import json

import chex
import etils.epath
import flax.nnx as nn
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from did.utils import static_batched_call
from nes import networks, samplers, solver, utils


class DEB:
  """A pretrained NES network, frozen, callable as sigma = deb(payoffs, mask)."""

  def __init__(
    self,
    checkpoint_dir: etils.epath.Path | str | None = None,
    solver_network: networks.NeuralEquilibriumModel | None = None,
    *,
    num_players: int,
    dual_channels: int,
    mode: networks.Mode,
    payoff_channel_list: list[int],
    dual_channel_list: list[int],
    welfare_coeff: float = 0.0,
    entropy_coeff: float,
    epsilon_max: float,
    epsilon: float = 0.01,
    objective: samplers.Objective | None = samplers.Objective.EPS_MWMRE,
    norm: int = 2,
    seed: int = 0,
  ) -> None:
    """Frozen version of the Neural Equilibrium solver. Constructor arguments
    mirror NESolver's constructor for the SAME values used to train the checkpoint 
    being loaded.

    NOTE:`epsilon_max` is the eps+ cap the DEB was trained with (used by recover_primals);
    `epsilon` is the TARGET slack to request at inference, in raw payoff units(0.01 in the
    paper). welfare_coeff must be 0.0: DID selects the eps-ME-Eql (pure max-entropy).

    """
    assert checkpoint_dir is not None or solver_network is not None, "\
      Must specify NES one way or another."

    if isinstance(checkpoint_dir, str):
      checkpoint_dir = etils.epath.Path(checkpoint_dir)

    self._num_players = num_players
    self._mode = mode
    self._mu = welfare_coeff
    self._rho = entropy_coeff
    self._eps_cap = epsilon_max
    self._epsilon = epsilon
    self._norm = norm
    self._obj = objective
    self._rngkey = jax.random.key(seed)

    if solver_network is not None:
      self._network = solver_network
      self._network.eval()  # freeze BatchNorm running stats

    elif checkpoint_dir is not None:

      self._network = networks.NeuralEquilibriumModel(
        num_players=num_players,
        dual_channels=dual_channels,
        mode=mode,
        payoff_channel_list=payoff_channel_list,
        dual_channel_list=dual_channel_list,
        rngs=nn.Rngs(seed),
      )
      self._network.eval()  # freeze BatchNorm running stats

      checkpointer = ocp.StandardCheckpointer()
      state_restored = checkpointer.restore(
        checkpoint_dir / "state", nn.state(self._network)
      )
      nn.update(self._network, state_restored)
      checkpointer.wait_until_finished()

    self._graphdef = nn.graphdef(self._network)
    self._state = nn.state(self._network)

  @classmethod
  def from_config(cls, config_file: etils.epath.Path | str) -> DEB:
    with open(config_file, "r") as f:
      config = json.load(f)

    config['mode'] = networks.Mode[config['mode']]
    config['objective'] = samplers.Objective[config['objective']]
    return cls(**config)

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _normalise_payoffs(
    self, payoffs: chex.Array, joint_mask: chex.Array
  ) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Masked GameSampler._compute_payoff_stats (NES paper, Eq. 16a).

    Statistics are taken over the valid joint actions only and Z_m comes
    from the effective (masked) size, so a game padded to a bigger shape is
    normalised exactly like the same game at its natural size.

    Returns:
      payoffs_norm: [N, A1, ..., AN], zero on invalid joint actions.
      scale: [N], the per-player factor Z_m / ||centred payoffs||_m. The
        target epsilon has to be multiplied by the same factor.
      z_m: [,].
    """
    action_axes = tuple(range(1, payoffs.ndim))
    valid = joint_mask[None]  # [1, A1, ..., AN]
    count = jnp.sum(joint_mask, keepdims=True)
    count_b = count.astype(payoffs.dtype).reshape(
      (-1,) + (1,) * (joint_mask.ndim-1)
    )

    mean = jnp.sum(payoffs, axis=action_axes, where=valid, keepdims=True) / count_b
    centred = jnp.where(valid, payoffs - mean, 0.0)
    norm = utils.compute_L_m_norm(centred, self._norm, action_axes)
    z_m = count_b ** (1.0 / self._norm)  # [1, 1, ...]
    scale = z_m / (norm + utils.SMALL_NUMBER)  # [N, 1, ...]
    return (
      centred * scale,
      jnp.squeeze(scale, axis=action_axes),
      z_m.reshape(-1),
    )

  def _intialise_strategy(
    self, joint_mask: chex.Array, rng: chex.PRNGKey
  ) -> tuple[chex.Array, chex.Array]:
    """Uniform sigma_hat over the valid joint actions and its scaled version
    (masked GameSampler._scale_strategy, Eq. 16d / 17)."""

    count = jnp.sum(joint_mask, keepdims=True)
    count_b = count.astype(jnp.float32).reshape(
      (-1,) + (1,) * (joint_mask.ndim-1)
    )
    strategy_base = jnp.where(joint_mask, 1.0 / count_b, 0.0)

    if self._obj in [samplers.Objective.MRE, samplers.Objective.EPS_MRE]:
      strategy_base = jnp.where(joint_mask, jax.random.gamma(rng, 1.0), 0.0)

    z_sigma = count_b * jnp.sqrt(
      (count_b + 1.0) / (count_b - 1.0 + utils.SMALL_NUMBER)
    )
    strategy_norm = jnp.where(
      joint_mask, z_sigma * (strategy_base - 1.0 / count_b), 0.0
    )
    return strategy_base, strategy_norm

  def _initialise_welfare(
    self, payoffs: chex.Array, joint_mask: chex.Array, z_m: chex.Array
  ) -> chex.Array:
    _, *action_shape = payoffs.shape
    if self._obj in [samplers.Objective.EPS_MWME, samplers.Objective.EPS_MWMRE]:
      welfare = jnp.sum(payoffs, axis=0, where=joint_mask)
    else:
      welfare = jnp.zeros(action_shape)

    mean = jnp.mean(
      welfare, axis=tuple(range(welfare.ndim)), keepdims=True, where=joint_mask
    )
    centered = welfare - mean
    welfare_norm = utils.compute_L_m_norm(
      centered, self._norm, tuple(range(welfare.ndim)), where=joint_mask
    )
    return z_m * centered / (welfare_norm + utils.SMALL_NUMBER)

  def __call__(
    self,
    payoffs: chex.Array,
    strat_mask_per_player: list[chex.Array],
    joint_mask: chex.Array | None = None,
  ) -> chex.Array:
    """Returns the eps-ME-(C)CE joint strategy sigma* for a batch of games.

    Args:
      payoffs: [B, N, A1, ..., AN] -- RAW, unnormalised (e.g. straight from
        the mechanism generator); this function applies the same
        normalisation the DEB was trained under. Entries on invalid joint
        actions are ignored.
      rng: chex.PRNGKey: random key
      strat_mask_per_player: per-player masks, each [B, Ap].
      joint_mask: optional [B, A1, ..., AN]; derived from
        strat_mask_per_player if omitted.

    Returns:
      sigma: [B, A1, ..., AN], exactly zero on invalid joint actions.
    """
    batch_size = payoffs.shape[0]

    if joint_mask is None:
      joint_mask = jax.vmap(utils.make_joint_mask_from_strat_masks)(
        strat_mask_per_player
      )

    _epsilon = self._epsilon

    def _initialise_single(payoff, strat_mask_per_player, joint_mask, rng):
      payoffs_norm, scale, z_m = self._normalise_payoffs(payoff, joint_mask)
      strategy_base, strategy_norm = self._intialise_strategy(joint_mask, rng)

      # target slack
      epsilon_target = jnp.minimum(_epsilon * scale, z_m)
      welfare = self._initialise_welfare(payoff, joint_mask, z_m)

      return samplers.Data(
        payoffs=payoffs_norm,
        strategy_base=strategy_base,
        strategy_norm=strategy_norm,
        epsilon_target=epsilon_target,
        welfare=welfare,
        strat_mask_per_player=strat_mask_per_player,
        joint_mask=joint_mask,
      )

    data = jax.vmap(_initialise_single)(
      payoffs,
      strat_mask_per_player,
      joint_mask,
      jax.random.split(jax.random.key(joint_mask.sum().astype(int)), batch_size),
    )

    broadcasted = jax.vmap(samplers.broadcast)(data)
    stacked = samplers.stack(broadcasted, axis=1)

    network = nn.merge(self._graphdef, self._state)
    duals = static_batched_call(network, stacked, data.strat_mask_per_player)

    _, strategy, _, _, _ = jax.vmap(
      solver.recover_primals, in_axes=(0, 0, None, None, None, None)
    )(duals, data, self._mu, self._rho, self._eps_cap, self._mode.value)

    return strategy


def compute_sigma_baseline(
  deb: DEB,
  baseline_payoffs: chex.Array,
  strat_mask_per_player: list[chex.Array],
  joint_mask: chex.Array | None = None,
) -> chex.Array:
  """sigma^0: the DEB's equilibrium of the game with no intervention
  (no contracts / no taxes).
  """
  return deb(baseline_payoffs, strat_mask_per_player, joint_mask)
