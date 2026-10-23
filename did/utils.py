import functools
from collections.abc import Callable

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import nes.equilibria_utils as eu
from nes import utils as nes_utils


@nn.vmap(
  in_axes=(None, 0, 0), out_axes=0, axis_name="batch", graph=False, graph_updates=False
)
def static_batched_call(model: nn.Module, x: chex.Array, mask: chex.Array) -> chex.Array:
  """Batched model call. Explicitly resticts graph propagation."""
  return model(x, mask)


@functools.partial(
  jax.jit,
  static_argnames=(
    "other_player_strat_reduction",
    "player_strat_reduction",
    "player_reduction",
    "include_identity",
    "include_all",
    "include_other",
  ),
)
def calc_pooled_payoffs_chan(
  *,
  payoffs_chan: chex.Array | None = None,
  joint_chan: chex.Array | None = None,
  joint_where: chex.Array | None = None,
  other_player_strat_reduction: nes_utils.Reduction = nes_utils.Reduction.SUM,
  player_strat_reduction: nes_utils.Reduction = nes_utils.Reduction.SUM,
  player_reduction: nes_utils.Reduction = nes_utils.Reduction.SUM,
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
) -> chex.Array:
  """Returns pooled payoffs channels with permutation-equivariant reductions.

  Applies 3-way reductions (identity / all / other) over:
    1. Opponent strategy axes (-p)
    2. Player p's own strategy axis
    3. The player axis (N)

  Resulting in up to 3x3x3 = 27 channel fold increase.

  Args:
      payoffs_chan: Array with shape [C, N, A1, ..., AN]. Any trailing axes
          after AN (e.g. an outcome axis O) are left untouched.
      joint_chan: Array with shape [C, A1, ..., AN].
      joint_where: Mask broadcastable (right-aligned) to the per-player
          slice [C, A1, ..., AN(, O)]: shape [A1, ..., AN], or [A1, ..., AN, O]
          if payoffs_chan has a trailing outcome axis. A [A1, ..., AN] mask
          against a tensor with a trailing axis does NOT line up and raises.
      other_player_strat_reduction: Reduction over opponent axes.
      player_strat_reduction: Reduction over player p's axis.
      player_reduction: Reduction over the player axis.
      include_identity: Include the unreduced (identity) channel.
      include_all: Include the "all" reduced channel.
      include_other: Include the "other" reduced channel.

  Returns:
      payoffs_chan: Array with shape [C', N, A1, ..., AN].
  """

  if payoffs_chan is not None:
    num_players = payoffs_chan.shape[1]

    # Reduce over strategies per player
    payoff_chan_per_player = []
    for player in range(num_players):
      payoff_chan = payoffs_chan[:, player]  # [C, A1, ..., AN]
      own_axis = player + 1
      opp_axes = tuple(q + 1 for q in range(num_players) if q != player)
      axes = [opp_axes, (own_axis,)]

      reductions = [other_player_strat_reduction, player_strat_reduction]
      for axis, reduction in zip(axes, reductions):
        assert 0 not in axis, "Cannot reduce over channel axis"

        payoff_chan = nes_utils.reduce(
          payoff_chan,
          axis,
          reduction,
          include_identity,
          include_all,
          include_other,
          where=joint_where,
        )
      payoff_chan_per_player.append(payoff_chan)
    payoffs_chan = jnp.stack(payoff_chan_per_player, axis=1)

    # Reduce over the player axis
    payoffs_chan = nes_utils.reduce(
      payoffs_chan,
      (1,),
      player_reduction,
      include_identity,
      include_all,
      include_other,
      where=joint_where,
    )

  elif joint_chan is not None:
    num_channels = joint_chan.shape[0]
    num_players = joint_chan.ndim - 1  # infer from joint shape [C, A1,...,AN]
    num_strats_per_player = joint_chan.shape[1:]

    payoffs_chan = jnp.broadcast_to(
      jnp.expand_dims(joint_chan, axis=1),  # [C, 1, A1, ..., AN]
      (num_channels, num_players, *num_strats_per_player),
    )
    payoffs_chan = calc_pooled_payoffs_chan(
      payoffs_chan=payoffs_chan,
      other_player_strat_reduction=other_player_strat_reduction,
      player_strat_reduction=player_strat_reduction,
      player_reduction=player_reduction,
      include_identity=include_identity,
      include_all=include_all,
      include_other=include_other,
      joint_where=joint_where,
    )

  return payoffs_chan


def collapse_meanvar(
  array: chex.Array,
  axis: tuple[int, ...],
  where: chex.Array | None = None,
) -> chex.Array:
  """Meanvar-collapses `axis`: sum / sqrt(count), with the axes REMOVED.

  Unlike `reduce_meanvar`, nothing is concatenated and nothing is broadcast
  back -- this is for layers that collapse an axis away entirely (e.g. PO2O
  collapsing the strategy axes, DID Appendix A.3.1).

  Args:
      array: Array to collapse.
      axis: Axes to collapse (removed from the output shape).
      where: Optional mask broadcastable (right-aligned) to `array`.

  Returns:
      Array with `axis` removed.
  """
  total = jnp.sum(array, axis=axis, where=where)
  if where is not None:
    count = jnp.sum(jnp.broadcast_to(where, array.shape), axis=axis)
    count = jnp.maximum(count, 1)
  else:
    count = jnp.prod([array.shape[a] for a in axis])
  return total / jnp.sqrt(count)


@functools.partial(
  jax.jit,
  static_argnames=(
    "player_reduction",
    "outcome_reduction",
    "include_identity",
    "include_all",
    "include_other",
  ),
)
def calc_pooled_outcome_chan(
  x: chex.Array,
  *,
  player_reduction: nes_utils.Reduction = nes_utils.Reduction.SUM,
  outcome_reduction: nes_utils.Reduction = nes_utils.Reduction.SUM,
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Returns pooled channels for outcome-shaped [C, N, O] tensors.

  Unlike `calc_pooled_payoffs_chan`, there are no per-player action axes
  here (e.g. after PO2O has collapsed them): just one shared player axis
  and one shared outcome axis, so there is no own-vs-opponent split.
  Applies identity/all/other reductions over the player axis (1) and the
  outcome axis (2), up to 3x3 = 9 channels (2x2 = 4 with
  include_other=False, matching DID's O2O layer).

  Args:
      x: Array with shape [C, N, O].
      player_reduction: Reduction over the player axis.
      outcome_reduction: Reduction over the outcome axis.
      include_identity: Include the unreduced (identity) channel.
      include_all: Include the "all" reduced channel.
      include_other: Include the "other" reduced channel.
      where: Optional outcome mask broadcastable (right-aligned) to x,
        e.g. shape [O].

  Returns:
      Array with shape [C', N, O].
  """
  x = nes_utils.reduce(
    x,
    (1,),
    player_reduction,
    include_identity,
    include_all,
    include_other,
    where=where,
  )
  x = nes_utils.reduce(
    x,
    (2,),
    outcome_reduction,
    include_identity,
    include_all,
    include_other,
    where=where,
  )
  return x


# Local (Nelder-Mead) refinement of a DID equilibrium solution.

# Reproduces the "+Local" polish of the DID paper (Sec 5.2): start from the
# joint strategy found by the DEB (or by ECOS) and run scipy's Nelder-Mead to
# quantify the local improvement of the designer's objective that was missed.

_GAP_FNS = {
  "CCE": jax.jit(eu.compute_cce_gap),
  "CE": jax.jit(eu.compute_ce_gap),
}


def build_gap_fn(
  payoffs: np.ndarray,
  strat_mask_per_player: list[chex.Array],
  joint_mask: np.ndarray,
  mode: str,
  epsilon: float | np.ndarray,
) -> Callable[[np.ndarray], float]:
  """Returns gap(sigma) = eu.compute_c(c)e_gap(payoffs, sigma, epsilon, ...).

  Args:
    payoffs: [N, A1, ..., AN], the (raw) game the equilibrium refers to.
    joint_mask: [A1, ..., AN] valid joint actions.
    mode: "CCE" or "CE".
    epsilon: scalar or [N] slack, in the same units as payoffs.

  Returns:
    A function of the FULL joint strategy sigma [A1, ..., AN], which must be
    zero on invalid joint actions (the refiner guarantees this).
  """
  if mode not in _GAP_FNS:
    raise ValueError(f"mode must be 'CE' or 'CCE', got {mode!r}")
  gap_fn = _GAP_FNS[mode]

  joint_mask = jnp.asarray(joint_mask, dtype=bool)
  payoffs = jnp.where(joint_mask[None], jnp.asarray(payoffs, jnp.float32), 0.0)
  num_players = payoffs.shape[0]
  strat_masks = strat_mask_per_player
  eps = jnp.broadcast_to(jnp.asarray(epsilon, jnp.float32), (num_players,))

  def gap(sigma: np.ndarray) -> float:
    return float(
      gap_fn(
        payoffs,
        jnp.asarray(sigma, jnp.float32),
        eps,
        joint_mask=joint_mask,
        strat_mask_per_player=strat_masks,
      )
    )

  return gap


def refine_did_solution(
  designer_loss_fn: Callable[[np.ndarray], float],
  initial_sigma: np.ndarray,
  *,
  payoffs: np.ndarray,
  strat_mask_per_player: list[chex.Array],
  joint_mask: np.ndarray | None = None,
  mode: str = "CCE",
  epsilon: float | np.ndarray = 0.01,
  penalty_coeff: float = 1e3,
  maxfev: int = 400,
  xatol: float = 1e-4,
  fatol: float = 1e-6,
  return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
  """Refines a DID equilibrium with Nelder-Mead, staying (softly) in the
  eps-(C)CE set of `payoffs`.

  Args:
    designer_loss_fn: sigma (np.ndarray, shape of `initial_sigma`, float64)
      -> scalar designer loss to MINIMISE.
    initial_sigma: [A1, ..., AN] joint strategy from the DEB or ECOS.
    payoffs: [N, A1, ..., AN] the game the equilibrium constraint refers
      to, in the same units as `epsilon`. Required: without it the search
      just moves all mass onto the designer's favourite joint action, which
      is not an equilibrium.
    joint_mask: [A1, ..., AN] valid joint actions (default: all valid).
    mode: "CCE" or "CE".
    epsilon: slack (scalar or [N]); 0.01 in the paper.
    penalty_coeff: weight on gap**2. Units are loss per payoff**2: it has to
      be large relative to the loss scale or the constraint is ignored.
    maxfev: function-evaluation budget (the paper uses 400 for contract
      design / scheduling). Hitting it is the normal way this ends, NOT a
      failure.
    xatol, fatol: Nelder-Mead tolerances. Keep them well above float32
      noise (the gap is float32).
    return_info: also return a dict with loss/gap before and after.

  Returns:
    refined sigma with the shape of `initial_sigma` (a probability
    distribution, exactly zero on invalid joint actions) -- or
    (sigma, info) if `return_info`. The result is never worse than the
    starting point on the penalised objective, so unlike a
    success-flag-gated return it keeps the progress made when the budget
    runs out.
  """
  sigma0 = np.asarray(initial_sigma, dtype=np.float64)
  shape = sigma0.shape
  mask = (
    np.ones(shape, dtype=bool)
    if joint_mask is None
    else np.asarray(joint_mask, dtype=bool)
  )
  valid_idx = np.flatnonzero(mask.reshape(-1))
  gap_fn = build_gap_fn(payoffs, strat_mask_per_player, mask, str(mode), epsilon)

  def to_full(sigma_valid: np.ndarray | chex.Array) -> np.ndarray:
    full = np.zeros(mask.size)
    full[valid_idx] = np.asarray(sigma_valid)
    return full.reshape(shape)

  def penalised(z: np.ndarray) -> float:
    full = to_full(jax.nn.softmax(z))
    return float(designer_loss_fn(full)) + penalty_coeff * gap_fn(full) ** 2

  sigma0_valid = sigma0.reshape(-1)[valid_idx]
  sigma0_valid = sigma0_valid / sigma0_valid.sum()
  info = {
    "loss_before": float(designer_loss_fn(sigma0)),
    "gap_before": gap_fn(to_full(sigma0_valid)),
  }

  if len(valid_idx) < 2:  # nothing to optimise
    info.update(
      loss_after=info["loss_before"],
      gap_after=info["gap_before"],
      nfev=0,
      success=True,
      message="single valid joint action",
    )
    return (sigma0, info) if return_info else sigma0

  z0 = np.log(np.clip(sigma0_valid, 1e-12, 1.0))
  result = scipy.optimize.minimize(
    penalised,
    z0,
    method="Nelder-Mead",
    # adaptive=True: scipy recommends it for high-dimensional problems
    # (up to |A| = 256 variables here).
    options={"maxfev": maxfev, "xatol": xatol, "fatol": fatol, "adaptive": True},
  )

  refined = to_full(jax.nn.softmax(result.x))
  info.update(
    loss_after=float(designer_loss_fn(refined)),
    gap_after=gap_fn(refined),
    nfev=int(result.nfev),
    success=bool(result.success),
    message=str(result.message),
  )
  return (refined, info) if return_info else refined
