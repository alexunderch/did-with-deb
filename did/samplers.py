import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from nes import utils


class ContractDesignContext(NamedTuple):
  """Contract design data structure."""

  # [B, N, max_A, max_A]   (agent costs, negative)
  cost: chex.Array
  # [B, max_A, max_A, max_O]  P(o|a) (row-stochastic for the valid outcomes)
  transition: chex.Array
  # [B, max_O]   W(o)
  principal_payoff: chex.Array
  # [[[A_p]] or [[B, A_p]] p={1,...,N} per-player valid action masks
  strategy_masks: list[chex.Array]
  # [B, max_A, max_A]      valid joint action mask
  joint_mask: chex.Array
  # [B, max_O] valid outcome masks
  outcome_mask: chex.Array

  @staticmethod
  def broadcast(context: ContractDesignContext) -> ContractDesignContext:
    """Broadcasts cost / transition / principal_payoff to [N, A_1, ..., A_N, O].
    Operates on a single (unbatched) context.
    """
    num_players = context.cost.shape[0]
    action_shape = context.cost.shape[1:]
    num_outcomes = context.principal_payoff.shape[-1]  # max_O
    full_shape = (num_players,) + action_shape + (num_outcomes,)
    return context._replace(
      cost=jnp.broadcast_to(context.cost[..., None], full_shape),
      transition=jnp.broadcast_to(context.transition[None], full_shape),
      principal_payoff=jnp.broadcast_to(context.principal_payoff, full_shape),
    )

  @staticmethod
  def stack(context: ContractDesignContext) -> chex.Array:
    """Channels-first generator input [3, N, A_1, ..., A_N, O].
    Operates on a single (unbatched) context.
    """
    b = ContractDesignContext.broadcast(context)
    return jnp.stack([b.cost, b.transition, b.principal_payoff], axis=0)


class SchedulingContext(NamedTuple):
  """Contract design data structure."""

  # [B, N, M]          t_{p,j} (p=player/job, j=machine)
  processing_time: chex.Array
  # [B, N, M, ..., M]  u_p(a) (N action dims)
  base_payoff: chex.Array
  # N arrays [B, M]  valid machines (identical for every player)
  strategy_masks: list[chex.Array]
  # [B, M, ..., M]     valid joint actions (N dims)
  joint_mask: chex.Array

  @staticmethod
  def stack(ctx: SchedulingContext) -> chex.Array:
    """Channels-first generator input [1, N, A_1, ..., A_N], A_i=M (single context)."""
    # As in the paper, we only use the base payoff, to comply with the
    # game-theoretic nature of the task
    return ctx.base_payoff[jnp.newaxis]


Context = ContractDesignContext | SchedulingContext


class ContextSampler:
  """Sampler for a task context."""

  @functools.partial(jax.jit, static_argnums=(0,))
  def _sample_single_random(self, rng: chex.PRNGKey) -> Context:
    raise NotImplementedError()

  def sample_random(
    self,
    batch_size: int,
    rng: chex.PRNGKey,
  ) -> Context:
    """DEB/DID input Context sampler."""
    # Vectorise over the batch size with unique random keys
    batch_keys = jax.random.split(rng, batch_size)
    return jax.vmap(self._sample_single_random)(batch_keys)


class ContractDesignSampler(ContextSampler):
  """Sampler for multi-agent contract design.
  Yields batches of (costs, transitions, principal_payoffs, action_mask, outcome_mask):
    - costs: [B, N, max_A, max_A]
    - transitions: [B, max_A, max_A, max_O]
    - principal_payoffs: [B, max_O]
    - valid action and outcome masks
  """

  def __init__(
    self,
    num_players: int = 2,
    min_actions: int = 2,
    max_actions: int = 16,
    min_outcomes: int = 2,
    max_outcomes: int = 10,
    min_cost: float = -4.0,
    max_cost: float = 0,
    min_outcome: float = -5,
    max_outcome: float = 5,
    transition_alpha: float = 0.1,
  ) -> None:
    self.num_players = num_players

    # Output space bounds
    self.min_outcomes = min_outcomes
    self.max_outcomes = max_outcomes
    assert min_outcome <= max_outcome
    self.outcome_bounds = (min_outcome, max_outcome)

    # Input space bounds
    self.min_actions = min_actions
    self.max_actions = max_actions

    # Cost bounds
    assert min_cost <= max_cost
    self.cost_bounds = (min_cost, max_cost)
    self.transition_alpha = transition_alpha

  def _sample_single_random(self, rng: chex.PRNGKey) -> ContractDesignContext:
    k_size, k_out, k_mask, k_cost, k_trans, k_w = jax.random.split(rng, 6)

    action_shape = (self.max_actions,) * self.num_players

    # Sample game size for the instance (used for masking)
    action_sizes = jax.random.randint(
      k_size, (self.num_players,), self.min_actions, self.max_actions + 1
    )

    # Masks
    strat_mask_per_player = utils.rand_choice_strat_mask(
      k_mask, action_shape, action_sizes
    )
    joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)

    # Sample outcome size for the instance (used for masking)
    outcome_sizes = jax.random.randint(
      k_out, (), self.min_outcomes, self.max_outcomes + 1
    )

    outcome_mask = jnp.arange(self.max_outcomes) < outcome_sizes  # [max_O]

    # Costs: [N, A_1, ..., A_N] in cost_bounds
    cost = jax.random.uniform(
      k_cost,
      (self.num_players,) + action_shape,
      minval=self.cost_bounds[0],
      maxval=self.cost_bounds[1],
    )
    cost = jnp.where(joint_mask[None], cost, 0.0)

    alpha = jnp.full((self.max_outcomes,), self.transition_alpha)
    transition = jax.random.dirichlet(
      k_trans, alpha, shape=action_shape
    )  # [A_1, ..., A_N, max_O]

    transition = transition * outcome_mask  # broadcast over [A_1, ..., A_N, O]

    mass = jnp.sum(transition, axis=-1, keepdims=True)
    tiny = jnp.finfo(transition.dtype).tiny
    uniform = outcome_mask.astype(transition.dtype) / jnp.sum(outcome_mask)
    transition = jnp.where(mass >= tiny, transition / jnp.maximum(mass, tiny), uniform)
    transition = jnp.where(joint_mask[..., None], transition, 0.0)

    principal_payoff = jax.random.uniform(
      k_w,
      (self.max_outcomes,),
      minval=self.outcome_bounds[0],
      maxval=self.outcome_bounds[1],
    )
    principal_payoff = jnp.where(outcome_mask, principal_payoff, 0.0)

    return ContractDesignContext(
      cost=cost,
      transition=transition,
      principal_payoff=principal_payoff,
      strategy_masks=strat_mask_per_player,
      joint_mask=joint_mask,
      outcome_mask=outcome_mask,
    )


class SchedulingSampler(ContextSampler):
  """Sampler for machine scheduling.

  Yields batches of (utilities, processing_times, masks):
    - utilities: [B, N, M, ..., M] (N action dims)  agent utilities u_p(a)
    - processing_times: [B, N, M]  t_{p,j}
    - masks: per-player machine masks [B, M] (shared across players) and
      the joint mask [B, M, ..., M]

  N players (jobs) each pick one of the machines {1..M}, with the number
  of machines M sampled per instance in [min_machines, max_machines] and
  padded/masked up to max_machines (the paper's games are 2 players, M
  from 2 to 12, i.e. sizes 2x2 .. 12x12). All players choose among the
  same machines, so every player gets the same machine mask.
  """

  # A guard against enumerating an enormous joint action space
  # due to the max_machines ** num_players entries per sample.
  _MAX_JOINT_ACTIONS: int = 2_000_000

  def __init__(
    self,
    num_players: int = 2,
    min_machines: int = 2,
    max_machines: int = 12,
    time_sigma: float = 0.5,
  ) -> None:
    # t_{p,j} ~ LogNormal(0, time_sigma) is the processing time of job p on
    # machine j; utilities are
    #   u_p(a) = -t_{p,a_p} - 0.5 * sum_{p'!=p, a_{p'}=a_p} t_{p',a_p}.
    assert num_players >= 2
    assert 2 <= min_machines <= max_machines
    if max_machines**num_players > self._MAX_JOINT_ACTIONS:
      raise ValueError(
        f"Scheduling with max_machines={max_machines} and "
        f"num_players={num_players} requires enumerating "
        f"{max_machines**num_players} joint actions per sample "
        f"(limit {self._MAX_JOINT_ACTIONS})."
      )

    self.num_players = num_players
    self.min_machines = min_machines
    self.max_machines = max_machines
    self.time_sigma = time_sigma

  def _sample_single_random(self, rng: chex.PRNGKey) -> SchedulingContext:
    k_m, k_mask, k_pt = jax.random.split(rng, 3)

    N, M = self.num_players, self.max_machines

    # Number of machines and a machine mask are shared by all players
    num_machines = jax.random.randint(k_m, (), self.min_machines, M + 1)
    machine_mask = jax.random.permutation(k_mask, jnp.arange(M) < num_machines)
    strategy_masks = tuple(machine_mask for _ in range(N))
    joint_mask = utils.make_joint_mask_from_strat_masks(strategy_masks)

    # t[p, j]: processing time of job p on machine j (zero on invalid machines)
    processing_time = (
      jax.random.lognormal(k_pt, sigma=self.time_sigma, shape=(N, M))
      * machine_mask[None, :]
    )

    # Enumerate M^N joint actions
    grids = jnp.meshgrid(*[jnp.arange(M) for _ in range(N)], indexing="ij")
    # a[p, idx] = machine chosen by player p in joint action idx
    a = jnp.stack(grids, axis=0).reshape(N, -1)  # [N, M^N]

    # Base cost: -t[p, a_p] for each player p and joint action idx
    base = -processing_time[jnp.arange(N)[:, None], a]  # [N, M^N]

    # Congestion: sum over other players p' on the same machine as p
    # same_machine[p, p', idx] = 1 iff a_p == a_{p'} (and p != p')
    same_machine = (a[:, None, :] == a[None, :, :]) & ~jnp.eye(N, dtype=bool)[:, :, None]

    # pt_on_p_machine[p', p, idx] = t[p', a_p]
    pt_on_p_machine = processing_time[:, a]  # [N, N, M^N]

    # Sum over p' (axis 0 of both arrays). `same_machine` is symmetric in
    # (p, p'), so same_machine[p', p] pairs correctly with t[p', a_p].
    congestion = jnp.sum(same_machine * pt_on_p_machine, axis=0)  # [N, M^N]
    utilities = base - 0.5 * congestion  # [N, M^N]

    payoff = utilities.reshape((N,) + (M,) * N)
    payoff = payoff * joint_mask[None]

    return SchedulingContext(
      processing_time=processing_time,
      base_payoff=payoff,
      strategy_masks=strategy_masks,
      joint_mask=joint_mask,
    )
