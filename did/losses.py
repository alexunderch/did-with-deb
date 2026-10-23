import chex
import jax
import jax.numpy as jnp

"""Loss functions for the contract design and scheduling ID tasks."""


def contract_design_loss(
  sigma_star: chex.Array,
  contracts: chex.Array,
  transitions: chex.Array,
  principal_payoffs: chex.Array,
  payment_weight: chex.Numeric = 1.0,
) -> chex.Array:
  """
  Contract Design loss: negative expected principal net utility.

  Args:
      sigma_star: array shaped as [A1, A2, ..., AN], equilibrium from DEB
      contracts: array shaped as [N, O], v_p(o) - principal payments
      transitions: array shaped as [A1, ..., AN, O], P(o|a) - outcome probabilities
      principal_payoffs: array shaped as [O], W(o) - principal payoff per outcome
      payment_weight: weight lambda on the paid contracts. The paper ramps it
        from 0 to 1 over the first 500k steps (A.3.2) so the network doesn't
        collapse to zero contracts early on; evaluate with 1.0.

  Returns:
      Scalar loss: -E_{a~sigma*, o~P}[W(o) - lambda * sum_p v_p(o)].
  """
  # sum_p v_p(o) over agents
  contracts_sum = jnp.sum(contracts, axis=0)  # [O]

  # Principal net payoff per outcome: W(o) - lambda * sum_p v_p(o)
  net_principal = principal_payoffs - payment_weight * contracts_sum  # [O]

  # Expected principal net payoff per joint action
  #  U_principal(a) = sum_o P(o|a) * (W(o) - lambda * sum_p v_p(o))
  principal_per_action = jnp.sum(
    transitions * net_principal, axis=-1
  )  # [A1, ..., AN]

  # Expected utility under equilibrium
  expected_utility = jnp.sum(sigma_star * principal_per_action)

  return -expected_utility


def scheduling_makespan_table(processing_times: chex.Array) -> chex.Array:
  """Makespan of every joint action.

  Paper convention: t[p, j] is the processing time of player p's
  job on machine j, so `processing_times` is [N players, M machines] and the
  joint action space is [M, ..., M] (N dims).

      makespan(a) = max_j sum_{p: a_p = j} t[p, j]

  Padded machines must have t = 0 (provided by the sampler); they then
  carry load 0 and never change the max, and sigma_star is 0 on joint
  actions that use them, so no mask is needed.

  Args:
      processing_times: [N, M].

  Returns:
      [M, ..., M] (N dims), same layout as sigma_star.
  """
  num_players, num_machines = processing_times.shape

  # Enumerate all M^N joint actions: assignments[idx, p] = machine of player p
  grids = jnp.meshgrid(
    *[jnp.arange(num_machines) for _ in range(num_players)], indexing="ij"
  )
  assignments = jnp.stack(grids, axis=-1).reshape(-1, num_players)

  # One-hot: [M^N, N, M], [idx, p, j] = 1{a_p = j}
  one_hot = jax.nn.one_hot(assignments, num_machines)

  # Load of machine j: sum_p 1{a_p = j} * t[p, j] -> [M^N, M]
  loads = jnp.sum(one_hot * processing_times[None], axis=1)

  makespans = jnp.max(loads, axis=1)  # [M^N]
  return makespans.reshape((num_machines,) * num_players)


def scheduling_makespan_loss(
  sigma_star: chex.Array,
  processing_times: chex.Array,
  taxes: chex.Array | None = None,
  tax_weight: chex.Numeric = 0.0,
  eps: float = 1e-12,
) -> chex.Array:
  """Machine Scheduling: expected makespan (+ tax-magnitude regulariser).

  Args:
      sigma_star: [M, ..., M] (N dims), equilibrium over assignments.
      processing_times: [N, M], t[p, j] = time of player p's job on machine j.
      taxes: optional [N, M, ..., M] generator output (the payoff CHANGE,
        <= 0), only its magnitude is used.
      tax_weight: weight lambda on ||taxes||_2. The paper ramps it 0 -> 0.1
        over the first 50k steps.
      eps: keeps the gradient of the (non-squared) norm finite at s = 0.

  Returns:
      Scalar: E_{a~sigma*}[makespan(a)] + lambda * ||taxes||_2.
  """
  expected_makespan = jnp.sum(
    sigma_star * scheduling_makespan_table(processing_times)
  )
  if taxes is None:
    return expected_makespan
  tax_norm = jnp.sqrt(jnp.sum(jnp.square(taxes)) + eps)
  return expected_makespan + tax_weight * tax_norm


def batched_contract_design_loss(
  sigma_star: chex.Array,  # [B, A1, ..., AN]
  contracts: chex.Array,  # [B, N, O]
  transitions: chex.Array,  # [B, A1, ..., AN, O]
  principal_payoffs: chex.Array,  # [B, O]
  payment_weight: chex.Numeric = 1.0,
) -> chex.Array:
  per_item = jax.vmap(contract_design_loss, in_axes=(0, 0, 0, 0, None))(
    sigma_star, contracts, transitions, principal_payoffs, payment_weight
  )
  return jnp.mean(per_item)


def batched_scheduling_makespan_loss(
  sigma_star: chex.Array,  # [B, M, ..., M]
  processing_times: chex.Array,  # [B, N, M]
  taxes: chex.Array | None = None,  # [B, N, M, ..., M]
  tax_weight: chex.Numeric = 0.0,
) -> chex.Array:
  if taxes is None:
    per_item = jax.vmap(scheduling_makespan_loss)(sigma_star, processing_times)
  else:
    per_item = jax.vmap(
      scheduling_makespan_loss, in_axes=(0, 0, 0, None)
    )(sigma_star, processing_times, taxes, tax_weight)
  return jnp.mean(per_item)
