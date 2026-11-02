import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import chex
import cvxpy as cp
import numpy as np

from did.utils import refine_did_solution
from nes import equilibria_utils as eu, networks, utils

"""Mostly numpy-based evaluation for the multi-threaded usage."""


def contract_design_value_table(
  transition: np.ndarray,  # [A1, ..., AN, O]
  principal_payoff: np.ndarray,  # [O]
  contracts: np.ndarray,  # [N, O]
) -> np.ndarray:
  """Per-joint-action net principal value W(o) - sum_p v_p(o), in
  expectation over the outcome distribution P(o|a), shaped as [A1, ..., AN].
  """
  net_payoff_per_outcome = principal_payoff - contracts.sum(axis=0)
  return np.einsum("...o,o->...", transition, net_payoff_per_outcome)


def scheduling_value_table(
  job_times: np.ndarray,  # [N, M]
  num_players: int,
  num_machines: int,
) -> np.ndarray:
  """Per-joint-action makespan max_j sum_{p: a_p=j} t_{p,j},
  shaped as [M, M, ..., M] (num_players times).
  """
  action_shape = (num_machines,) * num_players
  table = np.zeros(action_shape)
  for flat_idx in range(int(np.prod(action_shape))):
    a = np.unravel_index(flat_idx, action_shape)
    totals = np.zeros(num_machines)
    for p in range(num_players):
      totals[a[p]] += job_times[p, a[p]]
    table[a] = totals.max()
  return table


@dataclasses.dataclass
class EvalData:
  """Data used for evaluation.
  payoffs: np.ndarray ([N, A1, ..., AN]), the induced game G(theta; omega)
  welfare: np.ndarray ([A1, ..., AN]), welfare of the induced game
  joint_mask: np.ndarray ([A1, ..., AN]), joint action mask
  sigma_deb: np.ndarray ([A1, ..., AN]), the DEB's sigma*
  epsilon_target: np.ndarray ([N]=num_players), fixed slack
  value_table: np.ndarray ([A1, ..., AN]), per-joint-action objective
    coefficients (net principal value for contract design, makespan for
    scheduling
  baseline_value: float, objective value at the precomputed (no incentive) sigma^0
  """

  payoffs: np.ndarray
  welfare: np.ndarray | None
  strat_masks: list[np.ndarray]
  joint_mask: np.ndarray
  sigma_deb: np.ndarray
  epsilon_target: np.ndarray
  value_table: np.ndarray
  baseline_value: float


def _linear_objective(
  sigma: np.ndarray, value_table: np.ndarray, baseline_value: float
) -> float:
  return float(np.sum(sigma * value_table)) - baseline_value


def local_solution_polish(
  sigma_init: np.ndarray,
  item: EvalData,
  mode: networks.Mode,
  maximise: bool,
  penalty_coeff: float,
  maxfev: int,
) -> tuple[np.ndarray, float]:
  """Nelder-Mead local polish of sigma_init."""

  sign = -1.0 if maximise else 1.0

  def loss(sigma: np.ndarray) -> float:
    return sign * _linear_objective(
      sigma, item.value_table, item.baseline_value
    )

  sigma_polished = refine_did_solution(
    loss,
    sigma_init,
    payoffs=item.payoffs,
    strat_mask_per_player=item.strat_masks,
    joint_mask=item.joint_mask,
    mode=mode,
    epsilon=item.epsilon_target,
    penalty_coeff=penalty_coeff,
    maxfev=maxfev,
  )
  value = _linear_objective(
    sigma_polished, item.value_table, item.baseline_value
  )
  return sigma_polished, value


# As entropy_coeff (rho) approaches the infinity, in ME problems,
# we might need a special solver for these type of problems,
def solve_me_eql(
  payoffs: chex.Array,
  strat_mask_per_player: list[chex.Array],
  mode: str,
  epsilon: float | chex.Array,
  *,
  verbose: bool = False,
  abstol: float = 1e-3,
  reltol: float = 1e-3
) -> dict:
  """Exact eps-ME-(C)CE: max entropy over {sigma : eps-(C)CE}, with a fixed epsilon.

  This is the problem DID asks the DEB to approximate (a max-entropy
  equilibrium of the induced game at eps = .01 in the paper), so it is the right
  reference to compare the DEB against. There is no slack variable, therefore, 
  no epsilon cap and no relative entropy coefficient:

      max_sigma  sum_a entr(sigma(a))
      s.t.       sum_a sigma(a) = 1,  sigma >= 0,  M_p @ sigma <= eps_p

  (maximising entropy = minimising KL to the uniform distribution on the
  valid joint actions).

  Args:
    payoffs: [N, A1, ..., AN], the RAW game (same units as `epsilon`).
    strat_mask_per_player: per-player valid-action masks, [A_p] each.
    mode: str,  "CCE" or "CE".
    epsilon: scalar or [num_players=N] slack.
    verbose: Where to verbose the `cvxpy` solver.
    abstol, reltol: ECOS tolerances.

  Returns:
    dict with strategy_target ([A1, ..., AN], zero on invalid joint actions),
    status, solver, entropy, ineq_violation (how far the returned point is
    outside the eps-(C)CE set.
  """
  payoffs = np.asarray(payoffs, dtype=np.float32)
  num_players, *action_shape = payoffs.shape
  joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)
  masks = [np.asarray(m, dtype=bool) for m in strat_mask_per_player]

  valid_idx = np.flatnonzero(joint_mask.reshape(-1))
  n_valid = len(valid_idx)
  eps = np.broadcast_to(np.asarray(epsilon, dtype=np.float32), (num_players,))

  mats = utils._build_ineq_matrices(payoffs, valid_idx, masks, mode)

  scale = max(float(np.max(np.abs(payoffs[:, joint_mask]))), utils.SMALL_NUMBER)

  x = cp.Variable(n_valid, nonneg=True)
  constraints = [cp.sum(x) == 1] + [
    (mat / scale) @ x <= eps[p] / scale for p, mat in mats
  ]
  prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), constraints)

  def _failed(status, error):
    # Same keys as the success report, so callers can index unconditionally.
    return {
      "solver_gap": np.nan, "ineq_violation": np.nan,
      "error": error, "status": status, "entropy": np.nan
    }

  try:
    prob.solve(solver=cp.ECOS, verbose=verbose, abstol=abstol, reltol=reltol)
  except cp.error.SolverError as e:
    # ECOS can raise (not just return a bad status) on hard instances.
    return _failed("solver_error", str(e))
   
  pred_value = np.clip(np.asarray(x.value, dtype=np.float64), 0.0, None)
  pred_value = pred_value / pred_value.sum()
  strat_pred = np.zeros(joint_mask.size)
  strat_pred[valid_idx] = pred_value

  violation = max(
    [0.0] + [float(np.max(mat @ pred_value)) - eps[p] for p, mat in mats]
  )
  return {
    "strategy_target": strat_pred.reshape(action_shape),
    "status": prob.status,
    "entropy": float(-np.sum(pred_value * np.log(pred_value + utils.SMALL_NUMBER))),
    "ineq_violation": violation,
   }


def _evaluate_one(
  item: EvalData,
  mode: networks.Mode,
  welfare_coeff: float,
  entropy_coeff: float,
  eps_max: float,
  maximise: bool,
  penalty_coeff: float,
  maxfev: int,
  # This solver is more general then eps-C(C)E-ME,
  # however, more fragile
  use_mwre_solver: bool = False
) -> dict:
  deb_value = _linear_objective(
    item.sigma_deb, item.value_table, item.baseline_value
  )
  if use_mwre_solver:

    ecos_report = eu.mwmre_solver(
      item.payoffs,
      item.welfare,
      item.sigma_deb,
      item.epsilon_target,
      item.strat_masks,
      welfare_coeff,
      entropy_coeff,
      eps_max,
      mode,
      verbose=False
    )
  else:
    ecos_report = solve_me_eql(
      item.payoffs,
      item.strat_masks,
      mode,
      epsilon=eps_max,
      verbose=False
    )

  sigma_ecos = ecos_report.get("strategy_target")
  ecos_value = (
    _linear_objective(sigma_ecos, item.value_table, item.baseline_value)
    if sigma_ecos is not None
    else float("nan")
  )

  _, deb_local_value = local_solution_polish(
    item.sigma_deb, item, mode.name, maximise, penalty_coeff, maxfev
  )
  if sigma_ecos is not None:
    _, ecos_local_value = local_solution_polish(
      sigma_ecos, item, mode.name, maximise, penalty_coeff, maxfev
    )
  else:
    ecos_local_value = float("nan")

  return {
    "deb": deb_value,
    "deb_local": deb_local_value,
    "ecos": ecos_value,
    "ecos_local": ecos_local_value,
    "ecos_status": ecos_report.get("status"),
  }


class AsyncEvaluator:
  """Thread-pool wrapper for solution polish approaches.

  scipy releases the GIL during the ECOS C call, so threads give
  real parallelism.
  """

  def __init__(
    self, 
    maximise: bool, 
    non_harmful_threshold: float = 1e-4,
    max_workers: int | None = None
  ) -> None:
    if max_workers is None:
      max_workers = max(1, (os.cpu_count() or 2) // 2)
    self._executor = ThreadPoolExecutor(max_workers=max_workers)
    self._pending = []
    self._maximise = maximise
    self._thr = non_harmful_threshold

  def shutdown(self):
    self._executor.shutdown(wait=True)

  def submit_batch(
    self,
    items: list[EvalData],
    *,
    mode: str,
    welfare_coeff: float = 0.0,
    entropy_coeff: float,
    eps_max: float,
    penalty_coeff: float = 1e3,
    maxfev: int = 400,
  ):
    """Run B solves asynchronously. Returns a Future."""

    # Submit all B tasks; ECOS runs in C and releases the GIL
    futures = [
      self._executor.submit(
        _evaluate_one,
        item,
        mode,
        welfare_coeff,
        entropy_coeff,
        eps_max,
        self._maximise,
        penalty_coeff,
        maxfev,
      )
      for item in items
    ]
    return futures

  def collect(self, futures: list, timeout: float | None = None):
    """Block until all futures done and return mean stats."""
    results = [f.result(timeout=timeout) for f in futures]

    # Stack scalar fields
    def _maybe_stack(key: str):
      vals = [r[key] for r in results if key in r and r[key] is not None]
      if not vals:
        return None
      return np.mean(np.stack([np.array(v) for v in vals]), axis=0)


    cmp = (
      (lambda v: v >= -self._thr)
      if self._maximise
      else (lambda v: v <= self._thr)
    )

    return {
      "deb_mean": float(np.nanmean(_maybe_stack("deb"))),
      "deb_local_mean": float(np.nanmean(_maybe_stack("deb_local"))),
      "ecos_mean": float(np.nanmean(_maybe_stack("ecos"))),
      "ecos_local_mean": float(np.nanmean(_maybe_stack("ecos_local"))),
      "deb_non_harmful_frac": float(np.mean(cmp(_maybe_stack("deb")))),
      "ecos_statuses": [r["ecos_status"] for r in results],
      "per_item": results,
    }
