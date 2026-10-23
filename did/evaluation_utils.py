import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from did.utils import refine_did_solution
from nes import equilibria_utils as eu, networks

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


def _evaluate_one(
  item: EvalData,
  mode: networks.Mode,
  welfare_coeff: float,
  entropy_coeff: float,
  eps_max: float,
  maximise: bool,
  penalty_coeff: float,
  maxfev: int,
) -> dict:
  deb_value = _linear_objective(
    item.sigma_deb, item.value_table, item.baseline_value
  )

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
