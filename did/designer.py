import dataclasses
import enum
from collections.abc import Callable

import chex
import etils.epath
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import trackio
from open_spiel.python.utils import data_logger
from tqdm.auto import tqdm

from did import equilibrium_block, evaluation_utils as eu, losses, samplers
from nes import utils

"""Differentiable incentive designer implemented in Jax.

  See the paper 'Deep Incentive Design with Differentiable Equilibrium Blocks.' See
  https://arxiv.org/abs/2603.07705 for more details.
"""

class Task(enum.StrEnum):
  CONTRACT_DESIGN = "contract_design"
  MACHINE_SCHEDULLING = "machine_schedulling"


# Task spec is the full definition of an ID task. We define it as a
# prime citizen of the incentive designer.


@dataclasses.dataclass(frozen=True)
class TaskSpec:
  """Task specification of the designer. Encompasses the following args:

  name: Task name
  sampler: samplers_did.ContextSampler, the data sampler
  generate: Callable[[nn.Module, object], chex.Array], context generator
      (generator, ctx) -> generator output, batched [B, ...]
  induced_game: Callable[[Context, chex.Array], chex.Array], a
      function that converts the context into the NFG:
      (ctx, out) -> payoffs [B, N, A1, ..., AN]

  baseline_game: Callable[[Context], chex.Array], the same as
      the `induced_game` but without a generator intervention:
      ctx -> payoffs of the game WITHOUT the intervention (for sigma^0)
  loss_fn: Callable[[chex.Array, Context, chex.Array, chex.Numeric], chex.Array],
      a scalar loss function: (sigma [B, A...], ctx, Context, weight) -> scalar loss
  value_table: Callable[[samplers.Context, chex.Array], chex.Array],
      a per-joint-action objective table [A1, ..., AN]
  weight_schedule: optax.Schedule, step -> regulariser weight
      (contract payment weight / tax weight)
  maximise: bool, direction of the optimisation objective (
      can vary dependent on the task: makespan minimisation/profit maximisation)
  """

  name: str
  sampler: samplers.ContextSampler
  generate: Callable[[nn.Module, object], chex.Array]

  induced_game: Callable[[samplers.Context, chex.Array], chex.Array]
  baseline_game: Callable[[samplers.Context], chex.Array]
  loss_fn: Callable[[chex.Array, samplers.Context, chex.Array, chex.Numeric], chex.Array]
  value_table: Callable[[samplers.Context, chex.Array], chex.Array]
  weight_schedule: optax.Schedule
  maximise: bool


def define_contract_design_task(
  sampler: samplers.ContractDesignSampler, payment_ramp_steps: int = 10_000
) -> TaskSpec:
  # Batched version of the execution
  @nn.vmap(in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
  def _contract_forward(
    generator, cost, transition, principal_payoff, joint_mask, outcome_mask
  ):
    return generator(cost, transition, principal_payoff, joint_mask, outcome_mask)

  def _generate(generator, ctx):
    return _contract_forward(
      generator,
      ctx.cost,
      ctx.transition,
      ctx.principal_payoff,
      ctx.joint_mask,
      ctx.outcome_mask,
    )  # contracts [B, N, O]

  def _induced_game(ctx, contracts):
    # G_p(a) = initial payoff_p(a) + sum_o P(o|a) v_p(o)
    gain = jnp.einsum("b...o,bpo->bp...", ctx.transition, contracts)
    return (ctx.cost + gain) * ctx.joint_mask[:, None]

  def _loss_fn(sigma, ctx, contracts, weight):
    return losses.batched_contract_design_loss(
      sigma, contracts, ctx.transition, ctx.principal_payoff, weight
    )

  def _value_table(ctx, contracts):  # single item
    net_diff = ctx.principal_payoff - jnp.sum(contracts, axis=0)
    return jnp.einsum("...o,o->...", ctx.transition, net_diff)

  return TaskSpec(
    name=Task.CONTRACT_DESIGN,
    sampler=sampler,
    generate=_generate,
    induced_game=_induced_game,
    baseline_game=lambda ctx: ctx.cost,
    loss_fn=_loss_fn,
    value_table=_value_table,
    weight_schedule=optax.linear_schedule(0.0, 1.0, payment_ramp_steps),
    maximise=True,
  )


def define_machine_schedulling_task(
  sampler: samplers.SchedulingSampler,
  tax_ramp_steps: int = 10_000,
  max_tax_weight: float = 0.1,
) -> TaskSpec:
  # Batched taxes estimation
  @nn.vmap(in_axes=(None, 0, 0), out_axes=0)
  def _scheduling_forward(generator, base_payoff, joint_mask):
    return generator(base_payoff, joint_mask)

  def _generate(generator, ctx):
    return _scheduling_forward(
      # we're using only payoffs!
      generator,
      ctx.base_payoff,
      ctx.joint_mask,
    )

  def _induced_game(ctx, taxes):
    # G_p(a) = u_p(a) - s_p(a); NOTE: the generator returns -s!
    return (ctx.base_payoff + taxes) * ctx.joint_mask[:, None]

  def _loss_fn(sigma, ctx, taxes, weight):
    return losses.batched_scheduling_makespan_loss(
      sigma, ctx.processing_time, taxes, weight
    )

  def _value_table(ctx, taxes):  # single item; taxes unused
    return losses.scheduling_makespan_table(ctx.processing_time)

  return TaskSpec(
    name=Task.MACHINE_SCHEDULLING,
    sampler=sampler,
    generate=_generate,
    induced_game=_induced_game,
    baseline_game=lambda ctx: ctx.base_payoff,
    loss_fn=_loss_fn,
    value_table=_value_table,
    weight_schedule=optax.linear_schedule(0.0, max_tax_weight, tax_ramp_steps),
    maximise=False,
  )


class MechanismGeneratorDesigner:
  def __init__(
    self,
    task: TaskSpec,
    generator: nn.Module,
    deb: equilibrium_block.DEB,
    num_players: int,
    *,
    batch_size: int = 64,
    eval_batch_size: int = 16,
    base_learning_rate: float = 1e-3,
    warmup_steps: int = 50_000,
    network_train_steps: int = 150_000,
    gradient_clipping: float | None = None,
    allow_checkpointing: bool = False,
    log_every: int = 1000,
    eval_every: int = 0,
    seed: int = 42,
  ) -> None:
    self._task = task
    self._generator = generator
    self._deb = deb
    self._batch_size = batch_size
    self._eval_batch_size = eval_batch_size
    self._rng = jax.random.key(seed)

    self._checkpointer = None
    self._num_players = num_players
    self._log_every = log_every
    self._eval_every = eval_every
    if allow_checkpointing:
      self._checkpointer = ocp.StandardCheckpointer()

    self._logger = None
    self._network_train_steps = network_train_steps
    tx = optax.adam(
      utils.lr_schedule_with_warmup(
        base_learning_rate, warmup_steps, network_train_steps
      )
    )
    if gradient_clipping:
      tx = optax.chain(optax.clip_by_global_norm(gradient_clipping), tx)

    self._optimizer = nn.Optimizer(self._generator, tx, wrt=nn.Param)
    self._graphdef = nn.graphdef((self._generator, self._optimizer))
    self._update_fn = self._get_jitted_update()

    self._evaluator = eu.AsyncEvaluator(maximise=task.maximise)

  def _loss_fn(
    self, generator: nn.Module, ctx: samplers.Context, weight: chex.Numeric
  ) -> tuple[float, dict]:
    # Batched version of the loss
    out = self._task.generate(generator, ctx)
    payoffs = self._task.induced_game(ctx, out)
    sigma = self._deb(payoffs, ctx.strategy_masks, ctx.joint_mask)
    loss = self._task.loss_fn(sigma, ctx, out, weight)
    axes = tuple(range(1, sigma.ndim))
    entropy = jnp.mean(-jnp.sum(sigma * jnp.log(sigma + utils.SMALL_NUMBER), axis=axes))
    return loss, {"sigma_entropy": entropy}

  def _get_jitted_update(self) -> Callable:
    @jax.jit
    def update(state, rng, weight):
      generator, optimiser = nn.merge(self._graphdef, state, copy=True)
      ctx = self._task.sampler.sample_random(self._batch_size, rng)
      (loss, aux), grads = nn.value_and_grad(self._loss_fn, has_aux=True)(
        generator, ctx, weight
      )
      optimiser.update(generator, grads)
      return loss, aux, nn.state((generator, optimiser))

    return update

  def save(self, checkpoint_dir: etils.epath.Path, save_optimiser: bool = True) -> None:
    """Saves a current DID generator state.

    Args:
      checkpoint_dir (etils.epath.Path): target checkpoint location.
      save_optimiser (bool, optional): whether save only the optimiser or just
        the network's weights. Defaults to True.
    """
    assert self._checkpointer, (
      "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    )
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = etils.epath.Path(checkpoint_dir)
    if save_optimiser:
      self._checkpointer.save(
        checkpoint_dir / "optimiser",
        nn.state((self._generator, self._optimizer)),
        force=True,
      )
    else:
      self._checkpointer.save(checkpoint_dir / "state", nn.state(self._generator))
    self._checkpointer.wait_until_finished()

  def load(self, checkpoint_dir: etils.epath.Path, load_optimiser: bool = True) -> None:
    """Restores the NES network state.

    Args:
      checkpoint_dir (etils.epath.Path): target checkpoint dir.
      load_optimiser (bool, optional): whether load only the optimiser or just
        the network's weights. Defaults to True.
    """
    assert self._checkpointer, (
      "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    )
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = etils.epath.Path(checkpoint_dir)

    if load_optimiser:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "optimiser",
        nn.state((self._generator, self._optimizer)),
      )
      nn.update((self._generator, self._optimizer), state_restored)

    else:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "state", nn.state(self._generator)
      )
      nn.update(self._generator, state_restored)
    self._checkpointer.wait_until_finished()

  def solve(
    self,
    num_steps: int  | None = None,
    batch_size: int | None = None,
    log_every: int  | None = None,
    eval_every: int | None = None,
    eval_batch_size: int | None = None,
    logging_path: etils.epath.Path | str | None = None,
  ):
    num_steps = self._network_train_steps if num_steps is None else num_steps
    batch_size = self._batch_size if batch_size is None else batch_size
    eval_batch_size = (
      self._eval_batch_size if eval_batch_size is None else eval_batch_size
    )
    log_every = self._log_every if log_every is None else log_every
    eval_every = self._eval_every if eval_every is None else eval_every

    trackio.init(
      project="DID",
    )
    logger = None
    if logging_path is not None:
      logger = data_logger.DataLoggerJsonLines(logging_path, "DID_learner", True)

    state = nn.state((self._generator, self._optimizer))
    history = []
    for step in tqdm(range(num_steps), total=num_steps):
      rng = jax.random.fold_in(self._rng, step)
      weight = jnp.asarray(self._task.weight_schedule(step), jnp.float32)
      loss, aux, state = self._update_fn(state, rng, weight)

      if log_every and (step + 1) % log_every == 0:
        nn.update((self._generator, self._optimizer), state)
        row = {
          "step": step + 1,
          "loss": float(loss),
          "weight": float(weight),
          **{k: float(v) for k, v in aux.items()},
        }
        if eval_every and (step + 1) % eval_every == 0:
          row.update(
            self.evaluate(
              step + 1,
              eval_batch_size,
              mode=self._deb._mode,
              welfare_coeff=self._deb._mu,
              entropy_coeff=self._deb._rho,
              eps_max=self._deb._eps_cap,
            )
          )
        trackio.log({k: v for k, v in row.items()}, step=step + 1)
        if logger is not None:
          logger.write({k: v for k, v in row.items()})
        history.append(row)
        print(row)
    nn.update((self._generator, self._optimizer), state)
    trackio.finish()
    return history

  def build_eval_items(
    self, ctx: samplers.Context, out: chex.Array, epsilon: float = 0.01
  ) -> list[eu.EvalData]:
    payoffs = self._task.induced_game(ctx, out)
    sigma = self._deb(payoffs, ctx.strategy_masks, ctx.joint_mask)
    # sigma^0: equilibrium of the game WITHOUT the intervention
    sigma0 = self._deb(self._task.baseline_game(ctx), ctx.strategy_masks, ctx.joint_mask)
    tables = jax.vmap(self._task.value_table)(ctx, out)  # [B, A1, ..., AN]
    axes = tuple(range(1, sigma.ndim))
    baseline = jnp.sum(sigma0 * tables, axis=axes)  # [B]

    payoffs, sigma, tables, baseline, stat_masks, joint_mask = map(
      np.asarray, (payoffs, sigma, tables, baseline, ctx.strategy_masks, ctx.joint_mask)
    )
    num_players = payoffs.shape[1]
    return [
      eu.EvalData(
        payoffs=payoffs[i],
        welfare=None,
        strat_masks=[mask[i] for mask in stat_masks],
        joint_mask=joint_mask[i],
        sigma_deb=sigma[i],
        epsilon_target=np.full(num_players, epsilon),
        value_table=tables[i],
        baseline_value=float(baseline[i]),
      )
      for i in range(payoffs.shape[0])
    ]

  def evaluate(
    self,
    step: int,
    batch_size: int,
    *,
    mode: str = "CCE",
    welfare_coeff: float = 0,
    entropy_coeff: float = 100.0,
    eps_max: float = 10.0,
    penalty_coeff: float = 100,
    maxfev: int = 400,
  ) -> dict:
    ctx = self._task.sampler.sample_random(batch_size, jax.random.key(10_000 + step))
    out = self._task.generate(self._generator, ctx)
    items = self.build_eval_items(ctx, out)

    futures = self._evaluator.submit_batch(
      items,
      mode=mode,
      welfare_coeff=welfare_coeff,
      entropy_coeff=entropy_coeff,
      eps_max=eps_max,
      penalty_coeff=penalty_coeff,
      maxfev=maxfev,
    )
    report = self._evaluator.collect(futures, timeout=30.0)

    return {k: v for k, v in report.items() if k not in ("per_item", "ecos_statuses")}
