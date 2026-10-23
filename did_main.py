import flax.nnx as nn
import jax
from absl import app, flags, logging

from did import designer, equilibrium_block, generators, samplers

logger = logging
FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string(
  "checkpoint_dir", "/tmp/did_test", "Directory to save/load the agent models."
)

flags.DEFINE_string(
  "deb_config_file",
  None,
  "A JSON file with the DEB configuration.",
  required=True,
)

flags.DEFINE_enum(
  "task",
  "CONTRACT_DESIGN",
  [m.name for m in designer.Task],
  "What task, MACHINE_SCHEDULLING or CONTRACT_DESIGN are we adapting to?",
)

flags.DEFINE_integer(
  "num_players", None, "Number of players for the task.", required=True
)

flags.DEFINE_integer(
  "cd_min_actions", 2, "Number of minimum actions for the contract design."
)

flags.DEFINE_integer(
  "cd_max_actions", 16, "Number of maximum actions for the contract design."
)

flags.DEFINE_integer(
  "cd_min_outcomes", 2, "Number of minimum actions for the contract design."
)

flags.DEFINE_integer(
  "cd_max_outcomes", 10, "Number of maximum outcomes for the contract design."
)

flags.DEFINE_float("cd_min_cost", -4, "Minimum cost of the contract design.")

flags.DEFINE_float("cd_max_cost", 0, "Maximum cost for the contract design.")

flags.DEFINE_float("cd_min_outcome", -5, "Minimum outcome of the contract design.")

flags.DEFINE_float("cd_max_outcome", 5, "Maximum outcome for the contract design.")

flags.DEFINE_float(
  "cd_transition_alpha", 0.1, "Discounting coeff. for the contract design."
)

flags.DEFINE_integer(
  "ms_min_machines", 2, "Number of minimum machines for the machine schedulling."
)

flags.DEFINE_integer(
  "ms_max_machines", 12, "Number of maximum machines for the machine schedulling."
)

flags.DEFINE_float(
  "ms_time_sigma", 0.5, "Std. deviation of the LogNormal times distribution."
)

flags.DEFINE_integer("task_warm_up_steps", 1000, "Warm up for the task schedule.")

flags.DEFINE_float("ms_max_tax_weight", 0.1, "Maximum weight for the tax regulariser.")
flags.DEFINE_float("ms_max_delta", 1.0, "Delta for the taxes.")


flags.DEFINE_list(
  "ms_channel_list",
  [32, 32],
  "Number of hidden units in the payoff-to-payoff machine schedulling pipeline",
)

flags.DEFINE_integer(
  "cd_po2po_channels",
  32,
  "Number of channels in the PO2PO layer",
)

flags.DEFINE_integer(
  "cd_o2o_channels",
  32,
  "Number of channels in the PO2PO layer",
)

flags.DEFINE_integer(
  "cd_num_po2po",
  2,
  "Number the PO2PO layers",
)

flags.DEFINE_integer(
  "cd_num_o2o",
  32,
  "Number the O2O layers",
)

flags.DEFINE_integer(
  "eval_every",
  int(10),
  "Episode frequency of the evaluation.",
)
flags.DEFINE_integer("iterations", int(6e4), "Number of training episodes.")
flags.DEFINE_integer(
  "log_every", 5000, "Episode frequency at which the mechanism design is evaluated."
)


flags.DEFINE_float("max_grad_norm", 1, "Max allowed gradient norm.")
flags.DEFINE_float("base_learning_rate", 1e-4, "Learning rate of the solver's updates.")
flags.DEFINE_integer("warmup_steps", 1000, "Warmup rate for the solver's updates.")
flags.DEFINE_integer("seed", 42, "A random seed.")

flags.DEFINE_integer(
  "batch_size", 64, "Number of transitions to sample at each learning step."
)
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")

flags.DEFINE_integer(
  "eval_batch_size", 10, "Number of transitions to sample at each evaluation step."
)


def main(_) -> None:
  task = designer.Task[FLAGS.task]
  generator_rng = jax.random.key(FLAGS.seed)

  if task == designer.Task.MACHINE_SCHEDULLING:
    sampler = samplers.SchedulingSampler(
      num_players=FLAGS.num_players,
      min_machines=FLAGS.ms_min_machines,
      max_machines=FLAGS.ms_max_machines,
      time_sigma=FLAGS.ms_time_sigma,
    )
    task_def = designer.define_machine_schedulling_task(
      sampler=sampler,
      tax_ramp_steps=FLAGS.task_warm_up_steps,
      max_tax_weight=FLAGS.ms_max_tax_weight,
    )
    generator = generators.SchedulingGenerator(
      num_players=FLAGS.num_players,
      channel_list=FLAGS.ms_channel_list,
      max_delta=FLAGS.ms_max_delta,
      rngs=nn.Rngs(generator_rng),
    )
  elif task == designer.Task.CONTRACT_DESIGN:
    sampler = samplers.ContractDesignSampler(
      num_players=FLAGS.num_players,
      min_actions=FLAGS.cd_min_actions,
      max_actions=FLAGS.cd_max_actions,
      min_outcomes=FLAGS.cd_min_outcomes,
      max_outcomes=FLAGS.cd_max_outcomes,
      min_cost=FLAGS.cd_min_cost,
      max_cost=FLAGS.cd_max_cost,
      min_outcome=FLAGS.cd_min_outcome,
      max_outcome=FLAGS.cd_max_outcome,
      transition_alpha=FLAGS.cd_transition_alpha,
    )
    task_def = designer.define_contract_design_task(
      sampler=sampler, payment_ramp_steps=FLAGS.task_warm_up_steps
    )
    generator = generators.ContractDesignGenerator(
      num_players=FLAGS.num_players,
      po2po_channels=FLAGS.cd_po2po_channels,
      o2o_channels=FLAGS.cd_o2o_channels,
      num_po2po=FLAGS.cd_num_po2po,
      num_o2o=FLAGS.cd_num_o2o,
      rngs=nn.Rngs(generator_rng),
    )
  else:
    raise ValueError("Unspecified task!")

  deb = equilibrium_block.DEB.from_config(config_file=FLAGS.deb_config_file)
  assert deb._num_players == FLAGS.num_players, "Incompatible DEB!"

  didesigner = designer.MechanismGeneratorDesigner(
    num_players=FLAGS.num_players,
    task=task_def,
    generator=generator,
    deb=deb,
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    base_learning_rate=FLAGS.base_learning_rate,
    network_train_steps=FLAGS.iterations,
    gradient_clipping=FLAGS.max_grad_norm,
    seed=FLAGS.seed,
    log_every=FLAGS.log_every,
    eval_every=FLAGS.eval_every,
    allow_checkpointing=FLAGS.use_checkpoints,
    warmup_steps=FLAGS.warmup_steps,
  )

  didesigner.solve()


if __name__ == "__main__":
  app.run(main)
