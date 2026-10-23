import flax.nnx as nn
import jax
from absl.testing import absltest

from did import designer, equilibrium_block, generators, samplers
from nes import networks
from nes.samplers import Objective

"""Tests for did.designer.py"""

# Test are still very arbitrary.
 
RNG = jax.random.key(0)

class DIDTest(absltest.TestCase):
  """Mock tests for the mechanism design."""
  
  def test_it_runs_contract_design(self):
    num_players = 3
    model_rng, _, _ = jax.random.split(RNG, 3)
    generator = generators.ContractDesignGenerator(
      num_players,
      32,
      32,
      2,
      3,
      rngs=nn.Rngs(model_rng),
    )

    sampler = samplers.ContractDesignSampler(num_players)
    task = designer.define_contract_design_task(sampler)

    common = {
      "num_players": num_players,
      "dual_channels": 32,
      "mode": networks.Mode.CCE,
      "payoff_channel_list": [32, 32],
      "dual_channel_list": [32, 32],
    }

    solver_network = networks.NeuralEquilibriumModel(
      **common,
      rngs=nn.Rngs(model_rng),
    )

    deb = equilibrium_block.DEB(
      solver_network=solver_network,
      welfare_coeff=10.0,
      entropy_coeff=10.0,
      epsilon_max=10.0,
      epsilon=0.01,
      objective=Objective.EPS_MWMRE,
      **common,
    )

    didesigner = designer.MechanismGeneratorDesigner(
      num_players=num_players,
      task=task,
      generator=generator,
      deb=deb,
      eval_every=0,
      log_every=20,
      batch_size=10,
      warmup_steps=20,
      network_train_steps=200,
    )

    didesigner.solve()


  def test_it_runs_machine_schedulling(self):
    num_players = 3
    model_rng, _, _ = jax.random.split(RNG, 3)
    generator = generators.SchedulingGenerator(
      num_players,
      [32, 32],
      rngs=nn.Rngs(model_rng),
    )

    sampler = samplers.SchedulingSampler(num_players)
    task = designer.define_machine_schedulling_task(sampler)

    common = {
      "num_players": num_players,
      "dual_channels": 32,
      "mode": networks.Mode.CCE,
      "payoff_channel_list": [32, 32],
      "dual_channel_list": [32, 32],
    }

    solver_network = networks.NeuralEquilibriumModel(
      **common,
      rngs=nn.Rngs(model_rng),
    )

    deb = equilibrium_block.DEB(
      solver_network=solver_network,
      welfare_coeff=10.0,
      entropy_coeff=10.0,
      epsilon_max=10.0,
      epsilon=0.01,
      objective=Objective.EPS_MWMRE,
      **common,
    )

    didesigner = designer.MechanismGeneratorDesigner(
      num_players=num_players,
      task=task,
      generator=generator,
      deb=deb,
      eval_every=0,
      log_every=20,
      batch_size=10,
      warmup_steps=20,
      network_train_steps=200,
    )

    didesigner.solve()


if __name__ == "__main__":
  absltest.main()
