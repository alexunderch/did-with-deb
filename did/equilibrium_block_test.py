import itertools

import flax.nnx as nn
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from did import equilibrium_block
from nes import networks, samplers, utils

"""Tests for did.equilibrium_block.py"""


class DEBTest(parameterized.TestCase):
  """Mock tests for the mechanism design."""

  @parameterized.parameters(
    itertools.product(
      (1, 20, 300),
      (samplers.Objective.EPS_MWMRE, samplers.Objective.EPS_ME),
    )
  )
  def test_it_runs(self, seed, obj):
    model_rng, mask_rng = jax.random.split(jax.random.key(seed))
    num_players = 2
    batch_size = 10
    action_sizes = (4,) * num_players

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
      objective=obj,
      **common,
    )
    strat_mask_per_player = [
      jnp.ones((batch_size, action_sizes[0]), dtype=bool)
      for i in range(num_players)
    ]

    joint_mask = jax.vmap(utils.make_joint_mask_from_strat_masks)(
      strat_mask_per_player
    )

    strategy = deb(
      payoffs=jnp.ones((batch_size, num_players, *action_sizes)),
      strat_mask_per_player=strat_mask_per_player,
      joint_mask=None,
    )
    self.assertEqual(strategy.shape, (batch_size, *action_sizes))

    self.assertAlmostEqual(
      strategy.sum(axis=range(1, strategy.ndim), where=joint_mask).mean(),
      1.0,
      places=2,
    )


if __name__ == "__main__":
  absltest.main()
