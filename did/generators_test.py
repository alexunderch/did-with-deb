import itertools

import flax.nnx as nn
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from did import generators

"""Tests for did.generators.py"""

RNG = jax.random.key(0)


class ContextGeneratorTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product((2, 3), (8, 12), (6, 16), (32, 64))
  )
  def test_contract_design(
    self, num_players, num_outcomes, max_actions, channels
  ):
    model_rng, data_rng, data_rng2 = jax.random.split(RNG, 3)
    generator = generators.ContractDesignGenerator(
      num_players,
      channels,
      channels,
      2,
      3,
      rngs=nn.Rngs(model_rng),
    )
    action_dim = (max_actions,) * num_players
    batch = {
      "costs": jnp.zeros((num_players, *action_dim)),
      "transition": jnp.zeros((*action_dim, num_outcomes)),
      "principal_payoff": jnp.zeros((num_outcomes,)),
      "mask": jax.random.bernoulli(data_rng, 0.5, action_dim),
      "outcome_mask": jax.random.bernoulli(data_rng2, 0.5, (num_outcomes,)),
    }
    contracts = generator(**batch)
    self.assertEqual(contracts.shape, (num_players, num_outcomes))

  @parameterized.parameters(
    itertools.product(
      (2, 3),
      (6, 12),
    )
  )
  def test_machine_schedulling(self, num_players, num_machines):
    model_rng, data_rng = jax.random.split(RNG)

    generator = generators.SchedulingGenerator(
      num_players,
      [32, 64, 32],
      rngs=nn.Rngs(model_rng),
    )
    batch = {
      "base_payoffs": jnp.zeros((num_players,) + (num_machines,) * num_players),
      "mask": jax.random.bernoulli(
        data_rng, p=0.1, shape=(num_machines,) * num_players
      ),
    }
    times = generator(**batch)
    self.assertEqual(
      times.shape, (num_players,) + (num_machines,) * num_players
    )
    self.assertTrue((times <= 0).all())
    self.assertTrue((times >= -1).all())


if __name__ == "__main__":
  absltest.main()
