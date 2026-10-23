import itertools

import jax
from absl.testing import absltest, parameterized

from did import samplers

"""Tests for did.samplers.py"""

RNG = jax.random.key(0)


class ContextSamplerTest(parameterized.TestCase):
  
  @parameterized.parameters(
    itertools.product(
      (1, 5, 10),
      (2, 4),
      (8, 16),
    )
  )
  def test_contract_design(
    self, batch_size, num_players, max_actions_or_outcomes
  ):
    max_actions = max_outcomes = max_actions_or_outcomes
    sampler = samplers.ContractDesignSampler(
      num_players=num_players,
      max_outcomes=max_outcomes,
      max_actions=max_actions,
    )
    batch = sampler.sample_random(batch_size, RNG)

    action_shape = (max_actions,) * num_players

    self.assertEqual(
      batch.transition.shape,
      (batch_size, *action_shape, max_outcomes),
    )

    self.assertEqual(batch.principal_payoff.shape, (batch_size, max_outcomes))

    self.assertEqual(
      batch.cost.shape, (batch_size, num_players, *action_shape)
    )

    self.assertEqual(
      batch.joint_mask.shape, (batch_size, *action_shape)
    )

  @parameterized.parameters(
    itertools.product(
      (1, 5, 10),
      (2, 4),
      (8, 12),
    )
  )
  def test_machine_schedulling(self, batch_size, num_players, max_machines):
    sampler = samplers.SchedulingSampler(
      num_players=num_players,
      max_machines=max_machines,
    )
    batch = sampler.sample_random(batch_size, RNG)

    self.assertEqual(
      batch.processing_time.shape, (batch_size, num_players, max_machines)
    )

    self.assertEqual(
      batch.base_payoff.shape,
      (batch_size, num_players) + (max_machines,) * num_players,
    )

    self.assertEqual(
      batch.joint_mask.shape, (batch_size,) + (max_machines,) * num_players
    )


if __name__ == "__main__":
  absltest.main()
