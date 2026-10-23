import itertools

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from did import utils
from nes.utils import Reduction

"""Tests for did.utils.py"""

RNG = jax.random.key(0)

C = 3
O_ = 10
N = 4
A = (2,) * N

class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
    p for p in itertools.product(
      (True, False),
      (True, False),
      (True, False),
    ) if any(p) # skipping (False, False, False)
  )
  def test_payoff_pooling(self, include_identity, include_all, include_other):
    # x: [C, N, A1, ..., AN, O]; mask: optional payoff-outcome mask [A1, ..., AN, O]
    x = jnp.zeros((C, N, *A, O_))
    mask = jax.random.bernoulli(RNG, p=0.5, shape=(*A, O_))
    x = utils.calc_pooled_payoffs_chan(
      payoffs_chan=x,
      joint_where=mask,
      other_player_strat_reduction=Reduction.MEANVAR,
      player_strat_reduction=Reduction.MEANVAR,
      player_reduction=Reduction.MEANVAR,
      include_identity=include_identity,
      include_all=include_all,
      include_other=include_other,
    )
    y = int(include_identity) + int(include_all) + int(include_other)
    self.assertEqual(x.shape[0], C * y**3)

  @parameterized.parameters(
    p for p in itertools.product(
      (True, False),
      (True, False),
      (True, False),
    ) if any(p)
  )
  def test_outcome_pooling(self, include_identity, include_all, include_other):
    # x is shaped as  [C, N, O]
    # outcome_mask: optional, shaped as [O]
    x = jnp.zeros((C, N, O_))
    outcome_mask = jax.random.bernoulli(RNG, p=0.5, shape=(O_,))
    x = utils.calc_pooled_outcome_chan(
      x,
      player_reduction=Reduction.MEANVAR,
      outcome_reduction=Reduction.MEANVAR,
      include_identity=include_identity,
      include_all=include_all,
      include_other=include_other,
      where=outcome_mask,
    )
    y = int(include_identity) + int(include_all) + int(include_other)
    self.assertEqual(x.shape[0], C * y ** 2)



if __name__ == "__main__":
  absltest.main()
