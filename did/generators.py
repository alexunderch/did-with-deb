import itertools

import chex
import flax.nnx as nn
import jax.numpy as jnp

from did.utils import (
  calc_pooled_outcome_chan,
  calc_pooled_payoffs_chan,
  collapse_meanvar,
)
from nes import networks
from nes.utils import Reduction, reduce

# NOTE:
# Pooling-family sizes use `calc_pooled_payoffs_chan` ([C, N, A1..AN(, O)])
# and `calc_pooled_outcome_chan` ([C, N, O]) with include_other=False
# to match DID's I=2^x channel counts (Appendix A.1/A.3.1).
# Both are independent of num_players/num_strats, which is what lets
# a single network share weights across game sizes.

# calc_pooled_payoffs_chan always folds in the player-axis reduction, so
# PO2PO stage 1 is 2x2x2 = 8 channels rather than the paper's staged 2x2 = 4.

# Masks: layers that see an outcome axis take a payoff-outcome mask of shape
# [A1, ..., AN, O] (joint mask AND outcome mask); a [A1, ..., AN] mask does
# not broadcast against [C, N, A1, ..., AN, O] (right-aligned).

PAYOFF_NUM_POOLS = 8  # opponent axes x own axis x player axis, 2x2x2
OUTCOME_NUM_POOLS = 4  # player axis x outcome axis, 2x2


class PayoffToPayoffLayer(nn.Module):
  """Equivariant Payoff to Payoffs layer (P2P), Appendix A.1/A.2.1."""

  def __init__(
    self,
    num_players: int,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.linear = nn.Linear(
      in_channels * PAYOFF_NUM_POOLS,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.gelu

  def __call__(
    self, x: chex.Array, mask: chex.Array | None = None
  ) -> chex.Array:
    pooled = calc_pooled_payoffs_chan(
      payoffs_chan=x,
      joint_where=mask,
      other_player_strat_reduction=Reduction.MEANVAR,
      player_strat_reduction=Reduction.MEANVAR,
      player_reduction=Reduction.MEANVAR,
      include_identity=True,
      include_all=True,
      include_other=False,
    )

    # [N, *A, C * num_pools]
    pooled_t = jnp.moveaxis(pooled, 0, -1)
    out_t = self.act(self.linear(pooled_t))
    return jnp.moveaxis(out_t, -1, 0)


class Outcome2OutcomeLayer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    last_activation: bool,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.linear = nn.Linear(
      in_channels * OUTCOME_NUM_POOLS,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.softplus if last_activation else nn.gelu

  def __call__(
    self, x: chex.Array, mask: chex.Array | None = None
  ) -> chex.Array:
    # x: [C, N, O]; mask: optional outcome mask [O]
    pooled = calc_pooled_outcome_chan(
      x,
      player_reduction=Reduction.MEANVAR,
      outcome_reduction=Reduction.MEANVAR,
      include_identity=True,
      include_all=True,
      include_other=False,
      where=mask,
    )
    pooled_t = jnp.moveaxis(pooled, 0, -1)  # [N, O, C*pools]
    out_t = self.act(self.linear(pooled_t))
    return jnp.moveaxis(out_t, -1, 0)  # [C', N, O]


class PayoffOutcome2PayoffOutcomeLayer(nn.Module):
  """PO2PO: Payoff-Outcome -> Payoff-Outcome,
  split into three pool + linear stages to save memory.
  """

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    # 1. Pool over strategy axes (own + opponent) and the player axis in one go
    self.linear_strat = nn.Linear(
      in_channels * PAYOFF_NUM_POOLS,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    # 2. Pool over player axis: [identity, all] -> 2 channels
    self.linear_player = nn.Linear(
      out_channels * 2,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    # 3. Pool over outcome axis
    self.linear_outcome = nn.Linear(
      out_channels * 2,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    self.act = nn.gelu

  def _apply_linear(self, x: chex.Array, linear: nn.Module) -> chex.Array:
    # x: [C*pools, N, A1, ..., AN, O] → move channels to end
    x_t = jnp.moveaxis(x, 0, -1)
    out_t = linear(x_t)
    return jnp.moveaxis(out_t, -1, 0)

  def __call__(
    self, x: chex.Array, mask: chex.Array | None = None
  ) -> chex.Array:
    # x: [C, N, A1, ..., AN, O]; mask: optional payoff-outcome mask [A1, ..., AN, O]
    # Pool over strategies (own + opponent) and players
    # The trailing outcome axis is left untouched.
    x = calc_pooled_payoffs_chan(
      payoffs_chan=x,
      joint_where=mask,
      other_player_strat_reduction=Reduction.MEANVAR,
      player_strat_reduction=Reduction.MEANVAR,
      player_reduction=Reduction.MEANVAR,
      include_identity=True,
      include_all=True,
      include_other=False,
    )
    x = self._apply_linear(x, self.linear_strat)
    x = self.act(x)
    # Pool over players
    x = reduce(x, (1,), Reduction.MEANVAR, include_other=False, where=mask)
    x = self._apply_linear(x, self.linear_player)
    x = self.act(x)
    # Pool over outcomes
    x = reduce(x, (-1,), Reduction.MEANVAR, include_other=False, where=mask)
    x = self._apply_linear(x, self.linear_outcome)
    x = self.act(x)
    return x


class PayoffOutcome2OutcomeLayer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.linear = nn.Linear(
      in_channels * OUTCOME_NUM_POOLS,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.gelu

  def __call__(
    self, x: chex.Array, mask: chex.Array | None = None
  ) -> chex.Array:
    # x: [C, N, A1, ..., AN, O]; mask: optional payoff-outcome mask [A1, ..., AN, O]
    # Collapse ALL the strategy axes (removed, not broadcast back) -> [C, N, O]
    collapsed = collapse_meanvar(x, tuple(range(2, x.ndim - 1)), where=mask)
    # An outcome is valid <=> some valid joint action has it.
    outcome_mask = (
      None if mask is None else jnp.any(mask, axis=tuple(range(mask.ndim - 1)))
    )
    pooled = calc_pooled_outcome_chan(
      collapsed,
      player_reduction=Reduction.MEANVAR,
      outcome_reduction=Reduction.MEANVAR,
      include_identity=True,
      include_all=True,
      include_other=False,
      where=outcome_mask,
    )
    pooled_t = jnp.moveaxis(pooled, 0, -1)
    out_t = self.act(self.linear(pooled_t))
    return jnp.moveaxis(out_t, -1, 0)


class ContractDesignGenerator(nn.Module):
  """Main block for the contract design generator.
  Matches Figure 5.
  """

  def __init__(
    self,
    num_players: int,
    po2po_channels: int,
    o2o_channels: int,
    num_po2po: int,
    num_o2o: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    # PO2PO layers
    self.po2po_layers = nn.List(
      [
        PayoffOutcome2PayoffOutcomeLayer(
          3 if i == 0 else po2po_channels,
          po2po_channels,
          rngs=rngs,
        )
        for i in range(num_po2po)
      ]
    )
    # PO2O: collapse actions
    self.po2o = PayoffOutcome2OutcomeLayer(
      po2po_channels, o2o_channels, rngs=rngs
    )
    # O2O layers
    self.o2o_layers = nn.List(
      [
        Outcome2OutcomeLayer(
          o2o_channels,
          1 if i == num_o2o - 1 else o2o_channels,
          last_activation=(i == num_o2o - 1),
          rngs=rngs,
        )
        for i in range(num_o2o)
      ]
    )

  def __call__(
    self,
    costs: chex.Array,
    transition: chex.Array,
    principal_payoff: chex.Array,
    mask: chex.Array | None = None,
    outcome_mask: chex.Array | None = None,
  ) -> chex.Array:
    """
    Args:
        costs (chex.Array): array of shape [N, A1, ..., AN]
        transition (chex.Array): array of shape [A1, ..., AN, O]
        principal_payoff (chex.Array): array of shape [O, ]
        mask (chex.Array, optional): joint action mask [A1, ..., AN].
        Defaults to None.
        outcome_mask (chex.Array, optional): valid outcomes [O]. Defaults
        to None.
    Returns:
        contracts: chex.Array, shaped as [N, O]
    """

    # Payoff-outcome mask [A1, ..., AN, O] for the layers that see the
    # outcome axis (a [A1, ..., AN] mask would not broadcast against it).
    if mask is None and outcome_mask is None:
      po_mask = None
    elif outcome_mask is None:
      po_mask = mask[..., None]
    elif mask is None:
      po_mask = outcome_mask
    else:
      po_mask = mask[..., None] & outcome_mask

    # Broadcast all to [3, N, A1, ..., AN, O]

    N, *A = costs.shape
    Oc = principal_payoff.shape[0]

    # Expand dimensions
    costs_exp = jnp.expand_dims(costs, axis=-1)  # [N, A..., 1]
    costs_exp = jnp.broadcast_to(costs_exp, (N,) + tuple(A) + (Oc,))

    trans_exp = jnp.expand_dims(transition, axis=0)  # [1, A..., O]
    trans_exp = jnp.broadcast_to(trans_exp, (N,) + tuple(A) + (Oc,))

    principal_exp = jnp.reshape(principal_payoff, (1,) + (1,) * len(A) + (Oc,))
    principal_exp = jnp.broadcast_to(principal_exp, (N,) + tuple(A) + (Oc,))

    x = jnp.stack(
      [costs_exp, trans_exp, principal_exp], axis=0
    )  # [3, N, A..., O]

    # PO2PO layers
    for layer in self.po2po_layers:
      x = layer(x, po_mask)

    # PO2O: collapse to [C, N, O]
    x = self.po2o(x, po_mask)

    # O2O layers
    for layer in self.o2o_layers:
      x = layer(x, outcome_mask)

    # Squeeze channel dim: [1, N, O] → [N, O]
    contracts = x.squeeze(0)

    if outcome_mask is not None:
      contracts = contracts * outcome_mask

    return contracts


class SchedulingGenerator(nn.Module):
  """Main block for the scheduling generator."""

  def __init__(
    self,
    num_players: int,
    channel_list: list,
    max_delta: float = 1.0,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.num_players = num_players
    self.max_delta = max_delta
    channel_list = [1] + channel_list
    # Hidden layers
    self.hidden = nn.List(
      [
        PayoffToPayoffLayer(num_players, in_c, out_c, rngs=rngs)
        for in_c, out_c in itertools.pairwise(channel_list)
      ]
    )
    self.final = nn.Linear(channel_list[-1], 1, rngs=rngs)

    # JUST AS AN EXPERIMENT
    # Scalar gate for residual: taxes = gate * (-max_delta * sigmoid(...))
    # Initialize gate near 0 so taxes start near 0
    self.gate = nn.Param(jnp.array(0.01))

  def __call__(
    self, base_payoffs: chex.Array, mask: chex.Array = None
  ) -> chex.Array:
    x = jnp.expand_dims(base_payoffs, axis=0)
    for layer in self.hidden:
      x = layer(x, mask)
    x_t = jnp.moveaxis(x, 0, -1)
    taxes_raw = self.final(x_t)
    taxes_raw = jnp.squeeze(taxes_raw, axis=-1)

    taxes = -self.max_delta * nn.sigmoid(taxes_raw)
    taxes = self.gate[...] * taxes
    # Zero taxes on padded joint actions: [N, M, ..., M] * [M, ..., M]
    if mask is not None:
      taxes = taxes * mask

    return taxes
