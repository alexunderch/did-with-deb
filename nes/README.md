# NES (Neural Equilibrium solver)

Minimcal API:
```Python
solver = NESolver(
    game=pyspiel.load_game("matrix_rps"),  # or games.Game.L2_INVARIANT
    mode=networks.Mode.CCE,                # or Mode.CE
    network_config={
        "dual_channels": 32,
        "payoff_channel_list": [128, 64, 128],
        "dual_channel_list": [32, 32],
    },
    objective=samplers.Objective.EPS_MWMRE,
    welfare_coeff=1.0,
    entropy_coeff=1.0,
    epsilon_max=10,
    batch_size=32,
    network_train_steps=100,
)
result = solver.solve()  # → Frozen Differentiable Block
```

The main driver file is

```Bash
>> python nes_main.py --help
>> nes_main.py:
  --batch_size: Number of transitions to sample at each learning step.
    (default: 64) (an integer)
  --checkpoint_dir: Directory to save/load the agent models.
    (default: '/tmp/nes_test')
  --decay: Weight decay of the optimiser.
    (default: 1e-7) (a number)
  --dual_channel_list: Number of hidden units in the dual-to-dual MLP
    (default: '32,32') (a comma separated list)
  --dual_channels: Number of hidden units in the payoff-to-dual layer
    (default: 64) (an integer)
  --entropy_coeff: Minimum Relative Entropy coefficient of the loss function.
    (default: 10.0) (a number)
  --epsilon_max: Epsilon plus coefficient of the loss function. (a number)
  --eval_batch_size: Number of transitions to sample at each evaluation step.
    (default: 10) (an integer)
  --iterations: Number of training episodes.
    (default: 60000) (an integer)
  --learning_rate: Learning rate of the solver`s updates.
    (default: 0.0001) (a number)
  --log_every: Episode frequency at which the NESolver is evaluated.
    (default: 5000) (an integer)
  --max_actions: Number of maximum actions for all games. (an integer)
  --max_grad_norm: Max allowed gradient norm.
    (default: 1.0) (a number)
  --mode: <CE|CCE>: What equilibrium, CE or CCE do we look for? (default: 'CCE')
  --norm: Norm of the payoff tensor.
    (default: 2) (an integer)
  --num_strategies: A tuple defining number of the players` actions (A1, ..., AN)
    (a comma separated list)
  --openspiel_games: OpenSpiel games to sample from. Can specify multiple times.;
    repeat this option to specify a list of values
    (default: '[]')
  --payoff_channel_list: Number of hidden units in the payoff-to-payoff MLP
    (default: '32,32,32,32,32') (a comma separated list)
  --random_games: <EMPIRICAL_DISC_GAME|L1_INVARIANT|L2_INVARIANT|LINF_INVARIANT>: Random game
    types to sample from. Can specify multiple times.;
    repeat this option to specify a list of values
    (default: '[]')
  --save_every: Episode frequency at which the the solver`s weights are saved.
    (default: 10000) (an integer)
  --[no]save_for_deb: Save json config for using it as a DEB later. (default: 'false')
  --seed: A random seed.
    (default: 42) (an integer)
  --[no]use_checkpoints: Save/load neural network weights.
    (default: 'false')
  --welfare_coeff: Maximum Welfare coefficient of the loss function.
    (default: 4.0) (a number)
```


### Motivation

This section implements a Neural Equilibrium Solver (NES) that learns to compute ε-approximate Correlated Equilibria (CE) and Correlated Course Equilibria (CCE) via duality of their respective linear programmes and their optimisation. The main idea of the papers is to train a neural network to predict dual variables α, from which primal variables (joint strategy σ and slacks ε) are recovered analytically through the ε-MWMRE (Maximum Welfare Minimum Relative Entropy) objective.

### Method

The network consumes a 4-channel tensor of shape `[B, 4, N, A1, ..., AN]`:

- Channel 0: Normalized payoffs `Ĝ`
- Channel 1: Target epsilon `ε̂` (broadcasted)
- Channel 2: Target joint strategy `σ̂` (broadcasted)
- Channel 3: Welfare `W` (broadcasted)

The pipeline is following

```
    Input: [B, 4, N, A1, ..., AN]
        ↓
    [1] EquivariantPayoffToPayoff × K   (maintains joint action space)
        ↓
    [2] PayoffsToDuals                  (marginalises to player-action space)
        ↓
    [3] EquivariantDualToDual × L       (refines in individual space)
        ↓
    [4] Final projection + Softplus     (ensures α ≥ 0)
    Output: α — [B, 1, N, A] for CCE, [B, 1, N, A, A] for CE
```

Given duals α, the closed-form primal recovery implements Equation (7) from the paper [1]:
1. Compute deviation contributions per player (`gain_p`)
2. Form logits: `l(a) = μ·W(a) − Σ_p gain_p(a)`
3. Recover `σ(a) ∝ σ̂(a) · exp(l(a))`
4. Recover `ε_p = (ε̂_p − ε⁺) · exp(−Σα_p / ρ) + ε⁺`


The dual loss minimized by `AdamW`:
```
L_dual = log_sum_exp + ε⁺ · Σ_p Σ_a α_p(a) − ρ · Σ_p ε_p
```

## Reproduction plan

| # | Test                                 | What It Proves                                   |
| - | ------------------------------------ | ------------------------------------------------ |
| 1 | Training convergence on `8×8`        | The unsupervised loss works                      |
| 2 | In-distribution accuracy (`8×8`)     | NES matches iterative solvers                    |
| 3 | Zero-shot on `2×2`, `4×4`, `8×8`     | Shape-independent generalization                 |
| 4 | MWME vs. MRE vs. ε variants          | Flexible selection framework                     |
| 5 | CCE vs. CE duals                     | Both solution concepts are learnable             |
| 6 | 2×2 canonical games                  | Qualitative correctness & polytope visualization |


```Bash
python examples/nes_study.py
```
