# Neural equilibrium solvers and differentiable mechanism design for game theory

> [!WARNING]
> Despite the implementations being aligned with common sense, there might be plenty silent bugs present,
> which I am constantly fixing. Stay tooned! 

This repository implements the following papers:
1. Turbocharging Solution Concepts: Solving NEs, CEs and CCEs with Neural Equilibrium Solvers ([arXiv:2210.09257](https://arxiv.org/abs/2210.09257))
2. Deep Incentive Design with Differentiable Equilibrium Blocks ([arXiv:2603.07705](https://arxiv.org/abs/2603.07705))

> [!IMPORTANT]
> Overall goal of the project is to provide support for differential game solvers and mechanism designers for normal form games (NFG).

Each subproject is equipped with own `Readme` documentation.

## Installation

We recommend doing it through [`uv`](https://docs.astral.sh/uv/) and use `uv sync`.

But the working alternative is (assuming you run in the `venv` for `python>=3.14`):

```Bash
pip install -e .
```

> [!NOTE]
> Unfortunately, due to absense of any hardware acceleration, we only use CPU-based solutions.

As the repository is written in `JAX`, you can use power of its device agnosticism prowess:
* GPU/TPU installation: [official support](https://docs.jax.dev/en/latest/installation.html)
* MPS installation (experimental): [repo](https://github.com/tillahoffmann/jax-mps)

## Testing

> [!NOTE]
> This section is the most up to redesign. Current version uses `abseil`'s legacy which is hard to automate 
> without bash scripts or `Bazel`.

```Bash
bash run_tests.sh
```

## Contents

### Neural equilibrium solvers (NES)

- Documentation: [link](./nes/README.md)
- Driver file: `python nes_main.py`
- Paper study: `python examples/nes_study.py`

### Differential Incentive Design with differentiable equilibrium blocks

> [!NOTE]
> This paper re-uses plenly of the NES content.

- Documentation: [link](./did/README.md)
- Driver file: `python did_main.py`
- Paper study: TODO

## Licensing, contribution, and AI

1. Do whatever you fancy with the code — it's `Apache 2.0` licensed.
2. Despite some free avilable AIs were used for bug checking and design, all the code is proofread mostly typed by myself.
3. Contributions are accepted but by your own risk: it's my repo, I do whatever I want with it. 