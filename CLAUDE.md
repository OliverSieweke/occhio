# occhio — Claude Code Guide

## What this project is

`occhio` is a Python research library for experimenting with [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — a mechanistic interpretability research setup where neural networks are trained to compress sparse feature distributions into lower-dimensional representations. The library provides composable building blocks (distributions, autoencoders, training loops, visualization) to make it easy to run and analyze these experiments.

## Key concepts

- **`ToyModel`** ([src/occhio/toy_model.py](src/occhio/toy_model.py)): Core experiment object. Combines a `Distribution` with an `AutoEncoderBase`. Provides `.fit()`, `.sample_latent()`, `.get_one_hot_embeddings()`, and geometric analysis properties (`W`, `feature_norms`, `interferences`, `feature_dimensionalities`, etc.).
- **`ModelGrid`** ([src/occhio/model_grid.py](src/occhio/model_grid.py)): Vectorized grid of `ToyModel`s parameterized over one or more `Axis` values. Uses `torch.vmap` + `torch.compile` for fast parallel training. Supports `snapshot_interval` to capture training dynamics as a `TrainingAxis`.
- **Distributions** ([src/occhio/distributions/](src/occhio/distributions/)): All subclass `Distribution`. Must implement `.sample(batch_size) -> Tensor`. See `distributions/README.md` for a full taxonomy.
- **AutoEncoders** ([src/occhio/autoencoder.py](src/occhio/autoencoder.py)): All subclass `AutoEncoderBase(nn.Module)`. Must implement `.encode()`, `.decode()`, `.resample_weights()`, and set `self.n_features` + `self.n_hidden` in `__init__`.
- **SAEs** ([src/occhio/sae/sae.py](src/occhio/sae/sae.py)): Sparse AutoEncoders with L1 sparsity penalties, separate from the `AutoEncoderBase` hierarchy.

## Where things live

```
src/occhio/
├── __init__.py              # Exports: ToyModel, ModelGrid, AutoEncoderBase
├── toy_model.py             # ToyModel class
├── model_grid.py            # ModelGrid, Axis, TrainingAxis
├── autoencoder.py           # AutoEncoderBase, TiedLinear, TiedLinearRelu, MLPEncoder, ComputeAutoEncoder
├── distributions/
│   ├── base.py              # Distribution (ABC), DistributionStack
│   ├── sparse.py            # SparseUniform, SparseExponential, SingleUniform
│   ├── correlated.py        # CorrelatedPairs, HierarchicalPairs, ScaledHierarchicalPairs, AnticorrelatedPairs
│   ├── relational.py        # RelationalSimple, MultiRelational
│   ├── hierarchical.py      # HierarchicalSparse
│   ├── dag.py               # DAGDistribution, DAGBayesianPropagation, DAGRandomWalkToRoot, PowerLawDigraph
│   └── README.md            # Distribution taxonomy docs
├── sae/sae.py               # SAESimple, TopKIgnoreSAE, CausalSAE
├── visualization/           # Plotly-based plotting (embedding, geometry, phase change, etc.)
├── utils/device.py          # _same_device() helper
└── examples/                # Runnable example scripts and notebooks

experiments/                 # Ad-hoc research notebooks and scripts (not packaged)
tests/                       # Top-level pytest tests
```

## How to run things

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a specific test file
uv run pytest src/occhio/tests/test_model_grid.py

# Lint / format
uv run ruff check .
uv run ruff format .
```

## Conventions

- **Device handling**: `ToyModel` resolves device from the `ae`, `distribution`, or explicit `device` argument. Distribution and AE can live on different devices (e.g. CPU distribution, MPS AE) — samples are moved automatically. Don't create tensors on hardcoded devices; let the model/distribution handle it.
- **`AutoEncoderBase` contract**: Every subclass `__init__` must set `self.n_features` and `self.n_hidden`. This is enforced at construction via `__init_subclass__`. Custom loss functions can be passed at construction time.
- **`Distribution` contract**: `sample(batch_size)` returns `Tensor` of shape `(batch_size, n_features)`, or a tuple where the first element is the input tensor. Pass `generator=` for reproducible sampling; required when using `ModelGrid` with `cache_samples=True`.
- **`ModelGrid` factory**: The `create_model` function must accept a `params: dict[str, Any]` keyword argument. All AEs in a grid must share the same architecture (shape-compatible state dicts).
- **Linting**: `ruff` is configured with docstring code formatting (`ruff.toml`). Type checking uses `ty` (not mypy).
- **`# ABOUTME:` comments**: Module-level files use two `# ABOUTME:` lines describing the file's purpose.
- **Notebook hygiene**: `nbstripout` is in dev deps — notebooks should have outputs stripped before commit.
