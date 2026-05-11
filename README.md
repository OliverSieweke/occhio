<p align="center">
  <img src="docs/_static/occhio-dark.svg" alt="occhio" width="120">
</p>

<h1 align="center">occhio</h1>

<p align="center">
  <em>/ˈɔk.kjo/ — like Tokyo</em>
</p>

<p align="center">
  Named for the Italian word for "eye" — because we're trying to see what's hiding in those hidden dimensions — and for a certain toy who wanted to be real, because these are, after all, toy models dreaming of becoming generalizable ;)
</p>

<p align="center">
  <a href="https://occhio.dev">Documentation</a> ·
  <a href="https://occhio.dev/source/getting_started.html">Getting Started</a> ·
  <a href="https://github.com/OliverSieweke/occhio/tree/main/examples">Examples</a>
</p>

---

## What is occhio?

occhio is a library for studying **superposition** in toy models of neural networks — the phenomenon where networks represent more features than they have dimensions. Built on the framework from Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html), occhio provides the tools to systematically investigate how features are encoded, how they interfere, and how sparse autoencoders can recover them.

## Installation

### From PyPI

```bash
pip install occhio
```

### From source

```bash
git clone https://github.com/OliverSieweke/occhio.git
cd occhio
```

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                    # Install all dependencies
uv run python -c "import occhio; print('Ready!')"
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .           # Editable install
```

For dev/docs extras: `uv sync --extra dev --extra docs`

### Device support

occhio runs on **CPU**, **CUDA**, and **Apple MPS**. All production development and the majority of testing was done on MPS. Pass `device="mps"` or `device="cuda"` to ToyModel, or let it auto-detect from your autoencoder.

## Quick Start

```python
from occhio import ToyModel
from occhio.distributions import SparseUniform
from occhio.autoencoders import TiedLinearRelu

model = ToyModel(
    distribution=SparseUniform(n_features=50, p_active=0.01),
    ae=TiedLinearRelu(n_features=50, n_hidden=15),
)
losses, _ = model.fit(n_epochs=25_000, learning_rate=3e-4)

print(f"Superposition: {model.superposition.item():.3f}")
print(f"Feature norms: {model.feature_norms}")
```

## Features

- **[ToyModel](https://occhio.dev/source/concepts.html)** — single-line setup for distribution + autoencoder experiments with built-in training, metrics, and SAE integration
- **[ModelGrid](https://occhio.dev/source/concepts.html)** — vectorized parameter sweeps across sparsity, importance, architecture, and more
- **[15+ distributions](https://occhio.dev/source/distributions.html)** — sparse, correlated, hierarchical, manifold-based, DAG-structured, simplicial, and synthetic
- **[9 autoencoder architectures](https://occhio.dev/source/autoencoders.html)** — tied linear, MLP, attention, compute, synthetic, with architecture-invariant save/load
- **[SAE training & evaluation](https://occhio.dev/source/concepts.html)** — train and evaluate sparse autoencoders via [SAE Lens](https://github.com/jbloomAus/SAELens) with automatic feature recovery metrics
- **[Visualization](https://occhio.dev/source/visualization.html)** — Plotly-based plots with automatic faceting, sliders, composite layouts, and light/dark mode
- **[HuggingFace Hub](https://occhio.dev/source/autoencoders.html)** — push and pull pretrained models with `model.push_to_hub()` / `AutoEncoderBase.from_hub()`

## Architecture at a Glance

```
Distribution ──┐
               ├── ToyModel ──── .fit() ───── Geometric Analysis
AutoEncoder ───┘       │                      (norms, dims, interference, superposition)
                       │
                       ├──── .train_saes() ── SAE Evaluation
                       │                      (F1, MCC, explained variance, L0)
                       │
                       └──── ModelGrid ────── Systematic Sweeps
                                              (vectorized training, snapshots, faceted viz)
```

## Examples

The [`examples/`](https://github.com/OliverSieweke/occhio/tree/main/examples) directory contains executed Jupyter notebooks:

| Notebook | What it demonstrates |
|---|---|
| [Getting Started](examples/getting_started.ipynb) | First experiment: train, inspect metrics, visualize |
| [Architecture Comparison](examples/architecture_comparison.ipynb) | TiedLinearRelu vs MLP vs TiedMLP |
| [SAE Feature Recovery](examples/sae_feature_recovery.ipynb) | L1 sweep, precision-recall tradeoff |
| [Correlated Features](examples/correlated_features.ipynb) | How correlation structure affects learning |
| [Distribution Zoo](examples/distribution_zoo.ipynb) | 6 distribution types compared |
| [Superposition Geometry](examples/superposition_geometry.ipynb) | Gram matrix, dimensionalities, density sweep |
| [ModelGrid Sweeps](examples/model_grid_sweeps.ipynb) | 1D/2D parameter sweeps with snapshots |

## Documentation

Full documentation is hosted at **[occhio.dev](https://occhio.dev)**.

To build locally:

```bash
pip install occhio[docs]
cd docs && make html
# Open docs/_build/html/index.html
```

## Citation

```bibtex
@software{occhio2025,
  title   = {occhio: A Library for Studying Superposition in Toy Models},
  author  = {Kupper, Niclas and Reddy, Kaushik and Sieweke, Oliver and Ayonrinde, Kola},
  year    = {2025},
  url     = {https://github.com/OliverSieweke/occhio},
}
```

## License

See [LICENSE](LICENSE) for details.
