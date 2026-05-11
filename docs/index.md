# occhio

A library for studying superposition in toy models of neural networks.

occhio provides controlled environments for investigating how neural networks
represent more features than they have dimensions, including distributions,
autoencoder architectures, geometric analysis, SAE training/evaluation, and
visualization tools.

## Quick Example

```python
from occhio import ToyModel
from occhio.distributions import SparseUniform
from occhio.autoencoders import TiedLinearRelu

model = ToyModel(
    distribution=SparseUniform(n_features=5, p_active=0.1),
    ae=TiedLinearRelu(n_features=5, n_hidden=2),
)
losses, _ = model.fit(n_epochs=10_000)

print(model.feature_norms)
print(model.superposition)
```

## Highlights

- **ToyModel** -- single-line setup for distribution + autoencoder experiments
- **ModelGrid** -- vectorized parameter sweeps with `torch.vmap`
- **15+ distributions** -- sparse, correlated, hierarchical, manifold-based, DAG-structured
- **9 autoencoder architectures** -- tied linear, MLP, attention, compute, synthetic
- **SAE integration** -- train and evaluate sparse autoencoders via SAE Lens
- **Interactive visualization** -- Plotly-based plots with automatic faceting over grids
- **HuggingFace Hub** -- save/load models and distributions from the Hub

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   source/getting_started
   source/concepts
   source/distributions
   source/autoencoders
   source/visualization

.. toctree::
   :maxdepth: 2
   :caption: Reference

   source/api
```

## Citation

If you use occhio in your research, please cite:

```bibtex
@software{occhio2025,
  title   = {occhio: A Library for Studying Superposition in Toy Models},
  author  = {Kupper, Niclas and Reddy, Kaushik and Sieweke, Oliver and Ayonrinde, Kola},
  year    = {2025},
  url     = {https://github.com/OliverSieweke/occhio},
}
```
