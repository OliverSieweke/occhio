# Getting Started

## Installation

```bash
pip install occhio
```

For GPU support, install PyTorch with CUDA first, then install occhio.

## First Experiment

A minimal end-to-end example: create a distribution, an autoencoder, combine them
into a `ToyModel`, train, and inspect the results.

```python
import torch
from occhio import ToyModel
from occhio.distributions import SparseUniform
from occhio.autoencoders import TiedLinearRelu

# 1. Distribution: 5 sparse features, each active 10% of the time
dist = SparseUniform(n_features=5, p_active=0.1)

# 2. Autoencoder: compress 5 features into 2 hidden dimensions
ae = TiedLinearRelu(n_features=5, n_hidden=2)

# 3. ToyModel: ties distribution + autoencoder together
model = ToyModel(distribution=dist, ae=ae)

# 4. Train
losses, _ = model.fit(n_epochs=10_000, batch_size=512)

# 5. Inspect learned geometry
print("Feature norms:", model.feature_norms)
print("Superposition:", model.superposition.item())
print("Final loss:", losses[-1])
```

## Inspecting Metrics

After training, `ToyModel` exposes geometric properties as cached tensors:

```python
model.W                    # Weight matrix (n_hidden, n_features)
model.feature_norms        # Per-feature L2 norms
model.feature_dimensionalities  # Effective dimensionality per feature
model.interferences        # Pairwise interference matrix
model.superposition        # Scalar measure of superposition (rho_mm)
```

## Visualization

```python
from occhio.visualization import EmbeddingPlot

plot = EmbeddingPlot()
fig = plot(model)
fig.show()
```

## Parameter Sweeps with ModelGrid

```python
from occhio import ModelGrid
from occhio.model_grid import Axis

def create_model(params):
    return ToyModel(
        distribution=SparseUniform(5, p_active=params["Density"]),
        ae=TiedLinearRelu(5, 2),
        importances=params["Importance"] ** torch.arange(5),
    )

grid = ModelGrid(
    create_model,
    axes=[
        Axis("Density", [0.01, 0.05, 0.1, 0.5]),
        Axis("Importance", [0.5, 1.0, 2.0]),
    ],
)
grid.fit(n_epochs=10_000)
```

## Next Steps

- {doc}`concepts` -- core abstractions and how they fit together
- {doc}`distributions` -- all distribution types and when to use each
- {doc}`autoencoders` -- autoencoder architectures and customization
- {doc}`visualization` -- plotting and export
- See the example notebooks in `examples/` for deeper explorations
