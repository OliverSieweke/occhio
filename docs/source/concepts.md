# Core Concepts

## Superposition

Neural networks often represent more features than they have dimensions -- a
phenomenon called **superposition**. Features share directions in activation
space, trading off interference for representational capacity. occhio provides
controlled environments for studying this tradeoff.

For background, see [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
(Elhage et al., 2022).

## ToyModel

{class}`~occhio.ToyModel` is the central object. It combines three ingredients:

1. **Distribution** -- how ground-truth feature activations are sampled
2. **AutoEncoder** -- the bottleneck that must learn to reconstruct features
3. **Importances** -- per-feature weights in the reconstruction loss

```python
model = ToyModel(
    distribution=SparseUniform(10, p_active=0.05),
    ae=TiedLinearRelu(10, 5),
    importances=torch.logspace(-1, 0, 10),
)
losses, _ = model.fit(n_epochs=20_000)
```

After training, `ToyModel` exposes the learned geometry as properties:

| Property | Description |
|---|---|
| `W` | Weight matrix (n_hidden, n_features) |
| `feature_norms` | L2 norm of each feature's embedding |
| `feature_dimensionalities` | Effective dimensionality per feature |
| `interferences` | Pairwise cosine similarity of feature embeddings |
| `superposition` | Mean max absolute cosine similarity (scalar) |

## ModelGrid

{class}`~occhio.ModelGrid` runs systematic parameter sweeps over a grid of
`ToyModel` instances. It uses `torch.vmap` for vectorized parallel training.

```python
grid = ModelGrid(create_model, axes=[
    Axis("Density", logspace(0, -2, 16)),
    Axis("Importance", logspace(-1, 1, 16)),
])
grid.fit(n_epochs=10_000)
```

Key features:

- **Vectorized training** via `torch.vmap` + `torch.compile` -- all models
  train in parallel
- **Sample broadcasting** -- models with equivalent distributions share
  samples, reducing memory and compute
- **Snapshot history** -- pass `snapshot_interval` to `fit()` to capture
  training dynamics as a new `ModelGrid` with a `TrainingAxis`
- **Save/load** -- `grid.save("grid.pkl")` / `ModelGrid.load("grid.pkl")`

## SAE Training and Evaluation

occhio integrates with [SAE Lens](https://github.com/jbloomAus/SAELens) for
training and evaluating Sparse Autoencoders on toy model activations.

```python
from sae_lens import TrainingSAE
from occhio import SAEEntry

# Define SAEs to train
model.train_saes([
    SAEEntry(
        sae=TrainingSAE.from_dict({...}),
        type="Standard",
    ),
])

# Evaluate against ground truth
results = model.evaluate_saes()
print(model.saes_f1_score)
print(model.saes_mcc)
```

Evaluation metrics include precision, recall, F1, MCC, explained variance,
L0 sparsity, dead latents, shrinkage, and uniqueness -- all accessible as
dict properties on `ToyModel` (e.g., `model.saes_f1_score`).

## Feature Geometry

The geometric relationship between learned feature embeddings reveals how the
autoencoder allocates capacity:

- **Feature norms** (`model.feature_norms`) -- features with norms near 1 are
  fully represented; near 0 means the feature is dropped
- **Dimensionality** (`model.feature_dimensionalities`) -- how many effective
  dimensions each feature occupies (1.0 = dedicated direction, lower =
  compressed/shared)
- **Interference** (`model.interferences`) -- pairwise cosine similarity
  between feature embeddings; high interference means features share directions
- **Superposition** (`model.superposition`) -- a scalar summary: mean of each
  feature's max absolute cosine similarity to any other feature
