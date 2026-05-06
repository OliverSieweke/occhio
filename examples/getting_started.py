# %% [markdown]
# # Getting Started with Occhio
#
# This tutorial walks through the core occhio workflow: defining a sparse
# feature distribution, training an autoencoder to compress it, and
# inspecting the geometric structure of the learned representations.
#
# Along the way you will see **superposition** in action -- the phenomenon
# where a neural network packs more features than it has dimensions by
# overlapping their representations.
#
# Requires: `pip install occhio` (or editable install from this repo).

# %%
# -- Imports ------------------------------------------------------------------

import torch
from occhio import ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.visualization import plot_embedding

# Reproducibility
torch.manual_seed(42)

# %% [markdown]
# ## 1. Create a sparse data distribution
#
# `SparseUniform` generates samples where each of `n_features` features is
# independently active with probability `p_active`.  When active, its value
# is drawn from Uniform(0, 1); otherwise it is zero.
#
# With `p_active=0.05`, roughly 5% of features fire in any given sample.
# This sparsity is what makes superposition possible -- if features rarely
# co-occur, the network can afford to let their representations overlap.

# %%
n_features = 8
n_hidden = 4
p_active = 0.05

dist = SparseUniform(n_features=n_features, p_active=p_active, device="cpu")

# Draw a small batch to see what the data looks like
samples = dist.sample(batch_size=5)
print("Sample shape:", samples.shape)
print("Expected active features per sample (L0):", dist.expected_l0)
print()
print("Five samples (rows=samples, cols=features):")
print(samples.numpy().round(3))
print()
# Count non-zeros to verify sparsity
nonzeros = (samples > 0).sum(dim=1).float()
print(f"Active features per sample: {nonzeros.tolist()}")

# %% [markdown]
# ## 2. Create an autoencoder
#
# `TiedLinearRelu` is a single-layer autoencoder with tied weights and a
# ReLU activation on the decoder output:
#
#     encode(x) = x @ W.T          (n_features -> n_hidden)
#     decode(z) = ReLU(z @ W + b)  (n_hidden -> n_features)
#
# The key constraint: `n_features=8 > n_hidden=4`, so the encoder must
# compress 8 features into 4 dimensions.  This is the bottleneck that
# forces the network to make representational trade-offs.

# %%
ae = TiedLinearRelu(n_features=n_features, n_hidden=n_hidden, device="cpu")

print(f"Autoencoder: {n_features} features -> {n_hidden} hidden dims")
print(f"Compression ratio: {n_features / n_hidden:.1f}x")
print(f"Learnable parameters: {sum(p.numel() for p in ae.parameters())}")

# %% [markdown]
# ## 3. Train with ToyModel
#
# `ToyModel` ties together a distribution and an autoencoder.  Calling
# `.fit()` runs standard SGD training: sample a batch from the distribution,
# encode-decode it, compute MSE loss, and update weights.

# %%
model = ToyModel(dist, ae, device="cpu")

losses, _ = model.fit(
    n_epochs=5000,
    batch_size=512,
    learning_rate=1e-3,
)

print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss:   {losses[-1]:.6f}")

# %% [markdown]
# ### Loss curve
#
# We can plot the training loss to verify convergence.  The loss drops
# quickly as the network learns to reconstruct the sparse inputs.

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=losses,
        mode="lines",
        line=dict(width=1),
        name="Training loss",
    )
)
fig.update_layout(
    title="Training Loss",
    xaxis_title="Epoch",
    yaxis_title="MSE Loss",
    yaxis_type="log",
    height=400,
    width=600,
)
fig.show()

# %% [markdown]
# ## 4. Inspect geometric properties
#
# After training, `ToyModel` exposes properties that describe how features
# are represented in the hidden space.  These are the core tools for
# studying superposition.

# %% [markdown]
# ### The embedding matrix W
#
# `model.W` is the weight matrix of shape `(n_hidden, n_features)`.
# Each column is a feature's embedding vector in the hidden space.

# %%
W = model.W
print(f"W shape: {W.shape}  (n_hidden x n_features)")
print()
print("Embedding matrix W:")
print(W.numpy().round(4))

# %% [markdown]
# ### Feature norms
#
# The norm of each feature's embedding vector tells you how strongly it
# is represented.  A norm near 1.0 means the feature is fully represented;
# a norm near 0 means the network has effectively dropped it.
#
# With equal importances (the default) and high sparsity, the network
# can represent all features with near-unit norm by using antipodal
# pairs (see the interference matrix below).

# %%
norms = model.feature_norms
print("Feature norms (how strongly each feature is represented):")
for i, n in enumerate(norms.tolist()):
    bar = "#" * int(n * 40)
    print(f"  Feature {i}: {n:.4f}  {bar}")

# %% [markdown]
# ### Superposition metric
#
# The `superposition` property computes the mean maximum absolute cosine
# similarity between feature embeddings.  It ranges from 0 (all features
# orthogonal -- no superposition) to 1 (features fully overlapping).
#
# With 8 features in 4 dimensions, the network cannot make all features
# orthogonal.  At high sparsity (p_active=0.05), superposition is the
# optimal strategy: overlap features that rarely co-occur.

# %%
rho = model.superposition.item()
print(f"Superposition (rho_mm): {rho:.4f}")
if rho > 0.5:
    print("  -> High superposition: features are overlapping significantly.")
elif rho > 0.1:
    print("  -> Moderate superposition: some feature overlap.")
else:
    print("  -> Low superposition: features are nearly orthogonal.")

# %% [markdown]
# ### Feature dimensionalities
#
# Each feature occupies a certain number of "effective dimensions" in
# the hidden space.  In a 4D space with 8 features, if all features
# are equally represented, each gets about 0.5 dimensions on average.
# Highly superposed features share dimensions.

# %%
dims = model.feature_dimensionalities
print("Feature dimensionalities (effective dims per feature):")
for i, d in enumerate(dims.tolist()):
    bar = "#" * int(d * 40)
    print(f"  Feature {i}: {d:.4f}  {bar}")
print(f"\nMean: {dims.mean():.4f}")
print(f"Sum:  {dims.sum():.4f}  (hidden dims = {n_hidden})")

# %% [markdown]
# ### Interference matrix
#
# The `interferences` matrix shows how much each pair of features
# interferes with each other.  Diagonal entries are self-similarity
# (always 1.0 for normalized features); off-diagonal entries reveal
# which features the network has allowed to overlap.
#
# In this case you should see **antipodal pairs**: pairs of features
# with cosine similarity near -1.0 (e.g. features 0 & 6, 2 & 5, etc.).
# The network packs 8 features into 4 dimensions by placing each pair
# at opposite ends of the same axis.  The ReLU in the decoder ensures
# that only the correct feature is reconstructed for a given input.

# %%
interference = model.interferences
print("Interference matrix (cosine similarities between features):")
print(interference.numpy().round(3))

# %% [markdown]
# ## 5. Visualize embeddings in 2D
#
# To directly see where features land in the hidden space, we train a
# model with `n_hidden=2`.  Each feature's embedding is a 2D vector
# that we can draw as an arrow from the origin.
#
# `plot_embedding` creates an interactive plotly figure with one arrow
# per feature.  With high sparsity, you should see features spreading
# out in many directions -- this is superposition in action.

# %%
# Train a 2D model for visualization
dist_2d = SparseUniform(n_features=n_features, p_active=p_active, device="cpu")
ae_2d = TiedLinearRelu(n_features=n_features, n_hidden=2, device="cpu")
model_2d = ToyModel(dist_2d, ae_2d, device="cpu")

model_2d.fit(n_epochs=5000, batch_size=512, learning_rate=1e-3)

print(f"2D model superposition: {model_2d.superposition.item():.4f}")
print(f"2D model final loss: unneeded (see plot)")

fig = plot_embedding(model_2d)
fig.update_layout(
    title="Feature Embeddings in 2D Hidden Space (high sparsity)",
    height=500,
    width=500,
)
fig.show()

# %% [markdown]
# ## 6. Explore the sparsity-superposition trade-off
#
# The key insight from the Toy Models of Superposition paper: sparsity
# controls how much superposition a network uses.
#
# - **High sparsity** (p_active=0.01): features rarely co-occur, so the
#   network freely overlaps them.  More features fit in fewer dims.
# - **Low sparsity** (p_active=0.5): features often co-occur, so overlap
#   causes large reconstruction errors.  The network may shrink feature
#   norms (effectively dropping some features) rather than finding
#   orthogonal arrangements, especially when `n_features >> n_hidden`.
#
# Let's train both and compare.

# %%
results = {}

for label, p in [("High sparsity (p=0.01)", 0.01), ("Low sparsity (p=0.5)", 0.5)]:
    d = SparseUniform(n_features=n_features, p_active=p, device="cpu")
    a = TiedLinearRelu(n_features=n_features, n_hidden=2, device="cpu")
    m = ToyModel(d, a, device="cpu")
    losses_i, _ = m.fit(n_epochs=5000, batch_size=512, learning_rate=1e-3)

    results[label] = {
        "model": m,
        "final_loss": losses_i[-1],
        "superposition": m.superposition.item(),
        "mean_norm": m.feature_norms.mean().item(),
        "mean_dimensionality": m.feature_dimensionalities.mean().item(),
    }

# %%
# -- Compare metrics side by side --
print(f"{'Metric':<30} {'High sparsity':>15} {'Low sparsity':>15}")
print("-" * 62)
for metric in ["final_loss", "superposition", "mean_norm", "mean_dimensionality"]:
    high = results["High sparsity (p=0.01)"][metric]
    low = results["Low sparsity (p=0.5)"][metric]
    print(f"{metric:<30} {high:>15.4f} {low:>15.4f}")

# %% [markdown]
# Both models show very high superposition (~1.0) because `n_features`
# far exceeds `n_hidden` in both cases.  The real difference is in
# **mean_norm**: the high-sparsity model keeps all features at full
# strength (~1.0) while the low-sparsity model shrinks norms (~0.5),
# effectively under-representing features it cannot reconstruct well.
# The low-sparsity model also has much higher final loss, reflecting
# the unavoidable interference when features frequently co-occur.

# %%
# -- Visualize both embeddings --
fig_high = plot_embedding(results["High sparsity (p=0.01)"]["model"])
fig_high.update_layout(
    title="High Sparsity (p=0.01) -- features spread via superposition",
    height=450,
    width=450,
)
fig_high.show()

fig_low = plot_embedding(results["Low sparsity (p=0.5)"]["model"])
fig_low.update_layout(
    title="Low Sparsity (p=0.5) -- features shrink in norm",
    height=450,
    width=450,
)
fig_low.show()

# %% [markdown]
# ## Summary
#
# You have seen the core occhio workflow:
#
# 1. **Define** a sparse data distribution with `SparseUniform`
# 2. **Build** an autoencoder bottleneck with `TiedLinearRelu`
# 3. **Train** by combining them in a `ToyModel` and calling `.fit()`
# 4. **Analyze** the learned geometry with properties like
#    `superposition`, `feature_norms`, `feature_dimensionalities`,
#    and `interferences`
# 5. **Visualize** feature embeddings with `plot_embedding`
# 6. **Compare** how sparsity controls the degree of superposition
#
# Next steps:
# - Try different autoencoder architectures (`TiedLinear`, `TiedMLPEncoder`)
# - Vary feature importances with the `importances` parameter
# - Use `ModelGrid` to sweep over parameters systematically
# - Explore correlated features with `CorrelatedPairs` or
#   `HierarchicalPairs` distributions
