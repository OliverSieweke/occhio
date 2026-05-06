# %% [markdown]
# # Architecture Comparison: Does the Bottleneck Shape What Gets Learned?
#
# This experiment trains four autoencoder architectures on identical data
# and compares what each one learns. The question is whether architecture
# matters — do different bottlenecks impose different inductive biases on
# the learned feature geometry, even when the data, training budget,
# and hyperparameters are held constant?
#
# Architectures tested:
# 1. **TiedLinearRelu** — Linear encoder, tied linear decoder + ReLU.
#    The simplest possible bottleneck.
# 2. **TiedMLPEncoder** — MLP encoder with tied (transposed) decoder.
#    More expressive encoding, but weights are shared.
# 3. **MLPEncoder** — Independent MLP encoder and decoder.
#    No weight tying between encoder and decoder.
# 4. **AttnLinearAE** — Multi-head softmax attention encoder with
#    a linear decoder. A fundamentally different bottleneck: the latent
#    is a convex combination of dictionary vectors.

# %%
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio import ToyModel
from occhio.autoencoders import (
    TiedLinearRelu,
    TiedMLPEncoder,
    MLPEncoder,
    AttnLinearAE,
)
from occhio.distributions import SparseUniform, HierarchicalPairs

# %%
# -- Shared experiment parameters --

N_FEATURES = 8
N_HIDDEN = 4
N_EPOCHS = 4000
BATCH_SIZE = 1024
LR = 1e-3
SEED = 42
DEVICE = "cpu"

# Importance decay: earlier features matter more (standard Anthropic setup)
importances = torch.tensor([0.9**i for i in range(N_FEATURES)])

# %% [markdown]
# ## Part 1: SparseUniform — The Standard Benchmark
#
# SparseUniform(p_active=0.1) gives each feature a 10% chance of firing,
# with magnitudes drawn from Uniform(0, 1). This is the standard setup
# from Toy Models of Superposition. With 8 features in 4 dimensions,
# the model must decide which features to represent faithfully and which
# to compress into superposition.

# %%
# -- Build one model per architecture --


def make_generator():
    """Fresh generator with the same seed for fair comparison."""
    return torch.Generator(DEVICE).manual_seed(SEED)


def build_models(distribution):
    """Construct a dict of {name: ToyModel} for each architecture."""
    models = {}

    # 1. TiedLinearRelu — the baseline
    gen = make_generator()
    ae_linear = TiedLinearRelu(
        N_FEATURES,
        N_HIDDEN,
        device=DEVICE,
        generator=gen,
    )
    models["TiedLinearRelu"] = ToyModel(
        distribution,
        ae_linear,
        importances=importances,
    )

    # 2. TiedMLPEncoder — MLP encoder, tied decoder
    # dims=[input, hidden_layer, latent]. The intermediate layer
    # gives the encoder nonlinear capacity without adding decoder params.
    gen = make_generator()
    ae_tied_mlp = TiedMLPEncoder(
        dims=[N_FEATURES, 16, N_HIDDEN],
        device=DEVICE,
        generator=gen,
    )
    models["TiedMLPEncoder"] = ToyModel(
        distribution,
        ae_tied_mlp,
        importances=importances,
    )

    # 3. MLPEncoder — fully independent encoder and decoder
    # embedding: input -> wider -> latent
    # unembedding: latent -> wider -> output
    gen = make_generator()
    ae_mlp = MLPEncoder(
        embedding=[N_FEATURES, 16, N_HIDDEN],
        unembedding=[N_HIDDEN, 16, N_FEATURES],
        device=DEVICE,
        generator=gen,
    )
    models["MLPEncoder"] = ToyModel(
        distribution,
        ae_mlp,
        importances=importances,
    )

    # 4. AttnLinearAE — multi-head softmax attention bottleneck
    # n_hidden must be divisible by n_heads. 2 heads with value_dim=2
    # each, concatenated to form the 4-dim latent.
    gen = make_generator()
    ae_attn = AttnLinearAE(
        n_features=N_FEATURES,
        n_hidden=N_HIDDEN,
        n_heads=2,
        dict_size=8,
        device=DEVICE,
        generator=gen,
    )
    models["AttnLinearAE"] = ToyModel(
        distribution,
        ae_attn,
        importances=importances,
    )

    return models


# %%
# -- Train all architectures --

dist_sparse = SparseUniform(N_FEATURES, p_active=0.1, device=DEVICE)
models_sparse = build_models(dist_sparse)

all_losses = {}
for name, tm in models_sparse.items():
    print(f"Training {name}...")
    losses, _ = tm.fit(
        N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        verbose=False,
    )
    all_losses[name] = losses
    final_loss = losses[-1]
    print(f"  Final loss: {final_loss:.6f}")

# %% [markdown]
# ## Loss Curves
#
# Three of the four architectures (TiedLinearRelu, TiedMLPEncoder,
# AttnLinearAE) converge to similar low losses. The outlier is
# **MLPEncoder**, which converges to a substantially higher loss
# despite having the most parameters. Without weight tying, the
# encoder and decoder can drift apart, making optimization harder
# in this small-scale setting.

# %%
# -- Plot loss curves --

colors = {
    "TiedLinearRelu": "#636EFA",
    "TiedMLPEncoder": "#EF553B",
    "MLPEncoder": "#00CC96",
    "AttnLinearAE": "#AB63FA",
}

fig_loss = go.Figure()
for name, losses in all_losses.items():
    # Subsample for readability
    step = max(1, len(losses) // 500)
    epochs = list(range(0, len(losses), step))
    sampled = [losses[i] for i in epochs]
    fig_loss.add_trace(
        go.Scatter(
            x=epochs,
            y=sampled,
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=2),
        )
    )

fig_loss.update_layout(
    title="Training Loss: SparseUniform(p_active=0.1)",
    xaxis_title="Epoch",
    yaxis_title="Loss (importance-weighted MSE)",
    yaxis_type="log",
    template="plotly_white",
    height=450,
    width=750,
)
fig_loss.show()

# %% [markdown]
# ## Feature Geometry: Norms and Dimensionalities
#
# **Feature norms** (||w_i||) tell us how much capacity the model
# allocates to each feature. In a sparse regime, features with higher
# importance should get larger norms.
#
# **Feature dimensionality** measures how many hidden dimensions each
# feature effectively occupies, accounting for interference from other
# features. A feature with dimensionality close to 1.0 has a clean,
# dedicated direction; values below 1.0 indicate superposition.

# %%
# -- Collect geometric metrics --

metrics = {}
for name, tm in models_sparse.items():
    metrics[name] = {
        "feature_norms": tm.feature_norms.cpu().numpy(),
        "feature_dimensionalities": tm.feature_dimensionalities.cpu().numpy(),
        "superposition": tm.superposition.item(),
        "total_interference": tm.total_feature_interferences.cpu().numpy(),
        "final_loss": all_losses[name][-1],
    }

# Print summary table
print(
    f"{'Architecture':<20} {'Loss':>8} {'Superposition':>14} "
    f"{'Mean Dim':>10} {'Mean Norm':>10}"
)
print("-" * 66)
for name, m in metrics.items():
    print(
        f"{name:<20} {m['final_loss']:>8.5f} "
        f"{m['superposition']:>14.4f} "
        f"{np.mean(m['feature_dimensionalities']):>10.4f} "
        f"{np.mean(m['feature_norms']):>10.4f}"
    )

# %%
# -- Bar chart: feature norms by architecture --

fig_norms = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Feature Norms (||w_i||)", "Feature Dimensionalities"],
    horizontal_spacing=0.12,
)

for name in models_sparse:
    fig_norms.add_trace(
        go.Bar(
            x=[f"f{i}" for i in range(N_FEATURES)],
            y=metrics[name]["feature_norms"],
            name=name,
            marker_color=colors[name],
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig_norms.add_trace(
        go.Bar(
            x=[f"f{i}" for i in range(N_FEATURES)],
            y=metrics[name]["feature_dimensionalities"],
            name=name,
            marker_color=colors[name],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig_norms.update_layout(
    title="Feature Geometry: SparseUniform(p_active=0.1)",
    barmode="group",
    template="plotly_white",
    height=400,
    width=900,
)
fig_norms.show()

# %% [markdown]
# ## W Matrix Visualization
#
# The W matrix (n_hidden x n_features) is the learned feature embedding.
# Each column is a feature's direction in the hidden space. Comparing
# these across architectures reveals whether they discover the same
# or different geometric arrangements.

# %%
# -- Heatmaps of W matrices --

fig_w = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=list(models_sparse.keys()),
    horizontal_spacing=0.06,
)

for col_idx, (name, tm) in enumerate(models_sparse.items(), 1):
    W = tm.W.cpu().numpy()
    fig_w.add_trace(
        go.Heatmap(
            z=W,
            x=[f"f{i}" for i in range(N_FEATURES)],
            y=[f"h{i}" for i in range(N_HIDDEN)],
            colorscale="RdBu_r",
            zmid=0,
            showscale=(col_idx == 4),
            colorbar=dict(title="Weight") if col_idx == 4 else None,
        ),
        row=1,
        col=col_idx,
    )

fig_w.update_layout(
    title="Learned W Matrices (n_hidden x n_features)",
    template="plotly_white",
    height=300,
    width=1000,
)
fig_w.show()

# %% [markdown]
# ## Interference Structure
#
# The cosine similarity matrix between feature embeddings shows which
# features interfere with each other. Ideally, features with similar
# importance but different identities should be orthogonal (zero cosine
# similarity). In practice, superposition forces some features to share
# directions.

# %%
# -- Cosine similarity matrices --

fig_cos = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=list(models_sparse.keys()),
    horizontal_spacing=0.06,
)

for col_idx, (name, tm) in enumerate(models_sparse.items(), 1):
    cos_sim = tm.cosine_similarity_matrix.cpu().numpy()
    labels = [f"f{i}" for i in range(N_FEATURES)]
    fig_cos.add_trace(
        go.Heatmap(
            z=cos_sim,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(col_idx == 4),
            colorbar=dict(title="cos sim") if col_idx == 4 else None,
        ),
        row=1,
        col=col_idx,
    )

fig_cos.update_layout(
    title="Feature Cosine Similarity (Interference Structure)",
    template="plotly_white",
    height=350,
    width=1000,
)
fig_cos.show()

# %% [markdown]
# ## Part 2: HierarchicalPairs — Does the Distribution Change the Ranking?
#
# SparseUniform treats all features as independent. HierarchicalPairs
# creates parent-child pairs where the child only fires when the parent
# does. This structured correlation changes the effective sparsity pattern,
# which may shift the architecture ranking.

# %%
# -- Build and train on HierarchicalPairs --

dist_hier = HierarchicalPairs(
    N_FEATURES,
    p_active=0.15,
    p_follow=0.7,
    device=DEVICE,
)
models_hier = build_models(dist_hier)

losses_hier = {}
for name, tm in models_hier.items():
    print(f"Training {name} on HierarchicalPairs...")
    losses, _ = tm.fit(
        N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        verbose=False,
    )
    losses_hier[name] = losses
    print(f"  Final loss: {losses[-1]:.6f}")

# %%
# -- Collect metrics for HierarchicalPairs --

metrics_hier = {}
for name, tm in models_hier.items():
    metrics_hier[name] = {
        "feature_norms": tm.feature_norms.cpu().numpy(),
        "feature_dimensionalities": tm.feature_dimensionalities.cpu().numpy(),
        "superposition": tm.superposition.item(),
        "total_interference": tm.total_feature_interferences.cpu().numpy(),
        "final_loss": losses_hier[name][-1],
    }

print(
    f"\n{'Architecture':<20} {'Loss':>8} {'Superposition':>14} "
    f"{'Mean Dim':>10} {'Mean Norm':>10}"
)
print("-" * 66)
for name, m in metrics_hier.items():
    print(
        f"{name:<20} {m['final_loss']:>8.5f} "
        f"{m['superposition']:>14.4f} "
        f"{np.mean(m['feature_dimensionalities']):>10.4f} "
        f"{np.mean(m['feature_norms']):>10.4f}"
    )

# %% [markdown]
# ## Side-by-Side Comparison
#
# The architecture ranking is **stable** across distributions:
# TiedMLPEncoder wins on loss in both cases, MLPEncoder loses in both,
# and TiedLinearRelu and AttnLinearAE land in the middle. No crossovers.

# %%
# -- Comparison bar chart --

arch_names = list(models_sparse.keys())

fig_compare = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Final Loss by Distribution",
        "Superposition (rho_mm) by Distribution",
    ],
    horizontal_spacing=0.15,
)

# Loss comparison
for i, name in enumerate(arch_names):
    fig_compare.add_trace(
        go.Bar(
            x=["SparseUniform", "HierarchicalPairs"],
            y=[metrics[name]["final_loss"], metrics_hier[name]["final_loss"]],
            name=name,
            marker_color=colors[name],
            showlegend=True,
        ),
        row=1,
        col=1,
    )

# Superposition comparison
for i, name in enumerate(arch_names):
    fig_compare.add_trace(
        go.Bar(
            x=["SparseUniform", "HierarchicalPairs"],
            y=[metrics[name]["superposition"], metrics_hier[name]["superposition"]],
            name=name,
            marker_color=colors[name],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig_compare.update_layout(
    title="Architecture Comparison Across Distributions",
    barmode="group",
    template="plotly_white",
    height=400,
    width=900,
)
fig_compare.show()

# %%
# -- Loss curves for HierarchicalPairs --

fig_loss_hier = go.Figure()
for name, losses in losses_hier.items():
    step = max(1, len(losses) // 500)
    epochs = list(range(0, len(losses), step))
    sampled = [losses[i] for i in epochs]
    fig_loss_hier.add_trace(
        go.Scatter(
            x=epochs,
            y=sampled,
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=2),
        )
    )

fig_loss_hier.update_layout(
    title="Training Loss: HierarchicalPairs(p_active=0.15, p_follow=0.7)",
    xaxis_title="Epoch",
    yaxis_title="Loss (importance-weighted MSE)",
    yaxis_type="log",
    template="plotly_white",
    height=450,
    width=750,
)
fig_loss_hier.show()

# %% [markdown]
# ## Interference Patterns Under Hierarchy
#
# With HierarchicalPairs, features come in parent-child pairs
# (f0-f1, f2-f3, ...). Paired features co-occur, so the model may
# place them in similar directions. The cosine similarity matrices
# below show whether the different architectures discover this
# pair structure.

# %%
# -- Cosine similarity for HierarchicalPairs --

fig_cos_hier = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=list(models_hier.keys()),
    horizontal_spacing=0.06,
)

for col_idx, (name, tm) in enumerate(models_hier.items(), 1):
    cos_sim = tm.cosine_similarity_matrix.cpu().numpy()
    labels = [f"f{i}" for i in range(N_FEATURES)]
    fig_cos_hier.add_trace(
        go.Heatmap(
            z=cos_sim,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(col_idx == 4),
            colorbar=dict(title="cos sim") if col_idx == 4 else None,
        ),
        row=1,
        col=col_idx,
    )

fig_cos_hier.update_layout(
    title="Feature Cosine Similarity: HierarchicalPairs",
    template="plotly_white",
    height=350,
    width=1000,
)
fig_cos_hier.show()

# %% [markdown]
# ## Summary
#
# Key findings from this experiment:
#
# 1. **Weight tying helps, not hurts.** TiedMLPEncoder consistently
#    achieves the lowest loss across both distributions. MLPEncoder
#    (no weight tying) performs worst by a large margin (~5-8x higher
#    loss), despite having the most parameters. In this small-scale
#    setting, the shared decoder constraint acts as beneficial
#    regularization.
#
# 2. **AttnLinearAE matches TiedLinearRelu, not exceeding it.** The
#    attention bottleneck does not produce qualitatively different
#    feature geometry here -- both converge to similar loss, superposition,
#    and dimensionality values. The softmax constraint neither helps
#    nor hurts at this scale.
#
# 3. **The architecture ranking is stable across distributions.**
#    TiedMLPEncoder wins on both SparseUniform and HierarchicalPairs.
#    MLPEncoder loses on both. No crossovers occur, suggesting the
#    ranking reflects optimization properties rather than data-specific
#    inductive biases.
#
# 4. **Superposition varies with architecture.** TiedMLPEncoder achieves
#    notably lower superposition (~0.7) than the others (~0.9-1.0),
#    indicating it finds a less compressed representation. This likely
#    stems from its nonlinear encoder providing better feature separation.
