# %% [markdown]
# # Superposition Phase Transitions
#
# When a neural network has more features than hidden dimensions, it faces a
# choice: represent some features faithfully and ignore others, or represent
# all features approximately via **superposition** — packing feature directions
# into a space too small to hold them orthogonally.
#
# This example systematically maps the conditions under which superposition
# emerges. We sweep sparsity (p_active), bottleneck ratio (n_hidden/n_features),
# and build a 2D phase diagram inspired by Anthropic's "Toy Models of
# Superposition" paper.
#
# All experiments use occhio's ToyModel with TiedLinearRelu autoencoders and
# SparseUniform distributions.

# %% Imports

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from occhio import ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import SparseUniform

# %% Configuration

DEVICE = "cpu"
SEED = 42
N_EPOCHS = 5000
BATCH_SIZE = 1024
LR = 1e-3

# ============================================================================
# Part 1: Sparsity Sweep
# ============================================================================
# Fix the bottleneck (10 features into 5 hidden dims) and vary how often
# each feature is active. The question: does sparsity affect the degree
# of superposition at this bottleneck ratio?

# %% Sparsity sweep — setup

N_FEATURES = 10
N_HIDDEN = 5

# Sweep from dense (50% active) to very sparse (1% active).
# We use a log-spaced sweep because the interesting transitions happen
# at low sparsity values.
p_active_values = np.logspace(np.log10(0.01), np.log10(0.5), 12)

sparsity_results = []

# %% Sparsity sweep — train models

print("=== Part 1: Sparsity Sweep ===")
print(f"n_features={N_FEATURES}, n_hidden={N_HIDDEN}")
print(f"Sweeping p_active over {len(p_active_values)} values\n")

for p in tqdm(p_active_values, desc="Sparsity sweep"):
    gen = torch.Generator(DEVICE).manual_seed(SEED)

    dist = SparseUniform(N_FEATURES, p_active=float(p), generator=gen)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen)
    tm = ToyModel(dist, ae, device=DEVICE)

    losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)

    # Collect geometric measurements
    result = {
        "p_active": float(p),
        "superposition": tm.superposition.item(),
        "feature_norms": tm.feature_norms.cpu().numpy().copy(),
        "feature_dims": tm.feature_dimensionalities.cpu().numpy().copy(),
        "total_interference": tm.total_feature_interferences.cpu().numpy().copy(),
        "mean_interference": tm.total_feature_interferences.mean().item(),
        "final_loss": losses[-1],
        "W": tm.W.cpu().numpy().copy(),
        "cosine_sim": tm.cosine_similarity_matrix.cpu().numpy().copy(),
    }
    sparsity_results.append(result)

    print(
        f"  p={p:.3f}  superposition={result['superposition']:.3f}  "
        f"mean_dim={result['feature_dims'].mean():.2f}  "
        f"loss={result['final_loss']:.6f}"
    )

# %% Sparsity sweep — plot metrics

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Superposition (mean max cosine sim)",
        "Feature Norms",
        "Feature Dimensionalities",
        "Total Feature Interference",
    ),
    horizontal_spacing=0.12,
    vertical_spacing=0.12,
)

p_vals = [r["p_active"] for r in sparsity_results]

# Panel 1: Superposition metric
fig.add_trace(
    go.Scatter(
        x=p_vals,
        y=[r["superposition"] for r in sparsity_results],
        mode="lines+markers",
        name="rho_mm",
        marker=dict(size=8),
        line=dict(width=2),
    ),
    row=1,
    col=1,
)

# Panel 2: Feature norms — each feature gets a trace
for i in range(N_FEATURES):
    fig.add_trace(
        go.Scatter(
            x=p_vals,
            y=[r["feature_norms"][i] for r in sparsity_results],
            mode="lines",
            name=f"feat {i}",
            line=dict(width=1.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

# Panel 3: Feature dimensionalities
for i in range(N_FEATURES):
    fig.add_trace(
        go.Scatter(
            x=p_vals,
            y=[r["feature_dims"][i] for r in sparsity_results],
            mode="lines",
            name=f"feat {i}",
            line=dict(width=1.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

# Panel 4: Mean total interference
fig.add_trace(
    go.Scatter(
        x=p_vals,
        y=[r["mean_interference"] for r in sparsity_results],
        mode="lines+markers",
        name="mean interference",
        marker=dict(size=8),
        line=dict(width=2, color="crimson"),
    ),
    row=2,
    col=2,
)

# All panels share log-scaled x-axes
for row in [1, 2]:
    for col in [1, 2]:
        fig.update_xaxes(type="log", title_text="p_active", row=row, col=col)

fig.update_yaxes(title_text="rho_mm", row=1, col=1)
fig.update_yaxes(title_text="||w_i||", row=1, col=2)
fig.update_yaxes(title_text="dims per feature", row=2, col=1)
fig.update_yaxes(title_text="interference", row=2, col=2)

fig.update_layout(
    title_text=(f"Sparsity Sweep: {N_FEATURES} features, {N_HIDDEN} hidden dims"),
    height=700,
    width=950,
    showlegend=False,
)
fig.show()

# %% [markdown]
# **What we observe:**
#
# With 10 features and 5 hidden dimensions (a 2:1 ratio), the model learns
# 5 antipodal pairs at **all** sparsity levels — superposition (rho_mm) stays
# pinned at 1.0 and mean dimensionality at 0.5 throughout the sweep. The
# model always chooses full superposition because the bottleneck is severe
# enough that packing features into antipodal pairs is optimal regardless of
# how often they co-activate.
#
# What **does** change with sparsity is loss: the reconstruction error grows
# steadily as p_active increases, because co-activation becomes more frequent
# and the antipodal encoding causes more interference. Feature norms stay
# near 1.0 everywhere — the model represents all 10 features via superposition
# rather than dropping any, even in the dense regime.
#
# To see a genuine sparsity-driven phase transition in rho_mm, a milder
# bottleneck (e.g., n_hidden=8 for 10 features) would be needed so the model
# has a real choice between orthogonal and superposed representations.

# ============================================================================
# Part 2: Bottleneck Ratio Sweep
# ============================================================================
# Fix sparsity at p_active=0.03 (fairly sparse) and vary the hidden
# dimension. When n_hidden >= n_features, there is no need for superposition.

# %% Bottleneck sweep — train models

P_ACTIVE_FIXED = 0.03
n_hidden_values = list(range(2, 13))

bottleneck_results = []

print("\n=== Part 2: Bottleneck Ratio Sweep ===")
print(f"n_features={N_FEATURES}, p_active={P_ACTIVE_FIXED}")
print(f"Sweeping n_hidden from {n_hidden_values[0]} to {n_hidden_values[-1]}\n")

for n_h in tqdm(n_hidden_values, desc="Bottleneck sweep"):
    gen = torch.Generator(DEVICE).manual_seed(SEED)

    dist = SparseUniform(N_FEATURES, p_active=P_ACTIVE_FIXED, generator=gen)
    ae = TiedLinearRelu(N_FEATURES, n_h, generator=gen)
    tm = ToyModel(dist, ae, device=DEVICE)

    losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)

    result = {
        "n_hidden": n_h,
        "ratio": n_h / N_FEATURES,
        "superposition": tm.superposition.item(),
        "feature_norms": tm.feature_norms.cpu().numpy().copy(),
        "feature_dims": tm.feature_dimensionalities.cpu().numpy().copy(),
        "mean_dim": tm.feature_dimensionalities.mean().item(),
        "mean_interference": tm.total_feature_interferences.mean().item(),
        "n_alive": int((tm.feature_norms > 0.1).sum().item()),
        "final_loss": losses[-1],
    }
    bottleneck_results.append(result)

    print(
        f"  n_hidden={n_h:2d}  ratio={result['ratio']:.1f}  "
        f"superposition={result['superposition']:.3f}  "
        f"alive={result['n_alive']:2d}/{N_FEATURES}  "
        f"loss={result['final_loss']:.6f}"
    )

# %% Bottleneck sweep — plot

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Superposition vs Bottleneck Ratio",
        "Alive Features",
        "Mean Feature Dimensionality",
    ),
    horizontal_spacing=0.1,
)

ratios = [r["ratio"] for r in bottleneck_results]

fig.add_trace(
    go.Scatter(
        x=ratios,
        y=[r["superposition"] for r in bottleneck_results],
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2),
    ),
    row=1,
    col=1,
)
# Mark ratio=1.0 (n_hidden == n_features) with a vertical line
fig.add_vline(
    x=1.0,
    line=dict(dash="dash", color="gray"),
    row=1,
    col=1,
    annotation_text="n_hidden = n_features",
    annotation_position="top right",
    annotation_font_size=10,
)

fig.add_trace(
    go.Scatter(
        x=ratios,
        y=[r["n_alive"] for r in bottleneck_results],
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2, color="green"),
    ),
    row=1,
    col=2,
)
fig.add_hline(
    y=N_FEATURES,
    line=dict(dash="dot", color="gray"),
    row=1,
    col=2,
    annotation_text=f"all {N_FEATURES} features",
    annotation_position="bottom right",
    annotation_font_size=10,
)

fig.add_trace(
    go.Scatter(
        x=ratios,
        y=[r["mean_dim"] for r in bottleneck_results],
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2, color="orange"),
    ),
    row=1,
    col=3,
)
fig.add_hline(
    y=1.0,
    line=dict(dash="dot", color="gray"),
    row=1,
    col=3,
    annotation_text="1D per feature",
    annotation_position="bottom right",
    annotation_font_size=10,
)

for col in [1, 2, 3]:
    fig.update_xaxes(title_text="n_hidden / n_features", row=1, col=col)
fig.update_yaxes(title_text="rho_mm", row=1, col=1)
fig.update_yaxes(title_text="# alive features (norm > 0.1)", row=1, col=2)
fig.update_yaxes(title_text="dims per feature", row=1, col=3)

fig.update_layout(
    title_text=f"Bottleneck Sweep: p_active={P_ACTIVE_FIXED}",
    height=400,
    width=1100,
    showlegend=False,
)
fig.show()

# %% [markdown]
# **What we observe:**
#
# Superposition decreases as n_hidden grows toward n_features. At ratio=1.0
# (n_hidden=10), rho_mm drops to near 0 and loss vanishes — every feature
# gets its own orthogonal direction. Below this threshold the model
# superposes, packing all 10 features into fewer dimensions by sharing
# directions.
#
# All 10 features remain "alive" (norm > 0.1) at every bottleneck setting,
# even n_hidden=2. Rather than dropping features, the model always represents
# all of them via superposition and tolerates the resulting interference —
# feasible here because p_active=0.03 makes co-activation rare.
#
# The superposition curve is not perfectly monotonic: there is some
# non-monotonicity at small n_hidden values (e.g., rho jumps at ratio=0.5)
# reflecting different local optima in how the model packs features.

# ============================================================================
# Part 3: 2D Phase Diagram
# ============================================================================
# The classic result: superposition as a function of both sparsity and
# bottleneck ratio. We sweep a coarse grid and plot a heatmap.

# %% Phase diagram — train grid

PHASE_N_FEATURES = 10
phase_p_actives = np.logspace(np.log10(0.01), np.log10(0.5), 8)
phase_n_hiddens = list(range(2, 12))

# Store results in a 2D array: rows = p_active, cols = n_hidden
superposition_grid = np.zeros((len(phase_p_actives), len(phase_n_hiddens)))
alive_grid = np.zeros_like(superposition_grid)
loss_grid = np.zeros_like(superposition_grid)

total = len(phase_p_actives) * len(phase_n_hiddens)
print(f"\n=== Part 3: 2D Phase Diagram ===")
print(f"Grid: {len(phase_p_actives)} sparsities x {len(phase_n_hiddens)} hidden dims")
print(f"Total models to train: {total}\n")

with tqdm(total=total, desc="Phase diagram") as pbar:
    for i, p in enumerate(phase_p_actives):
        for j, n_h in enumerate(phase_n_hiddens):
            gen = torch.Generator(DEVICE).manual_seed(SEED)

            dist = SparseUniform(PHASE_N_FEATURES, p_active=float(p), generator=gen)
            ae = TiedLinearRelu(PHASE_N_FEATURES, n_h, generator=gen)
            tm = ToyModel(dist, ae, device=DEVICE)

            losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)

            superposition_grid[i, j] = tm.superposition.item()
            alive_grid[i, j] = int((tm.feature_norms > 0.1).sum().item())
            loss_grid[i, j] = losses[-1]

            pbar.update(1)

# %% Phase diagram — plot superposition heatmap

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Superposition (rho_mm)",
        "Alive Features (norm > 0.1)",
    ),
    horizontal_spacing=0.15,
)

# Custom labels for axes
p_labels = [f"{p:.3f}" for p in phase_p_actives]
h_labels = [str(n) for n in phase_n_hiddens]

fig.add_trace(
    go.Heatmap(
        z=superposition_grid,
        x=h_labels,
        y=p_labels,
        colorscale="Viridis",
        colorbar=dict(title="rho_mm", x=0.42),
        zmin=0,
        zmax=1,
        hovertemplate=(
            "n_hidden: %{x}<br>p_active: %{y}<br>superposition: %{z:.3f}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=alive_grid,
        x=h_labels,
        y=p_labels,
        colorscale="YlGnBu",
        colorbar=dict(title="# alive", x=1.0),
        zmin=0,
        zmax=PHASE_N_FEATURES,
        hovertemplate=(
            "n_hidden: %{x}<br>p_active: %{y}<br>alive: %{z:.0f}<extra></extra>"
        ),
    ),
    row=1,
    col=2,
)

for col in [1, 2]:
    fig.update_xaxes(title_text="n_hidden", row=1, col=col)
    fig.update_yaxes(title_text="p_active", row=1, col=col)

fig.update_layout(
    title_text=(
        f"Phase Diagram: {PHASE_N_FEATURES} features "
        f"({len(phase_p_actives)}x{len(phase_n_hiddens)} grid)"
    ),
    height=500,
    width=1000,
)
fig.show()

# %% [markdown]
# **Interpreting the phase diagram:**
#
# The dominant factor is the **bottleneck ratio** (n_hidden / n_features).
# When n_hidden >= n_features, superposition drops to near 0 regardless of
# sparsity — the model has enough room for orthogonal features.
#
# When n_hidden < n_features, superposition is high across the board.
# Sparsity has a much weaker effect on rho_mm than the bottleneck ratio in
# this setup. This is because rho_mm measures the *geometric arrangement*
# of feature directions (max cosine similarity), not the expected
# reconstruction cost. A model can have high geometric superposition yet low
# loss if features rarely co-activate.
#
# The **alive features** heatmap shows that when features are sparse, the
# model keeps all 10 alive even at extreme compression. At higher p_active
# values with small n_hidden, some features are dropped (norm near 0)
# because the interference cost of superposing dense features becomes
# prohibitive.

# ============================================================================
# Part 4: Feature Geometry Snapshots
# ============================================================================
# For a few interesting parameter settings, visualize the W^T W matrix
# (cosine similarities between feature embeddings). This shows the
# geometric structure of how features are arranged in hidden space.

# %% Select interesting configurations and visualize W^T W

# Pick four parameter settings that span different regimes.
# With 10 features, n_hidden=8 gives mild compression (ratio=0.8) while
# n_hidden=3 creates severe compression (ratio=0.3).
snapshot_configs = [
    {"p_active": 0.3, "n_hidden": 8, "label": "Dense, wide bottleneck"},
    {"p_active": 0.03, "n_hidden": 8, "label": "Sparse, wide bottleneck"},
    {"p_active": 0.3, "n_hidden": 3, "label": "Dense, narrow bottleneck"},
    {"p_active": 0.03, "n_hidden": 3, "label": "Sparse, narrow bottleneck"},
]

N_SNAP_FEATURES = 10

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[c["label"] for c in snapshot_configs],
    horizontal_spacing=0.12,
    vertical_spacing=0.15,
)

for idx, cfg in enumerate(snapshot_configs):
    gen = torch.Generator(DEVICE).manual_seed(SEED)

    dist = SparseUniform(N_SNAP_FEATURES, p_active=cfg["p_active"], generator=gen)
    ae = TiedLinearRelu(N_SNAP_FEATURES, cfg["n_hidden"], generator=gen)
    tm = ToyModel(dist, ae, device=DEVICE)
    tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)

    cos_sim = tm.cosine_similarity_matrix.cpu().numpy()
    norms = tm.feature_norms.cpu().numpy()
    rho = tm.superposition.item()

    row = idx // 2 + 1
    col = idx % 2 + 1

    # W^T W heatmap (cosine similarities between feature directions)
    fig.add_trace(
        go.Heatmap(
            z=cos_sim,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(idx == 1),
            colorbar=dict(title="cos sim", x=1.02) if idx == 1 else None,
            hovertemplate="feat %{x} vs %{y}: %{z:.3f}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # Annotate with superposition metric and alive count
    n_alive = int((norms > 0.1).sum())
    # Plotly subplot axes: first is "x"/"y", subsequent are "x2","y2", etc.
    ax_suffix = "" if idx == 0 else str(idx + 1)
    fig.add_annotation(
        text=f"rho={rho:.2f}, alive={n_alive}/{N_SNAP_FEATURES}",
        xref=f"x{ax_suffix} domain",
        yref=f"y{ax_suffix} domain",
        x=0.5,
        y=-0.18,
        showarrow=False,
        font=dict(size=11),
    )

fig.update_layout(
    title_text="Feature Cosine Similarity (W^T W) Across Regimes",
    height=700,
    width=800,
)
fig.show()

# %% Print geometry summary for each snapshot

print("\n=== Part 4: Feature Geometry Summary ===\n")
for cfg in snapshot_configs:
    gen = torch.Generator(DEVICE).manual_seed(SEED)

    dist = SparseUniform(N_SNAP_FEATURES, p_active=cfg["p_active"], generator=gen)
    ae = TiedLinearRelu(N_SNAP_FEATURES, cfg["n_hidden"], generator=gen)
    tm = ToyModel(dist, ae, device=DEVICE)
    tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)

    norms = tm.feature_norms.cpu().numpy()
    dims = tm.feature_dimensionalities.cpu().numpy()
    interference = tm.total_feature_interferences.cpu().numpy()
    rho = tm.superposition.item()

    print(f"--- {cfg['label']} ---")
    print(f"  p_active={cfg['p_active']}, n_hidden={cfg['n_hidden']}")
    print(f"  Superposition (rho_mm):    {rho:.4f}")
    print(f"  Feature norms:             {np.array2string(norms, precision=2)}")
    print(f"  Feature dimensionalities:  {np.array2string(dims, precision=2)}")
    print(f"  Total interferences:       {np.array2string(interference, precision=3)}")
    print(f"  Alive features (>0.1):     {int((norms > 0.1).sum())}/{N_SNAP_FEATURES}")
    print(
        f"  Embedded feats / hidden:   {tm.embedded_features_per_hidden_dimensions:.2f}"
    )
    print()

# %% [markdown]
# **Key takeaways:**
#
# 1. **The bottleneck ratio is the primary driver of superposition.** When
#    n_hidden >= n_features, every feature gets its own orthogonal direction
#    and rho_mm drops to near 0. Below this threshold, the model superposes.
#
# 2. **Sparsity affects loss more than geometry.** In this setup, the model
#    adopts the same superposed geometry (e.g., antipodal pairs) regardless
#    of sparsity. What sparsity changes is the *cost* of that geometry:
#    sparser features co-activate less, so interference causes less
#    reconstruction error. The expected interference cost scales with
#    p_active^2 for a pair of features.
#
# 3. **Sparsity determines whether features are kept or dropped.** When
#    features are sparse (p_active=0.03), the model keeps all 10 alive even
#    at extreme compression (n_hidden=2). When features are dense
#    (p_active=0.3) and the bottleneck is narrow, the model drops some
#    features entirely (norm near 0) because the interference cost of
#    superposing frequently co-active features becomes too high.
#
# 4. **Feature dimensionality reveals the representation structure.** In the
#    non-superposed regime, each feature uses roughly 1 dimension. In the
#    superposed regime, features share dimensions and their effective
#    dimensionality drops below 1 (e.g., 0.5 for antipodal pairs, ~0.33
#    for triplet structures in 3D).
