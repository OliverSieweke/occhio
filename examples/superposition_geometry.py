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
# and build a 2D phase diagram replicating the central result from Anthropic's
# "Toy Models of Superposition" paper.
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
# each feature is active. The central question: how sparse must features be
# before the network decides to superpose them?

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
# **What we expect to observe:**
#
# As p_active decreases (features become sparser), superposition increases.
# This is because sparse features are rarely active simultaneously, so the
# interference from packing them non-orthogonally is tolerable — the network
# accepts some reconstruction error on the rare occasions two superposed
# features co-activate, in exchange for representing more features overall.
#
# Feature norms should remain near 1.0 for represented features (the model
# "keeps" them) and drop toward 0 for features it abandons. In the dense
# regime, the model can only faithfully represent n_hidden features. In the
# sparse regime, it represents all n_features via superposition.

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
# **What we expect to observe:**
#
# Superposition should "turn off" as n_hidden approaches n_features. When
# there is enough room for every feature to have its own dimension, the model
# has no incentive to superpose — each feature gets a near-orthogonal
# direction and dimensionality approaches 1.0.
#
# Below the critical ratio, the model packs more features than dimensions
# by exploiting sparsity. The number of "alive" features (norm > 0.1)
# should exceed n_hidden, indicating superposition.

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
# The diagram should show two regimes separated by a phase boundary:
#
# 1. **No superposition** (top-right): high p_active AND large n_hidden.
#    Features are dense and there is room for all of them. rho_mm is near 0.
#
# 2. **Full superposition** (bottom-left): low p_active AND small n_hidden.
#    Features are sparse and the bottleneck is severe. rho_mm approaches 1.
#
# The phase boundary is not a sharp line but a smooth transition. The
# "alive features" heatmap shows that in the superposition regime, the
# model represents MORE features than it has hidden dimensions — a hallmark
# of superposition.

# ============================================================================
# Part 4: Feature Geometry Snapshots
# ============================================================================
# For a few interesting parameter settings, visualize the W^T W matrix
# (cosine similarities between feature embeddings). This shows the
# geometric structure of how features are arranged in hidden space.

# %% Select interesting configurations and visualize W^T W

# Pick four parameter settings that span the transition:
# (1) Dense + wide  = no superposition
# (2) Sparse + wide = mild superposition
# (3) Dense + narrow = forced superposition
# (4) Sparse + narrow = full superposition
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
# 1. **Sparsity enables superposition.** When features fire rarely, the model
#    can pack them into shared directions because simultaneous activation
#    (and thus interference) is unlikely. The expected interference cost
#    scales with p_active^2 for a pair of features.
#
# 2. **The bottleneck ratio controls capacity.** When n_hidden >= n_features,
#    every feature can have its own orthogonal direction and superposition is
#    unnecessary. Below this threshold, the model must choose between dropping
#    features (norm -> 0) and superposing them (high cosine similarity).
#
# 3. **The tradeoff is between fidelity and capacity.** Superposition
#    increases the effective number of represented features at the cost of
#    reconstruction accuracy. The optimal tradeoff depends on sparsity:
#    sparser features can tolerate more interference.
#
# 4. **Feature dimensionality reveals the representation structure.** In the
#    non-superposed regime, each feature uses roughly 1 dimension. In the
#    superposed regime, features share dimensions and their effective
#    dimensionality drops below 1.
