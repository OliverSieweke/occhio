# %% [markdown]
# # ModelGrid: Systematic Parameter Sweeps
#
# `ModelGrid` trains an entire grid of ToyModels in parallel using
# `torch.vmap`, making parameter sweeps dramatically faster than a
# sequential for-loop. This example covers:
#
# 1. **1D sweep** -- superposition vs. feature density
# 2. **2D sweep** -- phase diagram over density and correlation strength
# 3. **Grid slicing** -- extracting and comparing sub-grids
#
# All models share the same autoencoder architecture (required by vmap),
# but may differ in distribution parameters, importances, or any other
# ToyModel config you route through the `create_model` factory.

# %% Imports
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio import ToyModel, ModelGrid
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.distributions.correlated import HierarchicalPairs
from occhio.model_grid import Axis

# %% Configuration
N_FEATURES = 8
N_HIDDEN = 3
SEED = 42
DEVICE = "cpu"

# ============================================================================
# Part 1: 1D Sweep -- Superposition vs. Feature Density
# ============================================================================

# %% Define the model factory
# ModelGrid requires a `create_model(params)` function that returns a
# ToyModel. The `params` dict maps axis labels to values at a grid point.
# Every model must use the same autoencoder class and shape (vmap
# constraint), but distributions and importances can vary freely.


def create_sparse_model(params):
    """Factory for a SparseUniform ToyModel parameterized by p_active."""
    p_active = params["p_active"]
    gen = torch.Generator(DEVICE).manual_seed(SEED)
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES, p_active=p_active, device=DEVICE, generator=gen
        ),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
        device=DEVICE,
    )


# %% Build a 1D grid over feature density (p_active)
# Log-spaced values give better resolution in the sparse regime.
p_active_values = torch.logspace(-2, 0, 8)  # 0.01 to 1.0

grid_1d = ModelGrid(
    create_model=create_sparse_model,
    axes=[Axis("p_active", p_active_values)],
)

print(f"Grid shape: {grid_1d.shape}")
print(f"Grid description: {grid_1d.description}")
print(f"Total models: {grid_1d.models.size}")

# %% Train all 8 models simultaneously
# Under the hood, ModelGrid stacks all autoencoder parameters and uses
# torch.vmap to compute forward passes and losses in a single batched
# operation. This is typically 3-10x faster than training sequentially.
losses_1d = grid_1d.fit(
    n_epochs=2000,
    batch_size=512,
    learning_rate=3e-4,
    verbose=True,
    track_losses=True,
)

# %% Extract metrics across the grid
# After training, each grid cell holds a fully trained ToyModel with all
# its properties available: feature_norms, superposition, interferences, etc.
superposition_values = []
mean_feature_norm_values = []

for i, p in enumerate(p_active_values):
    model = grid_1d[i]  # 1D grid + int index -> ToyModel directly
    superposition_values.append(model.superposition.item())
    mean_feature_norm_values.append(model.feature_norms.mean().item())
    print(
        f"p_active={p:.3f}  "
        f"superposition={superposition_values[-1]:.3f}  "
        f"mean_norm={mean_feature_norm_values[-1]:.3f}"
    )

# %% Plot: Superposition vs. density
fig_1d = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Superposition vs. Density", "Mean Feature Norm vs. Density"],
)

p_vals = p_active_values.numpy()

fig_1d.add_trace(
    go.Scatter(
        x=p_vals,
        y=superposition_values,
        mode="lines+markers",
        name="superposition",
        marker=dict(size=8),
    ),
    row=1,
    col=1,
)
fig_1d.add_trace(
    go.Scatter(
        x=p_vals,
        y=mean_feature_norm_values,
        mode="lines+markers",
        name="mean ||w_i||",
        marker=dict(size=8),
    ),
    row=1,
    col=2,
)

fig_1d.update_xaxes(type="log", title_text="p_active", row=1, col=1)
fig_1d.update_xaxes(type="log", title_text="p_active", row=1, col=2)
fig_1d.update_yaxes(title_text="superposition (rho_mm)", row=1, col=1)
fig_1d.update_yaxes(title_text="mean feature norm", row=1, col=2)
fig_1d.update_layout(
    title_text="1D Sweep: Superposition and Feature Norms vs. Density",
    height=400,
    width=900,
    showlegend=False,
)
fig_1d.show()


# ============================================================================
# Part 2: 2D Sweep -- Phase Diagram (Density x Correlation)
# ============================================================================

# %% Define a 2D factory using HierarchicalPairs
# HierarchicalPairs organizes features into parent-child pairs. The beta
# parameter controls magnitude coupling: at beta=1 the child copies the
# parent's value exactly; at beta=0 the child gets an independent draw
# scaled by the parent. This correlation structure interacts with
# superposition and is worth exploring across parameter ranges.

N_FEATURES_2D = 6  # must be even for HierarchicalPairs
N_HIDDEN_2D = 3


def create_hierarchical_model(params):
    """Factory for HierarchicalPairs over (p_active, beta)."""
    p_active = params["p_active"]
    beta = params["beta"]
    gen = torch.Generator(DEVICE).manual_seed(SEED)
    return ToyModel(
        distribution=HierarchicalPairs(
            N_FEATURES_2D,
            p_active=p_active,
            p_follow=0.8,  # high follow probability to make correlations visible
            beta=beta,
            device=DEVICE,
            generator=gen,
        ),
        ae=TiedLinearRelu(N_FEATURES_2D, N_HIDDEN_2D, generator=gen, device=DEVICE),
        device=DEVICE,
    )


# %% Build and train a 2D grid
p_active_axis = Axis("p_active", torch.logspace(-1.5, 0, 6))  # ~0.03 to 1.0
beta_axis = Axis("beta", torch.linspace(0.0, 1.0, 5))

grid_2d = ModelGrid(
    create_model=create_hierarchical_model,
    axes=[p_active_axis, beta_axis],
)

print(f"2D grid shape: {grid_2d.shape}  ({grid_2d.models.size} models)")

grid_2d.fit(
    n_epochs=3000,
    batch_size=512,
    learning_rate=3e-4,
    verbose=True,
)

# %% Build heatmaps of superposition and feature norms
# Iterate over the full grid to extract per-model metrics into 2D arrays.
n_pa, n_beta = grid_2d.shape
superposition_map = np.zeros((n_pa, n_beta))
mean_norm_map = np.zeros((n_pa, n_beta))

for i in range(n_pa):
    for j in range(n_beta):
        model = grid_2d[i, j]
        superposition_map[i, j] = model.superposition.item()
        mean_norm_map[i, j] = model.feature_norms.mean().item()

# %% Phase diagram: Superposition across (p_active, beta)
# The heatmaps show how superposition and feature norms vary across
# the (p_active, beta) grid. With 6 features packed into 3 hidden dims,
# superposition stays high throughout, but its magnitude shifts with
# both density and correlation strength.
p_labels = [f"{v:.3f}" for v in p_active_axis.values]
b_labels = [f"{v:.2f}" for v in beta_axis.values]

fig_phase = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Superposition Phase Diagram", "Mean Feature Norm"],
    horizontal_spacing=0.15,
)

fig_phase.add_trace(
    go.Heatmap(
        z=superposition_map,
        x=b_labels,
        y=p_labels,
        colorscale="Viridis",
        colorbar=dict(title="rho_mm", x=0.42),
    ),
    row=1,
    col=1,
)
fig_phase.add_trace(
    go.Heatmap(
        z=mean_norm_map,
        x=b_labels,
        y=p_labels,
        colorscale="Plasma",
        colorbar=dict(title="||w||", x=1.0),
    ),
    row=1,
    col=2,
)

fig_phase.update_yaxes(title_text="p_active", row=1, col=1)
fig_phase.update_yaxes(title_text="p_active", row=1, col=2)
fig_phase.update_xaxes(title_text="beta", row=1, col=1)
fig_phase.update_xaxes(title_text="beta", row=1, col=2)
fig_phase.update_layout(
    title_text="2D Sweep: Density x Correlation Strength (HierarchicalPairs)",
    height=500,
    width=950,
)
fig_phase.show()


# ============================================================================
# Part 3: Grid Slicing -- Comparing Specific Models
# ============================================================================

# %% Slice the 2D grid to get 1D cross-sections
# Integer indexing collapses an axis (NumPy convention):
#   grid[i]    -> 1D sub-grid along second axis
#   grid[:, j] -> 1D sub-grid along first axis
#   grid[i, j] -> single ToyModel
#
# Slice indexing preserves dimensionality:
#   grid[1:4]     -> 2D sub-grid
#   grid[1:4, :3] -> 2D sub-grid

# Fix beta=0 (independent children) and beta=1 (locked children)
slice_independent = grid_2d[:, 0]  # beta=0.0, all p_active values
slice_locked = grid_2d[:, -1]  # beta=1.0, all p_active values

print(f"Slice shape (beta=0): {slice_independent.shape}")
print(f"Slice shape (beta=1): {slice_locked.shape}")

# %% Compare parent-child angle dynamics across beta regimes
# When beta=0 (independent magnitudes), parent and child features behave
# more independently. When beta=1 (locked magnitudes), parent-child
# pairs carry redundant information, which modestly increases
# superposition relative to the beta=0 baseline.
fig_slice = go.Figure()

for label, sub_grid, dash in [
    ("beta=0.00 (independent)", slice_independent, "solid"),
    ("beta=1.00 (locked)", slice_locked, "dash"),
]:
    sup_vals = []
    for i in range(sub_grid.shape[0]):
        model = sub_grid[i]
        sup_vals.append(model.superposition.item())

    fig_slice.add_trace(
        go.Scatter(
            x=np.array([v.item() for v in p_active_axis.values]),
            y=sup_vals,
            mode="lines+markers",
            name=label,
            line=dict(dash=dash),
            marker=dict(size=7),
        )
    )

fig_slice.update_xaxes(type="log", title_text="p_active")
fig_slice.update_yaxes(title_text="superposition (rho_mm)")
fig_slice.update_layout(
    title_text="Grid Slices: Independent vs. Locked Correlations",
    height=400,
    width=700,
)
fig_slice.show()

# %% Inspect individual models from the grid
# Pull out a specific model to examine its learned geometry.
model_sparse_independent = grid_2d[0, 0]  # lowest density, beta=0
model_dense_locked = grid_2d[-1, -1]  # highest density, beta=1

print("--- Sparse + Independent (p_active~0.03, beta=0) ---")
print(f"  Feature norms: {model_sparse_independent.feature_norms}")
print(f"  Superposition: {model_sparse_independent.superposition:.4f}")
print(f"  Cosine sim matrix diagonal-excluded max per feature:")
cos_sim = model_sparse_independent.cosine_similarity_matrix.abs()
cos_sim.fill_diagonal_(0)
print(f"  {cos_sim.max(dim=1).values}")

print("\n--- Dense + Locked (p_active~1.0, beta=1) ---")
print(f"  Feature norms: {model_dense_locked.feature_norms}")
print(f"  Superposition: {model_dense_locked.superposition:.4f}")

# %% Sub-grid: slice a rectangular region and examine it
# Slicing with ranges gives a smaller 2D grid (views, not copies --
# mutations propagate back to the parent grid).
corner = grid_2d[:3, :2]
print(f"\nSub-grid shape: {corner.shape}")
print(f"Sub-grid axes: {corner.description}")

# Verify it shares model objects with the parent
assert corner[0, 0] is grid_2d[0, 0], "Slices are views, not copies"
print("Confirmed: sub-grid shares model objects with parent grid.")

# %% [markdown]
# ## Summary
#
# - **`ModelGrid`** trains all models in a grid simultaneously via
#   `torch.vmap`, giving a large speedup over sequential training.
# - **`Axis`** defines each sweep dimension with a label and values.
# - **`create_model(params)`** is the factory that maps axis values
#   to a `ToyModel`. All models must share the same AE architecture.
# - **Indexing** follows NumPy conventions: integers collapse axes,
#   slices preserve them. Sub-grids are views into the parent.
# - **Metrics** like `superposition`, `feature_norms`, and
#   `cosine_similarity_matrix` are available on each `ToyModel` and
#   can be assembled into heatmaps for phase-diagram analysis.
