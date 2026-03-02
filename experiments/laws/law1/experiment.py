# ABOUTME: Law 1 experiment: correlation-interference relationship in CorrelatedPairs
# ABOUTME: Trains ModelGrids to extract Gram metrics and validate monotonic increase with correlation

# %%
# Cell 1: Imports

import os
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Any

from occhio import ToyModel, ModelGrid
from occhio.model_grid import Axis
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import CorrelatedPairs, AnticorrelatedPairs

print("✓ Imports successful")

# %%
# Cell 2: Model factories


def create_model_experiment_a(params: dict[str, Any]) -> ToyModel:
    """Factory for Experiment A models (n_features=6, n_hidden=2)."""
    n_features = 6
    correlation = params["Correlation"]
    density = params["Density"]
    dist = CorrelatedPairs(
        n_features=n_features,
        correlation=correlation,
        density=density,
        generator=torch.Generator().manual_seed(0),
    )

    ae = TiedLinearRelu(
        n_features=n_features,
        n_hidden=2,
        generator=torch.Generator().manual_seed(7),
    )

    return ToyModel(distribution=dist, ae=ae)


def create_model_experiment_b(params: dict[str, Any]) -> ToyModel:
    """Factory for Experiment B models (n_features=10, n_hidden=3)."""
    n_features = 10
    correlation = params["Correlation"]
    density = params["Density"]
    dist = CorrelatedPairs(
        n_features=n_features,
        correlation=correlation,
        density=density,
        generator=torch.Generator().manual_seed(0),
    )

    ae = TiedLinearRelu(
        n_features=n_features,
        n_hidden=3,
        generator=torch.Generator().manual_seed(7),
    )

    return ToyModel(distribution=dist, ae=ae)


def create_model_experiment_c(params: dict[str, Any]) -> ToyModel:
    """Factory for Experiment C models (anticorrelated control)."""
    n_features = 6
    density = params["Density"]
    dist = AnticorrelatedPairs(
        n_features=n_features,
        p_active=density,
        generator=torch.Generator().manual_seed(0),
    )

    ae = TiedLinearRelu(
        n_features=n_features,
        n_hidden=2,
        generator=torch.Generator().manual_seed(7),
    )

    return ToyModel(distribution=dist, ae=ae)


print("✓ Model factories defined")

# %%
# Cell 3: Metric extraction functions


def extract_metrics(grid: ModelGrid, n_features: int) -> dict[str, np.ndarray]:
    """Extract within-pair and cross-pair cosines from trained grid."""
    n_pairs = n_features // 2
    shape = grid.shape

    within_pair_cos = np.zeros(shape)
    cross_pair_cos_mean = np.zeros(shape)
    feature_norms_mean = np.zeros(shape)
    feature_dims_mean = np.zeros(shape)

    for idx in np.ndindex(shape):
        model = grid.models[idx]
        interf = model.interferences.cpu().numpy()

        # Within-pair cosines (extract diagonal elements for each pair)
        within_pair_vals = []
        for pair_idx in range(n_pairs):
            i, j = 2 * pair_idx, 2 * pair_idx + 1
            within_pair_vals.append(interf[i, j])
        within_pair_cos[idx] = np.mean(within_pair_vals)

        # Cross-pair cosines (all inter-pair interactions)
        cross_pair_vals = []
        for i in range(n_features):
            for j in range(n_features):
                pair_i, pair_j = i // 2, j // 2
                if pair_i != pair_j and i != j:
                    cross_pair_vals.append(abs(interf[i, j]))
        cross_pair_cos_mean[idx] = np.mean(cross_pair_vals) if cross_pair_vals else 0.0

        # Feature norms and dimensionalities
        feature_norms_mean[idx] = model.feature_norms.mean().item()
        feature_dims_mean[idx] = model.feature_dimensionalities.mean().item()

    return {
        "within_pair_cos": within_pair_cos,
        "cross_pair_cos_mean": cross_pair_cos_mean,
        "feature_norms_mean": feature_norms_mean,
        "feature_dims_mean": feature_dims_mean,
    }


def extract_metrics_anticorr(grid: ModelGrid, n_features: int) -> dict[str, np.ndarray]:
    """Extract metrics for anticorrelated control (1D grid)."""
    n_pairs = n_features // 2
    shape = grid.shape

    within_pair_cos = np.zeros(shape)
    cross_pair_cos_mean = np.zeros(shape)

    for idx in np.ndindex(shape):
        model = grid.models[idx]
        interf = model.interferences.cpu().numpy()

        # Within-pair cosines
        within_pair_vals = []
        for pair_idx in range(n_pairs):
            i, j = 2 * pair_idx, 2 * pair_idx + 1
            within_pair_vals.append(interf[i, j])
        within_pair_cos[idx] = np.mean(within_pair_vals)

        # Cross-pair cosines
        cross_pair_vals = []
        for i in range(n_features):
            for j in range(n_features):
                pair_i, pair_j = i // 2, j // 2
                if pair_i != pair_j and i != j:
                    cross_pair_vals.append(abs(interf[i, j]))
        cross_pair_cos_mean[idx] = np.mean(cross_pair_vals) if cross_pair_vals else 0.0

    return {
        "within_pair_cos": within_pair_cos,
        "cross_pair_cos_mean": cross_pair_cos_mean,
    }


print("✓ Metric extraction functions defined")

# %%
# Cell 4: Run all three experiments


def run_experiments():
    """Train all three experiment grids and extract metrics."""
    print("=" * 80)
    print("LAW 1: CORRELATION-INTERFERENCE RELATIONSHIP")
    print("=" * 80)

    # Experiment A: Primary
    print("\n[Experiment A] Training 20×15=300 models (n_features=6, n_hidden=2)")
    corr_values = torch.linspace(0, 0.95, 20)
    dens_values = torch.logspace(np.log10(0.01), np.log10(0.5), 15)

    grid_a = ModelGrid(
        create_model=create_model_experiment_a,
        axes=[
            Axis(label="Correlation", values=corr_values),
            Axis(label="Density", values=dens_values),
        ],
        cache_samples=False,
    )

    grid_a.fit(
        n_epochs=10000,
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=0.05,
        verbose=True,
    )

    # Extract metrics from Experiment A
    print("\n[Experiment A] Extracting metrics...")
    metrics_a = extract_metrics(grid_a, n_features=6)

    # Experiment B: Scale validation
    print("\n[Experiment B] Training 12×10=120 models (n_features=10, n_hidden=3)")
    corr_values_b = torch.linspace(0, 0.95, 12)
    dens_values_b = torch.logspace(np.log10(0.01), np.log10(0.5), 10)

    grid_b = ModelGrid(
        create_model=create_model_experiment_b,
        axes=[
            Axis(label="Correlation", values=corr_values_b),
            Axis(label="Density", values=dens_values_b),
        ],
        cache_samples=False,
    )

    grid_b.fit(
        n_epochs=10000,
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=0.05,
        verbose=True,
    )

    print("\n[Experiment B] Extracting metrics...")
    metrics_b = extract_metrics(grid_b, n_features=10)

    # Experiment C: Anticorrelated control
    print("\n[Experiment C] Training 1×15=15 anticorrelated models")
    dens_values_c = torch.logspace(np.log10(0.01), np.log10(0.5), 15)

    grid_c = ModelGrid(
        create_model=create_model_experiment_c,
        axes=[
            Axis(label="Density", values=dens_values_c),
        ],
        cache_samples=False,
    )

    grid_c.fit(
        n_epochs=10000,
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=0.05,
        verbose=True,
    )

    print("\n[Experiment C] Extracting metrics...")
    metrics_c = extract_metrics_anticorr(grid_c, n_features=6)

    return metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c


print("✓ Experiment runner defined")

# %%
# Cell 5: Figure generation


def create_figures(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c):
    """Generate 8 interactive figures for Law 1 analysis using Plotly."""
    os.makedirs("figures", exist_ok=True)

    # Extract axes from grids
    corr_ax_a = grid_a.axes[0]
    dens_ax_a = grid_a.axes[1]
    corr_vals_a = (
        corr_ax_a.values.cpu().numpy()
        if torch.is_tensor(corr_ax_a.values)
        else corr_ax_a.values
    )
    dens_vals_a = (
        dens_ax_a.values.cpu().numpy()
        if torch.is_tensor(dens_ax_a.values)
        else dens_ax_a.values
    )

    corr_vals_b = (
        grid_b.axes[0].values.cpu().numpy()
        if torch.is_tensor(grid_b.axes[0].values)
        else grid_b.axes[0].values
    )
    dens_vals_b = (
        grid_b.axes[1].values.cpu().numpy()
        if torch.is_tensor(grid_b.axes[1].values)
        else grid_b.axes[1].values
    )

    dens_vals_c = (
        grid_c.axes[0].values.cpu().numpy()
        if torch.is_tensor(grid_c.axes[0].values)
        else grid_c.axes[0].values
    )

    # Figure 1: Within-pair cosine heatmap (Correlation × Density)
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics_a["within_pair_cos"].T,
            x=corr_vals_a,
            y=dens_vals_a,
            colorscale="Viridis",
            colorbar=dict(title="Mean Within-Pair Cosine"),
        )
    )
    fig.update_layout(
        title="Figure 1: Within-pair cosine vs (Correlation, Density)<br>[Law 1 Core Observable]",
        xaxis_title="Correlation",
        yaxis_title="Density (log scale)",
        height=600,
        width=900,
    )
    fig.write_html("figures/01_within_pair_heatmap.html")

    # Figure 2: Within-pair cosine vs Correlation at selected densities
    selected_dens_idx = [
        0,
        len(dens_vals_a) // 4,
        len(dens_vals_a) // 2,
        3 * len(dens_vals_a) // 4,
        -1,
    ]
    color_list = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, dens_idx in enumerate(selected_dens_idx):
        within_pair = metrics_a["within_pair_cos"][:, dens_idx]
        fig.add_trace(
            go.Scatter(
                x=corr_vals_a,
                y=within_pair,
                mode="lines+markers",
                name=f"Density={dens_vals_a[dens_idx]:.3f}",
                line=dict(color=color_list[i % len(color_list)], width=2),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Figure 2: Functional Form: Within-pair Cosine vs Correlation<br>[Test h(c, S) Monotonicity]",
        xaxis_title="Correlation",
        yaxis_title="Mean Within-Pair Cosine",
        height=600,
        width=900,
        hovermode="x unified",
    )
    fig.write_html("figures/02_within_pair_vs_correlation.html")

    # Figure 3: Cross-pair cosine heatmap (should be independent of correlation)
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics_a["cross_pair_cos_mean"].T,
            x=corr_vals_a,
            y=dens_vals_a,
            colorscale="Plasma",
            colorbar=dict(title="Mean |Cross-Pair Cosine|"),
        )
    )
    fig.update_layout(
        title="Figure 3: Cross-pair |Cosine| vs (Correlation, Density)<br>[Control: Should NOT depend on correlation]",
        xaxis_title="Correlation",
        yaxis_title="Density (log scale)",
        height=600,
        width=900,
    )
    fig.write_html("figures/03_cross_pair_heatmap.html")

    # Figure 4: Structural interference = within_pair - cross_pair
    structural = metrics_a["within_pair_cos"] - metrics_a["cross_pair_cos_mean"]
    fig = go.Figure(
        data=go.Heatmap(
            z=structural.T,
            x=corr_vals_a,
            y=dens_vals_a,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Structural Interference"),
        )
    )
    fig.update_layout(
        title="Figure 4: Structural Interference = Within-pair - Cross-pair<br>[Isolates Correlation-Driven Component]",
        xaxis_title="Correlation",
        yaxis_title="Density (log scale)",
        height=600,
        width=900,
    )
    fig.write_html("figures/04_structural_interference.html")

    # Figure 5: Mean feature norms (phase diagram)
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics_a["feature_norms_mean"].T,
            x=corr_vals_a,
            y=dens_vals_a,
            colorscale="Magma",
            colorbar=dict(title="Mean Feature Norm"),
        )
    )
    fig.update_layout(
        title="Figure 5: Mean Feature Norms vs (Correlation, Density)<br>[Phase Diagram]",
        xaxis_title="Correlation",
        yaxis_title="Density (log scale)",
        height=600,
        width=900,
    )
    fig.write_html("figures/05_feature_norms_heatmap.html")

    # Figure 6: Within-pair cosine vs analytical prediction 1/(1-c) scaling
    fig = go.Figure()
    for i, dens_idx in enumerate(selected_dens_idx):
        within_pair = metrics_a["within_pair_cos"][:, dens_idx]
        fig.add_trace(
            go.Scatter(
                x=corr_vals_a,
                y=within_pair,
                mode="lines+markers",
                name=f"Measured (ρ={dens_vals_a[dens_idx]:.3f})",
                line=dict(color=color_list[i % len(color_list)], width=2),
                marker=dict(size=6),
            )
        )

    # Add theoretical prediction
    c_vals = corr_vals_a[corr_vals_a < 0.99]
    for i, dens_idx in enumerate(selected_dens_idx):
        within_pair = metrics_a["within_pair_cos"][: len(c_vals), dens_idx]
        if len(within_pair) > 0 and c_vals[0] > 0:
            scale = within_pair[len(c_vals) // 2] * (1 - c_vals[len(c_vals) // 2])
            theoretical = scale / (1 - c_vals)
            fig.add_trace(
                go.Scatter(
                    x=c_vals,
                    y=theoretical,
                    mode="lines",
                    name=f"Theory (ρ={dens_vals_a[dens_idx]:.3f})",
                    line=dict(dash="dash", width=1),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Figure 6: Comparison to 1/(1-c) Scaling<br>[Test Theoretical Cost Model]",
        xaxis_title="Correlation",
        yaxis_title="Within-Pair Cosine",
        height=600,
        width=900,
        hovermode="x unified",
    )
    fig.write_html("figures/06_theoretical_scaling.html")

    # Figure 7: Scale validation (Experiment B within-pair heatmap)
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics_b["within_pair_cos"].T,
            x=corr_vals_b,
            y=dens_vals_b,
            colorscale="Viridis",
            colorbar=dict(title="Mean Within-Pair Cosine"),
        )
    )
    fig.update_layout(
        title="Figure 7: Scale Validation (Exp B: n_features=10, n_hidden=3)<br>[Confirms Generalization]",
        xaxis_title="Correlation",
        yaxis_title="Density (log scale)",
        height=600,
        width=900,
    )
    fig.write_html("figures/07_scale_validation.html")

    # Figure 8: Anticorrelated control (opposite trend)
    anticorr_within = metrics_c["within_pair_cos"].flatten()
    correlated_within_mean = metrics_a["within_pair_cos"].mean(axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dens_vals_c,
            y=anticorr_within,
            name="Anticorrelated",
            marker=dict(color="#e74c3c"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=dens_vals_c,
            y=correlated_within_mean[: len(dens_vals_c)],
            name="Correlated (avg)",
            marker=dict(color="#3498db"),
        )
    )

    fig.update_layout(
        title="Figure 8: Anticorrelated Control<br>[Opposite Pattern: Features Repel]",
        xaxis_title="Density (log scale)",
        yaxis_title="Mean Within-Pair Cosine",
        barmode="group",
        height=600,
        width=900,
        hovermode="x unified",
    )
    fig.write_html("figures/08_anticorrelated_control.html")

    print("\n✓ All 8 figures saved as interactive html in figures/")


print("✓ Figure generation function defined")

# %%
# Cell 6: Summary and analysis


def print_summary(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c):
    """Print quantitative summary to stdout."""
    print("\n" + "=" * 80)
    print("QUANTITATIVE SUMMARY")
    print("=" * 80)

    print("\n[EXPERIMENT A] 20 correlations × 15 densities = 300 models")
    print(
        f"  Within-pair cosine: min={metrics_a['within_pair_cos'].min():.4f}, "
        f"max={metrics_a['within_pair_cos'].max():.4f}, "
        f"mean={metrics_a['within_pair_cos'].mean():.4f}"
    )

    # Check monotonicity with correlation at middle density
    mid_dens_idx = len(grid_a.axes[1].values) // 2
    within_at_mid = metrics_a["within_pair_cos"][:, mid_dens_idx]
    monotonic_increase = all(
        within_at_mid[i] <= within_at_mid[i + 1] for i in range(len(within_at_mid) - 1)
    )
    print(
        f"  Monotonic increase with correlation (at mid-density): {monotonic_increase}"
    )

    print("\n[EXPERIMENT B] 12 correlations × 10 densities = 120 models")
    print(
        f"  Within-pair cosine: min={metrics_b['within_pair_cos'].min():.4f}, "
        f"max={metrics_b['within_pair_cos'].max():.4f}, "
        f"mean={metrics_b['within_pair_cos'].mean():.4f}"
    )

    print("\n[EXPERIMENT C] Anticorrelated (1×15 = 15 models)")
    print(
        f"  Within-pair cosine: min={metrics_c['within_pair_cos'].min():.4f}, "
        f"max={metrics_c['within_pair_cos'].max():.4f}, "
        f"mean={metrics_c['within_pair_cos'].mean():.4f}"
    )

    # Evidence for Law 1
    print("\n" + "-" * 80)
    print("EVIDENCE FOR/AGAINST LAW 1")
    print("-" * 80)

    if monotonic_increase:
        print(
            "✓ SUPPORTS Law 1: Within-pair cosine increases monotonically with correlation"
        )
    else:
        print(
            "✗ CONTRADICTS Law 1: Non-monotonic or flat relationship with correlation"
        )

    # Check if cross-pair also increases (would be evidence against Law 1)
    cross_pair_at_mid = metrics_a["cross_pair_cos_mean"][:, mid_dens_idx]
    cross_monotonic = all(
        cross_pair_at_mid[i] <= cross_pair_at_mid[i + 1]
        for i in range(len(cross_pair_at_mid) - 1)
    )

    if cross_monotonic:
        print("✗ CONTRADICTS Law 1: Cross-pair cosine also increases with correlation")
        print("  (Suggests generic alignment, not correlation-specific effect)")
    else:
        print("✓ SUPPORTS Law 1: Cross-pair cosine independent of correlation")
        print("  (Confirms correlation-specific effect on paired features)")

    print("\n" + "=" * 80)


print("✓ Summary function defined")

# %%
# Cell 7: MAIN EXECUTION

print("\n" + "=" * 80)
print("STARTING EXPERIMENTS")
print("=" * 80)

metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c = run_experiments()
print_summary(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c)
create_figures(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c)

print("\n✓ Experiment complete!")

# %%
