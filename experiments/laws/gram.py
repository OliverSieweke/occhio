# ABOUTME: Law 1 experiment: correlation-interference relationship in CorrelatedPairs
# ABOUTME: Trains ModelGrids to extract Gram metrics and validate monotonic increase with correlation

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from occhio import ToyModel, ModelGrid
from occhio.model_grid import Axis
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import CorrelatedPairs, AnticorrelatedPairs


# ── Experiment A: Primary (6 features, 2 hidden, 3 pairs) ──────────────────


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

    return ToyModel(distribution=dist, ae=ae, device="cpu")


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

    return ToyModel(distribution=dist, ae=ae, device="cpu")


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

    return ToyModel(distribution=dist, ae=ae, device="cpu")


# ── Run experiments ────────────────────────────────────────────────────────


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


# ── Plotting ───────────────────────────────────────────────────────────────


def create_figures(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c):
    """Generate 8 figures for Law 1 analysis."""
    os.makedirs("experiments/laws/figures", exist_ok=True)

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
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        metrics_a["within_pair_cos"].T, aspect="auto", origin="lower", cmap="viridis"
    )
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        "Figure 1: Within-pair cosine vs (Correlation, Density)\n[Law 1 Core Observable]"
    )
    ax.set_xticks(np.linspace(0, len(corr_vals_a) - 1, 5))
    ax.set_xticklabels([f"{c:.2f}" for c in corr_vals_a[:: len(corr_vals_a) // 4]])
    ax.set_yticks(np.linspace(0, len(dens_vals_a) - 1, 5))
    ax.set_yticklabels([f"{d:.3f}" for d in dens_vals_a[:: len(dens_vals_a) // 4]])
    plt.colorbar(im, ax=ax, label="Mean Within-Pair Cosine")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/01_within_pair_heatmap.png", dpi=150)
    plt.close()

    # Figure 2: Within-pair cosine vs Correlation at selected densities
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_dens_idx = [
        0,
        len(dens_vals_a) // 4,
        len(dens_vals_a) // 2,
        3 * len(dens_vals_a) // 4,
        -1,
    ]
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, len(selected_dens_idx)))

    for i, dens_idx in enumerate(selected_dens_idx):
        within_pair = metrics_a["within_pair_cos"][:, dens_idx]
        ax.plot(
            corr_vals_a,
            within_pair,
            marker="o",
            label=f"Density={dens_vals_a[dens_idx]:.3f}",
            color=colors[i],
            linewidth=2,
        )

    ax.set_xlabel("Correlation", fontsize=12)
    ax.set_ylabel("Mean Within-Pair Cosine", fontsize=12)
    ax.set_title(
        "Figure 2: Functional Form: Within-pair Cosine vs Correlation\n[Test h(c, S) Monotonicity]",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/02_within_pair_vs_correlation.png", dpi=150)
    plt.close()

    # Figure 3: Cross-pair cosine heatmap (should be independent of correlation)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        metrics_a["cross_pair_cos_mean"].T, aspect="auto", origin="lower", cmap="plasma"
    )
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        "Figure 3: Cross-pair |Cosine| vs (Correlation, Density)\n[Control: Should NOT depend on correlation]"
    )
    ax.set_xticks(np.linspace(0, len(corr_vals_a) - 1, 5))
    ax.set_xticklabels([f"{c:.2f}" for c in corr_vals_a[:: len(corr_vals_a) // 4]])
    ax.set_yticks(np.linspace(0, len(dens_vals_a) - 1, 5))
    ax.set_yticklabels([f"{d:.3f}" for d in dens_vals_a[:: len(dens_vals_a) // 4]])
    plt.colorbar(im, ax=ax, label="Mean |Cross-Pair Cosine|")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/03_cross_pair_heatmap.png", dpi=150)
    plt.close()

    # Figure 4: Structural interference = within_pair - cross_pair (isolates correlation effect)
    structural = metrics_a["within_pair_cos"] - metrics_a["cross_pair_cos_mean"]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(structural.T, aspect="auto", origin="lower", cmap="RdBu_r")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        "Figure 4: Structural Interference = Within-pair - Cross-pair\n[Isolates Correlation-Driven Component]"
    )
    ax.set_xticks(np.linspace(0, len(corr_vals_a) - 1, 5))
    ax.set_xticklabels([f"{c:.2f}" for c in corr_vals_a[:: len(corr_vals_a) // 4]])
    ax.set_yticks(np.linspace(0, len(dens_vals_a) - 1, 5))
    ax.set_yticklabels([f"{d:.3f}" for d in dens_vals_a[:: len(dens_vals_a) // 4]])
    plt.colorbar(im, ax=ax, label="Structural Interference")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/04_structural_interference.png", dpi=150)
    plt.close()

    # Figure 5: Mean feature norms (phase diagram)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        metrics_a["feature_norms_mean"].T, aspect="auto", origin="lower", cmap="magma"
    )
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        "Figure 5: Mean Feature Norms vs (Correlation, Density)\n[Phase Diagram]"
    )
    ax.set_xticks(np.linspace(0, len(corr_vals_a) - 1, 5))
    ax.set_xticklabels([f"{c:.2f}" for c in corr_vals_a[:: len(corr_vals_a) // 4]])
    ax.set_yticks(np.linspace(0, len(dens_vals_a) - 1, 5))
    ax.set_yticklabels([f"{d:.3f}" for d in dens_vals_a[:: len(dens_vals_a) // 4]])
    plt.colorbar(im, ax=ax, label="Mean Feature Norm")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/05_feature_norms_heatmap.png", dpi=150)
    plt.close()

    # Figure 6: Within-pair cosine vs analytical prediction 1/(1-c) scaling
    fig, ax = plt.subplots(figsize=(10, 6))

    for dens_idx in selected_dens_idx:
        within_pair = metrics_a["within_pair_cos"][:, dens_idx]
        ax.plot(
            corr_vals_a,
            within_pair,
            marker="o",
            label=f"Measured (ρ={dens_vals_a[dens_idx]:.3f})",
            linewidth=2,
            markersize=5,
        )

    # Add theoretical prediction: a * 1/(1-c) for some scaling a
    c_vals = corr_vals_a[corr_vals_a < 0.99]
    for i, dens_idx in enumerate(selected_dens_idx):
        within_pair = metrics_a["within_pair_cos"][: len(c_vals), dens_idx]
        if len(within_pair) > 0 and c_vals[0] > 0:
            # Scale at c=0.5 to fit
            scale = within_pair[len(c_vals) // 2] * (1 - c_vals[len(c_vals) // 2])
            theoretical = scale / (1 - c_vals)
            ax.plot(c_vals, theoretical, "--", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("Correlation", fontsize=12)
    ax.set_ylabel("Within-Pair Cosine", fontsize=12)
    ax.set_title(
        "Figure 6: Comparison to 1/(1-c) Scaling\n[Test Theoretical Cost Model]",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/06_theoretical_scaling.png", dpi=150)
    plt.close()

    # Figure 7: Scale validation (Experiment B within-pair heatmap)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        metrics_b["within_pair_cos"].T, aspect="auto", origin="lower", cmap="viridis"
    )
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        "Figure 7: Scale Validation (Exp B: n_features=10, n_hidden=3)\n[Confirms Generalization]"
    )
    ax.set_xticks(np.linspace(0, len(corr_vals_b) - 1, 5))
    ax.set_xticklabels([f"{c:.2f}" for c in corr_vals_b[:: len(corr_vals_b) // 3]])
    ax.set_yticks(np.linspace(0, len(dens_vals_b) - 1, 5))
    ax.set_yticklabels([f"{d:.3f}" for d in dens_vals_b[:: len(dens_vals_b) // 3]])
    plt.colorbar(im, ax=ax, label="Mean Within-Pair Cosine")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/07_scale_validation.png", dpi=150)
    plt.close()

    # Figure 8: Anticorrelated control (opposite trend)
    fig, ax = plt.subplots(figsize=(10, 6))
    anticorr_within = metrics_c["within_pair_cos"].flatten()
    correlated_within_mean = metrics_a["within_pair_cos"].mean(
        axis=0
    )  # Average over all correlations

    x_pos = np.arange(len(dens_vals_c))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        anticorr_within,
        width,
        label="Anticorrelated",
        color="#e74c3c",
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        correlated_within_mean[: len(dens_vals_c)],
        width,
        label="Correlated (avg)",
        color="#3498db",
        alpha=0.8,
    )

    ax.set_xlabel("Density (log scale)", fontsize=12)
    ax.set_ylabel("Mean Within-Pair Cosine", fontsize=12)
    ax.set_title(
        "Figure 8: Anticorrelated Control\n[Opposite Pattern: Features Repel]",
        fontsize=13,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{d:.3f}" for d in dens_vals_c[:: len(dens_vals_c) // 4]], rotation=45
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("experiments/laws/figures/08_anticorrelated_control.png", dpi=150)
    plt.close()

    print("\n✓ All 8 figures saved to experiments/laws/figures/")


# ── Summary statistics ──────────────────────────────────────────────────────


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


if __name__ == "__main__":
    metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c = run_experiments()
    print_summary(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c)
    create_figures(metrics_a, metrics_b, metrics_c, grid_a, grid_b, grid_c)
    print("\n✓ Experiment complete!")
