# ABOUTME: Law 1 experiment: correlation drives within-pair interference
# ABOUTME: Trains ModelGrids over (correlation, density, seed) and generates 8 diagnostic figures
# %%
import os
import torch
from torch.distributions import Beta
from torch import Tensor
import numpy as np
import plotly.graph_objects as go
from typing import Any

from occhio.toy_model import ToyModel
from occhio.model_grid import ModelGrid, Axis
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import (
    Distribution,
    CorrelatedPairs,
    AnticorrelatedPairs,
    SparseUniform,
    DistributionStack,
)
from datetime import datetime

# %%
# ── Constants ────────────────────────────────────────────────────────────────
DEVICE = "mps"
N_HIDDEN = 3
N_FEATURES = 9
N_EPOCHS = 20_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
DIST_SEED = 7

CORR_VALUES = torch.tensor(
    [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 1.0]
)
DENS_VALUES: Tensor = torch.cat(
    [
        torch.logspace(np.log10(0.01), np.log10(0.5), 12),
        torch.tensor([0.6, 0.7, 0.8, 0.9]),
    ]
)
RANDOM_SEEDS = torch.arange(start=0, end=80, step=8)
FIGURES_DIR = "figures"


# %%
# ── Model factories ──────────────────────────────────────────────────────────
def create_correlated(params: dict[str, Any]) -> ToyModel:
    """Features 0-1 are CorrelatedPairs, rest SparseUniform."""
    c = params["Correlation"]
    d = params["Density"]
    seed = int(params["Seed"])
    dists: list[Distribution] = [
        CorrelatedPairs(
            2,
            correlation=c,
            density=d,
            generator=torch.Generator(device=DEVICE).manual_seed(DIST_SEED),
        ),
    ]
    if N_FEATURES > 2:
        dists.append(
            SparseUniform(
                N_FEATURES - 2,
                p_active=d,
                generator=torch.Generator(device=DEVICE).manual_seed(DIST_SEED),
            )
        )
    return ToyModel(
        distribution=DistributionStack(dists),
        ae=TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=N_HIDDEN,
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
        ),
        importances=0.996 ** torch.arange(N_FEATURES),
    )


def create_anticorrelated(params: dict[str, Any]) -> ToyModel:
    """Features 0-1 are AnticorrelatedPairs, rest SparseUniform."""
    d = params["Density"]
    seed = int(params["Seed"])
    # p_active=2*d so each individual feature fires with prob d
    dists: list[Distribution] = [
        AnticorrelatedPairs(
            2,
            p_active=2 * d,
            generator=torch.Generator(device=DEVICE).manual_seed(DIST_SEED),
        ),
    ]
    if N_FEATURES > 2:
        dists.append(
            SparseUniform(
                N_FEATURES - 2,
                p_active=d,
                generator=torch.Generator(device=DEVICE).manual_seed(DIST_SEED),
            )
        )

    return ToyModel(
        distribution=DistributionStack(dists),
        ae=TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=N_HIDDEN,
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
        ),
        importances=0.996 ** torch.arange(N_FEATURES),
    )


def create_baseline(params: dict[str, Any]) -> ToyModel:
    """All features are independent SparseUniform."""
    d = params["Density"]
    seed = int(params["Seed"])
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES,
            p_active=d,
            generator=torch.Generator(device=DEVICE).manual_seed(DIST_SEED),
        ),
        ae=TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=N_HIDDEN,
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
        ),
        importances=0.996 ** torch.arange(N_FEATURES),
    )


Model1 = create_correlated(params={"Correlation": 0.0, "Density": 0.1, "Seed": 0})
print(Model1.distribution.sample(10))
print(Model1.distribution.device)
print(Model1.distribution.generator)
print(vars(Model1.distribution))


# %%
# ── Metric extraction ────────────────────────────────────────────────────────
def extract_metrics(grid: ModelGrid) -> dict[str, np.ndarray]:
    """Extract within-pair cos(W0,W1), cross-pair mean |cos|, and feature norms."""
    shape = grid.shape
    within_pair_cos = np.zeros(shape)
    cross_pair_cos_mean = np.zeros(shape)
    feature_norms_mean = np.zeros(shape)

    for idx in np.ndindex(shape):
        model = grid.models[idx]
        interf = model.interferences.cpu().numpy()

        within_pair_cos[idx] = interf[0, 1]

        cross_vals = []
        for i in range(N_FEATURES):
            for j in range(i + 1, N_FEATURES):
                if i == 0 and j == 1:
                    continue
                cross_vals.append(abs(interf[i, j]))
        cross_pair_cos_mean[idx] = np.mean(cross_vals) if cross_vals else 0.0
        feature_norms_mean[idx] = model.feature_norms.mean().item()

    return {
        "within_pair_cos": within_pair_cos,
        "cross_pair_cos_mean": cross_pair_cos_mean,
        "feature_norms_mean": feature_norms_mean,
    }


def extract_within_pair_cos(grid: ModelGrid) -> np.ndarray:
    """Extract just within-pair cos(W0,W1)."""
    shape = grid.shape
    result = np.zeros(shape)
    for idx in np.ndindex(shape):
        model = grid.models[idx]
        interf = model.interferences.cpu().numpy()
        result[idx] = interf[0, 1]
    return result


# %%
# ── Training ─────────────────────────────────────────────────────────────────
def train_grid(grid: ModelGrid):
    """Train a ModelGrid with standard hyperparameters."""
    grid.fit(
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        verbose=False,
        sample_every=100,
    )


def run_correlated_grid() -> tuple[ModelGrid, dict[str, np.ndarray]]:
    """Train correlated grid: axes = [Correlation, Density, Seed]."""
    n_models = len(CORR_VALUES) * len(DENS_VALUES) * len(RANDOM_SEEDS)
    print(
        f"\n[Correlated] {len(CORR_VALUES)}x{len(DENS_VALUES)}x{len(RANDOM_SEEDS)} = {n_models} models"
    )
    grid = ModelGrid(
        create_model=create_correlated,
        axes=[
            Axis(label="Correlation", values=CORR_VALUES),
            Axis(label="Density", values=DENS_VALUES),
            Axis(label="Seed", values=RANDOM_SEEDS),
        ],
        broadcast_samples=True,
    )
    train_grid(grid)
    return grid, extract_metrics(grid)


def run_anticorrelated_grid() -> tuple[ModelGrid, np.ndarray]:
    """Train anticorrelated grid: axes = [Density, Seed]."""
    n_models = len(DENS_VALUES) * len(RANDOM_SEEDS)
    print(
        f"\n[Anticorrelated] {len(DENS_VALUES)}x{len(RANDOM_SEEDS)} = {n_models} models"
    )
    grid = ModelGrid(
        create_model=create_anticorrelated,
        axes=[
            Axis(label="Density", values=DENS_VALUES),
            Axis(label="Seed", values=RANDOM_SEEDS),
        ],
        broadcast_samples=True,
    )
    train_grid(grid)
    return grid, extract_within_pair_cos(grid)


def run_baseline_grid() -> tuple[ModelGrid, np.ndarray]:
    """Train baseline grid: axes = [Density, Seed]."""
    n_models = len(DENS_VALUES) * len(RANDOM_SEEDS)
    print(f"\n[Baseline] {len(DENS_VALUES)}x{len(RANDOM_SEEDS)} = {n_models} models")
    grid = ModelGrid(
        create_model=create_baseline,
        axes=[
            Axis(label="Density", values=DENS_VALUES),
            Axis(label="Seed", values=RANDOM_SEEDS),
        ],
        broadcast_samples=True,
    )
    train_grid(grid)
    return grid, extract_within_pair_cos(grid)


# %%
# ── Figure helpers ───────────────────────────────────────────────────────────
def corr_vals_np():
    return CORR_VALUES.cpu().numpy()


def dens_vals_np():
    return DENS_VALUES.cpu().numpy()


def save_fig(fig: go.Figure, filename: str):
    fig.write_html(os.path.join(FIGURES_DIR, filename))
    print(f"  Saved {filename}")


def _density_slider_steps(traces_per_density: int) -> list[dict]:
    """Build Plotly slider steps that toggle trace visibility by density index."""
    dv = dens_vals_np()
    n_d = len(dv)
    total = n_d * traces_per_density
    steps = []
    for d_idx in range(n_d):
        visible = [False] * total
        for i in range(traces_per_density):
            visible[d_idx * traces_per_density + i] = True
        steps.append(
            dict(
                method="update",
                args=[{"visible": visible}],
                label=f"{dv[d_idx]:.3f}",
            )
        )
    return steps


def _add_seed_traces(
    fig: go.Figure,
    cv: np.ndarray,
    data_2d: np.ndarray,
    d_idx: int,
    color: str = "black",
    opacity: float = 0.2,
    dash: str | None = None,
    legend_name: str | None = None,
    legendgroup: str | None = None,
):
    """Add per-seed traces (thin, low opacity) for a (n_corr, n_seeds) array.

    If legend_name is provided, the first seed trace gets a legend entry.
    """
    n_seeds = data_2d.shape[1]
    line_kw = dict(color=color, width=1)
    if dash:
        line_kw["dash"] = dash
    for s in range(n_seeds):
        fig.add_trace(
            go.Scatter(
                x=cv,
                y=data_2d[:, s],
                mode="lines",
                line=line_kw,
                opacity=opacity,
                name=legend_name if (s == 0 and legend_name) else "",
                showlegend=(s == 0 and legend_name is not None),
                legendgroup=legendgroup,
                visible=(d_idx == 0),
            )
        )


def _add_mean_trace(
    fig: go.Figure,
    cv: np.ndarray,
    data_2d: np.ndarray,
    d_idx: int,
    color: str = "red",
    width: float = 2.5,
    dash: str | None = None,
    name: str = "mean",
    legendgroup: str | None = None,
):
    """Add a mean-across-seeds trace for a (n_corr, n_seeds) array."""
    line_kw = dict(color=color, width=width)
    if dash:
        line_kw["dash"] = dash
    fig.add_trace(
        go.Scatter(
            x=cv,
            y=data_2d.mean(axis=1),
            mode="lines",
            line=line_kw,
            name=name,
            showlegend=True,
            legendgroup=legendgroup,
            visible=(d_idx == 0),
        )
    )


# %%
# ── Figures ──────────────────────────────────────────────────────────────────

# Heatmaps average across seed axis (axis=-1).
# corr_metrics shapes: (n_corr, n_dens, n_seeds)
# anticorr/baseline shapes: (n_dens, n_seeds)


def _heatmap_layout(fig: go.Figure, title: str, xaxis_title: str = "Correlation"):
    """Apply shared heatmap layout: explicit density ticks, no log scale."""
    dv = dens_vals_np()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Density",
        yaxis=dict(
            tickmode="array",
            tickvals=dv,
            ticktext=[f"{v:.2f}" for v in dv],
        ),
        height=700,
        width=900,
    )


def fig1_within_pair_heatmap(corr_metrics: dict[str, np.ndarray]):
    """Within-pair cosine heatmap, averaged across seeds.

    Expected: higher correlation → higher cos(W₀,W₁). At low density, features
    are easily represented so cosine stays low; at high density superposition is
    forced and correlation amplifies within-pair alignment.
    """
    avg = corr_metrics["within_pair_cos"].mean(axis=2)  # (n_corr, n_dens)
    fig = go.Figure(
        data=go.Heatmap(
            z=avg.T,
            x=corr_vals_np(),
            y=dens_vals_np(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="cos(W₀, W₁)", dtick=0.25),
        )
    )
    _heatmap_layout(
        fig, f"Within-pair cos(W₀,W₁) [n={N_FEATURES}, avg {len(RANDOM_SEEDS)} seeds]"
    )
    save_fig(fig, "01_within_pair_heatmap.html")


def fig2_within_pair_vs_correlation(corr_metrics: dict[str, np.ndarray]):
    """Within-pair cos vs correlation. Black=individual seeds, red=mean. Density slider.

    Tests Law 1 directly: cos(W₀,W₁) should increase monotonically with
    correlation at each density. Spread across seeds shows sensitivity to
    initialization — wide spread means the effect is fragile at that density.
    """
    cv = corr_vals_np()
    n_seeds = len(RANDOM_SEEDS)
    traces_per_density = n_seeds + 1

    fig = go.Figure()
    for d_idx in range(len(DENS_VALUES)):
        data = corr_metrics["within_pair_cos"][:, d_idx, :]  # (n_corr, n_seeds)
        _add_seed_traces(fig, cv, data, d_idx, legend_name="individual seeds")
        _add_mean_trace(fig, cv, data, d_idx)

    fig.update_layout(
        title=f"Within-pair cos(W₀,W₁) vs Correlation [n={N_FEATURES}]",
        xaxis_title="Correlation",
        yaxis_title="cos(W₀, W₁)",
        yaxis_range=[-1, 1],
        height=600,
        width=900,
        hovermode="x unified",
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Density: "},
                steps=_density_slider_steps(traces_per_density),
                pad={"t": 50},
            )
        ],
    )
    save_fig(fig, "02_within_pair_vs_correlation.html")


def fig3_cross_pair_heatmap(corr_metrics: dict[str, np.ndarray]):
    """Cross-pair mean |cos| heatmap, averaged across seeds.

    Control: cross-pair cosines should NOT depend on correlation. They measure
    geometric crowding due to density/n_features, not correlation structure.
    If this heatmap shows a correlation gradient, the effect in fig1 is not
    specific to the paired features.
    """
    avg = corr_metrics["cross_pair_cos_mean"].mean(axis=2)
    fig = go.Figure(
        data=go.Heatmap(
            z=avg.T,
            x=corr_vals_np(),
            y=dens_vals_np(),
            colorscale="Viridis",
            colorbar=dict(title="Mean |cos| (cross-pair)"),
        )
    )
    _heatmap_layout(
        fig, f"Cross-pair mean |cos| [n={N_FEATURES}, avg {len(RANDOM_SEEDS)} seeds]"
    )
    save_fig(fig, "03_cross_pair_heatmap.html")


def fig4_structural_interference(corr_metrics: dict[str, np.ndarray]):
    """Structural interference = within-pair − cross-pair, averaged across seeds.

    Isolates the correlation-driven component of interference by subtracting
    the generic crowding baseline (cross-pair). Positive values mean correlated
    features align MORE than uncorrelated features at the same density.
    """
    within_avg = corr_metrics["within_pair_cos"].mean(axis=2)
    cross_avg = corr_metrics["cross_pair_cos_mean"].mean(axis=2)
    structural = within_avg - cross_avg
    fig = go.Figure(
        data=go.Heatmap(
            z=structural.T,
            x=corr_vals_np(),
            y=dens_vals_np(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="within − cross", dtick=0.25),
        )
    )
    _heatmap_layout(
        fig, f"Structural Interference [n={N_FEATURES}, avg {len(RANDOM_SEEDS)} seeds]"
    )
    save_fig(fig, "04_structural_interference.html")


def fig5_feature_norms_heatmap(corr_metrics: dict[str, np.ndarray]):
    """Feature norms phase diagram, averaged across seeds.

    Norms near 1 = feature is fully represented; near 0 = feature dropped.
    Correlation should shift the phase boundary: correlated features may be
    represented or dropped together, changing the effective capacity.
    """
    avg = corr_metrics["feature_norms_mean"].mean(axis=2)
    fig = go.Figure(
        data=go.Heatmap(
            z=avg.T,
            x=corr_vals_np(),
            y=dens_vals_np(),
            colorscale="Magma",
            colorbar=dict(title="Mean Feature Norm"),
        )
    )
    _heatmap_layout(
        fig, f"Mean Feature Norms [n={N_FEATURES}, avg {len(RANDOM_SEEDS)} seeds]"
    )
    save_fig(fig, "05_feature_norms_heatmap.html")


def fig6_scaling_comparison(corr_metrics: dict[str, np.ndarray]):
    """1/(1−c) scaling: black=seeds, red=mean, dashed blue=theory. Density slider.

    Theoretical prior: if the interference cost of representing correlated
    features grows as 1/(1−c), then cos(W₀,W₁) ~ A/(1−c) for some constant A
    that depends on density and architecture. We calibrate A at the midpoint
    of the correlation range: A = cos_mid * (1 − c_mid). The dashed line shows
    this prediction. Deviations indicate a different functional form.
    """
    cv = corr_vals_np()
    n_seeds = len(RANDOM_SEEDS)
    traces_per_density = n_seeds + 2

    fig = go.Figure()
    for d_idx in range(len(DENS_VALUES)):
        data = corr_metrics["within_pair_cos"][:, d_idx, :]
        _add_seed_traces(fig, cv, data, d_idx, legend_name="individual seeds")
        _add_mean_trace(fig, cv, data, d_idx)

        # Theoretical: cos ~ A/(1-c), calibrated at midpoint
        mean_vals = data.mean(axis=1)
        mid = len(cv) // 2
        if cv[mid] < 0.99 and mean_vals[mid] != 0:
            A = mean_vals[mid] * (1 - cv[mid])
            theoretical = A / (1 - cv)
        else:
            theoretical = np.zeros_like(cv)
        fig.add_trace(
            go.Scatter(
                x=cv,
                y=theoretical,
                mode="lines",
                line=dict(dash="dash", color="blue", width=1.5),
                name="A/(1−c) theory",
                showlegend=True,
                visible=(d_idx == 0),
            )
        )

    fig.update_layout(
        title=f"1/(1−c) Scaling Comparison [n={N_FEATURES}]",
        xaxis_title="Correlation",
        yaxis_title="cos(W₀, W₁)",
        yaxis_range=[-1, 1],
        height=600,
        width=900,
        hovermode="x unified",
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Density: "},
                steps=_density_slider_steps(traces_per_density),
                pad={"t": 50},
            )
        ],
    )
    save_fig(fig, "06_theoretical_scaling.html")


def fig7_scale_validation(corr_metrics: dict[str, np.ndarray]):
    """Seed variation view: same data as fig2 but titled for seed analysis.

    Use this to assess how robust the correlation→interference relationship is
    to random initialization. Tight seed bundles = robust effect.
    Wide spread or crossing seeds = noisy / not converged.
    """
    cv = corr_vals_np()
    n_seeds = len(RANDOM_SEEDS)
    traces_per_density = n_seeds + 1

    fig = go.Figure()
    for d_idx in range(len(DENS_VALUES)):
        data = corr_metrics["within_pair_cos"][:, d_idx, :]
        _add_seed_traces(fig, cv, data, d_idx, legend_name="individual seeds")
        _add_mean_trace(fig, cv, data, d_idx)

    fig.update_layout(
        title=f"Seed Variation: cos(W₀,W₁) vs Correlation [n={N_FEATURES}]",
        xaxis_title="Correlation",
        yaxis_title="cos(W₀, W₁)",
        yaxis_range=[-1, 1],
        height=600,
        width=900,
        hovermode="x unified",
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Density: "},
                steps=_density_slider_steps(traces_per_density),
                pad={"t": 50},
            )
        ],
    )
    save_fig(fig, "07_scale_validation.html")


def fig8_anticorrelated_baseline(
    corr_metrics: dict[str, np.ndarray],
    anticorr_cos: np.ndarray,
    baseline_cos: np.ndarray,
):
    """Correlated vs anticorrelated vs baseline. Density slider.

    Correlated features (blue, solid) should show increasing cos with correlation.
    Anticorrelated features (red, dashed) should show near-zero or negative cos —
    mutual exclusivity pushes embedding vectors apart. Baseline (green, dotted) has
    no pair structure so cos(W₀,W₁) should be near zero regardless of the
    x-axis position (it's a horizontal reference).
    """
    cv = corr_vals_np()
    n_seeds = len(RANDOM_SEEDS)
    # Per density: 3 types × (n_seeds + 1) traces
    traces_per_density = 3 * (n_seeds + 1)

    fig = go.Figure()
    for d_idx in range(len(DENS_VALUES)):
        # Correlated — varies with correlation
        corr_data = corr_metrics["within_pair_cos"][:, d_idx, :]  # (n_corr, n_seeds)
        _add_seed_traces(
            fig,
            cv,
            corr_data,
            d_idx,
            color="lightskyblue",
            opacity=0.3,
            legend_name="correlated (seeds)",
            legendgroup="corr",
        )
        _add_mean_trace(
            fig,
            cv,
            corr_data,
            d_idx,
            color="blue",
            name="correlated (mean)",
            legendgroup="corr",
        )

        # Anticorrelated — constant across correlation
        anti_vals = anticorr_cos[d_idx, :]  # (n_seeds,)
        anti_2d = np.tile(anti_vals, (len(cv), 1))  # (n_corr, n_seeds)
        _add_seed_traces(
            fig,
            cv,
            anti_2d,
            d_idx,
            color="lightsalmon",
            opacity=0.3,
            dash="dash",
            legend_name="anticorr (seeds)",
            legendgroup="anti",
        )
        _add_mean_trace(
            fig,
            cv,
            anti_2d,
            d_idx,
            color="red",
            dash="dash",
            name="anticorr (mean)",
            legendgroup="anti",
        )

        # Baseline — constant across correlation
        base_vals = baseline_cos[d_idx, :]  # (n_seeds,)
        base_2d = np.tile(base_vals, (len(cv), 1))  # (n_corr, n_seeds)
        _add_seed_traces(
            fig,
            cv,
            base_2d,
            d_idx,
            color="lightgreen",
            opacity=0.3,
            dash="dot",
            legend_name="baseline (seeds)",
            legendgroup="base",
        )
        _add_mean_trace(
            fig,
            cv,
            base_2d,
            d_idx,
            color="green",
            dash="dot",
            name="baseline (mean)",
            legendgroup="base",
        )

    fig.update_layout(
        title=f"Correlated vs Anticorrelated vs Baseline [n={N_FEATURES}]",
        xaxis_title="Correlation",
        yaxis_title="cos(W₀, W₁)",
        yaxis_range=[-1, 1],
        height=600,
        width=900,
        hovermode="x unified",
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Density: "},
                steps=_density_slider_steps(traces_per_density),
                pad={"t": 50},
            )
        ],
    )
    save_fig(fig, "08_anticorrelated_control.html")


# %%
# ── Train and save all grids ───────────────────────────────────────────────────
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("LAW 1: CORRELATION -> INTERFERENCE")
print("=" * 80)

corr_grid, corr_metrics = run_correlated_grid()
anticorr_grid, anticorr_cos = run_anticorrelated_grid()
baseline_grid, baseline_cos = run_baseline_grid()

corr_grid.save_models(f"corr_grid_{datetime.now().strftime('%Y%m%d_%H%M')}")
anticorr_grid.save_models(f"anticorr_grid_{datetime.now().strftime('%Y%m%d_%H%M')}")
baseline_grid.save_models(f"baseline_grid_{datetime.now().strftime('%Y%m%d_%H%M')}")

# %%
# ── Generate all figures ─────────────────────────────────────────────────────
print("\nGenerating figures...")

fig1_within_pair_heatmap(corr_metrics)
fig2_within_pair_vs_correlation(corr_metrics)
fig3_cross_pair_heatmap(corr_metrics)
fig4_structural_interference(corr_metrics)
fig5_feature_norms_heatmap(corr_metrics)
fig6_scaling_comparison(corr_metrics)
fig7_scale_validation(corr_metrics)
fig8_anticorrelated_baseline(corr_metrics, anticorr_cos, baseline_cos)

# %%
# ── Summary ──────────────────────────────────────────────────────────────────
within_avg = corr_metrics["within_pair_cos"].mean(axis=2)
print(f"\n[n={N_FEATURES}, {len(RANDOM_SEEDS)} seeds]")
print(
    f"  Within-pair cos (seed avg): min={within_avg.min():.4f}, max={within_avg.max():.4f}, mean={within_avg.mean():.4f}"
)

mid_dens_idx = len(DENS_VALUES) // 2
within_at_mid = within_avg[:, mid_dens_idx]
monotonic = all(
    within_at_mid[i] <= within_at_mid[i + 1] for i in range(len(within_at_mid) - 1)
)
print(f"  Monotonic increase at mid-density: {monotonic}")

print("\nDone.")
