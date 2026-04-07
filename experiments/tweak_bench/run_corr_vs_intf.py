"""Train all distributions and save corr-vs-interference plots to plots/."""

from pathlib import Path
import random

import numpy as np
import plotly.express as px
import torch

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.correlated import CorrelatedPairs, HierarchicalPairs
from occhio.distributions.sparse import SparseUniform
from occhio.distributions import (
    DAGRandomWalkToRoot,
    PowerLawDigraph,
    SimplicialComplexDistribution,
    SphericalDistribution,
    TorusDistribution,
)
from occhio.toy_model import ToyModel

# --- Config ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 1296
D_HIDDEN = 100
BATCH_SIZE = 512
N_SAMPLES = 100_000

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def _dist_generator() -> torch.Generator:
    """Fresh generator seeded from SEED for each distribution's internal RNG."""
    return torch.Generator(DEVICE).manual_seed(SEED)


OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# --- Paper-quality plot defaults ---
FONT = dict(family="Times New Roman, serif", size=24, color="#333333")
AXIS_STYLE = dict(
    showgrid=False,
    gridcolor="rgba(0,0,0,0.08)",
    gridwidth=1,
    zeroline=False,
    linecolor="#666666",
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickwidth=1,
    tickcolor="#666666",
    tickfont_size=18,
    minor=dict(ticks="outside", ticklen=2),
)
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=FONT,
    title_font_size=24,
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        font_size=24,
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
)


# --- Distribution factories ---
def make_sparse_unif():
    high = 0.3
    low = 1.28 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    return SparseUniform(
        N_FEATURES, p_active=firing_probs, device=DEVICE, generator=_dist_generator()
    )


def make_correlated_pairs():
    np.random.seed(8)
    high = 0.5
    low = 1.22 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    corrs = 0.5 + 0.5 * np.random.random(N_FEATURES)
    return CorrelatedPairs(
        N_FEATURES,
        p_active=firing_probs,
        p_individual=corrs,
        device=DEVICE,
        generator=_dist_generator(),
    )


def make_hierarchical_pairs():
    np.random.seed(8)
    high = 0.45
    low = 1.3 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    betas = np.random.random(N_FEATURES)
    return HierarchicalPairs(
        N_FEATURES,
        p_active=firing_probs,
        p_follow=0.6,
        beta=betas,
        device=DEVICE,
        generator=_dist_generator(),
    )


def make_RWTR():
    return DAGRandomWalkToRoot(
        n_features=N_FEATURES,
        p_edge=50 / N_FEATURES,
        beta=0.8,
        shrinking=True,
        generator=_dist_generator(),
    )


def make_digraph():
    return PowerLawDigraph(
        n_features=N_FEATURES,
        p_active=3.3 / N_FEATURES,
        alpha=1,
        p_edge=4.1 / N_FEATURES,
        p_child=(0.1, 0.4),
        generator=_dist_generator(),
    )


def make_simplicial_complex():
    random.seed(SEED)
    FACE_DIM = 4
    N_FACES = 4 * (N_FEATURES // (FACE_DIM + 1))
    all_verts = list(range(N_FEATURES))
    random.shuffle(all_verts)
    face_size = FACE_DIM + 1
    covering_faces = set()
    for i in range(0, N_FEATURES, face_size):
        chunk = all_verts[i : i + face_size]
        if len(chunk) < face_size:
            remaining = [v for v in all_verts if v not in chunk]
            chunk += random.sample(remaining, face_size - len(chunk))
        covering_faces.add(tuple(sorted(chunk)))
    while len(covering_faces) < N_FACES:
        covering_faces.add(tuple(sorted(random.sample(range(N_FEATURES), face_size))))
    faces = list(covering_faces)[:N_FACES]
    return SimplicialComplexDistribution(
        n_vertices=N_FEATURES,
        faces=faces,
        sampling_mode="sparse",
        p_active=1 / N_FACES,
        generator=_dist_generator(),
    )


def make_spherical():
    return SphericalDistribution(
        n_features=N_FEATURES,
        length_scale=0.245,
        manifold_dim=4,
        magnitude_range=(0.5, 1.0),
        generator=_dist_generator(),
    )


def make_torus():
    return TorusDistribution(
        n_features=N_FEATURES,
        length_scale=0.669,
        torus_dim=4,
        magnitude_range=(0.5, 1.0),
        generator=_dist_generator(),
    )


N_PAIRS = 200_000

EXPERIMENTS = [
    ("sparse_unif", make_sparse_unif, 20_000),
    ("correlated_pairs", make_correlated_pairs, 25_000),
    ("hierarchical_pairs", make_hierarchical_pairs, 25_000),
    ("RWTR", make_RWTR, 20_000),
    ("digraph", make_digraph, 20_000),
    ("simplicial_complex", make_simplicial_complex, 20_000),
    ("spherical", make_spherical, 25_000),
    ("torus", make_torus, 25_000),
]


def make_corr_vs_intf_plot(name: str, tm: ToyModel, samples: torch.Tensor):
    """Build and save a single corr-vs-interference scatter plot."""
    empirical_corr = torch.corrcoef(samples.T.cpu())
    interferences = tm.interferences.detach().cpu()
    rows_i, rows_j = torch.triu_indices(N_FEATURES, N_FEATURES, offset=1)
    pair_corr = empirical_corr[rows_i, rows_j].numpy()
    pair_interf = interferences[rows_i, rows_j].numpy()

    _x = pair_corr[:N_PAIRS]
    _y = pair_interf[:N_PAIRS]
    _slope, _intercept = np.polyfit(_x, _y, 1)
    _resid = _y - (_slope * _x + _intercept)
    _ss_res = np.sum(_resid**2)
    _ss_tot = np.sum((_y - _y.mean()) ** 2)
    _r2 = 1 - _ss_res / _ss_tot
    _n = len(_x)
    _se_slope = np.sqrt(_ss_res / (_n - 2) / np.sum((_x - _x.mean()) ** 2))
    _corr = np.corrcoef(_x, _y)[0, 1]
    _z = np.arctanh(_corr)
    _z_se = 1.0 / np.sqrt(_n - 3)
    _corr_lo = np.tanh(_z - 1.96 * _z_se)
    _corr_hi = np.tanh(_z + 1.96 * _z_se)

    fig = px.scatter(
        x=_x,
        y=_y,
        labels={"x": "Empirical correlation", "y": "Interference"},
        opacity=0.8,
        trendline="ols",
        trendline_color_override="black",
    )
    fig.update_traces(marker_size=3, selector=dict(mode="markers"))
    fig.update_traces(opacity=0.8, selector=dict(mode="lines"))
    fig.add_hline(y=0, line_color="gray", line_width=1, layer="below")
    fig.add_annotation(
        text=(
            f"slope = {_slope:.3f} \u00b1 {1.96 * _se_slope:.3f}<br>"
            f"R\u00b2 = {_r2:.3f}<br>"
            f"r = {_corr:.3f} [{_corr_lo:.3f}, {_corr_hi:.3f}]"
        ),
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=22),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#cccccc",
        borderwidth=1,
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    out_path = OUT_DIR / f"{name}_corr_vs_intf.png"
    fig.write_image(str(out_path), width=750, height=450, scale=2)
    print(f"  Saved {out_path}")


# --- Main loop ---
if __name__ == "__main__":
    for name, make_dist, n_epochs in EXPERIMENTS:
        print(f"\n{'=' * 60}")
        print(f"Running: {name} (epochs={n_epochs})")
        print(f"{'=' * 60}")

        dist = make_dist()
        samples = dist.sample(N_SAMPLES)
        mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
        print(f"  Average L0: {mean_l0:.2f}")

        gen = torch.Generator(DEVICE).manual_seed(SEED)
        ae = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen)
        tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
        tm.fit(n_epochs, batch_size=BATCH_SIZE, verbose=True)

        make_corr_vs_intf_plot(name, tm, samples)

    print(f"\nAll plots saved to {OUT_DIR}")
