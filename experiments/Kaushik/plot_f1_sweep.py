# %%
"""F1 vs L0 sweep — 4 SAE architectures across 8 benchmark distributions.

Data: experiments/Kaushik/hidden_200_l8_*/results.parquet
Styling: matches sae_l1_sweep_sparse.py (publication-ready, Times New Roman).
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# --- Load data ---
import pathlib as _pathlib


def _find_data_dir() -> str:
    def _valid(p: _pathlib.Path) -> bool:
        return (p / "hidden_200_l8_standard").exists()

    # 1. Script's own directory (works when __file__ points to the right place)
    if "__file__" in dir():
        _c = _pathlib.Path(__file__).parent.resolve()
        if _valid(_c):
            return str(_c)

    # 2. Walk up from cwd, checking common relative locations at each level
    for _p in [_pathlib.Path.cwd(), *_pathlib.Path.cwd().parents]:
        for _rel in (".", "experiments/Kaushik", "experiments/nic/Kaushik"):
            _c = (_p / _rel).resolve()
            if _valid(_c):
                return str(_c)

    raise RuntimeError(
        "Cannot locate data directory containing hidden_200_l8_standard/. "
        "Set _DATA_DIR manually."
    )


_DATA_DIR = _find_data_dir()

DATASETS = {
    "Standard": "hidden_200_l8_standard",
    "BatchTopK": "hidden_200_l8_batch",
    "Matryoshka": "hidden_200_l8_matryoshka",
    "MatchingPursuit": "hidden_200_l8_mp",
}

dfs: dict[str, pd.DataFrame] = {}
for sae_name, subdir in DATASETS.items():
    path = os.path.join(_DATA_DIR, subdir, "results.parquet")
    dfs[sae_name] = pd.read_parquet(path)

# %%
# --- Benchmark display names & order ---
BENCHMARK_DISPLAY = {
    "SPARSE_UNIFORM": "Zipfian",
    "CORRELATED_PAIRS": "Correlated Pairs",
    "HIERARCHICAL_PAIRS": "Hierarchical Pairs",
    "POWER_LAW_DIGRAPH": "Preferential Attachment",
    "DAG_RANDOM_WALK": "Deep Hierarchy",
    "SIMPLICIAL_COMPLEX": "Simplicial Complex",
    "SPHERICAL": "Spherical",
    "TORUS": "Torus",
}

BENCHMARK_ORDER = [
    "SPARSE_UNIFORM",
    "CORRELATED_PAIRS",
    "HIERARCHICAL_PAIRS",
    "POWER_LAW_DIGRAPH",
    "DAG_RANDOM_WALK",
    "SIMPLICIAL_COMPLEX",
    "SPHERICAL",
    "TORUS",
]

# %%
# --- Styling (mirrors sae_l1_sweep_sparse.py exactly) ---
SAE_COLORS = {
    "Standard": "#ffcc00",
    "BatchTopK": "#3360FF",
    "Matryoshka": "#ed5400",
    "MatchingPursuit": "#41b5b2",
}

SAE_LEGEND = {
    "Standard": "ReLU&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
    "BatchTopK": "BatchTopK&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
    "Matryoshka": "Matryoshka&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
    "MatchingPursuit": "Matching Pursuit&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
}

_AXIS = dict(
    showgrid=True,
    gridcolor="#E5E7EB",
    gridwidth=1,
    showline=True,
    linecolor="black",
    linewidth=2.5,
    mirror=True,
    ticks="outside",
    ticklen=8,
    tickwidth=1.5,
    tickcolor="black",
    minor=dict(ticks="", showgrid=False),
    zeroline=False,
    tickfont=dict(size=43, color="black"),
    title_font=dict(size=46, color="black"),
)


def style_fig(fig, nticks=6):
    """Apply publication-ready styling (identical to sae_l1_sweep_sparse.py)."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", size=46, color="black"),
        title_font=dict(size=46),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            itemsizing="constant",
            font=dict(size=39),
        ),
    )
    fig.update_xaxes(**_AXIS, nticks=nticks)
    fig.update_yaxes(**_AXIS, nticks=nticks)
    return fig


# %%
# --- Build 2x4 subplot figure ---
N_ROWS, N_COLS = 2, 4
_hs = 0.04  # horizontal spacing fraction
_vs = 0.10  # vertical spacing fraction

AE_BASELINES = {
    "SPARSE_UNIFORM": 0.9803585410118103,
    "CORRELATED_PAIRS": 0.9804293513298035,
    "HIERARCHICAL_PAIRS": 0.9843034148216248,
    "POWER_LAW_DIGRAPH": 0.9646085500717163,
    "DAG_RANDOM_WALK": 0.9777095317840576,
    "SIMPLICIAL_COMPLEX": 0.9962378740310669,
    "SPHERICAL": 0.9793679714202881,
    "TORUS": 0.978403627872467,
}

# Square-panel dimensions (same formula as sae_l1_sweep_sparse.py)
_fs_title = 24  # subplot panel titles ("Zipfian", "Correlated Pairs", …)Let
_fs = 26  # axis label font size (F₁, L⁰_SAE)
_fs_tick = 18  # tick label font size (0, 0.25, 2, 4, …)
_fs_leg = 20  # legend font size
_plot_h = 300  # panel side length in px
_margin = dict(l=70, r=10, t=45, b=90)  # tight margins; bottom holds x-label + legend

_fig_w = int(N_COLS * _plot_h / (1 - (N_COLS - 1) * _hs)) + _margin["l"] + _margin["r"]
_fig_h = int(N_ROWS * _plot_h / (1 - (N_ROWS - 1) * _vs)) + _margin["t"] + _margin["b"]

fig = make_subplots(
    rows=N_ROWS,
    cols=N_COLS,
    subplot_titles=[BENCHMARK_DISPLAY[b] for b in BENCHMARK_ORDER],
    horizontal_spacing=_hs,
    vertical_spacing=_vs,
)

_x_lo = 0
_x_hi = 12

# %%
# --- Phase 1: L0 reference lines as scatter traces (added first → lowest z-order,
#              so SAE data renders on top). Hatch is added later but has transparent
#              background so the line remains visible through the gaps. ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row = bi // N_COLS + 1
    col = bi % N_COLS + 1
    true_l0 = float(dfs["Standard"].xs(bmark, level="benchmark")["true_l0"].mean())
    fig.add_trace(
        go.Scatter(
            x=[true_l0, true_l0],
            y=[0, 1.05],
            mode="lines",
            line=dict(color="#9CA3AF", width=2.0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

# --- Phase 2: SAE data traces ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row = bi // N_COLS + 1
    col = bi % N_COLS + 1
    show_legend = bi == 0

    for sae_name, df in dfs.items():
        color = SAE_COLORS[sae_name]
        sub = df.xs(bmark, level="benchmark").sort_values("sae_l0")
        x = sub["sae_l0"].values
        y = sub["f1_score"].values

        fig.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=y.tolist(),
                mode="lines+markers",
                name=SAE_LEGEND[sae_name],
                legendgroup=sae_name,
                showlegend=show_legend,
                marker=dict(size=8, color=color, line=dict(width=2, color="white")),
                line=dict(color=color, width=2.5),
            ),
            row=row,
            col=col,
        )

# --- Phase 3: white fill to blank the hatch region, then diagonal black lines
#              drawn via add_shape — guaranteed opaque black, no compositing artefacts. ---
# Slope for 45-degree visual lines on a square panel in data coordinates
_slope = 1.0 / (_x_hi - _x_lo)
# Spacing between lines: ~10 px expressed in x-data units
_gap_x = 10 * (_x_hi - _x_lo) / _plot_h

for bi, bmark in enumerate(BENCHMARK_ORDER):
    row = bi // N_COLS + 1
    col = bi % N_COLS + 1
    baseline = AE_BASELINES[bmark]
    y_top = 1.02
    # Opaque white background to clear the region
    fig.add_shape(
        type="rect",
        x0=_x_lo,
        y0=baseline,
        x1=_x_hi,
        y1=y_top,
        fillcolor="white",
        line=dict(width=0),
        layer="above",
        row=row,
        col=col,
    )
    # Diagonal "/" lines across the region
    x_extent = (y_top - baseline) / _slope
    t = _x_lo - x_extent
    while t <= _x_hi:
        x0 = max(_x_lo, t)
        y0 = baseline + _slope * max(0.0, _x_lo - t)
        x1 = min(_x_hi, t + x_extent)
        y1 = min(y_top, baseline + _slope * (x1 - x0) + (y0 - baseline))
        if x1 > x0:
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="black", width=1),
                layer="above",
                row=row,
                col=col,
            )
        t += _gap_x

# --- Phase 4: bold bottom border of hatched region ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row = bi // N_COLS + 1
    col = bi % N_COLS + 1
    baseline = AE_BASELINES[bmark]
    fig.add_hline(
        y=baseline,
        line_color="black",
        line_width=2,
        layer="above",
        row=row,
        col=col,
    )

# %%
# --- Apply styling ---
fig.update_layout(width=_fig_w, height=_fig_h, margin=_margin)
style_fig(fig, nticks=6)

# Global axis overrides
fig.update_xaxes(
    tickfont=dict(size=_fs_tick),
    title_font=dict(size=_fs),
    title_standoff=6,
    automargin=False,
    dtick=2,
    range=[1.3, 11.7],
    minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)
fig.update_yaxes(
    tickfont=dict(size=_fs_tick),
    title_font=dict(size=_fs),
    title_standoff=5,
    automargin=False,
    dtick=0.25,
    range=[0.45, 1],
    tickangle=-90,
    minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)

# Single F₁ label centered vertically across both rows.
# Wrap in a span so both the F and ₁ share the same italic Times New Roman face,
# avoiding a font-mixing artifact between the italic <i> tag and the Unicode subscript.
fig.add_annotation(
    text='<span style="font-family: \'Times New Roman\', Times, serif; font-style: italic;">F<sub style="font-style: normal;">1</sub></span>',
    xref="paper",
    yref="paper",
    x=-0.05,
    y=0.5,
    showarrow=False,
    textangle=-90,
    font=dict(size=_fs, family="Times New Roman, Times, serif", color="black"),
)

# Hide x-axis tick labels on top row (redundant — shared with bottom row)
for ci in range(1, N_COLS + 1):
    fig.update_xaxes(showticklabels=False, row=1, col=ci)

# Hide y-axis tick labels on all but leftmost column
for ri in range(1, N_ROWS + 1):
    for ci in range(2, N_COLS + 1):
        fig.update_yaxes(showticklabels=False, row=ri, col=ci)

# Centered x-axis label via paper annotation (below all bottom panels)
fig.add_annotation(
    text=(
        '<span style="font-family:Times New Roman; font-style:italic;">L</span>'
        "<sup>0</sup><sub>SAE</sub>"
    ),
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.14,
    showarrow=False,
    font=dict(size=_fs, family="Times New Roman, Times, serif", color="black"),
)

# Legend: below the x-axis label
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.14,
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=_fs_leg),
        itemsizing="constant",
        itemwidth=50,
    ),
)

# Subplot title font (override make_subplots default) + shift up for breathing room
# Only apply yshift to subplot titles (those added by make_subplots), not our axis labels
_n_subplot_titles = len(BENCHMARK_ORDER)
for ann in list(fig.layout.annotations)[:_n_subplot_titles]:
    ann.font = dict(
        size=_fs_title,
        family="Times New Roman, Times, serif",
        color="black",
    )
    ann.yshift = 8

fig.show()

# %%
# --- Save as vector (PDF + SVG) ---
_fig_dir = os.path.join(_DATA_DIR, "figures")
os.makedirs(_fig_dir, exist_ok=True)

fig.write_image(os.path.join(_fig_dir, "f1_sweep.pdf"), engine="kaleido")
fig.write_image(os.path.join(_fig_dir, "f1_sweep.svg"), engine="kaleido")
print(f"Saved to {_fig_dir}/f1_sweep.{{pdf,svg}}")

# %%
