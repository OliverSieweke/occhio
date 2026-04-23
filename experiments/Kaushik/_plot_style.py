"""Shared styling constants and helpers for Kaushik experiment plots.

All publication-ready plots import from here to stay visually consistent.
"""

import os
import pathlib

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load the combined CSV and return a DataFrame with display-friendly columns."""
    csv = _find_csv()
    df = pd.read_csv(csv)
    df["benchmark_display"] = df["benchmark"].map(BENCHMARK_DISPLAY)
    return df


def _find_csv() -> str:
    name = "hidden_200_l8_combined.csv"

    if "__file__" in dir():
        p = pathlib.Path(__file__).parent / name
        if p.exists():
            return str(p)

    for base in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        for rel in (".", "experiments/Kaushik"):
            p = (base / rel / name).resolve()
            if p.exists():
                return str(p)

    raise FileNotFoundError(f"Cannot locate {name}")


def fig_dir() -> str:
    """Return (and create) the figures output directory next to the CSV."""
    d = os.path.join(os.path.dirname(_find_csv()), "figures")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Benchmark display names & canonical order
# ---------------------------------------------------------------------------

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

# Split into two groups of 4 for the two-panel layouts
BENCH_GROUP_1 = BENCHMARK_ORDER[:4]
BENCH_GROUP_2 = BENCHMARK_ORDER[4:]

# ---------------------------------------------------------------------------
# SAE styling
# ---------------------------------------------------------------------------

SAE_ORDER = ["Standard", "BatchTopK", "Matryoshka", "MatchingPursuit"]

SAE_COLORS = {
    "Standard": "#ffcc00",
    "BatchTopK": "#3360FF",
    "Matryoshka": "#ed5400",
    "MatchingPursuit": "#41b5b2",
}

SAE_LEGEND = {
    "Standard": "ReLU",
    "BatchTopK": "BatchTopK",
    "Matryoshka": "Matryoshka",
    "MatchingPursuit": "Matching Pursuit",
}

_LEGEND_PAD = "&nbsp;" * 10

# ---------------------------------------------------------------------------
# AE baselines (for F1 hatch region)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Axis & figure styling
# ---------------------------------------------------------------------------

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

# Font sizes used across all multi-panel plots
FS_TITLE = 24  # subplot panel titles
FS_AXIS = 26  # axis label font size
FS_TICK = 18  # tick label font size
FS_LEGEND = 20  # legend font size
PLOT_H = 300  # panel side length in px


def style_fig(fig, nticks=6):
    """Apply publication-ready styling."""
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


def apply_panel_style(fig, *, margin, x_label_y=-0.14, legend_y=-0.14):
    """Common layout overrides for multi-panel figures."""
    fig.update_layout(margin=margin)
    style_fig(fig, nticks=6)

    fig.update_xaxes(
        tickfont=dict(size=FS_TICK),
        title_font=dict(size=FS_AXIS),
        title_standoff=6,
        automargin=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=FS_TICK),
        title_font=dict(size=FS_AXIS),
        title_standoff=5,
        automargin=False,
        tickangle=-90,
    )

    # Legend below the x-axis label
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=legend_y,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=FS_LEGEND),
            itemsizing="constant",
            itemwidth=50,
        ),
    )
    return fig


def add_l0_vline(fig, true_l0, row, col):
    """Add a grey L0 reference line (as a scatter trace for z-order control)."""
    fig.add_trace(
        go.Scatter(
            x=[true_l0, true_l0],
            y=[-10, 10],
            mode="lines",
            line=dict(color="#9CA3AF", width=2.0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def set_subplot_title_style(fig, n_titles):
    """Style only the first n_titles annotations (subplot titles) with yshift."""
    for ann in list(fig.layout.annotations)[:n_titles]:
        ann.font = dict(
            size=FS_TITLE,
            family="Times New Roman, Times, serif",
            color="black",
        )
        ann.yshift = 8


def add_x_axis_label(fig, text, y=-0.14):
    """Add a centered x-axis label below all panels."""
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=y,
        showarrow=False,
        font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
    )


def add_y_axis_label(fig, text, x=-0.05, y=0.5):
    """Add a centered, rotated y-axis label."""
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        showarrow=False,
        textangle=-90,
        font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
    )


L0_SAE_LABEL = (
    '<span style="font-family:Times New Roman; font-style:italic;">L</span>'
    "<sup>0</sup><sub>SAE</sub>"
)

F1_LABEL = (
    "<span style=\"font-family: 'Times New Roman', Times, serif; "
    'font-style: italic;">F<sub style="font-style: normal;">1</sub></span>'
)
