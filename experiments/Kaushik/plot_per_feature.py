# %%
"""Per-feature F1 and MCC vs feature index — 4 SAE architectures at L0=4 and L0=8.

Layout: 1×4 (F1@L0=4, MCC@L0=4, F1@L0=8, MCC@L0=8).

Data: experiments/Kaushik/zipfian_sorted_curves_l0_4_8.csv
"""

import os as _os
import pathlib as _pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _plot_style import (
    FS_AXIS,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    SAE_COLORS,
    SAE_LEGEND,
    fig_dir,
    style_fig,
)

# %%
_csv_dir = (
    _pathlib.Path(__file__).parent.resolve()
    if "__file__" in dir()
    else _pathlib.Path(_os.environ.get("PLOT_DIR", "experiments/Kaushik"))
)
df = pd.read_csv(_csv_dir / "zipfian_sorted_curves_l0_4_8.csv")

METRICS = [
    ("f1", '<span style="font-style:italic;">F</span><sub>1</sub>'),
    ("mcc", "MCC"),
]
L0_TARGETS = [4, 8]
ARCHS = ["Standard", "BatchTopK", "Matryoshka", "MatchingPursuit"]

# Columns grouped by L0: (metric_key, metric_label, l0)
_PANELS = [(mk, ml, l0) for l0 in L0_TARGETS for mk, ml in METRICS]

# %%
N_DATA_COLS = len(_PANELS)  # 4
N_COLS = 5  # 2 + spacer + 2

_hs = 0.055
_PANEL = 280
_margin = dict(l=70, r=15, t=45, b=140)

_col_widths = [1, 1, 0.08, 1, 1]
_specs = [[{}, {}, None, {}, {}]]

_fig_w = (
    int(N_DATA_COLS * _PANEL / (1 - (N_DATA_COLS - 1) * _hs))
    + _margin["l"]
    + _margin["r"]
)
_fig_h = _PANEL + _margin["t"] + _margin["b"]

# Subplot titles: L⁰ value centered over each pair (assigned to cols 1 and 4)
_L0_LABEL = '<span style="font-style:italic;">L</span><sup>0</sup><sub>SAE</sub>'
_titles = [f"{_L0_LABEL} = 4", None, None, f"{_L0_LABEL} = 8", None]


def _data_col(panel_idx):
    """Map panel index (0-3) to subplot column (skipping spacer at col 3)."""
    return panel_idx + 1 if panel_idx < 2 else panel_idx + 2


fig = make_subplots(
    rows=1,
    cols=N_COLS,
    subplot_titles=_titles,
    horizontal_spacing=_hs,
    column_widths=_col_widths,
    specs=_specs,
)

# %%
# --- Add traces ---
for ci, (metric_key, _, l0) in enumerate(_PANELS):
    col = _data_col(ci)
    for arch in ARCHS:
        sub = df[(df["arch"] == arch) & (df["l0"] == l0)].sort_values("feature_index")
        color = SAE_COLORS[arch]
        show_leg = ci == 0
        fig.add_trace(
            go.Scatter(
                x=sub["feature_index"].tolist(),
                y=sub[metric_key].tolist(),
                mode="markers",
                name=SAE_LEGEND[arch] + "&nbsp;" * 10,
                legendgroup=arch,
                showlegend=False,
                marker=dict(size=3, color=color, opacity=0.4),
            ),
            row=1,
            col=col,
        )
        # Legend-only trace with full opacity
        if show_leg:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=SAE_LEGEND[arch] + "&nbsp;" * 10,
                    legendgroup=arch,
                    showlegend=True,
                    marker=dict(size=8, color=color, opacity=1),
                ),
                row=1,
                col=col,
            )

# %%
# --- Styling ---
fig.update_layout(width=_fig_w, height=_fig_h, margin=_margin)
style_fig(fig, nticks=6)

fig.update_xaxes(
    tickfont=dict(size=FS_TICK),
    title_font=dict(size=FS_AXIS),
    title_standoff=6,
    automargin=False,
    minor=dict(showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)
fig.update_yaxes(
    tickfont=dict(size=FS_TICK),
    title_font=dict(size=FS_AXIS),
    title_standoff=5,
    automargin=False,
    tickangle=-90,
    dtick=0.25,
    range=[-0.05, 1],
    minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)

# Y-axis metric labels:
# F₁ panels (cols 1, 4): use built-in title_text — space to the left (margin/spacer)
# MCC panels (cols 2, 5): use a paper annotation positioned in the gap to the RIGHT
#   of the MCC panel's tick labels, avoiding collision with the F₁ plot area to the left.
for ci, (_, label, _) in enumerate(_PANELS):
    col = _data_col(ci)
    if ci in (0, 2):  # F₁ panels — built-in title works fine
        fig.update_yaxes(
            title_text=label,
            title_standoff=3,
            title_font=dict(
                size=FS_AXIS,
                family="Times New Roman, Times, serif",
                color="black",
            ),
            row=1,
            col=col,
        )
    else:  # MCC panels — use built-in title, hide tick labels
        fig.update_yaxes(
            showticklabels=False,
            title_text=label,
            title_standoff=3,
            title_font=dict(
                size=FS_AXIS,
                family="Times New Roman, Times, serif",
                color="black",
            ),
            row=1,
            col=col,
        )

# x-axis label
fig.add_annotation(
    text="Feature Index",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.24,
    showarrow=False,
    font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
)

# Legend just below x-axis label
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.26,
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=FS_LEGEND),
        itemsizing="constant",
        itemwidth=50,
    ),
)

# Style and reposition title annotations to center over each L0 pair
_n_real_titles = sum(1 for t in _titles if t is not None)
# Group 1: cols 1,2 (indices 0,1); Group 2: cols 4,5 (indices 3,4)
_group_col_pairs = [(0, 1), (3, 4)]
_group_centers = []
for c1, c2 in _group_col_pairs:
    x1 = fig.layout[fig._grid_ref[0][c1][0].layout_keys[0]].domain
    x2 = fig.layout[fig._grid_ref[0][c2][0].layout_keys[0]].domain
    _group_centers.append((x1[0] + x2[1]) / 2)

for i, ann in enumerate(list(fig.layout.annotations)[:_n_real_titles]):
    ann.font = dict(
        size=FS_TITLE, family="Times New Roman, Times, serif", color="black"
    )
    ann.yshift = 10
    ann.x = _group_centers[i]

fig.show()

# %%
# --- Save ---
_fd = fig_dir()
fig.write_image(f"{_fd}/per_feature_zipfian.pdf")
fig.write_image(f"{_fd}/per_feature_zipfian.svg")
print(f"Saved to {_fd}/per_feature_zipfian.{{pdf,svg}}")


# %%
