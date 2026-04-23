# %%
"""Detection metrics: F1, Precision, Recall vs L0 — 4 SAE architectures × 8 benchmarks.

Layout: 3 metric rows × 9 columns (4 distributions + spacer + 4 distributions).

Data: experiments/Kaushik/hidden_200_l8_combined.csv
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _plot_style import (
    BENCHMARK_DISPLAY,
    BENCHMARK_ORDER,
    FS_AXIS,
    L0_SAE_LABEL,
    PLOT_H,
    SAE_COLORS,
    SAE_LEGEND,
    SAE_ORDER,
    add_l0_vline,
    apply_panel_style,
    fig_dir,
    load_data,
    set_subplot_title_style,
)

# %%
df = load_data()

METRICS = [
    ("f1", "F<sub>1</sub>"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]
METRIC_COLS = [m[0] for m in METRICS]

agg = (
    df.groupby(["benchmark", "sae_type", "sae"])
    .agg(
        sae_l0=("sae_l0", "mean"),
        true_l0=("true_l0", "mean"),
        **{m: (m, "mean") for m in METRIC_COLS},
    )
    .reset_index()
)

# %%
# --- Layout: 3 rows × 9 cols (col 5 is a spacer) ---
N_METRIC_ROWS = len(METRICS)
N_REAL_COLS = 8
N_COLS = 9  # 4 + spacer + 4
_GAP_COL = 5

_hs = 0.02
_vs = 0.08

# Column widths: real panels = 1, spacer = 0.15
_col_widths = [1, 1, 1, 1, 0.15, 1, 1, 1, 1]

# Specs: spacer column is None
_specs = [
    [{} if c != _GAP_COL - 1 else None for c in range(N_COLS)]
    for _ in range(N_METRIC_ROWS)
]

# Subplot titles: distribution names on top row, spacer is blank
_bench_titles = []
for i, b in enumerate(BENCHMARK_ORDER):
    if i == 4:
        _bench_titles.append(None)  # spacer
    _bench_titles.append(BENCHMARK_DISPLAY[b])
_titles = _bench_titles + [None] * (N_METRIC_ROWS - 1) * N_COLS

_margin = dict(l=80, r=10, t=45, b=90)
_fig_w = (
    int(N_REAL_COLS * PLOT_H / (1 - (N_REAL_COLS - 1) * _hs))
    + _margin["l"]
    + _margin["r"]
)
_fig_h = (
    int(N_METRIC_ROWS * PLOT_H / (1 - (N_METRIC_ROWS - 1) * _vs))
    + _margin["t"]
    + _margin["b"]
)

fig = make_subplots(
    rows=N_METRIC_ROWS,
    cols=N_COLS,
    subplot_titles=_titles,
    horizontal_spacing=_hs,
    vertical_spacing=_vs,
    column_widths=_col_widths,
    specs=_specs,
)

_x_lo, _x_hi = 0, 12


def _data_col(bi):
    """Map benchmark index (0–7) to actual subplot column (skipping spacer)."""
    return bi + 1 if bi < 4 else bi + 2


# %%
# --- Add traces ---
for mi, (metric_key, _) in enumerate(METRICS):
    row = mi + 1
    for bi, bmark in enumerate(BENCHMARK_ORDER):
        col = _data_col(bi)
        true_l0 = float(agg.loc[agg["benchmark"] == bmark, "true_l0"].mean())
        add_l0_vline(fig, true_l0, row, col)

        for sae_name in SAE_ORDER:
            color = SAE_COLORS[sae_name]
            sub = agg[
                (agg["benchmark"] == bmark) & (agg["sae_type"] == sae_name)
            ].sort_values("sae_l0")
            fig.add_trace(
                go.Scatter(
                    x=sub["sae_l0"].tolist(),
                    y=sub[metric_key].tolist(),
                    mode="lines+markers",
                    name=SAE_LEGEND[sae_name] + "&nbsp;" * 10,
                    legendgroup=sae_name,
                    showlegend=(mi == 0 and bi == 0),
                    marker=dict(
                        size=6, color=color, line=dict(width=1.5, color="white")
                    ),
                    line=dict(color=color, width=2),
                ),
                row=row,
                col=col,
            )

# %%
# --- Styling ---
fig.update_layout(width=_fig_w, height=_fig_h)
apply_panel_style(fig, margin=_margin)

fig.update_xaxes(
    dtick=2,
    range=[_x_lo, _x_hi],
    minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)
fig.update_yaxes(
    dtick=0.25,
    range=[0, 1],
    minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)

# Hide x-axis tick labels on all but the bottom row
for ri in range(1, N_METRIC_ROWS):
    for bi in range(N_REAL_COLS):
        fig.update_xaxes(showticklabels=False, row=ri, col=_data_col(bi))

# Hide y-axis tick labels on all but leftmost column of each group
for ri in range(1, N_METRIC_ROWS + 1):
    for bi in range(N_REAL_COLS):
        if bi not in (0, 4):  # keep leftmost of each group
            fig.update_yaxes(showticklabels=False, row=ri, col=_data_col(bi))

# Y-axis metric labels on leftmost column
for mi, (_, label) in enumerate(METRICS):
    fig.add_annotation(
        text=(
            f"<span style=\"font-family: 'Times New Roman', Times, serif;\">"
            f"{label}</span>"
        ),
        xref="paper",
        yref="paper",
        x=-0.06,
        y=1 - (mi + 0.5) / N_METRIC_ROWS,
        showarrow=False,
        textangle=-90,
        font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
    )

# x-axis label
fig.add_annotation(
    text=L0_SAE_LABEL,
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.12,
    showarrow=False,
    font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
)

# Style only the distribution name annotations (top row)
set_subplot_title_style(fig, N_REAL_COLS)

fig.show()

# %%
# --- Save ---
_fd = fig_dir()
fig.write_image(f"{_fd}/detection_metrics.pdf", engine="kaleido")
fig.write_image(f"{_fd}/detection_metrics.svg", engine="kaleido")
print(f"Saved to {_fd}/detection_metrics.{{pdf,svg}}")

# %%
