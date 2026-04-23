# %%
"""Detection metrics: F1, Precision, Recall vs L0 — 4 SAE architectures × 8 benchmarks.

Layout: two stacked 3×4 blocks (rows=metrics, cols=distributions),
with a spacer row between the two distribution groups.

Data: experiments/Kaushik/hidden_200_l8_combined.csv
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _plot_style import (
    BENCH_GROUP_1,
    BENCH_GROUP_2,
    BENCHMARK_DISPLAY,
    FS_AXIS,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    L0_SAE_LABEL,
    SAE_COLORS,
    SAE_LEGEND,
    SAE_ORDER,
    add_l0_vline,
    fig_dir,
    load_data,
    style_fig,
)

# %%
df = load_data()

METRICS = [
    ("f1", '<span style="font-style:italic;">F</span><sub>1</sub>'),
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
# --- Layout: 7 rows × 4 cols ---
# Rows 1–3: F1, Precision, Recall for distributions 1–4
# Row 4: spacer
# Rows 5–7: F1, Precision, Recall for distributions 5–8
N_M = len(METRICS)
N_COLS = 4
N_ROWS = 2 * N_M + 1  # 7
_GAP_ROW = N_M + 1  # row 4

_hs = 0.04
_vs = 0.025
_spacer_frac = 0.12

_row_heights = [1] * N_M + [_spacer_frac] + [1] * N_M

_specs = []
for r in range(N_ROWS):
    if r == _GAP_ROW - 1:
        _specs.append([None] * N_COLS)
    else:
        _specs.append([{}] * N_COLS)

# Subplot titles: distribution names above row 1 and row 5.
# make_subplots skips None-spec rows when assigning titles.
_titles = (
    [BENCHMARK_DISPLAY[b] for b in BENCH_GROUP_1]
    + [None] * N_COLS * (N_M - 1)  # rows 2–3
    + [BENCHMARK_DISPLAY[b] for b in BENCH_GROUP_2]  # row 5
    + [None] * N_COLS * (N_M - 1)  # rows 6–7
)

# --- Dimensions: square panels ---
_PANEL = 240  # panel side length in px
_margin = dict(l=90, r=15, t=45, b=80)

_fig_w = int(N_COLS * _PANEL / (1 - (N_COLS - 1) * _hs)) + _margin["l"] + _margin["r"]

_total_row_weight = 2 * N_M + _spacer_frac
_plot_area_h = (
    2 * N_M * _PANEL / ((2 * N_M / _total_row_weight) * (1 - (N_ROWS - 1) * _vs))
)
_fig_h = int(_plot_area_h) + _margin["t"] + _margin["b"]

fig = make_subplots(
    rows=N_ROWS,
    cols=N_COLS,
    subplot_titles=_titles,
    horizontal_spacing=_hs,
    vertical_spacing=_vs,
    row_heights=_row_heights,
    specs=_specs,
)

_x_lo, _x_hi = 0, 12

_GROUPS = [
    (BENCH_GROUP_1, 0),
    (BENCH_GROUP_2, N_M + 1),
]

# %%
# --- Add traces ---
for benchmarks, row_off in _GROUPS:
    for mi, (metric_key, _) in enumerate(METRICS):
        row = row_off + mi + 1
        for bi, bmark in enumerate(benchmarks):
            col = bi + 1
            true_l0 = float(agg.loc[agg["benchmark"] == bmark, "true_l0"].mean())
            add_l0_vline(fig, true_l0, row, col)

            for sae_name in SAE_ORDER:
                color = SAE_COLORS[sae_name]
                sub = agg[
                    (agg["benchmark"] == bmark) & (agg["sae_type"] == sae_name)
                ].sort_values("sae_l0")
                show_leg = row_off == 0 and mi == 0 and bi == 0
                fig.add_trace(
                    go.Scatter(
                        x=sub["sae_l0"].tolist(),
                        y=sub[metric_key].tolist(),
                        mode="lines+markers",
                        name=SAE_LEGEND[sae_name] + "&nbsp;" * 10,
                        legendgroup=sae_name,
                        showlegend=show_leg,
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
fig.update_layout(width=_fig_w, height=_fig_h, margin=_margin)
style_fig(fig, nticks=6)

fig.update_xaxes(
    tickfont=dict(size=FS_TICK),
    title_font=dict(size=FS_AXIS),
    title_standoff=6,
    automargin=False,
    dtick=2,
    range=[_x_lo, _x_hi],
    minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)
fig.update_yaxes(
    tickfont=dict(size=FS_TICK),
    title_font=dict(size=FS_AXIS),
    title_standoff=5,
    automargin=False,
    tickangle=-90,
    dtick=0.25,
    range=[0, 1],
    minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)

# Per-metric y-ranges
_Y_RANGES = [[0.3, 1], [0.3, 1], [0.18, 1]]  # F1, Precision, Recall
for _, row_off in _GROUPS:
    for mi, yrange in enumerate(_Y_RANGES):
        for ci in range(1, N_COLS + 1):
            fig.update_yaxes(range=yrange, row=row_off + mi + 1, col=ci)

# Hide x-axis tick labels on all but bottom row of each block
for _, row_off in _GROUPS:
    for mi in range(N_M - 1):  # all but last metric row
        for ci in range(1, N_COLS + 1):
            fig.update_xaxes(showticklabels=False, row=row_off + mi + 1, col=ci)

# Hide y-axis tick labels on all but leftmost column
for _, row_off in _GROUPS:
    for mi in range(N_M):
        for ci in range(2, N_COLS + 1):
            fig.update_yaxes(showticklabels=False, row=row_off + mi + 1, col=ci)

# Y-axis metric labels via built-in title (auto-centered)
for _, row_off in _GROUPS:
    for mi, (_, label) in enumerate(METRICS):
        row = row_off + mi + 1
        fig.update_yaxes(
            title_text=label,
            title_standoff=5,
            title_font=dict(
                size=FS_AXIS, family="Times New Roman, Times, serif", color="black"
            ),
            row=row,
            col=1,
        )

# x-axis label
fig.add_annotation(
    text=L0_SAE_LABEL,
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.06,
    showarrow=False,
    font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
)

# Legend just below x-axis label
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.062,
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=FS_LEGEND),
        itemsizing="constant",
        itemwidth=50,
    ),
)

# Style distribution name annotations
for ann in list(fig.layout.annotations)[: len(BENCH_GROUP_1) + len(BENCH_GROUP_2)]:
    ann.font = dict(
        size=FS_TITLE, family="Times New Roman, Times, serif", color="black"
    )
    ann.yshift = 10

fig.show()

# %%
# --- Save ---
_fd = fig_dir()
fig.write_image(f"{_fd}/detection_metrics.pdf")
fig.write_image(f"{_fd}/detection_metrics.svg")
print(f"Saved to {_fd}/detection_metrics.{{pdf,svg}}")

# %%
