# %%
"""F1 vs L0 sweep — 4 SAE architectures across 8 benchmark distributions.

Data: experiments/Kaushik/hidden_200_l8_combined.csv
Styling: publication-ready, Times New Roman.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _plot_style import (
    AE_BASELINES,
    BENCHMARK_DISPLAY,
    BENCHMARK_ORDER,
    F1_LABEL,
    FS_AXIS,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
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
    style_fig,
)

# %%
# --- Load data ---
df = load_data()

# Aggregate: mean over seeds per (benchmark, sae_type, sae)
# Use "sae" column (e.g. Standard_0, BatchTopK_3) as the sweep-point key
# because l0_target / k are NaN for some SAE types.
agg = (
    df.groupby(["benchmark", "sae_type", "sae"])
    .agg(
        sae_l0=("sae_l0", "mean"),
        f1_score=("f1", "mean"),
        true_l0=("true_l0", "mean"),
    )
    .reset_index()
)

# %%
# --- Build 2x4 subplot figure ---
N_ROWS, N_COLS = 2, 4
_hs = 0.04
_vs = 0.10
_margin = dict(l=70, r=10, t=45, b=90)

_fig_w = int(N_COLS * PLOT_H / (1 - (N_COLS - 1) * _hs)) + _margin["l"] + _margin["r"]
_fig_h = int(N_ROWS * PLOT_H / (1 - (N_ROWS - 1) * _vs)) + _margin["t"] + _margin["b"]

fig = make_subplots(
    rows=N_ROWS,
    cols=N_COLS,
    subplot_titles=[BENCHMARK_DISPLAY[b] for b in BENCHMARK_ORDER],
    horizontal_spacing=_hs,
    vertical_spacing=_vs,
)

_x_lo, _x_hi = 0, 12

# %%
# --- Phase 1: L0 reference lines (lowest z-order) ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row, col = bi // N_COLS + 1, bi % N_COLS + 1
    true_l0 = float(agg.loc[agg["benchmark"] == bmark, "true_l0"].mean())
    add_l0_vline(fig, true_l0, row, col)

# --- Phase 2: SAE data traces ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row, col = bi // N_COLS + 1, bi % N_COLS + 1
    show_legend = bi == 0

    for sae_name in SAE_ORDER:
        color = SAE_COLORS[sae_name]
        sub = agg[
            (agg["benchmark"] == bmark) & (agg["sae_type"] == sae_name)
        ].sort_values("sae_l0")
        fig.add_trace(
            go.Scatter(
                x=sub["sae_l0"].tolist(),
                y=sub["f1_score"].tolist(),
                mode="lines+markers",
                name=SAE_LEGEND[sae_name] + "&nbsp;" * 10,
                legendgroup=sae_name,
                showlegend=show_legend,
                marker=dict(size=8, color=color, line=dict(width=2, color="white")),
                line=dict(color=color, width=2.5),
            ),
            row=row,
            col=col,
        )

# --- Phase 3: hatched AE baseline region ---
_slope = 1.0 / (_x_hi - _x_lo)
_gap_x = 10 * (_x_hi - _x_lo) / PLOT_H

for bi, bmark in enumerate(BENCHMARK_ORDER):
    row, col = bi // N_COLS + 1, bi % N_COLS + 1
    baseline = AE_BASELINES[bmark]
    y_top = 1.02
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

# --- Phase 4: bold bottom border ---
for bi, bmark in enumerate(BENCHMARK_ORDER):
    row, col = bi // N_COLS + 1, bi % N_COLS + 1
    fig.add_hline(
        y=AE_BASELINES[bmark],
        line_color="black",
        line_width=2,
        layer="above",
        row=row,
        col=col,
    )

# %%
# --- Apply styling ---
fig.update_layout(width=_fig_w, height=_fig_h)
apply_panel_style(fig, margin=_margin)

fig.update_xaxes(
    dtick=2,
    range=[_x_lo, _x_hi],
    minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)
fig.update_yaxes(
    dtick=0.25,
    range=[0.45, 1],
    minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
)

# F₁ y-axis label
fig.add_annotation(
    text=F1_LABEL,
    xref="paper",
    yref="paper",
    x=-0.05,
    y=0.5,
    showarrow=False,
    textangle=-90,
    font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
)

# Hide redundant tick labels
for ci in range(1, N_COLS + 1):
    fig.update_xaxes(showticklabels=False, row=1, col=ci)
for ri in range(1, N_ROWS + 1):
    for ci in range(2, N_COLS + 1):
        fig.update_yaxes(showticklabels=False, row=ri, col=ci)

# x-axis label
fig.add_annotation(
    text=L0_SAE_LABEL,
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.14,
    showarrow=False,
    font=dict(size=FS_AXIS, family="Times New Roman, Times, serif", color="black"),
)

set_subplot_title_style(fig, len(BENCHMARK_ORDER))

fig.show()

# %%
# --- Save ---
_fd = fig_dir()
fig.write_image(f"{_fd}/f1_sweep.pdf", engine="kaleido")
fig.write_image(f"{_fd}/f1_sweep.svg", engine="kaleido")
print(f"Saved to {_fd}/f1_sweep.{{pdf,svg}}")

# %%
