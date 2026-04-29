# %%
"""Benchmark SAE splitting quality across architectures.

Loads ``csv/plot_splitting_quality_vs_l0.csv`` and plots splitting_quality vs.
SAE L0 for each architecture on a log y-axis.
"""

import os

import pandas as pd
import plotly.graph_objects as go

# %%
# --- Load data ---
_csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
    "csv",
    "plot_splitting_quality_vs_l0.csv",
)
df = pd.read_csv(_csv_path)

# Rename Standard → ReLU, and drop ReLU points outside the L0 ∈ [2, 11] window
# (the ends are noisy / out of the regime we want to compare).
df["arch"] = df["arch"].replace({"Standard": "ReLU", "MatchingPursuit": "MP-SAE"})
_relu_mask = df["arch"] == "ReLU"
df = df[~_relu_mask | df["sae_l0"].between(2, 11)].reset_index(drop=True)

# Floor zeros so they render at the bottom of the log axis (relabeled as "0").
_ZERO_FLOOR = 1e-4
df["splitting_quality"] = df["splitting_quality"].clip(lower=_ZERO_FLOOR)
print(df.head())

# %%
# --- Style (matches benchmark_absorption.py) ---
ARCH_COLORS = {
    "ReLU": "#fcba03",
    "BatchTopK": "#000c7a",
    "Matryoshka": "#DC2626",
    "MP-SAE": "#297a58",
}

_AXIS = dict(
    showgrid=True,
    gridcolor="#E5E7EB",
    showline=True,
    linecolor="black",
    linewidth=1.5,
    ticks="outside",
    ticklen=8,
    tickwidth=1.5,
    tickcolor="black",
    zeroline=False,
    tickfont=dict(size=35, color="black"),
    title_font=dict(size=41, color="black"),
    mirror=True,
)

# %%
# --- Plot ---
fig = go.Figure()

# True underlying L0 marker — added first so it sits behind the data traces.
fig.add_trace(
    go.Scatter(
        x=[8, 8],
        y=[1e-4, 1],
        mode="lines",
        line=dict(color="#9CA3AF", width=2.5),
        showlegend=False,
        hoverinfo="skip",
    )
)

for _arch, _color in ARCH_COLORS.items():
    _sub = df[df["arch"] == _arch].sort_values("sae_l0")
    if _sub.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=_sub["sae_l0"].to_numpy(),
            y=_sub["splitting_quality"].to_numpy(),
            mode="lines+markers",
            name=_arch,
            line=dict(color=_color, width=3),
            marker=dict(size=14, color=_color, line=dict(width=2, color="white")),
        )
    )

fig.update_xaxes(
    title_text="SAE <i>L</i><sup>0</sup>",
    **_AXIS,
    nticks=8,
)
fig.update_yaxes(
    title_text="Splitting Quality",
    **_AXIS,
    type="log",
    range=[-4, 0],
    tickvals=[1e-4, 1e-3, 1e-2, 1e-1, 1],
    ticktext=["0", "0.001", "0.01", "0.1", "1"],
    minor=dict(ticks="", showgrid=True, gridcolor="#F3F4F6"),
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Times New Roman, Times, serif", size=41, color="black"),
    width=900,
    height=600,
    margin=dict(l=100, r=40, t=60, b=80),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        itemsizing="constant",
        itemwidth=50,
        tracegroupgap=50,
        font=dict(size=28),
    ),
)

# %%
# --- Save as vector ---
_fig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
    "figures",
)
os.makedirs(_fig_dir, exist_ok=True)
fig.write_image(
    os.path.join(_fig_dir, "benchmark_splitting_quality.pdf"), engine="kaleido"
)
print(f"Saved to {_fig_dir}/benchmark_splitting_quality.pdf")

fig.show()

# %%
