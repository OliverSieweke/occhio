# %%
"""Benchmark SAE absorption rates across architectures on HierarchicalPairs.

Loads ``csv/absorption_hierarchical_pairs.csv`` and plots absorption_rate vs.
SAE L0 for each architecture.
"""

import os

import pandas as pd
import plotly.graph_objects as go

# %%
# --- Load data ---
_csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
    "csv",
    "absorption_hierarchical_pairs.csv",
)
df = pd.read_csv(_csv_path)

# Rename Standard → ReLU, and drop ReLU points outside the L0 ∈ [2, 10] window
# (the ends are noisy / out of the regime we want to compare).
df["arch"] = df["arch"].replace({"Standard": "ReLU", "MatchingPursuit": "MP-SAE"})
_relu_mask = df["arch"] == "ReLU"
df = df[~_relu_mask | df["sae_l0"].between(2, 11)].reset_index(drop=True)
print(df.head())

# %%
# --- Style (matches synth_v_trained_sparse.py) ---
ARCH_COLORS = {
    "ReLU": "#000c7a",
    "BatchTopK": "#DC2626",
    "Matryoshka": "#297a58",
    "MP-SAE": "#fcba03",
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

# True underlying L0 marker — added first so it sits behind the data traces,
# but above gridlines (traces always render above the gridlines).
fig.add_trace(
    go.Scatter(
        x=[8, 8],
        y=[0, 1],
        mode="lines",
        line=dict(color="#9CA3AF", width=2.5, dash="15px,10px"),
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
            y=_sub["absorption_rate"].to_numpy(),
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
    title_text="Absorption Rate",
    **_AXIS,
    range=[0, 1],
    dtick=0.2,
    minor=dict(ticks="", showgrid=True, gridcolor="#F3F4F6", dtick=0.05),
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
fig.write_image(os.path.join(_fig_dir, "benchmark_absorption.pdf"), engine="kaleido")
print(f"Saved to {_fig_dir}/benchmark_absorption.pdf")

fig.show()

# %%
