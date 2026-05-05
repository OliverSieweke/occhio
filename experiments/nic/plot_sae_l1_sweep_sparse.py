# %%
"""Replot script for SAE L1 sweep (F1, MCC, R² vs L0).

Uses hardcoded results — no training required.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# --- Hardcoded sweep results ---
sweep_results = {
    "Trained AE": {
        "target_l0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "l0": [1.0, 2.0, 3.0, 4.0, 5.0, 5.99, 6.98, 8.02, 9.0, 10.02, 11.08, 11.99],
        "f1": [
            0.4769,
            0.6639,
            0.6325,
            0.5614,
            0.5029,
            0.4642,
            0.4273,
            0.4014,
            0.3779,
            0.3587,
            0.3423,
            0.3285,
        ],
        "f1_std": [
            0.0038,
            0.0062,
            0.0082,
            0.0034,
            0.0055,
            0.0063,
            0.0051,
            0.0052,
            0.0037,
            0.0046,
            0.0038,
            0.0037,
        ],
        "mcc": [
            0.9032,
            0.9341,
            0.9465,
            0.9485,
            0.9428,
            0.9333,
            0.9192,
            0.9131,
            0.9017,
            0.8982,
            0.8915,
            0.885,
        ],
        "mcc_std": [
            0.0031,
            0.0043,
            0.0075,
            0.0051,
            0.0079,
            0.0059,
            0.0082,
            0.0054,
            0.0056,
            0.0029,
            0.0037,
            0.0039,
        ],
        "r2": [
            0.1644,
            0.4301,
            0.5958,
            0.6579,
            0.6947,
            0.7189,
            0.7395,
            0.7556,
            0.7702,
            0.7834,
            0.7947,
            0.805,
        ],
        "r2_std": [
            0.0017,
            0.0029,
            0.0023,
            0.0023,
            0.0037,
            0.0018,
            0.0035,
            0.0026,
            0.0026,
            0.0012,
            0.0009,
            0.0014,
        ],
    },
    "Trained AE w/ Unit Norms": {
        "target_l0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "l0": [1.0, 2.0, 3.0, 4.02, 5.02, 6.01, 7.0, 7.97, 9.02, 9.98, 10.99, 12.0],
        "f1": [
            0.3842,
            0.5943,
            0.5929,
            0.5446,
            0.4831,
            0.4465,
            0.408,
            0.3927,
            0.3652,
            0.3523,
            0.3385,
            0.3219,
        ],
        "f1_std": [
            0.006,
            0.0173,
            0.008,
            0.0052,
            0.007,
            0.0098,
            0.0072,
            0.0034,
            0.0062,
            0.0036,
            0.007,
            0.006,
        ],
        "mcc": [
            0.8456,
            0.8849,
            0.9128,
            0.9279,
            0.917,
            0.9104,
            0.8965,
            0.8988,
            0.8895,
            0.888,
            0.8803,
            0.8761,
        ],
        "mcc_std": [
            0.0106,
            0.0171,
            0.0087,
            0.0043,
            0.0085,
            0.008,
            0.0075,
            0.0059,
            0.0088,
            0.0045,
            0.0093,
            0.005,
        ],
        "r2": [
            0.1593,
            0.4644,
            0.6272,
            0.6895,
            0.7249,
            0.7463,
            0.7627,
            0.7824,
            0.7941,
            0.8064,
            0.8169,
            0.8247,
        ],
        "r2_std": [
            0.0003,
            0.0024,
            0.0045,
            0.0052,
            0.0037,
            0.003,
            0.0031,
            0.0016,
            0.0024,
            0.0025,
            0.0015,
            0.0016,
        ],
    },
    "Constructed AE": {
        "target_l0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "l0": [1.0, 2.0, 3.0, 3.95, 4.96, 6.0, 7.0, 8.0, 9.02, 10.03, 11.05, 11.96],
        "f1": [
            0.3556,
            0.46,
            0.4236,
            0.3869,
            0.3571,
            0.3313,
            0.308,
            0.2875,
            0.2723,
            0.2604,
            0.2503,
            0.2428,
        ],
        "f1_std": [
            0.0105,
            0.0113,
            0.0107,
            0.0115,
            0.0101,
            0.009,
            0.0068,
            0.012,
            0.0031,
            0.0062,
            0.007,
            0.0051,
        ],
        "mcc": [
            0.7774,
            0.7998,
            0.7974,
            0.806,
            0.8178,
            0.8244,
            0.8302,
            0.8273,
            0.8365,
            0.8359,
            0.8458,
            0.8475,
        ],
        "mcc_std": [
            0.0116,
            0.0105,
            0.0114,
            0.0104,
            0.0118,
            0.01,
            0.0091,
            0.0172,
            0.0051,
            0.0104,
            0.0136,
            0.0067,
        ],
        "r2": [
            0.1531,
            0.3779,
            0.5071,
            0.5818,
            0.6343,
            0.6703,
            0.6979,
            0.7174,
            0.7384,
            0.7519,
            0.768,
            0.7789,
        ],
        "r2_std": [
            0.0019,
            0.003,
            0.0038,
            0.0031,
            0.0035,
            0.0039,
            0.0042,
            0.0054,
            0.0015,
            0.0022,
            0.0044,
            0.0026,
        ],
    },
}

# Drop the last data point (L0≈12) from each SAE
for _name in sweep_results:
    for _key in list(sweep_results[_name]):
        if isinstance(sweep_results[_name][_key], list):
            sweep_results[_name][_key] = sweep_results[_name][_key][:-1]

# %%
# --- Config ---
N_FEATURES = 500
high = 0.3
low = 1.28 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
true_mean_l0 = sum(firing_probs)

# %%
# --- Styling ---
MODEL_COLORS = {
    "Trained AE": "#000c7a",
    "Constructed AE": "#fcba03",
    "Trained AE w/ Unit Norms": "#DC2626",
}

_LEGEND_NAMES = {
    "Trained AE": "ReLU SAE(Trained AE)",
    "Trained AE w/ Unit Norms": "ReLU SAE(Trained AE w/ Unit Norms)",
    "Constructed AE": "ReLU SAE(Constructed AE)",
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


def _hex_to_rgba(hex_color, alpha):
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_band(fig, x, y_mean, y_std, color, row=None, col=None):
    """Add a ±1 std shaded band (fully invisible on hover)."""
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)
    x_band = list(x) + list(reversed(x))
    y_band = list(y_mean + y_std) + list(reversed(y_mean - y_std))
    kw = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.15),
            line=dict(width=0),
            mode="none",
            showlegend=False,
            hoverinfo="skip",
        ),
        **kw,
    )


# %%
# --- Plot: F1, MCC, R² vs L0 (combined horizontal) ---
_MAIN_METRICS = [
    ("f1", '<span style="font-style:italic;">F</span><sub>1</sub>'),
    ("mcc", "MCC"),
    ("r2", '<span style="font-style:italic;">R</span><sup>2</sup>'),
]
_L0_DASH = "15px 10px"

fig_main = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.13,
)

_LEG_PAD = "&nbsp;" * 8

for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    # E[L0] vline as a trace (behind data, above gridlines)
    fig_main.add_trace(
        go.Scatter(
            x=[true_mean_l0, true_mean_l0],
            y=[-10, 10],
            mode="lines",
            line=dict(color="#9CA3AF", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=ci,
    )

    for name, res in reversed(list(sweep_results.items())):
        color = MODEL_COLORS[name]
        x = np.array(res["l0"])
        y = np.array(res[mk])
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        y_std_s = np.array(res[f"{mk}_std"])[order]

        _add_band(fig_main, x_s, y_s, y_std_s, color, row=1, col=ci)
        fig_main.add_trace(
            go.Scatter(
                x=x_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=_LEGEND_NAMES[name] + _LEG_PAD,
                legendgroup=name,
                showlegend=(ci == 1),
                marker=dict(size=10, color=color, line=dict(width=3, color="white")),
                line=dict(color=color, width=2.5),
            ),
            row=1,
            col=ci,
        )

# --- Dimensions (square panels) ---
_fs = 38
_fs_tick = 30

_plot_h = 400
_margin = dict(l=100, r=20, t=60, b=120)
_hs = 0.13

_fig_w = int(3 * _plot_h / (1 - 2 * _hs)) + _margin["l"] + _margin["r"]
_fig_h = _plot_h + _margin["t"] + _margin["b"]

fig_main.update_layout(width=_fig_w, height=_fig_h, margin=_margin)
style_fig(fig_main)

# Lock margins + override font sizes
fig_main.update_xaxes(
    tickfont=dict(size=_fs_tick),
    title_font=dict(size=_fs),
    title_standoff=12,
    automargin=False,
)
fig_main.update_yaxes(
    tickfont=dict(size=_fs_tick),
    title_font=dict(size=_fs),
    title_standoff=0,
    automargin=False,
)

# Data-driven x-range (shared across panels)
_all_l0 = [v for res in sweep_results.values() for v in res["l0"]]
_x_lo = 0
_x_hi = 11.8

# Per-panel axis overrides
for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    _all_y = [v for res in sweep_results.values() for v in res[mk]]
    _all_std = [v for res in sweep_results.values() for v in res[f"{mk}_std"]]
    _y_lo = max(0, min(yv - s for yv, s in zip(_all_y, _all_std)) - 0.05)
    _y_hi = max(yv + s for yv, s in zip(_all_y, _all_std)) + 0.05
    fig_main.update_yaxes(
        title_text=ylabel,
        range=[_y_lo, _y_hi],
        dtick=0.25,
        tickangle=-90,
        minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )
    fig_main.update_xaxes(
        range=[_x_lo, _x_hi],
        dtick=5,
        minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )

# First subplot (F1): extend upper bound to 0.8
_f1_vals = [v for res in sweep_results.values() for v in res["f1"]]
_f1_stds = [v for res in sweep_results.values() for v in res["f1_std"]]
_f1_lo = max(0, min(yv - s for yv, s in zip(_f1_vals, _f1_stds)) - 0.05)
fig_main.update_yaxes(range=[_f1_lo, 0.8], row=1, col=1)

# x-axis title on middle panel only
fig_main.update_xaxes(
    title_text='<span style="font-family:Times New Roman; font-style:italic;">L</span><sup>0</sup><sub>SAE</sub>',
    row=1,
    col=2,
)

# Legend
fig_main.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.02,
        yanchor="bottom",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=_fs_tick),
        itemsizing="constant",
        itemwidth=50,
    ),
)
fig_main.show()

# %%
# --- Save as vector (PDF + SVG) ---
_fig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
    "figures",
)
os.makedirs(_fig_dir, exist_ok=True)

fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.pdf"), engine="kaleido")
fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.svg"), engine="kaleido")
print(f"Saved to {_fig_dir}/")

# %%
