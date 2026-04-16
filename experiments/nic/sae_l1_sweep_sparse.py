# %%
"""SAE L1 sweep: Trained AE vs Trained AE (unit norm) vs Constructed AE.

Trains three base models, then sweeps SAE L1 coefficients on each.
Plots F1 score vs mean L0 sparsity for all runs.
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions import SparseUniform
from occhio.toy_model import ToyModel

# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 500
D_HIDDEN = 64
N_EPOCHS = 30_000

N_EPOCHS_SYNTH = 15_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14

# SAE sweep config
L1_VALUES = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
L0_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
N_DICT = N_FEATURES // 2
SAE_STEPS = 25_000
N_SEEDS = 5
SAE_BATCH = 1024
SAE_LR = 3e-4
DET_SAMPLES = 50_000

high = 0.3
low = 1.28 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]

# %%
dist = SparseUniform(
    N_FEATURES,
    firing_probs,
    device=DEVICE,
)


# %%
# --- Unit-norm init hook ---
def normalize_W(tm):
    with torch.no_grad():
        tm.ae.W.data /= tm.ae.W.data.norm(dim=0, keepdim=True).clamp(min=1e-8)


# %%
# --- Train Trained AE ---
print("Training Trained AE...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
ae_trained = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1)
tm_trained = ToyModel(distribution=dist, ae=ae_trained, device=DEVICE)
tm_trained.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)
print("  Done.")

# %%
# --- Train Trained AE (unit norm) ---
print("Training Trained AE (unit norm)...")
gen2 = torch.Generator(DEVICE).manual_seed(SEED)
ae_trained_normed = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen2)
tm_trained_normed = ToyModel(
    distribution=dist, ae=ae_trained_normed, device=DEVICE, hooks=[normalize_W]
)
tm_trained_normed.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)
print("  Done.")

# %%
# --- Train Constructed AE ---
print("Training Constructed AE (orthogonalized, bias only)...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_constructed = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_constructed = ToyModel(distribution=dist, ae=ae_constructed, device=DEVICE)
tm_constructed.fit(N_EPOCHS_SYNTH, batch_size=BATCH_SIZE, verbose=True)
print("  Done.")


# %%
# --- Helpers ---
def make_data_fn(tm_ref, device):
    def data_fn(n: int) -> torch.Tensor:
        x = tm_ref.distribution.sample(n).to(device)
        return tm_ref.ae.encode(x)

    return data_fn


# %%
# --- L1 sweep with multi-seed averaging ---
METRIC_KEYS = [
    "f1",
    "precision",
    "recall",
    "l0",
    "r2",
    "mcc",
    "purity",
    "enc_precision",
    "enc_recall",
    "enc_f1",
    "enc_mcc",
]

base_models = [
    ("Trained AE", tm_trained),
    ("Trained AE w/ Unit Norms", tm_trained_normed),
    ("Constructed AE", tm_constructed),
]


def eval_sae(sae, tm):
    """Compute all metrics for a trained SAE against a base ToyModel."""
    with torch.no_grad():
        eye = torch.eye(tm.ae.n_features, device=DEVICE)
        D_enc = tm.ae.encode(eye)
        D_enc_normed = D_enc / D_enc.norm(dim=1, keepdim=True)

        W_dec = sae.W_dec.data
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)

        cos_sim = (D_enc_normed @ W_dec_normed.T).abs().cpu().numpy()
        feat_idx, dict_idx = linear_sum_assignment(-cos_sim)

        det_x = dist.sample(DET_SAMPLES).to(DEVICE)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)
        det_recon = sae.decode(det_z)

        l0 = (det_z > 0).float().sum(dim=-1).mean().item()

        gt_active = det_x[:, feat_idx] > 0
        pred_active = det_z[:, dict_idx] > 0
        tp = (gt_active & pred_active).float().sum(dim=0)
        fp = (~gt_active & pred_active).float().sum(dim=0)
        fn = (gt_active & ~pred_active).float().sum(dim=0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_per = 2 * prec * rec / (prec + rec + 1e-8)

        ss_res = ((det_hidden - det_recon) ** 2).sum().item()
        ss_tot = ((det_hidden - det_hidden.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        W_ae = tm.W.detach()
        W_dec_t = sae.W_dec.detach().T
        W_norm = W_ae / W_ae.norm(dim=0, keepdim=True).clamp(min=1e-8)
        Wd_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_mcc = (W_norm.T @ Wd_norm).cpu().numpy()
        cos_mcc_abs = np.abs(cos_mcc)
        mcc_fi, mcc_di = linear_sum_assignment(-cos_mcc_abs)
        mcc = float(cos_mcc_abs[mcc_fi, mcc_di].mean())

        sae_acts = sae.encode(D_enc).cpu().numpy()
        cos_purity = (D_enc_normed @ W_dec_normed.T).cpu().numpy()
        pf_idx, pd_idx = linear_sum_assignment(-cos_purity)
        matched_feats = set(pf_idx)
        matched_dicts = set(pd_idx)
        row_order = list(pf_idx) + [
            f for f in range(N_FEATURES) if f not in matched_feats
        ]
        col_order = list(pd_idx) + [d for d in range(N_DICT) if d not in matched_dicts]
        sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
        n_matched = len(pf_idx)
        diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
        total_sum = sae_acts_matched.sum()
        purity = diag_sum / total_sum if total_sum > 0 else 0.0

        # --- Encoder-based matching: match by maximizing SAE activations on one-hots ---
        sae_acts_raw = sae.encode(D_enc).cpu().numpy()  # (N_FEATURES, N_DICT)
        enc_feat_idx, enc_dict_idx = linear_sum_assignment(-sae_acts_raw)

        # Encoder MCC: cosine similarity of encoder-matched pairs
        enc_mcc = float(cos_mcc_abs[enc_feat_idx, enc_dict_idx].mean())

        # Encoder detection metrics
        enc_gt_active = det_x[:, enc_feat_idx] > 0
        enc_pred_active = det_z[:, enc_dict_idx] > 0
        enc_tp = (enc_gt_active & enc_pred_active).float().sum(dim=0)
        enc_fp = (~enc_gt_active & enc_pred_active).float().sum(dim=0)
        enc_fn = (enc_gt_active & ~enc_pred_active).float().sum(dim=0)
        enc_prec = enc_tp / (enc_tp + enc_fp + 1e-8)
        enc_rec = enc_tp / (enc_tp + enc_fn + 1e-8)
        enc_f1 = 2 * enc_prec * enc_rec / (enc_prec + enc_rec + 1e-8)

    return {
        "f1": f1_per.mean().item(),
        "precision": prec.mean().item(),
        "recall": rec.mean().item(),
        "l0": l0,
        "r2": r2,
        "mcc": mcc,
        "purity": purity,
        "enc_precision": enc_prec.mean().item(),
        "enc_recall": enc_rec.mean().item(),
        "enc_f1": enc_f1.mean().item(),
        "enc_mcc": enc_mcc,
    }


# Collect per-seed metrics: sweep_raw[name][l1_idx][seed] = {metric: value}
sweep_raw: dict[str, list[list[dict]]] = {name: [] for name, _ in base_models}

for li, l1_coef in enumerate(L1_VALUES):
    for name, tm in base_models:
        if li == 0:
            sweep_raw[name] = []
        sweep_raw[name].append([])

        for seed_i in range(N_SEEDS):
            print(f"  {name} L1={l1_coef} seed={seed_i}...", end=" ", flush=True)
            sae = SAESimple(
                n_latent=D_HIDDEN,
                n_dict=N_DICT,
                l1_coef=l1_coef,
                device=DEVICE,
            ).to(DEVICE)
            sae.train_sae(
                data_fn=make_data_fn(tm, DEVICE),
                n_steps=SAE_STEPS,
                batch_size=SAE_BATCH,
                lr=SAE_LR,
            )
            metrics = eval_sae(sae, tm)
            sweep_raw[name][li].append(metrics)
            print(f"F1={metrics['f1']:.4f}  L0={metrics['l0']:.1f}")

# %%
# --- Aggregate across seeds (mean ± std) ---
sweep_results: dict[str, dict] = {}
for name, _ in base_models:
    res = {"l1": list(L1_VALUES)}
    for key in METRIC_KEYS:
        vals = np.array([[m[key] for m in seeds] for seeds in sweep_raw[name]])
        res[key] = vals.mean(axis=1).tolist()
        res[f"{key}_std"] = vals.std(axis=1).tolist()
    sweep_results[name] = res

for name in sweep_results:
    res = sweep_results[name]
    for i, l1 in enumerate(res["l1"]):
        print(
            f"  {name:25s}  L1={l1:.2f}"
            f"  F1={res['f1'][i]:.4f}±{res['f1_std'][i]:.4f}"
            f"  L0={res['l0'][i]:.1f}±{res['l0_std'][i]:.1f}"
            f"  R²={res['r2'][i]:.4f}±{res['r2_std'][i]:.4f}"
        )

# %%
# =============================================================================
#  VISUALIZATION — run from here to re-plot without retraining
# =============================================================================

# --- Rename keys if sweep_results used old naming convention ---
_RENAME = {
    "Trained": "Trained AE",
    "Trained (unit norm)": "Trained AE w/ Unit Norms",
    "Constructed": "Constructed AE",
}
sweep_results = {_RENAME.get(k, k): v for k, v in sweep_results.items()}

# --- Publication-ready figure styling (matches synth_v_trained_sparse.py) ---
MODEL_COLORS = {
    "Trained AE": "#000c7a",
    "Constructed AE": "#fcba03",
    "Trained AE w/ Unit Norms": "#DC2626",
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


def _add_band(fig, x, y_mean, y_std, color, name, row=None, col=None):
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
# --- Plot: F1, R², MCC vs L0 (combined horizontal) ---
true_mean_l0 = sum(firing_probs)

_MAIN_METRICS = [
    ("f1", '<span style="font-style:italic;">F</span><sub>1</sub>-score'),
    ("r2", '<span style="font-style:italic;">R</span><sup>2</sup>'),
    ("mcc", "MCC"),
]
_L0_DASH = "15px 10px"
_L0_COLOR = "#6B7280"
_N_INTERVALS = 5  # both axes: 5 intervals → square grid cells


def _nice_dtick(y_max, n=_N_INTERVALS):
    """Pick smallest nice step so that n intervals cover y_max."""
    raw = y_max / n
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if step >= raw:
            return step
    return raw


fig_main = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.10,
)

_y_dticks = []
for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    for name, res in sweep_results.items():
        color = MODEL_COLORS[name]
        x = np.array(res["l0"])
        y = np.array(res[mk])
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        y_std_s = np.array(res[f"{mk}_std"])[order]

        _add_band(fig_main, x_s, y_s, y_std_s, color, name, row=1, col=ci)
        fig_main.add_trace(
            go.Scatter(
                x=x_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=(ci == 1),
                marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=2.5),
            ),
            row=1,
            col=ci,
        )

    # vline per panel (no annotation — label is in legend)
    fig_main.add_vline(
        x=true_mean_l0,
        line_dash=_L0_DASH,
        line_color=_L0_COLOR,
        line_width=1.5,
        row=1,
        col=ci,
    )

    # compute nice y dtick for exactly 5 intervals
    _all_y = [v for res in sweep_results.values() for v in res[mk]]
    _all_std = [v for res in sweep_results.values() for v in res[f"{mk}_std"]]
    _y_raw_max = max(yv + s for yv, s in zip(_all_y, _all_std)) * 1.05
    _y_dticks.append(_nice_dtick(_y_raw_max))

# Dummy trace for True L0 legend entry
fig_main.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="True <i>L</i><sup>0</sup>",
        line=dict(color=_L0_COLOR, width=1.5, dash=_L0_DASH),
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig_main.update_shapes(layer="below")

# --- Dimensions for square grid cells ---
# X: range [0, 25], dtick=5 → 5 intervals
# Y: 5 intervals (forced via _nice_dtick)
# → need each panel to be a perfect square in pixel space.
# Solve: panel_px = plot_height; width chosen so 3 panels + spacing + margins fit.
_plot_h = 640  # target plot-area height (px)
_margin = dict(l=100, r=50, t=110, b=100)
_hs = 0.10  # horizontal_spacing fraction
# panel_px = _plot_h → total_width = (3*_plot_h + margin_l + margin_r) / (1 - 2*_hs)
_fig_w = int((3 * _plot_h + _margin["l"] + _margin["r"]) / (1 - 2 * _hs))
_fig_h = _plot_h + _margin["t"] + _margin["b"]

fig_main.update_layout(width=_fig_w, height=_fig_h, margin=_margin)
style_fig(fig_main)

# Per-panel axis overrides (after style_fig so dtick wins over nticks)
for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    _dt = _y_dticks[ci - 1]
    fig_main.update_yaxes(
        title_text=ylabel,
        range=[0, _N_INTERVALS * _dt],
        dtick=_dt,
        row=1,
        col=ci,
    )
    fig_main.update_xaxes(range=[0, 25], dtick=5, row=1, col=ci)

# x-axis title on middle panel only
fig_main.update_xaxes(
    title_text='<span style="font-family:Times New Roman; font-style:italic;">L</span><sup>0</sup><sub>SAE</sub>',
    row=1,
    col=2,
)

# Legend: full width across top, no box, equidistant entries
# Each of 4 entries gets _fig_w*0.9/4 px → equal start-to-start spacing across 90% of width
fig_main.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.06,
        yanchor="bottom",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=39),
        itemsizing="constant",
        itemwidth=50,
        entrywidthmode="pixels",
        entrywidth=int(_fig_w * 0.9 / 4),
    ),
)
fig_main.show()

# %%
# --- Plot: Other metrics vs L0 ---
_METRIC_PLOTS = {
    "enc_precision": "Encoder Precision",
    "enc_recall": "Encoder Recall",
    "enc_f1": 'Encoder <span style="font-style:italic;">F</span><sub>1</sub>',
    "enc_mcc": "Encoder MCC",
    "purity": "Purity",
}

metric_figs = {}
for _mk, _ml in _METRIC_PLOTS.items():
    _fig = go.Figure()
    for name, res in sweep_results.items():
        color = MODEL_COLORS[name]
        x = np.array(res["l0"])
        y = np.array(res[_mk])
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        y_std_s = np.array(res[f"{_mk}_std"])[order]

        _add_band(_fig, x_s, y_s, y_std_s, color, name)
        _fig.add_trace(
            go.Scatter(
                x=x_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=name,
                marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=2.5),
            )
        )

    _fig.add_vline(
        x=true_mean_l0,
        line_dash=_L0_DASH,
        line_color=_L0_COLOR,
        line_width=1.5,
    )
    _fig.update_shapes(layer="below")

    _all_y = [v for res in sweep_results.values() for v in res[_mk]]
    _all_std = [v for res in sweep_results.values() for v in res[f"{_mk}_std"]]
    _y_max = max(yv + s for yv, s in zip(_all_y, _all_std)) * 1.15
    _y_max = min(_y_max, 1.05)

    _fig.update_layout(
        xaxis_title='<span style="font-family:Times New Roman; font-style:italic;">L</span><sup>0</sup><sub>SAE</sub>',
        yaxis_title=_ml,
        xaxis_range=[0, 25],
        xaxis_dtick=5,
        yaxis_range=[0, _y_max],
        width=1000,
        height=1000,
        margin=dict(l=100, r=100, t=100, b=100),
    )
    style_fig(_fig)
    _fig.show()
    metric_figs[_mk] = _fig

# %%
# --- Plot: Precision & Recall vs L1 (stacked subplots, shared x-axis) ---
fig_pr = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=["Precision", "Recall"],
)

for name, res in sweep_results.items():
    color = MODEL_COLORS[name]
    l1_arr = np.array(res["l1"])
    order = np.argsort(l1_arr)
    l1_s = l1_arr[order]

    for ri, (metric, metric_std) in enumerate(
        [("precision", "precision_std"), ("recall", "recall_std")], start=1
    ):
        y_s = np.array(res[metric])[order]
        y_std_s = np.array(res[metric_std])[order]

        _add_band(fig_pr, l1_s, y_s, y_std_s, color, name, row=ri, col=1)
        fig_pr.add_trace(
            go.Scatter(
                x=l1_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=(ri == 2),
                marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=2.5, dash="dash"),
            ),
            row=ri,
            col=1,
        )

# Shared x-axis label on bottom only
fig_pr.update_xaxes(title_text="SAE L1 Coefficient", row=2, col=1)
fig_pr.update_yaxes(title_text="Precision", row=1, col=1)
fig_pr.update_yaxes(title_text="Recall", row=2, col=1)

# Style subplot titles
for ann in fig_pr.layout.annotations:
    ann.font = dict(size=27)

fig_pr.update_layout(
    width=1000,
    height=1200,
    margin=dict(l=100, r=100, t=100, b=100),
)
style_fig(fig_pr)
fig_pr.show()

# %%
# --- Print summary table (decoder matching) ---
print(
    f"\n{'Model':25s}  {'L1':>6s}  {'F1':>15s}  {'Prec':>15s}  {'Recall':>15s}"
    f"  {'L0':>12s}  {'R²':>15s}  {'MCC':>15s}"
)
print("-" * 140)
for name, res in sweep_results.items():
    for i, l1 in enumerate(res["l1"]):
        print(
            f"{name:25s}  {l1:6.2f}"
            f"  {res['f1'][i]:6.4f}±{res['f1_std'][i]:.4f}"
            f"  {res['precision'][i]:6.4f}±{res['precision_std'][i]:.4f}"
            f"  {res['recall'][i]:6.4f}±{res['recall_std'][i]:.4f}"
            f"  {res['l0'][i]:5.1f}±{res['l0_std'][i]:.1f}"
            f"  {res['r2'][i]:6.4f}±{res['r2_std'][i]:.4f}"
            f"  {res['mcc'][i]:6.4f}±{res['mcc_std'][i]:.4f}"
        )

# %%
# --- Print summary table (encoder matching) ---
print(
    f"\n{'Model':25s}  {'L1':>6s}  {'EncF1':>15s}  {'EncPrec':>15s}  {'EncRecall':>15s}"
    f"  {'EncMCC':>15s}"
)
print("-" * 100)
for name, res in sweep_results.items():
    for i, l1 in enumerate(res["l1"]):
        print(
            f"{name:25s}  {l1:6.2f}"
            f"  {res['enc_f1'][i]:6.4f}±{res['enc_f1_std'][i]:.4f}"
            f"  {res['enc_precision'][i]:6.4f}±{res['enc_precision_std'][i]:.4f}"
            f"  {res['enc_recall'][i]:6.4f}±{res['enc_recall_std'][i]:.4f}"
            f"  {res['enc_mcc'][i]:6.4f}±{res['enc_mcc_std'][i]:.4f}"
        )

# %%
# --- W^T W heatmaps ---
_wtw_models = [
    ("Trained AE", tm_trained),
    ("Trained AE w/ Unit Norms", tm_trained_normed),
]
for _wtw_name, _wtw_tm in _wtw_models:
    with torch.no_grad():
        _W = _wtw_tm.W.detach().cpu().numpy()
        _WtW = _W.T @ _W
    _fig_wtw = go.Figure(
        go.Heatmap(z=_WtW, colorscale="RdBu_r", zmid=0, showscale=True)
    )
    _fig_wtw.update_layout(
        title=f"W<sup>T</sup>W — {_wtw_name}",
        xaxis_title="Feature j",
        yaxis_title="Feature i",
        width=600,
        height=550,
        margin=dict(l=60, r=30, t=50, b=60),
    )
    style_fig(_fig_wtw)
    _fig_wtw.show()

# %%
# --- Save figures as vector (PDF + SVG) ---
_fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(_fig_dir, exist_ok=True)

fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.pdf"), engine="kaleido")
fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.svg"), engine="kaleido")
fig_pr.write_image(os.path.join(_fig_dir, "sae_prec_vs_recall.pdf"), engine="kaleido")
fig_pr.write_image(os.path.join(_fig_dir, "sae_prec_vs_recall.svg"), engine="kaleido")
for _mk, _mfig in metric_figs.items():
    _mfig.write_image(os.path.join(_fig_dir, f"sae_{_mk}_vs_l0.pdf"), engine="kaleido")
    _mfig.write_image(os.path.join(_fig_dir, f"sae_{_mk}_vs_l0.svg"), engine="kaleido")
print(f"Saved to {_fig_dir}/")

# %%
