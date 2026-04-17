# %%
"""SAE L1 Sweep — Post-Training Analysis.

Load checkpoint from sae_l1_sweep_sparse.py and explore all metrics
without retraining. Run cells interactively.
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy import stats

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions import SparseUniform

# %%
# --- Load checkpoint ---
_here = (
    Path(os.path.dirname(os.path.abspath(__file__)))
    if "__file__" in dir()
    else Path.cwd()
)
ckpt = torch.load(
    _here / "checkpoints" / "sae_sweep_checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)

sweep_raw = ckpt["sweep_raw"]
sweep_results = ckpt["sweep_results"]
cfg = ckpt["config"]
sae_states = ckpt["sae_states"]
base_states = ckpt["base_states"]

N_FEATURES = cfg["N_FEATURES"]
D_HIDDEN = cfg["D_HIDDEN"]
N_DICT = cfg["N_DICT"]
L0_VALUES = cfg["L0_VALUES"]
N_SEEDS = cfg["N_SEEDS"]
firing_probs = np.array(cfg["firing_probs"])
DET_SAMPLES = cfg["DET_SAMPLES"]
SEED = cfg["SEED"]
DEVICE = "cpu"

MODEL_NAMES = list(sweep_results.keys())
METRIC_KEYS = list(sweep_raw[MODEL_NAMES[0]][0][0].keys())
true_mean_l0 = float(firing_probs.sum())

print(
    f"Loaded: {len(MODEL_NAMES)} models × {len(L0_VALUES)} L0 values × {N_SEEDS} seeds"
)
print(f"Models: {MODEL_NAMES}")
print(f"Metrics: {METRIC_KEYS}")
print(f"True mean L0 = {true_mean_l0:.2f}")

# %%
# --- Styling (matches sae_l1_sweep_sparse.py, scaled down for exploration) ---
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
    linewidth=2,
    mirror=True,
    ticks="outside",
    ticklen=6,
    tickwidth=1.5,
    tickcolor="black",
    minor=dict(ticks="", showgrid=False),
    zeroline=False,
    tickfont=dict(size=14, color="black"),
    title_font=dict(size=16, color="black"),
)


def style_fig(fig, nticks=6):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            font=dict(size=13),
        ),
    )
    fig.update_xaxes(**_AXIS, nticks=nticks)
    fig.update_yaxes(**_AXIS, nticks=nticks)
    return fig


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# %%
# =========================================================================
# 1. FULL SUMMARY TABLE
# =========================================================================
print("\n" + "=" * 120)
print("FULL SUMMARY (mean ± std across seeds)")
print("=" * 120)

for name in MODEL_NAMES:
    res = sweep_results[name]
    print(f"\n--- {name} ---")
    print(
        f"  {'tL0':>4s}  {'L0':>10s}  {'F1':>12s}  {'R²':>12s}"
        f"  {'MCC':>12s}  {'Purity':>12s}  {'EncF1':>12s}  {'EncMCC':>12s}"
    )
    print("  " + "-" * 100)
    for i, tl0 in enumerate(res["target_l0"]):
        print(
            f"  {tl0:4d}"
            f"  {res['l0'][i]:5.1f}±{res['l0_std'][i]:<4.1f}"
            f"  {res['f1'][i]:.4f}±{res['f1_std'][i]:.4f}"
            f"  {res['r2'][i]:.4f}±{res['r2_std'][i]:.4f}"
            f"  {res['mcc'][i]:.4f}±{res['mcc_std'][i]:.4f}"
            f"  {res['purity'][i]:.4f}±{res['purity_std'][i]:.4f}"
            f"  {res['enc_f1'][i]:.4f}±{res['enc_f1_std'][i]:.4f}"
            f"  {res['enc_mcc'][i]:.4f}±{res['enc_mcc_std'][i]:.4f}"
        )


# %%
# =========================================================================
# 2. BEST OPERATING POINT PER MODEL
# =========================================================================
print("\n" + "=" * 80)
print("BEST OPERATING POINTS")
print("=" * 80)

for mk, ml in [("f1", "F1"), ("r2", "R²"), ("mcc", "MCC"), ("enc_f1", "Enc-F1")]:
    print(f"\n  Best {ml}:")
    for name in MODEL_NAMES:
        res = sweep_results[name]
        vals = np.array(res[mk])
        bi = int(np.argmax(vals))
        tl0 = res["target_l0"][bi]
        print(
            f"    {name:30s}  target_L0={tl0:2d}"
            f"  {ml}={vals[bi]:.4f}±{res[f'{mk}_std'][bi]:.4f}"
            f"  (L0={res['l0'][bi]:.1f})"
        )


# %%
# =========================================================================
# 3. WINNER AT EACH L0
# =========================================================================
print("\n" + "=" * 100)
print("WINNER AT EACH TARGET L0")
print("=" * 100)

for mk, ml in [("f1", "F1"), ("mcc", "MCC"), ("r2", "R²")]:
    print(f"\n  By {ml}:")
    print(
        f"    {'tL0':>4s}  {'Winner':30s}  {ml:>8s}"
        f"  {'Runner-up':30s}  {ml:>8s}  {'Gap':>8s}"
    )
    print("    " + "-" * 100)
    for i, tl0 in enumerate(L0_VALUES):
        scores = sorted(
            [(name, sweep_results[name][mk][i]) for name in MODEL_NAMES],
            key=lambda x: -x[1],
        )
        w, wv = scores[0]
        r, rv = scores[1]
        print(f"    {tl0:4d}  {w:30s}  {wv:.4f}  {r:30s}  {rv:.4f}  {wv - rv:+.4f}")


# %%
# =========================================================================
# 4. METRIC CORRELATIONS
# =========================================================================
print("\n" + "=" * 80)
print("METRIC CORRELATIONS (Pearson r across all model × L0 combos)")
print("=" * 80)

_corr_metrics = ["f1", "r2", "mcc", "purity", "enc_f1", "enc_mcc", "l0"]
_corr_data = {m: [] for m in _corr_metrics}
for name in MODEL_NAMES:
    for m in _corr_metrics:
        _corr_data[m].extend(sweep_results[name][m])
_corr_data = {m: np.array(v) for m, v in _corr_data.items()}

print(f"\n{'':>10s}", end="")
for m2 in _corr_metrics:
    print(f"  {m2:>8s}", end="")
print()
for m1 in _corr_metrics:
    print(f"{m1:>10s}", end="")
    for m2 in _corr_metrics:
        r, _ = stats.pearsonr(_corr_data[m1], _corr_data[m2])
        print(f"  {r:8.3f}", end="")
    print()


# %%
# =========================================================================
# 5. DECODER vs ENCODER MATCHING
# =========================================================================
print("\n" + "=" * 80)
print("DECODER vs ENCODER MATCHING: F1 gap (dec − enc)")
print("=" * 80)
print(f"{'Model':30s}  {'tL0':>4s}  {'DecF1':>7s}  {'EncF1':>7s}  {'gap':>7s}")
print("-" * 65)

for name in MODEL_NAMES:
    res = sweep_results[name]
    for i, tl0 in enumerate(res["target_l0"]):
        d, e = res["f1"][i], res["enc_f1"][i]
        delta = d - e
        flag = " <--" if abs(delta) > 0.02 else ""
        print(f"{name:30s}  {tl0:4d}  {d:.4f}  {e:.4f}  {delta:+.4f}{flag}")


# %%
# =========================================================================
# 6. SEED VARIANCE ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print("HIGHEST-VARIANCE CONFIGS (F1 std, top 10)")
print("=" * 80)

_var_rows = []
for name in MODEL_NAMES:
    res = sweep_results[name]
    for i, tl0 in enumerate(res["target_l0"]):
        _var_rows.append((res["f1_std"][i], name, tl0, res["f1"][i]))
_var_rows.sort(key=lambda r: -r[0])

print(f"{'Model':30s}  {'tL0':>4s}  {'F1':>8s}  {'std':>8s}  {'CV':>8s}")
for std, name, tl0, f1 in _var_rows[:10]:
    print(f"{name:30s}  {tl0:4d}  {f1:.4f}  {std:.4f}  {std / (f1 + 1e-8):.3f}")


# %%
# =========================================================================
# 7. PRECISION vs RECALL BALANCE
# =========================================================================
print("\n" + "=" * 80)
print("PRECISION vs RECALL BALANCE")
print("=" * 80)

for name in MODEL_NAMES:
    res = sweep_results[name]
    print(f"\n--- {name} ---")
    print(f"  {'tL0':>4s}  {'Prec':>8s}  {'Recall':>8s}  {'Ratio':>8s}  {'Regime'}")
    for i, tl0 in enumerate(res["target_l0"]):
        p, r = res["precision"][i], res["recall"][i]
        ratio = r / (p + 1e-8)
        regime = (
            "balanced"
            if 0.9 < ratio < 1.1
            else "prec-heavy"
            if ratio < 0.9
            else "recall-heavy"
        )
        print(f"  {tl0:4d}  {p:.4f}  {r:.4f}  {ratio:.3f}  {regime}")


# %%
# =========================================================================
# 8. RECONSTRUCT MODELS FOR PER-FEATURE & DEAD NEURON ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print("RECONSTRUCTING MODELS...")
print("=" * 80)

dist = SparseUniform(N_FEATURES, firing_probs.tolist(), device=DEVICE)

_AE_CLASSES = {"TiedLinearRelu": TiedLinearRelu, "SynthAE": SynthAE}
base_aes = {}
for name, info in base_states.items():
    cls = _AE_CLASSES[info["ae_class"]]
    ae = cls(N_FEATURES, D_HIDDEN, device=DEVICE)
    ae.load_state_dict(info["ae_state"])
    ae.eval()
    base_aes[name] = ae
    print(f"  {name}: {info['ae_class']}")

# Shared test data
torch.manual_seed(SEED + 999)
test_x = dist.sample(DET_SAMPLES)
print(f"  Test data: {test_x.shape}")

# Representative L0 (closest to true mean L0)
_rep_l0_idx = int(np.argmin(np.abs(np.array(L0_VALUES) - true_mean_l0)))
_rep_l0 = L0_VALUES[_rep_l0_idx]
print(f"  Representative L0: {_rep_l0} (true mean={true_mean_l0:.1f})")


# %%
# =========================================================================
# 9. PER-FEATURE ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print(f"PER-FEATURE ANALYSIS at target_L0={_rep_l0} (seed 0)")
print("=" * 80)

per_feature_data = {}

for name in MODEL_NAMES:
    ae = base_aes[name]
    sae = SAESimple(n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=0.1, device=DEVICE)
    sae.load_state_dict(sae_states[name][_rep_l0_idx][0])
    sae.eval()

    with torch.no_grad():
        hidden = ae.encode(test_x)
        z = sae.encode(hidden)

        eye = torch.eye(N_FEATURES, device=DEVICE)
        D_enc = ae.encode(eye)
        D_enc_n = D_enc / D_enc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_dec_n = sae.W_dec.data / sae.W_dec.data.norm(dim=1, keepdim=True).clamp(
            min=1e-8
        )
        cos_sim = (D_enc_n @ W_dec_n.T).abs().cpu().numpy()
        feat_idx, dict_idx = linear_sum_assignment(-cos_sim)

        gt = test_x[:, feat_idx] > 0
        pred = z[:, dict_idx] > 0
        tp = (gt & pred).float().sum(dim=0)
        fp = (~gt & pred).float().sum(dim=0)
        fn = (gt & ~pred).float().sum(dim=0)
        prec_pf = (tp / (tp + fp + 1e-8)).numpy()
        rec_pf = (tp / (tp + fn + 1e-8)).numpy()
        f1_pf = 2 * prec_pf * rec_pf / (prec_pf + rec_pf + 1e-8)
        match_cos = np.array(
            [cos_sim[feat_idx[j], dict_idx[j]] for j in range(len(feat_idx))]
        )

    freqs = firing_probs[feat_idx]
    per_feature_data[name] = {
        "feat_idx": feat_idx,
        "dict_idx": dict_idx,
        "f1": f1_pf,
        "precision": prec_pf,
        "recall": rec_pf,
        "match_cos": match_cos,
        "freqs": freqs,
    }

    r_corr, p_corr = stats.spearmanr(freqs, f1_pf)
    print(f"\n--- {name} ---")
    print(f"  Spearman(freq, F1) = {r_corr:.3f} (p={p_corr:.1e})")
    print(f"  Mean match cosine  = {match_cos.mean():.4f}")

    bins = [
        ("high  (>0.10)", freqs > 0.10),
        ("mid   (0.01-0.10)", (freqs >= 0.01) & (freqs <= 0.10)),
        ("low   (<0.01)", freqs < 0.01),
    ]
    print(
        f"  {'Bin':>20s}  {'N':>4s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Cos':>7s}"
    )
    for bl, mask in bins:
        if mask.sum() == 0:
            continue
        print(
            f"  {bl:>20s}  {mask.sum():4d}  {f1_pf[mask].mean():.4f}"
            f"  {prec_pf[mask].mean():.4f}  {rec_pf[mask].mean():.4f}"
            f"  {match_cos[mask].mean():.4f}"
        )

    worst = np.argsort(f1_pf)[:5]
    print("  Worst 5 features:")
    for w in worst:
        print(
            f"    feat={feat_idx[w]:3d}  freq={freqs[w]:.5f}"
            f"  F1={f1_pf[w]:.4f}  cos={match_cos[w]:.4f}"
        )


# %%
# =========================================================================
# 10. DEAD NEURON ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print("DEAD NEURON ANALYSIS")
print("=" * 80)
print(f"{'Model':30s}  {'tL0':>4s}  {'Dead':>8s}  {'%Dead':>7s}  {'Alive':>8s}")
print("-" * 70)

dead_counts = {name: [] for name in MODEL_NAMES}

for name in MODEL_NAMES:
    ae = base_aes[name]
    for li, tl0 in enumerate(L0_VALUES):
        dead_per_seed = []
        for si in range(N_SEEDS):
            sae = SAESimple(
                n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=0.1, device=DEVICE
            )
            sae.load_state_dict(sae_states[name][li][si])
            sae.eval()
            with torch.no_grad():
                z = sae.encode(ae.encode(test_x))
                n_dead = int((z.sum(dim=0) == 0).sum().item())
            dead_per_seed.append(n_dead)
        m, s = np.mean(dead_per_seed), np.std(dead_per_seed)
        dead_counts[name].append((m, s))
        print(
            f"{name:30s}  {tl0:4d}"
            f"  {m:5.1f}+-{s:<4.1f}  {m / N_DICT * 100:5.1f}%  {N_DICT - m:5.1f}"
        )


# %%
# =========================================================================
# 11. STATISTICAL SIGNIFICANCE (paired t-test on F1 across seeds)
# =========================================================================
print("\n" + "=" * 80)
print("PAIRWISE COMPARISONS (paired t-test on F1)")
print("=" * 80)

_pairs = [
    ("Constructed AE", "Trained AE"),
    ("Constructed AE", "Trained AE w/ Unit Norms"),
    ("Trained AE w/ Unit Norms", "Trained AE"),
]

for m1, m2 in _pairs:
    print(f"\n  {m1} vs {m2}:")
    print(f"  {'tL0':>4s}  {'dF1':>9s}  {'t':>7s}  {'p':>9s}  {'sig':>4s}")
    print("  " + "-" * 40)
    for i, tl0 in enumerate(L0_VALUES):
        a = np.array([sweep_raw[m1][i][s]["f1"] for s in range(N_SEEDS)])
        b = np.array([sweep_raw[m2][i][s]["f1"] for s in range(N_SEEDS)])
        t, p = stats.ttest_rel(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {tl0:4d}  {a.mean() - b.mean():+9.4f}  {t:7.2f}  {p:9.4f}  {sig:>4s}")


# %%
# =========================================================================
# PLOTS
# =========================================================================

# %% --- Plot: Feature Frequency vs F1 (scatter, per model) ---
fig_freq = make_subplots(
    rows=1,
    cols=len(MODEL_NAMES),
    subplot_titles=MODEL_NAMES,
    horizontal_spacing=0.08,
)

for ci, name in enumerate(MODEL_NAMES, start=1):
    pfd = per_feature_data[name]
    fig_freq.add_trace(
        go.Scatter(
            x=np.log10(pfd["freqs"]),
            y=pfd["f1"],
            mode="markers",
            marker=dict(size=4, color=MODEL_COLORS[name], opacity=0.5),
            showlegend=False,
        ),
        row=1,
        col=ci,
    )
    fig_freq.update_xaxes(
        title_text="log10(freq)" if ci == 2 else "",
        dtick=5,
        minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )
    fig_freq.update_yaxes(
        title_text="Per-feature F1" if ci == 1 else "",
        dtick=0.25,
        minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )

fig_freq.update_layout(
    width=int(350 * len(MODEL_NAMES) * 0.5),
    height=int(380 * 0.5),
    title_text=f"Feature Frequency vs F1 (target L0={_rep_l0})",
    margin=dict(l=50, r=15, t=50, b=35),
)
style_fig(fig_freq)
fig_freq.show()


# %% --- Plot: Dead Neurons vs L0 ---
fig_dead = go.Figure()
for name in MODEL_NAMES:
    means = [d[0] for d in dead_counts[name]]
    stds = [d[1] for d in dead_counts[name]]
    fig_dead.add_trace(
        go.Scatter(
            x=L0_VALUES,
            y=means,
            mode="lines+markers",
            name=name,
            marker=dict(size=8, color=MODEL_COLORS[name]),
            line=dict(color=MODEL_COLORS[name], width=2.5),
            error_y=dict(type="data", array=stds, visible=True),
        )
    )
fig_dead.update_layout(
    xaxis_title="Target L0",
    yaxis_title="Dead Dictionary Elements",
    width=700,
    height=450,
    margin=dict(l=70, r=30, t=40, b=50),
)
style_fig(fig_dead)
fig_dead.show()


# %% --- Plot: F1 Heatmap (model x L0) ---
_f1_mat = np.array([sweep_results[n]["f1"] for n in MODEL_NAMES])
fig_heat = go.Figure(
    go.Heatmap(
        z=_f1_mat,
        x=[str(v) for v in L0_VALUES],
        y=MODEL_NAMES,
        colorscale="Viridis",
        text=np.round(_f1_mat, 3).astype(str),
        texttemplate="%{text}",
        textfont=dict(size=13),
        showscale=True,
        colorbar=dict(title="F1"),
    )
)
fig_heat.update_layout(
    xaxis_title="Target L0",
    width=850,
    height=280,
    margin=dict(l=200, r=50, t=40, b=50),
    title_text="F1 Score Heatmap",
)
fig_heat.show()


# %% --- Plot: Decoder F1 vs Encoder F1 ---
fig_de = go.Figure()
for name in MODEL_NAMES:
    res = sweep_results[name]
    fig_de.add_trace(
        go.Scatter(
            x=res["f1"],
            y=res["enc_f1"],
            mode="markers+text",
            name=name,
            text=[str(t) for t in res["target_l0"]],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=MODEL_COLORS[name]),
        )
    )
fig_de.add_shape(
    type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray")
)
fig_de.update_layout(
    xaxis_title="Decoder-matched F1",
    yaxis_title="Encoder-matched F1",
    width=550,
    height=550,
    margin=dict(l=70, r=30, t=40, b=50),
    title_text="Decoder vs Encoder Matching Agreement",
)
style_fig(fig_de)
fig_de.show()


# %% --- Plot: R^2 vs F1 trade-off ---
fig_rf = go.Figure()
for name in MODEL_NAMES:
    res = sweep_results[name]
    fig_rf.add_trace(
        go.Scatter(
            x=res["r2"],
            y=res["f1"],
            mode="markers+lines+text",
            name=name,
            text=[str(t) for t in res["target_l0"]],
            textposition="top right",
            textfont=dict(size=9),
            marker=dict(size=8, color=MODEL_COLORS[name]),
            line=dict(color=MODEL_COLORS[name], width=1.5),
        )
    )
fig_rf.update_layout(
    xaxis_title="R²",
    yaxis_title="F1",
    width=600,
    height=550,
    margin=dict(l=70, r=30, t=40, b=50),
    title_text="R² vs F1 Trade-off (labels = target L0)",
)
style_fig(fig_rf)
fig_rf.show()


# %% --- Plot: Per-seed F1 box plots at representative L0 ---
fig_box = go.Figure()
for name in MODEL_NAMES:
    f1_seeds = [sweep_raw[name][_rep_l0_idx][s]["f1"] for s in range(N_SEEDS)]
    fig_box.add_trace(
        go.Box(
            y=f1_seeds,
            name=name,
            marker_color=MODEL_COLORS[name],
            boxmean="sd",
        )
    )
fig_box.update_layout(
    yaxis_title="F1",
    width=500,
    height=400,
    margin=dict(l=70, r=30, t=60, b=50),
    title_text=f"F1 Distribution across Seeds (target L0={_rep_l0})",
    showlegend=False,
)
style_fig(fig_box)
fig_box.show()


# %% --- Plot: Radar chart at representative L0 ---
_radar_metrics = ["f1", "r2", "mcc", "purity", "enc_f1", "enc_mcc"]
_radar_labels = ["F1", "R²", "MCC", "Purity", "Enc-F1", "Enc-MCC"]

fig_radar = go.Figure()
for name in MODEL_NAMES:
    res = sweep_results[name]
    vals = [res[m][_rep_l0_idx] for m in _radar_metrics]
    vals.append(vals[0])  # close polygon
    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=_radar_labels + [_radar_labels[0]],
            fill="toself",
            name=name,
            line=dict(color=MODEL_COLORS[name]),
            fillcolor=_hex_to_rgba(MODEL_COLORS[name], 0.1),
        )
    )
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    width=600,
    height=500,
    title_text=f"Metric Comparison at target L0={_rep_l0}",
    font=dict(size=13),
)
fig_radar.show()

# %%
