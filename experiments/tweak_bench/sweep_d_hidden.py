# %%
"""Sweep D_HIDDEN for HierarchicalPairs — compare Matryoshka vs BatchTopK SAEs."""

import tempfile

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sae_lens import SAE
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.correlated import HierarchicalPairs
from occhio.toy_model import SAEEntry, ToyModel

# --- Paper-quality plot defaults ---
FONT = dict(family="Times New Roman, serif", size=24, color="#333333")
AXIS_STYLE = dict(
    showgrid=False,
    gridcolor="rgba(0,0,0,0.08)",
    gridwidth=1,
    zeroline=False,
    linecolor="#666666",
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickwidth=1,
    tickcolor="#666666",
    tickfont_size=18,
    minor=dict(ticks="outside", ticklen=2),
)
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=FONT,
    title_font_size=24,
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        font_size=24,
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 1296
N_EPOCHS = 50_000
BATCH_SIZE = 512

# SAE config
N_DICT = N_FEATURES // 2
SAE_STEPS = 30_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_K = 3
N_TEST = 100_000

# Sweep
D_HIDDEN_VALUES = [100, 200, 250, 300, 350, 400]
MATRYOSHKA_WIDTHS_FN = lambda n_dict: [n_dict // 2, n_dict]  # noqa: E731

# %%
# --- Distribution (shared across all runs) ---
np.random.seed(8)

high = 0.45
low = 1.3 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
betas = np.random.random(N_FEATURES)

dist = HierarchicalPairs(
    N_FEATURES, p_active=firing_probs, p_follow=0.6, beta=betas, device=DEVICE
)
print(f"Distribution: HierarchicalPairs, N={N_FEATURES}, alpha={alpha:.3f}")
samples = dist.sample(100_000)
mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
print(f"Average L0: {mean_l0:.2f}")


# %%
# --- Helpers ---
def to_jumprelu(training_sae, device):
    """Convert a (Matryoshka)BatchTopK training SAE to JumpReLU for inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        training_sae.save_inference_model(tmpdir)
        return SAE.load_from_disk(tmpdir, device=device)


def compute_ae_best_f1(tm, n_test=N_TEST, n_thresholds=200):
    """Compute the best macro-F1 the AE round-trip achieves by sweeping thresholds.

    For each candidate threshold t in [0, 1], predicts feature i as active if
    ``x_hat[:, i] > t`` (ground truth stays ``x_true > 0``). Returns the best
    macro-F1 across all thresholds and the threshold that achieved it.
    """
    with torch.no_grad():
        test_x = dist.sample(n_test).to(DEVICE)
        test_xhat = tm.ae.decode(tm.ae.encode(test_x))

        gt = test_x > 0  # (N, F)
        gt_f = gt.float()
        pos_per_feat = gt_f.sum(dim=0)  # (F,)

        thresholds = torch.linspace(0.0, 1.0, n_thresholds, device=DEVICE)

        best_f1 = -1.0
        best_t = 0.0
        for t in thresholds:
            pred = test_xhat > t
            tp = (gt & pred).float().sum(dim=0)
            fp = (~gt & pred).float().sum(dim=0)
            fn = pos_per_feat - tp  # gt_f.sum - tp

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t.item())

    return {"f1_macro": best_f1, "threshold": best_t}


def compute_f1(sae_inf, tm, n_test=N_TEST):
    """Compute macro and micro F1 using JumpReLU inference SAE."""
    with torch.no_grad():
        test_x = dist.sample(n_test).to(DEVICE)
        test_h = tm.ae.encode(test_x)
        test_z = sae_inf.encode(test_h)

        # MCC matching
        D = tm.W.detach()
        W_dec_t = sae_inf.W_dec.detach().T
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = (D_norm.T @ W_norm).cpu().numpy()

        feat_idx, dict_idx = linear_sum_assignment(-cos_sim)
        mcc = float(cos_sim[feat_idx, dict_idx].mean())

        # F1
        gt = test_x[:, feat_idx] > 0
        pred = test_z[:, dict_idx] > 0
        tp = (gt & pred).float().sum(dim=0)
        fp = (~gt & pred).float().sum(dim=0)
        fn = (gt & ~pred).float().sum(dim=0)

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_macro = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()

        tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
        p_micro = (tp_t / (tp_t + fp_t + 1e-8)).item()
        r_micro = (tp_t / (tp_t + fn_t + 1e-8)).item()
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-8)

        # R²
        test_recon = sae_inf.decode(test_z)
        total_var = test_h.var(dim=0).sum().item()
        resid_var = (test_h - test_recon).var(dim=0).sum().item()
        r2 = 1 - resid_var / total_var

        # L0
        l0 = (test_z > 0).float().sum(dim=-1).mean().item()

        # Dead latents
        dead = int((~(test_z > 0).any(dim=0)).sum().item())

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "mcc": mcc,
        "r2": r2,
        "l0": l0,
        "dead": dead,
    }


# %%
# --- Sweep ---
results = []

for d_hidden in D_HIDDEN_VALUES:
    print(f"\n{'=' * 60}")
    print(f"D_HIDDEN = {d_hidden}")
    print(f"{'=' * 60}")

    # Train AE
    gen = torch.Generator(DEVICE).manual_seed(SEED)
    ae = TiedLinearRelu(N_FEATURES, d_hidden, device=DEVICE, generator=gen)
    tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
    tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

    # --- AE own F1 (best over threshold sweep) ---
    ae_metrics = compute_ae_best_f1(tm)
    print(
        f"  [AE]         f1_macro={ae_metrics['f1_macro']:.4f}  "
        f"threshold={ae_metrics['threshold']:.3f}"
    )

    train_samples = SAE_STEPS * SAE_BATCH

    # --- Matryoshka ---
    widths = MATRYOSHKA_WIDTHS_FN(N_DICT)
    mat_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=d_hidden,
        d_sae=N_DICT,
        matryoshka_widths=widths,
        k=SAE_K,
        device="cpu",
    )
    mat_sae = MatryoshkaBatchTopKTrainingSAE(mat_cfg)
    mat_sae.topk_threshold = mat_sae.topk_threshold.to(torch.float32)
    mat_sae.to(DEVICE)

    tm.train_saes(
        [SAEEntry(sae=mat_sae, type="Matryoshka", label="matryoshka")],
        training_samples=train_samples,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
        verbose=True,
    )

    mat_inf = to_jumprelu(mat_sae, DEVICE)
    mat_metrics = compute_f1(mat_inf, tm)
    print(f"  [Matryoshka] {mat_metrics}")

    # --- BatchTopK ---
    btk_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_hidden,
        d_sae=N_DICT,
        k=SAE_K,
        device="cpu",
    )
    btk_sae = BatchTopKTrainingSAE(btk_cfg)
    btk_sae.topk_threshold = btk_sae.topk_threshold.to(torch.float32)
    btk_sae.to(DEVICE)

    tm.train_saes(
        [SAEEntry(sae=btk_sae, type="BatchTopK", label="batchtopk")],
        training_samples=train_samples,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
        verbose=True,
    )

    btk_inf = to_jumprelu(btk_sae, DEVICE)
    btk_metrics = compute_f1(btk_inf, tm)
    print(f"  [BatchTopK]  {btk_metrics}")

    results.append(
        {
            "d_hidden": d_hidden,
            "ae": ae_metrics,
            "matryoshka": mat_metrics,
            "batchtopk": btk_metrics,
        }
    )

# %%
# --- Plot: F1 (macro) vs D_HIDDEN ---
d_vals = [r["d_hidden"] for r in results]
mat_f1 = [r["matryoshka"]["f1_macro"] for r in results]
btk_f1 = [r["batchtopk"]["f1_macro"] for r in results]
ae_f1 = [r["ae"]["f1_macro"] for r in results]
mat_mcc = [r["matryoshka"]["mcc"] for r in results]
btk_mcc = [r["batchtopk"]["mcc"] for r in results]
mat_r2 = [r["matryoshka"]["r2"] for r in results]
btk_r2 = [r["batchtopk"]["r2"] for r in results]

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["F1 (macro)", "MCC", "R²"],
    shared_yaxes=False,
)

COLOR_MAT = "#636EFA"
COLOR_BTK = "#EF553B"
COLOR_AE = "#888888"

# F1
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=ae_f1,
        mode="lines+markers",
        name="AE (best threshold)",
        marker_color=COLOR_AE,
        line=dict(color=COLOR_AE, dash="dot"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=mat_f1,
        mode="lines+markers",
        name="Matryoshka",
        marker_color=COLOR_MAT,
        line_color=COLOR_MAT,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=btk_f1,
        mode="lines+markers",
        name="BatchTopK",
        marker_color=COLOR_BTK,
        line_color=COLOR_BTK,
    ),
    row=1,
    col=1,
)

# MCC
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=mat_mcc,
        mode="lines+markers",
        name="Matryoshka",
        showlegend=False,
        marker_color=COLOR_MAT,
        line_color=COLOR_MAT,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=btk_mcc,
        mode="lines+markers",
        name="BatchTopK",
        showlegend=False,
        marker_color=COLOR_BTK,
        line_color=COLOR_BTK,
    ),
    row=1,
    col=2,
)

# R²
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=mat_r2,
        mode="lines+markers",
        name="Matryoshka",
        showlegend=False,
        marker_color=COLOR_MAT,
        line_color=COLOR_MAT,
    ),
    row=1,
    col=3,
)
fig.add_trace(
    go.Scatter(
        x=d_vals,
        y=btk_r2,
        mode="lines+markers",
        name="BatchTopK",
        showlegend=False,
        marker_color=COLOR_BTK,
        line_color=COLOR_BTK,
    ),
    row=1,
    col=3,
)

for col in range(1, 4):
    fig.update_xaxes(title_text="D_HIDDEN", row=1, col=col, **AXIS_STYLE)
fig.update_yaxes(row=1, col=1, **AXIS_STYLE)
fig.update_yaxes(row=1, col=2, **AXIS_STYLE)
fig.update_yaxes(row=1, col=3, **AXIS_STYLE)

fig.update_layout(
    title=f"Matryoshka vs BatchTopK — D_HIDDEN sweep (N={N_FEATURES}, k={SAE_K})",
    **LAYOUT_DEFAULTS,
    width=1200,
    height=450,
)
fig.show()

# %%
# --- Summary table ---
print(
    f"\n{'D_HIDDEN':>8} | {'AE F1':>6} {'thr':>5}"
    f" | {'Matryoshka F1':>13} {'MCC':>6} {'R²':>6} {'L0':>5} {'DL':>4}"
    f" | {'BatchTopK F1':>12} {'MCC':>6} {'R²':>6} {'L0':>5} {'DL':>4}"
)
print("-" * 110)
for r in results:
    a, m, b = r["ae"], r["matryoshka"], r["batchtopk"]
    print(
        f"{r['d_hidden']:>8} | "
        f"{a['f1_macro']:>6.4f} {a['threshold']:>5.3f}"
        f" | {m['f1_macro']:>13.4f} {m['mcc']:>6.3f} {m['r2']:>6.3f} {m['l0']:>5.1f} {m['dead']:>4}"
        f" | {b['f1_macro']:>12.4f} {b['mcc']:>6.3f} {b['r2']:>6.3f} {b['l0']:>5.1f} {b['dead']:>4}"
    )
# %%
