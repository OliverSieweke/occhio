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
from occhio.sae_lens_adapter.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)
from occhio.distributions import SparseUniform
from occhio.toy_model import ToyModel

# %%
# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {DEVICE}")

SEED = 42
N_FEATURES = 500
D_HIDDEN = 64
N_EPOCHS = 30_000

N_EPOCHS_SYNTH = 15_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14

# SAE sweep config
L0_VALUES = [3, 4]
BASE_L1_COEF = 0.3
N_DICT = N_FEATURES // 2
SAE_STEPS = 60_000
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
tm_trained.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True, sample_every=800)
print("  Done.")

# %%
# --- Train Trained AE (unit norm) ---
print("Training Trained AE (unit norm)...")
gen2 = torch.Generator(DEVICE).manual_seed(SEED)
ae_trained_normed = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen2)
tm_trained_normed = ToyModel(
    distribution=dist, ae=ae_trained_normed, device=DEVICE, hooks=[normalize_W]
)
tm_trained_normed.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True, sample_every=800)
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
tm_constructed.fit(
    N_EPOCHS_SYNTH, batch_size=BATCH_SIZE, verbose=True, sample_every=800
)
print("  Done.")


# %%
# --- Helpers ---
def make_data_fn(tm_ref, device):
    def data_fn(n: int) -> torch.Tensor:
        x = tm_ref.distribution.sample(n).to(device)
        return tm_ref.ae.encode(x)

    return data_fn


def train_sae_with_l0_target(
    sae: SAESimple,
    data_fn,
    target_l0: float,
    n_steps: int = 25_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    sample_every: int = 900,
) -> list[float]:
    """Train an SAE while autotuning l1_coef to hit a target L0."""
    from torch.optim import AdamW

    autotuner = CoefficientAutotuner(
        CoefficientAutotunerConfig(target_l0=target_l0),
    )
    base_l1 = sae.l1_coef
    optimizer = AdamW(sae.parameters(), lr=lr)
    sae_device = next(sae.parameters()).device
    loss_buffer = torch.empty(n_steps, device=sae_device)
    raw_buffer = None

    for step in range(n_steps):
        buf_offset = step % sample_every
        if buf_offset == 0:
            steps_left = min(sample_every, n_steps - step)
            total_samples = steps_left * batch_size
            raw_buffer = data_fn(total_samples).detach()

        start = buf_offset * batch_size
        end = start + batch_size
        x = raw_buffer[start:end]

        optimizer.zero_grad()
        x_hat, z = sae.forward(x)

        # Single GPU→CPU sync per step: .item() inside autotuner.update()
        batch_l0 = (z > 0).float().sum(dim=-1).mean()
        multiplier = autotuner.update(batch_l0, step)
        sae.l1_coef = base_l1 * multiplier

        loss = sae.loss(x, x_hat, z)
        loss.backward()
        optimizer.step()
        sae.constrain_weights()

        loss_buffer[step] = loss.detach()
        if (step + 1) % 5000 == 0:
            print(
                f"    step {step + 1}/{n_steps}  loss={loss.item():.4f}"
                f"  L0={autotuner.smoothed_l0:.1f}  mult={multiplier:.3f}"
                f"  l1={sae.l1_coef:.4f}"
            )

    return loss_buffer.cpu().tolist()


# %%
# --- L0 sweep with multi-seed averaging ---
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


# --- Phase 1: Train all SAEs (GPU-bound) ---
# Store trained SAEs for deferred evaluation
trained_saes: dict[str, list[list[tuple[SAESimple, "ToyModel"]]]] = {
    name: [] for name, _ in base_models
}

for li, target_l0 in enumerate(L0_VALUES):
    for name, tm in base_models:
        if li == 0:
            trained_saes[name] = []
        trained_saes[name].append([])

        for seed_i in range(N_SEEDS):
            print(f"  {name} target_L0={target_l0} seed={seed_i}...")
            sae = SAESimple(
                n_latent=D_HIDDEN,
                n_dict=N_DICT,
                l1_coef=BASE_L1_COEF,
                device=DEVICE,
            ).to(DEVICE)
            train_sae_with_l0_target(
                sae,
                data_fn=make_data_fn(tm, DEVICE),
                target_l0=target_l0,
                n_steps=SAE_STEPS,
                batch_size=SAE_BATCH,
                lr=SAE_LR,
            )
            trained_saes[name][li].append((sae, tm))
            print(f"    trained (target_l0={target_l0})")

# --- Phase 2: Evaluate all SAEs (CPU-bound due to linear_sum_assignment) ---
print("\nEvaluating all trained SAEs...")
sweep_raw: dict[str, list[list[dict]]] = {name: [] for name, _ in base_models}

for name, _ in base_models:
    sweep_raw[name] = []
    for li, target_l0 in enumerate(L0_VALUES):
        sweep_raw[name].append([])
        for seed_i, (sae, tm) in enumerate(trained_saes[name][li]):
            metrics = eval_sae(sae, tm)
            sweep_raw[name][li].append(metrics)
            print(
                f"  {name} target_L0={target_l0} seed={seed_i}"
                f"  F1={metrics['f1']:.4f}  L0={metrics['l0']:.1f}"
            )

# %%
# --- Aggregate across seeds (mean ± std) ---
sweep_results: dict[str, dict] = {}
for name, _ in base_models:
    res = {"target_l0": list(L0_VALUES)}
    for key in METRIC_KEYS:
        vals = np.array([[m[key] for m in seeds] for seeds in sweep_raw[name]])
        res[key] = vals.mean(axis=1).tolist()
        res[f"{key}_std"] = vals.std(axis=1).tolist()
    sweep_results[name] = res

for name in sweep_results:
    res = sweep_results[name]
    for i, tl0 in enumerate(res["target_l0"]):
        print(
            f"  {name:25s}  target_L0={tl0:2d}"
            f"  F1={res['f1'][i]:.4f}±{res['f1_std'][i]:.4f}"
            f"  L0={res['l0'][i]:.1f}±{res['l0_std'][i]:.1f}"
            f"  R²={res['r2'][i]:.4f}±{res['r2_std'][i]:.4f}"
        )

# %%
# --- Save checkpoint (run once after training — never retrain again) ---
from pathlib import Path

_ckpt_dir = (
    Path(
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in dir()
        else os.getcwd()
    )
    / "checkpoints"
)
_ckpt_dir.mkdir(exist_ok=True)

_sae_states = {}
for _ckpt_name in trained_saes:
    _sae_states[_ckpt_name] = [
        [sae.state_dict() for sae, _ in seed_list]
        for seed_list in trained_saes[_ckpt_name]
    ]

_base_states = {}
for _ckpt_name, _ckpt_tm in base_models:
    _base_states[_ckpt_name] = {
        "ae_state": _ckpt_tm.ae.state_dict(),
        "ae_class": type(_ckpt_tm.ae).__name__,
    }

torch.save(
    {
        "sae_states": _sae_states,
        "base_states": _base_states,
        "sweep_raw": sweep_raw,
        "sweep_results": sweep_results,
        "config": {
            "N_FEATURES": N_FEATURES,
            "D_HIDDEN": D_HIDDEN,
            "N_DICT": N_DICT,
            "L0_VALUES": L0_VALUES,
            "N_SEEDS": N_SEEDS,
            "SAE_STEPS": SAE_STEPS,
            "SEED": SEED,
            "firing_probs": firing_probs,
            "BASE_L1_COEF": BASE_L1_COEF,
            "DET_SAMPLES": DET_SAMPLES,
        },
    },
    _ckpt_dir / "sae_sweep_checkpoint.pt",
)
print(f"Checkpoint saved → {_ckpt_dir / 'sae_sweep_checkpoint.pt'}")

# %%
# =============================================================================
#  VISUALIZATION — run from here to re-plot without retraining
# =============================================================================

# --- Hardcoded sweep results (run training cells above to regenerate) ---
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

# --- Publication-ready figure styling (matches synth_v_trained_sparse.py) ---
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
    ("f1", '<span style="font-style:italic;">F</span><sub>1</sub>'),
    ("mcc", "MCC"),
    ("r2", '<span style="font-style:italic;">R</span><sup>2</sup>'),
]
_L0_DASH = "15px 10px"
_L0_COLOR = "#374151"
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
    horizontal_spacing=0.13,
)

_LEG_PAD = "&nbsp;" * 8  # trailing whitespace for legend spacing

_y_dticks = []
for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    # vline drawn first so it sits behind data traces but above the grid.
    # Huge y-range gets clipped to the panel extent by update_yaxes below.
    fig_main.add_trace(
        go.Scatter(
            x=[true_mean_l0, true_mean_l0],
            y=[-1e6, 1e6],
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

        _add_band(fig_main, x_s, y_s, y_std_s, color, name, row=1, col=ci)
        fig_main.add_trace(
            go.Scatter(
                x=x_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=_LEGEND_NAMES[name] + _LEG_PAD,
                legendgroup=name,
                showlegend=(ci == 1),
                marker=dict(size=12, color=color, line=dict(width=3, color="white")),
                line=dict(color=color, width=2.5),
            ),
            row=1,
            col=ci,
        )

    # compute nice y dtick for exactly 5 intervals
    _all_y = [v for res in sweep_results.values() for v in res[mk]]
    _all_std = [v for res in sweep_results.values() for v in res[f"{mk}_std"]]
    _y_raw_max = max(yv + s for yv, s in zip(_all_y, _all_std)) * 1.05
    _y_dticks.append(_nice_dtick(_y_raw_max))

# --- Dimensions (square panels) ---
_fs = 38  # axis titles
_fs_tick = 30  # tick labels
_fs_leg = _fs_tick  # legend font = tick font per user request

_plot_h = 400  # panel side length in px
_margin = dict(l=100, r=20, t=60, b=120)
_hs = 0.13  # horizontal_spacing — must match make_subplots above

# Square-panel formula: panel_width = (1-2*hs)/3 × plot_area_width == _plot_h
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
_x_lo = max(0, min(_all_l0) - 1)
_x_hi = max(_all_l0) + 1

# Per-panel axis overrides
for ci, (mk, ylabel) in enumerate(_MAIN_METRICS, start=1):
    _all_y = [v for res in sweep_results.values() for v in res[mk]]
    _all_std = [v for res in sweep_results.values() for v in res[f"{mk}_std"]]
    _y_lo = max(0, min(yv - s for yv, s in zip(_all_y, _all_std))) - 0.02
    _y_hi = max(yv + s for yv, s in zip(_all_y, _all_std)) + 0.02
    fig_main.update_yaxes(
        title_text=ylabel,
        range=[_y_lo, _y_hi],
        dtick=0.1 if ci < 3 else 0.2,
        tickangle=-90,
        minor=dict(dtick=0.05, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )
    fig_main.update_xaxes(
        range=[_x_lo, _x_hi],
        dtick=2,
        minor=dict(dtick=1, showgrid=True, gridcolor="#F0F0F0", gridwidth=1),
        row=1,
        col=ci,
    )

# First subplot (F1): extend upper bound to 0.8
# _f1_vals = [v for res in sweep_results.values() for v in res["f1"]]
# _f1_stds = [v for res in sweep_results.values() for v in res["f1_std"]]
# _f1_lo = max(0, min(yv - s for yv, s in zip(_f1_vals, _f1_stds)) - 0.05)
# fig_main.update_yaxes(range=[_f1_lo, 0.8], row=1, col=1)

# x-axis title on middle panel only
fig_main.update_xaxes(
    title_text='<span style="font-family:Times New Roman; font-style:italic;">L</span><sup>0</sup><sub>SAE</sub>',
    row=1,
    col=2,
)

# Legend: font = tick size, equal-width entries for even spacing
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
                name=_LEGEND_NAMES[name],
                marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=2.5),
            )
        )

    _fig.add_vline(
        x=true_mean_l0,
        line_dash=_L0_DASH,
        line_color="#1F2937",
        line_width=2.5,
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
    tl0_arr = np.array(res["target_l0"])
    order = np.argsort(tl0_arr)
    tl0_s = tl0_arr[order]

    for ri, (metric, metric_std) in enumerate(
        [("precision", "precision_std"), ("recall", "recall_std")], start=1
    ):
        y_s = np.array(res[metric])[order]
        y_std_s = np.array(res[metric_std])[order]

        _add_band(fig_pr, tl0_s, y_s, y_std_s, color, name, row=ri, col=1)
        fig_pr.add_trace(
            go.Scatter(
                x=tl0_s.tolist(),
                y=y_s.tolist(),
                mode="lines+markers",
                name=_LEGEND_NAMES[name],
                legendgroup=name,
                showlegend=(ri == 2),
                marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=2.5, dash="dash"),
            ),
            row=ri,
            col=1,
        )

# Shared x-axis label on bottom only
fig_pr.update_xaxes(
    title_text='Target <span style="font-family:Times New Roman; font-style:italic;">L</span><sup>0</sup>',
    row=2,
    col=1,
)
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
    f"\n{'Model':25s}  {'tL0':>4s}  {'F1':>15s}  {'Prec':>15s}  {'Recall':>15s}"
    f"  {'L0':>12s}  {'R²':>15s}  {'MCC':>15s}"
)
print("-" * 140)
for name, res in sweep_results.items():
    for i, tl0 in enumerate(res["target_l0"]):
        print(
            f"{name:25s}  {tl0:4d}"
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
    f"\n{'Model':25s}  {'tL0':>4s}  {'EncF1':>15s}  {'EncPrec':>15s}  {'EncRecall':>15s}"
    f"  {'EncMCC':>15s}"
)
print("-" * 100)
for name, res in sweep_results.items():
    for i, tl0 in enumerate(res["target_l0"]):
        print(
            f"{name:25s}  {tl0:4d}"
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
_fig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
    "figures",
)
os.makedirs(_fig_dir, exist_ok=True)

fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.pdf"), engine="kaleido")
# fig_main.write_image(os.path.join(_fig_dir, "sae_main_metrics.svg"), engine="kaleido")
# fig_pr.write_image(os.path.join(_fig_dir, "sae_prec_vs_recall.pdf"), engine="kaleido")
# fig_pr.write_image(os.path.join(_fig_dir, "sae_prec_vs_recall.svg"), engine="kaleido")
# for _mk, _mfig in metric_figs.items():
#     _mfig.write_image(os.path.join(_fig_dir, f"sae_{_mk}_vs_l0.pdf"), engine="kaleido")
#     _mfig.write_image(os.path.join(_fig_dir, f"sae_{_mk}_vs_l0.svg"), engine="kaleido")
print(f"Saved to {_fig_dir}/")

# %%
