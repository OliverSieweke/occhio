# %%
"""SAE L1 sweep on TiedLinearRelu vs SynthAE.

Trains both base models (TiedLinearRelu, SynthAE), then sweeps SAE L1
coefficients [0.1, 0.2, 0.5, 1.0, 1.5] on each. Plots F1 score vs
mean L0 sparsity for all runs.
"""

import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
L1_VALUES = [0.08, 0.09, 0.1, 0.2, 0.5, 0.8]
N_DICT = N_FEATURES // 2
SAE_STEPS = 25_000
SAE_BATCH = 1024
SAE_LR = 3e-4
DET_SAMPLES = 50_000

high = 0.2
low = 0.5 / N_FEATURES
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
# --- Train TiedLinearRelu ---
print("Training TiedLinearRelu...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
ae_tied = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1)
tm_tied = ToyModel(distribution=dist, ae=ae_tied, device=DEVICE)
tm_tied.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)
print("  Done.")

# %%
# --- Train SynthAE ---
print("Training SynthAE (orthogonalized, bias only)...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_synth = ToyModel(distribution=dist, ae=ae_synth, device=DEVICE)
tm_synth.fit(N_EPOCHS_SYNTH, batch_size=BATCH_SIZE, verbose=True)
print("  Done.")


# %%
# --- Helpers ---
def make_data_fn(tm_ref, device):
    def data_fn(n: int) -> torch.Tensor:
        x = tm_ref.distribution.sample(n).to(device)
        return tm_ref.ae.encode(x)

    return data_fn


# %%
# --- L1 sweep (training only) ---
base_models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth)]
sweep_results = {name: {"l1": [], "sae": []} for name, _ in base_models}

for l1_coef in L1_VALUES:
    for name, tm in base_models:
        print(f"  Training SAE on {name} with L1={l1_coef}...")

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

        sweep_results[name]["l1"].append(l1_coef)
        sweep_results[name]["sae"].append(sae)
        print(f"    L1={l1_coef} done.")

# %%
# --- Compute metrics for all SAEs ---
for name, tm in base_models:
    res = sweep_results[name]
    res["f1"] = []
    res["precision"] = []
    res["recall"] = []
    res["l0"] = []
    res["r2"] = []
    res["mcc"] = []
    res["purity"] = []

    for i, sae in enumerate(res["sae"]):
        l1_coef = res["l1"][i]
        with torch.no_grad():
            # --- Feature matching via cosine similarity ---
            eye = torch.eye(tm.ae.n_features, device=DEVICE)
            D_enc = tm.ae.encode(eye)  # (N_FEATURES, D_HIDDEN)
            D_enc_normed = D_enc / D_enc.norm(dim=1, keepdim=True)

            W_dec = sae.W_dec.data  # (N_DICT, D_HIDDEN)
            W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)

            cos_sim = (D_enc_normed @ W_dec_normed.T).abs().cpu().numpy()
            feat_idx, dict_idx = linear_sum_assignment(-cos_sim)

            # --- Sample data for detection and reconstruction metrics ---
            det_x = dist.sample(DET_SAMPLES).to(DEVICE)
            det_hidden = tm.ae.encode(det_x)
            det_z = sae.encode(det_hidden)
            det_recon = sae.decode(det_z)

            # --- L0 ---
            l0 = (det_z > 0).float().sum(dim=-1).mean().item()

            # --- F1 ---
            gt_active = det_x[:, feat_idx] > 0
            pred_active = det_z[:, dict_idx] > 0

            tp = (gt_active & pred_active).float().sum(dim=0)
            fp = (~gt_active & pred_active).float().sum(dim=0)
            fn = (gt_active & ~pred_active).float().sum(dim=0)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_per = 2 * precision * recall / (precision + recall + 1e-8)
            mean_f1 = f1_per.mean().item()

            # --- R² (reconstruction quality in hidden space) ---
            ss_res = ((det_hidden - det_recon) ** 2).sum().item()
            ss_tot = ((det_hidden - det_hidden.mean(dim=0)) ** 2).sum().item()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # --- MCC (mean cosine similarity on matched pairs, using W columns) ---
            W = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
            W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
            W_norm = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            Wd_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
            cos_mcc = (W_norm.T @ Wd_norm).cpu().numpy()
            cos_mcc_abs = np.abs(cos_mcc)
            mcc_feat_idx, mcc_dict_idx = linear_sum_assignment(-cos_mcc_abs)
            mcc = float(cos_mcc_abs[mcc_feat_idx, mcc_dict_idx].mean())

            # --- Purity (diagonality of SAE activations on one-hot inputs) ---
            sae_acts = sae.encode(D_enc).cpu().numpy()  # (N_FEATURES, N_DICT)
            cos_purity = (D_enc_normed @ W_dec_normed.T).cpu().numpy()
            pf_idx, pd_idx = linear_sum_assignment(-cos_purity)

            matched_feats = set(pf_idx)
            matched_dicts = set(pd_idx)
            unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
            unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]
            row_order = list(pf_idx) + unmatched_feats
            col_order = list(pd_idx) + unmatched_dicts
            sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]

            n_matched = len(pf_idx)
            diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
            total_sum = sae_acts_matched.sum()
            purity = diag_sum / total_sum if total_sum > 0 else 0.0

        res["f1"].append(mean_f1)
        res["precision"].append(precision.mean().item())
        res["recall"].append(recall.mean().item())
        res["l0"].append(l0)
        res["r2"].append(r2)
        res["mcc"].append(mcc)
        res["purity"].append(purity)
        print(
            f"  {name} L1={l1_coef}  F1={mean_f1:.4f}"
            f"  Prec={precision.mean().item():.4f}  Rec={recall.mean().item():.4f}"
            f"  L0={l0:.1f}  R²={r2:.4f}  MCC={mcc:.4f}  Purity={purity:.4f}"
        )

# %%
# --- Plot: F1 vs L0 ---
colors = {"TiedLinearRelu": "blue", "SynthAE (ortho)": "green"}

fig = go.Figure()
for name, res in sweep_results.items():
    fig.add_trace(
        go.Scatter(
            x=res["l0"],
            y=res["f1"],
            mode="lines+markers+text",
            name=name,
            text=[f"L1={l1}" for l1 in res["l1"]],
            textposition="top center",
            marker=dict(size=10, color=colors[name]),
            line=dict(color=colors[name]),
        )
    )

fig.update_layout(
    title=f"SAE F1 vs Mean L0 — L1 Sweep (N={N_FEATURES}, D={D_HIDDEN}, dict={N_DICT})",
    xaxis_title="Mean L0 (avg active dict elements per sample)",
    yaxis_title="Mean F1 Score",
    width=900,
    height=600,
)
fig.show()

# %%
# --- Plot: Recall vs Precision ---
fig2 = go.Figure()
for name, res in sweep_results.items():
    best_idx = int(np.argmax(res["f1"]))
    fig2.add_trace(
        go.Scatter(
            x=res["precision"],
            y=res["recall"],
            mode="lines+markers+text",
            name=name,
            text=[f"L1={l1}" for l1 in res["l1"]],
            textposition="top center",
            marker=dict(size=10, color=colors[name]),
            line=dict(color=colors[name]),
        )
    )
    # Highlight best F1 point
    fig2.add_trace(
        go.Scatter(
            x=[res["precision"][best_idx]],
            y=[res["recall"][best_idx]],
            mode="markers+text",
            name=f"{name} best F1",
            text=[f"★ F1={res['f1'][best_idx]:.3f}"],
            textposition="bottom center",
            marker=dict(
                size=16,
                color=colors[name],
                symbol="star",
                line=dict(width=2, color="black"),
            ),
            showlegend=False,
        )
    )

fig2.update_layout(
    title=f"SAE Recall vs Precision — L1 Sweep (N={N_FEATURES}, D={D_HIDDEN}, dict={N_DICT})",
    xaxis_title="Mean Precision",
    yaxis_title="Mean Recall",
    width=900,
    height=600,
)
fig2.show()

# %%
# --- Print summary table ---
print(
    f"\n{'Model':25s}  {'L1':>6s}  {'F1':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'L0':>8s}"
    f"  {'R²':>8s}  {'MCC':>8s}  {'Purity':>8s}"
)
print("-" * 103)
for name, res in sweep_results.items():
    for l1, f1, prec, rec, l0, r2, mcc, purity in zip(
        res["l1"],
        res["f1"],
        res["precision"],
        res["recall"],
        res["l0"],
        res["r2"],
        res["mcc"],
        res["purity"],
    ):
        print(
            f"{name:25s}  {l1:6.2f}  {f1:8.4f}  {prec:8.4f}  {rec:8.4f}  {l0:8.1f}"
            f"  {r2:8.4f}  {mcc:8.4f}  {purity:8.4f}"
        )

# %%
# --- W^T W for TiedLinearRelu ---
with torch.no_grad():
    W = tm_tied.W.detach().cpu().numpy()  # (D_HIDDEN, N_FEATURES)
    WtW = W.T @ W  # (N_FEATURES, N_FEATURES)

fig = px.imshow(
    WtW,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title="W^T W — TiedLinearRelu",
    labels=dict(x="Feature j", y="Feature i", color="W^T W"),
)
fig.show()

# %%
