# %%
"""SAE L1 sweep on TiedLinearRelu vs SynthAE.

Trains both base models (TiedLinearRelu, SynthAE), then sweeps SAE L1
coefficients [0.1, 0.2, 0.5, 1.0, 1.5] on each. Plots F1 score vs
mean L0 sparsity for all runs.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions.ssb import (
    SyntheticDataModel,
    SyntheticDataConfig,
    HierarchyNode,
)
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
L1_VALUES = [0.1, 0.2, 0.5, 1.0, 1.5]
N_DICT = N_FEATURES
SAE_STEPS = 25_000
SAE_BATCH = 1024
SAE_LR = 3e-4
DET_SAMPLES = 50_000


# %%
# --- Build hierarchy forest ---
def build_hierarchy(
    start_idx: int, n_roots: int, branching: int
) -> tuple[list[HierarchyNode], int]:
    idx = start_idx
    roots = []
    for _ in range(n_roots):
        root_idx = idx
        idx += 1
        children = []
        for _ in range(branching):
            children.append(HierarchyNode(feature_idx=idx))
            idx += 1
        roots.append(
            HierarchyNode(
                feature_idx=root_idx,
                children=children,
                mutually_exclusive_children=True,
                parent_scaled=True,
            )
        )
    return roots, idx


hierarchy_roots, _ = build_hierarchy(start_idx=0, n_roots=16, branching=5)

# %%
# --- Distribution ---
config = SyntheticDataConfig(
    n_features=N_FEATURES,
    firing_prob_distribution="zipfian",
    p_max=0.3,
    p_min=0.5 / N_FEATURES,
    alpha=0.7,
    mean_distribution="linear",
    mean_high=3.0,
    mean_low=1.0,
    std_distribution="folded_normal",
    folded_normal_mu=0.5,
    folded_normal_sigma=0.5,
    correlation_rank=4,
    correlation_scale=0.1,
    hierarchy=hierarchy_roots,
    compensate_probabilities=True,
    device="cpu",
)

dist = SyntheticDataModel(config, seed=SEED)

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


def compute_f1_l0(sae, tm, dist, device, n_samples=DET_SAMPLES):
    """Compute mean F1 and mean L0 for a trained SAE."""
    with torch.no_grad():
        # --- Matching via cosine similarity ---
        eye = torch.eye(tm.ae.n_features, device=device)
        D = tm.ae.encode(eye)  # (N_FEATURES, D_HIDDEN)
        D_normed = D / D.norm(dim=1, keepdim=True)

        W_dec = sae.W_dec.data  # (N_DICT, D_HIDDEN)
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)

        cos_sim = (D_normed @ W_dec_normed.T).abs().cpu().numpy()
        feat_idx, dict_idx = linear_sum_assignment(-cos_sim)

        # --- Detection metrics ---
        det_x = dist.sample(n_samples).to(device)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)

        # Mean L0 across all dict elements
        l0 = (det_z > 0).float().sum(dim=-1).mean().item()

        # Per-feature F1 on matched pairs
        gt_active = det_x[:, feat_idx] > 0
        pred_active = det_z[:, dict_idx] > 0

        tp = (gt_active & pred_active).float().sum(dim=0)
        fp = (~gt_active & pred_active).float().sum(dim=0)
        fn = (gt_active & ~pred_active).float().sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_per = 2 * precision * recall / (precision + recall + 1e-8)
        mean_f1 = f1_per.mean().item()

    return mean_f1, l0


# %%
# --- L1 sweep ---
base_models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth)]
sweep_results = {
    name: {"l1": [], "f1": [], "l0": [], "sae": []} for name, _ in base_models
}

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

        f1, l0 = compute_f1_l0(sae, tm, dist, DEVICE)
        sweep_results[name]["l1"].append(l1_coef)
        sweep_results[name]["f1"].append(f1)
        sweep_results[name]["l0"].append(l0)
        sweep_results[name]["sae"].append(sae)
        print(f"    L1={l1_coef}  F1={f1:.4f}  L0={l0:.1f}")

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
# --- Print summary table ---
print(f"\n{'Model':25s}  {'L1':>6s}  {'F1':>8s}  {'L0':>8s}")
print("-" * 55)
for name, res in sweep_results.items():
    for l1, f1, l0 in zip(res["l1"], res["f1"], res["l0"]):
        print(f"{name:25s}  {l1:6.2f}  {f1:8.4f}  {l0:8.1f}")

# %%
# --- Purity, MCC, F1 for L1=0.2 ---
L1_EVAL = 0.2

print(f"\n=== Detailed metrics for L1={L1_EVAL} ===")
print(f"{'Model':25s}  {'MCC↑':>8s}  {'Purity↑':>8s}  {'F1↑':>8s}")
print("-" * 55)

for name, tm in base_models:
    l1_idx = L1_VALUES.index(L1_EVAL)
    sae = sweep_results[name]["sae"][l1_idx]

    with torch.no_grad():
        # --- MCC: cosine similarity between ground-truth W columns and SAE decoder ---
        D = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
        W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = (D_norm.T @ W_norm).cpu().numpy()  # (N_FEATURES, N_DICT)
        cos_sim_abs = np.abs(cos_sim)

        mcc_feat_idx, mcc_dict_idx = linear_sum_assignment(-cos_sim_abs)
        mcc = float(cos_sim_abs[mcc_feat_idx, mcc_dict_idx].mean())

        # --- Purity (diagonality): fraction of SAE activation on matched diagonal ---
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (N_FEATURES, N_DICT)

        # Cosine matching for purity (same as synth_v_trained_sparse.py)
        D_enc = tm.ae.encode(eye)
        D_enc_normed = D_enc / D_enc.norm(dim=1, keepdim=True)
        W_dec = sae.W_dec.data
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)
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

        # --- F1 (detection) ---
        det_x = dist.sample(DET_SAMPLES).to(DEVICE)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)

        gt_active = det_x[:, mcc_feat_idx] > 0
        pred_active = det_z[:, mcc_dict_idx] > 0

        tp = (gt_active & pred_active).float().sum(dim=0)
        fp = (~gt_active & pred_active).float().sum(dim=0)
        fn = (gt_active & ~pred_active).float().sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_per = 2 * precision * recall / (precision + recall + 1e-8)
        mean_f1 = f1_per.mean().item()

    print(f"{name:25s}  {mcc:8.4f}  {purity:8.4f}  {mean_f1:8.4f}")

# %%
