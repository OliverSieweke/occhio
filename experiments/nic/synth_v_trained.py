# %%
"""Compare TiedLinearRelu vs SynthAE on SyntheticDataModel.

TiedLinearRelu: learns W from data via gradient descent.
SynthAE: unit-norm tied weights positioned roughly orthogonally (no training needed).
Both use ReLU in decode. The question is whether the learned W outperforms
the synthetic dictionary approach.

Scaled-down version of the SSB benchmark (Anthropic, 2025) for a weaker machine:
  - N = 100 ground-truth features, D = 10 hidden dim
  - All other parameters scaled proportionally from the original
    (N=16384, D=768) specification.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
N_FEATURES = 200
D_HIDDEN = 40
N_EPOCHS = 30_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14
EVAL_FREQ = 250


# --- Build hierarchy forest ---
# Original: 128 roots, branching factor 4, max depth 3, covering 10884 features.
# Scaled: 8 roots, branching factor 4, max depth 2, covering 8*(1+4+16) = 168
# but we only have 100 features, so: 8 roots * (1 + 4) = 40 features in hierarchy.
# Remaining 60 features have no hierarchy.
def build_hierarchy(
    start_idx: int, n_roots: int, branching: int
) -> tuple[list[HierarchyNode], int]:
    """Build a forest of trees with mutual exclusion and parent-scaled magnitudes."""
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
# 8 roots with 4 children each = 40 features in hierarchy, 60 free

# %%
# --- SyntheticDataModel config ---
config = SyntheticDataConfig(
    n_features=N_FEATURES,
    # Firing probabilities
    firing_prob_distribution="zipfian",
    p_max=0.4,
    p_min=0.5 / N_FEATURES,
    alpha=0.8,
    # Magnitudes — linear mean, folded-normal stdev
    mean_distribution="linear",
    mean_high=3.0,
    mean_low=1.0,
    std_distribution="folded_normal",
    folded_normal_mu=0.5,
    folded_normal_sigma=0.5,
    # Correlation
    correlation_rank=4,
    correlation_scale=0.1,
    # Hierarchy
    hierarchy=hierarchy_roots,
    compensate_probabilities=True,
    # Runtime
    device="cpu",
)

dist = SyntheticDataModel(config, seed=SEED)

# %%
# Quick sanity check
activations = dist.sample(1024)
print(f"Activations shape: {activations.shape}")
print(f"Mean L0: {(activations > 0).float().sum(dim=-1).mean():.1f}")
print(
    f"Mean L2 norm: {activations.norm(dim=-1).mean():.2f} ± {activations.norm(dim=-1).std():.2f}"
)


# %%
# --- Hooks for evaluation ---
def eval_hook(data):
    """Compute eval loss on a large fresh sample."""
    tm = data["tm"]
    x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(x, x_hat, tm.importances).item()


def per_feature_hook(data):
    """Per-feature reconstruction MSE on one-hot inputs."""
    tm = data["tm"]
    eye = torch.eye(N_FEATURES, device=tm.device)
    x_hat = tm.ae(eye)[0]
    return (eye - x_hat).pow(2).sum(dim=-1).cpu().numpy()


# %%
# --- Helper: evaluate a (non-trained) model once ---
def evaluate_model(tm):
    """Return (eval_loss, per_feature_mse) for a model without training."""
    with torch.no_grad():
        x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
        x_hat = tm.ae(x)[0]
        eval_loss = tm.ae.loss(x, x_hat, tm.importances).item()

        eye = torch.eye(N_FEATURES, device=tm.device)
        e_hat = tm.ae(eye)[0]
        pf_mse = (eye - e_hat).pow(2).sum(dim=-1).cpu().numpy()
    return eval_loss, pf_mse


# %%
# --- Train TiedLinearRelu ---
print("Training TiedLinearRelu...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
ae_tied = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1)
tm_tied = ToyModel(distribution=dist, ae=ae_tied, device=DEVICE)

_, hook_results_tied = tm_tied.fit(
    N_EPOCHS,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_tied = hook_results_tied[0]
per_feature_tied = hook_results_tied[1]
print(f"  Final eval loss: {eval_losses_tied[-1]:.6f}")

# %%
# --- SynthAE (train bias only) ---
print("Training SynthAE (orthogonalized, bias only)...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth_ortho = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_synth_ortho = ToyModel(distribution=dist, ae=ae_synth_ortho, device=DEVICE)

N_EPOCHS_SYNTH = 15_000
_, hook_results_synth = tm_synth_ortho.fit(
    N_EPOCHS_SYNTH,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_synth = hook_results_synth[0]
per_feature_synth = hook_results_synth[1]
loss_synth_ortho = eval_losses_synth[-1]
pf_synth_ortho = per_feature_synth[-1]
print(f"  Final eval loss: {loss_synth_ortho:.6f}")

# %%
# --- Plot: Eval loss curve (TiedLinearRelu) with SynthAE baselines ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=eval_epochs, y=eval_losses_tied, name="TiedLinearRelu", opacity=0.8)
)
fig.add_hline(
    y=loss_synth_ortho,
    line_dash="dash",
    line_color="green",
)
fig.update_layout(
    title=f"Eval Loss — SyntheticDataModel (N={N_FEATURES}, D={D_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
)
fig.show()

# %%
# --- Plot: Per-feature reconstruction MSE ---
final_tied = np.array(per_feature_tied[-1])

# Sort features by firing probability (most frequent first)
firing_probs = dist.firing_probabilities.cpu().numpy()
sort_idx = np.argsort(-firing_probs)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=final_tied[sort_idx],
        mode="lines",
        name="TiedLinearRelu",
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=pf_synth_ortho[sort_idx],
        mode="lines",
        name="SynthAE (ortho)",
    )
)
fig.update_layout(
    title="Per-Feature Reconstruction MSE (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE",
)
fig.show()

# %%
# --- Plot: Features recovered (MSE < threshold) vs epoch, with SynthAE baselines ---
THRESHOLDS = [0.2, 0.5, 1.0]
COLORS = ["blue", "orange", "red"]

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, COLORS):
    n_recovered_tied = [int((np.array(s) < thresh).sum()) for s in per_feature_tied]
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered_tied,
            name=f"TiedLinearRelu (τ={thresh})",
            mode="lines",
            line=dict(color=color),
        )
    )
    n_recovered_ortho = int((pf_synth_ortho < thresh).sum())
    fig.add_hline(
        y=n_recovered_ortho,
        line_dash="dash",
        line_color=color,
    )
fig.update_layout(
    title="Features Recovered (MSE < τ) Over Training",
    xaxis_title="Epoch",
    yaxis_title="# features recovered",
)
fig.show()

# %%
# --- Plot: Per-feature MSE over training (heatmap, TiedLinearRelu only) ---
arr = np.array(per_feature_tied).T[:, 1:]  # (N_FEATURES, n_eval_points), skip step 0
arr_sorted = arr[sort_idx]

fig = go.Figure(
    go.Heatmap(
        z=arr_sorted,
        x=eval_epochs[1:],
        y=np.arange(N_FEATURES),
        colorscale="tempo",
        colorbar=dict(title="MSE"),
    )
)
fig.update_layout(
    title="Per-Feature MSE Over Training — TiedLinearRelu",
    xaxis_title="Epoch",
    yaxis_title="Feature rank (most frequent → rarest)",
    height=500,
)
fig.show()

# %%
# --- W^T W comparison ---
models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["TiedLinearRelu", "SynthAE (ortho)"]
)

for i, (name, tm) in enumerate(
    [
        ("TiedLinearRelu", tm_tied),
        ("SynthAE (ortho)", tm_synth_ortho),
    ]
):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(z=WtW, colorscale="RdBu_r", zmid=0, showscale=(i == 1)),
        row=1,
        col=i + 1,
    )

fig.update_layout(title="W^T W Comparison", height=400, width=900)
fig.show()

# %%
# --- Plot: Spectrum of W W^T ---
fig = go.Figure()
for name, tm in models:
    W = tm.W.detach().cpu()
    eigvals = torch.linalg.eigvalsh(W @ W.T).flip(0).numpy()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, D_HIDDEN + 1),
            y=eigvals,
            name=name,
            mode="lines+markers",
        )
    )
fig.update_layout(
    title="Spectrum of W W^T",
    xaxis_title="Eigenvalue index",
    yaxis_title="Eigenvalue",
)
fig.show()

# %%
# --- Plot: Geometric properties comparison ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Feature Dimensionalities",
        "Feature Norms",
        "Total Interference",
    ],
)

for name, tm in models:
    fd = tm.feature_dimensionalities.detach().cpu().numpy()[sort_idx]
    fn = tm.feature_norms.detach().cpu().numpy()[sort_idx]
    ti = tm.total_feature_interferences.detach().cpu().numpy()[sort_idx]
    x = np.arange(N_FEATURES)

    fig.add_trace(go.Scatter(x=x, y=fd, name=name, mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=fn, name=name, mode="lines", showlegend=False), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x, y=ti, name=name, mode="lines", showlegend=False), row=1, col=3
    )

fig.update_layout(
    title="Geometric Properties (sorted by firing probability)",
    height=400,
    width=1200,
)
for col in range(1, 4):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col)
fig.show()

# %%
# --- Plot: Bias comparison ---
fig = go.Figure()
for name, tm in models:
    b = tm.ae.b.detach().cpu().numpy()[sort_idx]  # ty:ignore
    fig.add_trace(go.Scatter(x=np.arange(N_FEATURES), y=b, name=name, mode="lines"))
fig.update_layout(
    title="Learned Bias b (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="b",
)
fig.show()

# %% --- Plot: Feature norms² + bias (TiedLinearRelu) ---
# This plot makes slightly less sense in this place but for consistency.
fn2 = tm_tied.feature_norms.detach().cpu().numpy() ** 2
b_tied = tm_tied.ae.b.detach().cpu().numpy()  # ty:ignore
combined = (fn2 + b_tied)[sort_idx]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=np.arange(N_FEATURES), y=combined, mode="lines", name="‖w‖² + b")
)
fig.update_layout(
    title="TiedLinearRelu: ‖w‖² + b (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="‖w‖² + b",
)
fig.show()


# %%
# --- Summary statistics ---
print("\n=== Summary ===")
for name, eval_loss, pf in [
    ("TiedLinearRelu", eval_losses_tied[-1], per_feature_tied[-1]),
    ("SynthAE (ortho)", loss_synth_ortho, pf_synth_ortho),
]:
    final_mse = np.array(pf)
    recovered = "  ".join(f"τ={t}: {int((final_mse < t).sum())}" for t in THRESHOLDS)
    print(
        f"{name:25s}  eval_loss={eval_loss:.6f}  "
        f"recovered=[{recovered}]  "
        f"mean_feature_MSE={final_mse.mean():.4f}"
    )

# %% --- SAE training on both models ---
N_DICT = N_FEATURES + 4
SAE_STEPS = 50_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.2

sae_results = {}

for name, tm in [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]:
    print(f"\nTraining SAE on {name}...")

    sae = SAESimple(
        n_latent=D_HIDDEN,
        n_dict=N_DICT,
        l1_coef=SAE_L1,
        device=DEVICE,
    ).to(DEVICE)

    def make_data_fn(tm_ref):
        def data_fn(n: int) -> torch.Tensor:
            x = tm_ref.distribution.sample(n).to(DEVICE)
            return tm_ref.ae.encode(x)

        return data_fn

    sae_losses = sae.train_sae(
        data_fn=make_data_fn(tm),
        n_steps=SAE_STEPS,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
    )

    # Compute metrics
    with torch.no_grad():
        test_x = dist.sample(10_000).to(DEVICE)
        test_hidden = tm.ae.encode(test_x)
        test_z = sae.encode(test_hidden)
        test_recon = sae.decode(test_z)

        # L0 sparsity: mean number of active (> 0) dict elements per sample
        l0 = (test_z > 0).float().sum(dim=-1).mean().item()

        # Dead features: dict elements that never fire
        ever_active = (test_z > 0).any(dim=0)
        n_dead = int((~ever_active).sum().item())
        n_alive = int(ever_active.sum().item())

        # Reconstruction MSE in hidden space
        recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

        # Per-feature faithfulness: encode one-hot, round-trip through SAE
        eye = torch.eye(N_FEATURES, device=DEVICE)
        h_eye = tm.ae.encode(eye)
        h_eye_recon = sae.decode(sae.encode(h_eye))
        per_feat_sae_mse = (h_eye - h_eye_recon).pow(2).sum(dim=-1).cpu().numpy()

        # Explained variance ratio
        total_var = test_hidden.var(dim=0).sum().item()
        residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
        explained_var = 1 - residual_var / total_var

    sae_results[name] = {
        "sae": sae,
        "tm": tm,
        "losses": sae_losses,
        "l0": l0,
        "n_dead": n_dead,
        "n_alive": n_alive,
        "recon_mse": recon_mse,
        "per_feat_sae_mse": per_feat_sae_mse,
        "explained_var": explained_var,
    }
    print(
        f"  L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
    )

# %% --- SAE loss curves ---
fig = go.Figure()
for name, res in sae_results.items():
    fig.add_trace(go.Scatter(y=res["losses"], mode="lines", name=name, opacity=0.8))
fig.update_layout(
    title="SAE Training Loss Comparison",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
)
fig.show()

# %% --- Per-feature SAE reconstruction error ---
names = list(sae_results.keys())
colors = ["blue", "green"]
fig = go.Figure()
for name in names:
    res = sae_results[name]
    fig.add_trace(
        go.Scatter(
            x=np.arange(N_FEATURES),
            y=res["per_feat_sae_mse"][sort_idx],
            name=name,
            mode="lines",
        )
    )
fig.update_layout(
    title="SAE Per-Feature Reconstruction Error (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE (hidden space)",
)
fig.show()

# %% --- SAE activations on one-hot features (matched) ---

for name in names:
    sae = sae_results[name]["sae"]
    tm_ref = tm_tied if name == "TiedLinearRelu" else tm_synth_ortho
    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = (
            sae.encode(tm_ref.ae.encode(eye)).cpu().numpy()
        )  # (N_FEATURES, N_DICT)

    # Hungarian matching: maximize total activation along the matched pairs
    cost = -sae_acts  # (N_FEATURES, N_DICT)
    feat_idx, dict_idx = linear_sum_assignment(cost)

    # Reorder both rows and columns: matched pairs form the top-left diagonal,
    # then append unmatched features (rows) and unmatched dict elements (cols)
    matched_feats = set(feat_idx)
    matched_dicts = set(dict_idx)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts

    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
    row_labels = [f"f{f}" for f in row_order]
    col_labels = [f"d{d}" for d in col_order]

    # Compute diagonality: fraction of total activation on the matched diagonal
    n_matched = len(feat_idx)
    diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
    total_sum = sae_acts_matched.sum()
    diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
    sae_results[name]["diagonality"] = diagonality
    print(
        f"{name}: diagonality = {diagonality:.4f} (diag_sum={diag_sum:.2f}, total={total_sum:.2f})"
    )

    px.imshow(
        sae_acts_matched,
        labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
        x=col_labels,
        y=row_labels,
        title=f"SAE one-hot activations (Hungarian matched, diag={diagonality:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    ).show()

# %% --- Compare matching methods: activation-based vs cosine similarity ---
print("=== Matching comparison: activation-based vs cosine-similarity ===\n")

for name in names:
    sae = sae_results[name]["sae"]
    tm_ref = tm_tied if name == "TiedLinearRelu" else tm_synth_ortho

    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)

        # Ground-truth feature directions in hidden space (normalized)
        D = tm_ref.ae.encode(eye)  # (N_FEATURES, D_HIDDEN)
        D_normed = D / D.norm(dim=1, keepdim=True)

        # SAE decoder columns (normalized). W_dec: (n_dict, n_latent)
        W_dec = sae.W_dec.data  # (N_DICT, D_HIDDEN)
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)

        # Method 1: activation-based (current code)
        sae_acts = sae.encode(tm_ref.ae.encode(eye)).cpu().numpy()
        cost_act = -sae_acts
        feat_idx_act, dict_idx_act = linear_sum_assignment(cost_act)

        # Method 2: absolute cosine similarity (O'Neill et al.)
        cosine_sim = (
            (D_normed @ W_dec_normed.T).abs().cpu().numpy()
        )  # (N_FEATURES, N_DICT)
        cost_cos = -cosine_sim
        feat_idx_cos, dict_idx_cos = linear_sum_assignment(cost_cos)

    # Compare matchings
    matching_act = set(zip(feat_idx_act, dict_idx_act))
    matching_cos = set(zip(feat_idx_cos, dict_idx_cos))
    overlap = matching_act & matching_cos
    agreement = len(overlap) / len(matching_act) * 100

    mean_cosine_matched = cosine_sim[feat_idx_cos, dict_idx_cos].mean()
    mean_act_matched = sae_acts[feat_idx_act, dict_idx_act].mean()

    # Cross-evaluate: how does each matching score under the other's metric?
    cosine_of_act_matching = cosine_sim[feat_idx_act, dict_idx_act].mean()
    act_of_cos_matching = sae_acts[feat_idx_cos, dict_idx_cos].mean()

    print(f"--- {name} ---")
    print(
        f"  Matching agreement: {len(overlap)}/{len(matching_act)} ({agreement:.1f}%)"
    )
    print(
        f"  Activation matching:  mean_act={mean_act_matched:.4f}  mean_cosine={cosine_of_act_matching:.4f}"
    )
    print(
        f"  Cosine matching:      mean_act={act_of_cos_matching:.4f}  mean_cosine={mean_cosine_matched:.4f}"
    )
    print()

    # Reorder cosine similarity matrix by cosine matching
    matched_feats_cos = set(feat_idx_cos)
    matched_dicts_cos = set(dict_idx_cos)
    unmatched_feats_cos = [f for f in range(N_FEATURES) if f not in matched_feats_cos]
    unmatched_dicts_cos = [d for d in range(N_DICT) if d not in matched_dicts_cos]

    row_order_cos = list(feat_idx_cos) + unmatched_feats_cos
    col_order_cos = list(dict_idx_cos) + unmatched_dicts_cos

    cosine_matched = cosine_sim[np.ix_(row_order_cos, col_order_cos)]
    row_labels_cos = [f"f{f}" for f in row_order_cos]
    col_labels_cos = [f"d{d}" for d in col_order_cos]

    n_matched_cos = len(feat_idx_cos)
    diag_sum_cos = sum(cosine_matched[i, i] for i in range(n_matched_cos))
    total_sum_cos = cosine_matched.sum()
    diagonality_cos = diag_sum_cos / total_sum_cos if total_sum_cos > 0 else 0.0

    px.imshow(
        cosine_matched,
        labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
        x=col_labels_cos,
        y=row_labels_cos,
        title=f"Cosine similarity (cosine matched, diag={diagonality_cos:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    ).show()

    # SAE one-hot activations reordered by cosine matching
    sae_acts_cos_order = sae_acts[np.ix_(row_order_cos, col_order_cos)]
    diag_sum_acts_cos = sum(sae_acts_cos_order[i, i] for i in range(n_matched_cos))
    total_sum_acts_cos = sae_acts_cos_order.sum()
    diag_acts_cos = (
        diag_sum_acts_cos / total_sum_acts_cos if total_sum_acts_cos > 0 else 0.0
    )

    px.imshow(
        sae_acts_cos_order,
        labels=dict(
            x="SAE dict element (cosine matched)", y="Feature (cosine matched)"
        ),
        x=col_labels_cos,
        y=row_labels_cos,
        title=f"SAE one-hot activations (cosine matched, diag={diag_acts_cos:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    ).show()

# %% --- SAE evaluation: MCC, detection metrics (both matchings) ---


def compute_detection_metrics(
    sae, tm, feat_idx, dict_idx, dist, device, n_samples=50_000
):
    """Compute detection metrics for a given feature-to-dict matching."""
    with torch.no_grad():
        det_x = dist.sample(n_samples).to(device)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)

        gt_active = det_x[:, feat_idx] > 0
        pred_active = det_z[:, dict_idx] > 0

        tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
        fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
        fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()
        tn = (~gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

    precision_per = tp / (tp + fp + 1e-8)
    recall_per = tp / (tp + fn + 1e-8)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-8)
    fpr_per = fp / (fp + tn + 1e-8)

    return {
        "precision_per": precision_per,
        "recall_per": recall_per,
        "f1_per": f1_per,
        "fpr_per": fpr_per,
        "precision": float(precision_per.mean()),
        "recall": float(recall_per.mean()),
        "f1": float(f1_per.mean()),
        "fpr": float(fpr_per.mean()),
        "confusion": np.array([[tn.sum(), fp.sum()], [fn.sum(), tp.sum()]]),
    }


for name, res in sae_results.items():
    sae = res["sae"]
    tm = res["tm"]

    with torch.no_grad():
        D = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
        W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = (D_norm.T @ W_norm).abs().cpu().numpy()  # (N_FEATURES, N_DICT)

        # Matching 1: cosine similarity (O'Neill et al.)
        cos_feat_idx, cos_dict_idx = linear_sum_assignment(-cos_sim)

        # Matching 2: activation-based
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()
        act_feat_idx, act_dict_idx = linear_sum_assignment(-sae_acts)

    # MCC under both matchings
    res["mcc_cos"] = float(cos_sim[cos_feat_idx, cos_dict_idx].mean())
    res["mcc_act"] = float(cos_sim[act_feat_idx, act_dict_idx].mean())

    # Keep the cosine matching as the primary (for downstream use)
    res["mcc"] = res["mcc_cos"]
    res["mcc_feat_idx"] = cos_feat_idx
    res["mcc_dict_idx"] = cos_dict_idx

    # Detection metrics under both matchings
    cos_metrics = compute_detection_metrics(
        sae, tm, cos_feat_idx, cos_dict_idx, dist, DEVICE
    )
    act_metrics = compute_detection_metrics(
        sae, tm, act_feat_idx, act_dict_idx, dist, DEVICE
    )

    # Store cosine matching metrics as primary
    for k, v in cos_metrics.items():
        res[k] = v

    # Store activation matching metrics with _act suffix
    for k, v in act_metrics.items():
        res[f"{k}_act"] = v

    print(f"--- {name} ---")
    print(
        f"  Cosine matching:     MCC={res['mcc_cos']:.4f}  Prec={cos_metrics['precision']:.4f}  "
        f"Rec={cos_metrics['recall']:.4f}  F1={cos_metrics['f1']:.4f}  FPR={cos_metrics['fpr']:.4f}"
    )
    print(
        f"  Activation matching: MCC={res['mcc_act']:.4f}  Prec={act_metrics['precision']:.4f}  "
        f"Rec={act_metrics['recall']:.4f}  F1={act_metrics['f1']:.4f}  FPR={act_metrics['fpr']:.4f}"
    )

# %% --- SAE summary print ---
print("\n=== SAE Summary ===")
print(
    f"{'Model':25s}  {'MSE↓':>10s}  {'L0↓':>6s}  {'Dead↓':>6s}  {'Alive↑':>6s}  "
    f"{'ExplVar↑':>8s}  {'Diag↑':>6s}  {'MCC↑':>6s}  {'Prec↑':>6s}  {'Rec↑':>6s}  {'F1↑':>6s}  {'FPR↓':>6s}"
)
for name, res in sae_results.items():
    print(
        f"{name:25s}  {res['recon_mse']:10.6f}  {res['l0']:6.1f}  "
        f"{res['n_dead']:6d}  {res['n_alive']:6d}  {res['explained_var']:8.4f}  "
        f"{res.get('diagonality', 0):6.4f}  {res.get('mcc', 0):6.4f}  "
        f"{res['precision']:6.4f}  {res['recall']:6.4f}  {res['f1']:6.4f}  {res['fpr']:6.4f}"
    )

# %% --- SAE summary comparison ---
fig = make_subplots(
    rows=1,
    cols=6,
    subplot_titles=[
        "Recon MSE",
        "Mean L0",
        "Dead Features",
        "Explained Variance",
        "Diagonality",
        "MCC",
    ],
)

for i, (name, color) in enumerate(zip(names, colors)):
    res = sae_results[name]
    fig.add_trace(
        go.Bar(x=[name], y=[res["recon_mse"]], marker_color=color, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=[name], y=[res["l0"]], marker_color=color, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=[name], y=[res["n_dead"]], marker_color=color, showlegend=False),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=[name], y=[res["explained_var"]], marker_color=color, showlegend=False
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        go.Bar(
            x=[name],
            y=[res.get("diagonality", 0)],
            marker_color=color,
            showlegend=False,
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Bar(
            x=[name],
            y=[res.get("mcc", 0)],
            marker_color=color,
            showlegend=False,
        ),
        row=1,
        col=6,
    )

fig.update_layout(
    title=f"SAE Comparison (dict={N_DICT}, L1={SAE_L1})", height=400, width=1600
)
fig.show()

# %% --- Confusion matrices ---
fig = make_subplots(
    rows=1,
    cols=len(names),
    subplot_titles=names,
)
for i, name in enumerate(names):
    cm = sae_results[name]["confusion"]
    # Normalize to rates
    cm_norm = cm / cm.sum()
    labels = ["Inactive", "Active"]
    text = [
        [f"{cm[r, c]:.0f}<br>({cm_norm[r, c]:.3f})" for c in range(2)] for r in range(2)
    ]
    fig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=(i == len(names) - 1),
        ),
        row=1,
        col=i + 1,
    )
    fig.update_yaxes(title_text="Ground Truth" if i == 0 else None, row=1, col=i + 1)
    fig.update_xaxes(title_text="SAE Prediction", row=1, col=i + 1)

fig.update_layout(
    title="Confusion Matrices (aggregated over matched features)", height=400, width=800
)
fig.show()

# %% --- Per-feature detection metrics (sorted by firing probability) ---
fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=["Precision", "Recall (TPR)", "F1 Score", "FPR"],
)

model_colors = {"TiedLinearRelu": "blue", "SynthAE (ortho)": "green"}
for name in names:
    res = sae_results[name]
    # The per-feature arrays are indexed by matched order (mcc_feat_idx).
    # Re-sort by firing probability for display.
    feat_order = np.argsort(-firing_probs[res["mcc_feat_idx"]])
    color = model_colors[name]
    x = np.arange(len(feat_order))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["precision_per"][feat_order],
            name=name,
            mode="lines",
            line=dict(color=color),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["recall_per"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["f1_per"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["fpr_per"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color),
        ),
        row=1,
        col=4,
    )

fig.update_layout(
    title="Per-Feature Detection Metrics (sorted by firing probability)",
    height=400,
    width=1400,
)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col)
fig.show()

# %%
