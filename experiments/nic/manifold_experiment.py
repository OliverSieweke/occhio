# %%
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SphericalDistribution, DistributionStack
from occhio.sae.sae import SAESimple
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(6)

n_features = 42
length_scale = 1.5
manifold_dim = 1
n_hidden = 3

dist = DistributionStack(
    [
        SphericalDistribution(
            n_features // 3,
            length_scale=length_scale,
            manifold_dim=manifold_dim,
            magnitude_range=(0.9, 1.0),
            generator=gen,
            device=DEVICE,
        )
        for i in range(3)
    ],
    sampling_mode="single",
    p_meta=0.5,
)


ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
# %%

print(dist.sample(1))
# %%
N_EPOCHS = 100_000
EVAL_SAMPLES = 2**14
EVAL_FREQ = 500


def eval_hook(data):
    """Compute eval loss on a large fresh sample."""
    tm = data["tm"]
    x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(x, x_hat, tm.importances).item()


def per_feature_hook(data):
    """Per-feature reconstruction MSE on one-hot inputs."""
    tm = data["tm"]
    eye = torch.eye(n_features, device=tm.device)
    x_hat = tm.ae(eye)[0]
    return (eye - x_hat).pow(2).sum(dim=-1).cpu().numpy()


_, hook_results = tm.fit(
    N_EPOCHS, 256, hooks=[eval_hook, per_feature_hook], hook_freq=EVAL_FREQ
)
eval_losses = hook_results[0]
per_feature_mse = hook_results[1]

# %%
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]
px.line(
    x=eval_epochs, y=eval_losses, labels={"x": "Epoch", "y": "Loss"}, title="Eval loss"
).show()

# %%
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

# %%
# Sample points, encode into hidden space, and visualise
with torch.no_grad():
    samples = dist.sample(512)
    hidden = tm.ae.encode(samples).cpu().numpy()
    samples_cpu = samples.cpu().numpy()

# Label each point by which features are active
active_feats = []
for row in samples_cpu:
    feats = [str(v) for v in range(n_features) if row[v] > 0]
    active_feats.append(",".join(feats) if feats else "none")

fig = go.Figure()
# fig.add_trace(
#     go.Scatter3d(
#         x=hidden[:, 0],
#         y=hidden[:, 1],
#         z=hidden[:, 2],
#         mode="markers",
#         marker=dict(size=2, opacity=0.6, color="gray"),
#         text=active_feats,
#         hovertemplate="Active features: %{text}<extra></extra>",
#         name="Encoded samples",
#     )
# )

# Overlay feature embedding vectors, coloured by position on the manifold
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
fig.add_trace(
    go.Scatter3d(
        x=W[0],
        y=W[1],
        z=W[2],
        mode="markers+text",
        marker=dict(
            size=6,
            color=angles,
            colorscale="HSV",
            colorbar=dict(title="φ"),
        ),
        text=[f"f{j}" for j in range(n_features)],
        textposition="top center",
        name="Features",
    )
)

# Connect consecutive features with lines to show manifold structure
for j in range(n_features):
    if j < n_features // 2:
        k = (j + 1) % (n_features // 2)
    else:
        k = n_features // 2 + ((j + 1) % (n_features // 2))
    fig.add_trace(
        go.Scatter3d(
            x=[W[0, j], W[0, k]],
            y=[W[1, j], W[1, k]],
            z=[W[2, j], W[2, k]],
            mode="lines",
            line=dict(width=1, color="gray"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

fig.update_layout(
    title="Manifold feature embeddings in hidden space",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig.show()

# %%
WtW = W.T @ W  # (n_features, n_features)
px.imshow(
    WtW,
    title="W^T W",
    labels=dict(x="Feature", y="Feature"),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
).show()

# %%
# --- Per-feature MSE over training (heatmap) ---
pf_arr = np.array(per_feature_mse)  # (n_eval_points, n_features)
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure(
    go.Heatmap(
        z=pf_arr.T,
        x=eval_epochs,
        y=np.arange(n_features),
        colorscale="Viridis",
        colorbar=dict(title="MSE"),
    )
)
fig.update_layout(
    title="Per-Feature Reconstruction MSE Over Training",
    xaxis_title="Epoch",
    yaxis_title="Feature index",
    height=500,
)
fig.show()

# %%
# --- Final per-feature MSE (bar chart) ---
final_pf = np.array(per_feature_mse[-1])
fig = go.Figure(go.Bar(x=np.arange(n_features), y=final_pf, opacity=0.7))
fig.update_layout(
    title="Final Per-Feature Reconstruction MSE",
    xaxis_title="Feature index",
    yaxis_title="MSE",
)
fig.show()

# %%
# --- SAE training ---
N_DICT = n_features + 4
SAE_STEPS = 65_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.05

print("Training SAE on TiedLinearRelu hidden representations...")
sae = SAESimple(
    n_latent=n_hidden,
    n_dict=N_DICT,
    l1_coef=SAE_L1,
    device=DEVICE,
).to(DEVICE)


def sae_data_fn(n: int) -> torch.Tensor:
    x = dist.sample(n).to(DEVICE)
    return tm.ae.encode(x)


sae_losses = sae.train_sae(
    data_fn=sae_data_fn,
    n_steps=SAE_STEPS,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
)

# %%
# --- SAE basic metrics ---
with torch.no_grad():
    test_x = dist.sample(10_000).to(DEVICE)
    test_hidden = tm.ae.encode(test_x)
    test_z = sae.encode(test_hidden)
    test_recon = sae.decode(test_z)

    l0 = (test_z > 0).float().sum(dim=-1).mean().item()

    ever_active = (test_z > 0).any(dim=0)
    n_dead = int((~ever_active).sum().item())
    n_alive = int(ever_active.sum().item())

    recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

    eye = torch.eye(n_features, device=DEVICE)
    h_eye = tm.ae.encode(eye)
    h_eye_recon = sae.decode(sae.encode(h_eye))
    per_feat_sae_mse = (h_eye - h_eye_recon).pow(2).sum(dim=-1).cpu().numpy()

    total_var = test_hidden.var(dim=0).sum().item()
    residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
    explained_var = 1 - residual_var / total_var

print(
    f"  L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
)

# %%
# --- SAE loss curve ---
px.line(
    y=sae_losses,
    labels={"x": "Step", "y": "Loss"},
    title="SAE Training Loss",
    log_y=True,
).show()

# %%
# --- SAE one-hot activations + diagonality ---
with torch.no_grad():
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (n_features, N_DICT)

cost = -sae_acts
feat_idx, dict_idx = linear_sum_assignment(cost)

matched_feats = set(feat_idx)
matched_dicts = set(dict_idx)
unmatched_feats = [f for f in range(n_features) if f not in matched_feats]
unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]
row_order = list(feat_idx) + unmatched_feats
col_order = list(dict_idx) + unmatched_dicts
sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]

n_matched = len(feat_idx)
diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
total_sum = sae_acts_matched.sum()
diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
print(f"Diagonality = {diagonality:.4f}  (diag={diag_sum:.2f}, total={total_sum:.2f})")

px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
    x=[f"d{d}" for d in col_order],
    y=[f"f{f}" for f in row_order],
    title=f"SAE one-hot activations (Hungarian matched, diag={diagonality:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
).show()

# %%
# --- MCC + detection metrics ---
with torch.no_grad():
    D = tm.W.detach()  # (n_hidden, n_features) — columns are feature directions
    W_dec_t = sae.W_dec.detach().T  # (n_hidden, N_DICT)
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim = (D_norm.T @ W_norm).abs().cpu().numpy()  # (n_features, N_DICT)
    mcc_feat_idx, mcc_dict_idx = linear_sum_assignment(-cos_sim)
    mcc = float(cos_sim[mcc_feat_idx, mcc_dict_idx].mean())

    det_x = dist.sample(50_000).to(DEVICE)
    det_hidden = tm.ae.encode(det_x)
    det_z = sae.encode(det_hidden)

    gt_active = det_x[:, mcc_feat_idx] > 0
    pred_active = det_z[:, mcc_dict_idx] > 0

    tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()
    tn = (~gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

    precision_per = tp / (tp + fp + 1e-8)
    recall_per = tp / (tp + fn + 1e-8)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-8)
    fpr_per = fp / (fp + tn + 1e-8)

    precision = float(precision_per.mean())
    recall = float(recall_per.mean())
    f1 = float(f1_per.mean())
    fpr = float(fpr_per.mean())
    confusion = np.array([[tn.sum(), fp.sum()], [fn.sum(), tp.sum()]])

# %%
# --- SAE summary ---
print("\n=== SAE Summary ===")
print(
    f"{'MSE↓':>10s}  {'L0↓':>6s}  {'Dead↓':>6s}  {'Alive↑':>6s}  "
    f"{'ExplVar↑':>8s}  {'Diag↑':>6s}  {'MCC↑':>6s}  {'Prec↑':>6s}  {'Rec↑':>6s}  {'F1↑':>6s}  {'FPR↓':>6s}"
)
print(
    f"{recon_mse:10.6f}  {l0:6.1f}  {n_dead:6d}  {n_alive:6d}  "
    f"{explained_var:8.4f}  {diagonality:6.4f}  {mcc:6.4f}  "
    f"{precision:6.4f}  {recall:6.4f}  {f1:6.4f}  {fpr:6.4f}"
)

# %%
# --- Confusion matrix ---
cm_norm = confusion / confusion.sum()
labels = ["Inactive", "Active"]
text = [
    [f"{confusion[r, c]:.0f}<br>({cm_norm[r, c]:.3f})" for c in range(2)]
    for r in range(2)
]
go.Figure(
    go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
    )
).update_layout(
    title="Confusion Matrix (aggregated over matched features)",
    xaxis_title="SAE Prediction",
    yaxis_title="Ground Truth",
    height=400,
    width=450,
).show()

# %%
# --- Per-feature detection metrics ---
feat_order = np.argsort(mcc_feat_idx)  # sort by feature index
x = np.arange(len(feat_order))

fig = make_subplots(
    rows=1, cols=4, subplot_titles=["Precision", "Recall (TPR)", "F1 Score", "FPR"]
)
for col, arr in enumerate([precision_per, recall_per, f1_per, fpr_per], start=1):
    fig.add_trace(
        go.Scatter(x=x, y=arr[feat_order], mode="lines", showlegend=False),
        row=1,
        col=col,
    )
    fig.update_xaxes(title_text="Feature index", row=1, col=col)

fig.update_layout(
    title="Per-Feature Detection Metrics",
    height=400,
    width=1400,
)
fig.show()

# %%
