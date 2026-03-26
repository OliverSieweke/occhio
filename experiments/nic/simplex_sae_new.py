# %%
import random

import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(6)

n_features = 9
n_hidden = 3

random.seed(7)
FACE_DIM = 1
N_FACES = n_features
p_active = 1 / (N_FACES)

faces = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

dist = SimplicialComplexDistribution(
    n_vertices=n_features,
    faces=faces,
    p_active=p_active,
    sampling_mode="sparse",
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%
losses, _ = tm.fit(25_000, 256, verbose=True)

# %%
px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training loss").show()

# %%
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)


# %%
# Sample 256 points, encode into hidden space, and visualise
with torch.no_grad():
    samples = dist.sample(512)
    hidden = tm.ae.encode(samples).cpu().numpy()
    samples_cpu = samples.cpu().numpy()

# Label each point by which vertices are active
active_verts = []
for row in samples_cpu:
    verts = [str(v) for v in range(n_features) if row[v] > 0]
    active_verts.append(",".join(verts) if verts else "none")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter3d(
        x=hidden[:, 0],
        y=hidden[:, 1],
        z=hidden[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color="gray"),
        text=active_verts,
        hovertemplate="Active vertices: %{text}<extra></extra>",
        name="Encoded samples",
    )
)

# Overlay feature (vertex) vectors
fig2.add_trace(
    go.Scatter3d(
        x=W[0],
        y=W[1],
        z=W[2],
        mode="markers+text",
        marker=dict(size=6),
        text=[f"v{j}" for j in range(n_features)],
        textposition="top center",
        name="Vertices",
    )
)
# for j in range(n_features):
#     fig2.add_trace(
#         go.Scatter3d(
#             x=[0, W[0, j]],
#             y=[0, W[1, j]],
#             z=[0, W[2, j]],
#             mode="lines",
#             line=dict(width=2),
#             showlegend=False,
#             hoverinfo="skip",
#         )
#     )

# Draw edges between vertices that share a face
for i, face in enumerate(dist.faces):
    edges = list(zip(face, face[1:])) + [(face[-1], face[0])]
    for a, b in edges:
        fig2.add_trace(
            go.Scatter3d(
                x=[W[0, a], W[0, b]],
                y=[W[1, a], W[1, b]],
                z=[W[2, a], W[2, b]],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig2.update_layout(
    title="Encoded samples in hidden space",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig2.show()

# %%
WtW = (tm.W.T @ tm.W).detach().cpu().numpy()
px.imshow(
    WtW,
    title="W^T W (feature cosine structure)",
    labels=dict(x="Feature", y="Feature"),
    x=[f"v{j}" for j in range(n_features)],
    y=[f"v{j}" for j in range(n_features)],
).show()

# %% --- SAE training ---
from occhio.sae.sae import SimplexSAE

SAE_STEPS = 25_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.15

sae = SimplexSAE(
    d_input=n_hidden,
    n_simplexes=3,
    d_local=3,
    lambda_1=0.1,
    lambda_2=0.1,
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

# %%
px.line(
    y=sae_losses,
    title="SAE Training Loss",
    labels={"x": "Step", "y": "Loss"},
    log_y=True,
).show()

# %% --- SAE metrics ---
with torch.no_grad():
    test_x = dist.sample(10_000).to(DEVICE)
    test_hidden = tm.ae.encode(test_x)
    test_z = sae.encode(test_hidden)
    test_recon = sae.decode(test_z)

    # Strip gate entries (last K) to get only dictionary activations
    test_s = test_z[:, : -sae.n_simplexes]
    l0 = (test_s > 0).float().sum(dim=-1).mean().item()
    ever_active = (test_s > 0).any(dim=0)
    n_dead = int((~ever_active).sum().item())
    n_alive = int(ever_active.sum().item())
    recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

    total_var = test_hidden.var(dim=0).sum().item()
    residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
    explained_var = 1 - residual_var / total_var

print(
    f"L0={l0:.1f}  Dead={n_dead}/{sae.d_dict}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
)

# %% --- MCC: cosine similarity matching between SAE decoder and ground-truth features ---
with torch.no_grad():
    D = tm.W.detach()  # (n_hidden, n_features) — columns are feature directions
    W_dec_t = torch.cat(
        [sae.D[k].detach() for k in range(sae.n_simplexes)], dim=1
    )  # (d_input, d_dict)
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim_raw = (D_norm.T @ W_norm).cpu().numpy()  # (n_features, N_DICT)

    feat_idx, dict_idx = linear_sum_assignment(-cos_sim_raw)
    mcc = float(np.abs(cos_sim_raw)[feat_idx, dict_idx].mean())
    mcc_signed = float(cos_sim_raw[feat_idx, dict_idx].mean())

print(f"MCC (|cos|)={mcc:.4f}  MCC (cos)={mcc_signed:.4f}")

# %% --- SAE activations on one-hot features (cosine-matched) ---
with torch.no_grad():
    eye = torch.eye(n_features, device=DEVICE)
    full_z = sae.encode(tm.ae.encode(eye))
    sae_acts = full_z[:, : -sae.n_simplexes].cpu().numpy()  # (n_features, d_dict)

matched_feats = set(feat_idx)
matched_dicts = set(dict_idx)
unmatched_feats = [f for f in range(n_features) if f not in matched_feats]
unmatched_dicts = [d for d in range(sae.d_dict) if d not in matched_dicts]

row_order = list(feat_idx) + unmatched_feats
col_order = list(dict_idx) + unmatched_dicts

sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
row_labels = [f"v{f}" for f in row_order]
col_labels = [f"d{d}" for d in col_order]

n_matched = len(feat_idx)
diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
total_sum = sae_acts_matched.sum()
diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
print(f"Diagonality={diagonality:.4f}")

px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (cosine matched)", y="Feature (cosine matched)"),
    x=col_labels,
    y=row_labels,
    title=f"SAE one-hot activations (cosine matched, diag={diagonality:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
).show()

# %% --- Cosine similarity matrix (matched order) + encoder bias ---
from plotly.subplots import make_subplots

cos_sim_matched = cos_sim_raw[np.ix_(row_order, col_order)]

with torch.no_grad():
    b_enc_np = (
        torch.cat([sae.b_enc[k].detach() for k in range(sae.n_simplexes)])
        .cpu()
        .numpy()[col_order]
    )

fig_cos = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.85, 0.15],
    shared_xaxes=True,
    vertical_spacing=0.02,
)

fig_cos.add_trace(
    go.Heatmap(
        z=cos_sim_matched,
        x=col_labels,
        y=row_labels,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="cos sim"),
    ),
    row=1,
    col=1,
)

fig_cos.add_trace(
    go.Bar(
        x=col_labels,
        y=b_enc_np,
        marker_color="green",
        name="b_enc",
    ),
    row=2,
    col=1,
)

fig_cos.update_layout(
    title=f"Cosine similarity (matched, mean={mcc_signed:.3f})",
    height=700,
    showlegend=False,
)
fig_cos.update_yaxes(title_text="Feature (matched)", row=1, col=1)
fig_cos.update_yaxes(title_text="b_enc", row=2, col=1)
fig_cos.update_xaxes(title_text="SAE dict element (matched)", row=2, col=1)
fig_cos.show()

# %% --- Detection metrics ---
with torch.no_grad():
    det_x = dist.sample(50_000).to(DEVICE)
    det_hidden = tm.ae.encode(det_x)
    det_z = sae.encode(det_hidden)
    det_s = det_z[:, : -sae.n_simplexes]  # strip gates

    gt_active = det_x[:, feat_idx] > 0
    pred_active = det_s[:, dict_idx] > 0

    tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

print(f"Precision={prec.mean():.4f}  Recall={rec.mean():.4f}  F1={f1.mean():.4f}")

# %% --- Plot W embedding + matched SAE decoder dictionary ---
with torch.no_grad():
    W_dec_np = (
        torch.cat([sae.D[k].detach() for k in range(sae.n_simplexes)], dim=1)
        .T.cpu()
        .numpy()
    )  # (d_dict, n_hidden)

# Only keep matched dictionary elements
matched_W_dec = W_dec_np[dict_idx]  # (n_matched, n_hidden)

fig3 = go.Figure()

# Encoded samples
with torch.no_grad():
    samples = dist.sample(512)
    hidden = tm.ae.encode(samples).cpu().numpy()
    samples_cpu = samples.cpu().numpy()

active_verts = []
for row in samples_cpu:
    verts = [str(v) for v in range(n_features) if row[v] > 0]
    active_verts.append(",".join(verts) if verts else "none")

fig3.add_trace(
    go.Scatter3d(
        x=hidden[:, 0],
        y=hidden[:, 1],
        z=hidden[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color="gray"),
        text=active_verts,
        hovertemplate="Active vertices: %{text}<extra></extra>",
        name="Encoded samples",
    )
)

# Autoencoder W columns (feature embeddings)
fig3.add_trace(
    go.Scatter3d(
        x=W[0],
        y=W[1],
        z=W[2],
        mode="markers+text",
        marker=dict(size=6, color="blue"),
        text=[f"v{j}" for j in range(n_features)],
        textposition="top center",
        name="AE features (W)",
    )
)

# Matched SAE decoder dictionary elements
fig3.add_trace(
    go.Scatter3d(
        x=matched_W_dec[:, 0],
        y=matched_W_dec[:, 1],
        z=matched_W_dec[:, 2],
        mode="markers+text",
        marker=dict(size=6, symbol="diamond", color="red"),
        text=[f"d{d}→v{f}" for f, d in zip(feat_idx, dict_idx)],
        textposition="bottom center",
        name="SAE dict (matched)",
    )
)

# SAE decoder biases (vantage points)
with torch.no_grad():
    b_dec_all = (
        torch.stack([sae.b_dec[k].detach() for k in range(sae.n_simplexes)])
        .cpu()
        .numpy()
    )  # (K, n_hidden)
fig3.add_trace(
    go.Scatter3d(
        x=b_dec_all[:, 0],
        y=b_dec_all[:, 1],
        z=b_dec_all[:, 2],
        mode="markers+text",
        marker=dict(size=8, color="green", symbol="cross"),
        text=[f"b_dec[{k}]" for k in range(sae.n_simplexes)],
        textposition="top center",
        name="SAE vantage points",
    )
)

# Draw lines connecting matched pairs
for i in range(len(feat_idx)):
    f, d = feat_idx[i], dict_idx[i]
    fig3.add_trace(
        go.Scatter3d(
            x=[W[0, f], matched_W_dec[i, 0]],
            y=[W[1, f], matched_W_dec[i, 1]],
            z=[W[2, f], matched_W_dec[i, 2]],
            mode="lines",
            line=dict(width=1, color="rgba(255,0,0,0.3)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

fig3.update_layout(
    title="AE embedding (W) vs matched SAE decoder dictionary",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig3.show()

# %%
