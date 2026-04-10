# %%
import random

import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SimplexDistribution, SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# --- Paper-quality plot defaults ---
FONT = dict(family="Times New Roman, serif", size=24, color="#333333")
AXIS_STYLE = dict(
    showgrid=False,
    gridcolor="rgba(0,0,0,0.02)",
    gridwidth=1,
    zeroline=False,
    linecolor="#666666",
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickwidth=1,
    tickcolor="#666666",
    minor=dict(ticks="outside", ticklen=2),
    tickfont_size=18,
)
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=FONT,
    title_font_size=18,
    legend=dict(
        x=0.95,
        y=0.95,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        font_size=20,
        itemsizing="constant",
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(1)

# 3 simplices of size 3 → 9 features total
# simplex_sizes = [2, 2, 2]
# n_features = sum(simplex_sizes)
n_features = 9
n_hidden = 3


FACE_DIM = 1
N_FACES = n_features
p_active = 1 / (N_FACES)
faces = [(0, 1, 2), (1, 2, 3), (4, 5, 6), (6, 7, 8)]
# faces = [
#     (0, 1, 2),
#     (0, 2, 3),
#     (0, 3, 4),
#     (0, 4, 1),
#     (5, 1, 2),
#     (5, 2, 3),
#     (5, 3, 4),
#     (5, 4, 1),
# ]

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
losses, _ = tm.fit(40_000, 256, verbose=True)

# %%
fig_loss = px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training loss")
fig_loss.update_layout(**LAYOUT_DEFAULTS)
fig_loss.update_xaxes(**AXIS_STYLE)
fig_loss.update_yaxes(**AXIS_STYLE)
fig_loss

# %%
# Sample 256 points, encode into hidden space, and visualise
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)


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

# # Wall projection (shadow on h₀–h₂ plane at y-min) for depth cue
# y_wall = float(hidden[:, 1].min()) - 0.05
# fig2.add_trace(
#     go.Scatter3d(
#         x=hidden[:, 0],
#         y=[y_wall] * len(hidden),
#         z=hidden[:, 2],
#         mode="markers",
#         marker=dict(size=1.5, opacity=0.15, color="black"),
#         showlegend=False,
#         hoverinfo="skip",
#     )
# )

# # Shadow of face edges on the y-wall
# for i, face in enumerate(dist.faces):
#     edges = list(zip(face, face[1:])) + [(face[-1], face[0])]
#     for a, b in edges:
#         fig2.add_trace(
#             go.Scatter3d(
#                 x=[W[0, a], W[0, b]],
#                 y=[y_wall, y_wall],
#                 z=[W[2, a], W[2, b]],
#                 mode="lines",
#                 line=dict(width=1.5, color="rgba(0,0,0,0.15)"),
#                 showlegend=False,
#                 hoverinfo="skip",
#             )
#         )

# # Back wall projection (shadow on h₁–h₂ plane at x-min) for depth cue
# x_wall = float(hidden[:, 0].min()) - 0.05
# fig2.add_trace(
#     go.Scatter3d(
#         x=[x_wall] * len(hidden),
#         y=hidden[:, 1],
#         z=hidden[:, 2],
#         mode="markers",
#         marker=dict(size=1.5, opacity=0.15, color="black"),
#         showlegend=False,
#         hoverinfo="skip",
#     )
# )

# # Shadow of face edges on the x-wall
# for i, face in enumerate(dist.faces):
#     edges = list(zip(face, face[1:])) + [(face[-1], face[0])]
#     for a, b in edges:
#         fig2.add_trace(
#             go.Scatter3d(
#                 x=[x_wall, x_wall],
#                 y=[W[1, a], W[1, b]],
#                 z=[W[2, a], W[2, b]],
#                 mode="lines",
#                 line=dict(width=1.5, color="rgba(0,0,0,0.15)"),
#                 showlegend=False,
#                 hoverinfo="skip",
#             )
#         )

fig2.add_trace(
    go.Scatter3d(
        x=hidden[:, 0],
        y=hidden[:, 1],
        z=hidden[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color="black"),
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
        marker=dict(size=8, opacity=0.9, color="red"),
        text=[f"v{j}" for j in range(n_features)],
        textposition="top center",
        name="Vertices",
    )
)

# Draw filled triangles and edges for each face
face_colors = [
    "rgba(31,119,180,0.9)",
    "rgba(255,127,14,0.8)",
    "rgba(44,160,44,0.8)",
    "rgba(214,39,40,0.9)",
]
for i, face in enumerate(dist.faces):
    verts = list(face)
    # fig2.add_trace(
    #     go.Mesh3d(
    #         x=W[0, verts],
    #         y=W[1, verts],
    #         z=W[2, verts],
    #         i=[0], j=[1], k=[2],
    #         color=face_colors[i % len(face_colors)],
    #         flatshading=False,
    #         lighting=dict(ambient=0.5, diffuse=0.7, specular=0.9, fresnel=0.4),
    #         lightposition=dict(x=1000, y=1000, z=1000),
    #         showlegend=False,
    #         hoverinfo="skip",
    #     )
    # )
    edges = list(zip(face, face[1:])) + [(face[-1], face[0])]
    for a, b in edges:
        fig2.add_trace(
            go.Scatter3d(
                x=[W[0, a], W[0, b]],
                y=[W[1, a], W[1, b]],
                z=[W[2, a], W[2, b]],
                mode="lines",
                line=dict(width=2, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig2.update_layout(
    **LAYOUT_DEFAULTS,
    title="Encoded samples in hidden space",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        xaxis=dict(
            showticklabels=False,
            ticks="",
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.02)",
        ),
        yaxis=dict(
            showticklabels=False,
            ticks="",
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.02)",
        ),
        zaxis=dict(
            showticklabels=False,
            ticks="",
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.02)",
        ),
        aspectmode="cube",
        # camera=dict(
        #     eye=dict(x=1.4, y=0.8, z=-1.1),
        #     up=dict(x=0, y=0, z=1),
        # ),
    ),
    height=700,
)
fig2

# %%
WtW = (tm.W.T @ tm.W).detach().cpu().numpy()
fig_wtw = px.imshow(
    WtW,
    title="W^T W (feature cosine structure)",
    labels=dict(x="Feature", y="Feature"),
    x=[f"v{j}" for j in range(n_features)],
    y=[f"v{j}" for j in range(n_features)],
)
fig_wtw.update_layout(**LAYOUT_DEFAULTS)
fig_wtw

# %% --- SAE training ---
from occhio.sae.sae import SAESimple

N_DICT = 5
SAE_STEPS = 40_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.15

gen_sae = torch.Generator()
gen_sae.manual_seed(3)

sae = SAESimple(
    n_latent=n_hidden,
    n_dict=N_DICT,
    l1_coef=SAE_L1,
    device=DEVICE,
    generator=gen_sae,
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
fig_sae_loss = px.line(
    y=sae_losses,
    title="SAE Training Loss",
    labels={"x": "Step", "y": "Loss"},
    log_y=True,
)
fig_sae_loss.update_layout(**LAYOUT_DEFAULTS)
fig_sae_loss.update_xaxes(**AXIS_STYLE)
fig_sae_loss.update_yaxes(**AXIS_STYLE)
fig_sae_loss

# %%
# --- SAE metrics ---
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

    total_var = test_hidden.var(dim=0).sum().item()
    residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
    explained_var = 1 - residual_var / total_var

print(
    f"L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
)

# %%
# --- MCC: cosine similarity matching between SAE decoder and ground-truth features ---
with torch.no_grad():
    D = tm.W.detach()  # (n_hidden, n_features) — columns are feature directions
    W_dec_t = sae.W_dec.detach().T  # (n_hidden, N_DICT)
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim_raw = (D_norm.T @ W_norm).cpu().numpy()  # (n_features, N_DICT)

    feat_idx, dict_idx = linear_sum_assignment(-cos_sim_raw)
    mcc = float(np.abs(cos_sim_raw)[feat_idx, dict_idx].mean())
    mcc_signed = float(cos_sim_raw[feat_idx, dict_idx].mean())

print(f"MCC (|cos|)={mcc:.4f}  MCC (cos)={mcc_signed:.4f}")

# %%
# --- SAE activations on one-hot features (cosine-matched) ---
with torch.no_grad():
    eye = torch.eye(n_features, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (n_features, N_DICT)

matched_feats = set(feat_idx)
matched_dicts = set(dict_idx)
unmatched_feats = [f for f in range(n_features) if f not in matched_feats]
unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

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

fig_acts = px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (cosine matched)", y="Feature (cosine matched)"),
    x=col_labels,
    y=row_labels,
    title=f"SAE activations (cosine matched, diag={diagonality:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
)
fig_acts.update_layout(**LAYOUT_DEFAULTS)
fig_acts

# %%
# --- Dual: one-hot SAE latents → decoded feature activations (unmatched) ---
with torch.no_grad():
    # Compute mean non-zero activation per SAE latent
    avg_x = dist.sample(50_000).to(DEVICE)
    avg_z = sae.encode(tm.ae.encode(avg_x))
    # Median non-zero activation per SAE latent
    median_act_per_latent = torch.zeros(avg_z.shape[1], device=DEVICE)
    for j in range(avg_z.shape[1]):
        nonzero_vals = avg_z[:, j][avg_z[:, j] > 0]
        if len(nonzero_vals) > 0:
            median_act_per_latent[j] = nonzero_vals.quantile(0.6)
    avg_act_per_latent = median_act_per_latent
    print(avg_act_per_latent)
    eye_dict = torch.diag(avg_act_per_latent)

    decoded_acts = (
        tm.ae.decode(sae.decode(eye_dict)).cpu().numpy()
    )  # (N_DICT, n_features)
    decoded_acts = np.maximum(decoded_acts, 0)

fig_dual = px.imshow(
    decoded_acts,
    labels=dict(x="Feature", y="SAE dict element"),
    x=[f"v{f}" for f in range(n_features)],
    y=[f"d{d}" for d in range(N_DICT)],
    # title="Dual: one-hot SAE latent → ae.decode(sae.decode(·))",
    aspect="auto",
    color_continuous_scale="Blues",
)
fig_dual.update_layout(**LAYOUT_DEFAULTS)
fig_dual

# %% --- Activation-based matching (Hungarian on one-hot SAE activations) ---
feat_idx_act, dict_idx_act = linear_sum_assignment(-sae_acts)

matched_feats_act = set(feat_idx_act)
matched_dicts_act = set(dict_idx_act)
unmatched_feats_act = [f for f in range(n_features) if f not in matched_feats_act]
unmatched_dicts_act = [d for d in range(N_DICT) if d not in matched_dicts_act]

row_order_act = list(feat_idx_act) + unmatched_feats_act
col_order_act = list(dict_idx_act) + unmatched_dicts_act

sae_acts_matched_act = sae_acts[np.ix_(row_order_act, col_order_act)]
row_labels_act = [f"v{f}" for f in row_order_act]
col_labels_act = [f"d{d}" for d in col_order_act]

n_matched_act = len(feat_idx_act)
diag_sum_act = sum(sae_acts_matched_act[i, i] for i in range(n_matched_act))
total_sum_act = sae_acts_matched_act.sum()
diagonality_act = diag_sum_act / total_sum_act if total_sum_act > 0 else 0.0

# Also compute MCC using activation-based matching on the cosine matrix for comparison
cos_mcc_act = float(np.abs(cos_sim_raw)[feat_idx_act, dict_idx_act].mean())
cos_mcc_act_signed = float(cos_sim_raw[feat_idx_act, dict_idx_act].mean())

print(f"[Activation matching] Diagonality={diagonality_act:.4f}")
print(
    f"[Activation matching] MCC (|cos|)={cos_mcc_act:.4f}  MCC (cos)={cos_mcc_act_signed:.4f}"
)
print(f"[Cosine matching]     Diagonality={diagonality:.4f}")
print(f"[Cosine matching]     MCC (|cos|)={mcc:.4f}  MCC (cos)={mcc_signed:.4f}")


# %% --- Detection metrics (both matchings) ---
cos_sim_matched = cos_sim_raw[np.ix_(row_order, col_order)]

with torch.no_grad():
    b_enc_np = sae.b_enc.detach().cpu().numpy()[col_order]


cos_sim_matched_act = cos_sim_raw[np.ix_(row_order_act, col_order_act)]

with torch.no_grad():
    b_enc_np_act = sae.b_enc.detach().cpu().numpy()[col_order_act]

with torch.no_grad():
    det_x = dist.sample(50_000).to(DEVICE)
    det_hidden = tm.ae.encode(det_x)
    det_z = sae.encode(det_hidden)

for label, fi, di in [
    ("Cosine", feat_idx, dict_idx),
    ("Activation", feat_idx_act, dict_idx_act),
]:
    gt_active = det_x[:, fi] > 0
    pred_active = det_z[:, di] > 0

    tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    print(
        f"[{label:10s}] Precision={prec.mean():.4f}  Recall={rec.mean():.4f}  F1={f1.mean():.4f}"
    )

# %%
# --- Plot W embedding + matched SAE decoder dictionary ---
with torch.no_grad():
    W_dec_np = sae.W_dec.detach().cpu().numpy()  # (N_DICT, n_hidden)

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
        name="Samples",
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
        name="AE features",
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
        name="SAE dict",
    )
)

# SAE decoder bias
with torch.no_grad():
    b_dec_np = sae.b_dec.detach().cpu().numpy()
fig3.add_trace(
    go.Scatter3d(
        x=[b_dec_np[0]],
        y=[b_dec_np[1]],
        z=[b_dec_np[2]],
        mode="markers+text",
        marker=dict(size=8, color="green", symbol="cross"),
        text=["b_dec"],
        textposition="top center",
        name="SAE bias",
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

# Draw simplex edges
for i, face in enumerate(dist.faces):
    edges = list(zip(face, face[1:])) + [(face[-1], face[0])]
    for a, b in edges:
        fig3.add_trace(
            go.Scatter3d(
                x=[W[0, a], W[0, b]],
                y=[W[1, a], W[1, b]],
                z=[W[2, a], W[2, b]],
                mode="lines",
                line=dict(width=2, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig3.update_layout(
    **LAYOUT_DEFAULTS,
    title="AE embedding (W) vs matched SAE decoder dictionary",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig3

# %% --- Plot W embedding + activation-matched SAE decoder dictionary ---
matched_W_dec_act = W_dec_np[dict_idx_act]  # (n_matched, n_hidden)

fig4 = go.Figure()

fig4.add_trace(
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

fig4.add_trace(
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

fig4.add_trace(
    go.Scatter3d(
        x=matched_W_dec_act[:, 0],
        y=matched_W_dec_act[:, 1],
        z=matched_W_dec_act[:, 2],
        mode="markers+text",
        marker=dict(size=6, symbol="diamond", color="red"),
        text=[f"d{d}→v{f}" for f, d in zip(feat_idx_act, dict_idx_act)],
        textposition="bottom center",
        name="SAE dict (activation matched)",
    )
)

fig4.add_trace(
    go.Scatter3d(
        x=[b_dec_np[0]],
        y=[b_dec_np[1]],
        z=[b_dec_np[2]],
        mode="markers+text",
        marker=dict(size=8, color="green", symbol="cross"),
        text=["b_dec"],
        textposition="top center",
        name="SAE decoder bias",
    )
)

for i in range(len(feat_idx_act)):
    f, d = feat_idx_act[i], dict_idx_act[i]
    fig4.add_trace(
        go.Scatter3d(
            x=[W[0, f], matched_W_dec_act[i, 0]],
            y=[W[1, f], matched_W_dec_act[i, 1]],
            z=[W[2, f], matched_W_dec_act[i, 2]],
            mode="lines",
            line=dict(width=1, color="rgba(255,0,0,0.3)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

fig4.update_layout(
    **LAYOUT_DEFAULTS,
    title="AE embedding (W) vs activation-matched SAE decoder dictionary",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig4

# %%
