# %%
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from occhio.autoencoder import TiedLinearRelu
from occhio.sae.sae import SAESimple
from occhio.distributions import HypercubeDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(8)

n_features = 49
length_scale = 0.5
cube_dim = 2
n_hidden = 3

dist = HypercubeDistribution(
    n_features=n_features,
    length_scale=length_scale,
    cube_dim=cube_dim,
    magnitude_range=(0.7, 1.0),
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%
print(dist.sample(1))

# %%
N_EPOCHS = 100_000
EVAL_SAMPLES = 2**14
EVAL_FREQ = 500
BATCH_SIZE = 2048


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
    N_EPOCHS, BATCH_SIZE, hooks=[eval_hook, per_feature_hook], hook_freq=EVAL_FREQ
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

# Feature positions on [0,1] for colouring
positions = dist.feature_positions.cpu().numpy().squeeze()

# PCA via eigendecomposition of W @ W^T (hidden x hidden)
WWt = W @ W.T  # (n_hidden, n_hidden)
eigvals, eigvecs = np.linalg.eigh(WWt)
# Top 3 eigenvectors (eigh returns ascending order)
top3 = eigvecs[:, -1:-4:-1]  # (n_hidden, 3)
# Project feature vectors: W is (n_hidden, n_features), so top3.T @ W is (3, n_features)
W_proj = top3.T @ W  # (3, n_features)

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=W_proj[0],
        y=W_proj[1],
        z=W_proj[2],
        mode="markers+text",
        marker=dict(
            size=6,
            color=positions,
            colorscale="Viridis",
            colorbar=dict(title="position"),
        ),
        text=[f"f{j}" for j in range(n_features)],
        textposition="top center",
        name="Features",
    )
)

# Connect consecutive features with lines to show manifold structure
# for j in range(n_features - 1):
#     fig.add_trace(
#         go.Scatter3d(
#             x=[W_proj[0, j], W_proj[0, j + 1]],
#             y=[W_proj[1, j], W_proj[1, j + 1]],
#             z=[W_proj[2, j], W_proj[2, j + 1]],
#             mode="lines",
#             line=dict(width=1, color="gray"),
#             showlegend=False,
#             hoverinfo="skip",
#         )
#     )

fig.update_layout(
    title="Feature embeddings projected onto top-3 PCA directions of WW^T",
    scene=dict(
        xaxis_title="PC₁",
        yaxis_title="PC₂",
        zaxis_title="PC₃",
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
        z=pf_arr[1:].T,
        x=eval_epochs[1:],
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

# %% --- SAE training ---

N_DICT = n_features + 2
SAE_STEPS = 50_000
SAE_BATCH = 2048
SAE_LR = 3e-4
SAE_L1 = 0.05

sae = SAESimple(
    n_latent=n_hidden,
    n_dict=N_DICT,
    l1_coef=SAE_L1,
    device=DEVICE,
).to(DEVICE)


def sae_data_fn(n: int) -> torch.Tensor:
    raw = dist.sample(n)
    x = raw[0] if isinstance(raw, tuple) else raw
    return tm.ae.encode(x.to(DEVICE))


sae_losses = sae.train_sae(
    data_fn=sae_data_fn,
    n_steps=SAE_STEPS,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
)

# %%
px.line(
    y=sae_losses,
    title="SAE Training Loss",
    log_y=True,
    labels={"x": "Step", "y": "Loss"},
).show()

# %% --- Samples overlaid with SAE decoder vectors (3D) ---
with torch.no_grad():
    samples = dist.sample(1024)
    x = samples[0] if isinstance(samples, tuple) else samples
    hidden = tm.ae.encode(x.to(DEVICE)).cpu().numpy()

    sae_W_dec = sae.W_dec.detach().cpu().numpy()  # (N_DICT, n_hidden)
    W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

fig3 = go.Figure()

# Encoded samples
fig3.add_trace(
    go.Scatter3d(
        x=hidden[:, 0],
        y=hidden[:, 1],
        z=hidden[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color="lightgray"),
        name="Encoded samples",
    )
)

# Ground-truth feature vectors (ToyModel W columns)
for j in range(n_features):
    fig3.add_trace(
        go.Scatter3d(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            z=[0, W[2, j]],
            mode="lines+markers",
            line=dict(width=2, color="blue"),
            marker=dict(size=3),
            name="GT features" if j == 0 else None,
            legendgroup="gt",
            showlegend=(j == 0),
            hovertext=f"GT f{j}",
            hoverinfo="text",
        )
    )

# SAE decoder vectors (endpoint only)
for d in range(N_DICT):
    fig3.add_trace(
        go.Scatter3d(
            x=[sae_W_dec[d, 0]],
            y=[sae_W_dec[d, 1]],
            z=[sae_W_dec[d, 2]],
            mode="markers",
            marker=dict(size=5, symbol="diamond", color="red"),
            name="SAE dict" if d == 0 else None,
            legendgroup="sae",
            showlegend=(d == 0),
            hovertext=f"SAE d{d}",
            hoverinfo="text",
        )
    )

fig3.update_layout(
    title="Ground Truth vs SAE Decoder Directions",
    scene=dict(
        xaxis_title="h₀",
        yaxis_title="h₁",
        zaxis_title="h₂",
        aspectmode="cube",
    ),
    height=700,
)
fig3.show()

# %% --- Samples vs SAE reconstruction (PCA projected) ---
with torch.no_grad():
    hidden_t = tm.ae.encode(x.to(DEVICE))
    reconstructed = sae.decode(sae.encode(hidden_t)).cpu().numpy()
    hidden_np = hidden_t.cpu().numpy()

# Project both onto the same PCA directions computed earlier
hidden_pca = hidden_np @ top3  # (n_samples, 3)
recon_pca = reconstructed @ top3  # (n_samples, 3)

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter3d(
        x=hidden_pca[:, 0],
        y=hidden_pca[:, 1],
        z=hidden_pca[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.5, color="blue"),
        name="Original (encoded)",
    )
)

fig4.add_trace(
    go.Scatter3d(
        x=recon_pca[:, 0],
        y=recon_pca[:, 1],
        z=recon_pca[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.5, color="red"),
        name="SAE encode→decode",
    )
)

fig4.update_layout(
    title="Hidden Samples vs SAE Reconstruction (PCA projected)",
    scene=dict(
        xaxis_title="PC₁",
        yaxis_title="PC₂",
        zaxis_title="PC₃",
        aspectmode="cube",
    ),
    height=700,
)
fig4.show()

# %% --- Per-feature SAE activations ---
with torch.no_grad():
    eye = torch.eye(n_features, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (n_features, N_DICT)

px.imshow(
    sae_acts,
    labels=dict(x="SAE dict element", y="Feature"),
    x=[f"d{d}" for d in range(N_DICT)],
    y=[f"f{j}" for j in range(n_features)],
    title="SAE activations per one-hot feature",
    aspect="auto",
).show()

# %% --- SAE activations on real samples ---
with torch.no_grad():
    real_samples = dist.sample(8)
    real_sae_acts = sae.encode(tm.ae.encode(real_samples.to(DEVICE))).cpu().numpy()

px.imshow(
    real_sae_acts,
    labels=dict(x="SAE dict element", y="Sample"),
    x=[f"d{d}" for d in range(N_DICT)],
    y=[f"s{i}" for i in range(8)],
    title="SAE activations on sampled inputs",
    aspect="auto",
).show()

# %% --- SAE dict elements decoded to feature space ---
with torch.no_grad():
    eye_dict = torch.eye(N_DICT, device=DEVICE)
    decoded_features = (
        tm.ae.decode(sae.decode(eye_dict)).cpu().numpy()
    )  # (N_DICT, n_features)

px.imshow(
    decoded_features,
    labels=dict(x="Feature", y="SAE dict element"),
    x=[f"f{j}" for j in range(n_features)],
    y=[f"d{d}" for d in range(N_DICT)],
    title="SAE dict elements decoded to feature space",
    aspect="auto",
).show()
# %%
