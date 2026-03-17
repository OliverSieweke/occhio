# %%
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import ManifoldDistribution, DistributionStack
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(6)

n_features = 40
length_scale = 1.5
manifold_dim = 1
n_hidden = 3

dist = DistributionStack(
    [
        ManifoldDistribution(
            n_features // 2,
            length_scale=length_scale,
            manifold_dim=manifold_dim,
            magnitude_range=(0.9, 1.0),
            generator=gen,
            device=DEVICE,
        )
        for i in range(2)
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
