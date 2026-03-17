# %%
"""Train TiedLinearRelu on a TorusDistribution and visualize the embedding space."""

import torch
import plotly.express as px
import plotly.graph_objects as go

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.manifold import TorusDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE).manual_seed(42)

n_features = 36  # 6x6 grid on T²
length_scale = 0.8
torus_dim = 2
n_hidden = 3

dist = TorusDistribution(
    n_features,
    length_scale=length_scale,
    torus_dim=torus_dim,
    magnitude_range=(0.9, 1.0),
    generator=gen,
    device=DEVICE,
)
ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%
N_EPOCHS = 50_000
EVAL_SAMPLES = 2**14
EVAL_FREQ = 500


def eval_hook(data):
    """Compute eval loss on a large fresh sample."""
    tm = data["tm"]
    x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(x, x_hat, tm.importances).item()


_, hook_results = tm.fit(N_EPOCHS, 256, hooks=[eval_hook], hook_freq=EVAL_FREQ)
eval_losses = hook_results[0]

# %%
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]
px.line(
    x=eval_epochs, y=eval_losses, labels={"x": "Epoch", "y": "Loss"}, title="Eval loss"
).show()

# %%
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

# Feature positions on the torus: (n_features, 2) angles
angles = dist.feature_angles.cpu().numpy()  # (n_features, 2)
grid_size = dist.grid_size

# %%
# 3D scatter of feature embeddings, colored by torus position
# Use angle on first torus dimension for color
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=W[0],
        y=W[1],
        z=W[2],
        mode="markers+text",
        marker=dict(
            size=6,
            color=angles[:, 0],
            colorscale="HSV",
            colorbar=dict(title="θ₁"),
        ),
        text=[f"{i}" for i in range(n_features)],
        textposition="top center",
        name="Features",
    )
)

# Connect features along the torus grid lines
for row in range(grid_size):
    for col in range(grid_size):
        idx = row * grid_size + col
        # Horizontal neighbor (same row, next col)
        idx_h = row * grid_size + (col + 1) % grid_size
        # Vertical neighbor (next row, same col)
        idx_v = ((row + 1) % grid_size) * grid_size + col
        for nb in [idx_h, idx_v]:
            fig.add_trace(
                go.Scatter3d(
                    x=[W[0, idx], W[0, nb]],
                    y=[W[1, idx], W[1, nb]],
                    z=[W[2, idx], W[2, nb]],
                    mode="lines",
                    line=dict(width=1, color="gray"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

fig.update_layout(
    title="Torus feature embeddings in hidden space",
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
# Color by second torus dimension
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=W[0],
        y=W[1],
        z=W[2],
        mode="markers",
        marker=dict(
            size=6,
            color=angles[:, 1],
            colorscale="Viridis",
            colorbar=dict(title="θ₂"),
        ),
        name="Features",
    )
)
fig.update_layout(
    title="Torus feature embeddings — colored by θ₂",
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
# W^T W gram matrix
WtW = W.T @ W
px.imshow(
    WtW,
    title="W^T W",
    labels=dict(x="Feature", y="Feature"),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
).show()

# %%
# %%
WWt = W @ W.T
px.imshow(
    WWt,
    title="W W.T",
    labels=dict(x="hidden", y="hidden"),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
).show()

# %%
