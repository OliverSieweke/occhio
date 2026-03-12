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
gen.manual_seed(7)

n_features = 32
length_scale = 0.7
manifold_dim = 2  # circle
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
losses, _ = tm.fit(100_000, 256)

# %%
px.line(
    y=losses[0::10], labels={"x": "Epoch", "y": "Loss"}, title="Training loss"
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
