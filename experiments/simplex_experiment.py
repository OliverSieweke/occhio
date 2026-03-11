# %%
import torch
import plotly.express as px
import plotly.graph_objects as go

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SimplexDistribution, SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(42)

# 3 simplices of size 3 → 9 features total
# simplex_sizes = [2, 2, 2]
# n_features = sum(simplex_sizes)
n_features = 10
p_active = 1 / (n_features + 1)
n_hidden = 3

# dist = SimplexDistribution(simplex_sizes, p_active, generator=gen, device=DEVICE)
dist = SimplicialComplexDistribution(
    n_vertices=n_features,
    faces=[(i, (i + 1) % n_features) for i in range(n_features)],
    p_active=p_active,
    sampling_mode="single",
    generator=gen,
    device=DEVICE,
)
ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%
losses, _ = tm.fit(25_000, 256)

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
        marker=dict(size=2, opacity=0.6, color="gray"),
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
for j in range(n_features):
    fig2.add_trace(
        go.Scatter3d(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            z=[0, W[2, j]],
            mode="lines",
            line=dict(width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Draw edges between vertices that share a face
pal = px.colors.qualitative.Set1
for i, face in enumerate(dist.faces):
    for a, b in zip(face, face[1:]):
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
