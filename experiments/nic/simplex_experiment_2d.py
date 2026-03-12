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
gen.manual_seed(5)

n_features = 10
p_active = 1 / (n_features + 1)
n_hidden = 2

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
losses, _ = tm.fit(25_000, 512)

# %%
px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training loss").show()

# %%
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

# Sample points, encode into hidden space, and visualise
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
    go.Scatter(
        x=hidden[:, 0],
        y=hidden[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.6, color="gray"),
        text=active_verts,
        hovertemplate="Active vertices: %{text}<extra></extra>",
        name="Encoded samples",
    )
)

# Overlay feature (vertex) vectors
fig2.add_trace(
    go.Scatter(
        x=W[0],
        y=W[1],
        mode="markers+text",
        marker=dict(size=8),
        text=[f"v{j}" for j in range(n_features)],
        textposition="top center",
        name="Vertices",
    )
)
for j in range(n_features):
    fig2.add_trace(
        go.Scatter(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            mode="lines",
            line=dict(width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Draw edges between vertices that share a face
for i, face in enumerate(dist.faces):
    for a, b in zip(face, face[1:]):
        fig2.add_trace(
            go.Scatter(
                x=[W[0, a], W[0, b]],
                y=[W[1, a], W[1, b]],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig2.update_layout(
    title="Encoded samples in hidden space",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
)
fig2.show()
# %%
