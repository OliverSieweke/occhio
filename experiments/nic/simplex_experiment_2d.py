# %%
import torch
import plotly.express as px
import plotly.graph_objects as go

from occhio.autoencoder import TiedLinearRelu
from occhio.sae.sae import SAESimple
from occhio.distributions import SimplexDistribution, SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(5)

n_features = 7
p_active = 1 / (n_features + 1)
n_hidden = 2

dist = SimplicialComplexDistribution(
    n_vertices=n_features,
    faces=[(0, 1), (2, 3), (4, 5), (5, 6)],
    p_active=p_active,
    sampling_mode="sparse",
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

# %% --- SAE training ---

N_DICT = n_features + 2
SAE_STEPS = 50_000
SAE_BATCH = 512
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

# %% --- Samples overlaid with SAE decoder vectors ---
with torch.no_grad():
    samples = dist.sample(1024)
    x = samples[0] if isinstance(samples, tuple) else samples
    hidden = tm.ae.encode(x.to(DEVICE)).cpu().numpy()

    # SAE decoder directions in hidden space
    sae_W_dec = sae.W_dec.detach().cpu().numpy()  # (N_DICT, n_hidden)

    # Ground-truth ToyModel encoder columns
    W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

fig3 = go.Figure()

# Encoded samples
fig3.add_trace(
    go.Scatter(
        x=hidden[:, 0],
        y=hidden[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.4, color="lightgray"),
        name="Encoded samples",
    )
)

# Ground-truth feature vectors (ToyModel W columns)
for j in range(n_features):
    fig3.add_trace(
        go.Scatter(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            mode="lines+markers",
            line=dict(width=2, color="blue"),
            marker=dict(size=6),
            name=f"GT v{j}" if j == 0 else None,
            legendgroup="gt",
            showlegend=(j == 0),
            hovertext=f"GT v{j}",
            hoverinfo="text",
        )
    )

# SAE decoder vectors (endpoint only, no line from origin)
for d in range(N_DICT):
    fig3.add_trace(
        go.Scatter(
            x=[sae_W_dec[d, 0]],
            y=[sae_W_dec[d, 1]],
            mode="markers",
            marker=dict(size=8, symbol="diamond", color="red"),
            name="SAE dict" if d == 0 else None,
            legendgroup="sae",
            showlegend=(d == 0),
            hovertext=f"SAE d{d}",
            hoverinfo="text",
        )
    )

fig3.update_layout(
    title="Ground Truth vs SAE Decoder Directions",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
)
fig3.show()

# %% --- Samples vs SAE reconstruction ---
with torch.no_grad():
    hidden_t = tm.ae.encode(x.to(DEVICE))
    reconstructed = sae.decode(sae.encode(hidden_t)).cpu().numpy()
    hidden_np = hidden_t.cpu().numpy()

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter(
        x=hidden_np[:, 0],
        y=hidden_np[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.5, color="blue"),
        name="Original (encoded)",
    )
)

fig4.add_trace(
    go.Scatter(
        x=reconstructed[:, 0],
        y=reconstructed[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.5, color="red"),
        name="SAE encode→decode",
    )
)

fig4.update_layout(
    title="Hidden Samples vs SAE Reconstruction",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
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
    y=[f"v{j}" for j in range(n_features)],
    title="SAE activations per one-hot feature",
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
    x=[f"v{j}" for j in range(n_features)],
    y=[f"d{d}" for d in range(N_DICT)],
    title="SAE dict elements decoded to feature space",
    aspect="auto",
).show()
# %%
