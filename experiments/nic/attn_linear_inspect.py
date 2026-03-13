# %%
"""Train a single AttnLinearAE and inspect its learned matrices."""

from occhio.distributions import SparseUniform, SimplicialComplexDistribution
from occhio.autoencoder import AttnLinearAE
from occhio.toy_model import ToyModel
import torch
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F_torch
from plotly.subplots import make_subplots

n_features = 6
n_hidden = 2
n_heads = 2
dict_size = 4
N_EPOCHS = 30_000
batch_size = 256
p_active = 0.1

# %%
device = torch.device("mps")
importances = torch.tensor([0.9**i for i in range(n_features)], device=device)
gen = torch.Generator(device)
gen.manual_seed(7)

# dist = SparseUniform(n_features, p_active, generator=gen, device=device)
dist = SimplicialComplexDistribution(
    6,
    [(0, 1), (2, 3), (4, 5)],
    p_active=p_active,
    sampling_mode="sparse",
    generator=gen,
    device=device,
)
ae = AttnLinearAE(
    n_features,
    n_hidden,
    n_heads=n_heads,
    dict_size=dict_size,
    generator=gen,
    device=device,
)
tm = ToyModel(dist, ae, importances=importances)

EVAL_BATCH = 2**14
HOOK_FREQ = 500


def eval_loss_hook(data: dict) -> float:
    """Evaluate loss on a fresh large batch."""
    _tm: ToyModel = data["tm"]
    raw = _tm.distribution.sample(EVAL_BATCH)
    ae_device = _tm.ae.device
    if isinstance(raw, tuple):
        raw = tuple(
            t.to(ae_device, non_blocking=True) if isinstance(t, torch.Tensor) else t
            for t in raw
        )
        x = raw[0]
    else:
        raw = raw.to(ae_device, non_blocking=True)
        x = raw
    x_hat = _tm.ae(x)[0]
    return _tm.ae.loss(raw, x_hat, _tm.importances).item()


_, hook_out = tm.fit(
    N_EPOCHS,
    verbose=True,
    batch_size=batch_size,
    track_losses=False,
    hooks=[eval_loss_hook],
    hook_freq=HOOK_FREQ,
)

# %%
# Loss curve
loss_epochs = list(range(0, N_EPOCHS, HOOK_FREQ)) + [N_EPOCHS - 1]
loss_fig = go.Figure()
loss_fig.add_trace(
    go.Scatter(
        x=loss_epochs,
        y=hook_out[0],
        mode="lines",
        name="eval loss",
        line=dict(color="#1f77b4"),
    )
)
loss_fig.update_layout(
    title=f"AttnLinearAE training loss (p={p_active}, heads={n_heads}, dict_size={dict_size})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    height=400,
    width=700,
)
loss_fig.show()

# %%
# Extract matrices to CPU numpy
encoder_projs = [ae.encoder_projs[h].detach().cpu().numpy() for h in range(n_heads)]
value_matrices = [ae.value_matrices[h].detach().cpu().numpy() for h in range(n_heads)]
W_mix = ae.W_mix.detach().cpu().numpy()
W_out = ae.W_out.detach().cpu().numpy()

print("=== Shapes ===")
for h in range(n_heads):
    print(f"encoder_projs[{h}]: {encoder_projs[h].shape}  (n_features, dict_size)")
    print(f"value_matrices[{h}]: {value_matrices[h].shape}  (dict_size, value_dim)")
print(f"W_mix: {W_mix.shape}  (n_hidden, n_hidden)")
print(f"W_out: {W_out.shape}  (n_hidden, n_features)")

# %%
# Composed encoder matrix per head: P_h @ V_h → (n_features, value_dim)
composed = [encoder_projs[h] @ value_matrices[h] for h in range(n_heads)]

# Effective encoder after mixing: [P_0 @ V_0 | P_1 @ V_1] @ (I + W_mix)
value_dim = n_hidden // n_heads
full_encoder = np.concatenate(composed, axis=1)  # (n_features, n_hidden)
effective_encoder = full_encoder @ (np.eye(n_hidden) + W_mix)  # (n_features, n_hidden)

print("\n=== Effective encoder (n_features, n_hidden) ===")
print(effective_encoder.round(3))

print("\n=== W_out (n_hidden, n_features) ===")
print(W_out.round(3))

print("\n=== W_mix ===")
print(W_mix.round(3))

# %%
# Heatmaps of all matrices
n_matrices = 2 * n_heads + 3  # encoder_projs, value_matrices, W_mix, W_out, effective
titles = (
    [f"encoder_projs[{h}]" for h in range(n_heads)]
    + [f"value_matrices[{h}]" for h in range(n_heads)]
    + ["W_mix", "W_out", "Effective encoder"]
)

fig = make_subplots(
    rows=1,
    cols=n_matrices,
    subplot_titles=titles,
    horizontal_spacing=0.03,
)

matrices = encoder_projs + value_matrices + [W_mix, W_out, effective_encoder]

for i, mat in enumerate(matrices):
    fig.add_trace(
        go.Heatmap(
            z=mat[::-1],
            colorscale="RdBu",
            zmid=0,
            showscale=i == 0,
        ),
        row=1,
        col=i + 1,
    )

fig.update_layout(
    title="AttnLinearAE learned matrices",
    height=400,
    width=250 * n_matrices,
)
fig.show()

# %%
# W embeddings (same as mrh.py)
W = tm.W.detach().cpu().numpy()
theta = np.linspace(0, 2 * np.pi, 100)
colors = ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#fde725"]

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line=dict(color="black", dash="dash", width=0.5),
        opacity=0.3,
        showlegend=False,
    )
)
for i in range(n_features):
    fig2.add_trace(
        go.Scatter(
            x=[W[0, i]],
            y=[W[1, i]],
            mode="markers+text",
            marker=dict(color=colors[i], size=12),
            text=[str(i)],
            textposition="top center",
            name=f"feature {i}",
        )
    )

fig2.update_layout(
    title=f"W embeddings (p={p_active})",
    xaxis=dict(range=[-1.5, 1.5], scaleanchor="y"),
    yaxis=dict(range=[-1.5, 1.5]),
    height=500,
    width=500,
)
fig2.show()

# %%
# Project dict keys into the same 2D space as W embeddings
# W is (n_hidden, n_features); key columns are in R^n_features
# W @ key_col gives 2D coordinates in hidden space
W_t = W.T  # (n_features, n_hidden) — pseudoinverse direction
W_pinv = np.linalg.pinv(W)  # (n_features, n_hidden) — proper left-inverse

head_colors = ["#e41a1c", "#377eb8"]

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line=dict(color="black", dash="dash", width=0.5),
        opacity=0.3,
        showlegend=False,
    )
)

# Feature embeddings
for i in range(n_features):
    fig3.add_trace(
        go.Scatter(
            x=[W[0, i]],
            y=[W[1, i]],
            mode="markers+text",
            marker=dict(color=colors[i], size=12),
            text=[str(i)],
            textposition="top center",
            name=f"feature {i}",
            legendgroup="features",
        )
    )

# Dict keys projected via pseudoinverse: pinv(W) has shape (n_features, n_hidden),
# but we want to go from n_features -> n_hidden, so use W @ key
for h in range(n_heads):
    P = encoder_projs[h]  # (n_features, dict_size)
    for j in range(dict_size):
        key = P[:, j]  # (n_features,)
        proj = W @ key  # (n_hidden,) = (2,)
        fig3.add_trace(
            go.Scatter(
                x=[proj[0]],
                y=[proj[1]],
                mode="markers+text",
                marker=dict(
                    color=head_colors[h],
                    size=10,
                    symbol="diamond",
                ),
                text=[f"h{h}k{j}"],
                textposition="bottom center",
                textfont=dict(size=8),
                name=f"head {h} key {j}",
                legendgroup=f"head{h}",
                showlegend=j == 0,
            )
        )

fig3.update_layout(
    title=f"Feature embeddings + dict keys projected to hidden space (p={p_active})",
    xaxis=dict(scaleanchor="y"),
    height=600,
    width=600,
)
fig3.show()

# %%
# Softmax weights per head when firing a single feature

feature_i = 0  # <-- change this to cycle through features

x_i = torch.zeros(1, n_features, device=device)
x_i[0, feature_i] = 1.0

head_colors_bar = ["#e41a1c", "#377eb8"]
fig4 = make_subplots(
    rows=1,
    cols=n_heads,
    subplot_titles=[f"Head {h}" for h in range(n_heads)],
    horizontal_spacing=0.1,
)

for h in range(n_heads):
    logits = x_i @ ae.encoder_projs[h]  # (1, dict_size)
    weights = F_torch.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
    fig4.add_trace(
        go.Bar(
            x=[f"k{j}" for j in range(dict_size)],
            y=weights,
            marker_color=head_colors_bar[h],
            name=f"head {h}",
            showlegend=False,
        ),
        row=1,
        col=h + 1,
    )
    fig4.update_yaxes(range=[0, 1], row=1, col=h + 1)

fig4.update_layout(
    title=f"Softmax key weights when firing feature {feature_i}",
    height=350,
    width=250 * n_heads,
)
fig4.show()

# %%
