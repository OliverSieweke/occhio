# %%
"""
Train a ToyModel on HierarchicalPairs with beta-coupled magnitudes,
then train an SAE on the learned latent space.
"""

from occhio.distributions.correlated import HierarchicalPairs
from occhio.model_grid import ModelGrid, Axis
from occhio.sae import SAESimple
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
torch.set_printoptions(3, sci_mode=False)
gen = torch.Generator()
gen.manual_seed(3)

N_FEAT = 2
N_HIDDEN = 2

dist = HierarchicalPairs(
    n_features=N_FEAT,
    p_active=0.8,
    p_follow=0.6,
    beta=0.99,
    generator=gen,
)

# %%
ae = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen)
tm = ToyModel(dist, ae, importances=0.99 ** torch.arange(N_FEAT))
losses = tm.fit(25_000, verbose=True)[0]

# %%
px.line(losses, title="ToyModel loss")

# %%
W = tm.W  # (n_hidden, n_features)
for p in range(N_FEAT // 2):
    w_parent = W[:, 2 * p]
    w_child = W[:, 2 * p + 1]
    cos = torch.dot(w_parent, w_child) / (w_parent.norm() * w_child.norm())
    angle = torch.acos(cos.clamp(-1, 1)).item() * 180 / torch.pi
    print(
        f"Pair {p}: f{2 * p}–f{2 * p + 1}  angle = {angle:.1f}°  cos = {cos.item():.3f}"
    )

# %%
emb = tm.W.detach().numpy()
px.scatter(
    x=emb[0],
    y=emb[1],
    hover_name=[f"f{i}" for i in range(N_FEAT)],
    color=[f"f{i}" for i in range(N_FEAT)],
    title="Feature embeddings",
)

# %%
gen_sae = torch.Generator()
gen_sae.manual_seed(4)

sae = SAESimple(N_HIDDEN, N_FEAT + 4, l1_coef=0.05, generator=gen_sae)
sae_losses = sae.train_sae(tm.sample_latent, 20_000)

px.line(sae_losses, title="SAE loss")

# %%
samples = dist.sample(512)
embedded = tm.encode(samples).detach().numpy().T
reconstructed = sae.decode(sae.encode(tm.encode(samples))).detach().numpy().T

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=embedded[0], y=embedded[1], mode="markers", name="Ground Truth")
)
fig.add_trace(
    go.Scatter(x=reconstructed[0], y=reconstructed[1], mode="markers", name="SAE")
)
fig.update_layout(title="Latent space: ground truth vs SAE reconstruction")
fig

# %%
patterns = torch.eye(N_FEAT)
encoded_patterns = sae.encode(tm.encode(patterns)).detach().numpy()
px.imshow(
    encoded_patterns,
    title="Feature -> SAE Dictionary",
    labels=dict(x="Dictionary Dim", y="Feature"),
)

# %% --- Feature embeddings + matched SAE decoder + decoder bias ---

feat_idx, dict_idx = linear_sum_assignment(-encoded_patterns)
matched_W_dec = sae.W_dec.detach().numpy()[dict_idx]  # (n_matched, n_hidden)
b_dec_np = sae.b_dec.detach().numpy()

fig_match = go.Figure()

# Lines from origin to feature embeddings
for j in range(N_FEAT):
    fig_match.add_trace(
        go.Scatter(
            x=[0, emb[0, j]],
            y=[0, emb[1, j]],
            mode="lines",
            line=dict(width=3, color="rgba(0,0,255,0.3)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Feature embeddings
fig_match.add_trace(
    go.Scatter(
        x=emb[0],
        y=emb[1],
        mode="markers+text",
        marker=dict(size=12, color="blue"),
        text=[f"f{i}" for i in range(N_FEAT)],
        textposition="top center",
        name="AE features (W)",
    )
)

# Matched SAE decoder dictionary elements
fig_match.add_trace(
    go.Scatter(
        x=matched_W_dec[:, 0],
        y=matched_W_dec[:, 1],
        mode="markers+text",
        marker=dict(size=12, symbol="diamond", color="red"),
        text=[f"d{d}→f{f}" for f, d in zip(feat_idx, dict_idx)],
        textposition="bottom center",
        name="SAE dict (matched)",
    )
)

# Lines connecting matched pairs
for i in range(len(feat_idx)):
    f = feat_idx[i]
    fig_match.add_trace(
        go.Scatter(
            x=[emb[0, f], matched_W_dec[i, 0]],
            y=[emb[1, f], matched_W_dec[i, 1]],
            mode="lines",
            line=dict(width=3, color="rgba(255,0,0,0.3)", dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# f0 + f1 sum vector
f0f1_sum = emb[:, 0] + emb[:, 1]
fig_match.add_trace(
    go.Scatter(
        x=[0, f0f1_sum[0]],
        y=[0, f0f1_sum[1]],
        mode="lines",
        line=dict(width=3, color="rgba(128,0,128,0.3)"),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig_match.add_trace(
    go.Scatter(
        x=[f0f1_sum[0]],
        y=[f0f1_sum[1]],
        mode="markers+text",
        marker=dict(size=12, symbol="star", color="purple"),
        text=["f0+f1"],
        textposition="top center",
        name="f0 + f1",
    )
)

# SAE decoder bias
fig_match.add_trace(
    go.Scatter(
        x=[b_dec_np[0]],
        y=[b_dec_np[1]],
        mode="markers+text",
        marker=dict(size=12, color="green", symbol="cross"),
        text=["b_dec"],
        textposition="top center",
        name="SAE decoder bias",
    )
)

mcc = float(np.abs(encoded_patterns)[feat_idx, dict_idx].mean())
fig_match.update_layout(
    title=f"Feature embeddings vs matched SAE decoder (MCC={mcc:.3f})",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=600,
)
fig_match

# %%
n_pairs = N_FEAT // 2
rows = []
labels = []
for p in range(n_pairs):
    parent_idx = 2 * p
    child_idx = 2 * p + 1

    parent_only = torch.zeros(N_FEAT)
    parent_only[parent_idx] = 0.9
    rows.append(parent_only)
    labels.append(f"f{parent_idx} (parent)")

    both = torch.zeros(N_FEAT)
    both[parent_idx] = 0.9
    both[child_idx] = 0.9
    rows.append(both)
    labels.append(f"f{parent_idx}+f{child_idx} (pair)")

patterns = torch.stack(rows)
encoded = sae.encode(ae.encode(patterns)).detach().numpy()
px.imshow(
    encoded,
    title="Parent vs Parent+Child -> SAE activations",
    labels=dict(x="Dictionary Dim", y="Pattern"),
    y=labels,
    color_continuous_scale="Reds",
)

# %%
# Phase transition: angle between parent-child pairs vs beta

# %%
BETA_VALUES = torch.linspace(0.0, 1.0, 32)


def create_model(params: dict):
    gen_grid = torch.Generator()
    gen_grid.manual_seed(3)
    dist = HierarchicalPairs(
        n_features=N_FEAT,
        p_active=2 / (N_FEAT + 1),
        p_follow=0.5,
        beta=params["Beta"].item(),
        generator=gen_grid,
    )
    ae_grid = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen_grid)
    return ToyModel(dist, ae_grid, importances=0.99 ** torch.arange(N_FEAT))


grid = ModelGrid(
    create_model,
    axes=[Axis(label="Beta", values=BETA_VALUES)],
)

# %%
grid.fit(25_000, verbose=True)

# %%
n_pairs = N_FEAT // 2
betas = BETA_VALUES.numpy()

angles = {p: [] for p in range(n_pairs)}

for beta_idx in range(len(BETA_VALUES)):
    tm_i: ToyModel = grid[beta_idx]  # type: ignore[assignment]
    W_i = tm_i.W
    for p in range(n_pairs):
        w_parent = W_i[:, 2 * p]
        w_child = W_i[:, 2 * p + 1]
        cos = torch.dot(w_parent, w_child) / (w_parent.norm() * w_child.norm() + 1e-8)
        angle = torch.acos(cos.clamp(-1, 1)).item() * 180 / torch.pi
        angles[p].append(angle)

fig = go.Figure()
for p in range(n_pairs):
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=angles[p],
            mode="lines+markers",
            name=f"Pair {p}: f{2 * p}–f{2 * p + 1}",
        )
    )
fig.update_layout(
    title="Parent-Child Pair Angle vs Beta",
    xaxis_title="Beta",
    yaxis_title="Angle (degrees)",
)
fig

# %%
# Phase transition: angle vs beta AND p_active (2D grid)

BETA_VALUES_2D = torch.linspace(0.0, 1.0, 9)
P_ACTIVE_VALUES = torch.logspace(np.log10(0.05), np.log10(0.95), 9)


def create_model_2d(params: dict):
    gen_grid = torch.Generator()
    gen_grid.manual_seed(6)
    dist = HierarchicalPairs(
        n_features=N_FEAT,
        p_active=params["p_active"].item(),
        p_follow=0.5,
        beta=params["Beta"].item(),
        generator=gen_grid,
    )
    ae_grid = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen_grid)
    return ToyModel(dist, ae_grid, importances=0.9 ** torch.arange(N_FEAT))


grid_2d = ModelGrid(
    create_model_2d,
    axes=[
        Axis(label="p_active", values=P_ACTIVE_VALUES),
        Axis(label="Beta", values=BETA_VALUES_2D),
    ],
)

# %%
grid_2d.fit(25_000, verbose=True)

# %%
n_pairs = N_FEAT // 2
n_pa = len(P_ACTIVE_VALUES)
n_beta = len(BETA_VALUES_2D)

# angle_grids[p] has shape (n_pa, n_beta)
angle_grids = {p: torch.zeros(n_pa, n_beta) for p in range(n_pairs)}

for pa_idx in range(n_pa):
    for beta_idx in range(n_beta):
        tm_ij: ToyModel = grid_2d[pa_idx, beta_idx]  # type: ignore[assignment]
        W_ij = tm_ij.W
        for p in range(n_pairs):
            w_parent = W_ij[:, 2 * p]
            w_child = W_ij[:, 2 * p + 1]
            cos = torch.dot(w_parent, w_child) / (
                w_parent.norm() * w_child.norm() + 1e-8
            )
            angle_grids[p][pa_idx, beta_idx] = (
                torch.acos(cos.clamp(-1, 1)).item() * 180 / torch.pi
            )

fig = make_subplots(
    rows=1,
    cols=n_pairs,
    subplot_titles=[f"Pair {p}: f{2 * p}–f{2 * p + 1}" for p in range(n_pairs)],
    shared_yaxes=True,
    horizontal_spacing=0.05,
)

beta_labels = [f"{v:.2f}" for v in BETA_VALUES_2D.tolist()]
pa_labels = [f"{v:.2f}" for v in P_ACTIVE_VALUES.tolist()]

for p in range(n_pairs):
    fig.add_trace(
        go.Heatmap(
            z=angle_grids[p].numpy(),
            x=beta_labels,
            y=pa_labels,
            coloraxis="coloraxis",
        ),
        row=1,
        col=p + 1,
    )
    fig.update_xaxes(title_text="Beta", row=1, col=p + 1)

fig.update_yaxes(title_text="p_active", row=1, col=1)
fig.update_layout(
    title="Parent-Child Pair Angle (degrees) vs Beta and p_active",
    coloraxis=dict(colorscale="Reds", colorbar_title="Angle (°)"),
    height=450,
    width=300 * n_pairs,
)
fig

# %% --- Phase transition: angle vs beta AND p_follow (averaged over seeds) ---

BETA_VALUES = torch.linspace(0.0, 1.0, 9)
P_FOLLOW_VALUES = torch.linspace(0.05, 0.95, 9)
TRAINING_SEEDS = [1, 2]

n_pairs = N_FEAT // 2
n_pf = len(P_FOLLOW_VALUES)
n_beta = len(BETA_VALUES)

# Accumulate angle grids across seeds
angle_grids_sum = {p: torch.zeros(n_pf, n_beta) for p in range(n_pairs)}

for seed in TRAINING_SEEDS:

    def create_model_2d(params: dict, _seed=seed):
        gen_grid = torch.Generator()
        gen_grid.manual_seed(_seed)
        dist = HierarchicalPairs(
            n_features=N_FEAT,
            p_active=0.25,
            p_follow=params["p_follow"].item(),
            beta=params["Beta"].item(),
            generator=gen_grid,
        )
        ae_grid = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen_grid)
        return ToyModel(dist, ae_grid, importances=0.9 ** torch.arange(N_FEAT))

    grid_2d = ModelGrid(
        create_model_2d,
        axes=[
            Axis(label="p_follow", values=P_FOLLOW_VALUES),
            Axis(label="Beta", values=BETA_VALUES_2D),
        ],
    )
    grid_2d.fit(20_000, verbose=True)

    for pf_idx in range(n_pf):
        for beta_idx in range(n_beta):
            tm_ij: ToyModel = grid_2d[pf_idx, beta_idx]  # type: ignore[assignment]
            W_ij = tm_ij.W
            for p in range(n_pairs):
                w_parent = W_ij[:, 2 * p]
                w_child = W_ij[:, 2 * p + 1]
                cos = torch.dot(w_parent, w_child) / (
                    w_parent.norm() * w_child.norm() + 1e-8
                )
                angle_grids_sum[p][pf_idx, beta_idx] += (
                    torch.acos(cos.clamp(-1, 1)).item() * 180 / torch.pi
                )

angle_grids_avg = {p: angle_grids_sum[p] / len(TRAINING_SEEDS) for p in range(n_pairs)}

# %%
fig = make_subplots(
    rows=1,
    cols=n_pairs,
    subplot_titles=[f"Pair {p}: f{2 * p}–f{2 * p + 1}" for p in range(n_pairs)],
    shared_yaxes=True,
    horizontal_spacing=0.05,
)

beta_labels = [f"{v:.2f}" for v in BETA_VALUES_2D.tolist()]
pf_labels = [f"{v:.2f}" for v in P_FOLLOW_VALUES.tolist()]

for p in range(n_pairs):
    fig.add_trace(
        go.Heatmap(
            z=angle_grids_avg[p].numpy(),
            x=beta_labels,
            y=pf_labels,
            coloraxis="coloraxis",
        ),
        row=1,
        col=p + 1,
    )
    fig.update_xaxes(title_text="Beta", row=1, col=p + 1)

fig.update_yaxes(title_text="p_follow", row=1, col=1)
fig.update_layout(
    title=f"Parent-Child Pair Angle (degrees) vs Beta and p_follow (avg over {len(TRAINING_SEEDS)} seeds)",
    coloraxis=dict(colorscale="Reds", colorbar_title="Angle (°)"),
    height=450,
    width=300 * n_pairs,
)
fig

# %%
