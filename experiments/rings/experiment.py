# %%
# ── IMPORTS AND CONFIG ────────────────────────────────────────────────────
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu, MLPEncoder, TiedMLPEncoder
from occhio.distributions import SparseSpheres

from occhio.visualization import (
    plot_representation,
    plot_feature_geometry_3d,
    plot_geometry,
)


assert torch.backends.mps.is_available(), "MPS device is not available"
DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

# SparseSpheres params: 5 circles (S^1) tilted into 3D ambient → n_features = 5*3 = 15
K = 5
N = 1  # S^1 (circles)
M = 3  # ambient dim per feature (tilt from 2D into 3D)
N_FEATURES = K * M  # = 15
P_ACTIVE = 0.08
R = 1.0

# AE / training params
HIDDEN_DIM = 3
N_EPOCHS = 100_000
BATCH_SIZE = 256
LR = 3e-4
IMPORTANCE_BASE = 0.95

# Each circle i gets importance IMPORTANCE_BASE^(i+1), repeated M times for its coordinates
importances = torch.repeat_interleave(IMPORTANCE_BASE ** torch.arange(1, K + 1), M)

# %%
# ── DISTRIBUTION ───────────────────────────────────────────────────────
dist = SparseSpheres(
    k=K,
    n=N,
    m=M,
    p_active=P_ACTIVE,
    r=R,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)

samples, labels = dist.sample_with_labels(4)
print(f"SparseSpheres: {K} circles (S^{N}) in {M}D ambient, total dim={N_FEATURES}")
print(f"Tilt shapes: {dist.tilts.shape}")
print(f"Sample shape: {samples.shape}, Labels shape: {labels.shape}")


# %%
# ── AUTOENCODERS ─────────────────────────────────────────────────────────
def make_models() -> dict[str, ToyModel]:
    """Build the four architecture variants, each paired with a fresh distribution."""
    configs = {
        "LinearAE": lambda: MLPEncoder(
            embedding=[N_FEATURES, HIDDEN_DIM],
            unembedding=[HIDDEN_DIM, N_FEATURES],
        ),
        "TiedLinearAE": lambda: TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=HIDDEN_DIM,
        ),
        "MLPAE": lambda: MLPEncoder(
            embedding=[N_FEATURES, 64, 32, HIDDEN_DIM],
            unembedding=[HIDDEN_DIM, 32, 64, N_FEATURES],
        ),
        "TiedMLPAE": lambda: TiedMLPEncoder(
            dims=[N_FEATURES, 64, 32, HIDDEN_DIM],
        ),
    }

    models = {}
    for name, ae_factory in configs.items():
        dist_gen = torch.Generator().manual_seed(SEED)
        d = SparseSpheres(
            k=K,
            n=N,
            m=M,
            p_active=P_ACTIVE,
            r=R,
            generator=dist_gen,
        )
        ae = ae_factory()
        models[name] = ToyModel(
            distribution=d, ae=ae, device=DEVICE, importances=importances
        )
    return models


models = make_models()
for name, tm in models.items():
    n_params = sum(p.numel() for p in tm.ae.parameters())
    print(f"{name}: {n_params} parameters")

# %%
# ── TRAINING LOOP ─────────────────────────────────────────────────────────
loss_curves: dict[str, list[float]] = {}

for name, tm in models.items():
    print(f"\nTraining {name}...")
    losses, _ = tm.fit(
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.0,
        track_losses=True,
        verbose=True,
    )
    loss_curves[name] = losses
    print(f"  Final loss: {losses[-1]:.6f}")


# %%
# ── LOSS CURVE PLOT ───────────────────────────────────────────────────────
def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


loss_curves_path = "experiments/rings/figures/loss_curves.html"
ensure_dir_exists(loss_curves_path)

fig = go.Figure()

for name, losses in loss_curves.items():
    epochs = np.arange(1, len(losses) + 1)
    fig.add_trace(
        go.Scatter(
            x=epochs[::100],
            y=np.array(losses[::100]),
            mode="lines",
            name=name,
        )
    )

fig.update_layout(
    title=f"Reconstruction Loss: SparseSpheres (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    xaxis_title="Epoch",
    yaxis_title="MSE Loss",
    yaxis_type="log",
    template="plotly_white",
)
fig.write_html(loss_curves_path)
fig.show()

# %%
# ── EMBEDDING VISUALIZATION (3D) ────────────────────────────────────────
# Use return_labels=True to get ground-truth activation masks directly.
# Color by which single ring fired; multi-ring in grey.
N_VIS = 10_000
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

bottleneck_path = "experiments/rings/figures/bottleneck.html"
ensure_dir_exists(bottleneck_path)

model_names = list(models.keys())
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=model_names,
    specs=[
        [{"type": "scatter3d"}, {"type": "scatter3d"}],
        [{"type": "scatter3d"}, {"type": "scatter3d"}],
    ],
    vertical_spacing=0.05,
    horizontal_spacing=0.05,
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = idx // 2 + 1, idx % 2 + 1

    # Sample with ground-truth labels
    vis_dist = SparseSpheres(
        k=K,
        n=N,
        m=M,
        p_active=P_ACTIVE,
        r=R,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    samples, mask = vis_dist.sample_with_labels(N_VIS)
    samples = samples.to(DEVICE)

    n_active_per_sample = mask.sum(dim=1)
    is_single = n_active_per_sample == 1
    is_inactive = n_active_per_sample == 0
    is_multi = n_active_per_sample > 1
    single_ring_id = mask.float().argmax(dim=1)

    with torch.no_grad():
        z = tm.ae.encode(samples).cpu().numpy()

    # Zero-center on active samples
    active_mask = ~is_inactive.numpy()
    center = z[active_mask].mean(axis=0) if active_mask.any() else np.zeros(z.shape[1])
    z = z - center

    # Plot inactive in light grey
    inact = is_inactive.numpy()
    if inact.any():
        fig.add_trace(
            go.Scatter3d(
                x=z[inact, 0],
                y=z[inact, 1],
                z=z[inact, 2],
                mode="markers",
                marker=dict(color="black", size=1, opacity=0.15),
                name="inactive",
                showlegend=(idx == 0),
                legendgroup="inactive",
            ),
            row=row,
            col=col,
        )

    # Plot multi-ring in dark grey
    multi = is_multi.numpy()
    if multi.any():
        fig.add_trace(
            go.Scatter3d(
                x=z[multi, 0],
                y=z[multi, 1],
                z=z[multi, 2],
                mode="markers",
                marker=dict(color="#888888", size=1.5, opacity=0.1),
                name="multi-ring",
                showlegend=(idx == 0),
                legendgroup="multi",
            ),
            row=row,
            col=col,
        )

    # Plot single-ring samples colored by ring
    single = is_single.numpy()
    ring_ids = single_ring_id.numpy()
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=z[ring_mask, 0],
                    y=z[ring_mask, 1],
                    z=z[ring_mask, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j % len(COLORS)], size=2, opacity=0.6),
                    name=f"ring {j}",
                    showlegend=(idx == 0),
                    legendgroup=f"ring_{j}",
                ),
                row=row,
                col=col,
            )

    # Overlay ground-truth rings: sweep S^1 through tilt + encoder
    n_angles = 256
    theta = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
    circle_2d = R * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    with torch.no_grad():
        for j in range(K):
            Rj = dist.tilts[j].to(DEVICE)
            ring_in_ambient = circle_2d @ Rj.T
            full_input = torch.zeros(n_angles, N_FEATURES, device=DEVICE)
            full_input[:, j * M : (j + 1) * M] = ring_in_ambient
            zr = tm.ae.encode(full_input).cpu().numpy()
            zr = zr - center  # same centering as sample scatter
            zr = np.concatenate([zr, zr[:1]], axis=0)  # close the loop
            fig.add_trace(
                go.Scatter3d(
                    x=zr[:, 0],
                    y=zr[:, 1],
                    z=zr[:, 2],
                    mode="lines",
                    line=dict(color=COLORS[j % len(COLORS)], width=4),
                    name=f"gt ring {j}",
                    showlegend=(idx == 0),
                    legendgroup=f"gt_ring_{j}",
                ),
                row=row,
                col=col,
            )

fig.update_layout(
    title_text=f"Bottleneck (3D): SparseSpheres (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    height=800,
    width=800,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
scene_domains = [
    {"x": [0, 0.45], "y": [0.55, 1.0]},  # row 1, col 1
    {"x": [0.55, 1.0], "y": [0.55, 1.0]},  # row 1, col 2
    {"x": [0, 0.45], "y": [0, 0.45]},  # row 2, col 1
    {"x": [0.55, 1.0], "y": [0, 0.45]},  # row 2, col 2
]
for i in range(1, 5):
    scene_key = f"scene{i}" if i > 1 else "scene"
    dom = scene_domains[i - 1]
    fig.update_layout(
        **{
            scene_key: dict(
                xaxis_title="h₁",
                yaxis_title="h₂",
                zaxis_title="h₃",
                aspectmode="cube",
                domain=dom,
            )
        }
    )

fig.write_html(bottleneck_path)
fig.show()

# %%
# ── PAIRWISE COSINE SIMILARITY OF RING CENTROIDS ────────────────────────
# Sweep each ring's circle through the encoder using the tilt matrices,
# then compute pairwise cosine similarity of the centroids in bottleneck space.
cosine_path = "experiments/rings/figures/cosine_similarities.html"
ensure_dir_exists(cosine_path)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=list(models.keys()),
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = idx // 2 + 1, idx % 2 + 1

    # Sweep each ring: sample points on S^1, apply tilt, embed into full input space
    n_angles = 512
    theta = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
    circle_2d = R * torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )  # (n_angles, 2)

    centroids = []
    with torch.no_grad():
        for j in range(K):
            # Tilt: (m, n+1) @ (n+1,) for each angle → (n_angles, m)
            Rj = dist.tilts[j].to(DEVICE)  # (m, n+1) = (3, 2)
            ring_in_ambient = circle_2d @ Rj.T  # (n_angles, m)
            # Place into full input: zeros everywhere except feature j's slice
            full_input = torch.zeros(n_angles, N_FEATURES, device=DEVICE)
            full_input[:, j * M : (j + 1) * M] = ring_in_ambient
            z = tm.ae.encode(full_input)  # (n_angles, hidden_dim)
            centroids.append(z.mean(dim=0))
    centroids = torch.stack(centroids)  # (K, hidden_dim)
    centroids_norm = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_matrix = (centroids_norm @ centroids_norm.T).cpu().numpy()

    fig.add_trace(
        go.Heatmap(
            z=cos_matrix,
            x=[f"r{j}" for j in range(K)],
            y=[f"r{j}" for j in range(K)],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(idx == 0),
        ),
        row=row,
        col=col,
    )

fig.update_layout(
    title_text=f"Pairwise Cosine Similarity of Ring Centroids (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    height=900,
    width=900,
    template="plotly_white",
)
fig.write_html(cosine_path)
fig.show()

# %%
# ── EMBEDDING VECTORS 3D (TiedLinearAE) ─────────────────────────────────
# Feature weight vectors as 3D arrows from origin, colored by circle membership
tm_tied = models["TiedLinearAE"]
W = tm_tied.W.detach().cpu().numpy()  # (HIDDEN_DIM, N_FEATURES)

emb_fig = go.Figure()
for circle_idx in range(K):
    feat_start = circle_idx * M
    feat_end = feat_start + M
    color = COLORS[circle_idx % len(COLORS)]
    for f in range(feat_start, feat_end):
        v = W[:, f]
        emb_fig.add_trace(
            go.Scatter3d(
                x=[0, v[0]],
                y=[0, v[1]],
                z=[0, v[2]],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=[0, 3], color=color),
                name=f"circle {circle_idx}, feat {f}",
                legendgroup=f"circle_{circle_idx}",
                showlegend=(f == feat_start),
            )
        )

emb_fig.update_layout(
    title=f"Embedding Vectors: TiedLinearAE (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    scene=dict(
        xaxis_title="h₁",
        yaxis_title="h₂",
        zaxis_title="h₃",
        aspectmode="cube",
    ),
    height=700,
    width=700,
    template="plotly_white",
)
emb_path = "experiments/rings/figures/embedding_tied.html"
ensure_dir_exists(emb_path)
emb_fig.write_html(emb_path)
emb_fig.show()

# %%
# ── REPRESENTATION PLOT (TiedLinearAE) ──────────────────────────────────
# W^T W heatmap, bias vector, and feature norms colored by interference

rep_fig = plot_representation(tm_tied)
rep_path = "experiments/rings/figures/representation_tied.html"
ensure_dir_exists(rep_path)
rep_fig.write_html(rep_path)
rep_fig.show()

# %%
# ── FEATURE GEOMETRY 3D (TiedLinearAE) ──────────────────────────────────
# Interference network graph: nodes = features, edges = pairwise interferences
feat_fig = plot_feature_geometry_3d(tm_tied)
feat_path = "experiments/rings/figures/feature_geometry_tied.html"
ensure_dir_exists(feat_path)
feat_fig.write_html(feat_path)
feat_fig.show()

# %%
# ── GEOMETRY METRICS (TiedLinearAE) ─────────────────────────────────────
# Feature dimensionalities, hidden dims per embedded feature, etc.
geom_fig = plot_geometry(tm_tied)
geom_path = "experiments/rings/figures/geometry_tied.html"
ensure_dir_exists(geom_path)
geom_fig.write_html(geom_path)
geom_fig.show()

# %%
# -- MANUAL TESTING ───────────────────────────────────────────────────────
sample = dist.sample(25).to(DEVICE)
TiedMLP = models["TiedMLPAE"]
TiedLinear = models["TiedLinearAE"]

# Show input, TiedMLPAE recon, and TiedLinearAE recon for each sample, side by side.
mlp_recon, _ = TiedMLP.forward(sample)
linear_recon, _ = TiedLinear.forward(sample)

for i in range(sample.shape[0]):
    print(f"\nSample {i}:")
    print("GT:       ", torch.round(sample[i], decimals=4).tolist())
    print("MLP Recon:", torch.round(mlp_recon[i], decimals=4).tolist())
    print("Linear Recon:", torch.round(linear_recon[i], decimals=4).tolist())
    # Cosine similarity (handle non-1D tensors safely)
    s = sample[i].detach().cpu().flatten().float()
    mlp_r = mlp_recon[i].detach().cpu().flatten().float()
    lin_r = linear_recon[i].detach().cpu().flatten().float()
    mlp_cos = torch.nn.functional.cosine_similarity(
        s.unsqueeze(0), mlp_r.unsqueeze(0)
    ).item()
    lin_cos = torch.nn.functional.cosine_similarity(
        s.unsqueeze(0), lin_r.unsqueeze(0)
    ).item()
    print(f"Cosine similarity (GT vs MLP):    {mlp_cos:.4f}")
    print(f"Cosine similarity (GT vs Linear): {lin_cos:.4f}")

# %%
