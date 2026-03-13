# %%
# ── IMPORTS AND CONFIG ────────────────────────────────────────────────────
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu, TiedMLPEncoder
from occhio.distributions import SparseSpheres
from occhio.sae.sae import SAESimple
from occhio.visualization import (
    plot_representation,
    plot_feature_geometry_3d,
    plot_geometry,
)


assert torch.backends.mps.is_available(), "MPS device is not available"
DEVICE = "cpu"
SEED = 8
torch.manual_seed(SEED)

# SparseSpheres params: 5 circles (S^1) tilted into 3D ambient → n_features = 5*3 = 15
K = 5
N = 1  # S^1 (circles)
M = 3  # ambient dim per feature (tilt from 2D into 3D)
N_FEATURES = K * M  # = 15
P_ACTIVE = 0.01
R = 1.0

# AE / training params
HIDDEN_DIM = 3
N_EPOCHS = 30_000
BATCH_SIZE = 256
LR = 3e-4
IMPORTANCE_BASE = 0.95

# SAE params
SAE_LATENT = 20
SAE_L1 = 0.1
SAE_STEPS = 100_000

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
    noise_std=0.08,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)

samples, labels = dist.sample_with_args(4)
print(f"SparseSpheres: {K} circles (S^{N}) in {M}D ambient, total dim={N_FEATURES}")
print(f"Tilt shapes: {dist.tilts.shape}")
print(f"Sample shape: {samples.shape}, Labels shape: {labels.shape}")


# %%
# ── AUTOENCODERS ─────────────────────────────────────────────────────────
def make_models() -> dict[str, ToyModel]:
    """Build TiedLinearAE and TiedMLPAE, sharing the same distribution."""
    configs = {
        "TiedLinearAE": lambda: TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=HIDDEN_DIM,
        ),
        "TiedMLPAE": lambda: TiedMLPEncoder(
            dims=[N_FEATURES, 64, 32, HIDDEN_DIM],
        ),
    }

    models = {}
    for name, ae_factory in configs.items():
        ae = ae_factory()
        models[name] = ToyModel(
            distribution=dist, ae=ae, device=DEVICE, importances=importances
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
    rows=1,
    cols=2,
    subplot_titles=model_names,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.05,
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = 1, idx + 1

    # Sample with GT labels, no noise for clean visualization
    samples, mask = dist.sample_with_args(N_VIS, noise_std=0.0)
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

    # Plot inactive in black
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
                visible=True,
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
                visible=True,
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
                    visible=True,
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
            ring_in_ambient = circle_2d @ Rj.T + dist.centers[j].to(DEVICE)
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
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"gt_ring_{j}",
                ),
                row=row,
                col=col,
            )

fig.update_layout(
    title_text=f"Embedding Space (3D): SparseSpheres (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
for i in range(1, 3):
    scene_key = f"scene{i}" if i > 1 else "scene"
    fig.update_layout(
        **{
            scene_key: dict(
                xaxis_title="h1",
                yaxis_title="h2",
                zaxis_title="h3",
                aspectmode="cube",
            )
        }
    )

fig.write_html(bottleneck_path)
fig.show()

# %%
# ── RECONSTRUCTION VISUALIZATION (per-ring 3D) ─────────────────────────
recon_path = "experiments/rings/figures/reconstruction.html"
ensure_dir_exists(recon_path)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=model_names,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.05,
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = 1, idx + 1

    # Sample with GT labels, no noise for clean visualization
    samples, mask = dist.sample_with_args(N_VIS, noise_std=0.0)
    samples = samples.to(DEVICE)

    n_active_per_sample = mask.sum(dim=1)
    is_single = n_active_per_sample == 1
    is_inactive = n_active_per_sample == 0
    is_multi = n_active_per_sample > 1
    single_ring_id = mask.float().argmax(dim=1)

    # Forward pass: full encode -> decode
    with torch.no_grad():
        recon, _ = tm.forward(samples)
        recon = recon.cpu().numpy()

    single = is_single.numpy()
    ring_ids = single_ring_id.numpy()

    # Plot inactive reconstructions
    inact = is_inactive.numpy()
    if inact.any():
        blk = recon[inact, :M]
        fig.add_trace(
            go.Scatter3d(
                x=blk[:, 0],
                y=blk[:, 1],
                z=blk[:, 2],
                mode="markers",
                marker=dict(color="black", size=1, opacity=0.15),
                name="inactive",
                visible=True,
                showlegend=(idx == 0),
                legendgroup="inactive",
            ),
            row=row,
            col=col,
        )

    # Plot multi-ring reconstructions
    multi = is_multi.numpy()
    if multi.any():
        first_ring = mask[is_multi].float().argmax(dim=1).numpy()
        for mi, ri in enumerate(np.unique(first_ring)):
            sub = multi.copy()
            sub[multi] = first_ring == ri
            if sub.any():
                blk = recon[sub, ri * M : (ri + 1) * M]
                fig.add_trace(
                    go.Scatter3d(
                        x=blk[:, 0],
                        y=blk[:, 1],
                        z=blk[:, 2],
                        mode="markers",
                        marker=dict(color="#888888", size=1.5, opacity=0.1),
                        name="multi-ring",
                        visible=True,
                        showlegend=(idx == 0 and mi == 0),
                        legendgroup="multi",
                    ),
                    row=row,
                    col=col,
                )

    # Plot single-ring: for ring j, plot the j-th block of the reconstruction
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            blk = recon[ring_mask, j * M : (j + 1) * M]
            fig.add_trace(
                go.Scatter3d(
                    x=blk[:, 0],
                    y=blk[:, 1],
                    z=blk[:, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j % len(COLORS)], size=2, opacity=0.6),
                    name=f"ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"ring_{j}",
                ),
                row=row,
                col=col,
            )

    # Overlay ground-truth rings
    n_angles = 256
    theta_r = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
    circle_2d = R * torch.stack([torch.cos(theta_r), torch.sin(theta_r)], dim=-1)

    with torch.no_grad():
        for j in range(K):
            Rj = dist.tilts[j].to(DEVICE)
            ring_in_ambient = circle_2d @ Rj.T + dist.centers[j].to(DEVICE)
            gt = ring_in_ambient.cpu().numpy()
            gt = np.concatenate([gt, gt[:1]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=gt[:, 0],
                    y=gt[:, 1],
                    z=gt[:, 2],
                    mode="lines",
                    line=dict(color=COLORS[j % len(COLORS)], width=4),
                    name=f"gt ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"gt_ring_{j}",
                ),
                row=row,
                col=col,
            )

fig.update_layout(
    title_text=f"Reconstruction (3D): SparseSpheres (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
for i in range(1, 3):
    scene_key = f"scene{i}" if i > 1 else "scene"
    fig.update_layout(
        **{
            scene_key: dict(
                xaxis_title="d1",
                yaxis_title="d2",
                zaxis_title="d3",
                aspectmode="cube",
            )
        }
    )

fig.write_html(recon_path)
fig.show()

# %%
# ── SAE TRAINING ─────────────────────────────────────────────────────────
# Train one SAESimple per model on unnoised AE bottleneck activations.
saes: dict[str, SAESimple] = {}
sae_loss_curves: dict[str, list[float]] = {}

for name, tm in models.items():
    print(f"\nTraining SAE for {name}...")
    sae = SAESimple(
        n_latent=HIDDEN_DIM,
        n_dict=SAE_LATENT,
        l1_coef=SAE_L1,
        device=DEVICE,
    )
    sae = sae.to(DEVICE)

    # data_fn: sample unnoised data, pass through AE encoder
    def make_data_fn(model):
        def data_fn(n):
            with torch.no_grad():
                x = dist.sample(n, noise_std=0.0).to(DEVICE)
                return model.ae.encode(x)

        return data_fn

    sae_losses = sae.train_sae(
        make_data_fn(tm),
        n_steps=SAE_STEPS,
        batch_size=1024,
        lr=3e-4,
    )
    saes[name] = sae
    sae_loss_curves[name] = sae_losses
    print(f"  Final SAE loss: {sae_losses[-1]:.6f}")

# %%
# ── SAE LOSS CURVE PLOT ──────────────────────────────────────────────────
sae_loss_path = "experiments/rings/figures/sae_loss_curves.html"
ensure_dir_exists(sae_loss_path)

fig = go.Figure()
for name, losses in sae_loss_curves.items():
    steps = np.arange(1, len(losses) + 1)
    fig.add_trace(
        go.Scatter(
            x=steps[::100],
            y=np.array(losses[::100]),
            mode="lines",
            name=name,
        )
    )

fig.update_layout(
    title=f"SAE Loss: latent={SAE_LATENT}, L1={SAE_L1}",
    xaxis_title="Step",
    yaxis_title="Loss (MSE + L1)",
    yaxis_type="log",
    template="plotly_white",
)
fig.write_html(sae_loss_path)
fig.show()

# %%
# ── SAE EMBEDDING VISUALIZATION (3D) ────────────────────────────────────
# 1x2 grid: AE embedding, GT rings, SAE approximation of bottleneck.
# All legend items start hidden (legendonly).
sae_emb_path = "experiments/rings/figures/sae_embedding.html"
ensure_dir_exists(sae_emb_path)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=model_names,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.05,
)

for idx, (name, tm) in enumerate(models.items()):
    col = idx + 1
    sae = saes[name]

    # Sample unnoised with GT labels
    samples, mask = dist.sample_with_args(N_VIS, noise_std=0.0)
    samples = samples.to(DEVICE)

    n_active_per_sample = mask.sum(dim=1)
    is_single = n_active_per_sample == 1
    is_inactive = n_active_per_sample == 0
    single_ring_id = mask.float().argmax(dim=1)

    with torch.no_grad():
        z_ae = tm.ae.encode(samples)  # (N_VIS, 3)
        z_sae_recon = sae.decode(sae.encode(z_ae))  # SAE roundtrip: (N_VIS, 3)
        z_ae_np = z_ae.cpu().numpy()
        z_sae_np = z_sae_recon.cpu().numpy()

    # Zero-center on active samples (using AE embeddings as reference)
    active_mask = ~is_inactive.numpy()
    center = z_ae_np[active_mask].mean(axis=0) if active_mask.any() else np.zeros(3)
    z_ae_np = z_ae_np - center
    z_sae_np = z_sae_np - center

    single = is_single.numpy()
    ring_ids = single_ring_id.numpy()

    # ── AE embedding traces ──
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=z_ae_np[ring_mask, 0],
                    y=z_ae_np[ring_mask, 1],
                    z=z_ae_np[ring_mask, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j % len(COLORS)], size=2, opacity=0.6),
                    name=f"AE ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"ae_ring_{j}",
                ),
                row=1,
                col=col,
            )

    # ── SAE approximation traces ──
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=z_sae_np[ring_mask, 0],
                    y=z_sae_np[ring_mask, 1],
                    z=z_sae_np[ring_mask, 2],
                    mode="markers",
                    marker=dict(
                        color=COLORS[j % len(COLORS)],
                        size=2,
                        opacity=0.6,
                        symbol="diamond",
                    ),
                    name=f"SAE ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"sae_ring_{j}",
                ),
                row=1,
                col=col,
            )

    # ── GT ring traces ──
    n_angles = 256
    theta = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
    circle_2d = R * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    with torch.no_grad():
        for j in range(K):
            Rj = dist.tilts[j].to(DEVICE)
            ring_in_ambient = circle_2d @ Rj.T + dist.centers[j].to(DEVICE)
            full_input = torch.zeros(n_angles, N_FEATURES, device=DEVICE)
            full_input[:, j * M : (j + 1) * M] = ring_in_ambient
            zr = tm.ae.encode(full_input).cpu().numpy()
            zr = zr - center
            zr = np.concatenate([zr, zr[:1]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=zr[:, 0],
                    y=zr[:, 1],
                    z=zr[:, 2],
                    mode="lines",
                    line=dict(color=COLORS[j % len(COLORS)], width=4),
                    name=f"GT ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"gt_ring_{j}",
                ),
                row=1,
                col=col,
            )

fig.update_layout(
    title_text=f"SAE Embedding (3D): latent={SAE_LATENT}, L1={SAE_L1}",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
for i in range(1, 3):
    scene_key = f"scene{i}" if i > 1 else "scene"
    fig.update_layout(
        **{
            scene_key: dict(
                xaxis_title="h1",
                yaxis_title="h2",
                zaxis_title="h3",
                aspectmode="cube",
            )
        }
    )

fig.write_html(sae_emb_path)
fig.show()

# %%
# ── SAE RECONSTRUCTION VISUALIZATION (per-ring 3D) ──────────────────────
# 1x2 grid: GT rings, AE recon, AE+SAE recon in input-space blocks.
# All legend items start hidden (legendonly).
sae_recon_path = "experiments/rings/figures/sae_reconstruction.html"
ensure_dir_exists(sae_recon_path)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=model_names,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.05,
)

for idx, (name, tm) in enumerate(models.items()):
    col = idx + 1
    sae = saes[name]

    # Sample unnoised with GT labels
    samples, mask = dist.sample_with_args(N_VIS, noise_std=0.0)
    samples = samples.to(DEVICE)

    n_active_per_sample = mask.sum(dim=1)
    is_single = n_active_per_sample == 1
    is_inactive = n_active_per_sample == 0
    single_ring_id = mask.float().argmax(dim=1)

    with torch.no_grad():
        # AE reconstruction: encode → decode
        ae_recon, _ = tm.forward(samples)
        ae_recon_np = ae_recon.cpu().numpy()

        # AE+SAE reconstruction: encode → SAE roundtrip → decode
        z_ae = tm.ae.encode(samples)
        z_sae = sae.decode(sae.encode(z_ae))
        ae_sae_recon = tm.ae.decode(z_sae)
        ae_sae_recon_np = ae_sae_recon.cpu().numpy()

    single = is_single.numpy()
    ring_ids = single_ring_id.numpy()

    # ── AE recon traces (per ring block) ──
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            blk = ae_recon_np[ring_mask, j * M : (j + 1) * M]
            fig.add_trace(
                go.Scatter3d(
                    x=blk[:, 0],
                    y=blk[:, 1],
                    z=blk[:, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j % len(COLORS)], size=2, opacity=0.6),
                    name=f"AE recon ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"ae_recon_{j}",
                ),
                row=1,
                col=col,
            )

    # ── AE+SAE recon traces (per ring block) ──
    for j in range(K):
        ring_mask = single & (ring_ids == j)
        if ring_mask.any():
            blk = ae_sae_recon_np[ring_mask, j * M : (j + 1) * M]
            fig.add_trace(
                go.Scatter3d(
                    x=blk[:, 0],
                    y=blk[:, 1],
                    z=blk[:, 2],
                    mode="markers",
                    marker=dict(
                        color=COLORS[j % len(COLORS)],
                        size=2,
                        opacity=0.6,
                        symbol="diamond",
                    ),
                    name=f"AE+SAE recon ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"ae_sae_recon_{j}",
                ),
                row=1,
                col=col,
            )

    # ── GT ring traces (in input space) ──
    n_angles = 256
    theta_r = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
    circle_2d = R * torch.stack([torch.cos(theta_r), torch.sin(theta_r)], dim=-1)

    with torch.no_grad():
        for j in range(K):
            Rj = dist.tilts[j].to(DEVICE)
            ring_in_ambient = circle_2d @ Rj.T + dist.centers[j].to(DEVICE)
            gt = ring_in_ambient.cpu().numpy()
            gt = np.concatenate([gt, gt[:1]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=gt[:, 0],
                    y=gt[:, 1],
                    z=gt[:, 2],
                    mode="lines",
                    line=dict(color=COLORS[j % len(COLORS)], width=4),
                    name=f"GT ring {j}",
                    visible=True,
                    showlegend=(idx == 0),
                    legendgroup=f"gt_ring_{j}",
                ),
                row=1,
                col=col,
            )

fig.update_layout(
    title_text=f"SAE Reconstruction (3D): latent={SAE_LATENT}, L1={SAE_L1}",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
for i in range(1, 3):
    scene_key = f"scene{i}" if i > 1 else "scene"
    fig.update_layout(
        **{
            scene_key: dict(
                xaxis_title="d1",
                yaxis_title="d2",
                zaxis_title="d3",
                aspectmode="cube",
            )
        }
    )

fig.write_html(sae_recon_path)
fig.show()

# # %%
# # ── PAIRWISE COSINE SIMILARITY OF RING CENTROIDS ────────────────────────
# cosine_path = "experiments/rings/figures/cosine_similarities.html"
# ensure_dir_exists(cosine_path)

# fig = make_subplots(
#     rows=1,
#     cols=2,
#     subplot_titles=list(models.keys()),
# )

# for idx, (name, tm) in enumerate(models.items()):
#     row, col = 1, idx + 1

#     n_angles = 512
#     theta = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
#     circle_2d = R * torch.stack(
#         [torch.cos(theta), torch.sin(theta)], dim=-1
#     )

#     centroids = []
#     with torch.no_grad():
#         for j in range(K):
#             Rj = dist.tilts[j].to(DEVICE)
#             ring_in_ambient = circle_2d @ Rj.T + dist.centers[j].to(DEVICE)
#             full_input = torch.zeros(n_angles, N_FEATURES, device=DEVICE)
#             full_input[:, j * M : (j + 1) * M] = ring_in_ambient
#             z = tm.ae.encode(full_input)
#             centroids.append(z.mean(dim=0))
#     centroids = torch.stack(centroids)
#     centroids_norm = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
#     cos_matrix = (centroids_norm @ centroids_norm.T).cpu().numpy()

#     fig.add_trace(
#         go.Heatmap(
#             z=cos_matrix,
#             x=[f"r{j}" for j in range(K)],
#             y=[f"r{j}" for j in range(K)],
#             colorscale="RdBu_r",
#             zmid=0,
#             zmin=-1,
#             zmax=1,
#             showscale=(idx == 0),
#         ),
#         row=row,
#         col=col,
#     )

# fig.update_layout(
#     title_text=f"Pairwise Cosine Similarity of Ring Centroids (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
#     height=500,
#     width=1200,
#     template="plotly_white",
# )
# fig.write_html(cosine_path)
# fig.show()


# # %%
# # ── EMBEDDING VECTORS 3D (TiedLinearAE) ─────────────────────────────────
# tm_tied = models["TiedLinearAE"]
# W = tm_tied.W.detach().cpu().numpy()

# emb_fig = go.Figure()
# for circle_idx in range(K):
#     feat_start = circle_idx * M
#     feat_end = feat_start + M
#     color = COLORS[circle_idx % len(COLORS)]
#     for f in range(feat_start, feat_end):
#         v = W[:, f]
#         emb_fig.add_trace(
#             go.Scatter3d(
#                 x=[0, v[0]],
#                 y=[0, v[1]],
#                 z=[0, v[2]],
#                 mode="lines+markers",
#                 line=dict(color=color, width=4),
#                 marker=dict(size=[0, 3], color=color),
#                 name=f"circle {circle_idx}, feat {f}",
#                 legendgroup=f"circle_{circle_idx}",
#                 showlegend=(f == feat_start),
#             )
#         )

# emb_fig.update_layout(
#     title=f"Embedding Vectors: TiedLinearAE (k={K}, S^{N}, m={M}, hidden={HIDDEN_DIM})",
#     scene=dict(
#         xaxis_title="h1",
#         yaxis_title="h2",
#         zaxis_title="h3",
#         aspectmode="cube",
#     ),
#     height=700,
#     width=700,
#     template="plotly_white",
# )
# emb_path = "experiments/rings/figures/embedding_tied.html"
# ensure_dir_exists(emb_path)
# emb_fig.write_html(emb_path)
# emb_fig.show()

# # %%
# # ── REPRESENTATION PLOT (TiedLinearAE) ──────────────────────────────────
# rep_fig = plot_representation(tm_tied)
# rep_path = "experiments/rings/figures/representation_tied.html"
# ensure_dir_exists(rep_path)
# rep_fig.write_html(rep_path)
# rep_fig.show()

# # %%
# # ── FEATURE GEOMETRY 3D (TiedLinearAE) ──────────────────────────────────
# feat_fig = plot_feature_geometry_3d(tm_tied)
# feat_path = "experiments/rings/figures/feature_geometry_tied.html"
# ensure_dir_exists(feat_path)
# feat_fig.write_html(feat_path)
# feat_fig.show()

# # %%
# # ── GEOMETRY METRICS (TiedLinearAE) ─────────────────────────────────────
# geom_fig = plot_geometry(tm_tied)
# geom_path = "experiments/rings/figures/geometry_tied.html"
# ensure_dir_exists(geom_path)
# geom_fig.write_html(geom_path)
# geom_fig.show()

# # %%
# # -- MANUAL TESTING ───────────────────────────────────────────────────────


# def is_nonsparse(vec, tol=1e-6):
#     return torch.any(torch.abs(vec) > tol).item()


# n_runs = 100
# batch_size = 25
# for run in range(n_runs):
#     sample = dist.sample(batch_size).to(DEVICE)
#     TiedMLP = models["TiedMLPAE"]
#     TiedLinear = models["TiedLinearAE"]
#     mlp_recon, _ = TiedMLP.forward(sample)
#     linear_recon, _ = TiedLinear.forward(sample)

#     for i in range(sample.shape[0]):
#         s = sample[i].detach().cpu().flatten().float()
#         mlp_r = mlp_recon[i].detach().cpu().flatten().float()
#         lin_r = linear_recon[i].detach().cpu().flatten().float()
#         if is_nonsparse(s) or is_nonsparse(mlp_r) or is_nonsparse(lin_r):
#             print(f"\nRun {run}, Sample {i}:")
#             print("GT:       ", torch.round(s, decimals=4).tolist())
#             print("MLP Recon:", torch.round(mlp_r, decimals=4).tolist())
#             print("Linear Recon:", torch.round(lin_r, decimals=4).tolist())
#             mlp_dist = torch.norm(s - mlp_r, p=2).item()
#             lin_dist = torch.norm(s - lin_r, p=2).item()
#             print(f"L2 distance (GT vs MLP):    {mlp_dist:.4f}")
#             print(f"L2 distance (GT vs Linear): {lin_dist:.4f}")


# # %%
# print(TiedLinear.ae.W)
# print(TiedLinear.ae.W.T)
# # %%

# %%
