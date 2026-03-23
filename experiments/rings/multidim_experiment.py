# %%
# ── CELL 1: IMPORTS AND CONFIG ──────────────────────────────────────────
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from types import SimpleNamespace
import os

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu, TiedMLPEncoder
from occhio.distributions import SparseSpheres, DistributionStack, SparseUniform
from occhio.sae import SAESimple, MultiDimSAE

DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

# Feature counts
K_SPHERES = 3
K_LINEAR = 3
K_TOTAL = K_SPHERES + K_LINEAR  # 6 feature groups
N = 1  # S^1 circles
M = 3  # ambient dim per sphere
N_SPHERE_FEATURES = K_SPHERES * M  # 9
N_LINEAR_FEATURES = K_LINEAR  # 3
N_FEATURES = N_SPHERE_FEATURES + N_LINEAR_FEATURES  # 12
P_ACTIVE = 0.07
RADIUS = 1.0

# AE / training params
HIDDEN_DIM = 3
N_EPOCHS = 100_000
BATCH_SIZE = 256
LR = 1e-3
IB = 0.95  # importance base

# SAESimple params — 55 latents gives enough dictionary capacity to tessellate
# the manifold; high L1 then forces L0 ≈ 1 (only ~1 atom fires per sample).
SAE_LATENT = 55
SAE_STEPS = 100_000

# MultiDimSAE params
MDSAE_FEATURES = 8
MDSAE_SUBDIM = 2
MDSAE_LAMBDA_GROUP = 0.3
MDSAE_LAMBDA_COL = 0.1  # within-feature sparsity: shrinks V_i columns toward 0
MDSAE_THETA = 0.5
MDSAE_STEPS = 100_000

# Sparsity sweep
SWEEP_STEPS = 20_000
SAE_L1_SWEEP = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
MDSAE_THETA_SWEEP = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]

# Importances: interleaved values, concatenated layout
sphere_importances = torch.tensor([IB] * 3 + [IB**3] * 3 + [IB**5] * 3)
linear_importances = torch.tensor([IB**2, IB**4, IB**6])
importances = torch.cat([sphere_importances, linear_importances])

# Visualization
N_VIS = 50_000
COLORS = [
    "#1f77b4",  # sphere 0
    "#ff7f0e",  # sphere 1
    "#2ca02c",  # sphere 2
    "#d62728",  # linear 0
    "#9467bd",  # linear 1
    "#8c564b",  # linear 2
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
FEATURE_NAMES = [
    "sphere 0",
    "sphere 1",
    "sphere 2",
    "linear 0",
    "linear 1",
    "linear 2",
]
FIGURES_DIR = "experiments/rings/figures"

print(
    f"Mixed distribution: {K_SPHERES} spheres (S^{N}, m={M}) + {K_LINEAR} linear, "
    f"hidden={HIDDEN_DIM}"
)
print(
    f"Total features: {N_FEATURES} ({N_SPHERE_FEATURES} sphere + {N_LINEAR_FEATURES} linear)"
)
print(f"Importances: {importances.tolist()}")


# %%
# ── CELL 2: DISTRIBUTION ────────────────────────────────────────────────
sphere_dist = SparseSpheres(
    n_spheres=K_SPHERES,
    sphere_dim=N,
    ambient_dim=M,
    p_active=P_ACTIVE,
    p_infill=0.0,
    radius=RADIUS,
    noise_std=0.18,
    n_discretizations=11,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)

linear_dist = SparseUniform(
    n_features=K_LINEAR,
    p_active=P_ACTIVE,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED + 1),
    device=DEVICE,
)

dist = DistributionStack([sphere_dist, linear_dist], sampling_mode="independent")


def sample_mixed_with_labels(batch_size):
    """Sample from mixed distribution, returning (data, mask).

    data: (batch, N_FEATURES) — concatenated sphere + linear features
    mask: (batch, K_TOTAL) — one bool per feature group (3 spheres + 3 linear)
    """
    sphere_data, sphere_mask = sphere_dist.sample_with_args(batch_size)
    linear_data = linear_dist.sample(batch_size)
    linear_mask = linear_data > 0
    data = torch.cat([sphere_data, linear_data], dim=-1)
    mask = torch.cat([sphere_mask, linear_mask], dim=-1)
    return data, mask


samples, labels = sample_mixed_with_labels(4)
print(f"Sample shape: {samples.shape}, Labels shape: {labels.shape}")


# %%
# ── CELL 3: UTILITY FUNCTIONS ───────────────────────────────────────────
def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def fold_to_3d(arr: np.ndarray) -> np.ndarray:
    """Fold high-dimensional vectors to 3D by summing coordinates mod 3."""
    d = arr.shape[-1]
    if d == 3:
        return arr
    rem = d % 3
    if rem:
        pad = [(0, 0)] * (arr.ndim - 1) + [(0, 3 - rem)]
        arr = np.pad(arr, pad)
        d = arr.shape[-1]
    return arr.reshape(*arr.shape[:-1], d // 3, 3).sum(axis=-2)


def sample_sphere_surface(n, n_pts, r, device):
    """Sample points on r*S^n. Returns (points, is_ordered)."""
    if n == 1:
        theta = torch.linspace(0, 2 * np.pi, n_pts + 1)[:-1].to(device)
        return r * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1), True
    elif n == 2:
        indices = torch.arange(n_pts, dtype=torch.float32, device=device)
        golden = (1 + 5**0.5) / 2
        theta = torch.acos(1 - 2 * (indices + 0.5) / n_pts)
        phi = 2 * np.pi * indices / golden
        return r * torch.stack(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        ), False
    else:
        pts = torch.randn(n_pts, n + 1, device=device)
        return r * pts / pts.norm(dim=-1, keepdim=True), False


def add_manifold_trace(
    fig,
    pts_3d,
    *,
    color,
    name,
    showlegend,
    legendgroup,
    row,
    col,
    is_ordered,
    opacity=0.3,
):
    """Add a GT manifold trace — lines for ordered, markers otherwise."""
    if is_ordered:
        pts_3d = np.concatenate([pts_3d, pts_3d[:1]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=pts_3d[:, 0],
                y=pts_3d[:, 1],
                z=pts_3d[:, 2],
                mode="lines",
                line=dict(color=color, width=4),
                opacity=opacity,
                name=name,
                visible=True,
                showlegend=showlegend,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=pts_3d[:, 0],
                y=pts_3d[:, 1],
                z=pts_3d[:, 2],
                mode="markers",
                marker=dict(color=color, size=1.5, opacity=opacity),
                name=name,
                visible=True,
                showlegend=showlegend,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )


def make_single_feature_input_sphere(j, pts_ambient):
    """Create full input vector with sphere feature j active."""
    n_pts = pts_ambient.shape[0]
    full_input = torch.zeros(n_pts, N_FEATURES, device=pts_ambient.device)
    full_input[:, j * M : (j + 1) * M] = pts_ambient
    return full_input


def make_single_feature_input_linear(j, values):
    """Create full input vector with linear feature j active."""
    n_pts = values.shape[0]
    full_input = torch.zeros(n_pts, N_FEATURES, device=values.device)
    full_input[:, N_SPHERE_FEATURES + j] = values
    return full_input


def update_3d_scenes(fig, n_scenes, **scene_kwargs):
    """Update all 3D scene layouts."""
    for i in range(1, n_scenes + 1):
        scene_key = f"scene{i}" if i > 1 else "scene"
        fig.update_layout(**{scene_key: scene_kwargs})


def add_sphere_gt_manifold(fig, tm, center, row, col, show_legend, tag="gt"):
    """Add GT manifold traces for all sphere features to a latent-space plot."""
    sphere_pts, gt_ordered = sample_sphere_surface(N, 256, RADIUS, DEVICE)
    with torch.no_grad():
        for j in range(K_SPHERES):
            Rj = sphere_dist.tilts[j].to(DEVICE)
            pts_ambient = sphere_pts @ Rj.T + sphere_dist.centers[j].to(DEVICE)
            full_input = make_single_feature_input_sphere(j, pts_ambient)
            zr = fold_to_3d(tm.ae.encode(full_input).cpu().numpy() - center)
            add_manifold_trace(
                fig,
                zr,
                color=COLORS[j],
                name=f"{tag} {FEATURE_NAMES[j]}",
                showlegend=show_legend,
                legendgroup=f"{tag}_{j}",
                row=row,
                col=col,
                is_ordered=gt_ordered,
            )


def add_linear_gt_manifold(fig, tm, center, row, col, show_legend, tag="gt"):
    """Add GT manifold traces for all linear features to a latent-space plot."""
    values = torch.linspace(0, 1, 256, device=DEVICE)
    with torch.no_grad():
        for j in range(K_LINEAR):
            feat_idx = K_SPHERES + j
            full_input = make_single_feature_input_linear(j, values)
            zr = fold_to_3d(tm.ae.encode(full_input).cpu().numpy() - center)
            fig.add_trace(
                go.Scatter3d(
                    x=zr[:, 0],
                    y=zr[:, 1],
                    z=zr[:, 2],
                    mode="lines",
                    line=dict(color=COLORS[feat_idx], width=4),
                    opacity=0.3,
                    name=f"{tag} {FEATURE_NAMES[feat_idx]}",
                    visible=True,
                    showlegend=show_legend,
                    legendgroup=f"{tag}_{feat_idx}",
                ),
                row=row,
                col=col,
            )


def labeled_samples(n):
    """Sample from mixed distribution with pre-computed masks and feature ids.

    Returns a SimpleNamespace with fields: data, mask, is_single, is_inactive,
    is_multi, feat_id. Data is already on DEVICE.
    """
    data, mask = sample_mixed_with_labels(n)
    data = data.to(DEVICE)
    n_active = mask.sum(dim=1)
    return SimpleNamespace(
        data=data,
        mask=mask,
        is_single=(n_active == 1),
        is_inactive=(n_active == 0),
        is_multi=(n_active > 1),
        feat_id=mask.float().argmax(dim=1),
    )


def make_data_fn(model, active_only=False, norm_threshold=0.01):
    """Create a data function that samples from dist and encodes through AE.

    When active_only=True, filters out near-zero latents (||z|| < norm_threshold).
    This is needed for TiedLinearRelu where ~95% of latents are zero (zero input
    maps to zero latent), which otherwise causes the SAE to learn identity.
    """

    def data_fn(n):
        with torch.no_grad():
            if not active_only:
                x = dist.sample(n).to(DEVICE)
                return model.ae.encode(x)
            # Oversample and filter to active-only
            collected = []
            while sum(len(c) for c in collected) < n:
                x = dist.sample(n * 25).to(DEVICE)
                z = model.ae.encode(x)
                active = z[z.norm(dim=-1) >= norm_threshold]
                if len(active) > 0:
                    collected.append(active)
            return torch.cat(collected)[:n]

    return data_fn


def add_sphere_gt_recon(fig, row, col, show_legend, tag="gt"):
    """Add GT manifold traces for sphere features in reconstruction (input) space."""
    sphere_pts, gt_ordered = sample_sphere_surface(N, 256, RADIUS, DEVICE)
    with torch.no_grad():
        for j in range(K_SPHERES):
            Rj = sphere_dist.tilts[j].to(DEVICE)
            pts_ambient = sphere_pts @ Rj.T + sphere_dist.centers[j].to(DEVICE)
            gt = fold_to_3d(pts_ambient.cpu().numpy())
            add_manifold_trace(
                fig,
                gt,
                color=COLORS[j],
                name=f"{tag} {FEATURE_NAMES[j]}",
                showlegend=show_legend,
                legendgroup=f"{tag}_{j}",
                row=row,
                col=col,
                is_ordered=gt_ordered,
            )


# %%
# ── CELL 4: AUTOENCODERS ────────────────────────────────────────────────
def make_models() -> dict[str, ToyModel]:
    """Build TiedLinearAE and TiedMLPAE, sharing the same distribution."""
    configs = {
        "TiedLinearAE": lambda: TiedLinearRelu(
            n_features=N_FEATURES,
            n_hidden=HIDDEN_DIM,
        ),
        "TiedMLPAE": lambda: TiedMLPEncoder(
            dims=[N_FEATURES, 64, 32, 64, 32, HIDDEN_DIM],
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
# ── CELL 5: TRAINING LOOP ───────────────────────────────────────────────
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
# ── CELL 6: AE LOSS CURVE PLOT ──────────────────────────────────────────
loss_path = os.path.join(FIGURES_DIR, "loss_curves.html")
ensure_dir_exists(loss_path)

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
    title=(
        f"AE Reconstruction Loss: Mixed Spheres+Linear "
        f"({K_SPHERES}S + {K_LINEAR}L, hidden={HIDDEN_DIM})"
    ),
    xaxis_title="Epoch",
    yaxis_title="MSE Loss",
    yaxis_type="log",
    template="plotly_white",
)
fig.write_html(loss_path)
fig.show()


# %%
# ── CELL 7: LATENT VISUALIZATION (with sphere + linear GT manifolds) ───
latent_path = os.path.join(FIGURES_DIR, "latent.html")
ensure_dir_exists(latent_path)

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
    batch = labeled_samples(N_VIS)

    with torch.no_grad():
        z = tm.ae.encode(batch.data).cpu().numpy()

    active_np = ~batch.is_inactive.numpy()
    center = z[active_np].mean(axis=0) if active_np.any() else np.zeros(z.shape[1])
    z = fold_to_3d(z - center)

    for label, mask_t, style in [
        ("inactive", batch.is_inactive, dict(color="black", size=1, opacity=0.15)),
        ("multi-feature", batch.is_multi, dict(color="#888888", size=1.5, opacity=0.1)),
    ]:
        m = mask_t.numpy()
        if m.any():
            fig.add_trace(
                go.Scatter3d(
                    x=z[m, 0],
                    y=z[m, 1],
                    z=z[m, 2],
                    mode="markers",
                    marker=style,
                    name=label,
                    showlegend=(idx == 0),
                    legendgroup=label,
                ),
                row=row,
                col=col,
            )

    single = batch.is_single.numpy()
    feat_ids = batch.feat_id.numpy()
    for j in range(K_TOTAL):
        fm = single & (feat_ids == j)
        if fm.any():
            fig.add_trace(
                go.Scatter3d(
                    x=z[fm, 0],
                    y=z[fm, 1],
                    z=z[fm, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j], size=2, opacity=0.6),
                    name=FEATURE_NAMES[j],
                    showlegend=(idx == 0),
                    legendgroup=f"feat_{j}",
                ),
                row=row,
                col=col,
            )

    add_sphere_gt_manifold(fig, tm, center, row, col, show_legend=(idx == 0))
    add_linear_gt_manifold(fig, tm, center, row, col, show_legend=(idx == 0))

fig.update_layout(
    title_text=f"Latent Space: Mixed ({K_SPHERES}S + {K_LINEAR}L, hidden={HIDDEN_DIM})",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
update_3d_scenes(
    fig, 2, xaxis_title="h1", yaxis_title="h2", zaxis_title="h3", aspectmode="cube"
)
fig.write_html(latent_path)
fig.show()


# %%
# ── CELL 8: RECONSTRUCTION VISUALIZATION (with linear GT) ──────────────
recon_path = os.path.join(FIGURES_DIR, "reconstruction.html")
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
    batch = labeled_samples(N_VIS)

    with torch.no_grad():
        recon = tm.ae.decode(tm.ae.encode(batch.data)).cpu().numpy()

    single = batch.is_single.numpy()
    feat_ids = batch.feat_id.numpy()

    for j in range(K_SPHERES):
        feat_mask = single & (feat_ids == j)
        if feat_mask.any():
            blk = fold_to_3d(recon[feat_mask, j * M : (j + 1) * M])
            fig.add_trace(
                go.Scatter3d(
                    x=blk[:, 0],
                    y=blk[:, 1],
                    z=blk[:, 2],
                    mode="markers",
                    marker=dict(color=COLORS[j], size=2, opacity=0.6),
                    name=f"recon {FEATURE_NAMES[j]}",
                    showlegend=(idx == 0),
                    legendgroup=f"recon_{j}",
                ),
                row=row,
                col=col,
            )

    for j in range(K_LINEAR):
        feat_idx = K_SPHERES + j
        feat_mask = single & (feat_ids == feat_idx)
        if feat_mask.any():
            vals = recon[feat_mask, N_SPHERE_FEATURES + j]
            fig.add_trace(
                go.Scatter3d(
                    x=vals,
                    y=np.full_like(vals, j * 0.5),
                    z=np.zeros_like(vals),
                    mode="markers",
                    marker=dict(color=COLORS[feat_idx], size=3, opacity=0.6),
                    name=f"recon {FEATURE_NAMES[feat_idx]}",
                    showlegend=(idx == 0),
                    legendgroup=f"recon_{feat_idx}",
                ),
                row=row,
                col=col,
            )

    # GT manifolds for sphere and linear features
    add_sphere_gt_recon(fig, row, col, show_legend=(idx == 0))
    # Linear GT: plot identity line as (value, offset, 0)
    gt_vals = np.linspace(0, 1, 256)
    for j in range(K_LINEAR):
        feat_idx = K_SPHERES + j
        fig.add_trace(
            go.Scatter3d(
                x=gt_vals,
                y=np.full_like(gt_vals, j * 0.5),
                z=np.zeros_like(gt_vals),
                mode="lines",
                line=dict(color=COLORS[feat_idx], width=4),
                opacity=0.3,
                name=f"gt {FEATURE_NAMES[feat_idx]}",
                showlegend=(idx == 0),
                legendgroup=f"gt_{feat_idx}",
            ),
            row=row,
            col=col,
        )

fig.update_layout(
    title_text=f"Reconstruction: Mixed ({K_SPHERES}S + {K_LINEAR}L, hidden={HIDDEN_DIM})",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=13)),
)
update_3d_scenes(
    fig, 2, xaxis_title="d1", yaxis_title="d2", zaxis_title="d3", aspectmode="cube"
)
fig.write_html(recon_path)
fig.show()


# %%
# ── CELL 9: SPARSITY SWEEP + AUTO-SELECT ──────────────────────────────
# Sweep L1 (SAESimple) and theta (MultiDimSAE), pick Pareto-optimal config,
# then retrain the winners at full steps.


def evaluate_sae(sae, data_fn, n_total, n_eval=5000):
    """Evaluate an SAE: returns dict with l0, mse, dead, alive."""
    with torch.no_grad():
        x = data_fn(n_eval)
        z = sae.encode(x)
        x_hat = sae.decode(z)
    return {
        "l0": (z > 0).float().sum(1).mean().item(),
        "mse": ((x - x_hat) ** 2).sum(-1).mean().item(),
        "dead": int(((z > 0).float().mean(0) == 0).sum().item()),
        "alive": n_total - int(((z > 0).float().mean(0) == 0).sum().item()),
    }


def evaluate_mdsae(mdsae, data_fn, n_total, n_eval=5000):
    """Evaluate a MultiDimSAE: returns dict with l0, mse, dead, alive."""
    with torch.no_grad():
        x = data_fn(n_eval)
        x_hat, _, gate, _ = mdsae.forward(x)
    dead = int((gate.sum(0) == 0).sum().item())
    return {
        "l0": gate.sum(-1).mean().item(),
        "mse": ((x - x_hat) ** 2).sum(-1).mean().item(),
        "dead": dead,
        "alive": n_total - dead,
    }


def pick_best(results):
    """Pick the config with lowest reconstruction MSE."""
    return min(results, key=lambda r: r["mse"])


sweep_results_sae: dict[str, list[dict]] = {}
sweep_results_md: dict[str, list[dict]] = {}

for name, tm in models.items():
    # TiedLinearRelu maps zero→zero, so 95% of latents are zero. Use active-only
    # data + dead neuron resampling to prevent the SAE from learning identity.
    use_active = isinstance(tm.ae, TiedLinearRelu)
    data_fn = make_data_fn(tm, active_only=use_active)

    # ── SAESimple L1 sweep ──
    tag = " (active-only)" if use_active else ""
    print(f"\n{'=' * 60}")
    print(f"SAESimple L1 sweep — {name}{tag} (dict={SAE_LATENT}, {SWEEP_STEPS} steps)")
    sweep_results_sae[name] = []
    for l1 in SAE_L1_SWEEP:
        sae = SAESimple(
            n_latent=HIDDEN_DIM, n_dict=SAE_LATENT, l1_coef=l1, device=DEVICE
        )
        sae.train_sae(data_fn, n_steps=SWEEP_STEPS, batch_size=1024, lr=3e-4)
        m = evaluate_sae(sae, data_fn, SAE_LATENT)
        sweep_results_sae[name].append({"l1": l1, **m})
        print(
            f"  L1={l1:<7.1f}  L0={m['l0']:5.2f}  MSE={m['mse']:.6f}  "
            f"dead={m['dead']:3d}  alive={m['alive']:3d}"
        )

    # ── MultiDimSAE theta sweep ──
    print(
        f"\nMultiDimSAE θ sweep — {name} "
        f"(feat={MDSAE_FEATURES}, λ_g={MDSAE_LAMBDA_GROUP}, λ_c={MDSAE_LAMBDA_COL})"
    )
    sweep_results_md[name] = []
    for theta in MDSAE_THETA_SWEEP:
        mdsae = MultiDimSAE(
            n_input=HIDDEN_DIM,
            n_features=MDSAE_FEATURES,
            subspace_dim=MDSAE_SUBDIM,
            lambda_group=MDSAE_LAMBDA_GROUP,
            lambda_col=MDSAE_LAMBDA_COL,
            theta=theta,
            device=DEVICE,
        )
        mdsae.train_sae(data_fn, n_steps=SWEEP_STEPS, batch_size=1024, lr=3e-4)
        m = evaluate_mdsae(mdsae, data_fn, MDSAE_FEATURES)
        sweep_results_md[name].append({"theta": theta, **m})
        print(
            f"  θ={theta:<4.1f}  L0={m['l0']:5.2f}  MSE={m['mse']:.6f}  "
            f"dead={m['dead']:3d}  alive={m['alive']:3d}"
        )


# ── Auto-select and retrain at full steps ──
print(f"\n{'=' * 60}")
print("Auto-selecting best (lowest MSE) and retraining")
print(f"{'=' * 60}")

saes: dict[str, SAESimple] = {}
mdsaes: dict[str, MultiDimSAE] = {}
sae_loss_curves: dict[str, list[float]] = {}
mdsae_loss_curves: dict[str, list[float]] = {}
best_sae_configs: dict[str, dict] = {}
best_md_configs: dict[str, dict] = {}

for name, tm in models.items():
    best_sae = pick_best(sweep_results_sae[name])
    best_md = pick_best(sweep_results_md[name])
    best_sae_configs[name] = best_sae
    best_md_configs[name] = best_md
    print(
        f"\n  {name}: SAE L1={best_sae['l1']} (L0={best_sae['l0']:.2f}), "
        f"mdSAE θ={best_md['theta']} (L0={best_md['l0']:.2f})"
    )

    use_active = isinstance(tm.ae, TiedLinearRelu)
    data_fn = make_data_fn(tm, active_only=use_active)

    sae = SAESimple(
        n_latent=HIDDEN_DIM, n_dict=SAE_LATENT, l1_coef=best_sae["l1"], device=DEVICE
    )
    sae_loss_curves[name] = sae.train_sae(
        data_fn, n_steps=SAE_STEPS, batch_size=1024, lr=3e-4
    )
    saes[name] = sae

    mdsae = MultiDimSAE(
        n_input=HIDDEN_DIM,
        n_features=MDSAE_FEATURES,
        subspace_dim=MDSAE_SUBDIM,
        lambda_group=MDSAE_LAMBDA_GROUP,
        lambda_col=MDSAE_LAMBDA_COL,
        theta=best_md["theta"],
        device=DEVICE,
    )
    mdsae_loss_curves[name] = mdsae.train_sae(
        data_fn, n_steps=MDSAE_STEPS, batch_size=1024, lr=3e-4
    )
    mdsaes[name] = mdsae


# %%
# ── CELL 9b: PARETO PLOT — L0 vs Reconstruction MSE ──────────────────
pareto_path = os.path.join(FIGURES_DIR, "sparsity_sweep.html")
ensure_dir_exists(pareto_path)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["SAESimple: L1 Sweep", "MultiDimSAE: θ Sweep"],
    horizontal_spacing=0.12,
)

sweep_colors = {"TiedLinearAE": "#1f77b4", "TiedMLPAE": "#ff7f0e"}

for name in models:
    clr = sweep_colors.get(name, "gray")

    # Plot sweep points + selected star for each SAE type
    for col_idx, (results, best, label_key, sae_label) in enumerate(
        [
            (sweep_results_sae[name], best_sae_configs[name], "l1", "SAESimple"),
            (sweep_results_md[name], best_md_configs[name], "theta", "MultiDimSAE"),
        ],
        start=1,
    ):
        prefix = "L1" if label_key == "l1" else "θ"
        fig.add_trace(
            go.Scatter(
                x=[r["l0"] for r in results],
                y=[r["mse"] for r in results],
                mode="markers+text",
                text=[f"{prefix}={r[label_key]}" for r in results],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(color=clr, size=10),
                name=f"{sae_label} — {name}",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=[best["l0"]],
                y=[best["mse"]],
                mode="markers",
                marker=dict(
                    color=clr, size=18, symbol="star", line=dict(width=2, color="black")
                ),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

for c in [1, 2]:
    fig.update_xaxes(title_text="L0", row=1, col=c)
    fig.update_yaxes(title_text="Reconstruction MSE", row=1, col=c)

fig.update_layout(
    title_text="Sparsity Sweep: L0 vs MSE (star = selected, lowest MSE)",
    height=500,
    width=1200,
    template="plotly_white",
)
fig.write_html(pareto_path)
fig.show()


# %%
# ── CELL 10: SAE LOSS CURVES (both SAEs) ────────────────────────────────
sae_loss_path = os.path.join(FIGURES_DIR, "sae_loss_curves.html")
ensure_dir_exists(sae_loss_path)

fig = go.Figure()
for name in models:
    steps_sae = np.arange(1, len(sae_loss_curves[name]) + 1)
    fig.add_trace(
        go.Scatter(
            x=steps_sae[::100],
            y=np.array(sae_loss_curves[name][::100]),
            mode="lines",
            name=f"SAESimple — {name}",
        )
    )
    steps_md = np.arange(1, len(mdsae_loss_curves[name]) + 1)
    fig.add_trace(
        go.Scatter(
            x=steps_md[::100],
            y=np.array(mdsae_loss_curves[name][::100]),
            mode="lines",
            name=f"MultiDimSAE — {name}",
            line=dict(dash="dash"),
        )
    )

fig.update_layout(
    title="SAE Loss Curves: SAESimple vs MultiDimSAE",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
    template="plotly_white",
)
fig.write_html(sae_loss_path)
fig.show()


# %%
# ── CELL 11: SAE DIAGNOSTICS (enhanced, both SAEs) ─────────────────────
print("=" * 70)
print("SAE DIAGNOSTICS")
print(f"  Input dim:  {HIDDEN_DIM}")
print(
    f"  SAESimple dict: {SAE_LATENT}  |  MultiDimSAE features: {MDSAE_FEATURES} x {MDSAE_SUBDIM}D"
)
print("=" * 70)

for name, tm in models.items():
    sae = saes[name]
    mdsae = mdsaes[name]
    batch = labeled_samples(5000)

    with torch.no_grad():
        z_ae = tm.ae.encode(batch.data)
        z_sae = sae.encode(z_ae)
        z_sae_recon = sae.decode(z_sae)
        md_x_hat, md_z, md_gate, _ = mdsae.forward(z_ae)

    sae_m = evaluate_sae(sae, make_data_fn(tm), SAE_LATENT)
    md_m = evaluate_mdsae(mdsae, make_data_fn(tm), MDSAE_FEATURES)

    print(f"\n  {name}:")
    print(f"    {'Metric':<35} {'SAESimple':>12} {'MultiDimSAE':>12}")
    print(f"    {'-' * 35} {'-' * 12} {'-' * 12}")
    print(
        f"    {'L0 (active per sample)':<35} {sae_m['l0']:>12.2f} {md_m['l0']:>12.2f}"
    )
    print(f"    {'Recon MSE (latent)':<35} {sae_m['mse']:>12.6f} {md_m['mse']:>12.6f}")
    print(f"    {'Dead neurons/features':<35} {sae_m['dead']:>12d} {md_m['dead']:>12d}")

    # Feature -> SAE neuron mapping
    def _topk_str(activations, mask, n_dict, prefix):
        """Format top-k activated neurons/features for a GT feature."""
        if mask.sum() == 0:
            return "no samples"
        mean_act = activations[mask].mean(dim=0)
        top_k = torch.topk(mean_act, min(3, n_dict))
        total = mean_act.sum().item()
        parts = []
        for i in range(len(top_k.indices)):
            v = top_k.values[i].item()
            if v < 0.001:
                break
            pct = v / total * 100 if total > 0 else 0
            parts.append(f"{prefix}{top_k.indices[i].item()}({pct:.0f}%)")
        return ", ".join(parts) or "none"

    md_acts = md_z.norm(dim=-1) * md_gate

    for label, acts, n_dict, prefix in [
        ("SAESimple feature mapping", z_sae, SAE_LATENT, "n"),
        ("MultiDimSAE feature mapping", md_acts, MDSAE_FEATURES, "f"),
    ]:
        print(f"\n    {label}:")
        for j in range(K_TOTAL):
            fm = batch.is_single & (batch.feat_id == j)
            print(f"      {FEATURE_NAMES[j]} -> {_topk_str(acts, fm, n_dict, prefix)}")

    # Effective dimensionality per mdSAE feature
    eff_dims = mdsae.effective_dim_per_feature(z_ae)
    alive_md = md_gate.sum(dim=0) > 0
    print("\n    MultiDimSAE effective dimensionality (alive features):")
    for fi in range(MDSAE_FEATURES):
        if not alive_md[fi]:
            continue
        d = eff_dims[fi].item()
        print(
            f"      f{fi}: {d:.2f} ({'2D' if d > 1.5 else '1D' if d > 0.5 else '0D'})"
        )

    # Per-feature reconstruction MSE
    print("\n    Per-feature latent MSE:")
    print(f"      {'Feature':<12} {'SAESimple':>12} {'MultiDimSAE':>12}")
    for j in range(K_TOTAL):
        fm = batch.is_single & (batch.feat_id == j)
        if fm.sum() == 0:
            continue
        with torch.no_grad():
            z_f = z_ae[fm]
            sae_f_mse = ((z_f - sae.decode(sae.encode(z_f))) ** 2).sum(-1).mean().item()
            md_f_hat, _, _, _ = mdsae.forward(z_f)
            md_f_mse = ((z_f - md_f_hat) ** 2).sum(-1).mean().item()
        print(f"      {FEATURE_NAMES[j]:<12} {sae_f_mse:>12.6f} {md_f_mse:>12.6f}")


# %%
# ── CELL 12: SAE LATENT VIS (AE latent + both SAE roundtrips + GT) ─────
sae_emb_path = os.path.join(FIGURES_DIR, "sae_embedding.html")
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
    sae, mdsae = saes[name], mdsaes[name]
    batch = labeled_samples(N_VIS)

    with torch.no_grad():
        z_ae = tm.ae.encode(batch.data)
        z_sae_rt = sae.decode(sae.encode(z_ae))
        md_rt, _, _, _ = mdsae.forward(z_ae)

    active_np = ~batch.is_inactive.numpy()
    z_ae_np = z_ae.cpu().numpy()
    center = z_ae_np[active_np].mean(axis=0) if active_np.any() else np.zeros(3)
    roundtrips = {
        "AE": fold_to_3d(z_ae_np - center),
        "SAE": fold_to_3d(z_sae_rt.cpu().numpy() - center),
        "mdSAE": fold_to_3d(md_rt.cpu().numpy() - center),
    }

    single = batch.is_single.numpy()
    feat_ids = batch.feat_id.numpy()
    symbols = {"AE": "circle", "SAE": "diamond", "mdSAE": "square"}

    for label, z_3d in roundtrips.items():
        for j in range(K_TOTAL):
            fm = single & (feat_ids == j)
            if fm.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=z_3d[fm, 0],
                        y=z_3d[fm, 1],
                        z=z_3d[fm, 2],
                        mode="markers",
                        marker=dict(
                            color=COLORS[j], size=2, opacity=0.5, symbol=symbols[label]
                        ),
                        name=f"{label} {FEATURE_NAMES[j]}",
                        showlegend=(idx == 0),
                        legendgroup=f"{label.lower()}_{j}",
                    ),
                    row=1,
                    col=col,
                )

    add_sphere_gt_manifold(fig, tm, center, 1, col, show_legend=(idx == 0))
    add_linear_gt_manifold(fig, tm, center, 1, col, show_legend=(idx == 0))

fig.update_layout(
    title_text="SAE Latent: AE / SAESimple / MultiDimSAE roundtrips",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=11)),
)
update_3d_scenes(
    fig, 2, xaxis_title="h1", yaxis_title="h2", zaxis_title="h3", aspectmode="cube"
)
fig.write_html(sae_emb_path)
fig.show()


# %%
# ── CELL 13: GROUPED RECONSTRUCTION {GT, AE, SAE, mdSAE} ──────────────
sae_recon_path = os.path.join(FIGURES_DIR, "sae_reconstruction.html")
ensure_dir_exists(sae_recon_path)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=model_names,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.05,
)

n_recon = 256  # sweep points per feature

for idx, (name, tm) in enumerate(models.items()):
    col = idx + 1
    sae = saes[name]
    mdsae = mdsaes[name]

    # Sphere features: sweep S^1
    sphere_pts, _ord = sample_sphere_surface(N, n_recon, RADIUS, DEVICE)
    for j in range(K_SPHERES):
        Rj = sphere_dist.tilts[j].to(DEVICE)
        pts_amb = sphere_pts @ Rj.T + sphere_dist.centers[j].to(DEVICE)
        full_in = make_single_feature_input_sphere(j, pts_amb)

        gt_np = pts_amb.cpu().numpy()
        with torch.no_grad():
            ae_rec = tm.ae.decode(tm.ae.encode(full_in))
            z_lat = tm.ae.encode(full_in)
            sae_rec = tm.ae.decode(sae.decode(sae.encode(z_lat)))
            md_hat, _, _, _ = mdsae.forward(z_lat)
            md_rec = tm.ae.decode(md_hat)

        ae_blk = ae_rec[:, j * M : (j + 1) * M].cpu().numpy()
        sae_blk = sae_rec[:, j * M : (j + 1) * M].cpu().numpy()
        md_blk = md_rec[:, j * M : (j + 1) * M].cpu().numpy()

        # GT ring
        gt_closed = np.concatenate([gt_np, gt_np[:1]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=gt_closed[:, 0],
                y=gt_closed[:, 1],
                z=gt_closed[:, 2],
                mode="lines",
                line=dict(color=COLORS[j], width=5),
                opacity=0.3,
                name=f"GT {FEATURE_NAMES[j]}",
                showlegend=(idx == 0),
                legendgroup=f"gt_{j}",
            ),
            row=1,
            col=col,
        )
        # AE
        fig.add_trace(
            go.Scatter3d(
                x=ae_blk[:, 0],
                y=ae_blk[:, 1],
                z=ae_blk[:, 2],
                mode="markers",
                marker=dict(color=COLORS[j], size=2, opacity=0.6),
                name=f"AE {FEATURE_NAMES[j]}",
                showlegend=(idx == 0),
                legendgroup=f"ae_r_{j}",
            ),
            row=1,
            col=col,
        )
        # SAE
        fig.add_trace(
            go.Scatter3d(
                x=sae_blk[:, 0],
                y=sae_blk[:, 1],
                z=sae_blk[:, 2],
                mode="markers",
                marker=dict(color=COLORS[j], size=2, opacity=0.6, symbol="diamond"),
                name=f"SAE {FEATURE_NAMES[j]}",
                showlegend=(idx == 0),
                legendgroup=f"sae_r_{j}",
            ),
            row=1,
            col=col,
        )
        # mdSAE
        fig.add_trace(
            go.Scatter3d(
                x=md_blk[:, 0],
                y=md_blk[:, 1],
                z=md_blk[:, 2],
                mode="markers",
                marker=dict(color=COLORS[j], size=2, opacity=0.6, symbol="square"),
                name=f"mdSAE {FEATURE_NAMES[j]}",
                showlegend=(idx == 0),
                legendgroup=f"md_r_{j}",
            ),
            row=1,
            col=col,
        )

    # Linear features: sweep [0, 1]
    lin_vals = torch.linspace(0, 1, n_recon, device=DEVICE)
    for j in range(K_LINEAR):
        feat_idx = K_SPHERES + j
        full_in = make_single_feature_input_linear(j, lin_vals)
        gt_v = lin_vals.cpu().numpy()

        with torch.no_grad():
            ae_rec = tm.ae.decode(tm.ae.encode(full_in))
            z_lat = tm.ae.encode(full_in)
            sae_rec = tm.ae.decode(sae.decode(sae.encode(z_lat)))
            md_hat, _, _, _ = mdsae.forward(z_lat)
            md_rec = tm.ae.decode(md_hat)

        ae_v = ae_rec[:, N_SPHERE_FEATURES + j].cpu().numpy()
        sae_v = sae_rec[:, N_SPHERE_FEATURES + j].cpu().numpy()
        md_v = md_rec[:, N_SPHERE_FEATURES + j].cpu().numpy()
        y_off = j * 0.5

        fig.add_trace(
            go.Scatter3d(
                x=gt_v,
                y=np.full_like(gt_v, y_off),
                z=np.zeros_like(gt_v),
                mode="lines",
                line=dict(color=COLORS[feat_idx], width=5),
                opacity=0.3,
                name=f"GT {FEATURE_NAMES[feat_idx]}",
                showlegend=(idx == 0),
                legendgroup=f"gt_{feat_idx}",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=ae_v,
                y=np.full_like(ae_v, y_off),
                z=np.zeros_like(ae_v),
                mode="markers",
                marker=dict(color=COLORS[feat_idx], size=3, opacity=0.6),
                name=f"AE {FEATURE_NAMES[feat_idx]}",
                showlegend=(idx == 0),
                legendgroup=f"ae_r_{feat_idx}",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=sae_v,
                y=np.full_like(sae_v, y_off + 0.05),
                z=np.zeros_like(sae_v),
                mode="markers",
                marker=dict(
                    color=COLORS[feat_idx], size=3, opacity=0.6, symbol="diamond"
                ),
                name=f"SAE {FEATURE_NAMES[feat_idx]}",
                showlegend=(idx == 0),
                legendgroup=f"sae_r_{feat_idx}",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=md_v,
                y=np.full_like(md_v, y_off + 0.1),
                z=np.zeros_like(md_v),
                mode="markers",
                marker=dict(
                    color=COLORS[feat_idx], size=3, opacity=0.6, symbol="square"
                ),
                name=f"mdSAE {FEATURE_NAMES[feat_idx]}",
                showlegend=(idx == 0),
                legendgroup=f"md_r_{feat_idx}",
            ),
            row=1,
            col=col,
        )

fig.update_layout(
    title_text="Reconstruction Chain: GT / AE / SAESimple / MultiDimSAE",
    height=600,
    width=1200,
    template="plotly_white",
    legend=dict(itemsizing="constant", font=dict(size=11)),
)
update_3d_scenes(
    fig, 2, xaxis_title="d1", yaxis_title="d2", zaxis_title="d3", aspectmode="cube"
)
fig.write_html(sae_recon_path)
fig.show()


# %%
# ── CELL 14: L0 ACTIVATION HEATMAPS ────────────────────────────────────
N_HEAT = 500

for name, tm in models.items():
    sae, mdsae = saes[name], mdsaes[name]
    batch = labeled_samples(N_HEAT * 10)

    single_idx = torch.where(batch.is_single)[0]
    single_fids = batch.feat_id[single_idx]
    sort_order = single_fids.argsort()
    single_idx = single_idx[sort_order]
    single_fids = single_fids[sort_order]

    # Take up to N_HEAT
    if len(single_idx) > N_HEAT:
        single_idx = single_idx[:N_HEAT]
        single_fids = single_fids[:N_HEAT]

    sel_samples = batch.data[single_idx]
    with torch.no_grad():
        z_ae = tm.ae.encode(sel_samples)
        sae_acts = sae.encode(z_ae).cpu().numpy()  # (n, SAE_LATENT)
        md_x_hat, md_z, md_gate, _ = mdsae.forward(z_ae)
        md_acts = (md_z.norm(dim=-1) * md_gate).cpu().numpy()  # (n, MDSAE_FEATURES)

    fids_np = single_fids.cpu().numpy()

    # Sort columns by total firing rate
    sae_col_order = np.argsort(-sae_acts.sum(axis=0))
    sae_acts_sorted = sae_acts[:, sae_col_order]

    md_col_order = np.argsort(-md_acts.sum(axis=0))
    md_acts_sorted = md_acts[:, md_col_order]

    # Find GT feature group boundaries for horizontal lines
    boundaries = []
    for j in range(K_TOTAL):
        idxs = np.where(fids_np == j)[0]
        if len(idxs) > 0:
            boundaries.append(idxs[-1] + 0.5)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["SAESimple Activations", "MultiDimSAE Activations"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(
            z=sae_acts_sorted,
            colorscale="Blues",
            zmin=0,
            x=[f"n{sae_col_order[i]}" for i in range(SAE_LATENT)],
            colorbar=dict(title="act", x=0.45, len=0.8),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=md_acts_sorted,
            colorscale="Oranges",
            zmin=0,
            x=[f"f{md_col_order[i]}" for i in range(MDSAE_FEATURES)],
            colorbar=dict(title="act", x=1.0, len=0.8),
        ),
        row=1,
        col=2,
    )

    # Add feature group boundary lines and labels
    for b in boundaries[:-1]:
        for c in [1, 2]:
            fig.add_hline(
                y=b, line_dash="dash", line_color="red", line_width=1, row=1, col=c
            )

    # Y-axis tick labels at group centers
    ytick_vals = []
    ytick_text = []
    for j in range(K_TOTAL):
        idxs = np.where(fids_np == j)[0]
        if len(idxs) > 0:
            ytick_vals.append(int(idxs.mean()))
            ytick_text.append(FEATURE_NAMES[j])

    fig.update_yaxes(tickvals=ytick_vals, ticktext=ytick_text, row=1, col=1)
    fig.update_yaxes(tickvals=ytick_vals, ticktext=ytick_text, row=1, col=2)
    fig.update_xaxes(title_text="SAE latent (sorted by firing rate)", row=1, col=1)
    fig.update_xaxes(title_text="mdSAE feature (sorted by firing rate)", row=1, col=2)

    fig.update_layout(
        title_text=f"L0 Activation Heatmap — {name}",
        height=600,
        width=1400,
        template="plotly_white",
    )

    heat_path = os.path.join(FIGURES_DIR, f"l0_heatmap_{name}.html")
    ensure_dir_exists(heat_path)
    fig.write_html(heat_path)
    fig.show()
    print(f"  Saved: {heat_path}")


# %%
# ── CELL 15: COACTIVATION HEATMAPS (GT features x SAE features) ────────
for name, tm in models.items():
    sae, mdsae = saes[name], mdsaes[name]
    batch = labeled_samples(10_000)

    with torch.no_grad():
        z_ae = tm.ae.encode(batch.data)
        sae_acts = sae.encode(z_ae)
        _, md_z, md_gate, _ = mdsae.forward(z_ae)
        md_acts = md_z.norm(dim=-1) * md_gate

    # Build + normalize coactivation matrices
    sae_coact = np.zeros((K_TOTAL, SAE_LATENT))
    md_coact = np.zeros((K_TOTAL, MDSAE_FEATURES))
    for j in range(K_TOTAL):
        fm = (batch.is_single & (batch.feat_id == j)).numpy()
        if fm.sum() == 0:
            continue
        sae_coact[j] = sae_acts[fm].mean(0).cpu().numpy()
        md_coact[j] = md_acts[fm].mean(0).cpu().numpy()

    def _row_normalize(mat):
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        return mat / row_sum

    sae_coact_norm = _row_normalize(sae_coact)
    md_coact_norm = _row_normalize(md_coact)
    sae_col_order = np.argsort(-sae_coact_norm.max(axis=0))
    md_col_order = np.argsort(-md_coact_norm.max(axis=0))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["SAESimple Coactivation", "MultiDimSAE Coactivation"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=sae_coact_norm[:, sae_col_order],
            y=FEATURE_NAMES,
            x=[f"n{sae_col_order[i]}" for i in range(SAE_LATENT)],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="frac", x=0.45, len=0.8),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=md_coact_norm[:, md_col_order],
            y=FEATURE_NAMES,
            x=[f"f{md_col_order[i]}" for i in range(MDSAE_FEATURES)],
            colorscale="Oranges",
            zmin=0,
            zmax=1,
            colorbar=dict(title="frac", x=1.0, len=0.8),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="SAE latent", row=1, col=1)
    fig.update_xaxes(title_text="mdSAE feature", row=1, col=2)

    fig.update_layout(
        title_text=f"Coactivation: GT Feature vs SAE Feature — {name}",
        height=400,
        width=1400,
        template="plotly_white",
    )

    coact_path = os.path.join(FIGURES_DIR, f"coactivation_{name}.html")
    ensure_dir_exists(coact_path)
    fig.write_html(coact_path)
    fig.show()
    print(f"  Saved: {coact_path}")


# %%
# ── CELL 16: SUBSPACE WEIGHTS VISUALIZATION (MultiDimSAE V vs GT tilts) ─
for name in models:
    mdsae = mdsaes[name]
    V_np = mdsae.V.detach().cpu().numpy()  # (m, D, p)
    tilts_np = sphere_dist.tilts.cpu().numpy()  # (K_SPHERES, M, sphere_dim+1)

    fig = go.Figure()

    # Unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=x_s,
            y=y_s,
            z=z_s,
            opacity=0.08,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            name="unit sphere",
        )
    )

    # Plot V_i columns as arrows
    for i in range(MDSAE_FEATURES):
        for p_idx in range(MDSAE_SUBDIM):
            vi = V_np[i, :, p_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[0, vi[0]],
                    y=[0, vi[1]],
                    z=[0, vi[2]],
                    mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)], width=4),
                    marker=dict(size=[0, 4], color=COLORS[i % len(COLORS)]),
                    name=f"V_{i} col {p_idx}",
                    legendgroup=f"feat_{i}",
                    showlegend=(p_idx == 0),
                )
            )

    # Overlay GT tilt planes
    for j in range(K_SPHERES):
        tilt = tilts_np[j]  # (M, sphere_dim+1)
        for p_idx in range(tilt.shape[1]):
            vi = tilt[:, p_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[0, vi[0]],
                    y=[0, vi[1]],
                    z=[0, vi[2]],
                    mode="lines+markers",
                    line=dict(color="black", width=6, dash="dash"),
                    marker=dict(size=[0, 6], color="black", symbol="diamond"),
                    name=f"GT tilt {j} col {p_idx}",
                    legendgroup=f"gt_tilt_{j}",
                    showlegend=(p_idx == 0),
                    opacity=0.5,
                )
            )

    fig.update_layout(
        title=f"Subspace Weights vs GT Tilts — {name}",
        scene=dict(
            xaxis_title="d1", yaxis_title="d2", zaxis_title="d3", aspectmode="cube"
        ),
        height=700,
        width=800,
        template="plotly_white",
    )
    sw_path = os.path.join(FIGURES_DIR, f"subspace_weights_{name}.html")
    ensure_dir_exists(sw_path)
    fig.write_html(sw_path)
    fig.show()
    print(f"  Saved: {sw_path}")


# %%
# ── CELL 17: FEATURE ACTIVATION SCATTER (MultiDimSAE z_i projections) ──
for name, tm in models.items():
    mdsae = mdsaes[name]
    batch = labeled_samples(20_000)

    with torch.no_grad():
        z_ae = tm.ae.encode(batch.data)
        _, md_z, md_gate, _ = mdsae.forward(z_ae)

    feature_activity = md_gate.sum(dim=0)
    active_features = torch.where(feature_activity > 0)[0].tolist()
    n_active_feats = len(active_features)

    if n_active_feats == 0:
        print(f"  {name}: WARNING — No active MultiDimSAE features!")
        continue

    n_cols = min(n_active_feats, 5)
    n_rows = (n_active_feats + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"f{i} ({int(feature_activity[i].item())} active)" for i in active_features
        ],
    )

    for plot_idx, feat_i in enumerate(active_features):
        r = plot_idx // n_cols + 1
        c = plot_idx % n_cols + 1

        feat_gate = md_gate[:, feat_i].bool()
        z_feat = md_z[feat_gate, feat_i, :]  # (n_gated, p)
        if z_feat.shape[0] == 0:
            continue
        z_np = z_feat.cpu().numpy()

        feat_mask_full = batch.mask[feat_gate.cpu()]
        feat_n_active = feat_mask_full.sum(dim=1)
        feat_single = feat_n_active == 1
        feat_fid = feat_mask_full.float().argmax(dim=1)

        for j in range(K_TOTAL):
            rm = (feat_single & (feat_fid == j)).numpy()
            if rm.any():
                fig.add_trace(
                    go.Scatter(
                        x=z_np[rm, 0],
                        y=z_np[rm, 1],
                        mode="markers",
                        marker=dict(color=COLORS[j], size=3, opacity=0.5),
                        name=FEATURE_NAMES[j],
                        showlegend=(plot_idx == 0),
                        legendgroup=f"ring_{j}",
                    ),
                    row=r,
                    col=c,
                )

        feat_multi = (feat_n_active > 1).numpy()
        if feat_multi.any():
            fig.add_trace(
                go.Scatter(
                    x=z_np[feat_multi, 0],
                    y=z_np[feat_multi, 1],
                    mode="markers",
                    marker=dict(color="#888888", size=2, opacity=0.3),
                    name="multi",
                    showlegend=(plot_idx == 0),
                    legendgroup="multi",
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        title=f"mdSAE Feature Activation Scatter (z_i projections) — {name}",
        height=300 * n_rows,
        width=300 * n_cols,
        template="plotly_white",
    )
    scat_path = os.path.join(FIGURES_DIR, f"feature_scatter_{name}.html")
    ensure_dir_exists(scat_path)
    fig.write_html(scat_path)
    fig.show()
    print(f"  Saved: {scat_path}")


# %%
# ── CELL 18: PER-FEATURE RECONSTRUCTION MSE BAR CHART ──────────────────
bar_batch = labeled_samples(10_000)

fig = make_subplots(
    rows=1,
    cols=len(models),
    subplot_titles=list(models.keys()),
    horizontal_spacing=0.12,
)

for idx, (name, tm) in enumerate(models.items()):
    col = idx + 1
    sae, mdsae = saes[name], mdsaes[name]

    with torch.no_grad():
        z_ae = tm.ae.encode(bar_batch.data)
        sae_recon = sae.decode(sae.encode(z_ae))
        md_hat, _, _, _ = mdsae.forward(z_ae)

    per_feat_sae = []
    per_feat_md = []
    for j in range(K_TOTAL):
        fm = bar_batch.is_single & (bar_batch.feat_id == j)
        if fm.sum() == 0:
            per_feat_sae.append(0.0)
            per_feat_md.append(0.0)
            continue
        z_f = z_ae[fm]
        per_feat_sae.append(((z_f - sae_recon[fm]) ** 2).sum(-1).mean().item())
        per_feat_md.append(((z_f - md_hat[fm]) ** 2).sum(-1).mean().item())

    fig.add_trace(
        go.Bar(
            x=FEATURE_NAMES,
            y=per_feat_sae,
            name="SAESimple" if idx == 0 else None,
            marker_color=COLORS[0],
            showlegend=(idx == 0),
            legendgroup="sae_bar",
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            x=FEATURE_NAMES,
            y=per_feat_md,
            name="MultiDimSAE" if idx == 0 else None,
            marker_color=COLORS[1],
            showlegend=(idx == 0),
            legendgroup="md_bar",
        ),
        row=1,
        col=col,
    )

fig.update_layout(
    title_text="Per-Feature Reconstruction MSE (Latent Space)",
    barmode="group",
    template="plotly_white",
    height=400,
    width=1200,
)
for i in range(1, len(models) + 1):
    fig.update_yaxes(title_text="MSE", row=1, col=i)

mse_path = os.path.join(FIGURES_DIR, "per_feature_mse.html")
ensure_dir_exists(mse_path)
fig.write_html(mse_path)
fig.show()
print(f"  Saved: {mse_path}")


# %%
print("\nDone! All figures saved to:", FIGURES_DIR)
