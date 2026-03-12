# %%
# ── IMPORTS AND CONFIG ────────────────────────────────────────────────────
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu, MLPEncoder, TiedMLPEncoder
from occhio.distributions import Bump

# Ensure we actually run on mps if available
assert torch.backends.mps.is_available(), "MPS device is not available"
DEVICE = "mps"
SEED = 42
torch.manual_seed(SEED)

N_FEATURES = 7
HIDDEN_DIM = 2
P_ACTIVE = 0.3
BUMP_WIDTH = 1
N_EPOCHS = 20_000
BATCH_SIZE = 256
LR = 3e-4

# %%
# ── DISTRIBUTION ───────────────────────────────────────────────────────
AMP_LOW = 0.3
NOISE_STD = 0.1

dist = Bump(
    n_features=N_FEATURES,
    p_active=P_ACTIVE,
    bump_width=BUMP_WIDTH,
    amp_low=AMP_LOW,
    noise_std=NOISE_STD,
    generator=torch.Generator().manual_seed(SEED),
)

# Sanity check: print a few bump vectors
print("Bump matrix (each row = one state's activation pattern):")
print(dist._bump_matrix.numpy())


# %%
# ── AUTOENCODERS ─────────────────────────────────────────────────────────
def make_models() -> dict[str, ToyModel]:
    """Build the four architecture variants, each paired with a fresh copy of the distribution."""
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
            embedding=[N_FEATURES, 16, HIDDEN_DIM],
            unembedding=[HIDDEN_DIM, 16, N_FEATURES],
        ),
        "TiedMLPAE": lambda: TiedMLPEncoder(
            dims=[N_FEATURES, 16, HIDDEN_DIM],
        ),
    }

    models = {}
    for name, ae_factory in configs.items():
        dist_gen = torch.Generator().manual_seed(SEED)
        d = Bump(
            n_features=N_FEATURES,
            p_active=P_ACTIVE,
            bump_width=BUMP_WIDTH,
            amp_low=AMP_LOW,
            noise_std=NOISE_STD,
            generator=dist_gen,
        )
        ae = ae_factory()
        models[name] = ToyModel(distribution=d, ae=ae, device=DEVICE)
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
        weight_decay=0.0,  # plain Adam (AdamW with wd=0)
        track_losses=True,
        verbose=True,
    )
    loss_curves[name] = losses
    print(f"  Final loss: {losses[-1]:.6f}")


# %%
#  ── LOSS CURVE PLOT ───────────────────────────────────────────────────────
def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


loss_curves_path = "experiments/bump/loss_curves.html"
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
    title=f"Reconstruction Loss: Bump Feature (n={N_FEATURES}, hidden={HIDDEN_DIM})",
    xaxis_title="Epoch",
    yaxis_title="MSE Loss",
    yaxis_type="log",
    template="plotly_white",
)
fig.write_html(loss_curves_path)
fig.show()

# %%
# ── EMBEDDING VISUALIZATION ──────────────────────────────────────────────
N_VIS = 4096
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

bottleneck_path = "experiments/bump/bottleneck.html"
ensure_dir_exists(bottleneck_path)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=list(models.keys()),
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = idx // 2 + 1, idx % 2 + 1

    # Generate samples and track which state produced each one
    gen = torch.Generator().manual_seed(SEED + 1)
    vis_dist = Bump(
        n_features=N_FEATURES,
        p_active=P_ACTIVE,
        bump_width=BUMP_WIDTH,
        amp_low=AMP_LOW,
        noise_std=NOISE_STD,
        generator=gen,
    )
    samples = vis_dist.sample(N_VIS).to(DEVICE)

    is_active = samples.sum(dim=1) > 0
    states = samples.argmax(dim=1)

    with torch.no_grad():
        z = tm.ae.encode(samples).cpu().numpy()

    is_active_np = is_active.cpu().numpy()
    state_arr = states.cpu().numpy()

    # Zero-center the embeddings per subplot
    z_active = z[is_active_np]
    center = z_active.mean(axis=0) if len(z_active) > 0 else z.mean(axis=0)
    z = z - center

    # Plot inactive samples in grey
    inactive = ~is_active_np
    if inactive.any():
        # Use small dots on all subplots, but big dots only for the first (for legend)
        fig.add_trace(
            go.Scatter(
                x=z[inactive, 0],
                y=z[inactive, 1],
                mode="markers",
                marker=dict(
                    color="lightgrey",
                    size=3,
                    opacity=0.3,
                    sizeref=1,
                    sizemode="diameter",
                ),
                name="inactive",
                showlegend=False,
                legendgroup="inactive",
                marker_symbol="circle",
                marker_line_width=0,
                legendgrouptitle_text=None,
            ),
            row=row,
            col=col,
        )
        if idx == 0:
            # Add invisible trace with large size for legend only
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        color="lightgrey",
                        size=12,
                        opacity=1.0,
                        sizeref=1,
                        sizemode="diameter",
                    ),
                    name="inactive",
                    showlegend=True,
                    legendgroup="inactive",
                    marker_symbol="circle",
                    marker_line_width=0,
                ),
                row=row,
                col=col,
            )

    # Plot active samples colored by state
    for j in range(N_FEATURES):
        mask = is_active_np & (state_arr == j)
        if mask.any():
            # Use small dots on all subplots, big dots only in legend for first subplot
            fig.add_trace(
                go.Scatter(
                    x=z[mask, 0],
                    y=z[mask, 1],
                    mode="markers",
                    marker=dict(
                        color=COLORS[j],
                        size=4,
                        opacity=0.5 if idx != 0 else 1.0,
                    ),
                    name=f"state {j}",
                    showlegend=False,
                    legendgroup=f"state_{j}",
                    marker_symbol="circle",
                    marker_line_width=0,
                ),
                row=row,
                col=col,
            )
            if idx == 0:
                # Add invisible trace with large dot size for legend only
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            color=COLORS[j],
                            size=14,
                            opacity=1.0,
                        ),
                        name=f"state {j}",
                        showlegend=True,
                        legendgroup=f"state_{j}",
                        marker_symbol="circle",
                        marker_line_width=0,
                    ),
                    row=row,
                    col=col,
                )

fig.update_layout(
    title_text=f"Embedding Representations (n={N_FEATURES}, hidden={HIDDEN_DIM})",
    height=900,
    width=900,
    template="plotly_white",
)
for i in range(1, 5):
    fig.update_xaxes(
        title_text="h₁",
        scaleanchor=f"y{i if i > 1 else ''}",
        row=(i - 1) // 2 + 1,
        col=(i - 1) % 2 + 1,
    )
    fig.update_yaxes(title_text="h₂", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

fig.write_html(bottleneck_path)
fig.show()

# %%
# ── AVERAGE PAIRWISE COSINE SIMILARITIES ────────────────────────────────────
cosine_path = "experiments/bump/cosine_similarities.html"
ensure_dir_exists(cosine_path)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=list(models.keys()),
)

for idx, (name, tm) in enumerate(models.items()):
    row, col = idx // 2 + 1, idx % 2 + 1

    gen = torch.Generator().manual_seed(SEED + 2)
    cos_dist = Bump(
        n_features=N_FEATURES,
        p_active=P_ACTIVE,
        bump_width=BUMP_WIDTH,
        amp_low=AMP_LOW,
        noise_std=NOISE_STD,
        generator=gen,
    )
    samples = cos_dist.sample(N_VIS).to(DEVICE)
    is_active = samples.sum(dim=1) > 0
    states = samples.argmax(dim=1)

    with torch.no_grad():
        z = tm.ae.encode(samples)

    # Compute average pairwise cosine similarity between state centroids
    centroids = []
    for j in range(N_FEATURES):
        mask = is_active & (states == j)
        if mask.any():
            centroids.append(z[mask].mean(dim=0))
    centroids = torch.stack(centroids)
    centroids_norm = centroids / centroids.norm(dim=1, keepdim=True)
    cos_matrix = (centroids_norm @ centroids_norm.T).cpu().numpy()

    fig.add_trace(
        go.Heatmap(
            z=cos_matrix,
            x=[f"s{j}" for j in range(len(centroids))],
            y=[f"s{j}" for j in range(len(centroids))],
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
    title_text=f"Pairwise Cosine Similarity of State Centroids (n={N_FEATURES}, hidden={HIDDEN_DIM})",
    height=900,
    width=900,
    template="plotly_white",
)
fig.write_html(cosine_path)
fig.show()

# %%
# --- TESTING ─────────────────────────────────────────────────────────────────
dist.sample(10)

# %%
