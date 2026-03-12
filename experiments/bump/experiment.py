# %% Cell 1: Imports and config
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

# %% Cell 2: Distribution
dist = Bump(
    n_features=N_FEATURES,
    p_active=P_ACTIVE,
    bump_width=BUMP_WIDTH,
    generator=torch.Generator().manual_seed(SEED),
)

# Sanity check: print a few bump vectors
print("Bump matrix (each row = one state's activation pattern):")
print(dist._bump_matrix.numpy())


# %% Cell 3: Four autoencoder architectures
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
            generator=dist_gen,
        )
        ae = ae_factory()
        models[name] = ToyModel(distribution=d, ae=ae, device=DEVICE)
    return models


models = make_models()
for name, tm in models.items():
    n_params = sum(p.numel() for p in tm.ae.parameters())
    print(f"{name}: {n_params} parameters")

# %% Cell 4: Training loop
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

# %% Cell 5: Loss curve plot


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

# %% Cell 6: Bottleneck visualization
N_VIS = 2048
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
        n_features=N_FEATURES, p_active=P_ACTIVE, bump_width=BUMP_WIDTH, generator=gen
    )
    samples = vis_dist.sample(N_VIS).to(DEVICE)

    is_active = samples.sum(dim=1) > 0
    states = samples.argmax(dim=1)

    with torch.no_grad():
        z = tm.ae.encode(samples).cpu().numpy()

    is_active_np = is_active.cpu().numpy()
    state_arr = states.cpu().numpy()

    # Plot inactive samples in grey
    inactive = ~is_active_np
    if inactive.any():
        fig.add_trace(
            go.Scatter(
                x=z[inactive, 0],
                y=z[inactive, 1],
                mode="markers",
                marker=dict(color="lightgrey", size=4, opacity=0.3),
                name="inactive",
                showlegend=(idx == 0),
                legendgroup="inactive",
            ),
            row=row,
            col=col,
        )

    # Plot active samples colored by state
    for j in range(N_FEATURES):
        mask = is_active_np & (state_arr == j)
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=z[mask, 0],
                    y=z[mask, 1],
                    mode="markers",
                    marker=dict(color=COLORS[j], size=4, opacity=0.6),
                    name=f"state {j}",
                    showlegend=(idx == 0),
                    legendgroup=f"state_{j}",
                ),
                row=row,
                col=col,
            )

fig.update_layout(
    title_text=f"Bottleneck Representations (n={N_FEATURES}, hidden={HIDDEN_DIM})",
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
# --- TESTING ────────────────────────────────────────────────────────────────
dist.sample(10)
# %%
