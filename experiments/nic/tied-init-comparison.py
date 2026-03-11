# %%
"""Compare TiedMLPEncoder vs MLPEncoder with tied initialization."""

import torch
import numpy as np
import plotly.graph_objects as go
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedMLPEncoder, MLPEncoder
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
N_FEATURES = 200
N_HIDDEN = 20
N_EPOCHS = 50_000
BATCH_SIZE = 512
EVAL_SAMPLES = 10_000
EVAL_FREQ = 200

# Zipf-like per-feature firing probabilities: p_i = 1/(i+2)
p_active = [1.0 / (i + 2) for i in range(N_FEATURES)]

INTERMEDIATES = [2 * N_HIDDEN, 4 * N_HIDDEN, 8 * N_HIDDEN]


def eval_hook(data):
    """Compute eval loss on a large fresh sample every EVAL_FREQ epochs."""
    tm = data["tm"]
    raw = tm.distribution.sample(EVAL_SAMPLES)
    x = raw[0] if isinstance(raw, tuple) else raw
    x = x.to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(raw, x_hat, tm.importances).item()


def per_feature_hook(data):
    """Compute per-feature MSE on one-hot inputs."""
    tm = data["tm"]
    eye = torch.eye(tm.n_features, device=tm.device)
    x_hat = tm.ae(eye)[0]
    per_feature_mse = (eye - x_hat).pow(2).sum(dim=-1)  # (n_features,)
    return per_feature_mse.cpu().numpy()


# %%
# --- TiedMLPEncoder models ---
tied_mlp_eval_losses = {}
tied_mlp_per_feature = {}

for intermediate in INTERMEDIATES:
    gen = torch.Generator(DEVICE).manual_seed(42)
    dist = SparseUniform(N_FEATURES, p_active, device=DEVICE, generator=gen)
    ae = TiedMLPEncoder(
        dims=[N_FEATURES, intermediate, N_HIDDEN],
        device=DEVICE,
        generator=gen,
    )
    tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
    _, hook_results = tm.fit(
        N_EPOCHS,
        batch_size=BATCH_SIZE,
        hooks=[eval_hook, per_feature_hook],
        hook_freq=EVAL_FREQ,
    )
    tied_mlp_eval_losses[intermediate] = hook_results[0]
    tied_mlp_per_feature[intermediate] = hook_results[1]
    print(f"Done TiedMLP d={intermediate}")

# %%
# --- MLPEncoder with tied initialization ---
mlp_tied_eval_losses = {}
mlp_tied_per_feature = {}

for intermediate in INTERMEDIATES:
    gen = torch.Generator(DEVICE).manual_seed(42)
    dist = SparseUniform(N_FEATURES, p_active, device=DEVICE, generator=gen)
    ae = MLPEncoder(
        embedding=[N_FEATURES, intermediate, N_HIDDEN],
        unembedding=[N_HIDDEN, intermediate, N_FEATURES],
        tied_initialization=True,
        device=DEVICE,
        generator=gen,
    )
    tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
    _, hook_results = tm.fit(
        N_EPOCHS,
        batch_size=BATCH_SIZE,
        hooks=[eval_hook, per_feature_hook],
        hook_freq=EVAL_FREQ,
    )
    mlp_tied_eval_losses[intermediate] = hook_results[0]
    mlp_tied_per_feature[intermediate] = hook_results[1]
    print(f"Done MLPEncoder(tied_init) d={intermediate}")

# %%
# --- Plot eval loss curves ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]
fig = go.Figure()
for intermediate in INTERMEDIATES:
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=tied_mlp_eval_losses[intermediate],
            name=f"TiedMLP (d={intermediate})",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=mlp_tied_eval_losses[intermediate],
            name=f"MLPEncoder tied_init (d={intermediate})",
            opacity=0.7,
            line=dict(dash="dash"),
        )
    )
fig.update_layout(
    title=f"Eval Loss — TiedMLP vs MLPEncoder(tied_init) (N={N_FEATURES}, m={N_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
    yaxis_range=[None, np.log10(0.6)],
)
fig.show()

# %%
# --- Per-feature reconstruction loss at final epoch ---
feat_norm = np.linspace(0, 1, N_FEATURES)

for intermediate in INTERMEDIATES:
    tied_final = np.array(tied_mlp_per_feature[intermediate])[-1]
    mlp_final = np.array(mlp_tied_per_feature[intermediate])[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(N_FEATURES)),
            y=tied_final,
            name="TiedMLP",
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Bar(
            x=list(range(N_FEATURES)),
            y=mlp_final,
            name="MLPEncoder tied_init",
            opacity=0.6,
        )
    )
    fig.update_layout(
        title=f"Final Per-Feature MSE — d={intermediate}",
        xaxis_title="Feature index",
        yaxis_title="Per-feature MSE",
        barmode="overlay",
    )
    fig.show()

# %%
# --- Feature count summary ---
for intermediate in INTERMEDIATES:
    tied_final = np.array(tied_mlp_per_feature[intermediate])[-1]
    mlp_final = np.array(mlp_tied_per_feature[intermediate])[-1]
    n_tied = int((tied_final < 0.5).sum())
    n_mlp = int((mlp_final < 0.5).sum())
    print(
        f"d={intermediate}: TiedMLP {n_tied}/{N_FEATURES}, "
        f"MLPEncoder(tied_init) {n_mlp}/{N_FEATURES} features with MSE < 0.5"
    )

# %%
