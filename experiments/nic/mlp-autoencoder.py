# %%
"""Compare TiedLinearRelu vs MLPEncoder on Zipf-sparse SparseUniform distribution."""

import torch
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu, TiedMLPEncoder
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
N_FEATURES = 200
N_HIDDEN = 20
N_EPOCHS = 150_000
BATCH_SIZE = 512
EVAL_SAMPLES = 10_000
EVAL_FREQ = 200

# Zipf-like per-feature firing probabilities: p_i = 1/(i+1)
p_active = [1.0 / (i + 2) for i in range(N_FEATURES)]


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
# --- TiedLinearRelu model ---
gen1 = torch.Generator(DEVICE).manual_seed(42)
dist1 = SparseUniform(N_FEATURES, p_active, device=DEVICE, generator=gen1)
ae1 = TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE, generator=gen1)
tm_tied = ToyModel(distribution=dist1, ae=ae1, device=DEVICE)

_, hook_results_tied = tm_tied.fit(
    N_EPOCHS,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
)
eval_losses_tied = hook_results_tied[0]
per_feature_tied = hook_results_tied[1]

# %%
# --- TiedMLPEncoder models with varying intermediate sizes ---
INTERMEDIATES = [2 * N_HIDDEN, 4 * N_HIDDEN, 8 * N_HIDDEN]
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

# %%
# --- Plot eval loss curves ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=eval_epochs, y=eval_losses_tied, name="TiedLinearRelu", opacity=0.7)
)
for intermediate, losses in tied_mlp_eval_losses.items():
    fig.add_trace(
        go.Scatter(
            x=eval_epochs, y=losses, name=f"TiedMLP (d={intermediate})", opacity=0.7
        )
    )
fig.update_layout(
    title=f"Eval Loss (10k samples) — Zipf SparseUniform (N={N_FEATURES}, m={N_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
    yaxis_range=[None, np.log10(0.6)],
)
fig.show()

# %%
# --- Per-feature reconstruction loss plots ---
# Color by feature index (normalized to [0, 1])
feat_norm = np.linspace(0, 1, N_FEATURES)

# Collect all models: (name, per_feature_snapshots)
all_models = [("TiedLinearRelu", per_feature_tied)]
for intermediate in INTERMEDIATES:
    all_models.append(
        (f"TiedMLP (d={intermediate})", tied_mlp_per_feature[intermediate])
    )

for model_name, snapshots in all_models:
    # snapshots is a list of arrays, each (n_features,), one per eval point
    feature_curves = np.array(snapshots).T  # (n_features, n_eval_points)

    fig = go.Figure()
    for feat_idx in range(N_FEATURES):
        fig.add_trace(
            go.Scatter(
                x=eval_epochs,
                y=feature_curves[feat_idx],
                mode="lines",
                line=dict(
                    color=f"rgb({int(255 * (1 - feat_norm[feat_idx]))}, 40, {int(255 * feat_norm[feat_idx])})",
                    width=1,
                ),
                opacity=0.1,
                showlegend=False,
                hovertemplate=f"Feature {feat_idx} (p={p_active[feat_idx]:.4f})<br>"
                "Epoch: %{x}<br>MSE: %{y:.4e}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Per-Feature Reconstruction MSE — {model_name}",
        xaxis_title="Epoch",
        # xaxis_type="log",
        yaxis_title="Per-feature MSE",
    )
    fig.show()


# %%
for model_name, snapshots in all_models:
    feature_curves = np.array(snapshots).T  # (n_features, n_eval_points)
    # Count features with final reconstruction MSE < 0.5
    final_mse = feature_curves[:, -1]
    n_below = int((final_mse < 0.5).sum())
    print(f"{model_name}: {n_below}/{N_FEATURES} features with MSE < 0.5")

# %%
# --- Encoder linearity deviation: encode(α·e_i) - α·encode(e_i) ---
# Only meaningful for TiedMLPEncoder models (TiedLinearRelu encoder is linear by construction)
alphas = torch.linspace(0, 1, 20, device=DEVICE)

# Use the last TiedMLPEncoder (d=8*N_HIDDEN) as representative
ae_test = ae  # last ae from the loop above
with torch.no_grad():
    eye = torch.eye(N_FEATURES, device=DEVICE)
    enc_ei = ae_test.encode(eye)  # (N_FEATURES, N_HIDDEN)

    # Sum deviation over all alphas for each feature
    total_deviation = torch.zeros(N_FEATURES, device=DEVICE)
    for alpha in alphas:
        enc_alpha_ei = ae_test.encode(alpha * eye)
        total_deviation += (enc_alpha_ei - alpha * enc_ei).mean(dim=-1)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=list(range(N_FEATURES)),
        y=total_deviation.cpu().numpy(),
    )
)
fig.update_layout(
    title=f"Encoder Nonlinearity — TiedMLP (d={INTERMEDIATES[-1]})",
    xaxis_title="Feature index",
    yaxis_title="mean |enc(a * eᵢ) − a * enc(eᵢ)|",
)
fig.show()

# %%
# --- Encoder additivity deviation: encode(e_i + e_j) - encode(e_i) - encode(e_j) ---
# For a linear encoder this is zero; measures how much the MLP encoder creates cross-feature interference
N_PAIRS = 1000  # random pairs to sample
with torch.no_grad():
    eye = torch.eye(N_FEATURES, device=DEVICE)
    enc_ei = ae_test.encode(eye)  # (N_FEATURES, N_HIDDEN)

    # Sample random (i, j) pairs with i < j
    rng = np.random.default_rng(0)
    pairs_i = rng.integers(0, N_FEATURES, size=N_PAIRS)
    pairs_j = rng.integers(0, N_FEATURES, size=N_PAIRS)
    # Ensure i != j
    mask = pairs_i == pairs_j
    pairs_j[mask] = (pairs_j[mask] + 1) % N_FEATURES

    # encode(e_i + e_j)
    x_ij = eye[pairs_i] + eye[pairs_j]  # (N_PAIRS, N_FEATURES)
    enc_ij = ae_test.encode(x_ij)  # (N_PAIRS, N_HIDDEN)

    # encode(e_i) + encode(e_j)
    enc_sum = enc_ei[pairs_i] + enc_ei[pairs_j]  # (N_PAIRS, N_HIDDEN)

    # Per-pair deviation
    pair_deviation = (enc_ij - enc_sum).abs().sum(dim=-1).cpu().numpy()  # (N_PAIRS,)

    # Accumulate per-feature (each feature appears in multiple pairs)
    per_feature_interference = np.zeros(N_FEATURES)
    per_feature_count = np.zeros(N_FEATURES)
    for k in range(N_PAIRS):
        per_feature_interference[pairs_i[k]] += pair_deviation[k]
        per_feature_interference[pairs_j[k]] += pair_deviation[k]
        per_feature_count[pairs_i[k]] += 1
        per_feature_count[pairs_j[k]] += 1
    per_feature_count = np.maximum(per_feature_count, 1)
    per_feature_interference /= per_feature_count  # average per feature

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=list(range(N_FEATURES)),
        y=per_feature_interference,
    )
)
fig.update_layout(
    title=f"Encoder Additivity Deviation — TiedMLP (d={INTERMEDIATES[-1]})",
    xaxis_title="Feature index",
    yaxis_title="mean |enc(eᵢ+eⱼ) − enc(eᵢ) − enc(eⱼ)|",
)
fig.show()

# %%
# --- Metric tensor variation: g(x) = J(x)^T J(x), compare g(0) vs g(z) ---
N_METRIC_SAMPLES = 2048

# Jacobian at the origin
x0 = torch.zeros(N_FEATURES, device=DEVICE, requires_grad=True)
J0 = torch.autograd.functional.jacobian(ae_test.encode, x0)  # (N_HIDDEN, N_FEATURES)
g0 = J0 @ J0.T  # (N_HIDDEN, N_HIDDEN)

# Sample encoded points and compute metric at each
dist_metric = SparseUniform(N_FEATURES, p_active, device=DEVICE)
x_samples = dist_metric.sample(N_METRIC_SAMPLES).to(
    DEVICE
)  # (N_METRIC_SAMPLES, N_FEATURES)

metric_diffs = []
for i in range(N_METRIC_SAMPLES):
    xi = x_samples[i].requires_grad_(True)
    Ji = torch.autograd.functional.jacobian(
        ae_test.encode, xi
    )  # (N_HIDDEN, N_FEATURES)
    gi = Ji @ Ji.T
    metric_diffs.append(torch.linalg.norm(g0 - gi, ord="fro").item())

metric_diffs = np.array(metric_diffs)
g0_norm = torch.linalg.norm(g0, ord="fro").item()
metric_diffs_normed = metric_diffs / g0_norm
print(f"||g(0)|| = {g0_norm:.4f}")
print(
    f"mean ||g(0) - g(z)|| / ||g(0)|| = {metric_diffs_normed.mean():.4f} ± {metric_diffs_normed.std():.4f}"
)

fig = go.Figure()
fig.add_trace(go.Histogram(x=metric_diffs_normed, nbinsx=30))
fig.update_layout(
    title=f"Metric Tensor Variation — TiedMLP (d={INTERMEDIATES[-1]})",
    xaxis_title="||g(0) − g(z)||_F / ||g(0)||_F",
    yaxis_title="Count",
)
fig.show()

# %%
# --- Metric variation vs number of active features ---
n_active = (x_samples > 0).sum(dim=-1).cpu().numpy()  # (N_METRIC_SAMPLES,)

# Group by number of active features and compute mean/std of normed metric diff
by_n_active = defaultdict(list)
for i in range(N_METRIC_SAMPLES):
    by_n_active[int(n_active[i])].append(metric_diffs_normed[i])

counts_sorted = sorted(by_n_active.keys())
means = [np.mean(by_n_active[k]) for k in counts_sorted]
stds = [np.std(by_n_active[k]) for k in counts_sorted]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=counts_sorted,
        y=means,
        error_y=dict(type="data", array=stds, visible=True),
        mode="markers+lines",
    )
)
fig.update_layout(
    title=f"Metric Variation vs Active Features — TiedMLP (d={INTERMEDIATES[-1]})",
    xaxis_title="Number of active features in input",
    yaxis_title="||g(0) − g(z)||_F / ||g(0)||_F",
)
fig.show()

# %%
# --- Eigenspectrum of g(z) across samples ---
all_eigenvalues = []  # list of (N_HIDDEN,) arrays
for i in range(N_METRIC_SAMPLES):
    xi = x_samples[i].requires_grad_(True)
    Ji = torch.autograd.functional.jacobian(ae_test.encode, xi)
    gi = Ji @ Ji.T
    eigvals = torch.linalg.eigvalsh(gi.cpu()).numpy()  # sorted ascending
    all_eigenvalues.append(eigvals)

all_eigenvalues = np.array(all_eigenvalues)  # (N_METRIC_SAMPLES, N_HIDDEN)

# Eigenspectrum at origin for reference
eigvals_origin = torch.linalg.eigvalsh(g0.cpu()).numpy()

fig = go.Figure()
for k in range(N_HIDDEN):
    fig.add_trace(
        go.Violin(
            y=all_eigenvalues[:, k],
            name=f"λ_{k}",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
        )
    )
# Overlay origin eigenvalues as markers
fig.add_trace(
    go.Scatter(
        x=[f"λ_{k}" for k in range(N_HIDDEN)],
        y=eigvals_origin,
        mode="markers",
        marker=dict(color="red", size=8, symbol="x"),
        name="g(0) eigenvalues",
    )
)
fig.update_layout(
    title=f"Eigenspectrum of g(z) = JᵀJ — TiedMLP (d={INTERMEDIATES[-1]})",
    xaxis_title="Eigenvalue index (ascending)",
    yaxis_title="Eigenvalue",
    yaxis_type="log",
)
fig.show()

# %%
