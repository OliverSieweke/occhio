# %%
"""Geometry experiments on TiedLinearRelu — null baseline for MLP encoder comparisons."""

import torch
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
N_FEATURES = 200
N_HIDDEN = 20
N_EPOCHS = 150_000
BATCH_SIZE = 512
EVAL_SAMPLES = 10_000
EVAL_FREQ = 200

# Zipf-like per-feature firing probabilities: p_i = 1/(i+2)
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
# --- Train TiedLinearRelu model ---
gen = torch.Generator(DEVICE).manual_seed(42)
dist = SparseUniform(N_FEATURES, p_active, device=DEVICE, generator=gen)
ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE, generator=gen)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

_, hook_results = tm.fit(
    N_EPOCHS,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
)
eval_losses = hook_results[0]
per_feature = hook_results[1]

# %%
# --- Encoder linearity deviation: encode(α·e_i) - α·encode(e_i) ---
# Should be exactly zero for TiedLinearRelu (encoder is linear by construction)
alphas = torch.linspace(0, 1, 20, device=DEVICE)

with torch.no_grad():
    eye = torch.eye(N_FEATURES, device=DEVICE)
    enc_ei = ae.encode(eye)  # (N_FEATURES, N_HIDDEN)
    enc_0 = ae.encode(torch.zeros(N_FEATURES, device=DEVICE))  # (N_HIDDEN,)

    # Sum deviation over all alphas for each feature
    total_deviation = torch.zeros(N_FEATURES, device=DEVICE)
    for alpha in alphas:
        enc_alpha_ei = ae.encode(alpha * eye)
        expected = alpha * enc_ei + (1 - alpha) * enc_0
        total_deviation += (enc_alpha_ei - expected).abs().mean(dim=-1)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=list(range(N_FEATURES)),
        y=total_deviation.cpu().numpy(),
    )
)
fig.update_layout(
    title="Encoder Nonlinearity — TiedLinearRelu (null baseline)",
    xaxis_title="Feature index",
    yaxis_title="mean |enc(αeᵢ) − [α·enc(eᵢ) + (1−α)·enc(0)]|",
)
fig.show()

# %%
# --- Encoder additivity deviation: enc(eᵢ+eⱼ) vs enc(eᵢ) + enc(eⱼ) - enc(0) ---
# For an affine encoder f(x+y) = f(x) + f(y) - f(0); deviation measures nonlinearity
N_PAIRS = 1000  # random pairs to sample
with torch.no_grad():
    eye = torch.eye(N_FEATURES, device=DEVICE)
    enc_ei = ae.encode(eye)  # (N_FEATURES, N_HIDDEN)
    enc_0 = ae.encode(torch.zeros(N_FEATURES, device=DEVICE))  # (N_HIDDEN,)

    # Sample random (i, j) pairs with i < j
    rng = np.random.default_rng(0)
    pairs_i = rng.integers(0, N_FEATURES, size=N_PAIRS)
    pairs_j = rng.integers(0, N_FEATURES, size=N_PAIRS)
    # Ensure i != j
    mask = pairs_i == pairs_j
    pairs_j[mask] = (pairs_j[mask] + 1) % N_FEATURES

    # encode(e_i + e_j)
    x_ij = eye[pairs_i] + eye[pairs_j]  # (N_PAIRS, N_FEATURES)
    enc_ij = ae.encode(x_ij)  # (N_PAIRS, N_HIDDEN)

    # encode(e_i) + encode(e_j) - encode(0)
    enc_sum = enc_ei[pairs_i] + enc_ei[pairs_j] - enc_0  # (N_PAIRS, N_HIDDEN)

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
    title="Encoder Additivity Deviation — TiedLinearRelu (null baseline)",
    xaxis_title="Feature index",
    yaxis_title="mean |enc(eᵢ+eⱼ) − enc(eᵢ) − enc(eⱼ) + enc(0)|",
)
fig.show()

# %%
# --- Metric tensor variation: g(x) = J(x)^T J(x), compare g(0) vs g(z) ---
N_METRIC_SAMPLES = 2048

# Jacobian at the origin
x0 = torch.zeros(N_FEATURES, device=DEVICE, requires_grad=True)
J0 = torch.autograd.functional.jacobian(ae.encode, x0)  # (N_HIDDEN, N_FEATURES)
g0 = J0 @ J0.T  # (N_HIDDEN, N_HIDDEN)

# Sample encoded points and compute metric at each
dist_metric = SparseUniform(N_FEATURES, p_active, device=DEVICE)
x_samples = dist_metric.sample(N_METRIC_SAMPLES).to(
    DEVICE
)  # (N_METRIC_SAMPLES, N_FEATURES)

metric_diffs = []
for i in range(N_METRIC_SAMPLES):
    xi = x_samples[i].requires_grad_(True)
    Ji = torch.autograd.functional.jacobian(ae.encode, xi)  # (N_HIDDEN, N_FEATURES)
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
    title="Metric Tensor Variation — TiedLinearRelu (null baseline)",
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
    title="Metric Variation vs Active Features — TiedLinearRelu (null baseline)",
    xaxis_title="Number of active features in input",
    yaxis_title="||g(0) − g(z)||_F / ||g(0)||_F",
)
fig.show()

# %%
# --- Eigenspectrum of g(z) across samples ---
all_eigenvalues = []  # list of (N_HIDDEN,) arrays
for i in range(N_METRIC_SAMPLES):
    xi = x_samples[i].requires_grad_(True)
    Ji = torch.autograd.functional.jacobian(ae.encode, xi)
    gi = Ji @ Ji.T
    eigvals = torch.linalg.eigvalsh(gi.cpu()).numpy()  # sorted ascending
    all_eigenvalues.append(eigvals)

all_eigenvalues = np.array(all_eigenvalues)  # (N_METRIC_SAMPLES, N_HIDDEN)

# Eigenspectrum at origin for reference
eigvals_origin = torch.linalg.eigvalsh(g0.cpu()).numpy()

fig = go.Figure()
for k in range(N_HIDDEN):
    fig.add_trace(
        go.Box(
            y=all_eigenvalues[:, k],
            name=f"λ_{k}",
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
    title="Eigenspectrum of g(z) = JᵀJ — TiedLinearRelu (null baseline)",
    xaxis_title="Eigenvalue index (ascending)",
    yaxis_title="Eigenvalue",
    yaxis_type="log",
)
fig.show()

# %%
