# %%
"""Compare Trained AE vs Constructed AE on SparseUniform with zipfian firing probabilities.

Trained AE (TiedLinearRelu): learns W + bias from data via gradient descent.
Constructed AE (SynthAE): unit-norm tied weights positioned roughly orthogonally, bias only.
Uses a soft power-law firing decay (p_max=0.2, p_min=0.01).
SAEs trained via SAELens StandardTrainingSAE.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

# --- Publication-ready figure styling ---
MODEL_COLORS = {
    "Trained AE": "#000c7a",
    "Constructed AE": "#fcba03",
    "Trained AE w/ Unit Norms": "#DC2626",
    "Trained AE w/ Scalar Bias": "#297a58",
}
_THRESH_COLORS = ["#3B82F6", "#F59E0B", "#EF4444"]

_AXIS = dict(
    showgrid=True,
    gridcolor="#E5E7EB",
    showline=True,
    linecolor="#374151",
    linewidth=1.2,
    ticks="outside",
    tickcolor="#374151",
    minor=dict(ticks="outside", tickcolor="#9CA3AF"),
    zeroline=False,
    tickfont=dict(size=15),
    title_font=dict(size=15),
)


def style_fig(fig, nticksx=10, nticksy=8):
    """Apply publication-ready styling."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", size=13, color="#1F2937"),
        title_font=dict(size=15),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            itemsizing="constant",
            font=dict(size=16),
        ),
    )
    fig.update_xaxes(**_AXIS, nticks=nticksx)
    fig.update_yaxes(**_AXIS, nticks=nticksy)
    return fig


def make_epoch_slider(
    epoch_arrays,
    static_arrays,
    epochs,
    titles,
    x=None,
    xlabel="Feature Rank",
    extra_animated=None,
):
    """Build 1×N subplot figure with epoch slider (animated + static scatter).

    Parameters
    ----------
    extra_animated : list of (name, epoch_arrays_dict), optional
        Additional animated series. Each entry's epoch_arrays_dict has the same
        structure as *epoch_arrays* (``{prop: (N, n_snapshots)}``).
    """
    props = list(epoch_arrays.keys())
    n = len(props)
    if extra_animated is None:
        extra_animated = []
    if x is None:
        x = np.arange(epoch_arrays[props[0]].shape[0])

    fig = make_subplots(rows=1, cols=n, subplot_titles=titles)

    # Stable y-ranges across all snapshots
    y_ranges = {}
    for prop in props:
        vals = [epoch_arrays[prop].ravel()]
        if static_arrays and prop in static_arrays:
            vals.append(static_arrays[prop])
        for _, ea in extra_animated:
            if prop in ea:
                vals.append(ea[prop].ravel())
        all_vals = np.concatenate(vals)
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = (hi - lo) * 0.05
        y_ranges[prop] = [lo - pad, hi + pad]

    _ms = 3
    _opacity = 0.7

    # Variable-bandwidth Gaussian: exact at rank 0, very smooth at high ranks
    _n_pts = len(x)
    _xs = np.arange(_n_pts, dtype=float)
    _sigmas = 1.0 + (_n_pts / 4) * (_xs / _n_pts)  # 1 → n/4
    _diffs = _xs[:, None] - _xs[None, :]  # (n, n)
    _W = np.exp(-0.5 * (_diffs / _sigmas[:, None]) ** 2)
    _W /= _W.sum(axis=1, keepdims=True)

    def _smooth(y):
        return _W @ np.asarray(y)

    def _add_dots_and_curve(y, name, color, col, show_legend):
        """Add a scatter + faint trend curve for one series in one subplot."""
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=_ms, opacity=_opacity, color=color),
                showlegend=show_legend,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_smooth(y),
                legendgroup=name,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.388,
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    # Initial traces: dots + curves for each series per subplot
    for i, prop in enumerate(props):
        _add_dots_and_curve(
            epoch_arrays[prop][:, 0],
            "Trained AE",
            MODEL_COLORS["Trained AE"],
            i + 1,
            i == 0,
        )
        for ea_name, ea in extra_animated:
            if prop in ea:
                _add_dots_and_curve(
                    ea[prop][:, 0],
                    ea_name,
                    MODEL_COLORS[ea_name],
                    i + 1,
                    i == 0,
                )
        if static_arrays and prop in static_arrays:
            _add_dots_and_curve(
                static_arrays[prop],
                "Constructed AE",
                MODEL_COLORS["Constructed AE"],
                i + 1,
                i == 0,
            )

    # Slider: update dots + curves for all series
    has_static = static_arrays is not None
    steps = []
    for s in range(len(epochs)):
        y_update = []
        for prop in props:
            y_update.append(epoch_arrays[prop][:, s])
            y_update.append(_smooth(epoch_arrays[prop][:, s]))
            for _, ea in extra_animated:
                if prop in ea:
                    y_update.append(ea[prop][:, s])
                    y_update.append(_smooth(ea[prop][:, s]))
            if has_static and prop in static_arrays:
                y_update.append(static_arrays[prop])
                y_update.append(_smooth(static_arrays[prop]))
        steps.append(
            dict(method="update", label=str(epochs[s]), args=[{"y": y_update}])
        )

    fig.update_layout(
        height=500,
        width=max(600, 350 * n),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Epoch: "),
                pad=dict(t=50),
                steps=steps,
            )
        ],
    )
    # Subplot titles 40% larger than base
    for ann in fig.layout.annotations:
        ann.font = dict(size=22)
    for i, prop in enumerate(props):
        fig.update_yaxes(range=y_ranges[prop], row=1, col=i + 1)
        fig.update_xaxes(title_text=None, row=1, col=i + 1)
    # Single shared x-axis label centered below all subplots (clear of tick labels)
    fig.add_annotation(
        text=xlabel,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.23,
        showarrow=False,
        font=dict(size=22),
    )
    fig.update_layout(margin=dict(b=100))
    return style_fig(fig)


# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 500
D_HIDDEN = 64
N_EPOCHS = 30_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14
EVAL_FREQ = 250

# %%
# --- Zipfian firing probabilities (soft decay) ---
high = 0.3
low = 1.28 / N_FEATURES  # zipfian decay
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")

firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
firing_probs = torch.tensor(firing_probs, dtype=torch.float32)

dist = SparseUniform(N_FEATURES, p_active=firing_probs, device=DEVICE)

# Sort features by firing probability (most frequent first) — used throughout
fp_np = firing_probs.cpu().numpy()
sort_idx = np.argsort(-fp_np)
# Inverse map: rank_of[feature_idx] = its rank by firing probability
rank_of = torch.empty(N_FEATURES, device=DEVICE)
rank_of[torch.from_numpy(sort_idx).to(DEVICE)] = torch.arange(
    N_FEATURES, device=DEVICE, dtype=torch.float32
)

# %%
# Quick sanity check
activations = dist.sample(1024)
print(f"Activations shape: {activations.shape}")
print(f"Mean L0: {(activations > 0).float().sum(dim=-1).mean():.1f}")
print(
    f"Mean L2 norm: {activations.norm(dim=-1).mean():.2f} ± {activations.norm(dim=-1).std():.2f}"
)


# %%
# --- Hooks for evaluation ---
def every(freq, hook):
    """Wrap a hook so it only fires every `freq` epochs (returns None otherwise)."""

    def wrapper(data):
        if data["epoch"] % freq == 0:
            return hook(data)
        return None

    return wrapper


def eval_hook(data):
    """Compute eval loss on a large fresh sample."""
    tm = data["tm"]
    x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(x, x_hat, tm.importances).item()


def per_feature_hook(data):
    """Per-feature reconstruction MSE on one-hot inputs."""
    tm = data["tm"]
    eye = torch.eye(N_FEATURES, device=tm.device)
    x_hat = tm.ae(eye)[0]
    return (eye - x_hat).pow(2).sum(dim=-1).cpu().numpy()


_GEOM_N_GROUPS = 100
_GEOM_GROUP_SIZE = N_FEATURES // _GEOM_N_GROUPS


def _interference_group_matrix(I_sq_np):
    """Compute N_GROUPS×N_GROUPS mean squared interference between frequency groups."""
    I_sorted = I_sq_np[np.ix_(sort_idx, sort_idx)]
    mat = np.zeros((_GEOM_N_GROUPS, _GEOM_N_GROUPS))
    for a in range(_GEOM_N_GROUPS):
        for b in range(_GEOM_N_GROUPS):
            block = I_sorted[
                a * _GEOM_GROUP_SIZE : (a + 1) * _GEOM_GROUP_SIZE,
                b * _GEOM_GROUP_SIZE : (b + 1) * _GEOM_GROUP_SIZE,
            ]
            if a == b:
                mask = ~np.eye(_GEOM_GROUP_SIZE, dtype=bool)
                mat[a, b] = block[mask].mean()
            else:
                mat[a, b] = block.mean()
    return mat


def geometry_hook(data):
    """Capture geometric properties, mean partner rank, and group interference matrix."""
    tm = data["tm"]
    I_sq = tm.interferences_sq.detach().clone()
    I_sq.fill_diagonal_(0)
    mpr = (I_sq * rank_of.unsqueeze(0)).sum(dim=1) / I_sq.sum(dim=1).clamp(min=1e-8)
    I_sq_np = tm.interferences_sq.detach().cpu().numpy()
    return {
        "fd": tm.feature_dimensionalities.detach().cpu().numpy(),
        "fn": tm.feature_norms.detach().cpu().numpy(),
        "ti": tm.total_feature_interferences.detach().cpu().numpy(),
        "bias": tm.ae.b.detach().cpu().numpy() * np.ones(tm.ae.n_features),
        "mpr": mpr.cpu().numpy(),
        "group_mat": _interference_group_matrix(I_sq_np),
    }


def normalize_W(tm):
    """Project W columns to unit norm after each optimizer step."""
    with torch.no_grad():
        tm.ae.W.data /= tm.ae.W.data.norm(dim=0, keepdim=True).clamp(min=1e-8)


# %%
# --- Train Trained AE ---
print("Training Trained AE...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
tm_trained = ToyModel(
    distribution=dist,
    ae=TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1),
    device=DEVICE,
    # hooks=[normalize_W],
)
_, hook_results_trained = tm_trained.fit(
    30000,
    batch_size=BATCH_SIZE,
    hooks=[every(EVAL_FREQ, h) for h in [eval_hook, per_feature_hook, geometry_hook]],
    verbose=True,
)
eval_losses_trained = hook_results_trained[0]
per_feature_trained = hook_results_trained[1]
geometry_trained = hook_results_trained[2]
print(f"  Final eval loss: {eval_losses_trained[-1]:.6f}")

# %%
# --- Train Trained AE (unit norms) ---
print("Training Trained AE w/ Unit Norms...")
gen2 = torch.Generator(DEVICE).manual_seed(SEED)
tm_unit_norm = ToyModel(
    distribution=dist,
    ae=TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen2),
    device=DEVICE,
    hooks=[normalize_W],
)
_, hook_results_unit_norm = tm_unit_norm.fit(
    30000,
    batch_size=BATCH_SIZE,
    hooks=[every(EVAL_FREQ, h) for h in [eval_hook, per_feature_hook, geometry_hook]],
    verbose=True,
)
eval_losses_unit_norm = hook_results_unit_norm[0]
per_feature_unit_norm = hook_results_unit_norm[1]
geometry_unit_norm = hook_results_unit_norm[2]
print(f"  Final eval loss: {eval_losses_unit_norm[-1]:.6f}")

# %%
# --- Train Trained AE (scalar bias) ---
print("Training TiedLinearRelu (scalar bias shared across features)...")
gen_sb = torch.Generator(DEVICE).manual_seed(SEED)
ae_scalar_bias = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen_sb)
ae_scalar_bias.b = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
tm_scalar_bias = ToyModel(distribution=dist, ae=ae_scalar_bias, device=DEVICE)
_, hook_results_scalar_bias = tm_scalar_bias.fit(
    30000,
    batch_size=BATCH_SIZE,
    hooks=[every(EVAL_FREQ, h) for h in [eval_hook, per_feature_hook, geometry_hook]],
    verbose=True,
)
eval_losses_scalar_bias = hook_results_scalar_bias[0]
per_feature_scalar_bias = hook_results_scalar_bias[1]
geometry_scalar_bias = hook_results_scalar_bias[2]
print(f"  Final eval loss: {eval_losses_scalar_bias[-1]:.6f}")

# %%
# --- Constructed AE (bias only) ---
print("Training Constructed AE...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_constructed = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=1000,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_constructed = ToyModel(distribution=dist, ae=ae_constructed, device=DEVICE)
print("Initialized")
N_EPOCHS_CONSTRUCTED = 10_000
_, hook_results_constructed = tm_constructed.fit(
    N_EPOCHS_CONSTRUCTED,
    batch_size=BATCH_SIZE,
    hooks=[every(EVAL_FREQ, h) for h in [eval_hook, per_feature_hook, geometry_hook]],
    verbose=True,
)
eval_losses_constructed = hook_results_constructed[0]
per_feature_constructed = hook_results_constructed[1]
geometry_constructed = hook_results_constructed[2]
loss_constructed = eval_losses_constructed[-1]
pf_constructed = per_feature_constructed[-1]
print(f"  Final eval loss: {loss_constructed:.6f}")

# %% --- SAE training on both models (SAELens Standard SAE) ---
N_DICT = N_FEATURES // 2
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.2
SAE_TRAINING_SAMPLES = 200_000 * SAE_BATCH  # ~200k steps

sae_results = {}

for name, tm in [("Trained AE", tm_trained), ("Constructed AE", tm_constructed)]:
    print(f"\nTraining SAE on {name}...")

    sae_config = StandardTrainingSAEConfig(
        d_in=D_HIDDEN,
        d_sae=N_DICT,
        l1_coefficient=SAE_L1,
        device=DEVICE,
    )
    sae = StandardTrainingSAE(sae_config)

    tm.train_saes(
        {name: sae},
        training_samples=SAE_TRAINING_SAMPLES,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
        verbose=True,
    )

    # Compute metrics
    trained_sae = tm.saes[name].sae
    with torch.no_grad():
        test_x = dist.sample(10_000).to(DEVICE)
        test_hidden = tm.ae.encode(test_x)
        test_z = trained_sae.encode(test_hidden)
        test_recon = trained_sae.decode(test_z)

        # L0 sparsity: mean number of active (> 0) dict elements per sample
        l0 = (test_z > 0).float().sum(dim=-1).mean().item()

        # Dead features: dict elements that never fire
        ever_active = (test_z > 0).any(dim=0)
        n_dead = int((~ever_active).sum().item())
        n_alive = int(ever_active.sum().item())

        # Reconstruction MSE in hidden space
        recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

        # Per-feature faithfulness: encode one-hot, round-trip through SAE
        eye = torch.eye(N_FEATURES, device=DEVICE)
        h_eye = tm.ae.encode(eye)
        h_eye_recon = trained_sae.decode(trained_sae.encode(h_eye))
        per_feat_sae_mse = (h_eye - h_eye_recon).pow(2).sum(dim=-1).cpu().numpy()

        # Explained variance ratio
        total_var = test_hidden.var(dim=0).sum().item()
        residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
        explained_var = 1 - residual_var / total_var

    sae_results[name] = {
        "l0": l0,
        "n_dead": n_dead,
        "n_alive": n_alive,
        "recon_mse": recon_mse,
        "per_feat_sae_mse": per_feat_sae_mse,
        "explained_var": explained_var,
    }
    print(
        f"  L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
    )

# %%
# --- Plot: Eval loss curve ---
# Derive eval_epochs from actual hook data (avoids off-by-one with last epoch)
_n_evals = len(eval_losses_trained)
eval_epochs = sorted(set(range(0, N_EPOCHS, EVAL_FREQ)) | {N_EPOCHS - 1})[:_n_evals]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=eval_epochs,
        y=eval_losses_trained,
        name="Trained AE",
        mode="lines",
        line=dict(width=2, color=MODEL_COLORS["Trained AE"]),
    )
)
fig.add_hline(
    y=loss_constructed,
    line_dash="dash",
    line_color=MODEL_COLORS["Constructed AE"],
    annotation_text="Constructed AE",
)
fig.update_layout(
    title=f"Eval Loss — SparseUniform (N={N_FEATURES}, D={D_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: Per-feature reconstruction MSE (epoch slider) ---

pf_arr = np.array(per_feature_trained).T[sort_idx]  # (N_FEATURES, n_snapshots)
make_epoch_slider(
    epoch_arrays={"mse": pf_arr},
    static_arrays={"mse": pf_constructed[sort_idx]},
    epochs=np.array(eval_epochs),
    titles=["Per-Feature Reconstruction MSE"],
).show()

# %%
# --- Plot: Features recovered (MSE < threshold) vs epoch ---
THRESHOLDS = [0.2, 0.5, 1.0]
COLORS = _THRESH_COLORS

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, COLORS):
    n_recovered_trained = [
        int((np.array(s) < thresh).sum()) for s in per_feature_trained
    ]
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered_trained,
            name=f"Trained AE (τ={thresh})",
            mode="lines",
            line=dict(color=color, width=2),
        )
    )
    n_recovered_constructed = int((pf_constructed < thresh).sum())
    fig.add_hline(
        y=n_recovered_constructed,
        line_dash="dash",
        line_color=color,
    )
fig.update_layout(
    title="Features Recovered (MSE < τ) Over Training",
    xaxis_title="Epoch",
    yaxis_title="# features recovered",
)
style_fig(fig)
fig.show()

# %%
# --- W^T W comparison ---
models = [("Trained AE", tm_trained), ("Constructed AE", tm_constructed)]
models_dict = dict(models)

fig = make_subplots(rows=1, cols=2, subplot_titles=["Trained AE", "Constructed AE"])

for i, (name, tm) in enumerate(
    [
        ("Trained AE", tm_trained),
        ("Constructed AE", tm_constructed),
    ]
):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(z=WtW, colorscale="RdBu_r", zmid=0, showscale=(i == 1)),
        row=1,
        col=i + 1,
    )

fig.update_layout(title="W^T W Comparison", height=700, width=1400)
style_fig(fig)
fig.show()

# %%
# --- Plot: Spectrum of W W^T ---
fig = go.Figure()
for name, tm in models:
    W = tm.W.detach().cpu()
    eigvals = torch.linalg.eigvalsh(W @ W.T).flip(0).numpy()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, D_HIDDEN + 1),
            y=eigvals,
            name=name,
            mode="lines+markers",
            line=dict(width=2, color=MODEL_COLORS[name]),
            marker=dict(size=6, color=MODEL_COLORS[name]),
        )
    )
fig.update_layout(
    title="Spectrum of W W^T",
    xaxis_title="Eigenvalue index",
    yaxis_title="Eigenvalue",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: Geometric properties + bias over training (slider) ---
_geom_props = ["fn", "ti", "bias", "mpr", "fd"]
_geom_titles = [
    "Feature Norms",
    "Total Interference",
    "Learned Bias",
    "Mean Partner Rank",
    "Feature Dimensionalities",
]

geom_arrays = {}
for prop in _geom_props:
    arr = np.array([g[prop] for g in geometry_trained]).T  # (N_FEATURES, n_snapshots)
    geom_arrays[prop] = arr[sort_idx]

geom_arrays_unit_norm = {}
for prop in _geom_props:
    arr = np.array([g[prop] for g in geometry_unit_norm]).T
    geom_arrays_unit_norm[prop] = arr[sort_idx]

n_snapshots = geom_arrays["fd"].shape[1]
geom_epochs = (np.arange(n_snapshots) * EVAL_FREQ).astype(int)

_gc_final = geometry_constructed[-1]
constructed_props = {prop: _gc_final[prop][sort_idx] for prop in _geom_props}

make_epoch_slider(
    epoch_arrays=geom_arrays,
    static_arrays=constructed_props,
    epochs=geom_epochs,
    titles=_geom_titles,
    extra_animated=[("Trained AE w/ Unit Norms", geom_arrays_unit_norm)],
).show()

# %%
# --- Plot: Geometric properties (scalar bias ablation) ---
geom_arrays_scalar_bias = {}
for prop in _geom_props:
    arr = np.array([g[prop] for g in geometry_scalar_bias]).T
    geom_arrays_scalar_bias[prop] = arr[sort_idx]

make_epoch_slider(
    epoch_arrays=geom_arrays,
    static_arrays=constructed_props,
    epochs=geom_epochs,
    titles=_geom_titles,
    extra_animated=[("Trained AE w/ Scalar Bias", geom_arrays_scalar_bias)],
).show()

# %%
# --- Plot: ‖w‖² + b over training (epoch slider) ---
fn2_bias_arr = geom_arrays["fn"] ** 2 + geom_arrays["bias"]  # (N_FEATURES, n_snapshots)
constructed_fn2_bias = constructed_props["fn"] ** 2 + constructed_props["bias"]

make_epoch_slider(
    epoch_arrays={"fn2b": fn2_bias_arr},
    static_arrays={"fn2b": constructed_fn2_bias},
    epochs=geom_epochs,
    titles=["‖w‖² + b"],
).show()

# %%
# --- Frequency-group interference heatmap (epoch slider) ---
group_labels = [str(i * _GEOM_GROUP_SIZE) for i in range(_GEOM_N_GROUPS)]

group_mats_trained = np.array([g["group_mat"] for g in geometry_trained])
# Constructed AE: W is frozen, so interference is static — use final state
group_mat_constructed = geometry_constructed[-1]["group_mat"]

# Stable color range across all snapshots + constructed
_all_group_vals = np.concatenate(
    [group_mats_trained.ravel(), group_mat_constructed.ravel()]
)
_bz_min, _bz_max = float(_all_group_vals.min()), float(_all_group_vals.max())

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Trained AE (animated)", "Constructed AE (static)"]
)
fig.add_trace(
    go.Heatmap(
        z=group_mats_trained[0],
        x=group_labels,
        y=group_labels,
        colorscale="Viridis",
        zmin=_bz_min,
        zmax=_bz_max,
        showscale=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(
        z=group_mat_constructed,
        x=group_labels,
        y=group_labels,
        colorscale="Viridis",
        zmin=_bz_min,
        zmax=_bz_max,
        showscale=True,
    ),
    row=1,
    col=2,
)

_group_steps = []
for s in range(len(group_mats_trained)):
    _group_steps.append(
        dict(
            method="update",
            label=str(geom_epochs[s]),
            args=[{"z": [group_mats_trained[s], group_mat_constructed]}],
        )
    )

for col in range(1, 3):
    fig.update_xaxes(title_text="Feature Rank Group", row=1, col=col)
    fig.update_yaxes(title_text="Partner Rank Group", row=1, col=col)
fig.update_layout(
    title="Mean Squared Interference by Frequency Group (groups of 10)",
    height=700,
    width=1400,
    showlegend=False,
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Epoch: "),
            pad=dict(t=50),
            steps=_group_steps,
        )
    ],
)
style_fig(fig)
fig.show()

# %%
# --- Interference entropy per feature ---
# Normalized entropy of each feature's interference distribution (0=concentrated, 1=uniform)
for name, tm in models:
    I_sq = tm.interferences_sq.detach().clone()
    I_sq.fill_diagonal_(0)
    p = I_sq / I_sq.sum(dim=1, keepdim=True).clamp(min=1e-8)  # normalize rows
    log_p = torch.log(p.clamp(min=1e-12))
    H = -(p * log_p).sum(dim=1)  # Shannon entropy per feature
    H_max = np.log(N_FEATURES - 1)  # max possible entropy (uniform over N-1 partners)
    H_norm = (H / H_max).cpu().numpy()
    print(
        f"{name:20s}  mean_entropy={H_norm.mean():.4f}  "
        f"head_150={H_norm[sort_idx[:150]].mean():.4f}  "
        f"tail_150={H_norm[sort_idx[-150:]].mean():.4f}"
    )

# %%
# --- Summary statistics ---
N_HEAD = 150  # top features by firing probability
N_TAIL = 150  # bottom features by firing probability
expected_l0 = float(firing_probs.sum())
head_idx = sort_idx[:N_HEAD]
tail_idx = sort_idx[-N_TAIL:]

print(f"\n=== Summary  |  E[L0]={expected_l0:.2f}  |  head={N_HEAD}  tail={N_TAIL} ===")
for name, eval_loss, pf in [
    ("Trained AE", eval_losses_trained[-1], per_feature_trained[-1]),
    ("Constructed AE", loss_constructed, pf_constructed),
]:
    final_mse = np.array(pf)
    head_mse, tail_mse = final_mse[head_idx], final_mse[tail_idx]
    for scope, mse_slice in [
        ("all", final_mse),
        ("head", head_mse),
        ("tail", tail_mse),
    ]:
        recovered = "  ".join(
            f"τ={t}: {int((mse_slice < t).sum())}" for t in THRESHOLDS
        )
        label = name if scope == "all" else f"  ({scope})"
        extra = f"  eval_loss={eval_loss:.6f}" if scope == "all" else ""
        print(
            f"{label:25s}{extra}  "
            f"mean_MSE={mse_slice.mean():.4f}  "
            f"recovered=[{recovered}]"
        )

# %% --- Per-feature SAE reconstruction error ---
names = list(sae_results.keys())
fig = go.Figure()
for name in names:
    res = sae_results[name]
    fig.add_trace(
        go.Scatter(
            x=np.arange(N_FEATURES),
            y=res["per_feat_sae_mse"][sort_idx],
            name=name,
            mode="markers",
            marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS[name]),
        )
    )
fig.update_layout(
    title="SAE Per-Feature Reconstruction Error (sorted by firing probability)",
    xaxis_title="Feature Rank ",
    yaxis_title="MSE (hidden space)",
)
style_fig(fig)
fig.show()

# %% --- SAE activations on one-hot features (cosine matched) ---
_cos_match_data = {}
for name in names:
    tm_ref = models_dict[name]
    sae = tm_ref.saes[name].sae
    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm_ref.ae.encode(eye)).cpu().numpy()
        D = tm_ref.ae.encode(eye)
        D_normed = D / D.norm(dim=1, keepdim=True)
        W_dec = sae.W_dec.data
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)
        cosine_sim = (D_normed @ W_dec_normed.T).cpu().numpy()

    feat_idx, dict_idx = linear_sum_assignment(-cosine_sim)
    matched_feats, matched_dicts = set(feat_idx), set(dict_idx)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]
    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts

    acts_matched = sae_acts[np.ix_(row_order, col_order)]
    cos_matched = cosine_sim[np.ix_(row_order, col_order)]
    n_matched = len(feat_idx)
    diag_sum = sum(acts_matched[i, i] for i in range(n_matched))
    total_sum = acts_matched.sum()
    diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
    mean_cos = cosine_sim[feat_idx, dict_idx].mean()
    sae_results[name]["diagonality"] = diagonality

    _cos_match_data[name] = {
        "acts": acts_matched,
        "cos": cos_matched,
        "diag": diagonality,
        "mean_cos": mean_cos,
    }
    print(f"{name}: diag={diagonality:.4f}  mean_cos={mean_cos:.4f}")

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        f"{n} — Activations (diag={_cos_match_data[n]['diag']:.3f})" for n in names
    ]
    + [f"{n} — Cosine Sim (mean={_cos_match_data[n]['mean_cos']:.3f})" for n in names],
    vertical_spacing=0.12,
)
for i, name in enumerate(names):
    d = _cos_match_data[name]
    fig.add_trace(
        go.Heatmap(
            z=d["acts"],
            colorscale="ylgnbu_r",
            showscale=(i == 1),
            colorbar=dict(y=0.78, len=0.4, title="Act"),
        ),
        row=1,
        col=i + 1,
    )
    fig.add_trace(
        go.Heatmap(
            z=d["cos"],
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(i == 1),
            colorbar=dict(y=0.22, len=0.4, title="Cos"),
        ),
        row=2,
        col=i + 1,
    )
for row in range(1, 3):
    for col in range(1, 3):
        fig.update_xaxes(title_text="Matched SAE Dictionary Element", row=row, col=col)
        fig.update_yaxes(title_text="Feature", row=row, col=col)
fig.update_layout(
    title="Cosine-Matched SAE Heatmaps",
    height=1000,
    width=1200,
    showlegend=False,
)
style_fig(fig)
fig.show()

# %% --- SAE evaluation: MCC, detection metrics ---
for name, res in sae_results.items():
    tm = models_dict[name]
    sae = tm.saes[name].sae

    with torch.no_grad():
        # Mean Correlation Coefficient (MCC) — O'Neill et al. (2025)
        # Cosine similarity between SAE decoder columns and ground-truth features
        D = tm.W.detach()  # (D_HIDDEN, N_FEATURES) — columns are feature directions
        W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT) — columns are decoder dirs
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim_raw = (D_norm.T @ W_norm).cpu().numpy()  # (N_FEATURES, N_DICT)
        cos_sim_abs = np.abs(cos_sim_raw)

        # Matching with absolute cosine similarity (ignores sign flips)
        mcc_feat_idx_abs, mcc_dict_idx_abs = linear_sum_assignment(-cos_sim_abs)
        res["mcc_abs_abs"] = float(
            cos_sim_abs[mcc_feat_idx_abs, mcc_dict_idx_abs].mean()
        )
        res["mcc_abs_cos"] = float(
            cos_sim_raw[mcc_feat_idx_abs, mcc_dict_idx_abs].mean()
        )

        # Matching with cosine similarity (no abs, penalises anti-alignment)
        mcc_feat_idx_cos, mcc_dict_idx_cos = linear_sum_assignment(-cos_sim_raw)
        res["mcc_cos_cos"] = float(
            cos_sim_raw[mcc_feat_idx_cos, mcc_dict_idx_cos].mean()
        )
        res["mcc_cos_abs"] = float(
            cos_sim_abs[mcc_feat_idx_cos, mcc_dict_idx_cos].mean()
        )

        # Store both matchings
        res["mcc_feat_idx_abs"] = mcc_feat_idx_abs
        res["mcc_dict_idx_abs"] = mcc_dict_idx_abs
        res["mcc_feat_idx_cos"] = mcc_feat_idx_cos
        res["mcc_dict_idx_cos"] = mcc_dict_idx_cos

        # Detection metrics for both matchings
        det_x = dist.sample(50_000).to(DEVICE)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)  # (50000, N_DICT)

        for suffix, fi, di in [
            ("_abs", mcc_feat_idx_abs, mcc_dict_idx_abs),
            ("_cos", mcc_feat_idx_cos, mcc_dict_idx_cos),
        ]:
            gt_active = det_x[:, fi] > 0
            pred_active = det_z[:, di] > 0

            tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
            fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
            fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()
            tn = (~gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            fpr = fp / (fp + tn + 1e-8)

            res[f"precision{suffix}"] = float(prec.mean())
            res[f"recall{suffix}"] = float(rec.mean())
            res[f"f1{suffix}"] = float(f1.mean())
            res[f"fpr{suffix}"] = float(fpr.mean())
            res[f"precision_per{suffix}"] = prec
            res[f"recall_per{suffix}"] = rec
            res[f"f1_per{suffix}"] = f1
            res[f"fpr_per{suffix}"] = fpr
            res[f"confusion{suffix}"] = np.array(
                [[tn.sum(), fp.sum()], [fn.sum(), tp.sum()]]
            )

    print(
        f"{name}: abs_match(|cos|={res['mcc_abs_abs']:.4f}, cos={res['mcc_abs_cos']:.4f})  "
        f"cos_match(|cos|={res['mcc_cos_abs']:.4f}, cos={res['mcc_cos_cos']:.4f})"
    )

# %% --- SAE summary print ---
print(f"\n{'=' * 80}")
print(f"SAE Overview  |  L1={SAE_L1}  E[L0]={expected_l0:.2f}  head={N_HEAD}")
print(f"{'=' * 80}")
_hdr = f"{'Model':20s}  {'MSE↓':>10s}  {'L0':>6s}  {'Dead':>5s}  {'Alive':>5s}  {'ExplVar↑':>8s}  {'Diag↑':>6s}"
print(_hdr)
print("-" * len(_hdr))
for name, res in sae_results.items():
    print(
        f"{name:20s}  {res['recon_mse']:10.6f}  {res['l0']:6.1f}  "
        f"{res['n_dead']:5d}  {res['n_alive']:5d}  {res['explained_var']:8.4f}  "
        f"{res.get('diagonality', 0):6.4f}"
    )

for match_label, sfx in [("cos-sim", "_cos"), ("|cos-sim|", "_abs")]:
    print(f"\nDetection ({match_label})")
    _dhdr = f"{'Model':20s}  {'Scope':>7s}  {'|cos|↑':>7s}  {'cos↑':>7s}  {'Prec↑':>6s}  {'Rec↑':>6s}  {'F1↑':>6s}  {'FPR↓':>6s}"
    print(_dhdr)
    print("-" * len(_dhdr))
    for name, res in sae_results.items():
        mcc_abs = res.get(f"mcc{sfx}_abs", 0)
        mcc_cos = res.get(f"mcc{sfx}_cos", 0)
        print(
            f"{name:20s}  {'all':>7s}  {mcc_abs:7.4f}  {mcc_cos:7.4f}  "
            f"{res[f'precision{sfx}']:6.4f}  {res[f'recall{sfx}']:6.4f}  "
            f"{res[f'f1{sfx}']:6.4f}  {res[f'fpr{sfx}']:6.4f}"
        )
        for scope, slc in [
            ("head", slice(None, N_HEAD)),
            ("tail", slice(-N_TAIL, None)),
        ]:
            pp = res[f"precision_per{sfx}"][slc]
            pr = res[f"recall_per{sfx}"][slc]
            pf = res[f"f1_per{sfx}"][slc]
            pfpr = res[f"fpr_per{sfx}"][slc]
            n = N_HEAD if scope == "head" else N_TAIL
            print(
                f"{'':20s}  {f'{scope} {n}':>7s}  {'':>7s}  {'':>7s}  "
                f"{pp.mean():6.4f}  {pr.mean():6.4f}  "
                f"{pf.mean():6.4f}  {pfpr.mean():6.4f}"
            )


# %% --- SAE Comparison bar chart ---
_bar_metrics = [
    ("recon_mse", "Recon MSE", True),
    ("l0", "Mean L0", True),
    ("n_dead", "Dead Features", True),
    ("explained_var", "Explained Var", False),
    ("diagonality", "Diagonality", False),
    ("mcc_cos_cos", "MCC (cosine)", False),
]
fig = make_subplots(
    rows=1,
    cols=len(_bar_metrics),
    subplot_titles=[m[1] for m in _bar_metrics],
)
for col, (key, label, lower_better) in enumerate(_bar_metrics, 1):
    for name in names:
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[sae_results[name].get(key, 0)],
                name=name,
                legendgroup=name,
                marker_color=MODEL_COLORS[name],
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
fig.update_layout(title="SAE Comparison", height=400, width=1400, barmode="group")
style_fig(fig)
fig.show()

# %% --- Confusion Matrices ---
fig = make_subplots(
    rows=1,
    cols=len(names),
    subplot_titles=names,
)
for i, name in enumerate(names):
    cm = sae_results[name]["confusion_cos"]
    cm_norm = cm / cm.sum()
    fig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=["Pred Inactive", "Pred Active"],
            y=["GT Inactive", "GT Active"],
            colorscale="Blues",
            showscale=(i == len(names) - 1),
            text=[[f"{v:.4f}" for v in row] for row in cm_norm],
            texttemplate="%{text}",
        ),
        row=1,
        col=i + 1,
    )
fig.update_layout(title="Confusion Matrices (cosine matching)", height=400, width=800)
style_fig(fig)
fig.show()

# %% --- Encoder-based matching (using SAE activations on one-hot inputs) ---
for name in names:
    res = sae_results[name]
    tm = models_dict[name]
    sae = tm.saes[name].sae

    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (N_FEATURES, N_DICT)

        # Match by maximizing total activation on the diagonal
        enc_feat_idx, enc_dict_idx = linear_sum_assignment(-sae_acts)

        res["enc_feat_idx"] = enc_feat_idx
        res["enc_dict_idx"] = enc_dict_idx

        # Mean activation on matched diagonal (encoder analogue of MCC)
        res["enc_mean_act"] = float(sae_acts[enc_feat_idx, enc_dict_idx].mean())

        # Also compute cosine similarity for encoder-matched pairs (for comparison)
        D = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
        W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim_raw = (D_norm.T @ W_norm).cpu().numpy()  # (N_FEATURES, N_DICT)

        res["enc_mcc_cos"] = float(cos_sim_raw[enc_feat_idx, enc_dict_idx].mean())
        res["enc_mcc_abs"] = float(
            np.abs(cos_sim_raw)[enc_feat_idx, enc_dict_idx].mean()
        )

        # Detection metrics for encoder matching
        det_x = dist.sample(50_000).to(DEVICE)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)  # (50000, N_DICT)

        gt_active = det_x[:, enc_feat_idx] > 0
        pred_active = det_z[:, enc_dict_idx] > 0

        tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
        fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
        fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()
        tn = (~gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        res["precision_enc"] = float(prec.mean())
        res["recall_enc"] = float(rec.mean())
        res["f1_enc"] = float(f1.mean())
        res["fpr_enc"] = float(fpr.mean())
        res["precision_per_enc"] = prec
        res["recall_per_enc"] = rec
        res["f1_per_enc"] = f1
        res["fpr_per_enc"] = fpr

    # Compare encoder vs decoder matching: how many pairs agree?
    dec_pairs = set(zip(res["mcc_feat_idx_cos"], res["mcc_dict_idx_cos"]))
    enc_pairs = set(zip(enc_feat_idx, enc_dict_idx))
    n_agree = len(dec_pairs & enc_pairs)

    dec_pairs_abs = set(zip(res["mcc_feat_idx_abs"], res["mcc_dict_idx_abs"]))
    n_agree_abs = len(dec_pairs_abs & enc_pairs)

    res["n_agree_cos"] = n_agree
    res["n_agree_abs"] = n_agree_abs

    print(
        f"{name}: encoder match — mean_act={res['enc_mean_act']:.4f}  "
        f"cos={res['enc_mcc_cos']:.4f}  |cos|={res['enc_mcc_abs']:.4f}"
    )
    print(
        f"  Matching agreement: {n_agree}/{N_FEATURES} with cos-sim, "
        f"{n_agree_abs}/{N_FEATURES} with |cos-sim|"
    )

# %% --- Encoder-matched SAE heatmaps (activations + cosine similarity) ---
_enc_match_data = {}
for name in names:
    res = sae_results[name]
    tm = models_dict[name]
    sae = tm.saes[name].sae

    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()

        # Cosine similarity matrix (same as MCC computation)
        D = tm.W.detach()
        W_dec_t = sae.W_dec.detach().T
        D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = (D_norm.T @ W_norm).cpu().numpy()

    enc_fi, enc_di = res["enc_feat_idx"], res["enc_dict_idx"]
    matched_feats, matched_dicts = set(enc_fi), set(enc_di)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]
    row_order = list(enc_fi) + unmatched_feats
    col_order = list(enc_di) + unmatched_dicts

    acts_matched = sae_acts[np.ix_(row_order, col_order)]
    cos_matched = cos_sim[np.ix_(row_order, col_order)]
    n_matched = len(enc_fi)
    diag_sum = sum(acts_matched[i, i] for i in range(n_matched))
    total_sum = acts_matched.sum()
    diag_enc = diag_sum / total_sum if total_sum > 0 else 0.0
    mean_cos_enc = cos_sim[enc_fi, enc_di].mean()
    res["diagonality_enc"] = diag_enc

    _enc_match_data[name] = {
        "acts": acts_matched,
        "cos": cos_matched,
        "diag": diag_enc,
        "mean_cos": mean_cos_enc,
    }

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        f"{n} — Activations (diag={_enc_match_data[n]['diag']:.3f})" for n in names
    ]
    + [f"{n} — Cosine Sim (mean={_enc_match_data[n]['mean_cos']:.3f})" for n in names],
    vertical_spacing=0.12,
)
for i, name in enumerate(names):
    d = _enc_match_data[name]
    fig.add_trace(
        go.Heatmap(
            z=d["acts"],
            colorscale="ylgnbu_r",
            showscale=(i == 1),
            colorbar=dict(y=0.78, len=0.4, title="Act"),
        ),
        row=1,
        col=i + 1,
    )
    fig.add_trace(
        go.Heatmap(
            z=d["cos"],
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(i == 1),
            colorbar=dict(y=0.22, len=0.4, title="Cos"),
        ),
        row=2,
        col=i + 1,
    )
for row in range(1, 3):
    for col in range(1, 3):
        fig.update_xaxes(title_text="Matched SAE Dictionary Element", row=row, col=col)
        fig.update_yaxes(title_text="Feature", row=row, col=col)
fig.update_layout(
    title="Encoder-Matched SAE Heatmaps",
    height=1000,
    width=1200,
    showlegend=False,
)
style_fig(fig)
fig.show()

# %% --- Comparison table: decoder vs encoder matching ---
print(f"\n{'=' * 90}")
print(f"{'Matching comparison':^90s}")
print(f"{'=' * 90}")

_hdr = (
    f"{'Model':25s}  {'Match':>8s}  {'MCC|cos|↑':>9s}  {'MCCcos↑':>8s}  "
    f"{'Prec↑':>6s}  {'Rec↑':>6s}  {'F1↑':>6s}  {'FPR↓':>6s}  {'Diag↑':>6s}"
)
print(_hdr)
print("-" * len(_hdr))

for name, res in sae_results.items():
    # Decoder matching (cosine similarity)
    print(
        f"{name:25s}  {'dec_cos':>8s}  {res['mcc_cos_abs']:9.4f}  {res['mcc_cos_cos']:8.4f}  "
        f"{res['precision_cos']:6.4f}  {res['recall_cos']:6.4f}  "
        f"{res['f1_cos']:6.4f}  {res['fpr_cos']:6.4f}  {res.get('diagonality', 0):6.4f}"
    )
    # Decoder matching (abs cosine similarity)
    print(
        f"{'':25s}  {'dec_abs':>8s}  {res['mcc_abs_abs']:9.4f}  {res['mcc_abs_cos']:8.4f}  "
        f"{res['precision_abs']:6.4f}  {res['recall_abs']:6.4f}  "
        f"{res['f1_abs']:6.4f}  {res['fpr_abs']:6.4f}  {'':>6s}"
    )
    # Encoder matching
    print(
        f"{'':25s}  {'encoder':>8s}  {res['enc_mcc_abs']:9.4f}  {res['enc_mcc_cos']:8.4f}  "
        f"{res['precision_enc']:6.4f}  {res['recall_enc']:6.4f}  "
        f"{res['f1_enc']:6.4f}  {res['fpr_enc']:6.4f}  {res.get('diagonality_enc', 0):6.4f}"
    )
    print(
        f"{'':25s}  agree: {res['n_agree_cos']}/{N_FEATURES} (cos), "
        f"{res['n_agree_abs']}/{N_FEATURES} (|cos|)"
    )
    print()

# %% --- Per-feature detection metrics (matching method slider) ---
_det_metrics = ["precision", "recall", "f1", "fpr"]
_det_titles = ["Precision", "Recall (TPR)", "F1 Score", "FPR"]
_match_methods = [
    ("Cosine", "_cos", "mcc_feat_idx_cos"),
    ("|Cosine|", "_abs", "mcc_feat_idx_abs"),
    ("Encoder", "_enc", "enc_feat_idx"),
]

fig = make_subplots(rows=1, cols=4, subplot_titles=_det_titles)
x = np.arange(N_FEATURES)

# Initial traces: first matching method for all models × metrics
init_sfx, init_fidx_key = _match_methods[0][1], _match_methods[0][2]
for col, metric in enumerate(_det_metrics):
    for name in names:
        res = sae_results[name]
        feat_order = np.argsort(-fp_np[res[init_fidx_key]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=res[f"{metric}_per{init_sfx}"][feat_order],
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS[name]),
                showlegend=(col == 0),
            ),
            row=1,
            col=col + 1,
        )

# Slider: one step per matching method
steps = []
for label, sfx, fidx_key in _match_methods:
    y_update = []
    for metric in _det_metrics:
        for name in names:
            res = sae_results[name]
            feat_order = np.argsort(-fp_np[res[fidx_key]])
            y_update.append(res[f"{metric}_per{sfx}"][feat_order])
    steps.append(dict(method="update", label=label, args=[{"y": y_update}]))

fig.update_layout(
    title="Per-Feature Detection Metrics (sorted by firing prob.)",
    height=400,
    width=1400,
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Matching: "),
            pad=dict(t=50),
            steps=steps,
        )
    ],
)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature Rank", row=1, col=col)
style_fig(fig)
fig.show()

# %%
# =============================================================================
# EXPORT: Static geometric properties figure (no slider) — vector-ready
# =============================================================================
# Delete everything below this line when done exporting.

import os

_export_props = ["fn", "ti", "bias", "mpr", "fd"]
_export_titles = [
    "Feature Norms",
    "Total Interference",
    "Learned Bias",
    "Mean Partner Rank",
    "Feature Dimensionalities",
]
_n_export = len(_export_props)
_x = np.arange(N_FEATURES)

# Use the FINAL epoch snapshot for each animated series
_final_trained = {p: geom_arrays[p][:, -1] for p in _export_props}
_final_ablation = {p: geom_arrays_scalar_bias[p][:, -1] for p in _export_props}
_final_constructed = {p: constructed_props[p] for p in _export_props}

# Variable-bandwidth Gaussian smooth (same as make_epoch_slider)
_n_pts = len(_x)
_xs_sm = np.arange(_n_pts, dtype=float)
_sigmas_sm = 1.0 + (_n_pts / 4) * (_xs_sm / _n_pts)
_diffs_sm = _xs_sm[:, None] - _xs_sm[None, :]
_W_sm = np.exp(-0.5 * (_diffs_sm / _sigmas_sm[:, None]) ** 2)
_W_sm /= _W_sm.sum(axis=1, keepdims=True)


def _sm(y):
    return _W_sm @ np.asarray(y)


_ms = 3
_opacity = 0.7
_curve_opacity = 0.388

_series = [
    ("Trained AE", _final_trained),
    ("Trained AE w/ Scalar Bias", _final_ablation),
    ("Constructed AE", _final_constructed),
]

_sb_props = ["fn", "ti", "bias", "mpr"]
_sb_titles = [
    "Feature Norms",
    "Total Interference",
    "Learned Bias",
    "Mean Partner Rank",
]
_n_sb = len(_sb_props)

fig_export = make_subplots(rows=1, cols=_n_sb, subplot_titles=_sb_titles)

for i, prop in enumerate(_sb_props):
    for j, (name, data) in enumerate(_series):
        color = MODEL_COLORS[name]
        fig_export.add_trace(
            go.Scatter(
                x=_x,
                y=data[prop],
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=_ms, opacity=_opacity, color=color),
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )
        fig_export.add_trace(
            go.Scatter(
                x=_x,
                y=_sm(data[prop]),
                legendgroup=name,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=_curve_opacity,
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

# Subplot titles
for ann in fig_export.layout.annotations:
    ann.font = dict(size=22)

# Axes: no per-subplot xlabel
for i in range(_n_sb):
    fig_export.update_xaxes(title_text=None, row=1, col=i + 1)

# Shared x-axis label
fig_export.add_annotation(
    text="Feature Rank",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.23,
    showarrow=False,
    font=dict(size=22),
)

fig_export.update_layout(
    height=470,
    width=max(600, 350 * _n_sb),
    margin=dict(b=100),
    showlegend=True,
)

fig_export.update_yaxes(range=[215, 450], row=1, col=4)  # Mean Partner Rank

style_fig(fig_export)
fig_export.show()

# %%
# --- Save as vector (PDF + SVG) ---
_fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(_fig_dir, exist_ok=True)

fig_export.write_image(os.path.join(_fig_dir, "geom_scalar_bias.pdf"), engine="kaleido")
fig_export.write_image(os.path.join(_fig_dir, "geom_scalar_bias.svg"), engine="kaleido")
print(f"Saved to {_fig_dir}/geom_scalar_bias.{{pdf,svg}}")

# %%
# =============================================================================
# EXPORT: Unit Norms ablation — same layout as scalar bias export
# =============================================================================

_final_unit_norm = {p: geom_arrays_unit_norm[p][:, -1] for p in _export_props}

_series_un = [
    ("Trained AE", _final_trained),
    ("Trained AE w/ Unit Norms", _final_unit_norm),
    ("Constructed AE", _final_constructed),
]

fig_export_un = make_subplots(rows=1, cols=_n_export, subplot_titles=_export_titles)

for i, prop in enumerate(_export_props):
    for j, (name, data) in enumerate(_series_un):
        color = MODEL_COLORS[name]
        fig_export_un.add_trace(
            go.Scatter(
                x=_x,
                y=data[prop],
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=_ms, opacity=_opacity, color=color),
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )
        fig_export_un.add_trace(
            go.Scatter(
                x=_x,
                y=_sm(data[prop]),
                legendgroup=name,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=_curve_opacity,
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

for ann in fig_export_un.layout.annotations:
    ann.font = dict(size=22)
for i in range(_n_export):
    fig_export_un.update_xaxes(title_text=None, row=1, col=i + 1)
fig_export_un.add_annotation(
    text="Feature Rank",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.23,
    showarrow=False,
    font=dict(size=22),
)
fig_export_un.update_layout(
    height=470,
    width=max(600, 350 * _n_export),
    margin=dict(b=100),
    showlegend=True,
    legend=dict(
        x=0.89,
        y=0.8,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#D1D5DB",
        borderwidth=1,
        itemsizing="constant",
        font=dict(size=7),
    ),
)
fig_export_un.update_yaxes(range=[220, 450], row=1, col=4)
fig_export_un.update_yaxes(range=[0.1, 0.52], row=1, col=5)

style_fig(fig_export_un)
fig_export_un.show()

# %%
# --- Save unit norms as vector ---
fig_export_un.write_image(
    os.path.join(_fig_dir, "geom_unit_norms.pdf"), engine="kaleido"
)
fig_export_un.write_image(
    os.path.join(_fig_dir, "geom_unit_norms.svg"), engine="kaleido"
)
print(f"Saved to {_fig_dir}/geom_unit_norms.{{pdf,svg}}")

# %%
