# %%
"""Compare TiedLinearRelu vs SynthAE on SparseUniform with zipfian firing probabilities.

Same experiment structure as synth_v_trained.py but using a simpler SparseUniform
distribution instead of the full SyntheticDataModel. Uses a soft power-law firing
decay (p_max=0.2, p_min=0.01). SAEs trained via SAELens StandardTrainingSAE.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

# --- Publication-ready figure styling ---
MODEL_COLORS = {"Trained AE": "#2563EB", "Constructed AE": "#16A34A"}
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
)


def style_fig(fig, nticksx=10, nticksy=8):
    """Apply publication-ready styling."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#1F2937"),
        title_font=dict(size=15),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    fig.update_xaxes(**_AXIS, nticks=nticksx)
    fig.update_yaxes(**_AXIS, nticks=nticksy)
    return fig


# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 1000
D_HIDDEN = 64
N_EPOCHS = 30_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14
EVAL_FREQ = 250

# %%
# --- Zipfian firing probabilities (soft decay) ---
high = 0.2
low = 0.5 / N_FEATURES  # softer decay: 20x range instead of 400x
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")

firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
firing_probs = torch.tensor(firing_probs, dtype=torch.float32)

dist = SparseUniform(N_FEATURES, p_active=firing_probs, device=DEVICE)

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


def geometry_hook(data):
    """Capture feature dimensionalities, norms, total interference, and bias."""
    tm = data["tm"]
    return {
        "fd": tm.feature_dimensionalities.detach().cpu().numpy(),
        "fn": tm.feature_norms.detach().cpu().numpy(),
        "ti": tm.total_feature_interferences.detach().cpu().numpy(),
        "bias": tm.ae.b.detach().cpu().numpy(),
    }


# %%
# --- Helper: evaluate a (non-trained) model once ---
def evaluate_model(tm):
    """Return (eval_loss, per_feature_mse) for a model without training."""
    with torch.no_grad():
        x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
        x_hat = tm.ae(x)[0]
        eval_loss = tm.ae.loss(x, x_hat, tm.importances).item()

        eye = torch.eye(N_FEATURES, device=tm.device)
        e_hat = tm.ae(eye)[0]
        pf_mse = (eye - e_hat).pow(2).sum(dim=-1).cpu().numpy()
    return eval_loss, pf_mse


# %%
# --- Train TiedLinearRelu ---
print("Training TiedLinearRelu...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
ae_tied = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1)
tm_tied = ToyModel(distribution=dist, ae=ae_tied, device=DEVICE)

_, hook_results_tied = tm_tied.fit(
    30000,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook, geometry_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_tied = hook_results_tied[0]
per_feature_tied = hook_results_tied[1]
geometry_tied = hook_results_tied[2]
print(f"  Final eval loss: {eval_losses_tied[-1]:.6f}")

# %%
# --- SynthAE (train bias only) ---
print("Training SynthAE (orthogonalized, bias only)...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth_ortho = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=1000,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_synth_ortho = ToyModel(distribution=dist, ae=ae_synth_ortho, device=DEVICE)
print("Initialized")
N_EPOCHS_SYNTH = 10_000
_, hook_results_synth = tm_synth_ortho.fit(
    N_EPOCHS_SYNTH,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_synth = hook_results_synth[0]
per_feature_synth = hook_results_synth[1]
loss_synth_ortho = eval_losses_synth[-1]
pf_synth_ortho = per_feature_synth[-1]
print(f"  Final eval loss: {loss_synth_ortho:.6f}")

# %% --- SAE training on both models (SAELens Standard SAE) ---
N_DICT = 1100
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.1
SAE_TRAINING_SAMPLES = 200_000 * SAE_BATCH  # ~200k steps

sae_results = {}

for name, tm in [("Trained AE", tm_tied), ("Constructed AE", tm_synth_ortho)]:
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
# --- Plot: Eval loss curve (TiedLinearRelu) with SynthAE baselines ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=eval_epochs,
        y=eval_losses_tied,
        name="Trained AE",
        mode="lines",
        line=dict(width=2, color=MODEL_COLORS["Trained AE"]),
    )
)
fig.add_hline(
    y=loss_synth_ortho,
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
# --- Plot: Per-feature reconstruction MSE ---
final_tied = np.array(per_feature_tied[-1])

# Sort features by firing probability (most frequent first)
fp_np = firing_probs.cpu().numpy()
sort_idx = np.argsort(-fp_np)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=final_tied[sort_idx],
        name="Trained AE",
        mode="markers",
        marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS["Trained AE"]),
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=pf_synth_ortho[sort_idx],
        name="Constructed AE",
        mode="markers",
        marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS["Constructed AE"]),
    )
)
fig.update_layout(
    title="Per-Feature Reconstruction MSE (sorted by firing probability)",
    xaxis_title="Feature Rank ",
    yaxis_title="MSE",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: Features recovered (MSE < threshold) vs epoch, with SynthAE baselines ---
THRESHOLDS = [0.2, 0.5, 1.0]
COLORS = _THRESH_COLORS

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, COLORS):
    n_recovered_tied = [int((np.array(s) < thresh).sum()) for s in per_feature_tied]
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered_tied,
            name=f"Trained AE (τ={thresh})",
            mode="lines",
            line=dict(color=color, width=2),
        )
    )
    n_recovered_ortho = int((pf_synth_ortho < thresh).sum())
    fig.add_hline(
        y=n_recovered_ortho,
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
# --- Plot: Per-feature MSE over training (heatmap, TiedLinearRelu only) ---
arr = np.array(per_feature_tied).T[:, 1:]  # (N_FEATURES, n_eval_points), skip step 0
arr_sorted = arr[sort_idx]

fig = go.Figure(
    go.Heatmap(
        z=arr_sorted,
        x=eval_epochs[1:],
        y=np.arange(N_FEATURES),
        colorscale="tempo",
        colorbar=dict(title="MSE"),
    )
)
fig.update_layout(
    title="Per-Feature MSE Over Training — Trained AE",
    xaxis_title="Epoch",
    yaxis_title="Feature rank (firing frequency)",
    height=500,
)
style_fig(fig)
fig.show()

# %%
# --- W^T W comparison ---
models = [("Trained AE", tm_tied), ("Constructed AE", tm_synth_ortho)]
models_dict = dict(models)

fig = make_subplots(rows=1, cols=2, subplot_titles=["Trained AE", "Constructed AE"])

for i, (name, tm) in enumerate(
    [
        ("Trained AE", tm_tied),
        ("Constructed AE", tm_synth_ortho),
    ]
):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(z=WtW, colorscale="RdBu_r", zmid=0, showscale=(i == 1)),
        row=1,
        col=i + 1,
    )

fig.update_layout(title="W^T W Comparison", height=400, width=900)
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
_geom_props = ["fd", "fn", "ti", "bias"]
_geom_titles = [
    "Feature Dimensionalities",
    "Feature Norms",
    "Total Interference",
    "Learned Bias",
]

# Build per-snapshot arrays for Trained AE, sorted by firing prob
geom_arrays = {}
for prop in _geom_props:
    arr = np.array([g[prop] for g in geometry_tied]).T  # (N_FEATURES, n_snapshots)
    geom_arrays[prop] = arr[sort_idx]

n_snapshots = geom_arrays["fd"].shape[1]
geom_epochs = (np.arange(n_snapshots) * EVAL_FREQ).astype(int)

# Static Constructed AE reference (final state)
synth_props = {
    "fd": tm_synth_ortho.feature_dimensionalities.detach().cpu().numpy()[sort_idx],
    "fn": tm_synth_ortho.feature_norms.detach().cpu().numpy()[sort_idx],
    "ti": tm_synth_ortho.total_feature_interferences.detach().cpu().numpy()[sort_idx],
    "bias": tm_synth_ortho.ae.b.detach().cpu().numpy()[sort_idx],
}

# Compute y-axis ranges across all snapshots (both models) for stable axes
y_ranges = {}
for prop in _geom_props:
    all_vals = np.concatenate([geom_arrays[prop].ravel(), synth_props[prop]])
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = (hi - lo) * 0.05
    y_ranges[prop] = [lo - pad, hi + pad]

x_feat = np.arange(N_FEATURES)
fig = make_subplots(rows=1, cols=4, subplot_titles=_geom_titles)

# Initial traces (epoch 0): 4 Trained AE + 4 Constructed AE = 8 traces
for i, prop in enumerate(_geom_props):
    fig.add_trace(
        go.Scatter(
            x=x_feat,
            y=geom_arrays[prop][:, 0],
            name="Trained AE",
            legendgroup="Trained AE",
            mode="markers",
            marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS["Trained AE"]),
            showlegend=(i == 0),
        ),
        row=1,
        col=i + 1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_feat,
            y=synth_props[prop],
            name="Constructed AE",
            legendgroup="Constructed AE",
            mode="markers",
            marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS["Constructed AE"]),
            showlegend=(i == 0),
        ),
        row=1,
        col=i + 1,
    )

# Slider steps — update the 4 Trained AE traces (indices 0, 2, 4, 6)
steps = []
for s in range(n_snapshots):
    step = dict(
        method="update",
        label=str(geom_epochs[s]),
        args=[
            {
                "y": [
                    geom_arrays["fd"][:, s],
                    None,
                    geom_arrays["fn"][:, s],
                    None,
                    geom_arrays["ti"][:, s],
                    None,
                    geom_arrays["bias"][:, s],
                    None,
                ]
            },
        ],
    )
    steps.append(step)

fig.update_layout(
    title="Geometric Properties Over Training (sorted by firing probability)",
    height=500,
    width=1400,
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Epoch: "),
            pad=dict(t=50),
            steps=steps,
        )
    ],
)
for i, prop in enumerate(_geom_props):
    fig.update_yaxes(range=y_ranges[prop], row=1, col=i + 1)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature Rank ", row=1, col=col)
style_fig(fig)
fig.show()

# %%
# --- Plot: Feature norms² + bias (TiedLinearRelu) ---
fn2 = tm_tied.feature_norms.detach().cpu().numpy() ** 2
b_tied = tm_tied.ae.b.detach().cpu().numpy()  # ty:ignore
combined = (fn2 + b_tied)[sort_idx]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=combined,
        mode="markers",
        marker=dict(size=3.25, opacity=0.6, color=MODEL_COLORS["Trained AE"]),
        name="‖w‖² + b",
    )
)
fig.update_layout(
    title="Trained AE: ‖w‖² + b (sorted by firing probability)",
    xaxis_title="Feature Rank ",
    yaxis_title="‖w‖² + b",
)
style_fig(fig)
fig.show()

# %%
# --- Summary statistics ---
print("\n=== Summary ===")
for name, eval_loss, pf in [
    ("Trained AE", eval_losses_tied[-1], per_feature_tied[-1]),
    ("Constructed AE", loss_synth_ortho, pf_synth_ortho),
]:
    final_mse = np.array(pf)
    recovered = "  ".join(f"τ={t}: {int((final_mse < t).sum())}" for t in THRESHOLDS)
    print(
        f"{name:25s}  eval_loss={eval_loss:.6f}  "
        f"recovered=[{recovered}]  "
        f"mean_feature_MSE={final_mse.mean():.4f}"
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

# %% --- SAE activations on one-hot features (matched) ---


for name in names:
    tm_ref = models_dict[name]
    sae = tm_ref.saes[name].sae
    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)

        # SAE activations on one-hot features
        sae_acts = (
            sae.encode(tm_ref.ae.encode(eye)).cpu().numpy()
        )  # (N_FEATURES, N_DICT)

        # Cosine similarity matching
        D = tm_ref.ae.encode(eye)  # (N_FEATURES, D_HIDDEN)
        D_normed = D / D.norm(dim=1, keepdim=True)
        W_dec = sae.W_dec.data  # (N_DICT, D_HIDDEN)
        W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)
        cosine_sim = (D_normed @ W_dec_normed.T).cpu().numpy()  # (N_FEATURES, N_DICT)

    feat_idx, dict_idx = linear_sum_assignment(-cosine_sim)

    # Reorder both rows and columns: matched pairs form the top-left diagonal,
    # then append unmatched features (rows) and unmatched dict elements (cols)
    matched_feats = set(feat_idx)
    matched_dicts = set(dict_idx)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts

    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
    row_labels = [f"f{f}" for f in row_order]
    col_labels = [f"d{d}" for d in col_order]

    # Compute diagonality: fraction of total activation on the matched diagonal
    n_matched = len(feat_idx)
    diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
    total_sum = sae_acts_matched.sum()
    diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
    sae_results[name]["diagonality"] = diagonality

    mean_cosine = cosine_sim[feat_idx, dict_idx].mean()
    print(
        f"{name}: diagonality = {diagonality:.4f} (diag_sum={diag_sum:.2f}, total={total_sum:.2f})  "
        f"mean_cosine = {mean_cosine:.4f}"
    )

    fig_imshow = px.imshow(
        sae_acts_matched,
        labels=dict(
            x="SAE dict element (cosine matched)", y="Feature (cosine matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"SAE one-hot activations (cosine matched, diag={diagonality:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    )
    style_fig(fig_imshow)
    fig_imshow.show()

    cosine_sim_matched = cosine_sim[np.ix_(row_order, col_order)]
    fig_imshow = px.imshow(
        cosine_sim_matched,
        labels=dict(
            x="SAE dict element (cosine matched)", y="Feature (cosine matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"Cosine similarity (matched, mean={mean_cosine:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    style_fig(fig_imshow)
    fig_imshow.show()

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
_hdr1 = f"{'Model':25s}  {'MSE↓':>10s}  {'L0↓':>6s}  {'Dead↓':>6s}  {'Alive↑':>6s}  {'ExplVar↑':>8s}  {'Diag↑':>6s}"
_hdr2 = f"{'':25s}  {'MCC_abs↑':>8s}  {'MCC_cos↑':>8s}  {'Prec↑':>6s}  {'Rec↑':>6s}  {'F1↑':>6s}  {'FPR↓':>6s}"

for match_label, sfx in [("abs cos-sim", "_abs"), ("cos-sim", "_cos")]:
    print(f"\n|| SAE Summary (matched on {match_label}) || L1 = {SAE_L1}")
    print(_hdr1)
    print(_hdr2)
    for name, res in sae_results.items():
        print(
            f"{name:25s}  {res['recon_mse']:10.6f}  {res['l0']:6.1f}  "
            f"{res['n_dead']:6d}  {res['n_alive']:6d}  {res['explained_var']:8.4f}  "
            f"{res.get('diagonality', 0):6.4f}"
        )
        print(
            f"{'':25s}  {res.get(f'mcc{sfx}_abs', 0):8.4f}  {res.get(f'mcc{sfx}_cos', 0):8.4f}  "
            f"{res[f'precision{sfx}']:6.4f}  {res[f'recall{sfx}']:6.4f}  "
            f"{res[f'f1{sfx}']:6.4f}  {res[f'fpr{sfx}']:6.4f}"
        )


# %% --- Per-feature detection metrics (sorted by firing probability) ---
fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=["Precision", "Recall (TPR)", "F1 Score", "FPR"],
)

for name in names:
    res = sae_results[name]
    # Re-sort by firing probability for display.
    feat_order = np.argsort(-fp_np[res["mcc_feat_idx_cos"]])
    color = MODEL_COLORS[name]
    x = np.arange(len(feat_order))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["precision_per_cos"][feat_order],
            name=name,
            legendgroup=name,
            mode="markers",
            marker=dict(size=3.25, opacity=0.6, color=color),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["recall_per_cos"][feat_order],
            name=name,
            legendgroup=name,
            mode="markers",
            showlegend=False,
            marker=dict(size=3.25, opacity=0.6, color=color),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["f1_per_cos"][feat_order],
            name=name,
            legendgroup=name,
            mode="markers",
            showlegend=False,
            marker=dict(size=3.25, opacity=0.6, color=color),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["fpr_per_cos"][feat_order],
            name=name,
            legendgroup=name,
            mode="markers",
            showlegend=False,
            marker=dict(size=3.25, opacity=0.6, color=color),
        ),
        row=1,
        col=4,
    )

fig.update_layout(
    title="Per-Feature Detection Metrics (sorted by firing prob.)",
    height=400,
    width=1400,
)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature Rank ", row=1, col=col)
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

# %% --- Encoder-matched SAE activations heatmap ---
for name in names:
    res = sae_results[name]
    tm = models_dict[name]
    sae = tm.saes[name].sae

    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()

    enc_fi, enc_di = res["enc_feat_idx"], res["enc_dict_idx"]
    matched_feats = set(enc_fi)
    matched_dicts = set(enc_di)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

    row_order = list(enc_fi) + unmatched_feats
    col_order = list(enc_di) + unmatched_dicts

    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
    row_labels = [f"f{f}" for f in row_order]
    col_labels = [f"d{d}" for d in col_order]

    n_matched = len(enc_fi)
    diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
    total_sum = sae_acts_matched.sum()
    diag_enc = diag_sum / total_sum if total_sum > 0 else 0.0
    res["diagonality_enc"] = diag_enc

    fig_imshow = px.imshow(
        sae_acts_matched,
        labels=dict(
            x="SAE dict element (encoder matched)", y="Feature (encoder matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"SAE one-hot activations (encoder matched, diag={diag_enc:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    )
    style_fig(fig_imshow)
    fig_imshow.show()

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

# %% --- Per-feature detection metrics: encoder vs decoder matching ---
fig = make_subplots(
    rows=2,
    cols=4,
    subplot_titles=[
        "Precision (dec)",
        "Recall (dec)",
        "F1 (dec)",
        "FPR (dec)",
        "Precision (enc)",
        "Recall (enc)",
        "F1 (enc)",
        "FPR (enc)",
    ],
)

for name in names:
    res = sae_results[name]
    color = MODEL_COLORS[name]

    # Row 1: decoder matching (cos)
    feat_order_dec = np.argsort(-fp_np[res["mcc_feat_idx_cos"]])
    x = np.arange(len(feat_order_dec))
    for col, metric in enumerate(
        ["precision_per_cos", "recall_per_cos", "f1_per_cos", "fpr_per_cos"], 1
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=res[metric][feat_order_dec],
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=3.25, opacity=0.6, color=color),
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

    # Row 2: encoder matching
    feat_order_enc = np.argsort(-fp_np[res["enc_feat_idx"]])
    for col, metric in enumerate(
        ["precision_per_enc", "recall_per_enc", "f1_per_enc", "fpr_per_enc"], 1
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=res[metric][feat_order_enc],
                name=name,
                legendgroup=name,
                mode="markers",
                marker=dict(size=3.25, opacity=0.6, color=color),
                showlegend=False,
            ),
            row=2,
            col=col,
        )

fig.update_layout(
    title="Per-Feature Detection: Decoder (top) vs Encoder (bottom) Matching",
    height=600,
    width=1400,
)
for row in range(1, 3):
    for col in range(1, 5):
        fig.update_xaxes(title_text="Feature Rank ", row=row, col=col)
style_fig(fig)
fig.show()

# %%
