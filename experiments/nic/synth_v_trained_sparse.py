# %%
"""Compare TiedLinearRelu vs SynthAE on SparseUniform with zipfian firing probabilities.

Same experiment structure as synth_v_trained.py but using a simpler SparseUniform
distribution instead of the full SyntheticDataModel. The zipfian firing pattern
(p_max=0.4, p_min=0.5/N, alpha=0.5) is matched to the SyntheticDataModel config.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

# --- Paper-quality plot defaults ---
PALETTE = {"TiedLinearRelu": "#2166ac", "SynthAE (ortho)": "#4daf4a"}
FONT = dict(family="Times New Roman, serif", size=14, color="#333333")
AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    gridwidth=1,
    zeroline=False,
    linecolor="#666666",
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickwidth=1,
    tickcolor="#666666",
    minor=dict(ticks="outside", ticklen=2),
)
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=FONT,
    title_font_size=16,
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        font_size=12,
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
LINE_WIDTH = 2

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
# --- Zipfian firing probabilities ---
# Matches the SyntheticDataModel zipfian config: p_max=0.4, p_min=0.5/N, alpha=0.5
high = 0.46
low = 1.0 / N_FEATURES
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
    N_EPOCHS,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_tied = hook_results_tied[0]
per_feature_tied = hook_results_tied[1]
print(f"  Final eval loss: {eval_losses_tied[-1]:.6f}")

# %%
# --- SynthAE (train bias only) ---
print("Training SynthAE (orthogonalized, bias only)...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth_ortho = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
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

# %%
# --- Plot: Eval loss curve (TiedLinearRelu) with SynthAE baselines ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=eval_epochs,
        y=eval_losses_tied,
        name="TiedLinearRelu",
        line=dict(color=PALETTE["TiedLinearRelu"], width=LINE_WIDTH),
    )
)
fig.add_hline(
    y=loss_synth_ortho,
    line_dash="dash",
    line_color=PALETTE["SynthAE (ortho)"],
    line_width=LINE_WIDTH,
    annotation_text="SynthAE (ortho)",
    annotation_font_size=12,
    annotation_font_color=PALETTE["SynthAE (ortho)"],
)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title=f"Eval Loss — SparseUniform (N={N_FEATURES}, D={D_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
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
        name="TiedLinearRelu",
        mode="lines",
        line=dict(color=PALETTE["TiedLinearRelu"], width=LINE_WIDTH),
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=pf_synth_ortho[sort_idx],
        name="SynthAE (ortho)",
        mode="lines",
        line=dict(color=PALETTE["SynthAE (ortho)"], width=LINE_WIDTH),
    )
)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Per-Feature Reconstruction MSE",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %%
# --- Plot: Features recovered (MSE < threshold) vs epoch, with SynthAE baselines ---
THRESHOLDS = [0.2, 0.5, 1.0]
THRESHOLD_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, THRESHOLD_COLORS):
    n_recovered_tied = [int((np.array(s) < thresh).sum()) for s in per_feature_tied]
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered_tied,
            name=f"TiedLinearRelu (τ={thresh})",
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
        )
    )
    n_recovered_ortho = int((pf_synth_ortho < thresh).sum())
    fig.add_hline(
        y=n_recovered_ortho,
        line_dash="dash",
        line_color=color,
        line_width=1.5,
        annotation_text=f"SynthAE τ={thresh}",
        annotation_font_size=10,
        annotation_font_color=color,
    )
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Features Recovered (MSE < τ) Over Training",
    xaxis_title="Epoch",
    yaxis_title="Number of features recovered",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
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
        colorscale="Viridis",
        reversescale=True,
        colorbar=dict(title=dict(text="MSE", font=dict(size=13)), thickness=15),
    )
)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Per-Feature MSE Over Training — TiedLinearRelu",
    xaxis_title="Epoch",
    yaxis_title="Feature rank (most frequent → rarest)",
    height=500,
    width=750,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %%
# --- W^T W comparison ---
models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["TiedLinearRelu", "SynthAE (ortho)"],
    horizontal_spacing=0.08,
)

for i, (name, tm) in enumerate(
    [
        ("TiedLinearRelu", tm_tied),
        ("SynthAE (ortho)", tm_synth_ortho),
    ]
):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(
            z=WtW,
            colorscale="RdBu_r",
            zmid=0,
            showscale=(i == 1),
            colorbar=dict(thickness=15) if i == 1 else None,
        ),
        row=1,
        col=i + 1,
    )

fig.update_layout(
    **LAYOUT_DEFAULTS, title="W<sup>T</sup>W Comparison", height=450, width=950
)
fig.update_annotations(font_size=14)
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
            line=dict(color=PALETTE[name], width=LINE_WIDTH),
            marker=dict(size=5, color=PALETTE[name]),
        )
    )
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Spectrum of WW<sup>T</sup>",
    xaxis_title="Eigenvalue index",
    yaxis_title="Eigenvalue",
    height=450,
    width=700,
)
fig.update_layout(
    legend=dict(x=0.95, y=0.95, xanchor="right", yanchor="top"),
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %%
# --- Plot: Geometric properties comparison ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Feature Dimensionality",
        "Feature Norm",
        "Total Interference",
    ],
    horizontal_spacing=0.08,
)

for name, tm in models:
    fd = tm.feature_dimensionalities.detach().cpu().numpy()[sort_idx]
    fn = tm.feature_norms.detach().cpu().numpy()[sort_idx]
    ti = tm.total_feature_interferences.detach().cpu().numpy()[sort_idx]
    x = np.arange(N_FEATURES)
    color = PALETTE[name]

    fig.add_trace(
        go.Scatter(
            x=x, y=fd, name=name, mode="lines", line=dict(color=color, width=LINE_WIDTH)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fn,
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ti,
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=3,
    )

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Geometric Properties (sorted by firing probability)",
    height=400,
    width=1200,
)
fig.update_annotations(font_size=14)
for col in range(1, 4):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col, **AXIS_STYLE)
    fig.update_yaxes(row=1, col=col, **AXIS_STYLE)
fig.show()

# %%
# --- Plot: Bias comparison ---
fig = go.Figure()
for name, tm in models:
    b = tm.ae.b.detach().cpu().numpy()[sort_idx]  # ty:ignore
    fig.add_trace(
        go.Scatter(
            x=np.arange(N_FEATURES),
            y=b,
            name=name,
            mode="lines",
            line=dict(color=PALETTE[name], width=LINE_WIDTH),
        )
    )
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Learned Bias <i>b</i>",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="<i>b</i>",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
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
        mode="lines",
        name="‖w‖² + b",
        line=dict(color=PALETTE["TiedLinearRelu"], width=LINE_WIDTH),
    )
)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="TiedLinearRelu: ‖w‖² + b",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="‖w‖² + b",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %%
# --- Summary statistics ---
print("\n=== Summary ===")
for name, eval_loss, pf in [
    ("TiedLinearRelu", eval_losses_tied[-1], per_feature_tied[-1]),
    ("SynthAE (ortho)", loss_synth_ortho, pf_synth_ortho),
]:
    final_mse = np.array(pf)
    recovered = "  ".join(f"τ={t}: {int((final_mse < t).sum())}" for t in THRESHOLDS)
    print(
        f"{name:25s}  eval_loss={eval_loss:.6f}  "
        f"recovered=[{recovered}]  "
        f"mean_feature_MSE={final_mse.mean():.4f}"
    )

# %% --- SAE training on both models ---
N_DICT = N_FEATURES // 2
SAE_STEPS = 50_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.3

sae_results = {}

for name, tm in [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]:
    print(f"\nTraining SAE on {name}...")

    sae = SAESimple(
        n_latent=D_HIDDEN,
        n_dict=N_DICT,
        l1_coef=SAE_L1,
        device=DEVICE,
    ).to(DEVICE)

    def make_data_fn(tm_ref):
        def data_fn(n: int) -> torch.Tensor:
            x = tm_ref.distribution.sample(n).to(DEVICE)
            return tm_ref.ae.encode(x)

        return data_fn

    sae_losses = sae.train_sae(
        data_fn=make_data_fn(tm),
        n_steps=SAE_STEPS,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
    )

    # Compute metrics
    with torch.no_grad():
        test_x = dist.sample(10_000).to(DEVICE)
        test_hidden = tm.ae.encode(test_x)
        test_z = sae.encode(test_hidden)
        test_recon = sae.decode(test_z)

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
        h_eye_recon = sae.decode(sae.encode(h_eye))
        per_feat_sae_mse = (h_eye - h_eye_recon).pow(2).sum(dim=-1).cpu().numpy()

        # Explained variance ratio
        total_var = test_hidden.var(dim=0).sum().item()
        residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
        explained_var = 1 - residual_var / total_var

    sae_results[name] = {
        "sae": sae,
        "tm": tm,
        "losses": sae_losses,
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

# %% --- SAE loss curves ---
fig = go.Figure()
for name, res in sae_results.items():
    fig.add_trace(
        go.Scatter(
            y=res["losses"],
            mode="lines",
            name=name,
            line=dict(color=PALETTE[name], width=LINE_WIDTH),
        )
    )
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="SAE Training Loss",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()


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
            mode="lines",
            line=dict(color=PALETTE[name], width=LINE_WIDTH),
        )
    )
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="SAE Per-Feature Reconstruction Error",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE (hidden space)",
    height=450,
    width=700,
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %% --- SAE activations on one-hot features (matched) ---


for name in names:
    sae = sae_results[name]["sae"]
    tm_ref = tm_tied if name == "TiedLinearRelu" else tm_synth_ortho
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

    _fig = px.imshow(
        sae_acts_matched,
        labels=dict(
            x="SAE dict element (cosine matched)", y="Feature (cosine matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"SAE Activations (cosine matched, diag={diagonality:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    _fig.update_layout(**LAYOUT_DEFAULTS, height=550, width=700)
    _fig.show()

    cosine_sim_matched = cosine_sim[np.ix_(row_order, col_order)]
    _fig = px.imshow(
        cosine_sim_matched,
        labels=dict(
            x="SAE dict element (cosine matched)", y="Feature (cosine matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"Cosine Similarity (matched, mean={mean_cosine:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    _fig.update_layout(**LAYOUT_DEFAULTS, height=550, width=700)
    _fig.show()

# %% --- SAE evaluation: MCC, detection metrics ---
for name, res in sae_results.items():
    sae = res["sae"]
    tm = res["tm"]

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
    horizontal_spacing=0.06,
)

for name in names:
    res = sae_results[name]
    feat_order = np.argsort(-fp_np[res["mcc_feat_idx_cos"]])
    color = PALETTE[name]
    x = np.arange(len(feat_order))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["precision_per_cos"][feat_order],
            name=name,
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["recall_per_cos"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["f1_per_cos"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res["fpr_per_cos"][feat_order],
            name=name,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=LINE_WIDTH),
        ),
        row=1,
        col=4,
    )

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Per-Feature Detection Metrics",
    height=400,
    width=1400,
)
fig.update_annotations(font_size=14)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col, **AXIS_STYLE)
    fig.update_yaxes(row=1, col=col, **AXIS_STYLE)
fig.show()

# %% --- Encoder-based matching (using SAE activations on one-hot inputs) ---
for name in names:
    res = sae_results[name]
    sae = res["sae"]
    tm = res["tm"]

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
    sae = res["sae"]
    tm = res["tm"]

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

    _fig = px.imshow(
        sae_acts_matched,
        labels=dict(
            x="SAE dict element (encoder matched)", y="Feature (encoder matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"SAE Activations (encoder matched, diag={diag_enc:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    _fig.update_layout(**LAYOUT_DEFAULTS, height=550, width=700)
    _fig.show()

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
    horizontal_spacing=0.06,
    vertical_spacing=0.12,
)

for name in names:
    res = sae_results[name]
    color = PALETTE[name]

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
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH),
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
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
            ),
            row=2,
            col=col,
        )

fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Per-Feature Detection: Decoder (top) vs Encoder (bottom) Matching",
    height=650,
    width=1400,
)
fig.update_annotations(font_size=13)
for row in range(1, 3):
    for col in range(1, 5):
        fig.update_xaxes(title_text="Feature rank", row=row, col=col, **AXIS_STYLE)
        fig.update_yaxes(row=row, col=col, **AXIS_STYLE)
fig.show()

# %%
