# %%
"""TiedLinearRelu on SparseUniform — basic experiment boilerplate."""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import SAEEntry, ToyModel

# --- Paper-quality plot defaults ---
FONT = dict(family="Times New Roman, serif", size=24, color="#333333")
AXIS_STYLE = dict(
    showgrid=False,
    gridcolor="rgba(0,0,0,0.08)",
    gridwidth=1,
    zeroline=False,
    linecolor="#666666",
    linewidth=1,
    ticks="outside",
    ticklen=4,
    tickwidth=1,
    tickcolor="#666666",
    tickfont_size=18,
    minor=dict(ticks="outside", ticklen=2),
)
LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=FONT,
    title_font_size=24,
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        font_size=24,
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 1296
D_HIDDEN = 100
N_EPOCHS = 40_000
BATCH_SIZE = 512

# %%
# --- Distribution ---
high = 0.3
low = 1.28 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]

dist = SparseUniform(N_FEATURES, p_active=firing_probs, device=DEVICE)

# Average L0
samples = dist.sample(100_000)
mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
std_l0 = (samples > 0).float().sum(dim=-1).std().item()
print(f"Average L0: {mean_l0:.2f} +/- {std_l0:.2f}")

# %%
# --- Train ---
gen = torch.Generator(DEVICE).manual_seed(SEED)
ae = TiedLinearRelu(
    N_FEATURES,
    D_HIDDEN,
    device=DEVICE,
    generator=gen,
)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

# %%
# --- Autoencoder F1 (does the AE round-trip preserve feature activity?) ---
with torch.no_grad():
    ae_test_x = dist.sample(100_000).to(DEVICE)
    ae_test_xhat = tm.ae.decode(tm.ae.encode(ae_test_x))

    gt_active_ae = ae_test_x > 0
    pred_active_ae = ae_test_xhat > 0

    tp_ae = (gt_active_ae & pred_active_ae).float().sum(dim=0)
    fp_ae = (~gt_active_ae & pred_active_ae).float().sum(dim=0)
    fn_ae = (gt_active_ae & ~pred_active_ae).float().sum(dim=0)

    prec_ae = tp_ae / (tp_ae + fp_ae + 1e-8)
    rec_ae = tp_ae / (tp_ae + fn_ae + 1e-8)
    f1_ae = 2 * prec_ae * rec_ae / (prec_ae + rec_ae + 1e-8)

    tp_tot_ae = tp_ae.sum()
    fp_tot_ae = fp_ae.sum()
    fn_tot_ae = fn_ae.sum()
    prec_micro_ae = (tp_tot_ae / (tp_tot_ae + fp_tot_ae + 1e-8)).item()
    rec_micro_ae = (tp_tot_ae / (tp_tot_ae + fn_tot_ae + 1e-8)).item()
    f1_micro_ae = (
        2 * prec_micro_ae * rec_micro_ae / (prec_micro_ae + rec_micro_ae + 1e-8)
    )

print(
    f"[AE round-trip] "
    f"prec={prec_ae.mean():.4f}  rec={rec_ae.mean():.4f}  "
    f"F1(macro)={f1_ae.mean().item():.4f}  F1(micro)={f1_micro_ae:.4f}"
)

# %%
# --- Plot: Per-feature AE F1 score ---
f1_ae_np = f1_ae.cpu().numpy()
fig_ae_f1 = go.Figure()
fig_ae_f1.add_trace(
    go.Scatter(x=np.arange(N_FEATURES), y=f1_ae_np, mode="lines", name="F1")
)
fig_ae_f1.update_xaxes(title_text="Feature index", **AXIS_STYLE)
fig_ae_f1.update_yaxes(title_text="F1 score", range=[0, 1.05], **AXIS_STYLE)
fig_ae_f1.update_layout(title="Per-feature AE round-trip F1", **LAYOUT_DEFAULTS)
fig_ae_f1.show()

# %%
# --- Plot: Feature Norms and Feature Dimensionalities ---
fn = tm.feature_norms.detach().cpu().numpy()
fd = tm.feature_dimensionalities.detach().cpu().numpy()

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Feature Norms", "Feature Dimensionalities"],
)

fig.add_trace(
    go.Scatter(x=np.arange(N_FEATURES), y=fn, mode="lines", name="Norms"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=fd,
        mode="lines",
        name="Dimensionalities",
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Feature index", row=1, col=1)
fig.update_xaxes(title_text="Feature index", row=1, col=2)
fig.update_yaxes(title_text="‖w‖", row=1, col=1)
fig.update_yaxes(title_text="Dimensionality", row=1, col=2)
fig.update_layout(
    title=f"TiedLinearRelu — SparseUniform (N={N_FEATURES}, D={D_HIDDEN})",
    height=400,
    width=900,
)
fig.show()

# %%
# --- Plot: Empirical correlation vs interference ---
empirical_corr = torch.corrcoef(samples.T.cpu())
interferences = tm.interferences.detach().cpu()
rows_i, rows_j = torch.triu_indices(N_FEATURES, N_FEATURES, offset=1)
pair_corr = empirical_corr[rows_i, rows_j].numpy()
pair_interf = interferences[rows_i, rows_j].numpy()

_x = pair_corr[:100_000]
_y = pair_interf[:100_000]
_slope, _intercept = np.polyfit(_x, _y, 1)
_resid = _y - (_slope * _x + _intercept)
_ss_res = np.sum(_resid**2)
_ss_tot = np.sum((_y - _y.mean()) ** 2)
_r2 = 1 - _ss_res / _ss_tot
_n = len(_x)
_se_slope = np.sqrt(_ss_res / (_n - 2) / np.sum((_x - _x.mean()) ** 2))
_corr = np.corrcoef(_x, _y)[0, 1]
_z = np.arctanh(_corr)
_z_se = 1.0 / np.sqrt(_n - 3)
_corr_lo = np.tanh(_z - 1.96 * _z_se)
_corr_hi = np.tanh(_z + 1.96 * _z_se)

fig = px.scatter(
    x=_x,
    y=_y,
    labels={"x": "Empirical correlation", "y": "Interference"},
    opacity=0.8,
    trendline="ols",
    trendline_color_override="black",
)
fig.update_traces(marker_size=3, selector=dict(mode="markers"))
fig.update_traces(opacity=0.8, selector=dict(mode="lines"))
fig.add_hline(y=0, line_color="gray", line_width=1, layer="below")
fig.add_annotation(
    text=f"slope = {_slope:.3f} ± {1.96 * _se_slope:.3f}<br>R² = {_r2:.3f}<br>r = {_corr:.3f} [{_corr_lo:.3f}, {_corr_hi:.3f}]",
    xref="paper",
    yref="paper",
    x=0.95,
    y=0.05,
    showarrow=False,
    font=dict(size=22),
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="#cccccc",
    borderwidth=1,
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %%
# --- SAE Training ---
from scipy.optimize import linear_sum_assignment
from occhio.sae.sae import SAESimple

N_DICT = N_FEATURES // 2
SAE_STEPS = 15_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.3

sae_gen = torch.Generator().manual_seed(4)

sae = SAESimple(
    n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=SAE_L1, aux_k=True, device=DEVICE
).to(DEVICE)


def data_fn(n: int) -> torch.Tensor:
    x = dist.sample(n).to(DEVICE)
    return tm.ae.encode(x)


sae_losses = sae.train_sae(
    data_fn=data_fn, n_steps=SAE_STEPS, batch_size=SAE_BATCH, lr=SAE_LR
)

# %%
# --- SAE Metrics ---
with torch.no_grad():
    test_x = dist.sample(50_000).to(DEVICE)
    test_hidden = tm.ae.encode(test_x)
    test_z = sae.encode(test_hidden)
    test_recon = sae.decode(test_z)

    # L0
    l0 = (test_z > 0).float().sum(dim=-1).mean().item()

    # R² (explained variance)
    total_var = test_hidden.var(dim=0).sum().item()
    residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
    r2 = 1 - residual_var / total_var

    # MCC matching (cosine similarity, no abs)
    D = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
    W_dec_t = sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim = (D_norm.T @ W_norm).cpu().numpy()  # (N_FEATURES, N_DICT)

    feat_idx, dict_idx = linear_sum_assignment(-cos_sim)
    mcc = float(cos_sim[feat_idx, dict_idx].mean())

    # F1 (detection)
    gt_active = test_x[:, feat_idx] > 0
    pred_active = test_z[:, dict_idx] > 0
    tp = (gt_active & pred_active).float().sum(dim=0)
    fp = (~gt_active & pred_active).float().sum(dim=0)
    fn_ = (gt_active & ~pred_active).float().sum(dim=0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn_ + 1e-8)
    f1 = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()

    # Purity / diagonality
    eye = torch.eye(N_FEATURES, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (N_FEATURES, N_DICT)

    # Reorder rows/cols by MCC matching
    matched_feats = set(feat_idx)
    matched_dicts = set(dict_idx)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]
    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts
    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]

    n_matched = len(feat_idx)
    diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
    total_sum = sae_acts_matched.sum()
    purity = diag_sum / total_sum if total_sum > 0 else 0.0

print(f"L1={SAE_L1}, batch={SAE_BATCH}, LR={SAE_LR}")
print(f"prec={prec.mean():.4f}, reca={rec.mean():.4f}")
print(f"L0={l0:.1f}  MCC={mcc:.4f}  F1={f1:.4f}  R²={r2:.4f}  Purity={purity:.4f}")

# %%
# --- Plot: SAE one-hot activations (MCC matched) ---
row_labels = [f"f{f}" for f in row_order]
col_labels = [f"d{d}" for d in col_order]

px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (MCC matched)", y="Feature (MCC matched)"),
    x=col_labels,
    y=row_labels,
    title=f"SAE one-hot activations (MCC matched, purity={purity:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
).show()

# %%
# --- Matryoshka SAE Training (via SAELens) ---
from sae_lens import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)

# Nested cumulative widths; final width must equal d_sae.
MATRYOSHKA_WIDTHS = [N_DICT // 8, N_DICT // 4, N_DICT // 2, N_DICT]
MATRYOSHKA_K = max(1, int(round(mean_l0)))
MATRYOSHKA_TRAIN_SAMPLES = SAE_STEPS * SAE_BATCH

# Build on CPU first: SAELens registers `topk_threshold` as float64, which MPS
# rejects. We downcast that buffer to float32 before moving to the target device.
matryoshka_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
    d_in=D_HIDDEN,
    d_sae=N_DICT,
    matryoshka_widths=MATRYOSHKA_WIDTHS,
    k=MATRYOSHKA_K,
    device="cpu",
)
matryoshka_sae = MatryoshkaBatchTopKTrainingSAE(matryoshka_cfg)
matryoshka_sae.topk_threshold = matryoshka_sae.topk_threshold.to(torch.float32)
matryoshka_sae.to(DEVICE)

tm.train_saes(
    [SAEEntry(sae=matryoshka_sae, type="Matryoshka", label="matryoshka")],
    training_samples=MATRYOSHKA_TRAIN_SAMPLES,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
    verbose=True,
)
tm.evaluate_saes(["matryoshka"], num_samples=50_000, verbose=True)

print(f"Matryoshka widths={MATRYOSHKA_WIDTHS}, k={MATRYOSHKA_K}")
print(f"F1 = {tm.saes_f1_score['matryoshka']:.4f}")
print(f"MCC = {tm.saes_mcc['matryoshka']:.4f}")
print(f"L0 = {tm.saes_l0['matryoshka']:.2f}")
print(f"R² = {tm.saes_explained_variance['matryoshka']:.4f}")

# %%
# --- Matryoshka SAE Metrics (computed by hand, no abs in MCC) ---
with torch.no_grad():
    test_x_m = dist.sample(50_000).to(DEVICE)
    test_hidden_m = tm.ae.encode(test_x_m)
    test_z_m = matryoshka_sae.encode(test_hidden_m)
    test_recon_m = matryoshka_sae.decode(test_z_m)

    # L0
    l0_m = (test_z_m > 0).float().sum(dim=-1).mean().item()

    # R² (explained variance)
    total_var_m = test_hidden_m.var(dim=0).sum().item()
    residual_var_m = (test_hidden_m - test_recon_m).var(dim=0).sum().item()
    r2_m = 1 - residual_var_m / total_var_m

    # MCC matching (cosine similarity, NO abs)
    D_m = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
    W_dec_t_m = matryoshka_sae.W_dec.detach().T  # (D_HIDDEN, N_DICT)
    D_norm_m = D_m / D_m.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm_m = W_dec_t_m / W_dec_t_m.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim_m = (D_norm_m.T @ W_norm_m).cpu().numpy()  # (N_FEATURES, N_DICT)

    feat_idx_m, dict_idx_m = linear_sum_assignment(-cos_sim_m)
    mcc_m = float(cos_sim_m[feat_idx_m, dict_idx_m].mean())

    # F1 (detection)
    gt_active_m = test_x_m[:, feat_idx_m] > 0
    pred_active_m = test_z_m[:, dict_idx_m] > 0
    tp_m = (gt_active_m & pred_active_m).float().sum(dim=0)
    fp_m = (~gt_active_m & pred_active_m).float().sum(dim=0)
    fn_m = (gt_active_m & ~pred_active_m).float().sum(dim=0)
    prec_m = tp_m / (tp_m + fp_m + 1e-8)
    rec_m = tp_m / (tp_m + fn_m + 1e-8)
    f1_m = (2 * prec_m * rec_m / (prec_m + rec_m + 1e-8)).mean().item()

print(f"[Matryoshka, by hand] prec={prec_m.mean():.4f}, reca={rec_m.mean():.4f}")
print(f"L0={l0_m:.1f}  MCC={mcc_m:.4f}  F1={f1_m:.4f}  R²={r2_m:.4f}")

# %%
