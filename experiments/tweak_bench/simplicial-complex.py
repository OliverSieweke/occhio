# %%
"""TiedLinearRelu on CorrelatedPairs — basic experiment boilerplate."""

from sae_lens import SAE

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SimplicialComplexDistribution
from occhio.toy_model import ToyModel

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
N_EPOCHS = 20_000
BATCH_SIZE = 512

# %%
# --- Distribution ---
FACE_DIM = 4
N_FACES = 4 * (N_FEATURES // (FACE_DIM + 1))

all_verts = list(range(N_FEATURES))
random.shuffle(all_verts)
face_size = FACE_DIM + 1
covering_faces = set()
for i in range(0, N_FEATURES, face_size):
    chunk = all_verts[i : i + face_size]
    if len(chunk) < face_size:
        remaining = [v for v in all_verts if v not in chunk]
        chunk += random.sample(remaining, face_size - len(chunk))
    covering_faces.add(tuple(sorted(chunk)))
while len(covering_faces) < N_FACES:
    covering_faces.add(tuple(sorted(random.sample(range(N_FEATURES), face_size))))
faces = list(covering_faces)[:N_FACES]


dist = SimplicialComplexDistribution(
    n_vertices=N_FEATURES, faces=faces, sampling_mode="single", p_active=1 / N_FACES
)

# Average L0
samples = dist.sample(100_000)
mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
std_l0 = (samples > 0).float().sum(dim=-1).std().item()
print(f"Average L0: {mean_l0:.2f} +/- {std_l0:.2f}")

# %%
# --- Plot: Sorted feature firing probabilities ---
firing_probs = (samples > 0).float().mean(dim=0).cpu().numpy()
firing_probs_sorted = np.sort(firing_probs)[::-1]

fig_firing = go.Figure()
fig_firing.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=firing_probs_sorted,
        mode="lines",
        name="Firing prob",
    )
)
fig_firing.update_xaxes(title_text="Feature rank", **AXIS_STYLE)
fig_firing.update_yaxes(title_text="P(active)", **AXIS_STYLE)
fig_firing.update_layout(
    title="Feature firing probabilities (sorted)", **LAYOUT_DEFAULTS
)
fig_firing.show()

# %%
# --- Train ---
gen = torch.Generator(DEVICE).manual_seed(SEED)
ae = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

# %%
# --- Plot losses ---
px.line(losses)

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
    title=f"TiedLinearRelu — CorrelatedPairs (N={N_FEATURES}, D={D_HIDDEN})",
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
SAE_L1 = 0.2

sae_gen = torch.Generator().manual_seed(4)

sae = SAESimple(n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=SAE_L1, device=DEVICE).to(
    DEVICE
)


def data_fn(n: int) -> torch.Tensor:
    x = dist.sample(n).to(DEVICE)
    return tm.ae.encode(x)


sae_losses = sae.train_sae(
    data_fn=data_fn, n_steps=SAE_STEPS, batch_size=SAE_BATCH, lr=SAE_LR
)
# %%
px.line(sae_losses)

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
