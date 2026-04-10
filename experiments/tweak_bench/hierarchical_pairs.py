# %%
"""TiedLinearRelu on CorrelatedPairs — basic experiment boilerplate."""

from sae_lens import SAE

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.correlated import HierarchicalPairs
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
D_HIDDEN = 128
N_EPOCHS = 50_000
BATCH_SIZE = 512

# %%
# --- Distribution ---
np.random.seed(8)

high = 0.45
low = 1.3 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
betas = np.random.random(N_FEATURES)

dist = HierarchicalPairs(
    N_FEATURES, p_active=firing_probs, p_follow=0.6, beta=betas, device=DEVICE
)

# Average L0
samples = dist.sample(100_000)
mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
print(f"Average L0: {mean_l0:.2f}")

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

print(f1_ae[: N_FEATURES // 2].mean().item())
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
SAE_STEPS = 20_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.4

sae_gen = torch.Generator().manual_seed(4)

sae = SAESimple(
    n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=SAE_L1, ortho_coef=0.1, device=DEVICE
).to(DEVICE)


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
# --- SAE Absorption Test ---
PROBE_F0 = 0
PROBE_F1 = 1

fa_match_pos = int((feat_idx == PROBE_F0).nonzero()[0][0])
fb_match_pos = int((feat_idx == PROBE_F1).nonzero()[0][0])
da = dict_idx[fa_match_pos]  # SAE latent matched to PROBE_F0
db = dict_idx[fb_match_pos]  # SAE latent matched to PROBE_F1

print(f"f{PROBE_F0} → SAE latent d{da}   |   f{PROBE_F1} → SAE latent d{db}")

with torch.no_grad():
    # Fire PROBE_F0 alone
    x_fa = torch.zeros(1, N_FEATURES, device=DEVICE)
    x_fa[0, PROBE_F0] = 1.0
    z_fa = sae.encode(tm.ae.encode(x_fa))

    # Fire PROBE_F0 + PROBE_F1
    x_fab = torch.zeros(1, N_FEATURES, device=DEVICE)
    x_fab[0, PROBE_F0] = 1.0
    x_fab[0, PROBE_F1] = 1.0
    z_fab = sae.encode(tm.ae.encode(x_fab))

print(f"\n=== Fire f{PROBE_F0} only ===")
print(f"  SAE latent d{da} (matched f{PROBE_F0}): {z_fa[0, da].item():.4f}")
print(f"  SAE latent d{db} (matched f{PROBE_F1}): {z_fa[0, db].item():.4f}")

print(f"\n=== Fire f{PROBE_F0} + f{PROBE_F1} ===")
print(f"  SAE latent d{da} (matched f{PROBE_F0}): {z_fab[0, da].item():.4f}")
print(f"  SAE latent d{db} (matched f{PROBE_F1}): {z_fab[0, db].item():.4f}")

# %%
# --- Matryoshka SAE Training (via SAELens) ---
from sae_lens import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)

# Nested cumulative widths; final width must equal d_sae.
MATRYOSHKA_WIDTHS = [N_DICT // 4, N_DICT // 2, N_DICT]
MATRYOSHKA_K = 3
MATRYOSHKA_TRAIN_SAMPLES = 4 * SAE_STEPS * SAE_BATCH

print("widths:", MATRYOSHKA_WIDTHS)
print("k:", MATRYOSHKA_K)
print("samples", MATRYOSHKA_TRAIN_SAMPLES)

# Build on CPU first: SAELens registers `topk_threshold` as float64, which MPS
# rejects. We downcast that buffer to float32 before moving to the target device.
matryoshka_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
    d_in=D_HIDDEN,
    d_sae=N_DICT,
    matryoshka_widths=MATRYOSHKA_WIDTHS,
    k=MATRYOSHKA_K,
    device="cpu",
    use_matryoshka_aux_loss=False,
)
matryoshka_sae = MatryoshkaBatchTopKTrainingSAE(matryoshka_cfg)
matryoshka_sae.topk_threshold = matryoshka_sae.topk_threshold.to(torch.float32)
matryoshka_sae.to(DEVICE)

# --- Snapshot callback: capture loss + F1 every ~2000 steps ---
SNAPSHOT_EVERY = 2000
N_SNAPSHOTS = max(1, SAE_STEPS // SNAPSHOT_EVERY)
matryoshka_history: list[dict] = []


def matryoshka_snapshot(trainer):
    """Compute recon loss, L0, MCC, F1, R² on a fresh batch at each snapshot."""
    sae = trainer.sae
    sae.eval()
    with torch.no_grad():
        x_snap = dist.sample(20_000).to(DEVICE)
        h_snap = tm.ae.encode(x_snap)
        z_snap = sae.encode(h_snap)
        recon_snap = sae.decode(z_snap)

        recon_loss = (h_snap - recon_snap).pow(2).sum(dim=-1).mean().item()
        l0_snap = (z_snap > 0).float().sum(dim=-1).mean().item()
        # Dead latents: never fired across the snapshot batch
        dead_snap = int((~(z_snap > 0).any(dim=0)).sum().item())
        total_var_snap = h_snap.var(dim=0).sum().item()
        residual_var_snap = (h_snap - recon_snap).var(dim=0).sum().item()
        r2_snap = 1 - residual_var_snap / total_var_snap

        D_snap = tm.W.detach()
        W_dec_t_snap = sae.W_dec.detach().T
        D_norm_snap = D_snap / D_snap.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm_snap = W_dec_t_snap / W_dec_t_snap.norm(dim=0, keepdim=True).clamp(
            min=1e-8
        )
        cos_sim_snap = (D_norm_snap.T @ W_norm_snap).cpu().numpy()
        feat_idx_snap, dict_idx_snap = linear_sum_assignment(-cos_sim_snap)
        mcc_snap = float(cos_sim_snap[feat_idx_snap, dict_idx_snap].mean())

        gt_snap = x_snap[:, feat_idx_snap] > 0
        pred_snap = z_snap[:, dict_idx_snap] > 0
        tp_snap = (gt_snap & pred_snap).float().sum(dim=0)
        fp_snap = (~gt_snap & pred_snap).float().sum(dim=0)
        fn_snap = (gt_snap & ~pred_snap).float().sum(dim=0)
        prec_snap = tp_snap / (tp_snap + fp_snap + 1e-8)
        rec_snap = tp_snap / (tp_snap + fn_snap + 1e-8)
        f1_snap = (
            (2 * prec_snap * rec_snap / (prec_snap + rec_snap + 1e-8)).mean().item()
        )

    matryoshka_history.append(
        {
            "step": trainer.n_training_steps,
            "recon_loss": recon_loss,
            "l0": l0_snap,
            "r2": r2_snap,
            "mcc": mcc_snap,
            "f1": f1_snap,
            "dead": dead_snap,
        }
    )
    sae.train()
    print(
        f"  [snap step={trainer.n_training_steps}] "
        f"loss={recon_loss:.4f} L0={l0_snap:.1f} "
        f"MCC={mcc_snap:.3f} F1={f1_snap:.3f} R²={r2_snap:.3f} "
        f"DL={dead_snap}"
    )


tm.train_saes(
    [SAEEntry(sae=matryoshka_sae, type="Matryoshka", label="matryoshka")],
    training_samples=MATRYOSHKA_TRAIN_SAMPLES,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
    n_snapshots=N_SNAPSHOTS,
    snapshot_fn=matryoshka_snapshot,
    verbose=False,
)
# %%
tm.evaluate_saes(["matryoshka"], num_samples=30000, verbose=True)

# --- Plot Matryoshka training curves (recon loss + F1/MCC/R² over time) ---
hist_steps = [h["step"] for h in matryoshka_history]
hist_loss = [h["recon_loss"] for h in matryoshka_history]
hist_f1 = [h["f1"] for h in matryoshka_history]
hist_mcc = [h["mcc"] for h in matryoshka_history]
hist_r2 = [h["r2"] for h in matryoshka_history]

fig_mh = make_subplots(specs=[[{"secondary_y": True}]])
fig_mh.add_trace(
    go.Scatter(x=hist_steps, y=hist_loss, mode="lines+markers", name="recon loss"),
    secondary_y=False,
)
fig_mh.add_trace(
    go.Scatter(x=hist_steps, y=hist_f1, mode="lines+markers", name="F1"),
    secondary_y=True,
)
fig_mh.add_trace(
    go.Scatter(x=hist_steps, y=hist_mcc, mode="lines+markers", name="MCC"),
    secondary_y=True,
)
fig_mh.add_trace(
    go.Scatter(x=hist_steps, y=hist_r2, mode="lines+markers", name="R²"),
    secondary_y=True,
)
fig_mh.update_xaxes(title_text="training step", **AXIS_STYLE)
fig_mh.update_yaxes(title_text="recon loss", secondary_y=False, **AXIS_STYLE)
fig_mh.update_yaxes(title_text="F1 / MCC / R²", secondary_y=True, **AXIS_STYLE)
fig_mh.update_layout(title="Matryoshka training curves", **LAYOUT_DEFAULTS)
fig_mh.show()

print(f"Matryoshka widths={MATRYOSHKA_WIDTHS}, k={MATRYOSHKA_K}")
print(f"F1 = {tm.saes_f1_score['matryoshka']:.4f}")
print(f"MCC = {tm.saes_mcc['matryoshka']:.4f}")
print(f"L0 = {tm.saes_l0['matryoshka']:.2f}")
print(f"R² = {tm.saes_explained_variance['matryoshka']:.4f}")

# %%
# --- Matryoshka SAE Metrics (via JumpReLU inference conversion) ---
# SAELens converts BatchTopK / Matryoshka training SAEs into a JumpReLU SAE
# for inference: the learned `topk_threshold` scalar is broadcast to a
# per-feature `threshold` buffer, and gating becomes a stateless elementwise
# JumpReLU. This removes BatchTopK's batch-size dependence entirely, so we
# can encode arbitrary N_TEST in one shot without any chunking.
import tempfile  # noqa: E402


def to_jumprelu_inference_sae(training_sae, device):
    """Round-trip a (Matryoshka)BatchTopK training SAE through disk to get
    its JumpReLU inference form."""
    with tempfile.TemporaryDirectory() as tmpdir:
        training_sae.save_inference_model(tmpdir)
        return SAE.load_from_disk(tmpdir, device=device)


matryoshka_inf = to_jumprelu_inference_sae(matryoshka_sae, DEVICE)
print(f"Matryoshka inference SAE arch: {matryoshka_inf.cfg.architecture()}")

N_TEST = 100_000

with torch.no_grad():
    test_x_m = dist.sample(N_TEST).to(DEVICE)
    test_hidden_m = tm.ae.encode(test_x_m)

    # Single-shot encode/decode — JumpReLU is stateless w.r.t. batch size.
    test_z_m = matryoshka_inf.encode(test_hidden_m)
    test_recon_m = matryoshka_inf.decode(test_z_m)

    # L0
    l0_m = (test_z_m > 0).float().sum(dim=-1).mean().item()
    # Dead latents: never fired across the test batch
    dead_m = int((~(test_z_m > 0).any(dim=0)).sum().item())

    # R² (explained variance)
    total_var_m = test_hidden_m.var(dim=0).sum().item()
    residual_var_m = (test_hidden_m - test_recon_m).var(dim=0).sum().item()
    r2_m = 1 - residual_var_m / total_var_m

    # MCC matching (cosine similarity, NO abs)
    D_m = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
    W_dec_t_m = matryoshka_inf.W_dec.detach().T  # (D_HIDDEN, N_DICT)
    D_norm_m = D_m / D_m.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm_m = W_dec_t_m / W_dec_t_m.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim_m = (D_norm_m.T @ W_norm_m).cpu().numpy()  # (N_FEATURES, N_DICT)

    feat_idx_m, dict_idx_m = linear_sum_assignment(-cos_sim_m)
    mcc_m = float(cos_sim_m[feat_idx_m, dict_idx_m].mean())

    # F1 (detection) — macro
    gt_active_m = test_x_m[:, feat_idx_m] > 0
    pred_active_m = test_z_m[:, dict_idx_m] > 0
    tp_m = (gt_active_m & pred_active_m).float().sum(dim=0)
    fp_m = (~gt_active_m & pred_active_m).float().sum(dim=0)
    fn_m = (gt_active_m & ~pred_active_m).float().sum(dim=0)
    prec_m = tp_m / (tp_m + fp_m + 1e-8)
    rec_m = tp_m / (tp_m + fn_m + 1e-8)
    f1_m = (2 * prec_m * rec_m / (prec_m + rec_m + 1e-8)).mean().item()

    # F1 (detection) — micro (less sensitive to per-feature noise)
    tp_tot = tp_m.sum()
    fp_tot = fp_m.sum()
    fn_tot = fn_m.sum()
    prec_micro_m = (tp_tot / (tp_tot + fp_tot + 1e-8)).item()
    rec_micro_m = (tp_tot / (tp_tot + fn_tot + 1e-8)).item()
    f1_micro_m = 2 * prec_micro_m * rec_micro_m / (prec_micro_m + rec_micro_m + 1e-8)

print(
    f"[Matryoshka, JumpReLU inference] N_TEST={N_TEST} "
    f"prec={prec_m.mean():.4f} rec={rec_m.mean():.4f}"
)
print(
    f"L0={l0_m:.1f}  MCC={mcc_m:.4f}  F1(macro)={f1_m:.4f}  "
    f"F1(micro)={f1_micro_m:.4f}  R²={r2_m:.4f}  DL={dead_m}"
)

# %%
# --- Matryoshka Absorption Test ---
fa_match_pos_m = int((feat_idx_m == PROBE_F0).nonzero()[0][0])
fb_match_pos_m = int((feat_idx_m == PROBE_F1).nonzero()[0][0])
da_m = dict_idx_m[fa_match_pos_m]
db_m = dict_idx_m[fb_match_pos_m]

print(f"[Matryoshka] f{PROBE_F0} → latent d{da_m}   |   f{PROBE_F1} → latent d{db_m}")

with torch.no_grad():
    x_fa_m = torch.zeros(1, N_FEATURES, device=DEVICE)
    x_fa_m[0, PROBE_F0] = 1.0
    z_fa_m = matryoshka_inf.encode(tm.ae.encode(x_fa_m))

    x_fab_m = torch.zeros(1, N_FEATURES, device=DEVICE)
    x_fab_m[0, PROBE_F0] = 1.0
    x_fab_m[0, PROBE_F1] = 1.0
    z_fab_m = matryoshka_inf.encode(tm.ae.encode(x_fab_m))

print(f"\n=== Fire f{PROBE_F0} only ===")
print(
    f"  Matryoshka latent d{da_m} (matched f{PROBE_F0}): {z_fa_m[0, da_m].item():.4f}"
)
print(
    f"  Matryoshka latent d{db_m} (matched f{PROBE_F1}): {z_fa_m[0, db_m].item():.4f}"
)

print(f"\n=== Fire f{PROBE_F0} + f{PROBE_F1} ===")
print(
    f"  Matryoshka latent d{da_m} (matched f{PROBE_F0}): {z_fab_m[0, da_m].item():.4f}"
)
print(
    f"  Matryoshka latent d{db_m} (matched f{PROBE_F1}): {z_fab_m[0, db_m].item():.4f}"
)

# %%
# --- BatchTopK SAE Training (via SAELens) ---
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

BATCHTOPK_K = 3
BATCHTOPK_TRAIN_SAMPLES = 4 * SAE_STEPS * SAE_BATCH
print("k:", BATCHTOPK_K)
print("samples", BATCHTOPK_TRAIN_SAMPLES)

# Build on CPU first: SAELens registers `topk_threshold` as float64, which MPS
# rejects. We downcast that buffer to float32 before moving to the target device.
batchtopk_cfg = BatchTopKTrainingSAEConfig(
    d_in=D_HIDDEN,
    d_sae=N_DICT,
    k=BATCHTOPK_K,
    device="cpu",
)
batchtopk_sae = BatchTopKTrainingSAE(batchtopk_cfg)
batchtopk_sae.topk_threshold = batchtopk_sae.topk_threshold.to(torch.float32)
batchtopk_sae.to(DEVICE)

# --- Snapshot callback: capture loss + F1 every ~SNAPSHOT_EVERY steps ---
N_SNAPSHOTS_B = max(1, SAE_STEPS // SNAPSHOT_EVERY)
batchtopk_history: list[dict] = []


def batchtopk_snapshot(trainer):
    """Compute recon loss, L0, MCC, F1, R² on a fresh batch at each snapshot."""
    sae = trainer.sae
    sae.eval()
    with torch.no_grad():
        x_snap = dist.sample(20_000).to(DEVICE)
        h_snap = tm.ae.encode(x_snap)
        z_snap = sae.encode(h_snap)
        recon_snap = sae.decode(z_snap)

        recon_loss = (h_snap - recon_snap).pow(2).sum(dim=-1).mean().item()
        l0_snap = (z_snap > 0).float().sum(dim=-1).mean().item()
        # Dead latents: never fired across the snapshot batch
        dead_snap = int((~(z_snap > 0).any(dim=0)).sum().item())
        total_var_snap = h_snap.var(dim=0).sum().item()
        residual_var_snap = (h_snap - recon_snap).var(dim=0).sum().item()
        r2_snap = 1 - residual_var_snap / total_var_snap

        D_snap = tm.W.detach()
        W_dec_t_snap = sae.W_dec.detach().T
        D_norm_snap = D_snap / D_snap.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm_snap = W_dec_t_snap / W_dec_t_snap.norm(dim=0, keepdim=True).clamp(
            min=1e-8
        )
        cos_sim_snap = (D_norm_snap.T @ W_norm_snap).cpu().numpy()
        feat_idx_snap, dict_idx_snap = linear_sum_assignment(-cos_sim_snap)
        mcc_snap = float(cos_sim_snap[feat_idx_snap, dict_idx_snap].mean())

        gt_snap = x_snap[:, feat_idx_snap] > 0
        pred_snap = z_snap[:, dict_idx_snap] > 0
        tp_snap = (gt_snap & pred_snap).float().sum(dim=0)
        fp_snap = (~gt_snap & pred_snap).float().sum(dim=0)
        fn_snap = (gt_snap & ~pred_snap).float().sum(dim=0)
        prec_snap = tp_snap / (tp_snap + fp_snap + 1e-8)
        rec_snap = tp_snap / (tp_snap + fn_snap + 1e-8)
        f1_snap = (
            (2 * prec_snap * rec_snap / (prec_snap + rec_snap + 1e-8)).mean().item()
        )

    batchtopk_history.append(
        {
            "step": trainer.n_training_steps,
            "recon_loss": recon_loss,
            "l0": l0_snap,
            "r2": r2_snap,
            "mcc": mcc_snap,
            "f1": f1_snap,
            "dead": dead_snap,
        }
    )
    sae.train()
    print(
        f"  [snap step={trainer.n_training_steps}] "
        f"loss={recon_loss:.4f} L0={l0_snap:.1f} "
        f"MCC={mcc_snap:.3f} F1={f1_snap:.3f} R²={r2_snap:.3f} "
        f"DL={dead_snap}"
    )


tm.train_saes(
    [SAEEntry(sae=batchtopk_sae, type="BatchTopK", label="batchtopk")],
    training_samples=BATCHTOPK_TRAIN_SAMPLES,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
    n_snapshots=N_SNAPSHOTS_B,
    snapshot_fn=batchtopk_snapshot,
    verbose=False,
)

# %%
tm.evaluate_saes(["batchtopk"], num_samples=10_000, verbose=True)

# %%
# --- Plot BatchTopK training curves ---
hist_steps_b = [h["step"] for h in batchtopk_history]
hist_loss_b = [h["recon_loss"] for h in batchtopk_history]
hist_f1_b = [h["f1"] for h in batchtopk_history]
hist_mcc_b = [h["mcc"] for h in batchtopk_history]
hist_r2_b = [h["r2"] for h in batchtopk_history]

fig_bh = make_subplots(specs=[[{"secondary_y": True}]])
fig_bh.add_trace(
    go.Scatter(x=hist_steps_b, y=hist_loss_b, mode="lines+markers", name="recon loss"),
    secondary_y=False,
)
fig_bh.add_trace(
    go.Scatter(x=hist_steps_b, y=hist_f1_b, mode="lines+markers", name="F1"),
    secondary_y=True,
)
fig_bh.add_trace(
    go.Scatter(x=hist_steps_b, y=hist_mcc_b, mode="lines+markers", name="MCC"),
    secondary_y=True,
)
fig_bh.add_trace(
    go.Scatter(x=hist_steps_b, y=hist_r2_b, mode="lines+markers", name="R²"),
    secondary_y=True,
)
fig_bh.update_xaxes(title_text="training step", **AXIS_STYLE)
fig_bh.update_yaxes(title_text="recon loss", secondary_y=False, **AXIS_STYLE)
fig_bh.update_yaxes(title_text="F1 / MCC / R²", secondary_y=True, **AXIS_STYLE)
fig_bh.update_layout(title="BatchTopK training curves", **LAYOUT_DEFAULTS)
fig_bh.show()

print(f"BatchTopK k={BATCHTOPK_K}")
print(f"F1 = {tm.saes_f1_score['batchtopk']:.4f}")
print(f"MCC = {tm.saes_mcc['batchtopk']:.4f}")
print(f"L0 = {tm.saes_l0['batchtopk']:.2f}")
print(f"R² = {tm.saes_explained_variance['batchtopk']:.4f}")

# %%
# --- BatchTopK SAE Metrics (via JumpReLU inference conversion) ---
batchtopk_inf = to_jumprelu_inference_sae(batchtopk_sae, DEVICE)
print(f"BatchTopK inference SAE arch: {batchtopk_inf.cfg.architecture()}")

N_TEST_B = 100_000

with torch.no_grad():
    test_x_b = dist.sample(N_TEST_B).to(DEVICE)
    test_hidden_b = tm.ae.encode(test_x_b)

    # Single-shot encode/decode — JumpReLU is stateless w.r.t. batch size.
    test_z_b = batchtopk_inf.encode(test_hidden_b)
    test_recon_b = batchtopk_inf.decode(test_z_b)

    # L0
    l0_b = (test_z_b > 0).float().sum(dim=-1).mean().item()
    # Dead latents: never fired across the test batch
    dead_b = int((~(test_z_b > 0).any(dim=0)).sum().item())

    # R² (explained variance)
    total_var_b = test_hidden_b.var(dim=0).sum().item()
    residual_var_b = (test_hidden_b - test_recon_b).var(dim=0).sum().item()
    r2_b = 1 - residual_var_b / total_var_b

    # MCC matching (cosine similarity, NO abs)
    D_b = tm.W.detach()  # (D_HIDDEN, N_FEATURES)
    W_dec_t_b = batchtopk_inf.W_dec.detach().T  # (D_HIDDEN, N_DICT)
    D_norm_b = D_b / D_b.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm_b = W_dec_t_b / W_dec_t_b.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim_b = (D_norm_b.T @ W_norm_b).cpu().numpy()  # (N_FEATURES, N_DICT)

    feat_idx_b, dict_idx_b = linear_sum_assignment(-cos_sim_b)
    mcc_b = float(cos_sim_b[feat_idx_b, dict_idx_b].mean())

    # F1 (detection) — macro
    gt_active_b = test_x_b[:, feat_idx_b] > 0
    pred_active_b = test_z_b[:, dict_idx_b] > 0
    tp_b = (gt_active_b & pred_active_b).float().sum(dim=0)
    fp_b = (~gt_active_b & pred_active_b).float().sum(dim=0)
    fn_b = (gt_active_b & ~pred_active_b).float().sum(dim=0)
    prec_b = tp_b / (tp_b + fp_b + 1e-8)
    rec_b = tp_b / (tp_b + fn_b + 1e-8)
    f1_b = (2 * prec_b * rec_b / (prec_b + rec_b + 1e-8)).mean().item()

    # F1 (detection) — micro
    tp_tot_b = tp_b.sum()
    fp_tot_b = fp_b.sum()
    fn_tot_b = fn_b.sum()
    prec_micro_b = (tp_tot_b / (tp_tot_b + fp_tot_b + 1e-8)).item()
    rec_micro_b = (tp_tot_b / (tp_tot_b + fn_tot_b + 1e-8)).item()
    f1_micro_b = 2 * prec_micro_b * rec_micro_b / (prec_micro_b + rec_micro_b + 1e-8)

print(
    f"[BatchTopK, JumpReLU inference] N_TEST={N_TEST_B} "
    f"prec={prec_b.mean():.4f} rec={rec_b.mean():.4f}"
)
print(
    f"L0={l0_b:.1f}  MCC={mcc_b:.4f}  F1(macro)={f1_b:.4f}  "
    f"F1(micro)={f1_micro_b:.4f}  R²={r2_b:.4f}  DL={dead_b}"
)
# %%
