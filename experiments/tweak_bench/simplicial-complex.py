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
D_HIDDEN = 200
N_EPOCHS = 50_000
BATCH_SIZE = 512

# %%
# --- Distribution ---
FACE_DIM = 7
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
# --- Feature-collapse check ---
# Normalize columns of W and inspect the Gram matrix. Off-diagonal entries
# near ±1 indicate two features that have collapsed onto the same direction.
COLLAPSE_THRESH = 0.8
with torch.no_grad():
    W_n = tm.W_normalized_features  # (D_HIDDEN, N_FEATURES)
    gram = (W_n.T @ W_n).cpu()
    i_idx, j_idx = torch.triu_indices(N_FEATURES, N_FEATURES, offset=1)
    pair_sims = gram[i_idx, j_idx]
    max_abs, max_k = pair_sims.max(dim=0)
    collapsed_mask = pair_sims >= COLLAPSE_THRESH

n_collapsed = int(collapsed_mask.sum().item())
print(
    f"[collapse] max|cos|={max_abs.item():.4f} "
    f"(f{int(i_idx[max_k])}–f{int(j_idx[max_k])} = {pair_sims[max_k].item():+.4f})  "
    f"pairs ≥ {COLLAPSE_THRESH}: {n_collapsed}"
)
if n_collapsed > 0:
    top_k = collapsed_mask.nonzero(as_tuple=True)[0][:20]
    for k in top_k:
        k = int(k)
        print(f"  f{int(i_idx[k])}–f{int(j_idx[k])}: cos={pair_sims[k].item():+.4f}")

fig_gram = px.histogram(
    x=pair_sims.numpy(),
    nbins=120,
    labels={"x": "cos(w_i, w_j)"},
    title=f"Pairwise cosine similarity of W columns (max(·)={max_abs.item():.3f})",
)
fig_gram.add_vline(x=COLLAPSE_THRESH, line_color="red", line_dash="dash")
fig_gram.update_layout(**LAYOUT_DEFAULTS)
fig_gram.update_xaxes(**AXIS_STYLE)
fig_gram.update_yaxes(**AXIS_STYLE)
fig_gram.show()


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
SAE_STEPS = 100_000
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


HOOK_BATCH = 10_000
HOOK_FREQ = 500


def loss_hook(d):
    sae = d["sae"]
    sae.eval()
    with torch.no_grad():
        x = dist.sample(HOOK_BATCH).to(DEVICE)
        h = tm.ae.encode(x)
        x_hat, z = sae.forward(h)
        val = sae.loss(h, x_hat, z).item()
    sae.train()
    return (d["step"], val)


def f1_hook(d):
    sae = d["sae"]
    sae.eval()
    with torch.no_grad():
        x = dist.sample(HOOK_BATCH).to(DEVICE)
        h = tm.ae.encode(x)
        z = sae.encode(h)

        D_h = tm.W.detach()
        W_dec_t = sae.W_dec.detach().T
        D_norm = D_h / D_h.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = (D_norm.T @ W_norm).cpu().numpy()
        feat_idx, dict_idx = linear_sum_assignment(-cos_sim)

        gt = x[:, feat_idx] > 0
        pred = z[:, dict_idx] > 0
        tp = (gt & pred).float().sum(dim=0)
        fp = (~gt & pred).float().sum(dim=0)
        fn_ = (gt & ~pred).float().sum(dim=0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn_ + 1e-8)
        val = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()
    sae.train()
    return (d["step"], val)


loss_history, f1_history = sae.train_sae(
    data_fn=data_fn,
    n_steps=SAE_STEPS,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
    hooks=[loss_hook, f1_hook],
    hook_freq=HOOK_FREQ,
)
# %%
loss_steps, loss_values = zip(*loss_history)
f1_steps, f1_values = zip(*f1_history)
fig_sae_curves = make_subplots(specs=[[{"secondary_y": True}]])
fig_sae_curves.add_trace(
    go.Scatter(x=loss_steps, y=loss_values, mode="lines+markers", name="loss"),
    secondary_y=False,
)
fig_sae_curves.add_trace(
    go.Scatter(x=f1_steps, y=f1_values, mode="lines+markers", name="F1"),
    secondary_y=True,
)
fig_sae_curves.update_xaxes(title_text="training step", **AXIS_STYLE)
fig_sae_curves.update_yaxes(title_text="loss", secondary_y=False, **AXIS_STYLE)
fig_sae_curves.update_yaxes(title_text="F1", secondary_y=True, **AXIS_STYLE)
fig_sae_curves.update_layout(title="SAE training curves", **LAYOUT_DEFAULTS)
fig_sae_curves.show()

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
# --- Splitting Quality ---
# For each dict element d, collect the set of ground-truth features that make
# it fire (via one-hot probes). d is "happy" if there exists a simplex face
# that contains all of those features as a subset. The splitting quality is
# the fraction of *live* dicts that are happy; dead latents are excluded from
# the denominator.
#
# Reuses `sae_acts` (one-hot → dict activations) from the SAE Metrics cell.
_face_sets = [frozenset(f) for f in faces]
_face_size = FACE_DIM + 1
_sae_acts_bool = sae_acts > 0  # (N_FEATURES, N_DICT)

happy = 0
unhappy = 0
dead = 0
for d in range(N_DICT):
    firing_feats = frozenset(np.nonzero(_sae_acts_bool[:, d])[0].tolist())
    if not firing_feats:
        dead += 1
        continue
    # |firing_feats| > face_size can't fit in any simplex — short-circuit.
    if len(firing_feats) > _face_size:
        unhappy += 1
        continue
    if any(firing_feats.issubset(fs) for fs in _face_sets):
        happy += 1
    else:
        unhappy += 1

live = happy + unhappy
splitting_quality = happy / live if live > 0 else 0.0
print(
    f"Splitting quality: {splitting_quality:.4f}  "
    f"(happy={happy}, unhappy={unhappy}, dead={dead}, live={live}/{N_DICT})"
)

# %%
# --- L1 Sweep: splitting quality vs L0 ---
L1_SWEEP = [0.05, 0.08, 0.1, 0.15, 0.2]


def compute_splitting_quality(
    sae_model, face_sets, face_size: int
) -> tuple[float, int, int, int]:
    """Return (splitting_quality, happy, unhappy, dead) for a trained SAE."""
    sae_model.eval()
    with torch.no_grad():
        eye_sq = torch.eye(N_FEATURES, device=DEVICE)
        acts = sae_model.encode(tm.ae.encode(eye_sq)).cpu().numpy() > 0
    h, u, dd = 0, 0, 0
    for col in range(acts.shape[1]):
        feats = frozenset(np.nonzero(acts[:, col])[0].tolist())
        if not feats:
            dd += 1
            continue
        if len(feats) > face_size:
            u += 1
            continue
        if any(feats.issubset(fs) for fs in face_sets):
            h += 1
        else:
            u += 1
    live_ = h + u
    return (h / live_ if live_ > 0 else 0.0), h, u, dd


def compute_mean_l0(sae_model, n_samples: int = 20_000) -> float:
    sae_model.eval()
    with torch.no_grad():
        x = dist.sample(n_samples).to(DEVICE)
        z = sae_model.encode(tm.ae.encode(x))
    return (z > 0).float().sum(dim=-1).mean().item()


sweep_results = []
for _l1 in L1_SWEEP:
    print(f"\n=== Training SAE (L1={_l1}) ===")
    _sae = SAESimple(n_latent=D_HIDDEN, n_dict=N_DICT, l1_coef=_l1, device=DEVICE).to(
        DEVICE
    )
    _sae.train_sae(
        data_fn=data_fn,
        n_steps=SAE_STEPS,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
    )
    _l0 = compute_mean_l0(_sae)
    _sq, _h, _u, _dd = compute_splitting_quality(_sae, _face_sets, _face_size)
    print(
        f"  L1={_l1:.3f}  L0={_l0:.2f}  split-quality={_sq:.4f}  "
        f"(happy={_h}, unhappy={_u}, dead={_dd})"
    )
    sweep_results.append(
        {
            "l1": _l1,
            "l0": _l0,
            "splitting_quality": _sq,
            "happy": _h,
            "unhappy": _u,
            "dead": _dd,
        }
    )

# %%
# --- Plot: L0 vs Splitting Quality ---
_l0s = [r["l0"] for r in sweep_results]
_sqs = [r["splitting_quality"] for r in sweep_results]
_labels = [f"L1={r['l1']}" for r in sweep_results]

fig_sweep = go.Figure()
fig_sweep.add_trace(
    go.Scatter(
        x=_l0s,
        y=_sqs,
        mode="lines+markers+text",
        text=_labels,
        textposition="top center",
        marker=dict(size=12),
        line=dict(width=2),
        name="sweep",
    )
)
fig_sweep.update_xaxes(title_text="Mean L0", **AXIS_STYLE)
fig_sweep.update_yaxes(title_text="Splitting Quality", range=[0, 1.05], **AXIS_STYLE)
fig_sweep.update_layout(title="Splitting Quality vs L0 (L1 sweep)", **LAYOUT_DEFAULTS)
fig_sweep.show()

# %%
# --- BatchTopK Sweep: splitting quality vs L0 ---
import tempfile  # noqa: E402
from sae_lens import (  # noqa: E402
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from occhio.toy_model import SAEEntry  # noqa: E402

K_SWEEP = [2, 4, 6, 8, 10]
BATCHTOPK_TRAIN_SAMPLES = 10000 * SAE_BATCH


def build_batchtopk_sae(k: int, device: str) -> BatchTopKTrainingSAE:
    """Build a BatchTopK training SAE.

    SAELens registers `topk_threshold` as float64 which MPS rejects — we build
    on CPU, downcast the buffer, then move to the target device.
    """
    cfg = BatchTopKTrainingSAEConfig(
        d_in=D_HIDDEN,
        d_sae=N_DICT,
        k=k,
        device="cpu",
    )
    sae_m = BatchTopKTrainingSAE(cfg)
    sae_m.topk_threshold = sae_m.topk_threshold.to(torch.float32)
    sae_m.to(device)
    return sae_m


def to_jumprelu_inference_sae(training_sae, device: str) -> SAE:
    """Round-trip a BatchTopK training SAE through disk to get JumpReLU inference form.

    JumpReLU is stateless w.r.t. batch size, which makes eval batch-independent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        training_sae.save_inference_model(tmpdir)
        return SAE.load_from_disk(tmpdir, device=device)


btk_sweep_results = []
for _k in K_SWEEP:
    print(f"\n=== Training BatchTopK (k={_k}) ===")
    _btk = build_batchtopk_sae(_k, DEVICE)
    tm.train_saes(
        [SAEEntry(sae=_btk, type="BatchTopK", label=f"btk_k{_k}")],
        training_samples=BATCHTOPK_TRAIN_SAMPLES,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
        verbose=False,
    )
    _btk_inf = to_jumprelu_inference_sae(_btk, DEVICE)
    _l0 = compute_mean_l0(_btk_inf)
    _sq, _h, _u, _dd = compute_splitting_quality(_btk_inf, _face_sets, _face_size)
    print(
        f"  k={_k}  L0={_l0:.2f}  split-quality={_sq:.4f}  "
        f"(happy={_h}, unhappy={_u}, dead={_dd})"
    )
    btk_sweep_results.append(
        {
            "k": _k,
            "l0": _l0,
            "splitting_quality": _sq,
            "happy": _h,
            "unhappy": _u,
            "dead": _dd,
        }
    )

# %%
# --- Plot: L0 vs Splitting Quality (ReLU L1-sweep + BatchTopK k-sweep) ---
fig_sweep_combined = go.Figure()

fig_sweep_combined.add_trace(
    go.Scatter(
        x=[r["l0"] for r in sweep_results],
        y=[r["splitting_quality"] for r in sweep_results],
        mode="lines+markers+text",
        # text=[f"L1={r['l1']}" for r in sweep_results],
        textposition="top center",
        marker=dict(size=12),
        line=dict(width=2),
        name="ReLU SAE (L1 sweep)",
    )
)
fig_sweep_combined.add_trace(
    go.Scatter(
        x=[r["l0"] for r in btk_sweep_results],
        y=[r["splitting_quality"] for r in btk_sweep_results],
        mode="lines+markers+text",
        text=[f"k={r['k']}" for r in btk_sweep_results],
        textposition="bottom center",
        marker=dict(size=12),
        line=dict(width=2),
        name="BatchTopK SAE (k sweep)",
    )
)

fig_sweep_combined.update_xaxes(title_text="Mean L0", **AXIS_STYLE)
fig_sweep_combined.update_yaxes(
    title_text="Splitting Quality", range=[0, 1.05], **AXIS_STYLE
)
fig_sweep_combined.update_layout(title="Splitting Quality vs L0", **LAYOUT_DEFAULTS)
# Overlay legend inside the plot area (top-right).
fig_sweep_combined.update_layout(
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
    )
)
fig_sweep_combined.show()

# %%
