# %%
"""
High-dimensional simplicial complex experiment:
100 features, 16 hidden dims
Analyses: facet membership vs interference, dimensionality, norms, and neighbor count.
"""

import random
from collections import Counter

import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
LINE_WIDTH = 2

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(42)

N_FEAT = 1000
N_HIDDEN = 64
FACE_DIM = 8
N_FACES = 10 * (N_FEAT // FACE_DIM)

random.seed(42)
# Deterministic covering: shuffle vertices and chunk into faces so every vertex appears
all_verts = list(range(N_FEAT))
random.shuffle(all_verts)
face_size = FACE_DIM + 1
covering_faces = set()
for i in range(0, N_FEAT, face_size):
    chunk = all_verts[i : i + face_size]
    if len(chunk) < face_size:
        remaining = [v for v in all_verts if v not in chunk]
        chunk += random.sample(remaining, face_size - len(chunk))
    covering_faces.add(tuple(sorted(chunk)))

while len(covering_faces) < N_FACES:
    covering_faces.add(tuple(sorted(random.sample(range(N_FEAT), face_size))))
faces = list(covering_faces)[:N_FACES]

p_active = 4.6 / (N_FACES)

dist = SimplicialComplexDistribution(
    n_vertices=N_FEAT,
    faces=faces,
    p_active=p_active,
    sampling_mode="sparse",
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %% --- Empirical firing frequency ---
samples = dist.sample(10_000)
if isinstance(samples, tuple):
    samples = samples[0]
firing_freq = (samples > 0).float().mean(dim=0).cpu().numpy()
freq_sorted = np.sort(firing_freq)[::-1]

fig = px.line(
    y=freq_sorted,
    labels={"x": "Feature index (sorted by frequency)", "y": "Firing frequency"},
    title="Empirical firing frequency (sorted, descending)",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()


# %%
def loss_hook(d):
    return d["loss"].item()


_, (tracked_losses,) = tm.fit(
    40_000, 256, verbose=True, track_losses=False, hooks=[loss_hook], hook_freq=250
)
fig = px.line(
    y=tracked_losses,
    labels={"x": "Hook step (every 250 epochs)", "y": "Loss"},
    title="Training loss",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %% Compute per-vertex statistics from the simplicial complex
# Number of faces each vertex belongs to
facet_counts = Counter()
for face in faces:
    for v in face:
        facet_counts[v] += 1
facet_count_per_feat = torch.tensor(
    [facet_counts.get(i, 0) for i in range(N_FEAT)], dtype=torch.float32
)

# Number of unique neighbors per vertex (vertices that share at least one face)
neighbor_sets = [set() for _ in range(N_FEAT)]
for face in faces:
    for v in face:
        for u in face:
            if u != v:
                neighbor_sets[v].add(u)
neighbor_count_per_feat = torch.tensor(
    [len(neighbor_sets[i]) for i in range(N_FEAT)], dtype=torch.float32
)

# Number of shared faces for each pair of vertices
shared_faces = torch.zeros(N_FEAT, N_FEAT)
for face in faces:
    for i, v in enumerate(face):
        for u in face[i + 1 :]:
            shared_faces[v, u] += 1
            shared_faces[u, v] += 1

# %% Extract model geometry
W = tm.W.detach().cpu()
feature_norms = tm.feature_norms.detach().cpu()
feature_dims = tm.feature_dimensionalities.detach().cpu()
interferences = tm.interferences.detach().cpu()
interferences_sq = tm.interferences_sq.detach().cpu()
total_interference = tm.total_feature_interferences.detach().cpu()

# # %% --- Plot 1: Shared faces vs interference ---
# # For each pair of features, plot the number of shared faces against interference
rows_i, rows_j = torch.triu_indices(N_FEAT, N_FEAT, offset=1)
pair_shared = shared_faces[rows_i, rows_j].numpy()
pair_interference = interferences[rows_i, rows_j].numpy()

# fig = px.scatter(
#     x=pair_shared,
#     y=pair_interference,
#     labels={"x": "Shared faces", "y": "Interference"},
#     title="Shared faces between feature pairs vs Interference",
#     opacity=0.3,
# )
# fig.show()

# %% --- Plot 2: Facet count vs feature dimensionality ---
fig = px.scatter(
    x=facet_count_per_feat.numpy(),
    y=feature_dims.numpy(),
    labels={"x": "Number of faces containing vertex", "y": "Feature dimensionality"},
    title="Facet membership count vs Feature dimensionality",
    hover_name=[f"v{i}" for i in range(N_FEAT)],
    trendline="ols",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()


# %% --- Plot 3: Neighbor count vs total interference ---
fig = px.scatter(
    x=neighbor_count_per_feat.numpy(),
    y=total_interference.numpy(),
    labels={
        "x": "Number of unique neighbors",
        "y": "Total interference",
    },
    title="Graph degree (neighbor count) vs Total feature interference",
    hover_name=[f"v{i}" for i in range(N_FEAT)],
    trendline="ols",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %% --- Plot 3b: Empirical correlation vs interference ---
empirical_corr = torch.corrcoef(samples.T.cpu())
pair_corr = empirical_corr[rows_i, rows_j].numpy()
pair_interf = interferences[rows_i, rows_j].numpy()

_x = pair_corr[:100_000]
_y = pair_interf[:100_000]
_slope, _intercept = np.polyfit(_x, _y, 1)
_ss_res = np.sum((_y - (_slope * _x + _intercept)) ** 2)
_ss_tot = np.sum((_y - _y.mean()) ** 2)
_r2 = 1 - _ss_res / _ss_tot

fig = px.scatter(
    x=_x,
    y=_y,
    labels={"x": "Empirical correlation", "y": "Interference"},
    # title="Pairwise empirical correlation vs Interference",
    opacity=0.8,
    trendline="ols",
    trendline_color_override="black",
)
fig.update_traces(marker_size=3, selector=dict(mode="markers"))
fig.update_traces(opacity=0.8, selector=dict(mode="lines"))
fig.add_hline(y=0, line_color="gray", line_width=1, layer="below")
fig.add_annotation(
    text=f"slope = {_slope:.3f}<br>R² = {_r2:.3f}",
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

# %% --- Plot 4: Norm, interference, dimensionality (sorted by facet membership) ---
sort_idx = torch.argsort(neighbor_count_per_feat, descending=True)

sorted_norms = feature_norms[sort_idx].numpy()
sorted_interference = total_interference[sort_idx].numpy()
sorted_dims = feature_dims[sort_idx].numpy()
sorted_bias = tm.ae.b.detach().cpu()[sort_idx].numpy()  # ty:ignore

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Feature norm",
        "Total interference",
        "Feature dimensionality",
        "Bias",
    ],
)
fig.add_trace(go.Scatter(y=sorted_norms, mode="lines", name="Norm"), row=1, col=1)
fig.add_trace(
    go.Scatter(y=sorted_interference, mode="lines", name="Interference"), row=1, col=2
)
fig.add_trace(
    go.Scatter(y=sorted_dims, mode="lines", name="Dimensionality"), row=2, col=1
)
fig.add_trace(go.Scatter(y=sorted_bias, mode="lines", name="Bias"), row=2, col=2)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Feature properties (sorted by unique neighbor count, descending)",
    showlegend=False,
    height=600,
)
for i in range(1, 5):
    r, c = (i - 1) // 2 + 1, (i - 1) % 2 + 1
    fig.update_xaxes(title_text="Feature index (sorted)", row=r, col=c, **AXIS_STYLE)
    fig.update_yaxes(row=r, col=c, **AXIS_STYLE)
fig.show()

# %% --- SAE training ---
from scipy.optimize import linear_sum_assignment
from occhio.sae.sae import SAESimple

N_DICT = N_FEAT + 10
SAE_STEPS = 65_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.1

sae = SAESimple(
    n_latent=N_HIDDEN,
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

# %% --- SAE metrics ---
with torch.no_grad():
    test_x = dist.sample(10_000).to(DEVICE)
    test_hidden = tm.ae.encode(test_x)
    test_z = sae.encode(test_hidden)
    test_recon = sae.decode(test_z)

    l0 = (test_z > 0).float().sum(dim=-1).mean().item()
    ever_active = (test_z > 0).any(dim=0)
    n_dead = int((~ever_active).sum().item())
    n_alive = int(ever_active.sum().item())
    recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

    total_var = test_hidden.var(dim=0).sum().item()
    residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
    explained_var = 1 - residual_var / total_var

print(
    f"L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
)

# %% --- SAE loss curve ---
fig = px.line(
    y=sae_losses,
    labels={"x": "Step", "y": "Loss"},
    title="SAE Training Loss",
    log_y=True,
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %% --- SAE one-hot activations (Hungarian matched) ---
with torch.no_grad():
    eye = torch.eye(N_FEAT, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()

cost = -sae_acts
feat_idx, dict_idx = linear_sum_assignment(cost)

matched_feats = set(feat_idx)
matched_dicts = set(dict_idx)
unmatched_feats = [f for f in range(N_FEAT) if f not in matched_feats]
unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

# Sort matched pairs by unique neighbor count (descending)
neighbor_counts_matched = neighbor_count_per_feat.numpy()[feat_idx]
neighbor_sort = np.argsort(-neighbor_counts_matched)
feat_idx_sorted = feat_idx[neighbor_sort]
dict_idx_sorted = dict_idx[neighbor_sort]

row_order = list(feat_idx_sorted) + unmatched_feats
col_order = list(dict_idx_sorted) + unmatched_dicts
sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]

n_matched = len(feat_idx)
diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
total_sum = sae_acts_matched.sum()
diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
print(f"Diagonality = {diagonality:.4f}")

fig = px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
    title=f"SAE one-hot activations (Hungarian matched, diag={diagonality:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.show()

# %% --- Per-feature purity (sorted by neighbor count) ---
# Purity = activation of matched dict element / total activation across all dict elements
purity = np.array(
    [
        sae_acts[feat_idx[i], dict_idx[i]] / (sae_acts[feat_idx[i]].sum() + 1e-8)
        for i in range(n_matched)
    ]
)
# Sort by neighbor count of the matched features (descending)
purity_sort = np.argsort(-neighbor_count_per_feat.numpy()[feat_idx])
fig = px.line(
    y=purity[purity_sort],
    labels={"x": "Feature index (sorted by neighbor count)", "y": "Purity"},
    title="Per-feature SAE purity (matched activation / total activation)",
)
fig.update_layout(**LAYOUT_DEFAULTS)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.show()

# %% --- MCC and detection metrics ---
with torch.no_grad():
    D = tm.W.detach()
    W_dec_t = sae.W_dec.detach().T
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim = (D_norm.T @ W_norm).abs().cpu().numpy()
    mcc_feat_idx, mcc_dict_idx = linear_sum_assignment(-cos_sim)
    mcc = float(cos_sim[mcc_feat_idx, mcc_dict_idx].mean())

    det_x = dist.sample(50_000).to(DEVICE)
    det_hidden = tm.ae.encode(det_x)
    det_z = sae.encode(det_hidden)

    gt_active = det_x[:, mcc_feat_idx] > 0
    pred_active = det_z[:, mcc_dict_idx] > 0

    tp = (gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fp = (~gt_active & pred_active).float().sum(dim=0).cpu().numpy()
    fn = (gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()
    tn = (~gt_active & ~pred_active).float().sum(dim=0).cpu().numpy()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

print(
    f"MCC={mcc:.4f}  Prec={precision.mean():.4f}  "
    f"Rec={recall.mean():.4f}  F1={f1.mean():.4f}  FPR={fpr.mean():.4f}"
)

# %% --- Per-feature detection metrics (sorted by neighbor count) ---
feat_order = np.argsort(-neighbor_count_per_feat.numpy()[mcc_feat_idx])
x_axis = np.arange(len(feat_order))

fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=["Precision", "Recall (TPR)", "F1 Score", "FPR"],
)
fig.add_trace(
    go.Scatter(x=x_axis, y=precision[feat_order], mode="lines", name="Precision"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_axis, y=recall[feat_order], mode="lines", name="Recall", showlegend=False
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(x=x_axis, y=f1[feat_order], mode="lines", name="F1", showlegend=False),
    row=1,
    col=3,
)
fig.add_trace(
    go.Scatter(x=x_axis, y=fpr[feat_order], mode="lines", name="FPR", showlegend=False),
    row=1,
    col=4,
)
fig.update_layout(
    **LAYOUT_DEFAULTS,
    title="Per-Feature Detection Metrics (sorted by neighbor count)",
    height=400,
    width=1400,
    showlegend=False,
)
for col in range(1, 5):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col, **AXIS_STYLE)
    fig.update_yaxes(row=1, col=col, **AXIS_STYLE)
fig.show()

# %%
