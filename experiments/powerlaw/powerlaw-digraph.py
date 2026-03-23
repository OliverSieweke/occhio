# %%
"""
PowerLawDigraph – toy model experiment.

We train a TiedLinearRelu autoencoder on a power-law digraph distribution and
inspect how the learned geometry relates to graph structure (in-degree, cascade
activation rates, etc.).
"""

# %%
from occhio import ToyModel
from occhio.distributions import PowerLawDigraph
from occhio.autoencoder import TiedLinearRelu
from occhio.visualization.dynamic import plot_dynamic_scatter

import torch
import numpy as np
import networkx as nx
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.sae.sae import SAESimple

# %%  ── config ───────────────────────────────────────────────────────────────
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(1)

N_FEATURES = 200
N_HIDDEN = 20

dist = PowerLawDigraph(
    n_features=N_FEATURES,
    alpha=2,
    p_edge=15 / N_FEATURES,
    p_active=2 / N_FEATURES,
    p_child=(0, 0.5),
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%  ── graph structure ──────────────────────────────────────────────────────
in_deg = dist.in_degrees().cpu().numpy()
# Weighted in-degree: sum of p_child over incoming edges for each node
_p_child_matrix = (
    (1.0 - torch.exp(dist._log_survival)).cpu().numpy()
)  # (N, N) p_child[j,i]
weighted_in_deg = _p_child_matrix.sum(axis=0)  # sum over parents j for each child i
out_deg = dist.out_degrees().cpu().numpy()
tot_deg = in_deg + out_deg

print(f"Average in-degree = {in_deg.mean()}")

node_ids = np.arange(N_FEATURES)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["In-degree per node (power-law)", "Out-degree per node"],
)
fig.add_trace(
    go.Bar(x=node_ids, y=in_deg, name="in-degree", marker_color="steelblue"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(x=node_ids, y=out_deg, name="out-degree", marker_color="coral"), row=1, col=2
)
fig.update_layout(title="Graph degree distribution", showlegend=True)
fig.show()


# %%  ── adjacency heatmap ────────────────────────────────────────────────────
adj_np = dist.adjacency.float().cpu().numpy()

fig = px.imshow(
    adj_np,
    color_continuous_scale="Blues",
    labels=dict(x="target node i  (edge j→i)", y="source node j", color="edge"),
    title="Adjacency matrix  [adj[j,i] = 1 ⟹ j→i]",
    aspect="equal",
)
fig.show()

# %%  ── empirical activation rates ──────────────────────────────────────────
empirical_act = dist.get_expected_activation(n_samples=50_000).cpu().numpy()

# Theoretical lower bound: independent-only (no cascade)
p_active_np = dist.p_active.cpu().numpy()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=node_ids,
        y=empirical_act,
        mode="markers+lines",
        name="empirical (w/ cascade)",
        marker=dict(color="steelblue", size=8),
    )
)
fig.add_trace(
    go.Scatter(
        x=node_ids,
        y=p_active_np,
        mode="markers+lines",
        name="independent-only (p_active)",
        marker=dict(color="grey", size=6),
        line=dict(dash="dash"),
    )
)
fig.update_layout(
    title="Empirical activation rate vs independent baseline",
    xaxis_title="node index (0 = highest in-degree)",
    yaxis_title="P(active)",
)
fig.show()

# %%  ── train ─────────────────────────
in_deg_t = torch.from_numpy(in_deg).float()

# %%


def norm_hook(hook_data):
    epoch = hook_data["epoch"]
    norms_t = hook_data["tm"].feature_norms.cpu()  # Tensor[N_FEATURES]
    return (epoch, torch.stack([in_deg_t, norms_t]))


# %%

losses, hook_returns = tm.fit(
    50_000,
    batch_size=1024,
    learning_rate=3e-4,
    verbose=True,
    hooks=[norm_hook],
    hook_freq=500,
)

# %%  ── loss curve ─────────────────────────────
fig = px.line(y=losses, labels={"x": "epoch", "y": "loss"}, title="Training loss")
fig.show()

# %%  ── feature geometry ───────────────────────
with torch.no_grad():
    norms = tm.feature_norms.cpu().numpy()
    interferences = tm.total_feature_interferences.cpu().numpy()
    feat_dims = tm.feature_dimensionalities.cpu().numpy()

# %%  ── histogram: norms and interference ───────────────────────────────────
fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Feature norms", "Total interference"]
)
fig.add_trace(
    go.Histogram(x=norms, nbinsx=30, name="norm", marker_color="steelblue"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(x=interferences, nbinsx=30, name="interference", marker_color="coral"),
    row=1,
    col=2,
)
fig.update_layout(
    title="Distribution of feature norms and interferences", showlegend=True
)
fig.show()

# %%  ── scatter: norm vs interference, coloured by in-degree ───────────────
fig = px.scatter(
    x=norms,
    y=interferences,
    color=in_deg,
    hover_name=[f"node {i}" for i in node_ids],
    labels=dict(x="feature norm", y="total interference", color="in-degree"),
    title="Norm vs interference  (colour = in-degree)",
    color_continuous_scale="Plasma",
)
fig.update_traces(marker=dict(size=10))
fig.show()


# %%  ── dynamic: in-degree vs learned norm ────────────────────────────────
fig = plot_dynamic_scatter(losses, hook_returns[0], loss_stride=100)
fig.update_layout(
    title="In-degree vs feature norm over training",
    xaxis2_title="in-degree",
    yaxis2_title="feature norm",
)
fig.show()

# %%  ── full interference matrix ─────────────────────────────────────────────
with torch.no_grad():
    imat = tm.interferences.cpu().numpy()

# Sort nodes by in-degree for legible ordering
order = np.argsort(-in_deg)  # descending
imat_sorted = imat[np.ix_(order, order)]
vmax = np.max(np.abs(imat_sorted))

fig = px.imshow(
    imat_sorted,
    color_continuous_scale="RdBu",
    labels=dict(color="interference"),
    title="Interference matrix  (rows/cols sorted by in-degree, high→low)",
    zmax=vmax,
    zmin=-vmax,
)
fig.show()

# %%  ── feature dimensionalities ────────────────────────────────────────────
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Feature dimensionality per node",
        "Dimensionality vs weighted in-degree",
    ],
)
fig.add_trace(
    go.Bar(
        x=node_ids,
        y=feat_dims,
        marker_color=in_deg,
        marker_colorscale="Plasma",
        name="dim",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=weighted_in_deg,
        y=feat_dims,
        mode="markers",
        marker=dict(color="steelblue", size=8),
        name="dim",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="node index", row=1, col=1)
fig.update_xaxes(title_text="weighted in-degree (Σ p_child)", row=1, col=2)
fig.update_yaxes(title_text="dimensionality", row=1, col=1)
fig.update_yaxes(title_text="dimensionality", row=1, col=2)
fig.update_layout(title="Feature dimensionalities", showlegend=False)
fig.show()


# %%  ── summary stats ────────────────────────────────────────────────────────
print("=== Graph stats ===")
print(f"  Nodes: {N_FEATURES}   Hidden dims: {N_HIDDEN}")
print(
    f"  Edges: {int(dist.adjacency.sum().item())}  (density={dist.adjacency.float().mean().item():.3f})"
)
print(
    f"  In-degree  — mean: {in_deg.mean():.2f}, max: {in_deg.max():.0f}, min: {in_deg.min():.0f}"
)
print(
    f"  Out-degree — mean: {out_deg.mean():.2f}, max: {out_deg.max():.0f}, min: {out_deg.min():.0f}"
)

print("\n=== Learned geometry ===")
print(f"  Feature norms         — mean: {norms.mean():.3f}, std: {norms.std():.3f}")
print(
    f"  Interferences         — mean: {interferences.mean():.4f}, std: {interferences.std():.4f}"
)
print(f"  Feature dimensionalities — mean: {feat_dims.mean():.3f}")
print(
    f"  Embedded features / hidden dim: {tm.embedded_features_per_hidden_dimensions:.3f}"
)

corr_norm_indeg = float(np.corrcoef(in_deg, norms)[0, 1])
corr_int_indeg = float(np.corrcoef(in_deg, interferences)[0, 1])
corr_act_indeg = float(np.corrcoef(in_deg, empirical_act)[0, 1])
corr_norm_outdeg = float(np.corrcoef(out_deg, norms)[0, 1])
corr_int_outdeg = float(np.corrcoef(out_deg, interferences)[0, 1])

print("\n=== Correlations with in-degree ===")
print(f"  in-degree ↔ norm          : r = {corr_norm_indeg:+.3f}")
print(f"  in-degree ↔ interference  : r = {corr_int_indeg:+.3f}")
print(f"  in-degree ↔ P(active)     : r = {corr_act_indeg:+.3f}")

print("\n=== Correlations with out-degree ===")
print(f"  out-degree ↔ norm         : r = {corr_norm_outdeg:+.3f}")
print(f"  out-degree ↔ interference : r = {corr_int_outdeg:+.3f}")

# %%

dist.print_graph(center=1)

# %%  ── neighborhood interference heatmap ──────────────────────────────────
CENTER = 3

adj = dist.adjacency  # adj[j, i] = True  ⟹  j → i
# children = adj[CENTER, :].nonzero(as_tuple=True)[0].tolist()
parents = adj[:, CENTER].nonzero(as_tuple=True)[0].tolist()

# Build ordered index list: parents | center | children (sorted within groups)
neighborhood = [CENTER] + sorted(parents)  # + sorted(children)

# Slice the interference matrix to the neighborhood
nbr = np.array(neighborhood)
imat_nbr = imat[np.ix_(nbr, nbr)]
vmax = np.max(np.abs(imat_nbr))

# Build axis labels that show role
role = {CENTER: "center"}
# role.update({i: "child" for i in children})
role.update({i: "parent" for i in parents})
tick_labels = [f"{i} ({role[i]})" for i in neighborhood]

fig = px.imshow(
    imat_nbr,
    x=tick_labels,
    y=tick_labels,
    color_continuous_scale="RdBu_r",
    labels=dict(color="interference"),
    title=f"Interference sub-matrix — node {CENTER} and its direct neighbours",
    aspect="equal",
    zmax=vmax,
    zmin=-vmax,
)
fig.update_xaxes(tickangle=45)
fig.show()

# %%  ── interference: children / parents vs unrelated nodes ─────────────────
# adj[j, i] = True ⟹  j → i  (j is a parent of i, i is a child of j)
adj_bool = dist.adjacency.cpu().numpy().astype(bool)

child_interf = np.full(N_FEATURES, np.nan)
parent_interf = np.full(N_FEATURES, np.nan)
other_interf_c = np.full(
    N_FEATURES, np.nan
)  # "others" baseline for nodes with children
other_interf_p = np.full(N_FEATURES, np.nan)  # "others" baseline for nodes with parents
child_interf_sq = np.full(N_FEATURES, np.nan)
parent_interf_sq = np.full(N_FEATURES, np.nan)
other_interf_c_sq = np.full(N_FEATURES, np.nan)
other_interf_p_sq = np.full(N_FEATURES, np.nan)

for i in range(N_FEATURES):
    children_i = np.where(adj_bool[i, :])[0]  # i → j  (i is parent)
    parents_i = np.where(adj_bool[:, i])[0]  # j → i  (i is child)
    relatives = set(children_i) | set(parents_i) | {i}
    others_i = np.array([j for j in range(N_FEATURES) if j not in relatives])

    row = imat[i]

    if len(children_i) > 0:
        child_interf[i] = row[children_i].mean()
        child_interf_sq[i] = (row[children_i] ** 2).mean()
        other_interf_c[i] = row[others_i].mean() if len(others_i) else np.nan
        other_interf_c_sq[i] = (row[others_i] ** 2).mean() if len(others_i) else np.nan

    if len(parents_i) > 0:
        parent_interf[i] = row[parents_i].mean()
        parent_interf_sq[i] = (row[parents_i] ** 2).mean()
        other_interf_p[i] = row[others_i].mean() if len(others_i) else np.nan
        other_interf_p_sq[i] = (row[others_i] ** 2).mean() if len(others_i) else np.nan

has_children = ~np.isnan(child_interf)
has_parents = ~np.isnan(parent_interf)
has_both = has_children & has_parents

# ── summary stats ────────────────────────────────────────────────────────────
print("\n=== Interference: children vs unrelated nodes (per-node means) ===")
print(f"  children   mean interf     : {np.nanmean(child_interf):.5f}")
print(f"  unrelated  mean interf     : {np.nanmean(other_interf_c):.5f}")
print(f"  children   mean interf²    : {np.nanmean(child_interf_sq):.5f}")
print(f"  unrelated  mean interf²    : {np.nanmean(other_interf_c_sq):.5f}")

print("\n=== Interference: parents vs unrelated nodes (per-node means) ===")
print(f"  parents    mean interf     : {np.nanmean(parent_interf):.5f}")
print(f"  unrelated  mean interf     : {np.nanmean(other_interf_p):.5f}")
print(f"  parents    mean interf²    : {np.nanmean(parent_interf_sq):.5f}")
print(f"  unrelated  mean interf²    : {np.nanmean(other_interf_p_sq):.5f}")


# ── scatter plots ─────────────────────────────────────────────────────────────
# Each point is one node.  Points ABOVE the diagonal ⟹ relatives interfere more.
def _diag_range(a, b):
    return [min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b))]


fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Children: interference vs unrelated",
        "Parents: interference vs unrelated",
        "Children mean interf vs Parents mean interf (per node)",
        "",
    ],
    specs=[[{}, {}], [{"colspan": 2}, None]],
    horizontal_spacing=0.12,
    vertical_spacing=0.15,
)

_panels = [
    (
        child_interf[has_children],
        other_interf_c[has_children],
        in_deg[has_children],
        node_ids[has_children],
        1,
        1,
    ),
    (
        parent_interf[has_parents],
        other_interf_p[has_parents],
        in_deg[has_parents],
        node_ids[has_parents],
        1,
        2,
    ),
]

for y_vals, x_vals, color_vals, idx_vals, r, c in _panels:
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                color=color_vals,
                colorscale="turbo",
                size=7,
                showscale=(r == 1 and c == 2),
                colorbar=dict(title="in-degree", x=1.02),
            ),
            customdata=idx_vals,
            hovertemplate="node %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=r,
        col=c,
    )

_x_labels = ["unrelated mean interf", "unrelated mean interf"]
_y_labels = ["children mean interf", "parents mean interf"]
for idx, (_, _, _, _, r, c) in enumerate(_panels):
    fig.update_xaxes(title_text=_x_labels[idx], row=r, col=c)
    fig.update_yaxes(title_text=_y_labels[idx], row=r, col=c)

# Bottom panel: children mean interf (x) vs parents mean interf (y), one point per node
fig.add_trace(
    go.Scatter(
        x=child_interf[has_both],
        y=parent_interf[has_both],
        mode="markers",
        marker=dict(
            color=in_deg[has_both],
            colorscale="turbo",
            size=7,
            showscale=False,
        ),
        customdata=node_ids[has_both],
        hovertemplate="node %{customdata}<extra></extra>",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.update_xaxes(title_text="children mean interf", row=2, col=1)
fig.update_yaxes(title_text="parents mean interf", row=2, col=1)

_diag_lim = _diag_range(child_interf[has_both], parent_interf[has_both])
fig.add_trace(
    go.Scatter(
        x=_diag_lim,
        y=_diag_lim,
        mode="lines",
        line=dict(color="grey", dash="dot", width=1),
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.update_layout(
    title="Per-node interference with relatives vs unrelated  (above diagonal = relatives interfere more)",
    height=900,
)
fig.show()

# %%  ── directed interference asymmetry ────────────────────────────────────
# For each directed edge i→j plot imat[i,j] (parent→child direction) against
# imat[j,i] (child→parent direction).  Off-diagonal ⟹ the model encodes direction.
edge_src, edge_dst = np.where(adj_bool)  # i→j for each edge
ij_vals = imat[edge_src, edge_dst]  # interference i→j
ji_vals = imat[edge_dst, edge_src]  # interference j→i

lim = max(np.abs(ij_vals).max(), np.abs(ji_vals).max()) * 1.05
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=ji_vals,
        y=ij_vals,
        mode="markers",
        marker=dict(
            color=in_deg[edge_src],
            colorscale="inferno",
            size=5,
            opacity=0.6,
            colorbar=dict(title="in-degree of src"),
            showscale=True,
        ),
        customdata=np.stack([edge_src, edge_dst], axis=1),
        hovertemplate="edge %{customdata[0]}→%{customdata[1]}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[-lim, lim],
        y=[-lim, lim],
        mode="lines",
        line=dict(color="grey", dash="dash", width=1),
        showlegend=False,
    )
)
fig.update_layout(
    title="Directed interference asymmetry  (each point = one edge i→j)",
    xaxis_title="imat[j, i]  (child→parent direction)",
    yaxis_title="imat[i, j]  (parent→child direction)",
    xaxis=dict(range=[-lim, lim]),
    yaxis=dict(range=[-lim, lim], scaleanchor="x", scaleratio=1),
)
fig.show()

# %%  ── PCA of W columns colored by in-degree ───────────────────────────────
with torch.no_grad():
    W_np = tm.W.cpu().numpy()  # shape (n_hidden, n_features)

W_cols = W_np.T  # (n_features, n_hidden) — one row per feature
W_centered = W_cols - W_cols.mean(axis=0)
_, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
W_3d = W_centered @ Vt[:3].T  # (n_features, 3)
var_ratio = S[:3] ** 2 / (S**2).sum()

fig = px.scatter_3d(
    x=W_3d[:, 0],
    y=W_3d[:, 1],
    z=W_3d[:, 2],
    color=weighted_in_deg,
    hover_name=[f"node {i}" for i in node_ids],
    color_continuous_scale="Plasma",
    labels=dict(
        x=f"PC1 ({var_ratio[0]:.1%})",
        y=f"PC2 ({var_ratio[1]:.1%})",
        z=f"PC3 ({var_ratio[2]:.1%})",
        color="weighted in-deg",
    ),
    title="PCA of W columns  (each point = one feature vector in hidden space)",
)
fig.update_traces(marker=dict(size=4))
fig.show()

# %%  ── interference vs correlation comparison ─────────────────────────────
N_CORR_SAMPLES = 100_000
samples = dist.sample(N_CORR_SAMPLES).cpu().numpy()  # (N_CORR_SAMPLES, N_FEATURES)

# Pearson correlation matrix from samples
corr_mat = np.corrcoef(samples, rowvar=False)  # (N_FEATURES, N_FEATURES)

# Sort both matrices by in-degree (descending) for comparable layout
imat_sorted_corr = imat[np.ix_(order, order)]
corr_sorted = corr_mat[np.ix_(order, order)]


# ── scatter: imat[i,j] vs corr[i,j] for all off-diagonal pairs ───────────
mask = ~np.eye(N_FEATURES, dtype=bool)
imat_flat = imat[mask]
corr_flat = corr_mat[mask]

fig = px.scatter(
    x=corr_flat,
    y=imat_flat,
    opacity=0.3,
    labels=dict(x="Pearson correlation", y="Interference  imat[i,j]"),
    title="Per-pair: interference vs empirical correlation  (all off-diagonal pairs)",
    trendline="ols",
    trendline_color_override="black",
)
fig.show()

# %%  ── per-edge: p_child vs interference ────────────────────────────────────
edge_j, edge_i = np.where(adj_bool)  # j → i for each edge
edge_p_child = _p_child_matrix[edge_j, edge_i]
edge_interf = imat[edge_j, edge_i]

fig = px.scatter(
    x=edge_p_child,
    y=edge_interf,
    opacity=0.3,
    labels=dict(x="p_child (edge j→i)", y="Interference  imat[j,i]"),
    title="Per-edge: p_child vs interference",
    trendline="ols",
    trendline_color_override="black",
)
fig.show()

# %%  ── SAE training ───────────────────────────────────────────
N_DICT = N_FEATURES + 4
SAE_STEPS = 100_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.05

print("Training SAE on TiedLinearRelu...")
sae = SAESimple(
    n_latent=N_HIDDEN,
    n_dict=N_DICT,
    l1_coef=SAE_L1,
    device=DEVICE,
).to(DEVICE)


def sae_data_fn(n: int) -> torch.Tensor:
    x = dist.sample(n).to(DEVICE)
    return tm.ae.encode(x)


sae_losses = sae.train_sae(
    data_fn=sae_data_fn,
    n_steps=SAE_STEPS,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
)

# %%  ── SAE metrics ──────────────────────────────────────────────────────────
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

# %%  ── SAE loss curve ───────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(y=sae_losses, mode="lines", name="SAE loss", opacity=0.8))
fig.update_layout(
    title="SAE Training Loss",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
)
fig.show()

# %%  ── SAE one-hot activations (Hungarian matched) ──────────────────────────
with torch.no_grad():
    eye = torch.eye(N_FEATURES, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (N_FEATURES, N_DICT)

cost = -sae_acts
feat_idx, dict_idx = linear_sum_assignment(cost)

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
diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
print(
    f"Diagonality = {diagonality:.4f} (diag_sum={diag_sum:.2f}, total={total_sum:.2f})"
)

row_labels = [f"f{f}" for f in row_order]
col_labels = [f"d{d}" for d in col_order]
fig = px.imshow(
    sae_acts_matched,
    labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
    x=col_labels,
    y=row_labels,
    title=f"SAE one-hot activations (Hungarian matched, diag={diagonality:.3f})",
    aspect="auto",
    color_continuous_scale="ylgnbu_r",
)
fig.show()

# %%  ── SAE MCC & detection metrics ─────────────────────────────────────────
with torch.no_grad():
    D = tm.W.detach()  # (N_HIDDEN, N_FEATURES)
    W_dec_t = sae.W_dec.detach().T  # (N_HIDDEN, N_DICT)
    D_norm = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim = (D_norm.T @ W_norm).abs().cpu().numpy()  # (N_FEATURES, N_DICT)
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

    precision_per = tp / (tp + fp + 1e-8)
    recall_per = tp / (tp + fn + 1e-8)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-8)
    fpr_per = fp / (fp + tn + 1e-8)

print(
    f"MCC={mcc:.4f}  Prec={precision_per.mean():.4f}  "
    f"Rec={recall_per.mean():.4f}  F1={f1_per.mean():.4f}  FPR={fpr_per.mean():.4f}"
)

# %%  ── SAE per-feature detection metrics (sorted by weighted in-degree) ────
feat_weighted_indeg = weighted_in_deg[mcc_feat_idx]
feat_order = np.argsort(-feat_weighted_indeg)
x_rank = np.arange(len(feat_order))

fig = make_subplots(
    rows=1,
    cols=4,
    subplot_titles=["Precision", "Recall (TPR)", "F1 Score", "FPR"],
)
fig.add_trace(
    go.Scatter(x=x_rank, y=precision_per[feat_order], mode="lines", name="Precision"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=x_rank, y=recall_per[feat_order], mode="lines", showlegend=False),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(x=x_rank, y=f1_per[feat_order], mode="lines", showlegend=False),
    row=1,
    col=3,
)
fig.add_trace(
    go.Scatter(x=x_rank, y=fpr_per[feat_order], mode="lines", showlegend=False),
    row=1,
    col=4,
)
fig.update_layout(
    title="SAE Per-Feature Detection Metrics (sorted by weighted in-degree)",
    height=400,
    width=1400,
)
for col in range(1, 5):
    fig.update_xaxes(
        title_text="Feature rank (highest weighted in-deg → lowest)", row=1, col=col
    )
fig.show()

# %%  ── SAE summary ─────────────────────────────────────────────────────────
print("\n=== SAE Summary ===")
print(f"  Dict size: {N_DICT}   L1: {SAE_L1}   Steps: {SAE_STEPS}")
print(f"  L0:            {l0:.1f}")
print(f"  Dead features: {n_dead}/{N_DICT}")
print(f"  Recon MSE:     {recon_mse:.6f}")
print(f"  Explained var: {explained_var:.4f}")
print(f"  Diagonality:   {diagonality:.4f}")
print(f"  MCC:           {mcc:.4f}")
print(f"  Precision:     {precision_per.mean():.4f}")
print(f"  Recall:        {recall_per.mean():.4f}")
print(f"  F1:            {f1_per.mean():.4f}")
print(f"  FPR:           {fpr_per.mean():.4f}")

# %%
