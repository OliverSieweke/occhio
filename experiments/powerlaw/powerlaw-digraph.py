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

# %%  ── config ───────────────────────────────────────────────────────────────
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(1)

N_FEATURES = 200
N_HIDDEN = 10

dist = PowerLawDigraph(
    n_features=N_FEATURES,
    alpha=3,
    p_edge=10 / N_FEATURES,
    p_active=1 / N_FEATURES,
    p_child=0.2,
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)

# %%  ── graph structure ──────────────────────────────────────────────────────
in_deg = dist.in_degrees().cpu().numpy()
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

# %%  ── train ───────────────────────────────────────────────────────────────
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

# %%  ── loss curve ───────────────────────────────────────────────────────────
fig = px.line(y=losses, labels={"x": "epoch", "y": "loss"}, title="Training loss")
fig.show()

# %%  ── feature geometry ────────────────────────────────────────────────────
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
        "Dimensionality vs in-degree",
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
        x=in_deg,
        y=feat_dims,
        mode="markers",
        marker=dict(color="steelblue", size=8),
        name="dim",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="node index", row=1, col=1)
fig.update_xaxes(title_text="in-degree", row=1, col=2)
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
W_2d = W_centered @ Vt[:2].T  # (n_features, 2)
var_ratio = S[:2] ** 2 / (S**2).sum()

fig = px.scatter(
    x=W_2d[:, 0],
    y=W_2d[:, 1],
    color=in_deg,
    hover_name=[f"node {i}" for i in node_ids],
    color_continuous_scale="Plasma",
    labels=dict(
        x=f"PC1 ({var_ratio[0]:.1%})", y=f"PC2 ({var_ratio[1]:.1%})", color="in-degree"
    ),
    title="PCA of W columns  (each point = one feature vector in hidden space)",
)
fig.update_traces(marker=dict(size=8))
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

# %% ── ego-graph: centre node and its direct neighbourhood ───────────────────
# adj_np[j, i] > 0  ⟹  directed edge j → i
# Layout: parents left, centre middle, children right.
CENTER = 3  # node 0 has the highest expected in-degree by construction

parents = [j for j in range(N_FEATURES) if adj_np[j, CENTER] and j != CENTER]
children = [i for i in range(N_FEATURES) if adj_np[CENTER, i] and i != CENTER]

print(f"Node {CENTER}: {len(parents)} parents  |  {len(children)} children")

all_ego_nodes = [CENTER] + parents + children

role_of = {CENTER: "centre"}
role_of.update({p: "parent" for p in parents})
role_of.update({c: "child" for c in children})


def _column(nodes, x):
    """Evenly-spaced y-positions for a column of nodes at a fixed x."""
    ys = np.linspace(1, -1, len(nodes)) if len(nodes) > 1 else [0.0]
    return {nd: (float(x), float(y)) for nd, y in zip(nodes, ys)}


pos = {CENTER: (0.0, 0.0), **_column(parents, -2.0), **_column(children, 2.0)}

# ── ALL directed edges within the neighbourhood ───────────────────────────────
edge_data = []
for u in all_ego_nodes:
    for v in all_ego_nodes:
        if u != v and adj_np[u, v]:
            edge_data.append((u, v, float(imat[u, v]), f"{role_of[u]}→{role_of[v]}"))

interf_vals = (
    np.array([v for _, _, v, _ in edge_data]) if edge_data else np.array([0.0])
)
vmax_e = max(np.abs(interf_vals).max(), 1e-8)


def _clr_rgba(val, alpha=0.85):
    rgb = pc.sample_colorscale("RdBu_r", [(val + vmax_e) / (2 * vmax_e)])[0]
    r, g, b = pc.unlabel_rgb(rgb)
    return f"rgba({int(r)},{int(g)},{int(b)},{alpha})"


def _lw(val):
    return 0.5 + 4.5 * abs(val) / vmax_e


def _arrow_endpoints(x0, y0, x1, y1, r=0.12):
    """Shorten both ends of an edge by r data-units so arrows land at node border."""
    dx, dy = x1 - x0, y1 - y0
    L = np.sqrt(dx**2 + dy**2) or 1e-8
    return x0 + r * dx / L, y0 + r * dy / L, x1 - r * dx / L, y1 - r * dy / L


# ── build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

# Arrows: one annotation per edge, plus an invisible midpoint marker for hover.
for s, d, val, label in edge_data:
    x0, y0 = pos[s]
    x1, y1 = pos[d]
    ax, ay, x1a, y1a = _arrow_endpoints(x0, y0, x1, y1)
    color = _clr_rgba(val)
    fig.add_annotation(
        x=x1a,
        y=y1a,
        ax=ax,
        ay=ay,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.0,
        arrowwidth=_lw(val),
        arrowcolor=color,
        text="",
    )
    fig.add_trace(
        go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            mode="markers",
            marker=dict(size=12, opacity=0),
            customdata=[[s, d, val, label]],
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "imat[%{customdata[0]:.0f} → %{customdata[1]:.0f}]"
                " = %{customdata[2]:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

# Nodes — single trace so the colorscale maps correctly across all nodes.
norm_min_ego = min(norms[n] for n in all_ego_nodes)
norm_max_ego = max(norms[n] for n in all_ego_nodes)
indeg_min_ego = min(in_deg[n] for n in all_ego_nodes)
indeg_max_ego = max(in_deg[n] for n in all_ego_nodes) + 1e-8

border_color = {"centre": "black", "parent": "#4e79a7", "child": "#e15759"}
border_width = {"centre": 3, "parent": 1.5, "child": 1.5}
base_size = {"centre": 30, "parent": 18, "child": 18}

node_sizes = [
    base_size[role_of[nd]]
    + 12 * (in_deg[nd] - indeg_min_ego) / (indeg_max_ego - indeg_min_ego)
    for nd in all_ego_nodes
]
fig.add_trace(
    go.Scatter(
        x=[pos[nd][0] for nd in all_ego_nodes],
        y=[pos[nd][1] for nd in all_ego_nodes],
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=[norms[nd] for nd in all_ego_nodes],
            colorscale="plasma",
            cmin=norm_min_ego,
            cmax=norm_max_ego,
            showscale=True,
            colorbar=dict(title="feature norm", x=1.0, thickness=14),
            line=dict(
                width=[border_width[role_of[nd]] for nd in all_ego_nodes],
                color=[border_color[role_of[nd]] for nd in all_ego_nodes],
            ),
        ),
        text=[str(nd) for nd in all_ego_nodes],
        textposition="middle center",
        textfont=dict(size=9, color="white"),
        customdata=[
            [nd, in_deg[nd], out_deg[nd], norms[nd], role_of[nd]]
            for nd in all_ego_nodes
        ],
        hovertemplate=(
            "<b>node %{customdata[0]:.0f}</b>  (%{customdata[4]})<br>"
            "in-degree: %{customdata[1]:.0f}<br>"
            "out-degree: %{customdata[2]:.0f}<br>"
            "norm: %{customdata[3]:.3f}<extra></extra>"
        ),
        showlegend=False,
    )
)

# Dummy trace for the edge interference colorbar.
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="RdBu_r",
            showscale=True,
            cmin=-vmax_e,
            cmax=vmax_e,
            colorbar=dict(title="interference<br>imat[src→dst]", x=1.12, thickness=14),
        ),
        hoverinfo="none",
        showlegend=False,
    )
)

# Column header annotations.
for label, x in [
    (f"parents  (j → {CENTER})", -2.0),
    (f"node {CENTER}", 0.0),
    (f"children  ({CENTER} → i)", 2.0),
]:
    fig.add_annotation(
        x=x,
        y=1.05,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=12),
        xref="x",
        yref="paper",
    )

n_rows = max(len(parents), len(children), 1)
fig.update_layout(
    title=(
        f"Ego-graph — node {CENTER}  "
        f"({len(parents)} parents, {len(children)} children)<br>"
        "<sup>Node colour = feature norm  |  Node size ∝ in-degree  |  "
        "Edge colour = imat[src→dst]  (blue < 0, red > 0)  |  "
        "Edge width = |interference|  |  "
        "Border: blue = parent, coral = child</sup>"
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 2.8]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
    plot_bgcolor="white",
    hovermode="closest",
    width=1000,
    height=max(500, 60 * n_rows),
)
fig.show()

# %%
