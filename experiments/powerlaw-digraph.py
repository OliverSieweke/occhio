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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%  ── config ───────────────────────────────────────────────────────────────
DEVICE = "cpu"
gen = torch.Generator(DEVICE)
gen.manual_seed(42)

N_FEATURES = 100
N_HIDDEN = 10

dist = PowerLawDigraph(
    n_features=N_FEATURES,
    alpha=1.5,  # steepness of power-law in-degree
    p_edge=0.10,  # base edge probability
    p_active=0.03,  # unconditional firing probability
    p_child=0.3,  # cascade probability per active parent
    value_dist="uniform",
    generator=gen,
    device=DEVICE,
)

ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)

# %%  ── graph structure ──────────────────────────────────────────────────────
in_deg = dist.in_degrees().numpy()
out_deg = dist.out_degrees().numpy()
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
adj_np = dist.adjacency.float().numpy()

fig = px.imshow(
    adj_np,
    color_continuous_scale="Blues",
    labels=dict(x="target node i  (edge j→i)", y="source node j", color="edge"),
    title="Adjacency matrix  [adj[j,i] = 1 ⟹ j→i]",
    aspect="equal",
)
fig.show()

# %%  ── empirical activation rates ──────────────────────────────────────────
empirical_act = dist.get_expected_activation(n_samples=50_000).numpy()

# Theoretical lower bound: independent-only (no cascade)
p_active_np = dist.p_active.numpy()

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


def norm_hook(hook_data):
    epoch = hook_data["epoch"]
    norms_t = hook_data["tm"].feature_norms  # Tensor[N_FEATURES]
    return (epoch, torch.stack([in_deg_t, norms_t]))


losses, hook_returns = tm.fit(
    25_000,
    batch_size=512,
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
    norms = tm.feature_norms.numpy()
    interferences = tm.total_feature_interferences.numpy()
    feat_dims = tm.feature_dimensionalities.numpy()

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
    imat = tm.interferences.numpy()

# Sort nodes by in-degree for legible ordering
order = np.argsort(-in_deg)  # descending
imat_sorted = imat[np.ix_(order, order)]

fig = px.imshow(
    imat_sorted,
    color_continuous_scale="Reds",
    labels=dict(color="interference"),
    title="Interference matrix  (rows/cols sorted by in-degree, high→low)",
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

dist.print_graph(center=48)

# %%  ── neighborhood interference heatmap ──────────────────────────────────
CENTER = 2

adj = dist.adjacency  # adj[j, i] = True  ⟹  j → i
parents = adj[:, CENTER].nonzero(as_tuple=True)[0].tolist()
children = adj[CENTER, :].nonzero(as_tuple=True)[0].tolist()

# Build ordered index list: parents | center | children (sorted within groups)
neighborhood = sorted(parents) + [CENTER] + sorted(children)

# Slice the interference matrix to the neighborhood
nbr = np.array(neighborhood)
imat_nbr = imat[np.ix_(nbr, nbr)]

# Build axis labels that show role
role = {i: "parent" for i in parents}
role[CENTER] = "center"
role.update({i: "child" for i in children})
tick_labels = [f"{i} ({role[i]})" for i in neighborhood]

fig = px.imshow(
    imat_nbr,
    x=tick_labels,
    y=tick_labels,
    color_continuous_scale="Reds",
    labels=dict(color="interference"),
    title=f"Interference sub-matrix — node {CENTER} and its direct neighbours",
    aspect="equal",
)
fig.update_xaxes(tickangle=45)
fig.show()

# %%
