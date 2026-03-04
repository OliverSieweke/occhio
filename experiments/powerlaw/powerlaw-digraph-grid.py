# %%
"""
PowerLawDigraph – toy model experiment.

We train a TiedLinearRelu autoencoder on a power-law digraph distribution and
inspect how the learned geometry relates to graph structure (in-degree, cascade
activation rates, etc.).
"""

# %%
from occhio import ToyModel
from occhio.model_grid import ModelGrid, Axis, TrainingAxis
from occhio.distributions import PowerLawDigraph
from occhio.autoencoder import TiedLinearRelu
import occhio.visualization as ov

import networkx as nx
import plotly.graph_objects as go
import torch


# %%  ── config ───────────────────────────────────────────────────────────────
def create_model(params):
    DEVICE = "mps"
    gen = torch.Generator(DEVICE)
    gen.manual_seed(1)

    N_FEATURES = 20
    N_HIDDEN = 4

    dist = PowerLawDigraph(
        n_features=N_FEATURES,
        alpha=2,
        p_edge=0.20,
        p_active=params["p_active"],
        p_child=params["p_child"],
        generator=gen,
        device=DEVICE,
    )

    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
    return ToyModel(
        distribution=dist,
        ae=ae,
        generator=gen,
        importances=0.99 ** torch.arange(N_FEATURES),
        device=DEVICE,
    )


# %%
mg = ModelGrid(
    create_model,
    axes=[
        Axis(label="p_active", values=torch.logspace(0, -3, steps=24)),
        TrainingAxis(label="p_child", values=torch.linspace(0, 1, steps=11)),
    ],
)

# %%
dist = mg[0, 0].distribution  # ty:ignore
adj = dist.adjacency.cpu().numpy()  # adj[j, i] = True → edge j → i
G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

pos = nx.arf_layout(G, a=2.0, max_iter=15_000, seed=0)
in_deg = dict(G.in_degree())

annotations = [
    dict(
        ax=pos[u][0],
        ay=pos[u][1],
        x=pos[v][0],
        y=pos[v][1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor="#666",
        standoff=10,
        startstandoff=10,
    )
    for u, v in G.edges()
]

node_trace = go.Scatter(
    x=[pos[n][0] for n in G.nodes()],
    y=[pos[n][1] for n in G.nodes()],
    mode="markers+text",
    text=[str(n) for n in G.nodes()],
    textposition="top center",
    hovertext=[
        f"Node {n} | in: {in_deg[n]} | out: {G.out_degree(n)}" for n in G.nodes()
    ],
    hoverinfo="text",
    marker=dict(
        size=14,
        color=[in_deg[n] for n in G.nodes()],
        colorscale="Viridis",
        colorbar=dict(title="In-degree"),
        line=dict(width=1, color="black"),
    ),
)

fig = go.Figure(
    data=[node_trace],
    layout=go.Layout(
        title="PowerLawDigraph structure",
        showlegend=False,
        hovermode="closest",
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    ),
)
fig.show()

# %%
losses = mg.fit(25_000, track_losses=True, sample_every=50)


# %%
fig = ov.plot_geometry(mg)
fig.update_layout(height=600)
fig.show()

# %%
fig = ov.plot_representation(mg[:4])
fig.update_layout(height=600)
fig.show()

fig = ov.plot_representation(mg[-4:])
fig.update_layout(height=600)
fig.show()
# %%
ov.plot_feature_geometry(mg[:4])

# %%
ov.plot_feature_geometry_3d(mg[-1])

# %%
