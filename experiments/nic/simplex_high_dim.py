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

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(42)

N_FEAT = 100
N_HIDDEN = 16
FACE_DIM = 3
N_FACES = int(2 * N_FEAT)

random.seed(42)
faces = list(
    {
        tuple(sorted(random.sample(range(N_FEAT), FACE_DIM + 1)))
        for _ in range(N_FACES * 3)
    }
)[:N_FACES]

p_active = 1 / (N_FEAT)

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

# %%
losses, _ = tm.fit(30_000, 256, verbose=True)
px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training loss").show()

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

# %% --- Plot 1: Shared faces vs interference ---
# For each pair of features, plot the number of shared faces against interference
rows_i, rows_j = torch.triu_indices(N_FEAT, N_FEAT, offset=1)
pair_shared = shared_faces[rows_i, rows_j].numpy()
pair_interference = interferences[rows_i, rows_j].numpy()

fig = px.scatter(
    x=pair_shared,
    y=pair_interference,
    labels={"x": "Shared faces", "y": "Interference"},
    title="Shared faces between feature pairs vs Interference",
    opacity=0.3,
)
fig.show()

# %% --- Plot 2: Facet count vs feature dimensionality ---
fig = px.scatter(
    x=facet_count_per_feat.numpy(),
    y=feature_dims.numpy(),
    labels={"x": "Number of faces containing vertex", "y": "Feature dimensionality"},
    title="Facet membership count vs Feature dimensionality",
    hover_name=[f"v{i}" for i in range(N_FEAT)],
    trendline="ols",
)
fig.show()


# %% --- Plot 3: Neighbor count vs total interference ---
fig = px.scatter(
    x=neighbor_count_per_feat.numpy(),
    y=total_interference.numpy(),
    labels={
        "x": "Number of unique neighbors (shared-face adjacency)",
        "y": "Total interference (sum of squared off-diagonal)",
    },
    title="Graph degree (neighbor count) vs Total feature interference",
    hover_name=[f"v{i}" for i in range(N_FEAT)],
    trendline="ols",
)
fig.show()

# %%
