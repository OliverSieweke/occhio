"""
Phase change in optimal configuration in a small model.
This file might take some time to run.
"""

from occhio.distributions.sparse import SparseUniform
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import einops
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def get_phase_color(interference, norm):
    gray = 200

    r = gray + (interference * 255 - gray) * norm
    g = gray + (0 - gray) * norm
    b = gray + ((1 - interference) * 255 - gray) * norm

    return r, g, b


EXPERIMENT_SIZE = 32
n_features = 2
n_hidden = 1

matrix = torch.zeros((EXPERIMENT_SIZE, EXPERIMENT_SIZE, 4))

importances = torch.tensor([0.9**i for i in range(n_features)])
gen = torch.Generator("cpu")
gen.manual_seed(7)

densities = torch.logspace(0, -2, EXPERIMENT_SIZE)
importances = torch.logspace(-1, 1, EXPERIMENT_SIZE)

for i, density in enumerate(densities):
    for j, importance in enumerate(importances):
        gen.manual_seed(7)
        tm = ToyModel(
            distribution=SparseUniform(n_features, density, generator=gen),
            ae=TiedLinearRelu(n_features, n_hidden, generator=gen),
            importances=importance ** torch.arange(n_features),
        )
        tm.fit(4_000)

        W = tm.ae.W.detach().numpy()  # ty:ignore
        b = tm.ae.b.detach().numpy()  # ty:ignore
        WTW = W.T @ W
        W_norms = np.linalg.norm(W, axis=0, keepdims=True)
        W_normalized = W / W_norms

        dot_prods = einops.einsum(W_normalized, W, "d i, d j -> i j")
        dot_prods_diag = dot_prods * (1 - np.eye(dot_prods.shape[0]))
        interference = einops.reduce(dot_prods_diag**2, "i j -> i", "sum")

        matrix[i, j, 0] = density
        matrix[i, j, 1] = importance
        matrix[i, j, 2] = np.clip(interference[1].item(), 0.0, 1)
        matrix[i, j, 3] = np.clip(W_norms[0, 1].item(), 0, 1)

phase_rgb = [
    [get_phase_color(matrix[i, j, 2], matrix[i, j, 3]) for j in range(matrix.shape[1])]
    for i in range(matrix.shape[0])
]

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.8, 0.2],
    subplot_titles=("Phase Change", "Colormap"),
)

# Phase Change
fig.add_trace(
    go.Image(
        z=phase_rgb,
        customdata=matrix,
        hovertemplate="Density: %{customdata[0]:.2f}<br>Importance: %{customdata[1]:.2f}<br>Norm: %{customdata[2]:.4f}<br>Interference: %{customdata[3]:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.update_xaxes(
    tickmode="array",
    tickvals=[0, (EXPERIMENT_SIZE - 1) / 2, EXPERIMENT_SIZE - 1],
    ticktext=["0.1x", "1x", "10x"],
    title=dict(text="<b>Importance</b>", font=dict(size=10)),
    row=1,
    col=1,
)
fig.update_yaxes(
    autorange="reversed",
    tickmode="array",
    tickvals=[0, (EXPERIMENT_SIZE - 1) / 2, EXPERIMENT_SIZE - 1],
    ticktext=["1", "0.1", "0.01"],
    title=dict(text="<b>Density</b>", font=dict(size=10)),
    row=1,
    col=1,
)

# Colormap
COLORMAP_SIZE = 100
interference_vals = np.linspace(0, 1, COLORMAP_SIZE)
norm_vals = np.linspace(1, 0, COLORMAP_SIZE)
interference_grid, norm_grid = np.meshgrid(interference_vals, norm_vals)

colormap_rgb = [
    [get_phase_color(interference_vals[j], norm_vals[i]) for j in range(COLORMAP_SIZE)]
    for i in range(COLORMAP_SIZE)
]

fig.add_trace(
    go.Image(
        z=colormap_rgb,
        customdata=np.stack([norm_grid, interference_grid], axis=-1),
        hovertemplate="Interference: %{customdata[1]:.2f}<br>Norm: %{customdata[0]:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(
    tickmode="array",
    tickvals=[0, 99],
    ticktext=["0", "1"],
    side="top",
    row=1,
    col=2,
    title=dict(text="<b>Interference</b>", font=dict(size=10), standoff=5),
)
fig.update_yaxes(
    tickmode="array",
    tickvals=[0, 99],
    ticktext=["1", "0"],
    side="right",
    row=1,
    col=2,
    title=dict(text="<b>Norm</b>", font=dict(size=10), standoff=5),
)
fig.write_image("phase_changes.png", scale=3)
