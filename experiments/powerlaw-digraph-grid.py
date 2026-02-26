# %%
"""
PowerLawDigraph – toy model experiment.

We train a TiedLinearRelu autoencoder on a power-law digraph distribution and
inspect how the learned geometry relates to graph structure (in-degree, cascade
activation rates, etc.).
"""

# %%
from occhio import ToyModel
from occhio.model_grid import ModelGrid, Axis
from occhio.distributions import PowerLawDigraph
from occhio.autoencoder import TiedLinearRelu
import occhio.visualization as ov

import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# %%  ── config ───────────────────────────────────────────────────────────────
def create_model(params):
    DEVICE = "cpu"
    gen = torch.Generator(DEVICE)
    gen.manual_seed(1)

    N_FEATURES = 200
    N_HIDDEN = 16

    dist = PowerLawDigraph(
        n_features=N_FEATURES,
        alpha=1.5,
        p_edge=0.10,
        p_active=params["p_active"],
        p_child=0.2,
        generator=gen,
        device=DEVICE,
    )

    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
    return ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)


# %%
mg = ModelGrid(
    create_model,
    axes=[
        Axis(label="p_active", values=torch.logspace(-0.2, -2.5, steps=20)),
    ],
)

# %%
mg.fit(10_000)

# %%
ov.plot_geometry(mg)

# %%
torch.logspace(-0.1, -3, steps=10)

# %%
