# %%
# ABOUTME: Experiment: ModelGrid over sparsity p with TiedLinearRelu on SparseUniform.
# ABOUTME: The axis is `p`; each feature i has p_active = p / (i+1).

from occhio import ToyModel
from occhio.model_grid import ModelGrid, Axis
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu
import occhio.visualization as ov

import torch

# %% ── config ─────────────────────────────────────────────────────────────────
DEVICE = "mps"
N_FEATURES = 12
N_HIDDEN = 2


def create_model(params):
    gen = torch.Generator(DEVICE)
    gen.manual_seed(8)

    p = params["p"]
    p_active = [p / (i + 1) for i in range(N_FEATURES)]

    dist = SparseUniform(N_FEATURES, p_active, generator=gen, device=DEVICE)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
    return ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)


# %% ── build grid ─────────────────────────────────────────────────────────────
mg = ModelGrid(
    create_model,
    axes=[
        Axis(label="p", values=torch.logspace(0, -2, steps=24)),
    ],
)

# %% ── train ──────────────────────────────────────────────────────────────────
mg.fit(20_000, batch_size=128)

# %% ── visualize ──────────────────────────────────────────────────────────────
fig = ov.plot_geometry(mg)
fig.update_layout(height=600)
fig.show()
# %%

fig = ov.plot_representation(mg[:3])
fig.update_layout(height=600)
fig.show()

# %%
fig = ov.plot_representation(mg[-3:])
fig.update_layout(height=600)
fig.show()

# %%
fig = ov.plot_embedding(mg[-3:])
fig.update_layout(height=600)
fig.show()

# %%
