# %%
from occhio import AutoEncoderBase
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
from occhio.visualization import plot_dynamic_scatter, plot_phase_change_multi
from occhio.model_grid import ModelGrid

import torch
import plotly.express as px

# %%
DEVICE = "cpu"
gen = torch.Generator(DEVICE)
gen.manual_seed(2)

n_features = 5

importances = 0.95 ** torch.arange(n_features)

dist = SparseUniform(n_features, 0.2, generator=gen, device=DEVICE)
ae = TiedLinearRelu(n_features, 2, generator=gen, device=DEVICE)
tm = ToyModel(
    distribution=dist,
    ae=ae,
    generator=gen,
    device=DEVICE,
)

# %%
tm.fit(15_000, 256)


# %%
emb = tm.W.detach().numpy()
fig = px.scatter(x=emb[0], y=emb[1])
fig.update_traces(marker=dict(size=10))
fig.show()
# %%
