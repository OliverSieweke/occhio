# %%
"""
I investigate the Random Walk to Root distribution,
to see if it exhibits ocllapse.
I also look at the AE representation of these distributions.
"""

# %%
from occhio.distributions.dag import DAGRandomWalkToRoot
from occhio.sae import SAESimple, TopKIgnoreSAE
from occhio.autoencoder import MLPEncoder, TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
from occhio.model_grid import ModelGrid, Axis
import occhio.visualization as ov
from importlib import reload

# %%


torch.set_printoptions(3, sci_mode=False)


N_FEAT = 50
N_EMB = 10


# %%
def create_model(params):
    DEVICE = "cpu"
    gen = torch.Generator(DEVICE)
    gen.manual_seed(3)

    imp = params["importance"]
    beta = params["beta"]

    importances = imp ** torch.arange(N_FEAT)

    dist = DAGRandomWalkToRoot(
        n_features=N_FEAT,
        p_edge=3 / N_FEAT,
        beta=beta,
        generator=gen,
        shrinking=True,
        device=DEVICE,
    )

    ae = TiedLinearRelu(N_FEAT, N_EMB, generator=gen, device=DEVICE)
    return ToyModel(dist, ae, generator=gen, importances=importances, device=DEVICE)


# %%
mg = ModelGrid(
    create_model,
    axes=[
        Axis(label="beta", values=1 - torch.logspace(-0.5, -2, steps=16)),
        Axis(label="importance", values=torch.linspace(0, 1, steps=16)),
    ],
)


# %%
mg[0, 0].distribution.print_graph()
mg[0, 0].distribution.print_sources_and_sinks()
mg[0, 0].distribution.print_connected_components()

# %%
mg.fit(12_000)

# %%
ov.plot_geometry(mg[:, -1])  # ty:ignore[invalid-argument-type]


# %%
ov.plot_phase_change(mg, tracked_feature=40)


# %%
