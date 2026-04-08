# %%
import torch
from occhio import ToyModel, ModelGrid
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import CorrelatedPairs
from occhio.model_grid import Axis
from sae_lens import (
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
)
from occhio.visualization_2.plots import SAEClassificationMetricsPlot

# %%
DEVICE = "cpu"
GENERATOR = torch.Generator(device=DEVICE)
# %%
N_FEATURES = 100
N_HIDDEN = 20
# %%
N_FEATURES = 400
N_HIDDEN = 30


# %%
def create_model(params):
    generator = torch.Generator(device=DEVICE).manual_seed(199)

    return ToyModel(
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE, generator=generator),
        distribution=CorrelatedPairs(
            N_FEATURES,
            density=params["Density"],
            correlation=params["Correlation"],
            device=DEVICE,
            generator=generator,
        ),
        importances=params["Relative Importance"] ** torch.arange(N_FEATURES),
        device=DEVICE,
    )


grid = ModelGrid(
    create_model,
    axes=[
        Axis(label="Density", values=torch.logspace(-1, 0, 1)),
        Axis(label="Relative Importance", values=[1, 0.99, 0.9]),
        Axis(label="Correlation", values=torch.linspace(0, 1, 1)),
    ],
)

grid.fit(100)
# %%
standard_sae_config = StandardTrainingSAEConfig(
    d_in=N_HIDDEN,
    d_sae=N_FEATURES,
    l1_coefficient=5e-2,
    device=DEVICE,
)

matryoshka_sae_config = MatryoshkaBatchTopKTrainingSAEConfig(
    d_in=N_HIDDEN,
    d_sae=N_FEATURES,
    k=50,  # number of active features
    matryoshka_widths=[100, 200, 400],  # nested levels, final must equal d_sae
    device=DEVICE,
)

standard_sae = StandardTrainingSAE(standard_sae_config)
matryoshka_sae = MatryoshkaBatchTopKTrainingSAE(matryoshka_sae_config)

grid.train_saes(
    {
        "Standard": standard_sae,
        "Matryoshka": matryoshka_sae,
    },
    training_samples=100,
    verbose=True,
)
# %%
grid.evaluate_saes(verbose=True)
# %%
plot = SAEClassificationMetricsPlot(group_by="metric")  # plots all SAEs
plot(grid, height=800)
# %%
