# %%
from pathlib import Path

import numpy as np
import torch

from experiments.sae_training.utils import name_embedding_training
from occhio import AutoencoderType, ModelGrid, ToyModel
from occhio.autoencoder import SynthAE, TiedLinearRelu
from occhio.distributions import (
    SparseUniform,
)
from occhio.model_grid import Axis

# %%
DEVICE = "mps"
N_HIDDEN = 64
N_FEATURES = [200, 300, 500, 700]


# %%
def create_model(params):
    SEED = 199
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    high = 0.2
    low = 0.5 / int(params["Features"])
    alpha = np.log(high / low) / np.log(int(params["Features"]))
    firing_probs = [high / (i + 1) ** alpha for i in range(int(params["Features"]))]

    match params["Autoencoder"]:
        case AutoencoderType.TiedLinearRelu:
            ae = TiedLinearRelu(
                int(params["Features"]), N_HIDDEN, device=DEVICE, generator=generator
            )
        case AutoencoderType.SynthAE:
            ae = SynthAE(
                int(params["Features"]),
                N_HIDDEN,
                orthogonalize=True,
                device=DEVICE,
                generator=generator,
            )

    return ToyModel(
        ae=ae,
        distribution=SparseUniform(
            n_features=int(params["Features"]),
            p_active=firing_probs,
            generator=generator,
        ),
        device=DEVICE,
    )


grid = ModelGrid(
    create_model,
    axes=[
        Axis(
            label="Autoencoder",
            values=[AutoencoderType.SynthAE, AutoencoderType.TiedLinearRelu],
        ),
        Axis(
            label="Features",
            values=N_FEATURES,
        ),
    ],
)
# %%
grid.fit(batch_size=2048, n_epochs=15000, snapshot_interval=500)
# %%

GRIDS_DIR = Path(__file__).parent.parent / "grids"
grid.save(
    GRIDS_DIR
    / name_embedding_training("uniform", n_hidden=N_HIDDEN, n_features=N_FEATURES)
)
