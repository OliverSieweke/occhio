# %%
from pathlib import Path

import torch

from occhio import AutoencoderType, ModelGrid, ToyModel
from occhio.autoencoder import SynthAE, TiedLinearRelu
from occhio.distributions import (
    HierarchyNode,
    SyntheticDataConfig,
    SyntheticDataModel,
)
from occhio.model_grid import Axis

from ...utils import name_embedding_training

# %%
DEVICE = "cpu"
# %%
N_HIDDEN = 64
N_FEATURES = [350, 400]


# %%
def build_hierarchy(
    start_idx: int, n_roots: int, branching: int
) -> tuple[list[HierarchyNode], int]:
    """Build a forest of trees with mutual exclusion and parent-scaled magnitudes."""
    idx = start_idx
    roots = []
    for _ in range(n_roots):
        root_idx = idx
        idx += 1
        children = []
        for _ in range(branching):
            children.append(HierarchyNode(feature_idx=idx))
            idx += 1
        roots.append(
            HierarchyNode(
                feature_idx=root_idx,
                children=children,
                mutually_exclusive_children=True,
                parent_scaled=True,
            )
        )
    return roots, idx


# %%
def create_model(params):
    SEED = 199
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    hierarchy_roots, _ = build_hierarchy(start_idx=0, n_roots=16, branching=5)

    config = SyntheticDataConfig(
        n_features=int(params["Features"]),
        # Firing probabilities
        firing_prob_distribution="zipfian",
        p_max=0.3,
        p_min=0.5 / int(params["Features"]),
        alpha=0.7,
        # Magnitudes — linear mean, folded-normal stdev
        mean_distribution="linear",
        mean_high=3.0,
        mean_low=1.0,
        std_distribution="folded_normal",
        folded_normal_mu=0.5,
        folded_normal_sigma=0.5,
        # Correlation
        correlation_rank=4,
        correlation_scale=0.1,
        # Hierarchy
        hierarchy=hierarchy_roots,
        compensate_probabilities=True,
        # Runtime
        device=DEVICE,
    )

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
        distribution=SyntheticDataModel(config, seed=SEED),
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
    / name_embedding_training(
        "synth_sae_bench", n_hidden=N_HIDDEN, n_features=N_FEATURES
    )
)
