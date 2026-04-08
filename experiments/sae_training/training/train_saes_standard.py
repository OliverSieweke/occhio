# %%
from pathlib import Path

from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig
from tqdm.auto import tqdm

from experiments.sae_training.utils import name_embedding_training, name_sae_evaluation
from occhio import ModelGrid

GRIDS_DIR = Path(__file__).parent.parent / "grids"
# %%
DEVICE = "mps"
NAME = "synth_sae_bench"
N_HIDDEN = 64
N_FEATURES = [200, 250, 300]
# %%
grid = ModelGrid.load(
    GRIDS_DIR / name_embedding_training(NAME, n_hidden=N_HIDDEN, n_features=N_FEATURES)
)

grid = grid[:, 2:3]  # 300

# %%
l1_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sae_labels = ["Standard"]

for model in tqdm(grid.models.ravel()):
    saes = {}
    for l1 in l1_coefficients:
        config = StandardTrainingSAEConfig(
            d_in=N_HIDDEN,
            d_sae=model.ae.n_features * 2,
            l1_coefficient=l1,
            device=DEVICE,
        )
        saes[f"Standard_L1_{l1}"] = StandardTrainingSAE(config)

    model.train_saes(saes, training_samples=15_000_000, verbose=True)

# %%
grid.evaluate_saes(verbose=True)

# %%
grid.save(
    GRIDS_DIR
    / name_sae_evaluation(
        NAME,
        N_HIDDEN,
        n_features=[300],
        sae_labels=sae_labels,
        l1_coefficients=l1_coefficients,
    ),
)
