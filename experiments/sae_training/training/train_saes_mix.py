# %%
from pathlib import Path

from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

from experiments.sae_training.utils import name_embedding_training, name_sae_evaluation
from occhio import ModelGrid

# %%
GRIDS_DIR = Path(__file__).parent.parent / "grids"
DEVICE = "cpu"

NAME = "uniform"
N_FEATURES = [200, 300, 500, 700]
N_HIDDEN = 64
# %%
# %%
grid = ModelGrid.load(
    GRIDS_DIR / name_embedding_training(NAME, n_hidden=N_HIDDEN, n_features=N_FEATURES)
)

grid = grid[:, 2:3]  # 500


# %%
# Sparsity-aware hyperparameter sweeps
l1_coefficient = 0.2  # for Standard
k_values = [2, 5, 12]  # for BatchTopK, Matryoshka
mp_max_iters = [3, 8, 12]  # for MatchingPursuit

for model in grid.models.ravel():
    n_features = model.ae.n_features
    d_sae = n_features * 2

    jumprelu_sae = JumpReLUTrainingSAE(
        JumpReLUTrainingSAEConfig(
            d_in=N_HIDDEN,
            d_sae=d_sae,
            device=DEVICE,
        )
    )
    # --- L1-based architectures ---
    standard_sae = StandardTrainingSAE(
        StandardTrainingSAEConfig(
            d_in=N_HIDDEN,
            d_sae=d_sae,
            l1_coefficient=l1_coefficient,
            device=DEVICE,
        )
    )
    model.train_saes(
        {f"Std_L1_{l1_coefficient}": standard_sae, "JpReLU": jumprelu_sae},
        training_samples=15_000_000,
        verbose=True,
    )

    # --- TopK-based architectures ---
    for k in k_values:
        batch_topk_sae = BatchTopKTrainingSAE(
            BatchTopKTrainingSAEConfig(
                d_in=N_HIDDEN,
                d_sae=d_sae,
                k=k,
                device=DEVICE,
            )
        )
        matryoshka_sae = MatryoshkaBatchTopKTrainingSAE(
            MatryoshkaBatchTopKTrainingSAEConfig(
                d_in=N_HIDDEN,
                d_sae=d_sae,
                k=k,
                matryoshka_widths=[d_sae // 8, d_sae // 4, d_sae // 2, d_sae],
                device=DEVICE,
            )
        )
        model.train_saes(
            {f"BTK_{k}": batch_topk_sae, f"Ma_{k}": matryoshka_sae},
            training_samples=15_000_000,
            verbose=True,
        )

    # --- Matching Pursuit ---
    for mi in mp_max_iters:
        mp_sae = MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=N_HIDDEN,
                d_sae=d_sae,
                max_iterations=mi,
                device=DEVICE,
            )
        )
        model.train_saes(
            {f"MP_{mi}": mp_sae},
            training_samples=15_000_000,
            verbose=True,
        )


# %%
grid.evaluate_saes(verbose=True)

# %%
# %%
grid.save(
    GRIDS_DIR
    / name_sae_evaluation(
        NAME,
        N_HIDDEN,
        n_features=N_FEATURES[2:3],
        sae_labels=grid[0, 0].saes.keys(),
    ),
)
# %%
