from enum import Enum, unique
from typing import Callable

from sae_lens import (
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TrainingSAE,
)

from occhio import ToyModel

# [2026-03-25 | OliverSieweke] TODO: Only keep one for both here
OCCHIO_HF_DISTRIBUTIONS_REPO = "kaushikreddyxyz/occhio-distributions"
OCCHIO_HF_MODELS_REPO = "kaushikreddyxyz/occhio-models"


@unique
class BenchmarkDistributionName(str, Enum):
    CORRELATED_PAIRS = "correlated_pairs"
    HIERARCHICAL_PAIRS = "hierarchical_pairs"
    DAG_RANDOM_WALK = "dag_random_walk"
    POWER_LAW_DIGRAPH = "power_law_digraph"
    SIMPLICIAL_COMPLEX = "simplicial_complex"
    SPARSE_UNIFORM = "sparse_uniform"
    SPHERICAL = "spherical"
    TORUS = "torus"


BenchmarkSAEsInput = (
    dict[str, TrainingSAE]
    | Callable[[ToyModel], dict[str, TrainingSAE]]
    | dict[BenchmarkDistributionName, dict[str, TrainingSAE]]
    | dict[BenchmarkDistributionName, Callable[[ToyModel], dict[str, TrainingSAE]]]
)


def default_benchmark_saes(device="cpu") -> BenchmarkSAEsInput:
    return {
        BenchmarkDistributionName.CORRELATED_PAIRS: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.4,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=2,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=2,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.HIERARCHICAL_PAIRS: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.5,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=1,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=2,  # check for higher
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.DAG_RANDOM_WALK: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.6,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=1,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=5,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.POWER_LAW_DIGRAPH: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.9,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=2,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=2,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.SIMPLICIAL_COMPLEX: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.05,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=2,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=2,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.SPARSE_UNIFORM: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.6,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=1,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=3,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.SPHERICAL: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.05,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=2,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=2,
                    device=device,
                )
            ),
        },
        BenchmarkDistributionName.TORUS: {
            "Standard": StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    l1_coefficient=0.4,
                    device=device,
                )
            ),
            "Matryoshka": MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    matryoshka_widths=[
                        81,
                        324,
                        648,
                    ],
                    k=1,
                    use_matryoshka_aux_loss=True,
                    device=device,
                )
            ),
            "MatchingPursuit": MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=100,
                    d_sae=648,
                    max_iterations=5,
                    device=device,
                )
            ),
        },
    }
