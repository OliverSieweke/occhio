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
class BenchmarkName(str, Enum):
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
    | dict[BenchmarkName, dict[str, TrainingSAE]]
    | dict[BenchmarkName, Callable[[ToyModel], dict[str, TrainingSAE]]]
)

DEFAULT_BENCHMARK_SAEs: BenchmarkSAEsInput = {
    BenchmarkName.CORRELATED_PAIRS: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.4,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=2,
            )
        ),
    },
    BenchmarkName.HIERARCHICAL_PAIRS: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.5,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=2,  # check for higher
            )
        ),
    },
    BenchmarkName.DAG_RANDOM_WALK: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.6,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=5,
            )
        ),
    },
    BenchmarkName.POWER_LAW_DIGRAPH: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.9,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=2,
            )
        ),
    },
    BenchmarkName.SIMPLICIAL_COMPLEX: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.05,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=2,
            )
        ),
    },
    BenchmarkName.SPARSE_UNIFORM: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.6,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=3,
            )
        ),
    },
    BenchmarkName.SPHERICAL: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.05,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=2,
            )
        ),
    },
    BenchmarkName.TORUS: {
        "Standard": StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                l1_coefficient=0.4,
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
            )
        ),
        "MatchingPursuit": MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                max_iterations=5,
            )
        ),
    },
}
