from enum import Enum, unique
from itertools import product
from typing import Callable

from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

from occhio import ToyModel
from occhio.toy_model import SAEEntry

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
    list[SAEEntry]
    | Callable[[ToyModel], list[SAEEntry]]
    | dict[BenchmarkDistributionName, list[SAEEntry]]
    | dict[BenchmarkDistributionName, Callable[[ToyModel], list[SAEEntry]]]
)

k_values = [2, 3, 4, 5, 6, 7]
width_configs = {
    "2-level": [324, 648],
    "3-level": [162, 324, 648],
    "4-level": [81, 162, 324, 648],
}

standard_total = [
    SAEEntry(
        sae=StandardTrainingSAE(
            StandardTrainingSAEConfig(d_in=100, d_sae=648, l1_coefficient=l1)
        ),
        type="Standard",
        params={
            "l1_coefficient": l1,
        },
    )
    for l1 in [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
]


batchtopk_total: BenchmarkSAEsInput = [
    SAEEntry(
        sae=BatchTopKTrainingSAE(
            BatchTopKTrainingSAEConfig(d_in=100, d_sae=648, k=k, decoder_init_norm=1)
        ),
        type="BatchTopK",
        params={
            "k": k,
        },
    )
    for k in [1, 2, 3, 4, 5, 6, 7]
]

matryoshka_total: BenchmarkSAEsInput = [
    SAEEntry(
        sae=MatryoshkaBatchTopKTrainingSAE(
            MatryoshkaBatchTopKTrainingSAEConfig(
                d_in=100,
                d_sae=648,
                matryoshka_widths=widths,
                use_matryoshka_aux_loss=True,
                k=k,
            )
        ),
        type="Matryoshka",
        params={
            "k": k,
            "widths": widths_name,
        },
    )
    for k, (widths_name, widths) in product(k_values, width_configs.items())
]

matching_pursuit_total = [
    SAEEntry(
        sae=MatchingPursuitTrainingSAE(
            MatchingPursuitTrainingSAEConfig(
                d_in=100, d_sae=648, max_iterations=max_iterations
            )
        ),
        type="MatchingPursuit",
        params={
            "max_iterations": max_iterations,
        },
    )
    for max_iterations in [1, 2, 3, 4, 5, 6]
]


# --------------------------------------------------------------------------------------
def default_benchmark_saes() -> BenchmarkSAEsInput:
    return {
        BenchmarkDistributionName.CORRELATED_PAIRS: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.4,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.4},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=2,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 2},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=3,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 3},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=2,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 2},
            ),
        ],
        BenchmarkDistributionName.HIERARCHICAL_PAIRS: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.5,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.5},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=1,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 1},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=2,  # check for higher
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 2},
            ),
        ],
        BenchmarkDistributionName.DAG_RANDOM_WALK: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.6,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.6},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=1,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 1},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=5,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 5},
            ),
        ],
        BenchmarkDistributionName.POWER_LAW_DIGRAPH: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.9,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.9},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=2,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 2},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=2,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 2},
            ),
        ],
        BenchmarkDistributionName.SIMPLICIAL_COMPLEX: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.05,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.05},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=2,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 2},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=2,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 2},
            ),
        ],
        BenchmarkDistributionName.SPARSE_UNIFORM: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.6,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.6},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=1,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 1},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=3,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 3},
            ),
        ],
        BenchmarkDistributionName.SPHERICAL: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.05,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.05},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=2,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 2},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=2,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 2},
            ),
        ],
        BenchmarkDistributionName.TORUS: [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        l1_coefficient=0.4,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": 0.4},
            ),
            SAEEntry(
                sae=MatryoshkaBatchTopKTrainingSAE(
                    MatryoshkaBatchTopKTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        matryoshka_widths=[81, 324, 648],
                        k=1,
                        use_matryoshka_aux_loss=True,
                    )
                ),
                type="Matryoshka",
                params={"k": 1},
            ),
            SAEEntry(
                sae=MatchingPursuitTrainingSAE(
                    MatchingPursuitTrainingSAEConfig(
                        d_in=100,
                        d_sae=648,
                        max_iterations=5,
                    )
                ),
                type="MatchingPursuit",
                params={"max_iterations": 5},
            ),
        ],
    }
