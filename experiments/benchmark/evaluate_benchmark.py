from itertools import product

from sae_lens import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)

from occhio import SAEEntry, benchmark
from occhio.benchmark.configs import (
    BenchmarkDistributionName,
    BenchmarkSAEsInput,
)

k_values = [1, 2, 3, 4, 5]
width_configs = {
    "2-level": [324, 648],
    "3-level": [162, 324, 648],
    "4-level": [81, 162, 324, 648],
}

saes: BenchmarkSAEsInput = [
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

benchmark.evaluate(
    saes=saes,
    distributions=[BenchmarkDistributionName.HIERARCHICAL_PAIRS],
    training_samples=15_000_000,
    verbose=True,
    export_dir="experiments/benchmark/sweep-analysis/data/matryoshka_hierarchical_pairs",
    device="cpu",
    n_seeds=1,
)
