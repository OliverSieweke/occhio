from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)

from occhio import SAEEntry, benchmark
from occhio.benchmark.configs import (
    BenchmarkSAEsInput,
)
from occhio.sae_lens_adapter.standard_sae_autotuned import (
    StandardTrainingSAEAutotuned,
    StandardTrainingSAEConfigAutotuned,
)

D_IN = 200
D_SAE = 648
L0_TARGETS = [1, 2, 3, 4, 5, 6, 7]
DEVICE = "cpu"

saes: BenchmarkSAEsInput = [
    sae_entry
    for l0_target in L0_TARGETS
    for sae_entry in [
        SAEEntry(
            sae=StandardTrainingSAEAutotuned(
                StandardTrainingSAEConfigAutotuned(
                    d_in=D_IN,
                    d_sae=D_SAE,
                    l1_coefficient=0.5,
                    autotune_target_l0=float(l0_target),
                    device=DEVICE,
                )
            ),
            type="Standard",
            params={"l0_target": l0_target},
        ),
        SAEEntry(
            sae=BatchTopKTrainingSAE(
                BatchTopKTrainingSAEConfig(
                    d_in=D_IN, d_sae=D_SAE, k=l0_target, device=DEVICE
                )
            ),
            type="BatchTopK",
            params={
                "k": l0_target,
            },
        ),
        SAEEntry(
            sae=MatryoshkaBatchTopKTrainingSAE(
                MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=D_IN,
                    d_sae=D_SAE,
                    matryoshka_widths=[324, 648],
                    k=l0_target,
                    device=DEVICE,
                )
            ),
            type="Matryoshka",
            params={
                "k": l0_target,
                "widths": "2-level",
            },
        ),
        SAEEntry(
            sae=MatchingPursuitTrainingSAE(
                MatchingPursuitTrainingSAEConfig(
                    d_in=D_IN,
                    d_sae=D_SAE,
                    max_iterations=l0_target,
                    decoder_init_norm=1,
                    device=DEVICE,
                )
            ),
            type="MatchingPursuit",
            params={
                "max_iterations": l0_target,
            },
        ),
    ]
]

benchmark.evaluate(
    saes=saes,
    training_samples=60_000_000,
    verbose=True,
    export_dir="experiments/benchmark/sweep-analysis/data/hidden_200",
    device=DEVICE,
    n_seeds=1,
)
