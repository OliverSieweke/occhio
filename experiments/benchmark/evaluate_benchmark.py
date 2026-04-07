from occhio import benchmark
from occhio.benchmark.configs import BenchmarkDistributionName, matryoshka_targeted

benchmark.evaluate(
    saes=matryoshka_targeted,
    distributions=[BenchmarkDistributionName.HIERARCHICAL_PAIRS],
    training_samples=15_000_000,
    verbose=True,
    export_dir="experiments/benchmark/sweep-analysis/data/matryoshka_targeted",
    device="cuda"
)
