from occhio import benchmark
from occhio.benchmark.configs import BenchmarkDistributionName, matryoshka_targeted

benchmark.evaluate(
    saes=matryoshka_targeted,
    training_samples=10000,
    distributions=[BenchmarkDistributionName.HIERARCHICAL_PAIRS],
    verbose=True,
    n_loss_snapshots=300,
    export_dir="experiments/benchmark/sweep-analysis/data/matryoshka_targeted",
)
