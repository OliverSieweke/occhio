from typing import Callable, TypeGuard

from sae_lens import TrainingSAE
from tqdm.auto import tqdm

from occhio import ModelGrid, ToyModel
from occhio.benchmark.configs import (
    BenchmarkName,
    BenchmarkSAEsInput,
    DEFAULT_BENCHMARK_SAEs,
)
from occhio.benchmark.utils import toy_model_from_benchmark
from occhio.model_grid import Axis

# [2026-03-27 | OliverSieweke] TODO: make this a script


def evaluate(
    benchmark_distributions: list[BenchmarkName] = list(BenchmarkName),
    saes: BenchmarkSAEsInput = DEFAULT_BENCHMARK_SAEs,
    training_samples: int = 10_000_000,
    device: str | None = None,
    verbose: bool = True,
):
    """
    Evaluate SAEs on toy model benchmarks.

    Args:
        saes: Either:
            - dict[str, TrainingSAE]: Same SAEs evaluated on all benchmarks
            - dict[BenchmarkName, dict[str, TrainingSAE]]: Different SAEs per benchmark
            - Callable[[ToyModel], dict[str, TrainingSAE]]: Factory function for all benchmarks
            - dict[BenchmarkName, Callable[[ToyModel], dict[str, TrainingSAE]]]: Factory function per benchmark
        benchmark_distributions: Only used when saes is dict[str, TrainingSAE] or Callable. Ignored otherwise.
        ...
    """
    if is_per_benchmark_saes(saes):
        benchmark_list = list(saes.keys())
    else:
        benchmark_list = (
            benchmark_distributions
            if benchmark_distributions is not None
            else list(BenchmarkName)
        )

    grid = ModelGrid(
        lambda params: toy_model_from_benchmark(
            params["Benchmark"].value, device=device
        ),
        axes=[Axis("Benchmark", benchmark_list)],
    )

    if is_per_benchmark_saes(saes):
        for benchmark, benchmark_saes in tqdm(saes.items(), desc="Distribution"):
            grid[benchmark_list.index(benchmark)].train_saes(
                benchmark_saes, training_samples=training_samples, verbose=verbose
            )
    elif is_shared_saes(saes):
        grid.train_saes(saes, training_samples=training_samples, verbose=verbose)

    grid.evaluate_saes(verbose=verbose)

    return grid


def is_per_benchmark_saes(
    saes: BenchmarkSAEsInput,
) -> TypeGuard[dict[BenchmarkName, Callable[[ToyModel], dict[str, TrainingSAE]]]]:
    return not callable(saes) and isinstance(next(iter(saes.keys())), BenchmarkName)


def is_shared_saes(
    saes: BenchmarkSAEsInput,
) -> TypeGuard[dict[str, TrainingSAE] | Callable[[ToyModel], dict[str, TrainingSAE]]]:
    return callable(saes) or isinstance(next(iter(saes.keys())), str)
