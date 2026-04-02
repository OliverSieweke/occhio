from typing import Callable, TypeGuard

import torch
from datasets.packaged_modules.generator import generator
from sae_lens import TrainingSAE
from tqdm.auto import tqdm

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.benchmark.configs import (
    BenchmarkDistributionName,
    BenchmarkSAEsInput,
    default_benchmark_saes,
)
from occhio.benchmark.utils import toy_model_from_benchmark
from occhio.model_grid import Axis

# [2026-03-27 | OliverSieweke] TODO: make this a script


def evaluate(
    distributions: list[BenchmarkDistributionName] = list(BenchmarkDistributionName),
    saes: BenchmarkSAEsInput = default_benchmark_saes(),
    training_samples: int = 10_000_000,
    device: str | None = None,
    verbose: bool = False,
):
    """
    Evaluate SAEs on toy model benchmarks.

    Args:
        saes: Either:
            - dict[str, TrainingSAE]: Same SAEs evaluated on all benchmarks
            - dict[BenchmarkName, dict[str, TrainingSAE]]: Different SAEs per benchmark
            - Callable[[ToyModel], dict[str, TrainingSAE]]: Factory function for all benchmarks
            - dict[BenchmarkName, Callable[[ToyModel], dict[str, TrainingSAE]]]: Factory function per benchmark
        distributions: Only used when saes is dict[str, TrainingSAE] or Callable. Ignored otherwise.
        ...
    """
    if is_per_benchmark_saes(saes):
        benchmark_list = list(saes.keys())
    else:
        benchmark_list = (
            distributions
            if distributions is not None
            else list(BenchmarkDistributionName)
        )

    grid = ModelGrid(
        lambda params: toy_model_from_benchmark(
            params["Benchmark"].value,
            device=device,
            # generator=torch.Generator(device=device),
        ),
        axes=[
            Axis("Benchmark", benchmark_list),
            # Axis("Seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ],
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
) -> TypeGuard[
    dict[BenchmarkDistributionName, Callable[[ToyModel], dict[str, TrainingSAE]]]
]:
    return not callable(saes) and isinstance(
        next(iter(saes.keys())), BenchmarkDistributionName
    )


def is_shared_saes(
    saes: BenchmarkSAEsInput,
) -> TypeGuard[dict[str, TrainingSAE] | Callable[[ToyModel], dict[str, TrainingSAE]]]:
    return callable(saes) or isinstance(next(iter(saes.keys())), str)
