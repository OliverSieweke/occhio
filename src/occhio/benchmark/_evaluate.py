"""SAE benchmark evaluation and export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeGuard

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from occhio.benchmark.configs import (
    BenchmarkDistributionName,
    BenchmarkSAEsInput,
    default_benchmark_saes,
)
from occhio.benchmark.utils import toy_model_from_benchmark
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import SAEEntry, ToyModel

# [2026-03-27 | OliverSieweke] TODO: make this a script


@dataclass
class EvaluationResult:
    """Result of a benchmark evaluation run."""

    grid: ModelGrid
    df: pd.DataFrame
    losses_df: pd.DataFrame | None = None


def evaluate(
    distributions: list[BenchmarkDistributionName] = list(BenchmarkDistributionName),
    saes: BenchmarkSAEsInput = default_benchmark_saes(),
    training_samples: int = 10_000_000,
    device: str | None = None,
    verbose: bool = False,
    export_dir: str | Path | None = None,
    n_loss_snapshots: int | None = None,
    n_seeds: int | None = None,
) -> EvaluationResult:
    """Evaluate SAEs on toy model benchmarks.

    Args:
        distributions: Which benchmark distributions to evaluate on.
        saes: Either:
            - dict[str, TrainingSAE]: Same SAEs evaluated on all benchmarks
            - dict[BenchmarkName, dict[str, TrainingSAE]]: Different SAEs per benchmark
            - Callable[[ToyModel], dict[str, TrainingSAE]]: Factory function for all benchmarks
            - dict[BenchmarkName, Callable[[ToyModel], dict[str, TrainingSAE]]]:
              Factory function per benchmark
        training_samples: Number of training samples per SAE.
        device: Device for training (e.g. "cpu", "mps", "cuda").
        verbose: Whether to show progress bars.
        export_dir: If set, export results to this directory. The directory must not
            already exist and be non-empty.
        n_loss_snapshots: If set, record the overall loss at this many evenly-spaced
            snapshots per SAE and include in the export. None (default) disables loss tracking.
        n_seeds: If set, sweep over this many seeds (0..n_seeds-1). None (default) runs
            a single model with no seed axis.

    Returns:
        An EvaluationResult with the trained grid and a tidy DataFrame of SAE metrics.
    """
    # [2026-04-02 | OliverSieweke] TODO: dryify
    if export_dir is not None:
        export_path = Path(export_dir)
        if export_path.exists() and any(export_path.iterdir()):
            raise FileExistsError(
                f"Export directory '{export_dir}' already exists and is not empty."
            )

    if is_per_benchmark_saes(saes):
        benchmark_list = list(saes.keys())
    else:
        benchmark_list = (
            distributions
            if distributions is not None
            else list(BenchmarkDistributionName)
        )

    axes = [Axis("benchmark", benchmark_list)]
    if n_seeds is not None:
        axes.append(Axis("seed", list(range(n_seeds))))

    grid = ModelGrid(
        lambda params: toy_model_from_benchmark(
            params["benchmark"].value,
            device=device,
            generator=torch.Generator(device=device),
        ),
        axes=axes,
    )

    if is_per_benchmark_saes(saes):
        for benchmark, benchmark_saes in tqdm(saes.items(), desc="Distribution"):
            grid[benchmark_list.index(benchmark)].train_saes(
                benchmark_saes,
                training_samples=training_samples,
                verbose=verbose,
                n_loss_snapshots=n_loss_snapshots,
            )
    elif is_shared_saes(saes):
        grid.train_saes(
            saes,
            training_samples=training_samples,
            verbose=verbose,
            n_loss_snapshots=n_loss_snapshots,
        )

    grid.evaluate_saes(verbose=verbose)

    df = grid.sae_results_to_dataframe()

    losses_df = _build_losses_dataframe(grid) if n_loss_snapshots is not None else None

    if export_dir is not None:
        _export_results(
            grid,
            df,
            export_dir=Path(export_dir),
            training_samples=training_samples,
            device=device,
            per_benchmark=is_per_benchmark_saes(saes),
            losses_df=losses_df,
        )

    return EvaluationResult(grid=grid, df=df, losses_df=losses_df)


def _build_losses_dataframe(grid: ModelGrid) -> pd.DataFrame | None:
    """Build a tidy DataFrame of SAE training losses from a grid."""
    rows = []
    for idx in np.ndindex(*grid.shape):
        model = grid.models[idx]
        axis_values = {}
        for i, axis in enumerate(grid.axes):
            value = axis.values[idx[i]]
            axis_values[axis.label] = (
                value.name
                if hasattr(value, "name") and isinstance(value.name, str)
                else str(value)
            )
        for label, record in model.saes.items():
            if record.losses is not None:
                row_base: dict[str, Any] = {**axis_values, "sae": label}
                if record.sae_type is not None:
                    row_base["sae_type"] = record.sae_type
                if record.params:
                    row_base.update(record.params)
                for step, loss in record.losses:
                    rows.append({**row_base, "step": step, "loss": loss})
    return pd.DataFrame(rows) if rows else None


def _export_results(
    grid: ModelGrid,
    df: pd.DataFrame,
    export_dir: Path,
    training_samples: int,
    device: str | None,
    per_benchmark: bool,
    losses_df: pd.DataFrame | None = None,
) -> None:
    """Write grid, results, and run info to an export directory."""
    if export_dir.exists() and any(export_dir.iterdir()):
        raise FileExistsError(
            f"Export directory '{export_dir}' already exists and is not empty."
        )

    export_dir.mkdir(parents=True, exist_ok=True)

    grid.save(export_dir / "grid.pkl")
    df.to_parquet(export_dir / "results.parquet")
    df.reset_index().to_csv(export_dir / "results.csv", index=False)

    if losses_df is not None:
        losses_df.to_parquet(export_dir / "losses.parquet")
        losses_df.to_csv(export_dir / "losses.csv", index=False)

    nested = {
        benchmark: {
            str(k) if isinstance(k, tuple) else k: v
            for k, v in group.droplevel("benchmark").to_dict(orient="index").items()
        }
        for benchmark, group in df.groupby(level="benchmark")
    }
    (export_dir / "results.json").write_text(json.dumps(nested, indent=2))

    run_info = _build_run_info(grid, training_samples, device, per_benchmark)
    (export_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, default=str)
    )


# [2026-04-02 | OliverSieweke] TODO: check this
def _build_run_info(
    grid: ModelGrid,
    training_samples: int,
    device: str | None,
    per_benchmark: bool,
) -> dict[str, Any]:
    """Build a human-readable run metadata dict."""
    run_info: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "training_samples": training_samples,
        "device": device,
        "distributions": [],
        "saes": {},
    }

    benchmark_axis = grid.axes[0]
    distribution_names = [
        v.name if hasattr(v, "name") and isinstance(v.name, str) else str(v)
        for v in benchmark_axis.values
    ]
    run_info["distributions"] = distribution_names

    if per_benchmark:
        # Different SAEs per distribution — nest configs under distribution name
        saes_by_dist: dict[str, dict[str, Any]] = {}
        for i, dist_name in enumerate(distribution_names):
            model: ToyModel = grid.models.ravel()[i]
            saes_by_dist[dist_name] = {
                label: record.sae.cfg.to_dict() for label, record in model.saes.items()
            }
        run_info["saes"] = saes_by_dist
    else:
        # Shared SAEs — extract configs from first model
        first_model: ToyModel = grid.models.ravel()[0]
        run_info["saes"] = {
            label: record.sae.cfg.to_dict()
            for label, record in first_model.saes.items()
        }

    return run_info


def is_per_benchmark_saes(
    saes: BenchmarkSAEsInput,
) -> TypeGuard[
    dict[BenchmarkDistributionName, list[SAEEntry]]
    | dict[BenchmarkDistributionName, Callable[[ToyModel], list[SAEEntry]]]
]:
    return (
        not callable(saes)
        and not isinstance(saes, list)
        and isinstance(next(iter(saes.keys())), BenchmarkDistributionName)
    )


def is_shared_saes(
    saes: BenchmarkSAEsInput,
) -> TypeGuard[list[SAEEntry] | Callable[[ToyModel], list[SAEEntry]]]:
    return callable(saes) or isinstance(saes, list)
