"""Autoencoder F1 baseline: threshold-sweep on AE reconstruction vs ground truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from occhio.benchmark.configs import BenchmarkDistributionName
from occhio.benchmark.utils import toy_model_from_benchmark
from occhio.toy_model import ToyModel

N_TEST_DEFAULT = 100_000
N_THRESHOLDS_DEFAULT = 101


def compute_ae_baseline(
    tm: ToyModel,
    n_test: int = N_TEST_DEFAULT,
    n_thresholds: int = N_THRESHOLDS_DEFAULT,
) -> dict[str, float]:
    """Compute the best F1 achievable by the autoencoder alone (no SAE).

    Sweeps thresholds on ``ae.decode(ae.encode(x))`` against ``x > 0``
    and returns the best macro/micro F1 at the optimal threshold.

    Args:
        tm: A trained ToyModel.
        n_test: Number of test samples.
        n_thresholds: Number of thresholds to sweep in [0, 1].

    Returns:
        Dict with keys: threshold, f1_macro, f1_micro.
    """
    device = tm.device

    with torch.no_grad():
        raw = tm.distribution.sample(n_test)
        test_x = (raw[0] if isinstance(raw, tuple) else raw).to(device)
        test_xhat = tm.ae.decode(tm.ae.encode(test_x))

    # Restrict to the first half of features (n_features // 2) to match the SAE
    # latent space dimensionality, making the comparison fair.
    n_half = tm.ae.n_features // 2
    threshold, f1_macro, f1_micro = _best_threshold_f1(
        test_x[:, :n_half], test_xhat[:, :n_half], n_thresholds=n_thresholds
    )
    return {"threshold": threshold, "f1_macro": f1_macro, "f1_micro": f1_micro}


def benchmark_ae_baselines(
    distributions: list[BenchmarkDistributionName] | None = None,
    device: str | None = None,
    n_test: int = N_TEST_DEFAULT,
    n_thresholds: int = N_THRESHOLDS_DEFAULT,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute AE F1 baselines for all (or selected) benchmark distributions.

    Args:
        distributions: Which benchmarks to evaluate. Defaults to all.
        device: Device for computation.
        n_test: Number of test samples per distribution.
        n_thresholds: Number of thresholds to sweep.
        cache_path: If set, load results from this file when it exists,
            or save computed results there. Supports ``.parquet`` and ``.csv``.

    Returns:
        DataFrame indexed by benchmark name with columns:
        threshold, f1_macro, f1_micro.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if cache_path.suffix == ".parquet":
                return pd.read_parquet(cache_path).set_index("benchmark")
            return pd.read_csv(cache_path).set_index("benchmark")

    if distributions is None:
        distributions = list(BenchmarkDistributionName)

    rows: list[dict[str, object]] = []
    for dist_name in distributions:
        tm = toy_model_from_benchmark(dist_name.value, device=device)
        metrics = compute_ae_baseline(tm, n_test=n_test, n_thresholds=n_thresholds)
        rows.append({"benchmark": dist_name.value, **metrics})

    df = pd.DataFrame(rows).set_index("benchmark")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == ".parquet":
            df.reset_index().to_parquet(cache_path)
        else:
            df.reset_index().to_csv(cache_path, index=False)

    return df


def _best_threshold_f1(
    x: Tensor,
    xhat: Tensor,
    n_thresholds: int = N_THRESHOLDS_DEFAULT,
) -> tuple[float, float, float]:
    """Sweep thresholds on xhat, return (best_threshold, macro_f1, micro_f1).

    Optimises for micro F1.
    """
    gt_active = x > 0
    thresholds = torch.linspace(0, 1, n_thresholds, device=x.device)

    best_f1_micro = -1.0
    best_threshold = 0.0
    best_f1_macro = 0.0

    for t in thresholds:
        pred_active = xhat > t

        tp = (gt_active & pred_active).float().sum(dim=0)
        fp = (~gt_active & pred_active).float().sum(dim=0)
        fn = (gt_active & ~pred_active).float().sum(dim=0)

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_per_feature = 2 * prec * rec / (prec + rec + 1e-8)

        tp_tot = tp.sum()
        fp_tot = fp.sum()
        fn_tot = fn.sum()
        prec_micro = (tp_tot / (tp_tot + fp_tot + 1e-8)).item()
        rec_micro = (tp_tot / (tp_tot + fn_tot + 1e-8)).item()
        f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + 1e-8)

        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro
            best_threshold = t.item()
            best_f1_macro = f1_per_feature.mean().item()

    return best_threshold, best_f1_macro, best_f1_micro
