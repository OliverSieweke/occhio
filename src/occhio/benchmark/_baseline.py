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
    n_eval_features: int | None = None,
) -> dict[str, float]:
    """Compute the best F1 achievable by the autoencoder alone (no SAE).

    Sweeps thresholds on ``ae.decode(ae.encode(x))`` against ``x > 0``
    independently per feature and returns the macro-averaged F1 at the
    per-feature optimal thresholds.

    Args:
        tm: A trained ToyModel.
        n_test: Number of test samples.
        n_thresholds: Number of thresholds to sweep.
        n_eval_features: Number of leading features to evaluate on.
            Defaults to ``n_features // 2`` to match the SAE latent space
            dimensionality.

    Returns:
        Dict with keys: f1_macro, precision_macro, recall_macro,
        threshold_mean, threshold_min, threshold_max.
    """
    device = tm.device

    with torch.no_grad():
        raw = tm.distribution.sample(n_test)
        test_x = (raw[0] if isinstance(raw, tuple) else raw).to(device)
        test_xhat = tm.ae.decode(tm.ae.encode(test_x))

    if n_eval_features is None:
        n_eval_features = tm.ae.n_features // 2
    test_x = test_x[:, :n_eval_features]
    test_xhat = test_xhat[:, :n_eval_features]

    return _per_feature_threshold_f1(test_x, test_xhat, n_thresholds=n_thresholds)


def benchmark_ae_baselines(
    distributions: list[BenchmarkDistributionName] | None = None,
    device: str | None = None,
    n_test: int = N_TEST_DEFAULT,
    n_thresholds: int = N_THRESHOLDS_DEFAULT,
    n_eval_features: int | None = None,
    cache_path: str | Path | None = None,
    ae_type: str = "huggingface",
    ae_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Compute AE F1 baselines for all (or selected) benchmark distributions.

    Args:
        distributions: Which benchmarks to evaluate. Defaults to all.
        device: Device for computation.
        n_test: Number of test samples per distribution.
        n_thresholds: Number of thresholds to sweep.
        n_eval_features: Number of leading features to evaluate on.
            Forwarded to :func:`compute_ae_baseline`.
        cache_path: If set, load results from this file when it exists,
            or save computed results there. Supports ``.parquet`` and ``.csv``.
        ae_type: Which autoencoder to use: "huggingface" (default) or "synth".
        ae_kwargs: Extra keyword arguments forwarded to the autoencoder constructor
            when ae_type="synth" (e.g. ``{"n_hidden": 200}``).

    Returns:
        DataFrame indexed by benchmark name with columns:
        f1_macro, precision_macro, recall_macro, threshold_mean,
        threshold_min, threshold_max.
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
        tm = toy_model_from_benchmark(
            dist_name.value,
            device=device,
            ae_type=ae_type,
            ae_kwargs=ae_kwargs,
        )
        metrics = compute_ae_baseline(
            tm,
            n_test=n_test,
            n_thresholds=n_thresholds,
            n_eval_features=n_eval_features,
        )
        rows.append({"benchmark": dist_name.value, **metrics})

    df = pd.DataFrame(rows).set_index("benchmark")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == ".parquet":
            df.reset_index().to_parquet(cache_path)
        else:
            df.reset_index().to_csv(cache_path, index=False)

    return df


def _per_feature_threshold_f1(
    x: Tensor,
    xhat: Tensor,
    n_thresholds: int = N_THRESHOLDS_DEFAULT,
    threshold_chunk_size: int = 10,
) -> dict[str, float]:
    """Sweep thresholds independently per feature and return macro-averaged metrics.

    For each feature, selects the threshold that maximises its F1 score, then
    reports the mean (macro) precision, recall, and F1 across features.
    """
    gt_active = x > 0  # (N, F)
    thresholds = torch.linspace(0, 0.5, n_thresholds, device=x.device)  # (T,)

    # Process thresholds in chunks to avoid materialising the full (T, N, F) tensor.
    tp_t = torch.zeros(n_thresholds, x.shape[1], device=x.device)
    fp_t = torch.zeros(n_thresholds, x.shape[1], device=x.device)
    fn_t = torch.zeros(n_thresholds, x.shape[1], device=x.device)

    gt_unsqueezed = gt_active.unsqueeze(0)  # (1, N, F)
    for start in range(0, n_thresholds, threshold_chunk_size):
        chunk = thresholds[start : start + threshold_chunk_size]  # (C,)
        pred_chunk = xhat.unsqueeze(0) > chunk.view(-1, 1, 1)  # (C, N, F)
        end = start + len(chunk)
        tp_t[start:end] = (gt_unsqueezed & pred_chunk).float().sum(dim=1)
        fp_t[start:end] = (~gt_unsqueezed & pred_chunk).float().sum(dim=1)
        fn_t[start:end] = (gt_unsqueezed & ~pred_chunk).float().sum(dim=1)

    prec_t = tp_t / (tp_t + fp_t + 1e-8)  # (T, F)
    rec_t = tp_t / (tp_t + fn_t + 1e-8)
    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)

    # Best threshold per feature (argmax over T axis)
    f1_per_feat, best_t_idx = f1_t.max(dim=0)  # (F,), (F,)
    best_thresholds = thresholds[best_t_idx]  # (F,)

    feat_idx = torch.arange(x.shape[1], device=x.device)

    return {
        "f1_macro": f1_per_feat.mean().item(),
        "precision_macro": prec_t[best_t_idx, feat_idx].mean().item(),
        "recall_macro": rec_t[best_t_idx, feat_idx].mean().item(),
        "threshold_mean": best_thresholds.mean().item(),
        "threshold_min": best_thresholds.min().item(),
        "threshold_max": best_thresholds.max().item(),
    }
