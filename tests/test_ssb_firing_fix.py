"""Tests for FiringSampler probability correctness (rank=0 and rank>0).

Verifies that the FiringSampler produces firing rates matching target
probabilities for both the independent (rank=0) and copula (rank>0) paths.
"""

import math

import pytest
import torch

from occhio.distributions.ssb import (
    CorrelationStructure,
    FiringSampler,
)

BATCH = 200_000
RTOL = 0.10  # 10% relative tolerance — generous for statistical tests
ATOL = 0.005  # absolute tolerance for very small p values


def _measure_firing_rates(
    p_values: list[float], rank: int, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a FiringSampler and measure empirical firing rates.

    Returns (target_probs, measured_rates).
    """
    probs = torch.tensor(p_values)
    n = len(p_values)

    corr = CorrelationStructure(n_features=n, rank=rank)
    sampler = FiringSampler(probabilities=probs, correlation=corr)

    gen = torch.Generator()
    gen.manual_seed(seed)
    z = sampler.sample(BATCH, generator=gen)

    rates = z.mean(dim=0)
    return probs, rates


class TestFiringSamplerRank0:
    """Independent (rank=0) Bernoulli firing path."""

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.5])
    def test_single_feature_rate(self, p: float):
        target, rates = _measure_firing_rates([p], rank=0)
        measured = rates[0].item()
        assert measured == pytest.approx(p, rel=RTOL, abs=ATOL), (
            f"rank=0: target p={p}, measured={measured:.4f}"
        )

    def test_multiple_features(self):
        p_values = [0.01, 0.05, 0.1, 0.5]
        target, rates = _measure_firing_rates(p_values, rank=0)
        for i, p in enumerate(p_values):
            measured = rates[i].item()
            assert measured == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"rank=0 feature {i}: target p={p}, measured={measured:.4f}"
            )


class TestFiringSamplerCopula:
    """Copula (rank>0) firing path — regression tests."""

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.5])
    def test_single_feature_rate(self, p: float):
        target, rates = _measure_firing_rates([p], rank=1)
        measured = rates[0].item()
        assert measured == pytest.approx(p, rel=RTOL, abs=ATOL), (
            f"rank>0: target p={p}, measured={measured:.4f}"
        )

    def test_multiple_features(self):
        p_values = [0.01, 0.05, 0.1, 0.5]
        target, rates = _measure_firing_rates(p_values, rank=2)
        for i, p in enumerate(p_values):
            measured = rates[i].item()
            assert measured == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"rank>0 feature {i}: target p={p}, measured={measured:.4f}"
            )


class TestFiringSamplerConsistency:
    """Both paths should agree on marginal rates."""

    def test_rank0_and_rank1_agree(self):
        p_values = [0.01, 0.05, 0.1, 0.5]
        _, rates_r0 = _measure_firing_rates(p_values, rank=0, seed=1)
        _, rates_r1 = _measure_firing_rates(p_values, rank=1, seed=2)
        for i, p in enumerate(p_values):
            r0 = rates_r0[i].item()
            r1 = rates_r1[i].item()
            # Both should be close to p; they won't be identical due to
            # different sampling mechanisms and seeds, but should be in
            # the same ballpark.
            assert r0 == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"rank=0 feature {i}: target={p}, got={r0:.4f}"
            )
            assert r1 == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"rank>0 feature {i}: target={p}, got={r1:.4f}"
            )
