"""Deep audit of SyntheticDataModel (SSB) pipeline.

Post-fix verification of FiringSampler probability inversion fix,
plus exhaustive testing of CorrelationStructure, MagnitudeSampler,
HierarchyConstraints, _make_schedule, _compute_firing_probs, and
the full SyntheticDataModel end-to-end pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
from torch import Tensor

from occhio.distributions.ssb import (
    CorrelationStructure,
    FiringSampler,
    HierarchyConstraints,
    HierarchyNode,
    MagnitudeSampler,
    SyntheticBatch,
    SyntheticDataConfig,
    SyntheticDataModel,
    _compute_firing_probs,
    _make_schedule,
)

# ---------------------------------------------------------------------------
# Constants for statistical tests
# ---------------------------------------------------------------------------
N_SAMPLES = 200_000
RTOL = 0.10  # 10% relative tolerance
ATOL = 0.005  # absolute tolerance for small probabilities


# ===========================================================================
# 1. _make_schedule audit
# ===========================================================================


class TestMakeSchedule:
    """Verify _make_schedule for all distribution types."""

    def test_constant_scalar(self):
        out = _make_schedule(5, "constant", value=3.14)
        assert out.shape == (5,)
        assert torch.allclose(out, torch.full((5,), 3.14))

    def test_constant_single_feature(self):
        out = _make_schedule(1, "constant", value=2.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(2.0)

    def test_constant_requires_value(self):
        with pytest.raises(AssertionError):
            _make_schedule(5, "constant")

    def test_linear_monotonic(self):
        out = _make_schedule(10, "linear", high=10.0, low=1.0)
        assert out.shape == (10,)
        assert out[0].item() == pytest.approx(10.0)
        assert out[-1].item() == pytest.approx(1.0)
        # Monotonically non-increasing
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-6).all(), (
            "linear schedule should be monotonically decreasing"
        )

    def test_linear_single_feature(self):
        out = _make_schedule(1, "linear", high=5.0, low=1.0)
        assert out.shape == (1,)
        # With n=1, linspace(5, 1, 1) = [5.0]
        assert out.item() == pytest.approx(5.0)

    def test_linear_requires_high_low(self):
        with pytest.raises(AssertionError):
            _make_schedule(5, "linear", high=10.0)
        with pytest.raises(AssertionError):
            _make_schedule(5, "linear", low=1.0)

    def test_exponential_monotonic(self):
        out = _make_schedule(10, "exponential", high=100.0, low=1.0)
        assert out.shape == (10,)
        assert out[0].item() == pytest.approx(100.0, rel=0.01)
        assert out[-1].item() == pytest.approx(1.0, rel=0.01)
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-4).all(), "exponential schedule should be decreasing"

    def test_exponential_single_feature(self):
        out = _make_schedule(1, "exponential", high=10.0, low=0.1)
        assert out.shape == (1,)

    def test_exponential_values_positive(self):
        out = _make_schedule(100, "exponential", high=10.0, low=0.01)
        assert (out > 0).all()

    def test_folded_normal_non_negative(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        out = _make_schedule(
            100, "folded_normal", folded_mu=1.0, folded_sigma=0.5, generator=gen
        )
        assert out.shape == (100,)
        assert (out >= 0).all(), "folded_normal should produce non-negative values"

    def test_folded_normal_single_feature(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        out = _make_schedule(
            1, "folded_normal", folded_mu=1.0, folded_sigma=0.5, generator=gen
        )
        assert out.shape == (1,)
        assert out.item() >= 0

    def test_folded_normal_requires_params(self):
        with pytest.raises(AssertionError):
            _make_schedule(5, "folded_normal", folded_mu=1.0)
        with pytest.raises(AssertionError):
            _make_schedule(5, "folded_normal", folded_sigma=0.5)

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule distribution"):
            _make_schedule(5, "nonexistent")

    def test_100_features_shapes(self):
        """All schedule types produce correct shape for n=100."""
        assert _make_schedule(100, "constant", value=1.0).shape == (100,)
        assert _make_schedule(100, "linear", high=10.0, low=1.0).shape == (100,)
        assert _make_schedule(100, "exponential", high=10.0, low=0.1).shape == (100,)
        gen = torch.Generator()
        gen.manual_seed(0)
        assert _make_schedule(
            100, "folded_normal", folded_mu=1.0, folded_sigma=0.5, generator=gen
        ).shape == (100,)


# ===========================================================================
# 2. _compute_firing_probs audit
# ===========================================================================


class TestComputeFiringProbs:
    """Verify _compute_firing_probs for all distribution types."""

    def test_constant_default(self):
        out = _compute_firing_probs(5, "constant", p_min=0.01, p_max=0.1)
        assert out.shape == (5,)
        # When p_constant is None, falls back to p_min
        assert torch.allclose(out, torch.full((5,), 0.01))

    def test_constant_with_p_constant(self):
        out = _compute_firing_probs(
            5, "constant", p_min=0.01, p_max=0.1, p_constant=0.05
        )
        assert torch.allclose(out, torch.full((5,), 0.05))

    def test_linear_range_and_monotonicity(self):
        out = _compute_firing_probs(10, "linear", p_min=0.001, p_max=0.1)
        assert out.shape == (10,)
        assert out[0].item() == pytest.approx(0.1, rel=1e-5)
        assert out[-1].item() == pytest.approx(0.001, rel=1e-5)
        # Monotonically non-increasing
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-7).all()

    def test_linear_values_in_range(self):
        out = _compute_firing_probs(50, "linear", p_min=0.001, p_max=0.5)
        assert (out >= 0).all() and (out <= 1).all()

    def test_uniform_range(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        out = _compute_firing_probs(
            100, "uniform", p_min=0.01, p_max=0.1, generator=gen
        )
        assert out.shape == (100,)
        assert (out >= 0.01 - 1e-6).all()
        assert (out <= 0.1 + 1e-6).all()

    def test_zipfian_monotonic_and_range(self):
        out = _compute_firing_probs(50, "zipfian", p_min=0.001, p_max=0.1, alpha=1.0)
        assert out.shape == (50,)
        # Zipfian with alpha>0: first feature gets p_max, last gets p_min
        assert out[0].item() == pytest.approx(0.1, rel=1e-4)
        assert out[-1].item() == pytest.approx(0.001, rel=1e-4)
        # Monotonically non-increasing
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-6).all()

    def test_zipfian_single_feature(self):
        # n=1 means q_max == q_min, should get midpoint
        out = _compute_firing_probs(1, "zipfian", p_min=0.01, p_max=0.1, alpha=1.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(0.055, rel=1e-4)

    def test_all_probs_valid(self):
        """All distribution types return values in [0, 1]."""
        for dist in ["constant", "linear", "zipfian"]:
            out = _compute_firing_probs(20, dist, p_min=0.001, p_max=0.5)
            assert (out >= 0).all() and (out <= 1).all(), f"{dist}: invalid prob"
        gen = torch.Generator()
        gen.manual_seed(0)
        out = _compute_firing_probs(
            20, "uniform", p_min=0.001, p_max=0.5, generator=gen
        )
        assert (out >= 0).all() and (out <= 1).all()

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown firing prob distribution"):
            _compute_firing_probs(5, "nonexistent", p_min=0.01, p_max=0.1)


# ===========================================================================
# 3. FiringSampler post-fix verification
# ===========================================================================


class TestFiringSamplerPostFix:
    """Comprehensive post-fix verification: all firing distributions x both ranks."""

    @pytest.mark.parametrize("dist_type", ["constant", "linear", "zipfian"])
    def test_rank0_all_firing_distributions(self, dist_type: str):
        """rank=0 path: rates match targets for each distribution type."""
        probs = _compute_firing_probs(
            10, dist_type, p_min=0.01, p_max=0.1, p_constant=0.05
        )
        corr = CorrelationStructure(n_features=10, rank=0)
        sampler = FiringSampler(probabilities=probs, correlation=corr)

        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(N_SAMPLES, generator=gen)
        rates = z.mean(dim=0)

        for i in range(10):
            p = probs[i].item()
            r = rates[i].item()
            assert r == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"{dist_type} rank=0 feature {i}: target={p:.4f}, measured={r:.4f}"
            )

    @pytest.mark.parametrize("dist_type", ["constant", "linear", "zipfian"])
    def test_rank2_all_firing_distributions(self, dist_type: str):
        """rank>0 path: marginal rates still match targets."""
        probs = _compute_firing_probs(
            10, dist_type, p_min=0.01, p_max=0.1, p_constant=0.05
        )
        corr = CorrelationStructure(n_features=10, rank=2, correlation_scale=0.1)
        sampler = FiringSampler(probabilities=probs, correlation=corr)

        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(N_SAMPLES, generator=gen)
        rates = z.mean(dim=0)

        for i in range(10):
            p = probs[i].item()
            r = rates[i].item()
            assert r == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"{dist_type} rank=2 feature {i}: target={p:.4f}, measured={r:.4f}"
            )

    def test_rank0_and_rank2_agree_on_marginals(self):
        """Both paths should produce same marginal rates (different correlation)."""
        probs = _compute_firing_probs(10, "zipfian", p_min=0.01, p_max=0.1)

        corr0 = CorrelationStructure(n_features=10, rank=0)
        sampler0 = FiringSampler(probabilities=probs, correlation=corr0)
        gen0 = torch.Generator()
        gen0.manual_seed(1)
        z0 = sampler0.sample(N_SAMPLES, generator=gen0)
        rates0 = z0.mean(dim=0)

        corr2 = CorrelationStructure(n_features=10, rank=2, correlation_scale=0.1)
        sampler2 = FiringSampler(probabilities=probs, correlation=corr2)
        gen2 = torch.Generator()
        gen2.manual_seed(2)
        z2 = sampler2.sample(N_SAMPLES, generator=gen2)
        rates2 = z2.mean(dim=0)

        for i in range(10):
            p = probs[i].item()
            r0 = rates0[i].item()
            r2 = rates2[i].item()
            assert r0 == pytest.approx(p, rel=RTOL, abs=ATOL)
            assert r2 == pytest.approx(p, rel=RTOL, abs=ATOL)

    def test_firing_sampler_output_binary(self):
        """Output should contain only 0.0 and 1.0."""
        probs = torch.tensor([0.1, 0.3, 0.5, 0.9])
        corr = CorrelationStructure(n_features=4, rank=0)
        sampler = FiringSampler(probabilities=probs, correlation=corr)
        z = sampler.sample(1000)
        unique_vals = z.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_firing_sampler_output_binary_copula(self):
        """Copula path should also produce binary output."""
        probs = torch.tensor([0.1, 0.3, 0.5, 0.9])
        corr = CorrelationStructure(n_features=4, rank=2)
        sampler = FiringSampler(probabilities=probs, correlation=corr)
        z = sampler.sample(1000)
        unique_vals = z.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_extreme_prob_zero(self):
        """p=0 should produce no firings."""
        probs = torch.tensor([0.0])
        corr = CorrelationStructure(n_features=1, rank=0)
        sampler = FiringSampler(probabilities=probs, correlation=corr)
        z = sampler.sample(10_000)
        assert z.sum().item() == 0.0

    def test_extreme_prob_one(self):
        """p=1 should produce all firings."""
        probs = torch.tensor([1.0])
        corr = CorrelationStructure(n_features=1, rank=0)
        sampler = FiringSampler(probabilities=probs, correlation=corr)
        z = sampler.sample(10_000)
        assert z.sum().item() == 10_000.0


# ===========================================================================
# 4. CorrelationStructure audit
# ===========================================================================


class TestCorrelationStructure:
    """Verify CorrelationStructure for rank=0 and rank>0."""

    def test_rank0_no_factor_matrix(self):
        cs = CorrelationStructure(n_features=10, rank=0)
        assert cs.factor_matrix is None
        assert cs.diagonal is None

    def test_rank0_sample_shape(self):
        cs = CorrelationStructure(n_features=10, rank=0)
        g = cs.sample(32)
        assert g.shape == (32, 10)

    def test_rank0_sample_stats(self):
        """rank=0 samples should be standard normal."""
        cs = CorrelationStructure(n_features=5, rank=0)
        gen = torch.Generator()
        gen.manual_seed(42)
        g = cs.sample(100_000, generator=gen)
        # Mean ~ 0, std ~ 1
        assert g.mean().item() == pytest.approx(0.0, abs=0.02)
        assert g.std().item() == pytest.approx(1.0, abs=0.02)

    def test_rank2_factor_matrix_shape(self):
        cs = CorrelationStructure(n_features=10, rank=2, correlation_scale=0.1)
        assert cs.factor_matrix is not None
        assert cs.factor_matrix.shape == (10, 2)
        assert cs.diagonal is not None
        assert cs.diagonal.shape == (10,)

    def test_rank2_positive_semi_definite(self):
        """Sigma = F @ F.T + diag(delta) should be PSD."""
        cs = CorrelationStructure(n_features=10, rank=2, correlation_scale=0.1)
        F = cs.factor_matrix
        d = cs.diagonal
        Sigma = F @ F.T + torch.diag(d)
        eigenvalues = torch.linalg.eigvalsh(Sigma)
        assert (eigenvalues >= -1e-6).all(), (
            f"Sigma not PSD: min eigenvalue = {eigenvalues.min().item()}"
        )

    def test_rank2_diagonal_positive(self):
        """Diagonal entries should be >= delta_min after rescaling."""
        cs = CorrelationStructure(
            n_features=10, rank=2, correlation_scale=0.1, delta_min=0.01
        )
        assert (cs.diagonal >= 0.01 - 1e-7).all()

    def test_large_correlation_scale_triggers_rescaling(self):
        """With large scale, delta should be clamped to >= delta_min."""
        cs = CorrelationStructure(
            n_features=10,
            rank=5,
            correlation_scale=10.0,  # Very large, will trigger rescaling
            delta_min=0.05,
        )
        assert cs.diagonal is not None
        assert (cs.diagonal >= 0.05 - 1e-6).all(), (
            f"delta_min not enforced: min delta = {cs.diagonal.min().item()}"
        )

    def test_sample_covariance_approximately_sigma(self):
        """Empirical covariance should approximate Sigma."""
        n = 5
        cs = CorrelationStructure(n_features=n, rank=2, correlation_scale=0.3)
        gen = torch.Generator()
        gen.manual_seed(42)
        g = cs.sample(200_000, generator=gen)
        emp_cov = (g.T @ g) / g.shape[0]
        F = cs.factor_matrix
        d = cs.diagonal
        Sigma = F @ F.T + torch.diag(d)
        # Each entry should be close
        assert torch.allclose(emp_cov, Sigma, atol=0.03), (
            f"Max cov diff = {(emp_cov - Sigma).abs().max().item():.4f}"
        )

    def test_n_features_1_rank0(self):
        cs = CorrelationStructure(n_features=1, rank=0)
        g = cs.sample(100)
        assert g.shape == (100, 1)

    def test_n_features_1_rank1(self):
        """Edge: n_features=1 with rank=1."""
        cs = CorrelationStructure(n_features=1, rank=1, correlation_scale=0.1)
        assert cs.factor_matrix.shape == (1, 1)
        assert cs.diagonal.shape == (1,)
        g = cs.sample(100)
        assert g.shape == (100, 1)


# ===========================================================================
# 5. MagnitudeSampler audit
# ===========================================================================


class TestMagnitudeSampler:
    """Verify MagnitudeSampler: non-negative, gated by z, correct stats."""

    def test_output_non_negative(self):
        means = torch.tensor([1.0, 2.0, 3.0])
        stds = torch.tensor([0.5, 0.5, 0.5])
        ms = MagnitudeSampler(means, stds)
        z = torch.ones(10_000, 3)
        c = ms.sample(z)
        assert (c >= 0).all(), "MagnitudeSampler should produce non-negative output"

    def test_zero_z_produces_zero_magnitude(self):
        """Features that don't fire should have exactly zero activation."""
        means = torch.tensor([5.0, 5.0, 5.0])
        stds = torch.tensor([1.0, 1.0, 1.0])
        ms = MagnitudeSampler(means, stds)
        z = torch.zeros(100, 3)
        c = ms.sample(z)
        assert (c == 0).all(), "Zero z must produce zero magnitude"

    def test_partial_z_gating(self):
        """Only features with z=1 should have nonzero activations."""
        means = torch.tensor([10.0, 10.0])
        stds = torch.tensor([0.1, 0.1])  # Small std so ReLU rarely clips
        ms = MagnitudeSampler(means, stds)
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        c = ms.sample(z)
        assert c[0, 1].item() == 0.0
        assert c[1, 0].item() == 0.0
        assert c[0, 0].item() > 0.0
        assert c[1, 1].item() > 0.0

    def test_mean_magnitudes_match_configured(self):
        """Mean of sampled magnitudes (when z=1) should approximate configured mean."""
        mu = 3.0
        sigma = 0.5
        means = torch.full((1,), mu)
        stds = torch.full((1,), sigma)
        ms = MagnitudeSampler(means, stds)
        z = torch.ones(200_000, 1)
        gen = torch.Generator()
        gen.manual_seed(42)
        c = ms.sample(z, generator=gen)
        # Expected mean of ReLU(N(mu, sigma^2)) = mu * Phi(mu/sigma) + sigma * phi(mu/sigma)
        # For mu=3, sigma=0.5, mu/sigma=6, Phi(6)~1, phi(6)~0, so expected ~= mu
        assert c.mean().item() == pytest.approx(mu, rel=0.05)

    @pytest.mark.parametrize(
        "dist_type,kwargs",
        [
            ("constant", {"value": 2.0}),
            ("linear", {"high": 5.0, "low": 0.5}),
            ("exponential", {"high": 5.0, "low": 0.5}),
        ],
    )
    def test_magnitude_with_different_distributions(self, dist_type, kwargs):
        """MagnitudeSampler works with means from different schedule types."""
        means = _make_schedule(10, dist_type, **kwargs)
        stds = torch.full((10,), 0.5)
        ms = MagnitudeSampler(means, stds)
        z = torch.ones(1000, 10)
        c = ms.sample(z)
        assert (c >= 0).all()
        assert c.shape == (1000, 10)

    def test_zero_mean_positive_std(self):
        """With mean=0, output should still be non-negative (ReLU)."""
        means = torch.zeros(3)
        stds = torch.ones(3)
        ms = MagnitudeSampler(means, stds)
        z = torch.ones(10_000, 3)
        c = ms.sample(z)
        assert (c >= 0).all()
        # About half should be zero (ReLU of N(0,1))
        frac_zero = (c == 0).float().mean().item()
        assert frac_zero == pytest.approx(0.5, abs=0.05)


# ===========================================================================
# 6. HierarchyConstraints audit
# ===========================================================================


class TestHierarchyConstraints:
    """Verify hierarchy: gating, mutual exclusion, parent-scaling, compensation."""

    def _make_simple_forest(self) -> list[HierarchyNode]:
        """Parent (0) with children (1, 2)."""
        return [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=False,
                parent_scaled=False,
            )
        ]

    def test_parent_gating_zeros_children_when_parent_off(self):
        """When parent is off, children should be zeroed."""
        forest = self._make_simple_forest()
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)

        # Parent off (feature 0 = 0), children have values
        c = torch.tensor([[0.0, 5.0, 3.0], [2.0, 5.0, 3.0]])
        result = hc.apply(c)

        # Row 0: parent off -> children zeroed
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0
        # Row 1: parent on -> children preserved
        assert result[1, 1].item() == 5.0
        assert result[1, 2].item() == 3.0

    def test_mutual_exclusion_at_most_one_child(self):
        """Among mutually exclusive siblings, at most one should survive."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                    HierarchyNode(feature_idx=3),
                ],
                mutually_exclusive_children=True,
                parent_scaled=False,
            )
        ]
        hc = HierarchyConstraints(forest, n_features=4, compensate=False)

        # Parent on, all 3 children fire
        gen = torch.Generator()
        gen.manual_seed(42)
        c = torch.tensor([[2.0, 1.0, 1.0, 1.0]] * 1000)
        result = hc.apply(c, generator=gen)

        # Each row: exactly one child should survive
        child_fired = (result[:, 1:4] > 0).sum(dim=1)
        assert (child_fired <= 1).all(), "Mutual exclusion violated: >1 child fired"
        # At least some rows should have exactly 1 child
        assert (child_fired == 1).any()

    def test_mutual_exclusion_no_children_fire_when_parent_off(self):
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
                parent_scaled=False,
            )
        ]
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)

        c = torch.tensor([[0.0, 3.0, 5.0]])
        result = hc.apply(c)
        # Parent off => all children zeroed (gating before ME)
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0

    def test_parent_scaled_mode(self):
        """Child magnitude scaled by parent's value / mean_magnitude."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[HierarchyNode(feature_idx=1)],
                mutually_exclusive_children=False,
                parent_scaled=True,
            )
        ]
        mean_mags = torch.tensor([2.0, 1.0])
        hc = HierarchyConstraints(
            forest, n_features=2, compensate=False, mean_magnitudes=mean_mags
        )

        # Parent has value 4.0, mean_mag=2.0 => scale = 4.0/2.0 = 2.0
        c = torch.tensor([[4.0, 3.0]])
        result = hc.apply(c)
        assert result[0, 1].item() == pytest.approx(6.0)  # 3.0 * 2.0

    def test_parent_scaled_zero_mean_magnitude(self):
        """If mean_magnitude of parent is 0, should skip scaling (no div-by-zero)."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[HierarchyNode(feature_idx=1)],
                mutually_exclusive_children=False,
                parent_scaled=True,
            )
        ]
        mean_mags = torch.tensor([0.0, 1.0])
        hc = HierarchyConstraints(
            forest, n_features=2, compensate=False, mean_magnitudes=mean_mags
        )

        c = torch.tensor([[1.0, 3.0]])
        result = hc.apply(c)
        # mu_parent == 0 => scaling is skipped
        assert result[0, 1].item() == pytest.approx(3.0)

    def test_compensation_boost_increases_child_prob(self):
        """Compensation should increase effective child probs to offset gating."""
        forest = self._make_simple_forest()
        base_probs = torch.tensor([0.5, 0.1, 0.1])
        hc = HierarchyConstraints(
            forest, n_features=3, compensate=True, base_probs=base_probs
        )
        compensated = hc.get_compensated_probs(base_probs)
        # Children should have boosted probabilities
        # child_comp = base_prob * (1 / p_parent) = 0.1 * (1/0.5) = 0.2
        assert compensated[1].item() == pytest.approx(0.2, rel=0.01)
        assert compensated[2].item() == pytest.approx(0.2, rel=0.01)
        # Parent should be unaffected
        assert compensated[0].item() == pytest.approx(0.5, rel=0.01)

    def test_compensation_clamps_to_1(self):
        """Compensation shouldn't exceed 1.0."""
        forest = self._make_simple_forest()
        base_probs = torch.tensor([0.05, 0.9, 0.9])  # child prob > parent => boost > 1
        hc = HierarchyConstraints(
            forest, n_features=3, compensate=True, base_probs=base_probs
        )
        compensated = hc.get_compensated_probs(base_probs)
        assert (compensated <= 1.0).all()

    def test_empty_forest_passthrough(self):
        """With no hierarchy, constraints should be a no-op."""
        hc = HierarchyConstraints(forest=[], n_features=5, compensate=False)
        assert not hc.has_constraints
        c = torch.randn(10, 5).abs()
        result = hc.apply(c)
        # apply should not be called in practice when has_constraints is False,
        # but if it is, the tensor should pass through unchanged
        assert torch.equal(c, result)

    def test_multi_level_hierarchy(self):
        """Test grandparent -> parent -> child gating."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(
                        feature_idx=1,
                        children=[HierarchyNode(feature_idx=2)],
                    )
                ],
            )
        ]
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)

        # Grandparent off -> everything zeroed
        c = torch.tensor([[0.0, 1.0, 1.0]])
        result = hc.apply(c)
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0

        # Grandparent on, parent off -> grandchild zeroed
        c = torch.tensor([[1.0, 0.0, 1.0]])
        result = hc.apply(c)
        assert result[0, 1].item() == 0.0  # parent was zero
        assert result[0, 2].item() == 0.0  # grandchild gated by parent

        # All on -> all preserved
        c = torch.tensor([[1.0, 1.0, 1.0]])
        result = hc.apply(c)
        assert result[0, 0].item() == 1.0
        assert result[0, 1].item() == 1.0
        assert result[0, 2].item() == 1.0


# ===========================================================================
# 7. Full SyntheticDataModel end-to-end
# ===========================================================================


class TestSyntheticDataModelE2E:
    """End-to-end tests for the full sampling pipeline."""

    def test_basic_sample_shape(self):
        cfg = SyntheticDataConfig(n_features=20)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 20)

    def test_output_non_negative(self):
        cfg = SyntheticDataConfig(n_features=20)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(10_000)
        assert (c >= 0).all()

    def test_sparsity_reasonable(self):
        """Output should be sparse but not all zeros."""
        cfg = SyntheticDataConfig(n_features=50, p_min=0.01, p_max=0.1)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(10_000)
        frac_nonzero = (c > 0).float().mean().item()
        # Expected: mean(p_i) ~ 0.05 (midpoint) but ReLU clips some
        assert frac_nonzero > 0.005, "Too sparse — almost all zeros"
        assert frac_nonzero < 0.5, "Not sparse enough"

    def test_not_all_zeros_not_all_nonzero(self):
        cfg = SyntheticDataConfig(n_features=10, p_min=0.01, p_max=0.5)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(1000)
        assert c.sum().item() > 0, "All zeros"
        assert (c == 0).any(), "No zeros"

    def test_with_correlation_rank2(self):
        cfg = SyntheticDataConfig(
            n_features=20, correlation_rank=2, correlation_scale=0.2
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(1000)
        assert c.shape == (1000, 20)
        assert (c >= 0).all()

    def test_with_hierarchy(self):
        """Full pipeline with hierarchy constraints."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
                parent_scaled=False,
            )
        ]
        cfg = SyntheticDataConfig(
            n_features=5,
            hierarchy=forest,
            p_min=0.1,
            p_max=0.5,
            firing_prob_distribution="constant",
            p_constant=0.3,
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(10_000)

        # Children should never both be active (mutual exclusion)
        both_active = ((c[:, 1] > 0) & (c[:, 2] > 0)).sum().item()
        assert both_active == 0, (
            f"Mutual exclusion violated: {both_active} rows have both children active"
        )

        # Children should be zero whenever parent is zero
        parent_off = c[:, 0] == 0
        child_on_when_parent_off = (
            ((c[parent_off, 1] > 0) | (c[parent_off, 2] > 0)).sum().item()
        )
        assert child_on_when_parent_off == 0, "Child active when parent is off"

    def test_reproducibility_with_seed(self):
        cfg = SyntheticDataConfig(n_features=10)
        m1 = SyntheticDataModel(cfg, seed=42)
        m2 = SyntheticDataModel(cfg, seed=42)
        c1 = m1.sample(100)
        c2 = m2.sample(100)
        assert torch.equal(c1, c2), "Same seed should produce identical samples"

    def test_different_seeds_differ(self):
        cfg = SyntheticDataConfig(n_features=10)
        m1 = SyntheticDataModel(cfg, seed=42)
        m2 = SyntheticDataModel(cfg, seed=99)
        c1 = m1.sample(100)
        c2 = m2.sample(100)
        assert not torch.equal(c1, c2), "Different seeds should differ"

    def test_post_processing_applied(self):
        """post_processing callable should transform output."""
        cfg = SyntheticDataConfig(n_features=5, post_processing=lambda c: c * 2.0)
        model = SyntheticDataModel(cfg, seed=42)

        # Compare with no post-processing
        cfg2 = SyntheticDataConfig(n_features=5)
        model2 = SyntheticDataModel(cfg2, seed=42)

        c1 = model.sample(100)
        c2 = model2.sample(100)
        assert torch.allclose(c1, c2 * 2.0)

    def test_firing_probabilities_property(self):
        cfg = SyntheticDataConfig(n_features=10, p_min=0.01, p_max=0.1)
        model = SyntheticDataModel(cfg, seed=42)
        probs = model.firing_probabilities
        assert probs.shape == (10,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_compensated_probabilities_no_hierarchy(self):
        """Without hierarchy, compensated == base."""
        cfg = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(cfg, seed=42)
        assert torch.equal(model.firing_probabilities, model.compensated_probabilities)

    def test_compensated_probabilities_with_hierarchy(self):
        """With hierarchy, compensated should differ from base."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[HierarchyNode(feature_idx=1)],
            )
        ]
        cfg = SyntheticDataConfig(
            n_features=5,
            hierarchy=forest,
            p_constant=0.3,
            firing_prob_distribution="constant",
            compensate_probabilities=True,
        )
        model = SyntheticDataModel(cfg, seed=42)
        base = model.firing_probabilities
        comp = model.compensated_probabilities
        # Feature 1 (child) should be compensated
        assert comp[1].item() > base[1].item()

    def test_all_schedule_types_in_full_pipeline(self):
        """Smoke test: each schedule type runs without error in the full pipeline."""
        for mean_dist, mean_kw in [
            ("constant", {"mean_value": 1.0}),
            ("linear", {"mean_high": 3.0, "mean_low": 0.5}),
            ("exponential", {"mean_high": 3.0, "mean_low": 0.5}),
        ]:
            cfg = SyntheticDataConfig(
                n_features=10,
                mean_distribution=mean_dist,
                **mean_kw,
            )
            model = SyntheticDataModel(cfg, seed=42)
            c = model.sample(100)
            assert c.shape == (100, 10)
            assert (c >= 0).all()

    def test_all_firing_prob_types_in_full_pipeline(self):
        """Smoke test: each firing prob distribution runs without error."""
        for dist in ["constant", "linear", "zipfian"]:
            cfg = SyntheticDataConfig(
                n_features=10,
                firing_prob_distribution=dist,
                p_constant=0.05,
            )
            model = SyntheticDataModel(cfg, seed=42)
            c = model.sample(100)
            assert c.shape == (100, 10)

    def test_uniform_firing_prob_in_full_pipeline(self):
        cfg = SyntheticDataConfig(
            n_features=10,
            firing_prob_distribution="uniform",
            p_min=0.01,
            p_max=0.1,
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 10)

    def test_to_device_moves_all_state(self):
        """to() should not crash (CPU->CPU as device test)."""
        cfg = SyntheticDataConfig(n_features=5)
        model = SyntheticDataModel(cfg, seed=42)
        model.to("cpu")
        c = model.sample(10)
        assert c.shape == (10, 5)

    def test_folded_normal_std_in_full_pipeline(self):
        """Folded normal for std distribution."""
        cfg = SyntheticDataConfig(
            n_features=10,
            std_distribution="folded_normal",
            folded_normal_mu=0.5,
            folded_normal_sigma=0.2,
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 10)
        assert (c >= 0).all()

    def test_empirical_firing_rates_match_configured(self):
        """Empirical firing rates should approximately match configured probs."""
        cfg = SyntheticDataConfig(
            n_features=10,
            firing_prob_distribution="constant",
            p_constant=0.1,
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(N_SAMPLES)
        # Firing = nonzero (after magnitude sampling and ReLU)
        empirical_rates = (c > 0).float().mean(dim=0)
        for i in range(10):
            r = empirical_rates[i].item()
            # Rate should be somewhat less than 0.1 due to ReLU clipping
            # (mean=1.0, std=0.5 => ReLU clips about 2.3% of N(1,0.5))
            # So expected rate ~ 0.1 * 0.977 ~ 0.0977
            assert r == pytest.approx(0.1, rel=0.15, abs=0.01), (
                f"Feature {i}: expected ~0.1, got {r:.4f}"
            )


# ===========================================================================
# 8. Edge cases and combinatorial interactions
# ===========================================================================


class TestEdgeCases:
    """Edge cases and potential combinatorial failure modes."""

    def test_single_feature(self):
        """n_features=1 should work throughout the pipeline."""
        cfg = SyntheticDataConfig(n_features=1, p_constant=0.5)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 1)

    def test_single_feature_with_correlation(self):
        """n_features=1, rank=1."""
        cfg = SyntheticDataConfig(
            n_features=1, correlation_rank=1, correlation_scale=0.1
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 1)

    def test_many_features(self):
        """Large n_features should work."""
        cfg = SyntheticDataConfig(n_features=500, p_min=0.001, p_max=0.01)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 500)
        assert (c >= 0).all()

    def test_very_small_probabilities(self):
        """Very small p should produce very sparse output."""
        cfg = SyntheticDataConfig(
            n_features=10,
            firing_prob_distribution="constant",
            p_constant=0.001,
        )
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(10_000)
        rate = (c > 0).float().mean().item()
        assert rate < 0.01

    def test_hierarchy_with_all_features_as_roots(self):
        """Hierarchy where no node has children (all roots with no ops)."""
        forest = [HierarchyNode(feature_idx=i) for i in range(5)]
        cfg = SyntheticDataConfig(n_features=5, hierarchy=forest)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(100)
        assert c.shape == (100, 5)

    def test_batch_size_1(self):
        cfg = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(cfg, seed=42)
        c = model.sample(1)
        assert c.shape == (1, 10)

    def test_correlation_structure_reproducible(self):
        """Same seed should give same correlation structure."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        cs1 = CorrelationStructure(
            n_features=5, rank=2, correlation_scale=0.2, generator=gen1
        )

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        cs2 = CorrelationStructure(
            n_features=5, rank=2, correlation_scale=0.2, generator=gen2
        )

        assert torch.equal(cs1.factor_matrix, cs2.factor_matrix)
        assert torch.equal(cs1.diagonal, cs2.diagonal)

    def test_hierarchy_me_all_children_off(self):
        """ME with no children firing should be a no-op."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
            )
        ]
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)
        c = torch.tensor([[1.0, 0.0, 0.0]])  # parent on, children off
        result = hc.apply(c)
        assert torch.equal(c, result)

    def test_hierarchy_me_single_child_fires(self):
        """ME with exactly one child firing should preserve it."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
            )
        ]
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)
        c = torch.tensor([[1.0, 5.0, 0.0]])
        result = hc.apply(c)
        assert result[0, 1].item() == 5.0
        assert result[0, 2].item() == 0.0

    def test_inv_normal_cdf_roundtrip(self):
        """_inv_normal_cdf and _inv_normal_cdf_to_prob should be inverses."""
        probs = torch.tensor([0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.99])
        thresholds = FiringSampler._inv_normal_cdf(1.0 - probs)
        recovered = FiringSampler._inv_normal_cdf_to_prob(thresholds)
        assert torch.allclose(probs, recovered, atol=1e-6)

    def test_compensation_with_zero_parent_prob(self):
        """Compensation should handle p_parent=0 gracefully (no division by zero)."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[HierarchyNode(feature_idx=1)],
            )
        ]
        base_probs = torch.tensor([0.0, 0.1, 0.5])
        hc = HierarchyConstraints(
            forest, n_features=3, compensate=True, base_probs=base_probs
        )
        comp = hc.get_compensated_probs(base_probs)
        # Should not be inf or nan
        assert torch.isfinite(comp).all()
        # Child compensation: 1/0 is skipped (gamma stays 1), so child prob unchanged
        assert comp[1].item() == pytest.approx(0.1)

    def test_me_compensation_formula(self):
        """Verify ME compensation formula is correct.

        For parent P with ME children C1, C2:
        gamma_C1 = (1/p_P) * (1 + p_C2/p_P)
        """
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
            )
        ]
        base_probs = torch.tensor([0.5, 0.1, 0.2])
        hc = HierarchyConstraints(
            forest, n_features=3, compensate=True, base_probs=base_probs
        )
        comp = hc.get_compensated_probs(base_probs)

        # gamma_1 = (1/0.5) * (1 + 0.2/0.5) = 2.0 * 1.4 = 2.8
        # comp_1 = 0.1 * 2.8 = 0.28
        assert comp[1].item() == pytest.approx(0.28, rel=1e-4)

        # gamma_2 = (1/0.5) * (1 + 0.1/0.5) = 2.0 * 1.2 = 2.4
        # comp_2 = 0.2 * 2.4 = 0.48
        assert comp[2].item() == pytest.approx(0.48, rel=1e-4)

    def test_correlation_scale_zero(self):
        """correlation_scale=0 should produce identity covariance (all independent)."""
        cs = CorrelationStructure(
            n_features=5, rank=2, correlation_scale=0.0, delta_min=0.01
        )
        # F is all zeros, delta is all ones
        assert cs.factor_matrix is not None
        assert (cs.factor_matrix == 0).all()
        assert torch.allclose(cs.diagonal, torch.ones(5))

    def test_rank_greater_than_n_features(self):
        """rank > n_features should still work (over-parameterized factor matrix)."""
        cs = CorrelationStructure(
            n_features=3, rank=10, correlation_scale=0.05, delta_min=0.01
        )
        assert cs.factor_matrix.shape == (3, 10)
        assert cs.diagonal.shape == (3,)
        # Sigma should still be PSD
        F = cs.factor_matrix
        d = cs.diagonal
        Sigma = F @ F.T + torch.diag(d)
        eigenvalues = torch.linalg.eigvalsh(Sigma)
        assert (eigenvalues >= -1e-6).all()

    def test_hierarchy_does_not_mutate_input(self):
        """HierarchyConstraints.apply() should not mutate the input tensor."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[HierarchyNode(feature_idx=1)],
            )
        ]
        hc = HierarchyConstraints(forest, n_features=3, compensate=False)
        c = torch.tensor([[0.0, 5.0, 1.0], [1.0, 5.0, 1.0]])
        c_original = c.clone()
        _ = hc.apply(c)
        assert torch.equal(c, c_original), "apply() should not mutate input tensor"

    def test_magnitude_sampler_with_large_negative_mean(self):
        """With mean << 0, ReLU should clip all outputs to zero."""
        means = torch.full((5,), -10.0)
        stds = torch.ones(5)
        ms = MagnitudeSampler(means, stds)
        z = torch.ones(10_000, 5)
        c = ms.sample(z)
        # N(-10, 1) is almost surely negative => ReLU gives 0
        assert (c >= 0).all()
        frac_nonzero = (c > 0).float().mean().item()
        assert frac_nonzero < 0.001, (
            f"Expected near-zero nonzero fraction, got {frac_nonzero}"
        )

    def test_firing_sampler_with_uniform_probs(self):
        """FiringSampler with uniform-distributed probs should still match targets."""
        gen = torch.Generator()
        gen.manual_seed(42)
        probs = _compute_firing_probs(
            10, "uniform", p_min=0.01, p_max=0.2, generator=gen
        )
        corr = CorrelationStructure(n_features=10, rank=0)
        sampler = FiringSampler(probabilities=probs, correlation=corr)

        gen2 = torch.Generator()
        gen2.manual_seed(99)
        z = sampler.sample(N_SAMPLES, generator=gen2)
        rates = z.mean(dim=0)

        for i in range(10):
            p = probs[i].item()
            r = rates[i].item()
            assert r == pytest.approx(p, rel=RTOL, abs=ATOL), (
                f"uniform prob feature {i}: target={p:.4f}, measured={r:.4f}"
            )

    def test_synthetic_data_model_inherits_distribution(self):
        """SyntheticDataModel should be a valid Distribution subclass."""
        from occhio.distributions.base import Distribution

        cfg = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(cfg, seed=42)
        assert isinstance(model, Distribution)
        assert model.n_features == 10

    def test_hierarchy_with_parent_scaled_and_me_combined(self):
        """Test parent_scaled + mutually_exclusive combined."""
        forest = [
            HierarchyNode(
                feature_idx=0,
                children=[
                    HierarchyNode(feature_idx=1),
                    HierarchyNode(feature_idx=2),
                ],
                mutually_exclusive_children=True,
                parent_scaled=True,
            )
        ]
        mean_mags = torch.tensor([2.0, 1.0, 1.0])
        hc = HierarchyConstraints(
            forest,
            n_features=3,
            compensate=False,
            mean_magnitudes=mean_mags,
        )

        # Parent = 4.0 (mean = 2.0, so scale = 2.0), both children fire
        gen = torch.Generator()
        gen.manual_seed(42)
        c = torch.tensor([[4.0, 3.0, 5.0]] * 100)
        result = hc.apply(c, generator=gen)

        # After ME: exactly one child survives per row
        child_fired = (result[:, 1:3] > 0).sum(dim=1)
        assert (child_fired <= 1).all()

        # The surviving child should be scaled by 2.0
        for i in range(100):
            if result[i, 1].item() > 0:
                assert result[i, 1].item() == pytest.approx(3.0 * 2.0, rel=1e-4)
            if result[i, 2].item() > 0:
                assert result[i, 2].item() == pytest.approx(5.0 * 2.0, rel=1e-4)

    def test_exponential_schedule_high_equals_low(self):
        """When high == low for exponential, all values should be equal."""
        out = _make_schedule(5, "exponential", high=2.0, low=2.0)
        assert torch.allclose(out, torch.full((5,), 2.0))

    def test_linear_schedule_high_equals_low(self):
        """When high == low for linear, all values should be equal."""
        out = _make_schedule(5, "linear", high=3.0, low=3.0)
        assert torch.allclose(out, torch.full((5,), 3.0))

    def test_zipfian_equal_pmin_pmax(self):
        """When p_min == p_max, all probs should be equal."""
        out = _compute_firing_probs(10, "zipfian", p_min=0.05, p_max=0.05)
        # q_max != q_min (different rank positions), but mapped range is [0.05, 0.05]
        # Actually with p_min == p_max, result = p_min + 0 = p_min for all
        assert torch.allclose(out, torch.full((10,), 0.05))

    def test_folded_normal_schedule_mu_zero(self):
        """Folded normal with mu=0 is a half-normal distribution."""
        gen = torch.Generator()
        gen.manual_seed(42)
        out = _make_schedule(
            100_000, "folded_normal", folded_mu=0.0, folded_sigma=1.0, generator=gen
        )
        assert (out >= 0).all()
        # Half-normal mean = sigma * sqrt(2/pi) ~ 0.7979
        expected_mean = math.sqrt(2.0 / math.pi)
        assert out.mean().item() == pytest.approx(expected_mean, rel=0.05)
