"""Pytest tests for SparseUniform and SparseExponential."""

import pytest
import torch
from torch import Tensor
from ..sparse import SparseUniform, SparseExponential


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


class TestSparseUniformBasic:
    def test_sample_shape(self, seeded_generator):
        dist = SparseUniform(n_features=10, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_values_in_unit_interval(self, seeded_generator):
        dist = SparseUniform(n_features=20, p_active=0.8, generator=seeded_generator)
        samples = dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_sparse_has_zeros(self, seeded_generator):
        """With p_active < 1, we should see zeros."""
        dist = SparseUniform(n_features=50, p_active=0.3, generator=seeded_generator)
        samples = dist.sample(1000)
        assert (samples == 0).any(), "Expected some zeros with p_active=0.3"

    def test_p_active_zero_all_zeros(self, seeded_generator):
        dist = SparseUniform(n_features=10, p_active=0.0, generator=seeded_generator)
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_p_active_one_no_zeros(self, seeded_generator):
        """With p_active=1, all values should be positive (w.h.p.)."""
        dist = SparseUniform(n_features=10, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(1000)
        # Probability of a single 0 is vanishingly small for Uniform[0,1]
        # but mask guarantees activation, so check activation rate
        active = (samples > 0).float().mean()
        assert active > 0.99


class TestSparseUniformActivationRate:
    @pytest.mark.parametrize("p_active", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_empirical_activation_matches_p_active(self, p_active, seeded_generator):
        dist = SparseUniform(
            n_features=100, p_active=p_active, generator=seeded_generator
        )
        samples = dist.sample(10000)
        empirical = (samples > 0).float().mean().item()
        assert (
            abs(empirical - p_active) < 0.02
        ), f"Expected ~{p_active}, got {empirical}"


class TestSparseUniformBroadcasting:
    def test_scalar_p_active(self, seeded_generator):
        dist = SparseUniform(n_features=5, p_active=0.5, generator=seeded_generator)
        assert dist.p_active.shape == (5,)
        assert (dist.p_active == 0.5).all()

    def test_list_p_active(self, seeded_generator):
        p_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        dist = SparseUniform(n_features=5, p_active=p_list, generator=seeded_generator)
        assert dist.p_active.shape == (5,)
        assert torch.allclose(dist.p_active, torch.tensor(p_list))

    def test_tensor_p_active(self, seeded_generator):
        p_tensor = torch.tensor([0.2, 0.4, 0.6, 0.8])
        dist = SparseUniform(
            n_features=4, p_active=p_tensor, generator=seeded_generator
        )
        assert torch.equal(dist.p_active, p_tensor)

    def test_per_feature_activation_rates(self, seeded_generator):
        """Each feature should have its own activation rate."""
        p_list = [0.1, 0.5, 0.9]
        dist = SparseUniform(n_features=3, p_active=p_list, generator=seeded_generator)
        samples = dist.sample(10000)

        for i, p in enumerate(p_list):
            empirical = (samples[:, i] > 0).float().mean().item()
            assert (
                abs(empirical - p) < 0.03
            ), f"Feature {i}: expected ~{p}, got {empirical}"


class TestSparseUniformReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)

        dist1 = SparseUniform(n_features=20, p_active=0.5, generator=gen1)
        dist2 = SparseUniform(n_features=20, p_active=0.5, generator=gen2)

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestSparseExponentialBasic:
    def test_sample_shape(self, seeded_generator):
        dist = SparseExponential(
            n_features=10, p_active=0.5, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_values_non_negative(self, seeded_generator):
        dist = SparseExponential(
            n_features=20, p_active=0.8, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert samples.min() >= 0.0

    def test_sparse_has_zeros(self, seeded_generator):
        dist = SparseExponential(
            n_features=50, p_active=0.3, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert (samples == 0).any(), "Expected some zeros with p_active=0.3"

    def test_p_active_zero_all_zeros(self, seeded_generator):
        dist = SparseExponential(
            n_features=10, p_active=0.0, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert (samples == 0).all()


class TestSparseExponentialActivationRate:
    @pytest.mark.parametrize("p_active", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_empirical_activation_matches_p_active(self, p_active, seeded_generator):
        dist = SparseExponential(
            n_features=100, p_active=p_active, generator=seeded_generator
        )
        samples = dist.sample(10000)
        empirical = (samples > 0).float().mean().item()
        assert (
            abs(empirical - p_active) < 0.02
        ), f"Expected ~{p_active}, got {empirical}"


class TestSparseExponentialScale:
    def test_scale_affects_mean(self, seeded_generator):
        """Mean of Exp(scale) is 1/scale. Check active values have correct mean."""
        scale = 2.0
        dist = SparseExponential(
            n_features=100, p_active=1.0, scale=scale, generator=seeded_generator
        )
        samples = dist.sample(10000)

        # All values active since p_active=1
        empirical_mean = samples.mean().item()
        expected_mean = 1.0 / scale
        assert (
            abs(empirical_mean - expected_mean) < 0.05
        ), f"Expected mean ~{expected_mean}, got {empirical_mean}"

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
    def test_scale_parameter_values(self, scale, seeded_generator):
        dist = SparseExponential(
            n_features=50, p_active=1.0, scale=scale, generator=seeded_generator
        )
        samples = dist.sample(20000)

        empirical_mean = samples.mean().item()
        expected_mean = 1.0 / scale
        # Allow 10% relative error
        assert abs(empirical_mean - expected_mean) / expected_mean < 0.1

    def test_default_scale_is_one(self, seeded_generator):
        dist = SparseExponential(
            n_features=10, p_active=0.5, generator=seeded_generator
        )
        assert dist.scale == 1.0


class TestSparseExponentialDistribution:
    def test_exponential_distribution_shape(self, seeded_generator):
        """Active values should follow exponential distribution (check variance)."""
        scale = 1.0
        dist = SparseExponential(
            n_features=100, p_active=1.0, scale=scale, generator=seeded_generator
        )
        samples = dist.sample(10000).flatten()

        # Variance of Exp(scale) is 1/scale^2
        expected_var = 1.0 / (scale**2)
        empirical_var = samples.var().item()
        assert abs(empirical_var - expected_var) / expected_var < 0.1

    def test_values_can_exceed_one(self, seeded_generator):
        """Unlike Uniform, Exponential can produce values > 1."""
        dist = SparseExponential(
            n_features=100, p_active=1.0, scale=0.5, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert (
            samples.max() > 1.0
        ), "Exponential with scale=0.5 should have mean=2, expect values > 1"


class TestSparseExponentialReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)

        dist1 = SparseExponential(
            n_features=20, p_active=0.5, scale=1.5, generator=gen1
        )
        dist2 = SparseExponential(
            n_features=20, p_active=0.5, scale=1.5, generator=gen2
        )

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestDistributionComparison:
    def test_same_sparsity_pattern_different_values(self, seeded_generator):
        """Uniform and Exponential with same p_active should have similar sparsity."""
        p = 0.4
        n_samples = 5000

        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        dist_uniform = SparseUniform(n_features=50, p_active=p, generator=gen1)
        dist_exp = SparseExponential(n_features=50, p_active=p, generator=gen2)

        samples_u = dist_uniform.sample(n_samples)
        samples_e = dist_exp.sample(n_samples)

        sparsity_u = (samples_u > 0).float().mean().item()
        sparsity_e = (samples_e > 0).float().mean().item()

        # Both should be close to p
        assert abs(sparsity_u - p) < 0.02
        assert abs(sparsity_e - p) < 0.02
