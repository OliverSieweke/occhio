"""Pytest tests for CorrelatedPairs and AnticorrelatedPairs."""

import pytest
import torch
from ..correlated import CorrelatedPairs, AnticorrelatedPairs


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


class TestCorrelatedPairsBasic:
    def test_sample_shape(self, seeded_generator):
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.5, correlation=0.5, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 20)

    def test_values_in_unit_interval(self, seeded_generator):
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.8, correlation=0.7, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_requires_even_features(self):
        with pytest.raises(AssertionError, match="even"):
            CorrelatedPairs(n_features=11, sparsity=0.5)

    def test_sparse_has_zeros(self, seeded_generator):
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.3, correlation=0.5, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert (samples == 0).any()


class TestCorrelatedPairsHierarchy:
    def test_secondary_inactive_when_primary_inactive(self, seeded_generator):
        """If primary (even index) is inactive, secondary (odd index) must be inactive."""
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.5, correlation=0.8, generator=seeded_generator
        )
        samples = dist.sample(5000)
        
        for i in range(0, 20, 2):
            primary_inactive = samples[:, i] == 0
            secondary_active = samples[:, i + 1] > 0
            violations = (primary_inactive & secondary_active).sum().item()
            assert violations == 0, f"Pair {i//2}: secondary active without primary"

    def test_correlation_one_means_always_paired(self, seeded_generator):
        """With correlation=1, secondary always fires when primary fires."""
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.5, correlation=1.0, generator=seeded_generator
        )
        samples = dist.sample(5000)
        
        for i in range(0, 20, 2):
            primary_active = samples[:, i] > 0
            secondary_active = samples[:, i + 1] > 0
            # When primary active, secondary should be active
            mismatches = (primary_active & ~secondary_active).sum().item()
            assert mismatches == 0, f"Pair {i//2}: secondary should fire with primary"

    def test_correlation_zero_means_secondary_never_fires(self, seeded_generator):
        """With correlation=0, secondary never fires."""
        dist = CorrelatedPairs(
            n_features=20, sparsity=0.5, correlation=0.0, generator=seeded_generator
        )
        samples = dist.sample(1000)
        
        secondary_active = (samples[:, 1::2] > 0).any()
        assert not secondary_active, "Secondary features should never fire with correlation=0"


class TestCorrelatedPairsActivationRates:
    def test_primary_activation_rate(self, seeded_generator):
        """Primary features should have activation rate = sparsity."""
        sparsity = 0.4
        dist = CorrelatedPairs(
            n_features=100, sparsity=sparsity, correlation=0.5, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        primary_rate = (samples[:, 0::2] > 0).float().mean().item()
        assert abs(primary_rate - sparsity) < 0.02, \
            f"Primary rate {primary_rate} should be ~{sparsity}"

    def test_secondary_activation_rate(self, seeded_generator):
        """Secondary features should have activation rate = sparsity * correlation."""
        sparsity = 0.5
        correlation = 0.6
        dist = CorrelatedPairs(
            n_features=100, sparsity=sparsity, correlation=correlation, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        expected = sparsity * correlation
        secondary_rate = (samples[:, 1::2] > 0).float().mean().item()
        assert abs(secondary_rate - expected) < 0.02, \
            f"Secondary rate {secondary_rate} should be ~{expected}"

    @pytest.mark.parametrize("sparsity,correlation", [
        (0.3, 0.2),
        (0.5, 0.5),
        (0.7, 0.8),
        (0.9, 0.9),
    ])
    def test_activation_rates_parametrized(self, sparsity, correlation, seeded_generator):
        dist = CorrelatedPairs(
            n_features=100, sparsity=sparsity, correlation=correlation, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        primary_rate = (samples[:, 0::2] > 0).float().mean().item()
        secondary_rate = (samples[:, 1::2] > 0).float().mean().item()
        
        assert abs(primary_rate - sparsity) < 0.03
        assert abs(secondary_rate - sparsity * correlation) < 0.03


class TestCorrelatedPairsConditionalRate:
    def test_conditional_correlation(self, seeded_generator):
        """P(secondary | primary) should equal correlation parameter."""
        sparsity = 0.5
        correlation = 0.7
        dist = CorrelatedPairs(
            n_features=100, sparsity=sparsity, correlation=correlation, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        primary_active = samples[:, 0::2] > 0
        secondary_active = samples[:, 1::2] > 0
        
        # P(secondary | primary) = (secondary & primary).sum() / primary.sum()
        joint = (primary_active & secondary_active).float().sum().item()
        primary_total = primary_active.float().sum().item()
        
        conditional = joint / primary_total
        assert abs(conditional - correlation) < 0.03, \
            f"P(secondary|primary) = {conditional}, expected {correlation}"


class TestCorrelatedPairsReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        
        dist1 = CorrelatedPairs(n_features=20, sparsity=0.5, correlation=0.6, generator=gen1)
        dist2 = CorrelatedPairs(n_features=20, sparsity=0.5, correlation=0.6, generator=gen2)
        
        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestAnticorrelatedPairsBasic:
    def test_sample_shape(self, seeded_generator):
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.5, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 20)

    def test_values_in_unit_interval(self, seeded_generator):
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.8, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_requires_even_features(self):
        with pytest.raises(AssertionError, match="even"):
            AnticorrelatedPairs(n_features=11, sparsity=0.5)

    def test_sparse_has_zeros(self, seeded_generator):
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.3, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert (samples == 0).any()


class TestAnticorrelatedPairsMutualExclusion:
    def test_at_most_one_active_per_pair(self, seeded_generator):
        """Both features in a pair should never be simultaneously active."""
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.8, generator=seeded_generator
        )
        samples = dist.sample(5000)
        
        for i in range(0, 20, 2):
            both_active = (samples[:, i] > 0) & (samples[:, i + 1] > 0)
            violations = both_active.sum().item()
            assert violations == 0, f"Pair {i//2}: both features active in {violations} samples"

    def test_mutual_exclusion_high_sparsity(self, seeded_generator):
        """Even with high sparsity, pairs should be exclusive."""
        dist = AnticorrelatedPairs(
            n_features=100, sparsity=0.95, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        even_active = samples[:, 0::2] > 0
        odd_active = samples[:, 1::2] > 0
        both_active = (even_active & odd_active).sum().item()
        assert both_active == 0


class TestAnticorrelatedPairsActivationRates:
    def test_overall_activation_rate(self, seeded_generator):
        """Overall activation should be ~sparsity (one per pair fires)."""
        sparsity = 0.5
        dist = AnticorrelatedPairs(
            n_features=100, sparsity=sparsity, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        # Each pair contributes 1 active feature when active, 0 otherwise
        # Expected total = n_pairs * sparsity = n_features/2 * sparsity
        # Expected rate = sparsity / 2 (since n_features total)
        overall_rate = (samples > 0).float().mean().item()
        expected = sparsity / 2
        assert abs(overall_rate - expected) < 0.02, \
            f"Overall rate {overall_rate} should be ~{expected}"

    def test_per_feature_activation_rate(self, seeded_generator):
        """Each feature should have activation rate = sparsity / 2."""
        sparsity = 0.6
        dist = AnticorrelatedPairs(
            n_features=100, sparsity=sparsity, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        expected = sparsity / 2
        for i in range(100):
            rate = (samples[:, i] > 0).float().mean().item()
            assert abs(rate - expected) < 0.05, \
                f"Feature {i} rate {rate} should be ~{expected}"

    def test_even_odd_symmetry(self, seeded_generator):
        """Even and odd features should have equal marginal rates."""
        sparsity = 0.5
        dist = AnticorrelatedPairs(
            n_features=100, sparsity=sparsity, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        even_rate = (samples[:, 0::2] > 0).float().mean().item()
        odd_rate = (samples[:, 1::2] > 0).float().mean().item()
        
        assert abs(even_rate - odd_rate) < 0.02, \
            f"Even rate {even_rate} vs odd rate {odd_rate} should be symmetric"

    @pytest.mark.parametrize("sparsity", [0.2, 0.4, 0.6, 0.8])
    def test_activation_rates_parametrized(self, sparsity, seeded_generator):
        dist = AnticorrelatedPairs(
            n_features=100, sparsity=sparsity, generator=seeded_generator
        )
        samples = dist.sample(10000)
        
        overall_rate = (samples > 0).float().mean().item()
        expected = sparsity / 2
        assert abs(overall_rate - expected) < 0.02


class TestAnticorrelatedPairsEdgeCases:
    def test_sparsity_zero_all_zeros(self, seeded_generator):
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.0, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_sparsity_one_exactly_one_per_pair(self, seeded_generator):
        """With sparsity=1, exactly one feature per pair should be active."""
        dist = AnticorrelatedPairs(
            n_features=20, sparsity=1.0, generator=seeded_generator
        )
        samples = dist.sample(1000)
        
        for i in range(0, 20, 2):
            pair_sum = (samples[:, i] > 0).int() + (samples[:, i + 1] > 0).int()
            assert (pair_sum == 1).all(), f"Pair {i//2} should have exactly one active"


class TestAnticorrelatedPairsReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        
        dist1 = AnticorrelatedPairs(n_features=20, sparsity=0.5, generator=gen1)
        dist2 = AnticorrelatedPairs(n_features=20, sparsity=0.5, generator=gen2)
        
        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestCorrelationContrast:
    def test_correlated_vs_anticorrelated_structure(self, seeded_generator):
        """Correlated pairs can have both active; anticorrelated cannot."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        
        corr_dist = CorrelatedPairs(
            n_features=20, sparsity=0.8, correlation=0.9, generator=gen1
        )
        anti_dist = AnticorrelatedPairs(
            n_features=20, sparsity=0.8, generator=gen2
        )
        
        corr_samples = corr_dist.sample(1000)
        anti_samples = anti_dist.sample(1000)
        
        # Correlated: some pairs should have both active
        corr_both = 0
        for i in range(0, 20, 2):
            corr_both += ((corr_samples[:, i] > 0) & (corr_samples[:, i + 1] > 0)).sum().item()
        
        # Anticorrelated: no pairs should have both active
        anti_both = 0
        for i in range(0, 20, 2):
            anti_both += ((anti_samples[:, i] > 0) & (anti_samples[:, i + 1] > 0)).sum().item()
        
        assert corr_both > 0, "Correlated should have some joint activations"
        assert anti_both == 0, "Anticorrelated should have no joint activations"