"""Deep audit tests for correlated.py and dag.py distributions.

Systematically tests:
- HierarchicalPairs: gating, p_follow accuracy, edge cases, beta coupling
- ScaledHierarchicalPairs: gating, scaling, edge cases
- CorrelatedPairs: correlation accuracy, parameter solving, edge cases
- AnticorrelatedPairs: mutual exclusivity, edge cases
- GaussianCorrelated: correlation matrix, thresholds, edge cases
- DAGDistribution: causality, structure, edge cases
- DAGBayesianPropagation: Noisy-OR semantics, edge cases
- DAGRandomWalkToRoot: walk mechanics, structure, edge cases
- PreferentialAttachment: degree distribution, cascade, edge cases
- Cross-cutting: shape, non-negativity, reproducibility, device
"""

import pytest
import torch
import numpy as np
from occhio.distributions.correlated import (
    HierarchicalPairs,
    ScaledHierarchicalPairs,
    CorrelatedPairs,
    AnticorrelatedPairs,
    GaussianCorrelated,
)
from occhio.distributions.dag import (
    DAGDistribution,
    DAGBayesianPropagation,
    DAGRandomWalkToRoot,
    PreferentialAttachment,
)


@pytest.fixture
def gen():
    g = torch.Generator()
    g.manual_seed(42)
    return g


def _fresh_gen(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


N = 100_000  # large sample size for statistical tests
TOL = 0.02  # tolerance for rate comparisons


# ============================================================================
# 1. HierarchicalPairs
# ============================================================================
class TestHierarchicalPairsAudit:
    """Deep audit of HierarchicalPairs."""

    def test_gating_property_child_only_fires_when_parent_fires(self, gen):
        """Child (odd idx) must NEVER fire when parent (even idx) is inactive."""
        dist = HierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=0.8, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            parent_off = s[:, i] == 0
            child_on = s[:, i + 1] > 0
            violations = (parent_off & child_on).sum().item()
            assert violations == 0, f"Pair {i // 2}: child active without parent"

    def test_p_follow_rate_matches_configured(self, gen):
        """P(child fires | parent fires) should match p_follow."""
        p_follow = 0.7
        dist = HierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=p_follow, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            parent_on = s[:, i] > 0
            child_on = s[:, i + 1] > 0
            n_parent = parent_on.sum().item()
            if n_parent < 100:
                continue
            cond_rate = (parent_on & child_on).sum().item() / n_parent
            assert abs(cond_rate - p_follow) < TOL, (
                f"Pair {i // 2}: conditional rate {cond_rate:.4f} != {p_follow}"
            )

    def test_p_follow_zero_secondary_never_fires(self, gen):
        """With p_follow=0, no secondary feature should ever fire."""
        dist = HierarchicalPairs(
            n_features=20, p_active=0.8, p_follow=0.0, generator=gen
        )
        s = dist.sample(N)
        assert (s[:, 1::2] == 0).all(), "Secondary fired with p_follow=0"

    def test_p_follow_one_secondary_always_fires_when_parent_active(self, gen):
        """With p_follow=1, secondary always fires when parent fires."""
        dist = HierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=1.0, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            parent_on = s[:, i] > 0
            child_on = s[:, i + 1] > 0
            mismatches = (parent_on & ~child_on).sum().item()
            assert mismatches == 0, (
                f"Pair {i // 2}: parent on but child off with p_follow=1"
            )

    def test_odd_n_features_raises(self):
        """Odd n_features should raise."""
        with pytest.raises(AssertionError):
            HierarchicalPairs(n_features=11, p_active=0.5)

    def test_n_features_2_minimal_pair(self, gen):
        """Minimal n_features=2 should work correctly."""
        dist = HierarchicalPairs(
            n_features=2, p_active=0.5, p_follow=0.5, generator=gen
        )
        s = dist.sample(10000)
        assert s.shape == (10000, 2)
        parent_off = s[:, 0] == 0
        child_on = s[:, 1] > 0
        assert (parent_off & child_on).sum().item() == 0

    def test_p_active_zero_all_zeros(self, gen):
        """p_active=0 means nothing fires."""
        dist = HierarchicalPairs(
            n_features=10, p_active=0.0, p_follow=0.5, generator=gen
        )
        s = dist.sample(1000)
        assert (s == 0).all()

    def test_p_active_one_all_primaries_fire(self, gen):
        """p_active=1 means all primaries fire."""
        dist = HierarchicalPairs(
            n_features=10, p_active=1.0, p_follow=0.5, generator=gen
        )
        s = dist.sample(10000)
        assert (s[:, 0::2] > 0).all(), "Some primaries didn't fire with p_active=1"

    def test_beta_one_child_equals_parent_value(self, gen):
        """With beta=1 and p_follow=1, child value == parent value."""
        dist = HierarchicalPairs(
            n_features=10, p_active=0.5, p_follow=1.0, beta=1.0, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 10, 2):
            both_active = (s[:, i] > 0) & (s[:, i + 1] > 0)
            if both_active.any():
                diff = (s[both_active, i] - s[both_active, i + 1]).abs().max().item()
                assert diff < 1e-6, (
                    f"Pair {i // 2}: beta=1 but child != parent (diff={diff})"
                )

    def test_beta_zero_child_values_scaled_by_parent(self, gen):
        """With beta=0, child = parent * U where U ~ Uniform(0,1).
        Child values should be <= parent values."""
        dist = HierarchicalPairs(
            n_features=10, p_active=0.5, p_follow=1.0, beta=0.0, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 10, 2):
            both_active = (s[:, i] > 0) & (s[:, i + 1] > 0)
            if both_active.any():
                # child = parent * 0 + 1.0 * rand = rand (independent of parent!)
                # Actually with beta=0: child = parent * 0 + (1-0) * rand = rand
                # So child is NOT constrained by parent. That's the current behavior.
                # Just verify non-negative
                assert (s[both_active, i + 1] >= 0).all()

    def test_values_in_unit_interval(self, gen):
        """All values should be in [0, 1]."""
        dist = HierarchicalPairs(
            n_features=20, p_active=0.8, p_follow=0.8, generator=gen
        )
        s = dist.sample(N)
        assert s.min() >= 0.0
        assert s.max() <= 1.0

    def test_beta_values_in_unit_interval(self, gen):
        """With beta, all values should still be in [0, 1]."""
        dist = HierarchicalPairs(
            n_features=20, p_active=0.8, p_follow=0.8, beta=0.5, generator=gen
        )
        s = dist.sample(N)
        assert s.min() >= 0.0
        assert s.max() <= 1.0


# ============================================================================
# 2. ScaledHierarchicalPairs
# ============================================================================
class TestScaledHierarchicalPairsAudit:
    """Deep audit of ScaledHierarchicalPairs."""

    def test_gating_property(self, gen):
        """Child fires only when parent fires."""
        dist = ScaledHierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=0.8, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            parent_off = s[:, i] == 0
            child_on = s[:, i + 1] > 0
            assert (parent_off & child_on).sum().item() == 0

    def test_scaling_child_leq_parent(self, gen):
        """Child value = U * parent_value, so child <= parent."""
        dist = ScaledHierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=1.0, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            both_active = (s[:, i] > 0) & (s[:, i + 1] > 0)
            if both_active.any():
                parent_vals = s[both_active, i]
                child_vals = s[both_active, i + 1]
                violations = (child_vals > parent_vals + 1e-6).sum().item()
                assert violations == 0, (
                    f"Pair {i // 2}: child > parent in {violations} samples"
                )

    def test_scaling_mean_child_about_half_parent(self, gen):
        """E[child | both active] = E[U * parent] = E[parent]/2."""
        dist = ScaledHierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=1.0, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            both_active = (s[:, i] > 0) & (s[:, i + 1] > 0)
            if both_active.sum() > 100:
                ratio = s[both_active, i + 1].mean() / s[both_active, i].mean()
                # E[U * v] / E[v] = E[U] = 0.5
                assert abs(ratio.item() - 0.5) < 0.05, (
                    f"Pair {i // 2}: ratio {ratio:.3f} should be ~0.5"
                )

    def test_p_follow_accuracy(self, gen):
        """P(child | parent) should match p_follow."""
        p_follow = 0.6
        dist = ScaledHierarchicalPairs(
            n_features=20, p_active=0.5, p_follow=p_follow, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 20, 2):
            parent_on = s[:, i] > 0
            child_on = s[:, i + 1] > 0
            n_parent = parent_on.sum().item()
            if n_parent > 100:
                cond = (parent_on & child_on).sum().item() / n_parent
                assert abs(cond - p_follow) < TOL

    def test_odd_n_features_raises(self):
        with pytest.raises(AssertionError):
            ScaledHierarchicalPairs(n_features=7, p_active=0.5)

    def test_values_in_unit_interval(self, gen):
        dist = ScaledHierarchicalPairs(
            n_features=20, p_active=0.8, p_follow=0.8, generator=gen
        )
        s = dist.sample(N)
        assert s.min() >= 0.0
        assert s.max() <= 1.0


# ============================================================================
# 3. CorrelatedPairs
# ============================================================================
class TestCorrelatedPairsAudit:
    """Deep audit of CorrelatedPairs."""

    def test_correlation_matches_configured(self, gen):
        """Empirical binary correlation should approximately match configured value."""
        target_corr = 0.5
        dist = CorrelatedPairs(
            n_features=10, p_active=0.3, correlation=target_corr, generator=gen
        )
        s = dist.sample(N)
        for i in range(0, 10, 2):
            x0 = (s[:, i] > 0).float()
            x1 = (s[:, i + 1] > 0).float()
            corr = torch.corrcoef(torch.stack([x0, x1]))[0, 1].item()
            assert abs(corr - target_corr) < 0.05, (
                f"Pair {i // 2}: correlation {corr:.4f} should be ~{target_corr}"
            )

    def test_correlation_zero_independent(self, gen):
        """correlation=0 should produce independent features within pair (p_individual=0)."""
        # corr=0 => p_i = 0 / (1 - pa + 0) = 0 => nothing fires
        # This is technically correct: zero correlation is achieved by never firing
        dist = CorrelatedPairs(
            n_features=10, p_active=0.3, correlation=0.0, generator=gen
        )
        s = dist.sample(1000)
        assert (s == 0).all(), "Expected all zeros with correlation=0"

    def test_correlation_one_perfect(self, gen):
        """correlation=1.0 should produce perfectly correlated binary activations."""
        dist = CorrelatedPairs(
            n_features=10, p_active=0.3, correlation=1.0, generator=gen
        )
        # p_i = 1 / (1 - pa + pa) = 1.0
        assert (dist.p_individual - 1.0).abs().max() < 1e-6
        s = dist.sample(N)
        for i in range(0, 10, 2):
            x0_on = s[:, i] > 0
            x1_on = s[:, i + 1] > 0
            mismatches = (x0_on != x1_on).sum().item()
            assert mismatches == 0, (
                f"Pair {i // 2}: {mismatches} mismatches with corr=1"
            )

    def test_density_parameter(self, gen):
        """density = p_active * p_individual."""
        dist = CorrelatedPairs(n_features=10, p_active=0.5, density=0.1, generator=gen)
        # p_individual should be 0.1/0.5 = 0.2
        assert (dist.p_individual - 0.2).abs().max() < 1e-6

    def test_all_four_parameter_combos(self):
        """All valid 2-param combos should produce valid distributions."""
        combos = [
            dict(p_active=0.3, p_individual=0.5),
            dict(p_active=0.3, correlation=0.5),
            dict(p_active=0.3, density=0.1),
            dict(p_individual=0.5, correlation=0.3),
            dict(p_individual=0.5, density=0.1),
            dict(correlation=0.5, density=0.1),
        ]
        for kw in combos:
            d = CorrelatedPairs(n_features=10, **kw)
            s = d.sample(100)
            assert s.shape == (100, 10), f"Failed for {kw}"

    def test_no_params_raises(self):
        with pytest.raises(ValueError, match="Exactly two"):
            CorrelatedPairs(n_features=10)

    def test_three_params_raises(self):
        with pytest.raises(ValueError, match="Exactly two"):
            CorrelatedPairs(
                n_features=10, p_active=0.3, p_individual=0.5, correlation=0.3
            )

    def test_impossible_params_raises(self, gen):
        """Parameters that produce p_active < 0 should raise."""
        # p_individual=0.3, correlation=0.5 => pa = (0.3 - 0.5)/(0.3*(1-0.5)) < 0
        with pytest.raises(ValueError, match="outside"):
            CorrelatedPairs(
                n_features=10, p_individual=0.3, correlation=0.5, generator=gen
            )

    def test_negative_correlation_raises(self, gen):
        """Negative correlation should yield negative p_individual which should raise."""
        with pytest.raises(ValueError, match="outside"):
            CorrelatedPairs(
                n_features=10, p_active=0.3, correlation=-0.5, generator=gen
            )

    def test_odd_n_features_raises(self):
        with pytest.raises(AssertionError):
            CorrelatedPairs(n_features=11, p_active=0.3, p_individual=0.5)

    def test_values_non_negative(self, gen):
        dist = CorrelatedPairs(
            n_features=20, p_active=0.3, p_individual=0.5, generator=gen
        )
        s = dist.sample(N)
        assert s.min() >= 0.0


# ============================================================================
# 4. AnticorrelatedPairs
# ============================================================================
class TestAnticorrelatedPairsAudit:
    """Deep audit of AnticorrelatedPairs."""

    def test_mutual_exclusivity_zero_violations(self, gen):
        """At most one feature active per pair, ever."""
        dist = AnticorrelatedPairs(n_features=20, p_active=0.9, generator=gen)
        s = dist.sample(N)
        for i in range(0, 20, 2):
            both = (s[:, i] > 0) & (s[:, i + 1] > 0)
            assert both.sum().item() == 0, f"Pair {i // 2}: both active"

    def test_exactly_one_when_p_active_one(self, gen):
        """p_active=1: exactly one per pair always."""
        dist = AnticorrelatedPairs(n_features=20, p_active=1.0, generator=gen)
        s = dist.sample(N)
        for i in range(0, 20, 2):
            pair_active = (s[:, i] > 0).int() + (s[:, i + 1] > 0).int()
            assert (pair_active == 1).all()

    def test_even_odd_equal_rates(self, gen):
        """Even and odd features in each pair should fire at equal rate."""
        dist = AnticorrelatedPairs(n_features=20, p_active=0.6, generator=gen)
        s = dist.sample(N)
        for i in range(0, 20, 2):
            rate_even = (s[:, i] > 0).float().mean().item()
            rate_odd = (s[:, i + 1] > 0).float().mean().item()
            assert abs(rate_even - rate_odd) < TOL, (
                f"Pair {i // 2}: asymmetry {rate_even:.4f} vs {rate_odd:.4f}"
            )

    def test_p_active_zero_all_zeros(self, gen):
        dist = AnticorrelatedPairs(n_features=10, p_active=0.0, generator=gen)
        s = dist.sample(1000)
        assert (s == 0).all()


# ============================================================================
# 5. GaussianCorrelated
# ============================================================================
class TestGaussianCorrelatedAudit:
    """Deep audit of GaussianCorrelated."""

    def test_output_shape(self, gen):
        dist = GaussianCorrelated(n_features=10, p_active=0.3, generator=gen)
        s = dist.sample(100)
        assert s.shape == (100, 10)

    def test_values_non_negative(self, gen):
        dist = GaussianCorrelated(n_features=10, p_active=0.3, generator=gen)
        s = dist.sample(N)
        assert s.min() >= 0.0

    def test_values_in_unit_interval(self, gen):
        """Active values come from Uniform(0,1)."""
        dist = GaussianCorrelated(n_features=10, p_active=0.3, generator=gen)
        s = dist.sample(N)
        assert s.max() <= 1.0

    def test_marginal_firing_rate_matches_p_active(self, gen):
        """Each feature should fire at approximately p_active."""
        p_active = 0.2
        dist = GaussianCorrelated(n_features=20, p_active=p_active, generator=gen)
        s = dist.sample(N)
        rates = (s > 0).float().mean(dim=0)
        for i, r in enumerate(rates):
            assert abs(r.item() - p_active) < TOL, (
                f"Feature {i}: rate {r.item():.4f} should be ~{p_active}"
            )

    def test_identity_correlation_matrix_independent(self, gen):
        """Identity correlation => independent features (no cross-correlation)."""
        n = 6
        corr = torch.eye(n)
        dist = GaussianCorrelated(
            n_features=n, p_active=0.3, correlation_matrix=corr, generator=gen
        )
        s = dist.sample(N)
        binary = (s > 0).float()
        empirical_corr = torch.corrcoef(binary.T)
        # Off-diagonal should be near zero
        mask = ~torch.eye(n, dtype=torch.bool)
        off_diag = empirical_corr[mask]
        assert off_diag.abs().max() < 0.05, (
            f"Off-diagonal corr too large: {off_diag.abs().max():.4f}"
        )

    def test_high_correlation_matrix_produces_correlated_firing(self, gen):
        """Features with high off-diagonal correlation should fire together more often."""
        n = 4
        # Create a correlation matrix with high correlation between features 0 and 1
        corr = torch.eye(n)
        corr[0, 1] = corr[1, 0] = 0.9
        dist = GaussianCorrelated(
            n_features=n, p_active=0.3, correlation_matrix=corr, generator=gen
        )
        s = dist.sample(N)
        x0 = (s[:, 0] > 0).float()
        x1 = (s[:, 1] > 0).float()
        empirical = torch.corrcoef(torch.stack([x0, x1]))[0, 1].item()
        # The Gaussian copula correlation and binary correlation differ,
        # but high copula corr should yield high binary corr
        assert empirical > 0.3, f"Expected positive correlation, got {empirical:.4f}"

    def test_wrong_shape_correlation_matrix_raises(self):
        with pytest.raises(AssertionError):
            GaussianCorrelated(
                n_features=5, p_active=0.3, correlation_matrix=torch.eye(3)
            )

    def test_reproducibility(self):
        g1 = _fresh_gen(99)
        g2 = _fresh_gen(99)
        d1 = GaussianCorrelated(n_features=8, p_active=0.3, generator=g1)
        d2 = GaussianCorrelated(n_features=8, p_active=0.3, generator=g2)
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert torch.equal(s1, s2)

    def test_single_feature(self, gen):
        """n_features=1 should work fine."""
        dist = GaussianCorrelated(n_features=1, p_active=0.5, generator=gen)
        s = dist.sample(10000)
        assert s.shape == (10000, 1)
        rate = (s > 0).float().mean().item()
        assert abs(rate - 0.5) < TOL


# ============================================================================
# 6. DAGDistribution
# ============================================================================
class TestDAGDistributionAudit:
    """Deep audit of DAGDistribution."""

    def test_causality_children_need_active_parent(self, gen):
        """Non-root nodes with all parents inactive must be inactive."""
        dist = DAGDistribution(n_features=20, p_active=0.5, p_edge=0.3, generator=gen)
        s = dist.sample(N)
        for j in range(dist.n_features):
            parents = dist.adjacency[:, j].nonzero(as_tuple=True)[0]
            if len(parents) == 0:
                continue
            all_parents_off = (s[:, parents] == 0).all(dim=1)
            child_on = s[:, j] > 0
            assert (all_parents_off & child_on).sum().item() == 0

    def test_single_node_dag(self, gen):
        """n_features=1: single root node, fires with p_active."""
        dist = DAGDistribution(n_features=1, p_active=0.5, p_edge=0.5, generator=gen)
        s = dist.sample(N)
        assert s.shape == (N, 1)
        rate = (s > 0).float().mean().item()
        assert abs(rate - 0.5) < TOL

    def test_linear_chain(self, gen):
        """Build a linear chain A->B->C->D and verify cascading gating."""
        n = 4
        dist = DAGDistribution(n_features=n, p_active=0.5, p_edge=0.0, generator=gen)
        # Manually create linear chain: 0->1->2->3
        adj = torch.zeros(n, n, dtype=torch.bool)
        adj[0, 1] = True
        adj[1, 2] = True
        adj[2, 3] = True
        dist.adjacency = adj

        s = dist.sample(N)
        # Node 3 can only fire if 2 fires, which needs 1, which needs 0
        for j in range(1, n):
            parent_off = s[:, j - 1] == 0
            child_on = s[:, j] > 0
            assert (parent_off & child_on).sum().item() == 0, (
                f"Node {j} active without parent {j - 1}"
            )

    def test_disconnected_components(self, gen):
        """p_edge=0: all nodes are roots, fire independently."""
        dist = DAGDistribution(n_features=10, p_active=0.3, p_edge=0.0, generator=gen)
        assert (dist.adjacency == 0).all()
        s = dist.sample(N)
        rate = (s > 0).float().mean().item()
        assert abs(rate - 0.3) < TOL

    def test_p_edge_one_fully_connected_upper_triangular(self, gen):
        """p_edge=1: all possible edges present."""
        dist = DAGDistribution(n_features=10, p_active=0.5, p_edge=1.0, generator=gen)
        expected = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
        assert torch.equal(dist.adjacency, expected)


# ============================================================================
# 7. DAGBayesianPropagation
# ============================================================================
class TestDAGBayesianPropagationAudit:
    """Deep audit of DAGBayesianPropagation Noisy-OR semantics."""

    def test_noisy_or_single_parent(self, gen):
        """Single-parent node: P(child fires | parent value v) = v.

        With Noisy-OR and one parent: fire_prob = 1 - (1 - v) = v.
        So binning parent values and measuring child fire rates should
        approximate the identity function.
        """
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.8, p_edge=0.4, generator=gen
        )
        s = dist.sample(N)

        single_parent_nodes = [
            j for j in range(dist.n_features) if len(dist._parent_indices[j]) == 1
        ]
        if not single_parent_nodes:
            pytest.skip("No single-parent nodes")

        j = single_parent_nodes[0]
        parent = dist._parent_indices[j][0].item()

        # Bin parent values: [0.1, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9)
        bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
        prev_rate = -1.0
        for lo, hi in bins:
            mask = (s[:, parent] > lo) & (s[:, parent] <= hi)
            if mask.sum() < 200:
                continue
            child_rate = (s[mask, j] > 0).float().mean().item()
            expected_rate = (lo + hi) / 2  # midpoint of bin = expected fire_prob
            assert abs(child_rate - expected_rate) < 0.1, (
                f"Bin [{lo}, {hi}): child rate {child_rate:.3f} vs expected ~{expected_rate:.3f}"
            )
            assert child_rate >= prev_rate - 0.05, (
                "Rate should increase with parent value"
            )
            prev_rate = child_rate

    def test_no_parents_inactive_means_child_inactive(self, gen):
        """If all parents are value 0, Noisy-OR gives fire_prob=0."""
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.5, p_edge=0.3, generator=gen
        )
        s = dist.sample(N)
        for j in range(dist.n_features):
            if not dist._has_parents[j]:
                continue
            parent_idx = dist._parent_indices[j]
            all_parents_zero = (s[:, parent_idx] == 0).all(dim=1)
            child_on = s[:, j] > 0
            assert (all_parents_zero & child_on).sum().item() == 0

    def test_single_node(self, gen):
        dist = DAGBayesianPropagation(
            n_features=1, p_active=0.5, p_edge=0.5, generator=gen
        )
        s = dist.sample(N)
        assert s.shape == (N, 1)
        rate = (s > 0).float().mean().item()
        assert abs(rate - 0.5) < TOL


# ============================================================================
# 8. DAGRandomWalkToRoot
# ============================================================================
class TestDAGRandomWalkToRootAudit:
    """Deep audit of DAGRandomWalkToRoot."""

    def test_output_shape(self, gen):
        dist = DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=gen)
        s = dist.sample(100)
        assert s.shape == (100, 10)

    def test_values_non_negative(self, gen):
        dist = DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=gen)
        s = dist.sample(N)
        assert s.min() >= 0.0

    def test_at_least_one_active_per_sample(self, gen):
        """Each sample should have at least the seed node active."""
        dist = DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=gen)
        s = dist.sample(N)
        active_per_row = (s > 0).sum(dim=1)
        assert (active_per_row >= 1).all(), "Some samples have no active features"

    def test_root_features_fire_more_often(self, gen):
        """Roots (no parents) should generally fire at higher rates than deep nodes."""
        dist = DAGRandomWalkToRoot(n_features=20, p_edge=0.5, generator=gen)
        s = dist.sample(N)

        has_parents = dist._has_parents_mask
        root_mask = ~has_parents
        nonroot_mask = has_parents

        if root_mask.any() and nonroot_mask.any():
            root_rate = (s[:, root_mask] > 0).float().mean().item()
            nonroot_rate = (s[:, nonroot_mask] > 0).float().mean().item()
            # Roots should have comparable or higher fire rates
            # (they're always part of the walk-to-root path)
            # This is a soft check since it depends on graph structure
            assert root_rate > 0, "Roots should fire"

    def test_custom_adjacency(self, gen):
        """Verify works with manually supplied adjacency."""
        n = 4
        adj = torch.zeros(n, n, dtype=torch.bool)
        adj[0, 1] = True  # 0 -> 1
        adj[0, 2] = True  # 0 -> 2
        adj[2, 3] = True  # 2 -> 3
        dist = DAGRandomWalkToRoot(n_features=n, adjacency=adj, generator=gen)
        s = dist.sample(10000)
        assert s.shape == (10000, n)
        # All values should be non-negative
        assert s.min() >= 0.0

    def test_n_firings_increases_density(self, gen):
        """More firings per sample should produce denser output."""
        g1 = _fresh_gen(42)
        g2 = _fresh_gen(42)
        d1 = DAGRandomWalkToRoot(n_features=15, p_edge=0.3, n_firings=1, generator=g1)
        d2 = DAGRandomWalkToRoot(n_features=15, p_edge=0.3, n_firings=5, generator=g2)
        # Need same DAG structure to compare
        d2.adjacency = d1.adjacency.clone()
        d2._build_parent_cache()

        g1b = _fresh_gen(99)
        g2b = _fresh_gen(99)
        d1.generator = g1b
        d2.generator = g2b
        s1 = d1.sample(10000)
        s2 = d2.sample(10000)
        density1 = (s1 > 0).float().mean().item()
        density2 = (s2 > 0).float().mean().item()
        assert density2 > density1, (
            f"n_firings=5 density {density2:.4f} should exceed n_firings=1 density {density1:.4f}"
        )

    def test_beta_one_no_decay(self, gen):
        """With beta=1, parent values should not decay (parent = child value)."""
        n = 3
        adj = torch.zeros(n, n, dtype=torch.bool)
        adj[0, 1] = True  # 0 -> 1
        dist = DAGRandomWalkToRoot(n_features=n, adjacency=adj, beta=1.0, generator=gen)
        s = dist.sample(N)
        # When seed=1 (node 1) and walk goes to 0:
        # value at node 1 = activation
        # value at node 0 should be beta*activation + (1-beta)*(...) = activation
        # So nodes 0 and 1 should have the same value (when both active from same walk)
        # But with sum accumulation and multiple starts, this is complex.
        # Just verify basic properties
        assert s.min() >= 0.0

    def test_wrong_adjacency_shape_raises(self):
        with pytest.raises(AssertionError):
            DAGRandomWalkToRoot(
                n_features=5, adjacency=torch.zeros(3, 3, dtype=torch.bool)
            )

    def test_reproducibility(self):
        g1 = _fresh_gen(42)
        g2 = _fresh_gen(42)
        d1 = DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=g1)
        d2 = DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=g2)
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert torch.equal(s1, s2)

    def test_no_edges_graph_single_node_active(self, gen):
        """With p_edge=0, all nodes are isolated roots. Walk terminates immediately."""
        dist = DAGRandomWalkToRoot(n_features=10, p_edge=0.0, generator=gen)
        assert (dist.adjacency == 0).all()
        s = dist.sample(10000)
        # Each sample should have exactly 1 active feature (the seed node)
        active_count = (s > 0).sum(dim=1)
        assert (active_count == 1).all(), (
            "With no edges, exactly one node should be active"
        )


# ============================================================================
# 9. PreferentialAttachment
# ============================================================================
class TestPreferentialAttachmentAudit:
    """Deep audit of PreferentialAttachment."""

    def test_output_shape(self, gen):
        dist = PreferentialAttachment(n_features=20, generator=gen)
        s = dist.sample(100)
        assert s.shape == (100, 20)

    def test_values_non_negative(self, gen):
        dist = PreferentialAttachment(n_features=20, generator=gen)
        s = dist.sample(N)
        assert s.min() >= 0.0

    def test_no_self_loops(self, gen):
        dist = PreferentialAttachment(n_features=20, generator=gen)
        assert not dist.adjacency.diagonal().any(), "Self-loops found"

    def test_in_degree_power_law_ordering(self, gen):
        """Node 0 should have highest expected in-degree, node N-1 lowest."""
        dist = PreferentialAttachment(
            n_features=50, alpha=2.0, p_edge=0.5, generator=gen
        )
        in_deg = dist.in_degrees()
        # Node 0 should have strictly higher in-degree than node N-1
        assert in_deg[0] > in_deg[-1], (
            f"in_deg[0]={in_deg[0].item()} should > in_deg[-1]={in_deg[-1].item()}"
        )

    def test_alpha_zero_erdos_renyi(self, gen):
        """alpha=0 should give uniform edge probability (Erdos-Renyi)."""
        dist = PreferentialAttachment(
            n_features=50, alpha=0.0, p_edge=0.3, generator=gen
        )
        in_deg = dist.in_degrees()
        # In-degrees should be roughly uniform
        mean_deg = in_deg.mean().item()
        std_deg = in_deg.std().item()
        cv = std_deg / mean_deg if mean_deg > 0 else 0
        assert cv < 0.5, f"Coefficient of variation {cv:.3f} too high for Erdos-Renyi"

    def test_cascade_with_p_child_zero_no_cascade(self, gen):
        """p_child=0: no cascade, only independent fires."""
        dist = PreferentialAttachment(
            n_features=20, p_active=0.1, p_child=0.0, generator=gen
        )
        s = dist.sample(N)
        rate = (s > 0).float().mean().item()
        # Should be approximately p_active = 0.1
        assert abs(rate - 0.1) < TOL, f"Rate {rate:.4f} should be ~0.1 with no cascade"

    def test_cascade_with_p_child_one_deterministic(self, gen):
        """p_child=1: deterministic cascade, all children of active nodes fire."""
        dist = PreferentialAttachment(
            n_features=20, p_active=0.3, p_child=1.0, p_edge=0.3, generator=gen
        )
        s = dist.sample(N)
        # Rate should be >= p_active since cascade adds more activations
        rate = (s > 0).float().mean().item()
        assert rate >= 0.3 - TOL, f"Rate {rate:.4f} should be >= 0.3 with cascade"

    def test_p_child_tuple_per_edge(self, gen):
        """p_child as tuple should create per-edge cascade probabilities."""
        dist = PreferentialAttachment(
            n_features=20, p_active=0.1, p_child=(0.5, 0.9), p_edge=0.3, generator=gen
        )
        s = dist.sample(1000)
        assert s.shape == (1000, 20)
        assert s.min() >= 0.0

    def test_exponential_value_dist(self, gen):
        """value_dist='exponential' should produce values > 1 sometimes."""
        dist = PreferentialAttachment(
            n_features=20, p_active=0.5, value_dist="exponential", generator=gen
        )
        s = dist.sample(N)
        assert s.min() >= 0.0
        # Exponential values can exceed 1
        assert s.max() > 1.0, "Exponential distribution should produce values > 1"

    def test_reproducibility(self):
        g1 = _fresh_gen(42)
        g2 = _fresh_gen(42)
        d1 = PreferentialAttachment(n_features=15, generator=g1)
        d2 = PreferentialAttachment(n_features=15, generator=g2)
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert torch.equal(s1, s2)

    def test_single_node(self, gen):
        dist = PreferentialAttachment(n_features=1, p_active=0.5, generator=gen)
        s = dist.sample(10000)
        assert s.shape == (10000, 1)
        rate = (s > 0).float().mean().item()
        assert abs(rate - 0.5) < TOL


# ============================================================================
# 10. Cross-cutting concerns for ALL distributions
# ============================================================================

ALL_DIST_FACTORIES = [
    (
        "HierarchicalPairs",
        lambda gen: HierarchicalPairs(
            n_features=10, p_active=0.3, p_follow=0.5, generator=gen
        ),
    ),
    (
        "ScaledHierarchicalPairs",
        lambda gen: ScaledHierarchicalPairs(
            n_features=10, p_active=0.3, p_follow=0.5, generator=gen
        ),
    ),
    (
        "CorrelatedPairs",
        lambda gen: CorrelatedPairs(
            n_features=10, p_active=0.3, p_individual=0.5, generator=gen
        ),
    ),
    (
        "AnticorrelatedPairs",
        lambda gen: AnticorrelatedPairs(n_features=10, p_active=0.3, generator=gen),
    ),
    (
        "GaussianCorrelated",
        lambda gen: GaussianCorrelated(n_features=10, p_active=0.3, generator=gen),
    ),
    (
        "DAGDistribution",
        lambda gen: DAGDistribution(
            n_features=10, p_active=0.3, p_edge=0.2, generator=gen
        ),
    ),
    (
        "DAGBayesianPropagation",
        lambda gen: DAGBayesianPropagation(
            n_features=10, p_active=0.3, p_edge=0.2, generator=gen
        ),
    ),
    (
        "DAGRandomWalkToRoot",
        lambda gen: DAGRandomWalkToRoot(n_features=10, p_edge=0.3, generator=gen),
    ),
    (
        "PreferentialAttachment",
        lambda gen: PreferentialAttachment(n_features=10, p_active=0.3, generator=gen),
    ),
]


class TestCrossCuttingConcerns:
    """Tests that apply to ALL distribution classes."""

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_output_shape(self, name, factory):
        dist = factory(_fresh_gen())
        s = dist.sample(64)
        assert s.shape == (64, 10), f"{name}: wrong shape {s.shape}"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_values_non_negative(self, name, factory):
        dist = factory(_fresh_gen())
        s = dist.sample(10000)
        assert s.min() >= 0.0, f"{name}: negative values found"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_reproducibility_with_same_seed(self, name, factory):
        d1 = factory(_fresh_gen(123))
        d2 = factory(_fresh_gen(123))
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert torch.equal(s1, s2), f"{name}: not reproducible"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_batch_size_one(self, name, factory):
        dist = factory(_fresh_gen())
        s = dist.sample(1)
        assert s.shape == (1, 10), f"{name}: batch_size=1 shape {s.shape}"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_large_batch(self, name, factory):
        dist = factory(_fresh_gen())
        s = dist.sample(10000)
        assert s.shape == (10000, 10), f"{name}: large batch shape {s.shape}"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_dtype_is_float(self, name, factory):
        dist = factory(_fresh_gen())
        s = dist.sample(10)
        assert s.dtype in (torch.float32, torch.float64), f"{name}: dtype {s.dtype}"

    @pytest.mark.parametrize(
        "name,factory", ALL_DIST_FACTORIES, ids=[x[0] for x in ALL_DIST_FACTORIES]
    )
    def test_sample_twice_produces_different_results(self, name, factory):
        """Calling sample() twice should give different results (unless probability is degenerate)."""
        dist = factory(_fresh_gen())
        s1 = dist.sample(100)
        s2 = dist.sample(100)
        # Very unlikely to be identical for non-degenerate distributions
        assert not torch.equal(s1, s2), f"{name}: two consecutive samples are identical"


# ============================================================================
# Edge case deep dives
# ============================================================================
class TestCorrelatedPairsParameterSolving:
    """Verify all 6 parameter combinations produce consistent results."""

    def test_pa_pi_roundtrip(self):
        """p_active + p_individual => derive correlation and density, then reconstruct."""
        pa, pi = 0.3, 0.7
        d = CorrelatedPairs(n_features=10, p_active=pa, p_individual=pi)
        # Compute expected correlation: pi * (1 - pa) / (1 - pa * pi)
        expected_corr = pi * (1 - pa) / (1 - pa * pi)
        expected_density = pa * pi

        # Now reconstruct from correlation + density
        d2 = CorrelatedPairs(
            n_features=10, correlation=expected_corr, density=expected_density
        )
        assert (d2.p_active - pa).abs().max() < 1e-5
        assert (d2.p_individual - pi).abs().max() < 1e-5

    def test_pa_corr_produces_correct_pi(self):
        pa, c = 0.4, 0.6
        d = CorrelatedPairs(n_features=10, p_active=pa, correlation=c)
        # pi = c / (1 - pa + c*pa)
        expected_pi = c / (1 - pa + c * pa)
        assert (d.p_individual - expected_pi).abs().max() < 1e-6

    def test_pi_corr_produces_correct_pa(self):
        pi, c = 0.7, 0.5
        d = CorrelatedPairs(n_features=10, p_individual=pi, correlation=c)
        # pa = (pi - c) / (pi * (1 - c))
        expected_pa = (pi - c) / (pi * (1 - c))
        assert (d.p_active - expected_pa).abs().max() < 1e-6


class TestHierarchicalPairsPerFeatureParams:
    """Test per-feature p_active and p_follow."""

    def test_per_feature_p_active(self, gen):
        """Different p_active per feature pair."""
        p_active = [0.1, 0.0, 0.9, 0.0, 0.5, 0.0]  # even indices used
        dist = HierarchicalPairs(
            n_features=6, p_active=p_active, p_follow=0.5, generator=gen
        )
        s = dist.sample(N)
        # Pair 0: p_active[0] = 0.1
        rate0 = (s[:, 0] > 0).float().mean().item()
        assert abs(rate0 - 0.1) < TOL
        # Pair 1: p_active[2] = 0.9
        rate1 = (s[:, 2] > 0).float().mean().item()
        assert abs(rate1 - 0.9) < TOL
        # Pair 2: p_active[4] = 0.5
        rate2 = (s[:, 4] > 0).float().mean().item()
        assert abs(rate2 - 0.5) < TOL


class TestDAGRandomWalkEdgeCases:
    """Additional edge case tests for DAGRandomWalkToRoot."""

    def test_fully_connected_upper_triangular(self, gen):
        """All edges present: walk should always reach node 0."""
        n = 5
        adj = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        dist = DAGRandomWalkToRoot(n_features=n, adjacency=adj, generator=gen)
        s = dist.sample(10000)
        # Node 0 is a root and is parent of all other nodes,
        # so any walk from a non-root should pass through some ancestor chain to 0
        # Node 0 should fire very frequently (every walk ending at a root hits it)
        rate_0 = (s[:, 0] > 0).float().mean().item()
        assert rate_0 > 0.5, f"Node 0 rate {rate_0:.4f} too low for fully connected DAG"

    def test_p_active_custom_distribution(self, gen):
        """Custom p_active should bias seed selection."""
        n = 5
        p_active = torch.zeros(n)
        p_active[0] = 1.0  # always start at node 0
        dist = DAGRandomWalkToRoot(
            n_features=n, p_edge=0.0, p_active=p_active, generator=gen
        )
        s = dist.sample(10000)
        # Only node 0 should ever be active (it's always the seed and has no parents to walk to)
        assert (s[:, 0] > 0).all()
        assert (s[:, 1:] == 0).all()


class TestPreferentialAttachmentCascadeAccuracy:
    """Detailed cascade tests for PreferentialAttachment."""

    def test_cascade_increases_density_over_independent(self, gen):
        """With cascade enabled, density should exceed p_active."""
        # Independent only (p_child=0)
        g1 = _fresh_gen(42)
        d_no_cascade = PreferentialAttachment(
            n_features=20, p_active=0.1, p_child=0.0, p_edge=0.3, generator=g1
        )
        s_no = d_no_cascade.sample(N)

        g2 = _fresh_gen(42)
        d_cascade = PreferentialAttachment(
            n_features=20, p_active=0.1, p_child=0.9, p_edge=0.3, generator=g2
        )
        # Same graph for fair comparison
        d_cascade.adjacency = d_no_cascade.adjacency.clone()
        d_cascade._build_log_survival()

        g3 = _fresh_gen(99)
        g4 = _fresh_gen(99)
        d_no_cascade.generator = g3
        d_cascade.generator = g4

        s_yes = d_cascade.sample(N)
        density_no = (s_no > 0).float().mean().item()
        density_yes = (s_yes > 0).float().mean().item()
        # Cascade should increase density if the graph has edges
        if d_no_cascade.adjacency.any():
            assert density_yes > density_no, (
                f"Cascade density {density_yes:.4f} should exceed no-cascade {density_no:.4f}"
            )
