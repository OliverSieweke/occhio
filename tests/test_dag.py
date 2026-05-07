"""Pytest tests for DAGDistribution and DAGBayesianPropagation."""

import pytest
import torch
from ..dag import DAGDistribution, DAGBayesianPropagation


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


class TestDAGDistributionBasic:
    def test_sample_shape(self, seeded_generator):
        dist = DAGDistribution(
            n_features=20, p_active=0.3, p_edge=0.2, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 20)

    def test_values_in_unit_interval(self, seeded_generator):
        dist = DAGDistribution(
            n_features=20, p_active=0.5, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_adjacency_is_upper_triangular(self, seeded_generator):
        dist = DAGDistribution(
            n_features=15, p_active=0.3, p_edge=0.3, generator=seeded_generator
        )
        lower = torch.tril(dist.adjacency, diagonal=0)
        assert (lower == 0).all(), "Adjacency should be strictly upper triangular"

    def test_adjacency_shape(self, seeded_generator):
        n = 25
        dist = DAGDistribution(
            n_features=n, p_active=0.3, p_edge=0.2, generator=seeded_generator
        )
        assert dist.adjacency.shape == (n, n)


class TestDAGDistributionCausality:
    def test_child_inactive_when_all_parents_inactive(self, seeded_generator):
        """If all parents of a node are inactive, the node must be inactive (unless root)."""
        dist = DAGDistribution(
            n_features=20, p_active=0.5, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(5000)

        for j in range(dist.n_features):
            parent_mask = dist.adjacency[:, j]
            if not parent_mask.any():
                continue  # Root node, skip

            # Find samples where all parents are inactive
            parent_indices = parent_mask.nonzero(as_tuple=True)[0]
            all_parents_inactive = (samples[:, parent_indices] == 0).all(dim=1)

            # Child should be inactive in those samples
            child_active = samples[:, j] > 0
            violations = (all_parents_inactive & child_active).sum().item()
            assert violations == 0, f"Node {j} active without any active parent"

    def test_root_nodes_can_fire_independently(self, seeded_generator):
        """Nodes with no parents should fire with probability ~p_active."""
        p_active = 0.4
        dist = DAGDistribution(
            n_features=30, p_active=p_active, p_edge=0.2, generator=seeded_generator
        )
        samples = dist.sample(10000)

        for j in range(dist.n_features):
            parent_mask = dist.adjacency[:, j]
            if parent_mask.any():
                continue  # Not a root

            rate = (samples[:, j] > 0).float().mean().item()
            assert abs(rate - p_active) < 0.03, (
                f"Root {j} rate {rate} should be ~{p_active}"
            )


class TestDAGDistributionEdgeCases:
    def test_p_edge_zero_all_roots(self, seeded_generator):
        """With p_edge=0, all nodes are roots."""
        dist = DAGDistribution(
            n_features=20, p_active=0.5, p_edge=0.0, generator=seeded_generator
        )
        assert (dist.adjacency == 0).all()

        # All nodes should fire independently
        samples = dist.sample(5000)
        rate = (samples > 0).float().mean().item()
        assert abs(rate - 0.5) < 0.02

    def test_p_active_zero_all_zeros(self, seeded_generator):
        """With p_active=0, nothing fires."""
        dist = DAGDistribution(
            n_features=20, p_active=0.0, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_regenerate_dag_changes_structure(self, seeded_generator):
        dist = DAGDistribution(
            n_features=30, p_active=0.3, p_edge=0.3, generator=seeded_generator
        )
        adj1 = dist.adjacency.clone()

        dist.regenerate_dag()
        adj2 = dist.adjacency.clone()

        assert not torch.equal(adj1, adj2)


class TestDAGDistributionReproducibility:
    def test_same_seed_same_dag(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = DAGDistribution(n_features=20, p_active=0.3, p_edge=0.2, generator=gen1)
        dist2 = DAGDistribution(n_features=20, p_active=0.3, p_edge=0.2, generator=gen2)

        assert torch.equal(dist1.adjacency, dist2.adjacency)

    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = DAGDistribution(n_features=20, p_active=0.3, p_edge=0.2, generator=gen1)
        dist2 = DAGDistribution(n_features=20, p_active=0.3, p_edge=0.2, generator=gen2)

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestDAGBayesianBasic:
    def test_sample_shape(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.2, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 20)

    def test_values_in_unit_interval(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.5, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_adjacency_is_upper_triangular(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=15, p_active=0.3, p_edge=0.3, generator=seeded_generator
        )
        lower = torch.tril(dist.adjacency, diagonal=0)
        assert (lower == 0).all()

    def test_parent_cache_consistent(self, seeded_generator):
        """Parent cache should match adjacency matrix."""
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.3, generator=seeded_generator
        )

        for j in range(dist.n_features):
            expected_parents = dist.adjacency[:, j].nonzero(as_tuple=True)[0]
            cached_parents = dist._parent_indices[j]
            assert torch.equal(expected_parents, cached_parents)
            assert dist._has_parents[j] == (len(expected_parents) > 0)


class TestDAGBayesianNoisyOR:
    def test_inactive_parents_dont_trigger(self, seeded_generator):
        """If all parents are inactive (value=0), child cannot fire via Noisy-OR."""
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.5, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(5000)

        for j in range(dist.n_features):
            if not dist._has_parents[j]:
                continue

            parent_idx = dist._parent_indices[j]
            all_parents_inactive = (samples[:, parent_idx] == 0).all(dim=1)
            child_active = samples[:, j] > 0
            violations = (all_parents_inactive & child_active).sum().item()
            assert violations == 0, f"Node {j} fired without active parents"

    def test_high_parent_value_increases_child_probability(self, seeded_generator):
        """Noisy-OR: higher parent values → higher child activation probability."""
        dist = DAGBayesianPropagation(
            n_features=30, p_active=0.8, p_edge=0.4, generator=seeded_generator
        )
        samples = dist.sample(10000)

        # Find a node with exactly one parent for clean test
        single_parent_nodes = [
            j for j in range(dist.n_features) if len(dist._parent_indices[j]) == 1
        ]

        if not single_parent_nodes:
            pytest.skip("No single-parent nodes in this DAG")

        j = single_parent_nodes[0]
        parent_idx = dist._parent_indices[j][0].item()

        # Split by parent value: low (0, 0.3) vs high (0.7, 1.0)
        parent_vals = samples[:, parent_idx]
        low_mask = (parent_vals > 0) & (parent_vals < 0.3)
        high_mask = parent_vals > 0.7

        if low_mask.sum() < 100 or high_mask.sum() < 100:
            pytest.skip("Not enough samples in low/high bins")

        low_child_rate = (samples[low_mask, j] > 0).float().mean().item()
        high_child_rate = (samples[high_mask, j] > 0).float().mean().item()

        assert high_child_rate > low_child_rate, (
            f"High parent value should increase child rate: low={low_child_rate}, high={high_child_rate}"
        )

    def test_multiple_parents_compound(self, seeded_generator):
        """Multiple active parents should increase firing probability."""
        dist = DAGBayesianPropagation(
            n_features=30, p_active=0.9, p_edge=0.5, generator=seeded_generator
        )
        samples = dist.sample(10000)

        # Find node with 2+ parents
        multi_parent_nodes = [
            j for j in range(dist.n_features) if len(dist._parent_indices[j]) >= 2
        ]

        if not multi_parent_nodes:
            pytest.skip("No multi-parent nodes in this DAG")

        j = multi_parent_nodes[0]
        parent_idx = dist._parent_indices[j]
        parent_active_count = (samples[:, parent_idx] > 0).sum(dim=1)

        # Compare: 1 active parent vs 2+ active parents
        one_active = parent_active_count == 1
        multi_active = parent_active_count >= 2

        if one_active.sum() < 100 or multi_active.sum() < 100:
            pytest.skip("Not enough samples for comparison")

        rate_one = (samples[one_active, j] > 0).float().mean().item()
        rate_multi = (samples[multi_active, j] > 0).float().mean().item()

        assert rate_multi > rate_one, (
            f"More active parents should increase rate: one={rate_one}, multi={rate_multi}"
        )


class TestDAGBayesianEdgeCases:
    def test_p_edge_zero_all_roots(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.5, p_edge=0.0, generator=seeded_generator
        )
        assert (dist.adjacency == 0).all()
        assert all(not has for has in dist._has_parents)

    def test_p_active_zero_all_zeros(self, seeded_generator):
        """With p_active=0, roots don't fire, so nothing propagates."""
        dist = DAGBayesianPropagation(
            n_features=20, p_active=0.0, p_edge=0.3, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_regenerate_dag_updates_cache(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=30, p_active=0.3, p_edge=0.3, generator=seeded_generator
        )
        adj1 = dist.adjacency.clone()
        cache1 = [p.clone() for p in dist._parent_indices]

        dist.regenerate_dag()

        assert not torch.equal(adj1, dist.adjacency)
        # Cache should be rebuilt
        for j in range(dist.n_features):
            expected = dist.adjacency[:, j].nonzero(as_tuple=True)[0]
            assert torch.equal(expected, dist._parent_indices[j])


class TestDAGBayesianExpectedActivation:
    def test_get_expected_activation_shape(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=15, p_active=0.3, p_edge=0.2, generator=seeded_generator
        )
        expected = dist.get_expected_activation(n_samples=1000)
        assert expected.shape == (15,)

    def test_get_expected_activation_range(self, seeded_generator):
        dist = DAGBayesianPropagation(
            n_features=15, p_active=0.3, p_edge=0.2, generator=seeded_generator
        )
        expected = dist.get_expected_activation(n_samples=5000)
        assert (expected >= 0).all()
        assert (expected <= 1).all()


class TestDAGBayesianReproducibility:
    def test_same_seed_same_dag_and_cache(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.2, generator=gen1
        )
        dist2 = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.2, generator=gen2
        )

        assert torch.equal(dist1.adjacency, dist2.adjacency)
        for p1, p2 in zip(dist1._parent_indices, dist2._parent_indices):
            assert torch.equal(p1, p2)

    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.2, generator=gen1
        )
        dist2 = DAGBayesianPropagation(
            n_features=20, p_active=0.3, p_edge=0.2, generator=gen2
        )

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestDAGComparison:
    def test_both_respect_dag_structure(self, seeded_generator):
        """Both distributions should respect parent-child relationships."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        dist1 = DAGDistribution(n_features=20, p_active=0.5, p_edge=0.3, generator=gen1)
        dist2 = DAGBayesianPropagation(
            n_features=20, p_active=0.5, p_edge=0.3, generator=gen2
        )

        # Same seed means same DAG
        assert torch.equal(dist1.adjacency, dist2.adjacency)

        # Both should satisfy: child inactive when all parents inactive
        for dist in [dist1, dist2]:
            samples = dist.sample(2000)
            for j in range(dist.n_features):
                parent_mask = dist.adjacency[:, j]
                if not parent_mask.any():
                    continue
                parent_idx = parent_mask.nonzero(as_tuple=True)[0]
                all_parents_inactive = (samples[:, parent_idx] == 0).all(dim=1)
                child_active = samples[:, j] > 0
                violations = (all_parents_inactive & child_active).sum().item()
                assert violations == 0

    def test_bayesian_value_dependent_simple_not(self, seeded_generator):
        """Bayesian uses parent values; simple only uses binary activation."""
        dist = DAGBayesianPropagation(
            n_features=30, p_active=0.8, p_edge=0.4, generator=seeded_generator
        )
        samples = dist.sample(10000)

        # For Bayesian: among active parents, higher value → higher child rate
        single_parent = [j for j in range(30) if len(dist._parent_indices[j]) == 1]
        if single_parent:
            j = single_parent[0]
            parent = dist._parent_indices[j][0].item()

            active_parent = samples[:, parent] > 0
            if active_parent.sum() > 200:
                parent_vals = samples[active_parent, parent]
                child_active = samples[active_parent, j] > 0

                # Check correlation
                corr = torch.corrcoef(torch.stack([parent_vals, child_active.float()]))[
                    0, 1
                ]
                assert corr > 0.1, f"Expected positive correlation, got {corr}"
