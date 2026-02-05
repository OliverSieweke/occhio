"""Pytest tests for HierarchicalSparse."""

import pytest
import torch
from ..hierarchical import HierarchicalSparse


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture
def basic_dist(seeded_generator):
    """Basic distribution for general tests."""
    return HierarchicalSparse(
        n_features=20,
        p_base=0.7,
        depth_decay=0.8,
        max_children=4,
        generator=seeded_generator,
    )


class TestBasicSampling:
    def test_sample_shape(self, basic_dist):
        samples = basic_dist.sample(100)
        assert samples.shape == (100, 20)

    def test_values_in_unit_interval(self, basic_dist):
        samples = basic_dist.sample(1000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_root_always_fires(self, basic_dist):
        """Root node (index 0) should always be active."""
        samples = basic_dist.sample(1000)
        root_active = (samples[:, 0] > 0).all()
        assert root_active

    def test_hierarchy_constraint(self, basic_dist):
        """If child is active, parent must be active."""
        samples = basic_dist.sample(1000)

        for node in basic_dist.nodes:
            if node.parent is not None:
                child_active = samples[:, node.index] > 0
                parent_active = samples[:, node.parent] > 0
                violations = (child_active & ~parent_active).sum().item()
                assert (
                    violations == 0
                ), f"Node {node.index} active without parent {node.parent}"


class TestTreeGeneration:
    def test_correct_number_of_nodes(self, seeded_generator):
        for n in [10, 50, 100]:
            dist = HierarchicalSparse(n_features=n, generator=seeded_generator)
            assert len(dist.nodes) == n

    def test_root_exists(self, basic_dist):
        assert basic_dist.nodes[0].parent is None
        assert basic_dist.nodes[0].depth == 0

    def test_all_nodes_reachable_from_root(self, basic_dist):
        """Every node should have a path to root."""
        for node in basic_dist.nodes:
            current = node
            visited = set()
            while current.parent is not None:
                assert current.index not in visited, "Cycle detected"
                visited.add(current.index)
                current = basic_dist.nodes[current.parent]
            assert current.index == 0, "Path doesn't lead to root"

    def test_generate_new_tree_changes_structure(self, seeded_generator):
        dist = HierarchicalSparse(n_features=30, generator=seeded_generator)
        tree1_parents = dist.parent_tensor.clone()

        dist.generate_new_tree()
        tree2_parents = dist.parent_tensor.clone()

        assert not torch.equal(tree1_parents, tree2_parents)

    def test_max_children_respected(self, seeded_generator):
        max_children = 3
        dist = HierarchicalSparse(
            n_features=50,
            max_children=max_children,
            generator=seeded_generator,
        )
        for node in dist.nodes:
            assert len(node.children) <= max_children


class TestDepthDependentActivation:
    def test_activation_decreases_with_depth(self, seeded_generator):
        dist = HierarchicalSparse(
            n_features=100,
            p_base=0.8,
            depth_decay=0.7,
            max_children=5,
            generator=seeded_generator,
        )

        samples = dist.sample(5000)
        active = (samples > 0).float()

        prev_rate = 1.0
        for depth in range(1, dist.max_depth + 1):
            indices = dist.depth_indices[depth]
            if indices:
                rate = active[:, indices].mean().item()
                assert (
                    rate < prev_rate
                ), f"Depth {depth} rate {rate} >= previous {prev_rate}"
                prev_rate = rate

    def test_empirical_matches_theoretical(self, seeded_generator):
        dist = HierarchicalSparse(
            n_features=50,
            p_base=0.8,
            depth_decay=0.9,
            max_children=4,
            generator=seeded_generator,
        )

        samples = dist.sample(10000)
        empirical = (samples > 0).float().mean(dim=0)
        theoretical = dist.get_expected_active()

        # Should be close (allowing for sampling variance)
        max_diff = (empirical - theoretical).abs().max().item()
        assert max_diff < 0.05, f"Max difference {max_diff} too large"


class TestPByDepthOverride:
    def test_explicit_probabilities(self, seeded_generator):
        p_by_depth = [1.0, 0.5, 0.25, 0.1]
        dist = HierarchicalSparse(
            n_features=30,
            p_by_depth=p_by_depth,
            max_children=4,
            generator=seeded_generator,
        )

        # Check that _get_p_fire returns correct values
        for d, p in enumerate(p_by_depth):
            assert dist._get_p_fire(d) == p

    def test_p_by_depth_extends_with_last_value(self, seeded_generator):
        p_by_depth = [1.0, 0.5, 0.3]
        dist = HierarchicalSparse(
            n_features=100,
            p_by_depth=p_by_depth,
            max_children=3,
            generator=seeded_generator,
        )

        # Depths beyond the list should use last value
        assert dist._get_p_fire(5) == 0.3
        assert dist._get_p_fire(10) == 0.3


class TestTreeStats:
    def test_stats_keys(self, basic_dist):
        stats = basic_dist.get_tree_stats()
        expected_keys = {
            "n_nodes",
            "max_depth",
            "mean_depth",
            "n_leaves",
            "mean_children",
            "max_children",
            "depth_distribution",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_consistency(self, basic_dist):
        stats = basic_dist.get_tree_stats()
        assert stats["n_nodes"] == basic_dist.n_features
        assert stats["max_depth"] == basic_dist.max_depth
        assert sum(stats["depth_distribution"].values()) == stats["n_nodes"]


class TestDeviceHandling:
    def test_to_device(self, seeded_generator):
        dist = HierarchicalSparse(n_features=10, generator=seeded_generator)

        # Just test that to() doesn't error (can't test CUDA without GPU)
        dist.to("cpu")
        assert dist.device == torch.device("cpu")
        assert dist.parent_tensor.device == torch.device("cpu")


class TestReproducibility:
    def test_same_generator_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = HierarchicalSparse(n_features=20, generator=gen1)
        dist2 = HierarchicalSparse(n_features=20, generator=gen2)

        # Trees should be identical
        assert torch.equal(dist1.parent_tensor, dist2.parent_tensor)

        # Samples should be identical
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        assert torch.equal(samples1, samples2)
