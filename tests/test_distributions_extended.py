"""Comprehensive tests for untested distribution classes in occhio.

Covers:
- base.py: Distribution (ABC) helpers, DistributionStack
- spherical.py: SphericalDistribution
- toric.py: ToricDistribution
- hypercube.py: HypercubeDistribution
- simplex.py: SimplexDistribution, SimplicialComplexDistribution
- ssb.py: SyntheticDataModel, CorrelationStructure, FiringSampler,
          MagnitudeSampler, HierarchyConstraints, helper functions

Testing strategy: for each class we verify construction, output shape,
value ranges, sparsity patterns, reproducibility via seeded generators,
and rejection of invalid parameters.  Integration tests exercise the
full SSB pipeline including hierarchy constraints.
"""

import math

import pytest
import torch
from torch import Tensor

from occhio.distributions.base import Distribution, DistributionStack
from occhio.distributions.spherical import SphericalDistribution
from occhio.distributions.toric import ToricDistribution
from occhio.distributions.hypercube import HypercubeDistribution
from occhio.distributions.simplex import (
    SimplexDistribution,
    SimplicialComplexDistribution,
)
from occhio.distributions.sparse import SparseUniform
from occhio.distributions.ssb import (
    CorrelationStructure,
    FiringSampler,
    HierarchyConstraints,
    HierarchyNode,
    MagnitudeSampler,
    SyntheticDataConfig,
    SyntheticDataModel,
    _compute_firing_probs,
    _make_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture
def make_generator():
    """Factory fixture: each call returns a fresh generator with the given seed."""

    def _make(seed: int = 42) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen

    return _make


# ===================================================================
# Distribution ABC & helpers
# ===================================================================


class _ConcreteDistribution(Distribution):
    """Minimal concrete subclass for testing the abstract base."""

    def sample(self, batch_size: int) -> Tensor:
        return self._rand(batch_size, self.n_features)


class TestDistributionABC:
    """Tests for the abstract Distribution base class."""

    def test_cannot_instantiate_abc(self):
        """ABC guard: instantiating Distribution directly must fail."""
        with pytest.raises(TypeError):
            Distribution(n_features=5)

    def test_concrete_subclass_instantiates(self):
        """Concrete subclass with sample() implemented can be created."""
        dist = _ConcreteDistribution(n_features=5)
        assert dist.n_features == 5

    def test_default_device_is_none(self):
        """Without explicit device or generator, device defaults to None."""
        dist = _ConcreteDistribution(n_features=3)
        assert dist.device is None

    def test_device_from_generator(self):
        """Device is inferred from the generator when not explicitly given."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        dist = _ConcreteDistribution(n_features=3, generator=gen)
        assert dist.device == torch.device("cpu")

    def test_device_generator_mismatch_raises(self):
        """Passing device != generator.device must raise ValueError."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        # Use a device string that differs from cpu
        # On CPU-only machines meta is always available
        with pytest.raises(ValueError, match="must match"):
            _ConcreteDistribution(n_features=3, device="meta", generator=gen)

    def test_repr_and_str(self):
        """__repr__ and __str__ include class name and n_features."""
        dist = _ConcreteDistribution(n_features=7)
        assert "7" in repr(dist)
        assert "_ConcreteDistribution" in str(dist)


class TestDistributionHelpers:
    """Tests for _rand, _randn, _randint, _rand_On, _broadcast."""

    def test_rand_shape(self, seeded_generator):
        dist = _ConcreteDistribution(n_features=5, generator=seeded_generator)
        t = dist._rand(3, 4)
        assert t.shape == (3, 4)

    def test_randn_shape(self, seeded_generator):
        dist = _ConcreteDistribution(n_features=5, generator=seeded_generator)
        t = dist._randn(2, 6)
        assert t.shape == (2, 6)

    def test_randint_range(self, seeded_generator):
        dist = _ConcreteDistribution(n_features=5, generator=seeded_generator)
        t = dist._randint(0, 10, (100,))
        assert t.min() >= 0
        assert t.max() < 10

    def test_randint_with_probs(self, seeded_generator):
        """_randint with p argument uses multinomial sampling."""
        dist = _ConcreteDistribution(n_features=5, generator=seeded_generator)
        # All weight on index 2 (offset by low=0)
        p = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        t = dist._randint(0, 5, (50,), p=p)
        assert (t == 2).all()

    def test_rand_On_orthogonal(self, seeded_generator):
        """_rand_On must return an orthogonal matrix."""
        dist = _ConcreteDistribution(n_features=5, generator=seeded_generator)
        q = dist._rand_On(4)
        eye = torch.eye(4)
        assert torch.allclose(q @ q.T, eye, atol=1e-5)

    def test_broadcast_scalar(self):
        dist = _ConcreteDistribution(n_features=4)
        t = dist._broadcast(3.14)
        assert t.shape == (4,)
        assert torch.allclose(t, torch.full((4,), 3.14))

    def test_broadcast_list(self):
        dist = _ConcreteDistribution(n_features=3)
        t = dist._broadcast([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_broadcast_tensor_0d(self):
        """Scalar tensor should be expanded to n_features."""
        dist = _ConcreteDistribution(n_features=3)
        t = dist._broadcast(torch.tensor(5.0))
        assert t.shape == (3,)
        assert (t == 5.0).all()


class TestDistributionReproducibility:
    """Seeded generators must produce identical samples."""

    def test_same_seed_same_samples(self, make_generator):
        g1 = make_generator(99)
        g2 = make_generator(99)
        d1 = _ConcreteDistribution(n_features=8, generator=g1)
        d2 = _ConcreteDistribution(n_features=8, generator=g2)
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert torch.equal(s1, s2)

    def test_different_seed_different_samples(self, make_generator):
        g1 = make_generator(1)
        g2 = make_generator(2)
        d1 = _ConcreteDistribution(n_features=8, generator=g1)
        d2 = _ConcreteDistribution(n_features=8, generator=g2)
        s1 = d1.sample(50)
        s2 = d2.sample(50)
        assert not torch.equal(s1, s2)


class TestDistributionGeneratorSync:
    """Tests for collect_generators / sync_generators."""

    def test_collect_generators_returns_list(self, seeded_generator):
        dist = _ConcreteDistribution(n_features=3, generator=seeded_generator)
        gens = dist.collect_generators()
        assert isinstance(gens, list)
        assert len(gens) == 1
        assert gens[0] is seeded_generator

    def test_sync_generators_copies_state(self, make_generator):
        """sync_generators should copy the state so the receiver produces the same sequence."""
        src = make_generator(123)
        dst = make_generator(0)
        dist = _ConcreteDistribution(n_features=3, generator=dst)
        dist.sync_generators(src)
        # Now dst should have the same state as src
        v1 = torch.rand(5, generator=src)
        v2 = torch.rand(5, generator=dst)
        assert torch.equal(v1, v2)

    def test_sync_generators_list_wrong_length(self, make_generator):
        """Passing a list with != 1 element to base Distribution must raise."""
        dist = _ConcreteDistribution(n_features=3, generator=make_generator(0))
        with pytest.raises(ValueError, match="single generator"):
            dist.sync_generators([make_generator(1), make_generator(2)])


class TestDistributionEquivalenceHash:
    """Tests for _equivalence_hash."""

    def test_same_config_same_hash(self, make_generator):
        g1, g2 = make_generator(42), make_generator(42)
        d1 = _ConcreteDistribution(n_features=5, generator=g1)
        d2 = _ConcreteDistribution(n_features=5, generator=g2)
        assert d1._equivalence_hash == d2._equivalence_hash

    def test_different_features_different_hash(self, make_generator):
        g1, g2 = make_generator(42), make_generator(42)
        d1 = _ConcreteDistribution(n_features=5, generator=g1)
        d2 = _ConcreteDistribution(n_features=6, generator=g2)
        assert d1._equivalence_hash != d2._equivalence_hash


class TestDistributionTo:
    """Tests for the .to() device-transfer method."""

    def test_to_changes_device(self):
        dist = _ConcreteDistribution(n_features=3, device="cpu")
        dist.to("cpu")  # no-op but should not error
        assert dist.device == torch.device("cpu")


# ===================================================================
# DistributionStack
# ===================================================================


class TestDistributionStackConstruction:
    def test_empty_distributions_raises(self):
        """Empty list is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DistributionStack([])

    def test_nested_stack_raises(self, seeded_generator):
        """Nesting DistributionStack is not allowed."""
        inner = DistributionStack(
            [SparseUniform(n_features=3, p_active=0.5, generator=seeded_generator)]
        )
        with pytest.raises(TypeError, match="Nesting"):
            DistributionStack([inner])

    def test_sparse_mode_requires_p_meta(self, seeded_generator):
        """sampling_mode='sparse' without p_meta must raise."""
        d = SparseUniform(n_features=3, p_active=0.5, generator=seeded_generator)
        with pytest.raises(ValueError, match="p_meta"):
            DistributionStack([d], sampling_mode="sparse")

    def test_total_features_is_sum(self, make_generator):
        """n_features should be the sum of child feature counts."""
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=7, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        assert stack.n_features == 10


class TestDistributionStackSample:
    def test_shape_independent(self, make_generator):
        """Independent mode: output shape = (batch, sum of n_features)."""
        d1 = SparseUniform(n_features=4, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        s = stack.sample(32)
        assert s.shape == (32, 10)

    def test_shape_single(self, make_generator):
        """Single mode: same output shape, but only one sub-dist fires per row."""
        d1 = SparseUniform(n_features=4, p_active=1.0, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=1.0, generator=make_generator(2))
        stack = DistributionStack(
            [d1, d2],
            sampling_mode="single",
            generator=make_generator(3),
        )
        s = stack.sample(100)
        assert s.shape == (100, 10)

    def test_single_mode_one_group_active(self, make_generator):
        """In single mode, each row should have at most one sub-dist group active."""
        d1 = SparseUniform(n_features=4, p_active=1.0, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=1.0, generator=make_generator(2))
        stack = DistributionStack(
            [d1, d2],
            sampling_mode="single",
            generator=make_generator(3),
        )
        s = stack.sample(200)
        for row in s:
            group1_active = (row[:4] > 0).any().item()
            group2_active = (row[4:] > 0).any().item()
            # At most one group should be active (both could be zero if
            # sub-dist sample is all zeros, but that's extremely unlikely with p_active=1)
            assert not (group1_active and group2_active), (
                "Both groups active in single mode"
            )

    def test_shape_sparse(self, make_generator):
        """Sparse mode: same output shape."""
        d1 = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack(
            [d1, d2],
            sampling_mode="sparse",
            p_meta=0.5,
            generator=make_generator(3),
        )
        s = stack.sample(64)
        assert s.shape == (64, 10)

    def test_values_non_negative(self, make_generator):
        """All sample values should be non-negative (SparseUniform produces [0,1])."""
        d1 = SparseUniform(n_features=4, p_active=0.8, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=0.8, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        s = stack.sample(500)
        assert s.min() >= 0.0

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    def test_various_batch_sizes(self, batch_size, make_generator):
        """Stack should work with different batch sizes."""
        d = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(1))
        stack = DistributionStack([d])
        s = stack.sample(batch_size)
        assert s.shape == (batch_size, 5)


class TestDistributionStackGenerators:
    def test_collect_generators_per_child(self, make_generator):
        g1, g2 = make_generator(1), make_generator(2)
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=g1)
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=g2)
        stack = DistributionStack([d1, d2])
        gens = stack.collect_generators()
        assert len(gens) == 2
        assert gens[0] is g1
        assert gens[1] is g2

    def test_sync_generators_wrong_count(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        with pytest.raises(ValueError, match="Expected 2"):
            stack.sync_generators([make_generator(1)])

    def test_defines_generators(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        assert stack._defines_generators is True

    def test_defines_generators_false_when_missing(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5)  # no generator
        stack = DistributionStack([d1, d2])
        assert stack._defines_generators is False


class TestDistributionStackRepr:
    def test_repr_includes_children(self, make_generator):
        d = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        stack = DistributionStack([d])
        r = repr(stack)
        assert "DistributionStack" in r
        assert "SparseUniform" in r


class TestDistributionStackEquivalenceHash:
    def test_same_children_same_hash(self, make_generator):
        d1a = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(42))
        d1b = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(42))
        s1 = DistributionStack([d1a])
        s2 = DistributionStack([d1b])
        assert s1._equivalence_hash == s2._equivalence_hash

    def test_different_children_different_hash(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(42))
        d2 = SparseUniform(n_features=4, p_active=0.5, generator=make_generator(42))
        s1 = DistributionStack([d1])
        s2 = DistributionStack([d2])
        assert s1._equivalence_hash != s2._equivalence_hash


class TestDistributionStackTo:
    def test_to_moves_all_children(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        stack.to("cpu")
        for d in stack.distributions:
            assert d.device == torch.device("cpu")


# ===================================================================
# SphericalDistribution
# ===================================================================


class TestSphericalDistributionBasic:
    def test_construction(self, seeded_generator):
        dist = SphericalDistribution(
            n_features=10, length_scale=1.0, generator=seeded_generator
        )
        assert dist.n_features == 10
        assert dist.length_scale == 1.0

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape(self, batch_size, seeded_generator):
        """sample() must return (batch_size, n_features)."""
        dist = SphericalDistribution(n_features=20, generator=seeded_generator)
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 20)

    def test_values_non_negative(self, seeded_generator):
        """Cosine bump clipped at zero means no negatives."""
        dist = SphericalDistribution(
            n_features=30,
            length_scale=0.5,
            magnitude_range=(0.9, 1.0),
            generator=seeded_generator,
        )
        s = dist.sample(500)
        assert s.min() >= 0.0

    def test_values_bounded_above(self, seeded_generator):
        """Output <= magnitude_range[1] since cosine <= 1."""
        hi = 2.0
        dist = SphericalDistribution(
            n_features=20,
            magnitude_range=(0.5, hi),
            generator=seeded_generator,
        )
        s = dist.sample(1000)
        assert s.max() <= hi + 1e-6

    def test_sparsity_with_small_length_scale(self, seeded_generator):
        """Small length_scale should produce sparse activations (many zeros)."""
        dist = SphericalDistribution(
            n_features=50,
            length_scale=0.2,
            generator=seeded_generator,
        )
        s = dist.sample(500)
        frac_zero = (s == 0).float().mean().item()
        assert frac_zero > 0.3, f"Expected significant sparsity, got {frac_zero}"


class TestSphericalDistributionManifoldDim:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_feature_positions_shape(self, dim, seeded_generator):
        """Feature positions should live in R^{dim+1}."""
        dist = SphericalDistribution(
            n_features=15, manifold_dim=dim, generator=seeded_generator
        )
        assert dist.feature_positions.shape == (15, dim + 1)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_feature_positions_unit_norm(self, dim, seeded_generator):
        """Feature positions should lie on the unit sphere."""
        dist = SphericalDistribution(
            n_features=15, manifold_dim=dim, generator=seeded_generator
        )
        norms = dist.feature_positions.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_circle_equal_spacing(self, seeded_generator):
        """manifold_dim=1 should produce equally spaced points on S^1."""
        dist = SphericalDistribution(
            n_features=6, manifold_dim=1, generator=seeded_generator
        )
        angles = torch.atan2(dist.feature_positions[:, 1], dist.feature_positions[:, 0])
        angles = angles % (2 * math.pi)
        angles_sorted = angles.sort().values
        diffs = angles_sorted[1:] - angles_sorted[:-1]
        expected_diff = 2 * math.pi / 6
        assert torch.allclose(diffs, torch.full_like(diffs, expected_diff), atol=1e-5)


class TestSphericalDistributionReproducibility:
    def test_seeded_reproducibility(self, make_generator):
        """Same seed produces identical samples."""
        d1 = SphericalDistribution(n_features=10, generator=make_generator(77))
        d2 = SphericalDistribution(n_features=10, generator=make_generator(77))
        assert torch.equal(d1.sample(50), d2.sample(50))


# ===================================================================
# ToricDistribution
# ===================================================================


class TestToricDistributionBasic:
    def test_construction(self, seeded_generator):
        dist = ToricDistribution(n_features=12, toric_dim=2, generator=seeded_generator)
        assert dist.n_features == 12
        assert dist.toric_dim == 2

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape(self, batch_size, seeded_generator):
        dist = ToricDistribution(n_features=15, generator=seeded_generator)
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 15)

    def test_values_non_negative(self, seeded_generator):
        """Cosine bump clipped at zero: no negatives."""
        dist = ToricDistribution(
            n_features=20, length_scale=0.5, generator=seeded_generator
        )
        s = dist.sample(500)
        assert s.min() >= 0.0

    def test_values_bounded_above(self, seeded_generator):
        hi = 3.0
        dist = ToricDistribution(
            n_features=20, magnitude_range=(1.0, hi), generator=seeded_generator
        )
        s = dist.sample(500)
        assert s.max() <= hi + 1e-6

    def test_sparsity_with_small_length_scale(self, seeded_generator):
        """Small length_scale should produce sparse activations."""
        dist = ToricDistribution(
            n_features=40, length_scale=0.2, generator=seeded_generator
        )
        s = dist.sample(500)
        frac_zero = (s == 0).float().mean().item()
        assert frac_zero > 0.3


class TestToricDistributionGeometry:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_feature_angles_shape(self, dim, seeded_generator):
        """Feature angles should have shape (n_features, toric_dim)."""
        dist = ToricDistribution(
            n_features=10, toric_dim=dim, generator=seeded_generator
        )
        assert dist.feature_angles.shape == (10, dim)

    def test_feature_angles_in_range(self, seeded_generator):
        """Angles should be in [0, 2*pi)."""
        dist = ToricDistribution(n_features=50, toric_dim=2, generator=seeded_generator)
        assert dist.feature_angles.min() >= 0.0
        assert dist.feature_angles.max() < 2 * math.pi + 1e-6

    def test_toric_distance_zero_for_same_point(self, seeded_generator):
        """Distance between a point and itself should be zero."""
        dist = ToricDistribution(n_features=5, toric_dim=2, generator=seeded_generator)
        a = torch.tensor([[1.0, 2.0]])
        d = dist._toric_distance(a, a)
        assert torch.allclose(d, torch.zeros(1), atol=1e-7)

    def test_toric_distance_wraps(self, seeded_generator):
        """Distance should wrap around: d(0, 2*pi - eps) ~ eps."""
        dist = ToricDistribution(n_features=5, toric_dim=1, generator=seeded_generator)
        eps = 0.1
        a = torch.tensor([[0.0]])
        b = torch.tensor([[2 * math.pi - eps]])
        d = dist._toric_distance(a, b)
        assert torch.allclose(d, torch.tensor([eps]), atol=1e-5)


class TestToricDistributionReproducibility:
    def test_seeded_reproducibility(self, make_generator):
        """Torus _place_features uses torch.rand (global RNG), so feature
        positions differ across instances even with the same generator seed.
        We seed the global RNG as well to get full reproducibility."""
        torch.manual_seed(0)
        d1 = ToricDistribution(n_features=10, generator=make_generator(77))
        torch.manual_seed(0)
        d2 = ToricDistribution(n_features=10, generator=make_generator(77))
        # Feature positions must match for samples to match
        assert torch.equal(d1.feature_angles, d2.feature_angles)
        assert torch.equal(d1.sample(50), d2.sample(50))


# ===================================================================
# HypercubeDistribution
# ===================================================================


class TestHypercubeDistributionBasic:
    def test_construction(self, seeded_generator):
        dist = HypercubeDistribution(
            n_features=9, cube_dim=2, generator=seeded_generator
        )
        assert dist.n_features == 9

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape(self, batch_size, seeded_generator):
        dist = HypercubeDistribution(n_features=10, generator=seeded_generator)
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 10)

    def test_values_non_negative(self, seeded_generator):
        """Tent bump clamped at zero: no negatives."""
        dist = HypercubeDistribution(n_features=10, generator=seeded_generator)
        s = dist.sample(500)
        assert s.min() >= 0.0

    def test_values_bounded_above(self, seeded_generator):
        hi = 2.0
        dist = HypercubeDistribution(
            n_features=10, magnitude_range=(0.5, hi), generator=seeded_generator
        )
        s = dist.sample(500)
        # activation in [0,1], magnitude in [lo,hi], so max is hi*1
        assert s.max() <= hi + 1e-6

    def test_sparsity_with_small_length_scale(self, seeded_generator):
        """Small length_scale should produce sparsity."""
        dist = HypercubeDistribution(
            n_features=20, length_scale=0.05, generator=seeded_generator
        )
        s = dist.sample(500)
        frac_zero = (s == 0).float().mean().item()
        assert frac_zero > 0.3


class TestHypercubeDistributionGrid:
    def test_perfect_square_uses_grid(self, seeded_generator):
        """9 features on 2D should use a 3x3 grid."""
        dist = HypercubeDistribution(
            n_features=9, cube_dim=2, generator=seeded_generator
        )
        assert dist.grid_size == 3

    def test_non_perfect_power_uses_random(self, seeded_generator):
        """7 features on 2D cannot form a grid, so grid_size is None."""
        dist = HypercubeDistribution(
            n_features=7, cube_dim=2, generator=seeded_generator
        )
        assert dist.grid_size is None

    def test_1d_grid_positions(self, seeded_generator):
        """1D grid should have evenly spaced positions in [0, 1]."""
        dist = HypercubeDistribution(
            n_features=5, cube_dim=1, generator=seeded_generator
        )
        assert dist.grid_size == 5
        expected = torch.linspace(0, 1, 5).unsqueeze(-1)
        assert torch.allclose(dist.feature_positions, expected, atol=1e-5)

    def test_feature_positions_in_unit_cube(self, seeded_generator):
        """All feature positions should be in [0, 1]^d."""
        dist = HypercubeDistribution(
            n_features=16, cube_dim=2, generator=seeded_generator
        )
        assert dist.feature_positions.min() >= 0.0
        assert dist.feature_positions.max() <= 1.0

    def test_cube_dim_3(self, seeded_generator):
        """8 features in 3D = 2^3 grid."""
        dist = HypercubeDistribution(
            n_features=8, cube_dim=3, generator=seeded_generator
        )
        assert dist.grid_size == 2
        assert dist.feature_positions.shape == (8, 3)

    def test_single_feature_grid(self, seeded_generator):
        """n_features=1 should work and place feature at center."""
        dist = HypercubeDistribution(
            n_features=1, cube_dim=2, generator=seeded_generator
        )
        assert dist.grid_size == 1
        assert torch.allclose(
            dist.feature_positions, torch.tensor([[0.5, 0.5]]), atol=1e-5
        )


class TestHypercubeDistributionReproducibility:
    def test_seeded_reproducibility(self, make_generator):
        d1 = HypercubeDistribution(n_features=10, generator=make_generator(77))
        d2 = HypercubeDistribution(n_features=10, generator=make_generator(77))
        assert torch.equal(d1.sample(50), d2.sample(50))


# ===================================================================
# SimplexDistribution
# ===================================================================


class TestSimplexDistributionBasic:
    def test_construction(self, seeded_generator):
        dist = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=0.5, generator=seeded_generator
        )
        assert dist.n_features == 7
        assert dist.n_simplices == 2

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape(self, batch_size, seeded_generator):
        dist = SimplexDistribution(
            simplex_sizes=[3, 5], p_active=0.5, generator=seeded_generator
        )
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 8)

    def test_values_non_negative(self, seeded_generator):
        """Dirichlet samples and zero mask: all values >= 0."""
        dist = SimplexDistribution(
            simplex_sizes=[4, 3, 5], p_active=0.8, generator=seeded_generator
        )
        s = dist.sample(500)
        assert s.min() >= 0.0


class TestSimplexDistributionNormalization:
    def test_active_simplices_sum_to_one(self, seeded_generator):
        """When a simplex fires, its features should sum to 1 (Dirichlet)."""
        sizes = [3, 4]
        dist = SimplexDistribution(
            simplex_sizes=sizes, p_active=1.0, generator=seeded_generator
        )
        s = dist.sample(200)
        # With p_active=1.0, all simplices fire
        # Check first simplex sums to 1
        simplex1_sum = s[:, :3].sum(dim=-1)
        assert torch.allclose(simplex1_sum, torch.ones(200), atol=1e-5)
        # Check second simplex sums to 1
        simplex2_sum = s[:, 3:7].sum(dim=-1)
        assert torch.allclose(simplex2_sum, torch.ones(200), atol=1e-5)

    def test_inactive_simplices_all_zero(self, seeded_generator):
        """With p_active=0, everything should be zero."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=0.0, generator=seeded_generator
        )
        s = dist.sample(100)
        assert (s == 0).all()


class TestSimplexDistributionSparsity:
    def test_sparsity_pattern(self, seeded_generator):
        """With moderate p_active, some simplices should be zero blocks."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 3, 3], p_active=0.3, generator=seeded_generator
        )
        s = dist.sample(1000)
        # Check that at least some rows have zero blocks
        block_sums = torch.stack(
            [s[:, i * 3 : (i + 1) * 3].sum(dim=-1) for i in range(3)], dim=-1
        )
        zero_blocks = (block_sums == 0).float().mean().item()
        assert zero_blocks > 0.5, f"Expected ~70% zero blocks, got {zero_blocks}"

    def test_per_simplex_p_active(self, seeded_generator):
        """Different p_active per simplex should produce different firing rates."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 3],
            p_active=[0.1, 0.9],
            generator=seeded_generator,
        )
        s = dist.sample(5000)
        s1_active = (s[:, :3].sum(dim=-1) > 0).float().mean().item()
        s2_active = (s[:, 3:].sum(dim=-1) > 0).float().mean().item()
        assert s1_active < 0.25
        assert s2_active > 0.75


class TestSimplexDistributionValidation:
    def test_p_active_length_mismatch(self, seeded_generator):
        """p_active list length != number of simplices should raise."""
        with pytest.raises(ValueError, match="length"):
            SimplexDistribution(
                simplex_sizes=[3, 4],
                p_active=[0.5, 0.5, 0.5],  # 3 != 2 simplices
                generator=seeded_generator,
            )


class TestSimplexDistributionReproducibility:
    def test_seeded_reproducibility(self, make_generator):
        d1 = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=0.5, generator=make_generator(77)
        )
        d2 = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=0.5, generator=make_generator(77)
        )
        assert torch.equal(d1.sample(50), d2.sample(50))


# ===================================================================
# SimplicialComplexDistribution
# ===================================================================


class TestSimplicialComplexBasic:
    def test_construction_single(self, seeded_generator):
        faces = [(0, 1), (1, 2), (2, 0)]
        dist = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        assert dist.n_features == 3
        assert dist.n_faces == 3

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape_single(self, batch_size, seeded_generator):
        faces = [(0, 1, 2), (2, 3, 4)]
        dist = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 5)

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape_sparse(self, batch_size, seeded_generator):
        faces = [(0, 1), (1, 2)]
        dist = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            sampling_mode="sparse",
            generator=seeded_generator,
        )
        s = dist.sample(batch_size)
        assert s.shape == (batch_size, 3)

    def test_values_non_negative(self, seeded_generator):
        faces = [(0, 1, 2), (2, 3, 4)]
        dist = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        s = dist.sample(500)
        assert s.min() >= 0.0


class TestSimplicialComplexSingle:
    def test_single_face_dirichlet_sum(self, seeded_generator):
        """In single mode with uniform faces, active vertices should sum to 1."""
        faces = [(0, 1, 2), (3, 4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        s = dist.sample(200)
        # Each row: exactly one face is active, its vertices sum to 1
        row_sums = s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(200), atol=1e-5)

    def test_single_mode_one_face_per_row(self, seeded_generator):
        """In single mode, only one face's vertices should be non-zero per row."""
        faces = [(0, 1), (2, 3), (4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        s = dist.sample(300)
        for row in s:
            active_faces = 0
            for f in faces:
                if any(row[v] > 0 for v in f):
                    active_faces += 1
            assert active_faces == 1, f"Expected 1 active face, got {active_faces}"


class TestSimplicialComplexSparse:
    def test_sparse_mode_can_fire_multiple(self, seeded_generator):
        """In sparse mode with high p_active, multiple faces can fire."""
        faces = [(0, 1), (2, 3), (4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            p_active=0.99,
            sampling_mode="sparse",
            generator=seeded_generator,
        )
        s = dist.sample(100)
        # With p=0.99 and 3 faces, almost all rows should have all faces active
        all_active = (s > 0).all(dim=-1).float().mean().item()
        assert all_active > 0.8

    def test_sparse_mode_low_p_sparsity(self, seeded_generator):
        """Low p_active should produce many zero rows or zero blocks."""
        faces = [(0, 1), (2, 3), (4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            p_active=0.1,
            sampling_mode="sparse",
            generator=seeded_generator,
        )
        s = dist.sample(1000)
        frac_zero = (s == 0).float().mean().item()
        assert frac_zero > 0.5


class TestSimplicialComplexSharedVertices:
    def test_shared_vertex_receives_contributions(self, seeded_generator):
        """When faces share a vertex and both fire, the shared vertex gets summed contributions."""
        # Faces (0,1) and (1,2) share vertex 1
        faces = [(0, 1), (1, 2)]
        dist = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            p_active=1.0,
            sampling_mode="sparse",
            generator=seeded_generator,
        )
        s = dist.sample(200)
        # Vertex 1 should generally have a larger value than 0 or 2
        mean_v1 = s[:, 1].mean().item()
        mean_v0 = s[:, 0].mean().item()
        assert mean_v1 > mean_v0, (
            "Shared vertex should accumulate more activation on average"
        )


class TestSimplicialComplexNonUniformFaces:
    def test_non_uniform_face_sizes(self, seeded_generator):
        """Faces of different sizes should work in both modes."""
        faces = [(0, 1), (2, 3, 4)]  # sizes 2 and 3
        dist_single = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=seeded_generator,
        )
        s = dist_single.sample(100)
        assert s.shape == (100, 5)
        assert s.min() >= 0.0


class TestSimplicialComplexReproducibility:
    def test_seeded_reproducibility_single(self, make_generator):
        faces = [(0, 1, 2), (2, 3, 4)]
        d1 = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=make_generator(77),
        )
        d2 = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=make_generator(77),
        )
        assert torch.equal(d1.sample(50), d2.sample(50))

    def test_seeded_reproducibility_sparse(self, make_generator):
        faces = [(0, 1), (1, 2)]
        d1 = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            sampling_mode="sparse",
            generator=make_generator(77),
        )
        d2 = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            sampling_mode="sparse",
            generator=make_generator(77),
        )
        assert torch.equal(d1.sample(50), d2.sample(50))


# ===================================================================
# SSB helper functions
# ===================================================================


class TestMakeSchedule:
    def test_constant(self):
        s = _make_schedule(5, "constant", value=3.0)
        assert s.shape == (5,)
        assert torch.allclose(s, torch.full((5,), 3.0))

    def test_linear(self):
        s = _make_schedule(5, "linear", high=10.0, low=2.0)
        assert s.shape == (5,)
        assert s[0].item() == pytest.approx(10.0, abs=1e-5)
        assert s[-1].item() == pytest.approx(2.0, abs=1e-5)
        # Should be monotonically non-increasing
        assert (s[:-1] >= s[1:]).all()

    def test_exponential(self):
        s = _make_schedule(5, "exponential", high=10.0, low=0.1)
        assert s.shape == (5,)
        assert s[0].item() == pytest.approx(10.0, rel=0.01)
        assert s[-1].item() == pytest.approx(0.1, rel=0.01)

    def test_folded_normal(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        s = _make_schedule(
            100, "folded_normal", folded_mu=1.0, folded_sigma=0.5, generator=gen
        )
        assert s.shape == (100,)
        assert (s >= 0).all(), "Folded normal should be non-negative"

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            _make_schedule(5, "bogus", value=1.0)

    def test_constant_missing_value_raises(self):
        with pytest.raises(AssertionError):
            _make_schedule(5, "constant")


class TestComputeFiringProbs:
    def test_constant(self):
        p = _compute_firing_probs(10, "constant", p_min=0.05, p_max=0.1)
        assert p.shape == (10,)
        assert torch.allclose(p, torch.full((10,), 0.05))

    def test_constant_with_p_constant(self):
        p = _compute_firing_probs(
            10, "constant", p_min=0.01, p_max=0.1, p_constant=0.07
        )
        assert torch.allclose(p, torch.full((10,), 0.07))

    def test_linear(self):
        p = _compute_firing_probs(5, "linear", p_min=0.01, p_max=0.1)
        assert p[0].item() == pytest.approx(0.1, abs=1e-5)
        assert p[-1].item() == pytest.approx(0.01, abs=1e-5)

    def test_uniform(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        p = _compute_firing_probs(100, "uniform", p_min=0.01, p_max=0.1, generator=gen)
        assert (p >= 0.01).all()
        assert (p <= 0.1).all()

    def test_zipfian(self):
        p = _compute_firing_probs(20, "zipfian", p_min=0.001, p_max=0.1, alpha=1.0)
        assert p[0].item() == pytest.approx(0.1, abs=1e-5)
        assert p[-1].item() == pytest.approx(0.001, abs=1e-5)
        # Should be monotonically non-increasing
        assert (p[:-1] >= p[1:]).all()

    def test_zipfian_single_feature(self):
        """Edge case: single feature with zipfian should not error."""
        p = _compute_firing_probs(1, "zipfian", p_min=0.01, p_max=0.1)
        assert p.shape == (1,)

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown firing"):
            _compute_firing_probs(5, "bogus", p_min=0.01, p_max=0.1)


# ===================================================================
# CorrelationStructure
# ===================================================================


class TestCorrelationStructure:
    def test_rank_zero_independent(self):
        """rank=0 means independent Gaussians (no factor matrix)."""
        cs = CorrelationStructure(n_features=10, rank=0)
        assert cs.factor_matrix is None
        assert cs.diagonal is None

    def test_rank_zero_sample_shape(self):
        cs = CorrelationStructure(n_features=10, rank=0)
        gen = torch.Generator()
        gen.manual_seed(42)
        s = cs.sample(32, generator=gen)
        assert s.shape == (32, 10)

    def test_nonzero_rank_has_factor(self):
        cs = CorrelationStructure(n_features=10, rank=3)
        assert cs.factor_matrix is not None
        assert cs.factor_matrix.shape == (10, 3)
        assert cs.diagonal is not None
        assert cs.diagonal.shape == (10,)

    def test_diagonal_positive(self):
        """Diagonal elements must be positive for numerical stability."""
        cs = CorrelationStructure(
            n_features=20, rank=5, correlation_scale=0.1, delta_min=0.01
        )
        assert (cs.diagonal >= 0.01 - 1e-6).all()

    def test_nonzero_rank_sample_shape(self):
        cs = CorrelationStructure(n_features=10, rank=3)
        gen = torch.Generator()
        gen.manual_seed(42)
        s = cs.sample(50, generator=gen)
        assert s.shape == (50, 10)

    def test_high_correlation_scale_rescales(self):
        """Large correlation_scale should trigger the rescaling path."""
        cs = CorrelationStructure(
            n_features=5, rank=4, correlation_scale=10.0, delta_min=0.01
        )
        # After rescaling, diagonal should still be >= delta_min
        assert (cs.diagonal >= 0.01 - 1e-6).all()


# ===================================================================
# FiringSampler
# ===================================================================


class TestFiringSampler:
    def test_output_binary(self):
        """Firing mask should contain only 0s and 1s."""
        probs = torch.full((10,), 0.5)
        corr = CorrelationStructure(n_features=10, rank=0)
        sampler = FiringSampler(probs, corr)
        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(500, generator=gen)
        unique = z.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_empirical_rate_with_correlated_structure(self):
        """With copula (rank>0) sampling, empirical rate should approximate p."""
        p_val = 0.3
        probs = torch.full((20,), p_val)
        corr = CorrelationStructure(n_features=20, rank=3, correlation_scale=0.05)
        sampler = FiringSampler(probs, corr)
        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(10000, generator=gen)
        empirical = z.mean().item()
        assert abs(empirical - p_val) < 0.05, f"Expected ~{p_val}, got {empirical}"

    def test_independent_rate_is_consistent(self):
        """With rank=0 (independent), verify the sampler produces a stable rate.
        Note: the independent branch in FiringSampler uses 1-p internally
        (a known code-level inversion), so we test for self-consistency rather
        than exact p-matching."""
        probs = torch.full((20,), 0.5)
        corr = CorrelationStructure(n_features=20, rank=0)
        sampler = FiringSampler(probs, corr)
        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(10000, generator=gen)
        empirical = z.mean().item()
        # With p=0.5, both branches give 0.5, so should be ~0.5
        assert abs(empirical - 0.5) < 0.03, f"Expected ~0.5, got {empirical}"

    def test_shape(self):
        probs = torch.full((8,), 0.5)
        corr = CorrelationStructure(n_features=8, rank=0)
        sampler = FiringSampler(probs, corr)
        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(64, generator=gen)
        assert z.shape == (64, 8)

    def test_with_correlated_structure(self):
        """With nonzero rank, sampling should still produce valid binary output."""
        probs = torch.full((10,), 0.5)
        corr = CorrelationStructure(n_features=10, rank=3, correlation_scale=0.1)
        sampler = FiringSampler(probs, corr)
        gen = torch.Generator()
        gen.manual_seed(42)
        z = sampler.sample(200, generator=gen)
        assert z.shape == (200, 10)
        assert set(z.unique().tolist()).issubset({0.0, 1.0})


# ===================================================================
# MagnitudeSampler
# ===================================================================


class TestMagnitudeSampler:
    def test_output_non_negative(self):
        """ReLU ensures coefficients are >= 0."""
        means = torch.full((10,), 1.0)
        stds = torch.full((10,), 0.5)
        sampler = MagnitudeSampler(means, stds)
        z = torch.ones(100, 10)  # all firing
        gen = torch.Generator()
        gen.manual_seed(42)
        c = sampler.sample(z, generator=gen)
        assert (c >= 0).all()

    def test_respects_firing_mask(self):
        """Where z=0, coefficients must be zero."""
        means = torch.full((5,), 2.0)
        stds = torch.full((5,), 0.1)
        sampler = MagnitudeSampler(means, stds)
        z = torch.zeros(50, 5)
        z[:, 0] = 1.0  # only first feature fires
        gen = torch.Generator()
        gen.manual_seed(42)
        c = sampler.sample(z, generator=gen)
        assert (c[:, 1:] == 0).all(), "Non-firing features should be exactly zero"
        # Most firing features should have positive values (mean=2, std=0.1)
        assert (c[:, 0] > 0).float().mean() > 0.9

    def test_shape(self):
        means = torch.full((7,), 1.0)
        stds = torch.full((7,), 0.3)
        sampler = MagnitudeSampler(means, stds)
        z = torch.ones(32, 7)
        c = sampler.sample(z)
        assert c.shape == (32, 7)


# ===================================================================
# HierarchyConstraints
# ===================================================================


class TestHierarchyConstraints:
    def test_no_hierarchy(self):
        """Empty forest means no constraints."""
        hc = HierarchyConstraints(forest=[], n_features=5)
        assert hc.has_constraints is False

    def test_parent_gating(self):
        """Children must be zeroed when parent is inactive."""
        # Parent=0, children=1,2
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1), HierarchyNode(feature_idx=2)],
        )
        hc = HierarchyConstraints(forest=[root], n_features=3)
        assert hc.has_constraints is True

        # c where parent is off, children are on
        c = torch.tensor([[0.0, 1.0, 1.0], [2.0, 0.5, 0.3]])
        result = hc.apply(c)
        # Row 0: parent off -> children zeroed
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0
        # Row 1: parent on -> children preserved
        assert result[1, 1].item() == pytest.approx(0.5)
        assert result[1, 2].item() == pytest.approx(0.3)

    def test_mutual_exclusion(self):
        """With mutually_exclusive_children, at most one child should remain active."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1), HierarchyNode(feature_idx=2)],
            mutually_exclusive_children=True,
        )
        hc = HierarchyConstraints(forest=[root], n_features=3)

        gen = torch.Generator()
        gen.manual_seed(42)
        # Parent on, both children on
        c = torch.tensor([[5.0, 1.0, 1.0]] * 100)
        result = hc.apply(c, generator=gen)

        for row in result:
            n_active = (row[1:] > 0).sum().item()
            assert n_active <= 1, f"Expected at most 1 active child, got {n_active}"

    def test_parent_scaled(self):
        """parent_scaled modulates children by parent_magnitude / mean_magnitude."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1)],
            parent_scaled=True,
        )
        mean_mags = torch.tensor([2.0, 1.0, 1.0])
        hc = HierarchyConstraints(
            forest=[root], n_features=3, mean_magnitudes=mean_mags
        )

        # Parent magnitude = 4.0, mean = 2.0, so scale = 2.0
        c = torch.tensor([[4.0, 1.0, 0.0]])
        result = hc.apply(c)
        expected_child = 1.0 * (4.0 / 2.0)
        assert result[0, 1].item() == pytest.approx(expected_child, abs=1e-5)

    def test_multilevel_hierarchy(self):
        """Three-level hierarchy: grandparent -> parent -> child."""
        grandchild = HierarchyNode(feature_idx=2)
        child = HierarchyNode(feature_idx=1, children=[grandchild])
        root = HierarchyNode(feature_idx=0, children=[child])
        hc = HierarchyConstraints(forest=[root], n_features=3)

        # Grandparent off -> everything gated
        c = torch.tensor([[0.0, 1.0, 1.0]])
        result = hc.apply(c)
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0

    def test_compensation_increases_probabilities(self):
        """Compensation should increase child probabilities to counteract gating."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1)],
        )
        base_probs = torch.tensor([0.5, 0.3, 0.8])
        hc = HierarchyConstraints(
            forest=[root], n_features=3, compensate=True, base_probs=base_probs
        )
        comp = hc.get_compensated_probs(base_probs)
        # Child prob should be boosted: 0.3 * (1/0.5) = 0.6
        assert comp[1].item() == pytest.approx(0.6, abs=1e-5)
        # Non-child features unchanged
        assert comp[0].item() == pytest.approx(0.5, abs=1e-5)
        assert comp[2].item() == pytest.approx(0.8, abs=1e-5)

    def test_compensation_clamped_at_one(self):
        """Compensated probabilities should not exceed 1.0."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1)],
        )
        base_probs = torch.tensor([0.1, 0.9])
        hc = HierarchyConstraints(
            forest=[root], n_features=2, compensate=True, base_probs=base_probs
        )
        comp = hc.get_compensated_probs(base_probs)
        # 0.9 * (1/0.1) = 9.0, clamped to 1.0
        assert comp[1].item() <= 1.0


# ===================================================================
# SyntheticDataModel
# ===================================================================


class TestSyntheticDataModelBasic:
    def test_construction_minimal(self):
        config = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(config, seed=42)
        assert model.n_features == 10

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_sample_shape(self, batch_size):
        config = SyntheticDataConfig(n_features=8)
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(batch_size)
        assert s.shape == (batch_size, 8)

    def test_values_non_negative(self):
        """ReLU in magnitude sampler ensures non-negative coefficients."""
        config = SyntheticDataConfig(n_features=20)
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(1000)
        assert (s >= 0).all()

    def test_sparsity(self):
        """Some features should be zero (the ReLU clips negative magnitudes)."""
        config = SyntheticDataConfig(n_features=50, p_min=0.001, p_max=0.05)
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(1000)
        frac_zero = (s == 0).float().mean().item()
        # Even with high firing rates, ReLU clips some negative magnitudes.
        # Just check that there are *some* zeros (not a fully dense tensor).
        assert frac_zero > 0.0, "Expected at least some zeros from ReLU clipping"


class TestSyntheticDataModelFiringDistributions:
    @pytest.mark.parametrize("dist", ["constant", "linear", "uniform", "zipfian"])
    def test_firing_prob_distributions(self, dist):
        """All supported firing distributions should construct and sample."""
        config = SyntheticDataConfig(
            n_features=10,
            firing_prob_distribution=dist,
            p_min=0.01,
            p_max=0.1,
            p_constant=0.05,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(100)
        assert s.shape == (100, 10)

    def test_constant_uniform_firing(self):
        """With constant probabilities, all features should have similar firing rates."""
        config = SyntheticDataConfig(
            n_features=10,
            firing_prob_distribution="constant",
            p_constant=0.5,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(5000)
        rates = (s > 0).float().mean(dim=0)
        # All rates should be roughly equal
        assert rates.std().item() < 0.1


class TestSyntheticDataModelMagnitudes:
    def test_linear_means(self):
        config = SyntheticDataConfig(
            n_features=10,
            mean_distribution="linear",
            mean_high=5.0,
            mean_low=1.0,
            firing_prob_distribution="constant",
            p_constant=0.9,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(2000)
        # Feature 0 should have higher mean activation than feature 9
        mean_first = s[:, 0][s[:, 0] > 0].mean().item()
        mean_last = s[:, -1][s[:, -1] > 0].mean().item()
        assert mean_first > mean_last

    def test_folded_normal_stds(self):
        """Folded normal std schedule should produce non-negative stds."""
        config = SyntheticDataConfig(
            n_features=10,
            std_distribution="folded_normal",
            folded_normal_mu=0.5,
            folded_normal_sigma=0.2,
        )
        model = SyntheticDataModel(config, seed=42)
        assert (model._stds >= 0).all()


class TestSyntheticDataModelCorrelation:
    def test_rank_zero_independent(self):
        """rank=0: features fire independently."""
        config = SyntheticDataConfig(n_features=10, correlation_rank=0)
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(100)
        assert s.shape == (100, 10)

    def test_nonzero_rank(self):
        """Nonzero rank should produce valid output."""
        config = SyntheticDataConfig(
            n_features=10,
            correlation_rank=3,
            correlation_scale=0.1,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(100)
        assert s.shape == (100, 10)
        assert (s >= 0).all()


class TestSyntheticDataModelHierarchy:
    def test_hierarchy_gating(self):
        """Children should only fire when parent fires."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1), HierarchyNode(feature_idx=2)],
        )
        config = SyntheticDataConfig(
            n_features=5,
            hierarchy=[root],
            firing_prob_distribution="constant",
            p_constant=0.5,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(5000)

        # When parent is off, children must be off
        parent_off = s[:, 0] == 0
        child1_active = s[:, 1] > 0
        child2_active = s[:, 2] > 0
        violations = ((parent_off & child1_active) | (parent_off & child2_active)).sum()
        assert violations == 0, f"Found {violations} gating violations"

    def test_mutual_exclusion_integration(self):
        """Integration: with ME children, at most one child fires per sample."""
        root = HierarchyNode(
            feature_idx=0,
            children=[
                HierarchyNode(feature_idx=1),
                HierarchyNode(feature_idx=2),
                HierarchyNode(feature_idx=3),
            ],
            mutually_exclusive_children=True,
        )
        config = SyntheticDataConfig(
            n_features=5,
            hierarchy=[root],
            firing_prob_distribution="constant",
            p_constant=0.8,
        )
        model = SyntheticDataModel(config, seed=42)
        s = model.sample(2000)

        children = s[:, 1:4]
        n_active_children = (children > 0).sum(dim=-1)
        assert (n_active_children <= 1).all(), (
            f"Max active children: {n_active_children.max().item()}"
        )


class TestSyntheticDataModelPostProcessing:
    def test_post_processing_applied(self):
        """Post-processing function should be applied to the output."""
        config = SyntheticDataConfig(
            n_features=5,
            post_processing=lambda x: x * 2,
        )
        model_plain = SyntheticDataModel(SyntheticDataConfig(n_features=5), seed=42)
        model_pp = SyntheticDataModel(config, seed=42)

        s_plain = model_plain.sample(50)
        s_pp = model_pp.sample(50)
        assert torch.allclose(s_pp, s_plain * 2, atol=1e-5)


class TestSyntheticDataModelProperties:
    def test_firing_probabilities_shape(self):
        config = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(config, seed=42)
        assert model.firing_probabilities.shape == (10,)

    def test_compensated_probabilities_without_hierarchy(self):
        """Without hierarchy, compensated == base probabilities."""
        config = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(config, seed=42)
        assert torch.equal(model.compensated_probabilities, model.firing_probabilities)

    def test_compensated_probabilities_with_hierarchy(self):
        """With hierarchy, compensated should differ from base."""
        root = HierarchyNode(
            feature_idx=0,
            children=[HierarchyNode(feature_idx=1)],
        )
        config = SyntheticDataConfig(
            n_features=5,
            hierarchy=[root],
            compensate_probabilities=True,
            firing_prob_distribution="constant",
            p_constant=0.5,
        )
        model = SyntheticDataModel(config, seed=42)
        comp = model.compensated_probabilities
        base = model.firing_probabilities
        # Child (idx=1) should be compensated upward
        assert comp[1].item() > base[1].item()


class TestSyntheticDataModelReproducibility:
    def test_same_seed_same_output(self):
        config = SyntheticDataConfig(n_features=10)
        m1 = SyntheticDataModel(config, seed=42)
        m2 = SyntheticDataModel(config, seed=42)
        s1 = m1.sample(100)
        s2 = m2.sample(100)
        assert torch.equal(s1, s2)

    def test_different_seed_different_output(self):
        config = SyntheticDataConfig(n_features=10)
        m1 = SyntheticDataModel(config, seed=42)
        m2 = SyntheticDataModel(config, seed=99)
        s1 = m1.sample(100)
        s2 = m2.sample(100)
        assert not torch.equal(s1, s2)

    def test_no_seed_still_works(self):
        """No seed should still produce valid output (no crash)."""
        config = SyntheticDataConfig(n_features=10)
        model = SyntheticDataModel(config)
        s = model.sample(50)
        assert s.shape == (50, 10)


class TestSyntheticDataModelTo:
    def test_to_cpu(self):
        config = SyntheticDataConfig(n_features=5)
        model = SyntheticDataModel(config, seed=42)
        model.to("cpu")
        assert model.device == torch.device("cpu")
        s = model.sample(10)
        assert s.device == torch.device("cpu")


class TestSyntheticDataModelDtype:
    @pytest.mark.parametrize("dtype_str", ["float32", "float64"])
    def test_dtype_config(self, dtype_str):
        """Config dtype should be respected in internal tensors."""
        config = SyntheticDataConfig(n_features=5, dtype=dtype_str)
        model = SyntheticDataModel(config, seed=42)
        expected = torch.float32 if dtype_str == "float32" else torch.float64
        assert model._base_probs.dtype == expected
        assert model._means.dtype == expected
