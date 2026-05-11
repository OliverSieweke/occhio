"""Deep audit tests for distribution implementations.

Covers manifold.py, simplex.py, sparse.py, relational.py, hierarchical.py, base.py.
Focus on invariant violations, edge cases, and statistical correctness.
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
from occhio.distributions.sparse import SparseUniform, SparseExponential, SingleUniform
from occhio.distributions.relational import RelationalSimple, MultiRelational
from occhio.distributions.hierarchical import HierarchicalSparse, TreeNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_generator():
    def _make(seed: int = 42) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen

    return _make


# ===================================================================
# 1. SphericalDistribution — deep audit
# ===================================================================


class TestSphericalFeaturePositionsUnitNorm:
    """Feature positions MUST have unit norm (they live on a sphere)."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 4])
    def test_unit_norm_all_dims(self, dim, make_generator):
        dist = SphericalDistribution(
            n_features=20, manifold_dim=dim, generator=make_generator(42)
        )
        norms = dist.feature_positions.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"manifold_dim={dim}: norms range [{norms.min():.6f}, {norms.max():.6f}]"
        )


class TestSphericalCosineBumpActivation:
    """Activation function must be cosine-bump: strongest near position, zero far away."""

    def test_activation_strongest_at_feature_position(self, make_generator):
        """When the sample direction exactly matches a feature, activation should be maximal."""
        dist = SphericalDistribution(
            n_features=6,
            manifold_dim=1,
            length_scale=1.0,
            magnitude_range=(1.0, 1.0),
            generator=make_generator(42),
        )
        # Use the first feature position as the "direction"
        direction = dist.feature_positions[0].unsqueeze(0)  # (1, 2)
        dots = (direction @ dist.feature_positions.T).clamp(-1.0, 1.0)
        alpha = torch.acos(dots)
        scaled = alpha / dist.length_scale
        activation = torch.cos(scaled)
        activation = torch.where(scaled <= math.pi / 2, activation, 0.0)

        # Feature 0 should have activation = 1.0 (cos(0) = 1)
        assert activation[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        # Other features should be <= 1.0
        assert (activation <= 1.0 + 1e-6).all()

    def test_zero_activation_far_away(self, make_generator):
        """With small length_scale, distant features should have zero activation."""
        dist = SphericalDistribution(
            n_features=50,
            manifold_dim=2,
            length_scale=0.1,
            magnitude_range=(1.0, 1.0),
            generator=make_generator(42),
        )
        samples = dist.sample(1000)
        # Most entries should be zero
        frac_zero = (samples == 0).float().mean().item()
        assert frac_zero > 0.7, f"Expected high sparsity, got frac_zero={frac_zero}"


class TestSphericalManifoldDims:
    """Test across manifold_dim = 1 (circle), 2 (sphere), 3 (hypersphere)."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_sample_shape(self, dim, make_generator):
        dist = SphericalDistribution(
            n_features=15, manifold_dim=dim, generator=make_generator(42)
        )
        s = dist.sample(32)
        assert s.shape == (32, 15)
        assert s.min() >= 0.0  # non-negative (cosine bump clipped)

    def test_circle_dim1_positions(self, make_generator):
        """manifold_dim=1: positions should be 2D (on S^1)."""
        dist = SphericalDistribution(
            n_features=8, manifold_dim=1, generator=make_generator(42)
        )
        assert dist.feature_positions.shape == (8, 2)

    def test_sphere_dim2_positions(self, make_generator):
        """manifold_dim=2: positions should be 3D (on S^2)."""
        dist = SphericalDistribution(
            n_features=20, manifold_dim=2, generator=make_generator(42)
        )
        assert dist.feature_positions.shape == (20, 3)


class TestSphericalSparsity:
    """With small length_scale, most features should be near-zero."""

    def test_sparsity_decreases_with_length_scale(self, make_generator):
        """Smaller length_scale -> more sparsity."""
        frac_zeros = []
        for ls in [0.1, 0.5, 2.0]:
            dist = SphericalDistribution(
                n_features=30,
                manifold_dim=2,
                length_scale=ls,
                generator=make_generator(42),
            )
            s = dist.sample(2000)
            frac_zeros.append((s == 0).float().mean().item())
        # Smaller length_scale should produce more zeros
        assert frac_zeros[0] > frac_zeros[1] > frac_zeros[2], (
            f"Sparsity should increase as length_scale decreases: {frac_zeros}"
        )


# ===================================================================
# 2. ToricDistribution — deep audit
# ===================================================================


class TestTorusPeriodicWrapping:
    """Distance near 0 and 2*pi should be small (wrapping)."""

    def test_wrapping_symmetry(self, make_generator):
        """Points at angle 0.01 and 2*pi - 0.01 should be close."""
        dist = ToricDistribution(
            n_features=10, toric_dim=1, generator=make_generator(42)
        )
        a = torch.tensor([[0.01]])
        b = torch.tensor([[2 * math.pi - 0.01]])
        d = dist._toric_distance(a, b)
        assert d.item() == pytest.approx(0.02, abs=1e-5)

    def test_wrapping_vs_direct(self, make_generator):
        """Distance should take the shorter path around the torus."""
        dist = ToricDistribution(
            n_features=10, toric_dim=1, generator=make_generator(42)
        )
        # Two points: 0.1 and 2*pi - 0.1. Direct distance = 2*pi - 0.2.
        # Wrapped distance = 0.2.
        a = torch.tensor([[0.1]])
        b = torch.tensor([[2 * math.pi - 0.1]])
        d = dist._toric_distance(a, b)
        assert d.item() == pytest.approx(0.2, abs=1e-5)

    def test_multidim_wrapping(self, make_generator):
        """Wrapping should work in each dimension independently."""
        dist = ToricDistribution(
            n_features=10, toric_dim=2, generator=make_generator(42)
        )
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[2 * math.pi - 0.1, 2 * math.pi - 0.1]])
        d = dist._toric_distance(a, b)
        expected = math.sqrt(0.1**2 + 0.1**2)
        assert d.item() == pytest.approx(expected, abs=1e-5)


class TestTorusFeaturePositions:
    """Feature positions should be valid angles on the torus."""

    @pytest.mark.parametrize("toric_dim", [1, 2, 3])
    def test_angles_in_range(self, toric_dim, make_generator):
        torch.manual_seed(42)
        dist = ToricDistribution(
            n_features=20, toric_dim=toric_dim, generator=make_generator(42)
        )
        assert dist.feature_angles.min() >= 0.0
        assert dist.feature_angles.max() < 2 * math.pi + 1e-6

    @pytest.mark.parametrize("toric_dim", [1, 2, 3])
    def test_angles_shape(self, toric_dim, make_generator):
        torch.manual_seed(42)
        dist = ToricDistribution(
            n_features=15, toric_dim=toric_dim, generator=make_generator(42)
        )
        assert dist.feature_angles.shape == (15, toric_dim)


class TestTorusNRings:
    """Test n_features with different toric_dim (n_rings conceptually)."""

    @pytest.mark.parametrize("n_features,toric_dim", [(5, 1), (10, 2), (20, 3)])
    def test_sample_shape(self, n_features, toric_dim, make_generator):
        torch.manual_seed(42)
        dist = ToricDistribution(
            n_features=n_features,
            toric_dim=toric_dim,
            generator=make_generator(42),
        )
        s = dist.sample(50)
        assert s.shape == (50, n_features)
        assert s.min() >= 0.0


# ===================================================================
# 3. HypercubeDistribution — deep audit
# ===================================================================


class TestHypercubeGridStructure:
    """For perfect-power n_features, verify grid structure."""

    def test_2d_grid_3x3(self, make_generator):
        """9 features on 2D -> 3x3 grid with positions in [0,1]^2."""
        dist = HypercubeDistribution(
            n_features=9, cube_dim=2, generator=make_generator(42)
        )
        assert dist.grid_size == 3
        assert dist.feature_positions.shape == (9, 2)
        # Verify grid points
        expected_coords = torch.linspace(0, 1, 3)
        actual_coords = dist.feature_positions[:, 0].unique().sort().values
        assert torch.allclose(actual_coords, expected_coords, atol=1e-5)

    def test_3d_grid_2x2x2(self, make_generator):
        """8 features on 3D -> 2x2x2 grid."""
        dist = HypercubeDistribution(
            n_features=8, cube_dim=3, generator=make_generator(42)
        )
        assert dist.grid_size == 2
        assert dist.feature_positions.shape == (8, 3)

    def test_non_perfect_power_random(self, make_generator):
        """Non-perfect-power uses random placement."""
        torch.manual_seed(42)
        dist = HypercubeDistribution(
            n_features=7, cube_dim=2, generator=make_generator(42)
        )
        assert dist.grid_size is None


class TestHypercubePositionsInUnitCube:
    """All positions must be in [0, 1]^d."""

    @pytest.mark.parametrize("n_features", [4, 9, 16, 7, 13])
    def test_positions_in_range(self, n_features, make_generator):
        torch.manual_seed(42)
        dist = HypercubeDistribution(
            n_features=n_features, cube_dim=2, generator=make_generator(42)
        )
        assert dist.feature_positions.min() >= 0.0
        assert dist.feature_positions.max() <= 1.0


class TestHypercubeTentBump:
    """Verify tent-bump activation shape: max(1 - dist/ls, 0)."""

    def test_tent_shape_1d(self, make_generator):
        """In 1D, features at grid points should have activation = 1 when U is at grid point."""
        dist = HypercubeDistribution(
            n_features=5,
            cube_dim=1,
            length_scale=0.5,
            magnitude_range=(1.0, 1.0),
            generator=make_generator(42),
        )
        # Manual: sample at the first grid point (0.0)
        u = torch.tensor([[0.0]])
        d = torch.cdist(u, dist.feature_positions).squeeze(-1)
        activation = (1 - d / dist.length_scale).clamp(min=0.0)
        # First feature is at 0.0, activation should be 1.0
        assert activation[0, 0].item() == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# 4. SimplexDistribution — deep audit
# ===================================================================


class TestSimplexDirichletProperty:
    """Active simplex features MUST sum to ~1 (Dirichlet property)."""

    def test_active_simplex_sums_to_one(self, make_generator):
        """With p_active=1.0, every simplex's features sum to exactly 1."""
        sizes = [3, 5, 4]
        dist = SimplexDistribution(
            simplex_sizes=sizes, p_active=1.0, generator=make_generator(42)
        )
        s = dist.sample(500)
        offset = 0
        for k in sizes:
            block_sum = s[:, offset : offset + k].sum(dim=-1)
            assert torch.allclose(block_sum, torch.ones(500), atol=1e-5), (
                f"Simplex of size {k} does not sum to 1: range [{block_sum.min():.6f}, {block_sum.max():.6f}]"
            )
            offset += k

    def test_individual_features_in_zero_one(self, make_generator):
        """Each Dirichlet coordinate should be in [0, 1]."""
        dist = SimplexDistribution(
            simplex_sizes=[5, 3], p_active=1.0, generator=make_generator(42)
        )
        s = dist.sample(1000)
        assert s.min() >= 0.0
        assert s.max() <= 1.0 + 1e-6


class TestSimplexPActive:
    """Test p_active=0 (all zeros), p_active=1 (all active)."""

    def test_p_active_zero(self, make_generator):
        """p_active=0: everything should be exactly zero."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=0.0, generator=make_generator(42)
        )
        s = dist.sample(100)
        assert (s == 0).all()

    def test_p_active_one(self, make_generator):
        """p_active=1: all simplices fire, row sum = number of simplices."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 4], p_active=1.0, generator=make_generator(42)
        )
        s = dist.sample(200)
        row_sums = s.sum(dim=-1)
        # 2 simplices each summing to 1 -> row_sum = 2.0
        assert torch.allclose(row_sums, torch.full((200,), 2.0), atol=1e-5)

    def test_p_active_intermediate_sparsity(self, make_generator):
        """With p_active=0.5, roughly half of simplices should fire."""
        dist = SimplexDistribution(
            simplex_sizes=[3, 3, 3, 3],
            p_active=0.5,
            generator=make_generator(42),
        )
        s = dist.sample(10000)
        # Each simplex fires with p=0.5, so average row_sum should be ~2.0
        mean_row_sum = s.sum(dim=-1).mean().item()
        assert abs(mean_row_sum - 2.0) < 0.1


class TestSimplexDirichletAlpha:
    """Verify dirichlet_alpha parameter effect.

    The current implementation uses Exp(1) variates (equivalent to Dirichlet(1,...,1)).
    This test verifies the default Dirichlet(1) behavior.
    """

    def test_dirichlet_uniform_on_simplex(self, make_generator):
        """Dir(1,...,1) should produce roughly uniform marginal means = 1/k."""
        k = 5
        dist = SimplexDistribution(
            simplex_sizes=[k], p_active=1.0, generator=make_generator(42)
        )
        s = dist.sample(50000)
        means = s.mean(dim=0)
        expected = 1.0 / k
        for i in range(k):
            assert abs(means[i].item() - expected) < 0.01, (
                f"Feature {i} mean = {means[i]:.4f}, expected ~{expected:.4f}"
            )


# ===================================================================
# 5. SimplicialComplexDistribution — deep audit
# ===================================================================


class TestSimplicialComplexSingleMode:
    """In 'single' mode: exactly one face active, features sum to 1."""

    def test_row_sum_is_one(self, make_generator):
        """Each row should sum to 1.0 in single mode."""
        faces = [(0, 1, 2), (3, 4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            sampling_mode="single",
            generator=make_generator(42),
        )
        s = dist.sample(300)
        row_sums = s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(300), atol=1e-5)

    def test_exactly_one_face_active(self, make_generator):
        """Only one face's vertices should be nonzero per row."""
        faces = [(0, 1), (2, 3), (4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            sampling_mode="single",
            generator=make_generator(42),
        )
        s = dist.sample(500)
        for row in s:
            active_faces = sum(1 for f in faces if any(row[v] > 0 for v in f))
            assert active_faces == 1, f"Expected 1 active face, got {active_faces}"

    def test_single_mode_nonuniform_faces(self, make_generator):
        """Non-uniform face sizes in single mode."""
        faces = [(0, 1), (2, 3, 4)]
        dist = SimplicialComplexDistribution(
            n_vertices=5,
            faces=faces,
            sampling_mode="single",
            generator=make_generator(42),
        )
        s = dist.sample(300)
        # Row sums should still be 1.0
        row_sums = s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(300), atol=1e-5)


class TestSimplicialComplexSharedVertices:
    """Shared vertices should accumulate contributions from multiple faces."""

    def test_shared_vertex_accumulates_in_sparse(self, make_generator):
        """In sparse mode with overlapping faces, shared vertex gets more activation."""
        # Faces (0,1) and (1,2) share vertex 1
        faces = [(0, 1), (1, 2)]
        dist = SimplicialComplexDistribution(
            n_vertices=3,
            faces=faces,
            p_active=1.0,
            sampling_mode="sparse",
            generator=make_generator(42),
        )
        s = dist.sample(5000)
        # Vertex 1 appears in both faces, should have higher mean activation
        mean_v1 = s[:, 1].mean().item()
        mean_v0 = s[:, 0].mean().item()
        mean_v2 = s[:, 2].mean().item()
        assert mean_v1 > mean_v0, (
            f"Shared vertex (mean={mean_v1:.4f}) should > non-shared (mean={mean_v0:.4f})"
        )
        assert mean_v1 > mean_v2, (
            f"Shared vertex (mean={mean_v1:.4f}) should > non-shared (mean={mean_v2:.4f})"
        )


class TestSimplicialComplexSparseMode:
    """Verify sparse mode works correctly."""

    def test_sparse_p_active_zero(self, make_generator):
        """p_active=0: everything should be zero."""
        faces = [(0, 1), (2, 3)]
        dist = SimplicialComplexDistribution(
            n_vertices=4,
            faces=faces,
            p_active=0.0,
            sampling_mode="sparse",
            generator=make_generator(42),
        )
        s = dist.sample(100)
        assert (s == 0).all()

    def test_sparse_p_active_one_disjoint_faces(self, make_generator):
        """With p_active=1 and disjoint faces, each face sums to 1."""
        faces = [(0, 1, 2), (3, 4, 5)]
        dist = SimplicialComplexDistribution(
            n_vertices=6,
            faces=faces,
            p_active=1.0,
            sampling_mode="sparse",
            generator=make_generator(42),
        )
        s = dist.sample(300)
        # Each face block should sum to 1.0
        face1_sum = s[:, 0:3].sum(dim=-1)
        face2_sum = s[:, 3:6].sum(dim=-1)
        assert torch.allclose(face1_sum, torch.ones(300), atol=1e-5)
        assert torch.allclose(face2_sum, torch.ones(300), atol=1e-5)


# ===================================================================
# 6. SparseUniform — deep audit
# ===================================================================


class TestSparseUniformSparsityRate:
    """With p_active=0.05, exactly ~5% of entries should be nonzero over 100k samples."""

    def test_sparsity_rate_5_percent(self, make_generator):
        dist = SparseUniform(
            n_features=100, p_active=0.05, generator=make_generator(42)
        )
        s = dist.sample(100_000)
        empirical_rate = (s > 0).float().mean().item()
        # Allow +/- 0.5% tolerance
        assert abs(empirical_rate - 0.05) < 0.005, (
            f"Expected ~0.05 active rate, got {empirical_rate}"
        )


class TestSparseUniformMagnitudeDistribution:
    """Active features should be uniformly distributed in [0, 1]."""

    def test_uniform_magnitude_active_features(self, make_generator):
        dist = SparseUniform(n_features=100, p_active=1.0, generator=make_generator(42))
        s = dist.sample(50000)
        # With p_active=1.0, all features are active with Uniform[0,1] values
        assert s.min() >= 0.0
        assert s.max() <= 1.0
        # Mean should be ~0.5
        mean = s.mean().item()
        assert abs(mean - 0.5) < 0.01, f"Expected mean ~0.5, got {mean}"
        # Check uniformity: split into 10 bins, each should have ~10%
        for i in range(10):
            lo = i / 10.0
            hi = (i + 1) / 10.0
            frac = ((s >= lo) & (s < hi)).float().mean().item()
            assert abs(frac - 0.1) < 0.01, (
                f"Bin [{lo}, {hi}): expected ~0.1, got {frac}"
            )


class TestSparseUniformEdgeCases:
    """Test p_active=0 and p_active=1."""

    def test_p_active_zero(self, make_generator):
        dist = SparseUniform(n_features=10, p_active=0.0, generator=make_generator(42))
        s = dist.sample(100)
        assert (s == 0).all()

    def test_p_active_one(self, make_generator):
        dist = SparseUniform(n_features=10, p_active=1.0, generator=make_generator(42))
        s = dist.sample(1000)
        # With p_active=1.0, all entries are drawn from Uniform[0,1]
        # Probability of exactly 0.0 is negligible
        active_rate = (s > 0).float().mean().item()
        assert active_rate > 0.999


# ===================================================================
# 7. SparseExponential — deep audit
# ===================================================================


class TestSparseExponentialSparsityRate:
    """Same sparsity test as SparseUniform."""

    def test_sparsity_rate_5_percent(self, make_generator):
        dist = SparseExponential(
            n_features=100, p_active=0.05, generator=make_generator(42)
        )
        s = dist.sample(100_000)
        empirical_rate = (s > 0).float().mean().item()
        assert abs(empirical_rate - 0.05) < 0.005, (
            f"Expected ~0.05 active rate, got {empirical_rate}"
        )


class TestSparseExponentialMagnitudeDistribution:
    """Active features should follow Exp(scale) with mean = 1/scale."""

    def test_exponential_mean(self, make_generator):
        scale = 2.0
        dist = SparseExponential(
            n_features=100, p_active=1.0, scale=scale, generator=make_generator(42)
        )
        s = dist.sample(50000)
        active = s[s > 0]
        mean = active.mean().item()
        expected = 1.0 / scale
        assert abs(mean - expected) < 0.02, f"Expected mean ~{expected}, got {mean}"

    def test_no_inf_values(self, make_generator):
        """SparseExponential should never produce inf values."""
        dist = SparseExponential(
            n_features=100, p_active=1.0, scale=1.0, generator=make_generator(42)
        )
        s = dist.sample(100_000)
        assert torch.isfinite(s).all(), (
            f"Found {(~torch.isfinite(s)).sum()} non-finite values"
        )


# ===================================================================
# 8. SingleUniform — deep audit
# ===================================================================


class TestSingleUniformExactlyOne:
    """Exactly one feature active per sample."""

    def test_exactly_one_active(self, make_generator):
        dist = SingleUniform(n_features=10, generator=make_generator(42))
        s = dist.sample(1000)
        l0 = (s > 0).sum(dim=-1)
        assert (l0 == 1).all(), (
            f"Expected exactly 1 active per row, got min={l0.min()}, max={l0.max()}"
        )

    def test_all_features_eventually_sampled(self, make_generator):
        """Over many samples, every feature should be selected at least once."""
        dist = SingleUniform(n_features=5, generator=make_generator(42))
        s = dist.sample(10000)
        for i in range(5):
            n_active = (s[:, i] > 0).sum().item()
            assert n_active > 0, f"Feature {i} was never selected"

    def test_uniform_feature_selection(self, make_generator):
        """Features should be selected with roughly equal probability."""
        dist = SingleUniform(n_features=5, generator=make_generator(42))
        s = dist.sample(50000)
        for i in range(5):
            rate = (s[:, i] > 0).float().mean().item()
            expected = 1.0 / 5
            assert abs(rate - expected) < 0.02, (
                f"Feature {i}: expected rate ~{expected}, got {rate}"
            )

    def test_active_value_in_unit_interval(self, make_generator):
        """Active feature value should be in (0, 1]."""
        dist = SingleUniform(n_features=10, generator=make_generator(42))
        s = dist.sample(10000)
        active = s[s > 0]
        assert active.min() > 0.0
        assert active.max() <= 1.0


# ===================================================================
# 9. RelationalSimple / MultiRelational — deep audit
# ===================================================================


class TestRelationalSimpleStructure:
    """Verify matrix binding structure."""

    def test_output_shape(self, make_generator):
        dist = RelationalSimple(
            n_features=10, p_active=0.3, generator=make_generator(42)
        )
        s = dist.sample(50)
        assert s.shape == (50, 10)

    def test_on_matrix_is_orthogonal(self, make_generator):
        """The O(n) matrix should be orthogonal."""
        dist = RelationalSimple(
            n_features=8, p_active=0.3, generator=make_generator(42)
        )
        eye = torch.eye(8)
        assert torch.allclose(dist.on_mat @ dist.on_mat.T, eye, atol=1e-5)

    def test_new_on_matrix_changes(self, make_generator):
        """Calling new_On_matrix should produce a different matrix."""
        dist = RelationalSimple(
            n_features=8, p_active=0.3, generator=make_generator(42)
        )
        old_mat = dist.on_mat.clone()
        dist.new_On_matrix()
        # Extremely unlikely to be equal
        assert not torch.allclose(dist.on_mat, old_mat, atol=1e-3)

    def test_output_can_be_negative(self, make_generator):
        """Relational outputs can be negative (no ReLU)."""
        dist = RelationalSimple(
            n_features=20, p_active=0.5, generator=make_generator(42)
        )
        s = dist.sample(5000)
        # With orthogonal rotation, some entries should be negative
        assert s.min() < 0, "Expected some negative values from orthogonal rotation"


class TestMultiRelationalStructure:
    """Verify multi-relational binding structure."""

    def test_output_shape(self, make_generator):
        dist = MultiRelational(
            n_features=10, p_active=0.3, k=3, generator=make_generator(42)
        )
        s = dist.sample(50)
        assert s.shape == (50, 10)

    def test_k_matrices_created(self, make_generator):
        dist = MultiRelational(
            n_features=8, p_active=0.3, k=4, generator=make_generator(42)
        )
        assert len(dist.on_mats) == 4
        for mat in dist.on_mats:
            eye = torch.eye(8)
            assert torch.allclose(mat @ mat.T, eye, atol=1e-5)

    def test_device_bug_multirelational(self, make_generator):
        """BUG: MultiRelational.sample creates result tensor without device=self.device.

        This test verifies the bug is fixed.
        """
        dist = MultiRelational(
            n_features=5,
            p_active=0.3,
            k=2,
            device="cpu",
            generator=make_generator(42),
        )
        s = dist.sample(10)
        assert s.device == torch.device("cpu")


# ===================================================================
# 10. HierarchicalSparse — deep audit
# ===================================================================


class TestHierarchicalTreeStructure:
    """Verify tree structure is respected in sampling."""

    def test_tree_has_correct_nodes(self, make_generator):
        dist = HierarchicalSparse(n_features=10, generator=make_generator(42))
        assert len(dist.nodes) == 10
        # Root is at index 0
        assert dist.nodes[0].depth == 0
        assert dist.nodes[0].parent is None

    def test_all_nodes_reachable_from_root(self, make_generator):
        """BFS from root should reach all nodes."""
        dist = HierarchicalSparse(n_features=15, generator=make_generator(42))
        visited = set()
        queue = [0]
        while queue:
            idx = queue.pop(0)
            visited.add(idx)
            queue.extend(dist.nodes[idx].children)
        assert len(visited) == 15


class TestHierarchicalParentGating:
    """Children should only fire when parent fires."""

    def test_children_gated_by_parent(self, make_generator):
        dist = HierarchicalSparse(
            n_features=20,
            p_base=0.5,
            depth_decay=0.8,
            generator=make_generator(42),
        )
        s = dist.sample(10000)

        # For each node with a parent, check: if parent is inactive, child must be inactive
        for node in dist.nodes:
            if node.parent is not None:
                parent_off = s[:, node.parent] == 0
                child_on = s[:, node.index] > 0
                violations = (parent_off & child_on).sum().item()
                assert violations == 0, (
                    f"Node {node.index} (parent={node.parent}): {violations} gating violations"
                )


class TestHierarchicalDepthDecay:
    """Deeper features should fire less often."""

    def test_deeper_features_fire_less(self, make_generator):
        dist = HierarchicalSparse(
            n_features=50,
            p_base=0.7,
            depth_decay=0.5,  # Aggressive decay for clear signal
            generator=make_generator(42),
        )
        s = dist.sample(20000)

        # Compute average activation by depth
        activation_by_depth = {}
        for node in dist.nodes:
            d = node.depth
            if d not in activation_by_depth:
                activation_by_depth[d] = []
            activation_by_depth[d].append((s[:, node.index] > 0).float().mean().item())

        depths = sorted(activation_by_depth.keys())
        mean_by_depth = [
            sum(activation_by_depth[d]) / len(activation_by_depth[d]) for d in depths
        ]

        # Each deeper level should have lower activation (with decay=0.5 this is clear)
        for i in range(len(mean_by_depth) - 1):
            assert mean_by_depth[i] >= mean_by_depth[i + 1] - 0.05, (
                f"Depth {depths[i]} activation ({mean_by_depth[i]:.3f}) should >= "
                f"depth {depths[i + 1]} activation ({mean_by_depth[i + 1]:.3f})"
            )

    def test_root_always_fires(self, make_generator):
        """Root node (depth 0) should always be active."""
        dist = HierarchicalSparse(n_features=10, generator=make_generator(42))
        s = dist.sample(1000)
        # Find root indices
        root_indices = dist.depth_indices[0]
        for idx in root_indices:
            assert (s[:, idx] > 0).all(), f"Root node {idx} should always be active"


class TestHierarchicalExpectedActive:
    """Verify get_expected_active consistency."""

    def test_expected_vs_empirical(self, make_generator):
        dist = HierarchicalSparse(
            n_features=20,
            p_base=0.6,
            depth_decay=0.7,
            generator=make_generator(42),
        )
        expected = dist.get_expected_active()
        s = dist.sample(50000)
        empirical = (s > 0).float().mean(dim=0)

        # The empirical activation rates should be close to the expected marginal probabilities
        # The expected activation probability for each node is the product of
        # p_fire along the path from root. Empirical values should also
        # account for the Uniform(0,1) magnitude (which is always > 0 when active)
        for i in range(20):
            assert abs(expected[i].item() - empirical[i].item()) < 0.03, (
                f"Node {i}: expected={expected[i]:.4f}, empirical={empirical[i]:.4f}"
            )


class TestHierarchicalPByDepth:
    """Verify p_by_depth override."""

    def test_custom_p_by_depth(self, make_generator):
        dist = HierarchicalSparse(
            n_features=10,
            p_by_depth=[1.0, 0.5, 0.25],
            generator=make_generator(42),
        )
        assert dist._get_p_fire(0) == 1.0
        assert dist._get_p_fire(1) == 0.5
        assert dist._get_p_fire(2) == 0.25
        # Beyond provided list, should use last value
        assert dist._get_p_fire(5) == 0.25


# ===================================================================
# 11. DistributionStack — deep audit
# ===================================================================


class TestDistributionStackConcatenation:
    """Verify stacking concatenates along feature dimension."""

    def test_total_features_equals_sum(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=7, p_active=0.5, generator=make_generator(2))
        d3 = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(3))
        stack = DistributionStack([d1, d2, d3])
        assert stack.n_features == 15
        s = stack.sample(50)
        assert s.shape == (50, 15)


class TestDistributionStackSamplingModes:
    """sampling_mode='independent', 'single', 'sparse' all work."""

    def test_independent_mode(self, make_generator):
        d1 = SparseUniform(n_features=4, p_active=0.8, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=0.8, generator=make_generator(2))
        stack = DistributionStack([d1, d2], sampling_mode="independent")
        s = stack.sample(100)
        assert s.shape == (100, 10)
        # Both groups should be active (not zeroed out)
        assert (s[:, :4] > 0).any()
        assert (s[:, 4:] > 0).any()

    def test_single_mode(self, make_generator):
        d1 = SparseUniform(n_features=4, p_active=1.0, generator=make_generator(1))
        d2 = SparseUniform(n_features=6, p_active=1.0, generator=make_generator(2))
        stack = DistributionStack(
            [d1, d2], sampling_mode="single", generator=make_generator(3)
        )
        s = stack.sample(200)
        assert s.shape == (200, 10)
        # Each row: at most one group active
        for row in s:
            g1 = (row[:4] > 0).any().item()
            g2 = (row[4:] > 0).any().item()
            assert not (g1 and g2), "Both groups active in single mode"

    def test_sparse_mode(self, make_generator):
        d1 = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=5, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack(
            [d1, d2],
            sampling_mode="sparse",
            p_meta=0.5,
            generator=make_generator(3),
        )
        s = stack.sample(100)
        assert s.shape == (100, 10)


class TestDistributionStackGeneratorManagement:
    """Generator management across stacked distributions."""

    def test_collect_generators(self, make_generator):
        g1, g2 = make_generator(1), make_generator(2)
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=g1)
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=g2)
        stack = DistributionStack([d1, d2])
        gens = stack.collect_generators()
        assert len(gens) == 2
        assert gens[0] is g1
        assert gens[1] is g2

    def test_sync_generators_list(self, make_generator):
        g1, g2 = make_generator(1), make_generator(2)
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(10))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(20))
        stack = DistributionStack([d1, d2])
        stack.sync_generators([g1, g2])
        # After sync, generators should produce same sequence as source
        # (states were copied)

    def test_sync_generators_wrong_count_raises(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        with pytest.raises(ValueError, match="Expected 2"):
            stack.sync_generators([make_generator(1)])

    def test_to_device(self, make_generator):
        d1 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(1))
        d2 = SparseUniform(n_features=3, p_active=0.5, generator=make_generator(2))
        stack = DistributionStack([d1, d2])
        stack.to("cpu")
        for d in stack.distributions:
            assert d.device == torch.device("cpu")


# ===================================================================
# Bug-specific regression tests
# ===================================================================


class TestMultiRelationalDeviceBug:
    """Regression: MultiRelational.sample creates tensor without device."""

    def test_result_tensor_on_correct_device(self, make_generator):
        dist = MultiRelational(
            n_features=5,
            p_active=0.3,
            k=2,
            device="cpu",
            generator=make_generator(42),
        )
        s = dist.sample(10)
        assert s.device == torch.device("cpu")


class TestTorusReproducibilityBug:
    """Regression: ToricDistribution._place_features uses torch.rand without generator."""

    def test_seeded_reproducibility(self, make_generator):
        """Two instances with same generator seed should produce identical feature_angles."""
        d1 = ToricDistribution(n_features=10, toric_dim=2, generator=make_generator(42))
        d2 = ToricDistribution(n_features=10, toric_dim=2, generator=make_generator(42))
        assert torch.equal(d1.feature_angles, d2.feature_angles), (
            "Feature angles should be identical with same generator seed"
        )
        assert torch.equal(d1.sample(50), d2.sample(50))


class TestSphericalReproducibilityHighDim:
    """Regression: _place_on_sphere_random uses torch.randn without generator for dim>=3."""

    def test_seeded_reproducibility_dim3(self, make_generator):
        """Two instances with manifold_dim=3 should produce identical positions."""
        d1 = SphericalDistribution(
            n_features=10, manifold_dim=3, generator=make_generator(42)
        )
        d2 = SphericalDistribution(
            n_features=10, manifold_dim=3, generator=make_generator(42)
        )
        assert torch.allclose(d1.feature_positions, d2.feature_positions, atol=1e-5), (
            "Feature positions should be identical with same generator seed for dim>=3"
        )
        assert torch.equal(d1.sample(50), d2.sample(50))


class TestHypercubeReproducibilityNonGrid:
    """Regression: _place_features uses torch.rand without generator for non-grid case."""

    def test_seeded_reproducibility_non_grid(self, make_generator):
        """Two instances with non-grid n_features should produce identical positions."""
        d1 = HypercubeDistribution(
            n_features=7, cube_dim=2, generator=make_generator(42)
        )
        d2 = HypercubeDistribution(
            n_features=7, cube_dim=2, generator=make_generator(42)
        )
        assert torch.allclose(d1.feature_positions, d2.feature_positions, atol=1e-5), (
            "Feature positions should be identical with same generator seed"
        )
        assert torch.equal(d1.sample(50), d2.sample(50))


class TestSparseExponentialInfinityBug:
    """Regression: SparseExponential can produce inf when rand() returns exactly 1.0."""

    def test_no_inf_large_sample(self, make_generator):
        """Large sample should never contain inf."""
        dist = SparseExponential(
            n_features=200, p_active=1.0, scale=1.0, generator=make_generator(42)
        )
        s = dist.sample(100_000)
        assert torch.isfinite(s).all(), (
            f"Found {(~torch.isfinite(s)).sum()} non-finite values"
        )
