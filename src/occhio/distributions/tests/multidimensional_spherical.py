"""Tests for SparseSpheres distribution (axis-aligned S^n features, non-negative orthant)."""

import math

import pytest
import torch

from ..sphere import SparseSpheres


@pytest.fixture
def seeded_generator():
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


# ── Shape ────────────────────────────────────────────────────────────────


class TestSparseSpheresShape:
    def test_sample_shape_circles(self, seeded_generator):
        """k=5 circles (S^1, ambient_dim=2) → output (batch, 10)."""
        dist = SparseSpheres(
            n_spheres=5, sphere_dim=1, p_active=0.3, generator=seeded_generator
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_sample_shape_spheres(self, seeded_generator):
        """k=3 spheres (S^2, ambient_dim=3) → output (batch, 9)."""
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=2, p_active=0.5, generator=seeded_generator
        )
        samples = dist.sample(64)
        assert samples.shape == (64, 9)

    def test_sample_shape_tilted(self, seeded_generator):
        """k=4 circles tilted into 3D ambient → output (batch, 12)."""
        dist = SparseSpheres(
            n_spheres=4,
            sphere_dim=1,
            ambient_dim=3,
            p_active=0.3,
            generator=seeded_generator,
        )
        samples = dist.sample(50)
        assert samples.shape == (50, 12)

    def test_n_features_computed(self, seeded_generator):
        """n_features should be k * m."""
        dist = SparseSpheres(
            n_spheres=5, sphere_dim=1, p_active=0.3, generator=seeded_generator
        )
        assert dist.n_features == 10

    def test_n_features_computed_tilted(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            ambient_dim=4,
            p_active=0.3,
            generator=seeded_generator,
        )
        assert dist.n_features == 12


# ── Tilt matrices ────────────────────────────────────────────────────────


class TestSparseSpheresTilts:
    def test_tilts_shape_no_tilt(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.3, generator=seeded_generator
        )
        assert dist.tilts.shape == (3, 2, 2)

    def test_tilts_identity_no_tilt(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=2, p_active=0.3, generator=seeded_generator
        )
        for i in range(3):
            assert torch.allclose(dist.tilts[i], torch.eye(3), atol=1e-6)

    def test_tilts_shape_with_tilt(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=4,
            sphere_dim=1,
            ambient_dim=5,
            p_active=0.3,
            generator=seeded_generator,
        )
        assert dist.tilts.shape == (4, 5, 2)

    def test_tilts_orthonormal_columns(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=4,
            sphere_dim=1,
            ambient_dim=5,
            p_active=0.3,
            generator=seeded_generator,
        )
        for i in range(4):
            R = dist.tilts[i]
            gram = R.T @ R
            assert torch.allclose(gram, torch.eye(2), atol=1e-5)

    def test_tilts_no_grad(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.3, generator=seeded_generator
        )
        assert not dist.tilts.requires_grad


# ── Centers ──────────────────────────────────────────────────────────────


class TestSparseSpheresCenters:
    def test_centers_shape(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=4,
            sphere_dim=1,
            ambient_dim=3,
            p_active=0.3,
            generator=seeded_generator,
        )
        assert dist.centers.shape == (4, 3)

    def test_centers_identity_tilt(self, seeded_generator):
        """With identity tilt, center = r * ones(m) since each row of I has norm 1."""
        r = 2.0
        dist = SparseSpheres(
            n_spheres=2,
            sphere_dim=1,
            p_active=0.3,
            radius=r,
            generator=seeded_generator,
        )
        expected = torch.full((2, 2), r)
        assert torch.allclose(dist.centers, expected, atol=1e-6)

    def test_centers_positive(self, seeded_generator):
        """Centers should always be non-negative (they are norms scaled by r)."""
        dist = SparseSpheres(
            n_spheres=5,
            sphere_dim=1,
            ambient_dim=4,
            p_active=0.3,
            generator=seeded_generator,
        )
        assert (dist.centers >= 0).all()


# ── Non-negative orthant ─────────────────────────────────────────────────


class TestSparseSpheresNonNegative:
    def test_all_coordinates_non_negative(self, seeded_generator):
        """All active sample coordinates should be >= 0."""
        dist = SparseSpheres(
            n_spheres=5, sphere_dim=1, p_active=1.0, generator=seeded_generator
        )
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()

    def test_all_coordinates_non_negative_tilted(self, seeded_generator):
        """Non-negative should hold even with tilt."""
        dist = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            ambient_dim=4,
            p_active=1.0,
            generator=seeded_generator,
        )
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()

    def test_all_coordinates_non_negative_spheres(self, seeded_generator):
        """Non-negative should hold for S^2 as well."""
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=2, p_active=1.0, generator=seeded_generator
        )
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()


# ── Activity ─────────────────────────────────────────────────────────────


class TestSparseSpheresActivity:
    def test_all_inactive_gives_zeros(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=5, sphere_dim=1, p_active=0.0, generator=seeded_generator
        )
        samples = dist.sample(200)
        assert (samples == 0).all()

    def test_all_active_nonzero(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=5, sphere_dim=1, p_active=1.0, generator=seeded_generator
        )
        samples = dist.sample(500)
        assert (samples.abs().sum(dim=1) > 0).all()

    def test_empirical_activity_rate(self, seeded_generator):
        p = 0.3
        k = 5
        dist = SparseSpheres(
            n_spheres=k, sphere_dim=1, p_active=p, generator=seeded_generator
        )
        samples = dist.sample(10000)
        active_frac = (samples.abs().sum(dim=1) > 0).float().mean().item()
        expected_active = 1 - (1 - p) ** k
        assert abs(active_frac - expected_active) < 0.03


# ── Sphere sampling ──────────────────────────────────────────────────────


class TestSparseSpheresSampling:
    def test_centered_block_norms_circle(self, seeded_generator):
        """k=1, S^1, p_active=1: block - center should have norm r."""
        r = 2.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_centered_block_norms_sphere(self, seeded_generator):
        """k=1, S^2, p_active=1: block - center should have norm r."""
        r = 1.5
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=2,
            p_active=1.0,
            p_infill=0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_axis_aligned_single_feature(self, seeded_generator):
        """With n_spheres=2, p_active=[1.0, 0.0], only first m dims should be nonzero."""
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=[1.0, 0.0], generator=seeded_generator
        )
        samples = dist.sample(500)
        assert (samples[:, 2:] == 0).all()
        assert (samples[:, :2].abs().sum(dim=1) > 0).all()

    def test_axis_aligned_second_feature(self, seeded_generator):
        """With n_spheres=2, p_active=[0.0, 1.0], only last m dims should be nonzero."""
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=[0.0, 1.0], generator=seeded_generator
        )
        samples = dist.sample(500)
        assert (samples[:, :2] == 0).all()
        assert (samples[:, 2:].abs().sum(dim=1) > 0).all()

    def test_per_feature_centered_norms(self, seeded_generator):
        """With p_active=1, each (block - center) should have norm r."""
        r = 1.0
        k, n = 4, 1
        m = n + 1
        dist = SparseSpheres(
            n_spheres=k,
            sphere_dim=n,
            p_active=1.0,
            p_infill=0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(500)
        for i in range(k):
            block = samples[:, i * m : (i + 1) * m]
            centered = block - dist.centers[i]
            norms = centered.norm(dim=1)
            assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_tilted_feature_centered_norms(self, seeded_generator):
        """With tilt (m > n+1), each (block - center) should have norm r."""
        r = 1.0
        k, n, m = 3, 1, 4
        dist = SparseSpheres(
            n_spheres=k,
            sphere_dim=n,
            ambient_dim=m,
            p_active=1.0,
            p_infill=0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(500)
        for i in range(k):
            block = samples[:, i * m : (i + 1) * m]
            centered = block - dist.centers[i]
            norms = centered.norm(dim=1)
            assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)


# ── sample_with_args ─────────────────────────────────────────────────────


class TestSparseSpheresWithArgs:
    def test_sample_returns_tensor(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        result = dist.sample(100)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 6)

    def test_with_labels_true_returns_tuple(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        samples, labels = dist.sample_with_args(100, with_labels=True)
        assert samples.shape == (100, 6)
        assert labels.shape == (100, 3)
        assert labels.dtype == torch.bool

    def test_with_labels_false_returns_tensor(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        result = dist.sample_with_args(100, with_labels=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 6)

    def test_default_with_labels_is_true(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        result = dist.sample_with_args(100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_labels_match_activity(self, seeded_generator):
        k, n = 5, 1
        m = n + 1
        dist = SparseSpheres(
            n_spheres=k, sphere_dim=n, p_active=0.5, generator=seeded_generator
        )
        samples, labels = dist.sample_with_args(1000)
        for i in range(k):
            block = samples[:, i * m : (i + 1) * m]
            block_active = block.abs().sum(dim=1) > 0
            assert torch.equal(labels[:, i], block_active)

    def test_labels_all_true_p1(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=1.0, generator=seeded_generator
        )
        _, labels = dist.sample_with_args(100)
        assert labels.all()

    def test_labels_all_false_p0(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.0, generator=seeded_generator
        )
        _, labels = dist.sample_with_args(100)
        assert not labels.any()


# ── Gaussian noise ───────────────────────────────────────────────────────


class TestSparseSpheresNoise:
    def test_noise_std_stored(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            p_active=0.5,
            noise_std=0.1,
            generator=seeded_generator,
        )
        assert dist.noise_std == 0.1

    def test_noise_std_default_zero(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        assert dist.noise_std == 0.0

    def test_no_noise_exact_norms(self, seeded_generator):
        """Without noise, centered block norms should exactly equal r."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_noise_perturbs_norms_via_init(self, seeded_generator):
        """With noise_std on object, centered block norms should deviate from r."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            radius=r,
            noise_std=0.5,
            generator=seeded_generator,
        )
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        deviations = (norms - r).abs()
        assert deviations.max() > 0.05

    def test_noise_perturbs_norms_via_param(self, seeded_generator):
        """Passing noise_std at call time should perturb norms."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            radius=r,
            generator=seeded_generator,
        )
        samples = dist.sample(1000, noise_std=0.5)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        deviations = (norms - r).abs()
        assert deviations.max() > 0.05

    def test_param_overrides_init(self, seeded_generator):
        """noise_std param should override self.noise_std."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            radius=r,
            noise_std=0.5,
            generator=seeded_generator,
        )
        # Override with 0.0 → should get exact norms
        samples = dist.sample(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_sample_with_args_noise_override(self, seeded_generator):
        """sample_with_args should accept noise_std override."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            radius=r,
            noise_std=0.5,
            generator=seeded_generator,
        )
        samples, _ = dist.sample_with_args(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_noise_only_on_active_features(self, seeded_generator):
        """Inactive features should remain exactly zero even with noise_std > 0."""
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=[1.0, 0.0], generator=seeded_generator
        )
        samples = dist.sample(500, noise_std=0.5)
        # Second feature (indices 2:4) is always inactive → must be zero
        assert (samples[:, 2:] == 0).all()
        # First feature should be nonzero
        assert (samples[:, :2].abs().sum(dim=1) > 0).all()

    def test_noise_does_not_affect_inactive_samples(self, seeded_generator):
        """Fully inactive samples (p_active=0) should be all zeros even with noise."""
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.0, generator=seeded_generator
        )
        samples = dist.sample(200, noise_std=1.0)
        assert (samples == 0).all()

    def test_noise_preserves_same_tilts(self):
        """noise_std should not affect tilt matrices (same seed → same tilts)."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        dist1 = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            ambient_dim=4,
            p_active=0.3,
            noise_std=0.0,
            generator=gen1,
        )
        dist2 = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            ambient_dim=4,
            p_active=0.3,
            noise_std=0.5,
            generator=gen2,
        )
        assert torch.equal(dist1.tilts, dist2.tilts)
        assert torch.equal(dist1.centers, dist2.centers)


# ── Reproducibility ──────────────────────────────────────────────────────


class TestSparseSpheresReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(n_spheres=5, sphere_dim=1, p_active=0.3, generator=gen1)
        dist2 = SparseSpheres(n_spheres=5, sphere_dim=1, p_active=0.3, generator=gen2)
        s1 = dist1.sample(200)
        s2 = dist2.sample(200)
        assert torch.equal(s1, s2)

    def test_same_seed_same_tilts(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(
            n_spheres=3, sphere_dim=1, ambient_dim=4, p_active=0.3, generator=gen1
        )
        dist2 = SparseSpheres(
            n_spheres=3, sphere_dim=1, ambient_dim=4, p_active=0.3, generator=gen2
        )
        assert torch.equal(dist1.tilts, dist2.tilts)

    def test_same_seed_same_noisy_samples_via_init(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, noise_std=0.1, generator=gen1
        )
        dist2 = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, noise_std=0.1, generator=gen2
        )
        s1 = dist1.sample(200)
        s2 = dist2.sample(200)
        assert torch.equal(s1, s2)

    def test_same_seed_same_noisy_samples_via_param(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(n_spheres=3, sphere_dim=1, p_active=0.5, generator=gen1)
        dist2 = SparseSpheres(n_spheres=3, sphere_dim=1, p_active=0.5, generator=gen2)
        s1 = dist1.sample(200, noise_std=0.1)
        s2 = dist2.sample(200, noise_std=0.1)
        assert torch.equal(s1, s2)


# ── Validation ───────────────────────────────────────────────────────────


class TestSparseSpheresValidation:
    def test_m_less_than_n_plus_1_raises(self, seeded_generator):
        with pytest.raises(ValueError, match="ambient_dim.*must be.*>=.*sphere_dim"):
            SparseSpheres(
                n_spheres=3,
                sphere_dim=2,
                ambient_dim=2,
                p_active=0.3,
                generator=seeded_generator,
            )

    def test_n_features_mismatch_warns(self, seeded_generator):
        with pytest.warns(UserWarning, match="n_features"):
            SparseSpheres(
                n_features=999,
                n_spheres=3,
                sphere_dim=1,
                p_active=0.3,
                generator=seeded_generator,
            )


# ── Broadcast p_active ───────────────────────────────────────────────────


class TestSparseSpheresBroadcast:
    def test_scalar_p_active(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        assert dist.p_active.shape == (3,)
        assert (dist.p_active == 0.5).all()

    def test_list_p_active(self, seeded_generator):
        p_list = [0.1, 0.5, 0.9]
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=p_list, generator=seeded_generator
        )
        assert dist.p_active.shape == (3,)
        assert torch.allclose(dist.p_active, torch.tensor(p_list))

    def test_tensor_p_active(self, seeded_generator):
        p_t = torch.tensor([0.2, 0.8])
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=p_t, generator=seeded_generator
        )
        assert dist.p_active.shape == (2,)
        assert torch.allclose(dist.p_active, p_t)

    def test_per_feature_activity(self, seeded_generator):
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=[0.0, 1.0], generator=seeded_generator
        )
        samples = dist.sample(500)
        assert (samples[:, :2] == 0).all()
        assert (samples[:, 2:].abs().sum(dim=1) > 0).all()


# ── Angles uniform on S^1 ───────────────────────────────────────────────


class TestSparseSpheresAngles:
    def test_angles_uniform_circle(self, seeded_generator):
        """With n_spheres=1, S^1, p_active=1, angles should be uniformly distributed."""
        dist = SparseSpheres(
            n_spheres=1, sphere_dim=1, p_active=1.0, generator=seeded_generator
        )
        samples = dist.sample(10000)
        # Subtract center to recover the sphere point, then compute angles
        centered = samples - dist.centers.reshape(1, -1)
        angles = torch.atan2(centered[:, 1], centered[:, 0])
        for lo in torch.linspace(-math.pi, math.pi, 9)[:-1]:
            hi = lo + 2 * math.pi / 8
            count = ((angles >= lo) & (angles < hi)).sum().item()
            expected = 10000 / 8
            assert abs(count - expected) / expected < 0.15


# ── Discretization ──────────────────────────────────────────────────────


class TestSparseSpheresDiscretization:
    def test_discrete_points_shape_s1(self):
        """S^1 with 8 discretizations → (8, 2) points."""
        from ..sphere import _make_discrete_points

        pts = _make_discrete_points(8, sphere_dim=1, device="cpu")
        assert pts.shape == (8, 2)

    def test_discrete_points_shape_s2(self):
        """S^2 with 20 discretizations → (20, 3) points."""
        from ..sphere import _make_discrete_points

        pts = _make_discrete_points(20, sphere_dim=2, device="cpu")
        assert pts.shape == (20, 3)

    def test_discrete_points_shape_s3(self):
        """S^3 (general case) with 50 discretizations → (50, 4) points."""
        from ..sphere import _make_discrete_points

        pts = _make_discrete_points(50, sphere_dim=3, device="cpu")
        assert pts.shape == (50, 4)

    def test_discrete_points_unit_norm(self):
        """All discrete points should have unit norm."""
        from ..sphere import _make_discrete_points

        for sd in [1, 2, 3, 5]:
            pts = _make_discrete_points(30, sphere_dim=sd, device="cpu")
            norms = pts.norm(dim=-1)
            assert torch.allclose(norms, torch.ones(30), atol=1e-5), (
                f"Failed for sphere_dim={sd}"
            )

    def test_s1_equidistant(self):
        """S^1 discrete points should be exactly equidistant."""
        from ..sphere import _make_discrete_points

        n = 12
        pts = _make_discrete_points(n, sphere_dim=1, device="cpu")
        # All pairwise distances between consecutive points (and wrap-around)
        # should equal the chord length for 2*pi/n angular separation.
        expected_chord = 2 * math.sin(math.pi / n)
        for i in range(n):
            d = (pts[i] - pts[(i + 1) % n]).norm().item()
            assert abs(d - expected_chord) < 1e-5

    def test_n_disc_zero_means_none(self, seeded_generator):
        """n_discretizations=0 should leave _discrete_points as None."""
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        assert dist._discrete_points is None

    def test_n_disc_positive_stores_points(self, seeded_generator):
        """n_discretizations > 0 should precompute _discrete_points."""
        dist = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            p_active=0.5,
            n_discretizations=10,
            generator=seeded_generator,
        )
        assert dist._discrete_points is not None
        assert dist._discrete_points.shape == (10, 2)

    def test_warning_with_noise(self, seeded_generator):
        """n_discretizations + noise_std > 0 should emit a warning."""
        with pytest.warns(UserWarning, match="n_discretizations.*noise_std"):
            SparseSpheres(
                n_spheres=3,
                sphere_dim=1,
                p_active=0.5,
                n_discretizations=10,
                noise_std=0.1,
                generator=seeded_generator,
            )

    def test_discrete_samples_land_on_point_set(self, seeded_generator):
        """With p_infill=0, all active samples should exactly match a discrete point."""
        n_disc = 8
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            n_discretizations=n_disc,
            generator=seeded_generator,
        )
        samples, _ = dist.sample_with_args(500)
        # Recover sphere points by subtracting center
        centered = samples - dist.centers.reshape(1, -1)
        centered_normed = centered / centered.norm(dim=-1, keepdim=True)
        # Each sample direction should match one of the discrete points
        # Compute cosine similarity with all discrete points
        disc = dist._discrete_points  # (n_disc, 2)
        sims = centered_normed @ disc.T  # (500, n_disc)
        max_sim = sims.max(dim=-1).values
        assert torch.allclose(max_sim, torch.ones_like(max_sim), atol=1e-5)

    def test_infill_bypasses_discretization(self, seeded_generator):
        """With p_infill=1 and n_disc>0, infill samples get continuous directions."""
        n_disc = 4
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=1.0,
            n_discretizations=n_disc,
            generator=seeded_generator,
        )
        samples, _ = dist.sample_with_args(1000)
        centered = samples - dist.centers.reshape(1, -1)
        # Infill samples should have radius < self.radius (almost surely)
        norms = centered.norm(dim=-1)
        # With p_infill=1, all samples are inside the ball, so norms < radius
        assert (norms < dist.radius - 1e-6).all()

    def test_discrete_reproducibility(self):
        """Same seed → same discrete samples."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        d1 = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=0.5, n_discretizations=8, generator=gen1
        )
        d2 = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=0.5, n_discretizations=8, generator=gen2
        )
        s1 = d1.sample(100)
        s2 = d2.sample(100)
        assert torch.equal(s1, s2)


# ── sample_with_args overrides ──────────────────────────────────────────


class TestSampleWithArgsOverrides:
    def test_p_active_override_float(self, seeded_generator):
        """Overriding p_active=0 should produce all-zero samples."""
        dist = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=1.0, generator=seeded_generator
        )
        samples, mask = dist.sample_with_args(200, p_active=0.0)
        assert (samples == 0).all()
        assert not mask.any()

    def test_p_active_override_tensor(self, seeded_generator):
        """Overriding p_active with a per-sphere tensor."""
        dist = SparseSpheres(
            n_spheres=2, sphere_dim=1, p_active=0.5, generator=seeded_generator
        )
        samples, mask = dist.sample_with_args(500, p_active=torch.tensor([1.0, 0.0]))
        # Second sphere should always be inactive
        assert not mask[:, 1].any()
        # First sphere should always be active
        assert mask[:, 0].all()

    def test_p_infill_override(self, seeded_generator):
        """Overriding p_infill=0 should give exact surface norms."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0.5,
            radius=r,
            generator=seeded_generator,
        )
        samples, _ = dist.sample_with_args(500, p_infill=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_n_disc_override(self, seeded_generator):
        """Overriding n_discretizations on a continuous dist should use discrete points."""
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            generator=seeded_generator,
        )
        assert dist.n_discretizations == 0
        samples, _ = dist.sample_with_args(500, n_discretizations=6)
        centered = samples - dist.centers.reshape(1, -1)
        centered_normed = centered / centered.norm(dim=-1, keepdim=True)
        # Directions should cluster into exactly 6 distinct angles
        angles = torch.atan2(centered_normed[:, 1], centered_normed[:, 0])
        unique_angles = angles.unique()
        # Should have at most 6 unique angles (within tolerance)
        assert len(unique_angles) <= 6

    def test_none_means_default(self, seeded_generator):
        """Passing None for all overrides should behave identically to no overrides."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        d1 = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, p_infill=0.2, generator=gen1
        )
        d2 = SparseSpheres(
            n_spheres=3, sphere_dim=1, p_active=0.5, p_infill=0.2, generator=gen2
        )
        s1, m1 = d1.sample_with_args(100)
        s2, m2 = d2.sample_with_args(
            100, noise_std=None, p_active=None, p_infill=None, n_discretizations=None
        )
        assert torch.equal(s1, s2)
        assert torch.equal(m1, m2)

    def test_no_mutation_after_override(self, seeded_generator):
        """Overriding params should not mutate self attributes."""
        dist = SparseSpheres(
            n_spheres=3,
            sphere_dim=1,
            p_active=0.5,
            p_infill=0.1,
            noise_std=0.0,
            n_discretizations=0,
            generator=seeded_generator,
        )
        orig_p_active = dist.p_active.clone()
        orig_p_infill = dist.p_infill
        orig_noise = dist.noise_std
        orig_n_disc = dist.n_discretizations

        dist.sample_with_args(
            100,
            p_active=0.9,
            p_infill=0.5,
            noise_std=0.3,
            n_discretizations=10,
        )

        assert torch.equal(dist.p_active, orig_p_active)
        assert dist.p_infill == orig_p_infill
        assert dist.noise_std == orig_noise
        assert dist.n_discretizations == orig_n_disc

    def test_backward_compat_noise_only(self, seeded_generator):
        """Existing noise_std-only calls should still work."""
        r = 1.0
        dist = SparseSpheres(
            n_spheres=1,
            sphere_dim=1,
            p_active=1.0,
            p_infill=0,
            radius=r,
            noise_std=0.5,
            generator=seeded_generator,
        )
        samples, _ = dist.sample_with_args(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)
