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
        """k=5 circles (S^1, m=2) → output (batch, 10)."""
        dist = SparseSpheres(k=5, n=1, p_active=0.3, generator=seeded_generator)
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_sample_shape_spheres(self, seeded_generator):
        """k=3 spheres (S^2, m=3) → output (batch, 9)."""
        dist = SparseSpheres(k=3, n=2, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(64)
        assert samples.shape == (64, 9)

    def test_sample_shape_tilted(self, seeded_generator):
        """k=4 circles tilted into 3D ambient → output (batch, 12)."""
        dist = SparseSpheres(k=4, n=1, m=3, p_active=0.3, generator=seeded_generator)
        samples = dist.sample(50)
        assert samples.shape == (50, 12)

    def test_n_features_computed(self, seeded_generator):
        """n_features should be k * m."""
        dist = SparseSpheres(k=5, n=1, p_active=0.3, generator=seeded_generator)
        assert dist.n_features == 10

    def test_n_features_computed_tilted(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, m=4, p_active=0.3, generator=seeded_generator)
        assert dist.n_features == 12


# ── Tilt matrices ────────────────────────────────────────────────────────


class TestSparseSpheresTilts:
    def test_tilts_shape_no_tilt(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.3, generator=seeded_generator)
        assert dist.tilts.shape == (3, 2, 2)

    def test_tilts_identity_no_tilt(self, seeded_generator):
        dist = SparseSpheres(k=3, n=2, p_active=0.3, generator=seeded_generator)
        for i in range(3):
            assert torch.allclose(dist.tilts[i], torch.eye(3), atol=1e-6)

    def test_tilts_shape_with_tilt(self, seeded_generator):
        dist = SparseSpheres(k=4, n=1, m=5, p_active=0.3, generator=seeded_generator)
        assert dist.tilts.shape == (4, 5, 2)

    def test_tilts_orthonormal_columns(self, seeded_generator):
        dist = SparseSpheres(k=4, n=1, m=5, p_active=0.3, generator=seeded_generator)
        for i in range(4):
            R = dist.tilts[i]
            gram = R.T @ R
            assert torch.allclose(gram, torch.eye(2), atol=1e-5)

    def test_tilts_no_grad(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.3, generator=seeded_generator)
        assert not dist.tilts.requires_grad


# ── Centers ──────────────────────────────────────────────────────────────


class TestSparseSpheresCenters:
    def test_centers_shape(self, seeded_generator):
        dist = SparseSpheres(k=4, n=1, m=3, p_active=0.3, generator=seeded_generator)
        assert dist.centers.shape == (4, 3)

    def test_centers_identity_tilt(self, seeded_generator):
        """With identity tilt, center = r * ones(m) since each row of I has norm 1."""
        r = 2.0
        dist = SparseSpheres(k=2, n=1, p_active=0.3, r=r, generator=seeded_generator)
        expected = torch.full((2, 2), r)
        assert torch.allclose(dist.centers, expected, atol=1e-6)

    def test_centers_positive(self, seeded_generator):
        """Centers should always be non-negative (they are norms scaled by r)."""
        dist = SparseSpheres(k=5, n=1, m=4, p_active=0.3, generator=seeded_generator)
        assert (dist.centers >= 0).all()


# ── Non-negative orthant ─────────────────────────────────────────────────


class TestSparseSpheresNonNegative:
    def test_all_coordinates_non_negative(self, seeded_generator):
        """All active sample coordinates should be >= 0."""
        dist = SparseSpheres(k=5, n=1, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()

    def test_all_coordinates_non_negative_tilted(self, seeded_generator):
        """Non-negative should hold even with tilt."""
        dist = SparseSpheres(k=3, n=1, m=4, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()

    def test_all_coordinates_non_negative_spheres(self, seeded_generator):
        """Non-negative should hold for S^2 as well."""
        dist = SparseSpheres(k=3, n=2, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(5000)
        assert (samples >= -1e-6).all()


# ── Activity ─────────────────────────────────────────────────────────────


class TestSparseSpheresActivity:
    def test_all_inactive_gives_zeros(self, seeded_generator):
        dist = SparseSpheres(k=5, n=1, p_active=0.0, generator=seeded_generator)
        samples = dist.sample(200)
        assert (samples == 0).all()

    def test_all_active_nonzero(self, seeded_generator):
        dist = SparseSpheres(k=5, n=1, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(500)
        assert (samples.abs().sum(dim=1) > 0).all()

    def test_empirical_activity_rate(self, seeded_generator):
        p = 0.3
        k = 5
        dist = SparseSpheres(k=k, n=1, p_active=p, generator=seeded_generator)
        samples = dist.sample(10000)
        active_frac = (samples.abs().sum(dim=1) > 0).float().mean().item()
        expected_active = 1 - (1 - p) ** k
        assert abs(active_frac - expected_active) < 0.03


# ── Sphere sampling ──────────────────────────────────────────────────────


class TestSparseSpheresSampling:
    def test_centered_block_norms_circle(self, seeded_generator):
        """k=1, S^1, p_active=1: block - center should have norm r."""
        r = 2.0
        dist = SparseSpheres(k=1, n=1, p_active=1.0, r=r, generator=seeded_generator)
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_centered_block_norms_sphere(self, seeded_generator):
        """k=1, S^2, p_active=1: block - center should have norm r."""
        r = 1.5
        dist = SparseSpheres(k=1, n=2, p_active=1.0, r=r, generator=seeded_generator)
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_axis_aligned_single_feature(self, seeded_generator):
        """With k=2, p_active=[1.0, 0.0], only first m dims should be nonzero."""
        dist = SparseSpheres(k=2, n=1, p_active=[1.0, 0.0], generator=seeded_generator)
        samples = dist.sample(500)
        assert (samples[:, 2:] == 0).all()
        assert (samples[:, :2].abs().sum(dim=1) > 0).all()

    def test_axis_aligned_second_feature(self, seeded_generator):
        """With k=2, p_active=[0.0, 1.0], only last m dims should be nonzero."""
        dist = SparseSpheres(k=2, n=1, p_active=[0.0, 1.0], generator=seeded_generator)
        samples = dist.sample(500)
        assert (samples[:, :2] == 0).all()
        assert (samples[:, 2:].abs().sum(dim=1) > 0).all()

    def test_per_feature_centered_norms(self, seeded_generator):
        """With p_active=1, each (block - center) should have norm r."""
        r = 1.0
        k, n = 4, 1
        m = n + 1
        dist = SparseSpheres(k=k, n=n, p_active=1.0, r=r, generator=seeded_generator)
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
            k=k, n=n, m=m, p_active=1.0, r=r, generator=seeded_generator
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
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        result = dist.sample(100)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 6)

    def test_with_labels_true_returns_tuple(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        samples, labels = dist.sample_with_args(100, with_labels=True)
        assert samples.shape == (100, 6)
        assert labels.shape == (100, 3)
        assert labels.dtype == torch.bool

    def test_with_labels_false_returns_tensor(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        result = dist.sample_with_args(100, with_labels=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 6)

    def test_default_with_labels_is_true(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        result = dist.sample_with_args(100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_labels_match_activity(self, seeded_generator):
        k, n = 5, 1
        m = n + 1
        dist = SparseSpheres(k=k, n=n, p_active=0.5, generator=seeded_generator)
        samples, labels = dist.sample_with_args(1000)
        for i in range(k):
            block = samples[:, i * m : (i + 1) * m]
            block_active = block.abs().sum(dim=1) > 0
            assert torch.equal(labels[:, i], block_active)

    def test_labels_all_true_p1(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=1.0, generator=seeded_generator)
        _, labels = dist.sample_with_args(100)
        assert labels.all()

    def test_labels_all_false_p0(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.0, generator=seeded_generator)
        _, labels = dist.sample_with_args(100)
        assert not labels.any()


# ── Gaussian noise ───────────────────────────────────────────────────────


class TestSparseSpheresNoise:
    def test_noise_std_stored(self, seeded_generator):
        dist = SparseSpheres(
            k=3, n=1, p_active=0.5, noise_std=0.1, generator=seeded_generator
        )
        assert dist.noise_std == 0.1

    def test_noise_std_default_zero(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        assert dist.noise_std == 0.0

    def test_no_noise_exact_norms(self, seeded_generator):
        """Without noise, centered block norms should exactly equal r."""
        r = 1.0
        dist = SparseSpheres(k=1, n=1, p_active=1.0, r=r, generator=seeded_generator)
        samples = dist.sample(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_noise_perturbs_norms_via_init(self, seeded_generator):
        """With noise_std on object, centered block norms should deviate from r."""
        r = 1.0
        dist = SparseSpheres(
            k=1, n=1, p_active=1.0, r=r, noise_std=0.5, generator=seeded_generator
        )
        samples = dist.sample(1000)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        deviations = (norms - r).abs()
        assert deviations.max() > 0.05

    def test_noise_perturbs_norms_via_param(self, seeded_generator):
        """Passing noise_std at call time should perturb norms."""
        r = 1.0
        dist = SparseSpheres(k=1, n=1, p_active=1.0, r=r, generator=seeded_generator)
        samples = dist.sample(1000, noise_std=0.5)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        deviations = (norms - r).abs()
        assert deviations.max() > 0.05

    def test_param_overrides_init(self, seeded_generator):
        """noise_std param should override self.noise_std."""
        r = 1.0
        dist = SparseSpheres(
            k=1, n=1, p_active=1.0, r=r, noise_std=0.5, generator=seeded_generator
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
            k=1, n=1, p_active=1.0, r=r, noise_std=0.5, generator=seeded_generator
        )
        samples, _ = dist.sample_with_args(500, noise_std=0.0)
        centered = samples - dist.centers.reshape(1, -1)
        norms = centered.norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, r), atol=1e-5)

    def test_noise_only_on_active_features(self, seeded_generator):
        """Inactive features should remain exactly zero even with noise_std > 0."""
        dist = SparseSpheres(k=2, n=1, p_active=[1.0, 0.0], generator=seeded_generator)
        samples = dist.sample(500, noise_std=0.5)
        # Second feature (indices 2:4) is always inactive → must be zero
        assert (samples[:, 2:] == 0).all()
        # First feature should be nonzero
        assert (samples[:, :2].abs().sum(dim=1) > 0).all()

    def test_noise_does_not_affect_inactive_samples(self, seeded_generator):
        """Fully inactive samples (p_active=0) should be all zeros even with noise."""
        dist = SparseSpheres(k=3, n=1, p_active=0.0, generator=seeded_generator)
        samples = dist.sample(200, noise_std=1.0)
        assert (samples == 0).all()

    def test_noise_preserves_same_tilts(self):
        """noise_std should not affect tilt matrices (same seed → same tilts)."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        dist1 = SparseSpheres(
            k=3, n=1, m=4, p_active=0.3, noise_std=0.0, generator=gen1
        )
        dist2 = SparseSpheres(
            k=3, n=1, m=4, p_active=0.3, noise_std=0.5, generator=gen2
        )
        assert torch.equal(dist1.tilts, dist2.tilts)
        assert torch.equal(dist1.centers, dist2.centers)


# ── Reproducibility ──────────────────────────────────────────────────────


class TestSparseSpheresReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(k=5, n=1, p_active=0.3, generator=gen1)
        dist2 = SparseSpheres(k=5, n=1, p_active=0.3, generator=gen2)
        s1 = dist1.sample(200)
        s2 = dist2.sample(200)
        assert torch.equal(s1, s2)

    def test_same_seed_same_tilts(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(k=3, n=1, m=4, p_active=0.3, generator=gen1)
        dist2 = SparseSpheres(k=3, n=1, m=4, p_active=0.3, generator=gen2)
        assert torch.equal(dist1.tilts, dist2.tilts)

    def test_same_seed_same_noisy_samples_via_init(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(k=3, n=1, p_active=0.5, noise_std=0.1, generator=gen1)
        dist2 = SparseSpheres(k=3, n=1, p_active=0.5, noise_std=0.1, generator=gen2)
        s1 = dist1.sample(200)
        s2 = dist2.sample(200)
        assert torch.equal(s1, s2)

    def test_same_seed_same_noisy_samples_via_param(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)
        dist1 = SparseSpheres(k=3, n=1, p_active=0.5, generator=gen1)
        dist2 = SparseSpheres(k=3, n=1, p_active=0.5, generator=gen2)
        s1 = dist1.sample(200, noise_std=0.1)
        s2 = dist2.sample(200, noise_std=0.1)
        assert torch.equal(s1, s2)


# ── Validation ───────────────────────────────────────────────────────────


class TestSparseSpheresValidation:
    def test_m_less_than_n_plus_1_raises(self, seeded_generator):
        with pytest.raises(ValueError, match="m.*must be.*>=.*n"):
            SparseSpheres(k=3, n=2, m=2, p_active=0.3, generator=seeded_generator)

    def test_n_features_mismatch_warns(self, seeded_generator):
        with pytest.warns(UserWarning, match="n_features"):
            SparseSpheres(
                n_features=999, k=3, n=1, p_active=0.3, generator=seeded_generator
            )


# ── Broadcast p_active ───────────────────────────────────────────────────


class TestSparseSpheresBroadcast:
    def test_scalar_p_active(self, seeded_generator):
        dist = SparseSpheres(k=3, n=1, p_active=0.5, generator=seeded_generator)
        assert dist.p_active.shape == (3,)
        assert (dist.p_active == 0.5).all()

    def test_list_p_active(self, seeded_generator):
        p_list = [0.1, 0.5, 0.9]
        dist = SparseSpheres(k=3, n=1, p_active=p_list, generator=seeded_generator)
        assert dist.p_active.shape == (3,)
        assert torch.allclose(dist.p_active, torch.tensor(p_list))

    def test_tensor_p_active(self, seeded_generator):
        p_t = torch.tensor([0.2, 0.8])
        dist = SparseSpheres(k=2, n=1, p_active=p_t, generator=seeded_generator)
        assert dist.p_active.shape == (2,)
        assert torch.allclose(dist.p_active, p_t)

    def test_per_feature_activity(self, seeded_generator):
        dist = SparseSpheres(k=2, n=1, p_active=[0.0, 1.0], generator=seeded_generator)
        samples = dist.sample(500)
        assert (samples[:, :2] == 0).all()
        assert (samples[:, 2:].abs().sum(dim=1) > 0).all()


# ── Angles uniform on S^1 ───────────────────────────────────────────────


class TestSparseSpheresAngles:
    def test_angles_uniform_circle(self, seeded_generator):
        """With k=1, S^1, p_active=1, angles should be uniformly distributed."""
        dist = SparseSpheres(k=1, n=1, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(10000)
        # Subtract center to recover the sphere point, then compute angles
        centered = samples - dist.centers.reshape(1, -1)
        angles = torch.atan2(centered[:, 1], centered[:, 0])
        for lo in torch.linspace(-math.pi, math.pi, 9)[:-1]:
            hi = lo + 2 * math.pi / 8
            count = ((angles >= lo) & (angles < hi)).sum().item()
            expected = 10000 / 8
            assert abs(count - expected) / expected < 0.15
