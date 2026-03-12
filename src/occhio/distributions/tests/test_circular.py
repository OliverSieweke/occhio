"""Tests for Bump distribution."""

import pytest
import torch

from ..bump import Bump


@pytest.fixture
def seeded_generator():
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


class TestBumpShape:
    def test_sample_shape(self, seeded_generator):
        dist = Bump(n_features=7, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(100)
        assert samples.shape == (100, 7)

    def test_sample_shape_large(self, seeded_generator):
        dist = Bump(
            n_features=20, p_active=0.8, bump_width=3, generator=seeded_generator
        )
        samples = dist.sample(500)
        assert samples.shape == (500, 20)


class TestBumpValues:
    def test_known_bump_vectors_k7_bw1(self, seeded_generator):
        """Verify bump vectors match the spec example: n_features=7, bump_width=1."""
        dist = Bump(
            n_features=7, p_active=1.0, bump_width=1, generator=seeded_generator
        )

        expected = {
            0: [1.0, 0.5, 0, 0, 0, 0, 0.5],
            1: [0.5, 1.0, 0.5, 0, 0, 0, 0],
            2: [0, 0.5, 1.0, 0.5, 0, 0, 0],
            3: [0, 0, 0.5, 1.0, 0.5, 0, 0],
            6: [0.5, 0, 0, 0, 0, 0.5, 1.0],
        }
        for j, vec in expected.items():
            expected_t = torch.tensor(vec)
            actual = dist._bump_matrix[j]
            assert torch.allclose(actual, expected_t), (
                f"State {j}: expected {vec}, got {actual.tolist()}"
            )

    def test_bump_peak_is_one(self, seeded_generator):
        """The active state position should always have value 1."""
        dist = Bump(
            n_features=10, p_active=1.0, bump_width=2, generator=seeded_generator
        )
        for j in range(10):
            assert dist._bump_matrix[j, j].item() == 1.0

    def test_bump_wraps_around(self, seeded_generator):
        """State 0 should have nonzero activation at the last positions."""
        dist = Bump(
            n_features=8, p_active=1.0, bump_width=2, generator=seeded_generator
        )
        bump_0 = dist._bump_matrix[0]
        # Positions 1,2 should be nonzero (right neighbors)
        assert bump_0[1].item() > 0
        assert bump_0[2].item() > 0
        # Positions 7,6 should be nonzero (wrapped left neighbors)
        assert bump_0[7].item() > 0
        assert bump_0[6].item() > 0
        # Position 3 should be zero (too far)
        assert bump_0[3].item() == 0
        # Position 5 should be zero (too far)
        assert bump_0[5].item() == 0

    def test_bump_symmetry(self, seeded_generator):
        """Bump should be symmetric around the peak due to circular distance."""
        dist = Bump(
            n_features=12, p_active=1.0, bump_width=3, generator=seeded_generator
        )
        for j in range(12):
            bump = dist._bump_matrix[j]
            for d in range(1, 4):
                left = bump[(j - d) % 12].item()
                right = bump[(j + d) % 12].item()
                assert left == pytest.approx(right), (
                    f"State {j}, distance {d}: left={left}, right={right}"
                )

    def test_bump_values_are_non_negative(self, seeded_generator):
        dist = Bump(
            n_features=10, p_active=1.0, bump_width=3, generator=seeded_generator
        )
        assert (dist._bump_matrix >= 0).all()

    def test_bump_max_is_one(self, seeded_generator):
        dist = Bump(
            n_features=10, p_active=1.0, bump_width=3, generator=seeded_generator
        )
        assert dist._bump_matrix.max().item() == 1.0

    def test_bump_linear_decay(self, seeded_generator):
        """Values should decay linearly: 1 - d/(bump_width+1)."""
        bw = 3
        dist = Bump(
            n_features=15, p_active=1.0, bump_width=bw, generator=seeded_generator
        )
        j = 5
        bump = dist._bump_matrix[j]
        for d in range(bw + 2):
            expected = max(0.0, 1.0 - d / (bw + 1))
            actual = bump[(j + d) % 15].item()
            assert actual == pytest.approx(expected), (
                f"d={d}: expected {expected}, got {actual}"
            )


class TestBumpActivity:
    def test_inactive_samples_are_zero(self, seeded_generator):
        dist = Bump(n_features=7, p_active=0.0, generator=seeded_generator)
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_all_active_no_zeros(self, seeded_generator):
        """With p_active=1, every sample should have a nonzero peak."""
        dist = Bump(n_features=7, p_active=1.0, generator=seeded_generator)
        samples = dist.sample(500)
        # Each row should have max value 1.0 (the peak)
        assert (samples.max(dim=1).values == 1.0).all()

    def test_empirical_activity_rate(self, seeded_generator):
        p = 0.4
        dist = Bump(n_features=10, p_active=p, bump_width=2, generator=seeded_generator)
        samples = dist.sample(10000)
        # A row is active if it has any nonzero value
        active = (samples.sum(dim=1) > 0).float().mean().item()
        assert abs(active - p) < 0.03, f"Expected ~{p}, got {active}"


class TestBumpReproducibility:
    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)

        dist1 = Bump(n_features=10, p_active=0.6, bump_width=2, generator=gen1)
        dist2 = Bump(n_features=10, p_active=0.6, bump_width=2, generator=gen2)

        samples1 = dist1.sample(200)
        samples2 = dist2.sample(200)
        assert torch.equal(samples1, samples2)


class TestBumpValidation:
    def test_bump_width_too_large_raises(self, seeded_generator):
        with pytest.raises(ValueError, match="bump_width"):
            Bump(n_features=7, p_active=0.5, bump_width=3, generator=seeded_generator)

    def test_bump_width_at_boundary_raises(self, seeded_generator):
        """bump_width must be strictly less than n_features // 2."""
        with pytest.raises(ValueError, match="bump_width"):
            Bump(n_features=8, p_active=0.5, bump_width=4, generator=seeded_generator)

    def test_bump_width_just_below_boundary_ok(self, seeded_generator):
        # n_features=8, n_features//2=4, bump_width=3 should be fine
        dist = Bump(
            n_features=8, p_active=0.5, bump_width=3, generator=seeded_generator
        )
        assert dist.bump_width == 3


class TestBumpUniformStateSelection:
    def test_states_are_roughly_uniform(self, seeded_generator):
        """Each state should be selected approximately uniformly."""
        k = 6
        dist = Bump(
            n_features=k, p_active=1.0, bump_width=0, generator=seeded_generator
        )
        samples = dist.sample(12000)
        # With bump_width=0, only the peak position is nonzero
        states = samples.argmax(dim=1)
        for j in range(k):
            count = (states == j).sum().item()
            expected = 12000 / k
            assert abs(count - expected) / expected < 0.1, (
                f"State {j}: expected ~{expected}, got {count}"
            )


class TestBumpWidthZero:
    def test_bump_width_zero_is_one_hot(self, seeded_generator):
        """With bump_width=0, active samples should be one-hot vectors."""
        dist = Bump(
            n_features=5, p_active=1.0, bump_width=0, generator=seeded_generator
        )
        samples = dist.sample(200)
        # Each row should have exactly one nonzero entry equal to 1.0
        assert (samples.sum(dim=1) == 1.0).all()
        assert ((samples == 0) | (samples == 1)).all()
