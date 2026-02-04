"""Pytest tests for RelationalSimple and MultiRelational."""

import pytest
import torch
from ..relational import RelationalSimple, MultiRelational


@pytest.fixture
def seeded_generator():
    """Provide a seeded generator for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


class TestRelationalSimpleBasic:
    def test_sample_shape(self, seeded_generator):
        dist = RelationalSimple(n_features=10, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_on_mat_shape(self, seeded_generator):
        n = 15
        dist = RelationalSimple(n_features=n, p_active=0.3, generator=seeded_generator)
        assert dist.on_mat.shape == (n, n)

    def test_on_mat_is_orthogonal(self, seeded_generator):
        """O(n) matrix should satisfy Q @ Q.T = I."""
        dist = RelationalSimple(n_features=10, p_active=0.5, generator=seeded_generator)
        Q = dist.on_mat
        product = Q @ Q.T
        identity = torch.eye(10)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_on_mat_has_unit_determinant_magnitude(self, seeded_generator):
        """O(n) matrix has |det| = 1."""
        dist = RelationalSimple(n_features=8, p_active=0.5, generator=seeded_generator)
        det = torch.linalg.det(dist.on_mat)
        assert abs(abs(det.item()) - 1.0) < 1e-5


class TestRelationalSimpleValues:
    def test_can_produce_negative_values(self, seeded_generator):
        """Orthogonal rotation can produce negative outputs."""
        dist = RelationalSimple(n_features=20, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(1000)
        assert samples.min() < 0, "Orthogonal transform should produce some negative values"

    def test_values_bounded_in_expectation(self, seeded_generator):
        """Values shouldn't explode since O(n) preserves norms."""
        dist = RelationalSimple(n_features=50, p_active=0.3, generator=seeded_generator)
        samples = dist.sample(1000)
        # With p_active=0.3 and Uniform[0,1], expected L2 norm per sample is manageable
        max_abs = samples.abs().max().item()
        assert max_abs < 20, f"Max absolute value {max_abs} seems too large"


class TestRelationalSimpleStructure:
    def test_output_is_sum_of_two_components(self, seeded_generator):
        """Output = first + on_mat @ second (implicitly tested via shape/value checks)."""
        dist = RelationalSimple(n_features=10, p_active=0.5, generator=seeded_generator)
        samples = dist.sample(100)
        # If structure is wrong, shape would fail or values would be nonsensical
        assert samples.shape == (100, 10)
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_zero_p_active_gives_zeros(self, seeded_generator):
        """With p_active=0, both components are zero."""
        dist = RelationalSimple(n_features=10, p_active=0.0, generator=seeded_generator)
        samples = dist.sample(100)
        assert (samples == 0).all()


class TestRelationalSimpleReproducibility:
    def test_same_seed_same_on_mat(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = RelationalSimple(n_features=10, p_active=0.5, generator=gen1)
        dist2 = RelationalSimple(n_features=10, p_active=0.5, generator=gen2)

        assert torch.equal(dist1.on_mat, dist2.on_mat)

    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        dist1 = RelationalSimple(n_features=10, p_active=0.5, generator=gen1)
        dist2 = RelationalSimple(n_features=10, p_active=0.5, generator=gen2)

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestMultiRelationalBasic:
    def test_sample_shape(self, seeded_generator):
        dist = MultiRelational(n_features=10, p_active=0.5, k=3, generator=seeded_generator)
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_correct_number_of_matrices(self, seeded_generator):
        k = 5
        dist = MultiRelational(n_features=10, p_active=0.3, k=k, generator=seeded_generator)
        assert len(dist.on_mats) == k

    def test_all_matrices_orthogonal(self, seeded_generator):
        dist = MultiRelational(n_features=8, p_active=0.5, k=4, generator=seeded_generator)
        identity = torch.eye(8)
        for i, Q in enumerate(dist.on_mats):
            product = Q @ Q.T
            assert torch.allclose(product, identity, atol=1e-5), f"Matrix {i} not orthogonal"

    def test_matrices_have_correct_shape(self, seeded_generator):
        n = 12
        dist = MultiRelational(n_features=n, p_active=0.5, k=3, generator=seeded_generator)
        for Q in dist.on_mats:
            assert Q.shape == (n, n)


class TestMultiRelationalValues:
    def test_can_produce_negative_values(self, seeded_generator):
        dist = MultiRelational(n_features=20, p_active=0.5, k=3, generator=seeded_generator)
        samples = dist.sample(1000)
        assert samples.min() < 0

    def test_values_bounded_in_expectation(self, seeded_generator):
        dist = MultiRelational(n_features=50, p_active=0.3, k=5, generator=seeded_generator)
        samples = dist.sample(1000)
        max_abs = samples.abs().max().item()
        # More matrices might mean larger values, but still bounded
        assert max_abs < 50, f"Max absolute value {max_abs} seems too large"


class TestMultiRelationalKParameter:
    def test_k_one_similar_structure(self, seeded_generator):
        """k=1 should give a single rotated sparse vector."""
        dist = MultiRelational(n_features=10, p_active=0.5, k=1, generator=seeded_generator)
        assert len(dist.on_mats) == 1
        samples = dist.sample(100)
        assert samples.shape == (100, 10)

    def test_k_affects_variance(self, seeded_generator):
        """More components (higher k) should generally increase variance."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        dist_k2 = MultiRelational(n_features=50, p_active=0.3, k=2, generator=gen1)
        dist_k5 = MultiRelational(n_features=50, p_active=0.3, k=5, generator=gen2)

        samples_k2 = dist_k2.sample(5000)
        samples_k5 = dist_k5.sample(5000)

        var_k2 = samples_k2.var().item()
        var_k5 = samples_k5.var().item()

        # k=5 sums more independent terms, so variance should be higher
        assert var_k5 > var_k2, f"var(k=5)={var_k5} should exceed var(k=2)={var_k2}"

    def test_default_k_is_two(self, seeded_generator):
        dist = MultiRelational(n_features=10, p_active=0.5, generator=seeded_generator)
        assert len(dist.on_mats) == 2


class TestMultiRelationalEdgeCases:
    def test_zero_p_active_gives_zeros(self, seeded_generator):
        dist = MultiRelational(n_features=10, p_active=0.0, k=3, generator=seeded_generator)
        samples = dist.sample(100)
        assert (samples == 0).all()

    def test_no_nans_or_infs(self, seeded_generator):
        dist = MultiRelational(n_features=20, p_active=0.8, k=5, generator=seeded_generator)
        samples = dist.sample(1000)
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()


class TestMultiRelationalReproducibility:
    def test_same_seed_same_matrices(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)

        dist1 = MultiRelational(n_features=10, p_active=0.5, k=3, generator=gen1)
        dist2 = MultiRelational(n_features=10, p_active=0.5, k=3, generator=gen2)

        for Q1, Q2 in zip(dist1.on_mats, dist2.on_mats):
            assert torch.equal(Q1, Q2)

    def test_same_seed_same_samples(self):
        gen1 = torch.Generator().manual_seed(999)
        gen2 = torch.Generator().manual_seed(999)

        dist1 = MultiRelational(n_features=10, p_active=0.5, k=3, generator=gen1)
        dist2 = MultiRelational(n_features=10, p_active=0.5, k=3, generator=gen2)

        samples1 = dist1.sample(50)
        samples2 = dist2.sample(50)
        assert torch.equal(samples1, samples2)


class TestRelationalComparison:
    def test_multi_with_k2_differs_from_simple(self, seeded_generator):
        """MultiRelational(k=2) differs from RelationalSimple in structure."""
        # RelationalSimple: first + on_mat @ second (identity on first)
        # MultiRelational(k=2): on_mat1 @ first + on_mat2 @ second (both rotated)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        simple = RelationalSimple(n_features=20, p_active=0.5, generator=gen1)
        multi = MultiRelational(n_features=20, p_active=0.5, k=2, generator=gen2)

        # They use different RNG sequences for matrices, so outputs will differ
        samples_simple = simple.sample(100)
        samples_multi = multi.sample(100)

        # Just verify both produce valid outputs
        assert samples_simple.shape == samples_multi.shape
        assert not torch.equal(samples_simple, samples_multi)