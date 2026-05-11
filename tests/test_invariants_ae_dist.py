"""Adversarial invariant tests for autoencoders and distributions.

Tests mathematical properties, contracts, and invariants that MUST hold
regardless of implementation details. Written as a red-team exercise —
these test what the code SHOULD do, not what it currently DOES.
"""

import math

import pytest
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
#  Autoencoder imports
# ═══════════════════════════════════════════════════════════════════════════
from occhio.autoencoders import (
    TiedLinear,
    TiedLinearRelu,
    TiedMLPEncoder,
    MLPEncoder,
    SynthAE,
    ComputeAutoEncoder,
    AttnLinearAE,
    AttnAttnAE,
    LinearAttnAE,
)
from occhio.autoencoders.attention import softmax1

# ═══════════════════════════════════════════════════════════════════════════
#  Distribution imports
# ═══════════════════════════════════════════════════════════════════════════
from occhio.distributions import (
    SparseUniform,
    SparseExponential,
    SingleUniform,
    HierarchicalPairs,
    ScaledHierarchicalPairs,
    CorrelatedPairs,
    AnticorrelatedPairs,
    HierarchicalSparse,
    SimplexDistribution,
    SimplicialComplexDistribution,
    SphericalDistribution,
    ToricDistribution,
    HypercubeDistribution,
    DistributionStack,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42


def make_gen(seed=SEED):
    return torch.Generator().manual_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
#  AUTOENCODER INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestReconstructionQuality:
    """AE1: For non-compressive autoencoders, encode->decode should
    approximately reconstruct the input after training."""

    def _train_ae(self, ae, data, lr=0.01, steps=500):
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            x_hat, z = ae(data)
            loss = ae.loss(data, x_hat, None)
            loss.backward()
            optimizer.step()
        return ae

    def test_tied_linear_reconstructs_when_no_compression(self):
        """TiedLinear with n_hidden >= n_features should perfectly reconstruct."""
        n_features, n_hidden = 4, 8
        gen = make_gen()
        ae = TiedLinear(n_features, n_hidden, generator=gen)
        data = torch.randn(64, n_features)
        ae = self._train_ae(ae, data, steps=1000)
        with torch.no_grad():
            x_hat, _ = ae(data)
            mse = (data - x_hat).pow(2).mean().item()
        assert mse < 0.01, f"TiedLinear MSE {mse} too large for non-compressive case"

    def test_tied_linear_relu_reconstructs_nonneg_data(self):
        """TiedLinearRelu with n_hidden >= n_features should reconstruct non-negative data."""
        n_features, n_hidden = 4, 8
        gen = make_gen()
        ae = TiedLinearRelu(n_features, n_hidden, generator=gen)
        # Non-negative data (ReLU decoder produces non-negative output)
        data = torch.rand(64, n_features)
        ae = self._train_ae(ae, data, steps=1000)
        with torch.no_grad():
            x_hat, _ = ae(data)
            mse = (data - x_hat).pow(2).mean().item()
        assert mse < 0.01, f"TiedLinearRelu MSE {mse} too large"


class TestGradientSanity:
    """AE2: After loss.backward(), gradients should be finite and non-zero."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda: TiedLinear(4, 8, generator=make_gen()),
            lambda: TiedLinearRelu(4, 8, generator=make_gen()),
            lambda: TiedMLPEncoder([4, 8], generator=make_gen()),
            lambda: SynthAE(4, 3, generator=make_gen()),
            lambda: ComputeAutoEncoder(4, 3, seed=42),
        ],
        ids=[
            "TiedLinear",
            "TiedLinearRelu",
            "TiedMLPEncoder",
            "SynthAE",
            "ComputeAutoEncoder",
        ],
    )
    def test_gradients_finite_and_nonzero(self, ae_factory):
        ae = ae_factory()
        data = torch.rand(16, ae.n_features)
        x_hat, z = ae(data)
        loss = ae.loss(data, x_hat, None)
        loss.backward()

        for name, param in ae.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{name}: gradient contains NaN/Inf"
                )
                # At least some gradients should be nonzero after a random forward pass
                assert param.grad.abs().sum() > 0, f"{name}: gradient is all zeros"

    def test_attn_linear_ae_dead_gradient_at_init(self):
        """AttnLinearAE: encoder_projs and value_matrices have zero gradients
        at init because W_mix starts at zero, killing the attention branch.

        KNOWN ISSUE: This is by design (residual-style init), but means the
        attention-specific params are dead on step 1. After one optimizer step,
        W_mix becomes non-zero and gradients flow. We verify that:
        1. All gradients are finite
        2. At least W_mix, W_out, b, and alpha get nonzero gradients
        3. encoder_projs get zero gradients (documenting the dead path)
        """
        ae = AttnLinearAE(4, 4, 2, 3, generator=make_gen())
        data = torch.rand(16, ae.n_features)
        x_hat, z = ae(data)
        loss = ae.loss(data, x_hat, None)
        loss.backward()

        for name, param in ae.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{name}: gradient contains NaN/Inf"
                )

        # These params DO get gradients
        for name in ["W_mix", "W_out", "b", "alpha"]:
            param = dict(ae.named_parameters())[name]
            assert param.grad.abs().sum() > 0, f"{name} should have nonzero gradient"

        # KNOWN: attention params are dead at init due to W_mix=0
        for name, param in ae.named_parameters():
            if "encoder_projs" in name or "value_matrices" in name:
                assert param.grad.abs().sum() == 0, (
                    f"{name} should have zero gradient at init (W_mix=0)"
                )


class TestNumericalStability:
    """AE3: Very large, very small, all-zeros, all-ones — no NaN/Inf."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda: TiedLinear(4, 8, generator=make_gen()),
            lambda: TiedLinearRelu(4, 8, generator=make_gen()),
            lambda: TiedMLPEncoder([4, 8], generator=make_gen()),
            lambda: ComputeAutoEncoder(4, 3, seed=42),
        ],
        ids=["TiedLinear", "TiedLinearRelu", "TiedMLPEncoder", "ComputeAutoEncoder"],
    )
    @pytest.mark.parametrize(
        "input_factory,input_name",
        [
            (lambda n: torch.zeros(8, n), "all_zeros"),
            (lambda n: torch.ones(8, n), "all_ones"),
            (lambda n: torch.full((8, n), 1e-8), "very_small"),
            (lambda n: torch.full((8, n), 1e6), "very_large"),
        ],
        ids=["zeros", "ones", "tiny", "large"],
    )
    def test_no_nan_inf_in_outputs(self, ae_factory, input_factory, input_name):
        ae = ae_factory()
        ae.eval()
        x = input_factory(ae.n_features)
        with torch.no_grad():
            x_hat, z = ae(x)
        assert torch.isfinite(x_hat).all(), f"x_hat has NaN/Inf for {input_name}"
        assert torch.isfinite(z).all(), f"latent z has NaN/Inf for {input_name}"


class TestWeightNormInvariants:
    """AE4: TiedLinear/TiedLinearRelu columns of W should be unit-norm
    after resample_weights()."""

    @pytest.mark.parametrize(
        "ae_cls", [TiedLinear, TiedLinearRelu], ids=["TiedLinear", "TiedLinearRelu"]
    )
    def test_columns_unit_norm_after_resample(self, ae_cls):
        ae = ae_cls(5, 10, generator=make_gen())
        # W is (n_hidden, n_features) — columns are feature directions
        col_norms = ae.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), (
            f"Column norms after init: {col_norms}"
        )

    @pytest.mark.parametrize(
        "ae_cls", [TiedLinear, TiedLinearRelu], ids=["TiedLinear", "TiedLinearRelu"]
    )
    def test_columns_unit_norm_after_second_resample(self, ae_cls):
        ae = ae_cls(5, 10, generator=make_gen())
        # Perturb weights, then resample
        with torch.no_grad():
            ae.W.data *= 3.0
        ae.resample_weights()
        col_norms = ae.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), (
            f"Column norms after second resample: {col_norms}"
        )


class TestTiedWeightConsistency:
    """AE5: TiedMLPEncoder decoder uses encoder weights transposed."""

    def test_decoder_uses_encoder_weights(self):
        ae = TiedMLPEncoder([4, 8, 6], generator=make_gen())
        z = torch.randn(1, ae.n_hidden)

        with torch.no_grad():
            out_before = ae.decode(z.clone()).clone()

        # Modify the first encoder weight in place
        with torch.no_grad():
            ae.encoder_weights[0].data += 10.0

        with torch.no_grad():
            out_after = ae.decode(z.clone())

        # Decoder should produce different output because it uses encoder weights
        assert not torch.allclose(out_before, out_after, atol=1e-6), (
            "Modifying encoder weights did not change decoder output — "
            "weights may not be truly tied"
        )


class TestDeterminism:
    """AE6: Same generator seed -> identical weights and forward pass."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda g: TiedLinear(4, 8, generator=g),
            lambda g: TiedLinearRelu(4, 8, generator=g),
            lambda g: TiedMLPEncoder([4, 8], generator=g),
            lambda g: SynthAE(4, 3, generator=g),
        ],
        ids=["TiedLinear", "TiedLinearRelu", "TiedMLPEncoder", "SynthAE"],
    )
    def test_same_seed_same_weights(self, ae_factory):
        ae1 = ae_factory(make_gen(123))
        ae2 = ae_factory(make_gen(123))
        for (n1, p1), (n2, p2) in zip(ae1.named_parameters(), ae2.named_parameters()):
            assert torch.equal(p1, p2), f"Param {n1} differs between seeded runs"

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda g: TiedLinear(4, 8, generator=g),
            lambda g: TiedLinearRelu(4, 8, generator=g),
        ],
        ids=["TiedLinear", "TiedLinearRelu"],
    )
    def test_same_seed_same_forward(self, ae_factory):
        x = torch.rand(16, 4)
        ae1 = ae_factory(make_gen(123))
        ae2 = ae_factory(make_gen(123))
        with torch.no_grad():
            out1 = ae1(x)
            out2 = ae2(x)
        assert torch.equal(out1[0], out2[0]), "Forward pass outputs differ"
        assert torch.equal(out1[1], out2[1]), "Latent codes differ"


class TestSynthAEOrthogonalization:
    """AE7: With orthogonalize=True, pairwise cosine similarity should be
    LOW (close to 0). With orthogonalize=False, it should be HIGHER."""

    def test_orthogonalized_lower_rho_than_random(self):
        n_features, n_hidden = 20, 5
        gen_ortho = make_gen(42)
        gen_no_ortho = make_gen(42)

        ae_ortho = SynthAE(
            n_features,
            n_hidden,
            orthogonalize=True,
            ortho_steps=500,
            generator=gen_ortho,
        )
        ae_no_ortho = SynthAE(
            n_features, n_hidden, orthogonalize=False, generator=gen_no_ortho
        )

        rho_ortho = ae_ortho.rho_mm
        rho_no_ortho = ae_no_ortho.rho_mm

        assert rho_ortho < rho_no_ortho, (
            f"Orthogonalized rho_mm ({rho_ortho:.4f}) should be < "
            f"non-orthogonalized ({rho_no_ortho:.4f})"
        )

    def test_orthogonalized_columns_unit_norm(self):
        ae = SynthAE(10, 4, orthogonalize=True, ortho_steps=500, generator=make_gen())
        col_norms = ae.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4), (
            f"Column norms after ortho: {col_norms}"
        )


class TestComputeAutoEncoder:
    """AE8: compute_step is a residual h + h@Z.T."""

    def test_compute_step_is_residual(self):
        ae = ComputeAutoEncoder(6, 3, seed=42)
        h = torch.randn(4, 3)
        with torch.no_grad():
            result = ae.compute_step(h)
            expected = h + h @ ae.Z.T
        assert torch.allclose(result, expected, atol=1e-6), (
            "compute_step does not implement h + h@Z.T"
        )

    def test_compute_step_identity_when_Z_zero(self):
        """When Z=0, compute_step should be identity."""
        ae = ComputeAutoEncoder(6, 3, seed=42)
        with torch.no_grad():
            ae.Z.data.zero_()
        h = torch.randn(4, 3)
        with torch.no_grad():
            result = ae.compute_step(h)
        assert torch.allclose(result, h, atol=1e-6), (
            "compute_step with Z=0 should be identity"
        )


class TestSoftmax1:
    """AE9: softmax1 should sum to LESS than 1."""

    def test_softmax1_sums_to_less_than_one(self):
        x = torch.randn(32, 10)
        result = softmax1(x, dim=-1)
        row_sums = result.sum(dim=-1)
        assert (row_sums < 1.0).all(), (
            f"softmax1 row sums should be < 1, got max={row_sums.max():.6f}"
        )

    def test_softmax1_non_negative(self):
        x = torch.randn(32, 10)
        result = softmax1(x, dim=-1)
        assert (result >= 0).all(), "softmax1 produced negative values"

    def test_softmax1_bounded_above_by_one(self):
        x = torch.randn(32, 10)
        result = softmax1(x, dim=-1)
        assert (result <= 1.0).all(), "softmax1 produced values > 1"

    def test_softmax1_vs_regular_softmax_sum(self):
        """softmax sums to 1, softmax1 should always sum to less."""
        x = torch.randn(32, 10)
        regular = torch.softmax(x, dim=-1)
        s1 = softmax1(x, dim=-1)
        assert (s1.sum(dim=-1) < regular.sum(dim=-1)).all(), (
            "softmax1 should sum to less than regular softmax"
        )

    def test_softmax1_large_inputs_no_nan(self):
        """Very large inputs should not cause NaN (numerics)."""
        x = torch.full((4, 5), 1e6)
        result = softmax1(x, dim=-1)
        assert torch.isfinite(result).all(), (
            "softmax1 with large inputs produced NaN/Inf"
        )


class TestLossZeroForPerfectReconstruction:
    """AE10: If x_hat == x_true, loss should be exactly 0."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda: TiedLinear(4, 8, generator=make_gen()),
            lambda: TiedLinearRelu(4, 8, generator=make_gen()),
            lambda: TiedMLPEncoder([4, 8], generator=make_gen()),
        ],
        ids=["TiedLinear", "TiedLinearRelu", "TiedMLPEncoder"],
    )
    def test_loss_zero_when_x_hat_equals_x(self, ae_factory):
        ae = ae_factory()
        x = torch.rand(16, ae.n_features)
        loss = ae.loss(x, x, None)
        assert loss.item() == 0.0, (
            f"Loss should be 0 for perfect reconstruction, got {loss.item()}"
        )

    def test_loss_positive_when_imperfect(self):
        ae = TiedLinear(4, 8, generator=make_gen())
        x = torch.rand(16, 4)
        x_hat = x + 0.1
        loss = ae.loss(x, x_hat, None)
        assert loss.item() > 0, "Loss should be positive for imperfect reconstruction"


# ═══════════════════════════════════════════════════════════════════════════
#  DISTRIBUTION INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSparseUniformSparsityRate:
    """D1: SparseUniform with p_active=0.05 should produce ~5% nonzero entries."""

    def test_sparsity_rate_approximately_correct(self):
        n_features = 100
        p_active = 0.05
        dist = SparseUniform(n_features, p_active, generator=make_gen())
        samples = dist.sample(10000)
        observed_rate = (samples > 0).float().mean().item()
        # Should be within ~20% relative error for 10k samples
        assert 0.03 < observed_rate < 0.08, (
            f"Expected ~{p_active} active rate, got {observed_rate}"
        )

    def test_sparsity_not_inverted(self):
        """Common bug: p_active=0.05 should NOT give 95% nonzero."""
        n_features = 100
        p_active = 0.05
        dist = SparseUniform(n_features, p_active, generator=make_gen())
        samples = dist.sample(5000)
        observed_rate = (samples > 0).float().mean().item()
        assert observed_rate < 0.5, (
            f"Sparsity appears inverted: p_active={p_active} but "
            f"observed active rate = {observed_rate}"
        )

    def test_different_p_active_different_rates(self):
        """Higher p_active should give higher observed rate."""
        dist_low = SparseUniform(50, 0.1, generator=make_gen(1))
        dist_high = SparseUniform(50, 0.9, generator=make_gen(1))
        rate_low = (dist_low.sample(5000) > 0).float().mean().item()
        rate_high = (dist_high.sample(5000) > 0).float().mean().item()
        assert rate_low < rate_high, (
            f"p_active=0.1 rate ({rate_low}) >= p_active=0.9 rate ({rate_high})"
        )


class TestDistributionNonNegativity:
    """D2: All distributions should produce non-negative samples."""

    @pytest.mark.parametrize(
        "dist_factory",
        [
            lambda: SparseUniform(10, 0.5, generator=make_gen()),
            lambda: SparseExponential(10, 0.5, generator=make_gen()),
            lambda: SingleUniform(10, generator=make_gen()),
            lambda: HierarchicalPairs(10, 0.3, generator=make_gen()),
            lambda: ScaledHierarchicalPairs(10, 0.3, generator=make_gen()),
            lambda: AnticorrelatedPairs(10, 0.3, generator=make_gen()),
            lambda: HierarchicalSparse(10, generator=make_gen()),
            lambda: SimplexDistribution([3, 3, 4], 0.5, generator=make_gen()),
            lambda: SphericalDistribution(10, generator=make_gen()),
            lambda: ToricDistribution(10, generator=make_gen()),
            lambda: HypercubeDistribution(10, generator=make_gen()),
        ],
        ids=[
            "SparseUniform",
            "SparseExponential",
            "SingleUniform",
            "HierarchicalPairs",
            "ScaledHierarchicalPairs",
            "AnticorrelatedPairs",
            "HierarchicalSparse",
            "SimplexDistribution",
            "SphericalDistribution",
            "ToricDistribution",
            "HypercubeDistribution",
        ],
    )
    def test_all_samples_non_negative(self, dist_factory):
        dist = dist_factory()
        samples = dist.sample(1000)
        assert (samples >= 0).all(), (
            f"Negative values found! Min = {samples.min().item()}"
        )


class TestDistributionStackFeatureCount:
    """D3: DistributionStack total features = sum of component n_features."""

    def test_total_features_correct(self):
        d1 = SparseUniform(5, 0.3, device="cpu", generator=make_gen(1))
        d2 = SparseUniform(7, 0.3, device="cpu", generator=make_gen(2))
        d3 = SparseUniform(3, 0.3, device="cpu", generator=make_gen(3))
        stack = DistributionStack([d1, d2, d3], device="cpu")
        assert stack.n_features == 5 + 7 + 3, (
            f"Expected {5 + 7 + 3} features, got {stack.n_features}"
        )

    def test_sample_shape_matches_total_features(self):
        d1 = SparseUniform(5, 0.3, device="cpu", generator=make_gen(1))
        d2 = SparseUniform(7, 0.3, device="cpu", generator=make_gen(2))
        stack = DistributionStack([d1, d2], device="cpu")
        samples = stack.sample(32)
        assert samples.shape == (32, 12), f"Expected (32, 12), got {samples.shape}"


class TestHierarchicalPairsGating:
    """D4: HierarchicalPairs child features should ONLY fire when parent fires."""

    def test_child_never_fires_without_parent(self):
        n_features = 20
        dist = HierarchicalPairs(
            n_features, p_active=0.3, p_follow=0.8, generator=make_gen()
        )
        samples = dist.sample(10000)

        for pair_idx in range(n_features // 2):
            parent_col = samples[:, 2 * pair_idx]
            child_col = samples[:, 2 * pair_idx + 1]
            # Where parent is zero, child must ALWAYS be zero
            parent_inactive = parent_col == 0
            child_active_without_parent = child_col[parent_inactive] > 0
            assert not child_active_without_parent.any(), (
                f"Pair {pair_idx}: child fired {child_active_without_parent.sum()} times "
                f"without parent (out of {parent_inactive.sum()} inactive parent samples)"
            )

    def test_child_fires_less_than_parent(self):
        """Child rate should be <= parent rate (it's conditional)."""
        dist = HierarchicalPairs(20, p_active=0.5, p_follow=0.5, generator=make_gen())
        samples = dist.sample(10000)
        parent_rate = (samples[:, 0::2] > 0).float().mean().item()
        child_rate = (samples[:, 1::2] > 0).float().mean().item()
        assert child_rate <= parent_rate + 0.02, (
            f"Child rate ({child_rate}) > parent rate ({parent_rate})"
        )


class TestSphericalDistributionUnitNorm:
    """D5: SphericalDistribution feature positions should have unit norm."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_feature_positions_unit_norm(self, dim):
        dist = SphericalDistribution(20, manifold_dim=dim, generator=make_gen())
        norms = dist.feature_positions.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Feature positions not unit norm for dim={dim}: {norms}"
        )


class TestSimplexDistribution:
    """D6: Active simplex features should sum to ~1 (Dirichlet property)."""

    def test_active_simplex_sums_to_one(self):
        sizes = [3, 4, 5]
        dist = SimplexDistribution(sizes, p_active=1.0, generator=make_gen())
        samples = dist.sample(1000)

        offset = 0
        for k in sizes:
            block = samples[:, offset : offset + k]
            row_sums = block.sum(dim=-1)
            # Every simplex fires (p_active=1.0), so each block should sum to ~1
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
                f"Simplex of size {k}: sums should be 1.0, "
                f"got mean={row_sums.mean():.6f}, std={row_sums.std():.6f}"
            )
            offset += k

    def test_inactive_simplex_sums_to_zero(self):
        sizes = [3, 4]
        dist = SimplexDistribution(sizes, p_active=0.0, generator=make_gen())
        samples = dist.sample(100)
        assert (samples == 0).all(), "All simplices should be zero when p_active=0"


class TestDistributionReproducibility:
    """D7: Same seed -> same samples. Different seed -> different samples."""

    @pytest.mark.parametrize(
        "dist_factory",
        [
            lambda g: SparseUniform(10, 0.5, generator=g),
            lambda g: SparseExponential(10, 0.5, generator=g),
            lambda g: SingleUniform(10, generator=g),
            lambda g: HierarchicalPairs(10, 0.3, generator=g),
            lambda g: SimplexDistribution([3, 3, 4], 0.5, generator=g),
        ],
        ids=[
            "SparseUniform",
            "SparseExponential",
            "SingleUniform",
            "HierarchicalPairs",
            "SimplexDistribution",
        ],
    )
    def test_same_seed_same_samples(self, dist_factory):
        d1 = dist_factory(make_gen(99))
        d2 = dist_factory(make_gen(99))
        s1 = d1.sample(64)
        s2 = d2.sample(64)
        assert torch.equal(s1, s2), "Same seed should produce identical samples"

    @pytest.mark.parametrize(
        "dist_factory",
        [
            lambda g: SparseUniform(10, 0.5, generator=g),
            lambda g: SparseExponential(10, 0.5, generator=g),
            lambda g: SingleUniform(10, generator=g),
        ],
        ids=["SparseUniform", "SparseExponential", "SingleUniform"],
    )
    def test_different_seed_different_samples(self, dist_factory):
        d1 = dist_factory(make_gen(1))
        d2 = dist_factory(make_gen(2))
        s1 = d1.sample(64)
        s2 = d2.sample(64)
        assert not torch.equal(s1, s2), (
            "Different seeds should produce different samples"
        )


class TestDimensionConsistency:
    """D8: sample(batch_size).shape == (batch_size, n_features) for ALL distributions."""

    @pytest.mark.parametrize(
        "dist_factory,expected_n",
        [
            (lambda: SparseUniform(10, 0.5, generator=make_gen()), 10),
            (lambda: SparseExponential(10, 0.5, generator=make_gen()), 10),
            (lambda: SingleUniform(10, generator=make_gen()), 10),
            (lambda: HierarchicalPairs(10, 0.3, generator=make_gen()), 10),
            (lambda: ScaledHierarchicalPairs(10, 0.3, generator=make_gen()), 10),
            (
                lambda: CorrelatedPairs(
                    10, p_active=0.3, p_individual=0.5, generator=make_gen()
                ),
                10,
            ),
            (lambda: AnticorrelatedPairs(10, 0.3, generator=make_gen()), 10),
            (lambda: HierarchicalSparse(10, generator=make_gen()), 10),
            (lambda: SimplexDistribution([3, 3, 4], 0.5, generator=make_gen()), 10),
            (lambda: SphericalDistribution(10, generator=make_gen()), 10),
            (lambda: ToricDistribution(10, generator=make_gen()), 10),
            (lambda: HypercubeDistribution(10, generator=make_gen()), 10),
        ],
        ids=[
            "SparseUniform",
            "SparseExponential",
            "SingleUniform",
            "HierarchicalPairs",
            "ScaledHierarchicalPairs",
            "CorrelatedPairs",
            "AnticorrelatedPairs",
            "HierarchicalSparse",
            "SimplexDistribution",
            "SphericalDistribution",
            "ToricDistribution",
            "HypercubeDistribution",
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    def test_shape_correct(self, dist_factory, expected_n, batch_size):
        dist = dist_factory()
        samples = dist.sample(batch_size)
        assert samples.shape == (batch_size, expected_n), (
            f"Expected ({batch_size}, {expected_n}), got {samples.shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ADDITIONAL EDGE CASE AND ADVERSARIAL TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSparseExponentialNonNegativity:
    """SparseExponential values should always be non-negative."""

    def test_values_non_negative(self):
        dist = SparseExponential(20, 0.5, scale=2.0, generator=make_gen())
        samples = dist.sample(5000)
        assert (samples >= 0).all(), (
            f"Negative values found: min={samples.min().item()}"
        )


class TestSingleUniformExactlyOneActive:
    """SingleUniform should have exactly one active feature per sample."""

    def test_exactly_one_nonzero_per_row(self):
        dist = SingleUniform(20, generator=make_gen())
        samples = dist.sample(1000)
        active_count = (samples > 0).sum(dim=-1)
        assert (active_count == 1).all(), (
            f"Expected exactly 1 active per row, got counts: "
            f"min={active_count.min()}, max={active_count.max()}"
        )


class TestSimplicialComplexDistribution:
    """Simplicial complex: Dirichlet on chosen face, vertex sums."""

    def test_single_mode_active_features_sum_to_one(self):
        faces = [(0, 1, 2), (2, 3, 4)]
        dist = SimplicialComplexDistribution(
            5, faces, sampling_mode="single", generator=make_gen()
        )
        samples = dist.sample(1000)
        row_sums = samples.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Active vertices should sum to 1, got mean={row_sums.mean():.4f}"
        )


class TestAnticorrelatedPairsMutualExclusion:
    """AnticorrelatedPairs: at most one of each pair is active."""

    def test_at_most_one_per_pair(self):
        dist = AnticorrelatedPairs(20, p_active=0.8, generator=make_gen())
        samples = dist.sample(5000)
        for pair in range(10):
            both_active = (samples[:, 2 * pair] > 0) & (samples[:, 2 * pair + 1] > 0)
            assert not both_active.any(), (
                f"Pair {pair}: both features active in {both_active.sum()} samples"
            )


class TestSphericalDistributionNonNegativity:
    """SphericalDistribution may produce negative values from cosine bumps
    in certain configurations. Let's check."""

    def test_samples_non_negative(self):
        # With default length_scale=1.0, cosine bump should stay non-negative
        dist = SphericalDistribution(
            20, length_scale=1.0, manifold_dim=2, generator=make_gen()
        )
        samples = dist.sample(5000)
        assert (samples >= -1e-7).all(), (
            f"Negative values found: min={samples.min().item()}"
        )


class TestHierarchicalSparseRootAlwaysActive:
    """HierarchicalSparse: root (depth 0) should always be active."""

    def test_root_always_active(self):
        dist = HierarchicalSparse(20, generator=make_gen())
        samples = dist.sample(1000)
        root_indices = dist.depth_indices[0]
        for idx in root_indices:
            # Root should always have nonzero value (it fires with p=1,
            # value is Uniform(0,1) — vanishingly unlikely to be exactly 0)
            root_active = samples[:, idx] > 0
            assert root_active.float().mean() > 0.99, (
                f"Root node {idx} active rate = {root_active.float().mean()}"
            )


class TestHierarchicalSparseChildGating:
    """HierarchicalSparse: children should never fire when parent is inactive."""

    def test_child_gated_by_parent(self):
        dist = HierarchicalSparse(20, p_base=0.5, generator=make_gen())
        samples = dist.sample(5000)

        for node in dist.nodes:
            if node.parent is None:
                continue
            parent_inactive = samples[:, node.parent] == 0
            child_active = samples[:, node.index] > 0
            violations = (parent_inactive & child_active).sum().item()
            assert violations == 0, (
                f"Node {node.index} (parent={node.parent}): "
                f"{violations} violations of parent gating"
            )


class TestDistributionStackSamplingModes:
    """DistributionStack sampling modes should behave correctly."""

    def test_single_mode_activates_one_distribution(self):
        d1 = SparseUniform(5, 1.0, device="cpu", generator=make_gen(1))
        d2 = SparseUniform(5, 1.0, device="cpu", generator=make_gen(2))
        stack = DistributionStack(
            [d1, d2], sampling_mode="single", device="cpu", generator=make_gen(3)
        )
        samples = stack.sample(1000)
        # In single mode, only one sub-distribution fires per row.
        # So either first 5 or last 5 are nonzero, not both.
        first_active = (samples[:, :5] > 0).any(dim=-1)
        second_active = (samples[:, 5:] > 0).any(dim=-1)
        both_active = first_active & second_active
        assert not both_active.any(), (
            f"Single mode: both distributions active in {both_active.sum()} samples"
        )


class TestAutoEncoderForwardShapes:
    """Forward pass shapes should be consistent."""

    @pytest.mark.parametrize(
        "ae_factory,n_feat,n_hid",
        [
            (lambda: TiedLinear(4, 8, generator=make_gen()), 4, 8),
            (lambda: TiedLinearRelu(4, 8, generator=make_gen()), 4, 8),
            (lambda: TiedMLPEncoder([4, 8], generator=make_gen()), 4, 8),
            (lambda: ComputeAutoEncoder(4, 3, seed=42), 4, 3),
        ],
        ids=["TiedLinear", "TiedLinearRelu", "TiedMLPEncoder", "ComputeAutoEncoder"],
    )
    def test_output_shapes(self, ae_factory, n_feat, n_hid):
        ae = ae_factory()
        x = torch.rand(16, n_feat)
        x_hat, z = ae(x)
        assert x_hat.shape == (16, n_feat), f"x_hat shape: {x_hat.shape}"
        assert z.shape == (16, n_hid), f"z shape: {z.shape}"


class TestAutoEncoderBatchSizeOne:
    """Edge case: batch size = 1 should work."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda: TiedLinear(4, 8, generator=make_gen()),
            lambda: TiedLinearRelu(4, 8, generator=make_gen()),
            lambda: TiedMLPEncoder([4, 8], generator=make_gen()),
            lambda: ComputeAutoEncoder(4, 3, seed=42),
        ],
        ids=["TiedLinear", "TiedLinearRelu", "TiedMLPEncoder", "ComputeAutoEncoder"],
    )
    def test_batch_size_one(self, ae_factory):
        ae = ae_factory()
        x = torch.rand(1, ae.n_features)
        x_hat, z = ae(x)
        assert x_hat.shape[0] == 1
        assert z.shape[0] == 1
        assert torch.isfinite(x_hat).all()
        assert torch.isfinite(z).all()


class TestSynthAEFrozenW:
    """SynthAE: W should be frozen (not trainable) after init."""

    def test_w_is_frozen(self):
        ae = SynthAE(10, 4, generator=make_gen())
        assert not ae.W.requires_grad, "SynthAE.W should not require gradients"

    def test_b_is_trainable(self):
        ae = SynthAE(10, 4, generator=make_gen())
        assert ae.b.requires_grad, "SynthAE.b should require gradients"


class TestComputeAELoss:
    """ComputeAutoEncoder loss methods should be finite for valid inputs."""

    def test_mse_loss_finite(self):
        ae = ComputeAutoEncoder(6, 3, seed=42)
        x = torch.rand(16, 6)
        x_hat, z = ae(x)
        loss = ae.mse_loss(x_hat, x, None)
        assert torch.isfinite(loss), f"MSE loss is not finite: {loss}"

    def test_loss_method_finite(self):
        ae = ComputeAutoEncoder(6, 3, seed=42)
        x = torch.rand(16, 6)
        x_hat, z = ae(x)
        loss = ae.loss(x, x_hat, None)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"


class TestTiedLinearEncodeDecodeSymmetry:
    """TiedLinear: encode uses W.T, decode uses W. Verify this relationship."""

    def test_encode_decode_weight_relationship(self):
        ae = TiedLinear(4, 8, generator=make_gen())
        x = torch.randn(1, 4)
        z = ae.encode(x)
        # z = x @ W.T
        expected_z = x @ ae.W.T
        assert torch.allclose(z, expected_z, atol=1e-6)

        # decode: z @ W + b
        x_hat = ae.decode(z)
        expected_x_hat = z @ ae.W + ae.b
        assert torch.allclose(x_hat, expected_x_hat, atol=1e-6)


class TestCorrelatedPairsParameterValidation:
    """CorrelatedPairs should reject invalid parameter combinations."""

    def test_rejects_zero_params(self):
        with pytest.raises(ValueError, match="Exactly two"):
            CorrelatedPairs(10, generator=make_gen())

    def test_rejects_three_params(self):
        with pytest.raises(ValueError, match="Exactly two"):
            CorrelatedPairs(
                10,
                p_active=0.3,
                p_individual=0.5,
                correlation=0.2,
                generator=make_gen(),
            )

    def test_rejects_odd_features(self):
        with pytest.raises(AssertionError):
            CorrelatedPairs(11, p_active=0.3, p_individual=0.5, generator=make_gen())


class TestDistributionStackEmpty:
    """DistributionStack should reject empty list."""

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            DistributionStack([])


class TestSphericalManifoldDim:
    """SphericalDistribution feature_positions shape should match manifold_dim."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_feature_positions_shape(self, dim):
        n = 20
        dist = SphericalDistribution(n, manifold_dim=dim, generator=make_gen())
        assert dist.feature_positions.shape == (n, dim + 1), (
            f"Expected ({n}, {dim + 1}), got {dist.feature_positions.shape}"
        )


class TestScaledHierarchicalPairsValueScaling:
    """ScaledHierarchicalPairs: child value should be <= parent value."""

    def test_child_value_bounded_by_parent(self):
        dist = ScaledHierarchicalPairs(
            20, p_active=0.5, p_follow=0.9, generator=make_gen()
        )
        samples = dist.sample(5000)
        for pair in range(10):
            parent_val = samples[:, 2 * pair]
            child_val = samples[:, 2 * pair + 1]
            # Where both are active, child <= parent (since child = U * parent)
            both_active = (parent_val > 0) & (child_val > 0)
            if both_active.any():
                violations = child_val[both_active] > parent_val[both_active] + 1e-6
                assert not violations.any(), (
                    f"Pair {pair}: child > parent in {violations.sum()} cases"
                )
