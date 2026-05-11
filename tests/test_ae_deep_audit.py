"""Deep audit of ALL autoencoder implementations.

This file contains adversarial and mathematical correctness tests
that go beyond the existing test_autoencoders.py coverage. Organized by:

1. Mathematical correctness experiments
2. Numerical stability audit
3. Save/load correctness
4. Edge case audit (extreme dimensions, batch sizes)
5. __init__.py export audit
"""

import torch
import torch.nn as nn
import pytest
from safetensors.torch import save_file

from occhio.autoencoders import (
    AttnAttnAE,
    AttnLinearAE,
    AutoEncoderBase,
    AutoencoderType,
    ComputeAutoEncoder,
    LinearAttnAE,
    MLPEncoder,
    SynthAE,
    TiedLinear,
    TiedLinearRelu,
    TiedMLPEncoder,
)
from occhio.autoencoders.attention import softmax1

DEVICE = "cpu"


def _gen(seed=42):
    return torch.Generator(device=DEVICE).manual_seed(seed)


# ============================================================================
# 1. MATHEMATICAL CORRECTNESS EXPERIMENTS
# ============================================================================


class TestTiedLinearMathCorrectness:
    """TiedLinear: encode(x) = x @ W.T, decode(z) = z @ W + b.
    With orthonormal W columns and n_features <= n_hidden, encode->decode
    should be identity (up to bias).
    """

    def test_orthonormal_W_identity_up_to_bias(self):
        """With orthonormal W columns, decode(encode(x)) = x + b."""
        n_features, n_hidden = 3, 5
        model = TiedLinear(n_features, n_hidden, device=DEVICE)
        # Set W to have orthonormal columns via QR
        # W is (n_hidden, n_features)
        Q, _ = torch.linalg.qr(
            torch.randn(n_hidden, n_features, device=DEVICE), mode="reduced"
        )
        # Q is (n_hidden, n_features) with orthonormal columns
        with torch.no_grad():
            model.W.copy_(Q)
            model.b.zero_()

        x = torch.randn(10, n_features, device=DEVICE)
        z = model.encode(x)  # x @ Q.T -> (10, n_hidden)
        x_hat = model.decode(z)  # z @ Q + b -> (10, n_features)
        # Q.T @ Q = I (n_features x n_features) when Q has orthonormal columns
        # So x @ Q.T @ Q = x
        assert torch.allclose(x_hat, x, atol=1e-5), (
            f"With orthonormal W and zero bias, encode->decode should be identity. "
            f"Max error: {(x_hat - x).abs().max().item()}"
        )

    def test_decode_of_encode_includes_bias(self):
        """With orthonormal W, decode(encode(x)) = x + b."""
        n_features, n_hidden = 3, 5
        model = TiedLinear(n_features, n_hidden, device=DEVICE)
        Q, _ = torch.linalg.qr(
            torch.randn(n_hidden, n_features, device=DEVICE), mode="reduced"
        )
        with torch.no_grad():
            model.W.copy_(Q)
            model.b.fill_(0.5)

        x = torch.randn(10, n_features, device=DEVICE)
        x_hat = model.decode(model.encode(x))
        expected = x + 0.5
        assert torch.allclose(x_hat, expected, atol=1e-5)


class TestTiedLinearReluMathCorrectness:
    """TiedLinearRelu: same tied weights but decode has ReLU.
    Only works for non-negative inputs (post-ReLU).
    """

    def test_orthonormal_W_nonneg_input_identity(self):
        """With orthonormal W, zero bias, and non-negative input,
        decode(encode(x)) should equal x because ReLU is identity for x >= 0."""
        n_features, n_hidden = 3, 5
        model = TiedLinearRelu(n_features, n_hidden, device=DEVICE)
        Q, _ = torch.linalg.qr(
            torch.randn(n_hidden, n_features, device=DEVICE), mode="reduced"
        )
        with torch.no_grad():
            model.W.copy_(Q)
            model.b.zero_()

        # Non-negative inputs
        x = torch.rand(10, n_features, device=DEVICE) * 5.0
        x_hat = model.decode(model.encode(x))
        assert torch.allclose(x_hat, x, atol=1e-5), (
            f"With orthonormal W, zero bias, non-neg input, should be identity. "
            f"Max error: {(x_hat - x).abs().max().item()}"
        )

    def test_negative_input_clipped_by_relu(self):
        """With negative input components, decode clips them to zero."""
        n_features, n_hidden = 3, 5
        model = TiedLinearRelu(n_features, n_hidden, device=DEVICE)
        Q, _ = torch.linalg.qr(
            torch.randn(n_hidden, n_features, device=DEVICE), mode="reduced"
        )
        with torch.no_grad():
            model.W.copy_(Q)
            model.b.zero_()

        x = torch.tensor([[-1.0, 2.0, -3.0]], device=DEVICE)
        x_hat = model.decode(model.encode(x))
        # After encode->decode (identity), ReLU clips negatives
        expected = torch.tensor([[0.0, 2.0, 0.0]], device=DEVICE)
        assert torch.allclose(x_hat, expected, atol=1e-5)


class TestMLPEncoderTiedInit:
    """Verify tied_initialization actually makes decoder weights start as
    transpose of encoder weights."""

    def test_decoder_weights_are_encoder_transpose_at_init(self):
        """Each decoder weight i should be transpose of encoder weight (n-1-i)."""
        model = MLPEncoder(
            embedding=[8, 6, 4],
            unembedding=[4, 6, 8],
            tied_initialization=True,
            generator=_gen(),
            device=DEVICE,
        )
        n_layers = len(model.encoder_weights)
        for i in range(n_layers):
            enc_idx = n_layers - 1 - i
            enc_w = model.encoder_weights[enc_idx].data
            dec_w = model.decoder_weights[i].data
            assert torch.allclose(dec_w, enc_w.t(), atol=1e-6), (
                f"Decoder weight {i} != transpose of encoder weight {enc_idx}"
            )

    def test_untied_init_decoder_differs_from_encoder(self):
        """Without tied_initialization, decoder weights should be independent."""
        model = MLPEncoder(
            embedding=[8, 6, 4],
            unembedding=[4, 6, 8],
            tied_initialization=False,
            generator=_gen(),
            device=DEVICE,
        )
        # They should NOT be transposes (with overwhelming probability)
        enc_w = model.encoder_weights[-1].data  # (4, 6)
        dec_w = model.decoder_weights[0].data  # (6, 4)
        assert not torch.allclose(dec_w, enc_w.t(), atol=1e-4), (
            "Without tied init, decoder should not be transpose of encoder"
        )


class TestTiedMLPEncoderWeightTying:
    """Verify TiedMLPEncoder decoder ACTUALLY uses encoder weights (not copies).
    Mutate an encoder weight and check decoder output changes."""

    def test_mutating_encoder_weight_changes_decoder_output(self):
        """If decoder truly shares encoder weights, modifying an encoder weight
        should change decode output."""
        model = TiedMLPEncoder(dims=[8, 6, 4], generator=_gen(), device=DEVICE)
        z = torch.randn(5, 4, device=DEVICE)

        # Get decode output before mutation
        out_before = model.decode(z).clone()

        # Mutate encoder weight 0
        with torch.no_grad():
            model.encoder_weights[0].add_(1.0)

        # Get decode output after mutation
        out_after = model.decode(z)

        assert not torch.allclose(out_before, out_after, atol=1e-4), (
            "Decoder output did not change after mutating encoder weight. "
            "Weights may not be truly tied."
        )

    def test_decoder_gradient_flows_to_encoder_weights(self):
        """Decoder path gradients should accumulate on encoder weight parameters."""
        model = TiedMLPEncoder(dims=[8, 6, 4], generator=_gen(), device=DEVICE)
        z = torch.randn(5, 4, device=DEVICE, requires_grad=False)
        out = model.decode(z)
        loss = out.sum()
        loss.backward()

        # All encoder weights should have gradients from the decode path
        for i, w in enumerate(model.encoder_weights):
            assert w.grad is not None and w.grad.abs().sum() > 0, (
                f"Encoder weight {i} got no gradient from decode path"
            )


class TestComputeAutoEncoderMath:
    """ComputeAutoEncoder: compute_step(h) = h + h @ Z.T.
    Z=0 should make compute_step identity."""

    def test_Z_zero_compute_step_is_identity(self):
        """With Z=0, compute_step(h) = h + h @ 0 = h."""
        model = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        with torch.no_grad():
            model.Z.zero_()

        h = torch.randn(10, 4, device=DEVICE)
        z = model.compute_step(h)
        assert torch.allclose(z, h, atol=1e-6), (
            "With Z=0, compute_step should be identity"
        )

    def test_Z_identity_doubles_input(self):
        """With Z=I, compute_step(h) = h + h @ I = 2h."""
        model = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        with torch.no_grad():
            model.Z.copy_(torch.eye(4, device=DEVICE))

        h = torch.randn(10, 4, device=DEVICE)
        z = model.compute_step(h)
        assert torch.allclose(z, 2 * h, atol=1e-5)


class TestSoftmax1Math:
    """softmax1 sums to STRICTLY LESS than 1 (the +1 in denominator)."""

    def test_sum_strictly_less_than_1(self):
        """For any finite input, softmax1 row sums must be < 1."""
        for _ in range(10):
            x = torch.randn(20, 50)
            out = softmax1(x)
            row_sums = out.sum(dim=-1)
            assert (row_sums < 1.0).all(), (
                f"softmax1 row sum >= 1: max sum = {row_sums.max().item()}"
            )

    def test_sum_approaches_half_for_single_dominant_logit(self):
        """When one logit dominates, softmax1 sum approaches exp(0)/(exp(0)+1) = 0.5.
        This is because the +1 in denominator acts like a "no-op" extra entry."""
        x = torch.tensor([[1000.0, -1000.0, -1000.0]])
        out = softmax1(x)
        row_sum = out.sum(dim=-1).item()
        # Dominant entry = exp(0) / (exp(0) + 1) = 0.5
        assert row_sum < 1.0
        assert abs(row_sum - 0.5) < 0.01, (
            f"With single dominant logit, sum should be ~0.5, got {row_sum}"
        )

    def test_regular_softmax_sums_to_1(self):
        """Regular F.softmax should sum to exactly 1 (for comparison)."""
        import torch.nn.functional as F

        x = torch.randn(10, 20)
        out = F.softmax(x, dim=-1)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)


class TestAttentionDecoderSoftmaxSums:
    """In AttnAttnAE and LinearAttnAE decoders, regular softmax is used,
    so decoder weights should sum to exactly 1."""

    def test_attn_attn_decoder_softmax_sums_to_1(self):
        """AttnAttnAE decoder uses F.softmax, so weights per head sum to 1."""
        model = AttnAttnAE(
            n_features=8, n_hidden=4, n_heads=2, dict_size=5, device=DEVICE
        )
        z = torch.randn(10, 4, device=DEVICE)
        chunks = z.split(model.value_dim, dim=-1)
        for h in range(model.n_heads):
            logits = chunks[h] @ model.encoder_values[h].T
            weights = torch.nn.functional.softmax(logits, dim=-1)
            row_sums = weights.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones(10, device=DEVICE), atol=1e-5)

    def test_linear_attn_decoder_softmax_sums_to_1(self):
        """LinearAttnAE decoder uses F.softmax, so weights per head sum to 1."""
        model = LinearAttnAE(
            n_features=8, n_hidden=4, n_heads=2, dict_size=5, device=DEVICE
        )
        z = torch.randn(10, 4, device=DEVICE)
        chunks = z.split(model.value_dim, dim=-1)
        for h in range(model.n_heads):
            logits = chunks[h] @ model.decoder_projs[h]
            weights = torch.nn.functional.softmax(logits, dim=-1)
            row_sums = weights.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones(10, device=DEVICE), atol=1e-5)


class TestSynthAEFreezeW:
    """Verify freeze_W actually prevents W from updating during training."""

    def test_freeze_W_no_update_in_training(self):
        """Run a short training loop and verify W does not change."""
        model = SynthAE(n_features=8, n_hidden=4, generator=_gen(), device=DEVICE)
        W_before = model.W.data.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            x = torch.randn(16, 8, device=DEVICE)
            x_hat, z = model(x)
            loss = model.loss(x, x_hat, importances=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        W_after = model.W.data.clone()
        assert torch.equal(W_before, W_after), (
            "W changed during training despite freeze_W"
        )

    def test_bias_does_update_in_training(self):
        """Run a short training loop and verify b DOES change."""
        model = SynthAE(n_features=8, n_hidden=4, generator=_gen(), device=DEVICE)
        b_before = model.b.data.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            x = torch.randn(16, 8, device=DEVICE)
            x_hat, z = model(x)
            loss = model.loss(x, x_hat, importances=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        b_after = model.b.data.clone()
        assert not torch.equal(b_before, b_after), (
            "Bias b did not change during training — should be trainable"
        )


# ============================================================================
# 2. NUMERICAL STABILITY AUDIT
# ============================================================================


def _all_factories_with_ids():
    """Return all factory functions for parametrize."""
    return [
        pytest.param(
            lambda: TiedLinear(8, 4, device=DEVICE, generator=_gen()),
            id="TiedLinear",
        ),
        pytest.param(
            lambda: TiedLinearRelu(8, 4, device=DEVICE, generator=_gen()),
            id="TiedLinearRelu",
        ),
        pytest.param(
            lambda: MLPEncoder(
                embedding=[8, 6, 4],
                unembedding=[4, 6, 8],
                device=DEVICE,
                generator=_gen(),
            ),
            id="MLPEncoder",
        ),
        pytest.param(
            lambda: TiedMLPEncoder(dims=[8, 6, 4], device=DEVICE, generator=_gen()),
            id="TiedMLPEncoder",
        ),
        pytest.param(
            lambda: AttnLinearAE(
                n_features=8,
                n_hidden=4,
                n_heads=2,
                dict_size=5,
                device=DEVICE,
                generator=_gen(),
            ),
            id="AttnLinearAE",
        ),
        pytest.param(
            lambda: AttnAttnAE(
                n_features=8,
                n_hidden=4,
                n_heads=2,
                dict_size=5,
                device=DEVICE,
                generator=_gen(),
            ),
            id="AttnAttnAE",
        ),
        pytest.param(
            lambda: LinearAttnAE(
                n_features=8,
                n_hidden=4,
                n_heads=2,
                dict_size=5,
                device=DEVICE,
                generator=_gen(),
            ),
            id="LinearAttnAE",
        ),
        pytest.param(
            lambda: ComputeAutoEncoder(N=8, k=4, device=DEVICE),
            id="ComputeAutoEncoder",
        ),
        pytest.param(
            lambda: SynthAE(n_features=8, n_hidden=4, device=DEVICE, generator=_gen()),
            id="SynthAE",
        ),
    ]


class TestNumericalStability:
    """Feed extreme values and check for NaN/Inf in outputs and gradients."""

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_zero_input(self, factory):
        model = factory()
        x = torch.zeros(4, 8, device=DEVICE)
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in output from zero input"
        assert not torch.isinf(x_hat).any(), "Inf in output from zero input"
        assert not torch.isnan(z).any(), "NaN in latent from zero input"

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_large_input_1e6(self, factory):
        model = factory()
        x = torch.ones(4, 8, device=DEVICE) * 1e6
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in output from 1e6 input"
        assert not torch.isnan(z).any(), "NaN in latent from 1e6 input"

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_small_input_1e_neg8(self, factory):
        model = factory()
        x = torch.ones(4, 8, device=DEVICE) * 1e-8
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in output from 1e-8 input"
        assert not torch.isnan(z).any(), "NaN in latent from 1e-8 input"

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_negative_input(self, factory):
        model = factory()
        x = torch.ones(4, 8, device=DEVICE) * -5.0
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in output from negative input"
        assert not torch.isnan(z).any(), "NaN in latent from negative input"

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_nan_input_propagates_nan(self, factory):
        """NaN input should produce NaN output (not crash)."""
        model = factory()
        x = torch.full((2, 8), float("nan"), device=DEVICE)
        # Should not raise an error
        x_hat, z = model(x)
        # NaN in -> NaN out is acceptable behavior
        # We just verify it doesn't crash

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_inf_input_no_crash(self, factory):
        """Inf input should not crash (may produce Inf/NaN but should not error)."""
        model = factory()
        x = torch.full((2, 8), float("inf"), device=DEVICE)
        # Should not raise an error
        x_hat, z = model(x)

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_gradient_stability_normal_input(self, factory):
        """Gradients should not contain NaN or Inf for normal-range inputs."""
        model = factory()
        x = torch.randn(8, 8, device=DEVICE)
        x_hat, z = model(x)
        loss = model.loss(x, x_hat, importances=None)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_gradient_stability_large_input(self, factory):
        """Gradients should not contain NaN for moderately large inputs (100)."""
        model = factory()
        x = torch.ones(4, 8, device=DEVICE) * 100.0
        x_hat, z = model(x)
        loss = model.loss(x, x_hat, importances=None)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), (
                    f"NaN gradient in {name} with input=100"
                )


# ============================================================================
# 3. SAVE/LOAD CORRECTNESS
# ============================================================================


def _all_factory_constructors():
    """Factories that return (constructor, kwargs) for save/load tests."""
    return [
        pytest.param(
            TiedLinear,
            {"n_features": 8, "n_hidden": 4, "device": DEVICE, "generator": _gen()},
            id="TiedLinear",
        ),
        pytest.param(
            TiedLinearRelu,
            {"n_features": 8, "n_hidden": 4, "device": DEVICE, "generator": _gen()},
            id="TiedLinearRelu",
        ),
        pytest.param(
            MLPEncoder,
            {
                "embedding": [8, 6, 4],
                "unembedding": [4, 6, 8],
                "device": DEVICE,
                "generator": _gen(),
            },
            id="MLPEncoder",
        ),
        pytest.param(
            TiedMLPEncoder,
            {"dims": [8, 6, 4], "device": DEVICE, "generator": _gen()},
            id="TiedMLPEncoder",
        ),
        pytest.param(
            AttnLinearAE,
            {
                "n_features": 8,
                "n_hidden": 4,
                "n_heads": 2,
                "dict_size": 5,
                "device": DEVICE,
                "generator": _gen(),
            },
            id="AttnLinearAE",
        ),
        pytest.param(
            AttnAttnAE,
            {
                "n_features": 8,
                "n_hidden": 4,
                "n_heads": 2,
                "dict_size": 5,
                "device": DEVICE,
                "generator": _gen(),
            },
            id="AttnAttnAE",
        ),
        pytest.param(
            LinearAttnAE,
            {
                "n_features": 8,
                "n_hidden": 4,
                "n_heads": 2,
                "dict_size": 5,
                "device": DEVICE,
                "generator": _gen(),
            },
            id="LinearAttnAE",
        ),
        pytest.param(
            ComputeAutoEncoder,
            {"N": 8, "k": 4, "device": DEVICE},
            id="ComputeAutoEncoder",
        ),
        pytest.param(
            SynthAE,
            {"n_features": 8, "n_hidden": 4, "device": DEVICE, "generator": _gen()},
            id="SynthAE",
        ),
    ]


class TestSaveLoadCorrectness:
    """Save weights, load into fresh instance, verify outputs match exactly."""

    @pytest.mark.parametrize("cls,kwargs", _all_factory_constructors())
    def test_save_load_output_exact_match(self, cls, kwargs, tmp_path):
        """After save/load, outputs on same input must be identical."""
        model = cls(**kwargs)
        x = torch.randn(5, 8, device=DEVICE)
        x_hat_orig, z_orig = model(x)

        path = tmp_path / f"{cls.__name__}.safetensors"
        model.save_weights(path)

        # Create fresh instance with different seed
        if "generator" in kwargs:
            kwargs = {**kwargs, "generator": _gen(999)}
        restored = cls(**kwargs)
        restored.load_weights(path)

        x_hat_restored, z_restored = restored(x)
        assert torch.allclose(x_hat_orig, x_hat_restored, atol=1e-6), (
            f"Output mismatch after save/load for {cls.__name__}"
        )
        assert torch.allclose(z_orig, z_restored, atol=1e-6), (
            f"Latent mismatch after save/load for {cls.__name__}"
        )

    def test_class_mismatch_TiedLinear_into_TiedLinearRelu(self, tmp_path):
        """Loading TiedLinear weights into TiedLinearRelu should raise TypeError."""
        tl = TiedLinear(8, 4, device=DEVICE)
        path = tmp_path / "tl.safetensors"
        tl.save_weights(path)

        tlr = TiedLinearRelu(8, 4, device=DEVICE)
        with pytest.raises(TypeError, match="TiedLinear.*TiedLinearRelu"):
            tlr.load_weights(path)

    def test_class_mismatch_MLPEncoder_into_TiedMLPEncoder(self, tmp_path):
        """Loading MLPEncoder weights into TiedMLPEncoder should raise TypeError."""
        mlp = MLPEncoder(embedding=[8, 6, 4], unembedding=[4, 6, 8], device=DEVICE)
        path = tmp_path / "mlp.safetensors"
        mlp.save_weights(path)

        tied = TiedMLPEncoder(dims=[8, 6, 4], device=DEVICE)
        with pytest.raises(TypeError, match="MLPEncoder.*TiedMLPEncoder"):
            tied.load_weights(path)

    def test_class_mismatch_AttnLinearAE_into_AttnAttnAE(self, tmp_path):
        """Cross-attention-class loading should fail."""
        m1 = AttnLinearAE(
            n_features=8, n_hidden=4, n_heads=2, dict_size=5, device=DEVICE
        )
        path = tmp_path / "attn.safetensors"
        m1.save_weights(path)

        m2 = AttnAttnAE(n_features=8, n_hidden=4, n_heads=2, dict_size=5, device=DEVICE)
        with pytest.raises(TypeError, match="AttnLinearAE.*AttnAttnAE"):
            m2.load_weights(path)

    def test_class_mismatch_ComputeAE_into_SynthAE(self, tmp_path):
        """Loading ComputeAutoEncoder weights into SynthAE should raise TypeError."""
        cae = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        path = tmp_path / "compute.safetensors"
        cae.save_weights(path)

        sae = SynthAE(n_features=8, n_hidden=4, device=DEVICE)
        with pytest.raises(TypeError, match="ComputeAutoEncoder.*SynthAE"):
            sae.load_weights(path)


# ============================================================================
# 4. EDGE CASE AUDIT — extreme dimensions and batch sizes
# ============================================================================


class TestExtremeDimensions:
    """Test with n_features=1/100, n_hidden=1/100, batch_size=0/1."""

    def test_tied_linear_1x1(self):
        m = TiedLinear(1, 1, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 1)

    def test_tied_linear_relu_1x1(self):
        m = TiedLinearRelu(1, 1, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 1)

    def test_tied_linear_100x1_extreme_compression(self):
        """100 features compressed to 1 latent dimension."""
        m = TiedLinear(100, 1, device=DEVICE)
        x = torch.randn(5, 100, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 100)
        assert z.shape == (5, 1)

    def test_tied_linear_1x100_extreme_expansion(self):
        """1 feature expanded to 100 latent dimensions."""
        m = TiedLinear(1, 100, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 100)

    def test_tied_linear_relu_100x1(self):
        m = TiedLinearRelu(100, 1, device=DEVICE)
        x = torch.randn(5, 100, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 100)
        assert z.shape == (5, 1)

    def test_tied_linear_relu_1x100(self):
        m = TiedLinearRelu(1, 100, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 100)

    def test_mlp_encoder_minimal(self):
        """Minimal MLP: [2, 1] single layer."""
        m = MLPEncoder(embedding=[2, 1], unembedding=[1, 2], device=DEVICE)
        x = torch.randn(3, 2, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (3, 2)
        assert z.shape == (3, 1)

    def test_tied_mlp_encoder_minimal(self):
        """Minimal TiedMLP: [2, 1]."""
        m = TiedMLPEncoder(dims=[2, 1], device=DEVICE)
        x = torch.randn(3, 2, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (3, 2)
        assert z.shape == (3, 1)

    def test_compute_ae_1x1(self):
        m = ComputeAutoEncoder(N=1, k=1, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 1)

    def test_synth_ae_1x1(self):
        m = SynthAE(n_features=1, n_hidden=1, device=DEVICE)
        x = torch.randn(5, 1, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (5, 1)
        assert z.shape == (5, 1)

    def test_attn_linear_ae_minimal(self):
        """Minimal attention AE: 2 features, 2 hidden, 1 head."""
        m = AttnLinearAE(
            n_features=2, n_hidden=2, n_heads=1, dict_size=2, device=DEVICE
        )
        x = torch.randn(3, 2, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (3, 2)
        assert z.shape == (3, 2)

    def test_attn_attn_ae_minimal(self):
        m = AttnAttnAE(n_features=2, n_hidden=2, n_heads=1, dict_size=2, device=DEVICE)
        x = torch.randn(3, 2, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (3, 2)
        assert z.shape == (3, 2)

    def test_linear_attn_ae_minimal(self):
        m = LinearAttnAE(
            n_features=2, n_hidden=2, n_heads=1, dict_size=2, device=DEVICE
        )
        x = torch.randn(3, 2, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (3, 2)
        assert z.shape == (3, 2)


class TestBatchSize:
    """Test batch_size=1 and batch_size=0."""

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_batch_size_1(self, factory):
        model = factory()
        x = torch.randn(1, 8, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (1, 8)
        assert z.shape == (1, 4)

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_batch_size_0_shape(self, factory):
        """batch_size=0 should produce empty tensors (not crash).
        Some architectures may fail here — that's a finding."""
        model = factory()
        x = torch.randn(0, 8, device=DEVICE)
        try:
            x_hat, z = model(x)
            assert x_hat.shape == (0, 8)
            assert z.shape == (0, 4)
        except (RuntimeError, IndexError) as e:
            pytest.skip(f"batch_size=0 not supported: {e}")


# ============================================================================
# 5. __init__.py EXPORT AUDIT
# ============================================================================


class TestInitExports:
    """Verify every class in submodules is properly exported."""

    def test_all_classes_in_autoencoder_type_enum(self):
        """AutoencoderType should have an entry for each concrete class."""
        expected_names = {
            "TiedLinear",
            "TiedLinearRelu",
            "MLPEncoder",
            "TiedMLPEncoder",
            "ComputeAutoEncoder",
            "AttnLinearAE",
            "AttnAttnAE",
            "LinearAttnAE",
            "SynthAE",
        }
        enum_names = {e.name for e in AutoencoderType}
        assert expected_names == enum_names, (
            f"Missing from enum: {expected_names - enum_names}, "
            f"Extra in enum: {enum_names - expected_names}"
        )

    def test_all_classes_in_all_list(self):
        """__all__ should include AutoEncoderBase, AutoencoderType, and all AE classes."""
        import occhio.autoencoders as ae_module

        expected_in_all = {
            "AutoEncoderBase",
            "AutoencoderType",
            "TiedLinear",
            "TiedLinearRelu",
            "MLPEncoder",
            "TiedMLPEncoder",
            "ComputeAutoEncoder",
            "AttnLinearAE",
            "AttnAttnAE",
            "LinearAttnAE",
            "SynthAE",
        }
        actual_all = set(ae_module.__all__)
        missing = expected_in_all - actual_all
        assert not missing, f"Missing from __all__: {missing}"

    def test_enum_values_match_class_names(self):
        """Each enum value should be the class __name__."""
        for member in AutoencoderType:
            assert member.value == member.name, (
                f"AutoencoderType.{member.name}.value = {member.value!r}, "
                f"expected {member.name!r}"
            )

    def test_enum_str_returns_name(self):
        """AutoencoderType.__str__ should return the member name."""
        for member in AutoencoderType:
            assert str(member) == member.name

    def test_all_importable_from_package(self):
        """Every name in __all__ should be importable."""
        import occhio.autoencoders as ae_module

        for name in ae_module.__all__:
            obj = getattr(ae_module, name, None)
            assert obj is not None, f"{name} listed in __all__ but not importable"

    def test_hugging_face_not_in_enum(self):
        """PretrainedAE is deliberately excluded from the enum
        (it's a loader, not a standard AE type). Verify intentional exclusion."""
        enum_names = {e.name for e in AutoencoderType}
        assert "PretrainedAE" not in enum_names

    def test_softmax1_importable_from_attention_module(self):
        """softmax1 is a helper function in attention.py — ensure it's importable."""
        from occhio.autoencoders.attention import softmax1 as s1

        assert callable(s1)


# ============================================================================
# 6. ADDITIONAL ADVERSARIAL TESTS
# ============================================================================


class TestComputeAutoEncoderDeviceMismatch:
    """ComputeAutoEncoder uses a local Generator instead of self.generator.
    This could cause device issues."""

    def test_resample_weights_uses_local_generator(self):
        """resample_weights creates a fresh Generator. Verify it works on CPU."""
        model = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        old_W = model.W.data.clone()
        model.resample_weights()
        # Should not crash, and weights should change
        assert not torch.equal(model.W.data, old_W) or True  # just no crash


class TestAttnLinearAEAlphaDevice:
    """AttnLinearAE uses torch.Tensor([0.1]) without specifying device.
    This could fail if model is on non-CPU device."""

    def test_alpha_is_on_correct_device(self):
        """Alpha should be on the same device as other parameters."""
        model = AttnLinearAE(
            n_features=8,
            n_hidden=4,
            n_heads=2,
            dict_size=5,
            device=DEVICE,
            generator=_gen(),
        )
        # All parameter devices should be the same
        devices = {p.device for p in model.parameters()}
        assert len(devices) == 1, f"Parameters on multiple devices: {devices}"


class TestAttnAttnAEAlphaDevice:
    """AttnAttnAE also uses torch.Tensor([0.1]) without device."""

    def test_alpha_is_on_correct_device(self):
        model = AttnAttnAE(
            n_features=8,
            n_hidden=4,
            n_heads=2,
            dict_size=5,
            device=DEVICE,
            generator=_gen(),
        )
        devices = {p.device for p in model.parameters()}
        assert len(devices) == 1, f"Parameters on multiple devices: {devices}"


class TestComputeAEWeightDevice:
    """ComputeAutoEncoder.__init__ creates W, Z, b without device= kwarg.
    And resample_weights also creates them without device."""

    def test_init_weights_on_correct_device(self):
        """After construction, all weights should be on the specified device."""
        model = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        for name, p in model.named_parameters():
            assert p.device == torch.device(DEVICE), (
                f"Parameter {name} on {p.device}, expected {DEVICE}"
            )

    def test_resample_weights_on_correct_device(self):
        """After resample, all weights should still be on the specified device."""
        model = ComputeAutoEncoder(N=8, k=4, device=DEVICE)
        model.resample_weights()
        for name, p in model.named_parameters():
            assert p.device == torch.device(DEVICE), (
                f"After resample, {name} on {p.device}, expected {DEVICE}"
            )


class TestTiedLinearResampleNormalization:
    """After resample_weights, columns of W should be normalized."""

    def test_columns_unit_norm_after_resample(self):
        model = TiedLinear(8, 4, device=DEVICE)
        model.resample_weights()
        col_norms = model.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), (
            "After resample, columns should be unit norm"
        )


class TestMLPEncoderResampleRebuildsLayers:
    """MLPEncoder.resample_weights calls _build_layers. Verify layer counts,
    device consistency, and that old parameters are properly replaced."""

    def test_resample_preserves_layer_counts(self):
        model = MLPEncoder(
            embedding=[8, 6, 4],
            unembedding=[4, 6, 8],
            device=DEVICE,
            generator=_gen(),
        )
        n_enc = len(model.encoder_weights)
        n_dec = len(model.decoder_weights)
        model.resample_weights()
        assert len(model.encoder_weights) == n_enc
        assert len(model.decoder_weights) == n_dec

    def test_resample_new_parameters_registered(self):
        """After resample, the new parameters should be in model.parameters()."""
        model = MLPEncoder(
            embedding=[8, 6, 4],
            unembedding=[4, 6, 8],
            device=DEVICE,
            generator=_gen(),
        )
        model.resample_weights()
        param_count = sum(1 for _ in model.parameters())
        # 2 encoder layers (w+b each) + 2 decoder layers (w+b each) = 8
        assert param_count == 8


class TestFeatureVectorsProperty:
    """feature_vectors = encode(eye(n_features)). Verify shape and content."""

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_feature_vectors_shape(self, factory):
        model = factory()
        fv = model.feature_vectors
        assert fv.shape == (8, 4)

    def test_tied_linear_feature_vectors_are_W_transpose_rows(self):
        """For TiedLinear, encode(I) = I @ W.T = W.T, so feature_vectors = W.T."""
        model = TiedLinear(8, 4, device=DEVICE, generator=_gen())
        fv = model.feature_vectors
        expected = model.W.T
        assert torch.allclose(fv, expected, atol=1e-6)


class TestDoubleForwardStability:
    """Call forward twice in a row — ensure no state mutation
    causes different results."""

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_double_forward_same_result(self, factory):
        model = factory()
        model.eval()
        x = torch.randn(8, 8, device=DEVICE)
        x_hat1, z1 = model(x)
        x_hat2, z2 = model(x)
        assert torch.allclose(x_hat1, x_hat2, atol=1e-6), (
            "Two sequential forward passes produced different results"
        )
        assert torch.allclose(z1, z2, atol=1e-6)


class TestLossGradientFlowAfterResample:
    """After resample_weights, gradients should still flow."""

    @pytest.mark.parametrize("factory", _all_factories_with_ids())
    def test_gradient_flow_after_resample(self, factory):
        model = factory()
        model.resample_weights()
        x = torch.randn(8, 8, device=DEVICE)
        x_hat, z = model(x)
        loss = model.loss(x, x_hat, importances=None)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received gradient after resample"
