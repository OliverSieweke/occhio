"""Comprehensive tests for all autoencoder architectures in occhio.

Testing strategy:
- For each concrete autoencoder class: construction, encode/decode/forward shapes,
  resample_weights mutation, loss computation, gradient flow, device consistency,
  and feature_vectors property.
- For AutoEncoderBase: save/load round-trip, class mismatch validation,
  missing file errors, serialization helpers, __init_subclass__ validation.
- For SynthAE: orthogonalization effectiveness, freeze_W behavior, rho_mm metric.
- For ComputeAutoEncoder: compute_step, ce_loss/mse_loss, decode activations.
- For attention classes: softmax1 helper, n_hidden divisibility validation.
- For MLP classes: tied initialization, various layer configurations.
"""

import math

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from occhio.autoencoders import (
    AttnAttnAE,
    AttnLinearAE,
    AutoEncoderBase,
    ComputeAutoEncoder,
    LinearAttnAE,
    MLPEncoder,
    SynthAE,
    TiedLinear,
    TiedLinearRelu,
    TiedMLPEncoder,
)
from occhio.autoencoders.attention import softmax1

# ── constants ──────────────────────────────────────────────────────────────────

DEVICE = "cpu"
N_FEATURES = 8
N_HIDDEN = 4
BATCH = 16
SEED = 42


def _gen(seed=SEED):
    return torch.Generator(device=DEVICE).manual_seed(seed)


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def random_input():
    """Standard (BATCH, N_FEATURES) random input for shape tests."""
    return torch.randn(BATCH, N_FEATURES, device=DEVICE, generator=_gen(0))


@pytest.fixture
def random_latent():
    """Standard (BATCH, N_HIDDEN) random input for decode shape tests."""
    return torch.randn(BATCH, N_HIDDEN, device=DEVICE, generator=_gen(1))


# ── factory helpers ────────────────────────────────────────────────────────────


def make_tied_linear(seed=SEED):
    return TiedLinear(N_FEATURES, N_HIDDEN, generator=_gen(seed), device=DEVICE)


def make_tied_linear_relu(seed=SEED):
    return TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=_gen(seed), device=DEVICE)


def make_mlp_encoder(seed=SEED):
    return MLPEncoder(
        embedding=[N_FEATURES, 6, N_HIDDEN],
        unembedding=[N_HIDDEN, 6, N_FEATURES],
        generator=_gen(seed),
        device=DEVICE,
    )


def make_tied_mlp_encoder(seed=SEED):
    return TiedMLPEncoder(
        dims=[N_FEATURES, 6, N_HIDDEN],
        generator=_gen(seed),
        device=DEVICE,
    )


def make_attn_linear_ae(seed=SEED):
    return AttnLinearAE(
        n_features=N_FEATURES,
        n_hidden=N_HIDDEN,
        n_heads=2,
        dict_size=5,
        generator=_gen(seed),
        device=DEVICE,
    )


def make_attn_attn_ae(seed=SEED):
    return AttnAttnAE(
        n_features=N_FEATURES,
        n_hidden=N_HIDDEN,
        n_heads=2,
        dict_size=5,
        generator=_gen(seed),
        device=DEVICE,
    )


def make_linear_attn_ae(seed=SEED):
    return LinearAttnAE(
        n_features=N_FEATURES,
        n_hidden=N_HIDDEN,
        n_heads=2,
        dict_size=5,
        generator=_gen(seed),
        device=DEVICE,
    )


def make_compute_ae(seed=SEED):
    return ComputeAutoEncoder(
        N=N_FEATURES,
        k=N_HIDDEN,
        device=DEVICE,
        seed=seed,
    )


def make_synth_ae(seed=SEED):
    return SynthAE(
        n_features=N_FEATURES,
        n_hidden=N_HIDDEN,
        generator=_gen(seed),
        device=DEVICE,
    )


ALL_FACTORIES = [
    pytest.param(make_tied_linear, id="TiedLinear"),
    pytest.param(make_tied_linear_relu, id="TiedLinearRelu"),
    pytest.param(make_mlp_encoder, id="MLPEncoder"),
    pytest.param(make_tied_mlp_encoder, id="TiedMLPEncoder"),
    pytest.param(make_attn_linear_ae, id="AttnLinearAE"),
    pytest.param(make_attn_attn_ae, id="AttnAttnAE"),
    pytest.param(make_linear_attn_ae, id="LinearAttnAE"),
    pytest.param(make_compute_ae, id="ComputeAutoEncoder"),
    pytest.param(make_synth_ae, id="SynthAE"),
]


# ============================================================================
# 1. UNIVERSAL TESTS — run for every autoencoder class
# ============================================================================


class TestConstruction:
    """Verify all models construct without errors and store correct dimensions."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_n_features_stored(self, factory):
        """n_features must be accessible after construction."""
        model = factory()
        assert model.n_features == N_FEATURES

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_n_hidden_stored(self, factory):
        """n_hidden must be accessible after construction."""
        model = factory()
        assert model.n_hidden == N_HIDDEN

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_device_property(self, factory):
        """device property should report the construction device."""
        model = factory()
        assert model.device == torch.device(DEVICE)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_is_nn_module(self, factory):
        """All autoencoders must be nn.Module subclasses."""
        model = factory()
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_is_autoencoder_base(self, factory):
        """All autoencoders must descend from AutoEncoderBase."""
        model = factory()
        assert isinstance(model, AutoEncoderBase)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_has_trainable_parameters(self, factory):
        """Every autoencoder must have at least one parameter (possibly frozen)."""
        model = factory()
        assert sum(1 for _ in model.parameters()) > 0


class TestEncodeShape:
    """encode() must produce (batch, n_hidden) from (batch, n_features)."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_encode_output_shape(self, factory, random_input):
        """Catches shape mismatches in the encoder projection."""
        model = factory()
        z = model.encode(random_input)
        assert z.shape == (BATCH, N_HIDDEN)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_encode_single_sample(self, factory):
        """Ensures batch dim=1 works (no accidental squeeze)."""
        model = factory()
        x = torch.randn(1, N_FEATURES, device=DEVICE)
        z = model.encode(x)
        assert z.shape == (1, N_HIDDEN)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_encode_preserves_dtype(self, factory, random_input):
        """Output dtype should match input dtype (float32)."""
        model = factory()
        z = model.encode(random_input)
        assert z.dtype == random_input.dtype


class TestDecodeShape:
    """decode() must produce (batch, n_features) from (batch, n_hidden)."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_decode_output_shape(self, factory, random_latent):
        """Catches shape mismatches in the decoder projection."""
        model = factory()
        x_hat = model.decode(random_latent)
        assert x_hat.shape == (BATCH, N_FEATURES)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_decode_single_sample(self, factory):
        """Ensures batch dim=1 works (no accidental squeeze)."""
        model = factory()
        z = torch.randn(1, N_HIDDEN, device=DEVICE)
        x_hat = model.decode(z)
        assert x_hat.shape == (1, N_FEATURES)


class TestForward:
    """forward() must return (x_hat, z) with correct shapes."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_forward_returns_tuple(self, factory, random_input):
        """forward() contract: returns a 2-tuple."""
        model = factory()
        result = model(random_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_forward_x_hat_shape(self, factory, random_input):
        """x_hat must match input shape for reconstruction."""
        model = factory()
        x_hat, _ = model(random_input)
        assert x_hat.shape == random_input.shape

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_forward_z_shape(self, factory, random_input):
        """Latent z must be (batch, n_hidden)."""
        model = factory()
        _, z = model(random_input)
        assert z.shape == (BATCH, N_HIDDEN)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_forward_matches_encode_decode(self, factory, random_input):
        """forward() should be consistent with separate encode + decode calls.

        Note: ComputeAutoEncoder overrides forward with a compute_step,
        so this won't hold for it — we skip it.
        """
        model = factory()
        model.eval()
        x_hat_fwd, z_fwd = model(random_input)
        z_manual = model.encode(random_input)
        # For ComputeAutoEncoder, forward includes compute_step so z differs
        if not isinstance(model, ComputeAutoEncoder):
            assert torch.allclose(z_fwd, z_manual, atol=1e-6), (
                "encode() and forward() latent disagree"
            )
            x_hat_manual = model.decode(z_manual)
            assert torch.allclose(x_hat_fwd, x_hat_manual, atol=1e-6), (
                "decode(encode(x)) and forward(x) disagree"
            )


class TestResampleWeights:
    """resample_weights() must mutate parameters."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_resample_changes_parameters(self, factory):
        """After resampling, at least one parameter should differ.

        Uses a different generator seed to guarantee different random values.
        """
        model = factory()
        old_params = {k: v.clone() for k, v in model.state_dict().items()}

        # Change generator to ensure different random values
        model.generator = _gen(seed=99)
        model.resample_weights()

        new_params = model.state_dict()
        any_changed = any(
            not torch.equal(old_params[k], new_params[k]) for k in old_params
        )
        assert any_changed, "resample_weights() did not change any parameters"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_resample_preserves_shapes(self, factory):
        """Parameter shapes must not change after resampling."""
        model = factory()
        old_shapes = {k: v.shape for k, v in model.state_dict().items()}
        model.resample_weights()
        new_shapes = {k: v.shape for k, v in model.state_dict().items()}
        assert old_shapes == new_shapes


class TestLoss:
    """loss() must return a non-negative scalar."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_loss_returns_scalar(self, factory, random_input):
        """Loss must be a 0-dim tensor (scalar)."""
        model = factory()
        x_hat, _ = model(random_input)
        loss = model.loss(random_input, x_hat, importances=None)
        assert loss.dim() == 0

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_loss_non_negative(self, factory, random_input):
        """MSE-based loss must be >= 0."""
        model = factory()
        x_hat, _ = model(random_input)
        loss = model.loss(random_input, x_hat, importances=None)
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_loss_with_importances(self, factory, random_input):
        """Loss with explicit importances should still return a scalar."""
        model = factory()
        x_hat, _ = model(random_input)
        importances = torch.ones(N_FEATURES, device=DEVICE) * 2.0
        loss = model.loss(random_input, x_hat, importances)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_loss_zero_for_perfect_reconstruction(self, factory):
        """When x_hat == x_true, loss should be exactly 0."""
        model = factory()
        x = torch.randn(4, N_FEATURES, device=DEVICE)
        loss = model.loss(x, x, importances=None)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestFeatureVectors:
    """feature_vectors property must have shape (n_features, n_hidden)."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_feature_vectors_shape(self, factory):
        """The identity-derived feature vectors must be (n_features, n_hidden).

        feature_vectors = encode(eye(n_features)) which is (n_features, n_hidden).
        """
        model = factory()
        fv = model.feature_vectors
        assert fv.shape == (N_FEATURES, N_HIDDEN)


class TestGradientFlow:
    """loss.backward() must populate gradients without errors."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_backward_completes(self, factory, random_input):
        """Verifies the computation graph is connected and backward works."""
        model = factory()
        x_hat, z = model(random_input)
        loss = model.loss(random_input, x_hat, importances=None)
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received a gradient"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_no_nan_gradients(self, factory, random_input):
        """NaN gradients indicate numerical instability."""
        model = factory()
        x_hat, z = model(random_input)
        loss = model.loss(random_input, x_hat, importances=None)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


class TestDeviceConsistency:
    """All parameters must live on the same device."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_all_params_on_same_device(self, factory):
        """Catches accidental device mismatches between parameters."""
        model = factory()
        devices = {p.device for p in model.parameters()}
        assert len(devices) == 1, f"Parameters on multiple devices: {devices}"


class TestDifferentDimensions:
    """Verify models work with various feature/hidden dimension combos."""

    @pytest.mark.parametrize(
        "n_feat,n_hid",
        [(1, 1), (3, 10), (16, 4), (32, 32)],
        ids=["1x1", "3x10", "16x4", "32x32"],
    )
    def test_tied_linear_various_dims(self, n_feat, n_hid):
        """TiedLinear must handle degenerate and overcomplete settings."""
        m = TiedLinear(n_feat, n_hid, device=DEVICE)
        x = torch.randn(2, n_feat, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (2, n_feat)
        assert z.shape == (2, n_hid)

    @pytest.mark.parametrize(
        "n_feat,n_hid",
        [(1, 1), (3, 10), (16, 4), (32, 32)],
        ids=["1x1", "3x10", "16x4", "32x32"],
    )
    def test_tied_linear_relu_various_dims(self, n_feat, n_hid):
        """TiedLinearRelu must handle degenerate and overcomplete settings."""
        m = TiedLinearRelu(n_feat, n_hid, device=DEVICE)
        x = torch.randn(2, n_feat, device=DEVICE)
        x_hat, z = m(x)
        assert x_hat.shape == (2, n_feat)
        assert z.shape == (2, n_hid)


# ============================================================================
# 2. AutoEncoderBase SPECIFIC TESTS
# ============================================================================


class TestAutoEncoderBaseSaveLoad:
    """Save/load round-trip, error handling, and metadata validation."""

    def test_save_load_round_trip(self, tmp_path):
        """Weights must be identical after save + load."""
        model = make_tied_linear_relu()
        orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
        path = tmp_path / "model.safetensors"
        model.save_weights(path)

        restored = make_tied_linear_relu(seed=99)
        restored.load_weights(path)
        for k in orig_sd:
            assert torch.equal(orig_sd[k], restored.state_dict()[k])

    def test_load_class_mismatch_raises_type_error(self, tmp_path):
        """Loading TiedLinear weights into TiedLinearRelu must fail with TypeError."""
        tl = make_tied_linear()
        path = tmp_path / "tl.safetensors"
        tl.save_weights(path)

        tlr = make_tied_linear_relu()
        with pytest.raises(TypeError, match="TiedLinear.*TiedLinearRelu"):
            tlr.load_weights(path)

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        """Loading from a nonexistent path must raise FileNotFoundError."""
        model = make_tied_linear()
        with pytest.raises(FileNotFoundError):
            model.load_weights(tmp_path / "does_not_exist.safetensors")

    def test_load_missing_metadata_raises_value_error(self, tmp_path):
        """A safetensors file without class metadata must be rejected."""
        path = tmp_path / "bare.safetensors"
        save_file({"W": torch.randn(4, 8), "b": torch.zeros(8)}, str(path))

        model = make_tied_linear_relu()
        with pytest.raises(ValueError, match="no 'class' metadata"):
            model.load_weights(path)

    def test_save_auto_appends_extension(self, tmp_path):
        """save_weights without .safetensors extension must auto-append it."""
        model = make_tied_linear()
        path = model.save_weights(tmp_path / "model")
        assert path.suffix == ".safetensors"
        assert path.exists()

    def test_load_auto_appends_extension(self, tmp_path):
        """load_weights without .safetensors extension must auto-append it."""
        model = make_tied_linear()
        model.save_weights(tmp_path / "model.safetensors")
        fresh = make_tied_linear(seed=99)
        fresh.load_weights(tmp_path / "model")  # no extension
        assert torch.equal(model.state_dict()["W"], fresh.state_dict()["W"])

    def test_default_filename_contains_class_and_dims(self, tmp_path, monkeypatch):
        """Default path must include class name and dimensions."""
        monkeypatch.chdir(tmp_path)
        model = make_tied_linear()
        path = model.save_weights()
        assert path.exists()
        assert f"TiedLinear_{N_FEATURES}x{N_HIDDEN}" in path.name

    def test_json_companion_file_created(self, tmp_path):
        """save_weights must produce a .json companion alongside .safetensors."""
        model = make_tied_linear()
        path = model.save_weights(tmp_path / "model")
        assert path.with_suffix(".json").exists()


class TestCollectAttrs:
    """_collect_attrs and _serialize_value edge cases."""

    def test_collect_attrs_excludes_nn_module_internals(self):
        """nn.Module bookkeeping keys like _parameters must not appear."""
        model = make_tied_linear()
        attrs = model._collect_attrs()
        for internal in AutoEncoderBase._NN_MODULE_INTERNALS:
            assert internal not in attrs

    def test_collect_attrs_includes_user_attrs(self):
        """n_features, n_hidden, generator must be present in collected attrs."""
        model = make_tied_linear()
        attrs = model._collect_attrs()
        assert "n_features" in attrs
        assert "n_hidden" in attrs
        assert "generator" in attrs

    def test_serialize_none(self):
        """None should pass through as-is."""
        assert AutoEncoderBase._serialize_value(None) is None

    def test_serialize_primitives(self):
        """int, float, str, bool should pass through unchanged."""
        assert AutoEncoderBase._serialize_value(42) == 42
        assert AutoEncoderBase._serialize_value(3.14) == 3.14
        assert AutoEncoderBase._serialize_value("hello") == "hello"
        assert AutoEncoderBase._serialize_value(True) is True

    def test_serialize_device(self):
        """torch.device should become its string representation."""
        assert AutoEncoderBase._serialize_value(torch.device("cpu")) == "cpu"

    def test_serialize_generator(self):
        """Generator should serialize to a dict with type/device/seed."""
        gen = _gen(123)
        result = AutoEncoderBase._serialize_value(gen)
        assert result["type"] == "Generator"
        assert result["initial_seed"] == 123

    def test_serialize_parameter_returns_skip(self):
        """nn.Parameter should be skipped (returned via _SKIP sentinel)."""
        from occhio.autoencoders.base import _SKIP

        p = nn.Parameter(torch.randn(3))
        assert AutoEncoderBase._serialize_value(p) is _SKIP

    def test_serialize_tensor(self):
        """Plain tensors should serialize to shape+dtype dict."""
        t = torch.randn(3, 4)
        result = AutoEncoderBase._serialize_value(t)
        assert result["shape"] == [3, 4]
        assert "dtype" in result

    def test_serialize_list(self):
        """Lists should be recursively serialized."""
        result = AutoEncoderBase._serialize_value([1, 2.0, "three"])
        assert result == [1, 2.0, "three"]

    def test_serialize_dict(self):
        """Dicts should be recursively serialized with string keys."""
        result = AutoEncoderBase._serialize_value({"a": 1, "b": 2.0})
        assert result == {"a": 1, "b": 2.0}

    def test_serialize_callable_uses_repr(self):
        """Callables should fall through to repr."""
        fn = lambda x: x  # noqa: E731
        result = AutoEncoderBase._serialize_value(fn)
        assert isinstance(result, str)  # repr produces a string


class TestInitSubclass:
    """__init_subclass__ validation: n_features and n_hidden must be set."""

    def test_subclass_missing_n_features_raises(self):
        """A subclass that doesn't set n_features must error on instantiation."""

        class BadAE(AutoEncoderBase):
            def __init__(self):
                super().__init__(n_features=5, n_hidden=3)
                # Deliberately delete n_features to trigger check
                del self.n_features

            def encode(self, x):
                return x

            def decode(self, z):
                return z

            def resample_weights(self):
                pass

        with pytest.raises(AttributeError, match="n_features"):
            BadAE()

    def test_subclass_missing_n_hidden_raises(self):
        """A subclass that doesn't set n_hidden must error on instantiation."""

        class BadAE(AutoEncoderBase):
            def __init__(self):
                super().__init__(n_features=5, n_hidden=3)
                del self.n_hidden

            def encode(self, x):
                return x

            def decode(self, z):
                return z

            def resample_weights(self):
                pass

        with pytest.raises(AttributeError, match="n_hidden"):
            BadAE()

    def test_well_formed_subclass_ok(self):
        """A subclass that sets both n_features and n_hidden should work fine."""

        class GoodAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden):
                super().__init__(n_features, n_hidden)

            def encode(self, x):
                return x[:, : self.n_hidden]

            def decode(self, z):
                return torch.zeros(z.shape[0], self.n_features)

            def resample_weights(self):
                pass

        ae = GoodAE(8, 4)
        assert ae.n_features == 8
        assert ae.n_hidden == 4


class TestDeviceMismatch:
    """Generator/device mismatch at construction must raise ValueError."""

    def test_generator_device_mismatch_raises(self):
        """If generator is on CPU but device is 'meta', must raise ValueError.

        We use 'meta' as a non-CPU device that doesn't require hardware.
        """
        gen = torch.Generator(device="cpu").manual_seed(42)
        with pytest.raises(ValueError, match="Generator lives on"):
            TiedLinear(N_FEATURES, N_HIDDEN, device="meta", generator=gen)

    def test_device_from_generator_when_no_device_specified(self):
        """When device is None, _init_device should come from the generator."""
        gen = torch.Generator(device="cpu").manual_seed(42)
        model = TiedLinear(N_FEATURES, N_HIDDEN, generator=gen)
        assert model._init_device == torch.device("cpu")

    def test_no_device_no_generator(self):
        """When neither device nor generator is specified, _init_device is None."""
        model = TiedLinear(N_FEATURES, N_HIDDEN)
        assert model._init_device is None


class TestCustomLossFn:
    """Custom loss_fn passed at construction overrides the default loss."""

    def test_custom_loss_fn_used(self):
        """Passing loss_fn should replace the instance method."""

        def dummy_loss(x_true, x_hat, importances):
            return torch.tensor(999.0)

        model = TiedLinear(N_FEATURES, N_HIDDEN, loss_fn=dummy_loss, device=DEVICE)
        x = torch.randn(2, N_FEATURES, device=DEVICE)
        x_hat, _ = model(x)
        loss = model.loss(x, x_hat, None)
        assert loss.item() == pytest.approx(999.0)


# ============================================================================
# 3. TiedLinear / TiedLinearRelu SPECIFIC TESTS
# ============================================================================


class TestTiedLinear:
    """Tests specific to TiedLinear (no activation in decode)."""

    def test_encode_is_linear(self):
        """encode(x) = x @ W.T should be a purely linear operation."""
        model = make_tied_linear()
        x1 = torch.randn(1, N_FEATURES, device=DEVICE)
        x2 = torch.randn(1, N_FEATURES, device=DEVICE)
        alpha = 0.5
        # Linearity: encode(alpha*x1 + x2) == alpha*encode(x1) + encode(x2)
        z_combined = model.encode(alpha * x1 + x2)
        z_separate = alpha * model.encode(x1) + model.encode(x2)
        assert torch.allclose(z_combined, z_separate, atol=1e-5)

    def test_decode_includes_bias(self):
        """decode(z) = z @ W + b. With z=0 the output should be b."""
        model = make_tied_linear()
        z = torch.zeros(1, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert torch.allclose(out, model.b.unsqueeze(0), atol=1e-6)

    def test_W_columns_unit_norm(self):
        """After construction, W columns (dim=0 norms along the feature axis)
        should be approximately unit norm (the constructor normalizes them)."""
        model = make_tied_linear()
        col_norms = model.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


class TestTiedLinearRelu:
    """Tests specific to TiedLinearRelu (ReLU in decode)."""

    def test_decode_output_non_negative(self, random_latent):
        """ReLU activation means decode output must be >= 0."""
        model = make_tied_linear_relu()
        out = model.decode(random_latent)
        assert (out >= 0).all()

    def test_zero_input_produces_zero_latent(self):
        """encode(0) = 0 @ W.T = 0 for tied weights without encoder bias."""
        model = make_tied_linear_relu()
        x = torch.zeros(1, N_FEATURES, device=DEVICE)
        z = model.encode(x)
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-7)

    def test_W_columns_unit_norm(self):
        """After construction, W columns should be unit norm."""
        model = make_tied_linear_relu()
        col_norms = model.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


# ============================================================================
# 4. MLPEncoder SPECIFIC TESTS
# ============================================================================


class TestMLPEncoder:
    """Tests specific to MLPEncoder with multi-layer embedding/unembedding."""

    def test_embedding_unembedding_dim_mismatch_raises(self):
        """Latent dims of embedding[-1] and unembedding[0] must match."""
        with pytest.raises(AssertionError, match="latent dims must match"):
            MLPEncoder(
                embedding=[8, 6, 4],
                unembedding=[5, 6, 8],  # 5 != 4
                device=DEVICE,
            )

    def test_input_output_dim_mismatch_raises(self):
        """embedding[0] must equal unembedding[-1]."""
        with pytest.raises(AssertionError, match="input/output dims must match"):
            MLPEncoder(
                embedding=[8, 4],
                unembedding=[4, 10],  # 10 != 8
                device=DEVICE,
            )

    def test_too_short_embedding_raises(self):
        """embedding must have at least 2 elements."""
        with pytest.raises(AssertionError, match="at least"):
            MLPEncoder(embedding=[8], unembedding=[8], device=DEVICE)

    def test_tied_initialization(self):
        """With tied_initialization=True, decoder weights start as encoder transposed."""
        model = MLPEncoder(
            embedding=[8, 6, 4],
            unembedding=[4, 6, 8],
            tied_initialization=True,
            generator=_gen(),
            device=DEVICE,
        )
        # Decoder weight 0 should be transpose of encoder weight 1 at init
        enc_w_last = model.encoder_weights[-1].data  # (4, 6)
        dec_w_first = model.decoder_weights[0].data  # (6, 4)
        # decoder weight i is initialized as transpose of encoder weight (n-1-i)
        assert torch.allclose(dec_w_first, enc_w_last.t(), atol=1e-6)

    def test_tied_initialization_rejects_non_mirror(self):
        """tied_initialization requires unembedding == embedding[::-1]."""
        with pytest.raises(AssertionError, match="tied_initialization"):
            MLPEncoder(
                embedding=[8, 6, 4],
                unembedding=[4, 5, 8],  # not the reverse
                tied_initialization=True,
                device=DEVICE,
            )

    def test_deep_mlp(self):
        """An MLP with 4 layers should work end-to-end."""
        model = MLPEncoder(
            embedding=[16, 12, 8, 4],
            unembedding=[4, 8, 12, 16],
            generator=_gen(),
            device=DEVICE,
        )
        x = torch.randn(4, 16, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (4, 16)
        assert z.shape == (4, 4)

    def test_decode_output_non_negative(self):
        """MLPEncoder decode applies final ReLU, so output must be >= 0."""
        model = make_mlp_encoder()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()

    def test_resample_rebuilds_layers(self):
        """resample_weights calls _build_layers, so encoder/decoder counts must stay."""
        model = make_mlp_encoder()
        n_enc = len(model.encoder_weights)
        n_dec = len(model.decoder_weights)
        model.resample_weights()
        assert len(model.encoder_weights) == n_enc
        assert len(model.decoder_weights) == n_dec

    def test_different_embedding_unembedding_dims(self):
        """Asymmetric intermediate dimensions should work."""
        model = MLPEncoder(
            embedding=[8, 10, 4],
            unembedding=[4, 12, 8],
            generator=_gen(),
            device=DEVICE,
        )
        x = torch.randn(3, 8, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (3, 8)
        assert z.shape == (3, 4)


class TestTiedMLPEncoder:
    """Tests specific to TiedMLPEncoder with weight-tied decoder."""

    def test_decoder_uses_encoder_weights_transposed(self):
        """decode() should use encoder weights in reverse order, untransposed."""
        model = make_tied_mlp_encoder()
        # The decoder uses encoder_weights in reverse, w (not w.T)
        # verify by checking the computation graph is connected
        x = torch.randn(2, N_FEATURES, device=DEVICE)
        x_hat, z = model(x)
        loss = x_hat.sum()
        loss.backward()
        # Encoder weights should get gradients from both encode and decode paths
        for w in model.encoder_weights:
            assert w.grad is not None

    def test_dims_too_short_raises(self):
        """dims must have at least 2 elements."""
        with pytest.raises(AssertionError, match="at least"):
            TiedMLPEncoder(dims=[8], device=DEVICE)

    def test_decode_output_non_negative(self):
        """TiedMLPEncoder decode applies final ReLU, so output must be >= 0."""
        model = make_tied_mlp_encoder()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()

    def test_decoder_biases_count(self):
        """Decoder should have len(dims)-1 bias parameters (one per decode layer)."""
        model = make_tied_mlp_encoder()
        # dims = [8, 6, 4], decoder mirrors: [4, 6, 8], so 2 bias params
        assert len(model.decoder_biases) == len(model.dims) - 1

    def test_three_layer_dims(self):
        """3-layer config: [16, 8, 4] should work."""
        model = TiedMLPEncoder(dims=[16, 8, 4], device=DEVICE)
        x = torch.randn(3, 16, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (3, 16)
        assert z.shape == (3, 4)


# ============================================================================
# 5. ATTENTION AUTOENCODERS SPECIFIC TESTS
# ============================================================================


class TestSoftmax1:
    """Tests for the softmax1 helper function."""

    def test_output_sums_less_than_one(self):
        """softmax1 adds 1 to the denominator, so sum < 1 for finite inputs."""
        x = torch.randn(5, 10)
        out = softmax1(x)
        row_sums = out.sum(dim=-1)
        assert (row_sums < 1.0).all()

    def test_output_non_negative(self):
        """softmax1 uses exp, so output must be non-negative."""
        x = torch.randn(5, 10)
        out = softmax1(x)
        assert (out >= 0).all()

    def test_uniform_large_negative_inputs(self):
        """softmax1 subtracts the max before exp, so uniform large-negative inputs
        become exp(0)/(n*exp(0)+1) = 1/(n+1). This tests the +1 denominator behavior."""
        x = torch.full((2, 3), -100.0)
        out = softmax1(x)
        # All entries identical after max-subtraction, so each is 1/(3+1) = 0.25
        expected = torch.full((2, 3), 1.0 / 4.0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_single_large_positive_rest_very_negative(self):
        """When one logit dominates, the others should be near zero due to the
        +1 denominator suppressing all entries slightly."""
        x = torch.tensor([[100.0, -100.0, -100.0]])
        out = softmax1(x)
        # The dominant entry should be close to 1/(1+1) = 0.5 (exp(0)/(exp(0)+1))
        # The others should be near 0
        assert out[0, 0] > 0.49
        assert out[0, 1] < 1e-6
        assert out[0, 2] < 1e-6

    def test_uniform_inputs(self):
        """For identical logits, softmax1 should produce uniform weights."""
        x = torch.full((1, 4), 5.0)
        out = softmax1(x)
        # All entries should be equal
        assert torch.allclose(out[0], out[0, 0].expand(4), atol=1e-6)

    def test_1d_input(self):
        """softmax1 with dim=-1 should handle 1D input."""
        x = torch.tensor([1.0, 2.0, 3.0])
        out = softmax1(x, dim=-1)
        assert out.shape == (3,)
        assert (out >= 0).all()

    def test_gradient_flows(self):
        """softmax1 must be differentiable."""
        x = torch.randn(3, 4, requires_grad=True)
        out = softmax1(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAttentionNHeadsDivisibility:
    """n_hidden must be divisible by n_heads for all attention AEs."""

    @pytest.mark.parametrize(
        "cls",
        [AttnLinearAE, AttnAttnAE, LinearAttnAE],
        ids=["AttnLinearAE", "AttnAttnAE", "LinearAttnAE"],
    )
    def test_indivisible_raises(self, cls):
        """n_hidden=5, n_heads=2 should raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            cls(
                n_features=8,
                n_hidden=5,
                n_heads=2,
                dict_size=4,
                device=DEVICE,
            )


class TestAttnLinearAE:
    """Tests specific to AttnLinearAE (attention encoder, linear decoder)."""

    def test_encoder_projs_count(self):
        """Should have one encoder projection matrix per head."""
        model = make_attn_linear_ae()
        assert len(model.encoder_projs) == 2  # n_heads=2

    def test_value_matrices_count(self):
        """Should have one value matrix per head."""
        model = make_attn_linear_ae()
        assert len(model.value_matrices) == 2

    def test_alpha_parameter_exists(self):
        """alpha controls skip connection and must be a learnable parameter."""
        model = make_attn_linear_ae()
        assert hasattr(model, "alpha")
        assert isinstance(model.alpha, nn.Parameter)

    def test_single_head(self):
        """Single head (n_heads=1) should work without issues."""
        model = AttnLinearAE(
            n_features=8, n_hidden=4, n_heads=1, dict_size=6, device=DEVICE
        )
        x = torch.randn(3, 8, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (3, 8)
        assert z.shape == (3, 4)

    def test_decode_output_non_negative(self):
        """AttnLinearAE decode uses ReLU, so output must be >= 0."""
        model = make_attn_linear_ae()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()


class TestAttnAttnAE:
    """Tests specific to AttnAttnAE (attention encoder, attention decoder)."""

    def test_encoder_and_decoder_values_count(self):
        """Should have n_heads encoder_values and n_heads decoder_values."""
        model = make_attn_attn_ae()
        assert len(model.encoder_values) == 2
        assert len(model.decoder_values) == 2

    def test_decode_output_non_negative(self):
        """AttnAttnAE decode uses ReLU, so output must be >= 0."""
        model = make_attn_attn_ae()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()


class TestLinearAttnAE:
    """Tests specific to LinearAttnAE (linear encoder, attention decoder)."""

    def test_encode_is_linear(self):
        """LinearAttnAE encoder is x @ W_enc.T which is linear."""
        model = make_linear_attn_ae()
        x1 = torch.randn(1, N_FEATURES, device=DEVICE)
        x2 = torch.randn(1, N_FEATURES, device=DEVICE)
        z_sum = model.encode(x1 + x2)
        z_sep = model.encode(x1) + model.encode(x2)
        assert torch.allclose(z_sum, z_sep, atol=1e-5)

    def test_decoder_projs_and_values_count(self):
        """Should have n_heads decoder projection and value matrices."""
        model = make_linear_attn_ae()
        assert len(model.decoder_projs) == 2
        assert len(model.decoder_values) == 2

    def test_decode_output_non_negative(self):
        """LinearAttnAE decode uses ReLU, so output must be >= 0."""
        model = make_linear_attn_ae()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()

    def test_various_dict_sizes(self):
        """Different dict_size values should all work."""
        for ds in [2, 8, 16]:
            model = LinearAttnAE(
                n_features=8, n_hidden=4, n_heads=2, dict_size=ds, device=DEVICE
            )
            x = torch.randn(3, 8, device=DEVICE)
            x_hat, z = model(x)
            assert x_hat.shape == (3, 8)


class TestAttentionMultiHead:
    """Test multi-head configurations across all attention AEs."""

    @pytest.mark.parametrize(
        "cls",
        [AttnLinearAE, AttnAttnAE, LinearAttnAE],
        ids=["AttnLinearAE", "AttnAttnAE", "LinearAttnAE"],
    )
    def test_four_heads(self, cls):
        """4 heads with n_hidden=8 (value_dim=2) should work."""
        model = cls(
            n_features=8,
            n_hidden=8,
            n_heads=4,
            dict_size=5,
            device=DEVICE,
        )
        x = torch.randn(3, 8, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (3, 8)
        assert z.shape == (3, 8)


# ============================================================================
# 6. ComputeAutoEncoder SPECIFIC TESTS
# ============================================================================


class TestComputeAutoEncoder:
    """Tests specific to ComputeAutoEncoder with compute_step and dual losses."""

    def test_compute_step_shape(self):
        """compute_step(h) must return same shape as h."""
        model = make_compute_ae()
        h = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        z = model.compute_step(h)
        assert z.shape == h.shape

    def test_forward_includes_compute_step(self):
        """forward() uses compute_step, so z != encode(x)."""
        model = make_compute_ae()
        x = torch.randn(BATCH, N_FEATURES, device=DEVICE)
        _, z_fwd = model(x)
        h = model.encode(x)
        z_with_compute = model.compute_step(h)
        assert torch.allclose(z_fwd, z_with_compute, atol=1e-6)

    def test_compute_step_is_residual(self):
        """compute_step(h) = h + h @ Z.T includes a residual connection."""
        model = make_compute_ae()
        h = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        z = model.compute_step(h)
        expected = h + h @ model.Z.T
        assert torch.allclose(z, expected, atol=1e-6)

    def test_decode_softmax_sums_to_one(self):
        """With decode_activation='softmax', decode output rows sum to 1."""
        model = ComputeAutoEncoder(
            N=N_FEATURES, k=N_HIDDEN, decode_activation="softmax"
        )
        z = torch.randn(4, N_HIDDEN)
        out = model.decode(z)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_decode_relu_non_negative(self):
        """With decode_activation='relu', decode output must be >= 0."""
        model = ComputeAutoEncoder(N=N_FEATURES, k=N_HIDDEN, decode_activation="relu")
        z = torch.randn(4, N_HIDDEN)
        out = model.decode(z)
        assert (out >= 0).all()

    def test_ce_loss_returns_scalar(self):
        """ce_loss should return a 0-dim tensor."""
        model = ComputeAutoEncoder(
            N=N_FEATURES, k=N_HIDDEN, decode_activation="softmax"
        )
        x = torch.randn(BATCH, N_FEATURES)
        y_hat, _ = model(x)
        y_idx = torch.randint(0, N_FEATURES, (BATCH,))
        importances = torch.ones(N_FEATURES)
        loss = model.ce_loss(y_hat, y_idx, importances)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mse_loss_returns_scalar(self):
        """mse_loss should return a 0-dim tensor."""
        model = make_compute_ae()
        x = torch.randn(BATCH, N_FEATURES, device=DEVICE)
        y_hat, _ = model(x)
        loss = model.mse_loss(y_hat, x, importances=None)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ce_loss_gradient_flow(self):
        """ce_loss must allow gradient flow back to model parameters."""
        model = ComputeAutoEncoder(
            N=N_FEATURES, k=N_HIDDEN, decode_activation="softmax"
        )
        x = torch.randn(BATCH, N_FEATURES)
        y_hat, _ = model(x)
        y_idx = torch.randint(0, N_FEATURES, (BATCH,))
        importances = torch.ones(N_FEATURES)
        loss = model.ce_loss(y_hat, y_idx, importances)
        loss.backward()
        assert model.W.grad is not None

    def test_mse_loss_gradient_flow(self):
        """mse_loss must allow gradient flow back to model parameters."""
        model = make_compute_ae()
        x = torch.randn(BATCH, N_FEATURES, device=DEVICE)
        y_hat, _ = model(x)
        loss = model.mse_loss(y_hat, x, importances=None)
        loss.backward()
        assert model.W.grad is not None

    def test_resample_changes_W_Z(self):
        """resample_weights must produce new W and Z."""
        model = make_compute_ae()
        old_W = model.W.data.clone()
        old_Z = model.Z.data.clone()
        model.resample_weights()
        # At least one of W, Z should change
        changed = not torch.equal(model.W.data, old_W) or not torch.equal(
            model.Z.data, old_Z
        )
        assert changed


# ============================================================================
# 7. SynthAE SPECIFIC TESTS
# ============================================================================


class TestSynthAE:
    """Tests specific to SynthAE with orthogonalization and frozen W."""

    def test_W_columns_unit_norm(self):
        """After construction, W columns should be unit norm."""
        model = make_synth_ae()
        col_norms = model.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    def test_freeze_W_makes_W_non_trainable(self):
        """After freeze_W(), W.requires_grad must be False."""
        model = make_synth_ae()
        assert not model.W.requires_grad

    def test_freeze_W_leaves_b_trainable(self):
        """freeze_W() should only freeze W, not b."""
        model = make_synth_ae()
        assert model.b.requires_grad

    def test_orthogonalize_reduces_cosine_similarity(self):
        """Orthogonalization should reduce pairwise cosine similarity vs random init.

        Uses small dims so the orthogonalization procedure is fast.
        """
        torch.manual_seed(42)
        # Random (non-orthogonalized)
        random_ae = SynthAE(
            n_features=6,
            n_hidden=4,
            orthogonalize=False,
            generator=_gen(42),
            device=DEVICE,
        )
        rho_random = random_ae.rho_mm

        # Orthogonalized
        ortho_ae = SynthAE(
            n_features=6,
            n_hidden=4,
            orthogonalize=True,
            ortho_steps=200,
            ortho_lr=0.01,
            generator=_gen(42),
            device=DEVICE,
        )
        rho_ortho = ortho_ae.rho_mm

        assert rho_ortho < rho_random, (
            f"Orthogonalization did not reduce rho_mm: "
            f"ortho={rho_ortho:.4f} >= random={rho_random:.4f}"
        )

    def test_rho_mm_returns_float_in_range(self):
        """rho_mm should be a float in [0, 1]."""
        model = make_synth_ae()
        rho = model.rho_mm
        assert isinstance(rho, float)
        assert 0.0 <= rho <= 1.0

    def test_rho_mm_orthogonal_is_low(self):
        """When n_features == n_hidden (square W), columns can be orthogonal,
        and rho_mm should be near 0 after orthogonalization."""
        model = SynthAE(
            n_features=4,
            n_hidden=4,
            orthogonalize=True,
            ortho_steps=500,
            ortho_lr=0.01,
            generator=_gen(),
            device=DEVICE,
        )
        assert model.rho_mm < 0.1, (
            f"Expected near-0 rho_mm for square W, got {model.rho_mm}"
        )

    def test_decode_output_non_negative(self):
        """SynthAE decode uses ReLU, so output must be >= 0."""
        model = make_synth_ae()
        z = torch.randn(BATCH, N_HIDDEN, device=DEVICE)
        out = model.decode(z)
        assert (out >= 0).all()

    def test_gradient_only_flows_through_b(self):
        """With W frozen, only b should receive gradients."""
        model = make_synth_ae()
        x = torch.randn(BATCH, N_FEATURES, device=DEVICE)
        x_hat, z = model(x)
        loss = model.loss(x, x_hat, importances=None)
        loss.backward()
        assert model.W.grad is None or (model.W.grad == 0).all()
        assert model.b.grad is not None
        assert model.b.grad.abs().sum() > 0

    def test_orthogonalize_preserves_unit_norm(self):
        """After orthogonalization, columns should still be unit norm."""
        model = SynthAE(
            n_features=6,
            n_hidden=4,
            orthogonalize=True,
            ortho_steps=100,
            generator=_gen(),
            device=DEVICE,
        )
        col_norms = model.W.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4)

    def test_resample_re_freezes_W(self):
        """resample_weights() should re-freeze W."""
        model = make_synth_ae()
        model.resample_weights()
        assert not model.W.requires_grad


# ============================================================================
# 8. EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Boundary and degenerate input handling."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_zero_input(self, factory):
        """Zero input should not produce NaN."""
        model = factory()
        x = torch.zeros(1, N_FEATURES, device=DEVICE)
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in reconstruction from zero input"
        assert not torch.isnan(z).any(), "NaN in latent from zero input"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_large_input(self, factory):
        """Large input values should not produce NaN (tests numerical stability)."""
        model = factory()
        x = torch.ones(1, N_FEATURES, device=DEVICE) * 100.0
        x_hat, z = model(x)
        assert not torch.isnan(x_hat).any(), "NaN in reconstruction from large input"
        assert not torch.isnan(z).any(), "NaN in latent from large input"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_large_batch(self, factory):
        """Batch size 256 should not cause issues."""
        model = factory()
        x = torch.randn(256, N_FEATURES, device=DEVICE)
        x_hat, z = model(x)
        assert x_hat.shape == (256, N_FEATURES)
        assert z.shape == (256, N_HIDDEN)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_deterministic_with_same_seed(self, factory):
        """Two models from the same seed should produce identical outputs."""
        m1 = factory(seed=SEED)
        m2 = factory(seed=SEED)
        x = torch.randn(4, N_FEATURES, device=DEVICE, generator=_gen(0))
        x_hat1, z1 = m1(x)
        x_hat2, z2 = m2(x)
        assert torch.allclose(z1, z2, atol=1e-6), "Same seed produced different latents"
        assert torch.allclose(x_hat1, x_hat2, atol=1e-6), (
            "Same seed produced different reconstructions"
        )

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_different_seeds_produce_different_params(self, factory):
        """Two models from different seeds must have at least one different parameter.

        Catches bugs where the generator is silently ignored.
        """
        m1 = factory(seed=1)
        m2 = factory(seed=2)
        sd1 = m1.state_dict()
        sd2 = m2.state_dict()
        any_diff = any(not torch.equal(sd1[k], sd2[k]) for k in sd1)
        assert any_diff, "Different seeds produced identical parameters"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_eval_mode_consistency(self, factory, random_input):
        """Output should be the same in train and eval mode (no dropout/batchnorm)."""
        model = factory()
        model.train()
        x_hat_train, z_train = model(random_input)
        model.eval()
        x_hat_eval, z_eval = model(random_input)
        assert torch.allclose(x_hat_train, x_hat_eval, atol=1e-6)
        assert torch.allclose(z_train, z_eval, atol=1e-6)


class TestLossImportancesWeighting:
    """Verify that importances weight the loss correctly."""

    def test_double_importances_doubles_loss(self):
        """Doubling importances should double the base MSE loss."""
        model = make_tied_linear()
        x = torch.randn(4, N_FEATURES, device=DEVICE)
        x_hat, _ = model(x)

        imp1 = torch.ones(N_FEATURES, device=DEVICE)
        imp2 = torch.ones(N_FEATURES, device=DEVICE) * 2.0

        loss1 = model.loss(x, x_hat, imp1)
        loss2 = model.loss(x, x_hat, imp2)
        assert torch.allclose(loss2, loss1 * 2.0, atol=1e-5)

    def test_zero_importances_zero_loss(self):
        """Zero importances should produce zero loss regardless of reconstruction error."""
        model = make_tied_linear()
        x = torch.randn(4, N_FEATURES, device=DEVICE)
        x_hat = torch.randn(4, N_FEATURES, device=DEVICE)  # deliberately wrong
        imp = torch.zeros(N_FEATURES, device=DEVICE)
        loss = model.loss(x, x_hat, imp)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestSaveLoadAllClasses:
    """Round-trip save/load for ALL autoencoder classes (beyond the existing tests)."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_full_round_trip(self, factory, tmp_path):
        """Every class must survive a save/load cycle with weights intact."""
        model = factory()
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        path = tmp_path / "model.safetensors"
        model.save_weights(path)

        restored = factory(seed=99)  # different seed to prove load overwrites
        restored.load_weights(path)

        for k in orig:
            assert torch.equal(orig[k], restored.state_dict()[k]), (
                f"Round-trip mismatch on key '{k}'"
            )

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_output_matches_after_load(self, factory, tmp_path):
        """A loaded model must produce the same output as the original."""
        model = factory()
        x = torch.randn(4, N_FEATURES, device=DEVICE, generator=_gen(0))
        x_hat_orig, z_orig = model(x)

        path = tmp_path / "model.safetensors"
        model.save_weights(path)

        restored = factory(seed=99)
        restored.load_weights(path)
        x_hat_loaded, z_loaded = restored(x)

        assert torch.allclose(x_hat_orig, x_hat_loaded, atol=1e-6)
        assert torch.allclose(z_orig, z_loaded, atol=1e-6)
