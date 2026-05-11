# ABOUTME: Exhaustive tests for the architecture-invariant autoencoder save/load system.
# ABOUTME: Covers get_config protocol, registry, from_local round-trips, custom subclass
# ABOUTME: extensibility, save_weights metadata, from_hub/push_to_hub (mocked), and
# ABOUTME: ToyModel hub integration.
"""Tests for the pretrained save/load/hub protocol in occhio.autoencoders.

Testing strategy:
- Config protocol: verify get_config() for every built-in AE class returns the
  right keys and can reconstruct the same architecture.
- Registry: verify all 9 built-in classes register, custom subclasses auto-register,
  and class aliases resolve correctly.
- from_local round-trips: every AE class save -> from_local -> encode produces
  identical output; device override; legacy files; error paths.
- Custom subclass extensibility: subclasses with and without get_config overrides.
- save_weights metadata: safetensors metadata, JSON sidecar, class name.
- from_hub / push_to_hub: mocked HF Hub calls verify delegation.
- ToyModel integration: mocked from_hub and push_to_hub.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
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

# ── constants ────────────────────────────────────────────────────────────────

DEVICE = "cpu"
N_FEATURES = 8
N_HIDDEN = 4
BATCH = 6
SEED = 42


def _gen(seed=SEED):
    return torch.Generator(device=DEVICE).manual_seed(seed)


# ── factory helpers ──────────────────────────────────────────────────────────


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

ALL_BUILTIN_CLASSES = [
    TiedLinear,
    TiedLinearRelu,
    MLPEncoder,
    TiedMLPEncoder,
    AttnLinearAE,
    AttnAttnAE,
    LinearAttnAE,
    ComputeAutoEncoder,
    SynthAE,
]


def _random_input(seed=0):
    return torch.randn(BATCH, N_FEATURES, device=DEVICE, generator=_gen(seed))


# ============================================================================
# Config protocol
# ============================================================================


class TestGetConfig:
    """Verify get_config() returns correct keys for every built-in AE class."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_get_config_returns_dict(self, factory):
        """get_config() must return a plain dict, not a subclass or other mapping."""
        ae = factory()
        config = ae.get_config()
        assert isinstance(config, dict)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_get_config_all_values_json_serializable(self, factory):
        """Config values must be JSON-serializable so they survive safetensors metadata."""
        ae = factory()
        config = ae.get_config()
        # Should not raise
        json.dumps(config)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_get_config_excludes_non_serializable(self, factory):
        """Config must not include loss_fn, device, or generator."""
        ae = factory()
        config = ae.get_config()
        for key in ("loss_fn", "device", "generator"):
            assert key not in config, f"Config should not contain '{key}'"

    def test_mlp_encoder_config_maps_embedding_dims_to_embedding(self):
        """MLPEncoder stores embedding_dims internally but get_config must use
        the constructor param name 'embedding' so reconstruction works."""
        ae = make_mlp_encoder()
        config = ae.get_config()
        assert "embedding" in config
        assert "unembedding" in config
        assert config["embedding"] == [N_FEATURES, 6, N_HIDDEN]
        assert config["unembedding"] == [N_HIDDEN, 6, N_FEATURES]
        # Must NOT expose internal attr names
        assert "embedding_dims" not in config
        assert "unembedding_dims" not in config

    def test_mlp_encoder_config_includes_tied_initialization(self):
        """MLPEncoder get_config must include tied_initialization flag."""
        ae = MLPEncoder(
            embedding=[N_FEATURES, 6, N_HIDDEN],
            unembedding=[N_HIDDEN, 6, N_FEATURES],
            tied_initialization=False,
            generator=_gen(),
            device=DEVICE,
        )
        config = ae.get_config()
        assert "tied_initialization" in config
        assert config["tied_initialization"] is False

    def test_compute_ae_config_maps_n_features_to_N(self):
        """ComputeAutoEncoder stores n_features/n_hidden but get_config must
        use the constructor param names N and k."""
        ae = make_compute_ae()
        config = ae.get_config()
        assert "N" in config
        assert "k" in config
        assert config["N"] == N_FEATURES
        assert config["k"] == N_HIDDEN
        # Must NOT expose internal attr names
        assert "n_features" not in config
        assert "n_hidden" not in config

    def test_compute_ae_config_includes_decode_activation(self):
        """ComputeAutoEncoder config must capture the decode_activation param."""
        ae = ComputeAutoEncoder(N=N_FEATURES, k=N_HIDDEN, decode_activation="relu")
        config = ae.get_config()
        assert config["decode_activation"] == "relu"

    def test_synth_ae_config_captures_ortho_params_via_prefix(self):
        """SynthAE stores ortho params as _orthogonalize, _ortho_lambda, etc.
        The default get_config() must find them via the '_' prefix fallback."""
        ae = SynthAE(
            n_features=N_FEATURES,
            n_hidden=N_HIDDEN,
            orthogonalize=True,
            ortho_lambda=2.0,
            ortho_steps=500,
            ortho_lr=0.005,
            ortho_chunk_size=512,
            generator=_gen(),
            device=DEVICE,
        )
        config = ae.get_config()
        assert config["orthogonalize"] is True
        assert config["ortho_lambda"] == 2.0
        assert config["ortho_steps"] == 500
        assert config["ortho_lr"] == 0.005
        assert config["ortho_chunk_size"] == 512

    def test_tied_linear_config_has_n_features_n_hidden(self):
        """TiedLinear uses default get_config -- must capture n_features, n_hidden."""
        ae = make_tied_linear()
        config = ae.get_config()
        assert config["n_features"] == N_FEATURES
        assert config["n_hidden"] == N_HIDDEN

    def test_tied_linear_relu_config_has_n_features_n_hidden(self):
        """TiedLinearRelu uses default get_config -- same basic keys."""
        ae = make_tied_linear_relu()
        config = ae.get_config()
        assert config["n_features"] == N_FEATURES
        assert config["n_hidden"] == N_HIDDEN

    def test_tied_mlp_encoder_config_has_dims(self):
        """TiedMLPEncoder uses default get_config -- must capture dims."""
        ae = make_tied_mlp_encoder()
        config = ae.get_config()
        assert "dims" in config
        assert config["dims"] == [N_FEATURES, 6, N_HIDDEN]

    def test_attn_linear_ae_config_has_heads_and_dict_size(self):
        """AttnLinearAE config must include n_heads and dict_size."""
        ae = make_attn_linear_ae()
        config = ae.get_config()
        assert config["n_features"] == N_FEATURES
        assert config["n_hidden"] == N_HIDDEN
        assert config["n_heads"] == 2
        assert config["dict_size"] == 5

    def test_attn_attn_ae_config_has_heads_and_dict_size(self):
        """AttnAttnAE config must include n_heads and dict_size."""
        ae = make_attn_attn_ae()
        config = ae.get_config()
        assert config["n_heads"] == 2
        assert config["dict_size"] == 5

    def test_linear_attn_ae_config_has_heads_and_dict_size(self):
        """LinearAttnAE config must include n_heads and dict_size."""
        ae = make_linear_attn_ae()
        config = ae.get_config()
        assert config["n_heads"] == 2
        assert config["dict_size"] == 5


class TestGetConfigReconstruction:
    """Verify get_config() output can reconstruct the same architecture."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_config_reconstructs_same_class(self, factory):
        """Constructing cls(**config) must produce an instance of the same class
        with the same n_features and n_hidden. Catches config keys that don't
        match constructor params."""
        ae = factory()
        config = ae.get_config()
        cls = type(ae)
        reconstructed = cls(**config)
        assert type(reconstructed) is cls
        assert reconstructed.n_features == ae.n_features
        assert reconstructed.n_hidden == ae.n_hidden

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_config_reconstructs_compatible_state_dict(self, factory):
        """A model built from get_config() must accept the original state_dict.
        Catches shape mismatches from wrong config values."""
        ae = factory()
        config = ae.get_config()
        cls = type(ae)
        reconstructed = cls(**config)
        # Must not raise
        reconstructed.load_state_dict(ae.state_dict(), strict=True)


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    """Verify the auto-registration mechanism and alias resolution."""

    def test_all_9_builtin_classes_registered(self):
        """All 9 concrete AE classes must be present in the registry."""
        for cls in ALL_BUILTIN_CLASSES:
            assert cls.__name__ in AutoEncoderBase._registry, (
                f"{cls.__name__} not in registry"
            )
            assert AutoEncoderBase._registry[cls.__name__] is cls

    def test_registry_has_at_least_9_entries(self):
        """Baseline count: at least the 9 built-in classes. Custom subclasses
        from other tests may add more, so we check >=."""
        assert len(AutoEncoderBase._registry) >= 9

    def test_custom_subclass_auto_registers_at_definition_time(self):
        """Defining a new subclass must immediately add it to the registry."""

        class _TestAutoRegAE(AutoEncoderBase):
            def __init__(self, n_features=4, n_hidden=2, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.W = nn.Parameter(torch.randn(n_hidden, n_features))
                self.b = nn.Parameter(torch.zeros(n_features))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return z @ self.W + self.b

            def resample_weights(self):
                pass

        assert "_TestAutoRegAE" in AutoEncoderBase._registry
        assert AutoEncoderBase._registry["_TestAutoRegAE"] is _TestAutoRegAE

    def test_class_aliases_resolve_old_names(self):
        """_class_aliases must map HuggingFaceAutoEncoder and PretrainedAE
        to TiedLinearRelu."""
        aliases = AutoEncoderBase._class_aliases
        assert aliases["HuggingFaceAutoEncoder"] == "TiedLinearRelu"
        assert aliases["PretrainedAE"] == "TiedLinearRelu"


# ============================================================================
# from_local round-trips
# ============================================================================


class TestFromLocalRoundTrip:
    """Verify save -> from_local -> encode produces identical output."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_encode_identical(self, factory, tmp_path):
        """The core invariant: save, reload, encode the same input, get the
        same latent. Catches serialization bugs, config errors, weight
        corruption."""
        ae = factory()
        x = _random_input()
        z_original = ae.encode(x)

        path = ae.save_weights(tmp_path / "model.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        z_loaded = loaded.encode(x)

        assert torch.allclose(z_original, z_loaded, atol=1e-6), (
            f"Encode mismatch for {type(ae).__name__}"
        )

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_forward_identical(self, factory, tmp_path):
        """Full forward pass (encode + decode) must match after round-trip."""
        ae = factory()
        x = _random_input()
        x_hat_orig, z_orig = ae(x)

        path = ae.save_weights(tmp_path / "model.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        x_hat_loaded, z_loaded = loaded(x)

        assert torch.allclose(z_orig, z_loaded, atol=1e-6)
        assert torch.allclose(x_hat_orig, x_hat_loaded, atol=1e-6)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_preserves_class(self, factory, tmp_path):
        """from_local must return the exact same class, not AutoEncoderBase."""
        ae = factory()
        path = ae.save_weights(tmp_path / "model.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded) is type(ae)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_state_dict_keys_match(self, factory, tmp_path):
        """State dict keys must be identical after round-trip."""
        ae = factory()
        path = ae.save_weights(tmp_path / "model.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        assert set(ae.state_dict().keys()) == set(loaded.state_dict().keys())

    def test_device_override_cpu(self, tmp_path):
        """from_local with explicit device='cpu' must place all params on cpu."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        loaded = AutoEncoderBase.from_local(path, device="cpu")
        for p in loaded.parameters():
            assert p.device == torch.device("cpu")

    def test_auto_appends_safetensors_extension(self, tmp_path):
        """from_local must auto-append .safetensors if missing from path."""
        ae = make_tied_linear_relu()
        ae.save_weights(tmp_path / "model.safetensors")
        # Pass without extension
        loaded = AutoEncoderBase.from_local(tmp_path / "model")
        assert type(loaded) is TiedLinearRelu


class TestFromLocalLegacy:
    """Test legacy file loading (no config metadata, just class + W key)."""

    def test_legacy_file_with_W_key_loads(self, tmp_path):
        """A safetensors file with class metadata but no config, containing a W key,
        should be loadable for simple tied-weight architectures."""
        ae = make_tied_linear_relu()
        state = ae.state_dict()

        # Save with class but no config metadata
        path = tmp_path / "legacy.safetensors"
        save_file(state, str(path), metadata={"class": "TiedLinearRelu"})

        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded) is TiedLinearRelu
        assert loaded.n_features == N_FEATURES
        assert loaded.n_hidden == N_HIDDEN

    def test_legacy_file_without_W_key_raises(self, tmp_path):
        """A legacy file with no config AND no W key cannot be loaded."""
        path = tmp_path / "bad_legacy.safetensors"
        state = {"some_other_key": torch.randn(3, 3)}
        save_file(state, str(path), metadata={"class": "TiedLinearRelu"})

        with pytest.raises(ValueError, match="no 'config' metadata and no 'W' key"):
            AutoEncoderBase.from_local(path)


class TestFromLocalErrors:
    """Test error paths in from_local."""

    def test_missing_class_metadata_raises(self, tmp_path):
        """A safetensors file without 'class' metadata must raise ValueError."""
        path = tmp_path / "no_class.safetensors"
        save_file({"W": torch.randn(4, 8)}, str(path))

        with pytest.raises(ValueError, match="no 'class' metadata"):
            AutoEncoderBase.from_local(path)

    def test_unknown_class_name_raises(self, tmp_path):
        """An unrecognized class name must raise ValueError with available classes."""
        path = tmp_path / "unknown.safetensors"
        config = json.dumps({"n_features": 8, "n_hidden": 4})
        save_file(
            {"W": torch.randn(4, 8)},
            str(path),
            metadata={"class": "NonexistentAE", "config": config},
        )

        with pytest.raises(
            ValueError, match="Unknown autoencoder class.*NonexistentAE"
        ):
            AutoEncoderBase.from_local(path)

    def test_extra_config_keys_raises(self, tmp_path):
        """Config keys not in the constructor signature must raise ValueError."""
        ae = make_tied_linear_relu()
        state = ae.state_dict()
        config = {"n_features": N_FEATURES, "n_hidden": N_HIDDEN, "bogus_param": 999}

        path = tmp_path / "extra_keys.safetensors"
        save_file(
            state,
            str(path),
            metadata={"class": "TiedLinearRelu", "config": json.dumps(config)},
        )

        with pytest.raises(ValueError, match="unexpected keys.*bogus_param"):
            AutoEncoderBase.from_local(path)

    def test_file_not_found_raises(self, tmp_path):
        """A nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AutoEncoderBase.from_local(tmp_path / "does_not_exist.safetensors")

    def test_alias_resolves_old_class_name(self, tmp_path):
        """Saving under the alias 'HuggingFaceAutoEncoder' should resolve
        to TiedLinearRelu via _class_aliases."""
        ae = make_tied_linear_relu()
        state = ae.state_dict()
        config = {"n_features": N_FEATURES, "n_hidden": N_HIDDEN}

        path = tmp_path / "aliased.safetensors"
        save_file(
            state,
            str(path),
            metadata={
                "class": "HuggingFaceAutoEncoder",
                "config": json.dumps(config),
            },
        )

        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded) is TiedLinearRelu


# ============================================================================
# Custom subclass extensibility
# ============================================================================


class TestCustomSubclass:
    """Test that user-defined subclasses integrate with the save/load protocol."""

    def test_custom_subclass_with_matching_names_no_override(self, tmp_path):
        """A subclass whose constructor param names match self.<name> attributes
        should work with the default get_config() -- no override needed."""

        class SimpleCustomAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, scale=1.0, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.scale = scale
                self.W = nn.Parameter(torch.randn(n_hidden, n_features) * scale)
                self.b = nn.Parameter(torch.zeros(n_features))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return torch.relu(z @ self.W + self.b)

            def resample_weights(self):
                pass

        ae = SimpleCustomAE(N_FEATURES, N_HIDDEN, scale=2.5)
        config = ae.get_config()
        assert config["scale"] == 2.5

        x = _random_input()
        z_orig = ae.encode(x)

        path = ae.save_weights(tmp_path / "custom.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded).__name__ == "SimpleCustomAE"
        assert loaded.scale == 2.5
        assert torch.allclose(loaded.encode(x), z_orig, atol=1e-6)

    def test_custom_subclass_with_get_config_override(self, tmp_path):
        """A subclass that stores params under different attr names must
        override get_config() and still round-trip correctly."""

        class RenamedParamAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, multiplier=1.0, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self._my_mult = multiplier  # mismatched name
                self.W = nn.Parameter(torch.randn(n_hidden, n_features) * multiplier)
                self.b = nn.Parameter(torch.zeros(n_features))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return torch.relu(z @ self.W + self.b)

            def resample_weights(self):
                pass

            def get_config(self):
                return {
                    "n_features": self.n_features,
                    "n_hidden": self.n_hidden,
                    "multiplier": self._my_mult,
                }

        ae = RenamedParamAE(N_FEATURES, N_HIDDEN, multiplier=3.0)
        x = _random_input()
        z_orig = ae.encode(x)

        path = ae.save_weights(tmp_path / "renamed.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded).__name__ == "RenamedParamAE"
        assert loaded._my_mult == 3.0
        assert torch.allclose(loaded.encode(x), z_orig, atol=1e-6)

    def test_custom_subclass_with_prefix_attr_pattern(self, tmp_path):
        """A subclass storing params as self._<param> (SynthAE pattern)
        should auto-resolve via the '_' prefix fallback in default get_config."""

        class PrefixParamAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, temperature=1.0, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self._temperature = temperature
                self.W = nn.Parameter(torch.randn(n_hidden, n_features))
                self.b = nn.Parameter(torch.zeros(n_features))

            def encode(self, x):
                return x @ self.W.T * self._temperature

            def decode(self, z):
                return torch.relu(z @ self.W + self.b)

            def resample_weights(self):
                pass

        ae = PrefixParamAE(N_FEATURES, N_HIDDEN, temperature=0.5)
        config = ae.get_config()
        assert config["temperature"] == 0.5

        x = _random_input()
        z_orig = ae.encode(x)

        path = ae.save_weights(tmp_path / "prefix.safetensors")
        loaded = AutoEncoderBase.from_local(path)
        assert loaded._temperature == 0.5
        assert torch.allclose(loaded.encode(x), z_orig, atol=1e-6)


# ============================================================================
# save_weights metadata
# ============================================================================


class TestSaveWeightsMetadata:
    """Verify the metadata stored by save_weights."""

    def test_safetensors_metadata_contains_class(self, tmp_path):
        """The safetensors file must store the class name in metadata."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
        assert metadata["class"] == "TiedLinearRelu"

    def test_safetensors_metadata_contains_config(self, tmp_path):
        """The safetensors file must store the config as JSON in metadata."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
        config = json.loads(metadata["config"])
        assert config["n_features"] == N_FEATURES
        assert config["n_hidden"] == N_HIDDEN

    def test_json_sidecar_includes_config(self, tmp_path):
        """save_weights must produce a .json sidecar with the config."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        json_path = path.with_suffix(".json")
        assert json_path.exists()

        info = json.loads(json_path.read_text())
        assert "config" in info
        assert info["config"]["n_features"] == N_FEATURES

    def test_json_sidecar_includes_class(self, tmp_path):
        """The JSON sidecar must include the class name."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        json_path = path.with_suffix(".json")
        info = json.loads(json_path.read_text())
        assert info["class"] == "TiedLinearRelu"

    def test_json_sidecar_includes_parameters(self, tmp_path):
        """The JSON sidecar must include parameter shapes and dtypes."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        json_path = path.with_suffix(".json")
        info = json.loads(json_path.read_text())
        assert "parameters" in info
        assert "W" in info["parameters"]
        assert info["parameters"]["W"]["shape"] == [N_HIDDEN, N_FEATURES]

    def test_json_sidecar_includes_total_params(self, tmp_path):
        """The JSON sidecar must include the total parameter count."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        json_path = path.with_suffix(".json")
        info = json.loads(json_path.read_text())
        expected = sum(p.numel() for p in ae.parameters())
        assert info["total_params"] == expected

    def test_json_sidecar_includes_attributes(self, tmp_path):
        """The JSON sidecar must include instance attributes (non-parameter)."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        json_path = path.with_suffix(".json")
        info = json.loads(json_path.read_text())
        assert "attributes" in info
        assert info["attributes"]["n_features"] == N_FEATURES

    def test_save_weights_auto_appends_extension(self, tmp_path):
        """save_weights must auto-append .safetensors if the path lacks it."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model")
        assert path.suffix == ".safetensors"
        assert path.exists()

    def test_save_weights_default_path(self):
        """save_weights with path=None generates a timestamped filename."""
        ae = make_tied_linear_relu()
        path = ae.save_weights()
        try:
            assert path.exists()
            assert "TiedLinearRelu" in path.name
            assert path.suffix == ".safetensors"
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_every_class_config_stored_in_metadata(self, factory, tmp_path):
        """Every built-in AE class must embed its config in safetensors metadata."""
        ae = factory()
        path = ae.save_weights(tmp_path / "model.safetensors")
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
        assert "class" in metadata
        assert "config" in metadata
        config = json.loads(metadata["config"])
        assert isinstance(config, dict)
        assert len(config) > 0


# ============================================================================
# from_hub (mocked)
# ============================================================================


class TestFromHub:
    """Verify from_hub delegates to from_local via mocked hf_hub_download."""

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_from_hub_delegates_to_from_local(
        self, mock_download, mock_hf_api_cls, tmp_path
    ):
        """from_hub must download the file and pass it to from_local."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")

        # Mock hf_hub_download to return our local path
        mock_download.return_value = str(path)
        # Mock model_info to return an object with a sha attribute
        mock_api = MagicMock()
        mock_api.model_info.return_value.sha = "abc123"
        mock_hf_api_cls.return_value = mock_api

        loaded = AutoEncoderBase.from_hub("user/repo")
        assert type(loaded) is TiedLinearRelu

        x = _random_input()
        assert torch.allclose(ae.encode(x), loaded.encode(x), atol=1e-6)

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_from_hub_passes_revision(self, mock_download, mock_hf_api_cls, tmp_path):
        """from_hub must forward revision to hf_hub_download."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        mock_download.return_value = str(path)
        mock_api = MagicMock()
        mock_api.model_info.return_value.sha = "rev123"
        mock_hf_api_cls.return_value = mock_api

        AutoEncoderBase.from_hub("user/repo", revision="v1.0")

        mock_api.model_info.assert_called_once_with("user/repo", revision="v1.0")
        mock_download.assert_called_once_with(
            repo_id="user/repo",
            filename="model.safetensors",
            revision="rev123",
            repo_type="model",
        )

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_from_hub_passes_custom_filename(
        self, mock_download, mock_hf_api_cls, tmp_path
    ):
        """from_hub must forward a custom filename."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "custom.safetensors")
        mock_download.return_value = str(path)
        mock_api = MagicMock()
        mock_api.model_info.return_value.sha = "sha456"
        mock_hf_api_cls.return_value = mock_api

        AutoEncoderBase.from_hub("user/repo", filename="custom.safetensors")

        mock_download.assert_called_once_with(
            repo_id="user/repo",
            filename="custom.safetensors",
            revision="sha456",
            repo_type="model",
        )

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_from_hub_passes_device(self, mock_download, mock_hf_api_cls, tmp_path):
        """from_hub must forward the device parameter."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        mock_download.return_value = str(path)
        mock_api = MagicMock()
        mock_api.model_info.return_value.sha = "abc"
        mock_hf_api_cls.return_value = mock_api

        loaded = AutoEncoderBase.from_hub("user/repo", device="cpu")
        for p in loaded.parameters():
            assert p.device == torch.device("cpu")


# ============================================================================
# push_to_hub (mocked)
# ============================================================================


class TestPushToHub:
    """Verify push_to_hub saves weights and uploads both files."""

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_creates_repo(self, mock_hf_api_cls):
        """push_to_hub must call create_repo with the right args."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        ae.push_to_hub("user/my-model", private=True, token="tok123")

        mock_hf_api_cls.assert_called_once_with(token="tok123")
        mock_api.create_repo.assert_called_once_with(
            "user/my-model", private=True, exist_ok=True
        )

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_uploads_safetensors_and_json(self, mock_hf_api_cls):
        """push_to_hub must upload both the .safetensors and .json files."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        ae.push_to_hub("user/my-model")

        assert mock_api.upload_file.call_count == 2
        upload_calls = mock_api.upload_file.call_args_list

        # First call: safetensors
        first_call = upload_calls[0]
        assert first_call.kwargs["path_in_repo"] == "model.safetensors"
        assert first_call.kwargs["repo_id"] == "user/my-model"

        # Second call: json
        second_call = upload_calls[1]
        assert second_call.kwargs["path_in_repo"] == "model.json"
        assert second_call.kwargs["repo_id"] == "user/my-model"

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_returns_url(self, mock_hf_api_cls):
        """push_to_hub must return the HuggingFace URL."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        url = ae.push_to_hub("user/my-model")
        assert url == "https://huggingface.co/user/my-model"

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_custom_filename(self, mock_hf_api_cls):
        """push_to_hub must use the provided filename for upload."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        ae.push_to_hub("user/my-model", filename="best.safetensors")

        upload_calls = mock_api.upload_file.call_args_list
        assert upload_calls[0].kwargs["path_in_repo"] == "best.safetensors"
        assert upload_calls[1].kwargs["path_in_repo"] == "best.json"

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_auto_commit_message(self, mock_hf_api_cls):
        """push_to_hub with no commit_message must auto-generate one."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        ae.push_to_hub("user/my-model")

        call_kwargs = mock_api.upload_file.call_args_list[0].kwargs
        msg = call_kwargs["commit_message"]
        assert "TiedLinearRelu" in msg
        assert f"{N_FEATURES}x{N_HIDDEN}" in msg

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_custom_commit_message(self, mock_hf_api_cls):
        """push_to_hub with a custom commit_message must use it."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        ae.push_to_hub("user/my-model", commit_message="My custom message")

        call_kwargs = mock_api.upload_file.call_args_list[0].kwargs
        assert call_kwargs["commit_message"] == "My custom message"


# ============================================================================
# ToyModel integration (mocked)
# ============================================================================


class TestToyModelIntegration:
    """ToyModel.from_hub and push_to_hub delegate correctly."""

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_toymodel_from_hub_creates_correct_model(
        self, mock_download, mock_hf_api_cls, tmp_path
    ):
        """ToyModel.from_hub must load the AE and wrap it with the distribution."""
        from occhio.distributions.sparse import SparseUniform
        from occhio.toy_model import ToyModel

        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")
        mock_download.return_value = str(path)
        mock_api = MagicMock()
        mock_api.model_info.return_value.sha = "abc"
        mock_hf_api_cls.return_value = mock_api

        dist = SparseUniform(N_FEATURES, p_active=0.5, device=DEVICE)
        tm = ToyModel.from_hub("user/repo", distribution=dist, device="cpu")

        assert isinstance(tm, ToyModel)
        assert isinstance(tm.ae, TiedLinearRelu)
        assert tm.ae.n_features == N_FEATURES
        assert tm.ae.n_hidden == N_HIDDEN

    @patch("huggingface_hub.HfApi")
    def test_toymodel_push_to_hub_delegates_to_ae(self, mock_hf_api_cls):
        """ToyModel.push_to_hub must delegate to ae.push_to_hub."""
        from occhio.distributions.sparse import SparseUniform
        from occhio.toy_model import ToyModel

        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = None

        ae = make_tied_linear_relu()
        dist = SparseUniform(N_FEATURES, p_active=0.5, device=DEVICE)
        tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

        url = tm.push_to_hub("user/tm-model", private=True, token="tok")

        assert url == "https://huggingface.co/user/tm-model"
        mock_hf_api_cls.assert_called_once_with(token="tok")
        mock_api.create_repo.assert_called_once_with(
            "user/tm-model", private=True, exist_ok=True
        )
        assert mock_api.upload_file.call_count == 2


# ============================================================================
# load_weights instance method
# ============================================================================


class TestLoadWeights:
    """Tests for the instance-level load_weights method."""

    def test_load_weights_class_mismatch_raises(self, tmp_path):
        """Loading weights saved from a different class must raise TypeError."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")

        other = make_tied_linear()
        with pytest.raises(TypeError, match="TiedLinearRelu.*TiedLinear"):
            other.load_weights(path)

    def test_load_weights_missing_class_metadata_raises(self, tmp_path):
        """A file without 'class' metadata must raise ValueError."""
        path = tmp_path / "no_meta.safetensors"
        save_file({"W": torch.randn(4, 8)}, str(path))

        ae = make_tied_linear_relu()
        with pytest.raises(ValueError, match="no 'class' metadata"):
            ae.load_weights(path)

    def test_load_weights_file_not_found_raises(self, tmp_path):
        """Nonexistent file must raise FileNotFoundError."""
        ae = make_tied_linear_relu()
        with pytest.raises(FileNotFoundError):
            ae.load_weights(tmp_path / "nonexistent.safetensors")

    def test_load_weights_auto_appends_extension(self, tmp_path):
        """load_weights must auto-append .safetensors if missing."""
        ae = make_tied_linear_relu()
        path = ae.save_weights(tmp_path / "model.safetensors")

        ae2 = make_tied_linear_relu(seed=99)
        ae2.load_weights(tmp_path / "model")  # no extension
        for key in ae.state_dict():
            assert torch.equal(ae.state_dict()[key], ae2.state_dict()[key])


# ============================================================================
# __init_subclass__ validation
# ============================================================================


class TestInitSubclass:
    """Verify __init_subclass__ enforces n_features/n_hidden."""

    def test_subclass_missing_n_features_raises(self):
        """A subclass whose __init__ does not set n_features must raise
        AttributeError at construction time."""

        class BadAE(AutoEncoderBase):
            def __init__(self):
                nn.Module.__init__(self)
                # Deliberately skip super().__init__ and don't set n_features
                self.n_hidden = 4

            def encode(self, x):
                return x

            def decode(self, z):
                return z

            def resample_weights(self):
                pass

        with pytest.raises(AttributeError, match="n_features"):
            BadAE()

    def test_subclass_missing_n_hidden_raises(self):
        """A subclass whose __init__ does not set n_hidden must raise
        AttributeError at construction time."""

        class BadAE2(AutoEncoderBase):
            def __init__(self):
                nn.Module.__init__(self)
                self.n_features = 8

            def encode(self, x):
                return x

            def decode(self, z):
                return z

            def resample_weights(self):
                pass

        with pytest.raises(AttributeError, match="n_hidden"):
            BadAE2()


# ============================================================================
# _serialize_value edge cases
# ============================================================================


class TestSerializeValue:
    """Test the static _serialize_value helper for edge cases."""

    def test_none(self):
        assert AutoEncoderBase._serialize_value(None) is None

    def test_int(self):
        assert AutoEncoderBase._serialize_value(42) == 42

    def test_float(self):
        assert AutoEncoderBase._serialize_value(3.14) == 3.14

    def test_string(self):
        assert AutoEncoderBase._serialize_value("hello") == "hello"

    def test_bool(self):
        assert AutoEncoderBase._serialize_value(True) is True

    def test_torch_device(self):
        result = AutoEncoderBase._serialize_value(torch.device("cpu"))
        assert result == "cpu"

    def test_tensor_returns_shape_dtype(self):
        t = torch.randn(3, 4)
        result = AutoEncoderBase._serialize_value(t)
        assert result == {"shape": [3, 4], "dtype": "torch.float32"}

    def test_parameter_returns_skip(self):
        from occhio.autoencoders.base import _SKIP

        p = nn.Parameter(torch.randn(3))
        assert AutoEncoderBase._serialize_value(p) is _SKIP

    def test_list_of_ints(self):
        assert AutoEncoderBase._serialize_value([1, 2, 3]) == [1, 2, 3]

    def test_tuple_of_ints(self):
        assert AutoEncoderBase._serialize_value((1, 2)) == [1, 2]

    def test_dict_of_primitives(self):
        result = AutoEncoderBase._serialize_value({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    def test_callable_uses_repr(self):
        """Non-serializable types fall back to repr."""

        def my_fn():
            pass

        result = AutoEncoderBase._serialize_value(my_fn)
        assert isinstance(result, str)
        assert "my_fn" in result

    def test_generator_returns_dict(self):
        gen = torch.Generator(device="cpu").manual_seed(42)
        result = AutoEncoderBase._serialize_value(gen)
        assert result["type"] == "Generator"
        assert result["device"] == "cpu"
        assert result["initial_seed"] == 42


# ============================================================================
# Double round-trip (idempotency)
# ============================================================================


class TestDoubleRoundTrip:
    """Save -> load -> save -> load must produce identical weights."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_double_round_trip_identical(self, factory, tmp_path):
        """Two consecutive save/load cycles must produce bit-identical weights.
        Catches any non-determinism in serialization."""
        ae = factory()
        x = _random_input()

        path1 = ae.save_weights(tmp_path / "round1.safetensors")
        loaded1 = AutoEncoderBase.from_local(path1)
        path2 = loaded1.save_weights(tmp_path / "round2.safetensors")
        loaded2 = AutoEncoderBase.from_local(path2)

        z1 = loaded1.encode(x)
        z2 = loaded2.encode(x)
        assert torch.equal(z1, z2)
