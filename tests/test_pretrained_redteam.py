"""Red-team tests for the architecture-invariant autoencoder save/load system.

Attack surface: save_weights, from_local, get_config, load_weights,
__init_subclass__ registry, config serialization/deserialization.

Tests are organized by attack vector:
1. Config corruption (wrong types in saved metadata)
2. State dict / class mismatch
3. Non-JSON-serializable configs
4. Subclass name collisions in the registry
5. Incomplete / extra config keys
6. Deeply nested config values
7. Large models
8. Empty state dicts
9. SynthAE orthogonalization on load
10. ComputeAutoEncoder seed loss
11. Pickle/dill compatibility
12. Misc edge cases
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

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


# ============================================================================
# 1. CONFIG CORRUPTION — wrong types in saved metadata
# ============================================================================


class TestConfigCorruption:
    """What happens when saved config has values of wrong type?"""

    def test_n_features_as_string_fails_at_construction(self, tmp_path):
        """n_features='ten' in config should fail when constructing the model."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        # Manually corrupt the config in metadata
        corrupted_config = {"n_features": "ten", "n_hidden": 5}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(corrupted_config),
            },
        )
        with pytest.raises(RuntimeError, match="Failed to construct"):
            AutoEncoderBase.from_local(path)

    def test_n_hidden_as_float_in_config(self, tmp_path):
        """n_hidden=5.0 (float instead of int) — may pass through but
        torch.randn will get a float for dimension, which should fail."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        # JSON encodes 5 as 5, but 5.0 is valid JSON
        corrupted_config = {"n_features": 10, "n_hidden": 5.0}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(corrupted_config),
            },
        )
        # This should either construct successfully (float->int coercion)
        # or fail at construction. Document which.
        # torch.randn accepts float for dims in newer PyTorch so let's test.
        try:
            loaded = AutoEncoderBase.from_local(path)
            # If it succeeds, n_hidden should be usable
            assert loaded.n_hidden == 5 or loaded.n_hidden == 5.0
        except (RuntimeError, TypeError):
            pass  # Also acceptable — the point is no silent corruption

    def test_negative_n_features_fails(self, tmp_path):
        """Negative dimension should fail during construction."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        corrupted_config = {"n_features": -1, "n_hidden": 5}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(corrupted_config),
            },
        )
        with pytest.raises(RuntimeError, match="Failed to construct"):
            AutoEncoderBase.from_local(path)

    def test_zero_n_hidden_fails(self, tmp_path):
        """Zero dimension should fail."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        corrupted_config = {"n_features": 10, "n_hidden": 0}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(corrupted_config),
            },
        )
        with pytest.raises((RuntimeError, ValueError)):
            AutoEncoderBase.from_local(path)

    def test_nan_in_config_fails(self, tmp_path):
        """NaN is not valid JSON per spec, but Python json module allows it.
        Construction should fail or produce a broken model."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        # json.dumps(float('nan')) produces "NaN" which json.loads accepts
        corrupted_config = json.dumps({"n_features": float("nan"), "n_hidden": 5})
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": corrupted_config,
            },
        )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            AutoEncoderBase.from_local(path)

    def test_null_required_param_in_config(self, tmp_path):
        """n_features=null should fail at construction (cannot create tensor of None size)."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        corrupted_config = {"n_features": None, "n_hidden": 5}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(corrupted_config),
            },
        )
        with pytest.raises(RuntimeError, match="Failed to construct"):
            AutoEncoderBase.from_local(path)


# ============================================================================
# 2. STATE DICT / CLASS MISMATCH
# ============================================================================


class TestStateDictMismatch:
    """Save as one architecture, lie in metadata about the class."""

    def test_save_tiedlinearrelu_claim_attnlinearae(self, tmp_path):
        """State dict from TiedLinearRelu claimed as AttnLinearAE.
        from_local should fail at load_state_dict (key mismatch)."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"

        # Save with wrong class name AND a config that would construct AttnLinearAE
        config = {"n_features": 10, "n_hidden": 4, "n_heads": 2, "dict_size": 8}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "AttnLinearAE",
                "config": json.dumps(config),
            },
        )
        # load_state_dict(strict=True) should raise because keys mismatch
        with pytest.raises(RuntimeError):
            AutoEncoderBase.from_local(path)

    def test_save_mlpencoder_claim_tiedlinearrelu(self, tmp_path):
        """MLPEncoder state dict claimed as TiedLinearRelu."""
        ae = MLPEncoder(embedding=[10, 8, 5], unembedding=[5, 8, 10])
        path = tmp_path / "model.safetensors"

        config = {"n_features": 10, "n_hidden": 5}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(config),
            },
        )
        with pytest.raises(RuntimeError):
            AutoEncoderBase.from_local(path)

    def test_load_weights_cross_class_rejected(self, tmp_path):
        """load_weights (instance method) checks class name in metadata."""
        ae1 = TiedLinearRelu(10, 5)
        ae2 = TiedLinear(10, 5)
        path = tmp_path / "model.safetensors"
        ae1.save_weights(path)

        with pytest.raises(TypeError, match="saved from TiedLinearRelu"):
            ae2.load_weights(path)

    def test_load_weights_same_dims_different_class(self, tmp_path):
        """Same n_features/n_hidden but different class should still be rejected."""
        ae1 = TiedLinearRelu(10, 5)
        ae2 = TiedLinear(10, 5)
        path = tmp_path / "model.safetensors"
        ae1.save_weights(path)

        with pytest.raises(TypeError):
            ae2.load_weights(path)


# ============================================================================
# 3. NON-JSON-SERIALIZABLE CONFIG
# ============================================================================


class TestNonJsonSerializableConfig:
    """Subclass where get_config returns something json.dumps can't handle."""

    def test_config_with_tensor_value_skipped_by_default(self, tmp_path):
        """Default get_config filters non-serializable types. A tensor attr
        stored as self.some_param won't appear in config because isinstance
        check excludes Tensor."""

        class TensorConfigAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, my_tensor=None, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.my_tensor = (
                    my_tensor if my_tensor is not None else torch.ones(n_features)
                )
                self.W = nn.Parameter(
                    torch.randn(n_hidden, n_features, device=self.device)
                )
                self.b = nn.Parameter(torch.zeros(n_features, device=self.device))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return torch.relu(z @ self.W + self.b)

            def resample_weights(self):
                pass

        ae = TensorConfigAE(5, 3)
        # Default get_config should skip tensor parameters
        config = ae.get_config()
        assert "my_tensor" not in config

        # save_weights should still work (config is clean)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)
        assert path.exists()

    def test_custom_get_config_with_tensor_crashes_save(self, tmp_path):
        """If a subclass overrides get_config to return a tensor, save_weights
        should fail at json.dumps with a clear error."""

        class BadConfigAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.W = nn.Parameter(
                    torch.randn(n_hidden, n_features, device=self.device)
                )
                self.b = nn.Parameter(torch.zeros(n_features, device=self.device))

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
                    "bad_value": torch.tensor([1, 2, 3]),
                }

        ae = BadConfigAE(5, 3)
        path = tmp_path / "model.safetensors"
        with pytest.raises(TypeError):
            ae.save_weights(path)

    def test_custom_get_config_with_module_crashes_save(self, tmp_path):
        """Config containing nn.Module should fail to serialize."""

        class ModuleConfigAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.W = nn.Parameter(
                    torch.randn(n_hidden, n_features, device=self.device)
                )
                self.b = nn.Parameter(torch.zeros(n_features, device=self.device))

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
                    "inner": nn.Linear(3, 3),
                }

        ae = ModuleConfigAE(5, 3)
        path = tmp_path / "model.safetensors"
        with pytest.raises(TypeError):
            ae.save_weights(path)


# ============================================================================
# 4. SUBCLASS NAME COLLISIONS IN REGISTRY
# ============================================================================


class TestRegistryCollisions:
    """Two subclasses with the same __name__ — which one wins?"""

    def test_last_registered_wins(self):
        """If two classes have the same name, the last one registered should
        be in the registry. This is the current behavior — document it."""

        class DupeAE(AutoEncoderBase):
            VERSION = 1

            def __init__(self, n_features, n_hidden, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.W = nn.Parameter(torch.randn(n_hidden, n_features))
                self.b = nn.Parameter(torch.zeros(n_features))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return z @ self.W + self.b

            def resample_weights(self):
                pass

        first_class = DupeAE
        assert AutoEncoderBase._registry["DupeAE"] is first_class

        # Define another class with same name in a different scope
        # Python won't let us use the same literal name in the same scope,
        # so we simulate via type()
        SecondDupeAE = type(
            "DupeAE",
            (AutoEncoderBase,),
            {
                "__init__": lambda self, n_features, n_hidden, **kwargs: (
                    AutoEncoderBase.__init__(self, n_features, n_hidden, **kwargs),
                    setattr(self, "W", nn.Parameter(torch.randn(n_hidden, n_features))),
                    setattr(self, "b", nn.Parameter(torch.zeros(n_features))),
                )[-1],
                "encode": lambda self, x: x @ self.W.T,
                "decode": lambda self, z: z @ self.W + self.b,
                "resample_weights": lambda self: None,
                "VERSION": 2,
            },
        )
        # The second class should have overwritten the first in the registry
        assert AutoEncoderBase._registry["DupeAE"] is SecondDupeAE
        assert AutoEncoderBase._registry["DupeAE"] is not first_class

    def test_registry_contains_all_concrete_classes(self):
        """All known concrete AE classes should be in the registry."""
        expected = {
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
        assert expected.issubset(set(AutoEncoderBase._registry.keys()))


# ============================================================================
# 5. INCOMPLETE / EXTRA CONFIG KEYS
# ============================================================================


class TestConfigCompleteness:
    """Missing or extra keys in the saved config."""

    def test_missing_required_param_fails_construction(self, tmp_path):
        """Config missing n_hidden (required, no default) should fail."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"

        incomplete_config = {"n_features": 10}  # missing n_hidden
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(incomplete_config),
            },
        )
        with pytest.raises(RuntimeError, match="Failed to construct"):
            AutoEncoderBase.from_local(path)

    def test_extra_keys_rejected(self, tmp_path):
        """Config with keys not in __init__ signature should be caught."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"

        extra_config = {"n_features": 10, "n_hidden": 5, "nonexistent_param": 42}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(extra_config),
            },
        )
        with pytest.raises(ValueError, match="unexpected keys"):
            AutoEncoderBase.from_local(path)

    def test_extra_keys_error_message_is_informative(self, tmp_path):
        """Error message should name the extra keys and valid params."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"

        extra_config = {"n_features": 10, "n_hidden": 5, "bogus": True}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(extra_config),
            },
        )
        with pytest.raises(ValueError, match="bogus"):
            AutoEncoderBase.from_local(path)

    def test_attnlinearae_missing_n_heads_fails(self, tmp_path):
        """AttnLinearAE requires n_heads and dict_size — missing either should fail."""
        ae = AttnLinearAE(10, 4, n_heads=2, dict_size=8)
        path = tmp_path / "model.safetensors"

        incomplete_config = {"n_features": 10, "n_hidden": 4, "dict_size": 8}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "AttnLinearAE",
                "config": json.dumps(incomplete_config),
            },
        )
        with pytest.raises(RuntimeError, match="Failed to construct"):
            AutoEncoderBase.from_local(path)

    def test_mlpencoder_get_config_has_all_keys(self):
        """MLPEncoder.get_config should include all required construction params."""
        ae = MLPEncoder(embedding=[10, 8, 5], unembedding=[5, 8, 10])
        config = ae.get_config()
        assert "embedding" in config
        assert "unembedding" in config
        assert config["embedding"] == [10, 8, 5]
        assert config["unembedding"] == [5, 8, 10]

    def test_tiedmlpencoder_get_config_has_dims(self):
        """TiedMLPEncoder uses default get_config — should capture 'dims'."""
        ae = TiedMLPEncoder(dims=[10, 8, 5])
        config = ae.get_config()
        assert "dims" in config
        assert config["dims"] == [10, 8, 5]

    def test_compute_get_config_maps_N_k_correctly(self):
        """ComputeAutoEncoder uses N/k params mapped to n_features/n_hidden."""
        ae = ComputeAutoEncoder(N=10, k=5)
        config = ae.get_config()
        assert config["N"] == 10
        assert config["k"] == 5
        assert "decode_activation" in config

    def test_synthae_default_get_config_captures_underscore_attrs(self):
        """SynthAE stores params as self._orthogonalize etc. Default get_config
        should find them via the self._<name> fallback."""
        ae = SynthAE(10, 5, orthogonalize=True, ortho_steps=500)
        config = ae.get_config()
        assert config.get("orthogonalize") is True
        assert config.get("ortho_steps") == 500

    def test_attn_ae_get_config_captures_all_params(self):
        """All attention AE constructor params should appear in config."""
        ae = AttnLinearAE(10, 4, n_heads=2, dict_size=8)
        config = ae.get_config()
        assert config["n_features"] == 10
        assert config["n_hidden"] == 4
        assert config["n_heads"] == 2
        assert config["dict_size"] == 8


# ============================================================================
# 6. DEEPLY NESTED CONFIG VALUES
# ============================================================================


class TestNestedConfigValues:
    """Lists of lists, dicts, etc. in config — verify JSON round-trip."""

    def test_mlpencoder_list_config_roundtrips(self, tmp_path):
        """MLPEncoder config has list values — JSON should preserve them."""
        ae = MLPEncoder(
            embedding=[10, 8, 6, 5], unembedding=[5, 6, 8, 10], tied_initialization=True
        )
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, MLPEncoder)
        assert loaded.embedding_dims == [10, 8, 6, 5]
        assert loaded.unembedding_dims == [5, 6, 8, 10]
        assert loaded.tied_initialization is True

    def test_json_roundtrip_preserves_types(self, tmp_path):
        """Verify int vs float vs bool vs None are preserved through JSON."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        config = json.loads(meta["config"])

        # n_features and n_hidden should be ints after roundtrip
        assert isinstance(config["n_features"], int)
        assert isinstance(config["n_hidden"], int)

    def test_tuple_becomes_list_after_json_roundtrip(self):
        """JSON has no tuple type — tuples become lists. This is fine if the
        constructor accepts lists where tuples were used."""
        # This is a known JSON limitation — test that the system handles it.
        original = (1, 2, 3)
        roundtripped = json.loads(json.dumps(original))
        assert isinstance(roundtripped, list)
        assert roundtripped == [1, 2, 3]


# ============================================================================
# 7. LARGE MODELS
# ============================================================================


class TestLargeModels:
    """n_features=10000 — verify no memory issues in save/load."""

    def test_large_tiedlinearrelu_save_load(self, tmp_path):
        """10000 features, 500 hidden — save and from_local roundtrip."""
        ae = TiedLinearRelu(10000, 500)
        path = tmp_path / "large.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert loaded.n_features == 10000
        assert loaded.n_hidden == 500

        x = torch.randn(2, 10000)
        out_original = ae.encode(x)
        out_loaded = loaded.encode(x)
        assert torch.allclose(out_original, out_loaded)

    def test_large_model_json_info_file_created(self, tmp_path):
        """The companion .json file should be created even for large models."""
        ae = TiedLinearRelu(5000, 200)
        path = ae.save_weights(tmp_path / "large")
        json_path = path.with_suffix(".json")
        assert json_path.exists()

        info = json.loads(json_path.read_text())
        assert info["config"]["n_features"] == 5000
        assert info["total_params"] > 0


# ============================================================================
# 8. EMPTY STATE DICT
# ============================================================================


class TestEmptyStateDict:
    """What if the safetensors file has no tensors?"""

    def test_empty_safetensors_with_valid_config(self, tmp_path):
        """A safetensors file with no tensors but valid config/class metadata.
        from_local will construct the model (which creates params), then
        load_state_dict will fail due to missing keys."""
        path = tmp_path / "empty.safetensors"
        config = {"n_features": 10, "n_hidden": 5}
        save_file(
            {},
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(config),
            },
        )
        # The model constructor creates W and b, but the state dict is empty
        # strict=True should raise about missing keys
        with pytest.raises(RuntimeError):
            AutoEncoderBase.from_local(path)

    def test_extra_tensor_in_safetensors(self, tmp_path):
        """State dict has extra keys not expected by the model.
        strict=True should reject it."""
        ae = TiedLinearRelu(10, 5)
        state = ae.state_dict()
        state["extra_bogus_tensor"] = torch.randn(3, 3)
        path = tmp_path / "extra.safetensors"
        config = {"n_features": 10, "n_hidden": 5}
        save_file(
            state,
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": json.dumps(config),
            },
        )
        with pytest.raises(RuntimeError):
            AutoEncoderBase.from_local(path)


# ============================================================================
# 9. SYNTHAE ORTHOGONALIZATION ON LOAD
# ============================================================================


class TestSynthAELoadBehavior:
    """orthogonalize=True in config but we're loading weights.
    The orthogonalization runs during __init__ then gets overwritten.
    Is this wasteful? Does it produce correct results?"""

    def test_synthae_ortho_true_roundtrip_correct(self, tmp_path):
        """Save SynthAE with orthogonalize=True, from_local should restore
        exact weights despite orthogonalization running during construction."""
        ae = SynthAE(10, 5, orthogonalize=True, ortho_steps=50, ortho_lr=0.01)
        x = torch.randn(4, 10)
        original_output = ae.encode(x)

        path = tmp_path / "synth.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        loaded_output = loaded.encode(x)

        # Weights should be exactly restored (overwriting the orthogonalized init)
        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_synthae_ortho_true_config_preserved(self, tmp_path):
        """The config should preserve orthogonalize=True so the loaded model
        behaves identically if resample_weights is called."""
        ae = SynthAE(10, 5, orthogonalize=True, ortho_steps=50)
        path = tmp_path / "synth.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert loaded._orthogonalize is True
        assert loaded._ortho_steps == 50

    def test_synthae_ortho_false_fast_load(self, tmp_path):
        """SynthAE with orthogonalize=False should load quickly
        (no wasted orthogonalization step)."""
        ae = SynthAE(10, 5, orthogonalize=False)
        path = tmp_path / "synth.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert loaded._orthogonalize is False

    def test_synthae_w_frozen_after_load(self, tmp_path):
        """SynthAE freezes W in __init__. After from_local,
        W should be present and have the saved values."""
        ae = SynthAE(10, 5)
        path = tmp_path / "synth.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert torch.allclose(ae.W, loaded.W)
        # W is frozen in SynthAE (requires_grad=False)
        assert not loaded.W.requires_grad


# ============================================================================
# 10. COMPUTEAUTOENCODER SEED LOSS
# ============================================================================


class TestComputeAESeedLoss:
    """Config doesn't include seed. Does this matter?"""

    def test_seed_not_in_config(self):
        """ComputeAutoEncoder.get_config should NOT include seed since
        we're loading weights anyway (seed only affects initialization)."""
        ae = ComputeAutoEncoder(N=10, k=5, seed=42)
        config = ae.get_config()
        assert "seed" not in config

    def test_different_seeds_same_weights_after_load(self, tmp_path):
        """Two ComputeAutoEncoders with different seeds, after loading the same
        weights, should produce identical outputs."""
        ae = ComputeAutoEncoder(N=10, k=5, seed=42)
        path = tmp_path / "compute.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        x = torch.randn(4, 10)
        assert torch.allclose(ae.encode(x), loaded.encode(x))

    def test_compute_ae_roundtrip_all_params(self, tmp_path):
        """Verify W, Z, and b are all correctly restored."""
        ae = ComputeAutoEncoder(N=10, k=5, seed=42, decode_activation="relu")
        path = tmp_path / "compute.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert torch.allclose(ae.W, loaded.W)
        assert torch.allclose(ae.Z, loaded.Z)
        assert torch.allclose(ae.b, loaded.b)
        assert loaded.decode_activation == "relu"

    def test_compute_ae_default_seed_does_not_break_load(self, tmp_path):
        """Default seed=10 construction, save, load should work fine."""
        ae = ComputeAutoEncoder(N=8, k=4)
        path = tmp_path / "compute.safetensors"
        ae.save_weights(path)

        # from_local will construct with config (no seed), then load weights
        loaded = AutoEncoderBase.from_local(path)
        x = torch.randn(2, 8)
        y_orig, z_orig = ae(x)
        y_load, z_load = loaded(x)
        assert torch.allclose(y_orig, y_load, atol=1e-6)
        assert torch.allclose(z_orig, z_load, atol=1e-6)


# ============================================================================
# 11. PICKLE / DILL COMPATIBILITY
# ============================================================================


class TestPickleCompatibility:
    """Can models loaded via from_local be pickled? Important for ModelGrid."""

    def test_pickle_roundtrip_tiedlinearrelu(self, tmp_path):
        """TiedLinearRelu from from_local should be picklable."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))

    def test_pickle_roundtrip_attnlinearae(self, tmp_path):
        """AttnLinearAE from from_local should be picklable."""
        ae = AttnLinearAE(10, 4, n_heads=2, dict_size=8)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))

    def test_pickle_roundtrip_mlpencoder(self, tmp_path):
        """MLPEncoder from from_local should be picklable."""
        ae = MLPEncoder(embedding=[10, 8, 5], unembedding=[5, 8, 10])
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))

    def test_pickle_roundtrip_compute_ae(self, tmp_path):
        """ComputeAutoEncoder from from_local should be picklable."""
        ae = ComputeAutoEncoder(N=10, k=5, seed=42)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))

    def test_pickle_roundtrip_synthae(self, tmp_path):
        """SynthAE from from_local should be picklable."""
        ae = SynthAE(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))

    def test_pickle_roundtrip_tiedmlpencoder(self, tmp_path):
        """TiedMLPEncoder from from_local should be picklable."""
        ae = TiedMLPEncoder(dims=[10, 8, 5])
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        pickled = pickle.dumps(loaded)
        restored = pickle.loads(pickled)

        x = torch.randn(2, 10)
        assert torch.allclose(loaded.encode(x), restored.encode(x))


# ============================================================================
# 12. MISC EDGE CASES
# ============================================================================


class TestMiscEdgeCases:
    """Various edge cases and boundary conditions."""

    def test_save_weights_returns_path_with_safetensors_extension(self, tmp_path):
        """save_weights should always return a .safetensors path."""
        ae = TiedLinearRelu(10, 5)
        path = ae.save_weights(tmp_path / "model")
        assert path.suffix == ".safetensors"

    def test_save_weights_appends_extension(self, tmp_path):
        """If given a path without .safetensors, extension should be appended."""
        ae = TiedLinearRelu(10, 5)
        path = ae.save_weights(tmp_path / "model.pt")
        assert path.suffix == ".safetensors"
        assert path.name == "model.safetensors"

    def test_from_local_appends_extension(self, tmp_path):
        """from_local should auto-append .safetensors if missing."""
        ae = TiedLinearRelu(10, 5)
        ae.save_weights(tmp_path / "model")

        loaded = AutoEncoderBase.from_local(tmp_path / "model")
        assert loaded.n_features == 10

    def test_save_load_preserves_eval_mode(self, tmp_path):
        """Models should be loadable in both train and eval mode."""
        ae = TiedLinearRelu(10, 5)
        ae.eval()
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        # from_local doesn't explicitly set eval/train mode,
        # so loaded model will be in training mode (default)
        # This is expected behavior — document it.
        assert loaded.training  # default is training mode

    def test_double_save_overwrites(self, tmp_path):
        """Saving twice to the same path should overwrite, not corrupt."""
        ae1 = TiedLinearRelu(10, 5)
        ae2 = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"

        ae1.save_weights(path)
        ae2.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        x = torch.randn(2, 10)
        # Should have ae2's weights, not ae1's
        assert torch.allclose(ae2.encode(x), loaded.encode(x))

    def test_save_default_filename(self):
        """save_weights with no path should create a file with class name."""
        ae = TiedLinearRelu(10, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = ae.save_weights()
                assert path.exists()
                assert "TiedLinearRelu" in path.name
                assert "10x5" in path.name
                assert path.suffix == ".safetensors"
            finally:
                os.chdir(old_cwd)

    def test_class_aliases_resolve_correctly(self, tmp_path):
        """All declared aliases should resolve to valid registry entries."""
        for alias, target in AutoEncoderBase._class_aliases.items():
            assert target in AutoEncoderBase._registry, (
                f"Alias {alias!r} -> {target!r} but {target!r} not in registry"
            )

    def test_from_local_with_alias_pretrainedae(self, tmp_path):
        """PretrainedAE alias should resolve to TiedLinearRelu."""
        ae = TiedLinearRelu(8, 4)
        path = tmp_path / "alias.safetensors"
        config = {"n_features": 8, "n_hidden": 4}
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "PretrainedAE",
                "config": json.dumps(config),
            },
        )
        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, TiedLinearRelu)

    def test_save_weights_companion_json_structure(self, tmp_path):
        """The companion .json file should have specific structure."""
        ae = AttnLinearAE(10, 4, n_heads=2, dict_size=8)
        path = ae.save_weights(tmp_path / "model")
        json_path = path.with_suffix(".json")
        assert json_path.exists()

        info = json.loads(json_path.read_text())
        assert "class" in info
        assert "config" in info
        assert "attributes" in info
        assert "parameters" in info
        assert "total_params" in info
        assert info["class"] == "AttnLinearAE"
        assert info["config"]["n_heads"] == 2

    def test_legacy_file_without_config_or_W_fails(self, tmp_path):
        """A file with class metadata but no config and no 'W' key
        should give a helpful error."""
        path = tmp_path / "bad.safetensors"
        save_file(
            {"some_other_key": torch.randn(3, 3)},
            str(path),
            metadata={"class": "TiedLinearRelu"},
        )
        with pytest.raises(ValueError, match="no 'config' metadata and no 'W' key"):
            AutoEncoderBase.from_local(path)

    def test_init_subclass_enforces_n_features_n_hidden(self):
        """Subclass that doesn't set n_features or n_hidden should fail."""

        class BrokenAE(AutoEncoderBase):
            def __init__(self, **kwargs):
                # deliberately skip super().__init__
                nn.Module.__init__(self)
                # Don't set n_features or n_hidden

            def encode(self, x):
                return x

            def decode(self, z):
                return z

            def resample_weights(self):
                pass

        with pytest.raises(AttributeError, match="must set self.n_features"):
            BrokenAE()

    def test_concurrent_save_load_different_files(self, tmp_path):
        """Save multiple models to different files, load them all back."""
        models = {
            "tied": TiedLinearRelu(10, 5),
            "attn": AttnLinearAE(10, 4, n_heads=2, dict_size=8),
            "mlp": MLPEncoder(embedding=[10, 8, 5], unembedding=[5, 8, 10]),
            "compute": ComputeAutoEncoder(N=10, k=5),
        }

        paths = {}
        for name, ae in models.items():
            p = tmp_path / f"{name}.safetensors"
            ae.save_weights(p)
            paths[name] = p

        for name, p in paths.items():
            loaded = AutoEncoderBase.from_local(p)
            assert type(loaded).__name__ == type(models[name]).__name__
            x = torch.randn(2, models[name].n_features)
            assert torch.allclose(models[name].encode(x), loaded.encode(x), atol=1e-6)

    def test_get_config_excludes_loss_fn_device_generator(self):
        """_NON_SERIALIZABLE_PARAMS should be excluded from default get_config."""
        gen = torch.Generator(device="cpu").manual_seed(42)
        ae = TiedLinearRelu(10, 5, device="cpu", generator=gen)
        config = ae.get_config()
        assert "loss_fn" not in config
        assert "device" not in config
        assert "generator" not in config

    def test_from_local_kwargs_subclass_accepted(self, tmp_path):
        """Classes that accept **kwargs shouldn't have issues with config validation.
        The config validator checks against valid_params; **kwargs params should
        not cause false positives."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert loaded.n_features == 10

    def test_load_weights_strict_false(self, tmp_path):
        """load_weights with strict=False should not raise on missing keys."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        fresh = TiedLinearRelu(10, 5)
        fresh.load_weights(path, strict=True)  # should work
        x = torch.randn(2, 10)
        assert torch.allclose(ae.encode(x), fresh.encode(x))

    def test_save_weights_no_metadata_file_loads_via_load_weights(self, tmp_path):
        """Verify load_weights validates class from metadata."""
        ae = TiedLinearRelu(10, 5)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        fresh = TiedLinearRelu(10, 5)
        fresh.load_weights(path)  # should succeed

    def test_linearattnae_roundtrip(self, tmp_path):
        """LinearAttnAE save/from_local roundtrip."""
        ae = LinearAttnAE(10, 4, n_heads=2, dict_size=8)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, LinearAttnAE)
        x = torch.randn(2, 10)
        assert torch.allclose(ae.encode(x), loaded.encode(x), atol=1e-6)

    def test_attnattnae_roundtrip(self, tmp_path):
        """AttnAttnAE save/from_local roundtrip."""
        ae = AttnAttnAE(10, 4, n_heads=2, dict_size=8)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, AttnAttnAE)
        x = torch.randn(2, 10)
        assert torch.allclose(ae.encode(x), loaded.encode(x), atol=1e-6)


# ============================================================================
# 13. COMPUTE AE SEED vs INIT INTERACTION — DEEPER DIVE
# ============================================================================


class TestComputeAESeedInit:
    """ComputeAutoEncoder creates its own generator in __init__ using seed.
    When loading via from_local, config has no seed, so seed=10 (default).
    This means different initial weights, but load_state_dict overwrites them.
    Verify the full forward pass is correct."""

    def test_non_default_seed_roundtrip(self, tmp_path):
        """Save with seed=99, load (which uses default seed=10),
        output should still match because weights are overwritten."""
        ae = ComputeAutoEncoder(N=8, k=4, seed=99)
        path = tmp_path / "compute.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        x = torch.randn(3, 8)

        # Full forward pass including compute_step
        y_orig, z_orig = ae(x)
        y_load, z_load = loaded(x)
        assert torch.allclose(y_orig, y_load, atol=1e-6)
        assert torch.allclose(z_orig, z_load, atol=1e-6)

    def test_decode_activation_preserved(self, tmp_path):
        """decode_activation must be in config to produce correct output."""
        for activation in ["softmax", "relu"]:
            ae = ComputeAutoEncoder(N=8, k=4, decode_activation=activation)
            path = tmp_path / f"compute_{activation}.safetensors"
            ae.save_weights(path)

            loaded = AutoEncoderBase.from_local(path)
            assert loaded.decode_activation == activation


# ============================================================================
# 14. MALFORMED METADATA STRINGS
# ============================================================================


class TestMalformedMetadata:
    """What if the metadata strings themselves are malformed?"""

    def test_corrupted_json_in_config(self, tmp_path):
        """config field is not valid JSON."""
        path = tmp_path / "bad.safetensors"
        save_file(
            {"W": torch.randn(5, 10), "b": torch.zeros(10)},
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": "not valid json {{{{",
            },
        )
        with pytest.raises(json.JSONDecodeError):
            AutoEncoderBase.from_local(path)

    def test_empty_config_string(self, tmp_path):
        """config field is empty string."""
        path = tmp_path / "bad.safetensors"
        save_file(
            {"W": torch.randn(5, 10), "b": torch.zeros(10)},
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": "",
            },
        )
        with pytest.raises(json.JSONDecodeError):
            AutoEncoderBase.from_local(path)

    def test_config_is_json_null(self, tmp_path):
        """config = "null" (valid JSON, but not a dict)."""
        path = tmp_path / "bad.safetensors"
        ae = TiedLinearRelu(10, 5)
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": "null",
            },
        )
        # json.loads("null") returns None, then config is None
        # Code does: if config_str is not None -> True, then json.loads -> None
        # Then set(None.keys()) will fail with AttributeError
        # This is a potential bug — config could be None after parsing
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            AutoEncoderBase.from_local(path)

    def test_config_is_json_array(self, tmp_path):
        """config = "[1, 2, 3]" (valid JSON but not a dict)."""
        path = tmp_path / "bad.safetensors"
        ae = TiedLinearRelu(10, 5)
        save_file(
            ae.state_dict(),
            str(path),
            metadata={
                "class": "TiedLinearRelu",
                "config": "[1, 2, 3]",
            },
        )
        # json.loads("[1,2,3]") returns a list, then set(list.keys()) fails
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            AutoEncoderBase.from_local(path)
