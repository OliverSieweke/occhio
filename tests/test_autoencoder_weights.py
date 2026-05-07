# ABOUTME: Tests for AutoEncoderBase.save_weights / load_weights (safetensors).
# ABOUTME: Covers round-trip correctness, class validation, strict mode, and edge cases.

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import Generator

from occhio.autoencoders import (
    ComputeAutoEncoder,
    MLPEncoder,
    TiedLinearRelu,
)

DEVICE = "cpu"
N_FEATURES = 8
N_HIDDEN = 4


def _gen(seed=42):
    return Generator(device=DEVICE).manual_seed(seed)


# ── helpers: one factory per subclass ────────────────────────────────────────


def _make_tied_linear_relu(seed=42):
    return TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=_gen(seed), device=DEVICE)


def _make_mlp_encoder(seed=42):
    return MLPEncoder(
        embedding=[N_FEATURES, 6, N_HIDDEN],
        unembedding=[N_HIDDEN, 6, N_FEATURES],
        generator=_gen(seed),
        device=DEVICE,
    )


def _make_compute_ae(seed=42):
    return ComputeAutoEncoder(
        N=N_FEATURES,
        k=N_HIDDEN,
        generator=_gen(seed),
        device=DEVICE,
    )


ALL_FACTORIES = [
    pytest.param(_make_tied_linear_relu, id="TiedLinearRelu"),
    pytest.param(_make_mlp_encoder, id="MLPEncoder"),
    pytest.param(_make_compute_ae, id="ComputeAutoEncoder"),
]


# ── round-trip: weights are exactly preserved ────────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_weights_identical_after_round_trip(self, factory, tmp_path):
        """Save weights, load into a fresh model, verify every tensor matches."""
        original = factory()
        original_sd = {k: v.clone() for k, v in original.state_dict().items()}

        path = tmp_path / "model.safetensors"
        original.save_weights(path)

        restored = factory()
        restored.load_weights(path)

        restored_sd = restored.state_dict()
        assert set(original_sd.keys()) == set(restored_sd.keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], restored_sd[key]), f"Mismatch on {key}"

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_loaded_weights_overwrite_different_init(self, factory, tmp_path):
        """Weights from seed=42 loaded into a seed=99 model must match seed=42."""
        saved = factory(seed=42)
        saved_sd = {k: v.clone() for k, v in saved.state_dict().items()}

        path = tmp_path / "model.safetensors"
        saved.save_weights(path)

        different = factory(seed=99)
        different.load_weights(path)

        for key in saved_sd:
            assert torch.equal(saved_sd[key], different.state_dict()[key]), (
                f"Mismatch on {key}"
            )

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_double_round_trip_idempotent(self, factory, tmp_path):
        original = factory()
        original_sd = {k: v.clone() for k, v in original.state_dict().items()}

        p1 = tmp_path / "round1.safetensors"
        original.save_weights(p1)

        mid = factory()
        mid.load_weights(p1)
        p2 = tmp_path / "round2.safetensors"
        mid.save_weights(p2)

        final = factory()
        final.load_weights(p2)

        for key in original_sd:
            assert torch.equal(original_sd[key], final.state_dict()[key]), (
                f"Double round-trip mismatch on {key}"
            )

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_all_state_dict_keys_present(self, factory, tmp_path):
        """Every key in the original state_dict survives the round-trip."""
        model = factory()
        path = tmp_path / "model.safetensors"
        model.save_weights(path)

        restored = factory()
        restored.load_weights(path)

        assert model.state_dict().keys() == restored.state_dict().keys()


# ── class validation ─────────────────────────────────────────────────────────


class TestClassValidation:
    def test_wrong_class_raises_type_error(self, tmp_path):
        tied_relu = _make_tied_linear_relu()
        path = tmp_path / "relu.safetensors"
        tied_relu.save_weights(path)

        mlp = _make_mlp_encoder()
        with pytest.raises(TypeError, match="TiedLinearRelu.*MLPEncoder"):
            mlp.load_weights(path)

    def test_missing_metadata_raises_value_error(self, tmp_path):
        """A bare safetensors file without class metadata should be rejected."""
        path = tmp_path / "bare.safetensors"
        save_file({"W": torch.randn(4, 8), "b": torch.zeros(8)}, str(path))

        model = _make_tied_linear_relu()
        with pytest.raises(ValueError, match="no 'class' metadata"):
            model.load_weights(path)

    def test_same_shape_different_class_rejected(self, tmp_path):
        """TiedLinearRelu and ComputeAutoEncoder could have compatible shapes
        but different class names — the metadata check must catch this."""
        relu = _make_tied_linear_relu()
        path = tmp_path / "model.safetensors"
        relu.save_weights(path)

        compute = _make_compute_ae()
        with pytest.raises(TypeError, match="TiedLinearRelu.*ComputeAutoEncoder"):
            compute.load_weights(path)


# ── strict mode ──────────────────────────────────────────────────────────────


class TestStrictMode:
    def test_extra_keys_rejected_by_default(self, tmp_path):
        """If the file has keys the model doesn't expect, strict=True raises."""
        model = _make_tied_linear_relu()
        sd = model.state_dict()
        sd["extra_param"] = torch.zeros(3)
        path = tmp_path / "extra.safetensors"
        save_file(sd, str(path), metadata={"class": "TiedLinearRelu"})

        fresh = _make_tied_linear_relu()
        with pytest.raises(RuntimeError, match="extra_param"):
            fresh.load_weights(path)

    def test_extra_keys_allowed_with_strict_false(self, tmp_path):
        model = _make_tied_linear_relu()
        sd = model.state_dict()
        expected_W = sd["W"].clone()
        sd["extra_param"] = torch.zeros(3)
        path = tmp_path / "extra.safetensors"
        save_file(sd, str(path), metadata={"class": "TiedLinearRelu"})

        fresh = _make_tied_linear_relu()
        fresh.load_weights(path, strict=False)
        assert torch.equal(fresh.state_dict()["W"], expected_W)

    def test_missing_keys_rejected_by_default(self, tmp_path):
        """If the file is missing keys the model expects, strict=True raises."""
        model = _make_tied_linear_relu()
        sd = model.state_dict()
        del sd["b"]
        path = tmp_path / "missing.safetensors"
        save_file(sd, str(path), metadata={"class": "TiedLinearRelu"})

        fresh = _make_tied_linear_relu()
        with pytest.raises(RuntimeError, match="b"):
            fresh.load_weights(path)


# ── shape mismatch ───────────────────────────────────────────────────────────


class TestShapeMismatch:
    def test_different_n_hidden_raises(self, tmp_path):
        """Same class, different dimensions → RuntimeError from load_state_dict."""
        small = TiedLinearRelu(N_FEATURES, 2, generator=_gen(), device=DEVICE)
        path = tmp_path / "small.safetensors"
        small.save_weights(path)

        big = TiedLinearRelu(N_FEATURES, 6, generator=_gen(), device=DEVICE)
        with pytest.raises(RuntimeError):
            big.load_weights(path)

    def test_different_n_features_raises(self, tmp_path):
        a = TiedLinearRelu(4, N_HIDDEN, generator=_gen(), device=DEVICE)
        path = tmp_path / "a.safetensors"
        a.save_weights(path)

        b = TiedLinearRelu(10, N_HIDDEN, generator=_gen(), device=DEVICE)
        with pytest.raises(RuntimeError):
            b.load_weights(path)

    def test_mlp_different_layer_count_raises(self, tmp_path):
        """MLPEncoder with 3-layer embedding vs 2-layer → key mismatch."""
        deep = MLPEncoder(
            embedding=[N_FEATURES, 6, 4, N_HIDDEN],
            unembedding=[N_HIDDEN, 4, 6, N_FEATURES],
            generator=_gen(),
            device=DEVICE,
        )
        path = tmp_path / "deep.safetensors"
        deep.save_weights(path)

        shallow = _make_mlp_encoder()
        with pytest.raises(RuntimeError):
            shallow.load_weights(path)


# ── file handling ────────────────────────────────────────────────────────────


class TestFileHandling:
    def test_auto_appends_extension(self, tmp_path):
        model = _make_tied_linear_relu()
        returned = model.save_weights(tmp_path / "model")
        assert returned.suffix == ".safetensors"
        assert returned.exists()

    def test_does_not_double_extension(self, tmp_path):
        model = _make_tied_linear_relu()
        returned = model.save_weights(tmp_path / "model.safetensors")
        assert returned.name == "model.safetensors"
        assert returned.exists()

    def test_load_auto_appends_extension(self, tmp_path):
        model = _make_tied_linear_relu()
        model.save_weights(tmp_path / "model.safetensors")

        fresh = _make_tied_linear_relu()
        fresh.load_weights(tmp_path / "model")  # no extension
        assert torch.equal(model.state_dict()["W"], fresh.state_dict()["W"])

    def test_file_not_found_raises(self, tmp_path):
        model = _make_tied_linear_relu()
        with pytest.raises(FileNotFoundError):
            model.load_weights(tmp_path / "nonexistent.safetensors")

    def test_returns_path(self, tmp_path):
        model = _make_tied_linear_relu()
        result = model.save_weights(tmp_path / "out")
        assert result == tmp_path / "out.safetensors"

    def test_default_filename(self, tmp_path, monkeypatch):
        """When no path is given, defaults to <Class>_<f>x<h>_<timestamp>.safetensors."""
        monkeypatch.chdir(tmp_path)
        model = _make_tied_linear_relu()
        result = model.save_weights()
        assert result.suffix == ".safetensors"
        assert result.exists()
        assert result.name.startswith(f"TiedLinearRelu_{N_FEATURES}x{N_HIDDEN}_")


# ── JSON companion file ─────────────────────────────────────────────────────


class TestJsonInfo:
    def test_json_created_alongside_safetensors(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        json_path = path.with_suffix(".json")
        assert json_path.exists()

    def test_json_is_valid(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert isinstance(info, dict)

    def test_json_has_class_name(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["class"] == "TiedLinearRelu"

    def test_json_has_dimensions_in_attributes(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["attributes"]["n_features"] == N_FEATURES
        assert info["attributes"]["n_hidden"] == N_HIDDEN

    def test_json_has_parameter_shapes(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["parameters"]["W"]["shape"] == [N_HIDDEN, N_FEATURES]
        assert info["parameters"]["b"]["shape"] == [N_FEATURES]

    def test_json_has_total_params(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        expected = N_HIDDEN * N_FEATURES + N_FEATURES  # W + b
        assert info["total_params"] == expected

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_json_attributes_capture_subclass_specifics(self, factory, tmp_path):
        """Each subclass's non-private, JSON-serializable attrs are captured."""
        model = factory()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert isinstance(info["attributes"], dict)

    def test_mlp_json_has_layer_dims(self, tmp_path):
        model = _make_mlp_encoder()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["attributes"]["embedding_dims"] == [N_FEATURES, 6, N_HIDDEN]
        assert info["attributes"]["unembedding_dims"] == [N_HIDDEN, 6, N_FEATURES]

    def test_json_has_device_in_attributes(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["attributes"]["_init_device"] == "cpu"

    def test_json_has_generator_info(self, tmp_path):
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        gen = info["attributes"]["generator"]
        assert gen["type"] == "Generator"
        assert gen["device"] == "cpu"
        assert gen["initial_seed"] == 42

    def test_json_captures_private_attrs(self, tmp_path):
        """Private config attrs (like SynthAE._orthogonalize) are included."""
        model = _make_tied_linear_relu()
        path = model.save_weights(tmp_path / "model")
        info = json.loads(path.with_suffix(".json").read_text())
        # _init_device is a private attr that should be captured
        assert "_init_device" in info["attributes"]
