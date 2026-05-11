"""Tests for ToyModel.fit(precomputed_data=...) and Distribution.save_samples()."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from occhio.autoencoders import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

DEVICE = "cpu"
N_FEATURES = 8
N_HIDDEN = 4


def _make_model():
    dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE)
    return ToyModel(distribution=dist, ae=ae, device=DEVICE)


# ── basic training from data file ────────────────────────────────────────────


class TestFitFromData:
    def test_trains_and_returns_losses(self, tmp_path):
        """fit(data=...) runs without error and returns a loss per epoch."""
        tm = _make_model()
        data = torch.rand(500, N_FEATURES)
        path = tmp_path / "samples.safetensors"
        save_file({"samples": data}, str(path))

        losses, hooks = tm.fit(n_epochs=10, batch_size=32, precomputed_data=path)
        assert len(losses) == 10
        assert all(isinstance(v, float) for v in losses)

    def test_loss_decreases(self, tmp_path):
        """Loss should generally decrease over enough epochs."""
        tm = _make_model()
        data = torch.rand(1000, N_FEATURES)
        path = tmp_path / "samples.safetensors"
        save_file({"samples": data}, str(path))

        losses, _ = tm.fit(n_epochs=500, batch_size=64, precomputed_data=path)
        # First 10 avg vs last 10 avg
        assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10

    def test_hooks_fire_with_data(self, tmp_path):
        """Hooks should be called even when training from data."""
        tm = _make_model()
        data = torch.rand(200, N_FEATURES)
        path = tmp_path / "samples.safetensors"
        save_file({"samples": data}, str(path))

        called = []

        def hook(d):
            called.append(d["epoch"])

        _, hook_returns = tm.fit(
            n_epochs=5, batch_size=32, precomputed_data=path, hooks=[hook]
        )
        assert len(called) == 5

    def test_auto_appends_extension(self, tmp_path):
        """Passing a path without .safetensors should still work."""
        tm = _make_model()
        data = torch.rand(100, N_FEATURES)
        save_file({"samples": data}, str(tmp_path / "samples.safetensors"))

        losses, _ = tm.fit(
            n_epochs=3, batch_size=16, precomputed_data=tmp_path / "samples"
        )
        assert len(losses) == 3


# ── validation errors ────────────────────────────────────────────────────────


class TestFitDataValidation:
    def test_dimension_mismatch_raises(self, tmp_path):
        tm = _make_model()
        wrong = torch.rand(100, N_FEATURES + 2)
        path = tmp_path / "wrong.safetensors"
        save_file({"samples": wrong}, str(path))

        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            tm.fit(n_epochs=1, precomputed_data=path)

    def test_multi_key_file_raises(self, tmp_path):
        tm = _make_model()
        path = tmp_path / "multi.safetensors"
        save_file(
            {"a": torch.rand(10, N_FEATURES), "b": torch.rand(10, N_FEATURES)},
            str(path),
        )

        with pytest.raises(ValueError, match="Expected exactly 1 tensor key"):
            tm.fit(n_epochs=1, precomputed_data=path)

    def test_1d_tensor_raises(self, tmp_path):
        tm = _make_model()
        path = tmp_path / "flat.safetensors"
        save_file({"samples": torch.rand(N_FEATURES)}, str(path))

        with pytest.raises(ValueError, match="2-D tensor"):
            tm.fit(n_epochs=1, precomputed_data=path)

    def test_batch_size_warning(self, tmp_path):
        """Warn when batch_size > dataset size."""
        tm = _make_model()
        data = torch.rand(50, N_FEATURES)
        path = tmp_path / "small.safetensors"
        save_file({"samples": data}, str(path))

        with pytest.warns(UserWarning, match="batch_size.*>100%"):
            tm.fit(n_epochs=1, batch_size=60, precomputed_data=path)


# ── Distribution.save_samples() ─────────────────────────────────────────────


class TestSaveSamples:
    def test_round_trip(self, tmp_path):
        """save_samples → fit(precomputed_data=...) works end-to-end."""
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(500, tmp_path / "data")
        assert path.suffix == ".safetensors"
        assert path.exists()

        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE)
        tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
        losses, _ = tm.fit(n_epochs=5, batch_size=32, precomputed_data=path)
        assert len(losses) == 5

    def test_auto_appends_extension(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "out")
        assert path.suffix == ".safetensors"
        assert path.exists()

    def test_does_not_double_extension(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "out.safetensors")
        assert path.name == "out.safetensors"

    def test_default_filename(self, tmp_path, monkeypatch):
        """When no path is given, defaults to <Class>_<f>f_<s>s_<timestamp>.safetensors."""
        monkeypatch.chdir(tmp_path)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(50)
        assert path.suffix == ".safetensors"
        assert path.exists()
        assert path.name.startswith(f"SparseUniform_{N_FEATURES}f_50s_")

    def test_saved_shape(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(200, tmp_path / "data")

        tensors = load_file(str(path))
        assert "samples" in tensors
        assert tensors["samples"].shape == (200, N_FEATURES)


# ── JSON companion file ─────────────────────────────────────────────────────


class TestSaveSamplesJson:
    def test_json_created_alongside_safetensors(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "data")
        json_path = path.with_suffix(".json")
        assert json_path.exists()

    def test_json_is_valid(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "data")
        info = json.loads(path.with_suffix(".json").read_text())
        assert isinstance(info, dict)

    def test_json_has_class_name(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "data")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["class"] == "SparseUniform"

    def test_json_has_sample_metadata(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(200, tmp_path / "data")
        info = json.loads(path.with_suffix(".json").read_text())
        assert info["samples"]["n_samples"] == 200
        assert info["samples"]["n_features"] == N_FEATURES
        assert info["samples"]["shape"] == [200, N_FEATURES]
        assert "dtype" in info["samples"]

    def test_json_has_distribution_attributes(self, tmp_path):
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, device=DEVICE)
        path = dist.save_samples(100, tmp_path / "data")
        info = json.loads(path.with_suffix(".json").read_text())
        attrs = info["attributes"]
        assert attrs["n_features"] == N_FEATURES
        # Tensor attrs are serialized as shape/dtype dicts (same as AutoEncoderBase)
        assert "p_active" in attrs
        assert attrs["p_active"]["shape"] == [N_FEATURES]
