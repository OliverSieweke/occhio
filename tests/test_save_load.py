# ABOUTME: Tests for ModelGrid.save_models and ModelGrid.load_models.
# ABOUTME: Covers round-trip, shape validation, type checks, device handling, warnings, and edge cases.

import os
import pickle
import warnings

import numpy as np
import pytest
import torch
from torch import Generator, Tensor

from occhio.autoencoders import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import ToyModel

N_FEATURES = 4
N_HIDDEN = 2
DEVICE = "cpu"


def _make_create_model(*, seed: int = 42):
    def create_model(params: dict, **kwargs) -> ToyModel:
        density = params["density"]
        importance = params.get("importance", 1.0)
        gen = Generator(device=DEVICE).manual_seed(seed)
        return ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=density, device=DEVICE, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
            importances=importance ** torch.arange(N_FEATURES, dtype=torch.float32),
            device=DEVICE,
        )

    return create_model


def _make_grid(
    n_density: int = 4,
    n_importance: int = 3,
    cache: bool = True,
    seed: int = 42,
) -> ModelGrid:
    return ModelGrid(
        _make_create_model(seed=seed),
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 1.0, n_density)),
            Axis(label="importance", values=torch.linspace(0.5, 2.0, n_importance)),
        ],
        broadcast_samples=cache,
    )


def _make_1d_grid(n: int = 4, cache: bool = True, seed: int = 42) -> ModelGrid:
    def create_model(params: dict, **kwargs) -> ToyModel:
        density = params["density"]
        gen = Generator(device=DEVICE).manual_seed(seed)
        return ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=density, device=DEVICE, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
            importances=torch.ones(N_FEATURES),
            device=DEVICE,
        )

    return ModelGrid(
        create_model,
        axes=[Axis(label="density", values=torch.linspace(0.1, 1.0, n))],
        broadcast_samples=cache,
    )


# ── save_models ──────────────────────────────────────────────────────────────


class TestSaveModels:
    def test_creates_file(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        assert os.path.exists(path)

    def test_auto_appends_extension(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid")
        grid.save_models(path)
        assert os.path.exists(path + ".pkl")

    def test_does_not_double_extension(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".pkl")

    def test_file_contains_ndarray(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, np.ndarray)
        assert data.dtype == object

    def test_file_shape_matches_grid(self, tmp_path):
        grid = _make_grid(n_density=5, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert data.shape == (5, 3)

    def test_file_contains_toymodels(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        for m in data.ravel():
            assert isinstance(m, ToyModel)

    def test_empty_path_generates_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        grid = _make_grid()
        grid.save_models("")
        # Should create a file with default naming in cwd
        pkl_files = list(tmp_path.glob("model_grid*.pkl"))
        assert len(pkl_files) == 1

    def test_overwrite_existing_file(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)
        size1 = os.path.getsize(path)
        grid.save_models(path)
        size2 = os.path.getsize(path)
        assert size2 > 0
        assert size1 == size2


# ── load_models: happy path ──────────────────────────────────────────────────


class TestLoadModelsHappyPath:
    def test_round_trip_preserves_shape(self, tmp_path):
        grid = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=4, n_importance=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert grid2.models.shape == (4, 3)

    def test_round_trip_preserves_weights(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        grid.fit(n_epochs=20, batch_size=64)
        path = str(tmp_path / "trained.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        for orig, loaded in zip(grid.models.ravel(), grid2.models.ravel()):
            for key in orig.ae.state_dict():
                assert torch.equal(
                    orig.ae.state_dict()[key], loaded.ae.state_dict()[key]
                ), f"Weight mismatch on {key}"

    def test_round_trip_preserves_importances(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        for orig, loaded in zip(grid.models.ravel(), grid2.models.ravel()):
            assert torch.equal(orig.importances, loaded.importances)

    def test_round_trip_1d_grid(self, tmp_path):
        grid = _make_1d_grid(n=5)
        grid.fit(n_epochs=10, batch_size=32)
        path = str(tmp_path / "grid1d.pkl")
        grid.save_models(path)

        grid2 = _make_1d_grid(n=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert grid2.models.shape == (5,)
        for orig, loaded in zip(grid.models.ravel(), grid2.models.ravel()):
            assert torch.equal(orig.ae.state_dict()["W"], loaded.ae.state_dict()["W"])

    def test_auto_appends_extension_on_load(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid")
        grid.save_models(path)

        grid2 = _make_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert grid2.models.shape == grid.models.shape

    def test_loaded_models_are_toymodels(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        for m in grid2.models.ravel():
            assert isinstance(m, ToyModel)


# ── load_models: sample caching ──────────────────────────────────────────────


class TestLoadModelsCaching:
    def test_cache_rebuilt_after_load(self, tmp_path):
        grid = _make_grid(cache=True)
        grid.fit(n_epochs=10, batch_size=32)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        broadcasters, broadcast_map = grid2._build_broadcast()
        assert len(broadcasters) > 0
        assert len(broadcast_map) == grid2.models.size

    def test_no_cache_skips_rebuild(self, tmp_path):
        grid = _make_grid(cache=False)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(cache=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert not hasattr(grid2, "_unique_distributions")

    def test_fit_after_load_with_cache(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        grid.fit(n_epochs=10, batch_size=32)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        losses = grid2.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert len(losses) == 10


# ── load_models: warnings ────────────────────────────────────────────────────


class TestLoadModelsWarnings:
    def test_axes_warning_emitted(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grid2.load_models(path)

        axes_warnings = [w for w in caught if "may not match" in str(w.message)]
        assert len(axes_warnings) == 1

    def test_axes_warning_includes_labels(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grid2.load_models(path)

        axes_warnings = [w for w in caught if "may not match" in str(w.message)]
        msg = str(axes_warnings[0].message)
        assert "density" in msg
        assert "importance" in msg

    def test_axes_warning_includes_sizes(self, tmp_path):
        grid = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=4, n_importance=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grid2.load_models(path)

        axes_warnings = [w for w in caught if "may not match" in str(w.message)]
        msg = str(axes_warnings[0].message)
        assert "4 values" in msg
        assert "3 values" in msg


# ── load_models: print output ────────────────────────────────────────────────


class TestLoadModelsWarnings:
    def test_warns_about_axes(self, tmp_path):
        grid = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=4, n_importance=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid2.load_models(path)
        assert any(str(path) in str(warning.message) for warning in w)

    def test_warn_includes_path(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "my_grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid2.load_models(path)
        assert any("my_grid.pkl" in str(warning.message) for warning in w)


# ── load_models: type validation ─────────────────────────────────────────────


class TestLoadModelsTypeErrors:
    def test_empty_path_raises(self):
        grid = _make_grid()
        with pytest.raises(TypeError, match="non-empty string"):
            grid.load_models("")

    def test_non_string_path_raises(self):
        grid = _make_grid()
        with pytest.raises(TypeError, match="non-empty string"):
            grid.load_models(42)

    def test_non_ndarray_in_file_raises(self, tmp_path):
        path = str(tmp_path / "bad.pkl")
        with open(path, "wb") as f:
            pickle.dump({"not": "an array"}, f)

        grid = _make_grid()
        with pytest.raises(TypeError, match="numpy ndarray"):
            grid.load_models(path)

    def test_wrong_dtype_raises(self, tmp_path):
        path = str(tmp_path / "bad.pkl")
        with open(path, "wb") as f:
            pickle.dump(np.zeros((4, 3), dtype=float), f)

        grid = _make_grid(n_density=4, n_importance=3)
        with pytest.raises(TypeError, match="dtype"):
            grid.load_models(path)

    def test_non_toymodel_entries_raise(self, tmp_path):
        bad_models = np.empty((4, 3), dtype=object)
        for idx in np.ndindex(bad_models.shape):
            bad_models[idx] = "not a ToyModel"
        path = str(tmp_path / "bad.pkl")
        with open(path, "wb") as f:
            pickle.dump(bad_models, f)

        grid = _make_grid(n_density=4, n_importance=3)
        with pytest.raises(TypeError, match="ToyModel"):
            grid.load_models(path)

    def test_file_not_found_raises(self, tmp_path):
        grid = _make_grid()
        with pytest.raises(FileNotFoundError):
            grid.load_models(str(tmp_path / "nonexistent.pkl"))


# ── load_models: shape validation ────────────────────────────────────────────


class TestLoadModelsShapeErrors:
    def test_shape_mismatch_vs_axes_raises(self, tmp_path):
        grid_save = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid_save.save_models(path)

        grid_load = _make_grid(n_density=5, n_importance=3)
        with pytest.raises(ValueError, match="Shape mismatch"):
            grid_load.load_models(path)

    def test_shape_mismatch_second_dim_raises(self, tmp_path):
        grid_save = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid_save.save_models(path)

        grid_load = _make_grid(n_density=4, n_importance=5)
        with pytest.raises(ValueError, match="Shape mismatch"):
            grid_load.load_models(path)

    def test_ndim_mismatch_raises(self, tmp_path):
        grid_save = _make_1d_grid(n=4)
        path = str(tmp_path / "grid.pkl")
        grid_save.save_models(path)

        grid_load = _make_grid(n_density=4, n_importance=1)
        with pytest.raises(ValueError, match="Shape mismatch"):
            grid_load.load_models(path)

    def test_shape_error_message_includes_both_shapes(self, tmp_path):
        grid_save = _make_grid(n_density=4, n_importance=3)
        path = str(tmp_path / "grid.pkl")
        grid_save.save_models(path)

        grid_load = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(ValueError, match=r"\(4, 3\).*\(6, 5\)"):
            grid_load.load_models(path)


# ── load_models: autoencoder validation ──────────────────────────────────────


class TestLoadModelsAutoencoderValidation:
    def test_internally_inconsistent_ae_raises(self, tmp_path):
        """Corrupt a pickle so models have different AE architectures."""
        grid = _make_grid(n_density=2, n_importance=2, cache=False)
        models = grid.models.copy()

        gen = Generator(device=DEVICE).manual_seed(99)
        models[0, 0] = ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=0.5, device=DEVICE, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, 3, generator=gen, device=DEVICE),
            importances=torch.ones(N_FEATURES),
            device=DEVICE,
        )

        path = str(tmp_path / "corrupt.pkl")
        with open(path, "wb") as f:
            pickle.dump(models, f)

        grid_load = _make_grid(n_density=2, n_importance=2, cache=False)
        with pytest.raises(ValueError, match="architecture"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_load.load_models(path)

    def test_consistent_ae_does_not_raise(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=False)
        path = str(tmp_path / "good.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)


# ── load_models: trained weights survive round-trip ──────────────────────────


class TestLoadModelsTrainedRoundTrip:
    def test_trained_weights_differ_from_fresh(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        fresh_weights = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=30, batch_size=64)
        path = str(tmp_path / "trained.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        for i, m in enumerate(grid2.models.ravel()):
            loaded_w = m.ae.state_dict()["W"]
            assert not torch.equal(loaded_w, fresh_weights[i]), (
                f"Model {i}: loaded weights identical to fresh (untrained)"
            )

    def test_save_load_save_load_idempotent(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        grid.fit(n_epochs=20, batch_size=64)

        path1 = str(tmp_path / "round1.pkl")
        grid.save_models(path1)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path1)

        path2 = str(tmp_path / "round2.pkl")
        grid2.save_models(path2)

        grid3 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid3.load_models(path2)

        for orig, final in zip(grid.models.ravel(), grid3.models.ravel()):
            for key in orig.ae.state_dict():
                assert torch.equal(
                    orig.ae.state_dict()[key], final.ae.state_dict()[key]
                ), f"Mismatch on {key} after double round-trip"

    def test_continue_training_after_load(self, tmp_path):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        losses1 = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        path = str(tmp_path / "trained.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=3, n_importance=2, cache=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        losses2 = grid2.fit(n_epochs=20, batch_size=64, track_losses=True)
        assert len(losses2) == 20
        assert losses2[0] < losses1[0], (
            "Continued training should start from a lower loss than initial"
        )


# ── load_models: edge cases ──────────────────────────────────────────────────


class TestLoadModelsEdgeCases:
    def test_single_model_grid(self, tmp_path):
        grid = _make_grid(n_density=1, n_importance=1)
        path = str(tmp_path / "single.pkl")
        grid.save_models(path)

        grid2 = _make_grid(n_density=1, n_importance=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert grid2.models.shape == (1, 1)

    def test_load_replaces_models_not_appends(self, tmp_path):
        grid = _make_grid()
        path = str(tmp_path / "grid.pkl")
        grid.save_models(path)

        grid2 = _make_grid()
        old_id = id(grid2.models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid2.load_models(path)

        assert id(grid2.models) != old_id

    def test_load_twice_uses_second_file(self, tmp_path):
        grid_a = _make_grid(n_density=3, n_importance=2, seed=42)
        grid_a.fit(n_epochs=20, batch_size=64)
        path_a = str(tmp_path / "a.pkl")
        grid_a.save_models(path_a)

        grid_b = _make_grid(n_density=3, n_importance=2, seed=99)
        grid_b.fit(n_epochs=20, batch_size=64)
        path_b = str(tmp_path / "b.pkl")
        grid_b.save_models(path_b)

        grid_target = _make_grid(n_density=3, n_importance=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_target.load_models(path_a)
            grid_target.load_models(path_b)

        for loaded, expected in zip(grid_target.models.ravel(), grid_b.models.ravel()):
            assert torch.equal(
                loaded.ae.state_dict()["W"], expected.ae.state_dict()["W"]
            )
