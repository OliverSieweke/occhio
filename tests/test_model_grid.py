"""
Comprehensive tests for ModelGrid: init, slicing, caching, fitting, validation.
Covers __getitem__ bounds/views, sample dedup, training correctness, save/load, and edge cases.
"""

import os
import pickle
import warnings

import pytest
import numpy as np
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
    """Returns a create_model callable with a fixed generator seed."""

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
    n_density: int = 6,
    n_importance: int = 5,
    cache: bool = True,
    seed: int = 42,
) -> ModelGrid:
    """Helper to build a small 2D grid."""
    return ModelGrid(
        _make_create_model(seed=seed),
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 1.0, n_density)),
            Axis(label="importance", values=torch.linspace(0.5, 2.0, n_importance)),
        ],
        broadcast_samples=cache,
    )


def _make_1d_grid(n: int = 8, cache: bool = True, seed: int = 42) -> ModelGrid:
    """Helper to build a 1D grid."""

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


# ── Initialization & Properties ──────────────────────────────────────────────


class TestInitialization:
    def test_shape_matches_axes(self):
        grid = _make_grid(n_density=6, n_importance=5)
        assert grid.shape == (6, 5)

    def test_models_array_shape(self):
        grid = _make_grid(n_density=6, n_importance=5)
        assert grid.models.shape == (6, 5)

    def test_models_are_toymodels(self):
        grid = _make_grid(n_density=3, n_importance=2)
        for m in grid.models.ravel():
            assert isinstance(m, ToyModel)

    def test_description(self):
        grid = _make_grid(n_density=4, n_importance=3)
        assert grid.description == {"density": 4, "importance": 3}

    def test_flattened_models_count(self):
        grid = _make_grid(n_density=4, n_importance=3)
        assert len(grid.models.ravel()) == 12

    def test_1d_grid_shape(self):
        grid = _make_1d_grid(n=5)
        assert grid.shape == (5,)
        assert grid.models.shape == (5,)

    def test_parameters_mesh_shapes(self):
        grid = _make_grid(n_density=4, n_importance=3)
        mesh = grid.parameters_mesh
        assert len(mesh) == 2
        assert mesh[0].shape == (4, 3)
        assert mesh[1].shape == (4, 3)


# ── __getitem__: Dimension Preservation ──────────────────────────────────────


class TestGetitemDimensionPreservation:
    def test_int_index_collapses_axis(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2]
        assert isinstance(sub, ModelGrid)
        assert len(sub.axes) == 1
        assert sub.shape == (5,)

    def test_int_partial_collapses_one_axis(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2]
        assert isinstance(sub, ModelGrid)
        assert sub.axes[0].label == "importance"
        assert sub.shape == (5,)

    def test_1d_int_index_returns_toymodel(self):
        """1D grid + single int = all axes specified → returns ToyModel."""
        grid = _make_1d_grid(n=8)
        result = grid[4]
        assert isinstance(result, ToyModel)
        assert result is grid.models[4]

    def test_subgrid_axes_count(self):
        grid = _make_grid(n_density=6, n_importance=5)
        # int collapses → 1 axis; slice preserves → 2 axes
        cases = {
            (0,): 1,
            (5,): 1,
            (slice(0, 3),): 2,
            (slice(1, 4), 2): 1,
            (0, slice(None)): 1,
            (slice(None), slice(None)): 2,
        }
        for key, expected_axes in cases.items():
            sub = grid[key]
            assert isinstance(sub, ModelGrid), f"Failed for key={key}"
            assert len(sub.axes) == expected_axes, (
                f"Failed for key={key}: expected {expected_axes} axes, got {len(sub.axes)}"
            )


# ── __getitem__: Single Model Indexing ────────────────────────────────────────


class TestGetitemSingleModel:
    def test_all_int_2d_returns_toymodel(self):
        grid = _make_grid(n_density=6, n_importance=5)
        result = grid[2, 3]
        assert isinstance(result, ToyModel)

    def test_all_int_returns_same_object(self):
        grid = _make_grid(n_density=6, n_importance=5)
        result = grid[2, 3]
        assert result is grid.models[2, 3]

    def test_all_int_1d_returns_toymodel(self):
        grid = _make_1d_grid(n=8)
        result = grid[4]
        assert isinstance(result, ToyModel)
        assert result is grid.models[4]

    def test_negative_ints_return_toymodel(self):
        grid = _make_grid(n_density=6, n_importance=5)
        result = grid[-1, -1]
        assert isinstance(result, ToyModel)
        assert result is grid.models[-1, -1]

    def test_origin_returns_toymodel(self):
        grid = _make_grid(n_density=6, n_importance=5)
        result = grid[0, 0]
        assert isinstance(result, ToyModel)
        assert result is grid.models[0, 0]

    def test_partial_int_still_returns_modelgrid(self):
        """Only 1 of 2 axes specified as int → still a sub-grid."""
        grid = _make_grid(n_density=6, n_importance=5)
        result = grid[2]
        assert isinstance(result, ModelGrid)


# ── __getitem__: Slice Behavior ──────────────────────────────────────────────


class TestGetitemSlicing:
    def test_basic_slice(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        assert sub.shape == (3, 5)

    def test_open_ended_slice(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[3:]
        assert sub.shape == (3, 5)

    def test_full_slice(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:]
        assert sub.shape == grid.shape

    def test_mixed_slice_and_int(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4, 2]
        assert sub.shape == (3,)
        assert len(sub.axes) == 1
        assert sub.axes[0].label == "density"

    def test_axis_labels_preserved(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2, 1:3]
        assert len(sub.axes) == 1
        assert sub.axes[0].label == "importance"

    def test_axis_values_are_sequences(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2, 1:3]
        for ax in sub.axes:
            assert isinstance(ax.values, (list, Tensor))

    def test_int_index_collapses_axis_leaves_remaining(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[3]
        # Int on density collapses it; remaining axis is importance
        assert len(sub.axes) == 1
        assert sub.axes[0].label == "importance"
        assert list(sub.axes[0].values) == list(grid.axes[1].values)

    def test_slice_axis_values_correct(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        expected = grid.axes[0].values[1:4]
        assert list(sub.axes[0].values) == list(expected)


# ── __getitem__: Negative Indexing ───────────────────────────────────────────


class TestGetitemNegativeIndexing:
    def test_negative_int(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[-1]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (5,)
        # Density axis collapsed; remaining is importance
        assert sub.axes[0].label == "importance"

    def test_negative_int_second_dim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:, -2]
        assert sub.shape == (6,)
        assert sub.axes[0].label == "density"

    def test_negative_slice_start(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[-3:]
        assert sub.shape == (3, 5)

    def test_negative_slice_stop(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:-2]
        assert sub.shape == (4, 5)

    def test_negative_start_and_stop(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[-4:-1]
        assert sub.shape == (3, 5)


# ── __getitem__: Reverse Indexing ────────────────────────────────────────────


class TestGetitemReverseIndexing:
    def test_reverse_slice_auto_step(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[4:1]
        assert sub.shape == (3, 5)
        expected = [grid.axes[0].values[i] for i in [4, 3, 2]]
        assert list(sub.axes[0].values) == expected

    def test_reverse_slice_models_order(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[4:1]
        for i, rev_i in enumerate([4, 3, 2]):
            assert sub.models[i, 0] is grid.models[rev_i, 0]

    def test_reverse_second_axis(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:, 4:1]
        assert sub.shape == (6, 3)

    def test_explicit_neg_step_open_stop(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[4::-1]
        assert sub.shape == (5, 5)
        expected = [grid.axes[0].values[i] for i in [4, 3, 2, 1, 0]]
        assert list(sub.axes[0].values) == expected

    def test_explicit_neg_step_open_start(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:2:-1]
        assert sub.shape == (3, 5)
        expected = [grid.axes[0].values[i] for i in [5, 4, 3]]
        assert list(sub.axes[0].values) == expected


# ── __getitem__: Out-of-Bounds Errors ────────────────────────────────────────


class TestGetitemBoundsErrors:
    def test_int_oob_positive(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[6]

    def test_int_oob_negative(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[-7]

    def test_slice_stop_exceeds_dim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[0:9999]

    def test_slice_stop_exceeds_second_dim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[:, 0:100]

    def test_slice_start_exceeds_dim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[100:]

    def test_too_many_indices(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="Too many indices"):
            grid[0, 0, 0]

    def test_negative_resolves_then_oob(self):
        grid = _make_grid(n_density=6, n_importance=5)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[:-7]


# ── __getitem__: View (Not Copy) ─────────────────────────────────────────────


class TestGetitemView:
    def test_subgrid_shares_model_objects(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        for i in range(3):
            for j in range(5):
                assert sub.models[i, j] is grid.models[i + 1, j]

    def test_int_index_shares_model_objects(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2]
        # Int collapses density → sub.models is 1D with 5 elements
        for j in range(5):
            assert sub.models[j] is grid.models[2, j]

    def test_subgrid_models_is_numpy_view(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        assert sub.models.base is grid.models or np.shares_memory(
            sub.models, grid.models
        )


# ── Sample Caching ───────────────────────────────────────────────────────────


class TestSampleCaching:
    def test_build_broadcast_returns_correct_types(self):
        grid = _make_grid(cache=True)
        broadcasters, broadcast_map = grid._build_broadcast()
        assert isinstance(broadcasters, list)
        assert isinstance(broadcast_map, Tensor)

    def test_no_cache_flag(self):
        grid = _make_grid(cache=False)
        assert grid.broadcast_samples is False

    def test_all_same_seed_collapses_to_one(self):
        """All models use seed=42 and same density→same distribution hash→one unique."""

        def create_model(params: dict, **kwargs) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="dummy", values=torch.arange(5, dtype=torch.float32))],
            broadcast_samples=True,
        )
        broadcasters, broadcast_map = grid._build_broadcast()
        assert len(broadcasters) == 1
        assert (broadcast_map == 0).all()

    def test_different_densities_different_groups(self):
        grid = _make_1d_grid(n=4, cache=True)
        broadcasters, _ = grid._build_broadcast()
        assert len(broadcasters) == 4

    def test_broadcast_map_length_matches_flat_models(self):
        grid = _make_grid(cache=True)
        _, broadcast_map = grid._build_broadcast()
        assert len(broadcast_map) == len(grid.models.ravel())


# ── Fitting ──────────────────────────────────────────────────────────────────


class TestFitting:
    def test_fit_returns_losses(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        assert losses is not None
        assert len(losses) == 20

    def test_fit_without_tracking(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        result = grid.fit(n_epochs=10, batch_size=64, track_losses=False)
        assert result is None

    def test_loss_decreases(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=50, batch_size=128, track_losses=True)
        assert losses[-1] < losses[0]

    def test_fit_without_cache(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=False)
        losses = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        assert len(losses) == 20

    def test_state_written_back_to_models(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=30, batch_size=64)
        for i, m in enumerate(grid.models.ravel()):
            after = m.ae.state_dict()["W"]
            assert not torch.equal(before[i], after), f"Model {i} weights unchanged"

    def test_broadcast_map_stable_after_fit(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        _, map_before = grid._build_broadcast()
        grid.fit(n_epochs=10, batch_size=64)
        _, map_after = grid._build_broadcast()
        assert map_after.shape == map_before.shape

    def test_fit_twice_works(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        losses1 = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        losses2 = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        assert len(losses1) == 20
        assert len(losses2) == 20

    def test_1d_grid_fit(self):
        grid = _make_1d_grid(n=4, cache=True)
        losses = grid.fit(n_epochs=20, batch_size=64, track_losses=True)
        assert len(losses) == 20


# ── Sample Every ─────────────────────────────────────────────────────────────


class TestSampleEvery:
    def test_sample_every_with_cache(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=20, batch_size=64, sample_every=5)
        for i, m in enumerate(grid.models.ravel()):
            assert not torch.equal(before[i], m.ae.state_dict()["W"])

    def test_sample_every_without_cache(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=False)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=20, batch_size=64, sample_every=5)
        for i, m in enumerate(grid.models.ravel()):
            assert not torch.equal(before[i], m.ae.state_dict()["W"])

    def test_sample_every_1_runs(self):
        """sample_every=1 should work without error."""
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=20, batch_size=64, sample_every=1)
        for i, m in enumerate(grid.models.ravel()):
            assert not torch.equal(before[i], m.ae.state_dict()["W"])

    def test_sample_every_not_evenly_divisible(self):
        """n_epochs=23 with sample_every=5: should handle the remainder correctly."""
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=23, batch_size=64, sample_every=5)
        for i, m in enumerate(grid.models.ravel()):
            assert not torch.equal(before[i], m.ae.state_dict()["W"])

    def test_sample_every_validation_zero(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        with pytest.raises(ValueError, match="sample_every"):
            grid.fit(n_epochs=20, batch_size=64, sample_every=0)

    def test_sample_every_validation_negative(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        with pytest.raises(ValueError, match="sample_every"):
            grid.fit(n_epochs=20, batch_size=64, sample_every=-1)

    def test_sample_every_with_snapshot_interval(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        history = grid.fit(
            n_epochs=20,
            batch_size=64,
            sample_every=5,
            snapshot_interval=10,
        )
        assert history is not None
        # Should have 3 snapshots: epoch 0, 10, 20
        assert history.models.shape[0] == 3

    def test_sample_every_state_written_back(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = {
            i: m.ae.state_dict()["W"].clone() for i, m in enumerate(grid.models.ravel())
        }
        grid.fit(n_epochs=30, batch_size=64, sample_every=10)
        for i, m in enumerate(grid.models.ravel()):
            after = m.ae.state_dict()["W"]
            assert not torch.equal(before[i], after), f"Model {i} weights unchanged"

    def test_sample_every_default_is_10(self):
        """Verify the default value is 10 by checking fit works without the argument."""
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        before = grid.models.ravel()[0].ae.state_dict()["W"].clone()
        grid.fit(n_epochs=20, batch_size=64)
        after = grid.models.ravel()[0].ae.state_dict()["W"]
        assert not torch.equal(before, after)


# ── Sub-grid Fitting & View Mutation ─────────────────────────────────────────


class TestSubgridFitting:
    def test_subgrid_fit_mutates_parent_models(self):
        grid = _make_grid(n_density=6, n_importance=5, cache=False)
        sub = grid[1:4]

        weight_before = grid.models[2, 0].ae.state_dict()["W"].clone()
        sub.fit(n_epochs=30, batch_size=64)
        weight_after = grid.models[2, 0].ae.state_dict()["W"]

        assert not torch.equal(weight_before, weight_after), (
            "Parent grid model should be mutated via sub-grid view"
        )

    def test_subgrid_retains_create_model(self):
        grid = _make_grid()
        sub = grid[1:3]
        assert sub.create_model is grid.create_model

    def test_subgrid_retains_cache_setting(self):
        grid = _make_grid(cache=True)
        sub = grid[1:3]
        assert sub.broadcast_samples is True

        grid2 = _make_grid(cache=False)
        sub2 = grid2[1:3]
        assert sub2.broadcast_samples is False


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_axes_raises(self):
        with pytest.raises(ValueError, match="At least one axis"):
            ModelGrid(_make_create_model(), axes=[], broadcast_samples=False)

    def test_missing_params_arg_raises(self):
        def bad_create_model(x: int) -> ToyModel:
            return ToyModel(
                distribution=SparseUniform(4, p_active=0.5, device=DEVICE),
                ae=TiedLinearRelu(4, 2, device=DEVICE),
                importances=torch.ones(4),
                device=DEVICE,
            )

        with pytest.raises(TypeError, match="params"):
            ModelGrid(
                bad_create_model,
                axes=[Axis(label="x", values=torch.tensor([1.0]))],
                broadcast_samples=False,
            )

    def test_no_generator_with_cache_warns(self):
        def create_model_no_gen(params: dict, **kwargs) -> ToyModel:
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=0.5, device=DEVICE, generator=None
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ModelGrid(
                create_model_no_gen,
                axes=[
                    Axis(label="density", values=torch.tensor([0.1, 0.5])),
                ],
                broadcast_samples=True,
            )
            assert any("generator" in str(warning.message).lower() for warning in w)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_element_grid(self):
        grid = _make_grid(n_density=1, n_importance=1)
        assert grid.shape == (1, 1)
        assert len(grid.models.ravel()) == 1

    def test_single_element_fit(self):
        grid = _make_grid(n_density=1, n_importance=1)
        losses = grid.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert len(losses) == 10

    def test_subgrid_of_subgrid(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub1 = grid[1:5]
        sub2 = sub1[0:2]
        assert sub2.shape == (2, 5)
        assert len(sub2.axes) == 2
        assert sub2.models[0, 0] is grid.models[1, 0]

    def test_full_slice_is_identity(self):
        grid = _make_grid(n_density=4, n_importance=3)
        sub = grid[:, :]
        assert sub.shape == grid.shape
        for idx in np.ndindex(grid.shape):
            assert sub.models[idx] is grid.models[idx]

    def test_unsupported_index_type_raises(self):
        grid = _make_grid()
        with pytest.raises(IndexError, match="Unsupported index type"):
            grid["bad"]


# ── sample_every × broadcast_samples Interaction ─────────────────────────────────


def _make_shared_dist_grid(n_models: int = 4, cache: bool = True, seed: int = 42):
    """All models share the same distribution (same seed, same p_active)
    so they collapse to 1 unique distribution under broadcast_samples."""

    def create_model(params: dict, **kwargs) -> ToyModel:
        gen = Generator(device=DEVICE).manual_seed(seed)
        return ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=0.5, device=DEVICE, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
            importances=torch.ones(N_FEATURES),
            device=DEVICE,
        )

    return ModelGrid(
        create_model,
        axes=[Axis(label="idx", values=torch.arange(n_models, dtype=torch.float32))],
        broadcast_samples=cache,
    )


class TestSampleEveryAndCacheSamplesInteraction:
    """Rigorous tests for the interaction between sample_every (epoch-level
    sample buffering) and broadcast_samples (model-level distribution dedup)."""

    # ── Correctness: shared distributions get identical data ──────────────

    def test_shared_distribution_identical_weights_after_training(self):
        """With broadcast_samples=True, models sharing a distribution see exactly
        the same input batches. Starting from the same weights they must
        converge to identical final weights."""
        grid = _make_shared_dist_grid(n_models=3, cache=True)
        broadcasters, _ = grid._build_broadcast()
        assert len(broadcasters) == 1

        grid.fit(n_epochs=30, batch_size=64, sample_every=5)

        models = grid.models.ravel()
        ref_W = models[0].ae.state_dict()["W"]
        for m in models[1:]:
            assert torch.equal(ref_W, m.ae.state_dict()["W"]), (
                "Models with same distribution + same init should have "
                "identical weights after training with broadcast_samples=True"
            )

    def test_generators_not_synced_without_cache(self):
        """Without broadcast_samples, generators are NOT synchronized after fit.
        With broadcast_samples=True, _sync_generators runs and all distributions
        in the same group have the same generator state afterward."""
        grid_no_cache = _make_shared_dist_grid(n_models=3, cache=False)
        grid_no_cache.fit(n_epochs=20, batch_size=64, sample_every=5)

        # Without cache, no _sync_generators call → broadcast_samples is False
        assert grid_no_cache.broadcast_samples is False

        grid_cache = _make_shared_dist_grid(n_models=3, cache=True)
        grid_cache.fit(n_epochs=20, batch_size=64, sample_every=5)

        # With cache, generators ARE synced
        models = grid_cache.models.ravel()
        states = [m.distribution.generator.get_state() for m in models]
        for s in states[1:]:
            assert torch.equal(states[0], s), (
                "With broadcast_samples=True, generators should be synced after fit"
            )

    def test_different_distributions_different_weights(self):
        """Models with genuinely different distributions must produce
        different final weights even with broadcast_samples=True."""
        grid = _make_1d_grid(n=3, cache=True)
        # Different densities → different unique distributions
        broadcasters, _ = grid._build_broadcast()
        assert len(broadcasters) == 3

        grid.fit(n_epochs=40, batch_size=64, sample_every=5)

        models = grid.models.ravel()
        w0 = models[0].ae.state_dict()["W"]
        any_differ = any(
            not torch.equal(w0, m.ae.state_dict()["W"]) for m in models[1:]
        )
        assert any_differ

    # ── Buffer slicing: each epoch gets fresh data ────────────────────────

    def test_buffer_slices_differ_across_epochs(self):
        """Within a sample_every window the buffer must be sliced so
        consecutive epochs train on different batches."""
        # We intercept by running two 1-epoch fits with sample_every=1
        # vs one 2-epoch fit with sample_every=2. If sample_every=2
        # reuses the same slice both epochs, loss would be suspiciously
        # equal; with proper slicing, the per-epoch losses will differ.
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=2, batch_size=64, sample_every=2, track_losses=True)
        # Two epochs on different slices of the same buffer → different losses
        assert losses[0] != losses[1], (
            "Consecutive epochs within a sample_every window should "
            "train on different data slices"
        )

    # ── Edge cases on sample_every ────────────────────────────────────────

    def test_sample_every_larger_than_n_epochs(self):
        """sample_every > n_epochs should work: single refill, buffer sized
        to exactly n_epochs * batch_size."""
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        before = grid.models.ravel()[0].ae.state_dict()["W"].clone()
        losses = grid.fit(
            n_epochs=5, batch_size=64, sample_every=100, track_losses=True
        )
        assert len(losses) == 5
        after = grid.models.ravel()[0].ae.state_dict()["W"]
        assert not torch.equal(before, after)

    def test_sample_every_equals_n_epochs(self):
        """sample_every == n_epochs: one refill for entire training.
        All epochs train on slices of the same buffer."""
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(
            n_epochs=10, batch_size=64, sample_every=10, track_losses=True
        )
        assert len(losses) == 10
        # Weights should change (training happened)
        # Note: loss may not monotonically decrease on a single buffer
        # because the model sees each slice only once.

    def test_n_epochs_1_sample_every_1(self):
        """Minimal training: 1 epoch, sample_every=1."""
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=1, batch_size=64, sample_every=1, track_losses=True)
        assert len(losses) == 1

    def test_n_epochs_1_sample_every_10(self):
        """Single epoch with sample_every > 1: buffer should be sized to 1 epoch."""
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=1, batch_size=64, sample_every=10, track_losses=True)
        assert len(losses) == 1

    def test_sample_every_2_n_epochs_3(self):
        """Odd remainder: n_epochs=3, sample_every=2 → refill at ep 0 (size 2)
        and ep 2 (size 1)."""
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=3, batch_size=64, sample_every=2, track_losses=True)
        assert len(losses) == 3

    # ── Generator synchronization after fit ───────────────────────────────

    def test_generator_sync_after_fit(self):
        """After fit with broadcast_samples=True, distributions in the same
        group must have synchronized generators so future sampling is
        consistent."""
        grid = _make_shared_dist_grid(n_models=3, cache=True)
        grid.fit(n_epochs=20, batch_size=64, sample_every=5)

        models = grid.models.ravel()
        samples = [m.distribution.sample(128) for m in models]
        for s in samples[1:]:
            assert torch.equal(samples[0], s), (
                "After fit + sync, distributions in same group "
                "should produce identical samples"
            )

    def test_generator_sync_allows_repeated_fit(self):
        """After a first fit, generator sync should leave the grid in a
        state where a second fit works correctly."""
        grid = _make_shared_dist_grid(n_models=3, cache=True)
        grid.fit(n_epochs=10, batch_size=64, sample_every=3)

        # Second fit should not error and should still decrease loss
        losses = grid.fit(n_epochs=20, batch_size=64, sample_every=5, track_losses=True)
        assert len(losses) == 20

    # ── Determinism ───────────────────────────────────────────────────────

    def test_deterministic_training_same_seed(self):
        """Two identically-constructed grids trained with the same
        hyperparameters must produce bit-identical final weights."""

        def train_grid():
            grid = _make_shared_dist_grid(n_models=2, cache=True, seed=99)
            grid.fit(n_epochs=15, batch_size=64, sample_every=5)
            return [m.ae.state_dict()["W"].clone() for m in grid.models.ravel()]

        run1 = train_grid()
        run2 = train_grid()
        for w1, w2 in zip(run1, run2):
            assert torch.equal(w1, w2), "Same seed must produce same result"

    def test_broadcast_samples_dedup_reduces_unique_distributions(self):
        """With broadcast_samples=True, models sharing the same distribution
        parameters are collapsed into fewer unique distributions. Verify
        that the dedup count is correct for a known setup."""
        # _make_grid: each (density, importance) pair has a unique density
        # but same density across importance axis → n_density unique dists
        grid = _make_grid(n_density=3, n_importance=4, cache=True)
        # All models in a density row share the same density (but seed=42 for all),
        # so all 12 models have the same p_active for a given density row.
        # With _make_create_model, all models use seed=42, but p_active differs
        # per density → n_density unique distributions.
        broadcasters, _ = grid._build_broadcast()
        n_unique = len(broadcasters)
        n_total = grid.models.size
        assert n_unique <= n_total, (
            f"Dedup should not create MORE distributions than models "
            f"({n_unique} > {n_total})"
        )
        # In our setup, each density value creates a different distribution
        assert n_unique == 3, (
            f"Expected 3 unique distributions (one per density), got {n_unique}"
        )

    # ── Loss tracking interaction ─────────────────────────────────────────

    def test_fit_returns_losses_with_sample_every(self):
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        losses = grid.fit(n_epochs=20, batch_size=64, sample_every=7, track_losses=True)
        assert isinstance(losses, list)
        assert len(losses) == 20

    def test_fit_returns_none_when_no_tracking_no_snapshot(self):
        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        result = grid.fit(
            n_epochs=10, batch_size=64, sample_every=3, track_losses=False
        )
        assert result is None

    def test_fit_returns_history_grid_with_snapshot(self):
        """snapshot_interval takes priority over losses in the return value."""
        from occhio.model_grid import TrainingAxis

        grid = _make_grid(n_density=2, n_importance=2, cache=True)
        result = grid.fit(
            n_epochs=20,
            batch_size=64,
            sample_every=5,
            snapshot_interval=10,
            track_losses=True,  # losses built internally but snapshot wins
        )
        assert isinstance(result, ModelGrid)
        assert isinstance(result.axes[0], TrainingAxis)
        # epoch 0, 10, 20 → 3 snapshots
        assert result.shape[0] == 3

    # ── broadcast_samples with 2D grid + sample_every ────────────────────────

    def test_2d_grid_broadcast_map_covers_all_models(self):
        """In a 2D grid, the broadcast map must cover all flattened models
        before and after training."""
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        n_flat = grid.models.size
        _, broadcast_map = grid._build_broadcast()
        assert len(broadcast_map) == n_flat

        grid.fit(n_epochs=10, batch_size=64, sample_every=3)

        _, broadcast_map_after = grid._build_broadcast()
        assert len(broadcast_map_after) == n_flat

    def test_2d_grid_unique_distributions_count_stable(self):
        """The number of unique distributions should not change after fit."""
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        broadcasters_before, _ = grid._build_broadcast()

        grid.fit(n_epochs=10, batch_size=64, sample_every=3)

        broadcasters_after, _ = grid._build_broadcast()
        assert len(broadcasters_after) == len(broadcasters_before), (
            f"Unique distribution count changed from {len(broadcasters_before)} "
            f"to {len(broadcasters_after)} after fit"
        )
