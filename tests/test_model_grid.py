# ABOUTME: Comprehensive tests for ModelGrid: init, slicing, caching, fitting, validation.
# ABOUTME: Covers __getitem__ bounds/views, sample dedup, training correctness, and edge cases.

import pytest
import numpy as np
import torch
from torch import Generator, Tensor

from occhio.autoencoder import TiedLinearRelu
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
        cache_samples=cache,
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
        cache_samples=cache,
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

    def test_describe(self):
        grid = _make_grid(n_density=4, n_importance=3)
        assert grid.describe == {"density": 4, "importance": 3}

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
    def test_int_index_preserves_ndim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2]
        assert isinstance(sub, ModelGrid)
        assert len(sub.axes) == len(grid.axes)
        assert sub.shape == (1, 5)

    def test_int_partial_preserves_ndim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2]
        assert isinstance(sub, ModelGrid)
        assert len(sub.axes) == len(grid.axes)
        assert sub.shape == (1, 5)

    def test_1d_int_index_returns_toymodel(self):
        """1D grid + single int = all axes specified → returns ToyModel."""
        grid = _make_1d_grid(n=8)
        result = grid[4]
        assert isinstance(result, ToyModel)
        assert result is grid.models[4]

    def test_subgrid_always_has_same_axes_count(self):
        grid = _make_grid(n_density=6, n_importance=5)
        for key in [
            (0,),
            (5,),
            (slice(0, 3),),
            (slice(1, 4), 2),
            (0, slice(None)),
        ]:
            sub = grid[key]
            assert isinstance(sub, ModelGrid), f"Failed for key={key}"
            assert len(sub.axes) == len(grid.axes), f"Failed for key={key}"


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
        assert sub.shape == (3, 1)
        assert len(sub.axes) == 2

    def test_axis_labels_preserved(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2, 1:3]
        assert sub.axes[0].label == "density"
        assert sub.axes[1].label == "importance"

    def test_axis_values_are_tensors(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[2, 1:3]
        for ax in sub.axes:
            assert isinstance(ax.values, Tensor)

    def test_int_index_axis_value_correct(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[3]
        expected_val = grid.axes[0].values[3]
        assert torch.allclose(sub.axes[0].values, expected_val.unsqueeze(0))

    def test_slice_axis_values_correct(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        expected = grid.axes[0].values[1:4]
        assert torch.allclose(sub.axes[0].values, expected)


# ── __getitem__: Negative Indexing ───────────────────────────────────────────


class TestGetitemNegativeIndexing:
    def test_negative_int(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[-1]
        assert sub.shape == (1, 5)
        last_density = grid.axes[0].values[-1:]
        assert torch.allclose(sub.axes[0].values, last_density)

    def test_negative_int_second_dim(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:, -2]
        assert sub.shape == (6, 1)

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
        expected = grid.axes[0].values[[4, 3, 2]]
        assert torch.allclose(sub.axes[0].values, expected)

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
        expected = grid.axes[0].values[[4, 3, 2, 1, 0]]
        assert torch.allclose(sub.axes[0].values, expected)

    def test_explicit_neg_step_open_start(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[:2:-1]
        assert sub.shape == (3, 5)
        expected = grid.axes[0].values[[5, 4, 3]]
        assert torch.allclose(sub.axes[0].values, expected)


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
        for j in range(5):
            assert sub.models[0, j] is grid.models[2, j]

    def test_subgrid_models_is_numpy_view(self):
        grid = _make_grid(n_density=6, n_importance=5)
        sub = grid[1:4]
        assert sub.models.base is grid.models or np.shares_memory(
            sub.models, grid.models
        )


# ── Sample Caching ───────────────────────────────────────────────────────────


class TestSampleCaching:
    def test_cache_builds_sample_index(self):
        grid = _make_grid(cache=True)
        assert hasattr(grid, "_unique_distributions")
        assert hasattr(grid, "_sample_index")

    def test_no_cache_skips_sample_index(self):
        grid = _make_grid(cache=False)
        assert not hasattr(grid, "_unique_distributions")
        assert not hasattr(grid, "_sample_index")

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
            cache_samples=True,
        )
        assert len(grid._unique_distributions) == 1
        assert (grid._sample_index == 0).all()

    def test_different_densities_different_groups(self):
        grid = _make_1d_grid(n=4, cache=True)
        n_unique = len(grid._unique_distributions)
        assert n_unique == 4

    def test_sample_index_length_matches_flat_models(self):
        grid = _make_grid(cache=True)
        assert len(grid._sample_index) == len(grid.models.ravel())


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

    def test_sample_index_rebuilt_after_fit(self):
        grid = _make_grid(n_density=3, n_importance=2, cache=True)
        idx_before = grid._sample_index.clone()
        grid.fit(n_epochs=10, batch_size=64)
        assert grid._sample_index.shape == idx_before.shape

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
        assert sub.cache_samples is True

        grid2 = _make_grid(cache=False)
        sub2 = grid2[1:3]
        assert sub2.cache_samples is False


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_axes_raises(self):
        with pytest.raises(ValueError, match="At least one axis"):
            ModelGrid(_make_create_model(), axes=[], cache_samples=False)

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
                cache_samples=False,
            )

    def test_no_generator_with_cache_raises(self):
        def create_model_no_gen(params: dict, **kwargs) -> ToyModel:
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=0.5, device=DEVICE, generator=None
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        with pytest.raises(ValueError, match="generator"):
            ModelGrid(
                create_model_no_gen,
                axes=[
                    Axis(label="density", values=torch.tensor([0.1, 0.5])),
                ],
                cache_samples=True,
            )


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
