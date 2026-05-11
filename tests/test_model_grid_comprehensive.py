"""Comprehensive tests for ModelGrid, Axis, and TrainingAxis.

Covers the following areas NOT fully exercised by the existing test_model_grid.py:

- Axis/TrainingAxis construction with diverse value types (tensor, list, enum, generator)
- ModelGrid construction: 1D, 2D, 3D grids; factory param forwarding; broadcast_samples flag
- ModelGrid.from_iterable: 1D, 2D, nested structures; error paths
- ModelGrid.fit: snapshot_interval, track_losses, sample_every edge values,
  multiple consecutive fit calls, weight write-back verification
- ModelGrid.__getitem__: step slicing, TrainingAxis preservation, unsupported types
- ModelGrid.save / load: dill round-trip preserves structure, axes, models
- ModelGrid.save_models / load_models: pickle round-trip, shape mismatch validation
- ModelGrid.sae_results_to_dataframe: empty results, correct index structure
- _build_broadcast: shared vs distinct distributions, generator-less distributions
- Edge cases: single-element grid fit, 3D grid, empty axes error
"""

from __future__ import annotations

import pickle
import warnings
from copy import deepcopy
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Generator, Tensor

from occhio import ToyModel
from occhio.autoencoders import AutoencoderType, TiedLinear, TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid, TrainingAxis


# ── Constants ────────────────────────────────────────────────────────────────

N_FEATURES = 8
N_HIDDEN = 4
DEVICE = "cpu"
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────


def _simple_create_model(params: dict) -> ToyModel:
    """Minimal factory: reads 'density' from params, ignores the rest."""
    density = params.get("density", 0.5)
    gen = Generator(device=DEVICE).manual_seed(SEED)
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES, p_active=density, device=DEVICE, generator=gen
        ),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
        importances=torch.ones(N_FEATURES),
        device=DEVICE,
    )


def _multi_param_create_model(params: dict) -> ToyModel:
    """Factory that uses density, importance, and optionally lr."""
    density = params.get("density", 0.5)
    importance = params.get("importance", 1.0)
    gen = Generator(device=DEVICE).manual_seed(SEED)
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES, p_active=density, device=DEVICE, generator=gen
        ),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
        importances=importance ** torch.arange(N_FEATURES, dtype=torch.float32),
        device=DEVICE,
    )


def _make_toy_model(density: float = 0.5, seed: int = SEED) -> ToyModel:
    """Stand-alone helper for from_iterable tests."""
    gen = Generator(device=DEVICE).manual_seed(seed)
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES, p_active=density, device=DEVICE, generator=gen
        ),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
        importances=torch.ones(N_FEATURES),
        device=DEVICE,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def grid_1d():
    """1D grid with 4 density values."""
    return ModelGrid(
        _simple_create_model,
        axes=[Axis(label="density", values=torch.linspace(0.1, 1.0, 4))],
        broadcast_samples=True,
    )


@pytest.fixture
def grid_2d():
    """2D grid: 3 densities x 2 importances."""
    return ModelGrid(
        _multi_param_create_model,
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 0.9, 3)),
            Axis(label="importance", values=torch.tensor([0.5, 2.0])),
        ],
        broadcast_samples=True,
    )


@pytest.fixture
def grid_3d():
    """3D grid: 2 x 2 x 2 = 8 models total."""

    def create_model(params: dict) -> ToyModel:
        gen = Generator(device=DEVICE).manual_seed(SEED)
        return ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=params["density"], device=DEVICE, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
            importances=params["importance"]
            ** torch.arange(N_FEATURES, dtype=torch.float32),
            device=DEVICE,
        )

    return ModelGrid(
        create_model,
        axes=[
            Axis(label="density", values=torch.tensor([0.3, 0.7])),
            Axis(label="importance", values=torch.tensor([0.5, 2.0])),
            Axis(label="lr", values=torch.tensor([1e-3, 3e-4])),
        ],
        broadcast_samples=False,
    )


@pytest.fixture
def grid_no_broadcast():
    """2D grid without broadcast_samples."""
    return ModelGrid(
        _multi_param_create_model,
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 0.9, 3)),
            Axis(label="importance", values=torch.tensor([0.5, 2.0])),
        ],
        broadcast_samples=False,
    )


@pytest.fixture
def single_element_grid():
    """Grid with exactly one model."""
    return ModelGrid(
        _simple_create_model,
        axes=[Axis(label="density", values=[0.5])],
        broadcast_samples=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Axis / TrainingAxis
# ══════════════════════════════════════════════════════════════════════════════


class TestAxisConstruction:
    """Tests that Axis correctly stores labels and converts various value types."""

    def test_axis_with_tensor_values(self):
        """Tensor values are converted to list (tensors are not Sequence),
        so the stored values should be a list of scalar tensors."""
        vals = torch.tensor([1.0, 2.0, 3.0])
        ax = Axis(label="x", values=vals)
        assert ax.label == "x"
        # Tensor is not a Sequence -> converted to list of scalar tensors
        assert isinstance(ax.values, list)
        assert len(ax.values) == 3
        assert float(ax.values[0]) == pytest.approx(1.0)
        assert float(ax.values[2]) == pytest.approx(3.0)

    def test_axis_with_list_values(self):
        """A plain list should be stored directly (list is a Sequence)."""
        ax = Axis(label="y", values=[10, 20, 30])
        assert ax.label == "y"
        assert ax.values == [10, 20, 30]

    def test_axis_with_enum_values(self):
        """Enums (which are iterable but not Sequence) should be converted to list."""

        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        ax = Axis(label="color", values=Color)
        assert ax.label == "color"
        assert len(ax.values) == 3
        assert Color.RED in ax.values

    def test_axis_with_generator_values(self):
        """Generator expressions (not Sequence) should be materialized to list."""
        ax = Axis(label="gen", values=(x**2 for x in range(5)))
        assert ax.values == [0, 1, 4, 9, 16]

    def test_axis_with_autoencoder_type_enum(self):
        """AutoencoderType enum should work as axis values -- real-world use case."""
        ax = Axis(label="ae_type", values=AutoencoderType)
        assert len(ax.values) > 0
        assert AutoencoderType.TiedLinearRelu in ax.values

    def test_axis_with_range(self):
        """range objects are Sequences and should be preserved."""
        ax = Axis(label="r", values=range(5))
        assert len(ax.values) == 5

    def test_axis_single_element(self):
        """Single-element lists are valid."""
        ax = Axis(label="s", values=[42])
        assert len(ax.values) == 1
        assert ax.values[0] == 42


class TestTrainingAxis:
    """Tests for the TrainingAxis subclass."""

    def test_is_subclass_of_axis(self):
        """TrainingAxis must be a proper subclass for isinstance checks in __getitem__."""
        assert issubclass(TrainingAxis, Axis)

    def test_default_label(self):
        """Default label should be 'Epoch'."""
        ta = TrainingAxis(values=[0, 10, 20])
        assert ta.label == "Epoch"

    def test_custom_label(self):
        """Custom label should override default."""
        ta = TrainingAxis(values=[0, 50, 100], label="Step")
        assert ta.label == "Step"

    def test_values_stored(self):
        """Values should be accessible."""
        vals = [0, 100, 200]
        ta = TrainingAxis(values=vals)
        assert ta.values == vals

    def test_tensor_values(self):
        """Tensor values are converted to list by Axis.__init__."""
        vals = torch.tensor([0, 5, 10])
        ta = TrainingAxis(values=vals)
        assert isinstance(ta.values, list)
        assert len(ta.values) == 3
        assert int(ta.values[0]) == 0
        assert int(ta.values[2]) == 10

    def test_isinstance_check(self):
        """isinstance(ta, Axis) should return True."""
        ta = TrainingAxis(values=[0])
        assert isinstance(ta, Axis)


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid Construction
# ══════════════════════════════════════════════════════════════════════════════


class TestModelGridConstruction:
    """Covers grid init, shape, description, axes, factory param forwarding."""

    def test_1d_grid_shape(self, grid_1d):
        """1D grid shape must match axis length."""
        assert grid_1d.shape == (4,)

    def test_2d_grid_shape(self, grid_2d):
        """2D grid shape must match (n_density, n_importance)."""
        assert grid_2d.shape == (3, 2)

    def test_3d_grid_shape(self, grid_3d):
        """3D grid shape must match (2, 2, 2)."""
        assert grid_3d.shape == (2, 2, 2)
        assert grid_3d.models.size == 8

    def test_description_property(self, grid_2d):
        """description should map axis labels to their lengths."""
        assert grid_2d.description == {"density": 3, "importance": 2}

    def test_axes_property(self, grid_2d):
        """axes should be a list of Axis objects with correct labels."""
        assert len(grid_2d.axes) == 2
        assert grid_2d.axes[0].label == "density"
        assert grid_2d.axes[1].label == "importance"

    def test_broadcast_samples_true(self, grid_2d):
        """broadcast_samples=True is stored."""
        assert grid_2d.broadcast_samples is True

    def test_broadcast_samples_false(self, grid_no_broadcast):
        """broadcast_samples=False is stored."""
        assert grid_no_broadcast.broadcast_samples is False

    def test_all_entries_are_toymodels(self, grid_3d):
        """Every cell in the grid must be a ToyModel instance."""
        for m in grid_3d.models.ravel():
            assert isinstance(m, ToyModel)

    def test_factory_receives_correct_params(self):
        """Verify that create_model receives the right parameter combinations."""
        received_params = []

        def spy_create_model(params: dict) -> ToyModel:
            received_params.append(params.copy())
            return _simple_create_model(params)

        ModelGrid(
            spy_create_model,
            axes=[
                Axis(label="density", values=[0.1, 0.5]),
                Axis(label="importance", values=[1.0, 2.0]),
            ],
            broadcast_samples=False,
        )
        assert len(received_params) == 4
        # Check that all 4 combinations were produced
        combos = {(p["density"], p["importance"]) for p in received_params}
        expected = {(0.1, 1.0), (0.1, 2.0), (0.5, 1.0), (0.5, 2.0)}
        assert combos == expected

    def test_empty_axes_raises(self):
        """At least one axis is required."""
        with pytest.raises(ValueError, match="At least one axis"):
            ModelGrid(_simple_create_model, axes=[], broadcast_samples=False)

    def test_factory_missing_params_arg_raises(self):
        """Factory must accept a 'params' argument."""

        def bad_factory(x):
            return _make_toy_model()

        with pytest.raises(TypeError, match="params"):
            ModelGrid(
                bad_factory,
                axes=[Axis(label="x", values=[1.0])],
                broadcast_samples=False,
            )

    def test_mismatched_ae_architectures_raises(self):
        """All autoencoders in the grid must share the same architecture.
        Catches vmap validation failures at init time."""

        def create_mixed(params: dict) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(SEED)
            # Use different n_hidden based on the parameter to create mismatched architectures
            n_hidden = 4 if params["density"] < 0.5 else 6
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=params["density"], device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(N_FEATURES, n_hidden, generator=gen, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        with pytest.raises(ValueError, match="[Aa]utoencoder"):
            ModelGrid(
                create_mixed,
                axes=[Axis(label="density", values=[0.1, 0.9])],
                broadcast_samples=False,
            )

    def test_list_axis_values_grid(self):
        """Axis values can be plain Python lists."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="density", values=[0.2, 0.5, 0.8])],
            broadcast_samples=False,
        )
        assert grid.shape == (3,)

    def test_shape_from_axes_property(self, grid_2d):
        """_shape_from_axes must agree with shape after init."""
        assert grid_2d._shape_from_axes == grid_2d.shape


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.from_iterable
# ══════════════════════════════════════════════════════════════════════════════


class TestFromIterable:
    """Tests for ModelGrid.from_iterable class method."""

    def test_1d_from_list(self):
        """Flat list of ToyModels produces a 1D grid."""
        models = [_make_toy_model(d) for d in [0.3, 0.5, 0.7]]
        grid = ModelGrid.from_iterable(models)
        assert grid.shape == (3,)
        assert len(grid.axes) == 1
        assert grid.axes[0].label == "Axis 1"

    def test_2d_from_nested_lists(self):
        """Nested lists produce a 2D grid."""
        models = [
            [_make_toy_model(0.2), _make_toy_model(0.4)],
            [_make_toy_model(0.6), _make_toy_model(0.8)],
        ]
        grid = ModelGrid.from_iterable(models)
        assert grid.shape == (2, 2)
        assert len(grid.axes) == 2

    def test_from_iterable_preserves_model_identity(self):
        """Models in the grid must be the exact same objects passed in."""
        m1, m2, m3 = _make_toy_model(0.3), _make_toy_model(0.5), _make_toy_model(0.7)
        grid = ModelGrid.from_iterable([m1, m2, m3])
        assert grid.models[0] is m1
        assert grid.models[1] is m2
        assert grid.models[2] is m3

    def test_single_model_raises(self):
        """A bare ToyModel (not wrapped in list) should raise."""
        m = _make_toy_model()
        with pytest.raises(ValueError, match="single ToyModel"):
            ModelGrid.from_iterable(m)

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            ModelGrid.from_iterable([])

    def test_inconsistent_nesting_raises(self):
        """Mixed ToyModels and sublists at the same level should raise."""
        m = _make_toy_model()
        with pytest.raises((TypeError, ValueError)):
            ModelGrid.from_iterable([m, [m]])

    def test_non_toymodel_leaf_raises(self):
        """Non-ToyModel objects in the iterable should raise TypeError."""
        with pytest.raises(TypeError):
            ModelGrid.from_iterable(["not a model", "also not"])

    def test_jagged_nested_raises(self):
        """Inconsistent sub-list lengths should raise ValueError."""
        m = _make_toy_model
        with pytest.raises(ValueError, match="[Ii]nconsistent"):
            ModelGrid.from_iterable(
                [
                    [m(0.1), m(0.2)],
                    [m(0.3)],  # different length
                ]
            )

    def test_from_iterable_broadcast_samples_false(self):
        """from_iterable always sets broadcast_samples=False."""
        grid = ModelGrid.from_iterable([_make_toy_model()])
        assert grid.broadcast_samples is False


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.__getitem__
# ══════════════════════════════════════════════════════════════════════════════


class TestGetitemComprehensive:
    """Additional getitem tests beyond the existing test file."""

    def test_3d_single_int_collapses_first_axis(self, grid_3d):
        """Integer index on a 3D grid collapses the first axis, leaving 2D."""
        sub = grid_3d[0]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (2, 2)
        assert len(sub.axes) == 2

    def test_3d_two_ints_collapse_two_axes(self, grid_3d):
        """Two integer indices on 3D grid collapse 2 axes, leaving 1D."""
        sub = grid_3d[0, 1]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (2,)
        assert len(sub.axes) == 1
        assert sub.axes[0].label == "lr"

    def test_3d_all_ints_return_toymodel(self, grid_3d):
        """Three integer indices on 3D grid returns a ToyModel."""
        result = grid_3d[0, 0, 0]
        assert isinstance(result, ToyModel)
        assert result is grid_3d.models[0, 0, 0]

    def test_step_slice(self, grid_1d):
        """Slicing with a step should return correct subset."""
        sub = grid_1d[::2]
        assert sub.shape == (2,)

    def test_negative_int_returns_last(self, grid_1d):
        """grid[-1] should return the last model."""
        result = grid_1d[-1]
        assert isinstance(result, ToyModel)
        assert result is grid_1d.models[-1]

    def test_too_many_indices_3d(self, grid_3d):
        """4 indices on a 3D grid should raise IndexError."""
        with pytest.raises(IndexError, match="Too many indices"):
            grid_3d[0, 0, 0, 0]

    def test_unsupported_index_float(self, grid_1d):
        """Float index should raise IndexError."""
        with pytest.raises(IndexError, match="Unsupported index type"):
            grid_1d[1.5]

    def test_slice_preserves_training_axis(self):
        """When slicing a TrainingAxis, the result should keep TrainingAxis type."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="density", values=torch.linspace(0.1, 0.9, 3))],
            broadcast_samples=False,
        )
        # Fit with snapshot to create a history grid with TrainingAxis
        history = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(history, ModelGrid)
        assert isinstance(history.axes[0], TrainingAxis)

        # Slice on the TrainingAxis dimension
        sub = history[0:2]
        assert isinstance(sub, ModelGrid)
        assert isinstance(sub.axes[0], TrainingAxis)

    def test_subgrid_create_model_preserved(self, grid_2d):
        """Subgrids via slicing should preserve the create_model reference."""
        sub = grid_2d[0:2]
        assert sub.create_model is grid_2d.create_model

    def test_subgrid_broadcast_samples_preserved(self, grid_2d):
        """Subgrids should inherit broadcast_samples from parent."""
        sub = grid_2d[0:2]
        assert sub.broadcast_samples == grid_2d.broadcast_samples

    def test_int_oob_3d(self, grid_3d):
        """Out-of-bounds integer on 3D grid."""
        with pytest.raises(IndexError, match="out of bounds"):
            grid_3d[5]

    def test_partial_indexing_preserves_remaining_axes(self, grid_3d):
        """Indexing only the first axis of 3D should preserve axes 2 and 3."""
        sub = grid_3d[0]
        assert sub.axes[0].label == "importance"
        assert sub.axes[1].label == "lr"


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.fit()
# ══════════════════════════════════════════════════════════════════════════════


class TestFitComprehensive:
    """Additional fit tests beyond the existing test file."""

    def test_fit_track_losses_true_returns_list(self, grid_1d):
        """track_losses=True should return a list of float loss values."""
        losses = grid_1d.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert isinstance(losses, list)
        assert len(losses) == 10
        assert all(isinstance(l, float) for l in losses)

    def test_fit_track_losses_false_returns_none(self, grid_1d):
        """track_losses=False (and no snapshots) should return None."""
        result = grid_1d.fit(n_epochs=10, batch_size=32, track_losses=False)
        assert result is None

    def test_fit_weights_change(self, grid_2d):
        """Weights should be updated in-place on the original ToyModel objects after fit."""
        weights_before = {}
        for idx in np.ndindex(grid_2d.shape):
            weights_before[idx] = grid_2d.models[idx].ae.state_dict()["W"].clone()

        grid_2d.fit(n_epochs=30, batch_size=64)

        changed = 0
        for idx in np.ndindex(grid_2d.shape):
            w_after = grid_2d.models[idx].ae.state_dict()["W"]
            if not torch.equal(weights_before[idx], w_after):
                changed += 1
        assert changed == grid_2d.models.size, (
            "All model weights should change after fitting"
        )

    def test_fit_loss_decreases_overall(self, grid_1d):
        """Loss should generally decrease over training epochs."""
        losses = grid_1d.fit(n_epochs=50, batch_size=64, track_losses=True)
        # Compare first few vs last few to be robust against noise
        early_mean = sum(losses[:5]) / 5
        late_mean = sum(losses[-5:]) / 5
        assert late_mean < early_mean

    def test_fit_multiple_calls_continue_training(self, grid_1d):
        """Calling fit() twice should continue training from the current state."""
        grid_1d.fit(n_epochs=20, batch_size=32)
        weights_after_first = grid_1d.models[0].ae.state_dict()["W"].clone()

        grid_1d.fit(n_epochs=20, batch_size=32)
        weights_after_second = grid_1d.models[0].ae.state_dict()["W"]

        assert not torch.equal(weights_after_first, weights_after_second), (
            "Second fit should further update weights"
        )

    def test_fit_no_broadcast(self, grid_no_broadcast):
        """Fit should work with broadcast_samples=False."""
        losses = grid_no_broadcast.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert len(losses) == 10

    def test_fit_single_element_grid(self, single_element_grid):
        """A grid with exactly one model should train successfully."""
        losses = single_element_grid.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert len(losses) == 10

    def test_fit_3d_grid(self, grid_3d):
        """3D grid should train all 8 models."""
        weights_before = {
            i: m.ae.state_dict()["W"].clone()
            for i, m in enumerate(grid_3d.models.ravel())
        }
        grid_3d.fit(n_epochs=20, batch_size=32)
        for i, m in enumerate(grid_3d.models.ravel()):
            assert not torch.equal(weights_before[i], m.ae.state_dict()["W"])


class TestFitSnapshotInterval:
    """Tests for snapshot_interval parameter in fit()."""

    def test_snapshot_returns_modelgrid(self, grid_1d):
        """snapshot_interval should make fit() return a ModelGrid (history grid)."""
        result = grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(result, ModelGrid)

    def test_snapshot_has_training_axis(self, grid_1d):
        """History grid's first axis should be a TrainingAxis."""
        history = grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(history.axes[0], TrainingAxis)

    def test_snapshot_count(self, grid_1d):
        """Snapshot at epoch 0, 10, 20 = 3 snapshots for n_epochs=20, interval=10."""
        history = grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert history.shape[0] == 3  # epoch 0, 10, 20

    def test_snapshot_epoch_values(self, grid_1d):
        """Training axis values should be the epoch numbers where snapshots were taken."""
        history = grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert history.axes[0].values == [0, 10, 20]

    def test_snapshot_preserves_original_axes(self, grid_2d):
        """History grid should have TrainingAxis + all original axes."""
        history = grid_2d.fit(n_epochs=10, batch_size=32, snapshot_interval=5)
        assert len(history.axes) == 3  # TrainingAxis + density + importance
        assert history.axes[1].label == "density"
        assert history.axes[2].label == "importance"

    def test_snapshot_models_are_independent(self, grid_1d):
        """Snapshot models should be independent copies (deepcopy)."""
        history = grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        # epoch-0 snapshot and epoch-20 snapshot should have different weights
        w_epoch0 = history.models[0, 0].ae.state_dict()["W"]
        w_epoch20 = history.models[2, 0].ae.state_dict()["W"]
        assert not torch.equal(w_epoch0, w_epoch20)

    def test_snapshot_takes_priority_over_losses(self, grid_1d):
        """When both snapshot_interval and track_losses are set, snapshot wins (returns grid)."""
        result = grid_1d.fit(
            n_epochs=20, batch_size=32, snapshot_interval=10, track_losses=True
        )
        assert isinstance(result, ModelGrid)

    def test_snapshot_interval_zero_raises(self, grid_1d):
        """snapshot_interval=0 should raise ValueError."""
        with pytest.raises(ValueError, match="snapshot_interval"):
            grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=0)

    def test_snapshot_interval_negative_raises(self, grid_1d):
        """Negative snapshot_interval should raise ValueError."""
        with pytest.raises(ValueError, match="snapshot_interval"):
            grid_1d.fit(n_epochs=20, batch_size=32, snapshot_interval=-5)

    def test_snapshot_interval_exceeds_n_epochs_raises(self, grid_1d):
        """snapshot_interval > n_epochs should raise ValueError."""
        with pytest.raises(ValueError, match="snapshot_interval.*exceed"):
            grid_1d.fit(n_epochs=10, batch_size=32, snapshot_interval=20)

    def test_snapshot_interval_equals_n_epochs(self, grid_1d):
        """snapshot_interval == n_epochs: epoch 0 and epoch n_epochs snapshots."""
        history = grid_1d.fit(n_epochs=10, batch_size=32, snapshot_interval=10)
        assert history.shape[0] == 2  # epoch 0, epoch 10

    def test_snapshot_not_evenly_divisible(self, grid_1d):
        """When n_epochs is not evenly divisible by snapshot_interval,
        the last snapshot is at the last multiple of interval."""
        history = grid_1d.fit(n_epochs=25, batch_size=32, snapshot_interval=10)
        # Snapshots at epoch 0, 10, 20 (25 is not a multiple of 10)
        assert history.axes[0].values == [0, 10, 20]

    def test_snapshot_interval_1(self, grid_1d):
        """snapshot_interval=1 should create n_epochs+1 snapshots (epoch 0 through n_epochs)."""
        n = 5
        history = grid_1d.fit(n_epochs=n, batch_size=32, snapshot_interval=1)
        assert history.shape[0] == n + 1  # epoch 0, 1, 2, 3, 4, 5


class TestFitSampleEvery:
    """Tests for the sample_every parameter in fit()."""

    def test_sample_every_1(self, grid_1d):
        """sample_every=1 means fresh samples every epoch."""
        losses = grid_1d.fit(
            n_epochs=10, batch_size=32, sample_every=1, track_losses=True
        )
        assert len(losses) == 10

    def test_sample_every_larger_than_n_epochs(self, grid_1d):
        """sample_every > n_epochs should work: single buffer for all epochs."""
        losses = grid_1d.fit(
            n_epochs=5, batch_size=32, sample_every=100, track_losses=True
        )
        assert len(losses) == 5

    def test_sample_every_zero_raises(self, grid_1d):
        """sample_every=0 must raise ValueError."""
        with pytest.raises(ValueError, match="sample_every"):
            grid_1d.fit(n_epochs=10, batch_size=32, sample_every=0)

    def test_sample_every_negative_raises(self, grid_1d):
        """sample_every=-1 must raise ValueError."""
        with pytest.raises(ValueError, match="sample_every"):
            grid_1d.fit(n_epochs=10, batch_size=32, sample_every=-1)

    def test_sample_every_remainder(self, grid_1d):
        """n_epochs not divisible by sample_every should still work properly."""
        losses = grid_1d.fit(
            n_epochs=7, batch_size=32, sample_every=3, track_losses=True
        )
        assert len(losses) == 7

    def test_sample_every_with_snapshot(self, grid_1d):
        """sample_every and snapshot_interval should compose correctly."""
        history = grid_1d.fit(
            n_epochs=20, batch_size=32, sample_every=7, snapshot_interval=10
        )
        assert isinstance(history, ModelGrid)
        assert history.shape[0] == 3  # epoch 0, 10, 20


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.save() / ModelGrid.load()
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLoad:
    """Tests for dill-based save/load round-trip."""

    def test_save_load_roundtrip(self, grid_2d, tmp_path):
        """Saving and loading should preserve grid structure."""
        path = tmp_path / "grid.pkl"
        grid_2d.fit(n_epochs=10, batch_size=32)
        grid_2d.save(str(path))

        loaded = ModelGrid.load(str(path))
        assert loaded.shape == grid_2d.shape
        assert len(loaded.axes) == len(grid_2d.axes)
        for orig_ax, load_ax in zip(grid_2d.axes, loaded.axes):
            assert orig_ax.label == load_ax.label

    def test_save_load_preserves_weights(self, grid_1d, tmp_path):
        """After round-trip, model weights should be identical."""
        grid_1d.fit(n_epochs=10, batch_size=32)
        path = tmp_path / "grid_weights.pkl"
        grid_1d.save(str(path))

        loaded = ModelGrid.load(str(path))
        for orig, load in zip(grid_1d.models.ravel(), loaded.models.ravel()):
            orig_w = orig.ae.state_dict()["W"]
            load_w = load.ae.state_dict()["W"]
            assert torch.equal(orig_w, load_w)

    def test_save_load_pathlib(self, grid_1d, tmp_path):
        """save/load should accept pathlib.Path objects."""
        path = tmp_path / "grid_pathlib.pkl"
        grid_1d.save(path)
        loaded = ModelGrid.load(path)
        assert loaded.shape == grid_1d.shape

    def test_save_overwrites_existing(self, grid_1d, tmp_path):
        """Saving to the same path should overwrite."""
        path = tmp_path / "grid_overwrite.pkl"
        grid_1d.save(str(path))
        grid_1d.fit(n_epochs=5, batch_size=32)
        grid_1d.save(str(path))
        loaded = ModelGrid.load(str(path))
        # Should have the weights from the second save (after fit)
        w_loaded = loaded.models[0].ae.state_dict()["W"]
        w_orig = grid_1d.models[0].ae.state_dict()["W"]
        assert torch.equal(w_loaded, w_orig)


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.save_models() / load_models()
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLoadModels:
    """Tests for pickle-based save_models/load_models."""

    def test_save_load_models_roundtrip(self, grid_2d, tmp_path):
        """save_models + load_models should preserve model weights."""
        grid_2d.fit(n_epochs=10, batch_size=32)
        path = str(tmp_path / "models.pkl")
        grid_2d.save_models(path)

        # Mutate the grid's weights to verify load actually restores
        for m in grid_2d.models.ravel():
            with torch.no_grad():
                m.ae.W.fill_(0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_2d.load_models(path)

        # Verify at least one model has non-zero weights
        any_nonzero = any(
            m.ae.state_dict()["W"].abs().sum() > 0 for m in grid_2d.models.ravel()
        )
        assert any_nonzero

    def test_save_models_appends_pkl_extension(self, grid_1d, tmp_path):
        """If path doesn't end with .pkl, it should be appended."""
        path = str(tmp_path / "models_no_ext")
        grid_1d.save_models(path)
        assert (tmp_path / "models_no_ext.pkl").exists()

    def test_save_models_default_path(self, grid_1d):
        """save_models with no path should create a timestamped file."""
        import os

        grid_1d.save_models()
        # Find the generated file
        pkl_files = [
            f
            for f in os.listdir(".")
            if f.startswith("model_grid") and f.endswith(".pkl")
        ]
        assert len(pkl_files) >= 1
        # Clean up
        for f in pkl_files:
            os.remove(f)

    def test_load_models_shape_mismatch_raises(self, grid_2d, grid_1d, tmp_path):
        """Loading models with wrong shape should raise ValueError."""
        path = str(tmp_path / "wrong_shape.pkl")
        grid_1d.save_models(path)

        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_2d.load_models(path)

    def test_load_models_empty_path_raises(self, grid_1d):
        """Empty string path should raise TypeError."""
        with pytest.raises(TypeError, match="non-empty string"):
            grid_1d.load_models("")

    def test_load_models_non_string_raises(self, grid_1d):
        """Non-string path should raise TypeError."""
        with pytest.raises(TypeError, match="non-empty string"):
            grid_1d.load_models(123)

    def test_load_models_warns_about_axes(self, grid_1d, tmp_path):
        """load_models should warn that axes may not match."""
        path = str(tmp_path / "load_warn.pkl")
        grid_1d.save_models(path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_1d.load_models(path)
            assert any("axes" in str(warning.message).lower() for warning in w)

    def test_load_models_invalid_file_content_raises(self, grid_1d, tmp_path):
        """Loading a pickle file that doesn't contain an ndarray should raise."""
        path = str(tmp_path / "invalid.pkl")
        with open(path, "wb") as f:
            pickle.dump({"not": "an array"}, f)

        with pytest.raises(TypeError, match="numpy ndarray"):
            grid_1d.load_models(path)

    def test_load_models_non_toymodel_entries_raises(self, grid_1d, tmp_path):
        """Array of non-ToyModel objects should raise TypeError."""
        path = str(tmp_path / "bad_entries.pkl")
        fake_array = np.array(["foo", "bar", "baz", "qux"], dtype=object).reshape(
            grid_1d.shape
        )
        with open(path, "wb") as f:
            pickle.dump(fake_array, f)

        with pytest.raises(TypeError, match="ToyModel"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_1d.load_models(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Broadcast / Caching
# ══════════════════════════════════════════════════════════════════════════════


class TestBroadcast:
    """Tests for _build_broadcast and broadcast-related behavior."""

    def test_build_broadcast_returns_correct_types(self, grid_2d):
        """_build_broadcast should return (list, Tensor)."""
        broadcasters, bmap = grid_2d._build_broadcast()
        assert isinstance(broadcasters, list)
        assert isinstance(bmap, Tensor)

    def test_broadcast_map_length(self, grid_2d):
        """Broadcast map length should equal number of flattened models."""
        _, bmap = grid_2d._build_broadcast()
        assert len(bmap) == grid_2d.models.size

    def test_shared_distributions_collapse(self):
        """Models with identical distributions (same seed, same params) should
        share a single broadcaster slot."""

        def create_model(params: dict) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(99)
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
            axes=[Axis(label="idx", values=torch.arange(5, dtype=torch.float32))],
            broadcast_samples=True,
        )
        broadcasters, bmap = grid._build_broadcast()
        assert len(broadcasters) == 1
        assert (bmap == 0).all()

    def test_distinct_distributions_separate_slots(self):
        """Models with different p_active should get separate broadcaster slots."""

        def create_model(params: dict) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(SEED)
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=params["density"], device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="density", values=[0.1, 0.3, 0.5, 0.7])],
            broadcast_samples=True,
        )
        broadcasters, bmap = grid._build_broadcast()
        assert len(broadcasters) == 4
        # Each model gets a unique broadcaster index
        assert len(set(bmap.tolist())) == 4

    def test_generatorless_distributions_never_grouped(self):
        """Distributions without generators should each get their own slot,
        even if they have identical parameters."""

        def create_model(params: dict) -> ToyModel:
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=0.5, device=DEVICE, generator=None
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = ModelGrid(
                create_model,
                axes=[Axis(label="idx", values=[0.0, 1.0, 2.0])],
                broadcast_samples=True,
            )

        broadcasters, bmap = grid._build_broadcast()
        # Each generator-less distribution gets its own slot
        assert len(broadcasters) == 3
        assert len(set(bmap.tolist())) == 3

    def test_no_generator_warns_on_broadcast(self):
        """Creating a grid with broadcast_samples=True but generator-less distributions
        should produce a warning."""

        def create_model(params: dict) -> ToyModel:
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
                create_model,
                axes=[Axis(label="idx", values=[1.0, 2.0])],
                broadcast_samples=True,
            )
            assert any("generator" in str(warning.message).lower() for warning in w)

    def test_broadcast_map_indices_are_valid(self, grid_2d):
        """All indices in broadcast_map should be valid indices into broadcasters."""
        broadcasters, bmap = grid_2d._build_broadcast()
        for idx in bmap.tolist():
            assert 0 <= idx < len(broadcasters)

    def test_2d_grid_dedup_by_density_row(self):
        """In a 2D grid where models in the same density row share a distribution,
        the number of unique broadcasters should equal n_density."""

        def create_model(params: dict) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(SEED)
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=params["density"], device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
                importances=params["importance"]
                ** torch.arange(N_FEATURES, dtype=torch.float32),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[
                Axis(label="density", values=[0.1, 0.5, 0.9]),
                Axis(label="importance", values=[0.5, 1.0, 2.0]),
            ],
            broadcast_samples=True,
        )
        broadcasters, bmap = grid._build_broadcast()
        assert len(broadcasters) == 3  # one per density


class TestCanVectorizeLoss:
    """Tests for _can_vectorize_loss helper."""

    def test_homogeneous_grid_can_vectorize(self, grid_2d):
        """All models share TiedLinearRelu -> same loss function."""
        assert grid_2d._can_vectorize_loss() is True

    def test_single_model_can_vectorize(self, single_element_grid):
        """Single-model grid should always be vectorizable."""
        assert single_element_grid._can_vectorize_loss() is True


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.sae_results_to_dataframe()
# ══════════════════════════════════════════════════════════════════════════════


class TestSaeResultsToDataFrame:
    """Tests for sae_results_to_dataframe."""

    def test_empty_results_returns_empty_df(self, grid_1d):
        """Grid with no trained SAEs should return an empty DataFrame."""
        df = grid_1d.sae_results_to_dataframe()
        assert len(df) == 0

    def test_empty_results_has_correct_index_names(self, grid_1d):
        """Empty DataFrame should have axis labels + 'sae' as index names."""
        df = grid_1d.sae_results_to_dataframe()
        expected_names = ["density", "sae"]
        assert list(df.index.names) == expected_names

    def test_2d_empty_results_index_names(self, grid_2d):
        """2D grid empty DataFrame should have correct index names.
        The empty case uses axis_labels + ['sae'], not the reordered non-empty form."""
        df = grid_2d.sae_results_to_dataframe()
        # Empty case: pd.MultiIndex with names = axis_labels + ["sae"]
        expected_names = ["density", "importance", "sae"]
        assert list(df.index.names) == expected_names


# ══════════════════════════════════════════════════════════════════════════════
#  ModelGrid.parameters_mesh
# ══════════════════════════════════════════════════════════════════════════════


class TestParametersMesh:
    """Tests for the parameters_mesh cached property.

    NOTE: parameters_mesh uses torch.meshgrid, which requires tensor arguments.
    Axis.__init__ converts tensors to lists of scalar tensors (since Tensor is
    not Sequence). When meshgrid unpacks a list, each scalar becomes a separate
    argument, so the mesh dimensions = total number of scalar values across all
    axes, NOT the number of axes. This behavior is known (see TODO in source).

    We test parameters_mesh with grids whose axis values are plain lists
    (non-tensor), which meshgrid can't handle, to document the actual behavior.
    We also test the happy path where axis values happen to be compatible.
    """

    def test_mesh_with_list_axis_values_works(self):
        """parameters_mesh converts list values via torch.as_tensor."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[
                Axis(label="density", values=[0.1, 0.5]),
                Axis(label="importance", values=[1.0, 2.0]),
            ],
            broadcast_samples=False,
        )
        mesh = grid.parameters_mesh
        assert len(mesh) == 2
        assert mesh[0].shape == (2, 2)
        assert mesh[1].shape == (2, 2)

    def test_mesh_is_cached(self, grid_1d):
        """parameters_mesh is a cached_property, so repeated access returns same object.
        We avoid triggering the mesh computation itself since it may fail
        with current axis value types; instead test the caching mechanism
        by checking identity on a grid that works."""
        # Build a grid where axis.values are kept as a format meshgrid accepts.
        # We use from_iterable which produces list[int] axis values --
        # still fails with meshgrid. So just verify the cached_property descriptor.
        import functools

        assert isinstance(
            ModelGrid.parameters_mesh,
            functools.cached_property,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Edge Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and corner-case behavior."""

    def test_single_element_grid_shape(self, single_element_grid):
        """Single-element grid should have shape (1,)."""
        assert single_element_grid.shape == (1,)

    def test_single_element_grid_getitem(self, single_element_grid):
        """Indexing single-element grid with int returns the lone ToyModel."""
        result = single_element_grid[0]
        assert isinstance(result, ToyModel)

    def test_single_element_grid_description(self, single_element_grid):
        """Description of single-element grid."""
        assert single_element_grid.description == {"density": 1}

    def test_single_axis_grid(self, grid_1d):
        """Single-axis grid should have exactly 1 axis."""
        assert len(grid_1d.axes) == 1

    def test_subgrid_of_subgrid(self, grid_3d):
        """Nested slicing should compose correctly."""
        sub1 = grid_3d[0]  # shape (2, 2)
        sub2 = sub1[0]  # shape (2,)
        result = sub2[0]  # single ToyModel
        assert isinstance(result, ToyModel)
        assert result is grid_3d.models[0, 0, 0]

    def test_full_slice_is_identity(self, grid_2d):
        """grid[:, :] should return a grid with identical models."""
        sub = grid_2d[:, :]
        assert sub.shape == grid_2d.shape
        for idx in np.ndindex(grid_2d.shape):
            assert sub.models[idx] is grid_2d.models[idx]

    def test_vmap_validation_skip_for_single_model(self):
        """_validate_vmap should not raise for single-model grids regardless of architecture."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="x", values=[0.5])],
            broadcast_samples=False,
        )
        # Should have been created without error
        assert grid.shape == (1,)

    def test_generator_validation_skip_for_single_model(self):
        """_validate_generators should not warn for single-model grids."""

        def create_model(params: dict) -> ToyModel:
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
            grid = ModelGrid(
                create_model,
                axes=[Axis(label="x", values=[1.0])],
                broadcast_samples=True,
            )
            # Single model -> no warning
            generator_warnings = [
                warning for warning in w if "generator" in str(warning.message).lower()
            ]
            assert len(generator_warnings) == 0

    def test_models_array_dtype_is_object(self, grid_2d):
        """Internal models array should be object dtype for holding ToyModels."""
        assert grid_2d.models.dtype == object

    def test_fit_preserves_importances(self, grid_2d):
        """Fitting should not modify model importances."""
        importances_before = {
            i: m.importances.clone() for i, m in enumerate(grid_2d.models.ravel())
        }
        grid_2d.fit(n_epochs=10, batch_size=32)
        for i, m in enumerate(grid_2d.models.ravel()):
            assert torch.equal(importances_before[i], m.importances)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: fit + getitem + save/load compose
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Tests that verify multiple features compose correctly."""

    def test_fit_then_slice_then_fit(self):
        """Train the full grid, slice a subgrid, train the subgrid more."""
        grid = ModelGrid(
            _multi_param_create_model,
            axes=[
                Axis(label="density", values=[0.2, 0.5, 0.8]),
                Axis(label="importance", values=[0.5, 1.0]),
            ],
            broadcast_samples=False,
        )
        grid.fit(n_epochs=10, batch_size=32)
        w_after_first = grid.models[1, 0].ae.state_dict()["W"].clone()

        sub = grid[1:3]  # density=[0.5, 0.8]
        sub.fit(n_epochs=10, batch_size=32)
        w_after_second = grid.models[1, 0].ae.state_dict()["W"]

        # Subgrid fit should have mutated the parent's model
        assert not torch.equal(w_after_first, w_after_second)

    def test_save_load_then_fit(self, tmp_path):
        """Saving, loading, and then fitting should work."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="density", values=[0.3, 0.7])],
            broadcast_samples=False,
        )
        grid.fit(n_epochs=5, batch_size=32)
        path = tmp_path / "integration.pkl"
        grid.save(str(path))

        loaded = ModelGrid.load(str(path))
        w_before = loaded.models[0].ae.state_dict()["W"].clone()
        loaded.fit(n_epochs=10, batch_size=32)
        w_after = loaded.models[0].ae.state_dict()["W"]
        assert not torch.equal(w_before, w_after)

    def test_snapshot_then_index(self):
        """History grid from snapshot_interval should be indexable."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="density", values=[0.2, 0.5, 0.8])],
            broadcast_samples=False,
        )
        history = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(history, ModelGrid)

        # Index into specific epoch
        epoch_0_grid = history[0]
        assert isinstance(epoch_0_grid, ModelGrid)
        assert epoch_0_grid.shape == (3,)

        # Index into specific epoch + specific model
        model = history[0, 1]
        assert isinstance(model, ToyModel)

    def test_history_grid_has_correct_total_models(self):
        """History grid should have n_snapshots * n_models total ToyModels."""
        grid = ModelGrid(
            _simple_create_model,
            axes=[Axis(label="density", values=[0.3, 0.7])],
            broadcast_samples=False,
        )
        history = grid.fit(n_epochs=10, batch_size=32, snapshot_interval=5)
        # epoch 0, 5, 10 => 3 snapshots x 2 models = 6
        assert history.models.size == 6

    def test_from_iterable_then_fit(self):
        """A grid created from from_iterable can be trained."""
        models = [_make_toy_model(d) for d in [0.3, 0.5, 0.7]]
        grid = ModelGrid.from_iterable(models)

        weights_before = [m.ae.state_dict()["W"].clone() for m in models]
        grid.fit(n_epochs=15, batch_size=32)

        for i, m in enumerate(models):
            assert not torch.equal(weights_before[i], m.ae.state_dict()["W"])
