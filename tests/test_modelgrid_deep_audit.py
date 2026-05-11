"""Deep audit tests for ModelGrid.

Systematically tests construction, fit(), broadcast caching, snapshots,
__getitem__, save/load, train_saes/evaluate_saes, and edge cases.

Discovered bugs are noted with comments and severity markers.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Generator, Tensor

from occhio.autoencoders import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid, TrainingAxis
from occhio.toy_model import ToyModel

# ── Constants ────────────────────────────────────────────────────────────────

N_FEATURES = 5
N_HIDDEN = 3
DEVICE = "cpu"
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_model(density: float = 0.5, seed: int = SEED) -> ToyModel:
    """Stand-alone ToyModel with a seeded generator."""
    gen = Generator(device=DEVICE).manual_seed(seed)
    return ToyModel(
        distribution=SparseUniform(
            N_FEATURES, p_active=density, device=DEVICE, generator=gen
        ),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
        importances=torch.ones(N_FEATURES),
        device=DEVICE,
    )


def _create_model_fn(params: dict) -> ToyModel:
    """Factory for ModelGrid: reads 'density' and optional 'importance'."""
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


def _make_grid_1d(n: int = 4, broadcast: bool = True) -> ModelGrid:
    return ModelGrid(
        _create_model_fn,
        axes=[Axis(label="density", values=[0.1, 0.3, 0.5, 0.7][:n])],
        broadcast_samples=broadcast,
    )


def _make_grid_2d(
    n_density: int = 3, n_importance: int = 2, broadcast: bool = True
) -> ModelGrid:
    densities = [0.1, 0.3, 0.5, 0.7, 0.9][:n_density]
    importances = [0.5, 1.0, 2.0, 3.0][:n_importance]
    return ModelGrid(
        _create_model_fn,
        axes=[
            Axis(label="density", values=densities),
            Axis(label="importance", values=importances),
        ],
        broadcast_samples=broadcast,
    )


def _make_grid_3d() -> ModelGrid:
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

    return ModelGrid(
        create_model,
        axes=[
            Axis(label="density", values=[0.3, 0.7]),
            Axis(label="importance", values=[0.5, 2.0]),
            Axis(label="lr", values=[1e-3, 3e-4]),
        ],
        broadcast_samples=False,
    )


def _make_shared_dist_grid(n: int = 3, broadcast: bool = True) -> ModelGrid:
    """All models share the same distribution (same seed, same p_active)."""

    def create_model(params: dict) -> ToyModel:
        gen = Generator(device=DEVICE).manual_seed(SEED)
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
        axes=[Axis(label="idx", values=list(range(n)))],
        broadcast_samples=broadcast,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. Construction Correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_1d_grid_shape_and_models(self):
        grid = _make_grid_1d(n=4)
        assert grid.shape == (4,)
        assert grid.models.shape == (4,)
        for m in grid.models.ravel():
            assert isinstance(m, ToyModel)

    def test_2d_grid_shape_and_models(self):
        grid = _make_grid_2d(n_density=3, n_importance=4)
        assert grid.shape == (3, 4)
        assert grid.models.shape == (3, 4)
        assert grid.models.size == 12

    def test_3d_grid_shape_and_models(self):
        grid = _make_grid_3d()
        assert grid.shape == (2, 2, 2)
        assert grid.models.size == 8
        for m in grid.models.ravel():
            assert isinstance(m, ToyModel)

    def test_create_model_receives_correct_params(self):
        """Verify factory receives every combination of axis values."""
        received = []

        def spy(params: dict) -> ToyModel:
            received.append(params.copy())
            return _create_model_fn(params)

        ModelGrid(
            spy,
            axes=[
                Axis(label="density", values=[0.1, 0.9]),
                Axis(label="importance", values=[0.5, 2.0]),
            ],
            broadcast_samples=False,
        )
        combos = {(p["density"], p["importance"]) for p in received}
        assert combos == {(0.1, 0.5), (0.1, 2.0), (0.9, 0.5), (0.9, 2.0)}

    def test_description_matches_axes(self):
        grid = _make_grid_2d(n_density=3, n_importance=2)
        assert grid.description == {"density": 3, "importance": 2}

    def test_shape_matches_description(self):
        grid = _make_grid_2d(n_density=4, n_importance=3)
        desc = grid.description
        assert grid.shape == tuple(desc.values())


# ══════════════════════════════════════════════════════════════════════════════
#  2. fit() Correctness — THE MOST CRITICAL
# ══════════════════════════════════════════════════════════════════════════════


class TestFitCorrectness:
    def test_all_model_weights_change(self):
        """After fit, ALL models' weights must have changed."""
        grid = _make_grid_2d(n_density=2, n_importance=3)
        before = {
            idx: grid.models[idx].ae.state_dict()["W"].clone()
            for idx in np.ndindex(grid.shape)
        }
        grid.fit(n_epochs=50, batch_size=64)
        for idx in np.ndindex(grid.shape):
            after = grid.models[idx].ae.state_dict()["W"]
            assert not torch.equal(before[idx], after), (
                f"Model at {idx} weights unchanged after 50 epochs"
            )

    def test_losses_are_finite(self):
        grid = _make_grid_2d(n_density=2, n_importance=2)
        losses = grid.fit(n_epochs=30, batch_size=64, track_losses=True)
        for i, loss in enumerate(losses):
            assert np.isfinite(loss), f"Loss at epoch {i} is not finite: {loss}"

    def test_loss_decreases(self):
        grid = _make_grid_2d(n_density=2, n_importance=2)
        losses = grid.fit(n_epochs=100, batch_size=128, track_losses=True)
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, "Loss should decrease over training"

    def test_grid_loss_comparable_to_standalone_model(self):
        """A model trained via ModelGrid should achieve comparable loss to
        the same model trained standalone (via ToyModel.fit)."""
        # Build grid with 1 model
        grid = ModelGrid(
            _create_model_fn,
            axes=[Axis(label="density", values=[0.5])],
            broadcast_samples=False,
        )
        # Clone the model for standalone training
        standalone = deepcopy(grid.models[0])

        grid_losses = grid.fit(
            n_epochs=200, batch_size=128, track_losses=True, learning_rate=3e-4
        )
        standalone_losses, _ = standalone.fit(
            n_epochs=200, batch_size=128, learning_rate=3e-4, track_losses=True
        )

        # Final losses should be in the same ballpark (within 10x)
        grid_final = sum(grid_losses[-10:]) / 10
        standalone_final = sum(standalone_losses[-10:]) / 10
        ratio = max(grid_final, standalone_final) / max(
            min(grid_final, standalone_final), 1e-10
        )
        assert ratio < 10, (
            f"Grid final loss {grid_final:.6f} and standalone {standalone_final:.6f} "
            f"differ by ratio {ratio:.1f}x"
        )

    def test_track_losses_true(self):
        grid = _make_grid_1d(n=2)
        losses = grid.fit(n_epochs=15, batch_size=32, track_losses=True)
        assert isinstance(losses, list)
        assert len(losses) == 15
        assert all(isinstance(x, float) for x in losses)

    def test_track_losses_false(self):
        grid = _make_grid_1d(n=2)
        result = grid.fit(n_epochs=10, batch_size=32, track_losses=False)
        assert result is None

    def test_sample_every_1(self):
        grid = _make_grid_1d(n=2)
        losses = grid.fit(n_epochs=10, batch_size=32, sample_every=1, track_losses=True)
        assert len(losses) == 10

    def test_sample_every_5(self):
        grid = _make_grid_1d(n=2)
        losses = grid.fit(n_epochs=12, batch_size=32, sample_every=5, track_losses=True)
        assert len(losses) == 12

    def test_sample_every_10(self):
        grid = _make_grid_1d(n=2)
        losses = grid.fit(
            n_epochs=10, batch_size=32, sample_every=10, track_losses=True
        )
        assert len(losses) == 10

    def test_sample_every_larger_than_n_epochs(self):
        """sample_every > n_epochs should work: single buffer allocation."""
        grid = _make_grid_1d(n=2)
        losses = grid.fit(
            n_epochs=5, batch_size=32, sample_every=100, track_losses=True
        )
        assert len(losses) == 5


# ══════════════════════════════════════════════════════════════════════════════
#  3. Broadcast Caching Deep Dive
# ══════════════════════════════════════════════════════════════════════════════


class TestBroadcastCaching:
    def test_shared_distributions_collapse_to_one(self):
        grid = _make_shared_dist_grid(n=4, broadcast=True)
        broadcasters, bmap = grid._build_broadcast()
        assert len(broadcasters) == 1
        assert (bmap == 0).all()

    def test_broadcast_map_covers_all_models(self):
        grid = _make_grid_2d(n_density=3, n_importance=2, broadcast=True)
        _, bmap = grid._build_broadcast()
        assert len(bmap) == grid.models.size

    def test_shared_distribution_identical_weights_after_training(self):
        """Models sharing a distribution and starting from the same weights
        must converge to IDENTICAL final weights with broadcast_samples=True."""
        grid = _make_shared_dist_grid(n=3, broadcast=True)
        broadcasters, _ = grid._build_broadcast()
        assert len(broadcasters) == 1

        grid.fit(n_epochs=50, batch_size=64, sample_every=5)

        models = grid.models.ravel()
        ref_W = models[0].ae.state_dict()["W"]
        for i, m in enumerate(models[1:], start=1):
            assert torch.equal(ref_W, m.ae.state_dict()["W"]), (
                f"Model {i} has different weights despite shared distribution"
            )

    def test_without_broadcast_models_generally_differ(self):
        """Without broadcast_samples, models with different seeds should produce
        different final weights because they see different random samples."""

        def create_model_unique_seed(params: dict) -> ToyModel:
            # Each model gets a different seed based on its index
            seed = SEED + int(params["idx"])
            gen = Generator(device=DEVICE).manual_seed(seed)
            return ToyModel(
                distribution=SparseUniform(
                    N_FEATURES, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE),
                importances=torch.ones(N_FEATURES),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model_unique_seed,
            axes=[Axis(label="idx", values=list(range(3)))],
            broadcast_samples=False,
        )
        grid.fit(n_epochs=50, batch_size=64)

        models = grid.models.ravel()
        w0 = models[0].ae.state_dict()["W"]
        any_differ = any(
            not torch.equal(w0, m.ae.state_dict()["W"]) for m in models[1:]
        )
        assert any_differ, (
            "Models with different seeds should produce different weights"
        )

    def test_different_distributions_get_separate_slots(self):
        grid = _make_grid_1d(n=4, broadcast=True)
        broadcasters, bmap = grid._build_broadcast()
        # Each density value is unique -> 4 unique distributions
        assert len(broadcasters) == 4
        assert len(set(bmap.tolist())) == 4

    def test_generatorless_distributions_never_grouped(self):
        """Generator-less distributions must each get their own slot."""

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
                axes=[Axis(label="idx", values=[0, 1, 2])],
                broadcast_samples=True,
            )

        broadcasters, bmap = grid._build_broadcast()
        assert len(broadcasters) == 3  # each gets its own slot

    def test_generator_sync_after_fit(self):
        """After fit with broadcast_samples=True, generators should be synced
        so future sampling from distributions in the same group is identical."""
        grid = _make_shared_dist_grid(n=3, broadcast=True)
        grid.fit(n_epochs=20, batch_size=64, sample_every=5)

        models = grid.models.ravel()
        samples = [m.distribution.sample(128) for m in models]
        for i, s in enumerate(samples[1:], start=1):
            assert torch.equal(samples[0], s), (
                f"After fit+sync, distribution {i} produces different samples"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  4. Snapshot Interval
# ══════════════════════════════════════════════════════════════════════════════


class TestSnapshotInterval:
    def test_returns_modelgrid(self):
        grid = _make_grid_1d(n=2)
        result = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(result, ModelGrid)

    def test_has_training_axis_first(self):
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert isinstance(history.axes[0], TrainingAxis)

    def test_epoch_values_correct(self):
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)
        assert history.axes[0].values == [0, 10, 20]

    def test_epoch_values_not_evenly_divisible(self):
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=25, batch_size=32, snapshot_interval=10)
        # epoch 0, 10, 20 (25 is not a multiple)
        assert history.axes[0].values == [0, 10, 20]

    def test_earlier_snapshots_higher_loss(self):
        """Earlier snapshots should generally have higher reconstruction error."""
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=100, batch_size=128, snapshot_interval=50)
        # history.axes[0].values == [0, 50, 100]
        epoch_0_model = history[0, 0]
        epoch_100_model = history[2, 0]

        # Use a fixed generator for evaluation data so comparison is fair
        eval_gen = Generator(device=DEVICE).manual_seed(999)
        eval_dist = SparseUniform(
            N_FEATURES, p_active=0.5, device=DEVICE, generator=eval_gen
        )
        samples = eval_dist.sample(2048)

        loss_0 = epoch_0_model.ae.loss(
            samples, epoch_0_model.ae(samples)[0], epoch_0_model.importances
        )
        loss_100 = epoch_100_model.ae.loss(
            samples, epoch_100_model.ae(samples)[0], epoch_100_model.importances
        )
        assert loss_0.item() > loss_100.item(), (
            f"Epoch 0 loss ({loss_0.item():.4f}) should be > epoch 100 loss ({loss_100.item():.4f})"
        )

    def test_snapshots_are_independent_copies(self):
        """Modifying a snapshot model should NOT affect other snapshots."""
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=20, batch_size=32, snapshot_interval=10)

        # Get snapshots at epoch 0 and epoch 20
        model_epoch0 = history[0, 0]
        model_epoch20 = history[2, 0]

        w_epoch0_before = model_epoch0.ae.state_dict()["W"].clone()

        # Modify epoch 20 snapshot's weights
        with torch.no_grad():
            model_epoch20.ae.W.fill_(999.0)

        # Epoch 0 snapshot should be unaffected
        w_epoch0_after = model_epoch0.ae.state_dict()["W"]
        assert torch.equal(w_epoch0_before, w_epoch0_after), (
            "Modifying one snapshot should not affect another"
        )

    def test_snapshot_preserves_original_axes(self):
        grid = _make_grid_2d(n_density=2, n_importance=2)
        history = grid.fit(n_epochs=10, batch_size=32, snapshot_interval=5)
        # TrainingAxis + density + importance = 3 axes
        assert len(history.axes) == 3
        assert history.axes[1].label == "density"
        assert history.axes[2].label == "importance"

    def test_snapshot_interval_1(self):
        """snapshot_interval=1 creates n_epochs+1 snapshots."""
        n = 5
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=n, batch_size=32, snapshot_interval=1)
        assert history.shape[0] == n + 1

    def test_snapshot_interval_equals_n_epochs(self):
        grid = _make_grid_1d(n=2)
        history = grid.fit(n_epochs=10, batch_size=32, snapshot_interval=10)
        # epoch 0 + epoch 10 = 2 snapshots
        assert history.shape[0] == 2

    def test_snapshot_interval_zero_raises(self):
        grid = _make_grid_1d(n=2)
        with pytest.raises(ValueError, match="snapshot_interval"):
            grid.fit(n_epochs=10, batch_size=32, snapshot_interval=0)

    def test_snapshot_interval_negative_raises(self):
        grid = _make_grid_1d(n=2)
        with pytest.raises(ValueError, match="snapshot_interval"):
            grid.fit(n_epochs=10, batch_size=32, snapshot_interval=-1)

    def test_snapshot_interval_exceeds_n_epochs_raises(self):
        grid = _make_grid_1d(n=2)
        with pytest.raises(ValueError, match="snapshot_interval.*exceed"):
            grid.fit(n_epochs=10, batch_size=32, snapshot_interval=20)

    def test_snapshot_takes_priority_over_losses(self):
        """When both snapshot_interval and track_losses are set, snapshot wins."""
        grid = _make_grid_1d(n=2)
        result = grid.fit(
            n_epochs=10, batch_size=32, snapshot_interval=5, track_losses=True
        )
        assert isinstance(result, ModelGrid)


# ══════════════════════════════════════════════════════════════════════════════
#  5. __getitem__ Correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestGetitem:
    def test_int_index_gives_subgrid(self):
        grid = _make_grid_2d(n_density=3, n_importance=2)
        sub = grid[0]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (2,)
        assert sub.axes[0].label == "importance"

    def test_slice_index(self):
        grid = _make_grid_2d(n_density=4, n_importance=2)
        sub = grid[1:3]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (2, 2)

    def test_multi_axis_int_slice(self):
        grid = _make_grid_2d(n_density=4, n_importance=3)
        sub = grid[0, 1:3]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (2,)

    def test_negative_indexing(self):
        grid = _make_grid_2d(n_density=4, n_importance=3)
        sub = grid[-1]
        assert isinstance(sub, ModelGrid)
        assert sub.shape == (3,)

    def test_model_identity_preserved(self):
        """grid[i].models should contain the SAME objects as original grid."""
        grid = _make_grid_2d(n_density=3, n_importance=2)
        sub = grid[1]
        for j in range(2):
            assert sub.models[j] is grid.models[1, j]

    def test_all_ints_return_toymodel(self):
        grid = _make_grid_2d(n_density=3, n_importance=2)
        result = grid[1, 0]
        assert isinstance(result, ToyModel)
        assert result is grid.models[1, 0]

    def test_oob_positive_raises(self):
        grid = _make_grid_1d(n=4)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[4]

    def test_oob_negative_raises(self):
        grid = _make_grid_1d(n=4)
        with pytest.raises(IndexError, match="out of bounds"):
            grid[-5]

    def test_too_many_indices_raises(self):
        grid = _make_grid_2d()
        with pytest.raises(IndexError, match="Too many indices"):
            grid[0, 0, 0]

    def test_unsupported_index_type_raises(self):
        grid = _make_grid_1d()
        with pytest.raises(IndexError, match="Unsupported index type"):
            grid["bad"]

    def test_full_slice_preserves_identity(self):
        grid = _make_grid_2d(n_density=3, n_importance=2)
        sub = grid[:, :]
        assert sub.shape == grid.shape
        for idx in np.ndindex(grid.shape):
            assert sub.models[idx] is grid.models[idx]

    def test_3d_nested_getitem(self):
        grid = _make_grid_3d()
        sub1 = grid[0]
        assert sub1.shape == (2, 2)
        sub2 = sub1[1]
        assert sub2.shape == (2,)
        model = sub2[0]
        assert isinstance(model, ToyModel)
        assert model is grid.models[0, 1, 0]

    def test_slice_axis_values_correct(self):
        """Axis values on a subgrid should correspond to the sliced values."""
        grid = _make_grid_2d(n_density=4, n_importance=3)
        sub = grid[1:3]
        # Original density values: [0.1, 0.3, 0.5, 0.7]
        # Sliced: [0.3, 0.5]
        expected = [0.3, 0.5]
        assert sub.axes[0].values == expected


# ══════════════════════════════════════════════════════════════════════════════
#  6. save/load Round-Trip
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLoad:
    def test_dill_roundtrip_structure(self, tmp_path):
        grid = _make_grid_2d(n_density=2, n_importance=2, broadcast=True)
        grid.fit(n_epochs=10, batch_size=32)

        path = tmp_path / "grid.pkl"
        grid.save(str(path))
        loaded = ModelGrid.load(str(path))

        assert loaded.shape == grid.shape
        assert len(loaded.axes) == len(grid.axes)
        for orig, load in zip(grid.axes, loaded.axes):
            assert orig.label == load.label
            assert len(orig.values) == len(load.values)

    def test_dill_roundtrip_weights(self, tmp_path):
        grid = _make_grid_1d(n=3)
        grid.fit(n_epochs=15, batch_size=32)

        path = tmp_path / "grid_w.pkl"
        grid.save(str(path))
        loaded = ModelGrid.load(str(path))

        for orig, load in zip(grid.models.ravel(), loaded.models.ravel()):
            assert torch.equal(
                orig.ae.state_dict()["W"],
                load.ae.state_dict()["W"],
            )

    def test_dill_roundtrip_pathlib(self, tmp_path):
        grid = _make_grid_1d(n=2)
        path = tmp_path / "grid_path.pkl"
        grid.save(path)
        loaded = ModelGrid.load(path)
        assert loaded.shape == grid.shape

    def test_pickle_save_load_models(self, tmp_path):
        grid = _make_grid_2d(n_density=2, n_importance=2)
        grid.fit(n_epochs=10, batch_size=32)

        path = str(tmp_path / "models.pkl")
        grid.save_models(path)

        # Mutate weights
        for m in grid.models.ravel():
            with torch.no_grad():
                m.ae.W.fill_(0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.load_models(path)

        any_nonzero = any(
            m.ae.state_dict()["W"].abs().sum() > 0 for m in grid.models.ravel()
        )
        assert any_nonzero

    def test_save_load_with_broadcast_true(self, tmp_path):
        grid = _make_grid_2d(n_density=2, n_importance=2, broadcast=True)
        grid.fit(n_epochs=10, batch_size=32)
        path = tmp_path / "grid_bc.pkl"
        grid.save(str(path))
        loaded = ModelGrid.load(str(path))
        assert loaded.broadcast_samples == grid.broadcast_samples

    def test_save_load_with_broadcast_false(self, tmp_path):
        grid = _make_grid_2d(n_density=2, n_importance=2, broadcast=False)
        grid.fit(n_epochs=10, batch_size=32)
        path = tmp_path / "grid_nobc.pkl"
        grid.save(str(path))
        loaded = ModelGrid.load(str(path))
        assert loaded.broadcast_samples is False


# ══════════════════════════════════════════════════════════════════════════════
#  7. train_saes / evaluate_saes
#     (Lightweight: just verify the plumbing, not SAE quality)
# ══════════════════════════════════════════════════════════════════════════════


class TestSAEPlumbing:
    """These tests train tiny SAEs with minimal steps to verify the ModelGrid
    -> ToyModel.train_saes/evaluate_saes plumbing works."""

    @pytest.fixture
    def trained_grid(self):
        grid = _make_grid_1d(n=2, broadcast=False)
        grid.fit(n_epochs=50, batch_size=64)
        return grid

    def _make_sae_entries(self):
        """Create a minimal SAE entry list for testing."""
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig
        from occhio.toy_model import SAEEntry

        cfg = StandardTrainingSAEConfig(
            d_in=N_HIDDEN,
            d_sae=N_FEATURES * 2,
            l1_coefficient=0.01,
        )
        sae = StandardTrainingSAE(cfg)
        return [SAEEntry(sae=sae, type="Standard")]

    def test_train_saes_runs(self, trained_grid):
        entries = self._make_sae_entries()
        trained_grid.train_saes(
            saes=entries,
            training_samples=2048,
            batch_size=256,
        )
        # Verify each model now has SAEs
        for m in trained_grid.models.ravel():
            assert len(m.saes) > 0

    def test_evaluate_saes_runs(self, trained_grid):
        entries = self._make_sae_entries()
        trained_grid.train_saes(
            saes=entries,
            training_samples=2048,
            batch_size=256,
        )
        trained_grid.evaluate_saes(num_samples=1000)
        for m in trained_grid.models.ravel():
            for label, record in m.saes.items():
                assert record.results is not None

    def test_sae_results_to_dataframe(self, trained_grid):
        entries = self._make_sae_entries()
        trained_grid.train_saes(
            saes=entries,
            training_samples=2048,
            batch_size=256,
        )
        trained_grid.evaluate_saes(num_samples=1000)
        df = trained_grid.sae_results_to_dataframe()
        # 2 models x 1 SAE = 2 rows
        assert len(df) == 2

    def test_empty_sae_results_dataframe(self):
        grid = _make_grid_1d(n=2)
        df = grid.sae_results_to_dataframe()
        assert len(df) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  8. Edge Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_element_grid(self):
        grid = ModelGrid(
            _create_model_fn,
            axes=[Axis(label="density", values=[0.5])],
            broadcast_samples=False,
        )
        assert grid.shape == (1,)
        assert grid.models.size == 1
        losses = grid.fit(n_epochs=10, batch_size=32, track_losses=True)
        assert len(losses) == 10

    def test_fit_called_twice_continues_training(self):
        grid = _make_grid_1d(n=2)
        grid.fit(n_epochs=20, batch_size=32)
        w_after_first = grid.models[0].ae.state_dict()["W"].clone()

        grid.fit(n_epochs=20, batch_size=32)
        w_after_second = grid.models[0].ae.state_dict()["W"]
        assert not torch.equal(w_after_first, w_after_second)

    def test_1_axis_vs_multiple_axes(self):
        grid1d = _make_grid_1d(n=3)
        grid2d = _make_grid_2d(n_density=3, n_importance=2)

        assert len(grid1d.axes) == 1
        assert len(grid2d.axes) == 2

        # Both should train successfully
        grid1d.fit(n_epochs=5, batch_size=32)
        grid2d.fit(n_epochs=5, batch_size=32)

    def test_large_grid_does_not_crash(self):
        """10x10 grid should instantiate without error. Don't train long."""

        def create_model(params: dict) -> ToyModel:
            gen = Generator(device=DEVICE).manual_seed(SEED)
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
            axes=[
                Axis(label="x", values=list(range(10))),
                Axis(label="y", values=list(range(10))),
            ],
            broadcast_samples=True,
        )
        assert grid.shape == (10, 10)
        assert grid.models.size == 100
        # Quick training sanity check — 2 epochs
        grid.fit(n_epochs=2, batch_size=32)

    def test_sample_every_zero_raises(self):
        grid = _make_grid_1d(n=2)
        with pytest.raises(ValueError, match="sample_every"):
            grid.fit(n_epochs=10, batch_size=32, sample_every=0)

    def test_sample_every_negative_raises(self):
        grid = _make_grid_1d(n=2)
        with pytest.raises(ValueError, match="sample_every"):
            grid.fit(n_epochs=10, batch_size=32, sample_every=-1)

    def test_fit_preserves_importances(self):
        grid = _make_grid_2d(n_density=2, n_importance=2)
        before = {i: m.importances.clone() for i, m in enumerate(grid.models.ravel())}
        grid.fit(n_epochs=10, batch_size=32)
        for i, m in enumerate(grid.models.ravel()):
            assert torch.equal(before[i], m.importances)


# ══════════════════════════════════════════════════════════════════════════════
#  BUG DISCOVERY: __getitem__ reverse indexing with list axis values
# ══════════════════════════════════════════════════════════════════════════════


class TestGetitemReverseSlicingBug:
    """BUG: __getitem__ uses `axis.values[indices]` where `indices` is a list,
    but axis.values is a Python list (not a tensor or ndarray), so list-based
    fancy indexing fails with TypeError.

    This occurs in the negative-step branch of slice handling (line ~819).

    Severity: MEDIUM — reverse slicing is broken for list-valued axes.
    """

    def test_reverse_slice_with_list_values(self):
        """grid[4:1] should work even when axis.values are plain lists."""
        grid = ModelGrid(
            _create_model_fn,
            axes=[Axis(label="density", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])],
            broadcast_samples=False,
        )
        sub = grid[4:1]
        assert sub.shape == (3,)
        assert sub.axes[0].values == [0.5, 0.4, 0.3]

    def test_explicit_negative_step_with_list_values(self):
        """grid[4::-1] should work with list-valued axes."""
        grid = ModelGrid(
            _create_model_fn,
            axes=[Axis(label="density", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])],
            broadcast_samples=False,
        )
        sub = grid[4::-1]
        assert sub.shape == (5,)
        assert sub.axes[0].values == [0.5, 0.4, 0.3, 0.2, 0.1]


# ══════════════════════════════════════════════════════════════════════════════
#  BUG DISCOVERY: parameters_mesh with list axis values
# ══════════════════════════════════════════════════════════════════════════════


class TestParametersMesh:
    """parameters_mesh converts axis values via torch.as_tensor and produces
    correct meshgrid output."""

    def test_parameters_mesh_with_list_values_works(self):
        """parameters_mesh handles plain Python list axis values."""
        grid = ModelGrid(
            _create_model_fn,
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

    def test_parameters_mesh_with_tensor_values_correct_shape(self):
        """Tensor axis values produce correct 1D meshgrid."""
        grid = ModelGrid(
            _create_model_fn,
            axes=[Axis(label="density", values=torch.tensor([0.1, 0.5, 0.9]))],
            broadcast_samples=False,
        )
        mesh = grid.parameters_mesh
        assert len(mesh) == 1
        assert mesh[0].shape == (3,)
