# ABOUTME: Red-team tests for ToyModel and ModelGrid pipeline invariants.
# ABOUTME: Tests physical/mathematical invariants that MUST hold if training
# ABOUTME: is working correctly: loss reduction, superposition, feature norms,
# ABOUTME: SAE metric bounds, seed determinism, grid vs individual consistency,
# ABOUTME: broadcast caching, snapshot ordering, save/load round-trip.

"""Pipeline invariant tests for occhio.

These tests verify that the ToyModel and ModelGrid training pipelines
produce results consistent with the mathematical theory of superposition
and the documented behavior of SAEs. Each test targets a specific invariant
that, if violated, indicates a real bug in the training or evaluation logic.

Designed for fast execution: small models, short training, but enough
epochs/samples to see genuine effects.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from occhio import SAEEntry, ToyModel
from occhio.autoencoders import TiedLinear, TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.model_grid import Axis, ModelGrid, TrainingAxis

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_model(
    n_features: int = 8,
    n_hidden: int = 4,
    p_active: float = 0.5,
    importances: list[float] | None = None,
    seed: int = 42,
) -> ToyModel:
    """Create a small ToyModel with seeded generators."""
    g_dist = torch.Generator(device=DEVICE).manual_seed(seed)
    g_ae = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    dist = SparseUniform(n_features=n_features, p_active=p_active, generator=g_dist)
    ae = TiedLinearRelu(n_features, n_hidden, generator=g_ae, device=DEVICE)
    return ToyModel(
        dist,
        ae,
        device=DEVICE,
        importances=importances,
    )


def _make_sae_entry(
    d_in: int, d_sae: int = 16, l1: float = 0.01, label: str | None = None
):
    """Create an SAEEntry for testing."""
    from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

    cfg = StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae, l1_coefficient=l1)
    sae = StandardTrainingSAE(cfg)
    return SAEEntry(sae=sae, type="Standard", params={"l1": l1}, label=label)


# ===========================================================================
# 1. Training actually learns
# ===========================================================================


class TestTrainingLearns:
    """Invariant: After sufficient training, loss must drop significantly."""

    def test_loss_drops_significantly_sparse_uniform(self):
        """8 features into 4 hidden with SparseUniform: final loss should be
        significantly lower than initial loss. If loss stays flat, the
        training loop is broken."""
        tm = _make_toy_model(n_features=8, n_hidden=4, p_active=0.5, seed=10)
        losses, _ = tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)

        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10

        # With 8 features compressed into 4 hidden dims, there is an
        # irreducible information bottleneck loss. The model should still
        # reduce loss meaningfully -- at least 30% reduction.
        assert final < initial * 0.7, (
            f"Training did not reduce loss sufficiently: "
            f"initial_avg={initial:.6f}, final_avg={final:.6f}, "
            f"ratio={final / initial:.2f} (expected < 0.7)"
        )

    def test_loss_drops_no_superposition_regime(self):
        """4 features into 8 hidden (no superposition needed): should learn
        a near-perfect reconstruction, with very low final loss."""
        tm = _make_toy_model(n_features=4, n_hidden=8, p_active=0.5, seed=20)
        losses, _ = tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        final = sum(losses[-10:]) / 10
        # With n_hidden > n_features and no competition, loss should be tiny.
        assert final < 0.01, (
            f"With n_hidden > n_features, final loss should be near zero, "
            f"got {final:.6f}"
        )


# ===========================================================================
# 2. Superposition metric
# ===========================================================================


class TestSuperpositionMetric:
    """Invariant: superposition should be high when n_features > n_hidden
    and data is sparse, and low when n_features <= n_hidden."""

    def test_superposition_high_when_overparameterized(self):
        """With 12 features in 4 hidden dims and sparse data, the model
        must pack features into shared directions -> high superposition."""
        tm = _make_toy_model(n_features=12, n_hidden=4, p_active=0.3, seed=30)
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)

        sup = tm.superposition.item()
        assert sup > 0.3, (
            f"Superposition should be high (>0.3) with 12 features in 4 dims, "
            f"got {sup:.4f}"
        )

    def test_superposition_low_when_underparameterized(self):
        """With 3 features in 8 hidden dims, each feature can get its own
        orthogonal direction -> low superposition."""
        tm = _make_toy_model(n_features=3, n_hidden=8, p_active=0.5, seed=40)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        sup = tm.superposition.item()
        assert sup < 0.5, (
            f"Superposition should be low (<0.5) with 3 features in 8 dims, "
            f"got {sup:.4f}"
        )


# ===========================================================================
# 3. Feature norms reflect importance
# ===========================================================================


class TestFeatureNormsReflectImportance:
    """Invariant: features with higher importance weight should be allocated
    more capacity (larger norms) by the autoencoder."""

    def test_important_features_have_larger_norms(self):
        """With exponentially decaying importances, the top features
        should have strictly larger norms than the bottom features."""
        n_features = 8
        n_hidden = 4
        # Importances: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        importances = [0.5**i for i in range(n_features)]
        tm = _make_toy_model(
            n_features=n_features,
            n_hidden=n_hidden,
            importances=importances,
            seed=50,
        )
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)

        norms = tm.feature_norms
        # The most important feature should have a larger norm than
        # the least important feature.
        most_important_norm = norms[0].item()
        least_important_norm = norms[-1].item()

        assert most_important_norm > least_important_norm, (
            f"Most important feature norm ({most_important_norm:.4f}) should be "
            f"larger than least important ({least_important_norm:.4f})"
        )

    def test_zero_importance_features_shrink(self):
        """Features with importance=0 should not be learned at all:
        their norms should be very small after training."""
        n_features = 6
        n_hidden = 4
        # First 3 features important, last 3 have zero importance
        importances = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        tm = _make_toy_model(
            n_features=n_features,
            n_hidden=n_hidden,
            importances=importances,
            seed=55,
        )
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)

        norms = tm.feature_norms
        important_norms = norms[:3]
        zero_norms = norms[3:]

        # Zero-importance features should have smaller average norm
        assert zero_norms.mean() < important_norms.mean(), (
            f"Zero-importance features (mean norm {zero_norms.mean():.4f}) should "
            f"have smaller norms than important features ({important_norms.mean():.4f})"
        )


# ===========================================================================
# 4. SAE training improves reconstruction
# ===========================================================================


class TestSAETrainingImprovesReconstruction:
    """Invariant: A trained SAE should reconstruct hidden activations
    better than chance. explained_variance should be positive."""

    def test_sae_explained_variance_positive(self):
        """After training an SAE, explained_variance must be > 0.
        Uses a generous training budget and low L1 so the SAE can
        focus on reconstruction quality."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=60)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        # Low L1 and plenty of training so the SAE can actually learn
        # good reconstructions rather than being overwhelmed by sparsity penalty.
        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.001, label="ev_test")
        tm.train_saes([entry], training_samples=200_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=10_000)

        ev = results["ev_test"].explained_variance
        assert ev > 0, (
            f"SAE explained_variance should be > 0 after training, got {ev:.6f}"
        )


# ===========================================================================
# 5. SAE evaluation metrics are sensible
# ===========================================================================


class TestSAEMetricBounds:
    """Invariant: All SAE classification metrics must lie within their
    mathematically valid ranges."""

    @pytest.fixture(scope="class")
    def trained_model_with_sae(self):
        """Train a model and SAE once for all metric bound tests."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=70)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.01, label="bounds_test")
        tm.train_saes([entry], training_samples=50_000, batch_size=512)
        tm.evaluate_saes(num_samples=10_000)
        return tm

    def test_f1_in_range(self, trained_model_with_sae):
        f1 = trained_model_with_sae.saes_f1_score["bounds_test"]
        assert 0.0 <= f1 <= 1.0, f"F1 should be in [0,1], got {f1}"

    def test_precision_in_range(self, trained_model_with_sae):
        p = trained_model_with_sae.saes_precision["bounds_test"]
        assert 0.0 <= p <= 1.0, f"Precision should be in [0,1], got {p}"

    def test_recall_in_range(self, trained_model_with_sae):
        r = trained_model_with_sae.saes_recall["bounds_test"]
        assert 0.0 <= r <= 1.0, f"Recall should be in [0,1], got {r}"

    def test_mcc_in_range(self, trained_model_with_sae):
        mcc = trained_model_with_sae.saes_mcc["bounds_test"]
        assert -1.0 <= mcc <= 1.0, f"MCC should be in [-1,1], got {mcc}"

    def test_l0_non_negative(self, trained_model_with_sae):
        l0 = trained_model_with_sae.saes_l0["bounds_test"]
        assert l0 >= 0, f"L0 should be >= 0, got {l0}"

    def test_dead_latents_non_negative(self, trained_model_with_sae):
        dl = trained_model_with_sae.saes_dead_latents["bounds_test"]
        assert dl >= 0, f"dead_latents should be >= 0, got {dl}"

    def test_shrinkage_non_negative(self, trained_model_with_sae):
        s = trained_model_with_sae.saes_shrinkage["bounds_test"]
        assert s >= 0, f"Shrinkage should be >= 0, got {s}"


# ===========================================================================
# 6. Consistent results with same seed
# ===========================================================================


class TestSeedDeterminism:
    """Invariant: Two ToyModels with identical seeds and config must produce
    bit-identical results."""

    def test_same_seed_same_final_loss(self):
        """Two models with same seed -> same loss trajectory."""
        tm1 = _make_toy_model(seed=80)
        tm2 = _make_toy_model(seed=80)

        losses1, _ = tm1.fit(n_epochs=100, batch_size=128)
        losses2, _ = tm2.fit(n_epochs=100, batch_size=128)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, f"Loss diverged at epoch {i}: {l1} vs {l2}"

    def test_same_seed_same_weights(self):
        """Two models with same seed -> identical final weight matrices."""
        tm1 = _make_toy_model(seed=85)
        tm2 = _make_toy_model(seed=85)

        tm1.fit(n_epochs=100, batch_size=128)
        tm2.fit(n_epochs=100, batch_size=128)

        W1 = tm1.W
        W2 = tm2.W
        assert torch.equal(W1, W2), (
            f"Weight matrices differ with same seed. "
            f"Max diff: {(W1 - W2).abs().max().item():.2e}"
        )

    def test_same_seed_same_superposition(self):
        """Two models with same seed -> identical superposition metric."""
        tm1 = _make_toy_model(seed=90)
        tm2 = _make_toy_model(seed=90)

        tm1.fit(n_epochs=100, batch_size=128)
        tm2.fit(n_epochs=100, batch_size=128)

        assert tm1.superposition.item() == tm2.superposition.item()


# ===========================================================================
# 7. Importances affect learning
# ===========================================================================


class TestImportancesAffectLearning:
    """Invariant: Features with importance=0 should have high reconstruction
    error; the model should not waste capacity on them."""

    def test_zero_importance_features_poorly_reconstructed(self):
        """Per-feature reconstruction error should be higher for
        zero-importance features than for important features."""
        n_features = 8
        n_hidden = 4
        importances = [1.0] * 4 + [0.0] * 4
        tm = _make_toy_model(
            n_features=n_features,
            n_hidden=n_hidden,
            importances=importances,
            seed=100,
        )
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)

        # Compute per-feature reconstruction error
        with torch.no_grad():
            x = tm.distribution.sample(2000).to(DEVICE)
            x_hat = tm.ae(x)[0]
            per_feature_mse = ((x - x_hat) ** 2).mean(dim=0)

        important_error = per_feature_mse[:4].mean().item()
        zero_error = per_feature_mse[4:].mean().item()

        assert zero_error > important_error, (
            f"Zero-importance features should have higher reconstruction error "
            f"({zero_error:.6f}) than important features ({important_error:.6f})"
        )


# ===========================================================================
# 8. Grid fit ~ individual fit
# ===========================================================================


class TestGridFitConsistency:
    """Invariant: ModelGrid.fit() should produce similar (not identical)
    results to fitting each ToyModel individually with ToyModel.fit().
    The difference is that ModelGrid averages losses across all models
    in its backward pass, whereas individual fit only sees one model's loss."""

    def test_grid_fit_produces_similar_quality(self):
        """Each model in a fitted grid should have loss comparable to
        individually fitted models. We check that grid-fitted models
        actually learn (loss drops) rather than demanding exact equality."""

        def make_model(seed):
            g_dist = torch.Generator(device=DEVICE).manual_seed(seed)
            g_ae = torch.Generator(device=DEVICE).manual_seed(seed + 1)
            dist = SparseUniform(n_features=6, p_active=0.5, generator=g_dist)
            ae = TiedLinearRelu(6, 3, generator=g_ae, device=DEVICE)
            return ToyModel(dist, ae, device=DEVICE)

        # Grid fit
        def create_model(params):
            seed = int(params["seed"])
            return make_model(seed)

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="seed", values=[100, 200, 300])],
            broadcast_samples=False,
        )
        grid.fit(n_epochs=200, batch_size=128, learning_rate=1e-3)

        # Check each grid model actually learned something
        for i, model in enumerate(grid.models.ravel()):
            with torch.no_grad():
                x = model.distribution.sample(1000).to(DEVICE)
                x_hat = model.ae(x)[0]
                mse = ((x - x_hat) ** 2).mean().item()

            # MSE should be significantly below the baseline of no learning.
            # With random weights on unit-scale data, MSE is typically ~0.5-1.0.
            assert mse < 0.3, (
                f"Grid model {i} MSE={mse:.4f} too high -- grid training may "
                f"not be learning effectively"
            )


# ===========================================================================
# 9. Broadcast caching correctness
# ===========================================================================


class TestBroadcastCaching:
    """Invariant: With broadcast_samples=True, models sharing the same
    distribution should get identical training data and produce identical
    final weights if they start identical."""

    def test_identical_distributions_produce_identical_weights(self):
        """Models sharing same distribution + same init weights should
        converge to identical weights under broadcast_samples=True."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="idx", values=[0, 1, 2])],
            broadcast_samples=True,
        )
        grid.fit(n_epochs=100, batch_size=64)

        models = grid.models.ravel()
        ref_W = models[0].ae.state_dict()["W"]
        for i, m in enumerate(models[1:], 1):
            W = m.ae.state_dict()["W"]
            assert torch.equal(ref_W, W), (
                f"Model {i} weights differ from model 0 despite same distribution "
                f"and same init under broadcast_samples=True. "
                f"Max diff: {(ref_W - W).abs().max().item():.2e}"
            )

    def test_different_distributions_produce_different_weights(self):
        """Models with different distributions should NOT converge to
        identical weights, even with broadcast_samples=True."""

        def create_model(params):
            p = params["p_active"]
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(6, p_active=p, device=DEVICE, generator=gen),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="p_active", values=[0.1, 0.5, 0.9])],
            broadcast_samples=True,
        )
        grid.fit(n_epochs=200, batch_size=64)

        models = grid.models.ravel()
        W0 = models[0].ae.state_dict()["W"]
        any_differ = any(
            not torch.equal(W0, m.ae.state_dict()["W"]) for m in models[1:]
        )
        assert any_differ, (
            "Models with different distributions should have different weights"
        )


# ===========================================================================
# 10. Snapshot interval captures training progress
# ===========================================================================


class TestSnapshotInterval:
    """Invariant: Early snapshots should have higher loss than late
    snapshots, since the model is learning over time."""

    def test_early_snapshots_worse_than_late(self):
        """With snapshot_interval, losses computed on each snapshot's
        weights should decrease over time."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="idx", values=[0])],
            broadcast_samples=False,
        )

        history = grid.fit(
            n_epochs=200,
            batch_size=128,
            snapshot_interval=100,
            learning_rate=1e-3,
        )

        assert isinstance(history, ModelGrid)
        assert isinstance(history.axes[0], TrainingAxis)

        # history has shape (n_snapshots, 1), snapshots at epoch 0, 100, 200
        n_snapshots = history.shape[0]
        assert n_snapshots == 3, f"Expected 3 snapshots, got {n_snapshots}"

        # Compute loss for each snapshot's model weights
        snapshot_losses = []
        for s in range(n_snapshots):
            snapshot_model = history.models[s, 0]
            with torch.no_grad():
                # Use a fixed set of test data
                test_gen = torch.Generator(device=DEVICE).manual_seed(999)
                test_dist = SparseUniform(6, p_active=0.5, generator=test_gen)
                x = test_dist.sample(1000)
                x_hat = snapshot_model.ae(x)[0]
                loss = ((x - x_hat) ** 2).mean().item()
            snapshot_losses.append(loss)

        # The initial snapshot (epoch 0) should have higher loss than the
        # final snapshot (epoch 200).
        assert snapshot_losses[0] > snapshot_losses[-1], (
            f"Initial snapshot loss ({snapshot_losses[0]:.6f}) should be higher "
            f"than final snapshot loss ({snapshot_losses[-1]:.6f}). "
            f"All snapshot losses: {snapshot_losses}"
        )


# ===========================================================================
# 11. Save/load round-trip preserves behavior
# ===========================================================================


class TestSaveLoadRoundTrip:
    """Invariant: After save+load, a ModelGrid should produce identical
    outputs for the same input."""

    def test_dill_save_load_preserves_outputs(self):
        """ModelGrid.save() + ModelGrid.load() should produce a grid
        whose models give identical outputs."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="idx", values=[0, 1])],
            broadcast_samples=False,
        )
        grid.fit(n_epochs=50, batch_size=64)

        # Compute reference outputs before saving
        test_input = torch.randn(10, 6)
        ref_outputs = []
        for m in grid.models.ravel():
            with torch.no_grad():
                out = m.ae(test_input)[0]
            ref_outputs.append(out.clone())

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_path = f.name

        try:
            grid.save(save_path)
            loaded_grid = ModelGrid.load(save_path)

            for i, m in enumerate(loaded_grid.models.ravel()):
                with torch.no_grad():
                    out = m.ae(test_input)[0]
                assert torch.allclose(out, ref_outputs[i], atol=1e-6), (
                    f"Model {i} output changed after save/load. "
                    f"Max diff: {(out - ref_outputs[i]).abs().max().item():.2e}"
                )
        finally:
            Path(save_path).unlink(missing_ok=True)


# ===========================================================================
# 12. Slicing preserves model identity
# ===========================================================================


class TestSlicingPreservesIdentity:
    """Invariant: grid[i] should contain the SAME model objects (by identity)
    as the original grid, not copies."""

    def test_int_index_returns_same_object(self):
        """grid[i, j] should return the same ToyModel object as
        grid.models[i, j]."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[
                Axis(label="a", values=[0, 1, 2]),
                Axis(label="b", values=[0, 1]),
            ],
            broadcast_samples=False,
        )

        for i in range(3):
            for j in range(2):
                assert grid[i, j] is grid.models[i, j], (
                    f"grid[{i},{j}] returned a different object than grid.models[{i},{j}]"
                )

    def test_slice_returns_view_with_same_objects(self):
        """grid[1:3] should contain the same model objects as the
        corresponding slice of grid.models."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[
                Axis(label="a", values=[0, 1, 2, 3]),
                Axis(label="b", values=[0, 1]),
            ],
            broadcast_samples=False,
        )

        sub = grid[1:3]
        assert isinstance(sub, ModelGrid)
        for i in range(2):
            for j in range(2):
                assert sub.models[i, j] is grid.models[i + 1, j], (
                    f"sub.models[{i},{j}] is not grid.models[{i + 1},{j}]"
                )


# ===========================================================================
# 13. SAE with high L1 -> high sparsity (low L0)
# ===========================================================================


class TestSAEL1SparsitySweep:
    """Invariant: Increasing L1 coefficient should produce monotonically
    (or near-monotonically) decreasing L0."""

    def test_higher_l1_yields_lower_l0(self):
        """Sweep L1 from low to high; L0 should decrease."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=110)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        l1_values = [0.001, 0.1, 1.0]
        l0_results = []

        for i, l1 in enumerate(l1_values):
            entry = _make_sae_entry(d_in=4, d_sae=32, l1=l1, label=f"l1_{i}")
            tm.train_saes([entry], training_samples=50_000, batch_size=512)
            tm.evaluate_saes(labels=[f"l1_{i}"], num_samples=10_000)
            l0_results.append(tm.saes_l0[f"l1_{i}"])

        # L0 at lowest L1 should be >= L0 at highest L1
        assert l0_results[0] >= l0_results[-1], (
            f"L0 should decrease as L1 increases. "
            f"L1 values: {l1_values}, L0 values: {l0_results}"
        )


# ===========================================================================
# 14. SAE with L1=0 -> low sparsity (high L0)
# ===========================================================================


class TestSAENoL1:
    """Invariant: With L1=0 (no sparsity penalty), the SAE should use
    many latents, producing high L0."""

    def test_l1_zero_uses_many_latents(self):
        """L1=0 -> SAE has no reason to be sparse -> L0 should be high."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=120)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry_no_l1 = _make_sae_entry(d_in=4, d_sae=16, l1=0.0, label="no_l1")
        entry_high_l1 = _make_sae_entry(d_in=4, d_sae=16, l1=1.0, label="high_l1")

        tm.train_saes(
            [entry_no_l1, entry_high_l1], training_samples=50_000, batch_size=512
        )
        tm.evaluate_saes(num_samples=10_000)

        l0_no_l1 = tm.saes_l0["no_l1"]
        l0_high_l1 = tm.saes_l0["high_l1"]

        assert l0_no_l1 > l0_high_l1, (
            f"L0 with no L1 ({l0_no_l1:.2f}) should be higher than with high L1 "
            f"({l0_high_l1:.2f})"
        )


# ===========================================================================
# 15. Dead latent count with high L1
# ===========================================================================


class TestDeadLatents:
    """Invariant: With very high L1, some latents should die."""

    def test_high_l1_has_more_dead_latents_than_low_l1(self):
        """With much higher L1, the SAE should have at least as many
        dead latents as with low L1. This is a relative test rather than
        an absolute one, because what counts as 'dead' depends on the
        evaluation procedure in SAE-lens."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=130)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry_low = _make_sae_entry(d_in=4, d_sae=64, l1=0.001, label="dead_low")
        entry_high = _make_sae_entry(d_in=4, d_sae=64, l1=10.0, label="dead_high")

        tm.train_saes([entry_low, entry_high], training_samples=100_000, batch_size=512)
        tm.evaluate_saes(num_samples=10_000)

        dead_low = tm.saes_dead_latents["dead_low"]
        dead_high = tm.saes_dead_latents["dead_high"]

        # High L1 should produce at least as many dead latents as low L1
        assert dead_high >= dead_low, (
            f"High L1 should produce >= dead latents than low L1. "
            f"Low L1: {dead_low}, High L1: {dead_high}"
        )
        # Additionally, with L1=10 and 64 latents for a 4-dim problem,
        # we expect L0 to be very low (high sparsity)
        l0_high = tm.saes_l0["dead_high"]
        l0_low = tm.saes_l0["dead_low"]
        assert l0_high < l0_low, (
            f"High L1 should produce lower L0 than low L1. "
            f"Low L1 L0: {l0_low:.2f}, High L1 L0: {l0_high:.2f}"
        )


# ===========================================================================
# 16. Multiple SAEs can coexist
# ===========================================================================


class TestMultipleSAEsCoexist:
    """Invariant: Training multiple SAEs on the same model should work
    independently -- they should all produce valid results."""

    def test_multiple_saes_independent(self):
        """Train 3 SAEs with different L1 on the same model, evaluate all."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=140)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entries = []
        for i, l1 in enumerate([0.001, 0.01, 0.1]):
            entries.append(_make_sae_entry(d_in=4, d_sae=16, l1=l1, label=f"multi_{i}"))

        tm.train_saes(entries, training_samples=50_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=10_000)

        # All 3 should have valid results
        assert len(results) == 3
        for label in ["multi_0", "multi_1", "multi_2"]:
            assert label in results
            r = results[label]
            assert r.explained_variance is not None
            assert r.sae_l0 >= 0
            assert 0.0 <= r.classification.f1_score <= 1.0


# ===========================================================================
# Additional invariants: hooks called correctly during training
# ===========================================================================


class TestHooksCalledDuringFit:
    """Invariant: Instance hooks on ToyModel should fire every epoch,
    and hooks passed to fit() should also fire every epoch with correct data."""

    def test_instance_hooks_fire_every_epoch(self):
        """Hooks registered in ToyModel constructor fire once per epoch."""
        count = [0]

        def hook(tm_instance):
            count[0] += 1

        tm = _make_toy_model(seed=150)
        tm.hooks.append(hook)
        tm.fit(n_epochs=50, batch_size=64)

        assert count[0] == 50, (
            f"Instance hook should fire 50 times, fired {count[0]} times"
        )

    def test_fit_hooks_receive_correct_data(self):
        """Hooks passed to fit() receive a dict with expected keys and
        valid tensor shapes."""
        captured = []

        def hook(data):
            captured.append(
                {
                    "epoch": data["epoch"],
                    "loss_finite": torch.isfinite(data["loss"]).item(),
                    "x_shape": tuple(data["x"].shape),
                    "x_hat_shape": tuple(data["x_hat"].shape),
                }
            )

        tm = _make_toy_model(seed=155)
        tm.fit(n_epochs=10, batch_size=32, hooks=[hook])

        assert len(captured) == 10
        for i, c in enumerate(captured):
            assert c["epoch"] == i
            assert c["loss_finite"]
            assert c["x_shape"] == (32, 8)  # batch_size x n_features
            assert c["x_hat_shape"] == (32, 8)


# ===========================================================================
# ToyModel metric consistency checks
# ===========================================================================


class TestMetricConsistency:
    """Cross-check various metric properties for mathematical consistency."""

    def test_W_matches_one_hot_embeddings(self):
        """W property must equal get_one_hot_embeddings().T"""
        tm = _make_toy_model(seed=160)
        tm.fit(n_epochs=100, batch_size=128)

        W = tm.W
        ohe = tm.get_one_hot_embeddings()
        assert torch.allclose(W, ohe.T, atol=1e-6), (
            f"W does not match get_one_hot_embeddings().T. "
            f"Max diff: {(W - ohe.T).abs().max().item():.2e}"
        )

    def test_feature_norms_squared_equals_representations(self):
        """feature_norms**2 must equal feature_representations."""
        tm = _make_toy_model(seed=165)
        tm.fit(n_epochs=100, batch_size=128)

        norms_sq = tm.feature_norms**2
        reps = tm.feature_representations
        assert torch.allclose(norms_sq, reps, atol=1e-5), (
            f"feature_norms**2 does not match feature_representations. "
            f"Max diff: {(norms_sq - reps).abs().max().item():.2e}"
        )

    def test_cosine_similarity_diagonal_is_one(self):
        """Diagonal of cosine similarity matrix should be 1.0."""
        tm = _make_toy_model(seed=170)
        tm.fit(n_epochs=100, batch_size=128)

        cs = tm.cosine_similarity_matrix
        diag = cs.diag()
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4), (
            f"Cosine similarity diagonal should be 1.0, got {diag}"
        )

    def test_frobenius_norm_squared_equals_sum_feature_representations(self):
        """||W||_F^2 = sum of feature representations."""
        tm = _make_toy_model(seed=175)
        tm.fit(n_epochs=100, batch_size=128)

        fns = tm.frobenius_norm_squared
        reps_sum = tm.feature_representations.sum()
        assert torch.allclose(fns, reps_sum, atol=1e-4), (
            f"||W||_F^2 ({fns:.6f}) != sum(feature_representations) ({reps_sum:.6f})"
        )

    def test_superposition_bounded_zero_one(self):
        """Superposition metric must be in [0, 1]."""
        tm = _make_toy_model(seed=180)
        tm.fit(n_epochs=100, batch_size=128)

        sup = tm.superposition.item()
        assert 0.0 <= sup <= 1.0 + 1e-6, f"Superposition should be in [0,1], got {sup}"


# ===========================================================================
# ModelGrid: loss writeback and state consistency
# ===========================================================================


class TestModelGridStateWriteback:
    """Invariant: After ModelGrid.fit(), each individual model's weights
    should reflect the training that happened in the vectorized loop."""

    def test_individual_model_inference_after_grid_fit(self):
        """After grid.fit(), calling model.ae(x) should use trained weights,
        not the original random weights."""

        def create_model(params):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            return ToyModel(
                distribution=SparseUniform(
                    6, p_active=0.5, device=DEVICE, generator=gen
                ),
                ae=TiedLinearRelu(6, 3, generator=gen, device=DEVICE),
                device=DEVICE,
            )

        grid = ModelGrid(
            create_model,
            axes=[Axis(label="idx", values=[0, 1])],
            broadcast_samples=False,
        )

        # Record pre-training outputs
        test_input = torch.randn(10, 6)
        pre_outputs = []
        for m in grid.models.ravel():
            with torch.no_grad():
                pre_outputs.append(m.ae(test_input)[0].clone())

        grid.fit(n_epochs=100, batch_size=64, learning_rate=1e-3)

        # Post-training outputs should differ from pre-training
        for i, m in enumerate(grid.models.ravel()):
            with torch.no_grad():
                post_output = m.ae(test_input)[0]
            assert not torch.allclose(post_output, pre_outputs[i], atol=1e-4), (
                f"Model {i} output unchanged after grid.fit() -- "
                f"weights may not have been written back"
            )


# ===========================================================================
# ToyModel.fit() loss function uses importances
# ===========================================================================


class TestLossFunctionUsesImportances:
    """Invariant: The loss function should weight per-feature errors by
    the importance vector. This is a functional test, not just a unit test."""

    def test_loss_with_importances_differs_from_without(self):
        """A model with non-uniform importances should produce different
        losses than one with uniform importances on the same data."""
        g1 = torch.Generator(device=DEVICE).manual_seed(200)
        g2 = torch.Generator(device=DEVICE).manual_seed(200)
        dist1 = SparseUniform(n_features=6, p_active=0.5, generator=g1)
        dist2 = SparseUniform(n_features=6, p_active=0.5, generator=g2)

        g_ae1 = torch.Generator(device=DEVICE).manual_seed(201)
        g_ae2 = torch.Generator(device=DEVICE).manual_seed(201)
        ae1 = TiedLinearRelu(6, 3, generator=g_ae1, device=DEVICE)
        ae2 = TiedLinearRelu(6, 3, generator=g_ae2, device=DEVICE)

        tm_uniform = ToyModel(dist1, ae1, device=DEVICE)
        tm_weighted = ToyModel(
            dist2,
            ae2,
            device=DEVICE,
            importances=[10.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
        )

        losses_uniform, _ = tm_uniform.fit(n_epochs=100, batch_size=128)
        losses_weighted, _ = tm_weighted.fit(n_epochs=100, batch_size=128)

        # The initial losses should differ because the importance weighting
        # changes the loss computation. After training, the models diverge further.
        assert losses_uniform[0] != losses_weighted[0], (
            "Initial losses should differ between uniform and weighted importances"
        )


# ===========================================================================
# TiedLinear (no ReLU) vs TiedLinearRelu
# ===========================================================================


class TestAutoEncoderVariants:
    """Verify that different autoencoder types work correctly with ToyModel."""

    def test_tied_linear_learns(self):
        """TiedLinear (no ReLU in decode) should also learn."""
        g_dist = torch.Generator(device=DEVICE).manual_seed(210)
        g_ae = torch.Generator(device=DEVICE).manual_seed(211)
        dist = SparseUniform(n_features=6, p_active=0.5, generator=g_dist)
        ae = TiedLinear(6, 3, generator=g_ae, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)

        losses, _ = tm.fit(n_epochs=200, batch_size=128, learning_rate=1e-3)

        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10
        assert final < initial * 0.7, (
            f"TiedLinear should learn: initial={initial:.6f}, final={final:.6f}"
        )
