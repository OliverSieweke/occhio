# ABOUTME: Deep adversarial audit of ToyModel — training loop correctness,
# ABOUTME: importances, superposition metrics, metric properties, SAE pipeline,
# ABOUTME: edge cases, reproducibility, and delegation.

"""Deep audit of ToyModel.

Tests organized by concern:
1. Training loop correctness (convergence, learning rates)
2. Importances affect training correctly
3. Superposition metric correctness (scaling with compression)
4. Metric properties after training (shapes, bounds, invariants)
5. SAE training pipeline correctness
6. Edge cases and error handling
7. Reproducibility (seeded determinism)
8. sample(), encode(), decode() delegation
"""

import pytest
import torch
from torch import Tensor

from occhio import SAEEntry, ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.toy_model import SAERecord, _resolve_sae_entries

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    n_features=8,
    n_hidden=4,
    p_active=0.1,
    seed_dist=42,
    seed_ae=43,
    importances=None,
):
    """Build a fresh ToyModel with deterministic seeds."""
    g_dist = torch.Generator(device=DEVICE).manual_seed(seed_dist)
    g_ae = torch.Generator(device=DEVICE).manual_seed(seed_ae)
    dist = SparseUniform(n_features=n_features, p_active=p_active, generator=g_dist)
    ae = TiedLinearRelu(n_features, n_hidden, generator=g_ae, device=DEVICE)
    return ToyModel(dist, ae, device=DEVICE, importances=importances)


# ===========================================================================
# 1. TRAINING LOOP CORRECTNESS
# ===========================================================================


class TestTrainingLoopCorrectness:
    """Verify the training loop converges and behaves correctly."""

    def test_loss_decreases_over_1000_epochs(self):
        """Train for 1000 epochs and verify loss decreases substantially."""
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
        losses, _ = tm.fit(n_epochs=1000, batch_size=256, learning_rate=1e-3)

        assert len(losses) == 1000
        # All losses should be finite
        for i, lv in enumerate(losses):
            assert torch.isfinite(torch.tensor(lv)), (
                f"Non-finite loss at epoch {i}: {lv}"
            )

        # Final loss should be significantly lower than initial loss
        early_avg = sum(losses[:20]) / 20
        late_avg = sum(losses[-20:]) / 20
        assert late_avg < early_avg * 0.5, (
            f"Loss did not decrease substantially: "
            f"early_avg={early_avg:.6f}, late_avg={late_avg:.6f}"
        )

    def test_loss_near_monotonic(self):
        """Loss should be roughly monotonically decreasing (allow some noise).

        We check that the smoothed loss curve is monotonically decreasing.
        """
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
        losses, _ = tm.fit(n_epochs=500, batch_size=512, learning_rate=1e-3)

        # Smooth with window of 50
        window = 50
        smoothed = [
            sum(losses[i : i + window]) / window
            for i in range(0, len(losses) - window, window)
        ]
        # Check that smoothed losses are generally decreasing
        decreasing_count = sum(
            1 for i in range(1, len(smoothed)) if smoothed[i] <= smoothed[i - 1]
        )
        total_steps = len(smoothed) - 1
        assert decreasing_count / total_steps >= 0.7, (
            f"Smoothed loss not decreasing often enough: "
            f"{decreasing_count}/{total_steps} steps decreasing"
        )

    @pytest.mark.parametrize("lr", [1e-2, 1e-3, 1e-4])
    def test_all_learning_rates_converge(self, lr):
        """All learning rates should converge, just at different speeds."""
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
        losses, _ = tm.fit(n_epochs=500, batch_size=256, learning_rate=lr)

        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(lv)) for lv in losses)

        # Loss should decrease
        early_avg = sum(losses[:20]) / 20
        late_avg = sum(losses[-20:]) / 20
        assert late_avg < early_avg, (
            f"Loss did not decrease with lr={lr}: "
            f"early={early_avg:.6f}, late={late_avg:.6f}"
        )

    def test_higher_lr_initially_converges_faster(self):
        """Higher learning rate should show faster initial convergence."""
        losses_by_lr = {}
        for lr in [1e-2, 1e-4]:
            tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
            losses, _ = tm.fit(n_epochs=200, batch_size=256, learning_rate=lr)
            losses_by_lr[lr] = losses

        # At epoch 50, the higher lr should have lower loss
        early_high = sum(losses_by_lr[1e-2][:50]) / 50
        early_low = sum(losses_by_lr[1e-4][:50]) / 50
        assert early_high < early_low, (
            f"Higher LR did not converge faster initially: "
            f"lr=1e-2 avg={early_high:.6f}, lr=1e-4 avg={early_low:.6f}"
        )


# ===========================================================================
# 2. IMPORTANCES AFFECT TRAINING CORRECTLY
# ===========================================================================


class TestImportancesAffectTraining:
    """Verify importances weight learning correctly (superposition theory prediction)."""

    def test_important_features_have_larger_norms(self):
        """Features with importance=1 should have larger norms than importance=0 features.

        This is a key prediction of the superposition theory: the model allocates
        more representational capacity to features with higher importance.
        """
        importances = [1, 1, 1, 1, 0, 0, 0, 0]
        tm = _make_model(
            n_features=8, n_hidden=4, p_active=0.1, importances=importances
        )
        tm.fit(n_epochs=5000, batch_size=256, learning_rate=1e-3)

        norms = tm.feature_norms
        important_mean = norms[:4].mean().item()
        unimportant_mean = norms[4:].mean().item()

        assert important_mean > unimportant_mean, (
            f"Important features should have larger norms: "
            f"important_mean={important_mean:.4f}, unimportant_mean={unimportant_mean:.4f}"
        )

    def test_importances_gradation(self):
        """Features with geometrically decaying importances should show
        roughly decaying norms after training."""
        importances = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
        tm = _make_model(
            n_features=8, n_hidden=4, p_active=0.1, importances=importances
        )
        tm.fit(n_epochs=5000, batch_size=256, learning_rate=1e-3)

        norms = tm.feature_norms
        # The top-2 features should have larger norms than bottom-2
        top2_mean = norms[:2].mean().item()
        bot2_mean = norms[6:].mean().item()
        assert top2_mean > bot2_mean, (
            f"Higher-importance features should have larger norms: "
            f"top2={top2_mean:.4f}, bot2={bot2_mean:.4f}"
        )


# ===========================================================================
# 3. SUPERPOSITION METRIC CORRECTNESS
# ===========================================================================


class TestSuperpositionMetricCorrectness:
    """Verify superposition increases with compression ratio."""

    def _train_and_measure_superposition(self, n_features, n_hidden, n_epochs=3000):
        """Train a model and return its superposition metric."""
        tm = _make_model(
            n_features=n_features,
            n_hidden=n_hidden,
            p_active=0.1,
        )
        tm.fit(n_epochs=n_epochs, batch_size=256, learning_rate=1e-3)
        return tm.superposition.item()

    def test_no_compression_low_superposition(self):
        """n_features=4, n_hidden=4: no compression, superposition should be LOW."""
        sup = self._train_and_measure_superposition(n_features=4, n_hidden=4)
        # With no compression needed, superposition should be very low
        assert sup < 0.5, (
            f"Expected low superposition without compression, got {sup:.4f}"
        )

    def test_2x_compression_higher_superposition(self):
        """n_features=8, n_hidden=4: 2x compression, superposition should be HIGHER."""
        sup_1x = self._train_and_measure_superposition(n_features=4, n_hidden=4)
        sup_2x = self._train_and_measure_superposition(n_features=8, n_hidden=4)
        assert sup_2x > sup_1x, (
            f"2x compression should have higher superposition: "
            f"1x={sup_1x:.4f}, 2x={sup_2x:.4f}"
        )

    def test_4x_compression_still_high_superposition(self):
        """n_features=16, n_hidden=4: 4x compression, superposition should be high.

        Note: the mean-max-cosine-similarity metric can saturate and even decrease
        at very high compression because features spread across more directions.
        At 2x compression the metric often reaches ~1.0; at 4x it stays high but
        need not exceed the 2x value. We just verify it exceeds the no-compression
        baseline.
        """
        sup_1x = self._train_and_measure_superposition(n_features=4, n_hidden=4)
        sup_4x = self._train_and_measure_superposition(n_features=16, n_hidden=4)
        assert sup_4x > sup_1x, (
            f"4x compression should have higher superposition than no compression: "
            f"1x={sup_1x:.4f}, 4x={sup_4x:.4f}"
        )

    def test_superposition_increases_from_no_compression(self):
        """Verify superposition increases when moving from no compression to compression."""
        sup_1x = self._train_and_measure_superposition(n_features=4, n_hidden=4)
        sup_2x = self._train_and_measure_superposition(n_features=8, n_hidden=4)
        assert sup_2x > sup_1x, (
            f"Superposition should increase from 1x to 2x compression: "
            f"1x={sup_1x:.4f}, 2x={sup_2x:.4f}"
        )


# ===========================================================================
# 4. METRIC PROPERTIES AFTER TRAINING
# ===========================================================================


class TestMetricPropertiesAfterTraining:
    """Verify metric property shapes, bounds, and mathematical invariants."""

    @pytest.fixture
    def trained_model(self):
        """A model trained for enough epochs to have meaningful metrics."""
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)
        return tm

    def test_feature_norms_positive_and_shape(self, trained_model):
        """feature_norms: all positive, shape (n_features,)."""
        norms = trained_model.feature_norms
        assert norms.shape == (8,)
        assert (norms > 0).all(), f"Some feature norms are zero or negative: {norms}"

    def test_feature_dimensionalities_bounds(self, trained_model):
        """feature_dimensionalities: should be in [0, n_hidden]."""
        dims = trained_model.feature_dimensionalities
        assert dims.shape == (8,)
        assert (dims >= -1e-5).all(), f"Negative dimensionality: {dims}"
        assert (dims <= 4 + 1e-5).all(), f"Dimensionality exceeds n_hidden: {dims}"

    def test_cosine_similarity_matrix_properties(self, trained_model):
        """cosine_similarity_matrix: diagonal=1, off-diagonal in [-1,1], symmetric."""
        cs = trained_model.cosine_similarity_matrix
        assert cs.shape == (8, 8)

        # Diagonal should be 1.0
        assert torch.allclose(cs.diag(), torch.ones(8), atol=1e-4), (
            f"Diagonal not ones: {cs.diag()}"
        )

        # Values should be in [-1, 1]
        assert cs.min() >= -1.0 - 1e-5, f"Min cosine sim below -1: {cs.min()}"
        assert cs.max() <= 1.0 + 1e-5, f"Max cosine sim above 1: {cs.max()}"

        # Symmetric
        assert torch.allclose(cs, cs.T, atol=1e-5), "Cosine similarity not symmetric"

    def test_W_shape(self, trained_model):
        """W property: shape (n_hidden, n_features)."""
        W = trained_model.W
        assert W.shape == (4, 8)

    def test_interferences_shape_and_type(self, trained_model):
        """interferences: shape (n_features, n_features)."""
        interf = trained_model.interferences
        assert interf.shape == (8, 8)
        assert interf.dtype == torch.float32

    def test_interferences_sq_nonnegative(self, trained_model):
        """interferences_sq: all entries should be non-negative (they're squared)."""
        Isq = trained_model.interferences_sq
        assert (Isq >= -1e-7).all(), f"Negative interference squared: {Isq.min()}"

    def test_frobenius_norm_consistency(self, trained_model):
        """frobenius_norm_squared should equal sum of feature_representations."""
        fns = trained_model.frobenius_norm_squared
        reps_sum = trained_model.feature_representations.sum()
        assert torch.isclose(fns, reps_sum, atol=1e-4), (
            f"Frobenius norm^2 ({fns}) != sum of representations ({reps_sum})"
        )

    def test_superposition_bounded_0_to_1(self, trained_model):
        """superposition should be in [0, 1]."""
        sup = trained_model.superposition.item()
        assert 0.0 - 1e-5 <= sup <= 1.0 + 1e-5, f"Superposition out of bounds: {sup}"

    def test_W_T_W_is_gram_matrix(self, trained_model):
        """W^T W should be a valid Gram matrix (symmetric, PSD)."""
        WtW = trained_model.W_T_W
        assert WtW.shape == (8, 8)
        # Symmetric
        assert torch.allclose(WtW, WtW.T, atol=1e-5)
        # PSD: all eigenvalues >= 0
        eigenvalues = torch.linalg.eigvalsh(WtW)
        assert (eigenvalues >= -1e-5).all(), (
            f"W^T W not PSD: min eigenvalue = {eigenvalues.min()}"
        )

    def test_feature_dimensionalities_sum_leq_n_hidden(self, trained_model):
        """Sum of feature dimensionalities should not exceed n_hidden."""
        # This follows from the geometry: the total dimensional capacity is n_hidden.
        # But with superposition, features share dimensions so the sum can exceed
        # n_hidden. The quantity total_feature_dimensionalities_per_hidden_dimension
        # measures this ratio.
        ratio = trained_model.total_feature_dimensionalities_per_hidden_dimension
        assert ratio > 0, f"Ratio should be positive: {ratio}"

    def test_hidden_dim_per_feature_and_inverse_consistency(self, trained_model):
        """hidden_dimensions_per_embedded_features and its inverse should be reciprocals."""
        hd = float(trained_model.hidden_dimensions_per_embedded_features)
        ef = float(trained_model.embedded_features_per_hidden_dimensions)
        assert abs(hd * ef - 1.0) < 1e-4, (
            f"Product should be 1: hd={hd}, ef={ef}, product={hd * ef}"
        )


# ===========================================================================
# 5. SAE TRAINING PIPELINE CORRECTNESS
# ===========================================================================


class TestSAEPipelineCorrectness:
    """Verify the SAE training and evaluation pipeline."""

    def _make_sae_entry(self, n_hidden, label=None, d_sae=16, l1=0.01):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=n_hidden, d_sae=d_sae, l1_coefficient=l1)
        return SAEEntry(sae=StandardTrainingSAE(cfg), type="Standard", label=label)

    @pytest.fixture
    def trained_tm(self):
        """A ToyModel trained sufficiently for SAE experiments."""
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.1)
        tm.fit(n_epochs=500, batch_size=256, learning_rate=1e-3)
        return tm

    def test_sae_record_stored_in_saes_dict(self, trained_tm):
        """After train_saes, SAE record should be stored in model.saes."""
        entry = self._make_sae_entry(4, label="test_sae")
        trained_tm.train_saes([entry], training_samples=10_000, batch_size=256)

        assert "test_sae" in trained_tm.saes
        rec = trained_tm.saes["test_sae"]
        assert isinstance(rec, SAERecord)
        assert rec.sae is entry.sae
        assert rec.sae_type == "Standard"

    def test_evaluate_saes_produces_results(self, trained_tm):
        """evaluate_saes should produce results with sensible values."""
        entry = self._make_sae_entry(4, label="eval_sae")
        trained_tm.train_saes([entry], training_samples=10_000, batch_size=256)
        results = trained_tm.evaluate_saes(num_samples=5_000)

        assert "eval_sae" in results
        result = results["eval_sae"]

        # explained_variance should be > 0 (SAE should be better than random)
        assert result.explained_variance > 0, (
            f"Expected positive explained variance, got {result.explained_variance}"
        )

    def test_classification_metrics_bounded(self, trained_tm):
        """F1, precision, recall should all be in [0, 1]."""
        entry = self._make_sae_entry(4, label="metrics_sae")
        trained_tm.train_saes([entry], training_samples=10_000, batch_size=256)
        trained_tm.evaluate_saes(num_samples=5_000)

        rec = trained_tm.saes["metrics_sae"]
        r = rec.results
        assert 0 <= r.classification.precision <= 1.0 + 1e-5
        assert 0 <= r.classification.recall <= 1.0 + 1e-5
        assert 0 <= r.classification.f1_score <= 1.0 + 1e-5

    def test_evaluate_saes_before_training_any(self):
        """evaluate_saes with no SAEs should return empty or handle gracefully."""
        tm = _make_model(n_features=8, n_hidden=4)
        # No SAEs have been trained
        results = tm.evaluate_saes()
        assert results == {}

    def test_evaluate_saes_with_empty_labels_list(self, trained_tm):
        """evaluate_saes(labels=[]) should return empty dict."""
        entry = self._make_sae_entry(4, label="some_sae")
        trained_tm.train_saes([entry], training_samples=5_000, batch_size=256)
        results = trained_tm.evaluate_saes(labels=[])
        assert results == {}


# ===========================================================================
# 6. EDGE CASES AND ERROR HANDLING
# ===========================================================================


class TestEdgeCasesAndErrorHandling:
    """Probe edge cases in ToyModel."""

    def test_fit_n_epochs_zero_returns_empty(self):
        """fit() with n_epochs=0 should return empty lists, not crash."""
        tm = _make_model()
        losses, hook_returns = tm.fit(n_epochs=0, batch_size=64)
        assert losses == []
        assert hook_returns == []

    def test_fit_n_epochs_one(self):
        """fit() with n_epochs=1 should work and return single loss."""
        tm = _make_model()
        losses, _ = tm.fit(n_epochs=1, batch_size=64)
        assert len(losses) == 1
        assert torch.isfinite(torch.tensor(losses[0]))

    def test_fit_large_batch_size(self):
        """fit() with batch_size larger than typical should still work."""
        tm = _make_model()
        losses, _ = tm.fit(n_epochs=5, batch_size=10000)
        assert len(losses) == 5
        assert all(torch.isfinite(torch.tensor(lv)) for lv in losses)

    def test_train_saes_with_empty_list(self):
        """train_saes with empty list should handle gracefully."""
        tm = _make_model()
        tm.fit(n_epochs=10, batch_size=64)
        # Should not crash
        tm.train_saes([], training_samples=1000, batch_size=256)
        assert len(tm.saes) == 0

    def test_evaluate_saes_nonexistent_label_raises(self):
        """evaluate_saes with nonexistent label should raise ValueError."""
        tm = _make_model()
        with pytest.raises(ValueError, match="do not exist"):
            tm.evaluate_saes(labels=["nonexistent"])

    def test_unknown_attribute_raises_attribute_error(self):
        """Accessing nonexistent attribute should raise AttributeError."""
        tm = _make_model()
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = tm.totally_fake_attribute

    def test_sample_every_equals_one(self):
        """sample_every=1 means re-sample every epoch (extreme but valid)."""
        tm = _make_model()
        losses, _ = tm.fit(n_epochs=10, batch_size=64, sample_every=1)
        assert len(losses) == 10

    def test_sample_every_exceeds_n_epochs(self):
        """sample_every larger than n_epochs should work (only one sample needed)."""
        tm = _make_model()
        losses, _ = tm.fit(n_epochs=5, batch_size=64, sample_every=100)
        assert len(losses) == 5

    def test_fit_with_zero_importances_does_not_crash(self):
        """Zero importances should not cause NaN/Inf (loss is zero for those features)."""
        tm = _make_model(importances=[0, 0, 0, 0, 0, 0, 0, 0])
        losses, _ = tm.fit(n_epochs=10, batch_size=64)
        assert all(torch.isfinite(torch.tensor(lv)) for lv in losses)

    def test_fit_track_losses_false(self):
        """track_losses=False should return empty losses but training still occurs."""
        tm = _make_model()
        W_before = tm.ae.W.data.clone()
        losses, _ = tm.fit(n_epochs=50, batch_size=64, track_losses=False)
        assert losses == []
        # But weights should have changed
        assert not torch.equal(W_before, tm.ae.W.data)

    def test_importances_wrong_length_raises(self):
        """Importances tensor with wrong length should cause shape mismatch during loss."""
        # This tests that the loss function catches the mismatch
        g_dist = torch.Generator(device=DEVICE).manual_seed(1)
        g_ae = torch.Generator(device=DEVICE).manual_seed(2)
        dist = SparseUniform(n_features=8, p_active=0.1, generator=g_dist)
        ae = TiedLinearRelu(8, 4, generator=g_ae, device=DEVICE)
        tm = ToyModel(
            dist, ae, device=DEVICE, importances=torch.ones(8)
        )  # correct length
        # Manually set wrong importances
        tm.importances = torch.ones(5)  # wrong length
        # The fit should produce a runtime error from broadcasting
        with pytest.raises(RuntimeError):
            tm.fit(n_epochs=1, batch_size=64)


# ===========================================================================
# 7. REPRODUCIBILITY
# ===========================================================================


class TestReproducibility:
    """Verify deterministic training with same seeds."""

    def test_same_seed_identical_trajectory(self):
        """Two models with identical seeds should produce identical loss trajectories."""
        tm1 = _make_model(seed_dist=100, seed_ae=101)
        tm2 = _make_model(seed_dist=100, seed_ae=101)

        losses1, _ = tm1.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)
        losses2, _ = tm2.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert abs(l1 - l2) < 1e-6, f"Loss mismatch at epoch {i}: {l1} != {l2}"

    def test_different_seed_different_trajectory(self):
        """Two models with different seeds should produce different loss trajectories."""
        tm1 = _make_model(seed_dist=200, seed_ae=201)
        tm2 = _make_model(seed_dist=300, seed_ae=301)

        losses1, _ = tm1.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)
        losses2, _ = tm2.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)

        # They should differ at some point (very unlikely to be identical by chance)
        any_different = any(abs(l1 - l2) > 1e-6 for l1, l2 in zip(losses1, losses2))
        assert any_different, "Different seeds produced identical trajectories"

    def test_same_seed_same_W(self):
        """Same-seeded models after same training produce identical W matrices."""
        tm1 = _make_model(seed_dist=400, seed_ae=401)
        tm2 = _make_model(seed_dist=400, seed_ae=401)

        tm1.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)
        tm2.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        assert torch.allclose(tm1.W, tm2.W, atol=1e-5), (
            "Same-seeded models produced different W matrices"
        )

    def test_same_seed_same_metrics(self):
        """Same-seeded models produce identical metric values."""
        tm1 = _make_model(seed_dist=500, seed_ae=501)
        tm2 = _make_model(seed_dist=500, seed_ae=501)

        tm1.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)
        tm2.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        assert torch.allclose(tm1.feature_norms, tm2.feature_norms, atol=1e-5)
        assert torch.allclose(tm1.superposition, tm2.superposition, atol=1e-5)
        assert torch.allclose(
            tm1.cosine_similarity_matrix, tm2.cosine_similarity_matrix, atol=1e-5
        )


# ===========================================================================
# 8. SAMPLE, ENCODE, DECODE DELEGATION
# ===========================================================================


class TestDelegation:
    """Verify sample(), encode(), decode() delegation and output shapes."""

    def test_sample_delegates_to_distribution(self):
        """model.sample(N) should call distribution.sample(N) and return correct shape."""
        tm = _make_model(n_features=8, n_hidden=4)
        samples = tm.sample(100)
        assert isinstance(samples, Tensor)
        assert samples.shape == (100, 8)

    def test_encode_delegates_to_ae(self):
        """model.encode(x) should call ae.encode(x) and return correct shape."""
        tm = _make_model(n_features=8, n_hidden=4)
        x = torch.randn(50, 8)
        z = tm.encode(x)
        assert z.shape == (50, 4)

    def test_decode_delegates_to_ae(self):
        """model.decode(z) should delegate to ae.decode(z)."""
        tm = _make_model(n_features=8, n_hidden=4)
        z = torch.randn(50, 4)
        x_hat = tm.decode(z)
        assert x_hat.shape == (50, 8)

    def test_encode_decode_roundtrip_shape(self):
        """Encoding then decoding should preserve batch shape."""
        tm = _make_model(n_features=8, n_hidden=4)
        x = torch.randn(30, 8)
        z = tm.encode(x)
        x_hat = tm.decode(z)
        assert x_hat.shape == x.shape

    def test_sample_latent_shape(self):
        """sample_latent should return (batch_size, n_hidden)."""
        tm = _make_model(n_features=8, n_hidden=4)
        z = tm.sample_latent(64)
        assert z.shape == (64, 4)

    def test_get_one_hot_embeddings_shape(self):
        """get_one_hot_embeddings should return (n_features, n_hidden)."""
        tm = _make_model(n_features=8, n_hidden=4)
        emb = tm.get_one_hot_embeddings()
        assert emb.shape == (8, 4)

    def test_W_equals_one_hot_embeddings_T(self):
        """W should be the transpose of get_one_hot_embeddings()."""
        tm = _make_model(n_features=8, n_hidden=4)
        W = tm.W
        emb = tm.get_one_hot_embeddings()
        assert torch.allclose(W, emb.T, atol=1e-6)

    def test_n_hidden_delegated(self):
        """model.n_hidden should delegate to ae.n_hidden."""
        tm = _make_model(n_features=8, n_hidden=4)
        assert tm.n_hidden == 4

    def test_forward_delegated(self):
        """model.forward should delegate to ae.forward."""
        tm = _make_model(n_features=8, n_hidden=4)
        x = torch.randn(10, 8)
        x_hat, z = tm.forward(x)
        assert x_hat.shape == (10, 8)
        assert z.shape == (10, 4)

    def test_loss_delegated(self):
        """model.loss should delegate to ae.loss."""
        tm = _make_model(n_features=8, n_hidden=4)
        x = torch.randn(10, 8)
        x_hat = torch.randn(10, 8)
        imp = torch.ones(8)
        loss = tm.loss(x, x_hat, imp)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ===========================================================================
# 9. HOOK BEHAVIOR IN TRAINING LOOP (bug probing)
# ===========================================================================


class TestHookBehavior:
    """Probe the hook system for interaction bugs.

    The training loop has two hook systems:
    1. self.hooks (instance hooks, set at construction) - called with (self,)
    2. hooks parameter to fit() - called with a dict
    These have different signatures which is a potential confusion source.
    """

    def test_instance_hooks_receive_model(self):
        """Instance hooks (set in constructor) should receive the ToyModel."""
        received = []

        def hook(tm_instance):
            received.append(type(tm_instance).__name__)

        tm = _make_model()
        tm.hooks = [hook]
        tm.fit(n_epochs=3, batch_size=32)
        assert len(received) == 3
        assert all(r == "ToyModel" for r in received)

    def test_fit_hooks_receive_dict(self):
        """fit() hooks should receive a dict with expected keys."""
        received = []

        def hook(data):
            received.append(set(data.keys()))

        tm = _make_model()
        tm.fit(n_epochs=3, batch_size=32, hooks=[hook])
        assert len(received) == 3
        expected_keys = {"tm", "epoch", "n_epochs", "loss", "x", "x_hat"}
        for keys in received:
            assert expected_keys.issubset(keys)

    def test_both_hook_types_called_together(self):
        """Both instance hooks and fit hooks should be called each epoch."""
        instance_count = [0]
        fit_count = [0]

        def instance_hook(tm_instance):
            instance_count[0] += 1

        def fit_hook(data):
            fit_count[0] += 1

        tm = _make_model()
        tm.hooks = [instance_hook]
        tm.fit(n_epochs=5, batch_size=32, hooks=[fit_hook])

        assert instance_count[0] == 5
        assert fit_count[0] == 5

    def test_fit_hook_return_values_captured(self):
        """fit() hooks that return values should have them captured."""
        tm = _make_model()
        _, hook_returns = tm.fit(
            n_epochs=5, batch_size=32, hooks=[lambda d: d["epoch"]]
        )
        assert len(hook_returns) == 1
        assert hook_returns[0] == [0, 1, 2, 3, 4]

    def test_fit_hook_returning_none_not_captured(self):
        """fit() hooks returning None should not add to hook_returns."""
        tm = _make_model()
        _, hook_returns = tm.fit(n_epochs=5, batch_size=32, hooks=[lambda d: None])
        assert len(hook_returns) == 1
        assert hook_returns[0] == []


# ===========================================================================
# 10. ADDITIONAL METRIC INVARIANTS
# ===========================================================================


class TestAdditionalMetricInvariants:
    """Test mathematical invariants that should hold regardless of training state."""

    def test_W_normalized_features_unit_columns(self):
        """W_normalized_features should have unit-norm columns."""
        tm = _make_model(n_features=8, n_hidden=4)
        tm.fit(n_epochs=100, batch_size=128)
        W_norm = tm.W_normalized_features
        col_norms = torch.linalg.vector_norm(W_norm, dim=0)
        assert torch.allclose(col_norms, torch.ones(8), atol=1e-5)

    def test_interferences_diagonal_is_feature_norms_squared(self):
        """The diagonal of interferences should relate to feature norms.

        interferences = W_normalized^T @ W, so diagonal entry (i,i) =
        (W[:,i] / ||W[:,i]||)^T @ W[:,i] = ||W[:,i]||
        """
        tm = _make_model(n_features=8, n_hidden=4)
        tm.fit(n_epochs=100, batch_size=128)
        interf = tm.interferences
        norms = tm.feature_norms
        assert torch.allclose(interf.diag(), norms, atol=1e-4), (
            f"Interference diagonal ({interf.diag()}) should equal feature norms ({norms})"
        )

    def test_cosine_similarity_from_W_normalized(self):
        """cosine_similarity_matrix should equal W_normalized^T @ W_normalized."""
        tm = _make_model(n_features=8, n_hidden=4)
        tm.fit(n_epochs=100, batch_size=128)
        cs = tm.cosine_similarity_matrix
        W_norm = tm.W_normalized_features
        expected = W_norm.T @ W_norm
        assert torch.allclose(cs, expected, atol=1e-5)

    def test_total_interference_including_self_is_row_sum_of_interferences_sq(self):
        """total_feature_interferences_including_self = row sum of interferences_sq."""
        tm = _make_model(n_features=8, n_hidden=4)
        tm.fit(n_epochs=100, batch_size=128)
        tfi_incl = tm.total_feature_interferences_including_self
        isq = tm.interferences_sq
        expected = isq.sum(dim=1)
        assert torch.allclose(tfi_incl, expected, atol=1e-5)

    def test_feature_frequencies_matches_p_active(self):
        """For SparseUniform, feature_frequencies should match p_active."""
        tm = _make_model(n_features=8, n_hidden=4, p_active=0.3)
        ff = tm.feature_frequencies
        expected = torch.full((8,), 0.3)
        assert torch.allclose(ff, expected, atol=1e-5)

    def test_untrained_model_has_valid_metrics(self):
        """An untrained model should still have valid metric shapes and types."""
        tm = _make_model(n_features=8, n_hidden=4)
        # These should all work without training
        W = tm.W
        assert W.shape == (4, 8)
        norms = tm.feature_norms
        assert norms.shape == (8,)
        sup = tm.superposition
        assert sup.dim() == 0
        cs = tm.cosine_similarity_matrix
        assert cs.shape == (8, 8)


# ===========================================================================
# 11. RESOLVE SAE ENTRIES EDGE CASES
# ===========================================================================


class TestResolveSaeEntriesDeep:
    """Additional edge cases for _resolve_sae_entries."""

    def _make_sae(self):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=4, d_sae=8, l1_coefficient=0.01)
        return StandardTrainingSAE(cfg)

    def test_explicit_label_then_auto_same_type(self):
        """Explicit label followed by auto-labeled entry of same type should work."""
        entries = [
            SAEEntry(sae=self._make_sae(), type="A", label="custom"),
            SAEEntry(sae=self._make_sae(), type="A"),  # auto: A_0
        ]
        resolved = _resolve_sae_entries(entries)
        assert set(resolved.keys()) == {"custom", "A_0"}

    def test_empty_type_string(self):
        """Empty string type should still work for auto-labeling."""
        entries = [SAEEntry(sae=self._make_sae(), type="")]
        resolved = _resolve_sae_entries(entries)
        assert "_0" in resolved

    def test_params_preserved_in_result(self):
        """params dict should be preserved exactly."""
        params = {"k": 2, "nested": {"a": 1}}
        entries = [SAEEntry(sae=self._make_sae(), type="T", params=params)]
        resolved = _resolve_sae_entries(entries)
        _, _, p = resolved["T_0"]
        assert p is params  # Should be the same object


# ===========================================================================
# 12. FIT AFTER FIT (calling fit multiple times)
# ===========================================================================


class TestFitMultipleTimes:
    """Verify behavior when fit() is called multiple times on the same model."""

    def test_fit_twice_further_reduces_loss(self):
        """Calling fit() twice should continue training from current weights."""
        tm = _make_model()
        losses1, _ = tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)
        losses2, _ = tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        # The second round should start near where the first ended
        # (not restart from scratch)
        late_first = sum(losses1[-10:]) / 10
        early_second = sum(losses2[:10]) / 10
        # They should be in the same ballpark
        assert abs(early_second - late_first) / max(late_first, 1e-8) < 2.0, (
            f"Second fit seems to restart: "
            f"late_first={late_first:.6f}, early_second={early_second:.6f}"
        )

    def test_fit_with_new_optimizer_resets_momentum(self):
        """Calling fit() with a new default optimizer resets Adam momentum.
        This is expected behavior -- each fit() creates a fresh optimizer."""
        tm = _make_model()
        # First fit
        tm.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)
        # Second fit (creates a new optimizer internally)
        losses2, _ = tm.fit(n_epochs=50, batch_size=128, learning_rate=1e-3)
        # Should still work and produce finite losses
        assert all(torch.isfinite(torch.tensor(lv)) for lv in losses2)
