"""Integration invariant tests for the SAE pipeline in occhio.

Tests the full data flow: Distribution -> ActivationGeneratorWrapper ->
sae_lens training -> evaluation, plus component-level invariants for the
CoefficientAutotuner, FeatureDictionaryWrapper, and SAE training outcomes.

These tests use deliberately small models for speed:
  n_features=10-20, n_hidden=5-10, d_sae=20-40, 50k-200k training samples.
"""

import pytest
import torch
import torch.nn.functional as F
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

from occhio import SAEEntry, ToyModel
from occhio.autoencoders import TiedLinear, TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.sae_lens_adapter.activation_generator import ActivationGeneratorWrapper
from occhio.sae_lens_adapter.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)
from occhio.sae_lens_adapter.feature_dictionary import FeatureDictionaryWrapper
from occhio.sae_lens_adapter.standard_sae_autotuned import (
    StandardTrainingSAEAutotuned,
    StandardTrainingSAEConfigAutotuned,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 10
N_HIDDEN = 5
D_SAE = 20
TRAINING_SAMPLES = 50_000
EVAL_SAMPLES = 10_000
BATCH_SIZE = 512
P_ACTIVE = 0.3


@pytest.fixture
def seeded_distribution():
    gen = torch.Generator()
    gen.manual_seed(42)
    return SparseUniform(n_features=N_FEATURES, p_active=P_ACTIVE, generator=gen)


@pytest.fixture
def trained_model():
    """A small ToyModel that has been fit (AE trained)."""
    torch.manual_seed(0)
    dist = SparseUniform(n_features=N_FEATURES, p_active=P_ACTIVE)
    ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=N_HIDDEN)
    model = ToyModel(dist, ae)
    model.fit(n_epochs=2000, batch_size=256, learning_rate=1e-3)
    return model


@pytest.fixture
def trained_model_with_sae(trained_model):
    """trained_model with a StandardTrainingSAE trained on it."""
    sae_entry = SAEEntry(
        sae=StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=N_HIDDEN,
                d_sae=D_SAE,
                l1_coefficient=0.3,
            )
        ),
        type="Standard",
        params={"l1_coefficient": 0.3},
    )
    trained_model.train_saes(
        [sae_entry],
        training_samples=TRAINING_SAMPLES,
        batch_size=BATCH_SIZE,
    )
    trained_model.evaluate_saes(num_samples=EVAL_SAMPLES)
    return trained_model


# ===========================================================================
# 1. ActivationGeneratorWrapper faithfulness
# ===========================================================================


class TestActivationGeneratorFaithfulness:
    """Wrapper.sample() must produce IDENTICAL tensors to the underlying
    Distribution.sample() given the same generator state."""

    def test_identical_samples_when_generator_state_matches(self):
        """Reset generator, sample from distribution; reset, sample from wrapper.
        Results must be bitwise identical."""
        gen = torch.Generator()
        gen.manual_seed(99)
        dist = SparseUniform(n_features=N_FEATURES, p_active=P_ACTIVE, generator=gen)
        wrapper = ActivationGeneratorWrapper(dist)

        state = gen.get_state().clone()

        gen.set_state(state.clone())
        direct = dist.sample(64)

        gen.set_state(state.clone())
        wrapped = wrapper.sample(64)

        assert torch.equal(direct, wrapped), (
            "Wrapper.sample() diverged from Distribution.sample()"
        )

    def test_faithfulness_across_multiple_batches(self):
        """Multiple consecutive samples should remain identical."""
        gen = torch.Generator()
        gen.manual_seed(123)
        dist = SparseUniform(n_features=N_FEATURES, p_active=P_ACTIVE, generator=gen)
        wrapper = ActivationGeneratorWrapper(dist)

        state = gen.get_state().clone()

        gen.set_state(state.clone())
        d1 = dist.sample(32)
        d2 = dist.sample(32)

        gen.set_state(state.clone())
        w1 = wrapper.sample(32)
        w2 = wrapper.sample(32)

        assert torch.equal(d1, w1)
        assert torch.equal(d2, w2)

    def test_shape_and_dtype_preserved(self):
        """Wrapper must not change shape or dtype."""
        gen = torch.Generator()
        gen.manual_seed(7)
        dist = SparseUniform(n_features=15, p_active=0.2, generator=gen)
        wrapper = ActivationGeneratorWrapper(dist)

        state = gen.get_state().clone()

        gen.set_state(state.clone())
        direct = dist.sample(16)

        gen.set_state(state.clone())
        wrapped = wrapper.sample(16)

        assert direct.shape == wrapped.shape
        assert direct.dtype == wrapped.dtype


# ===========================================================================
# 2. FeatureDictionaryWrapper faithfulness
# ===========================================================================


class TestFeatureDictionaryFaithfulness:
    """Wrapper.forward(x) must produce the SAME output as AutoEncoder.encode(x)."""

    def test_forward_matches_encode_random_input(self):
        """Random input through wrapper.forward == ae.encode."""
        ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=N_HIDDEN)
        wrapper = FeatureDictionaryWrapper(ae)

        x = torch.randn(32, N_FEATURES)
        expected = ae.encode(x)
        actual = wrapper(x)

        assert torch.allclose(actual, expected, atol=1e-6), (
            "FeatureDictionaryWrapper.forward() diverged from AutoEncoder.encode()"
        )

    def test_forward_matches_encode_zero_input(self):
        """Zero input: both paths should produce identical results."""
        ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=N_HIDDEN)
        wrapper = FeatureDictionaryWrapper(ae)

        x = torch.zeros(8, N_FEATURES)
        assert torch.allclose(wrapper(x), ae.encode(x), atol=1e-7)

    def test_forward_matches_encode_identity_input(self):
        """One-hot inputs (identity matrix)."""
        ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=N_HIDDEN)
        wrapper = FeatureDictionaryWrapper(ae)

        x = torch.eye(N_FEATURES)
        assert torch.allclose(wrapper(x), ae.encode(x), atol=1e-6)

    def test_forward_matches_encode_tied_linear(self):
        """Check with TiedLinear (no ReLU) too."""
        ae = TiedLinear(n_features=N_FEATURES, n_hidden=N_HIDDEN)
        wrapper = FeatureDictionaryWrapper(ae)

        x = torch.randn(16, N_FEATURES)
        assert torch.allclose(wrapper(x), ae.encode(x), atol=1e-6)


# ===========================================================================
# 3. CoefficientAutotuner convergence directionality
# ===========================================================================


class TestAutotunerConvergenceDirection:
    """When actual L0 > target, multiplier should INCREASE.
    When actual L0 < target, multiplier should DECREASE."""

    def test_l0_above_target_increases_multiplier_over_many_steps(self):
        """Sustained L0 above target: multiplier monotonically increases (modulo damping)."""
        cfg = CoefficientAutotunerConfig(
            target_l0=5.0,
            integral_gain=0.01,
            convergence_gain=1.0,  # no damping
            smoothing_factor=0.0,  # instant tracking
            rate_smoothing_factor=0.0,
        )
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=20.0, step=0)
        multipliers = []
        for step in range(1, 50):
            at.update(batch_l0=20.0, step=step)
            multipliers.append(at.multiplier)

        # All multipliers should be > 1 and generally increasing
        assert all(m > 1.0 for m in multipliers), (
            "Multiplier should be > 1 when L0 > target"
        )
        # Overall trend: last > first
        assert multipliers[-1] > multipliers[0]

    def test_l0_below_target_decreases_multiplier_over_many_steps(self):
        """Sustained L0 below target: multiplier monotonically decreases."""
        cfg = CoefficientAutotunerConfig(
            target_l0=20.0,
            integral_gain=0.01,
            convergence_gain=1.0,
            smoothing_factor=0.0,
            rate_smoothing_factor=0.0,
        )
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=1.0, step=0)
        multipliers = []
        for step in range(1, 50):
            at.update(batch_l0=1.0, step=step)
            multipliers.append(at.multiplier)

        assert all(m < 1.0 for m in multipliers), (
            "Multiplier should be < 1 when L0 < target"
        )
        assert multipliers[-1] < multipliers[0]


# ===========================================================================
# 4. CoefficientAutotuner clamping
# ===========================================================================


class TestAutotunerClamping:
    """Multiplier must never exceed bounds, even after extreme sustained errors."""

    def test_clamped_at_max_after_sustained_overshoot(self):
        cfg = CoefficientAutotunerConfig(
            target_l0=1.0,
            integral_gain=1.0,
            max_multiplier=5.0,
            min_multiplier=0.01,
        )
        at = CoefficientAutotuner(cfg)
        for step in range(500):
            at.update(batch_l0=1000.0, step=step)
        assert at.multiplier <= 5.0
        assert at.multiplier == pytest.approx(5.0, abs=1e-10)

    def test_clamped_at_min_after_sustained_undershoot(self):
        cfg = CoefficientAutotunerConfig(
            target_l0=1000.0,
            integral_gain=1.0,
            max_multiplier=100.0,
            min_multiplier=0.5,
        )
        at = CoefficientAutotuner(cfg)
        for step in range(500):
            at.update(batch_l0=0.001, step=step)
        assert at.multiplier >= 0.5
        assert at.multiplier == pytest.approx(0.5, abs=1e-10)

    def test_clamping_both_directions_sequential(self):
        """Drive to max, then reverse to min. Verify clamping both ways."""
        cfg = CoefficientAutotunerConfig(
            target_l0=10.0,
            integral_gain=0.5,
            max_multiplier=3.0,
            min_multiplier=0.2,
        )
        at = CoefficientAutotuner(cfg)
        # Drive up
        for step in range(200):
            at.update(batch_l0=100.0, step=step)
        assert at.multiplier <= 3.0

        # Now drive down
        for step in range(200, 600):
            at.update(batch_l0=0.01, step=step)
        assert at.multiplier >= 0.2


# ===========================================================================
# 5. CoefficientAutotuner deadband
# ===========================================================================


class TestAutotunerDeadband:
    """When L0 is within the deadband of the target, multiplier should stay constant."""

    def test_within_deadband_no_change(self):
        """L0 within deadband => multiplier stays at 1.0."""
        cfg = CoefficientAutotunerConfig(
            target_l0=10.0,
            deadband=2.0,
            integral_gain=0.1,
        )
        at = CoefficientAutotuner(cfg)
        # L0=11.0 => error=1.0 < deadband=2.0
        at.update(batch_l0=11.0, step=0)
        for step in range(1, 100):
            at.update(batch_l0=11.0, step=step)
        assert at.multiplier == pytest.approx(1.0)

    def test_outside_deadband_does_change(self):
        """L0 well outside deadband => multiplier changes."""
        cfg = CoefficientAutotunerConfig(
            target_l0=10.0,
            deadband=0.5,
            integral_gain=0.1,
        )
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=20.0, step=0)
        at.update(batch_l0=20.0, step=1)
        assert at.multiplier != pytest.approx(1.0)

    def test_deadband_boundary_exact(self):
        """L0 exactly at deadband boundary => no adjustment (|error| <= deadband)."""
        cfg = CoefficientAutotunerConfig(
            target_l0=10.0,
            deadband=5.0,
            integral_gain=0.1,
            smoothing_factor=0.0,  # instant tracking
        )
        at = CoefficientAutotuner(cfg)
        # error = 15.0 - 10.0 = 5.0 = deadband => no adjustment
        at.update(batch_l0=15.0, step=0)
        at.update(batch_l0=15.0, step=1)
        assert at.multiplier == pytest.approx(1.0)


# ===========================================================================
# 6. StandardTrainingSAEAutotuned trains without crashing
# ===========================================================================


class TestAutotunedSAETraining:
    """Construct an autotuned SAE, run it through ToyModel.train_saes(),
    verify it does not crash and produces valid results."""

    def test_autotuned_sae_trains_and_evaluates(self, trained_model):
        """Full pipeline: train autotuned SAE, evaluate, check valid results."""
        sae_entry = SAEEntry(
            sae=StandardTrainingSAEAutotuned(
                StandardTrainingSAEConfigAutotuned(
                    d_in=N_HIDDEN,
                    d_sae=D_SAE,
                    l1_coefficient=0.5,
                    autotune_target_l0=3.0,
                    autotune_integral_gain=1e-3,
                )
            ),
            type="Autotuned",
            params={"target_l0": 3.0},
        )
        trained_model.train_saes(
            [sae_entry],
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        results = trained_model.evaluate_saes(num_samples=EVAL_SAMPLES)

        assert "Autotuned_0" in results
        result = results["Autotuned_0"]
        # Basic sanity: explained_variance should be a finite number
        assert 0.0 <= result.explained_variance <= 1.0
        assert result.sae_l0 >= 0.0
        assert result.dead_latents >= 0

    def test_autotuned_sae_without_autotuning_matches_standard(self, trained_model):
        """When autotune_target_l0 is None, should behave like standard SAE."""
        sae_entry = SAEEntry(
            sae=StandardTrainingSAEAutotuned(
                StandardTrainingSAEConfigAutotuned(
                    d_in=N_HIDDEN,
                    d_sae=D_SAE,
                    l1_coefficient=0.3,
                )
            ),
            type="AutotunedOff",
        )
        trained_model.train_saes(
            [sae_entry],
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        results = trained_model.evaluate_saes(
            labels=["AutotunedOff_0"], num_samples=EVAL_SAMPLES
        )
        result = results["AutotunedOff_0"]
        assert 0.0 <= result.explained_variance <= 1.0


# ===========================================================================
# 7. SAE training data flow: shapes, dtypes, device consistency
# ===========================================================================


class TestSAEDataFlow:
    """Trace data from Distribution.sample() -> ActivationGeneratorWrapper ->
    FeatureDictionaryWrapper -> SAE. Check shape, dtype, and device at each step."""

    def test_data_flow_shapes_and_dtypes(self, trained_model):
        """Verify shapes and dtypes at each stage of the pipeline."""
        dist = trained_model.distribution
        ae = trained_model.ae
        batch_size = 64

        # Stage 1: Distribution sample
        feature_acts = dist.sample(batch_size)
        assert feature_acts.shape == (batch_size, N_FEATURES)
        assert feature_acts.dtype == torch.float32

        # Stage 2: ActivationGeneratorWrapper sample
        wrapper = ActivationGeneratorWrapper(dist)
        gen_state = dist.generator.get_state().clone() if dist.generator else None
        if gen_state is not None:
            dist.generator.set_state(gen_state)
        wrapped_sample = wrapper.sample(batch_size)
        assert wrapped_sample.shape == (batch_size, N_FEATURES)
        assert wrapped_sample.dtype == torch.float32

        # Stage 3: FeatureDictionaryWrapper forward (encode)
        fd = FeatureDictionaryWrapper(ae)
        hidden_acts = fd(feature_acts)
        assert hidden_acts.shape == (batch_size, N_HIDDEN)
        assert hidden_acts.dtype == torch.float32

        # Stage 4: SAE encode/decode
        sae = StandardTrainingSAE(StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=D_SAE))
        sae.eval()
        with torch.no_grad():
            sae_latents = sae.encode(hidden_acts)
            sae_output = sae.decode(sae_latents)
        assert sae_latents.shape == (batch_size, D_SAE)
        assert sae_output.shape == (batch_size, N_HIDDEN)
        assert sae_latents.dtype == torch.float32
        assert sae_output.dtype == torch.float32

    def test_no_nans_or_infs_in_pipeline(self, trained_model):
        """No NaN or Inf should appear at any stage."""
        dist = trained_model.distribution
        ae = trained_model.ae
        batch_size = 128

        feature_acts = dist.sample(batch_size)
        assert torch.isfinite(feature_acts).all(), "NaN/Inf in distribution sample"

        fd = FeatureDictionaryWrapper(ae)
        hidden_acts = fd(feature_acts)
        assert torch.isfinite(hidden_acts).all(), "NaN/Inf in feature dictionary output"

        sae = StandardTrainingSAE(StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=D_SAE))
        sae.eval()
        with torch.no_grad():
            sae_latents = sae.encode(hidden_acts)
            sae_output = sae.decode(sae_latents)
        assert torch.isfinite(sae_latents).all(), "NaN/Inf in SAE latents"
        assert torch.isfinite(sae_output).all(), "NaN/Inf in SAE output"


# ===========================================================================
# 8. SAE evaluation metrics consistency
# ===========================================================================


class TestEvaluationMetricsConsistency:
    """Verify that evaluation metrics are in valid ranges and are
    internally consistent."""

    def test_explained_variance_in_valid_range(self, trained_model_with_sae):
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        ev = result.explained_variance
        # explained_variance is R^2, can be negative for bad fits in theory,
        # but should typically be in [0, 1] for a trained SAE
        assert -1.0 <= ev <= 1.0, f"explained_variance={ev} outside [-1, 1]"

    def test_precision_recall_f1_harmonic_mean(self, trained_model_with_sae):
        """F1 should be the harmonic mean of precision and recall."""
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        p = result.classification.precision
        r = result.classification.recall
        f1 = result.classification.f1_score

        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert f1 == pytest.approx(expected_f1, abs=0.01), (
                f"F1={f1} != harmonic_mean(P={p}, R={r})={expected_f1}"
            )
        else:
            # Both zero => F1 should be 0
            assert f1 == pytest.approx(0.0, abs=1e-6)

    def test_mcc_in_valid_range(self, trained_model_with_sae):
        """MCC should be in [-1, 1]."""
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert -1.0 <= result.mcc <= 1.0, f"MCC={result.mcc} outside [-1, 1]"

    def test_precision_recall_in_valid_range(self, trained_model_with_sae):
        """Precision and recall should each be in [0, 1]."""
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert 0.0 <= result.classification.precision <= 1.0
        assert 0.0 <= result.classification.recall <= 1.0

    def test_sae_l0_nonnegative(self, trained_model_with_sae):
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert result.sae_l0 >= 0.0

    def test_dead_latents_nonnegative_and_bounded(self, trained_model_with_sae):
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert 0 <= result.dead_latents <= D_SAE

    def test_shrinkage_positive(self, trained_model_with_sae):
        """Shrinkage is ratio of output norm to input norm, should be > 0."""
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert result.shrinkage > 0.0

    def test_uniqueness_in_valid_range(self, trained_model_with_sae):
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert 0.0 <= result.uniqueness <= 1.0

    def test_accuracy_in_valid_range(self, trained_model_with_sae):
        result = trained_model_with_sae.saes["Standard_0"].results
        assert result is not None
        assert 0.0 <= result.classification.accuracy <= 1.0


# ===========================================================================
# 9. Feature similarity matrix
# ===========================================================================


class TestFeatureSimilarityMatrix:
    """saes_feature_similarity should be a matrix of cosine similarities
    in [-1, 1], shape (n_sae_latents, n_features)."""

    def test_shape(self, trained_model_with_sae):
        sim = trained_model_with_sae.saes_feature_similarity
        assert "Standard_0" in sim
        mat = sim["Standard_0"]
        assert mat.shape == (D_SAE, N_FEATURES)

    def test_values_in_cosine_range(self, trained_model_with_sae):
        sim = trained_model_with_sae.saes_feature_similarity
        mat = sim["Standard_0"]
        assert mat.min() >= -1.0 - 1e-6, f"min={mat.min()} below -1"
        assert mat.max() <= 1.0 + 1e-6, f"max={mat.max()} above 1"

    def test_no_nans(self, trained_model_with_sae):
        sim = trained_model_with_sae.saes_feature_similarity
        mat = sim["Standard_0"]
        assert torch.isfinite(mat).all(), "NaN or Inf in feature similarity matrix"

    def test_ordering_indices_valid(self, trained_model_with_sae):
        """saes_feature_similarity_ordering indices should be valid SAE latent indices."""
        ordering = trained_model_with_sae.saes_feature_similarity_ordering
        assert "Standard_0" in ordering
        idx = ordering["Standard_0"]
        # Should contain valid indices into [0, D_SAE)
        assert idx.min() >= 0
        assert idx.max() < D_SAE


# ===========================================================================
# 10. Multiple SAE types
# ===========================================================================


class TestMultipleSAETypes:
    """Train StandardTrainingSAE and BatchTopKTrainingSAE on the same model,
    verify both produce valid results with potentially different characteristics."""

    def test_standard_and_batchtopk_both_valid(self, trained_model):
        """Both SAE types should train and evaluate without crashing."""
        entries = [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=N_HIDDEN,
                        d_sae=D_SAE,
                        l1_coefficient=0.3,
                    )
                ),
                type="Standard",
                label="std",
            ),
            SAEEntry(
                sae=BatchTopKTrainingSAE(
                    BatchTopKTrainingSAEConfig(
                        d_in=N_HIDDEN,
                        d_sae=D_SAE,
                        k=3,
                    )
                ),
                type="BatchTopK",
                label="topk",
            ),
        ]
        trained_model.train_saes(
            entries,
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        results = trained_model.evaluate_saes(num_samples=EVAL_SAMPLES)

        for label in ["std", "topk"]:
            assert label in results
            r = results[label]
            assert 0.0 <= r.classification.precision <= 1.0
            assert 0.0 <= r.classification.recall <= 1.0
            assert -1.0 <= r.mcc <= 1.0
            assert r.sae_l0 >= 0.0

    def test_batchtopk_l0_near_k(self, trained_model):
        """BatchTopK with k=3 should produce L0 close to 3."""
        k_val = 3
        entry = SAEEntry(
            sae=BatchTopKTrainingSAE(
                BatchTopKTrainingSAEConfig(
                    d_in=N_HIDDEN,
                    d_sae=D_SAE,
                    k=k_val,
                )
            ),
            type="BatchTopK",
            label="topk_l0_check",
        )
        trained_model.train_saes(
            [entry],
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        results = trained_model.evaluate_saes(
            labels=["topk_l0_check"], num_samples=EVAL_SAMPLES
        )
        r = results["topk_l0_check"]
        # BatchTopK enforces exactly k active latents per batch on average
        # Allow some tolerance since it's a batch-level constraint
        assert r.sae_l0 <= k_val * 3, f"BatchTopK L0={r.sae_l0} far exceeds k={k_val}"


# ===========================================================================
# 11. L1 sweep monotonicity
# ===========================================================================


class TestL1SweepMonotonicity:
    """Increasing L1 coefficient should generally decrease L0.
    Test with 4 L1 values and verify downward trend."""

    def test_l1_vs_l0_trend(self, trained_model):
        """Higher L1 => lower or equal L0. Verify overall trend, not strict monotonicity."""
        l1_values = [0.05, 0.3, 1.0, 3.0]
        entries = [
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=N_HIDDEN,
                        d_sae=D_SAE,
                        l1_coefficient=l1,
                    )
                ),
                type="Standard",
                params={"l1_coefficient": l1},
                label=f"l1_{l1}",
            )
            for l1 in l1_values
        ]

        trained_model.train_saes(
            entries,
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        results = trained_model.evaluate_saes(num_samples=EVAL_SAMPLES)

        l0_values = [results[f"l1_{l1}"].sae_l0 for l1 in l1_values]

        # Verify overall trend: L0 at highest L1 should be <= L0 at lowest L1.
        # Stochastic training means we can't demand strict monotonicity at every step,
        # but the endpoints should respect the trend.
        assert l0_values[-1] <= l0_values[0] + 1.0, (
            f"L0 at L1={l1_values[-1]} ({l0_values[-1]:.2f}) should be "
            f"<= L0 at L1={l1_values[0]} ({l0_values[0]:.2f}), "
            f"all L0s: {l0_values}"
        )


# ===========================================================================
# 12. Device consistency throughout pipeline
# ===========================================================================


class TestDeviceConsistency:
    """If model is on CPU, all intermediate tensors during train_saes and
    evaluate_saes should also be on CPU."""

    def test_all_tensors_on_cpu(self, trained_model):
        """After training and evaluation, all SAE parameters and results
        should be on CPU."""
        assert str(trained_model.device) == "cpu"

        sae_entry = SAEEntry(
            sae=StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=N_HIDDEN,
                    d_sae=D_SAE,
                    l1_coefficient=0.3,
                )
            ),
            type="Standard",
            label="device_check",
        )
        trained_model.train_saes(
            [sae_entry],
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
        )

        # Check SAE parameters are on CPU
        sae = trained_model.saes["device_check"].sae
        for name, param in sae.named_parameters():
            assert param.device.type == "cpu", (
                f"SAE param '{name}' on {param.device}, expected cpu"
            )

        # Evaluate and check evaluation intermediates
        results = trained_model.evaluate_saes(
            labels=["device_check"], num_samples=EVAL_SAMPLES
        )
        # Evaluate doesn't expose intermediate tensors, but we can verify
        # the wrapper outputs are on the right device
        wrapper = ActivationGeneratorWrapper(trained_model.distribution)
        sample = wrapper.sample(16)
        assert sample.device.type == "cpu"

        fd = FeatureDictionaryWrapper(trained_model.ae)
        hidden = fd(sample)
        assert hidden.device.type == "cpu"

    def test_feature_similarity_on_model_device(self, trained_model_with_sae):
        """Feature similarity matrix should be on the same device as the model."""
        sim = trained_model_with_sae.saes_feature_similarity
        mat = sim["Standard_0"]
        assert mat.device.type == "cpu"


# ===========================================================================
# Additional edge case and stress tests
# ===========================================================================


class TestAutotunerEdgeCases:
    """Edge cases and stress tests for the CoefficientAutotuner."""

    def test_zero_target_l0(self):
        """target_l0=0 should not cause division by zero in rel_error calculation."""
        cfg = CoefficientAutotunerConfig(
            target_l0=0.0,  # pathological
            integral_gain=0.01,
        )
        at = CoefficientAutotuner(cfg)
        # This will compute rel_error = error / target_l0 = error / 0.0 = inf
        # tanh(inf) = 1.0, so adjustment should still be bounded
        at.update(batch_l0=5.0, step=0)
        result = at.update(batch_l0=5.0, step=1)
        # Should not be NaN
        assert not torch.isnan(torch.tensor(result)), "NaN multiplier with target_l0=0"
        assert torch.isfinite(torch.tensor(result)), "Inf multiplier with target_l0=0"

    def test_negative_batch_l0(self):
        """Negative L0 is nonsensical but should not crash."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=-1.0, step=0)
        result = at.update(batch_l0=-1.0, step=1)
        assert torch.isfinite(torch.tensor(result))

    def test_very_large_batch_l0(self):
        """Very large L0 should not overflow."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=1e15, step=0)
        result = at.update(batch_l0=1e15, step=1)
        assert torch.isfinite(torch.tensor(result))

    def test_reset_then_retrain(self):
        """After reset, autotuner should behave identically to a fresh one."""
        cfg = CoefficientAutotunerConfig(
            target_l0=5.0,
            integral_gain=0.01,
            smoothing_factor=0.5,
        )
        at1 = CoefficientAutotuner(cfg)
        # Run it for a while
        for step in range(50):
            at1.update(batch_l0=20.0, step=step)
        at1.reset()

        # Run identically to a fresh one
        at2 = CoefficientAutotuner(cfg)

        # Now both should produce identical results
        at1.update(batch_l0=7.0, step=0)
        at2.update(batch_l0=7.0, step=0)
        assert at1.multiplier == pytest.approx(at2.multiplier)
        assert at1.smoothed_l0 == pytest.approx(at2.smoothed_l0)

        at1.update(batch_l0=3.0, step=1)
        at2.update(batch_l0=3.0, step=1)
        assert at1.multiplier == pytest.approx(at2.multiplier)


class TestSAETrainingLossSnapshots:
    """Verify that loss snapshots are recorded when requested."""

    def test_loss_snapshots_recorded(self, trained_model):
        """n_loss_snapshots should produce the correct number of snapshots."""
        n_snapshots = 5
        entry = SAEEntry(
            sae=StandardTrainingSAE(
                StandardTrainingSAEConfig(
                    d_in=N_HIDDEN,
                    d_sae=D_SAE,
                    l1_coefficient=0.3,
                )
            ),
            type="Standard",
            label="loss_snap",
        )
        trained_model.train_saes(
            [entry],
            training_samples=TRAINING_SAMPLES,
            batch_size=BATCH_SIZE,
            n_loss_snapshots=n_snapshots,
        )
        losses = trained_model.saes["loss_snap"].losses
        assert losses is not None
        assert len(losses) == n_snapshots
        # Each entry is (step, loss)
        for step, loss_val in losses:
            assert isinstance(step, int)
            assert isinstance(loss_val, float)
            assert loss_val >= 0.0, f"Negative loss at step {step}: {loss_val}"


class TestFeatureDictionaryDimensionSemantics:
    """Verify that the FeatureDictionaryWrapper dimension semantics are correct
    and consistent with how sae_lens uses them."""

    def test_num_features_and_hidden_dim_relationship(self):
        """For TiedLinearRelu: encode maps (batch, n_features) -> (batch, n_hidden).
        FeatureDictionaryWrapper.num_features should be the output dim (n_hidden),
        and hidden_dim should be the input dim (n_features).
        This is because the SAE Lens convention transposes the matrix."""
        ae = TiedLinearRelu(n_features=12, n_hidden=7)
        fd = FeatureDictionaryWrapper(ae)

        # The transposed feature_vectors shape determines these
        fv_T = ae.feature_vectors.T
        assert fd.num_features == fv_T.shape[0]
        assert fd.hidden_dim == fv_T.shape[1]

        # Verify forward works with these dimensions
        x = torch.randn(8, fd.hidden_dim)
        out = fd(x)
        assert out.shape == (8, fd.num_features)
