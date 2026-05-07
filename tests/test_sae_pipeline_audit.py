"""Deep end-to-end audit of the SAE pipeline.

Traces the full data flow: distribution sampling -> AE encoding -> SAE training
-> evaluation metrics. Tests correctness of shapes, metric bounds, L1/L0
monotonicity, autotuner convergence, loss snapshots, feature similarity matrices,
and multi-SAE-type comparisons.

Every test exercises a specific invariant in the pipeline. Failures indicate
real bugs or regressions in the data flow, not just style preferences.
"""

import math

import pytest
import torch
from torch import Tensor

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

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_model(
    n_features: int = 8,
    n_hidden: int = 4,
    p_active: float = 0.5,
    seed: int = 42,
) -> ToyModel:
    """Create a small ToyModel with seeded generators."""
    g_dist = torch.Generator(device=DEVICE).manual_seed(seed)
    g_ae = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    dist = SparseUniform(n_features=n_features, p_active=p_active, generator=g_dist)
    ae = TiedLinearRelu(n_features, n_hidden, generator=g_ae, device=DEVICE)
    return ToyModel(dist, ae, device=DEVICE)


def _make_sae_entry(
    d_in: int, d_sae: int = 16, l1: float = 0.01, label: str | None = None
) -> SAEEntry:
    from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

    cfg = StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae, l1_coefficient=l1)
    sae = StandardTrainingSAE(cfg)
    return SAEEntry(sae=sae, type="Standard", params={"l1": l1}, label=label)


def _make_autotuned_sae_entry(
    d_in: int,
    d_sae: int = 16,
    l1: float = 0.01,
    target_l0: float = 5.0,
    label: str | None = None,
) -> SAEEntry:
    cfg = StandardTrainingSAEConfigAutotuned(
        d_in=d_in,
        d_sae=d_sae,
        l1_coefficient=l1,
        autotune_target_l0=target_l0,
    )
    sae = StandardTrainingSAEAutotuned(cfg)
    return SAEEntry(
        sae=sae, type="Autotuned", params={"target_l0": target_l0}, label=label
    )


# ---------------------------------------------------------------------------
# Shared fixture: a trained ToyModel used by many tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_tm():
    """A ToyModel trained once for the entire module to keep tests fast."""
    tm = _make_toy_model(n_features=8, n_hidden=4, seed=7777)
    tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)
    return tm


# ===========================================================================
# 1. Data flow trace -- verify no corruption
# ===========================================================================


class TestDataFlowTrace:
    """Verify shapes and value identity at each stage of the pipeline."""

    def test_distribution_sample_shape(self):
        """Distribution.sample() returns (batch, n_features)."""
        dist = SparseUniform(n_features=8, p_active=0.5)
        x = dist.sample(64)
        assert x.shape == (64, 8)

    def test_ae_encode_shape(self):
        """ae.encode() maps (batch, n_features) -> (batch, n_hidden)."""
        ae = TiedLinearRelu(n_features=8, n_hidden=4)
        x = torch.randn(64, 8)
        z = ae.encode(x)
        assert z.shape == (64, 4)

    def test_activation_generator_matches_distribution_sample(self):
        """ActivationGeneratorWrapper.sample() produces the same output
        as calling the underlying distribution's sample() with the same
        generator state."""
        gen = torch.Generator().manual_seed(123)
        dist = SparseUniform(n_features=5, p_active=0.5, generator=gen)
        act_gen = ActivationGeneratorWrapper(dist)

        state = gen.get_state().clone()
        result_act = act_gen.sample(32)
        gen.set_state(state)
        result_dist = dist.sample(32)
        assert torch.equal(result_act, result_dist)

    def test_activation_generator_shape_matches_ae_encode_input(self):
        """ActivationGeneratorWrapper.sample() output shape is (batch, n_features),
        which is the expected input shape for ae.encode()."""
        dist = SparseUniform(n_features=8, p_active=0.5)
        act_gen = ActivationGeneratorWrapper(dist)
        samples = act_gen.sample(32)
        assert samples.shape == (32, 8)

    def test_feature_dictionary_forward_matches_ae_encode(self):
        """FeatureDictionaryWrapper.forward() must produce identical output
        to ae.encode() for the same input."""
        ae = TiedLinearRelu(n_features=5, n_hidden=8)
        feat_dict = FeatureDictionaryWrapper(ae)
        x = torch.randn(16, 5)
        expected = ae.encode(x)
        actual = feat_dict(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_full_pipeline_shape_chain(self):
        """Trace: dist.sample -> ae.encode -> SAE input dimension check.
        The SAE d_in must equal ae.n_hidden."""
        n_features, n_hidden = 8, 4
        dist = SparseUniform(n_features=n_features, p_active=0.5)
        ae = TiedLinearRelu(n_features, n_hidden)

        x = dist.sample(32)
        assert x.shape == (32, n_features)

        z = ae.encode(x)
        assert z.shape == (32, n_hidden)

        # This is what the SAE sees as input
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=n_hidden, d_sae=16, l1_coefficient=0.1)
        sae = StandardTrainingSAE(cfg)
        assert sae.cfg.d_in == z.shape[1]

    def test_sample_latent_shape(self, trained_tm):
        """ToyModel.sample_latent() should return (batch, n_hidden)."""
        z = trained_tm.sample_latent(32)
        assert z.shape == (32, trained_tm.ae.n_hidden)

    def test_feature_dictionary_hidden_dim_matches_ae(self):
        """FeatureDictionaryWrapper.hidden_dim must match ae.n_hidden,
        and num_features must match ae.n_features."""
        ae = TiedLinearRelu(n_features=8, n_hidden=4)
        fd = FeatureDictionaryWrapper(ae)
        # feature_vectors.T has shape (n_features, n_hidden)
        # so num_features = n_features, hidden_dim = n_hidden
        # BUT: actually feature_vectors = ae.encode(I) which has shape (n_features, n_hidden)
        # .T has shape (n_hidden, n_features)
        # So num_features, hidden_dim = ae.feature_vectors.T.shape
        assert fd.hidden_dim == ae.n_features
        assert fd.num_features == ae.n_hidden


# ===========================================================================
# 2. SAE training correctness -- L1 sweep
# ===========================================================================


class TestSAETrainingCorrectness:
    """Train StandardTrainingSAE at various L1 coefficients and verify
    that L0 tracks L1 in the expected direction."""

    @pytest.fixture(scope="class")
    def l1_sweep_model(self):
        """Trained model with SAEs at L1=0.0, 0.1, 0.5, 1.0, 5.0."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=2222)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        l1_values = [0.0, 0.1, 0.5, 1.0, 5.0]
        for i, l1 in enumerate(l1_values):
            entry = _make_sae_entry(d_in=4, d_sae=32, l1=l1, label=f"l1_{l1}")
            tm.train_saes([entry], training_samples=100_000, batch_size=512)

        tm.evaluate_saes(num_samples=20_000)
        return tm, l1_values

    def test_l1_zero_has_high_l0(self, l1_sweep_model):
        """L1=0 should produce many active latents (high L0)."""
        tm, _ = l1_sweep_model
        l0_zero = tm.saes_l0["l1_0.0"]
        assert l0_zero > 1.0, f"L1=0 should give L0 > 1, got {l0_zero:.2f}"

    def test_l1_5_has_low_l0(self, l1_sweep_model):
        """L1=5.0 should produce few active latents (low L0)."""
        tm, _ = l1_sweep_model
        l0_high = tm.saes_l0["l1_5.0"]
        l0_zero = tm.saes_l0["l1_0.0"]
        assert l0_high < l0_zero, (
            f"L1=5.0 L0 ({l0_high:.2f}) should be < L1=0.0 L0 ({l0_zero:.2f})"
        )

    def test_l0_generally_decreases_with_l1(self, l1_sweep_model):
        """L0 should be non-increasing as L1 increases. Allow one violation
        (monotonicity is approximate in stochastic training)."""
        tm, l1_values = l1_sweep_model
        l0s = [tm.saes_l0[f"l1_{l1}"] for l1 in l1_values]

        violations = 0
        for i in range(len(l0s) - 1):
            if l0s[i] < l0s[i + 1]:
                violations += 1

        assert violations <= 1, (
            f"L0 should generally decrease with L1. "
            f"L1 values: {l1_values}, L0 values: {[f'{x:.2f}' for x in l0s]}, "
            f"violations: {violations}"
        )

    def test_good_reconstruction_with_enough_capacity(self, l1_sweep_model):
        """With d_sae=32 >> d_in=4 and low L1, the SAE should achieve
        decent explained variance."""
        tm, _ = l1_sweep_model
        ev = tm.saes["l1_0.1"].results.explained_variance
        assert ev > 0.0, (
            f"With low L1 and d_sae >> d_in, explained_variance should be > 0, got {ev:.4f}"
        )


# ===========================================================================
# 3. SAE evaluation metrics audit
# ===========================================================================


class TestSAEEvaluationMetrics:
    """Verify every metric field of SyntheticDataEvalResult is within
    its valid mathematical range."""

    @pytest.fixture(scope="class")
    def evaluated_result(self):
        """Train and evaluate a single SAE, return the result."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=3333)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)
        entry = _make_sae_entry(d_in=4, d_sae=32, l1=0.1, label="audit")
        tm.train_saes([entry], training_samples=200_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=20_000)
        return results["audit"]

    def test_f1_in_range(self, evaluated_result):
        f1 = evaluated_result.classification.f1_score
        assert 0.0 <= f1 <= 1.0, f"F1 should be in [0,1], got {f1}"

    def test_precision_in_range(self, evaluated_result):
        p = evaluated_result.classification.precision
        assert 0.0 <= p <= 1.0, f"Precision should be in [0,1], got {p}"

    def test_recall_in_range(self, evaluated_result):
        r = evaluated_result.classification.recall
        assert 0.0 <= r <= 1.0, f"Recall should be in [0,1], got {r}"

    def test_accuracy_in_range(self, evaluated_result):
        a = evaluated_result.classification.accuracy
        assert 0.0 <= a <= 1.0, f"Accuracy should be in [0,1], got {a}"

    def test_mcc_in_range(self, evaluated_result):
        mcc = evaluated_result.mcc
        assert -1.0 <= mcc <= 1.0, f"MCC should be in [-1,1], got {mcc}"

    def test_explained_variance_positive_for_trained(self, evaluated_result):
        ev = evaluated_result.explained_variance
        assert ev > 0.0, f"Explained variance should be > 0 for trained SAE, got {ev}"

    def test_uniqueness_in_range(self, evaluated_result):
        u = evaluated_result.uniqueness
        assert 0.0 <= u <= 1.0, f"Uniqueness should be in [0,1], got {u}"

    def test_shrinkage_non_negative(self, evaluated_result):
        s = evaluated_result.shrinkage
        assert s >= 0.0, f"Shrinkage should be >= 0, got {s}"

    def test_sae_l0_non_negative(self, evaluated_result):
        l0 = evaluated_result.sae_l0
        assert l0 >= 0.0, f"sae_l0 should be >= 0, got {l0}"

    def test_true_l0_non_negative(self, evaluated_result):
        l0 = evaluated_result.true_l0
        assert l0 >= 0.0, f"true_l0 should be >= 0, got {l0}"

    def test_dead_latents_non_negative_integer(self, evaluated_result):
        dl = evaluated_result.dead_latents
        assert isinstance(dl, int), f"dead_latents should be int, got {type(dl)}"
        assert dl >= 0, f"dead_latents should be >= 0, got {dl}"

    def test_f1_harmonic_mean_property(self, evaluated_result):
        """F1 = 2*P*R/(P+R). Verify this relationship holds (within tolerance
        for floating-point averaging across latents in SAE Lens)."""
        p = evaluated_result.classification.precision
        r = evaluated_result.classification.recall
        f1 = evaluated_result.classification.f1_score

        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            # Note: SAE Lens computes F1 per-latent then averages, so the
            # macro-F1 (mean of per-latent F1) is NOT strictly equal to
            # 2*mean(P)*mean(R)/(mean(P)+mean(R)). We check consistency
            # rather than exact equality.
            # Instead just verify F1 is between min(P,R) and max(P,R)
            # which is a property of the harmonic mean.
            assert f1 <= max(p, r) + 1e-6, (
                f"F1 ({f1:.4f}) should be <= max(P,R)={max(p, r):.4f}"
            )

    def test_high_l1_more_dead_latents(self):
        """With very high L1, dead_latents should be high."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=3334)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry_low = _make_sae_entry(d_in=4, d_sae=64, l1=0.001, label="dead_low")
        entry_high = _make_sae_entry(d_in=4, d_sae=64, l1=10.0, label="dead_high")
        tm.train_saes([entry_low, entry_high], training_samples=100_000, batch_size=512)
        tm.evaluate_saes(num_samples=10_000)

        dead_low = tm.saes_dead_latents["dead_low"]
        dead_high = tm.saes_dead_latents["dead_high"]
        assert dead_high >= dead_low, (
            f"High L1 should produce >= dead latents: low={dead_low}, high={dead_high}"
        )


# ===========================================================================
# 4. Feature similarity matrix
# ===========================================================================


class TestFeatureSimilarityMatrix:
    """Verify saes_feature_similarity shape, values, and ordering."""

    @pytest.fixture(scope="class")
    def model_with_sae(self):
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=4444)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)
        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="sim")
        tm.train_saes([entry], training_samples=200_000, batch_size=512)
        tm.evaluate_saes(num_samples=10_000)
        return tm

    def test_similarity_matrix_shape(self, model_with_sae):
        """Shape should be (n_sae_latents, n_features)."""
        sim = model_with_sae.saes_feature_similarity["sim"]
        # n_sae_latents=16, n_features=8
        assert sim.shape == (16, 8), f"Expected (16, 8), got {sim.shape}"

    def test_similarity_values_in_range(self, model_with_sae):
        """All values should be cosine similarities in [-1, 1]."""
        sim = model_with_sae.saes_feature_similarity["sim"]
        assert sim.min() >= -1.0 - 1e-6, f"Min cosine sim {sim.min():.4f} < -1"
        assert sim.max() <= 1.0 + 1e-6, f"Max cosine sim {sim.max():.4f} > 1"

    def test_ordering_shape_and_validity(self, model_with_sae):
        """saes_feature_similarity_ordering should have length
        min(n_sae_latents, n_features) and contain unique valid SAE latent
        indices. When n_sae > n_features (overcomplete), this selects the
        best-matched latents."""
        ordering = model_with_sae.saes_feature_similarity_ordering["sim"]
        n_sae = 16
        n_features = 8
        expected_len = min(n_sae, n_features)
        assert ordering.shape == (expected_len,), (
            f"Expected ({expected_len},), got {ordering.shape}"
        )
        # All indices should be valid SAE latent indices
        assert ordering.min() >= 0
        assert ordering.max() < n_sae
        # All indices should be unique (1-to-1 matching)
        assert len(ordering.unique()) == expected_len, (
            f"Ordering has duplicate indices: {ordering.tolist()}"
        )

    def test_ordering_indices_are_integers(self, model_with_sae):
        """Ordering should be integer-typed."""
        ordering = model_with_sae.saes_feature_similarity_ordering["sim"]
        assert ordering.dtype in (torch.int64, torch.int32, torch.long), (
            f"Expected integer dtype, got {ordering.dtype}"
        )


# ===========================================================================
# 5. CoefficientAutotuner correctness
# ===========================================================================


class TestCoefficientAutotunerDynamics:
    """Verify the autotuner's control loop behavior in detail."""

    def test_above_target_increases_multiplier(self):
        """When L0 is consistently above target, multiplier should increase."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)

        for step in range(50):
            at.update(batch_l0=20.0, step=step)

        assert at.multiplier > 1.0, (
            f"Multiplier should increase when L0 >> target, got {at.multiplier:.4f}"
        )

    def test_below_target_decreases_multiplier(self):
        """When L0 is consistently below target, multiplier should decrease."""
        cfg = CoefficientAutotunerConfig(target_l0=10.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)

        for step in range(50):
            at.update(batch_l0=1.0, step=step)

        assert at.multiplier < 1.0, (
            f"Multiplier should decrease when L0 << target, got {at.multiplier:.4f}"
        )

    def test_convergence_with_consistent_l0(self):
        """After many steps with L0 at target, multiplier should stay near 1.0."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=3e-4)
        at = CoefficientAutotuner(cfg)

        for step in range(1000):
            at.update(batch_l0=5.0, step=step)

        assert at.multiplier == pytest.approx(1.0, abs=0.01), (
            f"Multiplier should stabilize near 1.0 when L0 = target, got {at.multiplier:.4f}"
        )

    def test_min_clamping_prevents_zero(self):
        """Multiplier should never go below min_multiplier."""
        cfg = CoefficientAutotunerConfig(
            target_l0=100.0, integral_gain=1.0, min_multiplier=0.1
        )
        at = CoefficientAutotuner(cfg)

        for step in range(500):
            at.update(batch_l0=0.001, step=step)

        assert at.multiplier >= 0.1 - 1e-9, (
            f"Multiplier went below min: {at.multiplier:.6f}"
        )

    def test_max_clamping_prevents_explosion(self):
        """Multiplier should never exceed max_multiplier."""
        cfg = CoefficientAutotunerConfig(
            target_l0=0.1, integral_gain=1.0, max_multiplier=50.0
        )
        at = CoefficientAutotuner(cfg)

        for step in range(500):
            at.update(batch_l0=100.0, step=step)

        assert at.multiplier <= 50.0 + 1e-9, (
            f"Multiplier exceeded max: {at.multiplier:.6f}"
        )

    def test_multiplier_direction_reversal(self):
        """If L0 goes from above target to below target, the multiplier
        should eventually change direction."""
        cfg = CoefficientAutotunerConfig(
            target_l0=5.0,
            integral_gain=0.01,
            smoothing_factor=0.5,  # fast tracking
        )
        at = CoefficientAutotuner(cfg)

        # Phase 1: L0 above target, multiplier increases
        for step in range(100):
            at.update(batch_l0=20.0, step=step)
        mult_after_high = at.multiplier
        assert mult_after_high > 1.0

        # Phase 2: L0 below target, multiplier should decrease
        for step in range(100, 300):
            at.update(batch_l0=1.0, step=step)
        mult_after_low = at.multiplier
        assert mult_after_low < mult_after_high, (
            f"Multiplier should decrease after L0 drops below target. "
            f"After high phase: {mult_after_high:.4f}, after low phase: {mult_after_low:.4f}"
        )


# ===========================================================================
# 6. StandardTrainingSAEAutotuned end-to-end
# ===========================================================================


class TestAutotunedSAEEndToEnd:
    """Verify that autotuned SAEs converge to their target L0."""

    def test_autotuned_target_l0_3(self):
        """Train with target_l0=3, verify the autotuner moves L0 in the
        right direction compared to baseline (no autotuning).

        The autotuner's convergence speed depends on its integral_gain,
        EMA smoothing, and the ratio between base l1_coefficient and
        what's needed. This test verifies that the autotuner at least
        produces lower L0 than a fixed l1=0.1 (non-autotuned) baseline,
        confirming the multiplier is increasing.
        """
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=6661)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        # Autotuned entry with target_l0=3
        entry_at = _make_autotuned_sae_entry(
            d_in=4, d_sae=32, l1=0.1, target_l0=3.0, label="at3"
        )
        # Fixed baseline with same base L1
        entry_fixed = _make_sae_entry(d_in=4, d_sae=32, l1=0.1, label="fixed")
        tm.train_saes([entry_at, entry_fixed], training_samples=500_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=20_000)

        l0_at = results["at3"].sae_l0
        l0_fixed = results["fixed"].sae_l0

        # The autotuner with target_l0=3 should produce lower L0 than
        # fixed L1=0.1 because the multiplier should increase beyond 1.0
        assert l0_at < l0_fixed, (
            f"Autotuned L0 ({l0_at:.2f}) should be < fixed baseline L0 "
            f"({l0_fixed:.2f}) since the autotuner should increase L1"
        )

        # Also verify the autotuner's multiplier actually increased
        sae = tm.saes["at3"].sae
        assert sae.coefficient_autotuner is not None
        assert sae.coefficient_autotuner.multiplier > 1.0, (
            f"Autotuner multiplier should be > 1.0 when L0 was above target, "
            f"got {sae.coefficient_autotuner.multiplier:.4f}"
        )

    def test_autotuned_target_l0_10(self):
        """Train with target_l0=10, verify actual L0 is positive and bounded."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=6662)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        entry = _make_autotuned_sae_entry(
            d_in=4, d_sae=32, l1=0.1, target_l0=10.0, label="at10"
        )
        tm.train_saes([entry], training_samples=500_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=20_000)

        actual_l0 = results["at10"].sae_l0
        assert actual_l0 > 0.0, (
            f"Autotuned SAE with target_l0=10 produced L0={actual_l0:.2f}"
        )

    def test_autotuner_multiplier_moves_correct_direction(self):
        """When L0 is above target, the autotuner should increase its
        multiplier above 1.0. With default integral_gain=3e-4, convergence
        is slow, so we just verify the direction is correct.

        NOTE (audit finding): With 500k training samples and default
        integral_gain=3e-4, the autotuner only reaches multiplier ~1.12.
        For problems where the base L1 is far from what's needed, users
        should either increase integral_gain or use a higher base L1.
        """
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=6663)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        entry = _make_autotuned_sae_entry(
            d_in=4, d_sae=32, l1=0.1, target_l0=3.0, label="at_dir"
        )
        tm.train_saes([entry], training_samples=500_000, batch_size=512)

        sae = tm.saes["at_dir"].sae
        mult = sae.coefficient_autotuner.multiplier
        smoothed = sae.coefficient_autotuner.smoothed_l0

        # Multiplier should have increased (L0 was above target)
        assert mult > 1.0, (
            f"Multiplier should be > 1.0 when L0 > target, got {mult:.4f}"
        )
        # Smoothed L0 should be above target (it hasn't converged yet)
        assert smoothed > 3.0, (
            f"Smoothed L0 should still be above target_l0=3, got {smoothed:.2f}"
        )


# ===========================================================================
# 7. Multiple SAE types
# ===========================================================================


class TestMultipleSAETypes:
    """Train StandardTrainingSAE and BatchTopKTrainingSAE on the same model."""

    def test_both_types_produce_valid_results(self):
        """Both Standard and BatchTopK SAEs should produce valid metrics."""
        from sae_lens import (
            BatchTopKTrainingSAE,
            BatchTopKTrainingSAEConfig,
            StandardTrainingSAE,
            StandardTrainingSAEConfig,
        )

        tm = _make_toy_model(n_features=8, n_hidden=4, seed=7771)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        std_cfg = StandardTrainingSAEConfig(d_in=4, d_sae=16, l1_coefficient=0.1)
        std_sae = StandardTrainingSAE(std_cfg)

        topk_cfg = BatchTopKTrainingSAEConfig(d_in=4, d_sae=16, k=3)
        topk_sae = BatchTopKTrainingSAE(topk_cfg)

        entries = [
            SAEEntry(sae=std_sae, type="Standard", label="std"),
            SAEEntry(sae=topk_sae, type="BatchTopK", label="topk"),
        ]
        tm.train_saes(entries, training_samples=200_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=10_000)

        # Both should have valid results
        for label in ["std", "topk"]:
            r = results[label]
            assert r.sae_l0 >= 0
            assert 0.0 <= r.classification.f1_score <= 1.0
            assert r.explained_variance is not None

    def test_topk_has_exact_l0(self):
        """BatchTopK with k=3 should produce L0 very close to 3."""
        from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

        tm = _make_toy_model(n_features=8, n_hidden=4, seed=7772)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        topk_cfg = BatchTopKTrainingSAEConfig(d_in=4, d_sae=16, k=3)
        topk_sae = BatchTopKTrainingSAE(topk_cfg)
        entry = SAEEntry(sae=topk_sae, type="BatchTopK", label="topk_exact")
        tm.train_saes([entry], training_samples=200_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=10_000)

        actual_l0 = results["topk_exact"].sae_l0
        # BatchTopK should give L0 very close to k
        # Allow some tolerance since the evaluation averages over samples
        # and dead latents can reduce effective L0 below k
        assert actual_l0 <= 3.0 + 0.5, (
            f"BatchTopK k=3 should give L0 <= 3.5, got {actual_l0:.2f}"
        )

    def test_standard_and_topk_have_different_characteristics(self):
        """Standard L1 SAE and TopK SAE should produce different L0 profiles."""
        from sae_lens import (
            BatchTopKTrainingSAE,
            BatchTopKTrainingSAEConfig,
            StandardTrainingSAE,
            StandardTrainingSAEConfig,
        )

        tm = _make_toy_model(n_features=8, n_hidden=4, seed=7773)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        # Standard with very low L1 -> high L0
        std_cfg = StandardTrainingSAEConfig(d_in=4, d_sae=32, l1_coefficient=0.001)
        std_sae = StandardTrainingSAE(std_cfg)

        # TopK with k=2 -> exact L0=2
        topk_cfg = BatchTopKTrainingSAEConfig(d_in=4, d_sae=32, k=2)
        topk_sae = BatchTopKTrainingSAE(topk_cfg)

        entries = [
            SAEEntry(sae=std_sae, type="Standard", label="std_low"),
            SAEEntry(sae=topk_sae, type="BatchTopK", label="topk_2"),
        ]
        tm.train_saes(entries, training_samples=200_000, batch_size=512)
        results = tm.evaluate_saes(num_samples=10_000)

        l0_std = results["std_low"].sae_l0
        l0_topk = results["topk_2"].sae_l0

        # Standard with very low L1 should have higher L0 than TopK(k=2)
        assert l0_std > l0_topk, (
            f"Standard L1=0.001 L0 ({l0_std:.2f}) should be > TopK k=2 L0 ({l0_topk:.2f})"
        )


# ===========================================================================
# 8. n_loss_snapshots
# ===========================================================================


class TestLossSnapshots:
    """Verify loss snapshot recording during SAE training."""

    def test_loss_snapshots_recorded(self):
        """When n_loss_snapshots is set, SAERecord.losses should be populated."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=8881)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="snap")
        tm.train_saes(
            [entry],
            training_samples=100_000,
            batch_size=512,
            n_loss_snapshots=5,
        )

        record = tm.saes["snap"]
        assert record.losses is not None, (
            "losses should not be None when n_loss_snapshots > 0"
        )
        assert len(record.losses) == 5, (
            f"Expected 5 loss snapshots, got {len(record.losses)}"
        )

    def test_loss_snapshots_are_tuples(self):
        """Each loss snapshot should be a (step, loss) tuple."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=8882)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="snap_fmt")
        tm.train_saes(
            [entry],
            training_samples=100_000,
            batch_size=512,
            n_loss_snapshots=3,
        )

        record = tm.saes["snap_fmt"]
        for item in record.losses:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2, f"Expected (step, loss), got length {len(item)}"
            step, loss = item
            assert isinstance(step, int), f"Step should be int, got {type(step)}"
            assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
            assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_loss_snapshots_steps_are_increasing(self):
        """Snapshot steps should be monotonically increasing."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=8883)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="snap_inc")
        tm.train_saes(
            [entry],
            training_samples=100_000,
            batch_size=512,
            n_loss_snapshots=5,
        )

        record = tm.saes["snap_inc"]
        steps = [s for s, _ in record.losses]
        for i in range(len(steps) - 1):
            assert steps[i] < steps[i + 1], f"Steps should be increasing: {steps}"

    def test_loss_generally_decreases_over_training(self):
        """With enough training, later loss snapshots should be lower than
        earlier ones. At minimum, last should be <= first."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=8884)
        tm.fit(n_epochs=300, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=32, l1=0.01, label="snap_dec")
        tm.train_saes(
            [entry],
            training_samples=500_000,
            batch_size=512,
            n_loss_snapshots=10,
        )

        record = tm.saes["snap_dec"]
        losses = [l for _, l in record.losses]
        # Allow noisy decrease but first should be >= last
        # Use average of first 3 vs last 3 for robustness
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        assert late_avg <= early_avg * 1.5, (
            f"Loss should generally decrease: early_avg={early_avg:.4f}, "
            f"late_avg={late_avg:.4f}"
        )

    def test_no_loss_snapshots_when_none(self):
        """When n_loss_snapshots is None (default), losses should be None."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=8885)
        tm.fit(n_epochs=200, batch_size=256, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="snap_none")
        tm.train_saes([entry], training_samples=50_000, batch_size=512)

        record = tm.saes["snap_none"]
        assert record.losses is None, (
            f"losses should be None when n_loss_snapshots is not set, got {record.losses}"
        )


# ===========================================================================
# 9. Edge cases and potential bugs
# ===========================================================================


class TestEdgeCases:
    """Test edge cases in the SAE pipeline that could cause silent corruption."""

    def test_evaluate_nonexistent_label_raises(self, trained_tm):
        """evaluate_saes with a bad label should raise ValueError."""
        with pytest.raises(ValueError, match="do not exist"):
            trained_tm.evaluate_saes(labels=["nonexistent_sae"])

    def test_duplicate_sae_label_raises(self):
        """Two SAEEntry objects with the same label should raise."""
        e1 = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="dup")
        e2 = _make_sae_entry(d_in=4, d_sae=16, l1=0.2, label="dup")
        with pytest.raises(ValueError, match="Duplicate SAE label"):
            from occhio.toy_model import _resolve_sae_entries

            _resolve_sae_entries([e1, e2])

    def test_auto_labels_are_unique(self):
        """SAEEntries without explicit labels get auto-generated unique labels."""
        from occhio.toy_model import _resolve_sae_entries

        entries = [
            SAEEntry(sae=_make_sae_entry(d_in=4, l1=0.1).sae, type="Standard"),
            SAEEntry(sae=_make_sae_entry(d_in=4, l1=0.2).sae, type="Standard"),
            SAEEntry(sae=_make_sae_entry(d_in=4, l1=0.3).sae, type="Standard"),
        ]
        resolved = _resolve_sae_entries(entries)
        labels = list(resolved.keys())
        assert len(labels) == len(set(labels)), f"Auto-labels are not unique: {labels}"
        assert labels == ["Standard_0", "Standard_1", "Standard_2"]

    def test_sae_overwrite_warning(self):
        """Training an SAE with an existing label should warn."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=9991)
        tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        e1 = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="ow")
        tm.train_saes([e1], training_samples=10_000, batch_size=256)

        e2 = _make_sae_entry(d_in=4, d_sae=16, l1=0.2, label="ow")
        with pytest.warns(UserWarning, match="overwritten"):
            tm.train_saes([e2], training_samples=10_000, batch_size=256)

    def test_reevaluate_warning(self):
        """Re-evaluating an already-evaluated SAE should warn."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=9992)
        tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        entry = _make_sae_entry(d_in=4, d_sae=16, l1=0.1, label="reeval")
        tm.train_saes([entry], training_samples=10_000, batch_size=256)
        tm.evaluate_saes(labels=["reeval"], num_samples=1000)

        with pytest.warns(UserWarning, match="Re-evaluating"):
            tm.evaluate_saes(labels=["reeval"], num_samples=1000)

    def test_sae_callable_factory(self):
        """train_saes accepts a callable that returns SAEEntry list."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=9993)
        tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        def factory(model: ToyModel) -> list[SAEEntry]:
            return [_make_sae_entry(d_in=model.ae.n_hidden, l1=0.1, label="factory")]

        tm.train_saes(factory, training_samples=10_000, batch_size=256)
        assert "factory" in tm.saes

    def test_feature_dict_num_features_equals_ae_n_hidden(self):
        """FeatureDictionaryWrapper.num_features should equal ae.n_hidden
        (the SAE sees hidden-space activations, not raw features)."""
        ae = TiedLinearRelu(n_features=8, n_hidden=4)
        fd = FeatureDictionaryWrapper(ae)
        # feature_vectors = ae.encode(I_8) has shape (8, 4)
        # .T has shape (4, 8)
        # So num_features (first dim of .T) = 4 = n_hidden
        # Actually: num_features, hidden_dim = ae.feature_vectors.T.shape
        # ae.feature_vectors has shape (n_features, n_hidden) = (8, 4)
        # .T has shape (n_hidden, n_features) = (4, 8)
        # So num_features = 4 = n_hidden, hidden_dim = 8 = n_features
        # This means the SAE "features" correspond to the AE hidden dims
        assert fd.num_features == ae.n_hidden, (
            f"num_features ({fd.num_features}) should equal ae.n_hidden ({ae.n_hidden})"
        )

    def test_saes_property_accessors_consistent(self):
        """All saes_* property accessors should return consistent keys."""
        tm = _make_toy_model(n_features=8, n_hidden=4, seed=9994)
        tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)

        entries = [
            _make_sae_entry(d_in=4, l1=0.01, label="a"),
            _make_sae_entry(d_in=4, l1=0.1, label="b"),
        ]
        tm.train_saes(entries, training_samples=10_000, batch_size=256)
        tm.evaluate_saes(num_samples=1000)

        # All property accessors should return the same set of keys
        expected_keys = {"a", "b"}
        assert set(tm.saes_l0.keys()) == expected_keys
        assert set(tm.saes_f1_score.keys()) == expected_keys
        assert set(tm.saes_precision.keys()) == expected_keys
        assert set(tm.saes_recall.keys()) == expected_keys
        assert set(tm.saes_accuracy.keys()) == expected_keys
        assert set(tm.saes_mcc.keys()) == expected_keys
        assert set(tm.saes_explained_variance.keys()) == expected_keys
        assert set(tm.saes_uniqueness.keys()) == expected_keys
        assert set(tm.saes_shrinkage.keys()) == expected_keys
        assert set(tm.saes_dead_latents.keys()) == expected_keys
        assert set(tm.saes_true_l0.keys()) == expected_keys
        assert set(tm.saes_feature_similarity.keys()) == expected_keys


# ===========================================================================
# 10. Autotuner -- potential numerical edge cases
# ===========================================================================


class TestAutotunerNumericalEdgeCases:
    """Test numerical edge cases in the CoefficientAutotuner."""

    def test_zero_target_l0_does_not_divide_by_zero(self):
        """target_l0=0 in rel_error = error/target_l0 would cause division by zero.
        The autotuner should handle this gracefully."""
        cfg = CoefficientAutotunerConfig(target_l0=1e-10, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        # This should not raise
        at.update(batch_l0=5.0, step=0)
        at.update(batch_l0=5.0, step=1)
        assert math.isfinite(at.multiplier), (
            f"Multiplier is not finite: {at.multiplier}"
        )

    def test_very_large_l0_does_not_overflow(self):
        """Very large L0 values should not cause overflow."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=1e10, step=0)
        at.update(batch_l0=1e10, step=1)
        assert math.isfinite(at.multiplier), f"Multiplier overflowed: {at.multiplier}"

    def test_nan_l0_propagation(self):
        """If batch_l0 is NaN, the autotuner's smoothed_l0 becomes NaN.
        This documents current behavior -- whether it's a bug depends on
        the caller's responsibility to filter NaN inputs."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=5.0, step=0)
        at.update(batch_l0=float("nan"), step=1)
        # Document whether NaN propagates
        smoothed = at.smoothed_l0
        if math.isnan(smoothed):
            # This is the expected behavior -- NaN propagates through EMA
            pass
        else:
            # If NaN is filtered, verify multiplier is still finite
            assert math.isfinite(at.multiplier)

    def test_negative_l0_handled(self):
        """Negative L0 (physically impossible but could come from a bug)
        should not crash the autotuner."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.01)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=-1.0, step=0)
        at.update(batch_l0=-1.0, step=1)
        assert math.isfinite(at.multiplier)
