"""Tests for the sae_lens_adapter and utils modules.

Strategy:
- ActivationGeneratorWrapper: verify construction, sampling shape, forward alias,
  and that unsupported statistics properties raise NotImplementedError.
- FeatureDictionaryWrapper: verify construction, dimension attributes, feature_vectors
  parameter, and forward pass shape.
- CoefficientAutotunerConfig: verify default and custom construction.
- CoefficientAutotuner: verify construction, multiplier initialization, update dynamics
  (integral control toward target L0), reset, deadband, clamping, and convergence gain.
- StandardTrainingSAEConfigAutotuned: verify config construction and architecture name.
- StandardTrainingSAEAutotuned: verify construction with and without autotuning.
- _same_device: exhaustive comparison of device type/index combinations.
"""

import pytest
import torch
import torch.nn as nn

from occhio.distributions import SparseUniform
from occhio.autoencoders import TiedLinearRelu
from occhio.sae_lens_adapter.activation_generator import ActivationGeneratorWrapper
from occhio.sae_lens_adapter.feature_dictionary import FeatureDictionaryWrapper
from occhio.sae_lens_adapter.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)
from occhio.sae_lens_adapter.standard_sae_autotuned import (
    StandardTrainingSAEAutotuned,
    StandardTrainingSAEConfigAutotuned,
)
from occhio.utils.device import _same_device


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_generator():
    """Deterministic generator for reproducible distribution sampling."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture
def distribution(seeded_generator):
    """Small SparseUniform distribution for adapter tests."""
    return SparseUniform(n_features=5, p_active=0.5, generator=seeded_generator)


@pytest.fixture
def autoencoder():
    """Small TiedLinearRelu autoencoder for adapter tests."""
    return TiedLinearRelu(n_features=5, n_hidden=8)


@pytest.fixture
def activation_gen(distribution):
    """ActivationGeneratorWrapper wrapping a SparseUniform."""
    return ActivationGeneratorWrapper(distribution)


@pytest.fixture
def feature_dict(autoencoder):
    """FeatureDictionaryWrapper wrapping a TiedLinearRelu."""
    return FeatureDictionaryWrapper(autoencoder)


@pytest.fixture
def default_autotuner_config():
    """CoefficientAutotunerConfig with target_l0=5.0 and defaults."""
    return CoefficientAutotunerConfig(target_l0=5.0)


@pytest.fixture
def autotuner(default_autotuner_config):
    """CoefficientAutotuner with default config."""
    return CoefficientAutotuner(cfg=default_autotuner_config)


# ===========================================================================
# ActivationGeneratorWrapper
# ===========================================================================


class TestActivationGeneratorWrapperConstruction:
    def test_wraps_distribution(self, distribution):
        """Verify wrapper stores the distribution and is an nn.Module."""
        wrapper = ActivationGeneratorWrapper(distribution)
        assert wrapper._distribution is distribution
        assert isinstance(wrapper, nn.Module)

    def test_num_features_matches_distribution(self, distribution):
        """num_features must mirror distribution.n_features exactly."""
        wrapper = ActivationGeneratorWrapper(distribution)
        assert wrapper.num_features == distribution.n_features

    def test_num_features_various_sizes(self, seeded_generator):
        """Verify num_features is correct for several distribution sizes."""
        for n in [1, 10, 100]:
            dist = SparseUniform(n_features=n, p_active=0.3, generator=seeded_generator)
            wrapper = ActivationGeneratorWrapper(dist)
            assert wrapper.num_features == n


class TestActivationGeneratorWrapperSampling:
    def test_sample_returns_correct_shape(self, activation_gen):
        """sample() must return (batch_size, num_features)."""
        result = activation_gen.sample(32)
        assert result.shape == (32, activation_gen.num_features)

    def test_sample_batch_size_one(self, activation_gen):
        """Edge case: batch_size=1 should still work."""
        result = activation_gen.sample(1)
        assert result.shape == (1, activation_gen.num_features)

    def test_sample_large_batch(self, activation_gen):
        """Ensure no issues with a larger batch size."""
        result = activation_gen.sample(1024)
        assert result.shape == (1024, activation_gen.num_features)

    def test_sample_returns_tensor(self, activation_gen):
        """Return type must be a Tensor."""
        result = activation_gen.sample(16)
        assert isinstance(result, torch.Tensor)

    def test_sample_values_nonnegative(self, activation_gen):
        """SparseUniform produces values in [0, 1], so all nonneg."""
        result = activation_gen.sample(256)
        assert result.min() >= 0.0

    def test_sample_no_grad(self, activation_gen):
        """sample() is decorated with @torch.no_grad, so result should not require grad."""
        result = activation_gen.sample(16)
        assert not result.requires_grad


class TestActivationGeneratorWrapperForward:
    def test_forward_is_alias_for_sample(self, activation_gen):
        """forward() should produce the same output as sample() (same generator state)."""
        # Reset generator state, call forward; reset again, call sample
        gen = activation_gen._distribution.generator
        state = gen.get_state().clone()
        gen.set_state(state)
        result_forward = activation_gen.forward(64)
        gen.set_state(state)
        result_sample = activation_gen.sample(64)
        assert torch.equal(result_forward, result_sample)

    def test_forward_shape(self, activation_gen):
        """forward() must return the same shape as sample()."""
        result = activation_gen(16)
        assert result.shape == (16, activation_gen.num_features)


class TestActivationGeneratorWrapperNotImplemented:
    def test_firing_probabilities_raises(self, activation_gen):
        """firing_probabilities is unsupported and must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="firing_probabilities"):
            _ = activation_gen.firing_probabilities

    def test_mean_firing_magnitudes_raises(self, activation_gen):
        """mean_firing_magnitudes is unsupported and must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="mean_firing_magnitudes"):
            _ = activation_gen.mean_firing_magnitudes

    def test_std_firing_magnitudes_raises(self, activation_gen):
        """std_firing_magnitudes is unsupported and must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="std_firing_magnitudes"):
            _ = activation_gen.std_firing_magnitudes

    def test_correlation_matrix_raises(self, activation_gen):
        """correlation_matrix is unsupported and must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="correlation_matrix"):
            _ = activation_gen.correlation_matrix

    def test_error_messages_include_distribution_class_name(self, activation_gen):
        """Error messages should name the underlying distribution class."""
        with pytest.raises(NotImplementedError, match="SparseUniform"):
            _ = activation_gen.firing_probabilities


# ===========================================================================
# FeatureDictionaryWrapper
# ===========================================================================


class TestFeatureDictionaryWrapperConstruction:
    def test_wraps_autoencoder(self, autoencoder):
        """Verify wrapper stores a reference to the autoencoder."""
        wrapper = FeatureDictionaryWrapper(autoencoder)
        assert wrapper._auto_encoder is autoencoder

    def test_is_nn_module(self, feature_dict):
        """Wrapper must be an nn.Module."""
        assert isinstance(feature_dict, nn.Module)

    def test_num_features_correct(self, autoencoder, feature_dict):
        """num_features should match the transposed weight matrix rows."""
        expected_num_features, _ = autoencoder.feature_vectors.T.shape
        assert feature_dict.num_features == expected_num_features

    def test_hidden_dim_correct(self, autoencoder, feature_dict):
        """hidden_dim should match the transposed weight matrix columns."""
        _, expected_hidden_dim = autoencoder.feature_vectors.T.shape
        assert feature_dict.hidden_dim == expected_hidden_dim

    def test_feature_vectors_parameter_exists(self, feature_dict):
        """feature_vectors should be an nn.Parameter on the wrapper."""
        assert isinstance(feature_dict.feature_vectors, nn.Parameter)

    def test_feature_vectors_shape(self, autoencoder, feature_dict):
        """feature_vectors parameter should have shape matching the autoencoder's."""
        expected = autoencoder.feature_vectors.shape
        assert feature_dict.feature_vectors.shape == expected


class TestFeatureDictionaryWrapperForward:
    def test_forward_returns_correct_shape(self, feature_dict):
        """forward() takes (batch, hidden_dim) feature activations and encodes
        them via the autoencoder, returning (batch, num_features).

        The wrapper maps SAE Lens convention (features in, hidden out) through
        the autoencoder's encode path: input dim = autoencoder.n_features =
        wrapper.hidden_dim, output dim = autoencoder.n_hidden = wrapper.num_features.
        """
        batch = torch.randn(16, feature_dict.hidden_dim)
        result = feature_dict(batch)
        assert result.shape == (16, feature_dict.num_features)

    def test_forward_single_sample(self, feature_dict):
        """forward should handle a single-sample batch."""
        batch = torch.randn(1, feature_dict.hidden_dim)
        result = feature_dict(batch)
        assert result.shape == (1, feature_dict.num_features)

    def test_forward_uses_autoencoder_encode(self, autoencoder, feature_dict):
        """forward() delegates to the underlying autoencoder's encode method."""
        batch = torch.randn(8, feature_dict.hidden_dim)
        expected = autoencoder.encode(batch)
        actual = feature_dict(batch)
        assert torch.allclose(actual, expected)


# ===========================================================================
# CoefficientAutotunerConfig
# ===========================================================================


class TestCoefficientAutotunerConfig:
    def test_default_values(self):
        """Verify all default field values match the documented defaults."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0)
        assert cfg.target_l0 == 5.0
        assert cfg.start_step == 0
        assert cfg.smoothing_factor == 0.99
        assert cfg.rate_smoothing_factor == 0.95
        assert cfg.integral_gain == 5e-3
        assert cfg.min_multiplier == 1e-2
        assert cfg.max_multiplier == 100.0
        assert cfg.deadband == 0.0
        assert cfg.gain_scale == 10.0
        assert cfg.convergence_gain == 0.1

    def test_custom_values(self):
        """Verify custom values override defaults correctly."""
        cfg = CoefficientAutotunerConfig(
            target_l0=2.0,
            start_step=100,
            smoothing_factor=0.9,
            rate_smoothing_factor=0.8,
            integral_gain=1e-3,
            min_multiplier=0.5,
            max_multiplier=50.0,
            deadband=0.1,
            gain_scale=5.0,
            convergence_gain=0.05,
        )
        assert cfg.target_l0 == 2.0
        assert cfg.start_step == 100
        assert cfg.smoothing_factor == 0.9
        assert cfg.rate_smoothing_factor == 0.8
        assert cfg.integral_gain == 1e-3
        assert cfg.min_multiplier == 0.5
        assert cfg.max_multiplier == 50.0
        assert cfg.deadband == 0.1
        assert cfg.gain_scale == 5.0
        assert cfg.convergence_gain == 0.05


# ===========================================================================
# CoefficientAutotuner
# ===========================================================================


class TestCoefficientAutotunerConstruction:
    def test_initial_multiplier_is_one(self, autotuner):
        """Multiplier should start at 1.0 before any updates."""
        assert autotuner.multiplier == pytest.approx(1.0)

    def test_initial_smoothed_l0_is_zero(self, autotuner):
        """Smoothed L0 should be 0 before any observations."""
        assert autotuner.smoothed_l0 == pytest.approx(0.0)

    def test_initial_l0_rate_is_zero(self, autotuner):
        """L0 rate should be 0 before any observations."""
        assert autotuner.l0_rate == pytest.approx(0.0)

    def test_is_nn_module(self, autotuner):
        """Autotuner must be an nn.Module (for buffer management)."""
        assert isinstance(autotuner, nn.Module)

    def test_buffers_registered(self, autotuner):
        """Critical state should be registered as buffers for state_dict."""
        buffer_names = {name for name, _ in autotuner.named_buffers()}
        assert "_smoothed_l0" in buffer_names
        assert "_multiplier" in buffer_names
        assert "_initialized" in buffer_names
        assert "_l0_rate" in buffer_names
        assert "_prev_smoothed_l0" in buffer_names


class TestCoefficientAutotunerUpdate:
    def test_first_update_initializes_smoothed_l0(self, autotuner):
        """First call to update should set smoothed_l0 to the observed batch_l0."""
        autotuner.update(batch_l0=10.0, step=0)
        assert autotuner.smoothed_l0 == pytest.approx(10.0)

    def test_l0_above_target_increases_multiplier(self):
        """When L0 >> target, multiplier should increase to add more sparsity."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, integral_gain=0.1)
        at = CoefficientAutotuner(cfg)
        # Initialize
        at.update(batch_l0=20.0, step=0)
        # After update, L0 is well above target
        at.update(batch_l0=20.0, step=1)
        assert at.multiplier > 1.0

    def test_l0_below_target_decreases_multiplier(self):
        """When L0 << target, multiplier should decrease to reduce sparsity."""
        cfg = CoefficientAutotunerConfig(target_l0=10.0, integral_gain=0.1)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=1.0, step=0)
        at.update(batch_l0=1.0, step=1)
        assert at.multiplier < 1.0

    def test_l0_at_target_keeps_multiplier_near_one(self, autotuner):
        """When L0 matches target exactly, multiplier should stay near 1.0."""
        target = autotuner.cfg.target_l0
        autotuner.update(batch_l0=target, step=0)
        autotuner.update(batch_l0=target, step=1)
        # With zero error, tanh(0)=0, so adjustment=0
        assert autotuner.multiplier == pytest.approx(1.0)

    def test_start_step_delays_adjustment(self):
        """No adjustment should happen before start_step."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, start_step=100)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=20.0, step=0)
        at.update(batch_l0=20.0, step=50)
        assert at.multiplier == pytest.approx(1.0)

    def test_start_step_allows_adjustment_after(self):
        """Adjustment should begin once step >= start_step."""
        cfg = CoefficientAutotunerConfig(
            target_l0=5.0, start_step=10, integral_gain=0.1
        )
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=20.0, step=0)
        # Step 10 is at start_step, should allow adjustment
        at.update(batch_l0=20.0, step=10)
        assert at.multiplier > 1.0

    def test_accepts_tensor_batch_l0(self, autotuner):
        """update() should accept a Tensor as batch_l0 (auto .item())."""
        autotuner.update(batch_l0=torch.tensor(5.0), step=0)
        assert autotuner.smoothed_l0 == pytest.approx(5.0)

    def test_multiplier_clamped_to_max(self):
        """Multiplier must never exceed max_multiplier."""
        cfg = CoefficientAutotunerConfig(
            target_l0=1.0, integral_gain=1.0, max_multiplier=2.0
        )
        at = CoefficientAutotuner(cfg)
        for step in range(200):
            at.update(batch_l0=100.0, step=step)
        assert at.multiplier <= 2.0

    def test_multiplier_clamped_to_min(self):
        """Multiplier must never go below min_multiplier."""
        cfg = CoefficientAutotunerConfig(
            target_l0=100.0, integral_gain=1.0, min_multiplier=0.5
        )
        at = CoefficientAutotuner(cfg)
        for step in range(200):
            at.update(batch_l0=0.1, step=step)
        assert at.multiplier >= 0.5

    def test_deadband_suppresses_small_errors(self):
        """When |error| <= deadband, no adjustment should occur."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, deadband=1.0, integral_gain=0.1)
        at = CoefficientAutotuner(cfg)
        # L0 = 5.5 => error = 0.5 < deadband 1.0 => no adjustment
        at.update(batch_l0=5.5, step=0)
        at.update(batch_l0=5.5, step=1)
        assert at.multiplier == pytest.approx(1.0)

    def test_deadband_allows_large_errors(self):
        """When |error| > deadband, adjustment should happen."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, deadband=0.5, integral_gain=0.1)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=20.0, step=0)
        at.update(batch_l0=20.0, step=1)
        assert at.multiplier != pytest.approx(1.0)

    def test_ema_smoothing(self):
        """Smoothed L0 should be an EMA of observed values, not jump instantly."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0, smoothing_factor=0.9)
        at = CoefficientAutotuner(cfg)
        at.update(batch_l0=10.0, step=0)
        assert at.smoothed_l0 == pytest.approx(10.0)  # first observation = direct set
        at.update(batch_l0=0.0, step=1)
        # EMA: 0.9 * 10 + 0.1 * 0 = 9.0
        assert at.smoothed_l0 == pytest.approx(9.0)

    def test_convergence_gain_dampens_when_approaching(self):
        """When error is decreasing (moving toward target), gain should be reduced.

        Compare adjustment magnitude: same error but opposite rate direction
        should produce different adjustments due to convergence_gain damping.
        """
        cfg = CoefficientAutotunerConfig(
            target_l0=5.0,
            integral_gain=0.01,
            convergence_gain=0.01,
            smoothing_factor=0.0,  # no smoothing, instant tracking
            rate_smoothing_factor=0.0,
        )

        # Case 1: Error positive, rate negative (moving toward target) => damped
        at1 = CoefficientAutotuner(cfg)
        at1.update(batch_l0=20.0, step=0)  # init
        at1.update(batch_l0=15.0, step=1)  # rate < 0, error > 0 => converging
        mult_converging = at1.multiplier

        # Case 2: Error positive, rate positive (moving away) => full gain
        at2 = CoefficientAutotuner(cfg)
        at2.update(batch_l0=10.0, step=0)  # init
        at2.update(batch_l0=15.0, step=1)  # rate > 0, error > 0 => diverging
        mult_diverging = at2.multiplier

        # Both above target so both > 1, but diverging case should adjust more
        assert mult_diverging > mult_converging

    def test_update_returns_float(self, autotuner):
        """update() should return a Python float."""
        result = autotuner.update(batch_l0=5.0, step=0)
        assert isinstance(result, float)


class TestCoefficientAutotunerReset:
    def test_reset_restores_initial_state(self, autotuner):
        """After reset(), multiplier should be 1.0 and smoothed_l0 should be 0."""
        autotuner.update(batch_l0=20.0, step=0)
        autotuner.update(batch_l0=20.0, step=1)
        autotuner.reset()
        assert autotuner.multiplier == pytest.approx(1.0)
        assert autotuner.smoothed_l0 == pytest.approx(0.0)
        assert autotuner.l0_rate == pytest.approx(0.0)

    def test_reset_allows_reinit_on_next_update(self, autotuner):
        """After reset, next update should re-initialize smoothed_l0 directly."""
        autotuner.update(batch_l0=20.0, step=0)
        autotuner.reset()
        autotuner.update(batch_l0=7.0, step=0)
        assert autotuner.smoothed_l0 == pytest.approx(7.0)


class TestCoefficientAutotunerStateDict:
    def test_state_dict_contains_buffers(self, autotuner):
        """All registered buffers should appear in the state dict."""
        sd = autotuner.state_dict()
        assert "_smoothed_l0" in sd
        assert "_multiplier" in sd
        assert "_initialized" in sd
        assert "_l0_rate" in sd
        assert "_prev_smoothed_l0" in sd

    def test_load_state_dict_restores_state(self):
        """Saving and loading state_dict should preserve autotuner state."""
        cfg = CoefficientAutotunerConfig(target_l0=5.0)
        at1 = CoefficientAutotuner(cfg)
        for step in range(10):
            at1.update(batch_l0=20.0, step=step)

        at2 = CoefficientAutotuner(cfg)
        at2.load_state_dict(at1.state_dict())
        assert at2.multiplier == pytest.approx(at1.multiplier)
        assert at2.smoothed_l0 == pytest.approx(at1.smoothed_l0)
        assert at2.l0_rate == pytest.approx(at1.l0_rate)


# ===========================================================================
# StandardTrainingSAEConfigAutotuned
# ===========================================================================


class TestStandardTrainingSAEConfigAutotuned:
    def test_construction_defaults(self):
        """Config should be constructable with minimal args and have correct defaults."""
        cfg = StandardTrainingSAEConfigAutotuned(d_in=16, d_sae=32)
        assert cfg.d_in == 16
        assert cfg.d_sae == 32
        assert cfg.autotune_target_l0 is None
        assert cfg.autotune_start_step == 0
        assert cfg.autotune_smoothing_factor == 0.99

    def test_inherits_standard_config(self):
        """Config must be a subclass of StandardTrainingSAEConfig."""
        from sae_lens import StandardTrainingSAEConfig

        assert issubclass(StandardTrainingSAEConfigAutotuned, StandardTrainingSAEConfig)

    def test_architecture_name(self):
        """Architecture string should be 'xstandard'."""
        assert StandardTrainingSAEConfigAutotuned.architecture() == "xstandard"

    def test_target_l0_parameter(self):
        """autotune_target_l0 should be settable to a float."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=16, d_sae=32, autotune_target_l0=3.0
        )
        assert cfg.autotune_target_l0 == 3.0

    def test_get_autotuner_config_none_when_disabled(self):
        """get_autotuner_config() returns None when autotune_target_l0 is None."""
        cfg = StandardTrainingSAEConfigAutotuned(d_in=16, d_sae=32)
        assert cfg.get_autotuner_config() is None

    def test_get_autotuner_config_returns_config_when_enabled(self):
        """get_autotuner_config() returns a CoefficientAutotunerConfig when target is set."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=16,
            d_sae=32,
            autotune_target_l0=3.0,
            autotune_start_step=50,
            autotune_integral_gain=1e-3,
        )
        ac = cfg.get_autotuner_config()
        assert isinstance(ac, CoefficientAutotunerConfig)
        assert ac.target_l0 == 3.0
        assert ac.start_step == 50
        assert ac.integral_gain == 1e-3

    def test_get_autotuner_config_passes_all_fields(self):
        """All autotune_* fields should be forwarded to CoefficientAutotunerConfig."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=8,
            d_sae=16,
            autotune_target_l0=2.0,
            autotune_start_step=10,
            autotune_smoothing_factor=0.95,
            autotune_rate_smoothing_factor=0.9,
            autotune_integral_gain=5e-4,
            autotune_min_multiplier=0.1,
            autotune_max_multiplier=20.0,
            autotune_deadband=0.5,
            autotune_gain_scale=8.0,
            autotune_convergence_gain=0.02,
        )
        ac = cfg.get_autotuner_config()
        assert ac.target_l0 == 2.0
        assert ac.start_step == 10
        assert ac.smoothing_factor == 0.95
        assert ac.rate_smoothing_factor == 0.9
        assert ac.integral_gain == 5e-4
        assert ac.min_multiplier == 0.1
        assert ac.max_multiplier == 20.0
        assert ac.deadband == 0.5
        assert ac.gain_scale == 8.0
        assert ac.convergence_gain == 0.02

    def test_get_inference_config_class(self):
        """Inference config class should be StandardSAEConfig."""
        from sae_lens import StandardSAEConfig

        cfg = StandardTrainingSAEConfigAutotuned(d_in=8, d_sae=16)
        assert cfg.get_inference_config_class() is StandardSAEConfig


# ===========================================================================
# StandardTrainingSAEAutotuned
# ===========================================================================


class TestStandardTrainingSAEAutotuned:
    def test_inherits_from_standard_training_sae(self):
        """Must be a subclass of StandardTrainingSAE."""
        from sae_lens import StandardTrainingSAE

        assert issubclass(StandardTrainingSAEAutotuned, StandardTrainingSAE)

    def test_construction_without_autotuning(self):
        """When autotune_target_l0 is None, coefficient_autotuner should be None."""
        cfg = StandardTrainingSAEConfigAutotuned(d_in=8, d_sae=16)
        sae = StandardTrainingSAEAutotuned(cfg)
        assert sae.coefficient_autotuner is None

    def test_construction_with_autotuning(self):
        """When autotune_target_l0 is set, coefficient_autotuner should be created."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=8, d_sae=16, autotune_target_l0=3.0
        )
        sae = StandardTrainingSAEAutotuned(cfg)
        assert isinstance(sae.coefficient_autotuner, CoefficientAutotuner)
        assert sae.coefficient_autotuner.cfg.target_l0 == 3.0

    def test_is_nn_module(self):
        """Must be an nn.Module."""
        cfg = StandardTrainingSAEConfigAutotuned(d_in=8, d_sae=16)
        sae = StandardTrainingSAEAutotuned(cfg)
        assert isinstance(sae, nn.Module)

    def test_process_state_dict_removes_autotuner_buffers(self):
        """Inference state dict should not contain coefficient_autotuner.* keys."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=8, d_sae=16, autotune_target_l0=3.0
        )
        sae = StandardTrainingSAEAutotuned(cfg)
        sd = dict(sae.state_dict())
        # Confirm autotuner keys exist in training state_dict
        autotuner_keys = [k for k in sd if k.startswith("coefficient_autotuner.")]
        assert len(autotuner_keys) > 0

        # Process for inference
        sae.process_state_dict_for_saving_inference(sd)
        remaining = [k for k in sd if k.startswith("coefficient_autotuner.")]
        assert len(remaining) == 0

    def test_cfg_stored_correctly(self):
        """The cfg attribute should be the autotuned config type."""
        cfg = StandardTrainingSAEConfigAutotuned(
            d_in=8, d_sae=16, autotune_target_l0=5.0
        )
        sae = StandardTrainingSAEAutotuned(cfg)
        assert sae.cfg is cfg


# ===========================================================================
# _same_device
# ===========================================================================


class TestSameDevice:
    def test_cpu_cpu(self):
        """Two bare CPU devices should be considered the same."""
        assert _same_device(torch.device("cpu"), torch.device("cpu")) is True

    def test_cpu_with_index_zero(self):
        """cpu (index None) and cpu:0 (index 0) should be treated as same.

        The function maps None -> 0, so cpu == cpu:0.
        """
        assert _same_device(torch.device("cpu"), torch.device("cpu", 0)) is True

    def test_cpu_vs_cuda(self):
        """Different device types should never be considered the same."""
        assert _same_device(torch.device("cpu"), torch.device("cuda", 0)) is False

    def test_same_explicit_index(self):
        """Same type with same explicit index should match."""
        assert _same_device(torch.device("cuda", 0), torch.device("cuda", 0)) is True

    def test_different_indices(self):
        """Same type but different indices should not match."""
        assert _same_device(torch.device("cuda", 0), torch.device("cuda", 1)) is False

    def test_none_index_vs_zero_symmetric(self):
        """Symmetry: cpu:0 vs cpu (reversed order) should also match."""
        assert _same_device(torch.device("cpu", 0), torch.device("cpu")) is True

    def test_mps_same(self):
        """MPS devices without index should be same."""
        assert _same_device(torch.device("mps"), torch.device("mps")) is True

    def test_mps_vs_cpu(self):
        """MPS and CPU are different device types."""
        assert _same_device(torch.device("mps"), torch.device("cpu")) is False

    def test_mps_index_none_vs_zero(self):
        """mps (index None) and mps:0 (index 0) should be treated as same."""
        assert _same_device(torch.device("mps"), torch.device("mps", 0)) is True

    @pytest.mark.parametrize(
        "dev_a, dev_b, expected",
        [
            (torch.device("cpu"), torch.device("cpu"), True),
            (torch.device("cpu"), torch.device("cpu", 0), True),
            (torch.device("cpu", 0), torch.device("cpu"), True),
            (torch.device("cpu"), torch.device("cuda", 0), False),
            (torch.device("cuda", 0), torch.device("cuda", 0), True),
            (torch.device("cuda", 0), torch.device("cuda", 1), False),
            (torch.device("mps"), torch.device("mps", 0), True),
            (torch.device("mps"), torch.device("cpu"), False),
        ],
        ids=[
            "cpu-cpu",
            "cpu-cpu0",
            "cpu0-cpu",
            "cpu-cuda0",
            "cuda0-cuda0",
            "cuda0-cuda1",
            "mps-mps0",
            "mps-cpu",
        ],
    )
    def test_parametrized_device_pairs(self, dev_a, dev_b, expected):
        """Parametrized sweep over representative device pair combinations."""
        assert _same_device(dev_a, dev_b) is expected
