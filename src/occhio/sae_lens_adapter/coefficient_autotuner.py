"""Adaptive L0 sparsity coefficient autotuner for SAE training.

Copied from https://github.com/decoderesearch/synth-sae-bench-experiments/blob/main/saes/components/coefficient_autotuner.py
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class CoefficientAutotunerConfig:
    """Configuration for the CoefficientAutotuner.

    Uses a rate-dampened integral controller:
    1. Tracks current position error (smoothed_l0 - target_l0)
    2. Tracks L0 rate of change
    3. Reduces gain when moving toward target to prevent overshoot

    This is simpler than predictive control and doesn't require tuning a
    prediction horizon.
    """

    target_l0: float
    start_step: int = 0
    # EMA smoothing factor for L0 measurements. Higher = smoother but slower.
    # Effective window ≈ 1/(1-α) steps.
    smoothing_factor: float = 0.99
    # EMA smoothing factor for L0 rate (derivative) estimation.
    rate_smoothing_factor: float = 0.95
    # Integral gain (Ki). Controls speed of multiplier adjustment.
    integral_gain: float = 3e-4
    min_multiplier: float = 1e-2
    max_multiplier: float = 100.0
    # Deadband: no adjustment if |error| <= deadband
    deadband: float = 0.0
    # Nonlinear gain parameter. Adjustment = Ki * tanh(|rel_error| * gain_scale)
    gain_scale: float = 10.0
    # Gain multiplier when error is decreasing (moving toward target).
    # Lower = more damping. 0.01 means 99% reduction when converging.
    convergence_gain: float = 0.01


class CoefficientAutotuner:
    """Autotuner that outputs a multiplier to achieve a target L0 sparsity.

    Uses an integral controller with EMA smoothing and gain scheduling. The
    controller reduces its gain when the error is decreasing (i.e., when the
    system is converging toward the target) to prevent overshoot. A tanh
    nonlinearity bounds the adjustment magnitude and provides smooth behavior
    near the setpoint.

    All internal state is stored as plain Python floats to avoid unnecessary
    GPU synchronization when called from a CUDA training loop.

    Algorithm:
        1. Smooth L0 measurements using exponential moving average (EMA)
        2. Estimate L0 rate of change (smoothed derivative)
        3. Compute position error: smoothed_l0 - target_l0
        4. Apply gain scheduling: reduce gain when moving toward target
        5. Compute bounded adjustment: Ki * gain * tanh(|rel_error| * scale)
        6. Multiplicative update: multiplier *= (1 ± adjustment)

    Example usage:
        autotuner = CoefficientAutotuner(cfg)
        for step, batch in enumerate(dataloader):
            feature_acts = sae.encode(batch)
            batch_l0 = (feature_acts != 0).float().sum(dim=-1).mean()
            multiplier = autotuner.update(batch_l0, step)
            effective_coefficient = base_coefficient * multiplier
    """

    def __init__(
        self,
        cfg: CoefficientAutotunerConfig,
        device: torch.device | str = "cpu",
    ):
        # device is accepted for API compatibility but unused — all state is
        # plain Python floats to avoid GPU synchronization overhead.
        self.cfg = cfg
        self._smoothed_l0: float = 0.0
        self._multiplier: float = 1.0
        self._initialized: bool = False
        self._l0_rate: float = 0.0
        self._prev_smoothed_l0: float = 0.0

    @property
    def multiplier(self) -> float:
        """Current multiplier value."""
        return self._multiplier

    @property
    def smoothed_l0(self) -> float:
        """Current smoothed L0 estimate."""
        return self._smoothed_l0

    @property
    def l0_rate(self) -> float:
        """Current smoothed rate of L0 change per step."""
        return self._l0_rate

    def update(self, batch_l0: float | torch.Tensor, step: int) -> float:
        """Update the autotuner state and return the new multiplier.

        Args:
            batch_l0: L0 sparsity from the current batch.
            step: Current training step.

        Returns:
            The updated multiplier value.
        """
        # Single GPU→CPU sync: extract the scalar measurement
        if isinstance(batch_l0, torch.Tensor):
            batch_l0 = batch_l0.item()

        # Update smoothed L0 estimate (EMA)
        if not self._initialized:
            self._smoothed_l0 = batch_l0
            self._prev_smoothed_l0 = batch_l0
            self._l0_rate = 0.0
            self._initialized = True
        else:
            self._prev_smoothed_l0 = self._smoothed_l0
            self._smoothed_l0 = (
                self.cfg.smoothing_factor * self._smoothed_l0
                + (1.0 - self.cfg.smoothing_factor) * batch_l0
            )
            instant_rate = self._smoothed_l0 - self._prev_smoothed_l0
            self._l0_rate = (
                self.cfg.rate_smoothing_factor * self._l0_rate
                + (1.0 - self.cfg.rate_smoothing_factor) * instant_rate
            )

        # No adjustment before start_step
        if step < self.cfg.start_step:
            return self._multiplier

        # Position error: positive means L0 is above target
        error = self._smoothed_l0 - self.cfg.target_l0

        # Apply deadband - no adjustment if within tolerance
        if abs(error) <= self.cfg.deadband:
            return self._multiplier

        # Determine if we're moving toward or away from target
        # Moving toward target: error and rate have opposite signs
        moving_toward_target = error * self._l0_rate < 0

        # Reduce gain when moving toward target to prevent overshoot
        gain = self.cfg.convergence_gain if moving_toward_target else 1.0

        # Nonlinear gain using tanh (bounded adjustment)
        rel_error = error / self.cfg.target_l0
        adjustment = (
            self.cfg.integral_gain
            * gain
            * math.tanh(abs(rel_error) * self.cfg.gain_scale)
        )

        # Multiplicative update
        if error > 0:
            # L0 too high, increase multiplier to make it more sparse
            new_multiplier = self._multiplier * (1.0 + adjustment)
        else:
            # L0 too low, decrease multiplier to make it less sparse
            new_multiplier = self._multiplier * (1.0 - adjustment)

        # Clamp to bounds
        self._multiplier = max(
            self.cfg.min_multiplier, min(self.cfg.max_multiplier, new_multiplier)
        )

        return self._multiplier

    def reset(self) -> None:
        """Reset smoothed L0 state and multiplier back to 1.0."""
        self._smoothed_l0 = 0.0
        self._multiplier = 1.0
        self._initialized = False
        self._l0_rate = 0.0
        self._prev_smoothed_l0 = 0.0
