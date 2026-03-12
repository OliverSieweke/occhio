"""Bump feature distribution with cyclic activation over discrete states."""

import torch
from torch import Tensor

from .base import Distribution


class Bump(Distribution):
    """A single irreducible multi-dimensional feature with cyclic structure.

    Each sample is either the zero vector (inactive) or a triangular bump
    encoding one of ``k`` cyclically-ordered discrete states, where ``k = n_features``.

    Args:
        n_features: Dimensionality and number of discrete states on the circle.
        p_active: Probability that the feature is active in a given sample.
        bump_width: Number of neighbors on each side that get nonzero activation.
            Must satisfy ``bump_width < n_features // 2``.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(
        self,
        n_features: int,
        p_active: float,
        bump_width: int = 2,
        amp_low: float = 1.0,
        noise_std: float = 0.0,
        **kwargs,
    ):
        if bump_width >= n_features // 2:
            raise ValueError(
                f"bump_width ({bump_width}) must be < n_features // 2 ({n_features // 2})"
            )
        super().__init__(n_features, **kwargs)
        self.p_active = p_active
        self.bump_width = bump_width
        self.amp_low = amp_low
        self.noise_std = noise_std
        self._bump_matrix = self._build_bump_matrix()

    def _build_bump_matrix(self) -> Tensor:
        """Precompute (k, k) matrix of bump vectors for each state."""
        k = self.n_features
        bw = self.bump_width
        # indices [0, 1, ..., k-1]
        i = torch.arange(k, device=self.device)
        # (k, k): row j, col i -> circular distance between j and i
        diff = (i.unsqueeze(0) - i.unsqueeze(1)).abs()  # |i - j|
        circ_dist = torch.min(diff, k - diff)  # min(|i-j|, k-|i-j|)
        bumps = (1.0 - circ_dist.float() / (bw + 1)).clamp(min=0.0)
        return bumps

    def sample(self, batch_size: int) -> Tensor:
        active = self._rand(batch_size) < self.p_active
        states = self._randint(0, self.n_features, (batch_size,))
        result = self._bump_matrix[states].clone()
        if self.amp_low < 1.0:
            amp = self.amp_low + (1.0 - self.amp_low) * self._rand(batch_size, 1)
            result = result * amp
        if self.noise_std > 0.0:
            on_spike = result > 0.0
            noise = self._randn(batch_size, self.n_features) * self.noise_std
            result[on_spike] += noise[on_spike]
        result[~active] = 0.0
        return result

    def to(self, device: torch.device | str):
        super().to(device)
        self._bump_matrix = self._bump_matrix.to(self.device)
        return self
