"""
Simple Sparse Feature Distributions.
Inspired / taken from Toy Model of Superposition paper.
"""

from .base import Distribution
from torch import Tensor
import torch


class SparseUniform(Distribution):
    """Sparse uniform distribution with per-feature sparsity.

    Args:
        n_features: Number of features.
        p_active: Probability each feature is non-zero. Scalar or per-feature.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(
        self, n_features: int, p_active: float | list[float] | Tensor, **kwargs
    ):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)

    @property
    def expected_l0(self) -> float:
        """Expected L0 norm (number of active features) per sample."""
        return float(self.p_active.sum())

    def sample(self, batch_size: int) -> Tensor:
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        return mask * values


class SparseExponential(Distribution):
    """Sparse exponential distribution with per-feature sparsity and rate.

    Args:
        n_features: Number of features.
        p_active: Probability each feature is non-zero. Scalar or per-feature.
        scale: Rate parameter (mean = 1/scale). Scalar or per-feature.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor,
        scale: float | list[float] | Tensor = 1.0,
        **kwargs,
    ):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.scale = self._broadcast(scale)

    def sample(self, batch_size: int) -> Tensor:
        mask = self._rand(batch_size, self.n_features) < self.p_active
        u = self._rand(batch_size, self.n_features).clamp(max=1.0 - 1e-7)
        values = -(1.0 / self.scale) * torch.log(1.0 - u)
        return mask * values


class SingleUniform(Distribution):
    """Single-feature uniform distribution.

    Exactly one feature fires per sample, chosen uniformly at random.
    The active feature takes a value sampled from Uniform(0, 1).

    Args:
        n_features: Number of features.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(self, n_features: int, **kwargs):
        super().__init__(n_features, **kwargs)

    def sample(self, batch_size: int) -> Tensor:
        indices = self._randint(0, self.n_features, (batch_size,))
        values = self._rand(batch_size)
        result = torch.zeros(batch_size, self.n_features, device=self.device)
        result[torch.arange(batch_size, device=self.device), indices] = values
        return result
