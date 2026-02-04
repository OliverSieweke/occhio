"""Simple Sparse Feature Distributions."""

from .base import Distribution
from torch import Tensor
import torch


class SparseUniform(Distribution):
    def __init__(
        self, n_features: int, p_active: float | list[float] | Tensor, **kwargs
    ):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)

    def sample(self, batch_size: int) -> Tensor:
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        return mask * values


class SparseExponential(Distribution):
    def __init__(self, n_features: int, p_active: float, scale: float = 1.0, **kwargs):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.scale = scale

    def sample(self, batch_size: int) -> Tensor:
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = -(1.0 / self.scale) * torch.log(
            1.0 - self._rand(batch_size, self.n_features)
        )
        return mask * values
