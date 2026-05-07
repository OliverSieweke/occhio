"""
Module for Relational Composition.
Based on https://arxiv.org/abs/2407.14662v1
"""

from .base import Distribution
import torch
from torch import Tensor


class RelationalSimple(Distribution):
    """Encodes two distinct matrix bindings (identity + O(n))

    Args:
        n_features: Number of features.
        p_active: Probability each feature is non-zero. Scalar or per-feature.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(self, n_features: int, p_active: float = 0.1, **kwargs):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.new_On_matrix()

    def sample(self, batch_size: int) -> Tensor:
        # first
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        first = mask * values

        # second
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        second = mask * values

        return first + second @ self.on_mat

    def new_On_matrix(self):
        self.on_mat = self._rand_On(self.n_features)


class MultiRelational(Distribution):
    """Encodes k distinct matrix bindings

    Args:
        n_features: Number of features.
        p_active: Probability each feature is non-zero. Scalar or per-feature.
        k: The number of representation to layer.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(self, n_features: int, p_active: float = 0.1, k: int = 2, **kwargs):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.k = k
        self.new_On_matricies()

    def sample(self, batch_size: int) -> Tensor:
        res = torch.zeros((batch_size, self.n_features), device=self.device)

        for mat in self.on_mats:
            mask = self._rand(batch_size, self.n_features) < self.p_active
            values = self._rand(batch_size, self.n_features)
            res += (mask * values) @ mat

        return res

    def new_On_matricies(self):
        self.on_mats: list[Tensor] = [
            self._rand_On(self.n_features) for _ in range(self.k)
        ]
