from .base import Distribution
import torch
from torch import Tensor


class RelationalSimple(Distribution):
    def __init__(self, n_features: int, p_active: float = 0.1, **kwargs):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.on_mat = self._rand_On(self.n_features)

    def sample(self, batch_size: int) -> Tensor:
        # First
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        first = mask * values

        # second
        mask = self._rand(batch_size, self.n_features) < self.p_active
        values = self._rand(batch_size, self.n_features)
        second = mask * values

        return first + second @ self.on_mat
