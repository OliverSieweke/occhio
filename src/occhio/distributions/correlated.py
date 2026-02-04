"""Simple correlation structures."""

import torch
from torch import Tensor
from .base import Distribution


class CorrelatedPairs(Distribution):
    """
    Features come in pars; if features 2i is active,
    feature 2i+1 is active with probability `correlation`.
    Correlation is not strictly speaking the correlation.

    A shallow form of hierarchy.
    """

    def __init__(
        self,
        n_features: int,
        sparsity: float,
        correlation: float = 0.5,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even `n_features` for pairs."
        super().__init__(n_features, **kwargs)
        self.correlation = correlation
        self.sparsity = sparsity

    def sample(self, batch_size: int) -> Tensor:
        n_pairs = self.n_features // 2
        primary_mask = self._rand(batch_size, n_pairs) < self.sparsity
        secondary_mask = primary_mask & (
            self._rand(batch_size, n_pairs) < self.correlation
        )

        mask = torch.empty(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        mask[:, 0::2] = primary_mask
        mask[:, 1::2] = secondary_mask

        values = self._rand(batch_size, self.n_features)
        return mask * values


class AnticorrelatedPairs(Distribution):
    """
    Features come in mutually exclusive pairs.
    At most one of (2i, 2i+1) is active per sample.
    """

    def __init__(
        self,
        n_features: int,
        sparsity: float,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even n_features for pairs"
        super().__init__(n_features, **kwargs)
        self.sparsity = sparsity

    def sample(self, batch_size: int) -> Tensor:
        n_pairs: int = self.n_features // 2

        pair_active = self._rand(batch_size, n_pairs) < self.sparsity

        which_one = self._randint(0, 2, (batch_size, n_pairs))

        mask = torch.zeros(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        mask[:, 0::2] = pair_active & (which_one == 0)
        mask[:, 1::2] = pair_active & (which_one == 1)

        values = self._rand(batch_size, self.n_features)
        return mask * values
