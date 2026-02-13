"""The base class for distributions"""

from abc import ABC, abstractmethod
from math import prod
import torch
from torch import Tensor


class Distribution(ABC):
    """Base class for all distributions."""

    def __init__(
        self,
        n_features: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.n_features = n_features
        self.device = device
        self.generator = generator

    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        """Returns (batch_size, n_features)"""

    def _rand(self, *shape) -> Tensor:
        """Random uniform generator respecting the self.generator"""
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def _randn(self, *shape) -> Tensor:
        """Random standard normal generator respecting the self.generator"""
        return torch.randn(*shape, device=self.device, generator=self.generator)

    def _rand_On(self, num_feat) -> Tensor:
        """Random O(n) generator respecting self.generator"""
        mat = self._randn(num_feat, num_feat)
        q, r = torch.linalg.qr(mat)
        return q * torch.sign(torch.diag(r))

    def _randint(
        self, low: int, high: int, shape: tuple[int, ...], p: Tensor | None = None
    ) -> Tensor:
        """Random generator respecting the self.generator"""
        if p is None:
            return torch.randint(
                low=low,
                high=high,
                size=shape,
                device=self.device,
                generator=self.generator,
            )
        else:
            return (
                low
                + torch.multinomial(
                    p[low:high],
                    prod(shape),
                    replacement=True,
                    generator=self.generator,
                )
            ).reshape(shape)

    def _broadcast(self, x: float | list[float] | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(self.device)
        if isinstance(x, (int, float)):
            return torch.full((self.n_features,), x, device=self.device)
        return torch.as_tensor(x, device=self.device)

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self.n_features}, {self.device})"

    def __str__(self):
        return f"{type(self).__name__}({self.n_features}, {self.device})"
