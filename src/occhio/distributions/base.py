"""The base class for distributions"""

from abc import ABC, abstractmethod
from math import prod
import torch
from torch import Tensor, hash_tensor
from hashlib import sha256
from functools import cached_property

class Distribution(ABC):
    """Abstract base class for sampling distributions.

    Provides a common interface for generating batched samples of shape
    ``(batch_size, n_features)``, along with utility methods for reproducible
    random number generation via an optional ``torch.Generator``.

    Subclasses must implement :meth:`sample`. Helper methods :meth:`_rand`,
    :meth:`_randn`, :meth:`_randint`, and :meth:`_rand_On` all respect the
    stored generator for reproducibility.

    Args:
        n_features: Dimensionality of the sample space.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    def __init__(
        self,
        n_features: int,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ):
        self.n_features = n_features
        if device is not None and generator is not None:
            gen_device = torch.device(generator.device)
            dev = torch.device(device)
            if not _same_device(gen_device, dev):
                raise ValueError(
                    f"Generator lives on {gen_device}, but device is {dev}. "
                    f"These must match."
                )
        if device is not None:
            self._init_device = torch.device(device)
        elif generator is not None:
            self._init_device = torch.device(generator.device)
        else:
            self._init_device = None
        self.device = self._init_device
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
        for attr_name in vars(self):
            val = getattr(self, attr_name)
            if isinstance(val, Tensor):
                setattr(self, attr_name, val.to(self.device))
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, Tensor):
                        val[i] = item.to(self.device)
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self.n_features}, {self.device})"

    def __str__(self):
        return f"{type(self).__name__}({self.n_features}, {self.device})"

    @cached_property
    def _sampling_equivalence_hash(self) -> str:
        """
        We use this hash to determine if two distributions are equivalent at initialization. This is useful for caching samples. It's not reccomended to modify this hash.
        """
        equivalence_dict = vars(self).copy()
        generator = equivalence_dict.pop("generator")
        equivalence_dict["distribution_type"] = type(self).__name__

        if generator:
            state = generator.get_state()
            state_hash = hash_tensor(state, mode=0)
            equivalence_dict["generator"] = state_hash
        for k, v in equivalence_dict.items():
            if isinstance(v, Tensor):
                equivalence_dict[k] = v.tolist()
        equivalence_dict = dict(sorted(equivalence_dict.items(), key=lambda x: x[0]))
        equivalence_string = str(equivalence_dict)
        return sha256(equivalence_string.encode("utf-8")).hexdigest()
