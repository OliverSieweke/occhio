"""The base class for distributions"""

from abc import ABC, abstractmethod
from functools import cached_property
from hashlib import sha256
from math import prod
from warnings import warn
from typing import Literal

import torch
from torch import Tensor, hash_tensor

from ..utils.device import _same_device


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
                    f"\nGenerator lives on {gen_device}, but device is {dev}. These must match."
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

    @property
    def _defines_generators(self) -> bool:
        return self.generator is not None

    def sync_generators(
        self, generators: torch.Generator | list[torch.Generator]
    ) -> None:
        """Set this distribution's generator state from a source generator.

        Used by ModelGrid to keep equivalent distributions synchronized
        after sample broadcasting.

        Args:
            generators: A single generator whose state is copied into
                ``self.generator``. A list is not accepted for base
                Distribution (only DistributionStack supports lists).
        """
        if isinstance(generators, list):
            if len(generators) != 1:
                raise ValueError(
                    f"Base Distribution expects a single generator, got {len(generators)}"
                )
            generators = generators[0]
        if self.generator is not None:
            self.generator.set_state(generators.get_state())

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
                low=int(low),
                high=int(high),
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
            if x.dim() == 0:
                return x.expand(self.n_features).clone().to(self.device)
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

    @property
    def _equivalence_hash(self) -> str:
        """This hash is used to determine if two distributions are equivalent at initialization.
        This is useful for caching samples. It's not recommended to modify this hash.
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


class DistributionStack(Distribution):
    """A composite distribution formed by stacking multiple independent distributions along the feature dimension.

    Concatenates samples from a list of `Distribution` instances, producing outputs
    of shape `(batch_size, sum(d.n_features for d in distributions))`.

    Args:
        distributions: List of `Distribution` instances to compose.
        sampling_mode: Controls how sub-distributions are activated:

            - ``"independent"``: Every sub-distribution is sampled for every
              row in the batch (default).
            - ``"single"``: Exactly one sub-distribution is sampled per row;
              the remaining feature positions are zero.
            - ``"sparse"``: Each sub-distribution fires independently with
              probability ``p_meta``; non-firing positions are zero.
        p_meta: Probability that each sub-distribution fires per sample.
            Required when ``sampling_mode="sparse"``.

    Example::

        d = DistributionStack([Uniform(3), Normal(2)])
        d.sample(64)  # shape: (64, 5)

    Note:
        Device and generator settings are inherited from each sub-distribution
        individually rather than from a single top-level config. Use the ``to()``
        method to move all sub-distributions to a common device. To ensure
        reproducible sampling, set the generator on each sub-distribution before
        passing them to DistributionStack.
    """

    def __init__(
        self,
        distributions: list[Distribution],
        sampling_mode: Literal["independent", "sparse", "single"] = "independent",
        p_meta: float | None = None,
        **kwargs,
    ):
        self._validate_stack(distributions, sampling_mode, p_meta, **kwargs)
        total_features = sum(dist.n_features for dist in distributions)
        self.distributions = distributions
        self.sampling_mode = sampling_mode
        self.p_meta = p_meta
        super().__init__(total_features, **kwargs)

    @property
    def _defines_generators(self) -> bool:
        return all(dist._defines_generators for dist in self.distributions)

    def sync_generators(
        self, generators: torch.Generator | list[torch.Generator]
    ) -> None:
        """Sync generators into each child distribution.

        Args:
            generators: If a single generator, every child distribution
                receives the same state. If a list, must have one generator
                per child distribution.
        """
        if isinstance(generators, list):
            if len(generators) != len(self.distributions):
                raise ValueError(
                    f"Expected {len(self.distributions)} generators, got {len(generators)}"
                )
            for child, gen in zip(self.distributions, generators):
                child.sync_generators(gen)
        else:
            for child in self.distributions:
                child.sync_generators(generators)

    def sample(self, batch_size):
        if self.sampling_mode == "independent":
            return torch.cat(
                [dist.sample(batch_size) for dist in self.distributions], dim=-1
            )

        result = torch.zeros(batch_size, self.n_features, device=self.device)
        offset = 0

        if self.sampling_mode == "single":
            indices = self._randint(0, len(self.distributions), (batch_size,))
            for i, dist in enumerate(self.distributions):
                mask = indices == i
                n_active = int(mask.sum().item())
                if n_active > 0:
                    result[mask, offset : offset + dist.n_features] = dist.sample(
                        n_active
                    )
                offset += dist.n_features

        elif self.sampling_mode == "sparse":
            assert self.p_meta is not None
            for dist in self.distributions:
                fire = self._rand(batch_size) < self.p_meta
                n_active = int(fire.sum().item())
                if n_active > 0:
                    result[fire, offset : offset + dist.n_features] = dist.sample(
                        n_active
                    )
                offset += dist.n_features

        return result

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        for dist in self.distributions:
            dist.to(device)
        return self

    def __repr__(self):
        dist_reprs = ", ".join(repr(d) for d in self.distributions)
        return f"DistributionStack([{dist_reprs}])"

    @property
    def _equivalence_hash(self) -> str:
        equivalence_dict = vars(self).copy()
        generator = equivalence_dict.pop("generator")
        equivalence_dict["distribution_type"] = type(self).__name__

        if generator:
            state = generator.get_state()
            state_hash = hash_tensor(state, mode=0)
            equivalence_dict["generator"] = state_hash

        equivalence_dict["distributions"] = [
            dist._equivalence_hash for dist in self.distributions
        ]

        for k, v in equivalence_dict.items():
            if isinstance(v, Tensor):
                equivalence_dict[k] = v.tolist()

        equivalence_dict = dict(sorted(equivalence_dict.items(), key=lambda x: x[0]))
        equivalence_string = str(equivalence_dict)
        return sha256(equivalence_string.encode("utf-8")).hexdigest()

    def _validate_stack(self, distributions, sampling_mode, p_meta, **kwargs) -> None:
        if not distributions:
            raise ValueError("\ndistributions list cannot be empty")

        if sampling_mode == "sparse" and p_meta is None:
            raise ValueError("\np_meta must be provided when sampling_mode='sparse'")

        reference_device = distributions[0].device
        for dist in distributions:
            if dist.device != reference_device:
                warn(
                    f"\nDetected device mismatch in DistributionStack."
                    f"reference device: {reference_device}"
                    f"received device: {dist.device} for {dist}"
                    f"All distributions in a DistributionStack should be located on the same device for reproducibility and efficiency."
                    f"Use the `.to(device)` to move all distributions to the same device.",
                    stacklevel=2,
                )

        if "generator" in kwargs and sampling_mode == "independent":
            warn(
                "\nDistributionStack does not use the generator parameter in 'independent' mode. "
                "Set the generator on each sub-distribution individually for reproducible sampling.",
                stacklevel=2,
            )
