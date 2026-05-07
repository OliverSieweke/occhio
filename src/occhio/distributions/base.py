"""The base class for distributions"""

import datetime
import json
from abc import ABC, abstractmethod
from functools import cached_property
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Literal
import numpy as np
from warnings import warn

import torch
from safetensors.torch import save_file
from torch import Tensor, hash_tensor

from ..utils.device import _same_device

# Sentinel for values that should be omitted from JSON (mirrors autoencoder._SKIP).
_SKIP = object()


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

    def clear_buffer(self) -> None:
        """Release any device-side sample buffer held by this distribution.

        No-op by default; subclasses with GPU buffers (e.g.
        :class:`HuggingFaceDistribution`) override this to free the memory.
        Call after training is done to reclaim GPU memory before the next run.
        """

    @property
    def _defines_generators(self) -> bool:
        return self.generator is not None

    def collect_generators(self) -> list[torch.Generator | None]:
        """Return this distribution's generators as a list.

        Returns:
            Single-element list containing ``self.generator`` (which may be ``None``).
        """
        return [self.generator]

    def sync_generators(
        self, generators: torch.Generator | None | list[torch.Generator | None]
    ) -> None:
        """Set this distribution's generator state from a source generator.

        Used by ModelGrid to keep equivalent distributions synchronized
        after sample broadcasting.

        Args:
            generators: A single generator whose state is copied into
                ``self.generator``, or ``None`` (no-op). A list is not
                accepted for base Distribution (only DistributionStack
                supports lists).
        """
        if isinstance(generators, list):
            if len(generators) != 1:
                raise ValueError(
                    f"Base Distribution expects a single generator, got {len(generators)}"
                )
            generators = generators[0]
        if generators is None or self.generator is None:
            return
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

    def _broadcast(self, x: float | list[float] | np.ndarray | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            if x.dim() == 0:
                return (
                    x.expand(self.n_features)
                    .clone()
                    .to(device=self.device, dtype=torch.float32)
                )
            return x.to(device=self.device, dtype=torch.float32)
        if isinstance(x, (int, float)):
            return torch.full((self.n_features,), x, device=self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def save_samples(self, n_samples: int, path: str | Path | None = None) -> Path:
        """Sample from this distribution and save to a ``.safetensors`` file and a companion ``.json``.

        The ``.safetensors`` file contains a single key ``"samples"`` with shape
        ``(n_samples, n_features)`` plus a ``class`` metadata field.

        The ``.json`` file is a human-readable summary: distribution class,
        constructor-relevant attributes, tensor shape/dtype, and sample count.
        It is *not* used by :meth:`ToyModel.fit` — it exists purely so users
        can inspect what a saved file contains without loading it.

        Args:
            n_samples: Number of samples to generate.
            path: Destination path (``.safetensors`` extension auto-appended).
                If ``None``, defaults to
                ``<ClassName>_<n_features>f_<n_samples>s_<YYYYMMDD_HHMMSS>.safetensors``.

        Returns:
            The resolved :class:`~pathlib.Path` that was written.
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(
                f"{type(self).__name__}_{self.n_features}f_{n_samples}s_{ts}"
                ".safetensors"
            )
        else:
            path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")

        raw = self.sample(n_samples)
        samples = raw[0] if isinstance(raw, tuple) else raw

        class_name = type(self).__name__
        save_file(
            {"samples": samples},
            str(path),
            metadata={"class": class_name},
        )

        info: dict = {
            "class": class_name,
            "attributes": self._collect_attrs(),
            "samples": {
                "shape": list(samples.shape),
                "dtype": str(samples.dtype),
                "n_samples": n_samples,
                "n_features": self.n_features,
            },
        }

        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(info, indent=2) + "\n")

        return path

    def _collect_attrs(self) -> dict:
        """Collect instance attributes into a JSON-serializable dict.

        Mirrors :meth:`AutoEncoderBase._collect_attrs`. Captures all instance
        attributes; generator state is serialized as seed + device.
        """
        out = {}
        for k, v in vars(self).items():
            serialized = self._serialize_value(v)
            if serialized is not _SKIP:
                out[k] = serialized
        return out

    @staticmethod
    def _serialize_value(v):
        """Convert a single value to a JSON-compatible representation.

        Mirrors :meth:`AutoEncoderBase._serialize_value`. Returns ``_SKIP``
        for values that should be omitted.
        """
        if v is None or isinstance(v, (int, float, str, bool)):
            return v
        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, torch.Generator):
            return {
                "type": "Generator",
                "device": str(v.device),
                "initial_seed": v.initial_seed(),
            }
        if isinstance(v, Tensor):
            return {"shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, (list, tuple)):
            items = [Distribution._serialize_value(x) for x in v]
            if any(x is _SKIP for x in items):
                return _SKIP
            return items
        if isinstance(v, dict):
            return {
                str(dk): Distribution._serialize_value(dv)
                for dk, dv in v.items()
                if Distribution._serialize_value(dv) is not _SKIP
            }
        return repr(v)

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

    def collect_generators(self) -> list[torch.Generator | None]:
        """Return one generator per child distribution.

        Returns:
            List of length ``len(self.distributions)``, where each element is
            the child's generator (or ``None`` if that child has no generator).
        """
        return [dist.generator for dist in self.distributions]

    def sync_generators(
        self, generators: torch.Generator | None | list[torch.Generator | None]
    ) -> None:
        """Sync generators into each child distribution.

        Args:
            generators: If a single generator (or ``None``), every child
                receives the same value. If a list, must have one entry per
                child distribution; ``None`` entries are skipped.
        """
        if isinstance(generators, list):
            if len(generators) != len(self.distributions):
                raise ValueError(
                    f"Expected {len(self.distributions)} generators, got {len(generators)}"
                )
            for dist, gen in zip(self.distributions, generators):
                dist.sync_generators(gen)
        else:
            for dist in self.distributions:
                dist.sync_generators(generators)

    def sample(self, batch_size):
        if self.sampling_mode == "independent":
            return torch.cat(
                [dist.sample(batch_size) for dist in self.distributions], dim=-1
            )

        result = torch.zeros(batch_size, self.n_features, device=self.device)
        offset = 0

        if self.sampling_mode == "single":
            indices = self._randint(0, len(self.distributions), (batch_size,))
            # CUDA perf: compute all per-distribution counts in one batched op,
            # then transfer counts to CPU in a single sync instead of one .item()
            # per sub-distribution.
            n_dists = len(self.distributions)
            counts = torch.bincount(indices, minlength=n_dists).cpu().tolist()
            for i, dist in enumerate(self.distributions):
                n_active = counts[i]
                if n_active > 0:
                    mask = indices == i
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

        for dist in distributions:
            if isinstance(dist, DistributionStack):
                raise TypeError(
                    "Nesting DistributionStack inside another DistributionStack is not "
                    "supported. Flatten all sub-distributions into a single stack."
                )

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
