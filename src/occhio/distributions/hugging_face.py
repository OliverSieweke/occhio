"""Generic distribution for loading pre-generated samples from HuggingFace Hub."""

import warnings
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from .base import Distribution


class HuggingFaceDistribution(Distribution):
    """A distribution that serves pre-generated samples from HuggingFace Hub.

    Samples are downloaded from a HuggingFace dataset repository and kept on CPU
    (pin-memory backed for fast transfers). When a ``device`` is given, a
    ``buffer_batches``-sized chunk is prefetched to that device; ``sample()`` draws
    from the on-device buffer and refills it only when exhausted, amortising
    CPU→device transfer cost across many training steps.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name").
        filename: Path to the safetensors file within the repository.
        revision: Optional branch, tag, or commit hash.
        device: Torch device for returned samples.
        generator: Optional generator for reproducible sampling order.
        buffer_batches: How many batches to prefetch to ``device`` per transfer.
            Higher values reduce transfer frequency at the cost of device memory.
            Set to ``0`` to disable buffering (transfers every call).

    Example:
        >>> dist = HuggingFaceDistribution(
        ...     repo_id="kaushikreddyxyz/occhio-distributions",
        ...     filename="sparse_uniform/samples/samples.safetensors",
        ... )
        >>> samples = dist.sample(64)  # shape: (64, 1296)
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        repo_type: str = "dataset",
        data_key: str = "samples",
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        buffer_batches: int = 32,
    ):
        resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=resolved_revision,
        )

        if not Path(path).suffix == ".safetensors":
            warnings.warn(
                f"File '{filename}' does not have expected .safetensors extension."
                f"This may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

        data = load_file(path)

        if data_key not in data:
            raise KeyError(
                f"Expected key '{data_key}' not found in safetensors file. "
                f"Available keys: {list(data.keys())}"
            )

        samples = data[data_key]

        if samples.ndim != 2:
            raise ValueError(
                f"Expected samples to be 2D (n_samples, n_features), "
                f"but got shape {samples.shape}"
            )

        super().__init__(samples.shape[1], device=device, generator=generator)

        self._n_samples = samples.shape[0]
        # Keep backing store on CPU. Pin memory when a device is given so that
        # CPU→device DMA transfers bypass the CPU entirely.
        self._samples = samples.pin_memory() if device else samples
        self._buffer_batches = buffer_batches
        self._buffer: Tensor | None = None
        self._buffer_pos: int = 0

        self.filename = filename
        self.revision = resolved_revision

    def _refill_buffer(self, batch_size: int) -> None:
        # Generate indices on self.device (respects self.generator), then move to
        # CPU for indexing the CPU-resident backing store.
        indices = self._randint(
            0, self._n_samples, (self._buffer_batches * batch_size,)
        ).cpu()
        self._buffer = self._samples[indices].to(self.device, non_blocking=True)
        self._buffer_pos = 0

    def sample(self, batch_size: int) -> Tensor:
        """Return random samples with replacement from the cached data.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Tensor of shape ``(batch_size, n_features)``.
        """
        if self.device and self._buffer_batches:
            if self._buffer is None or self._buffer_pos + batch_size > len(
                self._buffer
            ):
                indices = self._randint(
                    0, self._n_samples, (self._buffer_batches * batch_size,)
                ).cpu()
                self._buffer = self._samples[indices].to(self.device, non_blocking=True)
                self._buffer_pos = 0

            batch = self._buffer[self._buffer_pos : self._buffer_pos + batch_size]
            self._buffer_pos += batch_size
            return batch

        indices = self._randint(0, self._n_samples, (batch_size,))
        batch = self._samples[indices.cpu()]
        return batch.to(self.device) if self.device else batch

    def to(self, device: torch.device | str) -> "HuggingFaceDistribution":
        # _samples intentionally stays on CPU — moving it would load the full
        # dataset onto the device, defeating the memory optimisation.
        # Invalidate the buffer so the next sample() refills on the new device.
        self.device = torch.device(device)
        self._buffer = None
        self._buffer_pos = 0
        return self

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDistribution(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_samples={self._n_samples}, buffer_batches={self._buffer_batches}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
