"""Generic distribution for loading pre-generated samples from HuggingFace Hub."""

import warnings
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from ..utils.device import ensure_device
from .base import Distribution


class HuggingFaceDistribution(Distribution):
    """A distribution that serves pre-generated samples from HuggingFace Hub.

    Samples are downloaded from a HuggingFace dataset repository and kept on CPU
    (mmap-backed by safetensors; the OS manages paging). The ``sample()`` method
    returns random samples with replacement, moving each batch to ``device`` on
    demand to avoid loading the full dataset into device memory.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name").
        filename: Path to the safetensors file within the repository.
        revision: Optional branch, tag, or commit hash.
        device: Torch device for returned samples.
        generator: Optional generator for reproducible sampling order.
        buffer_size: Number of randomly-selected samples to pre-load onto
            ``device`` at once, refilling lazily when exhausted. Avoids a
            CPU→device transfer on every call to :meth:`sample`. Must be >=
            any ``batch_size`` passed to :meth:`sample`. On non-CPU devices
            defaults to ~1 GB worth of samples (computed from ``n_features``
            and dtype); ``None`` (per-batch transfer) on CPU.

    Example:
        >>> dist = HuggingFaceDistribution(
        ...     repo_id="your-org/occhio-distributions",
        ...     filename="sparse_uniform/samples/samples.safetensors",
        ... )
        >>> samples = dist.sample(64)  # shape: (64, 1296)
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        repo_type: str = "model",
        data_key: str = "samples",
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        buffer_size: int | None = None,
    ):

        # [2026-04-07 | OliverSieweke] TODO: Method should depend on repo type here / use dataset_info
        resolved_revision = (
            HfApi().repo_info(repo_id, revision=revision, repo_type=repo_type).sha
        )

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=resolved_revision,
        )

        if not Path(path).suffix == ".safetensors":
            warnings.warn(
                f"File '{filename}' does not have expected .safetensors extension. "
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
        # Keep backing store on CPU — moving it to device would load the full
        # dataset into device memory. Batches are transferred in sample() instead.
        self._samples = samples

        self.repo_id = repo_id
        self.filename = filename
        self.repo_type = repo_type
        self.data_key = data_key
        self.revision = resolved_revision
        if (
            buffer_size is None
            and self.device is not None
            and self.device.type != "cpu"
        ):
            target_bytes = 1 * 1024**3  # 1 GB
            bytes_per_sample = self.n_features * samples.element_size()
            buffer_size = target_bytes // bytes_per_sample
        self.buffer_size = buffer_size
        self._buffer: torch.Tensor | None = None
        self._buffer_ptr: int = 0

    def _refill_buffer(self) -> None:
        # CUDA perf: transfer indices to CPU once for indexing into CPU-resident
        # _samples, then transfer the sampled batch to device with non_blocking
        indices = self._randint(0, self._n_samples, (self.buffer_size,))
        cpu_indices = ensure_device(indices, "cpu", non_blocking=False)
        self._buffer = ensure_device(self._samples[cpu_indices], self.device)
        self._buffer_ptr = 0

    def sample(self, batch_size: int) -> Tensor:
        """Return random samples with replacement from the cached data.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Tensor of shape ``(batch_size, n_features)``.
        """
        if self.buffer_size is None:
            indices = self._randint(0, self._n_samples, (batch_size,))
            cpu_indices = ensure_device(indices, "cpu", non_blocking=False)
            batch = self._samples[cpu_indices]
            return ensure_device(batch, self.device) if self.device else batch

        if batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer_size ({self.buffer_size}). "
                "Increase buffer_size or reduce batch_size."
            )

        if self._buffer is None or self._buffer_ptr + batch_size > self.buffer_size:
            self._refill_buffer()

        batch = self._buffer[self._buffer_ptr : self._buffer_ptr + batch_size]
        self._buffer_ptr += batch_size
        return batch

    def clear_buffer(self) -> None:
        """Release the GPU buffer, freeing device memory.

        The buffer is refilled lazily on the next :meth:`sample` call.
        """
        self._buffer = None
        self._buffer_ptr = 0

    def to(self, device: torch.device | str) -> "HuggingFaceDistribution":
        # _samples intentionally stays on CPU — moving it would load the full
        # dataset onto the device, defeating the memory optimisation.
        self.device = torch.device(device)
        self.clear_buffer()  # buffer was built for the old device
        return self

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_samples"]
        del state["_buffer"]
        del state["_buffer_ptr"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        path = hf_hub_download(
            repo_id=state["repo_id"],
            filename=state["filename"],
            repo_type=state["repo_type"],
            revision=state["revision"],
        )
        # [2026-04-02 | OliverSieweke] TODO: make this a method to reuse?
        data = load_file(path)
        self._samples = data[state["data_key"]]
        self._buffer = None
        self._buffer_ptr = 0

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDistribution(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_samples={self._n_samples}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
