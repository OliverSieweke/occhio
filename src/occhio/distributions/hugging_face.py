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

    Samples are kept on CPU. When a CUDA ``device`` is given, double-buffering is
    used: while the GPU consumes the current buffer, a background stream transfers
    the next one. On non-CUDA devices a single buffer is used. ``sample()`` draws
    from the on-device buffer and refills only when exhausted, amortising
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
        # Keep backing store on CPU (mmap-backed by safetensors; OS manages paging).
        self._samples = samples
        self._buffer_batches = buffer_batches
        self._buffer: Tensor | None = None
        self._buffer_pos: int = 0
        # Double-buffering state (CUDA only).
        self._next_buffer: Tensor | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None

        self.filename = filename
        self.revision = resolved_revision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cuda(self) -> bool:
        return self.device is not None and self.device.type == "cuda"

    def _random_indices(self, n: int) -> Tensor:
        # Generate on CPU directly to avoid a GPU→CPU sync when we immediately
        # need CPU indices to index _samples.
        return torch.randint(0, self._n_samples, (n,))

    def _start_prefetch(self, batch_size: int) -> None:
        """Kick off an async transfer of the next buffer on the prefetch stream."""
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        indices = self._random_indices(self._buffer_batches * batch_size)
        with torch.cuda.stream(self._prefetch_stream):
            self._next_buffer = self._samples[indices].to(
                self.device, non_blocking=True
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Tensor:
        """Return random samples with replacement from the cached data.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Tensor of shape ``(batch_size, n_features)``.
        """
        if self.device and self._buffer_batches:
            if self._buffer is None:
                # Cold start: blocking fill, then kick off first prefetch.
                indices = self._random_indices(self._buffer_batches * batch_size)
                self._buffer = self._samples[indices].to(self.device)
                self._buffer_pos = 0
                if self._is_cuda():
                    self._start_prefetch(batch_size)

            elif self._buffer_pos + batch_size > len(self._buffer):
                if self._is_cuda() and self._next_buffer is not None:
                    # Wait for the background transfer, swap buffers, start next prefetch.
                    torch.cuda.current_stream(self.device).wait_stream(
                        self._prefetch_stream
                    )
                    self._buffer = self._next_buffer
                    self._next_buffer = None
                    self._buffer_pos = 0
                    self._start_prefetch(batch_size)
                else:
                    # Non-CUDA single buffer: synchronous refill.
                    indices = self._random_indices(self._buffer_batches * batch_size)
                    self._buffer = self._samples[indices].to(
                        self.device, non_blocking=True
                    )
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
        # Invalidate both buffers and the prefetch stream so everything is
        # recreated on the new device.
        self.device = torch.device(device)
        self._buffer = None
        self._buffer_pos = 0
        self._next_buffer = None
        self._prefetch_stream = None
        return self

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDistribution(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_samples={self._n_samples}, buffer_batches={self._buffer_batches}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
