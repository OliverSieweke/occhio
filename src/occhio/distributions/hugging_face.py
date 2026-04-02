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
    (mmap-backed by safetensors; the OS manages paging). The ``sample()`` method
    returns random samples with replacement, moving each batch to ``device`` on
    demand to avoid loading the full dataset into device memory.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name").
        filename: Path to the safetensors file within the repository.
        revision: Optional branch, tag, or commit hash.
        device: Torch device for returned samples.
        generator: Optional generator for reproducible sampling order.

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
        # Keep backing store on CPU — moving it to device would load the full
        # dataset into device memory. Batches are transferred in sample() instead.
        self._samples = samples

        self.repo_id = repo_id
        self.filename = filename
        self.repo_type = repo_type
        self.data_key = data_key
        self.revision = resolved_revision

    def sample(self, batch_size: int) -> Tensor:
        """Return random samples with replacement from the cached data.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Tensor of shape ``(batch_size, n_features)``.
        """
        indices = self._randint(0, self._n_samples, (batch_size,))
        batch = self._samples[indices.cpu()]
        return batch.to(self.device) if self.device else batch

    def to(self, device: torch.device | str) -> "HuggingFaceDistribution":
        # _samples intentionally stays on CPU — moving it would load the full
        # dataset onto the device, defeating the memory optimisation.
        self.device = torch.device(device)
        return self

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_samples"]
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

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDistribution(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_samples={self._n_samples}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
