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

    Samples are downloaded from a HuggingFace dataset repository and loaded eagerly
    into memory. The ``sample()`` method returns random samples with replacement
    from the cached data.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/dataset-name").
        filename: Path to the safetensors file within the repository.
        revision: Optional branch, tag, or commit hash.
        device: Torch device for samples.
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
        # Keep backing store on CPU; only sampled batches are moved to device.
        self._samples = samples

        self.filename = filename
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
        # Update device without moving _samples; only batches are transferred on demand.
        self.device = torch.device(device)
        return self

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDistribution(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_samples={self._n_samples}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
