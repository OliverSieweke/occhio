"""Generic autoencoder loader for pre-trained models from HuggingFace Hub."""

import warnings
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file

from .autoencoder import TiedLinearRelu
from ..benchmark.configs import OCCHIO_HF_MODELS_REPO


# [2026-03-26 | OliverSieweke] TODO: Think about how to import non TiedLinearRelu models here...
class HuggingFaceAutoEncoder(TiedLinearRelu):
    """A pre-trained autoencoder loaded from HuggingFace Hub.

    Downloads model weights from a HuggingFace model repository and loads them
    into a ``TiedLinearRelu`` architecture. Model dimensions are inferred from
    the saved weights.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/model-name").
        filename: Path to the safetensors file within the repository.
        revision: Optional branch, tag, or commit hash.
        device: Torch device for the model.

    Example:
        >>> ae = HuggingFaceAutoEncoder(
        ...     repo_id="kaushikreddyxyz/occhio-models",
        ...     filename="correlated_pairs/model.safetensors",
        ... )
        >>> z = ae.encode(x)  # x shape: (batch, 512)
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        device: torch.device | str | None = None,
    ):
        resolved_revision = HfApi().model_info(repo_id, revision=revision).sha

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=resolved_revision,
            repo_type="model",
        )

        if not Path(path).suffix == ".safetensors":
            warnings.warn(
                f"File '{filename}' does not have expected .safetensors extension."
                f"This may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

        state_dict = load_file(path)

        if "W" not in state_dict:
            raise KeyError(
                f"Expected key 'W' not found in state dict. "
                f"Available keys: {list(state_dict.keys())}"
            )

        W = state_dict["W"]

        if W.ndim != 2:
            raise ValueError(
                f"Expected weight matrix 'W' to be 2D (n_hidden, n_features), "
                f"but got shape {W.shape}"
            )

        n_hidden, n_features = W.shape

        # Initialize parent class (this will call resample_weights)
        super().__init__(n_features, n_hidden, device=device)
        # Load the actual weights (overwrites random init)
        self.load_state_dict(state_dict)

        if device is not None:
            self.to(device)

        self.filename = filename
        self.revision = resolved_revision

    def __repr__(self) -> str:
        return (
            f"HuggingFaceAutoEncoder(filename={self.filename!r}, n_features={self.n_features}, "
            f"n_hidden={self.n_hidden}, device={self.device})"
        )

    def __str__(self) -> str:
        return self.__repr__()
