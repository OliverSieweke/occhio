"""HuggingFace Hub integration for autoencoder models."""

import tempfile
from pathlib import Path

import torch

from .base import AutoEncoderBase


def load_autoencoder_from_hub(
    repo_id: str,
    filename: str = "model.safetensors",
    *,
    revision: str | None = None,
    device: torch.device | str | None = None,
) -> AutoEncoderBase:
    """Download and reconstruct an autoencoder from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID.
        filename: Path to the safetensors file within the repo.
        revision: Branch, tag, or commit hash.
        device: Device to place the model on.

    Returns:
        Reconstructed autoencoder instance.
    """
    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = HfApi().model_info(repo_id, revision=revision).sha

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=resolved_revision,
        repo_type="model",
    )

    return AutoEncoderBase.from_local(local_path, device=device)


def push_autoencoder_to_hub(
    ae: AutoEncoderBase,
    repo_id: str,
    *,
    filename: str = "model.safetensors",
    commit_message: str | None = None,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Save and upload an autoencoder to HuggingFace Hub.

    Uploads both the ``.safetensors`` weights file and the companion
    ``.json`` metadata file.

    Args:
        ae: The autoencoder to upload.
        repo_id: HuggingFace Hub repository ID.
        filename: Destination filename in the repo.
        commit_message: Commit message (auto-generated if None).
        private: Whether to create a private repo.
        token: HuggingFace API token.

    Returns:
        URL of the repository.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)

    if commit_message is None:
        commit_message = f"Upload {type(ae).__name__} ({ae.n_features}x{ae.n_hidden})"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = ae.save_weights(Path(tmpdir) / filename)
        json_path = local_path.with_suffix(".json")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        json_filename = str(Path(filename).with_suffix(".json"))
        if json_path.exists():
            api.upload_file(
                path_or_fileobj=str(json_path),
                path_in_repo=json_filename,
                repo_id=repo_id,
                commit_message=commit_message,
            )

    return f"https://huggingface.co/{repo_id}"
