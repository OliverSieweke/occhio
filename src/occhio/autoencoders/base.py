"""Base class for all autoencoders."""

import datetime
import functools
import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from ..utils.device import _same_device

# Params that cannot be serialized and should be excluded from config.
_NON_SERIALIZABLE_PARAMS = frozenset({"loss_fn", "device", "generator"})

# Sentinel for values that should be omitted from JSON.
_SKIP = object()


class AutoEncoderBase(nn.Module, ABC):
    """Abstract base class for all autoencoders.

    Subclasses must implement :meth:`encode`, :meth:`decode`, and
    :meth:`resample_weights`.  The base class provides:

    - Default ``forward()`` (encode + decode) and ``loss()`` (importance-weighted MSE)
    - Serialization via :meth:`save_weights` / :meth:`from_local` / :meth:`from_hub` /
      :meth:`push_to_hub`
    - Automatic config extraction from constructor signature (:meth:`get_config`)
    - Auto-registration of subclasses for deserialization

    Args:
        n_features: Input/output dimensionality (number of ground-truth features).
        n_hidden: Latent/hidden dimensionality.
        loss_fn: Optional custom loss function replacing the default MSE loss.
        device: Torch device for parameters.
        generator: Optional ``torch.Generator`` for reproducible weight init.
    """

    # Auto-populated registry: class_name -> class.
    _registry: ClassVar[dict[str, type["AutoEncoderBase"]]] = {}

    # Aliases for renamed classes (old_name -> current_name).
    _class_aliases: ClassVar[dict[str, str]] = {
        "HuggingFaceAutoEncoder": "TiedLinearRelu",
        "PretrainedAE": "TiedLinearRelu",
    }

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode feature-space input to hidden/latent representation.

        Args:
            x: Input tensor of shape ``(batch, n_features)``.

        Returns:
            Latent tensor of shape ``(batch, n_hidden)``.
        """

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode hidden/latent representation back to feature space.

        Args:
            z: Latent tensor of shape ``(batch, n_hidden)``.

        Returns:
            Reconstructed tensor of shape ``(batch, n_features)``.
        """

    @property
    def feature_vectors(self) -> Tensor:
        return self.encode(torch.eye(self.n_features, device=self.device))

    @abstractmethod
    def resample_weights(self):
        """Reset / reinitialize all learnable parameters.

        Called during ``__init__`` and available for manual reinitialization.
        """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape ``(batch, n_features)``.

        Returns:
            ``(x_hat, z)`` -- the reconstruction and latent representation.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x_true: Tensor, x_hat: Tensor, importances: Tensor | None):
        """The associated loss function."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        return torch.mean(torch.sum(importances * torch.square(x_true - x_hat), dim=-1))

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        loss_fn: Callable | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ):
        """Initialize the AutoEncoder class.

        Note that we write device to `_init_device`, which remembers where the user intends to store the device.
        """
        super().__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden

        if loss_fn is not None:
            self.loss = loss_fn  # type: ignore[method-assign]
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
        self.generator = generator

    @property
    def device(self) -> torch.device | None:
        """Return the device of the first parameter, falling back to the
        device passed at construction time (needed during ``__init__`` before
        any parameters have been created)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._init_device

    def save_weights(self, path: str | Path | None = None) -> Path:
        """Save model weights to a ``.safetensors`` file and a companion ``.json``.

        The ``.safetensors`` file contains the full ``state_dict`` plus a
        ``class`` metadata field for :meth:`load_weights` validation.

        The ``.json`` file is a human-readable summary of the model: class
        name, constructor-relevant attributes, and per-parameter shapes/dtypes.
        It is *not* used by :meth:`load_weights` — it exists purely so users
        can inspect what a saved file contains without loading it.

        Args:
            path: Destination path (``.safetensors`` extension auto-appended).
                If ``None``, defaults to
                ``<ClassName>_<n_features>x<n_hidden>_<YYYYMMDD_HHMMSS>.safetensors``.
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(
                f"{type(self).__name__}_{self.n_features}x{self.n_hidden}_{ts}"
                ".safetensors"
            )
        else:
            path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")

        class_name = type(self).__name__
        config = self.get_config()
        metadata = {
            "class": class_name,
            "config": json.dumps(config),
        }
        save_file(self.state_dict(), str(path), metadata=metadata)

        info: dict = {
            "class": class_name,
            "config": config,
            "attributes": self._collect_attrs(),
            "parameters": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in self.state_dict().items()
            },
            "total_params": sum(p.numel() for p in self.parameters()),
        }

        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(info, indent=2) + "\n")

        return path

    # nn.Module internal attrs that are bookkeeping, not model config.
    _NN_MODULE_INTERNALS = frozenset(
        {
            "_parameters",
            "_buffers",
            "_modules",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
            "_forward_pre_hooks_with_kwargs",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
            "_non_persistent_buffers_set",
            "_is_full_backward_hook",
            "training",
        }
    )

    def _collect_attrs(self) -> dict:
        """Collect all instance attributes into a JSON-serializable dict.

        Captures everything on the instance except nn.Module bookkeeping
        and nn.Parameter/ParameterList objects (those go in ``parameters``).
        """
        out = {}
        for k, v in vars(self).items():
            if k in self._NN_MODULE_INTERNALS:
                continue
            serialized = self._serialize_value(v)
            if serialized is not _SKIP:
                out[k] = serialized
        return out

    @staticmethod
    def _serialize_value(v):
        """Convert a single value to a JSON-compatible representation."""
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
        if isinstance(v, nn.Parameter):
            return _SKIP
        if isinstance(v, nn.ParameterList):
            return _SKIP
        if isinstance(v, Tensor):
            return {"shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, (list, tuple)):
            items = [AutoEncoderBase._serialize_value(x) for x in v]
            if any(x is _SKIP for x in items):
                return _SKIP
            return items
        if isinstance(v, dict):
            return {
                str(dk): AutoEncoderBase._serialize_value(dv)
                for dk, dv in v.items()
                if AutoEncoderBase._serialize_value(dv) is not _SKIP
            }
        # Fallback: repr for anything else (enums, callables, etc.)
        return repr(v)

    def get_config(self) -> dict[str, Any]:
        """Return kwargs needed to reconstruct this instance via ``cls(**config)``.

        The default implementation inspects the constructor signature and looks
        up each parameter as ``self.<param>`` or ``self._<param>``. Subclasses
        with mismatched parameter/attribute names should override this method.
        """
        sig = inspect.signature(type(self).__init__)
        config = {}
        for name, param in sig.parameters.items():
            if name == "self" or name in _NON_SERIALIZABLE_PARAMS:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            # Look up self.<name>, then self._<name> (SynthAE pattern)
            if hasattr(self, name):
                val = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                val = getattr(self, f"_{name}")
            else:
                continue
            # Only include JSON-serializable values
            if isinstance(val, (int, float, str, bool, list, tuple)):
                config[name] = val
            elif val is None:
                config[name] = val
        return config

    @classmethod
    def from_local(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> "AutoEncoderBase":
        """Reconstruct any autoencoder from a ``.safetensors`` file.

        Reads the class name and constructor config from file metadata,
        looks up the class in the auto-populated registry, constructs an
        instance, and loads the saved weights.

        Args:
            path: Path to a ``.safetensors`` file.
            device: Device to place the model on (overrides saved config).
        """
        path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

        if not metadata or "class" not in metadata:
            raise ValueError(
                f"File {path} has no 'class' metadata. "
                f"Was it saved with save_weights()?"
            )

        class_name = metadata["class"]

        # Check aliases for renamed classes
        class_name = cls._class_aliases.get(class_name, class_name)

        ae_cls = cls._registry.get(class_name)
        if ae_cls is None:
            raise ValueError(
                f"Unknown autoencoder class {class_name!r}. "
                f"Available: {sorted(cls._registry.keys())}. "
                f"Ensure the module defining {class_name} has been imported."
            )

        config_str = metadata.get("config")
        if config_str is not None:
            config = json.loads(config_str)
        else:
            # Legacy file: no config. Infer from state dict for simple cases.
            state_dict = load_file(str(path))
            if "W" in state_dict:
                n_hidden, n_features = state_dict["W"].shape
                config = {"n_features": int(n_features), "n_hidden": int(n_hidden)}
            else:
                raise ValueError(
                    f"File {path} has no 'config' metadata and no 'W' key "
                    f"to infer dimensions. Construct the model manually "
                    f"and use load_weights(path)."
                )

        # Validate config against constructor
        sig = inspect.signature(ae_cls.__init__)
        valid_params = {
            name
            for name, p in sig.parameters.items()
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        unknown = set(config.keys()) - valid_params
        if unknown:
            raise ValueError(
                f"Saved config for {class_name} has unexpected keys {unknown}. "
                f"Valid params: {sorted(valid_params)}"
            )

        if device is not None:
            config["device"] = device

        try:
            instance = ae_cls(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct {class_name} with config {config}: {e}"
            ) from e

        instance.load_state_dict(load_file(str(path)), strict=True)

        if device is not None:
            instance.to(device)

        return instance

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        filename: str = "model.safetensors",
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
    ) -> "AutoEncoderBase":
        """Download and reconstruct an autoencoder from HuggingFace Hub.

        Args:
            repo_id: HuggingFace Hub repository ID.
            filename: Path to the safetensors file within the repo.
            revision: Branch, tag, or commit hash.
            device: Device to place the model on.
        """
        from .hub import load_autoencoder_from_hub

        return load_autoencoder_from_hub(
            repo_id, filename, revision=revision, device=device
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        filename: str = "model.safetensors",
        commit_message: str | None = None,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Save and upload this autoencoder to HuggingFace Hub.

        Args:
            repo_id: HuggingFace Hub repository ID.
            filename: Destination filename in the repo.
            commit_message: Commit message (auto-generated if None).
            private: Whether to create a private repo.
            token: HuggingFace API token.

        Returns:
            URL of the uploaded model.
        """
        from .hub import push_autoencoder_to_hub

        return push_autoencoder_to_hub(
            self,
            repo_id,
            filename=filename,
            commit_message=commit_message,
            private=private,
            token=token,
        )

    def load_weights(self, path: str | Path, *, strict: bool = True) -> None:
        """Load weights from a .safetensors file into this model.

        Validates that the file was saved from the same ``AutoEncoderBase``
        subclass before loading.  The model must already be constructed
        with the desired architecture — this method only overwrites
        parameter data in-place.

        Parameters
        ----------
        path : str | Path
            Path to a ``.safetensors`` file (extension auto-appended).
        strict : bool
            Passed to ``nn.Module.load_state_dict``.  When *True* (default),
            raises on missing or unexpected keys.
        """
        path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

        saved_class = metadata.get("class") if metadata else None
        if saved_class is None:
            raise ValueError(
                f"File {path} has no 'class' metadata. Was it saved with save_weights()?"
            )
        if saved_class != type(self).__name__:
            raise TypeError(
                f"Weights were saved from {saved_class}, "
                f"but this model is {type(self).__name__}"
            )

        self.load_state_dict(load_file(str(path)), strict=strict)

    def __init_subclass__(cls, **kwargs):
        """Register subclass and ensure n_features/n_hidden are set."""
        super().__init_subclass__(**kwargs)
        AutoEncoderBase._registry[cls.__name__] = cls
        original_init = cls.__init__

        @functools.wraps(original_init)
        def checked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            for attr in ("n_features", "n_hidden"):
                if not hasattr(self, attr):
                    raise AttributeError(
                        f"{cls.__name__}.__init__ must set self.{attr}"
                    )

        cls.__init__ = checked_init  # ty:ignore
