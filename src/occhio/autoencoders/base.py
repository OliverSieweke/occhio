"""Base class for all autoencoders."""

import datetime
import functools
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from ..utils.device import _same_device


# Sentinel for values that should be omitted from JSON.
_SKIP = object()


class AutoEncoderBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """features --> latent"""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """latent --> features"""

    @property
    def feature_vectors(self) -> Tensor:
        return self.encode(torch.eye(self.n_features, device=self.device))

    @abstractmethod
    def resample_weights(self):
        """Reset / resample all weights"""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
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
        save_file(self.state_dict(), str(path), metadata={"class": class_name})

        info: dict = {
            "class": class_name,
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
        """This ensures that `n_features` and `n_hidden` are defined at creation"""
        super().__init_subclass__(**kwargs)
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
