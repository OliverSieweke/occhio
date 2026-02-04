# ABOUTME: Defines a pluggable autoencoder module for toy superposition experiments.
# ABOUTME: Supports arbitrary encoder/decoder modules with optional tied-weight decoding.

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _get_activation(name_or_fn: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    if callable(name_or_fn):
        return name_or_fn
    name = name_or_fn.lower()
    if name == "relu":
        return torch.relu
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name_or_fn}")


class AutoEncoder(nn.Module):
    """
    Autoencoder base: encode → activation → decode.
    Accepts arbitrary encoder/decoder modules; supports optional tied decoding.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        activation: str | Callable[[Tensor], Tensor] = "identity",
        tied_weights: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.activation = _get_activation(activation)
        self.tied_weights = tied_weights

        if decoder is None and not tied_weights:
            raise ValueError("decoder must be provided unless tied_weights=True")

        self.decoder = decoder

        if tied_weights:
            if not isinstance(self.encoder, nn.Linear):
                raise TypeError("tied_weights=True requires encoder to be nn.Linear")
            if decoder is None:
                bias = getattr(self.encoder, "bias", None)
                self.decoder_bias = nn.Parameter(torch.zeros(self.encoder.in_features)) if bias is not None else None
            else:
                self.decoder_bias = getattr(decoder, "bias", None)
        else:
            self.decoder_bias = None

    def encode(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        if self.decoder is None:
            raise RuntimeError("decoder is not set")
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    @property
    def W(self) -> Tensor:
        if hasattr(self.encoder, "weight"):
            return self.encoder.weight  # type: ignore[return-value]
        raise AttributeError("encoder has no weight attribute")
