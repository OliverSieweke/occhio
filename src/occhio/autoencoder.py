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
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        n_features: Optional[int] = None,
        n_hidden: Optional[int] = None,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        tied_weights: bool = False,
        validate_shapes: bool = True,
    ):
        super().__init__()
        if encoder is None:
            if n_features is None or n_hidden is None:
                raise ValueError("n_features and n_hidden are required when encoder is not provided")
            encoder = nn.Linear(n_features, n_hidden, bias=False)
        self.encoder = encoder
        self.activation = _get_activation(activation)
        self.tied_weights = tied_weights
        self.validate_shapes = validate_shapes

        encoder_out = getattr(self.encoder, "out_features", None)
        encoder_in = getattr(self.encoder, "in_features", None)

        if decoder is None and not tied_weights:
            if encoder_out is None:
                raise ValueError("decoder must be provided unless tied_weights=True")
            decoder = nn.Linear(encoder_out, encoder_in, bias=False) if encoder_in is not None else None
            if decoder is None:
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

        if self.validate_shapes:
            self._validate_shapes()

    def encode(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        if self.decoder is None:
            raise RuntimeError("decoder must be provided unless tied_weights=True")
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor: 
        return self.decode(self.encode(x))

    @property
    def W(self) -> Tensor:
        if hasattr(self.encoder, "weight"):
            return self.encoder.weight  # type: ignore[return-value]
        raise AttributeError("encoder has no weight attribute")

    def _validate_shapes(self) -> None:
        if isinstance(self.encoder, nn.Linear):
            if self.encoder.out_features >= self.encoder.in_features:
                raise ValueError("encoder output dimension must be smaller than input dimension to encourage compression")
        if not self.tied_weights and isinstance(self.decoder, nn.Linear) and isinstance(self.encoder, nn.Linear):
            if self.decoder.in_features != self.encoder.out_features or self.decoder.out_features != self.encoder.in_features:
                raise ValueError("decoder dimensions must mirror encoder dimensions (hidden → input)")


def create_autoencoder(
    n_features: int,
    n_hidden: int,
    *,
    activation: str | Callable[[Tensor], Tensor] = "relu",
    tied_weights: bool = False,
    validate_shapes: bool = True,
) -> AutoEncoder:
    encoder = nn.Linear(n_features, n_hidden, bias=False)
    decoder = None if tied_weights else nn.Linear(n_hidden, n_features, bias=False)
    return AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        n_features=n_features,
        n_hidden=n_hidden,
        activation=activation,
        tied_weights=tied_weights,
        validate_shapes=validate_shapes,
    )
