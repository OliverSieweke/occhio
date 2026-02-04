# ABOUTME: Defines a minimal autoencoder module for toy superposition experiments.
# ABOUTME: Provides encoding, decoding, and weight exposure utilities.

from __future__ import annotations

from typing import Callable

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
    Minimal autoencoder: Linear encode → activation → Linear decode.

    The typical setup: n_features → n_hidden → n_features where n_hidden < n_features.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        bias: bool = False,
        tied_weights: bool = True,
        activation: str | Callable[[Tensor], Tensor] = "relu",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.tied_weights = tied_weights
        self.activation = _get_activation(activation)

        self.encoder = nn.Linear(n_features, n_hidden, bias=bias)
        if tied_weights:
            self.decoder = None
            self.decoder_bias = nn.Parameter(torch.zeros(n_features)) if bias else None
        else:
            self.decoder = nn.Linear(n_hidden, n_features, bias=bias)
            self.decoder_bias = None

    def encode(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    @property
    def W(self) -> Tensor:
        return self.encoder.weight
