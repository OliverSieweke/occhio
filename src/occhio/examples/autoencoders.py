# ABOUTME: Example AutoEncoder subclasses illustrating custom architectures.
# ABOUTME: Shows how to implement create_encoder and create_decoder variants.

from __future__ import annotations

import torch
from torch import nn

from occhio.autoencoder import AutoEncoder, TiedLinear


class LinearBottleneckAE(AutoEncoder):
    """Two-layer linear bottleneck with ReLU on the latent."""

    def __init__(self, input_dim: int = 8, bottleneck: int = 3):
        self.input_dim = input_dim
        self.bottleneck = bottleneck
        super().__init__(activation=nn.ReLU(), tied_weights=False)

    def create_encoder(self) -> nn.Module:
        return nn.Sequential(nn.Linear(self.input_dim, self.bottleneck), nn.ReLU())

    def create_decoder(self) -> nn.Module:
        return nn.Linear(self.bottleneck, self.input_dim)


class DeepTiedLinearAE(AutoEncoder):
    """Multi-layer encoder with tied linear decoder."""

    def __init__(self, dims: list[int] | tuple[int, ...] = (16, 8, 4)):
        self.dims = list(dims)
        super().__init__(activation="identity", tied_weights=False)

    def create_encoder(self) -> nn.Module:
        layers = []
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            layers.append(nn.ReLU())
        layers.pop()
        self._encoder_linear = [m for m in layers if isinstance(m, nn.Linear)]
        return nn.Sequential(*layers)

    def create_decoder(self) -> nn.Module:
        linear_layers = [layer for layer in self._encoder_linear]
        decoder_layers = []
        for idx, layer in enumerate(reversed(linear_layers)):
            decoder_layers.append(TiedLinear(layer, bias=True))
            if idx != len(linear_layers) - 1:
                decoder_layers.append(nn.ReLU())
        return nn.Sequential(*decoder_layers)


class Conv1dAutoEncoder(AutoEncoder):
    """Narrow convolutional encoder/decoder for sequence features."""

    def __init__(self, channels: int = 4, latent_channels: int = 2, kernel_size: int = 3):
        self.channels = channels
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        super().__init__(activation="identity", tied_weights=False)

    def create_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(self.channels, self.latent_channels, kernel_size=self.kernel_size, padding="same"),
            nn.ReLU(),
        )

    def create_decoder(self) -> nn.Module:
        return nn.Conv1d(self.latent_channels, self.channels, kernel_size=self.kernel_size, padding="same")


class LayerNormAutoEncoder(AutoEncoder):
    """Adds layer normalization around linear projections."""

    def __init__(self, dim: int = 6, latent: int = 3):
        self.dim = dim
        self.latent = latent
        super().__init__(activation="gelu", tied_weights=False)

    def create_encoder(self) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.latent))

    def create_decoder(self) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(self.latent), nn.Linear(self.latent, self.dim))
