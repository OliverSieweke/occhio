# ABOUTME: Provides illustrative AutoEncoder subclasses showcasing extensibility.
# ABOUTME: Demonstrates varied encoder/decoder choices and constraints.

from __future__ import annotations

import torch
from torch import nn

from occhio.autoencoder import AutoEncoder


class SparseTiedAutoEncoder(AutoEncoder):
    """Linear encoder with tied decoding; biasless for clean analysis."""

    def __init__(self, n_features: int, n_hidden: int):
        encoder = nn.Linear(n_features, n_hidden, bias=False)
        super().__init__(encoder=encoder, decoder=None, tied_weights=True, validate_shapes=True)


class DeepNonlinearAutoEncoder(AutoEncoder):
    """Two-layer encoder/decoder with ReLU nonlinearity and untied weights."""

    def __init__(self, n_features: int, n_hidden: int, mid_hidden: int):
        encoder = nn.Sequential(
            nn.Linear(n_features, mid_hidden),
            nn.ReLU(),
            nn.Linear(mid_hidden, n_hidden),
        )
        decoder = nn.Sequential(
            nn.Linear(n_hidden, mid_hidden),
            nn.ReLU(),
            nn.Linear(mid_hidden, n_features),
        )
        super().__init__(encoder=encoder, decoder=decoder, activation="identity", tied_weights=False, validate_shapes=False)


class DropoutAutoEncoder(AutoEncoder):
    """Adds dropout in the encoder path to encourage robustness."""

    def __init__(self, n_features: int, n_hidden: int, p: float = 0.1):
        encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p),
        )
        decoder = nn.Linear(n_hidden, n_features)
        super().__init__(encoder=encoder, decoder=decoder, activation="identity", tied_weights=False, validate_shapes=False)


class BottleneckConvAutoEncoder(AutoEncoder):
    """1D convolutional bottleneck followed by linear projection."""

    def __init__(self, n_channels: int, seq_len: int, n_hidden: int, kernel_size: int = 3):
        conv = nn.Conv1d(n_channels, n_hidden, kernel_size, padding=kernel_size // 2)
        encoder = nn.Sequential(
            conv,
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(n_hidden * seq_len, n_hidden),
        )
        decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * seq_len),
            nn.Unflatten(1, (n_hidden, seq_len)),
            nn.ConvTranspose1d(n_hidden, n_channels, kernel_size, padding=kernel_size // 2),
        )
        super().__init__(encoder=encoder, decoder=decoder, activation="identity", tied_weights=False, validate_shapes=False)
