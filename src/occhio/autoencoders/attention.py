"""Attention-based autoencoders."""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import AutoEncoderBase


def softmax1(x, dim=-1):
    e = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return e / (e.sum(dim=dim, keepdim=True) + 1)


class AttnLinearAE(AutoEncoderBase):
    """Multi-head softmax bottleneck autoencoder for MRH-style experiments.

    Each encoder head projects the input to ``dict_size`` logits, applies softmax, then multiplies by a learned value matrix to produce a ``value_dim``-sized output.  The latent is the concatenation of all heads' outputs (``n_hidden = n_heads * value_dim``).  The decoder is a linear projection with ReLU, mirroring :class:`TiedLinearRelu`.

    The softmax constraint forces each head's contribution to be a convex combination of its dictionary vectors, providing the architectural inductive bias for Minkowski-style tile representations.

    Parameters
    ----------
    n_features : int
        Input / output dimensionality.
    n_hidden : int
        Total latent dimensionality (must be divisible by ``n_heads``).
    n_heads : int
        Number of independent softmax heads.
    dict_size : int
        Number of dictionary elements (archetypes) per head.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        dict_size: int,
        **kwargs,
    ) -> None:
        super().__init__(n_features, n_hidden, **kwargs)

        if n_hidden % n_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.dict_size = dict_size
        self.value_dim = n_hidden // n_heads

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        dev = self.device
        gen = self.generator

        # Per-head encoder projections: (n_features, dict_size)
        self.encoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.n_features,
                        self.dict_size,
                        device=dev,
                        generator=gen,
                    )
                    / sqrt(self.n_features)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.W_mix = nn.Parameter(torch.zeros(self.n_hidden, self.n_hidden, device=dev))

        # Per-head value matrices: (dict_size, value_dim)
        self.value_matrices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size,
                        self.value_dim,
                        device=dev,
                        generator=gen,
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.alpha = nn.Parameter(torch.tensor([0.1], device=dev))

        # Decoder: linear projection from latent to features
        self.W_out = nn.Parameter(
            torch.randn(
                self.n_hidden,
                self.n_features,
                device=dev,
                generator=gen,
            )
            / sqrt(self.n_hidden)
        )
        self.b = nn.Parameter(torch.zeros(self.n_features, device=dev))

    def encode(self, x: Tensor) -> Tensor:
        parts = []
        for h in range(self.n_heads):
            logits = x @ self.encoder_projs[h]  # (B, dict_size)
            weights = softmax1(logits, dim=-1)  # (B, dict_size)
            values = weights @ self.value_matrices[h]  # (B, value_dim)
            parts.append(values)
        z = torch.cat(parts, dim=-1) @ self.W_mix
        return self.alpha * z + (1 - self.alpha) * (x @ self.W_out.T)

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W_out + self.b)


class AttnAttnAE(AutoEncoderBase):
    """Multi-head softmax autoencoder with attention-like encoding and decoding.

    Encoder: ``softmax(x @ P_h) @ V_h`` per head, concatenated to latent.
    Decoder: ``softmax(z_h @ Q_h) @ U_h`` per head, summed + bias + ReLU.

    Both encoder and decoder use independent per-head softmax-weighted
    dictionary lookups with separate parameters.

    Parameters
    ----------
    n_features : int
        Input / output dimensionality.
    n_hidden : int
        Total latent dimensionality (must be divisible by ``n_heads``).
    n_heads : int
        Number of independent softmax heads.
    dict_size : int
        Number of dictionary elements (archetypes) per head.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        dict_size: int,
        **kwargs,
    ) -> None:
        super().__init__(n_features, n_hidden, **kwargs)

        if n_hidden % n_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.dict_size = dict_size
        self.value_dim = n_hidden // n_heads

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        dev = self.device
        gen = self.generator

        # --- Encoder ---
        # Per-head encoder projections: (n_features, dict_size)
        self.encoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.n_features, self.dict_size, device=dev, generator=gen
                    )
                    / sqrt(self.n_features)
                )
                for _ in range(self.n_heads)
            ]
        )
        # Per-head encoder value matrices: (dict_size, value_dim)
        self.encoder_values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size, self.value_dim, device=dev, generator=gen
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.W_mix = nn.Parameter(torch.zeros(self.n_hidden, self.n_hidden, device=dev))

        self.W_skip = nn.Parameter(
            torch.randn(
                self.n_features,
                self.n_hidden,
                device=dev,
                generator=gen,
            )
            / sqrt(self.n_hidden)
        )
        # --- Decoder ---
        # decoder_projs are tied to encoder_values (transposed)
        # Per-head decoder value matrices: (dict_size, n_features)
        self.decoder_values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size, self.n_features, device=dev, generator=gen
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.b = nn.Parameter(torch.zeros(self.n_features, device=dev))
        self.alpha = nn.Parameter(torch.tensor([0.1], device=dev))

    def encode(self, x: Tensor) -> Tensor:
        parts = []
        for h in range(self.n_heads):
            logits = x @ self.encoder_projs[h]  # (B, dict_size)
            weights = softmax1(logits, dim=-1)  # (B, dict_size)
            parts.append(weights @ self.encoder_values[h])  # (B, value_dim)
        z = torch.cat(parts, dim=-1)
        return z + z @ self.W_mix + x @ self.W_skip

    def decode(self, z: Tensor) -> Tensor:
        chunks = z.split(self.value_dim, dim=-1)  # n_heads x (B, value_dim)
        out = torch.zeros(z.shape[0], self.n_features, device=z.device)
        for h in range(self.n_heads):
            logits = chunks[h] @ self.encoder_values[h].T  # (B, dict_size)
            weights = F.softmax(logits, dim=-1)  # (B, dict_size)
            out = out + weights @ self.decoder_values[h]  # (B, n_features)
        return torch.relu(out + self.b)


class LinearAttnAE(AutoEncoderBase):
    """Linear encoder with multi-head softmax attention decoder.

    Encoder: ``x @ W_enc.T`` (standard linear projection).
    Decoder: ``softmax(z_h @ Q_h) @ U_h`` per head, summed + bias + ReLU.

    This is the complement of ``AttnLinearAE`` which uses attention for
    encoding and a linear projection for decoding.

    Parameters
    ----------
    n_features : int
        Input / output dimensionality.
    n_hidden : int
        Total latent dimensionality (must be divisible by ``n_heads``).
    n_heads : int
        Number of independent softmax heads in the decoder.
    dict_size : int
        Number of dictionary elements (archetypes) per decoder head.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        dict_size: int,
        **kwargs,
    ) -> None:
        super().__init__(n_features, n_hidden, **kwargs)

        if n_hidden % n_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.dict_size = dict_size
        self.value_dim = n_hidden // n_heads

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        dev = self.device
        gen = self.generator

        # --- Encoder: linear projection ---
        self.W_enc = nn.Parameter(
            torch.randn(self.n_hidden, self.n_features, device=dev, generator=gen)
            / sqrt(self.n_features)
        )

        # --- Decoder: multi-head softmax attention ---
        # Per-head decoder projections: (value_dim, dict_size)
        self.decoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.value_dim, self.dict_size, device=dev, generator=gen
                    )
                    / sqrt(self.value_dim)
                )
                for _ in range(self.n_heads)
            ]
        )
        # Per-head decoder value matrices: (dict_size, n_features)
        self.decoder_values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size, self.n_features, device=dev, generator=gen
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.b = nn.Parameter(torch.zeros(self.n_features, device=dev))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W_enc.T

    def decode(self, z: Tensor) -> Tensor:
        chunks = z.split(self.value_dim, dim=-1)  # n_heads x (B, value_dim)
        out = torch.zeros(z.shape[0], self.n_features, device=z.device)
        for h in range(self.n_heads):
            logits = chunks[h] @ self.decoder_projs[h]  # (B, dict_size)
            weights = F.softmax(logits, dim=-1)  # (B, dict_size)
            out = out + weights @ self.decoder_values[h]  # (B, n_features)
        return torch.relu(out + self.b)
