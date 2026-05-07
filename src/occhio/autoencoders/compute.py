"""Autoencoder with a linear compute step."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.device import seeded_generator
from .base import AutoEncoderBase


class ComputeAutoEncoder(AutoEncoderBase):
    """
    Autoencoder with a tied encoder/decoder and a linear compute step.

    Subclasses occhio's AutoEncoderBase so it exposes encode/decode and slots
    into ToyModel for geometric analysis (feature norms, interferences, etc.).

    Parameters
    ----------
    N : int   — number of features
    k : int   — hidden / latent dimension
    decode_activation : "softmax" | "relu"
        "softmax" — outputs a probability simplex; use for one-hot targets (CE/MSE).
        "relu"    — outputs non-negative values; use for continuous targets like x @ P.
    seed : int — weight init seed

    Weights
    -------
    W : (k, N) — tied encoder / decoder
    Z : (k, k) — linear compute step
    b : (N,)   — decode bias
    """

    def __init__(
        self,
        N: int,
        k: int,
        decode_activation: Literal["softmax", "relu"] = "softmax",
        seed: int = 10,
        **kwargs,
    ):
        super().__init__(N, k, **kwargs)
        self.decode_activation = decode_activation

        dev = self.device
        gen = seeded_generator(seed, dev)
        self.W = nn.Parameter(torch.randn(k, N, generator=gen, device=dev) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen, device=dev) / k)
        self.b = nn.Parameter(torch.zeros(N, device=dev))

    # ── core operations ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, k)  : embed into latent space."""
        return x @ self.W.T

    def compute_step(self, h: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, k)  : linear compute / routing step."""
        return h + h @ self.Z.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, N)  : project back, then activate."""
        logits = z @ self.W + self.b
        if self.decode_activation == "softmax":
            return F.softmax(logits, dim=-1)
        return F.relu(logits)

    def forward(self, x: torch.Tensor):
        """(B, N) → (y_hat, z)."""
        h = self.encode(x)
        z = self.compute_step(h)
        y_hat = self.decode(z)
        return y_hat, z

    def ce_loss(
        self, y_hat: torch.Tensor, y_idx: torch.Tensor, importances: torch.Tensor
    ) -> torch.Tensor:
        """Importance-weighted NLL given softmax output probabilities."""
        per_sample = F.nll_loss(y_hat.clamp(min=1e-9).log(), y_idx, reduction="none")
        weights = importances[y_idx]
        return (per_sample * weights).mean()

    def mse_loss(
        self, y_hat: torch.Tensor, y_true: torch.Tensor, importances: Tensor | None
    ) -> torch.Tensor:
        """Importance-weighted MSE. Prediction first, target second (mirrors ce_loss)."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        per_sample = (y_true - y_hat).pow(2).sum(dim=-1)
        weights = importances[y_hat.argmax(dim=-1)]
        return (per_sample * weights).mean()

    def loss(
        self, x_true: Tensor, x_hat: Tensor, importances: Tensor | None
    ) -> torch.Tensor:
        """Importance-weighted MSE between predicted probs and one-hot target."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        per_sample = (x_true - x_hat).pow(2).sum(dim=-1)
        weights = importances[x_hat.argmax(dim=-1)]
        return (per_sample * weights).mean()

    def get_config(self) -> dict:
        return {
            "N": self.n_features,
            "k": self.n_hidden,
            "decode_activation": self.decode_activation,
        }

    def resample_weights(self):
        dev = self.device
        gen = self.generator or torch.Generator(device=dev or "cpu")
        N, k = self.n_features, self.n_hidden
        self.W = nn.Parameter(torch.randn(k, N, generator=gen, device=dev) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen, device=dev) / k)
        self.b = nn.Parameter(torch.zeros(N, device=dev))
