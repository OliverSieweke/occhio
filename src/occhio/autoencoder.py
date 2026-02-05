# ABOUTME: Defines the AutoEncoder abstraction used across occhio.
# ABOUTME: Provides tied linear layers plus encode/decode utilities.

from __future__ import annotations
from math import sqrt
from torch import Tensor
import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.W = nn.Parameter(torch.randn(n_hidden, n_features) / sqrt(n_hidden))
        self.b = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W + self.b)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def get_feature_norms(self):
        return torch.linalg.norm(self.W, dim=0)

    # @property
    # def W(self) -> Dict[str, List[nn.Parameter]]:
    #     return {
    #         "encoder": list(self.encoder.parameters()),
    #         "decoder": list(self.decoder.parameters()),
    #     }
