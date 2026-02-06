"""Implements simple"""

import functools
from math import sqrt
from torch import Tensor
import torch.nn as nn
import torch
from abc import ABC, abstractmethod


class AutoEncoderBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """features --> latent"""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """latent --> features"""

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
            importances = torch.ones(self.n_features)  # ty:ignore
        return torch.mean(torch.sum(importances * torch.square(x_true - x_hat), dim=-1))

    def __init__(
        self,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.device = device
        self.generator = generator

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


class TiedLinear(AutoEncoderBase):
    def __init__(self, n_features: int, n_hidden: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.resample_weights()

    # def loss_func(
    #     self, x_true: Tensor, x_hat: Tensor, importances: Tensor | None = None
    # ):
    #     if importances is None:
    #         importances = torch.ones(self.n_features)
    #     return torch.mean(torch.sum(importances * torch.square(x_true - x_hat), dim=-1))

    def resample_weights(self, force_norm=False):
        self.W = nn.Parameter(
            torch.randn(
                self.n_hidden,
                self.n_features,
                generator=self.generator,
                device=self.device,
            )
            / sqrt(self.n_hidden)
        )
        with torch.no_grad():
            norms = self.W.data.norm(dim=0, keepdim=True)
            self.W.data /= norms
        self.b = nn.Parameter(torch.zeros(self.n_features, device=self.device))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W + self.b)


class MLPEncoder(AutoEncoderBase):
    def __init__(self, embedding: list[int], unembedding: list[int], **kwargs):
        super().__init__(**kwargs)

        assert len(embedding) >= 2, "embedding must have at least [input, latent]"
        assert len(unembedding) >= 2, "unembedding must have at least [latent, output]"
        assert embedding[-1] == unembedding[0], "latent dims must match"
        assert embedding[0] == unembedding[-1], "input/output dims must match"

        self.n_features = embedding[0]
        self.n_hidden = embedding[-1]

        self.embedding_dims = embedding
        self.unembedding_dims = unembedding

        self._build_layers()

    def _build_layers(self):
        encoder_layers = []
        for i in range(len(self.embedding_dims) - 1):
            encoder_layers.append(
                nn.Linear(
                    self.embedding_dims[i],
                    self.embedding_dims[i + 1],
                    device=self.device,
                )
            )
            # No ReLU on output
            if i < len(self.embedding_dims) - 2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(self.unembedding_dims) - 1):
            decoder_layers.append(
                nn.Linear(
                    self.unembedding_dims[i],
                    self.unembedding_dims[i + 1],
                    device=self.device,
                )
            )
            if i < len(self.unembedding_dims) - 2:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.ReLU())  # ReLU on final output, matching TiedLinear
        self.decoder = nn.Sequential(*decoder_layers)

    def resample_weights(self, force_norm=False):
        self._build_layers()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
