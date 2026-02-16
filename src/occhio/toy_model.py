from functools import cached_property
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .autoencoder import AutoEncoderBase
from .distributions.base import Distribution


class ToyModel:
    def __init__(
        self,
        distribution: Optional[Distribution],
        ae: Optional[AutoEncoderBase],
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        importances=None,
    ):

        self.distribution = distribution
        self.ae = ae

        assert distribution.n_features == ae.n_features  # ty:ignore
        self.n_features: int = ae.n_features  # ty:ignore

        if importances is None:
            self.importances = torch.ones(self.n_features)
        else:
            self.importances = importances

    def fit(
        self,
        n_epochs: int,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        track_losses: bool = True,
        optimizer: Optimizer | None = None,
        verbose: bool = False,
    ) -> list[float]:
        if optimizer is None:
            optimizer = AdamW(
                self.ae.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        losses = []

        for ep in range(n_epochs):
            x = self.distribution.sample(batch_size)
            optimizer.zero_grad()
            x_hat = self.ae.forward(x)[0]  # Only take x_hat
            loss = self.loss(x, x_hat, self.importances)
            loss.backward()
            optimizer.step()

            if track_losses:
                losses.append(loss.item())
            if verbose and (ep + 1) % 1000 == 0:
                print(f"AE Epoch {ep + 1}/{n_epochs}, Loss: {loss.item():.6f}")

        return losses

    def sample_latent(self, batch_size) -> Tensor:
        inputs = self.distribution.sample(batch_size)
        return self.ae.encode(inputs)

    def get_one_hot_embeddings(self) -> Tensor:
        return self.ae.encode(torch.eye(self.n_features))

    def __repr__(self):
        return f"ToyModel({self.distribution})"

    def __getattr__(self, name):
        if name in ("sample", "n_features"):
            return getattr(self.distribution, name)

        if name in ("encode", "decode", "forward", "resample_weights", "loss"):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @cached_property
    @torch.no_grad()
    def froebenius_norm_squared(self):
        return torch.linalg.norm(self.W, ord="fro") ** 2

    @cached_property
    @torch.no_grad()
    def hidden_dimensions_per_embedded_features(self) -> Any:
        return self.ae.n_hidden / self.froebenius_norm_squared

    @cached_property
    @torch.no_grad()
    def embedded_features_per_hidden_dimensions(self) -> Any:
        return self.froebenius_norm_squared / self.ae.n_hidden

    @cached_property
    @torch.no_grad()
    def feature_dimensionalities(self):
        return (
            self.feature_representations
            / self.total_feature_interferences_including_self
        )

    @cached_property
    @torch.no_grad()
    def mean_feature_dimensionalities(self):
        return self.feature_dimensionalities.mean()

    @cached_property
    @torch.no_grad()
    def total_feature_dimensionalities_per_hidden_dimension(self):
        return self.feature_dimensionalities.sum() / self.ae.n_hidden

    @cached_property
    @torch.no_grad()
    def W(self) -> Tensor:
        return self.get_one_hot_embeddings().T

    @cached_property
    @torch.no_grad()
    def W_normalized_features(self) -> Tensor:
        return F.normalize(self.W, dim=0)

    @cached_property
    @torch.no_grad()
    def feature_norms(self) -> Tensor:
        return torch.linalg.vector_norm(self.W, dim=0)

    @cached_property
    @torch.no_grad()
    def feature_representations(self) -> Tensor:
        return (self.W**2).sum(dim=0)

    @cached_property
    @torch.no_grad()
    def interferences(self) -> Tensor:
        return (self.W_normalized_features.T @ self.W) ** 2

    @cached_property
    @torch.no_grad()
    def total_feature_interferences(self) -> Tensor:
        interferences = self.interferences.clone()
        return interferences.fill_diagonal_(0).sum(dim=1)

    @cached_property
    @torch.no_grad()
    def total_feature_interferences_including_self(self) -> Tensor:
        return self.interferences.sum(dim=1)
