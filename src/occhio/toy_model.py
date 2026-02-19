from typing import Any, Optional, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .autoencoder import AutoEncoderBase
from .distributions.base import Distribution


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Compare two devices, treating a missing index as 0 (e.g. mps == mps:0)."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


class ToyModel:
    """This is the ToyModel class which is the base for most experiments.

    Args:
        distribution: A Distribution object.
        ae: An AutoEncoderBase object.
        device: Where the torch objects live.
        generator: For seeded experiments.
        importances: Weighing of the distribution.
    """

    distribution: Distribution
    ae: AutoEncoderBase

    def __init__(
        self,
        distribution: Distribution,
        ae: AutoEncoderBase,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        importances=None,
    ):

        self.distribution = distribution
        self.ae = ae

        assert distribution.n_features == ae.n_features
        self.n_features: int = ae.n_features

        if device is None:
            device = ae.device or torch.device("cpu")
        else:
            device = torch.device(device)
            if ae._init_device is not None and not _same_device(ae._init_device, device):
                raise ValueError(
                    f"AutoEncoder was explicitly created on {ae._init_device}, "
                    f"but ToyModel device is {device}. "
                    f"Either omit the device from the AutoEncoder or make them match."
                )
            if distribution._init_device is not None and not _same_device(distribution._init_device, device):
                raise ValueError(
                    f"Distribution was explicitly created on {distribution._init_device}, "
                    f"but ToyModel device is {device}. "
                    f"Either omit the device from the Distribution or make them match."
                )

        ae.to(device)
        distribution.to(device)
        self.device = device

        if importances is None:
            self.importances = torch.ones(self.n_features, device=ae.device)
        else:
            self.importances = importances.to(ae.device)

    # If you change the signature or implementation here, make sure you keep it
    # consistent with ModelGrid.fit()
    def fit(
        self,
        n_epochs: int,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        track_losses: bool = True,
        optimizer: Optimizer | None = None,
        hooks: list[Callable] = [],
        hook_freq: int = 1,
        verbose: bool = False,
    ) -> tuple[list[float], list]:
        if optimizer is None:
            optimizer = AdamW(
                self.ae.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        losses = []
        hook_returns = [[] for _ in hooks]

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
            if hooks and (ep % hook_freq == 0 or ep == n_epochs - 1):
                with torch.no_grad():
                    hook_data = dict(
                        tm=self, epoch=ep, loss=loss.item(), x=x, x_hat=x_hat
                    )
                    for i, h in enumerate(hooks):
                        hook_returns[i].append(h(hook_data))

        return losses, hook_returns

    def sample_latent(self, batch_size) -> Tensor:
        inputs = self.distribution.sample(batch_size)
        return self.ae.encode(inputs)

    def get_one_hot_embeddings(self) -> Tensor:
        return self.ae.encode(torch.eye(self.n_features, device=self.ae.device))

    def __repr__(self):
        return f"ToyModel({self.distribution})"

    def __getattr__(self, name):
        if name in ("sample", "n_features"):
            return getattr(self.distribution, name)

        if name in ("encode", "decode", "forward", "resample_weights", "loss"):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    @torch.no_grad()
    def froebenius_norm_squared(self):
        return torch.linalg.norm(self.W, ord="fro") ** 2

    @property
    @torch.no_grad()
    def hidden_dimensions_per_embedded_features(self) -> Any:
        return self.ae.n_hidden / self.froebenius_norm_squared

    @property
    @torch.no_grad()
    def embedded_features_per_hidden_dimensions(self) -> Any:
        return self.froebenius_norm_squared / self.ae.n_hidden

    @property
    @torch.no_grad()
    def feature_dimensionalities(self):
        return (
            self.feature_representations
            / self.total_feature_interferences_including_self
        )

    @property
    @torch.no_grad()
    def mean_feature_dimensionalities(self):
        return self.feature_dimensionalities.mean()

    @property
    @torch.no_grad()
    def total_feature_dimensionalities_per_hidden_dimension(self):
        return self.feature_dimensionalities.sum() / self.ae.n_hidden

    @property
    @torch.no_grad()
    def W(self) -> Tensor:
        return self.get_one_hot_embeddings().T

    @property
    @torch.no_grad()
    def W_normalized_features(self) -> Tensor:
        return F.normalize(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_norms(self) -> Tensor:
        return torch.linalg.vector_norm(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_representations(self) -> Tensor:
        return (self.W**2).sum(dim=0)

    @property
    @torch.no_grad()
    def interferences(self) -> Tensor:
        return (self.W_normalized_features.T @ self.W) ** 2

    @property
    @torch.no_grad()
    def total_feature_interferences(self) -> Tensor:
        interferences = self.interferences.clone()
        return interferences.fill_diagonal_(0).sum(dim=1)

    @property
    @torch.no_grad()
    def total_feature_interferences_including_self(self) -> Tensor:
        return self.interferences.sum(dim=1)
