import torch
from torch import Tensor
from torch.optim import AdamW
from .distributions.base import Distribution
from .distributions.sparse import SparseUniform
from .autoencoder import AutoEncoder
from typing import Optional


class ToyModel:
    def __init__(
        self,
        distribution: Optional[Distribution] = None,
        ae: Optional[AutoEncoder] = None,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        importances=None,
    ):

        if distribution is None:
            self.distribution = SparseUniform(10, 0.1)
        else:
            self.distribution = distribution

        if ae is None:
            self.ae = AutoEncoder(10, 3)
        else:
            self.ae = ae

        assert distribution.n_features == ae.n_features  # ty:ignore

        self.importances = torch.ones(distribution.n_features)  # ty:ignore

    def loss_func(self, x_true, x_hat):
        return torch.mean(
            torch.sum(self.importances * torch.square(x_true - x_hat), dim=-1)
        )

    def fit(
        self,
        n_epochs: int,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=0.01,
        track_losses=True,
        force_norm=False,
        verbose=False,
    ) -> list[float]:
        optimizer = AdamW(
            self.ae.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        losses = []

        for ep in range(n_epochs):
            x = self.distribution.sample(batch_size)
            optimizer.zero_grad()
            x_hat = self.ae.forward(x)[0]  # Only take x_hat
            loss = self.loss_func(x, x_hat)
            loss.backward()
            optimizer.step()

            if force_norm:
                with torch.no_grad():
                    self.ae.W.data = self.ae.W.data / self.ae.get_feature_norms()
            if track_losses:
                losses.append(loss)
            if verbose and (ep + 1) % 1000 == 0:
                print(f"AE Epoch {ep + 1}/{n_epochs}, Loss: {loss.item():.6f}")

        return losses

    def sample_latent(self, batch_size) -> Tensor:
        inputs = self.distribution.sample(batch_size)
        return self.ae.encode(inputs)

    def __repr__(self):
        return f"ToyModel({self.distribution})"

    def __getattr__(self, name):
        if name in ("sample", "n_features"):
            return getattr(self.distribution, name)

        if name in ("encode", "decode", "forward", "W"):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
