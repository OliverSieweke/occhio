import torch
from torch import Tensor
from torch.optim import AdamW
from .distributions.base import Distribution
from .distributions.sparse import SparseUniform
from .autoencoder import AutoEncoderBase, TiedLinear
from typing import Optional


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

    # def loss_func(self, x_true, x_hat):
    #     return torch.mean(
    #         torch.sum(self.importances * torch.square(x_true - x_hat), dim=-1)
    #     )

    def fit(
        self,
        n_epochs: int,
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=0.05,
        track_losses=True,
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
            loss = self.loss(x, x_hat, self.importances)
            loss.backward()
            optimizer.step()

            if track_losses:
                losses.append(loss)
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

        if name in ("encode", "decode", "forward", "W", "resample_weights", "loss"):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
