import torch
from torch.optim import AdamW
from ..distributions.base import Distribution
from ..distributions.sparse import SparseUniform
from ..autoencoders.autoencoder import AutoEncoder
from typing import Optional


class ToyModel:
    def __init__(
        self,
        distribution: Optional[Distribution] = None,
        ae: Optional[AutoEncoder] = None,
        device="cpu",
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

        assert distribution.n_features == ae.n_features

        self.importances = torch.ones(distribution.n_features)

    def loss_func(self, x_true, x_hat):
        return torch.mean(
            torch.sum(self.importances * torch.square(x_true - x_hat), dim=-1)
        )

    def fit(
        self,
        n_epochs: int,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.01,
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
            x_hat = self.ae.forward(x)
            loss = self.loss_func(x, x_hat)
            loss.backward()
            optimizer.step()
            if track_losses:
                losses.append(loss)
            if verbose and (ep + 1) % 1000 == 0:
                print(f"AE Epoch {ep + 1}/{ep}, Loss: {loss.item():.6f}")

        return losses

    def get_latent(self, batch_size):
        inputs = self.distribution.sample(batch_size)
        return self.ae.encode(inputs)
