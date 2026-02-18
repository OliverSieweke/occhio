"""
Implements Sparse AutoEncoders.
"""

from torch import Tensor
from torch.optim import AdamW
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SparseAutoEncoderBase(nn.Module, ABC):
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

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        """
        The standard SAE loss function
        """
        sparsity_loss = torch.mean(torch.sum(torch.abs(intermediate), dim=-1))
        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))
        return self.l1_coef * sparsity_loss + mse_loss

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
    ) -> list[float]:
        optimizer = AdamW(self.parameters(), lr=lr)
        losses = []
        for step in range(n_steps):
            x = data_fn(batch_size)
            optimizer.zero_grad()
            x_hat, z = self.forward(x)
            loss = self.loss(x, x_hat, z)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (step + 1) % 1000 == 0:
                print(f"  SAE step {step + 1}/{n_steps}  loss={loss.item():.4f}")
        return losses

    def __init__(
        self,
        l1_coef: float = 0.5,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.l1_coef = l1_coef
        self.device = device
        self.generator = generator


class SAESimple(SparseAutoEncoderBase):
    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec


class TopKIgnoreSAE(SparseAutoEncoderBase):
    """
    SAE where the L1 sparsity penalty ignores the top-k activations per sample.
    By not penalizing the strongest activations, this reduces the incentive for
    feature absorption (where one neuron absorbs multiple features to lower L1).
    """

    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        k: int = 2,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict
        self.k = k

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        abs_acts = torch.abs(intermediate)

        # Zero out the top-k activations so they don't contribute to L1
        _, topk_indices = torch.topk(abs_acts, self.k, dim=-1)
        mask = torch.ones_like(abs_acts)
        mask.scatter_(-1, topk_indices, 0.0)

        sparsity_loss = torch.mean(torch.sum(abs_acts * mask, dim=-1))
        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))
        return self.l1_coef * sparsity_loss + mse_loss


class CausalSAE(SparseAutoEncoderBase):
    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        l1_dirc: float = 0.01,
        l1_causal: float = 0.00,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)
        self.l1_causal = l1_causal
        self.l1_dirc = l1_dirc

        self.n_latent = n_latent
        self.n_dict = n_dict

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))
        self.causal = nn.Parameter(torch.zeros((n_dict, n_dict)))

        self.resample_weights()

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.causal)

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:

        residual = intermediate - intermediate @ torch.triu(self.causal, 1)
        sparsity_loss = torch.mean(
            torch.sum(
                torch.abs(residual),
                dim=-1,
            )
        )
        direct_sparsity = torch.mean(torch.sum(torch.abs(intermediate), dim=-1))

        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))

        causal_loss = torch.mean(torch.abs(self.causal))
        return (
            mse_loss
            + self.l1_coef * sparsity_loss
            + self.l1_dirc * direct_sparsity
            + self.l1_causal * causal_loss
        )
