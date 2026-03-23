"""
Causal SAE: learns an upper-triangular causal matrix between dictionary features.
"""

from torch import Tensor
import torch
import torch.nn as nn

from .base import SparseAutoEncoderBase


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

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict), device=self.device))
        self.b_enc = nn.Parameter(torch.zeros(n_dict, device=self.device))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent), device=self.device))
        self.causal = nn.Parameter(torch.zeros((n_dict, n_dict), device=self.device))

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
