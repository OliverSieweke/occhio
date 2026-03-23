"""
Top-K Ignore SAE: L1 sparsity penalty that ignores the top-k activations.
"""

from torch import Tensor
import torch
import torch.nn as nn

from .base import SparseAutoEncoderBase


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

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict), device=self.device))
        self.b_enc = nn.Parameter(torch.zeros(n_dict, device=self.device))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent), device=self.device))

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
