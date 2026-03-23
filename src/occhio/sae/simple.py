"""
Simple (vanilla) Sparse AutoEncoder with L1 sparsity penalty.
"""

from torch import Tensor
import torch
import torch.nn as nn

from .base import SparseAutoEncoderBase


class SAESimple(SparseAutoEncoderBase):
    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        dec_bias: bool = False,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict
        self.dec_bias = dec_bias

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict), device=self.device))
        self.b_enc = nn.Parameter(torch.zeros(n_dict, device=self.device))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent), device=self.device))
        self.b_dec = nn.Parameter(torch.zeros(n_latent, device=self.device))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        if self.dec_bias:
            return z @ self.W_dec + self.b_dec
        else:
            return z @ self.W_dec

    def _resample_dead(self, data_fn, batch_size: int) -> int:
        """Reinitialize dead neurons from high-reconstruction-error examples.

        Both encoder and decoder get the unit-norm direction from a high-loss
        example. The encoder bias inherits the median of alive neurons' biases
        so resampled neurons start with appropriate selectivity.
        """
        with torch.no_grad():
            x = data_fn(batch_size * 4)
            z = self.encode(x)
            dead = (z > 0).float().mean(0) == 0
            if not dead.any():
                return 0

            x_hat = self.decode(z)
            losses = (x - x_hat).square().sum(-1)
            probs = losses / (losses.sum() + 1e-8)
            n_dead = int(dead.sum().item())
            # multinomial is unsupported on MPS — round-trip through CPU
            idx = torch.multinomial(probs.cpu(), n_dead, replacement=True).to(
                probs.device
            )

            alive = ~dead
            if alive.any():
                init_bias = self.b_enc.data[alive].median().item()
            else:
                init_bias = 0.0

            for i, di in enumerate(torch.where(dead)[0]):
                d = x[idx[i]]
                d_hat = d / (d.norm() + 1e-8)
                self.W_dec.data[di] = d_hat
                self.W_enc.data[:, di] = d_hat
                self.b_enc.data[di] = init_bias

            return n_dead
