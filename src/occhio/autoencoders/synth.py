"""Synthetic autoencoder with orthogonalization."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import AutoEncoderBase


class SynthAE(AutoEncoderBase):
    """Autoencoder with unit-norm tied weights and optional orthogonalization.

    Feature vectors (columns of W) are initialized as random unit vectors
    sampled from a standard normal distribution:

        d_i = g_i / ||g_i||_2,  g_i ~ N(0, I_D)

    When ``orthogonalize=True``, a gradient descent procedure minimizes
    pairwise cosine similarity between columns:

        L_ortho = Σ_{i≠j} (d_i^T d_j)^2 + λ Σ_i (||d_i||_2 - 1)^2

    After orthogonalization, all vectors are renormalized to unit length.
    Uses a chunked implementation to reduce memory from O(N^2) to
    O(chunk_size × N).

    Parameters
    ----------
    n_features : int
        Number of ground-truth features N.
    n_hidden : int
        Hidden dimension D (must satisfy N ≥ D for meaningful superposition).
    orthogonalize : bool
        Run the orthogonalization procedure on init.
    ortho_lambda : float
        Norm penalty weight λ in L_ortho.
    ortho_steps : int
        Number of gradient descent iterations.
    ortho_lr : float
        Learning rate for the orthogonalization optimizer.
    ortho_chunk_size : int
        Block size for chunked pairwise dot products.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        orthogonalize: bool = False,
        ortho_lambda: float = 1.0,
        ortho_steps: int = 1000,
        ortho_lr: float = 0.01,
        ortho_chunk_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(n_features, n_hidden, **kwargs)

        self._orthogonalize = orthogonalize
        self._ortho_lambda = ortho_lambda
        self._ortho_steps = ortho_steps
        self._ortho_lr = ortho_lr
        self._ortho_chunk_size = ortho_chunk_size

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        # W is (n_hidden, n_features) — each column is a unit-norm feature direction in R^D
        g = torch.randn(
            self.n_hidden,
            self.n_features,
            generator=self.generator,
            device=self.device,
        )
        # Normalize columns to unit length
        g = g / g.norm(dim=0, keepdim=True)

        if self._orthogonalize:
            g = self._run_orthogonalization(g)

        self.W = nn.Parameter(g)
        self.b = nn.Parameter(torch.zeros(self.n_features, device=self.device))
        self.freeze_W()

    def _run_orthogonalization(self, W: Tensor) -> Tensor:
        """Minimize pairwise cosine similarity via gradient descent.

        Operates on W of shape (D, N) where columns are feature directions.
        """
        D = W.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([D], lr=self._ortho_lr)
        N = D.shape[1]
        chunk = self._ortho_chunk_size

        for _ in range(self._ortho_steps):
            optimizer.zero_grad()

            # Chunked pairwise dot products between columns
            # D^T D gives (N, N) cosine similarities (columns are unit-norm)
            ortho_loss = torch.tensor(0.0, device=D.device, dtype=D.dtype)
            for i in range(0, N, chunk):
                cols_i = D[:, i : i + chunk]  # (D, chunk)
                dots = cols_i.T @ D  # (chunk, N)
                # Zero self-similarities
                end = min(i + chunk, N)
                for k in range(end - i):
                    dots[k, i + k] = 0.0
                ortho_loss = ortho_loss + (dots**2).sum()

            # Norm penalty on columns
            col_norms = D.norm(dim=0)
            norm_loss = self._ortho_lambda * ((col_norms - 1.0) ** 2).sum()

            loss = ortho_loss + norm_loss
            loss.backward()
            optimizer.step()

        # Re-normalize columns to unit length
        with torch.no_grad():
            D.div_(D.norm(dim=0, keepdim=True))
        return D.detach()

    def freeze_W(self):
        """Freeze W so only the bias b is updated during training."""
        self.W.requires_grad_(False)

    @property
    def rho_mm(self) -> float:
        """Mean max absolute cosine similarity (superposition metric).

        ρ_mm = (1/N) Σ_i max_{j≠i} |d_i^T d_j|

        Returns 0 for fully orthogonal features, approaches 1 for full superposition.
        """
        W = self.W.detach()
        N = W.shape[1]
        max_cos = torch.zeros(N, device=W.device, dtype=W.dtype)
        chunk = min(self._ortho_chunk_size, N)
        for i in range(0, N, chunk):
            sims = (W[:, i : i + chunk].T @ W).abs()  # (chunk, N)
            end = min(i + chunk, N)
            for k in range(end - i):
                sims[k, i + k] = 0.0
            max_cos[i : i + chunk] = sims.max(dim=1).values
        return max_cos.mean().item()

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W + self.b)
