"""
Matching Pursuit SAE: greedy iterative encoding without a learned encoder matrix.
"""

from torch import Tensor
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SparseAutoEncoderBase


class MatchingPursuitSAE(SparseAutoEncoderBase):
    """Matching Pursuit Sparse Autoencoder.

    Uses iterative greedy matching pursuit for encoding instead of a learned
    encoder matrix. A single tied dictionary ``W`` serves as both the analysis
    and synthesis basis. Sparsity is implicitly controlled by the pursuit
    stopping criteria (residual threshold and maximum iterations) rather than
    an L1 penalty.

    The algorithm (from "From Flat to Hierarchical: Extracting Sparse
    Representations with Matching Pursuit", Costa et al. 2025) works as
    follows at each forward pass:

    1. Initialize residual r = x (- b_pre if used).
    2. For each step t = 0 … T-1:
       a. Select atom j = argmax_j  relu(W @ r)  (greedy, no gradient).
       b. Compute coefficient c = relu(W_j · r)  (differentiable).
       c. Accumulate z_j += c.
       d. Update r -= c · W_j.
       e. Stop early when the support set stabilises or ||r|| < threshold.
    3. Decode: x_hat = z @ W (+ b_pre).

    Parameters
    ----------
    n_latent : int
        Input / reconstruction dimension (the space being decomposed).
    n_dict : int
        Number of dictionary atoms (overcomplete basis size).
    threshold : float
        Residual norm below which a sample is considered converged.
    max_iterations : int | None
        Maximum number of pursuit steps per forward pass. Defaults to
        ``n_latent`` when ``None``.
    pre_bias : bool
        If True, learn a bias that centres the input before pursuit and is
        added back after decoding.
    """

    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        threshold: float = 1e-3,
        max_iterations: int | None = None,
        pre_bias: bool = False,
        **kwargs,
    ):
        super().__init__(l1_coef=0.0, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict
        self.threshold = threshold
        self.max_iterations = max_iterations if max_iterations is not None else n_latent
        self._use_pre_bias = pre_bias

        self.W = nn.Parameter(torch.empty(n_dict, n_latent, device=self.device))
        if pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(n_latent, device=self.device))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W, generator=self.generator)
        with torch.no_grad():
            self.W.data = F.normalize(self.W.data, dim=1)
        if self._use_pre_bias:
            nn.init.zeros_(self.b_pre)

    def encode(self, x: Tensor) -> Tensor:
        """Run matching pursuit to produce sparse codes."""
        return self._matching_pursuit(x)

    def decode(self, z: Tensor) -> Tensor:
        """Linear decode: ``z @ W`` (+ ``b_pre`` if used)."""
        out = z @ self.W
        if self._use_pre_bias:
            out = out + self.b_pre
        return out

    def _matching_pursuit(self, x: Tensor) -> Tensor:
        """Greedy matching pursuit encoding.

        At each step, select the dictionary atom most aligned with the
        current residual, compute the projection coefficient, accumulate
        into the sparse code, and subtract from the residual. Stops when
        the support set stabilises or the residual norm drops below
        ``self.threshold``.
        """
        residual = x - self.b_pre if self._use_pre_bias else x

        z = x.new_zeros(x.shape[0], self.n_dict)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        prev_support = torch.zeros_like(z, dtype=torch.bool)

        for _ in range(self.max_iterations):
            if done.all():
                break

            # --- atom selection (discrete, no gradient needed) ---
            with torch.no_grad():
                projections = torch.relu(residual @ self.W.T)
                best_idx = projections.argmax(dim=1)  # (batch,)

            # --- coefficient computation (differentiable through W) ---
            selected = self.W[best_idx]  # (batch, n_latent)
            coeff = torch.relu((residual * selected).sum(dim=-1))  # (batch,)

            # --- sparse update for this step ---
            z_step = torch.zeros_like(z)
            z_step.scatter_(1, best_idx.unsqueeze(1), coeff.unsqueeze(1))

            # --- accumulate, masking converged samples ---
            z = torch.where(done.unsqueeze(1), z, z + z_step)

            # --- update residual ---
            residual = torch.where(
                done.unsqueeze(1),
                residual,
                residual - coeff.unsqueeze(1) * selected,
            )

            # --- convergence check ---
            support = z != 0
            support_unchanged = (support == prev_support).all(dim=1)
            below_threshold = residual.norm(dim=1) < self.threshold
            done = done | support_unchanged | below_threshold
            prev_support = support

        return z

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        """Pure reconstruction MSE — sparsity is implicit in the pursuit."""
        return torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))

    def constrain_decoder_norms(self):
        """Project dictionary atoms back to unit norm."""
        with torch.no_grad():
            self.W.data = F.normalize(self.W.data, dim=1)

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
        sample_every: int = 25,
    ) -> list[float]:
        """Train the MP-SAE, projecting atoms to unit norm after each step."""
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        optimizer = AdamW(self.parameters(), lr=lr)
        sae_device = next(self.parameters()).device
        loss_buffer = torch.empty(n_steps, device=sae_device)

        raw_buffer: Tensor | None = None

        for step in range(n_steps):
            buf_offset = step % sample_every
            if buf_offset == 0:
                steps_left = min(sample_every, n_steps - step)
                total_samples = steps_left * batch_size
                raw_buffer = data_fn(total_samples).detach()

            assert raw_buffer is not None
            start = buf_offset * batch_size
            end = start + batch_size
            x = raw_buffer[start:end]

            optimizer.zero_grad()
            x_hat, z = self.forward(x)
            loss = self.loss(x, x_hat, z)
            loss.backward()
            optimizer.step()
            self.constrain_decoder_norms()

            loss_buffer[step] = loss.detach()
            if (step + 1) % 5000 == 0:
                print(f"  MP-SAE step {step + 1}/{n_steps}  loss={loss.item():.4f}")

        return loss_buffer.cpu().tolist()
