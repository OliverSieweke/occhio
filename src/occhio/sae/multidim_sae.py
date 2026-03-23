"""Multidimensional Sparse AutoEncoder with subspace dictionary elements.

Each dictionary element is a D x p subspace matrix, so features natively
represent p-dimensional manifolds. The gate signal comes from the projection
norm itself (no separate gate encoder), with a fixed threshold theta.
"""

from torch import Tensor
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BinarySTE(torch.autograd.Function):
    """Straight-through estimator for binary thresholding.

    Forward: ``(x > 0).float()``
    Backward: identity (gradient passes through unchanged).
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class MultiDimSAE(nn.Module):
    """Multidimensional Sparse AutoEncoder.

    Replaces 1D direction vectors with D x p subspace matrices, so each
    dictionary element natively represents a p-dimensional feature manifold.
    Uses the projection norm as the gate signal (no separate gate encoder),
    with a fixed threshold ``theta``. Group and column sparsity penalties
    are applied only to gated (active) features.

    Parameters
    ----------
    n_input : int
        Input activation dimension (D).
    n_features : int
        Number of dictionary features (m).
    subspace_dim : int
        Maximum dimensionality per feature (p).
    lambda_group : float
        Group sparsity coefficient. L2 norm per feature encourages
        entire features to turn off.
    lambda_col : float
        Column sparsity coefficient. L1 norm within features encourages
        lower effective dimensionality.
    theta : float
        Fixed global threshold for norm-based gating. A feature activates
        when its projection norm exceeds theta.
    device : str
        Device for parameters.
    """

    def __init__(
        self,
        n_input: int,
        n_features: int,
        subspace_dim: int,
        lambda_group: float,
        lambda_col: float,
        theta: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_input = n_input
        self.n_features = n_features
        self.subspace_dim = subspace_dim
        self.lambda_group = lambda_group
        self.lambda_col = lambda_col
        self.theta = theta

        # Subspace matrices: (m, D, p) -- columns normalized to unit norm
        self.V = nn.Parameter(
            torch.empty(n_features, n_input, subspace_dim, device=device)
        )

        # Decoder bias (data centering)
        self.b_dec = nn.Parameter(torch.zeros(n_input, device=device))

        self._init_weights()

    def _init_weights(self):
        """Initialize V with xavier + column normalization."""
        nn.init.xavier_normal_(self.V)
        with torch.no_grad():
            self.V.data = F.normalize(self.V.data, dim=1)
        nn.init.zeros_(self.b_dec)

    def normalize_columns(self):
        """Project columns of V back to unit norm (along the D dimension)."""
        with torch.no_grad():
            self.V.data = F.normalize(self.V.data, dim=1)

    def _safe_norm(self, z: Tensor) -> Tensor:
        """NaN-safe L2 norm along the last dimension."""
        return (z.square().sum(-1) + 1e-8).sqrt()

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode into per-feature subspace projections and binary gate.

        The gate is derived from the projection norm: a feature activates
        when ``||V_i^T x_tilde||_2 > theta``.

        Parameters
        ----------
        x : Tensor, shape ``(batch, D)``

        Returns
        -------
        z : Tensor, shape ``(batch, m, p)``
            Subspace projections for each feature.
        gate : Tensor, shape ``(batch, m)``
            Binary gate (0 or 1).
        """
        x_tilde = x - self.b_dec  # (batch, D)

        # Subspace projection: z_i = V_i^T x_tilde
        z = torch.einsum("mdp,bd->bmp", self.V, x_tilde)  # (batch, m, p)

        # Gate from projection norm
        z_norm = self._safe_norm(z)  # (batch, m)
        gate = _BinarySTE.apply(z_norm - self.theta)

        return z, gate

    def decode(self, z: Tensor, gate: Tensor) -> Tensor:
        """Reconstruct from gated subspace projections.

        Parameters
        ----------
        z : Tensor, shape ``(batch, m, p)``
        gate : Tensor, shape ``(batch, m)``

        Returns
        -------
        x_hat : Tensor, shape ``(batch, D)``
        """
        gated_z = z * gate.unsqueeze(-1)  # (batch, m, p)
        x_hat = torch.einsum("mdp,bmp->bd", self.V, gated_z) + self.b_dec
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Full forward pass with loss computation.

        Sparsity penalties are applied only to gated (active) features.

        Returns
        -------
        x_hat : Tensor, shape ``(batch, D)``
        z : Tensor, shape ``(batch, m, p)``
        gate : Tensor, shape ``(batch, m)``
        loss_dict : dict
            Keys: ``"reconstruction"``, ``"group_sparsity"``,
            ``"column_sparsity"``, ``"total"``.
        """
        z, gate = self.encode(x)
        x_hat = self.decode(z, gate)

        # Reconstruction: mean-over-batch of sum-over-D of (x - x_hat)^2
        reconstruction = torch.mean(torch.sum(torch.square(x - x_hat), dim=-1))

        # Group sparsity: only on gated features
        z_l2 = self._safe_norm(z)  # (batch, m)
        gated_z_l2 = z_l2 * gate  # zero for inactive
        group_sparsity = self.lambda_group * torch.mean(torch.sum(gated_z_l2, dim=-1))

        # Column sparsity: only on gated features
        z_l1 = torch.sum(torch.abs(z), dim=-1)  # (batch, m)
        gated_z_l1 = z_l1 * gate
        column_sparsity = self.lambda_col * torch.mean(torch.sum(gated_z_l1, dim=-1))

        total = reconstruction + group_sparsity + column_sparsity

        loss_dict = {
            "reconstruction": reconstruction,
            "group_sparsity": group_sparsity,
            "column_sparsity": column_sparsity,
            "total": total,
        }
        return x_hat, z, gate, loss_dict

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
        sample_every: int = 25,
    ) -> list[float]:
        """Train the MultiDimSAE with column normalization after each step.

        Parameters
        ----------
        data_fn : callable
            ``data_fn(n)`` returns a Tensor of shape ``(n, D)``.
        n_steps : int
            Number of training steps.
        batch_size : int
            Samples per step.
        lr : float
            Adam learning rate.
        sample_every : int
            Re-sample data buffer every this many steps.

        Returns
        -------
        list[float]
            Total loss at each step.
        """
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        optimizer = Adam(self.parameters(), lr=lr)
        sae_device = next(self.parameters()).device
        loss_buffer = torch.empty(n_steps, device=sae_device)

        raw_buffer: Tensor | None = None

        for step in range(n_steps):
            buf_offset = step % sample_every
            if buf_offset == 0:
                steps_left = min(sample_every, n_steps - step)
                total_samples = steps_left * batch_size
                raw_buffer = data_fn(total_samples).detach().to(sae_device)

            assert raw_buffer is not None
            start = buf_offset * batch_size
            end = start + batch_size
            x = raw_buffer[start:end]

            optimizer.zero_grad()
            x_hat, z, gate, loss_dict = self.forward(x)
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()
            self.normalize_columns()

            loss_buffer[step] = loss.detach()
            if (step + 1) % 5000 == 0:
                l0 = gate.sum(dim=-1).mean().item()
                print(
                    f"  MultiDimSAE step {step + 1}/{n_steps}"
                    f"  loss={loss.item():.4f}"
                    f"  recon={loss_dict['reconstruction'].item():.4f}"
                    f"  L0={l0:.1f}"
                )

        return loss_buffer.cpu().tolist()

    # ---- Diagnostics ----

    @torch.no_grad()
    def effective_l0(self, x: Tensor) -> float:
        """Average number of active features (gate > 0) per sample."""
        _, gate = self.encode(x)
        return gate.sum(dim=-1).mean().item()

    @torch.no_grad()
    def effective_dim_per_feature(self, x: Tensor, threshold: float = 0.01) -> Tensor:
        """Average effective dimensionality per feature across a batch.

        A component of ``z_i`` is "significant" if
        ``|z_i_j| > threshold * ||z_i||_2``.

        Returns
        -------
        Tensor, shape ``(m,)``
        """
        z, gate = self.encode(x)
        z_norms = self._safe_norm(z).unsqueeze(-1)  # (batch, m, 1)
        relative = torch.abs(z) / z_norms  # (batch, m, p)
        significant = (relative > threshold).float()  # (batch, m, p)
        dims_per_sample = significant.sum(dim=-1)  # (batch, m)
        active_count = gate.sum(dim=0).clamp(min=1)  # (m,)
        total_dims = (dims_per_sample * gate).sum(dim=0)  # (m,)
        return total_dims / active_count

    @torch.no_grad()
    def dead_features(self, x: Tensor) -> int:
        """Number of features that never activate over the given batch."""
        _, gate = self.encode(x)
        ever_active = gate.sum(dim=0) > 0
        return int((~ever_active).sum().item())
