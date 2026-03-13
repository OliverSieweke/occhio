"""SparseSpheres: k independent features on S^n, axis-aligned in R^{k*m}."""

import warnings

import torch
from torch import Tensor

from .base import Distribution


class SparseSpheres(Distribution):
    """Distribution of k independent features on S^n, concatenated axis-aligned.

    Each feature lives on the n-dimensional unit sphere S^n, embedded in an
    ambient space R^m where m >= n+1. When m = n+1, the sphere sits naturally
    (no tilt). When m > n+1, a random orthonormal tilt matrix embeds R^{n+1}
    into R^m.

    The total output dimensionality is n_features = k * m. Each feature occupies
    its own axis-aligned m-dimensional slice. Superposition pressure comes from
    the autoencoder bottleneck, not from overlapping embeddings.

    Args:
        k: Number of independent ring features.
        n: Intrinsic dimension of each feature's manifold S^n. Default 1 (circles).
        m: Ambient dimension per feature. Must satisfy m >= n+1. Defaults to n+1.
        p_active: Per-feature activation probability. Scalar, list, or tensor
            of length k.
        r: Fixed radius for all features. Default 1.0.
        noise_std: Standard deviation of elementwise Gaussian noise added only to
            active features. Default 0.0 (no noise).
        **kwargs: Passed to Distribution (device, generator).
    """

    def __init__(
        self,
        *,
        k: int,
        n: int = 1,
        m: int | None = None,
        p_active: float | list[float] | Tensor,
        r: float = 1.0,
        noise_std: float = 0.0,
        n_features: int | None = None,
        **kwargs,
    ):
        if m is None:
            m = n + 1
        if m < n + 1:
            raise ValueError(
                f"m ({m}) must be >= n + 1 ({n + 1}). "
                f"The ambient dimension must be at least n+1 to embed S^n."
            )

        computed_n_features = k * m
        if n_features is not None and n_features != computed_n_features:
            warnings.warn(
                f"n_features={n_features} does not match k * m = {computed_n_features}. "
                f"Overwriting n_features with {computed_n_features}.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(computed_n_features, **kwargs)
        self.k = k
        self.n = n
        self.m = m
        self.r = r
        self.noise_std = noise_std

        # Broadcast p_active to (k,) — can't use _broadcast since that targets n_features
        if isinstance(p_active, Tensor):
            if p_active.dim() == 0:
                self.p_active = p_active.expand(k).clone().to(self.device)
            else:
                self.p_active = p_active.to(self.device)
        elif isinstance(p_active, (int, float)):
            self.p_active = torch.full((k,), p_active, device=self.device)
        else:
            self.p_active = torch.as_tensor(p_active, device=self.device)

        self.tilts = self._build_tilts()

        # Per-feature centers that shift each sphere into the non-negative orthant.
        # For coordinate d of feature i, the tilted point ranges in
        # [-r * ||R_i[d,:]||, +r * ||R_i[d,:]||]. Shifting by r * ||R_i[d,:]||
        # maps this to [0, 2r * ||R_i[d,:]||].
        self.centers = self.r * self.tilts.norm(dim=-1)  # (k, m)

    def _build_tilts(self) -> Tensor:
        """Build (k, m, n+1) tilt matrices.

        Identity when m = n+1, random orthonormal via QR when m > n+1.
        """
        n_plus_1 = self.n + 1
        if self.m == n_plus_1:
            return (
                torch.eye(self.m, device=self.device)
                .unsqueeze(0)
                .expand(self.k, -1, -1)
                .clone()
            )
        else:
            tilts = torch.zeros(self.k, self.m, n_plus_1, device=self.device)
            for i in range(self.k):
                A = self._randn(self.m, n_plus_1)
                Q, R = torch.linalg.qr(A)
                # Fix sign ambiguity for deterministic QR
                Q = Q * torch.sign(torch.diag(R))
                tilts[i] = Q
            return tilts

    def _sample_core(
        self, batch_size: int, noise_std: float | None = None
    ) -> tuple[Tensor, Tensor]:
        """Shared sampling logic. Returns (output, mask).

        Args:
            batch_size: Number of samples.
            noise_std: Override for Gaussian noise std. None falls back to self.noise_std.
        """
        k = self.k
        n_plus_1 = self.n + 1
        effective_noise = self.noise_std if noise_std is None else noise_std

        # 1. Activation mask: (batch_size, k)
        mask = self._rand(batch_size, k) < self.p_active

        # 2. Sample uniformly on S^n: draw from N(0,1) and normalize
        z = self._randn(batch_size, k, n_plus_1)
        z = z / z.norm(dim=-1, keepdim=True)
        z = self.r * z

        # 3. Apply tilt: (batch, k, n+1) @ (k, n+1, m) -> (batch, k, m)
        tilted = torch.einsum("bkj,kjm->bkm", z, self.tilts.transpose(-1, -2))

        # 4. Shift into non-negative orthant (center + sphere point)
        tilted = tilted + self.centers

        # 5. Zero out inactive features (center included)
        tilted = tilted * mask.unsqueeze(-1)

        # 6. Add Gaussian noise to active features only
        if effective_noise > 0:
            noise = self._randn(batch_size, k, self.m) * effective_noise
            tilted = tilted + noise * mask.unsqueeze(-1)

        # 7. Reshape to (batch_size, k * m)
        output = tilted.reshape(batch_size, k * self.m)
        return output, mask

    def sample(self, batch_size: int, noise_std: float | None = None):
        """Sample from the distribution.

        Args:
            batch_size: Number of samples.
            noise_std: Override for Gaussian noise std. None falls back to self.noise_std.
        """
        output, _ = self._sample_core(batch_size, noise_std=noise_std)
        return output

    def sample_with_args(
        self,
        batch_size: int,
        with_labels: bool = True,
        noise_std: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Like sample(), but optionally returns the (batch_size, k) boolean activation mask.

        Args:
            batch_size: Number of samples.
            with_labels: If True, return (output, mask) tuple. If False, return output only.
            noise_std: Override for Gaussian noise std. None falls back to self.noise_std.
        """
        output, mask = self._sample_core(batch_size, noise_std=noise_std)
        if with_labels:
            return output, mask
        return output
