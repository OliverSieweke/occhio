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

    The total output dimensionality is n_features = n_spheres * ambient_dim.
    Each feature occupies its own axis-aligned ambient_dim-dimensional slice.
    Superposition pressure comes from the autoencoder bottleneck, not from
    overlapping embeddings.

    Args:
        n_spheres: Number of independent sphere features.
        sphere_dim: Intrinsic dimension of each feature's manifold S^n.
            Default 1 (circles).
        ambient_dim: Ambient dimension per feature. Must satisfy
            ambient_dim >= sphere_dim + 1. Defaults to sphere_dim + 1.
        p_active: Per-feature activation probability. Scalar, list, or tensor
            of length n_spheres.
        p_infill: Probability that an active feature samples uniformly inside
            the n-ball instead of on its surface. Default 0.2. Set to 0 for
            surface-only sampling.
        radius: Fixed radius for all features. Default 1.0.
        noise_std: Standard deviation of elementwise Gaussian noise added only
            to active features. Default 0.0 (no noise).
        **kwargs: Passed to Distribution (device, generator).
    """

    def __init__(
        self,
        *,
        n_spheres: int,
        sphere_dim: int = 1,
        ambient_dim: int | None = None,
        p_active: float | list[float] | Tensor,
        p_infill: float = 0.0,
        radius: float = 1.0,
        noise_std: float = 0.0,
        n_features: int | None = None,
        **kwargs,
    ):
        if ambient_dim is None:
            ambient_dim = sphere_dim + 1
        if ambient_dim < sphere_dim + 1:
            raise ValueError(
                f"ambient_dim ({ambient_dim}) must be >= sphere_dim + 1 "
                f"({sphere_dim + 1}). The ambient dimension must be at least "
                f"sphere_dim+1 to embed S^sphere_dim."
            )

        computed_n_features = n_spheres * ambient_dim
        if n_features is not None and n_features != computed_n_features:
            warnings.warn(
                f"n_features={n_features} does not match "
                f"n_spheres * ambient_dim = {computed_n_features}. "
                f"Overwriting n_features with {computed_n_features}.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(computed_n_features, **kwargs)
        self.n_spheres = n_spheres
        self.sphere_dim = sphere_dim
        self.ambient_dim = ambient_dim
        self.radius = radius
        self.p_infill = p_infill
        self.noise_std = noise_std

        # Broadcast p_active to (n_spheres,)
        if isinstance(p_active, Tensor):
            if p_active.dim() == 0:
                self.p_active = p_active.expand(n_spheres).clone().to(self.device)
            else:
                self.p_active = p_active.to(self.device)
        elif isinstance(p_active, (int, float)):
            self.p_active = torch.full((n_spheres,), p_active, device=self.device)
        else:
            self.p_active = torch.as_tensor(p_active, device=self.device)

        self.tilts = self._build_tilts()

        # Per-feature centers that shift each sphere into the non-negative orthant.
        # For coordinate d of feature i, the tilted point ranges in
        # [-radius * ||R_i[d,:]||, +radius * ||R_i[d,:]||].
        # Shifting by radius * ||R_i[d,:]|| maps to [0, 2*radius * ||R_i[d,:]||].
        self.centers = self.radius * self.tilts.norm(dim=-1)  # (n_spheres, ambient_dim)

    def _build_tilts(self) -> Tensor:
        """Build (n_spheres, ambient_dim, sphere_dim+1) tilt matrices.

        Identity when ambient_dim = sphere_dim+1, random orthonormal via QR
        when ambient_dim > sphere_dim+1.
        """
        n_plus_1 = self.sphere_dim + 1
        if self.ambient_dim == n_plus_1:
            return (
                torch.eye(self.ambient_dim, device=self.device)
                .unsqueeze(0)
                .expand(self.n_spheres, -1, -1)
                .clone()
            )
        else:
            tilts = torch.zeros(
                self.n_spheres, self.ambient_dim, n_plus_1, device=self.device
            )
            for i in range(self.n_spheres):
                A = self._randn(self.ambient_dim, n_plus_1)
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
            noise_std: Override for Gaussian noise std. None falls back to
                self.noise_std.
        """
        ns = self.n_spheres
        n_plus_1 = self.sphere_dim + 1
        effective_noise = self.noise_std if noise_std is None else noise_std

        # 1. Activation mask: (batch_size, n_spheres)
        mask = self._rand(batch_size, ns) < self.p_active

        # 2. Sample uniformly on S^n: draw from N(0,1) and normalize
        z = self._randn(batch_size, ns, n_plus_1)
        z = z / z.norm(dim=-1, keepdim=True)
        z = self.radius * z

        # 2b. Infill: with probability p_infill, sample inside the ball
        if self.p_infill > 0:
            infill = self._rand(batch_size, ns) < self.p_infill
            # Uniform-in-ball: scale radius by u^(1/(n+1)) where u ~ U(0,1)
            u = self._rand(batch_size, ns)
            radial = u.pow(1.0 / n_plus_1)
            scale = torch.where(infill, radial, torch.ones_like(radial))
            z = z * scale.unsqueeze(-1)

        # 3. Apply tilt: (batch, ns, n+1) @ (ns, n+1, ambient) -> (batch, ns, ambient)
        tilted = torch.einsum("bkj,kjm->bkm", z, self.tilts.transpose(-1, -2))

        # 4. Shift into non-negative orthant (center + sphere point)
        tilted = tilted + self.centers

        # 5. Zero out inactive features (center included)
        tilted = tilted * mask.unsqueeze(-1)

        # 6. Add Gaussian noise to active features only
        if effective_noise > 0:
            noise = self._randn(batch_size, ns, self.ambient_dim) * effective_noise
            tilted = tilted + noise * mask.unsqueeze(-1)

        # 7. Reshape to (batch_size, n_spheres * ambient_dim)
        output = tilted.reshape(batch_size, ns * self.ambient_dim)
        return output, mask

    def sample(self, batch_size: int, noise_std: float | None = None):
        """Sample from the distribution.

        Args:
            batch_size: Number of samples.
            noise_std: Override for Gaussian noise std. None falls back to
                self.noise_std.
        """
        output, _ = self._sample_core(batch_size, noise_std=noise_std)
        return output

    def sample_with_args(
        self,
        batch_size: int,
        with_labels: bool = True,
        noise_std: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Like sample(), but optionally returns the (batch_size, n_spheres) boolean mask.

        Args:
            batch_size: Number of samples.
            with_labels: If True, return (output, mask) tuple. If False,
                return output only.
            noise_std: Override for Gaussian noise std. None falls back to
                self.noise_std.
        """
        output, mask = self._sample_core(batch_size, noise_std=noise_std)
        if with_labels:
            return output, mask
        return output
