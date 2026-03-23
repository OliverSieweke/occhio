"""SparseSpheres: k independent features on S^n, axis-aligned in R^{k*m}.

Note on generator hygiene: the _rand* methods share a single torch.Generator
that advances in-place on every call. When sampling parameters are overridden
via sample_with_args, the number of generator draws may differ from the default
path (e.g. discrete vs continuous, infill vs surface-only), so generator state
after a call depends on the effective parameter values. Callers who need
bit-exact reproducibility across different override combinations should use
fresh generators.

To Claude: Don't do anything about this for now, just keep this doctring for reference
so I (Kaushik) can fix this down the line.
"""

import math
import warnings

import torch
from torch import Tensor

from .base import Distribution


def _halton_sequence(n_points: int, base: int) -> list[float]:
    """Generate a 1-D Halton sequence of length *n_points* for the given prime *base*.

    Returns values in (0, 1), deterministic and low-discrepancy.
    """
    result: list[float] = []
    for i in range(1, n_points + 1):
        f, r = 1.0, 0.0
        n = i
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        result.append(r)
    return result


def _make_discrete_points(
    n_disc: int, sphere_dim: int, device: torch.device | str
) -> Tensor:
    """Build *n_disc* approximately equidistant points on S^{sphere_dim}.

    Returns a (n_disc, sphere_dim+1) tensor of unit-norm points. Fully
    deterministic (no generator needed).

    * S^1: exact equidistant via linspace on angles.
    * S^2: Fibonacci lattice (golden-angle method).
    * S^n (n >= 3): Halton sequence -> inverse normal CDF -> L2-normalize.
    """
    if sphere_dim == 1:
        # Equidistant on the circle
        angles = torch.linspace(0, 2 * math.pi, n_disc + 1, device=device)[:n_disc]
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    if sphere_dim == 2:
        # Fibonacci lattice on S^2
        golden_ratio = (1 + math.sqrt(5)) / 2
        indices = torch.arange(n_disc, dtype=torch.float64, device=device)
        theta = torch.acos(1 - 2 * (indices + 0.5) / n_disc)
        phi = 2 * math.pi * indices / golden_ratio
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1).float()

    # General S^n (n >= 3): Halton -> inverse normal CDF -> normalize
    dim = sphere_dim + 1
    # Use first `dim` primes as bases
    primes = _first_primes(dim)
    cols = []
    for d in range(dim):
        h = torch.tensor(
            _halton_sequence(n_disc, primes[d]), dtype=torch.float32, device=device
        )
        # Clamp away from 0/1 to avoid inf in erfinv
        h = h.clamp(1e-6, 1 - 1e-6)
        cols.append(torch.erfinv(2 * h - 1) * math.sqrt(2))
    pts = torch.stack(cols, dim=-1)
    pts = pts / pts.norm(dim=-1, keepdim=True)
    return pts


def _first_primes(n: int) -> list[int]:
    """Return the first *n* prime numbers."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


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
        n_discretizations: If > 0, sample from a fixed set of this many
            approximately equidistant points on S^n rather than continuously.
            Default 0 (continuous sampling).
        **kwargs: Passed to Distribution (device, generator).
    """

    def __init__(
        self,
        *,
        n_spheres: int,
        sphere_dim: int = 1,
        ambient_dim: int | None = None,
        radius: float = 1.0,
        p_active: float | list[float] | Tensor,
        p_infill: float = 0.0,
        noise_std: float = 0.0,
        n_discretizations: int = 0,
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
        self.n_discretizations = n_discretizations

        if n_discretizations > 0 and noise_std > 0:
            warnings.warn(
                "n_discretizations with noise_std > 0: noise will be added on top "
                "of discrete points, which may defeat the purpose of discretization.",
                UserWarning,
                stacklevel=2,
            )

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

        if n_discretizations > 0:
            self._discrete_points = _make_discrete_points(
                n_discretizations, self.sphere_dim, device=self.device
            )
        else:
            self._discrete_points = None

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
                # QR is unsupported on MPS — compute on CPU, move back
                Q, R = torch.linalg.qr(A.cpu())
                # Fix sign ambiguity for deterministic QR
                Q = (Q * torch.sign(torch.diag(R))).to(self.device)
                tilts[i] = Q
            return tilts

    def _sample_core(
        self,
        batch_size: int,
        *,
        noise_std: float | None = None,
        p_active: float | list[float] | Tensor | None = None,
        p_infill: float | None = None,
        n_discretizations: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Shared sampling logic. Returns (output, mask).

        All keyword arguments override the corresponding ``self.*`` attribute
        for this call only — no mutation of ``self``.

        Args:
            batch_size: Number of samples.
            noise_std: Override for Gaussian noise std.
            p_active: Override for per-feature activation probability.
            p_infill: Override for infill probability.
            n_discretizations: Override for discrete-point count (0 = continuous).
        """
        ns = self.n_spheres
        n_plus_1 = self.sphere_dim + 1

        # Effective parameter resolution
        eff_noise = self.noise_std if noise_std is None else noise_std
        eff_p_infill = self.p_infill if p_infill is None else p_infill
        eff_n_disc = (
            self.n_discretizations if n_discretizations is None else n_discretizations
        )

        if p_active is None:
            eff_p_active = self.p_active
        elif isinstance(p_active, Tensor):
            eff_p_active = (
                p_active.to(self.device)
                if p_active.dim() > 0
                else p_active.expand(ns).to(self.device)
            )
        elif isinstance(p_active, (int, float)):
            eff_p_active = torch.full((ns,), p_active, device=self.device)
        else:
            eff_p_active = torch.as_tensor(p_active, device=self.device)

        # Resolve discrete points: use cached if matching, else compute on-the-fly
        if eff_n_disc > 0:
            if (
                eff_n_disc == self.n_discretizations
                and self._discrete_points is not None
            ):
                disc_pts = self._discrete_points
            else:
                disc_pts = _make_discrete_points(
                    eff_n_disc, self.sphere_dim, device=self.device
                )
        else:
            disc_pts = None

        # 1. Activation mask: (batch_size, n_spheres)
        mask = self._rand(batch_size, ns) < eff_p_active

        # 2. Sphere points: (batch_size, ns, n+1)
        if disc_pts is not None:
            # Pick random indices into the discrete point set
            indices = self._randint(0, eff_n_disc, (batch_size, ns))  # (batch, ns)
            z = disc_pts[indices]  # (batch, ns, n+1)
            z = self.radius * z
        else:
            # Continuous: draw from N(0,1) and normalize
            z = self._randn(batch_size, ns, n_plus_1)
            z = z / z.norm(dim=-1, keepdim=True)
            z = self.radius * z

        # 2b. Infill: with probability p_infill, sample inside the ball
        if eff_p_infill > 0:
            infill = self._rand(batch_size, ns) < eff_p_infill
            # Uniform-in-ball: scale radius by u^(1/(n+1)) where u ~ U(0,1)
            u = self._rand(batch_size, ns)
            radial = u.pow(1.0 / n_plus_1)
            scale = torch.where(infill, radial, torch.ones_like(radial))

            if disc_pts is not None:
                # For infill samples, replace discrete direction with fresh continuous
                continuous_dir = self._randn(batch_size, ns, n_plus_1)
                continuous_dir = continuous_dir / continuous_dir.norm(
                    dim=-1, keepdim=True
                )
                continuous_dir = self.radius * continuous_dir
                # infill samples get continuous direction + radial scaling;
                # surface samples keep their discrete points unscaled
                z = torch.where(
                    infill.unsqueeze(-1),
                    continuous_dir * radial.unsqueeze(-1),
                    z,
                )
            else:
                z = z * scale.unsqueeze(-1)

        # 3. Apply tilt: (batch, ns, n+1) @ (ns, n+1, ambient) -> (batch, ns, ambient)
        tilted = torch.einsum("bkj,kjm->bkm", z, self.tilts.transpose(-1, -2))

        # 4. Shift into non-negative orthant (center + sphere point)
        tilted = tilted + self.centers

        # 5. Zero out inactive features (center included)
        tilted = tilted * mask.unsqueeze(-1)

        # 6. Add Gaussian noise to active features only
        if eff_noise > 0:
            noise = self._randn(batch_size, ns, self.ambient_dim) * eff_noise
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
        *,
        noise_std: float | None = None,
        p_active: float | list[float] | Tensor | None = None,
        p_infill: float | None = None,
        n_discretizations: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Like sample(), but all sampling params can be temporarily overridden.

        Args:
            batch_size: Number of samples.
            with_labels: If True, return (output, mask) tuple. If False,
                return output only.
            noise_std: Override for Gaussian noise std.
            p_active: Override for per-feature activation probability.
            p_infill: Override for infill probability.
            n_discretizations: Override for discrete-point count (0 = continuous).
        """
        output, mask = self._sample_core(
            batch_size,
            noise_std=noise_std,
            p_active=p_active,
            p_infill=p_infill,
            n_discretizations=n_discretizations,
        )
        if with_labels:
            return output, mask
        return output
