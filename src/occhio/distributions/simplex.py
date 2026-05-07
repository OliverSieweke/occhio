"""Simplex-based sparse distributions.

Each sample consists of a collection of simplices. Each simplex independently
fires with probability p, and when active its features are drawn from a
symmetric Dirichlet (via normalized Exp(1) variates).

Also provides ``SimplicialComplexDistribution`` for simplices that share
vertices (glued along faces/edges/vertices).
"""

import torch
from torch import Tensor
from typing import Literal

from .base import Distribution


class SimplexDistribution(Distribution):
    """A distribution over a collection of sparse simplices.

    The feature space is partitioned into groups (simplices). Each simplex of
    size k_i fires independently with probability p_i. When a simplex fires,
    its k_i features are sampled as normalized Exp(1) random variables
    (i.e. a draw from a symmetric Dirichlet(1, ..., 1) distribution).
    When a simplex does not fire, all its features are zero.

    Args:
        simplex_sizes: List of simplex sizes ``[k_1, k_2, ...]``.
            Total features = ``sum(simplex_sizes)``.
        p_active: Firing probability per simplex. Either a scalar (same for
            all simplices) or a list of per-simplex probabilities.
        **kwargs: Passed to ``Distribution`` (device, generator).
    """

    def __init__(
        self,
        simplex_sizes: list[int],
        p_active: float | list[float] | Tensor,
        **kwargs,
    ):
        n_features = sum(simplex_sizes)
        super().__init__(n_features, **kwargs)
        self.simplex_sizes = simplex_sizes
        self.n_simplices = len(simplex_sizes)

        if isinstance(p_active, (int, float)):
            self.p_active = torch.full(
                (self.n_simplices,), p_active, device=self.device
            )
        elif isinstance(p_active, list):
            self.p_active = torch.tensor(p_active, device=self.device)
        else:
            self.p_active = p_active.to(self.device)

        if len(self.p_active) != self.n_simplices:
            raise ValueError(
                f"p_active has length {len(self.p_active)}, "
                f"expected {self.n_simplices} (one per simplex)"
            )

        # CUDA perf: precompute on device to avoid creating this tensor
        # on every sample() call
        self._simplex_sizes_tensor = torch.tensor(
            self.simplex_sizes, device=self.device
        )

    def sample(self, batch_size: int) -> Tensor:
        # Sample all Exp(1) variates jointly: -log(U)
        u = self._rand(batch_size, self.n_features).clamp(min=1e-10)
        exp_samples = -torch.log(u)

        # Normalize each simplex group independently
        result = torch.zeros_like(exp_samples)
        offset = 0
        for k in self.simplex_sizes:
            block = exp_samples[:, offset : offset + k]
            result[:, offset : offset + k] = block / block.sum(dim=-1, keepdim=True)
            offset += k

        # Zero out non-firing simplices: expand p_active per-feature
        fire = self._rand(batch_size, self.n_simplices) < self.p_active
        mask = fire.repeat_interleave(self._simplex_sizes_tensor, dim=1)
        result *= mask

        return result


class SimplicialComplexDistribution(Distribution):
    """A distribution over a simplicial complex with shared vertices.

    Faces are defined as tuples of vertex indices into a shared vertex set.
    Faces may overlap (share vertices), which is how gluing is expressed.

    In ``"single"`` mode, exactly one face is sampled per row and a Dirichlet
    draw is placed on its vertices. In ``"sparse"`` mode, each face fires
    independently with probability ``p_active``; contributions from
    overlapping faces are summed and then re-normalized so active vertices
    sum to 1.

    Args:
        n_vertices: Total number of vertices (= number of features).
        faces: List of tuples, each containing vertex indices for one face.
            E.g. ``[(0,1), (1,2), (2,0)]`` for a triangle of edges.
        p_active: Firing probability per face. Scalar or per-face list.
            Required for ``"sparse"`` mode; ignored for ``"single"``.
        sampling_mode: ``"single"`` (pick one face) or ``"sparse"``
            (each face fires independently).
        **kwargs: Passed to ``Distribution`` (device, generator).

    Example — circle of 6 edges::

        faces = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        dist = SimplicialComplexDistribution(6, faces, sampling_mode="single")
        dist.sample(64)  # shape: (64, 6)
    """

    def __init__(
        self,
        n_vertices: int,
        faces: list[tuple[int, ...]],
        p_active: float | list[float] | Tensor = 0.5,
        sampling_mode: Literal["single", "sparse"] = "single",
        **kwargs,
    ):
        super().__init__(n_vertices, **kwargs)
        self.faces = faces
        self.n_faces = len(faces)
        self.sampling_mode = sampling_mode

        # Pre-compute a (n_faces, n_vertices) binary mask for scatter_add
        face_mask = torch.zeros(self.n_faces, n_vertices, device=self.device)
        for i, face in enumerate(faces):
            for v in face:
                face_mask[i, v] = 1.0
        self.face_mask = face_mask

        # Store face sizes for Dirichlet sampling
        self.face_sizes = [len(f) for f in faces]
        self.max_face_size = max(self.face_sizes)
        self.uniform_faces = len(set(self.face_sizes)) == 1

        # Index tensor: (n_faces, max_face_size) — no padding needed when uniform
        padded = torch.zeros(
            self.n_faces, self.max_face_size, dtype=torch.long, device=self.device
        )
        valid = torch.zeros(
            self.n_faces, self.max_face_size, dtype=torch.bool, device=self.device
        )
        for i, face in enumerate(faces):
            for j, v in enumerate(face):
                padded[i, j] = v
                valid[i, j] = True
        self.face_indices = padded  # (n_faces, max_face_size)
        self.face_valid = valid  # (n_faces, max_face_size)

        if isinstance(p_active, (int, float)):
            self.p_active = torch.full((self.n_faces,), p_active, device=self.device)
        elif isinstance(p_active, list):
            self.p_active = torch.tensor(p_active, device=self.device)
        else:
            self.p_active = p_active.to(self.device)

    def sample(self, batch_size: int) -> Tensor:
        if self.sampling_mode == "single":
            return self._sample_single(batch_size)
        else:
            return self._sample_sparse(batch_size)

    def _sample_single(self, batch_size: int) -> Tensor:
        """Pick one face uniformly per sample, draw Dirichlet on its vertices."""
        face_idx = self._randint(0, self.n_faces, (batch_size,))

        if self.uniform_faces:
            return self._sample_single_fast(batch_size, face_idx)

        result = torch.zeros(batch_size, self.n_features, device=self.device)
        # CUDA perf: batch-compute per-face counts in one op + one CPU transfer,
        # instead of .sum().item() per face (n_faces CUDA syncs)
        counts = torch.bincount(face_idx, minlength=self.n_faces).cpu().tolist()
        for i in range(self.n_faces):
            n_active = counts[i]
            if n_active == 0:
                continue
            mask = face_idx == i
            face = self.faces[i]
            k = len(face)
            u = self._rand(n_active, k).clamp(min=1e-10)
            dirichlet = -torch.log(u)
            dirichlet = dirichlet / dirichlet.sum(dim=-1, keepdim=True)
            for j, v in enumerate(face):
                result[mask, v] = dirichlet[:, j]
        return result

    def _sample_single_fast(self, batch_size: int, face_idx: Tensor) -> Tensor:
        """Vectorized single-face sampling when all faces have the same size."""
        k = self.face_sizes[0]
        # Dirichlet via normalized Exp(1): (batch, k)
        u = self._rand(batch_size, k).clamp(min=1e-10)
        dirichlet = -torch.log(u)
        dirichlet = dirichlet / dirichlet.sum(dim=-1, keepdim=True)

        # Gather the vertex indices for each sample's chosen face: (batch, k)
        target_indices = self.face_indices[face_idx]  # (batch, k)

        result = torch.zeros(batch_size, self.n_features, device=self.device)
        result.scatter_(1, target_indices, dirichlet)
        return result

    def _sample_sparse(self, batch_size: int) -> Tensor:
        """Each face fires independently; sum contributions, re-normalize."""
        if self.uniform_faces:
            return self._sample_sparse_fast(batch_size)

        result = torch.zeros(batch_size, self.n_features, device=self.device)

        for i in range(self.n_faces):
            fire = self._rand(batch_size) < self.p_active[i]
            n_active = int(fire.sum().item())
            if n_active == 0:
                continue
            face = self.faces[i]
            k = len(face)
            u = self._rand(n_active, k).clamp(min=1e-10)
            dirichlet = -torch.log(u)
            dirichlet = dirichlet / dirichlet.sum(dim=-1, keepdim=True)
            for j, v in enumerate(face):
                result[fire, v] += dirichlet[:, j]

        return result

    def _sample_sparse_fast(self, batch_size: int) -> Tensor:
        """Vectorized sparse sampling when all faces have the same size."""
        k = self.face_sizes[0]
        n_f = self.n_faces

        # Determine which faces fire: (batch, n_faces)
        fire = self._rand(batch_size, n_f) < self.p_active.unsqueeze(0)

        # Dirichlet draws for all (batch, face) pairs: (batch, n_faces, k)
        u = self._rand(batch_size, n_f, k).clamp(min=1e-10)
        dirichlet = -torch.log(u)
        dirichlet = dirichlet / dirichlet.sum(dim=-1, keepdim=True)

        # Zero out non-firing faces
        dirichlet = dirichlet * fire.unsqueeze(-1)

        # Scatter-add into vertex space: (batch, n_features)
        # Expand face_indices (n_faces, k) -> (batch, n_faces, k)
        indices = self.face_indices.unsqueeze(0).expand(batch_size, -1, -1)
        result = torch.zeros(batch_size, self.n_features, device=self.device)
        result.scatter_add_(
            1, indices.reshape(batch_size, -1), dirichlet.reshape(batch_size, -1)
        )

        return result
