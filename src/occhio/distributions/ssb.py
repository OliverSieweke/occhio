"""SynthSAEBench data generator.

Produces batches of hidden activations a = D^T c + b from a ground-truth feature
dictionary D ∈ R^{N×D}, where N > D (superposition). Supports correlated firings
via a Gaussian copula, hierarchical feature dependencies, and configurable
probability/magnitude distributions.

Used downstream to benchmark SAE variants against known ground-truth features.

Reference: `SynthSAEBench <https://arxiv.org/abs/2602.14687>`_
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .base import Distribution


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HierarchyNode:
    """A single node in the hierarchy forest.

    Args:
        feature_idx: Index of the feature this node represents.
        children: Child nodes in the hierarchy.
        mutually_exclusive_children: If True, at most one child fires per sample.
        parent_scaled: If True, child magnitudes are modulated by parent magnitude.
    """

    feature_idx: int
    children: list[HierarchyNode] = field(default_factory=list)
    mutually_exclusive_children: bool = False
    parent_scaled: bool = False


@dataclass
class SyntheticBatch:
    """Output of a single sampling call.

    Attributes:
        activations: Hidden activations a = D^T c + b, shape ``(batch_size, D)``.
        coefficients: Sparse coefficients c (post-hierarchy), shape ``(batch_size, N)``.
        firing_mask: Binary firing indicators z (pre-hierarchy), shape ``(batch_size, N)``.
    """

    activations: Tensor
    coefficients: Tensor
    firing_mask: Tensor


@dataclass
class SyntheticDataConfig:
    """Full configuration for :class:`SyntheticDataModel`.

    Groups all hyperparameters for the feature dictionary, firing probabilities,
    magnitudes, correlation structure, hierarchy, bias, and runtime settings.
    """

    # Features
    n_features: int

    # Firing probabilities
    firing_prob_distribution: str = "zipfian"
    p_min: float = 0.001
    p_max: float = 0.1
    alpha: float = 1.0
    p_constant: Optional[float] = None

    # Firing magnitudes
    mean_distribution: str = "constant"
    mean_value: float = 1.0
    mean_high: Optional[float] = None
    mean_low: Optional[float] = None
    std_distribution: str = "constant"
    std_value: float = 0.5
    std_high: Optional[float] = None
    std_low: Optional[float] = None
    folded_normal_mu: Optional[float] = None
    folded_normal_sigma: Optional[float] = None

    # Correlation
    correlation_rank: int = 0
    correlation_scale: float = 0.1
    delta_min: float = 0.01

    # Hierarchy
    hierarchy: Optional[list[HierarchyNode]] = None
    compensate_probabilities: bool = True

    # Post-processing
    post_processing: Optional[Callable[[Tensor], Tensor]] = None

    # Runtime
    device: str = "cpu"
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Helper: parameter schedule generation
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
}


def _make_schedule(
    n: int,
    distribution: str,
    *,
    value: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    folded_mu: Optional[float] = None,
    folded_sigma: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build a per-feature parameter vector according to a named schedule."""
    if distribution == "constant":
        assert value is not None, "value required for constant schedule"
        return torch.full((n,), value, device=device)
    elif distribution == "linear":
        assert high is not None and low is not None
        return torch.linspace(high, low, n, device=device)
    elif distribution == "exponential":
        assert high is not None and low is not None
        log_high, log_low = math.log(high), math.log(low)
        return torch.logspace(log_high, log_low, n, base=math.e, device=device)
    elif distribution == "folded_normal":
        assert folded_mu is not None and folded_sigma is not None
        return (
            folded_mu
            + folded_sigma * torch.randn(n, device=device, generator=generator)
        ).abs()
    else:
        raise ValueError(f"Unknown schedule distribution: {distribution}")


# ---------------------------------------------------------------------------
# CorrelationStructure
# ---------------------------------------------------------------------------


class CorrelationStructure:
    """Low-rank correlation Σ = FF^T + diag(δ) for Gaussian copula sampling."""

    def __init__(
        self,
        n_features: int,
        rank: int = 0,
        correlation_scale: float = 0.1,
        delta_min: float = 0.01,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ):
        self.n_features = n_features
        self.rank = rank
        self.device = torch.device(device)
        self.dtype = dtype

        if rank == 0:
            self.factor_matrix = None
            self.diagonal = None
            return

        # F ~ scale * N(0, 1)
        F = correlation_scale * torch.randn(
            n_features, rank, device=self.device, dtype=dtype, generator=generator
        )

        # δ_i = 1 - Σ_j F²_ij
        row_norms_sq = (F**2).sum(dim=1)
        delta = 1.0 - row_norms_sq

        # Numerical stability: if any δ_i < delta_min, rescale F
        if (delta < delta_min).any():
            scale = math.sqrt((1.0 - delta_min) / row_norms_sq.max().item())
            F = F * scale
            row_norms_sq = (F**2).sum(dim=1)
            delta = 1.0 - row_norms_sq

        self.factor_matrix = F
        self.diagonal = delta

    def sample(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Sample correlated Gaussians g ∈ R^{batch_size × N}."""
        if self.rank == 0:
            return torch.randn(
                batch_size,
                self.n_features,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )

        eps = torch.randn(
            batch_size,
            self.rank,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        eta = torch.randn(
            batch_size,
            self.n_features,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        assert self.factor_matrix is not None and self.diagonal is not None
        return eps @ self.factor_matrix.T + self.diagonal.sqrt() * eta


# ---------------------------------------------------------------------------
# FiringProbDistribution
# ---------------------------------------------------------------------------


def _compute_firing_probs(
    n_features: int,
    distribution: str,
    p_min: float,
    p_max: float,
    alpha: float = 1.0,
    p_constant: Optional[float] = None,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Compute per-feature firing probabilities p_i."""
    if distribution == "constant":
        p = p_constant if p_constant is not None else p_min
        return torch.full((n_features,), p, device=device)
    elif distribution == "linear":
        return torch.linspace(p_max, p_min, n_features, device=device)
    elif distribution == "uniform":
        return p_min + (p_max - p_min) * torch.rand(
            n_features, device=device, generator=generator
        )
    elif distribution == "zipfian":
        i = torch.arange(1, n_features + 1, device=device, dtype=torch.float)
        q = i.pow(-alpha)
        q_min, q_max = q[-1], q[0]
        if q_max == q_min:
            return torch.full((n_features,), (p_min + p_max) / 2, device=device)
        return p_min + (p_max - p_min) * (q - q_min) / (q_max - q_min)
    else:
        raise ValueError(f"Unknown firing prob distribution: {distribution}")


# ---------------------------------------------------------------------------
# FiringSampler
# ---------------------------------------------------------------------------


class FiringSampler:
    """Produce binary firing indicators z ∈ {0,1}^N via Gaussian copula."""

    def __init__(
        self,
        probabilities: Tensor,
        correlation: CorrelationStructure,
    ):
        self.correlation = correlation
        # Threshold: τ_i = Φ⁻¹(1 - p_i)
        # Φ⁻¹(q) = sqrt(2) * erfinv(2q - 1)
        self._thresholds = self._inv_normal_cdf(1.0 - probabilities)

    @staticmethod
    def _inv_normal_cdf(q: Tensor) -> Tensor:
        """Inverse standard normal CDF via erfinv."""
        return math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)

    def sample(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Sample binary firing mask z ∈ {0,1}^{batch_size × N}."""
        if self.correlation.rank == 0:
            # Independent Bernoulli — avoid copula overhead
            probs = self._inv_normal_cdf_to_prob(self._thresholds)
            u = torch.rand(
                batch_size,
                probs.shape[0],
                device=probs.device,
                dtype=probs.dtype,
                generator=generator,
            )
            return (u < probs).float()

        g = self.correlation.sample(batch_size, generator=generator)
        return (g > self._thresholds).float()

    @staticmethod
    def _inv_normal_cdf_to_prob(thresholds: Tensor) -> Tensor:
        """Convert thresholds back to probabilities: p = 1 - Φ(τ)."""
        return 1.0 - 0.5 * (1.0 + torch.erf(thresholds / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# MagnitudeSampler
# ---------------------------------------------------------------------------


class MagnitudeSampler:
    """Sample non-negative coefficients via rectified Gaussian: c = z · ReLU(μ + σε)."""

    def __init__(self, means: Tensor, stds: Tensor):
        self.means = means
        self.stds = stds

    def sample(self, z: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """Given firing mask z, produce coefficients c ≥ 0."""
        eps = torch.randn_like(z, generator=generator)
        raw = self.means + self.stds * eps
        return z * torch.relu(raw)


# ---------------------------------------------------------------------------
# HierarchyConstraints
# ---------------------------------------------------------------------------


class HierarchyConstraints:
    """Enforce parent-child gating, mutual exclusion, and parent-scaled magnitudes."""

    def __init__(
        self,
        forest: list[HierarchyNode],
        n_features: int,
        compensate: bool = True,
        base_probs: Optional[Tensor] = None,
        mean_magnitudes: Optional[Tensor] = None,
        device: torch.device | str = "cpu",
    ):
        self.n_features = n_features
        self.device = torch.device(device)
        self.compensate = compensate

        # Parse forest into level-order structures
        self._levels: list[list[_LevelOp]] = []
        self._parent_of: dict[int, int] = {}
        self._me_groups: dict[int, list[int]] = {}  # parent_idx -> child indices
        self._parent_scaled: dict[int, bool] = {}  # parent_idx -> bool

        # Mean magnitudes for parent-scaling
        self._mean_magnitudes = mean_magnitudes

        self._parse_forest(forest)

        # Precompute compensation factors
        self._compensation = torch.ones(n_features, device=self.device)
        if compensate and base_probs is not None:
            self._compute_compensation(base_probs)

    def _parse_forest(self, forest: list[HierarchyNode]) -> None:
        """Convert forest of HierarchyNodes into level-ordered operations."""
        # BFS to get levels
        queue: list[tuple[HierarchyNode, int]] = [(root, 0) for root in forest]
        level_ops: dict[int, list[_LevelOp]] = {}

        while queue:
            node, depth = queue.pop(0)
            if node.children:
                child_indices = [c.feature_idx for c in node.children]
                op = _LevelOp(
                    parent_idx=node.feature_idx,
                    child_indices=torch.tensor(child_indices, device=self.device),
                    mutually_exclusive=node.mutually_exclusive_children,
                    parent_scaled=node.parent_scaled,
                )
                level_ops.setdefault(depth, []).append(op)

                for child in node.children:
                    self._parent_of[child.feature_idx] = node.feature_idx
                    queue.append((child, depth + 1))

                if node.mutually_exclusive_children:
                    self._me_groups[node.feature_idx] = child_indices

                self._parent_scaled[node.feature_idx] = node.parent_scaled

        # Convert to sorted list of levels
        if level_ops:
            max_depth = max(level_ops.keys())
            self._levels = [level_ops.get(d, []) for d in range(max_depth + 1)]

    def _compute_compensation(self, base_probs: Tensor) -> None:
        """Compute hierarchy + ME probability compensation factors."""
        gamma = torch.ones(self.n_features, device=self.device)

        for child_idx, parent_idx in self._parent_of.items():
            p_parent = base_probs[parent_idx].item()
            if p_parent > 0:
                gamma[child_idx] *= 1.0 / p_parent

        # ME correction
        for parent_idx, children in self._me_groups.items():
            p_parent = base_probs[parent_idx].item()
            if p_parent <= 0:
                continue
            for i, ci in enumerate(children):
                me_factor = 1.0
                for j, cj in enumerate(children):
                    if i != j:
                        me_factor += base_probs[cj].item() / p_parent
                gamma[ci] *= me_factor

        self._compensation = gamma

    def get_compensated_probs(self, base_probs: Tensor) -> Tensor:
        """Return hierarchy-compensated firing probabilities."""
        return (base_probs * self._compensation).clamp(max=1.0)

    def apply(self, c: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """Apply hierarchy constraints level-by-level, root → leaves."""
        c = c.clone()
        for level in self._levels:
            for op in level:
                parent_active = c[:, op.parent_idx] > 0  # (batch,)

                # 1. Binary gating
                c[:, op.child_indices] *= parent_active.unsqueeze(1).float()

                # 2. Mutual exclusion
                if op.mutually_exclusive:
                    child_cols = op.child_indices  # (n_children,)
                    child_vals = c[:, child_cols]  # (batch, n_children)
                    fired = child_vals > 0  # (batch, n_children)
                    n_fired = fired.sum(dim=1)  # (batch,)

                    multi = n_fired > 1
                    if multi.any():
                        multi_fired = fired[multi]  # (n_multi, n_children)
                        # Pick random winner among fired children
                        rand_vals = torch.rand_like(
                            multi_fired.float(), generator=generator
                        )
                        rand_vals[~multi_fired] = -1.0
                        winners = rand_vals.argmax(dim=1)  # (n_multi,)

                        # Build mask: keep only the winner for each multi-fire row
                        winner_mask = torch.zeros_like(multi_fired)
                        winner_mask.scatter_(1, winners.unsqueeze(1), True)
                        # Zero losers
                        loser_mask = multi_fired & ~winner_mask  # (n_multi, n_children)

                        # Write zeros for losers back into c
                        batch_idx = multi.nonzero(as_tuple=True)[0]  # (n_multi,)
                        for ci in range(child_cols.shape[0]):
                            losers_in_col = loser_mask[:, ci]
                            if losers_in_col.any():
                                c[batch_idx[losers_in_col], child_cols[ci]] = 0.0

                # 3. Parent-scaled magnitudes
                if op.parent_scaled and self._mean_magnitudes is not None:
                    mu_parent = self._mean_magnitudes[op.parent_idx]
                    if mu_parent > 0:
                        scale = c[:, op.parent_idx] / mu_parent
                        c[:, op.child_indices] *= scale.unsqueeze(1)

        return c

    @property
    def has_constraints(self) -> bool:
        return len(self._levels) > 0


@dataclass
class _LevelOp:
    """A single parent→children operation at one level of the hierarchy."""

    parent_idx: int
    child_indices: Tensor
    mutually_exclusive: bool
    parent_scaled: bool


# ---------------------------------------------------------------------------
# SyntheticDataModel
# ---------------------------------------------------------------------------


class SyntheticDataModel(Distribution):
    """Synthetic sparse feature distribution for superposition experiments.

    Based on the data model from https://arxiv.org/abs/2602.14687.

    Produces batches of sparse coefficient vectors ``c`` of shape
    ``(batch_size, n_features)`` with configurable firing patterns,
    correlations, and hierarchical dependencies. The compression into a
    lower-dimensional hidden space is left to the autoencoder.

    Args:
        config: Full configuration specifying all generator parameters.
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, config: SyntheticDataConfig, seed: Optional[int] = None):
        device = torch.device(config.device)
        self._dtype = _DTYPE_MAP.get(config.dtype, torch.float32)

        # Build generator for reproducibility
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        super().__init__(
            n_features=config.n_features, device=device, generator=generator
        )

        self.config = config
        N = config.n_features

        # 1. Base firing probabilities
        self._base_probs = _compute_firing_probs(
            N,
            config.firing_prob_distribution,
            config.p_min,
            config.p_max,
            config.alpha,
            config.p_constant,
            device=device,
            generator=self.generator,
        ).to(self._dtype)

        # 2. Magnitude parameters
        self._means = _make_schedule(
            N,
            config.mean_distribution,
            value=config.mean_value,
            high=config.mean_high,
            low=config.mean_low,
            device=device,
        ).to(self._dtype)

        self._stds = _make_schedule(
            N,
            config.std_distribution,
            value=config.std_value,
            high=config.std_high,
            low=config.std_low,
            folded_mu=config.folded_normal_mu,
            folded_sigma=config.folded_normal_sigma,
            generator=self.generator,
            device=device,
        ).to(self._dtype)

        self._magnitude_sampler = MagnitudeSampler(self._means, self._stds)

        # 3. Hierarchy
        forest = config.hierarchy or []
        self._hierarchy = HierarchyConstraints(
            forest=forest,
            n_features=N,
            compensate=config.compensate_probabilities,
            base_probs=self._base_probs,
            mean_magnitudes=self._means,
            device=device,
        )

        # 4. Compute effective probabilities (compensated if hierarchy active)
        if self._hierarchy.has_constraints and config.compensate_probabilities:
            effective_probs = self._hierarchy.get_compensated_probs(self._base_probs)
        else:
            effective_probs = self._base_probs

        # 5. Correlation structure
        self._correlation = CorrelationStructure(
            n_features=N,
            rank=config.correlation_rank,
            correlation_scale=config.correlation_scale,
            delta_min=config.delta_min,
            device=device,
            dtype=self._dtype,
            generator=self.generator,
        )

        # 6. Firing sampler
        self._firing_sampler = FiringSampler(effective_probs, self._correlation)

    def sample(self, batch_size: int) -> Tensor:
        """Generate a batch of sparse coefficient vectors.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Sparse coefficients of shape ``(batch_size, n_features)``.
        """
        # 1. Firing indicators
        z = self._firing_sampler.sample(batch_size, generator=self.generator)

        # 2. Magnitudes
        c = self._magnitude_sampler.sample(z, generator=self.generator)

        # 3. Hierarchy constraints
        if self._hierarchy.has_constraints:
            c = self._hierarchy.apply(c, generator=self.generator)

        # 4. Post-processing
        if self.config.post_processing is not None:
            c = self.config.post_processing(c)

        return c

    def to(self, device: torch.device | str):
        super().to(device)
        self._base_probs = self._base_probs.to(self.device)
        self._means = self._means.to(self.device)
        self._stds = self._stds.to(self.device)
        self._magnitude_sampler = MagnitudeSampler(self._means, self._stds)
        self._firing_sampler._thresholds = self._firing_sampler._thresholds.to(
            self.device
        )
        self._correlation.device = self.device
        if self._correlation.factor_matrix is not None:
            self._correlation.factor_matrix = self._correlation.factor_matrix.to(
                self.device
            )
        if self._correlation.diagonal is not None:
            self._correlation.diagonal = self._correlation.diagonal.to(self.device)
        return self

    @property
    def firing_probabilities(self) -> Tensor:
        """Shape ``(N,)``, base (uncompensated) probabilities."""
        return self._base_probs

    @property
    def compensated_probabilities(self) -> Tensor:
        """Shape ``(N,)``, hierarchy-compensated probabilities."""
        if self._hierarchy.has_constraints and self.config.compensate_probabilities:
            return self._hierarchy.get_compensated_probs(self._base_probs)
        return self._base_probs
