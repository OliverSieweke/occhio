"""DAG-based distribution module."""

from .base import Distribution
from torch import Tensor
import torch
import numpy as np


class DAGDistribution(Distribution):
    """DAG-structured distribution with binary activation propagation.

    Nodes are organized in a directed acyclic graph (DAG) structure generated using
    an Erdős-Rényi process. The DAG is represented as an upper triangular adjacency
    matrix where ``adjacency[i, j] = True`` means edge i → j (i is parent of j).

    Root nodes (those without parents) activate independently with probability ``p_active``.
    Non-root nodes activate if at least one parent is active AND an independent coin flip
    with probability ``p_active`` succeeds. Active nodes take values ``~ Uniform(0, 1)``;
    inactive nodes have value 0.

    Args:
        n_features: Number of nodes/features in the DAG.
        p_active: Probability of activation. Scalar or per-feature. Defaults to 0.1.
            For root nodes, this is the independent activation probability.
            For non-root nodes, this is the conditional probability given that
            at least one parent is active.
        p_edge: Probability of edge i → j existing (for i < j) in the Erdős-Rényi
            generation process. Defaults to 0.1.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    def __init__(
        self, n_features: int, p_active: float = 0.1, p_edge: float = 0.1, **kwargs
    ):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.p_edge = p_edge

        self.regenerate_dag()

    def _generate_dag(self) -> Tensor:
        """Generate random DAG as upper triangular adjacency matrix."""
        adj = torch.triu(
            self._rand(self.n_features, self.n_features) < self.p_edge,
            diagonal=1,
        )
        return adj

    def regenerate_dag(self) -> None:
        """Generate a new random DAG structure."""
        self.adjacency = self._generate_dag()

    def sample(self, batch_size: int) -> Tensor:
        active = torch.zeros(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )

        for i in range(self.n_features):
            parent_mask = self.adjacency[:, i]
            has_parents = parent_mask.any()

            if not has_parents:
                active[:, i] = self._rand(batch_size, 1).squeeze(-1) < self.p_active[i]
            else:
                any_parent_active = active[:, parent_mask].any(dim=1)
                fires = self._rand(batch_size, 1).squeeze(-1) < self.p_active[i]
                active[:, i] = any_parent_active & fires

        values = self._rand(batch_size, self.n_features)
        return active.float() * values

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.adjacency = self.adjacency.to(device)
        return self


class DAGBayesianPropagation(Distribution):
    """DAG-structured distribution with Noisy-OR activation propagation.

    Root nodes (those without parents) activate with probability ``p_active`` and take
    values ``~ Uniform(0, 1)``. Non-root nodes use Noisy-OR propagation: given active
    parent values v₁, v₂, ..., vₖ, the node activates with probability:

    .. math::
        P(\\text{activate}) = 1 - \\prod_{j \\in \\text{active parents}} (1 - v_j)

    This implements a Bayesian causal model where activation magnitude represents causal
    influence. A parent with value v=0.9 almost certainly triggers its children, while
    v=0.1 rarely does. Active nodes take values ``~ Uniform(0, 1)``; inactive nodes
    have value 0.

    Args:
        n_features: Number of nodes/features in the DAG.
        p_active: Probability that root nodes activate. Scalar or per-feature.
            Defaults to 0.1.
        p_edge: Probability of edge i → j existing (for i < j) in the Erdős-Rényi
            generation process. Defaults to 0.1.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.

    Note:
        Unlike tree structures, DAG activation probabilities don't have simple closed
        forms due to the Noisy-OR combination over multiple parent paths. Use
        :meth:`get_expected_activation` to estimate marginal probabilities empirically.
    """

    def __init__(
        self,
        n_features: int,
        p_active: float = 0.1,
        p_edge: float = 0.1,
        **kwargs,
    ):
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.p_edge = p_edge

        self.regenerate_dag()

    def _generate_dag(self) -> Tensor:
        """Generate random DAG as upper triangular adjacency matrix."""
        adj = torch.triu(
            self._rand(self.n_features, self.n_features) < self.p_edge,
            diagonal=1,
        )
        return adj

    def _build_parent_cache(self) -> None:
        """Precompute parent indices for efficient sampling."""
        self._parent_indices = []
        self._has_parents = []
        for j in range(self.n_features):
            parents = self.adjacency[:, j].nonzero(as_tuple=True)[0]
            self._parent_indices.append(parents)
            self._has_parents.append(len(parents) > 0)

    def regenerate_dag(self) -> None:
        """Generate a new random DAG structure."""
        self.adjacency = self._generate_dag()
        self._build_parent_cache()

    def sample(self, batch_size: int) -> Tensor:
        """Sample from the DAG distribution with Noisy-OR propagation."""
        values = torch.zeros(batch_size, self.n_features, device=self.device)

        for j in range(self.n_features):
            if not self._has_parents[j]:
                fires = self._rand(batch_size) < self.p_active[j]
            else:
                parent_idx = self._parent_indices[j]
                parent_values = values[:, parent_idx]  # (batch_size, n_parents)

                survival_prob = (1 - parent_values).prod(dim=1)  # (batch_size,)
                fire_prob = 1 - survival_prob

                fires = self._rand(batch_size) < fire_prob

            n_fires = int(fires.sum().item())
            if n_fires > 0:
                values[fires, j] = self._rand(n_fires)

        return values

    def get_expected_activation(self, n_samples: int = 10000) -> Tensor:
        """Estimate marginal activation probabilities via Monte Carlo.

        Unlike tree structures, DAG activation probabilities don't have
        simple closed forms due to Noisy-OR over multiple parents.
        """
        samples = self.sample(n_samples)
        return (samples > 0).float().mean(dim=0)

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.adjacency = self.adjacency.to(device)
        self._parent_indices = [p.to(device) for p in self._parent_indices]
        return self


class DAGRandomWalkToRoot(Distribution):
    """DAG-structured distribution with maximally sparse random-walk-to-root activation.

    The sampling process:

    1. Select one starting node according to probabilities ``p_active``
       (uniform by default)
    2. Activate it with value ``~ Uniform(0, 1)``
    3. Perform a random walk upward: at each step, pick one parent uniformly at random
    4. Activate the chosen parent with a decayed value
    5. Repeat until reaching a root node (no parents)

    Value decay at each step is controlled by ``beta``:

    - If ``beta = 1.0``: deterministic decay, parent value = child value
    - If ``beta < 1.0``: ``parent_value = beta * child_value + (1 - beta) * child_value * U``
      where ``U ~ Uniform(0, 1)``

    Args:
        n_features: Number of nodes/features in the DAG.
        p_edge: Probability of edge i → j existing (for i < j) in the Erdős-Rényi
            generation process. Defaults to 0.1.
        beta: Multiplicative decay factor per step upward. Defaults to 1.0.
            Values in (0, 1] control how much parent activations decay relative
            to child activations.
        p_active: Probability distribution for selecting the starting node.
            If ``None``, uses uniform distribution. Can be a list or Tensor of
            length ``n_features``. Defaults to ``None``.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    adjacency: Tensor

    def __init__(
        self,
        n_features: int,
        p_edge: float = 0.1,
        adjacency: Tensor | np.ndarray | None = None,
        beta: float = 1.0,
        p_active: list[float] | Tensor | None = None,
        **kwargs,
    ):
        super().__init__(n_features, **kwargs)
        self.p_edge = p_edge
        self.beta = beta
        if p_active is None:
            self.p_active = torch.ones(n_features, device=self.device) / n_features
        else:
            self.p_active = torch.as_tensor(p_active)

        if adjacency is None:
            self.regenerate_dag()
        else:
            assert adjacency.shape == (n_features, n_features), (
                f"adjacency shape = {adjacency.shape} needs to equal (n_features, n_features)"
            )
            self.adjacency = torch.as_tensor(adjacency)

        self._build_parent_cache()

    def _generate_dag(self) -> Tensor:
        """Generate random DAG as upper triangular adjacency matrix."""
        adj = torch.triu(
            self._rand(self.n_features, self.n_features) < self.p_edge,
            diagonal=1,
        )
        return adj

    def regenerate_dag(self) -> None:
        """Generate a new random DAG structure."""
        self.adjacency = self._generate_dag()
        self._build_parent_cache()

    def _build_parent_cache(self) -> None:
        """Precompute padded parent tensor for vectorized sampling."""
        parent_lists = []
        parent_counts = []
        max_parents = 0
        for j in range(self.n_features):
            parents = self.adjacency[:, j].nonzero(as_tuple=True)[0]
            parent_lists.append(parents)
            parent_counts.append(len(parents))
            if len(parents) > max_parents:
                max_parents = len(parents)

        # Padded tensor: (n_features, max_parents), pad with 0 (arbitrary, masked out)
        max_parents = max(max_parents, 1)  # avoid zero-dim
        self._parent_padded = torch.zeros(
            self.n_features, max_parents, dtype=torch.long, device=self.device
        )
        self._parent_counts = torch.tensor(
            parent_counts, dtype=torch.long, device=self.device
        )
        self._has_parents_mask = self._parent_counts > 0

        for j, parents in enumerate(parent_lists):
            if len(parents) > 0:
                self._parent_padded[j, : len(parents)] = parents

    def sample(self, batch_size: int) -> Tensor:
        """Sample sparse activations via random walk to root (vectorized)."""
        values = torch.zeros(batch_size, self.n_features, device=self.device)

        seeds = self._randint(
            0,
            self.n_features,
            (batch_size,),
            p=self.p_active,
        )
        activations = self._rand(batch_size)

        batch_idx = torch.arange(batch_size, device=self.device)
        values[batch_idx, seeds] = activations

        current_nodes = seeds
        current_values = activations

        for _ in range(self.n_features):
            current_values = self.beta * current_values + (
                1.0 - self.beta
            ) * current_values * self._rand(batch_size)

            still_walking = self._has_parents_mask[current_nodes]  # (batch_size,)
            if not still_walking.any():
                break

            active_counts = self._parent_counts[
                current_nodes[still_walking]
            ]  # (n_active,)
            random_idx = (
                self._rand(active_counts.shape) * active_counts
            ).long()  # uniform in [0, count)

            active_nodes = current_nodes[still_walking]
            chosen_parents = self._parent_padded[
                active_nodes, random_idx
            ]  # (n_active,)

            # Update
            next_nodes = current_nodes.clone()
            next_nodes[still_walking] = chosen_parents

            active_idx = batch_idx[still_walking]
            values[active_idx, chosen_parents] += current_values[still_walking]
            current_nodes = next_nodes

        return values

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.adjacency = self.adjacency.to(device)
        self._parent_padded = self._parent_padded.to(device)
        self._parent_counts = self._parent_counts.to(device)
        self._has_parents_mask = self._has_parents_mask.to(device)
        return self
