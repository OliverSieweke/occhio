"""DAG-based distribution module."""

from .base import Distribution
from torch import Tensor
import torch


class DAGDistribution(Distribution):
    """
    DAG-structured distribution with simple binary activation.
    A node is active if any parent is active and a coin flip succeeds.
    Actual values are Uniform[0,1] when active, 0 otherwise.
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
    """
    DAG-structured distribution with Noisy-OR propagation.

    Structure:
    - Nodes are indexed 0, 1, ..., n_features-1
    - causal[i, j] = True means edge i → j (i is parent of j)
    - Upper triangular ensures DAG (processing in index order is topological)

    Sampling:
    - Root nodes (no parents) fire with probability p_active
    - Non-root nodes fire with probability: 1 - prod_{active parent j}(1 - v_j)
      where v_j is the realized value of parent j
    - Values are Uniform[0,1] when active, 0 otherwise

    Semantics: activation magnitude = causal influence. A parent with v=0.9
    almost certainly triggers its children; v=0.1 rarely does.
    """

    def __init__(
        self,
        n_features: int,
        p_active: float = 0.1,
        p_edge: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            n_features: Number of nodes/features
            p_active: Probability that root nodes fire
            p_edge: Probability of edge i → j existing (for i < j)
        """
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
    """
    DAG-structured distribution with random-walk-to-root activation.

    Sampling:
    1. Pick one node uniformly at random
    2. Activate it with value ~ Uniform(0,1)
    3. Random walk upward: at each node, pick one parent uniformly,
       activate it with decayed value (beta * current), repeat until root
    4. If a node has multiple parents, one is chosen uniformly at random

    Produces maximally sparse activations: exactly one path from a
    random node to a root is active per sample.
    """

    def __init__(
        self,
        n_features: int,
        p_edge: float = 0.1,
        beta: float = 1.0,
        p_active: list[float] | Tensor | None = None,
        **kwargs,
    ):
        """
        Args:
            n_features: Number of nodes/features
            p_edge: Probability of edge i -> j existing (for i < j)
            beta: Multiplicative decay per step upward
        """
        super().__init__(n_features, **kwargs)
        self.p_edge = p_edge
        self.beta = beta
        if p_active is None:
            self.p_active = torch.ones(n_features, device=self.device) / n_features
        else:
            self.p_active = torch.Tensor(p_active)

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
            current_values = current_values * self.beta

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
