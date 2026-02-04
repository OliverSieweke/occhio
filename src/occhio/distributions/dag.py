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
            # Parents of node i are nodes j where causal[j, i] = True
            parent_mask = self.adjacency[:, i]
            has_parents = parent_mask.any()

            if not has_parents:
                # Root node: fires with probability p_active
                active[:, i] = self._rand(batch_size, 1).squeeze(-1) < self.p_active[i]
            else:
                # Non-root: fires if (any parent active) AND (coin flip)
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
