"""Hierarchical tree-based sparse distribution."""

from .base import Distribution
from torch import Tensor
import torch
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Node in the feature hierarchy."""

    index: int
    depth: int
    parent: int | None  # None for root
    children: list[int]


class HierarchicalSparse(Distribution):
    """Sparse distribution with tree-structured dependencies.

    Sampling process:
    1. Root node always fires
    2. For each non-root node: fires with probability p(depth) IF parent fired
    3. Values are Uniform([0,1]) when active

    Tree structure is fixed at init (call generate_new_tree to resample).
    """

    def __init__(
        self,
        n_features: int,
        p_base: float = 0.8,
        depth_decay: float = 0.9,
        p_by_depth: list[float] | None = None,
        max_children: int = 5,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        """
        Args:
            n_features: Number of nodes/features in the tree
            p_base: Base firing probability (at depth 1)
            depth_decay: Geometric decay per depth level. p(d) = p_base * decay^(d-1)
            p_by_depth: If provided, overrides p_base/depth_decay. p_by_depth[d] is
                        firing prob at depth d (depth 0 = root, always fires).
                        Extended with last value if tree is deeper.
            max_children: Maximum children per node during tree generation
        """
        super().__init__(n_features, device, generator)

        self.p_base = p_base
        self.depth_decay = depth_decay
        self.p_by_depth = p_by_depth
        self.max_children = max_children

        # Tree structure (set by generate_new_tree)
        self.nodes: list[TreeNode] = []
        self.max_depth: int = 0
        self.depth_indices: list[list[int]] = []  # depth -> list of node indices
        self.parent_tensor: Tensor = torch.empty(0)

        self.generate_new_tree()

    def generate_new_tree(self) -> None:
        """Generate a new random tree structure.

        Uses random recursive algorithm: BFS from root, each node gets
        Uniform(0, max_children) children until we have n_features nodes.
        """
        self.nodes = []

        # Root node
        root = TreeNode(index=0, depth=0, parent=None, children=[])
        self.nodes.append(root)

        queue = [0]
        next_index = 1

        while next_index < self.n_features and queue:
            parent_idx = queue.pop(0)
            parent_node = self.nodes[parent_idx]

            # How many children? Random up to max, but don't exceed n_features
            remaining = self.n_features - next_index
            max_possible = min(self.max_children, remaining)

            if max_possible > 0:
                n_children = int(self._randint(1, max_possible + 1, (1, )).item())

                for _ in range(n_children):
                    if next_index >= self.n_features:
                        break

                    child = TreeNode(
                        index=next_index,
                        depth=parent_node.depth + 1,
                        parent=parent_idx,
                        children=[],
                    )
                    self.nodes.append(child)
                    parent_node.children.append(next_index)
                    queue.append(next_index)
                    next_index += 1

        # Build auxiliary structures
        self._build_auxiliary_structures()

    def _build_auxiliary_structures(self) -> None:
        """Build tensors and indices for efficient sampling."""
        self.max_depth = max(node.depth for node in self.nodes)

        # Group nodes by depth
        self.depth_indices = [[] for _ in range(self.max_depth + 1)]
        for node in self.nodes:
            self.depth_indices[node.depth].append(node.index)

        parents = []
        for node in self.nodes:
            parents.append(node.parent if node.parent is not None else 0)
        self.parent_tensor = torch.tensor(parents, device=self.device, dtype=torch.long)

    def _get_p_fire(self, depth: int) -> float:
        """Get firing probability at given depth."""
        if depth == 0:
            return 1.0  # Root always fires

        if self.p_by_depth is not None:
            if depth < len(self.p_by_depth):
                return self.p_by_depth[depth]
            else:
                return self.p_by_depth[-1]

        return self.p_base * (self.depth_decay ** (depth - 1))

    def sample(self, batch_size: int) -> Tensor:
        """Sample from the hierarchical distribution.

        Returns (batch_size, n_features) tensor.
        """
        active = torch.zeros(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )

        for depth in range(self.max_depth + 1):
            indices = self.depth_indices[depth]
            if not indices:
                continue

            indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)

            if depth == 0:
                active[:, indices_tensor] = True
            else:
                p_fire = self._get_p_fire(depth)

                parent_indices = self.parent_tensor[indices_tensor]
                parent_active = active[:, parent_indices]

                fires = self._rand(batch_size, len(indices)) < p_fire
                active[:, indices_tensor] = parent_active & fires

        values = self._rand(batch_size, self.n_features)
        return active.float() * values

    def get_tree_stats(self) -> dict:
        """Return statistics about the current tree structure."""
        depths = [node.depth for node in self.nodes]
        n_children = [len(node.children) for node in self.nodes]
        n_leaves = sum(1 for node in self.nodes if not node.children)

        return {
            "n_nodes": self.n_features,
            "max_depth": self.max_depth,
            "mean_depth": sum(depths) / len(depths),
            "n_leaves": n_leaves,
            "mean_children": sum(n_children) / len(n_children),
            "max_children": max(n_children),
            "depth_distribution": {
                d: len(self.depth_indices[d]) for d in range(self.max_depth + 1)
            },
        }

    def get_expected_active(self) -> Tensor:
        """Compute expected number of active features per sample.

        Returns tensor of shape (n_features,) with marginal activation probabilities.
        """
        p_active = torch.zeros(self.n_features, device=self.device)

        for depth in range(self.max_depth + 1):
            for idx in self.depth_indices[depth]:
                if depth == 0:
                    p_active[idx] = 1.0
                else:
                    parent_idx = int(self.parent_tensor[idx].item())
                    p_fire = self._get_p_fire(depth)
                    p_active[idx] = p_active[parent_idx] * p_fire

        return p_active

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.parent_tensor = self.parent_tensor.to(device)
        return self
