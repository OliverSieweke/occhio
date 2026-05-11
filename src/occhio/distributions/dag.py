"""DAG-based distribution module."""

from .base import Distribution
from torch import Tensor
from typing import Literal
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
        # CUDA perf: pre-generate all random values to avoid n_features tiny
        # _rand() kernel launches with varying sizes. Instead, use a single
        # (batch, n_features) draw and mask-assign.
        all_values = self._rand(batch_size, self.n_features)

        for j in range(self.n_features):
            if not self._has_parents[j]:
                fires = self._rand(batch_size) < self.p_active[j]
            else:
                parent_idx = self._parent_indices[j]
                parent_values = values[:, parent_idx]  # (batch_size, n_parents)

                survival_prob = (1 - parent_values).prod(dim=1)  # (batch_size,)
                fire_prob = 1 - survival_prob

                fires = self._rand(batch_size) < fire_prob

            # CUDA perf: avoid .sum().item() sync per feature by using
            # mask assignment unconditionally. When fires is all-False,
            # the masked assignment is a no-op.
            values[fires, j] = all_values[fires, j]

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
        n_firings: Number of independent walks per sample. Each walk picks its
            own starting node and propagates to a root; the resulting
            activations are summed into a single sample. Defaults to 1.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    adjacency: Tensor

    def __init__(
        self,
        n_features: int,
        p_edge: float = 0.5,
        adjacency: Tensor | np.ndarray | None = None,
        beta: float = 1.0,
        p_active: list[float] | Tensor | None = None,
        shrinking: bool = True,
        n_firings: int = 1,
        **kwargs,
    ):
        super().__init__(n_features, **kwargs)
        self.p_edge = p_edge
        self.beta = beta
        self.shrinking = shrinking
        self.n_firings = n_firings
        if p_active is None:
            self.p_active = torch.ones(n_features, device=self.device) / n_features
        else:
            self.p_active = torch.as_tensor(p_active)

        if adjacency is None:
            self.regenerate_dag()
        else:
            assert adjacency.shape == (
                n_features,
                n_features,
            ), (
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
        """Sample sparse activations via random walk to root (vectorized).

        When ``n_firings > 1``, runs that many independent walks per requested
        sample and sums their contributions.
        """
        effective_batch = batch_size * self.n_firings
        values = torch.zeros(effective_batch, self.n_features, device=self.device)

        seeds = self._randint(
            0,
            self.n_features,
            (effective_batch,),
            p=self.p_active,
        )
        activations = self._rand(effective_batch)

        batch_idx = torch.arange(effective_batch, device=self.device)
        values[batch_idx, seeds] = activations

        current_nodes = seeds
        current_values = activations

        for _ in range(self.n_features):
            current_values = self.beta * current_values + (1.0 - self.beta) * (
                current_values**self.shrinking
            ) * self._rand(effective_batch)

            still_walking = self._has_parents_mask[current_nodes]
            if not still_walking.any():
                break

            active_counts = self._parent_counts[current_nodes[still_walking]]
            random_idx = (self._rand(active_counts.shape) * active_counts).long()

            active_nodes = current_nodes[still_walking]
            chosen_parents = self._parent_padded[active_nodes, random_idx]

            next_nodes = current_nodes.clone()
            next_nodes[still_walking] = chosen_parents

            active_idx = batch_idx[still_walking]
            values[active_idx, chosen_parents] += current_values[still_walking]
            current_nodes = next_nodes

        if self.n_firings == 1:
            return values
        return values.view(batch_size, self.n_firings, self.n_features).sum(dim=1)

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.adjacency = self.adjacency.to(device)
        self._parent_padded = self._parent_padded.to(device)
        self._parent_counts = self._parent_counts.to(device)
        self._has_parents_mask = self._has_parents_mask.to(device)
        return self

    def print_graph(self, labels=None, center: int | None = None):
        n = self.adjacency.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n)]
        if center is not None:
            parents = [labels[i] for i in range(n) if self.adjacency[i, center]]
            children = [labels[j] for j in range(n) if self.adjacency[center, j]]
            print(f"  {labels[center]}")
            print(f"    parents:  {', '.join(parents) or '(none)'}")
            print(f"    children: {', '.join(children) or '(none)'}")
            return
        for i in range(n):
            targets = [labels[j] for j in range(n) if self.adjacency[i, j]]
            if targets:
                print(f"  {labels[i]} → {', '.join(targets)}")
            else:
                print(f"  {labels[i]}  (no outgoing)")

    def print_sources_and_sinks(self, labels=None):
        n = self.adjacency.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n)]

        has_outgoing = self.adjacency.any(dim=1)
        has_incoming = self.adjacency.any(dim=0)

        sources = [
            labels[i] for i in range(n) if has_outgoing[i] and not has_incoming[i]
        ]
        sinks = [labels[i] for i in range(n) if has_incoming[i] and not has_outgoing[i]]
        isolated = [
            labels[i] for i in range(n) if not has_outgoing[i] and not has_incoming[i]
        ]

        print(f"  Sources:  {', '.join(sources) or '(none)'}")
        print(f"  Sinks:    {', '.join(sinks) or '(none)'}")
        if isolated:
            print(f"  Isolated: {', '.join(isolated)}")

    def print_connected_components(self, labels=None):
        n = self.adjacency.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n)]

        # Build undirected adjacency (ignore direction)
        undirected = self.adjacency | self.adjacency.T

        visited = [False] * n
        components = []

        def bfs(start):
            queue = [start]
            visited[start] = True
            comp = [start]
            while queue:
                node = queue.pop(0)
                for j in range(n):
                    if undirected[node, j] and not visited[j]:
                        visited[j] = True
                        queue.append(j)
                        comp.append(j)
            return comp

        for i in range(n):
            if not visited[i]:
                components.append(bfs(i))

        print(f"  {len(components)} connected component(s):")
        for k, comp in enumerate(components):
            names = [labels[i] for i in comp]
            print(f"    Component {k}: {', '.join(names)}")


class PreferentialAttachment(Distribution):
    """Digraph with power-law in-degree distribution and one-step cascade propagation.

    Node **0** has the highest expected in-degree (many nodes point to it), and
    node **N-1** has the lowest.  The graph is a general directed graph — cycles
    are allowed and harmless because propagation is exactly one step.

    ``adjacency[j, i] = True`` encodes the directed edge j → i.  The per-edge
    connection probability for edges targeting node *i* follows a power law:

    .. math::
        p_i = \\min\\!\\left(1,\\; p_{\\text{edge}} \\cdot
              \\left(\\frac{N - i}{N}\\right)^{\\!\\alpha}\\right)

    so node 0 always gets connection probability ``p_edge`` (the maximum), and
    later nodes receive proportionally fewer in-edges as ``alpha`` grows.
    ``alpha=0`` recovers Erdős-Rényi with uniform edge probability.

    Sampling:

    1. Each node fires **independently** with probability ``p_active`` (like
       :class:`SparseUniform`).
    2. **One-step cascade**: every independently-fired node tries to trigger each
       of its out-neighbours.  Each out-edge fires the child independently with
       probability ``p_child``.  With *k* independently-fired parents, the
       combined probability that a child is cascade-triggered is
       :math:`1 - (1 - p_{\\text{child}})^{k}` (noisy-OR).
    3. Active nodes (independent OR cascade) take values drawn from ``value_dist``.

    Args:
        n_features: Number of nodes.
        alpha: Power-law exponent for the in-degree distribution.
            ``alpha=0`` gives Erdős-Rényi.  ``alpha=1`` makes node 0 receive
            roughly *N* times more in-edges than node N-1.  Default: ``1.0``.
        p_edge: Base edge probability (equals the per-edge probability for
            in-edges to node 0).  Controls overall graph density.
            Default: ``0.1``.
        p_active: Unconditional firing probability.  Scalar or per-feature.
            Default: ``0.05``.
        p_child: Per-edge cascade probability. Either a scalar ``float`` where
            every edge shares the same cascade probability, or a
            ``tuple(low, high)`` to sample an individual cascade probability
            per edge uniformly from ``[low, high]`` when the graph is
            generated.  ``1.0`` = deterministic cascade; ``0.0`` = no cascade.
            Default: ``0.9``.
        value_dist: Value distribution for active nodes.
            ``'uniform'`` — Uniform(0, 1).  ``'exponential'`` — Exponential(1).
            Default: ``'uniform'``.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        p_edge: float = 0.1,
        p_active: float | list[float] | Tensor = 0.05,
        p_child: float | tuple[float, float] = 0.9,
        value_dist: Literal["uniform", "exponential"] = "uniform",
        **kwargs,
    ):
        super().__init__(n_features, **kwargs)
        self.alpha = alpha
        self.p_edge = p_edge
        self.p_active = self._broadcast(p_active)
        self.p_child = p_child
        self.value_dist = value_dist
        self.regenerate_graph()

    def _generate_graph(self) -> Tensor:
        """Generate adjacency matrix with power-law in-degree distribution.

        Returns a boolean tensor where ``adj[j, i]`` is ``True`` iff there is a
        directed edge j → i.  Column *i* has per-entry probability
        ``p_edge * ((N - i) / N) ** alpha``; self-loops are excluded.
        """
        N = self.n_features
        node_idx = torch.arange(N, device=self.device, dtype=torch.float)
        p_per_target = (self.p_edge * ((N - node_idx) / N).pow(self.alpha)).clamp(
            max=1.0
        )
        # rand[j, i] < p_per_target[i] for all j — broadcast across rows.
        adj = self._rand(N, N) < p_per_target.unsqueeze(0)
        adj.fill_diagonal_(False)
        return adj

    def _build_log_survival(self) -> None:
        """Build the per-edge log-survival matrix for cascade computation.

        When ``p_child`` is a scalar, ``_log_survival[j, i] = log(1 - p_child)``
        for every edge j → i.  When ``p_child`` is a ``(low, high)`` tuple,
        each edge gets an independently sampled cascade probability and the
        log-survival is computed per edge.  Non-edges are 0 so they contribute
        nothing to the sum.
        """
        if isinstance(self.p_child, tuple):
            lo, hi = self.p_child
            per_edge_p = lo + (hi - lo) * self._rand(self.n_features, self.n_features)
            self._log_survival = torch.log(1.0 - per_edge_p) * self.adjacency.float()
        else:
            self._log_survival = (
                torch.log(torch.tensor(1.0 - self.p_child, device=self.device))
                * self.adjacency.float()
            )

    def regenerate_graph(self) -> None:
        """Sample a new graph from the power-law digraph prior."""
        self.adjacency = self._generate_graph()
        self._build_log_survival()

    def sample(self, batch_size: int) -> Tensor:
        """Sample activations via independent firing and one-step cascade."""
        # Step 1: independent fires — (batch_size, N)
        ind_active = self._rand(batch_size, self.n_features) < self.p_active

        # Step 2: cascade via log-survival sums (handles both scalar and
        # per-edge p_child).  sum_log_surv[b, i] = Σ_j 1[j fired] · log(1 - p_{j→i})
        sum_log_surv = ind_active.float() @ self._log_survival
        cascade_prob = 1.0 - torch.exp(sum_log_surv)
        cascade_active = self._rand(batch_size, self.n_features) < cascade_prob

        active = ind_active | cascade_active

        if self.value_dist == "uniform":
            values = self._rand(batch_size, self.n_features)
        else:  # "exponential"
            u = self._rand(batch_size, self.n_features).clamp(max=1.0 - 1e-6)
            values = -torch.log(1.0 - u)

        return active.float() * values

    def in_degrees(self) -> Tensor:
        """Return the in-degree (number of parents) for each node."""
        return self.adjacency.float().sum(dim=0)

    def out_degrees(self) -> Tensor:
        """Return the out-degree (number of children) for each node."""
        return self.adjacency.float().sum(dim=1)

    def get_expected_activation(self, n_samples: int = 10_000) -> Tensor:
        """Estimate marginal activation probabilities via Monte Carlo."""
        samples = self.sample(n_samples)
        return (samples > 0).float().mean(dim=0)

    def print_graph(self, labels=None, center: int | None = None):
        n = self.adjacency.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n)]
        if center is not None:
            parents = [labels[i] for i in range(n) if self.adjacency[i, center]]
            children = [labels[j] for j in range(n) if self.adjacency[center, j]]
            print(f"  {labels[center]}")
            print(f"    parents:  {', '.join(parents) or '(none)'}")
            print(f"    children: {', '.join(children) or '(none)'}")
            return
        for i in range(n):
            targets = [labels[j] for j in range(n) if self.adjacency[i, j]]
            if targets:
                print(f"  {labels[i]} → {', '.join(targets)}")
            else:
                print(f"  {labels[i]}  (no outgoing)")

    def to(self, device: torch.device | str):
        """Move distribution to device."""
        super().to(device)
        self.adjacency = self.adjacency.to(device)
        self._log_survival = self._log_survival.to(device)
        return self
