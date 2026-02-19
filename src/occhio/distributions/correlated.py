"""Simple correlation structures."""

import torch
from torch import Tensor
from .base import Distribution


class HierarchicalPairs(Distribution):
    """Hierarchical pair-based distribution where secondary features follow primary features.

    Features are organized in pairs (2i, 2i+1). The primary feature (2i) activates
    with probability ``p_active``. If active, the secondary feature (2i+1) activates
    independently with probability ``p_follow``. This creates a shallow hierarchical
    dependency structure.

    Args:
        n_features: Dimensionality of the sample space (must be even).
        p_active: Probability that a primary feature (2i) activates.
            Scalar or per-feature.
        p_follow: Conditional probability that the secondary feature (2i+1)
            activates given that the primary feature is active. Defaults to 0.5.
            Scalar or per-feature.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.

    Note:
        The correlation between paired features is:

        .. math::
            \\text{corr} = \\sqrt{\\frac{p_f (1 - p_a)}{1 - p_f p_a}}

        As ``p_a → 0``, the correlation approaches ``√p_f``.

        To achieve a target correlation ``c``, set:

        .. math::
            p_f = \\frac{c^2}{1 + p_a(c^2 - 1)}
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor,
        p_follow: float | list[float] | Tensor = 0.5,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even `n_features` for pairs."
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.p_follow = self._broadcast(p_follow)

    def sample(self, batch_size: int) -> Tensor:
        n_pairs = self.n_features // 2
        primary_mask = self._rand(batch_size, n_pairs) < self.p_active[0::2]
        secondary_mask = primary_mask & (
            self._rand(batch_size, n_pairs) < self.p_follow[1::2]
        )

        mask = torch.empty(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        mask[:, 0::2] = primary_mask
        mask[:, 1::2] = secondary_mask

        values = self._rand(batch_size, self.n_features)
        return mask * values

class ScaledHierarchicalPairs(Distribution):
    """Hierarchical pair-based distribution with value scaling from parent to child.

    Similar to :class:`HierarchicalPairs`, but the secondary feature's value is
    scaled by the primary feature's value. Features are organized in pairs (2i, 2i+1).
    The primary feature (2i) activates with probability ``p_active`` and takes a
    value ``v ~ Uniform(0, 1)``. If active, the secondary feature (2i+1) activates
    with probability ``p_follow`` and takes value ``U * v`` where ``U ~ Uniform(0, 1)``.

    This creates a hierarchical dependency where the child's magnitude is constrained
    by the parent's magnitude, representing a form of causal influence.

    Args:
        n_features: Dimensionality of the sample space (must be even).
        p_active: Probability that a primary feature (2i) activates.
            Scalar or per-feature.
        p_follow: Conditional probability that the secondary feature (2i+1)
            activates given that the primary feature is active. Defaults to 0.5.
            Scalar or per-feature.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor,
        p_follow: float | list[float] | Tensor = 0.5,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even `n_features` for pairs."
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.p_follow = self._broadcast(p_follow)

    def sample(self, batch_size: int) -> Tensor:
        n_pairs = self.n_features // 2
        primary_mask = self._rand(batch_size, n_pairs) < self.p_active[0::2]
        secondary_mask = primary_mask & (
            self._rand(batch_size, n_pairs) < self.p_follow[1::2]
        )

        primary_values = self._rand(batch_size, n_pairs)
        secondary_values = self._rand(batch_size, n_pairs) * primary_values

        out = torch.zeros(batch_size, self.n_features, device=self.device)
        out[:, 0::2] = primary_mask * primary_values
        out[:, 1::2] = secondary_mask * secondary_values
        return out


class CorrelatedPairs(Distribution):
    """Pair-based distribution with correlated but independent individual activations.

    Features are organized in pairs (2i, 2i+1). First, each pair activates with
    probability ``p_active``. If a pair is active, each individual feature within
    the pair independently activates with probability ``p_individual``. This creates
    positive correlation between paired features while allowing for independent
    variation within active pairs.

    Args:
        n_features: Dimensionality of the sample space (must be even).
        p_active: Probability that a pair becomes active. Defaults to 0.1.
            Scalar or per-feature (uses even indices for pair probabilities).
        p_individual: Conditional probability that each individual feature
            within an active pair activates. Defaults to 0.7.
            Scalar or per-feature.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.

    Note:
        The correlation between paired features is:

        .. math::
            \\text{corr}(X_{2i}, X_{2i+1}) = \\frac{p_i(1 - p_a)}{1 - p_a p_i}

        As ``p_a → 0``, the correlation approaches ``p_i``.

        To achieve a target correlation ``c``, set:

        .. math::
            p_i = \\frac{c}{1 - p_a + c p_a}
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor = 0.1,
        p_individual: float | list[float] | Tensor = 0.7,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even `n_features` for pairs."
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)
        self.p_individual = self._broadcast(p_individual)

    def sample(self, batch_size: int) -> Tensor:
        n_pairs = self.n_features // 2
        primary_mask = self._rand(batch_size, n_pairs) < self.p_active[0::2]

        mask = torch.empty(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        mask[:, 0::2] = primary_mask * (
            self._rand(batch_size, n_pairs) < self.p_individual[0::2]
        )
        mask[:, 1::2] = primary_mask * (
            self._rand(batch_size, n_pairs) < self.p_individual[1::2]
        )

        values = self._rand(batch_size, self.n_features)
        return mask * values

class AnticorrelatedPairs(Distribution):
    """Pair-based distribution with mutually exclusive (anticorrelated) features.

    Features are organized in pairs (2i, 2i+1) where at most one feature in each
    pair can be active per sample. Each pair activates with probability ``p_active``,
    and if active, exactly one of the two features is chosen uniformly at random.
    This creates maximal negative correlation (mutual exclusivity) between paired
    features.

    Args:
        n_features: Dimensionality of the sample space (must be even).
        p_active: Probability that a pair activates (with exactly one feature selected).
            Scalar or per-feature (uses even indices for pair probabilities).
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even n_features for pairs"
        super().__init__(n_features, **kwargs)
        self.p_active = self._broadcast(p_active)

    def sample(self, batch_size: int) -> Tensor:
        n_pairs: int = self.n_features // 2

        pair_active = self._rand(batch_size, n_pairs) < self.p_active[0::2]

        which_one = self._randint(0, 2, (batch_size, n_pairs))

        mask = torch.zeros(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        mask[:, 0::2] = pair_active & (which_one == 0)
        mask[:, 1::2] = pair_active & (which_one == 1)

        values = self._rand(batch_size, self.n_features)
        return mask * values