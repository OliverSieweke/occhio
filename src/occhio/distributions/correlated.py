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

    Exactly two of the four parameters must be specified:

    Args:
        n_features: Dimensionality of the sample space (must be even).
        p_active: Probability that a pair becomes active.
            Scalar or per-feature (uses even indices for pair probabilities).
        p_individual: Conditional probability that each individual feature
            within an active pair activates.
            Scalar or per-feature.
        correlation: Target correlation between paired features.
        density: Effective density, defined as ``p_active * p_individual``.
        device: Torch device for all generated tensors.
        generator: Optional ``torch.Generator`` for deterministic sampling.

    Note:
        The four parameters are related by:

        .. math::
            s = p_a \\cdot p_i, \\quad
            \\text{corr}(X_{2i}, X_{2i+1}) = \\frac{p_i(1 - p_a)}{1 - p_a p_i}

        The full inversion table (given any two, solve for the other two):

        +------------------+------------------+---------------------------------------+---------------------------------------+
        | Known 1          | Known 2          | Formula for ``p_i``                   | Formula for ``p_a``                   |
        +==================+==================+=======================================+=======================================+
        | ``p_a``          | ``p_i``          | —                                     | —                                     |
        +------------------+------------------+---------------------------------------+---------------------------------------+
        | ``p_a``          | ``correlation``  | ``c / (1 - p_a + c * p_a)``           | —                                     |
        +------------------+------------------+---------------------------------------+---------------------------------------+
        | ``p_a``          | ``density``     | ``d / p_a``                           | —                                     |
        +------------------+------------------+---------------------------------------+---------------------------------------+
        | ``p_i``          | ``correlation``  | —                                     | ``(p_i - c) / (p_i * (1 - c))``       |
        +------------------+------------------+---------------------------------------+---------------------------------------+
        | ``p_i``          | ``density``     | —                                     | ``d / p_i``                           |
        +------------------+------------------+---------------------------------------+---------------------------------------+
        | ``correlation``  | ``density``     | ``c + d * (1 - c)``                   | ``d / (c + d * (1 - c))``             |
        +------------------+------------------+---------------------------------------+---------------------------------------+
    """

    def __init__(
        self,
        n_features: int,
        p_active: float | list[float] | Tensor | None = None,
        p_individual: float | list[float] | Tensor | None = None,
        correlation: float | list[float] | Tensor | None = None,
        density: float | list[float] | Tensor | None = None,
        **kwargs,
    ):
        assert n_features % 2 == 0, "Need even `n_features` for pairs."
        super().__init__(n_features, **kwargs)

        given = {
            "p_active": p_active,
            "p_individual": p_individual,
            "correlation": correlation,
            "density": density,
        }
        provided = {k: v for k, v in given.items() if v is not None}
        if len(provided) != 2:
            raise ValueError(
                f"Exactly two of {{p_active, p_individual, correlation, density}} "
                f"must be specified; got {list(provided.keys()) or 'none'}."
            )

        pa = self._broadcast(p_active) if p_active is not None else None
        pi = self._broadcast(p_individual) if p_individual is not None else None
        c = self._broadcast(correlation) if correlation is not None else None
        s = self._broadcast(density) if density is not None else None

        match tuple(sorted(provided)):
            case ("p_active", "p_individual"):
                pass
            case ("correlation", "p_active"):
                pi: Tensor = c / (1 - pa + c * pa)  # ty:ignore
            case ("p_active", "density"):
                pi: Tensor = s / pa  # ty:ignore
            case ("correlation", "p_individual"):
                pa: Tensor = (pi - c) / (pi * (1 - c))  # ty:ignore
            case ("p_individual", "density"):
                pa: Tensor = s / pi  # ty:ignore
            case ("correlation", "density"):
                pi: Tensor = c + s * (1 - c)  # ty:ignore
                pa: Tensor = s / pi
            case _:
                raise ValueError(f"Unhandled parameter combination: {list(provided)}")

        self.p_active: Tensor = pa
        self.p_individual: Tensor = pi

        # Validate after solving
        for name, val in [
            ("p_active", self.p_active),
            ("p_individual", self.p_individual),
        ]:
            if not (val >= 0).all() or not (val <= 1).all():
                raise ValueError(
                    f"Derived `{name}` has values outside [0, 1]: {val}. "
                    f"Check that your input combination is achievable."
                )

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
