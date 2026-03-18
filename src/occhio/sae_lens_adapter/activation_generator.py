import torch
from sae_lens.synthetic import ActivationGenerator
from torch import Tensor, nn

from occhio.distributions import Distribution


class ActivationGeneratorWrapper(ActivationGenerator):
    """A SAE Lens-compatible ActivationGenerator for wrapping an occhio Distribution.

    This class partially implements an ActivationGenerator, matching the
    interface of sae_lens.synthetic.ActivationGenerator for the core sampling
    functionality. This allows occhio distributions to be used with SAE Lens pipelines.

    Note:
        The statistics attributes (firing_probabilities, mean_firing_magnitudes, etc.)
        raise NotImplementedError since occhio distributions don't uniformly expose
        these.

    Attributes:
        num_features: Number of features in the distribution.
    """

    num_features: int

    def __init__(self, distribution: Distribution):
        """Create an ActivationGeneratorWrapper from an occhio Distribution.

        Args:
            distribution: An occhio Distribution instance.
        """
        nn.Module.__init__(self)

        self._distribution = distribution
        self.num_features = distribution.n_features

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tensor:
        """Generate a batch of feature activations.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Tensor of shape [batch_size, num_features] with activations.
        """
        return self._distribution.sample(batch_size)

    def forward(self, batch_size: int) -> Tensor:
        """Generate a batch of feature activations (alias for sample)."""
        return self.sample(batch_size)

    @property
    def firing_probabilities(self) -> Tensor:
        """Per-feature firing probabilities (not supported by all distributions)."""
        raise NotImplementedError(
            f"firing_probabilities is not available for {type(self._distribution).__name__}. "
            "Use ActivationGeneratorWithStats or a distribution that exposes sparsity."
        )

    @property
    def mean_firing_magnitudes(self) -> Tensor:
        """Per-feature mean firing magnitudes (not supported by all distributions)."""
        raise NotImplementedError(
            f"mean_firing_magnitudes is not available for {type(self._distribution).__name__}. "
            "Use ActivationGeneratorWithStats or a distribution that exposes magnitude stats."
        )

    @property
    def std_firing_magnitudes(self) -> Tensor:
        """Per-feature std of firing magnitudes (not supported by all distributions)."""
        raise NotImplementedError(
            f"std_firing_magnitudes is not available for {type(self._distribution).__name__}. "
            "Use ActivationGeneratorWithStats or a distribution that exposes magnitude stats."
        )

    @property
    def correlation_matrix(self) -> Tensor | None:
        """Correlation matrix (not supported by all distributions)."""
        raise NotImplementedError(
            f"correlation_matrix is not available for {type(self._distribution).__name__}. "
            "Use ActivationGeneratorWithStats for distributions with correlation structure."
        )
