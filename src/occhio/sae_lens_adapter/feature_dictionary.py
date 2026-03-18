from sae_lens.synthetic import FeatureDictionary
from torch import Tensor, nn

from occhio import AutoEncoderBase


class FeatureDictionaryWrapper(FeatureDictionary):
    """A SAE Lens-compatible FeatureDictionary for wrapping occhio AutoEncoder.

    This class implements a FeatureDictionary, matching the interface of
    sae_lens.synthetic.FeatureDictionary. This allows occhio ToyModel embeddings
    to be used with SAE Lens pipelines.

    Attributes:
        num_features: Number of features in the dictionary.
        hidden_dim: Dimensionality of the hidden space.
        feature_vectors: Parameter of shape [num_features, hidden_dim] containing the
            feature embedding vectors.
        bias: Parameter of shape [hidden_dim] containing the bias term.
    """

    num_features: int
    hidden_dim: int
    feature_vectors: nn.Parameter
    bias: nn.Parameter

    def __init__(self, auto_encoder: AutoEncoderBase):
        """Create a FeatureDictionaryWrapper from ToyModel.

        Takes the weight matrix from the provided auto-encoder, transposes it to match
        the SAE Lens FeatureDictionary format.

        :param auto_encoder: The auto_encoder containing the embedding matrix
        :type auto_encoder: AutoEncoderBase

        """
        nn.Module.__init__(self)
        self._auto_encoder = auto_encoder
        # The SAE Lens FeatureDictionary works with the transposed matrix.
        self.num_features, self.hidden_dim = auto_encoder.feature_vectors.T.shape
        self.feature_vectors = nn.Parameter(auto_encoder.feature_vectors)

    def forward(self, feature_activations: Tensor) -> Tensor:
        """Convert feature activations to hidden activations.

        Args:
            feature_activations: Tensor of shape [batch, num_features] containing
                sparse feature activation values.

        Returns:
            Tensor of shape [batch, hidden_dim] containing dense hidden activations.
        """

        return self._auto_encoder.encode(feature_activations)
