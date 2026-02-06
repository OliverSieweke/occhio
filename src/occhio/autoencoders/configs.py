# ABOUTME: Configuration objects for constructing autoencoders.
# ABOUTME: Encapsulates dimensionality, activation, and tying preferences.

from dataclasses import dataclass
from typing import Sequence

from occhio.autoencoder import ActivationLike


class AutoEncoderConfig():
    pass

@dataclass
class LinearAutoEncoderConfig(AutoEncoderConfig):
    input_dim: int
    latent_dim: int
    activation: ActivationLike = "identity"
    tied_weights: bool = False
    bias: bool = True


@dataclass
class DeepAutoencoderConfig(AutoencoderConfig):
    dimensions: Sequence[int]
    hidden_activation: ActivationLike = "identity"
    tied_weights: bool = False
    validate_shapes: bool = False
    bias: bool = True
