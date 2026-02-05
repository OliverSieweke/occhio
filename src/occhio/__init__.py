# ABOUTME: Package exports for occhio.
# ABOUTME: Re-exports autoencoder conveniences for users.

from .autoencoder import AutoEncoder, TiedLinear
from .autoencoders import (
    DeepAutoencoderConfig,
    LinearAutoencoderConfig,
    create_autoencoder,
    deep_autoencoder,
    linear_autoencoder,
)

__all__ = [
    "AutoEncoder",
    "TiedLinear",
    "DeepAutoencoderConfig",
    "LinearAutoencoderConfig",
    "create_autoencoder",
    "deep_autoencoder",
    "linear_autoencoder",
]
