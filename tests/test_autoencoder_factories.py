# ABOUTME: Tests factory helpers for building configured autoencoders.
# ABOUTME: Ensures linear and deep factories tie weights and honor configs.

import unittest

import torch
from torch import nn

from occhio.autoencoder import TiedLinear
from occhio.autoencoders import (
    DeepAutoencoderConfig,
    LinearAutoencoderConfig,
    create_autoencoder,
    deep_autoencoder,
    linear_autoencoder,
)


class AutoEncoderFactoryTests(unittest.TestCase):
    def test_linear_autoencoder_builds_from_config(self):
        cfg = LinearAutoencoderConfig(input_dim=4, latent_dim=2, activation="relu", tied_weights=False)
        ae = linear_autoencoder(cfg)
        x = torch.randn(3, 4)
        out = ae(x)
        self.assertEqual(out.shape, (3, 4))
        relu_layers = [m for m in ae.encoder if isinstance(m, nn.ReLU)]
        self.assertEqual(len(relu_layers), 1)

    def test_linear_autoencoder_uses_tiedlinear_when_requested(self):
        cfg = LinearAutoencoderConfig(input_dim=6, latent_dim=3, activation="relu", tied_weights=True)
        ae = linear_autoencoder(cfg)
        self.assertIsInstance(ae.decoder[0], TiedLinear)
        self.assertIs(ae.decoder[0].tied_to, ae.encoder[0])

    def test_deep_autoencoder_supports_weight_tying_for_all_layers(self):
        cfg = DeepAutoencoderConfig(dimensions=[10, 6, 4, 2], hidden_activation="gelu", tied_weights=True)
        ae = deep_autoencoder(cfg)
        encoder_linears = [m for m in ae.encoder if isinstance(m, nn.Linear)]
        decoder_tied = [m for m in ae.decoder if isinstance(m, TiedLinear)]
        self.assertEqual(len(encoder_linears), len(decoder_tied))
        for enc, dec in zip(reversed(encoder_linears), decoder_tied):
            self.assertIs(dec.tied_to, enc)

    def test_deep_autoencoder_validates_monotonic_when_requested(self):
        cfg = DeepAutoencoderConfig(dimensions=[4, 5, 4], hidden_activation="relu", tied_weights=False, validate_shapes=True)
        with self.assertRaises(ValueError):
            deep_autoencoder(cfg)

    def test_create_autoencoder_wraps_modules(self):
        encoder = nn.Linear(3, 2)
        decoder = nn.Linear(2, 3)
        ae = create_autoencoder(encoder=encoder, decoder=decoder)
        x = torch.randn(4, 3)
        out = ae(x)
        self.assertEqual(out.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
