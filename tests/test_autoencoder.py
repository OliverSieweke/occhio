# ABOUTME: Tests for the AutoEncoder abstraction and helper factories.
# ABOUTME: Ensures activation handling, tying, factories, and examples behave.

import unittest

import torch
from torch import nn

from occhio.autoencoder import (
    AutoEncoder,
    TiedLinear,
    create_autoencoder,
    linear_autoencoder,
)


class ToyAutoEncoder(AutoEncoder):
    def __init__(self, activation="identity", tied_weights=False):
        self.input_dim = 4
        self.latent_dim = 2
        super().__init__(activation=activation, tied_weights=tied_weights)

    def create_encoder(self) -> nn.Module:
        return nn.Linear(self.input_dim, self.latent_dim, bias=True)

    def create_decoder(self) -> nn.Module:
        return nn.Linear(self.latent_dim, self.input_dim, bias=True)


class TiedBatchNormAE(AutoEncoder):
    def __init__(self):
        self.dim = 4
        super().__init__(activation="identity", tied_weights=True)

    def create_encoder(self) -> nn.Module:
        return nn.BatchNorm1d(self.dim, affine=True)

    def create_decoder(self) -> nn.Module:
        return nn.BatchNorm1d(self.dim, affine=True)


class AutoEncoderTests(unittest.TestCase):
    def test_autoencoder_is_abstract(self):
        with self.assertRaises(TypeError):
            AutoEncoder()  # type: ignore[arg-type]

    def test_forward_runs_with_subclass(self):
        ae = ToyAutoEncoder()
        x = torch.randn(3, 4)
        out = ae(x)
        self.assertEqual(out.shape, (3, 4))

    def test_activation_resolution_by_string(self):
        ae = ToyAutoEncoder(activation="GELU")
        x = torch.randn(2, 4)
        out = ae.encode(x)
        self.assertEqual(out.shape, (2, 2))

    def test_tied_weights_for_matching_modules(self):
        ae = TiedBatchNormAE()
        self.assertIs(ae.encoder.weight, ae.decoder.weight)
        x = torch.randn(5, 4)
        out = ae(x)
        self.assertEqual(out.shape, (5, 4))

    def test_linear_autoencoder_builds_multi_layer(self):
        ae = linear_autoencoder(dimensions=[8, 4, 2], hidden_activation="relu", tied_weights=False)
        x = torch.randn(1, 8)
        out = ae(x)
        self.assertEqual(out.shape, (1, 8))

    def test_linear_autoencoder_tied_uses_tiedlinear(self):
        ae = linear_autoencoder(dimensions=[6, 3], hidden_activation="relu", tied_weights=True)
        self.assertIsInstance(ae.decoder[0], TiedLinear)

    def test_linear_autoencoder_validates_monotonic(self):
        with self.assertRaises(ValueError):
            linear_autoencoder(dimensions=[4, 5, 4], validate_shapes=True)

    def test_create_autoencoder_wraps_modules(self):
        encoder = nn.Linear(3, 2)
        decoder = nn.Linear(2, 3)
        ae = create_autoencoder(encoder=encoder, decoder=decoder, activation="identity")
        x = torch.randn(4, 3)
        out = ae(x)
        self.assertEqual(out.shape, (4, 3))

    def test_W_exposes_encoder_and_decoder_params(self):
        ae = ToyAutoEncoder()
        weights = ae.W
        self.assertIn("encoder", weights)
        self.assertIn("decoder", weights)
        self.assertGreater(len(weights["encoder"]), 0)
        self.assertGreater(len(weights["decoder"]), 0)


if __name__ == "__main__":
    unittest.main()
