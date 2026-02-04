# ABOUTME: Tests for the AutoEncoder component in occhio.
# ABOUTME: Verifies defaults, pluggable modules, activation hook, validation, and tied-weight helper.

import unittest

import torch
from torch import nn

from occhio.autoencoder import AutoEncoder, deep_autoencoder, linear_autoencoder, TiedLinear


class AutoEncoderTests(unittest.TestCase):
    def test_forward_uses_custom_modules(self):
        encoder = nn.Sequential(nn.Linear(6, 4), nn.ReLU())
        decoder = nn.Linear(4, 6, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, activation="identity")

        x = torch.randn(5, 6)
        encoded = autoencoder.encode(x)
        decoded = autoencoder.forward(x)

        self.assertEqual(encoded.shape, (5, 4))
        self.assertEqual(decoded.shape, (5, 6))

    def test_activation_applied_after_encode(self):
        encoder = nn.Linear(3, 2, bias=False)
        decoder = nn.Linear(2, 3, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, activation="relu")

        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
            decoder.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

        x = torch.tensor([[-1.0, 2.0, -0.5]])
        output = autoencoder.forward(x)

        expected = torch.zeros(1, 3)
        self.assertTrue(torch.allclose(output, expected))

    def test_linear_autoencoder_factory_single_layer(self):
        autoencoder = linear_autoencoder(n_features=6, hidden_sizes=3, tied_weights=False)
        x = torch.randn(2, 6)
        out = autoencoder(x)
        self.assertEqual(out.shape, (2, 6))

    def test_tied_weights_without_decoder(self):
        encoder = nn.Linear(4, 2, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=None, tied_weights=True, activation="identity")

        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 0.0, 1.5, 1.0]]))

        x = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, -1.0, 0.0, 1.0]])
        decoded = autoencoder.forward(x)
        expected = x @ encoder.weight.t() @ encoder.weight

        self.assertTrue(torch.allclose(decoded, expected))

    def test_dimensionality_validation(self):
        encoder = nn.Linear(4, 5)
        decoder = nn.Linear(5, 4)
        with self.assertRaises(ValueError):
            AutoEncoder(encoder=encoder, decoder=decoder, validate_shapes=True)

    def test_factory_function_creates_autoencoder(self):
        ae = linear_autoencoder(n_features=5, hidden_sizes=2, activation="identity", tied_weights=False)
        self.assertIsInstance(ae, AutoEncoder)
        first_linear = ae._first_linear(ae.encoder)
        self.assertEqual(first_linear.out_features, 2)

    def test_W_property_exposes_encoder_weight(self):
        encoder = nn.Linear(5, 2, bias=False)
        decoder = nn.Linear(2, 5, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, activation="identity")
        self.assertIn("encoder", autoencoder.W)
        self.assertIn("decoder", autoencoder.W)

    def test_deep_autoencoder_builder(self):
        ae = deep_autoencoder([8, 4, 2], activation="relu", bias=True, validate_shapes=False)
        x = torch.randn(3, 8)
        out = ae(x)
        self.assertEqual(out.shape, (3, 8))

    def test_tied_weights_multi_layer_linear(self):
        ae = linear_autoencoder(n_features=6, hidden_sizes=[4, 3], activation="relu", tied_weights=True)
        x = torch.randn(2, 6)
        out = ae(x)
        self.assertEqual(out.shape, (2, 6))
        self.assertIsInstance(ae.decoder[0], TiedLinear)

    def test_tied_weights_arbitrary_module_batchnorm(self):
        encoder = nn.BatchNorm1d(4, affine=True)
        decoder = nn.BatchNorm1d(4, affine=True)
        ae = AutoEncoder(encoder=encoder, decoder=decoder, activation="relu", tied_weights=True, validate_shapes=False)
        self.assertIs(encoder.weight, decoder.weight)
        x = torch.randn(3, 4)
        out = ae(x)
        self.assertEqual(out.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
