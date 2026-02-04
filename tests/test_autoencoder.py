# ABOUTME: Tests for the AutoEncoder component in occhio.
# ABOUTME: Verifies encoding, decoding, activation, and weight exposure behaviors.

import unittest

import torch

from occhio.autoencoder import AutoEncoder


class AutoEncoderTests(unittest.TestCase):
    def test_forward_reconstruction_shape(self):
        autoencoder = AutoEncoder(n_features=6, n_hidden=3, activation="relu", tied_weights=True, bias=False)
        x = torch.randn(4, 6)

        encoded = autoencoder.encode(x)
        decoded = autoencoder.forward(x)

        self.assertEqual(encoded.shape, (4, 3))
        self.assertEqual(decoded.shape, (4, 6))

    def test_tied_weights_use_transpose(self):
        autoencoder = AutoEncoder(n_features=4, n_hidden=2, activation="identity", tied_weights=True, bias=False)
        weight = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 0.0, 1.5, 1.0]])
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, -1.0, 0.0, 1.0]])

        with torch.no_grad():
            autoencoder.encoder.weight.copy_(weight)

        decoded = autoencoder.forward(x)
        expected = x @ weight.T @ weight

        self.assertTrue(torch.allclose(decoded, expected))

    def test_activation_relu_applied(self):
        autoencoder = AutoEncoder(n_features=3, n_hidden=2, activation="relu", tied_weights=True, bias=False)
        with torch.no_grad():
            autoencoder.encoder.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

        x = torch.tensor([[-1.0, 2.0, -0.5]])
        output = autoencoder.forward(x)
        expected = torch.tensor([[0.0, 2.0, 0.0]])

        self.assertTrue(torch.allclose(output, expected))

    def test_W_property_exposes_encoder_weight(self):
        autoencoder = AutoEncoder(n_features=5, n_hidden=2, activation="identity", tied_weights=True, bias=False)
        self.assertIs(autoencoder.W, autoencoder.encoder.weight)


if __name__ == "__main__":
    unittest.main()
