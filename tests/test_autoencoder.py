# ABOUTME: Tests for the AutoEncoder component in occhio.
# ABOUTME: Verifies pluggable encoder/decoder, activation hook, tied-weight helper, and weight exposure.

import unittest

import torch
from torch import nn

from occhio.autoencoder import AutoEncoder


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

    def test_requires_decoder_when_not_tied(self):
        encoder = nn.Linear(4, 2)
        with self.assertRaises(ValueError):
            AutoEncoder(encoder=encoder, decoder=None, tied_weights=False)

    def test_tied_weights_without_decoder(self):
        encoder = nn.Linear(4, 2, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=None, tied_weights=True, activation="identity")

        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 0.0, 1.5, 1.0]]))

        x = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, -1.0, 0.0, 1.0]])
        decoded = autoencoder.forward(x)
        expected = x @ encoder.weight.t() @ encoder.weight

        self.assertTrue(torch.allclose(decoded, expected))

    def test_W_property_exposes_encoder_weight(self):
        encoder = nn.Linear(5, 2, bias=False)
        decoder = nn.Linear(2, 5, bias=False)
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, activation="identity")
        self.assertIs(autoencoder.W, encoder.weight)


if __name__ == "__main__":
    unittest.main()
