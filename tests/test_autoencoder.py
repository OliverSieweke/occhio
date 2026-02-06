# ABOUTME: Tests for the AutoEncoder abstraction and helper factories.
# ABOUTME: Ensures activation handling, tying, factories, and examples behave.

import unittest

import torch
from torch import nn

from occhio.autoencoder import AutoEncoder


class ToyAutoEncoder(AutoEncoder):
    def __init__(self):
        self.input_dim = 4
        self.latent_dim = 2
        super().__init__()

    def create_encoder(self) -> nn.Module:
        return nn.Linear(self.input_dim, self.latent_dim, bias=True)

    def create_decoder(self) -> nn.Module:
        return nn.Linear(self.latent_dim, self.input_dim, bias=True)


class TiedBatchNormAE(AutoEncoder):
    def __init__(self):
        self.dim = 4
        super().__init__()

    def create_encoder(self) -> nn.Module:
        return nn.BatchNorm1d(self.dim, affine=True)

    def create_decoder(self) -> nn.Module:
        return nn.BatchNorm1d(self.dim, affine=True)


class NoTieAutoEncoder(AutoEncoder):
    def __init__(self):
        self.dim = 3
        super().__init__()

    def create_encoder(self) -> nn.Module:
        return nn.Linear(self.dim, self.dim)

    def create_decoder(self) -> nn.Module:
        return nn.Linear(self.dim, self.dim)


class SharedLinearTieAE(AutoEncoder):
    def __init__(self):
        self.dim = 4
        super().__init__()

    def create_encoder(self) -> nn.Module:
        return nn.Linear(self.dim, self.dim)

    def create_decoder(self) -> nn.Module:
        return nn.Linear(self.dim, self.dim)

    def tie_weights(self, source: nn.Module, target: nn.Module) -> None:
        if not isinstance(source, nn.Linear) or not isinstance(target, nn.Linear):
            raise TypeError("tie_weights expects linear modules")
        target.weight = source.weight
        target.bias = source.bias


class AutoEncoderTests(unittest.TestCase):
    def test_autoencoder_is_abstract(self):
        with self.assertRaises(TypeError):
            AutoEncoder()  # type: ignore[arg-type]

    def test_forward_runs_with_subclass(self):
        ae = ToyAutoEncoder()
        x = torch.randn(3, 4)
        out = ae(x)
        self.assertEqual(out.shape, (3, 4))

    def test_forward_runs_when_modules_lack_weights(self):
        ae = TiedBatchNormAE()
        x = torch.randn(5, 4)
        out = ae(x)
        self.assertEqual(out.shape, (5, 4))

    def test_W_exposes_encoder_and_decoder_params(self):
        ae = ToyAutoEncoder()
        weights = ae.W
        self.assertIn("encoder", weights)
        self.assertIn("decoder", weights)
        self.assertGreater(len(weights["encoder"]), 0)
        self.assertGreater(len(weights["decoder"]), 0)

    def test_freeze_and_unfreeze_encoder(self):
        ae = ToyAutoEncoder()
        ae.freeze_encoder()
        self.assertTrue(all(not p.requires_grad for p in ae.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in ae.decoder.parameters()))
        ae.unfreeze_encoder()
        self.assertTrue(all(p.requires_grad for p in ae.encoder.parameters()))

    def test_freeze_and_unfreeze_decoder(self):
        ae = ToyAutoEncoder()
        ae.freeze_decoder()
        self.assertTrue(all(not p.requires_grad for p in ae.decoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in ae.encoder.parameters()))
        ae.unfreeze_decoder()
        self.assertTrue(all(p.requires_grad for p in ae.decoder.parameters()))

    def test_warns_when_tie_weights_not_overridden(self):
        with self.assertWarns(UserWarning):
            NoTieAutoEncoder()

    def test_set_encoder_ties_weights_when_requested(self):
        ae = SharedLinearTieAE()
        new_encoder = nn.Linear(ae.dim, ae.dim)
        ae.set_encoder(new_encoder, tie_weights=True)
        self.assertIs(ae.encoder.weight, ae.decoder.weight)
        self.assertIs(ae.encoder.bias, ae.decoder.bias)

    def test_set_encoder_requires_decoder_when_tying(self):
        ae = SharedLinearTieAE()
        ae.decoder = None  # simulate missing decoder
        with self.assertRaises(ValueError):
            ae.set_encoder(nn.Linear(ae.dim, ae.dim), tie_weights=True)

    def test_tie_weights_flag_raises_when_not_implemented(self):
        ae = NoTieAutoEncoder()
        with self.assertRaises(NotImplementedError):
            ae.set_encoder(nn.Linear(ae.dim, ae.dim), tie_weights=True)


if __name__ == "__main__":
    unittest.main()
