# ABOUTME: Defines the Linear AutoEncoder subclass used across occhio.
# ABOUTME: Provides tied implementation for linear autoencoders

from __future__ import annotations

from .autoencoder import AutoEncoder
from .configs import AutoEncoderConfig, LinearAutoEncoderConfig

from typing import Any, Dict, List, Optional
import torch.nn as nn
import torch.Tensor as Tensor
import warnings.warn as warn


class LinearAutoEncoder(AutoEncoder):
    def __init__(self, encoder: Optional[nn.Linear] = None, decoder: Optional[nn.Linear] = None, cfg: AutoEncoderConfig = None) -> None:
        super().__init__(encoder=encoder, decoder=decoder)
        pass
        # if (encoder and decoder and not(cfg.tie_weights)): return

        #  if cfg.encoder 

        # if cfg.tie_weights:
        #     if isinstance(encoder, nn.Linear) or not isinstance(decoder, nn.Linear) or cfg.encoder_dims or cfg.decoder_dims:
        #         raise ValueError("Cannot tie weights need values for AutoEncoderConfig.encoder or AutoEncoderConfig.decoder or AutoEncoderConfig.encoder_dims or AutoEncoderConfig.decoder_dims")

        #     if not isinstance(encoder, nn.Linear) or not isinstance(decoder, nn.Linear):
        #         raise ValueError("Encoder and decoder must be provided as instances of torch.nn.Linear")
        #     encoder = self.tie_weights(source=decoder, target=encoder, target_name="encoder")
        #     decoder = self.tie_weights(source=encoder, target=decoder, target_name="decoder")

        


    def build_encoder(self, in_dim: int, out_dim: int, bias: bool = True, tie_to_decoder: bool = False) -> nn.Linear:
        encoder = nn.Linear(in_dim, out_dim, bias=bias)
        if tie_to_decoder:
            encoder = self.tie_weights(source=self.decoder, target=encoder, target_name="encoder")
        return encoder

    def build_decoder(self, in_dim: int, out_dim: int, bias: bool = True, tie_to_encoder: bool = False) -> nn.Linear:
        decoder = nn.Linear(in_dim, out_dim, bias=bias)
        if tie_to_encoder:
            decoder = self.tie_weights(source=self.encoder, target=decoder, target_name="decoder")
        return decoder

    def set_encoder(self, encoder: nn.Linear, *, tie_weights: bool = False) -> None:
        if tie_weights:
            if self.decoder is None:
                raise ValueError("Cannot tie weights without an existing decoder.")
            if not isinstance(self.decoder, nn.Linear):
                raise ValueError("Decoder must be provided as an instance of torch.nn.Linear when tying weights.")
            encoder = self.tie_weights(source=self.encoder, target=self.decoder)
        self.encoder = encoder

    def set_decoder(self, decoder: nn.Linear, *, tie_weights: bool = False) -> None:
        if tie_weights:
            if self.encoder is None:
                raise ValueError("Cannot tie weights without an existing encoder.")
            if not isinstance(self.encoder, nn.Linear):
                raise ValueError("Encoder must be provided as an instance of torch.nn.Linear when tying weights.")
            decoder = self.tie_weights(source=self.encoder, target=self.decoder)
        self.decoder = decoder

    def validate(
        self,
        x: Optional[Tensor] = None,
        modules: Optional[List[nn.Linear]] = None,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:

        if not isinstance(self.encoder, nn.Linear) or not isinstance(self.decoder, nn.Linear):
            raise ValueError("Encoder and decoder must be provided as instances of torch.nn.Linear")

        log: List[Dict[str, Any]] = []

        def record(name: str, tensor: Tensor) -> None:
            log.append({
                "stage": name,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "device": tensor.device,
            })

        # Validate full autoencoder path
        if x is not None:
            record("input", x)
            z = self.encoder(x)
            record("encoded", z)
            y = self.decoder(z)
            record("decoded", y)

        if isinstance(modules, nn.Linear):
            if not isinstance(x, Tensor):
                raise ValueError("Input must be provided as an instance of torch.Tensor when validating a modules.")
            
            current = x
            for i, module in enumerate(modules):
                current = module(current)
                record(f"module[{i}]:{module.__class__.__name__}", current)
        return log

