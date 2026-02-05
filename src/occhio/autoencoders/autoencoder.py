# ABOUTME: Defines the AutoEncoder abstraction used across occhio.
# ABOUTME: Provides tied linear layers plus encode/decode utilities.

from __future__ import annotations
from typing import Any, Dict, List, Optional
from torch import Tensor, nn
from warnings import warn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ) -> None:

        self.encoder: Optional[nn.Module] = encoder
        self.decoder: Optional[nn.Module] = decoder
        self.n_features = n_features

        if self.tie_weights.__func__ is AutoEncoder.tie_weights:
            warn(
                f"{self.__class__.__name__} does not implement tie_weights; \
                passing tie_weights=True will fail.",
                UserWarning,
            )

        if self.build_encoder.__func__ is AutoEncoder.build_encoder:
            warn(
                f"{self.__class__.__name__} does not implement build_encoder",
                UserWarning,
            )

        if self.build_decoder.__func__ is AutoEncoder.build_decoder:
            warn(
                f"{self.__class__.__name__} does not implement build_decoder",
                UserWarning,
            )

    def set_encoder(self, encoder: nn.Module, *, tie_to_decoder: bool = False) -> None:
        if tie_to_decoder:
            if self.decoder is None:
                raise ValueError("Cannot tie weights without an existing decoder.")
            encoder = self.tie_weights(
                source=self.decoder, target=self.encoder, target_name="encoder"
            )
        self.encoder = encoder

    def set_decoder(self, decoder: nn.Module, *, tie_to_encoder: bool = False) -> None:
        if tie_to_encoder:
            if self.encoder is None:
                raise ValueError("Cannot tie weights without an existing encoder.")
            decoder = self.tie_weights(
                source=self.encoder, target=self.decoder, target_name="decoder"
            )
        self.decoder = decoder

    def build_encoder(
        self,
    ) -> nn.Module:
        raise NotImplementedError(
            "build_encoder is not implemented for this AutoEncoder instance."
        )

    def build_decoder(self) -> nn.Module:
        raise NotImplementedError(
            "build_decoder is not implemented for this AutoEncoder instance."
        )

    def tie_weights(
        self,
        source: nn.Module,
        target: nn.Module,
        target_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> nn.Module:
        raise NotImplementedError(
            "tie_weights is not implemented for this AutoEncoder instance."
        )

    @property
    def get_autoencoder(self) -> nn.Module:
        self.validate()
        return self.encoder

    @property
    def get_decoder(self) -> nn.Module:
        self.validate()
        return self.decoder

    @property
    def get_autoencoder(self) -> nn.Sequential:
        self.validate()
        return nn.Sequential(self.encoder, self.decoder)

    @property
    def W(self) -> Dict[str, List[nn.Parameter]]:
        self.validate()
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.decoder.parameters()),
        }

    def forward(self, x: Tensor) -> Tensor:
        self.validate()
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        self.validate()
        return self.encoder(x)

    def decode(self, encoded: Tensor) -> Tensor:
        self.validate()
        return self.decoder(encoded)

    def freeze_encoder(self) -> None:
        self.validate()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self) -> None:
        self.validate()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze(self) -> None:
        self.validate()
        self.freeze_encoder()
        self.freeze_decoder()

    def unfreeze_encoder(self) -> None:
        self.validate()
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self) -> None:
        self.validate()
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        self.validate()
        self.unfreeze_encoder()
        self.unfreeze_decoder()

    def validate(
        self,
        x: Optional[Tensor] = None,
        modules: Optional[List[nn.Module]] = None,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:

        if not isinstance(self.encoder, nn.Module) or not isinstance(
            self.decoder, nn.Module
        ):
            raise ValueError(
                "Encoder and decoder must be provided as instances of torch.nn.Module"
            )

        log: List[Dict[str, Any]] = []

        def record(name: str, tensor: Tensor) -> None:
            log.append(
                {
                    "stage": name,
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "device": tensor.device,
                }
            )

        # Validate full autoencoder path
        if x is not None:
            record("input", x)
            z = self.encoder(x)
            record("encoded", z)
            y = self.decoder(z)
            record("decoded", y)

        if modules is not None:
            if x is None:
                raise ValueError("Input must be provided when validating modules.")
            current = x

            for i, module in enumerate(modules):
                current = module(current)
                record(f"module[{i}]:{module.__class__.__name__}", current)
        return log
