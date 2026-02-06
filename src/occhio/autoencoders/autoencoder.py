# ABOUTME: Defines the AutoEncoder abstraction used across occhio.
# ABOUTME: Provides tied linear layers plus encode/decode utilities.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, zeros
import torch.nn as nn
from warnings import warn
from .configs import AutoEncoderConfig


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig) -> None:
        self.cfg = cfg
        self._validate_cfg(cfg)

        # 0) Warn if not implemented
        if self.tie_weights.__func__ is AutoEncoder.tie_weights:
            warn(f"{self.__class__.__name__} does not implement tie_weights; passing tie_weights=True will fail.", UserWarning)

        if self.build_encoder.__func__ is AutoEncoder.build_encoder:
            warn(f"{self.__class__.__name__} does not implement build_encoder", UserWarning)

        if self.build_decoder.__func__ is AutoEncoder.build_decoder:
            warn(f"{self.__class__.__name__} does not implement build_decoder", UserWarning)

        # 1) Encoder/decoder modules supplied directly
        if cfg.encoder is not None:
            self.set_encoder(cfg.encoder, tie_to_decoder=False)
        if cfg.decoder is not None:
            self.set_decoder(cfg.decoder, tie_to_encoder=False)

        # 2) If missing, build from dims
        if self.encoder is None and cfg.encoder_dims is not None:
            enc = self.build_encoder(cfg.encoder_dims)
            if not isinstance(enc, nn.Module):
                raise TypeError("build_encoder must return an nn.Module.")
            self.set_encoder(enc, tie_to_decoder=False)

        if self.decoder is None and cfg.decoder_dims is not None:
            dec = self.build_decoder(cfg.decoder_dims)
            if not isinstance(dec, nn.Module):
                raise TypeError("build_decoder must return an nn.Module.")
            self.set_decoder(dec, tie_to_encoder=False)

        # 3) Apply weight tying if requested
        if cfg.tie_weights:
            # Require both to exist before tying
            if self.encoder is None or self.decoder is None:
                raise ValueError("tie_weights=True requires both encoder and decoder to exist.")

            # Rebind modules returned by tie_weights (it may return wrapped modules)
            # Tie both directions via your helpers so that tie logic stays centralized.
            self.set_encoder(self.encoder, tie_to_decoder=True)
            self.set_decoder(self.decoder, tie_to_encoder=True)

        self._validate_autoencoder_structure(cfg)
        self.summary: Dict[str, Any] = self.describe()    

    def set_encoder(self, encoder: nn.Module, *, tie_to_decoder: bool = False) -> None:
        if tie_to_decoder:
            if self.decoder is None:
                raise ValueError("Cannot tie weights without an existing decoder.")
            encoder = self.tie_weights(source=self.decoder, target=self.encoder, target_name="encoder")
        self.encoder = encoder

    def set_decoder(self, decoder: nn.Module, *, tie_to_encoder: bool = False) -> None:
        if tie_to_encoder:
            if self.encoder is None:
                raise ValueError("Cannot tie weights without an existing encoder.")
            decoder = self.tie_weights(source=self.encoder, target=self.decoder, target_name="decoder")
        self.decoder = decoder
        
    def build_encoder(self, ) -> nn.Module:
        raise NotImplementedError("build_encoder is not implemented for this AutoEncoder instance.")

    def build_decoder(self) -> nn.Module:
        raise NotImplementedError("build_decoder is not implemented for this AutoEncoder instance.")

    def tie_weights(self, source: nn.Module, target: nn.Module, target_name: Optional[str] = None, *args, **kwargs) -> nn.Module:
        raise NotImplementedError("tie_weights is not implemented for this AutoEncoder instance.")

    @property
    def get_autoencoder(self) -> nn.Module:
        return self.encoder
    
    @property
    def get_decoder(self) -> nn.Module:
        return self.decoder
    
    @property
    def get_autoencoder(self) -> nn.Sequential:
        return nn.Sequential(self.encoder, self.decoder)
    
    @property
    def W(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.decoder.parameters()),
        }
    
    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, encoded: Tensor) -> Tensor:
        return self.decoder(encoded)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze(self) -> None:
        self.freeze_encoder()
        self.freeze_decoder()

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        self.unfreeze_encoder()
        self.unfreeze_decoder()
    
    def validate(
        self,
        x: Optional[Tensor] = None,
        modules: Optional[List[nn.Module]] = None,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:

        if not isinstance(self.encoder, nn.Module) or not isinstance(self.decoder, nn.Module):
            raise ValueError("Encoder and decoder must be provided as instances of torch.nn.Module")

        log: List[Dict[str, Any]] = []

        # Validate full autoencoder path
        if x is not None:
            log.append(self._record("input", x))
            z = self.encoder(x)
            log.append(self._record("encoded", z))
            y = self.decoder(z)
            log.append(self._record("decoded", y))

        if modules is not None:
            if x is None:
                raise ValueError("Input must be provided when validating modules.")
            current = x

            for i, module in enumerate(modules):
                current = module(current)
                log.append(self._record(f"module[{i}]:{module.__class__.__name__}", current))
        return log

    def _record(name: str, tensor: Tensor) -> None:
            return ({
                "stage": name,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "device": tensor.device,
            })


    def describe(self) -> Dict[str, Any]:
        """
        Return a structured, machine-readable description of the autoencoder.
        This makes no assumptions beyond what can be inferred from config and modules.
        """

        def summarize_dims(dims: Optional[List[Tuple[int, int]]]) -> Dict[str, Any]:
            if dims is None:
                return {"present": False}
            return {
                "present": True,
                "num_layers": len(dims),
                "layer_dims": dims,
                "input_dim": dims[0][0],
                "output_dim": dims[-1][1],
                "hidden_dims": [out for (_, out) in dims[:-1]],
            }

        encoder_dims_summary = summarize_dims(self.cfg.encoder_dims)
        decoder_dims_summary = summarize_dims(self.cfg.decoder_dims)

        # Infer latent / io dims when possible
        input_dim = None
        latent_dim = None
        output_dim = None

        if self.cfg.encoder_dims is not None:
            input_dim = self.cfg.encoder_dims[0][0]
            latent_dim = self.cfg.encoder_dims[-1][1]

        if self.cfg.decoder_dims is not None:
            output_dim = self.cfg.decoder_dims[-1][1]
            if latent_dim is None:
                latent_dim = self.cfg.decoder_dims[0][0]

        return {
            "class": self.__class__.__name__,
            "tie_weights": self.cfg.tie_weights,
            "encoder": {
                "provided_as_module": self.cfg.encoder is not None,
                "built_from_dims": self.cfg.encoder_dims is not None,
                "dims": encoder_dims_summary,
                "module_type": type(self.encoder).__name__ if self.encoder is not None else None,
                "num_parameters": (
                    sum(p.numel() for p in self.encoder.parameters())
                    if self.encoder is not None
                    else None
                ),
            },
            "decoder": {
                "provided_as_module": self.cfg.decoder is not None,
                "built_from_dims": self.cfg.decoder_dims is not None,
                "dims": decoder_dims_summary,
                "module_type": type(self.decoder).__name__ if self.decoder is not None else None,
                "num_parameters": (
                    sum(p.numel() for p in self.decoder.parameters())
                    if self.decoder is not None
                    else None
                ),
            },
            "architecture": {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "output_dim": output_dim,
                "encoder_num_layers": encoder_dims_summary.get("num_layers"),
                "decoder_num_layers": decoder_dims_summary.get("num_layers"),
                "is_symmetric": (
                    self.cfg.encoder_dims is not None
                    and self.cfg.decoder_dims is not None
                    and self.cfg.encoder_dims == list(reversed([(b, a) for (a, b) in self.cfg.decoder_dims]))
                ),
            },
        }

    def _validate_cfg(self, cfg: AutoEncoderConfig) -> None:
        # Disallow mixing explicit modules and dims for the same side (ambiguous precedence)
        if cfg.encoder is not None and cfg.encoder_dims is not None:
            raise ValueError("Provide either cfg.encoder or cfg.encoder_dims, not both.")
        if cfg.decoder is not None and cfg.decoder_dims is not None:
            raise ValueError("Provide either cfg.decoder or cfg.decoder_dims, not both.")

        # Must provide enough information to construct both halves
        if cfg.encoder is None and cfg.encoder_dims is None:
            raise ValueError("Must provide either cfg.encoder or cfg.encoder_dims.")
        if cfg.decoder is None and cfg.decoder_dims is None:
            raise ValueError("Must provide either cfg.decoder or cfg.decoder_dims.")

        # Validate dims format if present
        if cfg.encoder_dims is not None:
            self._validate_dims(cfg.encoder_dims, "encoder_dims")
        if cfg.decoder_dims is not None:
            self._validate_dims(cfg.decoder_dims, "decoder_dims")

        # If tying requested but tie_weights not overridden, fail early with clear error.
        if cfg.tie_weights and (self.tie_weights.__func__ is AutoEncoder.tie_weights):
            raise ValueError("cfg.tie_weights=True but tie_weights() is not implemented.")

    @staticmethod
    def _validate_dims(dims: List[Tuple[int, int]], name: str) -> None:
        if not isinstance(dims, list) or len(dims) < 1:
            raise ValueError(f"{name} must be a non-empty List[Tuple[int,int]].")
        for i, pair in enumerate(dims):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(f"{name}[{i}] must be a (in_dim, out_dim) tuple.")
            in_dim, out_dim = pair
            if not (isinstance(in_dim, int) and isinstance(out_dim, int)):
                raise ValueError(f"{name}[{i}] dims must be ints; got {type(in_dim)} and {type(out_dim)}.")
            if in_dim < 1 or out_dim < 1:
                raise ValueError(f"{name}[{i}] dims must be >= 1; got {(in_dim, out_dim)}.")

    def _infer_io_dims_from_dims(self, cfg: AutoEncoderConfig) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Returns (input_dim, latent_dim, output_dim) if inferable from dims, else None for unknowns.
        """
        input_dim = latent_dim = output_dim = None

        if cfg.encoder_dims is not None:
            input_dim = cfg.encoder_dims[0][0]
            latent_dim = cfg.encoder_dims[-1][1]

        if cfg.decoder_dims is not None:
            output_dim = cfg.decoder_dims[-1][1]
            # decoder input is its first in_dim
            dec_in = cfg.decoder_dims[0][0]
            # If encoder latent known, they should match
            if latent_dim is None:
                latent_dim = dec_in

        return input_dim, latent_dim, output_dim

    def _validate_autoencoder_structure(self, cfg: AutoEncoderConfig) -> None:
        assert self.encoder is not None and self.decoder is not None  # guaranteed earlier

        # If dims are provided, enforce basic AE compatibility conditions.
        # - encoder last out_dim == decoder first in_dim  (latent)
        # - encoder first in_dim == decoder last out_dim  (reconstruction space)
        if cfg.encoder_dims is not None and cfg.decoder_dims is not None:
            enc_in, enc_lat = cfg.encoder_dims[0][0], cfg.encoder_dims[-1][1]
            dec_lat, dec_out = cfg.decoder_dims[0][0], cfg.decoder_dims[-1][1]

            if enc_lat != dec_lat:
                raise ValueError(
                    f"Latent mismatch: encoder outputs {enc_lat} but decoder expects {dec_lat}."
                )
            if enc_in != dec_out:
                raise ValueError(
                    f"Input/output mismatch: encoder expects input dim {enc_in} but decoder outputs {dec_out}."
                )

        # If only one side dims is provided, we can still sanity-check internal consistency of that side.
        if cfg.encoder_dims is not None:
            self._validate_chain_dims(cfg.encoder_dims, "encoder_dims")
        if cfg.decoder_dims is not None:
            self._validate_chain_dims(cfg.decoder_dims, "decoder_dims")

        # Optional: cheap functional sanity check if dims allow us to create a dummy tensor.
        # This catches obvious forward incompatibilities for MLP-style encoders/decoders.
        input_dim, _, output_dim = self._infer_io_dims_from_dims(cfg)
        if input_dim is not None and output_dim is not None and input_dim == output_dim:
            try:
                x = zeros(2, input_dim)
                z = self.encoder(x)
                y = self.decoder(z)
                if not isinstance(y, Tensor):
                    raise TypeError("decoder output is not a Tensor.")
                if y.ndim != x.ndim:
                    raise ValueError(f"decoded tensor rank mismatch: got {tuple(y.shape)} expected rank {x.ndim}.")
                if y.shape[0] != x.shape[0]:
                    raise ValueError(f"batch size changed across AE: got {y.shape[0]} expected {x.shape[0]}.")
                if y.shape[-1] != x.shape[-1]:
                    raise ValueError(f"final dim mismatch: got {y.shape[-1]} expected {x.shape[-1]}.")
            except Exception as e:
                raise ValueError(f"Autoencoder forward sanity-check failed: {e}") from e

    @staticmethod
    def _validate_chain_dims(dims: List[Tuple[int, int]], name: str) -> None:
        # ensure out_dim of layer i matches in_dim of layer i+1
        for i in range(len(dims) - 1):
            _, out_dim = dims[i]
            next_in, _ = dims[i + 1]
            if out_dim != next_in:
                raise ValueError(
                    f"{name} is not a valid chain at index {i}: "
                    f"layer {i} out_dim={out_dim} != layer {i+1} in_dim={next_in}."
                )