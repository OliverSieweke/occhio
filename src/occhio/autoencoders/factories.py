# ABOUTME: Factory helpers for building configured autoencoders.
# ABOUTME: Provides linear and deep builders with optional weight tying.

from __future__ import annotations

import copy
from typing import List

import torch
from torch import nn

from occhio.autoencoder import ActivationLike, AutoEncoder, TiedLinear
from occhio.autoencoders.configs import DeepAutoencoderConfig, LinearAutoencoderConfig


class _ModuleAutoEncoder(AutoEncoder):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder=encoder, decoder=decoder)
        self._encoder_module = encoder
        self._decoder_module = decoder

    def create_encoder(self) -> nn.Module:
        return self._encoder_module

    def create_decoder(self) -> nn.Module:
        return self._decoder_module


def create_autoencoder(encoder: nn.Module, decoder: nn.Module) -> AutoEncoder:
    return _ModuleAutoEncoder(encoder=encoder, decoder=decoder)


def linear_autoencoder(cfg: LinearAutoencoderConfig) -> AutoEncoder:
    if cfg.input_dim <= 0 or cfg.latent_dim <= 0:
        raise ValueError("input_dim and latent_dim must be positive.")

    encoder_layers: List[nn.Module] = []
    encoder_linear = nn.Linear(cfg.input_dim, cfg.latent_dim, bias=cfg.bias)
    encoder_layers.append(encoder_linear)
    encoder_layers.append(_activation_module(cfg.activation))

    encoder = nn.Sequential(*encoder_layers)

    if cfg.tied_weights:
        decoder_linear: nn.Module = TiedLinear(encoder_linear, bias=cfg.bias)
    else:
        decoder_linear = nn.Linear(cfg.latent_dim, cfg.input_dim, bias=cfg.bias)

    decoder = nn.Sequential(decoder_linear)
    return create_autoencoder(encoder=encoder, decoder=decoder)


def deep_autoencoder(cfg: DeepAutoencoderConfig) -> AutoEncoder:
    dims = list(cfg.dimensions)
    if len(dims) < 2:
        raise ValueError("dimensions must contain at least two entries.")
    if cfg.validate_shapes:
        if any(next_dim >= curr_dim for curr_dim, next_dim in zip(dims, dims[1:])):
            raise ValueError("dimensions must be strictly decreasing when validate_shapes is True.")

    encoder_layers: List[nn.Module] = []
    encoder_linears: List[nn.Linear] = []

    for idx, (inp, out) in enumerate(zip(dims[:-1], dims[1:])):
        linear_layer = nn.Linear(inp, out, bias=cfg.bias)
        encoder_layers.append(linear_layer)
        encoder_linears.append(linear_layer)
        if idx < len(dims) - 2:
            encoder_layers.append(_activation_module(cfg.hidden_activation))

    encoder = nn.Sequential(*encoder_layers)

    decoder_layers: List[nn.Module] = []
    for idx, linear_layer in enumerate(reversed(encoder_linears)):
        if cfg.tied_weights:
            decoder_layers.append(TiedLinear(linear_layer, bias=cfg.bias))
        else:
            decoder_layers.append(nn.Linear(linear_layer.out_features, linear_layer.in_features, bias=cfg.bias))
        if idx < len(encoder_linears) - 1:
            decoder_layers.append(_activation_module(cfg.hidden_activation))

    decoder = nn.Sequential(*decoder_layers)
    return create_autoencoder(encoder=encoder, decoder=decoder)


def _activation_module(activation: ActivationLike) -> nn.Module:
    if isinstance(activation, str):
        return _resolve_activation(activation)
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    if callable(activation):
        class _CallableActivation(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fn(x)

        return _CallableActivation(activation)
    raise TypeError("Activation must be a string, torch.nn.Module, or callable.")


def _resolve_activation(activation: ActivationLike) -> nn.Module:
    if isinstance(activation, str):
        name = activation.lower()
        if name in {"identity", "linear", "none"}:
            return nn.Identity()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation '{activation}'.")
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation):
        class _CallableActivation(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fn(x)

        return _CallableActivation(activation)
    raise TypeError("Activation must be a string, torch.nn.Module, or callable.")
