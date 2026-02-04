# ABOUTME: Defines the abstract AutoEncoder container and activation utilities.
# ABOUTME: Provides factory helpers for linear autoencoders and module wrapping.

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


ActivationSpec = str | Callable[[torch.Tensor], torch.Tensor] | nn.Module | None


class _FunctionalActivation(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.fn(x)


def _resolve_activation(spec: ActivationSpec) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if spec is None or spec == "identity":
        return None
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, str):
        attr = getattr(nn, spec, None)
        if inspect.isclass(attr):
            return attr()
        func = getattr(F, spec, None)
        if callable(func):
            return _FunctionalActivation(func)
        raise ValueError("unknown activation")
    if callable(spec):
        return spec
    raise TypeError("invalid activation")


def _activation_module_factory(spec: ActivationSpec) -> Callable[[], nn.Module]:
    if spec is None or spec == "identity":
        return nn.Identity
    if isinstance(spec, str):
        attr = getattr(nn, spec, None)
        if inspect.isclass(attr):
            return attr
        func = getattr(F, spec, None)
        if callable(func):
            return lambda: _FunctionalActivation(func)
        raise ValueError("unknown activation")
    if isinstance(spec, nn.Module):
        return lambda: copy.deepcopy(spec)
    if callable(spec):
        return lambda: _FunctionalActivation(spec)
    raise TypeError("invalid activation")


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class AutoEncoder(nn.Module, ABC):
    def __init__(self, *, activation: ActivationSpec = "identity", tied_weights: bool = False):
        super().__init__()
        self.activation = _resolve_activation(activation)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        if self.encoder is None or self.decoder is None:
            raise ValueError("encoder and decoder required")
        if tied_weights:
            self._tie_parameters(self.encoder, self.decoder)

    @abstractmethod
    def create_encoder(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_decoder(self) -> nn.Module:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.activation is None:
            return h
        if isinstance(self.activation, nn.Module):
            return self.activation(h)
        return self.activation(h)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    @property
    def W(self) -> Dict[str, Dict[str, nn.Parameter]]:
        return {
            "encoder": dict(self.encoder.named_parameters()),
            "decoder": dict(self.decoder.named_parameters()),
        }

    def _tie_parameters(self, src: nn.Module, dst: nn.Module) -> None:
        src_params = dict(src.named_parameters())
        dst_params = dict(dst.named_parameters())
        for name, param in src_params.items():
            if name not in dst_params:
                continue
            target = dst_params[name]
            if param.shape != target.shape:
                raise ValueError("shape mismatch for tied parameter")
            parent, attr = _get_parameter_owner(dst, name)
            parent._parameters[attr] = param


class TiedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, *, bias: bool = True):
        super().__init__()
        self.linear = linear
        if bias:
            self.bias = nn.Parameter(torch.zeros(linear.in_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return F.linear(x, self.linear.weight.t(), self.bias)


class _ProvidedAutoEncoder(AutoEncoder):
    def __init__(self, *, encoder: nn.Module, decoder: nn.Module, activation: ActivationSpec, tied_weights: bool):
        self._encoder_factory = lambda: encoder
        self._decoder_factory = lambda: decoder
        super().__init__(activation=activation, tied_weights=tied_weights)

    def create_encoder(self) -> nn.Module:
        return self._encoder_factory()

    def create_decoder(self) -> nn.Module:
        return self._decoder_factory()


def create_autoencoder(*, encoder: nn.Module, decoder: nn.Module, activation: ActivationSpec = "identity", tied_weights: bool = False) -> AutoEncoder:
    return _ProvidedAutoEncoder(encoder=encoder, decoder=decoder, activation=activation, tied_weights=tied_weights)


def _build_linear_stack(dimensions: Sequence[int], *, bias: bool, activation: ActivationSpec) -> nn.Sequential:
    if len(dimensions) < 2:
        raise ValueError("need at least input and output dimension")
    layers: List[nn.Module] = []
    act_factory = _activation_module_factory(activation)
    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        if out_dim != dimensions[-1]:
            layers.append(act_factory())
    return nn.Sequential(*layers)


def _reverse_linear_stack(linear_stack: nn.Sequential, *, use_tied: bool, activation: ActivationSpec, bias: bool) -> nn.Sequential:
    reversed_layers: List[nn.Module] = []
    act_factory = _activation_module_factory(activation)
    linear_layers = [layer for layer in linear_stack if isinstance(layer, nn.Linear)]
    for idx, layer in enumerate(reversed(linear_layers)):
        if use_tied:
            reversed_layers.append(TiedLinear(layer, bias=bias))
        else:
            reversed_layers.append(nn.Linear(layer.out_features, layer.in_features, bias=bias))
        if idx != len(linear_layers) - 1:
            reversed_layers.append(act_factory())
    return nn.Sequential(*reversed_layers)


def _validate_monotonic(dimensions: Sequence[int]) -> None:
    if not all(dimensions[i] > dimensions[i + 1] for i in range(len(dimensions) - 1)):
        raise ValueError("encoder dimensions must strictly decrease")


def linear_autoencoder(
    *,
    dimensions: Sequence[int],
    hidden_activation: ActivationSpec = "relu",
    output_activation: ActivationSpec = "identity",
    bias: bool = True,
    tied_weights: bool = False,
    validate_shapes: bool = False,
) -> AutoEncoder:
    dims = list(dimensions)
    if validate_shapes:
        _validate_monotonic(dims)
    encoder = _build_linear_stack(dims, bias=bias, activation=hidden_activation)
    decoder = _reverse_linear_stack(encoder, use_tied=tied_weights, activation=hidden_activation, bias=bias)
    use_base_tie = tied_weights and not any(isinstance(layer, TiedLinear) for layer in decoder)
    return create_autoencoder(encoder=encoder, decoder=decoder, activation=output_activation, tied_weights=use_base_tie)
