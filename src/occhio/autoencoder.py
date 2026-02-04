# ABOUTME: Defines a flexible autoencoder base and quick-build helpers.
# ABOUTME: Supports arbitrary encoder/decoder modules with optional tying and validation.

from __future__ import annotations

from typing import Callable, Sequence, Union
import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ActivationLike = Union[str, Callable[[Tensor], Tensor], nn.Module]


class _FunctionalActivation(nn.Module):
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)

def _resolve_activation(activation: ActivationLike) -> Callable[[Tensor], Tensor]:
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation):
        return activation
    module_cls = getattr(nn, str(activation), None)
    if isinstance(module_cls, type) and issubclass(module_cls, nn.Module):
        try:
            return module_cls()
        except TypeError:
            pass
    if activation == "identity":
        return lambda x: x
    fn = getattr(F, activation, None)
    if fn is None or not callable(fn):
        raise ValueError(f"invalid activation '{activation}'")
    return fn

class AutoEncoder(nn.Module):
    """
    Autoencoder base: encode → activation → decode.
    Accepts arbitrary encoder/decoder modules; supports optional parameter tying.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        activation: ActivationLike = "identity",
        tied_weights: bool = False,
    ):
        super().__init__()
        if decoder is None:
            raise ValueError("decoder is required")
        self.encoder = encoder
        self.activation = _resolve_activation(activation)
        self.tied_weights = tied_weights

        self.decoder = decoder

        if tied_weights:
            self._tie_shared_parameters(self.encoder, self.decoder)

    def encode(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    @property
    def W(self) -> dict[str, Tensor]:
        weights: dict[str, Tensor] = {}
        enc_w = getattr(self.encoder, "weight", None)
        if enc_w is not None:
            weights["encoder"] = enc_w
        dec_w = getattr(self.decoder, "weight", None) if self.decoder is not None else None
        if dec_w is not None:
            weights["decoder"] = dec_w
        if not weights:
            raise AttributeError("no accessible weights on encoder/decoder")
        return weights

    @staticmethod
    def _first_linear(module: nn.Module) -> nn.Linear | None:
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    return layer
        return None

    @staticmethod
    def _tie_shared_parameters(src: nn.Module, dst: nn.Module) -> None:
        src_map = {name: param for name, param in src.named_parameters(recurse=True)}
        for name, param in dst.named_parameters(recurse=True):
            other = src_map.get(name)
            if other is not None and other.shape == param.shape:
                _set_param_by_name(dst, name, other)


def _make_mlp(sizes: Sequence[int], activation: ActivationLike, bias: bool) -> nn.Module:
    layers: list[nn.Module] = []
    act_fn = _resolve_activation(activation)
    def make_act() -> nn.Module:
        if isinstance(act_fn, nn.Module):
            return copy.deepcopy(act_fn)
        return _FunctionalActivation(act_fn)
    for in_s, out_s in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_s, out_s, bias=bias))
        layers.append(make_act())
    layers = layers[:-1] if len(layers) > 1 else layers
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def create_autoencoder(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    activation: ActivationLike = "identity",
    tied_weights: bool = False,
) -> AutoEncoder:
    return AutoEncoder(encoder=encoder, decoder=decoder, activation=activation, tied_weights=tied_weights)


def linear_autoencoder(
    n_features: int,
    hidden_sizes: Union[int, Sequence[int]],
    *,
    activation: ActivationLike = "relu",
    bias: bool = False,
    tied_weights: bool = False,
    validate_shapes: bool = True,
) -> AutoEncoder:
    hidden_list = [hidden_sizes] if isinstance(hidden_sizes, int) else list(hidden_sizes)
    if not hidden_list:
        raise ValueError("hidden_sizes must be non-empty")
    if validate_shapes and hidden_list[-1] >= n_features:
        raise ValueError("latent dimension must be smaller than input size")

    sizes = [n_features, *hidden_list]
    encoder = _make_mlp(sizes, activation, bias)

    if tied_weights:
        decoder = _build_tied_decoder(encoder, activation)
    else:
        decoder_sizes = [hidden_list[-1], *reversed(sizes[:-1])]
        decoder = _make_mlp(decoder_sizes, activation, bias)

    return AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        activation=activation,
        tied_weights=tied_weights,
    )


def deep_autoencoder(
    layer_sizes: Sequence[int],
    *,
    activation: ActivationLike = "relu",
    bias: bool = True,
    validate_shapes: bool = False,
) -> AutoEncoder:
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least input and output dims")
    if validate_shapes and layer_sizes[0] <= layer_sizes[-1]:
        raise ValueError("encoder must compress input when validate_shapes=True")
    encoder = _make_mlp(layer_sizes, activation, bias)
    decoder = _make_mlp([layer_sizes[-1], *reversed(layer_sizes[:-1])], activation, bias)
    return AutoEncoder(encoder=encoder, decoder=decoder, activation=activation)


class TiedLinear(nn.Module):
    """Linear decode that reuses source Linear weight with transpose."""

    def __init__(self, source: nn.Linear, bias: bool = True):
        super().__init__()
        self.source = source
        self.bias = nn.Parameter(torch.zeros(source.in_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.source.weight.t(), self.bias)


def _build_tied_decoder(encoder: nn.Module, activation: ActivationLike) -> nn.Module:
    if isinstance(encoder, nn.Linear):
        return TiedLinear(encoder, bias=getattr(encoder, "bias", None) is not None)
    if not isinstance(encoder, nn.Sequential):
        raise TypeError("tied_weights requires decoder or a Sequential/Linear encoder to mirror")

    layers: list[nn.Module] = []
    resolved_act = _resolve_activation(activation)
    def make_act() -> nn.Module:
        if isinstance(resolved_act, nn.Module):
            return copy.deepcopy(resolved_act)
        return _FunctionalActivation(resolved_act)

    for layer in reversed(encoder):
        if isinstance(layer, nn.Linear):
            layers.append(TiedLinear(layer, bias=getattr(layer, "bias", None) is not None))
        elif isinstance(layer, (nn.ReLU, nn.Identity, nn.Dropout, _FunctionalActivation)):
            layers.append(make_act())
        else:
            raise TypeError(f"tied_weights cannot mirror layer type {type(layer).__name__}")
    return nn.Sequential(*layers)


def _set_param_by_name(module: nn.Module, name: str, param: nn.Parameter) -> None:
    parts = name.split(".")
    target = module
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], param)
