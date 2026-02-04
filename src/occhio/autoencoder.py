# ABOUTME: Defines a flexible autoencoder base and quick-build helpers.
# ABOUTME: Supports arbitrary encoder/decoder modules with optional tying and validation.

from __future__ import annotations

from typing import Callable, Iterable, Sequence, Union

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
    if activation == "identity":
        return lambda x: x
    fn = getattr(F, activation, None)
    if fn is None or not callable(fn):
        raise ValueError(f"activation '{activation}' not found in torch.nn.functional")
    return fn


class AutoEncoder(nn.Module):
    """
    Autoencoder base: encode → activation → decode.
    Accepts arbitrary encoder/decoder modules; supports optional tied decoding.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        activation: ActivationLike = "relu",
        tied_weights: bool = False,
        validate_shapes: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.activation = _resolve_activation(activation)
        self.tied_weights = tied_weights
        self.validate_shapes = validate_shapes

        encoder_out = getattr(self.encoder, "out_features", None)
        encoder_in = getattr(self.encoder, "in_features", None)

        if decoder is None and not tied_weights:
            if encoder_out is None or encoder_in is None:
                raise ValueError("decoder missing and encoder lacks dimension metadata")
            decoder = nn.Linear(encoder_out, encoder_in, bias=False)

        self.decoder = decoder

        if tied_weights:
            if not isinstance(self.encoder, nn.Linear):
                raise TypeError("tied_weights only supported for single Linear encoder")
            if decoder is None:
                bias = getattr(self.encoder, "bias", None)
                self.decoder_bias = nn.Parameter(torch.zeros(self.encoder.in_features)) if bias is not None else None
            else:
                self.decoder_bias = getattr(decoder, "bias", None)
        else:
            self.decoder_bias = None

        if self.validate_shapes:
            self._validate_shapes()

    def encode(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        if self.decoder is None:
            raise RuntimeError("decoder missing (tied_weights=False)")
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

    def _validate_shapes(self) -> None:
        first_linear = self._first_linear(self.encoder)
        if first_linear and first_linear.out_features >= first_linear.in_features:
            raise ValueError("encoder out_features must be < in_features")
        if not self.tied_weights and self.decoder:
            dec_linear = self._first_linear(self.decoder)
            if first_linear and dec_linear:
                if dec_linear.in_features != first_linear.out_features or dec_linear.out_features != first_linear.in_features:
                    raise ValueError("decoder dims must mirror encoder (hidden→input)")

    @staticmethod
    def _first_linear(module: nn.Module) -> nn.Linear | None:
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    return layer
        return None


def _make_mlp(sizes: Sequence[int], activation: ActivationLike, bias: bool) -> nn.Module:
    layers: list[nn.Module] = []
    act_fn = _resolve_activation(activation)
    act_module: nn.Module = act_fn if isinstance(act_fn, nn.Module) else _FunctionalActivation(act_fn)
    for in_s, out_s in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_s, out_s, bias=bias))
        layers.append(act_module)
    layers = layers[:-1] if len(layers) > 1 else layers
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def linear_autoencoder(
    n_features: int,
    hidden_sizes: Union[int, Sequence[int]],
    *,
    activation: ActivationLike = "relu",
    bias: bool = False,
    tied_weights: bool = False,
    validate_shapes: bool = True,
) -> AutoEncoder:
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    sizes = [n_features, *hidden_sizes]
    encoder = _make_mlp(sizes, activation, bias)

    if tied_weights:
        if len(hidden_sizes) != 1 or not isinstance(encoder, nn.Linear):
            raise ValueError("tied_weights only supports single-layer linear encoder")
        decoder = None
    else:
        decoder_sizes = [hidden_sizes[-1], *reversed(sizes[:-1])]
        decoder = _make_mlp(decoder_sizes, activation, bias)

    return AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        activation=activation,
        tied_weights=tied_weights,
        validate_shapes=validate_shapes,
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
    encoder = _make_mlp(layer_sizes, activation, bias)
    decoder = _make_mlp([layer_sizes[-1], *reversed(layer_sizes[:-1])], activation, bias)
    return AutoEncoder(encoder=encoder, decoder=decoder, activation=activation, validate_shapes=validate_shapes)
