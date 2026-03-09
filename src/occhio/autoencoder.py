"""Implements simple"""

import functools
import math
from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils.device import _same_device


class AutoEncoderBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """features --> latent"""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """latent --> features"""

    @abstractmethod
    def resample_weights(self):
        """Reset / resample all weights"""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x_true: Tensor, x_hat: Tensor, importances: Tensor | None):
        """The associated loss function."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        return torch.mean(torch.sum(importances * torch.square(x_true - x_hat), dim=-1))

    def __init__(
        self,
        loss_fn: Callable | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ):
        """Initialize the AutoEncoder class.

        Note that we write device to `_init_device`, which remembers where the user intends to store the device.
        """
        super().__init__()
        if loss_fn is not None:
            self.loss = loss_fn  # type: ignore[method-assign]
        if device is not None and generator is not None:
            gen_device = torch.device(generator.device)
            dev = torch.device(device)
            if not _same_device(gen_device, dev):
                raise ValueError(
                    f"Generator lives on {gen_device}, but device is {dev}. "
                    f"These must match."
                )
        if device is not None:
            self._init_device = torch.device(device)
        elif generator is not None:
            self._init_device = torch.device(generator.device)
        else:
            self._init_device = None
        self.generator = generator

    @property
    def device(self) -> torch.device | None:
        """Return the device of the first parameter, falling back to the
        device passed at construction time (needed during ``__init__`` before
        any parameters have been created)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._init_device

    def __init_subclass__(cls, **kwargs):
        """This ensures that `n_features` and `n_hidden` are defined at creation"""
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        @functools.wraps(original_init)
        def checked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            for attr in ("n_features", "n_hidden"):
                if not hasattr(self, attr):
                    raise AttributeError(
                        f"{cls.__name__}.__init__ must set self.{attr}"
                    )

        cls.__init__ = checked_init  # ty:ignore


class TiedLinear(AutoEncoderBase):
    def __init__(self, n_features: int, n_hidden: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        self.W = nn.Parameter(
            torch.randn(
                self.n_hidden,
                self.n_features,
                generator=self.generator,
                device=self.device,
            )
            / sqrt(self.n_hidden)
        )
        with torch.no_grad():
            norms = self.W.data.norm(dim=0, keepdim=True)
            self.W.data /= norms
        self.b = nn.Parameter(torch.zeros(self.n_features, device=self.device))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W + self.b


class TiedLinearRelu(AutoEncoderBase):
    def __init__(self, n_features: int, n_hidden: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        self.W = nn.Parameter(
            torch.randn(
                self.n_hidden,
                self.n_features,
                generator=self.generator,
                device=self.device,
            )
            / sqrt(self.n_hidden)
        )
        with torch.no_grad():
            norms = self.W.data.norm(dim=0, keepdim=True)
            self.W.data /= norms
        self.b = nn.Parameter(torch.zeros(self.n_features, device=self.device))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W + self.b)


class MLPEncoder(AutoEncoderBase):
    def __init__(self, embedding: list[int], unembedding: list[int], **kwargs):
        super().__init__(**kwargs)

        assert len(embedding) >= 2, "embedding must have at least [input, latent]"
        assert len(unembedding) >= 2, "unembedding must have at least [latent, output]"
        assert embedding[-1] == unembedding[0], "latent dims must match"
        assert embedding[0] == unembedding[-1], "input/output dims must match"

        self.n_features = embedding[0]
        self.n_hidden = embedding[-1]

        self.embedding_dims = embedding
        self.unembedding_dims = unembedding

        self._build_layers()

    def _build_layers(self):
        self.encoder_weights = nn.ParameterList()
        self.encoder_biases = nn.ParameterList()
        for i in range(len(self.embedding_dims) - 1):
            w = nn.Parameter(
                torch.empty(
                    self.embedding_dims[i + 1],
                    self.embedding_dims[i],
                    device=self.device,
                )
            )
            b = nn.Parameter(
                torch.empty(self.embedding_dims[i + 1], device=self.device)
            )
            self._init_param(w, b)
            self.encoder_weights.append(w)
            self.encoder_biases.append(b)

        self.decoder_weights = nn.ParameterList()
        self.decoder_biases = nn.ParameterList()
        for i in range(len(self.unembedding_dims) - 1):
            w = nn.Parameter(
                torch.empty(
                    self.unembedding_dims[i + 1],
                    self.unembedding_dims[i],
                    device=self.device,
                )
            )
            b = nn.Parameter(
                torch.empty(self.unembedding_dims[i + 1], device=self.device)
            )
            self._init_param(w, b)
            self.decoder_weights.append(w)
            self.decoder_biases.append(b)

    def _init_param(self, w: nn.Parameter, b: nn.Parameter):
        nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=self.generator)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(b, -bound, bound, generator=self.generator)

    def encode(self, x: Tensor) -> Tensor:
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            x = x @ w.t()
            if i < len(self.encoder_weights) - 1:
                x = torch.relu(x)
        return x

    def decode(self, z: Tensor) -> Tensor:
        for i, (w, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            z = z @ w.t() + b
            if i < len(self.decoder_weights) - 1:
                z = torch.relu(z)
        z = torch.relu(z)  # ReLU on final output, matching your original
        return z

    def resample_weights(self, force_norm=False):
        self._build_layers()


class TiedMLPEncoder(AutoEncoderBase):
    """MLP autoencoder with tied weights: decoder reuses encoder weights transposed.

    Only the encoder weights and per-layer decoder biases are learned.
    This gives the MLP extra capacity over TiedLinearRelu while preserving
    the encoder-decoder symmetry that helps with superposition geometry.

    Parameters
    ----------
    dims : list[int]
        Layer dimensions from input to latent, e.g. [200, 200, 20].
        The decoder mirrors this in reverse.
    """

    def __init__(self, dims: list[int], **kwargs):
        super().__init__(**kwargs)

        assert len(dims) >= 2, "dims must have at least [input, latent]"

        self.n_features = dims[0]
        self.n_hidden = dims[-1]
        self.dims = dims

        self._build_layers()

    def _build_layers(self):
        self.encoder_weights = nn.ParameterList()
        self.encoder_biases = nn.ParameterList()
        for i in range(len(self.dims) - 1):
            w = nn.Parameter(
                torch.empty(self.dims[i + 1], self.dims[i], device=self.device)
            )
            b = nn.Parameter(torch.empty(self.dims[i + 1], device=self.device))
            self._init_param(w, b)
            self.encoder_weights.append(w)
            self.encoder_biases.append(b)

        # Decoder only needs its own biases; weights are tied (encoder transposed)
        self.decoder_biases = nn.ParameterList()
        for i in range(len(self.dims) - 2, -1, -1):
            b = nn.Parameter(torch.zeros(self.dims[i], device=self.device))
            self.decoder_biases.append(b)

    def _init_param(self, w: nn.Parameter, b: nn.Parameter):
        nn.init.kaiming_uniform_(w, a=0.01, generator=self.generator)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(b, -bound, bound, generator=self.generator)

    def encode(self, x: Tensor) -> Tensor:
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            x = x @ w.t() + b
            if i < len(self.encoder_weights) - 1:
                x = F.leaky_relu(x)
        return x

    def decode(self, z: Tensor) -> Tensor:
        # Walk encoder weights in reverse order
        rev_weights = list(reversed(list(self.encoder_weights)))
        for i, (w, b) in enumerate(zip(rev_weights, self.decoder_biases)):
            z = z @ w + b  # w (not w.t()) — transposed relative to encoder
            if i < len(rev_weights) - 1:
                z = F.leaky_relu(z)
        z = torch.relu(z)
        return z

    def resample_weights(self, force_norm=False):
        self._build_layers()


class MLPAutoencoder(AutoEncoderBase):
    """MLP-based autoencoder with nonlinear encoder for manifold representations.

    Uses a smooth nonlinearity (GELU by default) to enable curved feature manifolds,
    while keeping the decoder linear (or with optional activation) to isolate
    nonlinearity to the encoding side.

    Architecture:
        Encoder: x ∈ ℝⁿ → σ(V · σ(Wx + b₁) + b₂) → h ∈ ℝᵐ
        Decoder: h ∈ ℝᵐ → σ_dec(W_dec · h + b_dec) → x̂ ∈ ℝⁿ

    Parameters
    ----------
    n_features : int
        Input/output dimension (number of features).
    n_hidden : int
        Bottleneck/latent dimension.
    encoder_hidden_dim : int, optional
        Hidden layer width in the encoder MLP. Defaults to 2 * n_hidden.
    activation : str, optional
        Nonlinearity to use in encoder: "gelu" (default), "tanh", or "silu".
        ReLU is intentionally excluded as it's piecewise linear.
    decoder_activation : str or None, optional
        Activation on decoder output: None (default, linear), "relu", or "softplus".
        Use "relu" for non-negative data like SparseUniform.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        encoder_hidden_dim: int | None = None,
        activation: str = "gelu",
        decoder_activation: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.encoder_hidden_dim = (
            encoder_hidden_dim if encoder_hidden_dim is not None else 2 * n_hidden
        )

        # Select encoder activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Use 'gelu', 'tanh', or 'silu'."
            )
        self._activation_name = activation

        # Select decoder activation function
        if decoder_activation is None:
            self.decoder_activation = None
        elif decoder_activation == "relu":
            self.decoder_activation = torch.relu
        elif decoder_activation == "softplus":
            self.decoder_activation = F.softplus
        else:
            raise ValueError(
                f"Unsupported decoder_activation '{decoder_activation}'. "
                f"Use None, 'relu', or 'softplus'."
            )
        self._decoder_activation_name = decoder_activation

        self.resample_weights()

    def resample_weights(self) -> None:
        """Initialize or reset all weights using Kaiming initialization."""
        # Encoder: x -> hidden -> latent
        # Layer 1: n_features -> encoder_hidden_dim
        self.enc_W1 = nn.Parameter(
            torch.empty(
                self.encoder_hidden_dim,
                self.n_features,
                device=self.device,
            )
        )
        self.enc_b1 = nn.Parameter(
            torch.empty(self.encoder_hidden_dim, device=self.device)
        )

        # Layer 2: encoder_hidden_dim -> n_hidden
        self.enc_W2 = nn.Parameter(
            torch.empty(
                self.n_hidden,
                self.encoder_hidden_dim,
                device=self.device,
            )
        )
        self.enc_b2 = nn.Parameter(torch.empty(self.n_hidden, device=self.device))

        # Decoder: latent -> features (linear)
        self.dec_W = nn.Parameter(
            torch.empty(
                self.n_features,
                self.n_hidden,
                device=self.device,
            )
        )
        self.dec_b = nn.Parameter(torch.empty(self.n_features, device=self.device))

        # Initialize weights
        self._init_linear(self.enc_W1, self.enc_b1)
        self._init_linear(self.enc_W2, self.enc_b2)
        self._init_linear(self.dec_W, self.dec_b)

    def _init_linear(self, w: nn.Parameter, b: nn.Parameter) -> None:
        """Kaiming initialization for a linear layer."""
        nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=self.generator)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(b, -bound, bound, generator=self.generator)

    def encode(self, x: Tensor) -> Tensor:
        """Encode input through MLP with smooth nonlinearity."""
        # x: (batch, n_features)
        h = x @ self.enc_W1.T + self.enc_b1  # (batch, encoder_hidden_dim)
        h = self.activation(h)
        h = h @ self.enc_W2.T + self.enc_b2  # (batch, n_hidden)
        # h = self.activation(h)
        return h

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation, optionally applying activation."""
        out = z @ self.dec_W.T + self.dec_b
        if self.decoder_activation is not None:
            out = self.decoder_activation(out)
        return out


class ComputeAutoEncoder(AutoEncoderBase):
    """
    Autoencoder with a tied encoder/decoder and a linear compute step.

    Subclasses occhio's AutoEncoderBase so it exposes encode/decode and slots
    into ToyModel for geometric analysis (feature norms, interferences, etc.).

    Parameters
    ----------
    N : int   — number of features
    k : int   — hidden / latent dimension
    decode_activation : "softmax" | "relu"
        "softmax" — outputs a probability simplex; use for one-hot targets (CE/MSE).
        "relu"    — outputs non-negative values; use for continuous targets like x @ P.
    seed : int — weight init seed

    Weights
    -------
    W : (k, N) — tied encoder / decoder
    Z : (k, k) — linear compute step
    b : (N,)   — decode bias
    """

    def __init__(
        self,
        N: int,
        k: int,
        decode_activation: Literal["softmax", "relu"] = "softmax",
        seed: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_features = N
        self.n_hidden = k
        self.decode_activation = decode_activation

        gen = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(k, N, generator=gen) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen) / k)
        self.b = nn.Parameter(torch.zeros(N))

    # ── core operations ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, k)  : embed into latent space."""
        return x @ self.W.T

    def compute_step(self, h: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, k)  : linear compute / routing step."""
        return h + h @ self.Z.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, N)  : project back, then activate."""
        logits = z @ self.W + self.b
        if self.decode_activation == "softmax":
            return F.softmax(logits, dim=-1)
        return F.relu(logits)

    def forward(self, x: torch.Tensor):
        """(B, N) → (y_hat, z)."""
        h = self.encode(x)
        z = self.compute_step(h)
        y_hat = self.decode(z)
        return y_hat, z

    def ce_loss(
        self, y_hat: torch.Tensor, y_idx: torch.Tensor, importances: torch.Tensor
    ) -> torch.Tensor:
        """Importance-weighted NLL given softmax output probabilities."""
        per_sample = F.nll_loss(y_hat.clamp(min=1e-9).log(), y_idx, reduction="none")
        weights = importances[y_idx]
        return (per_sample * weights).mean()

    def mse_loss(
        self, y_hat: torch.Tensor, y_true: torch.Tensor, importances: Tensor | None
    ) -> torch.Tensor:
        """Importance-weighted MSE. Prediction first, target second (mirrors ce_loss)."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        per_sample = (y_true - y_hat).pow(2).sum(dim=-1)
        weights = importances[y_hat.argmax(dim=-1)]
        return (per_sample * weights).mean()

    def loss(
        self, x_true: Tensor, x_hat: Tensor, importances: Tensor | None
    ) -> torch.Tensor:
        """Importance-weighted MSE between predicted probs and one-hot target."""
        if importances is None:
            importances = torch.ones(self.n_features, device=self.device)  # ty:ignore
        per_sample = (x_true - x_hat).pow(2).sum(dim=-1)
        weights = importances[x_hat.argmax(dim=-1)]
        return (per_sample * weights).mean()

    def resample_weights(self):
        gen = self.generator or torch.Generator()
        N, k = self.n_features, self.n_hidden
        self.W = nn.Parameter(torch.randn(k, N, generator=gen) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen) / k)
        self.b = nn.Parameter(torch.zeros(N))
