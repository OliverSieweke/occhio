"""Implements simple"""

import functools
from math import sqrt
from torch import Tensor
import torch.nn.functional as F
from typing import Literal, Callable
import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import math
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
    def __init__(
        self,
        embedding: list[int],
        unembedding: list[int],
        tied_initialization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(embedding) >= 2, "embedding must have at least [input, latent]"
        assert len(unembedding) >= 2, "unembedding must have at least [latent, output]"
        assert embedding[-1] == unembedding[0], "latent dims must match"
        assert embedding[0] == unembedding[-1], "input/output dims must match"
        if tied_initialization:
            assert unembedding == embedding[::-1], (
                "tied_initialization requires unembedding to be the reverse of embedding, "
                f"got embedding={embedding}, unembedding={unembedding}"
            )

        self.n_features = embedding[0]
        self.n_hidden = embedding[-1]

        self.embedding_dims = embedding
        self.unembedding_dims = unembedding
        self.tied_initialization = tied_initialization

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
            if self.tied_initialization:
                # Initialize decoder layer i as transpose of mirrored encoder layer
                enc_idx = len(self.encoder_weights) - 1 - i
                w = nn.Parameter(self.encoder_weights[enc_idx].data.t().clone())
            else:
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
            if not self.tied_initialization:
                self._init_param(w, b)
            else:
                fan_in = self.unembedding_dims[i]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound, generator=self.generator)
            self.decoder_weights.append(w)
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
        for i, (w, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            z = z @ w.t() + b
            if i < len(self.decoder_weights) - 1:
                z = F.leaky_relu(z)
        z = torch.relu(z)  # ReLU on final output (outputs are non-negative)
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


class AttnLinearAE(AutoEncoderBase):
    """Multi-head softmax bottleneck autoencoder for MRH-style experiments.

    Each encoder head projects the input to ``dict_size`` logits, applies softmax, then multiplies by a learned value matrix to produce a ``value_dim``-sized output.  The latent is the concatenation of all heads' outputs (``n_hidden = n_heads * value_dim``).  The decoder is a linear projection with ReLU, mirroring :class:`TiedLinearRelu`.

    The softmax constraint forces each head's contribution to be a convex combination of its dictionary vectors, providing the architectural inductive bias for Minkowski-style tile representations.

    Parameters
    ----------
    n_features : int
        Input / output dimensionality.
    n_hidden : int
        Total latent dimensionality (must be divisible by ``n_heads``).
    n_heads : int
        Number of independent softmax heads.
    dict_size : int
        Number of dictionary elements (archetypes) per head.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        dict_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if n_hidden % n_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.dict_size = dict_size
        self.value_dim = n_hidden // n_heads

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        dev = self.device
        gen = self.generator

        # Per-head encoder projections: (n_features, dict_size)
        self.encoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.n_features,
                        self.dict_size,
                        device=dev,
                        generator=gen,
                    )
                    / sqrt(self.n_features)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.W_mix = nn.Parameter(torch.zeros(self.n_hidden, self.n_hidden, device=dev))

        # self.W_mix = nn.Parameter(
        #     torch.randn(self.n_hidden, self.n_hidden, device=dev, generator=gen)
        #     / sqrt(self.n_hidden)
        # )

        # self.W_skip = nn.Parameter(
        #     torch.randn(
        #         self.n_features,
        #         self.n_hidden,
        #         device=dev,
        #         generator=gen,
        #     )
        #     / sqrt(self.n_hidden)
        # )

        # Per-head value matrices: (dict_size, value_dim)
        self.value_matrices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size,
                        self.value_dim,
                        device=dev,
                        generator=gen,
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        # Decoder: linear projection from latent to features
        self.W_out = nn.Parameter(
            torch.randn(
                self.n_hidden,
                self.n_features,
                device=dev,
                generator=gen,
            )
            / sqrt(self.n_hidden)
        )
        self.b = nn.Parameter(torch.zeros(self.n_features, device=dev))

    def encode(self, x: Tensor) -> Tensor:
        parts = []
        for h in range(self.n_heads):
            logits = x @ self.encoder_projs[h]  # (B, dict_size)
            weights = F.softmax(logits, dim=-1)  # (B, dict_size)
            values = weights @ self.value_matrices[h]  # (B, value_dim)
            parts.append(values)
        z = torch.cat(parts, dim=-1) @ self.W_mix
        return z + x @ self.W_out.T

    def decode(self, z: Tensor) -> Tensor:
        return torch.relu(z @ self.W_out + self.b)


class AttnAttnAE(AutoEncoderBase):
    """Multi-head softmax autoencoder with attention-like encoding and decoding.

    Encoder: ``softmax(x @ P_h) @ V_h`` per head, concatenated to latent.
    Decoder: ``softmax(z_h @ Q_h) @ U_h`` per head, summed + bias + ReLU.

    Both encoder and decoder use independent per-head softmax-weighted
    dictionary lookups with separate parameters.

    Parameters
    ----------
    n_features : int
        Input / output dimensionality.
    n_hidden : int
        Total latent dimensionality (must be divisible by ``n_heads``).
    n_heads : int
        Number of independent softmax heads.
    dict_size : int
        Number of dictionary elements (archetypes) per head.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        dict_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if n_hidden % n_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.dict_size = dict_size
        self.value_dim = n_hidden // n_heads

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        dev = self.device
        gen = self.generator

        # --- Encoder ---
        # Per-head encoder projections: (n_features, dict_size)
        self.encoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.n_features, self.dict_size, device=dev, generator=gen
                    )
                    / sqrt(self.n_features)
                )
                for _ in range(self.n_heads)
            ]
        )
        # Per-head encoder value matrices: (dict_size, value_dim)
        self.encoder_values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size, self.value_dim, device=dev, generator=gen
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        # self.W_mix = nn.Parameter(
        #     torch.randn(self.n_hidden, self.n_hidden, device=dev, generator=gen)
        #     / sqrt(self.n_hidden)
        # )
        self.W_mix = nn.Parameter(torch.zeros(self.n_hidden, self.n_hidden, device=dev))

        self.W_skip = nn.Parameter(
            torch.randn(
                self.n_features,
                self.n_hidden,
                device=dev,
                generator=gen,
            )
            / sqrt(self.n_hidden)
        )
        # --- Decoder ---
        # Per-head decoder projections: (value_dim, dict_size)
        self.decoder_projs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.value_dim, self.dict_size, device=dev, generator=gen
                    )
                    / sqrt(self.value_dim)
                )
                for _ in range(self.n_heads)
            ]
        )
        # Per-head decoder value matrices: (dict_size, n_features)
        self.decoder_values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dict_size, self.n_features, device=dev, generator=gen
                    )
                    / sqrt(self.dict_size)
                )
                for _ in range(self.n_heads)
            ]
        )

        self.b = nn.Parameter(torch.zeros(self.n_features, device=dev))

    def encode(self, x: Tensor) -> Tensor:
        parts = []
        for h in range(self.n_heads):
            logits = x @ self.encoder_projs[h]  # (B, dict_size)
            weights = F.softmax(logits, dim=-1)  # (B, dict_size)
            parts.append(weights @ self.encoder_values[h])  # (B, value_dim)
        z = torch.cat(parts, dim=-1) @ self.W_mix  # (B, n_hidden)
        return z + x @ self.W_skip

    def decode(self, z: Tensor) -> Tensor:
        chunks = z.split(self.value_dim, dim=-1)  # n_heads x (B, value_dim)
        out = torch.zeros(z.shape[0], self.n_features, device=z.device)
        for h in range(self.n_heads):
            logits = chunks[h] @ self.decoder_projs[h]  # (B, dict_size)
            weights = F.softmax(logits, dim=-1)  # (B, dict_size)
            out = out + weights @ self.decoder_values[h]  # (B, n_features)
        return torch.relu(out + self.b)


class SynthAE(AutoEncoderBase):
    """Autoencoder with unit-norm tied weights and optional orthogonalization.

    Feature vectors (columns of W) are initialized as random unit vectors
    sampled from a standard normal distribution:

        d_i = g_i / ||g_i||_2,  g_i ~ N(0, I_D)

    When ``orthogonalize=True``, a gradient descent procedure minimizes
    pairwise cosine similarity between columns:

        L_ortho = Σ_{i≠j} (d_i^T d_j)^2 + λ Σ_i (||d_i||_2 - 1)^2

    After orthogonalization, all vectors are renormalized to unit length.
    Uses a chunked implementation to reduce memory from O(N^2) to
    O(chunk_size × N).

    Parameters
    ----------
    n_features : int
        Number of ground-truth features N.
    n_hidden : int
        Hidden dimension D (must satisfy N ≥ D for meaningful superposition).
    orthogonalize : bool
        Run the orthogonalization procedure on init.
    ortho_lambda : float
        Norm penalty weight λ in L_ortho.
    ortho_steps : int
        Number of gradient descent iterations.
    ortho_lr : float
        Learning rate for the orthogonalization optimizer.
    ortho_chunk_size : int
        Block size for chunked pairwise dot products.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        orthogonalize: bool = False,
        ortho_lambda: float = 1.0,
        ortho_steps: int = 1000,
        ortho_lr: float = 0.01,
        ortho_chunk_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_hidden = n_hidden
        self._orthogonalize = orthogonalize
        self._ortho_lambda = ortho_lambda
        self._ortho_steps = ortho_steps
        self._ortho_lr = ortho_lr
        self._ortho_chunk_size = ortho_chunk_size

        self.resample_weights()

    def resample_weights(self, force_norm=False):
        # W is (n_hidden, n_features) — each column is a unit-norm feature direction in R^D
        g = torch.randn(
            self.n_hidden,
            self.n_features,
            generator=self.generator,
            device=self.device,
        )
        # Normalize columns to unit length
        g = g / g.norm(dim=0, keepdim=True)

        if self._orthogonalize:
            g = self._run_orthogonalization(g)

        self.W = nn.Parameter(g)
        self.b = nn.Parameter(torch.zeros(self.n_features, device=self.device))

    def _run_orthogonalization(self, W: Tensor) -> Tensor:
        """Minimize pairwise cosine similarity via gradient descent.

        Operates on W of shape (D, N) where columns are feature directions.
        """
        D = W.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([D], lr=self._ortho_lr)
        N = D.shape[1]
        chunk = self._ortho_chunk_size

        for _ in range(self._ortho_steps):
            optimizer.zero_grad()

            # Chunked pairwise dot products between columns
            # D^T D gives (N, N) cosine similarities (columns are unit-norm)
            ortho_loss = torch.tensor(0.0, device=D.device, dtype=D.dtype)
            for i in range(0, N, chunk):
                cols_i = D[:, i : i + chunk]  # (D, chunk)
                dots = cols_i.T @ D  # (chunk, N)
                # Zero self-similarities
                end = min(i + chunk, N)
                for k in range(end - i):
                    dots[k, i + k] = 0.0
                ortho_loss = ortho_loss + (dots**2).sum()

            # Norm penalty on columns
            col_norms = D.norm(dim=0)
            norm_loss = self._ortho_lambda * ((col_norms - 1.0) ** 2).sum()

            loss = ortho_loss + norm_loss
            loss.backward()
            optimizer.step()

        # Re-normalize columns to unit length
        with torch.no_grad():
            D.div_(D.norm(dim=0, keepdim=True))
        return D.detach()

    @property
    def rho_mm(self) -> float:
        """Mean max absolute cosine similarity (superposition metric).

        ρ_mm = (1/N) Σ_i max_{j≠i} |d_i^T d_j|

        Returns 0 for fully orthogonal features, approaches 1 for full superposition.
        """
        W = self.W.detach()
        N = W.shape[1]
        max_cos = torch.zeros(N, device=W.device, dtype=W.dtype)
        chunk = min(self._ortho_chunk_size, N)
        for i in range(0, N, chunk):
            sims = (W[:, i : i + chunk].T @ W).abs()  # (chunk, N)
            end = min(i + chunk, N)
            for k in range(end - i):
                sims[k, i + k] = 0.0
            max_cos[i : i + chunk] = sims.max(dim=1).values
        return max_cos.mean().item()

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.W.T

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W + self.b
