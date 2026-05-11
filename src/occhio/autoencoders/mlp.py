"""MLP-based autoencoders."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import AutoEncoderBase


class MLPEncoder(AutoEncoderBase):
    def __init__(
        self,
        embedding: list[int],
        unembedding: list[int],
        tied_initialization: bool = False,
        **kwargs,
    ):
        assert len(embedding) >= 2, "embedding must have at least [input, latent]"
        assert len(unembedding) >= 2, "unembedding must have at least [latent, output]"
        assert embedding[-1] == unembedding[0], "latent dims must match"
        assert embedding[0] == unembedding[-1], "input/output dims must match"
        if tied_initialization:
            assert unembedding == embedding[::-1], (
                "tied_initialization requires unembedding to be the reverse of embedding, "
                f"got embedding={embedding}, unembedding={unembedding}"
            )

        super().__init__(embedding[0], embedding[-1], **kwargs)

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
                w = nn.Parameter(self.encoder_weights[enc_idx].data.t().contiguous())
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

    def get_config(self) -> dict:
        return {
            "embedding": self.embedding_dims,
            "unembedding": self.unembedding_dims,
            "tied_initialization": self.tied_initialization,
        }

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
        assert len(dims) >= 2, "dims must have at least [input, latent]"

        self.dims = dims

        super().__init__(dims[0], dims[-1], **kwargs)

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
