"""Autoencoder modules for occhio."""

from enum import Enum

from .attention import AttnAttnAE, AttnLinearAE, LinearAttnAE
from .base import AutoEncoderBase
from .compute import ComputeAutoEncoder
from .mlp import MLPEncoder, TiedMLPEncoder
from .synth import SynthAE
from .tied import TiedLinear, TiedLinearRelu


class AutoencoderType(Enum):
    """Enum for autoencoder types to enable picklable grid parameterization."""

    TiedLinear = TiedLinear.__name__
    TiedLinearRelu = TiedLinearRelu.__name__
    MLPEncoder = MLPEncoder.__name__
    TiedMLPEncoder = TiedMLPEncoder.__name__
    ComputeAutoEncoder = ComputeAutoEncoder.__name__
    AttnLinearAE = AttnLinearAE.__name__
    AttnAttnAE = AttnAttnAE.__name__
    LinearAttnAE = LinearAttnAE.__name__
    SynthAE = SynthAE.__name__

    def __str__(self) -> str:
        return self.name


__all__ = [
    "AutoEncoderBase",
    "AutoencoderType",
    "AttnAttnAE",
    "AttnLinearAE",
    "ComputeAutoEncoder",
    "LinearAttnAE",
    "MLPEncoder",
    "SynthAE",
    "TiedLinear",
    "TiedLinearRelu",
    "TiedMLPEncoder",
]
