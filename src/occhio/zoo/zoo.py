# ABOUTME: An interface to concurrently train and evaluate many toy models.
# ABOUTME: Build models in parallel with variation on some hyperparameter(s).

from __future__ import annotations
from ..autoencoder import AutoEncoderBase
from ..toy_model import ToyModel
from ..distributions import *
from ..datasets import *
from dataclasses import dataclass
# from .autoencoders import *

class Zoo:
    def __init__(self, ModelClass: type[AutoEncoderBase], config: ZooConfig, **kwargs):
        self.AutoEncoderClass = AutoEncoderClass
        self.config = config
        self.kwargs = kwargs
        self.box = None

    def build_models(self) -> list[ToyModel]:
        models = []
        for dist in self.config.distributions:
            for dist_param in self.config.distribution_params:
                for ae_param in self.config.autoencoder_params:
                    models.append(ToyModel(dist, ae, **ae_param))
        return models

@dataclass
class ZooConfig:
    """Configuration for a zoo to train and evaluate multiple models at once"""
    distributions: list[type[Distribution]]
    distribution_params: list[dict]
    autoencoder_params: list[dict]
    


"""
- build a bunch of models
- vary along:
-- distributions
-- distribution parameters
-- autoencoder parameters


"""