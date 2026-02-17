from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from itertools import product
from typing import Any, Callable, Dict, List

import numpy as np

from torch import Tensor, meshgrid, cartesian_prod, Generator, arange, as_tensor, logspace
from stack_data.utils import cached_property

from .occhio.toy_model import ToyModel
from .occhio.autoencoder import *
from .occhio.distributions import *


@dataclass
class Axis:
    label: str
    values: list | Tensor

class ModelGrid:
    def __init__(self, create_model: Callable[..., ToyModel], axes: List[Axis]):
        self._validate_args(create_model, axes)
        self.axes: list[Axis] = axes
        self.create_model: Callable[..., ToyModel] = create_model
        self.models: np.ndarray = self._initialize_models()

    def _initialize_models(self) -> np.ndarray:
        if not self.axes:
            raise ValueError("At least one axis must be provided.")

        shape: tuple[int, ...] = self.shape()
        models: np.ndarray = np.empty(shape, dtype=object)

        # Iterates over combinations of axis indices
        for indices in product(*[range(s) for s in shape]):
            # Build a parameter dictionary mapping axis labels to selected values at current index
            params: Dict[str, Any] = {str(axis.label): axis.values[i] for axis, i in zip(self.axes, indices)}
            models[indices] = self.create_model(params=params)
        return models

    @cached_property
    def parameters_mesh(self):
        """
        Returns a tuple of the meshgrid of the axes.
        """
        if all(isinstance(axis.values, Tensor) for axis in self.axes):
            return meshgrid(*(axis.values for axis in self.axes), indexing="ij")
        # If any axis' values are not Tensors, produce an object meshgrid manually
        index_mesh = meshgrid(*(arange(len(axis.values)) for axis in self.axes), indexing="ij")
        result = []
        for axis, idx_grid in zip(self.axes, index_mesh):
            vals = axis.values
            if isinstance(vals, Tensor):
                result.append(vals[idx_grid])
            else:
                # as_tensor might not work if vals is not numeric; fallback to pure Python
                try:
                    tensor_vals = as_tensor(vals)
                    result.append(tensor_vals[idx_grid])
                except Exception:
                    # Fallback: object array mesh, matching meshgrid shape
                    v = np.array(vals, dtype=object)
                    # mesh index will be the right shape; get values by advanced indexing
                    result.append(np.array(v)[idx_grid.numpy()])
        return tuple(result)


    @cached_property
    def cartesian_parameters(self):
        """
        Returns a tuple of the cartesian product of the axes.
        Warning: This may cause memory explosions for large grids.
        """
        return cartesian_prod(*(axis.values for axis in self.axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the axes that define the nested structure of the models."""
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def describe(self) -> dict[str, int]:
        """Returns a dictionary of the axis labels and their lengths."""
        return {axis.label: len(axis.values) for axis in self.axes}

    def __getitem__(self, key):
        return self.models[key]

    def _validate_args(self, create_model: Callable[..., ToyModel], axes: List[Axis]) -> None:
        if not all(isinstance(axis.values, Tensor) for axis in vectorized_axes):
            labels = [axis.label for axis in vectorized_axes if not isinstance(axis.values, Tensor)]
            raise TypeError(
                f"Axes {labels} must have values as torch.Tensor"
            )

        assert set(vectorized_axes).isdisjoint(set(stratified_axes)), "vectorized_axes and stratified_axes must be disjoint sets."

        if not vectorized_axes and not stratified_axes:
            raise ValueError("At least one of 'vectorized_axes' or 'stratified_axes' must be provided and non-empty.")

        if "params" not in signature(create_model).parameters:
            raise TypeError("create_model must accept a 'params' parameter (Dict[str, Any]).")




# CONSTANTS
N_FEATURES = 2
N_HIDDEN = 1
P_INDIVIDUAL = 1
P_FOLLOW = 1
DATA = "uniform"
EXPERIMENT_SIZE = 24
distribution_generator = Generator("cpu").manual_seed(7)

# EXPERIMENTAL SETUP
densities = logspace(0, -2, EXPERIMENT_SIZE)
importances = logspace(-1, 1, EXPERIMENT_SIZE)
random_seeds = range(10, 20)


def create_model(
    params: Dict[str, Any] = {}, 
    default_model: ToyModel | None = None, 
    *args, 
    **kwargs
) -> ToyModel:
    density = params["Density"]
    relative_importance = params["Importance"]
    random_seed = params["Random Seeds"]

    model = ToyModel(
        distribution=SparseUniform(N_FEATURES, p_active=density, generator=Generator(device="cpu").manual_seed(42)),
        ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=Generator(device="cpu").manual_seed(random_seed)),
        importances=relative_importance ** arange(N_FEATURES),
    ) 
    return model


model_grid = ModelGrid(

    create_model, 
    axes = [
        Axis(label="Density", values=densities),
        Axis(label="Importance", values=importances),
        Axis(label="Random Seeds", values=random_seeds)
        ],

    # vectorized_axes = [Axis(label="Density", values=densities), 
    #         Axis(label="Density", values=densities),
    #         Axis(label="Importance", values=importances),
    #         Axis(label="Random Seeds", values=random_seeds)
    #     ],
    # stratified_axes = [
    #     Axis(label="Distribution", values=distributions)
    #     ],
    )

"""
Things not to forget about:
- Vectorized axes are not implemented yet.
- Stratified axes are not implemented yet.
- Slice some subset of ModelGrid for training and evaluation independetly
---> if users slice subset of ModelGrid for training, and the slice returns a new ModelGrid object, make sure to update the original ModelGrid object in place.
---> Make sure we're not overriting the newly-trained models
- Have some kind of "get_attribute" method (or similar) that returns a subset of models
- Pass a default model for intialization simplicity, then update the model in place???
"""