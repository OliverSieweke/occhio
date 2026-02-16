from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from itertools import product

from torch import Tensor, vmap, meshgrid, cartesian_prod, prod, Generator, arange, logspace
from stack_data.utils import cached_property
from tqdm import tqdm

from .occhio.toy_model import ToyModel
from .occhio.autoencoder import *
from .occhio.distributions import *


@dataclass
class Axis:
    label: str
    values: list | Tensor

class ModelGrid:
    models: list[Any]

    def __init__(
        self,
        create_model: Callable[..., ToyModel],
        axes: list[Axis] = [],                  # TEMPORARY: will remove this in the future
        
        # currently not used, will be used in the future
        stratified_axes: list[Axis] = [],       
        vectorized_axes: list[Axis] = [],
        *args,
        **kwargs
    ):

        if not all(isinstance(axis.values, Tensor) for axis in vectorized_axes):
            labels = [axis.label for axis in vectorized_axes if not isinstance(axis.values, Tensor)]
            raise TypeError(
                f"Axes {labels} must have values as torch.Tensor"
            )

        assert set(vectorized_axes).isdisjoint(set(stratified_axes)), "vectorized_axes and stratified_axes must be disjoint sets."

        if not vectorized_axes and not stratified_axes:
            raise ValueError("At least one of 'vectorized_axes' or 'stratified_axes' must be provided and non-empty.")

        self.distribution_generator = kwargs.get(
            "distribution_generator",
            Generator(device="cpu").manual_seed(42)
        )

        self.create_model = create_model
        self.initialize_models()
        self.build_vmap()


    def initialize_models(self):
        # This initializes a model for every combination of parameters in self.axes,
        # building a nested list structure matching the shape of the parameter grid,
        # and uses tqdm for progress tracking.

        if not self.axes:
            self.models = []
            return
        
        with tqdm(
            total = int(prod(Tensor([len(v) for v in axis_values])).item()),
            desc = "Initializing Models",
            unit = "model",
        ) as pbar:
            axis_labels = [axis.label for axis in self.axes]
            axis_values = [axis.values for axis in self.axes]
            
            def build_grid(level, params_so_far):
                axis = self.axes[level]
                res = []
                for value in axis.values:
                    params = params_so_far.copy()
                    params[axis.label] = value
                    if level == len(self.axes) - 1:
                        model = self.create_model(params)
                        res.append(model)
                        pbar.update(1)
                    else:
                        res.append(build_grid(level + 1, params))
                return res

            self.models = build_grid(0, {})

        
    def _to_flat(self):
        pass

    def _to_nested(self):
        pass

    def __getitem__(self, key: tuple[int, ...]) -> ToyModel:
        """Returns the ToyModel at the given indices. Allows for arbitrary indexing."""
        item = self.models
        for idx in key:
            item = item[idx]
        return item

    # TODO: Figure out how to add compatibility for non-vectorized axes
    @cached_property
    def parameters_mesh(self):
        """Returns a tuple of the meshgrid of the axes."""
        return meshgrid(*(axis.values for axis in self.axes))

    # TODO: Figure out how to add compatibility for non-vectorized axes
    # Warning: This may cause memory explosions for large grids.
    @cached_property
    def cartesian_parameters(self):
        """Returns a tuple of the cartesian product of the axes."""
        return cartesian_prod(*(axis.values for axis in self.axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the self.models."""
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def describe(self) -> dict[str, int]:
        """Returns a dictionary of the axis labels and their lengths."""
        return {axis.label: len(axis.values) for axis in self.axes}




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
    default_model: ToyModel | None = None, 
    params: Dict[str, Any] = {}, 
    *args, 
    **kwargs
) -> ToyModel:
    density = params["Density"]
    relative_importance = params["Importance"]
    # distribution = params["Distribution"]
    random_seed = params["Random Seeds"]

    model = ToyModel(
        distribution=SparseUniform(N_FEATURES, p_active=density, generator=distribution_generator),
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