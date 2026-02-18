from dataclasses import dataclass
from functools import cached_property
from inspect import signature
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, meshgrid
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm import tqdm

from occhio.toy_model import ToyModel


@dataclass
class Axis:
    label: str
    values: list | Tensor


class ModelGrid:
    """A multi-dimensional grid of ``ToyModel`` instances, parameterized over one or
    more named axes.

    Each point in the grid corresponds to a ``ToyModel`` created by a factory
    function that receives the axis values as a ``params`` dict.

    Args:
        create_model: A factory function that accepts a ``Dict[str, Any]`` containing
         axes values at a given grid point, and returns an initialised ``ToyModel``.
        axes: An ordered list of ``Axis`` objects defining the grid dimensions.
            At least one axis must be provided.

    Example::
        def create_model(params):
            return ToyModel(
                distribution=SparseUniform(5, p_active=params["Density"]),
                ae=TiedLinearRelu(5, 2),
                importances=params["Relative Importance" ** torch.arange(5),
            )


        model_grid = ModelGrid(
            create_model,
            axes=[
                Axis(label="Density", values=logspace(0, -2, 32)),
                Axis(label="Relative Importance", values=logspace(-1, 1, 32)),
            ],
        )
    """

    models: NDArray[np.object_]

    def __init__(
        self, create_model: Callable[[Dict[str, Any]], ToyModel], axes: List[Axis]
    ):
        self._validate_args(
            create_model=create_model,
            axes=axes,
        )
        self.axes = axes

        self.models = np.empty(self.shape, dtype=object)
        for indices in np.ndindex(*self.shape):
            self.models[indices] = create_model(
                {
                    axis.label: axis.values[axis_index]
                    for axis, axis_index in zip(self.axes, indices)
                }
            )

        self._validate_autoencoders()

    def _validate_autoencoders(self):
        if len(self.models) < 2:
            return

        flattened_models = self.models.ravel()

        reference = flattened_models[0].ae
        reference_signature = (
            type(reference),
            {k: v.shape for k, v in reference.state_dict().items()},
            reference.device,
        )

        for i, model in enumerate(flattened_models[1:], start=1):
            ae = model.ae
            signature = (
                type(ae),
                {k: v.shape for k, v in ae.state_dict().items()},
                ae.device,
            )
            if signature != reference_signature:
                # [17.02.26 | OliverSieweke] TODO: unstack the index here
                raise ValueError(
                    f"All Autoencoders should share the same architecture"
                    f"Autoencoder at index {i} has incompatible architecture with the first Autoencoder."
                    f"received: {signature}, "
                    f"expected: {reference_signature}"
                )

    def _can_vectorize_loss(self) -> bool:
        flattened_models = self.models.ravel()

        if len(flattened_models) < 2:
            return True

        return all(
            type(model.ae).loss is type(flattened_models[0].ae).loss
            for model in flattened_models[1:]
        )

    # If you change the signature or implementation here, make sure you keep it
    # consistent with ToyModel.fit()
    def fit(
        self,
        n_epochs: int = 10000,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        verbose: bool = False,
        track_losses: bool = False,
    ):
        flattened_models = self.models.ravel()

        # Stack Model Characteristics --------------------------------------------------
        stacked_params, stacked_buffers = stack_module_state(
            [model.ae for model in flattened_models]
        )
        # NB: We enable gradients on params as stack_module_state returns detached
        # tensors
        stacked_params = {
            key: value.requires_grad_(True) for key, value in stacked_params.items()
        }
        stacked_importances = torch.stack(
            [model.importances for model in flattened_models]
        )

        # Optimizer --------------------------------------------------------------------
        optimizer = AdamW(
            list(stacked_params.values()), lr=learning_rate, weight_decay=weight_decay
        )

        # Define Stacked Forward Pass and Loss -----------------------------------------
        # The forward pass operation is based on the first Auto-Encoder, which stands as
        # a representative for all the Auto-Encoders. This relies on the models using
        # the same Auto-Encoder kind, which is enforced in the initialization.
        representative_ae = flattened_models[0].ae
        stacked_forward = torch.vmap(
            lambda params, buffers, x: functional_call(
                representative_ae, (params, buffers), (x,)
            )[0],
            in_dims=(0, 0, 0),
        )

        use_vectorized_loss = self._can_vectorize_loss()
        if use_vectorized_loss:
            stacked_loss = torch.vmap(
                lambda x_true, x_hat, importances: representative_ae.loss(
                    x_true, x_hat, importances
                ),
                in_dims=(0, 0, 0),
            )

        # Training ---------------------------------------------------------------------
        losses = [] if track_losses else None

        for ep in tqdm(range(n_epochs), desc="Training", unit="epoch"):
            # [17.02.26 | OliverSieweke] TODO: Could attempt to vectorize when possible
            # here. This is not trivial though, one would need to:
            #   - group the distributions of the same kind
            #   - think through which distributions are actually stackable
            #       (make this a property on the distribution?)
            #   - find a good way to expose/use this stackability
            stacked_samples = torch.stack(
                [model.distribution.sample(batch_size) for model in flattened_models]
            )

            optimizer.zero_grad()
            stacked_x_hat = stacked_forward(
                stacked_params, stacked_buffers, stacked_samples
            )
            if use_vectorized_loss:
                stacked_losses = stacked_loss(
                    stacked_samples, stacked_x_hat, stacked_importances
                )
            else:  # Fallback for heterogeneous losses
                stacked_losses = torch.stack(
                    [
                        model.ae.loss(samples, x_hat, importances)
                        for model, samples, x_hat, importances in zip(
                            flattened_models,
                            stacked_samples,
                            stacked_x_hat,
                            stacked_importances,
                        )
                    ]
                )

            total_loss = stacked_losses.mean()
            total_loss.backward()
            optimizer.step()

            if track_losses:
                losses.append(total_loss.item())
            if verbose and (ep + 1) % 1000 == 0:
                print(f"Epoch {ep + 1}/{n_epochs}, Mean Loss: {total_loss.item():.6f}")

        with torch.no_grad():
            for i, model in enumerate(flattened_models):
                model.ae.load_state_dict(
                    {
                        name: (
                            stacked_params[name]
                            if name in stacked_params
                            else stacked_buffers[name]
                        )[i]
                        for name in model.ae.state_dict()
                    }
                )

        return losses

    def __getitem__(self, key: tuple[int, ...]) -> ToyModel | NDArray[np.object_]:
        """Returns the ToyModel at the given indices. Allows for arbitrary indexing."""
        return self.models[key]

    @cached_property
    def parameters_mesh(self):
        """Returns a tuple of the meshgrid of the axes."""
        return meshgrid(*(axis.values for axis in self.axes), indexing="ij")

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the self.models."""
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def describe(self) -> dict[str, int]:
        """Returns a dictionary of the axis labels and their lengths."""
        return {axis.label: len(axis.values) for axis in self.axes}

    @staticmethod
    def _validate_args(create_model, axes) -> None:
        if not axes:
            raise ValueError("At least one axis must be provided.")

        if "params" not in signature(create_model).parameters:
            raise TypeError(
                "create_model must accept a 'params' parameter (Dict[str, Any])."
            )
