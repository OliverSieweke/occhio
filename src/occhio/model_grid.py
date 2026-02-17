from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Any, Callable, Dict, List

import torch
from torch import Generator, Tensor, arange, cartesian_prod, logspace, meshgrid
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm import tqdm

from occhio.autoencoder import *
from occhio.distributions import *
from occhio.toy_model import ToyModel


@dataclass
class Axis:
    label: str
    values: list | Tensor


class ModelGrid:
    models: List[ToyModel]

    def __init__(
        self,
        create_model: Callable[..., ToyModel],
        axes: List[Axis],
    ):
        if not axes:
            raise ValueError("At least one axis must be provided.")
        self.axes = axes

        self.models = []
        for values in product(*[axis.values for axis in self.axes]):
            params = {axis.label: value for axis, value in zip(self.axes, values)}
            self.models.append(create_model(params))

        self._validate_autoencoders()

    def _validate_autoencoders(self):
        if len(self.models) < 2:
            return

        reference = self.models[0].ae
        reference_signature = (
            type(reference),
            {k: v.shape for k, v in reference.state_dict().items()},
            reference.device,
        )

        for i, model in enumerate(self.models[1:], start=1):
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
        if len(self.models) < 2:
            return True

        return all(
            type(model.ae).loss is type(self.models[0].ae).loss
            for model in self.models[1:]
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
        # Stack Model Characteristics --------------------------------------------------
        stacked_params, stacked_buffers = stack_module_state(
            [model.ae for model in self.models]
        )
        # NB: We enable gradients on params as stack_module_state returns detached
        # tensors
        stacked_params = {
            key: value.requires_grad_(True) for key, value in stacked_params.items()
        }
        stacked_importances = torch.stack([model.importances for model in self.models])

        # Optimizer --------------------------------------------------------------------
        optimizer = AdamW(
            list(stacked_params.values()), lr=learning_rate, weight_decay=weight_decay
        )

        # Define Stacked Forward Pass and Loss -----------------------------------------
        # The forward pass operation is based on the first Auto-Encoder, which stands as
        # a representative for all the Auto-Encoders. This relies on the models using
        # the same Auto-Encoder kind, which is enforced in the initialization.
        representative_ae = self.models[0].ae
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
                [model.distribution.sample(batch_size) for model in self.models]
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
                            self.models,
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
            for i, model in enumerate(self.models):
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

    def __getitem__(self, key: tuple[int, ...]) -> ToyModel:
        """Returns the ToyModel at the given indices. Allows for arbitrary indexing."""
        item = self.models
        for i in key:
            item = item[i]
        return item

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
