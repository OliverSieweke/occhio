from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Any, Callable, Dict, List

import torch
from torch import Generator, Tensor, arange, cartesian_prod, logspace, meshgrid
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW

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

    def fit(
        self,
        batch_size: int = 1024,
        n_epochs: int = 10000,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        verbose: bool = False,
    ):
        """
        Train all models in parallel using batched operations.
        
        Instead of training each model sequentially, this method:
        1. Stacks all model parameters into batched tensors
        2. Samples from each distribution (still sequential, but fast)
        3. Performs batched forward pass across all models
        4. Computes losses in parallel with per-model importances
        5. Single backward pass updates all models simultaneously
        """
        if not self.models:
            return

        n_models = len(self.models)
        
        # Get device from first model's autoencoder
        device = next(self.models[0].ae.parameters()).device
        
        # Use the first model's AE as the "base" for functional_call
        base_ae = self.models[0].ae
        
        # Stack all model states: params will have shape {name: (n_models, *param_shape)}
        params, buffers = stack_module_state([m.ae for m in self.models])
        
        # Make stacked params require grad and move to correct device
        params = {k: v.to(device).requires_grad_(True) for k, v in params.items()}
        buffers = {k: v.to(device) for k, v in buffers.items()}
        
        # Stack importances: (n_models, n_features)
        importances = torch.stack([m.importances for m in self.models]).to(device)
        
        # Create optimizer over the stacked parameters
        optimizer = AdamW(
            list(params.values()), lr=learning_rate, weight_decay=weight_decay
        )

        # Define the forward function for a single model
        def single_forward(params, buffers, x):
            # functional_call runs the module with the given params/buffers
            x_hat, _ = functional_call(base_ae, (params, buffers), (x,))
            return x_hat

        # Vectorize over the model dimension (dim 0 of params/buffers/x)
        batched_forward = torch.vmap(single_forward, in_dims=(0, 0, 0))

        for ep in range(n_epochs):
            # Sample from each distribution: (n_models, batch_size, n_features)
            # This is still sequential but typically fast compared to forward/backward
            all_samples = torch.stack([
                m.distribution.sample(batch_size).to(device) 
                for m in self.models
            ])
            
            optimizer.zero_grad()
            
            # Batched forward pass: (n_models, batch_size, n_features)
            all_x_hat = batched_forward(params, buffers, all_samples)
            
            # Compute per-model losses in parallel
            # all_samples, all_x_hat: (n_models, batch_size, n_features)
            # importances: (n_models, n_features)
            squared_error = (all_samples - all_x_hat) ** 2  # (n_models, batch_size, n_features)
            weighted_error = squared_error * importances.unsqueeze(1)  # broadcast importances
            per_model_loss = weighted_error.sum(dim=-1).mean(dim=-1)  # (n_models,)
            
            # Total loss is mean across all models
            total_loss = per_model_loss.mean()
            
            total_loss.backward()
            optimizer.step()

            if verbose and (ep + 1) % 1000 == 0:
                print(f"Epoch {ep + 1}/{n_epochs}, Mean Loss: {total_loss.item():.6f}")

        # Write trained parameters back to original model objects
        self._unstack_params_to_models(params, buffers)

    def _unstack_params_to_models(self, params: dict, buffers: dict):
        """Copy the trained stacked parameters back into individual model AEs."""
        with torch.no_grad():
            for i, model in enumerate(self.models):
                state_dict = model.ae.state_dict()
                for name in state_dict:
                    if name in params:
                        state_dict[name].copy_(params[name][i])
                    elif name in buffers:
                        state_dict[name].copy_(buffers[name][i])
                
                # Clear any cached properties that depend on trained weights
                # These are computed lazily and would be stale after training
                for attr in list(model.__dict__.keys()):
                    if isinstance(getattr(type(model), attr, None), cached_property):
                        delattr(model, attr)

    def __getitem__(self, key: tuple[int, ...]) -> ToyModel:
        """Returns the ToyModel at the given indices. Allows for arbitrary indexing."""
        item = self.models
        for i in key:
            item = item[i]
        return item

    @cached_property
    def parameters_mesh(self):
        """Returns a tuple of the meshgrid of the axes."""
        return meshgrid(*(axis.values for axis in self.axes), indexing='ij')

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
