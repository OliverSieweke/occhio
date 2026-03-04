# ABOUTME: Defines ModelGrid, a vectorized multi-dimensional grid of ToyModels.
# ABOUTME: Uses torch.vmap + torch.compile for fast parallel training across grid points.
from __future__ import annotations

import pickle
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from inspect import signature
from itertools import product
from typing import Any, Callable
from warnings import warn

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, meshgrid
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm import tqdm

from occhio.autoencoder import AutoEncoderBase
from occhio.distributions.base import Distribution
from occhio.toy_model import ToyModel


@dataclass
class Axis:
    label: str
    values: Tensor

    def __init__(self, label: str, values: Tensor | Sequence[float | int]):
        self.label = label
        # Convert to tensor and ensure float dtype for meshgrid compatibility
        if not isinstance(values, Tensor):
            self.values = torch.as_tensor(values, dtype=torch.float32)
        elif values.dtype not in (torch.float32, torch.float64):
            self.values = values.to(dtype=torch.float32)
        else:
            self.values = values


class TrainingAxis(Axis):
    """Special axis representing snapshots taken during training.

    This axis is used to store model states at different epochs, enabling
    visualization of training dynamics over time.
    """

    def __init__(self, values: Tensor | Sequence[int], label: str = "Epoch"):
        super().__init__(label=label, values=values)


class ModelGrid:
    """A multi-dimensional grid of ``ToyModel`` instances, parameterized over one or
    more named axes.

    Each point in the grid corresponds to a ``ToyModel`` created by a factory
    function that receives the axis values as a ``params`` dict.

    Args:
        create_model: A factory function that accepts a ``dict[str, Any]`` containing
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
        self,
        create_model: Callable[[dict[str, Any]], ToyModel],
        axes: list[Axis],
        broadcast_samples: bool = True,
        *,
        _models: NDArray[np.object_] | None = None,
    ):
        self.broadcast_samples: bool = broadcast_samples
        self._validate_args(create_model, axes)
        self.axes: list[Axis] = axes
        self.create_model: Callable[[dict[str, Any]], ToyModel] = create_model

        if _models is not None:
            self.models: NDArray[np.object_] = _models
        else:
            self.models = self._initialize_models()
            self._validate_vmap()

            if self.broadcast_samples:
                self._validate_generators()
                self._unique_distributions, self._sample_index = (
                    self._build_sample_broadcast()
                )

    def _validate_args(
        self, create_model: Callable[..., ToyModel], axes: list[Axis]
    ) -> None:
        if not axes:
            raise ValueError("At least one axis must be provided.")

        if "params" not in signature(create_model).parameters:
            raise TypeError(
                "create_model must accept a 'params' parameter (dict[str, Any])."
            )

    def _initialize_models(self) -> NDArray[np.object_]:
        shape: tuple[int, ...] = tuple(len(axis.values) for axis in self.axes)
        models: NDArray[np.object_] = np.empty(shape, dtype=object)
        for indices in product(*[range(s) for s in shape]):
            params: dict[str, Any] = {
                axis.label: axis.values[i] for axis, i in zip(self.axes, indices)
            }
            models[indices] = self.create_model(params)
        return models

    def _validate_vmap(self) -> None:
        if self.models.size <= 1:
            return

        flattened_models: NDArray[np.object_] = self.models.ravel()

        reference: AutoEncoderBase = flattened_models[0].ae
        reference_signature: tuple = (
            type(reference),
            {k: v.shape for k, v in reference.state_dict().items()},
            reference.device,
        )

        for i, model in enumerate(flattened_models, start=1):
            ae: AutoEncoderBase = model.ae
            ae_signature = (
                type(ae),
                {k: v.shape for k, v in ae.state_dict().items()},
                ae.device,
            )
            if ae_signature != reference_signature:
                # [17.02.26 | OliverSieweke] TODO: unstack the index here
                raise ValueError(
                    f"All Autoencoders should share the same architecture. "
                    f"Autoencoder at index {i} has incompatible architecture with the first Autoencoder. "
                    f"received: {ae_signature}, "
                    f"expected: {reference_signature}"
                )

    def _validate_generators(self) -> None:
        if self.models.size <= 1:
            return

        flattened_models: NDArray[np.object_] = self.models.ravel()

        for i, model in enumerate(flattened_models, start=1):
            if not model.distribution.generator:
                raise ValueError(
                    f"All distributions should have a fixed generator. "
                    f"Distribution at index {i} does not have a fixed generator."
                )

    def _build_sample_broadcast(self) -> tuple[list[Distribution], Tensor]:
        """Precompute which models share a distribution so the training loop
        only needs to sample once per unique distribution and broadcast the
        results — without dict lookups at training time."""
        flattened_models: NDArray[np.object_] = self.models.ravel()
        hash_to_idx: dict[str, int] = {}
        unique_distributions: list[Distribution] = []
        sample_index: list[int] = []

        for model in flattened_models:
            dist: Distribution = model.distribution
            dist_hash: str = dist._sampling_equivalence_hash
            if dist_hash not in hash_to_idx:
                hash_to_idx[dist_hash] = len(unique_distributions)
                unique_distributions.append(dist)
            sample_index.append(hash_to_idx[dist_hash])

        device: torch.device | str = flattened_models[0].ae.device
        return unique_distributions, torch.tensor(
            sample_index, dtype=torch.long, device=device
        )

    def _sync_generators(self) -> None:
        """Copy each lead distribution's generator state to its followers
        so that all equivalent distributions stay synchronized."""
        flattened_models: NDArray[np.object_] = self.models.ravel()
        lead_states: list[Tensor] = [
            dist.generator.get_state() for dist in self._unique_distributions
        ]

        for model_idx, unique_idx in enumerate(self._sample_index.tolist()):
            follower_dist: Distribution = flattened_models[model_idx].distribution
            if follower_dist is not self._unique_distributions[unique_idx]:
                follower_dist.generator.set_state(lead_states[unique_idx])

    def _can_vectorize_loss(self) -> bool:
        flattened_models = self.models.ravel()

        if len(flattened_models) < 2:
            return True

        return all(
            type(model.ae).loss is type(flattened_models[0].ae).loss
            for model in flattened_models[1:]
        )

    @cached_property
    def parameters_mesh(self):
        """Returns a tuple of the meshgrid of the axes."""
        return meshgrid(*(axis.values for axis in self.axes), indexing="ij")

    @property
    def _shape_from_axes(self) -> tuple[int, ...]:
        """Returns the shape of the axes that define the nested structure of the models."""
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.models.shape

    @property
    def description(self) -> dict[str, int]:
        """Returns a dictionary of the axis labels and their lengths."""
        return {axis.label: len(axis.values) for axis in self.axes}

    def save_models(self, path: str) -> None:
        """Serialize the model grid to disk using pickle.

        Warning: pickle files are tied to the current Python and library versions.
        Loading in a different environment may fail silently or raise errors.
        """
        if not isinstance(path, str) or not path:
            raise TypeError("Path must be a non-empty string.")
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load_models(self, path: str) -> None:
        """Load a serialized model grid from disk, replacing ``self.models`` in-place.

        Warning: pickle files are tied to the Python and library versions used when
        saving. Axis labels and values are not stored in the file — verify that the
        current grid's axes match the file's original grid after loading.
        """
        if not isinstance(path, str) or not path:
            raise TypeError("Path must be a non-empty string.")
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, "rb") as f:
            models = pickle.load(f)
        if not isinstance(models, np.ndarray):
            raise TypeError(f"File at {path} does not contain a numpy ndarray.")
        if models.dtype != object:
            raise TypeError(
                f"Expected an object array of ToyModels, got dtype={models.dtype}."
            )

        expected_shape: tuple[int, ...] = self._shape_from_axes
        if models.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: file has {models.shape}, "
                f"but axes define {expected_shape}."
            )
        if models.shape != self.models.shape:
            raise ValueError(
                f"Shape mismatch: file has {models.shape}, "
                f"but current grid has {self.models.shape}."
            )

        for m in models.ravel():
            if not isinstance(m, ToyModel):
                raise TypeError(
                    f"Expected all entries to be ToyModel, got {type(m).__name__}."
                )

        loaded_device = models.ravel()[0].ae.device
        grid_device = self.models.ravel()[0].ae.device
        if str(loaded_device) != str(grid_device):
            warn(
                f"Device mismatch: loaded models are on '{loaded_device}', "
                f"but the current grid is on '{grid_device}'. "
                f"Moving loaded models to '{grid_device}'.",
                stacklevel=2,
            )
            for m in models.ravel():
                m.ae.to(grid_device)
                m.distribution.to(grid_device)
                m.importances = m.importances.to(grid_device)

        axes_summary = ", ".join(
            f"'{a.label}' ({len(a.values)} values)" for a in self.axes
        )
        warn(
            f"Loading models from '{path}'. The current axes [{axes_summary}] "
            f"may not match the axes used to generate the saved models. "
            f"Verify that axes labels, values, and ordering are consistent "
            f"with the file's original grid.",
            stacklevel=2,
        )

        self.models = models
        self._validate_vmap()

        if self.broadcast_samples:
            self._validate_generators()
            self._unique_distributions, self._sample_index = (
                self._build_sample_broadcast()
            )

    # If you change the signature or implementation here, make sure you keep it
    # consistent with ToyModel.fit()
    def fit(
        self,
        n_epochs: int = 10_000,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        verbose: bool = False,
        compile: bool = False,
        track_losses: bool = False,
        snapshot_interval: int | None = None,
        sample_every: int = 25,
    ) -> ModelGrid | list[float] | None:
        # Validate sample_every
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        # Validate snapshot_interval
        if snapshot_interval is not None:
            if snapshot_interval <= 0:
                raise ValueError(
                    f"snapshot_interval must be positive, got {snapshot_interval}"
                )
            if snapshot_interval > n_epochs:
                raise ValueError(
                    f"snapshot_interval ({snapshot_interval}) cannot exceed n_epochs ({n_epochs})"
                )

            # Memory warning
            n_snapshots = (n_epochs // snapshot_interval) + 1  # +1 for initial state
            total_snapshots = self.models.size * n_snapshots
            if total_snapshots > 10000:
                import warnings

                warnings.warn(
                    f"Large memory allocation: {self.models.size} models × {n_snapshots} snapshots "
                    f"= {total_snapshots} total model copies. This may consume significant memory.",
                    ResourceWarning,
                    stacklevel=2,
                )

        flattened_models: NDArray[np.object_] = self.models.ravel()

        # Stack Model Characteristics --------------------------------------------------
        stacked_params, stacked_buffers = stack_module_state(
            [model.ae for model in flattened_models]
        )
        # NB: We enable gradients on params as stack_module_state returns detached
        # tensors
        stacked_params = {
            key: value.requires_grad_(True) for key, value in stacked_params.items()
        }
        stacked_importances: Tensor = torch.stack(
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
        representative_ae: AutoEncoderBase = flattened_models[0].ae
        stacked_forward = torch.compile(
            torch.vmap(
                lambda params, buffers, x: functional_call(
                    representative_ae, (params, buffers), (x,)
                )[0],
                in_dims=(0, 0, 0),
            ),
            disable=not compile,
        )

        use_vectorized_loss: bool = self._can_vectorize_loss()
        if use_vectorized_loss:
            stacked_loss = torch.compile(
                torch.vmap(
                    lambda x_true, x_hat, importances: representative_ae.loss(
                        x_true, x_hat, importances
                    ),
                    in_dims=(0, 0, 0),
                ),
                disable=not compile,
            )

        # Training ---------------------------------------------------------------------
        losses: list[float] | None = [] if track_losses else None

        # Snapshot storage
        snapshots: list[tuple[int, dict, dict]] | None = (
            None  # [(epoch, params_copy, buffers_copy), ...]
        )
        if snapshot_interval is not None:
            snapshots = []
            # Capture initial state (epoch 0)
            snapshots.append(
                (
                    0,
                    {k: v.detach().clone() for k, v in stacked_params.items()},
                    {k: v.detach().clone() for k, v in stacked_buffers.items()},
                )
            )

        # Pre-allocated sample buffer: sample once every `sample_every` epochs
        # with sample_every × batch_size samples, then slice per epoch.
        sample_buffer: Tensor | None = None

        for ep in tqdm(range(n_epochs), unit="epoch"):
            buf_offset = ep % sample_every
            if buf_offset == 0:
                # Determine how many epochs remain to avoid over-sampling
                epochs_left = min(sample_every, n_epochs - ep)
                total_samples = epochs_left * batch_size

                if self.broadcast_samples:
                    unique_samples = torch.stack(
                        [
                            dist.sample(total_samples)
                            for dist in self._unique_distributions
                        ]
                    )
                    sample_buffer = unique_samples[self._sample_index]
                else:
                    sample_buffer = torch.stack(
                        [
                            model.distribution.sample(total_samples)
                            for model in flattened_models
                        ]
                    )

            start = buf_offset * batch_size
            end = start + batch_size
            stacked_samples = sample_buffer[:, start:end, :]

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

            total_loss: Tensor = stacked_losses.mean()
            total_loss.backward()
            optimizer.step()

            if track_losses:
                losses.append(total_loss.item())
            if verbose and (ep + 1) % 1000 == 0:
                print(f"Epoch {ep + 1}/{n_epochs}, Mean Loss: {total_loss.item():.6f}")

            # Capture snapshot if needed
            if snapshot_interval is not None and (ep + 1) % snapshot_interval == 0:
                snapshots.append(
                    (
                        ep + 1,
                        {k: v.detach().clone() for k, v in stacked_params.items()},
                        {k: v.detach().clone() for k, v in stacked_buffers.items()},
                    )
                )

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

        if self.broadcast_samples:
            self._sync_generators()
            for model in flattened_models:
                model.distribution.__dict__.pop("_sampling_equivalence_hash", None)
            self._unique_distributions, self._sample_index = (
                self._build_sample_broadcast()
            )

        # Build history grid if snapshots were captured
        if snapshots is not None:
            return self._build_history_grid(snapshots, flattened_models)

        return losses

    def _build_history_grid(
        self,
        snapshots: list[tuple[int, dict, dict]],
        flattened_models: NDArray[np.object_],
    ) -> ModelGrid:
        """Build a new ModelGrid with TrainingAxis from captured snapshots.

        Args:
            snapshots: List of (epoch, stacked_params, stacked_buffers) tuples
            flattened_models: Flattened array of original models (for reference)

        Returns:
            New ModelGrid with TrainingAxis prepended to axes
        """
        n_snapshots = len(snapshots)

        # Create new shape: (n_snapshots, *original_shape)
        history_shape = (n_snapshots,) + self.models.shape
        history_models = np.empty(history_shape, dtype=object)

        # Populate the history grid
        for snapshot_idx, (
            epoch,
            stacked_params_snapshot,
            stacked_buffers_snapshot,
        ) in enumerate(
            tqdm(snapshots, desc="Building history grid", unit="epoch", leave=True)
        ):
            for model_idx, original_model in enumerate(flattened_models):
                # Create a new ToyModel with the same distribution and architecture
                # but with snapshotted autoencoder weights
                snapshot_model = ToyModel(
                    distribution=original_model.distribution,
                    ae=deepcopy(original_model.ae),
                    importances=original_model.importances,
                )

                # Load the snapshotted state
                snapshot_model.ae.load_state_dict(
                    {
                        name: (
                            stacked_params_snapshot[name]
                            if name in stacked_params_snapshot
                            else stacked_buffers_snapshot[name]
                        )[model_idx]
                        for name in snapshot_model.ae.state_dict()
                    }
                )

                # Place in history grid (unravel model_idx to multi-dimensional index)
                multi_idx = np.unravel_index(model_idx, self.models.shape)
                history_models[(snapshot_idx,) + multi_idx] = snapshot_model

        # Create new axes with TrainingAxis prepended
        epoch_values = [snapshot[0] for snapshot in snapshots]
        new_axes = [TrainingAxis(values=epoch_values)] + self.axes

        # Create and return the history grid
        # Note: broadcast_samples=False because history grids are read-only snapshots
        # and won't be trained, so we don't need sample caching infrastructure
        return ModelGrid(
            create_model=self.create_model,
            axes=new_axes,
            broadcast_samples=False,
            _models=history_models,
        )

    def __getitem__(self, key) -> ModelGrid | ToyModel:
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) > len(self.axes):
            raise IndexError(
                f"Too many indices: got {len(key)}, grid has {len(self.axes)} axes"
            )

        numpy_key: list[slice] = []
        new_axes: list[Axis] = []

        for dim, k in enumerate(key):
            axis: Axis = self.axes[dim]
            dim_size: int = len(axis.values)

            if isinstance(k, int):
                idx: int = k + dim_size if k < 0 else k
                if idx < 0 or idx >= dim_size:
                    raise IndexError(
                        f"Index {k} out of bounds for axis '{axis.label}' "
                        f"with size {dim_size}"
                    )
                numpy_key.append(slice(idx, idx + 1))
                values = axis.values[idx : idx + 1]

            elif isinstance(k, slice):
                start, stop, step = k.start, k.stop, k.step
                if start is not None and start < 0:
                    start += dim_size
                if stop is not None and stop < 0:
                    stop += dim_size

                if start is not None and (start < 0 or start >= dim_size):
                    raise IndexError(
                        f"Slice start {k.start} out of bounds for axis "
                        f"'{axis.label}' with size {dim_size}"
                    )
                if stop is not None and (stop < 0 or stop > dim_size):
                    raise IndexError(
                        f"Slice stop {k.stop} out of bounds for axis "
                        f"'{axis.label}' with size {dim_size}"
                    )

                if (
                    step is None
                    and start is not None
                    and stop is not None
                    and start > stop
                ):
                    step = -1

                s = slice(start, stop, step)
                numpy_key.append(s)
                if step is not None and step < 0:
                    range_start = start if start is not None else dim_size - 1
                    range_stop = stop if stop is not None else -1
                    indices = list(range(range_start, range_stop, step))
                    values = axis.values[indices]
                else:
                    values = axis.values[s]
            else:
                raise IndexError(f"Unsupported index type: {type(k)}")

            # Preserve axis type (e.g., TrainingAxis)
            if isinstance(axis, TrainingAxis):
                new_axes.append(TrainingAxis(label=axis.label, values=values))
            else:
                new_axes.append(Axis(label=axis.label, values=values))

        for dim in range(len(key), len(self.axes)):
            new_axes.append(self.axes[dim])
            numpy_key.append(slice(None))

        all_int: bool = len(key) == len(self.axes) and all(
            isinstance(k, int) for k in key
        )
        if all_int:
            resolved_idx = tuple(s.start for s in numpy_key)
            return self.models[resolved_idx]

        sliced_models: NDArray[np.object_] = self.models[tuple(numpy_key)]

        return ModelGrid(
            create_model=self.create_model,
            axes=new_axes,
            broadcast_samples=self.broadcast_samples,
            _models=sliced_models,
        )
