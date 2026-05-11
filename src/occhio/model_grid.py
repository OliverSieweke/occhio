"""Defines ModelGrid, a vectorized multi-dimensional grid of ToyModels.

Uses torch.vmap + torch.compile for fast parallel training across grid points.
"""

from __future__ import annotations

import datetime
import pickle
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from inspect import signature
from itertools import product
from pathlib import Path
from typing import Any, Callable
from warnings import warn

import dill
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, meshgrid
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm.auto import tqdm

from occhio.autoencoders import AutoEncoderBase
from occhio.distributions.base import Distribution
from occhio.toy_model import SAEEntry, ToyModel


@dataclass
class Axis:
    """A named axis for a :class:`ModelGrid` parameter sweep.

    Args:
        label: Human-readable name (used in plot headers and DataFrame indices).
        values: Sequence of values to sweep over.
    """

    label: str
    values: Sequence

    def __init__(self, label: str, values: Iterable):
        self.label = label
        # Convert to list if not already indexable (handles Enums, generators, etc.)
        if not isinstance(values, Sequence):
            values = list(values)
        # Convert to tensor and ensure float dtype for meshgrid compatibility
        # [2026-03-25 | OliverSieweke] TODO: work on this, not sure we want to convert to tensor at this point....
        # if not isinstance(values, Tensor):
        #     self.values = torch.as_tensor(values, dtype=torch.float32)
        # elif values.dtype not in (torch.float32, torch.float64):
        #     self.values = values.to(dtype=torch.float32)
        # else:
        #     self.values = values
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

    @staticmethod
    def from_iterable(models: Iterable[ToyModel]) -> ModelGrid:
        """Create a ModelGrid from a (possibly nested) iterable of ToyModels.

        Args:
            models: An iterable of ToyModels, or nested iterables forming a
                multi-dimensional structure (e.g., list of lists of ToyModels).

        Returns:
            A ModelGrid with axes named "Axis 1", "Axis 2", etc., with integer indices.

        Raises:
            ValueError: If the iterable is empty or has inconsistent dimensions.
            TypeError: If leaf elements are not ToyModels.

        Example::
            # 1D grid from a list
            grid = ModelGrid.from_iterable([model1, model2, model3])

            # 2D grid from nested lists
            grid = ModelGrid.from_iterable(
                [
                    [model_a1, model_a2],
                    [model_b1, model_b2],
                ]
            )
        """

        def _to_nested_list(obj: Any) -> list | ToyModel:
            """Recursively convert iterables to nested lists, stopping at ToyModels."""
            if isinstance(obj, ToyModel):
                return obj
            if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                return [_to_nested_list(item) for item in obj]
            raise TypeError(f"Expected ToyModel or iterable, got {type(obj).__name__}")

        def _get_shape_and_validate(nested: list, depth: int = 0) -> tuple[int, ...]:
            """Recursively determine shape and validate consistency."""
            if not nested:
                raise ValueError(
                    f"Empty iterable at depth {depth}. All dimensions must be non-empty."
                )

            # Check if we're at leaf level (ToyModels)
            if isinstance(nested[0], ToyModel):
                for i, item in enumerate(nested):
                    if not isinstance(item, ToyModel):
                        raise TypeError(
                            f"Inconsistent structure: expected ToyModel at index {i}, "
                            f"got {type(item).__name__}"
                        )
                return (len(nested),)

            # We have nested lists - recurse
            if not isinstance(nested[0], list):
                raise TypeError(
                    f"Expected list or ToyModel at depth {depth}, "
                    f"got {type(nested[0]).__name__}"
                )

            child_shapes = []
            for i, child in enumerate(nested):
                if not isinstance(child, list):
                    raise TypeError(
                        f"Inconsistent structure at depth {depth}, index {i}: "
                        f"expected list, got {type(child).__name__}"
                    )
                child_shapes.append(_get_shape_and_validate(child, depth + 1))

            # Validate all children have the same shape
            first_shape = child_shapes[0]
            for i, shape in enumerate(child_shapes[1:], start=1):
                if shape != first_shape:
                    raise ValueError(
                        f"Inconsistent dimensions at depth {depth}: "
                        f"index 0 has shape {first_shape}, "
                        f"index {i} has shape {shape}"
                    )

            return (len(nested),) + first_shape

        def _flatten_to_array(
            nested: list, shape: tuple[int, ...]
        ) -> NDArray[np.object_]:
            """Flatten nested list structure into a numpy array with given shape."""
            models_array: NDArray[np.object_] = np.empty(shape, dtype=object)

            def _fill(current: list, indices: tuple[int, ...]) -> None:
                if len(indices) == len(shape) - 1:
                    for i, item in enumerate(current):
                        models_array[indices + (i,)] = item
                else:
                    for i, child in enumerate(current):
                        _fill(child, indices + (i,))

            _fill(nested, ())
            return models_array

        # Convert to nested lists
        nested_list = _to_nested_list(models)

        if isinstance(nested_list, ToyModel):
            raise ValueError(
                "Cannot convert a single ToyModel to ModelGrid. "
                "Wrap it in a list: ModelGrid.from_iterable([model])"
            )

        # Get shape and validate
        shape = _get_shape_and_validate(nested_list)

        # Create axes with integer indices
        axes = [
            Axis(label=f"Axis {i + 1}", values=list(range(dim_size)))
            for i, dim_size in enumerate(shape)
        ]

        # Flatten to numpy array
        models_array = _flatten_to_array(nested_list, shape)

        return ModelGrid(
            create_model=lambda params: None,
            axes=axes,
            broadcast_samples=False,
            _models=models_array,
        )

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

    def _initialize_models(self) -> NDArray[np.object_]:
        shape: tuple[int, ...] = tuple(len(axis.values) for axis in self.axes)
        models: NDArray[np.object_] = np.empty(shape, dtype=object)
        for indices in product(*[range(s) for s in shape]):
            params: dict[str, Any] = {
                axis.label: axis.values[i] for axis, i in zip(self.axes, indices)
            }
            models[indices] = self.create_model(params)
        return models

    def _validate_args(
        self, create_model: Callable[..., ToyModel], axes: list[Axis]
    ) -> None:
        if not axes:
            raise ValueError("At least one axis must be provided.")

        if "params" not in signature(create_model).parameters:
            raise TypeError(
                "create_model must accept a 'params' parameter (dict[str, Any])."
            )

    def _validate_vmap(self) -> None:
        if self.models.size <= 1:
            return

        first_index = next(np.ndindex(self.models.shape))
        reference: AutoEncoderBase = self.models[first_index].ae
        reference_signature: tuple = (
            type(reference),
            {k: v.shape for k, v in reference.state_dict().items()},
            reference.device,
        )

        for index in np.ndindex(self.models.shape):
            ae: AutoEncoderBase = self.models[index].ae
            ae_signature = (
                type(ae),
                {k: v.shape for k, v in ae.state_dict().items()},
                ae.device,
            )

            if ae_signature != reference_signature:
                raise ValueError(
                    f"\nAll Autoencoders should share the same architecture. "
                    f"Autoencoder at index {index} has incompatible architecture with the first Autoencoder. "
                    f"received: {ae_signature}, "
                    f"expected: {reference_signature}"
                )

    def _validate_generators(self) -> None:
        if self.models.size <= 1:
            return

        for index in np.ndindex(self.models.shape):
            distribution = self.models[index].distribution
            if self.broadcast_samples and not distribution._defines_generators:
                warn(
                    f"\nSample broadcasting requires every ToyModel.distribution to have defined generators for sample reproducibility. "
                    f"Distribution at position {index} does not have defined generators and will not participate in sample broadcasting. "
                    f"This may lead to unnecessary re-sampling or loss of determinism for this model.",
                    stacklevel=2,
                )

    def _build_broadcast(self) -> tuple[list[Distribution], Tensor]:
        """
        Groups models by which unique distribution instance they use, so that sample
        broadcasting (i.e., generating samples only once per set of equivalent distributions)
        is efficient during training.

        Returns:
            (broadcasters, broadcast_map):
                broadcasters: a list of unique Distribution objects used by the models.
                broadcast_map: a tensor that, for each model (flattened), gives the index
                                  into broadcasters for its distribution.

        Generator-less distributions are never grouped together because their
        sampling state cannot be synchronized — each gets its own broadcaster slot.
        """
        flattened_models: NDArray[np.object_] = self.models.ravel()
        # Maps each unique distribution hash to its assigned broadcaster index
        hash_to_idx: dict[str, int] = {}
        broadcasters: list[Distribution] = []
        broadcast_map: list[int] = []

        for model in flattened_models:
            distribution: Distribution = model.distribution
            # Generator-less distributions can't be synced, so each gets its
            # own broadcaster slot regardless of hash equivalence.
            if distribution._defines_generators:
                distribution_hash: str = distribution._equivalence_hash
                if distribution_hash not in hash_to_idx:
                    hash_to_idx[distribution_hash] = len(broadcasters)
                    broadcasters.append(distribution)
                broadcast_map.append(hash_to_idx[distribution_hash])
            else:
                broadcast_map.append(len(broadcasters))
                broadcasters.append(distribution)

        device: torch.device | str = flattened_models[0].ae.device
        return broadcasters, torch.tensor(
            broadcast_map, dtype=torch.long, device=device
        )

    def _sync_generators(
        self, broadcasters: list[Distribution], broadcast_map: Tensor
    ) -> None:
        """Copy each broadcaster distribution's generator state to all
        equivalent distributions so they stay synchronized."""
        flattened_models: NDArray[np.object_] = self.models.ravel()

        # Pre-collect generators from each broadcaster
        broadcaster_gens: list[list[torch.Generator | None]] = [
            dist.collect_generators() for dist in broadcasters
        ]

        for model_idx, broadcaster_idx in enumerate(broadcast_map.tolist()):
            dist: Distribution = flattened_models[model_idx].distribution
            gens = broadcaster_gens[broadcaster_idx]
            if dist is not broadcasters[broadcaster_idx]:
                dist.sync_generators(gens)

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
        tensors = [
            torch.as_tensor(axis.values, dtype=torch.float32) for axis in self.axes
        ]
        return meshgrid(*tensors, indexing="ij")

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

    def save_models(self, path: str | None = None) -> None:
        """Serialize the model grid to disk using pickle.

        Warning: pickle files are tied to the current Python and occhio versions.
        Loading in a different environment may fail silently or raise errors.
        """
        if not path:
            path = (
                "model_grid"
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".pkl"
            )
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load_models(self, path: str) -> None:
        """Load a serialized model grid from disk, replacing ``self.models`` in-place.

        Warning: pickle files are tied to the Python and occhio versions used when
        saving. Other attributes of ModelGrid (such as Axis labels and values) are not stored in the file.
        Ensure that the current ModelGrid's axes match the file's original ModelGrid after loading.
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
                f"\nDevice mismatch: loaded models are on '{loaded_device}', "
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
            f"\nLoading models from '{path}'. The current axes [{axes_summary}] "
            f"may not match the axes used to generate the saved models. "
            f"Verify that axes labels, values, and ordering are consistent "
            f"with the file's original grid.",
            stacklevel=2,
        )

        self.models = models
        self._validate_vmap()

        if self.broadcast_samples:
            self._validate_generators()

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
        """Train all models in the grid in parallel using ``torch.vmap``.

        Args:
            n_epochs: Number of training epochs.
            batch_size: Samples per model per epoch.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            verbose: Show a tqdm progress bar.
            compile: Apply ``torch.compile`` to the vectorized forward pass.
            track_losses: Return per-epoch mean losses.
            snapshot_interval: If set, capture model state every N epochs and
                return a new ``ModelGrid`` with a prepended ``TrainingAxis``.
            sample_every: Re-sample from distributions every N epochs.

        Returns:
            - If ``snapshot_interval`` is set: a new ``ModelGrid`` with
              ``TrainingAxis`` prepended.
            - If ``track_losses`` is True: list of per-epoch mean losses.
            - Otherwise: ``None``.
        """
        # Validate sample_every
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        # Validate snapshot_interval
        if snapshot_interval is not None:
            if snapshot_interval <= 0:
                raise ValueError(
                    f"\nsnapshot_interval must be positive, got {snapshot_interval}"
                )
            if snapshot_interval > n_epochs:
                raise ValueError(
                    f"\nsnapshot_interval ({snapshot_interval}) cannot exceed n_epochs ({n_epochs})"
                )

            # Memory warning
            n_snapshots = (n_epochs // snapshot_interval) + 1  # +1 for initial state
            total_snapshots = self.models.size * n_snapshots
            if total_snapshots > 10000:
                import warnings

                warnings.warn(
                    f"\nLarge memory allocation: {self.models.size} models × {n_snapshots} snapshots "
                    f"= {total_snapshots} total model copies. This may consume significant memory.",
                    ResourceWarning,
                    stacklevel=2,
                )

        if self.broadcast_samples:
            broadcasters, broadcast_map = self._build_broadcast()

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
        # Pre-allocate a device-side buffer so loss tracking never forces a
        # per-step GPU→CPU sync. Converted to a Python list in one transfer
        # at the end. (CUDA perf: .item() per epoch is a sync point.)
        ae_device = flattened_models[0].ae.device
        loss_buffer = torch.empty(n_epochs, device=ae_device) if track_losses else None
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

        for ep in tqdm(range(n_epochs), unit="epoch", disable=not verbose):
            buf_offset = ep % sample_every
            if buf_offset == 0:
                # Determine how many epochs remain to avoid over-sampling
                epochs_left = min(sample_every, n_epochs - ep)
                total_samples = epochs_left * batch_size

                if self.broadcast_samples:
                    broadcasted_samples = torch.stack(
                        [dist.sample(total_samples) for dist in broadcasters]
                    )
                    sample_buffer = broadcasted_samples[broadcast_map]
                else:
                    sample_buffer = torch.stack(
                        [
                            model.distribution.sample(total_samples)
                            for model in flattened_models
                        ]
                    )

            start = buf_offset * batch_size
            end = start + batch_size
            stacked_samples = sample_buffer[:, start:end, :]  # ty:ignore

            # CUDA perf: set_to_none=True avoids a memset kernel per parameter
            optimizer.zero_grad(set_to_none=True)
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

            if loss_buffer is not None:
                # CUDA perf: store on device to avoid a GPU→CPU sync every epoch.
                # Transferred to CPU in one batch after the loop.
                loss_buffer[ep] = total_loss.detach()

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
            self._sync_generators(broadcasters, broadcast_map)

        # Convert device-side loss buffer to a Python list in one transfer
        if loss_buffer is not None:
            losses = loss_buffer.cpu().tolist()

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

        numpy_key: list = []
        new_axes: list[Axis] = []

        for dim, k in enumerate(key):
            axis: Axis = self.axes[dim]
            dim_size: int = len(axis.values)

            if isinstance(k, int):
                idx: int = k + dim_size if k < 0 else k
                if idx < 0 or idx >= dim_size:
                    raise IndexError(
                        f"\nIndex {k} out of bounds for axis '{axis.label}' with size {dim_size}"
                    )
                # Integer index collapses the axis (NumPy convention)
                numpy_key.append(idx)

            elif isinstance(k, slice):
                start, stop, step = k.start, k.stop, k.step
                if start is not None and start < 0:
                    start += dim_size
                if stop is not None and stop < 0:
                    stop += dim_size

                if start is not None and (start < 0 or start >= dim_size):
                    raise IndexError(
                        f"\nSlice start {k.start} out of bounds for axis '{axis.label}' with size {dim_size}"
                    )
                if stop is not None and (stop < 0 or stop > dim_size):
                    raise IndexError(
                        f"\nSlice stop {k.stop} out of bounds for axis '{axis.label}' with size {dim_size}"
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
                    values = [axis.values[i] for i in indices]
                else:
                    values = axis.values[s]

                # Preserve axis type (e.g., TrainingAxis)
                if isinstance(axis, TrainingAxis):
                    new_axes.append(TrainingAxis(label=axis.label, values=values))
                else:
                    new_axes.append(Axis(label=axis.label, values=values))
            else:
                raise IndexError(f"\nUnsupported index type: {type(k)}")

        for dim in range(len(key), len(self.axes)):
            new_axes.append(self.axes[dim])
            numpy_key.append(slice(None))

        result = self.models[tuple(numpy_key)]

        # If result is a scalar (all indices were integers), return the ToyModel
        if not isinstance(result, np.ndarray):
            return result

        return ModelGrid(
            create_model=self.create_model,
            axes=new_axes,
            broadcast_samples=self.broadcast_samples,
            _models=result,
        )

    def train_saes(
        self,
        saes: list[SAEEntry] | Callable[[ToyModel], list[SAEEntry]],
        training_samples: int = 10_000_000,
        batch_size: int = 1024,
        lr: float = 0.0003,
        lr_warm_up_steps: int = 0,
        lr_decay_steps: int = 0,
        n_snapshots: int = 0,
        snapshot_fn: Callable[[Any], None] | None = None,
        autocast_sae: bool = False,
        autocast_data: bool = False,
        verbose: bool = False,
        n_loss_snapshots: int | None = None,
    ) -> None:
        """Train SAE(s) on each ToyModel in the grid.

        Args:
            saes: A list of :class:`SAEEntry` instances, or a callable that
                takes a :class:`ToyModel` and returns such a list.
            training_samples: Number of training samples (sae_lens param, default: 10M).
            batch_size: Training batch size (sae_lens param, default: 1024).
            lr: Learning rate (sae_lens param, default: 0.0003).
            lr_warm_up_steps: Number of warmup steps (sae_lens param, default: 0).
            lr_decay_steps: Number of decay steps (sae_lens param, default: 0).
            n_snapshots: Number of training snapshots (sae_lens param, default: 0).
            snapshot_fn: Optional callback for snapshots (sae_lens param).
            autocast_sae: Use autocast for SAE (sae_lens param, default: False).
            autocast_data: Use autocast for data (sae_lens param, default: False).
            verbose: Whether to show progress bars. Defaults to False.
            n_loss_snapshots: If set, record the overall loss at this many
                evenly-spaced snapshots and store in each SAERecord.losses. None
                (default) disables loss tracking.
        """
        flattened_models: NDArray[np.object_] = self.models.ravel()

        for model in tqdm(
            flattened_models, desc="Training SAEs", unit="model", disable=not verbose
        ):
            model.train_saes(
                saes=deepcopy(saes),
                training_samples=training_samples,
                batch_size=batch_size,
                lr=lr,
                lr_warm_up_steps=lr_warm_up_steps,
                lr_decay_steps=lr_decay_steps,
                n_snapshots=n_snapshots,
                snapshot_fn=snapshot_fn,
                autocast_sae=autocast_sae,
                autocast_data=autocast_data,
                verbose=verbose,
                n_loss_snapshots=n_loss_snapshots,
            )

    def evaluate_saes(
        self,
        labels: list[str] | None = None,
        num_samples: int = 100_000,
        verbose: bool = False,
    ) -> None:
        """Evaluate stored SAEs on each ToyModel in the grid.

        Args:
            labels: List of SAE labels to evaluate. Defaults to all stored SAEs.
            num_samples: Number of samples to use for evaluation.
            verbose: Whether to show progress bars. Defaults to False.

        Returns:
           None
        """
        flattened_models: NDArray[np.object_] = self.models.ravel()

        for model in tqdm(
            flattened_models, desc="Evaluating SAEs", unit="model", disable=not verbose
        ):
            model.evaluate_saes(labels=labels, num_samples=num_samples, verbose=verbose)

    def sae_results_to_dataframe(self):
        """Convert SAE evaluation results to a pandas DataFrame.

        Returns a DataFrame with a ``(distribution, sae)`` MultiIndex on the rows
        and one column per metric. Only includes SAEs that have been evaluated
        (results is not None).

        Returns:
            A DataFrame with ``(distribution, sae)`` row MultiIndex and metric names
            as columns. Metrics are derived dynamically from the result objects, with
            nested fields (e.g. ``classification``) flattened into the top level.

        Example::

            grid.evaluate_saes()
            df = grid.sae_results_to_dataframe()
            df.loc["SPARSE_UNIFORM"]  # all SAEs, all metrics
            df.loc["SPARSE_UNIFORM"].xs("Standard", level="sae")  # one SAE, all metrics
            df.loc["SPARSE_UNIFORM"].xs(
                ["Standard", "Matryoshka"], level="sae"
            )  # two SAEs, all metrics
            df[["f1_score", "mcc"]]  # filter metrics
        """
        axis_labels = [axis.label for axis in self.axes]

        rows: list[dict[str, Any]] = []
        for idx in np.ndindex(*self.shape):
            # [2026-04-02 | OliverSieweke] TODO: this feels like something that should be utility on model grid
            model: ToyModel = self.models[idx]
            axis_values = {}
            for i, axis in enumerate(self.axes):
                value = axis.values[idx[i]]
                axis_values[axis.label] = (
                    value.name
                    if hasattr(value, "name") and isinstance(value.name, str)
                    else str(value)
                )

            for sae_label, sae_record in model.saes.items():
                if sae_record.results is None:
                    continue
                metrics = asdict(sae_record.results)
                # Flatten any nested dataclass fields (e.g. classification)
                for key, value in list(metrics.items()):
                    if isinstance(value, dict):
                        metrics.update(metrics.pop(key))
                row = {**axis_values, "sae": sae_label, **metrics}
                if sae_record.sae_type is not None:
                    row["sae_type"] = sae_record.sae_type
                if sae_record.params:
                    for param_key, param_value in sae_record.params.items():
                        row[param_key] = param_value
                rows.append(row)

        if not rows:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=axis_labels + ["sae"])
            )

        tidy = pd.DataFrame(rows)
        # [2026-04-02 | OliverSieweke] TODO: based on benchmark axis here - don't assume it's the first.
        non_benchmark_axes = [label for label in axis_labels if label != axis_labels[0]]
        tidy = tidy.set_index(
            axis_labels[:1] + ["sae"] + non_benchmark_axes
        ).sort_index()
        tidy.columns.name = "metric"
        return tidy

    def save(self, path: str | Path) -> None:
        """Save grid to disk using dill.

        Args:
            path: File path to save to (will be created/overwritten).

        Example::
            grid.save("my_grid.pkl")

        Warning:
            Uses dill/pickle. If you refactor code (rename classes, change imports),
            old saves may fail to load. Just re-save after refactoring.
        """
        with open(path, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> ModelGrid:
        """Load a ModelGrid from disk.

        Args:
            path: File path to load from.

        Returns:
            A fully reconstructed ModelGrid.

        Example::
            grid = ModelGrid.load("my_grid.pkl")
        """
        with open(path, "rb") as file:
            return dill.load(file)
