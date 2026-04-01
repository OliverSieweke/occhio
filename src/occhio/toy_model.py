"""Defines ToyModel, the core experiment object combining a Distribution and AutoEncoderBase.

Provides fit(), geometric analysis properties (W, feature_norms, interferences, etc.), and sampling utilities.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sae_lens import TrainingSAE
from sae_lens.synthetic import (
    SyntheticDataEvalResult,
    eval_sae_on_synthetic_data,
    train_toy_sae,
)
from torch import Tensor
from torch.optim import AdamW, Optimizer
from tqdm.auto import tqdm

from .autoencoder import AutoEncoderBase
from .distributions import Distribution
from .sae_lens_adapter.activation_generator import ActivationGeneratorWrapper
from .sae_lens_adapter.feature_dictionary import FeatureDictionaryWrapper
from .utils.device import _same_device


@dataclass
class SAERecord:
    """Holds an SAE and its evaluation results."""

    sae: TrainingSAE
    results: SyntheticDataEvalResult | None = None


class ToyModel:
    """This is the ToyModel class which is the base for most experiments.

    Args:
        distribution: A Distribution object. May live on a different device from
            the AutoEncoder (e.g. CPU for fast sampling while ae runs on GPU/MPS).
            Samples are moved to the ae device automatically during training.
        ae: An AutoEncoderBase object.
        device: Device for the AutoEncoder and all computation. If the distribution
            has no explicit device it is also moved here for convenience. Pass
            ``None`` to infer from the ae or distribution.
        generator: For seeded experiments.
        importances: Weighing of the distribution.
    """

    distribution: Distribution
    ae: AutoEncoderBase
    saes: dict[str, SAERecord]

    def __init__(
        self,
        distribution: Distribution,
        ae: AutoEncoderBase,
        device: torch.device | str | None = None,
        importances: Tensor | list | None = None,
    ):

        self.distribution = distribution
        self.ae = ae
        self.saes = {}

        if distribution.n_features != ae.n_features:
            raise ValueError(
                f"Distribution has {distribution.n_features} features "
                f"but AutoEncoder has {ae.n_features}."
            )
        self.n_features: int = ae.n_features

        # Resolve the ae/computation device.
        if device is None:
            ae_device = (
                ae._init_device or distribution._init_device or torch.device("cpu")
            )
        else:
            ae_device = torch.device(device)
            if ae._init_device is not None and not _same_device(
                ae._init_device, ae_device
            ):
                raise ValueError(
                    f"AutoEncoder was explicitly created on {ae._init_device}, "
                    f"but ToyModel device is {ae_device}. "
                    f"Either omit the device from the AutoEncoder or make them match."
                )

        ae.to(ae_device)
        self.device = ae_device

        # Distribution keeps its own device when it was explicitly placed on one,
        # allowing distribution and ae to live on different devices (e.g. CPU vs MPS).
        # If the distribution has no explicit device, move it alongside the ae for
        # backward-compatible behaviour.
        if distribution._init_device is None:
            distribution.to(ae_device)

        if importances is None:
            self.importances = torch.ones(self.n_features, device=ae_device)
        else:
            if isinstance(importances, Tensor):
                self.importances = importances.to(ae_device)
            else:
                self.importances = torch.tensor(importances, device=ae_device)

    @staticmethod
    def _validate_data_file(
        tensors: dict[str, Tensor],
        path: Path,
        n_features: int,
        batch_size: int,
    ) -> None:
        """Validate an already-loaded safetensors dict for use with :meth:`fit`.

        Checks that the dict contains exactly one key, is 2-D, has the
        correct feature dimension, and warns if batch_size exceeds the
        dataset size.
        """
        if len(tensors) != 1:
            raise ValueError(
                f"Expected exactly 1 tensor key in {path.name}, "
                f"got {len(tensors)}: {list(tensors.keys())}"
            )
        tensor = next(iter(tensors.values()))
        if tensor.dim() != 2:
            raise ValueError(f"Expected a 2-D tensor, got shape {list(tensor.shape)}")
        if tensor.shape[1] != n_features:
            raise ValueError(
                f"Feature dimension mismatch: file has {tensor.shape[1]}, "
                f"model expects {n_features}"
            )
        n_samples = tensor.shape[0]
        if batch_size > n_samples:
            warnings.warn(
                f"batch_size ({batch_size}) is >100% of dataset size ({n_samples}). "
                f"Consider using a larger dataset or smaller batch_size to reduce "
                f"duplicate samples across epochs.",
                stacklevel=3,
            )
        print(
            f"[ToyModel.fit] Using precomputed data from {path.name} "
            f"({n_samples} samples, {n_features} features). "
            f"Distribution sampling is disabled."
        )

    # If you change the signature or implementation here, make sure you keep it
    # consistent with ModelGrid.fit()
    def fit(
        self,
        n_epochs: int,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        track_losses: bool = True,
        optimizer: Optimizer | None = None,
        hooks: list[Callable] | None = None,
        hook_freq: int = 1,
        verbose: bool = False,
        sample_every: int = 25,
        precomputed_data: str | Path | None = None,
    ) -> tuple[list[float], list]:
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        # --- Precomputed data overrides Distribution sampling ---
        # When precomputed_data is provided, the Distribution is NOT used at all.
        # Instead, each epoch takes the next sequential batch from the dataset,
        # wrapping around when exhausted. This is fully deterministic and does
        # not involve any random number generation.
        precomputed: Tensor | None = None
        if precomputed_data is not None:
            data_path = Path(precomputed_data)
            if data_path.suffix != ".safetensors":
                data_path = data_path.with_suffix(".safetensors")
            loaded = load_file(str(data_path))
            self._validate_data_file(loaded, data_path, self.ae.n_features, batch_size)
            precomputed = next(iter(loaded.values())).to(self.ae.device)

        if optimizer is None:
            optimizer = AdamW(
                self.ae.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        if hooks is None:
            hooks = []
        hook_returns = [[] for _ in hooks]

        ae_device = self.ae.device
        # Pre-allocate a device-side buffer so loss tracking never forces a
        # per-step MPS→CPU sync. Converted to a Python list in one transfer
        # at the end.
        loss_buffer = torch.empty(n_epochs, device=ae_device) if track_losses else None

        # Pre-allocated sample buffer: sample once every `sample_every` epochs
        # with sample_every × batch_size samples, then slice per epoch.
        raw_buffer: Tensor | tuple[Any, ...] | None = None

        for ep in range(n_epochs):
            if precomputed is not None:
                # Deterministic sequential iteration: cycle through the dataset
                # in fixed order. Wraps around when n_epochs * batch_size > N.
                n_total = precomputed.shape[0]
                start = (ep * batch_size) % n_total
                end = start + batch_size
                if end <= n_total:
                    raw = precomputed[start:end]
                else:
                    # Wrap around the end of the dataset
                    raw = torch.cat([precomputed[start:], precomputed[: end - n_total]])
                x = raw
            else:
                buf_offset = ep % sample_every
                if buf_offset == 0:
                    # Determine how many epochs remain to avoid over-sampling
                    epochs_left = min(sample_every, n_epochs - ep)
                    total_samples = epochs_left * batch_size

                    raw_buffer = self.distribution.sample(total_samples)
                    # Move samples to the ae device; distribution may be on a different device.
                    if isinstance(raw_buffer, tuple):
                        raw_buffer = tuple(
                            t.to(ae_device, non_blocking=True)
                            if isinstance(t, Tensor)
                            else t
                            for t in raw_buffer
                        )
                    else:
                        raw_buffer = raw_buffer.to(ae_device, non_blocking=True)

                assert raw_buffer is not None  # Always set on first iter (0 % n == 0)
                start = buf_offset * batch_size
                end = start + batch_size
                if isinstance(raw_buffer, tuple):
                    raw_buffer = tuple(
                        (
                            t.to(ae_device, non_blocking=True)
                            if isinstance(t, Tensor)
                            else t
                        )
                        for t in raw_buffer
                    )
                    raw = tuple(
                        t[start:end] if isinstance(t, Tensor) else t for t in raw_buffer
                    )
                    x = raw[0]
                else:
                    raw = raw_buffer[start:end]
                    x = raw

            optimizer.zero_grad(set_to_none=True)
            x_hat = self.ae(x)[0]  # Only take x_hat
            loss = self.ae.loss(raw, x_hat, self.importances)  # type: ignore[call-arg]
            loss.backward()
            optimizer.step()

            if loss_buffer is not None:
                loss_buffer[ep] = loss.detach()
            if verbose and (ep + 1) % 5000 == 0:
                print(f"AE Epoch {ep + 1}/{n_epochs}, Loss: {loss.item():.6f}")
            if hooks and (ep % hook_freq == 0 or ep == n_epochs - 1):
                with torch.no_grad():
                    hook_data = dict(
                        tm=self, epoch=ep, loss=loss.detach(), x=x, x_hat=x_hat
                    )
                    for i, h in enumerate(hooks):
                        hook_returns[i].append(h(hook_data))

        losses = loss_buffer.cpu().tolist() if loss_buffer is not None else []
        return losses, hook_returns

    def sample_latent(self, batch_size) -> Tensor:
        raw = self.distribution.sample(batch_size)
        x = raw[0] if isinstance(raw, tuple) else raw
        return self.ae.encode(x.to(self.ae.device))

    def get_one_hot_embeddings(self) -> Tensor:
        return self.ae.encode(torch.eye(self.n_features, device=self.ae.device))

    def __repr__(self):
        return f"ToyModel({self.distribution})"

    def __getattr__(self, name):
        if name == "sample":
            return getattr(self.distribution, name)

        if name in (
            "encode",
            "decode",
            "forward",
            "resample_weights",
            "loss",
            "n_features",
            "n_hidden",
        ):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def train_saes(
        self,
        saes: dict[str, TrainingSAE],
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
    ) -> None:
        """Train SAE(s) on this model's hidden activations using SAE Lens.

        Args:
            saes: Dict mapping labels to SAEs.
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

        Returns:
            None
        """
        for label, sae in saes.items():
            if label in self.saes.keys():
                warnings.warn(
                    f"An sae with the label '{label}' was already trained on this model and is being overwritten.",
                    UserWarning,
                    stacklevel=2,
                )

            train_toy_sae(
                sae=sae,
                feature_dict=FeatureDictionaryWrapper(self.ae),
                activations_generator=ActivationGeneratorWrapper(self.distribution),
                training_samples=training_samples,
                batch_size=batch_size,
                lr=lr,
                lr_warm_up_steps=lr_warm_up_steps,
                lr_decay_steps=lr_decay_steps,
                device=self.device,
                n_snapshots=n_snapshots,
                snapshot_fn=snapshot_fn,
                autocast_sae=autocast_sae,
                autocast_data=autocast_data,
            )
            self.saes[label] = SAERecord(sae=sae)

    def evaluate_saes(
        self,
        labels: list[str] | None = None,
        num_samples: int = 100_000,
        verbose: bool = False,
    ) -> dict[str, SyntheticDataEvalResult]:
        """Evaluate stored SAEs.

        Args:
            labels: List of SAE labels to evaluate. Defaults to all stored SAEs.
            num_samples: Number of samples to use for evaluation.
            verbose: Whether to show progress bars. Defaults to False.

        Returns:
            Dict of results keyed by SAE label.
        """
        if labels is None:
            labels = list(self.saes.keys())

        unmatched_labels = [label for label in labels if label not in self.saes]
        if unmatched_labels:
            raise ValueError(
                f"The following SAE labels do not exist: {', '.join(unmatched_labels)}. "
                f"Available labels: {', '.join(self.saes.keys())}"
            )

        results = {}
        with tqdm(
            labels, desc="SAEs", unit="SAE", leave=False, disable=not verbose
        ) as pbar:
            for label in pbar:
                sae_record = self.saes[label]
                if sae_record.results is not None:
                    warnings.warn(
                        f"SAE '{label}' was already evaluated. Re-evaluating and overwriting previous results.",
                        stacklevel=2,
                    )
                sae_record.sae.to(self.device)

                sae_record.results = eval_sae_on_synthetic_data(
                    sae=sae_record.sae,
                    feature_dict=FeatureDictionaryWrapper(self.ae),
                    activations_generator=ActivationGeneratorWrapper(self.distribution),
                    num_samples=num_samples,
                )
                results[label] = sae_record.results

        return results

    # ----------------------------------------------------------------------------------
    # Model Metrics --------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    @property
    @torch.no_grad()
    def frobenius_norm_squared(self):
        return torch.linalg.norm(self.W, ord="fro") ** 2

    @property
    @torch.no_grad()
    def hidden_dimensions_per_embedded_features(self) -> Any:
        return self.ae.n_hidden / self.frobenius_norm_squared

    @property
    @torch.no_grad()
    def embedded_features_per_hidden_dimensions(self) -> Any:
        return self.frobenius_norm_squared / self.ae.n_hidden

    @property
    @torch.no_grad()
    def feature_dimensionalities(self):
        return (
            self.feature_representations
            / self.total_feature_interferences_including_self
        )

    @property
    @torch.no_grad()
    def mean_feature_dimensionalities(self):
        return self.feature_dimensionalities.mean()

    @property
    @torch.no_grad()
    def total_feature_dimensionalities_per_hidden_dimension(self):
        return self.feature_dimensionalities.sum() / self.ae.n_hidden

    @property
    @torch.no_grad()
    def W(self) -> Tensor:
        return self.get_one_hot_embeddings().T

    @property
    @torch.no_grad()
    def W_T_W(self) -> Tensor:
        return self.W.T @ self.W

    @property
    @torch.no_grad()
    def W_normalized_features(self) -> Tensor:
        return F.normalize(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_norms(self) -> Tensor:
        return torch.linalg.vector_norm(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_representations(self) -> Tensor:
        return (self.W**2).sum(dim=0)

    @property
    @torch.no_grad()
    def interferences_sq(self) -> Tensor:
        return (self.W_normalized_features.T @ self.W) ** 2

    @property
    @torch.no_grad()
    def interferences(self) -> Tensor:
        return self.W_normalized_features.T @ self.W

    @property
    @torch.no_grad()
    def total_feature_interferences(self) -> Tensor:
        interferences = self.interferences_sq.clone()
        return interferences.fill_diagonal_(0).sum(dim=1)

    @property
    @torch.no_grad()
    def total_feature_interferences_including_self(self) -> Tensor:
        return self.interferences_sq.sum(dim=1)

    # ----------------------------------------------------------------------------------
    # SAE Metrics ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    @property
    def saes_precision(self) -> dict[str, float]:
        """Mean precision across SAE latents (TP / (TP + FP))"""
        return {
            label: sae_record.results.classification.precision
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_recall(self) -> dict[str, float]:
        """Mean recall across SAE latents (TP / (TP + FN))"""
        return {
            label: sae_record.results.classification.recall
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_f1_score(self) -> dict[str, float]:
        """Mean F1 score across SAE latents (harmonic mean of precision and recall)"""
        return {
            label: sae_record.results.classification.f1_score
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_accuracy(self) -> dict[str, float]:
        """Mean accuracy across SAE latents ((TP + TN) / total)"""
        return {
            label: sae_record.results.classification.accuracy
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_explained_variance(self) -> dict[str, float]:
        """Explained variance for evaluated SAEs."""
        return {
            label: sae_record.results.explained_variance
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_l0(self) -> dict[str, float]:
        """L0 sparsity for evaluated SAEs."""
        return {
            label: sae_record.results.sae_l0
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_dead_latents(self) -> dict[str, int]:
        """Dead latent count for evaluated SAEs."""
        return {
            label: sae_record.results.dead_latents
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_true_l0(self) -> dict[str, float]:
        """True L0 (ground truth feature activations) for evaluated SAEs."""
        return {
            label: sae_record.results.true_l0
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_shrinkage(self) -> dict[str, float]:
        """Shrinkage (ratio of SAE output norm to input norm) for evaluated SAEs."""
        return {
            label: sae_record.results.shrinkage
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_mcc(self) -> dict[str, float]:
        """Mean Correlation Coefficient between SAE decoder and ground truth features."""
        return {
            label: sae_record.results.mcc
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }

    @property
    def saes_uniqueness(self) -> dict[str, float]:
        """Fraction of SAE latents tracking unique ground-truth features."""
        return {
            label: sae_record.results.uniqueness
            for label, sae_record in self.saes.items()
            if sae_record.results is not None
        }
