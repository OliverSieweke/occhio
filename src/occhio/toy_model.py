"""Defines ToyModel, the core experiment object combining a Distribution and AutoEncoderBase.

Provides fit(), geometric analysis properties (W, feature_norms, interferences, etc.), and sampling utilities.
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .autoencoder import AutoEncoderBase
from .distributions import Distribution
from .utils.device import _same_device


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

    def __init__(
        self,
        distribution: Distribution,
        ae: AutoEncoderBase,
        device: torch.device | str | None = None,
        importances: Tensor | list | None = None,
    ):

        self.distribution = distribution
        self.ae = ae

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
    ) -> tuple[list[float], list]:
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

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
        raw_buffer: Tensor | tuple | None = None

        for ep in range(n_epochs):
            buf_offset = ep % sample_every
            if buf_offset == 0:
                # Determine how many epochs remain to avoid over-sampling
                epochs_left = min(sample_every, n_epochs - ep)
                total_samples = epochs_left * batch_size

                raw_buffer = self.distribution.sample(total_samples)
                # Move samples to the ae device; distribution may be on a different device.
                if isinstance(raw_buffer, tuple):
                    raw_buffer = tuple(
                        t.to(ae_device) if isinstance(t, Tensor) else t
                        for t in raw_buffer
                    )
                else:
                    raw_buffer = raw_buffer.to(ae_device)

            start = buf_offset * batch_size
            end = start + batch_size
            if isinstance(raw_buffer, tuple):
                raw = tuple(
                    t[start:end] if isinstance(t, Tensor) else t for t in raw_buffer
                )
                x = raw[0]
            else:
                raw = raw_buffer[start:end]
                x = raw

            optimizer.zero_grad()
            x_hat = self.ae(x)[0]  # Only take x_hat
            loss = self.ae.loss(raw, x_hat, self.importances)  # type: ignore[call-arg]
            loss.backward()
            optimizer.step()

            if track_losses:
                loss_buffer[ep] = loss.detach()  # type: ignore[index]
            if verbose and (ep + 1) % 1000 == 0:
                print(f"AE Epoch {ep + 1}/{n_epochs}, Loss: {loss.item():.6f}")
            if hooks and (ep % hook_freq == 0 or ep == n_epochs - 1):
                with torch.no_grad():
                    hook_data = dict(
                        tm=self, epoch=ep, loss=loss.item(), x=x, x_hat=x_hat
                    )
                    for i, h in enumerate(hooks):
                        hook_returns[i].append(h(hook_data))

        losses = loss_buffer.cpu().tolist() if track_losses else []
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

        if name in ("encode", "decode", "forward", "resample_weights", "loss"):
            return getattr(self.ae, name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

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
