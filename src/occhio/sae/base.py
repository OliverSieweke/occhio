"""
Base class for Sparse AutoEncoders.
"""

from torch import Tensor
from torch.optim import AdamW
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SparseAutoEncoderBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """features --> latent"""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """latent --> features"""

    @abstractmethod
    def resample_weights(self):
        """Reset / resample all weights"""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        """
        The standard SAE loss function
        """
        sparsity_loss = torch.mean(torch.sum(torch.abs(intermediate), dim=-1))
        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))
        return self.l1_coef * sparsity_loss + mse_loss

    def _resample_dead(self, data_fn, batch_size: int) -> int:
        """Resample dead neurons. Override in subclasses for targeted resampling.

        Returns the number of resampled neurons.
        """
        return 0

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
        sample_every: int = 25,
        resample_every: int = 0,
    ) -> list[float]:
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        optimizer = AdamW(self.parameters(), lr=lr)
        sae_device = next(self.parameters()).device
        loss_buffer = torch.empty(n_steps, device=sae_device)

        raw_buffer: Tensor | None = None

        for step in range(n_steps):
            buf_offset = step % sample_every
            if buf_offset == 0:
                steps_left = min(sample_every, n_steps - step)
                total_samples = steps_left * batch_size
                raw_buffer = data_fn(total_samples).detach().to(sae_device)

            assert raw_buffer is not None
            start = buf_offset * batch_size
            end = start + batch_size
            x = raw_buffer[start:end]

            optimizer.zero_grad()
            x_hat, z = self.forward(x)
            loss = self.loss(x, x_hat, z)
            loss.backward()
            optimizer.step()

            loss_buffer[step] = loss.detach()
            if (step + 1) % 5000 == 0:
                print(f"  SAE step {step + 1}/{n_steps}  loss={loss.item():.4f}")

            if resample_every > 0 and (step + 1) % resample_every == 0:
                n_resampled = self._resample_dead(data_fn, batch_size)
                if n_resampled > 0:
                    print(
                        f"  SAE step {step + 1}: resampled {n_resampled} dead neurons"
                    )

        return loss_buffer.cpu().tolist()

    def __init__(
        self,
        l1_coef: float = 0.5,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.l1_coef = l1_coef
        self.device = device
        self.generator = generator
