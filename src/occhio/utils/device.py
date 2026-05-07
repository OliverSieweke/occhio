"""Device utilities for consistent device handling across occhio."""

import torch
from torch import Tensor


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Compare two devices, treating a missing index as 0 (e.g. mps == mps:0)."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def ensure_device(
    tensor: Tensor,
    device: torch.device | str,
    *,
    non_blocking: bool = True,
) -> Tensor:
    """Move tensor to device only if needed, avoiding no-op .to() overhead."""
    target = torch.device(device)
    if tensor.device.type == target.type and (tensor.device.index or 0) == (
        target.index or 0
    ):
        return tensor
    return tensor.to(device, non_blocking=non_blocking)


def seeded_generator(
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Create a seeded torch.Generator on the specified device.

    Args:
        seed: Random seed.
        device: Device for the generator. Defaults to CPU.

    Returns:
        A seeded ``torch.Generator``.
    """
    dev = device or "cpu"
    return torch.Generator(device=dev).manual_seed(seed)
