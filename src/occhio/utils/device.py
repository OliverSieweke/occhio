import torch


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Compare two devices, treating a missing index as 0 (e.g. mps == mps:0)."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)
