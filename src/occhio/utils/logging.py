from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def suppress_tqdm():
    """Context manager to temporarily disable tqdm progress bars.

    This was added to suppress too verbose logging when training SAEs on model grids
    with SAE Lens.
    """
    import tqdm.auto as tqdm_auto

    original_init = tqdm_auto.tqdm.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["disable"] = True
        return original_init(self, *args, **kwargs)

    setattr(tqdm_auto.tqdm, "__init__", patched_init)
    try:
        yield
    finally:
        setattr(tqdm_auto.tqdm, "__init__", original_init)
