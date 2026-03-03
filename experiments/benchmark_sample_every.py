# ABOUTME: Benchmark script for sample_every optimization in ModelGrid.fit().
# ABOUTME: Reproduces profiling table comparing CPU/MPS × cached/uncached × sample_every values.

import time

import torch
from torch import Generator

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import ToyModel

SEED = 42
N_WARMUP = 3
N_RUNS = 5
N_EPOCHS = 200


def make_grid(
    n_grid: int,
    n_features: int,
    n_hidden: int,
    device: str,
    cache: bool,
) -> ModelGrid:
    def create_model(params: dict, **kwargs) -> ToyModel:
        density = params["density"]
        importance = params.get("importance", 1.0)
        gen = Generator(device=device).manual_seed(SEED)
        return ToyModel(
            distribution=SparseUniform(
                n_features, p_active=density, device=device, generator=gen
            ),
            ae=TiedLinearRelu(n_features, n_hidden, generator=gen, device=device),
            importances=importance ** torch.arange(n_features, dtype=torch.float32),
            device=device,
        )

    return ModelGrid(
        create_model,
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 1.0, n_grid)),
            Axis(label="importance", values=torch.linspace(0.5, 2.0, n_grid)),
        ],
        cache_samples=cache,
    )


def sync_device(device: str):
    if device == "mps":
        torch.mps.synchronize()


def benchmark_fit(
    n_grid: int,
    n_features: int,
    n_hidden: int,
    batch_size: int,
    device: str,
    cache: bool,
    sample_every: int,
) -> float:
    """Return median wall-clock ms per epoch."""
    times = []
    for run in range(N_WARMUP + N_RUNS):
        torch.manual_seed(SEED)
        grid = make_grid(n_grid, n_features, n_hidden, device, cache)
        sync_device(device)

        t0 = time.perf_counter()
        grid.fit(
            n_epochs=N_EPOCHS,
            batch_size=batch_size,
            sample_every=sample_every,
        )
        sync_device(device)
        elapsed = time.perf_counter() - t0

        if run >= N_WARMUP:
            times.append(elapsed / N_EPOCHS * 1000)  # ms per epoch

    times.sort()
    return times[len(times) // 2]  # median


def run_suite(label: str, n_grid: int, n_features: int, n_hidden: int, batch_size: int):
    print(f"\n{'=' * 80}")
    print(
        f"  {label} — {n_grid}×{n_grid} = {n_grid**2} models, "
        f"features={n_features}, hidden={n_hidden}, batch={batch_size}"
    )
    print(f"{'=' * 80}")

    devices = ["cpu", "mps"]
    cache_modes = [True, False]
    sample_every_values = [1, 10]

    # Header
    cols = []
    for dev in devices:
        for cached in cache_modes:
            c = "cached" if cached else "uncached"
            cols.append(f"{dev.upper()} {c}")
    header = f"{'sample_every':>14}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))

    for se in sample_every_values:
        row = f"{se:>14}"
        for dev in devices:
            for cached in cache_modes:
                ms = benchmark_fit(
                    n_grid, n_features, n_hidden, batch_size, dev, cached, se
                )
                row += f"{ms:>15.2f} ms"
        print(row)

    # Speedup row
    row = f"{'speedup':>14}"
    for dev in devices:
        for cached in cache_modes:
            ms_1 = benchmark_fit(
                n_grid, n_features, n_hidden, batch_size, dev, cached, 1
            )
            ms_10 = benchmark_fit(
                n_grid, n_features, n_hidden, batch_size, dev, cached, 10
            )
            speedup = ms_1 / ms_10 if ms_10 > 0 else float("inf")
            row += f"{speedup:>14.1f}×   "
    print(row)


if __name__ == "__main__":
    print("Benchmark: sample_every optimization in ModelGrid.fit()")
    print(
        f"Config: {N_EPOCHS} epochs, {N_WARMUP} warmup + {N_RUNS} timed runs (median)"
    )

    # SMALL grid — matches image: 10×10 = 100 models, features=5, hidden=2, batch=216
    run_suite("SMALL grid", n_grid=10, n_features=5, n_hidden=2, batch_size=216)

    # LARGE grid — matches image: 25×25 = 625 models, features=20, hidden=5, batch=512
    run_suite("LARGE grid", n_grid=25, n_features=20, n_hidden=5, batch_size=512)
