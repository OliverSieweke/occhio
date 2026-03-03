# ABOUTME: Phase-level benchmark of sample_every optimization in ModelGrid.fit().
# ABOUTME: Measures sample, zero_grad, forward, loss, backward, optim_step per epoch.

"""
Benchmark: sample_every phase-level profiling
==============================================
Large grid: 25×25 = 625 models, features=20, hidden=5, batch=512, epochs=2000

Measures wall-clock time per phase per epoch (ms), with MPS sync barriers
for accurate GPU timing. Compares sample_every=1, 10, 100 across
CPU cached, CPU uncached, MPS cached, MPS uncached.
"""

import time
from collections import defaultdict

import numpy as np
import torch
from torch import Generator, Tensor
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm import tqdm

from occhio.autoencoder import AutoEncoderBase, TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import ToyModel

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_GRID = 25
N_FEATURES = 20
N_HIDDEN = 5
BATCH_SIZE = 512
N_EPOCHS = 2_000
WARMUP_EPOCHS = 200  # warm up JIT / MPS pipeline before timing
SAMPLE_EVERY_VALUES = [1, 10, 100]


# ── Grid factory ──────────────────────────────────────────────────────────────


def make_grid(device: str, cache: bool) -> ModelGrid:
    def create_model(params: dict, **kwargs) -> ToyModel:
        density = params["density"]
        importance = params.get("importance", 1.0)
        gen = Generator(device=device).manual_seed(SEED)
        return ToyModel(
            distribution=SparseUniform(
                N_FEATURES, p_active=density, device=device, generator=gen
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=device),
            importances=importance ** torch.arange(N_FEATURES, dtype=torch.float32),
            device=device,
        )

    return ModelGrid(
        create_model,
        axes=[
            Axis(label="density", values=torch.linspace(0.1, 1.0, N_GRID)),
            Axis(label="importance", values=torch.linspace(0.5, 2.0, N_GRID)),
        ],
        cache_samples=cache,
    )


# ── Sync helper ───────────────────────────────────────────────────────────────


def sync(device: str):
    """Force MPS/CUDA synchronization so timers capture real GPU work."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


# ── Profiled training loop ────────────────────────────────────────────────────


def profiled_fit(
    grid: ModelGrid,
    n_epochs: int,
    batch_size: int,
    sample_every: int,
    device: str,
    warmup_epochs: int = 0,
) -> dict[str, float]:
    """
    Run the ModelGrid training loop with per-phase timing.

    Returns dict of phase_name -> mean ms/epoch (excluding warmup).
    Mirrors the exact logic of ModelGrid.fit() but with sync + timing barriers.
    """
    flattened_models = grid.models.ravel()

    # Stack params/buffers
    stacked_params, stacked_buffers = stack_module_state(
        [model.ae for model in flattened_models]
    )
    stacked_params = {k: v.requires_grad_(True) for k, v in stacked_params.items()}
    stacked_importances = torch.stack([model.importances for model in flattened_models])

    optimizer = AdamW(list(stacked_params.values()), lr=3e-4, weight_decay=0.05)

    representative_ae: AutoEncoderBase = flattened_models[0].ae
    stacked_forward = torch.vmap(
        lambda params, buffers, x: functional_call(
            representative_ae, (params, buffers), (x,)
        )[0],
        in_dims=(0, 0, 0),
    )

    stacked_loss_fn = torch.vmap(
        lambda x_true, x_hat, importances: representative_ae.loss(
            x_true, x_hat, importances
        ),
        in_dims=(0, 0, 0),
    )

    # Phase accumulators (only accumulate after warmup)
    phases = ["sample", "zero_grad", "forward", "loss", "backward", "optim_step"]
    totals = {p: 0.0 for p in phases}
    timed_epochs = 0

    sample_buffer: Tensor | None = None
    total_epochs = warmup_epochs + n_epochs

    for ep in tqdm(range(total_epochs), desc=f"se={sample_every}", leave=False):
        recording = ep >= warmup_epochs

        # ── sample ────────────────────────────────────────────────
        sync(device)
        t0 = time.perf_counter()

        buf_offset = ep % sample_every
        if buf_offset == 0:
            epochs_left = min(sample_every, total_epochs - ep)
            total_samples = epochs_left * batch_size

            if grid.cache_samples:
                unique_samples = torch.stack(
                    [dist.sample(total_samples) for dist in grid._unique_distributions]
                )
                sample_buffer = unique_samples[grid._sample_index]
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

        sync(device)
        t1 = time.perf_counter()

        # ── zero_grad ─────────────────────────────────────────────
        optimizer.zero_grad()
        sync(device)
        t2 = time.perf_counter()

        # ── forward ───────────────────────────────────────────────
        stacked_x_hat = stacked_forward(
            stacked_params, stacked_buffers, stacked_samples
        )
        sync(device)
        t3 = time.perf_counter()

        # ── loss ──────────────────────────────────────────────────
        stacked_losses = stacked_loss_fn(
            stacked_samples, stacked_x_hat, stacked_importances
        )
        total_loss: Tensor = stacked_losses.mean()
        sync(device)
        t4 = time.perf_counter()

        # ── backward ──────────────────────────────────────────────
        total_loss.backward()
        sync(device)
        t5 = time.perf_counter()

        # ── optim_step ────────────────────────────────────────────
        optimizer.step()
        sync(device)
        t6 = time.perf_counter()

        if recording:
            totals["sample"] += t1 - t0
            totals["zero_grad"] += t2 - t1
            totals["forward"] += t3 - t2
            totals["loss"] += t4 - t3
            totals["backward"] += t5 - t4
            totals["optim_step"] += t6 - t5
            timed_epochs += 1

    # Convert to mean ms/epoch
    return {p: (totals[p] / timed_epochs) * 1000 for p in phases}


# ── Pretty printing ──────────────────────────────────────────────────────────


def print_table(results: dict[tuple[str, bool, int], dict[str, float]]):
    """
    Print results as the comparison table.
    Columns: phase | CPU cached | CPU uncached | MPS cached | MPS uncached
    One table per sample_every value.
    """
    phases = ["sample", "zero_grad", "forward", "loss", "backward", "optim_step"]
    configs = [
        ("cpu", True, "CPU cached"),
        ("cpu", False, "CPU uncached"),
        ("mps", True, "MPS cached"),
        ("mps", False, "MPS uncached"),
    ]

    for se in SAMPLE_EVERY_VALUES:
        print(f"\n{'─' * 80}")
        print(f"  sample_every = {se}")
        print(f"{'─' * 80}")

        header = f"{'Phase':>14}" + "".join(f"{label:>16}" for _, _, label in configs)
        print(header)
        print(f"{'─' * 14}" + "─" * 16 * len(configs))

        row_total = {label: 0.0 for _, _, label in configs}

        for phase in phases:
            row = f"{phase:>14}"
            for dev, cached, label in configs:
                key = (dev, cached, se)
                ms = results[key][phase]
                row_total[label] += ms
                row += f"{ms:>13.2f} ms"
            print(row)

        # TOTAL row
        row = f"{'TOTAL':>14}"
        for _, _, label in configs:
            row += f"{row_total[label]:>13.2f} ms"
        print(f"{'─' * 14}" + "─" * 16 * len(configs))
        print(row)

        # sample % row
        row = f"{'sample %':>14}"
        for _, _, label in configs:
            pct = row_total[label] and (
                results[
                    (
                        [d for d, c, l in configs if l == label][0],
                        [c for d, c, l in configs if l == label][0],
                        se,
                    )
                ]["sample"]
                / row_total[label]
                * 100
            )
            row += f"{pct:>12.0f}%   "
        print(row)

    # ── Speedup comparison table ──────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("  SPEEDUP vs sample_every=1  (total ms/epoch)")
    print(f"{'=' * 80}")

    header = f"{'sample_every':>14}" + "".join(
        f"{label:>16}" for _, _, label in configs
    )
    print(header)
    print(f"{'─' * 14}" + "─" * 16 * len(configs))

    for se in SAMPLE_EVERY_VALUES:
        row = f"{se:>14}"
        for dev, cached, label in configs:
            baseline = sum(results[(dev, cached, 1)].values())
            current = sum(results[(dev, cached, se)].values())
            speedup = baseline / current if current > 0 else float("inf")
            if se == 1:
                row += f"{current:>11.2f} ms "
            else:
                row += f"{current:>7.2f} ms ({speedup:.1f}×)"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Benchmark: phase-level profiling of sample_every optimization")
    print(f"Grid: {N_GRID}×{N_GRID} = {N_GRID**2} models")
    print(f"features={N_FEATURES}, hidden={N_HIDDEN}, batch={BATCH_SIZE}")
    print(f"epochs={N_EPOCHS} (+ {WARMUP_EPOCHS} warmup)")
    print(f"sample_every values: {SAMPLE_EVERY_VALUES}")

    results: dict[tuple[str, bool, int], dict[str, float]] = {}

    devices = ["cpu", "mps"]
    cache_modes = [True, False]

    total_runs = len(devices) * len(cache_modes) * len(SAMPLE_EVERY_VALUES)
    run_idx = 0

    for dev in devices:
        for cached in cache_modes:
            for se in SAMPLE_EVERY_VALUES:
                run_idx += 1
                cache_label = "cached" if cached else "uncached"
                print(
                    f"\n[{run_idx}/{total_runs}] {dev.upper()} {cache_label}, "
                    f"sample_every={se}"
                )

                torch.manual_seed(SEED)
                grid = make_grid(dev, cached)
                phase_times = profiled_fit(
                    grid,
                    n_epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    sample_every=se,
                    device=dev,
                    warmup_epochs=WARMUP_EPOCHS,
                )
                results[(dev, cached, se)] = phase_times

                total_ms = sum(phase_times.values())
                print(f"  → {total_ms:.2f} ms/epoch total")

    print_table(results)
