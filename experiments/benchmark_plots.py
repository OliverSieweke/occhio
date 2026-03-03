# ABOUTME: Generates a 6-subplot plotly figure benchmarking sample_every optimization.
# ABOUTME: Row 1 = memory usage over epochs; Row 2 = epochs/s over epochs.

"""
Benchmark visualization: sample_every memory & throughput
=========================================================
Produces one HTML figure with 6 subplots (2 rows × 3 cols):

  Row 1: Memory (MB) over epochs for sample_every = 1, 10, 100
  Row 2: Epochs/s (rolling avg) over epochs for sample_every = 1, 10, 100

Each subplot has 4 lines: CPU cached, CPU uncached, MPS cached, MPS uncached.

Grid: 25×25 = 625 models, features=10, hidden=3, batch=512, 2000 epochs.
Distribution: HierarchicalSparse (max_children=3, depth_decay=0.85).
"""

import os
import time

import psutil
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import Generator, Tensor
from torch.func import functional_call, stack_module_state
from torch.optim import AdamW
from tqdm import tqdm

from occhio.autoencoder import AutoEncoderBase, TiedLinearRelu
from occhio.distributions.hierarchical import HierarchicalSparse
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import ToyModel

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_GRID = 25
N_FEATURES = 10
N_HIDDEN = 3
BATCH_SIZE = 512
MAX_CHILDREN = 3
DEPTH_DECAY = 0.85
N_EPOCHS = 2_000
WARMUP_EPOCHS = 100
SAMPLE_EVERY_VALUES = [1, 10, 100]
ROLLING_WINDOW = 50  # epochs for smoothing epochs/s
MEMORY_SAMPLE_INTERVAL = 10  # record memory every N epochs

COLORS = {
    ("cpu", True): "#636EFA",  # blue
    ("cpu", False): "#EF553B",  # red
    ("mps", True): "#00CC96",  # green
    ("mps", False): "#AB63FA",  # purple
}

LABELS = {
    ("cpu", True): "CPU cached",
    ("cpu", False): "CPU uncached",
    ("mps", True): "MPS cached",
    ("mps", False): "MPS uncached",
}


# ── Grid factory ──────────────────────────────────────────────────────────────


def make_grid(device: str, cache: bool) -> ModelGrid:
    def create_model(params: dict, **kwargs) -> ToyModel:
        p_base = params["p_base"]
        importance = params.get("importance", 1.0)
        gen = Generator(device=device).manual_seed(SEED)
        return ToyModel(
            distribution=HierarchicalSparse(
                N_FEATURES,
                p_base=p_base,
                depth_decay=DEPTH_DECAY,
                max_children=MAX_CHILDREN,
                device=device,
                generator=gen,
            ),
            ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=device),
            importances=importance ** torch.arange(N_FEATURES, dtype=torch.float32),
            device=device,
        )

    return ModelGrid(
        create_model,
        axes=[
            Axis(label="p_base", values=torch.linspace(0.1, 0.95, N_GRID)),
            Axis(label="importance", values=torch.linspace(0.5, 2.0, N_GRID)),
        ],
        cache_samples=cache,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def get_memory_mb(device: str) -> float:
    """Return current memory usage in MB for the relevant device."""
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    else:
        # CPU: use process RSS
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)


# ── Profiled training loop ────────────────────────────────────────────────────


def profiled_fit(
    grid: ModelGrid,
    n_epochs: int,
    batch_size: int,
    sample_every: int,
    device: str,
    warmup_epochs: int,
) -> dict:
    """
    Run training loop and record per-epoch memory + timing.

    Returns:
        {
            "epoch": [int, ...],
            "memory_mb": [float, ...],     # sampled every MEMORY_SAMPLE_INTERVAL
            "memory_epoch": [int, ...],    # corresponding epoch indices
            "epoch_time_s": [float, ...],  # wall-clock seconds per epoch
        }
    """
    flattened_models = grid.models.ravel()

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

    epoch_times: list[float] = []
    memory_mb: list[float] = []
    memory_epochs: list[int] = []

    sample_buffer: Tensor | None = None
    total_epochs = warmup_epochs + n_epochs

    for ep in tqdm(range(total_epochs), desc=f"se={sample_every}", leave=False):
        recording = ep >= warmup_epochs
        timed_ep = ep - warmup_epochs

        sync(device)
        t_start = time.perf_counter()

        # ── sample ────────────────────────────────────────────────
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

        # ── forward + loss + backward + step ──────────────────────
        optimizer.zero_grad()
        stacked_x_hat = stacked_forward(
            stacked_params, stacked_buffers, stacked_samples
        )
        stacked_losses = stacked_loss_fn(
            stacked_samples, stacked_x_hat, stacked_importances
        )
        total_loss: Tensor = stacked_losses.mean()
        total_loss.backward()
        optimizer.step()

        sync(device)
        t_end = time.perf_counter()

        if recording:
            epoch_times.append(t_end - t_start)

            # Record memory periodically
            if timed_ep % MEMORY_SAMPLE_INTERVAL == 0:
                memory_mb.append(get_memory_mb(device))
                memory_epochs.append(timed_ep)

    return {
        "epoch_time_s": epoch_times,
        "memory_mb": memory_mb,
        "memory_epoch": memory_epochs,
    }


# ── Rolling average ──────────────────────────────────────────────────────────


def rolling_epochs_per_sec(
    epoch_times: list[float], window: int
) -> tuple[list[int], list[float]]:
    """Compute rolling-window epochs/s from per-epoch wall-clock times."""
    epochs = []
    eps_values = []
    for i in range(window, len(epoch_times)):
        chunk = epoch_times[i - window : i]
        avg_time = sum(chunk) / len(chunk)
        eps = 1.0 / avg_time if avg_time > 0 else 0.0
        epochs.append(i)
        eps_values.append(eps)
    return epochs, eps_values


# ── Build figure ─────────────────────────────────────────────────────────────


def build_figure(
    all_results: dict[tuple[str, bool, int], dict],
) -> go.Figure:
    """Build a 2×3 subplot figure."""
    subplot_titles = []
    for row_label in ["Memory (MB)", "Epochs/s"]:
        for se in SAMPLE_EVERY_VALUES:
            subplot_titles.append(f"{row_label} — sample_every={se}")

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    configs = [("cpu", True), ("cpu", False), ("mps", True), ("mps", False)]

    for col_idx, se in enumerate(SAMPLE_EVERY_VALUES, start=1):
        for dev, cached in configs:
            key = (dev, cached, se)
            result = all_results[key]
            label = LABELS[(dev, cached)]
            color = COLORS[(dev, cached)]

            # Only show legend once per trace (first column)
            show_legend = col_idx == 1

            # Row 1: Memory
            fig.add_trace(
                go.Scatter(
                    x=result["memory_epoch"],
                    y=result["memory_mb"],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=col_idx,
            )

            # Row 2: Epochs/s (rolling)
            ep_x, ep_y = rolling_epochs_per_sec(result["epoch_time_s"], ROLLING_WINDOW)
            fig.add_trace(
                go.Scatter(
                    x=ep_x,
                    y=ep_y,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    line=dict(color=color, width=2),
                ),
                row=2,
                col=col_idx,
            )

    # Axis labels
    for col_idx in range(1, 4):
        fig.update_xaxes(title_text="Epoch", row=2, col=col_idx)
        fig.update_xaxes(title_text="Epoch", row=1, col=col_idx)
    fig.update_yaxes(title_text="MB", row=1, col=1)
    fig.update_yaxes(title_text="Epochs/s", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f"sample_every benchmark — {N_GRID}×{N_GRID}={N_GRID**2} models, "
                f"HierarchicalSparse(features={N_FEATURES}, max_children={MAX_CHILDREN}, "
                f"depth_decay={DEPTH_DECAY}), hidden={N_HIDDEN}, "
                f"batch={BATCH_SIZE}, epochs={N_EPOCHS}"
            ),
            font=dict(size=16),
        ),
        height=700,
        width=1400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Grid: {N_GRID}x{N_GRID} = {N_GRID**2} models")
    print(f"features={N_FEATURES}, hidden={N_HIDDEN}, batch={BATCH_SIZE}")
    print(f"epochs={N_EPOCHS} (+ {WARMUP_EPOCHS} warmup)")
    print(f"sample_every values: {SAMPLE_EVERY_VALUES}")
    print()

    all_results: dict[tuple[str, bool, int], dict] = {}

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
                    f"[{run_idx}/{total_runs}] {dev.upper()} {cache_label}, sample_every={se}"
                )

                torch.manual_seed(SEED)
                grid = make_grid(dev, cached)

                result = profiled_fit(
                    grid,
                    n_epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    sample_every=se,
                    device=dev,
                    warmup_epochs=WARMUP_EPOCHS,
                )
                all_results[(dev, cached, se)] = result

                avg_eps = 1.0 / (
                    sum(result["epoch_time_s"]) / len(result["epoch_time_s"])
                )
                print(f"  -> {avg_eps:.1f} epochs/s avg")

    print("\nBuilding figure...")
    fig = build_figure(all_results)

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_sample_every.html")
    fig.write_html(out_path)
    print(f"Saved to {out_path}")
