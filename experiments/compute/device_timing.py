# %%
"""
Timing experiment: distribution device × ae device combinations.

Tests all combinations of (cpu, mps) for distribution and ae across three
sync conditions:

  async   — track_losses=False, no hooks: all MPS work is queued and only
             flushed at the final synchronize(). Wall time reflects the MPS
             pipeline throughput, not per-step latency. This is the
             best-case (and misleading) scenario.

  per-step — track_losses=False, norm_hook at freq=1: forces an MPS→CPU
             transfer every epoch (matching what loss.item() does in the
             track_losses=True path). Wall time reflects actual per-step
             round-trip latency.

  track_losses — track_losses=True, no hooks: loss.item() called every
             epoch forces an MPS sync. This is what every real experiment
             does (verbose=True or loss curves).  Equivalent to per-step
             but via loss.item() rather than a hook.

The gap between 'async' and the other two columns reveals how much of the
apparent MPS speedup is an artefact of deferred execution vs real compute.

Usage:
    python experiments/compute/device_timing.py
"""

# %%
import time

import torch

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import PowerLawDigraph
from occhio.toy_model import ToyModel

# %% ── Config ────────────────────────────────────────────────────────────────────
N_FEATURES = 400
N_HIDDEN = 30
N_EPOCHS = 5_000
BATCH_SIZE = 512
N_RUNS = 3
SEED = 42
# %% ─────────────────────────────────────────────────────────────────────────────

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
else:
    print("MPS not available — only CPU will be tested.\n")

combinations = [(d, a) for d in DEVICES for a in DEVICES]


def make_tm(dist_device: str, ae_device: str) -> ToyModel:
    dist = PowerLawDigraph(N_FEATURES, device=dist_device)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, device=ae_device)
    return ToyModel(distribution=dist, ae=ae)


def _sync_if_mps(*devices: str) -> None:
    if "mps" in devices and torch.backends.mps.is_available():
        torch.mps.synchronize()


def norm_hook(hook_data: dict) -> tuple:
    """Pull feature norms to CPU — forces an MPS→CPU sync at call time."""
    return (hook_data["epoch"], hook_data["tm"].feature_norms.cpu())


def time_combination(
    dist_device: str,
    ae_device: str,
    track_losses: bool,
    hook_freq: int | None,
) -> float:
    """Return mean wall-clock seconds for N_RUNS training runs."""
    hooks = [] if hook_freq is None else [norm_hook]
    freq = hook_freq or 1
    times = []
    for _ in range(N_RUNS):
        torch.manual_seed(SEED)
        tm = make_tm(dist_device, ae_device)
        t0 = time.perf_counter()
        tm.fit(
            N_EPOCHS,
            batch_size=BATCH_SIZE,
            track_losses=track_losses,
            hooks=hooks,
            hook_freq=freq,
        )
        _sync_if_mps(dist_device, ae_device)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# Columns: (label, track_losses, hook_freq or None)
CONDITIONS: list[tuple[str, bool, int | None]] = [
    ("async", False, None),  # no per-step sync at all
    ("per-step", False, 1),  # norm_hook every epoch → sync every step
    ("track_losses", True, None),  # loss.item() every epoch → sync every step
]

# %% ── Run ───────────────────────────────────────────────────────────────────────
print(
    f"Config: n_features={N_FEATURES}, n_hidden={N_HIDDEN}, "
    f"n_epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, runs={N_RUNS}\n"
)

col_w = 15
cond_labels = [c[0] for c in CONDITIONS]
header = f"{'dist':<8} {'ae':<8}" + "".join(f"{lbl:>{col_w}}" for lbl in cond_labels)
print(header)
print("-" * len(header))

results: list[tuple[str, str, list[float]]] = []
for dist_dev, ae_dev in combinations:
    times: list[float] = []
    for label, track_losses, hook_freq in CONDITIONS:
        desc = f"dist={dist_dev}, ae={ae_dev}, cond={label}"
        print(f"  Running {desc} ...", flush=True)
        t = time_combination(dist_dev, ae_dev, track_losses, hook_freq)
        times.append(t)
    results.append((dist_dev, ae_dev, times))

print()
print(header)
print("-" * len(header))
for dist_dev, ae_dev, times in results:
    row = f"{dist_dev:<8} {ae_dev:<8}"
    row += "".join(f"{t:>{col_w}.3f}" for t in times)
    # Flag if async is notably faster than per-step (async speedup artefact)
    if times[0] < times[1] * 0.85:
        row += "  ← async gap!"
    print(row)

print("\nNote: 'async' < 'per-step' means the MPS speedup is partly an async")
print("      queuing artefact, not purely faster compute per step.")

# %%
