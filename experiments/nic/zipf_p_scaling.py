"""
In this document I want to check how large N_FEATURES can get when we fix
p and increase m (hidden dimension), and fix the target loss to some number.

The way to set this up is we start with p = 0.1, SparseUniform.
We set m = 2 and N = 5. We train for 15k epochs.
We then pass in 2**12 samples to estimate the loss.
This will be our target loss going forward.

So we move to m=3. Now we need to train multiple models with different N.
We want to find the smallest N which doesn't go across the loss threshold.
"""

# %%
import importlib, reg_utils

importlib.reload(reg_utils)


# %%
import numpy as np
import torch
import plotly.graph_objects as go
from reg_utils import (
    fit_boundary_model,
    fit_excess_model,
    fit_simple_excess_model,
    fit_power_excess_model,
)
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
N_EPOCHS = 20_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**13

M_VALUES = [3, 4, 6, 9, 13, 19, 28, 42, 63, 90, 100, 130]


def create_model(
    n_features: int, n_hidden: int, params: dict | None = None
) -> ToyModel:
    gen = torch.Generator(DEVICE).manual_seed(23)
    dist = SparseUniform(
        n_features,
        [1 / (i + 1) for i in range(n_features)],
        device=DEVICE,
        generator=gen,
    )
    ae = TiedLinearRelu(n_features, n_hidden, device=DEVICE, generator=gen)
    return ToyModel(
        distribution=dist,
        ae=ae,
        device=DEVICE,
    )


def estimate_loss(tm: ToyModel) -> float:
    with torch.no_grad():
        x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
        x_hat = tm.ae(x)[0]
        return torch.mean(torch.sum((x - x_hat) ** 2, dim=-1)).item()


# %% --- Step 1: Baseline (m=2, N=5) ---
print("Training baseline: m=2, N=5")
baseline = create_model(n_features=5, n_hidden=2)
baseline.fit(N_EPOCHS, batch_size=BATCH_SIZE)
target_loss = estimate_loss(baseline)
print(f"Target loss: {target_loss:.6f}")

# %% --- Step 2: Search N for each m ----
# Strategy: start from previous m's best N, double until we exceed target loss
# to bracket the answer, then binary search within that bracket.


def train_and_eval(n: int, m: int) -> float:
    tm = create_model(n_features=n, n_hidden=m)
    tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, track_losses=False)
    loss = estimate_loss(tm)
    print(f"  N={n:3d}  loss={loss:.6f}  {'OK' if loss <= target_loss else 'EXCEEDED'}")
    return loss


results: dict[int, int | None] = {}
all_evals: list[dict] = []  # records every (m, n, loss) evaluated
prev_best_n = 5  # baseline N; larger m can only do at least as well

# %%
for m in M_VALUES:
    print(f"\nm={m}: searching N...")

    def train_and_eval_recording(n: int, m: int = m) -> float:
        loss = train_and_eval(n, m)
        all_evals.append({"m": m, "n": n, "loss": loss})
        return loss

    # Phase 1: exponential expansion to find an upper bound
    lo = max(prev_best_n, m + 1)
    hi = lo
    while train_and_eval_recording(hi) <= target_loss:
        lo = hi
        hi = int(1.5 * hi)

    # Phase 2: binary search in [lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if train_and_eval_recording(mid) <= target_loss:
            lo = mid
        else:
            hi = mid

    best_n = lo
    results[m] = best_n
    prev_best_n = best_n

# %%
print("\n=== Results ===")
print(f"Target loss (m=2, N=5): {target_loss:.6f}")
print(f"{'m':>4}  {'max N':>6}")
for m, max_n in results.items():
    print(f"{m:>4}  {str(max_n):>6}")

# %% --- Plot: loss landscape across m and N ---
fig = go.Figure()

for m in M_VALUES:
    evals = sorted([e for e in all_evals if e["m"] == m], key=lambda e: e["n"])
    ns = [e["n"] for e in evals]
    losses = [e["loss"] for e in evals]
    ok = [loss <= target_loss for loss in losses]

    fig.add_trace(
        go.Scatter(
            x=ns,
            y=losses,
            mode="lines+markers",
            name=f"m={m}",
            marker=dict(
                symbol=["circle" if o else "x" for o in ok],
                size=10,
            ),
        )
    )

fig.add_hline(
    y=target_loss,
    line_dash="dash",
    line_color="black",
    annotation_text="target loss",
    annotation_position="bottom right",
)

# Mark the best N per m
for m, best_n in results.items():
    best_eval = next((e for e in all_evals if e["m"] == m and e["n"] == best_n), None)
    if best_eval:
        fig.add_trace(
            go.Scatter(
                x=[best_n],
                y=[best_eval["loss"]],
                mode="markers",
                marker=dict(symbol="star", size=14, color="black"),
                showlegend=False,
                hovertemplate=f"m={m}, best N={best_n}<extra></extra>",
            )
        )

fig.update_layout(
    title="Loss landscape: N features vs loss, by hidden dim m",
    xaxis_title="N (features)",
    yaxis_title="Loss",
    yaxis_type="log",
    xaxis_type="log",
    legend_title="Hidden dim m",
)
fig.show()

# %% --- Plot 2: m on x-axis, N on y-axis ---
fig2 = go.Figure()

ok_ms = [e["m"] for e in all_evals if e["loss"] <= target_loss]
ok_ns = [e["n"] for e in all_evals if e["loss"] <= target_loss]
bad_ms = [e["m"] for e in all_evals if e["loss"] > target_loss]
bad_ns = [e["n"] for e in all_evals if e["loss"] > target_loss]

fig2.add_trace(
    go.Scatter(
        x=ok_ms,
        y=ok_ns,
        mode="markers",
        name="loss ≤ target",
        marker=dict(symbol="circle", size=10),
    )
)
fig2.add_trace(
    go.Scatter(
        x=bad_ms,
        y=bad_ns,
        mode="markers",
        name="loss > target",
        marker=dict(symbol="x", size=10),
    )
)

# Star for the best N per m (closest to loss boundary)
fig2.add_trace(
    go.Scatter(
        x=list(results.keys()),
        y=list(results.values()),
        mode="markers",
        marker=dict(symbol="star", size=14, color="black"),
        name="best N (boundary)",
        hovertemplate=[f"m={m}, best N={n}<extra></extra>" for m, n in results.items()],
    )
)

fig2.update_layout(
    title="Max N features by hidden dim m",
    xaxis_title="m (hidden dim)",
    yaxis_title="N (features)",
    xaxis_type="log",
    yaxis_type="log",
    legend_title="Status",
)
fig2.show()

# %%
bnd_coeffs, n_bnd_boundary = fit_boundary_model(results)
print()
excess_coeffs, n_bnd_excess = fit_excess_model(all_evals, target_loss)
print()
simple_excess_coeffs, n_bnd_simple_excess = fit_simple_excess_model(
    all_evals, target_loss
)
print()
power_excess_fitted, n_bnd_power_excess = fit_power_excess_model(all_evals, target_loss)
_bnd_m = np.array(list(results.keys()), dtype=float)
_bnd_n = np.array(list(results.values()), dtype=float)
_bnd_n_pred = n_bnd_power_excess(_bnd_m)
_ss_res = np.sum((_bnd_n - _bnd_n_pred) ** 2)
_ss_tot = np.sum((_bnd_n - _bnd_n.mean()) ** 2)
print(f"  boundary R² (predicted vs actual best-N): {1 - _ss_res / _ss_tot:.4f}")
print()

# Overlay all fits on Plot 2
m_range = np.geomspace(min(M_VALUES), max(M_VALUES), 200)
delta, _ = bnd_coeffs
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_boundary(m_range),
        mode="lines",
        line=dict(dash="dot", color="gray"),
        name=f"boundary model",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_excess(m_range),
        mode="lines",
        line=dict(dash="dashdot", color="blue"),
        name="excess model",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_simple_excess(m_range),
        mode="lines",
        line=dict(dash="longdash", color="purple"),
        name="simple excess",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_power_excess(m_range),
        mode="lines",
        line=dict(dash="dashdot", color="teal"),
        name=f"power-excess model",
    )
)
fig2.show()

# %%
