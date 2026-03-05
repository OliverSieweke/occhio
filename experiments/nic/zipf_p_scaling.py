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
import numpy as np
import scipy.optimize
import torch
import plotly.graph_objects as go
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

# %% --- Regression helpers ---
"""
Three models (compression regime N > m, loss > 0 unless noted):

  loss_model:     log(loss) = α·log(N) + γ·log(m) + β
                  => N_boundary(m) = (L/exp(β))^(1/α) · m^(-γ/α)

  boundary_model: log(N_best) = δ·log(m) + ε   (boundary points only)
                  => N_best = exp(ε) · m^δ

  excess_model:   log(loss) = b1·log(m) + b2·log(N-m) + b3·log(N) + b0
                  => N_boundary(m) solved numerically via brentq

  simple_excess_model: log(loss) = b1·log(m) + b2·log(N-m) + b0
                  => N_boundary(m) solved numerically via brentq  [no log(N) term]
"""


def _r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def fit_loss_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = α·log(N) + γ·log(m) + β on compression-regime points.

    Returns the coefficient array [α, γ, β] and a callable n_boundary(m)
    that gives the predicted max N at target_loss for a given m.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([np.log(n), np.log(m), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    alpha, gamma, beta = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    print(
        f"loss model:     log(loss) = {alpha:.3f}·log(N) + {gamma:.3f}·log(m) + {beta:.3f}   R²={r2:.4f}"
    )
    print(f"  => loss = {np.exp(beta):.4f} · N^{alpha:.3f} · m^{gamma:.3f}")
    n_bnd_exp = -gamma / alpha
    n_bnd_coef = (target_loss / np.exp(beta)) ** (1 / alpha)
    print(f"  => N_boundary(m) = {n_bnd_coef:.3f} · m^{n_bnd_exp:.3f}")
    return coeffs, lambda m_arr: n_bnd_coef * np.asarray(m_arr) ** n_bnd_exp


def fit_boundary_model(results: dict[int, int | None]):
    """Fit log(N_best) = δ·log(m) + ε on boundary points only.

    Returns the coefficient array [δ, ε] and a callable n_boundary(m).
    """
    m = np.array(list(results.keys()), dtype=float)
    n = np.array(list(results.values()), dtype=float)
    X = np.column_stack([np.log(m), np.ones(len(m))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(n), rcond=None)
    delta, eps = coeffs
    r2 = _r2(np.log(n), X @ coeffs)
    print(f"boundary model: log(N) = {delta:.3f}·log(m) + {eps:.3f}")
    print(f"  => N_best = {np.exp(eps):.3f} · m^{delta:.3f}   R²={r2:.4f}")
    return coeffs, lambda m_arr: np.exp(eps) * np.asarray(m_arr) ** delta


def fit_simple_excess_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = b1·log(m) + b2·log(N-m) + b0 on compression-regime points.

    Like excess_model but without the log(N) term.
    Returns the coefficient array [b1, b2, b0] and a callable n_boundary(m)
    that gives the predicted max N at target_loss (solved numerically).
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([np.log(m), np.log(n - m), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    b1, b2, b0 = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    print(
        f"simple excess:  log(loss) = {b1:.3f}·log(m) + {b2:.3f}·log(N-m) + {b0:.3f}   R²={r2:.4f}"
    )
    print(f"  => loss = {np.exp(b0):.4f} · m^{b1:.3f} · (N-m)^{b2:.3f}")

    def n_boundary(m_arr: np.ndarray) -> np.ndarray:
        m_arr = np.asarray(m_arr, dtype=float)
        result = np.empty_like(m_arr)
        log_target = np.log(target_loss)
        for i, mi in enumerate(m_arr.flat):

            def residual(n_val):
                return b1 * np.log(mi) + b2 * np.log(n_val - mi) + b0 - log_target

            result.flat[i] = scipy.optimize.brentq(residual, mi + 1e-6, mi * 1e4)
        return result

    return coeffs, n_boundary


def fit_excess_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = b1·log(m) + b2·log(N-m) + b3·log(N) + b0 on compression-regime points.

    N-m is the number of "excess" features beyond perfect reconstruction capacity.
    Returns the coefficient array [b1, b2, b3, b0] and a callable n_boundary(m)
    that gives the predicted max N at target_loss for a given m (solved numerically).
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([np.log(m), np.log(n - m), np.log(n), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    b1, b2, b3, b0 = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    print(
        f"excess model:   log(loss) = {b1:.3f}·log(m) + {b2:.3f}·log(N-m) + {b3:.3f}·log(N) + {b0:.3f}   R²={r2:.4f}"
    )
    print(f"  => loss = {np.exp(b0):.4f} · m^{b1:.3f} · (N-m)^{b2:.3f} · N^{b3:.3f}")

    def n_boundary(m_arr: np.ndarray) -> np.ndarray:
        m_arr = np.asarray(m_arr, dtype=float)
        result = np.empty_like(m_arr)
        log_target = np.log(target_loss)
        for i, mi in enumerate(m_arr.flat):

            def residual(n_val):
                return (
                    b1 * np.log(mi)
                    + b2 * np.log(n_val - mi)
                    + b3 * np.log(n_val)
                    + b0
                    - log_target
                )

            result.flat[i] = scipy.optimize.brentq(residual, mi + 1e-6, mi * 1e4)
        return result

    return coeffs, n_boundary


# %%
loss_coeffs, n_bnd_loss = fit_loss_model(all_evals, target_loss)
print()
bnd_coeffs, n_bnd_boundary = fit_boundary_model(results)
print()
excess_coeffs, n_bnd_excess = fit_excess_model(all_evals, target_loss)
print()
simple_excess_coeffs, n_bnd_simple_excess = fit_simple_excess_model(
    all_evals, target_loss
)
print()

# Overlay all three fits on Plot 2
m_range = np.geomspace(min(M_VALUES), max(M_VALUES), 200)
alpha, gamma, _ = loss_coeffs
delta, _ = bnd_coeffs
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_loss(m_range),
        mode="lines",
        line=dict(dash="dash", color="black"),
        name=f"loss model: N ∝ m^{-gamma / alpha:.2f}",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_boundary(m_range),
        mode="lines",
        line=dict(dash="dot", color="gray"),
        name=f"boundary model: N ∝ m^{delta:.2f}",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_excess(m_range),
        mode="lines",
        line=dict(dash="dashdot", color="blue"),
        name="excess model: N = m + f(m)",
    )
)
fig2.add_trace(
    go.Scatter(
        x=m_range,
        y=n_bnd_simple_excess(m_range),
        mode="lines",
        line=dict(dash="longdash", color="purple"),
        name="simple excess: no log(N) term",
    )
)
fig2.show()

# %%
