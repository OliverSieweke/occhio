"""Regression utilities for scaling-law experiments.

OLS helpers fit log-linear models analytically; `fit_gd` handles nonlinear
models (e.g. power-law exponents that must be jointly optimised) via Adam.
"""

import numpy as np
import scipy.optimize
import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# OLS models
# ---------------------------------------------------------------------------


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


def fit_excess_only_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = b2·log(N-m) + b0 on compression-regime points.

    Only the excess (N-m) enters; m does not appear.
    Inversion is analytic: N_boundary(m) = m + exp((log(target_loss) - b0) / b2).
    The boundary excess N-m is constant across m under this model.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    excess = np.array([e["n"] - e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([np.log(excess), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    b2, b0 = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    n_bnd_excess = np.exp((np.log(target_loss) - b0) / b2)
    print(f"excess-only:    log(loss) = {b2:.3f}·log(N-m) + {b0:.3f}   R²={r2:.4f}")
    print(f"  => loss = {np.exp(b0):.4f} · (N-m)^{b2:.3f}")
    print(f"  => N_boundary(m) = m + {n_bnd_excess:.3f}  [constant excess]")
    return coeffs, lambda m_arr: np.asarray(m_arr, dtype=float) + n_bnd_excess


def fit_simple_excess_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = b1·log(m) + b2·log(N-m) + b0 on compression-regime points.

    Like excess_model but without the log(N) term.
    Inversion is analytic: N_boundary(m) = m + C · m^k where
    C = exp((log(L) - b0) / b2) and k = -b1/b2.
    Returns the coefficient array [b1, b2, b0] and a callable n_boundary(m).
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([np.log(m), np.log(n - m), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    b1, b2, b0 = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    log_target = np.log(target_loss)
    C = np.exp((log_target - b0) / b2)
    k = -b1 / b2
    print(
        f"simple excess:  log(loss) = {b1:.3f}·log(m) + {b2:.3f}·log(N-m) + {b0:.3f}   R²={r2:.4f}"
    )
    print(f"  => loss = {np.exp(b0):.4f} · m^{b1:.3f} · (N-m)^{b2:.3f}")
    print(f"  => N_boundary(m) = m + {C:.3f} · m^{k:.3f}")

    def n_boundary(m_arr):
        m_arr = np.asarray(m_arr, dtype=float)
        return m_arr + C * m_arr**k

    return coeffs, n_boundary


def fit_linear_m_model(evals: list[dict], target_loss: float):
    """Fit log(loss) = a·m + b·log(N) + c on compression-regime points.

    m enters linearly (not log-transformed).
    Returns the coefficient array [a, b, c] and a callable n_boundary(m)
    that gives the predicted max N at target_loss for a given m.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)
    X = np.column_stack([m, np.log(n), np.ones(len(pts))])
    coeffs, *_ = np.linalg.lstsq(X, np.log(loss), rcond=None)
    a, b, c = coeffs
    r2 = _r2(np.log(loss), X @ coeffs)
    print(
        f"linear-m model: log(loss) = {a:.3f}·m + {b:.3f}·log(N) + {c:.3f}   R²={r2:.4f}"
    )
    print(f"  => N_boundary(m) = exp((log(L) - {a:.3f}·m - {c:.3f}) / {b:.3f})")
    log_target = np.log(target_loss)
    return coeffs, lambda m_arr: np.exp((log_target - a * np.asarray(m_arr) - c) / b)


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


# ---------------------------------------------------------------------------
# Gradient-descent fitter (nonlinear models)
# ---------------------------------------------------------------------------


def fit_gd(
    data: dict[str, np.ndarray],
    model_fn,
    init_params: dict[str, float],
    y_key: str = "y",
    n_steps: int = 25_000,
    lr: float = 3e-3,
    verbose: bool = True,
) -> tuple[dict[str, float], float]:
    """Fit a model using gradient descent (Adam) via torch.

    Parameters
    ----------
    data:
        Dict of named numpy arrays. The target variable is identified by `y_key`.
    model_fn:
        ``(params, data) -> y_pred`` where both are dicts of torch tensors.
        The function must be differentiable w.r.t. ``params`` values.
    init_params:
        Initial parameter values as ``{name: float}``.
    y_key:
        Key in ``data`` for the target (dependent) variable.
    n_steps:
        Number of Adam steps.
    lr:
        Adam learning rate.
    verbose:
        Print fitted parameters and R² when done.

    Returns
    -------
    params : dict[str, float]
        Fitted parameter values.
    r2 : float
        R² of the fit on the (possibly transformed) target.
    """
    data_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in data.items()}
    y = data_t.pop(y_key)

    params = {
        k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for k, v in init_params.items()
    }
    optimizer = torch.optim.Adam(list(params.values()), lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        y_pred = model_fn(params, data_t)
        loss = torch.mean((y - y_pred) ** 2)
        loss.backward()
        optimizer.step()

    fitted = {k: v.item() for k, v in params.items()}
    with torch.no_grad():
        y_pred_np = model_fn(params, data_t).numpy()
    r2 = _r2(y.numpy(), y_pred_np)

    if verbose:
        for k, v in fitted.items():
            print(f"  {k} = {v:.4f}")
        print(f"  R² = {r2:.4f}")

    return fitted, r2


def fit_power_m_model(evals: list[dict], target_loss: float, **gd_kwargs):
    """Fit log(loss) = a·log(N) + b·m^κ + c using gradient descent.

    ``m^κ`` is a nonlinear term; κ is jointly optimised with the linear
    coefficients a, b, c.  The boundary N is solved analytically from the
    fitted parameters.

    Returns the fitted parameter dict and a callable ``n_boundary(m)``.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)

    def model_fn(params, data):
        return (
            params["a"] * data["log_N"]
            + params["b"] * data["m"] ** params["kappa"]
            + params["c"]
        )

    data = {"log_N": np.log(n), "m": m, "y": np.log(loss)}
    init_params = {"a": 1.0, "b": -0.1, "kappa": 1.0, "c": 0.0}

    print("power-m model:  log(loss) = a·log(N) + b·m^κ + c")
    fitted, r2 = fit_gd(data, model_fn, init_params, **gd_kwargs)
    a, b, kappa, c = fitted["a"], fitted["b"], fitted["kappa"], fitted["c"]
    print(
        f"  => log(loss) = {a:.3f}·log(N) + {b:.3f}·m^{kappa:.3f} + {c:.3f}   R²={r2:.4f}"
    )
    print(
        f"  => N_boundary(m) = exp((log(L) - {b:.3f}·m^{kappa:.3f} - {c:.3f}) / {a:.3f})"
    )

    log_target = np.log(target_loss)

    def n_boundary(m_arr):
        m_arr = np.asarray(m_arr, dtype=float)
        return np.exp((log_target - b * m_arr**kappa - c) / a)

    return fitted, n_boundary


def fit_ratio_power_model(evals: list[dict], target_loss: float, **gd_kwargs):
    """Fit log(loss) = C·(N/m)^κ + b using gradient descent.

    The ratio N/m captures the compression factor; κ and C are jointly
    optimised.  The boundary N is solved analytically:
        N_boundary(m) = m · ((log(L) - b) / C)^(1/κ)

    Returns the fitted parameter dict and a callable ``n_boundary(m)``.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)

    def model_fn(params, data):
        return params["C"] * (data["ratio"]) ** params["kappa"] + params["b"]

    data = {"ratio": n / m, "y": np.log(loss)}
    init_params = {"C": 1.0, "kappa": 1.0, "b": 0.0}

    print("ratio-power model: log(loss) = C·(N/m)^κ + b")
    fitted, r2 = fit_gd(data, model_fn, init_params, **gd_kwargs)
    C, kappa, b = fitted["C"], fitted["kappa"], fitted["b"]
    print(f"  => log(loss) = {C:.3f}·(N/m)^{kappa:.3f} + {b:.3f}   R²={r2:.4f}")
    print(f"  => N_boundary(m) = m · ((log(L) - {b:.3f}) / {C:.3f})^(1/{kappa:.3f})")

    log_target = np.log(target_loss)

    def n_boundary(m_arr):
        m_arr = np.asarray(m_arr, dtype=float)
        return m_arr * ((log_target - b) / C) ** (1 / kappa)

    return fitted, n_boundary


def fit_power_excess_model(evals: list[dict], target_loss: float, **gd_kwargs):
    """Fit log(loss) = a + b·m^c + d·log(N-m) using gradient descent.

    m^c is nonlinear; c is jointly optimised with the linear coefficients.
    The boundary N is solved analytically:
        N_boundary(m) = m + exp((log(L) - a - b·m^c) / d)

    Returns the fitted parameter dict and a callable ``n_boundary(m)``.
    """
    pts = [e for e in evals if e["n"] > e["m"] and e["loss"] > 0]
    n = np.array([e["n"] for e in pts], dtype=float)
    m = np.array([e["m"] for e in pts], dtype=float)
    loss = np.array([e["loss"] for e in pts], dtype=float)

    def model_fn(params, data):
        return (
            params["a"]
            + params["b"] * data["m"] ** params["c"]
            + params["d"] * data["log_excess"]
        )

    data = {"m": m, "log_excess": np.log(n - m), "y": np.log(loss)}
    init_params = {"a": 6.0, "b": -8.1, "c": 0.15, "d": 1.0}

    print("power-excess model: log(loss) = a + b·m^c + d·log(N-m)")
    fitted, r2 = fit_gd(data, model_fn, init_params, **gd_kwargs)
    a, b, c, d = fitted["a"], fitted["b"], fitted["c"], fitted["d"]
    print(
        f"  => log(loss) = {a:.3f} + {b:.3f}·m^{c:.3f} + {d:.3f}·log(N-m)   R²={r2:.4f}"
    )
    print(
        f"  => N_boundary(m) = m + exp((log(L) - {a:.3f} - {b:.3f}·m^{c:.3f}) / {d:.3f})"
    )

    log_target = np.log(target_loss)

    def n_boundary(m_arr):
        m_arr = np.asarray(m_arr, dtype=float)
        return m_arr + np.exp((log_target - a - b * m_arr**c) / d)

    return fitted, n_boundary
