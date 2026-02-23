import math

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid


def _compute_max_interference(model_grid: ModelGrid, features: list[int]) -> float:
    """Return ceil(max interference) across all models for the given features.

    Always returns at least 1.0 so the colormap range is never degenerate.
    """
    max_val = 0.0
    for feat in features:
        arr = np.vectorize(
            lambda m: m.total_feature_interferences[feat].cpu().item()
        )(model_grid.models)
        feat_max = float(arr.max())
        if feat_max > max_val:
            max_val = feat_max
    return float(max(math.ceil(max_val), 1))


def plot_phase_change_multi(
    model_grid: ModelGrid,
    *,
    up_to: int,
    max_cols: int = 4,
    importance_base_axis: int | None = None,
):
    """Plot phase-change heatmaps for features 0..up_to-1 plus a shared colormap.

    Args:
        model_grid: Trained 2-D ModelGrid.
        up_to: Number of features to plot (0 through up_to-1).
        max_cols: Maximum subplot columns.
        importance_base_axis: If 0 or 1, that axis contains the importance *base*
            parameter (i.e. ``importance = base ** feature_index``).  Tick labels
            for that axis are raised to the corresponding feature-index power so
            they show the actual per-feature importance rather than the shared base.
    """
    total_items = up_to + 1  # features + colormap
    n_cols = min(total_items, max_cols)
    n_rows = math.ceil(total_items / n_cols)

    specs = []
    for r in range(n_rows):
        row_specs = []
        for c in range(n_cols):
            if r * n_cols + c < total_items:
                row_specs.append({})
            else:
                row_specs.append(None)
        specs.append(row_specs)

    # Compute a single shared max_interference so all subplots use the same scale.
    max_interference = _compute_max_interference(model_grid, list(range(up_to)))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
    )

    for i in range(up_to):
        row = i // n_cols + 1
        col = i % n_cols + 1
        x_exponent = i if importance_base_axis is not None else None
        _add_model_phases_trace(
            model_grid, i, fig, col=col, row=row,
            max_interference=max_interference,
            importance_base_axis=importance_base_axis,
            x_exponent=x_exponent,
        )

    colormap_row = up_to // n_cols + 1
    colormap_col = up_to % n_cols + 1
    _add_colormap_trace(fig, col=colormap_col, row=colormap_row, max_interference=max_interference)

    return fig


def plot_phase_change(
    model_grid: ModelGrid,
    *,
    tracked_feature: int = 1,
    importance_base_axis: int | None = None,
):
    """Plot a single phase-change heatmap for ``tracked_feature``.

    Args:
        model_grid: Trained 2-D ModelGrid.
        tracked_feature: Which feature index to visualise.
        importance_base_axis: If 0 or 1, that axis contains the importance *base*
            parameter.  Tick labels are raised to ``tracked_feature`` power so they
            show the actual importance for this feature rather than the shared base.
    """
    if len(model_grid.shape) != 2:
        raise ValueError(
            f"plot_phase_change requires a 2-dimensional ModelGrid, "
            f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
        )

    max_interference = _compute_max_interference(model_grid, [tracked_feature])
    x_exponent = tracked_feature if importance_base_axis is not None else None

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        subplot_titles=(f"Phase Change [Feature {tracked_feature}]", "Colormap"),
    )

    _add_model_phases_trace(
        model_grid, tracked_feature, fig, col=1, row=1,
        max_interference=max_interference,
        importance_base_axis=importance_base_axis,
        x_exponent=x_exponent,
    )
    _add_colormap_trace(fig, col=2, row=1, max_interference=max_interference)

    return fig


def _get_phase_color(
    norm: NDArray,
    interference: NDArray,
    max_interference: float = 1.0,
) -> NDArray[np.uint8]:
    """Map (norm, interference) to an RGB colour in the grey-blue-red scheme.

    ``norm`` controls how saturated the colour is (0 = grey, 1 = fully saturated).
    ``interference`` in [0, max_interference] controls the hue:
      - 0            → blue  (feature is represented, no interference)
      - max_interference → red   (feature is entirely reconstructed via interference)
      - values between 1 and max_interference are intermediate reds, with 1 always
        mapping to the same position on the gradient regardless of max_interference.

    A ``max_interference > 1`` stretches the colour range so that pure red is
    reserved for ``max_interference`` rather than 1.
    """
    gray = 200
    # Scale interference to [0, 1] relative to the ceiling so pure red = max_interference.
    norm_interf = interference / max_interference
    r = np.clip(gray + (norm_interf * 255 - gray) * norm, 0, 255)
    g = np.clip(gray + (0 - gray) * norm, 0, 255)
    b = np.clip(gray + ((1 - norm_interf) * 255 - gray) * norm, 0, 255)

    return np.stack([r, g, b], axis=-1)


def _add_model_phases_trace(
    model_grid: ModelGrid,
    tracked_feature: int,
    fig,
    *,
    col: int,
    row: int,
    max_interference: float = 1.0,
    importance_base_axis: int | None = None,
    x_exponent: int | None = None,
) -> None:
    norm = np.vectorize(lambda m: m.feature_norms[tracked_feature].cpu().item())(
        model_grid.models
    )
    interference = np.vectorize(
        lambda m: m.total_feature_interferences[tracked_feature].cpu().item()
    )(model_grid.models)

    phase_colors = _get_phase_color(norm, interference, max_interference)

    metadata = np.stack(
        [
            norm,
            interference,
            model_grid.parameters_mesh[0],
            model_grid.parameters_mesh[1],
        ],
        axis=-1,
    )

    # Transpose so axes[0] maps to x (columns) and axes[1] maps to y (rows).
    # Flip vertically so row 0 (top) corresponds to the largest y-axis value.
    phase_colors = np.swapaxes(phase_colors, 0, 1)[::-1]
    metadata = np.swapaxes(metadata, 0, 1)[::-1]

    fig.add_trace(
        go.Image(
            z=phase_colors,
            customdata=metadata,
            hovertemplate=(
                f"Norm: %{{customdata[0]:.2f}}<br>"
                f"Interference: %{{customdata[1]:.2f}}<br>"
                f"{model_grid.axes[0].label}: %{{customdata[2]:.2f}}<br>"
                f"{model_grid.axes[1].label}: %{{customdata[3]:.2f}}<br>"
                f"<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    # --- X axis ---
    x_axis_values = model_grid.axes[0].values
    x_tick_indices = [0, len(x_axis_values) // 2, len(x_axis_values) - 1]
    if importance_base_axis == 0 and x_exponent is not None:
        x_tick_labels = [
            f"{float(x_axis_values[i]) ** x_exponent:.3f}" for i in x_tick_indices
        ]
        x_title = f"<b>{model_grid.axes[0].label}^{x_exponent}</b>"
    else:
        x_tick_labels = [f"{float(x_axis_values[i]):.3f}" for i in x_tick_indices]
        x_title = f"<b>{model_grid.axes[0].label}</b>"

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_indices,
        ticktext=x_tick_labels,
        title=dict(text=x_title, font=dict(size=10)),
        row=row,
        col=col,
    )

    # --- Y axis ---
    y_axis_values = model_grid.axes[1].values
    y_tick_indices = [0, len(y_axis_values) // 2, len(y_axis_values) - 1]
    # Reverse labels since go.Image has row 0 at the top.
    if importance_base_axis == 1 and x_exponent is not None:
        y_tick_labels = [
            f"{float(y_axis_values[i]) ** x_exponent:.3f}" for i in reversed(y_tick_indices)
        ]
        y_title = f"<b>{model_grid.axes[1].label}^{x_exponent}</b>"
    else:
        y_tick_labels = [f"{float(y_axis_values[i]):.3f}" for i in reversed(y_tick_indices)]
        y_title = f"<b>{model_grid.axes[1].label}</b>"

    fig.update_yaxes(
        tickmode="array",
        tickvals=y_tick_indices,
        ticktext=y_tick_labels,
        title=dict(text=y_title, font=dict(size=10)),
        row=row,
        col=col,
    )


def _add_colormap_trace(fig, *, col: int, row: int, max_interference: float = 1.0):
    COLORMAP_SIZE = 100

    # Interference axis spans [0, max_interference]; norm axis spans [0, 1].
    interference_mesh, norm_mesh = np.meshgrid(
        np.linspace(0, max_interference, COLORMAP_SIZE),
        np.linspace(1, 0, COLORMAP_SIZE),
    )
    colormap = _get_phase_color(norm_mesh, interference_mesh, max_interference)

    fig.add_trace(
        go.Image(
            z=colormap,
            customdata=np.stack([norm_mesh, interference_mesh], axis=-1),
            hovertemplate="Interference: %{customdata[1]:.2f}<br>Norm: %{customdata[0]:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # X-axis ticks.  When max_interference > 1 we add an explicit tick at 1 so
    # the split point is labelled; otherwise keep the original "≥1" label.
    if max_interference > 1.0:
        split_pos = (1.0 / max_interference) * (COLORMAP_SIZE - 1)
        x_tick_vals = [0, split_pos, COLORMAP_SIZE - 1]
        x_tick_texts = ["0", "1", str(int(max_interference))]
    else:
        x_tick_vals = [0, COLORMAP_SIZE - 1]
        x_tick_texts = ["0", "≥1"]

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_vals,
        ticktext=x_tick_texts,
        side="top",
        row=row,
        col=col,
        title=dict(text="<b>Interference</b>", font=dict(size=10), standoff=5),
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, COLORMAP_SIZE - 1],
        ticktext=["≥1", "0"],
        side="right",
        row=row,
        col=col,
        title=dict(text="<b>Norm</b>", font=dict(size=8), standoff=5),
    )

    # Draw a dotted white split line at interference = 1 when max > 1 so the
    # boundary between "in superposition" and "extreme interference" is visible.
    if max_interference > 1.0:
        split_x = (1.0 / max_interference) * (COLORMAP_SIZE - 1)
        fig.add_shape(
            type="line",
            x0=split_x,
            x1=split_x,
            y0=-0.5,
            y1=COLORMAP_SIZE - 0.5,
            line=dict(color="white", width=1.5, dash="dot"),
            row=row,
            col=col,
        )

    return fig
