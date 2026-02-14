import math

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid


def plot_phase_change_multi(model_grid: ModelGrid, *, up_to: int, max_cols: int = 4):
    # Extract feature norms and interferences for all features at once
    all_norms = []
    all_interferences = []

    for row in model_grid.models:
        row_norms = []
        row_interferences = []
        for model in row:
            # Convert tensors to numpy arrays and extract up to the specified number of features
            feature_norms = model.feature_norms.cpu().numpy()[:up_to]
            feature_interferences = model.total_feature_interferences.cpu().numpy()[
                :up_to
            ]
            row_norms.append(feature_norms)
            row_interferences.append(feature_interferences)
        all_norms.append(row_norms)
        all_interferences.append(row_interferences)

    # Convert to numpy arrays with shape (grid_height, grid_width, num_features)
    norm = np.array(all_norms)
    interference = np.array(all_interferences)

    max_norm = norm.max()
    max_interference = interference.max()

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

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        # subplot_titles=[f"Phase Change [Feature {i}]" for i in range(up_to)] + ["Colormaddp"],
        specs=specs,
        vertical_spacing=0.1,  # Add more space between rows (default is 0.03)
        horizontal_spacing=0.2,  # Add more space between rows (default is 0.03)
    )

    for i in range(up_to):
        row = i // n_cols + 1
        col = i % n_cols + 1

        _add_model_phases_trace(
            model_grid,
            i,
            fig,
            norm[:, :, i],  # Extract data for feature i
            interference[:, :, i],  # Extract data for feature i
            col=col,
            row=row,
        )

    colormap_row = up_to // n_cols + 1
    colormap_col = up_to % n_cols + 1
    _add_colormap_trace(
        max_norm, max_interference, fig, col=colormap_col, row=colormap_row
    )

    return fig


def plot_phase_change(model_grid: ModelGrid, *, tracked_feature=1):
    norm = np.vectorize(lambda m: m.feature_norms[tracked_feature])(model_grid.models)
    interference = np.vectorize(
        lambda m: m.total_feature_interferences[tracked_feature]
    )(model_grid.models)

    max_norm = norm.max()
    max_interference = interference.max()

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        subplot_titles=(f"Phase Change [Feature {tracked_feature}]"),
    )

    _add_model_phases_trace(
        model_grid,
        tracked_feature,
        fig,
        norm,
        interference,
        col=1,
        row=1,
    )
    _add_colormap_trace(max_norm, max_interference, fig, col=2, row=1)

    return fig


def _get_phase_color(
    norm: float, interference: float, max_interference, max_norm
) -> NDArray[np.uint8]:
    # [12.02.26 | OliverSieweke] TODO: Add docstring explanation
    gray = 200
    norm_scaled = norm / max_norm
    interference_scaled = interference / max_interference
    r = np.clip(gray + (interference_scaled * 255 - gray) * norm_scaled, 0, 255)
    g = np.clip(gray + (0 - gray) * norm_scaled, 0, 255)
    b = np.clip(gray + ((1 - interference_scaled) * 255 - gray) * norm_scaled, 0, 255)

    return np.stack([r, g, b], axis=-1)


def _add_model_phases_trace(
    model_grid: ModelGrid, tracked_feature, fig, norm, interference, *, col, row
):

    phase_colors = _get_phase_color(
        norm, interference, max_interference=interference.max(), max_norm=norm.max()
    )

    metadata = np.stack(
        [
            norm,
            interference,
            model_grid.parameters_mesh[0],
            model_grid.parameters_mesh[1],
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Image(
            z=phase_colors,
            customdata=metadata,
            hovertemplate=f"Norm: %{{customdata[0]:.2f}}<br>Interference: %{{customdata[1]:.2f}}<br>{model_grid.x_axis.label}: %{{customdata[2]:.2f}}<br>{model_grid.y_axis.label}: %{{customdata[3]:.2f}}<br><extra></extra>",
        ),
        row=row,
        col=col,
    )

    x_axis_values = model_grid.x_axis.values
    x_tick_indices = [0, len(x_axis_values) // 2, len(x_axis_values) - 1]
    x_tick_labels = [f"{x_axis_values[i]:.1f}" for i in x_tick_indices]
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_indices,
        ticktext=x_tick_labels,
        title=dict(text=f"<b>{model_grid.x_axis.label}</b>", font=dict(size=5)),
        row=row,
        col=col,
    )

    y_axis_values = model_grid.y_axis.values
    y_tick_indices = [0, len(y_axis_values) // 2, len(y_axis_values) - 1]
    y_tick_labels = [f"{y_axis_values[i]:.2f}" for i in y_tick_indices]
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=y_tick_indices,
        ticktext=y_tick_labels,
        title=dict(text=f"<b>{model_grid.y_axis.label}</b>", font=dict(size=5)),
        row=row,
        col=col,
    )


def _add_colormap_trace(max_norm, max_interference, fig, *, col, row):
    COLORMAP_SIZE = 100
    interference_mesh, norm_mesh = np.meshgrid(
        np.linspace(0, max_interference, COLORMAP_SIZE),
        np.linspace(max_norm, 0, COLORMAP_SIZE),
    )
    colormap = _get_phase_color(
        norm_mesh,
        interference_mesh,
        max_interference=max_interference,
        max_norm=max_norm,
    )

    fig.add_trace(
        go.Image(
            z=colormap,
            customdata=np.stack([norm_mesh, interference_mesh], axis=-1),
            hovertemplate="Interference: %{customdata[1]:.2f}<br>Norm: %{customdata[0]:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, COLORMAP_SIZE - 1],
        ticktext=[0, f"{max_norm:.1f}"],
        side="top",
        row=row,
        col=col,
        title=dict(text="<b>Interference</b>", font=dict(size=10), standoff=5),
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, COLORMAP_SIZE - 1],
        ticktext=[f"{max_interference:.1f}", 0],
        side="right",
        row=row,
        col=col,
        title=dict(text="<b>Norm</b>", font=dict(size=8), standoff=5),
    )

    return fig
