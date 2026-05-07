"""Phase change visualization ported to the v2 SinglePlot framework.

``PhaseChangePlot`` renders the 2D phase-change heatmap showing how a
feature's norm and interference vary across a 2-dimensional parameter grid.
Because it needs the full 2D grid to compute the image, it declares
``n_render_axes = 2``.

``plot_phase_change`` and ``plot_phase_change_multi`` are kept as standalone
convenience functions that delegate to the legacy implementation, since their
multi-grid and animated variants are complex and don't map cleanly onto the
SinglePlot orchestrator.
"""

import math

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid, TrainingAxis
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


# ---------------------------------------------------------------------------
# Colour helper (shared by SinglePlot and standalone functions)
# ---------------------------------------------------------------------------


def _get_phase_color(norm: float, interference: float) -> NDArray[np.uint8]:
    gray = 200
    r = np.clip(gray + (interference * 255 - gray) * norm, 0, 255)
    g = np.clip(gray + (0 - gray) * norm, 0, 255)
    b = np.clip(gray + ((1 - interference) * 255 - gray) * norm, 0, 255)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# SinglePlot wrapper
# ---------------------------------------------------------------------------


class PhaseChangePlot(SinglePlot):
    """2D heatmap of feature norm vs. interference across a parameter grid.

    Requires a 2D ModelGrid (``n_render_axes = 2``).
    """

    n_render_axes = 2

    def __init__(self, tracked_feature: int = 1, *, show_colormap: bool = True):
        self.tracked_feature = tracked_feature
        self.show_colormap = show_colormap

    def render(self, fig: FigureProxy, models: ModelGrid) -> None:
        tracked_feature = self.tracked_feature

        norm = np.vectorize(lambda m: m.feature_norms[tracked_feature].cpu().item())(
            models.models
        )
        interference = np.vectorize(
            lambda m: m.total_feature_interferences[tracked_feature].cpu().item()
        )(models.models)

        max_interference = interference.max()
        interference_normalized = (
            interference / max_interference if max_interference > 0 else interference
        )
        phase_colors = _get_phase_color(norm, interference_normalized)

        metadata = np.stack(
            [
                norm,
                interference,
                models.parameters_mesh[0],
                models.parameters_mesh[1],
            ],
            axis=-1,
        )

        # Transpose so axes[0] -> x (columns) and axes[1] -> y (rows), flip vertically
        phase_colors = np.swapaxes(phase_colors, 0, 1)[::-1]
        metadata = np.swapaxes(metadata, 0, 1)[::-1]

        fig.add_trace(
            go.Image(
                z=phase_colors,
                customdata=metadata,
                hovertemplate=(
                    f"Norm: %{{customdata[0]:.2f}}<br>"
                    f"Interference: %{{customdata[1]:.2f}}<br>"
                    f"{models.axes[0].label}: %{{customdata[2]:.2f}}<br>"
                    f"{models.axes[1].label}: %{{customdata[3]:.2f}}<br>"
                    f"<extra></extra>"
                ),
            )
        )

        # X-axis ticks
        x_vals = models.axes[0].values
        x_tick_indices = [0, len(x_vals) // 2, len(x_vals) - 1]
        x_tick_labels = [f"{x_vals[i]:.3f}" for i in x_tick_indices]
        fig.update_xaxes(
            tickmode="array",
            tickvals=x_tick_indices,
            ticktext=x_tick_labels,
            title_text=f"<b>{models.axes[0].label}</b>",
        )

        # Y-axis ticks
        y_vals = models.axes[1].values
        y_tick_indices = [0, len(y_vals) // 2, len(y_vals) - 1]
        y_tick_labels = [f"{y_vals[i]:.3f}" for i in reversed(y_tick_indices)]
        fig.update_yaxes(
            tickmode="array",
            tickvals=y_tick_indices,
            ticktext=y_tick_labels,
            title_text=f"<b>{models.axes[1].label}</b>",
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


# ---------------------------------------------------------------------------
# Standalone convenience functions (legacy API preserved)
# ---------------------------------------------------------------------------


def _create_phase_change_trace_data(model_grid: ModelGrid, tracked_feature: int):
    norm = np.vectorize(lambda m: m.feature_norms[tracked_feature].cpu().item())(
        model_grid.models
    )
    interference = np.vectorize(
        lambda m: m.total_feature_interferences[tracked_feature].cpu().item()
    )(model_grid.models)

    max_interference = interference.max()
    interference_normalized = (
        interference / max_interference if max_interference > 0 else interference
    )
    phase_colors = _get_phase_color(norm, interference_normalized)

    metadata = np.stack(
        [
            norm,
            interference,
            model_grid.parameters_mesh[0],
            model_grid.parameters_mesh[1],
        ],
        axis=-1,
    )

    phase_colors = np.swapaxes(phase_colors, 0, 1)[::-1]
    metadata = np.swapaxes(metadata, 0, 1)[::-1]

    trace = go.Image(
        z=phase_colors,
        customdata=metadata,
        hovertemplate=(
            f"Norm: %{{customdata[0]:.2f}}<br>Interference: %{{customdata[1]:.2f}}<br>"
            f"{model_grid.axes[0].label}: %{{customdata[2]:.2f}}<br>"
            f"{model_grid.axes[1].label}: %{{customdata[3]:.2f}}<br><extra></extra>"
        ),
    )

    x_axis_values = model_grid.axes[0].values
    x_tick_indices = [0, len(x_axis_values) // 2, len(x_axis_values) - 1]
    x_tick_labels = [f"{x_axis_values[i]:.3f}" for i in x_tick_indices]
    x_axis_config = dict(
        tickmode="array",
        tickvals=x_tick_indices,
        ticktext=x_tick_labels,
        title=dict(text=f"<b>{model_grid.axes[0].label}</b>", font=dict(size=10)),
    )

    y_axis_values = model_grid.axes[1].values
    y_tick_indices = [0, len(y_axis_values) // 2, len(y_axis_values) - 1]
    y_tick_labels = [f"{y_axis_values[i]:.3f}" for i in reversed(y_tick_indices)]
    y_axis_config = dict(
        tickmode="array",
        tickvals=y_tick_indices,
        ticktext=y_tick_labels,
        title=dict(text=f"<b>{model_grid.axes[1].label}</b>", font=dict(size=10)),
    )

    return trace, x_axis_config, y_axis_config


def _add_model_phases_trace(model_grid: ModelGrid, tracked_feature, fig, *, col, row):
    trace, x_axis_config, y_axis_config = _create_phase_change_trace_data(
        model_grid, tracked_feature
    )
    fig.add_trace(trace, row=row, col=col)
    fig.update_xaxes(**x_axis_config, row=row, col=col)
    fig.update_yaxes(**y_axis_config, row=row, col=col)

    interference = np.vectorize(
        lambda m: m.total_feature_interferences[tracked_feature].cpu().item()
    )(model_grid.models)
    return interference.max()


def _add_colormap_trace(fig, *, col, row, max_interference: float = 1.0):
    COLORMAP_SIZE = 100
    interference_mesh, norm_mesh = np.meshgrid(
        np.linspace(0, 1, COLORMAP_SIZE), np.linspace(1, 0, COLORMAP_SIZE)
    )
    colormap = _get_phase_color(norm_mesh, interference_mesh)

    fig.add_trace(
        go.Image(
            z=colormap,
            customdata=np.stack(
                [norm_mesh, interference_mesh * max_interference], axis=-1
            ),
            hovertemplate="Interference: %{customdata[1]:.2f}<br>Norm: %{customdata[0]:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, COLORMAP_SIZE - 1],
        ticktext=["0", f"≥{max_interference:.2f}"],
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
    return fig


def plot_phase_change_multi(model_grid: ModelGrid, *, up_to: int, max_cols: int = 4):
    total_items = up_to + 1
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

    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs)

    max_interferences = []
    for i in range(up_to):
        row = i // n_cols + 1
        col = i % n_cols + 1
        max_interferences.append(
            _add_model_phases_trace(model_grid, i, fig, col=col, row=row)
        )

    colormap_row = up_to // n_cols + 1
    colormap_col = up_to % n_cols + 1
    max_interference = max(max_interferences) if max_interferences else 1.0
    _add_colormap_trace(
        fig, col=colormap_col, row=colormap_row, max_interference=max_interference
    )
    return fig


def plot_phase_change(model_grid: ModelGrid | list[ModelGrid], *, tracked_feature=1):
    if isinstance(model_grid, list):
        return _plot_phase_change_multi_grids(model_grid, tracked_feature)

    training_axis_idx = None
    for idx, axis in enumerate(model_grid.axes):
        if isinstance(axis, TrainingAxis):
            if training_axis_idx is not None:
                raise ValueError(
                    "ModelGrid cannot have multiple TrainingAxis instances"
                )
            training_axis_idx = idx

    if training_axis_idx is None:
        if len(model_grid.shape) != 2:
            raise ValueError(
                f"plot_phase_change requires a 2-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_phase_change_static(model_grid, tracked_feature)
    else:
        if len(model_grid.shape) != 3:
            raise ValueError(
                f"plot_phase_change with TrainingAxis requires a 3-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_phase_change_animated(
            model_grid, training_axis_idx, tracked_feature
        )


def _plot_phase_change_multi_grids(model_grids: list[ModelGrid], tracked_feature: int):
    for i, grid in enumerate(model_grids):
        for axis in grid.axes:
            if isinstance(axis, TrainingAxis):
                raise ValueError(
                    f"Grid {i} contains a TrainingAxis. Multiple grids with TrainingAxis are not supported."
                )
        if len(grid.shape) != 2:
            raise ValueError(
                f"plot_phase_change with multiple grids requires all grids to be 2-dimensional, "
                f"got {len(grid.shape)}-dimensional for grid {i} (shape: {grid.shape})."
            )

    n_grids = len(model_grids)
    column_widths = [1.0] * n_grids + [0.2]

    fig = make_subplots(
        rows=1,
        cols=n_grids + 1,
        column_widths=column_widths,
        subplot_titles=[f"Phase Change [Feature {tracked_feature}]"] * n_grids
        + ["Colormap"],
    )

    max_interferences = []
    for i, grid in enumerate(model_grids):
        max_interference = _add_model_phases_trace(
            grid, tracked_feature, fig, col=i + 1, row=1
        )
        max_interferences.append(max_interference)

    global_max_interference = max(max_interferences) if max_interferences else 1.0
    _add_colormap_trace(
        fig, col=n_grids + 1, row=1, max_interference=global_max_interference
    )
    return fig


def _plot_phase_change_static(model_grid: ModelGrid, tracked_feature: int):
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        subplot_titles=(f"Phase Change [Feature {tracked_feature}]", "Colormap"),
    )

    max_interference = _add_model_phases_trace(
        model_grid, tracked_feature, fig, col=1, row=1
    )
    _add_colormap_trace(fig, col=2, row=1, max_interference=max_interference)
    return fig


def _plot_phase_change_animated(
    model_grid: ModelGrid, training_axis_idx: int, tracked_feature: int
):
    training_axis = model_grid.axes[training_axis_idx]
    n_epochs = len(training_axis.values)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        subplot_titles=(f"Phase Change [Feature {tracked_feature}]", "Colormap"),
    )

    def slice_at_epoch(epoch_idx: int) -> ModelGrid:
        if training_axis_idx == 0:
            return model_grid[epoch_idx, :, :]
        elif training_axis_idx == 1:
            return model_grid[:, epoch_idx, :]
        else:
            return model_grid[:, :, epoch_idx]

    initial_grid = slice_at_epoch(0)
    initial_trace, x_axis_config, y_axis_config = _create_phase_change_trace_data(
        initial_grid, tracked_feature
    )
    fig.add_trace(initial_trace, row=1, col=1)
    fig.update_xaxes(**x_axis_config, row=1, col=1)
    fig.update_yaxes(**y_axis_config, row=1, col=1)
    _add_colormap_trace(fig, col=2, row=1)

    frames = []
    for epoch_idx in range(n_epochs):
        grid_slice = slice_at_epoch(epoch_idx)
        trace, _, _ = _create_phase_change_trace_data(grid_slice, tracked_feature)
        frame = go.Frame(
            data=[trace],
            name=str(epoch_idx),
            traces=[0],
        )
        frames.append(frame)

    fig.frames = frames

    sliders = [
        {
            "active": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": f"{training_axis.values[i]}",
                    "method": "animate",
                }
                for i in range(n_epochs)
            ],
            "currentvalue": {
                "prefix": f"{training_axis.label}: ",
                "visible": True,
                "xanchor": "center",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig.update_layout(
        sliders=sliders,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "x": 0.05,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
    )
    return fig
