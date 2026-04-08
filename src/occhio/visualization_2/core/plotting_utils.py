import itertools

import plotly.graph_objects as go
import torch

from occhio import ModelGrid


def _format_axis_value(val: int | float | torch.Tensor) -> str:
    """Format axis values for display in subplot headers."""
    if isinstance(val, torch.Tensor):
        val = val.item()
    if isinstance(val, (int, float)):
        return f"{val:.4g}"
    return str(val)


def model_domain_center(
    fig: go.Figure,
    model_row: int,
    model_col: int,
    inner_rows: int,
    inner_cols: int,
) -> tuple[float, float]:
    """Compute the paper-coordinate center of a model's subplot block.

    Args:
        fig: Plotly figure with subplots already created.
        model_row: 0-indexed model row in the outer grid.
        model_col: 0-indexed model column in the outer grid.
        inner_rows: Number of subplot rows per model.
        inner_cols: Number of subplot columns per model.

    Returns:
        (x_center, y_center) in paper coordinates (0–1).
    """
    left, right = 1.0, 0.0
    bottom, top = 1.0, 0.0

    row_start = model_row * inner_rows + 1  # Plotly rows are 1-indexed
    row_end = row_start + inner_rows
    col_start = model_col * inner_cols + 1  # Plotly cols are 1-indexed
    col_end = col_start + inner_cols

    for row, col in itertools.product(
        range(row_start, row_end), range(col_start, col_end)
    ):
        subplot = fig.get_subplot(row=row, col=col)

        if subplot is None:
            continue

        # Domain subplots (e.g., Indicator, Pie) have x/y directly
        # XY subplots have xaxis/yaxis that reference layout axes
        if hasattr(subplot, "xaxis") and hasattr(subplot, "yaxis"):
            x_domain = fig.layout[subplot.xaxis.plotly_name].domain
            y_domain = fig.layout[subplot.yaxis.plotly_name].domain
        elif hasattr(subplot, "x") and hasattr(subplot, "y"):
            # SubplotDomain: x and y are tuples like (0.0, 0.45)
            x_domain = subplot.x
            y_domain = subplot.y
        else:
            continue

        left, right = (min(left, x_domain[0]), max(right, x_domain[1]))
        bottom, top = (min(bottom, y_domain[0]), max(top, y_domain[1]))

    if bottom >= top or left >= right:
        raise ValueError(
            f"No valid subplot found in model row {model_row} / col {model_col}."
        )

    x_center = (left + right) / 2
    y_center = (bottom + top) / 2

    return (x_center, y_center)


def add_grid_headers(
    fig: go.Figure,
    grid: ModelGrid,
    inner_rows: int = 1,
    inner_cols: int = 1,
    facet_axes: list[int] | None = None,
) -> None:
    """Add column and row header annotations to a faceted figure.

    Args:
        fig: Plotly figure to annotate.
        grid: The ModelGrid whose axis labels/values are used.
        inner_rows: Subplot rows per model (for composite layouts).
        inner_cols: Subplot columns per model (for composite layouts).
        facet_axes: Which grid axes map to columns (index 0) and rows (index 1).
            Defaults to the first two axes.
    """
    # Default to legacy behavior if facet_indices not provided
    if facet_axes is None:
        facet_axes = list(range(min(len(grid.shape), 2)))

    # Column headers (first facet axis)
    if len(facet_axes) >= 1:
        col_axis_idx = facet_axes[0]
        col_axis = grid.axes[col_axis_idx]
        for model_col in range(grid.shape[col_axis_idx]):
            fig.add_annotation(
                text=f"{col_axis.label}: {_format_axis_value(col_axis.values[model_col])}",
                x=model_domain_center(
                    fig,
                    model_row=0,
                    model_col=model_col,
                    inner_rows=inner_rows,
                    inner_cols=inner_cols,
                )[0],
                y=1.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11),
                xanchor="center",
                yanchor="bottom",
            )

    # Row headers (second facet axis)
    if len(facet_axes) >= 2:
        row_axis_idx = facet_axes[1]
        row_axis = grid.axes[row_axis_idx]
        for model_row in range(grid.shape[row_axis_idx]):
            fig.add_annotation(
                text=f"{row_axis.label}: {_format_axis_value(row_axis.values[model_row])}",
                x=-0.05,
                y=model_domain_center(
                    fig,
                    model_row=model_row,
                    model_col=0,
                    inner_rows=inner_rows,
                    inner_cols=inner_cols,
                )[1],
                xref="paper",
                yref="paper",
                showarrow=False,
                textangle=-90,
                font=dict(size=11),
                xanchor="right",
                yanchor="middle",
            )
