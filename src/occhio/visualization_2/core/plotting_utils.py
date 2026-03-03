import itertools

import plotly.graph_objects as go

from occhio import ModelGrid


def _format_axis_value(val) -> str:
    """Format axis values for display in subplot headers."""
    if hasattr(val, "item"):
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

        x_domain = fig.layout[subplot.xaxis.plotly_name].domain
        y_domain = fig.layout[subplot.yaxis.plotly_name].domain

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
) -> None:
    n_axes = len(grid.shape)

    # Column headers
    col_axis = grid.axes[0]
    for model_col in range(grid.shape[0]):
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

    # Row headers
    if n_axes >= 2:
        row_axis = grid.axes[1]
        for model_row in range(grid.shape[1]):
            fig.add_annotation(
                text=f"{row_axis.label}: {_format_axis_value(row_axis.values[model_row])}",
                x=-0.02,
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
