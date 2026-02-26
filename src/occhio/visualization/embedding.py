import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid


# [18.02.26 | OliverSieweke] TODO: Make this more general so it works with 1-dimensional embedding space (maybe 3 dimensional too)
def plot_embedding(
    model_grid: ModelGrid,
):
    if len(model_grid.shape) > 2:
        raise ValueError(
            f"plot_embedding requires a 1 or 2-dimensional ModelGrid, "
            f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
        )

    # Create subplots based on grid dimensions
    n_axes = len(model_grid.shape)
    if n_axes == 1:
        n_rows, n_cols = 1, model_grid.shape[0]
    else:  # n_axes == 2
        n_rows, n_cols = model_grid.shape

    fig = go.Figure()

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{model_grid.axes[0].label}: {model_grid.axes[0].values[i]}"
            for i in range(n_cols)
        ]
        if n_axes == 1
        else [
            f"{model_grid.axes[0].label}: {model_grid.axes[0].values[row]}, "
            f"{model_grid.axes[1].label}: {model_grid.axes[1].values[col]}"
            for row in range(n_rows)
            for col in range(n_cols)
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    fig.update_annotations(font_size=10)

    fig_width = 700  # Plotly default
    height = max(fig_width * n_rows / n_cols, 300)

    fig.update_layout(
        height=height,
        plot_bgcolor="#FCFBF8",
        showlegend=False,
    )

    # [18.02.26 | OliverSieweke] TODO: Make z dynamic here
    z = 1.2
    fig.update_xaxes(
        range=[-z, z],
        showticklabels=False,
        showgrid=False,
    )
    fig.update_yaxes(
        range=[-z, z],
        showticklabels=False,
        showgrid=False,
    )

    colorscale = px.colors.sequential.Plasma_r

    # Loop over grid positions and create subplots
    for idx in range(n_rows * n_cols):
        if n_axes == 1:
            row, col = 1, idx + 1
        else:  # n_axes == 2
            row = idx // n_cols + 1
            col = idx % n_cols + 1

        axis_idx = "" if idx == 0 else str(idx + 1)
        xref = f"x{axis_idx}"
        yref = f"y{axis_idx}"

        fig.update_xaxes(
            range=[-z, z], showticklabels=False, showgrid=False, row=row, col=col
        )
        fig.update_yaxes(
            range=[-z, z],
            showticklabels=False,
            showgrid=False,
            scaleanchor=xref,
            scaleratio=1,
            row=row,
            col=col,
        )

        model = model_grid.models.ravel()[idx]

        for feature_idx in range(model.W.shape[1]):
            # [18.02.26 | OliverSieweke] TODO: account for relative importance > 1 as well
            color_idx = int(
                model.importances[feature_idx].item() * 0.9 * (len(colorscale) - 1)
            )
            color = colorscale[color_idx]

            x_val = model.W[0, feature_idx].cpu().item()
            y_val = model.W[1, feature_idx].cpu().item()
            importance = model.importances[feature_idx].item()

            # Add arrow annotation (no hover support)
            fig.add_annotation(
                x=x_val,
                y=y_val,
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                ax=0,
                ay=0,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                opacity=0.7,
                row=row,
                col=col,
            )

            # Add invisible scatter point at arrow tip for hover
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0),
                    hovertemplate=(
                        f"<b>Feature {feature_idx}</b><br>"
                        f"Importance: {importance:.4f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    return fig
