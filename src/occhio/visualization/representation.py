import math

# Example that triggers ruff linting errors:
import os  # E401: multiple imports on one line
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel


def plot_representation(model_grid: ToyModel | ModelGrid):
    """Plot W^T W heatmaps, bias vectors, and feature norms/interferences for models in a 0D or 1D grid.

    Args:
        model_grid: A ToyModel or 0/1-dimensional ModelGrid containing the models to visualize.

    Returns:
        A plotly Figure with heatmaps and bar charts arranged in a grid.

    Raises:
        ValueError: If model_grid is not 0 or 1-dimensional or is empty.
    """
    # Convert ToyModel to ModelGrid
    if isinstance(model_grid, ToyModel):
        model_grid = ModelGrid(
            create_model=lambda: model_grid,
            axes=[],
            cache_samples=False,
            _models=np.array([model_grid]),
        )

    if len(model_grid.shape) > 1:
        raise ValueError(
            f"plot_representation requires a 0 or 1-dimensional ModelGrid, "
            f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
        )

    n_models = len(model_grid.models.flat)
    if n_models == 0:
        raise ValueError("Cannot plot representation for an empty ModelGrid.")

    # Layout configuration
    max_models_per_row = 5
    cols_per_model = 3  # W^T W heatmap, b heatmap, spacer
    n_model_rows = math.ceil(n_models / max_models_per_row)
    n_subplot_rows = (
        n_model_rows * 2
    )  # Each model row has 2 physical rows (heatmaps, bars)

    # Calculate column widths
    first_model = model_grid.models.flat[0]
    n_features = first_model.n_features
    column_widths = []
    for _ in range(min(n_models, max_models_per_row)):
        column_widths.extend([1, 1 / n_features, 0.1])  # Thin spacer between models

    fig = make_subplots(
        rows=n_subplot_rows,
        cols=len(column_widths),
        horizontal_spacing=0.005,  # Minimal spacing within models
        vertical_spacing=0.01,
        column_widths=column_widths,
    )

    colorscale = ["#6699FF", "#F0F0F0", "#FF6666"]

    for model_idx, model in enumerate(model_grid.models.flat):
        # Determine position in grid
        model_row = model_idx // max_models_per_row
        model_col_offset = (model_idx % max_models_per_row) * cols_per_model

        # Physical subplot rows: heatmap row and bar row
        heatmap_row = model_row * 2 + 1
        bar_row = model_row * 2 + 2

        # Get data from model
        W_T_W = model.W_T_W.detach().cpu()
        b = model.ae.b.detach().cpu().reshape(-1, 1)
        feature_norms = model.feature_norms.detach().cpu()
        interferences = model.total_feature_interferences.detach().cpu()

        # Determine if this is the last model (for colorbars)
        is_last_model = model_idx == n_models - 1

        # --- W^T W Heatmap ---
        fig.add_trace(
            go.Heatmap(
                z=W_T_W.numpy(),
                colorscale=colorscale,
                zmid=0,
                zmin=-1.2,
                zmax=1.2,
                hovertemplate="i: %{y}<br>j: %{x}<br>value: %{z:.2f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="W^T W", side="right", font=dict(size=10)),
                    len=0.3,
                    thickness=10,
                    y=0.75,
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=["-1", "0", "1"],
                ),
                showscale=is_last_model,
            ),
            row=heatmap_row,
            col=model_col_offset + 1,
        )

        # --- Bias Heatmap ---
        fig.add_trace(
            go.Heatmap(
                z=b.numpy(),
                colorscale=colorscale,
                zmid=0,
                zmin=-1.2,
                zmax=1.2,
                hovertemplate="i: %{y}<br>value: %{z:.2f}<extra></extra>",
                showscale=False,
            ),
            row=heatmap_row,
            col=model_col_offset + 2,
        )

        # Update axes for heatmaps
        fig.update_xaxes(
            side="top",
            showticklabels=False,
            row=heatmap_row,
            col=model_col_offset + 1,
        )
        fig.update_xaxes(
            side="top",
            showticklabels=False,
            row=heatmap_row,
            col=model_col_offset + 2,
        )
        fig.update_yaxes(
            showticklabels=False,
            autorange="reversed",
            scaleanchor=f"x{(heatmap_row - 1) * len(column_widths) + model_col_offset + 1}",
            scaleratio=1,
            constrain="domain",
            row=heatmap_row,
            col=model_col_offset + 1,
        )
        fig.update_yaxes(
            showticklabels=False,
            autorange="reversed",
            scaleanchor=f"x{(heatmap_row - 1) * len(column_widths) + model_col_offset + 2}",
            scaleratio=1,
            constrain="domain",
            row=heatmap_row,
            col=model_col_offset + 2,
        )

        # --- Feature Norms Bar Chart ---
        fig.add_trace(
            go.Bar(
                x=feature_norms.numpy(),
                y=list(reversed(range(len(feature_norms)))),
                orientation="h",
                marker=dict(
                    color=interferences.numpy(),
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(
                            text="Interference", side="right", font=dict(size=10)
                        ),
                        len=0.5,
                        thickness=10,
                        y=0.25,
                        yanchor="middle",
                        tickmode="array",
                        tickvals=[0, 1],
                        ticktext=["0", "1"],
                    ),
                    showscale=is_last_model,
                    cmin=0,
                    cmax=1.2,
                ),
                hovertemplate="||W<sub>%{y}</sub>||: %{x:.4f}<extra></extra>",
            ),
            row=bar_row,
            col=model_col_offset + 1,
        )

        # Update axes for bar chart
        norm_range_max = 1.2
        fig.update_xaxes(
            range=[0, norm_range_max],
            tickmode="array",
            tickvals=[0, 1],
            showgrid=True,
            gridcolor="gray",
            gridwidth=0.5,
            griddash="dot",
            row=bar_row,
            col=model_col_offset + 1,
        )
        fig.update_yaxes(
            showticklabels=False,
            row=bar_row,
            col=model_col_offset + 1,
        )

        # Add gray background rectangle
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=norm_range_max,
            y1=n_features,
            fillcolor="#F0F0F0",
            layer="below",
            line_width=0,
            row=bar_row,
            col=model_col_offset + 1,
        )

        # Add annotation with axis label and value
        if len(model_grid.axes) > 0:
            fig.add_annotation(
                text=f"{model_grid.axes[0].label} = {model_grid.axes[0].values[model_idx]:.3f}",
                xref=f"x{(heatmap_row - 1) * len(column_widths) + model_col_offset + 1}",
                yref=f"y{(heatmap_row - 1) * len(column_widths) + model_col_offset + 1}",
                x=-0.5,
                y=n_features,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=8),
            )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=20),
    )

    return fig
