"""Geometry visualization ported to the v2 SinglePlot framework.

``GeometryPlot`` shows geometry metrics across one axis (``n_render_axes = 1``).
``FeatureGeometryPlot`` renders the per-model network graph (``n_render_axes = 0``).

The standalone functions ``plot_geometry``, ``plot_feature_geometry``, and
``plot_feature_geometry_3d`` are preserved for backward compatibility.
"""

from enum import Enum

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid, TrainingAxis
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class GeometryPlotComponent(Enum):
    HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES = "hidden-dimensions-per-embedded-features"
    EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS = "embedded-features_per-hidden_dimensions"
    FEATURE_DIMENSIONALITIES = "feature-dimensionalities"
    MEAN_FEATURE_DIMENSIONALITIES = "mean-feature-dimensionalities"
    TOTAL_FEATURE_DIMENSIONALITIES = "total-feature-dimensionalities"
    GEOMETRIES = "geometries"


# Reference geometry lines used across plots
_GEOMETRIES: list[tuple[float, str, tuple[int, int, int]]] = [
    (1, "Dedicated Dimension", (255, 179, 186)),
    (3 / 4, "Tetrahedron", (186, 225, 255)),
    (2 / 3, "Triangle", (186, 255, 201)),
    (1 / 2, "Digon", (255, 223, 186)),
    (2 / 5, "Pentagon", (221, 186, 255)),
    (3 / 8, "Square Antiprism", (255, 235, 150)),
    (2 / 6, "Hexagon", (255, 200, 170)),
    (2 / 8, "Octagon", (186, 255, 255)),
    (0, "Not Learned", (255, 214, 229)),
]


# ---------------------------------------------------------------------------
# GeometryPlot (n_render_axes = 1)
# ---------------------------------------------------------------------------


class GeometryPlot(SinglePlot):
    """Geometry metrics across one model-grid axis.

    Displays hidden-dimensions-per-feature, feature dimensionalities, means,
    totals, and reference geometry lines.

    ``n_render_axes = 1`` -- the plot expects a 1D ModelGrid slice.
    """

    n_render_axes = 1

    def __init__(
        self,
        components: set[GeometryPlotComponent] | None = None,
    ):
        if components is None:
            components = set(GeometryPlotComponent) - {
                GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
            }
        self.components = components

    def render(self, fig: FigureProxy, models: ModelGrid) -> None:
        components = self.components

        if GeometryPlotComponent.HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES in components:
            fig.add_trace(
                go.Scatter(
                    x=list(models.axes[0].values),
                    y=[
                        model.hidden_dimensions_per_embedded_features.cpu()
                        for model in models.models
                    ],
                    mode="lines+markers",
                    line=dict(width=1, color="#333333", shape="spline"),
                    marker=dict(size=4, color="black"),
                    name="Hidden Dimensions / Learned Feature",
                    hovertemplate="Hidden Dimensions / Learned Feature: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS in components:
            fig.add_trace(
                go.Scatter(
                    x=list(models.axes[0].values),
                    y=[
                        model.embedded_features_per_hidden_dimensions.cpu()
                        for model in models.models
                    ],
                    mode="lines+markers",
                    line=dict(width=1, color="#333333", shape="spline"),
                    marker=dict(size=4, color="black"),
                    name="Learned Features / Hidden Dimensions",
                    hovertemplate="Learned Features / Hidden Dimensions: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.FEATURE_DIMENSIONALITIES in components:
            x_vals = []
            feature_dimensionalities = []
            for i, model in enumerate(models.models):
                x_vals.extend(
                    [models.axes[0].values[i]] * len(model.feature_dimensionalities)
                )
                feature_dimensionalities.extend(model.feature_dimensionalities.cpu())

            x_vals_jittered = np.array(x_vals) * np.exp(
                np.random.normal(0, 0.3 / len(models.models), len(x_vals))
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals_jittered.tolist(),
                    y=[float(v) for v in feature_dimensionalities],
                    mode="markers",
                    marker=dict(size=3, color="#333333", opacity=0.7),
                    name="Feature Dimensionality",
                    hovertemplate="Feature Dimensionality: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.MEAN_FEATURE_DIMENSIONALITIES in components:
            fig.add_trace(
                go.Scatter(
                    x=list(models.axes[0].values),
                    y=[
                        model.mean_feature_dimensionalities.cpu()
                        for model in models.models
                    ],
                    mode="lines+markers",
                    line=dict(width=1),
                    marker=dict(size=3, color="orange", opacity=0.6),
                    name="Mean Feature Dimensionality",
                    hovertemplate="Mean Feature Dimensionality: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.TOTAL_FEATURE_DIMENSIONALITIES in components:
            fig.add_trace(
                go.Scatter(
                    x=list(models.axes[0].values),
                    y=[
                        model.total_feature_dimensionalities_per_hidden_dimension.cpu()
                        for model in models.models
                    ],
                    mode="markers",
                    marker=dict(size=4, color="blue", opacity=0.6),
                    name="Total Feature Dimensionality / Hidden Dimension",
                    hovertemplate="Total Feature Dimensionality / Hidden Dimension: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_xaxes(
            title_text=models.axes[0].label,
            type="log",
            showgrid=False,
            dtick=0.2,
            tickformat=".3f",
            autorange="reversed",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="Hidden Dimensionality / Embedded Feature",
            showgrid=False,
            rangemode="tozero",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        )

    def configure_layout(self, fig: go.Figure) -> None:
        # Add geometry reference lines (figure-wide, not per-subplot)
        if GeometryPlotComponent.GEOMETRIES in self.components:
            for y, label, lc in _GEOMETRIES:
                fig.add_hline(
                    y=y,
                    line_color=f"rgba({lc[0]}, {lc[1]}, {lc[2]}, 0.5)",
                    line_width=5,
                )
                fig.add_annotation(
                    x=1.02,
                    xref="paper",
                    y=y,
                    yref="y",
                    text=label,
                    showarrow=False,
                    xanchor="left",
                    font=dict(
                        size=7,
                        color=f"rgb({lc[0]}, {lc[1]}, {lc[2]})",
                        weight="bold",
                    ),
                )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5
            ),
            margin=dict(r=100, b=120),
        )


# ---------------------------------------------------------------------------
# FeatureGeometryPlot (n_render_axes = 0)
# ---------------------------------------------------------------------------


class FeatureGeometryPlot(SinglePlot):
    """Network-graph visualization of feature interference for a single model.

    Each node is a feature; edges represent interference above the threshold.
    ``n_render_axes = 0`` -- renders per-model.
    """

    n_render_axes = 0

    def __init__(
        self,
        *,
        min_edge_interference: float = 0.1,
        feature_dimensionality_threshold: float = 0.1,
        dimensionality_range: list[float] | None = None,
    ):
        self.min_edge_interference = min_edge_interference
        self.feature_dimensionality_threshold = feature_dimensionality_threshold
        self.dimensionality_range = dimensionality_range

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        feature_dimensionalities = model.feature_dimensionalities.cpu()
        feature_interferences = (
            model.interferences.cpu()
            if hasattr(model.interferences, "cpu")
            else model.interferences
        )

        feature_dims_cpu = (
            feature_dimensionalities.cpu()
            if hasattr(feature_dimensionalities, "cpu")
            else feature_dimensionalities
        )

        if self.dimensionality_range is not None:
            active_feature_indices = np.where(
                (feature_dims_cpu >= self.feature_dimensionality_threshold)
                & (feature_dims_cpu >= self.dimensionality_range[0])
                & (feature_dims_cpu <= self.dimensionality_range[1])
            )[0]
        else:
            active_feature_indices = np.where(
                feature_dims_cpu >= self.feature_dimensionality_threshold
            )[0]

        if len(active_feature_indices) == 0:
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    showlegend=False,
                    name="No active features",
                )
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
            return

        G = nx.Graph()
        G.add_nodes_from(active_feature_indices)

        for i_idx, i in enumerate(active_feature_indices):
            for j_idx, j in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i, j]
                if abs(interference) >= self.min_edge_interference:
                    G.add_edge(i, j, weight=float(abs(interference).cpu()))

        if G.number_of_edges() > 0:
            positions = nx.spring_layout(
                G, weight="weight", k=15, iterations=5000, seed=42, scale=1.0, dim=2
            )
        else:
            positions = nx.circular_layout(G)

        node_x = [positions[n][0] for n in active_feature_indices]
        node_y = [positions[n][1] for n in active_feature_indices]

        max_dim = feature_dimensionalities[active_feature_indices].max()

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=7,
                    color="black",
                    opacity=(
                        feature_dimensionalities[active_feature_indices] / max_dim
                        if max_dim > 0
                        else 1.0
                    ),
                ),
                text=[
                    f"Feature {i}<br>Dimensionality: {feature_dimensionalities[i]:.3f}"
                    for i in active_feature_indices
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

        for i_idx, i_node in enumerate(active_feature_indices):
            for j_idx, j_node in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i_node, j_node]
                if abs(interference) > self.min_edge_interference:
                    x_coords = [positions[i_node][0], positions[j_node][0]]
                    y_coords = [positions[i_node][1], positions[j_node][1]]
                    edge_color = (
                        "rgba(255, 102, 102, 0.6)"
                        if interference > 0
                        else "rgba(51, 102, 204, 0.6)"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode="lines",
                            line=dict(color=edge_color, width=0.6),
                            hovertemplate=f"Interference: {interference:.3f}<extra></extra>",
                            showlegend=False,
                        )
                    )

        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )


# ---------------------------------------------------------------------------
# Standalone functions (backward-compatible legacy API)
# ---------------------------------------------------------------------------


def plot_geometry(
    model_grid: ToyModel | ModelGrid,
    components: set[GeometryPlotComponent] | None = None,
):
    if isinstance(model_grid, ToyModel):
        model_grid = ModelGrid(
            create_model=lambda: model_grid,
            axes=[],
            broadcast_samples=False,
            _models=np.array([model_grid]),
        )

    training_axis_idx = None
    for idx, axis in enumerate(model_grid.axes):
        if isinstance(axis, TrainingAxis):
            if training_axis_idx is not None:
                raise ValueError(
                    "ModelGrid cannot have multiple TrainingAxis instances"
                )
            training_axis_idx = idx

    if training_axis_idx is None:
        if len(model_grid.shape) > 1:
            raise ValueError(
                f"plot_geometry requires a 0 or 1-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_geometry_static(model_grid, components)
    else:
        if len(model_grid.shape) != 2:
            raise ValueError(
                f"plot_geometry with TrainingAxis requires a 2-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_geometry_animated(model_grid, training_axis_idx, components)


def _plot_geometry_static(
    model_grid: ModelGrid, components: set[GeometryPlotComponent] | None = None
):
    if components is None:
        components = set(GeometryPlotComponent) - {
            GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
        }

    fig = go.Figure()

    if len(model_grid.shape) == 0:
        model = model_grid.models.flat[0]

        if GeometryPlotComponent.HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES in components:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[model.hidden_dimensions_per_embedded_features.cpu()],
                    mode="markers",
                    marker=dict(size=8, color="black"),
                    name="Hidden Dimensions / Learned Feature",
                    hovertemplate="Hidden Dimensions / Learned Feature: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS in components:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[model.embedded_features_per_hidden_dimensions.cpu()],
                    mode="markers",
                    marker=dict(size=8, color="black"),
                    name="Learned Features / Hidden Dimensions",
                    hovertemplate="Learned Features / Hidden Dimensions: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.FEATURE_DIMENSIONALITIES in components:
            feature_dimensionalities = model.feature_dimensionalities.cpu()
            x_vals_jittered = np.random.normal(0, 0.05, len(feature_dimensionalities))
            fig.add_trace(
                go.Scatter(
                    x=x_vals_jittered,
                    y=feature_dimensionalities,
                    mode="markers",
                    marker=dict(size=3, color="#333333", opacity=0.7),
                    name="Feature Dimensionality",
                    hovertemplate="Feature Dimensionality: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.MEAN_FEATURE_DIMENSIONALITIES in components:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[model.mean_feature_dimensionalities.cpu()],
                    mode="markers",
                    marker=dict(size=8, color="orange", opacity=0.6),
                    name="Mean Feature Dimensionality",
                    hovertemplate="Mean Feature Dimensionality: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.TOTAL_FEATURE_DIMENSIONALITIES in components:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[model.total_feature_dimensionalities_per_hidden_dimension.cpu()],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.6),
                    name="Total Feature Dimensionality / Hidden Dimension",
                    hovertemplate="Total Feature Dimensionality / Hidden Dimension: %{y:.3f}<extra></extra>",
                )
            )

        if GeometryPlotComponent.GEOMETRIES in components:
            for y, label, line_color in _GEOMETRIES:
                fig.add_hline(
                    y=y,
                    line_color=f"rgba({line_color[0]}, {line_color[1]}, {line_color[2]}, 0.5)",
                    line_width=5,
                )
                fig.add_annotation(
                    x=1.02,
                    xref="paper",
                    y=y,
                    yref="y",
                    text=label,
                    showarrow=False,
                    xanchor="left",
                    font=dict(
                        size=7,
                        color=f"rgb({line_color[0]}, {line_color[1]}, {line_color[2]})",
                        weight="bold",
                    ),
                )

        fig.update_layout(
            xaxis_title="Model",
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=True,
                linewidth=1,
                linecolor="lightgray",
                mirror=True,
            ),
            yaxis_title="Hidden Dimensionality / Embedded Feature",
            yaxis=dict(
                showgrid=False,
                rangemode="tozero",
                showline=True,
                linewidth=1,
                linecolor="lightgray",
                mirror=True,
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(r=100, b=80),
        )
        return fig

    # Handle 1D grid
    if GeometryPlotComponent.HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES in components:
        fig.add_trace(
            go.Scatter(
                x=model_grid.axes[0].values,
                y=[
                    model.hidden_dimensions_per_embedded_features.cpu()
                    for model in model_grid.models
                ],
                mode="lines+markers",
                line=dict(width=1, color="#333333", shape="spline"),
                marker=dict(size=4, color="black"),
                name="Hidden Dimensions / Learned Feature",
                hovertemplate="Hidden Dimensions / Learned Feature: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS in components:
        fig.add_trace(
            go.Scatter(
                x=model_grid.axes[0].values,
                y=[
                    model.embedded_features_per_hidden_dimensions.cpu()
                    for model in model_grid.models
                ],
                mode="lines+markers",
                line=dict(width=1, color="#333333", shape="spline"),
                marker=dict(size=4, color="black"),
                name="Learned Features / Hidden Dimensions",
                hovertemplate="Learned Features / Hidden Dimensions: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.extend(
                [model_grid.axes[0].values[i]] * len(model.feature_dimensionalities)
            )
            feature_dimensionalities.extend(model.feature_dimensionalities.cpu())
        x_vals_jittered = np.array(x_vals) * np.exp(
            np.random.normal(0, 0.3 / len(model_grid.models), len(x_vals))
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals_jittered,
                y=feature_dimensionalities,
                mode="markers",
                marker=dict(size=3, color="#333333", opacity=0.7),
                name="Feature Dimensionality",
                hovertemplate="Feature Dimensionality: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.GEOMETRIES in components:
        for y, label, line_color in _GEOMETRIES:
            fig.add_hline(
                y=y,
                line_color=f"rgba({line_color[0]}, {line_color[1]}, {line_color[2]}, 0.5)",
                line_width=5,
            )
            fig.add_annotation(
                x=1.02,
                xref="paper",
                y=y,
                yref="y",
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(
                    size=7,
                    color=f"rgb({line_color[0]}, {line_color[1]}, {line_color[2]})",
                    weight="bold",
                ),
            )

    if GeometryPlotComponent.MEAN_FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        mean_feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.append(model_grid.axes[0].values[i])
            mean_feature_dimensionalities.append(
                model.mean_feature_dimensionalities.cpu()
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mean_feature_dimensionalities,
                mode="markers",
                marker=dict(size=4, color="orange", opacity=0.6),
                name="Mean Feature Dimensionality",
                hovertemplate="Mean Feature Dimensionality: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.TOTAL_FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        total_feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.append(model_grid.axes[0].values[i])
            total_feature_dimensionalities.append(
                model.total_feature_dimensionalities_per_hidden_dimension.cpu()
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=total_feature_dimensionalities,
                mode="markers",
                marker=dict(size=4, color="blue", opacity=0.6),
                name="Total Feature Dimensionality / Hidden Dimension",
                hovertemplate="Total Feature Dimensionality / Hidden Dimension: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title=model_grid.axes[0].label,
        xaxis_type="log",
        xaxis=dict(
            showgrid=False,
            dtick=0.2,
            tickformat=".3f",
            autorange="reversed",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        ),
        yaxis_title="Hidden Dimensionality / Embedded Feature",
        yaxis=dict(
            showgrid=False,
            rangemode="tozero",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(r=100, b=80),
    )
    return fig


def _plot_geometry_animated(
    model_grid: ModelGrid,
    training_axis_idx: int,
    components: set[GeometryPlotComponent] | None = None,
):
    if components is None:
        components = set(GeometryPlotComponent) - {
            GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
        }

    training_axis = model_grid.axes[training_axis_idx]
    n_epochs = len(training_axis.values)

    def slice_at_epoch(epoch_idx: int) -> ModelGrid:
        if training_axis_idx == 0:
            return model_grid[epoch_idx, :]
        else:
            return model_grid[:, epoch_idx]

    initial_grid = slice_at_epoch(0)
    fig = _create_geometry_figure(initial_grid, components)

    frames = []
    for epoch_idx in range(n_epochs):
        grid_slice = slice_at_epoch(epoch_idx)
        frame_fig = _create_geometry_figure(grid_slice, components)
        frame = go.Frame(data=frame_fig.data, name=str(epoch_idx))
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
                    "label": f"{int(training_axis.values[i])}",
                    "method": "animate",
                }
                for i in range(n_epochs)
            ],
            "currentvalue": {
                "prefix": f"{training_axis.label}: ",
                "visible": False,
                "xanchor": "center",
            },
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.05,
            "xanchor": "left",
            "y": -0.15,
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
                "y": 1.45,
                "yanchor": "top",
            }
        ],
    )
    return fig


def _create_geometry_figure(
    model_grid: ModelGrid, components: set[GeometryPlotComponent]
) -> go.Figure:
    fig = go.Figure()

    if GeometryPlotComponent.HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES in components:
        fig.add_trace(
            go.Scatter(
                x=model_grid.axes[0].values,
                y=[
                    model.hidden_dimensions_per_embedded_features.cpu()
                    for model in model_grid.models
                ],
                mode="lines+markers",
                line=dict(width=1, color="#333333", shape="spline"),
                marker=dict(size=4, color="black"),
                name="Hidden Dimensions / Learned Feature",
                hovertemplate="Hidden Dimensions / Learned Feature: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS in components:
        fig.add_trace(
            go.Scatter(
                x=model_grid.axes[0].values,
                y=[
                    model.embedded_features_per_hidden_dimensions.cpu()
                    for model in model_grid.models
                ],
                mode="lines+markers",
                line=dict(width=1, color="#333333", shape="spline"),
                marker=dict(size=4, color="black"),
                name="Learned Features / Hidden Dimensions",
                hovertemplate="Learned Features / Hidden Dimensions: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.extend(
                [model_grid.axes[0].values[i]] * len(model.feature_dimensionalities)
            )
            feature_dimensionalities.extend(model.feature_dimensionalities.cpu())
        x_vals_jittered = np.array(x_vals) * np.exp(
            np.random.normal(0, 0.3 / len(model_grid.models), len(x_vals))
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals_jittered,
                y=feature_dimensionalities,
                mode="markers",
                marker=dict(size=3, color="#333333", opacity=0.7),
                name="Feature Dimensionality",
                hovertemplate="Feature Dimensionality: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.GEOMETRIES in components:
        for y, label, line_color in _GEOMETRIES:
            fig.add_hline(
                y=y,
                line_color=f"rgba({line_color[0]}, {line_color[1]}, {line_color[2]}, 0.5)",
                line_width=5,
            )
            fig.add_annotation(
                x=1.02,
                xref="paper",
                y=y,
                yref="y",
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(
                    size=7,
                    color=f"rgb({line_color[0]}, {line_color[1]}, {line_color[2]})",
                    weight="bold",
                ),
            )

    if GeometryPlotComponent.MEAN_FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        mean_feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.append(model_grid.axes[0].values[i])
            mean_feature_dimensionalities.append(
                model.mean_feature_dimensionalities.cpu()
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mean_feature_dimensionalities,
                mode="lines+markers",
                line=dict(width=1),
                marker=dict(size=3, color="orange", opacity=0.6),
                name="Mean Feature Dimensionality",
                hovertemplate="Mean Feature Dimensionality: %{y:.3f}<extra></extra>",
            )
        )

    if GeometryPlotComponent.TOTAL_FEATURE_DIMENSIONALITIES in components:
        x_vals = []
        total_feature_dimensionalities = []
        for i, model in enumerate(model_grid.models):
            x_vals.append(model_grid.axes[0].values[i])
            total_feature_dimensionalities.append(
                model.total_feature_dimensionalities_per_hidden_dimension.cpu()
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=total_feature_dimensionalities,
                mode="markers",
                marker=dict(size=4, color="blue", opacity=0.6),
                name="Total Feature Dimensionality / Hidden Dimension",
                hovertemplate="Total Feature Dimensionality / Hidden Dimension: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title=model_grid.axes[0].label,
        xaxis_type="log",
        xaxis=dict(
            showgrid=False,
            dtick=0.2,
            tickformat=".3f",
            autorange="reversed",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        ),
        yaxis_title="Hidden Dimensionality / Embedded Feature",
        yaxis=dict(
            showgrid=False,
            rangemode="tozero",
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(r=100, b=120),
    )
    return fig


def plot_feature_geometry(
    models: ToyModel | ModelGrid | list[ToyModel],
    *,
    min_edge_interference: float = 0.1,
    feature_dimensionality_threshold: float = 0.1,
    dimensionality_range: list[float] = None,
    dimensions: int = 2,
):
    if isinstance(models, ToyModel):
        models = [models]
    elif isinstance(models, ModelGrid):
        models = list(models.models.flat)

    n_models = len(models)

    fig = make_subplots(
        rows=1,
        cols=n_models,
        horizontal_spacing=0.05,
        specs=[
            [
                {"type": "scatter3d" if dimensions == 3 else "xy"}
                for _ in range(n_models)
            ]
        ],
    )

    for model_idx, model in enumerate(models):
        feature_dimensionalities = model.feature_dimensionalities.cpu()
        feature_interferences = (
            model.interferences.cpu()
            if hasattr(model.interferences, "cpu")
            else model.interferences
        )

        feature_dims_cpu = (
            feature_dimensionalities.cpu()
            if hasattr(feature_dimensionalities, "cpu")
            else feature_dimensionalities
        )

        if dimensionality_range is not None:
            active_feature_indices = np.where(
                (feature_dims_cpu >= feature_dimensionality_threshold)
                & (feature_dims_cpu >= dimensionality_range[0])
                & (feature_dims_cpu <= dimensionality_range[1])
            )[0]
        else:
            active_feature_indices = np.where(
                feature_dims_cpu >= feature_dimensionality_threshold
            )[0]
        n_active_features = len(active_feature_indices)

        if n_active_features == 0:
            if dimensions == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="markers",
                        showlegend=False,
                        name="No active features",
                    ),
                    row=1,
                    col=model_idx + 1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        showlegend=False,
                        name="No active features",
                    ),
                    row=1,
                    col=model_idx + 1,
                )
                fig.update_xaxes(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    row=1,
                    col=model_idx + 1,
                )
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    row=1,
                    col=model_idx + 1,
                )
            continue

        G = nx.Graph()
        G.add_nodes_from(active_feature_indices)
        for i_idx, i in enumerate(active_feature_indices):
            for j_idx, j in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i, j]
                if abs(interference) >= min_edge_interference:
                    weight_value = float(abs(interference).cpu())
                    G.add_edge(i, j, weight=weight_value)

        if G.number_of_edges() > 0:
            k_value = 30 if dimensions == 3 else 15
            positions = nx.spring_layout(
                G,
                weight="weight",
                k=k_value,
                iterations=5000,
                seed=42,
                scale=1.0,
                dim=dimensions,
            )
        else:
            if dimensions == 3:
                positions_2d = nx.circular_layout(G)
                positions = {
                    node: [pos[0], pos[1], 0] for node, pos in positions_2d.items()
                }
            else:
                positions = nx.circular_layout(G)

        node_x = [positions[node][0] for node in active_feature_indices]
        node_y = [positions[node][1] for node in active_feature_indices]
        if dimensions == 3:
            node_z = [positions[node][2] for node in active_feature_indices]

        max_dimensionality = feature_dimensionalities[active_feature_indices].max()

        if dimensions == 3:
            opacity_values = (
                feature_dimensionalities[active_feature_indices] / max_dimensionality
                if max_dimensionality > 0
                else 1.0
            )
            if hasattr(opacity_values, "tolist"):
                opacity_values = opacity_values.tolist()
            elif hasattr(opacity_values, "numpy"):
                opacity_values = opacity_values.numpy().tolist()

            fig.add_trace(
                go.Scatter3d(
                    x=node_x,
                    y=node_y,
                    z=node_z,
                    mode="markers",
                    marker=dict(size=5, color="black"),
                    text=[
                        f"Feature {i}<br>Dimensionality: {feature_dimensionalities[i]:.3f}"
                        for i in active_feature_indices
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=model_idx + 1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker=dict(
                        size=7,
                        color="black",
                        opacity=(
                            feature_dimensionalities[active_feature_indices]
                            / max_dimensionality
                            if max_dimensionality > 0
                            else 1.0
                        ),
                    ),
                    text=[
                        f"Feature {i}<br>Dimensionality: {feature_dimensionalities[i]:.3f}"
                        for i in active_feature_indices
                    ],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=model_idx + 1,
            )

        for i_idx, i_node in enumerate(active_feature_indices):
            for j_idx, j_node in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i_node, j_node]
                if abs(interference) > min_edge_interference:
                    x_coords = [positions[i_node][0], positions[j_node][0]]
                    y_coords = [positions[i_node][1], positions[j_node][1]]
                    edge_color = (
                        "rgba(255, 102, 102, 0.6)"
                        if interference > 0
                        else "rgba(51, 102, 204, 0.6)"
                    )
                    if dimensions == 3:
                        z_coords = [positions[i_node][2], positions[j_node][2]]
                        fig.add_trace(
                            go.Scatter3d(
                                x=x_coords,
                                y=y_coords,
                                z=z_coords,
                                mode="lines",
                                line=dict(color=edge_color, width=3),
                                hovertemplate=f"Interference: {interference:.3f}<extra></extra>",
                                showlegend=False,
                            ),
                            row=1,
                            col=model_idx + 1,
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode="lines",
                                line=dict(color=edge_color, width=0.6),
                                hovertemplate=f"Interference: {interference:.3f}<extra></extra>",
                                showlegend=False,
                            ),
                            row=1,
                            col=model_idx + 1,
                        )

        if dimensions == 3:
            scene_name = "scene" if model_idx == 0 else f"scene{model_idx + 1}"
            fig.update_layout(
                **{
                    scene_name: dict(
                        xaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            showbackground=False,
                            title="",
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            showbackground=False,
                            title="",
                        ),
                        zaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            showbackground=False,
                            title="",
                        ),
                        bgcolor="white",
                    )
                }
            )
        else:
            fig.update_xaxes(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                row=1,
                col=model_idx + 1,
            )
            fig.update_yaxes(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                row=1,
                col=model_idx + 1,
            )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        width=500 * n_models,
    )
    return fig


def plot_feature_geometry_3d(
    models: ToyModel | ModelGrid | list[ToyModel],
    *,
    min_edge_interference: float = 0.1,
    feature_dimensionality_threshold: float = 0.1,
    dimensionality_range: list[float] = None,
):
    return plot_feature_geometry(
        models,
        min_edge_interference=min_edge_interference,
        feature_dimensionality_threshold=feature_dimensionality_threshold,
        dimensionality_range=dimensionality_range,
        dimensions=3,
    )
