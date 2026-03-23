from enum import Enum

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.distributions import CorrelatedPairs, HierarchicalPairs, SparseUniform
from occhio.model_grid import ModelGrid, TrainingAxis
from occhio.toy_model import ToyModel


class GeometryPlotComponent(Enum):
    HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES = "hidden-dimensions-per-embedded-features"
    EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS = "embedded-features_per-hidden_dimensions"
    FEATURE_DIMENSIONALITIES = "feature-dimensionalities"
    MEAN_FEATURE_DIMENSIONALITIES = "mean-feature-dimensionalities"
    TOTAL_FEATURE_DIMENSIONALITIES = "total-feature-dimensionalities"
    GEOMETRIES = "geometries"


def plot_geometry(
    model_grid: ToyModel | ModelGrid,
    components: set[GeometryPlotComponent] | None = None,
):
    # Convert ToyModel to ModelGrid
    if isinstance(model_grid, ToyModel):
        model_grid = ModelGrid(
            create_model=lambda: model_grid,
            axes=[],
            broadcast_samples=False,
            _models=np.array([model_grid]),
        )

    # Find TrainingAxis if present
    training_axis_idx = None
    for idx, axis in enumerate(model_grid.axes):
        if isinstance(axis, TrainingAxis):
            if training_axis_idx is not None:
                raise ValueError(
                    "ModelGrid cannot have multiple TrainingAxis instances"
                )
            training_axis_idx = idx

    # Validate dimensions
    if training_axis_idx is None:
        # Static case: require 0D or 1D grid
        if len(model_grid.shape) > 1:
            raise ValueError(
                f"plot_geometry requires a 0 or 1-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_geometry_static(model_grid, components)
    else:
        # Animated case: require 2D grid with one TrainingAxis
        if len(model_grid.shape) != 2:
            raise ValueError(
                f"plot_geometry with TrainingAxis requires a 2-dimensional ModelGrid, "
                f"got {len(model_grid.shape)}-dimensional (shape: {model_grid.shape})."
            )
        return _plot_geometry_animated(model_grid, training_axis_idx, components)


def _plot_geometry_static(
    model_grid: ModelGrid, components: set[GeometryPlotComponent] | None = None
):
    """Create static geometry plot for 0D or 1D grid."""
    if components is None:
        components = set(GeometryPlotComponent) - {
            GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
        }

    fig = go.Figure()

    # Handle 0D grid (single model)
    if len(model_grid.shape) == 0:
        # For a single model, show a single point plot
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

        # Add geometries reference lines
        if GeometryPlotComponent.GEOMETRIES in components:
            geometries: list[tuple[float, str, tuple[int, int, int]]] = [
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

            for y, label, line_color in geometries:
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

    # Handle 1D grid (multiple models)
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

        # [10.02.26 | OliverSieweke] TODO: Work on this to make it a sensible jitter based on number of models
        # [10.02.26 | OliverSieweke] TODO: also make sure the ticks don't expand to wide
        # log_x = np.log(x_vals)
        # log_x_range = np.max(log_x) - np.min(log_x)
        # sigma_log = (0.15 * log_x_range) / max(1, len(models))
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
        geometries: list[tuple[float, str, tuple[int, int, int]]] = [
            (1, "Dedicated Dimension", (255, 179, 186)),
            (3 / 4, "Tetrahedron", (186, 225, 255)),
            (2 / 3, "Triangle", (186, 255, 201)),
            (1 / 2, "Digon", (255, 223, 186)),
            (2 / 5, "Pentagon", (221, 186, 255)),
            (3 / 8, "Square Antiprism", (255, 235, 150)),
            (2 / 6, "Hexagon", (255, 200, 170)),
            (2 / 8, "Octagon", (186, 255, 255)),
            # [2 / 20, "Icosagon", (200, 220, 255)],
            # [1 / 12, "Dodecagon", (200, 220, 255)],
            (0, "Not Learned", (255, 214, 229)),
        ]

        for y, label, line_color in geometries:
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
    """Create animated geometry plot with slider for 2D grid with TrainingAxis."""
    if components is None:
        components = set(GeometryPlotComponent) - {
            GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
        }

    # Get the training axis
    training_axis = model_grid.axes[training_axis_idx]
    n_epochs = len(training_axis.values)

    # Helper to slice grid at a specific epoch
    def slice_at_epoch(epoch_idx: int) -> ModelGrid:
        """Extract 1D grid at specific epoch."""
        if training_axis_idx == 0:
            return model_grid[epoch_idx, :]
        else:  # training_axis_idx == 1
            return model_grid[:, epoch_idx]

    # Create initial figure with first epoch
    initial_grid = slice_at_epoch(0)
    fig = _create_geometry_figure(initial_grid, components)

    # Pre-compute all frame data for animation
    frames = []
    for epoch_idx in range(n_epochs):
        grid_slice = slice_at_epoch(epoch_idx)
        frame_fig = _create_geometry_figure(grid_slice, components)

        # Create frame with all traces from the frame figure
        frame = go.Frame(
            data=frame_fig.data,
            name=str(epoch_idx),
        )
        frames.append(frame)

    fig.frames = frames

    # Add slider for epoch selection
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
    """Create a geometry figure for a 1D model grid."""
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
        geometries: list[tuple[float, str, tuple[int, int, int]]] = [
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

        for y, label, line_color in geometries:
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
    # Convert ToyModel to list
    if isinstance(models, ToyModel):
        models = [models]
    # Convert ModelGrid to list
    elif isinstance(models, ModelGrid):
        models = list(models.models.flat)

    n_models = len(models)

    feature_probabilities = [
        # [10.02.26 | OliverSieweke] TODO: fix below once distributions updated
        (
            model.distribution.p_active[0]
            if hasattr(model.distribution.p_active, "__getitem__")
            and hasattr(model.distribution.p_active, "shape")
            and len(model.distribution.p_active.shape) > 0
            else (
                model.distribution.p_active
                if isinstance(model.distribution, SparseUniform)
                or isinstance(model.distribution, CorrelatedPairs)
                or isinstance(model.distribution, HierarchicalPairs)
                else (_ for _ in ()).throw(
                    UserWarning(
                        "plot_geometry currently expects ToyModel.distribution to be SparseUniform "
                        "(or at least to expose .p_active with the same meaning)."
                    )
                )
            )
        )
        for model in models
    ]

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
        n_features = len(feature_dimensionalities)

        # Filter features based on dimensionality threshold and range
        # Convert to CPU for NumPy compatibility if it's a GPU tensor
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

        # Skip this model if no features meet the criteria
        if n_active_features == 0:
            # Add empty subplot to maintain layout
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

                # Update subplot axes for empty plot
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

        # Create networkx graph
        G = nx.Graph()
        G.add_nodes_from(active_feature_indices)

        for i_idx, i in enumerate(active_feature_indices):
            for j_idx, j in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i, j]

                if abs(interference) >= min_edge_interference:
                    # Convert to Python scalar to avoid device type issues with networkx
                    weight_value = float(abs(interference).cpu())
                    G.add_edge(i, j, weight=weight_value)

        if G.number_of_edges() > 0:
            # [11.02.26 | OliverSieweke] TODO: check parameters here
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

            # positions = nx.kamada_kawai_layout(G, weight="weight")

            # positions = nx.spectral_layout(G, weight="weight")

            # try:
            #     positions = nx.nx_agraph.graphviz_layout(G, prog='neato')
            # except:
            #     positions = nx.spring_layout(G, weight="weight", k=3, iterations=800,
            #                                  seed=42)

        else:
            if dimensions == 3:
                # Use circular layout and add z=0 for 3D
                positions_2d = nx.circular_layout(G)
                positions = {
                    node: [pos[0], pos[1], 0] for node, pos in positions_2d.items()
                }
            else:
                positions = nx.circular_layout(G)

        # Extract node positions
        node_x = [positions[node][0] for node in active_feature_indices]
        node_y = [positions[node][1] for node in active_feature_indices]
        if dimensions == 3:
            node_z = [positions[node][2] for node in active_feature_indices]

        # Node representation
        max_dimensionality = feature_dimensionalities[active_feature_indices].max()

        if dimensions == 3:
            # Convert opacity values for 3D (plotly has issues with tensor types)
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
                    marker=dict(
                        size=5,
                        color="black",
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

        # Edge representation - individual traces to enable hover info
        # [11.02.26 | OliverSieweke] TODO: Check if this can be optimized for in the G loops
        for i_idx, i_node in enumerate(active_feature_indices):
            for j_idx, j_node in enumerate(
                active_feature_indices[i_idx + 1 :], start=i_idx + 1
            ):
                interference = feature_interferences[i_node, j_node]

                if abs(interference) > min_edge_interference:
                    x_coords = [positions[i_node][0], positions[j_node][0]]
                    y_coords = [positions[i_node][1], positions[j_node][1]]

                    if interference > 0:
                        edge_color = "rgba(255, 102, 102, 0.6)"
                    else:
                        edge_color = "rgba(51, 102, 204, 0.6)"

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

        # Update subplot axes
        if dimensions == 3:
            # Update 3D scene for this subplot
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

    # Update layout
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
    """Backwards compatibility wrapper for plot_feature_geometry with dimensions=3."""
    return plot_feature_geometry(
        models,
        min_edge_interference=min_edge_interference,
        feature_dimensionality_threshold=feature_dimensionality_threshold,
        dimensionality_range=dimensionality_range,
        dimensions=3,
    )
