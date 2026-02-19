from enum import Enum
import numpy as np
import plotly.graph_objects as go
from occhio.model_grid import ModelGrid
import numpy as np

class GeometryPlotComponent(Enum):
    HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES = "hidden-dimensions-per-embedded-features"
    EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS = "embedded-features_per-hidden_dimensions"
    FEATURE_DIMENSIONALITIES = "feature-dimensionalities"
    MEAN_FEATURE_DIMENSIONALITIES = "mean-feature-dimensionalities"
    TOTAL_FEATURE_DIMENSIONALITIES = "total-feature-dimensionalities"
    GEOMETRIES = "geometries"


def plot_geometry(
    model_grid: ModelGrid, components: set[GeometryPlotComponent] | None = None
):
    x_axis_index = np.where(np.array(model_grid.shape) != 1)[0]

    assert len(x_axis_index) == 1, (
        f"plot_phase_change only supports ModelGrids with exactly one non-singleton dimension."
        f"Got shape {model_grid.shape} with {int(len(x_axis_index))} non-singleton dims at indices {x_axis_index})."
    )
    x_axis_index: int = int(x_axis_index[0])

    if components is None:
        components = set(GeometryPlotComponent) - {
            GeometryPlotComponent.EMBEDDED_FEATURES_PER_HIDDEN_DIMENSIONS
        }

    fig = go.Figure()

    if GeometryPlotComponent.HIDDEN_DIMENSIONS_PER_EMBEDDED_FEATURES in components:
        fig.add_trace(
            go.Scatter(
                x=model_grid.axes[x_axis_index].values,
                y=[
                    model.hidden_dimensions_per_embedded_features.cpu()
                    for model in np.squeeze(model_grid.models)
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
                x=model_grid.axes[x_axis_index].values,
                y=[
                    model.embedded_features_per_hidden_dimensions.cpu()
                    for model in np.squeeze(model_grid.models)
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

        for i, model in enumerate(np.squeeze(model_grid.models)):
            x_vals.extend(
                [model_grid.axes[x_axis_index].values[i]]
                * len(model.feature_dimensionalities)
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
                marker=dict(size=1.5, color="#333333", opacity=0.4),
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

        for i, model in enumerate(np.squeeze(model_grid.models)):
            x_vals.append(model_grid.axes[x_axis_index].values[i])
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

        for i, model in enumerate(np.squeeze(model_grid.models)):
            x_vals.append(model_grid.axes[x_axis_index].values[i])
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
        xaxis_title=model_grid.axes[x_axis_index].label,
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
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(r=100),
    )

    return fig
