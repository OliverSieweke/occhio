"""Plots for per-feature representation metrics and their distributions.

This module provides single-metric panels and a combined 4x2 composite view for:
- superposition indicator (ρmm)
- feature dimensionalities
- feature norms
- total feature interferences
"""

from typing import ClassVar

import numpy as np
import plotly.colors
import plotly.graph_objects as go

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core import CompositePlot
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.composite_plot import Span
from occhio.visualization.core.figure_wrappers import FigureProxy


class SuperpositionIndicatorPlot(SinglePlot):
    """Indicator gauge for the superposition metric (ρmm).

    Use case:
        Quick at-a-glance view of degree of superposition in the model.

    Data:
        - `model.superposition`: Mean max absolute cosine similarity (ρmm).
          Range 0-1, where 0 = no superposition (orthogonal features).

    Visualization:
        Plotly Indicator with gauge showing value from 0 to 1.

    Customization:
        - `title`: Indicator title (default: "Superposition (ρmm)").
    """

    n_render_axes = 0
    subplot_type = "domain"

    def __init__(self, title: str = "Superposition"):
        self.title = title

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        value = model.superposition.detach().cpu().item()

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={"text": self.title, "font": {"size": 14}},
                number={"valueformat": ".3f", "font": {"size": 18}},
            )
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


plot_superposition_indicator = SuperpositionIndicatorPlot()


class _BaseFeatureMetricPlot(SinglePlot):
    """Base panel for a per-feature metric.

    Use case:
        Shared rendering for feature-level statistics pulled from a ToyModel.

    Data:
        - One `ToyModel` tensor metric of shape `(n_features,)`.

    Visualization:
        Implemented by subclasses as a bar chart by index or a histogram.

    Customization:
        - `color`: Trace color (default: steel blue).
    """

    n_render_axes = 0

    metric_label: ClassVar[str]
    metric_property: ClassVar[str]

    def __init__(self, color: str = "#4C78A8"):
        self.color = color

    def _metric_numpy(self, model: ToyModel):
        return getattr(model, self.metric_property).detach().cpu().numpy()

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.55)", ticksuffix="  "
        )


class _FeatureMetricByIndexPlot(_BaseFeatureMetricPlot):
    """Bar panel for a per-feature metric indexed by feature id.

    Use case:
        Inspect which specific features stand out on a chosen metric.

    Data:
        - One `ToyModel` metric tensor `(n_features,)`.

    Visualization:
        Bar chart with feature index on x-axis and metric value on y-axis.

    Customization:
        - `color`: Bar color (default: steel blue).
    """

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        values = self._metric_numpy(model)
        feature_indices = list(range(model.n_features))

        fig.add_trace(
            go.Bar(
                x=feature_indices,
                y=values,
                marker_color=self.color,
                hovertemplate="Feature: %{x}<br>Value: %{y:.4f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text="Index")
        fig.update_yaxes(title_text=self.metric_label)


class _FeatureMetricDistributionPlot(_BaseFeatureMetricPlot):
    """Histogram panel for a per-feature metric distribution.

    Use case:
        Check global distribution shape (e.g. skew, spread, multimodality).

    Data:
        - One `ToyModel` metric tensor `(n_features,)`.

    Visualization:
        Histogram of metric values across all features.

    Customization:
        - `color`: Histogram color (default: steel blue).
        - `bins`: Number of bins (default: 25).
    """

    def __init__(self, color: str = "#4C78A8", bins: int = 25):
        super().__init__(color=color)
        self.bins = bins

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        values = self._metric_numpy(model)

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=self.bins,
                marker_color=self.color,
                opacity=0.85,
                hovertemplate="Value: %{x:.4f}<br>Count: %{y}<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text=self.metric_label)
        # fig.update_yaxes(title_text="Count")


class FeatureDimensionalityByIndexPlot(_FeatureMetricByIndexPlot):
    """Feature dimensionality by feature index.

    Use case:
        Identify which features occupy higher/lower effective dimensionality.

    Data:
        - `model.feature_dimensionalities`: Effective dimensionality per feature.

    Visualization:
        Bar chart with feature index on x-axis and dimensionality on y-axis.

    Customization:
        - `color`: Bar color (default: steel blue).
    """

    metric_label = "Dimensionality"
    metric_property = "feature_dimensionalities"


class FeatureDimensionalityDistributionPlot(_FeatureMetricDistributionPlot):
    """Distribution of feature dimensionality across features.

    Use case:
        Spot modality or heavy tails in feature dimensionalities.

    Data:
        - `model.feature_dimensionalities`: Effective dimensionality per feature.

    Visualization:
        Histogram of dimensionality values.

    Customization:
        - `color`: Histogram color (default: steel blue).
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Dimensionality"
    metric_property = "feature_dimensionalities"


class FeatureNormByIndexPlot(_FeatureMetricByIndexPlot):
    """Feature norm by feature index.

    Use case:
        Inspect variation in embedding magnitudes across features.

    Data:
        - `model.feature_norms`: L2 norm per feature embedding.

    Visualization:
        Bar chart with feature index on x-axis and norm on y-axis.

    Customization:
        - `color`: Bar color (default: green).
    """

    metric_label = "Norm"
    metric_property = "feature_norms"

    def __init__(self, color: str = "#59A14F"):
        super().__init__(color=color)


class FeatureNormDistributionPlot(_FeatureMetricDistributionPlot):
    """Distribution of feature norms across features.

    Use case:
        Check global spread and concentration of embedding magnitudes.

    Data:
        - `model.feature_norms`: L2 norm per feature embedding.

    Visualization:
        Histogram of norm values.

    Customization:
        - `color`: Histogram color (default: green).
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Norm"
    metric_property = "feature_norms"

    def __init__(self, color: str = "#59A14F", bins: int = 25):
        super().__init__(color=color, bins=bins)


class FeatureInterferenceByIndexPlot(_FeatureMetricByIndexPlot):
    """Total feature interference by feature index.

    Use case:
        See which features interfere most with the rest of the representation.

    Data:
        - `model.total_feature_interferences`: Sum of squared off-diagonal interference per feature.

    Visualization:
        Bar chart with feature index on x-axis and total interference on y-axis.

    Customization:
        - `color`: Bar color (default: orange-red).
    """

    metric_label = "Interference"
    metric_property = "total_feature_interferences"

    def __init__(self, color: str = "#E15759"):
        super().__init__(color=color)


class FeatureInterferenceDistributionPlot(_FeatureMetricDistributionPlot):
    """Distribution of total feature interference across features.

    Use case:
        Detect heterogeneity or multimodality in feature interference levels.

    Data:
        - `model.total_feature_interferences`: Sum of squared off-diagonal interference per feature.

    Visualization:
        Histogram of total interference values.

    Customization:
        - `color`: Histogram color (default: orange-red).
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Interference"
    metric_property = "total_feature_interferences"

    def __init__(self, color: str = "#E15759", bins: int = 25):
        super().__init__(color=color, bins=bins)


plot_feature_dimensionality_by_index = FeatureDimensionalityByIndexPlot()
plot_feature_dimensionality_distribution = FeatureDimensionalityDistributionPlot()
plot_feature_norm_by_index = FeatureNormByIndexPlot()
plot_feature_norm_distribution = FeatureNormDistributionPlot()
plot_feature_interference_by_index = FeatureInterferenceByIndexPlot()
plot_feature_interference_distribution = FeatureInterferenceDistributionPlot()


class _BaseFeatureMetricOverlayPlot(SinglePlot):
    """Base class for overlaying feature metric distributions across a ModelGrid.

    Use case:
        Compare distribution shapes across multiple models in a grid.

    Data:
        - One `ToyModel` tensor metric of shape `(n_features,)` per model in grid.

    Visualization:
        Overlayed histograms with one trace per model, using distinct colors.

    Customization:
        - `bins`: Number of bins (default: 25).
    """

    n_render_axes = 1

    metric_label: ClassVar[str]
    metric_property: ClassVar[str]

    def __init__(self, bins: int = 50):
        self.bins = bins

    def _metric_numpy(self, model: ToyModel):
        return getattr(model, self.metric_property).detach().cpu().numpy()

    def render(self, fig: FigureProxy, models: ToyModel | ModelGrid) -> None:
        # Convert single model to a grid for uniform handling
        if isinstance(models, ToyModel):
            models = ModelGrid.from_iterable([models])

        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"{self.__class__.__name__} requires a ModelGrid, got {type(models).__name__}"
            )

        render_axis = models.axes[0]
        colors = plotly.colors.qualitative.Plotly

        # Compute global range across all models for uniform bin edges
        all_values = [self._metric_numpy(model) for model in models]
        global_min = min(v.min() for v in all_values)
        global_max = max(v.max() for v in all_values)
        bin_size = (global_max - global_min) / self.bins

        # Collect mean values for overlap detection
        mean_values = [(i, all_values[i].mean()) for i in range(len(all_values))]

        for i, (model, axis_value) in enumerate(zip(models, render_axis.values)):
            values = all_values[i]
            color = colors[i % len(colors)]

            # Format axis value: use .4g for numeric, str for non-numeric
            if isinstance(axis_value, (int, float)):
                formatted_value = f"{axis_value:.4g}"
            else:
                formatted_value = str(axis_value)

            fig.add_trace(
                go.Histogram(
                    x=values,
                    xbins=dict(
                        start=global_min,
                        end=global_max,
                        size=bin_size,
                    ),
                    marker_color=color,
                    opacity=0.6,
                    name=str(axis_value),
                    hovertemplate=f"{self.metric_label}: %{{x:.4f}}<br>Count: %{{y}}<extra>{render_axis.label}: {formatted_value}</extra>",
                    legendgroup=f"group_{i}",
                )
            )

        # Add mean lines as scatter traces (traces update with slider frames,
        # unlike vlines which are layout shapes and don't update)
        # Compute y positions for text labels with offsets to avoid overlap
        data_range = global_max - global_min
        threshold = data_range * 0.02  # 2% of range = close enough to offset

        # Assign vertical positions based on proximity of mean values
        positions = {}  # {index: y_position_level (0, 1, 2, ...)}
        sorted_means = sorted(mean_values, key=lambda x: x[1])

        for i, (idx, mean_val) in enumerate(sorted_means):
            overlaps_with = [
                pos_idx
                for pos_idx, pos_mean in sorted_means[:i]
                if abs(pos_mean - mean_val) < threshold
            ]
            if overlaps_with:
                max_level = max(positions[j] for j in overlaps_with)
                positions[idx] = max_level + 1
            else:
                positions[idx] = 0

        # Compute actual max bin count across all histograms for label positioning
        max_bin_count = 0
        for values in all_values:
            counts, _ = np.histogram(
                values, bins=self.bins, range=(global_min, global_max)
            )
            max_bin_count = max(max_bin_count, counts.max())

        # Base label height and offset per level
        label_y_base = max_bin_count * 0.9
        label_y_offset = max_bin_count * 0.08

        for idx, mean_value in mean_values:
            color = colors[idx % len(colors)]
            level = positions[idx]
            label_y = label_y_base + level * label_y_offset

            # Draw vertical line + text label as a single scatter trace
            fig.add_trace(
                go.Scatter(
                    x=[mean_value, mean_value, mean_value],
                    y=[0, label_y, label_y],
                    mode="lines+text",
                    line=dict(color=color, width=2, dash="dash"),
                    text=["", "", f"μ={mean_value:.3g}"],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.update_xaxes(title_text=self.metric_label)
        fig.update_yaxes(title_text="Count")

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay",
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.55)", ticksuffix="  "
        )


class FeatureDimensionalityDistributionOverlayPlot(_BaseFeatureMetricOverlayPlot):
    """Overlayed distribution of feature dimensionality across a ModelGrid.

    Use case:
        Compare how feature dimensionality distributions vary across models.

    Data:
        - `model.feature_dimensionalities`: Effective dimensionality per feature
          for each model in the grid.

    Visualization:
        Overlayed histograms with one trace per model. X-axis is dimensionality,
        Y-axis is count. Each model gets a distinct color from the Plotly palette.

    Customization:
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Dimensionality"
    metric_property = "feature_dimensionalities"


class FeatureNormDistributionOverlayPlot(_BaseFeatureMetricOverlayPlot):
    """Overlayed distribution of feature norms across a ModelGrid.

    Use case:
        Compare how feature norm distributions vary across models.

    Data:
        - `model.feature_norms`: L2 norm per feature embedding for each model
          in the grid.

    Visualization:
        Overlayed histograms with one trace per model. X-axis is norm,
        Y-axis is count. Each model gets a distinct color from the Plotly palette.

    Customization:
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Norm"
    metric_property = "feature_norms"


class FeatureInterferenceDistributionOverlayPlot(_BaseFeatureMetricOverlayPlot):
    """Overlayed distribution of feature interference across a ModelGrid.

    Use case:
        Compare how feature interference distributions vary across models.

    Data:
        - `model.total_feature_interferences`: Sum of squared off-diagonal
          interference per feature for each model in the grid.

    Visualization:
        Overlayed histograms with one trace per model. X-axis is interference,
        Y-axis is count. Each model gets a distinct color from the Plotly palette.

    Customization:
        - `bins`: Number of bins (default: 25).
    """

    metric_label = "Interference"
    metric_property = "total_feature_interferences"


plot_feature_representation = CompositePlot(
    layout=[
        [Span(plot_superposition_indicator, colspan=2)],
        [
            plot_feature_dimensionality_by_index,
            plot_feature_dimensionality_distribution,
        ],
        [
            plot_feature_norm_by_index,
            plot_feature_norm_distribution,
        ],
        [
            plot_feature_interference_by_index,
            plot_feature_interference_distribution,
        ],
    ],
    row_heights=[0.5, 1, 1, 1],
    share_axes_across_facets=True,
)

plot_feature_representation_overlay = CompositePlot(
    layout=[
        [FeatureDimensionalityDistributionOverlayPlot()],
        [FeatureNormDistributionOverlayPlot()],
        [FeatureInterferenceDistributionOverlayPlot()],
    ],
)
