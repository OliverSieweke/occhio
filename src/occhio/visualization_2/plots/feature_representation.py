"""Plots for per-feature representation metrics and their distributions.

This module provides single-metric panels and a combined 3x2 composite view for:
- feature dimensionalities
- feature norms
- total feature interferences
"""

from typing import ClassVar

import plotly.graph_objects as go
from torch import Tensor

from occhio.toy_model import ToyModel
from occhio.visualization_2.core import CompositePlot
from occhio.visualization_2.core.base_plot import SinglePlot
from occhio.visualization_2.core.figure_wrappers import FigureProxy


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
        fig.update_yaxes(title_text="Count")


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

plot_feature_representation = CompositePlot(
    layout=[
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
    share_axes_across_facets=True,
)
