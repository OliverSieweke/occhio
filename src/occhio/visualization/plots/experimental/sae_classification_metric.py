"""SAE classification metrics grid visualization.

This module provides line plots for visualizing SAE classification metrics
across a ModelGrid, with each model as a point on the x-axis.
"""

from dataclasses import dataclass
from enum import Enum

import plotly.colors
import plotly.graph_objects as go

from occhio import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core import CompositePlot
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


@dataclass
class MetricConfig:
    name: str
    model_property: str
    color: str


class SAEClassificationMetric(Enum):
    ACCURACY = MetricConfig("Accuracy", "saes_accuracy", "#BDC3C7")
    PRECISION = MetricConfig("Precision", "saes_precision", "#3498DB")
    RECALL = MetricConfig("Recall", "saes_recall", "#E67E22")
    F1 = MetricConfig("F1", "saes_f1_score", "#9B59B6")


class SAEClassificationMetricPlot(SinglePlot):
    """Plot SAE F1 scores as line chart across a ModelGrid.

    Each SAE is a line, x-axis is the first grid axis, y-axis is F1 score.

    Example::

        from occhio import ModelGrid
        from occhio.visualization.plots import SAEClassificationMetricsGridPlot

        grid = ModelGrid(create_model, axes=[Axis("sparsity", [0.1, 0.2, 0.3])])
        grid.fit(n_epochs=1000)
        grid.train_saes({"sae1": sae1, "sae2": sae2})
        grid.evaluate_saes()

        SAEClassificationMetricsGridPlot()(grid)
    """

    metric: SAEClassificationMetric
    sae_labels: list[str] | None

    n_render_axes = 1

    def __init__(
        self,
        metric: SAEClassificationMetric = SAEClassificationMetric.F1,
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAEClassificationMetricsGridPlot.

        Args:
            sae_labels: Optional list of SAE labels to plot. If None, plots all SAEs.
        """
        self.sae_labels = sae_labels
        self.metric = metric

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the F1 score line chart.

        Args:
            fig: FigureProxy for adding traces.
            models: ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAEClassificationMetricsGridPlot requires a ModelGrid, got {type(models).__name__}"
            )

        render_axis = models.axes[0]

        # Y-Axis Range -----------------------------------------------------------------
        fig.update_yaxes(range=[0, 1], title_text=self.metric.value.name)

        # Sanity checks ----------------------------------------------------------------
        sae_labels = self.sae_labels
        common_sae_labels = set.intersection(
            *({label for label in model.saes} for model in models)
        )

        if sae_labels is None:
            if not common_sae_labels:
                fig.add_annotation(
                    text="No SAE trained on all models.<br>Call grid.train_saes() first.",
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=12, color="firebrick"),
                )
                return
            else:
                sae_labels = list(common_sae_labels)

        else:
            unmatched_sae_labels = set.union(
                *(
                    {label for label in sae_labels if label not in common_sae_labels}
                    for model in models
                )
            )
            if unmatched_sae_labels:
                fig.add_annotation(
                    text=f"SAEs {unmatched_sae_labels} not trained on all models.<br>"
                    f"Available trained SAEs: {', '.join(common_sae_labels) or 'None'}",
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=12, color="firebrick"),
                )
                return

        evaluated_sae_labels = set.intersection(
            *(
                {label for label in model.saes if model.saes[label].results}
                for model in models
            )
        )
        unevaluated_model_sae_labels = set(sae_labels) - evaluated_sae_labels

        if unevaluated_model_sae_labels:
            fig.add_annotation(
                text=f"SAEs {unevaluated_model_sae_labels} not evaluated on all models.<br>"
                f"Call grid.evaluate_saes() first.<br>"
                f"Available evaluated SAEs: {', '.join(evaluated_sae_labels) or 'None'}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        colors = plotly.colors.qualitative.Plotly
        for i, sae_label in enumerate(sae_labels):
            fig.add_trace(
                go.Scatter(
                    name=sae_label,
                    x=render_axis.values,
                    y=[
                        getattr(model, self.metric.value.model_property)[sae_label]
                        for model in models
                    ],
                    mode="lines+markers",
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"{self.metric.value.name}: %{{y:.3f}}<extra>SAE:{sae_label}<br>{render_axis.label}: %{{x}}</extra>",
                )
            )
        # Grid lines -------------------------------------------------------------------
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.add_hline(y=1, line_color="rgba(211, 211, 211, 0.8)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(211, 211, 211, 0.8)", line_width=1)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


class SAEMetricsComparisonPlot(SinglePlot):
    """Plot multiple classification metrics for a single SAE across a ModelGrid.

    Each metric is a line, x-axis is the first grid axis, y-axis is the metric value.
    Uses consistent colors: Accuracy (gray), Precision (blue), Recall (orange), F1 (purple).

    Example::

        from occhio import ModelGrid
        from occhio.visualization.plots import SAEMetricsComparisonPlot

        grid = ModelGrid(create_model, axes=[Axis("sparsity", [0.1, 0.2, 0.3])])
        grid.fit(n_epochs=1000)
        grid.train_saes({"sae1": sae1, "sae2": sae2})
        grid.evaluate_saes()

        SAEMetricsComparisonPlot(sae_label="sae1")(grid)
    """

    sae_label: str
    metrics: list[SAEClassificationMetric]

    n_render_axes = 1

    def __init__(
        self,
        sae_label: str,
        metrics: list[SAEClassificationMetric] | None = None,
    ):
        """Initialize SAEMetricsComparisonPlot.

        Args:
            sae_label: The SAE label to plot metrics for.
            metrics: Optional list of metrics to plot. If None, plots all metrics.
        """
        self.sae_label = sae_label
        self.metrics = metrics if metrics is not None else list(SAEClassificationMetric)

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the metrics comparison line chart.

        Args:
            fig: FigureProxy for adding traces.
            models: ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAEMetricsComparisonPlot requires a ModelGrid, got {type(models).__name__}"
            )

        render_axis = models.axes[0]

        # Y-Axis Range -----------------------------------------------------------------
        fig.update_yaxes(range=[0, 1], title_text=self.sae_label)

        # Sanity checks ----------------------------------------------------------------
        sae_label = self.sae_label

        # Check SAE is trained on all models
        models_missing_sae = [
            i for i, model in enumerate(models) if sae_label not in model.saes
        ]
        if models_missing_sae:
            fig.add_annotation(
                text=f"SAE '{sae_label}' not trained on all models.<br>"
                f"Missing in models at indices: {models_missing_sae}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Check SAE is evaluated on all models
        models_not_evaluated = [
            i for i, model in enumerate(models) if not model.saes[sae_label].results
        ]
        if models_not_evaluated:
            fig.add_annotation(
                text=f"SAE '{sae_label}' not evaluated on all models.<br>"
                f"Call grid.evaluate_saes() first.<br>"
                f"Missing evaluations at indices: {models_not_evaluated}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Plot lines -------------------------------------------------------------------
        for metric in self.metrics:
            config = metric.value
            fig.add_trace(
                go.Scatter(
                    name=config.name,
                    x=render_axis.values,
                    y=[
                        getattr(model, config.model_property)[sae_label]
                        for model in models
                    ],
                    mode="lines+markers",
                    marker_color=config.color,
                    line_color=config.color,
                    hovertemplate=f"{config.name}: %{{y:.3f}}<extra>{sae_label}<br>{render_axis.label}: %{{x}}</extra>",
                )
            )

        # Grid lines -------------------------------------------------------------------
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.add_hline(y=1, line_color="rgba(211, 211, 211, 0.8)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(211, 211, 211, 0.8)", line_width=1)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


plot_sae_classification_metrics = CompositePlot(
    layout=[
        [
            SAEClassificationMetricPlot(SAEClassificationMetric.PRECISION),
            SAEClassificationMetricPlot(SAEClassificationMetric.RECALL),
        ],
        [
            SAEClassificationMetricPlot(SAEClassificationMetric.ACCURACY),
            SAEClassificationMetricPlot(SAEClassificationMetric.F1),
        ],
    ],
)
