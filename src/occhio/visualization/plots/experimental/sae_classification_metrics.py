"""SAE metrics visualization.

This module provides plotting utilities for visualizing SAE evaluation metrics
from toy models, including F1, precision, recall, explained variance, L0, and dead latents.
"""

from dataclasses import dataclass
from typing import Literal

import plotly.colors
import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


@dataclass
class MetricConfig:
    name: str
    values: dict[str, float]
    color: str


class SAEClassificationMetricsPlot(SinglePlot):
    """Plot SAE evaluation metrics as grouped bar charts.

    Creates a visualization with:
    - Main bars for classification metrics (F1, precision, recall) and reconstruction metrics
      (explained variance, L0)
    - Separate display for dead latent counts

    Example::

        from occhio import ToyModel
        from occhio.visualization.plots import SAEMetricsPlot

        # Create and train toy model with SAEs
        tm = ToyModel(distribution, ae)
        tm.fit(n_epochs=1000)
        tm.train_saes({"default": sae1, "topk": sae2})
        tm.evaluate_saes()

        # Plot metrics
        SAEMetricsPlot()(tm)
    """

    sae_labels: list[str] | None
    group_by: Literal["sae", "metric"] = "sae"

    def __init__(
        self,
        sae_labels: list[str] | None = None,
        group_by: Literal["sae", "metric"] = "sae",
    ):
        """Initialize SAEMetricsPlot.

        Args:
            sae_labels: Optional list of SAE labels to plot. If None, plots all SAEs.
        """
        self.sae_labels = sae_labels
        self.group_by = group_by

    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
    ) -> None:
        """Render the SAE metrics bar chart.

        Args:
            fig: FigureProxy for adding traces to the subplot.
            model: ToyModel containing trained and evaluated SAEs.
        """
        # Y-Axis Range -----------------------------------------------------------------
        fig.update_yaxes(range=[0, 1])

        # Sanity Checks ----------------------------------------------------------------
        model_sae_labels: list[str] = (
            self.sae_labels if self.sae_labels is not None else list(model.saes.keys())
        )

        if not model_sae_labels:
            fig.add_annotation(
                text="No SAEs trained on the model.<br>Call [model|grid].train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        unmatched_model_sae_labels = [
            label for label in model_sae_labels if label not in model.saes
        ]
        if unmatched_model_sae_labels:
            fig.add_annotation(
                text=f"SAEs {unmatched_model_sae_labels} not trained on the model.<br>"
                f"Available trained SAEs: {', '.join(model.saes.keys()) or 'None'}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        evaluated_sae_labels = [
            label for label in model.saes if model.saes[label].results
        ]
        unevaluated_model_sae_labels = [
            label for label in model_sae_labels if label not in evaluated_sae_labels
        ]
        if unevaluated_model_sae_labels:
            fig.add_annotation(
                text=f"SAEs {unevaluated_model_sae_labels} not evaluated on the model.<br>"
                f"Call [model|grid].evaluate_saes() first.<br>"
                f"Available evaluated SAEs: {', '.join(evaluated_sae_labels) or 'None'}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Bar Plot ---------------------------------------------------------------------
        metrics = [
            MetricConfig("Accuracy", model.saes_accuracy, "#BDC3C7"),
            MetricConfig("Precision", model.saes_precision, "#3498DB"),
            MetricConfig("Recall", model.saes_recall, "#E67E22"),
            MetricConfig("F1", model.saes_f1_score, "#9B59B6"),
        ]
        match self.group_by:
            case "sae":
                for metric in metrics:
                    values = [metric.values[label] for label in model_sae_labels]
                    fig.add_trace(
                        go.Bar(
                            name=metric.name,
                            x=model_sae_labels,
                            y=values,
                            marker_color=metric.color,
                            text=[f"{v:.3f}" for v in values],
                            textposition="outside",
                            hovertemplate=f"{metric.name}: %{{y:.3f}}<extra>%{{x}}</extra>",
                        )
                    )

            case "metric":
                colors = plotly.colors.qualitative.Plotly
                for i, sae in enumerate(model_sae_labels):
                    values = [metric.values.get(sae, 0) for metric in metrics]
                    fig.add_trace(
                        go.Bar(
                            name=sae,
                            x=[metric.name for metric in metrics],
                            y=values,
                            marker_color=colors[i % len(colors)],
                            text=[f"{v:.3f}" for v in values],
                            textposition="outside",
                            hovertemplate=f"%{{x}}: %{{y:.3f}}<extra>{sae}</extra>",
                        )
                    )

        # Grid lines -------------------------------------------------------------------
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.add_hline(y=1, line_color="rgba(211, 211, 211, 0.8)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(211, 211, 211, 0.8)", line_width=1)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.2,
            bargroupgap=0.05,
        )
