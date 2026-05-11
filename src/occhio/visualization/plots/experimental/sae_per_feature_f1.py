"""SAE per-feature F1 stratified by activation frequency.

This module provides a box/violin plot showing how SAE F1 scores vary across
feature frequency buckets, answering: "Does the SAE perform well on rare features?"
"""

from typing import Literal

import numpy as np
import plotly.colors
import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class SAEPerFeatureF1Plot(SinglePlot):
    """Box/violin plot of per-feature F1 scores stratified by feature frequency.

    Use case:
        Diagnose whether an SAE is only good at recovering common features or
        also handles rare ones. A high mean F1 can mask failure on rare features.

    Data:
        - `model.saes_per_feature_f1`: dict[str, np.ndarray] - F1 per ground-truth feature
        - `model.feature_frequencies`: Tensor (n_features,) - activation frequency per feature

    Visualization:
        - X-axis: Feature frequency buckets (percentile-based)
        - Y-axis: Per-feature F1 score [0, 1]
        - One box/violin per SAE within each bucket, color-coded by SAE label

    Customization:
        - `n_buckets`: Number of frequency buckets (default: 10)
        - `plot_type`: "box" or "violin" (default: "box")
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 0

    def __init__(
        self,
        n_buckets: int = 10,
        plot_type: Literal["box", "violin"] = "violin",
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAEPerFeatureF1Plot.

        Args:
            n_buckets: Number of frequency buckets (default: 10).
            plot_type: "box" or "violin" (default: "box").
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.n_buckets = n_buckets
        self.plot_type = plot_type
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
    ) -> None:
        """Render the per-feature F1 by frequency bucket plot.

        Args:
            fig: FigureProxy for adding traces.
            model: ToyModel with trained and evaluated SAEs.
        """
        # Sanity checks
        model_sae_labels = (
            self.sae_labels if self.sae_labels is not None else list(model.saes.keys())
        )

        if not model_sae_labels:
            fig.add_annotation(
                text="No SAEs trained.<br>Call model.train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Check all requested SAEs exist
        missing = [label for label in model_sae_labels if label not in model.saes]
        if missing:
            fig.add_annotation(
                text=f"SAEs {missing} not found.<br>"
                f"Available: {', '.join(model.saes.keys()) or 'None'}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Check SAEs are evaluated with per-feature metrics
        per_feature_f1 = model.saes_per_feature_f1
        unevaluated = [
            label for label in model_sae_labels if label not in per_feature_f1
        ]
        if unevaluated:
            fig.add_annotation(
                text=f"SAEs {unevaluated} not evaluated.<br>"
                "Call model.evaluate_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Get feature frequencies and compute bucket assignments
        freqs = model.feature_frequencies.detach().cpu().numpy()
        percentiles = np.linspace(0, 100, self.n_buckets + 1)
        bucket_edges = np.percentile(freqs, percentiles)
        # Assign features to buckets based on their frequency
        bucket_indices = np.digitize(freqs, bucket_edges[1:-1])  # 0 to n_buckets-1

        # Create bucket labels
        bucket_labels = []
        for i in range(self.n_buckets):
            low_pct = int(percentiles[i])
            high_pct = int(percentiles[i + 1])
            bucket_labels.append(f"{low_pct}-{high_pct}%")

        # Plot each SAE
        colors = plotly.colors.qualitative.Plotly
        for sae_idx, label in enumerate(model_sae_labels):
            f1_scores = per_feature_f1[label]
            color = colors[sae_idx % len(colors)]

            # Collect F1 values for each bucket
            for bucket_idx in range(self.n_buckets):
                mask = bucket_indices == bucket_idx
                bucket_f1 = f1_scores[mask]

                if len(bucket_f1) == 0:
                    continue

                x_values = [bucket_labels[bucket_idx]] * len(bucket_f1)

                if self.plot_type == "box":
                    fig.add_trace(
                        go.Box(
                            name=label,
                            x=x_values,
                            y=bucket_f1,
                            marker_color=color,
                            boxpoints="outliers",
                            legendgroup=label,
                            showlegend=(bucket_idx == 0),
                            hovertemplate="F1: %{y:.3f}<extra>%{x}<br>"
                            + label
                            + "</extra>",
                        )
                    )
                else:  # violin
                    fig.add_trace(
                        go.Violin(
                            name=label,
                            x=x_values,
                            y=bucket_f1,
                            line_color=color,
                            fillcolor=color,
                            opacity=0.6,
                            legendgroup=label,
                            showlegend=(bucket_idx == 0),
                            hovertemplate="F1: %{y:.3f}<extra>%{x}<br>"
                            + label
                            + "</extra>",
                        )
                    )

        # Axis styling
        fig.update_xaxes(
            title_text="Feature Frequency (percentile bucket)",
            categoryorder="array",
            categoryarray=bucket_labels,
        )
        fig.update_yaxes(title_text="F1 Score", range=[0, 1.05])

        # Grid lines
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.add_hline(y=1, line_color="rgba(211, 211, 211, 0.8)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(211, 211, 211, 0.8)", line_width=1)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            boxmode="group",
            violinmode="group",
        )


plot_sae_per_feature_f1 = SAEPerFeatureF1Plot()


class SAEPerFeatureF1DistributionPlot(SinglePlot):
    """Box/violin plot of overall per-feature F1 distribution (no frequency bucketing).

    Use case:
        Visualize the overall distribution of per-feature F1 scores across all
        ground-truth features, comparing multiple SAEs.

    Data:
        - `model.saes_per_feature_f1`: dict[str, np.ndarray] - F1 per ground-truth feature

    Visualization:
        - X-axis: SAE labels
        - Y-axis: Per-feature F1 score [0, 1]
        - One box/violin per SAE

    Customization:
        - `plot_type`: "box" or "violin" (default: "box")
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 0

    def __init__(
        self,
        plot_type: Literal["box", "violin"] = "box",
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAEPerFeatureF1DistributionPlot.

        Args:
            plot_type: "box" or "violin" (default: "box").
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.plot_type = plot_type
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
    ) -> None:
        """Render the per-feature F1 distribution plot.

        Args:
            fig: FigureProxy for adding traces.
            model: ToyModel with trained and evaluated SAEs.
        """
        model_sae_labels = (
            self.sae_labels if self.sae_labels is not None else list(model.saes.keys())
        )

        if not model_sae_labels:
            fig.add_annotation(
                text="No SAEs trained.<br>Call model.train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        missing = [label for label in model_sae_labels if label not in model.saes]
        if missing:
            fig.add_annotation(
                text=f"SAEs {missing} not found.<br>"
                f"Available: {', '.join(model.saes.keys()) or 'None'}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        per_feature_f1 = model.saes_per_feature_f1
        unevaluated = [
            label for label in model_sae_labels if label not in per_feature_f1
        ]
        if unevaluated:
            fig.add_annotation(
                text=f"SAEs {unevaluated} not evaluated.<br>"
                "Call model.evaluate_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        colors = plotly.colors.qualitative.Plotly
        for sae_idx, label in enumerate(model_sae_labels):
            f1_scores = per_feature_f1[label]
            color = colors[sae_idx % len(colors)]

            if self.plot_type == "box":
                fig.add_trace(
                    go.Box(
                        name=label,
                        y=f1_scores,
                        marker_color=color,
                        boxpoints="outliers",
                        hovertemplate="F1: %{y:.3f}<extra>" + label + "</extra>",
                    )
                )
            else:  # violin
                fig.add_trace(
                    go.Violin(
                        name=label,
                        y=f1_scores,
                        line_color=color,
                        fillcolor=color,
                        opacity=0.6,
                        hovertemplate="F1: %{y:.3f}<extra>" + label + "</extra>",
                    )
                )

        fig.update_xaxes(title_text="SAE")
        fig.update_yaxes(title_text="F1 Score", range=[0, 1.05])

        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.add_hline(y=1, line_color="rgba(211, 211, 211, 0.8)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(211, 211, 211, 0.8)", line_width=1)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


plot_sae_per_feature_f1_distribution = SAEPerFeatureF1DistributionPlot()
