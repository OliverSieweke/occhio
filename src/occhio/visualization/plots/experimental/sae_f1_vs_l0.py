"""SAE F1 vs L0 scatter plot visualization.

This module provides a scatter plot for comparing SAE F1 scores against L0 sparsity
across models in a 1D ModelGrid, with each model shown as a different colored series.
"""

import plotly.colors
import plotly.graph_objects as go

from occhio import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class SAEF1vsL0Plot(SinglePlot):
    """Scatter plot of F1 Score (x) vs L0 (y) for SAEs across models.

    Use case:
        Visualize the trade-off between classification performance (F1) and
        sparsity (L0) for SAEs trained on different models in a grid.

    Data:
        - `model.saes_f1_score`: dict[str, float] - F1 scores per SAE
        - `model.saes_l0`: dict[str, float] - L0 sparsity per SAE

    Visualization:
        - Scatter plot with F1 on x-axis, L0 on y-axis
        - Each model in the grid is a different color
        - Each point is an SAE, with label shown on hover

    Customization:
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 1

    def __init__(self, sae_labels: list[str] | None = None):
        """Initialize SAEF1vsL0Plot.

        Args:
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the F1 vs L0 scatter plot.

        Args:
            fig: FigureProxy for adding traces.
            models: 1D ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAEF1vsL0Plot requires a ModelGrid, got {type(models).__name__}"
            )

        if len(models.axes) < 1:
            fig.add_annotation(
                text="SAEF1vsL0Plot requires a 1D ModelGrid.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        render_axis = models.axes[0]

        # Find common SAE labels across all models
        common_sae_labels = set.intersection(
            *({label for label in model.saes} for model in models)
        )

        if not common_sae_labels:
            fig.add_annotation(
                text="No SAE trained on all models.<br>Call grid.train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        sae_labels = (
            [label for label in self.sae_labels if label in common_sae_labels]
            if self.sae_labels
            else sorted(common_sae_labels)
        )

        if not sae_labels:
            fig.add_annotation(
                text=f"Requested SAEs not found.<br>Available: {', '.join(common_sae_labels)}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Check SAEs are evaluated
        evaluated_sae_labels = set.intersection(
            *(
                {label for label in model.saes if model.saes[label].results}
                for model in models
            )
        )
        unevaluated = set(sae_labels) - evaluated_sae_labels
        if unevaluated:
            fig.add_annotation(
                text=f"SAEs {unevaluated} not evaluated.<br>Call grid.evaluate_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Plot each model as a separate series
        colors = plotly.colors.qualitative.Plotly
        for i, model in enumerate(models):
            axis_val = render_axis.values[i]
            color = colors[i % len(colors)]

            f1_scores = [model.saes_f1_score[label] for label in sae_labels]
            l0_values = [model.saes_l0[label] for label in sae_labels]

            fig.add_trace(
                go.Scatter(
                    name=f"{render_axis.label}={axis_val}",
                    x=f1_scores,
                    y=l0_values,
                    mode="markers",
                    marker=dict(color=color, size=8),
                    text=sae_labels,
                    hovertemplate="F1: %{x:.3f}<br>L0: %{y:.2f}<extra>%{text}<br>"
                    + f"{render_axis.label}={axis_val}</extra>",
                )
            )

        # Axis labels and styling (L0 reversed so lower=better is up)
        fig.update_xaxes(title_text="F1 Score", range=[0, 1])
        fig.update_yaxes(title_text="L0", autorange="reversed")

        # Grid lines
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(211, 211, 211, 0.8)", gridwidth=1
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


plot_sae_f1_vs_l0 = SAEF1vsL0Plot()
