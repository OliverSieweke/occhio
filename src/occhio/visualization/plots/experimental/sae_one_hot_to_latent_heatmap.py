"""SAE one-hot feature to latent activation heatmap visualization.

This module provides a plot for visualizing how one-hot input features are
mapped into SAE latent activations.
"""

import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class SAEOneHotToLatentHeatmapPlot(SinglePlot):
    """Heatmap of SAE latent activations for one-hot input features.

    Use case:
        Inspect how each one-hot input feature activates each SAE latent and
        identify sparse vs. distributed latent responses.

    Data:
        - `model.get_one_hot_embeddings()`: AE hidden representations for one-hot
          inputs with shape (n_features, n_hidden).
        - `model.saes[sae_label].sae.encode(...)`: SAE latent activations for each
          one-hot input with shape (n_features, n_sae_latents).

    Visualization:
        Heatmap with input feature index on X-axis and SAE latent index on Y-axis.
        Cell values are raw SAE latent activations (no normalization).

    Customization:
        - `sae_label`: Which SAE to plot (required).
        - `show_title`: Whether to include SAE label in Y-axis title (default: True).
        - `colorscale`: Plotly colorscale name (default: "Viridis").
    """

    n_render_axes = 0

    def __init__(
        self,
        sae_label: str,
        show_title: bool = True,
        colorscale: str = "Viridis",
    ):
        self.sae_label = sae_label
        self.show_title = show_title
        self.colorscale = colorscale

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        if not model.saes:
            fig.add_annotation(
                text="No SAEs trained.<br>Call model.train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        if self.sae_label not in model.saes:
            available = ", ".join(model.saes.keys())
            fig.add_annotation(
                text=f"SAE '{self.sae_label}' not found.<br>Available: {available}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        sae = model.saes[self.sae_label].sae
        sae_device = next(sae.parameters()).device
        one_hot_embeddings = model.get_one_hot_embeddings().to(sae_device)
        latent_activations = sae.encode(one_hot_embeddings)  # (n_features, n_latents)

        z = latent_activations.detach().cpu().numpy().T  # (n_latents, n_features)

        fig.add_trace(
            go.Heatmap(
                z=z,
                colorscale=self.colorscale,
                colorbar=dict(title="activation"),
                hovertemplate=(
                    "Input feature: %{x}<br>"
                    "SAE latent: %{y}<br>"
                    "Activation: %{z:.4f}<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(title_text="Input Feature Index")
        y_title = (
            f"{self.sae_label} (SAE Latent Index)"
            if self.show_title
            else "SAE Latent Index"
        )
        fig.update_yaxes(title_text=y_title)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


def plot_one_hot_to_latent_heatmap(
    sae_label: str,
    show_title: bool = True,
    colorscale: str = "Viridis",
) -> SAEOneHotToLatentHeatmapPlot:
    """Create an SAE one-hot input feature to latent activation heatmap plot."""
    return SAEOneHotToLatentHeatmapPlot(
        sae_label=sae_label,
        show_title=show_title,
        colorscale=colorscale,
    )
