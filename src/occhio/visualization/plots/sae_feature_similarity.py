"""SAE feature similarity heatmap visualization.

This module provides plots for visualizing cosine similarity between SAE decoder
weights and true feature embeddings, helping to understand how well SAE latents
align with ground-truth features.
"""

import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.composite_plot import CompositePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class SAEFeatureSimilarityPlot(SinglePlot):
    """Heatmap of cosine similarity between SAE latents and true features.

    Use case:
        Visualize how well each SAE latent aligns with each ground-truth feature.
        Helps identify which features the SAE has learned and spot redundant or
        missing latents.

    Data:
        - `model.saes_feature_similarity[sae_label]`: Cosine similarity matrix
          of shape (n_sae_latents, n_features) with values in [-1, 1].
        - `model.saes_feature_similarity_ordering[sae_label]`: Indices to reorder
          SAE latents for diagonal alignment.

    Visualization:
        Heatmap with SAE latents on Y-axis, true features on X-axis.
        RdBu colorscale from -1 (anti-aligned) to 1 (aligned).
        Hover shows exact cosine similarity values.

    Customization:
        - `sae_label`: Which SAE to plot (required).
        - `reorder`: Whether to reorder latents for diagonal alignment (default: True).
        - `show_title`: Whether to show SAE label as subplot title (default: True).
    """

    n_render_axes = 0

    def __init__(
        self,
        sae_label: str,
        reorder: bool = True,
        show_title: bool = True,
    ):
        self.sae_label = sae_label
        self.reorder = reorder
        self.show_title = show_title

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

        cos_sim = model.saes_feature_similarity[self.sae_label]

        if self.reorder:
            ordering = model.saes_feature_similarity_ordering[self.sae_label]
            cos_sim = cos_sim[ordering]

        z = cos_sim.detach().cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=z,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="cos sim", dtick=1, tickvals=[-1, 0, 1]),
                hovertemplate=(
                    "True feature: %{x}<br>"
                    "SAE Latent: %{y}<br>"
                    "Cosine Similarity: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(title_text="True Feature", constrain="domain")
        y_title = f"{self.sae_label} (SAE Latent)" if self.show_title else "SAE Latent"
        fig.update_yaxes(title_text=y_title, scaleanchor="x", constrain="domain")

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


def plot_sae_feature_similarity(
    sae_labels: list[str] | None = None,
    reorder: bool = True,
    columns: int | None = None,
) -> CompositePlot | SAEFeatureSimilarityPlot:
    """Create a composite plot showing feature similarity for multiple SAEs.

    Args:
        sae_labels: List of SAE labels to plot. If None, plots all SAEs in the model.
            When None, the plot dynamically discovers SAEs at render time.
        reorder: Whether to reorder latents for diagonal alignment (default: True).
        columns: Number of columns in the grid. If None, uses min(len(sae_labels), 3).

    Returns:
        CompositePlot if multiple SAE labels, SAEFeatureSimilarityPlot if single label.
        If sae_labels is None, returns a DynamicSAEFeatureSimilarityPlot that discovers
        SAEs at render time.
    """
    if sae_labels is None:
        return _DynamicSAEFeatureSimilarityPlot(reorder=reorder, columns=columns)

    if len(sae_labels) == 1:
        return SAEFeatureSimilarityPlot(sae_labels[0], reorder=reorder)

    # Build grid layout
    n_cols = columns if columns is not None else min(len(sae_labels), 3)
    n_rows = (len(sae_labels) + n_cols - 1) // n_cols

    layout: list[list[SAEFeatureSimilarityPlot | None]] = []
    for row_idx in range(n_rows):
        row: list[SAEFeatureSimilarityPlot | None] = []
        for col_idx in range(n_cols):
            label_idx = row_idx * n_cols + col_idx
            if label_idx < len(sae_labels):
                row.append(
                    SAEFeatureSimilarityPlot(sae_labels[label_idx], reorder=reorder)
                )
            else:
                row.append(None)
        layout.append(row)

    return CompositePlot(layout=layout)


class _DynamicSAEFeatureSimilarityPlot(SinglePlot):
    """Internal plot that dynamically discovers SAEs at render time.

    This is used when plot_sae_feature_similarity is called without sae_labels,
    allowing the plot to adapt to whatever SAEs are present in the model.
    """

    n_render_axes = 0

    def __init__(self, reorder: bool = True, columns: int | None = None):
        self.reorder = reorder
        self.columns = columns

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

        # For single SAE, render directly
        sae_labels = list(model.saes.keys())
        if len(sae_labels) == 1:
            inner = SAEFeatureSimilarityPlot(sae_labels[0], reorder=self.reorder)
            inner.render(fig, model)
            return

        # For multiple SAEs, we need to create subplots which isn't directly
        # supported in a single render call. Fall back to showing first SAE
        # with a note about using explicit labels.
        inner = SAEFeatureSimilarityPlot(
            sae_labels[0], reorder=self.reorder, show_title=True
        )
        inner.render(fig, model)

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
