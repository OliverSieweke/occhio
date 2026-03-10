import plotly.graph_objects as go
from plotly.graph_objs import Figure

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import BasePlot, GridPlot


class FeatureDimensionalityPlot(GridPlot):
    """Plot feature dimensionalities and embedded features per hidden dimensions across a grid axis.

    When n_render_axes=1, this plot receives a 1D ModelGrid and renders:
    - Per-feature dimensionality values as individual lines
    - Mean feature dimensionality as a bold black line
    - Embedded features per hidden dimensions as a dashed line

    Example::

        # Plot dimensionality vs Epoch for a training history grid
        FeatureDimensionalityPlot()(history_grid, render_axes=("Epoch",))

        # With faceting: one subplot per Sparsity, dimensionality vs Epoch in each
        FeatureDimensionalityPlot()(
            grid, render_axes=("Epoch",), facet_axes=("Sparsity",)
        )
    """

    n_render_axes = 1  # expects a 1D ModelGrid

    def render_grid(
        self,
        fig: Figure,
        grid: ModelGrid,
    ) -> None:
        if len(grid.axes) != 1:
            raise ValueError(
                f"FeatureDimensionalityPlot with n_render_axes=1 expects a 1D ModelGrid, "
                f"got {len(grid.axes)}D (shape: {grid.shape})."
            )

        axis = grid.axes[0]
        x_values = [v.item() if hasattr(v, "item") else v for v in axis.values]

        # Collect embedded features per hidden dimensions (scalar per model)
        embedded_features = [
            m.embedded_features_per_hidden_dimensions.detach().cpu().item()
            for m in grid.models.ravel()
        ]

        # Collect mean feature dimensionalities (scalar per model)
        mean_dimensionalities = [
            m.mean_feature_dimensionalities.detach().cpu().item()
            for m in grid.models.ravel()
        ]

        # Collect per-feature dimensionalities (shape: n_models x n_features)
        per_feature_dims = [
            m.feature_dimensionalities.detach().cpu() for m in grid.models.ravel()
        ]
        n_features = per_feature_dims[0].shape[0]

        # Add per-feature dimensionality lines (each with a different color)
        for feat_idx in range(n_features):
            feat_dims = [dims[feat_idx].item() for dims in per_feature_dims]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=feat_dims,
                    mode="lines",
                    line=dict(width=0.7),
                    opacity=0.5,
                    hovertemplate=f"{axis.label}: %{{x}}<br>Feature {feat_idx} Dimensionality: %{{y:.6f}}<extra></extra>",
                    showlegend=False,
                )
            )

        # Add mean feature dimensionality line (bold black)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=mean_dimensionalities,
                mode="lines",
                line=dict(color="black", width=1),
                hovertemplate=f"{axis.label}: %{{x}}<br>Mean Dimensionality: %{{y:.6f}}<extra></extra>",
                name="Mean Dimensionality",
                showlegend=True,
            )
        )

        # Add embedded features per hidden dimensions line (dashed)
        # fig.add_trace(
        #     go.Scatter(
        #         x=x_values,
        #         y=embedded_features,
        #         mode="lines",
        #         line=dict(color="red", width=1.5, dash="dash"),
        #         hovertemplate=f"{axis.label}: %{{x}}<br>Embedded Features / Hidden Dim: %{{y:.6f}}<extra></extra>",
        #         name="Embedded Features / Hidden Dim",
        #         showlegend=True,
        #     )
        # )

        fig.update_xaxes(title_text=axis.label)
        fig.update_yaxes(title_text="Dimensionality")
