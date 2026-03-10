import plotly.graph_objects as go
from plotly.graph_objs import Figure

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import BasePlot, GridPlot


class ReconstructionLossPlot(GridPlot):
    """Plot reconstruction loss across a grid axis as a line chart.

    When n_render_axes=1, this plot receives a 1D ModelGrid and renders
    loss values as a line/scatter plot with the axis values on the x-axis.

    Example::

        # Plot loss vs Epoch for a training history grid
        ReconstructionLossPlot()(history_grid, render_axes=("Epoch",))

        # With faceting: one subplot per Sparsity, loss vs Epoch in each
        ReconstructionLossPlot()(grid, render_axes=("Epoch",), facet_axes=("Sparsity",))
    """

    n_render_axes = 1  # expects a 1D ModelGrid

    # [05.03.26 | OliverSieweke] TODO:think about the necessity of render_grid here...
    def render_grid(
        self,
        fig: Figure,
        grid: ModelGrid,
    ) -> None:
        # [05.03.26 | OliverSieweke] TODO: Implement this check in parent class
        if len(grid.axes) != 1:
            raise ValueError(
                f"ReconstructionLossPlot with n_render_axes=1 expects a 1D ModelGrid, "
                f"got {len(grid.axes)}D (shape: {grid.shape})."
            )

        axis = grid.axes[0]
        x_values = [v.item() if hasattr(v, "item") else v for v in axis.values]

        # Collect mean losses
        mean_losses = [
            m.mean_feature_reconstruction_loss.detach().cpu().item()
            for m in grid.models.ravel()
        ]

        # Collect per-feature losses (shape: n_models x n_features)
        per_feature_losses = [
            m.per_feature_reconstruction_loss.detach().cpu()
            for m in grid.models.ravel()
        ]
        n_features = per_feature_losses[0].shape[0]

        # Add per-feature loss lines (each with a different color)
        for feat_idx in range(n_features):
            feat_losses = [losses[feat_idx].item() for losses in per_feature_losses]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=feat_losses,
                    mode="lines",
                    line=dict(width=0.7),
                    opacity=0.5,
                    hovertemplate=f"{axis.label}: %{{x}}<br>Feature {feat_idx} Loss: %{{y:.6f}}<extra></extra>",
                    showlegend=False,
                )
            )

        # Add mean loss line (thinner, no markers)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=mean_losses,
                mode="lines",
                line=dict(color="black", width=1),
                hovertemplate=f"{axis.label}: %{{x}}<br>Mean Loss: %{{y:.6f}}<extra></extra>",
                name="Mean",
                showlegend=True,
            )
        )

        fig.update_xaxes(title_text=axis.label)
        fig.update_yaxes(title_text="Reconstruction Loss", range=[0, 8])
