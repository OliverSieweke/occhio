import plotly.graph_objects as go
from plotly.graph_objs import Figure

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import BasePlot, GridPlot


class LossDistributionPlot(GridPlot):
    """Plot the distribution of per-feature reconstruction losses as violin plots.

    When n_render_axes=1, this plot receives a 1D ModelGrid and renders
    one violin plot per model, showing the distribution of reconstruction losses
    across all features in that model.

    Example::

        # Plot loss distribution across different sparsity values
        LossDistributionPlot()(grid, render_axes=("Sparsity",))

        # With faceting: one subplot per n_hidden, violins vs Sparsity in each
        LossDistributionPlot()(
            grid, render_axes=("Sparsity",), facet_axes=("n_hidden",)
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
                f"LossDistributionPlot with n_render_axes=1 expects a 1D ModelGrid, "
                f"got {len(grid.axes)}D (shape: {grid.shape})."
            )

        axis = grid.axes[0]
        axis_values = [v.item() if hasattr(v, "item") else v for v in axis.values]

        # Use categorical x-axis positions for equal spacing
        x_positions = list(range(len(axis_values)))
        x_labels = [str(v) for v in axis_values]

        # Add violin plot for each model
        for idx, (x_pos, axis_val, model) in enumerate(
            zip(x_positions, axis_values, grid.models.ravel())
        ):
            per_feature_losses = (
                model.per_feature_reconstruction_loss.detach().cpu().numpy()
            )

            fig.add_trace(
                go.Violin(
                    y=per_feature_losses,
                    x=[x_pos] * len(per_feature_losses),
                    name=f"{axis.label}={axis_val}",
                    box_visible=False,
                    meanline_visible=False,
                    showlegend=False,
                    hoverinfo="y",
                    points="all",
                    pointpos=0,
                    jitter=0.3,
                    scalemode="count",
                    bandwidth=0.05,
                    marker=dict(size=2, opacity=0.4, color="black"),
                    line=dict(width=1.5),
                    fillcolor="rgba(100, 100, 100, 0.5)",
                )
            )

        fig.update_xaxes(
            title_text=axis.label,
            tickmode="array",
            tickvals=x_positions,
            ticktext=x_labels,
        )
        fig.update_yaxes(title_text="Per-Feature Reconstruction Loss")
        # fig.update_layout(violinmode="group", plot_bgcolor="white")
