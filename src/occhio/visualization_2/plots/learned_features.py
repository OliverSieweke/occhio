import plotly.graph_objects as go
from plotly.graph_objs import Figure

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import BasePlot, GridPlot


class LearnedFeaturesPlot(GridPlot):
    """Plot the number of learned features across a grid axis with interference distribution overlay.

    A feature is considered "learned" if its reconstruction loss is below
    a specified threshold (default: 0.1).

    When n_render_axes=1, this plot receives a 1D ModelGrid and renders:
    - Count of learned features as a line plot (left y-axis)
    - Distribution of interferences for learned features as violin plots (right y-axis)

    Example::

        # Plot learned features vs Epoch for a training history grid
        LearnedFeaturesPlot()(history_grid, render_axes=("Epoch",))

        # With custom threshold
        LearnedFeaturesPlot(threshold=0.05)(grid, render_axes=("n_hidden",))

        # With faceting: one subplot per Sparsity
        LearnedFeaturesPlot()(grid, render_axes=("Epoch",), facet_axes=("Sparsity",))
    """

    n_render_axes = 1  # expects a 1D ModelGrid

    def __init__(
        self, threshold: float = 0.1, show_interference: bool = True, **kwargs
    ):
        """Initialize the LearnedFeaturesPlot.

        Args:
            threshold: Reconstruction loss threshold below which a feature is
                considered "learned". Default: 0.1
            show_interference: Whether to show interference distribution violin plots.
                Default: True
            **kwargs: Additional arguments passed to GridPlot
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.show_interference = show_interference

    def render_grid(
        self,
        fig: Figure,
        grid: ModelGrid,
    ) -> None:
        if len(grid.axes) != 1:
            raise ValueError(
                f"LearnedFeaturesPlot with n_render_axes=1 expects a 1D ModelGrid, "
                f"got {len(grid.axes)}D (shape: {grid.shape})."
            )

        axis = grid.axes[0]
        axis_values = [v.item() if hasattr(v, "item") else v for v in axis.values]

        # Use categorical x-axis positions for equal spacing
        x_positions = list(range(len(axis_values)))
        x_labels = [str(v) for v in axis_values]

        # Count learned features (loss < threshold) for each model
        learned_counts = [
            (m.per_feature_reconstruction_loss < self.threshold).sum().item()
            for m in grid.models.ravel()
        ]

        # Add count line plot on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=learned_counts,
                mode="lines+markers",
                line=dict(color="steelblue", width=2),
                marker=dict(size=6),
                hovertemplate=f"{axis.label}: %{{x}}<br>Learned Features: %{{y}}<extra></extra>",
                name=f"Count (threshold={self.threshold})",
                showlegend=True,
                yaxis="y1",
            )
        )

        # Add violin plots for interference distribution if requested
        if self.show_interference:
            for idx, (x_pos, axis_val, model) in enumerate(
                zip(x_positions, axis_values, grid.models.ravel())
            ):
                # Get learned feature mask
                learned_mask = model.per_feature_reconstruction_loss < self.threshold

                # Get interferences for learned features only
                learned_interferences = (
                    model.total_feature_interferences[learned_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Skip if no learned features
                if len(learned_interferences) == 0:
                    continue

                fig.add_trace(
                    go.Violin(
                        y=learned_interferences,
                        x=[x_pos] * len(learned_interferences),
                        name=f"Interference {axis.label}={axis_val}",
                        box_visible=False,
                        meanline_visible=False,
                        showlegend=False,
                        hoverinfo="y",
                        points="all",
                        pointpos=0,
                        jitter=0.3,
                        scalemode="width",
                        width=0.4,
                        marker=dict(size=3, opacity=0.5, color="darkred"),
                        line=dict(width=2, color="darkred"),
                        fillcolor="rgba(220, 20, 60, 0.4)",
                        yaxis="y2",
                    )
                )

        # Update axes
        fig.update_xaxes(
            title_text=axis.label,
            tickmode="array",
            tickvals=x_positions,
            ticktext=x_labels,
        )
        fig.update_yaxes(
            title_text="Number of Learned Features",
        )
        #
        # if self.show_interference:
        #     # Configure secondary y-axis for interference violins
        #     fig["layout"]["yaxis2"] = dict(
        #         title="Interference (Learned Features)",
        #         overlaying="y",
        #         side="right",
        #         showgrid=False,
        #     )
