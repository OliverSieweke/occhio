import plotly.express as px
import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class EmbeddingPlot(SinglePlot):
    """Plot encoder weight vectors as arrows from origin."""

    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
        **kwargs,
    ) -> None:
        colorscale = px.colors.sequential.Plasma_r

        # Check embedding dimension
        embedding_dim = model.W.shape[0]
        if embedding_dim not in [1, 2]:
            raise ValueError(
                f"EmbeddingPlot only supports 1 or 2-dimensional embedding spaces, "
                f"got {embedding_dim}-dimensional (W.shape = {model.W.shape})."
            )

        for feature_idx in range(model.W.shape[1]):
            color_idx = int(
                model.importances[feature_idx].item() * 0.9 * (len(colorscale) - 1)
            )
            color_idx = max(0, min(color_idx, len(colorscale) - 1))
            color = colorscale[color_idx]

            x_val = model.W[0, feature_idx].cpu().item()
            y_val = model.W[1, feature_idx].cpu().item() if embedding_dim == 2 else 0.0
            importance = model.importances[feature_idx].item()

            # Arrow from origin to embedding
            fig.add_annotation(
                x=x_val,
                y=y_val,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                opacity=0.7,
            )

            # Invisible scatter point for hover
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0),
                    hovertemplate=(
                        f"<b>Feature {feature_idx}</b><br>"
                        f"Importance: {importance:.4f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        # Configure axes
        z = 1.2
        fig.update_xaxes(range=[-z, z], showticklabels=False, showgrid=False)
        fig.update_yaxes(
            range=[-z, z],
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            scaleratio=1,
        )


plot_embedding = EmbeddingPlot()
