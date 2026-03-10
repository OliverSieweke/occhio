import plotly.graph_objects as go
from plotly.graph_objs import Figure

from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import SingleModelPlot


class RepresentationPlot(SingleModelPlot):
    """Plot W^T W heatmap showing feature interference/orthogonality.

    Note: The original plot_representation also showed:
    - Bias vector as a thin vertical heatmap (requires multi-column per model)
    - Feature norms bar chart (requires multi-row per model)
    These are not yet supported by the BasePlot layout.
    """

    def render(
        self,
        fig: Figure,
        model: ToyModel,
    ) -> None:
        colorscale = ["#6699FF", "#F0F0F0", "#FF6666"]
        W_T_W = model.W_T_W.detach().cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=W_T_W,
                colorscale=colorscale,
                zmid=0,
                zmin=-1.2,
                zmax=1.2,
                hovertemplate="i: %{y}<br>j: %{x}<br>W<sup>T</sup>W: %{z:.3f}<extra></extra>",
                showscale=False,
                xgap=1,
                ygap=1,
            )
        )

        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )
