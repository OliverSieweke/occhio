"""W^T W heatmap visualization for feature interference analysis."""

import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class RepresentationPlot(SinglePlot):
    """Heatmap of W^T W showing feature interference and orthogonality.

    Use case:
        Visualize the Gram matrix of encoder weights to understand how features
        interact. Diagonal entries show feature representation strength (norms²),
        off-diagonal entries show interference between feature pairs.

    Data:
        - `model.W_T_W`: Gram matrix of shape `(n_features, n_features)`.
          Diagonal ≈ squared norms, off-diagonal = dot products between features.

    Visualization:
        Heatmap with diverging colorscale centered at 0. Blue = negative correlation,
        white = orthogonal, red = positive correlation. Square aspect ratio preserved.

    Customization:
        - `colorscale`: Plotly colorscale (default: blue-white-red diverging).
        - `zmin`: Minimum value for color mapping (default: -1.2).
        - `zmax`: Maximum value for color mapping (default: 1.2).
        - `show_scale`: Show colorbar (default: False).
        - `gap`: Gap between cells in pixels (default: 0).
    """

    n_render_axes = 0

    def __init__(
        self,
        colorscale: list[str] | str = None,
        zmin: float = -1.2,
        zmax: float = 1.2,
        show_scale: bool = False,
        gap: int = 0,
    ):
        if colorscale is None:
            colorscale = ["#6699FF", "#F0F0F0", "#FF6666"]
        self.colorscale = colorscale
        self.zmin = zmin
        self.zmax = zmax
        self.show_scale = show_scale
        self.gap = gap

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        W_T_W = model.W_T_W.detach().cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=W_T_W,
                colorscale=self.colorscale,
                zmid=0,
                zmin=self.zmin,
                zmax=self.zmax,
                hovertemplate="i: %{y}<br>j: %{x}<br>W<sup>T</sup>W: %{z:.3f}<extra></extra>",
                showscale=self.show_scale,
                xgap=self.gap,
                ygap=self.gap,
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

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


plot_representation = RepresentationPlot()
