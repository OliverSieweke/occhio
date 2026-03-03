import itertools
from abc import ABC, abstractmethod
from typing import cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.figure_proxy import FigureProxy
from occhio.visualization_2.core.plotting_utils import add_grid_headers


class PlotRenderer(ABC):
    """Protocol for objects that can render a single model into a subplot cell."""

    @abstractmethod
    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
    ) -> None:
        """Add traces for a single model to the figure.

        Subclasses must implement this method. Write normal Plotly code
        against fig—subplot routing is handled automatically.

        Args:
            fig: A FigureProxy wrapping the Plotly figure.
            model: The ToyModel to visualize.
        """
        ...


class BasePlot(PlotRenderer, ABC):
    """Standalone entry point that renders a PlotRenderer across a ModelGrid."""

    def __call__(
        self,
        models: ToyModel | ModelGrid,
    ) -> go.Figure:
        """Main entry point—creates figure and delegates to render().

        Args:
            models: A single ToyModel or a ModelGrid.
        """
        if isinstance(models, ToyModel):
            fig = make_subplots(rows=1, cols=1)
            self.render(FigureProxy(fig, row=1, col=1), models)

        elif isinstance(models, ModelGrid):
            n_axes = len(models.shape)
            n_rows = models.shape[1] if len(models.shape) > 1 else 1
            n_cols = models.shape[0]

            fig = make_subplots(rows=n_rows, cols=n_cols)

            if n_axes in (1, 2):
                self._render_static_grid(fig, models, n_rows=n_rows, n_cols=n_cols)
            elif n_axes == 3:
                self._render_animated_grid(fig, models, n_rows=n_rows, n_cols=n_cols)
            else:
                raise ValueError(
                    f"BasePlot supports 1, 2, or 3-dimensional ModelGrids, "
                    f"got {n_axes}-dimensional (shape: {models.shape})."
                )

            add_grid_headers(fig, models)

        return fig

    def _render_static_grid(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        n_rows: int,
        n_cols: int,
    ) -> None:
        """Create a subplot figure, render a grid into it, and add headers."""

        for row_idx, col_idx in itertools.product(range(n_rows), range(n_cols)):
            model = grid[col_idx] if len(grid.shape) == 1 else grid[col_idx, row_idx]

            self.render(
                FigureProxy(fig, row=row_idx + 1, col=col_idx + 1),
                cast(ToyModel, model),
            )

    def _render_animated_grid(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        n_rows: int,
        n_cols: int,
    ) -> None:
        """Render a 3D ModelGrid with a slider for the third axis.

        Axes 0 and 1 define the subplot grid (cols × rows).
        Axis 2 (last) becomes the slider, with each value as a frame.
        """
        slider_size = grid.shape[2]
        slider_axis = grid.axes[2]

        # Initial frame ----------------------------------------------------------------
        first_grid_slice = cast(ModelGrid, grid[:, :, 0])

        self._render_static_grid(fig, first_grid_slice, n_rows=n_rows, n_cols=n_cols)

        # Frames for Animation ---------------------------------------------------------
        frames: list[go.Frame] = []
        for slider_idx in range(slider_size):
            grid_slice = cast(ModelGrid, grid[:, :, slider_idx])
            # Plotly frames only hold trace data, not layout/axis config.
            # Since render() combines both (adding traces + configuring axes),
            # we use a temporary figure to collect the traces for each frame.
            temp_fig = make_subplots(rows=n_rows, cols=n_cols)
            self._render_static_grid(temp_fig, grid_slice, n_rows=n_rows, n_cols=n_cols)

            frames.append(
                go.Frame(
                    data=list(temp_fig.data),
                    name=str(slider_idx),
                )
            )

        fig.frames = frames

        # Slider -----------------------------------------------------------------------
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f"{slider_axis.values[i]:.4g}",
                            "method": "animate",
                        }
                        for i in range(slider_size)
                    ],
                    "currentvalue": {
                        "prefix": f"{slider_axis.label}: ",
                        "visible": True,
                        "xanchor": "left",
                        "font": {"size": 11},
                    },
                    "pad": {"b": 10, "t": 20},
                    "len": 0.9,
                    "x": 0.05,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                    "font": {"size": 10},
                    "ticklen": 3,
                }
            ]
        )
