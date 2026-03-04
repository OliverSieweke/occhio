import itertools
from abc import ABC, abstractmethod
from typing import cast

import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.figure_proxy import FigureProxy
from occhio.visualization_2.core.plotting_utils import add_grid_headers


class InteractiveFigure(go.Figure):
    """A Plotly Figure that carries a post-render JavaScript snippet.

    Used for multi-slider animations where a JS callback coordinates
    slider state with frame selection. Behaves identically to a normal
    Figure when no script is attached.
    """

    _post_script: str

    def __init__(self, *args, post_script: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Plotly's Figure.__setattr__ blocks custom attributes,
        # so we bypass it with object.__setattr__.
        object.__setattr__(
            self,
            "_post_script",
            """
            const plot = document.getElementById('{plot_id}');
            
            plot.on('plotly_sliderchange', () => {
                const frameName = plot.layout.sliders.map(({active}) => active).join('_');
                Plotly.animate(plot, [frameName], {
                    frame: { duration: 0, redraw: true },
                    mode: 'immediate',
                    transition: { duration: 0 }
                });
            });
        """,
        )

    def _ipython_display_(self, **kwargs):
        html = self.to_html(
            post_script=self._post_script,
            full_html=False,
            include_plotlyjs="require",
            auto_play=False,
        )
        display(HTML(html))

    def show(self, *args, **kwargs):
        html = self.to_html(
            post_script=self._post_script,
            full_html=False,
            include_plotlyjs="require",
            auto_play=False,
        )
        display(HTML(html))


class PlotRenderer(ABC):
    """Protocol for objects that can render a single model into a subplot cell."""

    @abstractmethod
    def render(
        self,
        fig: FigureProxy,
        model: ToyModel,
    ) -> go.Figure:
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
        height: int | None = None,
        width: int | None = None,
    ) -> go.Figure:
        """Main entry point—creates figure and delegates to render().

        Args:
            models: A single ToyModel or a ModelGrid.
            height: Optional figure height in pixels.
            width: Optional figure width in pixels.
        """
        if not isinstance(models, (ToyModel, ModelGrid)):
            models = ModelGrid.from_iterable(models)

        if isinstance(models, ToyModel):
            fig = make_subplots(rows=1, cols=1)
            self.render(FigureProxy(fig, row=1, col=1), models)

        elif isinstance(models, ModelGrid):
            n_axes = len(models.shape)
            n_rows = models.shape[1] if len(models.shape) > 1 else 1
            n_cols = models.shape[0]

            fig = make_subplots(rows=n_rows, cols=n_cols)

            if n_axes in (1, 2):
                fig = self._render_static_grid(
                    fig, models, n_rows=n_rows, n_cols=n_cols
                )
            elif n_axes >= 3:
                fig = self._render_animated_grid(
                    fig, models, n_rows=n_rows, n_cols=n_cols, height=height
                )
            else:
                raise ValueError(
                    f"BasePlot supports 1, 2, or 3-dimensional ModelGrids, "
                    f"got {n_axes}-dimensional (shape: {models.shape})."
                )

            add_grid_headers(fig, models)

        fig.update_layout(
            height=height,
            width=width,
        )

        return fig

    def _render_static_grid(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        n_rows: int,
        n_cols: int,
    ) -> go.Figure:
        """Render a 2D grid of models into the given figure."""

        for row_idx, col_idx in itertools.product(range(n_rows), range(n_cols)):
            model = grid[col_idx] if len(grid.shape) == 1 else grid[col_idx, row_idx]

            self.render(
                FigureProxy(fig, row=row_idx + 1, col=col_idx + 1),
                cast(ToyModel, model),
            )

        return fig

    def _render_animated_grid(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        n_rows: int,
        n_cols: int,
        height: int | None = None,
    ) -> InteractiveFigure:
        """Render a 3+ dimensional ModelGrid with sliders for axes beyond the first two.

        Axes 0 and 1 define the subplot grid (cols × rows).
        Axes 2, 3, ... each become a slider, with frames for every
        combination of slider positions.
        """
        slider_axes = grid.axes[2:]
        sliders_shape = grid.shape[2:]

        # Initial frame: first position on every slider axis
        first_frame_index = (slice(None), slice(None)) + tuple([0] * len(slider_axes))
        first_grid_slice = cast(ModelGrid, grid[first_frame_index])

        self._render_static_grid(fig, first_grid_slice, n_rows=n_rows, n_cols=n_cols)
        # add_grid_headers(fig, grid)

        # Frames for every combination of slider positions
        frames: list[go.Frame] = []
        for slider_indices in itertools.product(
            *(range(slider_index) for slider_index in sliders_shape)
        ):
            frame_index = (slice(None), slice(None)) + slider_indices
            grid_slice = cast(ModelGrid, grid[frame_index])

            # Plotly frames only hold trace data, not layout/axis config.
            # Since render() combines both (adding traces + configuring axes),
            # we use a temporary figure to collect the traces for each frame.
            temp_fig = make_subplots(rows=n_rows, cols=n_cols)
            temp_fig = self._render_static_grid(
                temp_fig, grid_slice, n_rows=n_rows, n_cols=n_cols
            )

            frames.append(
                go.Frame(
                    name="_".join(str(i) for i in slider_indices),
                    data=list(temp_fig.data),
                )
            )

        fig.frames = frames

        # # Sliders each use method="skip", so they can track the state without
        # # firing commands. The JS callback coordinates all sliders.
        # # Position sliders below plot with constant pixel spacing
        # # Note: Plotly sliders use domain coordinates (0-1), so we convert pixel spacing
        # # to maintain constant pixel distance regardless of figure height
        # fig_height = height if height is not None else 450  # Plotly default
        # slider_total_height_px = 40  # Total pixels per slider (widget + gap)
        # Convert to domain coordinates based on figure height
        slider_spacing_domain = 0.3
        # slider_spacing_domain = 0.3

        sliders = []
        for i, axis in enumerate(slider_axes):
            slider = {
                "active": 0,
                "steps": [
                    {
                        "args": [None],
                        "label": f"{axis.values[i]:.4g}",
                        "method": "skip",
                    }
                    for i in range(len(axis.values))
                ],
                "currentvalue": {
                    "prefix": f"{axis.label}: ",
                    "visible": True,
                    "xanchor": "left",
                    "font": {"size": 11},
                },
                "pad": {"b": 0, "t": 0},
                "len": 0.9,
                "x": 0.05,
                "xanchor": "left",
                "y": -0.05 - slider_spacing_domain * i,
                "yanchor": "top",
                "font": {"size": 10},
                "ticklen": 4,
            }
            sliders.append(slider)

        fig.update_layout(sliders=sliders)

        return InteractiveFigure(fig)
