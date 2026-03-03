import itertools
from abc import ABC, abstractmethod

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
        **kwargs,
    ) -> None:
        """Add traces for a single model to the figure.

        Subclasses must implement this method. Write normal Plotly code
        against fig—subplot routing is handled automatically.

        Args:
            fig: A FigureProxy wrapping the Plotly figure.
            model: The ToyModel to visualize.
            **kwargs: Additional arguments passed from the caller.
        """
        ...


class BasePlot(PlotRenderer, ABC):
    """Standalone entry point that renders a PlotRenderer across a ModelGrid."""

    def __call__(self, models: ToyModel | ModelGrid, **kwargs) -> go.Figure:
        """Main entry point—creates figure and delegates to render().

        Args:
            models: A single ToyModel or a ModelGrid.
            **kwargs: Additional arguments passed to render().
        """
        if isinstance(models, ToyModel):
            fig = make_subplots(rows=1, cols=1)
            self.render(FigureProxy(fig, row=1, col=1), models, **kwargs)
            return fig

        if isinstance(models, ModelGrid):
            n_axes = len(models.shape)

            if n_axes not in (1, 2):
                raise ValueError(
                    f"BasePlot supports 1 or 2-dimensional ModelGrids, "
                    f"got {n_axes}-dimensional (shape: {models.shape})."
                )

            n_cols = models.shape[0]
            n_rows = models.shape[1] if n_axes > 1 else 1

            # SUBPLOTS -----------------------------------------------------------------
            fig = make_subplots(rows=n_rows, cols=n_cols)

            for row, col in itertools.product(range(n_rows), range(n_cols)):
                model = models[col] if n_axes == 1 else models[col, row]

                if not isinstance(model, ToyModel):
                    raise TypeError(
                        f"Expected ToyModel from grid indexing at position "
                        f"({col}, {row}), got {type(model).__name__}"
                    )

                self.render(FigureProxy(fig, row=row + 1, col=col + 1), model, **kwargs)

            # HEADERS ------------------------------------------------------------------
            add_grid_headers(fig, models, n_rows, n_cols, inner_rows=1, inner_cols=1)
            return fig

        raise TypeError(f"Expected ToyModel or ModelGrid, got {type(models).__name__}.")
