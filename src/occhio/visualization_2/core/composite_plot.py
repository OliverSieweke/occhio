import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import PlotRenderer
from occhio.visualization_2.core.figure_wrappers import FigureProxy
from occhio.visualization_2.core.plotting_utils import add_grid_headers


class PlotlySubplotSpecDict(TypedDict, total=False):
    """Specification dict for a single subplot in plotly.subplots.make_subplots.

    See: https://plotly.com/python/subplots/
    """

    type: Literal["xy", "scene", "polar", "ternary", "map", "mapbox", "domain"]
    secondary_y: bool
    colspan: int
    rowspan: int
    l: float  # padding left
    r: float  # padding right
    t: float  # padding top
    b: float  # padding bottom


PlotlySubplotSpec: TypeAlias = PlotlySubplotSpecDict | None
PlotlySpecsGrid: TypeAlias = list[list[PlotlySubplotSpec]]


# [03.03.26 | OliverSieweke] TODO:span never 0 or lower
@dataclass
class Span:
    """Wrap a PlotRenderer to span multiple rows/columns in a composite layout.

    Example::

        Span(MyPlot(), colspan=2)  # plot spans two columns
    """

    plot: PlotRenderer
    colspan: int = 1
    rowspan: int = 1


@dataclass
class SubplotSpec:
    """Specification for a subplot within the composite layout.

    Attributes:
        plot: The PlotRenderer to render in this subplot.
        row: 1-indexed row position in the inner grid.
        col: 1-indexed column position in the inner grid.
    """

    plot: PlotRenderer
    row: int
    col: int


# A layout cell is a PlotRenderer, a Span wrapping one, or None.
LayoutCell = PlotRenderer | Span | None

# The layout is a 2D list: layout[row_index][col_index].
Layout = list[list[LayoutCell]]


class CompositePlot:
    """Compose multiple PlotRenderer instances into a single figure.

    The layout is a 2D list describing the inner per-model grid. Each cell
    is a PlotRenderer, a Span(...) wrapper for multi-cell plots, or None.
    Cells consumed by a span are inferred automatically.

    Example::

        composite = CompositePlot(
            layout=[
                [Span(PlotA(), colspan=2)],
                [PlotB(), PlotC()],
            ],
            column_widths=[3, 1],
            row_heights=[2, 1],
        )
        fig = composite(model_grid)
    """

    _layout: Layout
    _column_widths: list[float] | None
    _row_heights: list[float] | None
    _subplots: list[SubplotSpec]
    _inner_rows: int
    _inner_cols: int
    _specs: PlotlySpecsGrid

    def __init__(
        self,
        layout: Layout,
        column_widths: list[float] | None = None,
        row_heights: list[float] | None = None,
    ):
        """Create a composite plot from a 2D layout of renderers.

        Args:
            layout: 2D list of ``PlotRenderer``, ``Span``, or ``None`` cells.
            column_widths: Relative column widths (length must match column count).
            row_heights: Relative row heights (length must match row count).
        """
        if not layout or not any(cell is not None for row in layout for cell in row):
            raise ValueError("Layout must contain at least one plot.")

        self._layout = layout
        self._column_widths = column_widths
        self._row_heights = row_heights

        self._inner_rows = len(layout)
        self._inner_cols = max(len(row) for row in layout)

        self._subplots, self._specs = self._resolve_layout()

    def _resolve_layout(
        self,
    ) -> tuple[list[SubplotSpec], PlotlySpecsGrid]:
        """Parse the raw layout into plot entries and Plotly specs.

        Returns:
            entries: List of SubplotSpec instances.
            specs: 2D list suitable for make_subplots(specs=...).
        """
        subplots: list[SubplotSpec] = []
        specs: PlotlySpecsGrid = [
            [None for _ in range(self._inner_cols)] for _ in range(self._inner_rows)
        ]

        for inner_row_index, row in enumerate(self._layout):
            for inner_column_index, cell in enumerate(row):
                # Skip cells already consumed by a previous span.
                if cell is None:
                    continue

                specs[inner_row_index][inner_column_index] = {
                    "colspan": cell.colspan if isinstance(cell, Span) else 1,
                    "rowspan": cell.rowspan if isinstance(cell, Span) else 1,
                }

                subplots.append(
                    SubplotSpec(
                        plot=cell.plot if isinstance(cell, Span) else cell,
                        row=inner_row_index + 1,
                        col=inner_column_index + 1,
                    )
                )

        return subplots, specs

    def _tile_specs(
        self,
        n_models_cols: int,
        n_models_rows: int,
    ) -> PlotlySpecsGrid:
        """Tile the inner specs grid across all model positions.

        Inner specs of shape ``(R, C)`` become
        ``(R * n_models_rows, C * n_models_cols)``.
        """

        return [
            [deepcopy(cell) for _ in range(n_models_cols) for cell in row]
            for _ in range(n_models_rows)
            for row in self._specs
        ]

    def __call__(self, models: ToyModel | ModelGrid, **kwargs) -> go.Figure:
        """Render the composite layout for a single model or a 1D/2D grid.

        Args:
            models: A single ``ToyModel`` or a 1D/2D ``ModelGrid``.
            **kwargs: Forwarded to each subplot's ``render()``.

        Returns:
            A Plotly ``Figure`` with all subplots populated.
        """
        if isinstance(models, ToyModel):
            fig = make_subplots(
                rows=self._inner_rows,
                cols=self._inner_cols,
                specs=self._specs,
                column_widths=self._column_widths,
                row_heights=self._row_heights,
            )
            legend_registry: set[str] = set()
            for subplot in self._subplots:
                subplot.plot.render(
                    FigureProxy(
                        fig,
                        row=subplot.row,
                        col=subplot.col,
                        legend_registry=legend_registry,
                    ),
                    models,
                    **kwargs,
                )
            return fig

        if isinstance(models, ModelGrid):
            n_axes = len(models.shape)

            if n_axes not in (1, 2):
                raise ValueError(
                    f"CompositePlot supports 1 or 2-dimensional ModelGrids, "
                    f"got {n_axes}-dimensional (shape: {models.shape})."
                )

            n_model_cols = models.shape[0]
            n_model_rows = models.shape[1] if n_axes > 1 else 1

            phys_cols = n_model_cols * self._inner_cols
            phys_rows = n_model_rows * self._inner_rows

            fig = make_subplots(
                rows=phys_rows,
                cols=phys_cols,
                specs=self._tile_specs(n_model_cols, n_model_rows),
                column_widths=self._column_widths * n_model_cols
                if self._column_widths
                else None,
                row_heights=self._row_heights * n_model_rows
                if self._row_heights
                else None,
            )

            legend_registry = set()
            for model_row, model_col in itertools.product(
                range(n_model_rows), range(n_model_cols)
            ):
                model = (
                    models[model_col] if n_axes == 1 else models[model_col, model_row]
                )

                if not isinstance(model, ToyModel):
                    raise TypeError(
                        f"Expected ToyModel from grid indexing at position "
                        f"({model_col}, {model_row}), got {type(model).__name__}"
                    )

                for subplot in self._subplots:
                    phys_row = model_row * self._inner_rows + subplot.row
                    phys_col = model_col * self._inner_cols + subplot.col
                    subplot.plot.render(
                        FigureProxy(
                            fig,
                            row=phys_row,
                            col=phys_col,
                            legend_registry=legend_registry,
                        ),
                        model,
                        **kwargs,
                    )

            add_grid_headers(
                fig,
                models,
                self._inner_rows,
                self._inner_cols,
            )

            return fig

        raise TypeError(f"Expected ToyModel or ModelGrid, got {type(models).__name__}.")
