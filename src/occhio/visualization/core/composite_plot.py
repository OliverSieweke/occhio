"""Composite plot for arranging multiple PlotRenderers in a grid layout."""

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import PlotOrchestrator, PlotRenderer
from occhio.visualization.core.figure_wrappers import FigureProxy
from occhio.visualization.core.plotting_utils import add_grid_headers


class PlotlySubplotSpecDict(TypedDict, total=False):
    """Specification dict for a single subplot in plotly.subplots.make_subplots.

    See: https://plotly.com/python/subplots/
    """

    type: Literal["xy", "scene", "polar", "ternary", "map", "mapbox", "domain"]
    secondary_y: bool
    colspan: int
    rowspan: int
    l: float  # padding left  # noqa: E741
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


class CompositePlot(PlotOrchestrator):
    """Compose multiple SinglePlot instances into a single figure.

    The layout is a 2D list describing the inner per-model grid. Each cell
    is a SinglePlot, a Span(...) wrapper for multi-cell plots, or None.
    Cells consumed by a span are inferred automatically.

    Example::

        composite = CompositePlot(
            layout=[
                [Span(PlotA(), colspan=2)],
                [PlotB(), PlotC()],
            ],
            column_widths=[3, 1],
            row_heights=[2, 1],
            share_axes_across_facets=True,  # Link axes for same subplot across facets
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
    _n_render_axes: int
    _share_axes_across_facets: bool
    _n_facet_cols: int
    _n_facet_rows: int

    def __init__(
        self,
        layout: Layout,
        column_widths: list[float] | None = None,
        row_heights: list[float] | None = None,
        share_axes_across_facets: bool = False,
    ):
        """Create a composite plot from a 2D layout of renderers.

        Args:
            layout: 2D list of ``PlotRenderer``, ``Span``, or ``None`` cells.
            column_widths: Relative column widths (length must match column count).
            row_heights: Relative row heights (length must match row count).
            share_axes_across_facets: If True, link x and y axes for the same
                subplot position across facets so they share the same range.

        Raises:
            ValueError: If no plots in layout or plots have inconsistent n_render_axes.
        """
        if not layout or not any(cell is not None for row in layout for cell in row):
            raise ValueError("Layout must contain at least one plot.")

        self._layout = layout
        self._column_widths = column_widths
        self._row_heights = row_heights
        self._share_axes_across_facets = share_axes_across_facets

        self._inner_rows = len(layout)
        self._inner_cols = max(len(row) for row in layout)

        self._subplots, self._specs = self._resolve_layout()
        self._n_render_axes = self._validate_n_render_axes()
        self._n_facet_cols = 1
        self._n_facet_rows = 1

    @property
    def n_render_axes(self) -> int:
        """Return the shared n_render_axes from all subplots."""
        return self._n_render_axes

    def _validate_n_render_axes(self) -> int:
        """Validate all plots have the same n_render_axes and return it."""
        values = {s.plot.n_render_axes for s in self._subplots}
        if len(values) > 1:
            raise ValueError(
                f"All plots in layout must have the same n_render_axes, "
                f"but found: {values}"
            )
        return values.pop() if values else 0

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

                plot = cell.plot if isinstance(cell, Span) else cell

                specs[inner_row_index][inner_column_index] = {
                    "colspan": cell.colspan if isinstance(cell, Span) else 1,
                    "rowspan": cell.rowspan if isinstance(cell, Span) else 1,
                    "type": plot.subplot_type,
                }

                subplots.append(
                    SubplotSpec(
                        plot=plot,
                        row=inner_row_index + 1,
                        col=inner_column_index + 1,
                    )
                )

        return subplots, specs

    def _tile_specs(
        self,
        n_facet_cols: int,
        n_facet_rows: int,
    ) -> PlotlySpecsGrid:
        """Tile the inner specs grid across all facet positions.

        Inner specs of shape ``(R, C)`` become
        ``(R * n_facet_rows, C * n_facet_cols)``.
        """
        return [
            [deepcopy(cell) for _ in range(n_facet_cols) for cell in row]
            for _ in range(n_facet_rows)
            for row in self._specs
        ]

    def configure_layout(self, fig: go.Figure) -> None:
        """Let each subplot configure the layout, then apply axis matching."""
        for subplot in self._subplots:
            subplot.plot.configure_layout(fig)

        if self._share_axes_across_facets:
            self._apply_axis_matching(fig)

    def _apply_axis_matching(self, fig: go.Figure) -> None:
        """Link axes for the same subplot position across facets.

        For each subplot in the inner layout, all facet copies share the same
        x and y axis range by setting `matches` to the reference axis of the
        first facet position.
        """
        if self._n_facet_cols <= 1 and self._n_facet_rows <= 1:
            return  # No faceting, nothing to match

        for subplot_spec in self._subplots:
            # Skip domain-type subplots (e.g., Indicator) - they don't have x/y axes
            if subplot_spec.plot.subplot_type == "domain":
                continue

            # Get reference axes from first facet position (facet_row=0, facet_col=0)
            ref_subplot = fig.get_subplot(row=subplot_spec.row, col=subplot_spec.col)
            ref_x = ref_subplot.xaxis.plotly_name.replace("axis", "")
            ref_y = ref_subplot.yaxis.plotly_name.replace("axis", "")

            # Apply matches to all other facet positions for this subplot
            for facet_row, facet_col in itertools.product(
                range(self._n_facet_rows), range(self._n_facet_cols)
            ):
                if facet_row == 0 and facet_col == 0:
                    continue  # Skip the reference facet

                phys_row = facet_row * self._inner_rows + subplot_spec.row
                phys_col = facet_col * self._inner_cols + subplot_spec.col
                target_subplot = fig.get_subplot(row=phys_row, col=phys_col)
                x_key = target_subplot.xaxis.plotly_name
                y_key = target_subplot.yaxis.plotly_name

                fig.layout[x_key].matches = ref_x
                fig.layout[y_key].matches = ref_y

    def _add_grid_headers(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        facet_axes: list[int],
    ) -> None:
        """Add headers with inner grid dimensions."""
        add_grid_headers(
            fig,
            grid,
            inner_rows=self._inner_rows,
            inner_cols=self._inner_cols,
            facet_axes=facet_axes,
        )

    def _render_static_subplots(
        self,
        grid: ModelGrid | ToyModel,
        *,
        render_axes: list[int] | None = None,
        facet_axes: list[int] | None = None,
    ) -> go.Figure:
        """Create a static figure with composite layout per facet combination.

        For a single ``ToyModel``, produces a figure with the inner layout.
        For a ``ModelGrid``, tiles the inner layout across facet positions.
        """
        legend_registry: set[str] = set()

        if isinstance(grid, ToyModel):
            fig = make_subplots(
                rows=self._inner_rows,
                cols=self._inner_cols,
                specs=self._specs,
                column_widths=self._column_widths,
                row_heights=self._row_heights,
            )

            for subplot in self._subplots:
                subplot.plot.render(
                    FigureProxy(
                        fig,
                        row=subplot.row,
                        col=subplot.col,
                        legend_registry=legend_registry,
                        is_composite=True,
                    ),
                    grid,
                )

        elif isinstance(grid, ModelGrid):
            if facet_axes is None:
                raise ValueError(
                    "facet_axes must be provided when rendering a ModelGrid"
                )
            if render_axes is None:
                raise ValueError(
                    "render_axes must be provided when rendering a ModelGrid"
                )
            n_facet_cols = grid.shape[facet_axes[0]] if len(facet_axes) >= 1 else 1
            n_facet_rows = grid.shape[facet_axes[1]] if len(facet_axes) >= 2 else 1

            # Store for axis matching in configure_layout
            self._n_facet_cols = n_facet_cols
            self._n_facet_rows = n_facet_rows

            phys_cols = n_facet_cols * self._inner_cols
            phys_rows = n_facet_rows * self._inner_rows

            fig = make_subplots(
                rows=phys_rows,
                cols=phys_cols,
                specs=self._tile_specs(n_facet_cols, n_facet_rows),
                column_widths=self._column_widths * n_facet_cols
                if self._column_widths
                else None,
                row_heights=self._row_heights * n_facet_rows
                if self._row_heights
                else None,
            )

            for facet_row, facet_col in itertools.product(
                range(n_facet_rows), range(n_facet_cols)
            ):
                # Build grid index: slice render_axes, set facet positions
                grid_index: list[int | slice] = [0] * len(grid.shape)
                for render_idx in render_axes:
                    grid_index[render_idx] = slice(None)
                if len(facet_axes) >= 1:
                    grid_index[facet_axes[0]] = facet_col
                if len(facet_axes) >= 2:
                    grid_index[facet_axes[1]] = facet_row

                model_or_slice = grid[tuple(grid_index)]

                for subplot in self._subplots:
                    phys_row = facet_row * self._inner_rows + subplot.row
                    phys_col = facet_col * self._inner_cols + subplot.col
                    subplot.plot.render(
                        FigureProxy(
                            fig,
                            row=phys_row,
                            col=phys_col,
                            legend_registry=legend_registry,
                            is_composite=True,
                        ),
                        model_or_slice,
                    )

        return fig
