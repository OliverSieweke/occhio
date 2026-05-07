"""Base classes for the visualization framework.

This module defines the core abstractions:

- ``PlotRenderer``: Objects that render a single model into a subplot cell.
- ``PlotOrchestrator``: Objects that orchestrate rendering across ModelGrids
  (handling faceting, sliders, figure construction).
- ``SinglePlot``: Combines both—a renderer that can also be called directly
  on ToyModels or ModelGrids.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Literal, Sequence, overload

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.model_grid import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core.figure_wrappers import FigureProxy, InteractiveFigure
from occhio.visualization.core.plotting_utils import add_grid_headers


class PlotRenderer(ABC):
    """Abstract base for objects that render a single model into a subplot cell.

    Subclasses implement ``render()`` to add Plotly traces to a ``FigureProxy``.
    The ``n_render_axes`` class attribute declares how many grid axes the plot
    expects to receive for rendering within a single subplot.
    """

    n_render_axes: int = 0
    """Number of grid axes this plot expects to iterate over within a single subplot.

    - 0: receives a single ToyModel (default, backward-compatible)
    - 1: receives a 1D ModelGrid (e.g., for line charts over Epoch)
    - 2: receives a 2D ModelGrid (e.g., x-axis + line series)
    """

    subplot_type: Literal[
        "xy", "scene", "polar", "ternary", "map", "mapbox", "domain", "table"
    ] = "xy"

    @abstractmethod
    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Add traces for a single model to the figure.

        Subclasses must implement this method. Write normal Plotly code
        against fig—subplot routing is handled automatically.

        Args:
            fig: A FigureProxy wrapping the Plotly figure.
            models: A ToyModel (when n_render_axes=0) or a ModelGrid with
                    dimensionality equal to n_render_axes.
        """
        ...

    def configure_layout(self, fig: go.Figure) -> None:
        """Configure figure-wide layout properties (optional).

        Called once after all render() calls complete. Use this for global
        styling like background color, bar gaps, font settings, etc.

        Per-subplot axis configuration should use update_xaxes/update_yaxes
        in render() instead.

        Args:
            fig: The raw Plotly Figure (not a FigureProxy).

        Example::

            def configure_layout(self, fig):
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    bargap=0.15,
                )
                # Style all axes globally
                fig.update_xaxes(showgrid=False)
        """
        pass  # Default no-op implementation


AxisSpec = int | str
"""An axis identifier—either an integer index or a string label."""


class PlotOrchestrator(ABC):
    """Abstract base for objects that orchestrate rendering across ModelGrids.

    Handles faceting, sliders, and figure construction. Subclasses must implement
    ``_render_static_subplots()`` and ``configure_layout()``.

    This class is separate from ``PlotRenderer`` because orchestrators (like
    ``CompositePlot``) don't render themselves—they delegate to child renderers.
    """

    @property
    @abstractmethod
    def n_render_axes(self) -> int:
        """Number of grid axes this plot expects for rendering within a single subplot."""
        ...

    @abstractmethod
    def configure_layout(self, fig: go.Figure) -> None:
        """Configure figure-wide layout properties."""
        ...

    @abstractmethod
    def _render_static_subplots(
        self,
        grid: ModelGrid | ToyModel,
        *,
        render_axes: list[int] | None = None,
        facet_axes: list[int] | None = None,
    ) -> go.Figure:
        """Create a static figure with subplots for each facet combination."""
        ...

    @staticmethod
    def _resolve_axis_index(spec: AxisSpec, grid: ModelGrid) -> int:
        """Convert an AxisSpec (int index or str label) to a validated int index."""
        if isinstance(spec, int):
            if spec < 0 or spec >= len(grid.axes):
                raise IndexError(
                    f"Axis index {spec} is out of range for a grid with "
                    f"{len(grid.axes)} axes."
                )
            return spec
        if isinstance(spec, str):
            labels = [a.label for a in grid.axes]
            try:
                return labels.index(spec)
            except ValueError:
                raise ValueError(
                    f"No axis with label {spec!r}. Available labels: {labels}"
                ) from None

        raise TypeError(
            f"Axis specifier must be int or str, got {type(spec).__name__}."
        )

    def _resolve_full_axes(
        self,
        grid: ModelGrid,
        *,
        facet_axes: Sequence[AxisSpec] | None,
        slider_axes: Sequence[AxisSpec] | None,
        render_axes: Sequence[AxisSpec] | None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Assign every grid axis to exactly one role: render, facet, or slider.

        Unassigned axes are auto-filled: render first (up to n_render_axes),
        then facet (up to 2), then slider (remaining).

        Args:
            grid: The ModelGrid to resolve axes for.
            facet_axes: User-specified facet axes (or None for auto).
            slider_axes: User-specified slider axes (or None for auto).
            render_axes: User-specified render axes (or None for auto).

        Returns:
            (render_axes, facet_axes, slider_axes) as resolved int indices.
        """
        expected_render_axes_count = self.n_render_axes

        resolved_render_axes: list[int] | None = (
            [PlotOrchestrator._resolve_axis_index(a, grid) for a in render_axes]
            if render_axes is not None
            else None
        )
        resolved_facet_axes: list[int] | None = (
            [PlotOrchestrator._resolve_axis_index(a, grid) for a in facet_axes]
            if facet_axes is not None
            else None
        )
        resolved_slider_axes: list[int] | None = (
            [PlotOrchestrator._resolve_axis_index(a, grid) for a in slider_axes]
            if slider_axes is not None
            else None
        )

        # Checks
        if (
            resolved_render_axes is not None
            and len(resolved_render_axes) != expected_render_axes_count
        ):
            raise ValueError(
                f"render_axes must have exactly {expected_render_axes_count} entries, "
                f"got {len(resolved_render_axes)}."
            )
        if resolved_facet_axes is not None and len(resolved_facet_axes) > 2:
            raise ValueError(
                f"facet_axes must have at most 2 entries, got {len(resolved_facet_axes)}."
            )

        assigned: dict[int, str] = {}
        axes_groups: list[tuple[str, list[int] | None]] = [
            ("render_axes", resolved_render_axes),
            ("facet_axes", resolved_facet_axes),
            ("slider_axes", resolved_slider_axes),
        ]
        for axis_role, axes_indices in axes_groups:
            if axes_indices is None:
                continue
            for idx in axes_indices:
                if idx in assigned:
                    if assigned[idx] == axis_role:
                        raise ValueError(
                            f"Axis {idx} ('{grid.axes[idx].label}') appears more than "
                            f"once in {axis_role}. Each axis may only appear once."
                        )
                    raise ValueError(
                        f"Axis {idx} ('{grid.axes[idx].label}') is assigned to "
                        f"both {assigned[idx]} and {axis_role}. "
                        f"Axis roles must be disjoint."
                    )
                assigned[idx] = axis_role

        #  Filling up --------------------------------------------------------------
        all_axes = set(range(len(grid.axes)))
        used_axes = set(
            (resolved_render_axes or [])
            + (resolved_facet_axes or [])
            + (resolved_slider_axes or [])
        )
        available_axes = all_axes - used_axes

        if resolved_render_axes is None:
            # render_axes is attributed axes starting from the left
            resolved_render_axes = sorted(available_axes)[:expected_render_axes_count]
            available_axes -= set(resolved_render_axes)

        if resolved_facet_axes is None:
            # facet_axes is attributed the next axes starting from the left, up-to-two
            resolved_facet_axes = sorted(available_axes)[: min(2, len(available_axes))]
            available_axes -= set(resolved_facet_axes)

        if resolved_slider_axes is None:
            # slider_axes gets attributed all remaining axes
            resolved_slider_axes = sorted(available_axes)

        return resolved_render_axes, resolved_facet_axes, resolved_slider_axes

    def __call__(
        self,
        models: ToyModel | ModelGrid,
        height: int | None = None,
        width: int | None = None,
        *,
        facet_axes: Sequence[AxisSpec] | None = None,
        slider_axes: Sequence[AxisSpec] | None = None,
        render_axes: Sequence[AxisSpec] | None = None,
    ) -> go.Figure:
        """Main entry point—creates figure and delegates to _render_static_subplots().

        Args:
            models: A single ToyModel or a ModelGrid.
            height: Optional figure height in pixels.
            width: Optional figure width in pixels.
            facet_axes: Grid axes to use for faceting (up to 2).
            slider_axes: Grid axes to use for sliders.
            render_axes: Grid axes passed to each subplot's render method.
        """
        if not isinstance(models, (ToyModel, ModelGrid)):
            models = ModelGrid.from_iterable(models)

        if isinstance(models, ToyModel):
            fig = self._render_static_subplots(models)

        elif isinstance(models, ModelGrid):
            render_axes_resolved, facet_axes_resolved, slider_axes_resolved = (
                self._resolve_full_axes(
                    models,
                    render_axes=render_axes,
                    facet_axes=facet_axes,
                    slider_axes=slider_axes,
                )
            )

            if len(slider_axes_resolved) == 0:
                fig = self._render_static_subplots(
                    models,
                    facet_axes=facet_axes_resolved,
                    render_axes=render_axes_resolved,
                )
            else:
                fig = self._render_animated_subplots(
                    models,
                    facet_axes=facet_axes_resolved,
                    slider_axes=slider_axes_resolved,
                    render_axes=render_axes_resolved,
                )

            self._add_grid_headers(fig, models, facet_axes=facet_axes_resolved)

        self.configure_layout(fig)

        fig.update_layout(
            height=height,
            width=width,
        )

        return fig

    def _add_grid_headers(
        self,
        fig: go.Figure,
        grid: ModelGrid,
        *,
        facet_axes: list[int],
    ) -> None:
        """Add grid headers to a faceted figure.

        Override in subclasses (e.g., CompositePlot) to handle custom inner layouts.
        """
        add_grid_headers(fig, grid, facet_axes=facet_axes)

    def _render_animated_subplots(
        self,
        grid: ModelGrid,
        *,
        render_axes: list[int],
        facet_axes: list[int],
        slider_axes: list[int],
    ) -> InteractiveFigure:
        """Create an interactive figure with Plotly sliders for the slider axes.

        Each combination of slider positions becomes a Plotly frame.
        Facet axes are laid out as static subplot rows/columns.
        """
        slider_axes_objects = [grid.axes[i] for i in slider_axes]
        sliders_shape = [len(axis.values) for axis in slider_axes_objects]

        # Remap facet and render indices for grid with removed slider axes
        remapped_facet_indices = [
            i - sum(slider_index < i for slider_index in slider_axes)
            for i in facet_axes
        ]
        remapped_render_indices = [
            i - sum(slider_index < i for slider_index in slider_axes)
            for i in render_axes
        ]

        # Create figure with first frame -----------------------------------------------
        # - index 0 for each slider axis
        # - slice over all other axes
        first_frame_index: list[int | slice] = [0] * len(grid.shape)
        for non_slider_axes in itertools.chain(facet_axes, render_axes):
            first_frame_index[non_slider_axes] = slice(None)

        fig = self._render_static_subplots(
            grid[tuple(first_frame_index)],
            render_axes=remapped_render_indices,
            facet_axes=remapped_facet_indices,
        )

        # Frames for every combination of slider positions -----------------------------
        frames: list[go.Frame] = []
        for slider_positions in itertools.product(
            *(range(size) for size in sliders_shape)
        ):
            frame_index: list[int | slice] = [0] * len(grid.shape)
            for facet_idx in facet_axes:
                frame_index[facet_idx] = slice(None)
            for render_idx in render_axes:
                frame_index[render_idx] = slice(None)
            for slider_idx, position in zip(slider_axes, slider_positions):
                frame_index[slider_idx] = position

            # Plotly frames only hold trace data, not layout/axis configs.
            # Since our render() methods combine both (adding traces + configuring layout/axis),
            # we use a temporary figure to collect only the traces for each frame.
            # [We could think about separating the render() method into two separate
            # methods, one for trace data and one for layout/axis configs. However, this
            # would push more complexity onto the user. This seems an ok tradeoff at the
            # moment]
            temp_fig = self._render_static_subplots(
                grid[tuple(frame_index)],
                facet_axes=remapped_facet_indices,
                render_axes=remapped_render_indices,
            )

            frames.append(
                go.Frame(
                    name="_".join(str(i) for i in slider_positions),
                    data=list(temp_fig.data),
                )
            )

        fig.frames = frames

        fig.update_layout(
            sliders=[
                {
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
                    "y": -0.05 - 0.3 * i,
                    "yanchor": "top",
                    "font": {"size": 10},
                    "ticklen": 4,
                }
                for i, axis in enumerate(slider_axes_objects)
            ]
        )

        return InteractiveFigure(fig)


class SinglePlot(PlotRenderer, PlotOrchestrator, ABC):
    """A plot that both renders a single model and orchestrates across ModelGrids.

    This is the main base class for user-defined plots. Subclasses implement
    ``render()`` to define visualization logic, and get ``__call__()`` for free.

    Example::

        class MyPlot(SinglePlot):
            def render(self, fig, model):
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))


        MyPlot()(grid, facet_axes=("Sparsity",), slider_axes=("Correlation",))

    Note:
        If you are using ToyModel attributes which might live on a different device
        then remember to use `.detach().cpu().numpy()`!
    """

    @property
    def n_render_axes(self) -> int:
        """Return n_render_axes from PlotRenderer (class attribute)."""
        return PlotRenderer.n_render_axes

    @overload
    def _render_static_subplots(
        self,
        grid: ToyModel,
    ) -> go.Figure: ...
    @overload
    def _render_static_subplots(
        self,
        grid: ModelGrid,
        *,
        render_axes: list[int],
        facet_axes: list[int],
    ) -> go.Figure: ...
    def _render_static_subplots(
        self,
        grid: ModelGrid | ToyModel,
        *,
        render_axes: list[int] | None = None,
        facet_axes: list[int] | None = None,
    ) -> go.Figure:
        """Create a static figure with one subplot per facet combination.

        For a single ``ToyModel``, produces a 1×1 figure.
        For a ``ModelGrid``, facet_axes and render_axes must be provided.
        """
        # Shared registry ensures legend entries are deduplicated across all subplots
        legend_registry: set[str] = set()

        if isinstance(grid, ToyModel):
            fig = make_subplots(rows=1, cols=1, specs=[[{"type": self.subplot_type}]])
            self.render(
                FigureProxy(fig, row=1, col=1, legend_registry=legend_registry),
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

            n_cols = grid.shape[facet_axes[0]] if len(facet_axes) >= 1 else 1
            n_rows = grid.shape[facet_axes[1]] if len(facet_axes) >= 2 else 1

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=[
                    [{"type": self.subplot_type} for _ in range(n_cols)]
                    for _ in range(n_rows)
                ],
            )

            for row_idx, col_idx in itertools.product(range(n_rows), range(n_cols)):
                grid_index: list[int | slice] = [0] * len(grid.shape)
                for render_idx in render_axes:
                    grid_index[render_idx] = slice(None)
                if len(facet_axes) >= 1:
                    grid_index[facet_axes[0]] = col_idx
                if len(facet_axes) >= 2:
                    grid_index[facet_axes[1]] = row_idx

                self.render(
                    FigureProxy(
                        fig,
                        row=row_idx + 1,
                        col=col_idx + 1,
                        legend_registry=legend_registry,
                    ),
                    grid[tuple(grid_index)],
                )

        return fig
