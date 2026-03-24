from typing import Any

import plotly.graph_objects as go
from IPython.display import HTML, display


class FigureProxy:
    """Proxy that intercepts subplot-aware Plotly methods to auto-inject row/col.

    Automatically handles:
    - **Legend deduplication**: Same trace names across subplots show only once
    - **Axis label deduplication**: X-axis labels only on bottom row, y-axis labels
      only on left column. Applied automatically after each add_trace() call and
      when update_xaxes()/update_yaxes() is called without explicit showticklabels.
      Override by passing showticklabels=True explicitly.

    Subclasses of SinglePlot can simply add traces and configure axes without
    worrying about grid context.
    """

    _SUBPLOT_METHODS = {
        "add_trace",
        "add_annotation",
        "add_shape",
        "add_hline",
        "add_vline",
        "update_xaxes",
        "update_yaxes",
    }

    _AXIS_REF_KEYS = {"xref", "yref", "axref", "ayref", "scaleanchor"}

    _BLOCKED_METHODS = {
        "update_layout": (
            "update_layout() is not allowed inside render(). "
            "Use update_xaxes/update_yaxes for per-subplot axis config, "
            "or set global layout properties at the SinglePlot/PlotGrid level."
        ),
    }

    def __init__(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        *,
        legend_registry: set[str] | None = None,
        is_composite: bool = False,
    ):
        """Create a proxy targeting a specific subplot cell.

        Args:
            fig: The underlying Plotly figure.
            row: 1-indexed subplot row.
            col: 1-indexed subplot column.
            legend_registry: Shared set for legend deduplication across subplots.
                If None, a new set is created (suitable for single-subplot figures).
            is_composite: If True, disables tick label deduplication since each
                subplot in a composite may have different axis scales.
        """
        self._fig = fig
        self.row = row
        self.col = col
        self._is_composite = is_composite

        # Extract grid dimensions from figure's internal grid reference
        if hasattr(fig, "_grid_ref") and fig._grid_ref:  # ty:ignore[has-type]
            self._n_rows = len(fig._grid_ref)  # ty:ignore[has-type]
            self._n_cols = len(fig._grid_ref[0]) if fig._grid_ref else 1  # ty:ignore[has-type]
        else:
            self._n_rows = 1
            self._n_cols = 1

        self._legend_registry: set[str] = (
            legend_registry if legend_registry is not None else set()
        )

    def _remap_axis_refs(self, kwargs: dict) -> dict[str, Any]:
        """Rewrite bare axis references (e.g. ``'x'`` → ``'x2'``) to target this subplot."""
        remapped_kwargs = kwargs.copy()

        for key in self._AXIS_REF_KEYS & kwargs.keys():
            if kwargs[key] in ("x", "y"):
                trace_kwargs = self._fig._grid_ref[self.row - 1][self.col - 1][  # ty:ignore[not-subscriptable]
                    0
                ].trace_kwargs
                remapped_kwargs[key] = trace_kwargs[f"{kwargs[key]}axis"]
            else:
                raise ValueError(
                    f"FigureProxy only supports bare 'x'/'y' axis references, "
                    f"got {key}={kwargs[key]!r}. The proxy automatically remaps "
                    f"these to target the correct subplot."
                )

        return remapped_kwargs

    def _dedupe_legend(self, trace: Any) -> None:
        """Apply legend deduplication to a trace.

        If the trace name has been seen before, sets showlegend=False and
        ensures legendgroup is set for proper hover/click behavior.
        """
        name: str | None = getattr(trace, "name", None)
        # Ignore traces with no name
        if name is None:
            return

        # Respect explicitly set showlegend=False
        if trace.showlegend is False:
            return

        # Set legendgroup if not already set (ensures clicking legend toggles all)
        if getattr(trace, "legendgroup", None) is None:
            trace.legendgroup = name

        # Only show legend for first occurrence of each name
        if name in self._legend_registry:
            trace.showlegend = False
        else:
            self._legend_registry.add(name)

    def _dedup_axis_label(self) -> None:
        """Apply axis label deduplication rules to current subplot.

        For composite plots, all subplots show tick labels. For single plots
        faceted across a grid, x-axis labels only show on the bottom row and
        y-axis labels only on the left column.
        Only applies if the user hasn't already explicitly set showticklabels to True.
        """
        # Get the axis names for this subplot
        if hasattr(self._fig, "_grid_ref") and self._fig._grid_ref:
            cell_ref = self._fig._grid_ref[self.row - 1][self.col - 1][0]
            trace_kwargs = getattr(cell_ref, "trace_kwargs")

            # Domain subplots (e.g., Indicator, Pie) have SubplotDomain objects
            # instead of refs with trace_kwargs - skip them
            # Skip traces that don't have x/y axes
            if (
                trace_kwargs is None
                or "xaxis" not in trace_kwargs
                or "yaxis" not in trace_kwargs
            ):
                return

            # Update x-axis: show ticks for composites or bottom row of faceted grids
            # trace_kwargs has 'xaxis': 'x', 'x2', etc. We need to convert to 'xaxis', 'xaxis2'
            xaxis_name = trace_kwargs["xaxis"].replace("x", "xaxis")
            xaxis = getattr(self._fig.layout, xaxis_name, None)
            if xaxis and xaxis.showticklabels is None:
                xaxis.showticklabels = self._is_composite or self.row == self._n_rows

            # Update y-axis: show ticks for composites or left column of faceted grids
            # trace_kwargs has 'yaxis': 'y', 'y2', etc. We need to convert to 'yaxis', 'yaxis2'
            yaxis_name = trace_kwargs["yaxis"].replace("y", "yaxis")
            yaxis = getattr(self._fig.layout, yaxis_name, None)
            if yaxis and yaxis.showticklabels is None:
                yaxis.showticklabels = self._is_composite or self.col == 1

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped figure.

        Subplot-aware methods (e.g. ``add_trace``) automatically receive
        ``row``/``col``. Methods in ``_BLOCKED_METHODS`` raise an error.
        """
        if name in self._BLOCKED_METHODS:
            msg = self._BLOCKED_METHODS[name]
            raise AttributeError(msg)

        attr = getattr(self._fig, name)

        if name == "add_trace" and callable(attr):

            def add_trace_wrapper(trace: Any, *args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("row", self.row)
                kwargs.setdefault("col", self.col)
                kwargs = self._remap_axis_refs(kwargs)
                self._dedupe_legend(trace)
                result = attr(trace, *args, **kwargs)

                # After adding trace, apply axis label deduplication
                # (Plotly shows tick labels by default when traces have x/y data)
                self._dedup_axis_label()

                return result

            return add_trace_wrapper

        if name == "update_xaxes" and callable(attr):

            def update_xaxes_wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("row", self.row)
                kwargs.setdefault("col", self.col)
                kwargs = self._remap_axis_refs(kwargs)
                # Show ticks for composites or bottom row of faceted grids
                if "showticklabels" not in kwargs:
                    kwargs["showticklabels"] = (
                        self._is_composite or self.row == self._n_rows
                    )
                # Show axis titles only on bottom row
                if (
                    "title_text" in kwargs
                    and not self._is_composite
                    and self.row != self._n_rows
                ):
                    kwargs["title_text"] = ""
                return attr(*args, **kwargs)

            return update_xaxes_wrapper

        if name == "update_yaxes" and callable(attr):

            def update_yaxes_wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("row", self.row)
                kwargs.setdefault("col", self.col)
                kwargs = self._remap_axis_refs(kwargs)
                # Show ticks for composites or left column of faceted grids
                if "showticklabels" not in kwargs:
                    kwargs["showticklabels"] = self._is_composite or self.col == 1
                # Show axis titles only on left column
                if "title_text" in kwargs and not self._is_composite and self.col != 1:
                    kwargs["title_text"] = ""
                return attr(*args, **kwargs)

            return update_yaxes_wrapper

        if name in self._SUBPLOT_METHODS and callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("row", self.row)
                kwargs.setdefault("col", self.col)
                kwargs = self._remap_axis_refs(kwargs)
                return attr(*args, **kwargs)

            return wrapper

        return attr


class InteractiveFigure(go.Figure):
    """A Plotly Figure that carries a post-render JavaScript snippet.

    Used for multi-slider animations where a JS callback coordinates
    slider state with frame selection. Behaves identically to a normal
    Figure when no script is attached.
    """

    _post_script: str

    def __init__(self, *args, post_script: str | None = None, **kwargs):
        """Wrap a Plotly figure with a JS callback for multi-slider sync."""
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

    def _ipython_display_(self, **kwargs: Any) -> None:
        """Render in Jupyter with the post-render JS script injected."""
        html = self.to_html(
            post_script=self._post_script,
            full_html=False,
            include_plotlyjs="require",
            auto_play=False,
        )
        display(HTML(html))

    def show(self, *args, **kwargs):
        """Display the figure with the post-render JS script injected."""
        html = self.to_html(
            post_script=self._post_script,
            full_html=False,
            include_plotlyjs="require",
            auto_play=False,
        )
        display(HTML(html))
