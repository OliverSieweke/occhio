from typing import Any

import plotly.graph_objects as go
from IPython.display import HTML, display


class FigureProxy:
    """Proxy that intercepts subplot-aware Plotly methods to auto-inject row/col."""

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
            "or set global layout properties at the BasePlot/PlotGrid level."
        ),
    }

    def __init__(self, fig: go.Figure, row: int, col: int):
        """Create a proxy targeting a specific subplot cell.

        Args:
            fig: The underlying Plotly figure.
            row: 1-indexed subplot row.
            col: 1-indexed subplot column.
        """
        self._fig = fig
        self.row = row
        self.col = col

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

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped figure.

        Subplot-aware methods (e.g. ``add_trace``) automatically receive
        ``row``/``col``. Methods in ``_BLOCKED_METHODS`` raise an error.
        """
        if name in self._BLOCKED_METHODS:
            msg = self._BLOCKED_METHODS[name]
            raise AttributeError(msg)

        attr = getattr(self._fig, name)

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
