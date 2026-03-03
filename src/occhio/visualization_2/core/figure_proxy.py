import plotly.graph_objects as go


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
        self._fig = fig
        self.row = row
        self.col = col

    def _remap_axis_refs(self, kwargs: dict) -> dict:
        """Rewrite bare axis references (e.g. 'x' → 'x2') to target this subplot."""
        remapped_kwargs = kwargs.copy()

        for key in self._AXIS_REF_KEYS & kwargs.keys():
            if kwargs[key] in ("x", "y"):
                trace_kwargs = self._fig._grid_ref[self.row - 1][self.col - 1][
                    0
                ].trace_kwargs  # ty:ignore[not-subscriptable]
                remapped_kwargs[key] = trace_kwargs[f"{kwargs[key]}axis"]
            else:
                raise ValueError(
                    f"FigureProxy only supports bare 'x'/'y' axis references, "
                    f"got {key}={kwargs[key]!r}. The proxy automatically remaps "
                    f"these to target the correct subplot."
                )

        return remapped_kwargs

    def __getattr__(self, name: str):
        if name in self._BLOCKED_METHODS:
            msg = self._BLOCKED_METHODS[name]
            raise AttributeError(msg)

        attr = getattr(self._fig, name)

        if name in self._SUBPLOT_METHODS and callable(attr):

            def wrapper(*args, **kwargs):
                kwargs.setdefault("row", self.row)
                kwargs.setdefault("col", self.col)
                kwargs = self._remap_axis_refs(kwargs)
                return attr(*args, **kwargs)

            return wrapper

        return attr
