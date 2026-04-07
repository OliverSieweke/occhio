# {py:mod}`occhio.visualization_2.core.figure_wrappers`

```{py:module} occhio.visualization_2.core.figure_wrappers
```

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FigureProxy <occhio.visualization_2.core.figure_wrappers.FigureProxy>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`InteractiveFigure <occhio.visualization_2.core.figure_wrappers.InteractiveFigure>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} FigureProxy(fig: plotly.graph_objects.Figure, row: int, col: int, *, legend_registry: set[str] | None = None)
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} _SUBPLOT_METHODS
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._SUBPLOT_METHODS
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._SUBPLOT_METHODS
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} _AXIS_REF_KEYS
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._AXIS_REF_KEYS
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._AXIS_REF_KEYS
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} _BLOCKED_METHODS
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._BLOCKED_METHODS
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._BLOCKED_METHODS
:parser: _ext.google_docstring_parser
```

````

````{py:method} _remap_axis_refs(kwargs: dict) -> dict[str, typing.Any]
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._remap_axis_refs

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._remap_axis_refs
:parser: _ext.google_docstring_parser
```

````

````{py:method} _dedupe_legend(trace: typing.Any) -> None
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._dedupe_legend

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._dedupe_legend
:parser: _ext.google_docstring_parser
```

````

````{py:method} _dedup_axis_label() -> None
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy._dedup_axis_label

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy._dedup_axis_label
:parser: _ext.google_docstring_parser
```

````

````{py:method} __getattr__(name: str) -> typing.Any
:canonical: occhio.visualization_2.core.figure_wrappers.FigureProxy.__getattr__

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.FigureProxy.__getattr__
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} InteractiveFigure(*args, post_script: str | None = None, **kwargs)
:canonical: occhio.visualization_2.core.figure_wrappers.InteractiveFigure

Bases: {py:obj}`plotly.graph_objects.Figure`

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} _post_script
:canonical: occhio.visualization_2.core.figure_wrappers.InteractiveFigure._post_script
:type: str
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure._post_script
:parser: _ext.google_docstring_parser
```

````

````{py:method} _ipython_display_(**kwargs: typing.Any) -> None
:canonical: occhio.visualization_2.core.figure_wrappers.InteractiveFigure._ipython_display_

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure._ipython_display_
:parser: _ext.google_docstring_parser
```

````

````{py:method} show(*args, **kwargs)
:canonical: occhio.visualization_2.core.figure_wrappers.InteractiveFigure.show

```{autodoc2-docstring} occhio.visualization_2.core.figure_wrappers.InteractiveFigure.show
:parser: _ext.google_docstring_parser
```

````

`````
