# {py:mod}`occhio.visualization_2.core.base_plot`

```{py:module} occhio.visualization_2.core.base_plot
```

```{autodoc2-docstring} occhio.visualization_2.core.base_plot
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PlotRenderer <occhio.visualization_2.core.base_plot.PlotRenderer>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.base_plot.PlotRenderer
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`BasePlot <occhio.visualization_2.core.base_plot.BasePlot>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AxisSpec <occhio.visualization_2.core.base_plot.AxisSpec>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.base_plot.AxisSpec
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PlotRenderer
:canonical: occhio.visualization_2.core.base_plot.PlotRenderer

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.PlotRenderer
:parser: _ext.google_docstring_parser
```

````{py:method} render(fig: occhio.visualization_2.core.figure_wrappers.FigureProxy, model: occhio.toy_model.ToyModel) -> None
:canonical: occhio.visualization_2.core.base_plot.PlotRenderer.render
:abstractmethod:

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.PlotRenderer.render
:parser: _ext.google_docstring_parser
```

````

````{py:method} configure_layout(fig: plotly.graph_objects.Figure) -> None
:canonical: occhio.visualization_2.core.base_plot.PlotRenderer.configure_layout

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.PlotRenderer.configure_layout
:parser: _ext.google_docstring_parser
```

````

`````

````{py:data} AxisSpec
:canonical: occhio.visualization_2.core.base_plot.AxisSpec
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.AxisSpec
:parser: _ext.google_docstring_parser
```

````

`````{py:class} BasePlot
:canonical: occhio.visualization_2.core.base_plot.BasePlot

Bases: {py:obj}`occhio.visualization_2.core.base_plot.PlotRenderer`, {py:obj}`abc.ABC`

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot
:parser: _ext.google_docstring_parser
```

````{py:method} _resolve_axis_index(spec: occhio.visualization_2.core.base_plot.AxisSpec, grid: occhio.model_grid.ModelGrid) -> int
:canonical: occhio.visualization_2.core.base_plot.BasePlot._resolve_axis_index
:staticmethod:

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot._resolve_axis_index
:parser: _ext.google_docstring_parser
```

````

````{py:method} _resolve_full_axes(grid: occhio.model_grid.ModelGrid, *, facet_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None, slider_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None, render_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None) -> tuple[list[int], list[int], list[int]]
:canonical: occhio.visualization_2.core.base_plot.BasePlot._resolve_full_axes
:staticmethod:

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot._resolve_full_axes
:parser: _ext.google_docstring_parser
```

````

````{py:method} __call__(models: occhio.toy_model.ToyModel | occhio.model_grid.ModelGrid, height: int | None = None, width: int | None = None, *, facet_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None = None, slider_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None = None, render_axes: typing.Sequence[occhio.visualization_2.core.base_plot.AxisSpec] | None = None) -> plotly.graph_objects.Figure
:canonical: occhio.visualization_2.core.base_plot.BasePlot.__call__

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot.__call__
:parser: _ext.google_docstring_parser
```

````

````{py:method} _render_static_subplots(grid: occhio.model_grid.ModelGrid | occhio.toy_model.ToyModel, *, render_axes: list[int] | None = None, facet_axes: list[int] | None = None) -> plotly.graph_objects.Figure
:canonical: occhio.visualization_2.core.base_plot.BasePlot._render_static_subplots

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot._render_static_subplots
:parser: _ext.google_docstring_parser
```

````

````{py:method} _render_animated_subplots(grid: occhio.model_grid.ModelGrid, *, render_axes: list[int], facet_axes: list[int], slider_axes: list[int]) -> occhio.visualization_2.core.figure_wrappers.InteractiveFigure
:canonical: occhio.visualization_2.core.base_plot.BasePlot._render_animated_subplots

```{autodoc2-docstring} occhio.visualization_2.core.base_plot.BasePlot._render_animated_subplots
:parser: _ext.google_docstring_parser
```

````

`````
