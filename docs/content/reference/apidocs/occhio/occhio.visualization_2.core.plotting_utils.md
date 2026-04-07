# {py:mod}`occhio.visualization_2.core.plotting_utils`

```{py:module} occhio.visualization_2.core.plotting_utils
```

```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_format_axis_value <occhio.visualization_2.core.plotting_utils._format_axis_value>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils._format_axis_value
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`model_domain_center <occhio.visualization_2.core.plotting_utils.model_domain_center>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils.model_domain_center
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`add_grid_headers <occhio.visualization_2.core.plotting_utils.add_grid_headers>`
  - ```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils.add_grid_headers
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

````{py:function} _format_axis_value(val: int | float | torch.Tensor) -> str
:canonical: occhio.visualization_2.core.plotting_utils._format_axis_value

```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils._format_axis_value
:parser: _ext.google_docstring_parser
```
````

````{py:function} model_domain_center(fig: plotly.graph_objects.Figure, model_row: int, model_col: int, inner_rows: int, inner_cols: int) -> tuple[float, float]
:canonical: occhio.visualization_2.core.plotting_utils.model_domain_center

```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils.model_domain_center
:parser: _ext.google_docstring_parser
```
````

````{py:function} add_grid_headers(fig: plotly.graph_objects.Figure, grid: occhio.ModelGrid, inner_rows: int = 1, inner_cols: int = 1, facet_axes: list[int] | None = None) -> None
:canonical: occhio.visualization_2.core.plotting_utils.add_grid_headers

```{autodoc2-docstring} occhio.visualization_2.core.plotting_utils.add_grid_headers
:parser: _ext.google_docstring_parser
```
````
