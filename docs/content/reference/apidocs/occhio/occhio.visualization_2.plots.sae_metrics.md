# {py:mod}`occhio.visualization_2.plots.sae_metrics`

```{py:module} occhio.visualization_2.plots.sae_metrics
```

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MetricConfig <occhio.visualization_2.plots.sae_metrics.MetricConfig>`
  - ```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.MetricConfig
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SAEClassificationMetricsPlot <occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot>`
  - ```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} MetricConfig
:canonical: occhio.visualization_2.plots.sae_metrics.MetricConfig

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.MetricConfig
:parser: _ext.google_docstring_parser
```

````{py:attribute} name
:canonical: occhio.visualization_2.plots.sae_metrics.MetricConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.MetricConfig.name
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} values
:canonical: occhio.visualization_2.plots.sae_metrics.MetricConfig.values
:type: dict[str, float]
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.MetricConfig.values
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} color
:canonical: occhio.visualization_2.plots.sae_metrics.MetricConfig.color
:type: str
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.MetricConfig.color
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} SAEClassificationMetricsPlot(sae_labels: list[str] | None = None, group_by: typing.Literal[sae, metric] = 'sae')
:canonical: occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot

Bases: {py:obj}`occhio.visualization_2.core.base_plot.BasePlot`

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} sae_labels
:canonical: occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.sae_labels
:type: list[str] | None
:value: >
   None

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.sae_labels
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} group_by
:canonical: occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.group_by
:type: typing.Literal[sae, metric]
:value: >
   'sae'

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.group_by
:parser: _ext.google_docstring_parser
```

````

````{py:method} render(fig: occhio.visualization_2.core.figure_wrappers.FigureProxy, model: occhio.toy_model.ToyModel) -> None
:canonical: occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.render

```{autodoc2-docstring} occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.render
:parser: _ext.google_docstring_parser
```

````

````{py:method} configure_layout(fig: plotly.graph_objects.Figure) -> None
:canonical: occhio.visualization_2.plots.sae_metrics.SAEClassificationMetricsPlot.configure_layout

````

`````
