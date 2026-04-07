# {py:mod}`occhio.distributions.dag`

```{py:module} occhio.distributions.dag
```

```{autodoc2-docstring} occhio.distributions.dag
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DAGDistribution <occhio.distributions.dag.DAGDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`DAGBayesianPropagation <occhio.distributions.dag.DAGBayesianPropagation>`
  - ```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`DAGRandomWalkToRoot <occhio.distributions.dag.DAGRandomWalkToRoot>`
  - ```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`PowerLawDigraph <occhio.distributions.dag.PowerLawDigraph>`
  - ```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DAGDistribution(n_features: int, p_active: float = 0.1, p_edge: float = 0.1, **kwargs)
:canonical: occhio.distributions.dag.DAGDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _generate_dag() -> torch.Tensor
:canonical: occhio.distributions.dag.DAGDistribution._generate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution._generate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} regenerate_dag() -> None
:canonical: occhio.distributions.dag.DAGDistribution.regenerate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution.regenerate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.dag.DAGDistribution.sample

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.dag.DAGDistribution.to

```{autodoc2-docstring} occhio.distributions.dag.DAGDistribution.to
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} DAGBayesianPropagation(n_features: int, p_active: float = 0.1, p_edge: float = 0.1, **kwargs)
:canonical: occhio.distributions.dag.DAGBayesianPropagation

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _generate_dag() -> torch.Tensor
:canonical: occhio.distributions.dag.DAGBayesianPropagation._generate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation._generate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_parent_cache() -> None
:canonical: occhio.distributions.dag.DAGBayesianPropagation._build_parent_cache

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation._build_parent_cache
:parser: _ext.google_docstring_parser
```

````

````{py:method} regenerate_dag() -> None
:canonical: occhio.distributions.dag.DAGBayesianPropagation.regenerate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation.regenerate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.dag.DAGBayesianPropagation.sample

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_expected_activation(n_samples: int = 10000) -> torch.Tensor
:canonical: occhio.distributions.dag.DAGBayesianPropagation.get_expected_activation

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation.get_expected_activation
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.dag.DAGBayesianPropagation.to

```{autodoc2-docstring} occhio.distributions.dag.DAGBayesianPropagation.to
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} DAGRandomWalkToRoot(n_features: int, p_edge: float = 0.5, adjacency: torch.Tensor | numpy.ndarray | None = None, beta: float = 1.0, p_active: list[float] | torch.Tensor | None = None, shrinking: bool = True, **kwargs)
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} adjacency
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.adjacency
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.adjacency
:parser: _ext.google_docstring_parser
```

````

````{py:method} _generate_dag() -> torch.Tensor
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot._generate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot._generate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} regenerate_dag() -> None
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.regenerate_dag

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.regenerate_dag
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_parent_cache() -> None
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot._build_parent_cache

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot._build_parent_cache
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.sample

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.to

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.to
:parser: _ext.google_docstring_parser
```

````

````{py:method} print_graph(labels=None, center: int | None = None)
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.print_graph

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.print_graph
:parser: _ext.google_docstring_parser
```

````

````{py:method} print_sources_and_sinks(labels=None)
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.print_sources_and_sinks

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.print_sources_and_sinks
:parser: _ext.google_docstring_parser
```

````

````{py:method} print_connected_components(labels=None)
:canonical: occhio.distributions.dag.DAGRandomWalkToRoot.print_connected_components

```{autodoc2-docstring} occhio.distributions.dag.DAGRandomWalkToRoot.print_connected_components
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} PowerLawDigraph(n_features: int, alpha: float = 1.0, p_edge: float = 0.1, p_active: float | list[float] | torch.Tensor = 0.05, p_child: float | tuple[float, float] = 0.9, value_dist: typing.Literal[uniform, exponential] = 'uniform', **kwargs)
:canonical: occhio.distributions.dag.PowerLawDigraph

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _generate_graph() -> torch.Tensor
:canonical: occhio.distributions.dag.PowerLawDigraph._generate_graph

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph._generate_graph
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_log_survival() -> None
:canonical: occhio.distributions.dag.PowerLawDigraph._build_log_survival

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph._build_log_survival
:parser: _ext.google_docstring_parser
```

````

````{py:method} regenerate_graph() -> None
:canonical: occhio.distributions.dag.PowerLawDigraph.regenerate_graph

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.regenerate_graph
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.dag.PowerLawDigraph.sample

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} in_degrees() -> torch.Tensor
:canonical: occhio.distributions.dag.PowerLawDigraph.in_degrees

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.in_degrees
:parser: _ext.google_docstring_parser
```

````

````{py:method} out_degrees() -> torch.Tensor
:canonical: occhio.distributions.dag.PowerLawDigraph.out_degrees

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.out_degrees
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_expected_activation(n_samples: int = 10000) -> torch.Tensor
:canonical: occhio.distributions.dag.PowerLawDigraph.get_expected_activation

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.get_expected_activation
:parser: _ext.google_docstring_parser
```

````

````{py:method} print_graph(labels=None, center: int | None = None)
:canonical: occhio.distributions.dag.PowerLawDigraph.print_graph

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.print_graph
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.dag.PowerLawDigraph.to

```{autodoc2-docstring} occhio.distributions.dag.PowerLawDigraph.to
:parser: _ext.google_docstring_parser
```

````

`````
