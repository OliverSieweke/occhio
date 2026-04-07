# {py:mod}`occhio.distributions.correlated`

```{py:module} occhio.distributions.correlated
```

```{autodoc2-docstring} occhio.distributions.correlated
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HierarchicalPairs <occhio.distributions.correlated.HierarchicalPairs>`
  - ```{autodoc2-docstring} occhio.distributions.correlated.HierarchicalPairs
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`ScaledHierarchicalPairs <occhio.distributions.correlated.ScaledHierarchicalPairs>`
  - ```{autodoc2-docstring} occhio.distributions.correlated.ScaledHierarchicalPairs
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`CorrelatedPairs <occhio.distributions.correlated.CorrelatedPairs>`
  - ```{autodoc2-docstring} occhio.distributions.correlated.CorrelatedPairs
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`AnticorrelatedPairs <occhio.distributions.correlated.AnticorrelatedPairs>`
  - ```{autodoc2-docstring} occhio.distributions.correlated.AnticorrelatedPairs
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`GaussianCorrelated <occhio.distributions.correlated.GaussianCorrelated>`
  - ```{autodoc2-docstring} occhio.distributions.correlated.GaussianCorrelated
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} HierarchicalPairs(n_features: int, p_active: float | list[float] | torch.Tensor, p_follow: float | list[float] | numpy.ndarray | torch.Tensor = 0.5, beta: float | list[float] | numpy.ndarray | torch.Tensor | None = None, **kwargs)
:canonical: occhio.distributions.correlated.HierarchicalPairs

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.correlated.HierarchicalPairs
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.correlated.HierarchicalPairs.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.correlated.HierarchicalPairs.sample

````

`````

`````{py:class} ScaledHierarchicalPairs(n_features: int, p_active: float | list[float] | torch.Tensor, p_follow: float | list[float] | torch.Tensor = 0.5, **kwargs)
:canonical: occhio.distributions.correlated.ScaledHierarchicalPairs

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.correlated.ScaledHierarchicalPairs
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.correlated.ScaledHierarchicalPairs.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.correlated.ScaledHierarchicalPairs.sample

````

`````

`````{py:class} CorrelatedPairs(n_features: int, p_active: float | list[float] | numpy.ndarray | torch.Tensor | None = None, p_individual: float | list[float] | numpy.ndarray | torch.Tensor | None = None, correlation: float | list[float] | numpy.ndarray | torch.Tensor | None = None, density: float | list[float] | numpy.ndarray | torch.Tensor | None = None, **kwargs)
:canonical: occhio.distributions.correlated.CorrelatedPairs

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.correlated.CorrelatedPairs
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.correlated.CorrelatedPairs.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.correlated.CorrelatedPairs.sample

````

`````

`````{py:class} AnticorrelatedPairs(n_features: int, p_active: float | list[float] | torch.Tensor, **kwargs)
:canonical: occhio.distributions.correlated.AnticorrelatedPairs

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.correlated.AnticorrelatedPairs
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.correlated.AnticorrelatedPairs.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.correlated.AnticorrelatedPairs.sample

````

`````

`````{py:class} GaussianCorrelated(n_features: int, p_active: float | list[float] | torch.Tensor, correlation_matrix: torch.Tensor | None = None, n_factors: int | None = None, factor_scale: float = 1.0, **kwargs)
:canonical: occhio.distributions.correlated.GaussianCorrelated

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.correlated.GaussianCorrelated
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.correlated.GaussianCorrelated.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _random_correlation_matrix(n: int, k: int, scale: float) -> torch.Tensor
:canonical: occhio.distributions.correlated.GaussianCorrelated._random_correlation_matrix

```{autodoc2-docstring} occhio.distributions.correlated.GaussianCorrelated._random_correlation_matrix
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.correlated.GaussianCorrelated.sample

````

`````
