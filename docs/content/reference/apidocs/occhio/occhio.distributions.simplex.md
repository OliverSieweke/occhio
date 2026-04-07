# {py:mod}`occhio.distributions.simplex`

```{py:module} occhio.distributions.simplex
```

```{autodoc2-docstring} occhio.distributions.simplex
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimplexDistribution <occhio.distributions.simplex.SimplexDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.simplex.SimplexDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SimplicialComplexDistribution <occhio.distributions.simplex.SimplicialComplexDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SimplexDistribution(simplex_sizes: list[int], p_active: float | list[float] | torch.Tensor, **kwargs)
:canonical: occhio.distributions.simplex.SimplexDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.simplex.SimplexDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.simplex.SimplexDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplexDistribution.sample

````

`````

`````{py:class} SimplicialComplexDistribution(n_vertices: int, faces: list[tuple[int, ...]], p_active: float | list[float] | torch.Tensor = 0.5, sampling_mode: typing.Literal[single, sparse] = 'single', **kwargs)
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution.sample

````

````{py:method} _sample_single(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution._sample_single

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution._sample_single
:parser: _ext.google_docstring_parser
```

````

````{py:method} _sample_single_fast(batch_size: int, face_idx: torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution._sample_single_fast

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution._sample_single_fast
:parser: _ext.google_docstring_parser
```

````

````{py:method} _sample_sparse(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution._sample_sparse

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution._sample_sparse
:parser: _ext.google_docstring_parser
```

````

````{py:method} _sample_sparse_fast(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.simplex.SimplicialComplexDistribution._sample_sparse_fast

```{autodoc2-docstring} occhio.distributions.simplex.SimplicialComplexDistribution._sample_sparse_fast
:parser: _ext.google_docstring_parser
```

````

`````
