# {py:mod}`occhio.distributions.sparse`

```{py:module} occhio.distributions.sparse
```

```{autodoc2-docstring} occhio.distributions.sparse
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparseUniform <occhio.distributions.sparse.SparseUniform>`
  - ```{autodoc2-docstring} occhio.distributions.sparse.SparseUniform
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SparseExponential <occhio.distributions.sparse.SparseExponential>`
  - ```{autodoc2-docstring} occhio.distributions.sparse.SparseExponential
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SingleUniform <occhio.distributions.sparse.SingleUniform>`
  - ```{autodoc2-docstring} occhio.distributions.sparse.SingleUniform
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SparseUniform(n_features: int, p_active: float | list[float] | torch.Tensor, **kwargs)
:canonical: occhio.distributions.sparse.SparseUniform

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.sparse.SparseUniform
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.sparse.SparseUniform.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.sparse.SparseUniform.sample

````

`````

`````{py:class} SparseExponential(n_features: int, p_active: float | list[float] | torch.Tensor, scale: float | list[float] | torch.Tensor = 1.0, **kwargs)
:canonical: occhio.distributions.sparse.SparseExponential

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.sparse.SparseExponential
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.sparse.SparseExponential.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.sparse.SparseExponential.sample

````

`````

`````{py:class} SingleUniform(n_features: int, **kwargs)
:canonical: occhio.distributions.sparse.SingleUniform

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.sparse.SingleUniform
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.sparse.SingleUniform.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.sparse.SingleUniform.sample

````

`````
