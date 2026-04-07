# {py:mod}`occhio.distributions.relational`

```{py:module} occhio.distributions.relational
```

```{autodoc2-docstring} occhio.distributions.relational
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RelationalSimple <occhio.distributions.relational.RelationalSimple>`
  - ```{autodoc2-docstring} occhio.distributions.relational.RelationalSimple
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`MultiRelational <occhio.distributions.relational.MultiRelational>`
  - ```{autodoc2-docstring} occhio.distributions.relational.MultiRelational
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} RelationalSimple(n_features: int, p_active: float = 0.1, **kwargs)
:canonical: occhio.distributions.relational.RelationalSimple

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.relational.RelationalSimple
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.relational.RelationalSimple.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.relational.RelationalSimple.sample

````

````{py:method} new_On_matrix()
:canonical: occhio.distributions.relational.RelationalSimple.new_On_matrix

```{autodoc2-docstring} occhio.distributions.relational.RelationalSimple.new_On_matrix
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} MultiRelational(n_features: int, p_active: float = 0.1, k: int = 2, **kwargs)
:canonical: occhio.distributions.relational.MultiRelational

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.relational.MultiRelational
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.relational.MultiRelational.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.relational.MultiRelational.sample

````

````{py:method} new_On_matricies()
:canonical: occhio.distributions.relational.MultiRelational.new_On_matricies

```{autodoc2-docstring} occhio.distributions.relational.MultiRelational.new_On_matricies
:parser: _ext.google_docstring_parser
```

````

`````
