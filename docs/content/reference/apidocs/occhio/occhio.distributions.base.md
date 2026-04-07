# {py:mod}`occhio.distributions.base`

```{py:module} occhio.distributions.base
```

```{autodoc2-docstring} occhio.distributions.base
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Distribution <occhio.distributions.base.Distribution>`
  - ```{autodoc2-docstring} occhio.distributions.base.Distribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`DistributionStack <occhio.distributions.base.DistributionStack>`
  - ```{autodoc2-docstring} occhio.distributions.base.DistributionStack
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SKIP <occhio.distributions.base._SKIP>`
  - ```{autodoc2-docstring} occhio.distributions.base._SKIP
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

````{py:data} _SKIP
:canonical: occhio.distributions.base._SKIP
:value: >
   'object(...)'

```{autodoc2-docstring} occhio.distributions.base._SKIP
:parser: _ext.google_docstring_parser
```

````

`````{py:class} Distribution(n_features: int, device: torch.device | str | None = None, generator: torch.Generator | None = None)
:canonical: occhio.distributions.base.Distribution

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} occhio.distributions.base.Distribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.base.Distribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution.sample
:abstractmethod:

```{autodoc2-docstring} occhio.distributions.base.Distribution.sample
:parser: _ext.google_docstring_parser
```

````

````{py:property} _defines_generators
:canonical: occhio.distributions.base.Distribution._defines_generators
:type: bool

```{autodoc2-docstring} occhio.distributions.base.Distribution._defines_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} collect_generators() -> list[torch.Generator | None]
:canonical: occhio.distributions.base.Distribution.collect_generators

```{autodoc2-docstring} occhio.distributions.base.Distribution.collect_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} sync_generators(generators: torch.Generator | None | list[torch.Generator | None]) -> None
:canonical: occhio.distributions.base.Distribution.sync_generators

```{autodoc2-docstring} occhio.distributions.base.Distribution.sync_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} _rand(*shape) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution._rand

```{autodoc2-docstring} occhio.distributions.base.Distribution._rand
:parser: _ext.google_docstring_parser
```

````

````{py:method} _randn(*shape) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution._randn

```{autodoc2-docstring} occhio.distributions.base.Distribution._randn
:parser: _ext.google_docstring_parser
```

````

````{py:method} _rand_On(num_feat) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution._rand_On

```{autodoc2-docstring} occhio.distributions.base.Distribution._rand_On
:parser: _ext.google_docstring_parser
```

````

````{py:method} _randint(low: int, high: int, shape: tuple[int, ...], p: torch.Tensor | None = None) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution._randint

```{autodoc2-docstring} occhio.distributions.base.Distribution._randint
:parser: _ext.google_docstring_parser
```

````

````{py:method} _broadcast(x: float | list[float] | numpy.ndarray | torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.base.Distribution._broadcast

```{autodoc2-docstring} occhio.distributions.base.Distribution._broadcast
:parser: _ext.google_docstring_parser
```

````

````{py:method} save_samples(n_samples: int, path: str | pathlib.Path | None = None) -> pathlib.Path
:canonical: occhio.distributions.base.Distribution.save_samples

```{autodoc2-docstring} occhio.distributions.base.Distribution.save_samples
:parser: _ext.google_docstring_parser
```

````

````{py:method} _collect_attrs() -> dict
:canonical: occhio.distributions.base.Distribution._collect_attrs

```{autodoc2-docstring} occhio.distributions.base.Distribution._collect_attrs
:parser: _ext.google_docstring_parser
```

````

````{py:method} _serialize_value(v)
:canonical: occhio.distributions.base.Distribution._serialize_value
:staticmethod:

```{autodoc2-docstring} occhio.distributions.base.Distribution._serialize_value
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.base.Distribution.to

```{autodoc2-docstring} occhio.distributions.base.Distribution.to
:parser: _ext.google_docstring_parser
```

````

````{py:method} __repr__()
:canonical: occhio.distributions.base.Distribution.__repr__

````

````{py:method} __str__()
:canonical: occhio.distributions.base.Distribution.__str__

````

````{py:property} _equivalence_hash
:canonical: occhio.distributions.base.Distribution._equivalence_hash
:type: str

```{autodoc2-docstring} occhio.distributions.base.Distribution._equivalence_hash
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} DistributionStack(distributions: list[occhio.distributions.base.Distribution], sampling_mode: typing.Literal[independent, sparse, single] = 'independent', p_meta: float | None = None, **kwargs)
:canonical: occhio.distributions.base.DistributionStack

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.base.DistributionStack
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.base.DistributionStack.__init__
:parser: _ext.google_docstring_parser
```

````{py:property} _defines_generators
:canonical: occhio.distributions.base.DistributionStack._defines_generators
:type: bool

```{autodoc2-docstring} occhio.distributions.base.DistributionStack._defines_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} collect_generators() -> list[torch.Generator | None]
:canonical: occhio.distributions.base.DistributionStack.collect_generators

```{autodoc2-docstring} occhio.distributions.base.DistributionStack.collect_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} sync_generators(generators: torch.Generator | None | list[torch.Generator | None]) -> None
:canonical: occhio.distributions.base.DistributionStack.sync_generators

```{autodoc2-docstring} occhio.distributions.base.DistributionStack.sync_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size)
:canonical: occhio.distributions.base.DistributionStack.sample

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.base.DistributionStack.to

```{autodoc2-docstring} occhio.distributions.base.DistributionStack.to
:parser: _ext.google_docstring_parser
```

````

````{py:method} __repr__()
:canonical: occhio.distributions.base.DistributionStack.__repr__

````

````{py:property} _equivalence_hash
:canonical: occhio.distributions.base.DistributionStack._equivalence_hash
:type: str

````

````{py:method} _validate_stack(distributions, sampling_mode, p_meta, **kwargs) -> None
:canonical: occhio.distributions.base.DistributionStack._validate_stack

```{autodoc2-docstring} occhio.distributions.base.DistributionStack._validate_stack
:parser: _ext.google_docstring_parser
```

````

`````
