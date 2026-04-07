# {py:mod}`occhio.distributions.manifold`

```{py:module} occhio.distributions.manifold
```

```{autodoc2-docstring} occhio.distributions.manifold
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SphericalDistribution <occhio.distributions.manifold.SphericalDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`TorusDistribution <occhio.distributions.manifold.TorusDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.manifold.TorusDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`HypercubeDistribution <occhio.distributions.manifold.HypercubeDistribution>`
  - ```{autodoc2-docstring} occhio.distributions.manifold.HypercubeDistribution
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SphericalDistribution(n_features: int, length_scale: float = 1.0, manifold_dim: int = 1, magnitude_range: tuple[float, float] = (0.9, 1.0), **kwargs)
:canonical: occhio.distributions.manifold.SphericalDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _place_features() -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution._place_features

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution._place_features
:parser: _ext.google_docstring_parser
```

````

````{py:method} _place_on_circle() -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution._place_on_circle

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution._place_on_circle
:parser: _ext.google_docstring_parser
```

````

````{py:method} _place_on_sphere_fibonacci() -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution._place_on_sphere_fibonacci

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution._place_on_sphere_fibonacci
:parser: _ext.google_docstring_parser
```

````

````{py:method} _place_on_sphere_random() -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution._place_on_sphere_random

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution._place_on_sphere_random
:parser: _ext.google_docstring_parser
```

````

````{py:method} _sample_direction(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution._sample_direction

```{autodoc2-docstring} occhio.distributions.manifold.SphericalDistribution._sample_direction
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.manifold.SphericalDistribution.sample

````

`````

`````{py:class} TorusDistribution(n_features: int, length_scale: float = 1.0, torus_dim: int = 1, magnitude_range: tuple[float, float] = (0.9, 1.0), **kwargs)
:canonical: occhio.distributions.manifold.TorusDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.manifold.TorusDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.manifold.TorusDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _place_features() -> torch.Tensor
:canonical: occhio.distributions.manifold.TorusDistribution._place_features

```{autodoc2-docstring} occhio.distributions.manifold.TorusDistribution._place_features
:parser: _ext.google_docstring_parser
```

````

````{py:method} _torus_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.manifold.TorusDistribution._torus_distance

```{autodoc2-docstring} occhio.distributions.manifold.TorusDistribution._torus_distance
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.manifold.TorusDistribution.sample

````

`````

`````{py:class} HypercubeDistribution(n_features: int, length_scale: float = 0.5, cube_dim: int = 1, magnitude_range: tuple[float, float] = (0.9, 1.0), **kwargs)
:canonical: occhio.distributions.manifold.HypercubeDistribution

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.manifold.HypercubeDistribution
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.manifold.HypercubeDistribution.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _place_features() -> torch.Tensor
:canonical: occhio.distributions.manifold.HypercubeDistribution._place_features

```{autodoc2-docstring} occhio.distributions.manifold.HypercubeDistribution._place_features
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.manifold.HypercubeDistribution.sample

````

`````
