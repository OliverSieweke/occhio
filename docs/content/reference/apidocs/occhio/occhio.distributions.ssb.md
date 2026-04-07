# {py:mod}`occhio.distributions.ssb`

```{py:module} occhio.distributions.ssb
```

```{autodoc2-docstring} occhio.distributions.ssb
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HierarchyNode <occhio.distributions.ssb.HierarchyNode>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SyntheticBatch <occhio.distributions.ssb.SyntheticBatch>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.SyntheticBatch
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SyntheticDataConfig <occhio.distributions.ssb.SyntheticDataConfig>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`CorrelationStructure <occhio.distributions.ssb.CorrelationStructure>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.CorrelationStructure
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`FiringSampler <occhio.distributions.ssb.FiringSampler>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`MagnitudeSampler <occhio.distributions.ssb.MagnitudeSampler>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.MagnitudeSampler
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`HierarchyConstraints <occhio.distributions.ssb.HierarchyConstraints>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`_LevelOp <occhio.distributions.ssb._LevelOp>`
  - ```{autodoc2-docstring} occhio.distributions.ssb._LevelOp
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SyntheticDataModel <occhio.distributions.ssb.SyntheticDataModel>`
  - ```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_make_schedule <occhio.distributions.ssb._make_schedule>`
  - ```{autodoc2-docstring} occhio.distributions.ssb._make_schedule
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`_compute_firing_probs <occhio.distributions.ssb._compute_firing_probs>`
  - ```{autodoc2-docstring} occhio.distributions.ssb._compute_firing_probs
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DTYPE_MAP <occhio.distributions.ssb._DTYPE_MAP>`
  - ```{autodoc2-docstring} occhio.distributions.ssb._DTYPE_MAP
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} HierarchyNode
:canonical: occhio.distributions.ssb.HierarchyNode

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode
:parser: _ext.google_docstring_parser
```

````{py:attribute} feature_idx
:canonical: occhio.distributions.ssb.HierarchyNode.feature_idx
:type: int
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode.feature_idx
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} children
:canonical: occhio.distributions.ssb.HierarchyNode.children
:type: list[occhio.distributions.ssb.HierarchyNode]
:value: >
   'field(...)'

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode.children
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mutually_exclusive_children
:canonical: occhio.distributions.ssb.HierarchyNode.mutually_exclusive_children
:type: bool
:value: >
   False

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode.mutually_exclusive_children
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} parent_scaled
:canonical: occhio.distributions.ssb.HierarchyNode.parent_scaled
:type: bool
:value: >
   False

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyNode.parent_scaled
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} SyntheticBatch
:canonical: occhio.distributions.ssb.SyntheticBatch

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticBatch
:parser: _ext.google_docstring_parser
```

````{py:attribute} activations
:canonical: occhio.distributions.ssb.SyntheticBatch.activations
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticBatch.activations
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} coefficients
:canonical: occhio.distributions.ssb.SyntheticBatch.coefficients
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticBatch.coefficients
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} firing_mask
:canonical: occhio.distributions.ssb.SyntheticBatch.firing_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticBatch.firing_mask
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} SyntheticDataConfig
:canonical: occhio.distributions.ssb.SyntheticDataConfig

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig
:parser: _ext.google_docstring_parser
```

````{py:attribute} n_features
:canonical: occhio.distributions.ssb.SyntheticDataConfig.n_features
:type: int
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.n_features
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} firing_prob_distribution
:canonical: occhio.distributions.ssb.SyntheticDataConfig.firing_prob_distribution
:type: str
:value: >
   'zipfian'

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.firing_prob_distribution
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} p_min
:canonical: occhio.distributions.ssb.SyntheticDataConfig.p_min
:type: float
:value: >
   0.001

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.p_min
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} p_max
:canonical: occhio.distributions.ssb.SyntheticDataConfig.p_max
:type: float
:value: >
   0.1

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.p_max
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} alpha
:canonical: occhio.distributions.ssb.SyntheticDataConfig.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.alpha
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} p_constant
:canonical: occhio.distributions.ssb.SyntheticDataConfig.p_constant
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.p_constant
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mean_distribution
:canonical: occhio.distributions.ssb.SyntheticDataConfig.mean_distribution
:type: str
:value: >
   'constant'

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.mean_distribution
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mean_value
:canonical: occhio.distributions.ssb.SyntheticDataConfig.mean_value
:type: float
:value: >
   1.0

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.mean_value
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mean_high
:canonical: occhio.distributions.ssb.SyntheticDataConfig.mean_high
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.mean_high
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mean_low
:canonical: occhio.distributions.ssb.SyntheticDataConfig.mean_low
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.mean_low
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} std_distribution
:canonical: occhio.distributions.ssb.SyntheticDataConfig.std_distribution
:type: str
:value: >
   'constant'

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.std_distribution
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} std_value
:canonical: occhio.distributions.ssb.SyntheticDataConfig.std_value
:type: float
:value: >
   0.5

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.std_value
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} std_high
:canonical: occhio.distributions.ssb.SyntheticDataConfig.std_high
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.std_high
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} std_low
:canonical: occhio.distributions.ssb.SyntheticDataConfig.std_low
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.std_low
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} folded_normal_mu
:canonical: occhio.distributions.ssb.SyntheticDataConfig.folded_normal_mu
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.folded_normal_mu
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} folded_normal_sigma
:canonical: occhio.distributions.ssb.SyntheticDataConfig.folded_normal_sigma
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.folded_normal_sigma
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} correlation_rank
:canonical: occhio.distributions.ssb.SyntheticDataConfig.correlation_rank
:type: int
:value: >
   0

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.correlation_rank
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} correlation_scale
:canonical: occhio.distributions.ssb.SyntheticDataConfig.correlation_scale
:type: float
:value: >
   0.1

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.correlation_scale
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} delta_min
:canonical: occhio.distributions.ssb.SyntheticDataConfig.delta_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.delta_min
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} hierarchy
:canonical: occhio.distributions.ssb.SyntheticDataConfig.hierarchy
:type: typing.Optional[list[occhio.distributions.ssb.HierarchyNode]]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.hierarchy
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} compensate_probabilities
:canonical: occhio.distributions.ssb.SyntheticDataConfig.compensate_probabilities
:type: bool
:value: >
   True

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.compensate_probabilities
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} post_processing
:canonical: occhio.distributions.ssb.SyntheticDataConfig.post_processing
:type: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.post_processing
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} device
:canonical: occhio.distributions.ssb.SyntheticDataConfig.device
:type: str
:value: >
   'cpu'

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.device
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} dtype
:canonical: occhio.distributions.ssb.SyntheticDataConfig.dtype
:type: str
:value: >
   'float32'

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataConfig.dtype
:parser: _ext.google_docstring_parser
```

````

`````

````{py:data} _DTYPE_MAP
:canonical: occhio.distributions.ssb._DTYPE_MAP
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb._DTYPE_MAP
:parser: _ext.google_docstring_parser
```

````

````{py:function} _make_schedule(n: int, distribution: str, *, value: typing.Optional[float] = None, high: typing.Optional[float] = None, low: typing.Optional[float] = None, folded_mu: typing.Optional[float] = None, folded_sigma: typing.Optional[float] = None, generator: typing.Optional[torch.Generator] = None, device: torch.device | str = 'cpu') -> torch.Tensor
:canonical: occhio.distributions.ssb._make_schedule

```{autodoc2-docstring} occhio.distributions.ssb._make_schedule
:parser: _ext.google_docstring_parser
```
````

`````{py:class} CorrelationStructure(n_features: int, rank: int = 0, correlation_scale: float = 0.1, delta_min: float = 0.01, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float32, generator: typing.Optional[torch.Generator] = None)
:canonical: occhio.distributions.ssb.CorrelationStructure

```{autodoc2-docstring} occhio.distributions.ssb.CorrelationStructure
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.ssb.CorrelationStructure.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: occhio.distributions.ssb.CorrelationStructure.sample

```{autodoc2-docstring} occhio.distributions.ssb.CorrelationStructure.sample
:parser: _ext.google_docstring_parser
```

````

`````

````{py:function} _compute_firing_probs(n_features: int, distribution: str, p_min: float, p_max: float, alpha: float = 1.0, p_constant: typing.Optional[float] = None, device: torch.device | str = 'cpu', generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: occhio.distributions.ssb._compute_firing_probs

```{autodoc2-docstring} occhio.distributions.ssb._compute_firing_probs
:parser: _ext.google_docstring_parser
```
````

`````{py:class} FiringSampler(probabilities: torch.Tensor, correlation: occhio.distributions.ssb.CorrelationStructure)
:canonical: occhio.distributions.ssb.FiringSampler

```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _inv_normal_cdf(q: torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.ssb.FiringSampler._inv_normal_cdf
:staticmethod:

```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler._inv_normal_cdf
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: occhio.distributions.ssb.FiringSampler.sample

```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} _inv_normal_cdf_to_prob(thresholds: torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.ssb.FiringSampler._inv_normal_cdf_to_prob
:staticmethod:

```{autodoc2-docstring} occhio.distributions.ssb.FiringSampler._inv_normal_cdf_to_prob
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} MagnitudeSampler(means: torch.Tensor, stds: torch.Tensor)
:canonical: occhio.distributions.ssb.MagnitudeSampler

```{autodoc2-docstring} occhio.distributions.ssb.MagnitudeSampler
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.ssb.MagnitudeSampler.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(z: torch.Tensor, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: occhio.distributions.ssb.MagnitudeSampler.sample

```{autodoc2-docstring} occhio.distributions.ssb.MagnitudeSampler.sample
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} HierarchyConstraints(forest: list[occhio.distributions.ssb.HierarchyNode], n_features: int, compensate: bool = True, base_probs: typing.Optional[torch.Tensor] = None, mean_magnitudes: typing.Optional[torch.Tensor] = None, device: torch.device | str = 'cpu')
:canonical: occhio.distributions.ssb.HierarchyConstraints

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _parse_forest(forest: list[occhio.distributions.ssb.HierarchyNode]) -> None
:canonical: occhio.distributions.ssb.HierarchyConstraints._parse_forest

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints._parse_forest
:parser: _ext.google_docstring_parser
```

````

````{py:method} _compute_compensation(base_probs: torch.Tensor) -> None
:canonical: occhio.distributions.ssb.HierarchyConstraints._compute_compensation

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints._compute_compensation
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_compensated_probs(base_probs: torch.Tensor) -> torch.Tensor
:canonical: occhio.distributions.ssb.HierarchyConstraints.get_compensated_probs

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints.get_compensated_probs
:parser: _ext.google_docstring_parser
```

````

````{py:method} apply(c: torch.Tensor, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: occhio.distributions.ssb.HierarchyConstraints.apply

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints.apply
:parser: _ext.google_docstring_parser
```

````

````{py:property} has_constraints
:canonical: occhio.distributions.ssb.HierarchyConstraints.has_constraints
:type: bool

```{autodoc2-docstring} occhio.distributions.ssb.HierarchyConstraints.has_constraints
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} _LevelOp
:canonical: occhio.distributions.ssb._LevelOp

```{autodoc2-docstring} occhio.distributions.ssb._LevelOp
:parser: _ext.google_docstring_parser
```

````{py:attribute} parent_idx
:canonical: occhio.distributions.ssb._LevelOp.parent_idx
:type: int
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb._LevelOp.parent_idx
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} child_indices
:canonical: occhio.distributions.ssb._LevelOp.child_indices
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb._LevelOp.child_indices
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} mutually_exclusive
:canonical: occhio.distributions.ssb._LevelOp.mutually_exclusive
:type: bool
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb._LevelOp.mutually_exclusive
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} parent_scaled
:canonical: occhio.distributions.ssb._LevelOp.parent_scaled
:type: bool
:value: >
   None

```{autodoc2-docstring} occhio.distributions.ssb._LevelOp.parent_scaled
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} SyntheticDataModel(config: occhio.distributions.ssb.SyntheticDataConfig, seed: typing.Optional[int] = None)
:canonical: occhio.distributions.ssb.SyntheticDataModel

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.ssb.SyntheticDataModel.sample

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.ssb.SyntheticDataModel.to

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel.to
:parser: _ext.google_docstring_parser
```

````

````{py:property} firing_probabilities
:canonical: occhio.distributions.ssb.SyntheticDataModel.firing_probabilities
:type: torch.Tensor

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel.firing_probabilities
:parser: _ext.google_docstring_parser
```

````

````{py:property} compensated_probabilities
:canonical: occhio.distributions.ssb.SyntheticDataModel.compensated_probabilities
:type: torch.Tensor

```{autodoc2-docstring} occhio.distributions.ssb.SyntheticDataModel.compensated_probabilities
:parser: _ext.google_docstring_parser
```

````

`````
