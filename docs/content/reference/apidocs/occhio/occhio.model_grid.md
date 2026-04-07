# {py:mod}`occhio.model_grid`

```{py:module} occhio.model_grid
```

```{autodoc2-docstring} occhio.model_grid
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Axis <occhio.model_grid.Axis>`
  - ```{autodoc2-docstring} occhio.model_grid.Axis
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`TrainingAxis <occhio.model_grid.TrainingAxis>`
  - ```{autodoc2-docstring} occhio.model_grid.TrainingAxis
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`ModelGrid <occhio.model_grid.ModelGrid>`
  - ```{autodoc2-docstring} occhio.model_grid.ModelGrid
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} Axis(label: str, values: torch.Tensor | collections.abc.Sequence[float | int])
:canonical: occhio.model_grid.Axis

```{autodoc2-docstring} occhio.model_grid.Axis
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.model_grid.Axis.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} label
:canonical: occhio.model_grid.Axis.label
:type: str
:value: >
   None

```{autodoc2-docstring} occhio.model_grid.Axis.label
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} values
:canonical: occhio.model_grid.Axis.values
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} occhio.model_grid.Axis.values
:parser: _ext.google_docstring_parser
```

````

`````

````{py:class} TrainingAxis(values: torch.Tensor | collections.abc.Sequence[int], label: str = 'Epoch')
:canonical: occhio.model_grid.TrainingAxis

Bases: {py:obj}`occhio.model_grid.Axis`

```{autodoc2-docstring} occhio.model_grid.TrainingAxis
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.model_grid.TrainingAxis.__init__
:parser: _ext.google_docstring_parser
```

````

`````{py:class} ModelGrid(create_model: typing.Callable[[dict[str, typing.Any]], occhio.toy_model.ToyModel], axes: list[occhio.model_grid.Axis], broadcast_samples: bool = True, *, _models: numpy.typing.NDArray[numpy.object_] | None = None)
:canonical: occhio.model_grid.ModelGrid

```{autodoc2-docstring} occhio.model_grid.ModelGrid
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.model_grid.ModelGrid.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} models
:canonical: occhio.model_grid.ModelGrid.models
:type: numpy.typing.NDArray[numpy.object_]
:value: >
   None

```{autodoc2-docstring} occhio.model_grid.ModelGrid.models
:parser: _ext.google_docstring_parser
```

````

````{py:method} from_iterable(models: collections.abc.Iterable[occhio.toy_model.ToyModel]) -> occhio.model_grid.ModelGrid
:canonical: occhio.model_grid.ModelGrid.from_iterable
:staticmethod:

```{autodoc2-docstring} occhio.model_grid.ModelGrid.from_iterable
:parser: _ext.google_docstring_parser
```

````

````{py:method} _initialize_models() -> numpy.typing.NDArray[numpy.object_]
:canonical: occhio.model_grid.ModelGrid._initialize_models

```{autodoc2-docstring} occhio.model_grid.ModelGrid._initialize_models
:parser: _ext.google_docstring_parser
```

````

````{py:method} _validate_args(create_model: typing.Callable[..., occhio.toy_model.ToyModel], axes: list[occhio.model_grid.Axis]) -> None
:canonical: occhio.model_grid.ModelGrid._validate_args

```{autodoc2-docstring} occhio.model_grid.ModelGrid._validate_args
:parser: _ext.google_docstring_parser
```

````

````{py:method} _validate_vmap() -> None
:canonical: occhio.model_grid.ModelGrid._validate_vmap

```{autodoc2-docstring} occhio.model_grid.ModelGrid._validate_vmap
:parser: _ext.google_docstring_parser
```

````

````{py:method} _validate_generators() -> None
:canonical: occhio.model_grid.ModelGrid._validate_generators

```{autodoc2-docstring} occhio.model_grid.ModelGrid._validate_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_broadcast() -> tuple[list[occhio.distributions.base.Distribution], torch.Tensor]
:canonical: occhio.model_grid.ModelGrid._build_broadcast

```{autodoc2-docstring} occhio.model_grid.ModelGrid._build_broadcast
:parser: _ext.google_docstring_parser
```

````

````{py:method} _sync_generators(broadcasters: list[occhio.distributions.base.Distribution], broadcast_map: torch.Tensor) -> None
:canonical: occhio.model_grid.ModelGrid._sync_generators

```{autodoc2-docstring} occhio.model_grid.ModelGrid._sync_generators
:parser: _ext.google_docstring_parser
```

````

````{py:method} _can_vectorize_loss() -> bool
:canonical: occhio.model_grid.ModelGrid._can_vectorize_loss

```{autodoc2-docstring} occhio.model_grid.ModelGrid._can_vectorize_loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} parameters_mesh()
:canonical: occhio.model_grid.ModelGrid.parameters_mesh

```{autodoc2-docstring} occhio.model_grid.ModelGrid.parameters_mesh
:parser: _ext.google_docstring_parser
```

````

````{py:property} _shape_from_axes
:canonical: occhio.model_grid.ModelGrid._shape_from_axes
:type: tuple[int, ...]

```{autodoc2-docstring} occhio.model_grid.ModelGrid._shape_from_axes
:parser: _ext.google_docstring_parser
```

````

````{py:property} shape
:canonical: occhio.model_grid.ModelGrid.shape
:type: tuple[int, ...]

```{autodoc2-docstring} occhio.model_grid.ModelGrid.shape
:parser: _ext.google_docstring_parser
```

````

````{py:property} description
:canonical: occhio.model_grid.ModelGrid.description
:type: dict[str, int]

```{autodoc2-docstring} occhio.model_grid.ModelGrid.description
:parser: _ext.google_docstring_parser
```

````

````{py:method} save_models(path: str | None = None) -> None
:canonical: occhio.model_grid.ModelGrid.save_models

```{autodoc2-docstring} occhio.model_grid.ModelGrid.save_models
:parser: _ext.google_docstring_parser
```

````

````{py:method} load_models(path: str) -> None
:canonical: occhio.model_grid.ModelGrid.load_models

```{autodoc2-docstring} occhio.model_grid.ModelGrid.load_models
:parser: _ext.google_docstring_parser
```

````

````{py:method} fit(n_epochs: int = 10000, batch_size: int = 512, learning_rate: float = 0.0003, weight_decay: float = 0.05, verbose: bool = False, compile: bool = False, track_losses: bool = False, snapshot_interval: int | None = None, sample_every: int = 25) -> occhio.model_grid.ModelGrid | list[float] | None
:canonical: occhio.model_grid.ModelGrid.fit

```{autodoc2-docstring} occhio.model_grid.ModelGrid.fit
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_history_grid(snapshots: list[tuple[int, dict, dict]], flattened_models: numpy.typing.NDArray[numpy.object_]) -> occhio.model_grid.ModelGrid
:canonical: occhio.model_grid.ModelGrid._build_history_grid

```{autodoc2-docstring} occhio.model_grid.ModelGrid._build_history_grid
:parser: _ext.google_docstring_parser
```

````

````{py:method} __getitem__(key) -> occhio.model_grid.ModelGrid | occhio.toy_model.ToyModel
:canonical: occhio.model_grid.ModelGrid.__getitem__

```{autodoc2-docstring} occhio.model_grid.ModelGrid.__getitem__
:parser: _ext.google_docstring_parser
```

````

````{py:method} train_saes(saes: dict[str, sae_lens.TrainingSAE], training_samples: int = 10000000, batch_size: int = 1024, lr: float = 0.0003, lr_warm_up_steps: int = 0, lr_decay_steps: int = 0, n_snapshots: int = 0, snapshot_fn: typing.Callable[[typing.Any], None] | None = None, autocast_sae: bool = False, autocast_data: bool = False, verbose: bool = False) -> None
:canonical: occhio.model_grid.ModelGrid.train_saes

```{autodoc2-docstring} occhio.model_grid.ModelGrid.train_saes
:parser: _ext.google_docstring_parser
```

````

````{py:method} evaluate_saes(labels: list[str] | None = None, num_samples: int = 100000, verbose: bool = False) -> None
:canonical: occhio.model_grid.ModelGrid.evaluate_saes

```{autodoc2-docstring} occhio.model_grid.ModelGrid.evaluate_saes
:parser: _ext.google_docstring_parser
```

````

`````
