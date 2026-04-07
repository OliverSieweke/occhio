# {py:mod}`occhio.toy_model`

```{py:module} occhio.toy_model
```

```{autodoc2-docstring} occhio.toy_model
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAERecord <occhio.toy_model.SAERecord>`
  - ```{autodoc2-docstring} occhio.toy_model.SAERecord
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`ToyModel <occhio.toy_model.ToyModel>`
  - ```{autodoc2-docstring} occhio.toy_model.ToyModel
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SAERecord
:canonical: occhio.toy_model.SAERecord

```{autodoc2-docstring} occhio.toy_model.SAERecord
:parser: _ext.google_docstring_parser
```

````{py:attribute} sae
:canonical: occhio.toy_model.SAERecord.sae
:type: sae_lens.TrainingSAE
:value: >
   None

```{autodoc2-docstring} occhio.toy_model.SAERecord.sae
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} results
:canonical: occhio.toy_model.SAERecord.results
:type: sae_lens.synthetic.SyntheticDataEvalResult | None
:value: >
   None

```{autodoc2-docstring} occhio.toy_model.SAERecord.results
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} ToyModel(distribution: occhio.distributions.Distribution, ae: occhio.autoencoder.AutoEncoderBase, device: torch.device | str | None = None, importances: torch.Tensor | list | None = None)
:canonical: occhio.toy_model.ToyModel

```{autodoc2-docstring} occhio.toy_model.ToyModel
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.toy_model.ToyModel.__init__
:parser: _ext.google_docstring_parser
```

````{py:attribute} distribution
:canonical: occhio.toy_model.ToyModel.distribution
:type: occhio.distributions.Distribution
:value: >
   None

```{autodoc2-docstring} occhio.toy_model.ToyModel.distribution
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} ae
:canonical: occhio.toy_model.ToyModel.ae
:type: occhio.autoencoder.AutoEncoderBase
:value: >
   None

```{autodoc2-docstring} occhio.toy_model.ToyModel.ae
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} saes
:canonical: occhio.toy_model.ToyModel.saes
:type: dict[str, occhio.toy_model.SAERecord]
:value: >
   None

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes
:parser: _ext.google_docstring_parser
```

````

````{py:method} _validate_data_file(tensors: dict[str, torch.Tensor], path: pathlib.Path, n_features: int, batch_size: int) -> None
:canonical: occhio.toy_model.ToyModel._validate_data_file
:staticmethod:

```{autodoc2-docstring} occhio.toy_model.ToyModel._validate_data_file
:parser: _ext.google_docstring_parser
```

````

````{py:method} fit(n_epochs: int, batch_size: int = 1024, learning_rate: float = 0.0003, weight_decay: float = 0.05, track_losses: bool = True, optimizer: torch.optim.Optimizer | None = None, hooks: list[typing.Callable] | None = None, hook_freq: int = 1, verbose: bool = False, sample_every: int = 25, precomputed_data: str | pathlib.Path | None = None) -> tuple[list[float], list]
:canonical: occhio.toy_model.ToyModel.fit

```{autodoc2-docstring} occhio.toy_model.ToyModel.fit
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample_latent(batch_size) -> torch.Tensor
:canonical: occhio.toy_model.ToyModel.sample_latent

```{autodoc2-docstring} occhio.toy_model.ToyModel.sample_latent
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_one_hot_embeddings() -> torch.Tensor
:canonical: occhio.toy_model.ToyModel.get_one_hot_embeddings

```{autodoc2-docstring} occhio.toy_model.ToyModel.get_one_hot_embeddings
:parser: _ext.google_docstring_parser
```

````

````{py:method} __repr__()
:canonical: occhio.toy_model.ToyModel.__repr__

````

````{py:method} __getattr__(name)
:canonical: occhio.toy_model.ToyModel.__getattr__

```{autodoc2-docstring} occhio.toy_model.ToyModel.__getattr__
:parser: _ext.google_docstring_parser
```

````

````{py:method} train_saes(saes: dict[str, sae_lens.TrainingSAE], training_samples: int = 10000000, batch_size: int = 1024, lr: float = 0.0003, lr_warm_up_steps: int = 0, lr_decay_steps: int = 0, n_snapshots: int = 0, snapshot_fn: typing.Callable[[typing.Any], None] | None = None, autocast_sae: bool = False, autocast_data: bool = False, verbose: bool = False) -> None
:canonical: occhio.toy_model.ToyModel.train_saes

```{autodoc2-docstring} occhio.toy_model.ToyModel.train_saes
:parser: _ext.google_docstring_parser
```

````

````{py:method} evaluate_saes(labels: list[str] | None = None, num_samples: int = 100000, verbose: bool = False) -> dict[str, sae_lens.synthetic.SyntheticDataEvalResult]
:canonical: occhio.toy_model.ToyModel.evaluate_saes

```{autodoc2-docstring} occhio.toy_model.ToyModel.evaluate_saes
:parser: _ext.google_docstring_parser
```

````

````{py:property} frobenius_norm_squared
:canonical: occhio.toy_model.ToyModel.frobenius_norm_squared

```{autodoc2-docstring} occhio.toy_model.ToyModel.frobenius_norm_squared
:parser: _ext.google_docstring_parser
```

````

````{py:property} hidden_dimensions_per_embedded_features
:canonical: occhio.toy_model.ToyModel.hidden_dimensions_per_embedded_features
:type: typing.Any

```{autodoc2-docstring} occhio.toy_model.ToyModel.hidden_dimensions_per_embedded_features
:parser: _ext.google_docstring_parser
```

````

````{py:property} embedded_features_per_hidden_dimensions
:canonical: occhio.toy_model.ToyModel.embedded_features_per_hidden_dimensions
:type: typing.Any

```{autodoc2-docstring} occhio.toy_model.ToyModel.embedded_features_per_hidden_dimensions
:parser: _ext.google_docstring_parser
```

````

````{py:property} feature_dimensionalities
:canonical: occhio.toy_model.ToyModel.feature_dimensionalities

```{autodoc2-docstring} occhio.toy_model.ToyModel.feature_dimensionalities
:parser: _ext.google_docstring_parser
```

````

````{py:property} mean_feature_dimensionalities
:canonical: occhio.toy_model.ToyModel.mean_feature_dimensionalities

```{autodoc2-docstring} occhio.toy_model.ToyModel.mean_feature_dimensionalities
:parser: _ext.google_docstring_parser
```

````

````{py:property} total_feature_dimensionalities_per_hidden_dimension
:canonical: occhio.toy_model.ToyModel.total_feature_dimensionalities_per_hidden_dimension

```{autodoc2-docstring} occhio.toy_model.ToyModel.total_feature_dimensionalities_per_hidden_dimension
:parser: _ext.google_docstring_parser
```

````

````{py:property} W
:canonical: occhio.toy_model.ToyModel.W
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.W
:parser: _ext.google_docstring_parser
```

````

````{py:property} W_T_W
:canonical: occhio.toy_model.ToyModel.W_T_W
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.W_T_W
:parser: _ext.google_docstring_parser
```

````

````{py:property} W_normalized_features
:canonical: occhio.toy_model.ToyModel.W_normalized_features
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.W_normalized_features
:parser: _ext.google_docstring_parser
```

````

````{py:property} feature_norms
:canonical: occhio.toy_model.ToyModel.feature_norms
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.feature_norms
:parser: _ext.google_docstring_parser
```

````

````{py:property} feature_representations
:canonical: occhio.toy_model.ToyModel.feature_representations
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.feature_representations
:parser: _ext.google_docstring_parser
```

````

````{py:property} interferences_sq
:canonical: occhio.toy_model.ToyModel.interferences_sq
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.interferences_sq
:parser: _ext.google_docstring_parser
```

````

````{py:property} interferences
:canonical: occhio.toy_model.ToyModel.interferences
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.interferences
:parser: _ext.google_docstring_parser
```

````

````{py:property} total_feature_interferences
:canonical: occhio.toy_model.ToyModel.total_feature_interferences
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.total_feature_interferences
:parser: _ext.google_docstring_parser
```

````

````{py:property} total_feature_interferences_including_self
:canonical: occhio.toy_model.ToyModel.total_feature_interferences_including_self
:type: torch.Tensor

```{autodoc2-docstring} occhio.toy_model.ToyModel.total_feature_interferences_including_self
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_precision
:canonical: occhio.toy_model.ToyModel.saes_precision
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_precision
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_recall
:canonical: occhio.toy_model.ToyModel.saes_recall
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_recall
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_f1_score
:canonical: occhio.toy_model.ToyModel.saes_f1_score
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_f1_score
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_accuracy
:canonical: occhio.toy_model.ToyModel.saes_accuracy
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_accuracy
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_explained_variance
:canonical: occhio.toy_model.ToyModel.saes_explained_variance
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_explained_variance
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_l0
:canonical: occhio.toy_model.ToyModel.saes_l0
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_l0
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_dead_latents
:canonical: occhio.toy_model.ToyModel.saes_dead_latents
:type: dict[str, int]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_dead_latents
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_true_l0
:canonical: occhio.toy_model.ToyModel.saes_true_l0
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_true_l0
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_shrinkage
:canonical: occhio.toy_model.ToyModel.saes_shrinkage
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_shrinkage
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_mcc
:canonical: occhio.toy_model.ToyModel.saes_mcc
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_mcc
:parser: _ext.google_docstring_parser
```

````

````{py:property} saes_uniqueness
:canonical: occhio.toy_model.ToyModel.saes_uniqueness
:type: dict[str, float]

```{autodoc2-docstring} occhio.toy_model.ToyModel.saes_uniqueness
:parser: _ext.google_docstring_parser
```

````

`````
