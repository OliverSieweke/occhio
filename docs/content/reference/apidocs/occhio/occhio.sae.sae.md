# {py:mod}`occhio.sae.sae`

```{py:module} occhio.sae.sae
```

```{autodoc2-docstring} occhio.sae.sae
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparseAutoEncoderBase <occhio.sae.sae.SparseAutoEncoderBase>`
  -
* - {py:obj}`SAESimple <occhio.sae.sae.SAESimple>`
  -
* - {py:obj}`TopKIgnoreSAE <occhio.sae.sae.TopKIgnoreSAE>`
  - ```{autodoc2-docstring} occhio.sae.sae.TopKIgnoreSAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SimplexSAE <occhio.sae.sae.SimplexSAE>`
  - ```{autodoc2-docstring} occhio.sae.sae.SimplexSAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`CausalSAE <occhio.sae.sae.CausalSAE>`
  -
````

### API

`````{py:class} SparseAutoEncoderBase(l1_coef: float = 0.5, device: torch.device | str = 'cpu', generator: torch.Generator | None = None)
:canonical: occhio.sae.sae.SparseAutoEncoderBase

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SparseAutoEncoderBase.encode
:abstractmethod:

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.encode
:parser: _ext.google_docstring_parser
```

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SparseAutoEncoderBase.decode
:abstractmethod:

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.decode
:parser: _ext.google_docstring_parser
```

````

````{py:method} resample_weights()
:canonical: occhio.sae.sae.SparseAutoEncoderBase.resample_weights
:abstractmethod:

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.resample_weights
:parser: _ext.google_docstring_parser
```

````

````{py:method} constrain_weights() -> None
:canonical: occhio.sae.sae.SparseAutoEncoderBase.constrain_weights

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.constrain_weights
:parser: _ext.google_docstring_parser
```

````

````{py:method} forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: occhio.sae.sae.SparseAutoEncoderBase.forward

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.forward
:parser: _ext.google_docstring_parser
```

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, intermediate: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SparseAutoEncoderBase.loss

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} train_sae(data_fn, n_steps: int = 10000, batch_size: int = 1024, lr: float = 0.0003, sample_every: int = 25) -> list[float]
:canonical: occhio.sae.sae.SparseAutoEncoderBase.train_sae

```{autodoc2-docstring} occhio.sae.sae.SparseAutoEncoderBase.train_sae
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} SAESimple(n_latent: int, n_dict: int, l1_coef: float = 0.1, dec_bias: bool = False, **kwargs)
:canonical: occhio.sae.sae.SAESimple

Bases: {py:obj}`occhio.sae.sae.SparseAutoEncoderBase`

````{py:method} resample_weights()
:canonical: occhio.sae.sae.SAESimple.resample_weights

````

````{py:method} constrain_weights() -> None
:canonical: occhio.sae.sae.SAESimple.constrain_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SAESimple.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SAESimple.decode

````

`````

`````{py:class} TopKIgnoreSAE(n_latent: int, n_dict: int, l1_coef: float = 0.1, k: int = 2, **kwargs)
:canonical: occhio.sae.sae.TopKIgnoreSAE

Bases: {py:obj}`occhio.sae.sae.SparseAutoEncoderBase`

```{autodoc2-docstring} occhio.sae.sae.TopKIgnoreSAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.sae.sae.TopKIgnoreSAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} resample_weights()
:canonical: occhio.sae.sae.TopKIgnoreSAE.resample_weights

````

````{py:method} constrain_weights() -> None
:canonical: occhio.sae.sae.TopKIgnoreSAE.constrain_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.TopKIgnoreSAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.TopKIgnoreSAE.decode

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, intermediate: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.TopKIgnoreSAE.loss

````

`````

`````{py:class} SimplexSAE(d_input: int, n_simplexes: int, d_local: int | list[int], lambda_1: float = 0.001, lambda_2: float = 0.001, **kwargs)
:canonical: occhio.sae.sae.SimplexSAE

Bases: {py:obj}`occhio.sae.sae.SparseAutoEncoderBase`

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _init_parameters() -> None
:canonical: occhio.sae.sae.SimplexSAE._init_parameters

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE._init_parameters
:parser: _ext.google_docstring_parser
```

````

````{py:method} resample_weights() -> None
:canonical: occhio.sae.sae.SimplexSAE.resample_weights

````

````{py:method} constrain_weights() -> None
:canonical: occhio.sae.sae.SimplexSAE.constrain_weights

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.constrain_weights
:parser: _ext.google_docstring_parser
```

````

````{py:method} initialize_from_data(data: torch.Tensor) -> None
:canonical: occhio.sae.sae.SimplexSAE.initialize_from_data

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.initialize_from_data
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_gates(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SimplexSAE.get_gates

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.get_gates
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_local_latents(x: torch.Tensor) -> list[torch.Tensor]
:canonical: occhio.sae.sae.SimplexSAE.get_local_latents

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.get_local_latents
:parser: _ext.google_docstring_parser
```

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SimplexSAE.encode

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.encode
:parser: _ext.google_docstring_parser
```

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SimplexSAE.decode

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.decode
:parser: _ext.google_docstring_parser
```

````

````{py:method} forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: occhio.sae.sae.SimplexSAE.forward

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.forward
:parser: _ext.google_docstring_parser
```

````

````{py:method} compute_loss(x: torch.Tensor, x_hat: torch.Tensor, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SimplexSAE.compute_loss

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.compute_loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, intermediate: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.SimplexSAE.loss

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} train_sae(data_fn, n_steps: int = 10000, batch_size: int = 1024, lr: float = 0.0003, sample_every: int = 25) -> list[float]
:canonical: occhio.sae.sae.SimplexSAE.train_sae

```{autodoc2-docstring} occhio.sae.sae.SimplexSAE.train_sae
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} CausalSAE(n_latent: int, n_dict: int, l1_coef: float = 0.1, l1_dirc: float = 0.01, l1_causal: float = 0.0, **kwargs)
:canonical: occhio.sae.sae.CausalSAE

Bases: {py:obj}`occhio.sae.sae.SparseAutoEncoderBase`

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.CausalSAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.CausalSAE.decode

````

````{py:method} resample_weights()
:canonical: occhio.sae.sae.CausalSAE.resample_weights

````

````{py:method} constrain_weights() -> None
:canonical: occhio.sae.sae.CausalSAE.constrain_weights

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, intermediate: torch.Tensor) -> torch.Tensor
:canonical: occhio.sae.sae.CausalSAE.loss

````

`````
