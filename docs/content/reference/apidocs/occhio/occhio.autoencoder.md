# {py:mod}`occhio.autoencoder`

```{py:module} occhio.autoencoder
```

```{autodoc2-docstring} occhio.autoencoder
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoEncoderBase <occhio.autoencoder.AutoEncoderBase>`
  -
* - {py:obj}`TiedLinear <occhio.autoencoder.TiedLinear>`
  -
* - {py:obj}`TiedLinearRelu <occhio.autoencoder.TiedLinearRelu>`
  -
* - {py:obj}`MLPEncoder <occhio.autoencoder.MLPEncoder>`
  -
* - {py:obj}`TiedMLPEncoder <occhio.autoencoder.TiedMLPEncoder>`
  - ```{autodoc2-docstring} occhio.autoencoder.TiedMLPEncoder
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`ComputeAutoEncoder <occhio.autoencoder.ComputeAutoEncoder>`
  - ```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`AttnLinearAE <occhio.autoencoder.AttnLinearAE>`
  - ```{autodoc2-docstring} occhio.autoencoder.AttnLinearAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`AttnAttnAE <occhio.autoencoder.AttnAttnAE>`
  - ```{autodoc2-docstring} occhio.autoencoder.AttnAttnAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`LinearAttnAE <occhio.autoencoder.LinearAttnAE>`
  - ```{autodoc2-docstring} occhio.autoencoder.LinearAttnAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`SynthAE <occhio.autoencoder.SynthAE>`
  - ```{autodoc2-docstring} occhio.autoencoder.SynthAE
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`softmax1 <occhio.autoencoder.softmax1>`
  - ```{autodoc2-docstring} occhio.autoencoder.softmax1
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SKIP <occhio.autoencoder._SKIP>`
  - ```{autodoc2-docstring} occhio.autoencoder._SKIP
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

````{py:data} _SKIP
:canonical: occhio.autoencoder._SKIP
:value: >
   'object(...)'

```{autodoc2-docstring} occhio.autoencoder._SKIP
:parser: _ext.google_docstring_parser
```

````

````{py:function} softmax1(x, dim=-1)
:canonical: occhio.autoencoder.softmax1

```{autodoc2-docstring} occhio.autoencoder.softmax1
:parser: _ext.google_docstring_parser
```
````

`````{py:class} AutoEncoderBase(n_features: int, n_hidden: int, loss_fn: typing.Callable | None = None, device: torch.device | str | None = None, generator: torch.Generator | None = None)
:canonical: occhio.autoencoder.AutoEncoderBase

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AutoEncoderBase.encode
:abstractmethod:

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.encode
:parser: _ext.google_docstring_parser
```

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AutoEncoderBase.decode
:abstractmethod:

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.decode
:parser: _ext.google_docstring_parser
```

````

````{py:property} feature_vectors
:canonical: occhio.autoencoder.AutoEncoderBase.feature_vectors
:type: torch.Tensor

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.feature_vectors
:parser: _ext.google_docstring_parser
```

````

````{py:method} resample_weights()
:canonical: occhio.autoencoder.AutoEncoderBase.resample_weights
:abstractmethod:

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.resample_weights
:parser: _ext.google_docstring_parser
```

````

````{py:method} forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: occhio.autoencoder.AutoEncoderBase.forward

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.forward
:parser: _ext.google_docstring_parser
```

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, importances: torch.Tensor | None)
:canonical: occhio.autoencoder.AutoEncoderBase.loss

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.loss
:parser: _ext.google_docstring_parser
```

````

````{py:property} device
:canonical: occhio.autoencoder.AutoEncoderBase.device
:type: torch.device | None

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.device
:parser: _ext.google_docstring_parser
```

````

````{py:method} save_weights(path: str | pathlib.Path | None = None) -> pathlib.Path
:canonical: occhio.autoencoder.AutoEncoderBase.save_weights

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.save_weights
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} _NN_MODULE_INTERNALS
:canonical: occhio.autoencoder.AutoEncoderBase._NN_MODULE_INTERNALS
:value: >
   'frozenset(...)'

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase._NN_MODULE_INTERNALS
:parser: _ext.google_docstring_parser
```

````

````{py:method} _collect_attrs() -> dict
:canonical: occhio.autoencoder.AutoEncoderBase._collect_attrs

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase._collect_attrs
:parser: _ext.google_docstring_parser
```

````

````{py:method} _serialize_value(v)
:canonical: occhio.autoencoder.AutoEncoderBase._serialize_value
:staticmethod:

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase._serialize_value
:parser: _ext.google_docstring_parser
```

````

````{py:method} load_weights(path: str | pathlib.Path, *, strict: bool = True) -> None
:canonical: occhio.autoencoder.AutoEncoderBase.load_weights

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.load_weights
:parser: _ext.google_docstring_parser
```

````

````{py:method} __init_subclass__(**kwargs)
:canonical: occhio.autoencoder.AutoEncoderBase.__init_subclass__
:classmethod:

```{autodoc2-docstring} occhio.autoencoder.AutoEncoderBase.__init_subclass__
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} TiedLinear(n_features: int, n_hidden: int, **kwargs)
:canonical: occhio.autoencoder.TiedLinear

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.TiedLinear.resample_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedLinear.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedLinear.decode

````

`````

`````{py:class} TiedLinearRelu(n_features: int, n_hidden: int, **kwargs)
:canonical: occhio.autoencoder.TiedLinearRelu

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.TiedLinearRelu.resample_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedLinearRelu.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedLinearRelu.decode

````

`````

`````{py:class} MLPEncoder(embedding: list[int], unembedding: list[int], tied_initialization: bool = False, **kwargs)
:canonical: occhio.autoencoder.MLPEncoder

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

````{py:method} _build_layers()
:canonical: occhio.autoencoder.MLPEncoder._build_layers

```{autodoc2-docstring} occhio.autoencoder.MLPEncoder._build_layers
:parser: _ext.google_docstring_parser
```

````

````{py:method} _init_param(w: torch.nn.Parameter, b: torch.nn.Parameter)
:canonical: occhio.autoencoder.MLPEncoder._init_param

```{autodoc2-docstring} occhio.autoencoder.MLPEncoder._init_param
:parser: _ext.google_docstring_parser
```

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.MLPEncoder.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.MLPEncoder.decode

````

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.MLPEncoder.resample_weights

````

`````

`````{py:class} TiedMLPEncoder(dims: list[int], **kwargs)
:canonical: occhio.autoencoder.TiedMLPEncoder

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.TiedMLPEncoder
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.TiedMLPEncoder.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} _build_layers()
:canonical: occhio.autoencoder.TiedMLPEncoder._build_layers

```{autodoc2-docstring} occhio.autoencoder.TiedMLPEncoder._build_layers
:parser: _ext.google_docstring_parser
```

````

````{py:method} _init_param(w: torch.nn.Parameter, b: torch.nn.Parameter)
:canonical: occhio.autoencoder.TiedMLPEncoder._init_param

```{autodoc2-docstring} occhio.autoencoder.TiedMLPEncoder._init_param
:parser: _ext.google_docstring_parser
```

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedMLPEncoder.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.TiedMLPEncoder.decode

````

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.TiedMLPEncoder.resample_weights

````

`````

`````{py:class} ComputeAutoEncoder(N: int, k: int, decode_activation: typing.Literal[softmax, relu] = 'softmax', seed: int = 10, **kwargs)
:canonical: occhio.autoencoder.ComputeAutoEncoder

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.encode

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.encode
:parser: _ext.google_docstring_parser
```

````

````{py:method} compute_step(h: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.compute_step

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.compute_step
:parser: _ext.google_docstring_parser
```

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.decode

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.decode
:parser: _ext.google_docstring_parser
```

````

````{py:method} forward(x: torch.Tensor)
:canonical: occhio.autoencoder.ComputeAutoEncoder.forward

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.forward
:parser: _ext.google_docstring_parser
```

````

````{py:method} ce_loss(y_hat: torch.Tensor, y_idx: torch.Tensor, importances: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.ce_loss

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.ce_loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor, importances: torch.Tensor | None) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.mse_loss

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.mse_loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} loss(x_true: torch.Tensor, x_hat: torch.Tensor, importances: torch.Tensor | None) -> torch.Tensor
:canonical: occhio.autoencoder.ComputeAutoEncoder.loss

```{autodoc2-docstring} occhio.autoencoder.ComputeAutoEncoder.loss
:parser: _ext.google_docstring_parser
```

````

````{py:method} resample_weights()
:canonical: occhio.autoencoder.ComputeAutoEncoder.resample_weights

````

`````

`````{py:class} AttnLinearAE(n_features: int, n_hidden: int, n_heads: int, dict_size: int, **kwargs)
:canonical: occhio.autoencoder.AttnLinearAE

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.AttnLinearAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.AttnLinearAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.AttnLinearAE.resample_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AttnLinearAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AttnLinearAE.decode

````

`````

`````{py:class} AttnAttnAE(n_features: int, n_hidden: int, n_heads: int, dict_size: int, **kwargs)
:canonical: occhio.autoencoder.AttnAttnAE

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.AttnAttnAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.AttnAttnAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.AttnAttnAE.resample_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AttnAttnAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.AttnAttnAE.decode

````

`````

`````{py:class} LinearAttnAE(n_features: int, n_hidden: int, n_heads: int, dict_size: int, **kwargs)
:canonical: occhio.autoencoder.LinearAttnAE

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.LinearAttnAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.LinearAttnAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.LinearAttnAE.resample_weights

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.LinearAttnAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.LinearAttnAE.decode

````

`````

`````{py:class} SynthAE(n_features: int, n_hidden: int, orthogonalize: bool = False, ortho_lambda: float = 1.0, ortho_steps: int = 1000, ortho_lr: float = 0.01, ortho_chunk_size: int = 1024, **kwargs)
:canonical: occhio.autoencoder.SynthAE

Bases: {py:obj}`occhio.autoencoder.AutoEncoderBase`

```{autodoc2-docstring} occhio.autoencoder.SynthAE
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.autoencoder.SynthAE.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} resample_weights(force_norm=False)
:canonical: occhio.autoencoder.SynthAE.resample_weights

````

````{py:method} _run_orthogonalization(W: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.SynthAE._run_orthogonalization

```{autodoc2-docstring} occhio.autoencoder.SynthAE._run_orthogonalization
:parser: _ext.google_docstring_parser
```

````

````{py:method} freeze_W()
:canonical: occhio.autoencoder.SynthAE.freeze_W

```{autodoc2-docstring} occhio.autoencoder.SynthAE.freeze_W
:parser: _ext.google_docstring_parser
```

````

````{py:property} rho_mm
:canonical: occhio.autoencoder.SynthAE.rho_mm
:type: float

```{autodoc2-docstring} occhio.autoencoder.SynthAE.rho_mm
:parser: _ext.google_docstring_parser
```

````

````{py:method} encode(x: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.SynthAE.encode

````

````{py:method} decode(z: torch.Tensor) -> torch.Tensor
:canonical: occhio.autoencoder.SynthAE.decode

````

`````
