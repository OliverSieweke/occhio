# Autoencoders

Autoencoders define the bottleneck architecture that learns to reconstruct
features from a lower-dimensional hidden space. All autoencoders inherit from
{class}`~occhio.autoencoders.AutoEncoderBase`.

## Architectures

| Class | Description |
|---|---|
| {class}`~occhio.autoencoders.TiedLinearRelu` | Tied-weight linear encoder + ReLU decoder. The standard architecture from Elhage et al. (2022). |
| {class}`~occhio.autoencoders.TiedLinear` | Tied-weight linear encoder + linear decoder (no activation). |
| {class}`~occhio.autoencoders.MLPEncoder` | Multi-layer perceptron encoder/decoder with configurable layer sizes. |
| {class}`~occhio.autoencoders.TiedMLPEncoder` | MLP with tied initialization between encoder and decoder layers. |
| {class}`~occhio.autoencoders.AttnLinearAE` | Multi-head softmax bottleneck encoder + linear decoder. |
| {class}`~occhio.autoencoders.AttnAttnAE` | Attention-based encoder and decoder. |
| {class}`~occhio.autoencoders.LinearAttnAE` | Linear encoder + attention-based decoder. |
| {class}`~occhio.autoencoders.ComputeAutoEncoder` | Tied encoder/decoder with a linear compute step in the hidden space. |
| {class}`~occhio.autoencoders.SynthAE` | Unit-norm tied weights with optional orthogonalization (synthetic ground truth). |

## Choosing an Architecture

**Default**: Use {class}`~occhio.autoencoders.TiedLinearRelu` for standard
superposition experiments. It matches the original paper's setup and is the
most well-studied.

**MLP variants**: Use {class}`~occhio.autoencoders.MLPEncoder` or
{class}`~occhio.autoencoders.TiedMLPEncoder` when you need nonlinear encoding
with biases (e.g., to avoid the identity-collapse issue with sparse inputs).

**Attention variants**: Use {class}`~occhio.autoencoders.AttnLinearAE` and
related classes for multi-head softmax bottleneck experiments
(Minkowski-style tile representations).

**Compute**: Use {class}`~occhio.autoencoders.ComputeAutoEncoder` to study
computation in the hidden space (linear transformation between encode and
decode).

**Synthetic**: Use {class}`~occhio.autoencoders.SynthAE` to construct
autoencoders with known ground-truth feature geometry (unit-norm, optionally
orthogonal weights).

## Usage

```python
from occhio.autoencoders import TiedLinearRelu

# Basic construction
ae = TiedLinearRelu(n_features=10, n_hidden=5)

# With specific device and generator
gen = torch.Generator(device="cpu").manual_seed(42)
ae = TiedLinearRelu(n_features=10, n_hidden=5, device="cpu", generator=gen)

# Encode / decode
z = ae.encode(x)        # (batch, n_features) -> (batch, n_hidden)
x_hat = ae.decode(z)     # (batch, n_hidden) -> (batch, n_features)
x_hat, z = ae(x)         # forward pass returns both
```

## Save, Load, and Hub

All autoencoders support persistence via safetensors:

```python
# Save to disk
path = ae.save_weights("my_model")  # creates .safetensors + .json

# Load from disk (auto-detects architecture from metadata)
ae = AutoEncoderBase.from_local("my_model.safetensors")

# Load into an existing instance (validates class match)
ae.load_weights("my_model.safetensors")

# HuggingFace Hub
ae.push_to_hub("username/my-model")
ae = AutoEncoderBase.from_hub("username/my-model")
```

The `.json` companion file is human-readable metadata (class name, config,
parameter shapes). It is not required for loading.

## Custom Autoencoders

To create a custom autoencoder, subclass {class}`~occhio.autoencoders.AutoEncoderBase`
and implement four methods:

```python
from occhio.autoencoders import AutoEncoderBase

class MyAE(AutoEncoderBase):
    def __init__(self, n_features, n_hidden, my_param=1.0, **kwargs):
        super().__init__(n_features, n_hidden, **kwargs)
        self.my_param = my_param
        # ... define parameters ...

    def encode(self, x):
        """Map from feature space to hidden space."""
        ...

    def decode(self, z):
        """Map from hidden space back to feature space."""
        ...

    def resample_weights(self):
        """Reinitialize all learnable parameters."""
        ...
```

The base class handles:

- Device management (`self.device`, `self._init_device`)
- Generator-aware random number generation (`self.generator`)
- `get_config()` for serialization (inspects `__init__` signature automatically)
- `save_weights()` / `from_local()` / `from_hub()` / `push_to_hub()`
- Auto-registration in the class registry for deserialization
- Default `loss()` (importance-weighted MSE) and `forward()` (encode + decode)

Override `get_config()` if your constructor parameter names do not match
instance attribute names.

## API Reference

See the full {doc}`api` for detailed class documentation.
