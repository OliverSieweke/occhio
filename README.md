> [!CAUTION]
>
> Work in Progress

# occhio
_/ˈɔk.kjo/ — like Tokyo_

A toy library for playing with [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html).

Named for the Italian word for "eye" — because we're trying to see what's hiding in those hidden dimensions — and for a certain wooden boy who wanted to be real, because these are, after all, toy models dreaming of becoming interpretable. 


## Actual Docs

The main idea of this library is to help researchers speed up their implementation of various Toy Models of superposition. To that end we have implemented a `ToyModel` class which takes in a feature distribution and some type of AutoEncoder. 

```python
tm = ToyModel(
    distribution = SparseUniform(5, p_active=0.2),
    ae = TiedLinear(5, 2),
    importances = torch.tensor([0.8**i for i in range(n_features)])
)
```

A reasonable training loop has already been implemented and can be performed easily by simply running `tm.fit(n_epochs=1000)`. Once trained, the latent space can be sampled either by `tm.get_one_hot_embeddings()`, which simply returns the embeddings of the one hot vectors of feature space (this corresponds exactly to `W` for the `TiedLinear` Autoencoder), or via `tm.sample_latent(batch_size)`, which samples the feature distribution and passes it through the encoder.

The abstract `AutoEncoderBase`, and `Distribution` classes can be easily extended and ensures compatibility with the `ToyModel` class. 


## Future directions

- [ ] Plotting
- [ ] SAE's
- [ ] Benchmarking tools
- [ ] Easy interference calculations