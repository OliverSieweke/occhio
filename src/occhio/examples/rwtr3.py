"""
I investigate the Random Walk to Root distribution,
to see if it exhibits ocllapse.
I also look at the AE representation of these distributions.
"""

# %%
from occhio.distributions.dag import DAGRandomWalkToRoot
from occhio.sae import SAESimple, TopKIgnoreSAE
from occhio.autoencoder import MLPEncoder, TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# %%
things: list[np.ndarray] = [
    np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.bool,
    ),
    np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    ),
    np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ),
    np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    ),
    np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    ),
]


# %%
torch.set_printoptions(3, sci_mode=False)
gen = torch.Generator()
gen.manual_seed(1)

p_active = [10.0, 1.0, 1.0]

dist = DAGRandomWalkToRoot(3, 0.5, 1.0, p_active, generator=gen)

# We force DAG layout
dist.adjacency = torch.Tensor(things[2])
dist._build_parent_cache()

# Validate p_active
torch.mean(1.0*(dist.sample(1000)>0.0), dim=0)


#%%
# ae = MLPEncoder([3, 2], [2, 6, 3], generator=gen)
ae = TiedLinearRelu(3, 2, generator=gen)
tm = ToyModel(dist, ae, generator=gen)
losses = tm.fit(20_000, verbose=True)

px.line(losses)


# %%
exsample = dist.sample(5)
print(exsample)

print(tm.encode(exsample))
print(tm.decode(tm.encode(exsample)))
# print(tm.ae.encoder_weights[0])


# %%
gen = torch.Generator()
gen.manual_seed(2)

# This can only learn if k=2!
# sae = TopKIgnoreSAE(2, 5, 0.01, k=2, generator=gen)
sae = SAESimple(2, 5, 0.02, generator=gen)

losses = sae.train_sae(tm.sample_latent, 20_000)

px.line(losses)


# %%
print(sae.W_enc, sae.b_enc)
print(sae.W_dec)

# %%
exsample = dist.sample(5)
print(exsample)
# print(tm.encode(exsample))
print(sae.encode(tm.encode(exsample)))
# print(sae.decode(sae.encode(tm.encode(exsample))))
print(tm.decode(sae.decode(sae.encode(tm.encode(exsample)))))

# %%
samples = dist.sample(256)
embedded_samples = tm.encode(samples).detach().numpy().T
em_hat_samples = sae.decode(sae.encode(tm.encode(samples))).detach().numpy().T
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=embedded_samples[0],
        y=embedded_samples[1],
        mode="markers",
        name="Ground Truth",
    )
)
fig.add_trace(
    go.Scatter(x=em_hat_samples[0], y=em_hat_samples[1], mode="markers", name="SAE")
)

# %%
patterns = torch.tensor(
    [[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32
)
px.imshow(patterns, title="Activation Patterns", labels=dict(x="Feature", y="Pattern"))

# %%
encoded_patterns = sae.encode(tm.encode(patterns)).detach().numpy()
px.imshow(
    encoded_patterns, title="Encoded Patterns", labels=dict(x="Latent Dim", y="Pattern")
)

# %%
