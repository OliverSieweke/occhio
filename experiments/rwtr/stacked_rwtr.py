# %%
"""
I investigate the Random Walk to Root distribution,
to see if it exhibits ocllapse.
I also look at the AE representation of these distributions.
"""

# %%
from occhio.distributions import DAGRandomWalkToRoot, DistributionStack
from occhio.sae import SAESimple, TopKIgnoreSAE
from occhio.autoencoder import MLPEncoder, TiedLinearRelu
from occhio.toy_model import ToyModel
from occhio.visualization import plot_embedding
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
gen.manual_seed(2)

N_STACKS = 2

p_active = [1.0, 1.0, 1.0]

dist = DistributionStack(
    [
        DAGRandomWalkToRoot(
            n_features=3, p_active=p_active, adjacency=things[2], generator=gen
        )
        for i in range(N_STACKS)
    ],
    sampling_mode="sparse",
    p_meta=0.3,
)
print(dist.n_features)

# Validate p_active
torch.mean(1.0 * (dist.sample(1000) > 0.0), dim=0)


# %%
ae = TiedLinearRelu(3 * N_STACKS, 2, generator=gen)
tm = ToyModel(dist, ae, generator=gen)
losses = tm.fit(15_000, verbose=True)[0]

# %%
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
# sae = TopKIgnoreSAE(2, 5, 0.01, k=1, generator=gen)
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
samples = dist.sample(512)
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
# fig.add_trace(
#     go.Scatter(x=em_hat_samples[0], y=em_hat_samples[1], mode="markers", name="SAE")
# )


# %%
emb = tm.W.detach().numpy()

px.scatter(x=emb[0], y=emb[1], hover_name=list(range(emb.shape[1])))

# %%
