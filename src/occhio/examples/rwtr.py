"""
I investigate the Random Walk to Root distribution,
to see if it exhibits ocllapse.
I also look at the AE representation of these distributions.
"""

# %%
from occhio.distributions.dag import DAGRandomWalkToRoot
from occhio.sae import SAESimple, TopKIgnoreSAE
from occhio.autoencoder import MLPEncoder
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
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    ),
]


# %%
gen = torch.Generator()
gen.manual_seed(5)

dist = DAGRandomWalkToRoot(2, 0.5, 1.0, generator=gen)

# We force DAG layout
dist.adjacency = torch.Tensor(things[0])
dist._build_parent_cache()


ae = MLPEncoder([2, 1], [1, 4, 2], generator=gen)
tm = ToyModel(dist, ae, generator=gen)
losses = tm.fit(15_000, verbose=True)
# %%
px.line(losses)

# %%
exsample = dist.sample(5)
exsample

# %%
print(tm.encode(exsample))
print(tm.decode(tm.encode(exsample)))
# print(tm.ae.encoder_weights[0])


# %%
gen = torch.Generator()
gen.manual_seed(6)
sae = TopKIgnoreSAE(1, 3, 0.01, k=1, generator=gen)
# sae = SAESimple(1, 3, 0.05, generator = gen)

losses = sae.train_sae(tm.sample_latent, 15_000)

px.line(losses)

# %%
print(sae.W_enc, sae.b_enc)
print(sae.W_dec)

# %%
# Feature absorption!!
exsample = dist.sample(5)
print(exsample)
print(tm.encode(exsample))
print(sae.encode(tm.encode(exsample)))
print(sae.decode(sae.encode(tm.encode(exsample))))
print(tm.decode(sae.decode(sae.encode(tm.encode(exsample)))))

# %%
# Input path: (1,1) -> (0,0) -> (1,0)
n_points = 10
t = torch.linspace(0.0, 1.0, n_points)

seg1 = torch.stack([1 - t, 1 - t], dim=1)  # (1,1) -> (0,0)
seg2 = torch.stack([t, torch.zeros(n_points)], dim=1)  # (0,0) -> (1,0)

inputs = torch.cat([seg1, seg2], dim=0)
path_pos = np.arange(len(inputs))

with torch.no_grad():
    tm_encoded = tm.encode(inputs).numpy()
    sae_encoded = sae.encode(torch.from_numpy(tm_encoded)).numpy()

# %%
fig0 = px.scatter(
    x=inputs[:, 0].numpy(),
    y=inputs[:, 1].numpy(),
    color=path_pos,
    color_continuous_scale="rainbow",
    labels={"x": "Feature 0", "y": "Feature 1", "color": "Position"},
    title="Input path (1,1)→(0,0)→(1,0)",
)
fig0.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(color="black", size=10),
        name="(0,0)",
    )
)
fig0

# %%
fig1 = px.scatter(
    x=path_pos,
    y=tm_encoded.flatten(),
    color=path_pos,
    color_continuous_scale="rainbow",
    labels={"x": "Path position", "y": "tm.encode", "color": "Position"},
    title="ToyModel encoding along (1,1)→(0,0)→(1,0)",
)
fig1.add_trace(
    go.Scatter(
        x=[path_pos[n_points - 1]],
        y=[tm_encoded.flatten()[n_points - 1]],
        mode="markers",
        marker=dict(color="black", size=10),
        name="(0,0)",
    )
)
fig1

# %%
fig2 = px.scatter_3d(
    x=sae_encoded[:, 0],
    y=sae_encoded[:, 1],
    z=sae_encoded[:, 2],
    color=path_pos,
    color_continuous_scale="rainbow",
    labels={"x": "Dict 0", "y": "Dict 1", "z": "Dict 2", "color": "Position"},
    title="SAE encoding along (1,1)→(0,0)→(1,0)",
)
i = n_points - 1
fig2.add_trace(
    go.Scatter3d(
        x=[sae_encoded[i, 0]],
        y=[sae_encoded[i, 1]],
        z=[sae_encoded[i, 2]],
        mode="markers",
        marker=dict(color="black", size=5),
        name="(0,0)",
    )
)
fig2

# %%
