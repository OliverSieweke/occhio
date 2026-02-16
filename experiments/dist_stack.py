"""Example with Distribution Stack. See nice hierarchical structure in embedding matrix."""

# %%
from occhio.distributions import SparseUniform, DistributionStack, CorrelatedPairs
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import plotly.express as px

# %%
gen = torch.Generator("cpu")
gen.manual_seed(2)


dist = DistributionStack(
    [SparseUniform(3, 0.5, generator=gen) for i in range(3)], "single"
)
# dist = DistributionStack(
#     [
#         CorrelatedPairs(4, p_active=0.5, p_individual=0.7, generator=gen)
#         for i in range(2)
#     ],
#     "single",
# )


n_hidden = 2
importances = torch.tensor([0.99**i for i in range(dist.n_features)])

# %%
ae = TiedLinearRelu(dist.n_features, n_hidden, generator=gen)
tm = ToyModel(dist, ae, importances=importances)
losses = tm.fit(40_000, verbose=False)

# %%
px.line(losses)

# %%
emb_mat = ae.W.detach().numpy()
print(emb_mat.shape)

fig = px.scatter(x=emb_mat[0], y=emb_mat[1], color=list(range(dist.n_features)))
fig.update_traces(marker={"size": 15})
# %%
