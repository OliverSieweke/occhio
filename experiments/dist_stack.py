"""Example with Distribution Stack. See nice hierarchical structure in embedding matrix."""

# %%
from occhio.distributions import SparseUniform, DistributionStack, CorrelatedPairs
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import plotly.express as px
from occhio.visualization import plot_dynamic_scatter

# %%
gen = torch.Generator("cpu")
gen.manual_seed(2)


def my_hook(hook_data):
    return hook_data["epoch"], hook_data["tm"].ae.W.detach().numpy().copy()


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
importances = torch.tensor([0.95**i for i in range(dist.n_features)])

# %%
ae = TiedLinearRelu(dist.n_features, n_hidden, generator=gen)
tm = ToyModel(dist, ae, importances=importances)
losses, hook_returns = tm.fit(40_000, verbose=False, hooks=[my_hook], hook_freq=500)

# %%
px.line(losses)

# %%
# Interactive version with slider to explore embeddings at different epochs
plot_dynamic_scatter(losses, hook_returns[0])
# %%
