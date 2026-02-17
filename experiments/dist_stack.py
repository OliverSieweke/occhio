"""Example with Distribution Stack. See nice hierarchical structure in embedding matrix."""

# %%
from occhio.distributions import SparseUniform, DistributionStack
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
from occhio.visualization import plot_dynamic_scatter

# %%
DEVICE = "cpu"
gen = torch.Generator("cpu")
gen.manual_seed(2)


def my_hook(hook_data):
    return hook_data["epoch"], hook_data["tm"].ae.W.detach().numpy().copy()


# # Another hook here!
# def my_hook(hook_data):
#     return hook_data["epoch"], torch.stack(
#         [hook_data["tm"].feature_dimensionalities, hook_data["tm"].total_feature_interferences]
#     )


dist = DistributionStack(
    [SparseUniform(3, 0.5, generator=gen, device=DEVICE) for i in range(3)],
    "single",
    device=DEVICE,
)


n_hidden = 2
importances = torch.tensor([0.95**i for i in range(dist.n_features)])

# %%
ae = TiedLinearRelu(dist.n_features, n_hidden, generator=gen)
tm = ToyModel(dist, ae, importances=importances, generator=gen, device=DEVICE)
losses, hook_returns = tm.fit(
    40_000, batch_size=256, verbose=False, hooks=[my_hook], hook_freq=1000
)


# %%
# Interactive version with slider to explore embeddings at different epochs
plot_dynamic_scatter(losses, hook_returns[0])

# %%
