"""Example with Distribution Stack. See nice hierarchical structure in embedding matrix."""

# %%
from occhio.distributions import SparseUniform, SingleUniform, DistributionStack
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
from occhio.visualization import plot_dynamic_scatter


# %%
# TiedLinearRelu(5, 2, generator=torch.Generator("mps"), device="cpu")
# tm = ToyModel(
#     SparseUniform(5, 0.3, device="mps"),
#     TiedLinearRelu(5, 2),
# )
# print(tm.device)
# tm.fit(100)

# %%
DEVICE = "cpu"
gen = torch.Generator("cpu")
gen.manual_seed(8)


def my_hook(hook_data):
    return hook_data["epoch"], hook_data["tm"].ae.W.detach().numpy().copy()


# Another hook here!
def feat_dim_and_interference(hook_data):
    return hook_data["epoch"], torch.stack(
        [
            hook_data["tm"].feature_dimensionalities,
            hook_data["tm"].total_feature_interferences,
        ]
    )


dist = DistributionStack(
    [
        SparseUniform(3, p_active=0.36, generator=gen, device=DEVICE),
        SparseUniform(3, 0.36, generator=gen, device=DEVICE),
        SparseUniform(3, 0.36, generator=gen, device=DEVICE),
    ],
    "sparse",
    p_meta=0.333,
    generator=gen,
    device=DEVICE,
)

# %%


n_hidden = 2
importances = torch.tensor([0.9**i for i in range(dist.n_features)])

# %%
ae = TiedLinearRelu(dist.n_features, n_hidden, generator=gen)
tm = ToyModel(dist, ae, importances=importances, generator=gen, device=DEVICE)
losses, hook_returns = tm.fit(
    n_epochs=60_000,
    batch_size=128,
    verbose=False,
    hooks=[my_hook, feat_dim_and_interference],
    hook_freq=250,
    learning_rate=3e-4,
    weight_decay=0.05,
)


# %%
plot_dynamic_scatter(losses, hook_returns[0], loss_stride=20)

# %%
plot_dynamic_scatter(losses, hook_returns[1], loss_stride=20)

# %%
