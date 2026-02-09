"""
Train on each p_active, evaluate loss on all p_actives, and show as a heatmap.
"""

from occhio.distributions.correlated import HierarchicalPairs
from occhio.distributions.sparse import SparseUniform
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import matplotlib.pyplot as plt
import numpy as np

n_features = 6
n_hidden = 2
importances = torch.tensor([1.0**i for i in range(n_features)])
gen = torch.Generator("cpu")

p_actives = torch.linspace(0.01, 0.99, 32)
n_p = len(p_actives)

loss_matrix = np.zeros((n_p, n_p))

for i, p_train in enumerate(p_actives):
    gen.manual_seed(7)
    # dist = HierarchicalPairs(n_features, p_train, p_follow=0.9, generator=gen)
    dist = SparseUniform(n_features, p_train, generator=gen)
    ae = TiedLinearRelu(n_features, n_hidden, generator=gen)
    tm = ToyModel(dist, ae, importances=importances)
    tm.fit(16_000, verbose=False)

    # Evaluate on all p_actives
    with torch.no_grad():
        for j, p_test in enumerate(p_actives):
            tm.distribution.p_active = p_test  # ty:ignore
            x = tm.distribution.sample(8192)  # ty:ignore
            x_hat, _ = tm.ae(x)  # ty:ignore
            loss_matrix[i, j] = tm.ae.loss(x, x_hat, importances).item()  # ty:ignore

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(loss_matrix, origin="lower")
ax.set_xticks(range(n_p), labels=[str(p.item())[:4] for p in p_actives])
ax.set_yticks(range(n_p), labels=[str(p.item())[:4] for p in p_actives])
ax.set_xlabel("Test p_active")
ax.set_ylabel("Train p_active")
ax.set_title("Loss: train vs test p_active (Sparse Uniform)")
fig.colorbar(im, ax=ax, label="Loss")
plt.tight_layout()
plt.savefig("density_sweep.png", dpi=150)
plt.show()
