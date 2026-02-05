"""Sweep over p_active values and visualize W embeddings"""

from occhio.distributions.sparse import SparseUniform
from occhio.autoencoder import AutoEncoder
from occhio.toy_model import ToyModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

n_features = 5
n_hidden = 2
importances = torch.tensor([0.8**i for i in range(n_features)])

# Sparsity values to sweep
p_actives = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# Color palette for features
colors = sns.color_palette("viridis", n_features)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, p_active in enumerate(p_actives):
    dist = SparseUniform(n_features, p_active)
    ae = AutoEncoder(n_features, n_hidden)

    tm = ToyModel(dist, ae, importances=importances)
    tm.fit(10_000, verbose=False)

    W: np.ndarray = tm.ae.W.detach().numpy()

    ax = axes[idx]

    for i in range(n_features):
        ax.scatter(W[0, i], W[1, i], c=[colors[i]], s=100)
        ax.annotate(f"{i}", (W[0, i], W[1, i]), fontsize=9, ha="center", va="bottom")

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=0.5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"p_active = {p_active}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

plt.suptitle(
    "Feature embeddings W as sparsity varies\n(colors = importance rank, 0=highest)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("sparsity_sweep.png", dpi=150)
plt.show()
