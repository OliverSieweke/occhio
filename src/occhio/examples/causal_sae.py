"""
Experiment: Can we recover causal structure from DAG distributions via SAEs?

Pipeline:
    1. Create DAGBayesianPropagation (ground truth causal graph, Noisy-OR propagation)
    2. Train a TiedLinearRelu autoencoder via ToyModel
    3. Collect latent representations from the trained AE
    4. Train SAESimple on latents (baseline)
    5. Train a CausalSAE on latents (with learned upper-triangular causal matrix)
    6. Compare the learned causal matrix to the ground truth DAG adjacency

"""

# %%
import torch
from torch import Tensor
from torch.optim import AdamW
import matplotlib.pyplot as plt
from occhio.distributions.dag import DAGDistribution
from occhio.distributions.base import Distribution
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
from occhio.sae.sae import SAESimple, CausalSAE, SparseAutoEncoderBase

# %%


# ---------------------------------------------------------------------------
# SAE training loop (works for both SAESimple and CausalSAE)
# ---------------------------------------------------------------------------
def train_sae(
    sae: SparseAutoEncoderBase,
    data_fn,  # callable() -> Tensor of latent batches
    n_steps: int = 10_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
) -> list[float]:
    optimizer = AdamW(sae.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x = data_fn(batch_size)
        optimizer.zero_grad()
        x_hat, z = sae.forward(x)
        loss = sae.loss(x, x_hat, z)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"  SAE step {step + 1}/{n_steps}  loss={loss.item():.4f}")
    return losses


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------
def match_dict_to_features(
    sae: SparseAutoEncoderBase,
    distribution: Distribution,
    ae: TiedLinearRelu,
    n_samples: int = 10_000,
) -> tuple[Tensor, Tensor]:
    """
    For each SAE dictionary element, find which ground-truth feature it best
    corresponds to by correlating SAE activations with ground-truth activations.

    Returns:
        corr_matrix: (n_dict, n_features) correlation matrix
        assignment:   (n_dict,) index of best-matching ground-truth feature
    """
    with torch.no_grad():
        x = distribution.sample(n_samples)  # (N, n_features)
        z_ae = ae.encode(x)  # (N, n_hidden)
        z_sae = sae.encode(z_ae)  # (N, n_dict)

    # Pearson correlation between each SAE dict element and each GT feature
    # x is binary-ish (active/inactive values), z_sae is ReLU activations
    x_centered = x - x.mean(dim=0, keepdim=True)
    z_centered = z_sae - z_sae.mean(dim=0, keepdim=True)

    x_std = x_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
    z_std = z_centered.std(dim=0, keepdim=True).clamp(min=1e-8)

    # (n_dict, n_features) correlation
    corr = (z_centered / z_std).T @ (x_centered / x_std) / n_samples

    assignment = corr.abs().argmax(dim=1)  # best GT feature per dict element
    return corr, assignment


def recover_causal_from_coactivation(
    sae: SparseAutoEncoderBase,
    distribution: DAGDistribution,
    ae: TiedLinearRelu,
    n_samples: int = 50_000,
) -> Tensor:
    """
    Estimate a causal-like adjacency from SAE feature *co-activation*.
    For each pair of SAE features (i, j), compute P(j active | i active).
    High conditional probability suggests i -> j (or shared cause).
    """
    with torch.no_grad():
        x = distribution.sample(n_samples)
        z_ae = ae.encode(x)
        z_sae = sae.encode(z_ae)
        active = (z_sae > 0.01).float()

    # P(j | i) = count(i AND j) / count(i)
    counts_i = active.sum(dim=0).clamp(min=1)  # (n_dict,)
    co_active = active.T @ active  # (n_dict, n_dict)
    cond_prob = co_active / counts_i.unsqueeze(1)
    return cond_prob


# %%


torch.manual_seed(5)

# ---- 1. Ground truth DAG distribution ----
n_features = 8
n_hidden = 5
n_dict = 16

dist = DAGDistribution(
    n_features=n_features,
    p_active=0.10,
    p_edge=0.25,
)
gt_adj = dist.adjacency.float()
print("Ground truth DAG adjacency (upper-triangular):")
print(gt_adj)
print(f"  # edges: {int(gt_adj.sum().item())}")
print()

# ---- 2. Train autoencoder via ToyModel ----
ae = TiedLinearRelu(n_features=n_features, n_hidden=n_hidden)
tm = ToyModel(distribution=dist, ae=ae)

print("Training autoencoder...")
ae_losses = tm.fit(n_epochs=15_000, batch_size=2048, verbose=True)
print(f"  Final AE loss: {ae_losses[-1].item():.6f}")  # ty:ignore
print()


# ---- 3. Latent data generator ----
def sample_latents(batch_size):
    with torch.no_grad():
        x = dist.sample(batch_size)
    return ae.encode(x)


# ---- 4. Train SAESimple (baseline) ----
print("Training SAESimple (baseline)...")
sae_simple = SAESimple(n_latent=n_hidden, n_dict=n_dict, l1_coef=0.02)
simple_losses = train_sae(sae_simple, sample_latents, n_steps=25_000)
print()

# ---- 5. Train CausalSAE ----
print("Training CausalSAE...")
sae_causal = CausalSAE(n_latent=n_hidden, n_dict=n_dict, l1_coef=0.02)
causal_losses = train_sae(sae_causal, sample_latents, n_steps=25_000)
print()

# ---- 6. Analysis ----
print("=" * 60)
print("ANALYSIS")
print("=" * 60)

# 6a. Match dictionary features to ground-truth features
corr_simple, assign_simple = match_dict_to_features(sae_simple, dist, ae)
corr_causal, assign_causal = match_dict_to_features(sae_causal, dist, ae)

print(f"\nSAESimple: dict→GT feature assignment: {assign_simple.tolist()}")
print(f"CausalSAE: dict→GT feature assignment: {assign_causal.tolist()}")

# 6b. Check if CausalSAE learned meaningful causal structure
learned_causal = torch.triu(sae_causal.causal.detach(), diagonal=1)
print("\nLearned causal matrix (upper tri) stats:")
print(
    f"  max={learned_causal.max():.3f}  min={learned_causal.min():.3f}  "
    f"mean(abs)={learned_causal.abs().mean():.4f}"
)

# 6c. Project learned causal matrix back to GT feature space
# For each GT feature, pick the dict element most correlated with it
gt_to_dict = corr_causal.abs().argmax(dim=0)  # (n_features,) best dict per GT feature
print(f"\nGT feature → best dict element: {gt_to_dict.tolist()}")

# Extract the sub-matrix of learned causal corresponding to GT features
projected_causal = learned_causal[gt_to_dict][:, gt_to_dict]  # (n_features, n_features)
projected_causal = torch.triu(projected_causal, diagonal=1)

print("\nProjected causal matrix (in GT feature space):")
print(projected_causal.numpy().round(3))
print("\nGround truth adjacency:")
print(gt_adj.numpy().astype(int))

# 6d. Correlation between learned and GT structure
gt_flat = gt_adj[torch.triu(torch.ones(n_features, n_features), diagonal=1).bool()]
pred_flat = projected_causal[
    torch.triu(torch.ones(n_features, n_features), diagonal=1).bool()
]
if gt_flat.std() > 0 and pred_flat.std() > 0:
    r = torch.corrcoef(torch.stack([gt_flat, pred_flat]))[0, 1].item()
    print(f"\nCorrelation (learned vs GT upper-tri): {r:.4f}")
else:
    print("\nCannot compute correlation (zero variance).")

# 6e. Co-activation based recovery (model-free baseline)
cond_simple = recover_causal_from_coactivation(sae_simple, dist, ae)
cond_causal = recover_causal_from_coactivation(sae_causal, dist, ae)

# Use causal SAE's co-activation for the projected comparison
proj_cond_causal = cond_causal[gt_to_dict][:, gt_to_dict]
proj_cond_causal = torch.triu(proj_cond_causal, diagonal=1)

proj_cond_simple = cond_simple[gt_to_dict][:, gt_to_dict]
proj_cond_simple = torch.triu(proj_cond_simple, diagonal=1)

cond_flat = proj_cond_simple[
    torch.triu(torch.ones(n_features, n_features), diagonal=1).bool()
]
if cond_flat.std() > 0:
    r_cond = torch.corrcoef(torch.stack([gt_flat, cond_flat]))[0, 1].item()
    print(f"Correlation (co-activation SAESimple vs GT): {r_cond:.4f}")

cond_causal_flat = proj_cond_causal[
    torch.triu(torch.ones(n_features, n_features), diagonal=1).bool()
]
if cond_causal_flat.std() > 0:
    r_cond_c = torch.corrcoef(torch.stack([gt_flat, cond_causal_flat]))[0, 1].item()
    print(f"Correlation (co-activation CausalSAE vs GT): {r_cond_c:.4f}")

# ---- 7. Plot ----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].set_title("GT DAG adjacency")
axes[0, 0].imshow(gt_adj.numpy(), cmap="Blues", vmin=0, vmax=1)

axes[0, 1].set_title("Learned causal (projected to GT)")
im = axes[0, 1].imshow(projected_causal.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[0, 1])

axes[0, 2].set_title("Co-activation P(j|i) (CausalSAE)")
im2 = axes[0, 2].imshow(proj_cond_causal.numpy(), cmap="Oranges", vmin=0, vmax=1)
plt.colorbar(im2, ax=axes[0, 2])

axes[1, 0].set_title("SAESimple: |correlation| dict↔GT")
axes[1, 0].imshow(
    corr_simple.abs().numpy(), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1
)
axes[1, 0].set_xlabel("GT feature")
axes[1, 0].set_ylabel("Dict element")

axes[1, 1].set_title("CausalSAE: |correlation| dict↔GT")
axes[1, 1].imshow(
    corr_causal.abs().numpy(), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1
)
axes[1, 1].set_xlabel("GT feature")
axes[1, 1].set_ylabel("Dict element")

axes[1, 2].set_title("Training losses")
axes[1, 2].plot(simple_losses, label="SAESimple", alpha=0.7)
axes[1, 2].plot(causal_losses, label="CausalSAE", alpha=0.7)
axes[1, 2].set_xlabel("Step")
axes[1, 2].set_ylabel("Loss")
axes[1, 2].legend()
axes[1, 2].set_yscale("log")

plt.tight_layout()
plt.savefig("causal_sae_experiment.png", dpi=150)
plt.show()
print("\nPlot saved to causal_sae_experiment.png")


# %%
learned_causal = sae_causal.causal.detach().numpy()
# %%
import plotly.express as px


# %%
px.imshow(learned_causal, zmin=-2, zmax=2, color_continuous_scale="RdBu_r")

# %%
px.imshow(corr_causal.T @ learned_causal)
# %%
px.imshow(sae_causal.W_dec.detach())

# %%
px.imshow((sae_causal.encode(ae.W.T) @ sae_causal.causal).detach().numpy())
# %%
px.imshow((sae_causal.encode(ae.W.T) @ sae_causal.causal).detach().numpy())

# %%
dict_samples = sae_causal.encode(tm.sample_latent(20))
# %%
px.imshow(dict_samples.detach().numpy())
# %%
