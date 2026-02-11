"""
Experiment: Can the CausalSAE recover hierarchical pair structure?

Pipeline:
    1. Create ScaledHierarchicalPairs distribution (even feature i → odd feature i+1)
    2. Train a TiedLinearRelu autoencoder via ToyModel
    3. Collect latent representations from the trained AE
    4. Train SAESimple on latents (baseline)
    5. Train a CausalSAE on latents (with learned upper-triangular causal matrix)
    6. Compare the learned causal matrix to the ground-truth pair structure

"""

# %%
import torch
from torch import Tensor
from torch.optim import AdamW
import matplotlib.pyplot as plt
from occhio.distributions.correlated import ScaledHierarchicalPairs
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
from occhio.sae.sae import SAESimple, CausalSAE, SparseAutoEncoderBase

# %%


# ---------------------------------------------------------------------------
# SAE training loop (works for both SAESimple and CausalSAE)
# ---------------------------------------------------------------------------
def train_sae(
    sae: SparseAutoEncoderBase,
    data_fn,  # callable(batch_size) -> Tensor of latent batches
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
    ae: TiedLinearRelu,
) -> tuple[Tensor, Tensor]:
    """
    For each SAE dictionary element, find which ground-truth feature it best
    corresponds to by passing one-hot feature vectors through the AE encoder
    (= ae.W.T) and then through the SAE encoder.

    Returns:
        activation_matrix: (n_dict, n_features) SAE activations per GT feature
        assignment:         (n_dict,) index of best-matching GT feature per dict element
    """
    with torch.no_grad():
        # ae.encode(one_hot) = one_hot @ W.T = W.T, shape (n_features, n_hidden)
        encoded_onehots = ae.W.T  # (n_features, n_hidden)
        sae_acts = sae.encode(encoded_onehots)  # (n_features, n_dict)

    # Transpose to (n_dict, n_features) to match old interface
    act_matrix = sae_acts.T
    assignment = act_matrix.abs().argmax(dim=1)  # best GT feature per dict element
    return act_matrix, assignment


def recover_causal_from_coactivation(
    sae: SparseAutoEncoderBase,
    distribution: ScaledHierarchicalPairs,
    ae: TiedLinearRelu,
    n_samples: int = 50_000,
) -> Tensor:
    """
    Estimate a causal-like adjacency from SAE feature *co-activation*.
    For each pair of SAE features (i, j), compute P(j active | i active).
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


torch.manual_seed(6)

# ---- 1. Ground truth ScaledHierarchicalPairs distribution ----
n_features = 8
n_hidden = 4
n_dict = 10

dist = ScaledHierarchicalPairs(
    n_features=n_features,
    p_active=0.10,
    p_follow=0.5,
)

# Ground-truth causal adjacency: feature 2i -> feature 2i+1
gt_adj = torch.zeros(n_features, n_features)
for i in range(n_features // 2):
    gt_adj[2 * i, 2 * i + 1] = 1.0

print("Ground truth pair structure (even → odd):")
print(gt_adj.int())
print(f"  # edges: {int(gt_adj.sum().item())}")
print()

# ---- 2. Train autoencoder via ToyModel ----
ae = TiedLinearRelu(n_features=n_features, n_hidden=n_hidden)
tm = ToyModel(distribution=dist, ae=ae)

print("Training autoencoder...")
ae_losses = tm.fit(n_epochs=15_000, batch_size=2048, verbose=True)
print(f"  Final AE loss: {ae_losses[-1].item():.6f}")
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
sae_causal = CausalSAE(
    n_latent=n_hidden, n_dict=n_dict, l1_coef=0.02, l1_dirc=0.0, l1_causal=0.01
)
causal_losses = train_sae(sae_causal, sample_latents, n_steps=25_000)
print()

# ---- 6. Analysis ----
print("=" * 60)
print("ANALYSIS")
print("=" * 60)

# 6a. Match dictionary features to ground-truth features
corr_simple, assign_simple = match_dict_to_features(sae_simple, ae)
corr_causal, assign_causal = match_dict_to_features(sae_causal, ae)

print(f"\nSAESimple: dict→GT feature assignment: {assign_simple.tolist()}")
print(f"CausalSAE: dict→GT feature assignment: {assign_causal.tolist()}")

# 6b. Raw learned causal matrix stats
learned_causal = sae_causal.causal.detach()
learned_upper = torch.triu(learned_causal, diagonal=1)
print("\nLearned causal matrix (upper tri) stats:")
print(
    f"  max={learned_upper.max():.3f}  min={learned_upper.min():.3f}  "
    f"mean(abs)={learned_upper.abs().mean():.4f}"
)

# 6c. Project learned causal matrix back to GT feature space
gt_to_dict = corr_causal.abs().argmax(dim=0)  # (n_features,) best dict per GT feature
print(f"\nGT feature → best dict element: {gt_to_dict.tolist()}")

projected_causal = learned_upper[gt_to_dict][:, gt_to_dict]  # (n_features, n_features)
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

# 6e. Co-activation based recovery
cond_simple = recover_causal_from_coactivation(sae_simple, dist, ae)
cond_causal = recover_causal_from_coactivation(sae_causal, dist, ae)

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

axes[0, 0].set_title("GT pair adjacency (even→odd)")
axes[0, 0].imshow(gt_adj.numpy(), cmap="Blues", vmin=0, vmax=1)

axes[0, 1].set_title("Learned causal (projected to GT)")
im = axes[0, 1].imshow(projected_causal.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[0, 1])

axes[0, 2].set_title("Raw CausalSAE.causal matrix")
im_raw = axes[0, 2].imshow(learned_causal.numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
plt.colorbar(im_raw, ax=axes[0, 2])
axes[0, 2].set_xlabel("Dict element")
axes[0, 2].set_ylabel("Dict element")

axes[1, 0].set_title("SAESimple: one-hot activations")
axes[1, 0].imshow(
    corr_simple.numpy(), cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5
)
axes[1, 0].set_xlabel("GT feature")
axes[1, 0].set_ylabel("Dict element")

axes[1, 1].set_title("CausalSAE: one-hot activations")
axes[1, 1].imshow(
    corr_causal.numpy(), cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5
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
plt.savefig("hierarchical_sae_experiment.png", dpi=150)
plt.show()
print("\nPlot saved to hierarchical_sae_experiment.png")


# %%
# ---- 8. SAE activations when hierarchical pairs co-fire ----
n_pairs = n_features // 2

with torch.no_grad():
    x_big = dist.sample(100_000)
    z_ae_big = ae.encode(x_big)
    z_simple_big = sae_simple.encode(z_ae_big)  # (N, n_dict)
    z_causal_big = sae_causal.encode(z_ae_big)  # (N, n_dict)

fig2, axes2 = plt.subplots(n_pairs, 2, figsize=(12, 3 * n_pairs))

for pair_idx in range(n_pairs):
    even = 2 * pair_idx
    odd = even + 1
    # Both features in the pair must be active (nonzero) in the original sample
    mask = (x_big[:, even] > 0) & (x_big[:, odd] > 0)
    n_co = mask.sum().item()

    # Mean SAE activations over co-firing samples
    mean_simple = z_simple_big[mask].mean(dim=0).numpy()
    mean_causal = z_causal_big[mask].mean(dim=0).numpy()

    ax_s = axes2[pair_idx, 0]
    ax_c = axes2[pair_idx, 1]

    ax_s.bar(range(n_dict), mean_simple)
    ax_s.set_title(f"SAESimple | pair ({even},{odd}) co-fire  (n={n_co})")
    ax_s.set_xlabel("Dict element")
    ax_s.set_ylabel("Mean activation")

    ax_c.bar(range(n_dict), mean_causal)
    ax_c.set_title(f"CausalSAE | pair ({even},{odd}) co-fire  (n={n_co})")
    ax_c.set_xlabel("Dict element")
    ax_c.set_ylabel("Mean activation")

fig2.suptitle(
    "SAE activations when both features in a hierarchical pair fire", fontsize=13
)
fig2.tight_layout()
plt.show()

# %%
dist.sample(1)

# %%
