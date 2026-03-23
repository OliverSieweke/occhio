# %% [markdown]
# # Matching Pursuit SAE — Toy Model Experiment
#
# Train a toy model in the superposition regime, then compare MP-SAE
# and standard SAE (SAESimple) at recovering features from the
# compressed latent representations.

# %% Imports
import torch
import matplotlib.pyplot as plt

from occhio import ToyModel
from occhio.autoencoder import TiedLinear
from occhio.distributions.sparse import SparseUniform
from occhio.sae import SAESimple, MatchingPursuitSAE

torch.manual_seed(42)

# %% Configuration
N_FEATURES = 12  # true features in the distribution
N_HIDDEN = 4  # bottleneck — forces superposition
P_ACTIVE = 0.05  # feature activation probability
N_DICT = 24  # SAE dictionary size (2× overcomplete)

# %% Create and train toy model
dist = SparseUniform(n_features=N_FEATURES, p_active=P_ACTIVE)
ae = TiedLinear(n_features=N_FEATURES, n_hidden=N_HIDDEN)
tm = ToyModel(distribution=dist, ae=ae)

tm_losses, _ = tm.fit(n_epochs=20_000, batch_size=512)

print(f"Final TM loss: {tm_losses[-1]:.6f}")
print(f"Feature norms: {tm.feature_norms.numpy().round(3)}")

# %% Plot toy model training
fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(tm_losses)
ax.set(title="Toy Model Training", xlabel="Epoch", ylabel="Loss")
plt.tight_layout()
plt.show()


# %% Data function: generate latent representations from the trained TM
def latent_data_fn(n_samples):
    with torch.no_grad():
        return tm.sample_latent(n_samples)


# %% Train MP-SAE
mp_sae = MatchingPursuitSAE(
    n_latent=N_HIDDEN,
    n_dict=N_DICT,
    threshold=1e-3,
    max_iterations=10,
)

mp_losses = mp_sae.train_sae(
    data_fn=latent_data_fn,
    n_steps=10_000,
    batch_size=512,
    lr=3e-4,
)

# %% Train standard SAE for comparison
std_sae = SAESimple(
    n_latent=N_HIDDEN,
    n_dict=N_DICT,
    l1_coef=0.01,
)

std_losses = std_sae.train_sae(
    data_fn=latent_data_fn,
    n_steps=10_000,
    batch_size=512,
    lr=3e-4,
)

# %% Compare training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(mp_losses)
axes[0].set(title="MP-SAE Training Loss", xlabel="Step", ylabel="Loss")
axes[1].semilogy(std_losses)
axes[1].set(title="SAESimple Training Loss", xlabel="Step", ylabel="Loss")
plt.tight_layout()
plt.show()

# %% Evaluate reconstruction quality and sparsity
with torch.no_grad():
    test_data = latent_data_fn(2000)

    mp_recon, mp_z = mp_sae(test_data)
    mp_mse = (test_data - mp_recon).pow(2).sum(-1).mean().item()
    mp_l0 = (mp_z != 0).float().sum(-1).mean().item()

    std_recon, std_z = std_sae(test_data)
    std_mse = (test_data - std_recon).pow(2).sum(-1).mean().item()
    std_l0 = (std_z > 0).float().sum(-1).mean().item()

print(f"{'':15s} {'MSE':>10s}  {'L0':>6s}")
print(f"{'MP-SAE':15s} {mp_mse:10.6f}  {mp_l0:6.2f}")
print(f"{'SAESimple':15s} {std_mse:10.6f}  {std_l0:6.2f}")

# %% Feature recovery analysis
#
# For each true feature direction (column of W from the toy model),
# find the best-matching dictionary atom (highest |cos-sim|).

W_tm = tm.W  # (n_hidden, n_features)
W_tm_norm = torch.nn.functional.normalize(W_tm.T, dim=1)  # (n_features, n_hidden)

# MP-SAE atoms (unit-norm by construction)
W_mp = mp_sae.W.data  # (n_dict, n_hidden)

# Standard SAE decoder rows, normalised
W_std = std_sae.W_dec.data  # (n_dict, n_latent)
W_std_norm = torch.nn.functional.normalize(W_std, dim=1)

mp_cos = (W_tm_norm @ W_mp.T).abs().max(dim=1).values
std_cos = (W_tm_norm @ W_std_norm.T).abs().max(dim=1).values

print(f"\nFeature recovery (max |cos-sim| to nearest atom):")
print(f"  {'Feature':>7s}  {'TM norm':>7s}  {'MP-SAE':>7s}  {'Simple':>7s}")
print(f"  {'-------':>7s}  {'-------':>7s}  {'------':>7s}  {'------':>7s}")
for i in range(N_FEATURES):
    print(f"  {i:7d}  {tm.feature_norms[i]:7.3f}  {mp_cos[i]:7.3f}  {std_cos[i]:7.3f}")
print(f"\n  Mean:  MP-SAE={mp_cos.mean():.3f}  SAESimple={std_cos.mean():.3f}")

# %% Visualise atom–feature cosine similarity matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow((W_tm_norm @ W_mp.T).abs().numpy(), aspect="auto", cmap="viridis")
axes[0].set(
    title="MP-SAE |cos-sim| (features × atoms)",
    xlabel="Dictionary atom",
    ylabel="True feature",
)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    (W_tm_norm @ W_std_norm.T).abs().numpy(), aspect="auto", cmap="viridis"
)
axes[1].set(
    title="SAESimple |cos-sim| (features × atoms)",
    xlabel="Dictionary atom",
    ylabel="True feature",
)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

# %% L0 distribution histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

with torch.no_grad():
    test = latent_data_fn(5000)
    _, mp_acts = mp_sae(test)
    _, std_acts = std_sae(test)

    mp_l0s = (mp_acts != 0).float().sum(-1).numpy()
    std_l0s = (std_acts > 0).float().sum(-1).numpy()

axes[0].hist(mp_l0s, bins=range(int(mp_l0s.max()) + 2), edgecolor="black")
axes[0].set(title="MP-SAE L0 distribution", xlabel="L0", ylabel="Count")

axes[1].hist(std_l0s, bins=range(int(std_l0s.max()) + 2), edgecolor="black")
axes[1].set(title="SAESimple L0 distribution", xlabel="L0", ylabel="Count")

plt.tight_layout()
plt.show()

# %% Vary max_iterations at inference time (adaptive sparsity)
#
# A key property of MP-SAE: you can change T at inference time and
# reconstruction quality monotonically improves.

max_iters_range = range(1, 16)
mses = []
l0s = []

with torch.no_grad():
    test = latent_data_fn(2000)
    for T in max_iters_range:
        # Temporarily override max_iterations
        old_T = mp_sae.max_iterations
        mp_sae.max_iterations = T
        recon, z = mp_sae(test)
        mp_sae.max_iterations = old_T

        mses.append((test - recon).pow(2).sum(-1).mean().item())
        l0s.append((z != 0).float().sum(-1).mean().item())

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(list(max_iters_range), mses, "o-b", label="MSE")
ax1.set(xlabel="Max iterations (T)", ylabel="MSE", title="MP-SAE: Adaptive Sparsity")
ax1.legend(loc="upper right")

ax2 = ax1.twinx()
ax2.plot(list(max_iters_range), l0s, "s-r", label="L0")
ax2.set_ylabel("L0")
ax2.legend(loc="center right")

plt.tight_layout()
plt.show()

# %%
