"""Investigation: Is SAESimple learning identity? How to get L0 ≈ 1?

Hypothesis: With TiedLinearRelu, encode(x) = x @ W.T (no bias). So zero input →
zero latent. Since p_active=0.01, ~94% of samples are zero. The SAE trivially
gets L0=0 on those, masking the behavior on the ~6% active samples. Overall
L0 ≈ 0.06 * L0_active.

This script tests:
  1. Data distribution (what fraction is zero?)
  2. Identity test (W_enc @ W_dec structure)
  3. L0 conditioned on active vs inactive samples
  4. FIX: centering data (subtract mean) before SAE training
  5. L1 sweep with centered vs uncentered data
"""

import torch
import numpy as np

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SparseSpheres, DistributionStack, SparseUniform
from occhio.sae import SAESimple

DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

# ── Same config as multidim_experiment.py ──
K_SPHERES, K_LINEAR, N, M = 3, 3, 1, 3
K_TOTAL = K_SPHERES + K_LINEAR
N_FEATURES = K_SPHERES * M + K_LINEAR  # 12
HIDDEN_DIM = 3
P_ACTIVE = 0.01
IB = 0.95

sphere_importances = torch.tensor([IB] * 3 + [IB**3] * 3 + [IB**5] * 3)
linear_importances = torch.tensor([IB**2, IB**4, IB**6])
importances = torch.cat([sphere_importances, linear_importances])

sphere_dist = SparseSpheres(
    n_spheres=K_SPHERES,
    sphere_dim=N,
    ambient_dim=M,
    p_active=P_ACTIVE,
    p_infill=0.0,
    radius=1.0,
    noise_std=0.0,
    n_discretizations=100,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)
linear_dist = SparseUniform(
    n_features=K_LINEAR,
    p_active=P_ACTIVE,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED + 1),
    device=DEVICE,
)
dist = DistributionStack([sphere_dist, linear_dist], sampling_mode="independent")

# Train AE
ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=HIDDEN_DIM)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE, importances=importances)
print("Training AE (50k epochs)...")
tm.fit(
    n_epochs=50_000, batch_size=256, learning_rate=1e-3, weight_decay=0.0, verbose=True
)


def make_data_fn(centered=False, active_only=False):
    """Factory for SAE data functions with optional centering/filtering."""
    # Precompute mean for centering
    if centered:
        with torch.no_grad():
            big_sample = dist.sample(100_000).to(DEVICE)
            mu = tm.ae.encode(big_sample).mean(dim=0)
        print(f"  Data mean (||μ||={mu.norm():.4f}): {mu.tolist()}")
    else:
        mu = None

    def data_fn(n):
        with torch.no_grad():
            if active_only:
                # Oversample and filter
                x = dist.sample(n * 25).to(DEVICE)
                z = tm.ae.encode(x)
                mask = z.norm(dim=-1) >= 0.01
                z = z[mask][:n]
            else:
                x = dist.sample(n).to(DEVICE)
                z = tm.ae.encode(x)
            if mu is not None:
                z = z - mu
            return z

    return data_fn


def eval_sae(sae, data_fn_all, n=10_000):
    """Evaluate SAE with breakdown by active/inactive input."""
    with torch.no_grad():
        x = data_fn_all(n)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        fired = (z > 0).float()

        norms = x.norm(dim=-1)
        active = norms >= 0.01
        inactive = ~active

        l0_all = fired.sum(1).mean().item()
        l0_act = fired[active].sum(1).mean().item() if active.any() else 0
        l0_inact = fired[inactive].sum(1).mean().item() if inactive.any() else 0
        mse = ((x - x_hat) ** 2).sum(-1).mean().item()
        alive = int((fired.mean(0) > 0).sum().item())
        frac_active = active.float().mean().item()

    return {
        "l0_all": l0_all,
        "l0_active": l0_act,
        "l0_inactive": l0_inact,
        "mse": mse,
        "alive": alive,
        "dead": 55 - alive,
        "frac_active": frac_active,
    }


# ══════════════════════════════════════════════════════════════════════
# TEST 1: Data distribution
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: Data distribution analysis")
print("=" * 60)

with torch.no_grad():
    x_raw = dist.sample(100_000).to(DEVICE)
    z_ae = tm.ae.encode(x_raw)
    norms = z_ae.norm(dim=-1)
    frac_zero = (norms < 0.01).float().mean().item()

print(f"  AE latent shape: {z_ae.shape}")
print(f"  Near-zero (||z||<0.01): {frac_zero * 100:.1f}%")
print(f"  Nonzero: {(1 - frac_zero) * 100:.1f}%")
print(
    f"  Active latent norms: mean={norms[norms >= 0.01].mean():.4f}, "
    f"std={norms[norms >= 0.01].std():.4f}"
)
print(f"  → Expected overall L0 ≈ {1 - frac_zero:.3f} × L0_active")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: Identity test — train SAE on raw data, inspect weights
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Identity test (L1=0.01, 55 latents, raw data)")
print("=" * 60)

sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=0.01, device=DEVICE)
sae.train_sae(make_data_fn(), n_steps=30_000, batch_size=1024, lr=3e-4)

m = eval_sae(sae, make_data_fn())
print(f"  Alive: {m['alive']}, Dead: {m['dead']}")
print(f"  L0_all: {m['l0_all']:.3f}")
print(f"  L0_active: {m['l0_active']:.3f}")
print(f"  L0_inactive: {m['l0_inactive']:.3f}")
print(f"  MSE: {m['mse']:.6f}")

# Check if alive neurons form identity-like structure
W_enc = sae.W_enc.detach()  # (3, 55)
W_dec = sae.W_dec.detach()  # (55, 3)
b_enc = sae.b_enc.detach()  # (55,)

with torch.no_grad():
    z_test = sae.encode(make_data_fn()(5000))
    alive_mask = (z_test > 0).float().mean(0) > 0

alive_idx = torch.where(alive_mask)[0]
print(f"\n  Alive neurons: {alive_idx.tolist()}")
print(f"  b_enc (alive): {[f'{b_enc[i].item():.4f}' for i in alive_idx]}")

if len(alive_idx) > 0:
    W_enc_alive = W_enc[:, alive_idx]  # (3, n_alive)
    W_dec_alive = W_dec[alive_idx, :]  # (n_alive, 3)
    product = W_enc_alive @ W_dec_alive  # (3, 3)
    identity_err = ((product - torch.eye(3)) ** 2).mean().item()
    print(f"\n  W_enc_alive @ W_dec_alive:")
    for row in product.numpy():
        print(f"    [{', '.join(f'{v:7.4f}' for v in row)}]")
    print(f"  MSE from identity: {identity_err:.6f}")

    # Cosine similarity between encoder and decoder direction pairs
    print(f"\n  Encoder-decoder direction alignment:")
    for i, idx in enumerate(alive_idx):
        enc = W_enc[:, idx]
        dec = W_dec[idx, :]
        cos = torch.cosine_similarity(enc.unsqueeze(0), dec.unsqueeze(0)).item()
        print(
            f"    n{idx.item()}: cos(enc,dec)={cos:+.4f}  "
            f"enc_norm={enc.norm():.3f}  dec_norm={dec.norm():.3f}"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST 3: L1 sweep — raw data vs centered data
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: L1 sweep with 55 latents")
print("=" * 60)

l1_values = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

for label, centered in [("RAW (uncentered)", False), ("CENTERED (z - μ)", True)]:
    print(f"\n  ── {label} ──")
    print(
        f"  {'L1':>8}  {'L0_all':>7}  {'L0_act':>7}  {'L0_inact':>8}  "
        f"{'MSE':>10}  {'alive':>5}  {'dead':>5}"
    )

    train_fn = make_data_fn(centered=centered)
    eval_fn = make_data_fn(centered=centered)

    for l1 in l1_values:
        sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
        sae.train_sae(train_fn, n_steps=20_000, batch_size=1024, lr=3e-4)
        m = eval_sae(sae, eval_fn)
        print(
            f"  {l1:>8.3f}  {m['l0_all']:>7.3f}  {m['l0_active']:>7.3f}  "
            f"{m['l0_inactive']:>8.3f}  {m['mse']:>10.6f}  "
            f"{m['alive']:>5}  {m['dead']:>5}"
        )


# ══════════════════════════════════════════════════════════════════════
# TEST 4: Feature splitting check — with centered data, does L1 cause
# individual features to split into multiple SAE atoms?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Feature splitting analysis (centered, best L1)")
print("=" * 60)

# Train with centered data at a moderate L1
print("  Training with centered data...")
train_fn = make_data_fn(centered=True)

# Try L1=1.0 (should give moderate sparsity)
for l1 in [0.3, 1.0, 3.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(train_fn, n_steps=30_000, batch_size=1024, lr=3e-4)
    m = eval_sae(sae, train_fn)
    print(
        f"\n  L1={l1}: L0_all={m['l0_all']:.3f}, alive={m['alive']}, MSE={m['mse']:.6f}"
    )

    # Check which GT features activate which SAE neurons
    with torch.no_grad():
        sphere_data, sphere_mask = sphere_dist.sample_with_args(20_000)
        linear_data = linear_dist.sample(20_000)
        linear_mask = linear_data > 0
        full_data = torch.cat([sphere_data, linear_data], dim=-1).to(DEVICE)
        gt_mask = torch.cat([sphere_mask, linear_mask], dim=-1)

        z_ae = tm.ae.encode(full_data)
        mu = z_ae.mean(0)  # for centering
        z_sae = sae.encode(z_ae - mu)

        n_active_gt = gt_mask.sum(1)
        is_single = n_active_gt == 1
        feat_id = gt_mask.float().argmax(1)

    feature_names = [f"sphere_{i}" for i in range(K_SPHERES)] + [
        f"linear_{i}" for i in range(K_LINEAR)
    ]

    print(f"  Feature → SAE neuron mapping:")
    for j in range(K_TOTAL):
        fm = is_single & (feat_id == j)
        if fm.sum() < 10:
            print(f"    {feature_names[j]}: too few samples ({fm.sum()})")
            continue
        acts = z_sae[fm]  # (n_samples, 55)
        mean_act = acts.mean(0)
        top3 = torch.topk(mean_act, 3)
        total = mean_act.sum().item()
        parts = []
        for k in range(3):
            v = top3.values[k].item()
            if v < 0.001:
                break
            pct = v / total * 100 if total > 0 else 0
            parts.append(f"n{top3.indices[k].item()}({pct:.0f}%)")
        print(f"    {feature_names[j]:>10} → {', '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: Sanity check — does centering harm reconstruction of original?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Centered SAE — reconstruction of ORIGINAL (uncentered) data")
print("=" * 60)

print("  Training centered SAE (L1=1.0)...")
train_fn_c = make_data_fn(centered=True)
sae_c = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=1.0, device=DEVICE)
sae_c.train_sae(train_fn_c, n_steps=30_000, batch_size=1024, lr=3e-4)

with torch.no_grad():
    big = dist.sample(50_000).to(DEVICE)
    z_ae = tm.ae.encode(big)
    mu = z_ae.mean(0)

    # Reconstruct via centered SAE: decode(encode(z - μ)) + μ
    z_centered = z_ae - mu
    z_sae = sae_c.encode(z_centered)
    recon_centered = sae_c.decode(z_sae) + mu  # add mean back

    mse_orig = ((z_ae - recon_centered) ** 2).sum(-1).mean().item()
    l0 = (z_sae > 0).float().sum(1).mean().item()

    norms = z_ae.norm(dim=-1)
    active = norms >= 0.01
    mse_active = ((z_ae[active] - recon_centered[active]) ** 2).sum(-1).mean().item()
    mse_inactive = (
        ((z_ae[~active] - recon_centered[~active]) ** 2).sum(-1).mean().item()
    )

print(f"  L0: {l0:.3f}")
print(f"  MSE (all, original space): {mse_orig:.6f}")
print(f"  MSE (active only): {mse_active:.6f}")
print(f"  MSE (inactive only): {mse_inactive:.6f}")
print(
    f"  → Inactive MSE = cost of centering: {mse_inactive:.6f} "
    f"(||μ - recon(−μ) + μ||² ≈ ||μ||² = {mu.norm().item() ** 2:.6f})"
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
