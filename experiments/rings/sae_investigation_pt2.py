"""Investigation part 2: Active-only SAE training.

Part 1 showed:
  - 95.5% of AE latents are exactly zero (TiedLinearRelu maps 0→0)
  - SAE learns identity (6 alive neurons = 3 pos/neg pairs, W_enc@W_dec ≈ 2I)
  - L0_all is dominated by 95.5% zero samples → can't reach L0_all ≈ 1

This script tests active-only training: filter out zero AE latents before
passing to SAE. This matches how SAEs are used in interpretability (trained
on meaningful activations, not padding/silence).
"""

import torch
from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SparseSpheres, DistributionStack, SparseUniform
from occhio.sae import SAESimple

DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

K_SPHERES, K_LINEAR, N, M = 3, 3, 1, 3
K_TOTAL = 6
N_FEATURES = K_SPHERES * M + K_LINEAR
HIDDEN_DIM = 3
P_ACTIVE = 0.01
IB = 0.95

importances = torch.cat(
    [
        torch.tensor([IB] * 3 + [IB**3] * 3 + [IB**5] * 3),
        torch.tensor([IB**2, IB**4, IB**6]),
    ]
)

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

ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=HIDDEN_DIM)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE, importances=importances)
print("Training AE...")
tm.fit(
    n_epochs=50_000, batch_size=256, learning_rate=1e-3, weight_decay=0.0, verbose=True
)


NORM_THRESHOLD = 0.01  # below this = "inactive"


def data_fn_active(n):
    """Only return AE latents with ||z|| >= threshold (i.e., active features)."""
    collected = []
    while sum(len(c) for c in collected) < n:
        with torch.no_grad():
            x = dist.sample(n * 25).to(DEVICE)
            z = tm.ae.encode(x)
            active = z[z.norm(dim=-1) >= NORM_THRESHOLD]
            if len(active) > 0:
                collected.append(active)
    return torch.cat(collected)[:n]


def data_fn_all(n):
    """All AE latents (including zeros)."""
    with torch.no_grad():
        return tm.ae.encode(dist.sample(n).to(DEVICE))


# ══════════════════════════════════════════════════════════════════════
# TEST 1: L1 sweep with active-only training, evaluated on active-only
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: L1 sweep — trained on ACTIVE-ONLY, eval on ACTIVE-ONLY")
print("=" * 60)

l1_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
print(f"  {'L1':>8}  {'L0':>7}  {'MSE':>10}  {'alive':>5}  {'dead':>5}")

for l1 in l1_values:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=20_000, batch_size=1024, lr=3e-4)

    with torch.no_grad():
        x = data_fn_active(5000)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        l0 = (z > 0).float().sum(1).mean().item()
        mse = ((x - x_hat) ** 2).sum(-1).mean().item()
        alive = int(((z > 0).float().mean(0) > 0).sum().item())

    print(f"  {l1:>8.3f}  {l0:>7.3f}  {mse:>10.6f}  {alive:>5}  {55 - alive:>5}")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: Active-trained SAE evaluated on FULL distribution
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Active-trained SAE evaluated on FULL distribution")
print("=" * 60)

# Pick two L1 values that gave L0 ≈ 1 on active data
for l1 in [0.03, 0.1, 0.3, 1.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=30_000, batch_size=1024, lr=3e-4)

    with torch.no_grad():
        # Eval on active
        x_act = data_fn_active(5000)
        z_act = sae.encode(x_act)
        l0_act = (z_act > 0).float().sum(1).mean().item()
        mse_act = ((x_act - sae.decode(z_act)) ** 2).sum(-1).mean().item()

        # Eval on all (including zeros)
        x_all = data_fn_all(50_000)
        z_all = sae.encode(x_all)
        x_hat_all = sae.decode(z_all)
        l0_all = (z_all > 0).float().sum(1).mean().item()
        mse_all = ((x_all - x_hat_all) ** 2).sum(-1).mean().item()

        # L0 on zeros specifically
        norms = x_all.norm(dim=-1)
        inactive = norms < NORM_THRESHOLD
        l0_zero = (z_all[inactive] > 0).float().sum(1).mean().item()
        mse_zero = ((x_all[inactive] - x_hat_all[inactive]) ** 2).sum(-1).mean().item()

        alive = int(((z_all > 0).float().mean(0) > 0).sum().item())

    print(f"\n  L1={l1}:")
    print(f"    L0_active={l0_act:.3f}  MSE_active={mse_act:.6f}")
    print(f"    L0_all={l0_all:.3f}     MSE_all={mse_all:.6f}")
    print(f"    L0_zero={l0_zero:.3f}   MSE_zero={mse_zero:.6f}")
    print(f"    alive={alive}  dead={55 - alive}")
    print(
        f"    b_enc range: [{sae.b_enc.min().item():.4f}, {sae.b_enc.max().item():.4f}]"
    )


# ══════════════════════════════════════════════════════════════════════
# TEST 3: Feature splitting analysis — does the active-trained SAE
# learn monosemantic features (1 GT feature → 1 SAE neuron)?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Feature splitting analysis (active-trained)")
print("=" * 60)

feature_names = [f"sphere_{i}" for i in range(K_SPHERES)] + [
    f"linear_{i}" for i in range(K_LINEAR)
]

for l1 in [0.03, 0.1, 0.3, 1.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=30_000, batch_size=1024, lr=3e-4)

    with torch.no_grad():
        x_act = data_fn_active(5000)
        z_sae = sae.encode(x_act)
        l0 = (z_sae > 0).float().sum(1).mean().item()
        alive = int(((z_sae > 0).float().mean(0) > 0).sum().item())

        # Also get GT labels for active samples
        # Sample with labels
        sphere_data, sphere_mask = sphere_dist.sample_with_args(200_000)
        linear_data = linear_dist.sample(200_000)
        linear_mask = linear_data > 0
        full_data = torch.cat([sphere_data, linear_data], dim=-1).to(DEVICE)
        gt_mask = torch.cat([sphere_mask, linear_mask], dim=-1)

        z_ae = tm.ae.encode(full_data)
        active_mask = z_ae.norm(dim=-1) >= NORM_THRESHOLD
        n_gt_active = gt_mask.sum(1)
        is_single = n_gt_active == 1
        feat_id = gt_mask.float().argmax(1)

        # Filter to active + single-feature samples
        use = active_mask & is_single
        z_ae_use = z_ae[use]
        z_sae_use = sae.encode(z_ae_use)
        feat_id_use = feat_id[use]

    print(f"\n  L1={l1}: L0_active={l0:.3f}, alive={alive}")

    # For each GT feature, which SAE neurons fire most?
    for j in range(K_TOTAL):
        fm = feat_id_use == j
        if fm.sum() < 10:
            print(f"    {feature_names[j]:>10}: too few samples")
            continue
        acts = z_sae_use[fm]
        mean_act = acts.mean(0)
        top3 = torch.topk(mean_act, min(5, 55))
        total = mean_act.sum().item()
        parts = []
        for k in range(5):
            v = top3.values[k].item()
            if v < 0.001:
                break
            pct = v / total * 100 if total > 0 else 0
            parts.append(f"n{top3.indices[k].item()}({pct:.0f}%)")
        n_neurons_used = (mean_act > 0.001).sum().item()
        print(
            f"    {feature_names[j]:>10} → {', '.join(parts)}  "
            f"[{int(n_neurons_used)} neurons total]"
        )

    # Check monosemanticity: for each alive neuron, which GT features activate it?
    alive_idx = torch.where((z_sae_use > 0).float().mean(0) > 0)[0]
    print(f"\n    Monosemanticity check ({len(alive_idx)} alive neurons):")
    for idx in alive_idx[:15]:  # cap at 15 for readability
        neuron_acts = z_sae_use[:, idx]
        fires = neuron_acts > 0
        if fires.sum() < 5:
            continue
        # Which GT features does this neuron respond to?
        gt_counts = []
        for j in range(K_TOTAL):
            fm = feat_id_use == j
            n_fires_for_j = (fires & fm).sum().item()
            n_total_j = fm.sum().item()
            if n_total_j > 0:
                gt_counts.append((feature_names[j], n_fires_for_j / n_total_j))
        gt_counts.sort(key=lambda x: -x[1])
        top_gt = [f"{name}({rate:.0%})" for name, rate in gt_counts if rate > 0.01]
        print(f"      n{idx.item():2d}: fires on {', '.join(top_gt[:4])}")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
