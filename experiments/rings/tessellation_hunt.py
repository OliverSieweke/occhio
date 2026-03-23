"""Can the optimizer find tessellation (1 atom per discretization point)?

With 11 discretizations × 3 spheres + 3 linear = 36 atoms needed.
n_dict=55 has plenty of capacity. Why doesn't it work?

Focused tests:
1. Baseline at various L1 (identity reference)
2. Negative bias init (narrow cones from start)
3. Long training at moderate L1 (duration hypothesis)
4. Neg bias + warmup + long training (best combined shot)
"""

import sys
import torch
from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import SparseSpheres, DistributionStack, SparseUniform
from occhio.sae import SAESimple

DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

K_SPHERES, K_LINEAR, N, M = 3, 3, 1, 3
N_FEATURES = K_SPHERES * M + K_LINEAR
HIDDEN_DIM = 3
P_ACTIVE = 0.07
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
    noise_std=0.18,
    n_discretizations=11,
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
print("Training AE...", flush=True)
tm.fit(
    n_epochs=20_000, batch_size=256, learning_rate=1e-3, weight_decay=0.0, verbose=True
)
print("AE training done.", flush=True)

NORM_THRESHOLD = 0.01


def data_fn_active(n):
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
    with torch.no_grad():
        return tm.ae.encode(dist.sample(n).to(DEVICE))


def eval_sae(sae, label=""):
    with torch.no_grad():
        x_act = data_fn_active(5000)
        z = sae.encode(x_act)
        x_hat = sae.decode(z)
        l0 = (z > 0).float().sum(1).mean().item()
        mse = ((x_act - x_hat) ** 2).sum(-1).mean().item()
        alive = int(((z > 0).float().mean(0) > 0).sum().item())
        b_min = sae.b_enc.min().item()
        b_max = sae.b_enc.max().item()
        b_neg = int((sae.b_enc < -0.3).sum().item())

        # Angular selectivity: what fraction of active data does each alive neuron fire on?
        fire_rates = (z > 0).float().mean(0)
        alive_rates = fire_rates[fire_rates > 0]
        max_rate = alive_rates.max().item() if len(alive_rates) > 0 else 0
        min_rate = alive_rates.min().item() if len(alive_rates) > 0 else 0
        med_rate = alive_rates.median().item() if len(alive_rates) > 0 else 0

    print(
        f"  {label:45s} L0={l0:.2f} MSE={mse:.4f} alive={alive:3d} "
        f"b=[{b_min:.2f},{b_max:.2f}] b<-0.3:{b_neg:3d} "
        f"fire_rate=[{min_rate:.3f},{med_rate:.3f},{max_rate:.3f}]",
        flush=True,
    )
    return {"l0": l0, "mse": mse, "alive": alive}


def train_warmup(sae, data_fn, n_steps, batch_size, lr, l1_final, warmup_frac=0.5):
    """Train with L1 warmup."""
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)
    warmup_steps = int(n_steps * warmup_frac)
    for step in range(n_steps):
        sae.l1_coef = l1_final * min(1.0, step / max(1, warmup_steps))
        x = data_fn(batch_size)
        optimizer.zero_grad()
        x_hat, z = sae.forward(x)
        loss = sae.loss(x, x_hat, z)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10000 == 0:
            with torch.no_grad():
                alive = int(((z > 0).float().mean(0) > 0).sum().item())
            print(
                f"    step {step + 1}/{n_steps} loss={loss.item():.4f} alive={alive}",
                flush=True,
            )


# ═══════════════════════════════════════════════════
# TEST 1: L1 sweep baseline (30k steps, standard init)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("TEST 1: L1 sweep baseline (30k steps)", flush=True)
print("=" * 70, flush=True)

for l1 in [0.3, 1.0, 3.0, 5.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=30_000, batch_size=1024, lr=3e-4)
    eval_sae(sae, f"baseline L1={l1}")

# ═══════════════════════════════════════════════════
# TEST 2: Negative bias init (the key lever for narrow cones)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("TEST 2: Negative bias init", flush=True)
print("=" * 70, flush=True)

for init_bias in [-0.5, -1.0, -1.5]:
    for l1 in [1.0, 3.0]:
        sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
        with torch.no_grad():
            sae.b_enc.fill_(init_bias)
        sae.train_sae(data_fn_active, n_steps=30_000, batch_size=1024, lr=3e-4)
        eval_sae(sae, f"neg_bias={init_bias} L1={l1}")

# ═══════════════════════════════════════════════════
# TEST 3: Long training (user's hypothesis: maybe just needs more time)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("TEST 3: Long training 80k steps", flush=True)
print("=" * 70, flush=True)

for l1 in [1.0, 3.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=80_000, batch_size=1024, lr=3e-4)
    eval_sae(sae, f"long 80k L1={l1}")

# With lower LR (slower, more careful optimization)
for l1 in [1.0, 3.0]:
    sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=80_000, batch_size=1024, lr=1e-4)
    eval_sae(sae, f"long 80k lr=1e-4 L1={l1}")

# ═══════════════════════════════════════════════════
# TEST 4: Best combined shot: neg bias + warmup + long training
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("TEST 4: Neg bias + warmup + 80k steps", flush=True)
print("=" * 70, flush=True)

for init_bias in [-0.5, -1.0]:
    for l1 in [1.0, 3.0]:
        sae = SAESimple(n_latent=HIDDEN_DIM, n_dict=55, l1_coef=0.0, device=DEVICE)
        with torch.no_grad():
            sae.b_enc.fill_(init_bias)
        train_warmup(sae, data_fn_active, 80_000, 1024, 1e-4, l1, warmup_frac=0.3)
        sae.l1_coef = l1
        eval_sae(sae, f"combined b={init_bias} L1={l1} 80k")


print("\n" + "=" * 70, flush=True)
print("DONE", flush=True)
print("=" * 70, flush=True)
