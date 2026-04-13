"""Quick eval: L1=0.4, 5 seeds, 3 models. Reports L0, R2, EP, MCC, F1."""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions import SparseUniform
from occhio.toy_model import ToyModel

# --- Config ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 500
D_HIDDEN = 64
N_EPOCHS = 30_000
N_EPOCHS_SYNTH = 15_000
BATCH_SIZE = 512

L1 = 0.4
N_DICT = N_FEATURES // 2
SAE_STEPS = 25_000
SAE_BATCH = 1024
SAE_LR = 3e-4
DET_SAMPLES = 50_000
N_SEEDS = 5

# --- Distribution ---
high = 0.3
low = 1.28 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
dist = SparseUniform(N_FEATURES, firing_probs, device=DEVICE)


def normalize_W(tm):
    with torch.no_grad():
        tm.ae.W.data /= tm.ae.W.data.norm(dim=0, keepdim=True).clamp(min=1e-8)


# --- Train base models ---
print("Training Trained AE...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
tm_trained = ToyModel(
    distribution=dist,
    ae=TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1),
    device=DEVICE,
)
tm_trained.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

print("Training Trained AE w/ Unit Norms...")
gen2 = torch.Generator(DEVICE).manual_seed(SEED)
tm_unit_norm = ToyModel(
    distribution=dist,
    ae=TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen2),
    device=DEVICE,
    hooks=[normalize_W],
)
tm_unit_norm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

print("Training Constructed AE...")
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_constructed = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_constructed = ToyModel(distribution=dist, ae=ae_constructed, device=DEVICE)
tm_constructed.fit(N_EPOCHS_SYNTH, batch_size=BATCH_SIZE, verbose=True)


def make_data_fn(tm_ref, device):
    def data_fn(n):
        x = tm_ref.distribution.sample(n).to(device)
        return tm_ref.ae.encode(x)

    return data_fn


def eval_sae(sae, tm):
    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        D_enc = tm.ae.encode(eye)

        # Detection samples
        det_x = dist.sample(DET_SAMPLES).to(DEVICE)
        det_hidden = tm.ae.encode(det_x)
        det_z = sae.encode(det_hidden)
        det_recon = sae.decode(det_z)

        l0 = (det_z > 0).float().sum(dim=-1).mean().item()

        # R2
        ss_res = ((det_hidden - det_recon) ** 2).sum().item()
        ss_tot = ((det_hidden - det_hidden.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # MCC (decoder-based, |cosine| matching)
        W_ae = tm.W.detach()
        W_dec_t = sae.W_dec.detach().T
        W_norm = W_ae / W_ae.norm(dim=0, keepdim=True).clamp(min=1e-8)
        Wd_norm = W_dec_t / W_dec_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_mcc = (W_norm.T @ Wd_norm).cpu().numpy()
        cos_mcc_abs = np.abs(cos_mcc)
        mcc_fi, mcc_di = linear_sum_assignment(-cos_mcc_abs)
        mcc = float(cos_mcc_abs[mcc_fi, mcc_di].mean())

        # Decoder-matched F1
        D_enc_normed = D_enc / D_enc.norm(dim=1, keepdim=True)
        W_dec_normed = sae.W_dec.data / sae.W_dec.data.norm(dim=1, keepdim=True)
        cos_dec = (D_enc_normed @ W_dec_normed.T).abs().cpu().numpy()
        dec_fi, dec_di = linear_sum_assignment(-cos_dec)

        gt = det_x[:, dec_fi] > 0
        pred = det_z[:, dec_di] > 0
        tp = (gt & pred).float().sum(dim=0)
        fp = (~gt & pred).float().sum(dim=0)
        fn = (gt & ~pred).float().sum(dim=0)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        # Encoder-based matching
        sae_acts = sae.encode(D_enc).cpu().numpy()
        enc_fi, enc_di = linear_sum_assignment(-sae_acts)

        enc_gt = det_x[:, enc_fi] > 0
        enc_pred = det_z[:, enc_di] > 0
        enc_tp = (enc_gt & enc_pred).float().sum(dim=0)
        enc_fp = (~enc_gt & enc_pred).float().sum(dim=0)
        enc_fn = (enc_gt & ~enc_pred).float().sum(dim=0)
        enc_prec = enc_tp / (enc_tp + enc_fp + 1e-8)
        enc_rec = enc_tp / (enc_tp + enc_fn + 1e-8)
        enc_f1 = 2 * enc_prec * enc_rec / (enc_prec + enc_rec + 1e-8)
        enc_mcc = float(cos_mcc_abs[enc_fi, enc_di].mean())

    return {
        "l0": l0,
        "r2": r2,
        "mcc": mcc,
        "f1": f1.mean().item(),
        "precision": prec.mean().item(),
        "recall": rec.mean().item(),
        "enc_precision": enc_prec.mean().item(),
        "enc_recall": enc_rec.mean().item(),
        "enc_f1": enc_f1.mean().item(),
        "enc_mcc": enc_mcc,
    }


# --- Sweep ---
base_models = [
    ("Trained AE", tm_trained),
    ("Trained AE w/ Unit Norms", tm_unit_norm),
    ("Constructed AE", tm_constructed),
]

results = {name: [] for name, _ in base_models}

for name, tm in base_models:
    for seed_i in range(N_SEEDS):
        print(f"  {name} seed={seed_i}...", end=" ", flush=True)
        sae = SAESimple(
            n_latent=D_HIDDEN,
            n_dict=N_DICT,
            l1_coef=L1,
            device=DEVICE,
        ).to(DEVICE)
        sae.train_sae(
            data_fn=make_data_fn(tm, DEVICE),
            n_steps=SAE_STEPS,
            batch_size=SAE_BATCH,
            lr=SAE_LR,
        )
        m = eval_sae(sae, tm)
        results[name].append(m)
        print(
            f"F1={m['f1']:.4f}  L0={m['l0']:.1f}  EP={m['enc_precision']:.4f}  MCC={m['mcc']:.4f}"
        )

# --- Report ---
print(f"\n{'=' * 120}")
print(f"L1={L1}  N_SEEDS={N_SEEDS}  N_DICT={N_DICT}")
print(f"{'=' * 120}")
header = (
    f"{'Model':30s}  {'L0':>12s}  {'R²':>12s}  {'MCC':>12s}  {'F1':>12s}"
    f"  {'EncPrec':>12s}  {'EncF1':>12s}  {'EncMCC':>12s}"
)
print(header)
print("-" * len(header))
for name, seeds in results.items():
    for key in ["l0", "r2", "mcc", "f1", "enc_precision", "enc_f1", "enc_mcc"]:
        vals = [s[key] for s in seeds]

    # Print mean ± std
    def fmt(key):
        vals = [s[key] for s in seeds]
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

    print(
        f"{name:30s}  {fmt('l0'):>12s}  {fmt('r2'):>12s}  {fmt('mcc'):>12s}  {fmt('f1'):>12s}"
        f"  {fmt('enc_precision'):>12s}  {fmt('enc_f1'):>12s}  {fmt('enc_mcc'):>12s}"
    )

# Also print per-seed raw values
print(f"\n--- Raw per-seed values ---")
for name, seeds in results.items():
    print(f"\n{name}:")
    for i, m in enumerate(seeds):
        print(
            f"  seed {i}: L0={m['l0']:.1f}  R²={m['r2']:.4f}  MCC={m['mcc']:.4f}"
            f"  F1={m['f1']:.4f}  EP={m['enc_precision']:.4f}  EF1={m['enc_f1']:.4f}  EMCC={m['enc_mcc']:.4f}"
        )
