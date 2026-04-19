# %%
"""Sweep all 8 tweak_bench distributions: train AE, threshold-sweep for best F1, report top-half F1."""

import random

import numpy as np
import torch

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions import (
    DAGRandomWalkToRoot,
    PowerLawDigraph,
    SimplicialComplexDistribution,
    SphericalDistribution,
    TorusDistribution,
)
from occhio.distributions.correlated import CorrelatedPairs, HierarchicalPairs
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

# %%
# ── Config ───────────────────────────────────────────────────────────
DEVICE = "mps"
SEED = 42
N_FEATURES = 1296
D_HIDDEN = 200
N_EPOCHS = 50_000
BATCH_SIZE = 512
N_TEST = 100_000
N_THRESHOLDS = 101


# ── Distribution factories ───────────────────────────────────────────
def make_sparse_uniform() -> SparseUniform:
    high = 0.3
    low = 1.28 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    return SparseUniform(N_FEATURES, p_active=firing_probs, device=DEVICE)


def make_correlated_pairs() -> CorrelatedPairs:
    np.random.seed(8)
    high = 0.5
    low = 1.22 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    corrs = 0.5 + 0.5 * np.random.random(N_FEATURES)
    return CorrelatedPairs(
        N_FEATURES, p_active=firing_probs, p_individual=corrs, device=DEVICE
    )


def make_hierarchical_pairs() -> HierarchicalPairs:
    np.random.seed(8)
    high = 0.45
    low = 1.3 / N_FEATURES
    alpha = np.log(high / low) / np.log(N_FEATURES)
    firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
    betas = np.random.random(N_FEATURES)
    return HierarchicalPairs(
        N_FEATURES, p_active=firing_probs, p_follow=0.6, beta=betas, device=DEVICE
    )


def make_digraph() -> PowerLawDigraph:
    return PowerLawDigraph(
        n_features=N_FEATURES,
        p_active=3.3 / N_FEATURES,
        alpha=1,
        p_edge=4.1 / N_FEATURES,
        p_child=(0.1, 0.4),
    )


def make_rwtr() -> DAGRandomWalkToRoot:
    return DAGRandomWalkToRoot(
        n_features=N_FEATURES, p_edge=50 / N_FEATURES, beta=0.8, shrinking=True
    )


def make_simplicial_complex() -> SimplicialComplexDistribution:
    random.seed(SEED)
    face_dim = 4
    n_faces = 4 * (N_FEATURES // (face_dim + 1))
    all_verts = list(range(N_FEATURES))
    random.shuffle(all_verts)
    face_size = face_dim + 1
    covering_faces: set[tuple[int, ...]] = set()
    for i in range(0, N_FEATURES, face_size):
        chunk = all_verts[i : i + face_size]
        if len(chunk) < face_size:
            remaining = [v for v in all_verts if v not in chunk]
            chunk += random.sample(remaining, face_size - len(chunk))
        covering_faces.add(tuple(sorted(chunk)))
    while len(covering_faces) < n_faces:
        covering_faces.add(tuple(sorted(random.sample(range(N_FEATURES), face_size))))
    faces = list(covering_faces)[:n_faces]
    return SimplicialComplexDistribution(
        n_vertices=N_FEATURES,
        faces=faces,
        sampling_mode="sparse",
        p_active=1 / n_faces,
    )


def make_spherical() -> SphericalDistribution:
    return SphericalDistribution(
        n_features=N_FEATURES,
        length_scale=0.245,
        manifold_dim=4,
        magnitude_range=(0.5, 1.0),
    )


def make_torus() -> TorusDistribution:
    return TorusDistribution(
        n_features=N_FEATURES,
        length_scale=0.669,
        torus_dim=4,
        magnitude_range=(0.5, 1.0),
    )


DISTRIBUTIONS: dict[str, callable] = {
    "SparseUniform": make_sparse_uniform,
    "CorrelatedPairs": make_correlated_pairs,
    "HierarchicalPairs": make_hierarchical_pairs,
    "PowerLawDigraph": make_digraph,
    "DAGRandomWalkToRoot": make_rwtr,
    "SimplicialComplex": make_simplicial_complex,
    "Spherical": make_spherical,
    "Torus": make_torus,
}


# ── Helpers ──────────────────────────────────────────────────────────
def best_threshold_f1(
    x: torch.Tensor, xhat: torch.Tensor, n_thresholds: int = N_THRESHOLDS
) -> tuple[float, torch.Tensor, float]:
    """Sweep thresholds on xhat, return (best_threshold, per_feature_f1, micro_f1)."""
    gt_active = x > 0
    thresholds = torch.linspace(0, 1, n_thresholds, device=x.device)

    best_f1_micro = -1.0
    best_threshold = 0.0
    best_per_feature_f1 = torch.zeros(x.shape[1], device=x.device)

    for t in thresholds:
        pred_active = xhat > t

        tp = (gt_active & pred_active).float().sum(dim=0)
        fp = (~gt_active & pred_active).float().sum(dim=0)
        fn = (gt_active & ~pred_active).float().sum(dim=0)

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        tp_tot = tp.sum()
        fp_tot = fp.sum()
        fn_tot = fn.sum()
        prec_micro = (tp_tot / (tp_tot + fp_tot + 1e-8)).item()
        rec_micro = (tp_tot / (tp_tot + fn_tot + 1e-8)).item()
        f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + 1e-8)

        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro
            best_threshold = t.item()
            best_per_feature_f1 = f1

    return best_threshold, best_per_feature_f1, best_f1_micro


def mean_max_cosine_sim(W: torch.Tensor) -> float:
    """Compute 1/N sum_i max_{j!=i} W_i^T W_j on column-normalised W."""
    W_norm = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos = W_norm.T @ W_norm  # (N, N)
    cos.fill_diagonal_(-float("inf"))
    return cos.max(dim=1).values.mean().item()


def top_half_mean_f1(per_feature_f1: torch.Tensor) -> float:
    """Sort features by F1 descending, return mean of the top half."""
    sorted_f1, _ = per_feature_f1.sort(descending=True)
    top_half = sorted_f1[: len(sorted_f1) // 2]
    return top_half.mean().item()


# %%
# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results: dict[str, dict] = {}

    for name, make_dist in DISTRIBUTIONS.items():
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        dist = make_dist()

        # Average L0
        samples = dist.sample(N_TEST)
        if isinstance(samples, tuple):
            samples = samples[0]
        mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
        print(f"  Average L0: {mean_l0:.2f}")

        # Train
        gen = torch.Generator(DEVICE).manual_seed(SEED)
        ae = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen)
        tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)
        tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

        # Evaluate
        with torch.no_grad():
            mmcs = mean_max_cosine_sim(tm.W)

            test_x = dist.sample(N_TEST)
            if isinstance(test_x, tuple):
                test_x = test_x[0]
            test_x = test_x.to(DEVICE)
            test_xhat = tm.ae.decode(tm.ae.encode(test_x))

            threshold, per_feature_f1, f1_micro = best_threshold_f1(test_x, test_xhat)
            top_half_f1 = top_half_mean_f1(per_feature_f1)

        results[name] = {
            "threshold": threshold,
            "f1_micro": f1_micro,
            "f1_macro": per_feature_f1.mean().item(),
            "top_half_f1": top_half_f1,
            "mean_l0": mean_l0,
            "mmcs": mmcs,
        }

        print(
            f"  threshold={threshold:.2f}  "
            f"F1(micro)={f1_micro:.4f}  F1(macro)={per_feature_f1.mean().item():.4f}  "
            f"F1(top-half)={top_half_f1:.4f}  MMCS={mmcs:.4f}"
        )

    # %%
    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print(
        f"  {'Distribution':<25} {'Thresh':>6} {'F1 micro':>9} "
        f"{'F1 macro':>9} {'F1 top½':>9} {'MMCS':>7}"
    )
    print(f"{'=' * 76}")
    for name, r in results.items():
        print(
            f"  {name:<25} {r['threshold']:>6.2f} {r['f1_micro']:>9.4f} "
            f"{r['f1_macro']:>9.4f} {r['top_half_f1']:>9.4f} {r['mmcs']:>7.4f}"
        )
    print(f"{'=' * 76}")

# %%
