# %%
"""
Inspect uploaded distributions from HuggingFace.

Downloads each distribution's samples.safetensors and prints thorough
diagnostics over the first 50 samples to verify data integrity.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "kaushikreddyxyz/occhio-distributions"

DISTRIBUTIONS = [
    "sparse_uniform",
    "correlated_pairs",
    "hierarchical_pairs",
    "power_law_digraph",
    "dag_random_walk",
    "simplicial_complex",
    "spherical",
    "torus",
]


def inspect(name: str) -> None:
    path = hf_hub_download(
        REPO_ID,
        f"{name}/samples/samples.safetensors",
    )
    data = load_file(path)
    samples = data["samples"]
    n_samples, n_feat = samples.shape

    # Global stats
    nan_total = samples.isnan().sum().item()
    nonzero_rows_total = (samples.sum(dim=-1) != 0).sum().item()
    active_per_row = (samples > 0).float().sum(dim=-1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  shape: {list(samples.shape)}, dtype: {samples.dtype}")
    print(f"  NaN count (full dataset): {nan_total}")
    print(f"  nonzero rows (full dataset): {nonzero_rows_total}/{n_samples}")
    # E[L0]: expected number of active features per sample (target ≈ 5.0)
    e_l0 = active_per_row.mean().item()
    zero_frac = 1.0 - nonzero_rows_total / n_samples

    print(
        f"  E[L0] = {e_l0:.2f}  (target ≈ 5.0) | "
        f"median={active_per_row.median():.0f}, max={active_per_row.max():.0f}"
    )
    print(f"  zero rows: {zero_frac:.1%}")
    print(f"  value range: [{samples.min().item():.6f}, {samples.max().item():.6f}]")

    # Per-feature activation rate (what fraction of samples have feature > 0)
    feat_activation_rate = (samples > 0).float().mean(dim=0)
    top_feats = feat_activation_rate.topk(10)
    print("  top 10 most active features (activation rate):")
    for idx, rate in zip(top_feats.indices.tolist(), top_feats.values.tolist()):
        print(f"    feature {idx}: {rate:.4f}")

    # Detailed look at first 50 samples
    head = samples[:50]
    print("\n  --- First 50 samples ---")
    for i in range(50):
        row = head[i]
        nz = torch.where(row > 0)[0]
        if len(nz) == 0:
            print(f"  sample[{i:>2d}]: all zeros")
        else:
            vals = row[nz]
            print(
                f"  sample[{i:>2d}]: {len(nz):>4d} active | "
                f"sum={row.sum():.4f} | "
                f"mean={vals.mean():.4f} | "
                f"max={vals.max():.4f} | "
                f"indices={nz[:10].tolist()}{'...' if len(nz) > 10 else ''}"
            )


# %%
for dist_name in DISTRIBUTIONS:
    inspect(dist_name)

# %%
