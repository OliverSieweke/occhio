# %%
"""
Validate trained weights from HuggingFace.

For each distribution:
1. Download weights + samples from HF
2. Load weights into a TiedLinearRelu
3. Inspect weight matrix properties (norms, rank, NaN)
4. Run reconstruction eval on a batch of the published samples
"""

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from occhio.autoencoder import TiedLinearRelu

DATASET_REPO = "kaushikreddyxyz/occhio-distributions"
MODEL_REPO = "kaushikreddyxyz/occhio-models"
N_HIDDEN = 200
DEVICE = "mps"

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


def validate(name: str) -> None:
    # --- Download ---
    weights_path = hf_hub_download(
        MODEL_REPO, f"{name}/weights/weights.safetensors", repo_type="model"
    )
    samples_path = hf_hub_download(
        DATASET_REPO,
        f"{name}/samples/samples.safetensors",
        repo_type="dataset",
    )

    # --- Load samples (first 10K on CPU for stats) ---
    samples_data = load_file(samples_path)
    all_samples = samples_data["samples"]
    n_samples, n_feat = all_samples.shape

    # --- Load AE ---
    ae = TiedLinearRelu(n_feat, N_HIDDEN, device="cpu")
    ae.load_weights(weights_path)
    ae = ae.to(DEVICE)
    ae.eval()

    # --- Weight inspection ---
    W = ae.W.detach()  # (n_hidden, n_features)
    b = ae.b.detach()  # (n_features,)

    col_norms = W.norm(dim=0)  # norm of each feature column
    row_norms = W.norm(dim=1)  # norm of each hidden row
    svd = torch.linalg.svdvals(W.float().cpu())  # SVD not supported on MPS

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  W shape: {list(W.shape)}, dtype: {W.dtype}")
    print(f"  W NaN: {W.isnan().sum().item()}, Inf: {W.isinf().sum().item()}")
    print(f"  W value range: [{W.min().item():.6f}, {W.max().item():.6f}]")
    print(
        f"  Column norms (per-feature): min={col_norms.min():.4f}, "
        f"max={col_norms.max():.4f}, mean={col_norms.mean():.4f}"
    )
    print(
        f"  Row norms (per-hidden): min={row_norms.min():.4f}, "
        f"max={row_norms.max():.4f}, mean={row_norms.mean():.4f}"
    )
    print(f"  Top 5 singular values: {svd[:5].tolist()}")
    print(
        f"  Effective rank (sv > 0.01*max): "
        f"{(svd > 0.01 * svd[0]).sum().item()}/{N_HIDDEN}"
    )
    print(
        f"  b range: [{b.min().item():.6f}, {b.max().item():.6f}], "
        f"mean={b.mean().item():.6f}"
    )

    # --- Reconstruction eval on 10K samples ---
    batch = all_samples[:10_000].to(DEVICE)
    with torch.no_grad():
        encoded = ae.encode(batch)
        decoded = ae.decode(encoded)
        mse = (batch - decoded).pow(2).mean().item()
        # Per-sample MSE
        per_sample_mse = (batch - decoded).pow(2).mean(dim=-1)
        # Only evaluate on nonzero rows (skip all-zero samples)
        nonzero_mask = batch.sum(dim=-1) > 0
        n_nonzero = nonzero_mask.sum().item()

        if n_nonzero > 0:
            nz_mse = per_sample_mse[nonzero_mask].mean().item()
            nz_mse_std = per_sample_mse[nonzero_mask].std().item()
            # Cosine similarity on nonzero rows
            cos = torch.nn.functional.cosine_similarity(
                batch[nonzero_mask], decoded[nonzero_mask], dim=-1
            )
            cos_mean = cos.mean().item()
            cos_min = cos.min().item()
        else:
            nz_mse = nz_mse_std = cos_mean = cos_min = float("nan")

        # L0 of encoded (how many hidden dims active)
        encoded_l0 = (encoded.abs() > 1e-8).float().sum(dim=-1).mean().item()

    print("\n  --- Reconstruction (10K samples) ---")
    print(f"  Overall MSE: {mse:.6f}")
    print(
        f"  Nonzero-only MSE: {nz_mse:.6f} +/- {nz_mse_std:.6f} "
        f"({n_nonzero}/{batch.shape[0]} nonzero)"
    )
    print(f"  Cosine similarity (nonzero): mean={cos_mean:.4f}, min={cos_min:.4f}")
    print(f"  Encoded L0 (active hidden dims): {encoded_l0:.1f}/{N_HIDDEN}")

    # Flag obvious problems
    issues = []
    if W.isnan().any():
        issues.append("W contains NaN")
    if mse > 1.0:
        issues.append(f"MSE suspiciously high ({mse:.4f})")
    if cos_mean < 0.5 and n_nonzero > 100:
        issues.append(f"Cosine similarity very low ({cos_mean:.4f})")
    if svd[0] < 0.01:
        issues.append("All singular values near zero — dead model")
    if encoded_l0 < 1.0:
        issues.append("Encoded L0 < 1 — model may be collapsed")

    if issues:
        print(f"\n  *** ISSUES: {'; '.join(issues)} ***")
    else:
        print("\n  OK")

    del batch, encoded, decoded, all_samples
    torch.mps.empty_cache()

    # Clean up downloaded HF cache files
    for path in (weights_path, samples_path):
        try:
            os.remove(path)
        except OSError:
            pass


# %%
for dist_name in DISTRIBUTIONS:
    validate(dist_name)

# %%
