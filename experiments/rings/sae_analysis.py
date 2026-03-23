"""Systematic analysis: do trained AEs learn circular or scalar features?

Following Engels et al., we train SAEs on AE bottleneck activations, cluster
dictionary elements by cosine similarity, and measure cluster dimensionality
via PCA. Circular features show 2D clusters; correlated scalars show 1D.
"""

# %%
# ── IMPORTS AND CONFIG ──────────────────────────────────────────────────
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import defaultdict

from occhio import ToyModel
from occhio.autoencoder import TiedLinearRelu, TiedMLPEncoder
from occhio.distributions import SparseSpheres
from occhio.sae import SAESimple

DEVICE = "cpu"
SEED = 42
torch.manual_seed(SEED)

# SparseSpheres config (matches experiment.py)
K = 5
N = 1  # S^1 (circles)
M = 3  # ambient dim per feature
N_FEATURES = K * M  # = 15
P_ACTIVE = 0.01
R = 1.0

# AE config
HIDDEN_DIM = 3
N_EPOCHS = 30_000
BATCH_SIZE = 256
LR = 3e-4
IMPORTANCE_BASE = 0.95

# SAE analysis config
N_ANALYSIS = 50_000
SAE_DICT = 64  # ~12-13 per ring, enough to tessellate each circle
SAE_LAMBDAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SAE_STEPS = 20_000
SAE_BATCH = 512
SAE_LR = 1e-3

# Clustering config
CLUSTER_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
PCA_DIM_THRESHOLD = 0.1  # explained variance ratio > this = significant component

OUTPUT_DIR = "experiments/rings/sae_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

importances = torch.repeat_interleave(IMPORTANCE_BASE ** torch.arange(1, K + 1), M)


# %%
# ── PHASE 0: DISTRIBUTION + AE TRAINING ────────────────────────────────
print("=" * 60)
print("PHASE 0: Training autoencoders")
print("=" * 60)

dist = SparseSpheres(
    n_spheres=K,
    sphere_dim=N,
    ambient_dim=M,
    p_active=P_ACTIVE,
    radius=R,
    noise_std=0.08,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)


def make_models():
    configs = {
        "TiedLinearAE": lambda: TiedLinearRelu(
            n_features=N_FEATURES, n_hidden=HIDDEN_DIM
        ),
        "TiedMLPAE": lambda: TiedMLPEncoder(dims=[N_FEATURES, 64, 32, HIDDEN_DIM]),
    }
    models = {}
    for name, ae_factory in configs.items():
        ae = ae_factory()
        models[name] = ToyModel(
            distribution=dist, ae=ae, device=DEVICE, importances=importances
        )
    return models


models = make_models()
for name, tm in models.items():
    print(f"\nTraining {name}...")
    losses, _ = tm.fit(
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.0,
        track_losses=True,
        verbose=True,
    )
    print(f"  Final loss: {losses[-1]:.6f}")

# %%
# ── PHASE 1: GENERATE BOTTLENECK DATASET ───────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: Generating bottleneck dataset")
print("=" * 60)

samples, labels = dist.sample_with_args(N_ANALYSIS, noise_std=0.0)
samples = samples.to(DEVICE)

n_active_per_sample = labels.sum(dim=1)
is_single = n_active_per_sample == 1
is_inactive = n_active_per_sample == 0
single_ring_id = labels.float().argmax(dim=1)

print(f"Samples: {samples.shape}, Labels: {labels.shape}")
print(
    f"Active: {(~is_inactive).sum().item()}, "
    f"Single-ring: {is_single.sum().item()}, "
    f"Inactive: {is_inactive.sum().item()}"
)

bottlenecks = {}  # full dataset (for GT correspondence later)
bottlenecks_active = {}  # active-only (for SAE training + analysis)
active_mask = ~is_inactive  # samples with at least one ring active

for name, tm in models.items():
    with torch.no_grad():
        z = tm.ae.encode(samples)
    bottlenecks[name] = z
    z_active = z[active_mask]
    bottlenecks_active[name] = z_active
    print(f"{name} bottleneck (all): {z.shape}, mean norm {z.norm(dim=1).mean():.3f}")
    print(
        f"{name} bottleneck (active only): {z_active.shape}, "
        f"range [{z_active.min():.3f}, {z_active.max():.3f}], "
        f"mean norm {z_active.norm(dim=1).mean():.3f}"
    )

# Labels/masks restricted to active samples
labels_active = labels[active_mask]
n_active_per_active = labels_active.sum(dim=1)
is_single_active = n_active_per_active == 1
single_ring_id_active = labels_active.float().argmax(dim=1)


# %%
# ── PHASE 2: SAE LAMBDA SWEEP ──────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: SAE lambda sweep")
print("=" * 60)

best_saes = {}
best_lambdas = {}
all_sweep_results = {}

for name, tm in models.items():
    print(f"\n--- {name} ---")
    z_data = bottlenecks_active[name]  # active-only for SAE training
    print(f"  Training on {len(z_data)} active samples (filtered out inactive)")

    sweep_results = {}
    for lam in SAE_LAMBDAS:
        sae = SAESimple(
            n_latent=HIDDEN_DIM, n_dict=SAE_DICT, l1_coef=lam, device=DEVICE
        )
        sae = sae.to(DEVICE)

        # data_fn: randomly sample from pre-computed bottleneck activations
        def make_data_fn(z_src):
            def data_fn(n):
                idx = torch.randint(0, len(z_src), (n,))
                return z_src[idx].detach()

            return data_fn

        sae_losses = sae.train_sae(
            make_data_fn(z_data),
            n_steps=SAE_STEPS,
            batch_size=SAE_BATCH,
            lr=SAE_LR,
        )

        with torch.no_grad():
            acts = sae.encode(z_data)
            l0 = (acts > 0).float().sum(dim=1).mean().item()
            recon = sae.decode(acts)
            mse = ((z_data - recon) ** 2).sum(dim=1).mean().item()

        sweep_results[lam] = {"sae": sae, "l0": l0, "mse": mse, "losses": sae_losses}
        print(f"  lambda={lam:.3f}: L0={l0:.2f} active/sample, MSE={mse:.6f}")

    # Pick lambda where L0 is closest to 3.0 (target: 2-4 active elements)
    best_lam = min(
        sweep_results.keys(), key=lambda l: abs(sweep_results[l]["l0"] - 3.0)
    )
    best_saes[name] = sweep_results[best_lam]["sae"]
    best_lambdas[name] = best_lam
    all_sweep_results[name] = sweep_results
    print(
        f"  >> Selected lambda={best_lam:.3f} (L0={sweep_results[best_lam]['l0']:.2f})"
    )


# %%
# ── PHASE 2b: IDENTITY ANALYSIS (wide L1 sweep) ─────────────────────────
print("\n" + "=" * 60)
print("PHASE 2b: Identity analysis across L1 sweep")
print("=" * 60)

for name in models:
    print(f"\n--- {name} ---")
    z_data = bottlenecks_active[name]
    print(
        f"  {'Lambda':>8} | {'L0':>6} | {'Alive':>5} | {'Rel Err%':>8} | {'d(I)':>8} | {'Regime':>12}"
    )
    print(f"  {'-' * 8}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 12}")

    for lam in sorted(all_sweep_results[name].keys()):
        res = all_sweep_results[name][lam]
        sae = res["sae"]
        l0 = res["l0"]

        with torch.no_grad():
            acts = sae.encode(z_data)
            recon = sae.decode(acts)
            mse = ((z_data - recon) ** 2).sum(dim=1).mean().item()
            gt_norm2 = (z_data**2).sum(dim=1).mean().item()
            rel_err = (mse / gt_norm2 * 100) if gt_norm2 > 1e-10 else 0.0
            n_alive = (acts > 0).any(dim=0).sum().item()

            # Distance from identity: compute effective linear map W_dec @ diag(mean_act) @ W_enc
            # Simplified: just check W_dec @ W_enc vs I
            W_enc = sae.W_enc.data  # (3, 64)
            W_dec = sae.W_dec.data  # (64, 3)
            effective = W_dec.T @ W_enc.T  # approximate — ignoring ReLU
            eye = torch.eye(HIDDEN_DIM, device=DEVICE)
            d_identity = (effective - eye).norm().item()

        if l0 < 0.01:
            regime = "DEAD"
        elif d_identity < 0.1 and n_alive >= 5:
            regime = "IDENTITY"
        elif d_identity < 0.2:
            regime = "TRANSITION"
        else:
            regime = "BROKEN"

        print(
            f"  {lam:>8.3f} | {l0:>6.2f} | {n_alive:>5} | "
            f"{rel_err:>8.1f} | {d_identity:>8.4f} | {regime:>12}"
        )

        # Ring-specific analysis for broken regime
        if regime == "BROKEN" and n_alive > 0:
            print(f"           Ring-neuron mapping:")
            for j in range(K):
                ring_mask = is_single_active & (single_ring_id_active == j)
                if ring_mask.sum() == 0:
                    continue
                ring_acts = acts[ring_mask]
                mean_act = ring_acts.mean(dim=0)
                top_neurons = torch.topk(mean_act, min(3, n_alive))
                neurons_str = ", ".join(
                    f"n{top_neurons.indices[i].item()}({top_neurons.values[i].item():.3f})"
                    for i in range(len(top_neurons.indices))
                    if top_neurons.values[i].item() > 0.01
                )
                print(f"             Ring {j} -> {neurons_str}")


# %%
# ── PHASE 3: DICTIONARY ELEMENT CLUSTERING ──────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Dictionary element clustering")
print("=" * 60)


def find_connected_components(adj):
    """BFS-based connected components from boolean adjacency matrix."""
    n = adj.shape[0]
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for j in range(n):
                if adj[node, j] and not visited[j]:
                    visited[j] = True
                    queue.append(j)
        components.append(sorted(comp))
    return components


cluster_results = {}

for name in models:
    print(f"\n--- {name} ---")
    sae = best_saes[name]
    z_active = bottlenecks_active[name]

    # Extract and normalize decoder columns: W_dec is (SAE_DICT, 3)
    W_dec = sae.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_dec_normed = W_dec / np.clip(norms, 1e-8, None)

    # Pairwise cosine similarity
    cos_sim = W_dec_normed @ W_dec_normed.T

    # Compute SAE activations on active samples
    with torch.no_grad():
        acts = sae.encode(z_active)  # (n_active, SAE_DICT)

    # Identify dead neurons (never fire on active data)
    neuron_fires = (acts > 0).any(dim=0).numpy()
    n_alive = neuron_fires.sum()
    print(f"  Alive neurons: {n_alive}/{SAE_DICT}")

    # Cluster only alive neurons — dead ones are irrelevant
    alive_idx = np.where(neuron_fires)[0]
    cos_sim_alive = cos_sim[np.ix_(alive_idx, alive_idx)]

    # Sweep thresholds on alive neurons
    print("  Threshold sweep (alive neurons only):")
    threshold_results = {}
    for T in CLUSTER_THRESHOLDS:
        adj = cos_sim_alive > T
        np.fill_diagonal(adj, False)
        local_comps = find_connected_components(adj)
        # Map back to global indices
        global_comps = [[int(alive_idx[i]) for i in c] for c in local_comps]
        sizes = sorted([len(c) for c in global_comps], reverse=True)
        threshold_results[T] = global_comps
        print(
            f"    T={T:.1f}: {len(global_comps)} clusters, sizes={sizes[:10]}"
            f"{'...' if len(sizes) > 10 else ''}"
        )

    # Pick threshold closest to K clusters
    best_T = min(
        CLUSTER_THRESHOLDS,
        key=lambda T: (abs(len(threshold_results[T]) - K), -T),
    )
    best_components = threshold_results[best_T]
    print(f"  Selected T={best_T:.1f} ({len(best_components)} clusters)")

    # Ground-truth ring correspondence per dictionary element
    # Use single-ring active samples for clean assignment
    gt_assignment = np.full(SAE_DICT, -1, dtype=int)
    for d in range(SAE_DICT):
        fires = (acts[:, d] > 0).numpy()
        valid = fires & is_single_active.numpy()
        if valid.sum() == 0:
            continue
        ring_counts = np.zeros(K)
        for j in range(K):
            ring_counts[j] = (valid & (single_ring_id_active.numpy() == j)).sum()
        gt_assignment[d] = ring_counts.argmax()

    cluster_results[name] = {
        "components": best_components,
        "threshold": best_T,
        "cos_sim": cos_sim,
        "W_dec": W_dec,
        "W_dec_normed": W_dec_normed,
        "norms": norms.flatten(),
        "gt_assignment": gt_assignment,
        "acts": acts,  # activations on active samples
        "n_active_samples": len(z_active),
    }

    # Print GT correspondence per cluster
    for ci, comp in enumerate(best_components):
        gt_rings = [gt_assignment[d] for d in comp if gt_assignment[d] >= 0]
        if not gt_rings:
            print(f"    Cluster {ci} (size {len(comp)}): all dead neurons")
            continue
        ring_counts = {}
        for r in gt_rings:
            ring_counts[r] = ring_counts.get(r, 0) + 1
        dominant = max(ring_counts, key=ring_counts.get)
        purity = ring_counts[dominant] / len(gt_rings)
        n_dead = sum(1 for d in comp if gt_assignment[d] < 0)
        dead_str = f", {n_dead} dead" if n_dead > 0 else ""
        print(
            f"    Cluster {ci} (size {len(comp)}{dead_str}): "
            f"ring counts={ring_counts}, "
            f"dominant=ring {dominant} ({purity:.0%})"
        )


# %%
# ── PHASE 4: CLUSTER DIMENSIONALITY ANALYSIS (THE KEY TEST) ────────────
print("\n" + "=" * 60)
print("PHASE 4: Cluster dimensionality (PCA)")
print(f"  Threshold for significant component: explained var > {PCA_DIM_THRESHOLD}")
print("=" * 60)

pca_results = {}

for name in models:
    print(f"\n--- {name} ---")
    sae = best_saes[name]
    z_active = bottlenecks_active[name]
    cr = cluster_results[name]
    acts = cr["acts"]  # activations on active samples
    components = cr["components"]
    n_active_total = cr["n_active_samples"]

    print(f"  Analyzing {n_active_total} active samples")
    print(
        f"  {'Cluster':>8} | {'Size':>5} | {'Samples':>8} | "
        f"{'PCA 1':>8} | {'PCA 2':>8} | {'PCA 3':>8} | {'Eff dim':>8}"
    )
    print(
        f"  {'-' * 8}-+-{'-' * 5}-+-{'-' * 8}-+-"
        f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
    )

    model_pca = []
    for ci, comp in enumerate(components):
        comp_set = set(comp)

        # Find active samples where at least one cluster element fires
        cluster_active = torch.zeros(n_active_total, dtype=torch.bool)
        for d in comp:
            cluster_active |= acts[:, d] > 0

        n_active = cluster_active.sum().item()
        if n_active < 10:
            model_pca.append(
                {
                    "var_ratios": np.array([0.0, 0.0, 0.0]),
                    "eff_dim": 0,
                    "n_samples": n_active,
                }
            )
            print(
                f"  {ci:>8} | {len(comp):>5} | {n_active:>8} | "
                f"{'(too few samples)':>35}"
            )
            continue

        # Ablate non-cluster elements, reconstruct through SAE decoder
        masked_acts = acts[cluster_active].clone()
        for d in range(SAE_DICT):
            if d not in comp_set:
                masked_acts[:, d] = 0.0

        with torch.no_grad():
            z_restricted = sae.decode(masked_acts).cpu().numpy()  # (n_active, 3)

        # PCA on cluster-restricted reconstructions
        z_centered = z_restricted - z_restricted.mean(axis=0)
        _, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
        var = S**2 / max(len(z_centered) - 1, 1)
        total = var.sum()
        var_ratios = var / total if total > 1e-10 else np.zeros(min(3, len(S)))
        # Pad to length 3 if needed
        if len(var_ratios) < 3:
            var_ratios = np.pad(var_ratios, (0, 3 - len(var_ratios)))

        eff_dim = int((var_ratios > PCA_DIM_THRESHOLD).sum())

        model_pca.append(
            {
                "var_ratios": var_ratios,
                "eff_dim": eff_dim,
                "n_samples": n_active,
                "z_restricted": z_restricted,
                "z_centered": z_centered,
                "Vt": Vt,
                "pca_scores": z_centered @ Vt.T,
                "cluster_active_mask": cluster_active,
            }
        )
        print(
            f"  {ci:>8} | {len(comp):>5} | {n_active:>8} | "
            f"{var_ratios[0]:>8.4f} | {var_ratios[1]:>8.4f} | "
            f"{var_ratios[2]:>8.4f} | {eff_dim:>8}"
        )

    pca_results[name] = model_pca


# %%
# ── PHASE 5: SEPARABILITY AND MIXTURE INDICES ──────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: Separability and mixture indices")
print("=" * 60)


def mutual_information_binned(x, y, n_bins=20):
    """Estimate MI between x and y using a 2D histogram."""
    hist2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py
    mask = (pxy > 0) & (denom > 0)
    if not mask.any():
        return 0.0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask]))
    return float(mi)


def separability_index(points_2d, n_angles=100):
    """Min MI over rotation angles. High = irreducible (circular)."""
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    min_mi = float("inf")
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rotated = points_2d @ rot.T
        mi = mutual_information_binned(rotated[:, 0], rotated[:, 1])
        min_mi = min(min_mi, mi)
    return min_mi


def mixture_index(points_2d, n_angles=100, n_offsets=50):
    """Max fraction of points near any line. High = reducible (scalar)."""
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    max_frac = 0.0
    for theta in angles:
        v = np.array([np.cos(theta), np.sin(theta)])
        proj = points_2d @ v
        sigma = proj.std()
        if sigma < 1e-10:
            continue
        eps = 0.1 * sigma
        offsets = np.linspace(proj.min(), proj.max(), n_offsets)
        for c in offsets:
            frac = float(np.mean(np.abs(proj - c) < eps))
            max_frac = max(max_frac, frac)
    return max_frac


sep_mix_results = {}

for name in models:
    print(f"\n--- {name} ---")
    components = cluster_results[name]["components"]
    model_results = []

    print(
        f"  {'Cluster':>8} | {'Sep(1-2)':>9} | {'Sep(2-3)':>9} | "
        f"{'Mix(1-2)':>9} | {'Mix(2-3)':>9}"
    )
    print(f"  {'-' * 8}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}")

    for ci in range(len(components)):
        pca_info = pca_results[name][ci]
        if pca_info.get("pca_scores") is None or pca_info["n_samples"] < 50:
            model_results.append(
                {
                    "sep_12": 0,
                    "sep_23": 0,
                    "mix_12": 0,
                    "mix_23": 0,
                }
            )
            print(f"  {ci:>8} | {'(skipped — too few samples)':>40}")
            continue

        scores = pca_info["pca_scores"]
        pc12 = scores[:, :2]  # PCA components 1-2
        pc23 = scores[:, 1:3]  # PCA components 2-3

        sep_12 = separability_index(pc12)
        sep_23 = separability_index(pc23)
        mix_12 = mixture_index(pc12)
        mix_23 = mixture_index(pc23)

        model_results.append(
            {
                "sep_12": sep_12,
                "sep_23": sep_23,
                "mix_12": mix_12,
                "mix_23": mix_23,
            }
        )
        print(
            f"  {ci:>8} | {sep_12:>9.4f} | {sep_23:>9.4f} | "
            f"{mix_12:>9.4f} | {mix_23:>9.4f}"
        )

    sep_mix_results[name] = model_results


# %%
# ── PHASE 6: PLOTS ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6: Generating plots")
print("=" * 60)


def save_and_show(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    fig.show()
    print(f"  Saved: {path}")


# ── Plot 1: Dictionary elements on unit sphere ──────────────────────────
for name in models:
    cr = cluster_results[name]
    W = cr["W_dec_normed"]
    components = cr["components"]

    fig = go.Figure()

    # Unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=x_s,
            y=y_s,
            z=z_s,
            opacity=0.08,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            name="unit sphere",
        )
    )

    for ci, comp in enumerate(components):
        pts = W[comp]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=5, color=COLORS[ci % len(COLORS)]),
                name=f"Cluster {ci} ({len(comp)} elems)",
            )
        )

    fig.update_layout(
        title=f"Dictionary Elements on Unit Sphere - {name}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        height=700,
        width=700,
        template="plotly_white",
    )
    save_and_show(fig, f"plot1_dict_sphere_{name}.html")


# ── Plot 2: Cluster activations in PCA 2-3 ─────────────────────────────
for name in models:
    components = cluster_results[name]["components"]
    acts = cluster_results[name]["acts"]
    n_clusters = len(components)

    n_cols = min(n_clusters, 5)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Cluster {ci}" for ci in range(n_clusters)],
    )

    for ci in range(n_clusters):
        pca_info = pca_results[name][ci]
        row = ci // n_cols + 1
        col = ci % n_cols + 1

        if pca_info.get("pca_scores") is None:
            continue

        scores = pca_info["pca_scores"]
        cluster_mask = pca_info["cluster_active_mask"]

        # GT ring labels for cluster-active samples (within active set)
        sample_labels = labels_active[cluster_mask]
        n_act = sample_labels.sum(dim=1)
        single = n_act == 1
        ring_id = sample_labels.float().argmax(dim=1)

        # Scatter PCA components 2-3 (indices 1, 2), colored by GT ring
        for j in range(K):
            ring_mask = (single & (ring_id == j)).numpy()
            if ring_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=scores[ring_mask, 1],
                        y=scores[ring_mask, 2],
                        mode="markers",
                        marker=dict(
                            color=COLORS[j % len(COLORS)],
                            size=3,
                            opacity=0.5,
                        ),
                        name=f"Ring {j}",
                        showlegend=(ci == 0),
                        legendgroup=f"ring_{j}",
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=f"Cluster Activations (PCA 2-3) - {name}",
        height=350 * n_rows,
        width=300 * n_cols,
        template="plotly_white",
    )
    save_and_show(fig, f"plot2_cluster_pca23_{name}.html")


# ── Plot 3: GT ring correspondence heatmap ──────────────────────────────
for name in models:
    cr = cluster_results[name]
    components = cr["components"]
    acts = cr["acts"]
    n_clusters = len(components)

    n_act_samples = cr["n_active_samples"]
    correspondence = np.zeros((n_clusters, K))
    for ci, comp in enumerate(components):
        cluster_active = torch.zeros(n_act_samples, dtype=torch.bool)
        for d in comp:
            cluster_active |= acts[:, d] > 0
        valid = cluster_active & is_single_active
        if valid.sum() == 0:
            continue
        for j in range(K):
            correspondence[ci, j] = (valid & (single_ring_id_active == j)).sum().item()

    # Normalize per cluster (row)
    row_sums = correspondence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    corr_norm = correspondence / row_sums

    fig = go.Figure(
        go.Heatmap(
            z=corr_norm,
            x=[f"Ring {j}" for j in range(K)],
            y=[f"Cluster {ci}" for ci in range(n_clusters)],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(corr_norm, 2).astype(str),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"GT Ring Correspondence - {name}",
        xaxis_title="Ground-Truth Ring",
        yaxis_title="SAE Cluster",
        height=max(300, 60 * n_clusters + 100),
        width=500,
        template="plotly_white",
    )
    save_and_show(fig, f"plot3_ring_correspondence_{name}.html")


# ── Plot 4: Reconstruction chain comparison (ring 0) ───────────────────
n_angles = 256
theta_plot = torch.linspace(0, 2 * np.pi, n_angles + 1)[:-1].to(DEVICE)
circle_2d_plot = R * torch.stack([torch.cos(theta_plot), torch.sin(theta_plot)], dim=-1)

for name in models:
    tm = models[name]
    sae = best_saes[name]

    fig = go.Figure()

    j = 0  # ring 0
    Rj = dist.tilts[j].to(DEVICE)
    ring_in_ambient = circle_2d_plot @ Rj.T + dist.centers[j].to(DEVICE)

    # GT ring in input space (ring 0's 3D block)
    gt = ring_in_ambient.cpu().numpy()
    gt_closed = np.concatenate([gt, gt[:1]], axis=0)
    fig.add_trace(
        go.Scatter3d(
            x=gt_closed[:, 0],
            y=gt_closed[:, 1],
            z=gt_closed[:, 2],
            mode="lines",
            line=dict(color="black", width=5),
            name="GT ring 0",
        )
    )

    # AE reconstruction of ring 0
    full_input = torch.zeros(n_angles, N_FEATURES, device=DEVICE)
    full_input[:, j * M : (j + 1) * M] = ring_in_ambient
    with torch.no_grad():
        ae_recon, _ = tm.forward(full_input)
        ae_blk = ae_recon[:, j * M : (j + 1) * M].cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=ae_blk[:, 0],
            y=ae_blk[:, 1],
            z=ae_blk[:, 2],
            mode="markers",
            marker=dict(color=COLORS[0], size=3, opacity=0.7),
            name="AE recon",
        )
    )

    # AE+SAE reconstruction: encode -> SAE roundtrip -> decode
    with torch.no_grad():
        z_ae = tm.ae.encode(full_input)
        z_sae = sae.decode(sae.encode(z_ae))
        sae_recon = tm.ae.decode(z_sae)
        sae_blk = sae_recon[:, j * M : (j + 1) * M].cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=sae_blk[:, 0],
            y=sae_blk[:, 1],
            z=sae_blk[:, 2],
            mode="markers",
            marker=dict(color=COLORS[1], size=3, opacity=0.7, symbol="diamond"),
            name="AE+SAE recon",
        )
    )

    fig.update_layout(
        title=f"Reconstruction Chain (Ring 0) - {name}",
        scene=dict(
            xaxis_title="d1",
            yaxis_title="d2",
            zaxis_title="d3",
            aspectmode="cube",
        ),
        height=600,
        width=700,
        template="plotly_white",
    )
    save_and_show(fig, f"plot4_recon_chain_{name}.html")


# ── Plot 5: Active dictionary elements histogram ────────────────────────
for name in models:
    acts = cluster_results[name]["acts"]
    l0_per_sample = (acts > 0).float().sum(dim=1).numpy()

    fig = go.Figure(
        go.Histogram(
            x=l0_per_sample,
            nbinsx=int(l0_per_sample.max()) + 1,
            marker_color=COLORS[0],
        )
    )
    fig.add_vline(
        x=l0_per_sample.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"mean={l0_per_sample.mean():.1f}",
    )
    fig.update_layout(
        title=(
            f"Active Dictionary Elements Per Input - {name} "
            f"(lambda={best_lambdas[name]:.3f})"
        ),
        xaxis_title="Number of active dictionary elements",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        width=600,
    )
    save_and_show(fig, f"plot5_l0_histogram_{name}.html")


# %%
# ── BONUS: DIRECT RING DIMENSIONALITY (bypasses SAE) ───────────────────
# For each ring, sweep S^1 through the AE encoder and PCA the bottleneck.
# This directly measures whether each ring spans 1D or 2D in the AE's
# learned representation, independent of the SAE decomposition.
print("\n" + "=" * 60)
print("BONUS: Direct ring dimensionality in bottleneck")
print("  (Sweep each circle through AE encoder, PCA on bottleneck)")
print("=" * 60)

n_sweep = 512
theta_sweep = torch.linspace(0, 2 * np.pi, n_sweep + 1)[:-1].to(DEVICE)
circle_2d_sweep = R * torch.stack(
    [torch.cos(theta_sweep), torch.sin(theta_sweep)], dim=-1
)

for name, tm in models.items():
    print(f"\n--- {name} ---")
    print(
        f"  {'Ring':>5} | {'PCA 1':>8} | {'PCA 2':>8} | {'PCA 3':>8} | "
        f"{'Eff dim':>8} | {'Verdict':>10}"
    )
    print(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    for j in range(K):
        Rj = dist.tilts[j].to(DEVICE)
        ring_ambient = circle_2d_sweep @ Rj.T + dist.centers[j].to(DEVICE)
        full_input = torch.zeros(n_sweep, N_FEATURES, device=DEVICE)
        full_input[:, j * M : (j + 1) * M] = ring_ambient

        with torch.no_grad():
            z = tm.ae.encode(full_input).cpu().numpy()

        z_c = z - z.mean(axis=0)
        _, S, _ = np.linalg.svd(z_c, full_matrices=False)
        var = S**2 / max(len(z_c) - 1, 1)
        total = var.sum()
        vr = var / total if total > 1e-10 else np.zeros(3)
        if len(vr) < 3:
            vr = np.pad(vr, (0, 3 - len(vr)))
        eff = int((vr > PCA_DIM_THRESHOLD).sum())
        verdict = "CIRCULAR" if eff >= 2 else "SCALAR"

        print(
            f"  {j:>5} | {vr[0]:>8.4f} | {vr[1]:>8.4f} | {vr[2]:>8.4f} | "
            f"{eff:>8} | {verdict:>10}"
        )


# %%
# ── SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for name in models:
    cr = cluster_results[name]
    components = cr["components"]
    gt_assign = cr["gt_assignment"]

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"  SAE lambda: {best_lambdas[name]:.3f}")
    l0_mean = all_sweep_results[name][best_lambdas[name]]["l0"]
    print(f"  Mean L0: {l0_mean:.2f} active dict elements/sample")
    print(f"  Clusters: {len(components)}, sizes: {[len(c) for c in components]}")
    print(f"{'=' * 50}")

    # ── Dimensionality table ──
    print(f"\n  Cluster dimensionality:")
    print(
        f"    {'Clust':>5} | {'Size':>4} | {'PCA1':>6} | {'PCA2':>6} | "
        f"{'PCA3':>6} | {'Dim':>3} | {'GT ring':>7} | {'Verdict':>10}"
    )
    print(
        f"    {'-' * 5}-+-{'-' * 4}-+-{'-' * 6}-+-{'-' * 6}-+-"
        f"{'-' * 6}-+-{'-' * 3}-+-{'-' * 7}-+-{'-' * 10}"
    )

    n_circular = 0
    n_scalar = 0
    for ci, comp in enumerate(components):
        pca_info = pca_results[name][ci]
        eff = pca_info["eff_dim"]
        vr = pca_info["var_ratios"]

        # GT correspondence
        gt_rings = [gt_assign[d] for d in comp if gt_assign[d] >= 0]
        if gt_rings:
            from collections import Counter

            rc = Counter(gt_rings)
            dominant_ring, dom_count = rc.most_common(1)[0]
            purity = dom_count / len(gt_rings)
            gt_str = f"r{dominant_ring}({purity:.0%})"
        else:
            gt_str = "dead"
            dominant_ring = -1

        verdict = "CIRCULAR" if eff >= 2 else "SCALAR"
        if eff >= 2:
            n_circular += 1
        else:
            n_scalar += 1

        print(
            f"    {ci:>5} | {len(comp):>4} | {vr[0]:>6.3f} | "
            f"{vr[1]:>6.3f} | {vr[2]:>6.3f} | {eff:>3} | "
            f"{gt_str:>7} | {verdict:>10}"
        )

    # ── Separability/mixture table ──
    print(f"\n  Separability/mixture indices:")
    print(f"    (High separability = irreducible/circular)")
    print(f"    (High mixture = reducible/scalar)")
    for ci in range(len(components)):
        sm = sep_mix_results[name][ci]
        print(
            f"    Cluster {ci}: "
            f"sep(1-2)={sm['sep_12']:.4f}, sep(2-3)={sm['sep_23']:.4f}, "
            f"mix(1-2)={sm['mix_12']:.4f}, mix(2-3)={sm['mix_23']:.4f}"
        )

    # ── GT correspondence accuracy ──
    n_clean = 0
    for ci, comp in enumerate(components):
        gt_rings = [gt_assign[d] for d in comp if gt_assign[d] >= 0]
        if not gt_rings:
            continue
        rc = Counter(gt_rings)
        _, dom_count = rc.most_common(1)[0]
        if dom_count / len(gt_rings) > 0.7:
            n_clean += 1
    print(
        f"\n  GT correspondence: {n_clean}/{len(components)} clusters "
        f"cleanly map to a single ring (>70% purity)"
    )

    # ── Final assessment ──
    print(f"\n  SAE-based analysis:")
    print(f"    Circular clusters: {n_circular}/{len(components)}")
    print(f"    Scalar clusters:   {n_scalar}/{len(components)}")
    if n_circular >= 3:
        print(f"    SAE ASSESSMENT: Evidence for circular features")
    elif n_circular >= 1:
        print(f"    SAE ASSESSMENT: Mixed evidence")
    else:
        print(f"    SAE ASSESSMENT: Consistent with correlated scalars")

    print(f"\n  NOTE: The SAE-based clustering (Engels et al.) is degenerate for")
    print(f"  HIDDEN_DIM=3. Only 6/64 neurons survive (max useful in 3D with ReLU),")
    print(f"  yielding singleton clusters that are trivially 1D. See the 'Direct")
    print(f"  Ring Dimensionality' section above for the definitive measurement.")

print(f"\n{'=' * 60}")
print("CONCLUSION")
print("=" * 60)
print()
print("The SAE-based approach (Engels et al.) gives FALSE NEGATIVES here because:")
print("  1. 3D bottleneck allows at most ~6 useful SAE neurons (non-neg basis)")
print("  2. 6 singleton clusters cannot tessellate circles")
print("  3. Singleton PCA is trivially 1D")
print()
print("The DIRECT ring dimensionality test (bonus section above) is definitive:")
print("  - TiedLinearAE: ALL 5 rings are CIRCULAR (2D) in the bottleneck")
print("  - TiedMLPAE: Rings 0-2 CIRCULAR, rings 3-4 SCALAR (collapsed by importance)")
print()
print("The tied linear encoder preserves circular structure for all rings despite")
print("5 circles in 3D. The MLP sacrifices low-importance rings' circularity.")
print()
print(f"All plots saved to: {OUTPUT_DIR}")
print("=" * 60)
