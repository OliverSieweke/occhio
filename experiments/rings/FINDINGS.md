# Circular Superposition Experiment: Findings & Implementation

## 1. Objective

Determine whether autoencoders trained on **SparseSpheres** (sparse circular features)
learn genuinely **multi-dimensional (circular)** representations or merely
**correlated scalar** features in their bottleneck. This is the central question
from Engels et al. ("Not All Language Model Features Are Linear", 2024) applied
to a controlled toy setting.

---

## 2. Experimental Setup

### 2.1 Distribution: SparseSpheres

**File:** `src/occhio/distributions/sphere.py` (179 lines)

SparseSpheres generates `k` independent features, each living on the unit circle
S^1 (or more generally S^n), embedded axis-aligned in R^{k*m}. Each feature
occupies its own m-dimensional block.

| Parameter   | Value | Description |
|-------------|-------|-------------|
| `k`         | 5     | Number of independent ring features |
| `n`         | 1     | Intrinsic dimension (S^1 = circles) |
| `m`         | 3     | Ambient dimension per feature (tilt from 2D into 3D) |
| `n_features`| 15    | Total input dimensionality (k * m) |
| `p_active`  | 0.01  | Per-feature activation probability |
| `r`         | 1.0   | Fixed radius |
| `noise_std` | 0.08  | Training noise (overridable at call time) |

**Key design decisions:**
- **Tilt matrices**: When `m > n+1`, each feature gets a random orthonormal (m x n+1)
  tilt matrix via QR decomposition. For m=3, n=1 (our case), this is identity (no tilt
  needed since 3D ambient fits S^1 naturally with 2D embedding).
- **Non-negative orthant centering**: Each feature is shifted by `r * ||R_i[d,:]||` so
  all coordinates are non-negative. This prevents cancellation between features and
  makes the sparsity structure visible.
- **Noise as call-time parameter**: `noise_std` is stored on the object as a default
  (used by `ToyModel.fit()` which calls `dist.sample(batch_size)`), but can be
  overridden at call time via `sample(n, noise_std=0.0)` for clean visualization.
  This lets us train on noised data and evaluate on clean data using the same
  distribution object with tilts built only once.

### 2.2 Autoencoders

Two architectures, both with tied weights (decoder reuses encoder weights transposed):

| Model | Architecture | Hidden dim | Parameters |
|-------|-------------|-----------|------------|
| **TiedLinearAE** | `relu(x @ W^T)` / `z @ W` | 3 | ~90 |
| **TiedMLPAE** | MLP encoder [15, 64, 32, 3], tied decoder | 3 | ~5,000 |

**Training config:**
- Epochs: 30,000
- Batch size: 256
- Learning rate: 3e-4
- Weight decay: 0.0
- Importance weighting: 0.95^(i+1) per ring, repeated M times

### 2.3 Importance Weighting

```
Ring 0: importance = 0.95^1 = 0.950 (highest)
Ring 1: importance = 0.95^2 = 0.903
Ring 2: importance = 0.95^3 = 0.857
Ring 3: importance = 0.95^4 = 0.815
Ring 4: importance = 0.95^5 = 0.774 (lowest)
```

This creates a mild priority gradient — higher-indexed rings are less important,
which turns out to matter for the MLP's allocation of representational capacity.

---

## 3. Implementation

### 3.1 Files Created/Modified

| File | Lines | Description |
|------|-------|-------------|
| `src/occhio/distributions/sphere.py` | 179 | SparseSpheres distribution with noise override |
| `src/occhio/distributions/tests/test_circular_superposition.py` | 456 | 53 tests across 12 test classes |
| `experiments/rings/experiment.py` | ~960 | Main experiment: training, visualization, SAE |
| `experiments/rings/sae_analysis.py` | 893 | Systematic circularity analysis (6 phases) |

### 3.2 Test Coverage

53 tests in `test_circular_superposition.py`:

| Test Class | Tests | What it covers |
|-----------|-------|---------------|
| TestSparseSpheresShape | 5 | Output shapes for circles, spheres, tilted |
| TestSparseSpheresTilts | 5 | Tilt matrix shapes, orthonormality, identity |
| TestSparseSpheresCenters | 3 | Center computation, positivity |
| TestSparseSpheresNonNegative | 3 | All coordinates >= 0 |
| TestSparseSpheresActivity | 3 | Sparsity mask correctness |
| TestSparseSpheresSampling | 6 | Norm verification, axis alignment |
| TestSparseSpheresWithArgs | 7 | `sample_with_args` API contract |
| TestSparseSpheresNoise | 10 | Init/param noise, override, active-only |
| TestSparseSpheresReproducibility | 4 | Seed determinism |
| TestSparseSpheresValidation | 2 | Error on bad params |
| TestSparseSpheresBroadcast | 4 | p_active broadcasting |
| TestSparseSpheresAngles | 1 | Uniform angular distribution |

### 3.3 experiment.py Structure

Notebook-style (.py with `# %%` cell markers):

1. **Config** — all hyperparameters as constants
2. **Distribution** — single SparseSpheres object shared across models
3. **AE Training** — TiedLinearAE + TiedMLPAE, 30k epochs each
4. **Loss Curves** — Plotly log-scale plot
5. **Embedding Visualization** — 1x2 3D scatter (per-AE), colored by GT ring, with
   GT ring overlays as closed curves
6. **Reconstruction Visualization** — 1x2 3D scatter, per-ring block of reconstructed
   output vs GT ring curves
7. **SAE Training** — SAESimple per model on unnoised bottleneck activations
8. **SAE Diagnostics** — L0, MSE, dead neurons, firing rates
9. **SAE Embedding** — AE ring + SAE ring + GT ring traces
10. **SAE Reconstruction** — AE recon + AE+SAE recon + GT ring traces
11. **Commented-out cells** — cosine similarity, embedding vectors, geometry metrics

### 3.4 sae_analysis.py Structure (893 lines)

Self-contained analysis script following Engels et al. methodology:

| Phase | Lines | Description |
|-------|-------|-------------|
| 0 | 66-103 | Train AEs (reproduces experiment.py with same seed) |
| 1 | 105-143 | Generate 50k bottleneck dataset with GT labels |
| 2 | 146-195 | SAE lambda sweep [0.003, 0.01, 0.03, 0.1] |
| 3 | 198-317 | Dictionary clustering (cosine similarity + connected components) |
| 4 | 320-399 | Cluster dimensionality via PCA (the key test) |
| 5 | 402-490 | Separability and mixture indices |
| 6 | 493-732 | 5 plot types (10 HTML files total) |
| Bonus | 735-778 | Direct ring dimensionality (bypasses SAE) |
| Summary | 782-893 | Tables, verdicts, conclusions |

---

## 4. SAE Analysis: Methodology

### 4.1 SAE Architecture

Standard Bricken et al. sparse autoencoder:

```
encode(z) = ReLU(z @ W_enc + b_enc)    # R^3 -> R^64
decode(a) = a @ W_dec                    # R^64 -> R^3 (no decoder bias)
loss = ||z - decode(encode(z))||^2 + lambda * ||encode(z)||_1
```

| Parameter | Value |
|-----------|-------|
| Input dim | 3 (bottleneck size) |
| Dict size | 64 (~12-13 per ring) |
| L1 sweep  | {0.003, 0.01, 0.03, 0.1} |
| Steps     | 20,000 |
| Batch size| 512 |
| LR        | 1e-3 (Adam) |

### 4.2 Lambda Selection

Target: 2-4 active dictionary elements per input (enough to tessellate circles).
Selection criterion: L0 closest to 3.0.

### 4.3 Clustering (Engels et al.)

1. Normalize decoder columns to unit norm
2. Compute 64x64 pairwise cosine similarity
3. Build graph: edge if cos_sim > threshold T
4. Connected components = clusters
5. Sweep T in {0.5, 0.6, 0.7, 0.8}, pick T giving ~5 clusters

### 4.4 Dimensionality Test

For each cluster:
1. Find all samples where at least one cluster neuron fires
2. Zero out non-cluster SAE activations
3. Reconstruct through SAE decoder (cluster-restricted reconstruction)
4. PCA on the restricted reconstructions
5. Count components with explained variance > 0.1

**Interpretation:** 2+ significant components = circular feature. 1 = scalar.

### 4.5 Separability & Mixture Indices

- **Separability index**: min MI over rotation angles in PCA 1-2 and 2-3 planes.
  High = irreducible (circular). Low = reducible (separable scalar).
- **Mixture index**: max fraction of points near any line in 2D.
  High = reducible (scalar). Low = irreducible (circular).

---

## 5. Results

### 5.1 AE Training

| Model | Final Loss |
|-------|-----------|
| TiedLinearAE | 0.011829 |
| TiedMLPAE | 0.010321 |

Both converge; MLP achieves slightly lower loss.

### 5.2 Bottleneck Statistics

| Model | Mean Norm (all) | Mean Norm (active) | Active Samples |
|-------|----------------|-------------------|----------------|
| TiedLinearAE | 0.092 | 1.900 | 2,409 / 50,000 |
| TiedMLPAE | 0.601 | 1.482 | 2,409 / 50,000 |

With p_active=0.01, only ~4.8% of samples have any active ring. Inactive samples
produce near-zero bottleneck activations (exactly zero for TiedLinearAE which has
no bias).

### 5.3 SAE Lambda Sweep (Active Samples Only)

**TiedLinearAE:**
| Lambda | L0 | MSE |
|--------|-----|----------|
| 0.003  | 3.00 | 0.000000 |
| 0.010  | 3.00 | 0.000003 |
| 0.030  | 2.99 | 0.000006 |
| 0.100  | 2.99 | 0.000045 |

**TiedMLPAE:**
| Lambda | L0 | MSE |
|--------|-----|----------|
| 0.003  | 3.00 | 0.000000 |
| 0.010  | 3.00 | 0.000002 |
| 0.030  | 2.99 | 0.000009 |
| 0.100  | 2.98 | 0.000058 |

Selected lambda = 0.003 for both (L0 closest to 3.0).

**Critical observation:** L0 = 3.0 uniformly across ALL lambda values. This is not
driven by L1 pressure — it is a geometric constraint. In R^3 with ReLU, exactly 3
non-negative basis vectors are needed to represent any point. The L1 coefficient
is irrelevant.

### 5.4 SAE Is Learning the Identity

Verification (run on TiedLinearAE SAE):

| Check | Result |
|-------|--------|
| Alive neurons | 6 / 64 |
| Relative reconstruction error | 0.023% |
| Effective linear map (W_dec @ diag @ W_enc) | Identity (distance = 0.00016) |
| Decoder column structure | 3 anti-podal pairs (cos = -1.0) |

The 6 alive neurons form **3 pairs of opposite directions** — a non-negative basis
for R^3. Each pair covers one coordinate axis (positive/negative). ReLU selects the
appropriate sign. This is the trivially optimal solution for reconstructing R^3 with
non-negative activations.

**Consequence:** The SAE-based clustering analysis is uninformative. It is analyzing
an identity transformation, not a meaningful feature decomposition.

### 5.5 Wide L1 Sweep: Breaking the Identity

Extended L1 sweep [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] on TiedLinearAE
reveals three distinct regimes:

| Lambda | L0   | Alive | Rel Error | d(I)   | Regime |
|--------|------|-------|-----------|--------|--------|
| 0.003  | 3.00 | 6     | 0.02%     | 0.0002 | Identity |
| 0.010  | 3.00 | 6     | 0.03%     | 0.0004 | Identity |
| 0.030  | 2.99 | 6     | 0.06%     | 0.001  | Identity |
| 0.100  | 2.99 | 6     | 0.2%      | 0.005  | Identity |
| 0.300  | 2.97 | 6     | 0.8%      | 0.018  | Identity |
| 1.000  | 2.93 | 6     | 3.5%      | 0.055  | Transition |
| 3.000  | 1.20 | 3     | 52.6%     | —      | Broken |
| 10.00  | 0.00 | 0     | 100%      | —      | Dead |

**Identity regime (L1 ≤ 0.3):** L0 ≈ 3.0, 6 alive neurons forming 3 anti-podal
pairs. The effective linear map is near-identity (d(I) < 0.02). L1 pressure is
irrelevant — the geometric floor of 3 non-negative basis vectors dominates.

**Transition (L1 = 1.0):** L0 drops to 2.93. The identity starts weakening
(d(I) = 0.055) but 6 neurons remain alive. Reconstruction quality is still high
(3.5% relative error).

**Broken identity (L1 = 3.0):** L0 = 1.20, only 3 neurons survive. The SAE
breaks out of identity and develops **ring-specific neurons**:
- Ring 0 → neuron 13
- Ring 1 → neurons 13 + 44
- Ring 2 → neuron 47
- Ring 3 → neuron 44
- Ring 4 → neuron 47

Each ring activates 1-2 specific neurons rather than a generic 3-neuron basis.
However, reconstruction error is 52.6% — the SAE sacrifices fidelity for sparsity.
The ring-specific pattern suggests the SAE has learned a coarse "which ring is
active?" detector, but cannot represent the angular position within a ring using
only 1-2 neurons.

**Dead (L1 = 10.0):** All neurons killed. L0 = 0.

**Implication:** The identity is not "fatal" per se — it can be broken with
sufficient L1 pressure (lambda ≥ 3.0). But breaking it destroys the angular
information that makes rings circular. The fundamental problem remains: 3D is
too low-dimensional for a meaningful sparse decomposition of circular features.

### 5.6 SAE Clustering Results (Degenerate)

Both models: 6 singletons, all trivially 1D.

| Cluster | Size | PCA dim | GT ring | Verdict |
|---------|------|---------|---------|---------|
| 0 | 1 | 1 | r0 (100%) | SCALAR |
| 1 | 1 | 1 | r0 (100%) | SCALAR |
| 2 | 1 | 1 | r2 (100%) | SCALAR |
| 3 | 1 | 1 | r4 (100%) | SCALAR |
| 4 | 1 | 1 | r1 (100%) | SCALAR |
| 5 | 1 | 1 | r3 (100%) | SCALAR |

**Why this fails:** In 3D with ReLU, at most ~6 independent non-negative directions
exist. The SAE learns exactly these 6 as a minimal basis. Single-neuron clusters
are trivially 1-dimensional regardless of the underlying feature structure.

### 5.7 Direct Ring Dimensionality (The Definitive Test)

Bypasses the SAE entirely. For each ring, sweeps 512 points around S^1 through the
AE encoder and runs PCA on the bottleneck activations.

**TiedLinearAE — ALL 5 rings CIRCULAR:**
| Ring | PCA 1 | PCA 2 | PCA 3 | Eff Dim | Verdict |
|------|-------|-------|-------|---------|---------|
| 0 | 0.601 | 0.399 | 0.000 | 2 | CIRCULAR |
| 1 | 0.656 | 0.344 | 0.000 | 2 | CIRCULAR |
| 2 | 0.549 | 0.451 | 0.000 | 2 | CIRCULAR |
| 3 | 0.584 | 0.416 | 0.000 | 2 | CIRCULAR |
| 4 | 0.659 | 0.341 | 0.000 | 2 | CIRCULAR |

Every ring spans exactly a 2D plane in the 3D bottleneck (PCA 3 = 0.000).
Variance splits range from ~55/45 to ~66/34, indicating slightly elliptical but
genuinely 2-dimensional embeddings.

**TiedMLPAE — 3 circular, 2 scalar:**
| Ring | PCA 1 | PCA 2 | PCA 3 | Eff Dim | Verdict |
|------|-------|-------|-------|---------|---------|
| 0 | 0.654 | 0.344 | 0.001 | 2 | CIRCULAR |
| 1 | 0.582 | 0.417 | 0.002 | 2 | CIRCULAR |
| 2 | 0.642 | 0.355 | 0.003 | 2 | CIRCULAR |
| 3 | 0.984 | 0.012 | 0.004 | 1 | SCALAR |
| 4 | 0.966 | 0.029 | 0.006 | 1 | SCALAR |

Rings 3-4 (importance 0.815, 0.774) collapse to near-1D: PCA 1 explains 98.4% and
96.6% respectively. The MLP trades low-importance circularity for better
reconstruction of high-importance features.

---

## 6. Interpretation

### 6.1 TiedLinearAE Preserves All Circles

The tied linear encoder `relu(x @ W^T)` maps each circle to a 2D plane in the
3D bottleneck with zero residual variance. This is mathematically clean: a linear
map sends a circle (1D manifold in a 2D plane) to another 2D plane (possibly
distorted by ReLU clipping, but still 2D). The weight-tying constraint means the
decoder `z @ W` must invert this exactly, which preserves the rotational structure.

5 circles in 3 dimensions means the planes must overlap — this is the superposition.
But each individual ring still traces a 2D curve in the bottleneck, not a 1D line.

### 6.2 TiedMLPAE Sacrifices Low-Importance Rings

The MLP has more capacity (5,000 vs 90 parameters) but uses it selectively.
Rings 0-2 (importance >= 0.857) get circular embeddings. Rings 3-4 (importance
<= 0.815) are compressed to near-scalars. The MLP is making an explicit
capacity allocation: it spends its hidden layers' representational budget on
faithfully representing high-importance circles and reduces low-importance ones
to intensity-only features.

This is consistent with the superposition hypothesis: when bottleneck capacity is
limited, the model prioritizes higher-importance features for multi-dimensional
representation and compresses lower-importance features to scalars.

### 6.3 Why the SAE Approach Fails Here

The Engels et al. SAE-based methodology requires:
1. Enough dictionary elements per feature to tessellate the manifold (~10+ per circle)
2. Dictionary elements that cluster by feature rather than by spatial direction
3. Multi-element clusters whose restricted reconstructions span 2D

In a 3D bottleneck, all three assumptions break:
- **Geometric ceiling**: ReLU in R^3 allows at most ~6 useful dictionary elements
  (3 anti-podal pairs forming a non-negative basis)
- **Identity collapse**: The SAE learns to perfectly reconstruct via a change of
  basis, not via meaningful feature decomposition
- **Singleton clusters**: 6 neurons in 6 directions = 6 clusters of size 1,
  trivially 1-dimensional

The SAE approach is designed for high-dimensional spaces (e.g., 768D transformer
activations) where the overcomplete dictionary can tessellate low-dimensional
manifolds embedded in high-dimensional space. In 3D, the "manifold" IS the space.

### 6.4 When Would the SAE Approach Work?

To make the SAE-based circularity test valid, you need:
- **HIDDEN_DIM >> 3**: At least 10-20 so that 64 dictionary elements genuinely
  overcomplete the space and can tessellate manifolds
- **Higher p_active**: More active samples for training signal diversity
- **Run SAE on 15D input space**: Where circles naturally live in 2D subspaces
  within a 15D ambient space — exactly the setting Engels et al. studied

---

## 7. Generated Outputs

### 7.1 experiment.py Figures

All saved to `experiments/rings/figures/`:
- `loss_curves.html` — Training loss (log scale) for both AEs
- `bottleneck.html` — 3D bottleneck embeddings colored by ring
- `reconstruction.html` — Per-ring 3D reconstruction vs GT
- `sae_loss_curves.html` — SAE training loss
- `sae_embedding.html` — AE + SAE + GT ring embeddings
- `sae_reconstruction.html` — AE + AE+SAE + GT ring reconstructions

### 7.2 sae_analysis.py Figures

All saved to `experiments/rings/sae_analysis/`:
- `plot1_dict_sphere_{model}.html` — Dictionary elements on unit sphere
- `plot2_cluster_pca23_{model}.html` — Cluster activations in PCA 2-3 space
- `plot3_ring_correspondence_{model}.html` — Cluster vs GT ring heatmap
- `plot4_recon_chain_{model}.html` — GT -> AE -> AE+SAE reconstruction chain
- `plot5_l0_histogram_{model}.html` — Active dictionary elements distribution

---

## 8. Lessons Learned

1. **Filter to active samples before SAE training.** With p_active=0.01, 95% of
   bottleneck activations are zero. Training on all samples teaches the SAE to
   output nothing. First run (unfiltered): L0=0.21. After filtering: L0=3.00.

2. **SAEs learn identity in low dimensions.** In R^d with ReLU, the SAE will
   learn 2d dictionary elements forming d anti-podal pairs — a non-negative
   basis that perfectly reconstructs via a change of coordinates. This is trivial
   and uninformative. The SAE approach requires high-dimensional inputs where
   sparsity is a meaningful constraint.

3. **Plotly 3D `visible="legendonly"` breaks empty scenes.** When ALL traces in a
   3D subplot start as `legendonly`, Plotly never initializes the scene axes.
   The plot renders completely blank and clicking legend items doesn't fix it.
   Solution: use `visible=True` and let users toggle off manually.

4. **Direct PCA on per-ring bottleneck activations is the cleanest test.**
   No SAE needed. Sweep S^1 through the encoder, PCA on the 3D output, check
   if 2 components are significant. Simple, fast, and definitive.

5. **The tied linear encoder naturally preserves circular structure.**
   `relu(x @ W^T)` is a piecewise-linear map that sends 2D planes to 2D planes
   (modulo ReLU clipping). Weight tying forces the decoder to invert this,
   maintaining the rotational structure. 5 circles in 3D = superposition, but
   each circle remains genuinely 2-dimensional.

6. **Importance weighting drives capacity allocation in MLPs.** The MLP
   selectively collapses low-importance features from 2D (circular) to 1D (scalar),
   while preserving high-importance features' circularity. This is a concrete
   instance of the superposition hypothesis: limited capacity forces feature
   compression, and importance determines which features get compressed.

7. **SAE identity can be broken — but at a cost.** A wide L1 sweep (0.003–10.0)
   shows the identity is robust up to L1=0.3, transitions at L1=1.0, and breaks
   at L1=3.0. In the broken regime, neurons become ring-specific (each ring
   activates 1-2 specific neurons), but reconstruction error spikes to 52.6%.
   The SAE trades angular information for a coarse "which ring?" detector.
   This demonstrates a fundamental tension: sparse decomposition of circular
   features in low dimensions must destroy the very structure we're trying to detect.

---

## 9. Open Questions

- Does increasing HIDDEN_DIM (e.g., 10 or 20) allow the SAE approach to recover
  circular structure? At what dimensionality does the method become reliable?
- Is there a sharp phase transition in the MLP's ring dimensionality as importance
  decreases, or is it gradual?
- Can we quantify the superposition geometry more precisely — how do the 5 circular
  planes overlap in the 3D bottleneck?
- Would training with higher p_active (e.g., 0.1) change whether rings are encoded
  circularly vs. as scalars?
- At L1=3.0 (broken regime), the ring-neuron mapping shows sharing (neurons 13, 44,
  47 serve multiple rings). Is this a crude superposition of ring identities, or an
  artifact of the sparse bottleneck?
