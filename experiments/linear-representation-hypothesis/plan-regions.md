# Testing the Region Hypothesis in Toy Models of Superposition

## Context

This plan extends the existing Linear Representation Hypothesis (LRH) testing infrastructure in `experiments/linear-representation-hypothesis/`. The previous work developed Angular Variance (AV) as a measure of manifold structure — where feature encoding *directions* vary smoothly across contexts. This plan targets a distinct hypothesis: that features are encoded not by directions (linear or curved) but by **regions** of activation space.

We are distinguishing between three hypotheses about how features are encoded in the embedding space $h \in \mathbb{R}^m$:

| Hypothesis | Encoding mechanism | Activation boundary shape | Local gradient behaviour |
|---|---|---|---|
| **Direction** | Projection $\hat{w}_i \cdot h$ | Hyperplane | Constant nonzero gradient everywhere |
| **Manifold** | Smooth nonlinear function of $h$ | Smooth curved surface | Nonzero gradient everywhere, but direction rotates continuously |
| **Region** | Membership in a polytope $\omega \subset \mathbb{R}^m$ | Piecewise-linear (polytope faces) | **Near-zero gradient in region interiors**, large gradient only at boundaries |

The key distinguishing prediction: under the region hypothesis, the Jacobian $\partial f_i / \partial h$ should "go dead" across large patches of the embedding space, with sensitivity concentrated at ReLU-induced polytope boundaries.

---

## Existing Infrastructure

The following are already implemented and available:

- **`occhio.analysis.jacobian`**: `compute_feature_jacobians()`, `compute_all_feature_jacobians()`, `angular_variance()`, `jacobian_pca()`, `direction_vs_context()`
- **`occhio.autoencoder`**: `TiedLinearRelu`, `TiedMLPEncoder` (the two architectures we compare)
- **`occhio.distributions.sparse.SparseUniform`**: with per-feature probability control
- **`occhio.toy_model.ToyModel`**: training wrapper
- **Notebook template**: `zipf_manifold_experiment.ipynb` provides the experimental pattern

The Jacobian infrastructure computes $\frac{\partial h}{\partial x_i}$ (encoder Jacobian w.r.t. input feature $i$). For the region hypothesis tests, we also need $\frac{\partial \hat{x}_i}{\partial h}$ (decoder Jacobian w.r.t. embedding), or the full round-trip Jacobian. See implementation notes below.

---

## Test 1: Gradient Magnitude Distribution (Primary Test)

### Concept

For each feature $i$ and each sample, compute the norm of the gradient of the reconstructed feature value with respect to the embedding:

$$g_i(h) = \left\| \frac{\partial \hat{x}_i}{\partial h} \right\|$$

This measures "how sensitive is feature $i$'s reconstruction to local perturbations of the embedding at point $h$?"

**Predictions:**

- **Direction hypothesis**: $g_i(h)$ is approximately constant across all $h$ (it equals $\|w_i\|$ for a linear decoder).
- **Manifold hypothesis**: $g_i(h)$ is consistently nonzero everywhere, though its value may vary smoothly.
- **Region hypothesis**: $g_i(h)$ is **bimodal** — near-zero in region interiors (where small perturbations don't change which region you're in), with large values concentrated near polytope boundaries (where a small perturbation flips a ReLU and changes the feature's reconstruction).

### Implementation

#### Step 1: Compute decoder Jacobians

We need $\frac{\partial \hat{x}_i}{\partial h}$ for each feature $i$ at each sample's embedding $h$.

```
Input: trained model, batch of samples x [B, n_features]
1. Compute embeddings h = model.encode(x)           # [B, n_hidden]
2. For each sample b in batch:
   a. Compute full decoder Jacobian J_dec(h_b) = ∂decode(h)/∂h |_{h=h_b}   # [n_features, n_hidden]
   b. For each feature i: gradient_norm[b, i] = || J_dec[i, :] ||_2
Output: gradient_norms [B, n_features]
```

Use `torch.func.jacrev` applied to `model.decode`, analogous to the existing `compute_feature_jacobians` pattern but on the decoder side. This can reuse the same vmap/jacrev pattern in `jacobian.py`.

#### Step 2: Analyse the distribution

For each feature $i$:
- Compute the histogram of $g_i$ across all samples.
- Compute the **coefficient of variation** (std/mean) of $g_i$.
- Compute a **bimodality coefficient**: fit a Gaussian mixture (k=2) to the gradient norms. If the two-component fit is significantly better than a single Gaussian (by BIC), the feature shows region-like encoding.

Aggregate metrics:
- **Fraction of "dead-gradient" samples**: For each feature, count the fraction of samples where $g_i < \epsilon \cdot \text{median}(g_i)$ for some threshold $\epsilon$ (e.g., 0.1). High fraction = region-like.
- **Per-feature bimodality score**: Ashman's D or Hartigan's dip test.

#### Step 3: Compare architectures

Run on both `TiedLinearRelu` and `TiedMLPEncoder` with the same distribution and dimensionality. The `TiedLinearRelu` with its piecewise-linear decoder should show *some* region structure (the ReLU creates genuine polytopes). The `TiedMLPEncoder` with LeakyReLU may show weaker region structure (LeakyReLU doesn't fully zero out gradients, just attenuates them).

### Experimental configurations

| Scale | n_features | n_hidden | Purpose |
|---|---|---|---|
| Visualisable | 10 | 2–3 | Direct scatter plots of gradient norms in embedding space |
| Medium | 200 | 20 | Statistical power, match existing AV experiments |

**Distributions**: Start with `SparseUniform` with Zipf probabilities (matching the existing notebook). Extend to correlated pairs and DAG distributions if initial results are interesting.

### Visualisations

1. **Gradient norm histogram per feature**: Overlaid for both architectures. Annotate with bimodality score.
2. **Embedding space heatmap** (2D/3D only): Colour each sample's embedding by $g_i(h)$. Under the region hypothesis, you should see sharp boundaries between high-gradient and low-gradient zones.
3. **Gradient norm vs. feature activation scatter**: Plot $g_i$ against the ground-truth feature magnitude $x_i$. Under direction/manifold, these should be uncorrelated (gradient norm doesn't depend on feature value). Under regions, you might see structure.

---

## Test 2: Perturbation Sensitivity Structure

### Concept

For each sample where feature $i$ is active, apply small random perturbations to the embedding in many directions and measure which directions cause feature $i$ to flip off.

Under the **direction** hypothesis, the sensitive directions should align with the feature's weight vector $\hat{w}_i$. Under the **region** hypothesis, the sensitive directions should align with ReLU boundary normals (polytope faces), which may not align with $\hat{w}_i$.

### Implementation

#### Step 1: Identify sensitive directions

```
Input: trained model, samples where feature i is active
For each sample x with embedding h = encode(x):
   1. Generate K random unit vectors u_k in R^m  (K ≈ 100–500)
   2. For each direction u_k and step size δ:
      a. Compute h_perturbed = h + δ * u_k
      b. Compute x_hat_perturbed = decode(h_perturbed)
      c. Record Δ_i = |x_hat_perturbed[i] - x_hat[i]|
   3. The "sensitivity profile" is the function u_k -> Δ_i(u_k)
   4. Fit the top-sensitivity directions: find the direction(s) that maximise Δ_i
```

For 2D/3D embeddings, sample directions densely (e.g., uniform on the circle/sphere). For 10D+, use random sampling with enough draws to cover the space.

#### Step 2: Compare sensitive directions to feature directions vs. ReLU boundaries

**Feature direction alignment**: Compute the cosine similarity between the maximally-sensitive direction and the putative feature direction $\hat{w}_i$ (from the decoder weight matrix or the mean Jacobian).

**ReLU boundary alignment**: For the `TiedMLPEncoder`, extract the ReLU boundary normals at each sample point. These are the rows of the weight matrix of each layer, masked by the activation pattern. Compute the cosine similarity between the sensitive direction and the nearest ReLU boundary normal.

If sensitive directions align with $\hat{w}_i$ → direction hypothesis.
If sensitive directions align with ReLU normals (and *not* with $\hat{w}_i$) → region hypothesis.
If sensitive directions align with neither in a structured way → manifold hypothesis (sensitivity rotates smoothly).

#### Step 3: Aggregate

For each feature, compute:
- Mean cosine similarity between sensitive direction and feature direction
- Mean cosine similarity between sensitive direction and nearest ReLU boundary normal
- The ratio of these two quantities

### Experimental configurations

Same as Test 1 (both scales, both architectures).

### Visualisations

1. **Sensitivity rose plot** (2D/3D): For a single feature at a single sample, polar plot of $\Delta_i$ as a function of perturbation direction. Overlay the feature direction and ReLU boundary normals.
2. **Alignment histogram**: Distribution of cosine similarities (sensitive direction vs. feature direction) across all features and samples. Separate panels for each architecture.
3. **Alignment scatter**: Feature direction alignment (x-axis) vs. ReLU boundary alignment (y-axis), one point per (feature, sample) pair. Points on the diagonal suggest the two coincide; points in the upper-left corner suggest region encoding.

---

## Test 3: Boundary Normal Clustering (Distinguishes Region from Manifold)

### Concept

This test directly distinguishes the region hypothesis from the manifold hypothesis by examining the geometry of the feature activation boundary.

For each feature $i$, find pairs of nearby points that straddle the active/inactive boundary. The vector connecting such a pair approximates the local boundary normal. Under the **region** hypothesis, these normals should cluster into a **discrete set** (corresponding to different polytope faces). Under the **manifold** hypothesis, they should rotate **continuously**.

### Implementation

#### Step 1: Find boundary-crossing pairs

```
Input: trained model, large batch of samples
1. Compute embeddings and reconstructions for all samples
2. For each feature i, define "active" as x_hat[i] > threshold
   (threshold = small positive value, e.g., 0.01 * mean activation)
3. For each active sample h_a, find the nearest inactive sample h_b
   (or: walk from h_a in random directions until feature i flips off)
4. Refine: binary search along the line h_a → h_b to find the
   precise boundary crossing point
5. Record the boundary normal: n = (h_a - h_b) / ||h_a - h_b||
```

The binary search refinement is important — coarse pair-finding gives noisy normals.

#### Step 2: Analyse normal distribution

Collect all boundary normals for feature $i$ and analyse their structure:

- **Clustering**: Run k-means on the normals for k = 1, 2, ..., K. Measure silhouette scores. If the optimal k > 1 with high silhouette, the boundary is piecewise-linear (region). If k = 1 fits well, the boundary is smooth (manifold) or flat (direction).
- **Angular jump distribution**: Sort boundary normals by their position along the boundary surface (e.g., by the angle of the crossing point in 2D). Compute the angular difference between consecutive normals. Bimodal distribution (many near-zero + some large jumps) → piecewise-linear. Unimodal smooth distribution → manifold.
- **Local flatness residual**: For each cluster of normals, compute how well a single hyperplane fits the boundary points in that cluster. Near-zero residuals → flat faces (region). Consistent small residuals → curvature (manifold).

### Experimental configurations

This test is most informative at the **visualisable scale** (n_hidden = 2–3) where boundaries can be directly inspected. At medium scale, the clustering analysis still works but is harder to validate visually.

### Visualisations

1. **Boundary normal quiver plot** (2D/3D): Plot the boundary crossing points with arrows showing the local normal direction. Colour by cluster membership.
2. **Angular jump histogram**: Distribution of angular differences between consecutive boundary normals.
3. **Silhouette score vs. k**: For each feature, how many discrete boundary faces does the data support?

---

## Test 4: ReLU Boundary Alignment (Architecture-Specific)

### Concept

This test is specific to architectures with ReLU (or LeakyReLU) nonlinearities and directly checks whether feature activation boundaries coincide with neuron sign-change surfaces.

### Implementation

#### Step 1: Enumerate ReLU boundaries

For the `TiedMLPEncoder` with dims `[n, h1, m]`:
- Layer 1 has $h_1$ neurons, each with a boundary hyperplane in input space
- These map to boundaries in embedding space via the subsequent layers

For each sample $x$ with embedding $h$:
1. Record the activation pattern: for each neuron in each layer, is it positive or negative?
2. This gives a binary code $c(h) \in \{0, 1\}^{n_\text{neurons}}$

#### Step 2: Correlate activation patterns with feature activations

For each feature $i$:
1. Collect all (activation pattern, feature active/inactive) pairs
2. Compute mutual information: $I(\text{pattern}; f_i \text{ active})$
3. Compute conditional entropy: $H(f_i \text{ active} | \text{pattern})$

If $H(f_i \text{ active} | \text{pattern}) \approx 0$, the activation pattern (i.e., which region) fully determines feature $i$'s activation → **region hypothesis**.

If $H(f_i \text{ active} | \text{pattern}) \approx H(f_i \text{ active})$, the region gives no information about feature activation → regions are irrelevant.

The intermediate case (some but not full information) is also interesting.

#### Step 3: Compare with linear projection information

Also compute $H(f_i \text{ active} | \hat{w}_i \cdot h > \theta)$ — how much information does the direction projection give?

The comparison between these two conditional entropies directly answers: does region membership explain feature activation above and beyond what the linear projection already explains?

### Experimental configurations

Both architectures. Note that `TiedLinearRelu` has no hidden-layer ReLUs in the encoder (it's linear), so for that model the only nonlinearity is in the decoder. The `TiedMLPEncoder` has LeakyReLU nonlinearities in the encoder, which create the partition structure.

### Visualisations

1. **Information bar chart**: For each feature, bars showing: mutual information from linear projection, mutual information from activation pattern, and the conditional mutual information (the "extra" information from knowing the region).
2. **Region-feature correspondence matrix** (if feasible at small scale): Heatmap where rows are activation patterns and columns are features. Cell colour = probability of feature being active given that pattern.

---

## Implementation Notes

### New functions needed in `occhio/analysis/jacobian.py`

1. **`compute_decoder_jacobians(model, embeddings) -> Tensor [B, n_features, n_hidden]`**
   Analogous to `compute_feature_jacobians` but for the decoder. Uses `jacrev` on `model.decode`.

2. **`gradient_norm_distribution(model, inputs) -> Tensor [B, n_features]`**
   Wraps the decoder Jacobian computation to return per-feature gradient norms at each sample.

### New module: `occhio/analysis/regions.py`

1. **`compute_activation_patterns(model, inputs) -> Tensor [B, n_neurons]`**
   For MLP-based models, extract the binary activation pattern (which neurons are positive) at each sample. This requires hooking into the model's intermediate activations.

2. **`boundary_crossing_pairs(model, feature_idx, inputs, n_pairs, n_refine_steps) -> tuple[Tensor, Tensor, Tensor]`**
   Find pairs of nearby points straddling the feature activation boundary. Returns (boundary_points, normals, crossing_values).

3. **`perturbation_sensitivity(model, feature_idx, embeddings, n_directions, step_size) -> dict`**
   The perturbation sensitivity test. Returns sensitivity profiles, maximally sensitive directions, and alignment scores.

4. **`region_feature_mutual_information(activation_patterns, feature_activations) -> Tensor [n_features]`**
   Compute mutual information between region membership and feature activation.

### Dependencies

- `scikit-learn` for k-means clustering and silhouette scores (likely already available)
- `scipy.stats` for distribution tests (Hartigan's dip test if available, otherwise use custom bimodality coefficient)

---

## Execution Order

| Step | Description | Depends on | Effort |
|---|---|---|---|
| 0 | Read existing jacobian.py and zipf_manifold_experiment.ipynb to understand patterns | — | 0.5 day |
| 1 | Implement `compute_decoder_jacobians` and `gradient_norm_distribution` in `jacobian.py` | 0 | 0.5 day |
| 2 | Create `regions.py` with `compute_activation_patterns` and `region_feature_mutual_information` | 0 | 1 day |
| 3 | **Run Test 1** (gradient magnitude distribution) at visualisable scale (n=10, m=2–3) | 1 | 1 day |
| 4 | Run Test 1 at medium scale (n=200, m=20) | 3 | 0.5 day |
| 5 | Implement `perturbation_sensitivity` in `regions.py` | 0 | 1 day |
| 6 | **Run Test 2** (perturbation sensitivity) at both scales | 5 | 1 day |
| 7 | Implement `boundary_crossing_pairs` in `regions.py` | 0 | 1 day |
| 8 | **Run Test 3** (boundary normal clustering) at visualisable scale | 7 | 1 day |
| 9 | **Run Test 4** (ReLU boundary alignment) at both scales | 2 | 1 day |
| 10 | Write up findings and create summary visualisations | 3–9 | 1 day |

**Total estimated effort: ~8.5 days**

Step 3 is the most important checkpoint. If gradient magnitude distributions are unimodal and constant for both architectures, the region hypothesis is unlikely and we can deprioritise Tests 2–4. If they show clear bimodality, proceed with full battery.

---

## Success Criteria

The experiment battery succeeds if it produces a clear classification for each (architecture, distribution, sparsity) configuration:

1. **"Regions"**: Bimodal gradient norms (Test 1) + sensitive directions align with ReLU boundaries rather than feature directions (Test 2) + clustered boundary normals (Test 3) + high mutual information between activation patterns and feature activations (Test 4).

2. **"Manifold"**: Consistently nonzero gradient norms (Test 1) + sensitive directions don't align with either feature directions or ReLU boundaries in a fixed way (Test 2) + continuously rotating boundary normals (Test 3).

3. **"Direction"**: Constant gradient norms (Test 1) + sensitive directions align with feature weight vectors (Test 2) + flat boundary (single normal cluster, Test 3).

Mixed results within a single model (some features region-encoded, others direction-encoded) are plausible and would be an interesting finding in their own right — analogous to the "two-population" result from the manifold experiments.

---

## Relationship to Existing Work

This plan builds directly on the Angular Variance infrastructure from `experiments/linear-representation-hypothesis/`. The key conceptual addition is that AV measures whether directions *rotate* (manifold), while the new tests measure whether gradients *vanish* (regions). These are complementary: a feature could have high AV (manifold) but nonzero gradients everywhere, or low AV (direction-like) but with dead-gradient zones (region-like with flat regions).

The Humayun et al. "local complexity" framework (spline partition regions, circuit correspondence) provides theoretical grounding for the region hypothesis. Their observation that ReLU networks partition input space into polytopes with unique circuits maps directly onto our Test 4.

Connection to the paper: if region encoding is found, this would strengthen the argument that SAEs (which assume linear feature geometry) are fundamentally limited — they can't detect region-based features at all, since no linear probe will recover a feature whose encoding is purely combinatorial.