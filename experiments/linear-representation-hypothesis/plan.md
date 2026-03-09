# Testing the Manifold Representation Hypothesis in Occhio

## Motivation

The linear representation hypothesis claims that features correspond to fixed directions in activation space. An alternative hypothesis posits that activation space is a curved manifold — the "direction" of a feature depends on where you are in the space (i.e., which other features are co-active). We want to test this using toy models where ground truth is known.

**Key insight:** Standard TMOS models with linear encoders ($h = Wx$ or $h = \text{ReLU}(Wx)$) are architecturally incapable of exhibiting smooth manifold structure. The Jacobian $\partial h / \partial x_i = w_i$ is constant, so feature directions are globally fixed by construction. We need nonlinear encoders to give the model the *capacity* for manifold representations, then check whether it *chooses* to use them.

---

## Phase 1: New Autoencoder Architecture

**Goal:** Add an MLP-based autoencoder to occhio that can represent nonlinear encodings.

### 1.1 Implement `MLPAutoencoder`

Create a new autoencoder class that follows occhio's existing interface (pluggable with `ModelGrid`, compatible with visualization tools).

```
Architecture:
  Encoder: x ∈ ℝⁿ → σ(V · σ(Wx + b₁) + b₂) → h ∈ ℝᵐ
  Decoder: h ∈ ℝᵐ → W_dec · h + b_dec → x̂ ∈ ℝⁿ
```

Design decisions:

- **Smooth nonlinearity** (GELU or Tanh, NOT ReLU) — ReLU is piecewise linear, which gives discrete context-dependence but not smooth manifold structure. GELU is preferred because it's smooth everywhere and matches what transformers actually use.
- **Keep the decoder linear** (or mildly nonlinear) — this isolates the question to the encoding side. If the decoder is also nonlinear, it becomes harder to attribute manifold structure to the encoder vs. decoder.
- **Hidden width of the encoder MLP** should be a hyperparameter. Start with `hidden_dim = 2 * m` (twice the bottleneck dimension).
- **Tied vs. untied weights:** Start untied. Tying would force symmetry that could suppress manifold structure.

### 1.2 Implement `ShallowNonlinearAutoencoder`

A simpler variant as a stepping stone:

```
Encoder: x ∈ ℝⁿ → σ(Wx + b) → h ∈ ℝᵐ
Decoder: h ∈ ℝᵐ → W_dec · h + b_dec → x̂ ∈ ℝⁿ
```

This has a single nonlinear layer in the encoder. It can produce *some* curvature but less than the MLP. Useful as an intermediate comparison point.

### 1.3 Training integration

- Both architectures should work with occhio's existing `ModelGrid` for parameter sweeps.
- Same loss function as standard TMOS: weighted MSE reconstruction loss.
- Same optimizer and training loop — the only change is the architecture.

**Files to create/modify:**

- `occhio/autoencoders/mlp.py` — new module
- `occhio/autoencoders/__init__.py` — register new architectures
- Ensure compatibility with existing `ModelGrid` and visualization tools

---

## Phase 2: Jacobian Analysis Tooling

**Goal:** Build tools to compute and analyze how feature encoding directions vary across contexts.

### 2.1 Local Feature Direction Extraction

For a trained model and a given feature $i$, compute the Jacobian of the encoder output with respect to $x_i$:

$$J_i(x) = \frac{\partial h}{\partial x_i} \bigg|_x \in \mathbb{R}^m$$

This vector is the local encoding direction for feature $i$ at input $x$.

**Implementation:**

```python
def compute_feature_jacobians(
        model,  # trained autoencoder
        feature_idx: int,  # which feature
        inputs: Tensor,  # batch of input samples [B, n]
) -> Tensor:  # returns [B, m] — one Jacobian vector per sample
    """
    For each input x in the batch, compute ∂h/∂x_i.
    Uses torch.autograd.functional.jacobian or manual gradient computation.
    """
```

**Sampling strategy:** For feature $i$, generate $N$ samples (e.g., $N = 1000$) where feature $i$ is active (nonzero) but the other active features vary. This is straightforward with occhio's distribution objects — sample from the distribution and filter for samples where feature $i$ fires.

**Important:** Also compute Jacobians for the *linear baseline model* trained on the same distribution. These should all be identical (equal to $w_i$), confirming the tool works correctly.

### 2.2 Angular Variance Metric

For feature $i$, given a collection of Jacobian vectors $\{J_i(x^{(k)})\}_{k=1}^N$:

1. Normalize each vector: $\hat{J}_i^{(k)} = J_i^{(k)} / \|J_i^{(k)}\|$
2. Compute the mean direction: $\bar{J}_i = \frac{1}{N}\sum_k \hat{J}_i^{(k)}$, then normalize
3. **Angular variance:** $\text{AV}_i = 1 - \|\bar{J}_i\|$

This ranges from 0 (all directions identical → linear) to ~1 (directions uniformly spread → maximally nonlinear). It's analogous to the circular variance in directional statistics.

```python
def angular_variance(jacobians: Tensor) -> float:
    """
    jacobians: [N, m] tensor of Jacobian vectors
    Returns scalar angular variance in [0, 1].
    """
    normed = jacobians / jacobians.norm(dim=1, keepdim=True)
    mean_dir = normed.mean(dim=0)
    return 1.0 - mean_dir.norm().item()
```

### 2.3 PCA of Jacobian Vectors

Run PCA on the normalized Jacobian vectors for each feature. The eigenvalue spectrum reveals the intrinsic dimensionality of the "feature direction manifold":

- **1 dominant eigenvalue:** feature direction is essentially fixed (linear)
- **$k$ significant eigenvalues:** feature direction varies on a $k$-dimensional submanifold
- **Flat spectrum:** directions are essentially random (pathological)

```python
def jacobian_pca(jacobians: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns (eigenvalues, eigenvectors) sorted descending."""
    normed = jacobians / jacobians.norm(dim=1, keepdim=True)
    centered = normed - normed.mean(dim=0)
    U, S, V = torch.svd(centered)
    return S ** 2 / (len(jacobians) - 1), V
```

### 2.4 Conditional Direction Analysis

Beyond just measuring variance, analyze *what* causes the direction to change:

```python
def direction_vs_context(
        model,
        feature_idx: int,
        inputs: Tensor,
        jacobians: Tensor,
) -> dict:
    """
    For each other feature j != feature_idx, compute the correlation
    between j's activation magnitude and the Jacobian direction for
    feature_idx. This reveals which co-active features cause the
    encoding direction to rotate.
    """
```

This gives interpretable results: "Feature 3's encoding direction rotates toward dimension X when feature 7 is co-active."

**Files to create:**

- `occhio/analysis/jacobian.py` — core Jacobian computation and angular variance
- `occhio/analysis/manifold.py` — PCA, conditional analysis, and manifold metrics

---

## Phase 3: Experiments

### 3.1 Experiment Configuration

**Dimensionality:** $n = 200$ features, $m = 50$ bottleneck dimensions (4:1 compression ratio). This is high enough for interesting manifold geometry but still computationally tractable for Jacobian computation.

**Distributions to test:**

1. **Sparse uniform (i.i.d.)** — the standard TMOS baseline
2. **Correlated pairs** — occhio's `CorrelatedPairs` distribution
3. **Hierarchical** — occhio's `HierarchicalSparse` distribution
4. **DAG-based** — the generalized DAG distributions from the paper

**Sparsity sweep:** Vary density from very sparse ($p = 0.01$) to moderate ($p = 0.3$). The hypothesis is that manifold structure is more likely at higher density where linear packing runs out of capacity.

**Architectures to compare:**
| Architecture | Purpose | |---|---| | `TiedLinearRelu` | Baseline — should show zero angular variance | | `ShallowNonlinearAutoencoder` | Minimal nonlinearity | | `MLPAutoencoder` (1 hidden layer) | Moderate capacity for curvature | | `MLPAutoencoder` (2 hidden layers) | High capacity for curvature |

### 3.2 Core Experiments

**Experiment A: Does manifold structure emerge?**

For each (architecture, distribution, sparsity) triple:

1. Train the model to convergence.
2. For each feature $i$, compute angular variance $\text{AV}_i$ from 1000 samples.
3. Report mean and max angular variance across features.

**Key comparison:** Do the nonlinear models develop significantly higher angular variance than the linear baseline? If they converge to similar angular variance (near zero), the linear representation hypothesis is a natural attractor even when the model could be nonlinear.

**Experiment B: Phase transition in manifold structure**

Fix architecture = `MLPAutoencoder`, distribution = sparse uniform. Sweep:

- Compression ratio $n/m$ from 1.5 to 10
- Sparsity $p$ from 0.01 to 0.5

Plot angular variance as a function of both parameters. Look for a phase transition: at what compression ratio does the model start using nonlinear encoding?

**Experiment C: Reconstruction quality comparison**

Compare reconstruction loss of linear vs. nonlinear models. If the MLP achieves *significantly* lower loss, that suggests the nonlinear encoding is doing useful work (not just noise). If losses are similar, linearity may genuinely be near-optimal.

**Experiment D: Conditional direction structure**

For models that DO show manifold structure:

- Run the conditional direction analysis from 2.4
- Identify which feature co-activations cause the largest direction rotations
- Check whether these correspond to correlated/hierarchical features in the distribution

### 3.3 Visualization

**Low-dimensional sanity checks:** Run the same experiments at $n = 10, m = 3$ where the Jacobian vectors can be directly visualized on a sphere (they live in $\mathbb{R}^3$). Plot the Jacobian vectors for a single feature across many contexts — if they cluster tightly, linear holds; if they spread on the sphere, there's manifold structure.

Use occhio's existing Plotly-based visualization tools where possible. New visualizations needed:

- **Jacobian direction scatter on a sphere** (for $m = 3$): 3D scatter of normalized Jacobians colored by co-active features
- **Angular variance heatmap:** features × sparsity, colored by angular variance
- **PCA eigenvalue spectrum:** bar chart of eigenvalues for each feature's Jacobian distribution
- **Direction rotation quiver plot** (for $m = 3$): arrows on a sphere showing how feature direction changes with context

**Files to create:**

- `occhio/visualization/manifold.py` — new visualization functions

---

## Phase 4: SAE Implications

**Goal:** If manifold structure IS found, test whether SAEs can handle it.

### 4.1 SAE on Nonlinear Representations

Train a standard SAE on the bottleneck activations of the MLP autoencoder. Compare feature recovery metrics against the linear baseline. If the activation space is curved, SAE (which assumes linear features) should show degraded performance — potentially new failure modes beyond absorption/splitting.

### 4.2 Nonlinear SAE Comparison

If standard SAEs fail, test whether a nonlinear SAE variant (e.g., with a nonlinear encoder) can recover features better. This connects to the broader question of whether we need fundamentally different interpretability tools for manifold representations.

---

## Implementation Order & Estimated Effort

| Step | Description                           | Depends on | Effort           |
|------|---------------------------------------|------------|------------------|
| 1.1  | `MLPAutoencoder` class                | —          | 1 day            |
| 1.2  | `ShallowNonlinearAutoencoder` class   | —          | 0.5 day          |
| 1.3  | Training integration with `ModelGrid` | 1.1, 1.2   | 0.5 day          |
| 2.1  | Jacobian computation                  | 1.1        | 1 day            |
| 2.2  | Angular variance metric               | 2.1        | 0.5 day          |
| 2.3  | Jacobian PCA                          | 2.1        | 0.5 day          |
| 2.4  | Conditional direction analysis        | 2.1        | 1 day            |
| 3.1  | Experiment scripts & configs          | 1.3, 2.2   | 1 day            |
| 3.2  | Run experiments A-D                   | 3.1        | 2 days (compute) |
| 3.3  | Visualization functions               | 2.1-2.4    | 1.5 days         |
| 4.1  | SAE on nonlinear activations          | 3.2        | 1 day            |
| 4.2  | Nonlinear SAE comparison              | 4.1        | 1 day            |

**Total estimated effort: ~11 days**

---

## Key Risks & Mitigations

**Risk:** The MLP encoder might overfit, creating apparent manifold structure that's just noise.
**Mitigation:** Use held-out test data for all Jacobian analyses. Check that angular variance is stable across different random seeds and sample sets.

**Risk:** Jacobian computation at $n = 200, m = 50$ might be slow for large batches.
**Mitigation:** Use `torch.autograd.functional.jacobian` with vectorized mode, or compute column-by-column using `torch.autograd.grad`. For 1000 samples, this should take seconds, not minutes.

**Risk:** The nonlinear model converges to a linear solution anyway, making the experiment uninformative.
**Mitigation:** This IS an informative result — it would provide evidence that linearity is a natural attractor. Additionally, try different initializations and learning rates to ensure the model isn't stuck in a linear local minimum.

**Risk:** Manifold structure appears but is an artifact of the toy model (wouldn't generalize to real networks).
**Mitigation:** This is inherent to all toy model research. Frame findings carefully. If manifold structure appears under specific distribution types (e.g., hierarchical but not i.i.d.), that itself is an interesting and potentially generalizable finding about when linear representations break down.

---

## Success Criteria

The experiment is successful if it produces a clear answer to at least one of:

1. **Do nonlinear models develop manifold structure when they could?** (Measured by angular variance significantly above baseline)
2. **At what compression ratio / sparsity does manifold structure emerge?** (Phase transition plot)
3. **Does distribution structure (correlations, hierarchy) promote or suppress manifold representations?** (Comparison across distribution types)
4. **If manifold structure exists, does it cause SAE failure?** (Feature recovery degradation)