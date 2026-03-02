# Law 1: The Correlation-Interference Law

## Background: What is Law 1?

Law 1 is a mechanistic principle governing how neural networks compress correlated features in sparse autoencoders. It proposes that:

**For paired features with correlation coefficient `c`, the optimal Gram matrix interference (cosine similarity) between the pair should increase monotonically with `c`.**

At the extremes:
- **c = 0** (independent features): features should maintain minimal interference (standard Toy Models of Superposition result)
- **c = 1** (perfectly correlated): features can be fully aligned since they always co-occur; interference is "free"

### Theoretical Motivation

The reconstruction loss cost of interference between two features depends on the probability that exactly one is active while the other is not:

```
Cost(interference) ∝ P(exactly one active)
                   = 2 · density · (1 - c) · (1 - density)
```

The critical term is **(1 - c)**: as correlation increases, this probability drops, making interference cheaper. Therefore, the model should *tolerate* more interference as correlation increases.

---

## Why It Matters

This law bridges **network geometry** (how features are arranged in latent space) and **feature statistics** (how often features co-occur). If true, it provides a principled way to predict geometric structure from data statistics—a key goal in mechanistic interpretability.

---

## Experimental Design

### Overview

Three complementary experiments validate Law 1 under different conditions:

```
Experiment A (Primary)        Experiment B (Scale)           Experiment C (Control)
n_features=6, n_hidden=2     n_features=10, n_hidden=3     n_features=6, n_hidden=2
20 correlations × 15 densities   12 × 10 grid              Anticorrelated pairs (1D)
300 models total             120 models total              15 models total
```

All experiments use identical hyperparameters to ensure fair comparison.

### Experiment A: Primary Evidence

**Grid Parameters:**
- **Correlations**: 0.0 → 0.95 (20 values, linear spacing)
- **Densities**: 0.01 → 0.5 (15 values, log spacing)
- **Distribution**: `CorrelatedPairs` with 3 feature pairs
- **Architecture**: TiedLinearRelu(n_features=6, n_hidden=2)
- **Training**: 10,000 epochs, batch_size=1024, lr=3e-4

**Rationale:**
- Small feature count (6) allows close inspection of individual pair behavior
- 3 pairs provide within-pair and cross-pair comparisons
- 20 × 15 resolution gives fine-grained monotonicity evidence
- CPU training for reproducibility

**Key Observable:**
- **Within-pair cosine** = `model.interferences[2i, 2i+1]` for each pair
- Should increase *monotonically* with correlation at fixed density

### Experiment B: Scale Validation

**Grid Parameters:**
- **Correlations**: 0.0 → 0.95 (12 values, linear spacing)
- **Densities**: 0.01 → 0.5 (10 values, log spacing)
- **Distribution**: `CorrelatedPairs` with 5 feature pairs
- **Architecture**: TiedLinearRelu(n_features=10, n_hidden=3)
- **Training**: Same hyperparameters as Experiment A

**Rationale:**
- Larger scale (10 features, 3 hidden) tests generalization
- More feature pairs (5) increase sample size for statistics
- Coarser resolution (12 × 10) for computational efficiency
- Validates that Law 1 is not an artifact of small-scale regime

**Expected Result:**
- Pattern should match Experiment A qualitatively
- Within-pair cosine ranges may differ (depends on capacity)
- Monotonicity should hold consistently

### Experiment C: Anticorrelated Control

**Grid Parameters:**
- **Densities**: 0.01 → 0.5 (15 values, log spacing)
- **Distribution**: `AnticorrelatedPairs` (mutually exclusive pairs)
- **Architecture**: TiedLinearRelu(n_features=6, n_hidden=2)
- **Training**: Same hyperparameters

**Rationale:**
- Tests whether Law 1 is specific to positive correlation
- Anticorrelated pairs should show **opposite** pattern (features repel)
- Within-pair cosine should be **minimal** and independent of density
- Validates that correlation-driven effects are real, not generic

**Expected Result:**
- Within-pair cosine ≈ 0 (features stay orthogonal)
- Should NOT increase with density (unlike correlated case)
- Provides evidence that effect is correlation-specific, not artifact

---

## Key Metrics

For each trained model, we extract:

| Metric | Source | Interpretation |
|--------|--------|-----------------|
| **Within-pair cosine** | `model.interferences[2i, 2i+1]` | Core Law 1 observable (should increase with c) |
| **Cross-pair cosine** | `model.interferences[i,j]` for i,j in different pairs | Baseline/noise (should be independent of c) |
| **Feature norms** | `model.feature_norms` | Phase/capacity indicator |
| **Feature dimensionality** | `model.feature_dimensionalities` | Representation efficiency |
| **Gram matrix** | `model.W_T_W` | Full pairwise interference structure |

---

## Falsification Criteria

Law 1 is **falsified** if any of these occur:

### Primary Falsification

❌ **Within-pair cosine does NOT increase monotonically with correlation** at fixed density.

**What it would mean:** Correlation alone doesn't predict optimal interference → Law 1 is incomplete or wrong.

### Secondary Falsification

❌ **Cross-pair cosine also increases with correlation.**

**What it would mean:** All features align with correlation, not just paired features → suggests generic capacity pressure, not correlation-specific effect.

### Tertiary Falsification

❌ **Anticorrelated pairs show similar patterns to correlated pairs.**

**What it would mean:** Pair structure itself drives alignment, not the correlation coefficient → Law 1 is measuring something else.

---

## Supporting Evidence Criteria

✓ **Strong support for Law 1:**

1. **Monotonic increase in within-pair cosine with correlation** (Figure 2)
   - Holds across multiple density levels
   - No plateaus or reversals

2. **Cross-pair cosine is independent of correlation** (Figure 3)
   - Shows no monotonic trend with correlation
   - Confirms effect is pair-specific

3. **Structural interference grows with correlation** (Figure 4)
   - Defined as: within-pair cosine − cross-pair cosine
   - Isolates the correlation-driven component

4. **Anticorrelated pairs show opposite pattern** (Figure 8)
   - Within-pair cosine ≈ 0
   - Features actively repel rather than align

5. **Pattern generalizes to larger scales** (Figure 7)
   - Experiment B heatmap matches Experiment A structure
   - Not restricted to small toy regime

6. **Scaling matches theoretical prediction** (Figure 6)
   - Within-pair cosine ∝ 1/(1−c) relationship
   - Validates cost-model hypothesis

---

## What We're Investigating at High Level

### The Core Question

> **How do network weights encode feature statistics?**

Specifically:

1. **Can we predict geometric structure from data alone?**
   - If Law 1 holds: yes, interference is predictable from correlation

2. **What is the relationship between physics (optimization) and statistics (feature distribution)?**
   - This law bridges the two: optimization exploits statistical structure

3. **Do networks discover universal principles for efficient representation?**
   - If Law 1 is principled rather than empirical, it should generalize

### Broader Implications

If Law 1 is confirmed:
- **Mechanistic interpretability becomes more tractable**: geometry → statistics → interpretability pipeline
- **Network behavior is more predictable**: we can anticipate alignment patterns from data properties
- **Universality exists in sparse autoencoders**: similar to universality in physics

If Law 1 is falsified:
- **Networks may use ad-hoc solutions** rather than principled ones
- **Feature geometry depends on architecture/training details**, not just data
- **Need richer models** to predict how networks use latent capacity

---

## How to Run

### Single Experiment

```bash
cd experiments/laws/law1
python experiment.py
```

This trains all three experiment grids (~435 total models) and generates 8 figures.

⏱️ **Expected time**: 4-8 hours on CPU (depends on machine)

### Test Mode (Optional)

To verify setup without full training:

```python
from experiment import create_model_experiment_a, extract_metrics
from occhio.model_grid import ModelGrid, Axis
import torch

# Create tiny test grid (2x2 = 4 models)
grid = ModelGrid(
    create_model=create_model_experiment_a,
    axes=[
        Axis(label="Correlation", values=torch.tensor([0.1, 0.5])),
        Axis(label="Density", values=torch.tensor([0.05, 0.2])),
    ],
    cache_samples=False,
)

# Train briefly
grid.fit(n_epochs=100, verbose=True)

# Extract metrics
metrics = extract_metrics(grid, n_features=6)
print(f"Within-pair cosine shape: {metrics['within_pair_cos'].shape}")
```

---

## Output Structure

```
experiments/laws/law1/
├── experiment.py                 # Main script (7 cells)
├── README.md                     # This file
└── figures/
    ├── 01_within_pair_heatmap.png           # Core observable (all c,ρ)
    ├── 02_within_pair_vs_correlation.png    # Monotonicity test
    ├── 03_cross_pair_heatmap.png            # Control (should be flat in c)
    ├── 04_structural_interference.png       # Within − cross
    ├── 05_feature_norms_heatmap.png         # Phase diagram
    ├── 06_theoretical_scaling.png           # 1/(1−c) comparison
    ├── 07_scale_validation.png              # Exp B heatmap
    └── 08_anticorrelated_control.png        # Opposite trend
```

---

## Technical Details

### Why These Distributions?

- **CorrelatedPairs**: Directly tests Law 1 hypothesis
- **AnticorrelatedPairs**: Control for "pairing effect" itself

Both are available in `occhio.distributions`.

### Why These Architectures?

- **TiedLinearRelu**:
  - Tied encoder/decoder (forces feature geometry to matter)
  - ReLU activation (sparse representation)
  - Simplest architecture where Law 1 can manifest
- Sizes chosen for representational capacity (roughly 1-2 bits per feature)

### Why CPU?

- Reproducibility (consistent randomness seeds)
- Sufficient for network sizes used
- Easier to scale across multiple runs

---

## References

- **Toy Models of Superposition**: https://transformer-circuits.pub/2022/toy_model/index.html
- **Feature Geometry**: https://transformer-circuits.pub/2024/feature-geometry/
- **Mechanistic Interpretability**: https://www.lesswrong.com/posts/8FcNS7eft59emt7HS/an-introduction-to-mechanistic-interpretability

---

## Contact

For questions or discussion about this experiment, see the occhio repository issues.
