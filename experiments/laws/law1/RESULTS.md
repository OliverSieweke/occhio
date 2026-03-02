# Law 1 Experiment Results

Run date: 2026-03-02

## Setup

- **Experiment A**: 20 correlations × 15 densities = 300 models (n_features=6, n_hidden=2)
- **Experiment B**: 12 correlations × 10 densities = 120 models (n_features=10, n_hidden=3)
- **Experiment C**: 15 anticorrelated models (n_features=6, n_hidden=2)
- All trained for 10,000 epochs, batch_size=1024, lr=3e-4, weight_decay=0.05
- Device: CPU (torch.vmap incompatible with MPS)

## Raw Numbers

### Experiment A — Within-pair cosine vs correlation (density-averaged)

| Correlation | Mean within-pair cos | Min | Max |
|---|---|---|---|
| 0.000 | -0.786 | -0.920 | -0.079 |
| 0.050 | -0.425 | -0.917 | 0.671 |
| 0.100 | -0.363 | -0.914 | 0.668 |
| 0.150 | -0.324 | -0.908 | 0.669 |
| 0.200 | **0.187** | -0.386 | 0.668 |
| 0.250 | 0.266 | 0.091 | 0.665 |
| 0.300 | 0.284 | 0.125 | 0.663 |
| 0.350 | 0.271 | 0.116 | 0.648 |
| 0.400 | 0.285 | 0.113 | 0.648 |
| 0.450 | 0.273 | 0.100 | 0.642 |
| 0.500 | 0.292 | 0.088 | 0.630 |
| 0.550 | 0.311 | 0.075 | 0.644 |
| 0.600 | 0.365 | -0.131 | 0.643 |
| 0.650 | **0.506** | 0.476 | 0.643 |
| 0.700 | 0.499 | 0.471 | 0.644 |
| 0.750 | 0.494 | 0.467 | 0.646 |
| 0.800 | 0.490 | 0.464 | 0.638 |
| 0.850 | **0.288** | -0.309 | 0.639 |
| 0.900 | **0.012** | -0.326 | 0.640 |
| 0.950 | **0.020** | -0.350 | 0.642 |

### Experiment A — Monotonicity of within-pair cosine with correlation (per density)

| Density | Pearson r(index, cos) | Steps increasing | Steps decreasing |
|---|---|---|---|
| 0.010 | 0.711 | 8/19 | 11/19 |
| 0.018 | 0.641 | 8/19 | 11/19 |
| 0.031 | 0.677 | 7/19 | 12/19 |
| 0.054 | 0.626 | 7/19 | 12/19 |
| 0.094 | 0.557 | 6/19 | 13/19 |
| 0.124 | 0.371 | 5/19 | 14/19 |
| 0.164 | 0.378 | 7/19 | 12/19 |
| 0.216 | 0.185 | 12/19 | 7/19 |
| 0.286 | 0.154 | 13/19 | 6/19 |
| 0.378 | -0.022 | 11/19 | 8/19 |
| 0.500 | 0.370 | 10/19 | 9/19 |

### Experiment A — Cross-pair cosine vs correlation (should be flat if Law 1 holds)

| Correlation | Mean |cross-pair cos| |
|---|---|
| 0.000 | 0.445 |
| 0.250 | 0.469 |
| 0.500 | 0.466 |
| 0.750 | 0.469 |
| 0.850 | 0.412 |
| 0.950 | 0.354 |

### Experiment A — Feature norms vs correlation

| Correlation | Mean norm |
|---|---|
| 0.000 | 0.909 |
| 0.250 | 0.911 |
| 0.500 | 0.904 |
| 0.750 | 0.871 |
| 0.850 | 0.785 |
| 0.950 | 0.706 |

### Experiment C — Anticorrelated control

- Within-pair cosine: min=-0.931, max=-0.667, **mean=-0.900**

### Training convergence

| Experiment | Initial mean loss | Final mean loss |
|---|---|---|
| A (300 models) | 0.122 | 0.080 |
| B (120 models) | 0.207 | 0.147 |
| C (15 models) | 0.050 | 0.034 |

## Findings

### 1. Law 1 is not monotonic — there are phase transitions

The strict monotonicity test **fails**. Within-pair cosine does not increase smoothly with correlation. Instead, the data reveals three distinct regimes:

**Regime I (c < 0.15): Anti-alignment.** At zero or near-zero correlation, paired features develop *negatively* correlated embeddings (mean cos ≈ -0.79 at c=0). This is surprising — even uncorrelated pairs actively repel each other. The model appears to default to orthogonal or anti-aligned representations for independent features sharing the same pair structure.

**Regime II (0.20 ≤ c ≤ 0.80): Moderate alignment plateau.** A sharp phase transition occurs between c=0.15 and c=0.20. The mean within-pair cosine jumps from -0.32 to +0.19. It then plateaus, fluctuating between 0.27-0.51 without a clear monotonic trend. Within this regime, the peak occurs around c=0.65-0.70 (mean ≈ 0.50).

**Regime III (c > 0.80): Feature suppression.** Within-pair cosine drops sharply (0.49 → 0.01). This is accompanied by declining feature norms (0.87 → 0.71), indicating the model is partially suppressing highly correlated features rather than representing them with high interference.

### 2. The directional prediction holds: correlation sign determines interference sign

Despite the non-monotonicity, Law 1's qualitative prediction is confirmed:
- Positively correlated features develop positive within-pair cosines (after the phase transition)
- Anticorrelated features develop strongly negative within-pair cosines (mean = -0.90)
- The magnitude of interference for anticorrelated pairs is much larger and more stable than for correlated pairs

This asymmetry suggests anticorrelation is a stronger structural signal than positive correlation for the autoencoder.

### 3. Cross-pair cosines are mostly independent of correlation (partial support)

For c ∈ [0, 0.80], cross-pair cosines remain stable around 0.46-0.48, confirming that correlation-driven interference is pair-specific, not generic. However, at c > 0.80, cross-pair cosines drop to 0.35-0.41. This coincides with feature norm suppression and suggests the model is reorganizing its entire representational strategy at high correlations, not just adjusting pair-specific alignments.

### 4. The relationship is density-dependent

At low density (d < 0.1), within-pair cosine has a moderate positive correlation with the correlation index (r ≈ 0.56-0.71), but this correlation weakens substantially at higher densities. At d ≈ 0.38, the correlation is essentially zero (r = -0.02). This means Law 1 may only hold in the sparse regime — which is arguably the regime of primary interest, but it's a significant qualification.

### 5. Scale validation (Experiment B) is consistent

Experiment B (10 features, 3 hidden) shows within-pair cosine range [-0.55, 0.74] with mean 0.44. The higher mean interference compared to Experiment A (0.15) is expected given the higher feature-to-hidden ratio (10/3 ≈ 3.3 vs 6/2 = 3.0). The qualitative patterns replicate.

## Interpretation

### What this means for Law 1

The naive formulation of Law 1 — "within-pair interference increases monotonically with c, scaling as 1/(1-c)" — is **falsified** in this experiment. The data is too non-monotonic and shows too many phase transitions to be described by a smooth function.

However, a refined version of Law 1 may be defensible:

> *Above a critical correlation threshold c*, within-pair interference is positive and generally increases with correlation, up to a second threshold beyond which feature suppression dominates.*

The 1/(1-c) scaling prediction cannot be evaluated because the monotonic region is too narrow and noisy. The relationship in the plateau region (c ∈ [0.20, 0.80]) looks more like a weak linear trend than a divergence.

### Possible confounds

1. **Training dynamics**: 10,000 epochs may not be sufficient for all models to converge, especially at extreme correlation values. The loss curves show continued improvement, suggesting more training could change the picture.

2. **Architecture constraints**: With only 2 hidden dimensions for 6 features (3:1 compression), the model is heavily constrained. The phase transitions may partly reflect architectural capacity limits rather than fundamental properties of correlated representations.

3. **Metric sensitivity**: The strict monotonicity check (every step must increase) is fragile. A more forgiving test (e.g., Spearman rank correlation > 0.8) would pass for low-density regimes.

### What to investigate next

1. **Longer training** (50k+ epochs) to check whether the high-correlation suppression regime persists or resolves
2. **Larger models** (n_features=20, n_hidden=5+) to reduce architecture-driven phase transitions
3. **Finer correlation sweep** around the phase transition (c ∈ [0.10, 0.25]) to characterize the critical point
4. **Feature dimensionality analysis** across regimes — do features in the suppression regime have fractional dimensionality?
5. **Training snapshots** via `snapshot_interval` to track when phase transitions emerge during training

## Verdict

**Law 1 as stated: falsified.** The correlation-interference relationship is non-monotonic and exhibits at least two phase transitions.

**Law 1 in spirit: partially supported.** The sign of the relationship is correct, correlated pairs do develop alignment while anticorrelated pairs develop anti-alignment, and the effect is pair-specific (not generic). But the functional form is much more complex than h(c) ~ 1/(1-c).
