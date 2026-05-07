# Distributions

Distributions define how ground-truth feature activations are sampled. Every
distribution produces tensors of shape `(batch_size, n_features)`.

## Overview

### Sparse Distributions

| Class | Description |
|---|---|
| {class}`~occhio.distributions.SparseUniform` | Each feature fires independently with probability `p_active`; values ~ Uniform(0, 1). The standard baseline from Elhage et al. |
| {class}`~occhio.distributions.SparseExponential` | Like SparseUniform but values ~ Exponential(scale). Useful for heavy-tailed feature magnitudes. |
| {class}`~occhio.distributions.SingleUniform` | Exactly one feature active per sample (one-hot sparsity). |

### Correlated Distributions

| Class | Description |
|---|---|
| {class}`~occhio.distributions.CorrelatedPairs` | Paired features where one co-activates with the other at rate `p_follow`. |
| {class}`~occhio.distributions.GaussianCorrelated` | Gaussian copula correlations via a user-specified correlation matrix. |
| {class}`~occhio.distributions.HierarchicalPairs` | Paired features with hierarchical parent-child activation dependency. |
| {class}`~occhio.distributions.ScaledHierarchicalPairs` | HierarchicalPairs with magnitude coupling between parent and child. |
| {class}`~occhio.distributions.AnticorrelatedPairs` | Paired features that are anti-correlated (mutually exclusive activation). |

### Structured / Graph Distributions

| Class | Description |
|---|---|
| {class}`~occhio.distributions.HierarchicalSparse` | Tree-structured hierarchy where activation cascades from root to leaves. |
| {class}`~occhio.distributions.DAGDistribution` | DAG-structured activation propagation via Erdos-Renyi graphs. |
| {class}`~occhio.distributions.DAGBayesianPropagation` | DAG with Bayesian (soft) propagation instead of binary gating. |
| {class}`~occhio.distributions.DAGRandomWalkToRoot` | DAG sampled by random walks to root nodes. |
| {class}`~occhio.distributions.PreferentialAttachment` | Scale-free DAG generated via preferential attachment. |

### Relational Distributions

| Class | Description |
|---|---|
| {class}`~occhio.distributions.RelationalSimple` | Two matrix bindings (identity + random O(n)) for relational composition. |
| {class}`~occhio.distributions.MultiRelational` | Multiple relational bindings with configurable relation count. |

### Manifold Distributions

| Class | Description |
|---|---|
| {class}`~occhio.distributions.SphericalDistribution` | Features on a sphere with cosine-bump activation from a random direction. |
| {class}`~occhio.distributions.ToricDistribution` | Features on a flat torus with cosine-bump activation. |
| {class}`~occhio.distributions.HypercubeDistribution` | Features on a hypercube grid with tent-bump activation. |
| {class}`~occhio.distributions.SimplexDistribution` | Groups of features forming sparse simplices (Dirichlet activations). |
| {class}`~occhio.distributions.SimplicialComplexDistribution` | Simplices with shared vertices (glued faces/edges). |

### Composite and External

| Class | Description |
|---|---|
| {class}`~occhio.distributions.DistributionStack` | Concatenates multiple distributions along the feature dimension. Supports independent, sparse, and single-active modes. |
| {class}`~occhio.distributions.HuggingFaceDistribution` | Serves pre-generated samples from a HuggingFace Hub dataset. |
| {class}`~occhio.distributions.SyntheticDataModel` | [SynthSAEBench](https://arxiv.org/abs/2602.14687) data generator with Gaussian copula correlations and hierarchical dependencies. |

## When to Use Each

- **Baseline experiments**: {class}`~occhio.distributions.SparseUniform` -- the
  standard from the original superposition paper.
- **Feature correlations**: {class}`~occhio.distributions.CorrelatedPairs` or
  {class}`~occhio.distributions.GaussianCorrelated` -- study how correlated
  features affect superposition.
- **Hierarchical structure**: {class}`~occhio.distributions.HierarchicalSparse`
  or {class}`~occhio.distributions.HierarchicalPairs` -- model parent-child
  feature dependencies.
- **Geometric structure**: {class}`~occhio.distributions.SphericalDistribution`,
  {class}`~occhio.distributions.ToricDistribution`, or
  {class}`~occhio.distributions.HypercubeDistribution` -- study how manifold
  topology affects representation.
- **Relational composition**: {class}`~occhio.distributions.RelationalSimple` --
  test binding via orthogonal matrices.
- **Real data**: {class}`~occhio.distributions.HuggingFaceDistribution` -- load
  activations from a pretrained model.
- **Mixed feature types**: {class}`~occhio.distributions.DistributionStack` --
  combine different distributions into one feature space.

## Common Parameters

All distributions accept:

- `n_features` (int) -- dimensionality of the sample space
- `device` (str | torch.device | None) -- torch device for generated tensors
- `generator` (torch.Generator | None) -- for reproducible sampling

Most sparse distributions also accept `p_active` (float or per-feature tensor)
controlling sparsity.

## API Reference

See the full {doc}`api` for detailed class documentation.
