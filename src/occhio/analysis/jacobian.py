"""Jacobian-based analysis for manifold representations.

This module provides tools to compute and analyze how feature encoding directions
vary across input contexts, enabling measurement of manifold structure in autoencoders.

Key metrics:
- Angular variance: Measures how much a feature's encoding direction varies (0 = linear, ~1 = maximally nonlinear)
- Jacobian PCA: Reveals the intrinsic dimensionality of the feature direction manifold
- Direction vs context: Identifies which co-active features cause direction rotations
"""

from typing import Tuple
import torch
from torch import Tensor
from torch.func import jacrev, vmap

from ..autoencoder import AutoEncoderBase
from ..toy_model import ToyModel


def compute_feature_jacobians(
    model: ToyModel | AutoEncoderBase,
    feature_idx: int,
    inputs: Tensor,
) -> Tensor:
    """Compute the Jacobian of the encoder output w.r.t. a specific input feature.

    For each input x in the batch, computes ∂h/∂x_i where h = encode(x) and i = feature_idx.
    This vector is the local encoding direction for feature i at input x.

    For linear encoders (e.g., TiedLinear), this should return identical vectors
    (equal to the i-th column of W) for all inputs. For nonlinear encoders,
    directions may vary depending on the input context.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model. If ToyModel, uses its ae attribute.
    feature_idx : int
        Which input feature to compute the Jacobian for.
    inputs : Tensor
        Batch of input samples, shape [B, n_features].

    Returns
    -------
    Tensor
        Jacobian vectors, shape [B, n_hidden]. Each row is ∂h/∂x_i for one input.

    Example
    -------
    >>> ae = MLPAutoencoder(n_features=10, n_hidden=3)
    >>> x = torch.randn(100, 10)
    >>> jacobians = compute_feature_jacobians(ae, feature_idx=0, inputs=x)
    >>> jacobians.shape
    torch.Size([100, 3])
    """
    # Extract autoencoder
    ae = model.ae if isinstance(model, ToyModel) else model

    # Ensure inputs require grad for Jacobian computation
    inputs = inputs.detach().requires_grad_(True)

    # Define function for a single input that returns encode output
    def encode_fn(x: Tensor) -> Tensor:
        return ae.encode(x.unsqueeze(0)).squeeze(0)

    # Compute Jacobian for each input using vmap
    # jacrev gives us [n_hidden, n_features] per input
    jacobian_fn = jacrev(encode_fn)

    jacobians = []
    for i in range(inputs.shape[0]):
        jac = jacobian_fn(inputs[i])  # [n_hidden, n_features]
        jacobians.append(jac[:, feature_idx])  # Extract column for feature_idx

    return torch.stack(jacobians)  # [B, n_hidden]


def compute_all_feature_jacobians(
    model: ToyModel | AutoEncoderBase,
    inputs: Tensor,
) -> Tensor:
    """Compute full Jacobians of the encoder for all features at once.

    More efficient than calling compute_feature_jacobians repeatedly when
    you need Jacobians for multiple features.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model.
    inputs : Tensor
        Batch of input samples, shape [B, n_features].

    Returns
    -------
    Tensor
        Full Jacobian tensors, shape [B, n_hidden, n_features].
        jacobians[b, :, i] is the encoding direction for feature i at input x[b].
    """
    ae = model.ae if isinstance(model, ToyModel) else model

    inputs = inputs.detach().requires_grad_(True)

    def encode_fn(x: Tensor) -> Tensor:
        return ae.encode(x.unsqueeze(0)).squeeze(0)

    jacobian_fn = jacrev(encode_fn)

    jacobians = []
    for i in range(inputs.shape[0]):
        jac = jacobian_fn(inputs[i])  # [n_hidden, n_features]
        jacobians.append(jac)

    return torch.stack(jacobians)  # [B, n_hidden, n_features]


def angular_variance(jacobians: Tensor, eps: float = 1e-8) -> float:
    """Compute angular variance of Jacobian vectors.

    Measures how much the encoding direction for a feature varies across
    different input contexts. Based on circular variance from directional statistics.

    The metric normalizes each Jacobian vector, computes the mean direction,
    and returns 1 - ||mean_direction||. This ranges from:
    - 0: All directions identical (perfectly linear encoding)
    - ~1: Directions uniformly spread (maximally nonlinear)

    Parameters
    ----------
    jacobians : Tensor
        Jacobian vectors from compute_feature_jacobians, shape [N, n_hidden].
    eps : float
        Small constant for numerical stability in normalization.

    Returns
    -------
    float
        Angular variance in [0, 1].

    Example
    -------
    >>> # For a linear encoder, angular variance should be ~0
    >>> ae = TiedLinear(n_features=10, n_hidden=3)
    >>> x = torch.randn(100, 10)
    >>> jacs = compute_feature_jacobians(ae, feature_idx=0, inputs=x)
    >>> av = angular_variance(jacs)
    >>> print(f"Angular variance: {av:.6f}")  # Should be very close to 0
    """
    # Normalize each Jacobian vector
    norms = jacobians.norm(dim=1, keepdim=True).clamp(min=eps)
    normed = jacobians / norms

    # Compute mean direction
    mean_dir = normed.mean(dim=0)

    # Angular variance = 1 - ||mean_direction||
    return 1.0 - mean_dir.norm().item()


def jacobian_pca(
    jacobians: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Run PCA on normalized Jacobian vectors.

    The eigenvalue spectrum reveals the intrinsic dimensionality of the
    "feature direction manifold":
    - 1 dominant eigenvalue: feature direction is essentially fixed (linear)
    - k significant eigenvalues: direction varies on a k-dimensional submanifold
    - Flat spectrum: directions are essentially random (pathological)

    Parameters
    ----------
    jacobians : Tensor
        Jacobian vectors from compute_feature_jacobians, shape [N, n_hidden].
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    eigenvalues : Tensor
        Eigenvalues sorted in descending order, shape [min(N-1, n_hidden)].
        Represents variance explained by each principal component.
    eigenvectors : Tensor
        Principal components, shape [n_hidden, min(N-1, n_hidden)].
        Column i is the i-th principal direction.

    Example
    -------
    >>> jacs = compute_feature_jacobians(ae, feature_idx=0, inputs=x)
    >>> eigenvalues, eigenvectors = jacobian_pca(jacs)
    >>> # Check if first eigenvalue dominates (indicates linear encoding)
    >>> print(f"Variance ratio of PC1: {eigenvalues[0] / eigenvalues.sum():.4f}")
    """
    # Normalize each Jacobian vector
    norms = jacobians.norm(dim=1, keepdim=True).clamp(min=eps)
    normed = jacobians / norms

    # Center the data
    centered = normed - normed.mean(dim=0, keepdim=True)

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Eigenvalues are S^2 / (N - 1)
    n_samples = jacobians.shape[0]
    eigenvalues = S**2 / (n_samples - 1)

    # Eigenvectors are rows of Vh (transposed to get columns)
    eigenvectors = Vh.T

    return eigenvalues.detach(), eigenvectors.detach()


def direction_vs_context(
    model: ToyModel | AutoEncoderBase,
    feature_idx: int,
    inputs: Tensor,
    jacobians: Tensor | None = None,
    eps: float = 1e-8,
) -> dict:
    """Analyze how co-active features affect a feature's encoding direction.

    For feature i, computes the correlation between each other feature j's
    activation magnitude and the Jacobian direction for feature i. This reveals
    which co-active features cause the encoding direction to rotate.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model.
    feature_idx : int
        Which feature's direction to analyze.
    inputs : Tensor
        Batch of input samples, shape [N, n_features].
    jacobians : Tensor, optional
        Pre-computed Jacobians from compute_feature_jacobians. If None,
        will be computed automatically.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    dict
        Dictionary containing:
        - 'correlations': Tensor [n_features, n_hidden] where entry [j, k] is
          the correlation between feature j's activation and Jacobian component k.
        - 'correlation_magnitudes': Tensor [n_features] giving the overall
          correlation magnitude between each feature j and direction changes.
        - 'most_influential': List of (feature_idx, magnitude) tuples for the
          top 5 most influential co-active features.
        - 'feature_idx': The analyzed feature index.

    Example
    -------
    >>> result = direction_vs_context(model, feature_idx=3, inputs=x)
    >>> print("Most influential features on feature 3's direction:")
    >>> for feat, mag in result["most_influential"]:
    ...     print(f"  Feature {feat}: {mag:.4f}")
    """
    ae = model.ae if isinstance(model, ToyModel) else model

    # Compute Jacobians if not provided
    if jacobians is None:
        jacobians = compute_feature_jacobians(model, feature_idx, inputs)

    # Normalize Jacobians
    jac_norms = jacobians.norm(dim=1, keepdim=True).clamp(min=eps)
    normed_jacs = jacobians / jac_norms

    # Center Jacobians for correlation computation
    jacs_centered = normed_jacs - normed_jacs.mean(dim=0, keepdim=True)

    # Get feature activations and center them
    # inputs: [N, n_features]
    inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)

    n_samples = inputs.shape[0]
    n_features = inputs.shape[1]
    n_hidden = jacobians.shape[1]

    # Compute correlations: for each feature j and each Jacobian component k
    # correlation[j, k] = cov(inputs[:, j], normed_jacs[:, k]) / (std_j * std_k)
    correlations = torch.zeros(n_features, n_hidden, device=inputs.device)

    for j in range(n_features):
        if j == feature_idx:
            continue  # Skip self-correlation

        feat_std = inputs_centered[:, j].std().clamp(min=eps)
        for k in range(n_hidden):
            jac_std = jacs_centered[:, k].std().clamp(min=eps)
            cov = (inputs_centered[:, j] * jacs_centered[:, k]).mean()
            correlations[j, k] = cov / (feat_std * jac_std)

    # Compute overall correlation magnitude per feature
    # Using L2 norm of correlation vector across Jacobian components
    correlation_magnitudes = correlations.norm(dim=1)

    # Find most influential features (excluding self)
    magnitudes_copy = correlation_magnitudes.clone()
    magnitudes_copy[feature_idx] = -float("inf")  # Exclude self
    top_k = min(5, n_features - 1)
    top_values, top_indices = magnitudes_copy.topk(top_k)

    most_influential = [
        (idx.item(), val.item()) for idx, val in zip(top_indices, top_values)
    ]

    return {
        "correlations": correlations,
        "correlation_magnitudes": correlation_magnitudes,
        "most_influential": most_influential,
        "feature_idx": feature_idx,
    }
