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


def compute_decoder_jacobians(
    model: ToyModel | AutoEncoderBase,
    embeddings: Tensor,
) -> Tensor:
    """Compute the Jacobian of the decoder output w.r.t. the embedding.

    For each embedding h in the batch, computes ∂x̂/∂h where x̂ = decode(h).
    This gives a matrix where row i is the gradient of reconstructed feature i
    with respect to the embedding.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model. If ToyModel, uses its ae attribute.
    embeddings : Tensor
        Batch of embeddings, shape [B, n_hidden].

    Returns
    -------
    Tensor
        Jacobian matrices, shape [B, n_features, n_hidden].
        jacobians[b, i, :] is ∂x̂_i/∂h for embedding h[b].
    """
    ae = model.ae if isinstance(model, ToyModel) else model

    embeddings = embeddings.detach().requires_grad_(True)

    def decode_fn(h: Tensor) -> Tensor:
        return ae.decode(h.unsqueeze(0)).squeeze(0)

    jacobian_fn = jacrev(decode_fn)

    jacobians = []
    for i in range(embeddings.shape[0]):
        jac = jacobian_fn(embeddings[i])  # [n_features, n_hidden]
        jacobians.append(jac)

    return torch.stack(jacobians)  # [B, n_features, n_hidden]


def gradient_norm_distribution(
    model: ToyModel | AutoEncoderBase,
    inputs: Tensor,
) -> Tensor:
    """Compute per-feature gradient norms at each sample's embedding.

    For each sample, computes the embedding h = encode(x), then for each feature i
    computes ||∂x̂_i/∂h||_2. This measures how sensitive the reconstruction of
    feature i is to local perturbations of the embedding.

    Under the region hypothesis, these norms should be bimodal: near-zero in
    region interiors and large at ReLU boundaries.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model.
    inputs : Tensor
        Batch of input samples, shape [B, n_features].

    Returns
    -------
    Tensor
        Gradient norms, shape [B, n_features].
        gradient_norms[b, i] = ||∂x̂_i/∂h||_2 at embedding h[b].
    """
    ae = model.ae if isinstance(model, ToyModel) else model

    with torch.no_grad():
        embeddings = ae.encode(inputs)

    decoder_jacobians = compute_decoder_jacobians(model, embeddings)
    # decoder_jacobians: [B, n_features, n_hidden]

    # Compute L2 norm over the hidden dimension for each feature
    gradient_norms = decoder_jacobians.norm(dim=2)  # [B, n_features]

    return gradient_norms


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


def perturbation_sensitivity(
    model: ToyModel | AutoEncoderBase,
    feature_idx: int,
    embeddings: Tensor,
    n_directions: int = 100,
    step_size: float = 0.1,
) -> dict:
    """Measure sensitivity of a feature's reconstruction to perturbations in different directions.

    For each embedding where feature_idx is active, apply perturbations in random directions
    and measure how much the reconstruction of feature_idx changes. Under the direction
    hypothesis, sensitivity should align with the feature's weight vector. Under the region
    hypothesis, sensitivity should align with ReLU boundary normals.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model.
    feature_idx : int
        Which feature to analyze.
    embeddings : Tensor
        Batch of embeddings, shape [B, n_hidden].
    n_directions : int
        Number of random perturbation directions to sample.
    step_size : float
        Magnitude of perturbations.

    Returns
    -------
    dict
        Dictionary containing:
        - 'sensitivity_profiles': Tensor [B, n_directions] sensitivity in each direction
        - 'directions': Tensor [n_directions, n_hidden] the random unit directions
        - 'max_sensitivity_directions': Tensor [B, n_hidden] direction of max sensitivity per sample
        - 'feature_direction': Tensor [n_hidden] the feature's decoder weight vector (if linear)
        - 'feature_alignment': Tensor [B] cosine similarity between max-sens direction and feature direction
        - 'mean_feature_alignment': float mean alignment across samples
    """
    ae = model.ae if isinstance(model, ToyModel) else model
    device = embeddings.device
    n_hidden = embeddings.shape[1]
    n_samples = embeddings.shape[0]

    # Generate random unit directions on the sphere
    directions = torch.randn(n_directions, n_hidden, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Get baseline reconstructions
    with torch.no_grad():
        baseline_recon = ae.decode(embeddings)[:, feature_idx]  # [B]

    # Compute sensitivity for each direction
    sensitivity_profiles = torch.zeros(n_samples, n_directions, device=device)

    with torch.no_grad():
        for d_idx in range(n_directions):
            direction = directions[d_idx]  # [n_hidden]
            perturbed = embeddings + step_size * direction  # [B, n_hidden]
            perturbed_recon = ae.decode(perturbed)[:, feature_idx]  # [B]
            sensitivity_profiles[:, d_idx] = torch.abs(perturbed_recon - baseline_recon)

    # Find direction of maximum sensitivity for each sample
    max_indices = sensitivity_profiles.argmax(dim=1)  # [B]
    max_sensitivity_directions = directions[max_indices]  # [B, n_hidden]

    # Get the feature's decoder weight vector (for comparison)
    # For TiedLinearRelu and TiedMLPEncoder, the final decoder layer weight gives the feature direction
    feature_direction = None
    if hasattr(ae, "W"):
        # TiedLinear, TiedLinearRelu: W is [n_hidden, n_features], feature direction is W[:, feature_idx]
        feature_direction = ae.W[:, feature_idx].detach()  # [n_hidden]
    elif hasattr(ae, "encoder_weights"):
        # TiedMLPEncoder: decoder uses reversed encoder weights
        # The last decoder step uses encoder_weights[0] (first encoder layer)
        # Feature direction is the feature_idx-th row of that weight matrix
        feature_direction = ae.encoder_weights[0][
            :, feature_idx
        ].detach()  # [n_hidden of first layer]
        # This might not match n_hidden if there are hidden layers, so check
        if feature_direction.shape[0] != n_hidden:
            # For multi-layer MLP, the "feature direction" concept is less clear
            # Use the decoder Jacobian mean instead
            feature_direction = None

    # If we couldn't get feature direction from weights, compute from decoder Jacobian
    if feature_direction is None:
        decoder_jacs = compute_decoder_jacobians(
            model, embeddings[: min(100, n_samples)]
        )
        # decoder_jacs: [B, n_features, n_hidden]
        feature_direction = decoder_jacs[:, feature_idx, :].mean(dim=0)  # [n_hidden]

    feature_direction = feature_direction / feature_direction.norm().clamp(min=1e-8)

    # Compute alignment between max-sensitivity direction and feature direction
    # Normalize max_sensitivity_directions
    max_sens_normed = max_sensitivity_directions / max_sensitivity_directions.norm(
        dim=1, keepdim=True
    ).clamp(min=1e-8)
    feature_alignment = (max_sens_normed * feature_direction).sum(
        dim=1
    )  # [B] cosine similarity

    return {
        "sensitivity_profiles": sensitivity_profiles,
        "directions": directions,
        "max_sensitivity_directions": max_sensitivity_directions,
        "feature_direction": feature_direction,
        "feature_alignment": feature_alignment,
        "mean_feature_alignment": feature_alignment.abs().mean().item(),
        "max_sensitivities": sensitivity_profiles.max(dim=1).values,
    }


def get_relu_boundary_normals(
    model: ToyModel | AutoEncoderBase,
    embeddings: Tensor,
) -> Tensor | None:
    """Extract ReLU boundary normals at each embedding point for MLP models.

    For each neuron in the decoder that uses ReLU/LeakyReLU, the boundary normal
    is the row of the weight matrix corresponding to that neuron, masked by
    whether the neuron is active at that point.

    Parameters
    ----------
    model : ToyModel or AutoEncoderBase
        The trained model (must be TiedMLPEncoder or similar with LeakyReLU).
    embeddings : Tensor
        Batch of embeddings, shape [B, n_hidden].

    Returns
    -------
    Tensor or None
        Boundary normals, shape [B, n_boundaries, n_hidden] where n_boundaries
        depends on the architecture. Returns None if model has no detectable
        ReLU boundaries in the decoder.
    """
    ae = model.ae if isinstance(model, ToyModel) else model

    # For TiedMLPEncoder, the decoder has LeakyReLU between layers
    if not hasattr(ae, "encoder_weights") or not hasattr(ae, "decoder_biases"):
        return None

    # The decoder walks encoder weights in reverse
    # For a [n_features, hidden, n_hidden] architecture:
    # - Layer 0: z @ encoder_weights[-1] + decoder_biases[0], then LeakyReLU
    # - Layer 1: z @ encoder_weights[-2] + decoder_biases[1], then ReLU (final)

    rev_weights = list(reversed(list(ae.encoder_weights)))
    n_samples = embeddings.shape[0]

    # Track which neurons are at their boundary (pre-activation near zero)
    # and their corresponding normals
    all_boundary_normals = []

    with torch.no_grad():
        z = embeddings
        for i, (w, b) in enumerate(zip(rev_weights, ae.decoder_biases)):
            pre_activation = z @ w + b  # [B, layer_dim]

            if i < len(rev_weights) - 1:
                # LeakyReLU layer - boundary at pre_activation = 0
                # Normal is the row of w that feeds into this neuron
                # For neuron j, the normal in the input space is w[j, :]

                # Identify neurons near boundary (|pre_activation| < threshold)
                near_boundary = pre_activation.abs() < 0.1  # [B, layer_dim]

                # The normals are rows of w (transposed from how we use it)
                # w shape: [output_dim, input_dim], so w[j, :] is normal for neuron j
                # But z @ w means input is z, so the gradient direction is w.T columns
                # Actually, d(z @ w)/dz = w, so the j-th output's gradient w.r.t. z is w[j, :]

                # We need to project back to embedding space through earlier layers
                # For simplicity, just collect the immediate normals
                layer_normals = w.unsqueeze(0).expand(
                    n_samples, -1, -1
                )  # [B, layer_dim, input_dim]

                # Mask by which neurons are near boundary
                # This gives [B, layer_dim, input_dim] where non-boundary neurons are zeroed
                masked_normals = layer_normals * near_boundary.unsqueeze(-1)
                all_boundary_normals.append(masked_normals)

                # Apply LeakyReLU for next layer
                z = torch.where(
                    pre_activation > 0, pre_activation, 0.01 * pre_activation
                )
            else:
                # Final ReLU layer
                near_boundary = pre_activation.abs() < 0.1
                layer_normals = w.unsqueeze(0).expand(n_samples, -1, -1)
                masked_normals = layer_normals * near_boundary.unsqueeze(-1)
                all_boundary_normals.append(masked_normals)

    if not all_boundary_normals:
        return None

    # Concatenate all boundary normals
    # This is approximate - proper computation would require backprop through layers
    return all_boundary_normals  # List of [B, layer_dim, layer_input_dim] tensors
