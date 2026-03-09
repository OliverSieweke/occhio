"""Analysis tools for investigating manifold representations in autoencoders.

This module provides Jacobian-based analysis for measuring how feature encoding
directions vary across different input contexts.
"""

from .jacobian import (
    compute_feature_jacobians,
    angular_variance,
    jacobian_pca,
    direction_vs_context,
    compute_all_feature_jacobians,
)

__all__ = [
    "compute_feature_jacobians",
    "angular_variance",
    "jacobian_pca",
    "direction_vs_context",
    "compute_all_feature_jacobians",
]
