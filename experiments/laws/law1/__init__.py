# ABOUTME: Law 1 experiment package for correlation-interference relationship
# ABOUTME: Provides model factories, metric extraction, and analysis for Law 1 validation

"""Law 1: The Correlation-Interference Law

Validates that optimal feature interference increases monotonically with feature
correlation in sparse autoencoders. Uses Toy Models of Superposition with
CorrelatedPairs and AnticorrelatedPairs distributions.

Usage:
    python experiment.py

    Or import factories and functions for custom analysis:
    from law1.experiment import (
        create_model_experiment_a,
        create_model_experiment_b,
        create_model_experiment_c,
        extract_metrics,
        extract_metrics_anticorr,
        run_experiments,
        create_figures,
        print_summary,
    )
"""

__version__ = "0.1.0"
