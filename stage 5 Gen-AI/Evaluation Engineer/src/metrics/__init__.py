"""Evaluation Engineer Metrics Package."""
from .continuous_metrics import (
    compute_mean_difference, compute_median_difference, compute_std_difference,
    compute_quantile_differences, compute_wasserstein_distance, compute_ks_statistic
)
from .categorical_metrics import (
    compute_frequency_difference, compute_total_variation_distance,
    compute_jensen_shannon_divergence, compute_category_coverage
)
from .distribution_metrics import (
    check_cooccurrence_constraints, compute_joint_frequency_divergence
)
from .confidence_metrics import (
    compute_confidence_spread, is_high_confidence_error, compute_confidence_by_outcome
)
from .disagreement_metrics import (
    calculate_sensitivity_drop, calculate_discordance_rate, build_pairwise_agreement_matrix
)

__all__ = [
    "compute_mean_difference", "compute_median_difference", "compute_std_difference",
    "compute_quantile_differences", "compute_wasserstein_distance", "compute_ks_statistic",
    "compute_frequency_difference", "compute_total_variation_distance",
    "compute_jensen_shannon_divergence", "compute_category_coverage",
    "check_cooccurrence_constraints", "compute_joint_frequency_divergence",
    "compute_confidence_spread", "is_high_confidence_error", "compute_confidence_by_outcome",
    "calculate_sensitivity_drop", "calculate_discordance_rate", "build_pairwise_agreement_matrix"
]
