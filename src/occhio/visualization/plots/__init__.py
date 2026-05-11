from occhio.visualization.plots.embedding import EmbeddingPlot, plot_embedding
from occhio.visualization.plots.experimental.sae_classification_metric import (
    SAEClassificationMetric,
    SAEClassificationMetricPlot,
    SAEMetricsComparisonPlot,
)
from occhio.visualization.plots.experimental.sae_classification_metrics import (
    SAEClassificationMetricsPlot,
)
from occhio.visualization.plots.experimental.sae_f1_vs_l0 import (
    SAEF1vsL0Plot,
    plot_sae_f1_vs_l0,
)
from occhio.visualization.plots.experimental.sae_metrics_summary import (
    DiagnosticTablePlot,
    InterpretabilityTablePlot,
    PerformanceFidelityTablePlot,
    plot_sae_metrics_summary,
)
from occhio.visualization.plots.experimental.sae_benchmark_table import (
    SAEBenchmarkTablePlot,
    plot_sae_benchmark_table,
)
from occhio.visualization.plots.experimental.sae_metrics_table import (
    SAECoreMetricsTablePlot,
    SAEMetricsTablePlot,
    SAESparsityMetricsTablePlot,
    plot_sae_core_metrics_table,
    plot_sae_metrics_table,
    plot_sae_sparsity_metrics_table,
)
from occhio.visualization.plots.experimental.sae_one_hot_to_latent_heatmap import (
    SAEOneHotToLatentHeatmapPlot,
    plot_one_hot_to_latent_heatmap,
)
from occhio.visualization.plots.experimental.sae_per_feature_f1 import (
    SAEPerFeatureF1DistributionPlot,
    SAEPerFeatureF1Plot,
    plot_sae_per_feature_f1,
    plot_sae_per_feature_f1_distribution,
)
from occhio.visualization.plots.feature_representation import (
    FeatureDimensionalityByIndexPlot,
    FeatureDimensionalityDistributionOverlayPlot,
    FeatureDimensionalityDistributionPlot,
    FeatureInterferenceByIndexPlot,
    FeatureInterferenceDistributionOverlayPlot,
    FeatureInterferenceDistributionPlot,
    FeatureNormByIndexPlot,
    FeatureNormDistributionOverlayPlot,
    FeatureNormDistributionPlot,
    SuperpositionIndicatorPlot,
    plot_feature_dimensionality_by_index,
    plot_feature_dimensionality_distribution,
    plot_feature_interference_by_index,
    plot_feature_interference_distribution,
    plot_feature_norm_by_index,
    plot_feature_norm_distribution,
    plot_feature_representation,
    plot_feature_representation_overlay,
    plot_superposition_indicator,
)
from occhio.visualization.plots.representation import (
    RepresentationPlot,
    plot_representation,
)
from occhio.visualization.plots.sae_feature_similarity import (
    SAEFeatureSimilarityPlot,
    plot_sae_feature_similarity,
)
from occhio.visualization.plots.phase_change import (
    PhaseChangePlot,
    plot_phase_change,
    plot_phase_change_multi,
)
from occhio.visualization.plots.geometry import (
    GeometryPlot,
    GeometryPlotComponent,
    FeatureGeometryPlot,
    plot_geometry,
    plot_feature_geometry,
    plot_feature_geometry_3d,
)
from occhio.visualization.plots.dynamic import plot_dynamic_scatter
from occhio.visualization.plots.compute import (
    DecodePlanePlot,
    plot_decode_plane,
)

__all__ = [
    "RepresentationPlot",
    "plot_representation",
    "EmbeddingPlot",
    "plot_embedding",
    "FeatureDimensionalityByIndexPlot",
    "FeatureDimensionalityDistributionPlot",
    "FeatureDimensionalityDistributionOverlayPlot",
    "FeatureNormByIndexPlot",
    "FeatureNormDistributionPlot",
    "FeatureNormDistributionOverlayPlot",
    "FeatureInterferenceByIndexPlot",
    "FeatureInterferenceDistributionPlot",
    "FeatureInterferenceDistributionOverlayPlot",
    "SuperpositionIndicatorPlot",
    "plot_feature_dimensionality_by_index",
    "plot_feature_dimensionality_distribution",
    "plot_feature_norm_by_index",
    "plot_feature_norm_distribution",
    "plot_feature_interference_by_index",
    "plot_feature_interference_distribution",
    "plot_feature_representation",
    "plot_feature_representation_overlay",
    "plot_superposition_indicator",
    "SAEClassificationMetric",
    "SAEClassificationMetricPlot",
    "SAEClassificationMetricsPlot",
    "SAEMetricsComparisonPlot",
    "SAEMetricsTablePlot",
    "SAECoreMetricsTablePlot",
    "SAESparsityMetricsTablePlot",
    "plot_sae_metrics_table",
    "plot_sae_core_metrics_table",
    "plot_sae_sparsity_metrics_table",
    "SAEFeatureSimilarityPlot",
    "plot_sae_feature_similarity",
    "SAEOneHotToLatentHeatmapPlot",
    "plot_one_hot_to_latent_heatmap",
    "SAEF1vsL0Plot",
    "plot_sae_f1_vs_l0",
    "PerformanceFidelityTablePlot",
    "InterpretabilityTablePlot",
    "DiagnosticTablePlot",
    "plot_sae_metrics_summary",
    "SAEPerFeatureF1Plot",
    "SAEPerFeatureF1DistributionPlot",
    "plot_sae_per_feature_f1",
    "plot_sae_per_feature_f1_distribution",
    "SAEBenchmarkTablePlot",
    "plot_sae_benchmark_table",
    # Ported from v1
    "PhaseChangePlot",
    "plot_phase_change",
    "plot_phase_change_multi",
    "GeometryPlot",
    "GeometryPlotComponent",
    "FeatureGeometryPlot",
    "plot_geometry",
    "plot_feature_geometry",
    "plot_feature_geometry_3d",
    "plot_dynamic_scatter",
    "DecodePlanePlot",
    "plot_decode_plane",
]
