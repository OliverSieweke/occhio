"""Deep audit tests for the visualization module.

Covers: import correctness, render smoke tests, data flow validation,
edge cases, CompositePlot/Span layout, and dead code detection.
"""

import plotly.graph_objects as go
import pytest
import torch

from occhio.autoencoders.tied import TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.model_grid import Axis, ModelGrid
from occhio.toy_model import ToyModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_model():
    """Minimal ToyModel: 5 features, 2D hidden, briefly trained."""
    dist = SparseUniform(5, p_active=0.5)
    ae = TiedLinearRelu(5, 2)
    tm = ToyModel(dist, ae)
    tm.fit(n_epochs=5, batch_size=64, verbose=False)
    return tm


@pytest.fixture()
def model_grid_1d(small_model):
    """1D ModelGrid with 3 copies of the same model (different sparsity values)."""
    models = []
    for _ in range(3):
        dist = SparseUniform(5, p_active=0.5)
        ae = TiedLinearRelu(5, 2)
        tm = ToyModel(dist, ae)
        tm.fit(n_epochs=5, batch_size=64, verbose=False)
        models.append(tm)
    return ModelGrid(
        create_model=lambda params: None,
        axes=[Axis("Sparsity", [0.1, 0.3, 0.5])],
        broadcast_samples=False,
        _models=__import__("numpy").array(models, dtype=object),
    )


@pytest.fixture()
def single_feature_model():
    """Edge case: single feature model."""
    dist = SparseUniform(1, p_active=0.5)
    ae = TiedLinearRelu(1, 1)
    tm = ToyModel(dist, ae)
    tm.fit(n_epochs=5, batch_size=64, verbose=False)
    return tm


# ---------------------------------------------------------------------------
# 1. Import Audit
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all visualization modules can be imported."""

    def test_import_visualization_star(self):
        from occhio.visualization import (
            GeometryPlotComponent,
            export_figure,
            plot_decode_plane,
            plot_dynamic_scatter,
            plot_embedding,
            plot_feature_geometry,
            plot_feature_geometry_3d,
            plot_geometry,
            plot_phase_change,
            plot_phase_change_multi,
            plot_representation,
        )

        # All should be importable (no circular import, no missing deps)
        assert callable(plot_embedding)
        assert callable(plot_representation)
        assert callable(export_figure)

    def test_import_visualization_classes(self):
        from occhio.visualization import (
            EmbeddingPlot,
            FeatureDimensionalityByIndexPlot,
            FeatureNormByIndexPlot,
            RepresentationPlot,
            SAEFeatureSimilarityPlot,
            SuperpositionIndicatorPlot,
        )
        from occhio.visualization.core import CompositePlot

        assert RepresentationPlot is not None
        assert EmbeddingPlot is not None
        assert CompositePlot is not None

    def test_import_core_submodules(self):
        from occhio.visualization.core import (
            CompositePlot,
            PlotOrchestrator,
            PlotRenderer,
            SinglePlot,
            Span,
        )

        assert issubclass(SinglePlot, PlotRenderer)
        assert issubclass(SinglePlot, PlotOrchestrator)

    def test_import_experimental_plots(self):
        from occhio.visualization.plots.experimental.sae_benchmark_table import (
            SAEBenchmarkTablePlot,
        )
        from occhio.visualization.plots.experimental.sae_classification_metric import (
            SAEClassificationMetricPlot,
        )
        from occhio.visualization.plots.experimental.sae_classification_metrics import (
            SAEClassificationMetricsPlot,
        )
        from occhio.visualization.plots.experimental.sae_f1_vs_l0 import (
            SAEF1vsL0Plot,
        )
        from occhio.visualization.plots.experimental.sae_metrics_summary import (
            DiagnosticTablePlot,
            InterpretabilityTablePlot,
            PerformanceFidelityTablePlot,
        )
        from occhio.visualization.plots.experimental.sae_metrics_table import (
            SAECoreMetricsTablePlot,
            SAEMetricsTablePlot,
            SAESparsityMetricsTablePlot,
        )
        from occhio.visualization.plots.experimental.sae_one_hot_to_latent_heatmap import (
            SAEOneHotToLatentHeatmapPlot,
        )
        from occhio.visualization.plots.experimental.sae_per_feature_f1 import (
            SAEPerFeatureF1DistributionPlot,
            SAEPerFeatureF1Plot,
        )

        assert SAEBenchmarkTablePlot is not None


# ---------------------------------------------------------------------------
# 2. Plot Rendering
# ---------------------------------------------------------------------------


class TestVisualization2Rendering:
    """Smoke tests: construct each plot and call render() with minimal valid data."""

    def test_representation_plot_renders(self, small_model):
        from occhio.visualization import RepresentationPlot

        plot = RepresentationPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0, "RepresentationPlot produced no traces"

    def test_embedding_plot_renders(self, small_model):
        from occhio.visualization import EmbeddingPlot

        plot = EmbeddingPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        # EmbeddingPlot adds invisible scatter markers for hover
        assert len(fig.data) > 0, "EmbeddingPlot produced no traces"

    def test_feature_norm_by_index_renders(self, small_model):
        from occhio.visualization import FeatureNormByIndexPlot

        plot = FeatureNormByIndexPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_norm_distribution_renders(self, small_model):
        from occhio.visualization import FeatureNormDistributionPlot

        plot = FeatureNormDistributionPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_dimensionality_by_index_renders(self, small_model):
        from occhio.visualization import FeatureDimensionalityByIndexPlot

        plot = FeatureDimensionalityByIndexPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_dimensionality_distribution_renders(self, small_model):
        from occhio.visualization import FeatureDimensionalityDistributionPlot

        plot = FeatureDimensionalityDistributionPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_interference_by_index_renders(self, small_model):
        from occhio.visualization import FeatureInterferenceByIndexPlot

        plot = FeatureInterferenceByIndexPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_interference_distribution_renders(self, small_model):
        from occhio.visualization import FeatureInterferenceDistributionPlot

        plot = FeatureInterferenceDistributionPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_superposition_indicator_renders(self, small_model):
        from occhio.visualization import SuperpositionIndicatorPlot

        plot = SuperpositionIndicatorPlot()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_representation_plot_faceted_on_grid(self, model_grid_1d):
        from occhio.visualization import RepresentationPlot

        plot = RepresentationPlot()
        fig = plot(model_grid_1d)
        assert isinstance(fig, go.Figure)
        # Should have one heatmap trace per model in the grid
        assert len(fig.data) >= 3

    def test_embedding_plot_faceted_on_grid(self, model_grid_1d):
        from occhio.visualization import EmbeddingPlot

        plot = EmbeddingPlot()
        fig = plot(model_grid_1d)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 2b. SAE-dependent plots — no SAEs scenario (graceful error annotation)
# ---------------------------------------------------------------------------


class TestSAEPlotGracefulDegradation:
    """SAE plots should display an error annotation rather than crash when no SAEs are present."""

    def test_sae_feature_similarity_no_saes(self, small_model):
        from occhio.visualization import SAEFeatureSimilarityPlot

        plot = SAEFeatureSimilarityPlot(sae_label="nonexistent")
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        # Should have an annotation mentioning "No SAEs"
        assert any(
            "No SAEs" in str(ann.text)
            for ann in fig.layout.annotations
            if ann.text is not None
        )

    def test_sae_one_hot_heatmap_no_saes(self, small_model):
        from occhio.visualization import SAEOneHotToLatentHeatmapPlot

        plot = SAEOneHotToLatentHeatmapPlot(sae_label="missing")
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert any(
            "No SAEs" in str(ann.text)
            for ann in fig.layout.annotations
            if ann.text is not None
        )

    def test_dynamic_sae_feature_similarity_no_saes(self, small_model):
        from occhio.visualization import plot_sae_feature_similarity

        plot = plot_sae_feature_similarity()
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert any(
            "No SAEs" in str(ann.text)
            for ann in fig.layout.annotations
            if ann.text is not None
        )


# ---------------------------------------------------------------------------
# 3. CompositePlot / Span Audit
# ---------------------------------------------------------------------------


class TestCompositePlot:
    """Test CompositePlot with multiple renderers and Span support."""

    def test_composite_2x2_renders(self, small_model):
        from occhio.visualization.core import CompositePlot
        from occhio.visualization.plots.feature_representation import (
            FeatureDimensionalityByIndexPlot,
            FeatureInterferenceByIndexPlot,
            FeatureNormByIndexPlot,
            FeatureNormDistributionPlot,
        )

        composite = CompositePlot(
            layout=[
                [FeatureNormByIndexPlot(), FeatureNormDistributionPlot()],
                [
                    FeatureDimensionalityByIndexPlot(),
                    FeatureInterferenceByIndexPlot(),
                ],
            ]
        )
        fig = composite(small_model)
        assert isinstance(fig, go.Figure)
        # Should have 4 traces minimum (one per subplot)
        assert len(fig.data) >= 4

    def test_composite_with_span_renders(self, small_model):
        from occhio.visualization.core import CompositePlot, Span
        from occhio.visualization.plots.feature_representation import (
            FeatureNormByIndexPlot,
            FeatureNormDistributionPlot,
            SuperpositionIndicatorPlot,
        )

        composite = CompositePlot(
            layout=[
                [Span(SuperpositionIndicatorPlot(), colspan=2)],
                [FeatureNormByIndexPlot(), FeatureNormDistributionPlot()],
            ],
            row_heights=[0.5, 1],
        )
        fig = composite(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_composite_empty_layout_raises(self):
        from occhio.visualization.core import CompositePlot

        with pytest.raises(ValueError, match="at least one plot"):
            CompositePlot(layout=[[None]])

    def test_composite_faceted_on_grid(self, model_grid_1d):
        from occhio.visualization.core import CompositePlot
        from occhio.visualization.plots.feature_representation import (
            FeatureNormByIndexPlot,
            FeatureNormDistributionPlot,
        )

        composite = CompositePlot(
            layout=[[FeatureNormByIndexPlot(), FeatureNormDistributionPlot()]]
        )
        fig = composite(model_grid_1d)
        assert isinstance(fig, go.Figure)
        # 3 models * 2 plots = 6+ traces
        assert len(fig.data) >= 6


# ---------------------------------------------------------------------------
# 4. Old Visualization Module Audit
# ---------------------------------------------------------------------------


class TestOldVisualization:
    """Tests for the old visualization module functions."""

    def test_plot_embedding_toymodel(self, small_model):
        from occhio.visualization import plot_embedding

        fig = plot_embedding(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_embedding_list(self, small_model):
        from occhio.visualization import plot_embedding

        fig = plot_embedding([small_model, small_model])
        assert isinstance(fig, go.Figure)

    def test_plot_representation_toymodel(self, small_model):
        from occhio.visualization import plot_representation

        fig = plot_representation(small_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    @pytest.mark.xfail(
        reason="BUG: plot_geometry wraps ToyModel in ModelGrid(axes=[]) "
        "which raises ValueError. Needs refactor to handle ToyModel directly.",
        raises=ValueError,
    )
    def test_plot_geometry_toymodel(self, small_model):
        from occhio.visualization import plot_geometry

        fig = plot_geometry(small_model)
        assert isinstance(fig, go.Figure)

    def test_plot_feature_geometry_toymodel(self, small_model):
        from occhio.visualization import plot_feature_geometry

        fig = plot_feature_geometry(small_model)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 5. Data Flow Audit — ToyModel attribute access
# ---------------------------------------------------------------------------


class TestDataFlow:
    """Verify that visualization code accesses ToyModel attributes that actually exist."""

    def test_toymodel_has_W_T_W(self, small_model):
        """RepresentationPlot accesses model.W_T_W."""
        W_T_W = small_model.W_T_W
        assert W_T_W.shape == (5, 5)

    def test_toymodel_has_feature_norms(self, small_model):
        """FeatureNormByIndexPlot accesses model.feature_norms."""
        norms = small_model.feature_norms
        assert norms.shape == (5,)

    def test_toymodel_has_feature_dimensionalities(self, small_model):
        """FeatureDimensionalityByIndexPlot accesses model.feature_dimensionalities."""
        dims = small_model.feature_dimensionalities
        assert dims.shape == (5,)

    def test_toymodel_has_total_feature_interferences(self, small_model):
        """FeatureInterferenceByIndexPlot accesses model.total_feature_interferences."""
        interferences = small_model.total_feature_interferences
        assert interferences.shape == (5,)

    def test_toymodel_has_superposition(self, small_model):
        """SuperpositionIndicatorPlot accesses model.superposition."""
        sup = small_model.superposition
        assert sup.dim() == 0  # scalar

    def test_toymodel_has_importances(self, small_model):
        """EmbeddingPlot accesses model.importances."""
        imp = small_model.importances
        assert imp.shape == (5,)

    def test_toymodel_has_W(self, small_model):
        """EmbeddingPlot accesses model.W."""
        W = small_model.W
        assert W.shape == (2, 5)

    def test_toymodel_has_saes_dict(self, small_model):
        """All SAE plots check model.saes."""
        assert isinstance(small_model.saes, dict)

    def test_toymodel_has_feature_frequencies(self, small_model):
        """SAEPerFeatureF1Plot accesses model.feature_frequencies."""
        freqs = small_model.feature_frequencies
        assert freqs.shape == (5,)
        assert (freqs >= 0).all() and (freqs <= 1).all()

    def test_toymodel_does_not_have_saes_per_feature_f1(self, small_model):
        """SAEPerFeatureF1Plot accesses model.saes_per_feature_f1 which does NOT exist.

        This is a known bug: the plot references an attribute that was never
        added to ToyModel. It will crash at runtime with an AttributeError.
        """
        assert not hasattr(small_model, "saes_per_feature_f1"), (
            "If this test fails, it means saes_per_feature_f1 was added to "
            "ToyModel and the per-feature F1 plots should now work."
        )


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: single feature, unfitted grid, etc."""

    def test_representation_plot_single_feature(self, single_feature_model):
        """RepresentationPlot should work with a 1-feature model."""
        from occhio.visualization import RepresentationPlot

        plot = RepresentationPlot()
        fig = plot(single_feature_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_feature_norm_single_feature(self, single_feature_model):
        from occhio.visualization import FeatureNormByIndexPlot

        plot = FeatureNormByIndexPlot()
        fig = plot(single_feature_model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_superposition_single_feature(self, single_feature_model):
        """Single-feature model: superposition should be 0 (no off-diagonal elements)."""
        from occhio.visualization import SuperpositionIndicatorPlot

        plot = SuperpositionIndicatorPlot()
        fig = plot(single_feature_model)
        assert isinstance(fig, go.Figure)
        val = single_feature_model.superposition.item()
        assert val == 0.0, f"Expected superposition=0 for 1 feature, got {val}"

    def test_sae_feature_similarity_wrong_label(self, small_model):
        """Asking for an SAE label that doesn't exist."""
        from occhio.visualization import SAEFeatureSimilarityPlot

        # Give it an SAE dict so it passes the "no SAEs" check
        # but fails on the specific label
        small_model.saes["dummy"] = type(
            "FakeRecord", (), {"sae": None, "results": None}
        )()
        plot = SAEFeatureSimilarityPlot(sae_label="wrong_label")
        fig = plot(small_model)
        assert isinstance(fig, go.Figure)
        assert any(
            "not found" in str(ann.text)
            for ann in fig.layout.annotations
            if ann.text is not None
        )

    def test_plot_representation_single_toymodel(self):
        """plot_representation (now RepresentationPlot()) works with single ToyModel."""
        from occhio.visualization import plot_representation

        dist = SparseUniform(3, p_active=0.5)
        ae = TiedLinearRelu(3, 2)
        tm = ToyModel(dist, ae)
        tm.fit(n_epochs=3, batch_size=32, verbose=False)
        fig = plot_representation(tm)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_composite_plot_feature_representation_renders(self, small_model):
        """The pre-built plot_feature_representation composite should render."""
        from occhio.visualization.plots.feature_representation import (
            plot_feature_representation,
        )

        fig = plot_feature_representation(small_model)
        assert isinstance(fig, go.Figure)
        # Should have traces from all 4 rows (indicator, dim, norm, interference)
        assert len(fig.data) >= 4


# ---------------------------------------------------------------------------
# 7. FigureProxy Audit
# ---------------------------------------------------------------------------


class TestFigureProxy:
    """Test FigureProxy behavior: axis remapping, legend dedup, blocked methods."""

    def test_update_layout_blocked(self, small_model):
        from plotly.subplots import make_subplots

        from occhio.visualization.core.figure_wrappers import FigureProxy

        fig = make_subplots(rows=1, cols=1)
        proxy = FigureProxy(fig, row=1, col=1)
        with pytest.raises(AttributeError, match="update_layout"):
            proxy.update_layout(title="test")

    def test_legend_dedup_across_subplots(self):
        from plotly.subplots import make_subplots

        from occhio.visualization.core.figure_wrappers import FigureProxy

        fig = make_subplots(rows=1, cols=2)
        registry: set[str] = set()
        proxy1 = FigureProxy(fig, row=1, col=1, legend_registry=registry)
        proxy2 = FigureProxy(fig, row=1, col=2, legend_registry=registry)

        proxy1.add_trace(go.Scatter(x=[1], y=[1], name="shared"))
        proxy2.add_trace(go.Scatter(x=[2], y=[2], name="shared"))

        # First trace should show legend, second should not
        assert fig.data[0].showlegend is not False
        assert fig.data[1].showlegend is False


# ---------------------------------------------------------------------------
# 8. Overlay Plots with 1D ModelGrid
# ---------------------------------------------------------------------------


class TestOverlayPlots:
    """Test overlay distribution plots that require n_render_axes=1."""

    def test_feature_norm_overlay_renders(self, model_grid_1d):
        from occhio.visualization import FeatureNormDistributionOverlayPlot

        plot = FeatureNormDistributionOverlayPlot()
        fig = plot(model_grid_1d)
        assert isinstance(fig, go.Figure)
        # Should have histogram traces + mean line traces
        assert len(fig.data) >= 3

    def test_feature_dimensionality_overlay_renders(self, model_grid_1d):
        from occhio.visualization import (
            FeatureDimensionalityDistributionOverlayPlot,
        )

        plot = FeatureDimensionalityDistributionOverlayPlot()
        fig = plot(model_grid_1d)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_feature_interference_overlay_renders(self, model_grid_1d):
        from occhio.visualization import (
            FeatureInterferenceDistributionOverlayPlot,
        )

        plot = FeatureInterferenceDistributionOverlayPlot()
        fig = plot(model_grid_1d)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3
