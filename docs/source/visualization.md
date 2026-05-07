# Visualization

occhio's visualization module produces interactive Plotly figures from
`ToyModel` and `ModelGrid` objects.

## SinglePlot Pattern

Every plot follows the same pattern: instantiate a plot object, then call it
on a model or grid.

```python
from occhio.visualization import EmbeddingPlot

plot = EmbeddingPlot()
fig = plot(model)       # returns a Plotly Figure
fig.show()
```

When called on a `ModelGrid`, the plot automatically creates a faceted grid
of subplots -- one per grid cell.

```python
fig = plot(grid)        # faceted across grid axes
fig.show()
```

## Available Plots

### Geometry & Representation

| Plot | Description |
|---|---|
| `EmbeddingPlot` | Feature embedding vectors in hidden space |
| `RepresentationPlot` | Feature representation (norm^2) as a bar chart |
| `FeatureNormByIndexPlot` | Per-feature L2 norms |
| `FeatureNormDistributionPlot` | Distribution of feature norms |
| `FeatureDimensionalityByIndexPlot` | Per-feature effective dimensionality |
| `FeatureDimensionalityDistributionPlot` | Distribution of feature dimensionalities |
| `FeatureInterferenceByIndexPlot` | Per-feature total interference |
| `FeatureInterferenceDistributionPlot` | Distribution of feature interferences |
| `SuperpositionIndicatorPlot` | Scalar superposition indicator (rho_mm) |
| `FeatureGeometryPlot` | Combined geometry view (norms + dimensionalities) |
| `GeometryPlot` | Configurable multi-component geometry plot |

### Phase Change & Dynamics

| Plot | Description |
|---|---|
| `PhaseChangePlot` | Feature norms across a parameter sweep (phase transitions) |
| `DecodePlanePlot` | Decode plane visualization for 2D hidden spaces |

### SAE Evaluation

| Plot | Description |
|---|---|
| `SAEFeatureSimilarityPlot` | Cosine similarity between SAE decoder and true features |
| `SAEOneHotToLatentHeatmapPlot` | One-hot input to SAE latent activation heatmap |
| `SAEF1vsL0Plot` | F1 score vs L0 sparsity scatter |
| `SAEPerFeatureF1Plot` | Per-feature F1 scores |
| `SAEPerFeatureF1DistributionPlot` | Distribution of per-feature F1 scores |

### Summary Tables

| Plot | Description |
|---|---|
| `SAEMetricsTablePlot` | Full SAE metrics table |
| `SAECoreMetricsTablePlot` | Core SAE metrics (F1, MCC, explained variance) |
| `SAESparsityMetricsTablePlot` | Sparsity-focused SAE metrics |
| `SAEBenchmarkTablePlot` | SAE benchmark comparison table |
| `DiagnosticTablePlot` | Diagnostic summary table |

### Overlay Plots

Overlay variants (`FeatureNormDistributionOverlayPlot`, etc.) plot multiple
distributions on the same axes for comparison.

## CompositePlot

Combine multiple plots into a single figure with
{class}`~occhio.visualization.core.composite_plot.CompositePlot`:

```python
from occhio.visualization import FeatureNormByIndexPlot, RepresentationPlot
from occhio.visualization.core.composite_plot import CompositePlot, Span

composite = CompositePlot([
    [FeatureNormByIndexPlot(), RepresentationPlot()],
    [Span(EmbeddingPlot(), colspan=2)],
])
fig = composite(model)
fig.show()
```

## Faceting with ModelGrid

When a `SinglePlot` is called on a `ModelGrid`, the grid axes become subplot
rows and columns automatically. Plots with `n_render_axes > 0` consume inner
axes for rendering (e.g., line charts over a `TrainingAxis`), while the
remaining axes are faceted.

## Exporting Figures

```python
from occhio.visualization import export_figure

export_figure(fig, labels={"model": "baseline", "metric": "norms"})
```

Saves to `../figures/` by default with auto-generated filenames.

## API Reference

See the full {doc}`api` for detailed class documentation.
