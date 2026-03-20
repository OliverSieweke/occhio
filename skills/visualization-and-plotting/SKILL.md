---
name: visualization-and-plotting
description: Create Plotly visualizations for ToyModel and ModelGrid objects using the visualization_2 module. Use whenever implementing new plots, modifying existing ones, understanding the plotting framework, composing multi-plot layouts, or debugging visualization code. Trigger on any mention of "plot", "chart", "heatmap", "visualization", "figure", "trace", "facet", "slider", or references to visualization_2, SinglePlot, CompositePlot, FigureProxy.
---

# Visualization and Plotting

Create Plotly-based visualizations for `ToyModel` and `ModelGrid` objects using the `visualization_2` module.

## Golden Rules

1. **Never modify `core/`** — The `visualization_2/core/` directory contains stable base classes. If the framework doesn't support your use case, explain the blocker to the user and ask permission before proposing changes.

2. **Read data from `ToyModel` properties** — Don't invent complex computation inside plots. If a needed metric doesn't exist, propose adding it as a `ToyModel` property first, and confirm with the user.

3. **Device safety** — Always `.detach().cpu().numpy()` (or `.cpu().item()` for scalars) before passing tensors to Plotly. Bare `.numpy()` fails on GPU/MPS.

4. **Keep plot code simple and tweakable** — Plots are the "last mile" that users customize. Prefer explicit, linear code over clever abstractions.

## Architecture

```
src/occhio/visualization_2/
├── core/                       # DO NOT MODIFY
│   ├── __init__.py             # Exports: SinglePlot, CompositePlot
│   ├── base_plot.py            # Plot (ABC), SinglePlot (ABC)
│   ├── composite_plot.py       # CompositePlot, Span, SubplotSpec
│   ├── figure_wrappers.py      # FigureProxy, InteractiveFigure
│   └── plotting_utils.py       # add_grid_headers, model_domain_center
└── plots/                      # ADD NEW PLOTS HERE
    ├── __init__.py             # Export all plot classes/instances
    ├── embedding.py            # EmbeddingPlot
    ├── feature_representation.py
    ├── representation.py       # RepresentationPlot
    ├── sae_classification_metric.py   # SAEClassificationMetricPlot, etc.
    └── sae_classification_metrics.py  # SAEClassificationMetricsPlot
```

## Creating a SinglePlot

### Minimal Template

```python
import plotly.graph_objects as go

from occhio.toy_model import ToyModel
from occhio.visualization_2.core.base_plot import SinglePlot
from occhio.visualization_2.core.figure_wrappers import FigureProxy


class MyPlot(SinglePlot):
    """One-line summary.

    Use case:
        When/why to use this plot.

    Data:
        - `model.property_name`: What it represents.

    Visualization:
        How the data is rendered (chart type, axes, colors).

    Customization:
        - `param`: What it controls (default: value).
    """

    n_render_axes = 0  # Single ToyModel per subplot

    def __init__(self, param: str = "default"):
        self.param = param

    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        data = model.some_property.detach().cpu().numpy()
        fig.add_trace(go.Scatter(x=[0, 1], y=data))
        fig.update_xaxes(title_text="X")
        fig.update_yaxes(title_text="Y")

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
```

### `render()` — Per-Subplot Logic

`render()` draws into a single subplot cell. The framework calls it once per facet position.

**Receives `FigureProxy`** (not raw `go.Figure`). FigureProxy auto-injects `row`/`col` for all subplot-aware methods and handles:
- **Legend deduplication**: Same trace name across subplots appears in legend only once.
- **Axis label deduplication**: X-axis tick labels only on bottom row, Y-axis only on left column. Override with `showticklabels=True`.

**Allowed FigureProxy methods:**
- `fig.add_trace(go.Scatter(...))` — routed to correct subplot
- `fig.update_xaxes(...)` / `fig.update_yaxes(...)` — scoped to this subplot
- `fig.add_annotation(...)`, `fig.add_hline(...)`, `fig.add_vline(...)`, `fig.add_shape(...)`

**Blocked:**
- `fig.update_layout(...)` — raises `AttributeError`. Use `configure_layout()` instead.

**Axis reference remapping:** FigureProxy remaps bare `'x'`/`'y'` axis refs (e.g., in `scaleanchor="x"`) to the correct subplot axis. Don't use numbered refs like `'x2'` — let the proxy handle it.

### `configure_layout()` — Global Styling

Called once after all `render()` calls complete. Receives the raw `go.Figure`.

```python
def configure_layout(self, fig: go.Figure) -> None:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.15,
    )
    # Global axis styling (applies to all subplots)
    fig.update_xaxes(showgrid=False)
```

### `n_render_axes` — What `render()` Receives

| Value | `models` parameter type | Use case |
|-------|------------------------|----------|
| `0` (default) | Single `ToyModel` | Per-model plots: heatmaps, bar charts, embeddings |
| `1` | 1D `ModelGrid` | Line charts where x-axis spans grid values |
| `2` | 2D `ModelGrid` | Phase diagrams, contour plots, surface plots |

When `n_render_axes > 0`, `render()` receives a `ModelGrid` and you access the render axis via `models.axes[0]`:

```python
from occhio.model_grid import ModelGrid

class MetricOverGridPlot(SinglePlot):
    n_render_axes = 1

    def render(self, fig: FigureProxy, models: ModelGrid) -> None:
        axis = models.axes[0]
        values = [m.superposition.cpu().item() for m in models]

        fig.add_trace(go.Scatter(
            x=axis.values,
            y=values,
            mode="lines+markers",
        ))
        fig.update_xaxes(title_text=axis.label)
```

### Error Handling

When data is missing (e.g., SAEs not trained), add an annotation instead of raising:

```python
def render(self, fig: FigureProxy, model: ToyModel) -> None:
    if not model.saes:
        fig.add_annotation(
            text="No SAEs trained.<br>Call model.train_saes() first.",
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="firebrick"),
        )
        return
    # Normal rendering...
```

## Composing Plots with CompositePlot

Arrange multiple `SinglePlot` instances in a grid layout rendered per model:

```python
from occhio.visualization_2.core.composite_plot import CompositePlot, Span

# 2x2 grid
plot = CompositePlot(
    layout=[
        [PrecisionPlot(), RecallPlot()],
        [AccuracyPlot(), F1Plot()],
    ],
    column_widths=[1, 1],
    row_heights=[1, 1],
)

# With spanning
plot = CompositePlot(
    layout=[
        [Span(WideHeaderPlot(), colspan=2)],
        [LeftPlot(), RightPlot()],
    ],
)
```

**Constraint:** All plots in a `CompositePlot` must share the same `n_render_axes`.

`CompositePlot` delegates `configure_layout()` to each child plot. The inner grid tiles across facet positions automatically.

## Calling Plots

```python
# Single model
plot(model)

# Grid — auto-assigns axes to facets/sliders
plot(grid)

# Explicit axis control (accepts int indices or string labels)
plot(grid, facet_axes=("Sparsity",), slider_axes=("Correlation",))

# Custom size
plot(model, height=400, width=600)
```

**Axis assignment rules** (from `resolve_full_axes`): unassigned axes fill render first (up to `n_render_axes`), then facet (up to 2), then slider (remaining). Facet axis 0 maps to columns, facet axis 1 maps to rows. Slider axes produce Plotly animation frames with interactive sliders.

## Naming and Exporting

**Class name:** `PascalCase` ending in `Plot` — e.g., `FeatureInterferencePlot`

**Instance name:** `plot_<domain>_<metric>` — e.g., `plot_feature_interference`

Exporting instances enables autocomplete: `viz.plot_feat...` lists all feature plots.

```python
# In plots/feature.py
class FeatureInterferencePlot(SinglePlot): ...

plot_feature_interference = FeatureInterferencePlot()

# In plots/__init__.py
from .feature import plot_feature_interference, FeatureInterferencePlot

# In visualization_2/__init__.py
from .plots import plot_feature_interference
```

## Docstring Template

Every plot class must document these four sections:

```python
class MyPlot(SinglePlot):
    """One-line summary.

    Use case:
        When/why to use this plot.

    Data:
        - `model.property`: What it represents (tensor shape, units).

    Visualization:
        Chart type, what maps to which axis, colorscale logic.

    Customization:
        - `param_name`: What it controls (default: value).
    """
```

## Available ToyModel Properties

### Geometric Properties (all `@property`, `@torch.no_grad()`)

| Property | Type | Description |
|----------|------|-------------|
| `W` | `Tensor (n_hidden x n_features)` | Encoder weight matrix (one-hot embeddings transposed) |
| `W_T_W` | `Tensor (n_features x n_features)` | Gram matrix W^T W |
| `W_normalized_features` | `Tensor (n_hidden x n_features)` | Column-normalized W |
| `feature_norms` | `Tensor (n_features,)` | L2 norm per feature embedding |
| `feature_representations` | `Tensor (n_features,)` | Squared norm per feature |
| `interferences` | `Tensor (n_features x n_features)` | W_norm^T @ W (not squared) |
| `interferences_sq` | `Tensor (n_features x n_features)` | (W_norm^T @ W)^2 |
| `total_feature_interferences` | `Tensor (n_features,)` | Sum of squared interferences, diagonal zeroed |
| `total_feature_interferences_including_self` | `Tensor (n_features,)` | Sum of squared interferences, with diagonal |
| `cosine_similarity_matrix` | `Tensor (n_features x n_features)` | Pairwise cosine similarities |
| `superposition` | `Tensor (scalar)` | Mean max abs cosine similarity (rho_mm) |
| `feature_dimensionalities` | `Tensor (n_features,)` | representations / total_interferences_including_self |
| `mean_feature_dimensionalities` | `Tensor (scalar)` | Mean of feature_dimensionalities |
| `frobenius_norm_squared` | `Tensor (scalar)` | norm(W)^2_F |

### SAE Metrics (all `dict[str, float]` keyed by SAE label)

| Property | Description |
|----------|-------------|
| `saes_precision` | TP / (TP + FP) |
| `saes_recall` | TP / (TP + FN) |
| `saes_f1_score` | Harmonic mean of precision and recall |
| `saes_accuracy` | (TP + TN) / total |
| `saes_explained_variance` | Variance explained by SAE reconstruction |
| `saes_l0` | L0 sparsity of SAE activations |
| `saes_true_l0` | Ground-truth feature activation L0 |
| `saes_dead_latents` | Count of dead SAE latents (`int`) |
| `saes_shrinkage` | Ratio of SAE output norm to input norm |
| `saes_mcc` | Mean Correlation Coefficient (SAE decoder vs ground truth) |
| `saes_uniqueness` | Fraction of SAE latents tracking unique features |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `importances` | `Tensor (n_features,)` | Feature importance weights |
| `n_features` | `int` | Number of features (delegated to ae) |
| `n_hidden` | `int` | Hidden dimension (delegated to ae via `__getattr__`) |
| `saes` | `dict[str, SAERecord]` | Trained SAEs and their eval results |
| `device` | `torch.device` | Computation device |

### ModelGrid / Axis

```python
from occhio.model_grid import ModelGrid, Axis

grid.axes          # list[Axis] — each has .label (str) and .values (Tensor or list)
grid.shape         # tuple of axis lengths
grid[i]            # index into grid; slicing with int/slice supported
len(grid)          # total number of models (product of shape)
for model in grid  # iterate over all models (flattened)
```

## Existing Plots as Reference

These are the current plots — read their source for patterns and conventions.

| Class | File | `n_render_axes` | Description |
|-------|------|-----------------|-------------|
| `EmbeddingPlot` | `embedding.py` | 0 | Arrow plot of 2D feature embeddings from origin |
| `RepresentationPlot` | `representation.py` | 0 | W^T W heatmap |
| `SAEClassificationMetricPlot` | `sae_classification_metric.py` | 1 | Line chart of one metric across a grid axis |
| `SAEMetricsComparisonPlot` | `sae_classification_metric.py` | 1 | Multiple metrics for one SAE across a grid |
| `SAEClassificationMetricsPlot` | `sae_classification_metrics.py` | 0 | Grouped bar chart of classification metrics per model |
| `plot_sae_classification_metrics` | `sae_classification_metric.py` | (composite) | 2x2 CompositePlot of precision/recall/accuracy/F1 |

## Workflow Summary

1. **Confirm data** — List which `ToyModel` properties the plot needs; get user sign-off.
2. **Create class** — Subclass `SinglePlot`, set `n_render_axes`, write `__init__`.
3. **Implement `render()`** — Add traces, configure subplot axes. Move tensors to CPU.
4. **Implement `configure_layout()`** — Set figure-wide styling (optional).
5. **Document** — Docstring with use case, data, visualization, customization.
6. **Export** — Create instance with `plot_<domain>_<metric>` naming.
7. **Update `__init__.py`** — Export from both `plots/__init__.py` and `visualization_2/__init__.py`.

## Skill Gaps

If you encounter issues while implementing a plot, suggest additions to this skill:
- Missing `ToyModel` property documentation
- Unclear `FigureProxy` behavior or edge cases
- Common Plotly patterns not covered
- `n_render_axes = 2` examples (none exist yet in the codebase)
