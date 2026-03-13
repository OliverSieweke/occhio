---
name: visualization-and-plotting
description: BasePlot pattern for creating Plotly visualizations of ToyModel and ModelGrid objects. Use when implementing custom plots, adding faceting/slider support, or working with the visualization module.
---

# Visualization and Plotting Skill

Plotly-based visualization for `ToyModel` and `ModelGrid` objects.

## BasePlot Pattern

`BasePlot` is the abstract base class for creating visualizations. Subclass it and implement the `render()` method to create custom plots that automatically handle:

- Single `ToyModel` rendering
- `ModelGrid` faceting (rows/columns for up to 2 axes)
- Interactive sliders for additional axes

### Usage

```python
from occhio.visualization.base_plot import BasePlot, FigureProxy
import plotly.graph_objects as go


class MyPlot(BasePlot):
    def render(self, fig: FigureProxy, model: ToyModel) -> None:
        # Add traces to the figure proxy
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))


# Single model
MyPlot()(model)

# Grid with automatic faceting
MyPlot()(grid)

# Explicit axis control
MyPlot()(grid, facet_axes=("Sparsity",), slider_axes=("Correlation",))
```

### Key Points

- Use `FigureProxy` (not raw `go.Figure`) in `render()`  it handles subplot positioning
- When accessing `ToyModel` tensor attributes, use `.detach().cpu().numpy()` for device safety
- `facet_axes` accepts up to 2 axes (mapped to columns and rows)
- Remaining axes become sliders automatically
