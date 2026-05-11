"""SAE benchmark table visualization.

This module provides a table plot for comparing SAE metrics (F1, MCC, EV, L0) across
models in a 1D ModelGrid, with SAEs as major column groups and metrics as sub-columns.
"""

from typing import Any

import plotly.graph_objects as go
from IPython.display import HTML, display

from occhio import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class ScrollableFigure(go.Figure):
    """A Plotly Figure that renders in a horizontally scrollable container."""

    def _ipython_display_(self, **kwargs: Any) -> None:
        """Render in Jupyter wrapped in a scrollable div."""
        html = self.to_html(full_html=False, include_plotlyjs="require")
        # Use max-width with overflow to create scrollable container
        scrollable_html = f"""
        <div style="overflow-x: scroll; overflow-y: hidden; max-width: 100%; border: 1px solid #eee;">
            <div style="min-width: {self.layout.width}px;">
                {html}
            </div>
        </div>
        """
        display(HTML(scrollable_html))

    def show(self, *args, **kwargs):
        """Display the figure wrapped in a scrollable div."""
        self._ipython_display_()


class SAEBenchmarkTablePlot(SinglePlot):
    """Table comparing SAE metrics across models with SAEs as column groups.

    Use case:
        Compare F1, MCC, EV, and L0 across multiple SAEs and models in a benchmark
        setting where rows are models (distributions) and columns are grouped by SAE.

    Data:
        - `model.saes_f1_score`: F1 score per SAE
        - `model.saes_mcc`: Mean Correlation Coefficient per SAE
        - `model.saes_explained_variance`: Explained variance per SAE
        - `model.saes_l0`: L0 sparsity per SAE

    Visualization:
        - Table with one row per model (axis value)
        - Columns grouped by SAE, with F1/MCC/EV/L0 sub-columns under each
        - Bold for best value per column (max F1/MCC/EV, min L0)
        - Underline for best value per row (max F1/MCC/EV, min L0)

    Customization:
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(
        self,
        sae_labels: list[str] | None = None,
        col_width_model: int = 120,
        col_width_metric: int = 55,
    ):
        """Initialize SAEBenchmarkTablePlot.

        Args:
            sae_labels: Optional list of SAE labels to include. If None, includes all.
            col_width_model: Width of the model column in pixels (default: 120).
            col_width_metric: Width of each metric column in pixels (default: 55).
        """
        self.sae_labels = sae_labels
        self.col_width_model = col_width_model
        self.col_width_metric = col_width_metric

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the benchmark table.

        Args:
            fig: FigureProxy for adding traces.
            models: 1D ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAEBenchmarkTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        if len(models.axes) < 1:
            fig.add_annotation(
                text="SAEBenchmarkTablePlot requires a 1D ModelGrid.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        render_axis = models.axes[0]

        # Find common SAE labels across all models
        common_sae_labels = set.intersection(
            *({label for label in model.saes} for model in models)
        )

        if not common_sae_labels:
            fig.add_annotation(
                text="No SAE trained on all models.<br>Call grid.train_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        sae_labels = (
            [lbl for lbl in self.sae_labels if lbl in common_sae_labels]
            if self.sae_labels
            else sorted(common_sae_labels)
        )

        if not sae_labels:
            fig.add_annotation(
                text=f"Requested SAEs not found.<br>Available: {', '.join(common_sae_labels)}",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        # Check SAEs are evaluated
        evaluated_sae_labels = set.intersection(
            *(
                {label for label in model.saes if model.saes[label].results}
                for model in models
            )
        )
        unevaluated = set(sae_labels) - evaluated_sae_labels
        if unevaluated:
            fig.add_annotation(
                text=f"SAEs {unevaluated} not evaluated.<br>Call grid.evaluate_saes() first.",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="firebrick"),
            )
            return

        n_models = models.shape[0]
        n_saes = len(sae_labels)
        model_labels = [str(render_axis.values[i]) for i in range(n_models)]

        # Collect raw values: [model_idx][sae_idx] for each metric
        f1_raw: list[list[float]] = []
        mcc_raw: list[list[float]] = []
        ev_raw: list[list[float]] = []
        l0_raw: list[list[float]] = []

        for model in models:
            f1_raw.append([model.saes_f1_score[label] for label in sae_labels])
            mcc_raw.append([model.saes_mcc[label] for label in sae_labels])
            ev_raw.append(
                [model.saes_explained_variance[label] for label in sae_labels]
            )
            l0_raw.append([model.saes_l0[label] for label in sae_labels])

        # Find best per column (across models for each SAE)
        f1_col_best = [
            max(f1_raw[m][s] for m in range(n_models)) for s in range(n_saes)
        ]
        mcc_col_best = [
            max(mcc_raw[m][s] for m in range(n_models)) for s in range(n_saes)
        ]
        ev_col_best = [
            max(ev_raw[m][s] for m in range(n_models)) for s in range(n_saes)
        ]
        l0_col_best = [
            min(l0_raw[m][s] for m in range(n_models)) for s in range(n_saes)
        ]

        # Find best per row (across SAEs for each model)
        f1_row_best = [max(f1_raw[m]) for m in range(n_models)]
        mcc_row_best = [max(mcc_raw[m]) for m in range(n_models)]
        ev_row_best = [max(ev_raw[m]) for m in range(n_models)]
        l0_row_best = [min(l0_raw[m]) for m in range(n_models)]

        # Build header: Model | SAE1 F1 | SAE1 MCC | SAE1 EV | SAE1 L0 | SAE2 F1 | ...
        # Two-row header: first row has SAE names spanning 4 cols, second row has metric names
        header_row1 = ["<b>Model</b>"]
        header_row2 = [""]
        for sae_label in sae_labels:
            header_row1.extend([f"<b>{sae_label}</b>", "", "", ""])
            header_row2.extend(["F1", "MCC", "EV", "L0"])

        header_values = [
            [header_row1[i], header_row2[i]] for i in range(len(header_row1))
        ]

        # Header colors
        header_colors = [["rgb(240, 240, 240)", "rgb(240, 240, 240)"]]  # Model column
        for _ in range(n_saes):
            for _ in range(4):  # F1, MCC, EV, L0
                header_colors.append(["rgb(220, 230, 240)", "rgb(240, 240, 240)"])

        # Build cell values with formatting
        model_col = model_labels
        metric_cols: list[list[str]] = []

        for sae_idx, sae_label in enumerate(sae_labels):
            f1_col: list[str] = []
            mcc_col: list[str] = []
            ev_col: list[str] = []
            l0_col: list[str] = []

            for model_idx in range(n_models):
                f1_val = f1_raw[model_idx][sae_idx]
                mcc_val = mcc_raw[model_idx][sae_idx]
                ev_val = ev_raw[model_idx][sae_idx]
                l0_val = l0_raw[model_idx][sae_idx]

                # Determine formatting
                f1_is_col_best = f1_val == f1_col_best[sae_idx]
                f1_is_row_best = f1_val == f1_row_best[model_idx]
                mcc_is_col_best = mcc_val == mcc_col_best[sae_idx]
                mcc_is_row_best = mcc_val == mcc_row_best[model_idx]
                ev_is_col_best = ev_val == ev_col_best[sae_idx]
                ev_is_row_best = ev_val == ev_row_best[model_idx]
                l0_is_col_best = l0_val == l0_col_best[sae_idx]
                l0_is_row_best = l0_val == l0_row_best[model_idx]

                f1_col.append(
                    self._format_value(f1_val, ".3f", f1_is_col_best, f1_is_row_best)
                )
                mcc_col.append(
                    self._format_value(mcc_val, ".3f", mcc_is_col_best, mcc_is_row_best)
                )
                ev_col.append(
                    self._format_value(ev_val, ".3f", ev_is_col_best, ev_is_row_best)
                )
                l0_col.append(
                    self._format_value(l0_val, ".2f", l0_is_col_best, l0_is_row_best)
                )

            metric_cols.extend([f1_col, mcc_col, ev_col, l0_col])

        cell_values = [model_col] + metric_cols

        # Cell colors (alternating by SAE for readability)
        cell_colors = [["white"] * n_models]  # Model column
        for sae_idx in range(n_saes):
            color = "rgb(250, 250, 250)" if sae_idx % 2 == 0 else "rgb(245, 245, 245)"
            for _ in range(4):
                cell_colors.append([color] * n_models)

        # Alignment and column widths
        align = ["left"] + ["center"] * (n_saes * 4)
        col_widths = [self.col_width_model] + [self.col_width_metric] * (n_saes * 4)

        # Store n_saes for configure_layout to compute width
        self._last_n_saes = n_saes

        fig.add_trace(
            go.Table(
                columnwidth=col_widths,
                header=dict(
                    values=header_values,
                    fill_color=header_colors,
                    align=align,
                    font=dict(size=11),
                    height=26,
                    line_color="white",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=cell_colors,
                    align=align,
                    font=dict(size=10),
                    height=24,
                ),
            )
        )

    def _format_value(
        self, value: float, fmt: str, is_col_best: bool, is_row_best: bool
    ) -> str:
        """Format value with bold for column best and underline for row best."""
        text = f"{value:{fmt}}"
        if is_col_best and is_row_best:
            return f"<b><u>{text}</u></b>"
        elif is_col_best:
            return f"<b>{text}</b>"
        elif is_row_best:
            return f"<u>{text}</u>"
        return text

    def configure_layout(self, fig: go.Figure) -> None:
        # Compute width based on column widths
        n_saes = getattr(self, "_last_n_saes", 10)
        total_width = self.col_width_model + self.col_width_metric * n_saes * 4 + 40
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            width=total_width,
        )

    def __call__(self, *args, **kwargs) -> ScrollableFigure:
        """Return a scrollable figure for wide tables."""
        fig = super().__call__(*args, **kwargs)
        return ScrollableFigure(fig)


plot_sae_benchmark_table = SAEBenchmarkTablePlot()
