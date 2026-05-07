"""SAE classification metrics table visualization.

This module provides a table plot for comparing SAE classification metrics
across models in a 1D ModelGrid, with color-coded cells and percentage
differences from a baseline row.
"""

from typing import Literal

import plotly.graph_objects as go

from occhio import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy


class SAEMetricsTablePlot(SinglePlot):
    """Table comparing SAE classification metrics across a 1D ModelGrid.

    Use case:
        Compare precision, recall, accuracy, and F1 scores across multiple
        SAEs or models in a tabular format with visual highlighting.

    Data:
        - `model.saes_precision`: Precision scores per SAE
        - `model.saes_recall`: Recall scores per SAE
        - `model.saes_accuracy`: Accuracy scores per SAE
        - `model.saes_f1_score`: F1 scores per SAE

    Visualization:
        - Table with rows for each SAE/model combination
        - Columns for 4 classification metrics
        - Cells color-coded 0 (white) to 1 (green)
        - Column max values bolded
        - Percentage difference from first row shown in parentheses

    Customization:
        - `group_by`: "model" or "sae" - determines row grouping (default: "model")
        - `sae_labels`: Optional list of SAE labels to include
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(
        self,
        group_by: Literal["model", "sae"] = "model",
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAEMetricsTablePlot.

        Args:
            group_by: How to group rows - "model" groups by model with SAEs
                as sub-rows, "sae" groups by SAE with models as sub-rows.
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.group_by = group_by
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the metrics table.

        Args:
            fig: FigureProxy for adding traces.
            models: 1D ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAEMetricsTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        if len(models.axes) < 1:
            fig.add_annotation(
                text="SAEMetricsTablePlot requires a 1D ModelGrid.",
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

        # Collect data
        metrics = ["Precision", "Recall", "Accuracy", "F1"]
        metric_props = [
            "saes_precision",
            "saes_recall",
            "saes_accuracy",
            "saes_f1_score",
        ]

        # Build rows based on grouping, tracking SAE label per row for baseline lookup
        row_labels: list[str] = []
        row_values: list[list[float]] = []
        row_sae_labels: list[str] = []  # Track which SAE each row belongs to

        # Get baseline values from first model for each SAE
        first_model = models[0]
        baselines: dict[str, list[float]] = {
            sae_label: [getattr(first_model, prop)[sae_label] for prop in metric_props]
            for sae_label in sae_labels
        }

        if self.group_by == "model":
            # Rows: Model (SAE) format
            for i, model in enumerate(models):
                axis_val = render_axis.values[i]
                for sae_label in sae_labels:
                    if len(sae_labels) == 1:
                        row_labels.append(f"{axis_val}")
                    else:
                        row_labels.append(f"{axis_val} ({sae_label})")
                    row_values.append(
                        [getattr(model, prop)[sae_label] for prop in metric_props]
                    )
                    row_sae_labels.append(sae_label)
        else:  # group_by == "sae"
            # Rows: SAE (Model) format
            for sae_label in sae_labels:
                for i, model in enumerate(models):
                    axis_val = render_axis.values[i]
                    if len(models) == 1:
                        row_labels.append(f"{sae_label}")
                    else:
                        row_labels.append(f"{sae_label} ({axis_val})")
                    row_values.append(
                        [getattr(model, prop)[sae_label] for prop in metric_props]
                    )
                    row_sae_labels.append(sae_label)

        # Calculate column maxes for bolding
        n_metrics = len(metrics)
        col_maxes = [
            max(row_values[r][c] for r in range(len(row_values)))
            for c in range(n_metrics)
        ]

        # Format cell text with bold max and % diff
        cell_text: list[list[str]] = []
        for row_idx, row in enumerate(row_values):
            row_text = []
            sae_label = row_sae_labels[row_idx]
            baseline = baselines[sae_label]
            is_baseline_row = row == baseline

            for col_idx, val in enumerate(row):
                # Format value
                val_str = f"{val:.3f}"

                # Add % diff (skip for baseline row of this SAE)
                if not is_baseline_row:
                    base_val = baseline[col_idx]
                    if base_val != 0:
                        pct_diff = ((val - base_val) / base_val) * 100
                        sign = "+" if pct_diff >= 0 else ""
                        val_str += f" ({sign}{pct_diff:.1f}%)"
                    else:
                        val_str += " (N/A)"

                # Bold if column max
                if val == col_maxes[col_idx]:
                    val_str = f"<b>{val_str}</b>"

                row_text.append(val_str)
            cell_text.append(row_text)

        # Transpose for plotly (columns, not rows)
        cell_values = [[row[c] for row in cell_text] for c in range(n_metrics)]
        cell_colors = [
            [self._value_to_color(row[c]) for row in row_values]
            for c in range(n_metrics)
        ]

        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Model/SAE</b>"] + [f"<b>{m}</b>" for m in metrics],
                    fill_color="rgb(240, 240, 240)",
                    align="center",
                    font=dict(size=12),
                    height=30,
                ),
                cells=dict(
                    values=[row_labels] + cell_values,
                    fill_color=[["white"] * len(row_labels)] + cell_colors,
                    align=["left"] + ["center"] * n_metrics,
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def _value_to_color(self, value: float) -> str:
        """Convert a 0-1 value to a white-to-blue color."""
        # Clamp value to [0, 1]
        v = max(0.0, min(1.0, value))
        # Interpolate from white (255, 255, 255) to blue (66, 133, 244)
        r = int(255 - v * (255 - 66))
        g = int(255 - v * (255 - 133))
        b = int(255 - v * (255 - 244))
        return f"rgb({r}, {g}, {b})"

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
        )


plot_sae_metrics_table = SAEMetricsTablePlot()


class SAECoreMetricsTablePlot(SinglePlot):
    """Table comparing core SAE metrics: Explained Variance, MCC, and F1.

    Use case:
        Compare reconstruction quality (explained variance), feature alignment
        (MCC), and classification performance (F1) across multiple SAEs or
        models in a tabular format with visual highlighting.

    Data:
        - `model.saes_explained_variance`: Variance explained by SAE reconstruction
        - `model.saes_mcc`: Mean Correlation Coefficient (decoder vs ground truth)
        - `model.saes_f1_score`: Harmonic mean of precision and recall

    Visualization:
        - Table with rows for each SAE/model combination
        - Columns for 3 core metrics
        - Cells color-coded 0 (white) to 1 (blue)
        - Column max values bolded
        - Percentage difference from first row shown in parentheses

    Customization:
        - `group_by`: "model" or "sae" - determines row grouping (default: "model")
        - `sae_labels`: Optional list of SAE labels to include
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(
        self,
        group_by: Literal["model", "sae"] = "model",
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAECoreMetricsTablePlot.

        Args:
            group_by: How to group rows - "model" groups by model with SAEs
                as sub-rows, "sae" groups by SAE with models as sub-rows.
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.group_by = group_by
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the metrics table.

        Args:
            fig: FigureProxy for adding traces.
            models: 1D ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAECoreMetricsTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        if len(models.axes) < 1:
            fig.add_annotation(
                text="SAECoreMetricsTablePlot requires a 1D ModelGrid.",
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

        # Collect data
        metrics = ["Explained Var", "MCC", "F1"]
        metric_props = ["saes_explained_variance", "saes_mcc", "saes_f1_score"]

        # Build rows based on grouping, tracking SAE label per row for baseline lookup
        row_labels: list[str] = []
        row_values: list[list[float]] = []
        row_sae_labels: list[str] = []  # Track which SAE each row belongs to

        # Get baseline values from first model for each SAE
        first_model = models[0]
        baselines: dict[str, list[float]] = {
            sae_label: [getattr(first_model, prop)[sae_label] for prop in metric_props]
            for sae_label in sae_labels
        }

        if self.group_by == "model":
            # Rows: Model (SAE) format
            for i, model in enumerate(models):
                axis_val = render_axis.values[i]
                for sae_label in sae_labels:
                    if len(sae_labels) == 1:
                        row_labels.append(f"{axis_val}")
                    else:
                        row_labels.append(f"{axis_val} ({sae_label})")
                    row_values.append(
                        [getattr(model, prop)[sae_label] for prop in metric_props]
                    )
                    row_sae_labels.append(sae_label)
        else:  # group_by == "sae"
            # Rows: SAE (Model) format
            for sae_label in sae_labels:
                for i, model in enumerate(models):
                    axis_val = render_axis.values[i]
                    if len(models) == 1:
                        row_labels.append(f"{sae_label}")
                    else:
                        row_labels.append(f"{sae_label} ({axis_val})")
                    row_values.append(
                        [getattr(model, prop)[sae_label] for prop in metric_props]
                    )
                    row_sae_labels.append(sae_label)

        # Calculate column maxes for bolding
        n_metrics = len(metrics)
        col_maxes = [
            max(row_values[r][c] for r in range(len(row_values)))
            for c in range(n_metrics)
        ]

        # Format cell text with bold max and % diff
        cell_text: list[list[str]] = []
        for row_idx, row in enumerate(row_values):
            row_text = []
            sae_label = row_sae_labels[row_idx]
            baseline = baselines[sae_label]
            is_baseline_row = row == baseline

            for col_idx, val in enumerate(row):
                # Format value
                val_str = f"{val:.3f}"

                # Add % diff (skip for baseline row of this SAE)
                if not is_baseline_row:
                    base_val = baseline[col_idx]
                    if base_val != 0:
                        pct_diff = ((val - base_val) / base_val) * 100
                        sign = "+" if pct_diff >= 0 else ""
                        val_str += f" ({sign}{pct_diff:.1f}%)"
                    else:
                        val_str += " (N/A)"

                # Bold if column max
                if val == col_maxes[col_idx]:
                    val_str = f"<b>{val_str}</b>"

                row_text.append(val_str)
            cell_text.append(row_text)

        # Transpose for plotly (columns, not rows)
        cell_values = [[row[c] for row in cell_text] for c in range(n_metrics)]
        cell_colors = [
            [self._value_to_color(row[c]) for row in row_values]
            for c in range(n_metrics)
        ]

        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Model/SAE</b>"] + [f"<b>{m}</b>" for m in metrics],
                    fill_color="rgb(240, 240, 240)",
                    align="center",
                    font=dict(size=12),
                    height=30,
                ),
                cells=dict(
                    values=[row_labels] + cell_values,
                    fill_color=[["white"] * len(row_labels)] + cell_colors,
                    align=["left"] + ["center"] * n_metrics,
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def _value_to_color(self, value: float) -> str:
        """Convert a 0-1 value to a white-to-blue color."""
        # Clamp value to [0, 1]
        v = max(0.0, min(1.0, value))
        # Interpolate from white (255, 255, 255) to blue (66, 133, 244)
        r = int(255 - v * (255 - 66))
        g = int(255 - v * (255 - 133))
        b = int(255 - v * (255 - 244))
        return f"rgb({r}, {g}, {b})"

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
        )


plot_sae_core_metrics_table = SAECoreMetricsTablePlot()


class SAESparsityMetricsTablePlot(SinglePlot):
    """Table comparing SAE sparsity metrics: L0, True L0, Dead Latents, Uniqueness.

    Use case:
        Compare sparsity behavior across multiple SAEs or models, showing
        L0 with ratio to true L0, dead latent counts/percentages, and uniqueness.

    Data:
        - `model.saes_l0`: L0 sparsity of SAE activations
        - `model.saes_true_l0`: Ground-truth feature activation L0 (for ratio)
        - `model.saes_dead_latents`: Count of dead SAE latents
        - `model.saes_uniqueness`: Fraction of SAE latents tracking unique features

    Visualization:
        - Table with rows for each SAE/model combination
        - L0 column shows value with (ratio to true L0) in brackets
        - Dead Latents shown as count and percentage
        - Uniqueness color-coded 0 (white) to 1 (blue)
        - Column max values bolded for uniqueness only
        - Percentage difference from first row shown in parentheses

    Customization:
        - `group_by`: "model" or "sae" - determines row grouping (default: "model")
        - `sae_labels`: Optional list of SAE labels to include
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(
        self,
        group_by: Literal["model", "sae"] = "model",
        sae_labels: list[str] | None = None,
    ):
        """Initialize SAESparsityMetricsTablePlot.

        Args:
            group_by: How to group rows - "model" groups by model with SAEs
                as sub-rows, "sae" groups by SAE with models as sub-rows.
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.group_by = group_by
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the metrics table.

        Args:
            fig: FigureProxy for adding traces.
            models: 1D ModelGrid containing trained and evaluated SAEs.
        """
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"SAESparsityMetricsTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        if len(models.axes) < 1:
            fig.add_annotation(
                text="SAESparsityMetricsTablePlot requires a 1D ModelGrid.",
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

        # Get total latents for percentage calculation (from first model, first SAE)
        first_sae = models[0].saes[sae_labels[0]].sae
        total_latents = first_sae.cfg.d_sae

        # Build rows based on grouping
        row_labels: list[str] = []
        # Store raw values: [l0, true_l0, dead_latents, uniqueness]
        row_values: list[list[float]] = []
        row_sae_labels: list[str] = []

        # Get baseline values from first model for each SAE
        first_model = models[0]
        baselines: dict[str, list[float]] = {}
        for sae_label in sae_labels:
            baselines[sae_label] = [
                first_model.saes_l0[sae_label],
                first_model.saes_true_l0[sae_label],
                first_model.saes_dead_latents[sae_label],
                first_model.saes_uniqueness[sae_label],
            ]

        if self.group_by == "model":
            for i, model in enumerate(models):
                axis_val = render_axis.values[i]
                for sae_label in sae_labels:
                    if len(sae_labels) == 1:
                        row_labels.append(f"{axis_val}")
                    else:
                        row_labels.append(f"{axis_val} ({sae_label})")
                    row_values.append(
                        [
                            model.saes_l0[sae_label],
                            model.saes_true_l0[sae_label],
                            model.saes_dead_latents[sae_label],
                            model.saes_uniqueness[sae_label],
                        ]
                    )
                    row_sae_labels.append(sae_label)
        else:  # group_by == "sae"
            for sae_label in sae_labels:
                for i, model in enumerate(models):
                    axis_val = render_axis.values[i]
                    if len(models) == 1:
                        row_labels.append(f"{sae_label}")
                    else:
                        row_labels.append(f"{sae_label} ({axis_val})")
                    row_values.append(
                        [
                            model.saes_l0[sae_label],
                            model.saes_true_l0[sae_label],
                            model.saes_dead_latents[sae_label],
                            model.saes_uniqueness[sae_label],
                        ]
                    )
                    row_sae_labels.append(sae_label)

        # Calculate column max for uniqueness (index 3)
        uniqueness_max = max(row[3] for row in row_values)

        # Format cells
        l0_col: list[str] = []
        dead_col: list[str] = []
        uniqueness_col: list[str] = []
        uniqueness_colors: list[str] = []

        for row_idx, row in enumerate(row_values):
            l0, true_l0, dead, uniqueness = row
            sae_label = row_sae_labels[row_idx]
            baseline = baselines[sae_label]
            is_baseline_row = row == baseline

            # L0 with ratio to true_l0
            ratio = l0 / true_l0 if true_l0 != 0 else float("inf")
            l0_str = f"{l0:.2f} ({ratio:.2f}x)"
            if not is_baseline_row and baseline[0] != 0:
                pct_diff = ((l0 - baseline[0]) / baseline[0]) * 100
                sign = "+" if pct_diff >= 0 else ""
                l0_str += f" [{sign}{pct_diff:.1f}%]"
            l0_col.append(l0_str)

            # Dead latents: count (percentage)
            dead_pct = (dead / total_latents) * 100 if total_latents > 0 else 0
            dead_str = f"{int(dead)} ({dead_pct:.1f}%)"
            if not is_baseline_row and baseline[2] != 0:
                pct_diff = ((dead - baseline[2]) / baseline[2]) * 100
                sign = "+" if pct_diff >= 0 else ""
                dead_str += f" [{sign}{pct_diff:.1f}%]"
            dead_col.append(dead_str)

            # Uniqueness (color-coded)
            uniqueness_str = f"{uniqueness:.3f}"
            if not is_baseline_row and baseline[3] != 0:
                pct_diff = ((uniqueness - baseline[3]) / baseline[3]) * 100
                sign = "+" if pct_diff >= 0 else ""
                uniqueness_str += f" ({sign}{pct_diff:.1f}%)"
            if uniqueness == uniqueness_max:
                uniqueness_str = f"<b>{uniqueness_str}</b>"
            uniqueness_col.append(uniqueness_str)

            # Color for uniqueness
            v = max(0.0, min(1.0, uniqueness))
            r = int(255 - v * (255 - 66))
            g = int(255 - v * (255 - 133))
            b = int(255 - v * (255 - 244))
            uniqueness_colors.append(f"rgb({r}, {g}, {b})")

        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "<b>Model/SAE</b>",
                        "<b>L0 (ratio)</b>",
                        "<b>Dead Latents</b>",
                        "<b>Uniqueness</b>",
                    ],
                    fill_color="rgb(240, 240, 240)",
                    align="center",
                    font=dict(size=12),
                    height=30,
                ),
                cells=dict(
                    values=[row_labels, l0_col, dead_col, uniqueness_col],
                    fill_color=[
                        ["white"] * len(row_labels),
                        ["white"] * len(row_labels),
                        ["white"] * len(row_labels),
                        uniqueness_colors,
                    ],
                    align=["left", "center", "center", "center"],
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
        )


plot_sae_sparsity_metrics_table = SAESparsityMetricsTablePlot()
