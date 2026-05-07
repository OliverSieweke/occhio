"""SAE metrics summary dashboard.

This module provides a three-tier table dashboard for comparing SAE evaluation
metrics across a ModelGrid. Each tier focuses on a different aspect:
- Performance & Fidelity: F1, MCC
- Interpretability & Usability: L0, True L0, Uniqueness
- Diagnostic: Explained Variance, Dead Latents

Tables are pivoted with SAEs as rows and models as sub-columns under each metric,
enabling easy side-by-side comparison across models. Values show ratio to first
model in brackets.
"""

import plotly.graph_objects as go

from occhio import ModelGrid
from occhio.toy_model import ToyModel
from occhio.visualization.core import CompositePlot
from occhio.visualization.core.base_plot import SinglePlot
from occhio.visualization.core.figure_wrappers import FigureProxy

# Alternating column colors for model distinction
MODEL_COLORS = ["rgb(240, 248, 255)", "rgb(255, 250, 240)"]  # AliceBlue, FloralWhite


def _get_sae_labels_and_check(
    fig: FigureProxy,
    models: ModelGrid,
    requested_sae_labels: list[str] | None,
) -> list[str] | None:
    """Get validated SAE labels or add error annotation and return None."""
    first_model = models[0]
    if not first_model.saes:
        fig.add_annotation(
            text="No SAEs trained.<br>Call model.train_saes() first.",
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12, color="firebrick"),
        )
        return None

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
        return None

    sae_labels = (
        [lbl for lbl in requested_sae_labels if lbl in common_sae_labels]
        if requested_sae_labels
        else sorted(common_sae_labels)
    )

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
        return None

    return sae_labels


def _format_with_pct_diff(
    value: float, baseline: float, fmt: str = ".3f", bold: bool = False
) -> str:
    """Format value with percentage difference from baseline in brackets."""
    if baseline == 0:
        text = f"{value:{fmt}}"
    else:
        pct_diff = ((value - baseline) / baseline) * 100
        sign = "+" if pct_diff >= 0 else ""
        text = f"{value:{fmt}} [{sign}{pct_diff:.0f}%]"
    return f"<b>{text}</b>" if bold else text


def _format_value(value: float, fmt: str = ".3f", bold: bool = False) -> str:
    """Format value with optional bold."""
    text = f"{value:{fmt}}"
    return f"<b>{text}</b>" if bold else text


def _get_model_column_colors(
    n_models: int, n_metrics: int, n_rows: int
) -> list[list[str]]:
    """Get column colors alternating by model index, repeated for each metric."""
    # SAE column is white
    colors = [["white"] * n_rows]
    # For each metric, repeat the model colors pattern
    for _ in range(n_metrics):
        for model_idx in range(n_models):
            colors.append([MODEL_COLORS[model_idx % len(MODEL_COLORS)]] * n_rows)
    return colors


class PerformanceFidelityTablePlot(SinglePlot):
    """Table of SAE performance and fidelity metrics: F1,  and MCC.

    Use case:
        Compare feature-finding accuracy across SAEs and models. Shows whether
        the SAE is finding the right features and using them correctly.

    Data:
        - `model.saes_f1_score`: Harmonic mean of precision and recall (per SAE latent)
        - `model.saes_macro_f1`: Unweighted mean of per-feature F1 (each GT feature counts equally)
        - `model.saes_mcc`: Mean Correlation Coefficient (decoder vs ground truth)

    Visualization:
        - Table with one row per SAE
        - Columns grouped by metric (F1 F1, MCC), with sub-columns for each model
        - Columns color-coded by model for easy comparison
        - Values show ratio to first model in brackets

    Customization:
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(self, sae_labels: list[str] | None = None):
        """Initialize PerformanceFidelityTablePlot.

        Args:
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the performance and fidelity metrics table."""
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"PerformanceFidelityTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        sae_labels = _get_sae_labels_and_check(fig, models, self.sae_labels)
        if sae_labels is None:
            return

        axis = models.axes[0]
        n_models = len(models)
        model_labels = [str(axis.values[i]) for i in range(n_models)]

        # Build header
        header_row1 = (
            ["<b>Performance & Fidelity</b>"]
            + ["<b>F1</b>"]
            + [""] * (n_models - 1)
            + ["<b>MCC</b>"]
            + [""] * (n_models - 1)
        )
        header_row2 = ["<b>SAE</b>"] + model_labels * 2

        header_values = [
            [header_row1[i], header_row2[i]] for i in range(len(header_row1))
        ]

        title_color = "rgb(66, 133, 244)"
        # Header colors: title row + model-colored second row
        header_colors = [[title_color, "rgb(240, 240, 240)"]]  # SAE column
        for _ in range(2):  # 2 metrics (F1, MCC)
            for model_idx in range(n_models):
                header_colors.append(
                    [title_color, MODEL_COLORS[model_idx % len(MODEL_COLORS)]]
                )

        # Collect raw values first to find best per column (maximize F1, MCC)
        f1_raw = [
            [model.saes_f1_score[label] for label in sae_labels] for model in models
        ]
        mcc_raw = [[model.saes_mcc[label] for label in sae_labels] for model in models]

        # Find best (max) value per column (across SAEs for each model)
        f1_best = [max(f1_raw[m]) for m in range(n_models)]
        mcc_best = [max(mcc_raw[m]) for m in range(n_models)]

        # Build formatted columns with bolding for best
        sae_col = sae_labels
        f1_cols = []
        mcc_cols = []
        for model_idx, model in enumerate(models):
            f1_col = []
            mcc_col = []
            for sae_idx, label in enumerate(sae_labels):
                f1_val = f1_raw[model_idx][sae_idx]
                mcc_val = mcc_raw[model_idx][sae_idx]
                f1_baseline = f1_raw[0][sae_idx]
                mcc_baseline = mcc_raw[0][sae_idx]
                f1_is_best = f1_val == f1_best[model_idx]
                mcc_is_best = mcc_val == mcc_best[model_idx]

                if model_idx == 0:
                    f1_col.append(_format_value(f1_val, ".3f", f1_is_best))
                    mcc_col.append(_format_value(mcc_val, ".3f", mcc_is_best))
                else:
                    f1_col.append(
                        _format_with_pct_diff(f1_val, f1_baseline, ".3f", f1_is_best)
                    )
                    mcc_col.append(
                        _format_with_pct_diff(mcc_val, mcc_baseline, ".3f", mcc_is_best)
                    )
            f1_cols.append(f1_col)
            mcc_cols.append(mcc_col)

        cell_values = [sae_col] + f1_cols + mcc_cols
        fill_color = _get_model_column_colors(n_models, 2, len(sae_labels))
        align = ["left"] + ["center"] * (n_models * 2)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color=header_colors,
                    font=dict(size=12, color="black"),
                    align=align,
                    height=28,
                    line_color="white",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fill_color,
                    align=align,
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))


class InterpretabilityTablePlot(SinglePlot):
    """Table of SAE interpretability and usability metrics: L0, True L0, Uniqueness.

    Use case:
        Assess whether a human or downstream circuit can use what the SAE learned.
        Shows sparsity behavior and feature uniqueness.

    Data:
        - `model.saes_l0`: L0 sparsity of SAE activations
        - `model.saes_true_l0`: Ground-truth feature activation L0
        - `model.saes_uniqueness`: Fraction of SAE latents tracking unique features

    Visualization:
        - Table with one row per SAE
        - Columns grouped by metric (L0, True L0, Uniqueness), with sub-columns for each model
        - Columns color-coded by model for easy comparison
        - Values show ratio to first model in brackets

    Customization:
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(self, sae_labels: list[str] | None = None):
        """Initialize InterpretabilityTablePlot.

        Args:
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the interpretability and usability metrics table."""
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"InterpretabilityTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        sae_labels = _get_sae_labels_and_check(fig, models, self.sae_labels)
        if sae_labels is None:
            return

        axis = models.axes[0]
        n_models = len(models)
        model_labels = [str(axis.values[i]) for i in range(n_models)]

        # Build header
        header_row1 = (
            ["<b>Interpretability & Usability</b>"]
            + ["<b>L0</b>"]
            + [""] * (n_models - 1)
            + ["<b>True L0</b>"]
            + [""] * (n_models - 1)
            + ["<b>Uniqueness</b>"]
            + [""] * (n_models - 1)
        )
        header_row2 = ["<b>SAE</b>"] + model_labels * 3

        header_values = [
            [header_row1[i], header_row2[i]] for i in range(len(header_row1))
        ]

        title_color = "rgb(52, 168, 83)"
        header_colors = [[title_color, "rgb(240, 240, 240)"]]
        for _ in range(3):  # 3 metrics
            for model_idx in range(n_models):
                header_colors.append(
                    [title_color, MODEL_COLORS[model_idx % len(MODEL_COLORS)]]
                )

        # Collect raw values first to find best per column
        # L0: minimize, True L0: no bolding (same across models), Uniqueness: maximize
        l0_raw = [[model.saes_l0[label] for label in sae_labels] for model in models]
        true_l0_raw = [
            [model.saes_true_l0[label] for label in sae_labels] for model in models
        ]
        uniqueness_raw = [
            [model.saes_uniqueness[label] for label in sae_labels] for model in models
        ]

        # Find best values per column (across SAEs for each model)
        l0_best = [min(l0_raw[m]) for m in range(n_models)]
        uniqueness_best = [max(uniqueness_raw[m]) for m in range(n_models)]

        # Build formatted columns with bolding for best
        sae_col = sae_labels
        l0_cols = []
        true_l0_cols = []
        uniqueness_cols = []
        for model_idx, model in enumerate(models):
            l0_col = []
            true_l0_col = []
            uniqueness_col = []
            for sae_idx, label in enumerate(sae_labels):
                l0_val = l0_raw[model_idx][sae_idx]
                true_l0_val = true_l0_raw[model_idx][sae_idx]
                uniqueness_val = uniqueness_raw[model_idx][sae_idx]
                l0_baseline = l0_raw[0][sae_idx]
                true_l0_baseline = true_l0_raw[0][sae_idx]
                uniqueness_baseline = uniqueness_raw[0][sae_idx]
                l0_is_best = l0_val == l0_best[model_idx]
                uniqueness_is_best = uniqueness_val == uniqueness_best[model_idx]

                if model_idx == 0:
                    l0_col.append(_format_value(l0_val, ".1f", l0_is_best))
                    true_l0_col.append(f"{true_l0_val:.1f}")
                    uniqueness_col.append(
                        _format_value(uniqueness_val, ".3f", uniqueness_is_best)
                    )
                else:
                    l0_col.append(
                        _format_with_pct_diff(l0_val, l0_baseline, ".1f", l0_is_best)
                    )
                    true_l0_col.append(
                        _format_with_pct_diff(true_l0_val, true_l0_baseline, ".1f")
                    )
                    uniqueness_col.append(
                        _format_with_pct_diff(
                            uniqueness_val,
                            uniqueness_baseline,
                            ".3f",
                            uniqueness_is_best,
                        )
                    )
            l0_cols.append(l0_col)
            true_l0_cols.append(true_l0_col)
            uniqueness_cols.append(uniqueness_col)

        cell_values = [sae_col] + l0_cols + true_l0_cols + uniqueness_cols
        fill_color = _get_model_column_colors(n_models, 3, len(sae_labels))
        align = ["left"] + ["center"] * (n_models * 3)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color=header_colors,
                    font=dict(size=12, color="black"),
                    align=align,
                    height=28,
                    line_color="white",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fill_color,
                    align=align,
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))


class DiagnosticTablePlot(SinglePlot):
    """Table of SAE diagnostic metrics: Explained Variance and Dead Latents.

    Use case:
        Diagnose why SAE performance is what it is. Shows reconstruction quality
        and latent utilization to identify failure modes.

    Data:
        - `model.saes_explained_variance`: Variance explained by SAE reconstruction
        - `model.saes_dead_latents`: Count of dead SAE latents

    Visualization:
        - Table with one row per SAE
        - Columns grouped by metric (Expl. Variance, Dead Latents), with sub-columns for each model
        - Columns color-coded by model for easy comparison
        - Values show ratio to first model in brackets

    Customization:
        - `sae_labels`: Optional list of SAE labels to include (default: all)
    """

    n_render_axes = 1
    subplot_type = "table"

    def __init__(self, sae_labels: list[str] | None = None):
        """Initialize DiagnosticTablePlot.

        Args:
            sae_labels: Optional list of SAE labels to include. If None, includes all.
        """
        self.sae_labels = sae_labels

    def render(
        self,
        fig: FigureProxy,
        models: ToyModel | ModelGrid,
    ) -> None:
        """Render the diagnostic metrics table."""
        if not isinstance(models, ModelGrid):
            raise TypeError(
                f"DiagnosticTablePlot requires a ModelGrid, got {type(models).__name__}"
            )

        sae_labels = _get_sae_labels_and_check(fig, models, self.sae_labels)
        if sae_labels is None:
            return

        axis = models.axes[0]
        n_models = len(models)
        model_labels = [str(axis.values[i]) for i in range(n_models)]

        # Build header
        header_row1 = (
            ["<b>Diagnostic</b>"]
            + ["<b>Expl. Variance</b>"]
            + [""] * (n_models - 1)
            + ["<b>Dead Latents</b>"]
            + [""] * (n_models - 1)
        )
        header_row2 = ["<b>SAE</b>"] + model_labels * 2

        header_values = [
            [header_row1[i], header_row2[i]] for i in range(len(header_row1))
        ]

        title_color = "rgb(251, 188, 4)"
        header_colors = [[title_color, "rgb(240, 240, 240)"]]
        for _ in range(2):  # 2 metrics
            for model_idx in range(n_models):
                header_colors.append(
                    [title_color, MODEL_COLORS[model_idx % len(MODEL_COLORS)]]
                )

        # Collect raw values first to find best per column
        # Expl. Variance: maximize, Dead Latents: minimize
        expl_var_raw = [
            [model.saes_explained_variance[label] for label in sae_labels]
            for model in models
        ]
        dead_latents_raw = [
            [model.saes_dead_latents[label] for label in sae_labels] for model in models
        ]

        # Find best values per column (across SAEs for each model)
        expl_var_best = [max(expl_var_raw[m]) for m in range(n_models)]
        dead_latents_best = [min(dead_latents_raw[m]) for m in range(n_models)]

        # Build formatted columns with bolding for best
        sae_col = sae_labels
        expl_var_cols = []
        dead_latents_cols = []
        for model_idx, model in enumerate(models):
            expl_var_col = []
            dead_latents_col = []
            for sae_idx, label in enumerate(sae_labels):
                expl_var_val = expl_var_raw[model_idx][sae_idx]
                dead_latents_val = dead_latents_raw[model_idx][sae_idx]
                expl_var_baseline = expl_var_raw[0][sae_idx]
                dead_latents_baseline = dead_latents_raw[0][sae_idx]
                expl_var_is_best = expl_var_val == expl_var_best[model_idx]
                dead_latents_is_best = dead_latents_val == dead_latents_best[model_idx]

                if model_idx == 0:
                    expl_var_col.append(
                        _format_value(expl_var_val, ".3f", expl_var_is_best)
                    )
                    text = f"{dead_latents_val}"
                    dead_latents_col.append(
                        f"<b>{text}</b>" if dead_latents_is_best else text
                    )
                else:
                    expl_var_col.append(
                        _format_with_pct_diff(
                            expl_var_val, expl_var_baseline, ".3f", expl_var_is_best
                        )
                    )
                    # For dead latents, show pct diff only if baseline > 0
                    if dead_latents_baseline > 0:
                        pct_diff = (
                            (dead_latents_val - dead_latents_baseline)
                            / dead_latents_baseline
                        ) * 100
                        sign = "+" if pct_diff >= 0 else ""
                        text = f"{dead_latents_val} [{sign}{pct_diff:.0f}%]"
                    else:
                        text = f"{dead_latents_val}"
                    dead_latents_col.append(
                        f"<b>{text}</b>" if dead_latents_is_best else text
                    )
            expl_var_cols.append(expl_var_col)
            dead_latents_cols.append(dead_latents_col)

        cell_values = [sae_col] + expl_var_cols + dead_latents_cols
        fill_color = _get_model_column_colors(n_models, 2, len(sae_labels))
        align = ["left"] + ["center"] * (n_models * 2)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color=header_colors,
                    font=dict(size=12, color="black"),
                    align=align,
                    height=28,
                    line_color="white",
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fill_color,
                    align=align,
                    font=dict(size=11),
                    height=28,
                ),
            )
        )

    def configure_layout(self, fig: go.Figure) -> None:
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))


# Composite plot instance for easy use
plot_sae_metrics_summary = CompositePlot(
    layout=[
        [PerformanceFidelityTablePlot()],
        [InterpretabilityTablePlot()],
        [DiagnosticTablePlot()],
    ],
    row_heights=[1, 1, 1],
    column_widths=[1],
)
