"""Extended Standard SAE with L1 coefficient autotuning for L0 targeting.

Copied from https://github.com/decoderesearch/synth-sae-bench-experiments/blob/main/saes/standard_sae.py
"""

from dataclasses import dataclass

import torch
from sae_lens import (
    SAEConfig,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from typing_extensions import override

from occhio.sae_lens_adapter.coefficient_autotuner import (
    CoefficientAutotuner,
    CoefficientAutotunerConfig,
)


@dataclass
class StandardTrainingSAEConfigAutotuned(StandardTrainingSAEConfig):
    """
    Extended standard SAE config that supports autotuning l1 coefficient.

    When autotune_target_l0 is not None:
    - A multiplier is autotuned during training to reach the target L0.
    - The effective coefficient is: l1_coefficient * multiplier
    """

    autotune_target_l0: float | None = None
    autotune_start_step: int = 0
    autotune_smoothing_factor: float = 0.99
    autotune_rate_smoothing_factor: float = 0.95
    autotune_integral_gain: float = 3e-4
    autotune_min_multiplier: float = 1e-2
    autotune_max_multiplier: float = 100.0
    autotune_deadband: float = 0.0
    autotune_gain_scale: float = 10.0
    autotune_convergence_gain: float = 0.01

    @override
    @classmethod
    def architecture(cls) -> str:
        return "xstandard"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return StandardSAEConfig

    def get_autotuner_config(self) -> CoefficientAutotunerConfig | None:
        """Returns autotuner config if autotuning is enabled, else None."""
        if self.autotune_target_l0 is None:
            return None
        return CoefficientAutotunerConfig(
            target_l0=self.autotune_target_l0,
            start_step=self.autotune_start_step,
            smoothing_factor=self.autotune_smoothing_factor,
            rate_smoothing_factor=self.autotune_rate_smoothing_factor,
            integral_gain=self.autotune_integral_gain,
            min_multiplier=self.autotune_min_multiplier,
            max_multiplier=self.autotune_max_multiplier,
            deadband=self.autotune_deadband,
            gain_scale=self.autotune_gain_scale,
            convergence_gain=self.autotune_convergence_gain,
        )


class StandardTrainingSAEAutotuned(StandardTrainingSAE):
    """
    Extended Standard Training SAE that supports autotuning l1 coefficient.
    """

    cfg: StandardTrainingSAEConfigAutotuned  # type: ignore
    coefficient_autotuner: CoefficientAutotuner | None

    def __init__(
        self, cfg: StandardTrainingSAEConfigAutotuned, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term=use_error_term)

        autotuner_cfg = cfg.get_autotuner_config()
        if autotuner_cfg is not None:
            self.coefficient_autotuner = CoefficientAutotuner(
                cfg=autotuner_cfg,
                device=cfg.device,
            )
        else:
            self.coefficient_autotuner = None

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        # Remove autotuner buffers — they're training-only state
        for key in list(state_dict.keys()):
            if key.startswith("coefficient_autotuner."):
                del state_dict[key]

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)

        if self.coefficient_autotuner is not None:
            # Calculate batch L0 from feature activations
            batch_l0 = (output.feature_acts != 0).float().sum(dim=-1).mean()

            # Update autotuner and get new multiplier
            new_multiplier = self.coefficient_autotuner.update(
                batch_l0, step_input.n_training_steps
            )

            # Log autotuner metrics
            output.metrics["autotuner/multiplier"] = new_multiplier
            output.metrics["autotuner/effective_l1_coefficient"] = (
                step_input.coefficients["l1"] * new_multiplier
            )
            output.metrics["autotuner/smoothed_l0"] = (
                self.coefficient_autotuner.smoothed_l0
            )
            output.metrics["autotuner/batch_l0"] = batch_l0.item()

        return output

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del hidden_pre, sae_out  # unused

        # Get base coefficient from step_input, apply multiplier if autotuning
        l1_coefficient = step_input.coefficients["l1"]
        if self.coefficient_autotuner is not None:
            l1_coefficient = l1_coefficient * self.coefficient_autotuner.multiplier

        # Calculate weighted feature acts and sparsity loss (same as parent)
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        l1_loss = (l1_coefficient * sparsity).mean()

        return {"l1_loss": l1_loss}
