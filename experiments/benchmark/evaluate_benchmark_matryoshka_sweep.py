# %%
from sae_lens import (
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

from occhio import ToyModel, benchmark

# %%
DEVICE = "cpu"


# %%
def create_saes(model: ToyModel):

    return {
        f"Matryoshka_{k}": MatryoshkaBatchTopKTrainingSAE(
            MatryoshkaBatchTopKTrainingSAEConfig(
                d_in=64,
                d_sae=648,
                matryoshka_widths=[
                    81,
                    324,
                    648,
                ],
                k=k,
                use_matryoshka_aux_loss=True,
                device=DEVICE,
            )
        )
        for k in [2, 4, 8, 12, 16]
    }


# %%

# %%
grid = benchmark.evaluate(create_saes, device=DEVICE, verbose=True)
# %%
grid.save("benchmark_standard_sweep_dag")
# %%
import importlib

import occhio.visualization_2.plots.experimental.sae_benchmark_table as sae_benchmark_table_module

importlib.reload(sae_benchmark_table_module)
import occhio.visualization_2.plots as plots_module

importlib.reload(plots_module)
import occhio.visualization_2 as viz_module

importlib.reload(viz_module)
from occhio.visualization_2 import plot_sae_benchmark_table

plot_sae_benchmark_table(grid, width=2000)
# %%
