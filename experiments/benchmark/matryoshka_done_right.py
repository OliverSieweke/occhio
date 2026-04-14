"""Load and analyze the matryoshka_total benchmark grid."""

from pathlib import Path

import dill

from occhio.model_grid import ModelGrid

GRID_PATH = Path(__file__).parent / "sweep-analysis/data/matryoshka_total/grid.pkl"


def load_grid(path: Path = GRID_PATH) -> ModelGrid:
    """Load a ModelGrid pickle."""
    with open(path, "rb") as f:
        return dill.load(f)


if __name__ == "__main__":
    grid = load_grid()

    print(f"Grid shape: {grid.shape}")
    print(f"Axes: {grid.description}")

    first_model = grid.models.ravel()[0]
    print(f"\nFirst model SAEs: {list(first_model.saes.keys())}")
    print(f"Number of models: {grid.models.size}")

    grid.evaluate_saes()

    sae = first_model.saes["Matryoshka_0"]
    print(sae.metrics)
