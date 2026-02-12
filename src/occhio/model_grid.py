from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
from stack_data.utils import cached_property
from tqdm import tqdm

from occhio import ToyModel

X = TypeVar("X")
Y = TypeVar("Y")


@dataclass
class Axis:
    label: str
    values: list


class ModelGrid:
    models: list[list[ToyModel]]

    def __init__(
        self,
        model_trainer: Callable[[X, Y], ToyModel],
        x_axis: Axis,
        y_axis: Axis,
    ):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.models = []

        with tqdm(
            total=len(x_axis.values) * len(y_axis.values),
            desc="Training Model Grid",
            unit="model",
        ) as pbar:
            for y_val in y_axis.values:
                row = []
                for x_val in x_axis.values:
                    # [11.02.26 | OliverSieweke] TODO: Should be vecotrized once we get there.
                    model = model_trainer(x_val, y_val)
                    row.append(model)
                    pbar.update(1)

                self.models.append(row)

    def __getitem__(self, key: tuple[int, int]) -> ToyModel:
        i, j = key
        return self.models[i][j]

    @cached_property
    def parameters_mesh(self):
        return np.meshgrid(
            self.x_axis.values,
            self.y_axis.values,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.x_axis.values), len(self.y_axis.values))
