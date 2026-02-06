# ABOUTME: List-like container for toy models with aligned variation metadata.
# ABOUTME: Supports selection, slicing, and mutation while keeping metadata in sync.

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..toy_model import ToyModel


class Box(list[ToyModel]):
    def __init__(self, models: Iterable[ToyModel]):
        super().__init__(models)
        if not self:
            msg = "Box must contain at least one model."
            raise ValueError(msg)
        self._variations: dict[str, list[Any]] = {}

    @property
    def models(self) -> list[ToyModel]:
        return self

    @property
    def size(self) -> int:
        return len(self)

    def variation_names(self) -> list[str]:
        return list(self._variations.keys())

    def add(self, model: ToyModel, index: int | None = None) -> None:
        if index is None:
            self.append(model)
            return
        self.insert(index, model)

    def add_variation(self, name: str, values: Iterable[Any]) -> None:
        if name in self._variations:
            msg = f"Variation '{name}' already exists."
            raise ValueError(msg)
        values_list = list(values)
        if len(values_list) != self.size:
            msg = (
                f"Variation '{name}' must have same length as models "
                f"({len(values_list)} != {self.size})."
            )
            raise ValueError(msg)
        self._variations[name] = values_list

    def select_ids(self, **criteria: Any) -> list[int]:
        self._validate_criteria(criteria)
        selected_ids: list[int] = []
        for idx in range(self.size):
            if all(self._variations[name][idx] == expected for name, expected in criteria.items()):
                selected_ids.append(idx)
        return selected_ids

    def slice(self, ids: Iterable[int]) -> Box:
        selected_ids = list(ids)
        return self._subset(selected_ids)

    def append(self, model: ToyModel) -> None:
        super().append(model)
        self._append_missing_variations()

    def extend(self, models: Iterable[ToyModel]) -> None:
        models_list = list(models)
        if not models_list:
            return
        super().extend(models_list)
        for values in self._variations.values():
            values.extend([None] * len(models_list))

    def insert(self, index: int, model: ToyModel) -> None:
        resolved_index = self._resolve_insert_index(index)
        super().insert(index, model)
        for values in self._variations.values():
            values.insert(resolved_index, None)

    def __getitem__(self, index: int | slice) -> ToyModel | Box:
        if isinstance(index, slice):
            ids = list(range(self.size))[index]
            return self._subset(ids)
        return super().__getitem__(index)

    def __setitem__(self, index: int | slice, value: ToyModel | Iterable[ToyModel]) -> None:
        if isinstance(index, slice):
            replacement = list(value)  # type: ignore[arg-type]
            super().__setitem__(index, replacement)
            for values in self._variations.values():
                values[index] = [None] * len(replacement)
            return
        super().__setitem__(index, value)  # type: ignore[arg-type]

    def __delitem__(self, index: int | slice) -> None:
        super().__delitem__(index)
        for values in self._variations.values():
            del values[index]

    def pop(self, index: int = -1) -> ToyModel:
        model = super().pop(index)
        for values in self._variations.values():
            values.pop(index)
        return model

    def remove(self, value: ToyModel) -> None:
        index = self.index(value)
        self.pop(index)

    def clear(self) -> None:
        super().clear()
        for values in self._variations.values():
            values.clear()

    def reverse(self) -> None:
        super().reverse()
        for values in self._variations.values():
            values.reverse()

    def sort(
        self,
        *,
        key: Callable[[ToyModel], Any] | None = None,
        reverse: bool = False,
    ) -> None:
        indices = sorted(
            range(self.size),
            key=(lambda idx: key(self[idx])) if key is not None else (lambda idx: self[idx]),
            reverse=reverse,
        )
        ordered_models = [self[idx] for idx in indices]
        super().__setitem__(slice(None), ordered_models)
        for values in self._variations.values():
            values[:] = [values[idx] for idx in indices]

    def __iadd__(self, values: Iterable[ToyModel]) -> Box:
        self.extend(values)
        return self

    def _append_missing_variations(self) -> None:
        for values in self._variations.values():
            values.append(None)

    def _subset(self, ids: list[int]) -> Box:
        subset_models = [self[idx] for idx in ids]
        subset_box = Box(subset_models)
        for name, values in self._variations.items():
            subset_box.add_variation(name, [values[idx] for idx in ids])
        return subset_box

    def _resolve_insert_index(self, index: int) -> int:
        if index >= self.size:
            return self.size
        if index < 0:
            return max(self.size + index, 0)
        return index

    def _validate_criteria(self, criteria: Mapping[str, Any]) -> None:
        for name in criteria:
            if name not in self._variations:
                msg = f"Unknown variation '{name}'."
                raise KeyError(msg)
