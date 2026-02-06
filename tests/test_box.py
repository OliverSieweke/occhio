# ABOUTME: Tests for the scheduler Box container used to manage many toy models.
# ABOUTME: Verifies variation registration, selection, and slicing behavior.

import pytest

from occhio.autoencoder import TiedLinear
from occhio.distributions.sparse import SparseUniform
from occhio.scheduler.box import Box
from occhio.toy_model import ToyModel


def _build_toy_model(n_features: int, n_hidden: int, p_active: float) -> ToyModel:
    distribution = SparseUniform(n_features, p_active=p_active)
    ae = TiedLinear(n_features=n_features, n_hidden=n_hidden)
    return ToyModel(distribution=distribution, ae=ae)


def test_box_stores_models_and_variations():
    models = [
        _build_toy_model(8, 2, 0.1),
        _build_toy_model(8, 3, 0.2),
        _build_toy_model(8, 4, 0.3),
    ]
    box = Box(models)

    box.add_variation("distribution_name", ["sparse", "sparse", "sparse"])
    box.add_variation("hidden_size", [2, 3, 4])

    assert isinstance(box, list)
    assert len(box) == 3
    assert box[1] is models[1]
    assert box.size == 3
    assert box.variation_names() == ["distribution_name", "hidden_size"]


def test_select_ids_filters_by_variations():
    models = [
        _build_toy_model(8, 2, 0.1),
        _build_toy_model(8, 2, 0.2),
        _build_toy_model(8, 4, 0.1),
    ]
    box = Box(models)
    box.add_variation("hidden_size", [2, 2, 4])
    box.add_variation("sparsity", [0.1, 0.2, 0.1])

    assert box.select_ids(hidden_size=2) == [0, 1]
    assert box.select_ids(hidden_size=2, sparsity=0.1) == [0]


def test_slice_keeps_models_and_variation_values():
    models = [
        _build_toy_model(8, 2, 0.1),
        _build_toy_model(8, 2, 0.2),
        _build_toy_model(8, 4, 0.3),
    ]
    box = Box(models)
    box.add_variation("hidden_size", [2, 2, 4])

    sliced = box[1:]

    assert sliced.size == 2
    assert isinstance(sliced, Box)
    assert list(sliced) == [models[1], models[2]]
    assert sliced.select_ids(hidden_size=4) == [1]


def test_append_keeps_variations_aligned():
    models = [_build_toy_model(8, 2, 0.1), _build_toy_model(8, 2, 0.2)]
    box = Box(models)
    box.add_variation("hidden_size", [2, 2])

    box.append(_build_toy_model(8, 3, 0.3))

    assert len(box) == 3
    assert box.select_ids(hidden_size=2) == [0, 1]
    assert box.select_ids(hidden_size=None) == [2]


def test_add_variation_rejects_wrong_length():
    models = [_build_toy_model(8, 2, 0.1), _build_toy_model(8, 3, 0.2)]
    box = Box(models)

    with pytest.raises(ValueError, match="same length"):
        box.add_variation("hidden_size", [2])


def test_select_ids_rejects_unknown_variation():
    models = [_build_toy_model(8, 2, 0.1)]
    box = Box(models)

    with pytest.raises(KeyError, match="Unknown variation"):
        box.select_ids(hidden_size=2)
