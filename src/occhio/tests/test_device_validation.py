"""Tests for device validation in ToyModel, AutoEncoderBase, and Distribution."""

import pytest
import torch

from ..autoencoder import TiedLinearRelu, TiedLinear
from ..distributions.sparse import SparseUniform
from ..toy_model import ToyModel, _same_device


class TestSameDevice:
    """Tests for the _same_device helper."""

    def test_identical_devices(self):
        assert _same_device(torch.device("cpu"), torch.device("cpu"))

    def test_index_none_vs_zero(self):
        """mps and mps:0 should be the same device."""
        assert _same_device(torch.device("cpu"), torch.device("cpu:0"))

    def test_different_types(self):
        assert not _same_device(torch.device("cpu"), torch.device("cuda:0"))

    def test_different_indices(self):
        """cuda:0 and cuda:1 should be different devices."""
        assert not _same_device(torch.device("cuda:0"), torch.device("cuda:1"))

    def test_same_index_explicit(self):
        assert _same_device(torch.device("cuda:0"), torch.device("cuda:0"))


class TestAutoEncoderDeviceValidation:
    """Tests for generator/device mismatch in AutoEncoderBase."""

    def test_no_device_no_generator(self):
        ae = TiedLinearRelu(5, 2)
        assert ae._init_device is None

    def test_explicit_device(self):
        ae = TiedLinearRelu(5, 2, device="cpu")
        assert ae._init_device == torch.device("cpu")

    def test_generator_infers_device(self):
        gen = torch.Generator(device="cpu")
        ae = TiedLinearRelu(5, 2, generator=gen)
        assert ae._init_device == torch.device("cpu")

    def test_matching_device_and_generator(self):
        gen = torch.Generator(device="cpu")
        ae = TiedLinearRelu(5, 2, device="cpu", generator=gen)
        assert ae._init_device == torch.device("cpu")

    def test_mismatched_device_and_generator_raises(self):
        gen = torch.Generator(device="cpu")
        with pytest.raises(ValueError, match="Generator lives on"):
            TiedLinearRelu(5, 2, device="cuda", generator=gen)


class TestDistributionDeviceValidation:
    """Tests for generator/device mismatch in Distribution."""

    def test_no_device_no_generator(self):
        dist = SparseUniform(5, 0.3)
        assert dist._init_device is None

    def test_explicit_device(self):
        dist = SparseUniform(5, 0.3, device="cpu")
        assert dist._init_device == torch.device("cpu")

    def test_generator_infers_device(self):
        gen = torch.Generator(device="cpu")
        dist = SparseUniform(5, 0.3, generator=gen)
        assert dist._init_device == torch.device("cpu")

    def test_matching_device_and_generator(self):
        gen = torch.Generator(device="cpu")
        dist = SparseUniform(5, 0.3, device="cpu", generator=gen)
        assert dist._init_device == torch.device("cpu")

    def test_mismatched_device_and_generator_raises(self):
        gen = torch.Generator(device="cpu")
        with pytest.raises(ValueError, match="Generator lives on"):
            SparseUniform(5, 0.3, device="cuda", generator=gen)


class TestToyModelDeviceConflict:
    """Tests for device conflict detection in ToyModel.__init__."""

    def test_no_devices_specified(self):
        """Should default to CPU without errors."""
        tm = ToyModel(SparseUniform(5, 0.3), TiedLinearRelu(5, 2))
        assert tm.device == torch.device("cpu")

    def test_only_toymodel_device(self):
        """Components without explicit device should be moved silently."""
        tm = ToyModel(SparseUniform(5, 0.3), TiedLinearRelu(5, 2), device="cpu")
        assert tm.device == torch.device("cpu")

    def test_matching_ae_device(self):
        """Matching explicit devices should work fine."""
        tm = ToyModel(
            SparseUniform(5, 0.3),
            TiedLinearRelu(5, 2, device="cpu"),
            device="cpu",
        )
        assert tm.device == torch.device("cpu")

    def test_matching_distribution_device(self):
        tm = ToyModel(
            SparseUniform(5, 0.3, device="cpu"),
            TiedLinearRelu(5, 2),
            device="cpu",
        )
        assert tm.device == torch.device("cpu")

    def test_conflicting_ae_device_raises(self):
        with pytest.raises(ValueError, match="AutoEncoder was explicitly created on"):
            ToyModel(
                SparseUniform(5, 0.3),
                TiedLinearRelu(5, 2, device="cpu"),
                device="cuda",
            )

    def test_conflicting_distribution_device_raises(self):
        with pytest.raises(ValueError, match="Distribution was explicitly created on"):
            ToyModel(
                SparseUniform(5, 0.3, device="cpu"),
                TiedLinearRelu(5, 2),
                device="cuda",
            )

    def test_ae_with_generator_infers_device_conflict(self):
        """Generator on CPU + ToyModel on cuda should conflict."""
        gen = torch.Generator(device="cpu")
        with pytest.raises(ValueError, match="AutoEncoder was explicitly created on"):
            ToyModel(
                SparseUniform(5, 0.3),
                TiedLinearRelu(5, 2, generator=gen),
                device="cuda",
            )

    def test_no_conflict_when_component_has_no_preference(self):
        """Components without device or generator should not conflict."""
        tm = ToyModel(
            SparseUniform(5, 0.3),
            TiedLinearRelu(5, 2),
            device="cpu",
        )
        assert tm.device == torch.device("cpu")
