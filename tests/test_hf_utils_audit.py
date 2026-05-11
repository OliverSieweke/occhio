"""Red-team audit tests for HuggingFace integration and utility modules.

Tests that exercise edge cases, boundary conditions, and failure modes
in the HuggingFace autoencoder/distribution loaders and utility functions.

Note: Tests that require network access (HuggingFace Hub) are skipped
unless the OCCHIO_HF_INTEGRATION environment variable is set.
"""

import os
import pickle
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ── Utility imports (always available, no network needed) ──────────────────
from occhio.utils.device import _same_device
from occhio.utils.logging import suppress_tqdm


# ═══════════════════════════════════════════════════════════════════════════
# 1. _same_device audit
# ═══════════════════════════════════════════════════════════════════════════


class TestSameDevice:
    """Verify _same_device covers all realistic device comparisons."""

    def test_cpu_cpu(self):
        assert _same_device(torch.device("cpu"), torch.device("cpu"))

    def test_cpu_with_and_without_index(self):
        """cpu has no index by default; cpu:0 should still match."""
        # torch.device("cpu").index is None
        assert _same_device(torch.device("cpu"), torch.device("cpu"))

    def test_mps_mps(self):
        assert _same_device(torch.device("mps"), torch.device("mps"))

    def test_mps_with_index(self):
        """mps and mps:0 should be treated as the same device."""
        assert _same_device(torch.device("mps"), torch.device("mps:0"))
        assert _same_device(torch.device("mps:0"), torch.device("mps"))

    def test_cross_device_cpu_mps(self):
        assert not _same_device(torch.device("cpu"), torch.device("mps"))

    def test_cross_device_mps_cpu(self):
        assert not _same_device(torch.device("mps"), torch.device("cpu"))

    def test_same_cuda_index(self):
        """cuda:0 == cuda:0 -- even if CUDA not available, the comparison should work."""
        assert _same_device(torch.device("cuda:0"), torch.device("cuda:0"))

    def test_different_cuda_index(self):
        assert not _same_device(torch.device("cuda:0"), torch.device("cuda:1"))

    def test_cuda_with_and_without_index(self):
        """cuda (no index => None) vs cuda:0 -- should be same."""
        assert _same_device(torch.device("cuda"), torch.device("cuda:0"))

    def test_none_index_both_sides(self):
        """Both devices have index=None -- the (a.index or 0) handles this."""
        d1 = torch.device("cpu")
        d2 = torch.device("cpu")
        assert d1.index is None
        assert d2.index is None
        assert _same_device(d1, d2)

    # ── Edge case: the function does NOT handle None inputs ──
    def test_none_input_raises(self):
        """_same_device expects torch.device, not None. Verify it fails clearly."""
        with pytest.raises(AttributeError):
            _same_device(None, torch.device("cpu"))

        with pytest.raises(AttributeError):
            _same_device(torch.device("cpu"), None)

    def test_string_input_raises(self):
        """_same_device expects torch.device objects, not raw strings."""
        with pytest.raises(AttributeError):
            _same_device("cpu", torch.device("cpu"))


# ═══════════════════════════════════════════════════════════════════════════
# 2. suppress_tqdm audit
# ═══════════════════════════════════════════════════════════════════════════


class TestSuppressTqdm:
    """Verify tqdm suppression context manager."""

    def test_basic_suppression(self):
        import tqdm.auto as tqdm_auto

        with suppress_tqdm():
            bar = tqdm_auto.tqdm(range(10))
            assert bar.disable is True

    def test_restoration_after_context(self):
        import tqdm.auto as tqdm_auto

        original = tqdm_auto.tqdm.__init__
        with suppress_tqdm():
            pass
        assert tqdm_auto.tqdm.__init__ is original

    def test_restoration_on_exception(self):
        """If code inside the context raises, tqdm should still be restored."""
        import tqdm.auto as tqdm_auto

        original = tqdm_auto.tqdm.__init__
        with pytest.raises(ValueError):
            with suppress_tqdm():
                raise ValueError("boom")
        assert tqdm_auto.tqdm.__init__ is original

    def test_nested_suppression(self):
        """Nested suppress_tqdm should not corrupt the original __init__."""
        import tqdm.auto as tqdm_auto

        original = tqdm_auto.tqdm.__init__
        with suppress_tqdm():
            with suppress_tqdm():
                bar = tqdm_auto.tqdm(range(5))
                assert bar.disable is True
            # After inner exits, the "original" it saved was the patched one
            # from the outer context -- so it stays patched (which is correct
            # since we're still inside the outer suppress_tqdm)
            bar2 = tqdm_auto.tqdm(range(5))
            # BUG DETECTION: After inner context exits, it restores its
            # saved "original_init" -- which was the PATCHED init from
            # the outer context. So suppression still works here.
            assert bar2.disable is True
        # After outer exits, the true original should be restored
        assert tqdm_auto.tqdm.__init__ is original


# ═══════════════════════════════════════════════════════════════════════════
# 3. AutoEncoderBase.from_local / from_hub / push_to_hub
# ═══════════════════════════════════════════════════════════════════════════


from occhio.autoencoders import (
    AutoEncoderBase,
    TiedLinearRelu,
    TiedLinear,
    MLPEncoder,
    TiedMLPEncoder,
    ComputeAutoEncoder,
    AttnLinearAE,
    AttnAttnAE,
    LinearAttnAE,
    SynthAE,
)


class TestFromLocalRoundTrip:
    """Tests for AutoEncoderBase.from_local() architecture-invariant loading."""

    @pytest.mark.parametrize(
        "make_ae",
        [
            pytest.param(lambda: TiedLinearRelu(10, 5), id="TiedLinearRelu"),
            pytest.param(lambda: TiedLinear(10, 5), id="TiedLinear"),
            pytest.param(
                lambda: MLPEncoder(embedding=[10, 8, 5], unembedding=[5, 8, 10]),
                id="MLPEncoder",
            ),
            pytest.param(lambda: TiedMLPEncoder(dims=[10, 8, 5]), id="TiedMLPEncoder"),
            pytest.param(
                lambda: ComputeAutoEncoder(N=10, k=5), id="ComputeAutoEncoder"
            ),
            pytest.param(
                lambda: AttnLinearAE(10, 4, n_heads=2, dict_size=8), id="AttnLinearAE"
            ),
            pytest.param(
                lambda: AttnAttnAE(10, 4, n_heads=2, dict_size=8), id="AttnAttnAE"
            ),
            pytest.param(
                lambda: LinearAttnAE(10, 4, n_heads=2, dict_size=8), id="LinearAttnAE"
            ),
            pytest.param(lambda: SynthAE(10, 5), id="SynthAE"),
        ],
    )
    def test_round_trip_all_architectures(self, make_ae, tmp_path):
        """Save → from_local round-trip preserves architecture and outputs."""
        ae = make_ae()
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)

        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded).__name__ == type(ae).__name__
        assert loaded.n_features == ae.n_features
        assert loaded.n_hidden == ae.n_hidden

        x = torch.randn(4, ae.n_features)
        assert torch.allclose(ae.encode(x), loaded.encode(x))

    def test_device_override(self, tmp_path):
        ae = TiedLinearRelu(8, 4)
        path = tmp_path / "model.safetensors"
        ae.save_weights(path)
        loaded = AutoEncoderBase.from_local(path, device="cpu")
        assert loaded.device == torch.device("cpu")

    def test_missing_class_metadata(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"W": torch.randn(4, 3)}, str(path))
        with pytest.raises(ValueError, match="no 'class' metadata"):
            AutoEncoderBase.from_local(path)

    def test_unknown_class(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"W": torch.randn(4, 3)}, str(path), metadata={"class": "FakeAE"})
        with pytest.raises(ValueError, match="Unknown autoencoder class"):
            AutoEncoderBase.from_local(path)

    def test_legacy_file_without_config(self, tmp_path):
        """Legacy files with class + W key but no config should still load."""
        path = tmp_path / "legacy.safetensors"
        ae = TiedLinearRelu(8, 4)
        # Save manually without config (simulating old format)
        save_file(ae.state_dict(), str(path), metadata={"class": "TiedLinearRelu"})
        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, TiedLinearRelu)
        assert loaded.n_features == 8
        assert loaded.n_hidden == 4

    def test_class_alias(self, tmp_path):
        """Old class names should resolve via _class_aliases."""
        ae = TiedLinearRelu(8, 4)
        path = tmp_path / "old.safetensors"
        save_file(
            ae.state_dict(),
            str(path),
            metadata={"class": "HuggingFaceAutoEncoder"},
        )
        loaded = AutoEncoderBase.from_local(path)
        assert isinstance(loaded, TiedLinearRelu)

    def test_config_in_metadata(self, tmp_path):
        """Config is stored in safetensors metadata after save_weights."""
        ae = AttnLinearAE(10, 4, n_heads=2, dict_size=8)
        path = ae.save_weights(tmp_path / "model")
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        assert "config" in meta
        import json

        config = json.loads(meta["config"])
        assert config["n_heads"] == 2
        assert config["dict_size"] == 8

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AutoEncoderBase.from_local("/nonexistent/path.safetensors")


# ═══════════════════════════════════════════════════════════════════════════
# 4. HuggingFaceDistribution audit (mocked, no network)
# ═══════════════════════════════════════════════════════════════════════════


class TestHuggingFaceDistributionMocked:
    """Tests using mocked HF Hub calls."""

    def _create_samples_file(
        self, tmp_path, n_samples=100, n_features=8, key="samples"
    ):
        path = tmp_path / "samples.safetensors"
        samples = torch.randn(n_samples, n_features)
        save_file({key: samples}, str(path))
        return path, samples

    def _make_dist(self, tmp_path, n_samples=100, n_features=8, **kwargs):
        """Helper that patches HF calls and returns a HuggingFaceDistribution."""
        path, samples = self._create_samples_file(tmp_path, n_samples, n_features)

        with (
            patch("occhio.distributions.hugging_face.HfApi") as mock_api_cls,
            patch("occhio.distributions.hugging_face.hf_hub_download") as mock_download,
        ):
            mock_info = MagicMock()
            mock_info.sha = "abc123"
            mock_api_cls.return_value.repo_info.return_value = mock_info
            mock_download.return_value = str(path)

            from occhio.distributions.hugging_face import HuggingFaceDistribution

            dist = HuggingFaceDistribution(
                repo_id="test/repo",
                filename="samples.safetensors",
                **kwargs,
            )
        return dist, samples

    def test_basic_sample(self, tmp_path):
        dist, _ = self._make_dist(tmp_path, n_samples=100, n_features=8)
        batch = dist.sample(16)
        assert batch.shape == (16, 8)

    def test_sample_device_cpu(self, tmp_path):
        dist, _ = self._make_dist(tmp_path, n_samples=50, n_features=4, device="cpu")
        batch = dist.sample(10)
        assert batch.device == torch.device("cpu")

    def test_missing_data_key(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"wrong_key": torch.randn(10, 3)}, str(path))

        with (
            patch("occhio.distributions.hugging_face.HfApi") as mock_api_cls,
            patch("occhio.distributions.hugging_face.hf_hub_download") as mock_download,
        ):
            mock_info = MagicMock()
            mock_info.sha = "abc123"
            mock_api_cls.return_value.repo_info.return_value = mock_info
            mock_download.return_value = str(path)

            from occhio.distributions.hugging_face import HuggingFaceDistribution

            with pytest.raises(KeyError, match="Expected key 'samples'"):
                HuggingFaceDistribution(repo_id="test/repo", filename="bad.safetensors")

    def test_1d_samples_rejected(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"samples": torch.randn(100)}, str(path))

        with (
            patch("occhio.distributions.hugging_face.HfApi") as mock_api_cls,
            patch("occhio.distributions.hugging_face.hf_hub_download") as mock_download,
        ):
            mock_info = MagicMock()
            mock_info.sha = "abc123"
            mock_api_cls.return_value.repo_info.return_value = mock_info
            mock_download.return_value = str(path)

            from occhio.distributions.hugging_face import HuggingFaceDistribution

            with pytest.raises(ValueError, match="2D"):
                HuggingFaceDistribution(repo_id="test/repo", filename="bad.safetensors")

    def test_custom_data_key(self, tmp_path):
        path = tmp_path / "custom.safetensors"
        save_file({"activations": torch.randn(50, 6)}, str(path))

        with (
            patch("occhio.distributions.hugging_face.HfApi") as mock_api_cls,
            patch("occhio.distributions.hugging_face.hf_hub_download") as mock_download,
        ):
            mock_info = MagicMock()
            mock_info.sha = "abc123"
            mock_api_cls.return_value.repo_info.return_value = mock_info
            mock_download.return_value = str(path)

            from occhio.distributions.hugging_face import HuggingFaceDistribution

            dist = HuggingFaceDistribution(
                repo_id="test/repo",
                filename="custom.safetensors",
                data_key="activations",
            )
            assert dist.n_features == 6
            batch = dist.sample(10)
            assert batch.shape == (10, 6)

    def test_buffered_sampling(self, tmp_path):
        """CPU buffered sampling: buffer_size > batch_size."""
        dist, _ = self._make_dist(
            tmp_path, n_samples=200, n_features=4, device="cpu", buffer_size=50
        )
        batch = dist.sample(10)
        assert batch.shape == (10, 4)

    def test_batch_exceeds_buffer_raises(self, tmp_path):
        dist, _ = self._make_dist(
            tmp_path, n_samples=200, n_features=4, device="cpu", buffer_size=5
        )
        with pytest.raises(ValueError, match="exceeds buffer_size"):
            dist.sample(10)

    def test_buffer_refill_on_exhaustion(self, tmp_path):
        """Drawing more samples than buffer holds triggers a refill."""
        dist, _ = self._make_dist(
            tmp_path, n_samples=200, n_features=4, device="cpu", buffer_size=20
        )
        # Draw 15, leaving 5 in buffer
        b1 = dist.sample(15)
        assert b1.shape == (15, 4)
        assert dist._buffer_ptr == 15

        # Draw 10, but only 5 remain => triggers refill
        b2 = dist.sample(10)
        assert b2.shape == (10, 4)
        assert dist._buffer_ptr == 10  # just drew 10 from fresh buffer

    def test_clear_buffer(self, tmp_path):
        dist, _ = self._make_dist(
            tmp_path, n_samples=50, n_features=4, device="cpu", buffer_size=20
        )
        dist.sample(10)
        assert dist._buffer is not None
        dist.clear_buffer()
        assert dist._buffer is None
        assert dist._buffer_ptr == 0

    def test_to_device_clears_buffer(self, tmp_path):
        dist, _ = self._make_dist(
            tmp_path, n_samples=50, n_features=4, device="cpu", buffer_size=20
        )
        dist.sample(5)
        assert dist._buffer is not None
        dist.to("cpu")  # even same device should clear buffer
        assert dist._buffer is None

    def test_repr(self, tmp_path):
        dist, _ = self._make_dist(tmp_path, n_samples=100, n_features=8)
        r = repr(dist)
        assert "n_features=8" in r
        assert "n_samples=100" in r

    def test_sample_single_row(self, tmp_path):
        """batch_size=1 edge case."""
        dist, _ = self._make_dist(tmp_path, n_samples=10, n_features=3)
        batch = dist.sample(1)
        assert batch.shape == (1, 3)

    def test_sample_all_rows(self, tmp_path):
        """batch_size == n_samples."""
        dist, samples = self._make_dist(tmp_path, n_samples=10, n_features=3)
        batch = dist.sample(10)
        assert batch.shape == (10, 3)

    def test_sample_more_than_n_samples(self, tmp_path):
        """batch_size > n_samples -- sampling with replacement, so this is valid."""
        dist, _ = self._make_dist(tmp_path, n_samples=5, n_features=3)
        batch = dist.sample(20)
        assert batch.shape == (20, 3)

    def test_zero_batch_size(self, tmp_path):
        """batch_size=0 -- should return empty tensor."""
        dist, _ = self._make_dist(tmp_path, n_samples=10, n_features=3)
        batch = dist.sample(0)
        assert batch.shape == (0, 3)

    def test_getstate_excludes_samples_and_buffer(self, tmp_path):
        dist, _ = self._make_dist(tmp_path, n_samples=10, n_features=3)
        state = dist.__getstate__()
        assert "_samples" not in state
        assert "_buffer" not in state
        assert "_buffer_ptr" not in state
        # But it should still have the metadata needed to re-download
        assert "repo_id" in state
        assert "filename" in state
        assert "revision" in state
        assert "data_key" in state

    def test_non_safetensors_extension_warns(self, tmp_path):
        """File without .safetensors extension should warn."""
        path = tmp_path / "samples.bin"
        save_file({"samples": torch.randn(10, 3)}, str(path))

        with (
            patch("occhio.distributions.hugging_face.HfApi") as mock_api_cls,
            patch("occhio.distributions.hugging_face.hf_hub_download") as mock_download,
        ):
            mock_info = MagicMock()
            mock_info.sha = "abc123"
            mock_api_cls.return_value.repo_info.return_value = mock_info
            mock_download.return_value = str(path)

            from occhio.distributions.hugging_face import HuggingFaceDistribution

            with pytest.warns(UserWarning, match="does not have expected .safetensors"):
                HuggingFaceDistribution(repo_id="test/repo", filename="samples.bin")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HuggingFace api.py audit
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 6. Warning message formatting bugs
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistryAndConfig:
    """Tests for the autoencoder registry and config protocol."""

    def test_registry_contains_all_builtin_classes(self):
        from occhio.autoencoders import AutoEncoderBase

        expected = {
            "TiedLinear",
            "TiedLinearRelu",
            "MLPEncoder",
            "TiedMLPEncoder",
            "ComputeAutoEncoder",
            "AttnLinearAE",
            "AttnAttnAE",
            "LinearAttnAE",
            "SynthAE",
        }
        assert expected.issubset(set(AutoEncoderBase._registry.keys()))

    def test_custom_subclass_auto_registers(self):
        from occhio.autoencoders import AutoEncoderBase

        class _TestCustomAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, my_param=42, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.my_param = my_param
                self.W = torch.nn.Parameter(torch.randn(n_hidden, n_features))

            def encode(self, x):
                return x @ self.W.T

            def decode(self, z):
                return z @ self.W

            def resample_weights(self):
                pass

        assert "_TestCustomAE" in AutoEncoderBase._registry

    def test_custom_subclass_round_trip(self, tmp_path):
        from occhio.autoencoders import AutoEncoderBase

        class _TestRoundTripAE(AutoEncoderBase):
            def __init__(self, n_features, n_hidden, scale=3.14, **kwargs):
                super().__init__(n_features, n_hidden, **kwargs)
                self.scale = scale
                self.W = torch.nn.Parameter(torch.randn(n_hidden, n_features))

            def encode(self, x):
                return x @ self.W.T * self.scale

            def decode(self, z):
                return z @ self.W

            def resample_weights(self):
                pass

        ae = _TestRoundTripAE(8, 4, scale=2.5)
        path = ae.save_weights(tmp_path / "custom")
        loaded = AutoEncoderBase.from_local(path)
        assert type(loaded).__name__ == "_TestRoundTripAE"
        assert loaded.scale == 2.5
        x = torch.randn(2, 8)
        assert torch.allclose(ae.encode(x), loaded.encode(x))
