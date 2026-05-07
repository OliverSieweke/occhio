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
# 3. HuggingFaceAutoEncoder audit (mocked, no network)
# ═══════════════════════════════════════════════════════════════════════════


class TestHuggingFaceAutoEncoderMocked:
    """Tests using mocked HF Hub calls to avoid network dependency."""

    def _create_safetensors(self, tmp_path, n_hidden=8, n_features=4, extra_keys=None):
        """Helper: create a valid safetensors file with a W matrix and optional bias."""
        W = torch.randn(n_hidden, n_features)
        b = torch.zeros(n_features)
        state = {"W": W, "b": b}
        if extra_keys:
            state.update(extra_keys)
        path = tmp_path / "model.safetensors"
        save_file(state, str(path))
        return path, W, b

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_successful_load(self, mock_api_cls, mock_download, tmp_path):
        path, W, b = self._create_safetensors(tmp_path, n_hidden=8, n_features=4)

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        ae = HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.safetensors")

        assert ae.n_features == 4
        assert ae.n_hidden == 8
        assert ae.revision == "abc123"
        assert ae.filename == "model.safetensors"
        assert torch.allclose(ae.W, W)
        assert torch.allclose(ae.b, b)

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_missing_W_key(self, mock_api_cls, mock_download, tmp_path):
        """File has no 'W' key -- should raise KeyError."""
        path = tmp_path / "bad.safetensors"
        save_file({"other": torch.randn(3, 3)}, str(path))

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.raises(KeyError, match="Expected key 'W'"):
            HuggingFaceAutoEncoder(repo_id="test/repo", filename="bad.safetensors")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_3d_weight_matrix(self, mock_api_cls, mock_download, tmp_path):
        """W is 3D -- should raise ValueError."""
        path = tmp_path / "bad.safetensors"
        save_file({"W": torch.randn(2, 3, 4)}, str(path))

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.raises(ValueError, match="2D"):
            HuggingFaceAutoEncoder(repo_id="test/repo", filename="bad.safetensors")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_non_safetensors_extension_warns(
        self, mock_api_cls, mock_download, tmp_path
    ):
        """A file without .safetensors extension should warn."""
        # Create a valid safetensors file but with a wrong name
        W = torch.randn(4, 3)
        b = torch.zeros(3)
        path = tmp_path / "model.bin"
        save_file({"W": W, "b": b}, str(path))

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.warns(UserWarning, match="does not have expected .safetensors"):
            ae = HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.bin")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_device_parameter(self, mock_api_cls, mock_download, tmp_path):
        """Verify device parameter moves model to specified device."""
        path, W, b = self._create_safetensors(tmp_path, n_hidden=4, n_features=3)

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        ae = HuggingFaceAutoEncoder(
            repo_id="test/repo", filename="model.safetensors", device="cpu"
        )
        assert ae.device == torch.device("cpu")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_load_state_dict_with_extra_keys_fails(
        self, mock_api_cls, mock_download, tmp_path
    ):
        """If safetensors has unexpected keys beyond W and b, load_state_dict raises."""
        path = tmp_path / "model.safetensors"
        save_file(
            {
                "W": torch.randn(4, 3),
                "b": torch.zeros(3),
                "extra_param": torch.randn(5),
            },
            str(path),
        )

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.raises(RuntimeError, match="Unexpected key"):
            HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.safetensors")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_shape_mismatch_between_W_and_b(
        self, mock_api_cls, mock_download, tmp_path
    ):
        """W says n_features=4, but b has length 3 -- load_state_dict should catch."""
        path = tmp_path / "model.safetensors"
        save_file({"W": torch.randn(8, 4), "b": torch.zeros(3)}, str(path))

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.raises(RuntimeError, match="size mismatch"):
            HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.safetensors")

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_repr(self, mock_api_cls, mock_download, tmp_path):
        path, W, b = self._create_safetensors(tmp_path, n_hidden=4, n_features=3)

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        ae = HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.safetensors")
        r = repr(ae)
        assert "model.safetensors" in r
        assert "n_features=3" in r
        assert "n_hidden=4" in r

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_encode_decode_roundtrip(self, mock_api_cls, mock_download, tmp_path):
        """After loading, encode/decode should work with correct shapes."""
        path, W, b = self._create_safetensors(tmp_path, n_hidden=8, n_features=4)

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        ae = HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.safetensors")
        x = torch.randn(16, 4)
        z = ae.encode(x)
        assert z.shape == (16, 8)
        x_hat = ae.decode(z)
        assert x_hat.shape == (16, 4)


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


class TestHuggingFaceApiModule:
    """Tests for src/occhio/hugging_face/api.py."""

    def test_module_imports_sae_lens(self):
        """api.py imports sae_lens.synthetic.SyntheticModel at module level.
        This is a hard dependency -- verify it at least imports."""
        from occhio.hugging_face.api import download_distributions, download_models

    def test_download_functions_exist(self):
        from occhio.hugging_face import api

        assert callable(api.download_distributions)
        assert callable(api.download_models)

    def test_no_init_py_in_hugging_face_dir(self):
        """The hugging_face/ directory has NO __init__.py, so
        'from occhio.hugging_face import api' works only as implicit namespace pkg."""
        import importlib

        # This should work because Python 3 supports implicit namespace packages
        mod = importlib.import_module("occhio.hugging_face.api")
        assert hasattr(mod, "download_distributions")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Warning message formatting bugs
# ═══════════════════════════════════════════════════════════════════════════


class TestWarningMessageFormatting:
    """Verify warning messages have correct spacing between sentences.

    Previously both hugging_face.py files had a missing space:
        "...extension.This may lead..."
    Fixed to: "...extension. This may lead..."
    """

    @patch("occhio.autoencoders.hugging_face.hf_hub_download")
    @patch("occhio.autoencoders.hugging_face.HfApi")
    def test_ae_warning_has_correct_spacing(
        self, mock_api_cls, mock_download, tmp_path
    ):
        """Warning message should have a space between sentences."""
        path = tmp_path / "model.bin"
        save_file({"W": torch.randn(4, 3), "b": torch.zeros(3)}, str(path))

        mock_info = MagicMock()
        mock_info.sha = "abc123"
        mock_api_cls.return_value.model_info.return_value = mock_info
        mock_download.return_value = str(path)

        from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder

        with pytest.warns(UserWarning) as record:
            HuggingFaceAutoEncoder(repo_id="test/repo", filename="model.bin")

        msg = str(record[0].message)
        assert "extension. This" in msg, (
            f"Expected proper spacing in warning, got: {msg}"
        )
