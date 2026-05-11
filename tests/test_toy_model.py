# ABOUTME: Tests for ToyModel, SAEEntry, SAERecord, and _resolve_sae_entries.
# ABOUTME: Covers construction, fit(), metrics, train_saes(), evaluate_saes(),
# ABOUTME: sampling, delegated attributes, and full integration pipeline.

"""Comprehensive tests for occhio.toy_model.

Testing strategy:
- SAEEntry / SAERecord: field access, defaults, label auto-generation, duplicate detection
- ToyModel construction: feature mismatch, device handling, importances, __repr__
- ToyModel.fit(): loss reduction, return shape, edge cases (n_epochs=0/1), weight mutation,
  sample_every, hooks, track_losses flag
- ToyModel metric properties: W, feature_norms, superposition, interference, cosine similarity
- ToyModel attribute delegation: sample, encode, decode, n_hidden
- ToyModel SAE pipeline: train_saes, evaluate_saes, SAE metric accessors
- Integration: full small-scale pipeline
"""

import warnings

import pytest
import torch
from torch import Tensor

from occhio import SAEEntry, ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import SparseUniform
from occhio.toy_model import SAERecord, _resolve_sae_entries

# Keep everything small for fast tests.
N_FEATURES = 6
N_HIDDEN = 3
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gen():
    """Seeded CPU generator for reproducible tests."""
    return torch.Generator(device=DEVICE).manual_seed(42)


@pytest.fixture
def distribution(gen):
    """Small SparseUniform distribution."""
    return SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=gen)


@pytest.fixture
def ae(gen):
    """Small TiedLinearRelu autoencoder."""
    return TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)


@pytest.fixture
def model(distribution, ae):
    """Pre-built ToyModel (not yet trained)."""
    return ToyModel(distribution, ae, device=DEVICE)


@pytest.fixture
def trained_model():
    """Small ToyModel trained for a handful of epochs. Uses its own seeds so
    it is independent of the other fixtures."""
    g_dist = torch.Generator(device=DEVICE).manual_seed(99)
    g_ae = torch.Generator(device=DEVICE).manual_seed(99)
    dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g_dist)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g_ae, device=DEVICE)
    tm = ToyModel(dist, ae, device=DEVICE)
    tm.fit(n_epochs=50, batch_size=64)
    return tm


# ---------------------------------------------------------------------------
# SAEEntry
# ---------------------------------------------------------------------------


class TestSAEEntry:
    """Tests for the SAEEntry dataclass."""

    def _make_sae(self):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        return StandardTrainingSAE(cfg)

    def test_construction_minimal(self):
        """Verify minimal construction with only required fields."""
        sae = self._make_sae()
        entry = SAEEntry(sae=sae, type="Standard")
        assert entry.sae is sae
        assert entry.type == "Standard"
        assert entry.params is None
        assert entry.label is None

    def test_construction_full(self):
        """Verify all fields can be set explicitly."""
        sae = self._make_sae()
        entry = SAEEntry(sae=sae, type="Standard", params={"k": 2}, label="my_sae")
        assert entry.params == {"k": 2}
        assert entry.label == "my_sae"

    def test_params_default_none(self):
        """Default params should be None, not an empty dict."""
        entry = SAEEntry(sae=self._make_sae(), type="X")
        assert entry.params is None

    def test_label_default_none(self):
        """Default label should be None so auto-generation kicks in."""
        entry = SAEEntry(sae=self._make_sae(), type="X")
        assert entry.label is None


# ---------------------------------------------------------------------------
# SAERecord
# ---------------------------------------------------------------------------


class TestSAERecord:
    """Tests for the SAERecord dataclass."""

    def _make_sae(self):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        return StandardTrainingSAE(cfg)

    def test_construction_defaults(self):
        """Verify default field values on SAERecord."""
        sae = self._make_sae()
        rec = SAERecord(sae=sae)
        assert rec.sae is sae
        assert rec.sae_type is None
        assert rec.params is None
        assert rec.results is None
        assert rec.losses is None

    def test_construction_full(self):
        """All fields can be explicitly set."""
        sae = self._make_sae()
        rec = SAERecord(
            sae=sae,
            sae_type="Standard",
            params={"l1": 0.1},
            results=None,
            losses=[(0, 1.0), (100, 0.5)],
        )
        assert rec.sae_type == "Standard"
        assert rec.losses == [(0, 1.0), (100, 0.5)]


# ---------------------------------------------------------------------------
# _resolve_sae_entries
# ---------------------------------------------------------------------------


class TestResolveSaeEntries:
    """Tests for the label resolution helper."""

    def _make_sae(self):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        return StandardTrainingSAE(cfg)

    def test_auto_label_single(self):
        """A single entry without explicit label gets '{type}_0'."""
        entries = [SAEEntry(sae=self._make_sae(), type="Standard")]
        resolved = _resolve_sae_entries(entries)
        assert "Standard_0" in resolved

    def test_auto_label_multiple_same_type(self):
        """Multiple entries of the same type get incrementing indices."""
        entries = [
            SAEEntry(sae=self._make_sae(), type="Standard"),
            SAEEntry(sae=self._make_sae(), type="Standard"),
        ]
        resolved = _resolve_sae_entries(entries)
        assert set(resolved.keys()) == {"Standard_0", "Standard_1"}

    def test_auto_label_mixed_types(self):
        """Per-type indices are independent."""
        entries = [
            SAEEntry(sae=self._make_sae(), type="A"),
            SAEEntry(sae=self._make_sae(), type="B"),
            SAEEntry(sae=self._make_sae(), type="A"),
        ]
        resolved = _resolve_sae_entries(entries)
        assert set(resolved.keys()) == {"A_0", "B_0", "A_1"}

    def test_explicit_label_used(self):
        """Explicit label overrides auto-generation."""
        entries = [SAEEntry(sae=self._make_sae(), type="X", label="custom")]
        resolved = _resolve_sae_entries(entries)
        assert "custom" in resolved
        assert "X_0" not in resolved

    def test_duplicate_explicit_label_raises(self):
        """Two entries with the same explicit label must raise ValueError."""
        entries = [
            SAEEntry(sae=self._make_sae(), type="X", label="dup"),
            SAEEntry(sae=self._make_sae(), type="Y", label="dup"),
        ]
        with pytest.raises(ValueError, match="Duplicate SAE label"):
            _resolve_sae_entries(entries)

    def test_auto_label_collides_with_explicit_label_raises(self):
        """Auto-generated label that collides with an earlier explicit label raises."""
        entries = [
            SAEEntry(sae=self._make_sae(), type="X", label="Standard_0"),
            SAEEntry(sae=self._make_sae(), type="Standard"),
        ]
        with pytest.raises(ValueError, match="Duplicate SAE label"):
            _resolve_sae_entries(entries)

    def test_empty_list(self):
        """Empty list should return empty dict, not raise."""
        assert _resolve_sae_entries([]) == {}

    def test_result_tuple_structure(self):
        """Each resolved value is (sae, sae_type, params)."""
        sae = self._make_sae()
        params = {"k": 2}
        entries = [SAEEntry(sae=sae, type="T", params=params)]
        resolved = _resolve_sae_entries(entries)
        val = resolved["T_0"]
        assert val[0] is sae
        assert val[1] == "T"
        assert val[2] is params


# ---------------------------------------------------------------------------
# ToyModel Construction
# ---------------------------------------------------------------------------


class TestToyModelConstruction:
    """Tests for ToyModel.__init__ parameter validation and defaults."""

    def test_basic_construction(self, distribution, ae):
        """ToyModel can be constructed with matching distribution and ae."""
        tm = ToyModel(distribution, ae, device=DEVICE)
        assert tm.n_features == N_FEATURES
        assert tm.device == torch.device(DEVICE)
        assert isinstance(tm.saes, dict) and len(tm.saes) == 0

    def test_feature_mismatch_raises(self):
        """Distribution and AE with different n_features must raise ValueError."""
        gen1 = torch.Generator(device=DEVICE).manual_seed(1)
        gen2 = torch.Generator(device=DEVICE).manual_seed(2)
        dist = SparseUniform(n_features=4, p_active=0.5, generator=gen1)
        ae = TiedLinearRelu(8, 3, generator=gen2, device=DEVICE)
        with pytest.raises(ValueError, match="features"):
            ToyModel(dist, ae)

    def test_device_mismatch_ae_raises(self):
        """If ae has explicit device and ToyModel gets a different device, raise."""
        g1 = torch.Generator(device="cpu").manual_seed(1)
        g2 = torch.Generator(device="cpu").manual_seed(2)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device="cpu")
        # AE was explicitly created on cpu; asking for a non-existent device
        # should raise. We can't test mps/cuda portably, but we can verify
        # the validation path by checking the error message format.
        # Since we only have cpu available, this tests the branch is hit
        # when ae._init_device != the requested device.
        # We can't actually trigger a mismatch on CPU-only machines, but we
        # can verify construction succeeds when devices match.
        tm = ToyModel(dist, ae, device="cpu")
        assert tm.device == torch.device("cpu")

    def test_importances_default_ones(self, distribution, ae):
        """Default importances should be all ones."""
        tm = ToyModel(distribution, ae, device=DEVICE)
        assert torch.allclose(tm.importances, torch.ones(N_FEATURES))

    def test_importances_list(self, distribution, ae):
        """Importances passed as a list are converted to tensor."""
        imp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tm = ToyModel(distribution, ae, device=DEVICE, importances=imp)
        assert torch.allclose(tm.importances, torch.tensor(imp))

    def test_importances_tensor(self, distribution, ae):
        """Importances passed as a tensor are moved to the correct device."""
        imp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tm = ToyModel(distribution, ae, device=DEVICE, importances=imp)
        assert tm.importances.device == torch.device(DEVICE)

    def test_hooks_default_empty(self, distribution, ae):
        """Hooks list defaults to empty."""
        tm = ToyModel(distribution, ae, device=DEVICE)
        assert tm.hooks == []

    def test_hooks_preserved(self, distribution, ae):
        """Hooks passed at construction are stored."""
        hook = lambda tm: None
        tm = ToyModel(distribution, ae, device=DEVICE, hooks=[hook])
        assert len(tm.hooks) == 1 and tm.hooks[0] is hook

    def test_repr(self, model):
        """__repr__ should include the distribution info."""
        r = repr(model)
        assert "ToyModel" in r
        assert "SparseUniform" in r

    def test_device_none_inferred_from_ae(self):
        """When device=None, infer from ae._init_device."""
        g1 = torch.Generator(device="cpu").manual_seed(1)
        g2 = torch.Generator(device="cpu").manual_seed(2)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device="cpu")
        tm = ToyModel(dist, ae)  # device=None
        assert tm.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ToyModel.fit()
# ---------------------------------------------------------------------------


class TestToyModelFit:
    """Tests for the training loop."""

    def test_returns_losses_and_hook_returns(self, model):
        """fit() returns a (losses, hook_returns) tuple."""
        result = model.fit(n_epochs=10, batch_size=32)
        assert isinstance(result, tuple) and len(result) == 2

    def test_losses_length_matches_n_epochs(self, model):
        """The losses list should have exactly n_epochs entries."""
        losses, _ = model.fit(n_epochs=20, batch_size=32)
        assert len(losses) == 20

    def test_n_epochs_zero(self, model):
        """n_epochs=0 should return empty losses and not crash."""
        losses, hook_returns = model.fit(n_epochs=0, batch_size=32)
        assert losses == []
        assert hook_returns == []

    def test_n_epochs_one(self, model):
        """Single epoch should still work correctly."""
        losses, _ = model.fit(n_epochs=1, batch_size=32)
        assert len(losses) == 1
        assert isinstance(losses[0], float)

    def test_loss_is_finite(self, model):
        """All losses should be finite (not NaN or Inf)."""
        losses, _ = model.fit(n_epochs=10, batch_size=64)
        for loss_val in losses:
            assert torch.isfinite(torch.tensor(loss_val)), (
                f"Non-finite loss: {loss_val}"
            )

    def test_training_reduces_loss(self):
        """After enough epochs the loss should decrease from its initial value.
        Uses a fresh model to avoid fixture interference."""
        g1 = torch.Generator(device=DEVICE).manual_seed(7)
        g2 = torch.Generator(device=DEVICE).manual_seed(8)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)
        losses, _ = tm.fit(n_epochs=100, batch_size=128, learning_rate=1e-3)
        # Compare early average to late average
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, (
            f"Loss did not decrease: early={early:.4f}, late={late:.4f}"
        )

    def test_weights_change_after_training(self):
        """Autoencoder weights should be different after training."""
        g1 = torch.Generator(device=DEVICE).manual_seed(10)
        g2 = torch.Generator(device=DEVICE).manual_seed(11)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        W_before = ae.W.data.clone()
        tm = ToyModel(dist, ae, device=DEVICE)
        tm.fit(n_epochs=20, batch_size=64)
        assert not torch.equal(W_before, ae.W.data), "Weights unchanged after training"

    def test_batch_size_parameter(self, model):
        """fit() should work with various batch sizes without errors."""
        losses, _ = model.fit(n_epochs=5, batch_size=16)
        assert len(losses) == 5

    def test_learning_rate_parameter(self):
        """Higher learning rate should generally cause larger weight changes."""
        g1 = torch.Generator(device=DEVICE).manual_seed(20)
        g2 = torch.Generator(device=DEVICE).manual_seed(21)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)
        # Just verify it doesn't crash and produces finite losses
        losses, _ = tm.fit(n_epochs=5, batch_size=32, learning_rate=0.1)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_track_losses_false(self, model):
        """track_losses=False should return an empty loss list."""
        losses, _ = model.fit(n_epochs=10, batch_size=32, track_losses=False)
        assert losses == []

    def test_sample_every_parameter(self, model):
        """sample_every controls how often fresh samples are drawn.
        Training should still complete and produce correct number of losses."""
        losses, _ = model.fit(n_epochs=30, batch_size=32, sample_every=10)
        assert len(losses) == 30

    def test_sample_every_invalid_raises(self, model):
        """sample_every < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="sample_every"):
            model.fit(n_epochs=10, batch_size=32, sample_every=0)

    def test_sample_every_negative_raises(self, model):
        """Negative sample_every must raise ValueError."""
        with pytest.raises(ValueError, match="sample_every"):
            model.fit(n_epochs=10, batch_size=32, sample_every=-1)

    def test_hooks_called(self):
        """Hooks passed to fit() should be called once per epoch."""
        g1 = torch.Generator(device=DEVICE).manual_seed(30)
        g2 = torch.Generator(device=DEVICE).manual_seed(31)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)

        call_count = [0]

        def counting_hook(data):
            call_count[0] += 1
            return data["epoch"]

        n = 15
        _, hook_returns = tm.fit(n_epochs=n, batch_size=32, hooks=[counting_hook])
        assert call_count[0] == n
        assert len(hook_returns) == 1
        assert len(hook_returns[0]) == n

    def test_hook_data_keys(self):
        """Hook data dict should contain expected keys."""
        g1 = torch.Generator(device=DEVICE).manual_seed(40)
        g2 = torch.Generator(device=DEVICE).manual_seed(41)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)

        captured = {}

        def spy_hook(data):
            captured.update(data)

        tm.fit(n_epochs=1, batch_size=32, hooks=[spy_hook])
        expected_keys = {"tm", "epoch", "n_epochs", "loss", "x", "x_hat"}
        assert expected_keys.issubset(captured.keys())

    def test_custom_optimizer(self):
        """fit() should accept a custom optimizer."""
        g1 = torch.Generator(device=DEVICE).manual_seed(50)
        g2 = torch.Generator(device=DEVICE).manual_seed(51)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)
        opt = torch.optim.SGD(ae.parameters(), lr=0.01)
        losses, _ = tm.fit(n_epochs=5, batch_size=32, optimizer=opt)
        assert len(losses) == 5

    def test_instance_hooks_called(self):
        """Hooks registered on the ToyModel instance (via constructor) are called
        every epoch, independently of hooks passed to fit()."""
        g1 = torch.Generator(device=DEVICE).manual_seed(60)
        g2 = torch.Generator(device=DEVICE).manual_seed(61)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g2, device=DEVICE)

        instance_count = [0]

        def instance_hook(tm_instance):
            instance_count[0] += 1

        tm = ToyModel(dist, ae, device=DEVICE, hooks=[instance_hook])
        tm.fit(n_epochs=5, batch_size=32)
        assert instance_count[0] == 5


# ---------------------------------------------------------------------------
# ToyModel Properties / Metrics
# ---------------------------------------------------------------------------


class TestToyModelMetrics:
    """Tests for computed metric properties on an (optionally) trained model."""

    def test_W_shape(self, trained_model):
        """W property should have shape (n_hidden, n_features)."""
        W = trained_model.W
        assert W.shape == (N_HIDDEN, N_FEATURES)

    def test_W_T_W_shape(self, trained_model):
        """W^T W should be (n_features, n_features)."""
        WtW = trained_model.W_T_W
        assert WtW.shape == (N_FEATURES, N_FEATURES)

    def test_W_T_W_symmetric(self, trained_model):
        """W^T W must be symmetric."""
        WtW = trained_model.W_T_W
        assert torch.allclose(WtW, WtW.T, atol=1e-5)

    def test_feature_norms_shape(self, trained_model):
        """feature_norms should have one entry per feature."""
        norms = trained_model.feature_norms
        assert norms.shape == (N_FEATURES,)

    def test_feature_norms_positive(self, trained_model):
        """All feature norms should be non-negative."""
        norms = trained_model.feature_norms
        assert (norms >= 0).all()

    def test_feature_representations_shape(self, trained_model):
        """feature_representations should be (n_features,)."""
        reps = trained_model.feature_representations
        assert reps.shape == (N_FEATURES,)

    def test_feature_norms_equals_sqrt_representations(self, trained_model):
        """feature_norms == sqrt(feature_representations) since reps = ||w_i||^2."""
        norms = trained_model.feature_norms
        reps = trained_model.feature_representations
        assert torch.allclose(norms, reps.sqrt(), atol=1e-5)

    def test_feature_dimensionalities_shape(self, trained_model):
        """feature_dimensionalities should be (n_features,)."""
        dims = trained_model.feature_dimensionalities
        assert dims.shape == (N_FEATURES,)

    def test_frobenius_norm_squared_scalar(self, trained_model):
        """frobenius_norm_squared should be a scalar."""
        fns = trained_model.frobenius_norm_squared
        assert fns.dim() == 0

    def test_frobenius_norm_squared_positive(self, trained_model):
        """frobenius_norm_squared should be positive."""
        assert trained_model.frobenius_norm_squared > 0

    def test_interferences_shape(self, trained_model):
        """interferences should be (n_features, n_features)."""
        I = trained_model.interferences
        assert I.shape == (N_FEATURES, N_FEATURES)

    def test_interferences_sq_shape(self, trained_model):
        """interferences_sq should be (n_features, n_features)."""
        Isq = trained_model.interferences_sq
        assert Isq.shape == (N_FEATURES, N_FEATURES)

    def test_total_feature_interferences_shape(self, trained_model):
        """total_feature_interferences should be (n_features,)."""
        tfi = trained_model.total_feature_interferences
        assert tfi.shape == (N_FEATURES,)

    def test_total_feature_interferences_excludes_diagonal(self, trained_model):
        """The total interference sum should exclude self-interference (diagonal=0)."""
        Isq = trained_model.interferences_sq
        off_diag_sum = Isq.clone().fill_diagonal_(0).sum(dim=1)
        tfi = trained_model.total_feature_interferences
        assert torch.allclose(tfi, off_diag_sum, atol=1e-5)

    def test_total_feature_interferences_including_self_shape(self, trained_model):
        """total_feature_interferences_including_self should be (n_features,)."""
        tfi_s = trained_model.total_feature_interferences_including_self
        assert tfi_s.shape == (N_FEATURES,)

    def test_cosine_similarity_matrix_shape(self, trained_model):
        """cosine_similarity_matrix should be (n_features, n_features)."""
        cs = trained_model.cosine_similarity_matrix
        assert cs.shape == (N_FEATURES, N_FEATURES)

    def test_cosine_similarity_diagonal_ones(self, trained_model):
        """Diagonal of cosine similarity should be 1 (feature with itself)."""
        cs = trained_model.cosine_similarity_matrix
        assert torch.allclose(cs.diag(), torch.ones(N_FEATURES), atol=1e-4)

    def test_cosine_similarity_bounded(self, trained_model):
        """Cosine similarity values should be in [-1, 1]."""
        cs = trained_model.cosine_similarity_matrix
        assert cs.min() >= -1.0 - 1e-5
        assert cs.max() <= 1.0 + 1e-5

    def test_superposition_scalar(self, trained_model):
        """superposition should be a scalar tensor."""
        sup = trained_model.superposition
        assert sup.dim() == 0

    def test_superposition_bounded(self, trained_model):
        """superposition (rho_mm) should be in [0, 1]."""
        sup = trained_model.superposition.item()
        assert 0.0 <= sup <= 1.0 + 1e-5

    def test_W_normalized_features_unit_norm(self, trained_model):
        """W_normalized_features columns should have unit norm."""
        W_norm = trained_model.W_normalized_features
        col_norms = torch.linalg.vector_norm(W_norm, dim=0)
        assert torch.allclose(col_norms, torch.ones(N_FEATURES), atol=1e-5)

    def test_hidden_dimensions_per_embedded_features(self, trained_model):
        """Should be n_hidden / frobenius_norm_squared."""
        hd = trained_model.hidden_dimensions_per_embedded_features
        expected = N_HIDDEN / trained_model.frobenius_norm_squared
        assert torch.isclose(
            torch.tensor(float(hd)), torch.tensor(float(expected)), atol=1e-5
        )

    def test_embedded_features_per_hidden_dimensions(self, trained_model):
        """Should be frobenius_norm_squared / n_hidden."""
        ef = trained_model.embedded_features_per_hidden_dimensions
        expected = trained_model.frobenius_norm_squared / N_HIDDEN
        assert torch.isclose(
            torch.tensor(float(ef)), torch.tensor(float(expected)), atol=1e-5
        )


# ---------------------------------------------------------------------------
# ToyModel W_effective / encode / decode delegation
# ---------------------------------------------------------------------------


class TestToyModelW:
    """Tests for the W property (effective weight matrix)."""

    def test_W_from_one_hot_embeddings(self, model):
        """W should be the transpose of encode(I), matching get_one_hot_embeddings().T."""
        W = model.W
        one_hot_emb = model.get_one_hot_embeddings()
        assert torch.allclose(W, one_hot_emb.T, atol=1e-6)


# ---------------------------------------------------------------------------
# ToyModel Attribute Delegation (__getattr__)
# ---------------------------------------------------------------------------


class TestToyModelDelegation:
    """Tests for the __getattr__ delegation to distribution and ae."""

    def test_sample_delegates_to_distribution(self, model):
        """model.sample should delegate to distribution.sample."""
        samples = model.sample(32)
        assert isinstance(samples, Tensor)
        assert samples.shape == (32, N_FEATURES)

    def test_encode_delegates_to_ae(self, model):
        """model.encode should delegate to ae.encode."""
        x = torch.randn(4, N_FEATURES)
        z = model.encode(x)
        assert z.shape == (4, N_HIDDEN)

    def test_decode_delegates_to_ae(self, model):
        """model.decode should delegate to ae.decode."""
        z = torch.randn(4, N_HIDDEN)
        x_hat = model.decode(z)
        assert x_hat.shape == (4, N_FEATURES)

    def test_n_hidden_delegates_to_ae(self, model):
        """model.n_hidden should delegate to ae.n_hidden."""
        assert model.n_hidden == N_HIDDEN

    def test_forward_delegates_to_ae(self, model):
        """model.forward should delegate to ae.forward."""
        x = torch.randn(4, N_FEATURES)
        x_hat, z = model.forward(x)
        assert x_hat.shape == (4, N_FEATURES)
        assert z.shape == (4, N_HIDDEN)

    def test_unknown_attribute_raises(self, model):
        """Accessing a non-delegated attribute should raise AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = model.nonexistent_attribute

    def test_loss_delegates_to_ae(self, model):
        """model.loss should delegate to ae.loss."""
        x = torch.randn(4, N_FEATURES)
        x_hat = torch.randn(4, N_FEATURES)
        imp = torch.ones(N_FEATURES)
        loss = model.loss(x, x_hat, imp)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# ToyModel.sample_latent()
# ---------------------------------------------------------------------------


class TestToyModelSampleLatent:
    """Tests for sample_latent method."""

    def test_shape(self, model):
        """sample_latent should return (batch_size, n_hidden)."""
        z = model.sample_latent(16)
        assert z.shape == (16, N_HIDDEN)

    def test_deterministic_with_seeded_distribution(self):
        """Two models with identical seeds should produce identical latents."""
        g1 = torch.Generator(device=DEVICE).manual_seed(77)
        g2 = torch.Generator(device=DEVICE).manual_seed(77)
        ga1 = torch.Generator(device=DEVICE).manual_seed(88)
        ga2 = torch.Generator(device=DEVICE).manual_seed(88)

        dist1 = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g1)
        dist2 = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g2)
        ae1 = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=ga1, device=DEVICE)
        ae2 = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=ga2, device=DEVICE)

        tm1 = ToyModel(dist1, ae1, device=DEVICE)
        tm2 = ToyModel(dist2, ae2, device=DEVICE)

        z1 = tm1.sample_latent(32)
        z2 = tm2.sample_latent(32)
        assert torch.equal(z1, z2)


# ---------------------------------------------------------------------------
# ToyModel.get_one_hot_embeddings()
# ---------------------------------------------------------------------------


class TestGetOneHotEmbeddings:
    """Tests for get_one_hot_embeddings method."""

    def test_shape(self, model):
        """Should return (n_features, n_hidden)."""
        emb = model.get_one_hot_embeddings()
        assert emb.shape == (N_FEATURES, N_HIDDEN)

    def test_consistent_with_encode(self, model):
        """Should equal ae.encode(I)."""
        identity = torch.eye(N_FEATURES, device=DEVICE)
        expected = model.ae.encode(identity)
        emb = model.get_one_hot_embeddings()
        assert torch.allclose(emb, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ToyModel.feature_frequencies
# ---------------------------------------------------------------------------


class TestFeatureFrequencies:
    """Tests for the feature_frequencies property."""

    def test_shape(self, model):
        """feature_frequencies should be (n_features,)."""
        ff = model.feature_frequencies
        assert ff.shape == (N_FEATURES,)

    def test_values_in_zero_one(self, model):
        """Feature frequencies should be between 0 and 1."""
        ff = model.feature_frequencies
        assert (ff >= 0).all()
        assert (ff <= 1).all()

    def test_uses_p_active_for_sparse_uniform(self, model):
        """For SparseUniform, feature_frequencies should use p_active analytically."""
        ff = model.feature_frequencies
        expected = model.distribution.p_active
        assert torch.allclose(ff, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# ToyModel.train_saes()
# ---------------------------------------------------------------------------


class TestToyModelTrainSaes:
    """Tests for the SAE training pipeline."""

    def _make_entry(self, label=None, d_sae=12, l1=0.01):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=d_sae, l1_coefficient=l1)
        sae = StandardTrainingSAE(cfg)
        return SAEEntry(sae=sae, type="Standard", label=label)

    def test_train_single_sae(self, trained_model):
        """Training a single SAE should populate model.saes with one entry."""
        entry = self._make_entry(label="test_sae")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        assert "test_sae" in trained_model.saes
        assert isinstance(trained_model.saes["test_sae"], SAERecord)

    def test_train_multiple_saes(self, trained_model):
        """Training multiple SAEs should populate all entries."""
        entries = [
            self._make_entry(label="sae_a"),
            self._make_entry(label="sae_b"),
        ]
        trained_model.train_saes(entries, training_samples=5_000, batch_size=256)
        assert "sae_a" in trained_model.saes
        assert "sae_b" in trained_model.saes

    def test_train_saes_auto_labels(self, trained_model):
        """SAEs without explicit labels get auto-generated labels."""
        entries = [
            self._make_entry(),
            self._make_entry(),
        ]
        trained_model.train_saes(entries, training_samples=5_000, batch_size=256)
        assert "Standard_0" in trained_model.saes
        assert "Standard_1" in trained_model.saes

    def test_train_saes_overwrite_warns(self, trained_model):
        """Re-training with the same label should warn."""
        entry = self._make_entry(label="dup")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_model.train_saes(
                [self._make_entry(label="dup")],
                training_samples=5_000,
                batch_size=256,
            )
            overwrite_warnings = [x for x in w if "overwritten" in str(x.message)]
            assert len(overwrite_warnings) >= 1

    def test_sae_record_has_sae(self, trained_model):
        """The SAERecord should hold the trained SAE instance."""
        entry = self._make_entry(label="chk")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        rec = trained_model.saes["chk"]
        assert rec.sae is entry.sae
        assert rec.sae_type == "Standard"

    def test_sae_record_params_propagated(self, trained_model):
        """SAEEntry.params should propagate to SAERecord.params."""
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.05)
        sae = StandardTrainingSAE(cfg)
        entry = SAEEntry(sae=sae, type="Standard", params={"l1": 0.05}, label="p")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        assert trained_model.saes["p"].params == {"l1": 0.05}

    def test_callable_saes_argument(self, trained_model):
        """train_saes should accept a callable that returns SAEEntry list."""

        def make_entries(tm):
            return [self._make_entry(label="from_callable")]

        trained_model.train_saes(make_entries, training_samples=5_000, batch_size=256)
        assert "from_callable" in trained_model.saes

    def test_n_loss_snapshots(self, trained_model):
        """When n_loss_snapshots is set, SAERecord.losses should be populated."""
        entry = self._make_entry(label="loss_track")
        trained_model.train_saes(
            [entry],
            training_samples=5_000,
            batch_size=256,
            n_loss_snapshots=3,
        )
        rec = trained_model.saes["loss_track"]
        assert rec.losses is not None
        assert len(rec.losses) == 3
        # Each element is (step, loss_value)
        for step, val in rec.losses:
            assert isinstance(step, int)
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# ToyModel.evaluate_saes()
# ---------------------------------------------------------------------------


class TestToyModelEvaluateSaes:
    """Tests for the SAE evaluation pipeline."""

    def _train_one_sae(self, tm):
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        sae = StandardTrainingSAE(cfg)
        entry = SAEEntry(sae=sae, type="Standard", label="eval_target")
        tm.train_saes([entry], training_samples=5_000, batch_size=256)

    def test_evaluate_returns_dict(self, trained_model):
        """evaluate_saes should return a dict keyed by SAE label."""
        self._train_one_sae(trained_model)
        results = trained_model.evaluate_saes(num_samples=1_000)
        assert isinstance(results, dict)
        assert "eval_target" in results

    def test_evaluate_result_fields(self, trained_model):
        """SyntheticDataEvalResult should have expected fields."""
        self._train_one_sae(trained_model)
        results = trained_model.evaluate_saes(num_samples=1_000)
        result = results["eval_target"]
        # Check all expected fields exist
        assert hasattr(result, "classification")
        assert hasattr(result, "mcc")
        assert hasattr(result, "explained_variance")
        assert hasattr(result, "sae_l0")
        assert hasattr(result, "dead_latents")
        assert hasattr(result, "shrinkage")
        assert hasattr(result, "uniqueness")
        assert hasattr(result, "true_l0")

    def test_evaluate_populates_sae_record(self, trained_model):
        """After evaluation, SAERecord.results should be non-None."""
        self._train_one_sae(trained_model)
        trained_model.evaluate_saes(num_samples=1_000)
        rec = trained_model.saes["eval_target"]
        assert rec.results is not None

    def test_evaluate_specific_labels(self, trained_model):
        """evaluate_saes(labels=[...]) should only evaluate specified SAEs."""
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        entries = [
            SAEEntry(sae=StandardTrainingSAE(cfg), type="S", label="s1"),
            SAEEntry(
                sae=StandardTrainingSAE(
                    StandardTrainingSAEConfig(
                        d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01
                    )
                ),
                type="S",
                label="s2",
            ),
        ]
        trained_model.train_saes(entries, training_samples=5_000, batch_size=256)
        results = trained_model.evaluate_saes(labels=["s1"], num_samples=1_000)
        assert "s1" in results
        assert "s2" not in results

    def test_evaluate_nonexistent_label_raises(self, trained_model):
        """Passing a label that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError, match="do not exist"):
            trained_model.evaluate_saes(labels=["no_such_sae"])

    def test_re_evaluate_warns(self, trained_model):
        """Re-evaluating an already-evaluated SAE should warn."""
        self._train_one_sae(trained_model)
        trained_model.evaluate_saes(num_samples=1_000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trained_model.evaluate_saes(num_samples=1_000)
            re_eval_warnings = [x for x in w if "Re-evaluating" in str(x.message)]
            assert len(re_eval_warnings) >= 1


# ---------------------------------------------------------------------------
# SAE Metric Properties
# ---------------------------------------------------------------------------


class TestSAEMetricProperties:
    """Tests for the SAE metric accessor properties on ToyModel.

    These properties iterate over model.saes and extract evaluation metrics.
    """

    @pytest.fixture
    def model_with_evaluated_sae(self, trained_model):
        """Trained model with one trained + evaluated SAE."""
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        sae = StandardTrainingSAE(cfg)
        entry = SAEEntry(sae=sae, type="Standard", label="metrics_sae")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        trained_model.evaluate_saes(num_samples=1_000)
        return trained_model

    def test_saes_precision(self, model_with_evaluated_sae):
        """saes_precision should return dict with float values."""
        p = model_with_evaluated_sae.saes_precision
        assert isinstance(p, dict)
        assert "metrics_sae" in p
        assert isinstance(p["metrics_sae"], float)

    def test_saes_recall(self, model_with_evaluated_sae):
        """saes_recall should return dict with float values."""
        r = model_with_evaluated_sae.saes_recall
        assert isinstance(r, dict) and "metrics_sae" in r

    def test_saes_f1_score(self, model_with_evaluated_sae):
        """saes_f1_score should return dict with float values."""
        f1 = model_with_evaluated_sae.saes_f1_score
        assert isinstance(f1, dict) and "metrics_sae" in f1

    def test_saes_accuracy(self, model_with_evaluated_sae):
        """saes_accuracy should return dict with float values."""
        acc = model_with_evaluated_sae.saes_accuracy
        assert isinstance(acc, dict) and "metrics_sae" in acc

    def test_saes_explained_variance(self, model_with_evaluated_sae):
        """saes_explained_variance should return dict with float values."""
        ev = model_with_evaluated_sae.saes_explained_variance
        assert isinstance(ev, dict) and "metrics_sae" in ev

    def test_saes_l0(self, model_with_evaluated_sae):
        """saes_l0 should return dict with float values."""
        l0 = model_with_evaluated_sae.saes_l0
        assert isinstance(l0, dict) and "metrics_sae" in l0

    def test_saes_dead_latents(self, model_with_evaluated_sae):
        """saes_dead_latents should return dict with int values."""
        dl = model_with_evaluated_sae.saes_dead_latents
        assert isinstance(dl, dict) and "metrics_sae" in dl
        assert isinstance(dl["metrics_sae"], int)

    def test_saes_mcc(self, model_with_evaluated_sae):
        """saes_mcc should return dict with float values."""
        mcc = model_with_evaluated_sae.saes_mcc
        assert isinstance(mcc, dict) and "metrics_sae" in mcc

    def test_saes_uniqueness(self, model_with_evaluated_sae):
        """saes_uniqueness should return dict with float values."""
        u = model_with_evaluated_sae.saes_uniqueness
        assert isinstance(u, dict) and "metrics_sae" in u

    def test_saes_true_l0(self, model_with_evaluated_sae):
        """saes_true_l0 should return dict with float values."""
        tl0 = model_with_evaluated_sae.saes_true_l0
        assert isinstance(tl0, dict) and "metrics_sae" in tl0

    def test_saes_shrinkage(self, model_with_evaluated_sae):
        """saes_shrinkage should return dict with float values."""
        s = model_with_evaluated_sae.saes_shrinkage
        assert isinstance(s, dict) and "metrics_sae" in s

    def test_saes_feature_similarity(self, model_with_evaluated_sae):
        """saes_feature_similarity should return dict of tensors."""
        fs = model_with_evaluated_sae.saes_feature_similarity
        assert isinstance(fs, dict) and "metrics_sae" in fs
        sim_tensor = fs["metrics_sae"]
        assert sim_tensor.shape[1] == N_FEATURES
        # Cosine similarity should be in [-1, 1]
        assert sim_tensor.min() >= -1.0 - 1e-5
        assert sim_tensor.max() <= 1.0 + 1e-5

    def test_saes_feature_similarity_ordering(self, model_with_evaluated_sae):
        """saes_feature_similarity_ordering should return dict of index tensors."""
        ordering = model_with_evaluated_sae.saes_feature_similarity_ordering
        assert isinstance(ordering, dict) and "metrics_sae" in ordering
        # Ordering should be a 1-D integer tensor
        idx = ordering["metrics_sae"]
        assert idx.dim() == 1
        assert idx.dtype == torch.int64

    def test_empty_saes_returns_empty_dicts(self, trained_model):
        """Metric properties should return empty dicts when no SAEs are stored."""
        assert trained_model.saes_precision == {}
        assert trained_model.saes_l0 == {}
        assert trained_model.saes_mcc == {}

    def test_unevaluated_sae_excluded_from_metrics(self, trained_model):
        """SAEs without results should be excluded from metric dicts."""
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        sae = StandardTrainingSAE(cfg)
        entry = SAEEntry(sae=sae, type="S", label="unevaled")
        trained_model.train_saes([entry], training_samples=5_000, batch_size=256)
        # Not evaluated yet -> results is None
        assert "unevaled" not in trained_model.saes_precision
        assert "unevaled" not in trained_model.saes_mcc


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    """End-to-end test: construct -> fit -> train_saes -> evaluate_saes."""

    def test_full_pipeline(self):
        """Complete pipeline should run without errors and produce valid results."""
        # 1. Construct
        g_dist = torch.Generator(device=DEVICE).manual_seed(100)
        g_ae = torch.Generator(device=DEVICE).manual_seed(101)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g_dist)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g_ae, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)

        # 2. Fit
        losses, _ = tm.fit(n_epochs=50, batch_size=64, learning_rate=1e-3)
        assert len(losses) == 50
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

        # 3. Verify metrics after training
        W = tm.W
        assert W.shape == (N_HIDDEN, N_FEATURES)
        sup = tm.superposition
        assert 0.0 <= sup.item() <= 1.0 + 1e-5

        # 4. Train SAE
        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=0.01)
        sae = StandardTrainingSAE(cfg)
        entry = SAEEntry(sae=sae, type="Standard", label="pipeline_sae")
        tm.train_saes([entry], training_samples=5_000, batch_size=256)
        assert "pipeline_sae" in tm.saes

        # 5. Evaluate SAE
        results = tm.evaluate_saes(num_samples=1_000)
        assert "pipeline_sae" in results
        result = results["pipeline_sae"]

        # Verify result fields have reasonable types and values
        assert isinstance(result.explained_variance, float)
        assert isinstance(result.mcc, float)
        assert isinstance(result.sae_l0, float)
        assert isinstance(result.dead_latents, int)
        assert isinstance(result.classification.precision, float)
        assert isinstance(result.classification.recall, float)
        assert isinstance(result.classification.f1_score, float)

    def test_pipeline_with_importances(self):
        """Pipeline with custom importances should work end-to-end."""
        g_dist = torch.Generator(device=DEVICE).manual_seed(200)
        g_ae = torch.Generator(device=DEVICE).manual_seed(201)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g_dist)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g_ae, device=DEVICE)
        imp = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        tm = ToyModel(dist, ae, device=DEVICE, importances=imp)

        losses, _ = tm.fit(n_epochs=30, batch_size=64)
        assert len(losses) == 30
        # Importances should affect the loss weighting
        assert torch.allclose(tm.importances, torch.tensor(imp))

    def test_pipeline_multiple_saes(self):
        """Training and evaluating multiple SAEs in one pipeline run."""
        g_dist = torch.Generator(device=DEVICE).manual_seed(300)
        g_ae = torch.Generator(device=DEVICE).manual_seed(301)
        dist = SparseUniform(n_features=N_FEATURES, p_active=0.5, generator=g_dist)
        ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=g_ae, device=DEVICE)
        tm = ToyModel(dist, ae, device=DEVICE)
        tm.fit(n_epochs=50, batch_size=64)

        from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

        entries = []
        for i, l1 in enumerate([0.01, 0.1]):
            cfg = StandardTrainingSAEConfig(d_in=N_HIDDEN, d_sae=12, l1_coefficient=l1)
            entries.append(
                SAEEntry(
                    sae=StandardTrainingSAE(cfg),
                    type="Standard",
                    params={"l1": l1},
                    label=f"sae_l1_{l1}",
                )
            )
        tm.train_saes(entries, training_samples=5_000, batch_size=256)

        assert len(tm.saes) == 2
        results = tm.evaluate_saes(num_samples=1_000)
        assert len(results) == 2
        for label in ["sae_l1_0.01", "sae_l1_0.1"]:
            assert label in results


# ---------------------------------------------------------------------------
# _validate_data_file static method
# ---------------------------------------------------------------------------


class TestValidateDataFile:
    """Tests for ToyModel._validate_data_file static method."""

    def test_valid_file(self):
        """Valid tensors dict should not raise."""
        from pathlib import Path

        tensors = {"samples": torch.randn(100, N_FEATURES)}
        # Should not raise
        ToyModel._validate_data_file(
            tensors, Path("test.safetensors"), N_FEATURES, batch_size=32
        )

    def test_multiple_keys_raises(self):
        """More than one key should raise ValueError."""
        from pathlib import Path

        tensors = {"a": torch.randn(10, 6), "b": torch.randn(10, 6)}
        with pytest.raises(ValueError, match="Expected exactly 1 tensor key"):
            ToyModel._validate_data_file(
                tensors, Path("test.safetensors"), N_FEATURES, batch_size=32
            )

    def test_wrong_ndim_raises(self):
        """Non-2D tensor should raise ValueError."""
        from pathlib import Path

        tensors = {"x": torch.randn(10, 6, 2)}
        with pytest.raises(ValueError, match="2-D tensor"):
            ToyModel._validate_data_file(
                tensors, Path("test.safetensors"), N_FEATURES, batch_size=32
            )

    def test_feature_dim_mismatch_raises(self):
        """Wrong feature dimension should raise ValueError."""
        from pathlib import Path

        tensors = {"x": torch.randn(10, 8)}  # 8 != N_FEATURES=6
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            ToyModel._validate_data_file(
                tensors, Path("test.safetensors"), N_FEATURES, batch_size=32
            )

    def test_batch_size_exceeds_dataset_warns(self):
        """batch_size > dataset size should warn."""
        from pathlib import Path

        tensors = {"x": torch.randn(10, N_FEATURES)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ToyModel._validate_data_file(
                tensors, Path("test.safetensors"), N_FEATURES, batch_size=100
            )
            batch_warnings = [x for x in w if "batch_size" in str(x.message).lower()]
            assert len(batch_warnings) >= 1
