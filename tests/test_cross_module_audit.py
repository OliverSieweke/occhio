"""Cross-module audit tests for the occhio package.

Tests export completeness, import chain health, device handling consistency,
serialization round-trips, API consistency, and dead code / stale references.
"""

import inspect
import torch
import pytest
from pathlib import Path
from unittest.mock import patch


# ============================================================================
# 1. __init__.py export completeness
# ============================================================================


class TestExportCompleteness:
    """Verify that all public classes are exported from their __init__.py."""

    def test_top_level_exports(self):
        """Top-level occhio exports AutoEncoderBase, AutoencoderType, SAEEntry,
        ToyModel, ModelGrid."""
        from occhio import (
            AutoEncoderBase,
            AutoencoderType,
            ModelGrid,
            SAEEntry,
            ToyModel,
        )

        # Verify they are real classes/enums
        assert inspect.isclass(AutoEncoderBase)
        assert inspect.isclass(ToyModel)
        assert inspect.isclass(ModelGrid)

    def test_autoencoders_export_all_classes(self):
        """Every concrete autoencoder class and the enum should be exported."""
        from occhio.autoencoders import (
            AttnAttnAE,
            AttnLinearAE,
            AutoEncoderBase,
            AutoencoderType,
            ComputeAutoEncoder,
            LinearAttnAE,
            MLPEncoder,
            SynthAE,
            TiedLinear,
            TiedLinearRelu,
            TiedMLPEncoder,
        )

        assert all(
            inspect.isclass(c)
            for c in [
                AutoEncoderBase,
                TiedLinear,
                TiedLinearRelu,
                MLPEncoder,
                TiedMLPEncoder,
                ComputeAutoEncoder,
                AttnLinearAE,
                AttnAttnAE,
                LinearAttnAE,
                SynthAE,
            ]
        )

    def test_distributions_export_all_classes(self):
        """Every concrete distribution class should be exported."""
        from occhio.distributions import (
            AnticorrelatedPairs,
            CorrelatedPairs,
            DAGBayesianPropagation,
            DAGDistribution,
            DAGRandomWalkToRoot,
            Distribution,
            DistributionStack,
            HierarchicalPairs,
            HierarchicalSparse,
            HypercubeDistribution,
            MultiRelational,
            PreferentialAttachment,
            RelationalSimple,
            ScaledHierarchicalPairs,
            SimplexDistribution,
            SimplicialComplexDistribution,
            SingleUniform,
            SparseExponential,
            SparseUniform,
            SphericalDistribution,
            SyntheticDataConfig,
            SyntheticDataModel,
            ToricDistribution,
            HierarchyNode,
        )

    def test_gaussian_correlated_exported(self):
        """GaussianCorrelated is defined in correlated.py and should be exported."""
        from occhio.distributions import GaussianCorrelated

        assert inspect.isclass(GaussianCorrelated)

    def test_hugging_face_distribution_exported(self):
        """HuggingFaceDistribution is defined in hugging_face.py and should be exported."""
        from occhio.distributions import HuggingFaceDistribution

        assert inspect.isclass(HuggingFaceDistribution)

    def test_autoencodertype_has_all_classes(self):
        """AutoencoderType enum should have one entry per concrete autoencoder."""
        from occhio.autoencoders import AutoencoderType

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
        actual = {e.name for e in AutoencoderType}
        assert actual == expected


# ============================================================================
# 2. Import chain health
# ============================================================================


class TestImportChainHealth:
    """Verify that imports work without errors or circular dependencies."""

    def test_import_occhio(self):
        import occhio

        assert hasattr(occhio, "ToyModel")

    def test_from_occhio_import_star(self):
        """from occhio import * should work."""
        exec("from occhio import *")

    def test_from_distributions_import_star(self):
        exec("from occhio.distributions import *")

    def test_from_autoencoders_import_star(self):
        exec("from occhio.autoencoders import *")

    def test_sae_lens_adapter_importable(self):
        from occhio.sae_lens_adapter.activation_generator import (
            ActivationGeneratorWrapper,
        )
        from occhio.sae_lens_adapter.feature_dictionary import (
            FeatureDictionaryWrapper,
        )

        assert inspect.isclass(ActivationGeneratorWrapper)
        assert inspect.isclass(FeatureDictionaryWrapper)

    def test_visualization_importable(self):
        from occhio.visualization import plot_embedding

        assert callable(plot_embedding)

    def test_visualization_classes_importable(self):
        from occhio.visualization import RepresentationPlot

        assert inspect.isclass(RepresentationPlot)


# ============================================================================
# 3. Device handling consistency
# ============================================================================


class TestDeviceHandling:
    """Device propagation and tensor creation tests."""

    def test_compute_ae_resample_uses_device(self):
        """ComputeAutoEncoder.resample_weights should create tensors on
        self.device (not hardcoded CPU)."""
        from occhio.autoencoders.compute import ComputeAutoEncoder

        ae = ComputeAutoEncoder(5, 3, device="cpu")
        ae.resample_weights()
        # After resample, W/Z/b should be on ae.device
        assert ae.W.device == ae.device
        assert ae.Z.device == ae.device
        assert ae.b.device == ae.device

    def test_multi_relational_sample_device(self):
        """MultiRelational.sample should create the result tensor on self.device."""
        from occhio.distributions.relational import MultiRelational

        d = MultiRelational(4, 0.1, k=2, device="cpu")
        result = d.sample(10)
        assert result.device == d.device

    def test_attn_linear_ae_alpha_on_device(self):
        """AttnLinearAE.alpha should be created on the correct device."""
        from occhio.autoencoders.attention import AttnLinearAE

        ae = AttnLinearAE(4, 4, n_heads=2, dict_size=3, device="cpu")
        assert ae.alpha.device.type == "cpu"

    def test_attn_attn_ae_alpha_on_device(self):
        """AttnAttnAE.alpha should be created on the correct device."""
        from occhio.autoencoders.attention import AttnAttnAE

        ae = AttnAttnAE(4, 4, n_heads=2, dict_size=3, device="cpu")
        assert ae.alpha.device.type == "cpu"

    def test_toymodel_device_propagation(self):
        """ToyModel should propagate device to ae and importances."""
        from occhio import ToyModel
        from occhio.autoencoders import TiedLinearRelu
        from occhio.distributions import SparseUniform

        dist = SparseUniform(5, 0.5)
        ae = TiedLinearRelu(5, 3)
        model = ToyModel(dist, ae, device="cpu")

        assert model.device == torch.device("cpu")
        assert model.ae.device == torch.device("cpu")
        assert model.importances.device == torch.device("cpu")

    def test_distribution_device_propagated_when_no_explicit_device(self):
        """When distribution has no explicit device, ToyModel moves it to ae device."""
        from occhio import ToyModel
        from occhio.autoencoders import TiedLinearRelu
        from occhio.distributions import SparseUniform

        dist = SparseUniform(5, 0.5)  # No explicit device
        ae = TiedLinearRelu(5, 3, device="cpu")
        model = ToyModel(dist, ae, device="cpu")

        assert dist.device == torch.device("cpu")

    def test_distribution_to_moves_tensors(self):
        """Distribution.to() should move all tensor attrs to the new device."""
        from occhio.distributions import SparseUniform

        d = SparseUniform(5, 0.5, device="cpu")
        d.to("cpu")  # should not error
        assert d.p_active.device == torch.device("cpu")


# ============================================================================
# 4. Serialization audit
# ============================================================================


class TestSerialization:
    """Test save/load round-trips for autoencoders and model grids."""

    def test_autoencoder_save_load_roundtrip(self, tmp_path):
        """AutoEncoderBase save_weights/load_weights should roundtrip perfectly."""
        from occhio.autoencoders import TiedLinearRelu

        ae1 = TiedLinearRelu(5, 3, device="cpu")
        path = ae1.save_weights(tmp_path / "test_ae.safetensors")

        ae2 = TiedLinearRelu(5, 3, device="cpu")
        ae2.load_weights(path)

        for (k1, v1), (k2, v2) in zip(
            ae1.state_dict().items(), ae2.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)

    def test_autoencoder_save_creates_json(self, tmp_path):
        """save_weights should create a companion .json metadata file."""
        from occhio.autoencoders import TiedLinearRelu

        ae = TiedLinearRelu(5, 3, device="cpu")
        path = ae.save_weights(tmp_path / "test_ae")

        json_path = path.with_suffix(".json")
        assert json_path.exists()

    def test_autoencoder_load_wrong_class_raises(self, tmp_path):
        """Loading weights saved from a different class should raise TypeError."""
        from occhio.autoencoders import TiedLinear, TiedLinearRelu

        ae1 = TiedLinearRelu(5, 3, device="cpu")
        path = ae1.save_weights(tmp_path / "test_ae")

        ae2 = TiedLinear(5, 3, device="cpu")
        with pytest.raises(TypeError, match="TiedLinearRelu"):
            ae2.load_weights(path)

    def test_autoencoder_load_nonexistent_raises(self):
        """Loading from a path that doesn't exist should raise FileNotFoundError."""
        from occhio.autoencoders import TiedLinearRelu

        ae = TiedLinearRelu(5, 3, device="cpu")
        with pytest.raises(FileNotFoundError):
            ae.load_weights("/nonexistent/path.safetensors")

    def test_model_grid_save_load_roundtrip(self, tmp_path):
        """ModelGrid.save / ModelGrid.load should roundtrip via dill."""
        from occhio import ModelGrid, ToyModel
        from occhio.autoencoders import TiedLinearRelu
        from occhio.distributions import SparseUniform
        from occhio.model_grid import Axis

        def create_model(params):
            return ToyModel(
                distribution=SparseUniform(3, params["p"]),
                ae=TiedLinearRelu(3, 2),
            )

        grid = ModelGrid(create_model, axes=[Axis("p", [0.1, 0.5])])
        save_path = tmp_path / "test_grid.pkl"
        grid.save(save_path)

        loaded = ModelGrid.load(save_path)
        assert loaded.shape == grid.shape
        assert len(loaded.axes) == len(grid.axes)

    def test_dill_is_required_dependency(self):
        """dill must be importable (listed in pyproject.toml dependencies)."""
        import dill

        assert hasattr(dill, "dump")


# ============================================================================
# 5. API consistency
# ============================================================================


class TestAPIConsistency:
    """Verify consistent interfaces across distribution and autoencoder classes."""

    @pytest.mark.parametrize(
        "cls_name",
        [
            "TiedLinear",
            "TiedLinearRelu",
            "SynthAE",
        ],
    )
    def test_simple_autoencoders_accept_standard_args(self, cls_name):
        """Simple autoencoders should accept (n_features, n_hidden, device, generator)."""
        import occhio.autoencoders as ae_mod

        cls = getattr(ae_mod, cls_name)
        gen = torch.Generator(device="cpu").manual_seed(42)
        ae = cls(5, 3, device="cpu", generator=gen)
        assert ae.n_features == 5
        assert ae.n_hidden == 3

    @pytest.mark.parametrize(
        "cls_name",
        [
            "SparseUniform",
            "SparseExponential",
            "SingleUniform",
        ],
    )
    def test_simple_distributions_accept_standard_args(self, cls_name):
        """Simple distributions should accept (n_features, device, generator)."""
        import occhio.distributions as dist_mod

        cls = getattr(dist_mod, cls_name)
        gen = torch.Generator(device="cpu").manual_seed(42)
        if cls_name == "SingleUniform":
            d = cls(5, device="cpu", generator=gen)
        else:
            d = cls(5, 0.5, device="cpu", generator=gen)
        assert d.n_features == 5

    def test_all_autoencoders_have_encode_decode_resample(self):
        """Every concrete AutoEncoderBase subclass must implement encode, decode,
        resample_weights."""
        from occhio.autoencoders import (
            AttnAttnAE,
            AttnLinearAE,
            ComputeAutoEncoder,
            LinearAttnAE,
            MLPEncoder,
            SynthAE,
            TiedLinear,
            TiedLinearRelu,
            TiedMLPEncoder,
        )

        for cls in [
            TiedLinear,
            TiedLinearRelu,
            SynthAE,
            ComputeAutoEncoder,
            AttnLinearAE,
            AttnAttnAE,
            LinearAttnAE,
        ]:
            for method in ["encode", "decode", "resample_weights"]:
                assert hasattr(cls, method), f"{cls.__name__} missing {method}"

    def test_all_distributions_have_sample(self):
        """Every concrete Distribution subclass must implement sample."""
        from occhio.distributions import (
            AnticorrelatedPairs,
            CorrelatedPairs,
            DAGBayesianPropagation,
            DAGDistribution,
            DAGRandomWalkToRoot,
            HierarchicalPairs,
            HierarchicalSparse,
            HypercubeDistribution,
            MultiRelational,
            PreferentialAttachment,
            RelationalSimple,
            ScaledHierarchicalPairs,
            SimplexDistribution,
            SimplicialComplexDistribution,
            SingleUniform,
            SparseExponential,
            SparseUniform,
            SphericalDistribution,
            SyntheticDataModel,
            ToricDistribution,
        )

        for cls in [
            SparseUniform,
            SparseExponential,
            SingleUniform,
            CorrelatedPairs,
            HierarchicalPairs,
            ScaledHierarchicalPairs,
            AnticorrelatedPairs,
            RelationalSimple,
            MultiRelational,
            HierarchicalSparse,
            DAGDistribution,
            DAGBayesianPropagation,
            DAGRandomWalkToRoot,
            PreferentialAttachment,
            SimplexDistribution,
            SimplicialComplexDistribution,
            SphericalDistribution,
            ToricDistribution,
            HypercubeDistribution,
            SyntheticDataModel,
        ]:
            assert hasattr(cls, "sample"), f"{cls.__name__} missing sample"


# ============================================================================
# 6. Device bugs in specific modules
# ============================================================================


class TestDeviceBugs:
    """Targeted tests for known device-related issues."""

    def test_multi_relational_sample_creates_on_device(self):
        """MultiRelational.sample creates result tensor with torch.zeros
        without device= argument. Fixed: should use device=self.device."""
        from occhio.distributions.relational import MultiRelational

        d = MultiRelational(4, 0.1, k=2, device="cpu")
        result = d.sample(8)
        assert result.device == torch.device("cpu")
        assert result.shape == (8, 4)

    def test_attn_linear_ae_alpha_uses_torch_tensor(self):
        """AttnLinearAE uses torch.Tensor([0.1]) which ignores device.
        Fixed: should use torch.tensor([0.1], device=dev)."""
        from occhio.autoencoders.attention import AttnLinearAE

        ae = AttnLinearAE(4, 4, n_heads=2, dict_size=3, device="cpu")
        # Alpha should be on the same device as everything else
        assert ae.alpha.device.type == "cpu"

    def test_attn_attn_ae_alpha_uses_torch_tensor(self):
        """AttnAttnAE uses torch.Tensor([0.1]) which ignores device.
        Fixed: should use torch.tensor([0.1], device=dev)."""
        from occhio.autoencoders.attention import AttnAttnAE

        ae = AttnAttnAE(4, 4, n_heads=2, dict_size=3, device="cpu")
        assert ae.alpha.device.type == "cpu"

    def test_torus_distribution_place_features_uses_generator(self):
        """ToricDistribution._place_features uses torch.rand without generator.
        This means feature placement is non-deterministic even with a generator."""
        from occhio.distributions.toric import ToricDistribution

        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        d1 = ToricDistribution(10, device="cpu", generator=gen1)
        d2 = ToricDistribution(10, device="cpu", generator=gen2)
        # After fix, these should be equal
        assert torch.allclose(d1.feature_angles, d2.feature_angles)

    def test_spherical_distribution_random_placement_uses_generator(self):
        """SphericalDistribution._place_on_sphere_random uses torch.randn
        without generator. Fixed: should use self._randn."""
        from occhio.distributions.spherical import SphericalDistribution

        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        d1 = SphericalDistribution(10, manifold_dim=3, device="cpu", generator=gen1)
        d2 = SphericalDistribution(10, manifold_dim=3, device="cpu", generator=gen2)
        # After fix, these should be equal
        assert torch.allclose(d1.feature_positions, d2.feature_positions)

    def test_hypercube_distribution_random_placement_uses_generator(self):
        """HypercubeDistribution._place_features uses torch.rand without
        generator when n_features is not a perfect power."""
        from occhio.distributions.hypercube import HypercubeDistribution

        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        # Use 7 features so it falls back to random placement (not a perfect square/cube)
        d1 = HypercubeDistribution(7, cube_dim=2, device="cpu", generator=gen1)
        d2 = HypercubeDistribution(7, cube_dim=2, device="cpu", generator=gen2)
        assert torch.allclose(d1.feature_positions, d2.feature_positions)


# ============================================================================
# 7. Stale test: MLPEncoder kwargs conflict
# ============================================================================


class TestMLPEncoderKwargs:
    """Test that MLPEncoder construction works correctly with kwargs passthrough."""

    def test_mlp_encoder_does_not_accept_redundant_n_features(self):
        """MLPEncoder gets n_features from embedding[0].
        Passing it as a kwarg should raise TypeError."""
        from occhio.autoencoders.mlp import MLPEncoder

        with pytest.raises(TypeError, match="multiple values"):
            MLPEncoder(
                embedding=[5, 6, 3],
                unembedding=[3, 6, 5],
                n_features=5,
                n_hidden=3,
            )

    def test_mlp_encoder_with_correct_args(self):
        """MLPEncoder should work with just embedding/unembedding."""
        from occhio.autoencoders.mlp import MLPEncoder

        ae = MLPEncoder(
            embedding=[5, 6, 3],
            unembedding=[3, 6, 5],
            device="cpu",
        )
        assert ae.n_features == 5
        assert ae.n_hidden == 3


# ============================================================================
# 8. Distribution.to() completeness
# ============================================================================


class TestDistributionToCompleteness:
    """Distributions with custom tensor attributes must override to() properly."""

    def test_hierarchical_sparse_to_moves_parent_tensor(self):
        """HierarchicalSparse has parent_tensor and depth index tensors
        that must be moved by to()."""
        from occhio.distributions import HierarchicalSparse

        d = HierarchicalSparse(10, device="cpu")
        d.to("cpu")
        assert d.parent_tensor.device == torch.device("cpu")

    def test_dag_distribution_to_moves_adjacency(self):
        from occhio.distributions import DAGDistribution

        d = DAGDistribution(5, device="cpu")
        d.to("cpu")
        assert d.adjacency.device == torch.device("cpu")

    def test_dag_random_walk_to_root_to_moves_all_tensors(self):
        from occhio.distributions import DAGRandomWalkToRoot

        d = DAGRandomWalkToRoot(5, device="cpu")
        d.to("cpu")
        assert d.adjacency.device == torch.device("cpu")
        assert d._parent_padded.device == torch.device("cpu")
        assert d._parent_counts.device == torch.device("cpu")


# ============================================================================
# 9. Autoencoder forward pass smoke tests
# ============================================================================


class TestAutoEncoderForwardPass:
    """Verify encode/decode/forward produce correct shapes."""

    @pytest.mark.parametrize(
        "ae_factory",
        [
            lambda: __import__(
                "occhio.autoencoders", fromlist=["TiedLinear"]
            ).TiedLinear(5, 3),
            lambda: __import__(
                "occhio.autoencoders", fromlist=["TiedLinearRelu"]
            ).TiedLinearRelu(5, 3),
            lambda: __import__("occhio.autoencoders", fromlist=["SynthAE"]).SynthAE(
                5, 3
            ),
            lambda: __import__(
                "occhio.autoencoders", fromlist=["ComputeAutoEncoder"]
            ).ComputeAutoEncoder(5, 3),
            lambda: __import__(
                "occhio.autoencoders", fromlist=["AttnLinearAE"]
            ).AttnLinearAE(5, 4, n_heads=2, dict_size=3),
            lambda: __import__(
                "occhio.autoencoders", fromlist=["AttnAttnAE"]
            ).AttnAttnAE(5, 4, n_heads=2, dict_size=3),
            lambda: __import__(
                "occhio.autoencoders", fromlist=["LinearAttnAE"]
            ).LinearAttnAE(5, 4, n_heads=2, dict_size=3),
        ],
        ids=[
            "TiedLinear",
            "TiedLinearRelu",
            "SynthAE",
            "ComputeAE",
            "AttnLinearAE",
            "AttnAttnAE",
            "LinearAttnAE",
        ],
    )
    def test_forward_shapes(self, ae_factory):
        ae = ae_factory()
        x = torch.randn(8, ae.n_features)
        x_hat, z = ae(x)
        assert x_hat.shape == x.shape
        assert z.shape[0] == 8
        assert z.shape[1] == ae.n_hidden


# ============================================================================
# 10. Distribution sample shape tests
# ============================================================================


class TestDistributionSampleShapes:
    """All distributions should produce (batch_size, n_features) output."""

    @pytest.mark.parametrize(
        "dist_factory",
        [
            lambda: __import__(
                "occhio.distributions", fromlist=["SparseUniform"]
            ).SparseUniform(5, 0.5),
            lambda: __import__(
                "occhio.distributions", fromlist=["SparseExponential"]
            ).SparseExponential(5, 0.5),
            lambda: __import__(
                "occhio.distributions", fromlist=["SingleUniform"]
            ).SingleUniform(5),
            lambda: __import__(
                "occhio.distributions", fromlist=["CorrelatedPairs"]
            ).CorrelatedPairs(4, p_active=0.5, p_individual=0.5),
            lambda: __import__(
                "occhio.distributions", fromlist=["HierarchicalPairs"]
            ).HierarchicalPairs(4, 0.5),
            lambda: __import__(
                "occhio.distributions", fromlist=["AnticorrelatedPairs"]
            ).AnticorrelatedPairs(4, 0.5),
            lambda: __import__(
                "occhio.distributions", fromlist=["HierarchicalSparse"]
            ).HierarchicalSparse(5),
            lambda: __import__(
                "occhio.distributions", fromlist=["DAGDistribution"]
            ).DAGDistribution(5),
            lambda: __import__(
                "occhio.distributions", fromlist=["RelationalSimple"]
            ).RelationalSimple(4),
            lambda: __import__(
                "occhio.distributions", fromlist=["MultiRelational"]
            ).MultiRelational(4),
        ],
        ids=[
            "SparseUniform",
            "SparseExponential",
            "SingleUniform",
            "CorrelatedPairs",
            "HierarchicalPairs",
            "AnticorrelatedPairs",
            "HierarchicalSparse",
            "DAGDistribution",
            "RelationalSimple",
            "MultiRelational",
        ],
    )
    def test_sample_shape(self, dist_factory):
        d = dist_factory()
        s = d.sample(16)
        assert s.shape == (16, d.n_features)
        assert s.device == (d.device or torch.device("cpu"))
