import torch

from occhio import ToyModel
from occhio.autoencoder import SynthAE
from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder
from occhio.benchmark.configs import (
    OCCHIO_HF_DISTRIBUTIONS_REPO,
    OCCHIO_HF_MODELS_REPO,
    BenchmarkDistributionName,
)
from occhio.distributions.hugging_face import HuggingFaceDistribution


def toy_model_from_benchmark(
    name: BenchmarkDistributionName,
    # [2026-03-25 | OliverSieweke] TODO: WARNING: this will only make sense if distribution and model live in the same repo
    revision: str | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    ae_type: str = "huggingface",
    ae_kwargs: dict | None = None,
):
    distribution = HuggingFaceDistribution(
        repo_id=OCCHIO_HF_DISTRIBUTIONS_REPO,
        filename=f"{name}/samples/samples.safetensors",
        revision=revision,
        device=device,
        generator=generator,
    )

    if ae_type == "huggingface":
        ae = HuggingFaceAutoEncoder(
            repo_id=OCCHIO_HF_MODELS_REPO,
            filename=f"{name}/weights/weights.safetensors",
            revision=revision,
            device=device,
        )
    elif ae_type == "synth":
        # Get n_features from the distribution
        kwargs = dict(
            n_features=distribution.n_features,
            device=device,
        )
        if ae_kwargs:
            kwargs.update(ae_kwargs)
        ae = SynthAE(**kwargs)
    else:
        raise ValueError(f"Unknown ae_type: {ae_type!r}. Use 'huggingface' or 'synth'.")

    return ToyModel(
        distribution=distribution,
        ae=ae,
        device=device,
    )
