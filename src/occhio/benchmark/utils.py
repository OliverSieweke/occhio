import torch

from occhio import ToyModel
from occhio.autoencoders.hugging_face import HuggingFaceAutoEncoder
from occhio.benchmark.configs import (
    OCCHIO_HF_DISTRIBUTIONS_REPO,
    OCCHIO_HF_MODELS_REPO,
    BenchmarkName,
)
from occhio.distributions.hugging_face import HuggingFaceDistribution


def toy_model_from_benchmark(
    name: BenchmarkName,
    # [2026-03-25 | OliverSieweke] TODO: WARNING: this will only make sense if distribution and model live in the same repo
    revision: str | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
):

    return ToyModel(
        distribution=HuggingFaceDistribution(
            repo_id=OCCHIO_HF_DISTRIBUTIONS_REPO,
            filename=f"{name}/samples/samples.safetensors",
            revision=revision,
            device=device,
            generator=generator,
        ),
        ae=HuggingFaceAutoEncoder(
            repo_id=OCCHIO_HF_MODELS_REPO,
            filename=f"{name}/weights/weights.safetensors",
            revision=revision,
            device=device,
        ),
        device=device,
    )
