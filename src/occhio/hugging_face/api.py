# %%
import json
from pathlib import Path

import torch
import tqdm.auto
from huggingface_hub import hf_hub_download, snapshot_download
from sae_lens.synthetic import SyntheticModel
from safetensors.torch import load_file


def download_distributions():
    return snapshot_download(
        "kaushikreddyxyz/occhio-distributions",
        repo_type="dataset",
        tqdm_class=tqdm.auto.tqdm,
    )


# [2026-03-25 | OliverSieweke] TODO: We probably want the models to be stored with the distributions instead of 2 repos
def download_models():
    return snapshot_download(
        "kaushikreddyxyz/occhio-models", repo_type="model", tqdm_class=tqdm.auto.tqdm
    )


#
# tensor = hf_hub_download(
#     REPO_ID,
#     repo_type="dataset",
#     filename="correlated_pairs/samples/samples.safetensors",
# )
# description = hf_hub_download(
#     REPO_ID,
#     repo_type="dataset",
#     filename="correlated_pairs/samples/samples.json",
# )
#
# samples_meta = json.loads(Path(description).read_text())
#
# # samples_meta = json.loads(description).read_text()
# print(samples_meta)
# description = samples_meta.get("description", "")
# # with open(path) as f:
# data = load_file(tensor)
#
# print(data["samples"])
#
# samples = torch.tensor(data)  # adjust depending on the JSON structure
# print(f"Shape: {samples.shape}, dtype: {samples.dtype}")
#
#
# # --- Download all distributions ---
# local_dir = snapshot_download(
#     REPO_ID,
#     repo_type="dataset",
# )
#
# # Load all distributions into a dict
# from pathlib import Path
#
# distributions = {}
# for samples_file in Path(local_dir).glob("*/samples/samples.json"):
#     dist_name = samples_file.parent.parent.name
#     with open(samples_file) as f:
#         distributions[dist_name] = torch.tensor(json.load(f))
#     print(f"{dist_name}: {distributions[dist_name].shape}")

#
# model = SyntheticModel.from_pretrained(
#     "decoderesearch/synth-sae-bench-16k-v1",
#     device=device,
# )
#
# hidden_acts, feature_acts = model.sample_with_features(batch_size=10_000)
