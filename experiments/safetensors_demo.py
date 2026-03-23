# ABOUTME: Demo of AutoEncoderBase.save_weights / load_weights with safetensors.
# %% Imports
import tempfile
from pathlib import Path

import torch

from occhio.autoencoder import MLPEncoder, TiedLinearRelu
from occhio.distributions.sparse import SparseUniform
from occhio.toy_model import ToyModel

# %%
# 1. Create and train a model
N_FEATURES, N_HIDDEN = 20, 5
gen = torch.Generator().manual_seed(42)

model = ToyModel(
    distribution=SparseUniform(N_FEATURES, p_active=0.1, generator=gen),
    ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen),
    importances=0.7 ** torch.arange(N_FEATURES, dtype=torch.float32),
)
losses, _ = model.fit(n_epochs=500, batch_size=256, track_losses=True)
print(f"Trained model final loss: {losses[-1]:.6f}")

# %%
# 2. Save weights
save_dir = Path(tempfile.mkdtemp())
path = model.ae.save_weights(save_dir / "trained_relu")
print(f"Saved to: {path}")
print(f"File size: {path.stat().st_size} bytes")

# %%
# 3. Load into a fresh model (different seed → different init weights)
fresh_ae = TiedLinearRelu(
    N_FEATURES, N_HIDDEN, generator=torch.Generator().manual_seed(999)
)

# 4. Verify weights differ before loading
assert not torch.equal(model.ae.state_dict()["W"], fresh_ae.state_dict()["W"])
print("Before load: weights differ ✓")

fresh_ae.load_weights(path)

# 5. Verify exact match after loading
for key in model.ae.state_dict():
    assert torch.equal(model.ae.state_dict()[key], fresh_ae.state_dict()[key])
print("After load:  all weights match exactly ✓")

# %%
# 6. Class mismatch is caught
mlp = MLPEncoder(
    embedding=[N_FEATURES, 10, N_HIDDEN],
    unembedding=[N_HIDDEN, 10, N_FEATURES],
    n_features=N_FEATURES,
    n_hidden=N_HIDDEN,
)
try:
    mlp.load_weights(path)
    assert False, "Should have raised"
except TypeError as e:
    print(f"Class mismatch caught: {e}")

# %%
# 7. Shape mismatch is caught
wrong_size = TiedLinearRelu(N_FEATURES, 10)  # n_hidden=10 vs saved n_hidden=5
try:
    wrong_size.load_weights(path)
    assert False, "Should have raised"
except RuntimeError as e:
    print(f"Shape mismatch caught: {str(e)[:80]}...")

# %%
# 8. Continue training with loaded weights
gen2 = torch.Generator().manual_seed(7)
model2 = ToyModel(
    distribution=SparseUniform(N_FEATURES, p_active=0.1, generator=gen2),
    ae=TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen2),
    importances=0.7 ** torch.arange(N_FEATURES, dtype=torch.float32),
)
model2.ae.load_weights(path)
print("\nContinuing training from loaded weights...")
losses2, _ = model2.fit(n_epochs=500, batch_size=256, track_losses=True)
print(f"Loss after further training:  {losses2[-1]:.6f}")

print("\nAll checks passed.")

# %%
