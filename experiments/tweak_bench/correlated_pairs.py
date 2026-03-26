# %%
"""TiedLinearRelu on CorrelatedPairs — basic experiment boilerplate."""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.autoencoder import TiedLinearRelu
from occhio.distributions.correlated import CorrelatedPairs
from occhio.toy_model import ToyModel

# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 1296
D_HIDDEN = 100
N_EPOCHS = 10_000
BATCH_SIZE = 512

# %%
# --- Distribution ---
np.random.seed(8)

high = 0.5
low = 1.0 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
corrs = 0.5 + np.random.random(N_FEATURES) / 2

dist = CorrelatedPairs(
    N_FEATURES, p_active=firing_probs, p_individual=corrs, device=DEVICE
)

# Average L0
samples = dist.sample(100_000)
mean_l0 = (samples > 0).float().sum(dim=-1).mean().item()
print(f"Average L0: {mean_l0:.2f}")

# %%
# --- Train ---
gen = torch.Generator(DEVICE).manual_seed(SEED)
ae = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

losses, _ = tm.fit(N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

# %%
# --- Plot: Feature Norms and Feature Dimensionalities ---
fn = tm.feature_norms.detach().cpu().numpy()
fd = tm.feature_dimensionalities.detach().cpu().numpy()

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Feature Norms", "Feature Dimensionalities"],
)

fig.add_trace(
    go.Scatter(x=np.arange(N_FEATURES), y=fn, mode="lines", name="Norms"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=np.arange(N_FEATURES),
        y=fd,
        mode="lines",
        name="Dimensionalities",
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Feature index", row=1, col=1)
fig.update_xaxes(title_text="Feature index", row=1, col=2)
fig.update_yaxes(title_text="‖w‖", row=1, col=1)
fig.update_yaxes(title_text="Dimensionality", row=1, col=2)
fig.update_layout(
    title=f"TiedLinearRelu — CorrelatedPairs (N={N_FEATURES}, D={D_HIDDEN})",
    height=400,
    width=900,
)
fig.show()

# %%
# --- Plot: Feature Norms vs Correlation ---
fig = go.Figure()
pair_corrs = corrs
fig.add_trace(
    go.Scatter(x=pair_corrs, y=fn, mode="markers", marker=dict(size=3), name="Features")
)
fig.update_layout(
    title="Feature Norms vs Correlation",
    xaxis_title="Correlation",
    yaxis_title="‖w‖",
)
fig.show()

# %%
