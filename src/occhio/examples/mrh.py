# %%
"""MRH experiment: compare TiedLinearRelu vs MultiHeadSoftmaxAE on SparseUniform.

Sweeps over p_active values and plots W embeddings (one-hot encodings projected
to the 2D latent space) for both architectures side by side.  The top row shows
the standard linear bottleneck (LRH-style), the bottom row shows the multi-head
softmax bottleneck (MRH-style).
"""

from occhio.distributions.sparse import SparseUniform
from occhio.autoencoder import TiedLinearRelu, MultiHeadSoftmaxAE
from occhio.toy_model import ToyModel
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_features = 6
n_hidden = 2
n_heads = 2
dict_size = 4

# %%
importances = torch.tensor([0.9**i for i in range(n_features)])
gen = torch.Generator("cpu")

p_actives = [0.01, 0.1, 0.2, 0.5, 0.75, 0.99]

# Train all models and collect W embeddings
results: dict[str, list[np.ndarray]] = {"TiedLinearRelu": [], "MultiHeadSoftmaxAE": []}

for p_active in p_actives:
    # --- TiedLinearRelu (LRH baseline) ---
    gen.manual_seed(7)
    dist = SparseUniform(n_features, p_active, generator=gen)
    ae_linear = TiedLinearRelu(n_features, n_hidden, generator=gen)
    tm_linear = ToyModel(dist, ae_linear, importances=importances)
    tm_linear.fit(16_000, verbose=False)
    results["TiedLinearRelu"].append(tm_linear.W.detach().numpy())

    # --- MultiHeadSoftmaxAE (MRH) ---
    gen.manual_seed(7)
    dist = SparseUniform(n_features, p_active, generator=gen)
    ae_mrh = MultiHeadSoftmaxAE(
        n_features, n_hidden, n_heads=n_heads, dict_size=dict_size, generator=gen
    )
    tm_mrh = ToyModel(dist, ae_mrh, importances=importances)
    tm_mrh.fit(16_000, verbose=False)
    results["MultiHeadSoftmaxAE"].append(tm_mrh.W.detach().numpy())

# %%
row_labels = ["TiedLinearRelu (LRH)", "MultiHeadSoftmaxAE (MRH)"]
row_keys = ["TiedLinearRelu", "MultiHeadSoftmaxAE"]

fig = make_subplots(
    rows=2,
    cols=len(p_actives),
    subplot_titles=[f"p={p}" for p in p_actives] + [f"p={p}" for p in p_actives],
    vertical_spacing=0.08,
    horizontal_spacing=0.04,
)

theta = np.linspace(0, 2 * np.pi, 100)
colors = [
    "#440154",
    "#443983",
    "#31688e",
    "#21918c",
    "#35b779",
    "#fde725",
]

for row_idx, key in enumerate(row_keys):
    for col_idx, (p_active, W) in enumerate(zip(p_actives, results[key])):
        r, c = row_idx + 1, col_idx + 1
        show_legend = row_idx == 0 and col_idx == 0

        # Unit circle
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="black", dash="dash", width=0.5),
                opacity=0.3,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Feature embeddings
        for i in range(n_features):
            fig.add_trace(
                go.Scatter(
                    x=[W[0, i]],
                    y=[W[1, i]],
                    mode="markers+text",
                    marker=dict(color=colors[i], size=10),
                    text=[str(i)],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"feature {i}",
                    showlegend=show_legend,
                    legendgroup=f"f{i}",
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(
            range=[-1.5, 1.5],
            scaleanchor=f"y{row_idx * len(p_actives) + col_idx + 1}",
            row=r,
            col=c,
        )
        fig.update_yaxes(range=[-1.5, 1.5], row=r, col=c)

# Row labels via y-axis titles on the first column
fig.update_yaxes(title_text=row_labels[0], row=1, col=1)
fig.update_yaxes(title_text=row_labels[1], row=2, col=1)

fig.update_layout(
    title_text=(
        f"Feature embeddings: LRH vs MRH bottleneck on SparseUniform<br>"
        f"<sub>n_features={n_features}, n_hidden={n_hidden}, "
        f"heads={n_heads}, dict_size={dict_size}</sub>"
    ),
    height=600,
    width=300 * len(p_actives),
)
fig.show()

# %%
