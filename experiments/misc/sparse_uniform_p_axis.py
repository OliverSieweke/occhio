# %%
# ABOUTME: Experiment: ModelGrid over sparsity p with TiedLinearRelu on SparseUniform.
# ABOUTME: The axis is `p`; each feature i has p_active = p / (i+1).

import plotly.graph_objects as go
import torch

from occhio import ToyModel
from occhio.model_grid import ModelGrid, Axis
from occhio.distributions import SparseUniform
from occhio.autoencoder import TiedLinearRelu
import occhio.visualization as ov

# %% ── config ───────────────────────────
DEVICE = "mps"
N_FEATURES = 24
N_HIDDEN = 3


def create_model(params):
    gen = torch.Generator(DEVICE)
    gen.manual_seed(8)

    p = params["p"]
    p_active = [p / (i + 1) for i in range(N_FEATURES)]

    dist = SparseUniform(N_FEATURES, p_active, generator=gen, device=DEVICE)
    ae = TiedLinearRelu(N_FEATURES, N_HIDDEN, generator=gen, device=DEVICE)
    return ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)


# %% ── build grid ───────────────────
mg = ModelGrid(
    create_model,
    axes=[
        Axis(label="p", values=torch.logspace(0, -3, steps=24)),
    ],
)

# %% ── train ─────────────────────
mg.fit(20_000, batch_size=512)

# %% ── firing probability vs feature dimensionality ─────────────────────────
all_p_active = []
all_dim = []
all_model_idx = []
avg_p_active = []
avg_dim = []
avg_model_idx = []


for k in range(len(mg.axes[0].values)):
    model = mg[k]
    assert isinstance(model, ToyModel)
    p = mg.axes[0].values[k].item()
    p_actives = [p / (i + 1) for i in range(N_FEATURES)]
    dims = model.feature_dimensionalities.cpu().tolist()
    all_p_active.extend(p_actives)
    all_dim.extend(dims)
    all_model_idx.extend([k] * N_FEATURES)
    avg_p_active.append(sum(p_actives) / N_FEATURES)
    avg_dim.append(sum(dims) / N_FEATURES)
    avg_model_idx.append(k)


fig_dim = go.Figure(
    [
        go.Scatter(
            x=all_p_active,
            y=all_dim,
            mode="markers",
            name="features",
            marker=dict(
                size=6,
                opacity=1.0,
                color=all_model_idx,
                colorscale="oxy",
            ),
        ),
        # go.Scatter(
        #     x=avg_p_active,
        #     y=avg_dim,
        #     mode="markers",
        #     name="model average",
        #     marker=dict(
        #         size=6,
        #         opacity=0.9,
        #         color=avg_model_idx,
        #         colorscale="oxy",
        #         colorbar=dict(title="Model (p index)"),
        #         line=dict(width=1, color="black"),
        #     ),
        # ),
    ]
)
fig_dim.update_layout(
    xaxis_title="Relative firing probability",
    yaxis_title="Feature dimensionality",
    xaxis_type="log",
    # yaxis_type="log",
    height=500,
)
fig_dim.show()

# %% ── visualize ────────────────────────
fig = ov.plot_geometry(mg)
fig.update_layout(height=600)
fig.show()
# %%

fig = ov.plot_representation(mg[:3])
fig.update_layout(height=600)
fig.show()

# %%
fig = ov.plot_representation(mg[-3:])
fig.update_layout(height=600)
fig.show()

# %%
fig = ov.plot_feature_geometry_3d(mg[-1])
fig.update_layout(height=600)
fig.show()

# %% ══════════════════════════════════════════════════════════════════════════
# Phase transition: 2 features → 1 hidden dim
# X axis: p_active of feature 0; Y axis: p_active of feature 1
# ══════════════════════════════════════════════════════════════════════════════

N_FEATURES_2 = 2
N_HIDDEN_2 = 1
N_STEPS = 11


def create_model_2d(params):
    gen = torch.Generator(DEVICE)
    gen.manual_seed(8)

    p1 = params["p1"].item()
    relp2 = params["relp2"].item()

    dist = SparseUniform(
        N_FEATURES_2, [p1, min(p1 * relp2, 1)], generator=gen, device=DEVICE
    )
    ae = TiedLinearRelu(N_FEATURES_2, N_HIDDEN_2, generator=gen, device=DEVICE)
    return ToyModel(distribution=dist, ae=ae, generator=gen, device=DEVICE)


# %% ── build grid ─────────────────────────────────────────────────────────────
mg2 = ModelGrid(
    create_model_2d,
    axes=[
        Axis(label="relp2", values=torch.logspace(-1, 1, steps=N_STEPS)),
        Axis(label="p1", values=torch.logspace(-2, 0, steps=N_STEPS)),
    ],
)

# %% ── train ──────────────────────────────────────────────────────────────────
mg2.fit(10_000, batch_size=128)

# %% ── visualize ──────────────────────────────────────────────────────────────
fig2 = ov.plot_phase_change(mg2, tracked_feature=0)
fig2.update_layout(title="Phase transition: 2→1, feature 0", height=600)
fig2.show()

# %%
fig3 = ov.plot_phase_change(mg2, tracked_feature=1)
fig3.update_layout(title="Phase transition: 2→1, feature 1", height=600)
fig3.show()

# %%
mg2[-1, -2].ae.b

# %%
