"""Example with Distribution Stack. See nice hierarchical structure in embedding matrix."""

# %%
from occhio.distributions import SparseUniform, DistributionStack, CorrelatedPairs
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
gen = torch.Generator("cpu")
gen.manual_seed(2)

list_of_Ws = []


def my_hook(hook_data):
    list_of_Ws.append(
        (hook_data["epoch"], hook_data["tm"].ae.W.detach().numpy().copy())
    )


dist = DistributionStack(
    [SparseUniform(3, 0.5, generator=gen) for i in range(3)], "single"
)
# dist = DistributionStack(
#     [
#         CorrelatedPairs(4, p_active=0.5, p_individual=0.7, generator=gen)
#         for i in range(2)
#     ],
#     "single",
# )


n_hidden = 2
importances = torch.tensor([0.99**i for i in range(dist.n_features)])

# %%
ae = TiedLinearRelu(dist.n_features, n_hidden, generator=gen)
tm = ToyModel(dist, ae, importances=importances)
losses = tm.fit(40_000, verbose=False, hooks=[my_hook], hook_freq=1000)

# %%
px.line(losses)

# %%
# Interactive version with slider to explore embeddings at different epochs

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Training Loss", "Embedding Scatter"),
    column_widths=[0.5, 0.5],
)

# Add loss line plot (static)
fig.add_trace(
    go.Scatter(x=list(range(len(losses))), y=losses, mode="lines", name="Loss"),
    row=1,
    col=1,
)

# Create frames for animation (one per epoch in list_of_Ws)
frames = []
for idx, (epoch, emb_mat) in enumerate(list_of_Ws):
    # Create marker trace for this epoch
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=list(range(len(losses))), y=losses, mode="lines"
                ),  # Keep loss plot
                go.Scatter(
                    x=emb_mat[0],
                    y=emb_mat[1],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=list(range(dist.n_features)),
                        colorscale="Viridis",
                    ),
                    showlegend=False,
                ),
            ],
            name=str(epoch),
            layout=go.Layout(
                title_text=f"Epoch {epoch}",
                shapes=[
                    dict(
                        type="line",
                        x0=epoch,
                        x1=epoch,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="y domain",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                ],
            ),
        )
    )

# Add initial scatter plot
emb_mat = list_of_Ws[0][1]
fig.add_trace(
    go.Scatter(
        x=emb_mat[0],
        y=emb_mat[1],
        mode="markers",
        marker=dict(size=15, color=list(range(dist.n_features)), colorscale="Viridis"),
        showlegend=False,
    ),
    row=1,
    col=2,
)

# Add frames to figure
fig.frames = frames

# Add slider
steps = []
for idx, (epoch, _) in enumerate(list_of_Ws):
    step = dict(
        method="animate",
        args=[
            [str(epoch)],
            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
        ],
        label=str(epoch),
    )
    steps.append(step)

sliders = [
    dict(
        active=0,
        yanchor="top",
        y=-0.1,
        xanchor="left",
        x=0.0,
        currentvalue=dict(prefix="Epoch: ", visible=True, xanchor="right"),
        pad=dict(b=10, t=50),
        len=0.9,
        steps=steps,
    )
]

fig.update_layout(
    sliders=sliders,
    height=500,
    showlegend=False,
    shapes=[
        dict(
            type="line",
            x0=list_of_Ws[0][0],
            x1=list_of_Ws[0][0],
            y0=0,
            y1=1,
            xref="x",
            yref="y domain",
            line=dict(color="red", width=2, dash="dash"),
        )
    ],
)

fig.show()

# %%
