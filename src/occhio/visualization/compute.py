from typing import Protocol, runtime_checkable
import numpy as np
import torch
from torch import Tensor
import plotly.express as px
import plotly.graph_objects as go
import torch.nn.functional as F
from occhio.toy_model import ToyModel


@runtime_checkable
class _ComputeAE(Protocol):
    n_features: int

    def encode(self, x: Tensor) -> Tensor: ...
    def compute_step(self, h: Tensor) -> Tensor: ...
    def decode(self, z: Tensor) -> Tensor: ...


def plot_decode_plane(tm: ToyModel, title: str = "Decode plane"):
    assert isinstance(tm.ae, _ComputeAE)
    ae = tm.ae
    """
    Grid the 2-D latent space, decode every point, colour by dominant feature,
    and overlay where each one-hot e_i maps after encode → compute_step.
    """
    pal = px.colors.qualitative.Set1
    N = ae.n_features

    with torch.no_grad():
        h1_pts = np.zeros((N, 2))  # after encode
        z_pts = np.zeros((N, 2))  # after compute_step

        for s in range(N):
            e = F.one_hot(torch.tensor([s]), N).float()
            h = ae.encode(e)
            z = ae.compute_step(h)
            h1_pts[s] = h[0].numpy()
            z_pts[s] = z[0].numpy()

        all_pts = np.vstack([h1_pts, z_pts])
        margin = 1.0
        x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
        y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

        res = 300
        xs = np.linspace(x_min, x_max, res)
        ys = np.linspace(y_min, y_max, res)
        xx, yy = np.meshgrid(xs, ys)
        pts = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
        )

        out = ae.decode(pts)  # (res², N)
        max_out = out.max(dim=1).values
        dominant_flat = out.argmax(dim=1).float()
        dominant_flat[max_out == 0] = float("nan")
        dominant = dominant_flat.numpy().reshape(res, res)

    colorscale = []
    for i in range(N):
        colorscale += [[i / N, pal[i % len(pal)]], [(i + 1) / N, pal[i % len(pal)]]]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=dominant,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=N - 0.5,
            opacity=0.30,
            showscale=False,
        )
    )

    for s in range(N):
        color = pal[s % len(pal)]

        fig.add_trace(
            go.Scatter(
                x=[h1_pts[s, 0]],
                y=[h1_pts[s, 1]],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="circle",
                    line=dict(width=1.5, color="black"),
                ),
                text=[f"We{s}"],
                textposition="bottom center",
                name=f"s{s} (We)",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[z_pts[s, 0]],
                y=[z_pts[s, 1]],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=color,
                    symbol="star",
                    line=dict(width=1.5, color="black"),
                ),
                text=[f"ZWe{s}"],
                textposition="top center",
                name=f"s{s} (ZWe)",
                showlegend=True,
            )
        )
        fig.add_annotation(
            x=z_pts[s, 0],
            y=z_pts[s, 1],
            ax=h1_pts[s, 0],
            ay=h1_pts[s, 1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=1.5,
            arrowcolor=color,
            showarrow=True,
        )

    fig.update_layout(
        title=title,
        xaxis_title="dim 0 (latent)",
        yaxis_title="dim 1 (latent)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600,
    )
    return fig
