"""Dynamic scatter animation -- standalone function (not a SinglePlot).

``plot_dynamic_scatter`` takes ``losses`` and ``hooks_list`` as constructor
args rather than model data, so it does not fit the SinglePlot protocol.
It is kept as a standalone function and re-exported from the package.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_dynamic_scatter(
    losses: list[float], hooks_list: list[tuple], loss_stride: int = 1
):
    """Creates a dynamic plot of loss and another property as a scatter.

    Args:
        losses: list of losses.
        hooks_list: list of tuples of (loss_epoch, Tensor[k, n]) with k>=2
        loss_stride: plot every nth loss entry (first and last are always included).
    """
    all_indices = list(range(len(losses)))
    if loss_stride > 1 and len(losses) > 2:
        sampled = list(range(0, len(losses), loss_stride))
        if sampled[-1] != len(losses) - 1:
            sampled.append(len(losses) - 1)
        loss_x = sampled
        loss_y = [losses[i] for i in sampled]
    else:
        loss_x = all_indices
        loss_y = losses

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training Loss", "Scatter"),
        column_widths=[0.5, 0.5],
    )
    n: int = hooks_list[0][-1].shape[-1]

    all_x = [emb_mat[0] for _, emb_mat in hooks_list]
    all_y = [emb_mat[1] for _, emb_mat in hooks_list]
    x_min = min(v.min().item() for v in all_x)
    x_max = max(v.max().item() for v in all_x)
    y_min = min(v.min().item() for v in all_y)
    y_max = max(v.max().item() for v in all_y)
    x_pad = (x_max - x_min) * 0.05 or 0.5
    y_pad = (y_max - y_min) * 0.05 or 0.5

    fig.add_trace(
        go.Scatter(x=loss_x, y=loss_y, mode="lines", name="Loss"),
        row=1,
        col=1,
    )

    frames = []
    for idx, (epoch, emb_mat) in enumerate(hooks_list):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=loss_x, y=loss_y, mode="lines"),
                    go.Scatter(
                        x=emb_mat[0],
                        y=emb_mat[1],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=list(range(n)),
                            colorscale="Viridis",
                        ),
                        text=list(range(n)),
                        hoverinfo="text",
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

    emb_mat = hooks_list[0][1]
    fig.add_trace(
        go.Scatter(
            x=emb_mat[0],
            y=emb_mat[1],
            mode="markers",
            marker=dict(size=10, color=list(range(n)), colorscale="Viridis"),
            text=list(range(n)),
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.frames = frames

    steps = []
    for idx, (epoch, _) in enumerate(hooks_list):
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
        xaxis2=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis2=dict(range=[y_min - y_pad, y_max + y_pad]),
        shapes=[
            dict(
                type="line",
                x0=hooks_list[0][0],
                x1=hooks_list[0][0],
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                line=dict(color="red", width=2, dash="dash"),
            )
        ],
    )
    return fig
