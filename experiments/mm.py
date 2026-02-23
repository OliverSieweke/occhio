# %%
"""
Toy Model of Superposition (TMoS) inspired mini-LLM — PyTorch.

Architecture:
  Encode:  h1 = W @ x                       (N -> k)
  Compute: h2 = h1 + Z.T @ ReLU(Z @ h1)     (k -> k, nonlinear residual)
  Decode:  y  = ReLU(W.T @ h2 + b)           (k -> N)

Experiment: same one-hot Markov chain data, two loss functions:
  1. Cross-entropy loss (target = integer class index)
  2. MSE loss          (target = one-hot float vector)
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Markov Chain ─────────────────────────────────────────────────────────────


# %%
def make_transition_matrix(N, sparsity=0.3, seed=42):
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.dirichlet(np.ones(N) * sparsity, size=N), dtype=torch.float32
    )


def sample_markov_chain(P, T, seed=0):
    rng = np.random.default_rng(seed)
    P_np = P.numpy() if isinstance(P, torch.Tensor) else np.asarray(P)
    N = P_np.shape[0]
    states = [int(rng.integers(N))]
    for _ in range(T - 1):
        p = P_np[states[-1]].astype(np.float64)
        p /= p.sum()
        states.append(int(rng.choice(N, p=p)))
    return torch.tensor(states, dtype=torch.long)


# %%


class TMoSModel(nn.Module):
    """
    W: (k, N)  — encoder/decoder (tied)
    Z: (h, k)  — compute step

    b: (N,)    — decode bias
    b2: (h, )  — compute bias
    """

    def __init__(self, N: int, k: int, h: int = 5, seed: int = 10):
        gen = torch.Generator().manual_seed(seed)
        super().__init__()
        self.N, self.k, self.h = N, k, h

        self.W = nn.Parameter(torch.randn(k, N, generator=gen) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen) / k)
        self.b = nn.Parameter(torch.zeros(N))

    def embedd(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def compute(self, h: torch.Tensor) -> torch.Tensor:
        return h + h @ self.Z.T

    def unembedd(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(z @ self.W + self.b, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N) -> (B, N)"""
        return self.unembedd(self.compute(self.embedd(x)))


# %%
def train(
    X_all,
    Y_all,
    N,
    k=2,
    h=5,
    loss_type="ce",
    epochs=3000,
    lr=3e-4,
    seed=42,
    batch_size=1024,
):
    """
    X_all:     (T, N) float one-hot inputs
    Y_all:     (T,)   int   indices    when loss_type='ce'
               (T, N) float one-hot    when loss_type='mse'
    """
    model = TMoSModel(N, k, h, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    rng = torch.Generator().manual_seed(seed)
    importance = 0.9 ** torch.arange(N)

    losses = []
    for epoch in range(epochs):
        idx = torch.randint(len(Y_all), (batch_size,), generator=rng)
        Xb, Yb = X_all[idx], Y_all[idx]

        y_pred = model(Xb)
        target_idx = Yb if Yb.dim() == 1 else Yb.argmax(dim=-1)
        weights = importance[target_idx]
        if loss_type == "ce":
            per_sample = F.nll_loss(torch.log(y_pred), target_idx, reduction="none")
        else:
            per_sample = F.mse_loss(y_pred, Yb, reduction="none").sum(dim=-1)
        loss = (per_sample * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            tag = "CE" if loss_type == "ce" else "MSE"
            print(f"  Epoch {epoch + 1:4d} | {tag} loss: {loss.item():.4f}")

    with torch.no_grad():
        P_learned = torch.zeros(N, N)
        for s in range(N):
            x = F.one_hot(torch.tensor([s]), N).float()
            P_learned[s] = model(x)[0]

    return model, P_learned, losses


# %%


def plot_transition_matrices(P_true, P_ce, P_mse, N=5):
    # Convert to numpy for plotly
    P_true = P_true.numpy() if isinstance(P_true, torch.Tensor) else P_true
    P_ce = P_ce.numpy() if isinstance(P_ce, torch.Tensor) else P_ce
    P_mse = P_mse.numpy() if isinstance(P_mse, torch.Tensor) else P_mse

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["True P", "CE Learned P", "MSE Learned P"],
    )

    for col, (P, title) in enumerate(
        [(P_true, "True P"), (P_ce, "CE Learned P"), (P_mse, "MSE Learned P")],
        start=1,
    ):
        fig.add_trace(
            go.Heatmap(
                z=P,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in P],
                texttemplate="%{text}",
                showscale=(col == 3),
                name=title,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Next state", row=1, col=col)
        fig.update_yaxes(title_text="Current state", row=1, col=col)

    fig.update_layout(title_text="Transition Matrices", height=420)
    fig.show()


def plot_results(model, losses, N, title, loss_label):
    colors = px.colors.qualitative.Set1
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{loss_label} Loss", "W columns in R²"],
    )

    fig.add_trace(
        go.Scatter(
            y=losses, mode="lines", name=f"{loss_label} Loss", line=dict(width=1)
        ),
        row=1,
        col=1,
    )

    with torch.no_grad():
        W = model.W.numpy()  # (k, N)

    for i in range(N):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[0, float(W[0, i])],
                y=[0, float(W[1, i])],
                mode="lines+markers+text",
                marker=dict(size=[4, 10], color=color),
                text=["", f"s{i}"],
                textposition="top center",
                name=f"s{i}",
                line=dict(color=color, width=2.5),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text=title,
        height=450,
        yaxis2=dict(scaleanchor="x2", scaleratio=1),
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text=loss_label, row=1, col=1)
    fig.update_xaxes(title_text="dim 0", row=1, col=2)
    fig.update_yaxes(title_text="dim 1", row=1, col=2)
    fig.show()


def plot_compute_step(model, N, title="Compute step: h₁ → h₂"):
    """
    For each one-hot input e_i, draws h₁ (encoder output) and h₂ (after the
    nonlinear residual) as points in the 2-D latent space, with an arrow
    showing the transformation.
    """
    colors = px.colors.qualitative.Set1

    H1 = np.zeros((N, 2))
    H2 = np.zeros((N, 2))
    with torch.no_grad():
        for s in range(N):
            x = F.one_hot(torch.tensor([s]), N).float()
            h1 = x @ model.W.T  # (1, k)
            h2 = h1 @ model.Z.T
            H1[s] = h1[0].numpy()
            H2[s] = h2[0].numpy()

    fig = go.Figure()

    for i in range(N):
        color = colors[i % len(colors)]

        # h1 — circle
        fig.add_trace(
            go.Scatter(
                x=[H1[i, 0]],
                y=[H1[i, 1]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="circle"),
                text=[f"h1·s{i}"],
                textposition="top right",
                name=f"s{i}",
                showlegend=True,
            )
        )

        # h2 — diamond
        fig.add_trace(
            go.Scatter(
                x=[H2[i, 0]],
                y=[H2[i, 1]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="diamond"),
                text=[f"h2·s{i}"],
                textposition="bottom right",
                showlegend=False,
            )
        )

        # Arrow from h1 to h2
        fig.add_annotation(
            x=H2[i, 0],
            y=H2[i, 1],
            ax=H1[i, 0],
            ay=H1[i, 1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
            text="",
            showarrow=True,
        )

    fig.update_layout(
        title=title,
        xaxis_title="dim 0",
        yaxis_title="dim 1",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=500,
    )
    fig.show()


def plot_decode_plane(model, N, title="Decode plane: ReLU(h·W + b)"):
    """
    Grid the 2-D latent (Zh) space, apply the decode step ReLU(h @ W + b) to
    every grid point, colour each point by the feature with the highest
    activation, and overlay where each one-hot input maps to in that space.
    """
    pal = px.colors.qualitative.Set1

    with torch.no_grad():
        W = model.W  # (k, N)
        b = model.b  # (N,)
        Z = model.Z  # (k, k)

        # Where does each one-hot e_i end up before and after Z?
        onehot_h1 = np.zeros((N, 2))  # h1  = e @ W.T       (before Z)
        onehot_Zh = np.zeros((N, 2))  # Zh1 = h1 @ Z.T      (after Z)
        for s in range(N):
            e = F.one_hot(torch.tensor([s]), N).float()
            h1 = model.embedd(e)
            Zh1 = model.compute(h1)
            onehot_h1[s] = h1[0].numpy()
            onehot_Zh[s] = Zh1[0].numpy()

        # Build a 2-D grid covering both sets of points
        all_pts = np.vstack([onehot_h1, onehot_Zh])
        margin = 1.0
        x_min = all_pts[:, 0].min() - margin
        x_max = all_pts[:, 0].max() + margin
        y_min = all_pts[:, 1].min() - margin
        y_max = all_pts[:, 1].max() + margin

        res = 400
        xs = np.linspace(x_min, x_max, res)
        ys = np.linspace(y_min, y_max, res)
        xx, yy = np.meshgrid(xs, ys)

        pts = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
        )
        out = F.relu(pts @ W + b)  # (res², N)

        # Most-activated feature; mark dead regions (all-zero) as NaN
        max_out = out.max(dim=1).values
        dominant_flat = out.argmax(dim=1).float()
        dominant_flat[max_out == 0] = float("nan")
        dominant = dominant_flat.numpy().reshape(res, res)

    # Step colorscale: feature i covers [i/N, (i+1)/N] of [0, 1]
    colorscale = []
    for i in range(N):
        colorscale += [
            [i / N, pal[i % len(pal)]],
            [(i + 1) / N, pal[i % len(pal)]],
        ]

    fig = go.Figure()

    # Background: dominant-feature heatmap
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=dominant,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=N - 0.5,
            opacity=0.35,
            showscale=False,
        )
    )

    # Overlay: h1 = We_i (before Z) as circles, Zh1 (after Z) as stars
    for s in range(N):
        color = pal[s % len(pal)]

        # h1 = e @ W.T  — circle
        fig.add_trace(
            go.Scatter(
                x=[onehot_h1[s, 0]],
                y=[onehot_h1[s, 1]],
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

        # Zh1 = h1 @ Z.T  — star
        fig.add_trace(
            go.Scatter(
                x=[onehot_Zh[s, 0]],
                y=[onehot_Zh[s, 1]],
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

        # Arrow from h1 to Zh1
        fig.add_annotation(
            x=onehot_Zh[s, 0],
            y=onehot_Zh[s, 1],
            ax=onehot_h1[s, 0],
            ay=onehot_h1[s, 1],
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
        xaxis_title="dim 0  (Zh space)",
        yaxis_title="dim 1  (Zh space)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600,
    )
    fig.show()


# %%
N, k = 7, 2
T = 20_000
N_EPOCHS = 25_000
seed = 4

# P = make_transition_matrix(N, sparsity=0.5 / N, seed=1)
P = (
    0.1 * torch.eye(N)
    + 0.2 * torch.roll(torch.eye(N), shifts=1, dims=1)
    + 0.7 * torch.roll(torch.eye(N), shifts=2, dims=1)
)

px.imshow(P.numpy())


# %%
states = sample_markov_chain(P, T, seed=seed)
X_all = F.one_hot(states[:-1], N).float()
Y_idx = states[1:]  # integer targets — for CE
Y_oh = F.one_hot(Y_idx, N).float()  # one-hot targets  — for MSE

# ─── Train: Cross-entropy ──────────────────────────────────────────────────────

# %%
print("=== Cross-Entropy Loss ===")
model_ce, P_ce, losses_ce = train(
    X_all, Y_oh, N=N, k=k, h=5, loss_type="ce", epochs=N_EPOCHS, seed=seed
)

# ─── Train: MSE ───────────────────────────────────────────────────────────────

# %%
print("\n=== MSE Loss ===")
model_mse, P_mse, losses_mse = train(
    X_all, Y_oh, N=N, k=k, loss_type="mse", epochs=N_EPOCHS, seed=seed
)

# ─── Evaluation ───────────────────────────────────────────────────────────────

# %%
with torch.no_grad():
    ce_ce = F.cross_entropy(model_ce(X_all), Y_idx).item()
    ce_mse = F.cross_entropy(model_mse(X_all), Y_idx).item()
    mse_ce = F.mse_loss(model_ce(X_all), Y_oh).item()
    mse_mse = F.mse_loss(model_mse(X_all), Y_oh).item()

print(f"{'':12s} {'CE loss':>10s}  {'MSE loss':>10s}")
print(f"{'CE  model':12s} {ce_ce:10.4f}  {mse_ce:10.4f}")
print(f"{'MSE model':12s} {ce_mse:10.4f}  {mse_mse:10.4f}")


# %%
plot_transition_matrices(P, P_ce, P_mse, N)

# %%
plot_results(
    model_ce, losses_ce, N, title="Cross-Entropy Model", loss_label="Cross-Entropy"
)

# %%
plot_results(model_mse, losses_mse, N, title="MSE Model", loss_label="MSE")

# %%
plot_decode_plane(model_ce, N, title="CE model — decode plane: ReLU(h·W + b)")

# %%
plot_decode_plane(model_mse, N, title="MSE model — decode plane: ReLU(h·W + b)")


# %%
# px.imshow((model_ce.W.T @ model_ce.Z @ model_ce.W).detach().numpy())

# # %%
# px.imshow((model_mse.W.T @ model_mse.Z @ model_mse.W).detach().numpy())

# %%
print(model_ce.W)
print(model_ce.Z)
print(model_ce.b)

# %%
print(model_mse.W)
print(model_mse.Z)
print(model_mse.b)

# %%


# %%
