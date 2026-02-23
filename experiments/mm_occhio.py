# %%
"""
TMoS Mini-LLM  ×  occhio library
=================================
Reimplements experiments/mm.py using occhio's Distribution and AutoEncoder
abstractions, then adds occhio-style geometric analysis.

Architecture:
  Encode:  h = x @ W.T              (N → k)
  Compute: z = h @ Z.T              (k → k, linear)
  Decode:  ŷ = softmax(z @ W + b)   (k → N, probability simplex)

Data:
  Markov chain over N states with transition matrix P.
  Each (x, y) pair is (one-hot current state, one-hot next state).

Two losses:
  1. Cross-entropy  (target = integer class index)
  2. MSE            (target = one-hot float vector)
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Literal

# occhio core abstractions
from occhio import ToyModel
from occhio.distributions import Distribution
from occhio.autoencoder import AutoEncoderBase
from occhio.distributions.sparse import SparseUniform


# ─── Distribution ─────────────────────────────────────────────────────────────


# %%
class MarkovChainDistribution(Distribution):
    """
    Distribution over one-hot states sampled from a Markov chain.

    Inherits from occhio's Distribution so it can be plugged into ToyModel
    for analysis, but adds sample_pairs() for the prediction task.

    Args:
        P:       (N, N) row-stochastic transition matrix.
        seq_len: Pre-generate a Markov sequence of this length at init time.
        seed:    Seed for the chain walk (numpy RNG).
    """

    def __init__(
        self,
        P: torch.Tensor,
        seq_len: int = 50_000,
        seed: int = 0,
        **kwargs,
    ):
        N = P.shape[0]
        super().__init__(n_features=N, **kwargs)

        self.P = P
        self._build_sequence(seq_len, seed)

    def _build_sequence(self, seq_len: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        P_np = self.P.cpu().numpy().astype(np.float64)
        N = self.n_features
        states = [int(rng.integers(N))]
        for _ in range(seq_len - 1):
            p = P_np[states[-1]]
            p = p / p.sum()
            states.append(int(rng.choice(N, p=p)))
        seq = torch.tensor(states, dtype=torch.long)
        self._x_idx = seq[:-1]  # current state indices
        self._y_idx = seq[1:]  # next    state indices

    def sample(self, batch_size: int):
        """Returns (x_oh, y_idx, y_oh): current one-hot, next index, next one-hot."""
        idx = torch.randint(len(self._x_idx), (batch_size,), generator=self.generator)
        x_oh = F.one_hot(self._x_idx[idx], self.n_features).float()
        y_i = self._y_idx[idx]
        y_oh = F.one_hot(y_i, self.n_features).float()
        return x_oh, y_i, y_oh

    def to(self, device):
        super().to(device)
        self.P = self.P.to(device)
        self._x_idx = self._x_idx.to(device)
        self._y_idx = self._y_idx.to(device)
        return self


# ─── AutoEncoder ──────────────────────────────────────────────────────────────


# %%
class ComputeAutoEncoder(AutoEncoderBase):
    """
    Autoencoder with a tied encoder/decoder and a linear compute step.

    Subclasses occhio's AutoEncoderBase so it exposes encode/decode and slots
    into ToyModel for geometric analysis (feature norms, interferences, etc.).

    Parameters
    ----------
    N : int   — number of features
    k : int   — hidden / latent dimension
    decode_activation : "softmax" | "relu"
        "softmax" — outputs a probability simplex; use for one-hot targets (CE/MSE).
        "relu"    — outputs non-negative values; use for continuous targets like x @ P.
    seed : int — weight init seed

    Weights
    -------
    W : (k, N) — tied encoder / decoder
    Z : (k, k) — linear compute step
    b : (N,)   — decode bias
    """

    def __init__(
        self,
        N: int,
        k: int,
        decode_activation: Literal["softmax", "relu"] = "softmax",
        seed: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_features = N
        self.n_hidden = k
        self.decode_activation = decode_activation

        gen = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(k, N, generator=gen) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen) / k)
        self.b = nn.Parameter(torch.zeros(N))

    # ── core operations ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, k)  : embed into latent space."""
        return x @ self.W.T

    def compute_step(self, h: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, k)  : linear compute / routing step."""
        return h + h @ self.Z.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, N)  : project back, then activate."""
        logits = z @ self.W + self.b
        if self.decode_activation == "softmax":
            return F.softmax(logits, dim=-1)
        return F.relu(logits)

    def forward(self, x: torch.Tensor):
        """(B, N) → (y_hat, z)."""
        h = self.encode(x)
        z = self.compute_step(h)
        y_hat = self.decode(z)
        return y_hat, z

    # ── losses ─────────────────────────────────────────────────────────────

    def ce_loss(
        self, y_hat: torch.Tensor, y_idx: torch.Tensor, importances: torch.Tensor
    ) -> torch.Tensor:
        """Importance-weighted NLL given softmax output probabilities."""
        per_sample = F.nll_loss(y_hat.clamp(min=1e-9).log(), y_idx, reduction="none")
        weights = importances[y_idx]
        return (per_sample * weights).mean()

    def mse_loss(
        self, y_hat: torch.Tensor, y_oh: torch.Tensor, importances: torch.Tensor
    ) -> torch.Tensor:
        """Importance-weighted MSE between predicted probs and one-hot target."""
        per_sample = (y_hat - y_oh).pow(2).sum(dim=-1)
        weights = importances[y_oh.argmax(dim=-1)]
        return (per_sample * weights).mean()

    def resample_weights(self):
        gen = self.generator or torch.Generator()
        N, k = self.n_features, self.n_hidden
        self.W = nn.Parameter(torch.randn(k, N, generator=gen) / N)
        self.Z = nn.Parameter(torch.randn(k, k, generator=gen) / k)
        self.b = nn.Parameter(torch.zeros(N))


# ─── Plotting ─────────────────────────────────────────────────────────────────


# %%
def plot_transition_matrices(P_true, P_ce, P_mse):
    """Heatmap comparison of true vs. learned transition matrices."""
    matrices = [
        (P_true.numpy(), "True P"),
        (P_ce.detach().numpy(), "CE Learned P"),
        (P_mse.detach().numpy(), "MSE Learned P"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in matrices])
    for col, (P, title) in enumerate(matrices, start=1):
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
    return fig


# %%
def plot_loss_and_embeddings(
    tm_ce: ToyModel,
    tm_mse: ToyModel,
    losses_ce: list[float],
    losses_mse: list[float],
):
    """
    2×2 grid: loss curves (left) + W-embedding arrows (right) for each model.
    """
    qual = px.colors.qualitative.Set1
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "CE Loss",
            "CE Embeddings W",
            "MSE Loss",
            "MSE Embeddings W",
        ],
    )

    for row, (tm, losses, label, lc) in enumerate(
        [
            (tm_ce, losses_ce, "CE", "#636EFA"),
            (tm_mse, losses_mse, "MSE", "#EF553B"),
        ],
        start=1,
    ):
        fig.add_trace(
            go.Scatter(
                y=losses,
                mode="lines",
                name=f"{label} Loss",
                line=dict(color=lc, width=1),
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text="Epoch", row=row, col=1)
        fig.update_yaxes(title_text=label, row=row, col=1)

        W = tm.W.detach().numpy()
        for s in range(tm.n_features):
            color = qual[s % len(qual)]
            fig.add_trace(
                go.Scatter(
                    x=[0, float(W[0, s])],
                    y=[0, float(W[1, s])],
                    mode="lines+markers+text",
                    marker=dict(size=[4, 10], color=color),
                    text=["", f"s{s}"],
                    textposition="top center",
                    name=f"s{s}",
                    line=dict(color=color, width=2.5),
                    showlegend=(row == 1),
                ),
                row=row,
                col=2,
            )
        # equal aspect ratio for embedding subplot
        xref = "x2" if row == 1 else "x4"
        fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=row, col=2)
        fig.update_xaxes(title_text="dim 0", row=row, col=2)
        fig.update_yaxes(title_text="dim 1", row=row, col=2)

    fig.update_layout(title_text="Training Results", height=800)
    return fig


# %%
def plot_decode_plane(ae: ComputeAutoEncoder, N: int, title: str):
    """
    Grid the 2-D latent space, decode every point with ReLU (for clean regions),
    colour by dominant feature, and overlay where each one-hot e_i maps after
    encode → compute_step.
    """
    pal = px.colors.qualitative.Set1

    with torch.no_grad():
        W = ae.W  # (k, N)
        b = ae.b  # (N,)

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

        # ReLU decode for clean Voronoi-like regions
        out = F.relu(pts @ W + b)  # (res², N)
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


# %%
def plot_geometry(tm_ce: ToyModel, tm_mse: ToyModel):
    """
    Occhio-inspired geometric analysis: feature norms and cross-feature
    interferences for both models, displayed as bar charts.
    """
    N = tm_ce.n_features
    states = [f"s{i}" for i in range(N)]
    colors = px.colors.qualitative.Set1

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "CE — Feature Norms ‖w_s‖₂",
            "MSE — Feature Norms ‖w_s‖₂",
            "CE — Total Cross-Interference",
            "MSE — Total Cross-Interference",
        ],
    )

    for col, tm, label in [(1, tm_ce, "CE"), (2, tm_mse, "MSE")]:
        norms = tm.feature_norms.numpy()
        interf = tm.total_feature_interferences.detach().numpy()

        fig.add_trace(
            go.Bar(
                x=states,
                y=norms,
                marker_color=[colors[i % len(colors)] for i in range(N)],
                name=f"{label} norms",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                x=states,
                y=interf,
                marker_color=[colors[i % len(colors)] for i in range(N)],
                name=f"{label} interference",
                showlegend=False,
            ),
            row=2,
            col=col,
        )

    fig.update_yaxes(title_text="‖w_s‖₂", row=1, col=1)
    fig.update_yaxes(title_text="‖w_s‖₂", row=1, col=2)
    fig.update_yaxes(title_text="Σ interference", row=2, col=1)
    fig.update_yaxes(title_text="Σ interference", row=2, col=2)
    fig.update_layout(title_text="Geometric Analysis (occhio-style)", height=600)
    return fig


# ─── Experiment setup ─────────────────────────────────────────────────────────

# %%
N, k = 7, 2
T = 50_000
seed = 5

# Cyclic-shift transition matrix (same as mm.py)
P = 0.1 * torch.eye(N) + 0.9 * torch.roll(torch.eye(N), shifts=1, dims=1)
px.imshow(
    P.numpy(), title="True Transition Matrix P", color_continuous_scale="Blues"
).show()

# %%
dist = MarkovChainDistribution(P, seq_len=T, seed=seed)

# ─── Train CE model ───────────────────────────────────────────────────────────

# %%
print("=== Cross-Entropy Loss ===")
ae_ce = ComputeAutoEncoder(N, k, seed=seed)
tm_ce = ToyModel(dist, ae_ce, importances=0.9 ** torch.arange(N))
losses_ce, _ = tm_ce.fit(
    n_epochs=25_000,
    loss_fn=lambda raw, x_hat, imp: ae_ce.ce_loss(x_hat, raw[1], imp),
    verbose=True,
)
with torch.no_grad():
    P_ce = torch.zeros(N, N)
    for s in range(N):
        e = F.one_hot(torch.tensor([s]), N).float()
        P_ce[s] = ae_ce(e)[0][0]

# ─── Train MSE model ──────────────────────────────────────────────────────────

# %%
print("\n=== MSE Loss ===")
ae_mse = ComputeAutoEncoder(N, k, seed=seed)
tm_mse = ToyModel(dist, ae_mse, importances=0.9 ** torch.arange(N))
losses_mse, _ = tm_mse.fit(
    n_epochs=25_000,
    loss_fn=lambda raw, x_hat, imp: ae_mse.mse_loss(x_hat, raw[2], imp),
    verbose=True,
)
with torch.no_grad():
    P_mse = torch.zeros(N, N)
    for s in range(N):
        e = F.one_hot(torch.tensor([s]), N).float()
        P_mse[s] = ae_mse(e)[0][0]

# ─── Evaluation ───────────────────────────────────────────────────────────────

# %%
x_oh_all = F.one_hot(dist._x_idx, N).float()
y_idx_all = dist._y_idx
y_oh_all = F.one_hot(y_idx_all, N).float()

with torch.no_grad():
    p_ce_out = ae_ce(x_oh_all)[0]
    p_mse_out = ae_mse(x_oh_all)[0]

    ce_ce = F.nll_loss(p_ce_out.clamp(1e-9).log(), y_idx_all).item()
    ce_mse = F.nll_loss(p_mse_out.clamp(1e-9).log(), y_idx_all).item()
    mse_ce = F.mse_loss(p_ce_out, y_oh_all).item()
    mse_mse = F.mse_loss(p_mse_out, y_oh_all).item()

print(f"{'':12s} {'CE loss':>10s}  {'MSE loss':>10s}")
print(f"{'CE  model':12s} {ce_ce:10.4f}  {mse_ce:10.4f}")
print(f"{'MSE model':12s} {ce_mse:10.4f}  {mse_mse:10.4f}")

# ─── Plots ────────────────────────────────────────────────────────────────────

# %%
plot_transition_matrices(P, P_ce, P_mse).show()

# %%
plot_loss_and_embeddings(tm_ce, tm_mse, losses_ce, losses_mse).show()


# %%
plot_decode_plane(ae_ce, N, title="CE model  — decode plane (Voronoi regions)").show()
plot_decode_plane(ae_mse, N, title="MSE model — decode plane (Voronoi regions)").show()

# %%
# Occhio-style geometric analysis
plot_geometry(tm_ce, tm_mse).show()

# %%
# W.T @ Z @ W  :  how does the compute step look from feature-space perspective?
fig = make_subplots(rows=1, cols=2, subplot_titles=["CE — W.T Z W", "MSE — W.T Z W"])
for col, ae in enumerate([ae_ce, ae_mse], start=1):
    M = (ae.W.T @ ae.Z @ ae.W).detach().numpy()
    fig.add_trace(
        go.Heatmap(
            z=M,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in M],
            texttemplate="%{text}",
            showscale=(col == 2),
        ),
        row=1,
        col=col,
    )
fig.update_layout(title_text="Compute-step projection  W.T @ Z @ W", height=420)
fig.show()


# %%


def plot_embedding_arrows(tm, title: str = "Feature Embeddings W"):
    """
    Occhio-style embedding plot for a single model.

    Arrows from origin to W[:, s], coloured by importance via Plasma_r
    (matches the aesthetic of occhio's plot_embedding from visualization/).
    Works with any model that exposes .W (k×N), .n_features, .importances.
    """
    colorscale = px.colors.sequential.Plasma_r
    W = tm.W.detach().numpy()
    N = tm.n_features
    imps = (
        tm.importances.detach().numpy()
        if isinstance(tm.importances, torch.Tensor)
        else tm.importances
    )

    fig = go.Figure()
    for s in range(N):
        imp = float(imps[s]) * 0.9
        color_idx = int(imp * (len(colorscale) - 1))
        color = colorscale[color_idx]

        fig.add_annotation(
            x=float(W[0, s]),
            y=float(W[1, s]),
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2.5,
            arrowcolor=color,
            opacity=0.85,
            text=f"s{s}",
            font=dict(size=11),
        )

    fig.update_layout(
        title=title,
        plot_bgcolor="#FCFBF8",
        xaxis=dict(title="dim 0", showgrid=False, zeroline=True),
        yaxis=dict(
            title="dim 1", showgrid=False, zeroline=True, scaleanchor="x", scaleratio=1
        ),
        showlegend=False,
        height=500,
    )
    return fig


# ─── Train sparse model ───────────────────────────────────────────────────────

# %%
p_active = 1.0

sparse_dist = SparseUniform(N, p_active=p_active)
ae_sparse = ComputeAutoEncoder(N, k, decode_activation="relu")
tm_sparse = ToyModel(sparse_dist, ae_sparse, importances=0.9 ** torch.arange(N))
losses_sparse, _ = tm_sparse.fit(
    n_epochs=25_000,
    loss_fn=lambda raw, x_hat, imp: ae_sparse.loss(raw @ P, x_hat, imp),
    verbose=True,
)
with torch.no_grad():
    P_sparse = torch.zeros(N, N)
    for s in range(N):
        e = F.one_hot(torch.tensor([s]), N).float()
        P_sparse[s] = ae_sparse(e)[0][0]

# ─── Evaluation ──────────────────────────────────────────────────────────────

# %%
# Evaluate on all one-hot inputs: predicted P vs true P
with torch.no_grad():
    eye = torch.eye(N)
    P_sparse_pred = ae_sparse(eye)[0]  # model's full transition matrix
    mse_sparse = F.mse_loss(P_sparse_pred, P).item()
    print(f"\n  MSE(P_learned, P_true) = {mse_sparse:.5f}")

# ─── Plots ────────────────────────────────────────────────────────────────────

# %%
# Transition matrix: true vs sparse-learned
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["True P", "Sparse MSE Learned P"],
)
for col, (M, _) in enumerate(
    [(P.numpy(), "True P"), (P_sparse.detach().numpy(), "Sparse")], start=1
):
    fig.add_trace(
        go.Heatmap(
            z=M,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in M],
            texttemplate="%{text}",
            showscale=(col == 2),
        ),
        row=1,
        col=col,
    )
    fig.update_xaxes(title_text="Next state", row=1, col=col)
    fig.update_yaxes(title_text="Current state", row=1, col=col)
fig.update_layout(title_text="Sparse Markov — Transition Matrices", height=420)
fig.show()

# %%
# Loss curve
px.line(
    y=losses_sparse,
    title="Sparse Markov — MSE Loss",
    labels={"y": "MSE loss", "index": "Epoch"},
).show()

# %%
# Occhio-style arrow embedding plot (Plasma_r = importance rank)
plot_embedding_arrows(
    tm_sparse,
    title="Sparse Markov — Feature Embeddings W  (Plasma = importance)",
).show()

# %%
# Decode plane — encode positions (circles) and after compute step (stars)
plot_decode_plane(
    ae_sparse,
    N,
    title="Sparse Markov — Decode Plane (ReLU regions)",
).show()

# %%
# Geometry: feature norms and interferences (inherited from occhio ToyModel)
states = [f"s{i}" for i in range(N)]
colors = px.colors.qualitative.Set1
norms = tm_sparse.feature_norms.numpy()
interf = tm_sparse.total_feature_interferences.detach().numpy()

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Feature Norms ‖w_s‖₂", "Total Cross-Interference"],
)
fig.add_trace(
    go.Bar(
        x=states,
        y=norms,
        marker_color=[colors[i % len(colors)] for i in range(N)],
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=states,
        y=interf,
        marker_color=[colors[i % len(colors)] for i in range(N)],
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.update_yaxes(title_text="‖w_s‖₂", row=1, col=1)
fig.update_yaxes(title_text="Σ interference", row=1, col=2)
fig.update_layout(
    title_text="Sparse Markov — Geometric Analysis (occhio-style)", height=400
)
fig.show()

# %%
