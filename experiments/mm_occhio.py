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
from occhio import ToyModel as OcchioToyModel
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

    def sample(self, batch_size: int) -> torch.Tensor:
        """Returns (B, N) one-hot current states — satisfies Distribution interface."""
        idx = torch.randint(len(self._x_idx), (batch_size,), generator=self.generator)
        return F.one_hot(self._x_idx[idx], self.n_features).float()

    def sample_pairs(self, batch_size: int, generator=None):
        """Returns (x_oh, y_idx, y_oh): current one-hot, next index, next one-hot."""
        idx = torch.randint(len(self._x_idx), (batch_size,), generator=generator)
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
        return h @ self.Z.T

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


# ─── MarkovToyModel  (thin wrapper matching occhio's ToyModel API) ─────────────


# %%
class MarkovToyModel:
    """
    Wraps MarkovChainDistribution + ComputeAutoEncoder in an occhio-compatible
    interface so we can call the same geometric analysis properties
    (W, feature_norms, interferences, …) that ToyModel exposes.

    Also exposes fit() which runs the prediction-task training loop.
    """

    def __init__(
        self,
        distribution: MarkovChainDistribution,
        ae: ComputeAutoEncoder,
    ):
        self.distribution = distribution
        self.ae = ae
        self.n_features = ae.n_features
        self.importances = 0.9 ** torch.arange(ae.n_features)

    def fit(
        self,
        loss_type: Literal["ce", "mse"] = "ce",
        epochs: int = 20_000,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        batch_size: int = 1024,
        seed: int = 42,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Train the model and return (P_learned, losses).

        P_learned : (N, N) learned transition matrix (one forward pass per state)
        losses    : per-epoch scalar losses
        """
        N = self.ae.n_features
        importances = self.importances.to(self.ae.device or "cpu")
        optimizer = torch.optim.AdamW(
            self.ae.parameters(), lr=lr, weight_decay=weight_decay
        )
        rng = torch.Generator().manual_seed(seed)

        losses = []
        for epoch in range(epochs):
            x_oh, y_idx, y_oh = self.distribution.sample_pairs(
                batch_size, generator=rng
            )
            y_hat, _ = self.ae(x_oh)

            if loss_type == "ce":
                loss = self.ae.ce_loss(y_hat, y_idx, importances)
            else:
                loss = self.ae.mse_loss(y_hat, y_oh, importances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if verbose and (epoch + 1) % 5_000 == 0:
                tag = "CE" if loss_type == "ce" else "MSE"
                print(f"  Epoch {epoch + 1:5d} | {tag} loss: {loss.item():.4f}")

        with torch.no_grad():
            P_learned = torch.zeros(N, N)
            for s in range(N):
                e = F.one_hot(torch.tensor([s]), N).float()
                P_learned[s] = self.ae(e)[0][0]

        return P_learned, losses

    # ── geometric analysis (mirrors ToyModel) ──────────────────────────────

    @property
    @torch.no_grad()
    def W(self) -> torch.Tensor:
        """(k, N) encoder weight matrix — identical to occhio ToyModel.W."""
        return self.ae.W

    @property
    @torch.no_grad()
    def W_normalized(self) -> torch.Tensor:
        return F.normalize(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.W, dim=0)

    @property
    @torch.no_grad()
    def feature_representations(self) -> torch.Tensor:
        return (self.W**2).sum(dim=0)

    @property
    @torch.no_grad()
    def interferences(self) -> torch.Tensor:
        return (self.W_normalized.T @ self.W) ** 2

    @property
    @torch.no_grad()
    def total_feature_interferences(self) -> torch.Tensor:
        I = self.interferences.clone()
        return I.fill_diagonal_(0).sum(dim=1)

    @property
    @torch.no_grad()
    def feature_dimensionalities(self) -> torch.Tensor:
        return self.feature_representations / self.interferences.sum(dim=1)


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
    tm_ce: MarkovToyModel,
    tm_mse: MarkovToyModel,
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
def plot_geometry(tm_ce: MarkovToyModel, tm_mse: MarkovToyModel):
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
tm_ce = MarkovToyModel(dist, ae_ce)
P_ce, losses_ce = tm_ce.fit(loss_type="ce", epochs=25_000, seed=seed)

# ─── Train MSE model ──────────────────────────────────────────────────────────

# %%
print("\n=== MSE Loss ===")
ae_mse = ComputeAutoEncoder(N, k, seed=seed)
tm_mse = MarkovToyModel(dist, ae_mse)
P_mse, losses_mse = tm_mse.fit(loss_type="mse", epochs=25_000, seed=seed)

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
# ─── Sparse Markov Variant (SparseUniform × occhio ToyModel) ──────────────────
#
# Instead of sampling one-hot states from a Markov chain walk, we:
#   1. Sample a sparse input  x ~ SparseUniform(N, p_active)   (multiple features on)
#   2. Compute the prediction target  y = x @ P                (linear mix of P rows)
#   3. Train the model to predict y from x with MSE loss
#
# This is a natural generalisation of the one-hot Markov task: a sparse x is a
# "soft" mixture of states, and y is the correspondingly mixed next-state dist.
# Using SparseUniform means the model sees richer superposition structure
# during training — closer to the standard occhio setup.
#
# Since targets are continuous (not one-hot), we use ComputeAutoEncoder with
# decode_activation="relu" (outputs ≥ 0) and MSE via the base-class loss.


# %%
class SparseMarkovToyModel(OcchioToyModel):
    """
    Extends occhio's ToyModel to learn the Markov prediction task on sparse inputs.

    Training target: y = x @ P  instead of  x  (reconstruction).

    A sparse input x represents a soft mixture of states; multiplying by P
    gives the expected next-state distribution under that mixture.  The model
    therefore must learn to "store" all rows of P in its 2-D embedding weight W.

    Uses ComputeAutoEncoder with decode_activation="relu" so outputs are ≥ 0,
    matching the non-simplex targets.  The geometric analysis properties
    inherited from ToyModel (W, feature_norms, interferences, …) work unchanged.
    """

    def __init__(self, P: torch.Tensor, distribution, ae, **kwargs):
        super().__init__(distribution=distribution, ae=ae, **kwargs)
        self.P = P

    def fit(
        self,
        n_epochs: int,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        track_losses: bool = True,
        optimizer=None,
        hooks: list = [],
        hook_freq: int = 1,
        verbose: bool = True,
    ) -> tuple[list[float], list]:
        """
        Train the model to predict y = x @ P from sparse inputs x.

        Drop-in replacement for ToyModel.fit — same signature and return type.
        Use the .P_learned property after training to read off the learned matrix.
        """
        from torch.optim import AdamW

        if optimizer is None:
            optimizer = AdamW(
                self.ae.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        losses = []
        hook_returns = [[] for _ in hooks]

        for ep in range(n_epochs):
            x = self.distribution.sample(batch_size)
            y = x @ self.P  # (B, N) — expected next-state mixture
            y_hat = self.ae.forward(x)[0]  # (B, N) — ReLU prediction
            loss = self.ae.loss(y, y_hat, self.importances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if track_losses:
                losses.append(loss.item())
            if verbose and (ep + 1) % 5_000 == 0:
                print(f"  Epoch {ep + 1:5d} | MSE loss: {loss.item():.5f}")
            if hooks and (ep % hook_freq == 0 or ep == n_epochs - 1):
                with torch.no_grad():
                    hook_data = dict(
                        tm=self, epoch=ep, loss=loss.item(), x=x, x_hat=y_hat
                    )
                    for i, h in enumerate(hooks):
                        hook_returns[i].append(h(hook_data))

        return losses, hook_returns

    @property
    @torch.no_grad()
    def P_learned(self) -> torch.Tensor:
        """Learned transition matrix: model output for each one-hot basis vector."""
        N = self.n_features
        P = torch.zeros(N, N)
        for s in range(N):
            e = F.one_hot(torch.tensor([s]), N).float()
            P[s] = self.ae(e)[0][0]
        return P


# ── helper: occhio-style arrow plot (single model) ─────────────────────────────


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
p_active = 0.1

print("\n=== Sparse Markov (SparseUniform × ComputeAutoEncoder[relu] × MSE) ===")
sparse_dist = SparseUniform(N, p_active=p_active)
ae_sparse = ComputeAutoEncoder(N, k, decode_activation="relu")
tm_sparse = SparseMarkovToyModel(
    P=P,
    distribution=sparse_dist,
    ae=ae_sparse,
    importances=0.9 ** torch.arange(N),
)
losses_sparse, _ = tm_sparse.fit(n_epochs=25_000)
P_sparse = tm_sparse.P_learned

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
