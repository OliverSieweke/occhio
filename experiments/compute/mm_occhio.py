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
import torch.nn.functional as F
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# occhio core abstractions
from occhio import ToyModel
from occhio.distributions import Distribution
from occhio.autoencoder import ComputeAutoEncoder
from occhio.model_grid import ModelGrid, Axis
from occhio.visualization import plot_embedding, plot_dynamic_scatter, plot_decode_plane


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
losses_ce, hooks_ce = tm_ce.fit(
    n_epochs=15000,
    loss_fn=lambda raw, x_hat, imp: ae_ce.ce_loss(x_hat, raw[1], imp),
    hooks=[lambda d: (d["epoch"], d["tm"].W)],
    hook_freq=250,
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
losses_mse, hooks_mse = tm_mse.fit(
    n_epochs=15000,
    loss_fn=lambda raw, x_hat, imp: ae_mse.mse_loss(x_hat, raw[2], imp),
    hooks=[lambda d: (d["epoch"], d["tm"].W)],
    hook_freq=250,
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
plot_decode_plane(ae_ce, title="CE model  — decode plane").show()
plot_decode_plane(ae_mse, title="MSE model — decode plane").show()

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
_models = np.empty(2, dtype=object)
_models[0] = tm_ce
_models[1] = tm_mse
mg = ModelGrid(
    create_model=lambda params: params,
    axes=[Axis(label="Loss", values=[0, 1])],
    cache_samples=False,
    _models=_models,
)
plot_embedding(mg).show()

# %%
plot_dynamic_scatter(losses_ce, hooks_ce[0], loss_stride=10).show()
plot_dynamic_scatter(losses_mse, hooks_mse[0], loss_stride=10).show()

# %%
