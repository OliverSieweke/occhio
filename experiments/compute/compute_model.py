# %%
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio.distributions import SparseUniform
from occhio.autoencoder import ComputeAutoEncoder
from occhio import ToyModel
from occhio.visualization import plot_decode_plane
import numpy as np


# %%
def make_transition_matrix(N, sparsity=0.3, seed=42):
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.dirichlet(np.ones(N) * sparsity, size=N), dtype=torch.float32
    )


# %%
N = 7
k = 2
p_active = 0.2

P = 0.1 * torch.eye(N) + 0.9 * torch.roll(torch.eye(N), shifts=1, dims=1)

sparse_dist = SparseUniform(N, p_active=p_active)
ae_sparse = ComputeAutoEncoder(
    N,
    k,
    decode_activation="relu",
    loss_fn=lambda raw, x_hat, imp: ae_sparse.mse_loss(x_hat, raw @ P, imp),
)
tm_sparse = ToyModel(sparse_dist, ae_sparse, importances=0.9 ** torch.arange(N))
losses_sparse, _ = tm_sparse.fit(
    n_epochs=25_000,
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
plot_decode_plane(ae_sparse)

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
