# %% [markdown]
# # Correlated Features and Autoencoder Representations
#
# When features in the data have statistical dependencies -- parent-child
# hierarchies, causal DAG structure, or mutual exclusivity -- does the
# autoencoder learn representations that reflect those relationships?
#
# This notebook investigates that question in three parts:
#
# 1. **Hierarchical Pairs** -- parents and children with tunable coupling
# 2. **DAG Causal Structure** -- Erdos-Renyi graph with Bayesian propagation
# 3. **Correlated vs Anticorrelated** -- co-occurring vs mutually exclusive
#
# Throughout, we use a TiedLinearRelu autoencoder in a bottleneck
# configuration (more features than hidden dims) and examine how the
# learned weight geometry (W^T W, cosine similarities) relates to the
# statistical structure of the input.
#
# Requires: `pip install occhio`

# %%
# -- Imports -----------------------------------------------------------------

import torch
import numpy as np
import plotly.graph_objects as go

from occhio import ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions.correlated import (
    HierarchicalPairs,
    CorrelatedPairs,
    AnticorrelatedPairs,
)
from occhio.distributions.dag import DAGBayesianPropagation

torch.manual_seed(0)

# %%
# -- Shared helpers ----------------------------------------------------------

N_FEATURES = 8
N_HIDDEN = 4
P_ACTIVE = 0.05
N_EPOCHS = 5000
BATCH_SIZE = 512
LR = 1e-3


def train_model(dist):
    """Train a TiedLinearRelu on the given distribution and return the model."""
    ae = TiedLinearRelu(n_features=N_FEATURES, n_hidden=N_HIDDEN, device="cpu")
    model = ToyModel(dist, ae, device="cpu")
    losses, _ = model.fit(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    return model, losses


def cosine_heatmap(matrix, title, labels=None, zmin=-1, zmax=1):  # noqa: ANN001
    """Create a heatmap figure from a square matrix."""
    n = matrix.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.numpy(),
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            text=np.round(matrix.numpy(), 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        title=title,
        height=450,
        width=500,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# %% [markdown]
# ---
# ## Part 1: Hierarchical Pairs
#
# `HierarchicalPairs` organizes features into parent-child pairs:
# (0,1), (2,3), (4,5), (6,7). The parent activates with probability
# `p_active`, and the child fires only when the parent is active
# (with probability `p_follow`).
#
# The `beta` parameter controls magnitude coupling: at `beta=1.0` the
# child copies the parent's value exactly; at `beta=0.0` the child gets
# an independent magnitude scaled by the parent.
#
# **Question:** As beta increases (tighter value coupling), do parent-child
# pairs converge to more similar embedding directions?

# %%
# -- Sweep beta and measure parent-child cosine similarities ----------------

betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
pair_cosines_by_beta = []  # mean parent-child cosine sim for each beta

for beta in betas:
    dist = HierarchicalPairs(
        n_features=N_FEATURES,
        p_active=P_ACTIVE,
        p_follow=0.8,
        beta=beta,
        device="cpu",
    )

    model, _ = train_model(dist)
    cos_mat = model.cosine_similarity_matrix

    # Extract parent-child cosine similarities: pairs (0,1), (2,3), ...
    n_pairs = N_FEATURES // 2
    pair_cos = [cos_mat[2 * i, 2 * i + 1].item() for i in range(n_pairs)]
    mean_pair_cos = np.mean(pair_cos)
    pair_cosines_by_beta.append(mean_pair_cos)

    print(
        f"beta={beta:.1f}  "
        f"pair cosines: [{', '.join(f'{c:.3f}' for c in pair_cos)}]  "
        f"mean={mean_pair_cos:.3f}"
    )

# %%
# -- Plot: parent-child cosine similarity vs beta --------------------------

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=betas,
        y=pair_cosines_by_beta,
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2),
        name="Mean |parent-child cos|",
    )
)
fig.update_layout(
    title="Parent-Child Cosine Similarity vs Magnitude Coupling (beta)",
    xaxis_title="beta (magnitude coupling)",
    yaxis_title="Mean parent-child cosine similarity",
    height=400,
    width=600,
)
fig.show()

# %% [markdown]
# **Interpretation:** When beta is high, the child's value closely tracks
# the parent's, making them statistically near-identical from the
# autoencoder's perspective. The network responds by aligning their
# embedding directions -- why waste separate dimensions on signals that
# always co-occur with similar magnitudes?

# %%
# -- Heatmaps at beta=0 and beta=1 for direct comparison -------------------

pair_labels = ["P0", "C0", "P1", "C1", "P2", "C2", "P3", "C3"]

for beta_val in [0.0, 1.0]:
    dist = HierarchicalPairs(
        n_features=N_FEATURES,
        p_active=P_ACTIVE,
        p_follow=0.8,
        beta=beta_val,
        device="cpu",
    )
    model, _ = train_model(dist)
    cos_mat = model.cosine_similarity_matrix

    fig = cosine_heatmap(
        cos_mat,
        title=f"Cosine Similarity (beta={beta_val:.1f})",
        labels=pair_labels,
    )
    fig.show()

# %% [markdown]
# In the `beta=1.0` heatmap, parent-child blocks along the diagonal
# should show strong positive cosine similarity -- the network has merged
# each pair into nearly the same direction. At `beta=0.0`, pairs are more
# independent and the structure is weaker.

# %% [markdown]
# ---
# ## Part 2: DAG Causal Structure
#
# `DAGBayesianPropagation` generates a random DAG via an Erdos-Renyi
# process and propagates activations using Noisy-OR: if any parent is
# active, the child fires with probability `1 - prod(1 - v_parent)`.
#
# **Question:** Does the interference pattern in the learned W^T W
# reflect the causal adjacency in the DAG? Features with a causal
# connection should statistically co-occur more, leading the autoencoder
# to place them in interfering (non-orthogonal) directions.

# %%
# -- Create a DAG with enough edges to see structure -----------------------

# Use a fixed seed for the DAG topology so results are reproducible
dag_gen = torch.Generator(device="cpu").manual_seed(12)

dag_dist = DAGBayesianPropagation(
    n_features=N_FEATURES,
    p_active=0.15,
    p_edge=0.35,
    device="cpu",
    generator=dag_gen,
)

# Print the graph structure
adj = dag_dist.adjacency.float()
print("DAG adjacency (i -> j means i causes j):")
print(adj.numpy().astype(int))
sources = [i for i in range(N_FEATURES) if adj[:, i].sum() == 0]
sinks = [i for i in range(N_FEATURES) if adj[i, :].sum() == 0]
print(f"Sources (root nodes): {sources}")
print(f"Sinks (leaf nodes): {sinks}")

# %%
# -- Visualize the ground-truth adjacency ----------------------------------

# Build the "reachability" matrix: which nodes can reach which via any path
# This captures indirect causal influence, not just direct edges
reach = adj.clone()
for _ in range(N_FEATURES):
    reach = (reach @ adj).clamp(max=1.0).max(reach)

print("\nDirect adjacency matrix:")
print(adj.numpy().astype(int))
print("\nReachability matrix (transitive closure):")
print(reach.numpy().astype(int))

# %%
# -- Train and compare cosine similarity to DAG structure ------------------

model_dag, losses_dag = train_model(dag_dist)

cos_mat = model_dag.cosine_similarity_matrix
wtw = model_dag.W_T_W

node_labels = [f"n{i}" for i in range(N_FEATURES)]

# Heatmap of learned cosine similarities
fig_cos = cosine_heatmap(
    cos_mat,
    title="Learned Cosine Similarities (DAG)",
    labels=node_labels,
)
fig_cos.show()

# Heatmap of W^T W (includes norm information)
fig_wtw = cosine_heatmap(
    wtw,
    title="W^T W Interference (DAG)",
    labels=node_labels,
    zmin=None,
    zmax=None,
)
fig_wtw.show()

# %%
# -- Quantify: do causally connected pairs have higher |cos similarity|? ---

cos_abs = cos_mat.abs()

# Symmetrize the adjacency for undirected comparison
adj_sym = (adj + adj.T).clamp(max=1.0)
reach_sym = (reach + reach.T).clamp(max=1.0)

# Collect cosine similarities for connected vs unconnected pairs
n = N_FEATURES
connected_direct = []
connected_reach = []
unconnected = []

for i in range(n):
    for j in range(i + 1, n):
        c = cos_abs[i, j].item()
        if adj_sym[i, j] > 0:
            connected_direct.append(c)
        if reach_sym[i, j] > 0:
            connected_reach.append(c)
        else:
            unconnected.append(c)

print("Mean |cosine similarity| for feature pairs:")
if connected_direct:
    print(
        f"  Directly connected:  {np.mean(connected_direct):.4f}"
        f"  (n={len(connected_direct)})"
    )
if connected_reach:
    print(
        f"  Reachable (any path): {np.mean(connected_reach):.4f}"
        f"  (n={len(connected_reach)})"
    )
if unconnected:
    print(f"  Unconnected:          {np.mean(unconnected):.4f}  (n={len(unconnected)})")

# %% [markdown]
# **Interpretation:** If the DAG has enough connectivity, causally linked
# features co-occur more frequently and the autoencoder gives them
# overlapping directions (higher cosine similarity). This is the network
# encoding statistical structure -- it does not "know" the DAG, but the
# co-occurrence patterns induced by the DAG leave a geometric fingerprint.
#
# Note: with only 8 features, results will vary by seed. The signal
# becomes clearer with more features and denser graphs.

# %% [markdown]
# ---
# ## Part 3: Correlated vs Anticorrelated Pairs
#
# **CorrelatedPairs:** features in each pair tend to fire together.
# A pair-level gate activates with probability `p_active`, then each
# member independently fires with `p_individual`.
#
# **AnticorrelatedPairs:** features in each pair are mutually exclusive.
# When a pair activates, exactly one of the two fires.
#
# **Question:** Do anticorrelated features get pushed to opposite
# directions (negative cosine similarity)? Do correlated features
# get pulled together?

# %%
# -- Train on CorrelatedPairs -----------------------------------------------

corr_dist = CorrelatedPairs(
    n_features=N_FEATURES,
    p_active=0.15,
    p_individual=0.9,
    device="cpu",
)

# Verify the correlation analytically
p_a = corr_dist.p_active[0].item()
p_i = corr_dist.p_individual[0].item()
analytic_corr = p_i * (1 - p_a) / (1 - p_a * p_i)
print(f"CorrelatedPairs:  p_active={p_a:.2f}, p_individual={p_i:.2f}")
print(f"  Analytic correlation within pairs: {analytic_corr:.4f}")

# Verify empirically
samples = corr_dist.sample(50_000)
active = (samples > 0).float()
for pair_idx in range(N_FEATURES // 2):
    i, j = 2 * pair_idx, 2 * pair_idx + 1
    empirical_corr = torch.corrcoef(active[:, [i, j]].T)[0, 1].item()
    if pair_idx == 0:
        print(f"  Empirical correlation (pair 0): {empirical_corr:.4f}")

model_corr, _ = train_model(corr_dist)
cos_corr = model_corr.cosine_similarity_matrix

# %%
# -- Train on AnticorrelatedPairs -------------------------------------------

anti_dist = AnticorrelatedPairs(
    n_features=N_FEATURES,
    p_active=0.15,
    device="cpu",
)

# Verify mutual exclusivity empirically
samples_anti = anti_dist.sample(50_000)
active_anti = (samples_anti > 0).float()
for pair_idx in range(N_FEATURES // 2):
    i, j = 2 * pair_idx, 2 * pair_idx + 1
    both_active = (active_anti[:, i] * active_anti[:, j]).sum().item()
    if pair_idx == 0:
        print(
            f"\nAnticorrelatedPairs: pair 0 both-active count "
            f"= {int(both_active)} / {len(samples_anti)}"
        )

model_anti, _ = train_model(anti_dist)
cos_anti = model_anti.cosine_similarity_matrix

# %%
# -- Side-by-side heatmaps -------------------------------------------------

for label, cos_mat in [
    ("Correlated Pairs", cos_corr),
    ("Anticorrelated Pairs", cos_anti),
]:
    fig = cosine_heatmap(
        cos_mat,
        title=f"Cosine Similarity -- {label}",
        labels=pair_labels,
    )
    fig.show()

# %%
# -- Compare within-pair cosine similarities directly ----------------------

n_pairs = N_FEATURES // 2
print(f"{'Pair':<8} {'Correlated':>12} {'Anticorrelated':>16}")
print("-" * 38)
for p in range(n_pairs):
    c_corr = cos_corr[2 * p, 2 * p + 1].item()
    c_anti = cos_anti[2 * p, 2 * p + 1].item()
    print(f"({2 * p},{2 * p + 1})   {c_corr:>+12.4f} {c_anti:>+16.4f}")

mean_corr = np.mean([cos_corr[2 * p, 2 * p + 1].item() for p in range(n_pairs)])
mean_anti = np.mean([cos_anti[2 * p, 2 * p + 1].item() for p in range(n_pairs)])
print(f"\n{'Mean':<8} {mean_corr:>+12.4f} {mean_anti:>+16.4f}")

# %% [markdown]
# **Interpretation:** Correlated features tend toward positive cosine
# similarity -- the autoencoder benefits from aligning features that
# co-occur, since their combined signal reinforces along a shared
# direction.
#
# Anticorrelated (mutually exclusive) features tend toward negative
# cosine similarity or orthogonality. Since they never co-occur, the
# network can safely place them in opposite directions. This is efficient:
# a single hidden dimension can encode "which one of the pair fired" via
# the sign of the activation.

# %% [markdown]
# ---
# ## Part 4: Pulling it together -- interference vs data correlation
#
# Across all three distribution types, a pattern emerges: the
# autoencoder's interference structure (W^T W) mirrors the statistical
# correlation structure of the data.
#
# Let's make this explicit by plotting the relationship between
# empirical feature correlation and learned cosine similarity across
# all feature pairs.

# %%
# -- Scatter: empirical correlation vs learned cosine similarity -----------

fig = go.Figure()

configs = [
    (
        "Hierarchical (beta=0.8)",
        HierarchicalPairs(
            n_features=N_FEATURES,
            p_active=P_ACTIVE,
            p_follow=0.8,
            beta=0.8,
            device="cpu",
        ),
    ),
    ("Correlated", corr_dist),
    ("Anticorrelated", anti_dist),
]

for label, dist in configs:
    # Empirical correlation matrix from samples
    samps = dist.sample(50_000)
    active = (samps > 0).float()
    emp_corr = torch.corrcoef(active.T)

    # Train model
    model, _ = train_model(dist)
    cos_mat = model.cosine_similarity_matrix

    # Collect all unique pairs
    emp_vals = []
    cos_vals = []
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            r = emp_corr[i, j].item()
            c = cos_mat[i, j].item()
            if not np.isnan(r):
                emp_vals.append(r)
                cos_vals.append(c)

    fig.add_trace(
        go.Scatter(
            x=emp_vals,
            y=cos_vals,
            mode="markers",
            marker=dict(size=6, opacity=0.7),
            name=label,
        )
    )

fig.add_shape(
    type="line",
    x0=-1,
    x1=1,
    y0=-1,
    y1=1,
    line=dict(dash="dash", color="gray", width=1),
)
fig.update_layout(
    title="Data Correlation vs Learned Cosine Similarity",
    xaxis_title="Empirical feature correlation",
    yaxis_title="Learned cosine similarity",
    height=500,
    width=600,
    showlegend=True,
)
fig.show()

# %% [markdown]
# **Takeaway:** There is a clear relationship between input correlation
# and representation geometry. Positively correlated features get aligned
# (positive cosine similarity), negatively correlated features get
# opposed, and independent features stay relatively orthogonal. The
# autoencoder does not receive the correlation structure as an explicit
# signal -- it emerges from the MSE loss minimization alone.
#
# This has implications for interpretability: when we observe structure
# in an autoencoder's weight matrix, it may be reflecting statistical
# dependencies in the data, not just individual feature importance.
