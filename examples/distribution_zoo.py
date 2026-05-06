# %% [markdown]
# # The Distribution Zoo
#
# A tour of occhio's exotic distribution types -- the ones that go far beyond
# independent sparse features. Each distribution imposes a different geometric
# structure on the data: curved manifolds, periodic grids, compositional
# simplices, correlated pairs. The question we explore throughout: **how does
# the structure of your data determine the structure of the learned
# representation?**
#
# For each distribution we:
# 1. Sample data and inspect its statistical fingerprint
# 2. Train a TiedLinearRelu autoencoder
# 3. Examine the learned feature geometry (W matrix, norms, cosine similarities)
# 4. Visualize the embedding space
#
# Requirements: occhio, torch, plotly, numpy

# %% Imports
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from occhio import ToyModel
from occhio.autoencoders import TiedLinearRelu
from occhio.distributions import (
    SphericalDistribution,
    TorusDistribution,
    SimplexDistribution,
    SimplicialComplexDistribution,
    CorrelatedPairs,
)

# %% Shared utilities

torch.manual_seed(42)
DEVICE = "cpu"


def data_fingerprint(dist, name, n_samples=4096):
    """Print basic statistics about a distribution's samples."""
    x = dist.sample(n_samples)
    sparsity = (x == 0).float().mean().item()
    mean_active = (x > 0).float().sum(dim=1).mean().item()
    corr = torch.corrcoef(x.T)
    # Off-diagonal correlations (ignoring self-correlations)
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool)
    off_diag = corr[mask]

    print(f"--- {name} ---")
    print(f"  Shape:            {list(x.shape)}")
    print(f"  Sparsity:         {sparsity:.3f}")
    print(f"  Mean active:      {mean_active:.1f} / {dist.n_features}")
    print(f"  Correlation range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
    print(f"  Mean |corr|:      {off_diag.abs().mean():.3f}")
    print()
    return x


def train_and_report(dist, n_features, n_hidden, n_epochs=3000, lr=3e-4):
    """Train a TiedLinearRelu AE and return the ToyModel."""
    gen = torch.Generator(DEVICE)
    gen.manual_seed(42)

    ae = TiedLinearRelu(n_features, n_hidden, device=DEVICE, generator=gen)
    tm = ToyModel(dist, ae, device=DEVICE)
    losses, _ = tm.fit(n_epochs, learning_rate=lr, batch_size=1024)

    final_loss = losses[-1]
    norms = tm.feature_norms
    superposition = tm.superposition.item()

    print(f"  Final loss:       {final_loss:.6f}")
    print(f"  Feature norms:    [{norms.min():.3f}, {norms.max():.3f}]")
    print(f"  Superposition:    {superposition:.3f}")
    print()
    return tm, losses


def plot_embedding_2d(tm, title, feature_labels=None):
    """Scatter plot of feature embeddings in 2D hidden space."""
    W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)
    n_feat = W.shape[1]
    if feature_labels is None:
        feature_labels = [f"f{i}" for i in range(n_feat)]

    norms = np.linalg.norm(W, axis=0)

    fig = go.Figure()

    # Feature embedding vectors as arrows from origin
    for i in range(n_feat):
        fig.add_trace(
            go.Scatter(
                x=[0, W[0, i]],
                y=[0, W[1, i]],
                mode="lines+markers",
                marker=dict(
                    size=[3, 8],
                    color=norms[i],
                    colorscale="Viridis",
                    cmin=0,
                    cmax=norms.max(),
                ),
                line=dict(width=2),
                name=feature_labels[i],
                hovertext=f"{feature_labels[i]}: norm={norms[i]:.3f}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="h0",
        yaxis_title="h1",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=500,
        height=500,
        showlegend=True,
    )
    fig.show()
    return fig


def plot_cosine_heatmap(tm, title, feature_labels=None):
    """Heatmap of pairwise cosine similarities between feature embeddings."""
    cos_sim = tm.cosine_similarity_matrix.detach().cpu().numpy()
    n_feat = cos_sim.shape[0]
    if feature_labels is None:
        feature_labels = [f"f{i}" for i in range(n_feat)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cos_sim,
            x=feature_labels,
            y=feature_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(cos_sim, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=title,
        width=500,
        height=450,
        xaxis_title="Feature",
        yaxis_title="Feature",
    )
    fig.show()
    return fig


def plot_loss_curve(losses, title):
    """Simple loss curve plot."""
    fig = go.Figure(
        data=go.Scatter(
            y=losses,
            mode="lines",
            line=dict(width=1.5, color="steelblue"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        width=500,
        height=300,
        yaxis_type="log",
    )
    fig.show()
    return fig


# %% [markdown]
# ## 1. SphericalDistribution: Features on a Circle
#
# Features are placed at equal angular spacing around S^1 (the unit circle).
# When we sample, a random direction is chosen and nearby features activate
# with a cosine bump profile. The length_scale controls how many features
# light up per sample -- smaller means sparser, more localized bumps.
#
# The key question: the AE must represent features that live on a *curved*
# manifold. How do the learned embeddings arrange themselves?

# %%
N_FEAT_SPHERE = 8
N_HIDDEN = 2

sphere_dist = SphericalDistribution(
    n_features=N_FEAT_SPHERE,
    length_scale=1.2,  # moderate width -- ~3-4 features active per sample
    manifold_dim=1,  # circle (S^1)
    device=DEVICE,
)

# Let's look at where features are placed on the circle
positions = sphere_dist.feature_positions.detach().cpu().numpy()
print("Feature positions on S^1 (unit circle):")
for i in range(N_FEAT_SPHERE):
    angle = np.arctan2(positions[i, 1], positions[i, 0])
    print(f"  f{i}: angle = {np.degrees(angle):+7.1f} deg")
print()

x_sphere = data_fingerprint(sphere_dist, "SphericalDistribution (S^1)")

# %%
# The correlation matrix should reveal the circular topology: each feature
# correlates most with its angular neighbors and anticorrelates with the
# feature on the opposite side of the circle.
corr_sphere = torch.corrcoef(x_sphere.T).numpy()
print("Correlation structure (first 3 features):")
for i in range(3):
    neighbors = np.argsort(-corr_sphere[i])[:3]
    print(f"  f{i} most correlated with: {['f' + str(n) for n in neighbors]}")
print()

# %%
print("Training AE on spherical data...")
tm_sphere, losses_sphere = train_and_report(
    sphere_dist,
    N_FEAT_SPHERE,
    N_HIDDEN,
    n_epochs=3000,
)

plot_loss_curve(losses_sphere, "Spherical: Training Loss")

# %% [markdown]
# With only 2 hidden dimensions, the AE must project 8 features from a circle
# into R^2. Let's see how the embeddings arrange -- they should form a
# polygon-like pattern reflecting the circular topology.

# %%
labels_sphere = [f"f{i}" for i in range(N_FEAT_SPHERE)]
plot_embedding_2d(tm_sphere, "Spherical: Feature Embeddings in R^2", labels_sphere)
plot_cosine_heatmap(tm_sphere, "Spherical: Cosine Similarity", labels_sphere)

# The embedding should approximate the circular arrangement: adjacent features
# (by angle) should have high cosine similarity, opposite features should
# have negative similarity. This is a hallmark of the manifold structure
# leaking through into the learned geometry.


# %% [markdown]
# ## 2. TorusDistribution: Periodic Grids
#
# The torus T^1 is topologically a circle (same as S^1 above), but the
# distance metric is different: it uses the flat (geodesic) distance on the
# torus rather than the angular distance on the sphere. With torus_dim=2
# we get T^2 = S^1 x S^1 -- a 2D periodic grid.
#
# Let's use torus_dim=1 to compare directly with the sphere, then explore
# T^2 to see how periodicity in two directions affects the embeddings.

# %%
N_FEAT_TORUS = 8
torus_dist = TorusDistribution(
    n_features=N_FEAT_TORUS,
    length_scale=1.5,
    torus_dim=1,  # 1D torus = circle with flat metric
    device=DEVICE,
)

x_torus = data_fingerprint(torus_dist, "TorusDistribution (T^1)")

# %%
print("Training AE on torus data...")
tm_torus, losses_torus = train_and_report(
    torus_dist,
    N_FEAT_TORUS,
    N_HIDDEN,
    n_epochs=3000,
)

plot_loss_curve(losses_torus, "Torus: Training Loss")

# %%
labels_torus = [f"f{i}" for i in range(N_FEAT_TORUS)]
plot_embedding_2d(tm_torus, "Torus: Feature Embeddings in R^2", labels_torus)
plot_cosine_heatmap(tm_torus, "Torus: Cosine Similarity", labels_torus)

# Compare with the sphere: the torus places features at *random* angles
# (not equally spaced), so the embedding geometry will differ. The key
# difference is that the torus uses wrapped L2 distance, not geodesic
# angular distance. In practice, for dim=1, the results are similar but
# the random placement creates asymmetry.


# %% [markdown]
# ## 3. SimplexDistribution: Compositional Features
#
# This is fundamentally different from manifold distributions. Features are
# organized into groups (simplices). When a group fires, its features are
# drawn from a Dirichlet distribution -- they must sum to 1 within the group.
# This is the natural distribution for compositional data: market shares,
# mixture proportions, probability vectors.
#
# One might expect the sum-to-one constraint to create negative correlations
# within a group, but because each group fires independently with probability
# p_active, the on/off covariance actually dominates: when a group is active
# all its features are nonzero simultaneously, producing *positive* marginal
# correlations. The Dirichlet anticorrelation only appears if you condition
# on the group being active.  How does the AE handle this mixed signal?

# %%
# Two groups: a 3-simplex and a 3-simplex. Each fires with p=0.5.
simplex_dist = SimplexDistribution(
    simplex_sizes=[3, 3],
    p_active=0.5,
    device=DEVICE,
)
LABELS_SIMPLEX = ["A0", "A1", "A2", "B0", "B1", "B2"]

x_simplex = data_fingerprint(simplex_dist, "SimplexDistribution ([3,3])")

# %%
# Check the correlation structure.  Within each group the marginal
# correlations are actually *positive* because the dominant effect is
# the shared on/off activation (p_active=0.5).  The Dirichlet
# anticorrelation is a second-order effect.  Between groups, features
# are uncorrelated.
print("Within-group correlations (positive due to shared activation):")
corr_simplex = torch.corrcoef(x_simplex.T).numpy()
print(f"  corr(A0, A1) = {corr_simplex[0, 1]:.3f}")
print(f"  corr(A0, A2) = {corr_simplex[0, 2]:.3f}")
print()
print("Between-group correlations (should be ~0):")
print(f"  corr(A0, B0) = {corr_simplex[0, 3]:.3f}")
print(f"  corr(A0, B1) = {corr_simplex[0, 4]:.3f}")
print()

# %%
N_HIDDEN_SIMPLEX = 3  # 3 hidden dims for 6 features in 2 groups
print("Training AE on simplex data...")
tm_simplex, losses_simplex = train_and_report(
    simplex_dist,
    6,
    N_HIDDEN_SIMPLEX,
    n_epochs=4000,
)

plot_loss_curve(losses_simplex, "Simplex: Training Loss")

# %%
# With 3 hidden dims we can only plot 2D projections, so use cosine sim
plot_cosine_heatmap(tm_simplex, "Simplex: Cosine Similarity", LABELS_SIMPLEX)

# The cosine similarity matrix may show some block structure: features within
# the same group share activation patterns that the AE can exploit. Features
# in different groups are more independent.

# %%
# Let's also look at the norms -- the AE may learn different norms for
# features within vs between groups.
norms_simplex = tm_simplex.feature_norms.detach().cpu().numpy()
print("Feature norms by group:")
print(f"  Group A: {norms_simplex[:3].round(3)}")
print(f"  Group B: {norms_simplex[3:].round(3)}")
print()

# Within each group, norms should be roughly equal (the Dirichlet is
# symmetric). Between groups, norms may differ if the AE allocates
# representation capacity unevenly.


# %% [markdown]
# ## 4. SimplicialComplexDistribution: Glued Geometry
#
# A simplicial complex glues simplices together along shared vertices. This
# is the natural distribution for data with overlapping compositional
# structure -- think of a chemical reaction network where species
# participate in multiple reactions, or a social network where people
# belong to multiple groups.
#
# We'll build a triangle of edges: 6 vertices connected as
# (0,1), (1,2), (2,3), (3,4), (4,5), (5,0) -- a hexagonal ring of edges.
# Each sample picks one edge and places a Dirichlet draw on its two vertices.

# %%
N_VERTICES = 6
faces = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

complex_dist = SimplicialComplexDistribution(
    n_vertices=N_VERTICES,
    faces=faces,
    sampling_mode="single",  # one face per sample
    device=DEVICE,
)

x_complex = data_fingerprint(complex_dist, "SimplicialComplex (hexagonal ring)")

# %%
# The correlation structure: with "single" sampling, each sample activates
# exactly one edge.  Adjacent vertices share an edge but the Dirichlet
# constraint within that edge washes out the co-occurrence signal, leaving
# neighbor correlations near zero.  Opposite vertices *never* co-occur
# (they share no edges), so they anticorrelate.
print("Correlations -- neighbor vs opposite vertex:")

corr_complex = torch.corrcoef(x_complex.T).numpy()
for i in range(N_VERTICES):
    j = (i + 1) % N_VERTICES
    k = (i + 3) % N_VERTICES  # opposite vertex
    print(
        f"  corr(v{i}, v{j}) = {corr_complex[i, j]:+.3f}   "
        f"corr(v{i}, v{k}) = {corr_complex[i, k]:+.3f}"
    )
print()

# %%
print("Training AE on simplicial complex data...")
tm_complex, losses_complex = train_and_report(
    complex_dist,
    N_VERTICES,
    2,
    n_epochs=4000,
)

plot_loss_curve(losses_complex, "Simplicial Complex: Training Loss")

# %%
labels_complex = [f"v{i}" for i in range(N_VERTICES)]
plot_embedding_2d(tm_complex, "Simplicial Complex: Embeddings", labels_complex)
plot_cosine_heatmap(tm_complex, "Simplicial Complex: Cosine Similarity", labels_complex)

# The 2D embedding may hint at the ring topology, though with only 2 hidden
# dims for 6 vertices the reconstruction is imperfect. The cosine similarity
# matrix should reflect the anticorrelation of opposite vertices more than
# the neighbor relationships (which are weak in the data).


# %% [markdown]
# ## 5. CorrelatedPairs: Paired Feature Dependencies
#
# Features come in pairs (2i, 2i+1). A pair first activates jointly with
# probability p_active, then each member independently appears with
# probability p_individual. This creates positive correlations between
# paired features while allowing for independent variation.
#
# The interesting parameter is the correlation target: we can specify
# the desired correlation directly and let the distribution solve for
# the underlying probabilities.

# %%
N_FEAT_CORR = 8  # 4 pairs
corr_dist = CorrelatedPairs(
    n_features=N_FEAT_CORR,
    correlation=0.7,  # strong positive correlation within pairs
    density=0.1,  # overall sparsity
    device=DEVICE,
)

# The distribution solved for p_active and p_individual from our targets
print("Derived parameters:")
print(f"  p_active:     {corr_dist.p_active[0].item():.4f}")
print(f"  p_individual: {corr_dist.p_individual[0].item():.4f}")
print()

x_corr = data_fingerprint(corr_dist, "CorrelatedPairs (corr=0.7)")

# %%
# Verify the correlation structure.  The target is 0.7 but the achieved
# correlations are typically lower (~0.5) because the Bernoulli sampling
# introduces variance.  Between-pair correlations should be near zero.
print("Within-pair correlations (target: 0.7, actual ~0.5):")
corr_pairs = torch.corrcoef(x_corr.T).numpy()
for p in range(4):
    i, j = 2 * p, 2 * p + 1
    print(f"  corr(f{i}, f{j}) = {corr_pairs[i, j]:.3f}")
print()
print("Between-pair correlations (should be ~0):")
print(f"  corr(f0, f2) = {corr_pairs[0, 2]:.3f}")
print(f"  corr(f0, f4) = {corr_pairs[0, 4]:.3f}")
print()

# %%
print("Training AE on correlated pairs data...")
tm_corr, losses_corr = train_and_report(
    corr_dist,
    N_FEAT_CORR,
    3,
    n_epochs=3000,
)

plot_loss_curve(losses_corr, "Correlated Pairs: Training Loss")

# %%
labels_corr = [f"p{i // 2}{'a' if i % 2 == 0 else 'b'}" for i in range(N_FEAT_CORR)]
plot_cosine_heatmap(tm_corr, "Correlated Pairs: Cosine Similarity", labels_corr)

# Paired features should have *higher* cosine similarity than unpaired ones,
# since they co-activate. In practice the effect is moderate (mean ~0.35)
# rather than dramatic -- the high sparsity limits the signal. The 2x2
# block structure may be visible but is not always crisp.

# %%
# Let's verify: do paired features end up closer in embedding space?
cos_sim = tm_corr.cosine_similarity_matrix.detach().cpu().numpy()
within_pair = [cos_sim[2 * i, 2 * i + 1] for i in range(4)]
between_pair = [
    cos_sim[0, 2],
    cos_sim[0, 4],
    cos_sim[0, 6],
    cos_sim[2, 4],
    cos_sim[2, 6],
    cos_sim[4, 6],
]
print("Cosine similarity summary:")
print(
    f"  Within pairs:  mean={np.mean(within_pair):.3f}  "
    f"(range [{min(within_pair):.3f}, {max(within_pair):.3f}])"
)
print(
    f"  Between pairs: mean={np.mean(between_pair):.3f}  "
    f"(range [{min(between_pair):.3f}, {max(between_pair):.3f}])"
)
print()


# %% [markdown]
# ## Comparison: How Data Structure Shapes Representation
#
# Let's bring it all together. Each distribution imposes a different
# geometry and the AE responds accordingly.

# %%
print("=" * 60)
print("SUMMARY: Data Structure -> Representation Structure")
print("=" * 60)

models = {
    "Sphere (S^1)": tm_sphere,
    "Torus (T^1)": tm_torus,
    "Simplex ([3,3])": tm_simplex,
    "Simplicial Complex": tm_complex,
    "Correlated Pairs": tm_corr,
}

for name, tm in models.items():
    norms = tm.feature_norms
    sup = tm.superposition.item()
    cos = tm.cosine_similarity_matrix.detach()
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
    off_diag = cos[mask]

    print(f"\n  {name}:")
    print(f"    Hidden dims:       {tm.ae.n_hidden}")
    print(f"    Features:          {tm.ae.n_features}")
    print(f"    Superposition:     {sup:.3f}")
    print(f"    Norm range:        [{norms.min():.3f}, {norms.max():.3f}]")
    print(f"    Cosine sim range:  [{off_diag.min():.3f}, {off_diag.max():.3f}]")

print()
print("Key observations:")
print("  - Manifold distributions (sphere, torus) produce embeddings that")
print("    mirror the original topology -- neighbors stay neighbors.")
print("  - Simplex distributions show positive within-group correlations")
print("    (driven by shared on/off activation), not the negative ones")
print("    one might naively expect from the sum-to-one constraint.")
print("  - Correlated pairs show moderately higher within-pair cosine")
print("    similarity than between-pair, but the effect is modest.")
print("  - The simplicial complex ring shows strong anticorrelation for")
print("    opposite vertices (which never co-occur) and near-zero")
print("    correlation for neighbors.")


# %% [markdown]
# ## Appendix: Side-by-Side Cosine Similarity Comparison
#
# A single figure comparing the cosine similarity matrices across all
# five distributions.

# %%
fig = make_subplots(
    rows=1,
    cols=5,
    subplot_titles=list(models.keys()),
    horizontal_spacing=0.03,
)

for idx, (name, tm) in enumerate(models.items()):
    cos_sim = tm.cosine_similarity_matrix.detach().cpu().numpy()
    n = cos_sim.shape[0]
    labels = [str(i) for i in range(n)]

    fig.add_trace(
        go.Heatmap(
            z=cos_sim,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=(idx == 4),  # only show colorbar on last
            colorbar=dict(title="cos sim") if idx == 4 else None,
        ),
        row=1,
        col=idx + 1,
    )

fig.update_layout(
    title_text="Cosine Similarity: How Data Structure Shapes the Learned Geometry",
    width=1400,
    height=350,
)
fig.show()

# %%
print("Done. The punchline: the structure of your data influences the")
print("structure of the learned representation. Different distributions")
print("impose different geometric constraints, and the autoencoder's")
print("feature embeddings partially reflect those constraints -- though")
print("the signal can be noisy, especially when the hidden dimension is")
print("small or the data is very sparse.")
