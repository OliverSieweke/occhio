# %%
"""
Train a ToyModel on HierarchicalPairs with beta-coupled magnitudes,
then train an SAE on the learned latent space.
"""

from occhio.distributions.correlated import HierarchicalPairs
from occhio.sae import SAESimple
from occhio.autoencoder import TiedLinearRelu
from occhio.toy_model import ToyModel
import torch
import plotly.express as px
import plotly.graph_objects as go

# %%
torch.set_printoptions(3, sci_mode=False)
gen = torch.Generator()
gen.manual_seed(3)

N_FEAT = 8
N_HIDDEN = 5

dist = HierarchicalPairs(
    n_features=N_FEAT,
    p_active=0.3,
    p_follow=0.8,
    beta=0.0,
    generator=gen,
)

# %%
ae = TiedLinearRelu(N_FEAT, N_HIDDEN, generator=gen)
tm = ToyModel(dist, ae, importances=0.99 ** torch.arange(N_FEAT))
losses = tm.fit(25_000, verbose=True)[0]

# %%
px.line(losses, title="ToyModel loss")

# %%
W = tm.W  # (n_hidden, n_features)
for p in range(N_FEAT // 2):
    w_parent = W[:, 2 * p]
    w_child = W[:, 2 * p + 1]
    cos = torch.dot(w_parent, w_child) / (w_parent.norm() * w_child.norm())
    angle = torch.acos(cos.clamp(-1, 1)).item() * 180 / torch.pi
    print(
        f"Pair {p}: f{2 * p}–f{2 * p + 1}  angle = {angle:.1f}°  cos = {cos.item():.3f}"
    )

# %%
emb = tm.W.detach().numpy()
px.scatter(
    x=emb[0],
    y=emb[1],
    hover_name=[f"f{i}" for i in range(N_FEAT)],
    color=[f"f{i}" for i in range(N_FEAT)],
    title="Feature embeddings",
)

# %%
gen_sae = torch.Generator()
gen_sae.manual_seed(41)

sae = SAESimple(N_HIDDEN, N_FEAT + 4, l1_coef=0.02, generator=gen_sae)
sae_losses = sae.train_sae(tm.sample_latent, 20_000)

px.line(sae_losses, title="SAE loss")

# %%
samples = dist.sample(512)
embedded = tm.encode(samples).detach().numpy().T
reconstructed = sae.decode(sae.encode(tm.encode(samples))).detach().numpy().T

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=embedded[0], y=embedded[1], mode="markers", name="Ground Truth")
)
fig.add_trace(
    go.Scatter(x=reconstructed[0], y=reconstructed[1], mode="markers", name="SAE")
)
fig.update_layout(title="Latent space: ground truth vs SAE reconstruction")
fig

# %%
patterns = torch.eye(N_FEAT)
encoded_patterns = sae.encode(tm.encode(patterns)).detach().numpy()
px.imshow(
    encoded_patterns,
    title="Feature -> SAE Dictionary",
    labels=dict(x="Dictionary Dim", y="Feature"),
)

# %%
n_pairs = N_FEAT // 2
rows = []
labels = []
for p in range(n_pairs):
    parent_idx = 2 * p
    child_idx = 2 * p + 1

    parent_only = torch.zeros(N_FEAT)
    parent_only[parent_idx] = 0.9
    rows.append(parent_only)
    labels.append(f"f{parent_idx} (parent)")

    both = torch.zeros(N_FEAT)
    both[parent_idx] = 0.9
    both[child_idx] = 0.9
    rows.append(both)
    labels.append(f"f{parent_idx}+f{child_idx} (pair)")

patterns = torch.stack(rows)
encoded = sae.encode(ae.encode(patterns)).detach().numpy()
px.imshow(
    encoded,
    title="Parent vs Parent+Child -> SAE activations",
    labels=dict(x="Dictionary Dim", y="Pattern"),
    y=labels,
    color_continuous_scale="Reds",
)

# %%
