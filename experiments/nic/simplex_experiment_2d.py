# %%
import torch
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

from occhio.autoencoder import TiedLinearRelu
from occhio.sae.sae import SAESimple
from occhio.distributions import SimplexDistribution, SimplicialComplexDistribution
from occhio.toy_model import ToyModel

# %%
DEVICE = "mps"
gen = torch.Generator(DEVICE)
gen.manual_seed(6)

n_features = 5
p_active = 1 / (n_features + 1)
n_hidden = 2

dist = SimplicialComplexDistribution(
    n_vertices=n_features,
    faces=[(0, 1), (1, 2), (3, 4)],
    p_active=p_active,
    sampling_mode="sparse",
    generator=gen,
    device=DEVICE,
)
ae = TiedLinearRelu(n_features, n_hidden, generator=gen, device=DEVICE)
tm = ToyModel(distribution=dist, ae=ae, device=DEVICE)

# %%
losses, _ = tm.fit(30_000, 512)

# %%
px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training loss").show()

# %%
W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

# Sample points, encode into hidden space, and visualise
with torch.no_grad():
    samples = dist.sample(512)
    hidden = tm.ae.encode(samples).cpu().numpy()
    samples_cpu = samples.cpu().numpy()

# Label each point by which vertices are active
active_verts = []
for row in samples_cpu:
    verts = [str(v) for v in range(n_features) if row[v] > 0]
    active_verts.append(",".join(verts) if verts else "none")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=hidden[:, 0],
        y=hidden[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.6, color="gray"),
        text=active_verts,
        hovertemplate="Active vertices: %{text}<extra></extra>",
        name="Encoded samples",
    )
)

# Overlay feature (vertex) vectors
fig2.add_trace(
    go.Scatter(
        x=W[0],
        y=W[1],
        mode="markers+text",
        marker=dict(size=8),
        text=[f"v{j}" for j in range(n_features)],
        textposition="top center",
        name="Vertices",
    )
)
for j in range(n_features):
    fig2.add_trace(
        go.Scatter(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            mode="lines",
            line=dict(width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Draw edges between vertices that share a face
for i, face in enumerate(dist.faces):
    for a, b in zip(face, face[1:]):
        fig2.add_trace(
            go.Scatter(
                x=[W[0, a], W[0, b]],
                y=[W[1, a], W[1, b]],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig2.update_layout(
    title="Encoded samples in hidden space",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
)
fig2.show()

# %% --- SAE training ---

N_DICT = n_features + 2
SAE_STEPS = 20_000
SAE_BATCH = 512
SAE_LR = 3e-4
SAE_L1 = 0.1

sae = SAESimple(
    n_latent=n_hidden,
    n_dict=N_DICT,
    l1_coef=SAE_L1,
    device=DEVICE,
).to(DEVICE)


def sae_data_fn(n: int) -> torch.Tensor:
    raw = dist.sample(n)
    x = raw[0] if isinstance(raw, tuple) else raw
    return tm.ae.encode(x.to(DEVICE))


sae_losses = sae.train_sae(
    data_fn=sae_data_fn,
    n_steps=SAE_STEPS,
    batch_size=SAE_BATCH,
    lr=SAE_LR,
)

# %%
px.line(
    y=sae_losses,
    title="SAE Training Loss",
    log_y=True,
    labels={"x": "Step", "y": "Loss"},
).show()

# %% --- Match SAE dict elements to features via cosine similarity ---
with torch.no_grad():
    sae_dirs = sae.W_dec.detach()  # (N_DICT, n_hidden)
    gt_dirs = tm.W.detach().T  # (n_features, n_hidden)

    # Normalise
    sae_normed = sae_dirs / sae_dirs.norm(dim=1, keepdim=True)
    gt_normed = gt_dirs / gt_dirs.norm(dim=1, keepdim=True)

    cos_sim = sae_normed @ gt_normed.T  # (N_DICT, n_features)

    # Greedy matching: for each feature pick the best unmatched dict element
    cos_np = cos_sim.cpu().numpy()
    from scipy.optimize import linear_sum_assignment

    # Hungarian matching (maximise similarity → minimise negative)
    row_ind, col_ind = linear_sum_assignment(-cos_np)
    # row_ind = dict element indices, col_ind = feature indices
    # Build ordering: matched dict elements first (sorted by feature index),
    # then unmatched dict elements
    matched_pairs = sorted(zip(col_ind, row_ind))  # sort by feature index
    matched_dict_order = [d for _, d in matched_pairs]
    unmatched = [d for d in range(N_DICT) if d not in matched_dict_order]
    dict_order = matched_dict_order + unmatched  # reordering of dict elements

    # Labels: matched get "d{i}→v{j}", unmatched get "d{i}"
    dict_labels = [""] * N_DICT
    feature_for_dict = {}
    for feat, dct in matched_pairs:
        dict_labels[dct] = f"d{dct}→v{feat}"
        feature_for_dict[dct] = feat
    for d in unmatched:
        dict_labels[d] = f"d{d} (unmatched)"

    reordered_labels = [dict_labels[d] for d in dict_order]

print("Matching (cosine similarity):")
for feat, dct in matched_pairs:
    print(f"  v{feat} ↔ d{dct}  (cos={cos_np[dct, feat]:.3f})")
for d in unmatched:
    print(f"  d{d} unmatched (best cos={cos_np[d].max():.3f})")

# %% --- Samples overlaid with SAE decoder vectors ---
with torch.no_grad():
    samples = dist.sample(1024)
    x = samples[0] if isinstance(samples, tuple) else samples
    hidden = tm.ae.encode(x.to(DEVICE)).cpu().numpy()

    # SAE decoder directions in hidden space
    sae_W_dec = sae.W_dec.detach().cpu().numpy()  # (N_DICT, n_hidden)

    # Ground-truth ToyModel encoder columns
    W = tm.W.detach().cpu().numpy()  # (n_hidden, n_features)

fig3 = go.Figure()

# Encoded samples
fig3.add_trace(
    go.Scatter(
        x=hidden[:, 0],
        y=hidden[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.4, color="gray"),
        name="Encoded samples",
    )
)

# Ground-truth feature vectors (ToyModel W columns), colored to match SAE pairing
_colors = pc.qualitative.Plotly
for j in range(n_features):
    fig3.add_trace(
        go.Scatter(
            x=[0, W[0, j]],
            y=[0, W[1, j]],
            mode="lines+markers",
            line=dict(width=2, color=_colors[j % len(_colors)]),
            marker=dict(size=6),
            name=f"GT v{j}",
            legendgroup="gt",
            showlegend=True,
            hovertext=f"GT v{j}",
            hoverinfo="text",
        )
    )

# SAE decoder vectors, colored by matched feature
_colors = pc.qualitative.Plotly
for d in range(N_DICT):
    if d in feature_for_dict:
        feat = feature_for_dict[d]
        color = _colors[feat % len(_colors)]
        label = f"d{d}→v{feat}"
    else:
        color = "gray"
        label = f"d{d} (unmatched)"
    fig3.add_trace(
        go.Scatter(
            x=[sae_W_dec[d, 0]],
            y=[sae_W_dec[d, 1]],
            mode="markers",
            marker=dict(size=8, symbol="diamond", color=color),
            name=label,
            legendgroup="sae",
            showlegend=True,
            hovertext=label,
            hoverinfo="text",
        )
    )

# SAE decoder bias (b_dec) in hidden space
with torch.no_grad():
    b_dec_np = sae.b_dec.detach().cpu().numpy()
fig3.add_trace(
    go.Scatter(
        x=[b_dec_np[0]],
        y=[b_dec_np[1]],
        mode="markers",
        marker=dict(size=12, symbol="star", color="green"),
        name="b_dec",
        hovertext=f"b_dec ({b_dec_np[0]:.3f}, {b_dec_np[1]:.3f})",
        hoverinfo="text",
    )
)

fig3.update_layout(
    title="Ground Truth vs SAE Decoder Directions",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
)
fig3.show()

# %% --- Samples vs SAE reconstruction ---
with torch.no_grad():
    hidden_t = tm.ae.encode(x.to(DEVICE))
    reconstructed = sae.decode(sae.encode(hidden_t)).cpu().numpy()
    hidden_np = hidden_t.cpu().numpy()

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter(
        x=hidden_np[:, 0],
        y=hidden_np[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.5, color="blue"),
        name="Original (encoded)",
    )
)

fig4.add_trace(
    go.Scatter(
        x=reconstructed[:, 0],
        y=reconstructed[:, 1],
        mode="markers",
        marker=dict(size=3, opacity=0.5, color="red"),
        name="SAE encode→decode",
    )
)

fig4.update_layout(
    title="Hidden Samples vs SAE Reconstruction",
    xaxis_title="h₀",
    yaxis_title="h₁",
    height=700,
    width=700,
)
fig4.show()

# %% --- Per-feature SAE activations ---
with torch.no_grad():
    eye = torch.eye(n_features, device=DEVICE)
    sae_acts = sae.encode(tm.ae.encode(eye)).cpu().numpy()  # (n_features, N_DICT)

px.imshow(
    sae_acts[:, dict_order],
    labels=dict(x="SAE dict element", y="Feature"),
    x=reordered_labels,
    y=[f"v{j}" for j in range(n_features)],
    title="SAE activations per one-hot feature (matched order)",
    aspect="auto",
).show()

# %% --- SAE dict elements decoded to feature space ---
with torch.no_grad():
    eye_dict = torch.eye(N_DICT, device=DEVICE)
    decoded_features = (
        tm.ae.decode(sae.decode(eye_dict)).cpu().numpy()
    )  # (N_DICT, n_features)

px.imshow(
    decoded_features[dict_order],
    labels=dict(x="Feature", y="SAE dict element"),
    x=[f"v{j}" for j in range(n_features)],
    y=reordered_labels,
    title="SAE dict elements decoded to feature space (matched order)",
    aspect="auto",
).show()
# %%
