# %%
"""MRH experiment: compare TiedLinearRelu vs AttnLinearAE on SparseUniform.

Sweeps over p_active values and plots W embeddings (one-hot encodings projected
to the 2D latent space) for both architectures side by side.  The top row shows
the standard linear bottleneck (LRH-style), the bottom row shows the multi-head
softmax bottleneck (MRH-style).
"""

from occhio.distributions import SparseUniform, SimplicialComplexDistribution
from occhio.autoencoder import (
    TiedLinearRelu,
    AttnLinearAE,
    LinearAttnAE,
    AttnAttnAE,
)
from occhio.toy_model import ToyModel
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_features = 6
n_hidden = 2
n_heads = 2
dict_size = 4
N_EPOCHS = 50_000
batch_size = 256

# %%
device = torch.device("mps")
importances = torch.tensor([0.95**i for i in range(n_features)], device=device)
gen = torch.Generator(device)

p_actives = [0.1, 0.5, 0.75]

# Train all models and collect W embeddings
results: dict[str, list[np.ndarray]] = {
    "TiedLinearRelu": [],
    "AttnLinearAE": [],
    "LinearAttnAE": [],
    "AttnAttnAE": [],
}
losses: dict[str, list[list[float]]] = {
    "TiedLinearRelu": [],
    "AttnLinearAE": [],
    "LinearAttnAE": [],
    "AttnAttnAE": [],
}
# Store encoded samples (64 per model per sparsity) for overlay on embedding plots
encoded_samples: dict[str, list[np.ndarray]] = {
    "TiedLinearRelu": [],
    "AttnLinearAE": [],
    "LinearAttnAE": [],
    "AttnAttnAE": [],
}
N_SAMPLES = 64

EVAL_BATCH = 2**14


def eval_loss_hook(data: dict) -> float:
    """Evaluate loss on a fresh large batch."""
    tm: ToyModel = data["tm"]
    raw = tm.distribution.sample(EVAL_BATCH)
    ae_device = tm.ae.device
    if isinstance(raw, tuple):
        raw = tuple(
            t.to(ae_device, non_blocking=True) if isinstance(t, torch.Tensor) else t
            for t in raw
        )
        x = raw[0]
    else:
        raw = raw.to(ae_device, non_blocking=True)
        x = raw
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(raw, x_hat, tm.importances).item()


HOOK_FREQ = 500

for p_active in p_actives:
    print(f"starting on p_active = {p_active}")
    # --- TiedLinearRelu (LRH baseline) ---
    gen.manual_seed(7)
    # dist = SparseUniform(n_features, p_active, generator=gen, device=device)
    dist = SimplicialComplexDistribution(
        6,
        [(0, 1), (2, 3), (4, 5)],
        p_active=p_active,
        sampling_mode="sparse",
        generator=gen,
        device=device,
    )

    ae_linear = TiedLinearRelu(n_features, n_hidden, generator=gen, device=device)
    tm_linear = ToyModel(dist, ae_linear, importances=importances)
    _, hook_out = tm_linear.fit(
        20_000,
        verbose=True,
        batch_size=batch_size,
        track_losses=False,
        hooks=[eval_loss_hook],
        hook_freq=HOOK_FREQ,
    )
    results["TiedLinearRelu"].append(tm_linear.W.detach().cpu().numpy())
    losses["TiedLinearRelu"].append(hook_out[0])
    with torch.no_grad():
        s = dist.sample(N_SAMPLES)
        s_in = s[0] if isinstance(s, tuple) else s
        encoded_samples["TiedLinearRelu"].append(
            tm_linear.ae.encode(s_in.to(device)).cpu().numpy()
        )

    # --- AttnLinearAE (MRH) ---
    gen.manual_seed(7)
    # dist = SparseUniform(n_features, p_active, generator=gen, device=device)
    dist = SimplicialComplexDistribution(
        6,
        [(0, 1), (2, 3), (4, 5)],
        p_active=p_active,
        sampling_mode="sparse",
        generator=gen,
        device=device,
    )
    ae_mrh = AttnLinearAE(
        n_features,
        n_hidden,
        n_heads=n_heads,
        dict_size=dict_size,
        generator=gen,
        device=device,
    )
    tm_mrh = ToyModel(dist, ae_mrh, importances=importances)
    _, hook_out = tm_mrh.fit(
        N_EPOCHS,
        verbose=True,
        batch_size=batch_size,
        track_losses=False,
        hooks=[eval_loss_hook],
        hook_freq=HOOK_FREQ,
    )
    results["AttnLinearAE"].append(tm_mrh.W.detach().cpu().numpy())
    losses["AttnLinearAE"].append(hook_out[0])
    with torch.no_grad():
        s = dist.sample(N_SAMPLES)
        s_in = s[0] if isinstance(s, tuple) else s
        encoded_samples["AttnLinearAE"].append(
            tm_mrh.ae.encode(s_in.to(device)).cpu().numpy()
        )

    # --- LinearAttnAE (MRH tied) ---
    gen.manual_seed(7)
    # dist = SparseUniform(n_features, p_active, generator=gen, device=device)
    dist = SimplicialComplexDistribution(
        6,
        [(0, 1), (2, 3), (4, 5)],
        p_active=p_active,
        sampling_mode="sparse",
        generator=gen,
        device=device,
    )
    ae_sym = LinearAttnAE(
        n_features,
        n_hidden,
        n_heads=n_heads,
        dict_size=dict_size,
        generator=gen,
        device=device,
    )
    tm_sym = ToyModel(dist, ae_sym, importances=importances)
    _, hook_out = tm_sym.fit(
        N_EPOCHS,
        verbose=True,
        batch_size=batch_size,
        track_losses=False,
        hooks=[eval_loss_hook],
        hook_freq=HOOK_FREQ,
    )
    results["LinearAttnAE"].append(tm_sym.W.detach().cpu().numpy())
    losses["LinearAttnAE"].append(hook_out[0])
    with torch.no_grad():
        s = dist.sample(N_SAMPLES)
        s_in = s[0] if isinstance(s, tuple) else s
        encoded_samples["LinearAttnAE"].append(
            tm_sym.ae.encode(s_in.to(device)).cpu().numpy()
        )

    # --- AttnAttnAE ---
    gen.manual_seed(7)
    dist = SimplicialComplexDistribution(
        6,
        [(0, 1), (2, 3), (4, 5)],
        p_active=p_active,
        sampling_mode="sparse",
        generator=gen,
        device=device,
    )
    ae_aa = AttnAttnAE(
        n_features,
        n_hidden,
        n_heads=n_heads,
        dict_size=dict_size,
        generator=gen,
        device=device,
    )
    tm_aa = ToyModel(dist, ae_aa, importances=importances)
    _, hook_out = tm_aa.fit(
        N_EPOCHS,
        verbose=True,
        batch_size=batch_size,
        track_losses=False,
        hooks=[eval_loss_hook],
        hook_freq=HOOK_FREQ,
    )
    results["AttnAttnAE"].append(tm_aa.W.detach().cpu().numpy())
    losses["AttnAttnAE"].append(hook_out[0])
    with torch.no_grad():
        s = dist.sample(N_SAMPLES)
        s_in = s[0] if isinstance(s, tuple) else s
        encoded_samples["AttnAttnAE"].append(
            tm_aa.ae.encode(s_in.to(device)).cpu().numpy()
        )


# %%
row_labels = [
    "TiedLinearRelu (LRH)",
    "AttnLinearAE (MRH)",
    "LinearAttnAE (MRH tied)",
    "AttnAttnAE",
]
row_keys = [
    "TiedLinearRelu",
    "AttnLinearAE",
    "LinearAttnAE",
    "AttnAttnAE",
]

fig = make_subplots(
    rows=4,
    cols=len(p_actives),
    subplot_titles=[f"p={p}" for p in p_actives] * 4,
    vertical_spacing=0.08,
    horizontal_spacing=0.04,
)

theta = np.linspace(0, 2 * np.pi, 100)
colors = [
    "#440154",
    "#443983",
    "#31688e",
    "#21918c",
    "#35b779",
    "#fde725",
]

for row_idx, key in enumerate(row_keys):
    for col_idx, (p_active, W) in enumerate(zip(p_actives, results[key])):
        r, c = row_idx + 1, col_idx + 1
        show_legend = row_idx == 0 and col_idx == 0

        # Unit circle
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="black", dash="dash", width=0.5),
                opacity=0.3,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Feature embeddings
        for i in range(n_features):
            fig.add_trace(
                go.Scatter(
                    x=[W[0, i]],
                    y=[W[1, i]],
                    mode="markers+text",
                    marker=dict(color=colors[i], size=10),
                    text=[str(i)],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"feature {i}",
                    showlegend=show_legend,
                    legendgroup=f"f{i}",
                ),
                row=r,
                col=c,
            )

        # Encoded samples as low-opacity points
        z = encoded_samples[key][col_idx]  # (N_SAMPLES, n_hidden)
        fig.add_trace(
            go.Scatter(
                x=z[:, 0],
                y=z[:, 1],
                mode="markers",
                marker=dict(color="black", size=3, opacity=0.25),
                name="samples",
                showlegend=show_legend,
                legendgroup="samples",
            ),
            row=r,
            col=c,
        )

        fig.update_xaxes(
            range=[-1.5, 1.5],
            scaleanchor=f"y{row_idx * len(p_actives) + col_idx + 1}",
            row=r,
            col=c,
        )
        fig.update_yaxes(range=[-1.5, 1.5], row=r, col=c)

# Row labels via y-axis titles on the first column
for i, label in enumerate(row_labels):
    fig.update_yaxes(title_text=label, row=i + 1, col=1)

fig.update_layout(
    title_text=(
        f"Feature embeddings: LRH vs MRH bottleneck on SparseUniform<br>"
        f"<sub>n_features={n_features}, n_hidden={n_hidden}, "
        f"heads={n_heads}, dict_size={dict_size}</sub>"
    ),
    height=1200,
    width=300 * len(p_actives),
)
fig.show()

# %%
# Loss curves: one subplot per p_active, both architectures overlaid
loss_fig = make_subplots(
    rows=1,
    cols=len(p_actives),
    subplot_titles=[f"p={p}" for p in p_actives],
    horizontal_spacing=0.04,
)

loss_epochs = list(range(0, N_EPOCHS, HOOK_FREQ)) + [N_EPOCHS - 1]

for col_idx, p_active in enumerate(p_actives):
    c = col_idx + 1
    for key, color in zip(row_keys, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        loss_fig.add_trace(
            go.Scatter(
                x=loss_epochs,
                y=losses[key][col_idx],
                mode="lines",
                name=key,
                line=dict(color=color),
                showlegend=col_idx == 0,
                legendgroup=key,
            ),
            row=1,
            col=c,
        )
    loss_fig.update_xaxes(title_text="Epoch", row=1, col=c)
    loss_fig.update_yaxes(
        title_text="Loss" if c == 1 else None, type="log", row=1, col=c
    )

loss_fig.update_layout(
    title_text=(
        f"Training loss: LRH vs MRH on SparseUniform<br>"
        f"<sub>n_features={n_features}, n_hidden={n_hidden}, "
        f"heads={n_heads}, dict_size={dict_size}</sub>"
    ),
    height=400,
    width=300 * len(p_actives),
)
loss_fig.show()

# %%
print("=== Frobenius norms (last p_active) ===")
print(f"TiedLinearRelu:  W={ae_linear.W.norm().item():.4f}")
print(
    f"AttnLinearAE:    W_out={ae_mrh.W_out.norm().item():.4f}  "
    f"W_mix={ae_mrh.W_mix.norm().item():.4f}"
)
print(f"LinearAttnAE:    W_enc={ae_sym.W_enc.norm().item():.4f}")
print(f"AttnAttnAE:      W_mix={ae_aa.W_mix.norm().item():.4f}")

# %%
