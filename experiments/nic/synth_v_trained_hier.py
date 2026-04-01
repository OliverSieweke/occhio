# %%
"""Compare TiedLinearRelu vs SynthAE on HierarchicalPairs.

Features are organized in pairs: primary features activate with p_active,
secondary features follow with probability p_follow given the primary is active.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.distributions.correlated import HierarchicalPairs
from occhio.sae.sae import SAESimple
from occhio.toy_model import ToyModel

# --- Publication-ready figure styling ---
MODEL_COLORS = {"TiedLinearRelu": "#2563EB", "SynthAE (ortho)": "#16A34A"}
_THRESH_COLORS = ["#3B82F6", "#F59E0B", "#EF4444"]

_AXIS = dict(
    showgrid=True,
    gridcolor="#E5E7EB",
    showline=True,
    linecolor="#374151",
    linewidth=1.2,
    ticks="outside",
    tickcolor="#374151",
    minor=dict(ticks="outside", tickcolor="#9CA3AF"),
    zeroline=False,
)


def style_fig(fig, nticksx=10, nticksy=8):
    """Apply publication-ready styling."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#1F2937"),
        title_font=dict(size=15),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D1D5DB",
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    fig.update_xaxes(**_AXIS, nticks=nticksx)
    fig.update_yaxes(**_AXIS, nticks=nticksy)
    return fig


# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 100  # must be even (50 primary/secondary pairs)
D_HIDDEN = 20
N_EPOCHS = 40_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14
EVAL_FREQ = 250

# %%
high = 0.3
low = 1.0 / N_FEATURES
alpha = np.log(high / low) / np.log(N_FEATURES)
print(f"{alpha=}")
firing_probs = [high / (i + 1) ** alpha for i in range(N_FEATURES)]
# --- Distribution ---
dist = HierarchicalPairs(
    n_features=N_FEATURES,
    p_active=firing_probs,
    p_follow=0.7,
    beta=0.99,
    device=DEVICE,
)

# Quick sanity check
activations = dist.sample(1024)
print(f"Activations shape: {activations.shape}")
print(f"Mean L0: {(activations > 0).float().sum(dim=-1).mean():.1f}")
print(
    f"Mean L2 norm: {activations.norm(dim=-1).mean():.2f} ± {activations.norm(dim=-1).std():.2f}"
)


# %%
# --- Hooks for evaluation ---
def eval_hook(data):
    """Compute eval loss on a large fresh sample."""
    tm = data["tm"]
    x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
    x_hat = tm.ae(x)[0]
    return tm.ae.loss(x, x_hat, tm.importances).item()


def per_feature_hook(data):
    """Per-feature reconstruction MSE on one-hot inputs."""
    tm = data["tm"]
    eye = torch.eye(N_FEATURES, device=tm.device)
    x_hat = tm.ae(eye)[0]
    return (eye - x_hat).pow(2).sum(dim=-1).cpu().numpy()


# %%
# --- Train TiedLinearRelu ---
print("Training TiedLinearRelu...")
gen1 = torch.Generator(DEVICE).manual_seed(SEED)
ae_tied = TiedLinearRelu(N_FEATURES, D_HIDDEN, device=DEVICE, generator=gen1)
tm_tied = ToyModel(distribution=dist, ae=ae_tied, device=DEVICE)

_, hook_results_tied = tm_tied.fit(
    N_EPOCHS,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_tied = hook_results_tied[0]
per_feature_tied = hook_results_tied[1]
print(f"  Final eval loss: {eval_losses_tied[-1]:.6f}")

# %%
# --- SynthAE (train bias only) ---
print("Training SynthAE (orthogonalized, bias only)...")
gen2 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=100,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen2,
)
tm_synth = ToyModel(distribution=dist, ae=ae_synth, device=DEVICE)

N_EPOCHS_SYNTH = 10_000
_, hook_results_synth = tm_synth.fit(
    N_EPOCHS_SYNTH,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
loss_synth = hook_results_synth[0][-1]
pf_synth = hook_results_synth[1][-1]
print(f"  Final eval loss: {loss_synth:.6f}")

# %%
# --- Plot: Eval loss curve ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=eval_epochs,
        y=eval_losses_tied,
        name="TiedLinearRelu",
        mode="lines",
        line=dict(width=2, color=MODEL_COLORS["TiedLinearRelu"]),
    )
)
fig.add_hline(
    y=loss_synth,
    line_dash="dash",
    line_color=MODEL_COLORS["SynthAE (ortho)"],
    annotation_text="SynthAE",
)
fig.update_layout(
    title=f"Eval Loss — HierarchicalPairs (N={N_FEATURES}, D={D_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: Per-feature reconstruction MSE ---
# Features ordered as pairs: 0,1 | 2,3 | ... (primary, secondary)
final_tied = np.array(per_feature_tied[-1])
x = np.arange(N_FEATURES)
pair_labels = [f"{'P' if i % 2 == 0 else 'S'}{i // 2}" for i in range(N_FEATURES)]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=final_tied,
        name="TiedLinearRelu",
        mode="markers",
        marker=dict(size=5, opacity=0.7, color=MODEL_COLORS["TiedLinearRelu"]),
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=pf_synth,
        name="SynthAE (ortho)",
        mode="markers",
        marker=dict(size=5, opacity=0.7, color=MODEL_COLORS["SynthAE (ortho)"]),
    )
)
fig.update_layout(
    title="Per-Feature Reconstruction MSE",
    xaxis_title="Feature index (P=primary, S=secondary within pair)",
    yaxis_title="MSE",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: Features recovered (MSE < threshold) over training ---
THRESHOLDS = [0.2, 0.5, 1.0]
COLORS = _THRESH_COLORS

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, COLORS):
    n_recovered = [int((np.array(s) < thresh).sum()) for s in per_feature_tied]
    n_synth = int((pf_synth < thresh).sum())
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered,
            name=f"TiedLinearRelu (τ={thresh})",
            mode="lines",
            line=dict(color=color, width=2),
        )
    )
    fig.add_hline(y=n_synth, line_dash="dash", line_color=color)
fig.update_layout(
    title="Features Recovered (MSE < τ) Over Training",
    xaxis_title="Epoch",
    yaxis_title="# features recovered",
)
style_fig(fig)
fig.show()

# %%
# --- Plot: W^T W comparison ---
models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth)]

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["TiedLinearRelu", "SynthAE (ortho)"]
)
for i, (name, tm) in enumerate(models):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(z=WtW, colorscale="RdBu_r", zmid=0, showscale=(i == 1)),
        row=1,
        col=i + 1,
    )
fig.update_layout(title="W^T W Comparison", height=400, width=900)
style_fig(fig)
fig.show()

# %%
# --- Plot: Geometric properties ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Feature Dimensionalities", "Feature Norms", "Total Interference"],
)
model_colors = MODEL_COLORS
for name, tm in models:
    fd = tm.feature_dimensionalities.detach().cpu().numpy()
    fn = tm.feature_norms.detach().cpu().numpy()
    ti = tm.total_feature_interferences.detach().cpu().numpy()
    color = model_colors[name]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fd,
            name=name,
            mode="markers",
            marker=dict(size=5, opacity=0.7, color=color),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=fn,
            name=name,
            mode="markers",
            showlegend=False,
            marker=dict(size=5, opacity=0.7, color=color),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ti,
            name=name,
            mode="markers",
            showlegend=False,
            marker=dict(size=5, opacity=0.7, color=color),
        ),
        row=1,
        col=3,
    )
fig.update_layout(title="Geometric Properties", height=400, width=1200)
for col in range(1, 4):
    fig.update_xaxes(title_text="Feature index", row=1, col=col)
style_fig(fig)
fig.show()

# %%
# --- SAE training ---
N_DICT = N_FEATURES + 4
SAE_STEPS = 30_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.25

sae_results = {}

for name, tm in [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth)]:
    print(f"\nTraining SAE on {name}...")

    sae = SAESimple(
        n_latent=D_HIDDEN,
        n_dict=N_DICT,
        l1_coef=SAE_L1,
        device=DEVICE,
    ).to(DEVICE)

    def make_data_fn(tm_ref):
        def data_fn(n: int) -> torch.Tensor:
            x = tm_ref.distribution.sample(n).to(DEVICE)
            return tm_ref.ae.encode(x)

        return data_fn

    sae_losses = sae.train_sae(
        data_fn=make_data_fn(tm),
        n_steps=SAE_STEPS,
        batch_size=SAE_BATCH,
        lr=SAE_LR,
    )
    sae_results[name] = {"sae": sae, "tm": tm, "losses": sae_losses}
    print("  Done.")

# %%
# --- SAE loss curves ---
fig = go.Figure()
for name, res in sae_results.items():
    fig.add_trace(
        go.Scatter(
            y=res["losses"],
            mode="lines",
            name=name,
            line=dict(width=2, color=MODEL_COLORS[name]),
        )
    )
fig.update_layout(
    title="SAE Training Loss",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
)
style_fig(fig)
fig.show()

# %%
# --- SAE one-hot activations with cosine matching ---
for name, res in sae_results.items():
    sae = res["sae"]
    tm = res["tm"]

    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        h_eye = tm.ae.encode(eye)
        sae_acts = sae.encode(h_eye).cpu().numpy()  # (N_FEATURES, N_DICT)

    # Match features to dict elements by one-hot firing strength
    feat_idx, dict_idx = linear_sum_assignment(-sae_acts)
    mean_firing = sae_acts[feat_idx, dict_idx].mean()

    unmatched_feats = [f for f in range(N_FEATURES) if f not in set(feat_idx)]
    unmatched_dicts = [d for d in range(N_DICT) if d not in set(dict_idx)]
    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts

    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
    row_labels = [f"f{f}" for f in row_order]
    col_labels = [f"d{d}" for d in col_order]

    diag_sum = sum(sae_acts_matched[i, i] for i in range(len(feat_idx)))
    total_sum = sae_acts_matched.sum()
    diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
    res["col_order"] = col_order
    res["feat_to_dict"] = {int(f): int(d) for f, d in zip(feat_idx, dict_idx)}

    print(f"{name}: diagonality={diagonality:.4f}  mean_firing={mean_firing:.4f}")

    fig_imshow = px.imshow(
        sae_acts_matched,
        labels=dict(
            x="SAE dict element (firing matched)", y="Feature (firing matched)"
        ),
        x=col_labels,
        y=row_labels,
        title=f"SAE one-hot activations (firing matched, diag={diagonality:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    )
    style_fig(fig_imshow)
    fig_imshow.show()

# %%
# --- f0, f1, f0+f1 probe ---
# Rows = activation patterns; cols = matched SAE dict elements that fire for any probe.
probe_inputs = {
    "f0": torch.zeros(N_FEATURES),
    "f1": torch.zeros(N_FEATURES),
    "f0+f1": torch.zeros(N_FEATURES),
}
activation_size = 0.6
probe_inputs["f0"][0] = activation_size
probe_inputs["f1"][1] = activation_size
probe_inputs["f0+f1"][0] = activation_size
probe_inputs["f0+f1"][1] = activation_size

probe_names = list(probe_inputs.keys())
model_names = ["TiedLinearRelu", "SynthAE (ortho)"]

fig = make_subplots(
    rows=1,
    cols=len(model_names),
    subplot_titles=model_names,
)

for col, name in enumerate(model_names, start=1):
    sae = sae_results[name]["sae"]
    tm = sae_results[name]["tm"]
    feat_to_dict = sae_results[name]["feat_to_dict"]

    with torch.no_grad():
        # Compute activations for all 3 probes: (3, N_DICT)
        all_acts = []
        for p_name in probe_names:
            inp = probe_inputs[p_name].unsqueeze(0).to(DEVICE)
            h = tm.ae.encode(inp)
            acts = sae.encode(h).squeeze(0).cpu().numpy()
            all_acts.append(acts)
        all_acts = np.stack(all_acts)  # (3, N_DICT)

    # Keep only matched dict elements (from feat_to_dict), ordered by feature index
    matched_feat_order = sorted(feat_to_dict.keys())
    matched_dict_cols = [feat_to_dict[f] for f in matched_feat_order]

    # Filter to those with any nonzero activation across the 3 probes
    active_mask = all_acts[:, matched_dict_cols].max(axis=0) > 0
    active_feat_order = [f for f, m in zip(matched_feat_order, active_mask) if m]
    active_dict_cols = [feat_to_dict[f] for f in active_feat_order]

    z = all_acts[:, active_dict_cols]
    x_labels = [f"d→f{f}" for f in active_feat_order]

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=probe_names,
            colorscale="ylgnbu_r",
            showscale=(col == len(model_names)),
            zmin=0,
        ),
        row=1,
        col=col,
    )

fig.update_layout(
    title="SAE activations for f0, f1, f0+f1 (matched dict elements only)",
    height=300,
    width=500 * len(model_names),
)
style_fig(fig)
fig.show()

# %%
