# %%
"""Compare TiedLinearRelu vs SynthAE on SyntheticDataModel.

TiedLinearRelu: learns W from data via gradient descent.
SynthAE: unit-norm tied weights positioned roughly orthogonally (no training needed).
Both use ReLU in decode. The question is whether the learned W outperforms
the synthetic dictionary approach.

Scaled-down version of the SSB benchmark (Anthropic, 2025) for a weaker machine:
  - N = 100 ground-truth features, D = 10 hidden dim
  - All other parameters scaled proportionally from the original
    (N=16384, D=768) specification.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from occhio.autoencoder import TiedLinearRelu, SynthAE
from occhio.sae.sae import SAESimple
from occhio.distributions.ssb import (
    SyntheticDataModel,
    SyntheticDataConfig,
    HierarchyNode,
)
from occhio.toy_model import ToyModel

# %%
# --- Configuration ---
DEVICE = "mps"
SEED = 42
N_FEATURES = 100
D_HIDDEN = 24
N_EPOCHS = 30_000
BATCH_SIZE = 512
EVAL_SAMPLES = 2**14
EVAL_FREQ = 250


# --- Build hierarchy forest ---
# Original: 128 roots, branching factor 4, max depth 3, covering 10884 features.
# Scaled: 8 roots, branching factor 4, max depth 2, covering 8*(1+4+16) = 168
# but we only have 100 features, so: 8 roots * (1 + 4) = 40 features in hierarchy.
# Remaining 60 features have no hierarchy.
def build_hierarchy(
    start_idx: int, n_roots: int, branching: int
) -> tuple[list[HierarchyNode], int]:
    """Build a forest of trees with mutual exclusion and parent-scaled magnitudes."""
    idx = start_idx
    roots = []
    for _ in range(n_roots):
        root_idx = idx
        idx += 1
        children = []
        for _ in range(branching):
            children.append(HierarchyNode(feature_idx=idx))
            idx += 1
        roots.append(
            HierarchyNode(
                feature_idx=root_idx,
                children=children,
                mutually_exclusive_children=True,
                parent_scaled=True,
            )
        )
    return roots, idx


hierarchy_roots, _ = build_hierarchy(start_idx=0, n_roots=8, branching=4)
# 8 roots with 4 children each = 40 features in hierarchy, 60 free

# %%
# --- SyntheticDataModel config ---
config = SyntheticDataConfig(
    n_features=N_FEATURES,
    # Firing probabilities
    firing_prob_distribution="zipfian",
    p_max=0.4,
    p_min=0.5 / N_FEATURES,
    alpha=0.5,
    # Magnitudes — linear mean, folded-normal stdev
    mean_distribution="linear",
    mean_high=3.0,
    mean_low=1.0,
    std_distribution="folded_normal",
    folded_normal_mu=0.5,
    folded_normal_sigma=0.5,
    # Correlation
    correlation_rank=4,
    correlation_scale=0.1,
    # Hierarchy
    hierarchy=hierarchy_roots,
    compensate_probabilities=True,
    # Runtime
    device=DEVICE,
)

dist = SyntheticDataModel(config, seed=SEED)

# %%
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
# --- Helper: evaluate a (non-trained) model once ---
def evaluate_model(tm):
    """Return (eval_loss, per_feature_mse) for a model without training."""
    with torch.no_grad():
        x = tm.distribution.sample(EVAL_SAMPLES).to(tm.device)
        x_hat = tm.ae(x)[0]
        eval_loss = tm.ae.loss(x, x_hat, tm.importances).item()

        eye = torch.eye(N_FEATURES, device=tm.device)
        e_hat = tm.ae(eye)[0]
        pf_mse = (eye - e_hat).pow(2).sum(dim=-1).cpu().numpy()
    return eval_loss, pf_mse


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
gen3 = torch.Generator(DEVICE).manual_seed(SEED)
ae_synth_ortho = SynthAE(
    N_FEATURES,
    D_HIDDEN,
    orthogonalize=True,
    ortho_steps=2_000,
    ortho_lr=3e-4,
    device=DEVICE,
    generator=gen3,
)
tm_synth_ortho = ToyModel(distribution=dist, ae=ae_synth_ortho, device=DEVICE)

N_EPOCHS_SYNTH = 10_000
_, hook_results_synth = tm_synth_ortho.fit(
    N_EPOCHS_SYNTH,
    batch_size=BATCH_SIZE,
    hooks=[eval_hook, per_feature_hook],
    hook_freq=EVAL_FREQ,
    verbose=True,
)
eval_losses_synth = hook_results_synth[0]
per_feature_synth = hook_results_synth[1]
loss_synth_ortho = eval_losses_synth[-1]
pf_synth_ortho = per_feature_synth[-1]
print(f"  Final eval loss: {loss_synth_ortho:.6f}")

# %%
# --- Plot: Eval loss curve (TiedLinearRelu) with SynthAE baselines ---
eval_epochs = list(range(0, N_EPOCHS, EVAL_FREQ)) + [N_EPOCHS - 1]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=eval_epochs, y=eval_losses_tied, name="TiedLinearRelu", opacity=0.8)
)
fig.add_hline(
    y=loss_synth_ortho,
    line_dash="dash",
    line_color="green",
)
fig.update_layout(
    title=f"Eval Loss — SyntheticDataModel (N={N_FEATURES}, D={D_HIDDEN})",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    yaxis_type="log",
)
fig.show()

# %%
# --- Plot: Per-feature reconstruction MSE ---
final_tied = np.array(per_feature_tied[-1])

# Sort features by firing probability (most frequent first)
firing_probs = dist.firing_probabilities.cpu().numpy()
sort_idx = np.argsort(-firing_probs)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=np.arange(N_FEATURES),
        y=final_tied[sort_idx],
        name="TiedLinearRelu",
        opacity=0.7,
    )
)
fig.add_trace(
    go.Bar(
        x=np.arange(N_FEATURES),
        y=pf_synth_ortho[sort_idx],
        name="SynthAE (ortho)",
        opacity=0.7,
    )
)
fig.update_layout(
    title="Per-Feature Reconstruction MSE (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE",
    barmode="group",
)
fig.show()

# %%
# --- Plot: Features recovered (MSE < threshold) vs epoch, with SynthAE baselines ---
THRESHOLDS = [0.2, 0.5, 1.0]
COLORS = ["blue", "orange", "red"]

fig = go.Figure()
for thresh, color in zip(THRESHOLDS, COLORS):
    n_recovered_tied = [int((np.array(s) < thresh).sum()) for s in per_feature_tied]
    fig.add_trace(
        go.Scatter(
            x=eval_epochs,
            y=n_recovered_tied,
            name=f"TiedLinearRelu (τ={thresh})",
            mode="lines",
            line=dict(color=color),
        )
    )
    n_recovered_ortho = int((pf_synth_ortho < thresh).sum())
    fig.add_hline(
        y=n_recovered_ortho,
        line_dash="dash",
        line_color=color,
    )
fig.update_layout(
    title="Features Recovered (MSE < τ) Over Training",
    xaxis_title="Epoch",
    yaxis_title="# features recovered",
)
fig.show()

# %%
# --- Plot: Per-feature MSE over training (heatmap, TiedLinearRelu only) ---
arr = np.array(per_feature_tied).T[:, 1:]  # (N_FEATURES, n_eval_points), skip step 0
arr_sorted = arr[sort_idx]

fig = go.Figure(
    go.Heatmap(
        z=arr_sorted,
        x=eval_epochs[1:],
        y=np.arange(N_FEATURES),
        colorscale="tempo",
        colorbar=dict(title="MSE"),
    )
)
fig.update_layout(
    title="Per-Feature MSE Over Training — TiedLinearRelu",
    xaxis_title="Epoch",
    yaxis_title="Feature rank (most frequent → rarest)",
    height=500,
)
fig.show()

# %%
# --- W^T W comparison ---
models = [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["TiedLinearRelu", "SynthAE (ortho)"]
)

for i, (name, tm) in enumerate(
    [
        ("TiedLinearRelu", tm_tied),
        ("SynthAE (ortho)", tm_synth_ortho),
    ]
):
    W = tm.W.detach().cpu().numpy()
    WtW = W.T @ W
    fig.add_trace(
        go.Heatmap(z=WtW, colorscale="RdBu_r", zmid=0, showscale=(i == 1)),
        row=1,
        col=i + 1,
    )

fig.update_layout(title="W^T W Comparison", height=400, width=900)
fig.show()

# %%
# --- Plot: Spectrum of W W^T ---
fig = go.Figure()
for name, tm in models:
    W = tm.W.detach().cpu()
    eigvals = torch.linalg.eigvalsh(W @ W.T).flip(0).numpy()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, D_HIDDEN + 1),
            y=eigvals,
            name=name,
            mode="lines+markers",
        )
    )
fig.update_layout(
    title="Spectrum of W W^T",
    xaxis_title="Eigenvalue index",
    yaxis_title="Eigenvalue",
)
fig.show()

# %%
# --- Plot: Geometric properties comparison ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Feature Dimensionalities",
        "Feature Norms",
        "Total Interference",
    ],
)

for name, tm in models:
    fd = tm.feature_dimensionalities.detach().cpu().numpy()[sort_idx]
    fn = tm.feature_norms.detach().cpu().numpy()[sort_idx]
    ti = tm.total_feature_interferences.detach().cpu().numpy()[sort_idx]
    x = np.arange(N_FEATURES)

    fig.add_trace(go.Scatter(x=x, y=fd, name=name, mode="lines"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=fn, name=name, mode="lines", showlegend=False), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x, y=ti, name=name, mode="lines", showlegend=False), row=1, col=3
    )

fig.update_layout(
    title="Geometric Properties (sorted by firing probability)",
    height=400,
    width=1200,
)
for col in range(1, 4):
    fig.update_xaxes(title_text="Feature rank", row=1, col=col)
fig.show()

# %%
# --- Plot: Bias comparison ---
fig = go.Figure()
for name, tm in models:
    b = tm.ae.b.detach().cpu().numpy()[sort_idx]  # ty:ignore
    fig.add_trace(go.Scatter(x=np.arange(N_FEATURES), y=b, name=name, mode="lines"))
fig.update_layout(
    title="Learned Bias b (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="b",
)
fig.show()

# %% --- Plot: Feature norms² + bias (TiedLinearRelu) ---
# This plot makes slightly less sense in this place but for consistency.
fn2 = tm_tied.feature_norms.detach().cpu().numpy() ** 2
b_tied = tm_tied.ae.b.detach().cpu().numpy()  # ty:ignore
combined = (fn2 + b_tied)[sort_idx]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=np.arange(N_FEATURES), y=combined, mode="lines", name="‖w‖² + b")
)
fig.update_layout(
    title="TiedLinearRelu: ‖w‖² + b (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="‖w‖² + b",
)
fig.show()


# %%
# --- Summary statistics ---
print("\n=== Summary ===")
for name, eval_loss, pf in [
    ("TiedLinearRelu", eval_losses_tied[-1], per_feature_tied[-1]),
    ("SynthAE (ortho)", loss_synth_ortho, pf_synth_ortho),
]:
    final_mse = np.array(pf)
    recovered = "  ".join(f"τ={t}: {int((final_mse < t).sum())}" for t in THRESHOLDS)
    print(
        f"{name:25s}  eval_loss={eval_loss:.6f}  "
        f"recovered=[{recovered}]  "
        f"mean_feature_MSE={final_mse.mean():.4f}"
    )

# %% --- SAE training on both models ---
N_DICT = N_FEATURES + 4
SAE_STEPS = 65_000
SAE_BATCH = 1024
SAE_LR = 3e-4
SAE_L1 = 0.05

sae_results = {}

for name, tm in [("TiedLinearRelu", tm_tied), ("SynthAE (ortho)", tm_synth_ortho)]:
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

    # Compute metrics
    with torch.no_grad():
        test_x = dist.sample(10_000).to(DEVICE)
        test_hidden = tm.ae.encode(test_x)
        test_z = sae.encode(test_hidden)
        test_recon = sae.decode(test_z)

        # L0 sparsity: mean number of active (> 0) dict elements per sample
        l0 = (test_z > 0).float().sum(dim=-1).mean().item()

        # Dead features: dict elements that never fire
        ever_active = (test_z > 0).any(dim=0)
        n_dead = int((~ever_active).sum().item())
        n_alive = int(ever_active.sum().item())

        # Reconstruction MSE in hidden space
        recon_mse = (test_hidden - test_recon).pow(2).sum(dim=-1).mean().item()

        # Per-feature faithfulness: encode one-hot, round-trip through SAE
        eye = torch.eye(N_FEATURES, device=DEVICE)
        h_eye = tm.ae.encode(eye)
        h_eye_recon = sae.decode(sae.encode(h_eye))
        per_feat_sae_mse = (h_eye - h_eye_recon).pow(2).sum(dim=-1).cpu().numpy()

        # Explained variance ratio
        total_var = test_hidden.var(dim=0).sum().item()
        residual_var = (test_hidden - test_recon).var(dim=0).sum().item()
        explained_var = 1 - residual_var / total_var

    sae_results[name] = {
        "sae": sae,
        "losses": sae_losses,
        "l0": l0,
        "n_dead": n_dead,
        "n_alive": n_alive,
        "recon_mse": recon_mse,
        "per_feat_sae_mse": per_feat_sae_mse,
        "explained_var": explained_var,
    }
    print(
        f"  L0={l0:.1f}  Dead={n_dead}/{N_DICT}  MSE={recon_mse:.6f}  ExplVar={explained_var:.4f}"
    )

# %% --- SAE loss curves ---
fig = go.Figure()
for name, res in sae_results.items():
    fig.add_trace(go.Scatter(y=res["losses"], mode="lines", name=name, opacity=0.8))
fig.update_layout(
    title="SAE Training Loss Comparison",
    xaxis_title="Step",
    yaxis_title="Loss",
    yaxis_type="log",
)
fig.show()

# %% --- Per-feature SAE reconstruction error ---
names = list(sae_results.keys())
colors = ["blue", "green"]
fig = go.Figure()
for name in names:
    res = sae_results[name]
    fig.add_trace(
        go.Scatter(
            x=np.arange(N_FEATURES),
            y=res["per_feat_sae_mse"][sort_idx],
            name=name,
            mode="lines",
        )
    )
fig.update_layout(
    title="SAE Per-Feature Reconstruction Error (sorted by firing probability)",
    xaxis_title="Feature rank (most frequent → rarest)",
    yaxis_title="MSE (hidden space)",
)
fig.show()

# %% --- SAE activations on one-hot features (matched) ---

for name in names:
    sae = sae_results[name]["sae"]
    tm_ref = tm_tied if name == "TiedLinearRelu" else tm_synth_ortho
    with torch.no_grad():
        eye = torch.eye(N_FEATURES, device=DEVICE)
        sae_acts = (
            sae.encode(tm_ref.ae.encode(eye)).cpu().numpy()
        )  # (N_FEATURES, N_DICT)

    # Hungarian matching: maximize total activation along the matched pairs
    cost = -sae_acts  # (N_FEATURES, N_DICT)
    feat_idx, dict_idx = linear_sum_assignment(cost)

    # Reorder both rows and columns: matched pairs form the top-left diagonal,
    # then append unmatched features (rows) and unmatched dict elements (cols)
    matched_feats = set(feat_idx)
    matched_dicts = set(dict_idx)
    unmatched_feats = [f for f in range(N_FEATURES) if f not in matched_feats]
    unmatched_dicts = [d for d in range(N_DICT) if d not in matched_dicts]

    row_order = list(feat_idx) + unmatched_feats
    col_order = list(dict_idx) + unmatched_dicts

    sae_acts_matched = sae_acts[np.ix_(row_order, col_order)]
    row_labels = [f"f{f}" for f in row_order]
    col_labels = [f"d{d}" for d in col_order]

    # Compute diagonality: fraction of total activation on the matched diagonal
    n_matched = len(feat_idx)
    diag_sum = sum(sae_acts_matched[i, i] for i in range(n_matched))
    total_sum = sae_acts_matched.sum()
    diagonality = diag_sum / total_sum if total_sum > 0 else 0.0
    sae_results[name]["diagonality"] = diagonality
    print(
        f"{name}: diagonality = {diagonality:.4f} (diag_sum={diag_sum:.2f}, total={total_sum:.2f})"
    )

    px.imshow(
        sae_acts_matched,
        labels=dict(x="SAE dict element (matched)", y="Feature (matched)"),
        x=col_labels,
        y=row_labels,
        title=f"SAE one-hot activations (Hungarian matched, diag={diagonality:.3f}) — {name}",
        aspect="auto",
        color_continuous_scale="ylgnbu_r",
    ).show()

# %% --- SAE summary print ---
print("\n=== SAE Summary ===")
print(
    f"{'Model':25s}  {'MSE':>10s}  {'L0':>6s}  {'Dead':>6s}  {'Alive':>6s}  {'ExplVar':>8s}  {'Diag':>6s}"
)
for name, res in sae_results.items():
    print(
        f"{name:25s}  {res['recon_mse']:10.6f}  {res['l0']:6.1f}  "
        f"{res['n_dead']:6d}  {res['n_alive']:6d}  {res['explained_var']:8.4f}  "
        f"{res.get('diagonality', 0):6.4f}"
    )

# %% --- SAE summary comparison ---
fig = make_subplots(
    rows=1,
    cols=5,
    subplot_titles=[
        "Recon MSE",
        "Mean L0",
        "Dead Features",
        "Explained Variance",
        "Diagonality",
    ],
)

for i, (name, color) in enumerate(zip(names, colors)):
    res = sae_results[name]
    fig.add_trace(
        go.Bar(x=[name], y=[res["recon_mse"]], marker_color=color, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=[name], y=[res["l0"]], marker_color=color, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=[name], y=[res["n_dead"]], marker_color=color, showlegend=False),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=[name], y=[res["explained_var"]], marker_color=color, showlegend=False
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        go.Bar(
            x=[name],
            y=[res.get("diagonality", 0)],
            marker_color=color,
            showlegend=False,
        ),
        row=1,
        col=5,
    )

fig.update_layout(
    title=f"SAE Comparison (dict={N_DICT}, L1={SAE_L1})", height=400, width=1400
)
fig.show()

# %%
