"""SAE directly on the raw distribution (no AE bottleneck).

Does the SAE learn tessellation when there's no superposition?
K_SPHERES SparseSpheres (dim=1, ambient=3).
n_dict scales with K_SPHERES (11 discretizations per sphere + headroom).
"""

# %%
# IMPORTS AND CONFIG

import math
import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from occhio.distributions import SparseSpheres
from occhio.sae import SAESimple

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

K_SPHERES = 1
SPHERE_DIM = 1
AMBIENT_DIM = 3
N_FEATURES = K_SPHERES * AMBIENT_DIM
P_ACTIVE = 0.04
N_DISC = 11
N_DICT = N_DISC * K_SPHERES + 5  # 11 atoms per sphere + headroom
N_STEPS = 30_000
NOISE_STD = 0.04
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %%
# DISTRIBUTION SETUP

dist = SparseSpheres(
    n_spheres=K_SPHERES,
    sphere_dim=SPHERE_DIM,
    ambient_dim=AMBIENT_DIM,
    p_active=P_ACTIVE,
    p_infill=0.0,
    radius=4.0,
    noise_std=NOISE_STD,
    n_discretizations=N_DISC,
    generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    device=DEVICE,
)

# %%
# DATA FUNCTIONS


def data_fn_all(n):
    with torch.no_grad():
        return dist.sample(n).to(DEVICE)


def data_fn_active(n):
    collected = []
    while sum(len(c) for c in collected) < n:
        with torch.no_grad():
            x = dist.sample(n * 25).to(DEVICE)
            active = x[x.norm(dim=-1) > 0]
            if len(active) > 0:
                collected.append(active)
    return torch.cat(collected)[:n]


# %%
# VISUALIZE RAW SAMPLES IN 3D PER SPHERE
# Each sphere's 3D ambient subspace shown separately with GT circle overlay.

print("VISUALIZE RAW SAMPLES IN 3D PER SPHERE", flush=True)

with torch.no_grad():
    samples = data_fn_active(5000).cpu().numpy()

SPHERE_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]

fig = make_subplots(
    rows=1,
    cols=K_SPHERES,
    subplot_titles=[f"Sphere {k}" for k in range(K_SPHERES)],
    specs=[[{"type": "scene"}] * K_SPHERES],
    horizontal_spacing=0.02,
)

for k in range(K_SPHERES):
    s = samples[:, k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM]
    active = np.linalg.norm(s, axis=1) > 0.01
    s_active = s[active]

    fig.add_trace(
        go.Scatter3d(
            x=s_active[:, 0],
            y=s_active[:, 1],
            z=s_active[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=SPHERE_COLORS[k % len(SPHERE_COLORS)], opacity=0.5
            ),
            name=f"Sphere {k}",
        ),
        row=1,
        col=k + 1,
    )

    T = dist.tilts[k].cpu().numpy()  # (3, 2)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    centers = dist.centers[k].cpu().numpy() if hasattr(dist, "centers") else np.zeros(3)
    circle_3d = circle_2d @ T.T + centers
    fig.add_trace(
        go.Scatter3d(
            x=circle_3d[:, 0],
            y=circle_3d[:, 1],
            z=circle_3d[:, 2],
            mode="lines",
            line=dict(color="black", width=4),
            name="GT circle" if k == 0 else None,
            showlegend=(k == 0),
        ),
        row=1,
        col=k + 1,
    )

for k in range(K_SPHERES):
    sk = f"scene{k + 1}" if k > 0 else "scene"
    fig.update_layout(
        **{
            sk: dict(
                aspectmode="data",
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
            )
        }
    )

fig.update_layout(
    title="Raw samples per sphere (3D ambient space)", height=500, width=350 * K_SPHERES
)
fig.write_html(f"{FIG_DIR}/samples_3d.html")
print(f"  Saved: {FIG_DIR}/samples_3d.html", flush=True)


# %%
# EVAL HELPER


def eval_sae(sae, label=""):
    with torch.no_grad():
        x = data_fn_active(10_000).to(next(sae.parameters()).device)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        l0 = (z > 0).float().sum(1).mean().item()
        mse = ((x - x_hat) ** 2).sum(-1).mean().item()
        alive = int(((z > 0).float().mean(0) > 0).sum().item())
        b_min = sae.b_enc.min().item()
        b_max = sae.b_enc.max().item()
        fire_rates = (z > 0).float().mean(0)
        alive_rates = fire_rates[fire_rates > 0]
        max_rate = alive_rates.max().item() if len(alive_rates) > 0 else 0
        min_rate = alive_rates.min().item() if len(alive_rates) > 0 else 0
        med_rate = alive_rates.median().item() if len(alive_rates) > 0 else 0
    print(
        f"  {label:40s} L0={l0:.2f} MSE={mse:.4f} alive={alive:3d} "
        f"b=[{b_min:.2f},{b_max:.2f}] "
        f"fire=[{min_rate:.3f},{med_rate:.3f},{max_rate:.3f}]",
        flush=True,
    )
    return {"l0": l0, "mse": mse, "alive": alive}


# %%
# L1 SWEEP — TRAIN SAES AT DIFFERENT SPARSITY PENALTIES

print("L1 SWEEP", flush=True)
print(
    f"  {K_SPHERES} spheres x {N_DISC} disc = {K_SPHERES * N_DISC} atoms needed, n_dict={N_DICT}",
    flush=True,
)

best_sae = None
best_l0_gap = float("inf")
trained_saes = {}

for l1 in [0.01, 0.03, 0.1, 0.3, 0.7]:
    sae = SAESimple(n_latent=N_FEATURES, n_dict=N_DICT, l1_coef=l1, device=DEVICE)
    sae.train_sae(data_fn_active, n_steps=N_STEPS, batch_size=1024, lr=3e-4)
    result = eval_sae(sae, f"L1={l1}")
    trained_saes[l1] = sae
    gap = abs(result["l0"] - 1.0)
    if gap < best_l0_gap:
        best_l0_gap = gap
        best_sae = sae
        best_l1 = l1

print(f"\nBest L0~1 at L1={best_l1}", flush=True)


# %%
# SELECT SAE AND PRECOMPUTE PROJECTIONS ONTO EACH SPHERE'S CIRCLE PLANE
# For each dictionary atom, project its encoder/decoder onto each sphere's
# 2D circle plane via tilts[k].T. This gives the atom's angle on each circle,
# its projection magnitude, and its firing arc half-width.

print("SELECT SAE AND PRECOMPUTE PROJECTIONS", flush=True)

sae = trained_saes[0.03]

# Move everything to CPU for analysis — training is done
W_dec = sae.W_dec.detach().cpu()  # (n_dict, N_FEATURES)
W_enc = sae.W_enc.detach().cpu()  # (N_FEATURES, n_dict)
b_enc = sae.b_enc.detach().cpu()  # (n_dict,)
tilts = dist.tilts.cpu()  # (K_SPHERES, 3, 2)

dec_proj = torch.zeros(K_SPHERES, N_DICT, 2)
enc_proj = torch.zeros(K_SPHERES, N_DICT, 2)
dec_mags = torch.zeros(K_SPHERES, N_DICT)
enc_mags = torch.zeros(K_SPHERES, N_DICT)
enc_angles = torch.zeros(K_SPHERES, N_DICT)
arc_hws = torch.zeros(K_SPHERES, N_DICT)

for k in range(K_SPHERES):
    T = tilts[k]  # (3, 2)
    for j in range(N_DICT):
        dp = T.T @ W_dec[j, k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM]
        dec_proj[k, j] = dp
        dec_mags[k, j] = dp.norm()

        ep = T.T @ W_enc[k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM, j]
        enc_proj[k, j] = ep
        enc_mags[k, j] = ep.norm()
        enc_angles[k, j] = torch.atan2(ep[1], ep[0])

        en = ep.norm().item()
        if en > 1e-6:
            r = max(-1.0, min(1.0, -b_enc[j].item() / en))
            arc_hws[k, j] = math.acos(r)

primary_sphere = dec_mags.argmax(dim=0)  # which sphere each latent belongs to

with torch.no_grad():
    x_eval = data_fn_active(10_000).to(next(sae.parameters()).device)
    z_eval = sae.encode(x_eval).cpu()
    alive_mask = (z_eval > 0).float().mean(0) > 0
    alive_idx = torch.where(alive_mask)[0]

n_alive = int(alive_mask.sum().item())
per_sphere = [(primary_sphere[alive_mask] == k).sum().item() for k in range(K_SPHERES)]
print(f"Alive latents: {n_alive} / {N_DICT}", flush=True)
print(f"Latents per sphere: {per_sphere}", flush=True)


# %%
# Q1: MONOSEMANTICITY — DOES EACH LATENT RESPOND TO ONLY ONE CIRCLE?
# For each alive latent, what fraction of its decoder norm lives in its
# primary sphere's 3D subspace vs the other spheres.

print("\nQ1: MONOSEMANTICITY", flush=True)

for j in alive_idx:
    j = j.item()
    total_dec_norm = W_dec[j].norm().item()
    sphere_norms = [dec_mags[k, j].item() for k in range(K_SPHERES)]
    primary_k = primary_sphere[j].item()
    primary_frac = sphere_norms[primary_k] / (total_dec_norm + 1e-8)
    bar = "#" * int(primary_frac * 30)
    secondary = sorted(enumerate(sphere_norms), key=lambda x: -x[1])
    sec_str = " ".join(f"s{k}:{v:.2f}" for k, v in secondary[:3])
    print(
        f"  n{j:2d} -> sphere {primary_k}  {primary_frac:5.1%} {bar:30s}  [{sec_str}]",
        flush=True,
    )

mono_fracs = []
for j in alive_idx:
    j = j.item()
    total = W_dec[j].norm().item()
    primary_k = primary_sphere[j].item()
    mono_fracs.append(dec_mags[primary_k, j].item() / (total + 1e-8))

mono_fracs_t = torch.tensor(mono_fracs)
print(
    f"\n  Summary: mean primary fraction = {mono_fracs_t.mean().item():.1%}, "
    f"min = {mono_fracs_t.min().item():.1%}, "
    f">90% monosemantic: {(mono_fracs_t > 0.9).float().mean().item():.0%} of latents",
    flush=True,
)


# %%
# Q2: 3-HOTNESS — ARE LATENT VECTORS SPARSE IN THE 15D SPACE?
# Each W_dec row is N_FEATURES-D. If the latent is monosemantic, only one
# 3D block (one sphere's subspace) should carry significant weight.

print("\nQ2: 3-HOTNESS (are latent vectors sparse in {N_FEATURES}D?)", flush=True)

for j in alive_idx:
    j = j.item()
    w = W_dec[j].abs()
    total = w.sum().item()
    block_fracs = []
    for k in range(K_SPHERES):
        block = w[k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM].sum().item()
        block_fracs.append(block / (total + 1e-8))
    significant_blocks = sum(1 for f in block_fracs if f > 0.05)
    top_block = max(range(K_SPHERES), key=lambda k: block_fracs[k])
    bars = "".join(f"[{'#' * int(f * 20):20s}]" for f in block_fracs)
    print(
        f"  n{j:2d}: {significant_blocks}-hot  top=s{top_block}({block_fracs[top_block]:.0%})  {bars}",
        flush=True,
    )

sig_counts = []
for j in alive_idx:
    j = j.item()
    w = W_dec[j].abs()
    total = w.sum().item()
    n_sig = sum(
        1
        for k in range(K_SPHERES)
        if w[k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM].sum().item() / (total + 1e-8)
        > 0.05
    )
    sig_counts.append(n_sig)
sig_t = torch.tensor(sig_counts, dtype=torch.float)
print(
    f"\n  Summary: mean significant blocks = {sig_t.mean().item():.1f}, "
    f"exactly 1-hot: {(sig_t == 1).float().mean().item():.0%}",
    flush=True,
)


# %%
# Q3: TESSELLATION AND ANTIPODALITY STRUCTURE
# For each sphere: angular distribution of assigned latents, pairwise cosine
# similarities, arc half-widths. Also check antipodality — ReLU neurons are
# half-space detectors, so the SAE often learns antipodal pairs (theta and
# theta+180) to cover both halves of each circle.

print("\nQ3: TESSELLATION AND ANTIPODALITY STRUCTURE", flush=True)

for k in range(K_SPHERES):
    assigned = [j.item() for j in alive_idx if primary_sphere[j] == k]
    if len(assigned) < 2:
        print(f"\n  Sphere {k}: {len(assigned)} latents (too few)", flush=True)
        continue

    angles = torch.tensor([enc_angles[k, j].item() for j in assigned])
    sorted_angles, order = angles.sort()

    # Angular gaps between consecutive latents
    gaps = torch.diff(sorted_angles)
    wrap_gap = 2 * math.pi - sorted_angles[-1] + sorted_angles[0]
    all_gaps = torch.cat([gaps, wrap_gap.unsqueeze(0)])
    ideal_gap = 2 * math.pi / len(assigned)

    # Pairwise cosine similarity in circle plane
    projs = torch.stack([enc_proj[k, j] for j in assigned])
    projs_normed = projs / projs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = projs_normed @ projs_normed.T
    mask = ~torch.eye(len(assigned), dtype=torch.bool)
    off_diag = cos_sim[mask]

    # Arc half-widths
    hws = torch.tensor([math.degrees(arc_hws[k, j].item()) for j in assigned])

    # Antipodality check: for each latent, find the closest to 180 degrees away
    n_antipodal = 0
    for i, a in enumerate(angles):
        diffs = torch.abs(angles - a)
        # Wrap to [0, pi]
        diffs = torch.min(diffs, 2 * math.pi - diffs)
        anti_diff = torch.abs(diffs - math.pi)
        anti_diff[i] = float("inf")  # exclude self
        if anti_diff.min().item() < math.radians(30):  # within 30 deg of antipodal
            n_antipodal += 1

    print(f"\n  Sphere {k}: {len(assigned)} latents", flush=True)
    print(
        f"    Angular gaps (deg): min={all_gaps.min().item() * 180 / math.pi:.1f}  "
        f"max={all_gaps.max().item() * 180 / math.pi:.1f}  "
        f"std={all_gaps.std().item() * 180 / math.pi:.1f}  "
        f"ideal={ideal_gap * 180 / math.pi:.1f}",
        flush=True,
    )
    print(
        f"    Cosine sim: mean={off_diag.mean().item():.3f}  "
        f"min={off_diag.min().item():.3f}  max={off_diag.max().item():.3f}",
        flush=True,
    )
    print(
        f"    Arc half-widths (deg): mean={hws.mean().item():.1f}  "
        f"min={hws.min().item():.1f}  max={hws.max().item():.1f}",
        flush=True,
    )
    print(
        f"    Antipodal pairs (within 30 deg): {n_antipodal}/{len(assigned)} latents have an antipodal partner",
        flush=True,
    )

    for i, idx in enumerate(order):
        j = assigned[idx]
        gap_deg = all_gaps[i].item() * 180 / math.pi
        print(
            f"      n{j:2d}: angle={sorted_angles[i].item() * 180 / math.pi:6.1f} deg  "
            f"gap_to_next={gap_deg:5.1f} deg  arc_hw={hws[idx].item():5.1f} deg",
            flush=True,
        )


# %%
# VISUALIZATION: CIRCLES IN 3D WITH DICTIONARY SPOKES AND FIRING ARCS
# For each sphere: grey samples, black GT circle, black diamond GT discretization
# points, colored spokes (decoder direction of each assigned dictionary atom),
# and colored arcs (angular firing range from encoder direction + b_enc).

print("\nVISUALIZATION: DICTIONARY ON SPHERES (3D)", flush=True)

ATOM_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#e6beff",
    "#fffac8",
]

with torch.no_grad():
    samp = data_fn_active(3000).cpu().numpy()

centers_np = (
    dist.centers.cpu().numpy()
    if hasattr(dist, "centers")
    else np.zeros((K_SPHERES, AMBIENT_DIM))
)

fig3d = make_subplots(
    rows=1,
    cols=K_SPHERES,
    subplot_titles=[f"Sphere {k}" for k in range(K_SPHERES)],
    specs=[[{"type": "scene"}] * K_SPHERES],
    horizontal_spacing=0.02,
)

for k in range(K_SPHERES):
    T = tilts[k].cpu().numpy()
    c = centers_np[k]
    color_k = SPHERE_COLORS[k % len(SPHERE_COLORS)]

    # Samples — colored by sphere, saturated
    s = samp[:, k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM]
    active = np.linalg.norm(s, axis=1) > 0.01
    s_act = s[active]
    fig3d.add_trace(
        go.Scatter3d(
            x=s_act[:, 0],
            y=s_act[:, 1],
            z=s_act[:, 2],
            mode="markers",
            marker=dict(size=2, color=color_k, opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=k + 1,
    )

    # GT circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    circle_3d = circle_2d @ T.T + c
    fig3d.add_trace(
        go.Scatter3d(
            x=circle_3d[:, 0],
            y=circle_3d[:, 1],
            z=circle_3d[:, 2],
            mode="lines",
            line=dict(color="black", width=4),
            showlegend=False,
        ),
        row=1,
        col=k + 1,
    )

    # GT discretization points
    disc_2d = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, N_DISC, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, N_DISC, endpoint=False)),
        ],
        axis=1,
    )
    disc_3d = disc_2d @ T.T + c
    fig3d.add_trace(
        go.Scatter3d(
            x=disc_3d[:, 0],
            y=disc_3d[:, 1],
            z=disc_3d[:, 2],
            mode="markers",
            marker=dict(size=5, color="black", symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=k + 1,
    )

    # Dictionary atoms assigned to this sphere
    ci = 0
    for j in range(N_DICT):
        if not alive_mask[j] or primary_sphere[j] != k:
            continue
        color = ATOM_COLORS[ci % len(ATOM_COLORS)]
        ci += 1

        w = W_dec[j, k * AMBIENT_DIM : (k + 1) * AMBIENT_DIM].cpu().numpy()
        wnorm = np.linalg.norm(w)
        if wnorm < 1e-6:
            continue
        tip = c + (w / wnorm)
        fig3d.add_trace(
            go.Scatter3d(
                x=[c[0], tip[0]],
                y=[c[1], tip[1]],
                z=[c[2], tip[2]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
            ),
            row=1,
            col=k + 1,
        )

        hw = arc_hws[k, j].item()
        ang = enc_angles[k, j].item()
        if 0.01 < hw < math.pi - 0.01:
            arc_t = np.linspace(ang - hw, ang + hw, 60)
            arc_2d = np.stack([np.cos(arc_t), np.sin(arc_t)], axis=1)
            arc_3d = arc_2d @ T.T * 1.08 + c
            fig3d.add_trace(
                go.Scatter3d(
                    x=arc_3d[:, 0],
                    y=arc_3d[:, 1],
                    z=arc_3d[:, 2],
                    mode="lines",
                    line=dict(color=color, width=5),
                    showlegend=False,
                ),
                row=1,
                col=k + 1,
            )

for k in range(K_SPHERES):
    sk = f"scene{k + 1}" if k > 0 else "scene"
    fig3d.update_layout(
        **{
            sk: dict(
                aspectmode="data",
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
            )
        }
    )

fig3d.update_layout(
    title="Dictionary atoms on spheres (3D)", height=550, width=350 * K_SPHERES
)
fig3d.write_html(f"{FIG_DIR}/dict_3d.html")
print(f"  Saved: {FIG_DIR}/dict_3d.html", flush=True)


# %%
# VISUALIZATION: POLAR VIEW — CIRCLE + SPOKES + FIRING ARCS PER SPHERE
# Projects each dictionary atom's encoder direction onto the circle's 2D plane
# (via tilts[k].T), removing the 3D tilt. Each spoke shows the angle the atom
# responds to. Each arc shows the angular range where relu(x.w + b) > 0 for
# points on that circle. Arc half-width = arccos(-b_enc / ||proj_enc||).

print("\nVISUALIZATION: DICTIONARY POLAR VIEW", flush=True)

fig_polar = make_subplots(
    rows=1,
    cols=K_SPHERES,
    subplot_titles=[f"Sphere {k}" for k in range(K_SPHERES)],
    specs=[[{"type": "polar"}] * K_SPHERES],
)

disc_angles_deg = np.degrees(np.linspace(0, 2 * np.pi, N_DISC, endpoint=False))

for k in range(K_SPHERES):
    # Unit circle
    fig_polar.add_trace(
        go.Scatterpolar(
            r=[1.0] * 361,
            theta=np.linspace(0, 360, 361).tolist(),
            mode="lines",
            line=dict(color="black", width=3),
            showlegend=False,
        ),
        row=1,
        col=k + 1,
    )

    # GT discretization points
    fig_polar.add_trace(
        go.Scatterpolar(
            r=[1.0] * N_DISC,
            theta=disc_angles_deg.tolist(),
            mode="markers",
            marker=dict(color="black", size=8, symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=k + 1,
    )

    # Dictionary atoms sorted by angle for consistent arc stacking
    assigned = [
        (j.item(), enc_angles[k, j.item()].item())
        for j in alive_idx
        if primary_sphere[j] == k
    ]
    assigned.sort(key=lambda x: x[1])

    for ci, (j, _) in enumerate(assigned):
        color = ATOM_COLORS[ci % len(ATOM_COLORS)]
        ang_deg = math.degrees(enc_angles[k, j].item())
        hw_deg = math.degrees(arc_hws[k, j].item())

        # Spoke from center
        fig_polar.add_trace(
            go.Scatterpolar(
                r=[0, 1],
                theta=[ang_deg, ang_deg],
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"n{j}",
                showlegend=(k == 0),
            ),
            row=1,
            col=k + 1,
        )

        # Firing arc
        if 0.5 < hw_deg < 179:
            arc_t = np.linspace(ang_deg - hw_deg, ang_deg + hw_deg, max(5, int(hw_deg)))
            arc_r = 1.06 + 0.05 * ci
            fig_polar.add_trace(
                go.Scatterpolar(
                    r=[arc_r] * len(arc_t),
                    theta=arc_t.tolist(),
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                ),
                row=1,
                col=k + 1,
            )

max_arcs = max(
    (sum(1 for j in alive_idx if primary_sphere[j] == k) for k in range(K_SPHERES)),
    default=1,
)
for k in range(K_SPHERES):
    pk = f"polar{k + 1}" if k > 0 else "polar"
    fig_polar.update_layout(
        **{
            pk: dict(
                radialaxis=dict(visible=False, range=[0, 1.2 + 0.05 * max_arcs]),
                angularaxis=dict(direction="counterclockwise"),
            )
        }
    )

fig_polar.update_layout(
    title="Dictionary polar view (spokes + firing arcs)",
    height=550,
    width=350 * K_SPHERES,
)
fig_polar.write_html(f"{FIG_DIR}/dict_polar.html")
print(f"  Saved: {FIG_DIR}/dict_polar.html", flush=True)


# %%
# ACTIVATION DISTRIBUTION — HOW UNEVEN ARE ACTIVATIONS ACROSS FIRING LATENTS?
# For each sample, sort activations descending and report cumulative fraction.
# If top-1 carries most of the activation, the SAE is close to tessellation
# even if L0 > 1.

print("\nACTIVATION DISTRIBUTION", flush=True)

with torch.no_grad():
    x_ad = data_fn_active(20_000).to(next(sae.parameters()).device)
    z_ad = sae.encode(x_ad).cpu()

    sorted_acts, _ = z_ad.sort(dim=1, descending=True)
    total_act = z_ad.sum(1, keepdim=True).clamp(min=1e-8)
    cum_frac = sorted_acts.cumsum(1) / total_act

    print(f"\n{'Rank':>4}  {'mean_act':>10}  {'cum_frac':>10}  {'fire_prob':>10}")
    for rank in range(8):
        a = sorted_acts[:, rank]
        fires = (a > 0).float().mean().item()
        print(
            f"  {rank + 1:>2}    {a.mean().item():>10.4f}    "
            f"{cum_frac[:, rank].mean().item():>10.3f}    "
            f"{fires:>10.3f}",
            flush=True,
        )

    l0_per = (z_ad > 0).float().sum(1)
    print(f"\nL0 distribution:")
    for v in range(10):
        pct = (l0_per == v).float().mean().item() * 100
        if pct > 0.05:
            print(f"  L0={v}: {pct:.1f}%", flush=True)

    gt1 = l0_per > 1
    if gt1.sum() > 10:
        s, _ = z_ad[gt1].sort(dim=1, descending=True)
        top1_frac = s[:, 0] / z_ad[gt1].sum(1).clamp(min=1e-8)
        ratio_12 = s[:, 0] / s[:, 1].clamp(min=1e-8)
        print(f"\nL0>1 samples ({gt1.sum().item()}):")
        print(
            f"  top1/top2 ratio: mean={ratio_12.mean().item():.2f} median={ratio_12.median().item():.2f}"
        )
        print(
            f"  top1 >50% of total: {(top1_frac > 0.5).float().mean().item() * 100:.1f}%"
        )
        print(
            f"  top1 >80% of total: {(top1_frac > 0.8).float().mean().item() * 100:.1f}%",
            flush=True,
        )


print("\nDONE", flush=True)

# %%
