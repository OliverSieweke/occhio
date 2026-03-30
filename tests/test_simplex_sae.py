"""Tests for SimplexSAE."""

import torch
import pytest
from occhio.sae import SimplexSAE


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, K, d_local, B",
    [
        (10, 3, 4, 32),
        (5, 1, 5, 16),
        (20, 5, 8, 64),
        (8, 2, [3, 5], 10),
    ],
)
def test_forward_shapes(d, K, d_local, B):
    sae = SimplexSAE(d_input=d, n_simplexes=K, d_local=d_local)
    x = torch.randn(B, d)
    x_hat, s, g = sae.forward(x)

    d_dict = sum(sae.d_locals)
    assert x_hat.shape == (B, d)
    assert s.shape == (B, d_dict)
    assert g.shape == (B, K)


# ---------------------------------------------------------------------------
# 2. Gate independence — sigmoid outputs in [0,1], don't sum to 1
# ---------------------------------------------------------------------------


def test_gate_independence():
    sae = SimplexSAE(d_input=10, n_simplexes=4, d_local=6)
    x = torch.randn(64, 10)
    g = sae.get_gates(x)

    assert (g >= 0).all() and (g <= 1).all(), "Gates must be in [0, 1]"
    # With 4 independent sigmoids, row sums should NOT all be ~1
    row_sums = g.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1), (
        "Gate sums should not all equal 1 (they are independent sigmoids, not softmax)"
    )


# ---------------------------------------------------------------------------
# 3. Column norm constraint
# ---------------------------------------------------------------------------


def test_column_norm_after_step():
    sae = SimplexSAE(d_input=8, n_simplexes=2, d_local=5, lambda_2=1e-2)
    x = torch.randn(32, 8)

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    optimizer.zero_grad()
    x_hat, s, g = sae.forward(x)
    loss = sae.compute_loss(x, x_hat, s, g)
    loss.backward()
    optimizer.step()
    sae.constrain_weights()

    for k in range(sae.n_simplexes):
        col_norms = sae.D[k].data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-6), (
            f"D[{k}] columns should be unit norm after constrain_weights"
        )


# ---------------------------------------------------------------------------
# 4. Gradient flow to all parameters
# ---------------------------------------------------------------------------


def test_gradient_flow():
    sae = SimplexSAE(d_input=6, n_simplexes=2, d_local=4)
    x = torch.randn(16, 6)

    x_hat, s, g = sae.forward(x)
    loss = sae.compute_loss(x, x_hat, s, g)
    loss.backward()

    for name, param in sae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# 5. encode/decode roundtrip matches forward
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    sae = SimplexSAE(d_input=10, n_simplexes=3, d_local=4)
    x = torch.randn(16, 10)

    x_hat_fwd, _, _ = sae.forward(x)
    z = sae.encode(x)
    x_hat_rt = sae.decode(z)

    assert torch.allclose(x_hat_fwd, x_hat_rt, atol=1e-6), (
        "decode(encode(x)) must equal x_hat from forward"
    )


# ---------------------------------------------------------------------------
# 6. K=1 behaves like a gated single-dictionary SAE
# ---------------------------------------------------------------------------


def test_k1_single_simplex():
    """With K=1, the model should reconstruct reasonably on centred data."""
    torch.manual_seed(42)
    d = 6
    sae = SimplexSAE(d_input=d, n_simplexes=1, d_local=d, lambda_1=0.0, lambda_2=1e-4)

    # Centre data at origin so gate ≈ 0.5 (b_d dot x ≈ 0 + c_k=0 → σ(0)=0.5)
    x = torch.randn(256, d) * 0.1
    x_hat, s, g = sae.forward(x)

    # Gate should be roughly 0.5 for small zero-centred data
    assert g.mean().item() == pytest.approx(0.5, abs=0.2)


# ---------------------------------------------------------------------------
# 7. Variable d_local per simplex
# ---------------------------------------------------------------------------


def test_variable_d_local():
    d_locals = [3, 5, 7]
    sae = SimplexSAE(d_input=10, n_simplexes=3, d_local=d_locals)

    assert sae.d_dict == 15
    x = torch.randn(8, 10)
    x_hat, s, g = sae.forward(x)
    assert s.shape == (8, 15)

    latents = sae.get_local_latents(x)
    assert len(latents) == 3
    for k, lat in enumerate(latents):
        assert lat.shape == (8, d_locals[k])


# ---------------------------------------------------------------------------
# 8. initialize_from_data sets vantage points near cluster centroids
# ---------------------------------------------------------------------------


def test_initialize_from_data():
    torch.manual_seed(0)
    K = 3
    d = 8
    # Create 3 tight clusters
    centres = torch.randn(K, d) * 5
    data = torch.cat(
        [centres[k].unsqueeze(0) + torch.randn(100, d) * 0.1 for k in range(K)]
    )

    sae = SimplexSAE(d_input=d, n_simplexes=K, d_local=4)
    sae.initialize_from_data(data)

    learned = torch.stack([sae.b_dec[k].data for k in range(K)])

    # Each true centre should be close to some learned vantage (up to permutation)
    dists = torch.cdist(centres, learned)
    min_dists = dists.min(dim=1).values
    assert (min_dists < 1.0).all(), (
        f"Vantage points should be near true cluster centres, got min dists {min_dists}"
    )


# ---------------------------------------------------------------------------
# 9. Synthetic simplex smoke test — loss converges
# ---------------------------------------------------------------------------


def test_synthetic_simplex_convergence():
    torch.manual_seed(7)
    K_true = 3
    d = 10
    d_k = 4

    # Ground-truth vantage points
    vantage = torch.randn(K_true, d)

    # Ground-truth dictionary directions (unit norm columns)
    dicts = []
    for _ in range(K_true):
        D = torch.randn(d, d_k)
        D = D / D.norm(dim=0, keepdim=True)
        dicts.append(D)

    def sample_fn(n):
        xs = []
        for _ in range(n):
            k = torch.randint(K_true, (1,)).item()
            coeffs = torch.relu(torch.randn(d_k)) * 0.5
            xs.append(vantage[k] + dicts[k] @ coeffs)
        return torch.stack(xs)

    sae = SimplexSAE(
        d_input=d,
        n_simplexes=K_true,
        d_local=d_k,
        lambda_1=1e-4,
        lambda_2=1e-4,
    )
    sae.initialize_from_data(sample_fn(500))

    losses = sae.train_sae(sample_fn, n_steps=2000, batch_size=128, lr=1e-3)

    # Loss should decrease substantially
    early = sum(losses[:100]) / 100
    late = sum(losses[-100:]) / 100
    assert late < early * 0.5, (
        f"Loss should converge: early={early:.4f}, late={late:.4f}"
    )
