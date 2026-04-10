"""
Implements Sparse AutoEncoders.
"""

from torch import Tensor
from torch.optim import AdamW
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SparseAutoEncoderBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """features --> latent"""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """latent --> features"""

    @abstractmethod
    def resample_weights(self):
        """Reset / resample all weights"""

    def constrain_weights(self) -> None:
        """Apply post-step weight constraints (e.g. unit-norm decoder rows)."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        """
        The standard SAE loss function
        """
        sparsity_loss = torch.mean(torch.sum(torch.abs(intermediate), dim=-1))
        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))
        return self.l1_coef * sparsity_loss + mse_loss

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
        sample_every: int = 25,
    ) -> list[float]:
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        optimizer = AdamW(self.parameters(), lr=lr)
        sae_device = next(self.parameters()).device
        loss_buffer = torch.empty(n_steps, device=sae_device)

        raw_buffer: Tensor | None = None

        for step in range(n_steps):
            buf_offset = step % sample_every
            if buf_offset == 0:
                steps_left = min(sample_every, n_steps - step)
                total_samples = steps_left * batch_size
                raw_buffer = data_fn(total_samples).detach()

            assert raw_buffer is not None
            start = buf_offset * batch_size
            end = start + batch_size
            x = raw_buffer[start:end]

            optimizer.zero_grad()
            x_hat, z = self.forward(x)
            if hasattr(self, "_update_dead_latent_tracker"):
                self._update_dead_latent_tracker(z)  # ty:ignore
            loss = self.loss(x, x_hat, z)
            loss.backward()
            optimizer.step()
            self.constrain_weights()

            loss_buffer[step] = loss.detach()
            if (step + 1) % 5000 == 0:
                print(f"  SAE step {step + 1}/{n_steps}  loss={loss.item():.4f}")

        return loss_buffer.cpu().tolist()

    def __init__(
        self,
        l1_coef: float = 0.5,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.l1_coef = l1_coef
        self.device = device
        self.generator = generator


class SAESimple(SparseAutoEncoderBase):
    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        dec_bias: bool = False,
        aux_k: bool = False,
        aux_coef: float = 1 / 32,
        dead_latent_window: int = 1000,
        ortho_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict
        self.dec_bias = dec_bias
        self.aux_k = aux_k
        self.aux_coef = aux_coef
        self.dead_latent_window = dead_latent_window
        self.ortho_coef = ortho_coef

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))
        self.b_dec = nn.Parameter(torch.zeros(n_latent))

        # Dead latent tracking (not a parameter, survives .to(device))
        self.register_buffer("steps_since_fired", torch.zeros(n_dict, dtype=torch.long))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        self.constrain_weights()

    def constrain_weights(self) -> None:
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def _encode_pre_act(self, x: Tensor) -> Tensor:
        """Encoder pre-activations (before ReLU)."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, z: Tensor) -> Tensor:
        if self.dec_bias:
            return z @ self.W_dec + self.b_dec
        else:
            return z @ self.W_dec

    def _auxk_loss(self, x: Tensor, x_hat: Tensor, hidden_pre: Tensor) -> Tensor:
        """AuxK auxiliary loss for dead latent revival (Gao et al., 2024).

        Dead latents (those that haven't fired in ``dead_latent_window`` steps)
        are asked to reconstruct the residual error from the main forward pass.
        """
        dead_mask = self.steps_since_fired >= self.dead_latent_window
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return x.new_tensor(0.0)

        k_aux = self.n_latent // 2
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Among dead latents, pick the top-k_aux by pre-activation magnitude
        dead_pre = torch.where(dead_mask[None], hidden_pre, torch.tensor(-torch.inf))
        topk = dead_pre.topk(k_aux, dim=-1)
        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, topk.indices, topk.values.relu())

        # Dead latents reconstruct the residual
        residual = (x - x_hat).detach()
        recon = auxk_acts @ self.W_dec
        return scale * (recon - residual).pow(2).sum(dim=-1).mean()

    def _ortho_loss(self) -> Tensor:
        """Penalize off-diagonal cosine similarity of decoder rows.

        W_dec rows are already unit-normalized (via ``constrain_weights``),
        so ``W_dec @ W_dec.T`` gives the Gram (cosine-similarity) matrix.
        We penalize the Frobenius norm of the off-diagonal entries.
        """
        gram = self.W_dec @ self.W_dec.T
        off_diag = gram - torch.diag(gram.diag())
        return off_diag.pow(2).sum() / self.n_dict

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        base_loss = super().loss(x_true, x_hat, intermediate)

        if self.aux_k and self.training:
            hidden_pre = self._encode_pre_act(x_true)
            aux_loss = self._auxk_loss(x_true, x_hat, hidden_pre)
            base_loss = base_loss + self.aux_coef * aux_loss

        if self.ortho_coef > 0:
            base_loss = base_loss + self.ortho_coef * self._ortho_loss()

        return base_loss

    @torch.no_grad()
    def _update_dead_latent_tracker(self, z: Tensor) -> None:
        """Update steps-since-fired counter based on current batch activations."""
        fired = (z > 0).any(dim=0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0


class TopKIgnoreSAE(SparseAutoEncoderBase):
    """
    SAE where the L1 sparsity penalty ignores the top-k activations per sample.
    By not penalizing the strongest activations, this reduces the incentive for
    feature absorption (where one neuron absorbs multiple features to lower L1).
    """

    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        k: int = 2,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)

        self.n_latent = n_latent
        self.n_dict = n_dict
        self.k = k

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))

        self.resample_weights()

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)
        self.constrain_weights()

    def constrain_weights(self) -> None:
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        abs_acts = torch.abs(intermediate)

        # Zero out the top-k activations so they don't contribute to L1
        _, topk_indices = torch.topk(abs_acts, self.k, dim=-1)
        mask = torch.ones_like(abs_acts)
        mask.scatter_(-1, topk_indices, 0.0)

        sparsity_loss = torch.mean(torch.sum(abs_acts * mask, dim=-1))
        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))
        return self.l1_coef * sparsity_loss + mse_loss


class SimplexSAE(SparseAutoEncoderBase):
    """Sparse AutoEncoder with K local dictionaries centered at learned vantage points.

    Each simplex has its own vantage point b_d^(k), encoder V^(k), decoder
    dictionary D^(k), and an independent sigmoid gate g_k.  The reconstruction
    is a *superposition* of local decompositions — multiple simplexes can be
    active simultaneously.

    A standard SAE is the special case K=1 with gate always ~1.

    The ``encode``/``decode`` interface packs gate values onto the sparse latent:
    ``encode`` returns ``[s^(1), …, s^(K), g_1, …, g_K]`` in R^{d_dict + K}
    and ``decode`` splits off the last K entries as gates.
    """

    def __init__(
        self,
        d_input: int,
        n_simplexes: int,
        d_local: int | list[int],
        lambda_1: float = 1e-3,
        lambda_2: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_input = d_input
        self.n_simplexes = n_simplexes
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        if isinstance(d_local, int):
            self.d_locals = [d_local] * n_simplexes
        else:
            if len(d_local) != n_simplexes:
                raise ValueError(
                    f"d_local list length ({len(d_local)}) must match "
                    f"n_simplexes ({n_simplexes})"
                )
            self.d_locals = list(d_local)

        self.d_dict = sum(self.d_locals)

        self._init_parameters()

    # -- Parameter initialization ------------------------------------------------

    def _init_parameters(self) -> None:
        dev = self.device
        gen = self.generator
        d = self.d_input

        self.b_dec = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        self.V = nn.ParameterList()
        self.D = nn.ParameterList()
        self.c = nn.ParameterList()

        for k in range(self.n_simplexes):
            d_k = self.d_locals[k]

            self.b_dec.append(
                nn.Parameter(torch.randn(d, device=dev, generator=gen) / (d**0.5))
            )
            self.b_enc.append(nn.Parameter(torch.zeros(d_k, device=dev)))
            self.V.append(
                nn.Parameter(torch.randn(d_k, d, device=dev, generator=gen) / (d**0.5))
            )

            D_k = torch.randn(d, d_k, device=dev, generator=gen) / (d_k**0.5)
            D_k = D_k / D_k.norm(dim=0, keepdim=True)
            self.D.append(nn.Parameter(D_k))

            self.c.append(nn.Parameter(torch.zeros(1, device=dev)))

    def resample_weights(self) -> None:
        self._init_parameters()

    def constrain_weights(self) -> None:
        """Normalize columns of each D^(k) to unit norm."""
        for k in range(self.n_simplexes):
            self.D[k].data = self.D[k].data / self.D[k].data.norm(dim=0, keepdim=True)

    def initialize_from_data(self, data: Tensor) -> None:
        """Set vantage points b_d^(k) via k-means on *data*."""
        K = self.n_simplexes
        n = data.shape[0]

        indices = torch.randperm(n, generator=self.generator)[:K]
        centroids = data[indices].clone()

        for _ in range(100):
            dists = torch.cdist(data, centroids)
            assignments = dists.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        with torch.no_grad():
            for k in range(K):
                self.b_dec[k].copy_(centroids[k])

    # -- Forward pass ------------------------------------------------------------

    def get_gates(self, x: Tensor) -> Tensor:
        """Gate activations g in R^{B x K}."""
        gates = []
        for k in range(self.n_simplexes):
            g_k = torch.sigmoid(x @ self.b_dec[k] + self.c[k])
            gates.append(g_k)
        return torch.stack(gates, dim=-1)

    def get_local_latents(self, x: Tensor) -> list[Tensor]:
        """Per-simplex latents [s^(1), …, s^(K)]."""
        g = self.get_gates(x)
        latents = []
        for k in range(self.n_simplexes):
            residual = x - self.b_dec[k]
            pre_act = residual @ self.V[k].T + self.b_enc[k]
            s_k = g[:, k : k + 1] * torch.relu(pre_act)
            latents.append(s_k)
        return latents

    def encode(self, x: Tensor) -> Tensor:
        """Returns [s^(1), …, s^(K), g_1, …, g_K] in R^{d_dict + K}."""
        g = self.get_gates(x)
        parts: list[Tensor] = []
        for k in range(self.n_simplexes):
            residual = x - self.b_dec[k]
            pre_act = residual @ self.V[k].T + self.b_enc[k]
            s_k = g[:, k : k + 1] * torch.relu(pre_act)
            parts.append(s_k)
        parts.append(g)
        return torch.cat(parts, dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct from [s | g] latent (last K entries are gates)."""
        K = self.n_simplexes
        g = z[:, -K:]
        s = z[:, :-K]

        x_hat = torch.zeros(z.shape[0], self.d_input, device=z.device)
        offset = 0
        for k in range(K):
            d_k = self.d_locals[k]
            s_k = s[:, offset : offset + d_k]
            x_hat = x_hat + s_k @ self.D[k].T + g[:, k : k + 1] * self.b_dec[k]
            offset += d_k
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # ty:ignore
        """Returns (x_hat, s, g)."""
        g = self.get_gates(x)

        latents = []
        for k in range(self.n_simplexes):
            residual = x - self.b_dec[k]
            pre_act = residual @ self.V[k].T + self.b_enc[k]
            s_k = g[:, k : k + 1] * torch.relu(pre_act)
            latents.append(s_k)
        s = torch.cat(latents, dim=-1)

        x_hat = torch.zeros_like(x)
        offset = 0
        for k in range(self.n_simplexes):
            d_k = self.d_locals[k]
            s_k = s[:, offset : offset + d_k]
            x_hat = x_hat + s_k @ self.D[k].T + g[:, k : k + 1] * self.b_dec[k]
            offset += d_k

        return x_hat, s, g

    # -- Loss --------------------------------------------------------------------

    def compute_loss(self, x: Tensor, x_hat: Tensor, s: Tensor, g: Tensor) -> Tensor:
        """Full loss: reconstruction + gate sparsity + within-simplex sparsity."""
        recon = torch.mean(torch.sum(torch.square(x - x_hat), dim=-1))
        gate_sparse = torch.mean(torch.sum(g, dim=-1))
        latent_sparse = torch.mean(torch.sum(torch.abs(s), dim=-1))
        return recon + self.lambda_1 * gate_sparse + self.lambda_2 * latent_sparse

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:
        """SparseAutoEncoderBase-compatible loss (intermediate = [s | g])."""
        K = self.n_simplexes
        g = intermediate[:, -K:]
        s = intermediate[:, :-K]
        return self.compute_loss(x_true, x_hat, s, g)

    # -- Training ----------------------------------------------------------------

    def train_sae(
        self,
        data_fn,
        n_steps: int = 10_000,
        batch_size: int = 1024,
        lr: float = 3e-4,
        sample_every: int = 25,
    ) -> list[float]:
        """Train loop using the 3-tuple forward."""
        if sample_every < 1:
            raise ValueError(f"sample_every must be positive, got {sample_every}")

        optimizer = AdamW(self.parameters(), lr=lr)
        sae_device = next(self.parameters()).device
        loss_buffer = torch.empty(n_steps, device=sae_device)
        raw_buffer: Tensor | None = None

        for step in range(n_steps):
            buf_offset = step % sample_every
            if buf_offset == 0:
                steps_left = min(sample_every, n_steps - step)
                total_samples = steps_left * batch_size
                raw_buffer = data_fn(total_samples).detach()

            assert raw_buffer is not None
            start = buf_offset * batch_size
            end = start + batch_size
            x = raw_buffer[start:end]

            optimizer.zero_grad()
            x_hat, s, g = self.forward(x)
            cur_loss = self.compute_loss(x, x_hat, s, g)
            cur_loss.backward()
            optimizer.step()
            self.constrain_weights()

            loss_buffer[step] = cur_loss.detach()
            if (step + 1) % 5000 == 0:
                print(f"  SAE step {step + 1}/{n_steps}  loss={cur_loss.item():.4f}")

        return loss_buffer.cpu().tolist()


class CausalSAE(SparseAutoEncoderBase):
    def __init__(
        self,
        n_latent: int,
        n_dict: int,
        l1_coef: float = 0.1,
        l1_dirc: float = 0.01,
        l1_causal: float = 0.00,
        **kwargs,
    ):
        super().__init__(l1_coef, **kwargs)
        self.l1_causal = l1_causal
        self.l1_dirc = l1_dirc

        self.n_latent = n_latent
        self.n_dict = n_dict

        self.W_enc = nn.Parameter(torch.empty((n_latent, n_dict)))
        self.b_enc = nn.Parameter(torch.zeros(n_dict))

        self.W_dec = nn.Parameter(torch.empty((n_dict, n_latent)))
        self.causal = nn.Parameter(torch.zeros((n_dict, n_dict)))

        self.resample_weights()

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec

    def resample_weights(self):
        nn.init.xavier_normal_(self.W_enc, generator=self.generator)
        nn.init.xavier_normal_(self.W_dec, generator=self.generator)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.causal)
        self.constrain_weights()

    def constrain_weights(self) -> None:
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def loss(self, x_true: Tensor, x_hat: Tensor, intermediate: Tensor) -> Tensor:

        residual = intermediate - intermediate @ torch.triu(self.causal, 1)
        sparsity_loss = torch.mean(
            torch.sum(
                torch.abs(residual),
                dim=-1,
            )
        )
        direct_sparsity = torch.mean(torch.sum(torch.abs(intermediate), dim=-1))

        mse_loss = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=-1))

        causal_loss = torch.mean(torch.abs(self.causal))
        return (
            mse_loss
            + self.l1_coef * sparsity_loss
            + self.l1_dirc * direct_sparsity
            + self.l1_causal * causal_loss
        )
