# Multi-Channel BDH (MC-BDH)
# Extends pathwaycom/bdh with k weight channels per synapse + a context gate.
# The idea: each synapse holds k weights (like k neurotransmitters). A low-dim
# context signal ("neuromodulator") blends them per forward pass, so the same
# network expresses k modes of cognition without separate experts.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    n_channels: int = 2          # k: weights per synapse (the new dimension)
    context_dim: int = 8         # size of the neuromodulator signal


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class MCBDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        k = config.n_channels

        # --- THE KEY CHANGE: a leading channel dim k on every synapse matrix ---
        # original: (nh, D, N) ; (nh*N, D)
        # now:   (k, nh, D, N) ; (k, nh*N, D)
        self.encoder = nn.Parameter(torch.zeros((k, nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((k, nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((k, nh * N, D)).normal_(std=0.02))

        # neuromodulator: maps a context vector -> mixing weights over k channels
        self.context_proj = nn.Linear(config.context_dim, k)

        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _blend(self, param, gate):
        # param: (k, ...) ; gate: (B, k) -> per-batch blended weight (B, ...)
        # einsum over the channel dim: weighted sum of the k weight tensors.
        return torch.einsum("bk,k...->b...", gate, param)

    def forward(self, idx, targets=None, context=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # context -> channel mixing weights (the neuromodulator signal).
        # default: zero context -> softmax gives uniform blend, so it degrades
        # gracefully to "average of channels" when no task signal is supplied.
        if context is None:
            context = torch.zeros(B, C.context_dim, device=idx.device)
        gate = F.softmax(self.context_proj(context), dim=-1)  # (B, k)

        enc = self._blend(self.encoder, gate)      # (B, nh, D, N)
        enc_v = self._blend(self.encoder_v, gate)  # (B, nh, D, N)
        dec = self._blend(self.decoder, gate)      # (B, nh*N, D)

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        for _ in range(C.n_layer):
            # x: (B,1,T,D) ; enc: (B,nh,D,N) -> broadcast over the head dim
            x_latent = torch.einsum("bhtd,bhdn->bhtn", x.expand(B, nh, T, D), enc)
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = torch.einsum("bhtd,bhdn->bhtn", yKV.expand(B, nh, T, D), enc_v)
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = torch.einsum(
                "btm,bmd->btd",
                xy_sparse.transpose(1, 2).reshape(B, T, N * nh),
                dec,
            ).unsqueeze(1)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    # smoke test: does it run, and do different contexts give different outputs?
    torch.manual_seed(0)
    cfg = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=16,
                    n_channels=2, context_dim=8)
    model = MCBDH(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))

    # context A = channel 0, context B = channel 1 (one-hot-ish signals)
    ctx_a = torch.zeros(4, 8); ctx_a[:, 0] = 5.0
    ctx_b = torch.zeros(4, 8); ctx_b[:, 1] = 5.0

    logits_a, loss_a = model(idx, targets, context=ctx_a)
    logits_b, loss_b = model(idx, targets, context=ctx_b)

    diff = (logits_a - logits_b).abs().mean().item()
    print(f"loss A: {loss_a.item():.4f} | loss B: {loss_b.item():.4f}")
    print(f"mean |logit_A - logit_B|: {diff:.4f}  (non-zero => channels diverge)")

    loss_a.backward()
    g = model.encoder.grad
    print(f"encoder.grad shape: {tuple(g.shape)}  (leading dim k={cfg.n_channels})")
    print("per-channel grad norms:", [round(g[i].norm().item(), 4) for i in range(cfg.n_channels)])
    print("OK" if diff > 1e-4 else "WARN: channels not diverging")
