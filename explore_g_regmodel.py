#!/usr/bin/env python
"""
explore_g_regmodel.py — EXPLORATORY, not a result. Session G's model for S56 (and S57): the window gate (LOCAL3) at layer 1
and a per-layer gate over the residual stream and a register at the later layers. It changes no existing module.

THE MODEL (RegBDH, a subclass of explore_b16_gates.LocalBDH; built as build(task, arm A, ARCH, seed) -> to_local(m, seed)
-> to_reg(m, seed, ...), so every existing parameter, the window and every batch are LOCAL3_SLOW's for the seed).
  v_t       the raw token embedding (embed(tokens)), as every gate in this project reads.
  Layer 1   LOCAL3's gate, unchanged: g^(1)_t = softmax(W_g tanh(W_in conv3(v)_t)).
  Register  R_t = R_{t-1}(1 - beta_t) + beta_t W_R v_t,  beta_t = sigmoid(w_R . v_t + b_R),  R_0 = 0 before the first token
            (R_{-1} = 0), R_t in R^16. W_R ~ N(0, 0.1^2) (16 x 32), w_R ~ N(0, 0.01^2), b_R = logit(0.1); all three from a
            generator seeded seed + REG_OFFSET.
            CAUSALITY: the gate at t reads R_{t-1}, the register as written by tokens 0..t-1 (Rprev_t = R_{t-1}, Rprev_0 = 0);
            token t's own write enters the gate only from t+1 (CHECK: the gates at t do not change when token t's register
            write changes, i.e. when v_t changes only through the register path, and do change at t+1).
  Layers l in GATED (2 and 3 by default)
            u^(l)_t = A_r LN(r^(l)_t) + A_v v_t + A_R Rprev_t,   g^(l)_t = softmax(W_g2 tanh(u^(l)_t)),
            r^(l)_t the residual stream entering layer l (bdh.BDH.forward's x at the top of its loop for that level; LN
            without affine). A_v = W_in (a copy, its own parameter) at init, so u = W_in v_t exactly at step 0 if A_r = A_R
            = 0; A_r ~ N(0, 0.01^2), A_R ~ N(0, 0.01^2) (the residual and register paths start as perturbations, and every
            new parameter receives a gradient from update 1); W_g2 = W_g (a copy) at init; A_r, A_v, A_R, W_g2 are SHARED
            by the gated layers. Read gate = write gate at every layer (as LOCAL3). Each layer's memory uses its own
            G^(l) = g^(l) g^(l)T; layer 1's is LOCAL3's.
            sg=True (variant SG): LN(r) is computed from r.detach() — no gradient from the gate into the stack.
            sg=False (NOSG): the gradient flows from the gate into the stack.
            reg=False (RESGATE_ONLY): no register (A_R and the register's parameters do not exist).
            res_layers: the gated layers whose gate reads the residual (default: all of GATED); a gated layer not in
            res_layers has A_r's term dropped (it reads v_t and the register only).
  Memory    the model's own GatedAttention (Hebbian, decay 0.95) with G set per layer; or, with mem="delta", Session F's
            explore_delta_mem.DeltaMemory (copied from claude/explore-F unchanged) given the per-layer gates.
forward returns (logits, None, g_read, g_write) with the LAST gated layer's gate as g_read = g_write (so the harness's
statistics read the layer-3 gate), and keeps every layer's gate and the register's beta in self.diag (detached) when
self.keep_diag is set.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdh import BDH
from test_multilayer_binding import D
import explore_b16_gates as g16

REG_OFFSET = 56_000
R_DIM = 16
GATED = (2, 3)
B_R0 = math.log(0.1 / 0.9)                       # logit(0.1)


class RegBDH(g16.LocalBDH):
    """See the module docstring. Created only by to_reg."""

    def register(self, v):
        """(Rprev (B,T,R_DIM), beta (B,T)): Rprev_t = R_{t-1}, the register written by tokens < t."""
        beta = torch.sigmoid(v @ self.w_R + self.b_R)                     # (B, T)
        w = v @ self.W_R.T                                                # (B, T, R_DIM)
        B, T, _ = v.shape
        R = v.new_zeros(B, R_DIM)
        prev = []
        for t in range(T):
            prev.append(R)
            b = beta[:, t:t + 1]
            R = R * (1 - b) + b * w[:, t]
        return torch.stack(prev, 1), beta

    def layer_gate(self, level, h, v, Rprev):
        u = v @ self.A_v.T
        if level in self.res_layers:
            r = h.detach() if self.sg else h
            u = u + F.layer_norm(r, (r.shape[-1],)) @ self.A_r.T
        if Rprev is not None:
            u = u + Rprev @ self.A_R.T
        return F.softmax(torch.tanh(u) @ self.W_g2.T, dim=-1)

    def forward(self, tokens, tau=None, track_sat=False):
        C = self.config
        B, T = tokens.size()
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        v = self.embed(tokens)
        g1, _ = self.local_gates(v)
        Rprev, beta = self.register(v) if self.reg else (None, None)
        gates = []
        G1 = None if self.mem == "delta" else torch.einsum("btk,bsk->bts", g1, g1).unsqueeze(1)   # one tensor, as LocalBDH
        x = self.embed(tokens).unsqueeze(1)          # a second lookup, as LocalBDH (gate) + BDH.forward (stack) do
        x = self.ln(x)
        for level in range(C.n_layer):
            lv = level + 1
            if lv in self.gated:
                g = self.layer_gate(lv, x.view(B, T, D_), v, Rprev)
            else:
                g = g1
            gates.append(g)
            if self.mem == "delta":
                self.attn.set_gates(g, g)
            else:
                self.attn.G = G1 if g is g1 else torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        logits = x.view(B, T, D_) @ self.lm_head
        if self.keep_diag:
            self.diag = dict(gates=[g.detach() for g in gates], beta=None if beta is None else beta.detach())
        gl = gates[max(self.gated) - 1] if self.gated else g1
        return logits, None, gl, gl


def to_reg(m, seed, sg=True, reg=True, gated=GATED, res_layers=None, mem="hebb"):
    """Convert a LocalBDH (explore_b16_gates.to_local) in place. New parameters are registered after every existing one,
    drawn from their own generator; no existing parameter changes."""
    assert type(m) is g16.LocalBDH, type(m)
    assert m.conv is None and not m.gate_to_readout
    m.__class__ = RegBDH
    m.sg, m.reg, m.gated = bool(sg), bool(reg), tuple(gated)
    m.res_layers = tuple(gated if res_layers is None else res_layers)
    m.mem, m.keep_diag, m.diag = mem, False, None
    g = torch.Generator().manual_seed(int(seed) + REG_OFFSET)
    hg = m.W_in.shape[0]
    with torch.no_grad():
        m.A_v = nn.Parameter(m.W_in.detach().clone())
        m.A_r = nn.Parameter(torch.randn(hg, D, generator=g) * 0.01)
        m.W_g2 = nn.Parameter(m.W_g.detach().clone())
        if m.reg:
            m.A_R = nn.Parameter(torch.randn(hg, R_DIM, generator=g) * 0.01)
            m.W_R = nn.Parameter(torch.randn(R_DIM, D, generator=g) * 0.1)
            m.w_R = nn.Parameter(torch.randn(D, generator=g) * 0.01)
            m.b_R = nn.Parameter(torch.tensor(B_R0))
    return m


GATE_NEW = ("A_v", "A_r", "W_g2", "A_R", "W_R", "w_R", "b_R")


def gate_names(m):
    """The gate group of the slow schedule: LOCAL3's gate (W_in, W_g, window) and every new gate/register parameter."""
    names = [n for n, _ in m.named_parameters()]
    return [n for n in ("W_in", "W_g", "gate_conv.conv_w") + GATE_NEW if n in names]
