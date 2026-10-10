#!/usr/bin/env python
"""
explore_b16_gates.py — EXPLORATORY, not a result. Batch 16's model-side helpers, shared by its child processes
(S43 on this branch's modules; S44-S47 on the main line's modules at explore_main9c.MAIN_SHA). It imports only
modules that are the same on both trees (bdh, test_multilayer_binding, test_instrument_v2), so a child of either
kind can use it. Nothing here draws from the global RNG.

LOCAL3 (batch 1's S1, explore_local_gate.LocalGateBDH): the gate from a width-3 depthwise causal convolution of the
raw embeddings, no recurrence,
    u_t = sum_{j<3} w_j * v_{t-j} (zero-padded),  h_t = tanh(W_in u_t),  g_t = softmax(W_g h_t),
read gate = write gate. to_local(m, seed) converts a built MultiBDH in place: the class becomes LocalBDH
(LocalGateBDH.forward line for line), the window is drawn exactly as batch 1's (its own generator seeded
seed + 31_337, uniform +-1/sqrt(3), no bias) and registered after every existing parameter; W_h stays in the
model, unused, frozen (requires_grad False) and outside every optimizer group. No other parameter changes.

RESERVOIR (S45): to_res(m) rescales the recurrent gate's W_h to spectral radius 0.5 (float64 eigenvalues, the
product cast back to float32) and freezes it.

DECODER (S38's): a nearest-class-mean decoder (Euclidean, one mean per stream), fit on the probe's first half of
sequences and scored on the second half (the halves share no sequence), on S38's probe (512 sequences of the
configuration's task from torch.Generator seeded 38_000). Positions: CTX 3j, KEY 3j+1, VAL 3j+2 (j = 0 .. S*P-1,
the query excluded); label = the stream token at 3j. Features: the gate state h_t and the gate's input
pre-activation u_t (recurrent gate: W_in v_t [+ W_prev v_(t-1) for S40's GATE_PREV]; LOCAL3: W_in conv3(v)_t),
recomputed from the model's own parameters; the read gate recomputed from h is compared with the model's.
"""

import math

import torch
import torch.nn.functional as F

from bdh import BDH
from test_multilayer_binding import MultiBDH, CausalConv, D
from test_instrument_v2 import TAU_END

PROBE_N, PROBE_SEED = 512, 38_000           # S38's probe
WIDTH, GCONV_OFFSET = 3, 31_337             # batch 1's LOCAL3 window (explore_local_gate.WIDTH, GCONV_OFFSET)
RES_RHO = 0.5
POS = (("ctx", 0), ("key", 1), ("val", 2))


# ── LOCAL3 ───────────────────────────────────────────────────────────────────
class LocalBDH(MultiBDH):
    """MultiBDH with LOCAL3's gate: explore_local_gate.LocalGateBDH's local_gates and forward, line for line."""

    def local_gates(self, v):
        u = self.gate_conv(v)                                         # (B, T, D), causal
        h = torch.tanh(u @ self.W_in.T)
        g = F.softmax(h @ self.W_g.T, dim=-1)
        return g, g

    def forward(self, tokens, tau=None, track_sat=False):
        gr, gw = self.local_gates(self.embed(tokens))
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        if self.conv is None:
            logits, _ = BDH.forward(self, tokens)
        else:
            logits = self.forward_conv(tokens)
        return logits, None, gr, gw


def window_init(seed):
    """Batch 1's window for a seed: uniform +-1/sqrt(WIDTH) from a generator seeded seed + GCONV_OFFSET."""
    g = torch.Generator().manual_seed(int(seed) + GCONV_OFFSET)
    return (torch.rand(D, WIDTH, generator=g) * 2 - 1) * (1.0 / math.sqrt(WIDTH))


def to_local(m, seed):
    assert type(m) is MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
    assert not m.gate_to_readout and not m.gate_ln and m.gate_noise == 0.0
    m.__class__ = LocalBDH
    m.local = True
    m.gate_conv = CausalConv(D, WIDTH)
    with torch.no_grad():
        m.gate_conv.conv_w.copy_(window_init(seed))
    m.W_h.requires_grad_(False)
    return m


def freeze_wh(m):
    m.W_h.requires_grad_(False)
    return m


def is_local(m):
    return bool(getattr(m, "local", False)) and getattr(m, "gate_conv", None) is not None


# ── Reservoir ────────────────────────────────────────────────────────────────
def rho(W):
    return float(torch.linalg.eigvals(W.detach().double()).abs().max())


def sigma_max(W):
    return float(torch.linalg.matrix_norm(W.detach().double(), 2))


def to_res(m, target=RES_RHO):
    assert type(m) is MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
    r0, s0 = rho(m.W_h), sigma_max(m.W_h)
    with torch.no_grad():
        m.W_h.copy_((m.W_h.detach().double() * (target / r0)).float())
    m.W_h.requires_grad_(False)
    m.res_init = dict(rho0=r0, sigma0=s0, rho=rho(m.W_h), sigma=sigma_max(m.W_h))
    return m


# ── Decoder ──────────────────────────────────────────────────────────────────
def probe_data(task):
    return task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]


def ncm(Xtr, ytr, Xte, yte, n_cls):
    """S38's nearest-class-mean accuracy (explore_gate_state_child.ncm, line for line; CHECK in S47)."""
    means, ok = [], []
    for c in range(n_cls):
        sel = ytr == c
        ok.append(bool(sel.any()))
        means.append(Xtr[sel].mean(0) if sel.any() else torch.full((Xtr.shape[1],), float("inf")))
    M = torch.stack(means)
    d = torch.cdist(Xte, M)
    return float((d.argmin(1) == yte).float().mean())


@torch.no_grad()
def gate_state(m, x):
    """(H, U, diff): the gate state and the gate's input pre-activation at every position, and max |softmax(H W_g^T)
    - the model's read gate|."""
    was = m.training
    m.eval()
    try:
        v = m.embed(x)
        if is_local(m):
            U = m.gate_conv(v) @ m.W_in.T
            H = torch.tanh(U)
        else:
            assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
            prev = getattr(m, "W_prev", None)
            h = torch.zeros(v.shape[0], m.h_gate, dtype=v.dtype)
            vp = torch.zeros(v.shape[0], v.shape[-1], dtype=v.dtype)
            hs, us = [], []
            for t in range(v.shape[1]):
                u = v[:, t] @ m.W_in.T
                if prev is not None:
                    u = u + vp @ prev.T
                h = torch.tanh(u + h @ m.W_h.T)
                hs.append(h)
                us.append(u)
                vp = v[:, t]
            H, U = torch.stack(hs, 1), torch.stack(us, 1)
        gr = m(x, TAU_END)[2]
    finally:
        m.train(was)
    return H, U, float((torch.softmax(H @ m.W_g.T, -1) - gr).abs().max())


def halves(B):
    """The decoder's fit and score halves (sequence indices)."""
    return torch.arange(B // 2), torch.arange(B // 2, B)


def decode(m, task, data):
    x = data[:, :-1]
    S, P = task.S, task.P
    j = torch.arange(S * P)
    lab = x[:, 3 * j]
    assert int(lab.min()) >= 0 and int(lab.max()) < S, "stream tokens are 0 .. S-1"
    H, U, diff = gate_state(m, x)
    fit, score = halves(x.shape[0])

    def acc(F_):
        return ncm(F_[fit].reshape(-1, F_.shape[-1]), lab[fit].reshape(-1),
                   F_[score].reshape(-1, F_.shape[-1]), lab[score].reshape(-1), S)
    out = {}
    for name, off in POS:
        out[f"h_{name}"] = acc(H[:, 3 * j + off])
        out[f"u_{name}"] = acc(U[:, 3 * j + off])
    out.update(gate_recompute_diff=diff, chance=1.0 / S)
    return out
