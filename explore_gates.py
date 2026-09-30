#!/usr/bin/env python
"""
explore_gates.py — EXPLORATORY, not a result. Batch 3's two new channel gates, as MultiBDH
subclasses. Both replace only how the gate is computed; the BDH stack, the channels and the
training are arm A's.

far_EMA / EMA_near — a fading-memory gate with several time scales:
    u_t = W_in v_t
    c_t = a * c_{t-1} + (1 - a) * u_t          per unit, c_{-1} = 0, a = sigmoid(theta)
    g_t = softmax(W_g tanh(c_t))
  theta (one per unit, learnable) is initialized so the 32 units' half-lives are log-spaced from
  1 to 32 tokens: h_k = 32^(k/31), a = 2^(-1/h). Which unit gets which half-life is a random
  permutation drawn from the gate's own generator (seed + EMA_OFFSET); nothing else is drawn.

far_SEL / SEL_near — a selective recurrence (minGRU-style, Feng et al. 2024, "Were RNNs all we
needed?"):
    z_t = sigmoid(W_z v_t + b_z)                 b_z = -1 at init
    h_t = (1 - z_t) * h_{t-1} + z_t * (W_in v_t)  h_{-1} = 0
    g_t = softmax(W_g tanh(h_t))
  W_z (32 x D, randn * 0.1, the scale of arm A's gate matrices) is drawn from the gate's own
  generator (seed + SEL_OFFSET). The forget rate depends on the token, so a context token can
  overwrite the state and other tokens can leave it alone.

In both, v_t is the raw token embedding (arm A's gate input), W_in and W_g are arm A's (drawn by
MultiBDH exactly as for arm A, so every shared parameter is bitwise arm A's for the seed), and W_h
exists but is unused (no gradient; Adam skips it). Read and write gates are the same, as in arm A.

IMPLEMENTATION. Both recurrences are computed in closed form over all positions at once (a lower-
triangular weight per unit; for SEL the weights come from cumulative sums of log(1 - z), masked
before the exp so future positions get weight exactly 0), in batch chunks of CHUNK sequences.
reference_loop() is the explicit per-step loop, written independently, that the CHECKs compare
against.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_common as ec
from test_multilayer_binding import MultiBDH, BDH, D, build
from test_channel_binding import ARCH

EMA_OFFSET, SEL_OFFSET = 51_515, 62_626
HALF_MAX = 32.0
B_Z0 = -1.0
CHUNK = 256
KINDS = ("ema", "sel")


def log_spaced_half_lives(n):
    return HALF_MAX ** (torch.arange(n, dtype=torch.float64) / (n - 1))


class GateVariantBDH(MultiBDH):
    def __init__(self, *args, kind=None, variant_seed=0, **kw):
        super().__init__(*args, **kw)                    # arm A's parameters, same RNG draws
        assert kind in KINDS
        assert self.gate_kind == "recurrent" and not self.gate_to_readout
        assert not self.gate_ln and self.gate_noise == 0.0
        self.kind = kind
        hg = self.h_gate
        g = torch.Generator().manual_seed(int(variant_seed))
        if kind == "ema":
            h = log_spaced_half_lives(hg)
            a = 2.0 ** (-1.0 / h)
            theta = torch.log(a) - torch.log1p(-a)       # logit(a), float64 then float32
            perm = torch.randperm(hg, generator=g)
            self.theta = nn.Parameter(theta[perm].float().clone())
        else:
            self.W_z = nn.Parameter(torch.randn(hg, D, generator=g) * 0.1)
            self.b_z = nn.Parameter(torch.full((hg,), B_Z0))

    # ── the gate, closed form ────────────────────────────────────────────────
    def _state(self, v):
        u = v @ self.W_in.T                                            # (B, T, H)
        T = v.shape[1]
        t = torch.arange(T)
        lag = t[:, None] - t[None, :]                                  # (T, S): t - s
        mask = lag >= 0
        if self.kind == "ema":
            la = F.logsigmoid(self.theta)                              # log a
            l1a = F.logsigmoid(-self.theta)                            # log (1 - a)
            logw = torch.where(mask[..., None], lag[..., None].to(v.dtype) * la,
                               torch.tensor(float("-inf"), dtype=v.dtype))
            w = torch.exp(logw + l1a)                                  # a^(t-s) (1-a), (T,S,H)
            return torch.einsum("tsh,bsh->bth", w, u)
        pre = v @ self.W_z.T + self.b_z                                # (B, T, H)
        z = torch.sigmoid(pre)
        L = F.logsigmoid(-pre).cumsum(1)                               # sum_{r<=t} log(1 - z_r)
        zu = z * u
        out = []
        for i in range(0, v.shape[0], CHUNK):
            Li = L[i:i + CHUNK]
            diff = Li[:, :, None, :] - Li[:, None, :, :]               # (b, T, S, H): L_t - L_s
            logw = torch.where(mask[None, :, :, None], diff,
                               torch.tensor(float("-inf"), dtype=v.dtype))
            out.append(torch.einsum("btsh,bsh->bth", torch.exp(logw), zu[i:i + CHUNK]))
        return torch.cat(out, 0)

    def variant_gates(self, v):
        g = F.softmax(torch.tanh(self._state(v)) @ self.W_g.T, dim=-1)
        return g, g

    def forward(self, tokens, tau=None, track_sat=False):
        gr, gw = self.variant_gates(self.embed(tokens))
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        if self.conv is None:
            logits, _ = BDH.forward(self, tokens)
        else:
            logits = self.forward_conv(tokens)
        return logits, None, gr, gw


def reference_loop(m, tokens):
    """The explicit per-step loop of the recurrence, independent of _state."""
    v = m.embed(tokens)
    B, T, _ = v.shape
    c = torch.zeros(B, m.h_gate, dtype=v.dtype)
    gs = []
    for t in range(T):
        u = v[:, t] @ m.W_in.T
        if m.kind == "ema":
            a = torch.sigmoid(m.theta)
            c = a * c + (1 - a) * u
        else:
            z = torch.sigmoid(v[:, t] @ m.W_z.T + m.b_z)
            c = (1 - z) * c + z * u
        gs.append(F.softmax(torch.tanh(c) @ m.W_g.T, dim=-1))
    return torch.stack(gs, 1)


def make_model(task, a, seed, kind):
    torch.manual_seed(seed)                              # build()'s seeding, then the same ctor
    return GateVariantBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                          gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                          ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), kind=kind,
                          variant_seed=seed + (EMA_OFFSET if kind == "ema" else SEL_OFFSET))


def builder(task, kind):
    return lambda a, seed: (lambda: make_model(task, a, seed, kind))


# ── CHECKs (explore_batch3 runs them) ────────────────────────────────────────
def check_gate(task, kind, label, seed=160):
    ok = True
    m = make_model(task, ec.ARM_A, seed, kind)
    ref = build(task, ec.ARM_A, ARCH, seed)
    pr = dict(ref.named_parameters())
    new = {"theta"} if kind == "ema" else {"W_z", "b_z"}
    names = {n for n, _ in m.named_parameters()}
    shared = (names - new == set(pr)
              and all(torch.equal(p, pr[n]) for n, p in m.named_parameters() if n in pr))
    g = torch.Generator().manual_seed(11)
    x, _ = task.make_batch(64, g)
    x = x[:, :-1]
    res = {}
    for mode in ("train", "eval"):
        m.train(mode == "train")
        with torch.no_grad():
            gv = m(x)[2]
            gl = reference_loop(m, x)
            res[f"loop_{mode}"] = (gv - gl).abs().max().item()
            T = x.shape[1]
            caus = True
            for t in (0, T // 3, T // 2, T - 2):
                y = x.clone()
                y[:, t + 1:] = (y[:, t + 1:] + 1 + torch.randint(0, task.vocab - 1, y[:, t + 1:].shape,
                                                                  generator=g)) % task.vocab
                gy = m(y)[2]
                caus &= bool(torch.equal(gy[:, :t + 1], gv[:, :t + 1]))
                caus &= not torch.equal(gy[:, t + 1:], gv[:, t + 1:])
            res[f"causal_{mode}"] = caus
    # the same comparisons on a copy with inflated weights, so the gates are far from uniform
    import copy
    ms = copy.deepcopy(m)
    with torch.no_grad():
        ms.embed.weight.mul_(50.0); ms.W_g.mul_(10.0)
        if kind == "ema":
            ms.theta.add_(torch.randn(ms.theta.shape, generator=g))
        else:
            ms.W_z.mul_(3.0)
        ms.eval()
        gv = ms(x)[2]
        res["loop_scaled"] = (gv - reference_loop(ms, x)).abs().max().item()
        res["spread_scaled"] = (2 * (gv[..., 0] - 0.5).abs()).mean().item()
        y = x.clone()
        t = x.shape[1] // 2
        y[:, t + 1:] = (y[:, t + 1:] + 1) % task.vocab
        res["causal_scaled"] = bool(torch.equal(ms(y)[2][:, :t + 1], gv[:, :t + 1]))
    items = [(f"{label}: every parameter shared with arm A is bitwise arm A's (new: {sorted(new)})",
              shared),
             (f"{label}: closed form equals the loop on inflated weights (gates far from uniform, "
              f"spread {res['spread_scaled']:.2f}; max diff {res['loop_scaled']:.2e} < 1e-6) and "
              f"stays causal there", res["loop_scaled"] < 1e-6 and res["causal_scaled"]
              and res["spread_scaled"] > 0.2),
             (f"{label}: closed form equals the explicit per-step loop, train (max diff "
              f"{res['loop_train']:.2e} < 1e-6)", res["loop_train"] < 1e-6),
             (f"{label}: closed form equals the explicit per-step loop, eval (max diff "
              f"{res['loop_eval']:.2e} < 1e-6)", res["loop_eval"] < 1e-6),
             (f"{label}: causal in train (tokens after t changed: gates <= t identical, later ones "
              f"change)", res["causal_train"]),
             (f"{label}: causal in eval", res["causal_eval"])]
    if kind == "ema":
        h = (-math.log(2.0) / F.logsigmoid(m.theta.detach().double())).sort().values
        want = log_spaced_half_lives(m.h_gate)
        rel = ((h - want).abs() / want).max().item()
        items.append((f"{label}: initial half-lives are the log-spaced set 1..32 (max rel. diff "
                      f"{rel:.1e}; {h[0]:.3f}, {h[1]:.3f}, ..., {h[-1]:.3f})", rel < 1e-5))
        g2 = make_model(task, ec.ARM_A, seed + 1, kind).theta.detach()
        items.append((f"{label}: theta is a permutation drawn from its own generator (differs "
                      f"across seeds, same set)", not torch.equal(g2, m.theta.detach())
                      and torch.equal(g2.sort().values, m.theta.detach().sort().values)))
    else:
        items.append((f"{label}: b_z = -1 at init", bool((m.b_z == B_Z0).all())))
    for nm, v in items:
        print(f"  CHECK {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok
