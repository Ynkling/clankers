#!/usr/bin/env python
"""
explore_b18_gates.py — EXPLORATORY, not a result. Batch 18's shared helpers. It imports only torch, so a child of either
kind (this branch's modules for S52; the main line's at explore_main9c.MAIN_SHA for S50, S51) can use it.

eta2_pool(x, lab, G)   eta^2 of a vector-valued x (..., k) grouped by integer labels lab in 0..G-1, pooled over the k
                       channels: between-group over total sum of squares, both summed over channels, total + EPS;
                       differentiable. test_slow_start.eta2_hinge's computation (main at 9c5939e) with the grouping
                       left open (CHECK: with its groupings, by triple index and by sequence half, it equals eta2_hinge's
                       two values; check_vs_slow_start below, run in a main-line child).
eta2_key(gr, tokens, S, P)  eta2_pool of the read gate at the body's key positions 3j+1 grouped by the key id there
                       (token - S): W2_KEYHINGE's term (S52), the key-position analogue of the hinge's index / half terms.
zero_state(o, W)       S36's split_op's optimizer-state reset of W, line for line, without the row copy: every tensor in
                       o.state[W] zeroed in place (Adam: exp_avg, exp_avg_sq, step). W_RESET's operation (S50).
"""

import torch
import torch.nn.functional as F

EPS = 1e-12                                     # test_slow_start.EPS (explore_slow_pos.EPS)


def eta2_pool(x, lab, G, eps=EPS):
    k = x.shape[-1]
    x = x.reshape(-1, k)
    lab = lab.reshape(-1)
    mu = x.mean(0)
    tot = ((x - mu) ** 2).sum() + eps
    oh = F.one_hot(lab, G).to(x.dtype)                                   # (N, G)
    cnt = oh.sum(0)                                                      # (G,)
    gm = (oh.t() @ x) / cnt.clamp(min=1).unsqueeze(1)                    # group means (G, k)
    between = (cnt.unsqueeze(1) * (gm - mu) ** 2).sum()
    return between / tot


def eta2_key(gr, tokens, S, P, eps=EPS):
    j = torch.arange(S * P)
    return eta2_pool(gr[:, 3 * j + 1, :], tokens[:, 3 * j + 1] - S, P, eps)


def zero_state(o, W):
    st = o.state.get(W)
    zeroed = []
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                v.zero_()
                zeroed.append(k)
    return zeroed


# ── CHECK (run in a main-line child) ─────────────────────────────────────────
def check_vs_slow_start(p):
    """eta2_pool with test_slow_start.eta2_hinge's groupings (triple index; sequence half) equals eta2_hinge's two values on
    random read gates (softmax of N(0, 2^2) logits), k = 2, 4, 16, S = 2, 4, 8, P = 4."""
    import test_slow_start as tss
    rows, worst = [], 0.0
    g = torch.Generator().manual_seed(18)
    for S, k in ((2, 2), (2, 4), (4, 4), (8, 16)):
        P = 4
        n = S * P
        T = 3 * n + 2
        gr = torch.softmax(2.0 * torch.randn(64, T, k, generator=g), -1)
        e_i, e_h = (float(v) for v in tss.eta2_hinge(gr, S, P))
        j = torch.arange(n)
        x = gr[:, 3 * j + 1, :]
        mi = float(eta2_pool(x, j.expand(64, n), n))
        mh = float(eta2_pool(x, (j >= n // 2).long().expand(64, n), 2))
        d = max(abs(mi - e_i), abs(mh - e_h))
        worst = max(worst, d)
        rows.append(dict(S=S, k=k, index=[round(e_i, 6), round(mi, 6)], half=[round(e_h, 6), round(mh, 6)], diff=d))
    return dict(ok=worst <= 1e-6 and tss.EPS == EPS, worst=worst, rows=rows, tss_eps=tss.EPS)
