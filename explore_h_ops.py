#!/usr/bin/env python
"""
explore_h_ops.py — EXPLORATORY, not a result. Session H's split operations on W_g, pure torch (no project imports), so a
child of either kind (this branch's modules for S78; the main line's for S79) can use them.

  copy_exact(m, o, cs, c0, seed)   W_g[c0] := W_g[c*] exactly; W_g[c*] and the optimizer state untouched (S61's
                                   COPY_NONOISE operation).
  split_noise_reset(...)           S36's split_op (explore_split_child.split_op) line for line: w = W_g[c*];
                                   sd = 0.1 std(w); n1, n2 ~ N(0, sd^2) drawn in that order from a generator seeded with
                                   seed; W_g[c0] = w + n2, W_g[c*] = w + n1; then every tensor in W_g's optimizer state
                                   zeroed in place. CHECK (in a main-line child): equals split_op bit for bit.
"""

import torch

NOISE_REL = 0.1


def copy_exact(m, o, cs, c0, seed):
    W = m.W_g
    with torch.no_grad():
        before = float((W[cs] - W[c0]).norm())
        W[c0] = W[cs].clone()
    return dict(row_dist_before=before, row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=[])


def noise_for(w, seed):
    sd = NOISE_REL * w.std()
    g = torch.Generator().manual_seed(seed)
    n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd
    n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd
    return sd, n1, n2


def split_noise_reset(m, o, cs, c0, seed):
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = noise_for(w, seed)
        W[c0] = w + n2
        W[cs] = w + n1
    st = o.state.get(W)
    zeroed = []
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                v.zero_()
                zeroed.append(k)
    return dict(noise_sd=float(sd), noise_seed=seed, row_dist_before=before,
                row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=zeroed)
