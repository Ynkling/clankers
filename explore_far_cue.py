#!/usr/bin/env python
"""
explore_far_cue.py — EXPLORATORY, not a result. Screen S6 (batch 2), descriptive: does LOCAL3
(batch 1's gate from a width-3 causal convolution) still find the routing when the stream token
is out of its view? A new "header" layout.

BACKGROUND. LOCAL3 routed 15/20 against arm A's 7/20 (batch 1). Its gate sees the stream token at
every key, value and query (at t-1 or t-2); this follows from how BindTask builds each triple
[CTX_s, KEY_i, VAL_{i,s}].

THE HEADER LAYOUT (HeaderTask, a BindTask subclass; test_binding_capacity is not changed)
- Each stream's block is [CTX_s, KEY, VAL, KEY, VAL, ...]: the stream token once, at the start of
  the block, then its P (key, value) pairs. Blocks come in random stream order; keys come in a
  random order within each block, drawn independently per block.
- Values and queries as in make_batch: S distinct values per key; queries [CTX_s, KEY_i, VAL]
  at the end, drawn without replacement.
- L = S*(1+2P) + 3*n_q; qpos = S*(1+2P) + 3j + 1 (the query's KEY position; its logits predict
  the query's value). At P=4, S=2: L = 21, qpos = [19].
- At P=4 the stream token falls inside a width-3 window (t-2..t) only for the first pair of each
  block; the query's KEY position sees its CTX at t-1.
- Recipe otherwise explore_common's: lr test_channel_binding.SUB_LR = 1e-3 (passed explicitly),
  MAX_ITERS 24000, evaluation every 1200 on eval_batch(HEADER), early stop after 3 >= 0.95.

ARMS (seeds paired within the batch: same seed = same initial parameters and the same batches)
    far_ceil  perfect gate (validity)   seeds 160-162
    far_A     arm A                     seeds 160-169
    far_L3    LOCAL3                    seeds 160-169
If far_ceil binds fewer than 2/3, S6 is reported UNTESTED.

STATISTICS. gate_stats_k (test_router_reliability) at every evaluation, and header_routing_stats
at 1200, 2400, 3600 and the end: test_router_layout.routing_stats line for line with the header
positions (key of pair i in block b at b*(1+2P) + 1 + 2i, its value one later; pairs indexed in
order of appearance; "half" = the second half of the pairs, i.e. the second block at S=2, which
is not the stream since block order is random). ROUTED* = margin >= 0.9 and eta_key_by_stream >
0.9. Failure classes: test_router_layout.fail_class on these statistics.

REPORT (descriptive). Per arm: bound, ROUTED* at 1200, failure types; far_L3 minus far_A in bound
runs, with the paired seeds. Readings, fixed before any run:
  far_L3 bound >= far_A bound + 3 : "LOCAL3 survives a distant cue"
  far_L3 bound <= far_A bound     : "LOCAL3's gain needs the cue in view"
  otherwise                       : neither reading applies

CHECKS: every (stream, key) appears once, with S distinct values per key; each query is answered
by its own stream's value; exactly one CTX token per block, the blocks' streams a permutation;
stream_labels gives the block's stream at every body position and the query's stream at the
query; the perfect gate zeroes every cross-stream score at every layer (and leaves same-stream
scores non-zero); L, qpos and select as stated; the stream token is in a width-3 window only for
each block's first pair; header_routing_stats gives margin 1 and eta_key_by_stream 1 under the
perfect gate; far_A and far_L3 share their initial parameters and vocabulary with arm A's.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import explore_common as ec
import explore_common2 as c2
import explore_local_gate as lg
import test_channel_binding as tcb
import test_short_conv as tsc
from test_binding_capacity import BindTask
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import gate_stats_k
from test_router_layout import eta2, fail_class
from test_instrument_v2 import TAU_END

NAME = "far_cue"
IDEA = "LOCAL3 when the stream token is a block header, out of a width-3 window for most pairs"
SOURCE = "batch 1's LOCAL3 (Mamba / H3 / RWKV local gate); a new header layout"
CHANGE = ("HeaderTask: each stream's block [CTX_s, K, V, K, V, ...], blocks in random stream order, "
          "keys random within a block; queries at the end as in make_batch")
PAIRING = "within the batch: far_A and far_L3 on seeds 160-169 (same initial parameters, same batches)"


class HeaderTask(BindTask):
    def __init__(self, P, S, n_vals, n_q):
        super().__init__(P, S=S, n_vals=n_vals, n_q=n_q)      # grouped, then re-laid out below
        self.layout = "header"
        self.key = f"P{P}hdr"
        self.label += ", layout=header"
        self.blk = 1 + 2 * P
        self.L = S * self.blk + 3 * n_q
        self.T = self.L - 1
        self.qpos = [S * self.blk + 3 * j + 1 for j in range(n_q)]

    def make_batch(self, B, gen):
        S, P, nv, nq = self.S, self.P, self.n_vals, self.n_q
        vals = torch.rand(B, P, nv, generator=gen).argsort(-1)[..., :S]    # (B,P,S) distinct
        sblk = torch.rand(B, S, generator=gen).argsort(-1)                  # stream of block b
        kblk = torch.rand(B, S, P, generator=gen).argsort(-1)               # key order in block b
        ss = sblk.unsqueeze(-1).expand(B, S, P).reshape(B, -1)
        kk = kblk.reshape(B, -1)
        vv = vals.reshape(B, P * S).gather(1, kk * S + ss)
        pairs = torch.stack([S + kk, S + P + vv], dim=-1).reshape(B, S, 2 * P)
        body = torch.cat([sblk.unsqueeze(-1), pairs], dim=-1).reshape(B, -1)
        q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :nq]         # no replacement
        qs, qk = q // P, q % P
        qv = vals.reshape(B, P * S).gather(1, qk * S + qs)
        queries = torch.stack([qs, S + qk, S + P + qv], dim=-1).reshape(B, -1)
        return torch.cat([body, queries], dim=1), qs


HEADER = HeaderTask(ec.TASK.P, S=tcb.S_STREAMS, n_vals=tcb.N_VALS, n_q=tcb.N_Q)


def positions(task):
    S, P, blk = task.S, task.P, task.blk
    kpos = torch.tensor([b * blk + 1 + 2 * i for b in range(S) for i in range(P)])
    return kpos, kpos + 1, [b * blk for b in range(S)]


@torch.no_grad()
def header_routing_stats(model, task, probe):
    """test_router_layout.routing_stats line for line, at the header layout's positions."""
    assert task.n_q == 1
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        _, _, gr, gw = model(pinp, TAU_END)
    finally:
        model.train(was)
    S, P, n = task.S, task.P, task.S * task.P
    B = pinp.shape[0]
    kpos, vpos, cpos = positions(task)
    j = torch.arange(n)
    ctx = pinp[:, cpos].repeat_interleave(P, dim=1)                         # (B, n)
    key = pinp[:, kpos] - S
    labels = dict(stream=ctx, key=key, half=(j >= P).long().expand(B, n), index=j.expand(B, n))
    p = gr[..., 0]
    out = {}
    for role, pos in (("key", kpos), ("val", vpos)):
        for name, lab in labels.items():
            out[f"eta_{role}_by_{name}"] = eta2(p[:, pos], lab)
    qpos = task.qpos[0]
    qs, qk = pinp[:, qpos - 1], pinp[:, qpos] - S
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    assert bool((tgt.sum(1) == 1).all()) and bool((dis.sum(1) == S - 1).all())
    dots = (gw[:, vpos] * gr[:, qpos][:, None, :]).sum(-1)                 # (B, n)
    marg = (dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)
    out["margin"] = marg.mean().item()
    return out


@torch.no_grad()
def header_stats(model, task, probe, step):
    out = gate_stats_k(model, task, probe, step)
    if step in c2.ROUTE_AT:
        out.update(header_routing_stats(model, task, probe))
    return out


_CEIL = {a["key"]: a for a in cb_arms(HEADER)}["ceiling"]
ARMS = {
    "far_ceil": dict(_CEIL, key="far_ceil", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=(160, 161, 162),
                     label="perfect gate, header layout (validity)", prio=1),
    "far_A": dict(ec.ARM_A, key="far_A", lr=ec.SUB_LR, iters=ec.MAX_ITERS,
                  seeds=tuple(range(160, 170)), label="arm A, header layout", prio=1),
    "far_L3": dict(ec.ARM_A, key="far_L3", lr=ec.SUB_LR, iters=ec.MAX_ITERS,
                   seeds=tuple(range(160, 170)), label="LOCAL3, header layout", prio=1),
}


def builder_for(arm):
    if arm == "far_L3":
        return lg.builder                                  # arm A's parameters + the gate conv
    return lambda a, seed: (lambda: build(HEADER, a, ARCH, seed))


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_one(a, seed, a["iters"], task=HEADER, stats_fn=header_stats,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder_for(arm))


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], HEADER, builder_for(arm)(a, 0), a["lr"], header_stats)]


def check():
    ok = True
    t = HEADER
    S, P, blk = t.S, t.P, t.blk
    g = torch.Generator().manual_seed(7)
    x, qs_ret = t.make_batch(512, g)
    B = x.shape[0]
    body, qry = x[:, :S * blk], x[:, S * blk:]
    kpos, vpos, cpos = positions(t)
    ctx = body[:, cpos]                                                     # (B, S)
    st = ctx.repeat_interleave(P, dim=1)
    keys, vals = body[:, kpos] - S, body[:, vpos]
    pairs_once = all(sorted(zip(st[b].tolist(), keys[b].tolist())) ==
                     [(s, k) for s in range(S) for k in range(P)] for b in range(B))
    vtab = torch.full((B, S, P), -1, dtype=torch.long)
    vtab[torch.arange(B)[:, None], st, keys] = vals
    distinct = bool((vtab >= S + P).all()) and all(
        len(set(vtab[b, :, k].tolist())) == S for b in range(B) for k in range(P))
    qs, qk, qv = qry[:, 0], qry[:, 1] - S, qry[:, 2]
    answered = (bool((vtab[torch.arange(B), qs, qk] == qv).all()) and qs_ret.shape == (B, t.n_q)
                and bool((qs_ret[:, 0] == qs).all()))
    is_ctx = body < S
    one_ctx = (bool(is_ctx[:, cpos].all()) and int(is_ctx.sum()) == B * S
               and all(sorted(ctx[b].tolist()) == list(range(S)) for b in range(B)))
    inp = x[:, :-1]
    lab = t.stream_labels(inp)
    want = torch.cat([ctx.repeat_interleave(blk, dim=1),
                      qs[:, None].expand(B, inp.shape[1] - S * blk)], dim=1)
    labels_ok = bool((lab == want).all())
    shape_ok = (x.shape[1] == t.L == S * (1 + 2 * P) + 3 and t.qpos == [S * (1 + 2 * P) + 1]
                and t.L == 21 and t.qpos == [19])
    ql, qt, _ = t.select(torch.zeros(B, inp.shape[1], t.vocab), x[:, 1:], None)
    select_ok = bool((qt == qv).all()) and bool((inp[:, t.qpos[0]] == S + qk).all())
    win = []
    for b in range(S):
        for i in range(P):
            for pos in (b * blk + 1 + 2 * i, b * blk + 2 + 2 * i):
                win.append((i == 0) == bool((body[:, max(0, pos - 2):pos + 1] < S).any(1).all())
                           and ((i == 0) or not bool((body[:, max(0, pos - 2):pos + 1] < S).any())))
    window_ok = all(win) and bool((inp[:, t.qpos[0] - 1] < S).all())
    # the perfect gate zeroes every cross-stream score at every layer
    ceil = build(t, ARMS["far_ceil"], ARCH, 160)
    ceil.eval()
    ceil.attn.record = []
    with torch.no_grad():
        ceil(inp[:64])
    rec, ceil.attn.record = ceil.attn.record, None
    cross = (lab[:64, :, None] != lab[:64, None, :]).unsqueeze(1)
    same = (~cross) & torch.ones_like(cross).tril(-1).bool()
    perfect_ok = (len(rec) == ARCH["n_layer"] and all(bool((sc[cross] == 0).all()) for sc in rec)
                  and all(bool((sc[same] != 0).any()) for sc in rec))
    probe = (inp[:256], lab[:256], t.role_masks(inp[:256]))
    hr = header_routing_stats(ceil, t, probe)
    hstats_ok = abs(hr["margin"] - 1.0) < 1e-6 and abs(hr["eta_key_by_stream"] - 1.0) < 1e-6
    fa = build(t, ARMS["far_A"], ARCH, 160)
    fl = lg.builder(ARMS["far_L3"], 160)()
    ref = build(ec.TASK, ec.ARM_A, ARCH, 160)
    pr = dict(ref.named_parameters())
    share = (t.vocab == ec.TASK.vocab and t.ctx_tokens == ec.TASK.ctx_tokens
             and all(torch.equal(p, pr[n]) for n, p in fa.named_parameters())
             and all(torch.equal(p, pr[n]) for n, p in fl.named_parameters() if n in pr))
    for nm, v in (("every (stream, key) appears once", pairs_once),
                  ("S distinct values per key", distinct),
                  ("each query is answered by its own stream's value", answered),
                  ("exactly one CTX token per block; block streams a permutation", one_ctx),
                  ("stream_labels = the block's stream (body), the query's stream (query)", labels_ok),
                  (f"L = {t.L}, qpos = {t.qpos}", shape_ok),
                  ("select reads the query's KEY position and targets its value", select_ok),
                  ("stream token in a width-3 window only for each block's first pair; the query "
                   "sees its CTX at t-1", window_ok),
                  ("perfect gate zeroes every cross-stream score at every layer (same-stream "
                   "scores non-zero)", perfect_ok),
                  (f"header_routing_stats under the perfect gate: margin {hr['margin']:.4f}, "
                   f"eta_key_by_stream {hr['eta_key_by_stream']:.4f}", hstats_ok),
                  ("far_A and far_L3 share arm A's initial parameters and vocabulary", share)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report(store, refs=None):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store, extra_recorded=())
    ceil = [store["runs"].get(f"far_ceil|{s}") for s in ARMS["far_ceil"]["seeds"]]
    nb = sum(ec.bound(r) for r in ceil)
    need = 2 if len(ceil) == 3 else len(ceil)
    print(f"  validity: far_ceil bound {nb}/{len(ceil)} "
          f"(transitions {[r['transition'] if r and r.get('ok') else None for r in ceil]})")
    if nb < need:
        print("  S6 UNTESTED: the perfect gate does not bind the header layout "
              f"({nb}/{len(ceil)} < {need}/{len(ceil)})")
        return dict(valid=False, reading="UNTESTED")
    seeds = sorted(set(ARMS["far_A"]["seeds"]) & set(ARMS["far_L3"]["seeds"]))
    res = {}
    for k in ("far_A", "far_L3"):
        runs = [store["runs"].get(f"{k}|{s}") for s in seeds]
        runs = [r for r in runs if r and r.get("ok")]
        fails = {}
        for r in runs:
            if not ec.bound(r):
                fails[fail_class(r)] = fails.get(fail_class(r), 0) + 1
        res[k] = {r["seed"]: ec.bound(r) for r in runs}
        print(f"    {k:<7} bound {sum(ec.bound(r) for r in runs)}/{len(runs)}   discovered "
              f"{sum(ec.discovered(r) for r in runs)}/{len(runs)}   ROUTED*@1200 "
              f"{sum(c2.routed_at(r, 1200) for r in runs)}/{len(runs)}   ROUTED*@end "
              f"{sum(c2.routed_at(r, 'end') for r in runs)}/{len(runs)}   failures {fails}")
    both = [s for s in seeds if s in res["far_A"] and s in res["far_L3"]]
    nA = sum(res["far_A"][s] for s in both)
    nL = sum(res["far_L3"][s] for s in both)
    only_L = [s for s in both if res["far_L3"][s] and not res["far_A"][s]]
    only_A = [s for s in both if res["far_A"][s] and not res["far_L3"][s]]
    diff = nL - nA
    print(f"    far_L3 - far_A in bound runs: {nL} - {nA} = {diff:+d}   (paired seeds {both[0]}-"
          f"{both[-1]}: far_L3 only {only_L}, far_A only {only_A})")
    if diff >= 3:
        reading = "LOCAL3 survives a distant cue"
    elif diff <= 0:
        reading = "LOCAL3's gain needs the cue in view"
    else:
        reading = f"neither reading applies (far_L3 - far_A = {diff:+d})"
    print(f"  READING S6: {reading}")
    return dict(valid=True, reading=reading, diff=diff)
