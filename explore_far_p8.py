#!/usr/bin/env python
"""
explore_far_p8.py — EXPLORATORY, not a result. Screen S28 (batch 9): the header layout at P=8.
Does it need the partition, and does S19's FAR_TOK route there?

BACKGROUND (docstring)
- S23: one channel binds the header layout at P=4 (B_FAR 3/10, B_FAR_SLOW 7/10). On the grouped
  layout at P=8 one channel with the same memory bound 0/24 on the main line.
- S19/S22: FAR_TOK (the HINGE recipe plus a key-token term, header layout, P=4) DISCOVERED 10/20,
  ROUTED* at the end 7/20.

THE TASK: explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1): each stream's block is its CTX token
followed by its 8 KEY/VAL pairs (keys in random order), the two blocks in random stream order, then
one query triple (CTX, KEY, VAL) as before. L = 2 x 17 + 3 = 37, the query's KEY at position 35,
vocabulary 26 (S + P + n_vals).

THE ARMS (MAX_ITERS 24000 = test_short_conv.MAX_ITERS, evaluation every 1200, early stop after 3
evaluations >= 0.95, lr 1e-3 = test_channel_binding.SUB_LR passed explicitly; S6's, S23's and S19's
run paths with the P=8 task in place of the P=4 one)
  FAR8_CEIL    perfect gate (test_channel_binding's ceiling arm for this task, k = S), lr 1e-3
               throughout; seeds 160-162 (validity). Statistics explore_far_cue.header_stats.
  B_FAR8_SLOW  single channel (test_short_conv's arm B: k = 1, gate 'none', no conv) with the slow
               schedule: every parameter at 1e-4 for updates 1-2400, 1e-3 after (onset_run's lr_at),
               as S23's B_FAR_SLOW. Seeds 160-169. Statistics test_short_conv.conv_stats.
  FAR_TOK8     S19's FAR_TOK: arm A + SLOW_MEM's groups and lr switch + 1.0 x [relu(eta2_index - 0.2)
               + relu(eta2_half - 0.2) + relu(eta2_key - 0.2)] at the header layout's 16 KEY positions
               (S19's TokHingeBDH, S17's position terms with half = block, S19's key term with 8 keys).
               Seeds 160-169. Statistics S19's (header_stats + hinge counts).
The seeds pair with S23 and S19/S22 by number only (a different task: different batches and, the
vocabulary being larger, different initial parameters); those outcomes are printed alongside.

READINGS (fixed before any run)
- "P=8 header needs the partition" if FAR8_CEIL binds >= 2/3 and B_FAR8_SLOW binds <= 2/10;
  otherwise "it does not".
- FAR_TOK8: DISCOVERED and ROUTED* at the end printed; "promising" if ROUTED*@end >= 4/10 and the
  first reading holds (otherwise "not promising").

CHECKS: the P=8 header task gives each (stream, key) once, with S distinct values per key, one CTX per
block (block streams a permutation), stream labels as the blocks, each query answered by its own
stream's value, L = 37 and qpos = [35]; the perfect gate zeroes every cross-stream score at every
layer (same-stream scores non-zero) and scores margin 1 / eta_key_by_stream 1 in header_routing_stats;
the three run paths, given the P=4 header task, reproduce S6's far_ceil, S23's B_FAR_SLOW and S19's
FAR_TOK records (seed 160, 3600 steps).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_far_cue as s6
import explore_far_single as s23
import explore_token_hinge as s19
import test_channel_binding as tcb
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY

NAME = "far_p8"
IDEA = "the header layout at P=8: does it need the partition, and does FAR_TOK route there?"
SOURCE = ("batch 8's S23 (one channel binds the P=4 header layout); the main line (one channel bound 0/24 "
          "on the grouped layout at P=8)")
CHANGE = "HeaderTask(P=8): perfect gate (validity), single channel with the slow schedule, S19's FAR_TOK"
PAIRING = ("seeds 160-169; S23's and S19/S22's P=4 header outcomes printed by seed number only (a different task, "
           "different batches)")
HEADER8 = s6.HeaderTask(8, S=tcb.S_STREAMS, n_vals=tcb.N_VALS, n_q=tcb.N_Q)
HEADER4 = s6.HEADER
KPOS8 = s6.positions(HEADER8)[0]
SEEDS = tuple(range(160, 170))
CEIL_MIN, BSLOW_MAX, TOK_MIN = 2, 2, 4

_CEIL = {a["key"]: a for a in cb_arms(HEADER8)}["ceiling"]
ARMS = {
    "FAR8_CEIL": dict(_CEIL, key="FAR8_CEIL", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=(160, 161, 162), prio=1,
                      label="perfect gate, header layout P=8 (validity)"),
    "B_FAR8_SLOW": dict(s23.ARMS["B_FAR_SLOW"], key="B_FAR8_SLOW", seeds=SEEDS, prio=1, layout="header8",
                        label="single channel (arm B), slow schedule, header layout P=8"),
    "FAR_TOK8": dict(s19.ARMS["FAR_TOK"], key="FAR_TOK8", seeds=SEEDS, prio=1, layout="header8",
                     label="S19's FAR_TOK, header layout P=8 (16 KEY positions)",
                     sched=s19.ARMS["FAR_TOK"]["sched"].replace("(header KEY positions)",
                                                                "(header KEY positions, P=8)")),
}


# ── run paths (task-parametrised copies of S6's, S23's and S19's) ────────────
def ceil_path(task, a, seed, iters, keep=None):
    return ec.run_one(a, seed, iters, keep=keep, task=task, stats_fn=s6.header_stats,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=lambda aa, s: (lambda: build(task, aa, ARCH, s)))


def bslow_path(task, a, seed, iters, keep=None):
    return ec.run_one(a, seed, iters, keep=keep, task=task, stats_fn=tsc.conv_stats, grad_fn=tsc.conv_grad_norms,
                      lr=a["lr"], builder=lambda aa, s: (lambda: build(task, aa, ARCH, s)),
                      run_kw=dict(lr_at=s23.make_lr_at(a)))


def tok_model(task, a, seed):
    torch.manual_seed(seed)
    return s19.TokHingeBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                           gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                           ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), pen_lambda=s19.LAMBDA,
                           tau=s19.TAU, tau_key=s19.TAU, kpos=s6.positions(task)[0], pen_SP=(task.S, task.P))


def tok_path(task, a, seed, iters, keep=None):
    holder = {}
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=task, stats_fn=s19.stats_far,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=lambda aa, s: (lambda: tok_model(task, aa, s)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


PATH = {"FAR8_CEIL": ceil_path, "B_FAR8_SLOW": bslow_path, "FAR_TOK8": tok_path}


def run_job(arm, seed):
    a = ARMS[arm]
    return PATH[arm](HEADER8, a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    if arm == "FAR_TOK8":
        return [(a["iters"], HEADER8, lambda: tok_model(HEADER8, a, 0), a["lr"], s19.stats_far)]
    sf = s6.header_stats if arm == "FAR8_CEIL" else tsc.conv_stats
    return [(a["iters"], HEADER8, lambda: build(HEADER8, a, ARCH, 0), a["lr"], sf)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def task_checks(t):
    S, P, blk = t.S, t.P, t.blk
    g = torch.Generator().manual_seed(7)
    x, qs_ret = t.make_batch(512, g)
    B = x.shape[0]
    body, qry = x[:, :S * blk], x[:, S * blk:]
    kpos, vpos, cpos = s6.positions(t)
    ctx = body[:, cpos]
    st = ctx.repeat_interleave(P, dim=1)
    keys, vals = body[:, kpos] - S, body[:, vpos]
    pairs_once = all(sorted(zip(st[b].tolist(), keys[b].tolist())) == [(s, k) for s in range(S) for k in range(P)]
                     for b in range(B))
    vtab = torch.full((B, S, P), -1, dtype=torch.long)
    vtab[torch.arange(B)[:, None], st, keys] = vals
    distinct = bool((vtab >= S + P).all()) and all(len(set(vtab[b, :, k].tolist())) == S
                                                   for b in range(B) for k in range(P))
    qs, qk, qv = qry[:, 0], qry[:, 1] - S, qry[:, 2]
    answered = (bool((vtab[torch.arange(B), qs, qk] == qv).all()) and qs_ret.shape == (B, t.n_q)
                and bool((qs_ret[:, 0] == qs).all()))
    is_ctx = body < S
    one_ctx = (bool(is_ctx[:, cpos].all()) and int(is_ctx.sum()) == B * S
               and all(sorted(ctx[b].tolist()) == list(range(S)) for b in range(B)))
    inp = x[:, :-1]
    lab = t.stream_labels(inp)
    want = torch.cat([ctx.repeat_interleave(blk, dim=1), qs[:, None].expand(B, inp.shape[1] - S * blk)], dim=1)
    labels_ok = bool((lab == want).all())
    shape_ok = x.shape[1] == t.L == S * (1 + 2 * P) + 3 == 37 and t.qpos == [35] and t.vocab == S + P + t.n_vals
    ql, qt, _ = t.select(torch.zeros(B, inp.shape[1], t.vocab), x[:, 1:], None)
    select_ok = bool((qt == qv).all()) and bool((inp[:, t.qpos[0]] == S + qk).all())
    ceil = build(t, ARMS["FAR8_CEIL"], ARCH, 160)
    ceil.eval()
    ceil.attn.record = []
    with torch.no_grad():
        ceil(inp[:64])
    rec, ceil.attn.record = ceil.attn.record, None
    cross = (lab[:64, :, None] != lab[:64, None, :]).unsqueeze(1)
    same = (~cross) & torch.ones_like(cross).tril(-1).bool()
    perfect_ok = (len(rec) == ARCH["n_layer"] and all(bool((sc[cross] == 0).all()) for sc in rec)
                  and all(bool((sc[same] != 0).any()) for sc in rec))
    hr = s6.header_routing_stats(ceil, t, (inp[:256], lab[:256], t.role_masks(inp[:256])))
    hstats_ok = abs(hr["margin"] - 1.0) < 1e-6 and abs(hr["eta_key_by_stream"] - 1.0) < 1e-6
    return [("every (stream, key) appears once", pairs_once),
            ("S distinct values per key", distinct),
            ("each query is answered by its own stream's value", answered),
            ("exactly one CTX token per block; block streams a permutation", one_ctx),
            ("stream_labels = the block's stream (body), the query's stream (query)", labels_ok),
            (f"L = {t.L}, qpos = {t.qpos}, vocab = {t.vocab}", shape_ok),
            ("select reads the query's KEY position and targets its value", select_ok),
            (f"the perfect gate zeroes every cross-stream score at every layer ({len(rec)} layers; same-stream "
             f"scores non-zero)", perfect_ok),
            (f"header_routing_stats under the perfect gate: margin {hr['margin']:.4f}, eta_key_by_stream "
             f"{hr['eta_key_by_stream']:.4f}", hstats_ok)]


def check():
    ok = True
    rows = [(f"P=8 header task: {nm}", v) for nm, v in task_checks(HEADER8)]
    it = 3 * EVAL_EVERY
    for arm, path, store, key, a4 in (
            ("FAR8_CEIL", ceil_path, "far_cue", "far_ceil", s6.ARMS["far_ceil"]),
            ("B_FAR8_SLOW", bslow_path, s23.NAME, "B_FAR_SLOW", s23.ARMS["B_FAR_SLOW"]),
            ("FAR_TOK8", tok_path, s19.NAME, "FAR_TOK", s19.ARMS["FAR_TOK"])):
        r = path(HEADER4, a4, 160, it)
        rec = [c for c in ec.load_store(store)["runs"][f"{key}|160"]["curve"] if c[0] <= it]
        rows.append((f"{arm}'s run path with the P=4 header task reproduces the {key} record (seed 160, to {it}; "
                     f"{len(rec)} evaluations)", r["curve"] == rec))
    m = tok_model(HEADER8, ARMS["FAR_TOK8"], 160)
    rows.append((f"FAR_TOK8's hinge reads the 16 KEY positions {m.kpos.tolist()} with (S, P) = {m.pen_SP}",
                 m.kpos.tolist() == [b * 17 + 1 + 2 * i for b in range(2) for i in range(8)] and m.pen_SP == (2, 8)))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def runs_of(store_name, key):
    return {int(k.split("|")[1]): r for k, r in ec.load_store(store_name)["runs"].items()
            if k.startswith(key + "|") and r.get("ok")}


def acc_at(r, step):
    return next((c[1] for c in r["curve"] if c[0] == step), None)


def reading_part(nc, nb):
    return "P=8 header needs the partition" if nc >= CEIL_MIN and nb <= BSLOW_MAX else "it does not"


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    b4 = runs_of(s23.NAME, "B_FAR_SLOW")
    t4 = runs_of(s19.NAME, "FAR_TOK")
    t4.update(runs_of("far_tok_check", "FAR_TOK_NEW"))
    ce = got["FAR8_CEIL"]
    print(f"  FAR8_CEIL (validity): " + "; ".join(
        f"seed {s} {('BOUND at ' + str(ce[s]['transition'])) if ec.bound(ce[s]) else 'unbound'} (acc {ce[s]['acc']:.3f})"
        if s in ce else f"seed {s} not run" for s in ARMS["FAR8_CEIL"]["seeds"]))
    print("  per seed (acc at 2400/4800/12000; P=4 columns: the same seed number on the P=4 header task, S23's "
          "B_FAR_SLOW and S19/S22's FAR_TOK):")
    print(f"    {'seed':>4}  {'B_FAR8_SLOW acc@2400/4800/12000':>31} {'end':>6} {'trans':>6} {'outcome':<8} "
          f"{'P=4 B_FAR_SLOW':<15} | {'FAR_TOK8 acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8} {'outcome':<15} {'hinge i/h/k/any of n':>24} {'P=4 FAR_TOK':<15}")
    for s in SEEDS:
        rb, rt = got["B_FAR8_SLOW"].get(s), got["FAR_TOK8"].get(s)
        if rb:
            ac = "/".join("--" if acc_at(rb, t) is None else f"{acc_at(rb, t):.3f}" for t in (2400, 4800, 12000))
            bs = f"{ac:>31} {rb['acc']:6.3f} {str(rb['transition']):>6} {'BOUND' if ec.bound(rb) else 'unbound':<8}"
        else:
            bs = f"{'not run':>31} {'':>6} {'':>6} {'':<8}"
        if rt:
            m = rt["end"].get("margin")
            t, hc = s19.counts(rt)
            ts = (f"{rt['acc']:6.3f} {str(rt['transition']):>6} {rt['val_cos']:7.3f} "
                  f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(rt):>20} {c2.rflags(rt):>8} {ec.tag(rt):<15} "
                  f"{f'{t[0]}/{t[1]}/{t[2]}/{hc[0]} of {hc[1]}':>24}")
        else:
            ts = f"{'not run':<104}"
        print(f"    {s:>4}  {bs} {ec.tag(b4.get(s)):<15} | {ts} {ec.tag(t4.get(s)):<15}")
    nc = sum(ec.bound(r) for r in ce.values())
    bsl = got["B_FAR8_SLOW"]
    nb = sum(ec.bound(r) for r in bsl.values())
    tk = got["FAR_TOK8"]
    nd = sum(ec.discovered(r) for r in tk.values())
    nr = sum(c2.routed_at(r, "end") for r in tk.values())
    nbt = sum(ec.bound(r) for r in tk.values())
    fails = s19.fails(tk.get, tk)
    print(f"  FAR8_CEIL BOUND {nc}/{len(ce)}; B_FAR8_SLOW BOUND {nb}/{len(bsl)} (P=4 B_FAR_SLOW on these seeds "
          f"{sum(ec.bound(b4.get(s)) for s in SEEDS)}/10)")
    print(f"  FAR_TOK8: DISCOVERED {nd}/{len(tk)}, ROUTED*@end {nr}/{len(tk)}, BOUND {nbt}/{len(tk)}, "
          f"ROUTED*@1200 {sum(c2.routed_at(r, EVAL_EVERY) for r in tk.values())}/{len(tk)}; failures {fails}")
    s19.firings(tk)
    rp = reading_part(nc, nb)
    rt = "promising" if nr >= TOK_MIN and rp == "P=8 header needs the partition" else "not promising"
    if nc < CEIL_MIN:
        print(f"  NOTE: the validity arm FAR8_CEIL bound {nc}/{len(ce)} (< {CEIL_MIN}/3): the perfect gate does not bind "
              f"this task within {ARMS['FAR8_CEIL']['iters']} steps, so 'it does not' below says nothing about the partition")
    print(f"  RULE S28 (FAR8_CEIL {nc}/{len(ce)}, B_FAR8_SLOW {nb}/{len(bsl)}; 'P=8 header needs the partition' if "
          f">= {CEIL_MIN}/3 and <= {BSLOW_MAX}/10): {rp}")
    print(f"  RULE S28 FAR_TOK8 (ROUTED*@end {nr}/{len(tk)}; 'promising' if >= {TOK_MIN}/10 and the first reading "
          f"holds): {rt}")
    return dict(ceil=nc, n_ceil=len(ce), bslow=nb, n_bslow=len(bsl), tok_disc=nd, tok_routed=nr, n_tok=len(tk),
                reading=rp, tok_reading=rt)
