#!/usr/bin/env python
"""
test_router_layout.py — does the order in which the streams' bindings are presented change
how often arm A's learned gate finds the stream routing, and does each layout still require
channels? Everything is fixed here before any result.

Run it directly:

    python test_router_layout.py                   # CHECKs, projection, all arms
    python test_router_layout.py --also results/L/router_layout_results.json

BACKGROUND
- Arm A (k=2 recurrent gate on raw embeddings, no readout) finds the stream routing on about
  55-65% of seeds across both machines. More channels do not significantly change that
  (test_router_reliability: K4, K8 NOT SHOWN on X and L). Restarts chosen by held-out
  accuracy make it reliable (RESTART RELIABLE, 30/30 on both machines).
- Scratch diagnostic (a container that matched X exactly), six arm-A seeds that failed on X.
  - Five failed gates split by POSITION: the gate switches channel partway through the
    sequence, so key positions in the first key groups go to one channel and later ones to
    the other. One failed gate split by key identity (three keys vs one).
  - All six had settled by step 1200-2400, and the query routing margin was ~0 in each.
  - The key_part statistic had labelled two of the position splits "locked on keys". It
    measures how hard key tokens are routed, not whether different keys go to different
    channels.
- Hypothesis. BindTask groups both streams' triples for a key next to each other, so an
  early/late split of the sequence is a coarse key split. That reduces key confusion, which
  dominates early training, and can never separate the streams.
  - In a STREAM-BLOCKED layout, an early/late split no longer coincides with key groups, and
    a recurrent state that integrates the block's repeated context token carries the stream.
  - In a SHUFFLED layout, neither key groups nor streams are contiguous.
- Scratch check (not pre-registered), on arm-A seeds 0-3, all of which failed in the grouped
  layout on X and on L:
  - in the blocked layout, three routed by stream (bound by step 1200-6000) and one mixed
    stream and key (0.74 at step 6000);
  - no gate split by position;
  - single-channel B stayed at 0.21-0.22 at step 12000 in the blocked layout (seeds 0 and
    1), where the grouped layout reaches ~0.5 by then.
- Caution: the blocked layout also makes binding harder for a model WITHOUT routing, so it
  changes the payoff of routing as well as the layout. The shuffled layout and the
  diagnostics below help separate the two.
- Seed-level outcomes differ between machines; the verdict is per machine.

TASK. test_binding_capacity.BindTask gained a backward-compatible layout argument:
  grouped   (the default) the existing make_batch, unchanged (CHECK 48);
  blocked   the same S*P triples [CTX_s, KEY_i, VAL_{i,s}], each stream's P triples
            contiguous; block order a random permutation of the streams per sequence, key
            order within each block random and independent per block;
  shuffled  all S*P triples in a uniformly random order per sequence.
In every layout the values (S distinct per key, from n_vals) and the single query are drawn
as in BindTask, and the query is answered by its own stream's value (CHECK 49). Evaluation
and probe batches use the arm's own layout (CHECK 50). test_router_reliability.run_one
gained task and stats_fn knobs, inert at their defaults (CHECK 48).
Substrate and training are test_router_confirm's: P=4, S=2, n_vals=16, n_q=1; MultiBDH
n_layer=3, decay, N=256; lr 1e-3, BATCH 32; evaluation every 1200 steps on 2048 queries;
early stop after 3 evaluations >= 0.95; MAX_ITERS = 24000.
DISCOVERED = bound AND final VAL cos < 0.5.

ARMS AND SEEDS (fresh; disjoint from every earlier seed)
  A_grouped, A_blocked, A_shuffled   arm A in each layout, seeds 120-159 (40 each); paired:
                                     same seed, same initial parameters
  B_blocked (seeds 120-139), B_shuffled (seeds 120-129)   the plain single channel
  ceiling_blocked, ceiling_shuffled  the perfect gate, seeds 120-124
Fixed before any result: the layouts, arms, seeds, MAX_ITERS and every threshold below.

VALIDITY AND PRE-REGISTERED CLAIMS (each printed with its numbers)
  VALID (per new layout): the ceiling binds 5/5, AND the recurrent gate fits the perfect
    routing in that layout (argmax >= 0.99; test_router_reliability.gate_fit_k on arm A's
    model). Otherwise that layout's claim is UNTESTED.
  DISCRIMINATIVE (per new layout): single-channel B binds 0 of its seeds. Otherwise the
    layout is COMPROMISED, and its claim is printed but not interpreted as evidence for
    channels.
  L1 BLOCKS HELP     A_blocked vs A_grouped, discovered of 40, Fisher's exact test one-sided
                     (blocked higher). HELPS if p < 0.05, else NOT SHOWN.
  L2 SHUFFLING HELPS A_shuffled vs A_grouped, the same test.
  Band for each A arm: RELIABLE >= 36/40, MAJORITY 20-35, MINORITY <= 19.
Fisher and McNemar are test_router_confirm's.

DIAGNOSTICS, not part of the verdict. For every arm-A run, at the first evaluation (1200)
and at the end, on the probe batch; p is the read gate's channel-0 probability.
- eta^2 of p (between-group over total variance; 0 when p is constant), separately over the
  body's key positions and value positions, grouped by stream, key id, HALF (triple index
  < P vs >= P; in the blocked layout the block) and triple index.
- The query routing margin: mean over probe sequences of g_q.g_tgt - g_q.g_dis, g_q the read
  gate at the query's key position (the last input position), g_tgt and g_dis the write
  gates at the value positions of the body triples holding the queried key in the queried
  stream and in the other stream. 1 = perfect stream routing, 0 = stream-blind.
- Failure classes of non-discovering runs, from the end statistics, first match wins:
  STREAM-PARTIAL margin >= 0.25; KEY eta^2 by key at key positions >= 0.5; POSITION eta^2 by
  half or by triple index at key positions >= 0.5; OTHER. The old key_part classification is
  printed alongside.
- Learning speed without routing: B's accuracy curves per layout, and arm A's accuracy at
  1200 and 2400.
- The restart check per layout: runs with held-out accuracy >= 0.6 at 2400, and how many of
  those discovered.
- Escape step; per-seed outcomes across the three layouts with exact McNemar p (blocked vs
  grouped, shuffled vs grouped); gradient norms at steps 1, 10 and 100; --also pooled counts.

CHECKS (printed before any training): test_router_reliability's verification (which runs
every earlier test's), then
  48 layout="grouped" is bit-identical to the old make_batch and every earlier test's task is
     unchanged (vs the pre-change test_binding_capacity); run_one's new knobs are inert;
  49 the blocked and shuffled generators over 10,000 sequences;
  50 evaluation, probe and training batches follow the arm's layout (one decoded sequence per
     layout);
  51 the new statistics: perfect gate eta^2 by stream = 1 and margin = 1; uniform gate
     margin = 0;
  52 a worker's run is bit-identical to the same run in the main process.

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to router_layout_results.json (gitignored; copied into
results/X/ after the run).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.nn.modules.module import register_module_forward_pre_hook

import test_binding_capacity as tbc
from test_binding_capacity import BindTask
import test_binding_onset as tbo
from test_binding_onset import fmt_step, EVAL_EVERY
import test_binding_recipe as trec
import test_multilayer_binding as tmb
from test_multilayer_binding import MultiBDH, role_cosines, FIT_PASS
import test_channel_binding as tcb
from test_channel_binding import eval_batch, arms as cb_arms
from test_router_confirm import fisher_greater, mcnemar_exact
import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, make_fn, escape, get, count, mode_of, load_store, init_worker, makespan, cpu_model,
    save_results, probe_batch, time_arm, DISC_COS, COLLAPSE, TAU_END,
)
from test_router_discovery import load_at
import test_router_reliability as trr
from test_router_reliability import (
    run_one, gate_stats_k, gate_fit_k, strip_all, curve_acc, med, T_CHECK, A_CHECK,
)
from test_readout_path import GRAD_STEPS

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
LAYOUTS = ("grouped", "blocked", "shuffled")
NEW_LAYOUTS = ("blocked", "shuffled")
ALPHA = 0.05
RELIABLE_K, MAJORITY_K = 36, 20            # bands, of 40
MARGIN_PARTIAL = 0.25                      # failure class STREAM-PARTIAL
ETA_SPLIT = 0.5                            # failure classes KEY and POSITION
LEGACY_SHA = "e5291da"                     # head before this test's changes
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "router_layout_results.json"
EARLIER_SEEDS = tuple(range(120))          # every seed used by earlier router tests (0-119)

TASKS = {"grouped": TASK}
TASKS.update({lay: BindTask(TASK.P, S=TASK.S, n_vals=TASK.n_vals, n_q=TASK.n_q, layout=lay)
              for lay in NEW_LAYOUTS})
_BASE = {a["key"]: a for a in cb_arms(TASK)}
A = trc.ARM["A"]
ARMS = [dict(A, key=f"A_{lay}", layout=lay, label=f"A_{lay:<16} arm A, {lay} layout")
        for lay in LAYOUTS] + [
    dict(_BASE["B"], key="B_blocked", layout="blocked", label="B_blocked        single channel"),
    dict(_BASE["B"], key="B_shuffled", layout="shuffled", label="B_shuffled       single channel"),
    dict(_BASE["ceiling"], key="ceiling_blocked", layout="blocked",
         label="ceiling_blocked  perfect gate"),
    dict(_BASE["ceiling"], key="ceiling_shuffled", layout="shuffled",
         label="ceiling_shuffled perfect gate"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
A_KEYS = tuple(f"A_{lay}" for lay in LAYOUTS)
SEEDS = {k: tuple(range(120, 160)) for k in A_KEYS}
SEEDS.update(B_blocked=tuple(range(120, 140)), B_shuffled=tuple(range(120, 130)),
             ceiling_blocked=tuple(range(120, 125)), ceiling_shuffled=tuple(range(120, 125)))
ETA_GROUPS = ("stream", "key", "half", "index")


# ── The new statistics ───────────────────────────────────────────────────────
def eta2(x, lab):
    """Between-group variance over total variance of x grouped by lab; 0 if x is constant."""
    x, lab = x.reshape(-1).double(), lab.reshape(-1)
    mu = x.mean()
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    between = sum((lab == g).sum() * (x[lab == g].mean() - mu) ** 2 for g in lab.unique())
    return float(between / tot)


@torch.no_grad()
def routing_stats(model, task, probe):
    """eta^2 of the read gate's channel-0 probability over the body's key and value positions,
    by stream / key / half / triple index, and the query routing margin."""
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
    j = torch.arange(n)
    ctx, key = pinp[:, 3 * j], pinp[:, 3 * j + 1] - S
    labels = dict(stream=ctx, key=key, half=(j >= P).long().expand(B, n), index=j.expand(B, n))
    p = gr[..., 0]
    out = {}
    for role, off in (("key", 1), ("val", 2)):
        for name, lab in labels.items():
            out[f"eta_{role}_by_{name}"] = eta2(p[:, 3 * j + off], lab)
    qpos = task.qpos[0]
    qs, qk = pinp[:, qpos - 1], pinp[:, qpos] - S
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    assert bool((tgt.sum(1) == 1).all()) and bool((dis.sum(1) == S - 1).all())
    dots = (gw[:, 3 * j + 2] * gr[:, qpos][:, None, :]).sum(-1)          # (B, n)
    marg = (dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)
    out["margin"] = marg.mean().item()
    return out


@torch.no_grad()
def layout_stats(model, task, probe, step):
    """run_one's stats_fn: gate_stats_k (k >= 2) or the role cosines alone (k = 1, no gate);
    at the first evaluation and at the end, plus routing_stats (k = 2)."""
    if model.n_ch == 1:
        was = model.training
        model.eval()
        try:
            d = role_cosines(model, probe, step)
        finally:
            model.train(was)
        return dict(step=step, k=1, key_cos=d["key_cos"], val_cos=d["val_cos"],
                    ctx_cos=d["ctx_cos"])
    out = gate_stats_k(model, task, probe, step)
    if step in (EVAL_EVERY, "end"):
        out.update(routing_stats(model, task, probe))
    return out


def fail_class(r):
    e = r["end"]
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


# ── Layout structure (CHECKs 49 and 50) ──────────────────────────────────────
def triples(x, task):
    """Stream, key id and value token of each body triple of token rows x."""
    j = torch.arange(task.S * task.P)
    return x[:, 3 * j], x[:, 3 * j + 1] - task.S, x[:, 3 * j + 2]


def structure(x, task):
    """Fractions of rows in the grouped structure (triples 2m, 2m+1 share a key) and in the
    blocked structure (each stream's P triples contiguous), and the fraction of adjacent
    triple pairs that share a key."""
    P = task.P
    ctx, key, _ = triples(x, task)
    grouped = ((key[:, 0::2] == key[:, 1::2]) & (ctx[:, 0::2] != ctx[:, 1::2])).all(1)
    blocked = ((ctx[:, :P] == ctx[:, :1]).all(1) & (ctx[:, P:] == ctx[:, P:P + 1]).all(1)
               & (ctx[:, 0] != ctx[:, P]))
    adj = (key[:, 1:] == key[:, :-1]).float().mean().item()
    return grouped.float().mean().item(), blocked.float().mean().item(), adj


def decode(row, task):
    S, P, n = task.S, task.P, task.S * task.P
    name = lambda t: f"S{t}" if t < S else f"K{t - S}" if t < S + P else f"v{t - S - P}"
    body = " | ".join(" ".join(name(int(t)) for t in row[3 * i:3 * i + 3]) for i in range(n))
    q = row[3 * n:]
    return f"{body} || {name(int(q[0]))} {name(int(q[1]))} -> {name(int(q[2]))}"


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_job(sp):
    a = ARM[sp["arm"]]
    return run_one(a, sp["seed"], sp["iters"], task=TASKS[a["layout"]], stats_fn=layout_stats)


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_router_reliability.py's verification (which runs test_readout_path's, and so on")
    print("down to test_multilayer_binding's):")
    trr.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 48 layout=\"grouped\" is the old make_batch, and every earlier test's task is "
          f"unchanged, vs {LEGACY_SHA}'s")
    print("         test_binding_capacity (same constructor arguments, same generator seeds):")
    leg = load_at(LEGACY_SHA, "test_binding_capacity.py", "test_binding_capacity_legacy")
    earlier = ([(f"test_binding_capacity P={P}", tbc.BindTask(P)) for P in tbc.P_LIST]
               + [("test_multilayer_binding S=1", tmb.BindTask(4, S=1, n_vals=tmb.N_VALS,
                                                               n_q=tmb.N_Q))]
               + [("test_binding_onset PRIMARY", tbo.PRIMARY), ("test_binding_onset SECONDARY",
                                                                tbo.SECONDARY)]
               + [(f"test_binding_recipe P={P}", trec.task_for(P)) for P in (4, 8, 16)]
               + [(f"test_channel_binding P={P} S={S}", tcb.task_for(P, S))
                  for P in tcb.P_LIST for S in (1, 2)]
               + [("test_router_* TASK", TASK)])
    good = True
    for name, t in earlier:
        L = leg.BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q)
        L.key = t.key
        explicit = BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q, layout="grouped")
        explicit.key = t.key
        attrs = ({k: v for k, v in vars(t).items() if k != "layout"} == vars(L)
                 and t.layout == "grouped" and vars(explicit) == vars(t))
        same = True
        for gs in (0, 1):
            outs = []
            for task in (t, L, explicit):
                g = torch.Generator().manual_seed(gs)
                tok, qs = task.make_batch(64, g)
                outs.append((tok, qs, g.get_state()))
            same &= all(torch.equal(outs[0][i], o[i]) for o in outs[1:] for i in range(3))
        g_ = attrs and same
        good &= g_
        if not g_:
            print(f"         {name}: attributes equal {attrs}   batches and generator states equal "
                  f"{same}  -> DIFFERS")
    print(f"         {len(earlier)} tasks (S=1 and S=2, n_q=1 and 4, P=4-32): attributes (other than "
          f"the new layout = 'grouped'), batches")
    print(f"         and generator states after make_batch equal the old ones, implicit and explicit "
          f"default: {good}")
    leg_trr = load_at(LEGACY_SHA, "test_router_reliability.py", "test_router_reliability_legacy")
    r_leg = leg_trr.run_one(A, 3, 60, eval_every=20)
    r_new = run_one(A, 3, 60, eval_every=20)
    r_exp = run_one(A, 3, 60, eval_every=20, task=TASK, stats_fn=gate_stats_k)
    same_run = strip_all(r_leg) == strip_all(r_new) == strip_all(r_exp) and r_new["ok"]
    print(f"         test_router_reliability.run_one's new knobs (task, stats_fn) at their defaults "
          f"and explicit, vs {LEGACY_SHA}'s run_one,")
    print(f"         arm A seed 3, 60 steps (evaluations every 20): records equal {same_run}  -> "
          f"{'UNCHANGED' if good and same_run else 'CHANGED'}")
    ok &= good and same_run
    print()

    N = 10_000
    print(f"CHECK 49 the generators, over {N:,} sequences each (generator seed 49):")
    good = True
    for lay in LAYOUTS:
        t = TASKS[lay]
        S, P, n = t.S, t.P, t.S * t.P
        tok, qs = t.make_batch(N, torch.Generator().manual_seed(49))
        ctx, key, val = triples(tok, t)
        once = bool((torch.sort(ctx * P + key, 1).values == torch.arange(n)).all())
        val_of = torch.zeros(N, S * P, dtype=tok.dtype).scatter_(1, ctx * P + key, val).view(N, S, P)
        distinct = bool((torch.sort(val_of, 1).values.diff(dim=1) != 0).all())
        qc, qkey, qv = tok[:, 3 * n], tok[:, 3 * n + 1] - S, tok[:, 3 * n + 2]
        answered = bool((qc == qs.reshape(-1)).all()) and bool(
            (qv == val_of[torch.arange(N), qc, qkey]).all())
        fg, fb, adj = structure(tok, t)
        line = (f"         {lay:<8} each (stream, key) once: {once}   S distinct values per key: "
                f"{distinct}   query answered by its own stream's value: {answered}")
        g = once and distinct and answered
        if lay == "grouped":
            extra = (f"grouped structure {fg:.3f}   adjacent triples sharing a key {adj:.4f} "
                     f"(4 of the 7 adjacent pairs are a key's own two triples: 4/7 = {4 / 7:.4f})")
            g &= fg == 1.0
        elif lay == "blocked":
            first0 = (ctx[:, 0] == 0).float().mean().item()
            same_korder = (key[:, :P] == key[:, P:]).all(1).float().mean().item()
            g &= fb == 1.0 and abs(first0 - 0.5) <= 0.02
            extra = (f"streams contiguous {fb:.3f}   block order (S0 first) {first0:.4f} (0.5 +- 0.02)"
                     f"   both blocks in the same key order {same_korder:.4f} (1/P! = "
                     f"{1 / math.factorial(P):.4f} if independent)")
        else:
            slot0 = [(ctx[:, i] == 0).float().mean().item() for i in range(n)]
            g &= all(abs(f - 0.5) <= 0.02 for f in slot0) and abs(adj - (S - 1) / (n - 1)) <= 0.01
            extra = (f"stream 0 per slot {[round(f, 3) for f in slot0]} (0.5 +- 0.02)   adjacent "
                     f"triples sharing a key {adj:.4f} (1/7 = {1 / 7:.4f} +- 0.01)   grouped "
                     f"structure {fg:.4f}, blocked {fb:.4f} by chance")
        good &= g
        print(line)
        print(f"         {'':<8} {extra}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 50 evaluation, probe and training batches follow the arm's layout: arm A seed 3, "
          "20 steps,")
    print("         evaluations every 10, every MultiBDH input captured by a global forward pre-hook:")
    good = True
    for lay in LAYOUTS:
        t = TASKS[lay]
        cap = {"train": [], "eval": []}
        h = register_module_forward_pre_hook(
            lambda mod, args: cap["train" if mod.training else "eval"].append(args[0].clone())
            if isinstance(mod, MultiBDH) else None)
        try:
            r = run_one(A, 3, 20, eval_every=10, task=t, stats_fn=layout_stats)
        finally:
            h.remove()
        ev_in = eval_batch(t)[:, :-1]
        pr_in = probe_batch(t, 3)[0]
        has_eval = any(x.shape == ev_in.shape and torch.equal(x, ev_in) for x in cap["eval"])
        has_probe = any(x.shape == pr_in.shape and torch.equal(x, pr_in) for x in cap["eval"])
        rows = {}
        for mode in ("train", "eval"):
            rows[mode] = structure(torch.cat(cap[mode]), t) + (sum(len(x) for x in cap[mode]),)
        if lay == "grouped":
            g = all(rows[m][0] == 1.0 for m in rows)
        elif lay == "blocked":
            g = all(rows[m][1] == 1.0 for m in rows)
        else:
            g = all(rows[m][0] < 0.05 and rows[m][1] < 0.1 and abs(rows[m][2] - 1 / 7) < 0.03
                    for m in rows)
        g &= has_eval and has_probe and r["ok"] and "margin" in r["end"]
        good &= g
        print(f"         {lay:<8} the arm's eval_batch and probe batch were used: {has_eval}, "
              f"{has_probe}")
        for m in ("train", "eval"):
            fg, fb, adj, nn = rows[m]
            print(f"         {'':<8} {m:<5} inputs ({nn:>5} rows): grouped structure {fg:.3f}   "
                  f"blocked structure {fb:.3f}   adjacent same key {adj:.3f}")
        print(f"         {'':<8} -> {'FOLLOWS THE LAYOUT' if g else 'WRONG'}")
        print(f"         {'':<8} e.g. {decode(t.make_batch(1, torch.Generator().manual_seed(50))[0][0], t)}")
    ok &= good
    print()

    print("CHECK 51 the new statistics, on the probe batch of seed 0 in each layout:")
    good = True
    for lay in LAYOUTS:
        t = TASKS[lay]
        probe = probe_batch(t, 0)
        perf = routing_stats(make_fn(_BASE["ceiling"], 0)(), t, probe)
        uni = routing_stats(make_fn(_BASE["floor"], 0)(), t, probe)
        g = (abs(perf["eta_key_by_stream"] - 1) < 1e-9 and abs(perf["eta_val_by_stream"] - 1) < 1e-9
             and perf["margin"] == 1.0 and uni["margin"] == 0.0)
        good &= g
        print(f"         {lay:<8} perfect gate: eta^2 by stream key {perf['eta_key_by_stream']:.6f} "
              f"val {perf['eta_val_by_stream']:.6f}   by key {perf['eta_key_by_key']:.4f}   by half "
              f"{perf['eta_key_by_half']:.4f}   margin {perf['margin']}")
        print(f"         {'':<8} uniform gate: margin {uni['margin']}   eta^2 by stream "
              f"{uni['eta_key_by_stream']} (p constant)  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 52 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("A_blocked", "B_shuffled"):
        sp = spec(key, 1, EVAL_EVERY)
        rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        extra = (f"margin at the evaluation {rh['stats'][-1]['margin']:.4f}" if key.startswith("A")
                 else f"acc {rh['acc']:.4f}")
        print(f"         {key:<10} seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   "
              f"{extra}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def band(c):
    return "RELIABLE" if c >= RELIABLE_K else "MAJORITY" if c >= MAJORITY_K else "MINORITY"


def ok_runs(store, key):
    return [(s, r) for s in SEEDS[key] if (r := get(store, key, s)) and r.get("ok")]


def stat_at(r, step):
    return next((x for x in r["stats"] if x["step"] == step), None)


def report(store, wall, path, also):
    n = len(SEEDS["A_grouped"])
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<11} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>6} "
          f"{'spread':>6} {'key_part':>8} {'margin':>7}  eta^2 at key positions by stream/key/half/"
          f"index   flags")
    for k in A_KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<11} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<11} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            fl = (["DISCOVERED"] if r["discovered"] else [fail_class(r)]) + (
                ["collapsed"] if r["collapsed"] else [])
            etas = "/".join(f"{e[f'eta_key_by_{g}']:.2f}" for g in ETA_GROUPS)
            print(f"  {k:<11} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                  f"{e['val_cos']:>8.4f} {e['sep']:>6.3f} {e['spread']:>6.3f} {e['key_part']:>8.4f} "
                  f"{e['margin']:>7.4f}  {etas:<23}   {' '.join(fl)}")
        print()
    print(f"  {'arm':<16} {'seed':>4} {'acc':>7} {'transition':>10} {'stopped':>7}  flags")
    for k in KEYS:
        if k in A_KEYS:
            continue
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<16} {s:>4}  NOT RUN")
            elif not r.get("ok"):
                print(f"  {k:<16} {s:>4}  FAILED — {r['error']}")
            else:
                print(f"  {k:<16} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{r['stopped_at']:>7}  {'BOUND' if r['transition'] else ''}")
        print()

    print("=" * 100)
    print(f"COUNTS (discovered = bound AND final VAL cos < {DISC_COS})")
    print("=" * 100)
    c, bound = {}, {}
    for a in ARMS:
        k = a["key"]
        ns = len(SEEDS[k])
        c[k] = sum(disc(store, k, s) for s in SEEDS[k])
        bound[k] = count(store, k, SEEDS[k], "transition")
        rs = [get(store, k, s) for s in SEEDS[k]]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        print(f"  {a['label']:<38} discovered {c[k]:>2}/{ns}   bound {bound[k]:>2}/{ns}   completed "
              f"{sum(1 for r in rs if r and r.get('ok'))}/{ns}" + (f"   [{failed} FAILED]" if failed else ""))
    print()

    print("#" * 100)
    print("VALIDITY (per new layout)")
    print("#" * 100)
    fit = store.get("fit", {})
    valid, discr = {}, {}
    for lay in NEW_LAYOUTS:
        ck, bk = f"ceiling_{lay}", f"B_{lay}"
        am = fit.get(lay, {}).get("argmax", float("nan"))
        valid[lay] = bound[ck] == len(SEEDS[ck]) and am >= FIT_PASS
        discr[lay] = bound[bk] == 0
        print(f"  {lay:<8} ceiling binds {bound[ck]}/{len(SEEDS[ck])}   recurrent gate fits the perfect "
              f"routing: argmax {am:.4f} (>= {FIT_PASS}), mse {fit.get(lay, {}).get('mse', float('nan')):.2e}")
        print(f"     *** {lay}: {'VALID' if valid[lay] else 'UNTESTED'} ***")
        print(f"  {lay:<8} single-channel B binds {bound[bk]}/{len(SEEDS[bk])}")
        print(f"     *** {lay}: {'DISCRIMINATIVE' if discr[lay] else 'COMPROMISED'} ***")
    print(f"  (grouped layout, for reference: the recurrent gate fit gives argmax "
          f"{fit.get('grouped', {}).get('argmax', float('nan')):.4f})")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS")
    print("#" * 100)
    v = dict(counts=c, bound=bound, valid=valid, discriminative=discr)
    for name, lay, title in (("L1", "blocked", "BLOCKS HELP"), ("L2", "shuffled", "SHUFFLING HELPS")):
        k = f"A_{lay}"
        p = fisher_greater(c[k], n, c["A_grouped"], n)
        raw = "HELPS" if p < ALPHA else "NOT SHOWN"
        verdict = raw if valid[lay] else "UNTESTED"
        print(f"  {name} {title} (Fisher one-sided): {k} {c[k]}/{n} vs A_grouped {c['A_grouped']}/{n}; "
              f"p = {p:.3g}")
        note = ("" if not valid[lay] or discr[lay] else
                f"   (layout {lay} COMPROMISED: printed, not interpreted as evidence for channels)")
        print(f"     *** {name}: {verdict} ***{note}"
              + ("" if valid[lay] else f"   (layout {lay} not VALID; the test result would be {raw})"))
        v[name], v[f"{name}_p"], v[f"{name}_interpreted"] = verdict, p, valid[lay] and discr[lay]
    print(f"  BANDS (RELIABLE >= {RELIABLE_K}/40, MAJORITY {MAJORITY_K}-{RELIABLE_K - 1}, MINORITY "
          f"<= {MAJORITY_K - 1}):")
    for k in A_KEYS:
        print(f"     *** {k:<10} {c[k]}/{n}: {band(c[k])} ***")
        v[f"band_{k}"] = band(c[k])
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    seeds = SEEDS["A_grouped"]
    print("  per-seed outcome (D = discovered, . = not):")
    print("    seed  " + "  ".join(f"{k:>10}" for k in A_KEYS))
    for s in seeds:
        print(f"    {s:>4}  " + "  ".join(f"{('D' if disc(store, k, s) else '.'):>10}" for k in A_KEYS))
    for k1 in ("A_blocked", "A_shuffled"):
        b = sum(disc(store, k1, s) and not disc(store, "A_grouped", s) for s in seeds)
        cc = sum(disc(store, "A_grouped", s) and not disc(store, k1, s) for s in seeds)
        print(f"    McNemar exact two-sided, {k1} vs A_grouped: {k1} only {b}, A_grouped only {cc}; "
              f"p = {mcnemar_exact(b, cc):.3g}")
    print()
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}) per discovering run "
          f"(escape/transition):")
    for k in A_KEYS:
        es = [(s, escape(r), r["transition"]) for s, r in ok_runs(store, k) if r["discovered"]]
        print(f"    {k:<10} " + ("  ".join(f"s{s}:{fmt_step(e)}/{fmt_step(tr)}" for s, e, tr in es)
                              if es else "none"))
    print()
    print("  eta^2 of the read gate's channel-0 probability and the query routing margin, median "
          "[min, max],")
    print("  at the first evaluation (1200) and at the end, discovered (D) and not (N) runs apart:")
    for k in A_KEYS:
        for lab, want in (("D", True), ("N", False)):
            rs = [r for _, r in ok_runs(store, k) if r["discovered"] == want]
            for when in (EVAL_EVERY, "end"):
                xs = [x for r in rs if (x := (r["end"] if when == "end" else stat_at(r, when)))
                      and "margin" in x]
                if not xs:
                    continue
                print(f"    {k:<10} {lab} ({len(rs):>2}) @{str(when):<4} margin {med([x['margin'] for x in xs])}")
                for role in ("key", "val"):
                    print(f"    {'':<10} {'':<6} {'':<5} {role} positions: " + "  ".join(
                        f"{g} {med([x[f'eta_{role}_by_{g}'] for x in xs])}" for g in ETA_GROUPS))
    print()
    print(f"  failure classes of non-discovering runs (end statistics; STREAM-PARTIAL margin >= "
          f"{MARGIN_PARTIAL}; KEY eta^2 by key at key")
    print(f"  positions >= {ETA_SPLIT}; POSITION eta^2 by half or index at key positions >= {ETA_SPLIT}; "
          f"OTHER), with the old key_part classification:")
    for k in A_KEYS:
        classes = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        old = {"saddle": 0, "locked on keys": 0, "other": 0}
        for s, r in ok_runs(store, k):
            if not r["discovered"]:
                classes[fail_class(r)].append(s)
                old[mode_of(r)] += 1
        print(f"    {k:<10} " + "   ".join(f"{m}: {len(x)}" for m, x in classes.items())
              + "      old: " + "   ".join(f"{m}: {x}" for m, x in old.items()))
        for m, x in classes.items():
            if x:
                print(f"    {'':<10}   {m:<14} " + " ".join(f"s{s}" for s in x))
    print()
    print("  learning speed without routing: B's held-out accuracy (x100) at 1200, 2400, 6000, 12000 "
          "and the end:")
    for k in ("B_blocked", "B_shuffled"):
        for s, r in ok_runs(store, k):
            print(f"    {k:<10} s{s} " + " ".join(
                f"{round(a * 100) if (a := curve_acc(r, st)) is not None else '--':>4}"
                for st in (1200, 2400, 6000, 12000)) + f"  end {round(r['acc'] * 100):>3}")
        accs = [r["acc"] for _, r in ok_runs(store, k)]
        if accs:
            print(f"    {k:<10} final accuracy median {med(accs)}")
    print("  arm A's held-out accuracy at 1200 and 2400, median [min, max]:")
    for k in A_KEYS:
        rs = [r for _, r in ok_runs(store, k)]
        for lab, sub in (("all", rs), ("discovered", [r for r in rs if r["discovered"]]),
                         ("not", [r for r in rs if not r["discovered"]])):
            a1 = [x for r in sub if (x := curve_acc(r, 1200)) is not None]
            a2 = [x for r in sub if (x := curve_acc(r, 2400)) is not None]
            print(f"    {k:<10} {lab:<10} ({len(sub):>2})  @1200 {med(a1)}   @2400 {med(a2)}")
    print()
    print(f"  the restart check per layout (held-out accuracy >= {A_CHECK} at step {T_CHECK}):")
    for k in A_KEYS:
        rs = ok_runs(store, k)
        passed = [(s, r) for s, r in rs if (x := curve_acc(r, T_CHECK)) is not None and x >= A_CHECK]
        late = [s for s, r in rs if r["discovered"] and (curve_acc(r, T_CHECK) or 0.0) < A_CHECK]
        miss = [s for s, r in passed if not r["discovered"]]
        print(f"    {k:<10} pass {len(passed)}/{len(rs)}; of those discovered "
              f"{len(passed) - len(miss)}/{len(passed)}"
              + (f" (not: {' '.join(f's{s}' for s in miss)})" if miss else "")
              + f"; discoveries that would fail the check: {len(late)}")
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; rest = all other parameters), median [min, max] "
          "over seeds:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            ga, re = [x[0] for x in g], [x[1] for x in g]
            print(f"    step {step:>3} {k:<16} gate {statistics.median(ga):.3e} [{min(ga):.3e}, "
                  f"{max(ga):.3e}]   rest {statistics.median(re):.3e} [{min(re):.3e}, {max(re):.3e}]")
    print()
    print("  runs that bound but did not discover:")
    for k in A_KEYS:
        bd = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None and not r["discovered"]]
        print(f"    {k:<10} " + ("  ".join(f"s{s} (bound @{r['transition']}, VAL cos {r['val_cos']:.3f}, "
                                          f"margin {r['end']['margin']:.3f})" for s, r in bd)
                              if bd else "none"))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for k in KEYS:
                runs = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                d2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                b2 = sum(1 for r in runs if r.get("ok") and r.get("transition"))
                ns = len(SEEDS[k])
                print(f"    {k:<16} discovered here {c[k]}/{ns}   other {d2}/{len(runs)}   pooled "
                      f"{c[k] + d2}/{ns + len(runs)}      bound here {bound[k]}/{ns}   other "
                      f"{b2}/{len(runs)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — accuracy x100 and VAL cos x100 at each evaluation every {EVAL_EVERY} steps")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            trr.curve_lines(f"{k:<16} s{s}", r)
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Does the binding layout change router discovery?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's router_layout_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    all_seeds = set().union(*SEEDS.values())

    print("=" * 100)
    print("Router layout: does the order of the streams' bindings change router discovery?")
    print(f"  test_router_confirm's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{trc.SUB_LR}), MAX_ITERS={MAX_ITERS}; layouts {', '.join(LAYOUTS)}")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<38} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  all seeds disjoint from earlier seeds 0-119: {not all_seeds & set(EARLIER_SEEDS)}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        store["fit"] = {}
        for lay in LAYOUTS:
            mse, am = gate_fit_k(TASKS[lay], A)
            store["fit"][lay] = dict(mse=mse, argmax=am)
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step (200 steps) inside the pool, evaluation charged once per "
              f"{EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_arm, [ARM[k] for k in KEYS])))
        for k in KEYS:
            print(f"  {k:<16} {costs[k][0] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds")
        allspecs = [spec(k, s, MAX_ITERS) for k in KEYS for s in SEEDS[k]]
        durs = [costs[sp["arm"]][0] * sp["iters"] for sp in allspecs]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(allspecs)} runs")
        print("=" * 100)
        todo = []
        for sp in allspecs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<16} seed {sp['seed']}  cached")
            else:
                todo.append(sp)
        todo.sort(key=lambda sp: -costs[sp["arm"]][0])
        futs = {pool.submit(run_job, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            if rec.get("ok"):
                e, tr = rec["end"], rec["transition"]
                etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS) if "margin" in e else ""
                extra = (f"VALcos={e['val_cos']:.3f} margin={e['margin']:.3f} "
                         f"eta_key(stream/key/half/idx)={etas}" if "margin" in e else "")
                print(f"   {sp['arm']:<16} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}  {extra}"
                      f"{'  DISCOVERED' if rec['discovered'] else ''}  {rec['secs']:.0f}s  "
                      f"(elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<16} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
