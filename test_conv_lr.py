#!/usr/bin/env python
"""
test_conv_lr.py — does a higher learning rate, 4e-3, change what test_short_conv found at
1e-3, when the short causal convolution is present? Everything below is fixed before any
result.

Run it directly:

    python test_conv_lr.py --short results/X/short_conv_results.json
    python test_conv_lr.py --short ... --also results/L/conv_lr_results.json

BACKGROUND
- test_short_conv, both machines, seeds 160-199, lr 1e-3:
  - K1 NOT SHOWN: B_conv 2/40 on X and 5/40 on L; B 0/20 on each.
  - K2 DIFFERENT, with A higher.
  - K3 SHOWN: A_conv 23/40 on X and 28/40 on L.
  - Post hoc: on the same seeds, A_conv bound where A did not far more than the reverse
    (X 13 vs 2, L 16 vs 2). A itself was low on this seed set (12/40 and 14/40, against
    ~64% pooled before).
  - Diagnostics: runs that bound WITHOUT stream routing ended with a larger lag-2
    convolution weight, lag 2 being where a value position sees its context token.
    - A_conv without routing: lag 2 at 0.05-0.15, against about 0.01-0.02 for routed binders.
    - B_conv binders: 0.053-0.057, against 0.031-0.033 for non-binders.
- Scratch at lr 4e-3, through test_short_conv's own run path (run_one with SUB_LR
  overridden). At 1e-3 the same path reproduces X's recorded step-1200 accuracies exactly.
  Exploratory only:
  - B_conv, seeds 160-163: bound 2/4 (162 at step 1200, 163 at 10800). At 1e-3 these bound
    0/4 on X and 1/4 on L.
  - A_conv, seeds 161 and 162 (both failed at 1e-3 on both machines): bound 2/2 at step 3600,
    with routing.
  - Without the convolution a higher lr hurts: A seed 160 stuck at 0.31, and B seed 3 at 0.20.
  - B_wide_conv (below), seed 160: stayed at 0.51 (B_conv 160 also failed at 4e-3).
  - Lag 2 at the end: the single-channel binders had 0.19-0.21, the non-binders 0.06-0.10.
- Seed-level outcomes differ between machines; the verdict is per machine.

DESIGN
- Paired with test_short_conv's recorded runs. The new arms reuse seeds 160-199 on purpose:
  on each machine the lr 1e-3 outcome of every seed is already in short_conv_results.json.
  Each new run therefore shares its initial parameters and batches with a recorded run, and
  only the learning rate differs.
- test_router_reliability.run_one gained an lr knob (default None = SUB_LR, unchanged) and a
  builder knob (a model builder for Part 2's tasks, whose vocabulary differs; default None =
  make_fn, unchanged) (CHECK 58). The task, substrate, conv and everything else are
  test_short_conv's.
- --short points to this machine's test_short_conv results (default short_conv_results.json;
  on X, results/X/short_conv_results.json). The file's meta CPU must match this machine and
  two recorded runs must reproduce exactly (CHECK 59). Otherwise the paired claims are
  UNTESTED.

ARMS (lr 4e-3, P=4, S=2, grouped layout, MAX_ITERS 24000; test_short_conv's recipe otherwise)
  B_conv4        single channel + conv 'layer'                                seeds 160-199 (40)
  A_conv4        arm A + conv 'layer'                                         seeds 160-199 (40)
  B_wide_conv4   single channel at N=512 (mult 16) + conv 'layer'; its fast-
                 weight state equals A_conv's (1 x 512 x D vs 2 x 256 x D per
                 layer, CHECK 60)                                             seeds 160-179 (20)
  A4             arm A, no conv (descriptive)                                 seeds 160-169 (10)
  B4             plain single channel (control)                               seeds 160-169 (10)
  ceiling_conv4  perfect gate + conv 'layer' (validity)                       seeds 160-164 (5)
Part 2, descriptive only; it sets up P-scaling (test_channel_binding.task_for, CHECK 62):
  G0_P8_conv4       S=1, P=8, single channel + conv (does the substrate bind P=8?) 160-164
  ceiling_P8_conv4  S=2, P=8, perfect gate + conv                                  160-164

VALIDITY AND PRE-REGISTERED CLAIMS (Part 1; outcome BOUND = transition not None; DISCOVERED
printed for gated arms)
  VALID: ceiling_conv4 binds 5/5, AND the pairing CHECK (59) passes. Otherwise every claim is
    UNTESTED.
  LR1 ONE CHANNEL BINDS MORE AT 4e-3: B_conv4 vs the recorded B_conv at 1e-3, same seeds;
      exact McNemar, one-sided (4e-3 higher).
  LR2 GATE + CONV BINDS MORE AT 4e-3: A_conv4 vs the recorded A_conv, the same test.
  LR3 THE GATE STILL ADDS AT 4e-3: A_conv4 vs B_conv4; Fisher one-sided (A_conv4 higher),
      the McNemar test printed alongside.
  LR4 TWO CHANNELS BEAT ONE CHANNEL WITH THE SAME MEMORY: A_conv4 (40) vs B_wide_conv4 (20);
      Fisher one-sided (A_conv4 higher), the McNemar test on seeds 160-179 alongside.
  M1  LAG 2 MARKS SINGLE-CHANNEL BINDING: B_conv4 and B_wide_conv4 runs pooled; bound runs
      have a larger end lag-2 mean |w| than unbound runs: exact Mann-Whitney U, one-sided
      (conditional on ties). UNTESTABLE if either group has fewer than 3 runs.
  Each claim is SHOWN if p < 0.05, else NOT SHOWN.
  Bands for B_conv4 and A_conv4, on BOUND: RELIABLE >= 36/40, MAJORITY 20-35, MINORITY 1-19,
  NEVER 0.
  Readings, each printed verbatim when it applies:
  - LR3 NOT SHOWN: "at 4e-3 one channel with the convolution binds about as often as the
    gated model; channels must earn their place on capacity."
  - LR3 SHOWN and LR4 SHOWN: "two channels beat one channel of the same total memory: the
    partition helps, not just the memory size."
  - LR3 SHOWN and LR4 NOT SHOWN: "one channel with the same total memory does about as well
    as two: the gain may be memory size."
  - A_conv4 RELIABLE: "gate + conv at 4e-3 binds reliably without restarts; carry it into
    P-scaling."
Fisher and McNemar (two-sided) are test_router_confirm's; the one-sided McNemar and the
Mann-Whitney test are here, checked against hand values (CHECK 61).

DIAGNOSTICS, not part of the verdict
- Per seed: the outcome at 1e-3 (recorded) and at 4e-3 for B_conv, A_conv and A; McNemar
  tests for every arm pair within 4e-3.
- test_short_conv's diagnostics for the new arms: routed vs unrouted binders and failure
  classes; the convolution's mean |w| per lag by outcome; transition steps and plateau
  length; the restart check; gradient norms at steps 1, 10 and 100; collapsed runs (final
  accuracy < 0.15; A4 and B4 may collapse at this lr).
- Part 2: per seed, bound or not, transition step and final accuracy.
- --also pooled counts.

CHECKS (printed before any training): test_short_conv's verification (which runs every
earlier test's), then
  58 the lr and builder knobs are inert at their defaults (B_conv, A_conv vs 6c90a19's
     run_one, weights and records); every new arm's optimizer runs at 4e-3;
  59 the pairing: the --short file, its CPU, and two recorded runs reproduced exactly
     (recorded, not asserted: a failure makes the claims UNTESTED);
  60 B_wide_conv4 is k=1, N=512, conv 'layer', with A_conv's fast-weight state;
  61 the one-sided McNemar and the Mann-Whitney code reproduce hand values, ties included;
  62 Part 2's tasks are task_for(8, 1) and task_for(8, 2); the perfect gate at P=8 applies at
     every layer (CHECK 6's test, conv on);
  63 a worker's run is bit-identical to the same run in the main process.

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to conv_lr_results.json (gitignored; copied into results/X/
after the run).

RESULT (full run, 135/135 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz,
4 workers x 1 thread; 249.2 min after verification, projection 7.79 h worst case; paired with
results/X/short_conv_results.json; no --also file; machine X, commit 3c69afd).
  VALID: ceiling_conv4 bound 5/5 (all at step 1200) and the pairing CHECK (59) passed.
  LR1 ONE CHANNEL BINDS MORE AT 4e-3: SHOWN. B_conv4 18/40 vs the recorded B_conv 2/40 at
      1e-3; 4e-3 only 17, 1e-3 only 1; one-sided McNemar p = 7.25e-05.
  LR2 GATE + CONV BINDS MORE AT 4e-3: NOT SHOWN. A_conv4 22/40 vs the recorded A_conv 23/40;
      4e-3 only 10, 1e-3 only 11; p = 0.668.
  LR3 THE GATE STILL ADDS AT 4e-3: NOT SHOWN. A_conv4 22/40 vs B_conv4 18/40; Fisher one-sided
      p = 0.251; McNemar on the pairs: A_conv4 only 12, B_conv4 only 8, p = 0.503.
  LR4 TWO CHANNELS BEAT ONE CHANNEL WITH THE SAME MEMORY: NOT SHOWN. A_conv4 22/40 vs
      B_wide_conv4 10/20; Fisher one-sided p = 0.463; McNemar on seeds 160-179: A_conv4 only 5,
      B_wide_conv4 only 3, p = 0.727.
  M1 LAG 2 MARKS SINGLE-CHANNEL BINDING: SHOWN. B_conv4 + B_wide_conv4: the 28 bound runs have
      end lag-2 mean |w| median 0.158 [0.075, 0.295], the 32 unbound runs 0.082 [0.050, 0.120];
      U = 856, one-sided p = 2.07e-12.
  Bands (bound): B_conv4 18/40 MINORITY, A_conv4 22/40 MAJORITY.
  The pre-registered reading: LR3 NOT SHOWN: "at 4e-3 one channel with the convolution binds
  about as often as the gated model; channels must earn their place on capacity."

  Diagnostics (not part of the verdict):
  - The BACKGROUND's scratch runs reproduced exactly: B_conv4 162 bound at step 1200 and 163
    at 10800; A_conv4 161 and 162 at 3600; B_wide_conv4 160 not bound; A4 160 ended at 0.31.
  - B_conv's two binders at 1e-3 (seeds 181, 184): at 4e-3, 181 bound (step 21600) and 184 did
    not. Single-channel binding at 4e-3 is late: B_conv4 transition median 15000 [1200, 24000]
    after a plateau at ~0.50 of median 12600 steps; B_wide_conv4 median 16800 [2400, 21600].
    Every unbound B_conv4, B_wide_conv4 and A_conv4 run but one ended at 0.48-0.52 (A_conv4
    seed 170: 0.62).
  - B_conv4 vs B_wide_conv4 on seeds 160-179: 10/20 each (6 vs 6 discordant). Doubling one
    channel's memory to A_conv's fast-weight state did not change its binding rate here.
  - The gated model at 4e-3 mostly bound WITHOUT routing: of A_conv4's 22 binders, 6 route
    (margin >= 0.9) and 16 do not; only 7/40 were DISCOVERED (VAL cos < 0.5). At 1e-3 the same
    seeds had 23 bound, 16 routed, 17 discovered. Post hoc (not a claim): discovered at 1e-3
    only 14, at 4e-3 only 4, McNemar p = 0.0309. So the higher learning rate lifted one channel
    to about the gated model's rate, and moved the gated model from routing towards binding
    the way one channel does, at the same overall rate.
  - The unrouted A_conv4 binders carry the single-channel signature: lag 2 median 0.165, vs
    0.082 for the routed ones (and 0.149 / 0.081 for B_conv4 bound / unbound, 0.165 / 0.086 for
    B_wide_conv4).
  - A_conv4's 18 failures: POSITION 10, OTHER 5, KEY 3.
  - Without the convolution the higher lr hurt: A4 0/10 (the recorded A bound 4/10 on these
    seeds at 1e-3), B4 0/10; final accuracies 0.21-0.51 for both; none collapsed (< 0.15). Two A4
    runs separated the streams (VAL cos < 0.5, STREAM-PARTIAL; seeds 162, 165) and still did
    not bind.
  - The restart check (>= 0.6 at step 2400) does not transfer to 4e-3: it passed 3/40 A_conv4
    runs (all bound) and would have thrown away 19 of its 22 binders; for B_conv4 it passed 1/40
    and would have thrown away 17 of 18. No A_conv4 run bound at step 1200 (A_conv at 1e-3:
    13 of 23).
  - Step-1 gradient norms: conv median 5.6e-02, gate ~8e-06, as at 1e-3.
  - Part 2 (descriptive): G0_P8_conv4 (S=1, P=8, one channel + conv) bound 5/5, four of them at
    step 1200 and one at 3600. ceiling_P8_conv4 (S=2, P=8, perfect gate + conv) bound 5/5, at
    steps 1200, 2400, 8400, 9600 and 15600.
"""

import argparse
import itertools
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
from torch.optim.optimizer import register_optimizer_step_pre_hook

from test_binding_onset import evaluate, time_per_step, fmt_step, EVAL_EVERY
from test_multilayer_binding import n_params, n_state, D
import test_channel_binding as tcb
from test_channel_binding import eval_batch, arms as cb_arms
from test_router_confirm import fisher_greater, mcnemar_exact
import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, make_fn, get, count, load_store, init_worker, makespan, cpu_model,
    same_weights, save_results, COLLAPSE, TAU_END, TIME_STEPS,
)
from test_router_discovery import load_at, model_fn
from test_router_reliability import run_one, strip_all, curve_acc, med, curve_lines, T_CHECK, A_CHECK
from test_router_layout import fail_class, ETA_GROUPS
from test_readout_path import GRAD_STEPS
import test_short_conv as tsc
from test_short_conv import conv_stats, conv_grad_norms, plateau, routed, wstr, band, LAYER, CONV_K

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
LR4 = 4e-3
ALPHA = 0.05
M1_MIN = 3                                 # M1 is UNTESTABLE below 3 runs in either group
LEGACY_SHA = "6c90a19"                     # head before this test's changes
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "conv_lr_results.json"
SHORT_FILE = "short_conv_results.json"

TASKS = {"P4S2": TASK, "P8S1": tcb.task_for(8, 1), "P8S2": tcb.task_for(8, 2)}
_BASE = {a["key"]: a for a in cb_arms(TASK)}
A, B = trc.ARM["A"], _BASE["B"]
ARMS = [
    dict(tsc.ARM["B_conv"], key="B_conv4", task="P4S2", lr=LR4,
         label="B_conv4           single channel + conv 'layer'"),
    dict(tsc.ARM["A_conv"], key="A_conv4", task="P4S2", lr=LR4,
         label="A_conv4           arm A + conv 'layer'"),
    dict(B, key="B_wide_conv4", mult=B["mult"] * 2, model_kw=LAYER, task="P4S2", lr=LR4,
         label="B_wide_conv4      single channel N=512 + conv 'layer'"),
    dict(A, key="A4", task="P4S2", lr=LR4, label="A4                arm A, no conv"),
    dict(B, key="B4", task="P4S2", lr=LR4, label="B4                plain single channel"),
    dict(tsc.ARM["ceiling_conv"], key="ceiling_conv4", task="P4S2", lr=LR4,
         label="ceiling_conv4     perfect gate + conv 'layer'"),
    dict(B, key="G0_P8_conv4", model_kw=LAYER, task="P8S1", lr=LR4,
         label="G0_P8_conv4       S=1 P=8, single channel + conv"),
    dict({a["key"]: a for a in cb_arms(TASKS["P8S2"])}["ceiling"], key="ceiling_P8_conv4",
         model_kw=LAYER, task="P8S2", lr=LR4,
         label="ceiling_P8_conv4  S=2 P=8, perfect gate + conv"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
PART2 = ("G0_P8_conv4", "ceiling_P8_conv4")
PART1 = tuple(k for k in KEYS if k not in PART2)
GATED = tuple(k for k in KEYS if ARM[k]["gate"] != "none")
CONV_KEYS = tuple(k for k in KEYS if ARM[k].get("model_kw", {}).get("conv"))
SEEDS = dict(B_conv4=tuple(range(160, 200)), A_conv4=tuple(range(160, 200)),
             B_wide_conv4=tuple(range(160, 180)), A4=tuple(range(160, 170)),
             B4=tuple(range(160, 170)), ceiling_conv4=tuple(range(160, 165)),
             G0_P8_conv4=tuple(range(160, 165)), ceiling_P8_conv4=tuple(range(160, 165)))
RECORDED = dict(B_conv4="B_conv", A_conv4="A_conv", A4="A", B4="B")   # 4e-3 arm -> 1e-3 arm
READINGS = {
    "LR3-": "at 4e-3 one channel with the convolution binds about as often as the gated model; "
            "channels must earn their place on capacity.",
    "LR3+ LR4+": "two channels beat one channel of the same total memory: the partition helps, "
                 "not just the memory size.",
    "LR3+ LR4-": "one channel with the same total memory does about as well as two: the gain "
                 "may be memory size.",
    "RELIABLE": "gate + conv at 4e-3 binds reliably without restarts; carry it into P-scaling.",
}


# ── Exact tests ──────────────────────────────────────────────────────────────
def mcnemar_greater(b, c):
    """One-sided exact McNemar: P(Bin(b + c, 1/2) >= b), b = pairs where the first arm alone
    succeeded."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def mann_whitney_greater(x, y):
    """One-sided exact Mann-Whitney U (x larger), conditional on ties: U from midranks, and
    p = P(rank sum of x >= observed) over all C(N, len(x)) equally likely assignments of the
    observed midranks."""
    allv = sorted(x + y)
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        rank[allv[i]] = i + j + 2                 # doubled midrank: 2 * (i + j + 2) / 2
        i = j + 1
    r2 = [rank[v] for v in x + y]
    n1 = len(x)
    obs = sum(r2[:n1])
    U = obs / 2 - n1 * (n1 + 1) / 2
    ways = [dict() for _ in range(n1 + 1)]
    ways[0][0] = 1
    for r in r2:
        for k in range(n1 - 1, -1, -1):
            for s, c in ways[k].items():
                ways[k + 1][s + r] = ways[k + 1].get(s + r, 0) + c
    tot = math.comb(len(r2), n1)
    return U, sum(c for s, c in ways[n1].items() if s >= obs) / tot


# ── Worker-side ──────────────────────────────────────────────────────────────
def builder_for(a):
    """Part 1 keeps run_one's own builder (test_short_conv's path, so the pairing holds);
    Part 2's tasks have another vocabulary and are built with test_router_discovery.model_fn."""
    if a["task"] == "P4S2":
        return None
    task = TASKS[a["task"]]
    return lambda aa, seed: model_fn(task, aa, seed)


def run_job(sp):
    a = ARM[sp["arm"]]
    return run_one(a, sp["seed"], sp["iters"], task=TASKS[a["task"]], stats_fn=conv_stats,
                   grad_fn=conv_grad_norms, lr=a["lr"], builder=builder_for(a))


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def time_job(key):
    """test_router_curriculum.time_arm on the arm's own task (Part 2's sequences are longer)."""
    a = ARM[key]
    task = TASKS[a["task"]]
    mk = lambda: model_fn(task, a, 0)()
    per = time_per_step(task, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(task)
    t0 = time.time()
    evaluate(m, task, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


# ── The pairing (CHECK 59) ───────────────────────────────────────────────────
def pairing_check(pool, short_path):
    """The --short file exists, was written on this CPU, and two of its runs reproduce."""
    info = dict(path=short_path, exists=os.path.exists(short_path))
    if not info["exists"]:
        info.update(cpu_match=False, reproduced=False, ok=False)
        return info
    short = load_store(short_path)
    info["cpu"] = short.get("meta", {}).get("cpu")
    info["git"] = short.get("meta", {}).get("git")
    info["cpu_match"] = info["cpu"] == cpu_model()
    rep = {}
    for key in ("B_conv", "A_conv"):
        rec = get(short, key, 160)
        r = pool.submit(tsc.run_job, tsc.spec(key, 160, EVAL_EVERY)).result()
        now = json.loads(json.dumps(r["curve"]))          # the stored form: lists, not tuples
        old = (rec or {}).get("curve", [])[:len(now)]
        rep[key] = bool(rec and rec.get("ok") and r["ok"] and now and now == old)
        info[f"{key}_curve"] = (now, old)
    info["reproduced"] = all(rep.values())
    info["ok"] = info["cpu_match"] and info["reproduced"]
    return info


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool, short_path):
    print("test_short_conv.py's verification (which runs test_router_layout's, and so on down")
    print("to test_multilayer_binding's):")
    tsc.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 58 the lr and builder knobs are inert at their defaults: 60-step runs (evaluations "
          f"every 20, seed 3) vs {LEGACY_SHA}'s run_one:")
    leg = load_at(LEGACY_SHA, "test_router_reliability.py", "test_router_reliability_legacy")
    good = True
    for key in ("B_conv", "A_conv"):
        a = tsc.ARM[key]
        kw = dict(eval_every=20, task=TASK, stats_fn=conv_stats, grad_fn=conv_grad_norms)
        k0, k1, k2 = {"at": -1}, {"at": -1}, {"at": -1}
        r0 = leg.run_one(a, 3, 60, keep=k0, **kw)
        r1 = run_one(a, 3, 60, keep=k1, **kw)
        r2 = run_one(a, 3, 60, keep=k2, lr=SUB_LR, builder=make_fn, **kw)
        g = (same_weights(k0["model"], k1["model"]) and same_weights(k0["model"], k2["model"])
             and strip_all(r0) == strip_all(r1) == strip_all(r2) and r1["ok"])
        good &= g
        print(f"         {key:<7} weights equal (default, explicit) {same_weights(k0['model'], k1['model'])}, "
              f"{same_weights(k0['model'], k2['model'])}   records equal {strip_all(r0) == strip_all(r1) == strip_all(r2)}"
              f"  -> {'UNCHANGED' if g else 'CHANGED'}")
    seen = []
    h = register_optimizer_step_pre_hook(
        lambda opt, args, kw: seen.append(sorted({g_["lr"] for g_ in opt.param_groups})))
    try:
        lrs = {}
        for key in KEYS:
            seen.clear()
            run_job(spec(key, 1, 2))
            lrs[key] = sorted({v for s in seen for v in s})
        seen.clear()
        tsc.run_job(tsc.spec("B_conv", 1, 2))
        lr_short = sorted({v for s in seen for v in s})
    finally:
        h.remove()
    g = all(v == [LR4] for v in lrs.values()) and lr_short == [SUB_LR]
    good &= g
    print(f"         optimizer lr captured by a step pre-hook (2 steps each): "
          + "  ".join(f"{k} {v}" for k, v in lrs.items()))
    print(f"         test_short_conv's own B_conv for reference: {lr_short}  -> {'4e-3 IN EVERY NEW ARM' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 59 the pairing with this machine's test_short_conv results ({short_path}):")
    pair = pairing_check(pool, short_path)
    if not pair["exists"]:
        print(f"         file exists: False  -> PAIRING FAILS (the paired claims will be UNTESTED)")
    else:
        print(f"         file exists: True   written on: {pair['cpu']} (git {pair['git']})   this machine: "
              f"{cpu_model()}   match: {pair['cpu_match']}")
        for key in ("B_conv", "A_conv"):
            now, rec = pair[f"{key}_curve"]
            print(f"         {key} seed 160 re-run at lr {SUB_LR}, {EVAL_EVERY} steps: {now} vs recorded {rec}")
        print(f"         reproduced exactly: {pair['reproduced']}  -> "
              f"{'PAIRED' if pair['ok'] else 'PAIRING FAILS (the paired claims will be UNTESTED)'}")
    print()

    print("CHECK 60 B_wide_conv4's model:")
    mw = make_fn(ARM["B_wide_conv4"], 0)()
    ma = make_fn(tsc.ARM["A_conv"], 0)()
    good = (mw.n_ch == 1 and mw.n_feat == 512 and mw.conv == "layer" and mw.gate_kind == "none"
            and n_state(mw) == n_state(ma))
    print(f"         B_wide_conv4: k={mw.n_ch}, N={mw.n_feat}, conv {mw.conv!r}, gate {mw.gate_kind!r}, "
          f"fast-weight state {n_state(mw)}, parameters {n_params(mw)}")
    print(f"         A_conv:       k={ma.n_ch}, N={ma.n_feat}, conv {ma.conv!r}, gate {ma.gate_kind!r}, "
          f"fast-weight state {n_state(ma)}, parameters {n_params(ma)}")
    print(f"         -> {'STATE MATCHED' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 61 the exact tests reproduce hand values:")
    cases = [
        ("McNemar one-sided b=5 c=0", mcnemar_greater(5, 0), 1 / 32),
        ("McNemar one-sided b=3 c=1", mcnemar_greater(3, 1), 5 / 16),
        ("McNemar one-sided b=0 c=0", mcnemar_greater(0, 0), 1.0),
        ("Mann-Whitney x=[3,4] y=[1,2]", mann_whitney_greater([3, 4], [1, 2]), (4.0, 1 / 6)),
        ("Mann-Whitney x=[1,2] y=[3,4]", mann_whitney_greater([1, 2], [3, 4]), (0.0, 1.0)),
        ("Mann-Whitney x=[5,6,7] y=[1,2,3,4]", mann_whitney_greater([5, 6, 7], [1, 2, 3, 4]), (12.0, 1 / 35)),
        ("Mann-Whitney ties x=[2,3] y=[1,2]", mann_whitney_greater([2, 3], [1, 2]), (3.5, 1 / 3)),
        ("Mann-Whitney ties x=[1,1] y=[1,1]", mann_whitney_greater([1, 1], [1, 1]), (2.0, 1.0)),
    ]
    good = True
    for name, got, want in cases:
        g = (all(abs(u - v) < 1e-12 for u, v in zip(got, want)) if isinstance(want, tuple)
             else abs(got - want) < 1e-12)
        good &= g
        gs = f"U {got[0]}, p {got[1]:.5f}" if isinstance(got, tuple) else f"p {got:.5f}"
        ws = f"U {want[0]}, p {want[1]:.5f}" if isinstance(want, tuple) else f"p {want:.5f}"
        print(f"         {name:<36} {gs:<22} (hand {ws})  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 62 Part 2's tasks, and the perfect gate at P=8 at every layer (conv on, random weights):")
    good = True
    for key, (P, S) in (("P8S1", (8, 1)), ("P8S2", (8, 2))):
        ref = tcb.task_for(P, S)
        t = TASKS[key]
        g1 = torch.Generator().manual_seed(62)
        g2 = torch.Generator().manual_seed(62)
        same = vars(t) == vars(ref) and all(torch.equal(u, v) for u, v in
                                            zip(t.make_batch(8, g1), ref.make_batch(8, g2)))
        good &= same
        print(f"         {key}: equals test_channel_binding.task_for({P}, {S}) (attributes and batches): {same}   "
              f"vocab {t.vocab}, length {t.L}")
    t = TASKS["P8S2"]
    x = t.make_batch(16, torch.Generator().manual_seed(620))[0][:, :-1]
    m = builder_for(ARM["ceiling_P8_conv4"])(ARM["ceiling_P8_conv4"], 6)().eval()
    with torch.no_grad():
        m.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(621)))
    m.attn.record = []
    with torch.no_grad():
        m(x, TAU_END)
    lab = t.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s_[:, 0][cross].abs().max().item() for s_ in m.attn.record]
    live = [s_[:, 0][~cross].abs().max().item() for s_ in m.attn.record]
    m.attn.record = None
    g = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0
    good &= g
    for i, (c, l) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l:.2e}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 63 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("A_conv4", "B_wide_conv4"):
        sp = spec(key, 1, EVAL_EVERY)
        rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        print(f"         {key:<12} seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   "
              f"acc {rh['acc']:.4f}   conv mean |w| per lag {[round(v, 3) for v in rh['end']['conv_absmean']]}"
              f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}"
          + ("" if pair["ok"] else "   (CHECK 59, the pairing, did not pass: see VALIDITY)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return pair


# ── Reporting ────────────────────────────────────────────────────────────────
def bound_r(r):
    return bool(r and r.get("ok") and r.get("transition") is not None)


def bound(store, key, s):
    return bound_r(get(store, key, s))


def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def ok_runs(store, key):
    return [(s, r) for s in SEEDS[key] if (r := get(store, key, s)) and r.get("ok")]


def raw_tables(store):
    print(f"  {'arm':<16} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'margin':>7}  "
          f"eta^2 key pos s/k/h/i   conv mean|w| lag 0/1/2/3   flags")
    for k in GATED:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<16} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<16} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS)
            fl = ((["BOUND"] + (["DISCOVERED"] if r["discovered"] else [])
                   + ["routed" if routed(r) else "NOT routed"]) if r["transition"] is not None
                  else [fail_class(r)]) + (["collapsed"] if r["collapsed"] else [])
            print(f"  {k:<16} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {e['val_cos']:>8.4f} "
                  f"{e['margin']:>7.4f}  {etas:<21}   {wstr(e):<25}   {' '.join(fl)}")
        print()
    print(f"  {'arm':<16} {'seed':>4} {'acc':>7} {'transition':>10} {'stopped':>7}   conv mean|w| lag "
          f"0/1/2/3   flags")
    for k in KEYS:
        if k in GATED:
            continue
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<16} {s:>4}  NOT RUN")
            elif not r.get("ok"):
                print(f"  {k:<16} {s:>4}  FAILED — {r['error']}")
            else:
                fl = (["BOUND"] if r["transition"] else []) + (["collapsed"] if r["collapsed"] else [])
                print(f"  {k:<16} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{r['stopped_at']:>7}   {wstr(r['end']):<25}   {' '.join(fl)}")
        print()


def report(store, short, pair, wall, path, also):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    raw_tables(store)

    print("=" * 100)
    print("COUNTS (BOUND = transition not None; DISCOVERED = bound AND final VAL cos < 0.5, gated arms)")
    print("=" * 100)
    c, d = {}, {}
    for a in ARMS:
        k, ns = a["key"], len(SEEDS[a["key"]])
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        d[k] = sum(disc(store, k, s) for s in SEEDS[k])
        rs = [get(store, k, s) for s in SEEDS[k]]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        extra = ""
        if k in RECORDED and short is not None:
            rk = RECORDED[k]
            c1 = sum(bound_r(get(short, rk, s)) for s in SEEDS[k])
            extra = f"   (recorded {rk} at lr 1e-3, same seeds: bound {c1}/{ns})"
        print(f"  {a['label']:<54} bound {c[k]:>2}/{ns}"
              + (f"   discovered {d[k]:>2}/{ns}" if k in GATED else " " * 18)
              + f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{ns}"
              + (f"   [{failed} FAILED]" if failed else "") + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["ceiling_conv4"])
    valid = c["ceiling_conv4"] == nc and pair["ok"]
    print(f"  ceiling_conv4 binds {c['ceiling_conv4']}/{nc}   the pairing CHECK (59) passed: {pair['ok']} "
          f"(file {pair['path']} exists {pair['exists']}, CPU match {pair.get('cpu_match')}, "
          f"reproduced {pair.get('reproduced')})")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND)")
    print("#" * 100)
    v = dict(bound=c, discovered=d, valid=valid, pairing=pair["ok"])
    verdict = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    for name, k, title in (("LR1", "B_conv4", "ONE CHANNEL BINDS MORE AT 4e-3"),
                           ("LR2", "A_conv4", "GATE + CONV BINDS MORE AT 4e-3")):
        rk = RECORDED[k]
        seeds = SEEDS[k]
        if short is None:
            p, b_, c_ = 1.0, 0, 0
        else:
            b_ = sum(bound(store, k, s) and not bound_r(get(short, rk, s)) for s in seeds)
            c_ = sum(bound_r(get(short, rk, s)) and not bound(store, k, s) for s in seeds)
            p = mcnemar_greater(b_, c_)
        c1 = sum(bound_r(get(short, rk, s)) for s in seeds) if short else float("nan")
        print(f"  {name} {title} (exact McNemar, one-sided): {k} {c[k]}/{len(seeds)} at 4e-3 vs recorded "
              f"{rk} {c1}/{len(seeds)} at 1e-3; 4e-3 only {b_}, 1e-3 only {c_}; p = {p:.3g}")
        print(f"     *** {name}: {verdict(p)} ***")
        v[name], v[f"{name}_p"] = verdict(p), p
    n40, n20 = len(SEEDS["A_conv4"]), len(SEEDS["B_wide_conv4"])
    p3 = fisher_greater(c["A_conv4"], n40, c["B_conv4"], n40)
    s3 = SEEDS["A_conv4"]
    b3 = sum(bound(store, "A_conv4", s) and not bound(store, "B_conv4", s) for s in s3)
    c3 = sum(bound(store, "B_conv4", s) and not bound(store, "A_conv4", s) for s in s3)
    print(f"  LR3 THE GATE STILL ADDS AT 4e-3 (Fisher one-sided): A_conv4 {c['A_conv4']}/{n40} vs B_conv4 "
          f"{c['B_conv4']}/{n40}; p = {p3:.3g}")
    print(f"     exact McNemar two-sided on the pairs: A_conv4 only {b3}, B_conv4 only {c3}; p = "
          f"{mcnemar_exact(b3, c3):.3g}")
    print(f"     *** LR3: {verdict(p3)} ***")
    p4 = fisher_greater(c["A_conv4"], n40, c["B_wide_conv4"], n20)
    s4 = SEEDS["B_wide_conv4"]
    b4 = sum(bound(store, "A_conv4", s) and not bound(store, "B_wide_conv4", s) for s in s4)
    c4 = sum(bound(store, "B_wide_conv4", s) and not bound(store, "A_conv4", s) for s in s4)
    print(f"  LR4 TWO CHANNELS BEAT ONE CHANNEL WITH THE SAME MEMORY (Fisher one-sided): A_conv4 "
          f"{c['A_conv4']}/{n40} vs B_wide_conv4 {c['B_wide_conv4']}/{n20}; p = {p4:.3g}")
    print(f"     exact McNemar two-sided on seeds {s4[0]}-{s4[-1]}: A_conv4 only {b4}, B_wide_conv4 only "
          f"{c4}; p = {mcnemar_exact(b4, c4):.3g}")
    print(f"     *** LR4: {verdict(p4)} ***")
    pool_runs = ok_runs(store, "B_conv4") + ok_runs(store, "B_wide_conv4")
    xb = [r["end"]["conv_absmean"][2] for _, r in pool_runs if r["transition"] is not None]
    xn = [r["end"]["conv_absmean"][2] for _, r in pool_runs if r["transition"] is None]
    if len(xb) < M1_MIN or len(xn) < M1_MIN:
        m1, pm1, U = ("UNTESTABLE" if valid else "UNTESTED"), float("nan"), float("nan")
    else:
        U, pm1 = mann_whitney_greater(xb, xn)
        m1 = verdict(pm1)
    print(f"  M1 LAG 2 MARKS SINGLE-CHANNEL BINDING (exact Mann-Whitney, one-sided): B_conv4 + B_wide_conv4, "
          f"bound {len(xb)} (lag-2 median {med(xb)}) vs not bound {len(xn)} (median {med(xn)}); U = {U}, "
          f"p = {pm1:.3g}")
    print(f"     *** M1: {m1} ***")
    v.update(LR3=verdict(p3), LR3_p=p3, LR4=verdict(p4), LR4_p=p4, M1=m1, M1_p=pm1)
    print(f"  BANDS (RELIABLE >= 36/40, MAJORITY 20-35, MINORITY 1-19, NEVER 0), on BOUND:")
    for k in ("B_conv4", "A_conv4"):
        print(f"     *** {k:<8} {c[k]}/{n40}: {band(c[k])} ***")
        v[f"band_{k}"] = band(c[k])
    print()
    print("  READINGS:")
    rd = []
    if not valid:
        print("     (UNTESTED: no reading)")
    else:
        if v["LR3"] == "NOT SHOWN":
            rd.append(("LR3 NOT SHOWN", READINGS["LR3-"]))
        elif v["LR4"] == "SHOWN":
            rd.append(("LR3 SHOWN and LR4 SHOWN", READINGS["LR3+ LR4+"]))
        else:
            rd.append(("LR3 SHOWN and LR4 NOT SHOWN", READINGS["LR3+ LR4-"]))
        if band(c["A_conv4"]) == "RELIABLE":
            rd.append(("A_conv4 RELIABLE", READINGS["RELIABLE"]))
        for cond, text in rd:
            print(f"     {cond}: \"{text}\"")
    v["readings"] = [cond for cond, _ in rd]
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  per-seed outcome at lr 1e-3 (recorded) and 4e-3 (B = bound, D = bound and discovered, "
          ". = neither, - = not run):")
    cols = [("B_conv", "B_conv4"), ("A_conv", "A_conv4"), ("A", "A4")]

    def mk(store_, k, s, seeds):
        if store_ is None or s not in seeds:
            return "-"
        r = get(store_, k, s)
        if not (r and r.get("ok")):
            return "?"
        return "D" if r.get("discovered") else "B" if r["transition"] is not None else "."
    print("    seed  " + "  ".join(f"{k1 + '@1e-3':>12} {k4 + '@4e-3':>13}" for k1, k4 in cols)
          + f"  {'B_wide_conv4':>12}")
    for s in SEEDS["A_conv4"]:
        print(f"    {s:>4}  " + "  ".join(f"{mk(short, k1, s, tsc.SEEDS[k1]):>12} {mk(store, k4, s, SEEDS[k4]):>13}"
                                        for k1, k4 in cols)
              + f"  {mk(store, 'B_wide_conv4', s, SEEDS['B_wide_conv4']):>12}")
    print("  exact McNemar two-sided (bound), every pair of arms within 4e-3, on their shared seeds:")
    for k1, k2 in itertools.combinations(("B_conv4", "A_conv4", "B_wide_conv4", "A4", "B4"), 2):
        sh = sorted(set(SEEDS[k1]) & set(SEEDS[k2]))
        b_ = sum(bound(store, k1, s) and not bound(store, k2, s) for s in sh)
        c_ = sum(bound(store, k2, s) and not bound(store, k1, s) for s in sh)
        print(f"    {k1:<12} vs {k2:<12} ({len(sh):>2} seeds): {k1} only {b_}, {k2} only {c_}; "
              f"p = {mcnemar_exact(b_, c_):.3g}")
    print()
    print("  gated runs at the end: bound WITH routing (margin >= 0.9) or WITHOUT; failures by "
          "test_router_layout's classes:")
    for k in ("A_conv4", "A4"):
        rs = ok_runs(store, k)
        w_ = [s for s, r in rs if r["transition"] is not None and routed(r)]
        wo = [s for s, r in rs if r["transition"] is not None and not routed(r)]
        classes = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        for s, r in rs:
            if r["transition"] is None:
                classes[fail_class(r)].append(s)
        print(f"    {k:<8} bound with routing {len(w_)}, WITHOUT routing {len(wo)}"
              + (f" ({' '.join(f's{s}' for s in wo)})" if wo else "")
              + "   failures: " + "   ".join(f"{m_} {len(x)}" for m_, x in classes.items()))
        for m_, xs in classes.items():
            if xs:
                print(f"    {'':<8}   {m_:<14} " + " ".join(f"s{s}" for s in xs))
    print()
    print("  the convolution's mean |w| per lag at the end, median [min, max] (lag 2 = where a value "
          "position sees its context token):")
    for k in CONV_KEYS:
        rs = ok_runs(store, k)
        groups = (("bound", [r for _, r in rs if r["transition"] is not None]),
                  ("not bound", [r for _, r in rs if r["transition"] is None]))
        if k in GATED:
            b_all = [r for _, r in rs if r["transition"] is not None]
            groups = (("bound, routed", [r for r in b_all if routed(r)]),
                      ("bound, NOT routed", [r for r in b_all if not routed(r)]),
                      ("not bound", [r for _, r in rs if r["transition"] is None]))
        for lab, sub in groups:
            if sub:
                print(f"    {k:<16} {lab:<17} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  transition steps and plateau length (steps at held-out accuracy 0.45-0.55 before the "
          "transition), per bound run:")
    for k in KEYS:
        rs = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None]
        if not rs:
            print(f"    {k:<16} none bound")
            continue
        trs = [r["transition"] for _, r in rs]
        pls = [plateau(r) for _, r in rs]
        print(f"    {k:<16} transition median {statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   "
              f"plateau median {statistics.median(pls):.0f} [{min(pls)}, {max(pls)}]")
        print(f"    {'':<16} " + "  ".join(f"s{s}:{r['transition']}/{plateau(r)}" for s, r in rs))
    print()
    print(f"  the restart check per arm (held-out accuracy >= {A_CHECK} at step {T_CHECK}):")
    for k in KEYS:
        rs = ok_runs(store, k)
        passed = [(s, r) for s, r in rs if (a_ := curve_acc(r, T_CHECK)) is not None and a_ >= A_CHECK]
        pb = sum(1 for _, r in passed if r["transition"] is not None)
        thrown = [s for s, r in rs if r["transition"] is not None and (curve_acc(r, T_CHECK) or 0.0) < A_CHECK]
        print(f"    {k:<16} pass {len(passed)}/{len(rs)}; of those bound {pb}/{len(passed)}; binders the "
              f"check would throw away: {len(thrown)}" + (f" ({' '.join(f's{s}' for s in thrown)})" if thrown else ""))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median "
          "[min, max] over seeds:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>3} {k:<16} " + "   ".join(parts))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
    print()

    print("=" * 100)
    print("PART 2 — descriptive (sets up P-scaling)")
    print("=" * 100)
    for k in PART2:
        rs = ok_runs(store, k)
        print(f"  {ARM[k]['label']}: bound {sum(1 for _, r in rs if r['transition'] is not None)}/{len(SEEDS[k])}")
        for s, r in rs:
            print(f"    s{s}  {'BOUND at ' + str(r['transition']) if r['transition'] else 'not bound':<16} "
                  f"final accuracy {r['acc']:.4f}   stopped at {r['stopped_at']}")
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
                b2 = sum(1 for r in runs if r.get("ok") and r.get("transition") is not None)
                d2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                ns = len(SEEDS[k])
                print(f"    {k:<16} bound here {c[k]}/{ns}   other {b2}/{len(runs)}   pooled "
                      f"{c[k] + b2}/{ns + len(runs)}"
                      + (f"      discovered here {d[k]}/{ns}   other {d2}/{len(runs)}" if k in GATED else ""))
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — accuracy x100 at each evaluation every {EVAL_EVERY} steps, plus VAL cos x100 for "
          f"gated arms")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            tag = f"{k:<16} s{s}"
            if k in GATED:
                curve_lines(tag, r)
            else:
                accs = "".join(f" {round(acc * 100):>3}" for _, acc, _ in r["curve"])
                print(f"  {tag} acc {accs}   -> "
                      f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Does lr 4e-3 change what test_short_conv found?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's conv_lr_results.json, for pooled counts")
    ap.add_argument("--short", default=SHORT_FILE,
                    help="this machine's short_conv_results.json (the lr 1e-3 runs to pair with)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("Conv learning rate: does lr 4e-3 change what test_short_conv found at 1e-3?")
    print(f"  test_short_conv's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, conv width "
          f"{CONV_K}), lr {LR4} (test_short_conv: {SUB_LR}), MAX_ITERS={MAX_ITERS}")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<54} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  paired with {args.short}: seeds 160-199 reused on purpose (same initial parameters and "
          f"batches, only the lr differs)")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        pair = verify(pool, args.short)
        short = load_store(args.short) if pair["exists"] else None
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, lr=LR4, short=args.short,
                             pairing={k: v for k, v in pair.items() if not k.endswith("_curve")},
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on each arm's own task, "
              f"evaluation charged once per {EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_job, KEYS)))
        for k in KEYS:
            print(f"  {k:<16} {costs[k] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds")
        allspecs = [spec(k, s, MAX_ITERS) for k in KEYS for s in SEEDS[k]]
        durs = [costs[sp["arm"]] * sp["iters"] for sp in allspecs]
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
        todo.sort(key=lambda sp: -costs[sp["arm"]])
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
                extra = (f"VALcos={e['val_cos']:.3f} margin={e['margin']:.3f} " if "margin" in e else "")
                print(f"   {sp['arm']:<16} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}  {extra}conv|w|={wstr(e)}"
                      f"{'  DISCOVERED' if rec['discovered'] else ''}  {rec['secs']:.0f}s  "
                      f"(elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<16} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, short, pair, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
