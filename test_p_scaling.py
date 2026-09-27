#!/usr/bin/env python
"""
test_p_scaling.py — does two channels' advantage over one grow when each sequence holds more
bindings? P=8 instead of P=4, at lr 4e-3 with the convolution, the recipe under which P=8
trains. Everything below is fixed before any result.

Run it directly:

    python test_p_scaling.py --p4 results/X/conv_lr_results.json
    python test_p_scaling.py --p4 ... --also results/L/p_scaling_results.json

BACKGROUND
test_conv_lr, both machines (seeds 160-199, lr 4e-3, paired with test_short_conv's 1e-3 runs);
L ran merge ce62073, whose code is identical to 3c69afd.
- LR1 SHOWN on both: one channel + conv bound 18/40 (X; 2/40 at 1e-3) and 16/40 (L; 5/40).
- LR2 NOT SHOWN on both: gate + conv 22/40 (X; 23 at 1e-3) and 28/40 (L; 28).
- LR3 split: X NOT SHOWN (22 vs 18, p = 0.25), L SHOWN (28 vs 16, p = 0.006). Pooled, post hoc:
  50/80 vs 34/80, McNemar 29 vs 13, p = 0.02.
- LR4 NOT SHOWN on both (A_conv4 vs B_wide_conv4, which bound 10/20 on each). On the shared
  seeds 160-179, doubling one channel's memory did not help it: B_wide_conv4 vs B_conv4 was
  12 vs 11 discordant, pooled.
- M1 SHOWN on both (p = 2e-12 and 4e-15): single-channel binders end with a larger lag-2 weight.
- Routing: at 4e-3 the gated binders mostly do NOT route by stream. X had 6 routed of 22
  (16 of 23 at 1e-3), L 12 of 28 (23 of 28).
- Without the conv, 4e-3 fails: A4 and B4 bound 0/10 on each machine.
- P=8 at 4e-3 with the conv is trainable. One stream (S=1) bound 5/5 on both machines, and the
  perfect gate at S=2 bound 5/5 on both (transitions 1200-15600).
- Seed-level outcomes differ between machines; the verdict is per machine.

DESIGN (lr 4e-3, S=2, P=8, grouped layout, conv 'layer' width 4, MAX_ITERS 24000)
- The arms are test_conv_lr's, on test_channel_binding.task_for(8, 2), built and run by
  test_conv_lr's own path (run_one with its lr and builder knobs; no code changes).
- Seeds reuse 160-199 on purpose: each P=8 run is paired by seed with this machine's recorded
  P=4 run in conv_lr_results.json. --p4 points to that file (default conv_lr_results.json; on
  X, results/X/conv_lr_results.json). Its meta CPU must match this machine, two of its runs
  must reproduce exactly, and it must hold every A_conv4 and B_conv4 run for seeds 160-189
  (CHECK 64). Otherwise the pairing fails and every claim is UNTESTED (see VALIDITY).

ARMS
  B_conv8        single channel + conv                          seeds 160-189 (30)
  A_conv8        arm A + conv                                   seeds 160-189 (30)
  B_wide_conv8   single channel, N=512, + conv (A_conv8's fast-
                 weight state, CHECK 66)                        seeds 160-175 (16)
  ceiling_conv8  perfect gate + conv (validity)                 seeds 160-167 (8)

VALIDITY AND PRE-REGISTERED CLAIMS (outcome BOUND = transition not None; DISCOVERED printed for
gated arms)
  VALID: ceiling_conv8 binds >= 7/8, AND the pairing CHECK (64) passes. Otherwise every claim is
    UNTESTED.
  P1 THE GATE ADDS AT P=8: A_conv8 vs B_conv8. Fisher one-sided (A higher); McNemar printed
     alongside.
  P2 THE GATE'S ADVANTAGE GROWS WITH P: for each seed s in 160-189,
     D_P(s) = bound(A_conv at P) - bound(B_conv at P), with P=4 taken from the recorded A_conv4
     and B_conv4. Exact sign test, one-sided, on D_8(s) - D_4(s) over the seeds where it is
     nonzero; the alternative is an increase. (Under no change D_4 and D_8 are independent and
     identically distributed, so the differences are symmetric about 0.)
  P3 TWO CHANNELS BEAT ONE CHANNEL WITH THE SAME MEMORY AT P=8: A_conv8 vs B_wide_conv8. Fisher
     one-sided (A higher); McNemar on seeds 160-175 printed alongside.
  M1-P8 LAG 2 STILL MARKS SINGLE-CHANNEL BINDING AT P=8: test_conv_lr's M1 (exact Mann-Whitney,
     one-sided: bound runs have a larger end lag-2 mean |w|) on B_conv8 + B_wide_conv8.
     UNTESTABLE if either group has fewer than 3 runs.
  Each claim is SHOWN if p < 0.05, else NOT SHOWN.
  Readings, each printed verbatim when it applies:
  - P1 SHOWN and P2 SHOWN: "two channels' advantage over one grows with load at this scale:
    evidence for capacity, the case for scaling them."
  - P1 SHOWN and P2 NOT SHOWN: "the gate helps at P=8, but not detectably more than at P=4."
  - P1 NOT SHOWN: "at P=8 one channel with the convolution keeps up with the gated model: no
    capacity advantage from two channels at this load."
  - P3 SHOWN: "at P=8 two channels beat one channel with the same total memory: the partition
    helps, not the memory size."
Fisher and McNemar (two-sided) are test_router_confirm's; the one-sided McNemar and the
Mann-Whitney test are test_conv_lr's. The sign test is the one-sided McNemar applied to the
signs of the nonzero differences (CHECK 67).

DIAGNOSTICS, not part of the verdict
- Per seed: bound at P=4 (recorded) and at P=8, for every arm.
- Final accuracy per arm, as median [min, max] and the count >= 0.9. At P=8 runs may plateau
  below binding, and the plateau level matters.
- test_conv_lr's diagnostics for the new arms: routed vs unrouted binders and failure classes
  (test_router_layout's classes, computed at P=8); the convolution's mean |w| per lag by
  outcome; transitions and plateaus (0.45-0.55, as before); the restart check; gradient norms
  at steps 1, 10 and 100; collapsed runs.
- The P=4 vs P=8 transition-step distributions per arm, on the same seeds.
- --also pooled counts.

CHECKS (printed before any training): test_conv_lr's verification (which runs every earlier
test's; its CHECK 59 pairs with the short_conv file named in the --p4 file's meta), then
  64 the pairing: the --p4 file exists, its meta CPU equals this machine's, it holds every
     recorded A_conv4 and B_conv4 run for seeds 160-189, and re-running B_conv4 seed 162 and
     A_conv4 seed 160 at P=4 for 1200 steps reproduces their recorded curves exactly (recorded,
     not asserted: a failure makes the claims UNTESTED);
  65 every arm runs task_for(8, 2) at lr 4e-3 (a 20-step run's record, evaluations every 10,
     equals the same run on a fresh task_for(8, 2); optimizer lr captured by a step
     pre-hook); the perfect gate at P=8 applies at every layer;
  66 B_wide_conv8's fast-weight state equals A_conv8's; both parameter counts printed;
  67 the sign-test code reproduces hand values, including no nonzero differences (p = 1);
  68 a worker's run is bit-identical to the same run in the main process (A_conv8 and
     B_wide_conv8, 1200 steps).

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to p_scaling_results.json (gitignored; copied into
results/X/ after the run).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import itertools
import json
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
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater, mcnemar_exact
from test_router_curriculum import (
    ARCH, get, count, load_store, init_worker, makespan, cpu_model, save_results, COLLAPSE,
    TAU_END, TIME_STEPS,
)
from test_router_discovery import model_fn
from test_router_reliability import run_one, strip_all, curve_acc, med, curve_lines, T_CHECK, A_CHECK
from test_router_layout import fail_class, ETA_GROUPS
from test_readout_path import GRAD_STEPS
from test_short_conv import conv_stats, conv_grad_norms, plateau, routed, wstr, CONV_K
import test_conv_lr as tcl
from test_conv_lr import LR4, ALPHA, M1_MIN, mcnemar_greater, mann_whitney_greater, bound_r

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
CEILING_MIN = 7                            # VALID needs ceiling_conv8 >= 7/8
ACC_HIGH = 0.9                             # diagnostic: final accuracy >= 0.9
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "p_scaling_results.json"
P4_FILE = "conv_lr_results.json"

TASK8 = tcl.TASKS["P8S2"]                  # test_channel_binding.task_for(8, 2) (CHECK 65)
ARMS = [
    dict(tcl.ARM["B_conv4"], key="B_conv8", task="P8S2",
         label="B_conv8        single channel + conv"),
    dict(tcl.ARM["A_conv4"], key="A_conv8", task="P8S2",
         label="A_conv8        arm A + conv"),
    dict(tcl.ARM["B_wide_conv4"], key="B_wide_conv8", task="P8S2",
         label="B_wide_conv8   single channel N=512 + conv"),
    dict(tcl.ARM["ceiling_P8_conv4"], key="ceiling_conv8",
         label="ceiling_conv8  perfect gate + conv"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
GATED = tuple(k for k in KEYS if ARM[k]["gate"] != "none")
SEEDS = dict(B_conv8=tuple(range(160, 190)), A_conv8=tuple(range(160, 190)),
             B_wide_conv8=tuple(range(160, 176)), ceiling_conv8=tuple(range(160, 168)))
P4 = dict(B_conv8="B_conv4", A_conv8="A_conv4", B_wide_conv8="B_wide_conv4",
          ceiling_conv8="ceiling_conv4")                              # P=8 arm -> recorded P=4 arm
P2_SEEDS = tuple(range(160, 190))
READINGS = {
    "P1+ P2+": "two channels' advantage over one grows with load at this scale: evidence for "
               "capacity, the case for scaling them.",
    "P1+ P2-": "the gate helps at P=8, but not detectably more than at P=4.",
    "P1-": "at P=8 one channel with the convolution keeps up with the gated model: no capacity "
           "advantage from two channels at this load.",
    "P3+": "at P=8 two channels beat one channel with the same total memory: the partition "
           "helps, not the memory size.",
}


# ── The sign test ────────────────────────────────────────────────────────────
def sign_test_greater(diffs):
    """One-sided exact sign test (an increase) over the nonzero differences: the number
    positive against Bin(n, 1/2), n = nonzero count; test_conv_lr's one-sided McNemar is the
    same computation. Returns (positive, negative, p); no nonzero differences gives p = 1."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return pos, neg, mcnemar_greater(pos, neg)


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_job(sp):
    a = ARM[sp["arm"]]
    return run_one(a, sp["seed"], sp["iters"], task=TASK8, stats_fn=conv_stats,
                   grad_fn=conv_grad_norms, lr=a["lr"], builder=tcl.builder_for(a))


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def time_job(key):
    """test_conv_lr.time_job for this test's arms (all on task_for(8, 2))."""
    a = ARM[key]
    mk = tcl.builder_for(a)(a, 0)
    per = time_per_step(TASK8, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(TASK8)
    t0 = time.time()
    evaluate(m, TASK8, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


# ── The pairing (CHECK 64) ───────────────────────────────────────────────────
def p4_pairing(pool, path):
    """The --p4 file exists, was written on this CPU, holds every recorded A_conv4 and
    B_conv4 run for seeds 160-189, and two of its runs reproduce."""
    info = dict(path=path, exists=os.path.exists(path))
    if not info["exists"]:
        info.update(cpu_match=False, complete=False, reproduced=False, ok=False)
        return info
    p4 = load_store(path)
    info["cpu"] = p4.get("meta", {}).get("cpu")
    info["git"] = p4.get("meta", {}).get("git")
    info["cpu_match"] = info["cpu"] == cpu_model()
    have = [bool((r := get(p4, k, s)) and r.get("ok")) for k in ("A_conv4", "B_conv4") for s in P2_SEEDS]
    info["n_recorded"] = sum(have)
    info["complete"] = all(have)
    rep = {}
    for key, seed in (("B_conv4", 162), ("A_conv4", 160)):
        rec = get(p4, key, seed)
        r = pool.submit(tcl.run_job, tcl.spec(key, seed, EVAL_EVERY)).result()
        now = json.loads(json.dumps(r["curve"]))          # the stored form: lists, not tuples
        old = (rec or {}).get("curve", [])[:len(now)]
        rep[key] = bool(rec and rec.get("ok") and r["ok"] and now and now == old)
        info[f"{key}_curve"] = (seed, now, old)
    info["reproduced"] = all(rep.values())
    info["ok"] = info["cpu_match"] and info["complete"] and info["reproduced"]
    return info


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool, p4_path):
    short_path = tcl.SHORT_FILE
    if os.path.exists(p4_path):
        short_path = load_store(p4_path).get("meta", {}).get("short") or short_path
    print(f"test_conv_lr.py's verification (which runs test_short_conv's, and so on down to")
    print(f"test_multilayer_binding's); its CHECK 59 pairs with {short_path}, the short_conv file")
    print(f"named in the --p4 file's meta:")
    tcl.verify(pool, short_path)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 64 the pairing with this machine's test_conv_lr results ({p4_path}):")
    pair = p4_pairing(pool, p4_path)
    if not pair["exists"]:
        print(f"         file exists: False  -> PAIRING FAILS (every claim will be UNTESTED)")
    else:
        print(f"         file exists: True   written on: {pair['cpu']} (git {pair['git']})   this machine: "
              f"{cpu_model()}   match: {pair['cpu_match']}")
        print(f"         recorded A_conv4 and B_conv4 runs for seeds {P2_SEEDS[0]}-{P2_SEEDS[-1]}: "
              f"{pair['n_recorded']}/{2 * len(P2_SEEDS)}")
        for key in ("B_conv4", "A_conv4"):
            seed, now, rec = pair[f"{key}_curve"]
            print(f"         {key} seed {seed} re-run at P=4, {EVAL_EVERY} steps: {now} vs recorded {rec}")
        print(f"         reproduced exactly: {pair['reproduced']}  -> "
              f"{'PAIRED' if pair['ok'] else 'PAIRING FAILS (every claim will be UNTESTED)'}")
    print()

    print(f"CHECK 65 every arm runs task_for(8, 2) at lr {LR4}:")
    ref = tcb.task_for(8, 2)
    g1, g2 = torch.Generator().manual_seed(65), torch.Generator().manual_seed(65)
    same_task = vars(TASK8) == vars(ref) and all(torch.equal(u, v) for u, v in
                                                 zip(TASK8.make_batch(8, g1), ref.make_batch(8, g2)))
    good = same_task
    print(f"         the task equals a fresh test_channel_binding.task_for(8, 2) (attributes and batches): "
          f"{same_task}   P {TASK8.P}, S {TASK8.S}, vocab {TASK8.vocab}, length {TASK8.L}")
    seen, widths = [], set()
    h1 = register_optimizer_step_pre_hook(
        lambda opt, args, kw: seen.append(sorted({g_["lr"] for g_ in opt.param_groups})))
    h2 = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda mod, inp: widths.add(inp[0].shape[1])
        if type(mod).__name__ == "MultiBDH" and inp and torch.is_tensor(inp[0]) else None)
    try:
        for key in KEYS:
            a = ARM[key]
            seen.clear()
            widths.clear()
            keep = {"at": -1}
            r = run_one(a, 1, 20, eval_every=10, task=TASK8, stats_fn=conv_stats,
                        grad_fn=conv_grad_norms, lr=a["lr"], builder=tcl.builder_for(a), keep=keep)
            lrs = sorted({v for s in seen for v in s})
            w_now = sorted(widths)
            widths.clear()
            r_ref = run_one(a, 1, 20, eval_every=10, task=ref, stats_fn=conv_stats,
                            grad_fn=conv_grad_norms, lr=LR4, builder=lambda aa, s: model_fn(ref, aa, s))
            vocab = keep["model"].embed.weight.shape[0]
            g = (lrs == [LR4] and strip_all(r) == strip_all(r_ref) and r["ok"] and vocab == ref.vocab
                 and w_now == sorted(widths) and max(w_now) > tcl.TASKS["P4S2"].L)
            good &= g
            print(f"         {key:<14} optimizer lr {lrs}   20-step record == fresh task_for(8, 2)'s: "
                  f"{strip_all(r) == strip_all(r_ref)} (acc {r['acc']:.4f})   input widths {w_now}   "
                  f"model vocab {vocab}  "
                  f"-> {'OK' if g else 'WRONG'}")
    finally:
        h1.remove()
        h2.remove()
    x = TASK8.make_batch(16, torch.Generator().manual_seed(650))[0][:, :-1]
    a = ARM["ceiling_conv8"]
    m = tcl.builder_for(a)(a, 6)().eval()
    with torch.no_grad():
        m.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(651)))
    m.attn.record = []
    with torch.no_grad():
        m(x, TAU_END)
    lab = TASK8.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s_[:, 0][cross].abs().max().item() for s_ in m.attn.record]
    live = [s_[:, 0][~cross].abs().max().item() for s_ in m.attn.record]
    m.attn.record = None
    g = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0
    good &= g
    print(f"         ceiling_conv8, random conv weights: layers seen {len(maxc)} (n_layer={ARCH['n_layer']})")
    for i, (c, l) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l:.2e}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 66 B_wide_conv8's model:")
    mw = tcl.builder_for(ARM["B_wide_conv8"])(ARM["B_wide_conv8"], 0)()
    ma = tcl.builder_for(ARM["A_conv8"])(ARM["A_conv8"], 0)()
    good = (mw.n_ch == 1 and mw.n_feat == 512 and mw.conv == "layer" and mw.gate_kind == "none"
            and n_state(mw) == n_state(ma))
    for name, mm in (("B_wide_conv8", mw), ("A_conv8", ma)):
        print(f"         {name + ':':<13} k={mm.n_ch}, N={mm.n_feat}, conv {mm.conv!r}, gate {mm.gate_kind!r}, "
              f"fast-weight state {n_state(mm)}, parameters {n_params(mm)}")
    print(f"         -> {'STATE MATCHED' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 67 the sign test reproduces hand values:")
    cases = [
        ("[1, 1, 1, 1, 1]", [1, 1, 1, 1, 1], (5, 0, 1 / 32)),
        ("[1, 1, 1, -1]", [1, 1, 1, -1], (3, 1, 5 / 16)),
        ("[2, -1, 1, 0]", [2, -1, 1, 0], (2, 1, 1 / 2)),
        ("[1, 2, 1, 1, 1, 1, 0, -1]", [1, 2, 1, 1, 1, 1, 0, -1], (6, 1, 1 / 16)),
        ("[-1, -2]", [-1, -2], (0, 2, 1.0)),
        ("[0, 0, 0] (no nonzero)", [0, 0, 0], (0, 0, 1.0)),
        ("[] (no seeds)", [], (0, 0, 1.0)),
    ]
    good = True
    for name, diffs, want in cases:
        got = sign_test_greater(diffs)
        g = got[:2] == want[:2] and abs(got[2] - want[2]) < 1e-12
        good &= g
        print(f"         {name:<26} + {got[0]}  - {got[1]}  p {got[2]:.5f}   (hand + {want[0]}  - {want[1]}  "
              f"p {want[2]:.5f})  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 68 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("A_conv8", "B_wide_conv8"):
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
          + ("" if pair["ok"] else "   (CHECK 64, the pairing, did not pass: see VALIDITY)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return pair


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds)
            if (r := get(store, key, s)) and r.get("ok")]


def mark(store, key, s):
    if store is None:
        return "-"
    r = get(store, key, s)
    if r is None:
        return "-"
    if not r.get("ok"):
        return "?"
    return "D" if r.get("discovered") else "B" if r["transition"] is not None else "."


def d_of(store, a_key, b_key, s):
    return int(bound(store, a_key, s)) - int(bound(store, b_key, s))


def raw_tables(store, p4):
    print(f"  {'arm':<14} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'margin':>7}  "
          f"eta^2 key pos s/k/h/i   conv mean|w| lag 0/1/2/3   flags")
    for k in GATED:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<14} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<14} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS)
            fl = ((["BOUND"] + (["DISCOVERED"] if r["discovered"] else [])
                   + ["routed" if routed(r) else "NOT routed"]) if r["transition"] is not None
                  else [fail_class(r)]) + (["collapsed"] if r["collapsed"] else [])
            print(f"  {k:<14} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {e['val_cos']:>8.4f} "
                  f"{e['margin']:>7.4f}  {etas:<21}   {wstr(e):<25}   {' '.join(fl)}")
        print()
    print(f"  {'arm':<14} {'seed':>4} {'acc':>7} {'transition':>10} {'stopped':>7}   conv mean|w| lag "
          f"0/1/2/3   flags")
    for k in KEYS:
        if k in GATED:
            continue
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<14} {s:>4}  NOT RUN")
            elif not r.get("ok"):
                print(f"  {k:<14} {s:>4}  FAILED — {r['error']}")
            else:
                fl = (["BOUND"] if r["transition"] else []) + (["collapsed"] if r["collapsed"] else [])
                print(f"  {k:<14} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{r['stopped_at']:>7}   {wstr(r['end']):<25}   {' '.join(fl)}")
        print()
    print("  P2 per seed (1 = bound, 0 = not): D_P = A_conv - B_conv at P; P=4 recorded (--p4)")
    print(f"    {'seed':>4}  {'A_conv4':>7} {'B_conv4':>7} {'D_4':>4}   {'A_conv8':>7} {'B_conv8':>7} {'D_8':>4}   "
          f"{'D_8 - D_4':>9}")
    for s in P2_SEEDS:
        if p4 is None:
            a4 = b4 = d4 = "-"
        else:
            a4, b4 = int(bound(p4, "A_conv4", s)), int(bound(p4, "B_conv4", s))
            d4 = a4 - b4
        a8, b8 = int(bound(store, "A_conv8", s)), int(bound(store, "B_conv8", s))
        d8 = a8 - b8
        diff = "-" if p4 is None else f"{d8 - d4:+d}"
        print(f"    {s:>4}  {a4!s:>7} {b4!s:>7} {d4!s:>4}   {a8:>7} {b8:>7} {d8:>4}   {diff:>9}")
    print()


def report(store, p4, pair, wall, path, also):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    raw_tables(store, p4)

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
        if p4 is not None:
            have = [s for s in SEEDS[k] if get(p4, P4[k], s) is not None]
            c4 = sum(bound(p4, P4[k], s) for s in have)
            extra = f"   (recorded {P4[k]} at P=4, same seeds: bound {c4}/{len(have)})"
        print(f"  {a['label']:<46} bound {c[k]:>2}/{ns}"
              + (f"   discovered {d[k]:>2}/{ns}" if k in GATED else " " * 18)
              + f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{ns}"
              + (f"   [{failed} FAILED]" if failed else "") + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["ceiling_conv8"])
    valid = c["ceiling_conv8"] >= min(CEILING_MIN, nc) and pair["ok"]
    print(f"  ceiling_conv8 binds {c['ceiling_conv8']}/{nc} (needs >= {min(CEILING_MIN, nc)})   the pairing "
          f"CHECK (64) passed: {pair['ok']} (file {pair['path']} exists {pair['exists']}, CPU match "
          f"{pair.get('cpu_match')}, complete {pair.get('complete')}, reproduced {pair.get('reproduced')})")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND)")
    print("#" * 100)
    v = dict(bound=c, discovered=d, valid=valid, pairing=pair["ok"])
    verdict = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    nA, nB, nW = len(SEEDS["A_conv8"]), len(SEEDS["B_conv8"]), len(SEEDS["B_wide_conv8"])
    p1 = fisher_greater(c["A_conv8"], nA, c["B_conv8"], nB)
    s1 = SEEDS["A_conv8"]
    b1 = sum(bound(store, "A_conv8", s) and not bound(store, "B_conv8", s) for s in s1)
    c1 = sum(bound(store, "B_conv8", s) and not bound(store, "A_conv8", s) for s in s1)
    print(f"  P1 THE GATE ADDS AT P=8 (Fisher one-sided): A_conv8 {c['A_conv8']}/{nA} vs B_conv8 "
          f"{c['B_conv8']}/{nB}; p = {p1:.3g}")
    print(f"     exact McNemar two-sided on the pairs: A_conv8 only {b1}, B_conv8 only {c1}; p = "
          f"{mcnemar_exact(b1, c1):.3g}")
    print(f"     *** P1: {verdict(p1)} ***")
    if p4 is None:
        pos = neg = 0
        p2 = float("nan")
        v2 = "UNTESTED"
        d4s = d8s = []
    else:
        d4s = [d_of(p4, "A_conv4", "B_conv4", s) for s in P2_SEEDS]
        d8s = [d_of(store, "A_conv8", "B_conv8", s) for s in P2_SEEDS]
        pos, neg, p2 = sign_test_greater([b - a for a, b in zip(d4s, d8s)])
        v2 = verdict(p2)
    print(f"  P2 THE GATE'S ADVANTAGE GROWS WITH P (exact sign test, one-sided, on D_8 - D_4 over seeds "
          f"{P2_SEEDS[0]}-{P2_SEEDS[-1]}):")
    if p4 is not None:
        print(f"     sum D_4 = {sum(d4s):+d} (A_conv4 {sum(bound(p4, 'A_conv4', s) for s in P2_SEEDS)}, "
              f"B_conv4 {sum(bound(p4, 'B_conv4', s) for s in P2_SEEDS)})   sum D_8 = {sum(d8s):+d} "
              f"(A_conv8 {sum(bound(store, 'A_conv8', s) for s in P2_SEEDS)}, B_conv8 "
              f"{sum(bound(store, 'B_conv8', s) for s in P2_SEEDS)})")
    print(f"     D_8 - D_4: positive {pos}, negative {neg}, zero {len(P2_SEEDS) - pos - neg}; p = {p2:.3g}")
    print(f"     *** P2: {v2} ***")
    p3 = fisher_greater(c["A_conv8"], nA, c["B_wide_conv8"], nW)
    s3 = SEEDS["B_wide_conv8"]
    b3 = sum(bound(store, "A_conv8", s) and not bound(store, "B_wide_conv8", s) for s in s3)
    c3 = sum(bound(store, "B_wide_conv8", s) and not bound(store, "A_conv8", s) for s in s3)
    print(f"  P3 TWO CHANNELS BEAT ONE CHANNEL WITH THE SAME MEMORY AT P=8 (Fisher one-sided): A_conv8 "
          f"{c['A_conv8']}/{nA} vs B_wide_conv8 {c['B_wide_conv8']}/{nW}; p = {p3:.3g}")
    print(f"     exact McNemar two-sided on seeds {s3[0]}-{s3[-1]}: A_conv8 only {b3}, B_wide_conv8 only "
          f"{c3}; p = {mcnemar_exact(b3, c3):.3g}")
    print(f"     *** P3: {verdict(p3)} ***")
    pool_runs = ok_runs(store, "B_conv8") + ok_runs(store, "B_wide_conv8")
    xb = [r["end"]["conv_absmean"][2] for _, r in pool_runs if r["transition"] is not None]
    xn = [r["end"]["conv_absmean"][2] for _, r in pool_runs if r["transition"] is None]
    if len(xb) < M1_MIN or len(xn) < M1_MIN:
        m1, pm1, U = ("UNTESTABLE" if valid else "UNTESTED"), float("nan"), float("nan")
    else:
        U, pm1 = mann_whitney_greater(xb, xn)
        m1 = verdict(pm1)
    print(f"  M1-P8 LAG 2 STILL MARKS SINGLE-CHANNEL BINDING AT P=8 (exact Mann-Whitney, one-sided): "
          f"B_conv8 + B_wide_conv8, bound {len(xb)} (lag-2 median {med(xb)}) vs not bound {len(xn)} "
          f"(median {med(xn)}); U = {U}, p = {pm1:.3g}")
    print(f"     *** M1-P8: {m1} ***")
    v.update(P1=verdict(p1), P1_p=p1, P2=v2, P2_p=p2, P2_pos=pos, P2_neg=neg, P3=verdict(p3),
             P3_p=p3, M1_P8=m1, M1_P8_p=pm1)
    print()
    print("  READINGS:")
    rd = []
    if not valid:
        print("     (UNTESTED: no reading)")
    else:
        if v["P1"] == "SHOWN" and v["P2"] == "SHOWN":
            rd.append(("P1 SHOWN and P2 SHOWN", READINGS["P1+ P2+"]))
        elif v["P1"] == "SHOWN":
            rd.append(("P1 SHOWN and P2 NOT SHOWN", READINGS["P1+ P2-"]))
        else:
            rd.append(("P1 NOT SHOWN", READINGS["P1-"]))
        if v["P3"] == "SHOWN":
            rd.append(("P3 SHOWN", READINGS["P3+"]))
        for cond, text in rd:
            print(f"     {cond}: \"{text}\"")
    v["readings"] = [cond for cond, _ in rd]
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  per-seed outcome at P=4 (recorded) and P=8 (B = bound, D = bound and discovered, . = neither, "
          "- = not run):")
    print("    seed  " + "  ".join(f"{P4[k] + ' P4':>16} {k + ' P8':>16}" for k in KEYS))
    for s in P2_SEEDS:
        print(f"    {s:>4}  " + "  ".join(
            f"{mark(p4, P4[k], s):>16} {(mark(store, k, s) if s in SEEDS[k] else '-'):>16}" for k in KEYS))
    print("  exact McNemar two-sided (bound), every pair of arms at P=8, on their shared seeds:")
    for k1, k2 in itertools.combinations(("B_conv8", "A_conv8", "B_wide_conv8"), 2):
        sh = sorted(set(SEEDS[k1]) & set(SEEDS[k2]))
        b_ = sum(bound(store, k1, s) and not bound(store, k2, s) for s in sh)
        c_ = sum(bound(store, k2, s) and not bound(store, k1, s) for s in sh)
        print(f"    {k1:<12} vs {k2:<12} ({len(sh):>2} seeds): {k1} only {b_}, {k2} only {c_}; "
              f"p = {mcnemar_exact(b_, c_):.3g}")
    print()
    print(f"  final accuracy per arm, median [min, max], and the count >= {ACC_HIGH}:")
    for k in KEYS:
        accs = [r["acc"] for _, r in ok_runs(store, k)]
        un = [r["acc"] for _, r in ok_runs(store, k) if r["transition"] is None]
        print(f"    {k:<14} all {med(accs)}   >= {ACC_HIGH}: {sum(1 for x in accs if x >= ACC_HIGH)}/{len(accs)}   "
              f"not bound: {med(un)} ({len(un)})")
    print()
    print("  gated runs at the end: bound WITH routing (margin >= 0.9) or WITHOUT; failures by "
          "test_router_layout's classes (at P=8):")
    for k in GATED:
        rs = ok_runs(store, k)
        w_ = [s for s, r in rs if r["transition"] is not None and routed(r)]
        wo = [s for s, r in rs if r["transition"] is not None and not routed(r)]
        classes = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        for s, r in rs:
            if r["transition"] is None:
                classes[fail_class(r)].append(s)
        print(f"    {k:<14} bound with routing {len(w_)}, WITHOUT routing {len(wo)}"
              + (f" ({' '.join(f's{s}' for s in wo)})" if wo else "")
              + "   failures: " + "   ".join(f"{m_} {len(x)}" for m_, x in classes.items()))
        for m_, xs in classes.items():
            if xs:
                print(f"    {'':<14}   {m_:<14} " + " ".join(f"s{s}" for s in xs))
    print()
    print("  the convolution's mean |w| per lag at the end, median [min, max] (lag 2 = where a value "
          "position sees its context token):")
    for k in KEYS:
        rs = ok_runs(store, k)
        b_all = [r for _, r in rs if r["transition"] is not None]
        groups = (("bound", b_all), ("not bound", [r for _, r in rs if r["transition"] is None]))
        if k in GATED:
            groups = (("bound, routed", [r for r in b_all if routed(r)]),
                      ("bound, NOT routed", [r for r in b_all if not routed(r)]), groups[1])
        for lab, sub in groups:
            if sub:
                print(f"    {k:<14} {lab:<17} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  transition steps and plateau length (steps at held-out accuracy 0.45-0.55 before the "
          "transition), per bound run:")
    for k in KEYS:
        rs = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None]
        if not rs:
            print(f"    {k:<14} none bound")
            continue
        trs = [r["transition"] for _, r in rs]
        pls = [plateau(r) for _, r in rs]
        print(f"    {k:<14} transition median {statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   "
              f"plateau median {statistics.median(pls):.0f} [{min(pls)}, {max(pls)}]")
        print(f"    {'':<14} " + "  ".join(f"s{s}:{r['transition']}/{plateau(r)}" for s, r in rs))
    print()
    print("  transition steps at P=4 (recorded) vs P=8, on the same seeds; median [min, max] (bound runs):")
    for k in KEYS:
        t8 = [r["transition"] for _, r in ok_runs(store, k) if r["transition"] is not None]
        t4 = ([] if p4 is None else
              [r["transition"] for _, r in ok_runs(p4, P4[k], SEEDS[k]) if r["transition"] is not None])
        n4 = 0 if p4 is None else len(ok_runs(p4, P4[k], SEEDS[k]))
        f = lambda ts: (f"{statistics.median(ts):.0f} [{min(ts)}, {max(ts)}]" if ts else "--")
        print(f"    {k:<14} P=4 ({P4[k]}) bound {len(t4)}/{n4}: {f(t4):<22}   P=8 bound {len(t8)}/"
              f"{len(ok_runs(store, k))}: {f(t8)}")
    print()
    print(f"  the restart check per arm (held-out accuracy >= {A_CHECK} at step {T_CHECK}):")
    for k in KEYS:
        rs = ok_runs(store, k)
        passed = [(s, r) for s, r in rs if (a_ := curve_acc(r, T_CHECK)) is not None and a_ >= A_CHECK]
        pb = sum(1 for _, r in passed if r["transition"] is not None)
        thrown = [s for s, r in rs if r["transition"] is not None and (curve_acc(r, T_CHECK) or 0.0) < A_CHECK]
        print(f"    {k:<14} pass {len(passed)}/{len(rs)}; of those bound {pb}/{len(passed)}; binders the "
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
            print(f"    step {step:>3} {k:<14} " + "   ".join(parts))
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
                b2 = sum(1 for r in runs if r.get("ok") and r.get("transition") is not None)
                d2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                ns = len(SEEDS[k])
                print(f"    {k:<14} bound here {c[k]}/{ns}   other {b2}/{len(runs)}   pooled "
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
            tag = f"{k:<14} s{s}"
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
    ap = argparse.ArgumentParser(description="Does two channels' advantage grow from P=4 to P=8?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's p_scaling_results.json, for pooled counts")
    ap.add_argument("--p4", default=P4_FILE,
                    help="this machine's conv_lr_results.json (the P=4 runs to pair with)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("P-scaling: does two channels' advantage over one grow from P=4 to P=8?")
    print(f"  task_for(8, 2) (S=2 P=8 n_vals=16 n_q=1, length {TASK8.L}), n_layer=3 decay N=256, conv "
          f"'layer' width {CONV_K}, lr {LR4}, MAX_ITERS={MAX_ITERS}")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<46} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  paired with {args.p4}: seeds reused on purpose (each P=8 run is paired by seed with the "
          f"recorded P=4 run)")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        pair = verify(pool, args.p4)
        p4 = load_store(args.p4) if pair["exists"] else None
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, lr=LR4, p4=args.p4,
                             pairing={k: v for k, v in pair.items() if not k.endswith("_curve")},
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on task_for(8, 2), "
              f"evaluation charged once per {EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_job, KEYS)))
        for k in KEYS:
            print(f"  {k:<14} {costs[k] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds")
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
                print(f"   {sp['arm']:<14} seed {sp['seed']}  cached")
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
                print(f"   {sp['arm']:<14} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}  {extra}conv|w|={wstr(e)}"
                      f"{'  DISCOVERED' if rec['discovered'] else ''}  {rec['secs']:.0f}s  "
                      f"(elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<14} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, p4, pair, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
