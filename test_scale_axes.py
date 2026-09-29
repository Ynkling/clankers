#!/usr/bin/env python
"""
test_scale_axes.py — two gaps to close before any claim about scaling, both at lr 1e-3 (SUB_LR,
passed explicitly) with conv 'layer' width 4:
(A) at P=8, does one channel with the SAME total memory bind as well as the gate?
(B) with S=4 streams and k=4 channels, does the gate still find the stream routing?
Everything below is fixed before any result.

Run it directly:

    python test_scale_axes.py
    python test_scale_axes.py --also results/L/scale_axes_results.json

BACKGROUND
test_curriculum_confirm, both machines (seeds 200-219; L ran c5ec279):
- K1 NOT SHOWN on both: CUR_lo 16/20 (X), 17/20 (L); DIRECT_lo, P=8 from scratch at 1e-3,
  12/15 on each.
- K2 SHOWN on both: CUR_B_lo 0/15 (X), 3/15 (L).
- K3 RELIABLE on both: CUR_lo_R 20/20 on each.
- Every gated P=8 binder at 1e-3 was routed by stream (X 53/53; L: all).
- P=16 (descriptive, 3-stage curriculum): CUR16 4/5 on each (3 of the 8 binders not routed);
  CUR16_ceiling 3/3 on each.
- Correction to the reading of test_p_scaling: its P=8 collapse (A_conv8 12/60 at 4e-3) was
  mostly the learning rate. At 1e-3, P=8 from scratch bound 24/30. The curriculum's apparent
  rescue in test_load_curriculum compared against 4e-3 runs.
- The only memory-matched single channel so far (B_wide_conv at N=512) ran at 4e-3: 20/40 at
  P=4 and 5/32 at P=8. It has never run at 1e-3.
- Seed-level outcomes differ between machines; the verdict is per machine.

PART A: memory at P=8 (S=2, task_for(8, 2), up to 28800 steps, the same as DIRECT_lo)
- DIRECT8:   arm A + conv, all 8 keys from step 1 (test_p_scaling's A_conv8 at 1e-3, which is
             test_curriculum_confirm's DIRECT_lo).                        Seeds 220-239 (20)
- B_wide8:   single channel at N=512 (mult 16) + conv (test_p_scaling's B_wide_conv8 at 1e-3).
             Its fast-weight state equals DIRECT8's (CHECK 81).          Seeds 220-231 (12)
- B8:        single channel at N=256 + conv (reference; B_conv8 at 1e-3). Seeds 220-229 (10)
- ceiling8:  perfect gate + conv (validity; ceiling_conv8 at 1e-3).      Seeds 220-222 (3)

PART B: four streams (S=4, P=4, n_vals=16, n_q=1: task_for(4, 4); length 51; up to 28800 steps)
- A4k4:       arm A with k=4 channels (test_router_reliability's A_k4 construction,
              dict(A, n_ch=4)) + conv, i.e. test_short_conv's A_conv with n_ch=4.
                                                                          Seeds 220-239 (20)
- B4s:        single channel + conv (test_short_conv's B_conv).          Seeds 220-229 (10)
- ceiling4k4: perfect gate, k=4, stream s -> channel s (test_multilayer_binding.arms_for's
              ceiling at S=4: the existing perfect_gate_general, no new model code) + conv.
                                                                          Seeds 220-224 (5)
ceiling8 and ceiling4k4 run first. If ceiling4k4 binds fewer than 4/5, Part B is UNTESTED and
not run; if ceiling8 binds fewer than 2/3, Part A is UNTESTED and not run. Only the failing
part is skipped.

DEFINITIONS AT k=4
- BOUND: as before (transition not None: held-out accuracy >= 0.95 held to the end).
- Routing margin at the query (test_router_layout.routing_stats, already defined for any S):
  g_q.g_tgt minus the mean over the S-1 other streams' bindings of the queried key, g_q.g_dis.
  1 = perfect routing, 0 = stream-blind. Routed = margin >= 0.9.
- Also reported: the stream -> channel map at the end (the argmax of each stream's mean read
  gate at value positions) and whether it is one-to-one; eff_ch; per-stream held-out accuracy
  at the end, to show which streams merged in a partial failure.
- Failure classes: test_router_layout's, generalized to k channels by a multichannel eta^2
  (between-group over total sum of squares, summed over channels; groups stream / key /
  block / index, where block = the triple's index // P, the S-stream form of "half"). At k=2
  it equals the channel-0 eta^2 up to float32 rounding (CHECK 83; the two channel
  probabilities sum to 1 only to rounding). Margin and eta^2 are recorded at every
  evaluation for gated arms.

VALIDITY AND CLAIMS (outcome BOUND)
- VALID A: ceiling8 binds >= 2/3.
- VALID B: ceiling4k4 binds >= 4/5, AND the recurrent k=4 gate fits the perfect 4-way routing
  (the CHECK 45 recipe, test_router_reliability.gate_fit_k, on A4k4's model: argmax >= 0.99).
- M1 THE GATE BEATS ONE CHANNEL OF THE SAME MEMORY AT P=8: DIRECT8 (20) vs B_wide8 (12). Fisher
  one-sided (DIRECT8 higher); exact McNemar (two-sided) on seeds 220-231 alongside.
- S1 FOUR STREAMS BIND: A4k4's band: RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9, NEVER 0.
- S2 THE GATE BEATS ONE CHANNEL AT S=4: A4k4 (20) vs B4s (10). Fisher one-sided.
- M1 and S2 are SHOWN if p < 0.05, else NOT SHOWN.
- Readings, printed verbatim where they apply:
  - M1 SHOWN: "at P=8 the gate beats a single channel with the same total memory: the
    partition, not the memory size."
  - M1 NOT SHOWN: "at P=8 a single channel with the same total memory binds about as well as
    the gate: the channel advantage may be memory size."
  - S1 RELIABLE or MAJORITY, and S2 SHOWN: "the gate finds a 4-way stream routing: the
    mechanism extends beyond two streams."
  - S1 MINORITY or NEVER: "four streams defeat routing discovery from scratch at this size."
Fisher and McNemar are test_router_confirm's.

DIAGNOSTICS, not part of the verdict
- Per seed: outcome, transition, final accuracy; margin at 1200, 2400 and the end; for the k=4
  arms the stream -> channel map and eff_ch; per-stream accuracy.
- Failure classes (generalized); for A4k4 failures, how many streams share a channel.
- The restart check per arm (held-out accuracy >= 0.6 at 2400): pass count and how many of
  those bind.
- Lag weights by outcome; gradient norms at steps 1, 10 and 100.
- DIRECT8 vs test_curriculum_confirm's DIRECT_lo: a descriptive line (different seeds).

CHECKS (printed before any training): test_curriculum_confirm's verification (which runs every
earlier test's), then
  80 every arm's optimizer runs at lr 1e-3 (step pre-hook, through run_job), and every arm is
     built through the existing code paths: its arm dict equals its source arm's except the
     named fields, and its model at init equals the source path's (test_conv_lr.builder_for
     for Part A; test_multilayer_binding.build of test_router_reliability's A_k4 + conv, and of
     arms_for's S=4 ceiling + conv, for Part B);
  81 B_wide8's fast-weight state equals DIRECT8's (n_state); both parameter counts;
  82 task_for(4, 4) over 10,000 sequences: each (stream, key) once, S distinct values per key,
     the query answered by its own stream's value, vocabulary S+P+n_vals, length 51;
  83 the k=4 perfect gate zeroes every cross-stream score at every layer (CHECK 6's test at
     S=4, conv on with random weights); the margin is 1 on the perfect gate and 0 on the
     uniform gate at k=4; the multichannel eta^2 equals routing_stats' channel-0 eta^2 at k=2
     (to 1e-6: float32 rounding);
     the recurrent k=4 gate's fit to the perfect 4-way routing (recorded for VALID B, not
     asserted);
  84 a worker's run is bit-identical to the same run in the main process (A4k4 and B_wide8,
     1200 steps).

LOGISTICS. Parallel single-thread workers; wall clock projected before training (worst case,
no drop rule). Results persist atomically to scale_axes_results.json (gitignored; copied into
results/X/ after the run).

RESULT (full run, 80/80 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz, 4 workers
x 1 thread; projection 8.61 h worst case; 362.8 min after verification; no --also file; machine
X, commit 248c482).
  VALID A: ceiling8 bound 3/3. VALID B: ceiling4k4 bound 5/5 (all at 1200), and the recurrent
  k=4 gate fits the perfect 4-way routing (argmax 1.0000, mse 3.9e-06).
  M1 THE GATE BEATS ONE CHANNEL OF THE SAME MEMORY AT P=8: SHOWN. DIRECT8 13/20 vs B_wide8 0/12;
     Fisher one-sided p = 0.000223; McNemar on seeds 220-231: DIRECT8 only 8, B_wide8 only 0,
     p = 0.0078.
  S1 FOUR STREAMS BIND: MINORITY. A4k4 bound 9/20.
  S2 THE GATE BEATS ONE CHANNEL AT S=4: SHOWN. A4k4 9/20 vs B4s 0/10; Fisher one-sided p = 0.0117.
  The pre-registered readings: M1 SHOWN: "at P=8 the gate beats a single channel with the same
  total memory: the partition, not the memory size." S1 MINORITY: "four streams defeat routing
  discovery from scratch at this size." (S1 MINORITY rules out the S1 + S2 reading.)

  Diagnostics (not part of the verdict):
  - The memory-matched single channel does not bind at 1e-3 either: B_wide8 0/12 and B8 0/10,
    every run at 0.48-0.53 with both streams at ~0.5 (keys bound, streams at chance). B_wide8 has
    DIRECT8's fast-weight state (49152) and 1.8x its parameters (50944 vs 28480).
  - Every DIRECT8 binder ended routed (13/13, one-to-one maps). Its 7 failures: 5 POSITION near
    0.5 (s222, s223, s229, s236, s239) and 2 STREAM-PARTIAL at 0.89 (s221, s232; margins 0.87 and
    0.98 but accuracy still under 0.95 at 28800). DIRECT8 13/20 vs test_curriculum_confirm's
    DIRECT_lo 12/15 (different seeds).
  - Four streams: A4k4's 9 binders came late (transition median 14400, range 6000-26400); the
    margin was ~0 at 1200 and 2400 in 19/20 runs. 8 binders ended routed with a one-to-one
    stream -> channel map (margin 0.93-0.98); s237 bound WITHOUT routing (map [3, 0, 0, 0], margin
    0.34, end lag-2 |w| 0.089 vs 0.033 median for the routed binders).
  - A4k4's 11 failures fall into two groups, none one-to-one: 5 STREAM-PARTIAL at 0.74-0.75 with
    exactly two streams sharing a channel, the other two solved (per-stream accuracy 1.0, 1.0,
    ~0.5, ~0.5; the merged pair was streams 0+1 in s221, s226, s239 and 2+3 in s223, s225), and 6
    with every stream on one channel at 0.24-0.38 (POSITION s227, s232, s234, s235; OTHER s224;
    KEY s238).
  - The restart check (>= 0.6 at 2400) does not transfer to S=4: A4k4 passed it in 1/20 runs (s226,
    which did not bind), and all 9 binders failed it. DIRECT8 passed in 10/20, all 10 bound; 3 of
    the 10 that failed also bound.
  - One channel at S=4: B4s 0/10, every run at 0.23-0.26 (chance over the 4 streams' values).
"""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import time_per_step, fmt_step, EVAL_EVERY
from test_multilayer_binding import D, n_params, n_state, build, probe_batch
import test_channel_binding as tcb
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater, mcnemar_exact
from test_router_curriculum import (
    ARCH, SUB_LR, get, count, load_store, init_worker, makespan, cpu_model, same_weights,
    save_results, COLLAPSE, TAU_END, TIME_STEPS,
)
from test_router_discovery import model_fn
import test_router_reliability as trr
from test_router_reliability import run_one, strip_all, med, gate_fit_k, T_CHECK, A_CHECK
from test_router_layout import routing_stats, fail_class, eta2, MARGIN_PARTIAL, ETA_SPLIT
from test_readout_path import GRAD_STEPS
import test_short_conv as tsc
from test_short_conv import conv_stats, conv_grad_norms, wstr, CONV_K, LAYER, ROUTED_MARGIN
import test_conv_lr as tcl
from test_conv_lr import bound_r
import test_p_scaling as tps
import test_curriculum_confirm as tcc

# ── Settings (fixed before any result) ───────────────────────────────────────
LR = SUB_LR                                # 1e-3, passed explicitly to every run
MAX_ITERS = 28800
CEIL8_MIN, CEIL44_MIN = 2, 4               # VALID A needs ceiling8 >= 2/3; VALID B ceiling4k4 >= 4/5
FIT_MIN = 0.99                             # VALID B: the k=4 gate's argmax fit
ALPHA = 0.05
RELIABLE_S, MAJORITY_S = 18, 10            # S1 bands, of 20
GROUPS = ("stream", "key", "block", "index")
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "scale_axes_results.json"

TASK8 = tps.TASK8                          # task_for(8, 2)
TASK44 = tcb.task_for(4, 4)                # S=4, P=4, n_vals=16, n_q=1
TASKS = {"P8S2": TASK8, "P4S4": TASK44}
CEIL44_BASE = {a["key"]: a for a in tcb.arms(TASK44)}["ceiling"]      # perfect gate, k=S=4
A_K4_BASE = trr.ARM["A_k4"]                                          # dict(A, n_ch=4)
ARMS = [
    dict(tps.ARM["ceiling_conv8"], key="ceiling8", lr=LR, part="A",
         label="ceiling8    perfect gate + conv, P=8"),
    dict(tps.ARM["A_conv8"], key="DIRECT8", lr=LR, part="A",
         label="DIRECT8     arm A + conv, P=8"),
    dict(tps.ARM["B_wide_conv8"], key="B_wide8", lr=LR, part="A",
         label="B_wide8     single channel N=512 + conv, P=8"),
    dict(tps.ARM["B_conv8"], key="B8", lr=LR, part="A",
         label="B8          single channel N=256 + conv, P=8"),
    dict(CEIL44_BASE, key="ceiling4k4", model_kw=LAYER, task="P4S4", lr=LR, part="B",
         label="ceiling4k4  perfect gate k=4 + conv, S=4"),
    dict(tsc.ARM["A_conv"], key="A4k4", n_ch=4, task="P4S4", lr=LR, part="B",
         label="A4k4        arm A k=4 + conv, S=4"),
    dict(tsc.ARM["B_conv"], key="B4s", task="P4S4", lr=LR, part="B",
         label="B4s         single channel + conv, S=4"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
GATED = tuple(k for k in KEYS if ARM[k]["gate"] != "none")
PART_KEYS = {"A": ("ceiling8", "DIRECT8", "B_wide8", "B8"), "B": ("ceiling4k4", "A4k4", "B4s")}
CEIL = {"A": "ceiling8", "B": "ceiling4k4"}


def ceil_min(part):
    return CEIL8_MIN if part == "A" else CEIL44_MIN
SEEDS = dict(ceiling8=tuple(range(220, 223)), DIRECT8=tuple(range(220, 240)),
             B_wide8=tuple(range(220, 232)), B8=tuple(range(220, 230)),
             ceiling4k4=tuple(range(220, 225)), A4k4=tuple(range(220, 240)),
             B4s=tuple(range(220, 230)))
READINGS = {
    "M1+": "at P=8 the gate beats a single channel with the same total memory: the partition, not "
           "the memory size.",
    "M1-": "at P=8 a single channel with the same total memory binds about as well as the gate: the "
           "channel advantage may be memory size.",
    "S1+ S2+": "the gate finds a 4-way stream routing: the mechanism extends beyond two streams.",
    "S1-": "four streams defeat routing discovery from scratch at this size.",
}


def band(c):
    return ("RELIABLE" if c >= RELIABLE_S else "MAJORITY" if c >= MAJORITY_S
            else "MINORITY" if c >= 1 else "NEVER")


def builder_for(a):
    task = TASKS[a["task"]]
    return lambda aa, seed: model_fn(task, aa, seed)


# ── Statistics at k channels ─────────────────────────────────────────────────
def eta2_multi(x, lab):
    """eta^2 of a vector-valued x (..., k) grouped by lab: between-group over total sum of
    squares, both summed over the k channels; 0 if x is constant."""
    x, lab = x.reshape(-1, x.shape[-1]).double(), lab.reshape(-1)
    mu = x.mean(0)
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    between = sum((lab == g).sum() * ((x[lab == g].mean(0) - mu) ** 2).sum() for g in lab.unique())
    return float(between / tot)


@torch.no_grad()
def routing_k(model, task, probe):
    """The multichannel eta^2 of the read gate over the body's key and value positions (groups
    stream / key / block / index), and the stream -> channel map at value positions."""
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    S, P, n = task.S, task.P, task.S * task.P
    B = pinp.shape[0]
    j = torch.arange(n)
    ctx, key = pinp[:, 3 * j], pinp[:, 3 * j + 1] - S
    labels = dict(stream=ctx, key=key, block=(j // P).expand(B, n), index=j.expand(B, n))
    out = {}
    for role, off in (("key", 1), ("val", 2)):
        for name, lab in labels.items():
            out[f"etak_{role}_by_{name}"] = eta2_multi(gr[:, 3 * j + off], lab)
    gv = gr[:, 3 * j + 2]
    means = [gv[ctx == s].mean(0) for s in range(S)]
    cmap = [int(m_.argmax()) for m_ in means]
    out["ch_map"] = cmap
    out["one_to_one"] = len(set(cmap)) == S
    out["shared"] = sum(1 for c in cmap if cmap.count(c) > 1)
    out["stream_gate"] = [[round(v, 4) for v in m_.tolist()] for m_ in means]
    return out


@torch.no_grad()
def stream_acc(model, task, data):
    """Held-out accuracy per queried stream (tbo.evaluate's computation, split by stream)."""
    was = model.training
    model.eval()
    try:
        ql, qt, _ = task.select(tbo.logits_of(model, data[:, :-1]), data[:, 1:], None)
    finally:
        model.train(was)
    qs = data[:, task.qpos[0] - 1]
    hit = (ql.argmax(-1) == qt).float()
    return [hit[qs == s].mean().item() for s in range(task.S)]


def scale_stats(model, task, probe, step):
    """run_one's stats_fn: test_short_conv.conv_stats, plus (gated arms) routing_stats and
    routing_k at every evaluation, and per-stream held-out accuracy at the end."""
    out = conv_stats(model, task, probe, step)
    if model.n_ch >= 2:
        out.update(routing_stats(model, task, probe))
        out.update(routing_k(model, task, probe))
    if step == "end":
        out["stream_acc"] = stream_acc(model, task, eval_batch(task))
    return out


def fail_class_k(r):
    """test_router_layout.fail_class on the multichannel eta^2 (block for half)."""
    e = r["end"]
    return fail_class({"end": dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                                   eta_key_by_half=e["etak_key_by_block"],
                                   eta_key_by_index=e["etak_key_by_index"])})


# ── Jobs ─────────────────────────────────────────────────────────────────────
def run_job(sp):
    a = ARM[sp["arm"]]
    return run_one(a, sp["seed"], sp["iters"], task=TASKS[a["task"]], stats_fn=scale_stats,
                   grad_fn=conv_grad_norms, lr=LR, builder=builder_for(a))


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def lr_job(key):
    """CHECK 80: the optimizer lr over 2 steps of run_job for arm `key` (step pre-hook)."""
    seen = []
    h = register_optimizer_step_pre_hook(lambda opt, args, kw: seen.append(sorted({g["lr"] for g in opt.param_groups})))
    try:
        r = run_job(spec(key, 3, 2))
    finally:
        h.remove()
    return seen, r["ok"]


def time_job(key):
    """test_p_scaling.time_job's recipe on the arm's own task."""
    a = ARM[key]
    task = TASKS[a["task"]]
    mk = builder_for(a)(a, 0)
    per = time_per_step(task, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(task)
    t0 = time.time()
    tbo.evaluate(m, task, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_curriculum_confirm.py's verification (which runs test_load_curriculum's, and so on")
    print(f"down to test_multilayer_binding's), with its default --prev ({tcc.PREV_FILE}):")
    tcc.verify(pool, tcc.PREV_FILE)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f84 = {k: pool.submit(run_job, spec(k, 1, EVAL_EVERY)) for k in ("A4k4", "B_wide8")}
    f80 = {k: pool.submit(lr_job, k) for k in KEYS}

    print(f"CHECK 80 every arm runs at lr {LR} and is built through the existing code paths:")
    src = dict(ceiling8=(tps.ARM["ceiling_conv8"], "test_p_scaling's ceiling_conv8", ()),
               DIRECT8=(tps.ARM["A_conv8"], "test_p_scaling's A_conv8", ()),
               B_wide8=(tps.ARM["B_wide_conv8"], "test_p_scaling's B_wide_conv8", ()),
               B8=(tps.ARM["B_conv8"], "test_p_scaling's B_conv8", ()),
               ceiling4k4=(CEIL44_BASE, "arms_for(task_for(4, 4))'s ceiling", ("model_kw", "task")),
               A4k4=(tsc.ARM["A_conv"], "test_short_conv's A_conv", ("n_ch", "task")),
               B4s=(tsc.ARM["B_conv"], "test_short_conv's B_conv", ("task",)))
    good = True
    for k in KEYS:
        ref, name, extra = src[k]
        drop = ("key", "label", "lr", "part") + extra
        same = {x: v for x, v in ARM[k].items() if x not in drop} == {x: v for x, v in ref.items() if x not in drop}
        seen, rok = f80[k].result()
        lr_ok = seen == [[LR], [LR]] and rok
        a = ARM[k]
        m_new = builder_for(a)(a, 3)()
        if a["part"] == "A":
            m_ref, ref_path = tcl.builder_for(ref)(ref, 3)(), "test_conv_lr.builder_for"
        elif k == "A4k4":
            m_ref, ref_path = build(TASK44, dict(A_K4_BASE, model_kw=LAYER), ARCH, 3), "build(A_k4 + conv)"
        elif k == "ceiling4k4":
            m_ref, ref_path = build(TASK44, dict(CEIL44_BASE, model_kw=LAYER), ARCH, 3), "build(arms_for ceiling + conv)"
        else:
            m_ref, ref_path = build(TASK44, ref, ARCH, 3), "build(B_conv)"
        w_ok = same_weights(m_new, m_ref)
        g = same and lr_ok and w_ok
        good &= g
        chg = ", ".join(("lr",) + extra)
        print(f"         {k:<10} = {name:<37} except key, label, {chg}: {same}   optimizer lr over 2 steps "
              f"{seen}   init == {ref_path} (seed 3): {w_ok}  -> {'OK' if g else 'WRONG'}")
    print(f"         A4k4's source: test_router_reliability's A_k4 = dict(A, n_ch=4); A_conv = A + conv: "
          f"{({x: v for x, v in A_K4_BASE.items() if x not in ('key', 'label')} == {x: v for x, v in dict(tsc.ARM['A_conv'], n_ch=4).items() if x not in ('key', 'label', 'model_kw')})}")
    ok &= good
    print()

    print("CHECK 81 B_wide8's memory:")
    md = builder_for(ARM["DIRECT8"])(ARM["DIRECT8"], 0)()
    mw = builder_for(ARM["B_wide8"])(ARM["B_wide8"], 0)()
    g = n_state(md) == n_state(mw)
    print(f"         DIRECT8: k={md.n_ch}, N={ARM['DIRECT8']['mult'] * D}, fast-weight state {n_state(md)}, parameters "
          f"{n_params(md)}")
    print(f"         B_wide8: k={mw.n_ch}, N={ARM['B_wide8']['mult'] * D}, fast-weight state {n_state(mw)}, parameters "
          f"{n_params(mw)}")
    print(f"         -> {'STATE MATCHED' if g else 'STATE DIFFERS'}")
    ok &= g
    print()

    N = 10_000
    t = TASK44
    S, P = t.S, t.P
    print(f"CHECK 82 task_for(4, 4), over {N:,} sequences (generator seed 82):")
    tok, qs = t.make_batch(N, torch.Generator().manual_seed(82))
    j = torch.arange(S * P)
    ctx, key, val = tok[:, 3 * j], tok[:, 3 * j + 1] - S, tok[:, 3 * j + 2]
    once = bool((torch.sort(ctx * P + key, 1).values == torch.arange(S * P)).all())
    kv = key.view(N, P, S)
    grouped = bool((kv == kv[..., :1]).all()) and bool((torch.sort(ctx.view(N, P, S), -1).values == torch.arange(S)).all())
    distinct = bool((torch.sort(val.view(N, P, S), -1).values.diff(dim=-1) != 0).all())
    n = S * P
    qc, qk, qv = tok[:, 3 * n], tok[:, 3 * n + 1] - S, tok[:, 3 * n + 2]
    hit = (ctx == qc[:, None]) & (key == qk[:, None])
    answered = (bool((hit.sum(1) == 1).all()) and bool((val[hit] == qv).all())
                and bool((qc == qs.reshape(-1)).all()))
    voc = t.vocab == S + P + t.n_vals and bool((tok >= 0).all()) and bool((tok < t.vocab).all())
    length = t.L == 51 and tok.shape[1] == 51 and t.qpos == [49]
    good = once and grouped and distinct and answered and voc and length
    print(f"         each (stream, key) once: {once}   grouped (each key's S triples adjacent, one per stream): {grouped}   "
          f"S distinct values per key: {distinct}")
    print(f"         the query answered by its own stream's value: {answered}   vocabulary {t.vocab} = S+P+n_vals = "
          f"{S}+{P}+{t.n_vals}, token ids in range: {voc}   length {t.L}, query position {t.qpos}: {length}")
    print(f"         e.g. {tcc.tlc.tps_decode(tok[0], t)}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 83 the k=4 definitions:")
    a = ARM["ceiling4k4"]
    x = TASK44.make_batch(16, torch.Generator().manual_seed(830))[0][:, :-1]
    mdl = builder_for(a)(a, 6)().eval()
    with torch.no_grad():
        mdl.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(831)))
    mdl.attn.record = []
    with torch.no_grad():
        mdl(x, TAU_END)
    lab = TASK44.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s_[:, 0][cross].abs().max().item() for s_ in mdl.attn.record]
    live = [s_[:, 0][~cross].abs().max().item() for s_ in mdl.attn.record]
    mdl.attn.record = None
    g1 = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0
    print(f"     (a) ceiling4k4, random conv weights: layers seen {len(maxc)} (n_layer={ARCH['n_layer']})")
    for i, (c, l_) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l_:.2e}")
    probe44 = probe_batch(TASK44, 3)
    m_perf = builder_for(a)(a, 3)()
    uni = dict(a, gate="uniform")
    m_uni = builder_for(uni)(uni, 3)()
    mp_, mu_ = routing_stats(m_perf, TASK44, probe44)["margin"], routing_stats(m_uni, TASK44, probe44)["margin"]
    rk = routing_k(m_perf, TASK44, probe44)
    g2 = abs(mp_ - 1.0) < 1e-6 and abs(mu_) < 1e-6 and rk["ch_map"] == [0, 1, 2, 3] and rk["one_to_one"]
    print(f"     (b) margin at k=4: perfect gate {mp_:.6f} (want 1), uniform gate {mu_:.6f} (want 0); perfect gate's "
          f"stream -> channel map {rk['ch_map']} (one-to-one {rk['one_to_one']}), eta^2 by stream key "
          f"{rk['etak_key_by_stream']:.4f} val {rk['etak_val_by_stream']:.4f}: {g2}")
    diffs = []
    for k8 in ("DIRECT8", "ceiling8"):
        a8 = ARM[k8]
        m8 = builder_for(a8)(a8, 3)()
        if k8 == "DIRECT8":
            with torch.no_grad():
                m8.W_g.normal_(0, 1.0, generator=torch.Generator().manual_seed(832))
        pr8 = probe_batch(TASK8, 3)
        r0, r1 = routing_stats(m8, TASK8, pr8), routing_k(m8, TASK8, pr8)
        for role in ("key", "val"):
            for g0, g_ in (("stream", "stream"), ("key", "key"), ("half", "block"), ("index", "index")):
                diffs.append(abs(r0[f"eta_{role}_by_{g0}"] - r1[f"etak_{role}_by_{g_}"]))
        fc0 = fail_class({"end": r0})
        fck = fail_class_k({"end": dict(r0, **r1)})
        diffs.append(0.0 if fc0 == fck else 1.0)
    g3 = max(diffs) < 1e-6
    print(f"     (c) at k=2 (DIRECT8 with a random W_g, and ceiling8; seed-3 probe) the multichannel eta^2 equals "
          f"routing_stats' channel-0 eta^2 (block = half) to float32 rounding, and fail_class_k == fail_class: "
          f"max |diff| {max(diffs):.1e} (< 1e-6): {g3}")
    fit_mse, fit_am = gate_fit_k(TASK44, ARM["A4k4"])
    print(f"     (d) the recurrent k=4 gate (A4k4's model) fit to the perfect 4-way routing, test_router_reliability."
          f"gate_fit_k (CHECK 45's recipe): mse {fit_mse:.2e}   argmax match {fit_am:.4f}  -> "
          f"{'FITS' if fit_am >= FIT_MIN else 'DOES NOT FIT'} (>= {FIT_MIN}; recorded for VALID B, not asserted)")
    good = g1 and g2 and g3
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 84 a worker's run is bit-identical to the same run here:")
    good = True
    for k in ("A4k4", "B_wide8"):
        rw, rh = f84[k].result(), run_job(spec(k, 1, EVAL_EVERY))
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        extra = (f"   margin {rh['end']['margin']:.4f}   map {rh['end']['ch_map']}   eff_ch {rh['end']['eff_ch']:.3f}"
                 if k == "A4k4" else "")
        print(f"         {k:<8} seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   acc {rh['acc']:.4f}   "
              f"per-stream acc {[round(v, 3) for v in rh['end']['stream_acc']]}{extra}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print("ALL VERIFICATION CHECKS PASSED\n" if ok else "SOME VERIFICATION CHECKS FAILED\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(fit_mse=fit_mse, fit_argmax=fit_am)


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def ok_runs(store, key):
    return [(s, r) for s in SEEDS[key] if (r := get(store, key, s)) and r.get("ok")]


def at(curve, step):
    return next((a for st, a, _ in curve if st == step), None)


def st_at(r, step):
    return next((x for x in r["stats"] if x["step"] == step), None)


def margin_at(r, step):
    x = st_at(r, step)
    return None if x is None else x.get("margin")


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def routed(r):
    return r["end"].get("margin", -1) >= ROUTED_MARGIN


def flags(k, r):
    if r["transition"] is not None:
        fl = ["BOUND"]
        if k in GATED:
            fl.append("routed" if routed(r) else "NOT routed")
    else:
        fl = [fail_class_k(r)] if k in GATED else []
    return fl + (["collapsed"] if r["collapsed"] else [])


def raw_tables(store):
    print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'acc@2400':>8} {'m@1200':>7} {'m@2400':>7} "
          f"{'m end':>7} {'eff_ch':>6} {'map':>10} {'1-1':>4}  per-stream acc          conv |w| lag 0/1/2/3   flags")
    for k in KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<10} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<10} {s:>4}  FAILED — {r.get('error')}")
                continue
            e = r["end"]
            g = k in GATED
            cmap = "".join(str(c) for c in e["ch_map"]) if g else "--"
            psa = "/".join(f"{v:.2f}" for v in e["stream_acc"])
            print(f"  {k:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {fmt(at(r['curve'], T_CHECK)):>8} "
                  f"{fmt(margin_at(r, EVAL_EVERY)):>7} {fmt(margin_at(r, 2 * EVAL_EVERY)):>7} {fmt(e.get('margin')):>7} "
                  f"{(format(e['eff_ch'], '.3f') if g else '--'):>6} {cmap:>10} {(str(e['one_to_one'])[0] if g else '-'):>4}  "
                  f"{psa:<22} {wstr(e):<23}  {' '.join(flags(k, r))}")
        print()


def report(store, fit, wall, path, also, ran):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  (m = routing margin at the query; map = the stream -> channel argmax at value positions, stream 0 first; "
          f"1-1 = one-to-one; per-stream acc at the end, stream 0 first)")
    raw_tables(store)

    print("=" * 100)
    print("COUNTS (BOUND = transition not None; routed = end margin >= 0.9)")
    print("=" * 100)
    c = {}
    for a in ARMS:
        k, ns = a["key"], len(SEEDS[a["key"]])
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        rs = [get(store, k, s) for s in SEEDS[k]]
        done = sum(1 for r in rs if r and r.get("ok"))
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        extra = ""
        if k in GATED:
            extra = f"   bound and routed {sum(1 for _, r in ok_runs(store, k) if r['transition'] is not None and routed(r))}"
        print(f"  {a['label']:<46} bound {c[k]:>2}/{ns}   completed {done}/{ns}" + (f"   [{failed} FAILED]" if failed else "")
              + ("" if ran[a["part"]] else "   (part not run)") + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nA, nB = len(SEEDS["ceiling8"]), len(SEEDS["ceiling4k4"])
    needA, needB = min(CEIL8_MIN, nA), min(CEIL44_MIN, nB)
    validA = ran["A"] and c["ceiling8"] >= needA
    fit_ok = fit["fit_argmax"] >= FIT_MIN
    validB = ran["B"] and c["ceiling4k4"] >= needB and fit_ok
    print(f"  A: ceiling8 binds {c['ceiling8']}/{nA} (needs >= {needA})" + ("" if ran["A"] else "; Part A not run"))
    print(f"     *** VALID A: {'VALID' if validA else 'UNTESTED'} ***")
    print(f"  B: ceiling4k4 binds {c['ceiling4k4']}/{nB} (needs >= {needB}); the k=4 gate's fit argmax "
          f"{fit['fit_argmax']:.4f} (needs >= {FIT_MIN}): {fit_ok}" + ("" if ran["B"] else "; Part B not run"))
    print(f"     *** VALID B: {'VALID' if validB else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND)")
    print("#" * 100)
    v = dict(bound=c, validA=validA, validB=validB, fit=fit)
    ver = lambda p, valid: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    nd, nw = len(SEEDS["DIRECT8"]), len(SEEDS["B_wide8"])
    p1 = fisher_greater(c["DIRECT8"], nd, c["B_wide8"], nw)
    sh = SEEDS["B_wide8"]
    b_ = sum(bound(store, "DIRECT8", s) and not bound(store, "B_wide8", s) for s in sh)
    c_ = sum(bound(store, "B_wide8", s) and not bound(store, "DIRECT8", s) for s in sh)
    print(f"  M1 THE GATE BEATS ONE CHANNEL OF THE SAME MEMORY AT P=8 (Fisher one-sided): DIRECT8 {c['DIRECT8']}/{nd} vs "
          f"B_wide8 {c['B_wide8']}/{nw}; p = {p1:.3g}")
    print(f"     exact McNemar two-sided on seeds {sh[0]}-{sh[-1]}: DIRECT8 only {b_}, B_wide8 only {c_}; p = "
          f"{mcnemar_exact(b_, c_):.3g}")
    print(f"     *** M1: {ver(p1, validA)} ***")
    na, nb = len(SEEDS["A4k4"]), len(SEEDS["B4s"])
    vs1 = band(c["A4k4"]) if validB else "UNTESTED"
    print(f"  S1 FOUR STREAMS BIND: A4k4 bound {c['A4k4']}/{na} (RELIABLE >= {RELIABLE_S}, MAJORITY {MAJORITY_S}-"
          f"{RELIABLE_S - 1}, MINORITY 1-{MAJORITY_S - 1}, NEVER 0)")
    print(f"     *** S1: {vs1} ***")
    p2 = fisher_greater(c["A4k4"], na, c["B4s"], nb)
    print(f"  S2 THE GATE BEATS ONE CHANNEL AT S=4 (Fisher one-sided): A4k4 {c['A4k4']}/{na} vs B4s {c['B4s']}/{nb}; "
          f"p = {p2:.3g}")
    print(f"     *** S2: {ver(p2, validB)} ***")
    v.update(M1=ver(p1, validA), M1_p=p1, M1_mcnemar=[b_, c_, mcnemar_exact(b_, c_)], S1=vs1, S2=ver(p2, validB),
             S2_p=p2)
    print()
    print("  READINGS:")
    rd = []
    if v["M1"] == "SHOWN":
        rd.append(("M1 SHOWN", READINGS["M1+"]))
    if v["M1"] == "NOT SHOWN":
        rd.append(("M1 NOT SHOWN", READINGS["M1-"]))
    if v["S1"] in ("RELIABLE", "MAJORITY") and v["S2"] == "SHOWN":
        rd.append((f"S1 {v['S1']} and S2 SHOWN", READINGS["S1+ S2+"]))
    if v["S1"] in ("MINORITY", "NEVER"):
        rd.append((f"S1 {v['S1']}", READINGS["S1-"]))
    for cond, text in rd:
        print(f"     {cond}: \"{text}\"")
    if not rd:
        print("     (none of the pre-registered readings applies)")
    v["readings"] = [cond for cond, _ in rd]
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  per seed (B = bound and routed or ungated, b = bound NOT routed, . = not bound, - = not run):")
    print(f"    seed  " + " ".join(f"{k:>10}" for k in KEYS))
    for s in range(220, 240):
        row = []
        for k in KEYS:
            r = get(store, k, s) if s in SEEDS[k] else None
            if r is None or not r.get("ok"):
                row.append("-" if r is None else "?")
            elif r["transition"] is None:
                row.append(".")
            else:
                row.append("b" if k in GATED and not routed(r) else "B")
        print(f"    {s:>4}  " + " ".join(f"{x:>10}" for x in row))
    print()
    print(f"  the restart check (held-out accuracy >= {A_CHECK} at {T_CHECK}): passed, and of those bound; failed, and "
          f"of those bound:")
    for k in KEYS:
        rs = ok_runs(store, k)
        ps = [r for _, r in rs if (x := at(r["curve"], T_CHECK)) is not None and x >= A_CHECK]
        fs = [r for _, r in rs if not ((x := at(r["curve"], T_CHECK)) is not None and x >= A_CHECK)]
        print(f"    {k:<10} passed {len(ps):>2}/{len(rs):<2} bound {sum(1 for r in ps if r['transition'] is not None):>2}   "
              f"failed {len(fs):>2} bound {sum(1 for r in fs if r['transition'] is not None):>2}")
    print()
    print("  transitions, median [min, max]:")
    for k in KEYS:
        trs = [r["transition"] for _, r in ok_runs(store, k) if r["transition"] is not None]
        print(f"    {k:<10} " + (f"{statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   " + " ".join(
            f"s{s}:{r['transition']}" for s, r in ok_runs(store, k) if r["transition"] is not None) if trs else "none bound"))
    print()
    print("  gated runs: margin at 1200, 2400 and the end, median [min, max]; eff_ch at the end; failure classes "
          "(generalized to k channels) at the end:")
    for k in GATED:
        rs = ok_runs(store, k)
        if not rs:
            continue
        print(f"    {k:<10} m@1200 {med([x for _, r in rs if (x := margin_at(r, EVAL_EVERY)) is not None]):<22} "
              f"m@2400 {med([x for _, r in rs if (x := margin_at(r, 2 * EVAL_EVERY)) is not None]):<22} "
              f"m end {med([r['end']['margin'] for _, r in rs]):<22} eff_ch {med([r['end']['eff_ch'] for _, r in rs])}")
        cls = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        for s, r in rs:
            if r["transition"] is None:
                cls[fail_class_k(r)].append(s)
        bw = [s for s, r in rs if r["transition"] is not None and routed(r)]
        bwo = [s for s, r in rs if r["transition"] is not None and not routed(r)]
        print(f"    {'':<10} bound with routing {len(bw)}, WITHOUT routing {len(bwo)}"
              + (f" ({' '.join(f's{s}' for s in bwo)})" if bwo else "")
              + "   failures: " + "   ".join(f"{m_} {len(x)}" for m_, x in cls.items()))
        for m_, xs in cls.items():
            if xs:
                print(f"    {'':<10}   {m_:<14} " + " ".join(f"s{s}" for s in xs))
    print()
    if ok_runs(store, "A4k4"):
        print("  A4k4 at the end: the stream -> channel map, streams sharing a channel, eff_ch, per-stream accuracy:")
        for s, r in ok_runs(store, "A4k4"):
            e = r["end"]
            print(f"    s{s}: {'BOUND' if r['transition'] is not None else 'not bound':<9} map {e['ch_map']} one-to-one "
                  f"{e['one_to_one']!s:<5} sharing {e['shared']}   eff_ch {e['eff_ch']:.3f}   margin {e['margin']:.3f}   "
                  f"per-stream acc {[round(v_, 3) for v_ in e['stream_acc']]}")
        fl = [r for _, r in ok_runs(store, "A4k4") if r["transition"] is None]
        if fl:
            sh_ = [r["end"]["shared"] for r in fl]
            print(f"    A4k4 failures: streams sharing a channel per run {sh_}; one-to-one in "
                  f"{sum(1 for r in fl if r['end']['one_to_one'])}/{len(fl)}")
        print()
    print("  the convolution's mean |w| per lag at the end, median [min, max]:")
    for k in KEYS:
        rs = ok_runs(store, k)
        b_all = [r for _, r in rs if r["transition"] is not None]
        grp = [("bound", b_all), ("not bound", [r for _, r in rs if r["transition"] is None])]
        if k in GATED:
            grp = [("bound, routed", [r for r in b_all if routed(r)]),
                   ("bound, NOT routed", [r for r in b_all if not routed(r)]), grp[1]]
        for lab, sub in grp:
            if sub:
                print(f"    {k:<10} {lab:<17} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median [min, max]:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>4} {k:<10} " + "   ".join(parts))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
    print()
    cc = load_store(tcc.RESULTS_FILE) if os.path.exists(tcc.RESULTS_FILE) else None
    if cc and cc.get("runs"):
        rs = [r for kk, r in cc["runs"].items() if kk.split("|")[0] == "DIRECT_lo"]
        print(f"  DIRECT8 (seeds 220-239) bound {c['DIRECT8']}/{nd}; test_curriculum_confirm's DIRECT_lo (the same arm, "
              f"seeds 200-214; {tcc.RESULTS_FILE}, git {cc.get('meta', {}).get('git')}) bound "
              f"{sum(1 for r in rs if bound_r(r))}/{len(rs)}. Different seeds: descriptive only.")
    else:
        print(f"  DIRECT8 bound {c['DIRECT8']}/{nd}; {tcc.RESULTS_FILE} not found for the DIRECT_lo line.")
    print()
    print("  exact McNemar two-sided (bound), per seed:")
    for k1, k2 in (("DIRECT8", "B8"), ("B_wide8", "B8"), ("A4k4", "B4s")):
        sh2 = [s for s in SEEDS[k1] if s in SEEDS[k2]]
        b2 = sum(bound(store, k1, s) and not bound(store, k2, s) for s in sh2)
        c2 = sum(bound(store, k2, s) and not bound(store, k1, s) for s in sh2)
        print(f"    {k1:<8} vs {k2:<4} ({len(sh2):>2} seeds): {k1} only {b2}, {k2} only {c2}; p = {mcnemar_exact(b2, c2):.3g}")
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git {other.get('meta', {}).get('git')}")
            for k in KEYS:
                rs2 = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                b2_ = sum(1 for r in rs2 if r.get("ok") and r.get("transition") is not None)
                nh = sum(1 for s in SEEDS[k] if get(store, k, s) is not None)
                print(f"    {k:<10} bound here {c[k]}/{nh}   other {b2_}/{len(rs2)}   pooled {c[k] + b2_}/{nh + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — per evaluation (every {EVAL_EVERY} steps): held-out accuracy x100, and for gated arms the routing "
          f"margin x100")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            tag = f"{k:<10} s{s}"
            acc = "".join(f"{round(a_ * 100):>4}" for _, a_, _ in r["curve"])
            end = f"bound @{r['transition']}" if r["transition"] else "not bound"
            if k in GATED:
                mg = "".join((f"{round(m_ * 100):>4}" if (m_ := margin_at(r, st)) is not None else "  --")
                             for st, _, _ in r["curve"])
                print(f"  {tag} acc {acc}")
                print(f"  {'':<{len(tag)}} mrg {mg}  -> {end}")
            else:
                print(f"  {tag} acc {acc}  -> {end}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def describe(rec, key):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr, e = rec["transition"], rec["end"]
    g = key in GATED
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} "
            + (f"margin {fmt(margin_at(rec, EVAL_EVERY))}->{fmt(e.get('margin'))} map {e['ch_map']} eff_ch "
               f"{e['eff_ch']:.2f} " if g else "")
            + f"streams {[round(v, 2) for v in e['stream_acc']]} conv|w|={wstr(e)}")


def main():
    ap = argparse.ArgumentParser(description="Memory-matched single channel at P=8; four streams at k=4.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None, help="a second machine's scale_axes_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    iters = MAX_ITERS

    print("=" * 100)
    print("Scale axes: (A) the gate vs one channel of the same memory at P=8; (B) four streams with k=4")
    print(f"  lr {LR} (SUB_LR, explicit), n_layer=3 decay, conv 'layer' width {CONV_K}; up to {iters} steps, evaluation "
          f"every {EVAL_EVERY}, early stop after 3 evaluations >= 0.95")
    print(f"  Part A: task_for(8, 2) (length {TASK8.L}, vocab {TASK8.vocab});  Part B: task_for(4, 4) (S=4, P=4, length "
          f"{TASK44.L}, vocab {TASK44.vocab})")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<46} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        fit = verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR,
                             max_iters=iters, fit=fit, seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on the arm's task, one evaluation "
              f"charged per {EVAL_EVERY}; worst case = every run to {iters} steps")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        for k in KEYS:
            print(f"  {k:<10} {cost[k] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds   (a full run {cost[k] * iters / 60:.1f} min)")
        d_ceil = [cost[k] * iters for k in ("ceiling8", "ceiling4k4") for _ in SEEDS[k]]
        d_rest = [cost[k] * iters for k in KEYS if not k.startswith("ceiling") for _ in SEEDS[k]]
        mk1, mk2 = makespan(d_ceil, args.workers), makespan(d_rest, args.workers)
        print(f"  ceilings first: {len(d_ceil)} runs, {mk1 / 3600:.2f} h;  then {len(d_rest)} runs: serial "
              f"{sum(d_rest) / 3600:.2f} h, {mk2 / 3600:.2f} h on {args.workers} workers;  total {(mk1 + mk2) / 3600:.2f} h")
        store["meta"]["projected_wall_h"] = (mk1 + mk2) / 3600
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<10} seed {sp['seed']}  {describe(rec, sp['arm'])}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)")

        def run_specs(specs):
            pending, inflight = list(specs), {}
            while pending or inflight:
                while pending and len(inflight) < args.workers:
                    sp = pending.pop(0)
                    inflight[pool.submit(run_job, sp)] = sp
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done:
                    sp = inflight.pop(f)
                    try:
                        rec = f.result()
                    except Exception as e:
                        rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
                    record(sp, rec)

        def todo(k):
            out = []
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<10} seed {s}  cached")
                else:
                    out.append(spec(k, s, iters))
            return out

        print("=" * 100)
        print(f"RUNS — the ceilings first (ceiling8 {len(SEEDS['ceiling8'])}, ceiling4k4 {len(SEEDS['ceiling4k4'])})")
        print("=" * 100)
        run_specs(todo("ceiling4k4") + todo("ceiling8"))
        ran = {}
        for part, ck in CEIL.items():
            n_ = sum(bound(store, ck, s) for s in SEEDS[ck])
            need = min(ceil_min(part), len(SEEDS[ck]))
            ran[part] = n_ >= need
            print(f"   {ck} bound {n_}/{len(SEEDS[ck])} (needs >= {need}): Part {part} "
                  + ("runs" if ran[part] else "is UNTESTED and NOT run"))
        store["meta"]["parts_run"] = ran
        save_results(args.results, store)
        print()
        order = ("B_wide8", "A4k4", "DIRECT8", "B8", "B4s")
        rest = [sp for k in order if ran[ARM[k]["part"]] for sp in todo(k)]
        print("=" * 100)
        print(f"RUNS — the rest ({len(rest)} runs)")
        print("=" * 100)
        run_specs(rest)
        print()

    report(store, fit, time.time() - t0, args.results, args.also, ran)


if __name__ == "__main__":
    main()
