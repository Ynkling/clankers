#!/usr/bin/env python
"""
test_stream_curriculum.py — does a curriculum over the number of streams per sequence (2, then 4,
then 8 of 8) let the gate learn eight streams, which failed from scratch? lr 1e-3 (SUB_LR, passed
explicitly) in every stage; conv 'layer' width 4; k=16. Everything below is fixed before any
result.

Run it directly:

    python test_stream_curriculum.py
    python test_stream_curriculum.py --also results/L/stream_curriculum_results.json

BACKGROUND
test_stream_recipe, both machines, seeds 240-259 (L ran 092937b):
- Four streams:
  - A4k16_R bound 17/20 (X, R1 MAJORITY) and 19/20 (L, RELIABLE): 36/40 pooled.
  - R2: NOT PRECISE on X (17/20; the three that passed and failed were merges, three streams
    sharing at ~0.5 or two at 0.88), PRECISE on L (19/19).
  - R3 SHOWN on both (A4k4_R 10/20 and 8/20; pooled McNemar 20 vs 2).
  - R4 NOT SHOWN on both (A4k16 12/20 and 9/20 vs A4k4 10/20 and 5/20).
- Correction to test_stream_channels' reading: partial merges at k=16 were 8/40 here (1/40
  there); pooled 9/80 at k=16 vs 32/80 at k=4.
- Eight streams (L only; X dropped Part 2 under the 12 h rule):
  - ceiling8k16 bound 2/2 (steps 2400 and 3600).
  - A8k16_R bound 0/5. All 25 attempts were at 0.06-0.09 at step 4800, below the 1/8 that binding
    keys alone gives. The continued attempts ended at 0.09-0.30.
  - Reading (post hoc): without routing, eight streams leave nothing learnable early, so the gate
    gets no signal. A stream curriculum gives the model a learnable start; test_curriculum_
    confirm's key curriculum reached P=16 this way (descriptive).

TASK KNOB
BindTask gained s_active (default None = S; None or S leaves the task bit-identical to the old
make_batch, CHECK 95). Each sequence draws s_active distinct streams uniformly from the S; for
each of the P keys it draws s_active distinct values and places the s_active triples
[CTX_s, KEY_i, VAL] adjacent, in random stream order (the grouped layout), keys in random order.
The query is drawn from the s_active*P triples and answered by its own stream's value. The
vocabulary stays S's, so one model trains on every stage. s_active and n_active are not combined
(asserted). Length 3*(s_active*P + 1): 27, 51 and 99 at s_active 2, 4 and 8 (CHECK 96).

CURRICULUM (task_for(4, 8)'s vocabulary; P=4, S=8, n_vals=16, n_q=1)
- Stage 1: s_active=2, steps 1-4800. Stage 2: s_active=4, steps 4801-9600. Stage 3: all 8 streams
  (task_for(4, 8) itself), up to 19200 more steps (28800 in total).
- One optimizer throughout (Adam state carried), lr 1e-3.
- Evaluation every 1200 steps on the current stage's held-out set (eval_batch of the stage task)
  AND on the 8-stream set (eval_batch(task_for(4, 8)), test_stream_recipe's). In stage 3 the two
  are the same set.
- BOUND = the transition on the 8-stream curve, counted in stage 3 only. The early-stop streak
  counts only evaluations after step 9600 (onset_run's early_stop_after).
- Run path: test_stream_recipe's (test_router_reliability.run_one on task_for(4, 8): its probe,
  test_scale_axes's scale_stats, conv_grad_norms, lr 1e-3), with each stage's batches through
  onset_run's task_at (test_curriculum_confirm's run_cur16 mechanism). The per-stream 8-stream
  held-out accuracy is also recorded at steps 4800 and 9600 (it is always recorded at the end),
  so the 8-stream map, one_to_one, shared, eff_ch and per-stream accuracy exist at 4800, 9600 and
  the end.
- D8 runs the same path with all 8 streams from step 1 and the usual early stop (no curriculum).

ARMS (fresh seeds, disjoint from every earlier seed; models built as test_stream_recipe's, so
the same seed gives the same initial parameters in every arm with the same gate)
- SC8_ceil: perfect gate k=16 (stream s -> channel s; test_stream_recipe's ceiling8k16),
            curriculum. Validity; runs first.                            Seeds 260-262 (3)
- SC8:      arm A k=16 + conv (test_stream_recipe's A8k16_R without the restart rule),
            curriculum.                                                  Seeds 260-279 (20)
- SC8_R:    SC8 with a restart rule in stage 1: continue if the 2-stream held-out accuracy at
            step 2400 is >= 0.6 (test_router_reliability.make_decide(2400, 0.6)); else restart with
            seed s + 1000*j (j = 1..4), at most 5 attempts; the last attempt continues. Attempt 0
            is SC8's run with the same seed: its record is reused when it passes; otherwise the
            trial runs attempts 1-4 (SC8's run still continues as SC8).    Seeds 260-279 (20)
- D8:       arm A k=16 + conv, all 8 streams from step 1, up to 28800 steps.
                                                                         Seeds 260-269 (10)
If SC8_ceil binds fewer than 2/3, every claim is UNTESTED and nothing else runs.
Drop rule, applied to the printed worst-case projection (every continued run to 28800, every
trial needing all its attempts), in this order:
- above 12 h on the workers: D8 is cut to seeds 260-265;
- still above: SC8 and SC8_R are cut to seeds 260-275.
What was cut is printed. The decision is stored in the results file, and a resumed run keeps it.
Bands keep their fractions if SC8_R is cut: RELIABLE >= 0.9 of the seeds (rounded up), MAJORITY
>= 0.5, MINORITY >= 1, NEVER 0 (18 and 10 of 20; 15 and 8 of 16).

DEFINITIONS (test_stream_recipe's)
- Stream -> channel map (argmax of each stream's mean read gate at value positions on the
  8-stream probe), one_to_one, shared (streams on the most crowded channel), eff_ch, margin.
- ROUTED: a one-to-one 8-stream map AND every stream's 8-stream held-out accuracy >= 0.9.
- MERGED: shared >= 2. Channels used: >= 5% of any stream's mean read gate.
- The check passes: the 2-stream held-out accuracy at step 2400 is >= 0.6.

CLAIMS (outcome BOUND on the 8-stream set)
- VALID: SC8_ceil binds >= 2/3, and the k=16 gate fits the perfect 8-way routing (CHECK 45's
  recipe, test_router_reliability.gate_fit_k: argmax >= 0.99; CHECK 98).
- C1 THE CURRICULUM MAKES EIGHT STREAMS LEARNABLE: SC8 vs D8 on D8's seeds (260-269), exact
  McNemar one-sided (SC8 higher); Fisher on all SC8 vs all D8 printed alongside.
- C2 RELIABLE WITH RESTARTS: SC8_R's band: RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9,
  NEVER 0.
- C3 ROUTING TRANSFERS ACROSS STREAM COUNT: over the distinct SC8 and SC8_R runs (a reused
  attempt 0 counts once; abandoned attempts are not runs), runs ROUTED on the 8-stream probe and
  set at the end of stage 1 (step 4800) bind more often than runs not routed there: Fisher
  one-sided. UNTESTABLE if either group has fewer than 3 runs.
- C1 and C3 are SHOWN if p < 0.05, else NOT SHOWN. If not VALID, every claim is UNTESTED.
- Readings, printed verbatim where they apply:
  - C1 SHOWN and C2 RELIABLE: "a stream curriculum plus an early check makes eight streams
    reliable at this size."
  - C1 SHOWN and C2 not RELIABLE: "the stream curriculum makes eight streams learnable, but not
    reliably."
  - C1 NOT SHOWN: "the stream curriculum does not rescue eight streams at this size."
Fisher is test_router_confirm's; the one-sided McNemar is test_conv_lr's.

DIAGNOSTICS, not part of the verdict
- Per seed: the outcomes of SC8, SC8_R and D8 with transitions; the attempts used and each
  attempt's accuracy at 2400; at 4800, 9600 and the end, the 8-stream map, one_to_one, shared,
  eff_ch and per-stream accuracy.
- Transfer: the 8-stream accuracy at 4800 and at 9600, before any 8-stream training.
- Merges by how many streams share; non-stream failures (POSITION, KEY, OTHER); collapsed.
- Check precision in stage 1: passes, passes that bound, binders missed; for reference, the same
  threshold on the 4-stream set at 7200.
- Channels used and eff_ch by outcome; transitions and plateaus in stage 3; gradient norms at
  steps 1, 10, 100, 4801 and 9601.

CHECKS (printed before any training): test_stream_recipe's verification (which runs every
earlier test's), then
  95 s_active=None (implicit and explicit) and s_active=S leave every earlier task bit-identical
     to 18d5ae1's make_batch (attributes, batches and generator states); every arm dict equals
     its source arm's except the named fields; every arm's optimizer runs at lr 1e-3 at every
     step of every stage (step pre-hook, through run_job; for SC8_R through a trial);
  96 s_active=2 and 4 of S=8 over 10,000 sequences: exactly s_active distinct streams per
     sequence; each stream used with frequency s_active/8 +- 0.02; every active (stream, key) pair
     once, grouped, with s_active distinct values per key; the query answered by its own stream's
     value; token ids within task_for(4, 8)'s vocabulary; lengths 27 and 51;
  97 the schedule (SC8 seed 1, the real schedule run 9602 steps in a worker, hooks on every step):
     training inputs of width 26, 50 and 98 switching exactly after steps 4800 and 9600; 2, 4 and
     8 active streams per training sequence; one optimizer, Adam's step count carried over; no
     early stop before stage 3 with every held-out accuracy stubbed to 1.0 (and without
     early_stop_after the same run would stop in stage 1);
  98 the perfect gate (SC8_ceil's model) zeroes every cross-stream score at every layer with 2, 4
     and 8 active streams, channels 8-15 at 0; the 8-stream margin is 1 on the perfect gate and 0
     on the uniform gate; the recurrent k=16 gate's fit to the perfect 8-way routing (recorded for
     VALID, not asserted);
  99 the restart rule reads only the stage-1 held-out accuracy at 2400 (make_decide's closure);
     attempt seeds s + 1000*j are disjoint from every earlier seed, test_stream_recipe's
     included; attempt 0's record equals SC8's when it passes; on synthetic curves (tbo.evaluate
     stubbed) a trial continues and restarts as specified, including "none passes -> the last
     attempt continues", and an SC8 run that fails the check continues;
  100 a worker's run is bit-identical to the same run in the main process (SC8_R through one
      restart and the first switch, 6000 steps; decisions forced: attempt 0 fails, attempt 1
      passes).

OUTPUT AND LOGISTICS. Parallel single-thread workers; the wall clock is projected before training
(worst case). Results persist atomically to stream_curriculum_results.json (gitignored; copied into
results/X/ after the run). Flags: --workers, --force, --results, --also (a second machine's file,
for pooled counts).

RESULT (full run, 41/41 records complete, no failures; torch 2.14.0, Intel Xeon @ 2.80GHz, 4 workers
x 1 thread; machine X after its container moved to a new host (earlier X results were written on
the 2.10GHz part; the inherited pairing CHECKs 72 and 85 reproduce their recorded runs exactly
and fail only on that CPU label); commit ffdf0aa; 360.7 min after verification; no --also file.)
  Drop rule: at the first start the projection was 14.44 h, so D8 was cut to seeds 260-265 (13.29 h)
  and then SC8 and SC8_R to 260-275 (11.21 h). That start was killed by a container restart during
  the SC8_ceil runs (nothing recorded); the resumed start printed 12.18 h for the uncut design and
  kept the recorded decision, as designed.
  VALID: SC8_ceil bound 3/3 (all at 10800, the first stage-3 evaluation); the k=16 gate fits the
  perfect 8-way routing (argmax 1.0000, mse 2.9e-06).
  C1 THE CURRICULUM MAKES EIGHT STREAMS LEARNABLE: NOT SHOWN. On seeds 260-265, SC8 3/6 vs D8 0/6;
     SC8 only 3, D8 only 0; one-sided McNemar p = 0.125 (Fisher alongside, all SC8 6/16 vs D8 0/6:
     p = 0.107).
  C2 RELIABLE WITH RESTARTS: MINORITY. SC8_R bound 7/16 (26 attempts, 10 of them the reused SC8 run).
  C3 ROUTING TRANSFERS ACROSS STREAM COUNT: UNTESTABLE. None of the 22 distinct SC8 / SC8_R runs was
     ROUTED on the 8-stream set at step 4800 (0 vs 22 runs; 7 of the 22 bound).
  The pre-registered reading: C1 NOT SHOWN: "the stream curriculum does not rescue eight streams at
  this size."

  Diagnostics (not part of the verdict):
  - Descriptively the curriculum is not inert: D8 bound 0/6, every run at chance (0.08-0.10 at the
    end; all streams on one channel in five runs, on three in one), while SC8 bound 6/16 and SC8_R 7/16. With D8 cut to 6
    seeds the most C1 could reach with the observed 3 vs 0 split was p = 0.125.
  - Transfer: no curriculum run was routed at 4800 (8-stream accuracy at 4800 median 0.21, range
    0.10-0.44). Routing appeared in stage 2: at 9600 the eventual binders were at 0.68-0.95 (median
    0.77) and the rest at 0.11-0.80 (SC8 median 0.33). Every binder bound at 10800 or 12000, the first or
    second stage-3 evaluation; no run bound later in stage 3.
  - The stage-1 check (0.6 on the 2-stream set at 2400) missed no binder: SC8 passed 10/16, 6 of
    which bound; the 6 that failed it never bound (four POSITION collapses near 0.12, two merges).
    Restarts replaced the collapses: in SC8_R all four of those seeds passed a later attempt, but only
    s266 then bound; s263, s268 and s273 ended as merges. Passing runs that failed were all merges
    (2 or 3 streams sharing, at 0.48-0.89). For reference, 0.6 on the 4-stream set at 7200 passed 11,
    6 bound, missed 0.
  - Three binders ended without a one-to-one map, every stream at 1.00 (s260: two pairs sharing,
    s269: six streams on channel 9, s270: four on each of two channels).
  - The switch to 4 streams hits the gate hard: gate gradient norm at step 4801 median 3.8 (step 100:
    9e-7); at 9601 median 0.97.
"""

import argparse
import inspect
import itertools
import json
import math
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
from test_binding_onset import time_per_step, fmt_step, EVAL_EVERY, transition
from test_multilayer_binding import probe_batch
import test_multilayer_binding as tmb
import test_binding_capacity as tbc
from test_binding_capacity import BindTask
import test_binding_recipe as trec
import test_channel_binding as tcb
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater
from test_router_curriculum import (
    ARCH, SUB_LR, get, load_store, init_worker, makespan, cpu_model, save_results, TIME_STEPS,
)
from test_router_discovery import load_at, TASK
import test_router_reliability as trr
from test_router_reliability import run_one, strip_all, med, gate_fit_k
from test_router_layout import routing_stats
from test_readout_path import GRAD_STEPS
from test_short_conv import conv_grad_norms
from test_conv_lr import bound_r, mcnemar_greater
import test_load_curriculum as tlc
import test_curriculum_confirm as tcc
from test_scale_axes import scale_stats, routing_k, stream_acc
from test_stream_channels import shared_max, merged, used, med_int, mstr, fmt, at
import test_stream_recipe as tsr
from test_stream_recipe import outcome, routed, apath, chk

# ── Settings (fixed before any result) ───────────────────────────────────────
LR = SUB_LR                                # 1e-3, passed explicitly, every stage
T1, T2, TOTAL = 4800, 9600, 28800          # stage switches and the total budget
T_CHECK, A_CHECK = 2400, 0.6               # the restart rule's check, on the stage-1 set
R_MAX, STRIDE = 5, 1000
REAL = dict(t1=T1, t2=T2, total=TOTAL, eval_every=EVAL_EVERY, t_check=T_CHECK, a_check=A_CHECK)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
CEIL_MIN = 2                               # VALID needs SC8_ceil >= 2/3
FIT_MIN = 0.99
ALPHA = 0.05
C3_MIN = 3                                 # C3 is UNTESTABLE below 3 runs in either group
RELIABLE_F, MAJORITY_F = 0.9, 0.5          # bands, as fractions of the seeds (18 and 10 of 20)
REF_STEP = 7200                            # the check's threshold on the 4-stream set, for reference
PLATEAUS = ((0.20, 0.30), (0.45, 0.55), (0.70, 0.80))
DROP_H = 12.0
LEGACY_SHA = "18d5ae1"                     # head before this test's changes
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "stream_curriculum_results.json"

TASK48 = tsr.TASK48                        # task_for(4, 8): S=8, P=4, n_vals=16, n_q=1, length 99
STAGE = {2: BindTask(4, S=8, n_vals=tcb.N_VALS, n_q=tcb.N_Q, s_active=2),
         4: BindTask(4, S=8, n_vals=tcb.N_VALS, n_q=tcb.N_Q, s_active=4),
         8: TASK48}
A8 = {k: v for k, v in tsr.ARM["A8k16_R"].items() if k != "restart"}
ARMS = [
    dict(tsr.ARM["ceiling8k16"], key="SC8_ceil", cur=True, label="SC8_ceil  perfect gate k=16 + conv, curriculum"),
    dict(A8, key="SC8", cur=True, label="SC8       arm A k=16 + conv, curriculum"),
    dict(A8, key="SC8_R", cur=True, restart=True, label="SC8_R     SC8 + restart rule in stage 1"),
    dict(A8, key="D8", cur=False, label="D8        arm A k=16 + conv, 8 streams from step 1"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
PLAIN = {"SC8_R": "SC8"}
SEEDS = dict(SC8_ceil=tuple(range(260, 263)), SC8=tuple(range(260, 280)), SC8_R=tuple(range(260, 280)),
             D8=tuple(range(260, 270)))
CUTS = (("D8", tuple(range(260, 266))), ("SC8 and SC8_R", tuple(range(260, 276))))
READINGS = {
    "C1+ C2+": "a stream curriculum plus an early check makes eight streams reliable at this size.",
    "C1+ C2-": "the stream curriculum makes eight streams learnable, but not reliably.",
    "C1-": "the stream curriculum does not rescue eight streams at this size.",
}


def band_n(c, n):
    return ("RELIABLE" if c >= math.ceil(RELIABLE_F * n) else "MAJORITY" if c >= math.ceil(MAJORITY_F * n)
            else "MINORITY" if c >= 1 else "NEVER")


def attempt_seeds(s):
    return [s + STRIDE * j for j in range(R_MAX)]


def earlier_seeds():
    """test_stream_recipe's earlier seeds, plus its own base seeds (240-259) and attempt seeds."""
    out = tsr.earlier_seeds() | {x for v in tsr.SEEDS.values() for x in v}
    out |= {x for s in tsr.SEEDS["A4k16_R"] + tsr.SEEDS["A8k16_R"] for x in tsr.attempt_seeds(s)}
    return out


def stage_of(sched, step):
    return 2 if step <= sched["t1"] else 4 if step <= sched["t2"] else 8


def grad_sc(sched):
    return tuple(GRAD_STEPS) + (sched["t1"] + 1, sched["t2"] + 1)


def sc_stats(at_steps):
    """test_scale_axes.scale_stats (routing margin, multichannel eta^2 and the 8-stream map at
    every evaluation, per-stream accuracy at the end), plus the per-stream 8-stream held-out
    accuracy at the steps in at_steps."""
    def stats(model, task, probe, step):
        out = scale_stats(model, task, probe, step)
        if step in at_steps:
            out["stream_acc"] = stream_acc(model, task, eval_batch(task))
        return out
    return stats


# ── Runs ─────────────────────────────────────────────────────────────────────
def run_sc(a, seed, sched, decide=None, abandon=False, keep=None):
    """One run of arm a from `seed` on test_stream_recipe's run path, on task_for(4, 8) (its held-
    out set, probe and statistics). Curriculum arms (a["cur"]) train on each stage's batches
    (task_at), count the early stop only after t2, and check() evaluates the current stage's set
    at every evaluation up to t2; at t_check it gives decide(step, stage-1 accuracy), and with
    abandon=True a failing attempt stops there (trr.Abandon). D8 (cur False) trains on all 8
    streams from step 1 with the usual early stop."""
    t1, t2, ee, tc = sched["t1"], sched["t2"], sched["eval_every"], sched["t_check"]
    cur = a["cur"]
    decide = trr.make_decide(tc, sched["a_check"]) if decide is None else decide
    sets = {n: eval_batch(STAGE[n]) for n in (2, 4)} if cur else {}
    info = dict(curve_stage=[])

    def check(m, step, data):
        if not cur:
            return
        n = stage_of(sched, step)
        if n == 8:
            return
        acc, el = tbo.evaluate(m, STAGE[n], sets[n])
        info["curve_stage"].append([step, acc, el])
        if step == tc:
            info["acc_check"] = acc
            info["passed"] = bool(decide(step, acc))
            if abandon and not info["passed"]:
                raise trr.Abandon(f"stage-1 held-out accuracy {acc:.4f} at step {step}")

    run_kw = (dict(task_at=lambda step: STAGE[stage_of(sched, step)], early_stop_after=t2) if cur else None)
    rec = run_one(a, seed, sched["total"], ee, check=check, keep=keep, task=TASK48, stats_fn=sc_stats((t1, t2)),
                  grad_fn=conv_grad_norms, lr=LR, builder=tsr.builder_for(a), grad_steps=grad_sc(sched), run_kw=run_kw)
    rec.update(curve_stage=info["curve_stage"], acc_check=info.get("acc_check"), passed=info.get("passed"))
    if rec["ok"]:
        rec["transition_full"] = rec["transition"]
        if cur:
            rec["transition"] = transition([c for c in rec["curve"] if c[0] > t2])
        rec["st1"] = tcc.stats_at_step(rec, t1)
        rec["st2"] = tcc.stats_at_step(rec, t2)
    return rec


def run_trial(a, seed, sched, start=0, decide=None):
    """A trial from attempt `start`: attempts s + 1000*j in order; a failing attempt is abandoned
    at t_check, except the last (j = R_MAX-1), which continues (test_stream_recipe.run_trial's
    loop, on this test's run path)."""
    decide = trr.make_decide(sched["t_check"], sched["a_check"]) if decide is None else decide
    ts0 = time.time()
    attempts = []
    for j in range(start, R_MAX):
        sj, last = seed + STRIDE * j, j == R_MAX - 1
        rec = run_sc(a, sj, sched, decide=decide, abandon=not last)
        cut = rec["passed"] is False and not last
        attempts.append(dict(j=j, seed=sj, acc_check=rec["acc_check"], passed=rec["passed"],
                             early=[c for c in rec["curve_stage"] if c[0] <= sched["t_check"]],
                             steps=sched["t_check"] if cut else rec.get("stopped_at")))
        if cut:
            continue
        rec["trial"] = dict(start=start, attempts=attempts, used=j + 1, continued=sj, reused=False,
                            secs=time.time() - ts0)
        return rec
    raise AssertionError("unreachable")


def run_job(sp):
    a = sp.get("arm_dict") or ARM[sp["arm"]]
    sched = sp["sched"]
    decide = tsr.Forced(sp["force"], sched["t_check"]) if sp.get("force") else None
    if sp.get("trial"):
        return run_trial(a, sp["seed"], sched, start=sp["start"], decide=decide)
    return run_sc(a, sp["seed"], sched, decide=decide)


def spec(key, seed, sched, trial=False, start=0):
    sp = dict(arm=key, seed=seed, sched=dict(sched))
    if trial:
        sp.update(trial=True, start=start)
    return sp


TINY = dict(t1=2, t2=4, total=6, eval_every=2, t_check=2, a_check=A_CHECK)


def lr_job(key):
    """CHECK 95: the optimizer lr at every step through run_job (step pre-hook), stages of 2 steps
    (6 in all); for SC8_R a trial whose attempts all fail the check at step 2 (attempts 0-3 stop
    there, the last continues to step 6)."""
    seen, n_opt = [], [0]

    def pre(opt, args, kw):
        if not getattr(opt, "_c95", False):
            opt._c95 = True
            n_opt[0] += 1
        seen.append(sorted({g["lr"] for g in opt.param_groups}))
    h = register_optimizer_step_pre_hook(pre)
    try:
        r = run_job(spec(key, 3, TINY, trial=bool(ARM[key].get("restart"))))
    finally:
        h.remove()
    return seen, n_opt[0], r["ok"], (r.get("trial") or {}).get("used")


def schedule_job(seed, total):
    """CHECK 97: SC8 on the real schedule for `total` steps with hooks on every step: the width and
    the active streams per sequence of every training input, the optimizers, and Adam's step count
    entering each step."""
    widths, nstreams, adam, opts = [], {}, {}, []
    step = [0]

    def pre(opt, args, kw):
        if not opts or opts[-1] is not opt:
            opts.append(opt)
        step[0] += 1
        st = opt.state.get(opt.param_groups[0]["params"][0], {})
        adam[step[0]] = float(st["step"]) if "step" in st else 0.0

    def fwd(mod, args):
        if type(mod).__name__ == "MultiBDH" and mod.training:
            x = args[0]
            s = step[0] + 1
            widths.append((s, x.shape[1]))
            nstreams[s] = sorted({len(set(row[row < TASK48.S].tolist())) for row in x})

    h1 = register_optimizer_step_pre_hook(pre)
    h2 = torch.nn.modules.module.register_module_forward_pre_hook(fwd)
    try:
        rec = run_sc(ARM["SC8"], seed, dict(REAL, total=total))
    finally:
        h1.remove()
        h2.remove()
    return dict(ok=rec["ok"], widths=widths, nstreams={k: nstreams[k] for k in
                (T1 - 1, T1, T1 + 1, T1 + 2, T2 - 1, T2, T2 + 1, T2 + 2) if k in nstreams},
                all_streams={n: sorted({tuple(v) for k, v in nstreams.items() if stage_of(REAL, k) == n})
                             for n in (2, 4, 8)},
                n_opt=len(opts), adam={k: adam.get(k) for k in (T1, T1 + 1, T2, T2 + 1)},
                stopped=rec.get("stopped_at"))


def time_job(key):
    """ms/step on each stage's task (test_curriculum_confirm.time_job16's recipe), each charged
    with its evaluations once per EVAL_EVERY: the stage set and the 8-stream set in stages 1 and 2,
    the 8-stream set alone in stage 3."""
    a = ARM[key]
    mk = tsr.builder_for(a)(a, 0)
    out = {}
    for n in ((2, 4, 8) if a["cur"] else (8,)):
        task = STAGE[n]
        per = time_per_step(task, mk, steps=TIME_STEPS)
        m, data = mk(), eval_batch(task)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        ev = time.time() - t0
        out[n] = (max(per - ev / TIME_STEPS, 1e-6), ev)
    ev8 = out[8][1]
    return {n: out[n][0] + ((out[n][1] if n != 8 else 0.0) + ev8) / EVAL_EVERY for n in out}


# ── Verification ─────────────────────────────────────────────────────────────
def earlier_tasks():
    """Every task an earlier test built (test_load_curriculum's CHECK 69 list, plus the n_active
    tasks and the S=4 and S=8 tasks)."""
    out = ([(f"test_binding_capacity P={P}", tbc.BindTask(P)) for P in tbc.P_LIST]
           + [("test_multilayer_binding S=1", tmb.BindTask(4, S=1, n_vals=tmb.N_VALS, n_q=tmb.N_Q))]
           + [("test_binding_onset PRIMARY", tbo.PRIMARY), ("test_binding_onset SECONDARY", tbo.SECONDARY)]
           + [(f"test_binding_recipe P={P}", trec.task_for(P)) for P in (4, 8, 16)]
           + [(f"test_channel_binding P={P} S={S}", tcb.task_for(P, S)) for P in tcb.P_LIST for S in (1, 2)]
           + [("test_router_* TASK", TASK)]
           + [(f"test_router_layout {lay}", BindTask(4, S=2, n_vals=16, n_q=1, layout=lay)) for lay in tbc.LAYOUTS]
           + [("test_load_curriculum TASK1 (n_active=4 of 8)", tlc.TASK1)]
           + [(f"test_curriculum_confirm stage n_active={n}", t) for n, t in tcc.STAGES16.items()]
           + [("task_for(4, 4)", tcb.task_for(4, 4)), ("task_for(4, 8)", tcb.task_for(4, 8))])
    return out


def gen_check(t, N=10_000, gseed=96):
    """CHECK 96's checks for an s_active task; prints its lines and returns whether all hold."""
    S, P, Z = t.S, t.P, t.s_active
    tok, qs = t.make_batch(N, torch.Generator().manual_seed(gseed))
    n = Z * P
    j = torch.arange(n)
    ctx, key, val = tok[:, 3 * j], tok[:, 3 * j + 1] - S, tok[:, 3 * j + 2]
    pres = torch.zeros(N, S).scatter_(1, ctx, 1.0)
    exact = bool((pres.sum(1) == Z).all())
    freq = pres.mean(0)
    freq_ok = bool(((freq - Z / S).abs() <= 0.02).all())
    once = bool((torch.sort(ctx * P + key, 1).values.diff(dim=1) != 0).all())
    kv, cv = key.view(N, P, Z), ctx.view(N, P, Z)
    act = torch.sort(pres.nonzero()[:, 1].view(N, Z), -1).values
    grouped = bool((kv == kv[..., :1]).all()) and bool((torch.sort(cv, -1).values == act.unsqueeze(1)).all())
    distinct = bool((torch.sort(val.view(N, P, Z), -1).values.diff(dim=-1) != 0).all())
    qc, qk, qv = tok[:, 3 * n], tok[:, 3 * n + 1] - S, tok[:, 3 * n + 2]
    hit = (ctx == qc[:, None]) & (key == qk[:, None])
    answered = (bool((hit.sum(1) == 1).all()) and bool((val[hit] == qv).all())
                and bool((qc == qs.reshape(-1)).all()))
    voc = t.vocab == TASK48.vocab and bool((tok >= 0).all()) and bool((tok < TASK48.vocab).all())
    length = t.L == 3 * (Z * P + 1) and tok.shape[1] == t.L and t.qpos == [3 * Z * P + 1]
    good = exact and freq_ok and once and grouped and distinct and answered and voc and length
    print(f"     s_active={Z} of S={S}, over {N:,} sequences (generator seed {gseed}):")
    print(f"         exactly {Z} distinct streams in every sequence: {exact}   stream frequencies "
          f"{[round(x, 3) for x in freq.tolist()]} ({Z / S:g} +- 0.02): {freq_ok}")
    print(f"         every active (stream, key) pair once: {once}   grouped (each key's {Z} triples adjacent, one per active "
          f"stream): {grouped}   {Z} distinct values per key: {distinct}")
    print(f"         the query answered by its own stream's value: {answered}   token ids in [0, {TASK48.vocab}) "
          f"(task_for(4, 8)'s vocabulary): {voc}   length {t.L}, query position {t.qpos}: {length}")
    print(f"         e.g. {tlc.tps_decode(tok[0], t)}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    return good


def verify(pool):
    print("test_stream_recipe.py's verification (which runs test_stream_channels', and so on down to")
    print("test_multilayer_binding's):")
    tsr.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f95 = {k: pool.submit(lr_job, k) for k in KEYS}
    f97 = pool.submit(schedule_job, 1, T2 + 2)
    uni = dict(ARM["SC8"], key="uniform8k16", gate="uniform")
    f100 = pool.submit(run_job, dict(spec("SC8_R", 1, dict(REAL, total=6000), trial=True), force=[False, True]))

    print(f"CHECK 95 s_active=None is inert, the arms are the source arms, and every arm runs at lr {LR} in every stage:")
    leg = load_at(LEGACY_SHA, "test_binding_capacity.py", "test_binding_capacity_legacy")
    good = True
    tasks = earlier_tasks()
    for name, t in tasks:
        na = t.n_active
        L = leg.BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q, layout=t.layout, n_active=na)
        L.key = t.key
        kw = [dict(s_active=None)] + ([dict(s_active=t.S)] if na is None else [])
        variants = [BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q, layout=t.layout, n_active=na, **k_) for k_ in kw]
        for v_ in variants:
            v_.key = t.key
        attrs = all(vars(x) == vars(L) for x in [t] + variants) and t.s_active is None
        same = True
        for gs in (0, 1):
            outs = []
            for task in [t, L] + variants:
                g = torch.Generator().manual_seed(gs)
                tok, qs = task.make_batch(64, g)
                outs.append((tok, qs, g.get_state()))
            same &= all(torch.equal(outs[0][i], o[i]) for o in outs[1:] for i in range(3))
        good &= attrs and same
        if not (attrs and same):
            print(f"         {name}: attributes equal {attrs}   batches and generator states equal {same}  -> DIFFERS")
    try:
        BindTask(8, S=2, n_vals=16, n_q=1, n_active=4, s_active=1)
        both = False
    except AssertionError:
        both = True
    good &= both
    print(f"     (a) {len(tasks)} earlier tasks (S=1, 2, 4 and 8; n_q=1 and 4; P=4-32; all layouts; n_active tasks): "
          f"attributes, batches and generator")
    print(f"         states with s_active=None (implicit and explicit) and s_active=S equal {LEGACY_SHA}'s: {good}   "
          f"s_active with n_active refused: {both}")
    src = [("SC8_ceil", tsr.ARM["ceiling8k16"], "test_stream_recipe's ceiling8k16", ("key", "label", "cur")),
           ("SC8", tsr.ARM["A8k16_R"], "test_stream_recipe's A8k16_R", ("key", "label", "cur", "restart")),
           ("SC8_R", ARM["SC8"], "SC8", ("key", "label", "restart")),
           ("D8", ARM["SC8"], "SC8", ("key", "label", "cur"))]
    for k, ref, name, drop in src:
        same = ({x: v for x, v in ARM[k].items() if x not in drop} == {x: v for x, v in ref.items() if x not in drop})
        good &= same
        print(f"     (b) {k:<8} = {name:<32} except {', '.join(drop)}: {same}")
    for k in KEYS:
        seen, n_opt, rok, used_ = f95[k].result()
        lrs = sorted({tuple(x) for x in seen})
        if ARM[k].get("restart"):
            want_steps = (R_MAX - 1) * TINY["t_check"] + TINY["total"]
            g = lrs == [(LR,)] and rok and n_opt == R_MAX and len(seen) == want_steps and used_ == R_MAX
            how = f"through a trial ({n_opt} optimizers, {len(seen)} steps, attempts used {used_})"
        else:
            g = lrs == [(LR,)] and rok and len(seen) == TINY["total"] and n_opt == 1
            how = f"over {len(seen)} steps, {n_opt} optimizer"
        good &= g
        stg = "stages of 2 steps: 2, 4, 8 streams" if ARM[k]["cur"] else "all 8 streams"
        print(f"     (c) {k:<8} optimizer lr at every step {[list(x) for x in lrs]} {how} ({stg})  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 96 the stage tasks:")
    good = gen_check(STAGE[2]) and gen_check(STAGE[4], gseed=97)
    same_voc = all(STAGE[n].vocab == TASK48.vocab and STAGE[n].ctx_tokens == TASK48.ctx_tokens and STAGE[n].P == 4
                   for n in (2, 4))
    good &= same_voc
    print(f"     the stage tasks share task_for(4, 8)'s P, S, vocabulary and context tokens: {same_voc}   stage 3 is "
          f"task_for(4, 8) itself: {STAGE[8] is TASK48}")
    ok &= good
    print()

    print(f"CHECK 97 the schedule (SC8 seed 1, the real schedule run {T2 + 2} steps in a worker; hooks on every step):")
    sj = f97.result()
    w = sj["widths"]
    wmap = {n: sorted({x for s, x in w if stage_of(REAL, s) == n}) for n in (2, 4, 8)}
    g1 = wmap == {2: [STAGE[2].T], 4: [STAGE[4].T], 8: [TASK48.T]} and len(w) == T2 + 2
    g2 = (sj["all_streams"] == {2: [(2,)], 4: [(4,)], 8: [(8,)]}
          and [sj["nstreams"][k] for k in sorted(sj["nstreams"])] == [[2], [2], [4], [4], [4], [4], [8], [8]])
    g3 = sj["n_opt"] == 1 and sj["adam"] == {T1: T1 - 1.0, T1 + 1: float(T1), T2: T2 - 1.0, T2 + 1: float(T2)} and sj["ok"]
    print(f"         training input widths at steps 1-{T1}: {wmap[2]}, {T1 + 1}-{T2}: {wmap[4]}, after: {wmap[8]}   "
          f"({len(w)} training forwards): {g1}")
    print(f"         active streams per training sequence in each stage {sj['all_streams']}; at steps "
          f"{', '.join(str(k) for k in sorted(sj['nstreams']))}: {[sj['nstreams'][k] for k in sorted(sj['nstreams'])]}: {g2}")
    print(f"         one optimizer throughout: {sj['n_opt'] == 1}   Adam's step count entering steps {T1}, {T1 + 1}, {T2}, "
          f"{T2 + 1}: {', '.join(str(sj['adam'][k]) for k in (T1, T1 + 1, T2, T2 + 1))} (carried over)   run ok "
          f"{sj['ok']}: {g3}")
    real_eval = tbo.evaluate
    tiny = dict(t1=40, t2=80, total=200, eval_every=10, t_check=20, a_check=A_CHECK)
    tbo.evaluate = lambda model, task, d: (1.0, 0.0)
    try:
        r_es = run_sc(ARM["SC8"], 3, tiny)
        r_no = run_one(ARM["SC8"], 3, 200, 10, task=TASK48, stats_fn=scale_stats, grad_fn=conv_grad_norms, lr=LR,
                       builder=tsr.builder_for(ARM["SC8"]),
                       run_kw=dict(task_at=lambda step: STAGE[stage_of(tiny, step)]))
    finally:
        tbo.evaluate = real_eval
    g4 = r_es["ok"] and r_es["stopped_at"] == 110 and r_es["transition"] == 90 and r_no["stopped_at"] == 30
    print(f"         early stop, every held-out accuracy stubbed to 1.0 (stages switch at 40 and 80, evaluations every 10): "
          f"the curriculum stops at step {r_es['stopped_at']} (stage 3's third evaluation), transition {r_es['transition']};")
    print(f"         without early_stop_after the same run would stop at step {r_no['stopped_at']}, in stage 1: {g4}")
    good = g1 and g2 and g3 and g4
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 98 the perfect gate with 2, 4 and 8 active streams, and the 8-stream margin:")
    good = True
    for n in (2, 4, 8):
        maxc, live, gr = tsr.cross_scores(ARM["SC8_ceil"], STAGE[n], 980 + n)
        unused = gr.shape[-1] == 16 and bool((gr[..., TASK48.S:] == 0).all())
        g = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0 and unused
        good &= g
        print(f"     (a) {n} active streams, random conv weights: layers seen {len(maxc)} (n_layer={ARCH['n_layer']}); read "
              f"gate {tuple(gr.shape)}, channels 8-15 exactly 0: {unused}; max |cross-stream score| per layer "
              f"{[f'{c:.2e}' for c in maxc]}; max |same-stream score| {[f'{l_:.2e}' for l_ in live]}")
    probe = probe_batch(TASK48, 3)
    ap = ARM["SC8_ceil"]
    mp_, mu_ = tsr.builder_for(ap)(ap, 3)(), tsr.builder_for(uni)(uni, 3)()
    rp_, ru_ = routing_stats(mp_, TASK48, probe)["margin"], routing_stats(mu_, TASK48, probe)["margin"]
    rk = routing_k(mp_, TASK48, probe)
    g = abs(rp_ - 1.0) < 1e-6 and abs(ru_) < 1e-6 and rk["ch_map"] == list(range(8)) and rk["one_to_one"]
    good &= g
    print(f"     (b) 8-stream margin on the perfect gate {rp_:.6f} (want 1), on the uniform gate {ru_:.6f} (want 0); perfect "
          f"gate's map {rk['ch_map']} one-to-one {rk['one_to_one']}: {g}")
    mse, am = gate_fit_k(TASK48, ARM["SC8"])
    fits = dict(SC8=dict(mse=mse, argmax=am))
    print(f"     (c) SC8's recurrent k=16 gate fit to the perfect 8-way routing (test_router_reliability.gate_fit_k): mse "
          f"{mse:.2e}   argmax match {am:.4f}  -> {'FITS' if am >= FIT_MIN else 'DOES NOT FIT'} (>= {FIT_MIN}; recorded "
          f"for VALID, not asserted)")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 99 the restart rule:")
    dec = trr.make_decide(T_CHECK, A_CHECK)
    params = list(inspect.signature(dec).parameters)
    cv = inspect.getclosurevars(dec)
    good_a = (params == ["step", "acc"] and not cv.globals and not cv.builtins
              and cv.nonlocals == {"t_check": T_CHECK, "a_check": A_CHECK})
    print(f"     (a) decide = test_router_reliability.make_decide({T_CHECK}, {A_CHECK}), given the stage-1 held-out accuracy: "
          f"parameters {params}; it reads no global name ({dict(cv.globals)}) and only the constants {dict(cv.nonlocals)}: "
          f"{good_a}")
    earlier = earlier_seeds()
    base = sorted({s for k in KEYS for s in SEEDS[k]})
    new = [x for s in SEEDS["SC8_R"] for x in attempt_seeds(s)[1:]]
    good_b = (len(set(new)) == len(new) == len(SEEDS["SC8_R"]) * (R_MAX - 1) and not (set(new) | set(base)) & earlier
              and all(x == s + STRIDE * j for s in SEEDS["SC8_R"] for j, x in enumerate(attempt_seeds(s))))
    print(f"     (b) base seeds {tsr.ranges(base)}; attempt seeds s + {STRIDE}*j (j = 1..{R_MAX - 1}): {tsr.ranges(new)}; "
          f"{len(set(new))} distinct, all disjoint from every earlier seed: {good_b}")
    print(f"         (earlier seeds: {tsr.ranges(earlier)})")
    mid = dict(t1=40, t2=80, total=100, eval_every=20, t_check=20, a_check=0.0)
    tr0 = run_trial(ARM["SC8_R"], 3, mid, start=0)
    rs = run_sc(ARM["SC8"], 3, mid)
    body0 = {k: v for k, v in tr0.items() if k != "trial"}
    good_c = strip_all(body0) == strip_all(rs) and tr0["trial"]["used"] == 1 and rs["ok"]
    print(f"     (c) attempt 0 passing (a_check 0, stages switch at 40 and 80 of 100, check at 20): its record == SC8's run of "
          f"seed 3: {strip_all(body0) == strip_all(rs)}  -> {'OK' if good_c else 'WRONG'}")
    tiny = dict(t1=8, t2=16, total=24, eval_every=2, t_check=4, a_check=A_CHECK)
    cnt = {"opt": None, "n": 0, "k": -1, "steps": []}

    def pre(opt, args, kw):
        if cnt["opt"] is not opt:
            cnt["opt"], cnt["n"], cnt["k"] = opt, 0, cnt["k"] + 1
            cnt["steps"].append(0)
        cnt["n"] += 1
        cnt["steps"][-1] += 1

    def run_synth(script, fn):
        cnt.update(opt=None, n=0, k=-1, steps=[])
        tbo.evaluate = lambda model, task, d: (script(cnt["k"], cnt["n"]), 0.0)
        h = register_optimizer_step_pre_hook(pre)
        try:
            return fn(), list(cnt["steps"])
        finally:
            h.remove()
            tbo.evaluate = real_eval
    cases = [("passes at once", lambda j, st: 0.65 if st == 4 else 0.3,
              dict(calls=[(4, 0.65)], used=1, continued=3, steps=[24])),
             ("restarts, then passes", lambda j, st: (0.7 if j == 2 else 0.5) if st == 4 else 0.2,
              dict(calls=[(4, 0.5), (4, 0.5), (4, 0.7)], used=3, continued=2003, steps=[4, 4, 24])),
             ("none passes", lambda j, st: 0.3,
              dict(calls=[(4, 0.3)] * R_MAX, used=R_MAX, continued=4003, steps=[4, 4, 4, 4, 24]))]
    good_d = True
    for name, script, want in cases:
        calls, rule = [], trr.make_decide(tiny["t_check"], A_CHECK)

        def spy(step, acc, rule=rule, calls=calls):
            calls.append((step, acc))
            return rule(step, acc)
        tr, steps = run_synth(script, lambda: run_trial(ARM["SC8_R"], 3, tiny, start=0, decide=spy))
        t_ = tr["trial"]
        g = (calls == want["calls"] and t_["used"] == want["used"] and t_["continued"] == want["continued"]
             and steps == want["steps"] and [x["seed"] for x in t_["attempts"]] == attempt_seeds(3)[:want["used"]]
             and tr["ok"] and tr["stopped_at"] == tiny["total"] and [x["steps"] for x in t_["attempts"]] == want["steps"])
        good_d &= g
        print(f"     (d) synthetic '{name}' (check at step 4, stages switch at 8 and 16 of 24, evaluations every 2): decide saw "
              f"{calls}; attempts used {t_['used']}, seeds {[x['seed'] for x in t_['attempts']]}, steps per attempt {steps}, "
              f"continued {t_['continued']}  -> {'OK' if g else 'WRONG'}")
    rp, steps = run_synth(lambda j, st: 0.3, lambda: run_sc(ARM["SC8"], 3, tiny))
    g_e = rp["ok"] and rp["passed"] is False and rp["stopped_at"] == tiny["total"] and steps == [tiny["total"]]
    print(f"     (e) an SC8 run failing the check (synthetic 'none passes'): passed {rp['passed']}, continued to step "
          f"{rp['stopped_at']} of {tiny['total']}  -> {'OK' if g_e else 'WRONG'}")
    good = good_a and good_b and good_c and good_d and g_e
    ok &= good
    print()

    print("CHECK 100 a worker's run is bit-identical to the same run here:")
    rw = f100.result()
    rh = run_job(dict(spec("SC8_R", 1, dict(REAL, total=6000), trial=True), force=[False, True]))
    t_ = rh.get("trial", {})
    g = (strip_all(rw) == strip_all(rh) and rh["ok"] and t_.get("used") == 2 and t_.get("continued") == 1001
         and rh["stopped_at"] == 6000)
    stg = [round(a_, 3) for _, a_, _ in rh.get("curve_stage", [])]
    print(f"         SC8_R seed 1, one restart (decisions forced: attempt 0 fails, attempt 1 passes) and the switch at {T1}, "
          f"6000 steps: equal {strip_all(rw) == strip_all(rh)}   attempts "
          f"{[(x['seed'], round(x['acc_check'], 4)) for x in t_.get('attempts', [])]}   continued {t_.get('continued')} to "
          f"step {rh.get('stopped_at')}   stage acc {stg}   8-stream acc {[round(a_, 3) for _, a_, _ in rh.get('curve', [])]}"
          f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return fits


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds) if (r := get(store, key, s)) and r.get("ok")]


def routed_st(st):
    """ROUTED at an intermediate step: stats at that step with the per-stream 8-stream accuracy."""
    return bool(st and "stream_acc" in st and routed({"end": st}))


def shared_st(st):
    cm = st["ch_map"]
    return max(cm.count(c) for c in cm)


def sc_at(curve, step):
    return at(curve, step)


def distinct_runs(store):
    """C3's runs: every distinct SC8 / SC8_R run (continued attempt), by attempt seed -> record."""
    out = {}
    for s in SEEDS["SC8"]:
        r = get(store, "SC8", s)
        if r and r.get("ok"):
            out[s] = r
    for s in SEEDS["SC8_R"]:
        r = get(store, "SC8_R", s)
        if r and r.get("ok") and r["trial"]["continued"] not in out:
            out[r["trial"]["continued"]] = r
    return out


def raw_rows(store, key, seeds, sched):
    tc, t1, t2 = sched["t_check"], sched["t1"], sched["t2"]
    trial = bool(ARM[key].get("restart"))
    print(f"  {'arm':<8} {'seed':>4} {'acc':>7} {'transition':>10} {'stg@' + str(tc):>8} {'chk':>4} {'8@' + str(t1):>7} "
          f"{'8@' + str(t2):>7} {'m end':>6} {'eff_ch':>6} {'map':>16} {'1:1':>3} {'sh':>2} {'used':>4}  per-stream acc"
          f"{'':<27} outcome" + ("                 attempts (seed:stage acc@check)" if trial else ""))
    for s in seeds:
        r = get(store, key, s)
        if r is None:
            print(f"  {key:<8} {s:>4}  NOT RUN")
            continue
        if not r.get("ok"):
            print(f"  {key:<8} {s:>4}  FAILED — {r.get('error')}" + (f"   attempts {apath(r)}" if trial else ""))
            continue
        e = r["end"]
        extra = ""
        if trial:
            t_ = r["trial"]
            extra = f"  {t_['used']} used: {apath(r)}" + ("  (reused)" if t_["reused"] else "")
        print(f"  {key:<8} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {fmt(r.get('acc_check')):>8} "
              f"{chk(r.get('passed')):>4} {fmt(sc_at(r['curve'], t1)):>7} {fmt(sc_at(r['curve'], t2)):>7} "
              f"{fmt(e.get('margin')):>6} {e['eff_ch']:>6.2f} {mstr(e['ch_map']):>16} {'y' if e['one_to_one'] else 'n':>3} "
              f"{shared_max(r):>2} {len(used(r)):>4}  {'/'.join(f'{v:.2f}' for v in e['stream_acc']):<41} "
              f"{outcome(r) + ('  collapsed' if r['collapsed'] else ''):<22}" + extra)
    print()


def plateau3(r, lo, hi, sched):
    tr, t2, ee = r["transition"], sched["t2"], sched["eval_every"]
    return sum(ee for st, a_, _ in r["curve"] if st > t2 and (tr is None or st < tr) and lo <= a_ <= hi)


def report_stop(store, sched, wall, path):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — SC8_ceil only (nothing else was run)")
    print("#" * 100)
    raw_rows(store, "SC8_ceil", SEEDS["SC8_ceil"], sched)
    c = sum(bound(store, "SC8_ceil", s) for s in SEEDS["SC8_ceil"])
    need = min(CEIL_MIN, len(SEEDS["SC8_ceil"]))
    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    print(f"  SC8_ceil binds {c}/{len(SEEDS['SC8_ceil'])} (needs >= {need}): the validity condition FAILED; nothing else "
          f"was run")
    print("     *** UNTESTED ***")
    print()
    print("PRE-REGISTERED CLAIMS: C1, C2 and C3 UNTESTED; no reading.")
    store["verdict"] = dict(valid=False, stopped="ceiling", ceiling=c, C1="UNTESTED", C2="UNTESTED", C3="UNTESTED",
                            readings=[])
    save_results(path, store)
    print(f"\n  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}\n")


def report(store, sched, fits, drop, wall, path, also):
    tc, t1, t2, ee = sched["t_check"], sched["t1"], sched["t2"], sched["eval_every"]
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  (acc = final 8-stream held-out accuracy; transition on the 8-stream curve, stage 3 only for the curriculum; "
          f"stg@{tc} and chk = the stage-1 (2-stream) held-out accuracy at the check and the rule's decision; 8@{t1}, "
          f"8@{t2} = 8-stream accuracy at the switches; m end = routing margin; map = the 8-stream stream -> channel argmax, "
          f"stream 0 first; 1:1 = one-to-one; sh = streams on the most crowded channel; used = channels with >= 5% of any "
          f"stream's mean gate; ROUTED = one-to-one and every stream >= 0.9; for SC8_R the row is the continued attempt)")
    if drop:
        print(f"  seeds cut by the drop rule: {'; '.join(drop)}")
    for k in KEYS:
        raw_rows(store, k, SEEDS[k], sched)

    print("=" * 100)
    print("COUNTS (BOUND = 8-stream transition, stage 3 only for the curriculum; ROUTED = one-to-one and every stream >= 0.9; "
          "MERGED = shared >= 2 at the end)")
    print("=" * 100)
    c = {}
    for k in KEYS:
        rs = ok_runs(store, k)
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        extra = ""
        if ARM[k].get("restart"):
            extra = (f"   attempts used {sum(r['trial']['used'] for _, r in rs)} (reused attempt 0: "
                     f"{sum(1 for _, r in rs if r['trial']['reused'])})")
        print(f"  {ARM[k]['label']:<50} bound {c[k]:>2}/{len(SEEDS[k])}   bound and ROUTED "
              f"{sum(1 for _, r in rs if r['transition'] is not None and routed(r)):>2}   one-to-one at the end "
              f"{sum(1 for _, r in rs if r['end']['one_to_one']):>2}   MERGED {sum(1 for _, r in rs if merged(r)):>2}/{len(rs)}   "
              f"completed {len(rs)}/{len(SEEDS[k])}" + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["SC8_ceil"])
    need = min(CEIL_MIN, nc)
    fit_ok = fits["SC8"]["argmax"] >= FIT_MIN
    valid = c["SC8_ceil"] >= need and fit_ok
    print(f"  SC8_ceil binds {c['SC8_ceil']}/{nc} (needs >= {need})   the k=16 gate fits the perfect 8-way routing: argmax "
          f"{fits['SC8']['argmax']:.4f} (need >= {FIT_MIN}): {fit_ok}")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND on the 8-stream set)")
    print("#" * 100)
    v = dict(bound=c, valid=valid, fits=fits, drop=drop)
    ver = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    sd = SEEDS["D8"]
    b1 = sum(bound(store, "SC8", s) and not bound(store, "D8", s) for s in sd)
    c1 = sum(bound(store, "D8", s) and not bound(store, "SC8", s) for s in sd)
    p1 = mcnemar_greater(b1, c1)
    pf = fisher_greater(c["SC8"], len(SEEDS["SC8"]), c["D8"], len(sd))
    v.update(C1=ver(p1), C1_p=p1, C1_discordant=[b1, c1], C1_fisher=pf)
    print(f"  C1 THE CURRICULUM MAKES EIGHT STREAMS LEARNABLE (exact McNemar one-sided, seeds {sd[0]}-{sd[-1]}): SC8 "
          f"{sum(bound(store, 'SC8', s) for s in sd)}/{len(sd)} vs D8 {c['D8']}/{len(sd)}; SC8 only {b1}, D8 only {c1}; "
          f"p = {p1:.3g}   (Fisher one-sided alongside, all SC8 {c['SC8']}/{len(SEEDS['SC8'])} vs D8 {c['D8']}/{len(sd)}: "
          f"p = {pf:.3g})")
    print(f"     *** C1: {v['C1']} ***")
    n2 = len(SEEDS["SC8_R"])
    b2 = band_n(c["SC8_R"], n2)
    v["C2"] = b2 if valid else "UNTESTED"
    print(f"  C2 RELIABLE WITH RESTARTS (band: RELIABLE >= {math.ceil(RELIABLE_F * n2)}, MAJORITY {math.ceil(MAJORITY_F * n2)}-"
          f"{math.ceil(RELIABLE_F * n2) - 1}, MINORITY 1-{math.ceil(MAJORITY_F * n2) - 1}, NEVER 0, of {n2}): SC8_R "
          f"{c['SC8_R']}/{n2}")
    print(f"     *** C2: {v['C2']} ***")
    runs = distinct_runs(store)
    rt = {s: r for s, r in runs.items() if routed_st(r.get("st1"))}
    nr = {s: r for s, r in runs.items() if s not in rt}
    kr, kn = sum(1 for r in rt.values() if r["transition"] is not None), sum(1 for r in nr.values() if r["transition"] is not None)
    if len(rt) < C3_MIN or len(nr) < C3_MIN:
        p3, v3 = float("nan"), ("UNTESTABLE" if valid else "UNTESTED")
    else:
        p3 = fisher_greater(kr, len(rt), kn, len(nr))
        v3 = ver(p3)
    v.update(C3=v3, C3_p=p3, C3_groups=dict(routed=[kr, len(rt)], not_routed=[kn, len(nr)]))
    print(f"  C3 ROUTING TRANSFERS ACROSS STREAM COUNT (Fisher one-sided; {len(runs)} distinct SC8 / SC8_R runs): ROUTED on "
          f"the 8-stream probe and set at step {t1}: bound {kr}/{len(rt)}; not routed: bound {kn}/{len(nr)}; "
          + (f"p = {p3:.3g}" if p3 == p3 else f"UNTESTABLE (a group has fewer than {C3_MIN} runs)"))
    if rt:
        print(f"     routed at {t1}: " + " ".join(f"s{s}" for s in sorted(rt)))
    print(f"     *** C3: {v3} ***")
    print()
    print("  READINGS:")
    rd = []
    if valid:
        if v["C1"] == "SHOWN" and v["C2"] == "RELIABLE":
            rd.append(("C1 SHOWN and C2 RELIABLE", READINGS["C1+ C2+"]))
        if v["C1"] == "SHOWN" and v["C2"] != "RELIABLE":
            rd.append((f"C1 SHOWN and C2 {v['C2']}", READINGS["C1+ C2-"]))
        if v["C1"] == "NOT SHOWN":
            rd.append(("C1 NOT SHOWN", READINGS["C1-"]))
        for cond, text in rd:
            print(f"     {cond}: \"{text}\"")
    else:
        print("     (UNTESTED: no reading)")
    v["readings"] = [cond for cond, _ in rd]
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print(f"  per seed: SC8, SC8_R (attempts used; each attempt's stage-1 accuracy at {tc}, + passed, - failed) and D8; B = bound "
          f"(transition), . = not bound:")
    for s in SEEDS["SC8"]:
        cells = []
        for k in ("SC8", "SC8_R", "D8"):
            r = get(store, k, s)
            if s not in SEEDS[k]:
                cells.append("")
                continue
            if not r or not r.get("ok"):
                cells.append(f"{k}: --")
                continue
            head = f"{k}: {'B' if r['transition'] is not None else '.'} {fmt_step(r['transition']):>5}"
            if ARM[k].get("restart"):
                head += f" [{r['trial']['used']}: " + " ".join(
                    f"{fmt(a['acc_check'], 2)}{'+' if a['passed'] else '-'}" for a in r["trial"]["attempts"]) + "]"
            cells.append(head)
        print(f"    s{s}  {cells[0]:<16} {cells[1]:<50} {cells[2]}")
    print()
    print(f"  per seed at {t1}, {t2} and the end (8-stream probe and set): map; one-to-one; shared; eff_ch; per-stream accuracy "
          f"(ROUTED marked *):")
    for k in ("SC8", "SC8_R", "D8"):
        for s, r in ok_runs(store, k):
            parts = []
            for lab, st in ((str(t1), r.get("st1")), (str(t2), r.get("st2")), ("end", r["end"])):
                if not st:
                    parts.append(f"{lab}: --")
                    continue
                sa = st.get("stream_acc")
                parts.append(f"{lab}: [{mstr(st['ch_map'])}] {'y' if st['one_to_one'] else 'n'} sh {shared_st(st)} eff "
                             f"{st['eff_ch']:.1f} " + ("/".join(f"{x:.2f}" for x in sa) if sa else "--")
                             + ("*" if routed_st(st) else ""))
            print(f"    {k:<6} s{s}  " + "   ".join(parts))
    print()
    print(f"  transfer: 8-stream held-out accuracy at {t1} and {t2} (before any 8-stream training for the curriculum), median "
          f"[min, max]:")
    for k in KEYS:
        rs = [r for _, r in ok_runs(store, k)]
        for step in (t1, t2):
            xs = [x for r in rs if (x := sc_at(r["curve"], step)) is not None]
            print(f"    {k:<8} at {step:>5}: {med(xs)}   (bound runs: {med([x for r in rs if r['transition'] is not None and (x := sc_at(r['curve'], step)) is not None])}; "
                  f"not bound: {med([x for r in rs if r['transition'] is None and (x := sc_at(r['curve'], step)) is not None])})")
    print()
    print("  failure classes and outcomes (MERGED = STREAM-PARTIAL with streams sharing a channel; non-stream = test_scale_axes's "
          "generalized classes):")
    for k in KEYS:
        rs = ok_runs(store, k)
        tally = {}
        for s, r in rs:
            tally.setdefault(outcome(r), []).append(s)
        print(f"    {k:<8} " + "   ".join(f"{o} {len(x)}" for o, x in sorted(tally.items())))
        for o, xs in sorted(tally.items()):
            print(f"    {'':<8}   {o:<26} " + " ".join(f"s{s}" for s in xs))
        mgs = {}
        for _, r in rs:
            mgs[shared_max(r)] = mgs.get(shared_max(r), 0) + 1
        print(f"    {'':<8}   shared at the end, all runs: " + "   ".join(f"{n}: {mgs[n]}" for n in sorted(mgs))
              + f"   collapsed {sum(1 for _, r in rs if r['collapsed'])}")
    print()
    print(f"  check precision in stage 1 ({A_CHECK} on the 2-stream set at {tc}) — passes, passes that bound, binders missed:")
    rs = [r for _, r in ok_runs(store, "SC8")]
    ps = [r for r in rs if r.get("passed")]
    fs = [r for r in rs if r.get("passed") is False]
    print(f"    SC8      ({len(rs)} runs)   passed {len(ps):>2}   passed and bound {sum(1 for r in ps if r['transition'] is not None):>2}"
          f"   missed {sum(1 for r in fs if r['transition'] is not None):>2}")
    rs = [r for _, r in ok_runs(store, "SC8_R")]
    att = [a for r in rs for a in r["trial"]["attempts"]]
    cont_fail = [r for r in rs if r.get("passed") is False]
    print(f"    SC8_R    ({len(att)} attempts in {len(rs)} trials)   passed {sum(1 for a in att if a['passed']):>2}   passed and "
          f"bound {sum(1 for r in rs if r.get('passed') and r['transition'] is not None):>2}   abandoned "
          f"{sum(1 for a in att if a['passed'] is False) - len(cont_fail):>2}   last attempts continued after failing "
          f"{len(cont_fail)}, of which bound {sum(1 for r in cont_fail if r['transition'] is not None)}")
    rs = [r for _, r in ok_runs(store, "SC8")]
    ref = [r for r in rs if (x := at(r["curve_stage"], REF_STEP)) is not None]
    rp = [r for r in ref if at(r["curve_stage"], REF_STEP) >= A_CHECK]
    rf = [r for r in ref if at(r["curve_stage"], REF_STEP) < A_CHECK]
    print(f"    for reference, {A_CHECK} on the 4-stream set at {REF_STEP}, SC8 ({len(ref)} runs): passed {len(rp)}   passed and "
          f"bound {sum(1 for r in rp if r['transition'] is not None)}   missed {sum(1 for r in rf if r['transition'] is not None)}")
    print()
    print("  channels used (>= 5% of any stream's mean gate at the end), median [min, max], and eff_ch, by outcome:")
    for k in KEYS:
        rs = ok_runs(store, k)
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {k:<8} {lab:<9} ({len(sub):>2})  {med_int([len(used(r)) for r in sub]):<14}  eff_ch "
                      f"{med([r['end']['eff_ch'] for r in sub])}")
    print()
    print(f"  transitions (stage 3 for the curriculum), median [min, max]; plateaus in stage 3 before the transition (or the "
          f"whole stage), steps at 8-stream accuracy {', '.join(f'{lo:.2f}-{hi:.2f}' for lo, hi in PLATEAUS)}, median:")
    for k in KEYS:
        rs = [r for _, r in ok_runs(store, k)]
        trs = [r["transition"] for r in rs if r["transition"] is not None]
        pl = "   ".join(f"{lo:.2f}-{hi:.2f}: {med_int([plateau3(r, lo, hi, sched) for r in rs])}" for lo, hi in PLATEAUS)
        print(f"    {k:<8} " + (f"{statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]" if trs else "none bound")
              + ("" if not ARM[k]["cur"] else f"   plateaus {pl}"))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median [min, max]:")
    for step in grad_sc(sched):
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>5} {k:<8} " + "   ".join(parts))
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
                bh = sum(bound(store, k, s) for s in SEEDS[k])
                print(f"    {k:<8} bound here {bh}/{nh}   other {b2_}/{len(rs2)}   pooled {bh + b2_}/{nh + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — per evaluation (every {ee} steps), x100: stg = the current stage's held-out set (stages 1-2), 8st = the "
          f"8-stream set; '^' marks the check at {tc}, '|' the switches at {t1} and {t2}")
    print("=" * 100)
    for k in KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is not None:
                curve_lines(k, s, r, sched)
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


def curve_str(curve, sched, marks=True):
    tc, t1, t2 = sched["t_check"], sched["t1"], sched["t2"]
    mk = lambda st: (" " if not marks else "^" if st == tc else "|" if st in (t1, t2) else " ")
    return "".join(f"{mk(st)}{round(acc * 100):>3}" for st, acc, _ in curve)


def curve_lines(k, s, r, sched):
    cur = ARM[k]["cur"]
    tag = f"{k:<8} s{s}"
    if ARM[k].get("restart"):
        t_ = r.get("trial") or {}
        for a in t_.get("attempts", []):
            if a["passed"] is False and a["seed"] != t_.get("continued"):
                print(f"  {tag} a{a['j']} {a['seed']:>4} stg {curve_str(a['early'] or [], sched)}  -> abandoned "
                      f"({fmt(a['acc_check'], 4)} < {A_CHECK})")
    if not r.get("ok"):
        print(f"  {tag} FAILED — {r.get('error')}")
        return
    if ARM[k].get("restart"):
        t_ = r["trial"]
        j = next(a["j"] for a in t_["attempts"] if a["seed"] == t_["continued"])
        tag += f" a{j} {t_['continued']:>4}"
    end = f"bound @{r['transition']}" if r["transition"] else "not bound"
    note = "  (attempt 0 = the SC8 run)" if (r.get("trial") or {}).get("reused") else ""
    if cur:
        print(f"  {tag} stg {curve_str(r['curve_stage'], sched)}")
    print(f"  {'' if not cur else ' ' * len(tag)}{tag if not cur else ''} 8st {curve_str(r['curve'], sched, marks=cur)}  -> "
          f"{end}  [{mstr(r['end']['ch_map'])}]{note}")


# ── Main ─────────────────────────────────────────────────────────────────────
def describe(rec, sched):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr, e = rec["transition"], rec["end"]
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} chk "
            f"{fmt(rec.get('acc_check'))} {chk(rec.get('passed')):<4} 8@{sched['t1']} {fmt(sc_at(rec['curve'], sched['t1']))} "
            f"8@{sched['t2']} {fmt(sc_at(rec['curve'], sched['t2']))} map [{mstr(e['ch_map'])}] shared {shared_max(rec)} "
            f"eff_ch {e['eff_ch']:.2f} streams {[round(v, 2) for v in e['stream_acc']]}")


def att0(rp, s, tc):
    return dict(j=0, seed=s, acc_check=rp.get("acc_check"), passed=rp.get("passed"),
                early=[c for c in rp.get("curve_stage", []) if c[0] <= tc],
                steps=rp.get("stopped_at") if rp.get("passed") else tc)


def r_step(store, s, sched, force, record):
    """SC8_R for seed s once SC8 seed s is known: reuse SC8's record if attempt 0 passed, else a
    trial from attempt 1 (from attempt 0 if SC8's run failed)."""
    if get(store, "SC8_R", s) is not None and not force:
        print(f"   {'SC8_R':<8} seed {s}  cached")
        return None
    rp = get(store, "SC8", s)
    if rp and rp.get("ok") and rp.get("passed"):
        rec = dict(rp, trial=dict(start=0, attempts=[att0(rp, s, sched["t_check"])], used=1, continued=s, reused=True,
                                  secs=0.0))
        record(dict(arm="SC8_R", seed=s), rec)
        return None
    return spec("SC8_R", s, sched, trial=True, start=1 if (rp and rp.get("ok")) else 0)


def finish_trial(store, sp, rec):
    """Prepend attempt 0 (SC8's run) to a trial that started at attempt 1."""
    if "trial" in rec and sp["start"] == 1:
        rp = get(store, "SC8", sp["seed"])
        rec["trial"]["attempts"] = [att0(rp, sp["seed"], sp["sched"]["t_check"])] + rec["trial"]["attempts"]
    elif "trial" not in rec:
        rec["trial"] = dict(start=sp["start"], attempts=[], used=None, continued=None, reused=False, secs=0.0)
    return rec


def projection(cost, sched, workers):
    t1, t2, total, tc = sched["t1"], sched["t2"], sched["total"], sched["t_check"]
    full = {k: (t1 * cost[k][2] + (t2 - t1) * cost[k][4] + (total - t2) * cost[k][8]) if ARM[k]["cur"]
            else total * cost[k][8] for k in KEYS}
    d_ceil = [full["SC8_ceil"]] * len(SEEDS["SC8_ceil"])
    d_rest = ([full["SC8"]] * len(SEEDS["SC8"]) + [full["D8"]] * len(SEEDS["D8"])
              + [full["SC8_R"] + (R_MAX - 2) * tc * cost["SC8_R"][2]] * len(SEEDS["SC8_R"]))
    mk1, mk2 = makespan(d_ceil, workers), makespan(d_rest, workers)
    return full, d_ceil, d_rest, mk1, mk2, (mk1 + mk2) / 3600


def main():
    ap = argparse.ArgumentParser(description="A stream curriculum (2, 4, then 8 streams) for eight streams at k=16.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None, help="a second machine's stream_curriculum_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    sched = dict(SCHED)
    tc, t1, t2, total = sched["t_check"], sched["t1"], sched["t2"], sched["total"]

    print("=" * 100)
    print("Stream curriculum: 2, then 4, then 8 of 8 streams per sequence, for eight streams at k=16")
    print(f"  task_for(4, 8) vocabulary (S=8, P=4, n_vals=16); stage 1 s_active=2 (length {STAGE[2].L}) steps 1-{t1}; stage 2 "
          f"s_active=4 (length {STAGE[4].L}) steps {t1 + 1}-{t2}; stage 3 all 8 (length {TASK48.L}) up to step {total}")
    print(f"  lr {LR} (SUB_LR, explicit) in every stage, one optimizer; conv 'layer' width 4; evaluation every {sched['eval_every']} "
          f"on the stage set and the 8-stream set; early stop in stage 3 only")
    print(f"  restart rule (SC8_R): 2-stream held-out accuracy >= {sched['a_check']} at step {tc}, else restart with seed s + "
          f"{STRIDE}*j, at most {R_MAX} attempts, the last continues; attempt 0 = SC8's run")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<50} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        fits = verify(pool)
        store = load_store(args.results)
        old_drop = None if args.force else store.get("meta", {}).get("drop_decision")
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR, sched=sched,
                             r_max=R_MAX, stride=STRIDE, fits=fits, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on each stage's task, its evaluations charged "
              f"once per {EVAL_EVERY}; worst case = every continued run to {total} steps, every SC8_R trial needing "
              f"all {R_MAX} attempts (attempts 1-3 to {tc} + attempt 4 in full)")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        cost = {k: {int(n): v for n, v in c_.items()} for k, c_ in cost.items()}
        full, d_ceil, d_rest, mk1, mk2, total_h = projection(cost, sched, args.workers)
        for k in KEYS:
            per = ", ".join(f"{n} streams {cost[k][n] * 1000:.1f}" for n in sorted(cost[k]))
            print(f"  {k:<8} ms/step: {per:<50} x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        print(f"  SC8_ceil first: {len(d_ceil)} runs, {mk1 / 3600:.2f} h;  then {len(d_rest)} runs/trials: serial "
              f"{sum(d_rest) / 3600:.2f} h, {mk2 / 3600:.2f} h on {args.workers} workers;  total {total_h:.2f} h")
        drop = []
        if old_drop is not None:
            for what, seeds in old_drop:
                for k in (("D8",) if what == "D8" else ("SC8", "SC8_R")):
                    SEEDS[k] = tuple(seeds)
                drop.append(f"{what} cut to seeds {seeds[0]}-{seeds[-1]}")
            print(f"  the drop decision recorded at the first start of this results file is kept: "
                  f"{'; '.join(drop) if drop else 'nothing cut'}")
        else:
            decision = []
            for what, seeds in CUTS:
                if total_h <= DROP_H:
                    break
                for k in (("D8",) if what == "D8" else ("SC8", "SC8_R")):
                    SEEDS[k] = seeds
                decision.append([what, list(seeds)])
                drop.append(f"{what} cut to seeds {seeds[0]}-{seeds[-1]}")
                full, d_ceil, d_rest, mk1, mk2, total_h = projection(cost, sched, args.workers)
                print(f"  *** above {DROP_H:g} h: {what} cut to seeds {seeds[0]}-{seeds[-1]} (drop rule); projection now "
                      f"{total_h:.2f} h ***")
            if not drop:
                print(f"  within {DROP_H:g} h: nothing is cut")
            elif total_h > DROP_H:
                print(f"  (still above {DROP_H:g} h after both cuts; the drop rule has no further step)")
            store["meta"]["drop_decision"] = decision
        if old_drop is not None:
            store["meta"]["drop_decision"] = old_drop
        store["meta"].update(projected_wall_h=total_h, drop=drop, seeds={k: list(v) for k, v in SEEDS.items()})
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            extra, secs = "", rec.get("secs", 0.0)
            if ARM[sp["arm"]].get("restart"):
                t_ = rec.get("trial") or {}
                secs = t_.get("secs", secs)
                extra = f"  attempts {t_.get('used')} [{apath(rec)}]" + ("  (SC8 run reused)" if t_.get("reused") else "")
            print(f"   {sp['arm']:<8} seed {sp['seed']}  {describe(rec, sched)}{extra}  {secs:.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)")

        def run_specs(specs, on_done=None):
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
                    if sp.get("trial"):
                        rec = finish_trial(store, sp, rec)
                    record(sp, rec)
                    if on_done is not None:
                        pending[:0] = on_done(sp)

        def todo(k):
            out = []
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<8} seed {s}  cached")
                else:
                    out.append(spec(k, s, sched))
            return out

        print("=" * 100)
        print(f"RUNS — SC8_ceil first ({len(SEEDS['SC8_ceil'])} runs)")
        print("=" * 100)
        run_specs(todo("SC8_ceil"))
        n_ceil = sum(bound(store, "SC8_ceil", s) for s in SEEDS["SC8_ceil"])
        need = min(CEIL_MIN, len(SEEDS["SC8_ceil"]))
        print(f"   SC8_ceil bound {n_ceil}/{len(SEEDS['SC8_ceil'])} (needs >= {need})")
        print()
        if n_ceil < need:
            print("   *** the ceiling failed: nothing else is run ***\n")
            report_stop(store, sched, time.time() - t0, args.results)
            return

        def after(sp):
            if sp["arm"] == "SC8":
                nxt = r_step(store, sp["seed"], sched, args.force, record)
                return [] if nxt is None else [nxt]
            return []

        n_rest = sum(len(SEEDS[k]) for k in ("SC8", "SC8_R", "D8"))
        print("=" * 100)
        print(f"RUNS — the rest ({n_rest} runs/trials; each SC8_R trial is queued, first, when its SC8 run is known)")
        print("=" * 100)
        pre = []
        for s in SEEDS["SC8_R"]:
            if get(store, "SC8", s) is not None and not args.force:
                nxt = r_step(store, s, sched, args.force, record)
                if nxt is not None:
                    pre.append(nxt)
        rest = [sp for pr in itertools.zip_longest(todo("SC8"), todo("D8")) for sp in pr if sp is not None]
        run_specs(pre + rest, on_done=after)
        print()

    report(store, sched, fits, drop, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
