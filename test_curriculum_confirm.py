#!/usr/bin/env python
"""
test_curriculum_confirm.py — does the constant-lr load curriculum reliably make the gate find
and keep the stream routing at P=8 on fresh seeds? And a first, descriptive look at P=16.
Everything below is fixed before any result.

Run it directly:

    python test_curriculum_confirm.py --prev results/X/load_curriculum_results.json
    python test_curriculum_confirm.py --prev ... --also results/L/curriculum_confirm_results.json

BACKGROUND
test_load_curriculum, both machines: seeds 160-189, conv width 4. Phase 1: 4 of 8 keys at lr
1e-3, 4800 steps. Phase 2: all 8 keys. L ran merge 98f3bd3, code identical to b7c18f0.
- Bound, X / L:
  - CUR_A (phase 2 at 4e-3): 20/30 / 16/30
  - CUR_A_R (with phase-1 restarts): 28/30 / 25/30
  - CUR_B (one channel): 4/20 / 1/20
  - CUR_ceiling: 5/5 / 5/5
  - CUR_A_lo (phase 2 at 1e-3, seeds 160-169, descriptive): 9/10 / 9/10
  - Recorded A_conv8 (P=8 from scratch at 4e-3): 4/30 / 8/30
- Claims:
  - C1: SHOWN on X (16 vs 0 discordant), NOT SHOWN on L (14 vs 6, p = 0.058).
  - C2 and C3: SHOWN on both. Pooled, runs routed at the switch bound 52/59; runs not routed
    bound 2/20.
  - C4: RELIABLE on X, MAJORITY on L.
- Post hoc: the lr jump 1e-3 -> 4e-3 at the switch broke the routing within 1200 steps in 14
  of the 40 routed CUR_A runs. At a constant 1e-3, 0 of 12 routed runs lost it. CUR_A_lo vs
  CUR_A on seeds 160-169: 18/20 vs 11/20, discordant 7 vs 0, p = 0.016.
- Transfer is mostly immediate: 27/60 CUR_A runs were already >= 0.95 on the 8-key set at the
  switch.
- Seed-level outcomes differ between machines; the verdict is per machine.

PART 1: P=8, a constant lr of 1e-3 in every phase. Fresh seeds 200-219.
test_load_curriculum's knobs and arms; the only change from CUR_A_lo is the seeds (CHECK 75).
- CUR_lo: arm A + conv. Phase 1: 4 of 8 keys, steps 1-4800. Phase 2: all 8 keys, up to 24000
  more steps. Seeds 200-219 (20).
- CUR_lo_R: CUR_lo with the phase-1 restart rule (accuracy >= 0.6 at 2400; attempt seeds
  s + 1000*j, j < 5; attempt 0's record reused when it passes). Seeds 200-219 (20).
- DIRECT_lo: arm A + conv, all 8 keys from step 1, up to 28800 steps (the same total): test_p_
  scaling's A_conv8 run path at lr 1e-3. Early stop and the transition over the whole run.
  Seeds 200-214 (15).
- CUR_B_lo: single channel + conv, the curriculum. Seeds 200-214 (15).
- CUR_ceiling_lo: perfect gate + conv, the curriculum (validity; runs first). Seeds 200-204.
If CUR_ceiling_lo binds fewer than 4/5, the test stops: Part 1 is reported UNTESTED and
nothing else runs.
Every Part 1 run also records the gate statistics at the first evaluation after the switch
(step 6000; DIRECT_lo at the same steps): run_cur's stats_at knob, which adds statistics only.
DIRECT_lo evaluates the phase-1 set too (descriptive; evaluation does not perturb training).

PART 2: P=16, descriptive only, at a constant lr of 1e-3. Vocabulary task_for(16, 2).
- A three-stage curriculum: 4 of 16 keys, steps 1-4800; 8 of 16, steps 4801-9600; all 16, up
  to 24000 more steps (total 33600).
- Evaluation every 1200 steps on the current stage's held-out set and on the 16-key set
  (eval_batch(task_for(16, 2))); at each stage end (4800, 9600) both the 4- and 8-key sets.
  BOUND = the transition on the 16-key curve in the last stage. Early stopping only in the last
  stage. Gate statistics at 4800, 6000, 9600, 10800 and the end.
- CUR16: arm A + conv. Seeds 200-204 (5).
- CUR16_ceiling: perfect gate + conv, the same curriculum. Seeds 200-202 (3).
- CUR16_ceiling runs first in Part 2. If it binds 0/3, CUR16 is skipped (said so).
Drop rule: if the printed projection exceeds 12 h on the workers, Part 2 is dropped (said so).

CLAIMS (Part 1; outcome BOUND on the 8-key set)
- VALID: CUR_ceiling_lo binds >= 4/5.
- K1 THE CURRICULUM BEATS TRAINING FROM SCRATCH: CUR_lo (20) vs DIRECT_lo (15). Fisher one-sided
  (CUR_lo higher); exact McNemar (two-sided) on seeds 200-214 printed alongside.
- K2 THE GAIN NEEDS CHANNELS: CUR_lo (20) vs CUR_B_lo (15). Fisher one-sided; McNemar alongside.
- K3 RELIABLE: CUR_lo_R's band: RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9, NEVER 0.
  CUR_lo's band is printed alongside.
- K1 and K2 are SHOWN if p < 0.05, else NOT SHOWN.
- Readings, printed verbatim where they apply:
  - K1 SHOWN, K2 SHOWN and K3 RELIABLE: "a constant-lr load curriculum makes the gate find and
    keep the stream routing at P=8 on almost every run: the recipe to carry to P=16."
  - K1 NOT SHOWN: "at a constant 1e-3, training at P=8 from scratch does about as well as the
    curriculum."
  - K2 NOT SHOWN: "a single channel gains about as much from the curriculum: not a channel
    effect."
Fisher and McNemar (two-sided) are test_router_confirm's.

DIAGNOSTICS, not part of the verdict
- Per seed: every Part 1 arm's outcome, the margin at the switch and at the end, attempts used.
- Routing persistence: of the runs routed at the switch (margin >= 0.9), how many lost the
  routing by the first evaluation after it (8-key accuracy < 0.8, or margin < 0.5). Compared with
  test_load_curriculum's CUR_A (lr jump) on this machine, read from --prev; that file has no
  margin at step 6000, so its runs are counted on the accuracy criterion, with VAL cos > 0.5 at
  6000 (its post-hoc proxy) printed alongside, and both are printed for this test's runs too.
- 8-key accuracy at the switch (the immediate transfer); transitions and plateaus after the
  switch; failure classes; lag weights by outcome; gradient norms at steps 1, 10, 100, 4801
  and 4810.
- DIRECT_lo at 1e-3 vs the recorded A_conv8 at 4e-3 (the p_scaling file named in --prev's
  meta): a descriptive line only (different seeds).
- Part 2, per seed: bound, transition; each stage's set accuracy at each stage end; the margin
  at each stage end; the final 16-key accuracy.

CHECKS (printed before any training): test_load_curriculum's verification (which runs every
earlier test's; its CHECK 72 pairs with the p_scaling file named in the --prev file's meta), then
  75 the Part 1 arms equal test_load_curriculum's arms except for seeds and phase-2 lr: arm
     dicts; a 6000-step CUR_lo run through the new code equals test_load_curriculum's CUR_A_lo
     code path for the same seed (weights and curves; the record too, once the statistics added
     at step 6000 are removed); DIRECT_lo equals test_p_scaling's A_conv8 call at lr 1e-3; the
     knobs added to test_load_curriculum (run_cur's stats_at, run_trial's arm and stats_at) are
     inert at their defaults and explicit, vs 1062bb6's module;
  76 the optimizer lr is 1e-3 at steps 1, 4801 and 20000 (and every other step) in every Part 1
     arm, through run_job (step pre-hook; the model's forward stubbed to a cheap function and
     every held-out accuracy to 0.5, so the full 28800-step schedule runs in seconds and
     CUR_lo_R uses all its attempts);
  77 Part 2's schedule: the stages switch exactly at 4800 and 9600; the batches have 4, 8 and 16
     distinct keys, from task_for(16, 2)'s vocabulary; Adam's state is carried over; no early
     stop before the last stage even with accuracy stubbed to 1.0; n_active=4 and 8 of P=16 pass
     CHECK 70's generator checks; the perfect gate zeroes every cross-stream score at P=16;
  78 the restart seeds (s + 1000*j for s in 200-219) are disjoint from every earlier seed,
     test_load_curriculum's included;
  79 a worker's run is bit-identical to the same run in the main process (CUR_lo 6000 steps;
     CUR16 10000 steps).

LOGISTICS. Parallel single-thread workers; wall clock projected before training (worst case).
CUR_lo_R's trials are queued as soon as their CUR_lo run is known; CUR16 is queued when the
last CUR16_ceiling run is known. Results persist atomically to curriculum_confirm_results.json
(gitignored; copied into results/X/ after the run).

RESULT (full run, 83/83 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz, 4 workers
x 1 thread; projection 8.72 h worst case, so Part 2 ran; 216.9 min after verification; --prev
results/X/load_curriculum_results.json; no --also file; machine X, commit c5ec279).
  VALID: CUR_ceiling_lo bound 5/5.
  K1 THE CURRICULUM BEATS TRAINING FROM SCRATCH: NOT SHOWN. CUR_lo 16/20 vs DIRECT_lo 12/15;
     Fisher one-sided p = 0.66; McNemar on seeds 200-214: CUR_lo only 2 (s208, s213), DIRECT_lo
     only 3 (s203, s204, s214), p = 1.
  K2 THE GAIN NEEDS CHANNELS: SHOWN. CUR_lo 16/20 vs CUR_B_lo 0/15; Fisher one-sided p = 1.19e-06;
     McNemar on seeds 200-214: CUR_lo only 11, CUR_B_lo only 0, p = 0.000977.
  K3 RELIABLE: RELIABLE. CUR_lo_R bound 20/20 (CUR_lo alone 16/20: MAJORITY).
  The pre-registered reading: K1 NOT SHOWN: "at a constant 1e-3, training at P=8 from scratch
  does about as well as the curriculum." (K1 NOT SHOWN rules out the reading that needs K1, K2
  and K3 together.)

  Diagnostics (not part of the verdict):
  - At lr 1e-3 the gate binds P=8 from scratch: DIRECT_lo 12/15 (transition median 3600, three
    runs at 1200). The recorded A_conv8 at 4e-3 bound 4/30 on seeds 160-189; the seeds differ, so
    this is descriptive, but it points at the lr, not the curriculum, as what rescued the gate at
    P=8. DIRECT_lo's failures: s207 and s213 KEY, s208 OTHER.
  - Every binder of every gated Part 1 arm ended routed (end margin >= 0.9): CUR_ceiling_lo 5/5,
    CUR_lo 16/16, CUR_lo_R 20/20, DIRECT_lo 12/12. test_load_curriculum's CUR_A (4e-3 in phase 2)
    had 4 unrouted binders of 20.
  - Routing persistence: of the 21 distinct CUR_lo / CUR_lo_R runs routed at the switch, 0 lost it
    by step 6000 (on accuracy < 0.8, on margin < 0.5, and on VAL cos > 0.5 alike); all 21 bound.
    test_load_curriculum's CUR_A on this machine: 21/30 routed at the switch, 5 lost on accuracy
    alone, 6 with VAL cos > 0.5 at 6000; its CUR_A_lo: 0 of 7. CUR_ceiling_lo s200 is counted as
    lost on accuracy alone (0.634 at the switch, 0.670 at 6000) with its margin at 1.000
    throughout: the accuracy criterion also counts runs that were never above 0.8.
  - The phase-1 check sorts CUR_lo: the 13 runs that passed it (>= 0.6 at 2400) all bound, 12 of
    them at 6000 (s200 at 9600); of the 7 that failed it, 3 bound (s215 at 6000; s205 and s210 at 13200 and 14400,
    routing found in phase 2) and 4 did not (s203, s204, s207, s214, all POSITION-split at the
    switch and at the end, near 0.34-0.48). CUR_lo_R restarted those 7: attempt 2 for 203, 204,
    207, 210 and 214, attempt 3 for 205 and 215; all 7 passed, routed at the switch and bound (s214
    at 10800, the rest at 6000). CUR_lo_R vs CUR_lo: 4 vs 0, McNemar p = 0.125.
  - The transfer is immediate: 8-key accuracy at the switch, before any 8-key training, median
    0.949 in CUR_lo (>= 0.95 in 10/20) and 0.967 in CUR_lo_R (14/20).
  - One channel does not bind at 1e-3 with or without the curriculum: CUR_B_lo 0/15, every run
    between 0.39 and 0.51 at 28800; its phase-1 accuracy stayed near 0.5 (median 0.48 at 4800).
  - Part 2 (P=16, descriptive): CUR16_ceiling bound 3/3 (10800, 10800, 22800). CUR16 bound 4/5:
    s203 and s204 routed from stage 1 (margin >= 0.999 at 4800) and bound at 10800, the first
    16-key evaluation after the second switch; s200 (12000) and s201 (22800) bound WITHOUT the
    stream routing (margin ~0 and VAL cos ~1 at every stage; end lag-2 |w| 0.09 and 0.06 vs 0.04 and
    0.05 for the routed binders); s202 did not bind (0.40 at 33600, OTHER; 0.38 on the 4-key set at
    4800). 16-key accuracy at 9600, before any 16-key training: routed CUR16 0.77 and 0.88, the
    ceiling 0.62-0.87, the unrouted s200 0.91 and s201 0.71.
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
from torch.nn.modules.module import register_module_forward_pre_hook

from test_binding_capacity import BindTask
import test_binding_onset as tbo
from test_binding_onset import transition, time_per_step, fmt_step, EVAL_EVERY
from test_multilayer_binding import D
import test_channel_binding as tcb
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater, mcnemar_exact
from test_router_curriculum import (
    ARCH, get, count, load_store, init_worker, makespan, cpu_model, same_weights, save_results,
    DISC_COS, COLLAPSE, TAU_END, TIME_STEPS,
)
from test_router_discovery import load_at, model_fn
import test_router_reliability as trr
from test_router_reliability import run_one, strip_all, med, T_CHECK, A_CHECK
from test_router_layout import fail_class, ETA_GROUPS
from test_readout_path import GRAD_STEPS
from test_short_conv import conv_stats, conv_grad_norms, routed, wstr, CONV_K, ROUTED_MARGIN, PLATEAU
import test_conv_lr as tcl
from test_conv_lr import bound_r
import test_p_scaling as tps
import test_load_curriculum as tlc
from test_load_curriculum import at, sw_margin, routed_sw, plateau2, fmt, mark, curve_lines2, describe

# ── Settings (fixed before any result) ───────────────────────────────────────
LR1 = tlc.LR1                              # 1e-3, every phase and stage
R_MAX, STRIDE = tlc.R_MAX, tlc.STRIDE      # 5 attempts, seed s + 1000*j
CEIL_MIN = 4                               # VALID needs CUR_ceiling_lo >= 4/5
CEIL16_MIN = 1                             # CUR16 runs only if CUR16_ceiling binds >= 1/3
ALPHA = 0.05
RELIABLE_R, MAJORITY_R = 18, 10            # K3 bands, of 20
LOST_ACC, LOST_MARGIN = 0.8, 0.5           # routing lost by the first evaluation after the switch
DROP_H = 12.0                              # Part 2 is dropped if the projection exceeds this
LEGACY_SHA = "1062bb6"                     # head before this test's changes
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "curriculum_confirm_results.json"
PREV_FILE = "load_curriculum_results.json"

TASK1, TASK8 = tlc.TASK1, tlc.TASK8        # n_active=4 of P=8; task_for(8, 2)
TASK16 = tcb.task_for(16, 2)
STAGES16 = {4: BindTask(16, S=2, n_vals=tcb.N_VALS, n_q=tcb.N_Q, n_active=4),
            8: BindTask(16, S=2, n_vals=tcb.N_VALS, n_q=tcb.N_Q, n_active=8),
            16: TASK16}
SCHED = dict(tlc.REAL)                     # t_switch 4800, total 28800, eval_every 1200, t_check 2400, a_check 0.6
SCHED16 = dict(t1=4800, t2=9600, total=33600, eval_every=EVAL_EVERY)
REAL, REAL16 = dict(SCHED), dict(SCHED16)  # the CHECKs use these (a dry run shortens SCHED and SCHED16)

ARMS1 = [
    dict(tlc.ARM["CUR_ceiling"], key="CUR_ceiling_lo", lr2=LR1,
         label="CUR_ceiling_lo  perfect gate + conv, curriculum"),
    dict(tlc.ARM["CUR_A_lo"], key="CUR_lo", label="CUR_lo          arm A + conv, curriculum"),
    dict(tlc.ARM["CUR_A_R"], key="CUR_lo_R", lr2=LR1,
         label="CUR_lo_R        CUR_lo + restarts in phase 1"),
    dict(tps.ARM["A_conv8"], key="DIRECT_lo", lr=LR1,
         label="DIRECT_lo       arm A + conv, all 8 keys from step 1"),
    dict(tlc.ARM["CUR_B"], key="CUR_B_lo", lr2=LR1,
         label="CUR_B_lo        single channel + conv, curriculum"),
]
ARMS2 = [
    dict(tps.ARM["ceiling_conv8"], key="CUR16_ceiling", task="P16S2", lr=LR1,
         label="CUR16_ceiling   perfect gate + conv, 3 stages"),
    dict(tps.ARM["A_conv8"], key="CUR16", task="P16S2", lr=LR1,
         label="CUR16           arm A + conv, 3 stages"),
]
ARM = {a["key"]: a for a in ARMS1 + ARMS2}
PART1 = tuple(a["key"] for a in ARMS1)
PART2 = tuple(a["key"] for a in ARMS2)
KEYS = PART1 + PART2
GATED1 = tuple(k for k in PART1 if ARM[k]["gate"] != "none")
BASE = dict(CUR_ceiling_lo="CUR_ceiling", CUR_lo="CUR_A_lo", CUR_lo_R="CUR_A_R", CUR_B_lo="CUR_B")
SEEDS = dict(CUR_ceiling_lo=tuple(range(200, 205)), CUR_lo=tuple(range(200, 220)),
             CUR_lo_R=tuple(range(200, 220)), DIRECT_lo=tuple(range(200, 215)),
             CUR_B_lo=tuple(range(200, 215)), CUR16_ceiling=tuple(range(200, 203)),
             CUR16=tuple(range(200, 205)))
READINGS = {
    "K1+ K2+ K3 RELIABLE": "a constant-lr load curriculum makes the gate find and keep the stream "
                           "routing at P=8 on almost every run: the recipe to carry to P=16.",
    "K1-": "at a constant 1e-3, training at P=8 from scratch does about as well as the curriculum.",
    "K2-": "a single channel gains about as much from the curriculum: not a channel effect.",
}
_REAL_EVALUATE, _REAL_LOGITS = tbo.evaluate, tbo.logits_of
_PROBE = {"on": False}                     # CHECK 76's stubs are installed (in that worker, for that job only)


def band(c):
    return ("RELIABLE" if c >= RELIABLE_R else "MAJORITY" if c >= MAJORITY_R
            else "MINORITY" if c >= 1 else "NEVER")


def attempt_seeds(s):
    return [s + STRIDE * j for j in range(R_MAX)]


def stats_at_step(rec, step):
    return next((x for x in rec["stats"] if x["step"] == step), None)


# ── Part 1: test_load_curriculum's run path ──────────────────────────────────
def run_direct(a, seed, sched, keep=None):
    """DIRECT_lo: test_p_scaling's A_conv8 call (run_one on task_for(8, 2)) at the arm's lr, for
    sched["total"] steps, with the statistics at the switch step and after it (cur_stats_fn), the
    phase-1 set evaluated at every evaluation (descriptive) and the curriculum's gradient steps."""
    ts, ee = sched["t_switch"], sched["eval_every"]
    data1 = eval_batch(TASK1)
    curve1 = []

    def check(m, step, data):
        acc, el = tbo.evaluate(m, TASK1, data1)
        curve1.append([step, acc, el])

    rec = run_one(a, seed, sched["total"], ee, check=check, keep=keep, task=TASK8,
                  stats_fn=tlc.cur_stats_fn(ts, also=(ts + ee,)), grad_fn=conv_grad_norms, lr=a["lr"],
                  builder=tcl.builder_for(a), grad_steps=tlc.grad_steps_of(sched))
    rec["curve1"] = curve1
    if rec["ok"]:
        rec["switch"] = stats_at_step(rec, ts)
    return rec


def run_part1(a, seed, sched, decide=None, keep=None):
    ts, ee = sched["t_switch"], sched["eval_every"]
    if a["key"] == "DIRECT_lo":
        rec = run_direct(a, seed, sched, keep=keep)
    else:
        rec = tlc.run_cur(a, seed, sched, decide=decide, keep=keep, stats_at=(ts + ee,))
    if rec["ok"]:
        rec["after"] = stats_at_step(rec, ts + ee)
    return rec


def run_trial(seed, sched, start=1, decide=None):
    ts, ee = sched["t_switch"], sched["eval_every"]
    rec = tlc.run_trial(seed, sched, start=start, decide=decide, arm=ARM["CUR_lo_R"], stats_at=(ts + ee,))
    if rec["ok"]:
        rec["after"] = stats_at_step(rec, ts + ee)
    return rec


# ── Part 2: three stages at P=16 ─────────────────────────────────────────────
def builder16(aa, seed):
    return model_fn(TASK16, aa, seed)


def stage_of(sched, step):
    return 4 if step <= sched["t1"] else 8 if step <= sched["t2"] else 16


def grad16(sched):
    t1, t2 = sched["t1"], sched["t2"]
    return tuple(GRAD_STEPS) + (t1 + 1, t1 + 10, t2 + 1, t2 + 10)


def run_cur16(a, seed, sched, keep=None):
    """run_one on task_for(16, 2) (its held-out set, probe, statistics and early stop), with each
    stage's batches; check() evaluates the current stage's set at every evaluation up to t2, and
    the 4- and 8-key sets at t1 and t2."""
    t1, t2, ee = sched["t1"], sched["t2"], sched["eval_every"]
    sets = {n: eval_batch(STAGES16[n]) for n in (4, 8)}
    info = dict(curve_stage=[], ends={})

    def check(m, step, data):
        n = stage_of(sched, step)
        if n != 16:
            acc, el = tbo.evaluate(m, STAGES16[n], sets[n])
            info["curve_stage"].append([step, acc, el])
        if step in (t1, t2):
            info["ends"][str(step)] = {str(k): tbo.evaluate(m, STAGES16[k], sets[k])[0] for k in (4, 8)}

    rec = run_one(a, seed, sched["total"], ee, check=check, keep=keep, task=TASK16,
                  stats_fn=tlc.cur_stats_fn(t1, also=(t1 + ee, t2, t2 + ee)), grad_fn=conv_grad_norms,
                  lr=LR1, builder=builder16, grad_steps=grad16(sched),
                  run_kw=dict(task_at=lambda step: STAGES16[stage_of(sched, step)], early_stop_after=t2))
    rec.update(curve_stage=info["curve_stage"], ends=info["ends"])
    if rec["ok"]:
        rec["transition_full"] = rec["transition"]
        rec["transition"] = transition([c for c in rec["curve"] if c[0] > t2])
        rec["discovered"] = bool(rec["transition"] is not None and rec["val_cos"] < DISC_COS)
        for name, st in (("st1", t1), ("st1a", t1 + ee), ("st2", t2), ("st2a", t2 + ee)):
            rec[name] = stats_at_step(rec, st)
    return rec


# ── Jobs ─────────────────────────────────────────────────────────────────────
def run_job(sp):
    assert _PROBE["on"] or (tbo.evaluate is _REAL_EVALUATE and tbo.logits_of is _REAL_LOGITS), \
        "a CHECK stub is still installed"
    sched = sp["sched"]
    if sp["arm"] in PART2:
        return run_cur16(ARM[sp["arm"]], sp["seed"], sched)
    decide = trr.make_decide(sched["t_check"], sched["a_check"])
    if sp.get("trial"):
        return run_trial(sp["seed"], sched, start=sp["start"], decide=decide)
    return run_part1(ARM[sp["arm"]], sp["seed"], sched, decide=decide)


def spec(key, seed, sched, trial=False, start=1):
    sp = dict(arm=key, seed=seed, sched=dict(sched))
    if trial:
        sp.update(trial=True, start=start)
    return sp


def ref_job(seed, sched):
    """CHECK 75: test_load_curriculum's own CUR_A_lo code path (its run_job's call), returning
    the record and the trained weights."""
    keep = {"at": -1}
    r = tlc.run_cur(tlc.ARM["CUR_A_lo"], seed, sched,
                    decide=trr.make_decide(sched["t_check"], sched["a_check"]), keep=keep)
    return r, {n: p.detach().clone() for n, p in keep["model"].named_parameters()}


def lr_probe_job(key, sched):
    """CHECK 76: every optimizer step's lr through run_job for arm `key`, seed 1. The model's
    forward is stubbed to a cheap function of one parameter and every held-out accuracy to 0.5
    (no early stop; the restart check fails), so the full schedule runs in seconds. Stubs and
    hook are removed before returning (the worker is reused)."""
    opts = []

    def pre(opt, args, kw):
        if not hasattr(opt, "_check76"):
            opt._check76 = len(opts)
            opts.append([])
        opts[opt._check76].append(sorted({g["lr"] for g in opt.param_groups}))

    def cheap(model, inp):
        p = next(model.parameters())
        return p.reshape(-1)[0] * torch.ones(inp.shape[0], inp.shape[1], TASK8.vocab)

    h = register_optimizer_step_pre_hook(pre)
    _PROBE["on"] = True
    tbo.evaluate, tbo.logits_of = (lambda model, task, d: (0.5, 0.0)), cheap
    try:
        rec = run_job(spec(key, 1, sched, trial=key == "CUR_lo_R", start=0))
    finally:
        h.remove()
        tbo.evaluate, tbo.logits_of = _REAL_EVALUATE, _REAL_LOGITS
        _PROBE["on"] = False
    return dict(ok=rec["ok"], n_opt=len(opts), steps=[len(x) for x in opts], last=opts[-1],
                all_lrs=sorted({v for ls in opts for x in ls for v in x}),
                attempts=rec.get("trial", {}).get("used"))


def time_job16(key):
    """ms/step on each stage's task (test_load_curriculum.time_job's recipe), each charged with its
    evaluations once per EVAL_EVERY (the stage set and the 16-key set; the 16-key set alone last)."""
    a = ARM[key]
    mk = builder16(a, 0)
    out = {}
    for n in (4, 8, 16):
        task = STAGES16[n]
        per = time_per_step(task, mk, steps=TIME_STEPS)
        m, data = mk(), eval_batch(task)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        ev = time.time() - t0
        out[n] = (max(per - ev / TIME_STEPS, 1e-6), ev)
    ev16 = out[16][1]
    return tuple(out[n][0] + ((out[n][1] if n != 16 else 0.0) + ev16) / EVAL_EVERY for n in (4, 8, 16))


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool, prev_path):
    p8_path = tlc.P8_FILE
    if os.path.exists(prev_path):
        p8_path = load_store(prev_path).get("meta", {}).get("p8") or p8_path
    print("test_load_curriculum.py's verification (which runs test_p_scaling's, and so on down to")
    print(f"test_multilayer_binding's); its CHECK 72 pairs with {p8_path}, the p_scaling file named in")
    print("the --prev file's meta:")
    tlc.verify(pool, p8_path)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    ts, ee = REAL["t_switch"], REAL["eval_every"]
    N75, N79_16 = ts + ee, REAL16["t2"] + 400
    # the long runs go to the pool first; the main process runs its side meanwhile
    f_ref = pool.submit(ref_job, 1, dict(REAL, total=N75))
    f_w1 = pool.submit(run_job, spec("CUR_lo", 1, dict(REAL, total=N75)))
    f_w16 = pool.submit(run_job, spec("CUR16", 1, dict(REAL16, total=N79_16)))
    f_lr = {k: pool.submit(lr_probe_job, k, REAL) for k in PART1}

    print(f"CHECK 75 the Part 1 arms are test_load_curriculum's arms except for seeds and phase-2 lr:")
    good = True
    for k in PART1:
        if k == "DIRECT_lo":
            ref, ref_name, diff = tps.ARM["A_conv8"], "test_p_scaling's A_conv8", "lr"
        else:
            ref, ref_name, diff = tlc.ARM[BASE[k]], f"test_load_curriculum's {BASE[k]}", "lr2"
        drop = ("key", "label", diff)
        same = ({x: v for x, v in ARM[k].items() if x not in drop} == {x: v for x, v in ref.items() if x not in drop}
                and ARM[k][diff] == LR1)
        good &= same
        print(f"     (a) {k:<14} = {ref_name:<36} except key, label and {diff:<3} ({ref[diff]} -> {ARM[k][diff]}): {same}")
    leg = load_at(LEGACY_SHA, "test_load_curriculum.py", "test_load_curriculum_legacy")
    small = dict(t_switch=40, total=80, eval_every=20, t_check=20, a_check=A_CHECK)
    good_b = True
    for key in ("CUR_A", "CUR_B"):
        dec = trr.make_decide(20, A_CHECK)
        r0 = leg.run_cur(leg.ARM[key], 3, small, decide=dec)
        r1 = tlc.run_cur(tlc.ARM[key], 3, small, decide=dec)
        r2 = tlc.run_cur(tlc.ARM[key], 3, small, decide=dec, stats_at=())
        g = strip_all(r0) == strip_all(r1) == strip_all(r2) and r1["ok"]
        good_b &= g
        print(f"     (b) run_cur {key:<5} (switch at 40 of 80, evaluations every 20) vs {LEGACY_SHA}'s: records equal "
              f"(default, explicit stats_at) {g}  -> {'UNCHANGED' if g else 'CHANGED'}")
    tiny = dict(t_switch=8, total=16, eval_every=2, t_check=4, a_check=A_CHECK)
    dec = trr.make_decide(4, A_CHECK)
    t0_ = leg.run_trial(3, tiny, start=0, decide=dec)
    t1_ = tlc.run_trial(3, tiny, start=0, decide=dec)
    t2_ = tlc.run_trial(3, tiny, start=0, decide=dec, arm=tlc.ARM["CUR_A_R"], stats_at=())
    g = strip_all(t0_) == strip_all(t1_) == strip_all(t2_) and t1_["ok"]
    good_b &= g
    print(f"     (b) run_trial (switch at 8 of 16, check at 4; {t1_['trial']['used']} attempts) vs {LEGACY_SHA}'s: records "
          f"equal (default, explicit arm and stats_at) {g}  -> {'UNCHANGED' if g else 'CHANGED'}")
    good &= good_b
    keep1 = {"at": -1}
    h1 = run_part1(ARM["CUR_lo"], 1, dict(REAL, total=N75), decide=trr.make_decide(REAL["t_check"], REAL["a_check"]),
                   keep=keep1)
    ref_rec, ref_w = f_ref.result()
    w_eq = (set(ref_w) == {n for n, _ in keep1["model"].named_parameters()}
            and all(torch.equal(p.detach(), ref_w[n]) for n, p in keep1["model"].named_parameters()))
    c_eq = h1["curve"] == ref_rec["curve"] and h1["curve1"] == ref_rec["curve1"]
    body = {x: v for x, v in h1.items() if x != "after"}
    added = [x for x in (stats_at_step(h1, ts + ee) or {}) if x.startswith("eta_") or x == "margin"]
    body["stats"] = [{x: v for x, v in st.items() if not (st["step"] == ts + ee and x in added)} for st in h1["stats"]]
    r_eq = strip_all(body) == strip_all(ref_rec)
    g = w_eq and c_eq and r_eq and h1["ok"] and len(added) == 9
    good &= g
    print(f"     (c) CUR_lo seed 1, {N75} steps, vs test_load_curriculum's CUR_A_lo code path (its run_cur, in a worker):")
    print(f"         weights equal {w_eq}   curves (8-key and phase-1 sets) equal {c_eq}   record equal once the {len(added)} "
          f"routing statistics added at step {ts + ee} are removed {r_eq}")
    print(f"         8-key acc {[round(c[1], 3) for c in h1['curve']]}   margin at {ts} {fmt(sw_margin(h1))}, at {ts + ee} "
          f"{fmt((h1['after'] or {}).get('margin'))}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    kd, kp = {"at": -1}, {"at": -1}
    rd = run_direct(ARM["DIRECT_lo"], 3, dict(REAL, total=60, eval_every=20), keep=kd)
    ra = tps.ARM["A_conv8"]
    rp = run_one(ra, 3, 60, 20, task=TASK8, stats_fn=conv_stats, grad_fn=conv_grad_norms, lr=LR1,
                 builder=tcl.builder_for(ra), keep=kp)
    bd = {x: v for x, v in rd.items() if x not in ("curve1", "switch")}
    g = strip_all(bd) == strip_all(rp) and same_weights(kd["model"], kp["model"]) and rd["ok"]
    good &= g
    print(f"     (d) DIRECT_lo seed 3, 60 steps (evaluations every 20) vs test_p_scaling's A_conv8 call (run_one on "
          f"task_for(8, 2)) at lr {LR1}: weights equal {same_weights(kd['model'], kp['model'])}   record equal except "
          f"the added curve1 and switch {strip_all(bd) == strip_all(rp)}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 76 the optimizer lr in every Part 1 arm, through run_job with the real schedule ({REAL['total']} steps; "
          f"step pre-hook; forward stubbed, held-out accuracy stubbed to 0.5):")
    good = True
    for k in PART1:
        r = f_lr[k].result()
        last = r["last"]
        pick = {s_: (last[s_ - 1] if len(last) >= s_ else None) for s_ in (1, ts + 1, 20000)}
        g = (r["ok"] and r["all_lrs"] == [LR1] and len(last) == REAL["total"]
             and all(v == [LR1] for v in pick.values())
             and (r["n_opt"] == R_MAX if k == "CUR_lo_R" else r["n_opt"] == 1))
        good &= g
        att = (f"   {r['n_opt']} optimizers (attempts, the first {R_MAX - 1} abandoned at {REAL['t_check']}: steps "
               f"{r['steps']})" if k == "CUR_lo_R" else "")
        print(f"         {k:<14} lr at steps 1, {ts + 1}, 20000: {pick[1]} {pick[ts + 1]} {pick[20000]}   every step "
              f"({len(last)} steps): {r['all_lrs']}{att}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 77 Part 2's schedule (CUR16 seed 1, the real schedule run {N79_16} steps, hooks on every step; the "
          f"run CHECK 79 compares):")
    rec_opt, widths, keysets, toks = [], [], {}, [0, 10 ** 9]
    cnt = {"fw": 0}
    t1, t2 = REAL16["t1"], REAL16["t2"]
    watch = (t1 - 1, t1, t1 + 1, t1 + 2, t2 - 1, t2, t2 + 1, t2 + 2)

    def pre_opt(opt, args, kw):
        p0 = opt.param_groups[0]["params"][0]
        st = opt.state.get(p0, {}).get("step")
        rec_opt.append((id(opt), sorted({g_["lr"] for g_ in opt.param_groups}), None if st is None else float(st)))

    def pre_fw(mod, inp):
        if type(mod).__name__ == "MultiBDH" and mod.training:
            cnt["fw"] += 1
            x = inp[0]
            widths.append(x.shape[1])
            toks[0], toks[1] = max(toks[0], int(x.max())), min(toks[1], int(x.min()))
            if cnt["fw"] in watch:
                j_ = torch.arange((x.shape[1] + 1) // 3 - 1)
                keysets[cnt["fw"]] = sorted({len(set(row.tolist())) for row in x[:, 3 * j_ + 1]})
    ha = register_optimizer_step_pre_hook(pre_opt)
    hb = register_module_forward_pre_hook(pre_fw)
    try:
        h16 = run_cur16(ARM["CUR16"], 1, dict(REAL16, total=N79_16))
    finally:
        ha.remove()
        hb.remove()
    n = N79_16
    lr_ok = len(rec_opt) == n and all(x[1] == [LR1] for x in rec_opt)
    w_ok = (len(widths) == n and set(widths[:t1]) == {STAGES16[4].L - 1} and set(widths[t1:t2]) == {STAGES16[8].L - 1}
            and set(widths[t2:]) == {TASK16.L - 1})
    want = {t1 - 1: [4], t1: [4], t1 + 1: [8], t1 + 2: [8], t2 - 1: [8], t2: [8], t2 + 1: [16], t2 + 2: [16]}
    k_ok = keysets == want
    v_ok = toks[1] >= 0 and toks[0] < TASK16.vocab and all(STAGES16[m].vocab == TASK16.vocab for m in (4, 8))
    same_opt = len({x[0] for x in rec_opt}) == 1
    adam_ok = (rec_opt[t1][2] == float(t1) and rec_opt[t2][2] == float(t2) and rec_opt[t1 - 1][2] == float(t1 - 1)
               and rec_opt[t2 - 1][2] == float(t2 - 1))
    print(f"         lr at every step: {sorted({v for x in rec_opt for v in x[1]})} ({len(rec_opt)} steps): {lr_ok}")
    print(f"         training input widths at steps 1-{t1}: {sorted(set(widths[:t1]))}, {t1 + 1}-{t2}: "
          f"{sorted(set(widths[t1:t2]))}, {t2 + 1}-{n}: {sorted(set(widths[t2:]))}: {w_ok}")
    print(f"         distinct keys per sequence at steps {', '.join(str(s_) for s_ in watch)}: "
          f"{[keysets.get(s_) for s_ in watch]}: {k_ok}")
    print(f"         training token ids in [{toks[1]}, {toks[0]}], within task_for(16, 2)'s vocabulary [0, {TASK16.vocab}) "
          f"(every stage's task has that vocabulary): {v_ok}")
    print(f"         one optimizer throughout: {same_opt}   Adam's step count entering steps {t1}, {t1 + 1}, {t2}, "
          f"{t2 + 1}: {rec_opt[t1 - 1][2]}, {rec_opt[t1][2]}, {rec_opt[t2 - 1][2]}, {rec_opt[t2][2]} (carried over): "
          f"{adam_ok}   run ok {h16['ok']}")
    good = lr_ok and w_ok and k_ok and v_ok and same_opt and adam_ok and h16["ok"]
    sm = dict(t1=40, t2=80, total=200, eval_every=10)
    tbo.evaluate = lambda model, task, d: (1.0, 0.0)
    try:
        r_cur = run_cur16(ARM["CUR16"], 1, sm)
        r_plain = run_one(ARM["CUR16"], 1, sm["total"], sm["eval_every"], task=TASK16, stats_fn=conv_stats,
                          grad_fn=conv_grad_norms, lr=LR1, builder=builder16,
                          run_kw=dict(task_at=lambda step: STAGES16[stage_of(sm, step)]))
    finally:
        tbo.evaluate = _REAL_EVALUATE
    stop_ok = (r_cur["ok"] and r_cur["stopped_at"] == sm["t2"] + 3 * sm["eval_every"]
               and r_cur["transition"] == sm["t2"] + sm["eval_every"] and r_plain["stopped_at"] == 3 * sm["eval_every"])
    print(f"         early stop, every held-out accuracy stubbed to 1.0 (stages switch at {sm['t1']} and {sm['t2']}, "
          f"evaluations every {sm['eval_every']}): the curriculum stops at step {r_cur['stopped_at']} (the last stage's "
          f"third evaluation), transition {r_cur['transition']};")
    print(f"         without early_stop_after the same run would stop at step {r_plain['stopped_at']}, in the first "
          f"stage: {stop_ok}")
    good &= stop_ok
    for m in (4, 8):
        print(f"         n_active={m} of P=16, over {10_000:,} sequences (generator seed 77{m}), CHECK 70's checks:")
        good &= tlc.generator_check(STAGES16[m], TASK16, 10_000, 770 + m)
    t_eq = all(vars(STAGES16[m])[x] == vars(TASK16)[x] for m in (4, 8) for x in ("P", "S", "n_vals", "n_q", "vocab",
                                                                                   "ctx_tokens", "layout"))
    print(f"         the stage tasks share task_for(16, 2)'s P, S, n_vals, n_q, vocab, context tokens and layout: {t_eq}")
    good &= t_eq
    a = ARM["CUR16_ceiling"]
    for m in (4, 16):
        x = STAGES16[m].make_batch(16, torch.Generator().manual_seed(7700 + m))[0][:, :-1]
        mdl = builder16(a, 6)().eval()
        with torch.no_grad():
            mdl.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(771)))
        mdl.attn.record = []
        with torch.no_grad():
            mdl(x, TAU_END)
        lab = STAGES16[m].stream_labels(x)
        cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
        maxc = [s_[:, 0][cross].abs().max().item() for s_ in mdl.attn.record]
        live = [s_[:, 0][~cross].abs().max().item() for s_ in mdl.attn.record]
        mdl.attn.record = None
        g = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0
        good &= g
        print(f"         CUR16_ceiling at P=16, {m}-key input (conv on, random weights): layers seen {len(maxc)}; max "
              f"|cross-stream score| per layer {[f'{c:.2e}' for c in maxc]}; max |same-stream score| "
              f"{[f'{l_:.2e}' for l_ in live]}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 78 the seeds:")
    tlc_attempts = {x for s in tlc.SEEDS["CUR_A_R"] for x in tlc.attempt_seeds(s)[1:]}
    earlier = tlc.EARLIER_SEEDS | tlc_attempts | set(tlc.SEEDS["CUR_A"])
    primary = set(SEEDS["CUR_lo"])
    new_seeds = [x for s in SEEDS["CUR_lo_R"] for x in attempt_seeds(s)[1:]]
    g1 = not primary & earlier and all(set(SEEDS[k]) <= primary for k in KEYS)
    g2 = (len(set(new_seeds)) == len(new_seeds) == len(SEEDS["CUR_lo_R"]) * (R_MAX - 1)
          and not set(new_seeds) & (earlier | primary)
          and all(x == s + STRIDE * j for s in SEEDS["CUR_lo_R"] for j, x in enumerate(attempt_seeds(s))))
    rng_txt = ", ".join(f"{STRIDE * j + SEEDS['CUR_lo_R'][0]}-{STRIDE * j + SEEDS['CUR_lo_R'][-1]}" for j in range(1, R_MAX))
    tlc_txt = ", ".join(f"{STRIDE * j + 160}-{STRIDE * j + 189}" for j in range(1, R_MAX))
    print(f"         every arm's seeds lie in 200-219, disjoint from every earlier seed (0-199, 1000-1149, and "
          f"test_load_curriculum's attempt seeds {tlc_txt}): {g1}")
    print(f"         restart seeds s + 1000*j (j = 1..{R_MAX - 1}): {rng_txt}; {len(set(new_seeds))} distinct, disjoint "
          f"from every earlier seed and from 200-219: {g2}")
    good = g1 and g2
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 79 a worker's run is bit-identical to the same run here:")
    good = True
    for name, rw, rh, nsteps in (("CUR_lo", f_w1.result(), h1, N75), ("CUR16", f_w16.result(), h16, N79_16)):
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        if name == "CUR_lo":
            detail = (f"phase-1 acc {[round(c[1], 3) for c in rh['curve1']]}   8-key acc "
                      f"{[round(c[1], 3) for c in rh['curve']]}")
        else:
            detail = (f"stage-set acc {[round(c[1], 3) for c in rh['curve_stage']]}   16-key acc "
                      f"{[round(c[1], 3) for c in rh['curve']]}")
        print(f"         {name:<6} seed 1, {nsteps} steps: equal {strip_all(rw) == strip_all(rh)}   {detail}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print("ALL VERIFICATION CHECKS PASSED\n" if ok else "SOME VERIFICATION CHECKS FAILED\n")
    assert ok, "verification failed — do not trust the results below"
    return p8_path


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds)
            if (r := get(store, key, s)) and r.get("ok")]


def after_margin(r):
    return (r.get("after") or {}).get("margin")


def lost(r, ts, ee, acc_only=False):
    """Routing lost by the first evaluation after the switch: 8-key accuracy < 0.8, or margin < 0.5."""
    a8 = at(r["curve"], ts + ee)
    m = None if acc_only else after_margin(r)
    return (a8 is not None and a8 < LOST_ACC) or (m is not None and m < LOST_MARGIN)


def cos_after(r, ts, ee):
    st = stats_at_step(r, ts + ee)
    return None if st is None else st["val_cos"]


def distinct_runs(store):
    out = [("CUR_lo", s, r) for s, r in ok_runs(store, "CUR_lo")]
    out += [("CUR_lo_R", s, r) for s, r in ok_runs(store, "CUR_lo_R") if not r["trial"]["reused"]]
    return out


def flags1(k, r, ts, ee):
    if r["transition"] is not None:
        fl = ["BOUND"] + (["DISCOVERED"] if r["discovered"] else [])
        if k in GATED1:
            fl.append("routed" if routed(r) else "NOT routed")
    else:
        fl = [fail_class(r)] if k in GATED1 else []
    if k in GATED1 and k != "DIRECT_lo" and routed_sw(r) and lost(r, ts, ee):
        fl.append("LOST after the switch")
    fl += (["collapsed"] if r["collapsed"] else [])
    if k == "CUR_lo_R":
        t_ = r["trial"]
        fl.append(f"attempts {t_['used']}" + (" (CUR_lo reused)" if t_["reused"] else ""))
    return fl


def raw_tables(store, sched):
    ts, tc, ee = sched["t_switch"], sched["t_check"], sched["eval_every"]
    print(f"  (DIRECT_lo trains on all 8 keys throughout: its 'sw' columns are the same steps, and p1 its accuracy "
          f"on the phase-1 set)")
    print(f"  {'arm':<14} {'seed':>4} {'run':>5} {'acc P8':>7} {'transition':>10} {'p1@' + str(tc):>8} "
          f"{'p1@' + str(ts):>8} {'P8@' + str(ts):>8} {'P8@' + str(ts + ee):>8} {'margin sw':>9} {'+1 eval':>7} "
          f"{'end':>6} {'VAL cos':>7}  eta^2 key s/k/h/i end   conv |w| lag 0/1/2/3   flags")
    for k in PART1:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<14} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<14} {s:>4}  FAILED — {r.get('error')}")
                continue
            e = r["end"]
            etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS) if k in GATED1 else "--"
            print(f"  {k:<14} {s:>4} {r['seed']:>5} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                  f"{fmt(at(r['curve1'], tc)):>8} {fmt(at(r['curve1'], ts)):>8} {fmt(at(r['curve'], ts)):>8} "
                  f"{fmt(at(r['curve'], ts + ee)):>8} {fmt(sw_margin(r)):>9} {fmt(after_margin(r)):>7} "
                  f"{fmt(e.get('margin')):>6} {e['val_cos']:>7.3f}  {etas:<22}   {wstr(e):<23}  "
                  f"{' '.join(flags1(k, r, ts, ee))}")
        print()
    print(f"  CUR_lo_R attempts (seed: phase-1 accuracy at {tc}, pass/fail; the last attempt continues):")
    for s, r in ok_runs(store, "CUR_lo_R"):
        t_ = r["trial"]
        print(f"    s{s}: " + "   ".join(f"{x['seed']}: {fmt(x['acc_check'], 4)} "
                                        f"{'pass' if x['passed'] else 'fail'}" for x in t_["attempts"])
              + f"   -> continued {t_['continued']}" + ("  (CUR_lo's record reused)" if t_["reused"] else ""))
    print()


def report_stop(store, sched, wall, path):
    """The ceiling failed: Part 1 UNTESTED, nothing else run."""
    print("#" * 100)
    print("PER-SEED RAW RESULTS — CUR_ceiling_lo only (nothing else was run)")
    print("#" * 100)
    k = "CUR_ceiling_lo"
    for s in SEEDS[k]:
        r = get(store, k, s)
        if r and r.get("ok"):
            print(f"  {k} s{s}: acc P8 {r['acc']:.4f}  transition {fmt_step(r['transition'])}  "
                  f"phase-1 at {sched['t_switch']} {fmt(at(r['curve1'], sched['t_switch']))}")
        else:
            print(f"  {k} s{s}: {'NOT RUN' if r is None else 'FAILED — ' + str(r.get('error'))}")
    c = sum(bound(store, k, s) for s in SEEDS[k])
    need = min(CEIL_MIN, len(SEEDS[k]))
    print()
    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    print(f"  CUR_ceiling_lo binds {c}/{len(SEEDS[k])} (needs >= {need}): the validity condition FAILED; nothing "
          f"else was run (Part 2 included)")
    print("     *** Part 1 UNTESTED ***")
    print()
    print("PRE-REGISTERED CLAIMS: K1 UNTESTED, K2 UNTESTED, K3 UNTESTED; no reading. Part 2 not run.")
    store["verdict"] = dict(valid=False, stopped="ceiling", ceiling=c, K1="UNTESTED", K2="UNTESTED",
                            K3="UNTESTED", readings=[], part2="not run")
    save_results(path, store)
    print(f"\n  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}\n")


def persistence_line(name, rs, ts, ee, has_margin=True):
    rsw = [r for r in rs if routed_sw(r)]
    acc = sum(1 for r in rsw if lost(r, ts, ee, acc_only=True))
    cos = sum(1 for r in rsw if (c := cos_after(r, ts, ee)) is not None and c > 0.5)
    full = f"lost (acc < {LOST_ACC} or margin < {LOST_MARGIN}) {sum(1 for r in rsw if lost(r, ts, ee))}   " \
        if has_margin else f"{'(no margin at ' + str(ts + ee) + ' recorded)':<36}"
    print(f"    {name:<34} routed at the switch {len(rsw):>2}/{len(rs):<2}  {full}lost on accuracy alone {acc}   "
          f"VAL cos > 0.5 at {ts + ee}: {cos}   of the routed: bound {sum(1 for r in rsw if r['transition'] is not None)}")


def report(store, sched, sched16, wall, path, also, prev_path, p8_path, part2):
    ts, tc, ee = sched["t_switch"], sched["t_check"], sched["eval_every"]
    print("#" * 100)
    print("PER-SEED RAW RESULTS (PART 1) — every value, before any aggregate")
    print("#" * 100)
    raw_tables(store, sched)

    print("=" * 100)
    print("COUNTS, PART 1 (BOUND = transition on the 8-key curve, after the switch for the curriculum arms; "
          "DISCOVERED = bound AND final VAL cos < 0.5)")
    print("=" * 100)
    c, d = {}, {}
    for k in PART1:
        ns = len(SEEDS[k])
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        d[k] = sum(tlc.disc(store, k, s) for s in SEEDS[k])
        rs = [get(store, k, s) for s in SEEDS[k]]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        print(f"  {ARM[k]['label']:<52} bound {c[k]:>2}/{ns}"
              + (f"   discovered {d[k]:>2}/{ns}" if k in GATED1 else " " * 18)
              + f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{ns}"
              + (f"   [{failed} FAILED]" if failed else ""))
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["CUR_ceiling_lo"])
    need = min(CEIL_MIN, nc)
    valid = c["CUR_ceiling_lo"] >= need
    print(f"  CUR_ceiling_lo binds {c['CUR_ceiling_lo']}/{nc} (needs >= {need})")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (Part 1; outcome: BOUND on the 8-key set)")
    print("#" * 100)
    verdict = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    v = dict(bound=c, discovered=d, valid=valid)
    nl = len(SEEDS["CUR_lo"])
    for kk, other, title in (("K1", "DIRECT_lo", "THE CURRICULUM BEATS TRAINING FROM SCRATCH"),
                             ("K2", "CUR_B_lo", "THE GAIN NEEDS CHANNELS")):
        no = len(SEEDS[other])
        p = fisher_greater(c["CUR_lo"], nl, c[other], no)
        sh = SEEDS[other]
        b_ = sum(bound(store, "CUR_lo", s) and not bound(store, other, s) for s in sh)
        c_ = sum(bound(store, other, s) and not bound(store, "CUR_lo", s) for s in sh)
        print(f"  {kk} {title} (Fisher one-sided): CUR_lo {c['CUR_lo']}/{nl} vs {other} {c[other]}/{no}; p = {p:.3g}")
        print(f"     exact McNemar two-sided on seeds {sh[0]}-{sh[-1]}: CUR_lo only {b_}, {other} only {c_}; p = "
              f"{mcnemar_exact(b_, c_):.3g}")
        print(f"     *** {kk}: {verdict(p)} ***")
        v.update({kk: verdict(p), kk + "_p": p, kk + "_mcnemar": [b_, c_, mcnemar_exact(b_, c_)]})
    nr = len(SEEDS["CUR_lo_R"])
    v3 = band(c["CUR_lo_R"]) if valid else "UNTESTED"
    print(f"  K3 RELIABLE: CUR_lo_R bound {c['CUR_lo_R']}/{nr} (RELIABLE >= {RELIABLE_R}, MAJORITY {MAJORITY_R}-"
          f"{RELIABLE_R - 1}, MINORITY 1-{MAJORITY_R - 1}, NEVER 0); alongside, CUR_lo {c['CUR_lo']}/{nl}: "
          f"{band(c['CUR_lo'])}")
    print(f"     *** K3: {v3} ***")
    v.update(K3=v3, K3_CUR_lo=band(c["CUR_lo"]))
    print()
    print("  READINGS:")
    rd = []
    if not valid:
        print("     (UNTESTED: no reading)")
    else:
        if v["K1"] == "SHOWN" and v["K2"] == "SHOWN" and v["K3"] == "RELIABLE":
            rd.append(("K1 SHOWN, K2 SHOWN and K3 RELIABLE", READINGS["K1+ K2+ K3 RELIABLE"]))
        if v["K1"] == "NOT SHOWN":
            rd.append(("K1 NOT SHOWN", READINGS["K1-"]))
        if v["K2"] == "NOT SHOWN":
            rd.append(("K2 NOT SHOWN", READINGS["K2-"]))
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
    print("  per seed (B = bound, D = bound and discovered, . = neither, - = not run); CUR_lo margins at the switch "
          "-> first evaluation after -> end; CUR_lo_R attempts used:")
    print(f"    seed  {'CUR_lo':>6} {'CUR_lo_R':>8} {'used':>4} {'DIRECT_lo':>9} {'CUR_B_lo':>8} {'ceiling':>7}   "
          f"CUR_lo margin switch -> +1 -> end")
    for s in SEEDS["CUR_lo"]:
        rL, rR = get(store, "CUR_lo", s), get(store, "CUR_lo_R", s)
        used = rR["trial"]["used"] if rR and rR.get("ok") else "-"
        mg = (f"{fmt(sw_margin(rL))} -> {fmt(after_margin(rL))} -> {fmt(rL['end'].get('margin'))}"
              if rL and rL.get("ok") else "--")
        mk = lambda k: mark(store, k, s) if s in SEEDS[k] else "-"
        print(f"    {s:>4}  {mk('CUR_lo'):>6} {mk('CUR_lo_R'):>8} {used!s:>4} {mk('DIRECT_lo'):>9} {mk('CUR_B_lo'):>8} "
              f"{mk('CUR_ceiling_lo'):>7}   {mg}")
    print()
    print(f"  routing persistence: of the runs routed at the switch (margin >= {ROUTED_MARGIN} at {ts}), how many "
          f"lost the routing by the first evaluation after it ({ts + ee}):")
    runs = [r for _, _, r in distinct_runs(store) if sw_margin(r) is not None]
    persistence_line("CUR_lo (lr 1e-3 throughout)", [r for _, r in ok_runs(store, "CUR_lo") if sw_margin(r) is not None],
                     ts, ee)
    persistence_line("CUR_lo + CUR_lo_R, distinct runs", runs, ts, ee)
    persistence_line("CUR_ceiling_lo", [r for _, r in ok_runs(store, "CUR_ceiling_lo") if sw_margin(r) is not None],
                     ts, ee)
    prev = load_store(prev_path) if os.path.exists(prev_path) else None
    if prev is None or not prev.get("runs"):
        print(f"    test_load_curriculum's runs: {prev_path} not found")
    else:
        pts = prev.get("meta", {}).get("sched", {}).get("t_switch", ts)
        pee = prev.get("meta", {}).get("sched", {}).get("eval_every", ee)
        pm = prev.get("meta", {})
        print(f"    test_load_curriculum on {pm.get('cpu', 'unknown')} (git {pm.get('git')}; {prev_path}):")
        for k, lab in (("CUR_A", "CUR_A (lr jump 1e-3 -> 4e-3)"), ("CUR_A_lo", "CUR_A_lo (lr 1e-3 throughout)")):
            rs = [r for kk, r in prev["runs"].items() if kk.split("|")[0] == k and r.get("ok") and sw_margin(r) is not None]
            persistence_line(f"  {lab}", rs, pts, pee, has_margin=False)
    print()
    print(f"  the immediate transfer: 8-key accuracy at the switch (step {ts}, before any 8-key training; DIRECT_lo: "
          f"at the same step, after {ts} steps on 8 keys), median [min, max], and runs already >= 0.95; phase-1 "
          f"accuracy at {tc} and {ts}; the switch margin:")
    for k in PART1:
        rs = [r for _, r in ok_runs(store, k)]
        a8 = [x for r in rs if (x := at(r["curve"], ts)) is not None]
        print(f"    {k:<14} P8@{ts} {med(a8):<22} >= 0.95: {sum(1 for x in a8 if x >= 0.95)}/{len(a8):<3} "
              f"p1@{tc} {med([x for r in rs if (x := at(r['curve1'], tc)) is not None]):<22} "
              f"p1@{ts} {med([x for r in rs if (x := at(r['curve1'], ts)) is not None]):<22}"
              + (f" margin {med([sw_margin(r) for r in rs if sw_margin(r) is not None])}" if k in GATED1 else ""))
    print()
    print(f"  transitions and plateau length (steps at 8-key accuracy {PLATEAU[0]}-{PLATEAU[1]} before the transition; "
          f"after the switch for the curriculum arms, from step 0 for DIRECT_lo):")
    for k in PART1:
        rs = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None]
        if not rs:
            print(f"    {k:<14} none bound")
            continue
        t0_ = 0 if k == "DIRECT_lo" else ts
        trs = [r["transition"] for _, r in rs]
        pls = [plateau2(r, t0_) for _, r in rs]
        print(f"    {k:<14} transition median {statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   plateau "
              f"median {statistics.median(pls):.0f} [{min(pls)}, {max(pls)}]")
        print(f"    {'':<14} " + "  ".join(f"s{s}:{r['transition']}/{plateau2(r, t0_)}" for s, r in rs))
    print()
    print("  gated runs: bound WITH routing (end margin >= 0.9) or WITHOUT; test_router_layout's classes at the "
          "switch and, for failures, at the end:")
    for k in GATED1:
        rs = ok_runs(store, k)
        w_ = [s for s, r in rs if r["transition"] is not None and routed(r)]
        wo = [s for s, r in rs if r["transition"] is not None and not routed(r)]
        cls_end = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        cls_sw = {"ROUTED": [], "STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        for s, r in rs:
            if r["transition"] is None:
                cls_end[fail_class(r)].append(s)
            if sw_margin(r) is not None:
                cls_sw["ROUTED" if routed_sw(r) else fail_class({"end": r["switch"]})].append(s)
        print(f"    {k:<14} bound with routing {len(w_)}, WITHOUT routing {len(wo)}"
              + (f" ({' '.join(f's{s}' for s in wo)})" if wo else ""))
        print(f"    {'':<14} at the switch: " + "   ".join(f"{m_} {len(x)}" for m_, x in cls_sw.items()))
        print(f"    {'':<14} failures at the end: " + "   ".join(f"{m_} {len(x)}" for m_, x in cls_end.items()))
        for m_, xs in cls_end.items():
            if xs:
                print(f"    {'':<14}   {m_:<14} " + " ".join(f"s{s}" for s in xs))
    print()
    print("  the convolution's mean |w| per lag at the end, median [min, max] (lag 2 = where a value position "
          "sees its context token):")
    for k in PART1:
        rs = ok_runs(store, k)
        b_all = [r for _, r in rs if r["transition"] is not None]
        grp = (("bound", b_all), ("not bound", [r for _, r in rs if r["transition"] is None]))
        if k in GATED1:
            grp = (("bound, routed", [r for r in b_all if routed(r)]),
                   ("bound, NOT routed", [r for r in b_all if not routed(r)]), grp[1])
        for lab, sub in grp:
            if sub:
                print(f"    {k:<14} {lab:<17} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median [min, max]:")
    for step in tlc.grad_steps_of(sched):
        for k in PART1:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>5} {k:<14} " + "   ".join(parts))
    print(f"  collapsed runs (final 8-key accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in PART1))
    print()
    p8 = load_store(p8_path) if os.path.exists(p8_path) else None
    if p8 is not None and p8.get("runs"):
        rs8 = [r for kk, r in p8["runs"].items() if kk.split("|")[0] == "A_conv8"]
        b8 = sum(1 for r in rs8 if bound_r(r))
        its = "/".join(str(x) for x in sorted({r.get("iters") for r in rs8 if r.get("iters")}))
        print(f"  DIRECT_lo (lr {LR1}, seeds {SEEDS['DIRECT_lo'][0]}-{SEEDS['DIRECT_lo'][-1]}, up to {sched['total']} "
              f"steps) bound {c['DIRECT_lo']}/{len(SEEDS['DIRECT_lo'])}; the recorded A_conv8 (lr {tps.ARM['A_conv8']['lr']}, "
              f"seeds 160-189, up to {its} steps; {p8_path}) bound {b8}/{len(rs8)}. Different seeds: descriptive only.")
    else:
        print(f"  DIRECT_lo bound {c['DIRECT_lo']}/{len(SEEDS['DIRECT_lo'])}; the recorded A_conv8 ({p8_path}) not found.")
    print()
    print("  exact McNemar two-sided (bound), per seed:")
    for k1, k2 in (("CUR_lo_R", "CUR_lo"), ("CUR_lo", "DIRECT_lo"), ("CUR_lo", "CUR_B_lo")):
        sh = [s for s in SEEDS[k1] if s in SEEDS[k2]]
        b_ = sum(bound(store, k1, s) and not bound(store, k2, s) for s in sh)
        c_ = sum(bound(store, k2, s) and not bound(store, k1, s) for s in sh)
        print(f"    {k1:<9} vs {k2:<10} ({len(sh):>2} seeds): {k1} only {b_}, {k2} only {c_}; p = {mcnemar_exact(b_, c_):.3g}")
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for k in KEYS:
                rs2 = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                b2_ = sum(1 for r in rs2 if r.get("ok") and r.get("transition") is not None)
                ch = sum(bound(store, k, s) for s in SEEDS[k])
                nh = sum(1 for s in SEEDS[k] if get(store, k, s) is not None)
                print(f"    {k:<14} bound here {ch}/{nh}   other {b2_}/{len(rs2)}   pooled {ch + b2_}/{nh + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    report_part2(store, sched16, part2)
    save_results(path, store)

    print("=" * 100)
    print(f"CURVES, PART 1 — per evaluation (every {ee} steps): phase-1 set accuracy x100, 8-key set accuracy x100, "
          f"VAL cos x100 on the 8-key probe; | = the switch after step {ts} (DIRECT_lo: the same step, no switch)")
    print("=" * 100)
    for k in PART1:
        for s, r in ok_runs(store, k):
            curve_lines2(f"{k:<14} s{s}", r, ts)
    print()
    if part2.get("ran"):
        print("=" * 100)
        print(f"CURVES, PART 2 — per evaluation (every {sched16['eval_every']} steps): the current stage's set accuracy "
              f"x100 (stages 1-2; stage 3's set is the 16-key set), 16-key set accuracy x100, VAL cos x100 on the "
              f"16-key probe; | = a stage switch (after steps {sched16['t1']} and {sched16['t2']})")
        print("=" * 100)
        for k in PART2:
            for s, r in ok_runs(store, k):
                curve_lines16(f"{k:<14} s{s}", r, sched16)
        print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


def report_part2(store, sched16, part2):
    t1, t2, ee = sched16["t1"], sched16["t2"], sched16["eval_every"]
    print("=" * 100)
    print("PART 2 — P=16, descriptive only (BOUND = transition on the 16-key curve in the last stage)")
    print("=" * 100)
    if part2.get("dropped"):
        print(f"  PART 2 DROPPED: the projection was {part2['projected_h']:.2f} h > {DROP_H:g} h on the workers; nothing "
              f"was run.")
        print()
        store["verdict"]["part2"] = dict(part2)
        return
    if not part2.get("ran"):
        print("  Part 2 was not run.")
        print()
        return
    print(f"  per seed: accuracy on each stage's set at each stage end (4-key and 8-key sets at {t1} and {t2}; the "
          f"16-key set at {t1}, {t2} and the end); margins at {t1}, {t1 + ee}, {t2}, {t2 + ee} and the end:")
    print(f"  {'arm':<14} {'seed':>4} {'transition':>10}  {'4@' + str(t1):>7} {'8@' + str(t1):>7} {'16@' + str(t1):>8} "
          f"{'4@' + str(t2):>7} {'8@' + str(t2):>7} {'16@' + str(t2):>8} {'16 end':>7}   {'m ' + str(t1):>7} "
          f"{'+1':>6} {'m ' + str(t2):>7} {'+1':>6} {'end':>6} {'VAL cos':>7}  flags")
    counts = {}
    for k in PART2:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<14} {s:>4}  NOT RUN" + (" (skipped: CUR16_ceiling bound 0)" if part2.get("skipped")
                                                      and k == "CUR16" else ""))
                continue
            if not r.get("ok"):
                print(f"  {k:<14} {s:>4}  FAILED — {r.get('error')}")
                continue
            e1, e2 = r["ends"].get(str(t1), {}), r["ends"].get(str(t2), {})
            mg = lambda x: fmt((r.get(x) or {}).get("margin"))
            if r["transition"] is not None:
                fl = ["BOUND"] + (["DISCOVERED"] if r["discovered"] else []) + ["routed" if routed(r) else "NOT routed"]
            else:
                fl = [fail_class(r)]
            print(f"  {k:<14} {s:>4} {fmt_step(r['transition']):>10}  {fmt(e1.get('4')):>7} {fmt(e1.get('8')):>7} "
                  f"{fmt(at(r['curve'], t1)):>8} {fmt(e2.get('4')):>7} {fmt(e2.get('8')):>7} {fmt(at(r['curve'], t2)):>8} "
                  f"{r['acc']:>7.4f}   {mg('st1'):>7} {mg('st1a'):>6} {mg('st2'):>7} {mg('st2a'):>6} "
                  f"{fmt(r['end'].get('margin')):>6} {r['end']['val_cos']:>7.3f}  {' '.join(fl)}")
        print()
    for k in PART2:
        rs = [get(store, k, s) for s in SEEDS[k]]
        counts[k] = sum(bound(store, k, s) for s in SEEDS[k])
        done = sum(1 for r in rs if r and r.get("ok"))
        print(f"  {ARM[k]['label']:<52} bound {counts[k]}/{len(SEEDS[k])}   completed {done}/{len(SEEDS[k])}"
              + ("   (skipped: CUR16_ceiling bound 0/3)" if k == "CUR16" and part2.get("skipped") else ""))
    if part2.get("skipped"):
        print(f"  CUR16_ceiling bound 0/{len(SEEDS['CUR16_ceiling'])}: CUR16 was skipped, as pre-registered.")
    print()
    store["verdict"]["part2"] = dict(part2, bound=counts)


def curve_lines16(tag, r, sched16):
    t1, t2 = sched16["t1"], sched16["t2"]
    ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
    sep = lambda st: " |" if st in (t1, t2) else "  "
    stg = {st: a for st, a, _ in r["curve_stage"]}
    row_s = "".join((f"{round(stg[st] * 100):>3}" if st in stg else "   ") + sep(st) for st, _, _ in r["curve"])
    row_16 = "".join(f"{round(a * 100):>3}{sep(st)}" for st, a, _ in r["curve"])
    cs = "".join((f"{round(ev[st]['val_cos'] * 100):>3}" if st in ev else " --") + sep(st) for st, _, _ in r["curve"])
    end = f"bound @{r['transition']}" if r["transition"] else "not bound"
    print(f"  {tag} st  {row_s.rstrip()}")
    print(f"  {'':<{len(tag)}} 16  {row_16}")
    print(f"  {'':<{len(tag)}} cos {cs}  -> {end}{'  DISCOVERED' if r.get('discovered') else ''}")


# ── Main ─────────────────────────────────────────────────────────────────────
def describe16(rec):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr, e = rec["transition"], rec["end"]
    mg = lambda x: fmt((rec.get(x) or {}).get("margin"))
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc16={rec['acc']:.4f} @{rec['stopped_at']:<5} "
            f"margin {mg('st1')}->{mg('st2')}->{fmt(e.get('margin'))} VALcos={e['val_cos']:.3f} conv|w|={wstr(e)}"
            f"{'  DISCOVERED' if rec['discovered'] else ''}")


def main():
    ap = argparse.ArgumentParser(description="Does the constant-lr load curriculum bind P=8 reliably? P=16 look.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's curriculum_confirm_results.json, for pooled counts")
    ap.add_argument("--prev", default=PREV_FILE,
                    help="this machine's load_curriculum_results.json (routing persistence under the lr jump)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    sched, sched16 = dict(SCHED), dict(SCHED16)
    ts = sched["t_switch"]

    print("=" * 100)
    print("Curriculum confirm: does the constant-lr load curriculum bind P=8 reliably on fresh seeds? And P=16.")
    print(f"  Part 1: phase 1 n_active=4 of P=8 (length {TASK1.L}), steps 1-{ts}; phase 2 task_for(8, 2) (length "
          f"{TASK8.L}), up to {sched['total'] - ts} more steps; lr {LR1} throughout")
    print(f"  Part 2: task_for(16, 2) (length {TASK16.L}); stages of 4, 8 and 16 of 16 keys: steps 1-{sched16['t1']}, "
          f"{sched16['t1'] + 1}-{sched16['t2']}, then up to {sched16['total'] - sched16['t2']} more; lr {LR1}; descriptive")
    print(f"  n_layer=3 decay N=256, conv 'layer' width {CONV_K}; evaluation every {sched['eval_every']}; early stop "
          f"only in the last phase/stage (DIRECT_lo: throughout)")
    for k in KEYS:
        s = SEEDS[k]
        print(f"  {ARM[k]['label']:<52} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  restart rule (CUR_lo_R): phase-1 accuracy >= {sched['a_check']} at step {sched['t_check']}, attempts "
          f"s + {STRIDE}*j, j < {R_MAX}")
    print(f"  --prev {args.prev}   torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        p8_path = verify(pool, args.prev)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head,
                             sched=sched, sched16=sched16, lr=LR1, prev=args.prev, p8=p8_path,
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on each phase's/stage's task, its "
              f"evaluations charged once per {EVAL_EVERY}; worst case = every run to its full length, every CUR_lo_R "
              f"trial needing all {R_MAX} attempts")
        print("=" * 100)
        c1 = dict(zip(PART1, pool.map(tlc.time_job, [BASE.get(k, "CUR_A") for k in PART1])))
        c2 = dict(zip(PART2, pool.map(time_job16, PART2)))
        t1, t2, T2 = sched16["t1"], sched16["t2"], sched16["total"]
        full = {k: (sched["total"] * c1[k][1] if k == "DIRECT_lo" else ts * c1[k][0] + (sched["total"] - ts) * c1[k][1])
                for k in PART1}
        full.update({k: t1 * c2[k][0] + (t2 - t1) * c2[k][1] + (T2 - t2) * c2[k][2] for k in PART2})
        for k in PART1:
            ph = (f"all 8 keys {c1[k][1] * 1000:6.1f} ms/step" if k == "DIRECT_lo" else
                  f"phase 1 {c1[k][0] * 1000:6.1f} ms/step, phase 2 {c1[k][1] * 1000:6.1f} ms/step")
            print(f"  {k:<14} {ph:<48} x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        for k in PART2:
            print(f"  {k:<14} stages {c2[k][0] * 1000:.1f} / {c2[k][1] * 1000:.1f} / {c2[k][2] * 1000:.1f} ms/step"
                  f"{'':<14} x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        trial_extra = (R_MAX - 2) * sched["t_check"] * c1["CUR_lo_R"][0]
        d_ceil = [full["CUR_ceiling_lo"]] * len(SEEDS["CUR_ceiling_lo"])
        d_rest1 = ([full[k] for k in ("CUR_lo", "DIRECT_lo", "CUR_B_lo") for _ in SEEDS[k]]
                   + [full["CUR_lo_R"] + trial_extra] * len(SEEDS["CUR_lo_R"]))
        d_rest2 = [full[k] for k in PART2 for _ in SEEDS[k]]
        mk1 = makespan(d_ceil, args.workers)
        mk2 = makespan(d_rest1 + d_rest2, args.workers)
        total_h = (mk1 + mk2) / 3600
        print(f"  CUR_ceiling_lo first: {len(d_ceil)} runs, {mk1 / 3600:.2f} h;  then Part 1's {len(d_rest1)} runs/trials "
              f"and Part 2's {len(d_rest2)} runs: serial {(sum(d_rest1) + sum(d_rest2)) / 3600:.2f} h, {mk2 / 3600:.2f} h "
              f"on {args.workers} workers;  total {total_h:.2f} h")
        part2 = dict(projected_h=total_h, dropped=total_h > DROP_H, ran=False, skipped=False)
        if part2["dropped"]:
            mk2b = makespan(d_rest1, args.workers)
            print(f"  *** the projection exceeds {DROP_H:g} h: PART 2 IS DROPPED (drop rule). Part 1 alone: "
                  f"{(mk1 + mk2b) / 3600:.2f} h ***")
            store["meta"]["projected_wall_h"] = (mk1 + mk2b) / 3600
        else:
            print(f"  within {DROP_H:g} h: Part 2 runs")
            store["meta"]["projected_wall_h"] = total_h
        store["meta"]["projected_with_part2_h"] = total_h
        store["meta"]["part2_dropped"] = part2["dropped"]
        save_results(args.results, store)
        print()

        def record(sp, rec):
            key = f"{sp['arm']}|{sp['seed']}"
            store["runs"][key] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            extra = ""
            if sp["arm"] == "CUR_lo_R" and rec.get("ok"):
                t_ = rec["trial"]
                extra = f"  attempts {t_['used']} (continued {t_['continued']})" + ("  CUR_lo reused" if t_["reused"] else "")
            secs = rec.get("trial", {}).get("secs", rec.get("secs", 0.0)) if sp["arm"] == "CUR_lo_R" else rec.get("secs", 0.0)
            desc = describe16(rec) if sp["arm"] in PART2 else describe(rec)
            print(f"   {sp['arm']:<14} seed {sp['seed']}  {desc}{extra}  {secs:.0f}s  (elapsed {el:.1f}m)")

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
            sch = sched16 if k in PART2 else sched
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<14} seed {s}  cached")
                else:
                    out.append(spec(k, s, sch))
            return out

        print("=" * 100)
        print(f"RUNS — CUR_ceiling_lo first ({len(SEEDS['CUR_ceiling_lo'])} runs)")
        print("=" * 100)
        run_specs(todo("CUR_ceiling_lo"))
        n_ceil = sum(bound(store, "CUR_ceiling_lo", s) for s in SEEDS["CUR_ceiling_lo"])
        need = min(CEIL_MIN, len(SEEDS["CUR_ceiling_lo"]))
        print(f"   CUR_ceiling_lo bound {n_ceil}/{len(SEEDS['CUR_ceiling_lo'])} (needs >= {need})")
        print()
        if n_ceil < need:
            print("   *** the ceiling failed: nothing else is run ***\n")
            report_stop(store, sched, time.time() - t0, args.results)
            return

        def ceiling16_known():
            return all(get(store, "CUR16_ceiling", s) is not None for s in SEEDS["CUR16_ceiling"])

        def cur16_step():
            n16 = sum(bound(store, "CUR16_ceiling", s) for s in SEEDS["CUR16_ceiling"])
            print(f"   CUR16_ceiling bound {n16}/{len(SEEDS['CUR16_ceiling'])}"
                  + ("" if n16 >= CEIL16_MIN else ": CUR16 is skipped, as pre-registered"))
            if n16 < CEIL16_MIN:
                part2["skipped"] = True
                return []
            return todo("CUR16")

        def after(sp):
            if sp["arm"] == "CUR_lo":
                nxt = r_step(store, sp["seed"], sched, args.force, record)
                return [] if nxt is None else [nxt]
            if sp["arm"] == "CUR16_ceiling" and ceiling16_known():
                return cur16_step()
            return []

        n_rest = sum(len(SEEDS[k]) for k in PART1 if k != "CUR_ceiling_lo")
        part2["ran"] = not part2["dropped"]
        print("=" * 100)
        print(f"RUNS — the rest ({n_rest} Part 1 runs/trials" + ("" if part2["dropped"] else
              f", and Part 2's {len(SEEDS['CUR16_ceiling'])} + {len(SEEDS['CUR16'])} runs, CUR16_ceiling first") +
              "; each CUR_lo_R trial is queued, first, when its CUR_lo run is known)")
        print("=" * 100)
        first = []
        if part2["ran"]:
            first = todo("CUR16_ceiling")
            if ceiling16_known():
                first = cur16_step() + first
        lo = todo("CUR_lo")
        for s in SEEDS["CUR_lo"]:
            if get(store, "CUR_lo", s) is not None and not args.force:
                nxt = r_step(store, s, sched, args.force, record)
                if nxt is not None:
                    lo.insert(0, nxt)
        run_specs(first + lo + todo("DIRECT_lo") + todo("CUR_B_lo"), on_done=after)
        print()

    report(store, sched, sched16, time.time() - t0, args.results, args.also, args.prev, p8_path, part2)


def r_step(store, s, sched, force, record):
    """CUR_lo_R seed s once CUR_lo seed s is known: reuse CUR_lo's record if its attempt passed,
    else a trial from attempt 1 (from attempt 0 if CUR_lo's run failed)."""
    if get(store, "CUR_lo_R", s) is not None and not force:
        print(f"   {'CUR_lo_R':<14} seed {s}  cached")
        return None
    rL = get(store, "CUR_lo", s)
    if rL and rL.get("ok") and rL.get("passed"):
        rec = dict(rL, trial=dict(start=0, attempts=[dict(j=0, seed=s, acc_check=rL["acc_check"], passed=True)],
                                  used=1, continued=s, reused=True, secs=0.0))
        record(dict(arm="CUR_lo_R", seed=s), rec)
        return None
    return spec("CUR_lo_R", s, sched, trial=True, start=1 if (rL and rL.get("ok")) else 0)


def finish_trial(store, sp, rec):
    """Prepend attempt 0 (CUR_lo's run) to a trial that started at attempt 1."""
    if rec.get("ok") and sp["start"] == 1:
        rL = get(store, "CUR_lo", sp["seed"])
        a0 = dict(j=0, seed=sp["seed"], acc_check=rL.get("acc_check"), passed=rL.get("passed"))
        rec["trial"]["attempts"] = [a0] + rec["trial"]["attempts"]
    elif rec.get("ok") is False and "trial" not in rec:
        rec["trial"] = dict(start=sp["start"], attempts=[], used=None, continued=None, reused=False, secs=0.0)
    return rec


if __name__ == "__main__":
    main()
