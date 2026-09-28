#!/usr/bin/env python
"""
test_load_curriculum.py — does the gate, having found the stream routing on an easy load (4
active keys), keep it and bind when the load rises to 8 keys, where routing from scratch
mostly fails? Everything below is fixed before any result.

Run it directly:

    python test_load_curriculum.py --p8 results/X/p_scaling_results.json
    python test_load_curriculum.py --p8 ... --also results/L/load_curriculum_results.json

BACKGROUND
test_p_scaling, both machines: seeds 160-189, S=2, P=8, lr 4e-3, conv width 4, 24000 steps.
L ran merge 3b5e041, whose code is identical to 745564f.
- Bound, X / L:
  - A_conv8 4/30 / 8/30
  - B_conv8 2/30 / 1/30
  - B_wide_conv8 1/16 / 4/16
  - ceiling_conv8 8/8 / 8/8
- Claims: P1 NOT SHOWN on X (p = 0.335), SHOWN on L (p = 0.013). P2 and P3 NOT SHOWN on both.
  M1-P8 SHOWN on both.
- Pooled, post hoc: A_conv8 vs B_conv8 was 11 vs 2 discordant (p = 0.02), but only 5 of
  A_conv8's 12 binders routed by stream. On the same seeds at P=4 (test_conv_lr), A_conv4
  bound 38/60 and B_conv4 26/60.
- Unbound runs sit at ~0.50: keys bound, streams confused. The restart check passed 0/30
  A_conv8 runs on X.
- At lr 1e-3 and P=4, most gated binders route by stream (test_short_conv: 39 of 51); at 4e-3
  most do not (test_conv_lr: 18 of 50).
- Hypothesis: the stream routing does not depend on which keys are present. If found at a low
  load, it should survive the switch to a high load.
- Seed-level outcomes differ between machines; the verdict is per machine.

TASK KNOB
BindTask gained n_active (default None = P, bit-identical: CHECK 69). Each sequence then draws
n_active distinct keys uniformly from the P keys and builds its S*n_active triples and its
query from those keys only, with the grouped layout rule unchanged (CHECK 70). The vocabulary
stays P's, so one model trains on both loads.

CURRICULUM
- Phase 1: n_active=4 of P=8, lr 1e-3, steps 1-4800.
- Phase 2: n_active=8 (the plain task_for(8, 2)), lr 4e-3, up to 24000 more steps (total
  28800). The same optimizer carries over; only the lr in its param groups changes (CHECK 71).
- Evaluation every 1200 steps on BOTH held-out sets: the phase-1 set (n_active=4, its own
  EVAL_SEED stream) and the P=8 set (eval_batch(task_for(8, 2)), exactly test_p_scaling's).
- BOUND = the transition on the P=8 curve, counted in phase 2 only. Early stopping only in
  phase 2, on P=8 accuracy.
- The gate statistics (margin, VAL cos, eta^2 classes) are recorded at step 4800 (the switch)
  and at the end, on the P=8 probe batch (the inputs the claims are about).
- The phase-2 budget of 24000 steps matches test_p_scaling's; the 4800 phase-1 steps are extra
  and cheap. The recorded P=8 runs had mostly plateaued at 0.50 by step ~3600.
- Mechanism: onset_run gained task_at(step) and lr_at(step), and test_router_reliability.run_one
  gained grad_steps and run_kw (onset_run keywords passed through); all inert at their
  defaults (CHECK 69). The rest is test_p_scaling's run path.

ARMS (conv 'layer' width 4, identity init; the substrate of test_p_scaling)
- CUR_A: arm A + conv, curriculum, seeds 160-189 (30). Paired with the recorded A_conv8: same
  seeds, same vocabulary, so the same initial parameters (CHECK 72).
- CUR_A_R: CUR_A with the restart rule in phase 1, seeds 160-189 (30).
  - At phase-1 step 2400, continue if phase-1 held-out accuracy >= 0.6 (test_router_
    reliability's rule); otherwise restart with attempt seed s + 1000*j (j = 1..4), for at most
    5 attempts. If no attempt passes, the last attempt continues.
  - Attempt seeds 1160-1189, 2160-2189, 3160-3189, 4160-4189: disjoint from every earlier seed.
  - Attempt 0 is CUR_A's run: its record is reused when it passes, rather than re-run (CHECK 73).
- CUR_B: single channel + conv, curriculum, seeds 160-179 (20). Paired with the recorded B_conv8.
- CUR_A_lo: CUR_A, but phase 2 at lr 1e-3, seeds 160-169 (10). Descriptive only.
- CUR_ceiling: perfect gate + conv, curriculum, seeds 160-164 (5). Validity.

ORDER OF EXECUTION
CUR_ceiling runs first. If it binds fewer than 4/5, the test stops: it prints the validity
failure and reports every claim UNTESTED without running the rest.

VALIDITY AND PRE-REGISTERED CLAIMS (outcome: BOUND on the P=8 set)
- VALID: CUR_ceiling binds >= 4/5, AND the pairing CHECK (72) passes. Otherwise UNTESTED.
- C1 THE CURRICULUM HELPS THE GATE: CUR_A vs the recorded A_conv8, same seeds. Exact McNemar,
  one-sided (CUR_A higher).
- C2 THE GAIN NEEDS CHANNELS: CUR_A (seeds 160-179) vs CUR_B. Fisher one-sided (CUR_A higher);
  McNemar printed alongside.
- C3 ROUTING TRANSFERS: over the distinct runs of CUR_A and CUR_A_R (a reused attempt 0 counts
  once; abandoned attempts are not runs), runs routed at the switch (margin >= 0.9 at step
  4800) bind at P=8 more often than runs not routed: Fisher one-sided. UNTESTABLE if either
  group has fewer than 3 runs.
- C4 A RELIABLE RECIPE: CUR_A_R's band: RELIABLE >= 27/30, MAJORITY 15-26, MINORITY 1-14,
  NEVER 0.
- C1-C3 are SHOWN if p < 0.05, else NOT SHOWN.
- Readings, printed verbatim where they apply:
  - C1 SHOWN and C3 SHOWN: "routing found at a low load carries over to a high load: learn the
    partition where it is easy."
  - C1 SHOWN and C2 NOT SHOWN: "the curriculum helps, but a single channel gains about as much:
    not a channel effect."
  - C1 NOT SHOWN: "the curriculum did not rescue the gate at P=8."
  - C4 RELIABLE: "curriculum + restarts binds P=8 reliably."
Fisher and McNemar (two-sided) are test_router_confirm's; the one-sided McNemar is test_conv_lr's.

DIAGNOSTICS, not part of the verdict
- Per seed: the recorded P=8 outcome; CUR_A / CUR_A_R / CUR_B / CUR_A_lo outcomes; margin at
  the switch and at the end; attempts used (CUR_A_R).
- Routing persistence: of the runs routed at the switch, how many are still routed at the end,
  split by bound and not.
- Phase-1 accuracy at 2400 and 4800, and P=8 accuracy at the switch.
- Transitions after the switch, and plateau lengths (phase-2 steps at P=8 accuracy 0.45-0.55
  before the transition).
- test_p_scaling's diagnostics: failure classes (at the switch and at the end), the
  convolution's lag weights by outcome, and gradient norms at steps 1, 10, 100, 4801 and 4810.
- CUR_A_lo vs CUR_A per seed (the phase-2 lr question); CUR_B vs the recorded B_conv8 per seed.
- --also pooled counts.

CHECKS (printed before any training): test_p_scaling's verification (which runs every
earlier test's; its CHECK 64 pairs with the conv_lr file named in the --p8 file's meta), then
  69 n_active=None (and n_active=P) is bit-identical to 57479b8's make_batch on every earlier
     task (attributes, batches, generator states); onset_run's task_at and lr_at and run_one's
     grad_steps and run_kw are inert at their defaults and explicit, vs 57479b8's modules;
  70 n_active=4 of P=8, over 10,000 sequences: 4 distinct keys per sequence, each key used
     with frequency 0.5 +- 0.02, every used (stream, key) pair once, the grouped structure,
     the query answered by its own stream's value, token ids within task_for(8, 2)'s vocabulary;
  71 the switch happens exactly at step 4800 (hooks): lr 1e-3 -> 4e-3, batches from 4 to 8
     active keys, Adam's state carried over; no early stop in phase 1 even with every held-out
     accuracy forced to 1.0 (evaluate stubbed; a scaled schedule);
  72 the pairing: the --p8 file exists, its CPU matches, it holds every recorded A_conv8
     (160-189) and B_conv8 (160-179) run; curriculum models at step 0 equal the recorded arms'
     for seeds 160 and 161; re-running A_conv8 seed 160 for 1200 steps reproduces its curve
     (recorded, not asserted: a failure makes the claims UNTESTED);
  73 the restart rule reads only the phase-1 accuracy at 2400; attempt seeds are s + 1000*j,
     distinct and disjoint from every earlier seed; synthetic pass/fail curves; when attempt 0
     passes its record equals CUR_A's run of the same seed;
  74 a worker's run is bit-identical to the same run in the main process (CUR_A and CUR_B,
     through the switch, 6000 steps).

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). CUR_A_R's trials are queued as soon as their CUR_A run is known. Results persist
atomically to load_curriculum_results.json (gitignored; copied into results/X/ after the run).

RESULT (full run, 95/95 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz,
4 workers x 1 thread; projection 11.83 h worst case; paired with results/X/p_scaling_results.json;
no --also file; machine X, commit b7c18f0). The container restarted after 94 of 95 runs (216.9
min after verification); the same command was re-run on the same results file, which re-ran the
full verification (all passed) and only the missing run, CUR_A_lo seed 169 (24.2 min). Runs are
deterministic per seed, so the cached records stand. meta.started is the first start; meta.resumed
records the second.
  VALID: CUR_ceiling bound 5/5 and the pairing CHECK (72) passed.
  C1 THE CURRICULUM HELPS THE GATE: SHOWN. CUR_A 20/30 vs the recorded A_conv8 4/30 on the same
     seeds; CUR_A only 16, A_conv8 only 0; one-sided McNemar p = 1.53e-05.
  C2 THE GAIN NEEDS CHANNELS: SHOWN. CUR_A 14/20 (seeds 160-179) vs CUR_B 4/20; Fisher one-sided
     p = 0.00182; McNemar on the pairs: CUR_A only 11, CUR_B only 1, p = 0.00635.
  C3 ROUTING TRANSFERS: SHOWN. Over 38 distinct runs of CUR_A and CUR_A_R, runs routed at the
     switch bound 27/29, runs not routed 1/9; Fisher one-sided p = 7.79e-06.
  C4 A RELIABLE RECIPE: RELIABLE. CUR_A_R bound 28/30.
  The pre-registered readings: C1 SHOWN and C3 SHOWN: "routing found at a low load carries over
  to a high load: learn the partition where it is easy." C4 RELIABLE: "curriculum + restarts
  binds P=8 reliably."

  Diagnostics (not part of the verdict):
  - Much of the transfer is immediate: at the switch, before any P=8 training, P=8 held-out
    accuracy was >= 0.95 in 16/30 CUR_A runs (median 0.952), and 17 of CUR_A's 20 binders held
    from the first phase-2 evaluation (step 6000); the other 3 bound at 14400 (s166), 21600
    (s177) and 24000 (s183). CUR_ceiling was >= 0.95 at the switch in 3/5 runs.
  - The restart rule: 22/30 CUR_A runs passed the phase-1 check (>= 0.6 at 2400); every CUR_A
    binder was among them (20 of the 22; s175 and s186 passed and did not bind). The 8 restarted
    seeds passed at attempt 2 (161, 170, 171) or 3 (163, 169, 181, 187, 188) and all 8 bound; no
    trial needed the last-attempt fallback. CUR_A_R vs CUR_A: 8 vs 0, McNemar p = 0.0078.
  - Routing at the switch was the gate: routed at the switch 29/38 distinct runs. Of the 27
    routed binders, 22 were still routed at the end; 5 lost the routing in phase 2 (end margin
    ~0: CUR_A s166, s177, s182; CUR_A_R attempts 1170 and 2181) yet stayed bound, with the
    single-channel lag-2 signature (unrouted binders lag 2 median 0.158-0.196, routed 0.049).
    The one binder not routed at the switch was CUR_A s172 (bound at 6000, unrouted). CUR_A's
    10 failures: KEY 5, POSITION 5 at the end; all 9 of CUR_A's runs not routed at the switch
    were POSITION-split there.
  - Phase 2 at lr 1e-3 (CUR_A_lo) bound 9/10 and every binder ended routed (CUR_A at 4e-3 on the
    same seeds: 7/10; CUR_A_lo only 2, CUR_A only 0). At 4e-3 some routed binders drift to
    unrouted binding after the switch; at 1e-3 none did.
  - A single channel gains little from the curriculum: CUR_B 4/20 vs the recorded B_conv8 1/20
    (4 vs 1, p = 0.375); its phase-1 accuracy stayed at the ~0.5 plateau (median 0.49 at 4800),
    and its binders came late (median 23400, plateau median 12600 steps).
  - Unbound runs end near the 0.50 plateau (CUR_A 0.33-0.53, CUR_B 0.29-0.51); none collapsed.
"""

import argparse
import inspect
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

import test_binding_capacity as tbc
from test_binding_capacity import BindTask
import test_binding_onset as tbo
from test_binding_onset import onset_run, transition, time_per_step, fmt_step, EVAL_EVERY
import test_binding_recipe as trec
import test_multilayer_binding as tmb
import test_channel_binding as tcb
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater, mcnemar_exact
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, make_fn, get, count, load_store, init_worker, makespan, cpu_model,
    same_weights, save_results, DISC_COS, COLLAPSE, TIME_STEPS,
)
from test_router_discovery import load_at, load_legacy_pair
import test_router_reliability as trr
from test_router_reliability import run_one, strip_all, med, T_CHECK, A_CHECK
from test_router_layout import routing_stats, fail_class, ETA_GROUPS
from test_readout_path import GRAD_STEPS
import test_short_conv as tsc
from test_short_conv import conv_stats, conv_grad_norms, routed, wstr, CONV_K, ROUTED_MARGIN, PLATEAU
import test_conv_lr as tcl
from test_conv_lr import mcnemar_greater, bound_r
import test_p_scaling as tps

# ── Settings (fixed before any result) ───────────────────────────────────────
T_SWITCH = 4800
PHASE2 = 24000
TOTAL = T_SWITCH + PHASE2                  # 28800
LR1, LR2 = SUB_LR, tcl.LR4                 # 1e-3, 4e-3
N_ACTIVE = 4
R_MAX = 5
STRIDE = 1000                              # attempt seed s + 1000*j
CEIL_MIN = 4                               # VALID needs CUR_ceiling >= 4/5
C3_MIN = 3                                 # C3 is UNTESTABLE below 3 runs in either group
ALPHA = 0.05
RELIABLE_R, MAJORITY_R = 27, 15            # C4 bands, of 30
LEGACY_SHA = "57479b8"                     # head before this test's changes
EARLIER_SEEDS = set(range(200)) | set(range(1000, 1150))   # every seed used by earlier tests
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "load_curriculum_results.json"
P8_FILE = "p_scaling_results.json"

TASK8 = tps.TASK8                          # test_channel_binding.task_for(8, 2)
TASK1 = BindTask(8, S=2, n_vals=16, n_q=1, n_active=N_ACTIVE)
REAL = dict(t_switch=T_SWITCH, total=TOTAL, eval_every=EVAL_EVERY, t_check=T_CHECK, a_check=A_CHECK)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ARMS = [
    dict(tps.ARM["ceiling_conv8"], key="CUR_ceiling", lr2=LR2,
         label="CUR_ceiling  perfect gate + conv, curriculum"),
    dict(tps.ARM["A_conv8"], key="CUR_A", lr2=LR2, label="CUR_A        arm A + conv, curriculum"),
    dict(tps.ARM["A_conv8"], key="CUR_A_R", lr2=LR2,
         label="CUR_A_R      CUR_A + restarts in phase 1"),
    dict(tps.ARM["B_conv8"], key="CUR_B", lr2=LR2,
         label="CUR_B        single channel + conv, curriculum"),
    dict(tps.ARM["A_conv8"], key="CUR_A_lo", lr2=LR1,
         label="CUR_A_lo     CUR_A, phase 2 at lr 1e-3"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
GATED = tuple(k for k in KEYS if ARM[k]["gate"] != "none")
SEEDS = dict(CUR_ceiling=tuple(range(160, 165)), CUR_A=tuple(range(160, 190)),
             CUR_A_R=tuple(range(160, 190)), CUR_B=tuple(range(160, 180)),
             CUR_A_lo=tuple(range(160, 170)))
RECORDED = dict(CUR_ceiling="ceiling_conv8", CUR_A="A_conv8", CUR_A_R="A_conv8",
                CUR_B="B_conv8", CUR_A_lo="A_conv8")          # curriculum arm -> recorded P=8 arm
READINGS = {
    "C1+ C3+": "routing found at a low load carries over to a high load: learn the partition "
               "where it is easy.",
    "C1+ C2-": "the curriculum helps, but a single channel gains about as much: not a channel "
               "effect.",
    "C1-": "the curriculum did not rescue the gate at P=8.",
    "C4 RELIABLE": "curriculum + restarts binds P=8 reliably.",
}


def band(c):
    return ("RELIABLE" if c >= RELIABLE_R else "MAJORITY" if c >= MAJORITY_R
            else "MINORITY" if c >= 1 else "NEVER")


def attempt_seeds(s):
    return [s + STRIDE * j for j in range(R_MAX)]


# ── The curriculum run ───────────────────────────────────────────────────────
def cur_stats_fn(t_switch):
    """test_short_conv.conv_stats, plus test_router_layout.routing_stats at the switch (it is
    already in conv_stats at step 1200 and at the end)."""
    def stats(model, task, probe, step):
        out = conv_stats(model, task, probe, step)
        if step == t_switch and model.n_ch == 2:
            out.update(routing_stats(model, task, probe))
        return out
    return stats


def grad_steps_of(sched):
    return tuple(GRAD_STEPS) + (sched["t_switch"] + 1, sched["t_switch"] + 10)


def run_cur(a, seed, sched, decide=None, abandon=False, keep=None):
    """One curriculum run: test_router_reliability.run_one on task_for(8, 2) (the P=8 held-out
    set, probe, statistics and early stop, as test_p_scaling), with phase 1's batches and lr up
    to the switch. check() evaluates the phase-1 set at every evaluation; at t_check it gives
    decide(step, phase-1 accuracy), and with abandon=True a failing attempt stops there."""
    ts = sched["t_switch"]
    lr2 = a["lr2"]
    data1 = eval_batch(TASK1)
    info = dict(curve1=[])

    def check(m, step, data):
        acc, el = tbo.evaluate(m, TASK1, data1)
        info["curve1"].append([step, acc, el])
        if decide is not None and step == sched["t_check"]:
            info["acc_check"] = acc
            info["passed"] = bool(decide(step, acc))
            if abandon and not info["passed"]:
                raise trr.Abandon(f"phase-1 held-out accuracy {acc:.4f} at step {step}")

    rec = run_one(a, seed, sched["total"], sched["eval_every"], check=check, keep=keep, task=TASK8,
                  stats_fn=cur_stats_fn(ts), grad_fn=conv_grad_norms, lr=LR1,
                  builder=tcl.builder_for(a), grad_steps=grad_steps_of(sched),
                  run_kw=dict(task_at=lambda step: TASK1 if step <= ts else TASK8,
                              lr_at=lambda step: LR1 if step <= ts else lr2,
                              early_stop_after=ts))
    rec.update(curve1=info["curve1"], acc_check=info.get("acc_check"), passed=info.get("passed"))
    if rec["ok"]:
        rec["transition_full"] = rec["transition"]
        rec["transition"] = transition([c for c in rec["curve"] if c[0] > ts])
        rec["discovered"] = bool(rec["transition"] is not None and rec["val_cos"] < DISC_COS)
        rec["switch"] = next((x for x in rec["stats"] if x["step"] == ts), None)
    return rec


def run_trial(seed, sched, start=1, decide=None):
    """CUR_A_R from attempt `start`: attempts s + 1000*j in order; a failing attempt is
    abandoned at t_check, except the last (j = R_MAX-1), which continues."""
    decide = trr.make_decide(sched["t_check"], sched["a_check"]) if decide is None else decide
    ts0 = time.time()
    attempts = []
    for j in range(start, R_MAX):
        sj, last = seed + STRIDE * j, j == R_MAX - 1
        rec = run_cur(ARM["CUR_A_R"], sj, sched, decide=decide, abandon=not last)
        attempts.append(dict(j=j, seed=sj, acc_check=rec.get("acc_check"), passed=rec.get("passed")))
        if rec.get("passed") is False and not last:
            continue
        rec["trial"] = dict(start=start, attempts=attempts, used=j + 1, continued=sj,
                            reused=False, secs=time.time() - ts0)
        return rec
    raise AssertionError("unreachable")


def run_job(sp):
    sched = sp["sched"]
    decide = trr.make_decide(sched["t_check"], sched["a_check"])
    if sp.get("trial"):
        return run_trial(sp["seed"], sched, start=sp["start"], decide=decide)
    return run_cur(ARM[sp["arm"]], sp["seed"], sched, decide=decide)


def spec(key, seed, sched, trial=False, start=1):
    sp = dict(arm=key, seed=seed, sched=dict(sched))
    if trial:
        sp.update(trial=True, start=start)
    return sp


def time_job(key):
    """ms/step on each phase's task (test_conv_lr.time_job's recipe), each charged with both
    evaluations once per EVAL_EVERY."""
    a = ARM[key]
    mk = tcl.builder_for(a)(a, 0)
    out = []
    for task in (TASK1, TASK8):
        per = time_per_step(task, mk, steps=TIME_STEPS)
        m, data = mk(), eval_batch(task)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        ev = time.time() - t0
        out.append((max(per - ev / TIME_STEPS, 1e-6), ev))
    ev_both = (out[0][1] + out[1][1]) / EVAL_EVERY
    return out[0][0] + ev_both, out[1][0] + ev_both


# ── The pairing (CHECK 72) ───────────────────────────────────────────────────
def p8_pairing(pool, path):
    info = dict(path=path, exists=os.path.exists(path))
    if not info["exists"]:
        info.update(cpu_match=False, complete=False, params=False, reproduced=False, ok=False)
        return info
    p8 = load_store(path)
    info["cpu"], info["git"] = p8.get("meta", {}).get("cpu"), p8.get("meta", {}).get("git")
    info["cpu_match"] = info["cpu"] == cpu_model()
    need = [("A_conv8", s) for s in range(160, 190)] + [("B_conv8", s) for s in range(160, 180)]
    have = [bool((r := get(p8, k, s)) and r.get("ok")) for k, s in need]
    info["n_recorded"], info["n_needed"], info["complete"] = sum(have), len(need), all(have)
    params = []
    for cur, recd in (("CUR_A", "A_conv8"), ("CUR_B", "B_conv8")):
        for s in (160, 161):
            k1, k2 = {"at": -1}, {"at": -1}
            run_cur(ARM[cur], s, dict(REAL, total=0), keep=k1)
            ra = tps.ARM[recd]
            run_one(ra, s, 0, task=TASK8, stats_fn=conv_stats, grad_fn=conv_grad_norms, lr=ra["lr"],
                    builder=tcl.builder_for(ra), keep=k2)
            params.append((cur, recd, s, same_weights(k1["model"], k2["model"])))
    info["params_list"] = params
    info["params"] = all(x[3] for x in params)
    rec = get(p8, "A_conv8", 160)
    r = pool.submit(tps.run_job, tps.spec("A_conv8", 160, EVAL_EVERY)).result()
    now = json.loads(json.dumps(r["curve"]))
    old = (rec or {}).get("curve", [])[:len(now)]
    info["A_conv8_curve"] = (now, old)
    info["reproduced"] = bool(rec and rec.get("ok") and r["ok"] and now and now == old)
    info["ok"] = info["cpu_match"] and info["complete"] and info["params"] and info["reproduced"]
    return info


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool, p8_path):
    p4_path = tps.P4_FILE
    if os.path.exists(p8_path):
        p4_path = load_store(p8_path).get("meta", {}).get("p4") or p4_path
    print("test_p_scaling.py's verification (which runs test_conv_lr's, and so on down to")
    print(f"test_multilayer_binding's); its CHECK 64 pairs with {p4_path}, the conv_lr file named")
    print("in the --p8 file's meta:")
    tps.verify(pool, p4_path)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 69 the new knobs are inert at their defaults, vs {LEGACY_SHA}'s modules:")
    leg = load_at(LEGACY_SHA, "test_binding_capacity.py", "test_binding_capacity_legacy")
    earlier = ([(f"test_binding_capacity P={P}", tbc.BindTask(P)) for P in tbc.P_LIST]
               + [("test_multilayer_binding S=1", tmb.BindTask(4, S=1, n_vals=tmb.N_VALS, n_q=tmb.N_Q))]
               + [("test_binding_onset PRIMARY", tbo.PRIMARY), ("test_binding_onset SECONDARY", tbo.SECONDARY)]
               + [(f"test_binding_recipe P={P}", trec.task_for(P)) for P in (4, 8, 16)]
               + [(f"test_channel_binding P={P} S={S}", tcb.task_for(P, S)) for P in tcb.P_LIST for S in (1, 2)]
               + [("test_router_* TASK", TASK)]
               + [(f"test_router_layout {lay}", BindTask(4, S=2, n_vals=16, n_q=1, layout=lay))
                  for lay in tbc.LAYOUTS])
    good = True
    for name, t in earlier:
        L = leg.BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q, layout=t.layout)
        L.key = t.key
        variants = [BindTask(t.P, S=t.S, n_vals=t.n_vals, n_q=t.n_q, layout=t.layout, n_active=na)
                    for na in (None, t.P)]
        for v_ in variants:
            v_.key = t.key
        attrs = all(vars(x) == vars(L) for x in [t] + variants) and t.n_active is None
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
    print(f"         (a) {len(earlier)} earlier tasks (S=1 and 2, n_q=1 and 4, P=4-32, all layouts): attributes, "
          f"batches and generator")
    print(f"             states with n_active=None (implicit and explicit) and n_active=P equal the old ones: {good}")
    leg_ml, leg_on = load_legacy_pair(LEGACY_SHA)
    data = eval_batch(TASK)
    good_b = True
    for key, a in (("A", tsc.ARM["A"]), ("B_conv", tsc.ARM["B_conv"])):
        m0, c0, _ = leg_on.onset_run(TASK, lambda: leg_ml.build(TASK, a, ARCH, 3), 3, max_iters=60,
                                     eval_every=20, data=data, early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(TASK, make_fn(a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR)
        m2, c2, _ = onset_run(TASK, make_fn(a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR, task_at=lambda step: TASK,
                              lr_at=lambda step: SUB_LR)
        g = same_weights(m0, m1) and same_weights(m0, m2) and c0 == c1 == c2
        good_b &= g
        print(f"         (b) onset_run {key:<6} 60 steps: weights equal (default, explicit task_at/lr_at) "
              f"{same_weights(m0, m1)}, {same_weights(m0, m2)}   curves equal {c0 == c1 == c2}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    leg_trr = load_at(LEGACY_SHA, "test_router_reliability.py", "test_router_reliability_legacy")
    for key in ("A_conv", "B_conv"):
        a = tsc.ARM[key]
        kw = dict(eval_every=20, task=TASK, stats_fn=conv_stats, grad_fn=conv_grad_norms)
        r0 = leg_trr.run_one(a, 3, 60, **kw)
        r1 = run_one(a, 3, 60, **kw)
        r2 = run_one(a, 3, 60, grad_steps=GRAD_STEPS, run_kw={}, **kw)
        g = strip_all(r0) == strip_all(r1) == strip_all(r2) and r1["ok"]
        good_b &= g
        print(f"         (c) run_one {key:<6} 60 steps: records equal (default, explicit grad_steps/run_kw) "
              f"{g}  -> {'UNCHANGED' if g else 'CHANGED'}")
    good &= good_b
    ok &= good
    print()

    N = 10_000
    print(f"CHECK 70 n_active={N_ACTIVE} of P=8, over {N:,} sequences (generator seed 70):")
    t = TASK1
    S, P, A = t.S, t.P, t.n_active
    tok, qs = t.make_batch(N, torch.Generator().manual_seed(70))
    j = torch.arange(S * A)
    ctx, key, val = tok[:, 3 * j], tok[:, 3 * j + 1] - S, tok[:, 3 * j + 2]
    act = torch.zeros(N, P, dtype=torch.long).scatter_(1, key, 1)
    n_keys = act.sum(1)
    four = bool((n_keys == A).all())
    freq = act.float().mean(0)
    freq_ok = bool(((freq - A / P).abs() <= 0.02).all())
    pairs = ctx * P + key
    once = bool((torch.sort(pairs, 1).values.diff(dim=1) != 0).all())
    per_key = torch.zeros(N, P, dtype=torch.long).scatter_add_(1, key, torch.ones_like(key))
    both = bool((per_key[act.bool()] == S).all())
    grouped = bool(((key[:, 0::2] == key[:, 1::2]) & (ctx[:, 0::2] != ctx[:, 1::2])).all())
    vv = val.view(N, A, S)
    distinct = bool((vv[..., 0] != vv[..., 1]).all())
    n = S * A
    qc, qk, qv = tok[:, 3 * n], tok[:, 3 * n + 1] - S, tok[:, 3 * n + 2]
    hit = (ctx == qc[:, None]) & (key == qk[:, None])
    answered = (bool((hit.sum(1) == 1).all()) and bool((val[hit] == qv).all())
                and bool((qc == qs.reshape(-1)).all()))
    in_vocab = bool((tok >= 0).all()) and bool((tok < TASK8.vocab).all()) and t.vocab == TASK8.vocab
    lens = (t.L == 3 * (S * A + 1), t.qpos == [3 * S * A + 1])
    good = four and freq_ok and once and both and grouped and distinct and answered and in_vocab and all(lens)
    print(f"         {A} distinct keys in every sequence: {four}   key frequencies "
          f"{[round(x, 3) for x in freq.tolist()]} (0.5 +- 0.02): {freq_ok}")
    print(f"         every used (stream, key) pair once: {once} and both streams per active key: {both}   "
          f"grouped structure: {grouped}   S distinct values per key: {distinct}")
    print(f"         the query answered by its own stream's value: {answered}   token ids in [0, {TASK8.vocab}) "
          f"(task_for(8, 2)'s vocabulary): {in_vocab}")
    print(f"         length {t.L}, query position {t.qpos} (P=8: length {TASK8.L}, query position {TASK8.qpos}); "
          f"e.g. {tps_decode(tok[0], t)}")
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 71 the switch at step {T_SWITCH} (CUR_B seed 1, the real schedule run {T_SWITCH + 2} steps; "
          f"hooks on every step):")
    rec_opt, widths, keysets = [], [], {}
    cnt = {"fw": 0}

    def pre_opt(opt, args, kw):
        p0 = opt.param_groups[0]["params"][0]
        st = opt.state.get(p0, {}).get("step")
        rec_opt.append((id(opt), sorted({g_["lr"] for g_ in opt.param_groups}),
                        None if st is None else float(st)))

    def pre_fw(mod, inp):
        if type(mod).__name__ == "MultiBDH" and mod.training:
            cnt["fw"] += 1
            x = inp[0]
            widths.append(x.shape[1])
            if cnt["fw"] in (T_SWITCH - 1, T_SWITCH, T_SWITCH + 1, T_SWITCH + 2):
                j_ = torch.arange((x.shape[1] + 1) // 3 - 1)
                ks = x[:, 3 * j_ + 1]
                keysets[cnt["fw"]] = sorted({len(set(row.tolist())) for row in ks})
    h1 = register_optimizer_step_pre_hook(pre_opt)
    h2 = register_module_forward_pre_hook(pre_fw)
    try:
        r71 = run_cur(ARM["CUR_B"], 1, dict(REAL, total=T_SWITCH + 2))
    finally:
        h1.remove()
        h2.remove()
    lrs = [x[1] for x in rec_opt]
    lr_ok = (len(lrs) == T_SWITCH + 2 and all(x == [LR1] for x in lrs[:T_SWITCH])
             and all(x == [LR2] for x in lrs[T_SWITCH:]))
    w_ok = (len(widths) == T_SWITCH + 2 and set(widths[:T_SWITCH]) == {TASK1.L - 1}
            and set(widths[T_SWITCH:]) == {TASK8.L - 1})
    k_ok = (keysets.get(T_SWITCH - 1) == [N_ACTIVE] and keysets.get(T_SWITCH) == [N_ACTIVE]
            and keysets.get(T_SWITCH + 1) == [TASK8.P] and keysets.get(T_SWITCH + 2) == [TASK8.P])
    same_opt = len({x[0] for x in rec_opt}) == 1
    adam_ok = rec_opt[T_SWITCH][2] == float(T_SWITCH) and rec_opt[T_SWITCH - 1][2] == float(T_SWITCH - 1)
    print(f"         lr at steps 1-{T_SWITCH}: {sorted({v for x in lrs[:T_SWITCH] for v in x})}   at steps "
          f"{T_SWITCH + 1}-{T_SWITCH + 2}: {sorted({v for x in lrs[T_SWITCH:] for v in x})}: {lr_ok}")
    print(f"         training input widths at steps 1-{T_SWITCH}: {sorted(set(widths[:T_SWITCH]))}, after: "
          f"{sorted(set(widths[T_SWITCH:]))}   distinct keys per sequence at steps {T_SWITCH - 1}-"
          f"{T_SWITCH + 2}: {[keysets.get(s_) for s_ in range(T_SWITCH - 1, T_SWITCH + 3)]}: {w_ok and k_ok}")
    print(f"         one optimizer throughout: {same_opt}   Adam's step count entering steps {T_SWITCH} and "
          f"{T_SWITCH + 1}: {rec_opt[T_SWITCH - 1][2]}, {rec_opt[T_SWITCH][2]} (carried over): {adam_ok}   "
          f"run ok {r71['ok']}")
    good = lr_ok and w_ok and k_ok and same_opt and adam_ok and r71["ok"]
    small = dict(t_switch=40, total=120, eval_every=10, t_check=20, a_check=A_CHECK)
    real_eval = tbo.evaluate
    tbo.evaluate = lambda model, task, d: (1.0, 0.0)
    try:
        r_cur = run_cur(ARM["CUR_B"], 1, small, decide=trr.make_decide(20, A_CHECK))
        ts_ = small["t_switch"]
        r_plain = run_one(ARM["CUR_B"], 1, small["total"], small["eval_every"], task=TASK8,
                          stats_fn=conv_stats, grad_fn=conv_grad_norms, lr=LR1,
                          builder=tcl.builder_for(ARM["CUR_B"]),
                          run_kw=dict(task_at=lambda step: TASK1 if step <= ts_ else TASK8,
                                      lr_at=lambda step: LR1 if step <= ts_ else LR2))
    finally:
        tbo.evaluate = real_eval
    stop_ok = (r_cur["ok"] and r_cur["stopped_at"] == ts_ + 3 * small["eval_every"]
               and r_plain["stopped_at"] == 3 * small["eval_every"] and r_cur["transition"] == ts_ + small["eval_every"])
    print(f"         early stop, every held-out accuracy stubbed to 1.0 (switch at {ts_}, evaluations every "
          f"{small['eval_every']}): the curriculum stops at step {r_cur['stopped_at']} (phase 2's third "
          f"evaluation), transition {r_cur['transition']};")
    print(f"         without early_stop_after the same run would stop at step {r_plain['stopped_at']}, in "
          f"phase 1: {stop_ok}")
    good &= stop_ok
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 72 the pairing with this machine's test_p_scaling results ({p8_path}):")
    pair = p8_pairing(pool, p8_path)
    if not pair["exists"]:
        print("         file exists: False  -> PAIRING FAILS (every claim will be UNTESTED)")
    else:
        print(f"         file exists: True   written on: {pair['cpu']} (git {pair['git']})   this machine: "
              f"{cpu_model()}   match: {pair['cpu_match']}")
        print(f"         recorded A_conv8 (160-189) and B_conv8 (160-179) runs: {pair['n_recorded']}/"
              f"{pair['n_needed']}")
        for cur, recd, s, eq in pair["params_list"]:
            print(f"         {cur} seed {s} at step 0 == recorded {recd}'s build: {eq}")
        now, old = pair["A_conv8_curve"]
        print(f"         A_conv8 seed 160 re-run, {EVAL_EVERY} steps: {now} vs recorded {old}")
        print(f"         reproduced exactly: {pair['reproduced']}  -> "
              f"{'PAIRED' if pair['ok'] else 'PAIRING FAILS (every claim will be UNTESTED)'}")
    print()

    print("CHECK 73 the restart rule:")
    dec = trr.make_decide(T_CHECK, A_CHECK)
    params = list(inspect.signature(dec).parameters)
    cv = inspect.getclosurevars(dec)
    good_a = (params == ["step", "acc"] and not cv.globals and not cv.builtins
              and cv.nonlocals == {"t_check": T_CHECK, "a_check": A_CHECK})
    print(f"     (a) decide's parameters: {params}; it reads no global name ({dict(cv.globals)}) and only "
          f"the constants {dict(cv.nonlocals)}: {good_a}")
    new_seeds = [x for s in SEEDS["CUR_A_R"] for x in attempt_seeds(s)[1:]]
    good_b = (len(set(new_seeds)) == len(new_seeds) == len(SEEDS["CUR_A_R"]) * (R_MAX - 1)
              and not set(new_seeds) & (EARLIER_SEEDS | set(SEEDS["CUR_A"]))
              and all(x == s + STRIDE * j for s in SEEDS["CUR_A_R"] for j, x in enumerate(attempt_seeds(s))))
    rng_txt = ", ".join(f"{STRIDE * j + SEEDS['CUR_A_R'][0]}-{STRIDE * j + SEEDS['CUR_A_R'][-1]}"
                        for j in range(1, R_MAX))
    print(f"     (b) attempt seeds s + 1000*j (j = 1..{R_MAX - 1}): {rng_txt}; {len(set(new_seeds))} distinct, "
          f"disjoint from every earlier seed (0-199, 1000-1149): {good_b}")
    synth = {"pass": ({2: 0.31, 4: 0.64, 6: 0.70, 8: 0.80, 10: 0.9, 12: 0.9, 14: 0.9, 16: 0.9},
                      {2: 0.05, 4: 0.10, 6: 0.20, 8: 0.30, 10: 0.5, 12: 0.5, 14: 0.5, 16: 0.5}),
             "fail": ({2: 0.35, 4: 0.52, 6: 0.55, 8: 0.58, 10: 0.6, 12: 0.6, 14: 0.6, 16: 0.6},
                      {2: 0.05, 4: 0.97, 6: 0.20, 8: 0.30, 10: 0.5, 12: 0.5, 14: 0.5, 16: 0.5})}
    tiny = dict(t_switch=8, total=16, eval_every=2, t_check=4, a_check=A_CHECK)
    cnt_o = {"opt": None, "n": 0}

    def pre(opt, args, kw):
        if cnt_o["opt"] is not opt:
            cnt_o["opt"], cnt_o["n"] = opt, 0
        cnt_o["n"] += 1
    good_c = True
    for name, (sc1, sc8) in synth.items():
        calls, rule = [], trr.make_decide(4, A_CHECK)

        def spy(step, acc, rule=rule, calls=calls):
            calls.append((step, acc))
            return rule(step, acc)
        tbo.evaluate = lambda model, task, d, sc1=sc1, sc8=sc8: ((sc1 if task is TASK1 else sc8)[cnt_o["n"]], 0.0)
        h = register_optimizer_step_pre_hook(pre)
        try:
            tr = run_trial(3, tiny, start=0, decide=spy)
        finally:
            h.remove()
            tbo.evaluate = real_eval
        info_ = tr["trial"]
        if name == "pass":
            g = (calls == [(4, 0.64)] and info_["used"] == 1 and info_["continued"] == 3
                 and [c[1] for c in tr["curve1"]] == list(sc1.values()) and tr["ok"])
        else:
            g = (calls == [(4, 0.52)] * R_MAX and info_["used"] == R_MAX
                 and info_["continued"] == 3 + STRIDE * (R_MAX - 1)
                 and [x["seed"] for x in info_["attempts"]] == attempt_seeds(3) and tr["ok"]
                 and tr["stopped_at"] == tiny["total"])
        good_c &= g
        print(f"     (c) synthetic '{name}' (phase-1 accuracy {list(sc1.values())[:4]}, P=8 accuracy "
              f"{list(sc8.values())[:4]}, check at step 4 of 16, switch at 8): decide saw {calls}; attempts "
              f"used {info_['used']}, seeds {[x['seed'] for x in info_['attempts']]}, continued {info_['continued']}"
              f"  -> {'OK' if g else 'WRONG'}")
    mid = dict(t_switch=40, total=80, eval_every=20, t_check=20, a_check=0.0)
    tr0 = run_trial(3, mid, start=0, decide=trr.make_decide(20, 0.0))
    ra = run_cur(ARM["CUR_A"], 3, mid, decide=trr.make_decide(20, 0.0))
    body0 = {k: v for k, v in tr0.items() if k != "trial"}
    good_d = strip_all(body0) == strip_all(ra) and tr0["trial"]["used"] == 1 and ra["ok"]
    print(f"     (d) attempt 0 passing (A_CHECK forced to 0, switch at 40 of 80, evaluations every 20): its "
          f"record == CUR_A's run of seed 3: {strip_all(body0) == strip_all(ra)}  -> {'OK' if good_d else 'WRONG'}")
    good = good_a and good_b and good_c and good_d
    ok &= good
    print()

    print("CHECK 74 a worker's run is bit-identical to the same run here (through the switch):")
    good = True
    for key in ("CUR_A", "CUR_B"):
        sp = spec(key, 1, dict(REAL, total=T_SWITCH + 1200))
        rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        print(f"         {key:<6} seed 1, {T_SWITCH + 1200} steps: equal {strip_all(rw) == strip_all(rh)}   "
              f"phase-1 acc {[round(c[1], 3) for c in rh['curve1']]}   P=8 acc {[round(c[1], 3) for c in rh['curve']]}"
              f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}"
          + ("" if pair["ok"] else "   (CHECK 72, the pairing, did not pass: see VALIDITY)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return pair


def tps_decode(row, task):
    S, P = task.S, task.P
    n = (len(row) - 3) // 3
    name = lambda t: f"S{t}" if t < S else f"K{t - S}" if t < S + P else f"v{t - S - P}"
    body = " | ".join(" ".join(name(int(t)) for t in row[3 * i:3 * i + 3]) for i in range(n))
    q = row[3 * n:]
    return f"{body} || {name(int(q[0]))} {name(int(q[1]))} -> {name(int(q[2]))}"


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds)
            if (r := get(store, key, s)) and r.get("ok")]


def at(curve, step):
    return next((a for st, a, _ in curve if st == step), None)


def sw_margin(r):
    return (r.get("switch") or {}).get("margin")


def routed_sw(r):
    m = sw_margin(r)
    return m is not None and m >= ROUTED_MARGIN


def plateau2(r, ts):
    tr = r["transition"]
    return sum(EVAL_EVERY for st, acc, _ in r["curve"]
               if ts < st < tr and PLATEAU[0] <= acc <= PLATEAU[1])


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def mark(store, key, s):
    if store is None:
        return "-"
    r = get(store, key, s)
    if r is None:
        return "-"
    if not r.get("ok"):
        return "?"
    return "D" if r.get("discovered") else "B" if r["transition"] is not None else "."


def distinct_runs(store):
    """C3's runs: every CUR_A run, and every CUR_A_R run that is not a reused CUR_A run."""
    out = [("CUR_A", s, r) for s, r in ok_runs(store, "CUR_A")]
    out += [("CUR_A_R", s, r) for s, r in ok_runs(store, "CUR_A_R") if not r["trial"]["reused"]]
    return out


def raw_tables(store, sched):
    ts, tc = sched["t_switch"], sched["t_check"]
    print(f"  {'arm':<11} {'seed':>4} {'run':>5} {'acc P8':>7} {'transition':>10} {'p1@' + str(tc):>8} "
          f"{'p1@' + str(ts):>8} {'P8@' + str(ts):>8} {'margin sw':>9} {'margin end':>10} {'VAL cos':>7}  "
          f"eta^2 key s/k/h/i end   conv |w| lag 0/1/2/3   flags")
    for k in KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<11} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<11} {s:>4}  FAILED — {r.get('error')}")
                continue
            e = r["end"]
            gated = k in GATED
            etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS) if gated else "--"
            if r["transition"] is not None:
                fl = ["BOUND"] + (["DISCOVERED"] if r["discovered"] else [])
                if gated:
                    fl.append("routed" if routed(r) else "NOT routed")
            else:
                fl = [fail_class(r)] if gated else []
            fl += (["collapsed"] if r["collapsed"] else [])
            if k == "CUR_A_R":
                t_ = r["trial"]
                fl.append(f"attempts {t_['used']}" + (" (CUR_A reused)" if t_["reused"] else ""))
            print(f"  {k:<11} {s:>4} {r['seed']:>5} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                  f"{fmt(at(r['curve1'], tc)):>8} {fmt(at(r['curve1'], ts)):>8} {fmt(at(r['curve'], ts)):>8} "
                  f"{fmt(sw_margin(r)):>9} {fmt(e.get('margin')):>10} {e['val_cos']:>7.3f}  {etas:<22}   "
                  f"{wstr(e):<23}  {' '.join(fl)}")
        print()
    print(f"  CUR_A_R attempts (seed: phase-1 accuracy at {tc}, pass/fail; the last attempt continues):")
    for s, r in ok_runs(store, "CUR_A_R"):
        t_ = r["trial"]
        print(f"    s{s}: " + "   ".join(f"{x['seed']}: {fmt(x['acc_check'], 4)} "
                                        f"{'pass' if x['passed'] else 'fail'}" for x in t_["attempts"])
              + f"   -> continued {t_['continued']}" + ("  (CUR_A's record reused)" if t_["reused"] else ""))
    print()


def report_stop(store, pair, sched, wall, path):
    """The ceiling failed: validity failure, every claim UNTESTED, nothing else run."""
    print("#" * 100)
    print("PER-SEED RAW RESULTS — CUR_ceiling only (the rest was not run)")
    print("#" * 100)
    k = "CUR_ceiling"
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
    print(f"  CUR_ceiling binds {c}/{len(SEEDS[k])} (needs >= {need}): the validity condition FAILED; the "
          f"other arms were not run")
    print("     *** UNTESTED ***")
    print()
    print("PRE-REGISTERED CLAIMS: C1 UNTESTED, C2 UNTESTED, C3 UNTESTED, C4 UNTESTED; no reading.")
    store["verdict"] = dict(valid=False, stopped="ceiling", ceiling=c, pairing=pair["ok"], C1="UNTESTED",
                            C2="UNTESTED", C3="UNTESTED", C4="UNTESTED", readings=[])
    save_results(path, store)
    print(f"\n  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}\n")


def report(store, p8, pair, sched, wall, path, also):
    ts, tc = sched["t_switch"], sched["t_check"]
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    raw_tables(store, sched)

    print("=" * 100)
    print(f"COUNTS (BOUND = transition on the P=8 curve after the switch; DISCOVERED = bound AND final "
          f"VAL cos < 0.5, gated arms)")
    print("=" * 100)
    c, d = {}, {}
    for a in ARMS:
        k, ns = a["key"], len(SEEDS[a["key"]])
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        d[k] = sum(disc(store, k, s) for s in SEEDS[k])
        rs = [get(store, k, s) for s in SEEDS[k]]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        extra = ""
        if p8 is not None:
            have = [s for s in SEEDS[k] if get(p8, RECORDED[k], s) is not None]
            extra = (f"   (recorded {RECORDED[k]}, same seeds: bound "
                     f"{sum(bound(p8, RECORDED[k], s) for s in have)}/{len(have)})")
        print(f"  {a['label']:<46} bound {c[k]:>2}/{ns}"
              + (f"   discovered {d[k]:>2}/{ns}" if k in GATED else " " * 18)
              + f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{ns}"
              + (f"   [{failed} FAILED]" if failed else "") + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["CUR_ceiling"])
    need = min(CEIL_MIN, nc)
    valid = c["CUR_ceiling"] >= need and pair["ok"]
    print(f"  CUR_ceiling binds {c['CUR_ceiling']}/{nc} (needs >= {need})   the pairing CHECK (72) passed: "
          f"{pair['ok']} (file {pair['path']} exists {pair['exists']}, CPU match {pair.get('cpu_match')}, "
          f"complete {pair.get('complete')}, step-0 parameters {pair.get('params')}, reproduced "
          f"{pair.get('reproduced')})")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND on the P=8 set)")
    print("#" * 100)
    v = dict(bound=c, discovered=d, valid=valid, pairing=pair["ok"])
    verdict = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    s1 = SEEDS["CUR_A"]
    if p8 is None:
        b1 = c1 = 0
        p1 = float("nan")
        v1 = "UNTESTED"
    else:
        b1 = sum(bound(store, "CUR_A", s) and not bound(p8, "A_conv8", s) for s in s1)
        c1 = sum(bound(p8, "A_conv8", s) and not bound(store, "CUR_A", s) for s in s1)
        p1 = mcnemar_greater(b1, c1)
        v1 = verdict(p1)
    n8 = sum(bound(p8, "A_conv8", s) for s in s1) if p8 is not None else "?"
    print(f"  C1 THE CURRICULUM HELPS THE GATE (exact McNemar, one-sided): CUR_A {c['CUR_A']}/{len(s1)} vs "
          f"recorded A_conv8 {n8}/{len(s1)}; CUR_A only {b1}, A_conv8 only {c1}; p = {p1:.3g}")
    print(f"     *** C1: {v1} ***")
    s2 = SEEDS["CUR_B"]
    ca2 = sum(bound(store, "CUR_A", s) for s in s2)
    cb2 = sum(bound(store, "CUR_B", s) for s in s2)
    p2 = fisher_greater(ca2, len(s2), cb2, len(s2))
    b2 = sum(bound(store, "CUR_A", s) and not bound(store, "CUR_B", s) for s in s2)
    c2_ = sum(bound(store, "CUR_B", s) and not bound(store, "CUR_A", s) for s in s2)
    print(f"  C2 THE GAIN NEEDS CHANNELS (Fisher one-sided): CUR_A {ca2}/{len(s2)} (seeds {s2[0]}-{s2[-1]}) vs "
          f"CUR_B {cb2}/{len(s2)}; p = {p2:.3g}")
    print(f"     exact McNemar two-sided on the pairs: CUR_A only {b2}, CUR_B only {c2_}; p = "
          f"{mcnemar_exact(b2, c2_):.3g}")
    print(f"     *** C2: {verdict(p2)} ***")
    runs = [(k, s, r) for k, s, r in distinct_runs(store) if sw_margin(r) is not None]
    ro = [r for _, _, r in runs if routed_sw(r)]
    nro = [r for _, _, r in runs if not routed_sw(r)]
    bro = sum(1 for r in ro if r["transition"] is not None)
    bnro = sum(1 for r in nro if r["transition"] is not None)
    if len(ro) < C3_MIN or len(nro) < C3_MIN:
        p3, v3 = float("nan"), ("UNTESTABLE" if valid else "UNTESTED")
    else:
        p3 = fisher_greater(bro, len(ro), bnro, len(nro))
        v3 = verdict(p3)
    print(f"  C3 ROUTING TRANSFERS (Fisher one-sided; {len(runs)} distinct runs of CUR_A and CUR_A_R): routed at "
          f"the switch (margin >= {ROUTED_MARGIN}) bound {bro}/{len(ro)} vs not routed {bnro}/{len(nro)}; "
          f"p = {p3:.3g}")
    print(f"     *** C3: {v3} ***")
    cr = c["CUR_A_R"]
    v4 = band(cr) if valid else "UNTESTED"
    print(f"  C4 A RELIABLE RECIPE: CUR_A_R bound {cr}/{len(SEEDS['CUR_A_R'])} (RELIABLE >= {RELIABLE_R}, "
          f"MAJORITY {MAJORITY_R}-{RELIABLE_R - 1}, MINORITY 1-{MAJORITY_R - 1}, NEVER 0)")
    print(f"     *** C4: {v4} ***")
    v.update(C1=v1, C1_p=p1, C2=verdict(p2), C2_p=p2, C3=v3, C3_p=p3, C4=v4,
             C3_groups=dict(routed=[bro, len(ro)], not_routed=[bnro, len(nro)]))
    print()
    print("  READINGS:")
    rd = []
    if not valid:
        print("     (UNTESTED: no reading)")
    else:
        if v["C1"] == "SHOWN" and v["C3"] == "SHOWN":
            rd.append(("C1 SHOWN and C3 SHOWN", READINGS["C1+ C3+"]))
        if v["C1"] == "SHOWN" and v["C2"] == "NOT SHOWN":
            rd.append(("C1 SHOWN and C2 NOT SHOWN", READINGS["C1+ C2-"]))
        if v["C1"] == "NOT SHOWN":
            rd.append(("C1 NOT SHOWN", READINGS["C1-"]))
        if v["C4"] == "RELIABLE":
            rd.append(("C4 RELIABLE", READINGS["C4 RELIABLE"]))
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
    print("  per seed (B = bound, D = bound and discovered, . = neither, - = not run); margins of CUR_A at the "
          "switch -> end; CUR_A_R attempts used:")
    print(f"    seed  {'A_conv8':>8} {'CUR_A':>6} {'CUR_A_R':>8} {'used':>4} {'CUR_A_lo':>8}   {'B_conv8':>8} "
          f"{'CUR_B':>6}   CUR_A margin switch -> end")
    for s in SEEDS["CUR_A"]:
        rA, rR = get(store, "CUR_A", s), get(store, "CUR_A_R", s)
        used = rR["trial"]["used"] if rR and rR.get("ok") else "-"
        mg = (f"{fmt(sw_margin(rA))} -> {fmt(rA['end'].get('margin'))}" if rA and rA.get("ok") else "--")
        in_b = s in SEEDS["CUR_B"]
        print(f"    {s:>4}  {mark(p8, 'A_conv8', s):>8} {mark(store, 'CUR_A', s):>6} {mark(store, 'CUR_A_R', s):>8} "
              f"{used!s:>4} {(mark(store, 'CUR_A_lo', s) if s in SEEDS['CUR_A_lo'] else '-'):>8}   "
              f"{(mark(p8, 'B_conv8', s) if in_b else '-'):>8} {(mark(store, 'CUR_B', s) if in_b else '-'):>6}   {mg}")
    print()
    print(f"  routing persistence (distinct CUR_A / CUR_A_R runs, and CUR_A_lo): routed at the switch -> still "
          f"routed at the end (margin >= {ROUTED_MARGIN}):")
    groups = [("CUR_A + CUR_A_R", [r for _, _, r in runs])]
    groups.append(("CUR_A_lo", [r for _, r in ok_runs(store, "CUR_A_lo") if sw_margin(r) is not None]))
    groups.append(("CUR_ceiling", [r for _, r in ok_runs(store, "CUR_ceiling") if sw_margin(r) is not None]))
    for name, rs in groups:
        rsw = [r for r in rs if routed_sw(r)]
        for lab, sub in (("bound", [r for r in rsw if r["transition"] is not None]),
                         ("not bound", [r for r in rsw if r["transition"] is None])):
            print(f"    {name:<16} routed at the switch {len(rsw):>2}/{len(rs):<2}  {lab:<9} {len(sub):>2}: still routed "
                  f"at the end {sum(1 for r in sub if routed(r))}")
        nsw = [r for r in rs if not routed_sw(r)]
        print(f"    {name:<16} NOT routed at the switch {len(nsw)}: bound {sum(1 for r in nsw if r['transition'] is not None)}, "
              f"routed at the end {sum(1 for r in nsw if routed(r))}")
    print()
    print(f"  phase-1 accuracy at {tc} and {ts}, P=8 accuracy at the switch, and the switch margin, median "
          f"[min, max]:")
    for k in KEYS:
        rs = [r for _, r in ok_runs(store, k)]
        print(f"    {k:<11} p1@{tc} {med([x for r in rs if (x := at(r['curve1'], tc)) is not None]):<22} "
              f"p1@{ts} {med([x for r in rs if (x := at(r['curve1'], ts)) is not None]):<22} "
              f"P8@{ts} {med([x for r in rs if (x := at(r['curve'], ts)) is not None]):<22}"
              + (f" margin {med([sw_margin(r) for r in rs if sw_margin(r) is not None])}" if k in GATED else ""))
    print()
    print(f"  transitions after the switch and phase-2 plateau length (steps at P=8 accuracy "
          f"{PLATEAU[0]}-{PLATEAU[1]} before the transition):")
    for k in KEYS:
        rs = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None]
        if not rs:
            print(f"    {k:<11} none bound")
            continue
        trs = [r["transition"] for _, r in rs]
        pls = [plateau2(r, ts) for _, r in rs]
        print(f"    {k:<11} transition median {statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   plateau "
              f"median {statistics.median(pls):.0f} [{min(pls)}, {max(pls)}]")
        print(f"    {'':<11} " + "  ".join(f"s{s}:{r['transition']}/{plateau2(r, ts)}" for s, r in rs))
    print()
    print("  gated runs: bound WITH routing (end margin >= 0.9) or WITHOUT; test_router_layout's classes at the "
          "switch and, for failures, at the end:")
    for k in GATED:
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
        print(f"    {k:<11} bound with routing {len(w_)}, WITHOUT routing {len(wo)}"
              + (f" ({' '.join(f's{s}' for s in wo)})" if wo else ""))
        print(f"    {'':<11} at the switch: " + "   ".join(f"{m_} {len(x)}" for m_, x in cls_sw.items()))
        print(f"    {'':<11} failures at the end: " + "   ".join(f"{m_} {len(x)}" for m_, x in cls_end.items()))
        for m_, xs in cls_end.items():
            if xs:
                print(f"    {'':<11}   {m_:<14} " + " ".join(f"s{s}" for s in xs))
    print()
    print("  the convolution's mean |w| per lag at the end, median [min, max] (lag 2 = where a value position "
          "sees its context token):")
    for k in KEYS:
        rs = ok_runs(store, k)
        b_all = [r for _, r in rs if r["transition"] is not None]
        grp = (("bound", b_all), ("not bound", [r for _, r in rs if r["transition"] is None]))
        if k in GATED:
            grp = (("bound, routed", [r for r in b_all if routed(r)]),
                   ("bound, NOT routed", [r for r in b_all if not routed(r)]), grp[1])
        for lab, sub in grp:
            if sub:
                print(f"    {k:<11} {lab:<17} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median [min, max]:")
    for step in grad_steps_of(sched):
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>5} {k:<11} " + "   ".join(parts))
    print(f"  collapsed runs (final P=8 accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
    print()
    print("  exact McNemar two-sided (bound), per seed:")
    pairs = [("CUR_A_lo", "CUR_A", store, store, SEEDS["CUR_A_lo"]), ("CUR_B", "B_conv8", store, p8, SEEDS["CUR_B"]),
             ("CUR_A_R", "CUR_A", store, store, SEEDS["CUR_A_R"]), ("CUR_A", "CUR_B", store, store, SEEDS["CUR_B"])]
    for k1, k2, st1, st2, sh in pairs:
        if st2 is None:
            continue
        b_ = sum(bound(st1, k1, s) and not bound(st2, k2, s) for s in sh)
        c_ = sum(bound(st2, k2, s) and not bound(st1, k1, s) for s in sh)
        print(f"    {k1:<9} vs {k2 + (' (recorded)' if st2 is p8 else ''):<19} ({len(sh):>2} seeds): {k1} only {b_}, "
              f"{k2} only {c_}; p = {mcnemar_exact(b_, c_):.3g}")
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
                ns = len(SEEDS[k])
                print(f"    {k:<11} bound here {c[k]}/{ns}   other {b2_}/{len(rs2)}   pooled {c[k] + b2_}/{ns + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — per evaluation (every {EVAL_EVERY} steps): phase-1 set accuracy x100, P=8 set accuracy x100, "
          f"VAL cos x100 on the P=8 probe; | = the switch after step {ts}")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            curve_lines2(f"{k:<11} s{s}", r, ts)
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


def curve_lines2(tag, r, ts):
    ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
    sep = lambda st: " |" if st == ts else "  "
    p1 = "".join(f"{round(a * 100):>3}{sep(st)}" for st, a, _ in r["curve1"])
    p8 = "".join(f"{round(a * 100):>3}{sep(st)}" for st, a, _ in r["curve"])
    cs = "".join((f"{round(ev[st]['val_cos'] * 100):>3}" if st in ev else " --") + sep(st) for st, _, _ in r["curve"])
    end = f"bound @{r['transition']}" if r["transition"] else "not bound"
    print(f"  {tag} p1  {p1}")
    print(f"  {'':<{len(tag)}} P8  {p8}")
    print(f"  {'':<{len(tag)}} cos {cs}  -> {end}{'  DISCOVERED' if r.get('discovered') else ''}")


# ── Main ─────────────────────────────────────────────────────────────────────
def describe(rec):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr = rec["transition"]
    e = rec["end"]
    m = (f" margin {fmt(sw_margin(rec))}->{fmt(e.get('margin'))}" if "margin" in e else "")
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} "
            f"p1@check={fmt(rec.get('acc_check'))}{m} VALcos={e['val_cos']:.3f} conv|w|={wstr(e)}"
            f"{'  DISCOVERED' if rec['discovered'] else ''}")


def main():
    ap = argparse.ArgumentParser(description="Does routing found at 4 active keys carry to 8?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's load_curriculum_results.json, for pooled counts")
    ap.add_argument("--p8", default=P8_FILE,
                    help="this machine's p_scaling_results.json (the recorded P=8 runs to pair with)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    sched = dict(SCHED)
    ts = sched["t_switch"]

    print("=" * 100)
    print("Load curriculum: does routing found at 4 active keys survive the switch to 8?")
    print(f"  phase 1: n_active={N_ACTIVE} of P=8 (length {TASK1.L}), lr {LR1}, steps 1-{ts};  phase 2: "
          f"task_for(8, 2) (length {TASK8.L}), lr {LR2} (CUR_A_lo {LR1}), up to {sched['total'] - ts} more steps")
    print(f"  n_layer=3 decay N=256, conv 'layer' width {CONV_K}; evaluation every {sched['eval_every']} on both "
          f"held-out sets; early stop only in phase 2")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<46} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  restart rule (CUR_A_R): phase-1 accuracy >= {sched['a_check']} at step {sched['t_check']}, attempts "
          f"s + {STRIDE}*j, j < {R_MAX}")
    print(f"  paired with {args.p8}: same seeds, same initial parameters as the recorded P=8 runs")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        pair = verify(pool, args.p8)
        p8 = load_store(args.p8) if pair["exists"] else None
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head,
                             sched=sched, lr1=LR1, lr2=LR2, n_active=N_ACTIVE, p8=args.p8,
                             pairing={k: v for k, v in pair.items() if not k.endswith("_curve")},
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool on each phase's task, both "
              f"evaluations charged once per {EVAL_EVERY}; worst case = every run to {sched['total']}, every "
              f"CUR_A_R trial needing all {R_MAX} attempts")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_job, KEYS)))
        full = {k: ts * c1 + (sched["total"] - ts) * c8 for k, (c1, c8) in costs.items()}
        for k in KEYS:
            print(f"  {k:<11} phase 1 {costs[k][0] * 1000:6.1f} ms/step, phase 2 {costs[k][1] * 1000:6.1f} ms/step   "
                  f"x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        trial_extra = (R_MAX - 2) * sched["t_check"] * costs["CUR_A_R"][0]
        d_ceil = [full["CUR_ceiling"]] * len(SEEDS["CUR_ceiling"])
        d_rest = ([full[k] for k in ("CUR_A", "CUR_B", "CUR_A_lo") for _ in SEEDS[k]]
                  + [full["CUR_A_R"] + trial_extra] * len(SEEDS["CUR_A_R"]))
        mk1, mk2 = makespan(d_ceil, args.workers), makespan(d_rest, args.workers)
        print(f"  CUR_ceiling first: {len(d_ceil)} runs, {mk1 / 3600:.2f} h;  then {len(d_rest)} runs/trials: serial "
              f"{sum(d_rest) / 3600:.2f} h, {mk2 / 3600:.2f} h on {args.workers} workers;  total {(mk1 + mk2) / 3600:.2f} h "
              f"(no drop rule)")
        store["meta"]["projected_wall_h"] = (mk1 + mk2) / 3600
        print()

        def record(sp, rec):
            key = f"{sp['arm']}|{sp['seed']}"
            store["runs"][key] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            extra = ""
            if sp["arm"] == "CUR_A_R" and rec.get("ok"):
                t_ = rec["trial"]
                extra = f"  attempts {t_['used']} (continued {t_['continued']})" + ("  CUR_A reused" if t_["reused"] else "")
            secs = rec.get("trial", {}).get("secs", rec.get("secs", 0.0)) if sp["arm"] == "CUR_A_R" else rec.get("secs", 0.0)
            print(f"   {sp['arm']:<11} seed {sp['seed']}  {describe(rec)}{extra}  {secs:.0f}s  (elapsed {el:.1f}m)")

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
                        nxt = on_done(sp)
                        if nxt is not None:
                            pending.insert(0, nxt)

        def todo(k):
            out = []
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<11} seed {s}  cached")
                else:
                    out.append(spec(k, s, sched))
            return out

        print("=" * 100)
        print(f"RUNS — CUR_ceiling first ({len(SEEDS['CUR_ceiling'])} runs)")
        print("=" * 100)
        run_specs(todo("CUR_ceiling"))
        n_ceil = sum(bound(store, "CUR_ceiling", s) for s in SEEDS["CUR_ceiling"])
        need = min(CEIL_MIN, len(SEEDS["CUR_ceiling"]))
        print(f"   CUR_ceiling bound {n_ceil}/{len(SEEDS['CUR_ceiling'])} (needs >= {need})")
        print()
        if n_ceil < need:
            print("   *** the ceiling failed: the rest is NOT run ***\n")
            report_stop(store, pair, sched, time.time() - t0, args.results)
            return

        def after(sp):
            if sp["arm"] != "CUR_A":
                return None
            return r_step(store, sp["seed"], sched, args.force, record)

        n_rest = sum(len(SEEDS[k]) for k in ("CUR_A", "CUR_A_R", "CUR_B", "CUR_A_lo"))
        print("=" * 100)
        print(f"RUNS — the rest ({n_rest} runs/trials; each CUR_A_R trial is queued, first, when its CUR_A run "
              f"is known)")
        print("=" * 100)
        first = todo("CUR_A")
        for s in SEEDS["CUR_A"]:
            if get(store, "CUR_A", s) is not None and not args.force:
                nxt = r_step(store, s, sched, args.force, record)
                if nxt is not None:
                    first.insert(0, nxt)
        order = {"CUR_A": 0, "CUR_B": 1, "CUR_A_lo": 2}
        rest = sorted(todo("CUR_B") + todo("CUR_A_lo"), key=lambda sp: order[sp["arm"]])
        run_specs(first + rest, on_done=after)
        print()

    report(store, p8, pair, sched, time.time() - t0, args.results, args.also)


def r_step(store, s, sched, force, record):
    """CUR_A_R seed s once CUR_A seed s is known: reuse CUR_A's record if its attempt passed,
    else a trial from attempt 1 (from attempt 0 if CUR_A's run failed)."""
    if get(store, "CUR_A_R", s) is not None and not force:
        print(f"   CUR_A_R     seed {s}  cached")
        return None
    rA = get(store, "CUR_A", s)
    if rA and rA.get("ok") and rA.get("passed"):
        rec = dict(rA, trial=dict(start=0, attempts=[dict(j=0, seed=s, acc_check=rA["acc_check"], passed=True)],
                                  used=1, continued=s, reused=True, secs=0.0))
        record(dict(arm="CUR_A_R", seed=s), rec)
        return None
    return spec("CUR_A_R", s, sched, trial=True, start=1 if (rA and rA.get("ok")) else 0)


def finish_trial(store, sp, rec):
    """Prepend attempt 0 (CUR_A's run) to a trial that started at attempt 1."""
    if rec.get("ok") and sp["start"] == 1:
        rA = get(store, "CUR_A", sp["seed"])
        a0 = dict(j=0, seed=sp["seed"], acc_check=rA.get("acc_check"), passed=rA.get("passed"))
        rec["trial"]["attempts"] = [a0] + rec["trial"]["attempts"]
    elif rec.get("ok") is False and "trial" not in rec:
        rec["trial"] = dict(start=sp["start"], attempts=[], used=None, continued=None, reused=False, secs=0.0)
    return rec


if __name__ == "__main__":
    main()
