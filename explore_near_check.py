#!/usr/bin/env python
"""
explore_near_check.py — EXPLORATORY, not a result. Screen S9 (batch 3), descriptive: do S8's two
new gates keep arm A's behaviour on the standard (grouped) task, where the stream token is next
to every key and value?

ARMS (explore_common's TASK: BindTask P=4, S=2, n_vals=16, n_q=1, grouped; seeds 160-169, paired
with X's recorded arm A, which DISCOVERED 4/10 on these seeds)
  EMA_near   explore_gates' multi-scale EMA gate (as far_EMA)
  SEL_near   explore_gates' selective, minGRU-style gate (as far_SEL)
Recipe explore_common's: lr test_channel_binding.SUB_LR = 1e-3 (passed explicitly), MAX_ITERS
24000; statistics explore_common2.stats_b2 (ROUTED* at 1200, 2400, 3600 and the end).
X's arm A ROUTED* at 2400 comes from batch 2's explore_ref_a2400 re-run (seeds 160-179).

READING (descriptive, per arm, fixed before any run): "keeps the near case" if DISCOVERED >= X's
arm A's DISCOVERED on these seeds - 1 (i.e. >= 3/10); otherwise "does not keep the near case".

CHECKS: explore_gates.check_gate on the grouped task for both gates.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_gates as eg
import test_short_conv as tsc

NAME = "near_check"
IDEA = "S8's EMA and selective gates on the standard grouped task (the near case)"
SOURCE = "explore_gates (multi-timescale EMA; minGRU, Feng et al. 2024)"
CHANGE = "the gate's recurrence only (explore_gates); grouped layout, arm A otherwise"
PAIRING = ("seeds 160-169, paired with X's recorded arm A (test_short_conv, lr 1e-3; same shared "
           "initial parameters and batches); X's ROUTED* at 2400 from batch 2's ref_a2400")
SEEDS = tuple(range(160, 170))
REF_STORE = "ref_a2400"

_A = dict(ec.ARM_A, lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1)
ARMS = {
    "EMA_near": dict(_A, key="EMA_near", label="multi-scale EMA gate, grouped layout"),
    "SEL_near": dict(_A, key="SEL_near", label="selective (minGRU-style) gate, grouped layout"),
}
KIND = {"EMA_near": "ema", "SEL_near": "sel"}


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_one(a, seed, a["iters"], task=ec.TASK, stats_fn=c2.stats_b2,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=eg.builder(ec.TASK, KIND[arm]))


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, eg.builder(ec.TASK, KIND[arm])(a, 0), a["lr"], c2.stats_b2)]


def check():
    ok = eg.check_gate(ec.TASK, "ema", f"{NAME}/EMA_near")
    ok &= eg.check_gate(ec.TASK, "sel", f"{NAME}/SEL_near")
    return ok


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    refs = c2.ref_runs(ec.load_store(REF_STORE))
    c2.per_seed_table2(me, store, refs=refs)
    xa = sum(ec.discovered(ec.recorded("A")[s]) for s in SEEDS)
    out = {}
    for k in ARMS:
        print(f"  {k}, paired with X's arm A:")
        d = ec.paired_vs_recorded(store, k, SEEDS, ec.discovered, label="DISCOVERED")
        ec.paired_vs_recorded(store, k, SEEDS, ec.bound, label="BOUND")
        for step in (1200, 2400, "end"):
            c2.paired_routed(store, k, SEEDS, step, refs)
        print(f"    failures {k} {ec.fail_counts(store, k, SEEDS)}   "
              f"X arm A {ec.recorded_fail_counts('A', SEEDS)}")
        reading = ("keeps the near case" if d["new"] >= xa - 1 else "does not keep the near case")
        print(f"  READING {k}: DISCOVERED {d['new']}/{d['n']} vs X's arm A {xa}/10 (threshold "
              f">= {xa - 1}): {reading}")
        out[k] = dict(d, reading=reading)
    return out
