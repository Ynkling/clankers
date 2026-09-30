#!/usr/bin/env python
"""
explore_ref_a2400.py — EXPLORATORY, not a result. Batch 2's reference: X's arm A re-run here
through step 2400, to read its ROUTED* at step 2400.

WHY. Batch 2 prints, for S4 and S5, ROUTED* at 1200, 2400 and the end, paired with X's arm A. X's
record (results/X/short_conv_results.json) has routing statistics (margin, eta^2) only at step
1200 and at the end. Arm A is therefore re-run here on seeds 160-179 through step 2400 with
explore_common2.stats_b2 (routing statistics at 1200 and 2400). This container reproduces X's
arm A bit for bit at 1 thread (explore_common.repro_check). Each re-run is compared with X's
record (curve points at 1200 and 2400, and every routing statistic at 1200); a seed whose re-run
differs is left out of the ROUTED*@2400 rows, and the report names it.

THE RUN. test_short_conv's arm A, explore_common's recipe (lr test_channel_binding.SUB_LR = 1e-3,
passed explicitly), 2400 steps, evaluation every 1200. No change to the model or the training.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH

NAME = "ref_a2400"
IDEA = "reference: X's arm A through step 2400, for ROUTED* at 2400"
SOURCE = "test_short_conv's arm A (X's record)"
CHANGE = "none (a re-run with routing statistics at 1200 and 2400)"
PAIRING = "each seed compared with X's record of the same run"
ITERS = 2400

ARMS = {
    "A_2400": dict(ec.ARM_A, key="A_2400", lr=ec.SUB_LR, iters=ITERS, seeds=ec.SEEDS,
                   label="X's arm A, re-run through step 2400", prio=0),
}


def builder(a, seed):
    return lambda: build(ec.TASK, a, ARCH, seed)


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_one(a, seed, a["iters"], task=ec.TASK, stats_fn=c2.stats_b2,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder)


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, builder(a, 0), a["lr"], c2.stats_b2)]


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    refs = c2.ref_runs(store)
    ok = [s for s in sorted(refs) if refs[s]["reproduces"]]
    bad = [s for s in sorted(refs) if not refs[s]["reproduces"]]
    print(f"  re-runs that reproduce X's record exactly (curve at 1200 and 2400, routing statistics "
          f"at 1200): {len(ok)}/{len(refs)}" + (f"; DIFFER: {bad}" if bad else ""))
    print("  X's arm A ROUTED* at 1200 / 2400 (re-run) / end (record), per seed: " +
          " ".join(f"{s}:{'Y' if c2.x_routed(s, 1200, refs) else '-'}"
                   f"{'?' if c2.x_routed(s, 2400, refs) is None else ('Y' if c2.x_routed(s, 2400, refs) else '-')}"
                   f"{'Y' if c2.x_routed(s, 'end', refs) else '-'}" for s in sorted(refs)))
    return refs
