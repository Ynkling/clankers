#!/usr/bin/env python
"""
explore_g_layers.py — EXPLORATORY, not a result. Session G follow-up, screen S71: which layers must be routed for the
memory to bind RandHeaderTask-lite? Everything below was fixed and committed before any S71 run.

BACKGROUND. S56 (header P = 8): REG_NUDGE on Hebbian memory had a clean latch and perfect routing at layers 2-3 and sat at
0.32; the post-hoc ORACLE23 (LOCAL3 at layer 1, the perfect gate at layers 2-3) bound 0/3 while ORACLE_H (all layers
perfect) bound seed 371 (one paired seed). On delta memory REG_NUDGE_D bound 8/10. S71 measures this on RandHeaderTask-lite,
where both perfect gates bind by 1200 (F's S70).

TASK, MODELS, STATISTICS, OUTCOMES: explore_g_rhl_child and explore_g_rhl_parent (RandHeaderTask-lite exactly as F's S70;
24000 updates; k = 2; no conv). Recipe: Adam 1e-3 on every trainable parameter throughout, for every S71 arm (the LOCAL3
gate at the non-perfect layers trains at 1e-3 too).
ARMS (seeds 700-709, paired: the same batches and the same stack parameters)
  Hebbian: ORACLE_ALL (perfect at layers 1-3; S56's ORACLE_H code path), ORACLE_23 (perfect at 2-3, LOCAL3 at 1),
           ORACLE_1 (perfect at 1, LOCAL3 at 2-3)
  delta (beta 1, L2 keys, decay 0.95): ORACLE_ALL_D, ORACLE_23_D
  50 runs.
VALIDITY: the Hebbian family is VALID if ORACLE_ALL binds >= 8/10; the delta family if ORACLE_ALL_D binds >= 8/10. Otherwise
that family's readings are UNTESTED. ORACLE_ALL and ORACLE_ALL_D are also the validity oracles of S72 and S73.
READINGS (BOUND, paired on 700-709; "beats" = one-sided exact McNemar p < 0.05; "A >= B - 1 pair" = (B only) - (A only) <= 1)
  "layer 1 is the block"                    if ORACLE_ALL beats ORACLE_23
  "layer 1 is not needed"                   if ORACLE_23 >= ORACLE_ALL - 1 pair
  "layer 1 alone suffices"                  if ORACLE_1 >= ORACLE_ALL - 1 pair
  "delta tolerates an unrouted layer 1"     if ORACLE_23_D >= ORACLE_ALL_D - 1 pair
  (each printed with b, c, p; none of them is printed as applying if its family is UNTESTED)
ALSO REPORTED: transitions; final accuracy; the in-run probe at L2 / L3 / final; per-layer margins and contingency tables;
held-out accuracy by (query stream, block, pair index) for unbound runs; write-order accuracy.
CHECKS (explore_g_rhl_child.checks_s71, asserted before any run, and X's arm A bit for bit): F's check_rand_header and the
task's parameters; the in-run probe's hooked features equal S55's extraction on LOCAL3 and respect per-layer gates; the
per-cell accuracy covers every query; each oracle is one-hot on the stream at its perfect layers and LOCAL3's gate at the
others; ORACLE_ALL (and ORACLE_ALL_D) equal S56's ORACLE_H (ORACLE_D) code path; the arms share the stack parameters; the
all-layer oracles take nothing from the other stream.

Run:  python explore_g_layers.py [--report-only]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_g_common as gc
import explore_g_rhl_parent as hp

NAME = "s71_layers"
CHILD = hp.CHILD
SEEDS = tuple(range(700, 710))
ARMS = {a: dict(iters=24000, seeds=SEEDS, prio=1) for a in
        ("ORACLE_ALL", "ORACLE_23", "ORACLE_1", "ORACLE_ALL_D", "ORACLE_23_D")}
line = hp.line


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    print(hp.count_table(R, list(ARMS)))
    b = hp.bound
    vh = sum(b(r) for r in R["ORACLE_ALL"].values()) >= 8
    vd = sum(b(r) for r in R["ORACLE_ALL_D"].values()) >= 8
    rd = {"validity": f"Hebbian {'VALID' if vh else 'UNTESTED'} (ORACLE_ALL {sum(b(r) for r in R['ORACLE_ALL'].values())}/10); "
                      f"delta {'VALID' if vd else 'UNTESTED'} (ORACLE_ALL_D {sum(b(r) for r in R['ORACLE_ALL_D'].values())}/10)"}
    fmt = lambda t: f"b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]}"
    x, t = hp.beats(R["ORACLE_ALL"], R["ORACLE_23"], b)
    rd["layer 1 is the block (ALL beats 23)"] = f"{'APPLIES' if x and vh else 'does not apply'} ({fmt(t)})"
    x, t = hp.not_worse_1(R["ORACLE_23"], R["ORACLE_ALL"], b)
    rd["layer 1 is not needed (23 >= ALL - 1)"] = f"{'APPLIES' if x and vh else 'does not apply'} ({fmt(t)})"
    x, t = hp.not_worse_1(R["ORACLE_1"], R["ORACLE_ALL"], b)
    rd["layer 1 alone suffices (1 >= ALL - 1)"] = f"{'APPLIES' if x and vh else 'does not apply'} ({fmt(t)})"
    x, t = hp.not_worse_1(R["ORACLE_23_D"], R["ORACLE_ALL_D"], b)
    rd["delta tolerates an unrouted layer 1 (23_D >= ALL_D - 1)"] = f"{'APPLIES' if x and vd else 'does not apply'} ({fmt(t)})"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, list(ARMS)))
    print("\nper run:\n" + hp.per_run(R, list(ARMS)))
    with open(os.path.join(gc.G_DIR, "s71_summary.json"), "w") as f:
        json.dump(dict(readings=rd, valid_hebb=vh, valid_delta=vd), f, indent=1)


if __name__ == "__main__":
    hp.drive(sys.modules[__name__], "S71")
