#!/usr/bin/env python
"""
explore_g_register3.py — EXPLORATORY, not a result. Session G follow-up, screen S72: the register gate at all three layers
on the main line's (Hebbian) memory, on RandHeaderTask-lite. Everything below was fixed and committed before any S72 run.

BACKGROUND. S56 (header P = 8, Hebbian): REG_NUDGE with the gate at layers 2-3 only had a clean latch (beta^R 0.96 at CTX)
and perfect routing at layers 2-3, and sat at 0.32 on both streams in every run (R1 "architecture insufficient"); the spec
kept layer 1 on LOCAL3. S71 measures whether layer 1 must be routed. F's S70_HEBB (LOCAL3+SLOW, 330-349): DISCOVERED
0/20, BOUND 1/20, ROUTED*@end 0/20, KEY 19.

TASK, MODELS, RECIPES, STATISTICS, OUTCOMES: explore_g_rhl_child and explore_g_rhl_parent (Hebbian memory; SLOW; no hinge).
ARMS (seeds 700-719, paired: the same batches and stack / LOCAL3 parameters)
  BASELINE     LOCAL3 + SLOW (F's S70_HEBB recipe; F's counts are the reference, other seeds: counts only)
  REG3         Reg3BDH: one gate shared by layers 1-3 over [LN(r^(l)); window_t; R_{t-1}], S56's register, SG, no nudge
  REG3_NUDGE   REG3 + the 5% labelled nudge at all three layers (validity)
  60 runs.
VALIDITY: S71's ORACLE_ALL (the Hebbian perfect gate, same seeds) binds >= 8/10 on 700-709; otherwise UNTESTED.
READINGS (ROUTED-BOUND = BOUND and F's ROUTED*@end; paired on 700-719; "beats" = one-sided exact McNemar p < 0.05)
  "the register carries the cue at all three layers"  REG3_NUDGE ROUTED-BOUND >= 17/20
  "architecture insufficient"                         REG3_NUDGE ROUTED-BOUND < 12/20 (then: does the accuracy by (query
                                                      stream, block, pair index) of its unbound runs match S56's flat 0.32
                                                      plateau — every cell within 0.1 of the run's mean — or depend on the
                                                      block / pair?)
  otherwise                                           "neither"
  "discovered"                                        REG3 beats BASELINE;  "not discovered" otherwise
  Printed beside every count: BOUND, ROUTED*@end, DISCOVERED, my BOUND ROUTED, failure classes, transitions.
ALSO REPORTED: beta^R at CTX / KEY / VAL by step; the routing margin at VAL per layer; the CTX / KEY / VAL x stream
contingency of g at every layer; the in-run probe; held-out accuracy by (query stream, block, pair index) for unbound runs.
CHECKS (explore_g_rhl_child.checks_reg3(hebb), asserted before any run, and X's arm A bit for bit): the task; the in-run
probe; S56's 12 CHECKs; REG3 with the layer-1 path disabled reproduces S56's REG|370 on H8 through 2400 bit for bit; with
every new path disabled REG3's gate at every layer equals LOCAL3's and its logits equal BASELINE's; init; register
causality at every layer and strict causality; delta beta 0 equals Hebbian; the SLOW groups; the nudge at all three layers.

Run:  python explore_g_register3.py [--report-only]
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_g_common as gc
import explore_g_rhl_parent as hp
import explore_g_layers as s71

NAME = "s72_register3"
CHILD = hp.CHILD
S20 = tuple(range(700, 720))
ARMS = {a: dict(iters=24000, seeds=S20, prio=1) for a in ("BASELINE", "REG3", "REG3_NUDGE")}
line = hp.line


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    print(hp.count_table(R, list(ARMS)))
    orc = gc.runs(s71, "ORACLE_ALL")
    nv = sum(hp.bound(r) for r in orc.values())
    valid = nv >= 8
    rb = hp.routed_bound
    rd = {"validity": f"S71 ORACLE_ALL {nv}/{len(orc)} -> {'VALID' if valid else 'UNTESTED'}"}
    n_rb = sum(rb(r) for r in R["REG3_NUDGE"].values())
    r1 = "the register carries the cue at all three layers" if n_rb >= 17 else "architecture insufficient" if n_rb < 12 else "neither"
    rd["R1 (REG3_NUDGE ROUTED-BOUND)"] = f"{r1 if valid else 'UNTESTED'} ({n_rb}/{len(R['REG3_NUDGE'])})"
    if r1 == "architecture insufficient":
        flat = []
        for r in R["REG3_NUDGE"].values():
            if hp.bound(r):
                continue
            c = [v["acc"] for v in r["acc_by_cell"].values()]
            flat.append(max(abs(x - statistics.fmean(c)) for x in c) <= 0.1)
        rd["R1 follow-up: unbound REG3_NUDGE runs with a flat (stream, block, index) pattern (every cell within 0.1 of the mean)"] = \
            f"{sum(flat)}/{len(flat)}"
    x, t = hp.beats(R["REG3"], R["BASELINE"], rb)
    rd["R2 discovered (REG3 beats BASELINE)"] = (("discovered" if x else "not discovered") if valid else "UNTESTED") + \
        f" (b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]})"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, list(ARMS)))
    print("\nper run:\n" + hp.per_run(R, list(ARMS)))
    with open(os.path.join(gc.G_DIR, "s72_summary.json"), "w") as f:
        json.dump(dict(readings=rd, valid=valid), f, indent=1)


if __name__ == "__main__":
    hp.drive(sys.modules[__name__], "S72")
