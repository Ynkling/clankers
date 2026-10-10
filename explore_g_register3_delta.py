#!/usr/bin/env python
"""
explore_g_register3_delta.py — EXPLORATORY, not a result. Session G follow-up, screen S73: the discovery devices on the
memory where routing pays (delta: beta 1 fixed, L2 keys, tied write, decay 0.95), on RandHeaderTask-lite, with the gate at
all three layers. Everything below was fixed and committed before any S73 run.

BACKGROUND. S56 (header P = 8) on delta memory: REG_NUDGE_D bound 8/10 (7 routed); REG_D bound 6/10 but only 2/10 routed,
3 by key splits. F's S70_DELTA (LOCAL3+SLOW, RandHeaderTask-lite, 330-349): BOUND 6/20, ROUTED*@end 0/20, KEY 14. S57's
latch ran on Hebbian memory only.

TASK, MODELS, RECIPES, STATISTICS, OUTCOMES: explore_g_rhl_child and explore_g_rhl_parent. Every arm here is SLOW (S56's
groups) except SINGLE_D (Adam 1e-3 throughout). No forget gate anywhere.
ARMS (seeds paired: the same batches and stack/LOCAL3 parameters; 20-seed arms on 700-719, 10-seed arms on 700-709)
  BASELINE_D      LOCAL3 + SLOW on delta (F's S70_DELTA recipe; F's counts are the reference, other seeds: counts only)   700-719
  REG3_D          Reg3BDH: one gate shared by layers 1-3 over [LN(r^(l)); window_t; R_{t-1}], S56's register, SG           700-719
  REG3_NUDGE_D    REG3_D + the 5% labelled nudge at all three layers (validity)                                             700-709
  RESGATE_ONLY_D  the per-layer residual gate without the register, at all three layers (can the gate read the stream tag
                  the delta memory writes into its own keys?)                                                              700-709
  LATCH3_D        S57's HM-RNN latch (slope 1 -> 5 over updates 8001-24000, no prior) feeding the gate at all three layers 700-709
  LATCH3_NUDGE_D  LATCH3_D + the nudge                                                                                      700-709
  SINGLE_D        one delta channel, no gate (k = 1, fixed decay; F's S53 / S68 single-channel path): is the gate needed?    700-709
  90 runs.
VALIDITY: S71's ORACLE_ALL_D (the perfect gate on delta, same seeds) binds >= 8/10 on 700-709; otherwise UNTESTED.
READINGS (ROUTED-BOUND = BOUND and F's ROUTED*@end; paired on the shared seeds; "beats" = one-sided exact McNemar p < 0.05)
  "the register carries the cue"        REG3_NUDGE_D ROUTED-BOUND >= 9/10
  "discovered (register)"               REG3_D beats BASELINE_D (700-719)
  "discovered (residual gate)"          RESGATE_ONLY_D beats BASELINE_D (700-709)
  "discovered (latch)"                  LATCH3_D beats BASELINE_D (700-709)
  "the latch holds on delta memory"     LATCH3_NUDGE_D's end z >= 0.9 at CTX and <= 0.1 at KEY and at VAL on >= 8/10 runs
  "the delta stack carries the cue"     BASELINE_D's in-run probe at the residual entering layer 3 >= 0.9 at K and at V on
                                        >= 10/20 runs (descriptive: the same count for every arm, and whether the runs whose
                                        probe reads the stream are the ones that route: a 2x2 of probe >= 0.9 x ROUTED-BOUND)
  "the gate is not needed on this task" SINGLE_D BOUND >= 8/10 (descriptive; write_order_acc printed)
  Printed beside every count: BOUND, ROUTED*@end, DISCOVERED, my BOUND ROUTED, failure classes, transitions.
ALSO REPORTED: beta^R / z at CTX / KEY / VAL by step; the routing margin at VAL per layer; the CTX / KEY / VAL x stream
contingency of g at every layer; held-out accuracy by (query stream, block, pair index) for every arm's unbound runs.
CHECKS (explore_g_rhl_child.checks_reg3(delta), asserted before any run, and X's arm A bit for bit): the task (F's
check_rand_header); the in-run probe; S56's 12 CHECKs; REG3 with the layer-1 path disabled reproduces S56's REG|370 on H8
through 2400 bit for bit; with every new path disabled REG3_D's gate at every layer equals LOCAL3's and its logits equal
BASELINE_D's; REG3_D's init; register and latch causality at every layer (layer 1 included) and strict causality; delta
beta 0 / write 1 / raw keys equals Hebbian (1e-5); the SLOW groups; the nudge at all three layers; LATCH3_D's dead-STE guard;
RESGATE_ONLY_D has no register; SINGLE_D equals F's single-channel path; BASELINE_D's memory.

Run:  python explore_g_register3_delta.py [--report-only]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_g_common as gc
import explore_g_rhl_parent as hp
import explore_g_layers as s71

NAME = "s73_register3_delta"
CHILD = hp.CHILD
S20, S10 = tuple(range(700, 720)), tuple(range(700, 710))
ARMS = {
    "BASELINE_D": dict(iters=24000, seeds=S20, prio=1),
    "REG3_D": dict(iters=24000, seeds=S20, prio=1),
    "REG3_NUDGE_D": dict(iters=24000, seeds=S10, prio=1),
    "RESGATE_ONLY_D": dict(iters=24000, seeds=S10, prio=1),
    "LATCH3_D": dict(iters=24000, seeds=S10, prio=1),
    "LATCH3_NUDGE_D": dict(iters=24000, seeds=S10, prio=1),
    "SINGLE_D": dict(iters=24000, seeds=S10, prio=1),
}
line = hp.line


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    print(hp.count_table(R, list(ARMS)))
    orc = gc.runs(s71, "ORACLE_ALL_D")
    nv = sum(hp.bound(r) for r in orc.values())
    valid = nv >= 8
    rb = hp.routed_bound
    fmt = lambda t: f"b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]}"
    rd = {"validity": f"S71 ORACLE_ALL_D {nv}/{len(orc)} -> {'VALID' if valid else 'UNTESTED'}"}
    ap = lambda x: ("APPLIES" if x else "does not apply") if valid else "UNTESTED"
    n_rb = sum(rb(r) for r in R["REG3_NUDGE_D"].values())
    rd["the register carries the cue (REG3_NUDGE_D >= 9/10)"] = f"{ap(n_rb >= 9)} ({n_rb}/{len(R['REG3_NUDGE_D'])})"
    for arm, nm in (("REG3_D", "register"), ("RESGATE_ONLY_D", "residual gate"), ("LATCH3_D", "latch")):
        x, t = hp.beats(R[arm], R["BASELINE_D"], rb)
        rd[f"discovered ({nm}): {arm} beats BASELINE_D"] = f"{ap(x)} ({fmt(t)})"
    zs = [r["end"]["beta"] for r in R["LATCH3_NUDGE_D"].values() if r["end"].get("beta")]
    nz = sum(z["CTX"] >= 0.9 and z["KEY"] <= 0.1 and z["VAL"] <= 0.1 for z in zs)
    rd["the latch holds on delta memory (LATCH3_NUDGE_D z, >= 8/10 runs)"] = f"{ap(nz >= 8)} ({nz}/{len(zs)})"
    pres = lambda r: r["probe"]["L3|K"]["acc"] >= 0.9 and r["probe"]["L3|V"]["acc"] >= 0.9
    npb = sum(pres(r) for r in R["BASELINE_D"].values())
    rd["the delta stack carries the cue (BASELINE_D probe L3 >= 0.9 at K and V on >= 10/20)"] = \
        f"{'APPLIES' if npb >= 10 else 'does not apply'} ({npb}/{len(R['BASELINE_D'])}; descriptive)"
    for arm, rr in R.items():
        if not rr or arm == "SINGLE_D":
            continue
        t2 = [[sum(1 for r in rr.values() if pres(r) == p and rb(r) == q) for q in (True, False)] for p in (True, False)]
        rd[f"probe x route, {arm}"] = (f"probe L3 >= 0.9 on {sum(pres(r) for r in rr.values())}/{len(rr)}; "
                                       f"[[probe & routed-bound, probe & not], [no probe & routed-bound, no probe & not]] = {t2}")
    ns = sum(hp.bound(r) for r in R["SINGLE_D"].values())
    rd["the gate is not needed on this task (SINGLE_D BOUND >= 8/10)"] = f"{'APPLIES' if ns >= 8 else 'does not apply'} ({ns}/{len(R['SINGLE_D'])}; descriptive)"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, list(ARMS)))
    print("\nper run:\n" + hp.per_run(R, list(ARMS)))
    with open(os.path.join(gc.G_DIR, "s73_summary.json"), "w") as f:
        json.dump(dict(readings=rd, valid=valid), f, indent=1)


if __name__ == "__main__":
    hp.drive(sys.modules[__name__], "S73")
