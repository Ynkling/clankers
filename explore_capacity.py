#!/usr/bin/env python
"""
explore_capacity.py — EXPLORATORY, not a result. Screen S67 (batch 19): is four-stream binding capacity-bound at N=256?

CONFIGURATION (explore_scale_child, "k16" on longer tasks): test_stream_recipe.run_attempt with ARM["A4k16"] (S=4, k=16,
conv 'layer' width 4, evaluation every 1200, early stop after 4800) on test_channel_binding.task_for(P, 4) for P = 8 and
P = 16 (grouped layout, n_vals 16, n_q 1; length 99 and 195), 43200 updates (the budget). Seeds 270-279; the oracle on
270-271.
ARMS (fixed by the driver from S64 when it is complete: W = S64's best width, by explore_scale_width.best_candidate):
  ORC_P8_N256, ORC_P16_N256, ORC_P8_<W>, ORC_P16_<W>   the oracle (the perfect gate, the recorded validity recipe: a single
                                                        Adam at 1e-3), first
  P8_N256, P16_N256, P8_<W>, P16_<W>                   LOCAL3 + SLOW (batch 16's recipe, unchanged)
ORACLE RULE (the user's): a learned arm runs once its oracle has BOUND on a seed; if the oracle binds on neither seed
within 43200 updates the arm is UNTESTED (not run).
READING (the user's, fixed before any run; on the oracle): "capacity-bound at N=256" if the oracle binds P=16 only at the
larger width (ORC_P16_N256 0/2 and ORC_P16_<W> >= 1/2). Otherwise printed as it is: "not capacity-bound at N=256" (the
oracle binds P=16 at N=256) or "the oracle binds P=16 at neither width". The learned arms' BOUND ROUTED counts are printed
alongside (paired by seed: <W> against N256 at each P), with no reading.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19
import explore_scale_width as s64

NAME = "capacity"
CHILD = "explore_scale_child"
MAIN = True
IDEA = "is four-stream binding capacity-bound at N=256? P = 8 and 16 at N=256 and at S64's best width, oracle first"
SOURCE = "the user's batch 19 (budget 43200; the oracle-first rule)"
CHANGE = "the task's P (8, 16) and the model's size; recipes unchanged"
PAIRING = "seeds 270-279: <W> with N256 at each P (the same batches); oracles on 270-271"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(270, 280))
ORACLE_SEEDS = (270, 271)
PS = (8, 16)
ARMS = {}
CUT = set()
DECISION = {}


def configure(dec):
    DECISION.clear()
    DECISION.update(dec or {})
    ARMS.clear()
    if dec is None:
        return
    widths = ["N256"] + ([dec["best_width"]] if dec.get("best_width") else [])
    for P in PS:
        for w in widths:
            mult, d = s64.SIZES[w]
            ARMS[f"ORC_P{P}_{w}"] = dict(key=f"ORC_P{P}_{w}", spec=dict(cfg="k16", task=f"P{P}S4", gate="perfect", size=[mult, d, 3]),
                                         seeds=ORACLE_SEEDS, iters=ITERS, lr=LR, opt="adam", role="oracle", sched=s64._OR,
                                         label=f"oracle, P={P}, {w} (N={mult * d}, D={d})")
    for P in PS:
        for w in widths:
            mult, d = s64.SIZES[w]
            ARMS[f"P{P}_{w}"] = dict(key=f"P{P}_{w}", spec=dict(cfg="k16", task=f"P{P}S4", gate="local", size=[mult, d, 3]),
                                     seeds=SEEDS, iters=ITERS, lr=LR, opt="adam", role="ref" if w == "N256" else "arm",
                                     oracle=f"ORC_P{P}_{w}", sched=s64._SL,
                                     label=f"LOCAL3 + SLOW, P={P}, {w} (N={mult * d}, D={d})")


def cap_reading(o256, ow, w):
    """The user's reading on the P=16 oracles: o256 / ow = dict(bound, complete) at N256 / at the best width w."""
    if o256 is None or not o256["complete"] or (w and (ow is None or not ow["complete"])):
        return "INCOMPLETE, no reading"
    if not w:
        return "UNTESTED (S64 gave no larger width)"
    if o256["bound"] == 0 and ow["bound"] >= 1:
        return f"capacity-bound at N=256 (the oracle binds P=16 at {w} {ow['bound']}/{ow.get('n', 2)} and at N256 0/{o256.get('n', 2)})"
    if o256["bound"] >= 1:
        return (f"not capacity-bound at N=256 (the oracle binds P=16 at N256 {o256['bound']}/{o256.get('n', 2)}; at {w} "
                f"{ow['bound']}/{ow.get('n', 2)})")
    return f"the oracle binds P=16 at neither width (N256 0/{o256.get('n', 2)}, {w} 0/{ow.get('n', 2)})"


def report():
    import explore_common2 as c2
    me = sys.modules[__name__]
    if not ARMS:
        print("-" * 100)
        print(f"[{ec.BANNER}] SCREEN {NAME}: not configured yet (S64 must complete first)")
        return {}
    c2.print_screen_header2(me)
    print(f"  phase-1 decision: {DECISION}")
    st = ec.load_store(NAME)
    w = DECISION.get("best_width")
    orc = {}
    print("  ORACLES (BOUND = held-out accuracy >= 0.95 held to the end, within 43200 updates):")
    for arm, a in ARMS.items():
        if a["role"] != "oracle":
            continue
        runs = c16.runs_of(me, arm, store=st)
        nb = sum(c19.bound(r) for r in runs.values())
        comp = c19.complete(me, arm, st)
        orc[arm] = dict(bound=nb, n=len(runs), complete=comp)
        print(f"    {arm:<16} bound {nb}/{len(runs)} of {len(a['seeds'])}; transitions {[r.get('transition') for _, r in sorted(runs.items())]}; "
              f"final accuracy {[round(r.get('acc', 0), 3) for _, r in sorted(runs.items())]}; stop {[r.get('stopped_at') for _, r in sorted(runs.items())]}"
              f" -> {'VALID' if nb else ('UNTESTED' if comp else 'pending')}")
    out = dict(oracles=orc)
    for P in PS:
        base = c16.runs_of(me, f"P{P}_N256", store=st) if f"P{P}_N256" in ARMS else {}
        for arm in [k for k in ARMS if k.startswith(f"P{P}_")]:
            a = ARMS[arm]
            state, why = c19.arm_state(me, arm, st, False)
            runs = c16.runs_of(me, arm, store=st)
            print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}" + (f": {why}" if state != "ready" else ""))
            if runs:
                c19.arm_table(arm, runs, a["seeds"], None if arm.endswith("N256") else base, None if arm.endswith("N256") else f"P{P}_N256")
                tm, tl = c19.trans_med(runs)
                print(f"    BOUND ROUTED {sum(c19.br(r) for r in runs.values())}/{len(runs)}; transitions (BOUND ROUTED) {tl}; classes "
                      f"{c19.classes(runs)}")
                if not arm.endswith("N256") and base:
                    d = c19.counts_vs(runs, base, a["seeds"])
                    print(f"    vs P{P}_N256 (printed, no reading): {arm} only {d['b']}, P{P}_N256 only {d['c']}, p = {d['p']:.3g}")
            out[arm] = dict(state=state, why=why, br=sum(c19.br(r) for r in runs.values()), n=len(runs))
    rd = cap_reading(orc.get("ORC_P16_N256"), orc.get(f"ORC_P16_{w}") if w else None, w)
    print(f"  READING S67: {rd}")
    out["reading"] = rd
    print()
    return out
