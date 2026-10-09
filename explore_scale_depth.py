#!/usr/bin/env python
"""
explore_scale_depth.py — EXPLORATORY, not a result. Screen S65 (batch 19): does the window recipe survive depth?

CONFIGURATION: S64's (explore_scale_width): S=4, P=4, k=16, conv, 28800 updates, LOCAL3 + SLOW unchanged, N=256, D=32;
only ARCH's n_layer changes. One gate shared across layers, as now: the LOCAL3 gate is computed once from the embeddings
and multiplies every layer's attention scores. Note (CHECK, explore_scale_child): bdh.BDH shares its weights across
layers (one encoder, encoder_v, decoder applied at every layer), so 2, 4 and 6 layers have exactly the parameters (and,
for a seed, the initial values) of 3 layers; depth is the number of times the shared layer is applied.
ARMS (seeds 280-289; the oracle on 280-281):
  L2, L3 (the reference, the current model), L4, L6;  ORC_L<n>: the oracle at that depth (the perfect gate, the recorded
  validity recipe), the same oracle rule as S64 (an untested depth is not run and takes no part in the reading).
  RES6 (diagnostic, printed, no reading): run only if 6 layers fails, which is fixed here as: L6 is tested and its BOUND
  ROUTED count is below L3's by 2 or more (it misses the "scale-free" margin of 1). Session G's per-layer gate over the
  residual stream at 6 layers (explore_b19_model.ResGateBDH: layer 1 keeps the window gate; layer l >= 2 gates by
  softmax(W_g tanh(A [RMSNorm(r^(l)); v_t])) with A = [0 | W_in] at init; W_g shared; the equations of
  docs/reading/distant_cues_2026-10.md 4.3), with the SLOW recipe (A in the gate group), seeds 280-289. Its outcome is
  measured on layer 1's gate (the run path's statistics); every layer's routing is printed at the end.
READING (the user's "same readings", on counts, d = arm's BOUND ROUTED - L3's over the paired seeds; exact McNemar
printed): "scale-free" if every tested depth has d >= -1; "depth-sensitive" if any tested depth has d <= -4; otherwise
neither. Transition medians per depth printed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19
import explore_scale_width as s64

NAME = "scale_depth"
CHILD = "explore_scale_child"
MAIN = True
IDEA = "does the window recipe survive depth? 2, 4 and 6 layers against 3 (one gate shared across layers)"
SOURCE = "batch 16's S44(a) (3 layers); the user's batch 19; RES6: Session G's design (docs/reading/distant_cues_2026-10.md 4.3)"
CHANGE = "only ARCH's n_layer (the shared BDH layer applied n times); RES6 adds a per-layer gate over the residual stream"
PAIRING = "seeds 280-289 with L3 (the same batches and initial parameters); oracles on 280-281"
LR = s64.LR
ITERS = s64.ITERS
SEEDS = tuple(range(280, 290))
ORACLE_SEEDS = (280, 281)
DEPTHS = (2, 3, 4, 6)
REF = "L3"
FAIL_D = -2
CUT = set()


def depth_desc(L):
    return f"N=256, D=32, {L} layers"


ARMS = {}
for _L in DEPTHS:
    ARMS[f"ORC_L{_L}"] = dict(key=f"ORC_L{_L}", spec=dict(cfg="k16", task="P4S4", gate="perfect", size=[8, 32, _L]),
                              seeds=ORACLE_SEEDS, iters=ITERS, lr=LR, opt="adam", role="oracle", size=f"L{_L}", sched=s64._OR,
                              label=f"oracle at {depth_desc(_L)}")
    ARMS[f"L{_L}"] = dict(key=f"L{_L}", spec=dict(cfg="k16", task="P4S4", gate="local", size=[8, 32, _L]), seeds=SEEDS, iters=ITERS,
                          lr=LR, opt="adam", role="ref" if _L == 3 else "arm", oracle=f"ORC_L{_L}", size=f"L{_L}", sched=s64._SL,
                          label=f"LOCAL3 + SLOW at {depth_desc(_L)}" + (" (the reference)" if _L == 3 else ""))
ARMS["RES6"] = dict(key="RES6", spec=dict(cfg="k16", task="P4S4", gate="resgate", size=[8, 32, 6]), seeds=SEEDS, iters=ITERS, lr=LR,
                    opt="adam", role="diag", needs="res6_state", size="L6", oracle="ORC_L6",
                    sched=s64._SL.replace("gate W_in/W_g/window", "gate W_in/W_g/window/A"),
                    label="Session G's per-layer residual gate at 6 layers (diagnostic; only if L6 fails)")


def res6_state(store):
    me = sys.modules[__name__]
    if not (c19.complete(me, "L3", store) and c19.complete(me, "L6", store)):
        return "wait", "waiting for L3 and L6 to complete"
    if c19.oracle_state(me, "ORC_L6", store) == "invalid":
        return "never", "L6 is UNTESTED (its oracle did not bind)"
    l3 = c16.runs_of(me, "L3", store=store)
    l6 = c16.runs_of(me, "L6", store=store)
    d = c19.counts_vs(l6, l3, SEEDS)["d"]
    if d <= FAIL_D:
        return "ready", f"6 layers fails (L6 - L3 = {d:+d} <= {FAIL_D})"
    return "never", f"6 layers did not fail (L6 - L3 = {d:+d} > {FAIL_D}): RES6 not run"


def res6_block(me):
    st = ec.load_store(me.NAME)
    state, why = c19.arm_state(me, "RES6", st, False)
    runs = c16.runs_of(me, "RES6", store=st)
    dry = me.NAME.startswith("dry_")
    print(f"  DIAGNOSTIC RES6 ({me.ARMS['RES6']['label']}): {why or state}"
          + (" (the dry run ran it regardless, to exercise the code)" if dry and runs and state != "ready" else ""))
    if not runs:
        return dict(state=state, why=why)
    seeds = me.ARMS["RES6"]["seeds"]
    c19.arm_table("RES6", runs, seeds, c16.runs_of(me, "L6", store=st), "L6")
    d = c19.counts_vs(runs, c16.runs_of(me, "L6", store=st), seeds)
    tm, tl = c19.trans_med(runs)
    print(f"    BOUND ROUTED (layer 1's gate) {d['new']}/{d['n']}; bound {sum(c19.bound(r) for r in runs.values())}/{len(runs)}; vs L6: "
          f"RES6 only {d['b']}, L6 only {d['c']}, p = {d['p']:.3g}; transitions {tl}; classes {c19.classes(runs)} (printed, no reading)")
    print("    every layer's routing at the end (stream -> channel map; eta^2 of the gate at VAL positions by stream / by key):")
    for s, r in sorted(runs.items()):
        lm = r.get("layer_maps") or []
        print(f"      {s}: " + "  ".join(f"L{i + 1} {c19.per_channel(x.get('ch_map'))} {x.get('etak_val_by_stream', float('nan')):.2f}/"
                                       f"{x.get('etak_val_by_key', float('nan')):.2f}" for i, x in enumerate(lm)))
    return dict(state=state, why=why, br=d["new"], n=d["n"])


def report():
    me = sys.modules[__name__]
    saved = dict(me.ARMS)
    me.ARMS = {k: a for k, a in saved.items() if k != "RES6"}
    try:
        out = s64.report_block(me, REF, "L3", "scale-free", "depth-sensitive",
                               lambda a: depth_desc(a["spec"]["size"][2]))
    finally:
        me.ARMS = saved
    out["res6"] = res6_block(me)
    print()
    return out
