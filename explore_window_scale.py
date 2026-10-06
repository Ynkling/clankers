#!/usr/bin/env python
"""
explore_window_scale.py — EXPLORATORY, not a result. Screen S44 (batch 16): the window gate with slow memory at four and
eight streams (conv on, 28800 steps; outcome BOUND ROUTED; S37's KEYMASS trigger OFF except in the SPLIT arm).

BACKGROUND: see explore_window_recipe (S43) and the audit of 6 October 2026, section 3.1: the window gate's state
carries the stream at every write position by construction; if Revision 7's diagnosis (the recurrent gate's state
loses the stream at eight streams) is right, it should bind on some eight-stream seeds where every recurrent arm bound
none.

ARMS (explore_window_scale_child; main's run paths at 9c5939e in child processes; Adam lr 1e-3 = test_slow_start.LR,
read in the child; Muon lr 0.005 = S33's validity choice; 1 torch thread per run):
(a) S=4, P=4, k=16, seeds 240-259:
    LOCAL3_SLOW16_A    LOCAL3 + SLOW, no hinge (Adam); paired with X's HINGE4k16 (17/20 BOUND).
    LOCAL3_SLOW16_M    LOCAL3 + S25's groups (Muon 0.005), no hinge; paired with MUON_HINGE16_REF.
    MUON_HINGE16_REF   S35's MUON_HINGE16 (S35's recipe), fresh on this CPU (a reference; no reading).
(b) S=8, P=4, k=16, all 8 streams from step 1, seeds 260-269:
    LOCAL3_SLOW_D8_A   LOCAL3 + SLOW, no hinge (Adam); paired with X's HINGE_D8 (0/10).
    LOCAL3_SLOW_D8_M   LOCAL3 + S25's groups (Muon 0.005), no hinge; paired with D8_HINGE_M_REF.
    LOCAL3_SLOW_D8_A_SPLIT  (the user's addition) LOCAL3_SLOW_D8_A with S37's KEYMASS plateau trigger ON (checks at
                       4800, 7200, ...; at most 2 splits per run, >= 4800 apart); paired with LOCAL3_SLOW_D8_A; scored
                       BOUND ROUTED. Cut before any other S44 arm if the projection is over.
    D8_HINGE_M_REF     S33's D8_HINGE (S33's recipe), fresh on this CPU (a reference; no reading; also S45's Muon pair).
A reference arm runs only while an arm it pairs with runs.
READINGS (BOUND ROUTED at 28800; fixed before any run):
(a) "carries to four streams" if either arm has >= 16/20 and is not worse than its reference by more than 1 discordant
    pair (reference only - arm only <= 1); otherwise "not shown".
(b) "breaks the eight-stream stall" if either of LOCAL3_SLOW_D8_A and LOCAL3_SLOW_D8_M has >= 3/10; "does not" if both
    have 0/10; otherwise neither. An arm cut by the runtime rule drops out of "either"/"both".
    LOCAL3_SLOW_D8_A_SPLIT: its own line with the same thresholds (>= 3/10, 0/10), and McNemar against LOCAL3_SLOW_D8_A;
    it does not enter (b)'s "either arm".
DIAGNOSTICS: decodability of the gate state (S38's decoder) at CTX, KEY and VAL positions at 1200/2400/4800/9600;
distinct channels and streams per channel at 4800, 9600 and the end; merges (failure classes); transitions; the SPLIT
arm's checks and splits.
CHECKS (explore_window_scale_child.checks): LOCAL3's conversion keeps every parameter of the run path's model and adds
batch 1's window; the gate at t ignores t-3 and sees t-2; with the window at [1, 0, 0] and W_h = 0 it equals arm A's gate
with W_h = 0; the Adam and Muon groups cover every trainable parameter once (gate group W_in, W_g, window; W_h frozen, in
no group, unchanged) with their lrs at updates 1, 2400, 2401 through the real paths; muon_groups with the recurrent
gate's names equals S33's groups; (if the SPLIT arm runs) with the trigger's threshold at 0 the SPLIT arm equals
LOCAL3_SLOW_D8_A bit for bit through 6000.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_b16_report as rp

NAME = "window_scale"
CHILD = "explore_window_scale_child"
MAIN = True
IDEA = "the window gate (LOCAL3) with slow memory, no hinge, at four and eight streams (k=16, conv)"
SOURCE = "the audit of 6 October 2026, sections 3.1 and 3.2 (the window sees CTX at the read and the write positions)"
CHANGE = ("main's A4k16 / D8 paths with LOCAL3's gate in place of the recurrent gate, the SLOW schedule (Adam) or S25's "
          "groups (Muon 0.005), no hinge; SPLIT: + S37's KEYMASS trigger")
PAIRING = ("(a) seeds 240-259: Adam with X's HINGE4k16, Muon with MUON_HINGE16_REF (S35's recipe, fresh on this CPU); (b) "
           "seeds 260-269: Adam with X's HINGE_D8, Muon with D8_HINGE_M_REF (S33's recipe, fresh on this CPU), SPLIT with "
           "LOCAL3_SLOW_D8_A; same initial parameters (+ the window) and batches")
LR = 1e-3
MUON_LR = 0.005
ITERS = 28800
SA = tuple(range(240, 260))
SB = tuple(range(260, 270))
CARRY_N, CARRY_D = 16, 1
BREAK_N, NOT_N = 3, 0
SPLIT_CHECK_ITERS = 6000
_sl = (f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} "
       f"after; W_h frozen (unused); no hinge")
_mu = (f"Muon lr {MUON_LR:g}: W_in/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam: the window {LR:g} "
       f"throughout, embedding/conv/1-D {LR / 10:g} for 1-2400, {LR:g} after; W_h frozen; no hinge")
_ref = (f"Muon lr {MUON_LR:g}: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding "
        f"and conv {LR / 10:g} for 1-2400, {LR:g} after; + main's HINGE hinge")
_trig = ("; + S37's KEYMASS trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02; <= 2 splits, >= 4800 apart) -> "
         "S24's SPLIT of W_g's row c* onto c0, W_g's Adam state zeroed")
ARMS = {
    "LOCAL3_SLOW16_A": dict(key="LOCAL3_SLOW16_A", part="a", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SA,
                            label="LOCAL3 + SLOW, no hinge (Adam), S=4, k=16", sched=_sl),
    "LOCAL3_SLOW16_M": dict(key="LOCAL3_SLOW16_M", part="a", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SA,
                            label="LOCAL3 + S25's groups (Muon), no hinge, S=4, k=16", sched=_mu),
    "MUON_HINGE16_REF": dict(key="MUON_HINGE16_REF", part="a", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SA,
                             label="S35's MUON_HINGE16, fresh on this CPU (reference)", sched=_ref,
                             ref=True, needed_by=(("window_scale", "LOCAL3_SLOW16_M"),)),
    "LOCAL3_SLOW_D8_A": dict(key="LOCAL3_SLOW_D8_A", part="b", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB,
                             label="LOCAL3 + SLOW, no hinge (Adam), S=8, k=16", sched=_sl),
    "LOCAL3_SLOW_D8_M": dict(key="LOCAL3_SLOW_D8_M", part="b", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB,
                             label="LOCAL3 + S25's groups (Muon), no hinge, S=8, k=16", sched=_mu),
    "LOCAL3_SLOW_D8_A_SPLIT": dict(key="LOCAL3_SLOW_D8_A_SPLIT", part="b", opt="adam", lr=LR, muon_lr=None, iters=ITERS,
                                   seeds=SB, label="LOCAL3_SLOW_D8_A + S37's KEYMASS trigger", sched=_sl + _trig),
    "D8_HINGE_M_REF": dict(key="D8_HINGE_M_REF", part="b", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB,
                           label="S33's D8_HINGE, fresh on this CPU (reference)", sched=_ref, ref=True,
                           needed_by=(("window_scale", "LOCAL3_SLOW_D8_M"), ("reservoir_gate", "RES_D8_M"))),
}
CUT = set()                                    # arms cut by the runtime rule (set by the driver)


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300, mlr=MUON_LR))


def n_meas(arm):
    return 4


def n_extra(arm):
    return 11 if arm.endswith("SPLIT") else 0              # the trigger's checks (2400 .. 26400), about an evaluation each


def split_check():
    """With the trigger's threshold at 0 (it never fires; its measurements still run), LOCAL3_SLOW_D8_A_SPLIT equals
    LOCAL3_SLOW_D8_A bit for bit through SPLIT_CHECK_ITERS (curve, statistics, decoder); two children at once."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(child, "run", dict(arm="LOCAL3_SLOW_D8_A_SPLIT", seed=260, iters=SPLIT_CHECK_ITERS, thr=0.0))
        f2 = ex.submit(child, "run", dict(arm="LOCAL3_SLOW_D8_A", seed=260, iters=SPLIT_CHECK_ITERS))
        r1, r2 = f1.result(), f2.result()
    same = r1["curve"] == r2["curve"] and r1["stats"] == r2["stats"] and r1["decode"] == r2["decode"]
    return [(f"with the trigger's threshold at 0, LOCAL3_SLOW_D8_A_SPLIT equals LOCAL3_SLOW_D8_A bit for bit through "
             f"{SPLIT_CHECK_ITERS} (curves {r1['curve']} vs {r2['curve']}; statistics and decoder equal {same}; checks logged at "
             f"{[c['step'] for c in r1['checks']]}, {r1['splits']} splits)",
             same and r1["splits"] == 0 and len(r1["checks"]) >= 2)]


def check():
    rows = [tuple(x) for x in child("checks", dict(mlr=MUON_LR))]
    if "LOCAL3_SLOW_D8_A_SPLIT" in ARMS:
        rows += split_check()
    else:
        rows.append(("the SPLIT arm's inertness check is skipped: the arm is cut", True))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
_X = {}


def x_runs(prefix, seeds):
    keys = [f"{prefix}|{s}" for s in seeds]
    need = [k for k in keys if k not in _X]
    if need:
        _X.update(child("recorded", dict(keys=need)))
    return {int(k.split("|")[1]): _X[k] for k in keys if k in _X}


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    got = {arm: c16.runs_of(me, arm) for arm in ARMS}
    xa, xb = x_runs("HINGE4k16", SA), x_runs("HINGE_D8", SB)
    pair = {"LOCAL3_SLOW16_A": ("X HINGE4k16", xa), "LOCAL3_SLOW16_M": ("MUON_HINGE16_REF", got.get("MUON_HINGE16_REF", {})),
            "LOCAL3_SLOW_D8_A": ("X HINGE_D8", xb), "LOCAL3_SLOW_D8_M": ("D8_HINGE_M_REF", got.get("D8_HINGE_M_REF", {})),
            "LOCAL3_SLOW_D8_A_SPLIT": ("LOCAL3_SLOW_D8_A", got.get("LOCAL3_SLOW_D8_A", {})),
            "MUON_HINGE16_REF": ("X HINGE4k16", xa), "D8_HINGE_M_REF": ("X HINGE_D8", xb)}
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs, no reading)")
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        plab, pr = pair[arm]
        rp.arm_table(arm, rs, a["seeds"], [(plab, pr)], a["label"])
        d = rp.compare(rs, pr, a["seeds"], arm, plab.replace(" ", "_"))
        print(f"    failure classes at the end: {arm} {rp.classes(rs)}; {plab} {rp.classes({s: pr[s] for s in a['seeds'] if s in pr})}")
        print(f"    transitions: {rp.transitions(rs)}")
        rp.maps_block(rs)
        meds = rp.decode_block(rs, chance=1 / (4 if a["part"] == "a" else 8))
        if arm.endswith("SPLIT"):
            print("    the trigger's checks and splits (step: probe acc, eligible/fired/blocked, KEYMASS c* -> c0, labelled "
                  "targeting ok, map before -> after):")
            for s, r in sorted(rs.items()):
                fired = [c for c in r.get("checks") or [] if c.get("fired")]
                print(f"      {s}: {r.get('splits')} split(s) " + "; ".join(
                    f"{c['step']}: acc {c['acc']:.3f} c* {c['key_cs']} -> c0 {c['key_c0']} ok {c['key_ok']} "
                    f"{rp.s33.per_channel(c['map_before'])} -> {rp.s33.per_channel(c.get('map_after'))}" for c in fired)
                    + (f"; blocked at {[c['step'] for c in r.get('checks') or [] if c.get('blocked')]}"
                       if any(c.get('blocked') for c in r.get('checks') or []) else ""))
        nb, n = sum(rp.br(r) for r in rs.values()), len(rs)
        out[arm] = dict(d, bound_routed=nb, n=n, complete=n == len(a["seeds"]), classes=rp.classes(rs), meds=meds)
        if a.get("ref"):
            print(f"  S44 {arm}: reference (no reading): BOUND ROUTED {nb}/{n}")
    # readings
    ra = [arm for arm in ("LOCAL3_SLOW16_A", "LOCAL3_SLOW16_M") if arm in ARMS]

    def counts(arms):
        return ", ".join(f"{arm} {out[arm]['n']}/{len(ARMS[arm]['seeds'])}" for arm in arms)
    if any(not out[arm]["complete"] for arm in ra) or not ra:
        rda = f"INCOMPLETE ({counts(ra)}), no reading"
    else:
        hit = [arm for arm in ra if out[arm]["bound_routed"] >= CARRY_N and out[arm]["c"] - out[arm]["b"] <= CARRY_D]
        rda = (("carries to four streams" if hit else "not shown") + " ("
               + "; ".join(f"{arm} {out[arm]['bound_routed']}/{out[arm]['n']}, reference only {out[arm]['c']}, arm only "
                           f"{out[arm]['b']}" for arm in ra)
               + f"; carries: >= {CARRY_N}/20 and reference only - arm only <= {CARRY_D}"
               + ("; LOCAL3_SLOW16_M cut by the runtime rule" if "LOCAL3_SLOW16_M" in CUT else "") + ")")
    print(f"  READING S44(a): {rda}")
    rb = [arm for arm in ("LOCAL3_SLOW_D8_A", "LOCAL3_SLOW_D8_M") if arm in ARMS]
    if any(not out[arm]["complete"] for arm in rb):
        rdb = f"INCOMPLETE ({counts(rb)}), no reading"
    else:
        nbs = [out[arm]["bound_routed"] for arm in rb]
        rdb = (("breaks the eight-stream stall" if max(nbs) >= BREAK_N else "does not" if max(nbs) == NOT_N else "neither")
               + " (" + "; ".join(f"{arm} {out[arm]['bound_routed']}/{out[arm]['n']}" for arm in rb)
               + f"; breaks: either >= {BREAK_N}/10; does not: both 0/10"
               + ("; LOCAL3_SLOW_D8_M cut by the runtime rule" if "LOCAL3_SLOW_D8_M" in CUT else "") + ")")
    print(f"  READING S44(b): {rdb}")
    rds = None
    if "LOCAL3_SLOW_D8_A_SPLIT" in ARMS:
        o = out["LOCAL3_SLOW_D8_A_SPLIT"]
        if not o["complete"]:
            rds = f"INCOMPLETE ({o['n']}/{len(ARMS['LOCAL3_SLOW_D8_A_SPLIT']['seeds'])}), no reading"
        else:
            nb = o["bound_routed"]
            rds = (("breaks the eight-stream stall" if nb >= BREAK_N else "does not" if nb == NOT_N else "neither")
                   + f" (BOUND ROUTED {nb}/{o['n']}; vs LOCAL3_SLOW_D8_A: SPLIT only {o['b']}, LOCAL3_SLOW_D8_A only {o['c']}, "
                     f"p = {o['p']:.3g})")
        print(f"  READING S44(b) LOCAL3_SLOW_D8_A_SPLIT (own line, same thresholds): {rds}")
    return dict(a=rda, b=rdb, split=rds, arms=out)
