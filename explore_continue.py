#!/usr/bin/env python
"""
explore_continue.py — EXPLORATORY, not a result. Screen S68 (batch 20): are batch 19's unbound P = 16 runs slow or stuck?

QUESTION (the user's batch 20): S67's lead, P = 16 at D = 128 against D = 32 (N = 256; LOCAL3 + SLOW, 4/10 against 0/10,
4 / 0, p = 0.125), has a confound: 12 of its 16 failures ended with a one-to-one stream -> channel map and unbound streams
(STREAM-PARTIAL one-to-one) at the 43,200 budget, and the N = 256 oracle needed 24,000-27,600 updates with the routing
given. S68 runs every unbound S67 learned run on to 86,400 under the same recipe, so the main-line test can fix its budget.

HOW (a deviation from the specification, agreed with the user before any run): batch 19 deleted each run's checkpoint when
the run was saved, so no 43,200 checkpoint exists. The training loop does not depend on the run's total (no schedule reads
it: test_binding_onset.onset_run; the SLOW switch is at 2400), so a run of the same arm and seed with iters 86,400 is the
S67 run through 43,200. S68 therefore regenerates every run from update 0, and a run counts only if it REPRODUCES its S67
record through 43,200 bit for bit: the held-out curve (every evaluation, accuracy and loss), the statistics at every
evaluation, the early curve, and the SHA-1 of every weight at 43,200 equal to the recorded final weights' SHA-1. A run
that does not reproduce is printed NOT REPRODUCED and takes no part in any reading.
  S67 had no trigger at S = 4 (no checks, no splits): "the same trigger schedule" and "spent splits" have nothing to apply
to; the MERGED runs are MERGED with no split ever fired.

THE REPRODUCTION OF A RESUMED RUN (the user's premise check, kept): C_P8_N256 seed 272 halts right after its checkpoint at
43,200 and a new process resumes it from that checkpoint; RCHK_P8_N256 seed 272 is the same run computed without a
checkpoint after 42,000 (so its 43,200 -> 44,400 updates are computed in one process). The two must agree at 44,400 bit for
bit: the evaluation (accuracy, loss), the statistics, the per-stream accuracy and the weights' SHA-1.

ARMS (S67's spec, recipe and seeds; iters 86,400; evaluation every 1200; early stop as before):
  C_P16_N256      P = 16, N = 256, D = 32, LOCAL3 + SLOW; seeds 270-279 (S67: 6 STREAM-PARTIAL one-to-one: 270, 272, 273,
                  274, 275, 276; 4 MERGED: 271, 277, 278, 279)
  C_P16_D128      P = 16, N = 256, D = 128, LOCAL3 + SLOW; seeds 271, 272, 273, 275, 276, 279 (all STREAM-PARTIAL one-to-one)
  C_P8_N256       P = 8, N = 256, D = 32; seed 272 (MERGED (2 share) at 0.74); the halt at 43,200 above
  C_ORC_P16_D128  the oracle (the perfect gate, a single Adam at 1e-3), P = 16, D = 128; seed 270 (not bound, 0.32)
  RCHK_P8_N256    the reproduction's uninterrupted run, seed 272, to 44,400
MEASURED: outcome (test_stream_recipe.outcome; BOUND ROUTED = transition, a one-to-one map and every stream >= 0.9 at the
end) with the transition; the per-stream held-out accuracy at 43,200 and every 4800 after (48,000, ..., 86,400).

READINGS (fixed before any S68 run; on the one-to-one runs; "binds" = ends BOUND ROUTED by 86,400):
  per width W (N256: seeds 270, 272-276; D128: 271-273, 275, 276, 279), k_W = how many of its 6 bind:
    "slow, not stuck"  k_W >= 4;   "stuck"  k_W = 0;   "unresolved"  1 <= k_W <= 3.
  joint:  "the budget is the lead"  k_D128 >= 4 and k_N256 >= 4 (S67's 4/10 against 0/10 was a budget effect);
          "the width is the lead"   k_D128 >= 4 and k_N256 <= 1;
          "neither"                 every other pair (named here so each outcome has exactly one name).
  A width with a run NOT REPRODUCED or not complete has no reading ("INCOMPLETE"), and then neither does the joint one.
  BUDGET for the main-line test: B = the latest transition among every BOUND ROUTED P = 16 learned run (S67 and S68, both
  widths) + 2400 (one check interval), rounded up to a multiple of 2400. Printed with it: "86,400 is not enough" if any of
  the 12 one-to-one runs is unbound at 86,400 with its held-out accuracy still rising (>= 0.02 from 76,800 to 86,400).
DESCRIPTIVE (no reading): the four MERGED P16_N256 runs, P8_N256 272, the D128 oracle on seed 270.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19

NAME = "s67_continue"
CHILD = "explore_continue_child"
MAIN = True
IDEA = "are batch 19's unbound P = 16 runs slow or stuck? every unbound S67 learned run on to 86,400, same recipe"
SOURCE = "batch 19's S67 (P16_D128 4/10 vs P16_N256 0/10; 12 of 16 failures one-to-one and unbound); the user's batch 20"
CHANGE = "only the budget (43,200 -> 86,400); every run regenerated from update 0 and checked against its S67 record"
PAIRING = "each run is its own S67 run (same arm, seed, recipe), continued"
LR = 1e-3
ITERS = 86400
S67_ITERS = 43200
S67_NAME = "capacity"
S67_SHA = "17e56410a1db"
MARKS = tuple(range(43200, 86400 + 1, 4800))
REPRO_AT = 44400
ONE2ONE = {"N256": (270, 272, 273, 274, 275, 276), "D128": (271, 272, 273, 275, 276, 279)}
MERGED_N256 = (271, 277, 278, 279)
SLOW_K, STUCK_K, WIDTH_N256_MAX = 4, 0, 1
STEP_B = 2400
RISE = 0.02
RISE_FROM = 76800
CUT = set()
_SL = ("gate W_in/W_g/window 0.001 throughout; every other trainable parameter 0.0001 for updates 1-2400, 0.001 after; "
       "W_h frozen (unused); no hinge; no trigger (as S67)")
_OR = "the perfect gate; a single Adam at 0.001 on every parameter (the recorded validity recipe)"


UPTO = S67_ITERS          # the reproduction against S67 runs through this step (a dry run: its own iters)
HALT_AT = S67_ITERS       # C_P8_N256 272 halts here and resumes in a new process


def _spec(P, mult, d, gate, marks, sha_at, **extra):
    return dict(cfg="k16", task=f"P{P}S4", gate=gate, size=[mult, d, 3], marks=list(marks), sha_at=list(sha_at), **extra)


def build_arms(iters=ITERS, marks=MARKS, halt_at=HALT_AT, repro_at=REPRO_AT, ckpt_until=42000, seeds=None):
    """The arms (a dry run passes small steps and one seed per arm)."""
    sd = seeds or {}
    m2 = sorted(set(marks) | {repro_at})
    return {
        "RCHK_P8_N256": dict(key="RCHK_P8_N256", spec=_spec(8, 8, 32, "local", [halt_at, repro_at], [halt_at, repro_at],
                                                            ckpt_until=ckpt_until),
                             seeds=sd.get("RCHK_P8_N256", (272,)), iters=repro_at, lr=LR, opt="adam", role="check", sched=_SL,
                             s67="P8_N256", label=f"the reproduction's run computed through (no checkpoint after {ckpt_until}), "
                                                  f"P=8, N256, to {repro_at}"),
        "C_P8_N256": dict(key="C_P8_N256", spec=_spec(8, 8, 32, "local", m2, [halt_at, repro_at], halt_once_at=halt_at),
                          seeds=sd.get("C_P8_N256", (272,)), iters=iters, lr=LR, opt="adam", role="arm", sched=_SL, s67="P8_N256",
                          label=f"P=8, N256 (N=256, D=32): halts at {halt_at}, a new process resumes it"),
        "C_ORC_P16_D128": dict(key="C_ORC_P16_D128", spec=_spec(16, 2, 128, "perfect", marks, [halt_at]),
                               seeds=sd.get("C_ORC_P16_D128", (270,)), iters=iters, lr=LR, opt="adam", role="oracle", sched=_OR,
                               s67="ORC_P16_D128", label="oracle, P=16, D128 (N=256, D=128)"),
        "C_P16_N256": dict(key="C_P16_N256", spec=_spec(16, 8, 32, "local", marks, [halt_at]),
                           seeds=sd.get("C_P16_N256", tuple(range(270, 280))), iters=iters, lr=LR, opt="adam", role="arm",
                           sched=_SL, s67="P16_N256", label="LOCAL3 + SLOW, P=16, N256 (N=256, D=32)"),
        "C_P16_D128": dict(key="C_P16_D128", spec=_spec(16, 2, 128, "local", marks, [halt_at]),
                           seeds=sd.get("C_P16_D128", ONE2ONE["D128"]), iters=iters, lr=LR, opt="adam", role="arm", sched=_SL,
                           s67="P16_D128", label="LOCAL3 + SLOW, P=16, D128 (N=256, D=128)"),
    }


ARMS = build_arms()


def make_dry(iters=2400):
    """The labelled dry run: one seed per arm, every step scaled down (halt and the first mark at 1200, the end at 2400)."""
    global NAME, UPTO, HALT_AT, REPRO_AT, ITERS, RISE_FROM, ARMS
    NAME = "dry_" + NAME
    UPTO, HALT_AT, REPRO_AT, ITERS, RISE_FROM = iters, iters // 2, iters, iters, iters // 2
    ARMS = build_arms(iters=iters, marks=(iters // 2, iters), halt_at=iters // 2, repro_at=iters, ckpt_until=0,
                      seeds={"C_P16_N256": (270,), "C_P16_D128": (271,)})


# ── The S67 records ──────────────────────────────────────────────────────────
def s67_store():
    return ec.load_store(S67_NAME)


def s67_record(arm, seed, store=None):
    st = s67_store() if store is None else store
    for k, r in st["runs"].items():
        a, s, h, _ = k.split("|", 3)
        if a == arm and int(s) == seed and h == S67_SHA and r.get("ok"):
            return r
    return None


def reproduced(r, rec, upto=None):
    """(ok, why): r (S68) equals rec (S67) through `upto`: curve, statistics, early curve, the weights' SHA-1 at `upto`."""
    if rec is None:
        return False, "no S67 record"
    if not r or not r.get("ok"):
        return False, "S68 run not ok"
    upto = UPTO if upto is None else upto
    upto = min(upto, rec.get("stopped_at") or upto)
    c1 = [x for x in r.get("curve") or [] if x[0] <= upto]
    c0 = [x for x in rec.get("curve") or [] if x[0] <= upto]
    if c1 != c0:
        return False, f"curve differs (first at {next((a[0] for a, b in zip(c1, c0) if a != b), min(len(c1), len(c0)))})"
    n = len(c0) + 1                                                      # stats[0] is update 0
    if (r.get("stats") or [])[:n] != (rec.get("stats") or [])[:n]:
        return False, "statistics differ"
    if [x for x in r.get("early") or [] if x[0] <= upto] != [x for x in rec.get("early") or [] if x[0] <= upto]:
        return False, "early curve differs"
    sh = dict((s, h) for s, h in r.get("shas") or [])
    if upto == rec.get("stopped_at") and sh.get(upto) != rec.get("w_sha"):
        return False, f"weights' SHA-1 at {upto} differs from the recorded final weights' ({sh.get(upto)} vs {rec.get('w_sha')})"
    return True, ""


def mark_rows(r):
    return {s: v for s, v in r.get("marks") or []}


def acc_at(r, step):
    for s, a, _ in r.get("curve") or []:
        if s == step:
            return a
    return None


def br(r):
    return c19.br(r)


# ── Readings ─────────────────────────────────────────────────────────────────
def width_reading(k, complete):
    if not complete:
        return "INCOMPLETE"
    if k >= SLOW_K:
        return "slow, not stuck"
    if k == STUCK_K:
        return "stuck"
    return "unresolved"


def joint_reading(kd, kn, complete):
    if not complete:
        return "INCOMPLETE, no reading"
    if kd >= SLOW_K and kn >= SLOW_K:
        return "the budget is the lead"
    if kd >= SLOW_K and kn <= WIDTH_N256_MAX:
        return "the width is the lead"
    return "neither"


def budget_of(transitions):
    if not transitions:
        return None
    t = max(transitions) + STEP_B
    return int(math.ceil(t / STEP_B) * STEP_B)


def still_rising(r):
    a1, a0 = acc_at(r, ITERS), acc_at(r, RISE_FROM)
    return (not br(r)) and a1 is not None and a0 is not None and a1 - a0 >= RISE


def evaluate_all(store=None):
    """Every number the report and the readings use, from the stores."""
    me = sys.modules[__name__]
    st = ec.load_store(NAME) if store is None else store
    s67 = s67_store()
    out = dict(rows={}, valid={}, why={})
    for arm, a in ARMS.items():
        runs = c16.runs_of(me, arm, store=st)
        for s in a["seeds"]:
            r = runs.get(s)
            rec = s67_record(a["s67"], s, s67)
            ok, why = (reproduced(r, rec) if r else (False, "not run"))
            out["rows"][(arm, s)] = r
            out["valid"][(arm, s)] = ok
            out["why"][(arm, s)] = why
    ks, comp = {}, {}
    for w, arm in (("N256", "C_P16_N256"), ("D128", "C_P16_D128")):
        seeds = [s for s in ONE2ONE[w] if s in ARMS[arm]["seeds"]]
        comp[w] = all(out["valid"][(arm, s)] for s in seeds)
        ks[w] = sum(1 for s in seeds if out["valid"][(arm, s)] and br(out["rows"][(arm, s)]))
    out["k"], out["complete"] = ks, comp
    out["width"] = {w: width_reading(ks[w], comp[w]) for w in ks}
    out["joint"] = joint_reading(ks["D128"], ks["N256"], comp["N256"] and comp["D128"])
    tr = []
    for arm in ("P16_N256", "P16_D128"):
        for k, r in s67["runs"].items():
            if k.startswith(arm + "|") and k.split("|")[2] == S67_SHA and br(r):
                tr.append(r["transition"])
    for arm in ("C_P16_N256", "C_P16_D128"):
        for s in ARMS[arm]["seeds"]:
            r = out["rows"][(arm, s)]
            if out["valid"][(arm, s)] and br(r):
                tr.append(r["transition"])
    out["transitions"] = sorted(tr)
    out["budget"] = budget_of(tr)
    rising = []
    for w, arm in (("N256", "C_P16_N256"), ("D128", "C_P16_D128")):
        for s in ONE2ONE[w]:
            r = out["rows"].get((arm, s))
            if out["valid"].get((arm, s)) and still_rising(r):
                rising.append(f"{arm}|{s}")
    out["rising"] = rising
    # the reproduction of a resumed run
    a, b = out["rows"].get(("C_P8_N256", 272)), out["rows"].get(("RCHK_P8_N256", 272))
    out["repro44"] = repro44(a, b)
    return out


def repro44(a, b):
    if not a or not b or not a.get("ok") or not b.get("ok"):
        return dict(ok=None, why="not run yet")
    why = []
    if a.get("halted_once_at") != HALT_AT:
        why.append(f"C_P8_N256 272 was not resumed at {HALT_AT} (marker {a.get('halted_once_at')})")
    if HALT_AT not in (a.get("resumed") or []):
        why.append(f"C_P8_N256 272's resumes {a.get('resumed')} do not include {HALT_AT}")
    if any(x >= HALT_AT for x in (b.get("resumed") or [])):
        why.append(f"RCHK resumed at {b.get('resumed')} (must be < {HALT_AT})")
    ca, cb = [x for x in a["curve"] if x[0] == REPRO_AT], [x for x in b["curve"] if x[0] == REPRO_AT]
    if not ca or ca != cb:
        why.append(f"evaluation at {REPRO_AT}: {ca} vs {cb}")
    ia = [i for i, x in enumerate(a["curve"]) if x[0] == REPRO_AT]
    ib = [i for i, x in enumerate(b["curve"]) if x[0] == REPRO_AT]
    if ia and ib and a["stats"][ia[0] + 1] != b["stats"][ib[0] + 1]:
        why.append(f"statistics at {REPRO_AT} differ")
    if mark_rows(a).get(REPRO_AT) != mark_rows(b).get(REPRO_AT):
        why.append(f"per-stream accuracy at {REPRO_AT}: {mark_rows(a).get(REPRO_AT)} vs {mark_rows(b).get(REPRO_AT)}")
    sa, sb = dict((s, h) for s, h in a.get("shas") or []), dict((s, h) for s, h in b.get("shas") or [])
    if sa.get(REPRO_AT) is None or sa.get(REPRO_AT) != sb.get(REPRO_AT):
        why.append(f"weights' SHA-1 at {REPRO_AT}: {sa.get(REPRO_AT)} vs {sb.get(REPRO_AT)}")
    return dict(ok=not why, why="; ".join(why), acc=ca, sha=sa.get(REPRO_AT))


# ── Report ───────────────────────────────────────────────────────────────────
def _fmt_marks(r):
    m = mark_rows(r)
    return "  ".join(f"{s}:{'/'.join(f'{v:.2f}' for v in m[s])}" for s in sorted(m))


def report():
    import explore_common2 as c2
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    st = ec.load_store(NAME)
    ev = evaluate_all(st)
    s67 = s67_store()
    print(f"  REPRODUCTION OF A RESUMED RUN (C_P8_N256 272 resumed from its {HALT_AT} checkpoint in a new process vs RCHK_P8_N256 272 "
          f"computed through, at {REPRO_AT}): "
          + ("not run yet" if ev["repro44"]["ok"] is None else ("BIT-IDENTICAL" if ev["repro44"]["ok"] else f"DIFFERS: {ev['repro44']['why']}"))
          + (f" (evaluation {ev['repro44'].get('acc')}, weights' SHA-1 {str(ev['repro44'].get('sha'))[:12]})" if ev["repro44"]["ok"] else ""))
    for arm, a in ARMS.items():
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}")
        print(f"    {'seed':>4} | {'S67 at 43200':<26} | {'reproduced':<10} | {'S68 outcome':<26} {'trans':>6} {'stop':>6} {'acc':>5} "
              f"{'end map':>9} | per-stream accuracy (step:s0/s1/s2/s3)")
        for s in a["seeds"]:
            r = ev["rows"][(arm, s)]
            rec = s67_record(a["s67"], s, s67)
            s67o = f"{rec.get('outcome')} {rec.get('acc', float('nan')):.2f}" if rec else "no record"
            if r is None:
                print(f"    {s:>4} | {s67o:<26} | not run")
                continue
            rp = "yes" if ev["valid"][(arm, s)] else f"NO: {ev['why'][(arm, s)]}"
            print(f"    {s:>4} | {s67o:<26} | {rp:<10} | {str(r.get('outcome')):<26} {str(r.get('transition')):>6} "
                  f"{str(r.get('stopped_at')):>6} {r.get('acc', float('nan')):5.2f} {c19.per_channel((r.get('end') or {}).get('ch_map')):>9} | "
                  f"{_fmt_marks(r)}" + (f"  [resumed at {r['resumed']}]" if r.get("resumed") else "")
                  + (f"  [halted once at {r['halted_once_at']}]" if r.get("halted_once_at") else ""))
        print()
    for w, arm in (("N256", "C_P16_N256"), ("D128", "C_P16_D128")):
        bound_s = [s for s in ONE2ONE[w] if ev["valid"].get((arm, s)) and br(ev["rows"].get((arm, s)))]
        print(f"  READING S68 {w} (its 6 one-to-one runs, seeds {ec.fmt_seeds(ONE2ONE[w])}): {ev['width'][w]} (bind by {ITERS}: "
              f"{ev['k'][w]}/{len([x for x in ONE2ONE[w] if x in ARMS['C_P16_' + w]['seeds']])} {bound_s}; slow, not stuck: >= {SLOW_K}; stuck: {STUCK_K}; unresolved otherwise)")
    print(f"  READING S68 joint: {ev['joint']} (D128 {ev['k']['D128']}/6, N256 {ev['k']['N256']}/6; the budget is the lead: both >= "
          f"{SLOW_K}; the width is the lead: D128 >= {SLOW_K} and N256 <= {WIDTH_N256_MAX}; neither otherwise)")
    b = ev["budget"]
    w6 = {w: len([x for x in ONE2ONE[w] if x in ARMS["C_P16_" + w]["seeds"]]) for w in ONE2ONE}
    print(f"  BUDGET for the main-line test: {b} (the latest transition {max(ev['transitions']) if ev['transitions'] else None} of the "
          f"BOUND ROUTED P = 16 learned runs in S67 and S68 {ev['transitions']}, + {STEP_B}, rounded up to a multiple of {STEP_B})"
          + (f"; 86,400 IS NOT ENOUGH: still rising at {ITERS} (>= {RISE} since {RISE_FROM}): {ev['rising']}" if ev["rising"] else
             f"; no one-to-one run still rising at {ITERS}"))
    print("  DESCRIPTIVE (no reading): the MERGED P16_N256 runs "
          + ", ".join(f"{s} {(ev['rows'].get(('C_P16_N256', s)) or {}).get('outcome', 'not run')}" for s in MERGED_N256)
          + "; P8_N256 272 " + str((ev["rows"].get(("C_P8_N256", 272)) or {}).get("outcome"))
          + "; the D128 oracle on 270 " + str((ev["rows"].get(("C_ORC_P16_D128", 270)) or {}).get("outcome")))
    print()
    return dict(width=ev["width"], joint=ev["joint"], k=ev["k"], budget=b, rising=ev["rising"], repro44=ev["repro44"])
