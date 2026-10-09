#!/usr/bin/env python
"""
explore_scale_width.py — EXPLORATORY, not a result. Screen S64 (batch 19): does the window recipe survive width?

BACKGROUND: batch 16's S44(a): LOCAL3_SLOW16_A (LOCAL3 + SLOW, Adam) bound and routed 20/20 at S=4, P=4, k=16, conv (seeds
240-259; transitions 3600 on 15, at most 10800). Every screen since ran the model at N=256 (BDH sparse width, mult 8 x
D 32), embedding width D=32, 3 layers. Does the recipe, unchanged, carry to a wider or a fatter model?

CONFIGURATION (explore_scale_child, "k16"): test_stream_recipe.run_attempt with ARM["A4k16"] (S=4, P=4, k=16, conv 'layer'
width 4, 28800 updates, evaluation every 1200, early stop after 4800), LOCAL3 + SLOW exactly as batch 16 (gate W_in, W_g,
window at 1e-3 throughout; every other trainable parameter 1e-4 for updates 1-2400, then 1e-3; window width 3; W_h
frozen, unused; no hinge; no split: four streams on 16 channels). Only the size changes (explore_b19_model.builder_of);
initial scales are the code's defaults at the new size (recorded per arm in the report: bdh's std 0.02 for the stack
and the embedding, 0.1 for the gate's matrices, the window +-1/sqrt(3), whatever the size).
ARMS (seeds 270-279; the oracle on 270-271):
  N256   mult 8,  D 32  (the current model: the reference, rerun here)
  N512   mult 16, D 32
  N1024  mult 32, D 32
  D64    mult 4,  D 64  (N = 256)
  D128   mult 2,  D 128 (N = 256)
  ORC_<size>  the oracle at that size: the perfect gate (stream s -> channel s, channels 4-15 unused; test_stream_recipe's
              ceiling4k16), on the recorded validity recipe (a single Adam at 1e-3), seeds 270-271, 28800 updates.
ORACLE RULE (fixed before any run; the user's S67 rule applied here too): a size's learned arm runs once its oracle has
BOUND on a seed; if neither oracle seed binds within 28800 the size is UNTESTED (not run, no part in the reading).
PAIRING: by seed (the same batches; the initial parameters differ with the size).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome: transition not None, one-to-one stream -> channel map and every stream's
held-out accuracy >= 0.9 at the end); transition time; failure classes.
READING (the user's, fixed before any run; on counts, with d = arm's BOUND ROUTED - N256's over the paired seeds = arm-only
minus N256-only discordant pairs; exact McNemar two-sided printed):
  "scale-free"       if every tested size has d >= -1;
  "width-sensitive"  if any tested size has d <= -4;
  otherwise neither. Transition medians (over BOUND ROUTED runs) per size are printed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19

NAME = "scale_width"
CHILD = "explore_scale_child"
MAIN = True
IDEA = "does the window recipe survive width? N = 256, 512, 1024 at D = 32; D = 64, 128 at N = 256"
SOURCE = "batch 16's S44(a) LOCAL3_SLOW16_A (20/20 at S=4, k=16, N=256, D=32); the user's batch 19"
CHANGE = "only the model's size (mult = N/D, the embedding width D); every recipe and initial scale rule as is"
PAIRING = "seeds 270-279 with N256 (the same batches); oracles on 270-271"
LR = 1e-3
ITERS = 28800
SEEDS = tuple(range(270, 280))
ORACLE_SEEDS = (270, 271)
SIZES = {"N256": (8, 32), "N512": (16, 32), "N1024": (32, 32), "D64": (4, 64), "D128": (2, 128)}
REF = "N256"
FREE_D, SENS_D = -1, -4
_SL = ("gate W_in/W_g/window 0.001 throughout; every other trainable parameter 0.0001 for updates 1-2400, 0.001 after; "
       "W_h frozen (unused); no hinge")
_OR = "the perfect gate; a single Adam at 0.001 on every parameter (the recorded validity recipe)"
CUT = set()


def size_desc(name):
    mult, d = SIZES[name]
    return f"N={mult * d} (mult {mult}), D={d}, 3 layers"


ARMS = {}
for _n, (_m, _d) in SIZES.items():
    ARMS[f"ORC_{_n}"] = dict(key=f"ORC_{_n}", spec=dict(cfg="k16", task="P4S4", gate="perfect", size=[_m, _d, 3]), seeds=ORACLE_SEEDS,
                             iters=ITERS, lr=LR, opt="adam", role="oracle", size=_n, sched=_OR,
                             label=f"oracle at {size_desc(_n)} (S=4, P=4, k=16, conv)")
    ARMS[_n] = dict(key=_n, spec=dict(cfg="k16", task="P4S4", gate="local", size=[_m, _d, 3]), seeds=SEEDS, iters=ITERS, lr=LR,
                    opt="adam", role="ref" if _n == REF else "arm", oracle=f"ORC_{_n}", size=_n, sched=_SL,
                    label=f"LOCAL3 + SLOW at {size_desc(_n)}" + (" (the reference)" if _n == REF else ""))


# ── Reading ──────────────────────────────────────────────────────────────────
def reading_of(rows, ref_complete, ref_label, free_word="scale-free", sens_word="width-sensitive", ref_tested=True):
    """rows: {arm: dict(tested, complete, d)} for the non-reference arms."""
    if not ref_tested:
        return f"UNTESTED (the reference's oracle did not bind)"
    tested = {k: r for k, r in rows.items() if r["tested"]}
    if not ref_complete or not all(r["complete"] for r in tested.values()):
        return "INCOMPLETE, no reading"
    if not tested:
        return "UNTESTED (no other size's oracle bound)"
    ds = {k: r["d"] for k, r in tested.items()}
    untested = sorted(k for k, r in rows.items() if not r["tested"])
    if all(v >= FREE_D for v in ds.values()):
        v = free_word
    elif any(v <= SENS_D for v in ds.values()):
        v = sens_word
    else:
        v = "neither"
    return (f"{v} (d = arm - {ref_label} BOUND ROUTED: " + ", ".join(f"{k} {d:+d}" for k, d in ds.items())
            + f"; {free_word}: every d >= {FREE_D}; {sens_word}: any d <= {SENS_D}"
            + (f"; UNTESTED: {untested}" if untested else "") + ")")


def summarize(me, store=None):
    st = ec.load_store(me.NAME) if store is None else store
    out = {}
    for arm, a in me.ARMS.items():
        runs = c16.runs_of(me, arm, store=st)
        o = a.get("oracle")
        ost = c19.oracle_state(me, o, st) if o else None
        tm, tl = c19.trans_med(runs)
        out[arm] = dict(runs=runs, n=len(runs), complete=c19.complete(me, arm, st), br=sum(c19.br(r) for r in runs.values()),
                        bound=sum(c19.bound(r) for r in runs.values()), oracle_state=ost, tested=(ost != "invalid") if o else True,
                        trans_med=tm, trans=tl, stop_med=c16.med([r.get("stopped_at") for r in runs.values()]),
                        classes=c19.classes(runs), params=c19.params_of(runs))
    return out


def best_candidate(sm, ref):
    """The phase-2 rule (fixed before any run): among the tested, complete non-reference learned arms, the most BOUND
    ROUTED; ties -> the earlier median transition (over BOUND ROUTED runs; none = last); ties -> the larger model (more
    parameters)."""
    cands = [k for k, v in sm.items() if v.get("role") in ("arm",) and v["tested"] and v["complete"] and v["n"] > 0]
    if not cands:
        return None
    return sorted(cands, key=lambda k: (-sm[k]["br"], sm[k]["trans_med"] if sm[k]["trans_med"] is not None else float("inf"),
                                        -(sm[k]["params"] or 0)))[0]


def report_block(me, ref, ref_label, free_word, sens_word, desc):
    import explore_common2 as c2
    c2.print_screen_header2(me)
    st = ec.load_store(me.NAME)
    sm = summarize(me, st)
    for k in sm:
        sm[k]["role"] = me.ARMS[k].get("role")
    refr = sm[ref]["runs"]
    print(f"  ORACLES (the perfect gate; BOUND = held-out accuracy >= 0.95 held to the end):")
    for arm, a in me.ARMS.items():
        if a.get("role") != "oracle":
            continue
        s = sm[arm]
        print(f"    {arm:<12} {desc(a):<34} bound {s['bound']}/{s['n']} of {len(a['seeds'])}  transitions "
              f"{[r.get('transition') for _, r in sorted(s['runs'].items())]}  stop {[r.get('stopped_at') for _, r in sorted(s['runs'].items())]}"
              f"  -> {'VALID' if s['bound'] else ('UNTESTED' if s['complete'] else 'pending')}")
    rows = {}
    for arm, a in me.ARMS.items():
        if a.get("role") not in ("ref", "arm"):
            continue
        s = sm[arm]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}")
        r0 = next(iter(s["runs"].values()), None)
        print(f"    initial scales ({'seed ' + str(r0['seed']) if r0 else 'no run yet'}): {c19.init_line((r0 or {}).get('init'))}")
        if not s["tested"]:
            print(f"    UNTESTED: its oracle did not bind within budget (not run)")
            if arm != ref:
                rows[arm] = dict(tested=False, complete=True, d=None)
            continue
        c19.arm_table(arm, s["runs"], a["seeds"], None if arm == ref else refr, None if arm == ref else ref_label)
        print(f"    BOUND ROUTED {s['br']}/{s['n']}; bound {s['bound']}/{s['n']}; transition median (BOUND ROUTED) {s['trans_med']} "
              f"{s['trans']}; stop median {s['stop_med']}; classes {s['classes']}; trainable parameters {s['params']}")
        if arm != ref:
            d = c19.counts_vs(s["runs"], refr, a["seeds"])
            print(f"    vs {ref_label}: {arm} {d['new']}/{d['n']}, {ref_label} {d['old']}/{d['n']}; {arm} only {d['b']}, {ref_label} only "
                  f"{d['c']}; d = {d['d']:+d}; McNemar two-sided p = {d['p']:.3g}")
            rows[arm] = dict(tested=True, complete=s["complete"] and sm[ref]["complete"], d=d["d"], p=d["p"])
    rd = reading_of(rows, sm[ref]["complete"], ref_label, free_word, sens_word, ref_tested=sm[ref]["tested"])
    print(f"  transition medians per size (BOUND ROUTED runs): " + ", ".join(
        f"{k} {sm[k]['trans_med']}" for k, a in me.ARMS.items() if a.get("role") in ("ref", "arm") and sm[k]["tested"]))
    print(f"  READING {me.NAME}: {rd}")
    print()
    return dict(reading=rd, summary={k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in sm.items()},
                best=best_candidate(sm, ref))


def report():
    return report_block(sys.modules[__name__], REF, "N256", "scale-free", "width-sensitive",
                        lambda a: size_desc(a["size"]))
