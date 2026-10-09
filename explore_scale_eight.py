#!/usr/bin/env python
"""
explore_scale_eight.py — EXPLORATORY, not a result. Screen S66 (batch 19): does the window recipe carry to eight streams
at the best width and the best depth of S64 / S65?

CONFIGURATION (explore_scale_child, "b"): test_stream_curriculum.run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams
from step 1), 43200 updates, LOCAL3 + SLOW + S37's KEYMASS split as run in batches 17-18 (W_SPLIT: checks every 2400 from
4800 through 40800 on S36's 64-sequence probe, 2400 the reference; fire if the probe accuracy < 0.95 and it rose < 0.02;
<= 3 splits, >= 4800 apart; S24's split of W_g's row c* onto c0, W_g's Adam state zeroed). Seeds 290-299.
ARMS (fixed by the driver from S64 / S65 when both are complete, by the rule in explore_scale_width.best_candidate, and
stored before any S66 run):
  N256_L3        the reference (N=256, D=32, 3 layers)
  <W>_L3         S64's best width (the tested, complete non-reference size with the most BOUND ROUTED; ties -> the earlier
                 median transition; ties -> the larger model), 3 layers
  N256_L<n>      S65's best depth by the same rule (L2 / L4 / L6), N=256, D=32
  No arm if no candidate was tested.
READING (the user's, fixed before any run; per arm, d = arm's BOUND ROUTED - the reference's over the paired seeds):
  "carries" if d >= -1; otherwise "does not carry". Exact McNemar two-sided printed.
ALSO REPORTED: splits fired per run (more or less often than the reference); each run's transition against the 2400
slow-memory switch and the 4800 first check: if an arm's median transition is at or before 4800, the trigger's schedule is
size-dependent and a main-line test needs a relative schedule (printed as such).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19
import explore_scale_width as s64

NAME = "scale_eight"
CHILD = "explore_scale_child"
MAIN = True
IDEA = "the window recipe with the split at eight streams, at the best width and the best depth of S64 / S65"
SOURCE = "batch 17's S48 W_SPLIT (9/10 at N=256, 3 layers); S64 / S65; the user's batch 19"
CHANGE = "the model's size (S64's best width; S65's best depth); recipe and trigger unchanged; 43200 updates"
PAIRING = "seeds 290-299 with N256_L3 (the same batches)"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(290, 300))
REF = "N256_L3"
CARRY_D = -1
SWITCH, FIRST_CHECK = 2400, 4800
_SL = s64._SL + ("; + S37's KEYMASS trigger (every 2400 from 4800 through 40800: probe acc < 0.95 and rise < 0.02; <= 3 "
                 "splits, >= 4800 apart) -> S24's SPLIT of W_g's row c* onto c0, W_g's Adam state zeroed")
ARMS = {}
CUT = set()
DECISION = {}


def arm_of(key, mult, d, L, role, label):
    return dict(key=key, spec=dict(cfg="b", task="P4S8", gate="local", size=[mult, d, L], trigger=True), seeds=SEEDS, iters=ITERS,
                lr=LR, opt="adam", role=role, sched=_SL, label=label)


def configure(dec):
    """dec: the stored phase-1 decision {best_width: size name or None, best_depth: n_layer or None}."""
    DECISION.clear()
    DECISION.update(dec or {})
    ARMS.clear()
    if dec is None:
        return
    ARMS[REF] = arm_of(REF, 8, 32, 3, "ref", "N=256, D=32, 3 layers (the reference)")
    w = dec.get("best_width")
    if w:
        mult, d = s64.SIZES[w]
        ARMS[f"{w}_L3"] = arm_of(f"{w}_L3", mult, d, 3, "arm", f"S64's best width {w}: N={mult * d}, D={d}, 3 layers")
    L = dec.get("best_depth")
    if L:
        ARMS[f"N256_L{L}"] = arm_of(f"N256_L{L}", 8, 32, L, "arm", f"S65's best depth: N=256, D=32, {L} layers")


def splits_block(runs):
    for s, r in sorted(runs.items()):
        cks = r.get("checks") or []
        f = [c["step"] for c in cks if c.get("fired")]
        lab = [c.get("key_ok") for c in cks if c.get("fired")]
        print(f"      {s}: {r.get('splits')} split(s) at {f} (labelled ok {lab}); checks "
              + " ".join(f"{c['step']}:{c['acc']:.2f}" + ("F" if c["fired"] else "B" if c["blocked"] else "") for c in cks))


def report():
    import explore_common2 as c2
    me = sys.modules[__name__]
    if not ARMS:
        print("-" * 100)
        print(f"[{ec.BANNER}] SCREEN {NAME}: not configured yet (S64 and S65 must complete first)")
        return {}
    c2.print_screen_header2(me)
    print(f"  phase-1 decision: {DECISION}")
    st = ec.load_store(NAME)
    ref = c16.runs_of(me, REF, store=st)
    out = {}
    for arm, a in ARMS.items():
        runs = c16.runs_of(me, arm, store=st)
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}")
        r0 = next(iter(runs.values()), None)
        print(f"    initial scales: {c19.init_line((r0 or {}).get('init'))}")
        c19.arm_table(arm, runs, a["seeds"], None if arm == REF else ref, None if arm == REF else REF)
        tm, tl = c19.trans_med(runs, only_br=False)
        nbr = sum(c19.br(r) for r in runs.values())
        sp = [r.get("splits") for r in runs.values()]
        early = [t for t in tl if t <= FIRST_CHECK]
        print(f"    BOUND ROUTED {nbr}/{len(runs)}; bound {sum(c19.bound(r) for r in runs.values())}/{len(runs)}; classes {c19.classes(runs)}")
        print(f"    splits fired: {sum(x or 0 for x in sp)} in {len(runs)} runs (per run {sp}); transitions (bound runs) {tl}, median {tm}; "
              f"at or before the {SWITCH} switch: {sum(1 for t in tl if t <= SWITCH)}; at or before the {FIRST_CHECK} first check: "
              f"{len(early)}")
        if tm is not None and tm <= FIRST_CHECK:
            print(f"    -> the median transition is at or before {FIRST_CHECK}: at this size the trigger's schedule is size-dependent, "
                  f"and a main-line test needs a relative schedule")
        print("    the trigger's checks per run (step:probe acc; F fired, B blocked):")
        splits_block(runs)
        o = dict(br=nbr, n=len(runs), complete=c19.complete(me, arm, st), splits=sum(x or 0 for x in sp), trans_med=tm, trans=tl,
                 classes=c19.classes(runs))
        if arm != REF:
            d = c19.counts_vs(runs, ref, a["seeds"])
            o.update(d=d["d"], p=d["p"], b=d["b"], c=d["c"])
            refc = c19.complete(me, REF, st)
            if not (o["complete"] and refc):
                o["reading"] = f"INCOMPLETE ({len(runs)}/{len(a['seeds'])}, reference {len(ref)}/{len(SEEDS)}), no reading"
            else:
                v = "carries" if d["d"] >= CARRY_D else "does not carry"
                o["reading"] = (f"{v} (BOUND ROUTED {d['new']}/{d['n']} vs {REF} {d['old']}/{d['n']}; d = {d['d']:+d}; carries: d >= "
                                f"{CARRY_D}; {arm} only {d['b']}, {REF} only {d['c']}, p = {d['p']:.3g}; splits {o['splits']} vs "
                                f"{sum(r.get('splits') or 0 for r in ref.values())})")
            print(f"  READING S66 {arm}: {o['reading']}")
        out[arm] = o
        print()
    return out
