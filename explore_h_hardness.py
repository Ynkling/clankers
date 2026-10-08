#!/usr/bin/env python
"""
explore_h_hardness.py — EXPLORATORY, not a result. Session H, screen S63: gate hardness at eight streams — an annealed gate
temperature (1 -> 0.25) and a hard (one-hot) write gate with a soft read gate — on top of the KEYMASS split recipe.
Branch claude/explore-H; seeds 490-499; outputs under explore_out/H/. Child: explore_h_hardness_child (main line at 9c5939e).

BACKGROUND: docs/reading/distant_cues_2026-10.md 4.7 (Active Dendrites: hardness, kWTA, is the component whose removal causes
the "sharp drop"; anneal softmax -> top-m late in training). S60: a learned temperature ran straight to its 0.5 floor (the
gate wants to be sharper). S58b: LOCAL3 + SLOW + KEYMASS bound routed 10/10 at eight streams (transitions 14400-27600).

CONFIGURATION: S=8, P=4, k=16, conv, all 8 streams from step 1 (run_sc with ARM["D8"]), LOCAL3 gate, SLOW (Adam), no hinge,
43200 updates, evaluation every 1200, S48's KEYMASS split (<= 3, >= 4800 apart, checks every 2400 from 4800 through 40800)
on every learned arm; seeds 490-499, paired. 1 torch thread per run.
ARMS:
  REF     LOCAL3 + SLOW + KEYMASS (the reference).
  ANNEAL  REF + gate temperature tau linear from 1 (update 2400) to 0.25 (update 24000), 0.25 after; read and write gates
          both softmax(z / tau); evaluations at the current tau.
  HARD_W  REF + from update 2400 on the write gate one-hot(argmax z) (no gradient), the read gate softmax(z) (the gradient
          path into the gate); evaluations with the same gates.
  ORACLE  perfect gate ceil8_D8 + conv, seeds 490-491 (validity).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome), failure classes, transition times, splits fired.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READING (fixed now; per arm X in {ANNEAL, HARD_W} vs REF on the 10 paired seeds):
  "hardness helps" (for X) if  (a) X only >= 4 and REF only = 0 on BOUND ROUTED,  or  (b) X's transition is strictly earlier
  than REF's on >= 8 of the 10 seeds (a seed counts only if both are BOUND ROUTED there) and X's BOUND ROUTED count >= REF's.
  Otherwise "does not help". Exact one-sided McNemar both ways, Wilson intervals and bands printed.
CHECKS (before any run): explore_common.repro_check; the child's unit checks (the tau schedule's values; HARD_W's gates
before / from 2400 — one-hot write without gradient, soft read with gradient, the global RNG untouched; ANNEAL's gates at
n = 13200 and 2400); with the trigger's threshold at 0 and the new code path disabled, a run equal BIT FOR BIT to batch 16's
recorded LOCAL3_SLOW_D8_A|260 through 6000: REF; ANNEAL with the end temperature 1 (tau = 1 throughout); HARD_W with the
hard switch never reached.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc
import explore_h_collapse as s58

NAME = "H/hardness"
CHILD = "explore_h_hardness_child"
MAIN = True
IDEA = "gate hardness at eight streams: annealed temperature, hard write gate"
SOURCE = "distant_cues 4.7 (Active Dendrites' kWTA); S60's temperature ran to its floor"
CHANGE = "LOCAL3 + SLOW + KEYMASS + annealed tau 1 -> 0.25 / one-hot write gate from 2400"
PAIRING = "seeds 490-499, all learned arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(490, 500))
ORACLE_SEEDS = (490, 491)
REC_SEED = 260
_sl = "LOCAL3 + SLOW + KEYMASS (<= 3); no hinge"
ARMS = {"REF": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="LOCAL3 + SLOW + KEYMASS", sched=_sl),
        "ANNEAL": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ tau 1 -> 0.25 over 2400-24000",
                       sched=_sl + "; annealed tau"),
        "HARD_W": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ one-hot write gate from 2400",
                       sched=_sl + "; hard write"),
        "ORACLE": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8 + conv",
                       sched="one Adam 1e-3")}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", REC_SEED)
    base = dict(seed=REC_SEED, iters=6000, thr=0.0, total=ITERS)
    jobs = [("REF", dict(base, arm="REF"), "as is"), ("ANNEAL", dict(base, arm="ANNEAL", tau_end=1.0), "end temperature 1"),
            ("HARD_W", dict(base, arm="HARD_W", hard_from=10 ** 9), "hard switch never reached")]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    for (arm, p, how), r in res:
        eq, c = s58.eq_upto(r, ref, 6000)
        eq = eq and r["splits"] == 0 and len(r["checks"]) >= 2 and r.get("n_upd") == 6000
        rows.append((f"{arm} ({how}, threshold 0) equals batch 16's LOCAL3_SLOW_D8_A|{REC_SEED} ({ref and ref['cpu']}) bit for bit "
                     f"through 6000 (curve {[x[:2] for x in c] if c else c}; statistics); updates counted {r.get('n_upd')}", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={}, earlier={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    learned = ("REF", "ANNEAL", "HARD_W")
    for arm in learned:
        rs = got[arm]
        n, k = len(rs), sum(hc.br(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        res["arms"][arm] = dict(
            n=n, br=k, bound=sum(1 for r in rs.values() if r.get("transition") is not None), band=hc.band(k, n),
            wilson=hc.wilson(k, n), classes=hc.classes(rs), merged=sum(1 for r in rs.values() if hc.fclass(r) == "MERGED"),
            trans_med=c16.med(tr), trans=tr, ops=sum(r.get("splits") or 0 for r in rs.values()), complete=n == len(SEEDS),
            bound_seeds=sorted(s for s, r in rs.items() if hc.br(r)),
            maps_end=sorted(hc.per_channel(r["end"].get("ch_map")) for r in rs.values() if not hc.br(r)))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for a in ("ANNEAL", "HARD_W"):
        res["cmp"][a] = hc.paired(B[a], B["REF"], SEEDS)
        res["earlier"][a] = sum(1 for s in SEEDS if s in got[a] and s in got["REF"] and hc.br(got[a][s]) and hc.br(got["REF"][s])
                                and got[a][s]["transition"] < got["REF"][s]["transition"])
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {a: "INCOMPLETE, no reading" for a in ("ANNEAL", "HARD_W")}
    elif not valid:
        rd = {a: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for a in ("ANNEAL", "HARD_W")}
    else:
        rd = {}
        for a in ("ANNEAL", "HARD_W"):
            c, e = res["cmp"][a], res["earlier"][a]
            ra = c["b"] >= 4 and c["c"] == 0
            rb = e >= 8 and res["arms"][a]["br"] >= res["arms"]["REF"]["br"]
            rd[a] = (("hardness helps" if (ra or rb) else "does not help")
                     + f" ({a} vs REF {c['b']} vs {c['c']} [>= 4 vs 0: {ra}]; earlier on {e}/10 with BR {res['arms'][a]['br']} vs "
                       f"{res['arms']['REF']['br']} [>= 8 and no loss: {rb}])")
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
