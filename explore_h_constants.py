#!/usr/bin/env python
"""
explore_h_constants.py — EXPLORATORY, not a result. Session H, screen S79: which of the split's constants matter, with the
simplified split (an exact row copy) as the base. The report states the constants without having varied them; an
independent replication will have to guess the ones it does not state. Branch claude/explore-H; seeds 1320-1329 (the
specification's 1120-1129, shifted with S78's block: see explore_h_two); outputs under explore_out/H/. Child:
explore_h_constants_child (main line's modules at 9c5939e).

CONFIGURATION: test_window_gate's Part C / D path (run_sc with ARM["D8"]: S=8, P=4, k=16, conv, all 8 streams from step 1),
LOCAL3 gate, SLOW (Adam), no hinge, 43200 updates, evaluation every 1200; seeds 1320-1329, all arms paired (same initial
parameters, window and batches). 1 torch thread per run.
ARMS (the operation in every trigger arm: the exact copy W_g[c0] := W_g[c*] onto the KEYMASS target, no noise, no Adam
reset; the target c* / c0 = the largest / smallest mean read-gate mass at the probe's KEY positions; last check 40800):
  BASE     checks every 2400 from 4800 (2400 reference); fire if probe acc < 0.95 and it rose < 0.02 since the previous
           check; 64-sequence probe; cap 3; gap 4800.
  INT1200  checks every 1200 from 2400 (1200 reference), gap 2400.
  CAP1     cap 1.
  CAP6     cap 6, gap 2400 (so it can fire).
  PROBE16  a 16-sequence probe (the first 16 of the 64).
  THR05    rose < 0.05.
  ORACLE   perfect gate ceil8_D8 + conv, seeds 1320-1321 (validity).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition; every firing recorded (update, c* -> c0, labelled
ok = c* holds >= 2 streams and c0 none on the labelled map before it). Failure classes under v1 and v2 (outcome_v2 of
fail_class_v2 at ba8901a), both reported.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise every reading is UNTESTED.
READINGS (fixed now; per variant V vs BASE, BOUND ROUTED on the 10 paired seeds; "V only" = V bound routed and BASE not):
  "robust"       if (BASE only - V only) <= 1;
  "sensitive"    if BASE only >= 4 and V only = 0;
  "inconclusive" otherwise.
  CAP6 also: "the extra splits are harmless" if CAP6's BOUND ROUTED >= BASE's - 1 and every CAP6 firing beyond a run's
  third is labelled ok (c* >= 2 streams, c0 none); if CAP6 fires no fourth operation in any run, printed as "no extra
  splits fired" (the reading does not apply).
DESCRIPTIVE: transitions by arm; for INT1200, per seed whether its first split comes earlier than BASE's first split.
CHECKS (before any run): explore_common.repro_check; make_trigger_cfg at BASE's constants equals S58's make_trigger_op row
for row; every variant's firing pattern with every check eligible (BASE 4800/9600/14400, INT1200 2400/4800/7200, CAP1 4800,
CAP6 4800-16800 every 2400, PROBE16 and THR05 as BASE); PROBE16's probe = the first 16 of the 64; each arm (BASE and the
five variants) with the threshold at 0 (never fires) equals batch 16's recorded LOCAL3_SLOW_D8_A|260 BIT FOR BIT through
6000; explore_h_ops.copy_exact equals S61's exact copy (main-line child).
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

NAME = "H/constants"
CHILD = "explore_h_constants_child"
MAIN = True
IDEA = "the split's constants (check interval, cap, gap, probe size, rise threshold) around the exact-copy base"
SOURCE = "S61 (the split is an exact row copy); the replication's need for stated constants"
CHANGE = "LOCAL3 + SLOW + exact-copy trigger with one constant changed"
PAIRING = "seeds 1320-1329 (shifted from 1120-1129), all arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(1320, 1330))
ORACLE_SEEDS = (1320, 1321)
VARIANTS = ("INT1200", "CAP1", "CAP6", "PROBE16", "THR05")
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="LOCAL3 + SLOW + copy trigger " + a)
        for a in ("BASE",) + VARIANTS}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8", sched="one Adam 1e-3")
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", 260)
    jobs = [(a, dict(arm=a, seed=260, iters=6000, thr=0.0, total=ITERS)) for a in ("BASE",) + VARIANTS]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        fo = ex.submit(lambda: c16.run_child("explore_h_classes_child", "ops_check", {}, main=True))
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()] + [tuple(x) for x in fo.result()]
    for (a, p), r in res:
        eq, c = s58.eq_upto(r, ref, 6000)
        eq = eq and r["splits"] == 0 and len(r["checks"]) >= 2
        rows.append((f"{a} (threshold 0) equals batch 16's LOCAL3_SLOW_D8_A|260 ({ref and ref['cpu']}) bit for bit through 6000 "
                     f"(curve {[x[:2] for x in c] if c else c}; statistics; checks at {[x['step'] for x in r['checks']]})", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    learned = ("BASE",) + VARIANTS
    for a in learned:
        rs = got[a]
        n, k = len(rs), sum(hc.br(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        fires = {s: [c for c in r.get("checks") or [] if c.get("fired")] for s, r in rs.items()}
        nf = [len(f) for f in fires.values()]
        res["arms"][a] = dict(
            n=n, br=k, band=hc.band(k, n), wilson=hc.wilson(k, n), bound=sum(1 for r in rs.values() if r.get("transition") is not None),
            v1=hc.classes(rs), v2=dict(__import__("collections").Counter(
                r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
            trans_med=c16.med(tr), trans=tr, ops=sum(nf), ops_max=max(nf) if nf else 0,
            ops_ok=sum(1 for f in fires.values() for c in f if c.get("key_ok")),
            extra=[(s, c["step"], c["key_ok"]) for s, f in fires.items() for c in f[3:]],
            first={s: (f[0]["step"] if f else None) for s, f in fires.items()},
            complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
            v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for v in VARIANTS:
        res["cmp"][v] = hc.paired(B[v], B["BASE"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    rd = {}
    if not complete:
        rd = {v: "INCOMPLETE, no reading" for v in VARIANTS}
    elif not valid:
        rd = {v: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for v in VARIANTS}
    else:
        for v in VARIANTS:
            c = res["cmp"][v]
            lab = ("robust" if c["c"] - c["b"] <= 1 else "sensitive" if (c["c"] >= 4 and c["b"] == 0) else "inconclusive")
            rd[v] = f"{lab} (BASE only {c['c']}, {v} only {c['b']})"
        A = res["arms"]
        ex = A["CAP6"]["extra"]
        if not ex:
            rd["CAP6_extra"] = "no extra splits fired (the reading does not apply)"
        else:
            ok = A["CAP6"]["br"] >= A["BASE"]["br"] - 1 and all(k for _, _, k in ex)
            rd["CAP6_extra"] = (("the extra splits are harmless" if ok else "does not apply")
                                + f" (CAP6 {A['CAP6']['br']} vs BASE {A['BASE']['br']}; extra firings {len(ex)}, labelled ok "
                                  f"{sum(1 for _, _, k in ex if k)})")
    fb, fi = res["arms"]["BASE"]["first"], res["arms"]["INT1200"]["first"]
    res["int1200_earlier"] = {s: (fi.get(s), fb.get(s)) for s in SEEDS}
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
