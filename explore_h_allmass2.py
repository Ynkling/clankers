#!/usr/bin/env python
"""
explore_h_allmass2.py — EXPLORATORY, not a result. Session H, round 5 (1), screen S87a: S86's ALLMASS and BASE on twenty
more paired seeds, pooled with S86 descriptively. Branch claude/explore-H; seeds 2600-2619 (session H's round-5 block
2600-2659; scanned before any run: it appears nowhere in explore_out/, results/ or specs/ on any branch); outputs under
explore_out/H/. Child: explore_h_allmass_child, unchanged (S86's code; the code SHA is the same).

CONFIGURATION: S86's (S=8, P=4, k=16, conv, all 8 streams from step 1; LOCAL3 + SLOW, Adam, Hebbian memory, no hinge; 43200
updates; the exact-copy split; checks every 2400 from 4800, 2400 reference, fire if accuracy < 0.95 and rose < 0.02, cap 3,
gap 4800, last check 40800; 64-sequence probe as the rule's source).
ARMS: BASE (KEYMASS target), ALLMASS (all-position target), ORACLE (perfect gate ceil8_D8 + conv, seeds 2600-2601).
OUTCOME: BOUND ROUTED with the transition; every firing with its key-label and its all-label. v1 and v2 classes.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READINGS (fixed now; the 20 new pairs; S86's rules):
  "keeps it"        ALLMASS >= BASE - 1 pair ((BASE only - ALLMASS only) <= 1).
  "costs binding"   BASE only >= 4 and ALLMASS only = 0.
  "aims"            >= 90% of ALLMASS's firings labelled ok (its own pair, all_ok).
  Pooled with S86 (30 pairs): the counts and the discordant pairs printed, with the exact one-sided McNemar both ways; no
  reading.
CHECKS (before any run): explore_common.repro_check; the child's unit checks (S86's); ALLMASS at threshold 0 equals BASE at
threshold 0 bit for bit through 6000 on seed 2600 (curve, statistics, the trigger's rows), and equals batch 16's recorded
LOCAL3_SLOW_D8_A|260 through 6000.
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
import explore_h_allmass as s86

NAME = "H/allmass2"
CHILD = "explore_h_allmass_child"
MAIN = True
IDEA = "S86's all-position trigger vs KEYMASS on twenty more paired seeds"
SOURCE = "S86 (ALLMASS 8/10 vs BASE 9/10)"
CHANGE = "as S86"
PAIRING = "seeds 2600-2619, paired; pooled with S86 (2330-2339) descriptively"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2600, 2620))
ORACLE_SEEDS = (2600, 2601)
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="LOCAL3 + SLOW + exact-copy trigger " + a)
        for a in ("BASE", "ALLMASS")}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8", sched="one Adam 1e-3")
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", 260)
    jobs = [("A2600", dict(arm="ALLMASS", seed=2600, iters=6000, thr=0.0, total=ITERS)),
            ("B2600", dict(arm="BASE", seed=2600, iters=6000, thr=0.0, total=ITERS)),
            ("A260", dict(arm="ALLMASS", seed=260, iters=6000, thr=0.0, total=ITERS))]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    a, b = res["A2600"], res["B2600"]
    rk = lambda r: [{k: x.get(k) for k in s86.SEL + ("step", "acc", "fired")} for x in r["checks"]]   # noqa: E731
    same = a["curve"] == b["curve"] and a["stats"] == b["stats"] and rk(a) == rk(b) and a["splits"] == b["splits"] == 0
    rows.append((f"ALLMASS|2600 at threshold 0 equals BASE|2600 at threshold 0 bit for bit through 6000 (curve "
                 f"{[x[:2] for x in a['curve']]}; statistics; trigger rows at {[x['step'] for x in a['checks']]})", same))
    eq, c = s58.eq_upto(res["A260"], ref, 6000)
    rows.append((f"ALLMASS|260 at threshold 0 equals batch 16's LOCAL3_SLOW_D8_A|260 bit for bit through 6000", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def summarize(rs):
    n, k = len(rs), sum(hc.br(r) for r in rs.values())
    tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
    f = [c for r in rs.values() for c in r.get("checks") or [] if c.get("fired")]
    return dict(n=n, br=k, band=hc.band(k, n), wilson=hc.wilson(k, n), v1=hc.classes(rs),
                v2=dict(__import__("collections").Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
                trans_med=c16.med(tr), trans=tr, ops=len(f), key_ok=sum(bool(c["key_ok"]) for c in f),
                all_ok=sum(bool(c["all_ok"]) for c in f), op_ok=sum(bool(c.get("op_ok")) for c in f),
                same=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
                one_to_one_before=sum(len(set(c["map_before"])) == 8 for c in f),
                v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")),
                bad_fires=[(s, c["step"], c["key_cs"], c["key_c0"], c["all_cs"], c["all_c0"], c["key_ok"], c["all_ok"])
                           for s, r in rs.items() for c in r.get("checks") or [] if c.get("fired") and not c.get("op_ok")])


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    for a in ("BASE", "ALLMASS"):
        res["arms"][a] = dict(summarize(got[a]), complete=len(got[a]) == len(SEEDS),
                              seeds_ok=sorted(s for s, r in got[a].items() if hc.br(r)))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in ("BASE", "ALLMASS")}
    res["cmp"]["ALLMASS"] = c = hc.paired(B["ALLMASS"], B["BASE"], SEEDS)
    g86 = ec.load_store(s86.NAME)
    c16.STATE["shas"].setdefault(s86.NAME, g86["meta"]["code_shas"][-1])
    p86 = {a: {s: hc.br(r) for s, r in c16.runs_of(s86, a, store=g86).items()} for a in ("BASE", "ALLMASS")}
    res["pooled"] = hc.paired({**p86["ALLMASS"], **B["ALLMASS"]}, {**p86["BASE"], **B["BASE"]}, s86.SEEDS + SEEDS)
    complete = all(res["arms"][a]["complete"] for a in ("BASE", "ALLMASS")) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {"S87a": "INCOMPLETE, no reading"}
    elif not valid:
        rd = {"S87a": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        al = res["arms"]["ALLMASS"]
        frac = al["all_ok"] / al["ops"] if al["ops"] else None
        rd = {"keeps": ("keeps it" if c["c"] - c["b"] <= 1 else "does not apply") + f" (BASE only {c['c']}, ALLMASS only {c['b']})",
              "costs": ("costs binding" if (c["c"] >= 4 and c["b"] == 0) else "no cost by the rule")
              + f" (BASE only {c['c']}, ALLMASS only {c['b']})",
              "aims": ("aims" if (frac is not None and frac >= 0.9) else "does not apply")
              + f" (ALLMASS firings labelled ok {al['all_ok']}/{al['ops']})"}
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
