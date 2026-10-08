#!/usr/bin/env python
"""
explore_h_eight.py — EXPLORATORY, not a result. Session H, screen S58b: the two best arms of S58 against SPLIT at eight
streams. Branch claude/explore-H; seeds 460-469 (session H's block); outputs under explore_out/H/. Child:
explore_h_eight_child (main line's modules at 9c5939e).

CONFIGURATION: S=8, P=4, k=16, conv, all 8 streams from step 1 (test_stream_curriculum.run_sc with ARM["D8"], batch 16's
S44(b) / batch 17's S48 path), LOCAL3 gate, SLOW (Adam), no hinge, 43200 updates, evaluation every 1200. Paired seeds.
Batch 17's W_SPLIT (the same split, 28800 updates) bound routed 9/10 on seeds 260-269; batch 16's no-split
LOCAL3_SLOW_D8_A 3/10 (other seeds and CPU: compared by counts only).

ARM SELECTION (fixed now, before S58's results exist; applied once S58 is complete and stored in this screen's store):
the two arms of S58 with the most BOUND ROUTED among {RESET, GUMBEL_W, GUMBEL_RW, SWITCH} (the devices; SPLIT is the
comparator, NONE the no-device baseline); ties broken by the earlier median transition time, then by fewer MERGED runs,
then by that listed order. If S58 is UNTESTED (its oracle failed), S58b does not run.
ARMS: SPLIT (S48's KEYMASS split, checks every 2400 from 4800 through 40800, <= 3 splits, >= 4800 apart), the two selected
arms (S58's definitions), ORACLE (perfect gate ceil8_D8 + conv, own recipe) on seeds 460-461.
OUTCOME, failure classes, transition, KEY-position routing: as S58 (test_stream_recipe.outcome; routing_k's VAL map; the
KEY-position map beside it).
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READING (fixed now): for each selected arm X, "X carries to eight streams" if (SPLIT only - X only) <= 1 on the 10 paired
seeds (BOUND ROUTED). The screen's reading: "carries to eight streams" if it holds for the better of the two arms at eight
streams (more BOUND ROUTED; if tied, if it holds for either), else "does not carry". Exact one-sided McNemar both ways,
Wilson intervals and bands printed.
CHECKS (before any run): explore_common.repro_check; with each new code path disabled, a run equal BIT FOR BIT to batch
16's recorded LOCAL3_SLOW_D8_A|260 (curve and every statistic): SPLIT (and RESET if selected) with the trigger's threshold at
0 through 6000; GUMBEL_RW at noise scale 0 and SWITCH at alpha 0 through 2400 (if selected); GUMBEL_W (if selected) at noise
scale 0 equal bit for bit to the TWO_NODE control through 2400 (S58's amended CHECK); the ORACLE's path runs (validity is
read from its runs).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc
import explore_h_collapse as s58

NAME = "H/eight"
CHILD = "explore_h_eight_child"
MAIN = True
IDEA = "S58's two best anti-collapse arms vs the KEYMASS split at eight streams"
SOURCE = "S58 (session H); batch 17's S48 W_SPLIT 9/10"
CHANGE = "LOCAL3_SLOW_D8_A + the split / S58's two best devices, 43200 updates"
PAIRING = "seeds 460-469, all learned arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(460, 470))
ORACLE_SEEDS = (460, 461)
CANDIDATES = ("RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH")
B16 = "window_scale"
B16_SHA = "01767493e840"
REC_SEED = 260
CHECK_SHORT, CHECK_TRIG = 2400, 6000
ARMS = {}
SELECTED = []
CUT = set()


def select(res):
    """S58's report -> the two selected arms (the rule in the docstring)."""
    A = res["arms"]
    order = {a: i for i, a in enumerate(CANDIDATES)}
    key = lambda a: (-A[a]["br"], A[a]["trans_med"] if A[a]["trans_med"] is not None else 1e9, A[a]["merged"], order[a])  # noqa: E731
    return sorted(CANDIDATES, key=key)[:2]


def configure():
    """Fix the selection once (stored in this screen's store); returns False if S58 is incomplete or UNTESTED."""
    st = ec.load_store(NAME)
    sel = st["meta"].get("selected")
    if sel is None:
        c16.STATE["shas"].setdefault(s58.NAME, ec.load_store(s58.NAME)["meta"]["code_shas"][-1])
        c16.STATE["batch_cpu"] = c16.STATE["batch_cpu"] or ec.cpu_model()
        res, _ = s58.report()
        if not (res["complete"] and res["valid"]):
            print(f"  S58b: S58 complete {res['complete']}, valid {res['valid']}: no selection, S58b does not run", flush=True)
            return False
        sel = select(res)
        st["meta"]["selected"] = sel
        st["meta"]["selection_basis"] = {a: dict(br=res["arms"][a]["br"], trans_med=res["arms"][a]["trans_med"],
                                                 merged=res["arms"][a]["merged"]) for a in CANDIDATES}
        ec.save_store(NAME, st)
        print(f"  S58b: selected {sel} from S58 ({st['meta']['selection_basis']}) (stored)", flush=True)
    else:
        print(f"  S58b: selection {sel} (stored)", flush=True)
    SELECTED[:] = sel
    ARMS.clear()
    for a in ["SPLIT"] + list(sel):
        ARMS[a] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=s58.ARMS[a]["label"], sched=s58.ARMS[a]["sched"])
    ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8 + conv",
                          sched="one Adam 1e-3 for every parameter")
    return True


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def check():
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", REC_SEED)
    rows = []
    jobs = []
    for a in ["SPLIT"] + SELECTED:
        if a in ("SPLIT", "RESET"):
            jobs.append((a, dict(arm=a, seed=REC_SEED, iters=CHECK_TRIG, thr=0.0, total=ITERS), CHECK_TRIG, "threshold 0"))
        elif a.startswith("GUMBEL"):
            jobs.append((a, dict(arm=a, seed=REC_SEED, iters=CHECK_SHORT, gumbel_scale=0.0), CHECK_SHORT, "noise scale 0"))
        elif a == "SWITCH":
            jobs.append((a, dict(arm=a, seed=REC_SEED, iters=CHECK_SHORT, alpha=0.0), CHECK_SHORT, "alpha 0"))
    if "GUMBEL_W" in SELECTED:
        jobs.append(("TWO_NODE", dict(arm="TWO_NODE", seed=REC_SEED, iters=CHECK_SHORT), CHECK_SHORT, "control"))
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
    two = next((r for (j, r) in res if j[0] == "TWO_NODE"), None)
    for (arm, p, t, how), r in res:
        if arm == "TWO_NODE":
            continue
        if arm == "GUMBEL_W":
            eq, c = s58.eq_upto(r, two, t)
            gc = (r.get("keyroute") or {}).get("end", {}).get("gumbel_calls")
            rows.append((f"GUMBEL_W (noise scale 0) equals the TWO_NODE control bit for bit through {t} on config b (curve {c}); "
                         f"noisy-path forwards {gc}", eq and gc == t))
            continue
        eq, c = s58.eq_upto(r, ref, t) if ref is not None else (False, None)
        if arm in ("SPLIT", "RESET"):
            eq = eq and r["splits"] == 0 and len(r["checks"]) >= 2
        if arm == "GUMBEL_RW":
            eq = eq and (r.get("keyroute") or {}).get("end", {}).get("gumbel_calls") == t
        if arm == "SWITCH":
            eq = eq and r.get("switch_n") == t
        rows.append((f"{arm} ({how}) equals batch 16's LOCAL3_SLOW_D8_A|{REC_SEED} ({ref and ref['cpu']}) bit for bit through {t} "
                     f"(curve {c}; statistics)", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={}, selected=list(SELECTED), basis=store["meta"].get("selection_basis"))
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    learned = [a for a in ARMS if a != "ORACLE"]
    for arm in learned:
        rs = got[arm]
        n, k = len(rs), sum(hc.br(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        fired = [sum(1 for c in r.get("checks") or [] if c.get("fired")) for r in rs.values()]
        res["arms"][arm] = dict(
            n=n, br=k, bound=sum(1 for r in rs.values() if r.get("transition") is not None), band=hc.band(k, n),
            wilson=hc.wilson(k, n), classes=hc.classes(rs), merged=sum(1 for r in rs.values() if hc.fclass(r) == "MERGED"),
            trans_med=c16.med(tr), trans=tr, br_by_28800=sum(1 for r in rs.values() if hc.br(r) and r["transition"] <= 28800),
            br_key_too=sum(1 for r in rs.values() if hc.br(r) and s58.kr_end(r).get("one_to_one_key")),
            ops=sum(fired), complete=n == len(SEEDS), bound_seeds=sorted(s for s, r in rs.items() if hc.br(r)),
            maps_end=sorted(hc.per_channel(r["end"].get("ch_map")) for r in rs.values() if not hc.br(r)))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for a in SELECTED:
        res["cmp"][a] = hc.paired(B[a], B["SPLIT"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        res["readings"] = {"S58b": "INCOMPLETE, no reading"}
    elif not valid:
        res["readings"] = {"S58b": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        per = {a: (res["cmp"][a]["c"] - res["cmp"][a]["b"] <= 1) for a in SELECTED}
        a1, a2 = SELECTED
        k1, k2 = res["arms"][a1]["br"], res["arms"][a2]["br"]
        best = [a1] if k1 > k2 else [a2] if k2 > k1 else [a1, a2]
        holds = any(per[a] for a in best)
        res["readings"] = {a: (f"{a} carries to eight streams" if per[a] else f"{a} does not carry")
                              + f" (SPLIT only {res['cmp'][a]['c']} - {a} only {res['cmp'][a]['b']} [<= 1])" for a in SELECTED}
        res["readings"]["S58b"] = ("carries to eight streams" if holds else "does not carry") + f" (better arm: {'/'.join(best)})"
    res["complete"], res["valid"] = complete, valid
    return res, got
