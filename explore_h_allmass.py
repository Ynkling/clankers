#!/usr/bin/env python
"""
explore_h_allmass.py — EXPLORATORY, not a result. Session H, screen S86: what the split trigger reads — the target chosen
by the mean read-gate mass over all body positions (no task structure) instead of the key positions (KEYMASS), and a second
look at S84's held-out-evaluation trigger. Branch claude/explore-H; seeds 2330-2339 (inside session H's shifted block
2300-2359, which appears nowhere in explore_out/ or results/); outputs under explore_out/H/. Child: explore_h_allmass_child.

BACKGROUND: the main line's 50 split runs (WIN3_SPLIT_D8: window_gate X 20, L 20; split_copy X 10) store at every check
both pairs; at their 107 firings the key-position pair was a labelled target at 107, the all-position pair at 97 (X 46/47,
L 30/38, split_copy X 21/22), and the two pairs coincide at 46 (explore_h_allmass_counts.py, recomputed before any run:
explore_out/H/allmass_counts.txt). S37 (batch 13): the all-position rule mistargeted 6 of 11 splits under Muon at S=4,
k=4. S84: EVAL 10/10 vs BASE 9/10 (1 vs 0).

CONFIGURATION: S79 / S84's (S=8, P=4, k=16, conv, all 8 streams from step 1; LOCAL3 + SLOW, Adam, Hebbian memory, no
hinge; 43200 updates; the exact-copy split W_g[c0] := W_g[c*], no noise, no Adam reset; checks every 2400 from 4800 with
2400 as reference, fire if accuracy < 0.95 and it rose < 0.02 since the previous check, cap 3, gap 4800, last check 40800);
seeds 2330-2339, all arms paired. 1 torch thread per run.
ARMS:
  BASE     S79's BASE: KEYMASS target (key-position masses), the 64-sequence probe as the rule's source.
  ALLMASS  c*, c0 by the mean read-gate mass over all positions of the probe's inputs (explore_split_target_child.select's
           all-position pair, the computation test_window_gate.select() also does; it reads no task structure); the probe as
           source.
  EVAL     S84's EVAL: KEYMASS target, the rule reading the 2048-query held-out evaluation.
  ORACLE   perfect gate ceil8_D8 + conv, seeds 2330-2331 (validity).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition; every firing with its key-label and its all-label
(key_ok, all_ok on the labelled map before it; the operation's own label op_ok = the label of the pair actually used).
Failure classes v1 and v2.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READINGS (fixed now; 10 paired seeds; "A >= BASE - 1 pair" = (BASE only - A only) <= 1):
  "all-position targeting keeps it"     ALLMASS >= BASE - 1 pair.
  "aims"                                >= 90% of ALLMASS's firings are labelled ok (its own pair, all_ok).
  "costs binding" (ALLMASS)             BASE only >= 4 and ALLMASS only = 0.
  "the evaluation is the better probe"  EVAL >= BASE - 1 pair.
  Printed, descriptive (no reading): EVAL vs BASE pooled over S84 (seeds 2300-2309) and S86 (20 pairs), with the exact
  one-sided McNemar both ways.
CHECKS (before any run): explore_common.repro_check; the trigger on S48's harness (target key equals S84's trigger row for
row; target all copies onto the all-position pair); ALLMASS with the threshold at 0 (never fires) equals batch 16's
recorded LOCAL3_SLOW_D8_A|260 BIT FOR BIT through 6000 and equals BASE at threshold 0 there (curve, statistics, the
trigger's rows); the two selections recomputed on stored checks equal the records': ALLMASS rerun on seed 2300 through
4800 against S84's BASE|2300 (this CPU; checks at 2400 and 4800, the latter before its first split's operation: key_cs,
key_c0, all_cs, all_c0, both mass vectors, both labels) — asserted; and against the main line's X record
WIN3_SPLIT_D8|540 (window_gate, a Xeon @ 2.80GHz; checks at 2400 and 4800) — printed.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc
import explore_h_collapse as s58

NAME = "H/allmass"
CHILD = "explore_h_allmass_child"
MAIN = True
IDEA = "the split's target from all positions (no task structure) vs key positions; the held-out evaluation as the reading"
SOURCE = "the main line's 50 split runs (all-position pair labelled at 97/107); S37; S84's EVAL 1 vs 0"
CHANGE = "S79's BASE with the all-position target / with the evaluation as the reading"
PAIRING = "seeds 2330-2339, all arms paired; EVAL pooled with S84 descriptively"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2330, 2340))
ORACLE_SEEDS = (2330, 2331)
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="LOCAL3 + SLOW + exact-copy trigger " + a)
        for a in ("BASE", "ALLMASS", "EVAL")}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8", sched="one Adam 1e-3")
CUT = set()
SEL = ("key_cs", "key_c0", "all_cs", "all_c0", "key_masses", "all_masses", "key_ok", "all_ok")
MAINREF = "origin/claude/bdh-growth-hebbian-inference-w90069"


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def store_run(name, arm, seed):
    for k, r in ec.load_store(name)["runs"].items():
        a, s = k.split("|")[:2]
        if a == arm and int(s) == seed and r.get("ok"):
            return r
    return None


def check():
    import json
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", 260)
    s84 = store_run("H/guard", "BASE", 2300)
    mx = json.loads(subprocess.run(["git", "-C", HERE, "show", f"{MAINREF}:results/X/window_gate_results.json"],
                                   capture_output=True, text=True, check=True).stdout)["runs"]["WIN3_SPLIT_D8|540"]
    jobs = [("ALLMASS260", dict(arm="ALLMASS", seed=260, iters=6000, thr=0.0, total=ITERS)),
            ("BASE260", dict(arm="BASE", seed=260, iters=6000, thr=0.0, total=ITERS)),
            ("ALLMASS2300", dict(arm="ALLMASS", seed=2300, iters=4800, thr=0.0, total=ITERS)),
            ("ALLMASS540", dict(arm="ALLMASS", seed=540, iters=4800, thr=0.0, total=ITERS))]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    a, b = res["ALLMASS260"], res["BASE260"]
    eq, c = s58.eq_upto(a, ref, 6000)
    same = a["curve"] == b["curve"] and a["stats"] == b["stats"] and \
        [{k: x.get(k) for k in SEL + ("step", "acc", "fired")} for x in a["checks"]] == \
        [{k: x.get(k) for k in SEL + ("step", "acc", "fired")} for x in b["checks"]]
    rows.append((f"ALLMASS|260 with the threshold at 0 equals batch 16's LOCAL3_SLOW_D8_A|260 ({ref and ref['cpu']}) bit for bit through "
                 f"6000 (curve {[x[:2] for x in c] if c else c}; statistics) and equals BASE|260 at threshold 0 (curve, statistics, "
                 f"trigger rows: {same}); fired {a['splits']}", eq and same and a["splits"] == 0))

    def sel_eq(r, rec, steps):
        m = {x["step"]: x for x in r["checks"]}
        n = {x["step"]: x for x in rec["checks"]}
        return all(s in m and s in n and {k: m[s].get(k) for k in SEL} == {k: n[s].get(k) for k in SEL} for s in steps)
    first84 = next(x["step"] for x in s84["checks"] if x["fired"])
    ok = sel_eq(res["ALLMASS2300"], s84, (2400, 4800)) and [x[:2] for x in res["ALLMASS2300"]["curve"]] == \
        [x[:2] for x in s84["curve"] if x[0] <= 4800]
    rows.append((f"the two selections recomputed at S84's BASE|2300 stored checks 2400 and 4800 (its first split at {first84}) equal the "
                 f"record's ({', '.join(SEL)})", ok and first84 >= 4800))
    okm = sel_eq(res["ALLMASS540"], mx, (2400, 4800))
    m4800 = next(x for x in mx["checks"] if x["step"] == 4800)
    print(f"  (printed) the two selections recomputed at the main line's X WIN3_SPLIT_D8|540 checks 2400 and 4800 (a Xeon @ 2.80GHz; "
          f"its first split at 4800: key {m4800['key_cs']}->{m4800['key_c0']}, all {m4800['all_cs']}->{m4800['all_c0']}) equal the "
          f"record's: {okm}", flush=True)
    ok_all = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok_all &= bool(v)
    return ok_all


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    learned = ("BASE", "ALLMASS", "EVAL")
    for a in learned:
        rs = got[a]
        n, k = len(rs), sum(hc.br(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        f = [c for r in rs.values() for c in r.get("checks") or [] if c.get("fired")]
        res["arms"][a] = dict(
            n=n, br=k, band=hc.band(k, n), wilson=hc.wilson(k, n), v1=hc.classes(rs),
            v2=dict(__import__("collections").Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
            trans_med=c16.med(tr), trans=tr, ops=len(f), key_ok=sum(bool(c["key_ok"]) for c in f), all_ok=sum(bool(c["all_ok"]) for c in f),
            op_ok=sum(bool(c.get("op_ok")) for c in f), same=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
            fire_acc=sorted(round(c["acc"], 2) for c in f), complete=n == len(SEEDS),
            seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
            v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")),
            bad_fires=[(s, c["step"], c["key_cs"], c["key_c0"], c["all_cs"], c["all_c0"], c["key_ok"], c["all_ok"])
                       for s, r in rs.items() for c in r.get("checks") or [] if c.get("fired") and not c.get("op_ok")])
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for v in ("ALLMASS", "EVAL"):
        res["cmp"][v] = hc.paired(B[v], B["BASE"], SEEDS)
    g84 = ec.load_store("H/guard")
    import explore_h_guard as s84
    c16.STATE["shas"].setdefault(s84.NAME, g84["meta"]["code_shas"][-1])
    e84 = {s: hc.br(r) for s, r in c16.runs_of(s84, "EVAL", store=g84).items()}
    b84 = {s: hc.br(r) for s, r in c16.runs_of(s84, "BASE", store=g84).items()}
    res["pooled_eval"] = hc.paired(dict(e84, **B["EVAL"]), dict(b84, **B["BASE"]), s84.SEEDS + SEEDS)
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    A, C = res["arms"], res["cmp"]
    if not complete:
        rd = {"S86": "INCOMPLETE, no reading"}
    elif not valid:
        rd = {"S86": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        c = C["ALLMASS"]
        al = A["ALLMASS"]
        frac = al["all_ok"] / al["ops"] if al["ops"] else None
        rd = {"keeps": ("all-position targeting keeps it" if c["c"] - c["b"] <= 1 else "does not apply")
              + f" (BASE only {c['c']}, ALLMASS only {c['b']})",
              "aims": ("aims" if (frac is not None and frac >= 0.9) else "does not apply")
              + f" (ALLMASS firings labelled ok {al['all_ok']}/{al['ops']})",
              "costs": ("costs binding" if (c["c"] >= 4 and c["b"] == 0) else "no cost by the rule")
              + f" (BASE only {c['c']}, ALLMASS only {c['b']})"}
        c = C["EVAL"]
        rd["eval"] = (("the evaluation is the better probe" if c["c"] - c["b"] <= 1 else "does not apply")
                      + f" (BASE only {c['c']}, EVAL only {c['b']})")
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
