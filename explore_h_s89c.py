#!/usr/bin/env python
"""
explore_h_s89c.py — EXPLORATORY, not a result. Session H, round 6, screen S89c: was the trigger's schedule what failed on
the 32-key pool task? S89 (report 10) found the labelled recipe itself failing there (BASE 1/10). Every run had fired
at 4800 in the flat start and used its cap of 3 by 14400-36000, and the late merges found no split left. That is S79's
cap failure at a cap of three. Two schedule changes, each alone, against a BASE rerun on the same ten seeds. Branch
claude/explore-H; seeds 2924-2933 (round-6 block 2900-2959; 2900-2923 are S89's and S89b's; no 2900-2959 elsewhere on any
branch, Referee V's scan, screen_review_s89_H.md §3); outputs under explore_out/H/. Child: explore_h_s89c_child.

TASK, MODEL AND RECIPE: S89's, unchanged (explore_h_s89_child: PoolTask, keys drawn from a pool of 32 on test_window_gate
Part C's grouped layout; S = 8, P = 4, k = 16, conv, 43200 updates, Hebbian memory, LOCAL3 + SLOW, Adam; the exact copy;
the key-position target).
ARMS (seeds 2924-2933, all three paired):
  BASE        S89's BASE rerun (first eligible check 4800, cap 3).
  BASE_LATE   the trigger's first eligible check moved from 4800 to 12000; every other constant unchanged (checks
              every 2400 from the 2400 reference, so the rise is measured from 9600 at 12000; cap 3; gap 4800).
  BASE_CAP6   the cap raised from 3 to 6; the first eligible check unchanged (4800).
VALIDITY: S89's perfect gate, 2/2 on 2900-2901 (at 16800 and 10800) on the same task and code path
(explore_h_s89_child, code SHA 6e0a80a55be3). No new perfect-gate run.
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition; v1 and v2 classes.
READINGS (fixed now; 10 paired seeds; "V only" = V bound routed and BASE not; exact one-sided McNemar printed with each):
  "the schedule was the limit"   BASE_LATE or BASE_CAP6 beats BASE by >= 4 vs 0 (V only >= 4, BASE only = 0).
  "the channel is the limit"     neither BASE_LATE nor BASE_CAP6 reaches 4/10.
  "no reading applies"           otherwise.
  The first two cannot both hold (4 vs 0 means at least 4/10). The comparison of the two new arms with each other is
  printed, with no reading. If "the schedule was the limit" applies, the all-position trigger question reopens on Hebbian
  memory with the new schedule, and S89b's delta arms get it too (the user's plan; not run here).
CAVEAT (printed with every reading; Referee V's fix 1 for S89): the pool makes key types rare across the corpus (0.0105
per type against 0.084 at pool size 4); the keys stay at 3j+1, are the most repeated types within a sequence and sit
next to their stream's CTX token. A reading is about rare key types on a layout with positional keys.
OPERATING CHARACTERISTICS (explore_h_oc.py, V's model: exact; rho = 0 / 0.5 / 1, the share of seeds with one shared
difficulty), at BASE 0.1 (S89's 1/10):
  both new arms at 0.1: schedule 0.01/0.00/0.00, channel 0.97/0.98/0.99;
  both at 0.3: schedule 0.27/0.23/0.12, channel 0.42/0.50/0.65, none 0.31/0.27/0.23;
  both at 0.5: schedule 0.64/0.69/0.62, channel 0.03/0.07/0.17;
  both at 0.7: schedule 0.87/0.93/0.95.
  One arm at 0.5 and the other at BASE's 0.1: schedule 0.47/0.55/0.62, channel 0.17; one at 0.7: schedule 0.72/0.83/0.95.
  At BASE 0.3 with both new arms at 0.5, "no reading applies" is 0.68-0.76.
  So "the channel is the limit" reads a schedule change worth less than about +0.2. It also reads half the time when
  both arms are at 0.3, and it is not a test that the schedule cannot matter. "The schedule was the limit" needs a gain
  of about +0.4 to be likely. 4 vs 0 is one-sided p = 0.0625.
DESCRIPTIVE (no reading): per run every firing (update, probe accuracy, the pair, op_ok, the map before), the checks
blocked by the cap, the probe accuracy at every check, the transitions, the held-out accuracy at the end and at 24000.
CHECKS (before any run): explore_common.repro_check; the child's mechanics on S48's harness (BASE fires at 4800, 9600,
14400 and is cap-blocked from 16800; BASE_LATE first fires at 12000; BASE_CAP6 fires six times from 4800 and is
cap-blocked from 31200; BASE_LATE's rows at 2400 and 4800 equal BASE's before BASE's first operation); asserted, this
CPU, on S89's seed 2902 (not an S89c seed): BASE through this child equals S89's BASE|2902 bit for bit through 12000
(curve, statistics, the trigger rows' common fields; it fired at 4800); BASE_CAP6 equals the same record through 12000;
BASE_LATE equals S89's NOSPLIT|2902 through 12000 (curve, statistics; it cannot fire before the evaluation at 12000).
"""

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc
import explore_h_collapse as s58
import explore_h_s89 as s89

NAME = "H/s89c"
CHILD = "explore_h_s89c_child"
MAIN = True
IDEA = "S89's labelled recipe with the trigger's first check at 12000 or its cap at 6, on the 32-key pool task"
SOURCE = "S89 (report 10: BASE 1/10; every cap spent by 14400-36000) and S79 (CAP1's flat-start failures)"
CHANGE = "first eligible check 4800 -> 12000; cap 3 -> 6"
PAIRING = "seeds 2924-2933, all three arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2924, 2934))
ARM_NAMES = ("BASE", "BASE_LATE", "BASE_CAP6")
NEW = ("BASE_LATE", "BASE_CAP6")
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a + ", key pool 32",
                sched="LOCAL3 + SLOW + exact-copy trigger " + a) for a in ARM_NAMES}
CUT = set()
CAVEAT = s89.CAVEAT
OC = ("operating characteristics (explore_h_oc.py, exact, BASE 0.1, rho 0 / 0.5 / 1): both new arms at 0.3: schedule "
      "0.27/0.23/0.12, channel 0.42/0.50/0.65; at 0.5: schedule 0.64/0.69/0.62, channel 0.03/0.07/0.17; at 0.7: schedule "
      "0.87/0.93/0.95; one arm at 0.5, the other at 0.1: schedule 0.47/0.55/0.62, channel 0.17. 'The channel is the limit' "
      "reads a gain below about +0.2 (and half the time at +0.2); 'the schedule was the limit' needs about +0.4 to be "
      "likely; 4 vs 0 is one-sided p = 0.0625.")
SEL = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
       "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
       "row_dist_after", "rep_cs", "rep_c0", "rep_masses", "rep_ok", "op_cs", "op_c0", "op_ok")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    base, nos = s89.store_run(s89.NAME, "BASE", 2902), s89.store_run(s89.NAME, "NOSPLIT", 2902)
    jobs = [(a, dict(arm=a, seed=2902, iters=12000, total=ITERS)) for a in ARM_NAMES]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    for a in ("BASE", "BASE_CAP6"):
        r = res[a]
        eq, c = s58.eq_upto(r, base, 12000)
        rq = s89.rows_eq(r, base, 12000, SEL)
        fired = [x["step"] for x in r["checks"] if x["fired"]]
        rows.append((f"{a}|2902 equals S89's BASE|2902 ({base['cpu']}) bit for bit through 12000 (curve {[x[:2] for x in c]}; "
                     f"statistics; trigger rows {rq}; fired at {fired}, the record "
                     f"{[x['step'] for x in base['checks'] if x['fired'] and x['step'] <= 12000]})", eq and rq and fired[:1] == [4800]))
    r = res["BASE_LATE"]
    eq, c = s58.eq_upto(r, nos, 12000)
    rows.append((f"BASE_LATE|2902 equals S89's NOSPLIT|2902 bit for bit through 12000 (curve {[x[:2] for x in c]}; statistics; "
                 f"checks at {[x['step'] for x in r['checks']]}, eligible from {r['trig_cfg']['first']})",
                 eq and r["trig_cfg"]["first"] == 12000))
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
                v2=dict(Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
                trans_med=c16.med(tr), trans=tr, complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
                ops=len(f), op_ok=sum(bool(c.get("op_ok")) for c in f), flat=sum(c["acc"] < 0.2 for c in f),
                one_to_one_before=sum(len(set(c["map_before"])) == 8 for c in f),
                capped=sum(1 for r in rs.values() for c in r.get("checks") or [] if c.get("blocked_by") == "cap"),
                last_fire=sorted(max((c["step"] for c in r.get("checks") or [] if c.get("fired")), default=0) for r in rs.values()),
                acc_end=sorted(round(r["acc"], 2) for r in rs.values()),
                v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")))


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={a: summarize(got[a]) for a in ARM_NAMES}, cmp={}, readings={}, caveat=CAVEAT, oc=OC)
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in ARM_NAMES}
    for x, y in (("BASE_LATE", "BASE"), ("BASE_CAP6", "BASE"), ("BASE_LATE", "BASE_CAP6")):
        res["cmp"][f"{x} vs {y}"] = hc.paired(B[x], B[y], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in ARM_NAMES)
    A, C = res["arms"], res["cmp"]
    if not complete:
        rd = {"S89c": "INCOMPLETE, no reading"}
    else:
        beat = [v for v in NEW if C[f"{v} vs BASE"]["b"] >= 4 and C[f"{v} vs BASE"]["c"] == 0]
        counts = "; ".join(f"{v} {A[v]['br']}/10 (BASE only {C[f'{v} vs BASE']['c']}, {v} only {C[f'{v} vs BASE']['b']}, "
                           f"p({v}>BASE) {C[f'{v} vs BASE']['p']:.3g})" for v in NEW) + f"; BASE {A['BASE']['br']}/10"
        if beat:
            s = "the schedule was the limit (" + ", ".join(beat) + ")"
        elif all(A[v]["br"] < 4 for v in NEW):
            s = "the channel is the limit"
        else:
            s = "no reading applies"
        rd = {"S89c": f"{s} ({counts})"}
    res["readings"], res["complete"], res["valid"] = rd, complete, True
    return res, got


def per_seed_log(got):
    for a in ARM_NAMES:
        for s, r in sorted(got.get(a, {}).items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            print(f"    {a:<10} {s}: {r.get('outcome')} (transition {r.get('transition')}); fired " + "; ".join(
                f"{c['step']} (probe {c['acc']:.2f}) {c['op_cs']}->{c['op_c0']} op_ok {c['op_ok']}" for c in f)
                + f"; probe at the checks {[(c['step'], round(c['acc'], 2)) for c in r.get('checks') or []]}", flush=True)
