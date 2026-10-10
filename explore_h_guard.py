#!/usr/bin/env python
"""
explore_h_guard.py — EXPLORATORY, not a result. Session H, screen S84: the split trigger's flat start at eight streams — a
guard against firing before any stream has bound, the faster check interval with the guard, and the held-out evaluation in
place of the 64-sequence probe. Arm names and rules as given (Session F's S81 runs the same guard on delta channels and
pairs with these by name). Branch claude/explore-H; outputs under explore_out/H/. Child: explore_h_guard_child (main line
at 9c5939e).

SEEDS: the specification gave 2200-2209 (S84) and 2210-2229 (S85), with the standing rule "shift by 100 if any appears".
2200-2229 appear in results/ (curriculum_confirm's restart-attempt seeds 2204, 2205, 2215 in results/X and results/L;
2200 and 2219 in results/L/recipe_scope_L_final.log); 2300-2359 appears nowhere in explore_out/ or results/ on any branch
(scanned before any run). So S84 uses 2300-2309 (ORACLE 2300-2301) and S85 2310-2329.

BACKGROUND: S79 (seeds 1320-1329): with the exact-copy split, CAP1 bound routed 5/10 against BASE's 9/10 because its one
split fired during the flat start (held-out accuracy 0.09-0.50), and INT1200 (7/10) spent all three splits by 7200-10800 at
chance accuracy in its three failures, while binding much earlier when it worked (median 14400 vs 24000). Session R found
that a 64-sequence probe reading >= 0.95 at a true 0.88 stops the rule.

CONFIGURATION: S79's (S=8, P=4, k=16, conv, all 8 streams from step 1; LOCAL3 + SLOW, Adam, Hebbian memory, no hinge;
43200 updates; exact-copy split onto the KEYMASS target, last check 40800); seeds 2300-2309, all arms paired.
ARMS:
  BASE                S79's BASE (checks every 2400 from 4800, 2400 reference; fire if acc < 0.95 and rose < 0.02 since
                      the previous check; 64-sequence probe; cap 3; gap 4800).
  GUARD               BASE, and no split while the probe's accuracy is below 0.5.
  GUARD_INT1200       the guard with checks every 1200 from 2400 (1200 reference), gap 2400.
  EVAL                BASE with the rule reading the run's 2048-query held-out evaluation instead of the 64-sequence probe
                      (the KEYMASS target is still chosen on the 64-sequence probe).
  EVAL_GUARD_INT1200  GUARD_INT1200 reading the held-out evaluation (the guard reads it too).
  ORACLE              perfect gate ceil8_D8 + conv, seeds 2300-2301 (validity).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition; every firing with its label (labelled ok: c* holds
>= 2 streams and c0 none on the labelled map before it). Failure classes under v1 and v2.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READINGS (fixed now; 10 paired seeds; "X only" = X bound routed and BASE not; "A >= BASE - 1 pair" means (BASE only - A
only) <= 1):
  "the guard aims the split"            GUARD's firings labelled ok >= 95% (of all its firings) AND GUARD >= BASE - 1 pair.
  "faster without the cost"             GUARD_INT1200 >= BASE - 1 pair AND its median transition (over its BOUND ROUTED
                                        runs) is earlier than BASE's by >= 2400.
  "the evaluation is the better probe"  EVAL >= BASE - 1 pair.
  "costs binding" (each arm)            BASE only >= 4 and the arm only = 0.
  Exact one-sided McNemar both ways, Wilson intervals and bands printed.
DESCRIPTIVE: per arm the firings (update, accuracy read, labelled ok), the guard-blocked checks, transitions; per seed the
first split's update.
CHECKS (before any run): explore_common.repro_check; the guard and the source on the harness (with the guard off and the
probe as source, make_trigger_guard equals S79's make_trigger_cfg row for row at BASE's and INT1200's constants; with the
guard at 0.5 a model at chance never fires and every eligible check is guard-blocked; source eval reads evaluate on the
2048-query eval batch); and with the condition disabled (guard off, probe as source) each arm equals S79's recorded run on
seed 1321 BIT FOR BIT through its first split and the next check: GUARD and EVAL against S79's BASE|1321 (first split at
4800) through 7200; GUARD_INT1200 and EVAL_GUARD_INT1200 against S79's INT1200|1321 (first split at 2400) through 4800 —
curve, every statistic and the trigger's rows.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

NAME = "H/guard"
CHILD = "explore_h_guard_child"
MAIN = True
IDEA = "a guard on the split trigger's flat start; INT1200 with the guard; the held-out evaluation as the trigger's probe"
SOURCE = "S79 (CAP1 and INT1200 lost runs to splits fired at chance accuracy); Session R (probe 0.95 at true 0.88)"
CHANGE = "S79's BASE + guard 0.5 / + guard and 1200-update checks / reading the held-out evaluation / both"
PAIRING = "seeds 2300-2309 (shifted from 2200-2209), all arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2300, 2310))
ORACLE_SEEDS = (2300, 2301)
VARIANTS = ("GUARD", "GUARD_INT1200", "EVAL", "EVAL_GUARD_INT1200")
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="LOCAL3 + SLOW + exact-copy trigger " + a)
        for a in ("BASE",) + VARIANTS}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8", sched="one Adam 1e-3")
CUT = set()
S79 = "H/constants"
S79_KEYS = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "key_ok", "fired", "blocked",
            "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before", "row_dist_after")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def s79_record(arm, seed):
    for k, r in ec.load_store(S79)["runs"].items():
        a, s = k.split("|")[:2]
        if a == arm and int(s) == seed and r.get("ok"):
            return r
    return None


def check():
    jobs = [("GUARD", "BASE", 7200), ("EVAL", "BASE", 7200), ("GUARD_INT1200", "INT1200", 4800),
            ("EVAL_GUARD_INT1200", "INT1200", 4800)]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = list(ex.map(lambda j: (j, child("run", dict(arm=j[0], seed=1321, iters=j[2], disable=True, total=ITERS))), jobs))
        rows += [tuple(x) for x in fu.result()]
    for (arm, ref_arm, t), r in res:
        ref = s79_record(ref_arm, 1321)
        c1 = [c for c in r["curve"] if c[0] <= t]
        s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
        s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
        k1 = [{k: c.get(k) for k in S79_KEYS} for c in r["checks"] if c["step"] <= t]
        k2 = [{k: c.get(k) for k in S79_KEYS} for c in ref["checks"] if c["step"] <= t]
        first = next((c["step"] for c in ref["checks"] if c["fired"]), None)
        eq = bool(c1) and c1 == [c for c in ref["curve"] if c[0] <= t] and s1 == s2 and k1 == k2 and first is not None and first < t
        rows.append((f"{arm} with the condition disabled equals S79's {ref_arm}|1321 ({ref.get('cpu')}) bit for bit through {t} "
                     f"(its first split at {first}; curve, statistics, trigger rows at {[c['step'] for c in r['checks']]}; "
                     f"fired {[c['step'] for c in r['checks'] if c['fired']]})", eq))
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
        nf = sum(len(f) for f in fires.values())
        res["arms"][a] = dict(
            n=n, br=k, band=hc.band(k, n), wilson=hc.wilson(k, n), v1=hc.classes(rs),
            v2=dict(__import__("collections").Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
            trans_med=c16.med(tr), trans=tr, ops=nf, ops_ok=sum(1 for f in fires.values() for c in f if c.get("key_ok")),
            fire_acc=sorted(round(c["acc"], 2) for f in fires.values() for c in f),
            guard_blocked=sum(1 for r in rs.values() for c in r.get("checks") or [] if c.get("blocked_by") == "guard"),
            first={s: (f[0]["step"] if f else None) for s, f in fires.items()}, complete=n == len(SEEDS),
            seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
            v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for v in VARIANTS:
        res["cmp"][v] = hc.paired(B[v], B["BASE"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    A, C = res["arms"], res["cmp"]
    keep = lambda v: C[v]["c"] - C[v]["b"] <= 1                                        # noqa: E731
    if not complete:
        rd = {"S84": "INCOMPLETE, no reading"}
    elif not valid:
        rd = {"S84": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        g = A["GUARD"]
        frac = g["ops_ok"] / g["ops"] if g["ops"] else None
        rd = {"guard aims": ("the guard aims the split" if (frac is not None and frac >= 0.95 and keep("GUARD")) else "does not apply")
              + f" (labelled ok {g['ops_ok']}/{g['ops']}; BASE only {C['GUARD']['c']}, GUARD only {C['GUARD']['b']})"}
        gi, b = A["GUARD_INT1200"], A["BASE"]
        earlier = (gi["trans_med"] is not None and b["trans_med"] is not None and b["trans_med"] - gi["trans_med"] >= 2400)
        rd["faster"] = (("faster without the cost" if (keep("GUARD_INT1200") and earlier) else "does not apply")
                        + f" (BASE only {C['GUARD_INT1200']['c']}, GUARD_INT1200 only {C['GUARD_INT1200']['b']}; median transition "
                          f"{gi['trans_med']} vs BASE {b['trans_med']})")
        rd["eval"] = (("the evaluation is the better probe" if keep("EVAL") else "does not apply")
                      + f" (BASE only {C['EVAL']['c']}, EVAL only {C['EVAL']['b']})")
        for v in VARIANTS:
            rd["cost " + v] = ("costs binding" if (C[v]["c"] >= 4 and C[v]["b"] == 0) else "no cost by the rule") \
                + f" (BASE only {C[v]['c']}, {v} only {C[v]['b']})"
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
