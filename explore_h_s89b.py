#!/usr/bin/env python
"""
explore_h_s89b.py — EXPLORATORY, not a result. Session H, round 6, screen S89b: the split with rare key types on delta
memory, at eight streams. Does the all-position trigger, with the exact-copy split, still bind eight streams on delta
channels when each key type is rare across the corpus (32-key pool), on the grouped layout where key positions are
still fixed (3j+1)? Branch claude/explore-H; seeds 2912-2923 (round-6 block 2900-2959; S89 used 2900-2911, S89c
2924-2933; no 2900-2959 elsewhere on any branch, Referee V's scan, screen_review_s89_H.md §3); outputs under
explore_out/H/. Child: explore_h_s89b_child.

TASK: S89's (explore_h_s89_child.PoolTask, K = 32: test_window_gate Part C's grouped layout with each sequence's 4 keys
drawn without replacement from 32 key types; equal to the main line's batches bit for bit at pool size 4). F's
explore_f_tasks.py has no grouped-layout pool variant at F's head 52e2ea7 (RandHeaderTask-pool / -allpool are S70's
lite layout, S = 2), so nothing of F's is vendored and S89's own pool draw is used with its CHECK.
MEMORY: test_delta_recipe's delta memory: explore_delta_mem (F's 9a6c2bf, byte for byte), beta 1 fixed, L2 keys, tied
write, decay 0.95, applied after the LOCAL3 conversion by explore_f_delta_fast.to_delta_fast. Both modules are served
from the main line at a581027 (test_delta_recipe's X commit).
CONFIGURATION: S = 8, P = 4, k = 16, conv, 43200 updates; LOCAL3 + SLOW (the gate group at 1e-3, the rest at 1e-4 until
2400), Adam; no hinge. The trigger: S89's (S84's constants: checks every 2400 from a 2400 reference, fire if the
64-sequence probe's accuracy < 0.95 and rose < 0.02, gap 4800, last check 40800), the exact copy W_g[c0] := W_g[c*].
THE SCHEDULE (fixed now; the user's plan): first eligible check 4800 and cap 3, as S89, unless S89c reads "the schedule
was the limit". Then DELTA_SPLIT and DELTA_ALLMASS take that arm's schedule (BASE_LATE: first check 12000; BASE_CAP6:
cap 6). If both S89c arms meet the rule, the one with more "V only" pairs is taken; at a tie, BASE_LATE. The learned
arms are queued only after S89c is complete, and the schedule used is stored in every run record (trig_cfg) and printed
with the readings.
ARMS:
  ORACLE_DELTA   the perfect gate ceil8_D8 + conv, converted to delta (test_delta_recipe's CEIL_DELTA form), one Adam
                 1e-3, seeds 2912-2913 (validity). It runs first.
  DELTA_SPLIT    delta + the recipe + the key-position trigger (KEYMASS at positions 3j+1, which the generator defines:
                 the labelled ceiling), seeds 2914-2923.
  DELTA_NOSPLIT  delta + the recipe, no trigger (test_delta_recipe's DELTA_SLOW).
  DELTA_ALLMASS  delta + the recipe + the all-position trigger (S87a / S89's ALLMASS).
  The learned arms are paired on 2914-2923.
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition; v1 and v2 classes.
VALIDITY (fixed now): the learned arms are queued only if ORACLE_DELTA binds both seeds within 43200; otherwise the screen
is UNTESTED and no learned arm runs.
READINGS (fixed now; 10 paired seeds; "X only" = X bound routed and the other not; "X within 1 pair of Y" = (Y only - X
only) <= 1; each printed with its exact one-sided McNemar p):
  "the labelled recipe itself fails here"   DELTA_SPLIT < 7/10. Then no other reading.
  "the split is not needed here"            DELTA_NOSPLIT within 1 pair of DELTA_SPLIT. test_delta_recipe's "no room" at
                                            pool size 4 (DELTA_SLOW 38/40 pooled) makes this the expected outcome. Then
                                            the trigger's labels are moot on delta memory at rare keys, and DELTA_ALLMASS
                                            is printed 'DELTA_ALLMASS: moot (the split is not needed here); <counts>',
                                            with no other name (Referee V's fix 3).
  Otherwise, for DELTA_ALLMASS:
    "DELTA_ALLMASS: the trigger survives"             within 1 pair of DELTA_SPLIT AND beats DELTA_NOSPLIT by >= 4 vs 0.
    "DELTA_ALLMASS: the trigger needs key positions"  DELTA_SPLIT beats it by >= 4 vs 0.
    "DELTA_ALLMASS: no reading applies"               otherwise.
CAVEAT (printed with every reading; Referee V's fix 1): the pool makes key types rare across the corpus (0.0105 per type
against 0.084 at pool size 4). It does not move the keys: they stay at 3j+1, are the most repeated types within a
sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. The
key-position mask is the main line's, unchanged. A reading is about rare key types on a layout with positional keys,
not about unmarked key positions.
OPERATING CHARACTERISTICS (explore_h_oc.py, Referee V's model, exact; rho = 0 / 0.5 / 1, the share of seeds with one
shared difficulty; printed with the readings):
  SPLIT 0.9 and NOSPLIT 0.9 (test_delta_recipe's rates at pool size 4): "not needed" 0.87/0.94/0.99 at any ALLMASS rate;
  "fails" 0.01.
  SPLIT 0.95, NOSPLIT 0.95: "not needed" 0.94/0.98/1.00.
  SPLIT 0.9, NOSPLIT 0.5: ALLMASS 0.9: "survives" 0.44/0.53/0.61, "no reading" 0.46/0.39/0.33; ALLMASS 0.7: "survives"
  0.10/0.11/0.07, "needs key positions" 0.15/0.15/0.12, "no reading" 0.66-0.75; ALLMASS 0.5: "needs key positions"
  0.45/0.54/0.61.
  P(SPLIT < 7/10 | true 0.9) = 0.013. 4 vs 0 is one-sided p = 0.0625. "No reading applies" means ten pairs do not decide.
DESCRIPTIVE (no reading): per run every firing (update, probe accuracy, the pair copied, op_ok and the key-position and
all-position pairs' labels on routing_k's stream -> channel map at VAL positions, the map before); with each trigger arm,
its firings, how many had a one-to-one map before the fire and how many were labelled ok (Referee V's fix 5); the probe
accuracy at every check; the cap-blocked checks; transitions; held-out accuracy at 24000 and at the end; v1 / v2.
Side by side with S89 (Hebbian, same task; descriptive, different seeds).
CHECKS (before any run): explore_common.repro_check; the child's unit checks (the delta modules served at a581027,
explore_delta_mem equal to F's 9a6c2bf and both to the main line's head ed9ca2b; F's fast_checks; the conversion adds and
changes no parameter; the perfect gate's arm equal to test_window_gate's CEIL_C but for key and label); asserted, this CPU
(test_delta_recipe's X records are from a Xeon @ 2.10GHz, torch 2.14.0+cu130, read at the main-line commit 8ae660d that
recorded them): at pool size 4, DELTA_SPLIT|583 equals X's DELTA_SPLIT|583 bit for bit through 7200 (curve, statistics,
gradient norms, the trigger rows' common fields; its first firing at 4800), DELTA_NOSPLIT|583 equals X's DELTA_SLOW|583
through 4800, ORACLE_DELTA|580 equals X's CEIL_DELTA|580 through 2400; at pool size 32 on seed 2959 (not a screen seed):
DELTA_SPLIT and DELTA_ALLMASS with the threshold at 0 equal DELTA_NOSPLIT bit for bit through 4800. S89's child unit
checks (the pool task, the exact copy, token-ids-only statistics, the trigger) passed with explore_h_s89_child's code SHA
6e0a80a55be3, which this child imports unchanged.
STAGES: H_S89B_STAGE=oracle runs ORACLE_DELTA only. Any other value queues the learned arms only after ORACLE_DELTA bound
2/2 and S89c is complete.
CORRECTION (21 October 2026, after the validity stage; no learned arm had run): s89c_schedule() read S89c as incomplete,
because the harness starts STATE["batch_cpu"] at None and setdefault left it there, so no S89c run was found. It would
have kept the learned arms unqueued (never run on a wrong schedule). ORACLE_DELTA bound 1/2, so the screen is UNTESTED
by its own rule and no learned arm runs either way. The lookup now sets the CPU.
"""

import json
import os
import subprocess
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

NAME = "H/s89b"
CHILD = "explore_h_s89b_child"
MAIN = True
IDEA = "the split's trigger at eight streams on delta memory with keys drawn from a pool of 32"
SOURCE = "S89 (Hebbian: the labelled recipe fails at pool 32) and test_delta_recipe (delta needs no split at pool 4)"
CHANGE = "delta memory (beta 1, L2 keys) on S89's pool task: DELTA_SPLIT / DELTA_NOSPLIT / DELTA_ALLMASS"
PAIRING = "seeds 2914-2923, the three learned arms paired; ORACLE_DELTA 2912-2913"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2914, 2924))
ORACLE_SEEDS = (2912, 2913)
LEARNED = ("DELTA_SPLIT", "DELTA_NOSPLIT", "DELTA_ALLMASS")
TRIGGERED = ("DELTA_SPLIT", "DELTA_ALLMASS")
ARMS = {"ORACLE_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8, delta, key pool 32",
                             sched="one Adam 1e-3")}
ARMS.update({a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a + ", key pool 32",
                     sched="LOCAL3 + SLOW, delta" + ("" if a == "DELTA_NOSPLIT" else " + exact-copy trigger")) for a in LEARNED})
ALL_ARMS = dict(ARMS)
CUT = set()
DELTA_REC = "8ae660d"      # the main-line commit that recorded results/X/delta_recipe_results.json
CAVEAT = s89.CAVEAT
OC = ("operating characteristics (explore_h_oc.py, exact, rho 0 / 0.5 / 1): at SPLIT 0.9 and NOSPLIT 0.9 'the split is not "
      "needed here' 0.87/0.94/0.99 at any ALLMASS rate; at SPLIT 0.9 and NOSPLIT 0.5, ALLMASS 0.9: 'survives' 0.44/0.53/0.61; "
      "ALLMASS 0.7: 'survives' 0.10/0.11/0.07, 'needs key positions' 0.15/0.15/0.12; ALLMASS 0.5: 'needs key positions' "
      "0.45/0.54/0.61; P(SPLIT < 7/10 | 0.9) = 0.013; 4 vs 0 is one-sided p = 0.0625; 'no reading applies' means ten pairs "
      "do not decide.")
SCHEDULE = {}              # resolved by configure() from S89c's reading (empty = S89's: first 4800, cap 3)
SEL = s89.FIELDS + ("op_cs", "op_c0", "op_ok", "state_zeroed")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS, schedule=SCHEDULE))


def s89c_schedule():
    """(complete, schedule) from S89c's stored runs and its reading rule."""
    import explore_h_s89c as s89c
    st = ec.load_store(s89c.NAME)
    if not st["meta"].get("code_shas"):
        return False, {}
    if not isinstance(c16.STATE.get("shas"), dict):
        c16.STATE["shas"] = {}
    c16.STATE["shas"][s89c.NAME] = st["meta"]["code_shas"][-1]
    c16.STATE["batch_cpu"] = c16.STATE.get("batch_cpu") or ec.cpu_model()      # the harness starts it at None
    c16.STATE["other_cpus"] = c16.STATE.get("other_cpus") or []
    res, _ = s89c.report()
    if not res["complete"]:
        return False, {}
    C = res["cmp"]
    win = [(C[f"{v} vs BASE"]["b"], -i, v) for i, v in enumerate(s89c.NEW) if C[f"{v} vs BASE"]["b"] >= 4 and C[f"{v} vs BASE"]["c"] == 0]
    if not win:
        return True, {}
    v = max(win)[2]
    import explore_h_s89c_child as cc
    return True, dict(cc.ARM[v], source=f"S89c {v}")


def configure():
    ARMS.clear()
    if os.environ.get("H_S89B_STAGE") == "oracle":
        ARMS["ORACLE_DELTA"] = ALL_ARMS["ORACLE_DELTA"]
        return True
    ARMS.update(ALL_ARMS)
    sha = c16.code_sha(CHILD, MAIN)[0]
    st = ec.load_store(NAME)
    o = {s: st["runs"].get(c16.key("ORACLE_DELTA", s, sha, ec.cpu_model())) for s in ORACLE_SEEDS}
    tr = {s: (r.get("transition") if r is not None and r.get("ok") else None) for s, r in o.items()}
    done = all(r is not None and r.get("ok") for r in o.values())
    cdone, sched = s89c_schedule()
    SCHEDULE.clear()
    SCHEDULE.update(sched)
    ok = done and all(t is not None for t in tr.values()) and cdone
    print(f"  {NAME}: ORACLE_DELTA {tr} (both stored: {done}); S89c complete: {cdone}; schedule "
          f"{SCHEDULE or 'S89s (first 4800, cap 3)'} -> " + ("the learned arms are queued" if ok else "no learned arm is queued"),
          flush=True)
    return ok


def record(path, key):
    return json.loads(subprocess.run(["git", "-C", HERE, "show", f"{DELTA_REC}:{path}"], capture_output=True, text=True,
                                     check=True).stdout)["runs"][key]


def check():
    xs, xn, xc = (record("results/X/delta_recipe_results.json", k) for k in ("DELTA_SPLIT|583", "DELTA_SLOW|583", "CEIL_DELTA|580"))
    jobs = [("SPLIT583", dict(arm="DELTA_SPLIT", seed=583, iters=7200, K=4, total=ITERS)),
            ("NOSPLIT583", dict(arm="DELTA_NOSPLIT", seed=583, iters=4800, K=4, total=ITERS)),
            ("ORACLE580", dict(arm="ORACLE_DELTA", seed=580, iters=2400, K=4))]
    jobs += [(f"{a}2959", dict(arm=a, seed=2959, iters=4800, thr=0.0, total=ITERS)) for a in TRIGGERED]
    jobs += [("DELTA_NOSPLIT2959", dict(arm="DELTA_NOSPLIT", seed=2959, iters=4800, total=ITERS))]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]

    def same(r, rec, t, grad=True):
        cq = [x for x in r["curve"] if x[0] <= t] == [x for x in rec["curve"] if x[0] <= t] and len([x for x in r["curve"] if x[0] <= t]) == t // 1200
        sq = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t] == [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= t]
        gq = (not grad) or {k: v for k, v in r["grad"].items() if int(k) <= t} == {k: v for k, v in rec["grad"].items() if int(k) <= t}
        return cq, sq, gq
    r = res["SPLIT583"]
    cq, sq, gq = same(r, xs, 7200)
    tq = s89.rows_eq(r, xs, 7200, SEL)
    fired = [x["step"] for x in r["checks"] if x["fired"]]
    rows.append((f"pool 4: DELTA_SPLIT|583 equals X's DELTA_SPLIT|583 ({xs['cpu']}; {DELTA_REC}) bit for bit through 7200 (curve "
                 f"{cq}: {[x[:2] for x in r['curve']]}; statistics {sq}; gradient norms {gq}; trigger rows {tq}; fired at {fired}, "
                 f"the record {[x['step'] for x in xs['checks'] if x['fired'] and x['step'] <= 7200]})",
                 cq and sq and gq and tq and fired == [4800]))
    cq, sq, gq = same(res["NOSPLIT583"], xn, 4800)
    rows.append((f"pool 4: DELTA_NOSPLIT|583 equals X's DELTA_SLOW|583 bit for bit through 4800 (curve {cq}, statistics {sq}, "
                 f"gradient norms {gq})", cq and sq and gq))
    o = res["ORACLE580"]
    cq, sq, gq = same(o, xc, 2400)
    rows.append((f"pool 4: ORACLE_DELTA|580 equals X's CEIL_DELTA|580 through 2400 (curve {[x[:2] for x in o['curve']]}: {cq}; "
                 f"statistics {sq}; gradient norms {gq})", cq and sq and gq))
    ns = res["DELTA_NOSPLIT2959"]
    for a in TRIGGERED:
        r = res[f"{a}2959"]
        eq = r["curve"] == ns["curve"] and r["stats"] == ns["stats"] and r["splits"] == 0
        rows.append((f"pool 32: {a}|2959 with the threshold at 0 equals DELTA_NOSPLIT|2959 bit for bit through 4800 (curve "
                     f"{[x[:2] for x in r['curve']]}; statistics; checks at {[x['step'] for x in r['checks']]}, none fired)", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def summarize(rs, arm):
    n, k = len(rs), sum(hc.br(r) for r in rs.values())
    tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
    f = [c for r in rs.values() for c in r.get("checks") or [] if c.get("fired")]
    out = dict(n=n, br=k, band=hc.band(k, n), wilson=hc.wilson(k, n), v1=hc.classes(rs),
               v2=dict(Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
               trans_med=c16.med(tr), trans=tr, complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
               acc_end=sorted(round(r["acc"], 2) for r in rs.values()),
               schedules=sorted({json.dumps({k: (r.get("trig_cfg") or {}).get(k) for k in ("first", "cap")}) for r in rs.values()}),
               v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")))
    if arm in TRIGGERED:
        out.update(ops=len(f), op_ok=sum(bool(c.get("op_ok")) for c in f), key_ok=sum(bool(c["key_ok"]) for c in f),
                   all_ok=sum(bool(c["all_ok"]) for c in f),
                   same_key_all=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
                   one_to_one_before=sum(len(set(c["map_before"])) == 8 for c in f), flat=sum(c["acc"] < 0.2 for c in f),
                   capped=sum(1 for r in rs.values() for c in r.get("checks") or [] if c.get("blocked_by") == "cap"))
    return out


def report():
    me = sys.modules[__name__]
    ARMS.clear()
    ARMS.update(ALL_ARMS)
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={}, caveat=CAVEAT, oc=OC)
    orc = got["ORACLE_DELTA"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    for a in LEARNED:
        res["arms"][a] = summarize(got[a], a)
    A = res["arms"]
    res["trigger_lines"] = {t: (f"{t}: {A[t]['ops']} firings, {A[t]['one_to_one_before']} with a one-to-one map before the fire, "
                                f"{A[t]['op_ok']} labelled ok (op_ok)") for t in TRIGGERED if A[t].get("ops") is not None}
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in LEARNED}
    for x, y in (("DELTA_NOSPLIT", "DELTA_SPLIT"), ("DELTA_ALLMASS", "DELTA_SPLIT"), ("DELTA_ALLMASS", "DELTA_NOSPLIT")):
        res["cmp"][f"{x} vs {y}"] = hc.paired(B[x], B[y], SEEDS)
    C = res["cmp"]
    complete = all(A[a]["complete"] for a in LEARNED) and len(orc) == len(ORACLE_SEEDS)
    if len(orc) == len(ORACLE_SEEDS) and not valid:
        rd = {"S89b": f"UNTESTED (ORACLE_DELTA bound {n_or}/{len(ORACLE_SEEDS)}); no learned arm runs"}
    elif not complete:
        rd = {"S89b": "INCOMPLETE, no reading"}
    elif A["DELTA_SPLIT"]["br"] < 7:
        rd = {"S89b": f"the labelled recipe itself fails here (DELTA_SPLIT {A['DELTA_SPLIT']['br']}/10); no other reading"}
    else:
        ns, cb, cn = C["DELTA_NOSPLIT vs DELTA_SPLIT"], C["DELTA_ALLMASS vs DELTA_SPLIT"], C["DELTA_ALLMASS vs DELTA_NOSPLIT"]
        moot = ns["c"] - ns["b"] <= 1
        rd = {"split": ("the split is not needed here" if moot else "the split is needed (no reading)")
              + f" (DELTA_SPLIT only {ns['c']}, DELTA_NOSPLIT only {ns['b']}; p(SPLIT>NOSPLIT) {ns['p_rev']:.3g})"}
        counts = (f"DELTA_SPLIT only {cb['c']}, DELTA_ALLMASS only {cb['b']} (p(SPLIT>ALLMASS) {cb['p_rev']:.3g}); DELTA_ALLMASS "
                  f"only {cn['b']} vs DELTA_NOSPLIT only {cn['c']} (p(ALLMASS>NOSPLIT) {cn['p']:.3g})")
        if moot:
            rd["DELTA_ALLMASS"] = f"DELTA_ALLMASS: moot (the split is not needed here); {counts}"
        elif cb["c"] - cb["b"] <= 1 and cn["b"] >= 4 and cn["c"] == 0:
            rd["DELTA_ALLMASS"] = f"DELTA_ALLMASS: the trigger survives ({counts})"
        elif cb["c"] >= 4 and cb["b"] == 0:
            rd["DELTA_ALLMASS"] = f"DELTA_ALLMASS: the trigger needs key positions ({counts})"
        else:
            rd["DELTA_ALLMASS"] = f"DELTA_ALLMASS: no reading applies ({counts})"
    res["schedule"] = A["DELTA_SPLIT"]["schedules"]
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got


def per_seed_log(got):
    for a in LEARNED:
        for s, r in sorted(got.get(a, {}).items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            print(f"    {a:<13} {s}: {r.get('outcome')} (transition {r.get('transition')}); " + "; ".join(
                f"{c['step']} (probe {c['acc']:.2f}) {c['op_cs']}->{c['op_c0']} op_ok {c['op_ok']} (key {c['key_cs']}->{c['key_c0']} "
                f"{c['key_ok']}, all {c['all_cs']}->{c['all_c0']} {c['all_ok']}) map before {c['map_before']}" for c in f), flush=True)
