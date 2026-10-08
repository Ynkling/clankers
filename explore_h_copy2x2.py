#!/usr/bin/env python
"""
explore_h_copy2x2.py — EXPLORATORY, not a result. Session H, screen S61: the 2x2 of the KEYMASS split's two actions (row
copy x Adam reset) under one plateau trigger, at eight streams. Branch claude/explore-H; seeds 480-489 (session H's
follow-up block 480-499); outputs under explore_out/H/. Child: explore_h_copy2x2_child (main line's modules at 9c5939e).

BACKGROUND: S58b (seeds 460-469): SPLIT (copy + reset) bound routed 10/10, RESET (reset only) 2/10 (8 MERGED, every target
labelled correct). S24 (batch 8): the copy makes the busy and idle channels' logits nearly tie (row distance 1.96-3.32 ->
0.24-0.40); noise of the same size without the tie did not split. Untested: the copy without the reset, and the copy
without noise.

CONFIGURATION: S=8, P=4, k=16, conv, all 8 streams from step 1 (run_sc with ARM["D8"]), LOCAL3 gate, SLOW (Adam), no hinge,
43200 updates, evaluation every 1200; seeds 480-489, all arms paired (same initial parameters, window and batches; the
trigger's decisions are each run's own). 1 torch thread per run.
ARMS (one plateau trigger for all: S48's rule, checks every 2400 from 4800 through 40800, probe acc < 0.95 and rise < 0.02,
<= 3 operations, >= 4800 apart, KEYMASS target c* -> c0):
  SPLIT         copy with N(0, (0.1 std(w))^2) noise on both rows + W_g's Adam state zeroed (rerun of S58b's arm)
  RESET         W_g's Adam state zeroed only (S58b's arm)
  COPY          copy with noise (split_op's row lines, the same noise draws), Adam state untouched
  COPY_NONOISE  W_g[c0] = W_g[c*] exactly, Adam state untouched
  NONE          no trigger
  ORACLE        perfect gate ceil8_D8 + conv, own recipe, seeds 480-481 (validity)
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome; routing_k's VAL-position map one-to-one and every stream >= 0.9 at the
end), failure classes (MERGED etc.), transition times; operations fired and their targets.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise every reading is UNTESTED.
READINGS (fixed now; BOUND ROUTED on the 10 paired seeds; "X only" = X bound routed and the other arm not; exact one-sided
McNemar both ways, Wilson intervals and bands printed):
  R1 "the copy alone suffices"  if (SPLIT only - COPY only) <= 1 in COPY vs SPLIT.
  R2 "the noise matters"        if COPY only >= 4 and COPY_NONOISE only = 0 in COPY vs COPY_NONOISE.
  R3 "the reset adds nothing"   if |SPLIT only - COPY only| <= 1 (SPLIT vs COPY within one discordant pair either way).
  Printed alongside (no reading): RESET vs SPLIT (S58b's comparison on new seeds), every arm vs NONE.
CHECKS (before any run; a failure stops the invocation): explore_common.repro_check; the operations on a LOCAL3 model after
one Adam step (COPY's rows = SPLIT's bit for bit and Adam state untouched; SPLIT zeroes it; COPY_NONOISE's exact copy;
RESET leaves W_g; only rows c*, c0 change); with the trigger's threshold at 0 (never fires), each of SPLIT, RESET, COPY and
COPY_NONOISE equals batch 16's recorded LOCAL3_SLOW_D8_A|260 BIT FOR BIT through 6000 (curve and every statistic); NONE
through 2400.
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

NAME = "H/copy2x2"
CHILD = "explore_h_copy2x2_child"
MAIN = True
IDEA = "the split's two actions (row copy, Adam reset) separated under one plateau trigger, at eight streams"
SOURCE = "S58b (SPLIT 10/10 vs RESET 2/10); S24's tie mechanism"
CHANGE = "LOCAL3_SLOW_D8_A + the trigger with: copy+reset / reset / copy / exact copy; and no trigger"
PAIRING = "seeds 480-489, all learned arms paired"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(480, 490))
ORACLE_SEEDS = (480, 481)
REC_SEED = 260
_sl = "LOCAL3 + SLOW (gate 1e-3 throughout; the rest 1e-4 for 1-2400, 1e-3 after); no hinge"
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=lab, sched=_sl + "; " + lab)
        for a, lab in (("SPLIT", "trigger: copy + noise + reset"), ("RESET", "trigger: reset only"),
                       ("COPY", "trigger: copy + noise"), ("COPY_NONOISE", "trigger: exact copy"), ("NONE", "no trigger"))}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8 + conv",
                      sched="one Adam 1e-3")
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    ref = s58.b16_record("LOCAL3_SLOW_D8_A", REC_SEED)
    rows = []
    jobs = [(a, dict(arm=a, seed=REC_SEED, iters=6000, thr=0.0, total=ITERS), 6000)
            for a in ("SPLIT", "RESET", "COPY", "COPY_NONOISE")] + [("NONE", dict(arm="NONE", seed=REC_SEED, iters=2400), 2400)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    for (arm, p, t), r in res:
        eq, c = s58.eq_upto(r, ref, t)
        if arm != "NONE":
            eq = eq and r["splits"] == 0 and len(r["checks"]) >= 2
        rows.append((f"{arm} ({'threshold 0' if arm != 'NONE' else 'as is'}) equals batch 16's LOCAL3_SLOW_D8_A|{REC_SEED} "
                     f"({ref and ref['cpu']}) bit for bit through {t} (curve {[x[:2] for x in c] if c else c}; statistics)", eq))
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
    learned = [a for a in ARMS if a != "ORACLE"]
    for arm in learned:
        rs = got[arm]
        n, k = len(rs), sum(hc.br(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        fired = [sum(1 for c in r.get("checks") or [] if c.get("fired")) for r in rs.values()]
        res["arms"][arm] = dict(
            n=n, br=k, bound=sum(1 for r in rs.values() if r.get("transition") is not None), band=hc.band(k, n),
            wilson=hc.wilson(k, n), classes=hc.classes(rs), merged=sum(1 for r in rs.values() if hc.fclass(r) == "MERGED"),
            trans_med=c16.med(tr), trans=tr, ops=sum(fired), ops_runs=sum(1 for f in fired if f),
            targets_ok=sum(1 for r in rs.values() for c in r.get("checks") or [] if c.get("fired") and c.get("key_ok")),
            complete=n == len(SEEDS), bound_seeds=sorted(s for s, r in rs.items() if hc.br(r)),
            maps_end=sorted(hc.per_channel(r["end"].get("ch_map")) for r in rs.values() if not hc.br(r)))
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in learned}
    for x, y in (("COPY", "SPLIT"), ("COPY", "COPY_NONOISE"), ("RESET", "SPLIT"), ("SPLIT", "NONE"), ("COPY", "NONE"),
                 ("COPY_NONOISE", "NONE"), ("RESET", "NONE")):
        res["cmp"][(x, y)] = hc.paired(B[x], B[y], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    cm = res["cmp"]
    if not complete:
        rd = {k: "INCOMPLETE, no reading" for k in ("R1", "R2", "R3")}
    elif not valid:
        rd = {k: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for k in ("R1", "R2", "R3")}
    else:
        c = cm[("COPY", "SPLIT")]
        rd = {"R1": ("the copy alone suffices" if c["c"] - c["b"] <= 1 else "does not apply")
              + f" (SPLIT only {c['c']} - COPY only {c['b']} = {c['c'] - c['b']} [<= 1])"}
        d = cm[("COPY", "COPY_NONOISE")]
        rd["R2"] = (("the noise matters" if (d["b"] >= 4 and d["c"] == 0) else "does not apply")
                    + f" (COPY vs COPY_NONOISE {d['b']} vs {d['c']} [>= 4 vs 0])")
        rd["R3"] = (("the reset adds nothing" if abs(c["c"] - c["b"]) <= 1 else "does not apply")
                    + f" (|SPLIT only {c['c']} - COPY only {c['b']}| = {abs(c['c'] - c['b'])} [<= 1])")
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
