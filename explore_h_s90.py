#!/usr/bin/env python
"""
explore_h_s90.py — EXPLORATORY, not a result. Session H, round 7, screen S92 (the round-7 prompt's name; the module,
store and logs keep the key "s90", which G's S90 and F's S90-F do not share): the pool size and the budget, the perfect
gates first and the recipe second, so that no learned arm runs on a configuration whose oracle is not reliable (the
lesson of S89b and of E's batch 21). Report 11's two candidates (a longer budget; a smaller key pool). Branch
claude/explore-H; seeds 3020-3033 (scanned before any run; Referee V, 22 October: no 3020-3059, the block, in any JSON
record on any head or in history); outputs under explore_out/H/. Child: explore_h_s90_child.

TASK: S89's PoolTask (explore_h_s89_child) on test_window_gate Part C's grouped layout (S = 8, P = 4, n_vals = 16, n_q =
1, L = 99), each sequence's 4 keys drawn without replacement from a pool of K key types, K a parameter: 4 (the main
line's task, bit for bit), 8, 16, 32. Model: k = 16, conv.
PHASE 1, THE ORACLES (seeds 3020-3023, 4 per cell; 32 runs): ORACLE_HEBB (the perfect gate ceil8_D8 + conv, one Adam
1e-3; S89's ORACLE) and ORACLE_DELTA (the same converted to test_delta_recipe's delta memory, beta 1, L2 keys; S89b's
ORACLE_DELTA), at K = 4, 8, 16, 32, each run with a budget of 86400 updates, evaluation every 1200 on the 2048 held-out
sequences, and run_one's early stop as before (three evaluations >= 0.95). Each cell's transitions are printed.
READINGS, PHASE 1 (fixed now; per memory and pool; "bind" = a transition, run_one's bound):
  "the oracle is reliable at pool p"        4/4 bind by 43200.
  "needs the longer budget at p"            4/4 bind by 86400, and not 4/4 by 43200.
  "the oracle is the limit at p"            otherwise.
  "Bind by 43200" = the main line's transition() (test_binding_onset) on the run's curve cut at 43200, which is what a
  43200-budget run of the same trajectory gives; "bind by 86400" = the run's own transition (Referee V's fix 2). A cell
  where the two definitions differ is printed.
  "the reference fails" if either memory at pool 4 (the main line's task) reads "the oracle is the limit": no reading
  about the pool, and phase 2 does not run (V's fix 3).
  P* = the largest pool at which both memories read "reliable" or "needs the longer budget". The budget named for a
  memory at P* is 43200 if it reads reliable there and 86400 if it needs the longer budget. If a pool below P* fails on
  either memory, the reading prints "pools not nested: <list>".
  Operating characteristics: an oracle that binds by a budget with probability q passes 4/4 with probability q^4 (q =
  0.95: 0.81; 0.9: 0.66; 0.8: 0.41). One failure in four is enough to move a cell down a reading, and "the oracle is
  the limit at p" can mean 3 of 4 (Wilson 0.30-0.95); each cell is printed with its k/4. Both memories pass a pool with
  probability (q_h q_d)^4 (0.66 at 0.95 each, 0.43 at 0.9). P* is the largest pool that passed, not the largest
  reliable one: with q = 0.95 at every pool, P* = 32 with probability 0.66, and 4 or none 0.04; with 1.0 / 0.95 / 0.9 /
  0.7 at 4 / 8 / 16 / 32, P* is 16 (0.41), 8 (0.36) or 4 (0.18) (Referee V, screen_review_s92_H.md §B; V's fix 4).
PHASE 2, THE RECIPE AT P* (seeds 3024-3033, paired; 30 runs; only if P* > 4; queued only after phase 1 is complete):
  BASE           Hebbian: LOCAL3 + SLOW + the key-position trigger with the exact copy (S89's BASE; the main line's
                 recipe with H's exact-copy split), at the budget phase 1 named for Hebbian memory.
  DELTA_SPLIT    delta: test_delta_recipe's delta recipe (beta 1, L2 keys) with the exact copy and the key-position
                 trigger, at the budget phase 1 named for delta memory.
  DELTA_NOSPLIT  the delta recipe without a trigger, at the same budget.
  The trigger's constants are S84's (checks every 2400 from a 2400 reference, first eligible 4800, fire if the probe's
  accuracy < 0.95 and rose < 0.02, cap 3, gap 4800), with the last check at budget - 2400 (40800 or 84000).
  Validity is phase 1's (the oracles at P* read reliable or needs the longer budget by definition).
READINGS, PHASE 2 (fixed now; 10 paired seeds; BOUND ROUTED; each printed with the caveat below):
  per memory (BASE for Hebbian, DELTA_SPLIT for delta):
    "the recipe carries to P*"          the arm binds >= 7/10.
    "the recipe does not carry to P*"   the arm binds <= 3/10.
    "no reading applies"                otherwise.
  "delta split: moot"                        both DELTA_SPLIT and DELTA_NOSPLIT bind <= 3/10: neither delta arm binds, so
                                              the split is read by its counts only, with no name (V's fix 5). Otherwise:
  "the split is not needed on delta at P*"   DELTA_NOSPLIT within 1 pair of DELTA_SPLIT ((DELTA_SPLIT only -
                                              DELTA_NOSPLIT only) <= 1; one-sided, so it also covers NOSPLIT ahead).
                                              Printed beside it: "the split costs on delta at P*" when DELTA_NOSPLIT only -
                                              DELTA_SPLIT only >= 2 (with p).
  "no reading applies (DELTA_SPLIT ahead by >= 2 pairs)"   otherwise (V's fix 6). Exact one-sided McNemar printed.
  If P* = 4, phase 2 does not run and S92 reads "no pool above 4 binds 4/4 by 86400 on both memories (P* = 4)"; if
  the reference fails or no pool passes, "the reference fails" (above).
  BUDGET GUARD (V's fix 9): the readings are "INCOMPLETE (budget mismatch)" unless every phase-1 record has budget 86400
  and every phase-2 record its memory's budget, each ending at its budget or at a three-evaluation early stop.
OPERATING CHARACTERISTICS (exact; printed with the readings): P(>= 7/10 | true rate) = 0.001, 0.011, 0.172, 0.382, 0.650,
0.879, 0.987 at 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9; P(<= 3/10) = 0.879, 0.650, 0.172, 0.055, 0.011, 0.001, 0.000. So
"carries" is likely only at a true rate of 0.8 or more and comes about one time in six at 0.5; "does not carry" is
likely at 0.3 or less and comes about one time in six at 0.5. "Split not needed" (explore_h_oc.py, V's model, rho 0 /
0.5 / 1): at SPLIT 0.9 and NOSPLIT 0.9, 0.88/0.95/1.00; at 0.9 / 0.7, 0.39/0.38/0.38; at 0.9 / 0.5, 0.09/0.07/0.05;
at 0.7 / 0.7, 0.77/0.86/1.00; at 0.7 / 0.4, 0.24/0.20/0.15. The moot branch takes 0.86/0.93/0.99 at SPLIT = NOSPLIT =
0.1 and 0.38/0.47/0.65 at 0.3 (V's §C).
LABELS: the perfect gates read stream labels (the gate is one-hot by stream). BASE and DELTA_SPLIT read the key
positions 3j+1 (the generator's layout) for the trigger's target pair; the decision to fire reads the probe's accuracy
only. op_ok, the pairs' labels and map_before are descriptive (routing_k's stream -> channel map from stream tokens) and
are read by nothing that acts. No label-free claim is made.
CAVEAT (printed with every reading; Referee V's fix 1 for S89): the pool makes key types rare across the corpus (at K =
32, 0.0105 per type; 16: 0.021; 8: 0.042; 4: 0.084). It does not move the keys: they stay at 3j+1, are the most repeated
types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate
reads. The key-position mask is the main line's, unchanged. A reading is about rare key types on a layout with
positional keys, not about unmarked key positions.
DESCRIPTIVE (no reading): per phase-1 cell the transitions and the held-out accuracy at 43200 and at the end; per
phase-2 run every firing (update, probe accuracy, the pair copied, op_ok, the key-position and all-position pairs'
labels, the stream -> channel map before it), the cap-blocked checks, the transition, the held-out accuracy at the
end.
CHECKS (before any run): explore_common.repro_check; the child's checks (the delta recipe's converter and exact copy are
test_delta_recipe's at a581027 by syntax tree, BETA equal; PoolTask(8) and PoolTask(16) well formed); at pool 4 with the
main line's split op (S89's MAINSPLIT path), the main line's X record WIN3_SPLIT_D8|540 (read at 628eb7c) through 4800,
printed and not asserted, since the record is from a Xeon @ 2.80GHz; asserted, this CPU: through this child at K = 32,
ORACLE_HEBB|2900 equals S89's ORACLE|2900 and ORACLE_DELTA|2913 equals S89b's ORACLE_DELTA|2913 through 2400 (curve,
statistics, gradient norms; the records first committed at 4324d2eb and d0095a8c, read from the working-tree store);
BASE|2902 (first committed at 7939dd4c) with an 86400 budget equals S89's BASE|2902 through 4800 (curve, statistics, the
trigger rows' common fields) and records its last check at 84000; at K = 4, DELTA_SPLIT|583 equals X's DELTA_SPLIT|583
(test_delta_recipe, 8ae660d, a Xeon @ 2.10GHz) through 4800. ADDED AFTER PHASE 2 STARTED (V's fix 8; check_extra(), run by
hand, asserted, this CPU; logged in explore_out/H/s92_checks_extra.log): DELTA_NOSPLIT|583 (K 4) equals X's
DELTA_SLOW|583 through 4800; ORACLE_HEBB|2600 (K 4) equals S87a's ORACLE|2600 through 2400; ORACLE_DELTA|580 (K 4)
equals X's CEIL_DELTA|580 through 2400.
BUDGET: phase 1 is 32 runs. A bound run stops early; a run that never binds runs all 86400 updates. At K = 32 the delta
perfect gate took 161 min for 43200 updates (S89b), so such a run may take 5 h or more. Phase 2 is about 30 runs at
43200 or 86400 updates.
STAGES: H_S90_STAGE=oracle runs phase 1 only. Any other value queues phase 2 only after phase 1 is complete and P* > 4.
Run with --push-each (explore_h_run.py; commits and pushes after every saved run, no pull).
REVISION (22 October 2026): Referee V's screen review (claude/review-V 980498e, docs/review_2026-10/screen_review_s92_H.md,
"fit to run phase 1, fix before phase 2") was written at 05:30 UTC, before phase 1's first record, and read at 17:00 UTC,
after phase 1 had completed (P* = 8) and after phase 2's first record (BASE|3026, 16:36:45 UTC). Fixes 1-7, 9 and 10 are
applied to this docstring and to report() / phase1(); fix 8's two CHECK rows are check_extra(), run by hand after phase
2 started. This file is not in the child's code SHA, so no run or record changes. Fix 2 changes no phase-1 cell (every
bound phase-1 run stopped early within 2400 updates of its transition), and the reference passes, so P* = 8 and phase
2's plan are unchanged.
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
import explore_h_s89b as s89b

NAME = "H/s90"
CHILD = "explore_h_s90_child"
MAIN = True
IDEA = "the oracles at key pools 4/8/16/32 and 86400 updates, then the recipe at the largest pool whose oracles are reliable"
SOURCE = ("report 11 (S89c: the pre-fixed reading 'the channel is the limit' -- neither schedule change reaches 4/10; memory "
          "and budget not varied; S89b: the delta oracle 1/2 at pool 32)")
CHANGE = "pool size 4/8/16/32; budget 86400 for the oracles; BASE / DELTA_SPLIT / DELTA_NOSPLIT at P*"
PAIRING = "phase 1 seeds 3020-3023 per cell; phase 2 seeds 3024-3033, paired"
LR = 1e-3
LONG = 86400
SHORT = 43200
POOLS = (4, 8, 16, 32)
MEMS = ("HEBB", "DELTA")
P1_SEEDS = (3020, 3021, 3022, 3023)
SEEDS = tuple(range(3024, 3034))
P2 = ("BASE", "DELTA_SPLIT", "DELTA_NOSPLIT")
P2_MEM = {"BASE": "HEBB", "DELTA_SPLIT": "DELTA", "DELTA_NOSPLIT": "DELTA"}
CELL = {f"{m}_P{k}": dict(arm=f"ORACLE_{m}", K=k, mem=m) for m in MEMS for k in POOLS}
P1_ARMS = {c: dict(opt="adam", lr=LR, iters=LONG, seeds=P1_SEEDS, label=f"perfect gate, {c}", sched="one Adam 1e-3")
           for c in CELL}
ARMS = dict(P1_ARMS)
PLAN = {}                          # phase 2: {"K": P*, "budget": {mem: iters}}, set by configure() from phase 1
CUT = set()
CAVEAT = ("the pool makes key types rare across the corpus (at K = 32, 0.0105 per type; 16: 0.021; 8: 0.042; 4: 0.084); it "
          "does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), "
          "and each sits next to its stream's CTX token, which the window gate reads. The key-position mask is the main "
          "line's, unchanged. A reading is about rare key types on a layout with positional keys, not about unmarked key "
          "positions.")
OC = ("operating characteristics (exact): P(>= 7/10) = 0.001, 0.011, 0.172, 0.382, 0.650, 0.879, 0.987 and P(<= 3/10) = "
      "0.879, 0.650, 0.172, 0.055, 0.011, 0.001, 0.000 at a true rate of 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9; 'split not "
      "needed' (V's model, rho 0 / 0.5 / 1) at SPLIT / NOSPLIT 0.9 / 0.9: 0.88/0.95/1.00, 0.9 / 0.7: 0.39/0.38/0.38, "
      "0.9 / 0.5: 0.09/0.07/0.05, 0.7 / 0.7: 0.77/0.86/1.00; the moot branch 0.86/0.93/0.99 at SPLIT = NOSPLIT = 0.1 and "
      "0.38/0.47/0.65 at 0.3; phase 1: a cell passes 4/4 with probability q^4 (0.81 at q = 0.95), both memories (q_h q_d)^4 "
      "(0.66 at 0.95 each); P* is the largest pool that passed, not the largest reliable one (with q = 0.95 everywhere, "
      "P* = 32 with probability 0.66); 'the oracle is the limit' can mean 3 of 4 (Wilson 0.30-0.95).")
SEL = s89.FIELDS + ("op_cs", "op_c0", "op_ok", "rep_cs", "rep_c0", "rep_ok")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def payload(arm, seed):
    if arm in CELL:
        c = CELL[arm]
        return dict(arm=c["arm"], seed=seed, iters=LONG, K=c["K"])
    return dict(arm=arm, seed=seed, iters=PLAN["budget"][P2_MEM[arm]], K=PLAN["K"])


def run_job(arm, seed):
    rec = child("run", payload(arm, seed))
    return rec


def cell_reading(t43, t86):
    """t43: the four transitions of the curves cut at 43200; t86: the runs' own transitions (None = not bound)."""
    if len(t86) < len(P1_SEEDS):
        return None
    if all(t is not None for t in t43):
        return "reliable"
    if all(t is not None for t in t86):
        return "needs the longer budget"
    return "the oracle is the limit"


def t_cut(r, t=SHORT):
    """The main line's transition() on the record's curve cut at t: what a t-budget run of the same trajectory gives."""
    import test_binding_onset as tbo
    return tbo.transition([c for c in r["curve"] if c[0] <= t])


def budget_ok(r, budget):
    """The record ran at its budget: it says so, and it ends at the budget or at a three-evaluation early stop."""
    cv = r["curve"]
    if r.get("budget") != budget or not cv:
        return False
    return cv[-1][0] == budget or (len(cv) >= 3 and all(c[1] >= 0.95 for c in cv[-3:]) and r.get("transition") is not None)


def phase1(runs_of_cell):
    """runs_of_cell(cell) -> {seed: record}. Returns per-cell transitions and readings, P* and the budgets."""
    out = {}
    for c, spec in CELL.items():
        rs = runs_of_cell(c)
        t86 = [rs[s].get("transition") for s in P1_SEEDS if s in rs]
        t43 = [t_cut(rs[s]) for s in P1_SEEDS if s in rs]
        differ = [s for s in sorted(rs) if (t_cut(rs[s]) is not None) != (rs[s].get("transition") is not None and
                                                                          rs[s]["transition"] <= SHORT)]
        out[c] = dict(trans={s: rs[s].get("transition") for s in sorted(rs)}, n=len(rs), reading=cell_reading(t43, t86),
                      k43=sum(t is not None for t in t43), k86=sum(t is not None for t in t86), differ=differ,
                      budget_ok=all(budget_ok(r, LONG) for r in rs.values()),
                      acc_end={s: round(rs[s]["acc"], 3) for s in sorted(rs)},
                      acc_43200={s: next((x[1] for x in rs[s]["curve"] if x[0] == SHORT), None) for s in sorted(rs)},
                      stopped={s: rs[s].get("stopped_at") for s in sorted(rs)})
    complete = all(v["n"] == len(P1_SEEDS) for v in out.values())
    passes = lambda k: all(out[f"{m}_P{k}"]["reading"] in ("reliable", "needs the longer budget") for m in MEMS)  # noqa: E731
    ok = [k for k in POOLS if passes(k)]
    reference_ok = passes(POOLS[0])
    pstar = max(ok) if (complete and ok and reference_ok) else None
    budget = ({m: (SHORT if out[f"{m}_P{pstar}"]["reading"] == "reliable" else LONG) for m in MEMS} if pstar else None)
    return out, complete, pstar, budget


def not_nested(out, pstar):
    return [k for k in POOLS if pstar and k < pstar and not all(out[f"{m}_P{k}"]["reading"] in ("reliable", "needs the longer budget")
                                                               for m in MEMS)]


def configure():
    """Phase 1 alone (H_S90_STAGE=oracle); otherwise phase 2, only after phase 1 is complete and P* > 4."""
    ARMS.clear()
    PLAN.clear()
    if os.environ.get("H_S90_STAGE") == "oracle":
        ARMS.update(P1_ARMS)
        return True
    sha = c16.code_sha(CHILD, MAIN)[0]
    st = ec.load_store(NAME)
    cpu = ec.cpu_model()

    def roc(c):
        got = {s: st["runs"].get(c16.key(c, s, sha, cpu)) for s in P1_SEEDS}
        return {s: r for s, r in got.items() if r is not None and r.get("ok")}
    out, complete, pstar, budget = phase1(roc)
    ARMS.update(P1_ARMS)
    if complete and pstar and pstar > 4:
        PLAN.update(K=pstar, budget=budget)
        for a in P2:
            ARMS[a] = dict(opt="adam", lr=LR, iters=budget[P2_MEM[a]], seeds=SEEDS, label=f"{a}, key pool {pstar}",
                           sched=f"LOCAL3 + SLOW, {P2_MEM[a].lower()}" + ("" if a == "DELTA_NOSPLIT" else " + exact-copy trigger"))
    cells = ", ".join(f"{c}: {v['reading']}" for c, v in out.items())
    print(f"  {NAME}: phase 1 complete {complete}; readings {{{cells}}}; P* {pstar}; plan {PLAN or 'none'}", flush=True)
    return bool(PLAN)


def check():
    mx = json.loads(subprocess.run(["git", "-C", HERE, "show", f"{s89.MAINREF}:results/X/window_gate_results.json"],
                                   capture_output=True, text=True, check=True).stdout)["runs"]["WIN3_SPLIT_D8|540"]
    xs = s89b.record("results/X/delta_recipe_results.json", "DELTA_SPLIT|583")
    o89 = s89.store_run(s89.NAME, "ORACLE", 2900)
    o89b = s89b_store("ORACLE_DELTA", 2913)
    b89 = s89.store_run(s89.NAME, "BASE", 2902)
    jobs = [("OH2900", dict(arm="ORACLE_HEBB", seed=2900, iters=2400, K=32)),
            ("OD2913", dict(arm="ORACLE_DELTA", seed=2913, iters=2400, K=32)),
            ("BASE2902", dict(arm="BASE", seed=2902, iters=4800, K=32, total=LONG)),
            ("DS583", dict(arm="DELTA_SPLIT", seed=583, iters=4800, K=4, total=SHORT))]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        fm = ex.submit(lambda: c16.run_child(s89.CHILD, "run", dict(arm="MAINSPLIT", seed=540, iters=4800, K=4, total=SHORT),
                                             main=True))
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
        rm = fm.result()

    def same(r, rec, t):
        c = [x for x in r["curve"] if x[0] <= t]
        cq = c == [x for x in rec["curve"] if x[0] <= t] and len(c) == t // 1200
        sq = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t] == \
            [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= t]
        gq = {k: v for k, v in r["grad"].items() if int(k) <= t} == {k: v for k, v in rec["grad"].items() if int(k) <= t}
        return cq, sq, gq, c
    for lab, key, rec in (("ORACLE_HEBB|2900 (K 32)", "OH2900", o89), ("ORACLE_DELTA|2913 (K 32)", "OD2913", o89b)):
        cq, sq, gq, c = same(res[key], rec, 2400)
        rows.append((f"{lab} equals the recorded S89/S89b run ({rec['cpu']}) through 2400 (curve {[x[:2] for x in c]}: {cq}; "
                     f"statistics {sq}; gradient norms {gq})", cq and sq and gq))
    r = res["BASE2902"]
    cq, sq, gq, c = same(r, b89, 4800)
    tq = s89.rows_eq(r, b89, 4800, SEL)
    rows.append((f"BASE|2902 (K 32, budget 86400) equals S89's BASE|2902 through 4800 (curve {cq}; statistics {sq}; gradient norms "
                 f"{gq}; trigger rows {tq}); its last check is {r['trig_cfg']['last']}", cq and sq and gq and tq
                 and r["trig_cfg"]["last"] == LONG - 2400))
    r = res["DS583"]
    cq, sq, gq, c = same(r, xs, 4800)
    tq = s89.rows_eq(r, xs, 4800, SEL)
    rows.append((f"DELTA_SPLIT|583 (K 4) equals X's DELTA_SPLIT|583 ({xs['cpu']}; {s89b.DELTA_REC}) through 4800 (curve {cq}; "
                 f"statistics {sq}; gradient norms {gq}; trigger rows {tq})", cq and sq and gq and tq))
    cq, sq, gq, c = same(rm, mx, 4800)
    tq = s89.rows_eq(rm, mx, 4800, s89.FIELDS + ("noise_sd", "noise_seed", "state_zeroed"))
    print(f"  (printed, not asserted: CPU {mx['cpu']} vs this {ec.cpu_model()}) pool 4 with the main line's split op against X's "
          f"WIN3_SPLIT_D8|540 at {s89.MAINREF} through 4800: curve {cq}; statistics {sq}; gradient norms {gq}; trigger rows {tq}",
          flush=True)
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return check_extra() and ok


def check_extra():
    """Referee V's fix 8 (added after phase 2 started; run by hand): the K = 4 paths of this child not otherwise tied to
    a record. Asserted, this CPU."""
    import explore_h_allmass2 as s87a
    xn = s89b.record("results/X/delta_recipe_results.json", "DELTA_SLOW|583")
    xc = s89b.record("results/X/delta_recipe_results.json", "CEIL_DELTA|580")
    o87 = s89.store_run(s87a.NAME, "ORACLE", 2600)
    jobs = [("DN583", dict(arm="DELTA_NOSPLIT", seed=583, iters=4800, K=4, total=SHORT)),
            ("OH2600", dict(arm="ORACLE_HEBB", seed=2600, iters=2400, K=4)),
            ("OD580", dict(arm="ORACLE_DELTA", seed=580, iters=2400, K=4))]
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
    rows = []
    for lab, key, rec, t in (("DELTA_NOSPLIT|583 (K 4) = X's DELTA_SLOW|583 (8ae660d)", "DN583", xn, 4800),
                             ("ORACLE_HEBB|2600 (K 4) = S87a's ORACLE|2600", "OH2600", o87, 2400),
                             ("ORACLE_DELTA|580 (K 4) = X's CEIL_DELTA|580 (8ae660d)", "OD580", xc, 2400)):
        r = res[key]
        c = [x for x in r["curve"] if x[0] <= t]
        cq = c == [x for x in rec["curve"] if x[0] <= t] and len(c) == t // 1200
        sq = [x for x in r["stats"] if x["step"] != "end" and x["step"] <= t] == \
            [x for x in rec["stats"] if x["step"] != "end" and x["step"] <= t]
        gq = {k: v for k, v in r["grad"].items() if int(k) <= t} == {k: v for k, v in rec["grad"].items() if int(k) <= t}
        rows.append((f"{lab} bit for bit through {t} ({rec['cpu']}): curve {[x[:2] for x in c]} {cq}; statistics {sq}; gradient "
                     f"norms {gq}", cq and sq and gq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME} (added after phase 2 started): {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def s89b_store(arm, seed):
    return s89.store_run(s89b.NAME, arm, seed)


def summarize(rs):
    n, k = len(rs), sum(hc.br(r) for r in rs.values())
    tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
    f = [c for r in rs.values() for c in r.get("checks") or [] if c.get("fired")]
    return dict(n=n, br=k, band=hc.band(k, n) if n else "--", wilson=hc.wilson(k, n) if n else (0.0, 1.0), v1=hc.classes(rs),
                v2=dict(Counter(r.get("outcome_v2") for r in rs.values() if r.get("outcome_v2") != "BOUND ROUTED")),
                trans_med=c16.med(tr), trans=tr, complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if hc.br(r)),
                ops=len(f), op_ok=sum(bool(c.get("op_ok")) for c in f), flat=sum(c["acc"] < 0.2 for c in f),
                one_to_one_before=sum(len(set(c["map_before"])) == 8 for c in f),
                capped=sum(1 for r in rs.values() for c in r.get("checks") or [] if c.get("blocked_by") == "cap"),
                acc_end=sorted(round(r["acc"], 2) for r in rs.values()))


def report():
    me = sys.modules[__name__]
    st = ec.load_store(NAME)
    ARMS.clear()
    ARMS.update(P1_ARMS)
    sha = c16.STATE["shas"][NAME]
    got1 = {c: c16.runs_of(me, c, store=st) for c in CELL}
    out, complete1, pstar, budget = phase1(lambda c: got1[c])
    res = dict(phase1=out, complete1=complete1, pstar=pstar, budget=budget, readings={}, caveat=CAVEAT, oc=OC, arms={}, cmp={})
    rd = {}
    for m in MEMS:
        rd[f"phase 1 {m}"] = "; ".join(f"pool {k}: {out[f'{m}_P{k}']['reading'] or 'incomplete'} "
                                       f"({out[f'{m}_P{k}']['k43']}/4 by 43200, {out[f'{m}_P{k}']['k86']}/4 by 86400)" for k in POOLS)
    differ = {c: v["differ"] for c, v in out.items() if v["differ"]}
    rd["phase 1, 43200 by the cut curve vs the run's transition"] = f"differ in {differ or 'no run'}"
    got = dict(got1)
    reference_ok = all(out[f"{m}_P{POOLS[0]}"]["reading"] in ("reliable", "needs the longer budget") for m in MEMS)
    if not complete1:
        rd["S92"] = "INCOMPLETE (phase 1), no reading"
    elif not all(v["budget_ok"] for v in out.values()):
        rd["S92"] = "INCOMPLETE (budget mismatch in phase 1), no reading"
    elif not reference_ok:
        rd["S92"] = "the reference fails (pool 4 does not bind 4/4 by 86400 on both memories): no reading about the pool; phase 2 does not run"
    elif not pstar or pstar <= 4:
        rd["S92"] = f"no pool above 4 binds 4/4 by 86400 on both memories (P* = {pstar}); phase 2 does not run"
    else:
        nn = not_nested(out, pstar)
        rd["P*"] = f"P* = {pstar}; budgets {budget}" + (f"; pools not nested: {nn}" if nn else "; the passing pools are nested")
        for a in P2:
            ARMS[a] = dict(opt="adam", lr=LR, iters=budget[P2_MEM[a]], seeds=SEEDS)
            got[a] = c16.runs_of(me, a, store=st)
            res["arms"][a] = summarize(got[a])
        A = res["arms"]
        B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in P2}
        res["cmp"]["DELTA_NOSPLIT vs DELTA_SPLIT"] = c = hc.paired(B["DELTA_NOSPLIT"], B["DELTA_SPLIT"], SEEDS)
        res["cmp"]["DELTA_SPLIT vs BASE"] = hc.paired(B["DELTA_SPLIT"], B["BASE"], SEEDS)
        bud = all(budget_ok(r, budget[P2_MEM[a]]) for a in P2 for r in got[a].values())
        if not all(A[a]["complete"] for a in P2):
            rd["S92"] = "INCOMPLETE (phase 2), no reading"
        elif not bud:
            rd["S92"] = "INCOMPLETE (budget mismatch in phase 2), no reading"
        else:
            for a, mem in (("BASE", "Hebbian"), ("DELTA_SPLIT", "delta")):
                k = A[a]["br"]
                s = ("the recipe carries to P*" if k >= 7 else "the recipe does not carry to P*" if k <= 3 else "no reading applies")
                rd[f"{mem} ({a})"] = f"{s} ({a} {k}/10 at pool {pstar}, budget {budget[P2_MEM[a]]})"
            counts = (f"DELTA_SPLIT {A['DELTA_SPLIT']['br']}/10, DELTA_NOSPLIT {A['DELTA_NOSPLIT']['br']}/10; DELTA_SPLIT only "
                      f"{c['c']}, DELTA_NOSPLIT only {c['b']}; p(SPLIT>NOSPLIT) {c['p_rev']:.3g}, p(NOSPLIT>SPLIT) {c['p']:.3g}")
            if A["DELTA_SPLIT"]["br"] <= 3 and A["DELTA_NOSPLIT"]["br"] <= 3:
                rd["delta split"] = f"moot (neither delta arm binds): {counts}"
            elif c["c"] - c["b"] <= 1:
                rd["delta split"] = ("the split is not needed on delta at P*"
                                     + ("; the split costs on delta at P*" if c["b"] - c["c"] >= 2 else "") + f" ({counts})")
            else:
                rd["delta split"] = f"no reading applies (DELTA_SPLIT ahead by >= 2 pairs) ({counts})"
    rd["caveat (with every reading)"] = CAVEAT
    rd["operating characteristics"] = OC
    res["readings"] = rd
    res["complete"] = complete1 and (not pstar or pstar <= 4 or all(res["arms"].get(a, {}).get("complete") for a in P2))
    res["valid"] = True
    ARMS.clear()
    ARMS.update(P1_ARMS)
    if pstar and pstar > 4:
        for a in P2:
            ARMS[a] = dict(opt="adam", lr=LR, iters=budget[P2_MEM[a]], seeds=SEEDS)
    return res, got


def per_seed_log(got):
    for c in CELL:
        rs = got.get(c, {})
        print(f"    {c:<9} transitions {[rs[s].get('transition') for s in sorted(rs)]}; accuracy at the end "
              f"{[round(rs[s]['acc'], 3) for s in sorted(rs)]}", flush=True)
    for a in P2:
        for s, r in sorted(got.get(a, {}).items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            print(f"    {a:<13} {s}: {r.get('outcome')} (transition {r.get('transition')}); " + "; ".join(
                f"{c['step']} (probe {c['acc']:.2f}) {c['op_cs']}->{c['op_c0']} op_ok {c['op_ok']} map before {c['map_before']}"
                for c in f), flush=True)
