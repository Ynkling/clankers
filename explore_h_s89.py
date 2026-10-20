#!/usr/bin/env python
"""
explore_h_s89.py — EXPLORATORY, not a result. Session H, round 6, screen S89: the split with rare key types, at eight
streams. Does the all-position trigger, with the exact-copy split, still bind eight streams when each key type is rare
across the corpus (32-key pool), on the grouped layout where key positions are still fixed (3j+1)? Branch claude/explore-H; seeds 2900-2911 (round-6 block 2900-2959; scanned before any run: no
2900-2959 in explore_out/, results/ or specs/ on any branch); outputs under explore_out/H/. Child: explore_h_s89_child
(the main line's module tree at 9c5939e, as S86 / S87a).

TASK: test_window_gate Part C's task (S = 8, P = 4, n_vals = 16, n_q = 1, the grouped layout, length 99), with the P = 4
keys of each sequence drawn without replacement from a pool of 32 key types (explore_h_s89_child.PoolTask; F's module
has no grouped-layout pool variant at 50dc8e4 (RandHeaderTask-pool / -allpool are S70's lite layout, S = 2; the
same at F's head 52e2ea7), so the draw is built into a subclass of the main line's task; at pool
size 4 it equals the main line's batches bit for bit, CHECKed). Tokens CTX 0-7, KEY 8-39, VAL 40-55 (vocab 56).
DESCRIBED BEFORE ANY RUN (explore_h_s89_child.describe; 4096 sequences, generator 89_000; no model):
- Corpus frequency per token type: CTX 0.042 each, VAL 0.020, KEY 0.0097-0.0112 (at pool size 4 a key type is 0.084).
  The keys are now the rarest class across sequences.
- Within one sequence the layout is unchanged: each of its 4 keys occurs 8 times (once per stream, in one group of 8
  adjacent triples, at the fixed key positions 3j+1), each stream token 4 times, a value token 2.1 times on average.
  The key-position rule (BASE) therefore still reads the true key positions, which are positional on this layout.
- The repeat statistic (token types occurring more than once in the sequence's input, as G's S89 hinge reads it) does
  NOT select the keys alone on this layout. Over the 4096 sequences' 401,408 input positions it selects 385,149 (96%):
  every CTX position (131,072), every KEY position (131,072), 114,813 of the 131,072 VAL positions (88%), and the query's
  CTX and KEY. The key positions are 131,072 of the 385,149 (34%), so REPEATMASS is expected to choose close to ALLMASS.
  On this layout the repeat set is 94 of 98 input positions (every CTX and KEY position, 7/8 of the VAL positions, the
  query's CTX and KEY); REPEATMASS is ALLMASS without ~4 singleton-value positions, not a key-selecting statistic. Its
  reading is printed beside ALLMASS's as 'REPEATMASS (repeat set = X% of input positions; same pair as ALLMASS at s of n
  firings): <reading>' and is not a second finding; if the two arms chose the same pair at every firing, it is printed
  as 'REPEATMASS: as ALLMASS (same pair at every firing)'. (Referee V's fix 2.)
  The arm is run as defined (positions of repeated token types), and its overlap with the key positions is printed per
  run on its probe (key positions in the set; positions in the set that are not key positions).
CONFIGURATION: S=8, P=4, k=16, conv, 43200 updates, Hebbian memory; LOCAL3 + SLOW (gate group at 1e-3 throughout, the
rest at 1e-4 until 2400), Adam; no hinge. The trigger: S86 / S87a's rule (checks every 2400 from 4800 after a 2400
reference, fire if the 64-sequence probe's accuracy < 0.95 and rose < 0.02, cap 3, gap 4800, last check 40800), the exact
copy W_g[c0] := W_g[c*] (explore_h_ops.copy_exact = S61's copy_nonoise_op). 1 torch thread per run.
ARMS:
  ORACLE      perfect gate ceil8_D8 + conv (one Adam 1e-3), seeds 2900-2901 (validity).
  BASE        the main line's recipe with the key-position target (KEYMASS at positions 3j+1, which the generator
              defines: the labelled ceiling), seeds 2902-2911.
  ALLMASS     S87a's arm: the target pair from the mean read-gate mass over every input position.
  NOSPLIT     the recipe without a trigger.
  REPEATMASS  the target pair from the mean read-gate mass over the positions whose token type occurs more than once in
              that sequence's input (token ids only).
  All learned arms paired on 2902-2911.
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with the transition. Failure classes v1 (KEY / POSITION /
STREAM-PARTIAL / OTHER / MERGED) and v2.
VALIDITY (fixed now): the perfect gate runs first; the learned arms are queued only if it binds both seeds within 43200.
Otherwise the screen is UNTESTED and no learned arm runs.
READINGS (fixed now; 10 paired seeds; exact one-sided McNemar printed for every comparison; "X only" = X bound routed
and the other not; "X within 1 pair of BASE" = (BASE only - X only) <= 1):
  "the labelled recipe itself fails here"   BASE < 7/10. Then no other reading.
  "the split is not needed here"            NOSPLIT within 1 pair of BASE. Then each trigger arm is printed 'T: moot (the
                                            split is not needed here); BASE only b, T only c; T only d vs NOSPLIT only e',
                                            with no other name. (Referee V's fix 3.)
  Per trigger arm T in (ALLMASS, REPEATMASS):
    "T: the trigger survives"               T within 1 pair of BASE AND T beats NOSPLIT by >= 4 vs 0 (T only >= 4 and
                                            NOSPLIT only = 0).
    "T: the trigger needs key positions"    BASE beats T by >= 4 vs 0 (BASE only >= 4 and T only = 0).
    "T: no reading applies"                 otherwise.
  Each printed reading carries its exact one-sided McNemar p.
CAVEAT (printed with every reading; Referee V's fix 1): the pool makes key types rare across the corpus (0.0105 per type
against 0.084 at pool size 4). It does not move the keys: they stay at 3j+1, are the most repeated types within a
sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. BASE's
key-position mask is the main line's, unchanged. A reading here is about rare key types on a layout with positional
keys, not about unmarked key positions.
OPERATING CHARACTERISTICS (Referee V's fix 4; V's screen_checks_s89_H_out.txt §5; exact, BASE 0.9, NOSPLIT 0.3, 10 pairs;
independent / half-shared / fully shared seed difficulty): P('survives') at a true trigger rate of 0.9, 0.8, 0.7, 0.5 =
0.66/0.80/0.94, 0.39/0.50/0.64, 0.20/0.26/0.29, 0.03/0.03/0.02; P('needs key positions') = 0.005/0.001/0, 0.05/0.03/0.01,
0.16/0.15/0.12, 0.47/0.55/0.62. 'Survives' does not distinguish a trigger as good as BASE from one 10 points worse; 'needs
key positions' detects a loss of about 40 points half the time; 4 vs 0 is one-sided p = 0.0625. 'No reading applies'
means ten pairs do not decide. P(BASE < 7/10 | true 0.9) = 0.013.
DESCRIPTIVE (no reading): per run the splits (update, probe accuracy, the pair copied), the three pairs' labels at each
firing (key_ok, all_ok, rep_ok, op_ok: c* holds >= 2 streams and c0 none on routing_k's stream -> channel map at VAL
positions; the stream labels come from the generator) and the map before each firing; firings labelled ok per arm;
how often the pairs coincide; printed with each trigger arm's reading: its firings, how many had a one-to-one map before
the fire (one_to_one_before) and how many were labelled ok (op_ok) (Referee V's fix 5); transitions; v1 / v2 classes; REPEATMASS's overlap counts; inherited statistics that index
key ids 8-11 only (key_sep, key_part) are descriptive and used by no outcome.
CHECKS (before any run): explore_common.repro_check; the child's unit checks (PoolTask(4)'s batches and attributes equal
TASK48's bit for bit on the training, evaluation and both probe generators; PoolTask(32)'s structure; repeat_mask reads
token ids only (one argument, invariant under a permutation of the vocabulary, equal to a Counter reference); the exact
copy is S61's copy_nonoise_op statement for statement and bit for bit; targets key and all equal S86's trigger row for
row on S48's harness and target repeat copies onto the repeat pair); at pool size 4 (asserted, this CPU): BASE|2600 and
ALLMASS|2600 equal S87a's records bit for bit through 9600 (curve, statistics, the trigger rows' common fields; both
fired at 4800), NOSPLIT|260 equals batch 16's LOCAL3_SLOW_D8_A|260 through 6000, ORACLE|2600 equals S87a's
ORACLE|2600 through 2400; at pool size 32 (asserted): BASE, ALLMASS and REPEATMASS with the threshold at 0 equal NOSPLIT
bit for bit through 4800 on seed 2902; at pool size 4 with the main line's split op (printed, not asserted: the record
is from a Xeon @ 2.80GHz): the main line's X record WIN3_SPLIT_D8|540 through 4800 (curve, statistics, gradient norms,
the trigger rows at 2400 and 4800); the record is read at the main-line commit that recorded it, 628eb7c (Referee V's fix
6); it printed True on curve, statistics, gradient norms and trigger rows through 4800 at 0dd88bf (fix 8).
STAGES: H_S89_STAGE=oracle runs ORACLE only; any other value queues the learned arms only after ORACLE bound 2/2.
REVISION (20 October 2026, after every run): Referee V's screen review (claude/review-V, 3c772c3 / dcd549d,
docs/review_2026-10/screen_review_s89_H.md, "fix first", text and the report path only) was written before the learned
arms ran and read after they had finished. Its fixes 1-6 (and the text of 7 and 8) are applied to this docstring and to
report(); this file is not in the child's code SHA, so no run, CHECK or record changes. The reading rules are unchanged
except fix 3's print of a moot reading, which does not arise in S89's outcome ("the labelled recipe itself fails here").
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
import explore_h_allmass2 as s87a

NAME = "H/s89"
CHILD = "explore_h_s89_child"
MAIN = True
IDEA = "the split's trigger at eight streams with keys drawn from a pool of 32: all-position and repeat-position targets"
SOURCE = "S86 + S87a (ALLMASS 24/30 vs BASE 26/30 on the main line's task)"
CHANGE = "the main line's task with a 32-key pool; ALLMASS / REPEATMASS / NOSPLIT against BASE"
PAIRING = "seeds 2902-2911, all learned arms paired; ORACLE 2900-2901"
LR = 1e-3
ITERS = 43200
SEEDS = tuple(range(2902, 2912))
ORACLE_SEEDS = (2900, 2901)
LEARNED = ("BASE", "ALLMASS", "REPEATMASS", "NOSPLIT")
TRIGGERED = ("BASE", "ALLMASS", "REPEATMASS")
ARMS = {"ORACLE": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceil8_D8, key pool 32",
                       sched="one Adam 1e-3")}
ARMS.update({a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a + ", key pool 32",
                     sched="LOCAL3 + SLOW" + ("" if a == "NOSPLIT" else " + exact-copy trigger " + a)) for a in LEARNED})
ALL_ARMS = dict(ARMS)
CUT = set()
MAINREF = "628eb7c"   # the main-line commit that recorded results/X/window_gate_results.json (WIN3_SPLIT_D8|540)
CAVEAT = ("the pool makes key types rare across the corpus (0.0105 per type against 0.084 at pool size 4); it does not move "
          "the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits "
          "next to its stream's CTX token, which the window gate reads. BASE's key-position mask is the main line's, unchanged. "
          "A reading here is about rare key types on a layout with positional keys, not about unmarked key positions.")
OC = ("operating characteristics (Referee V, exact, BASE 0.9, NOSPLIT 0.3, 10 pairs; independent / half-shared / fully shared "
      "seed difficulty): P('survives') at a true trigger rate of 0.9, 0.8, 0.7, 0.5 = 0.66/0.80/0.94, 0.39/0.50/0.64, "
      "0.20/0.26/0.29, 0.03/0.03/0.02; P('needs key positions') = 0.005/0.001/0, 0.05/0.03/0.01, 0.16/0.15/0.12, "
      "0.47/0.55/0.62; 4 vs 0 is one-sided p = 0.0625; 'no reading applies' means ten pairs do not decide; "
      "P(BASE < 7/10 | true 0.9) = 0.013.")
FIELDS = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
          "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
          "row_dist_after")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def oracle_state():
    """{seed: transition or None} of the stored ORACLE runs under this code SHA on this CPU, and whether both exist."""
    sha = c16.code_sha(CHILD, MAIN)[0]
    st = ec.load_store(NAME)
    got = {s: st["runs"].get(c16.key("ORACLE", s, sha, ec.cpu_model())) for s in ORACLE_SEEDS}
    got = {s: (r if r is not None and r.get("ok") else None) for s, r in got.items()}
    return {s: (None if r is None else r.get("transition")) for s, r in got.items()}, all(r is not None for r in got.values())


def configure():
    """The stage: ORACLE alone first (H_S89_STAGE=oracle); otherwise every arm, queued only after ORACLE bound both seeds
    under this code SHA on this CPU (the validity rule). ORACLE's runs are then cached and not rerun."""
    ARMS.clear()
    if os.environ.get("H_S89_STAGE") == "oracle":
        ARMS["ORACLE"] = ALL_ARMS["ORACLE"]
        return True
    ARMS.update(ALL_ARMS)
    tr, done = oracle_state()
    ok = done and all(t is not None for t in tr.values())
    print(f"  {NAME}: ORACLE {tr} (both stored: {done}) -> " + ("the learned arms are queued" if ok else
          "UNTESTED or not yet run: no learned arm is queued"), flush=True)
    return ok


def store_run(name, arm, seed):
    for k, r in ec.load_store(name)["runs"].items():
        a, s = k.split("|")[:2]
        if a == arm and int(s) == seed and r.get("ok"):
            return r
    return None


def rows_eq(r, rec, upto, fields=FIELDS):
    m = [{k: x.get(k) for k in fields if k in x and k in y}
         for x, y in zip([c for c in r["checks"] if c["step"] <= upto], [c for c in rec["checks"] if c["step"] <= upto])]
    n = [{k: y.get(k) for k in fields if k in x and k in y}
         for x, y in zip([c for c in r["checks"] if c["step"] <= upto], [c for c in rec["checks"] if c["step"] <= upto])]
    same_len = len([c for c in r["checks"] if c["step"] <= upto]) == len([c for c in rec["checks"] if c["step"] <= upto])
    return same_len and m == n


def check():
    s87 = {a: store_run(s87a.NAME, a, 2600) for a in ("BASE", "ALLMASS", "ORACLE")}
    ref260 = s58.b16_record("LOCAL3_SLOW_D8_A", 260)
    mx = json.loads(subprocess.run(["git", "-C", HERE, "show", f"{MAINREF}:results/X/window_gate_results.json"],
                                   capture_output=True, text=True, check=True).stdout)["runs"]["WIN3_SPLIT_D8|540"]
    jobs = [("BASE2600", dict(arm="BASE", seed=2600, iters=9600, K=4, total=ITERS)),
            ("ALLMASS2600", dict(arm="ALLMASS", seed=2600, iters=9600, K=4, total=ITERS)),
            ("NOSPLIT260", dict(arm="NOSPLIT", seed=260, iters=6000, K=4, total=ITERS)),
            ("ORACLE2600", dict(arm="ORACLE", seed=2600, iters=2400, K=4)),
            ("MAINSPLIT540", dict(arm="MAINSPLIT", seed=540, iters=4800, K=4, total=ITERS))]
    jobs += [(f"{a}2902", dict(arm=a, seed=2902, iters=4800, thr=0.0, total=ITERS)) for a in TRIGGERED]
    jobs += [("NOSPLIT2902", dict(arm="NOSPLIT", seed=2902, iters=4800, total=ITERS))]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = dict(ex.map(lambda j: (j[0], child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    for a in ("BASE", "ALLMASS"):
        r, rec = res[f"{a}2600"], s87[a]
        eq, c = s58.eq_upto(r, rec, 9600)
        rq = rows_eq(r, rec, 9600)
        fired = [x["step"] for x in r["checks"] if x["fired"]]
        rows.append((f"pool 4: {a}|2600 equals S87a's {a}|2600 ({rec['cpu']}) bit for bit through 9600 (curve {[x[:2] for x in c]}; "
                     f"statistics; trigger rows: {rq}; fired at {fired} as the record: "
                     f"{[x['step'] for x in rec['checks'] if x['fired'] and x['step'] <= 9600]})", eq and rq and fired == [4800]))
    eq, c = s58.eq_upto(res["NOSPLIT260"], ref260, 6000)
    rows.append((f"pool 4: NOSPLIT|260 equals batch 16's LOCAL3_SLOW_D8_A|260 ({ref260['cpu']}) bit for bit through 6000 "
                 f"(curve {[x[:2] for x in c]}; statistics)", eq and res["NOSPLIT260"]["splits"] == 0))
    o, ro = res["ORACLE2600"], s87["ORACLE"]
    oq = [x for x in o["curve"] if x[0] <= 2400] == [x for x in ro["curve"] if x[0] <= 2400] and \
        [s for s in o["stats"] if s["step"] != "end" and s["step"] <= 2400] == \
        [s for s in ro["stats"] if s["step"] != "end" and s["step"] <= 2400]
    rows.append((f"pool 4: ORACLE|2600 equals S87a's ORACLE|2600 through 2400 (curve {[x[:2] for x in o['curve']]}; statistics)", oq))
    ns = res["NOSPLIT2902"]
    for a in TRIGGERED:
        r = res[f"{a}2902"]
        eq = r["curve"] == ns["curve"] and r["stats"] == ns["stats"] and r["splits"] == 0
        rows.append((f"pool 32: {a}|2902 with the threshold at 0 equals NOSPLIT|2902 bit for bit through 4800 (curve "
                     f"{[x[:2] for x in r['curve']]}; statistics; checks at {[x['step'] for x in r['checks']]}, none fired)", eq))
    ov = res["REPEATMASS2902"]["overlap"]
    print(f"  (printed) REPEATMASS's probe (64 sequences): {ov['repeat']} of {ov['positions']} input positions repeat; key positions "
          f"in it {ov['key_in_repeat']} of {ov['key_positions']}; not key positions {ov['repeat_not_key']} (CTX {ov['repeat_ctx']}, "
          f"VAL {ov['repeat_val']} of {ov['all_val']}, query CTX {ov['repeat_q_ctx']}, query KEY {ov['repeat_q_key']})", flush=True)
    r = res["MAINSPLIT540"]
    cq = [x for x in r["curve"] if x[0] <= 4800] == [x for x in mx["curve"] if x[0] <= 4800]
    sq = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 4800] == \
        [s for s in mx["stats"] if s["step"] != "end" and s["step"] <= 4800]
    gq = {k: v for k, v in r["grad"].items() if int(k) <= 4800} == {k: v for k, v in mx["grad"].items() if int(k) <= 4800}
    tq = rows_eq(r, mx, 4800, FIELDS + ("noise_sd", "noise_seed", "state_zeroed"))
    print(f"  (printed, not asserted: CPU {mx['cpu']} vs this {ec.cpu_model()}) pool 4 with the main line's split op reproduces X's "
          f"WIN3_SPLIT_D8|540 through 4800: curve {cq} ({[x[:2] for x in r['curve']]} vs {[x[:2] for x in mx['curve'] if x[0] <= 4800]}); "
          f"statistics {sq}; gradient norms {gq}; trigger rows at 2400 and 4800 {tq} (fired at "
          f"{[x['step'] for x in r['checks'] if x['fired']]} vs {[x['step'] for x in mx['checks'] if x['fired'] and x['step'] <= 4800]})",
          flush=True)
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
               v1v2_differ=sorted(s for s, r in rs.items() if r.get("outcome_v2") != r.get("outcome")))
    if arm in TRIGGERED:
        out.update(ops=len(f), op_ok=sum(bool(c.get("op_ok")) for c in f), key_ok=sum(bool(c["key_ok"]) for c in f),
                   all_ok=sum(bool(c["all_ok"]) for c in f), rep_ok=sum(bool(c["rep_ok"]) for c in f),
                   same_key_all=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
                   same_key_rep=sum(c["key_cs"] == c["rep_cs"] and c["key_c0"] == c["rep_c0"] for c in f),
                   same_all_rep=sum(c["all_cs"] == c["rep_cs"] and c["all_c0"] == c["rep_c0"] for c in f),
                   one_to_one_before=sum(len(set(c["map_before"])) == 8 for c in f),
                   flat=sum(c["acc"] < 0.2 for c in f), fire_acc=sorted(round(c["acc"], 2) for c in f))
    return out


def report():
    me = sys.modules[__name__]
    ARMS.clear()
    ARMS.update(ALL_ARMS)
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    for a in LEARNED:
        res["arms"][a] = summarize(got[a], a)
    B = {a: {s: hc.br(r) for s, r in got[a].items()} for a in LEARNED}
    for x, y in (("ALLMASS", "BASE"), ("REPEATMASS", "BASE"), ("NOSPLIT", "BASE"), ("ALLMASS", "NOSPLIT"),
                 ("REPEATMASS", "NOSPLIT"), ("ALLMASS", "REPEATMASS")):
        res["cmp"][f"{x} vs {y}"] = hc.paired(B[x], B[y], SEEDS)
    rep = [r.get("overlap") for r in got["REPEATMASS"].values() if r.get("overlap")]
    res["overlap"] = rep[0] if rep else None
    complete = all(res["arms"][a]["complete"] for a in LEARNED) and len(orc) == len(ORACLE_SEEDS)
    A, C = res["arms"], res["cmp"]
    res["caveat"] = CAVEAT
    res["oc"] = OC
    res["trigger_lines"] = {t: (f"{t}: {A[t]['ops']} firings, {A[t]['one_to_one_before']} with a one-to-one map before the fire, "
                                f"{A[t]['op_ok']} labelled ok (op_ok)") for t in TRIGGERED if A[t].get("ops") is not None}
    ra = A["REPEATMASS"]
    if res["overlap"] and ra.get("ops"):
        share = 100 * res["overlap"]["repeat"] / res["overlap"]["positions"]
        res["repeat_note"] = ("REPEATMASS: as ALLMASS (same pair at every firing)" if ra["same_all_rep"] == ra["ops"] else
                              f"REPEATMASS (repeat set = {share:.1f}% of input positions; same pair as ALLMASS at "
                              f"{ra['same_all_rep']} of {ra['ops']} firings)")
    if len(orc) == len(ORACLE_SEEDS) and not valid:
        rd = {"S89": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)}); no learned arm runs"}
    elif not complete:
        rd = {"S89": "INCOMPLETE, no reading"}
    elif A["BASE"]["br"] < 7:
        rd = {"S89": f"the labelled recipe itself fails here (BASE {A['BASE']['br']}/10); no other reading"}
    else:
        within = lambda c: c["c"] - c["b"] <= 1                                        # noqa: E731  (x vs BASE)
        ns = C["NOSPLIT vs BASE"]
        moot = within(ns)
        rd = {"split": ("the split is not needed here" if moot else "the split is needed (no reading)")
              + f" (BASE only {ns['c']}, NOSPLIT only {ns['b']}; p(BASE>NOSPLIT) {ns['p_rev']:.3g})"}
        for t in ("ALLMASS", "REPEATMASS"):
            cb, cn = C[f"{t} vs BASE"], C[f"{t} vs NOSPLIT"]
            if within(cb) and cn["b"] >= 4 and cn["c"] == 0:
                s = f"{t}: the trigger survives"
            elif cb["c"] >= 4 and cb["b"] == 0:
                s = f"{t}: the trigger needs key positions"
            else:
                s = f"{t}: no reading applies"
            counts = (f"BASE only {cb['c']}, {t} only {cb['b']} (p(BASE>{t}) {cb['p_rev']:.3g}); {t} only {cn['b']} vs NOSPLIT "
                      f"only {cn['c']} (p({t}>NOSPLIT) {cn['p']:.3g})")
            rd[t] = f"{t}: moot (the split is not needed here); {counts}" if moot else f"{s} ({counts})"
            if t == "REPEATMASS" and res.get("repeat_note"):
                rd[t] = res["repeat_note"] + ": " + rd[t]
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got


def per_seed_log(got):
    for a in TRIGGERED + ("NOSPLIT",):
        for s, r in sorted(got.get(a, {}).items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            print(f"    {a:<10} {s}: {r.get('outcome')} (transition {r.get('transition')}); " + "; ".join(
                f"{c['step']} acc {c['acc']:.2f} {c['op_cs']}->{c['op_c0']} op_ok {c['op_ok']} (key {c['key_cs']}->{c['key_c0']} "
                f"{c['key_ok']}, all {c['all_cs']}->{c['all_c0']} {c['all_ok']}, rep {c['rep_cs']}->{c['rep_c0']} {c['rep_ok']}) "
                f"map before {c['map_before']}" for c in f), flush=True)
