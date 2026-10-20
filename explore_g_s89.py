#!/usr/bin/env python
"""
explore_g_s89.py — EXPLORATORY, not a result. Session G, screen S89 (round 6): does S87's frequency hinge do anything on a
task where keys are not a frequent class, and does a within-sequence repeat hinge replace the key hinge? Delta memory is
the primary memory. Everything below was fixed and committed before any S89 run (the first version at 7ab98c1 had readings
and no code; Referee V's review, claude/review-V docs/review_2026-10/screen_review_s89_G.md at dcd549d, "fix first";
this version answers its fixes 1-10 and the user's re-plan of 20 October).

BACKGROUND. S87 (RandHeaderTask-lite): Hebbian REG3_FREQHINGE 17/20 (17 vs 0 against BASELINE, two pairs short of
REG3_KEYHINGE 20/20); delta (2720-2729) REG3_FREQHINGE_D 10/10, REG3_KEYHINGE_D 10/10, BASELINE_D 0/10, REG3_D 0/10. On
lite the largest-ratio cut selected the 2 CTX and the 4 key types. Round 6: F's RandHeaderTask-pool (each sequence's 4 keys
renamed to 4 of 32 types) and RandHeaderTask-allpool (the 2 CTX tokens also renamed to 2 of 16).
WHY DELTA. F's pilots (claude/explore-F report_9.md, 52e2ea7; Intel Xeon @ 2.80GHz, counts only here): the Hebbian perfect
gate binds neither pool task at 24000 or 48000, under SLOW or S70's Adam (0/2 each), while the delta perfect gate (beta 1,
L2 keys) binds both 2/2 at 2400-3600. Both tasks read "untested" for Hebbian and valid for delta. So every arm here is
on delta memory. NO HEBBIAN LEARNED ARM RUNS: F's convolution pilot (explore_f_s89_post at 3dd5b74, results 73035d3; the
Hebbian perfect gate + conv 'layer', seeds 2800-2801) bound POOL 2/2 and ALLPOOL 1/2, which F's own pre-fixed rule reads
"partly" (4/4 "restores"). That is a Hebbian configuration valid on POOL only, and with a convolution this branch's
register models do not have (RegBDH's layer loop has no conv path; building and CHECKing one is a separate screen). It
is not run here; the report says so.

TASKS: explore_f_tasks.py vendored byte for byte from claude/explore-F at b0c6332 (blob e24d5e10815eb1234d1947aa6b74b128d2696882,
the later hash F named after its own fixes, published before any S89 run; asserted; its first 35,605 bytes equal 50dc8e4's
blob ffe249bb... and its first 14,101 a199d5e's lite module, both asserted): "pool" = ft.RandHeaderTaskPool(), "allpool" =
ft.RandHeaderTaskAllPool() (vocabularies 50 and 64; L 23, qpos 21, as lite). Lite enters only through S87's records.
THE ROUTING CAP ON ALLPOOL (F's b0c6332 comment, CHECKed by ft.check_inv_stats): no fixed map from CTX type to channel
separates a sequence's two contexts in more than 64/120 = 0.533 of sequences; a gate reading token identity routes the
rest only by comparing the two contexts within the sequence (the register can carry the first one). task.perfect_gate is
handed that comparison. So a valid perfect gate on ALLPOOL says the memory can use a relative routing, not that a gate
under test can express one, and on ALLPOOL any null ("fails here", "does nothing") does not separate "keys are not
frequent" from "contexts have no fixed identity". The report states this beside every ALLPOOL reading.
MODEL, RECIPE, HINGES, TASK-AWARE STATISTICS, RECORDS: explore_g_s89_child (S87's model: k = 2, no conv, 24000 updates,
REG3 at all three layers, delta memory beta 1 / L2 keys / tied write / decay 0.95; SLOW for every arm, the perfect gate
included; every hinge LAM 1.0 x mean over layers of relu(eta2 by its ids at its positions - 0.2), updates 1-2400 only, with
a switch that makes it REG3_D bit for bit when off).
  REG3_KEYHINGE_D    the key term with the true key positions and identities (task.body_positions, task.key_index over the
                     32 key types): the labelled ceiling.
  REG3_FREQHINGE_D   S87's frequency hinge exactly, the largest-ratio cut recomputed on the task's own sample (F's table at
                     50dc8e4 agrees). WHAT IT IS APPLIED TO: on -pool CTX 0.114 per type, VAL 0.023, KEY 0.013: the cut
                     selects the 2 CTX types only, so it is a CTX HINGE (5 CTX positions per sequence, none of the key
                     term's; grouping by CTX id is grouping by stream there). On -allpool CTX 0.0142 per type (5 of 22
                     tokens over 16 types), VAL 0.0227, KEY 0.0128: the largest ratio (1.60) cuts after the 16 VALUE types,
                     so it is a VALUE HINGE (8 VAL positions per sequence, none of the key term's). (The 7ab98c1 docstring's
                     "CTX 0.028" and "16 CTX and 16 VAL types" were wrong.)
  REG3_REPEATHINGE_D the positions of token types occurring more than once in the input sequence, grouped by token id. On
                     both tasks, position by position the same as on lite: 5 CTX + 9 KEY + about 2.63 VAL positions per
                     sequence (16.6), covering the key term's 8 (48% of what it selects). It is not "the keys and nothing
                     else": it is S87's lite frequency hinge's positions plus the chance value repeats, grouped by pool ids.
ARMS per task, seeds 2842-2861 (20, paired across arms and across the two tasks): BASELINE_D (LOCAL3 + SLOW), REG3_D,
  REG3_KEYHINGE_D, REG3_FREQHINGE_D, REG3_REPEATHINGE_D. REG3_D is descriptive (the hinge's contribution: KH vs REG3_D
  printed). VALIDITY: the delta perfect gate ORACLE_D (this branch's code path, SLOW) on 2840-2841 per task, run first; a
  task's learned arms are queued only if it BINDS 2/2; else the task is UNTESTED here (not a negative) and nothing else runs
  on it. The decision is written to explore_out/G/s89_validity.json before any learned arm is queued. 204 runs at most.
  Seeds 2862-2899 are not used.
LITE COMPARISON (descriptive, counts only, other seeds): S87's delta references on 2720-2729, read from s87_results.json.
OUTCOME: ROUTED-BOUND = explore_g_rhl_parent.routed_bound (BOUND and F's ROUTED*@end: end margin >= 0.9 and end eta^2 by
stream at KEY > 0.9). On POOL the pooled eta^2 (explore_f_tasks.pool_routing_stats); on ALLPOOL the within-sequence eta^2
(pool_routing_stats_inv: the pooled one reads the first-appearance labelling, ~0 for a gate that separates every sequence
but not with the same channel for the first context), applied as routed_bound(inv_view(record)); CHECKed equal to F's
pool_outcome(rec, invariant) on a grid; the other form printed beside it. The margin is within-sequence in both. paired,
beats and not_worse_1 from explore_g_rhl_parent (imported, not re-implemented). Exact one-sided McNemar on seed pairs (b = the first arm only);
"discovered" = beats BASELINE_D with p < 0.05 (this is the word's meaning in S82-S89; explore_common.discovered's
"bound and VAL cos < 0.5" is printed beside it as DISCOVERED, as before); "p <= 0.0625" = at least as strong as 4 vs 0.
READINGS (fixed now; per task; computed by readings() below; n = the paired seeds):
  0. UNTESTED               if the perfect gate did not bind 2/2, or a run did not complete (a failed run is retried by
                            the harness, up to 3 attempts; if none completes, the task's readings are UNTESTED).
  1. "the task does not need the hinge"   BASELINE_D ROUTED-BOUND on >= n - 4 of the n seeds (16 of 20: REG3_KEYHINGE_D
                            could then win at most 4 vs 0, p 0.0625, and could not be discovered). Then no other reading.
  2. "the labelled hinge itself fails here"   REG3_KEYHINGE_D not discovered (and BASELINE_D <= n - 5). Then no other
                            reading on that task; the report says what that means for the register spec.
  3. Else each of REG3_FREQHINGE_D and REG3_REPEATHINGE_D gets exactly one of four names (KH = REG3_KEYHINGE_D; "KH only",
     "arm only" on the paired seeds):
       FREQ:   "the frequency hinge survives"                  discovered and (KH only) - (FREQ only) <= 1
               "the frequency hinge is weaker than the key hinge"  discovered, (KH only) - (FREQ only) >= 2, not "needs"
               "the frequency hinge needs frequent keys" [-allpool: "needs the key positions", since the cut there
                                                                selects values]    p(KH > FREQ) <= 0.0625
               "the frequency hinge does nothing here"         not discovered and not "needs"
       REPEAT: "the repeat hinge replaces the key hinge" / "is weaker than the key hinge" / "needs key identities" /
               "does nothing here", with the same four conditions.
     The four are exclusive and cover every outcome (a CHECK enumerates every per-seed pattern count at n = 10 and 20).
     And, for these two arms only: "clamps"   (arm only) - (BASELINE_D only) <= 1 AND the arm's median commitment at 2400
                            (gate_stats_k's spread) below BASELINE_D's (printed for every arm; it can accompany a name).
  ACROSS TASKS (descriptive, not a reading): the ROUTED-BOUND drop of BASELINE_D, REG3_D, REG3_KEYHINGE_D and
  REG3_REPEATHINGE_D from -pool to -allpool on the same seed numbers (pool only, allpool only, one-sided McNemar);
  REG3_FREQHINGE_D is NOT compared across tasks (a CTX hinge on -pool, a value hinge on -allpool).
OPERATING CHARACTERISTICS (V's screen_checks_s89_G_out.txt §5, exact): at 20 paired seeds with BASELINE ~0 and KH ~0.9,
  "survives" fires with probability ~0.8-1.0 for an arm at the ceiling, ~0.4 at 0.8, 0.07-0.15 at 0.7, < 0.01 at 0.5: it
  separates the ceiling from half of it, not from 0.8. "discovered" is near certain for any arm binding 5 or more of 20.
  Between 0.7 and 0.8 the most likely name is "is weaker than the key hinge" (formerly no name).
ALSO REPORTED: the hinges' selected positions per role and the frequency cuts; firings and values; beta^R by role and step;
the margin per layer; the contingency tables; the probe; the plateaus by (stream, block, pair); the commitment by step.
CHECKS (asserted before any run; explore_g_s89_child unless stated): X's arm A bit for bit (explore_common.repro_check);
  - the vendored module: its blob equals F's at b0c6332; its first 35,605 bytes equal 50dc8e4's and 14,101 a199d5e's; F's
    check_pool_task and check_inv_stats on both tasks; routed_bound (pooled / inv_view) = F's pool_outcome on a grid;
  - token ids only: frequent() equals F's largest_ratio_cut on ft.freq_sample(task) and raises when handed
    stream_labels(x), the key ids, a float tensor, a narrower tensor or another sample; one parameter each; repeat_mask by
    hand; the repeat positions identical on the three tasks; what each hinge selects per role (pool: CTX only; allpool:
    VAL only; asserted);
  - the task-aware key term equals explore_g_devices_child.key_eta bit for bit on lite; on the new tasks its positions are
    task.key_mask's 8 per sequence; the task-aware statistics equal the lite functions on lite;
  - the perfect gate: task.perfect_gate (k = 2) at every layer, equal to explore_g_rhl_child's ORACLE_ALL_D model on lite
    and pool (parameters, logits, gates); the UNTRAINED perfect gate reads ROUTED* on each task (eta^2 by stream at KEY 1,
    margin >= 0.9);
  - S87's records reproduce bit for bit through 2400 through this screen's run() on lite (CPU-tolerant lookup, printed):
    REG3_FREQHINGE_D, REG3_KEYHINGE_D, BASELINE_D, REG3_D at 2720 and REG3_FREQHINGE, REG3_KEYHINGE, BASELINE at 2700;
  - per new task: each hinge with its switch off equals REG3_D through 2400 bit for bit; every arm runs end to end (20
    updates; the statistics and post-run records); each hinge's eta^2 gradient reaches the gate's parameters (and the
    embedding through its input), not the stack or the memory; the schedule over training forwards; the lrs (the perfect
    gate's gate group empty);
  - (parent) report's readings on S87's records re-labelled as one task: Hebbian (BASELINE, REG3_KEYHINGE, REG3_FREQHINGE,
    2700-2719) must read "the frequency hinge is weaker than the key hinge" and not "needs frequent keys"; delta (2720-2729)
    must read "the frequency hinge survives"; the four names exclusive and exhaustive by enumeration.
  F's BASELINE -pool 2802 cannot be reproduced: F ran no baseline (its phase 2 queued none).
BUDGET: about 200 runs, 2.3 min (bound) to 8.5 min (unbound) each at S87's delta rate: 2-7 h on 4 workers.

Run:  python explore_g_s89.py [--report-only]
"""

import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_g_common as gc
import explore_g_rhl_parent as hp

NAME = "s89"
CHILD = "explore_g_s89_child"
TASK_NAMES = ("pool", "allpool")
S2, S20 = (2840, 2841), tuple(range(2842, 2862))
LEARNED = ("BASELINE_D", "REG3_D", "REG3_KEYHINGE_D", "REG3_FREQHINGE_D", "REG3_REPEATHINGE_D")
ROLE = dict(B="BASELINE_D", R="REG3_D", KH="REG3_KEYHINGE_D", FQ="REG3_FREQHINGE_D", RP="REG3_REPEATHINGE_D")
VALID = os.path.join(gc.G_DIR, "s89_validity.json")
P4 = 0.0625
NAMES = {
    "FQ": ("the frequency hinge survives", "the frequency hinge is weaker than the key hinge",
           "the frequency hinge needs frequent keys", "the frequency hinge does nothing here"),
    "RP": ("the repeat hinge replaces the key hinge", "the repeat hinge is weaker than the key hinge",
           "the repeat hinge needs key identities", "the repeat hinge does nothing here"),
}
NEEDS_ALLPOOL = "the frequency hinge needs the key positions"
S87_DELTA = dict(B=("BASELINE_D", tuple(range(2720, 2730))), R=("REG3_D", tuple(range(2720, 2730))),
                 KH=("REG3_KEYHINGE_D", tuple(range(2720, 2730))), FQ=("REG3_FREQHINGE_D", tuple(range(2720, 2730))))
S87_HEBB = dict(B=("BASELINE", tuple(range(2700, 2720))), KH=("REG3_KEYHINGE", tuple(range(2700, 2720))),
                FQ=("REG3_FREQHINGE", tuple(range(2700, 2720))))


def inv_view(r):
    """The record with eta^2 by stream at KEY replaced by its within-sequence form (ALLPOOL's outcome); the pooled one kept."""
    e = dict(r["end"], eta_key_by_stream=r["end"].get("eta_key_by_stream_inv"),
             eta_key_by_stream_pooled=r["end"].get("eta_key_by_stream"))
    return dict(r, end=e)


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _arms():
    a = {f"{t}.ORACLE_D": dict(iters=24000, seeds=S2, prio=2) for t in TASK_NAMES}
    v = _load(VALID)
    if v:
        for t in TASK_NAMES:
            if v[t]["valid"]:
                for k in LEARNED:
                    a[f"{t}.{k}"] = dict(iters=24000, seeds=S20, prio=1)
    return a


ARMS = _arms()
line = hp.line
rb = hp.routed_bound


def commit(r, step=2400):
    for s in r["stats"]:
        if s["step"] == step:
            return s.get("spread")
    return None


def med_commit(rr):
    v = [commit(r) for r in rr.values() if commit(r) is not None]
    return statistics.median(v) if v else float("nan")


# ── the readings (a pure function of the records) ────────────────────────────
def arm_name(kh, arm, base, role, task):
    """Exactly one of the four names for a learned hinge arm (role FQ or RP), with its numbers."""
    disc, (db, dc_, dp, dn) = hp.beats(arm, base, rb)
    b, c, p, n = hp.paired(kh, arm, rb)
    names = NAMES[role]
    if p <= P4:
        name = NEEDS_ALLPOOL if (role == "FQ" and task == "allpool") else names[2]
    elif disc and b - c <= 1:
        name = names[0]
    elif disc:
        name = names[1]
    else:
        name = names[3]
    return name, (f"discovered {'yes' if disc else 'no'} (arm only {db}, BASELINE only {dc_}, p {dp:.3g}); vs KH: KH only "
                  f"{b}, arm only {c}, p(KH >) {p:.3g}, n {n}")


def clamps(arm, base):
    b, c, _, _ = hp.paired(arm, base, rb)
    ca, cb = med_commit(arm), med_commit(base)
    return (b - c <= 1 and ca < cb), f"arm only {b}, BASELINE only {c}; commitment at 2400 {ca:.4f} vs BASELINE {cb:.4f}"


def readings(R, seeds, task, valid=True, valid_note="", label=None):
    """R: role -> {seed: record} (roles B, KH required; R, FQ, RP optional); label: role -> printed arm name (default
    ROLE). Returns an ordered dict of readings."""
    label = ROLE if label is None else label
    rd = {"validity": valid_note}
    n = len(seeds)
    missing = {label.get(k, k): sorted(set(seeds) - set(v)) for k, v in R.items() if set(seeds) - set(v)}
    if not valid:
        rd["first"] = "UNTESTED (the perfect gate did not bind 2/2)"
        return rd
    if missing:
        rd["first"] = f"UNTESTED (runs that did not complete: {missing})"
        return rd
    B, KH = R["B"], R["KH"]
    nb = sum(rb(r) for r in B.values())
    kh_disc, t = hp.beats(KH, B, rb)
    rd["counts"] = ", ".join(f"{label.get(k, k)} {sum(rb(r) for r in v.values())}/{n}" for k, v in R.items())
    rd["REG3_KEYHINGE vs BASELINE"] = f"KH only {t[0]}, BASELINE only {t[1]}, p {t[2]:.3g}"
    if "R" in R:
        b, c, p, _ = hp.paired(KH, R["R"], rb)
        rd["REG3_KEYHINGE vs REG3 (descriptive: the hinge's contribution)"] = f"KH only {b}, REG3 only {c}, p {p:.3g}"
    if nb >= n - 4:
        rd["first"] = f"the task does not need the hinge (BASELINE {nb}/{n} >= n - 4)"
        return rd
    if not kh_disc:
        rd["first"] = "the labelled hinge itself fails here"
        return rd
    rd["first"] = "the labelled hinge is discovered; per-arm names below"
    for role in ("FQ", "RP"):
        if role in R:
            name, why = arm_name(KH, R[role], B, role, task)
            cl, cwhy = clamps(R[role], B)
            rd[label[role]] = f"{name.upper()} ({why}); clamps: {'APPLIES' if cl else 'does not apply'} ({cwhy})"
    return rd


# ── the reading CHECKs (parent; no torch) ────────────────────────────────────
def recorded(store, arm, seeds):
    """{seed: record}: each seed's single ok record of arm under the store's SHA, whatever its CPU (prefix lookup)."""
    st = gc.load(store)
    sha = st["meta"]["sha"][:12]
    out = {}
    for s in seeds:
        hits = [v for k, v in st["runs"].items() if k.startswith(f"{arm}|{s}|{sha}|") and v.get("ok")]
        assert len(hits) <= 1, (arm, s)
        if hits:
            out[s] = hits[0]
    return out


def _fake(outcomes):
    """records with the given ROUTED-BOUND outcomes (True / False) per seed index."""
    good = dict(transition=1200, end=dict(margin=1.0, eta_key_by_stream=1.0), stats=[dict(step=2400, spread=0.5)])
    bad = dict(transition=None, end=dict(margin=0.0, eta_key_by_stream=0.0), stats=[dict(step=2400, spread=0.5)])
    return {i: (good if o else bad) for i, o in enumerate(outcomes)}


def check_readings():
    rows = []
    for tag, spec, want in (("Hebbian 2700-2719", S87_HEBB, NAMES["FQ"][1]), ("delta 2720-2729", S87_DELTA, NAMES["FQ"][0])):
        R = {k: recorded("s87", a, s) for k, (a, s) in spec.items()}
        seeds = next(iter(spec.values()))[1]
        label = {k: a for k, (a, _) in spec.items()}
        rd = readings(R, seeds, "lite", valid_note="S87's validity: ORACLE_ALL 2/2 (Hebbian; S87 ran no delta perfect gate)",
                      label=label)
        print(f"  S87's records re-labelled as one task ({tag}):\n" + "\n".join(f"      {k}: {v}" for k, v in rd.items()),
              flush=True)
        got = rd.get(label["FQ"], "")
        rows.append((f"S87 {tag} re-labelled: REG3_FREQHINGE reads '{want}' and not 'needs frequent keys' "
                     f"(got: {got.split(' (')[0]})", got.startswith(want.upper()) and "NEEDS" not in got.split(" (")[0]))
    # the four names are exclusive and exhaustive: the docstring's conditions, written independently, on every count
    # vector of the 8 per-seed patterns (KH, arm, BASELINE) at n = 10 and n = 20
    mc = lru_cache(None)(gc.mcnemar_greater)
    for n in (10, 20):
        bad = total = 0
        for comp in itertools.combinations_with_replacement(range(8), n):
            cnt = [0] * 8
            for i in comp:
                cnt[i] += 1
            # pattern bits: 4 = KH, 2 = arm, 1 = BASELINE
            kh_only = sum(cnt[i] for i in range(8) if i & 4 and not i & 2)
            arm_only = sum(cnt[i] for i in range(8) if i & 2 and not i & 4)
            a_nb = sum(cnt[i] for i in range(8) if i & 2 and not i & 1)
            b_na = sum(cnt[i] for i in range(8) if i & 1 and not i & 2)
            disc = mc(a_nb, b_na) < 0.05
            needs = mc(kh_only, arm_only) <= P4
            preds = [disc and kh_only - arm_only <= 1, disc and kh_only - arm_only >= 2 and not needs, needs,
                     not disc and not needs]
            total += 1
            bad += sum(preds) != 1
        rows.append((f"the four names are exclusive and exhaustive over all {total} count vectors at n = {n} "
                     f"({bad} with not exactly one)", bad == 0))
    # and readings() agrees with them on a few hand cases
    s = list(range(20))
    case = readings(dict(B=_fake([0] * 20), KH=_fake([1] * 20), FQ=_fake([1] * 17 + [0] * 3)), s, "pool")
    rows.append((f"hand case KH 20, FQ 17 nested, B 0 -> {case['REG3_FREQHINGE_D'].split(' (')[0]}",
                 case["REG3_FREQHINGE_D"].startswith(NAMES["FQ"][1].upper())))
    case = readings(dict(B=_fake([0] * 20), KH=_fake([1] * 20), FQ=_fake([1] * 10 + [0] * 10)), s, "allpool")
    rows.append((f"hand case KH 20, FQ 10, B 0 on allpool -> {case['REG3_FREQHINGE_D'].split(' (')[0]}",
                 case["REG3_FREQHINGE_D"].startswith(NEEDS_ALLPOOL.upper())))
    case = readings(dict(B=_fake([1] * 16 + [0] * 4), KH=_fake([1] * 20)), s, "pool")
    rows.append((f"hand case B 16/20 -> {case['first']}", case["first"].startswith("the task does not need the hinge")))
    case = readings(dict(B=_fake([1] * 15 + [0] * 5), KH=_fake([1] * 18 + [0] * 2)), s, "pool")
    rows.append((f"hand case B 15/20, KH 18 (3 vs 0) -> {case['first']}", case["first"] == "the labelled hinge itself fails here"))
    case = readings(dict(B=_fake([0] * 20), KH=_fake([1] * 20)), s[:19] + [99], "pool")
    rows.append((f"hand case: a missing seed -> {case['first'][:40]}", case["first"].startswith("UNTESTED")))
    return rows


# ── report ───────────────────────────────────────────────────────────────────
def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    other = {}
    for a in list(R):
        other[a] = sum(rb(inv_view(r) if a.startswith("pool.") else r) for r in R[a].values())
        if a.startswith("allpool."):
            R[a] = {s: inv_view(r) for s, r in R[a].items()}
    print("OUTCOME FORM: pool pooled eta^2, allpool within-sequence eta^2. ROUTED-BOUND under the other form "
          "(pool within-sequence, allpool pooled), descriptive: " + ", ".join(f"{a} {k}/{len(R[a])}" for a, k in other.items()))
    v = _load(VALID) or {}
    order = [a for t in TASK_NAMES for a in [f"{t}.ORACLE_D"] + [f"{t}.{k}" for k in LEARNED] if a in ARMS]
    print(hp.count_table(R, order))
    out = {}
    for t in TASK_NAMES:
        o = R.get(f"{t}.ORACLE_D", {})
        nb = sum(hp.bound(r) for r in o.values())
        note = f"ORACLE_D (delta perfect gate, SLOW) on 2840-2841 bound {nb}/{len(o)}: " + ", ".join(
            f"{s} {'bound @' + str(r['transition']) if hp.bound(r) else 'unbound'}" for s, r in sorted(o.items()))
        valid = bool(v.get(t, {}).get("valid"))
        RR = {k: R[f"{t}.{a}"] for k, a in ROLE.items() if f"{t}.{a}" in R}
        out[t] = readings(RR, S20, t, valid=valid, valid_note=note) if valid else dict(validity=note, first="UNTESTED")
        if t == "allpool":
            out[t]["caveat"] = ("the routing cap: a fixed CTX-type map separates at most 0.533 of sequences; a null here does "
                                "not separate 'keys are not frequent' from 'contexts have no fixed identity'")
        out[t]["commitment at 2400 (median spread)"] = ", ".join(f"{a} {med_commit(R[f'{t}.{a}']):.4f}" for a in LEARNED
                                                                 if R.get(f"{t}.{a}"))
    across = {}
    for a in ("BASELINE_D", "REG3_D", "REG3_KEYHINGE_D", "REG3_REPEATHINGE_D"):
        rp, ra = R.get(f"pool.{a}"), R.get(f"allpool.{a}")
        if rp and ra:
            b, c, p, n = hp.paired(rp, ra, rb)
            across[a] = (f"pool {sum(rb(r) for r in rp.values())}/{len(rp)}, allpool {sum(rb(r) for r in ra.values())}/{len(ra)}; "
                         f"pool only {b}, allpool only {c}, p(pool >) {p:.3g}, n {n}")
    lite = {}
    for k, (a, s) in S87_DELTA.items():
        rr = recorded("s87", a, s)
        k_ = sum(rb(r) for r in rr.values())
        row = [f"lite (S87, 2720-2729) {gc.cnt(k_, len(rr))}"]
        for t in TASK_NAMES:
            rt = R.get(f"{t}.{a}")
            if rt:
                kt = sum(rb(r) for r in rt.values())
                row.append(f"{t} {gc.cnt(kt, len(rt))} (Fisher p(lite >) {gc.fisher_greater(k_, len(rr), kt, len(rt)):.3g})")
        lite[a] = "; ".join(row)
    print("\nREADINGS:")
    for t in TASK_NAMES:
        print(f"  {t}:\n" + "\n".join(f"      {k}: {x}" for k, x in out[t].items()))
    print("  ACROSS TASKS (descriptive; REG3_FREQHINGE_D not compared):\n" + "\n".join(f"      {k}: {x}" for k, x in across.items()))
    print("  LITE COMPARISON (descriptive, counts only, other seeds):\n" + "\n".join(f"      {k}: {x}" for k, x in lite.items()))
    for a in order:
        tv = [r["term"] for r in R.get(a, {}).values() if r.get("term")]
        if tv:
            f = [x["fired"] for x in tv]
            vals = {u: statistics.median(statistics.fmean(x["value_at"][u]) for x in tv if u in x["value_at"])
                    for u in ("1", "300", "600", "1200", "2400") if any(u in x["value_at"] for x in tv)}
            print(f"  {a}: firings median {statistics.median(f)} ({min(f)}-{max(f)}); value (mean over layers, median) "
                  + ", ".join(f"{u} {x:.4f}" for u, x in vals.items()))
    for a in order:
        if "ORACLE" in a or not R.get(a):
            continue
        bys = {st_: statistics.median([commit(r, st_) for r in R[a].values() if commit(r, st_) is not None])
               for st_ in (0, 1200, 2400, 3600) if any(commit(r, st_) is not None for r in R[a].values())}
        print(f"  {a}: commitment (spread) by step " + ", ".join(f"{k} {x:.4f}" for k, x in bys.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, order))
    print("\nper run:\n" + hp.per_run(R, order))
    with open(os.path.join(gc.G_DIR, "s89_summary.json"), "w") as fh:
        json.dump(dict(readings=out, across=across, lite=lite), fh, indent=1)


# ── main ─────────────────────────────────────────────────────────────────────
def run_checks():
    jobs = [("checks", {}), ("lr_rows", {})]
    jobs += [("checks_repro", dict(arm=a, seed=s)) for a, s in (
        ("REG3_FREQHINGE_D", 2720), ("REG3_KEYHINGE_D", 2720), ("BASELINE_D", 2720), ("REG3_D", 2720),
        ("REG3_FREQHINGE", 2700), ("REG3_KEYHINGE", 2700), ("BASELINE", 2700))]
    jobs += [(f, dict(task=t)) for t in TASK_NAMES for f in ("checks_off", "checks_on")]
    with ThreadPoolExecutor(gc.WORKERS) as ex:
        res = list(ex.map(lambda j: gc.run_child(CHILD, j[0], j[1]), jobs))
    return [r for rr in res for r in rr]


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S89")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = [list(x) for x in check_readings()] + run_checks()
        for nm, ok in rows:
            print(f"  CHECK S89: {nm}: {'ok' if ok else 'FAIL'}", flush=True)
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s89_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        assert all(ok for _, ok in rows), "a CHECK failed"
        gc.run_screen(me)                                   # phase 1: the perfect gate per task
        if _load(VALID) is None:
            R = {a: gc.runs(me, a) for a in ARMS}
            v = {t: dict(bound=sum(hp.bound(r) for r in R[f"{t}.ORACLE_D"].values()), n=len(R[f"{t}.ORACLE_D"]))
                 for t in TASK_NAMES}
            for t in TASK_NAMES:
                v[t]["valid"] = v[t]["bound"] == 2 and v[t]["n"] == 2
            v["time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            json.dump(v, open(VALID, "w"), indent=1)
            print(f"  S89 validity (the delta perfect gate 2/2 per task): {v}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)                                   # phase 2: the learned arms on the valid tasks
        print(f"  S89 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
