#!/usr/bin/env python
"""
explore_g_s89h.py — EXPLORATORY, not a result. Session G, screen S89H (round 6, the Hebbian half of S89): does the repeat
hinge's S89 result hold on Hebbian memory, where S82 and S87 were? Everything below was fixed and committed before any
S89H run.

WHY, AND THE CONFIGURATION. S89 (report 8) ran on delta memory only, because the Hebbian perfect gate binds neither pool
task without a convolution (F's report 9). F's conv pilots (claude/explore-F explore_f_s89_post, 3dd5b74; results 73035d3
and 129215e) bound the Hebbian perfect gate + conv 'layer' on POOL 2/2 at 24000 and on ALLPOOL 2/2 at 48000 (1/2 at
24000); F's docstring records the user's decision that this conv configuration is round 6's valid Hebbian configuration
("ALLPOOL: the conv configuration needs the longer budget"). The user's instruction of 21 October to this session: "the
Hebbian-conv half still stands as asked (both tasks; ALLPOOL at 48,000) ... Same seeds, readings fixed before any run, V's
two caveats printed". V's two caveats are those of claude/review-V docs/review_2026-10/screen_review_s89_G_v2.md (c42ace8):
(1) REG3 must be read (a hinge is named only against BASELINE and the key hinge, so a hinge that adds nothing to a register
that binds without it would read "replaces"); (2) on POOL the frequency hinge penalises the stream (its cut is the 2 CTX
types, and CTX id is the stream there). Both are built into the readings below (2b and the POOL frequency rule).
MODEL, RECIPE, HINGES: explore_g_s89h_child. Hebbian memory (decay 0.95) + test_short_conv.LAYER (conv 'layer', K = 4, on
each layer's memory input; identity at init; trained in the rest group), k = 2, SLOW for every arm. The arms are S89's,
with _H for this configuration: BASELINE_H (LOCAL3 + conv), REG3_H, REG3_KEYHINGE_H, REG3_FREQHINGE_H, REG3_REPEATHINGE_H
(the hinges are explore_g_s89_child's, unchanged: on POOL the frequency hinge is a CTX hinge, 5 CTX positions; on ALLPOOL a
value hinge, 8 VAL positions; the repeat hinge 5 CTX + 9 KEY + ~2.6 VAL positions on both); PERFECT_H the perfect gate +
conv (validity). The tasks are S89's (F's b0c6332 module, vendored and asserted by S89's CHECKs).
BUDGET: POOL 24000 updates; ALLPOOL 48000 (every arm, the perfect gate included). ERRATUM (22 October; Referee V's
check_G_report9.md): ALLPOOL RAN 24000. ITERS below never reached the child (explore_g_common.run_screen sent only the
arm and the seed; the child's default is 24000), so every S89H record, ALLPOOL included, carries iters 24000. The
perfect gate bound 2/2 within 24000, so the ALLPOOL null stands at that budget. Fixed in explore_g_common.py on
22 October (the payload carries the arm's iters). Evaluation every 1200, early stop after
3 evaluations >= 0.95.
SEEDS (S89's, as asked): PERFECT_H on 2840-2841 per task, run first; the five arms on 2842-2861 per task (20, paired
across arms, across the two tasks and with S89's delta arms: the same batches; the same initial parameters wherever the
models share them). 204 runs.
VALIDITY: a task's learned arms are queued only if PERFECT_H binds 2/2 on it at its budget; else the task is UNTESTED (not
a negative) and nothing else runs on it. The decision is written to explore_out/G/s89h_validity.json with the code SHA
before any learned arm is queued, and is read only under that SHA.
OUTCOME: ROUTED-BOUND = explore_g_rhl_parent.routed_bound (BOUND and end margin >= 0.9 and end eta^2 by stream at KEY >
0.9). POOL: the pooled eta^2; ALLPOOL: inv_view(record), the within-sequence eta^2 and VAL cos (V's v2 fix 4: DISCOVERED is
printed in its within-sequence form on ALLPOOL too); the other form printed beside it. Imported: routed_bound, paired,
beats. Exact one-sided McNemar on seed pairs; "discovered" = beats BASELINE_H with p < 0.05; "p <= 0.0625" = at least as
strong as 4 vs 0.
READINGS (fixed now; per task; computed by readings() below; n = 20 paired seeds):
  0. UNTESTED               the perfect gate did not bind 2/2, or a run of BASELINE_H, REG3_KEYHINGE_H, REG3_FREQHINGE_H or
                            REG3_REPEATHINGE_H did not complete (3 attempts). A missing REG3_H seed does not void the task:
                            2b is computed on the seeds that exist, with their n printed.
  1. "the task does not need the hinge"   BASELINE_H ROUTED-BOUND on >= n - 4. Then no other reading.
  2. "the labelled hinge itself fails here"   REG3_KEYHINGE_H not discovered. Then no other reading.
  2b. (V's caveat 1) "the register does not need the hinge here"   REG3_KEYHINGE_H does not beat REG3_H (p >= 0.05). The
                            names in 3 are then printed but read as descriptive: no hinge can show what it adds over a
                            register that binds without one. Every name in 3 prints the arm against REG3_H both ways; where
                            REG3_H beats the arm (p < 0.05), "and the term hurts the register here" is added.
  3. Each of REG3_FREQHINGE_H and REG3_REPEATHINGE_H gets exactly one of four names (S89's, unchanged):
       FREQ: "survives" (discovered, (KH only) - (FREQ only) <= 1) / "is weaker than the key hinge" (discovered, >= 2, not
             "needs") / "needs frequent keys" [ALLPOOL: "needs the key positions"] (p(KH > FREQ) <= 0.0625) / "does nothing
             here" (not discovered, not "needs").
             (V's caveat 2) On POOL the cut is the 2 CTX types, which grouped by CTX id are the stream: the term penalises
             the stream at the 5 CTX positions (the perfect gate's eta^2 there is 1.0, hinge value 0.80; lite's 0.36 / 0.16;
             V's screen_checks_s89_G_v2_out.txt §6). A POOL "needs frequent keys" is read with the arm against REG3_H:
             where REG3_H beats it, the report reads it as the CTX term hurting the register, not the keys' frequency.
       REPEAT: "replaces the key hinge" / "is weaker than the key hinge" / "needs key identities" / "does nothing here",
             with the same four conditions.
     "clamps" for these two arms only: (arm only) - (BASELINE_H only) <= 1 AND the arm's median commitment at 2400 below
     BASELINE_H's.
  THE QUESTION (per task; from the names above, fixed now): does the repeat hinge's S89 result hold on Hebbian memory?
     "holds, and the hinge is needed here"   REPEAT reads "replaces the key hinge" and 2b does not apply (KH beats REG3_H);
     "holds (descriptive: the register does not need the hinge here either)"   REPEAT "replaces" and 2b applies;
     "does not hold"                         REPEAT reads any other name (KH discovered);
     "cannot be asked here"                  the first reading is 0, 1 or 2.
     S89's delta result, read through the same function after the fact (descriptive, dated: S89's readings at db14e9b had
     no 2b): POOL REPEAT "replaces the key hinge" with KH vs REG3_D 4 / 0, p 0.0625, so 2b applies there; ALLPOOL "the
     labelled hinge itself fails here". Both are printed beside S89H's.
  ACROSS MEMORIES (descriptive): each arm's ROUTED-BOUND, Hebbian-conv vs S89's delta, on the same seeds (paired; one-sided
     McNemar both ways). The two differ in the memory AND the convolution together.
  ACROSS TASKS (descriptive): the drop of BASELINE_H, REG3_H, REG3_KEYHINGE_H, REG3_REPEATHINGE_H from POOL to ALLPOOL on the
     same seeds; FREQ not compared (a CTX hinge on POOL, a value hinge on ALLPOOL). The budgets differ (24000 / 48000). The
     repeat hinge groups the same positions by token id, so its pressure on the routed solution differs between tasks (the
     perfect gate pays 0.13 on POOL, 0 on ALLPOOL; a random gate 0.09 / 0.12; the key term's chance level 0.12 on both:
     V's §6); the drop is read as a change of task and of pressure together.
  LITE REFERENCE (descriptive, other seeds, no conv): S87's Hebbian BASELINE 0/20, REG3_KEYHINGE 20/20, REG3_FREQHINGE 17/20
     (2700-2719).
  ALLPOOL CAVEATS (printed beside every ALLPOOL reading): the routing cap (a fixed CTX-type -> channel map separates at most
     0.533 of sequences; the rest needs a within-sequence comparison, which task.perfect_gate is handed), so a null does not
     separate "keys are not frequent" from "contexts have no fixed identity"; F's type-level perfect gate ran on Hebbian
     without the conv only (O_TYPE_HEBB_ALLPOOL 0/2).
OPERATING CHARACTERISTICS (V's v2 §2.10, exact, n = 20, BASELINE ~0): at a ceiling KH ~1.0 "survives" fires with
  probability 1.0 / 0.74 / 0.39 / 0.07 / 0.01 for an arm at 1.0 / 0.95 / 0.9 / 0.8 / 0.7 (at KH ~0.9: 0.79-1.0 at the ceiling,
  ~0.4 at 0.8). With a strong ceiling "does nothing here" cannot fire: an inert arm reads "needs ...", which is why every name
  is read with the arm against REG3_H (2b). The validity gate passes a perfect gate binding at p with probability p^2.
ALSO REPORTED: the hinges' firings and values; beta^R by role and step; the margin per layer; the contingency tables; the
  probe (its per-layer features are the conv output here); the plateaus; the commitment by step; within-sequence routed
  fractions on ALLPOOL.
CHECKS (asserted before any run): X's arm A bit for bit; S89's full CHECK set (explore_g_s89_child: the vendored module,
  token ids only, the task-aware code, the perfect gate, S87's 7 records through S89's run()); and the child's:
  - per task and arm, the conv is the only parameter beyond the no-conv model's, the shared ones equal, and at
    initialisation (identity conv) logits and gate equal the no-conv model's bit for bit; with the conv off REG3_H and
    BASELINE_H equal explore_g_rhl_child's REG3 and BASELINE;
  - the conv insertion equals MultiBDH.forward_conv (REG3_H with A_r = A_R = 0 and random conv weights = BASELINE_H, bit for
    bit) and is strictly causal;
  - PERFECT_H's gate is task.perfect_gate and equals the built-in ceiling+conv model on POOL; untrained it reads ROUTED*;
  - F's O_HEBB_CONV_POOL|2800 and O_HEBB_CONV_ALLPOOL|2800 (2.10GHz records, vendored in explore_out/G/refs/ from
    claude/explore-F 129215e) reproduce through 2400 through PERFECT_H (curve, bit for bit), and S87's REG3_KEYHINGE|2700,
    REG3_FREQHINGE|2700 and BASELINE|2700 through this run() with the conv off;
  - per task: each hinge off = REG3_H through 2400 bit for bit; every arm end to end; each term's gradient reaches the gate
    only (not the stack, not the conv); the lrs (the conv in the rest group; the perfect gate's gate group empty);
  - (parent) readings(): exclusive and exhaustive (as S89's) and the hand cases for 2b and the question.
RUNTIME: about 100 POOL runs (2-9 min) and 100 ALLPOOL runs at 48000 (up to ~20 min unbound): about 8-10 h on 4 workers.

Run:  python explore_g_s89h.py [--report-only]
"""

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_g_common as gc
import explore_g_rhl_parent as hp
import explore_g_s89 as s89p

NAME = "s89h"
CHILD = "explore_g_s89h_child"
TASK_NAMES = ("pool", "allpool")
ITERS = dict(pool=24000, allpool=48000)
S2, S20 = (2840, 2841), tuple(range(2842, 2862))
LEARNED = ("BASELINE_H", "REG3_H", "REG3_KEYHINGE_H", "REG3_FREQHINGE_H", "REG3_REPEATHINGE_H")
ROLE = dict(B="BASELINE_H", R="REG3_H", KH="REG3_KEYHINGE_H", FQ="REG3_FREQHINGE_H", RP="REG3_REPEATHINGE_H")
ROLE_D = dict(B="BASELINE_D", R="REG3_D", KH="REG3_KEYHINGE_D", FQ="REG3_FREQHINGE_D", RP="REG3_REPEATHINGE_D")
REQUIRED = ("B", "KH", "FQ", "RP")
VALID = os.path.join(gc.G_DIR, "s89h_validity.json")
P4 = s89p.P4
NAMES, NEEDS_ALLPOOL = s89p.NAMES, s89p.NEEDS_ALLPOOL
rb = hp.routed_bound
line = hp.line
commit, med_commit = s89p.commit, s89p.med_commit
REG2B = "the register does not need the hinge here"
HURTS = "and the term hurts the register here"
Q = ("holds, and the hinge is needed here", "holds (descriptive: the register does not need the hinge here either)",
     "does not hold", "cannot be asked here")


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _sha():
    return gc.load(NAME)["meta"].get("sha")


def _valid():
    v = _load(VALID)
    return v if v and v.get("sha") == _sha() else None


def _arms():
    a = {f"{t}.PERFECT_H": dict(iters=ITERS[t], seeds=S2, prio=2) for t in TASK_NAMES}
    v = _valid()
    if v:
        for t in TASK_NAMES:
            if v[t]["valid"]:
                for k in LEARNED:
                    a[f"{t}.{k}"] = dict(iters=ITERS[t], seeds=S20, prio=1)
    return a


ARMS = _arms()


def inv_view(r):
    """ALLPOOL's outcome: eta^2 by stream at KEY and VAL cos in their within-sequence forms; the pooled ones kept."""
    e = r["end"]
    assert e.get("eta_key_by_stream_inv") is not None and e.get("val_cos_inv") is not None, "no within-sequence fields"
    return dict(r, val_cos=e["val_cos_inv"], val_cos_pooled=r.get("val_cos"),
                end=dict(e, eta_key_by_stream=e["eta_key_by_stream_inv"], eta_key_by_stream_pooled=e.get("eta_key_by_stream")))


# ── the readings (a pure function of the records) ────────────────────────────
def readings(R, seeds, task, valid=True, valid_note="", label=None):
    """R: role -> {seed: record} (B, KH, FQ, RP required; R optional). Ordered dict of readings; rd['question'] always set."""
    label = ROLE if label is None else label
    rd = {"validity": valid_note}
    n = len(seeds)
    if not valid:
        rd["first"] = "UNTESTED (the perfect gate did not bind 2/2)"
        rd["question"] = Q[3]
        return rd
    missing = {label[k]: sorted(set(seeds) - set(R.get(k, {}))) for k in REQUIRED if set(seeds) - set(R.get(k, {}))}
    if missing:
        rd["first"] = f"UNTESTED (runs that did not complete: {missing})"
        rd["question"] = Q[3]
        return rd
    B, KH, REG = R["B"], R["KH"], R.get("R", {})
    rd["counts"] = ", ".join(f"{label[k]} {sum(rb(r) for r in v.values())}/{len(v)}" for k, v in R.items())
    if set(seeds) - set(REG):
        rd["REG3 seeds missing (2b on the rest)"] = sorted(set(seeds) - set(REG))
    nb = sum(rb(r) for r in B.values())
    kh_disc, t = hp.beats(KH, B, rb)
    rd["REG3_KEYHINGE vs BASELINE"] = f"KH only {t[0]}, BASELINE only {t[1]}, p {t[2]:.3g}"
    if nb >= n - 4:
        rd["first"] = f"the task does not need the hinge (BASELINE {nb}/{n} >= n - 4)"
        rd["question"] = Q[3]
        return rd
    if not kh_disc:
        rd["first"] = "the labelled hinge itself fails here"
        rd["question"] = Q[3]
        return rd
    rd["first"] = "the labelled hinge is discovered; per-arm names below"
    need, (b, c, p, nr) = hp.beats(KH, REG, rb) if REG else (False, (0, 0, 1.0, 0))
    rd["2b"] = (f"{REG2B} (KH only {b}, REG3 only {c}, p {p:.3g}, n {nr}): the names below are descriptive" if not need else
                f"the hinge is needed over the register (KH only {b}, REG3 only {c}, p {p:.3g}, n {nr})")
    names = {}
    for role in ("FQ", "RP"):
        name, why = s89p.arm_name(KH, R[role], B, role, task)
        names[role] = name
        hurt = ""
        if REG:
            ba, ca, pa, na = hp.paired(R[role], REG, rb)
            pr = hp.paired(REG, R[role], rb)[2]
            hurt = f"; vs REG3: arm only {ba}, REG3 only {ca}, p(arm >) {pa:.3g}, p(REG3 >) {pr:.3g}, n {na}"
            if pr < 0.05:
                hurt += f" -- {HURTS}"
                if role == "FQ" and task == "pool" and name == NAMES["FQ"][2]:
                    hurt += " (POOL: read as the CTX term hurting the register, not the keys' frequency)"
        cl, cwhy = s89p.clamps(R[role], B)
        rd[label[role]] = f"{name.upper()} ({why}{hurt}); clamps: {'APPLIES' if cl else 'does not apply'} ({cwhy})"
    rep = names["RP"] == NAMES["RP"][0]
    rd["question"] = (Q[0] if need else Q[1]) if rep else Q[2]
    return rd


# ── reading CHECKs (parent; no torch) ────────────────────────────────────────
def check_readings():
    rows = []
    f = s89p._fake
    s = list(range(20))
    one, zero = [1] * 20, [0] * 20
    cases = [
        ("REG3 20, KH 20, FQ 20, RP 20, B 0", dict(B=f(zero), R=f(one), KH=f(one), FQ=f(one), RP=f(one)),
         lambda rd: rd["2b"].startswith(REG2B) and rd["question"] == Q[1] and rd["REG3_FREQHINGE_H"].startswith(NAMES["FQ"][0].upper())),
        ("REG3 0, KH 20, RP 20, FQ 0, B 0", dict(B=f(zero), R=f(zero), KH=f(one), FQ=f(zero), RP=f(one)),
         lambda rd: rd["2b"].startswith("the hinge is needed") and rd["question"] == Q[0]),
        ("REG3 16, KH 20, FQ 7 (inside REG3's), RP 20, B 0 (S89's delta POOL pattern)",
         dict(B=f(zero), R=f([1] * 16 + [0] * 4), KH=f(one), FQ=f([1] * 7 + [0] * 13), RP=f(one)),
         lambda rd: rd["2b"].startswith(REG2B) and HURTS in rd["REG3_FREQHINGE_H"] and "CTX term" in rd["REG3_FREQHINGE_H"]
         and rd["question"] == Q[1]),
        ("REG3 0, KH 20, RP 10, B 0", dict(B=f(zero), R=f(zero), KH=f(one), FQ=f(one), RP=f([1] * 10 + [0] * 10)),
         lambda rd: rd["question"] == Q[2]),
        ("KH 0 = B 0", dict(B=f(zero), R=f(zero), KH=f(zero), FQ=f(zero), RP=f(zero)),
         lambda rd: rd["first"] == "the labelled hinge itself fails here" and rd["question"] == Q[3]),
        ("B 17", dict(B=f([1] * 17 + [0] * 3), R=f(one), KH=f(one), FQ=f(one), RP=f(one)),
         lambda rd: rd["first"].startswith("the task does not need") and rd["question"] == Q[3]),
        ("a missing REG3 seed does not void", dict(B=f(zero), R={i: r for i, r in f(one).items() if i}, KH=f(one), FQ=f(one), RP=f(one)),
         lambda rd: rd["first"].startswith("the labelled hinge is discovered") and "n 19" in rd["2b"]),
        ("a missing RP seed voids", dict(B=f(zero), R=f(one), KH=f(one), FQ=f(one), RP={i: r for i, r in f(one).items() if i}),
         lambda rd: rd["first"].startswith("UNTESTED") and rd["question"] == Q[3]),
    ]
    for nm, R, ok in cases:
        rd = readings(R, s, "pool")
        rows.append((f"readings hand case [{nm}]: first '{rd['first'][:50]}', 2b '{rd.get('2b', '-')[:45]}', question "
                     f"'{rd['question']}'", bool(ok(rd))))
    # the four names: S89's exclusive / exhaustive enumeration (run again; the name function is S89's, imported)
    rows += [r for r in s89p.check_readings() if "count vectors" in r[0]]
    # inv_view: DISCOVERED and ROUTED* from the within-sequence fields
    r = dict(transition=1200, val_cos=0.9, end=dict(margin=1.0, eta_key_by_stream=0.0, eta_key_by_stream_inv=0.95,
                                                    val_cos_inv=0.1))
    rows.append(("inv_view: routed_bound and DISCOVERED read the within-sequence fields (pooled eta^2 0, VAL cos 0.9; "
                 "within 0.95, 0.1)", rb(inv_view(r)) and hp.discovered(inv_view(r)) and not rb(r) and not hp.discovered(r)))
    return rows


# ── report ───────────────────────────────────────────────────────────────────
def _view(task, rr):
    return {s: inv_view(r) for s, r in rr.items()} if task == "allpool" else rr


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    other = {a: sum(rb(inv_view(r) if a.startswith("pool.") else r) for r in R[a].values()) for a in R}
    for a in list(R):
        R[a] = _view(a.split(".")[0], R[a])
    v = _valid() or {}
    order = [a for t in TASK_NAMES for a in [f"{t}.PERFECT_H"] + [f"{t}.{k}" for k in LEARNED] if a in ARMS]
    print(hp.count_table(R, order))
    print("OUTCOME FORM: pool pooled eta^2, allpool within-sequence (eta^2 and VAL cos). ROUTED-BOUND under the other form, "
          "descriptive: " + ", ".join(f"{a} {k}/{len(R[a])}" for a, k in other.items()))
    # S89's delta records, read through the same function (descriptive, after the fact)
    s89R = {}
    for t in TASK_NAMES:
        s89R[t] = {k: _view(t, s89p.recorded("s89", f"{t}.{a}", S20)) for k, a in ROLE_D.items()}
    out = {}
    for t in TASK_NAMES:
        o = R.get(f"{t}.PERFECT_H", {})
        nb = sum(hp.bound(r) for r in o.values())
        note = (f"PERFECT_H (Hebbian + conv, SLOW, {ITERS[t]}) on 2840-2841 bound {nb}/{len(o)}: "
                + ", ".join(f"{s} {'bound @' + str(r['transition']) if hp.bound(r) else 'unbound'}" for s, r in sorted(o.items())))
        RR = {k: R[f"{t}.{a}"] for k, a in ROLE.items() if f"{t}.{a}" in R}
        out[t] = readings(RR, S20, t, valid=bool(v.get(t, {}).get("valid")), valid_note=note)
        d = readings(s89R[t], S20, t, valid=True, valid_note="S89's ORACLE_D 2/2", label=ROLE_D)
        out[t]["S89 delta, read through the same function (descriptive, after the fact)"] = (
            f"first: {d['first']}; 2b: {d.get('2b', '-')}; REPEAT: {d.get('REG3_REPEATHINGE_D', '-').split(' (')[0]}; "
            f"question: {d['question']}")
        if t == "allpool":
            out[t]["caveats"] = ("the routing cap: a fixed CTX-type map separates at most 0.533 of sequences, so a null does "
                                 "not separate 'keys are not frequent' from 'contexts have no fixed identity'; F's type-level "
                                 "perfect gate ran on Hebbian without the conv only (O_TYPE_HEBB_ALLPOOL 0/2)")
        out[t]["commitment at 2400 (median spread)"] = ", ".join(f"{a} {med_commit(R[f'{t}.{a}']):.4f}" for a in LEARNED
                                                                 if R.get(f"{t}.{a}"))
    mem = {}
    for t in TASK_NAMES:
        for k in ROLE:
            rh, rd_ = R.get(f"{t}.{ROLE[k]}"), s89R[t].get(k)
            if rh and rd_:
                b, c, p, n = hp.paired(rh, rd_, rb)
                p2 = hp.paired(rd_, rh, rb)[2]
                mem[f"{t}.{ROLE[k]} vs {ROLE_D[k]}"] = (f"Hebbian-conv {sum(rb(r) for r in rh.values())}/{len(rh)}, delta "
                                                       f"{sum(rb(r) for r in rd_.values())}/{len(rd_)}; Hebbian only {b}, delta only "
                                                       f"{c}, p(Hebbian >) {p:.3g}, p(delta >) {p2:.3g}, n {n}")
    across = {}
    for a in ("BASELINE_H", "REG3_H", "REG3_KEYHINGE_H", "REG3_REPEATHINGE_H"):
        rp, ra = R.get(f"pool.{a}"), R.get(f"allpool.{a}")
        if rp and ra:
            b, c, p, n = hp.paired(rp, ra, rb)
            across[a] = (f"pool {sum(rb(r) for r in rp.values())}/{len(rp)}, allpool {sum(rb(r) for r in ra.values())}/{len(ra)}; "
                         f"pool only {b}, allpool only {c}, p(pool >) {p:.3g}, n {n}")
    lite = {a: gc.cnt(sum(rb(r) for r in s89p.recorded("s87", a, range(2700, 2720)).values()), 20)
            for a in ("BASELINE", "REG3_KEYHINGE", "REG3_FREQHINGE")}
    print("\nREADINGS:")
    for t in TASK_NAMES:
        print(f"  {t}:\n" + "\n".join(f"      {k}: {x}" for k, x in out[t].items()))
    print("  ACROSS MEMORIES (descriptive; memory and conv differ together):\n" + "\n".join(f"      {k}: {x}" for k, x in mem.items()))
    print("  ACROSS TASKS (descriptive; FREQ not compared; budgets 24000 / 48000):\n" + "\n".join(f"      {k}: {x}" for k, x in across.items()))
    print("  LITE REFERENCE (S87, Hebbian, no conv, 2700-2719): " + ", ".join(f"{k} {x}" for k, x in lite.items()))
    for a in order:
        tv = [r["term"] for r in R.get(a, {}).values() if r.get("term")]
        if tv:
            fr = [x["fired"] for x in tv]
            print(f"  {a}: firings median {statistics.median(fr)} ({min(fr)}-{max(fr)})")
        if R.get(a) and "PERFECT" not in a:
            e = [r["end"] for r in R[a].values()]
            print(f"  {a}: end median margin {statistics.median(x['margin'] for x in e):.3f}, eta^2 by stream at KEY pooled "
                  f"{statistics.median(x['eta_key_by_stream_pooled'] if 'eta_key_by_stream_pooled' in x else x['eta_key_by_stream'] for x in e):.3f}"
                  f" / within {statistics.median(x['eta_key_by_stream_inv'] for x in e):.3f}, eta^2 by key at KEY "
                  f"{statistics.median(x['eta_key_by_key'] for x in e):.3f}, sequences routed median "
                  f"{statistics.median(x['seq_routed_frac'] for x in e):.2f} (max {max(x['seq_routed_frac'] for x in e):.2f})")
            bys = {st_: statistics.median([commit(r, st_) for r in R[a].values() if commit(r, st_) is not None])
                   for st_ in (0, 1200, 2400, 3600) if any(commit(r, st_) is not None for r in R[a].values())}
            print(f"  {a}: commitment (spread) by step " + ", ".join(f"{k} {x:.4f}" for k, x in bys.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, order))
    print("\nper run:\n" + hp.per_run(R, order))
    with open(os.path.join(gc.G_DIR, "s89h_summary.json"), "w") as fh:
        json.dump(dict(readings=out, memories=mem, across=across, lite=lite), fh, indent=1)


# ── main ─────────────────────────────────────────────────────────────────────
def run_checks():
    jobs = [("explore_g_s89_child", "checks", {}), ("explore_g_s89_child", "lr_rows", {}),
            (CHILD, "checks_static", {}), (CHILD, "checks_lrs", {})]
    jobs += [("explore_g_s89_child", "checks_repro", dict(arm=a, seed=s)) for a, s in (
        ("REG3_FREQHINGE_D", 2720), ("REG3_KEYHINGE_D", 2720), ("BASELINE_D", 2720), ("REG3_D", 2720),
        ("REG3_FREQHINGE", 2700), ("REG3_KEYHINGE", 2700), ("BASELINE", 2700))]
    jobs += [(CHILD, "checks_repro", dict(kind="f", key=f"O_HEBB_CONV_{t.upper()}|2800", seed=2800, task=t)) for t in TASK_NAMES]
    jobs += [(CHILD, "checks_repro", dict(kind="s87", arm=a + "_H", ref_arm=a, seed=2700))
             for a in ("REG3_KEYHINGE", "REG3_FREQHINGE", "BASELINE")]
    jobs += [(CHILD, f, dict(task=t)) for t in TASK_NAMES for f in ("checks_off", "checks_on")]

    def one(j):
        r = gc.run_child(j[0], j[1], j[2])
        return [[n if j[0] == CHILD else f"(S89) {n}", bool(v)] for n, v in r]
    with ThreadPoolExecutor(gc.WORKERS) as ex:
        res = list(ex.map(one, jobs))
    return [r for rr in res for r in rr]


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S89H")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = [list(x) for x in check_readings()] + run_checks()
        for nm, ok in rows:
            print(f"  CHECK S89H: {nm}: {'ok' if ok else 'FAIL'}", flush=True)
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s89h_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        assert all(ok for _, ok in rows), "a CHECK failed"
        gc.run_screen(me)                                   # phase 1: the perfect gate per task
        if _valid() is None:
            R = {a: gc.runs(me, a) for a in ARMS}
            v = {t: dict(bound=sum(hp.bound(r) for r in R[f"{t}.PERFECT_H"].values()), n=len(R[f"{t}.PERFECT_H"]),
                         iters=ITERS[t]) for t in TASK_NAMES}
            for t in TASK_NAMES:
                v[t]["valid"] = v[t]["bound"] == 2 and v[t]["n"] == 2
            v["sha"] = _sha()
            v["time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            json.dump(v, open(VALID, "w"), indent=1)
            print(f"  S89H validity (the Hebbian+conv perfect gate 2/2 per task): {v}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)                                   # phase 2: the learned arms on the valid tasks
        print(f"  S89H runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
