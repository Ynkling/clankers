#!/usr/bin/env python
"""
explore_g_s90.py — EXPLORATORY, not a result. Session G, screen S90 (round 7): the repeat hinge on S87's exact lite
setting, the register spec's own setting. Everything below was fixed and committed before any S90 run.

WHY. On POOL the repeat hinge equals the labelled key hinge on both memories (S89 delta 20/20 vs 20/20; S89H Hebbian + conv
19/20 vs 18/20, where the register alone binds 9/20 and the hinge is needed, p 0.002). The register spec draft is Hebbian on
RandHeaderTask-lite without a conv, where S87 has the key hinge 20/20, BASELINE 0/20 and the frequency hinge 17/20 on seeds
2700-2719 (S82: REG3 alone 2/20 on 2100-2119). S90 asks the spec's question of the repeat hinge in that setting.
SETTING (S87's exactly): RandHeaderTask-lite (explore_g_rhl_child.RHL), Hebbian memory (decay 0.95), k = 2, no conv, 24000
updates, SLOW, the REG3 model; seeds 2700-2719 (S87's, as the prompt of 21 October says). Evaluation every 1200, early stop
after 3 evaluations >= 0.95.
ARMS AND PAIRING. This CPU reproduces S87's records bit for bit (S89's CHECKs: REG3_KEYHINGE|2700, REG3_FREQHINGE|2700,
BASELINE|2700 through S89's run(); asserted again here), so S90 pairs with S87's records: BASELINE, REG3_KEYHINGE and
REG3_FREQHINGE are S87's (explore_out/G/s87_results.json, read by a CPU-tolerant prefix lookup), not rerun. S87 ran no
hingeless REG3 on these seeds, so REG3 runs here (it is needed for reading 2b). New runs (explore_g_s89_child.run on the
lite task, the path that reproduced S87's records):
  lite.REG3              S72's register gate shared by three layers, SLOW, no term (20 runs);
  lite.REG3_REPEATHINGE  REG3 + the repeat hinge (explore_g_s89_child: the key term's form, LAM 1.0, TAU 0.2, updates
                         1-2400, at the positions of token types that occur more than once in the input sequence, grouped
                         by token id; it receives the token ids only) (20 runs).
  VALIDITY: S87's ORACLE_ALL 2/2 on 2700-2701 stands (and S71's 10/10); ORACLE_ALL|2700 reproduces through 2400 (CHECK).
THE REPEAT SET ON LITE (CHECKed counts, 4096 sequences): 16.63 positions per sequence = 5 CTX (4 block headers + the
  query's) + 9 KEY (8 body keys + the query's) + 2.63 VAL (values repeated by chance across keys). It contains all 8 of the
  key term's positions, and there its grouping (token id) is the key term's (key id = token - S) one to one. It reads no
  label (no stream, no role, no key index) and no corpus statistic (no frequency table): only which token types repeat
  within the sequence. Beyond the key term it also groups the CTX positions by CTX id (= the stream on lite, as S87's
  frequency hinge did) and the chance value repeats.
OUTCOME: ROUTED-BOUND = explore_g_rhl_parent.routed_bound (pooled eta^2; lite's stream index is the CTX id). Paired, beats
  imported. Exact one-sided McNemar on seed pairs; "discovered" = beats BASELINE with p < 0.05; "p <= 0.0625" = at least as
  strong as 4 vs 0.
READINGS (fixed now; computed by explore_g_s89h.readings, the function S89H committed at d5d48ee, imported; n = 20). No
  relay text with S90's own reading names reached this session; the readings are S89H's, applied to lite, with the round-7
  prompt's name "the repeat hinge falls short" defined below:
  0. UNTESTED      a run of REG3_REPEATHINGE did not complete (3 attempts); a missing REG3 seed does not void (2b on the rest).
  1. "the task does not need the hinge"        BASELINE >= n - 4 (S87: 0/20; it cannot fire).
  2. "the labelled hinge itself fails here"    REG3_KEYHINGE not discovered (S87: 20 / 0; it cannot fire).
  2b. "the register does not need the hinge here"   REG3_KEYHINGE does not beat REG3 (p >= 0.05); the names in 3 are then
                   descriptive. Every name prints the arm against REG3 both ways; "and the term hurts the register here"
                   where REG3 beats the arm.
  3. REG3_REPEATHINGE gets exactly one of: "replaces the key hinge" (discovered, (KH only) - (REPEAT only) <= 1) / "is
     weaker than the key hinge" (discovered, >= 2, not "needs") / "needs key identities" (p(KH > REPEAT) <= 0.0625) /
     "does nothing here" (not discovered, not "needs"); and "clamps" if (REPEAT only) - (BASELINE only) <= 1 and its
     median commitment at 2400 is below BASELINE's. (REG3_FREQHINGE, S87's, is named by the same function, descriptive.)
  THE QUESTION: "holds, and the hinge is needed here" (REPEAT "replaces" and KH beats REG3) / "holds (descriptive: the
     register does not need the hinge here either)" (REPEAT "replaces", 2b applies) / "does not hold" (another name) /
     "cannot be asked here" (0, 1 or 2).
  FOR THE SPEC (fixed now): the register spec's label-free arm is the repeat hinge if THE QUESTION reads "holds, and the hinge
     is needed here". Every other outcome reads "the repeat hinge falls short" for the spec, which then carries the key
     hinge as its only hinge arm and says why a label-free arm is not in it.
OPERATING CHARACTERISTICS (V's v2 §2.10): at a ceiling of 20/20, "replaces" (within 1 pair) fires with probability 1.0 /
  0.74 / 0.39 / 0.07 for an arm at 1.0 / 0.95 / 0.9 / 0.8; an arm at 0.9 reads "is weaker" about half the time. So "falls
  short" here can mean a true rate as high as 0.9.
ALSO REPORTED: the term's firings and values; the commitment by step; beta^R by role and step; the margin per layer; the
  contingency tables; the probe; the plateaus; S87's SELFKL and TEMP are not re-read.
CHECKS (asserted before any run): X's arm A bit for bit; S87's REG3_KEYHINGE|2700, REG3_FREQHINGE|2700, BASELINE|2700 through
  S89's run() on lite and ORACLE_ALL|2700 through S87's own path, bit for bit through 2400 (the pairing condition); the
  repeat set's counts on lite and the key term inside it (grouping one to one); REG3_REPEATHINGE with its switch off equals
  REG3|2700 through 2400 bit for bit; its term's gradient reaches the gate only; S89's static CHECKs (token ids only: the
  repeat statistic receives the token ids only); readings() hand cases (S89H's check_readings).
RUNTIME: 40 runs, 2-9 min each: about 1.5 h on 4 workers.

Run:  python explore_g_s90.py [--report-only]
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
import explore_g_s89h as s89h

NAME = "s90"
CHILD = "explore_g_s89_child"
S20 = tuple(range(2700, 2720))
ARMS = {"lite.REG3": dict(iters=24000, seeds=S20, prio=1), "lite.REG3_REPEATHINGE": dict(iters=24000, seeds=S20, prio=1)}
LABEL = dict(B="BASELINE (S87)", R="REG3 (S90)", KH="REG3_KEYHINGE (S87)", FQ="REG3_FREQHINGE (S87)", RP="REG3_REPEATHINGE (S90)")
line = hp.line
rb = hp.routed_bound


def records():
    me = sys.modules[__name__]
    R = dict(R=gc.runs(me, "lite.REG3"), RP=gc.runs(me, "lite.REG3_REPEATHINGE"))
    for k, a in (("B", "BASELINE"), ("KH", "REG3_KEYHINGE"), ("FQ", "REG3_FREQHINGE")):
        R[k] = s89p.recorded("s87", a, S20)
    return R


def spec_decision(rd):
    return ("the spec's label-free arm is the repeat hinge" if rd["question"] == s89h.Q[0] else
            "the repeat hinge falls short: the spec carries the key hinge only")


def report():
    R = records()
    order = ["B", "R", "KH", "FQ", "RP"]
    print(hp.count_table({LABEL[k]: R[k] for k in order}, [LABEL[k] for k in order]))
    o = s89p.recorded("s87", "ORACLE_ALL", (2700, 2701))
    rd = s89h.readings(R, S20, "lite", valid=True, label=LABEL,
                       valid_note=f"S87's ORACLE_ALL {sum(hp.bound(r) for r in o.values())}/{len(o)} on 2700-2701 (and S71's 10/10)")
    rd["FOR THE SPEC"] = spec_decision(rd)
    rd["commitment at 2400 (median spread)"] = ", ".join(f"{LABEL[k]} {s89p.med_commit(R[k]):.4f}" for k in order)
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    for k in ("RP", "R"):
        tv = [r["term"] for r in R[k].values() if r.get("term")]
        if tv:
            f = [x["fired"] for x in tv]
            vals = {u: statistics.median(statistics.fmean(x["value_at"][u]) for x in tv if u in x["value_at"])
                    for u in ("1", "300", "600", "1200", "2400") if any(u in x["value_at"] for x in tv)}
            print(f"  {LABEL[k]}: firings median {statistics.median(f)} ({min(f)}-{max(f)}); value " +
                  ", ".join(f"{u} {v:.4f}" for u, v in vals.items()))
        bys = {st_: statistics.median([s89p.commit(r, st_) for r in R[k].values() if s89p.commit(r, st_) is not None])
               for st_ in (0, 1200, 2400, 3600) if any(s89p.commit(r, st_) is not None for r in R[k].values())}
        print(f"  {LABEL[k]}: commitment (spread) by step " + ", ".join(f"{a} {v:.4f}" for a, v in bys.items()))
    named = {LABEL[k]: R[k] for k in order}
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(named, [LABEL[k] for k in ("R", "RP")]))
    print("\nper run:\n" + hp.per_run(named, [LABEL[k] for k in ("R", "RP")]))
    with open(os.path.join(gc.G_DIR, "s90_summary.json"), "w") as fh:
        json.dump(dict(readings=rd), fh, indent=1)


def run_checks():
    jobs = [("explore_g_s89_child", "checks_repro", dict(arm=a, seed=2700)) for a in ("REG3_KEYHINGE", "REG3_FREQHINGE", "BASELINE")]
    jobs += [("explore_g_s90_child", "check_oracle", {}), ("explore_g_s90_child", "check_lite", {}),
             ("explore_g_s89_child", "checks", {})]
    with ThreadPoolExecutor(gc.WORKERS) as ex:
        res = list(ex.map(lambda j: gc.run_child(j[0], j[1], j[2]), jobs))
    return [r for rr in res for r in rr]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S90")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = [list(x) for x in s89h.check_readings()] + run_checks()
        for nm, ok in rows:
            print(f"  CHECK S90: {nm}: {'ok' if ok else 'FAIL'}", flush=True)
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s90_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        assert all(ok for _, ok in rows), "a CHECK failed"
        gc.run_screen(me)
        print(f"  S90 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
