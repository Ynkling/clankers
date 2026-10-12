#!/usr/bin/env python
"""
explore_g_s87.py — EXPLORATORY, not a result. Session G, screen S87: can a term that reads no task labels do the key
term's work on RandHeaderTask-lite? Everything below was fixed and committed before any S87 run.

BACKGROUND. S82/S83 (seeds 2100-2119): REG3_KEYHINGE (the register gate shared by three layers + S52's key term on
updates 1-2400) ROUTED-BOUND 20/20, BASELINE 0/20, REG3 2/20; the key term's identity part is needed (a KL-from-uniform
term at key positions 1/20, at every body position 0/20: both clamp the gate at uniform). A language model has no marked
keys.

DEVICES, CONTROLS, COMMITMENT, THE FREQUENCY TABLE: explore_g_s87_child's docstring (REG3_FREQHINGE: the key term at the
positions of the frequent token types, grouped by token id; REG3_SELFKL: KL of g_t from the sequence's mean gate at every
body position; REG3_TEMP: the gate's softmax at temperature 4 for updates 1-2400). NOTE fixed there: on this task the
frequent types are both CTX types and all four key types (the context tokens are the most frequent, against the premise
that they are rare); the frequent positions are 14 per sequence, 8 of them the key term's 8.
OUTCOMES: explore_g_rhl_parent (ROUTED-BOUND = BOUND and F's ROUTED*@end; BOUND, DISCOVERED, my BOUND ROUTED printed).
PHASE 0: S82's BASELINE and REG3_KEYHINGE at seed 2100 rerun through 2400 under the current code must equal S82's records
  (curve and every statistic at 0 / 1200 / 2400; asserted). With phase 2: S82's BASELINE_D, REG3_D, REG3_KEYHINGE_D too.
PHASE 1 (Hebbian; seeds 2700-2719, paired): BASELINE, REG3_KEYHINGE (controls, rerun), REG3_FREQHINGE, REG3_SELFKL,
  REG3_TEMP; ORACLE_ALL on 2700-2701 (validity). 102 runs.
PHASE 2 (delta, beta 1, L2 keys; seeds 2720-2729): the new arm that reads least and is discovered on Hebbian memory, in the
  order REG3_TEMP, REG3_SELFKL (both read nothing; the schedule before the penalty), REG3_FREQHINGE (a frequency table of
  token ids); with BASELINE_D, REG3_D and REG3_KEYHINGE_D (S82's delta references) rerun on 2720-2729. None discovered ->
  no phase 2. The choice is written to explore_out/G/s87_delta.json before phase 2 is queued.
VALIDITY: S71's ORACLE_ALL 10/10 stands; ORACLE_ALL on 2700-2701 printed (UNTESTED if 0/2).
READINGS (ROUTED-BOUND; exact one-sided McNemar on seed pairs, b = the first arm only; "discovered" = beats BASELINE with
p < 0.05; "within 1 pair of REG3_KEYHINGE" = (REG3_KEYHINGE only) - (arm only) <= 1; "beats by >= 4 vs 0" = one-sided
p <= p(4 vs 0) = 0.0625):
  "a frequency hinge replaces the key hinge"   REG3_FREQHINGE discovered and within 1 pair of REG3_KEYHINGE
  "a hinge that reads nothing suffices"        REG3_SELFKL or REG3_TEMP discovered and within 1 pair of REG3_KEYHINGE
  "clamps" (per arm)                           (arm only) - (BASELINE only) <= 1 AND the arm's median commitment at 2400
                                               is below BASELINE's (both printed for every arm)
  "the identity term is still needed"          REG3_KEYHINGE beats every new arm by >= 4 vs 0
  Each arm's "discovered" printed with b, c, p.
ALSO REPORTED: the terms' values and firings; beta^R at CTX / KEY / VAL by step; the margin per layer; the contingency
tables; the probe; accuracy by (stream, block, pair) for the plateaus; the commitment by step.
CHECKS (explore_g_s87_child.checks, asserted before any run, and X's arm A bit for bit): S83's 61 (S82's 48 + 13); each new
device with its switch off equals REG3|2700 through 2400 bit for bit; the frequency table from token ids only (one
parameter, a LongTensor asserted, a float tensor refused) and its frequent types; the coincidence count over the sample
(14 frequent positions per sequence, 8 key-term positions, 8 in both); the token-id grouping at key positions equals the key
term's; freq and selfkl gradients reach the gate's parameters (and the embedding through its input) and not the stack;
TEMP's gates flatter inside the window and REG3's after it, no parameter added; the schedules over real training forwards.

Run:  python explore_g_s87.py [--report-only]
"""

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_g_common as gc
import explore_g_rhl_parent as hp

NAME = "s87"
CHILD = "explore_g_s87_child"
S20, S2, D10 = tuple(range(2700, 2720)), (2700, 2701), tuple(range(2720, 2730))
NEW = ("REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP")
DELTA_ORDER = ("REG3_TEMP", "REG3_SELFKL", "REG3_FREQHINGE")
DELTA = os.path.join(gc.G_DIR, "s87_delta.json")
REPRO = os.path.join(gc.G_DIR, "s87_repro.json")
P4 = 0.0625


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _arms():
    a = {k: dict(iters=24000, seeds=S20, prio=1) for k in ("BASELINE", "REG3_KEYHINGE") + NEW}
    a["ORACLE_ALL"] = dict(iters=24000, seeds=S2, prio=2)
    d = _load(DELTA)
    if d and d.get("arm"):
        for k in (d["arm"] + "_D", "BASELINE_D", "REG3_D", "REG3_KEYHINGE_D"):
            a[k] = dict(iters=24000, seeds=D10, prio=1)
    return a


ARMS = _arms()
line = hp.line


def commit(r, step=2400):
    for s in r["stats"]:
        if s["step"] == step:
            return s.get("spread")
    return None


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    order = ["BASELINE", "REG3_KEYHINGE"] + list(NEW) + ["ORACLE_ALL"] + [a for a in ARMS if a.endswith("_D")]
    print(hp.count_table(R, order))
    rb = hp.routed_bound
    fmt = lambda t: f"b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]}"
    cm = {a: statistics.median([commit(r) for r in R[a].values()]) for a in ("BASELINE", "REG3_KEYHINGE") + NEW if R.get(a)}
    rd = {"reproduction": str((_load(REPRO) or {}).get("result")),
          "validity": f"S71 ORACLE_ALL 10/10 stands; ORACLE_ALL on 2700-2701 bound {sum(hp.bound(r) for r in R['ORACLE_ALL'].values())}/{len(R['ORACLE_ALL'])}",
          "commitment at 2400 (median spread)": ", ".join(f"{a} {v:.4f}" for a, v in cm.items())}
    disc, within, beat = {}, {}, {}
    for a in NEW:
        x, t = hp.beats(R[a], R["BASELINE"], rb)
        disc[a] = x
        b, c, p, n = hp.paired(R["REG3_KEYHINGE"], R[a], rb)
        within[a] = (b - c) <= 1
        beat[a] = p <= P4
        cl = (hp.paired(R[a], R["BASELINE"], rb)[0] - hp.paired(R[a], R["BASELINE"], rb)[1]) <= 1 and cm[a] < cm["BASELINE"]
        rd[f"{a}"] = (f"discovered {'yes' if x else 'no'} ({fmt(t)}); vs REG3_KEYHINGE: KH only {b}, {a} only {c}, p(KH >) {p:.3g}; "
                      f"clamps: {'APPLIES' if cl else 'does not apply'} (commitment {cm[a]:.4f} vs BASELINE {cm['BASELINE']:.4f})")
    x, t = hp.beats(R["REG3_KEYHINGE"], R["BASELINE"], rb)
    rd["REG3_KEYHINGE (control rerun) vs BASELINE"] = fmt(t)
    rd["a frequency hinge replaces the key hinge"] = "APPLIES" if disc["REG3_FREQHINGE"] and within["REG3_FREQHINGE"] else "does not apply"
    rd["a hinge that reads nothing suffices"] = ("APPLIES (" + ", ".join(a for a in ("REG3_SELFKL", "REG3_TEMP") if disc[a] and within[a]) + ")"
                                                 if any(disc[a] and within[a] for a in ("REG3_SELFKL", "REG3_TEMP")) else "does not apply")
    rd["the identity term is still needed"] = "APPLIES" if all(beat.values()) else "does not apply"
    d = _load(DELTA)
    if d:
        rd["delta arm (rule)"] = str(d)
        if d.get("arm") and R.get(d["arm"] + "_D"):
            dd = d["arm"] + "_D"
            for ref in ("BASELINE_D", "REG3_D", "REG3_KEYHINGE_D"):
                b, c, p, n = hp.paired(R[dd], R[ref], rb)
                b2, c2, p2, _ = hp.paired(R[ref], R[dd], rb)
                rd[f"{dd} vs {ref} (descriptive)"] = f"{dd} only {b}, {ref} only {c}, p({dd} >) {p:.3g}, p({ref} >) {p2:.3g}"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    for a in NEW + tuple(x for x in ARMS if x.endswith("_D") and x.startswith("REG3_") and x[:-2] in NEW):
        tv = [r["term"] for r in R.get(a, {}).values() if r.get("term")]
        if tv:
            f = [t["fired"] for t in tv]
            vals = {u: statistics.median(statistics.fmean(t["value_at"][u]) for t in tv if u in t["value_at"])
                    for u in ("1", "300", "600", "1200", "2400") if any(u in t["value_at"] for t in tv)}
            print(f"  {a}: firings median {statistics.median(f)} ({min(f)}-{max(f)}); value (mean over layers, median) "
                  + ", ".join(f"{u} {v:.4f}" for u, v in vals.items()))
    for a in ("BASELINE", "REG3_KEYHINGE") + NEW:
        bys = {st_: statistics.median([commit(r, st_) for r in R[a].values() if commit(r, st_) is not None])
               for st_ in (0, 1200, 2400, 3600) if any(commit(r, st_) is not None for r in R[a].values())}
        print(f"  {a}: commitment (spread) by step " + ", ".join(f"{k} {v:.4f}" for k, v in bys.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, order))
    print("\nper run:\n" + hp.per_run(R, order))
    with open(os.path.join(gc.G_DIR, "s87_summary.json"), "w") as fh:
        json.dump(dict(readings=rd), fh, indent=1)


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S87")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        if _load(REPRO) is None:
            res = gc.run_child(CHILD, "reproduce", dict(arms=["BASELINE", "REG3_KEYHINGE"]))
            json.dump(dict(result=res, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())), open(REPRO, "w"), indent=1)
        rp = _load(REPRO)["result"]
        print(f"  S87 phase 0 (S82 records through 2400 under the current code): {rp}", flush=True)
        assert all(rp.values()), "an S82 record does not reproduce"
        rows = gc.run_child(CHILD, "checks", {})
        for nm, v in rows:
            print(f"  CHECK S87: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s87_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        gc.run_screen(me)
        if _load(DELTA) is None:
            R = {a: gc.runs(me, a) for a in ("BASELINE",) + NEW}
            arm = next((a for a in DELTA_ORDER if hp.beats(R[a], R["BASELINE"], hp.routed_bound)[0]), None)
            json.dump(dict(arm=arm, order=DELTA_ORDER, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())),
                      open(DELTA, "w"), indent=1)
            print(f"  S87 phase 2 arm (rule): {arm}", flush=True)
            if arm:
                d = gc.run_child(CHILD, "reproduce", dict(arms=["BASELINE_D", "REG3_D", "REG3_KEYHINGE_D"]))
                print(f"  S87 phase 2 reproduction of S82's delta references at 2100 through 2400: {d}", flush=True)
                assert all(d.values()), "an S82 delta record does not reproduce"
        ARMS = _arms()
        gc.run_screen(me)
        print(f"  S87 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
