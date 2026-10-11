#!/usr/bin/env python
"""
explore_g_s83.py — EXPLORATORY, not a result. Session G, screen S83: what the key-term hinge must read, whether the register
is needed, and how long the window must be. Everything below was fixed and committed before any S83 run.

BACKGROUND (S82, seeds 2100-2119, Hebbian, RandHeaderTask-lite, 24000, SLOW): REG3_KEYHINGE (the register gate shared by
all three layers + S52's key-term hinge for updates 1-2400) ROUTED-BOUND 20/20, BASELINE 0/20, REG3 2/20; on delta
REG3_KEYHINGE_D 10/10. The hinge reads key positions and key identities.

TERMS, MODELS, RECORDS: explore_g_s83_child (the key, pos and flat terms, what each reads, and the scale caveat).
OUTCOMES: explore_g_rhl_parent (ROUTED-BOUND = BOUND and F's ROUTED*@end; BOUND, DISCOVERED, my BOUND ROUTED printed).
PHASE 0, REPRODUCTION (before any S83 run): S82's BASELINE, REG3 and REG3_KEYHINGE at seed 2100 (and BASELINE_D, REG3_D,
  REG3_KEYHINGE_D at 2100) are rerun under the current code through 2400; each must equal its S82 record (curve and every
  statistic at 0 / 1200 / 2400). An arm that does not reproduce is rerun in full on its S82 seeds as <ARM>_RERUN, and the
  rerun replaces its S82 records as the reference. The decision is written to explore_out/G/s83_repro.json.
PHASE 1 (Hebbian; seeds 2100-2119, paired with S82's records): LOCAL3_KEYHINGE, REG3_KEYHINGE_POS, REG3_FLAT,
  REG3_KEYHINGE_1200; REG3_NUDGE on 2100-2101 (validity of the code path). 82 runs.
PHASE 2 (delta, beta 1, L2 keys; seeds 2100-2109): the arm of (1)-(2) that reads least and is discovered on Hebbian
  memory, in the order REG3_FLAT (reads nothing), REG3_KEYHINGE_POS (key positions), LOCAL3_KEYHINGE (key positions and
  identities); none discovered -> no phase 2. Paired, descriptive, with S82's BASELINE_D, REG3_D and REG3_KEYHINGE_D. The
  choice is written to explore_out/G/s83_delta.json before phase 2 is queued.
VALIDITY: S71's ORACLE_ALL (10/10) and S82's rerun on 2100-2101 (2/2) stand; REG3_NUDGE 2/2 printed.
READINGS (ROUTED-BOUND; exact one-sided McNemar on seed pairs, b = the first arm only; "discovered" = beats S82's BASELINE
with p < 0.05; "within 1 pair of REG3_KEYHINGE" = (REG3_KEYHINGE only) - (arm only) <= 1; "beats by >= 4 vs 0" = one-sided
p <= p(4 vs 0) = 0.0625, as in S82):
  the register:      "the register is not needed" if LOCAL3_KEYHINGE is discovered and within 1 pair of REG3_KEYHINGE;
                     "the register is needed" if REG3_KEYHINGE beats LOCAL3_KEYHINGE by >= 4 vs 0; otherwise "neither".
  key identity:      "key identity need not be read" if REG3_KEYHINGE_POS is discovered and within 1 pair of REG3_KEYHINGE;
                     "the identity term is needed" if REG3_KEYHINGE beats REG3_KEYHINGE_POS by >= 4 vs 0; else "neither".
  structure:         "no structure need be read" if REG3_FLAT is discovered and within 1 pair of REG3_KEYHINGE; else it
                     does not apply.
  the window:        "the short window suffices" if REG3_KEYHINGE_1200 is within 1 pair of REG3_KEYHINGE; else it does not.
  Each arm's "discovered" is printed with b, c, p.
ALSO REPORTED: the terms' firings and values at updates 1-2400; eta^2 by key per layer; beta^R at CTX / KEY / VAL by step;
the margin per layer; the contingency tables; the probe; accuracy by (stream, block, pair) for the plateaus.
CHECKS (explore_g_s83_child.checks, asserted before any run, and X's arm A bit for bit): S82's 48; each new term with its
switch off equals S82's REG3 bit for bit through 2400, LOCAL3_KEYHINGE with the term off equals S82's BASELINE; the key term
in S83's class equals S82's REG3_KEYHINGE through 2400; the pos and flat terms' (and the key term's) gradients reach the
gate's parameters and, through the gate's input, the embedding, and nothing in the stack; the terms' values by hand; the
schedules; LOCAL3_KEYHINGE has BASELINE's parameters; the delta variants' memory.

Run:  python explore_g_s83.py [--report-only]
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_g_common as gc
import explore_g_rhl_parent as hp
import explore_g_devices as s82

NAME = "s83"
CHILD = "explore_g_s83_child"
S20, S10, S2 = tuple(range(2100, 2120)), tuple(range(2100, 2110)), (2100, 2101)
NEW = ("LOCAL3_KEYHINGE", "REG3_KEYHINGE_POS", "REG3_FLAT", "REG3_KEYHINGE_1200")
DELTA_ORDER = ("REG3_FLAT", "REG3_KEYHINGE_POS", "LOCAL3_KEYHINGE")
REPRO = os.path.join(gc.G_DIR, "s83_repro.json")
DELTA = os.path.join(gc.G_DIR, "s83_delta.json")
REFS_H = ("BASELINE", "REG3", "REG3_KEYHINGE")
REFS_D = ("BASELINE_D", "REG3_D", "REG3_KEYHINGE_D")
P4 = 0.0625


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _arms():
    a = {k: dict(iters=24000, seeds=S20, prio=1) for k in NEW}
    a["REG3_NUDGE"] = dict(iters=24000, seeds=S2, prio=1)
    rp = _load(REPRO) or {}
    for arm, ok in rp.get("result", {}).items():
        if not ok:
            a[f"{arm}_RERUN"] = dict(iters=24000, seeds=S20 if not arm.endswith("_D") else S10, prio=2)
    d = _load(DELTA)
    if d and d.get("arm"):
        a[d["arm"] + "_D"] = dict(iters=24000, seeds=S10, prio=1)
    return a


ARMS = _arms()
line = hp.line


def refs():
    """The paired references: S82's records, or the reruns where S82's did not reproduce."""
    rp = (_load(REPRO) or {}).get("result", {})
    me = sys.modules[__name__]
    out = {}
    for arm in REFS_H + REFS_D:
        out[arm] = gc.runs(me, f"{arm}_RERUN") if rp.get(arm) is False else gc.runs(s82, arm)
    return out


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    R.update(refs())
    order = list(REFS_H) + list(NEW) + ["REG3_NUDGE"] + [a for a in ARMS if a.endswith("_D")] + [a for a in REFS_D if a in R]
    print(hp.count_table(R, order))
    rb = hp.routed_bound
    fmt = lambda t: f"b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]}"
    disc, within = {}, {}
    rd = {"reproduction": str((_load(REPRO) or {}).get("result")),
          "validity": f"S71 ORACLE_ALL 10/10 and S82's rerun 2/2 stand; REG3_NUDGE ROUTED-BOUND "
                      f"{sum(rb(r) for r in R['REG3_NUDGE'].values())}/{len(R['REG3_NUDGE'])}"}
    for a in NEW:
        x, t = hp.beats(R[a], R["BASELINE"], rb)
        disc[a] = x
        b, c, p, n = hp.paired(R["REG3_KEYHINGE"], R[a], rb)
        within[a] = (b - c) <= 1
        rd[f"{a}: discovered (beats BASELINE)"] = f"{'yes' if x else 'no'} ({fmt(t)}); vs REG3_KEYHINGE: KH only {b}, {a} only {c}"
    p_reg = hp.paired(R["REG3_KEYHINGE"], R["LOCAL3_KEYHINGE"], rb)[2]
    rd["the register"] = ("the register is not needed" if disc["LOCAL3_KEYHINGE"] and within["LOCAL3_KEYHINGE"] else
                          "the register is needed" if p_reg <= P4 else "neither") + f" (p(KH > LOCAL3_KH) {p_reg:.3g})"
    p_id = hp.paired(R["REG3_KEYHINGE"], R["REG3_KEYHINGE_POS"], rb)[2]
    rd["key identity"] = ("key identity need not be read" if disc["REG3_KEYHINGE_POS"] and within["REG3_KEYHINGE_POS"] else
                          "the identity term is needed" if p_id <= P4 else "neither") + f" (p(KH > POS) {p_id:.3g})"
    rd["structure"] = "no structure need be read" if disc["REG3_FLAT"] and within["REG3_FLAT"] else "does not apply"
    rd["the window"] = "the short window suffices" if within["REG3_KEYHINGE_1200"] else "does not apply"
    d = _load(DELTA)
    if d:
        rd["delta arm (rule)"] = str(d)
        if d.get("arm") and R.get(d["arm"] + "_D"):
            dd = d["arm"] + "_D"
            for ref in REFS_D:
                b, c, p, n = hp.paired(R[dd], R[ref], rb)
                b2, c2, p2, _ = hp.paired(R[ref], R[dd], rb)
                rd[f"{dd} vs {ref} (descriptive)"] = f"{dd} only {b}, {ref} only {c}, p({dd} >) {p:.3g}, p({ref} >) {p2:.3g}"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    for a in list(NEW) + [x for x in ARMS if x.endswith("_D")]:
        rr = R.get(a, {})
        tv = [r["term"] for r in rr.values() if r.get("term")]
        if tv:
            import statistics as st
            f = [t["fired"] for t in tv]
            vals = {u: st.median(st.fmean(t["value_at"][u]) for t in tv if u in t["value_at"]) for u in ("1", "300", "600", "1200", "2400")
                    if any(u in t["value_at"] for t in tv)}
            print(f"  {a}: term firings median {st.median(f)} ({min(f)}-{max(f)}); term value (mean over layers, median) "
                  + ", ".join(f"{u} {v:.4f}" for u, v in vals.items()))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, order))
    print("\nper run:\n" + hp.per_run(R, [a for a in order if a in ARMS]))
    with open(os.path.join(gc.G_DIR, "s83_summary.json"), "w") as fh:
        json.dump(dict(readings=rd), fh, indent=1)


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S83")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        if _load(REPRO) is None:
            res = gc.run_child(CHILD, "reproduce", dict(arms=list(REFS_H) + list(REFS_D)))
            with open(REPRO, "w") as f:
                json.dump(dict(result=res, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), cpu=ec.cpu_model()), f, indent=1)
        print(f"  S83 phase 0 (S82 records through 2400 under the current code): {_load(REPRO)['result']}", flush=True)
        rows = gc.run_child(CHILD, "checks", {})
        for nm, v in rows:
            print(f"  CHECK S83: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        with open(os.path.join(gc.G_DIR, f"checks_s83_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        ARMS = _arms()
        gc.run_screen(me)
        if _load(DELTA) is None:
            R = {a: gc.runs(me, a) for a in NEW}
            R.update(refs())
            arm = next((a for a in DELTA_ORDER if hp.beats(R[a], R["BASELINE"], hp.routed_bound)[0]), None)
            with open(DELTA, "w") as f:
                json.dump(dict(arm=arm, order=DELTA_ORDER, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())), f, indent=1)
            print(f"  S83 phase 2 arm (rule): {arm}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)
        print(f"  S83 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
