#!/usr/bin/env python
"""
explore_g_devices.py — EXPLORATORY, not a result. Session G, screen S82: devices that delay or prevent the key split so the
register's CTX write can emerge. Everything below was fixed and committed before any S82 run.

BACKGROUND (S72 / S73, RandHeaderTask-lite): the register gate shared by all three layers (REG3) routes and binds 20/20
(Hebbian) and 10/10 (delta) with the 5% nudge, and label-free on 4/20 on each memory (BASELINE 0/20); in the other 16 the
gate locks into a key split before the register's write at CTX appears (beta^R flat at about 0.2-0.3 at every role).

TASK, MODELS, RECIPES, STATISTICS, OUTCOMES: explore_g_devices_child (devices), explore_g_rhl_child (task, REG3, statistics,
probe, per-cell accuracy) and explore_g_rhl_parent (outcomes: ROUTED-BOUND = BOUND and F's ROUTED*@end is primary; BOUND,
DISCOVERED and my BOUND ROUTED printed; failure classes). RandHeaderTask-lite, k = 2, no conv, 24000 updates, SLOW.
PHASE 1 (Hebbian; seeds 2100-2119, paired: the same batches and the same initial parameters wherever they exist)
  BASELINE        LOCAL3 + SLOW
  REG3            S72's arm (the reference on new seeds)
  REG3_KEYHINGE   REG3 + S52's key-term hinge (LAM 1.0, TAU 0.2, updates 1-2400) on the three layers' gates (mean over
                  layers). It reads key positions and key identities (token types), as the key split reads key positions;
                  it reads no stream label.
  REG3_SLOWGATE   REG3 with the gate group at 1e-4 for updates 1-1200, then 1e-3; the rest as SLOW
  REG3_NOWIN      REG3 whose gate reads [LN(r^(l)); R_{t-1}] only (no window input; no window parameters)
  REG3_NUDGE      REG3 + the 5% labelled nudge, seeds 2100-2101 (validity of the code path; expected 2/2)
  ORACLE_ALL      the perfect gate at all three layers, seeds 2100-2101 (S71's oracle rerun on this block)
  104 runs.
PHASE 2 (delta: beta 1, L2 keys, tied write, decay 0.95; seeds 2100-2109): the BEST device (rule below) as <DEVICE>_D,
  with BASELINE_D and REG3_D on the same seeds as its pairs (added so that the delta arm has a paired reference), and
  ORACLE_ALL_D on 2100-2101. 32 runs. The choice is written to explore_out/G/s82_best.json before phase 2 is queued.
  BEST-DEVICE RULE (fixed now): among REG3_KEYHINGE, REG3_SLOWGATE, REG3_NOWIN the most ROUTED-BOUND runs on 2100-2119; a
  tie goes to the fewer KEY failures, then the order KEYHINGE, SLOWGATE, NOWIN.

VALIDITY: S71's ORACLE_ALL (10/10 on 700-709) stands for the model; the rerun on 2100-2101 is printed (expected 2/2), and
REG3_NUDGE's 2/2 is printed as the code path's validity. If the rerun oracle binds 0/2 the screen is reported UNTESTED.
READINGS (ROUTED-BOUND, paired on 2100-2119; exact one-sided McNemar, b = seeds where the first arm alone succeeds)
  for each device D in {REG3_KEYHINGE, REG3_SLOWGATE, REG3_NOWIN} (and printed for REG3 too):
  "discovered"                         D beats BASELINE: p < 0.05 (5 vs 0, 7 vs 1, 9 vs 2 or better)
  "the device helps"                   D beats REG3 by at least 4 vs 0: one-sided p(D > REG3) <= p(4 vs 0) = 0.0625
                                       (4 vs 0, 5 vs 0, 6 vs 1, ... — at least as strong as 4 vs 0)
  "the device hurts"                   REG3 beats D by at least 4 vs 0 (the same rule the other way)
  "a working write without the device's cost"   D's unbound runs have no KEY failures, AND its end beta^R at CTX exceeds
                                       both KEY's and VAL's in >= 15/20 runs
  Phase 2 (descriptive, 10 seeds): the same comparisons for <DEVICE>_D against BASELINE_D and REG3_D, printed with b, c, p.
ALSO REPORTED: beta^R at CTX / KEY / VAL by step; the routing margin at VAL per layer; the CTX / KEY / VAL x stream
contingency of g at every layer; the in-run probe; accuracy by (query stream, block, pair index) for the plateaus; the
hinge's firings per window and eta2_key at updates 1-2400; the slow gate's lrs.
CHECKS (explore_g_devices_child.checks, asserted before any run, and X's arm A bit for bit): S73's 37; each device with its
switch off (lam 0; gate lr 1e-3 from update 1; window on) equals REG3|2100 through 2400 bit for bit (curve, statistics,
probe); the hinge's gradient through _Inject equals autograd of the term; eta2_key at RandHeaderTask's KEY positions equals
a hand computation; the hinge's schedule (LAM at 1 and 2400, 0 at 2401, 2402); the slow gate's lrs at 1 / 1200 / 1201 /
2400 / 2401 and its groups; NOWIN has no window parameters, is causal, and with the window on equals REG3.

Run:  python explore_g_devices.py [--report-only]     (phase 2 is queued automatically after phase 1)
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

NAME = "s82_devices"
CHILD = "explore_g_devices_child"
S20, S10, S2 = tuple(range(2100, 2120)), tuple(range(2100, 2110)), (2100, 2101)
DEVICES = ("REG3_KEYHINGE", "REG3_SLOWGATE", "REG3_NOWIN")
BEST = os.path.join(gc.G_DIR, "s82_best.json")
P1 = {"ORACLE_ALL": S2, "REG3_NUDGE": S2, "BASELINE": S20, "REG3": S20,
      "REG3_KEYHINGE": S20, "REG3_SLOWGATE": S20, "REG3_NOWIN": S20}
P4 = 0.0625                                    # one-sided McNemar p of 4 vs 0


def best():
    return json.load(open(BEST)) if os.path.exists(BEST) else None


def _arms():
    a = {k: dict(iters=24000, seeds=v, prio=2 if k == "ORACLE_ALL" else 1) for k, v in P1.items()}
    b = best()
    if b:
        for k in ("ORACLE_ALL_D", "BASELINE_D", "REG3_D", b["device"] + "_D"):
            a[k] = dict(iters=24000, seeds=S2 if k == "ORACLE_ALL_D" else S10, prio=2 if k == "ORACLE_ALL_D" else 1)
    return a


ARMS = _arms()
line = hp.line


def write_ok(r):
    b = r["end"].get("beta")
    return bool(b and b["CTX"] > max(b["KEY"], b["VAL"]))


def decide(R):
    rb = hp.routed_bound
    score = []
    for i, d in enumerate(DEVICES):
        rr = R[d]
        key = sum(1 for r in rr.values() if not hp.bound(r) and hp.outcome(r) == "KEY")
        score.append((-sum(rb(r) for r in rr.values()), key, i, d))
    s = sorted(score)[0]
    return dict(device=s[3], routed_bound=-s[0], key_failures=s[1], table={d: [-a, k] for a, k, _, d in score},
                time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    order = list(P1)[2:] + list(P1)[:2] + [a for a in ARMS if a.endswith("_D")]
    print(hp.count_table(R, order))
    rb = hp.routed_bound
    fmt = lambda t: f"b {t[0]}, c {t[1]}, p {t[2]:.3g}, n {t[3]}"
    no = sum(hp.bound(r) for r in R["ORACLE_ALL"].values())
    nn = sum(rb(r) for r in R["REG3_NUDGE"].values())
    rd = {"validity": f"S71 ORACLE_ALL 10/10 stands; rerun on 2100-2101 bound {no}/{len(R['ORACLE_ALL'])}; REG3_NUDGE "
                      f"ROUTED-BOUND {nn}/{len(R['REG3_NUDGE'])} -> {'VALID' if no >= 1 else 'UNTESTED'}"}
    for d in ("REG3",) + DEVICES:
        x, t = hp.beats(R[d], R["BASELINE"], rb)
        rd[f"{d}: discovered (beats BASELINE)"] = f"{'APPLIES' if x else 'does not apply'} ({fmt(t)})"
        if d == "REG3":
            continue
        b, c, p, n = hp.paired(R[d], R["REG3"], rb)
        rd[f"{d}: the device helps (p(D > REG3) <= {P4})"] = f"{'APPLIES' if p <= P4 else 'does not apply'} (b {b}, c {c}, p {p:.3g})"
        b2, c2, p2, _ = hp.paired(R["REG3"], R[d], rb)
        rd[f"{d}: the device hurts (p(REG3 > D) <= {P4})"] = f"{'APPLIES' if p2 <= P4 else 'does not apply'} (b {b2}, c {c2}, p {p2:.3g})"
        keyf = sum(1 for r in R[d].values() if not hp.bound(r) and hp.outcome(r) == "KEY")
        nw = sum(write_ok(r) for r in R[d].values())
        rd[f"{d}: a working write without the device's cost"] = (f"{'APPLIES' if keyf == 0 and nw >= 15 else 'does not apply'} "
                                                                  f"(KEY failures {keyf}; beta^R CTX > KEY, VAL in {nw}/{len(R[d])})")
    rd["REG3: beta^R CTX > KEY, VAL"] = f"{sum(write_ok(r) for r in R['REG3'].values())}/{len(R['REG3'])} (printed)"
    b = best()
    if b:
        rd["best device (rule)"] = f"{b['device']} ({b['table']})"
        dd = b["device"] + "_D"
        if R.get(dd):
            for ref in ("BASELINE_D", "REG3_D"):
                t = hp.paired(R[dd], R[ref], rb)
                t2 = hp.paired(R[ref], R[dd], rb)
                rd[f"{dd} vs {ref} (descriptive, 10 seeds)"] = f"{dd} only {t[0]}, {ref} only {t[1]}, p({dd} > {ref}) {t[2]:.3g}, p({ref} > {dd}) {t2[2]:.3g}"
            rd[f"{dd}: beta^R CTX > KEY, VAL"] = f"{sum(write_ok(r) for r in R[dd].values())}/{len(R[dd])}"
    print("\nREADINGS:\n" + "\n".join(f"  {k}: {v}" for k, v in rd.items()))
    hk = R.get("REG3_KEYHINGE", {})
    if hk:
        f = [r["hinge"]["fired"] for r in hk.values()]
        e0 = [r["hinge"]["eta_at"] for r in hk.values()]
        print(f"\n  REG3_KEYHINGE: firings in updates 1-2400 median {statistics.median(f)} ({min(f)}-{max(f)}); eta2_key (mean over "
              "layers, median over runs) at " + ", ".join(
                  f"{u} {statistics.median(statistics.fmean(e[u]) for e in e0 if u in e):.3f}" for u in ("1", "300", "600", "1200", "2400")
                  if any(u in e for e in e0)))
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(R, order))
    print("\nper run:\n" + hp.per_run(R, order))
    with open(os.path.join(gc.G_DIR, "s82_summary.json"), "w") as fh:
        json.dump(dict(readings=rd, best=b), fh, indent=1)


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S82 devices")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = gc.run_child(CHILD, "checks", {})
        for nm, v in rows:
            print(f"  CHECK S82: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        with open(os.path.join(gc.G_DIR, f"checks_s82_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        gc.run_screen(me, only=list(P1))
        if best() is None:
            d = decide({a: gc.runs(me, a) for a in DEVICES})
            with open(BEST, "w") as f:
                json.dump(d, f, indent=1)
            print(f"  S82 best device (phase 2): {d}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)
        print(f"  S82 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
