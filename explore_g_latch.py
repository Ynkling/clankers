#!/usr/bin/env python
"""
explore_g_latch.py — EXPLORATORY, not a result. Session G, screen S57: an HM-RNN latch in place of S56's leaky register.
Everything below was fixed and committed before any S57 run.

TRIGGER (fixed in explore_g_register before any S56 run): S56's R2 "not discovered" AND REG's median end-of-run beta^R at
CTX tokens in [0.2, 0.8] (a leaky compromise). This screen runs only if explore_out/G/s56_hebb_summary.json says
TRIGGERED.

MODEL: explore_g_latch_child.LatchBDH — S56's model (layer 1 LOCAL3; layers 2-3 the per-layer gate over LN(r^(l)), v_t and
the latch state r_{t-1}; SG) with the register replaced by the latch
    u_t = w_z . v_t + b_z,  z~ = clamp((a u + 1)/2, 0, 1),  z_t = 1[z~ > 0.5] forward, straight-through backward with the
    hard sigmoid's derivative,  r_t = z_t W_r v_t + (1 - z_t) r_{t-1};
a = 1 until update 8000, then linear 1 -> 5 over updates 8001-24000; b_z = 0, w_z ~ N(0, 0.01^2) so |u| < 1 at init
(CHECKed: max |u| 2.4e-3). PRIOR: lambda (mean_t z_t - 1/17)^2, lambda = 0.1. Task, recipe (SLOW), nudge, statistics
and outcome definitions: S56's (explore_g_register, explore_g_register_child), unchanged; the "beta^R" fields of the
records hold the latch's firing rate z by role.

ARMS (Hebbian memory; seeds 410-419 paired: the same batches and initial parameters wherever they exist)
  ORACLE_L     the perfect gate, Adam 1e-3 (seeds 410-411; validity)
  BASELINE_L   LOCAL3 + SLOW (S56's BASELINE on these seeds)
  LATCH        the latch, no prior, no nudge
  LATCH_NUDGE  the latch + S56's 5% nudge (validity of the architecture)
  LATCH_PRIOR  the latch + the firing-rate prior (the second arm)
POST HOC (descriptive; not a reading; added after S56's REG_NUDGE runs were seen to route perfectly at layers 2-3 and
still not bind): ORACLE23 — LOCAL3 at layer 1, the perfect gate at layers 2-3, SLOW — on seeds 370-372 (S56's seeds, so
paired with S56's REG_NUDGE and ORACLE_H). It asks whether the unrouted layer-1 memory blocks binding.

VALIDITY: ORACLE_L binds on >= 1 of 2 within 24000, else UNTESTED.
READINGS (S56's, applied to the latch; counts out of 10, Wilson 95% and band with every count; one-sided exact McNemar)
  R1 "the latch carries the cue"      LATCH_NUDGE BOUND ROUTED >= 9/10;  "architecture insufficient" if < 6/10 (then the
                                      firing rate z at CTX vs KEY/VAL: the write decision or the read path); else "neither"
  R2 "discovered"                     LATCH or LATCH_PRIOR BOUND >= BASELINE_L BOUND + 3 with that arm's KEY failures <= 2;
                                      else "not discovered"
ALWAYS REPORTED: z at CTX / KEY / VAL by step (median over runs), the routing margin at VAL per layer, the CTX / KEY / VAL x
stream contingency table of g at layers 2 and 3, transitions, failure classes.
CHECKS (explore_g_latch_child.checks, asserted before any run; S56's CHECKs rerun too): the dead-STE guard at step 0; the
a schedule; the STE's forward and gradient; the latch's hold / write recurrence and its causality (the gate at t reads
r_{t-1}); the prior's gradient on training forwards; LATCH shares S56 REG's parameters except b_R; the nudge moves only the
new gate parameters; ORACLE23's gates.

Run:  python explore_g_latch.py [--report-only] [--force]
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_g_common as gc
import explore_common as ec
import explore_g_register as s56p

NAME = "s57_latch"
CHILD = "explore_g_latch_child"
LS = tuple(range(410, 420))
ARMS = {
    "ORACLE_L": dict(iters=24000, seeds=LS[:2], prio=2),
    "BASELINE_L": dict(iters=24000, seeds=LS, prio=1),
    "LATCH": dict(iters=24000, seeds=LS, prio=1),
    "LATCH_NUDGE": dict(iters=24000, seeds=LS, prio=1),
    "LATCH_PRIOR": dict(iters=24000, seeds=LS, prio=1),
    "ORACLE23": dict(iters=24000, seeds=(370, 371, 372), prio=1, post_hoc=True),
}
GATED = {"BASELINE_L": [3]}


def line(arm, r):
    return s56p.line(arm, r) if r.get("ok") else f"FAILED {r.get('error')}"


def bound_routed(r):
    if r["transition"] is None:
        return False
    lay = r["end"]["layers"]
    return bool(all(lay[f"L{l}"]["one_to_one_val"] for l in GATED.get(r["arm"], [2, 3])) and min(r["stream_acc"]) >= 0.9)


def outcome(r):
    if r["transition"] is not None:
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = ec.fail_class(r)
    if fc == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return fc


def report():
    st = gc.load(NAME)
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a, st) for a in ARMS}
    bnd = lambda r: r["transition"] is not None
    print("| arm | n | BOUND | BOUND ROUTED | failures (unbound) | transitions | median acc |")
    print("|---|---|---|---|---|---|---|")
    summ = {}
    for a, rr in R.items():
        if not rr:
            continue
        n = len(rr)
        nb = sum(bnd(r) for r in rr.values())
        nbr = sum(bound_routed(r) for r in rr.values())
        fails = {}
        for r in rr.values():
            if not bnd(r):
                fails[outcome(r)] = fails.get(outcome(r), 0) + 1
        tr = sorted(r["transition"] for r in rr.values() if bnd(r))
        summ[a] = dict(n=n, bound=nb, bound_routed=nbr, fails=fails, transitions=tr)
        print(f"| {a}{' (post hoc)' if ARMS[a].get('post_hoc') else ''} | {n} | {gc.cnt(nb, n)} | {gc.cnt(nbr, n)} | "
              f"{fails or '-'} | {tr or '-'} | {statistics.median(r['acc'] for r in rr.values()):.3f} |")
    g = lambda a, k: summ.get(a, {}).get(k, 0)
    valid = g("ORACLE_L", "bound") >= 1
    rd = {"validity": f"ORACLE_L {g('ORACLE_L', 'bound')}/2 -> {'VALID' if valid else 'UNTESTED'}"}
    if valid:
        x = g("LATCH_NUDGE", "bound_routed")
        rd["R1"] = "the latch carries the cue" if x >= 9 else "architecture insufficient" if x < 6 else "neither"
        disc = [c for c in ("LATCH", "LATCH_PRIOR") if g(c, "bound") >= g("BASELINE_L", "bound") + 3
                and g(c, "fails").get("KEY", 0) <= 2]
        rd["R2"] = f"discovered ({', '.join(disc)})" if disc else "not discovered"
        for c in ("LATCH", "LATCH_PRIOR"):
            if R.get(c) and R.get("BASELINE_L"):
                b_, c_, pv = s56p.mcn(R[c], R["BASELINE_L"], bnd)
                print(f"  {c} vs BASELINE_L BOUND: {b_} / {c_}, one-sided McNemar p = {pv:.3g}")
    diag = {}
    for a, rr in R.items():
        if not rr:
            continue
        dd = {}
        ends = [r["end"].get("beta") for r in rr.values() if r["end"].get("beta")]
        if ends:
            dd["z_end"] = {k: statistics.median(v[k] for v in ends) for k in ("CTX", "KEY", "VAL", "QUERY")}
            by = {}
            one = next(iter(rr.values()))
            for i, s_ in enumerate(one["stats"]):
                vals = [r["stats"][i]["beta"] for r in rr.values() if i < len(r["stats"]) and r["stats"][i].get("beta")]
                if vals and str(s_["step"]) in ("0", "1200", "2400", "4800", "9600", "14400", "19200", "24000"):
                    by[str(s_["step"])] = {k: round(statistics.median(v[k] for v in vals), 3) for k in ("CTX", "KEY", "VAL")}
            dd["z_by_step"] = by
        dd["margin_val"] = {f"L{l}": statistics.median(r["end"]["layers"][f"L{l}"]["margin"] for r in rr.values())
                            for l in (1, 2, 3)}
        dd["table"] = {f"L{l}": {role: [[round(statistics.median(r["end"]["layers"][f"L{l}"]["table"][role][s][c]
                                                                  for r in rr.values()), 3) for c in range(2)]
                                       for s in range(2)] for role in ("CTX", "KEY", "VAL")} for l in (2, 3)}
        diag[a] = dd
        print(f"  {a}: margin at VAL {dd['margin_val']}" + (f"; z end {dd['z_end']}; z by step {dd['z_by_step']}" if ends else "")
              + f"; table {dd['table']}")
    print("\nREADINGS: " + "; ".join(f"{k}: {v}" for k, v in rd.items()))
    print("\nper run:")
    for a, rr in R.items():
        for s, r in sorted(rr.items()):
            lay = r["end"]["layers"]
            print(f"  {a} {s}: {outcome(r):<16} acc {r['acc']:.3f} trans {r['transition']} stream_acc "
                  f"{[round(v, 2) for v in r['stream_acc']]}" + "".join(
                      f" | L{l} map {lay[f'L{l}']['ch_map_val']} m {lay[f'L{l}']['margin']:.2f} etaS K/V "
                      f"{lay[f'L{l}']['eta_key_by_stream']:.2f}/{lay[f'L{l}']['eta_val_by_stream']:.2f} etaK "
                      f"{lay[f'L{l}']['eta_key_by_key']:.2f}" for l in (1, 2, 3))
                  + (f" | z C/K/V {r['end']['beta']['CTX']:.2f}/{r['end']['beta']['KEY']:.2f}/{r['end']['beta']['VAL']:.2f}"
                     if r["end"].get("beta") else ""))
    with open(os.path.join(gc.G_DIR, "s57_summary.json"), "w") as f:
        json.dump(dict(summary=summ, readings=rd, diag=diag, valid=valid), f, indent=1)


def triggered():
    with open(os.path.join(gc.G_DIR, "s56_hebb_summary.json")) as f:
        return json.load(f)["readings"].get("S57", "").startswith("TRIGGERED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--post-hoc-only", action="store_true", help="run only the post-hoc ORACLE23 arm")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S57 latch")
    if not args.report_only:
        t0 = time.time()
        trig = triggered()
        print(f"  S57 trigger (from S56's summary): {'TRIGGERED' if trig else 'not triggered'}", flush=True)
        assert ec.repro_check(), "repro check failed"
        rows = gc.run_child(CHILD, "checks", {})
        for nm, v in rows:
            print(f"  CHECK S57: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        with open(os.path.join(gc.G_DIR, f"checks_s57_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        only = ["ORACLE23"] if (args.post_hoc_only or not trig) else None
        gc.run_screen(sys.modules[__name__], only=only)
        print(f"  S57 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
