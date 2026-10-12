#!/usr/bin/env python
"""
explore_h_dose.py — EXPLORATORY, not a result. Session H, round 5 (3), screen S87c: the key term's dose at two streams —
the weight at 0.3x and 3x and the window at 1200, each against the drafted constants (weight 1.0, window 2400, from S52
and S85; explore_out/H/spec_draft_two_keyterm.md). Branch claude/explore-H; seeds 2630-2639 (round-5 block 2600-2659,
unused elsewhere); outputs under explore_out/H/. Child: explore_h_dose_child (this branch's modules, S85's path).

CONFIGURATION: S=2, P=4, k=2, grouped, no conv, 24000 updates; test_window_gate's Part A path (WIN3_SLOW = S43's
LOCAL3_SLOW: LOCAL3, SLOW, Hebbian), + S52's key term L = lam * relu(eta2_key - 0.2) on the read gate at key positions,
weight lam on updates 1..window and 0 after; seeds 2630-2639, all arms paired. 1 torch thread per run.
ARMS:
  KEYHINGE   lam 1.0, window 2400 (the drafted constants; S85's arm).
  LAM03      lam 0.3, window 2400.
  LAM3       lam 3.0, window 2400.
  WIN1200    lam 1.0, window 1200.
  ORACLE     ceiling_conv, seeds 2630-2631 (validity).
OUTCOME: DISCOVERED (explore_common.discovered). Failure classes v1 and v2.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 24000; otherwise UNTESTED.
READINGS (fixed now; 10 paired seeds; per variant V vs KEYHINGE: "V only" = V discovered and KEYHINGE not):
  "robust"   if every variant is within 1 pair of KEYHINGE: |V only - KEYHINGE only| <= 1 for LAM03, LAM3 and WIN1200.
  Otherwise "not robust", printing each failing variant with its discordant pairs.
  AMENDMENT RULE: the draft's constant is changed only if a variant beats KEYHINGE by >= 4 vs 0 (V only >= 4, KEYHINGE
  only = 0); the draft then says so with the date. No other outcome changes the draft.
DESCRIPTIVE: transitions; the update at which eta^2 by key at KEY positions first exceeds 0.5 and whether it does at the end
(S85's measure); the term's firings; the maximum eta^2 by key at KEY positions over updates 25-window.
CHECKS (before any run): explore_common.repro_check; the term's weight at S52's logged updates in every arm (lam at 1; lam
or 0 at 2399-2400 by window; 0 at 2401-2402); KEYHINGE (the drafted constants, through this child) equals S85's recorded
KEYHINGE|2310 bit for bit through 2400 (curve, statistics, decoder, dense cells, the hinge's counts); LAM03|2310's curve at
1200 printed beside it (not asserted).
"""

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc
import explore_h_two as s78
import explore_h_two_b as s85

NAME = "H/dose"
CHILD = "explore_h_dose_child"
MAIN = False
IDEA = "the key term's dose at two streams: weight 0.3x / 3x, window 1200, vs the drafted 1.0 / 2400"
SOURCE = "spec_draft_two_keyterm.md (constants from S52 and S85)"
CHANGE = "WIN3_SLOW + the key term at lam 0.3 / 3.0 / window 1200"
PAIRING = "seeds 2630-2639, all arms paired"
LR = 1e-3
ITERS = 24000
SEEDS = tuple(range(2630, 2640))
ORACLE_SEEDS = (2630, 2631)
VARIANTS = ("LAM03", "LAM3", "WIN1200")
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="WIN3_SLOW + key term " + a)
        for a in ("KEYHINGE",) + VARIANTS}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="ceiling_conv", sched="one Adam 1e-3")
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    rec = None
    for k, r in ec.load_store(s85.NAME)["runs"].items():
        if k.startswith("KEYHINGE|2310|") and r.get("ok"):
            rec = r
    rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        fu = ex.submit(child, "checks", {})
        rk = ex.submit(child, "run", dict(arm="KEYHINGE", seed=2310, iters=2400))
        rl = ex.submit(child, "run", dict(arm="LAM03", seed=2310, iters=1200))
        rows += [tuple(x) for x in fu.result()]
        r1, r3 = rk.result(), rl.result()
    c1 = [c for c in r1["curve"] if c[0] <= 2400]
    s1 = [s for s in r1["stats"] if s["step"] != "end" and s["step"] <= 2400]
    s2 = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= 2400]
    dn = {k: v for k, v in rec["dense"].items() if int(k) <= 2400}
    h1, h2 = r1["hinge"], rec["hinge"]
    d1, d2 = r1.get("decode") or {}, rec.get("decode") or {}
    eq = (c1 == [c for c in rec["curve"] if c[0] <= 2400] and s1 == s2 and dn == {k: v for k, v in r1["dense"].items() if int(k) <= 2400}
          and h1["fired"] == h2["fired"] and h1["eta_at"] == h2["eta_at"] and h1["windows"] == h2["windows"]
          and all(d1[k] == d2[k] for k in d1 if k.isdigit() and int(k) <= 2400 and k in d2))
    rows.append((f"KEYHINGE|2310 (the drafted constants through this child) equals S85's recorded KEYHINGE|2310 bit for bit through 2400 "
                 f"(curve {[x[:2] for x in c1]}; statistics; decoder; dense cells; hinge fired {h1['fired']} vs {h2['fired']})", eq))
    print(f"  (printed) LAM03|2310 at 1200: {[x[:2] for x in r3['curve']]} (KEYHINGE: {[x[:2] for x in c1[:1]]})", flush=True)
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
    arms = ("KEYHINGE",) + VARIANTS
    recs = {f"{a}|{s}": dict(r, _kind="k2") for a in arms for s, r in got[a].items() if r.get("transition") is None}
    cls = s78.classify(recs) if recs else {}
    res["v2_property"] = s78.v2_property(cls) if cls else (True, [], [])
    res["classes"] = cls
    for a in arms:
        rs = got[a]
        n, k = len(rs), sum(ec.discovered(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if ec.discovered(r))
        win = 1200 if a == "WIN1200" else 2400
        mx = [max((v["etak_key_by_key"] for t, v in r["dense"].items() if 25 <= int(t) <= win), default=None) for r in rs.values()]
        res["arms"][a] = dict(
            n=n, k=k, band=hc.band(k, n), wilson=hc.wilson(k, n),
            v1=dict(Counter(cls[f"{a}|{s}"]["v1"] for s in rs if f"{a}|{s}" in cls)),
            v2=dict(Counter(cls[f"{a}|{s}"]["v2"] for s in rs if f"{a}|{s}" in cls)),
            bound_not=sum(1 for r in rs.values() if r.get("transition") is not None and not ec.discovered(r)),
            trans_med=c16.med(tr), trans=tr, onset={s: s85.key_onset(r) for s, r in sorted(rs.items())},
            fired=[(r.get("hinge") or {}).get("fired") for r in rs.values()], eta_max=max(x for x in mx if x is not None) if mx else None,
            complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if ec.discovered(r)))
    D = {a: {s: ec.discovered(r) for s, r in got[a].items()} for a in arms}
    for v in VARIANTS:
        res["cmp"][v] = hc.paired(D[v], D["KEYHINGE"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in arms) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {"S87c": "INCOMPLETE, no reading"}
    elif not valid:
        rd = {"S87c": f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        C = res["cmp"]
        fails = [v for v in VARIANTS if abs(C[v]["b"] - C[v]["c"]) > 1]
        rd = {"robust": ("robust" if not fails else "not robust: " + "; ".join(
            f"{v} {C[v]['b']} vs {C[v]['c']} (by {C[v]['b'] - C[v]['c']:+d})" for v in fails))
              + " (" + "; ".join(f"{v} only {C[v]['b']}, KEYHINGE only {C[v]['c']}" for v in VARIANTS) + ")"}
        amend = [v for v in VARIANTS if C[v]["b"] >= 4 and C[v]["c"] == 0]
        rd["amend"] = ("amend the draft: " + ", ".join(amend)) if amend else "no variant beats the drafted constants by >= 4 vs 0: the draft stands"
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
