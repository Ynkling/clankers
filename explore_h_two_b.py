#!/usr/bin/env python
"""
explore_h_two_b.py — EXPLORATORY, not a result. Session H, screen S85: the two-stream key split by prevention rather than
repair. Branch claude/explore-H; seeds 2310-2329 (the specification's 2210-2229 shifted by 100: see explore_h_guard);
outputs under explore_out/H/. Child: explore_h_two_b_child (this branch's modules, S43 / S52's path).

BACKGROUND: S78: at k = S = 2 the exact-copy split erases the gate's partition and the same key split re-forms by the next
check; REF (WIN3_SLOW) 15/20 DISCOVERED, its failures key splits. S49 / S52: the window gate's key splits form within the
first 600 updates, while every non-gate parameter (the memory) is still at 1e-4. S52's W2_KEYHINGE (the hinge's key term on
updates 1-2400) discovered 36/40 on seeds 160-199 under v1, with three boundary cases at eta^2 by key 0.5.

CONFIGURATION: S=2, P=4, k=2, grouped, no conv, 24000 updates (test_window_gate's Part A path = S43's LOCAL3_SLOW), LOCAL3,
SLOW, Hebbian memory; seeds 2310-2329, all arms paired (same initial parameters, window and batches). 1 torch thread per run.
ARMS (explore_h_two_b_child.run_b = S52's run at k = 2 with two switches):
  REF                 WIN3_SLOW.
  SLOWGATE            the gate group (W_in, W_g, window) at 1e-4 for updates 1-1200, then 1e-3; the rest as SLOW.
  KEYHINGE            S52's W2_KEYHINGE: + 1.0 x relu(eta2_key - 0.2) on the read gate at key positions, updates 1-2400.
  SLOWGATE_KEYHINGE   both.
  ORACLE              ceiling_conv (the main line's CEIL_A), seeds 2310-2311 (validity).
OUTCOME: DISCOVERED (explore_common.discovered: BOUND and final VAL cos < 0.5). Failure classes of the unbound runs under v1
(test_router_layout.fail_class) and v2 (fail_class_v2 at ba8901a, copied unchanged), both reported.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 24000; otherwise UNTESTED.
READINGS (fixed now; 20 paired seeds; per arm A in {SLOWGATE, KEYHINGE, SLOWGATE_KEYHINGE}):
  "prevention works" (A)           A beats REF on DISCOVERED with exact one-sided McNemar p < 0.05 (5 vs 0, 7 vs 1, 9 vs 2
                                   or better).
  "the key split only moves" (A)   A's KEY failures under v2 >= REF's KEY failures under v2 - 1.
  Both may hold for one arm; both are printed per arm with the counts.
DESCRIPTIVE: per run, the update at which eta^2 by key of the read gate at KEY positions (S52's dense cells: routing_k's
pooled etak_key_by_key on the run's probe, after updates 0, 25, ..., 1200, 1300, ..., 2400, then every 1200) first exceeds
0.5 (">", the first measured update; "never" if it does not), by arm; and whether it still exceeds 0.5 at the end. The
hinge's firings (KEYHINGE arms).
CHECKS (before any run): explore_common.repro_check; explore_keysplits_child.py equals claude/outside-ideas' file (git blob
SHA); REF|160 equals S43's recorded LOCAL3_SLOW|160 bit for bit through 2400 (curve, statistics, decoder) and S52's
W2_KEYHINGE|160 is reproduced by the copied code (KEYHINGE through run_b) bit for bit through 2400 (curve, statistics,
decoder, dense cells, the hinge's counts); SLOWGATE's groups and lrs through the real path (gate 1e-4 at updates 1-1200,
1e-3 from 1201; the rest 1e-4 through 2400, 1e-3 from 2401) and REF's.
"""

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
import explore_h_two as s78

NAME = "H/two_b"
CHILD = "explore_h_two_b_child"
MAIN = False
IDEA = "prevent the two-stream key split: hold the gate until the memory moves; the hinge's key term; both"
SOURCE = "S78 (repair by copy fails at k=2); S49 / S52 (key splits form within 600 updates); S52's W2_KEYHINGE"
CHANGE = "WIN3_SLOW + gate at 1e-4 for updates 1-1200 / + S52's key term on 1-2400 / both"
PAIRING = "seeds 2310-2329 (shifted from 2210-2229), all arms paired"
LR = 1e-3
ITERS = 24000
SEEDS = tuple(range(2310, 2330))
ORACLE_SEEDS = (2310, 2311)
DEVICES = ("SLOWGATE", "KEYHINGE", "SLOWGATE_KEYHINGE")
ARMS = {a: dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=a, sched="LOCAL3 + SLOW " + a) for a in ("REF",) + DEVICES}
ARMS["ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="ceiling_conv", sched="one Adam 1e-3")
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ITERS))


def check():
    rows = []
    k1 = subprocess.run(["git", "-C", HERE, "hash-object", "explore_keysplits_child.py"], capture_output=True, text=True).stdout.strip()
    k2 = subprocess.run(["git", "-C", HERE, "rev-parse", f"{s78.OI}:explore_keysplits_child.py"], capture_output=True, text=True).stdout.strip()
    rows.append((f"explore_keysplits_child.py {k1[:12]} = claude/outside-ideas' {k2[:12]}", k1 == k2 and len(k1) == 40))
    s43 = next(r for k, r in ec.load_store("window_recipe")["runs"].items() if k.startswith("LOCAL3_SLOW|160|46867e792241|"))
    kh = next(r for k, r in s78.oi_store("two_stream_keysplits").items() if k.startswith("W2_KEYHINGE|160|") and r.get("ok"))
    with ThreadPoolExecutor(max_workers=3) as ex:
        fu = ex.submit(child, "checks", {})
        rr = ex.submit(child, "run", dict(arm="REF", seed=160, iters=2400))
        rk = ex.submit(child, "run", dict(arm="KEYHINGE", seed=160, iters=2400))
        rows += [tuple(x) for x in fu.result()]
        r0, r1 = rr.result(), rk.result()

    def upto(r, ref, t):
        c1 = [c for c in r["curve"] if c[0] <= t]
        s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
        s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
        d1, d2 = r.get("decode") or {}, ref.get("decode") or {}
        d = all(d1[k] == d2[k] for k in d1 if k.isdigit() and int(k) <= t and k in d2)
        return bool(c1) and c1 == [c for c in ref["curve"] if c[0] <= t] and s1 == s2 and d, c1
    eq, c = upto(r0, s43, 2400)
    rows.append((f"REF|160 (run_b) equals S43's recorded LOCAL3_SLOW|160 bit for bit through 2400 (curve {[x[:2] for x in c]})", eq))
    eq, c = upto(r1, kh, 2400)
    dn = {k: v for k, v in (kh.get("dense") or {}).items() if int(k) <= 2400}
    dense_eq = dn == {k: v for k, v in r1["dense"].items() if int(k) <= 2400}
    h1, h2 = r1.get("hinge") or {}, kh.get("hinge") or {}
    hinge_eq = h1.get("fired") == h2.get("fired") and h1.get("eta_at") == h2.get("eta_at") and h1.get("windows") == h2.get("windows")
    rows.append((f"KEYHINGE|160 (run_b, the copied S52 code) equals S52's recorded W2_KEYHINGE|160 (claude/outside-ideas, "
                 f"{kh.get('cpu')}) bit for bit through 2400 (curve {[x[:2] for x in c]}; statistics; decoder; dense cells "
                 f"{dense_eq} ({len(dn)}); the hinge fired {h1.get('fired')} vs {h2.get('fired')}, eta at {h1.get('eta_at')})",
                 eq and dense_eq and hinge_eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def key_onset(r):
    """First measured update with etak_key_by_key > 0.5 at KEY positions, and whether it is > 0.5 at the last measurement."""
    d = sorted(((int(k), v) for k, v in (r.get("dense") or {}).items()), key=lambda x: x[0])
    first = next((t for t, v in d if v["etak_key_by_key"] > 0.5), None)
    return first, (d[-1][1]["etak_key_by_key"] > 0.5 if d else None)


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    arms = ("REF",) + DEVICES
    recs = {f"{a}|{s}": dict(r, _kind="k2") for a in arms for s, r in got[a].items() if r.get("transition") is None}
    cls = s78.classify(recs) if recs else {}
    res["v2_property"] = s78.v2_property(cls) if cls else (True, [], [])
    res["classes"] = cls
    for a in arms:
        rs = got[a]
        n, k = len(rs), sum(ec.discovered(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if ec.discovered(r))
        c1 = dict(Counter(cls[f"{a}|{s}"]["v1"] for s in rs if f"{a}|{s}" in cls))
        c2 = dict(Counter(cls[f"{a}|{s}"]["v2"] for s in rs if f"{a}|{s}" in cls))
        onset = {s: key_onset(r) for s, r in sorted(rs.items())}
        res["arms"][a] = dict(n=n, k=k, band=hc.band(k, n), wilson=hc.wilson(k, n), v1=c1, v2=c2, key_v2=c2.get("KEY", 0),
                              bound_not=sum(1 for r in rs.values() if r.get("transition") is not None and not ec.discovered(r)),
                              trans_med=c16.med(tr), trans=tr, onset=onset, complete=n == len(SEEDS),
                              seeds_ok=sorted(s for s, r in rs.items() if ec.discovered(r)),
                              hinge_fired=[(r.get("hinge") or {}).get("fired") for r in rs.values()] if "HINGE" in a else None)
    D = {a: {s: ec.discovered(r) for s, r in got[a].items()} for a in arms}
    for a in DEVICES:
        res["cmp"][a] = hc.paired(D[a], D["REF"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in arms) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {a: "INCOMPLETE, no reading" for a in DEVICES}
    elif not valid:
        rd = {a: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for a in DEVICES}
    else:
        rk = res["arms"]["REF"]["key_v2"]
        rd = {}
        for a in DEVICES:
            c = res["cmp"][a]
            ka = res["arms"][a]["key_v2"]
            rd[a] = ((("prevention works" if c["p"] < 0.05 else "prevention not shown")
                      + f" ({a} vs REF {c['b']} vs {c['c']}, one-sided p = {c['p']:.3g}); ")
                     + ("the key split only moves" if ka >= rk - 1 else "the key split does not only move")
                     + f" (KEY failures v2: {a} {ka}, REF {rk})")
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
