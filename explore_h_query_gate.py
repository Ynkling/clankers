#!/usr/bin/env python
"""
explore_h_query_gate.py — EXPLORATORY, not a result. Session H, screen S62: the gate at the QUERY's KEY position, on S58's
and S58b's saved runs. Descriptive (no reading); one table. Child: explore_h_query_child (main line's modules at 9c5939e).

WHY: S58 found bound-routed runs whose stream -> channel map at body KEY positions differs from the map at body VAL
positions (34 of SWITCH's 36; 2-5 in other arms). The answer is read at the query's KEY position (CTX one back) from what
was written at VAL positions, so the query map, not the body KEY map, is the one that has to match the VAL map.

RUNS: every saved S58 run (explore_out/H/collapse_results.json, code 9c9172a89d60) and S58b run (eight_results.json, code
91494b36315e) with a transition (BOUND, routed or not), the ORACLE arms excluded — rerun with the screen's own unchanged
child code plus a measurement callback after each evaluation (explore_h_query_child; eval mode, no gradient, no RNG). The
saved runs hold no weights, hence the reruns. A rerun counts only if it reproduces its saved run's curve and every
statistic BIT FOR BIT (this CPU is the one the saved runs were made on); the count of reproduced runs is printed.
MEASURED at the run's last evaluation (its stop), on the run's routing probe: the query map (stream -> argmax channel of the
mean read gate at the query KEY position), the body VAL map (routing_k's), the body KEY map; per run: query map = VAL map,
= KEY map, both, or neither; margin_val (the query read gate's mass on its stream's VAL-map channel minus its mean mass on
the other streams' VAL-map channels) beside the harness's margin.
TABLE (per screen and arm): bound runs; reproduced; query = both / VAL only / KEY only / neither; among BOUND ROUTED and
among BOUND NOT routed; median margin_val and median harness margin.
CHECKS: explore_common.repro_check; on the perfect gate (both configurations) the three maps are the identity and
margin_val = 1; the reruns' bit-for-bit reproduction (every run).
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

NAME = "H/query_gate"
CHILD = "explore_h_query_child"
MAIN = True
IDEA = "the gate at the query's KEY position vs the body VAL and KEY maps, on S58/S58b's bound runs"
SOURCE = "S58's KEY-vs-VAL map differences (SWITCH 34/36)"
CHANGE = "reruns of saved runs with a query-gate measurement"
PAIRING = "each rerun against its saved run (bit for bit)"
SRC = {"collapse": ("H/collapse", "9c9172a89d60", 28800), "eight": ("H/eight", "91494b36315e", 43200)}
ARMS = {}
CUT = set()


def saved(src):
    name, sha, _ = SRC[src]
    st = ec.load_store(name)["runs"]
    out = {}
    for k, r in st.items():
        a, s, h, cpu = k.split("|", 3)
        if h == sha and r.get("ok") and a != "ORACLE":
            out.setdefault(a, {})[int(s)] = r
    return out


def _build():
    for src in SRC:
        for a, rs in sorted(saved(src).items()):
            seeds = tuple(sorted(s for s, r in rs.items() if r.get("transition") is not None))
            if seeds:
                ARMS[f"{src}.{a}"] = dict(opt="adam", lr=1e-3, iters=SRC[src][2], seeds=seeds, label=f"rerun of {src} {a}",
                                          sched="as saved")


_build()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    src, a = arm.split(".", 1)
    return child("run", dict(src=src, arm=a, seed=seed, iters=SRC[src][2]))


def check():
    rows = [tuple(x) for x in child("checks", {})]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def reproduces(r, ref):
    return r["curve"] == ref["curve"] and r["stats"] == ref["stats"]


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    rows, per = [], {}
    for arm in ARMS:
        src, a = arm.split(".", 1)
        ref = saved(src)[a]
        rs = c16.runs_of(me, arm, store=store)
        rep = {s: reproduces(r, ref[s]) for s, r in rs.items()}
        cnt = {"BOUND ROUTED": Counter(), "BOUND NOT routed": Counter()}
        mv, mh = [], []
        for s, r in rs.items():
            if not rep[s]:
                continue
            q = r["query"]["end"]
            kind = "both" if q["eq_val"] and q["eq_key"] else "VAL only" if q["eq_val"] else "KEY only" if q["eq_key"] else "neither"
            o = "BOUND ROUTED" if hc.br(ref[s]) else "BOUND NOT routed"
            cnt[o][kind] += 1
            mv.append(q["margin_val"])
            mh.append(ref[s]["end"].get("margin"))
            per[(arm, s)] = dict(kind=kind, outcome=o, q=q)
        rows.append(dict(arm=arm, n=len(ARMS[arm]["seeds"]), got=len(rs), reproduced=sum(rep.values()), cnt=cnt,
                         margin_val=c16.med(mv), margin=c16.med(mh)))
    res = dict(rows=rows, per=per, complete=all(r["got"] == r["n"] for r in rows), valid=True,
               readings={"S62": "descriptive"})
    return res, None
