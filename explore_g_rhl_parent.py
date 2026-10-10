#!/usr/bin/env python
"""
explore_g_rhl_parent.py — EXPLORATORY, not a result. Parent-side helpers of Session G's follow-up screens S71-S73
(outcomes, paired statistics, the report tables and the driver). Run code: explore_g_rhl_child.

OUTCOMES (fixed for S71-S73; from a run record):
  BOUND          transition is not None (explore_common.bound: the run ended >= 0.95 from its transition on).
  ROUTED*@end    F's: end margin >= 0.9 and end eta^2 by stream at KEY > 0.9 (explore_f_tasks.rand_header_routing_stats on
                 the model's returned gate; explore_common2.routed_star's thresholds).
  DISCOVERED     F's: BOUND and the end VAL cos < 0.5 (test_router_discovery.DISC_COS).
  ROUTED-BOUND   BOUND and ROUTED*@end. The readings use this (S72, S73) or BOUND (S71).
  BOUND ROUTED   (mine, printed beside it): BOUND, the VAL stream -> channel map one-to-one at every layer, every stream's
                 held-out accuracy >= 0.9.
  failure class  of an unbound k = 2 run: test_router_layout.fail_class on the end statistics; MERGED if STREAM-PARTIAL and
                 the layer-3 VAL map is not one-to-one.
STATISTICS: Wilson 95% and bands with every count (explore_g_common); paired by seed, exact one-sided McNemar
(test_conv_lr.mcnemar_greater: P(Bin(b + c, 1/2) >= b), b = seeds where the first arm alone succeeds); "A beats B" means
p < 0.05. "A >= B - 1 pair" means (B only) - (A only) <= 1 on the paired seeds.
PLATEAU: an arm's unbound runs; their held-out accuracy by (query stream, block, pair index) is reported (median over runs).
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

CHILD = "explore_g_rhl_child"
ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9
DISC_COS = 0.5


def bound(r):
    return r["transition"] is not None


def routed_star(r):
    e = r["end"]
    return bool(e.get("margin", 0.0) >= ROUTE_MARGIN and e.get("eta_key_by_stream", 0.0) > ROUTE_ETA)


def discovered(r):
    return bool(bound(r) and r["val_cos"] < DISC_COS)


def routed_bound(r):
    return bool(bound(r) and routed_star(r))


def bound_routed_mine(r):
    lay = r["end"].get("layers")
    return bool(bound(r) and lay and all(lay[f"L{l}"]["one_to_one_val"] for l in (1, 2, 3)) and min(r["stream_acc"]) >= 0.9)


def outcome(r):
    if r.get("k", 2) < 2:
        return "BOUND" if bound(r) else "unbound"
    if bound(r):
        return "ROUTED-BOUND" if routed_bound(r) else "BOUND unrouted"
    fc = ec.fail_class(r)
    if fc == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return fc


def paired(ra, rb, f):
    seeds = sorted(set(ra) & set(rb))
    b = sum(1 for s in seeds if f(ra[s]) and not f(rb[s]))
    c = sum(1 for s in seeds if f(rb[s]) and not f(ra[s]))
    return b, c, gc.mcnemar_greater(b, c), len(seeds)


def beats(ra, rb, f):
    b, c, p, n = paired(ra, rb, f)
    return p < 0.05, (b, c, p, n)


def not_worse_1(ra, rb, f):
    """A >= B - 1 pair: (B only) - (A only) <= 1."""
    b, c, p, n = paired(ra, rb, f)
    return c - b <= 1, (b, c, p, n)


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else float("nan")


def count_table(R, arms):
    L = ["| arm | n | BOUND | ROUTED-BOUND | ROUTED*@end | DISCOVERED | BOUND ROUTED (mine) | unbound classes | transitions | median acc |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rr = R.get(a, {})
        n = len(rr)
        if not n:
            L.append(f"| {a} | 0 | not run | | | | | | | |")
            continue
        k1 = next(iter(rr.values())).get("k", 2) < 2
        cls = {}
        for r in rr.values():
            if not bound(r):
                o = outcome(r)
                cls[o] = cls.get(o, 0) + 1
        tr = sorted(r["transition"] for r in rr.values() if bound(r))
        L.append(f"| {a} | {n} | {gc.cnt(sum(bound(r) for r in rr.values()), n)} | "
                 + (" -- | -- | -- | -- | " if k1 else
                    f"{gc.cnt(sum(routed_bound(r) for r in rr.values()), n)} | {sum(routed_star(r) for r in rr.values())}/{n} | "
                    f"{sum(discovered(r) for r in rr.values())}/{n} | {sum(bound_routed_mine(r) for r in rr.values())}/{n} | ")
                 + f"{', '.join(f'{k} {v}' for k, v in sorted(cls.items())) or '-'} | "
                 + (f"median {med(tr):.0f} ({min(tr)}-{max(tr)})" if tr else "-")
                 + f" | {med([r['acc'] for r in rr.values()]):.3f} |")
    return "\n".join(L)


def diagnostics(R, arms):
    L = []
    for a in arms:
        rr = R.get(a, {})
        if not rr:
            continue
        one = next(iter(rr.values()))
        L.append(f"- {a}:")
        if one.get("k", 2) >= 2 and one["end"].get("layers"):
            mv = {f"L{l}": med([r["end"]["layers"][f"L{l}"]["margin"] for r in rr.values()]) for l in (1, 2, 3)}
            L.append("  margin at VAL by layer (median) " + ", ".join(f"{k} {v:.2f}" for k, v in mv.items()))
            for l in (1, 2, 3):
                tab = {role: [[round(med([r["end"]["layers"][f"L{l}"]["table"][role][s][c] for r in rr.values()]), 2)
                               for c in range(2)] for s in range(2)] for role in ("CTX", "KEY", "VAL")}
                L.append(f"  contingency L{l} (mean g; rows stream 0/1, cols ch 0/1): " +
                         "; ".join(f"{role} {t}" for role, t in tab.items()))
        if one["end"].get("beta"):
            steps = [s_["step"] for s_ in one["stats"]]
            sel = [i for i, s_ in enumerate(steps) if str(s_) in ("0", "1200", "2400", "4800", "9600", "14400", "19200", "24000")]
            parts = []
            for i in sel:
                vals = [r["stats"][i]["beta"] for r in rr.values() if i < len(r["stats"]) and r["stats"][i].get("beta")]
                if vals:
                    parts.append(f"{steps[i]} " + "/".join(f"{med([v[k] for v in vals]):.2f}" for k in ("CTX", "KEY", "VAL")))
            ends = [r["end"]["beta"] for r in rr.values()]
            parts.append("end " + "/".join(f"{med([v[k] for v in ends]):.2f}" for k in ("CTX", "KEY", "VAL")))
            L.append("  write rate (beta^R or z) CTX/KEY/VAL by step (median; runs stop early when bound): " + "; ".join(parts))
        pr = {k: med([r["probe"][k]["acc"] for r in rr.values()]) for k in one["probe"]}
        L.append("  in-run probe (median held-out acc) " + ", ".join(f"{k} {v:.2f}" for k, v in pr.items()))
        unb = [r for r in rr.values() if not bound(r)]
        if unb:
            cells = sorted(set(k for r in unb for k in r["acc_by_cell"]))
            L.append(f"  plateau ({len(unb)} unbound runs; acc {med([r['acc'] for r in unb]):.2f}): acc by (stream|block|index) "
                     + ", ".join(f"{c} {med([r['acc_by_cell'][c]['acc'] for r in unb if c in r['acc_by_cell']]):.2f}" for c in cells))
        wo = [r["write_order"] for r in rr.values()]
        L.append(f"  write order: acc last-written {med([w['acc_last'] for w in wo]):.2f}, not-last {med([w['acc_notlast'] for w in wo]):.2f}")
    return "\n".join(L)


def per_run(R, arms):
    L = []
    for a in arms:
        for s, r in sorted(R.get(a, {}).items()):
            lay = r["end"].get("layers") or {}
            pr = r["probe"]
            L.append(f"  {a} {s}: {outcome(r):<15} acc {r['acc']:.3f} trans {r['transition']} margin {r['end'].get('margin', float('nan')):.2f} "
                     f"etaS/K@KEY {r['end'].get('eta_key_by_stream', float('nan')):.2f}/{r['end'].get('eta_key_by_key', float('nan')):.2f}"
                     + "".join(f" | L{l} map {lay[f'L{l}']['ch_map_val']} m {lay[f'L{l}']['margin']:.2f}" for l in (1, 2, 3) if lay)
                     + f" | probe L3 {pr['L3|K']['acc']:.2f}/{pr['L3|V']['acc']:.2f}"
                     + (f" | w C/K/V {r['end']['beta']['CTX']:.2f}/{r['end']['beta']['KEY']:.2f}/{r['end']['beta']['VAL']:.2f}"
                        if r["end"].get("beta") else ""))
    return "\n".join(L)


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return f"acc {r['acc']:.3f} trans {str(r['transition']):>5} {outcome(r)}"


def drive(screen, tag):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner(f"Session G {tag}")
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = gc.run_child(CHILD, "checks", dict(screen=tag.lower()))
        for nm, v in rows:
            print(f"  CHECK {tag}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        with open(os.path.join(gc.G_DIR, f"checks_{tag.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        gc.run_screen(screen)
        print(f"  {tag} runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    screen.report()
