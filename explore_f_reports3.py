#!/usr/bin/env python
"""
explore_f_reports3.py — EXPLORATORY, not a result. Report tables of Session F's S77 (probe) and S76 (eight streams), with
the readings exactly as each screen's docstring fixes them. Writes explore_out/F/<screen>_tables.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc
from explore_f_reports import fmt, bound, trans_str, acc_med, store_meta

FEATS = ("emb", "L1", "L2", "L3", "final")


def _runs(s):
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    return {arm: fc.runs(s, arm) for arm in s.ARMS}


def _write(name, L):
    md = "\n".join(L)
    print(md)
    open(os.path.join(fc.F_DIR, f"{name}_tables.md"), "w").write(md + "\n")
    return md


def l3(r):
    return 0.5 * (r["probe"]["L3|K"]["acc"] + r["probe"]["L3|V"]["acc"])


def report_s77():
    import explore_f_probe as s
    R = _runs(s)
    L = []
    P = L.append
    P("### S77 probe — counts and probe accuracy (held-out; chance 0.5)\n")
    P("| arm | n | reruns equal to records | BOUND | " + " | ".join(f"{f} K/V" for f in FEATS) + " | at init: L3 K/V |")
    P("|---|---|---|---|" + "---|" * len(FEATS) + "---|")
    for arm, rs in R.items():
        n = len(rs)
        cells = []
        for f in FEATS:
            cells.append(f"{fmt(fc.med([r['probe'][f + '|K']['acc'] for r in rs.values()]))} / "
                         f"{fmt(fc.med([r['probe'][f + '|V']['acc'] for r in rs.values()]))}")
        P(f"| {arm} | {n}/10 | {sum(bool(r.get('repro')) for r in rs.values())}/{n} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), n)} | " + " | ".join(cells) + " | "
          f"{fmt(fc.med([r['probe_init']['L3|K']['acc'] for r in rs.values()]))} / "
          f"{fmt(fc.med([r['probe_init']['L3|V']['acc'] for r in rs.values()]))} |")
    P("\n(medians over the arm's runs; full per-run values in the store)\n")
    P("By outcome (medians):\n")
    groups = {}
    for arm, rs in R.items():
        for sd, r in rs.items():
            groups.setdefault((arm, bound(r)), []).append(r)
    for (arm, b), rs in sorted(groups.items()):
        P(f"- {arm} {'bound' if b else 'unbound'} (n {len(rs)}): " + "; ".join(
            f"{f} {fmt(fc.med([r['probe'][f + '|K']['acc'] for r in rs]))}/{fmt(fc.med([r['probe'][f + '|V']['acc'] for r in rs]))}"
            for f in FEATS))
    complete = all(len(rs) == 10 for rs in R.values())
    bd = [r for arm in ("S53_DELTA", "S59_GDN") for r in R[arm].values() if bound(r)]
    tag = [r for r in bd if r["probe"]["L3|K"]["acc"] >= 0.9 and r["probe"]["L3|V"]["acc"] >= 0.9
           and r["probe"]["L1|K"]["acc"] <= 0.6 and r["probe"]["L1|V"]["acc"] <= 0.6]
    allb = [r for rs in R.values() for r in rs.values() if bound(r)]
    allu = [r for rs in R.values() for r in rs.values() if not bound(r)]
    mb, mu = fc.med([l3(r) for r in allb]), fc.med([l3(r) for r in allu])
    rd = []
    if not complete:
        rd = ["INCOMPLETE, no reading"]
    else:
        if len(tag) >= 8:
            rd.append("the delta stack tags its keys by stream")
        if mb is not None and mu is not None and mb - mu >= 0.2:
            rd.append("the tag is where binding is")
        if not rd:
            rd = ["descriptive (neither reading applies)"]
    P(f"\nREADING S77: **{'; '.join(rd)}** (bound delta runs with L3 >= 0.9 at K and V and L1 <= 0.6 at K and V: {len(tag)} of "
      f"{len(bd)}; rule >= 8. Median L3 (mean K, V): bound {fmt(mb)} (n {len(allb)}), unbound {fmt(mu)} (n {len(allu)}); "
      f"difference {fmt(None if mb is None or mu is None else mb - mu)}; rule >= 0.2)")
    P("\nPer run (L1 K/V, L2 K/V, L3 K/V; B = bound):\n")
    for arm, rs in R.items():
        P(f"- {arm}: " + "; ".join(
            f"{sd}{' B' if bound(r) else ''} " + " ".join(
                f"{r['probe'][f + '|K']['acc']:.2f}/{r['probe'][f + '|V']['acc']:.2f}" for f in ("L1", "L2", "L3"))
            for sd, r in sorted(rs.items())))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    return dict(rd=rd, md=_write("probe", L))
