#!/usr/bin/env python
"""
explore_f_reports3.py — EXPLORATORY, not a result. Report tables of Session F's S77 (probe) and S76 (eight streams), with
the readings exactly as each screen's docstring fixes them. Writes explore_out/F/<screen>_tables.md.
"""
import json
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


def _acc_at(r, step):
    for st, a, _ in r.get("curve", []):
        if st == step:
            return a
    return None


def report_s76():
    import math
    import explore_f_eight as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    L = []
    P = L.append
    P("### S76 eight streams — Phase 1, the oracle (seeds 1300-1303; BOUND; Wilson 95%)\n")
    P("| arm | BOUND | per seed: transition (or final acc / DIVERGED) | acc @12000 / @24000 (median) | meets the 2x mark |")
    P("|---|---|---|---|---|")
    R = {a: s.all_records(a) for a in ["O_HEBB"] + list(s.VARIANTS)}
    hebb = R["O_HEBB"]
    meets = {}
    for arm, rs in R.items():
        cells, m = [], 0
        for sd in s.OSEEDS:
            r = rs.get(sd)
            if r is None:
                cells.append(f"{sd} --")
                continue
            if not r.get("ok"):
                cells.append(f"{sd} DIVERGED")
                continue
            cells.append(f"{sd} " + (f"B@{r['transition']}" if r["transition"] is not None else f"{r['acc']:.2f}"))
            if arm != "O_HEBB":
                h = hebb.get(sd)
                th = s.trans(h) if h else math.inf
                tx = s.trans(r)
                m += int(tx < math.inf and (tx <= 2 * th or th == math.inf))
        meets[arm] = m
        nb = sum(1 for r in rs.values() if r.get("ok") and r["transition"] is not None)
        a12 = fc.med([_acc_at(r, 12000) if _acc_at(r, 12000) is not None else (r["acc"] if r.get("ok") and r["transition"] else None)
                      for r in rs.values() if r.get("ok")])
        a24 = fc.med([_acc_at(r, 24000) if _acc_at(r, 24000) is not None else (r["acc"] if r.get("ok") and r["transition"] else None)
                      for r in rs.values() if r.get("ok")])
        P(f"| {arm} | {fc.count_str(nb, len(rs))} | {'; '.join(cells)} | {fmt(a12)} / {fmt(a24)} | "
          f"{'--' if arm == 'O_HEBB' else f'{m}/4'} |")
    complete = all(len(rs) == 4 for rs in R.values())
    w = {a: meets[a] >= 3 for a in s.VARIANTS}
    if not complete:
        r1 = "INCOMPLETE, no reading"
    elif w["O_B025_L2"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B1_RAW"]:
        r1 = "the slowness is the overwrite"
    elif w["O_B1_RAW"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B025_L2"]:
        r1 = "the slowness is the key normalisation"
    else:
        r1 = "neither"
    P(f"\nREADING Phase 1: **{r1}** (seeds meeting the 2x mark: " + ", ".join(f"{a} {meets[a]}/4" for a in s.VARIANTS)
      + "; within 2x = >= 3/4)")
    star = json.load(open(s.STAR)) if os.path.exists(s.STAR) else None
    P(f"\nDELTA* = {star}")
    P("\nAccuracy curves (held-out, every 2400):\n")
    for arm, rs in R.items():
        for sd, r in sorted(rs.items()):
            if r.get("ok"):
                P(f"- {arm} {sd}: " + " ".join(f"{a:.2f}" for st, a, _ in r["curve"] if st % 2400 == 0))
    out = dict(r1=r1, star=star)
    if "HEBB_SPLIT" in s.ARMS:
        RR = {a: fc.runs(s, a) for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
        P("\n### S76 Phase 2, the recipe (seeds 1310-1319; BOUND ROUTED; Wilson 95%)\n")
        P("| arm | n | BOUND ROUTED | BOUND | outcomes | MERGED | splits (total; runs; targets ok) | transitions of BOUND ROUTED |")
        P("|---|---|---|---|---|---|---|---|")
        for arm, rs in RR.items():
            oc = {}
            for r in rs.values():
                oc[r.get("outcome")] = oc.get(r.get("outcome"), 0) + 1
            spl = [c for r in rs.values() for c in r.get("checks", []) if c.get("fired")]
            tr = sorted(r["transition"] for r in rs.values() if r.get("outcome") == "BOUND ROUTED")
            P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(r.get('outcome') == 'BOUND ROUTED' for r in rs.values()), len(rs))} | "
              f"{fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {oc} | "
              f"{sum(str(r.get('outcome')).startswith('MERGED') for r in rs.values())} | {len(spl)}; "
              f"{sum(1 for r in rs.values() if r.get('splits'))}; {sum(bool(c.get('key_ok')) for c in spl)}/{len(spl)} | "
              f"{(str(fc.med(tr)) + ' (' + ', '.join(map(str, tr)) + ')') if tr else '--'} |")
        A = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in RR["DELTA_SPLIT"].items()}
        B = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in RR["HEBB_SPLIT"].items()}
        d = fc.paired(A, B, s.SEEDS)
        vh = sum(1 for r in R["O_HEBB"].values() if r.get("ok") and r["transition"] is not None) >= 2
        vd = star is not None and sum(1 for r in R[star["arm"]].values() if r.get("ok") and r["transition"] is not None) >= 2
        if d["n"] < 10:
            r2 = "INCOMPLETE, no reading"
        elif not (vh and vd):
            r2 = "UNTESTED (an oracle bound < 2/4)"
        elif d["c"] - d["b"] <= 1:
            r2 = "the delta recipe keeps eight streams"
        elif d["c"] >= 4 and d["b"] == 0:
            r2 = "the delta recipe loses eight streams"
        else:
            r2 = "inconclusive"
        P(f"\nREADING Phase 2: **{r2}** (BOUND ROUTED DELTA_SPLIT {d['new']}/{d['n']} vs HEBB_SPLIT {d['old']}/{d['n']}; delta only "
          f"{d['b']}, hebb only {d['c']}; p(hebb > delta) = {d['p_old_gt_new']:.3g}, p(delta > hebb) = {d['p_new_gt_old']:.3g}; "
          f"rules: keeps c − b <= 1, loses c >= 4 and b = 0)")
        P("\nSplits per run (update c*->c0 ok?):\n")
        for arm, rs in RR.items():
            P(f"- {arm}: " + "; ".join(
                f"{sd} {r.get('outcome')}" + (f"@{r['transition']}" if r.get('transition') else f" {r['acc']:.2f}") + " ["
                + ", ".join(f"{c['step']} {c['key_cs']}->{c['key_c0']} {'ok' if c.get('key_ok') else 'NOT ok'}"
                            for c in r.get("checks", []) if c.get("fired")) + "]"
                for sd, r in sorted(rs.items())))
        out["r2"] = r2
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed or diverged records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU "
      f"{meta['cpus']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    out["md"] = _write("eight", L)
    return out
