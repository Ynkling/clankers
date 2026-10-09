#!/usr/bin/env python
"""
explore_f_reports2.py — EXPLORATORY, not a result. Report tables of Session F's follow-up screens (S68, S69, S70), with
the readings exactly as each screen's docstring fixes them. Writes explore_out/F/<screen>_tables.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc
from explore_f_reports import fmt, bound, trans_str, acc_med, disc, routed_star_end, fails, store_meta


def _runs(s):
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    return {arm: fc.runs(s, arm) for arm in s.ARMS}


def _write(name, L):
    md = "\n".join(L)
    print(md)
    open(os.path.join(fc.F_DIR, f"{name}_tables.md"), "w").write(md + "\n")
    return md


def _count_rows(s, R, P, extra=None):
    P("| arm | n | BOUND | transitions: median (all) | final acc median | write order acc last / not-last |"
      + (" " + extra[0] + " |" if extra else ""))
    P("|---|---|---|---|---|---|" + ("---|" if extra else ""))
    for arm in s.ARMS:
        rs = R[arm]
        n = len(rs)
        wo = [r.get("write_order") or {} for r in rs.values()]
        P(f"| {arm} | {n}/{len(s.ARMS[arm]['seeds'])} | {fc.count_str(sum(bound(r) for r in rs.values()), n)} | "
          f"{trans_str(rs)} | {acc_med(rs)} | {fmt(fc.med([w.get('acc_last') for w in wo]))} / "
          f"{fmt(fc.med([w.get('acc_notlast') for w in wo]))} |" + (f" {extra[1](arm, rs)} |" if extra else ""))


def _decay_lines(R, arms, P):
    for arm in arms:
        dd = [r.get("decay_by_role") for r in R[arm].values() if r.get("decay_by_role")]
        if not dd:
            continue
        keys = [k for k in dd[0][0] if k != "beta"]
        P(f"- {arm}: " + "; ".join(f"L{i + 1} " + ", ".join(
            f"{k} " + "/".join(fmt(fc.med([x[i][k][role] for x in dd]), 3) for role in ("ctx", "key", "val")) for k in keys)
            for i in range(len(dd[0]))))


def report_s68():
    import explore_f_premise as s
    R = _runs(s)
    L = []
    P = L.append
    P("### S68 premise — counts (BOUND; Wilson 95%; band)\n")
    _count_rows(s, R, P)
    val = {}
    for S in (4, 8):
        rs = R[f"S68_ORACLE_S{S}"]
        nb = sum(bound(r) for r in rs.values())
        val[S] = (nb >= 1) if len(rs) == 2 else None
        P(f"\nValidity S={S}: S68_ORACLE_S{S} {nb}/{len(rs)} (transitions {[r['transition'] for r in rs.values()]}) -> "
          f"{'VALID' if val[S] else ('incomplete' if val[S] is None else 'NOT VALID')}")
    k4 = sum(bound(r) for r in R["S68_SINGLE_S4"].values())
    k8 = sum(bound(r) for r in R["S68_SINGLE_S8"].values())
    n4, n8 = len(R["S68_SINGLE_S4"]), len(R["S68_SINGLE_S8"])
    if n4 < 10 or n8 < 10 or None in val.values():
        rd = "INCOMPLETE, no reading"
    elif not all(val.values()):
        rd = "UNTESTED (an oracle did not bind)"
    elif k8 >= 7:
        rd = "context-tagged keys scale"
    elif k8 <= 3 and k4 >= 7:
        rd = "the partition's advantage is in the number of streams"
    else:
        rd = "neither"
    P(f"\nREADING S68: **{rd}** (SINGLE_DELTA S=4 {k4}/{n4}, S=8 {k8}/{n8}; rules: scale >= 7/10 at S=8; number of streams "
      f"<= 3/10 at S=8 with >= 7/10 at S=4)")
    for S, (arm, br_, n, b) in (("4", s.REC["S4"]), ("8", s.REC["S8"])):
        k = k4 if S == "4" else k8
        nn = n4 if S == "4" else n8
        P(f"- S={S}: one delta channel BOUND {fc.count_str(k, nn)} vs recorded gated Hebbian S54 {arm} BOUND ROUTED "
          f"{fc.count_str(br_, n)} (BOUND {b}/{n}); counts only, other seeds; Fisher one-sided p(gated > single) = "
          f"{fc.fisher_greater(br_, n, k, nn):.3g}")
    P("\nPer seed (B@transition or final acc): " + "; ".join(
        f"{arm} " + ", ".join((f"{sd} B@{r['transition']}" if bound(r) else f"{sd} {r['acc']:.2f}") for sd, r in sorted(R[arm].items()))
        for arm in ("S68_SINGLE_S4", "S68_SINGLE_S8")))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    return dict(rd=rd, md=_write("premise", L))


def report_s69():
    import explore_f_hebb_decay as s
    R = _runs(s)
    L = []
    P = L.append
    P("### S69 forget gate on a single Hebbian channel — counts (BOUND; Wilson 95%; band)\n")
    _count_rows(s, R, P, extra=("S59 delta, same seeds", lambda arm, rs: f"{s.S59_DELTA.get(arm[4:], '--')}/10"))
    rs = R["S69_ORACLE"]
    nb = sum(bound(r) for r in rs.values())
    v = (nb >= 1) if len(rs) == 2 else None
    P(f"\nValidity: S69_ORACLE {nb}/{len(rs)} (transitions {[r['transition'] for r in rs.values()]}) -> "
      f"{'VALID' if v else ('incomplete' if v is None else 'NOT VALID')}")
    k = {d: sum(bound(r) for r in R[f"S69_{d}"].values()) for d in ("FIXED", "ROUTED", "GDN")}
    complete = all(len(R[f"S69_{d}"]) == 10 for d in k)
    if not complete or v is None:
        rd = "INCOMPLETE, no reading"
    elif not v:
        rd = "UNTESTED (the oracle did not bind)"
    else:
        needs = k["ROUTED"] <= k["FIXED"] + 1 and k["GDN"] <= k["FIXED"] + 1
        alone = k["ROUTED"] >= 8 or k["GDN"] >= 8
        rd = ("the forget gate needs the delta rule" if needs and not alone else "the forget gate alone suffices" if alone
              and not needs else "both rules hold" if needs and alone else "neither")
    P(f"\nREADING S69: **{rd}** (Hebbian FIXED {k['FIXED']}/10, ROUTED {k['ROUTED']}/10, GDN {k['GDN']}/10; rules: needs "
      f"delta if ROUTED and GDN <= FIXED + 1; alone suffices if either >= 8/10)")
    P("\nPaired with S59's delta single channel on the same seeds and batches (BOUND; delta only / hebb only): " + "; ".join(
        f"{d}: S59 delta {s.S59_DELTA[d]}/10 vs Hebbian {k[d]}/10" for d in k))
    P("\nLearned decay at the end (median α per layer at CTX/KEY/VAL):\n")
    _decay_lines(R, list(s.ARMS), P)
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    return dict(rd=rd, md=_write("hebb_decay", L))


def report_s70():
    import explore_f_rh_lite as s
    R = _runs(s)
    d = s.decision()
    L = []
    P = L.append
    P("### S70 RandHeaderTask-lite — counts (Wilson 95%; band)\n")
    P(f"Phase 1 (oracle first): {d}\n")
    P("| arm | n | BOUND | DISCOVERED | ROUTED*@end | failures (unbound) | transitions: median (all) | final acc median |")
    P("|---|---|---|---|---|---|---|---|")
    for arm in s.ARMS:
        rs = R[arm]
        n = len(rs)
        o = "ORACLE" in arm
        P(f"| {arm} | {n}/{len(s.ARMS[arm]['seeds'])} | {fc.count_str(sum(bound(r) for r in rs.values()), n)} | "
          f"{'--' if o else fc.count_str(sum(disc(r) for r in rs.values()), n)} | "
          f"{'--' if o else fc.count_str(sum(routed_star_end(r) for r in rs.values()), n)} | {'--' if o else fails(rs)} | "
          f"{trans_str(rs)} | {acc_med(rs)} |")
    if "S70_HEBB" in R and len(R["S70_HEBB"]) == len(s.SEEDS):
        h = R["S70_HEBB"]
        P(f"\n**BASELINE for Session G (S70_HEBB, LOCAL3 + SLOW, Hebbian, RandHeaderTask-lite, {d.get('budget')} updates, "
          f"seeds 330-349): DISCOVERED {fc.count_str(sum(disc(r) for r in h.values()), 20)}; BOUND "
          f"{fc.count_str(sum(bound(r) for r in h.values()), 20)}; ROUTED*@end "
          f"{fc.count_str(sum(routed_star_end(r) for r in h.values()), 20)}; failures {fails(h)}**")
        for nm, f in (("DISCOVERED", disc), ("BOUND", bound), ("ROUTED*@end", routed_star_end)):
            A = {sd: f(r) for sd, r in R["S70_DELTA"].items()}
            B = {sd: f(r) for sd, r in h.items()}
            pp = fc.paired(A, B, s.SEEDS)
            P(f"- paired {nm}: DELTA {pp['new']}/{pp['n']} vs HEBB {pp['old']}/{pp['n']}: delta only {pp['b']}, hebb only "
              f"{pp['c']}; p(delta > hebb) = {pp['p_new_gt_old']:.3g}, p(hebb > delta) = {pp['p_old_gt_new']:.3g}")
        P("\nη² of the read gate at the end (median; by stream / key / half / index), KEY and VAL; margin:\n")
        for arm in ("S70_HEBB", "S70_DELTA"):
            es = [r["end"] for r in R[arm].values() if "eta_key_by_stream" in r["end"]]
            if es:
                P(f"- {arm}: KEY " + "/".join(fmt(fc.med([e[f'eta_key_by_{g}'] for e in es])) for g in ("stream", "key", "half", "index"))
                  + "; VAL " + "/".join(fmt(fc.med([e[f'eta_val_by_{g}'] for e in es])) for g in ("stream", "key", "half", "index"))
                  + f"; margin {fmt(fc.med([e['margin'] for e in es]))}")
        P("\nPer seed: " + "; ".join(f"{arm} " + ", ".join(
            f"{sd} {r.get('tag')}" + (f"@{r['transition']}" if bound(r) else f" {r['acc']:.2f}") for sd, r in sorted(R[arm].items()))
            for arm in ("S70_HEBB", "S70_DELTA")))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    return dict(md=_write("rh_lite", L))
