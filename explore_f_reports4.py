#!/usr/bin/env python
"""
explore_f_reports4.py — EXPLORATORY, not a result. Report tables of Session F's S80 (configurations) and S81 (the split's
trigger on delta), with the readings exactly as each screen's docstring fixes them. Writes explore_out/F/<screen>_tables.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc
from explore_f_reports import fmt, bound, store_meta
from explore_f_reports3 import _write, FEATS


def _runs(s):
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    return {arm: fc.runs(s, arm) for arm in s.ARMS}


def _trs(rs, ok=bound):
    tr = sorted(r["transition"] for r in rs.values() if ok(r))
    return f"{fc.med(tr):g} ({', '.join(map(str, tr))})" if tr else "--"


def _reading(d, valid):
    if d["n"] < 10:
        return "INCOMPLETE, no reading"
    if not valid:
        return "UNTESTED (the delta perfect gate did not bind on both seeds)"
    if d["c"] - d["b"] <= 1:
        return "the delta recipe keeps it"
    if d["c"] >= 4 and d["b"] == 0:
        return "the delta recipe loses it"
    return "inconclusive"


def _validity(P, R, arms):
    for arm in arms:
        rs = R[arm]
        P(f"- {arm}: " + "; ".join(f"{sd} " + (f"bound @{r['transition']}" if bound(r) else
                                              (f"unbound {r['acc']:.2f}" if r.get("ok") else "FAILED"))
                                   for sd, r in sorted(rs.items())))


def _probe_table(P, R, arms):
    P("| arm | n | " + " | ".join(f"{f} K / V" for f in FEATS) + " | L3 >= 0.9 at K and V | at init: L3 K / V |")
    P("|---|---|" + "---|" * len(FEATS) + "---|---|")
    for arm in arms:
        rs = {sd: r for sd, r in R[arm].items() if r.get("probe")}
        if not rs:
            continue
        cells = [f"{fmt(fc.med([r['probe'][f + '|K']['acc'] for r in rs.values()]))} / "
                 f"{fmt(fc.med([r['probe'][f + '|V']['acc'] for r in rs.values()]))}" for f in FEATS]
        hi = sum(1 for r in rs.values() if r["probe"]["L3|K"]["acc"] >= 0.9 and r["probe"]["L3|V"]["acc"] >= 0.9)
        P(f"| {arm} | {len(rs)} | " + " | ".join(cells) + f" | {hi}/{len(rs)} | "
          f"{fmt(fc.med([r['probe_init']['L3|K']['acc'] for r in rs.values()]))} / "
          f"{fmt(fc.med([r['probe_init']['L3|V']['acc'] for r in rs.values()]))} |")


# ── S80 ──────────────────────────────────────────────────────────────────────
def report_s80():
    import explore_f_configs as sa
    import explore_f_configs_b as sb
    Ra, Rb = _runs(sa), _runs(sb)
    L = []
    P = L.append
    out = {}
    # (a)
    P("### S80 (a) two streams (S = 2, P = 4, k = 2, no conv, 24000; WIN3_SLOW; seeds 2000-2009; DISCOVERED; Wilson 95%)\n")
    P("Validity (the delta perfect gate binds on both seeds):\n")
    _validity(P, Ra, ("O_DELTA_A", "O_HEBB_A"))
    va = sum(bound(r) for r in Ra["O_DELTA_A"].values()) == 2
    P("\n| arm | n | DISCOVERED | BOUND | transitions of DISCOVERED: median (all) | not discovered: bound (VAL cos) / unbound |")
    P("|---|---|---|---|---|---|")
    for arm in ("HEBB_A", "DELTA_A"):
        rs = Ra[arm]
        nd = [(sd, r) for sd, r in sorted(rs.items()) if not r.get("discovered")]
        P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(bool(r.get('discovered')) for r in rs.values()), len(rs))} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {_trs(rs, lambda r: bool(r.get('discovered')))} | "
          + (", ".join(f"{sd} " + (f"bound ({r['val_cos']:.2f})" if bound(r) else "unbound") for sd, r in nd) or "none") + " |")
    A = {sd: bool(r.get("discovered")) for sd, r in Ra["DELTA_A"].items()}
    B = {sd: bool(r.get("discovered")) for sd, r in Ra["HEBB_A"].items()}
    da = fc.paired(A, B, sa.SEEDS)
    ra = _reading(da, va)
    P(f"\nREADING (a): **{ra}** (DISCOVERED DELTA_A {da['new']}/{da['n']} vs HEBB_A {da['old']}/{da['n']}; delta only {da['b']}, "
      f"hebb only {da['c']}; p(hebb > delta) = {da['p_old_gt_new']:.3g}, p(delta > hebb) = {da['p_new_gt_old']:.3g}; rules: keeps "
      f"c − b <= 1, loses c >= 4 and b = 0). Reference (counts only, other CPUs and seeds): WIN3_SLOW pooled {sa.REF[1]}/{sa.REF[2]}.")
    # failure classes v1 / v2 (unbound runs), through the main-line child
    two = {f"{arm}|{sd}": r["end"] for arm in ("HEBB_A", "DELTA_A") for sd, r in Ra[arm].items() if not bound(r)}
    cl = c16.run_child("explore_f_configs_b_child", "classify", dict(two=two), main=True) if two else {}
    P("\nFailure classes of the unbound runs (v2 = fail_class_v2, ba8901a; v1 = test_router_layout.fail_class):\n")
    for arm in ("HEBB_A", "DELTA_A"):
        rows = [(k, v) for k, v in cl.items() if k.startswith(arm + "|")]
        P(f"- {arm}: " + ("; ".join(f"{k.split('|')[1]} v2 {v['v2']}" + (f" (v1 {v['v1']})" if v["v1"] != v["v2"] else "")
                                    + f" [eta2 by key {two[k]['eta_key_by_key']:.3f}, margin {two[k]['margin']:.3f}]"
                                    for k, v in sorted(rows)) or "none"))
    diff_a = [k for k, v in cl.items() if v["v1"] != v["v2"]]
    P(f"- v1 and v2 differ on: {', '.join(diff_a) or 'none'}")
    P("\nProbe at the end of every run (S77's: G's S55 logistic regression on the residual; stream of the body triple; "
      "held-out 400 sequences; chance 0.5):\n")
    _probe_table(P, Ra, ("HEBB_A", "DELTA_A", "O_HEBB_A", "O_DELTA_A"))
    P("\nPer seed, L3 K / V (HEBB_A | DELTA_A; D = discovered):\n")
    for sd in sa.SEEDS:
        h, d = Ra["HEBB_A"].get(sd), Ra["DELTA_A"].get(sd)
        if not (h and d and h.get("probe") and d.get("probe")):
            continue
        P(f"- {sd}: {h['probe']['L3|K']['acc']:.2f} / {h['probe']['L3|V']['acc']:.2f}{' D' if h.get('discovered') else ''} | "
          f"{d['probe']['L3|K']['acc']:.2f} / {d['probe']['L3|V']['acc']:.2f}{' D' if d.get('discovered') else ''}")
    out["ra"] = ra
    # (b)
    P("\n### S80 (b) four streams (S = 4, P = 4, k = 16, conv, 28800; WIN3_SLOW16 + exact-copy split; seeds 2010-2019; "
      "BOUND ROUTED; Wilson 95%)\n")
    P("Validity:\n")
    _validity(P, Rb, ("O_DELTA_B", "O_HEBB_B"))
    vb = sum(bound(r) for r in Rb["O_DELTA_B"].values()) == 2
    P("\n| arm | n | BOUND ROUTED | BOUND | outcomes (v1) | splits (total; runs; targets ok) | transitions of BOUND ROUTED |")
    P("|---|---|---|---|---|---|---|")
    for arm in ("HEBB_B", "DELTA_B"):
        rs = Rb[arm]
        oc = {}
        for r in rs.values():
            oc[r.get("outcome")] = oc.get(r.get("outcome"), 0) + 1
        spl = [c for r in rs.values() for c in r.get("checks", []) if c.get("fired")]
        P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(r.get('outcome') == 'BOUND ROUTED' for r in rs.values()), len(rs))} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {oc} | {len(spl)}; "
          f"{sum(1 for r in rs.values() if r.get('splits'))}; {sum(bool(c.get('key_ok')) for c in spl)}/{len(spl)} | "
          f"{_trs(rs, lambda r: r.get('outcome') == 'BOUND ROUTED')} |")
    A = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in Rb["DELTA_B"].items()}
    B = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in Rb["HEBB_B"].items()}
    db = fc.paired(A, B, sb.SEEDS)
    rb = _reading(db, vb)
    P(f"\nREADING (b): **{rb}** (BOUND ROUTED DELTA_B {db['new']}/{db['n']} vs HEBB_B {db['old']}/{db['n']}; delta only {db['b']}, "
      f"hebb only {db['c']}; p(hebb > delta) = {db['p_old_gt_new']:.3g}, p(delta > hebb) = {db['p_new_gt_old']:.3g}). Reference "
      f"(counts only): WIN3_SLOW16 pooled {sb.REF[1]}/{sb.REF[2]} (no split).")
    P("\nPer run (outcome v1 [v2 where it differs] @transition or final acc; splits update c*->c0 ok?):\n")
    for arm in ("HEBB_B", "DELTA_B"):
        P(f"- {arm}: " + "; ".join(
            f"{sd} {r.get('outcome')}" + (f" [v2 {r.get('outcome_v2')}]" if r.get("outcome_v2") not in (None, r.get("outcome")) else "")
            + (f"@{r['transition']}" if r.get("transition") else f" {r['acc']:.2f}") + " ["
            + ", ".join(f"{c['step']} {c['key_cs']}->{c['key_c0']} {'ok' if c.get('key_ok') else 'NOT ok'}"
                        for c in r.get("checks", []) if c.get("fired")) + "]"
            for sd, r in sorted(Rb[arm].items()) if r.get("ok")))
    out["rb"] = rb
    # the classifier CHECK on every stored Session F record at report time
    cr = c16.run_child("explore_f_configs_b_child", "classifier_rows", {}, main=True)
    P(f"\nClassifier CHECK at report time: v2 equals v1 on {cr['n']} stored Session F records ({cr['n_two']} two-stream, "
      f"{cr['n_multi']} multichannel) except where eta2 by key >= 0.5 and the margin rule fired: {len(cr['diffs'])} differ"
      f"{':' if cr['diffs'] else ''} " + "; ".join(f"{t} ({'bound' if bd else 'unbound'}) {a} -> {b} (eta2 by key {eta:.4f}, "
                                                   f"margin {mg:.3f})" for t, kind, bd, a, b, eta, mg in cr["diffs"])
      + f"; violations {cr['bad'] or 'none'} ({'ok' if cr['ok'] else 'FAILED'})")
    for s in (sa, sb):
        meta = store_meta(s)
        P(f"\n{s.NAME}: {meta['n']} runs ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
          f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    out["md"] = _write("configs", L)
    return out


# ── S81 ──────────────────────────────────────────────────────────────────────
def report_s81():
    import explore_f_trigger as s
    R = _runs(s)
    L = []
    P = L.append
    out = {}
    arms = ("BASE", "GUARD", "EVAL", "GUARD_EVAL")
    P("### S81 the split's trigger on delta memory (S = 8, P = 4, k = 16, conv, 43200; DELTA*; seeds 2020-2029; BOUND ROUTED; "
      "Wilson 95%)\n")
    P("Validity (the delta perfect gate binds on both seeds):\n")
    _validity(P, R, ("O_DELTA_C",))
    valid = sum(bound(r) for r in R["O_DELTA_C"].values()) == 2
    P("\n| arm | n | BOUND ROUTED | BOUND | outcomes (v1) | firings (total; runs; labelled ok) | firings at a disagreement | "
      "transitions of BOUND ROUTED |")
    P("|---|---|---|---|---|---|---|---|")
    st = {}
    for arm in arms:
        rs = R[arm]
        oc = {}
        for r in rs.values():
            oc[r.get("outcome")] = oc.get(r.get("outcome"), 0) + 1
        fires = [c for r in rs.values() for c in r.get("checks", []) if c.get("fired")]
        chk = [c for r in rs.values() for c in r.get("checks", [])]
        nok = sum(bool(c.get("key_ok")) for c in fires)
        nd = sum(bool(c.get("disagree")) for c in fires)
        st[arm] = dict(fires=len(fires), ok=nok, dis=nd, chk=chk)
        P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(r.get('outcome') == 'BOUND ROUTED' for r in rs.values()), len(rs))} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {oc} | {len(fires)}; "
          f"{sum(1 for r in rs.values() if r.get('splits'))}; {nok}/{len(fires)}"
          f"{f' ({100 * nok / len(fires):.0f}%)' if fires else ''} | {nd} | "
          f"{_trs(rs, lambda r: r.get('outcome') == 'BOUND ROUTED')} |")
    P("\nChecks (all arms; printed, not readings): disagreement = the rule's eligibility read on the probe differs from it read "
      "on the evaluation (with the arm's guard).\n")
    P("| arm | checks | guarded (eligible but < 0.5) | disagreements: probe yes / eval no | probe no / eval yes | probe >= 0.95 "
      "while eval < 0.95 | eval >= 0.95 while probe < 0.95 |")
    P("|---|---|---|---|---|---|---|")
    for arm in arms:
        chk = st[arm]["chk"]
        P(f"| {arm} | {len(chk)} | {sum(bool(c.get('guarded')) for c in chk)} | "
          f"{sum(1 for c in chk if c.get('elig_probe') and not c.get('elig_eval'))} | "
          f"{sum(1 for c in chk if c.get('elig_eval') and not c.get('elig_probe'))} | "
          f"{sum(1 for c in chk if c.get('acc_probe', 0) >= 0.95 and c.get('acc_eval', 1) < 0.95)} | "
          f"{sum(1 for c in chk if c.get('acc_eval', 0) >= 0.95 and c.get('acc_probe', 1) < 0.95)} |")
    base = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in R["BASE"].items()}
    pr = {}
    for arm in arms[1:]:
        x = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in R[arm].items()}
        pr[arm] = fc.paired(x, base, s.SEEDS)
    P("\nPaired against BASE (BOUND ROUTED; b = the arm alone, c = BASE alone; exact one-sided McNemar):\n")
    for arm, d in pr.items():
        P(f"- {arm} {d['new']}/{d['n']} vs BASE {d['old']}/{d['n']}: b {d['b']}, c {d['c']}; p(arm > BASE) = "
          f"{d['p_new_gt_old']:.3g}, p(BASE > arm) = {d['p_old_gt_new']:.3g}")
    complete = all(len(R[a]) == 10 for a in arms) and len(R["O_DELTA_C"]) == 2
    rd = []
    if not complete:
        rd = ["INCOMPLETE, no reading"]
    elif not valid:
        rd = ["UNTESTED (the delta perfect gate did not bind on both seeds)"]
    else:
        g, e, ge = pr["GUARD"], pr["EVAL"], pr["GUARD_EVAL"]
        sg, se = st["GUARD"], st["EVAL"]
        if sg["fires"] >= 1 and sg["ok"] >= 0.9 * sg["fires"] and g["c"] - g["b"] <= 1:
            rd.append("the guard aims the split")
        if e["c"] - e["b"] <= 1 and se["dis"] == 0:
            rd.append("the evaluation is the better probe")
        if g["c"] >= 4 and g["b"] == 0:
            rd.append("the guard costs binding")
        if e["c"] >= 4 and e["b"] == 0:
            rd.append("the evaluation costs binding")
        if ge["c"] >= 4 and ge["b"] == 0:
            rd.append("(printed) both devices together cost binding")
        if not any(x.startswith("the guard") for x in rd):
            rd.append("guard: no reading applies")
        if not any(x.startswith("the evaluation") for x in rd):
            rd.append("evaluation: no reading applies")
    sb = st["BASE"]
    P(f"\nREADINGS S81: **{'; '.join(rd)}** (GUARD firings labelled ok {st['GUARD']['ok']}/{st['GUARD']['fires']}, rule >= 90%; "
      f"EVAL firings at a disagreement {st['EVAL']['dis']}, rule 0; BASE labelled ok {sb['ok']}/{sb['fires']} (S76: 6/12); "
      f"GUARD_EVAL labelled ok {st['GUARD_EVAL']['ok']}/{st['GUARD_EVAL']['fires']}, at a disagreement {st['GUARD_EVAL']['dis']})")
    out["rd"] = rd
    P("\nFirings per run (update c*->c0, labelled ok?, probe / evaluation reading, D = at a disagreement):\n")
    for arm in arms:
        P(f"- {arm}: " + "; ".join(
            f"{sd} {r.get('outcome')}" + (f"@{r['transition']}" if r.get('transition') else f" {r['acc']:.2f}") + " ["
            + ", ".join(f"{c['step']} {c['key_cs']}->{c['key_c0']} {'ok' if c.get('key_ok') else 'NOT ok'} "
                        f"{c['acc_probe']:.2f}/{c['acc_eval']:.2f}{' D' if c.get('disagree') else ''}"
                        for c in r.get("checks", []) if c.get("fired")) + "]"
            for sd, r in sorted(R[arm].items()) if r.get("ok")))
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed or diverged records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU "
      f"{meta['cpus']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    out["md"] = _write("trigger", L)
    return out
