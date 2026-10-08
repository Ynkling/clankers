#!/usr/bin/env python
"""
explore_f_reports.py — EXPLORATORY, not a result. Session F's report tables (markdown), computed from the stores with the
readings exactly as each screen's docstring fixes them. Prints the markdown and writes explore_out/F/<screen>_tables.md;
the batch report (report_<n>.md) is written from these tables.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc


def fmt(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def bound(r):
    return bool(r and r.get("ok") and r["transition"] is not None)


def trans_str(rs):
    tr = sorted(r["transition"] for r in rs.values() if bound(r))
    return f"{fc.med(tr):g} ({', '.join(str(t) for t in tr)})" if tr else "--"


def acc_med(rs):
    return fmt(fc.med([r["acc"] for r in rs.values()]))


def checks_summary(prefix):
    fs = sorted(glob.glob(os.path.join(fc.F_DIR, f"checks_{prefix}*.json")))
    out = []
    for f in fs:
        d = json.load(open(f))
        rows = d["rows"]
        out.append(f"{os.path.basename(f)}: {'ALL PASSED' if d['ok'] else 'FAILED'} ({sum(v is True for _, v in rows)} ok, "
                   f"{sum(v is None for _, v in rows)} n/a) on {d['provenance']['cpu']}, torch {d['provenance']['torch']}, "
                   f"git {d['provenance']['git']}")
    return out


def store_meta(screen):
    st = fc.load(screen)
    rs = [r for r in st["runs"].values() if r.get("ok") and r.get("code_sha", "").startswith(c16.STATE["shas"][screen.NAME][:12])]
    secs = sum(r.get("secs_wall", 0) for r in rs)
    gits = sorted({r.get("git") for r in rs})
    cpus = sorted({r.get("cpu") for r in rs})
    return dict(n=len(rs), hours=secs / 3600, gits=gits, cpus=cpus, failed=sum(1 for r in st["runs"].values() if not r.get("ok")))


# ── S53 ──────────────────────────────────────────────────────────────────────
def report_s53(write=True):
    import explore_f_delta_controls as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    L = []
    P = L.append
    P("### S53 delta_controls — counts (BOUND; Wilson 95%; band)\n")
    P("| arm | layout | BOUND | transitions: median (all) | final acc median | write order: acc last / not-last / "
      "last-written value among not-last (medians) |")
    P("|---|---|---|---|---|---|")
    for arm, a in s.ARMS.items():
        rs = R[arm]
        n, k = len(rs), sum(bound(r) for r in rs.values())
        wo = [r.get("write_order") or {} for r in rs.values()]
        P(f"| {arm} | {a['layout']} | {fc.count_str(k, n)} of {len(a['seeds'])} | {trans_str(rs)} | {acc_med(rs)} | "
          f"{fmt(fc.med([w.get('acc_last') for w in wo]))} / {fmt(fc.med([w.get('acc_notlast') for w in wo]))} / "
          f"{fmt(fc.med([w.get('lastval_notlast') for w in wo]))} |")
    # validity
    P("\nValidity (perfect gate binds on >= 2 of seeds 300-302):\n")
    valid = {}
    for l in ("G", "H"):
        for kind in ("ORACLE_DELTA", "ORACLE_HEBB"):
            rs = R[f"{kind}_{l}"]
            nb = sum(bound(rs.get(sd)) for sd in s.VALID_SEEDS)
            done = all(sd in rs for sd in s.VALID_SEEDS)
            v = (nb >= 2) if done else None
            if kind == "ORACLE_DELTA":
                valid[l] = v
            P(f"- {s.LAYOUTS[l]}: {kind} {nb}/3 -> {'VALID' if v else ('incomplete' if v is None else 'NOT VALID')}"
              f"{' (the delta reading needs this)' if kind == 'ORACLE_DELTA' else ' (printed)'}")
    # paired
    P("\nPaired (same seeds and batches; exact one-sided McNemar):\n")
    for l in ("G", "H"):
        for x, y in (("SINGLE_HEBB", "SINGLE_DELTA"), ("SINGLE_HEBB", "SINGLE_DELTA_L"), ("ORACLE_DELTA", "ORACLE_HEBB")):
            A = {sd: bound(r) for sd, r in R[f"{x}_{l}"].items()}
            B = {sd: bound(r) for sd, r in R[f"{y}_{l}"].items()}
            d = fc.paired(A, B, s.SEEDS)
            P(f"- {s.LAYOUTS[l]}: {x} {d['new']}/{d['n']} vs {y} {d['old']}/{d['n']}: {x} only {d['b']}, {y} only {d['c']}, "
              f"p({x} > {y}) = {d['p_new_gt_old']:.3g}")
    A = {sd: bound(r) for sd, r in R["ORACLE_DELTA_P8"].items()}
    B = {sd: bound(r) for sd, r in R["ORACLE_HEBB_P8"].items()}
    d = fc.paired(A, B, s.SEEDS_P8)
    P(f"- header P=8 (48000): ORACLE_DELTA {d['new']}/{d['n']} vs ORACLE_HEBB {d['old']}/{d['n']}: delta only {d['b']}, "
      f"hebb only {d['c']}, p = {d['p_new_gt_old']:.3g}")
    # readings
    n_sd = {l: sum(bound(r) for r in R[f"SINGLE_DELTA_{l}"].values()) for l in ("G", "H")}
    n_sh = {l: sum(bound(r) for r in R[f"SINGLE_HEBB_{l}"].values()) for l in ("G", "H")}
    complete = all(len(R[f"{k}_{l}"]) == len(s.SEEDS) for k in ("SINGLE_DELTA", "SINGLE_HEBB") for l in ("G", "H"))
    if not complete or any(v is None for v in valid.values()):
        r1 = "INCOMPLETE, no reading"
    elif not all(valid.values()):
        r1 = "UNTESTED (ORACLE_DELTA did not bind >= 2/3 on " + ", ".join(s.LAYOUTS[l] for l, v in valid.items() if not v) + ")"
    elif all(n_sd[l] <= 1 for l in n_sd) and any(n_sh[l] >= 3 for l in n_sh):
        r1 = "delta makes the single channel fail"
    else:
        r1 = "neither"
    P(f"\nREADING R1: **{r1}** (SINGLE_DELTA {n_sd['G']}/10 grouped, {n_sd['H']}/10 header; SINGLE_HEBB {n_sh['G']}/10, "
      f"{n_sh['H']}/10; rule: SINGLE_DELTA <= 1/10 on both and SINGLE_HEBB >= 3/10 on one, both layouts valid)")
    p8 = all(sd in R["ORACLE_DELTA_P8"] and sd in R["ORACLE_HEBB_P8"] for sd in s.SEEDS_P8)
    n_r2 = sum(1 for sd in s.SEEDS_P8 if bound(R["ORACLE_DELTA_P8"].get(sd)) and not bound(R["ORACLE_HEBB_P8"].get(sd)))
    r2 = ("INCOMPLETE, no reading" if not p8 else "delta raises the oracle" if n_r2 >= 2 else "neither")
    P(f"\nREADING R2: **{r2}** (seeds where ORACLE_DELTA binds and ORACLE_HEBB does not: {n_r2} of 3; rule >= 2)")
    # write order, descriptive
    P("\nWrite order (descriptive; median run per layout; held-out 2048):\n")
    for l in ("G", "H"):
        for kind in ("SINGLE_DELTA", "SINGLE_DELTA_L", "SINGLE_HEBB"):
            wo = [r.get("write_order") or {} for r in R[f"{kind}_{l}"].values()]
            al, an, lv = (fc.med([w.get(f) for w in wo]) for f in ("acc_last", "acc_notlast", "lastval_notlast"))
            bys = [w.get("acc_by_stream") for w in wo if w.get("acc_by_stream")]
            s0, s1 = (fc.med([b[i] for b in bys]) if bys else None for i in (0, 1))
            keeps = (al is not None and al >= 0.9 and an <= 0.1 and lv >= 0.8)
            tag = (" -> keeps exactly the last-written stream" if keeps else " -> does not") if kind == "SINGLE_DELTA" else ""
            P(f"- {s.LAYOUTS[l]} {kind}: acc last {fmt(al)}, not-last {fmt(an)}, last-written value among not-last "
              f"{fmt(lv)}; by query stream {fmt(s0)} / {fmt(s1)}{tag}")
    P("\nPer seed (BOUND transition or final acc; write order last/not-last):\n")
    P("| seed | " + " | ".join(a for a in s.ARMS if not a.endswith("P8")) + " |")
    P("|---|" + "---|" * (len(s.ARMS) - 2))
    for sd in s.SEEDS:
        cells = []
        for arm in s.ARMS:
            if arm.endswith("P8"):
                continue
            r = R[arm].get(sd)
            if r is None:
                cells.append("--")
                continue
            wo = r.get("write_order") or {}
            cells.append((f"B@{r['transition']}" if bound(r) else f"{r['acc']:.2f}") +
                         f" ({fmt(wo.get('acc_last'))}/{fmt(wo.get('acc_notlast'))})")
        P(f"| {sd} | " + " | ".join(cells) + " |")
    P("\nP = 8 header per seed: " + "; ".join(
        f"{sd}: HEBB {('B@' + str(R['ORACLE_HEBB_P8'][sd]['transition'])) if bound(R['ORACLE_HEBB_P8'].get(sd)) else fmt(R['ORACLE_HEBB_P8'].get(sd, {}).get('acc'))}"
        f", DELTA {('B@' + str(R['ORACLE_DELTA_P8'][sd]['transition'])) if bound(R['ORACLE_DELTA_P8'].get(sd)) else fmt(R['ORACLE_DELTA_P8'].get(sd, {}).get('acc'))}"
        for sd in s.SEEDS_P8))
    # beta
    P("\nSINGLE_DELTA_L: mean β at CTX / KEY / VAL per layer at the end (median over runs):\n")
    for l in ("G", "H"):
        bb = [r.get("beta_by_role") for r in R[f"SINGLE_DELTA_L_{l}"].values() if r.get("beta_by_role")]
        if bb:
            P(f"- {s.LAYOUTS[l]}: " + "; ".join(
                f"layer {i + 1} " + "/".join(fmt(fc.med([b[i][role] for b in bb])) for role in ("ctx", "key", "val"))
                for i in range(len(bb[0]))))
    # oracle routing
    P("\nOracle routing check (end, median): " + "; ".join(
        f"{arm} margin {fmt(fc.med([r['end'].get('margin') for r in R[arm].values() if r['end'].get('margin') is not None]))}"
        for arm in ("ORACLE_HEBB_G", "ORACLE_DELTA_G", "ORACLE_HEBB_H", "ORACLE_DELTA_H")))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"git {meta['gits']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    md = "\n".join(L)
    print(md)
    if write:
        open(os.path.join(fc.F_DIR, "delta_controls_tables.md"), "w").write(md + "\n")
    return dict(r1=r1, r2=r2, md=md)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "s53"
    globals()[f"report_{which}"]()
