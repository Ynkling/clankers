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


# ── S54 (a), (d) ─────────────────────────────────────────────────────────────
def disc(r):
    import explore_common as ec
    return bool(r and r.get("ok") and ec.discovered(r))


def routed_star_end(r):
    e = (r or {}).get("end") or {}
    return bool(e.get("margin") is not None and e["margin"] >= 0.9 and e.get("eta_key_by_stream", 0) > 0.9)


def fails(rs):
    import explore_common as ec
    out = {}
    for r in rs.values():
        if not bound(r):
            c = ec.fail_class(r)
            out[c] = out.get(c, 0) + 1
    return out


def valid_oracle(rs, need=1):
    done = len(rs)
    nb = sum(bound(r) for r in rs.values())
    return nb, done, (nb >= need) if done >= 2 else None


def beta_line(rs):
    bb = [r.get("beta_by_role") for r in rs.values() if r.get("beta_by_role")]
    if not bb:
        return "--"
    return "; ".join(f"L{i + 1} " + "/".join(fmt(fc.med([b[i][role] for b in bb])) for role in ("ctx", "key", "val"))
                     for i in range(len(bb[0])))


def report_s54ad(write=True):
    import explore_f_window_delta as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    L = []
    P = L.append
    P("### S54 (a), (d) — counts (Wilson 95%; band)\n")
    P("| arm | n | BOUND | DISCOVERED | ROUTED*@end | failures (unbound) | transitions: median (all) | final acc median |")
    P("|---|---|---|---|---|---|---|---|")
    for arm in s.ARMS:
        rs = R[arm]
        n = len(rs)
        oracle = "ORACLE" in arm
        P(f"| {arm} | {n}/{len(s.ARMS[arm]['seeds'])} | {fc.count_str(sum(bound(r) for r in rs.values()), n)} | "
          f"{'--' if oracle else fc.count_str(sum(disc(r) for r in rs.values()), n)} | "
          f"{'--' if oracle else fc.count_str(sum(routed_star_end(r) for r in rs.values()), n)} | "
          f"{'--' if oracle else fails(rs)} | {trans_str(rs)} | {acc_med(rs)} |")
    P("\nValidity (perfect gate binds on >= 1 of its 2 seeds):\n")
    val = {}
    for arm in ("A_ORACLE_DELTA", "A_ORACLE_HEBB", "D_ORACLE_DELTA", "D_ORACLE_HEBB"):
        nb, done, v = valid_oracle(R[arm])
        val[arm] = v
        P(f"- {arm}: {nb}/{done} -> {'VALID' if v else ('incomplete' if v is None else 'NOT VALID')} "
          f"(transitions {[r['transition'] for r in R[arm].values()]})")
    # (a)
    n_a = len(R["A_DELTA"])
    k_a = sum(disc(r) for r in R["A_DELTA"].values())
    if n_a < len(s.A_SEEDS) or val["A_ORACLE_DELTA"] is None:
        ra = "INCOMPLETE, no reading"
    elif not val["A_ORACLE_DELTA"]:
        ra = "UNTESTED (the delta oracle did not bind)"
    elif k_a >= s.BETTER:
        ra = "delta better (and not worse)"
    elif k_a >= s.NOT_WORSE:
        ra = "delta not worse"
    else:
        ra = "delta worse"
    pf = fc.fisher_greater(k_a, n_a, s.REC_LOCAL3_SLOW, 40) if n_a else None
    pf2 = fc.fisher_greater(s.REC_LOCAL3_SLOW, 40, k_a, n_a) if n_a else None
    P(f"\nREADING (a): **{ra}** (A_DELTA DISCOVERED {k_a}/{n_a} vs recorded Hebbian LOCAL3_SLOW 31/40 on 160-199, counts "
      f"only; rules: not worse >= {s.NOT_WORSE}, better >= {s.BETTER}; one-sided Fisher p(delta > hebb) = {fmt(pf, 3)}, "
      f"p(hebb > delta) = {fmt(pf2, 3)})")
    # (d)
    P("\n(d) BASELINE (RandHeaderTask, LOCAL3 + SLOW, k = 2, 24000):\n")
    for arm in ("D_HEBB", "D_DELTA"):
        rs = R[arm]
        n = len(rs)
        P(f"- BASELINE {arm}: DISCOVERED {fc.count_str(sum(disc(r) for r in rs.values()), n)}; BOUND "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), n)}; ROUTED*@end "
          f"{fc.count_str(sum(routed_star_end(r) for r in rs.values()), n)}; failures {fails(rs)}; validity "
          f"{'VALID' if val[arm.replace('D_', 'D_ORACLE_')] else val[arm.replace('D_', 'D_ORACLE_')]}")
    for nm, f in (("DISCOVERED", disc), ("BOUND", bound), ("ROUTED*@end", routed_star_end)):
        A = {sd: f(r) for sd, r in R["D_DELTA"].items()}
        B = {sd: f(r) for sd, r in R["D_HEBB"].items()}
        d = fc.paired(A, B, s.D_SEEDS)
        P(f"- paired {nm}: D_DELTA {d['new']}/{d['n']} vs D_HEBB {d['old']}/{d['n']}: delta only {d['b']}, hebb only "
          f"{d['c']}; p(delta > hebb) = {d['p_new_gt_old']:.3g}, p(hebb > delta) = {d['p_old_gt_new']:.3g}")
    P("\nη² of the read gate at the end (median over runs; by stream / key / half / index), KEY and VAL positions; margin:\n")
    for arm in ("A_DELTA", "D_DELTA", "D_HEBB"):
        es = [r["end"] for r in R[arm].values() if "eta_key_by_stream" in r["end"]]
        if es:
            k = "/".join(fmt(fc.med([e[f"eta_key_by_{g}"] for e in es])) for g in ("stream", "key", "half", "index"))
            v = "/".join(fmt(fc.med([e[f"eta_val_by_{g}"] for e in es])) for g in ("stream", "key", "half", "index"))
            P(f"- {arm}: KEY {k}; VAL {v}; margin {fmt(fc.med([e['margin'] for e in es]))}")
    P("\nLearned β at CTX/KEY/VAL per layer, end (median): " + "; ".join(
        f"{arm} {beta_line(R[arm])}" for arm in ("A_DELTA", "D_DELTA")))
    dec = [r.get("decode") for r in R["A_DELTA"].values() if r.get("decode")]
    if dec:
        P("\nA_DELTA gate-state decodability at KEY / VAL (median; chance 0.5): " + "; ".join(
            f"{t} {fmt(fc.med([d[t]['h_key'] for d in dec if t in d]))}/{fmt(fc.med([d[t]['h_val'] for d in dec if t in d]))}"
            for t in ("1200", "2400", "4800", "end")))
    P("\nPer seed (outcome tag; transition):\n")
    for arm in ("A_DELTA", "D_DELTA", "D_HEBB"):
        P(f"- {arm}: " + ", ".join(f"{sd} {r.get('tag')}" + (f"@{r['transition']}" if bound(r) else f" {r['acc']:.2f}")
                                   for sd, r in sorted(R[arm].items())))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"git {meta['gits']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    md = "\n".join(L)
    print(md)
    if write:
        open(os.path.join(fc.F_DIR, "window_delta_tables.md"), "w").write(md + "\n")
    return dict(ra=ra, md=md)


# ── S54 (b), (c) ─────────────────────────────────────────────────────────────
def br(r):
    return bool(r and r.get("ok") and r.get("outcome") == "BOUND ROUTED")


def report_s54bc(write=True):
    import explore_f_window_delta_scale as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    L = []
    P = L.append
    P("### S54 (b), (c) — counts (BOUND ROUTED; Wilson 95%; band)\n")
    P("| arm | n | BOUND ROUTED | BOUND | outcomes | splits fired (total; runs) | transitions of BOUND ROUTED: median (all) | "
      "final acc median |")
    P("|---|---|---|---|---|---|---|---|")
    for arm in s.ARMS:
        rs = R[arm]
        n = len(rs)
        oc = {}
        for r in rs.values():
            o = r.get("outcome")
            oc[o] = oc.get(o, 0) + 1
        tr = sorted(r["transition"] for r in rs.values() if br(r))
        sp = [r.get("splits") or 0 for r in rs.values()]
        P(f"| {arm} | {n}/{len(s.ARMS[arm]['seeds'])} | {fc.count_str(sum(br(r) for r in rs.values()), n)} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), n)} | {oc if 'ORACLE' not in arm else '--'} | "
          f"{sum(sp)}; {sum(1 for x in sp if x)} | {(str(fc.med(tr)) + ' (' + ', '.join(map(str, tr)) + ')') if tr else '--'} | "
          f"{acc_med(rs)} |")
    P("\nValidity (perfect gate binds on >= 1 of its 2 seeds):\n")
    val = {}
    for arm in ("B_ORACLE_DELTA", "B_ORACLE_HEBB", "C_ORACLE_DELTA", "C_ORACLE_HEBB"):
        nb, done, v = valid_oracle(R[arm])
        val[arm] = v
        P(f"- {arm}: {nb}/{done} -> {'VALID' if v else ('incomplete' if v is None else 'NOT VALID')} "
          f"(transitions {[r['transition'] for r in R[arm].values()]})")
    out = {}
    for cfg, seeds in (("B", s.B_SEEDS), ("C", s.C_SEEDS)):
        A = {sd: br(r) for sd, r in R[f"{cfg}_DELTA"].items()}
        B = {sd: br(r) for sd, r in R[f"{cfg}_HEBB"].items()}
        d = fc.paired(A, B, seeds)
        v = [val[f"{cfg}_ORACLE_DELTA"], val[f"{cfg}_ORACLE_HEBB"]]
        if d["n"] < len(seeds) or any(x is None for x in v):
            rd = "INCOMPLETE, no reading"
        elif not all(v):
            rd = "UNTESTED (a perfect gate did not bind)"
        elif d["b"] >= s.BETTER_B and d["c"] == 0:
            rd = "delta better (and not worse)"
        elif d["c"] - d["b"] <= s.NOT_WORSE_D:
            rd = "delta not worse"
        else:
            rd = "delta worse"
        out[cfg] = rd
        P(f"\nREADING ({cfg.lower()}): **{rd}** (BOUND ROUTED DELTA {d['new']}/{d['n']} vs HEBB {d['old']}/{d['n']}; delta "
          f"only {d['b']}, hebb only {d['c']}; one-sided McNemar p(delta > hebb) = {d['p_new_gt_old']:.3g}, p(hebb > delta) "
          f"= {d['p_old_gt_new']:.3g}; rules: not worse c − b <= 2, better b >= 4 and c = 0)")
    P("\nPer seed (outcome@transition, splits, end map):\n")
    for arm in ("B_DELTA", "B_HEBB", "C_DELTA", "C_HEBB"):
        P(f"- {arm}: " + "; ".join(
            f"{sd} {r.get('outcome')}" + (f"@{r['transition']}" if bound(r) else f" {r['acc']:.2f}")
            + f" s{r.get('splits')} {c16.chmap_str(r['end'].get('ch_map'))}" for sd, r in sorted(R[arm].items())))
    for arm in ("B_DELTA", "B_HEBB", "C_DELTA", "C_HEBB"):
        dec = [r.get("decode") for r in R[arm].values() if r.get("decode")]
        if dec:
            P(f"\n{arm} gate-state decodability at KEY / VAL (median): " + "; ".join(
                f"{t} {fmt(fc.med([d[t]['h_key'] for d in dec if t in d]))}/{fmt(fc.med([d[t]['h_val'] for d in dec if t in d]))}"
                for t in ("1200", "2400", "4800", "9600", "end")))
    P("\nLearned β at CTX/KEY/VAL per layer, end (median): " + "; ".join(
        f"{arm} {beta_line(R[arm])}" for arm in ("B_DELTA", "C_DELTA")))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"git {meta['gits']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    md = "\n".join(L)
    print(md)
    if write:
        open(os.path.join(fc.F_DIR, "window_delta_scale_tables.md"), "w").write(md + "\n")
    return dict(out=out, md=md)


# ── S59 ──────────────────────────────────────────────────────────────────────
def report_s59(write=True):
    import explore_f_decay as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    L = []
    P = L.append
    P("### S59 decay — counts (BOUND; Wilson 95%; band)\n")
    P("| arm | n | BOUND | transitions: median (all) | final acc median | write order acc last / not-last (median) |")
    P("|---|---|---|---|---|---|")
    for arm in s.ARMS:
        rs = R[arm]
        if not rs and s.ARMS[arm].get("conditional"):
            continue
        n = len(rs)
        wo = [r.get("write_order") or {} for r in rs.values()]
        P(f"| {arm} | {n}/{len(s.ARMS[arm]['seeds'])} | {fc.count_str(sum(bound(r) for r in rs.values()), n)} | "
          f"{trans_str(rs)} | {acc_med(rs)} | {fmt(fc.med([w.get('acc_last') for w in wo]))} / "
          f"{fmt(fc.med([w.get('acc_notlast') for w in wo]))} |")
    val = {}
    P("\nValidity (ORACLE binds on >= 2 of 350-352):\n")
    for d in s.DECAYS:
        rs = R[f"ORACLE_{d}"]
        nb = sum(bound(rs.get(sd)) for sd in s.VALID_SEEDS)
        v = (nb >= 2) if all(sd in rs for sd in s.VALID_SEEDS) else None
        val[d] = v
        P(f"- {d}: {nb}/3 -> {'VALID' if v else ('incomplete' if v is None else 'NOT VALID')}")
    gaps = {}
    for d in s.DECAYS:
        gaps[d] = {sd: R[f"ORACLE_{d}"][sd]["acc"] - R[f"SINGLE_{d}"][sd]["acc"] for sd in s.SEEDS
                   if sd in R[f"ORACLE_{d}"] and sd in R[f"SINGLE_{d}"]}
    P("\nOracle gap per seed (ORACLE acc − SINGLE acc), and gap_d − gap_FIXED:\n")
    P("| seed | FIXED | ROUTED | GDN | ROUTED − FIXED | GDN − FIXED |")
    P("|---|---|---|---|---|---|")
    cnt, dif = {}, {}
    for d in ("ROUTED", "GDN"):
        dif[d] = {sd: gaps[d][sd] - gaps["FIXED"][sd] for sd in s.SEEDS if sd in gaps[d] and sd in gaps["FIXED"]}
        cnt[d] = sum(1 for v in dif[d].values() if v >= s.GAP)
    for sd in s.SEEDS:
        P(f"| {sd} | " + " | ".join(fmt(gaps[d].get(sd), 3) for d in s.DECAYS) + " | "
          + " | ".join(fmt(dif[d].get(sd), 3) for d in ("ROUTED", "GDN")) + " |")
    complete = all(len(dif[d]) == len(s.SEEDS) for d in dif)
    if not complete or any(v is None for v in val.values()):
        rd, win = "INCOMPLETE, no reading", None
    else:
        ok = {d: val[d] and val["FIXED"] and cnt[d] >= s.N_SEEDS for d in ("ROUTED", "GDN")}
        unt = [d for d in ("ROUTED", "GDN") if not (val[d] and val["FIXED"])]
        if any(ok.values()):
            cands = [d for d in ok if ok[d]]
            win = max(cands, key=lambda d: (cnt[d], fc.med(list(dif[d].values()))))
            rd = f"the horizon knob matters (winner {win})"
        else:
            win = None
            rd = "it does not" + (f" (UNTESTED for {unt})" if unt else "")
    P(f"\nREADING: **{rd}** (seeds with gap_d − gap_FIXED >= {s.GAP}: ROUTED {cnt.get('ROUTED')}, GDN {cnt.get('GDN')} of "
      f"10; rule >= {s.N_SEEDS}; medians of gap_d − gap_FIXED: ROUTED {fmt(fc.med(list(dif['ROUTED'].values())), 3)}, GDN "
      f"{fmt(fc.med(list(dif['GDN'].values())), 3)})")
    P("\nLearned decay at the end (median α per layer at CTX/KEY/VAL):\n")
    for arm in s.ARMS:
        dd = [r.get("decay_by_role") for r in R[arm].values() if r.get("decay_by_role")]
        if not dd:
            continue
        keys = [k for k in dd[0][0] if k != "beta"]
        P(f"- {arm}: " + "; ".join(f"L{i + 1} " + ", ".join(
            f"{k} " + "/".join(fmt(fc.med([x[i][k][role] for x in dd]), 3) for role in ("ctx", "key", "val")) for k in keys)
            for i in range(len(dd[0]))))
    meta = store_meta(s)
    P(f"\nRuns: {meta['n']} ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"git {meta['gits']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    md = "\n".join(L)
    print(md)
    if write:
        open(os.path.join(fc.F_DIR, "decay_tables.md"), "w").write(md + "\n")
    return dict(rd=rd, win=win, md=md)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "s53"
    globals()[f"report_{which}"]()
