#!/usr/bin/env python
"""
explore_f_reports5.py — EXPLORATORY, not a result. Report tables of Session F's S88a (the write strength) and S88b (the
horizon), with the readings exactly as each screen's docstring fixes them. Writes explore_out/F/<screen>_tables.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc
from explore_f_reports import fmt, bound, store_meta
from explore_f_reports3 import _write
from explore_f_reports4 import _runs, _trs, _validity


def _bt_end(r):
    bt = r.get("beta_traj") or {}
    return bt[max(bt, key=int)] if bt else None


# ── S88a ─────────────────────────────────────────────────────────────────────
def _with_diverged(s):
    """{arm: {seed: record}}: the ok records plus the runs stored as DIVERGED (counted as not successful)."""
    R = _runs(s)
    st = fc.load(s)
    sha = c16.STATE["shas"][s.NAME][:12]
    for arm, a in s.ARMS.items():
        for sd in a["seeds"]:
            r = st["runs"].get(f"{arm}|{sd}|{sha}|{c16.STATE['batch_cpu']}")
            if sd not in R[arm] and r is not None and "DIVERGED" in str(r.get("error")):
                R[arm][sd] = r
    return R


def report_s88a():
    import explore_f_write as s
    R = _with_diverged(s)
    L = []
    P = L.append
    out = {}
    arms = ("DELTA_FIXED", "DELTA_LEARNED", "DELTA_LEARNED_SLOW")
    P("### S88a the write strength (S = 8, P = 4, k = 16, conv, 43200; the delta recipe with the exact-copy split; seeds "
      "2400-2409; BOUND ROUTED; Wilson 95%)\n")
    P("Validity (O_DELTA binds on both seeds) and the learned-β perfect gate (printed):\n")
    _validity(P, R, ("O_DELTA", "O_DELTA_LEARNED"))
    valid = sum(bound(r) for r in R["O_DELTA"].values()) == 2
    P("\n| arm | n | BOUND ROUTED | BOUND | outcomes (v1) | splits (total; runs; targets ok) | transitions of BOUND ROUTED |")
    P("|---|---|---|---|---|---|---|")
    for arm in arms:
        rs = R[arm]
        oc = {}
        for r in rs.values():
            oc[r.get("outcome") if r.get("ok") else "DIVERGED"] = oc.get(r.get("outcome") if r.get("ok") else "DIVERGED", 0) + 1
        spl = [c for r in rs.values() for c in r.get("checks", []) if c.get("fired")]
        P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(r.get('outcome') == 'BOUND ROUTED' for r in rs.values()), len(rs))} | "
          f"{fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {oc} | {len(spl)}; "
          f"{sum(1 for r in rs.values() if r.get('splits'))}; {sum(bool(c.get('key_ok')) for c in spl)}/{len(spl)} | "
          f"{_trs(rs, lambda r: r.get('outcome') == 'BOUND ROUTED')} |")
    base = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in R["DELTA_FIXED"].items()}
    pr = {}
    P("\nPaired against DELTA_FIXED (b = the learned arm alone, c = DELTA_FIXED alone; exact one-sided McNemar):\n")
    for arm in arms[1:]:
        x = {sd: r.get("outcome") == "BOUND ROUTED" for sd, r in R[arm].items()}
        d = pr[arm] = fc.paired(x, base, s.SEEDS)
        P(f"- {arm} {d['new']}/{d['n']} vs DELTA_FIXED {d['old']}/{d['n']}: b {d['b']}, c {d['c']}; p(arm > FIXED) = "
          f"{d['p_new_gt_old']:.3g}, p(FIXED > arm) = {d['p_old_gt_new']:.3g}")
    complete = all(len(R[a]) == 10 for a in arms) and len(R["O_DELTA"]) == 2
    if not complete:
        rd = "INCOMPLETE, no reading"
    elif not valid:
        rd = "UNTESTED (the perfect gate did not bind on both seeds)"
    else:
        best = max(arms[1:], key=lambda a: (pr[a]["new"], -pr[a]["c"]))
        keeps = pr[best]["c"] - pr[best]["b"] <= 1
        costs = all(pr[a]["c"] >= 4 and pr[a]["b"] == 0 for a in arms[1:])
        rd = ("a learned write strength keeps the partition" if keeps else
              "a learned write strength costs it" if costs else "no reading applies")
        rd += f" (the better learned arm: {best})"
    P(f"\nREADING S88a: **{rd}** (rules: keeps if c − b <= 1 for the better learned arm; costs if c >= 4 and b = 0 for both)")
    out["rd"] = rd
    # β
    P("\nβ per run (learned arms; probe batch; layers 1 / 2 / 3). Trajectory: the median over positions at 1200, 2400, 4800, "
      "9600 and the end; then at the end the medians at CTX / KEY / VAL, the fraction > 1, η² by stream at KEY, and the spread of "
      "the 8 per-stream medians at KEY (max − min); D = β diverges by stream (η² at KEY >= 0.5 in a layer at the end):\n")
    div = {}
    for arm in arms[1:] + ("O_DELTA_LEARNED",):
        for sd, r in sorted(R[arm].items()):
            bt = r.get("beta_traj") or {}
            if not bt:
                continue
            steps = [st for st in ("1200", "2400", "4800", "9600") if st in bt] + [max(bt, key=int)]
            traj = "; ".join(f"{st}: " + "/".join(f"{Ly['median']:.2f}" for Ly in bt[st]) for st in dict.fromkeys(steps))
            e = bt[max(bt, key=int)]
            d = any(Ly["eta_key_by_stream"] >= s.ETA for Ly in e)
            div.setdefault(arm, []).append(d)
            P(f"- {arm} {sd} ({r.get('outcome') if r.get('ok') else 'DIVERGED'}): {traj} | end CTX/KEY/VAL "
              + " ".join("/".join(f"{Ly['by_role'][k]:.2f}" for k in ("ctx", "key", "val")) for Ly in e)
              + " | >1 " + "/".join(f"{Ly['frac_gt1']:.2f}" for Ly in e)
              + " | η² KEY " + "/".join(f"{Ly['eta_key_by_stream']:.2f}" for Ly in e)
              + " | spread " + "/".join(f"{max(Ly['key_by_stream']) - min(Ly['key_by_stream']):.2f}" for Ly in e)
              + (" | D" if d else ""))
    P("\nβ diverges by stream (end): " + "; ".join(f"{a} {sum(v)}/{len(v)}" for a, v in div.items()))
    out["div"] = {a: sum(v) for a, v in div.items()}
    P("\nSplits per run (update c*->c0 ok?):\n")
    for arm in arms:
        P(f"- {arm}: " + "; ".join(
            f"{sd} {r.get('outcome')}" + (f"@{r['transition']}" if r.get('transition') else f" {r['acc']:.2f}") + " ["
            + ", ".join(f"{c['step']} {c['key_cs']}->{c['key_c0']} {'ok' if c.get('key_ok') else 'NOT ok'}"
                        for c in r.get("checks", []) if c.get("fired")) + "]"
            for sd, r in sorted(R[arm].items()) if r.get("ok")))
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed or diverged records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU "
      f"{meta['cpus']}; code SHA {c16.STATE['shas'][s.NAME][:12]}")
    out["md"] = _write("write", L)
    return out


# ── S88b ─────────────────────────────────────────────────────────────────────
def report_s88b():
    import explore_f_horizon as s
    R = _runs(s)
    L = []
    P = L.append
    out = {}
    P("### S88b the horizon (one delta channel, RandHeaderTask-lite at 1x and 2x cue-to-key distance, 24000; seeds 2420-2429; "
      "BOUND; Wilson 95%)\n")
    P("Validity (the perfect gate binds on both seeds at each length):\n")
    _validity(P, R, ("O_FIXED_1X", "O_FIXED_2X"))
    v1 = sum(bound(r) for r in R["O_FIXED_1X"].values()) == 2
    v2 = sum(bound(r) for r in R["O_FIXED_2X"].values()) == 2
    P("\n| arm | n | BOUND | transitions (median; all) | final accuracy (median) | accuracy by cue-to-key distance at the end "
      "(median over runs) |")
    P("|---|---|---|---|---|---|")
    for arm in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X"):
        rs = R[arm]
        bd = {}
        for r in rs.values():
            ab = r.get("acc_by_distance") or {}
            if ab:
                for d, (n, a) in ab[max(ab, key=int)].items():
                    bd.setdefault(int(d), []).append(a)
        P(f"| {arm} | {len(rs)}/10 | {fc.count_str(sum(bound(r) for r in rs.values()), len(rs))} | {_trs(rs)} | "
          f"{fmt(fc.med([r['acc'] for r in rs.values() if r.get('ok')]))} | "
          + ", ".join(f"{d}: {fmt(fc.med(v))}" for d, v in sorted(bd.items())) + " |")
    B = {a: {sd: bound(r) for sd, r in R[a].items()} for a in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X")}
    within = {L_: fc.paired(B[f"ROUTED_{L_}"], B[f"FIXED_{L_}"], s.SEEDS) for L_ in s.LENGTHS}
    P("\nWithin a length (b = ROUTED alone, c = FIXED alone):\n")
    for L_, d in within.items():
        P(f"- {L_}: ROUTED {d['new']}/{d['n']} vs FIXED {d['old']}/{d['n']}: b {d['b']}, c {d['c']}; p(ROUTED > FIXED) = "
          f"{d['p_new_gt_old']:.3g}, p(FIXED > ROUTED) = {d['p_old_gt_new']:.3g}")

    def drop(x):
        a, b = B[f"{x}_1X"], B[f"{x}_2X"]
        ss = [sd for sd in s.SEEDS if sd in a and sd in b]
        return sum(1 for sd in ss if a[sd] and not b[sd]) - sum(1 for sd in ss if b[sd] and not a[sd])
    dF, dR = drop("FIXED"), drop("ROUTED")
    P(f"\nAcross lengths (same seed numbers): drop_FIXED = {dF}, drop_ROUTED = {dR}")
    complete = all(len(R[a]) == 10 for a in B) and all(len(R[f"O_FIXED_{L_}"]) == 2 for L_ in s.LENGTHS)
    n1F, n1R = within["1X"]["old"], within["1X"]["new"]
    if not complete:
        rd = ["INCOMPLETE, no reading"]
    elif not v1:
        rd = ["UNTESTED (the perfect gate did not bind on both seeds at 1x)"]
    elif n1F < 8 or n1R < 8:
        rd = [f"untested (at 1x FIXED {n1F}/10, ROUTED {n1R}/10; rule >= 8/10 each)"]
    elif not v2:
        rd = ["UNTESTED at 2x (the perfect gate did not bind on both seeds at 2x)"]
    else:
        rd = []
        d2 = within["2X"]
        if d2["new"] - d2["old"] >= 0 and d2["b"] >= 4 and d2["c"] == 0:
            rd.append("the forget gate extends the horizon")
        if dF >= 4 and abs(dR - dF) <= 1:
            rd.append("the horizon is the decay's")
        if not rd:
            rd = ["no reading applies"]
    P(f"\nREADING S88b: **{'; '.join(rd)}** (rules: untested if at 1x either arm < 8/10; extends if at 2x b >= 4 and c = 0; "
      f"the decay's if drop_FIXED >= 4 and |drop_ROUTED − drop_FIXED| <= 1)")
    out["rd"] = rd
    P("\nThe learned decay at the end (ROUTED; median over runs; α at CTX / KEY / VAL / FILL per layer) and Δ:\n")
    for arm in ("ROUTED_1X", "ROUTED_2X"):
        rows = [r for r in R[arm].values() if r.get("decay_by_role")]
        if not rows:
            continue
        cells = []
        for ly in range(3):
            roles = sorted({k for r in rows for k in r["decay_by_role"][ly]})
            cells.append("L%d " % (ly + 1) + " ".join(f"{k} {fmt(fc.med([r['decay_by_role'][ly][k] for r in rows if k in r['decay_by_role'][ly]]), 3)}"
                                                       for k in roles))
        P(f"- {arm}: " + "; ".join(cells) + f"; Δ median {fmt(fc.med([r['decay_params']['Delta'] for r in rows]), 3)} "
          f"(init {fmt(-2.6037, 3)}), |w| median {fmt(fc.med([r['decay_params']['w_norm'] for r in rows]), 3)}")
    P("\nPer run (B = bound @transition, else final accuracy; write order: accuracy when the queried pair was the last write "
      "of its key / not):\n")
    for arm in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X"):
        P(f"- {arm}: " + "; ".join(
            f"{sd} " + (f"B@{r['transition']}" if bound(r) else f"{r['acc']:.2f}")
            + (f" ({fmt(r['write_order'].get('acc_last'))}/{fmt(r['write_order'].get('acc_notlast'))})" if r.get("write_order") else "")
            for sd, r in sorted(R[arm].items()) if r.get("ok")))
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    out["md"] = _write("horizon", L)
    return out
