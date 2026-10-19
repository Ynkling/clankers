#!/usr/bin/env python
"""
explore_f_reports6.py — EXPLORATORY, not a result. Report tables of Session F's round-6 screen (explore_f_s89_tasks), with the
rules and readings exactly as its docstring fixes them. Writes explore_out/F/s89_tasks_tables.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc
from explore_f_reports import fmt, bound, store_meta
from explore_f_reports3 import _write
from explore_f_reports4 import _trs


def routed_end(r):
    e = (r or {}).get("end") or {}
    return bool(e.get("margin", 0) >= 0.9 and e.get("eta_key_by_stream", 0) > 0.9)


def report_s89():
    import explore_f_s89_tasks as s
    import explore_f_tasks as ft
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    d = s.decision()
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    L = []
    P = L.append
    out = {}
    P("### S89-F the lite task without marked keys (seeds 2800-2811)\n")
    P("Token-type frequencies over S87's sample (4096 sequences, torch.Generator(87_000), input tokens) and the two cuts:\n")
    P("| task | CTX types (frequency) | KEY types | VAL types | largest-ratio cut selects | half-mass cut selects |")
    P("|---|---|---|---|---|---|")
    tasks = {"LITE": ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), "POOL": ft.RandHeaderTaskPool(),
             "ALLPOOL": ft.RandHeaderTaskAllPool()}
    for t, T in tasks.items():
        r = ft.freq_report(T)
        br = r["by_role"]
        cell = lambda k: f"{br[k]['n']} ({br[k]['min']:.4f}-{br[k]['max']:.4f})"
        cnt = lambda roles: ", ".join(f"{roles.count(x)} {x}" for x in ("CTX", "KEY", "VAL") if roles.count(x))
        P(f"| {t} | {cell('CTX')} | {cell('KEY')} | {cell('VAL')} | {cnt(r['largest_ratio_roles'])} | "
          f"{cnt(r['half_mass_roles'])} (mass {r['half_mass_mass']:.3f}) |")
    P("\nPhase 0, positional solvability (no memory; held-out accuracy at the end, seeds 2800 / 2801; rule: stopped if > 0.6):\n")
    P("| task | NOMEM (zero read) | NOMEM_SELF (own V only) | stopped |")
    P("|---|---|---|---|")
    for t in ("LITE",) + s.TASKS:
        cells = []
        for v in ("NOMEM", "NOMEM_SELF"):
            rs = R.get(f"{v}_{t}", {})
            cells.append(" / ".join(f"{rs[sd]['acc']:.3f}" if sd in rs else "--" for sd in s.P0_SEEDS))
        st = d.get("phase0", {}).get(t, {}).get("stopped") if t != "LITE" else "(reference)"
        P(f"| {t} | {cells[0]} | {cells[1]} | {st} |")
    P("\nPhase 1, validity pilots (the perfect gate, SLOW; transition per seed, or final accuracy if unbound):\n")
    rd = {}
    for t in s.TASKS:
        if d.get("phase0", {}).get(t, {}).get("stopped"):
            rd[t] = "solvable without memory (stopped)"
            P(f"- {t}: stopped by phase 0")
            continue
        for suf in ("", "_48K"):
            for m in ("HEBB", "DELTA"):
                arm = f"O_{m}_{t}{suf}"
                if arm not in R:
                    continue
                rs = R[arm]
                P(f"- {arm}: " + "; ".join(f"{sd} " + (f"bound @{r['transition']}" if bound(r) else f"unbound {r['acc']:.2f}")
                                           for sd, r in sorted(rs.items())))
        b = d.get("budget", {}).get(t)
        rd[t] = ("the task is valid" if b == 24000 else "the task needs the longer budget" if b == 48000 else
                 "untested" if b == "UNTESTED" else "INCOMPLETE, no reading")
    P("\nREADINGS (per task): " + "; ".join(f"**{t}: {v}**" for t, v in rd.items()))
    out["rd"] = rd
    P("\nPhase 2, baselines (seeds 2802-2811; no reading; ROUTED*@end = end margin >= 0.9 and η² by stream at KEY > 0.9):\n")
    P("| task | arm | n | BOUND | ROUTED-BOUND | DISCOVERED | transitions (median; all) | unbound: final acc / class |")
    P("|---|---|---|---|---|---|---|---|")
    for t in s.TASKS:
        for a in ("BASELINE", "SINGLE_HEBB", "SINGLE_DELTA"):
            arm = f"{a}_{t}"
            if arm not in R:
                continue
            rs = R[arm]
            n = len(rs)
            nb = sum(bound(r) for r in rs.values())
            rb = sum(bound(r) and routed_end(r) for r in rs.values()) if a == "BASELINE" else None
            dc = sum(bool(r.get("discovered")) for r in rs.values()) if a == "BASELINE" else None
            ub = ", ".join(f"{sd} {r['acc']:.2f}" + (f" {r.get('fail')}" if r.get("fail") else "")
                           for sd, r in sorted(rs.items()) if not bound(r)) or "none"
            P(f"| {t} | {a} | {n}/10 | {fc.count_str(nb, n)} | {fc.count_str(rb, n) if rb is not None else '--'} | "
              f"{fc.count_str(dc, n) if dc is not None else '--'} | {_trs(rs)} | {ub} |")
    P("\nPer run (BASELINE: end margin / η² by stream at KEY / η² by key at KEY; B = bound):\n")
    for t in s.TASKS:
        arm = f"BASELINE_{t}"
        if arm in R:
            P(f"- {arm}: " + "; ".join(
                f"{sd}{' B@' + str(r['transition']) if bound(r) else ''} {fmt((r.get('end') or {}).get('margin'))}/"
                f"{fmt((r.get('end') or {}).get('eta_key_by_stream'))}/{fmt((r.get('end') or {}).get('eta_key_by_key'))}"
                for sd, r in sorted(R[arm].items())))
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPU {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}; decision {d}")
    out["md"] = _write("s89_tasks", L)
    return out
