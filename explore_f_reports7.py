#!/usr/bin/env python
"""
explore_f_reports7.py — EXPLORATORY, not a result. Report tables of Session F's round-7 screen (explore_f_s90_ctx), with the
rules and readings exactly as its docstring fixes them. Writes explore_out/F/s90_ctx_tables.md.
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


def _accept_store_cpus(s):
    """The report reads every CPU this screen's records are on (each stage's check() reproduced the others first)."""
    st = fc.load(s)
    for c in sorted({r.get("cpu") for r in st["runs"].values() if r.get("ok")} - {None}):
        if c != c16.STATE["batch_cpu"] and c not in c16.STATE["other_cpus"]:
            c16.STATE["other_cpus"].append(c)


def _cell(o, a, r):
    """one run's outcome: bound @t / unbound acc / FAILED / --."""
    if r is not None:
        return f"@{r['transition']}" if bound(r) else f"unbound {r['acc']:.2f}"
    return "FAILED" if o is False else "--"


def report_s90():
    import torch
    import explore_f_s90_ctx as s
    import explore_f_tasks as ft
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    _accept_store_cpus(s)
    d = s.decision()
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    tasks = {"CTXPOOL": ft.RandHeaderTaskCtxPool(), "DOCS3": ft.RandHeaderTaskDocs(3), "DOCS4": ft.RandHeaderTaskDocs(4)}
    L = []
    P = L.append
    P("### S90-F the context-pool family (seeds 3060-3071)\n")
    P("Token-type frequencies over S87's sample (4096 sequences, torch.Generator(87_000), input tokens), both cuts, and the "
      "cap of a fixed type -> channel map (the fraction of sequences whose D contexts land on D distinct channels; exact):\n")
    P("| task | CTX types | KEY types | VAL types | largest-ratio cut | half-mass cut | cap at k = D | k = 2D | k = 16 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for t, T in tasks.items():
        r = ft.freq_report(T)
        br = r["by_role"]
        cell = lambda k: f"{br[k]['n']} ({br[k]['min']:.4f}-{br[k]['max']:.4f})"
        cnt = lambda roles: ", ".join(f"{roles.count(x)} {x}" for x in ("CTX", "KEY", "VAL") if roles.count(x))
        caps = [ft.type_map_cap(T.n_ctx, T.S, k) for k in (T.S, 2 * T.S, 16)]
        P(f"| {t} (D = {T.S}) | {cell('CTX')} | {cell('KEY')} | {cell('VAL')} | {cnt(r['largest_ratio_roles'])} | "
          f"{cnt(r['half_mass_roles'])} (mass {r['half_mass_mass']:.3f}) | {caps[0][2]}/{caps[0][3]} = {caps[0][0]:.3f} | "
          f"{caps[1][0]:.3f} | {caps[2][0]:.3f} |")
    P("\nPhase 0, no-memory solvability (held-out accuracy at the end, seeds 3060 / 3061; rule: stopped if > 0.6), with "
      "P(the target pair inside the reach) and the positional bound P + (1 - P)/(n_vals - S + 1):\n")
    P("| task | NOMEM (reach 4) | bound | NOMEM_SELF (reach 10) | bound | stopped |")
    P("|---|---|---|---|---|---|")
    for t, T in tasks.items():
        x = T.make_batch(4096, torch.Generator().manual_seed(87_000))[0]
        qp = T.qpos[0]
        lab = T.stream_labels(x)
        hit = (x[:, :qp - 1] == x[:, qp][:, None]) & (lab[:, :qp - 1] == lab[:, qp][:, None])
        kp = torch.where(hit, torch.arange(qp - 1).expand_as(hit), torch.full_like(hit, -1, dtype=torch.long)).max(1).values
        bd = {}
        for reach in (4, 10):
            seen = (kp >= qp - reach + 1).double().mean().item()
            bd[reach] = f"P {seen:.3f}, bound {seen + (1 - seen) / (T.n_vals - T.S + 1):.3f}"
        cells = []
        for v in ("NOMEM", "NOMEM_SELF"):
            rs = R.get(f"{v}_{t}", {})
            cells.append(" / ".join(f"{rs[sd]['acc']:.3f}" if sd in rs else "--" for sd in s.P0_SEEDS))
        st = d.get("phase0", {}).get(t, {}).get("stopped")
        P(f"| {t} | {cells[0]} | {bd[4]} | {cells[1]} | {bd[10]} | {st} |")
    P("\nPhase 1, validity pilots (the perfect gate handed the per-sequence index, k = S, SLOW; seeds 3060 / 3061; "
      "transition or final accuracy), and the readings per task and memory:\n")
    P("| task | memory | 24000 | 48000 (fallback) | reading |")
    P("|---|---|---|---|---|")
    rd = {}
    for t in s.TASKS:
        for m in s.MEMS:
            a24, a48 = f"O_{m}_{t}", f"O_{m}_{t}_48K"
            c24 = "; ".join(_cell(None, None, R.get(a24, {}).get(sd)) for sd in s.P1_SEEDS) if a24 in R else "--"
            c48 = "; ".join(_cell(None, None, R.get(a48, {}).get(sd)) for sd in s.P1_SEEDS) if a48 in R else "not run"
            rd[(t, m)] = s.reading_task(d, t, m)
            P(f"| {t} | {m} | {c24} | {c48} | **{rd[(t, m)]}** |")
    P("\nThe type-level perfect gate (balanced map of the 16 types to k = S channels; 24000; descriptive): final accuracy; end "
      "fraction of probe sequences separated / the cap; end margin; sep_key_inv; VAL cos within sequences:\n")
    P("| task | memory | 3060 | 3061 | separated (cap) | margin | sep_key_inv | VAL cos (within) |")
    P("|---|---|---|---|---|---|---|---|")
    for t in s.TASKS:
        T = tasks[t]
        cap = ft.type_map_cap(T.n_ctx, T.S, T.S)[0]
        for m in s.MEMS:
            a = f"TL_{m}_{t}"
            if a not in R:
                continue
            rs = R[a]
            ends = [(rs[sd].get("end") or {}) for sd in s.P1_SEEDS if sd in rs]
            mean = lambda f: (sum(e.get(f, float("nan")) for e in ends) / len(ends)) if ends else None
            P(f"| {t} | {m} | " + " | ".join(_cell(None, None, rs.get(sd)) for sd in s.P1_SEEDS)
              + f" | {fmt(mean('seq_separated_frac'))} ({cap:.3f}) | {fmt(mean('margin'))} | {fmt(mean('sep_key_inv'))} | "
              f"{fmt(mean('val_cos_inv_k'))} |")
    P("\nPhase 2, the screen (one channel, no gate; seeds 3062-3071 paired; at each (task, memory)'s budget):\n")
    P("| task | arm | budget | BOUND | transitions (median; all) | unbound: final acc | write order: last / not last |")
    P("|---|---|---|---|---|---|---|")
    single = {}
    for t in s.TASKS:
        for m in s.MEMS:
            a = f"SINGLE_{m}_{t}"
            if a not in s.ARMS:
                continue
            o, acc = s.outcomes(a, s.P2_SEEDS)
            rs = R[a]
            n = sum(1 for x in o.values() if x is not None)
            nb = sum(1 for x in o.values() if x)
            single[(t, m)] = o
            ub = ", ".join(f"{sd} {acc[sd]:.2f}" for sd in s.P2_SEEDS if o[sd] is False) or "none"
            wo = [rs[sd].get("write_order") or {} for sd in s.P2_SEEDS if sd in rs]
            mw = lambda f: sum(w.get(f) or 0 for w in wo) / len(wo) if wo else None
            P(f"| {t} | {a} | {s.ARMS[a]['iters']} | {fc.count_str(nb, n) if n else '--'} | {_trs(rs)} | {ub} | "
              f"{fmt(mw('acc_last'))} / {fmt(mw('acc_notlast'))} |")
    P("")
    for t in s.TASKS:
        if (t, "DELTA") in single and (t, "HEBBCONV") in single:
            dd, hh = single[(t, "DELTA")], single[(t, "HEBBCONV")]
            if all(v is not None for v in list(dd.values()) + list(hh.values())):
                pr = fc.paired(dd, hh, s.P2_SEEDS)
                P(f"- {t}: SINGLE_DELTA {pr['new']}/{pr['n']} vs SINGLE_HEBBCONV {pr['old']}/{pr['n']}: delta only {pr['b']}, "
                  f"Hebbian + conv only {pr['c']}; exact one-sided McNemar p (delta > Hebbian) = {pr['p_new_gt_old']:.4g}, "
                  f"(Hebbian > delta) = {pr['p_old_gt_new']:.4g}")
    P("\nPer run (phase 2): seed outcome (B@t bound at t; else final accuracy):\n")
    for t in s.TASKS:
        for m in s.MEMS:
            a = f"SINGLE_{m}_{t}"
            if a in R:
                one = lambda r: f"B@{r['transition']}" if bound(r) else f"{r['acc']:.2f}"
                P(f"- {a}: " + "; ".join(f"{sd} {one(r)}" for sd, r in sorted(R[a].items())))
    scr = s.reading_screen(d)
    P(f"\nREADINGS per task and memory: " + "; ".join(f"**{t} {m}: {v}**" for (t, m), v in rd.items()))
    P(f"\nSCREEN READING (CTXPOOL, one delta channel): **{scr}**")
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPUs {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}; decision {d}")
    return dict(rd=rd, screen=scr, md=_write("s90_ctx", L))


def report_s90_k16():
    """explore_f_s90_k16: the type-level perfect gate at k = 16, beside the per-sequence gate (k = 2). Writes
    explore_out/F/s90_k16_tables.md."""
    import explore_common as ec
    import explore_f_s90_k16 as s
    if s.NAME not in c16.STATE["shas"]:
        fc.setup([s])
    _accept_store_cpus(s)
    R = {arm: fc.runs(s, arm) for arm in s.ARMS}
    st = fc.load(s)
    failed = [k for k, r in st["runs"].items() if not r.get("ok") and r.get("attempts", 1) >= c16.MAX_ATTEMPTS]
    L = []
    P = L.append
    P("### The type-level perfect gate at k = 16 (explore_f_s90_k16; seed 3060; SLOW)\n")
    P("| task | memory | budget | k = 16 type-level gate (3060) | end margin / separated | per-sequence gate, k = 2 (2/2) |")
    P("|---|---|---|---|---|---|")
    nb = 0
    for arm in s.ARMS:
        r = R[arm].get(3060)
        _, t = arm.split("_")[1], arm.split("_")[2]
        cell = (f"bound @{r['transition']} (final {r['acc']:.3f})" if bound(r) else f"unbound {r['acc']:.3f}") if r else "--"
        nb += bound(r)
        e = (r or {}).get("end") or {}
        store, a2, seeds, sha = s.PER_SEQ[arm]
        d2 = ec.load_store(store)["runs"]
        cells = []
        for sd in seeds:
            k2 = [k for k in d2 if k.startswith(f"{a2}|{sd}|{sha}|") and d2[k].get("ok")]
            r2 = d2[k2[0]] if k2 else None
            cells.append(f"{sd} " + (f"@{r2['transition']}" if bound(r2) else (f"unbound {r2['acc']:.2f}" if r2 else "--")))
        P(f"| {t} | {arm.split('_')[1]} | {s.ARMS[arm]['iters']} | {cell} | {fmt(e.get('margin'))} / "
          f"{fmt(e.get('seq_separated_frac'))} | {a2}: {'; '.join(cells)} |")
    P("\nCurves (held-out accuracy every 2400):\n")
    for arm in s.ARMS:
        for sd, r in sorted(R[arm].items()):
            P(f"- {arm}|{sd}: " + " ".join(f"{c[1]:.2f}" for c in r["curve"][1::2]))
    n = sum(len(v) for v in R.values())
    rd = s.reading(nb, len(failed), n)
    P(f"\nREADING (fixed before the runs): **{rd}**")
    meta = store_meta(s)
    P(f"\n{s.NAME}: {meta['n']} runs ok (failed records {meta['failed']}), {meta['hours']:.2f} h of run time; CPUs {meta['cpus']}; "
      f"code SHA {c16.STATE['shas'][s.NAME][:12]}")
    return dict(reading=rd, md=_write("s90_k16", L))
