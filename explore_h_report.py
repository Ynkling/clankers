#!/usr/bin/env python
"""
explore_h_report.py — EXPLORATORY, not a result. Session H's report writer: one markdown report per batch from the saved
runs (no runs). Batch 1 = S58; batch 2 = S58b + S60. Counts table first; the pre-fixed readings and which applies;
CHECK results; runtime; anything unexpected (from --notes, a text file written by hand after reading the log).

Run:  python explore_h_report.py 1 [--notes FILE]    -> explore_out/H/report_1.md
      python explore_h_report.py 2 [--notes FILE]    -> explore_out/H/report_2.md
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

OUT = os.path.join(ec.OUT_DIR, "H")


def setup(mods):
    c16.STATE["batch_cpu"] = ec.cpu_model()
    c16.STATE["other_cpus"] = []
    for m in mods:
        st = ec.load_store(m.NAME)
        c16.STATE["shas"][m.NAME] = st["meta"]["code_shas"][-1]


def fmt_w(w):
    return f"{w[0]:.2f}-{w[1]:.2f}"


def fmt_t(x):
    return "--" if x is None else f"{x:.0f}"


def runtime(m):
    st = ec.load_store(m.NAME)
    rs = [r for r in st["runs"].values() if r.get("ok")]
    tot = sum(r.get("secs_wall", 0) for r in rs) / 3600
    return len(rs), tot, sorted({r.get("cpu") for r in rs}), sorted({r.get("code_sha", "")[:12] for r in rs})


def checks_line(name):
    with open(os.path.join(OUT, "checks.json")) as f:
        c = json.load(f).get(name)
    return c


def s58_section(L):
    import explore_h_collapse as s58
    setup([s58])
    res, got = s58.report()
    A = res["arms"]
    L.append("## S58 explore_h_collapse — S=4, P=4, k=16, conv, 28800 updates, LOCAL3 + SLOW, seeds 420-459\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | BOUND (any) | failure classes | MERGED | transition median (range) | "
             "unbound@14400 → bound | flat@14400 → bound | BR & 1:1 at KEY | margin (median, end) | ops fired (runs) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ("SPLIT", "RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH", "NONE"):
        x = A[a]
        cl = ", ".join(f"{k} {v}" for k, v in sorted(x["classes"].items())) or "—"
        rng = f"{fmt_t(x['trans_med'])} ({fmt_t(x['trans_min'])}-{fmt_t(x['trans_max'])})"
        ops = f"{x['ops']} ({x['ops_runs']})" if a in ("SPLIT", "RESET") else "—"
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {x['bound']}/{x['n']} | {cl} | {x['merged']} | "
                 f"{rng} | {x['unbound14']} → {x['unbound14_bind']} | {x['flat14']} → {x['flat14_bind']} | {x['br_key_too']}/{x['br']} | "
                 f"{'--' if x['margin_med'] is None else round(x['margin_med'], 3)} | {ops} |")
    o = res["oracle"]
    L.append(f"\nORACLE (perfect gate ceiling4k16 + conv, seeds 420-421): bound {o['bound']}/{o['n']}, transitions "
             f"{o['transitions']} → screen {'VALID' if o['valid'] else 'UNTESTED'}.\n")
    L.append("**Paired comparisons** (BOUND ROUTED, 40 seeds; x only / y only; exact one-sided McNemar p for x > y and y > x):\n")
    L.append("| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |")
    L.append("|---|---|---|---|---|---|---|")
    for (x, y), c in res["cmp"].items():
        L.append(f"| {x} vs {y} | {c['x']}/{c['n']} | {c['y']}/{c['n']} | {c['b']} | {c['c']} | {c['p']:.3g} | {c['p_rev']:.3g} |")
    L.append("\n**Readings (fixed in explore_h_collapse's docstring before any run):**\n")
    names = {"R1": "noise replaces the split", "R2": "the split is the reset", "R3": "the balance loss helps"}
    for k, v in res["readings"].items():
        L.append(f"- {k} ({names[k]}): **{v}**")
    return res, got


def s58b_section(L):
    import explore_h_eight as s58b
    s58b.configure()
    setup([s58b])
    res, got = s58b.report()
    A = res["arms"]
    L.append("## S58b explore_h_eight — S=8, P=4, k=16, conv, 43200 updates, LOCAL3 + SLOW, seeds 460-469\n")
    L.append(f"Selected from S58 by the fixed rule: {res['selected']} (S58 basis {res['basis']}).\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | BR by 28800 | BOUND (any) | failure classes | MERGED | transition median "
             "(all) | BR & 1:1 at KEY | unbound maps at end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, x in A.items():
        cl = ", ".join(f"{k} {v}" for k, v in sorted(x["classes"].items())) or "—"
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {x['br_by_28800']} | {x['bound']}/{x['n']} | {cl} | "
                 f"{x['merged']} | {fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['br_key_too']}/{x['br']} | "
                 f"{' '.join(x['maps_end']) or '—'} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8 + conv, seeds 460-461): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for a, c in res["cmp"].items():
        L.append(f"- {a} vs SPLIT: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {a} only {c['b']}, SPLIT only {c['c']}; p({a}>SPLIT) "
                 f"{c['p']:.3g}, p(SPLIT>{a}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_eight's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    return res, got


def s60_section(L):
    import explore_h_stability as s60
    setup([s60])
    res, got = s60.report()
    A = res["arms"]
    L.append("## S60 explore_h_stability — S=2, P=4, k=2, no conv, 24000 updates, LOCAL3 + SLOW, seeds 470-479\n")
    L.append("| arm | DISCOVERED | band | Wilson 95% | BOUND | bound not disc. | failures | transition median (all) | sat. evals before "
             "trans. | at/after | unbound runs | runs sat. before binding | single-final BOUND | runs with flips | tau at end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, x in A.items():
        fl = ", ".join(f"{k} {v}" for k, v in sorted(x["fails"].items())) or "—"
        tau = (f"{min(x['tau_end']):.2f}-{max(x['tau_end']):.2f}" if x["tau_end"] else "—")
        L.append(f"| {a} | {x['disc']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {x['bound']} | {x['not_disc_bound']} | {fl} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['pre'][0]}/{x['pre'][1]} | {x['post'][0]}/{x['post'][1]} | "
                 f"{x['unbound'][0]}/{x['unbound'][1]} | {x['sat_before_runs']} | {x['single_final']} | {x['flips']} | {tau} |")
    o = res["oracle"]
    L.append(f"\nORACLE_CONV (ceiling_conv, seeds 470-471): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}; ORACLE_PLAIN (printed): transitions {o['plain']}, final acc {o['plain_acc']}.\n")
    for a, c in res["cmp"].items():
        L.append(f"- {a} vs REF (DISCOVERED): {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {a} only {c['b']}, REF only {c['c']}; "
                 f"p({a}>REF) {c['p']:.3g}, p(REF>{a}) {c['p_rev']:.3g}")
    L.append("\n**Readings (descriptive, fixed in explore_h_stability's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    return res, got


def d8_table(L, res, arms, extra=None):
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | BOUND (any) | failure classes | transition median (all BR) | ops fired |"
             + (" " + extra[0] + " |" if extra else ""))
    L.append("|---|---|---|---|---|---|---|---|" + ("---|" if extra else ""))
    for a in arms:
        x = res["arms"][a]
        cl = ", ".join(f"{k} {v}" for k, v in sorted(x["classes"].items())) or "—"
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {x['bound']}/{x['n']} | {cl} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x.get('ops', '—')} |"
                 + (f" {extra[1](a)} |" if extra else ""))


def s61_section(L):
    import explore_h_copy2x2 as m
    setup([m])
    res, got = m.report()
    L.append("## S61 explore_h_copy2x2 — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + one plateau trigger, seeds 480-489\n")
    d8_table(L, res, ("SPLIT", "RESET", "COPY", "COPY_NONOISE", "NONE"))
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 480-481): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    L.append("| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |")
    L.append("|---|---|---|---|---|---|---|")
    for (x, y), c in res["cmp"].items():
        L.append(f"| {x} vs {y} | {c['x']}/{c['n']} | {c['y']}/{c['n']} | {c['b']} | {c['c']} | {c['p']:.3g} | {c['p_rev']:.3g} |")
    L.append("\n**Readings (fixed before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    return m


def s63_section(L):
    import explore_h_hardness as m
    setup([m])
    res, got = m.report()
    L.append("## S63 explore_h_hardness — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + KEYMASS, seeds 490-499\n")
    d8_table(L, res, ("REF", "ANNEAL", "HARD_W"),
             ("earlier than REF (both BR)", lambda a: "—" if a == "REF" else f"{res['earlier'][a]}/10"))
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 490-491): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for a, c in res["cmp"].items():
        L.append(f"- {a} vs REF: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {a} only {c['b']}, REF only {c['c']}; p({a}>REF) "
                 f"{c['p']:.3g}, p(REF>{a}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    return m


def s62_section(L):
    import explore_h_query_gate as m
    setup([m])
    res, _ = m.report()
    L.append("## S62 explore_h_query_gate — the query-KEY gate vs the body maps, reruns of S58 / S58b's bound runs\n")
    L.append("Per run at its stop: query map (stream -> argmax of the read gate at the query KEY position) = body VAL map, = body "
             "KEY map, both, or neither. margin_val = query gate mass on its stream's VAL-map channel minus mean mass on the "
             "other streams' VAL-map channels.\n")
    L.append("| screen.arm | bound runs | reproduced bit for bit | BOUND ROUTED: both / VAL only / KEY only / neither | "
             "BOUND NOT routed: both / VAL only / KEY only / neither | margin_val (median) | harness margin (median) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        f = lambda c: " / ".join(str(c.get(k, 0)) for k in ("both", "VAL only", "KEY only", "neither"))  # noqa: E731
        L.append(f"| {r['arm']} | {r['n']} | {r['reproduced']}/{r['got']} | {f(r['cnt']['BOUND ROUTED'])} | "
                 f"{f(r['cnt']['BOUND NOT routed'])} | {'--' if r['margin_val'] is None else round(r['margin_val'], 3)} | "
                 f"{'--' if r['margin'] is None else round(r['margin'], 3)} |")
    return m


def cls_str(d):
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "—"


def s78_section(L):
    import explore_h_two as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S78 explore_h_two — S=2, P=4, k=2 (K4: k=4), no conv, 24000, WIN3_SLOW path, seeds 1300-1319\n")
    L.append("| arm | outcome | count | band | Wilson 95% | bound, not successful | unbound: classes v1 | unbound: classes v2 | "
             "transition median (all) | splits fired (runs) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4"):
        x = A[a]
        L.append(f"| {a} | {'BOUND ROUTED' if a == 'K4' else 'DISCOVERED'} | {x['k']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | "
                 f"{x['bound_not']} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | {fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | "
                 + (f"{x['ops']} ({x['ops_runs']})" if a.startswith("SPLIT") else "—") + " |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceiling_conv, seeds 1300-1301): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    L.append("| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |")
    L.append("|---|---|---|---|---|---|---|")
    for (x, y), c in res["cmp"].items():
        L.append(f"| {x} vs {y} | {c['x']}/{c['n']} | {c['y']}/{c['n']} | {c['b']} | {c['c']} | {c['p']:.3g} | {c['p_rev']:.3g} |")
    L.append("\n**Readings (fixed in explore_h_two's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    D = m.split_diag(got)
    L.append(f"\n**Splits** ({len(D)} firings). eta^2 of the read gate at KEY positions by key / by stream, and at VAL by key / "
             "by stream, at the firing check -> right after the copy -> at the next check; run's end tag:\n")
    L.append("| arm | seed | update | KEY key/stream before | after | next | VAL key/stream before | after | next | end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    f = lambda d, a, b: "--" if not d else f"{d[a]:.2f}/{d[b]:.2f}"  # noqa: E731
    for d in D:
        L.append(f"| {d['arm']} | {d['seed']} | {d['step']} | {f(d['before'], 'eta_key_by_key', 'eta_key_by_stream')} | "
                 f"{f(d['after'], 'eta_key_by_key', 'eta_key_by_stream')} | {f(d['next'], 'eta_key_by_key', 'eta_key_by_stream')} | "
                 f"{f(d['before'], 'eta_val_by_key', 'eta_val_by_stream')} | {f(d['after'], 'eta_val_by_key', 'eta_val_by_stream')} | "
                 f"{f(d['next'], 'eta_val_by_key', 'eta_val_by_stream')} | {d['outcome']} |")
    ok, dis, pred = res["v2_property"]
    L.append(f"\n**v1 / v2** on S78's unbound runs: differ on {len(dis)}; exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25: "
             f"{ok}. " + "; ".join(f"{k} {res['classes'][k]['v1']} -> {res['classes'][k]['v2']} (key {res['classes'][k]['eta_key']:.4f}, "
                                  f"margin {res['classes'][k]['margin']:.3f})" for k in dis))
    return m


def s79_section(L):
    import explore_h_constants as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S79 explore_h_constants — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, seeds 1320-1329\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits (max/run) | "
             "labelled ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in ("BASE",) + m.VARIANTS:
        x = A[a]
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['ops']} ({x['ops_max']}) | {x['ops_ok']}/{x['ops']} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 1320-1321): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for v, c in res["cmp"].items():
        L.append(f"- {v} vs BASE: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {v} only {c['b']}, BASE only {c['c']}; p({v}>BASE) "
                 f"{c['p']:.3g}, p(BASE>{v}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_constants' docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    e = res["int1200_earlier"]
    n_e = sum(1 for s, (i, b) in e.items() if i is not None and (b is None or i < b))
    L.append(f"\nINT1200's first split vs BASE's (per seed, update or none): " + ", ".join(f"{s}: {i}/{b}" for s, (i, b) in e.items())
             + f" — earlier on {n_e} of 10.")
    L.append(f"CAP6's firings beyond the third: {A['CAP6']['extra'] or 'none'}. v1/v2 outcome differences: "
             + "; ".join(f"{a} {A[a]['v1v2_differ']}" for a in A if A[a]['v1v2_differ']) + ".")
    return m


def s84_section(L):
    import explore_h_guard as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S84 explore_h_guard — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, Hebbian, seeds 2300-2309\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | labelled ok | "
             "accuracy read at the splits | guard-blocked checks |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ("BASE",) + m.VARIANTS:
        x = A[a]
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['ops']} | {x['ops_ok']}/{x['ops']} | "
                 f"{', '.join(str(v) for v in x['fire_acc'])} | {x['guard_blocked']} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 2300-2301): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for v, c in res["cmp"].items():
        L.append(f"- {v} vs BASE: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {v} only {c['b']}, BASE only {c['c']}; p({v}>BASE) "
                 f"{c['p']:.3g}, p(BASE>{v}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_guard's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append("\nFirst split per seed (update or none): " + "; ".join(
        f"{a} " + " ".join(f"{s}:{A[a]['first'][s]}" for s in m.SEEDS) for a in ("BASE",) + m.VARIANTS))
    L.append("v1/v2 outcome differences: " + ("; ".join(f"{a} {A[a]['v1v2_differ']}" for a in A if A[a]['v1v2_differ']) or "none") + ".")
    return m


def s85_section(L):
    import explore_h_two_b as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S85 explore_h_two_b — S=2, P=4, k=2, no conv, 24000, WIN3_SLOW path, seeds 2310-2329\n")
    L.append("| arm | DISCOVERED | band | Wilson 95% | bound, not discovered | unbound: classes v1 | unbound: classes v2 | "
             "transition median (all) | key split formed (eta^2 by key > 0.5 at KEY), median update [runs] | > 0.5 at the end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in ("REF",) + m.DEVICES:
        x = A[a]
        on = [v[0] for v in x["onset"].values() if v[0] is not None]
        L.append(f"| {a} | {x['k']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {x['bound_not']} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {fmt_t(c16.med(on))} [{len(on)}/{x['n']}] | "
                 f"{sum(1 for v in x['onset'].values() if v[1])} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceiling_conv, seeds 2310-2311): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for a, c in res["cmp"].items():
        L.append(f"- {a} vs REF: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {a} only {c['b']}, REF only {c['c']}; p({a}>REF) {c['p']:.3g}, "
                 f"p(REF>{a}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_two_b's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append("\n**When the key split forms** (first measured update with eta^2 by key at KEY positions > 0.5; '-' never; "
             "* = still > 0.5 at the end), per seed:\n")
    L.append("| seed | " + " | ".join(("REF",) + m.DEVICES) + " |")
    L.append("|---|" + "---|" * (1 + len(m.DEVICES)))
    for sd in m.SEEDS:
        cells = []
        for a in ("REF",) + m.DEVICES:
            v = A[a]["onset"].get(sd)
            r = got[a].get(sd)
            tag = "" if r is None else (" D" if ec.discovered(r) else "")
            cells.append("--" if v is None else (f"{v[0] if v[0] is not None else '-'}{'*' if v[1] else ''}{tag}"))
        L.append(f"| {sd} | " + " | ".join(cells) + " |")
    L.append("(D = DISCOVERED.)")
    ok, dis, pred = res["v2_property"]
    L.append(f"\n**v1 / v2** on S85's unbound runs: differ on {len(dis)}; exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25: "
             f"{ok}. " + "; ".join(f"{k} {res['classes'][k]['v1']} -> {res['classes'][k]['v2']} (key {res['classes'][k]['eta_key']:.4f}, "
                                  f"margin {res['classes'][k]['margin']:.3f})" for k in dis))
    return m


def s86_section(L):
    import explore_h_allmass as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S86 explore_h_allmass — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, Hebbian, seeds 2330-2339\n")
    with open(os.path.join(OUT, "allmass_counts.txt")) as f:
        L.append("**First, the main line's 50 split runs** (explore_h_allmass_counts.py, committed before the screen):\n")
        L.append("```\n" + f.read().rstrip() + "\n```\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | key-label ok | "
             "all-label ok | own pair ok | pairs equal | accuracy read at the splits |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ("BASE", "ALLMASS", "EVAL"):
        x = A[a]
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['ops']} | {x['key_ok']}/{x['ops']} | "
                 f"{x['all_ok']}/{x['ops']} | {x['op_ok']}/{x['ops']} | {x['same']}/{x['ops']} | {', '.join(str(v) for v in x['fire_acc'])} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 2330-2331): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for v, c in res["cmp"].items():
        L.append(f"- {v} vs BASE: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {v} only {c['b']}, BASE only {c['c']}; p({v}>BASE) "
                 f"{c['p']:.3g}, p(BASE>{v}) {c['p_rev']:.3g}")
    pe = res["pooled_eval"]
    L.append(f"- (descriptive) EVAL vs BASE pooled over S84 and S86: {pe['x']}/{pe['n']} vs {pe['y']}/{pe['n']}; EVAL only {pe['b']}, BASE "
             f"only {pe['c']}; p(EVAL>BASE) {pe['p']:.3g}, p(BASE>EVAL) {pe['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_allmass' docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    bad = A["ALLMASS"]["bad_fires"]
    L.append(f"\nALLMASS firings whose own (all-position) pair was not a labelled target (seed, update, key c*->c0, all c*->c0, "
             f"key_ok, all_ok): {bad or 'none'}.")
    L.append("v1/v2 outcome differences: " + ("; ".join(f"{a} {A[a]['v1v2_differ']}" for a in A if A[a]['v1v2_differ']) or "none") + ".")
    return m


def s87_section(L):
    import explore_h_allmass2 as m1
    import explore_h_dose as m2
    setup([m1, m2])
    res, got = m1.report()
    A = res["arms"]
    L.append("## (1) S87a explore_h_allmass2 — S=8, P=4, k=16, conv, 43200, Hebbian, exact copy, seeds 2600-2619 (as S86)\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | key-label ok | "
             "all-label ok | own pair ok | pairs equal | map 1:1 before a split |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ("BASE", "ALLMASS"):
        x = A[a]
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | {x['ops']} | {x['key_ok']}/{x['ops']} | "
                 f"{x['all_ok']}/{x['ops']} | {x['op_ok']}/{x['ops']} | {x['same']}/{x['ops']} | {x['one_to_one_before']} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8, seeds 2600-2601): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    c = res["cmp"]["ALLMASS"]
    L.append(f"- ALLMASS vs BASE (20 new pairs): {c['x']}/{c['n']} vs {c['y']}/{c['n']}; ALLMASS only {c['b']}, BASE only {c['c']}; "
             f"p(ALLMASS>BASE) {c['p']:.3g}, p(BASE>ALLMASS) {c['p_rev']:.3g}")
    pp = res["pooled"]
    L.append(f"- pooled with S86 (30 pairs, no reading): ALLMASS {pp['x']}/{pp['n']}, BASE {pp['y']}/{pp['n']}; ALLMASS only {pp['b']}, "
             f"BASE only {pp['c']}; p(ALLMASS>BASE) {pp['p']:.3g}, p(BASE>ALLMASS) {pp['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_allmass2's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\nALLMASS firings whose own pair was not a labelled target (seed, update, key c*->c0, all c*->c0, key_ok, all_ok): "
             f"{A['ALLMASS']['bad_fires'] or 'none'}; BASE's: {A['BASE']['bad_fires'] or 'none'}. v1/v2 differences: "
             + ("; ".join(f"{a} {A[a]['v1v2_differ']}" for a in A if A[a]['v1v2_differ']) or "none") + ".\n")
    L.append("## (2) The one-to-one guard on the main line's split (no runs)\n")
    with open(os.path.join(OUT, "onetoone_counts.txt")) as f:
        L.append("```\n" + f.read().rstrip() + "\n```\n")
    res2, got2 = m2.report()
    B = res2["arms"]
    L.append("## (3) S87c explore_h_dose — S=2, P=4, k=2, no conv, 24000, WIN3_SLOW + the key term, seeds 2630-2639\n")
    L.append("| arm | lam | window | DISCOVERED | band | Wilson 95% | unbound: v1 | unbound: v2 | transition median (all) | "
             "max eta^2 by key (25..window) | key split > 0.5 at the end | firings (per run) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    import explore_h_dose_child as dc
    for a in ("KEYHINGE",) + m2.VARIANTS:
        x = B[a]
        L.append(f"| {a} | {dc.ARM[a]['lam']} | {dc.ARM[a]['window']} | {x['k']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | "
                 f"{cls_str(x['v1'])} | {cls_str(x['v2'])} | {fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])}) | "
                 f"{x['eta_max']:.3f} | {sum(1 for v in x['onset'].values() if v[1])} | {', '.join(str(f) for f in x['fired'])} |")
    o = res2["oracle"]
    L.append(f"\nORACLE (ceiling_conv, seeds 2630-2631): bound {o['bound']}/{o['n']}, transitions {o['transitions']} → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for v, c in res2["cmp"].items():
        L.append(f"- {v} vs KEYHINGE: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {v} only {c['b']}, KEYHINGE only {c['c']}; "
                 f"p({v}>KEYHINGE) {c['p']:.3g}, p(KEYHINGE>{v}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_dose's docstring before any run):**\n")
    for k, v in res2["readings"].items():
        L.append(f"- {k}: **{v}**")
    ok, dis, pred = res2["v2_property"]
    L.append(f"\nv1 / v2 on S87c's unbound runs: differ on {len(dis)}, exactly the predicted ones: {ok}.")
    return [m1, m2]


def s89_section(L):
    import explore_h_s89 as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S89 explore_h_s89 — S=8, P=4, k=16, conv, 43200, Hebbian, exact copy; keys drawn from a pool of 32; "
             "seeds 2902-2911 paired\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | own pair ok | "
             "key-pair ok | all-pair ok | repeat-pair ok | splits at probe < 0.2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in m.LEARNED:
        x = A[a]
        tr = f"{fmt_t(x['trans_med'])} ({', '.join(str(t) for t in x['trans'])})" if x["trans"] else "--"
        if a in m.TRIGGERED:
            extra = (f"{x['ops']} | {x['op_ok']}/{x['ops']} | {x['key_ok']}/{x['ops']} | {x['all_ok']}/{x['ops']} | "
                     f"{x['rep_ok']}/{x['ops']} | {x['flat']}")
        else:
            extra = "-- | -- | -- | -- | -- | --"
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{tr} | {extra} |")
    o = res["oracle"]
    L.append(f"\nORACLE (ceil8_D8 + conv on the pool task, seeds 2900-2901; run first, gating the learned arms): bound "
             f"{o['bound']}/{o['n']}, transitions {o['transitions']} → {'VALID' if o['valid'] else 'UNTESTED'}.\n")
    for k, c in res["cmp"].items():
        x, y = k.split(" vs ")
        L.append(f"- {k}: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {x} only {c['b']}, {y} only {c['c']}; p({x}>{y}) {c['p']:.3g}, "
                 f"p({y}>{x}) {c['p_rev']:.3g}")
    L.append("\n**Readings (fixed in explore_h_s89's docstring before any run; printed with Referee V's fixes, applied after the "
             "runs to the text and report path only):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\n*Caveat, with every reading:* {res['caveat']}\n")
    L.append("Per trigger arm (V's fix 5): " + "; ".join(res["trigger_lines"].values()) + ".")
    if res.get("repeat_note"):
        L.append(f"\n{res['repeat_note']}: not a second finding (V's fix 2); its runs are ALLMASS's bit for bit on 8 of 10 seeds.")
    L.append(f"\n*The rule's {res['oc']}*")
    ov = res["overlap"]
    if ov:
        L.append(f"\nREPEATMASS's repeat set on its 64-sequence probe: {ov['repeat']} of {ov['positions']} input positions; key "
                 f"positions in it {ov['key_in_repeat']} of {ov['key_positions']}; not key positions {ov['repeat_not_key']} (CTX "
                 f"{ov['repeat_ctx']} of {ov['all_ctx']}, VAL {ov['repeat_val']} of {ov['all_val']}, the query's CTX and KEY "
                 f"{ov['repeat_q_ctx'] + ov['repeat_q_key']}).")
    L.append("\nPairs at the firings (descriptive): " + "; ".join(
        f"{a}: key = all {A[a]['same_key_all']}/{A[a]['ops']}, key = repeat {A[a]['same_key_rep']}/{A[a]['ops']}, all = repeat "
        f"{A[a]['same_all_rep']}/{A[a]['ops']}, map one-to-one before {A[a]['one_to_one_before']}" for a in m.TRIGGERED)
        + ". v1/v2 differences: " + ("; ".join(f"{a} {A[a]['v1v2_differ']}" for a in A if A[a]['v1v2_differ']) or "none")
        + ". Per run: explore_out/H/s89_runs.md.")
    rows = ["# S89 per run (EXPLORATORY, not a result)\n",
            "Each firing: update, probe accuracy, the pair copied (c* -> c0) and whether it is a labelled target (op_ok), the "
            "key-position, all-position and repeat pairs with their labels, and the stream -> channel map before it.\n",
            "| arm | seed | outcome | transition | firings |", "|---|---|---|---|---|"]
    for a in m.LEARNED:
        for s, r in sorted(got[a].items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            fs = "<br>".join(f"{c['step']} ({c['acc']:.2f}): {c['op_cs']}->{c['op_c0']} {'ok' if c['op_ok'] else 'NOT ok'}; key "
                             f"{c['key_cs']}->{c['key_c0']} {c['key_ok']}, all {c['all_cs']}->{c['all_c0']} {c['all_ok']}, rep "
                             f"{c['rep_cs']}->{c['rep_c0']} {c['rep_ok']}; map {c['map_before']}" for c in f) or "--"
            rows.append(f"| {a} | {s} | {r.get('outcome')} | {r.get('transition')} | {fs} |")
    with open(os.path.join(OUT, "s89_runs.md"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return m


def s89c_section(L):
    import explore_h_s89c as m
    setup([m])
    res, got = m.report()
    A = res["arms"]
    L.append("## S89c explore_h_s89c — the trigger's schedule on the 32-key pool task (S=8, k=16, conv, 43200, Hebbian, exact "
             "copy); seeds 2924-2933 paired\n")
    L.append("| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transitions | splits | own pair ok | at probe < 0.2 "
             "| map 1:1 before | cap-blocked checks | last firing per run | held-out accuracy at the end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in m.ARM_NAMES:
        x = A[a]
        L.append(f"| {a} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | {cls_str(x['v1'])} | {cls_str(x['v2'])} | "
                 f"{', '.join(str(t) for t in x['trans']) or '--'} | {x['ops']} | {x['op_ok']}/{x['ops']} | {x['flat']} | "
                 f"{x['one_to_one_before']} | {x['capped']} | {', '.join(str(t) for t in x['last_fire'])} | "
                 f"{', '.join(str(v) for v in x['acc_end'])} |")
    L.append("\nValidity: S89's perfect gate on the same task and code path, 2/2 (at 16800 and 10800).\n")
    for k, c in res["cmp"].items():
        x, y = k.split(" vs ")
        L.append(f"- {k}: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {x} only {c['b']}, {y} only {c['c']}; p({x}>{y}) {c['p']:.3g}, "
                 f"p({y}>{x}) {c['p_rev']:.3g}")
    L.append("\n**Reading (fixed in explore_h_s89c's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\n*Caveat, with every reading:* {res['caveat']}\n")
    L.append(f"*The rule's {res['oc']}*")
    rows = ["# S89c per run (EXPLORATORY, not a result)\n", "Each firing: update (probe accuracy) c* -> c0, labelled ok or not. Then the "
            "probe accuracy at every check.\n", "| arm | seed | outcome | transition | firings | probe at the checks |",
            "|---|---|---|---|---|---|"]
    for a in m.ARM_NAMES:
        for s, r in sorted(got[a].items()):
            f = [c for c in r.get("checks") or [] if c.get("fired")]
            fs = "<br>".join(f"{c['step']} ({c['acc']:.2f}) {c['op_cs']}->{c['op_c0']} {'ok' if c['op_ok'] else 'NOT ok'}" for c in f)
            pr = ", ".join(f"{c['step'] // 1200 * 1.2:g}k {c['acc']:.2f}" for c in r.get("checks") or [])
            rows.append(f"| {a} | {s} | {r.get('outcome')} | {r.get('transition')} | {fs or '--'} | {pr} |")
    with open(os.path.join(OUT, "s89c_runs.md"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return m


def s89b_section(L):
    import explore_h_s89b as m
    setup([m])
    res, got = m.report()
    o = res["oracle"]
    L.append("## S89b explore_h_s89b — the split on delta memory (beta 1, L2 keys) on the 32-key pool task (S=8, k=16, conv, "
             "43200); the delta perfect gate on 2912-2913, learned arms on 2914-2923\n")
    L.append("| run | outcome | transition | held-out accuracy at 1200 / 12000 / 24000 / end | streams at the end | runtime |")
    L.append("|---|---|---|---|---|---|")
    for s, r in sorted(got["ORACLE_DELTA"].items()):
        cv = {c[0]: c[1] for c in r["curve"]}
        sa = r["end"].get("stream_acc") or []
        L.append(f"| ORACLE_DELTA {s} | {r.get('outcome')} | {r.get('transition')} | "
                 f"{' / '.join(f'{cv[t]:.2f}' if t in cv else '--' for t in (1200, 12000, 24000))} / {r['acc']:.3f} | "
                 f"{min(sa):.2f}-{max(sa):.2f}, map {r['end'].get('ch_map')} | {r.get('secs_wall', 0) / 60:.0f} min |")
    L.append(f"\nValidity: ORACLE_DELTA bound {o['bound']}/{o['n']} ({o['transitions']}) → "
             f"{'VALID' if o['valid'] else 'UNTESTED'}. Learned-arm runs: "
             f"{sum(len(got[a]) for a in m.LEARNED)} (queued only after the perfect gate binds both seeds).\n")
    L.append("**Reading (fixed in explore_h_s89b's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\n*Caveat, with every reading:* {res['caveat']}")
    return m


def s90_section(L):
    import explore_h_s90 as m
    setup([m])
    res, got = m.report()
    P = res["phase1"]
    L.append("## S92 (explore_h_s90; the round-7 prompt's name, the key s90 kept) — the oracles at key pools 4 / 8 / 16 / 32 "
             "and 86400 updates, then the recipe at P* (S=8, P=4, k=16, conv)\n")
    L.append("### Phase 1: the perfect gates (seeds 3020-3023 per cell; budget 86400, early stop)\n")
    L.append("| memory | pool | transitions (3020-3023) | bound by 43200 (cut curve) | bound by 86400 | held-out accuracy at the end "
             "| reading |")
    L.append("|---|---|---|---|---|---|---|")
    for mem in m.MEMS:
        for k in m.POOLS:
            c = P[f"{mem}_P{k}"]
            tr = ", ".join(str(c["trans"].get(s)) for s in m.P1_SEEDS)
            ae = ", ".join(f"{c['acc_end'].get(s, float('nan')):.3f}" for s in m.P1_SEEDS)
            L.append(f"| {mem} | {k} | {tr} | {c['k43']}/4 | {c['k86']}/4 | {ae} | {c['reading']} |")
    L.append(f"\n**P* = {res['pstar']}**; budgets {res['budget']}. Runs where the 43200 reading by the cut curve differs from "
             f"the run's own transition: {res['readings'].get('phase 1, 43200 by the cut curve vs the run' + chr(39) + 's transition')}.\n")
    if res["arms"]:
        A = res["arms"]
        L.append(f"### Phase 2: the recipe at pool {res['pstar']} (seeds 3024-3033, paired)\n")
        L.append("| arm | budget | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transitions | splits | own pair ok | "
                 "at probe < 0.2 | map 1:1 before | cap-blocked checks | held-out accuracy at the end |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in m.P2:
            x = A[a]
            L.append(f"| {a} | {res['budget'][m.P2_MEM[a]]} | {x['br']}/{x['n']} | {x['band']} | {fmt_w(x['wilson'])} | "
                     f"{cls_str(x['v1'])} | {cls_str(x['v2'])} | {', '.join(str(t) for t in x['trans']) or '--'} | "
                     + (f"{x['ops']} | {x['op_ok']}/{x['ops']} | {x['flat']} | {x['one_to_one_before']} | {x['capped']}"
                        if a != "DELTA_NOSPLIT" else "-- | -- | -- | -- | --")
                     + f" | {', '.join(str(v) for v in x['acc_end'])} |")
        for k, c in res["cmp"].items():
            x, y = k.split(" vs ")
            L.append(f"\n- {k}: {c['x']}/{c['n']} vs {c['y']}/{c['n']}; {x} only {c['b']}, {y} only {c['c']}; p({x}>{y}) "
                     f"{c['p']:.3g}, p({y}>{x}) {c['p_rev']:.3g}")
        rows = ["# S90 phase 2 per run (EXPLORATORY, not a result)\n", "Each firing: update (probe accuracy) c* -> c0, labelled ok "
                "or not, the key-position and all-position pairs and their labels, and the stream -> channel map before it.\n",
                "| arm | seed | outcome | transition | firings |", "|---|---|---|---|---|"]
        for a in m.P2:
            for s, r in sorted(got.get(a, {}).items()):
                f = [c for c in r.get("checks") or [] if c.get("fired")]
                fs = "<br>".join(f"{c['step']} ({c['acc']:.2f}) {c['op_cs']}->{c['op_c0']} {'ok' if c['op_ok'] else 'NOT ok'}; key "
                                 f"{c['key_cs']}->{c['key_c0']} {c['key_ok']}, all {c['all_cs']}->{c['all_c0']} {c['all_ok']}; map "
                                 f"{c['map_before']}" for c in f)
                rows.append(f"| {a} | {s} | {r.get('outcome')} | {r.get('transition')} | {fs or '--'} |")
        with open(os.path.join(OUT, "s90_runs.md"), "w") as fh:
            fh.write("\n".join(rows) + "\n")
    L.append("\n**Readings (fixed in explore_h_s90's docstring before any run):**\n")
    for k, v in res["readings"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\n*Caveat, with every reading:* {res['caveat']}\n")
    L.append(f"*The rules' {res['oc']}*")
    return m


def write(n, body, notes):
    p = os.path.join(OUT, f"report_{n}.md")
    with open(p, "w") as f:
        f.write("\n".join(body) + "\n")
        if notes:
            f.write("\n" + notes.rstrip() + "\n")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()
    notes = open(args.notes).read() if args.notes else ""
    L = [f"# Session H — report {args.batch} (EXPLORATORY, not a result)\n"]
    if args.batch == 1:
        res, got = s58_section(L)
        import explore_h_collapse as s58
        nr, h, cpus, shas = runtime(s58)
        c = checks_line(s58.NAME)
        L.append(f"\n**Runtime:** {nr} runs, {h:.1f} CPU-hours (1 thread each, 4 workers) on {', '.join(cpus)}; code SHA {', '.join(shas)}.")
        L.append(f"**CHECKs:** all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min); see explore_out/H/s58.log.")
    elif args.batch == 9:
        for m in s87_section(L):
            nr, h, cpus, shas = runtime(m)
            c = checks_line(m.NAME)
            L.append(f"\n**{m.NAME}:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}; CHECKs "
                     + (f"all passed ({c['time']}, git {c['git']})" if c else "NOT RECORDED"))
    elif args.batch == 12:
        m = s90_section(L)
        nr, h, cpus, shas = runtime(m)
        c = checks_line(m.NAME)
        L.append(f"\n**Runtime:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}. **CHECKs:** "
                 + (f"all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min)" if c else "NOT RECORDED"))
    elif args.batch == 11:
        mods = [s89c_section(L)]
        L.append("")
        mods.append(s89b_section(L))
        for m in mods:
            nr, h, cpus, shas = runtime(m)
            c = checks_line(m.NAME)
            L.append(f"\n**{m.NAME}:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}; CHECKs "
                     + (f"all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min)" if c else "NOT RECORDED"))
    elif args.batch == 10:
        m = s89_section(L)
        nr, h, cpus, shas = runtime(m)
        c = checks_line(m.NAME)
        L.append(f"\n**Runtime:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}. **CHECKs:** "
                 + (f"all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min)" if c else "NOT RECORDED"))
    elif args.batch in (4, 5, 6, 7, 8):
        m = {4: s78_section, 5: s79_section, 6: s84_section, 7: s85_section, 8: s86_section}[args.batch](L)
        nr, h, cpus, shas = runtime(m)
        c = checks_line(m.NAME)
        L.append(f"\n**Runtime:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}. **CHECKs:** "
                 + (f"all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min)" if c else "NOT RECORDED"))
    elif args.batch == 3:
        mods = [s61_section(L)]
        L.append("")
        mods.append(s63_section(L))
        L.append("")
        mods.append(s62_section(L))
        for m in mods:
            nr, h, cpus, shas = runtime(m)
            c = checks_line(m.NAME)
            L.append(f"\n**{m.NAME}:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}; CHECKs "
                     + (f"all passed ({c['time']}, git {c['git']})" if c else "NOT RECORDED"))
    else:
        import explore_h_eight as s58b
        import explore_h_stability as s60
        s58b_section(L)
        L.append("")
        s60_section(L)
        for m in (s58b, s60):
            nr, h, cpus, shas = runtime(m)
            c = checks_line(m.NAME)
            L.append(f"\n**{m.NAME}:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}; CHECKs "
                     + (f"all passed ({c['time']}, git {c['git']})" if c else "NOT RECORDED"))
    write(args.batch, L, notes)


if __name__ == "__main__":
    main()
