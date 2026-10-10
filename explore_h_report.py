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


def write(n, body, notes):
    p = os.path.join(OUT, f"report_{n}.md")
    with open(p, "w") as f:
        f.write("\n".join(body) + "\n")
        if notes:
            f.write("\n" + notes.rstrip() + "\n")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int, choices=(1, 2, 3, 4, 5))
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
    elif args.batch in (4, 5):
        m = s78_section(L) if args.batch == 4 else s79_section(L)
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
