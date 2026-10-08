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


def write(n, body, notes):
    p = os.path.join(OUT, f"report_{n}.md")
    with open(p, "w") as f:
        f.write("\n".join(body) + "\n")
        if notes:
            f.write("\n" + notes.rstrip() + "\n")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int, choices=(1, 2))
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
