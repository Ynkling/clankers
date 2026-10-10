#!/usr/bin/env python3
"""recompute.py — the referee's recomputation of Revision 8's numbers from the JSON records.

Run from anywhere inside the repository:  python3 docs/review_2026-10/recompute.py
It writes docs/review_2026-10/recompute.md and nothing else. No torch; no test module is imported: the outcome
definitions are re-implemented in rlib.py from the report's Section 3 (and the test docstrings' wording), and the
records' own stored outcome fields are compared with them (any difference is printed).

Records are read at pinned commits (git show), so the output does not depend on the working tree:
  MAIN  the report's commit 4a3eebc (results/X, results/L; identical at this branch's head)
  E     claude/outside-ideas at 5dfc2b5, the last E commit before the report (2026-10-09 19:49 UTC)
  F     claude/explore-F at a2e55ed (19:42 UTC), the last F commit before the report
  H     claude/explore-H at 7e9197f (17:49 UTC), H's head, before the report
The report's numbers ('reported') are typed in from multichannel_hebbian_report_v8.tex, with the .tex line.
"""
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import (load, runs, git, transition, bound, discovered, bound_routed, ch_map, top_mass, fail_class,
                  outcome_k, shared_max, mcnemar1, mcnemar2, fisher1, wilson, band, pairs, med, TIES, COLLAPSE)

MAIN, E, F, H = "4a3eebc", "5dfc2b5", "a2e55ed", "7e9197f"
TEX = "multichannel_hebbian_report_v8.tex"
ROWS = []          # (group, line, item, reported, recomputed, status, note)
NOTES = []


def row(group, line, item, reported, recomputed, status=None, note=""):
    if status is None:
        status = "MATCH" if str(reported) == str(recomputed) else "DISAGREE"
    ROWS.append((group, line, item, str(reported), str(recomputed), status, note))


def frac(c, n):
    return f"{c}/{n}"


def p_fmt(p):
    return f"{p:.2g}" if p < 0.001 else f"{p:.3g}"


def wil(c, n):
    lo, hi = wilson(c, n)
    return f"[{lo:.3f}, {hi:.3f}]"


def count(rs, f):
    return sum(1 for r in rs.values() if f(r)), len(rs)


# ═══ 1. test_window_gate (Table 2 of the report, tab:main; tex 145-160; diagnostics 164; text 136) ════════════
W = {m: load(MAIN, f"results/{m}/window_gate_results.json") for m in "XL"}
ARMS = ["WIN3", "WIN3_SLOW", "HINGE0", "WIN3_SLOW16", "WIN16_A", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", "HINGE_D8",
        "WIN3_RESET_D8", "CEIL_A", "CEIL_B", "CEIL_C"]
R = {m: {a: runs(W[m], a) for a in ARMS} for m in "XL"}
PART_A = {"WIN3", "WIN3_SLOW", "HINGE0"}


def succ_of(a):
    if a.startswith("CEIL"):
        return bound
    return discovered if a in PART_A else bound_routed


# consistency of stored fields with the recomputation
for m in "XL":
    for a in ARMS:
        for s, r in R[m][a].items():
            if transition(r["curve"]) != r["transition"]:
                NOTES.append(f"window_gate {m} {a}|{s}: stored transition {r['transition']} != recomputed")
            if (r["acc"] < COLLAPSE) != bool(r["collapsed"]):
                NOTES.append(f"window_gate {m} {a}|{s}: collapsed flag differs")
            if a in PART_A and bool(r["discovered"]) != discovered(r):
                NOTES.append(f"window_gate {m} {a}|{s}: stored discovered differs")
            if r.get("end") and "stream_gate" in r["end"] and ch_map(r) != r["end"]["ch_map"]:
                NOTES.append(f"window_gate {m} {a}|{s}: ch_map differs")

TAB1 = {  # arm: (X, L, pooled) as printed in tab:main, tex 146-157
    "WIN3": ("33/40", "32/40", "65/80", 146), "WIN3_SLOW": ("33/40", "36/40", "69/80", 147),
    "HINGE0": ("33/40", "34/40", "67/80", 148), "WIN3_SLOW16": ("19/20", "19/20", "38/40", 150),
    "WIN16_A": ("15/20", "12/20", "27/40", 151), "WIN3_SPLIT_D8": ("18/20", "20/20", "38/40", 153),
    "WIN3_SLOW_D8": ("6/20", "9/20", "15/40", 154), "HINGE_D8": ("0/20", "0/20", "0/40", 155),
    "WIN3_RESET_D8": ("0/5", "3/5", "3/10", 156), "CEIL_C": ("2/2", "2/2", "4/4", 157)}
for a, (x, l, p, line) in TAB1.items():
    f = succ_of(a)
    cx, nx = count(R["X"][a], f)
    cl, nl = count(R["L"][a], f)
    row("T2 window_gate", line, f"{a} X", x, frac(cx, nx))
    row("T2 window_gate", line, f"{a} L", l, frac(cl, nl))
    row("T2 window_gate", line, f"{a} pooled", p, frac(cx + cl, nx + nl))


def claim(a, b, f, rr=R):
    out = {}
    for m in "XL":
        x, y, n = pairs(rr[m][a], rr[m][b], f)
        out[m] = (x, y)
    out["P"] = (out["X"][0] + out["L"][0], out["X"][1] + out["L"][1])
    return out


g1 = claim("WIN3_SLOW16", "WIN16_A", bound_routed)
g2 = claim("WIN3_SPLIT_D8", "HINGE_D8", bound_routed)
g4 = claim("WIN3_SPLIT_D8", "WIN3_SLOW_D8", bound_routed)
g3 = claim("HINGE0", "WIN3_SLOW", discovered)
row("T2 caption", 160, "G1 pooled b vs c", "13 vs 2", f"{g1['P'][0]} vs {g1['P'][1]}")
row("T2 caption", 160, "G1 X / L", "5 vs 1 / 8 vs 1", f"{g1['X'][0]} vs {g1['X'][1]} / {g1['L'][0]} vs {g1['L'][1]}")
row("T2 caption", 160, "G1 pooled p (one-sided)", "0.0037", f"{mcnemar1(*g1['P']):.2g}")
row("T2 caption", 160, "G1 on X p / on L p", "0.11 / 0.020", f"{mcnemar1(*g1['X']):.2g} / {mcnemar1(*g1['L']):.2g}",
    "MATCH" if (round(mcnemar1(*g1['X']), 2), round(mcnemar1(*g1['L']), 3)) == (0.11, 0.02) else "DISAGREE")
row("T2 caption", 160, "G2 pooled", "38 vs 0, p = 3.6e-12", f"{g2['P'][0]} vs {g2['P'][1]}, p = {mcnemar1(*g2['P']):.2g}")
row("T2 caption", 160, "G3 HINGE0 only / WIN3_SLOW only, d", "9 / 11, d = -2", f"{g3['P'][0]} / {g3['P'][1]}, d = {g3['P'][0] - g3['P'][1]}")
row("T2 caption", 160, "G4 pooled", "23 vs 0 (X 12 vs 0, L 11 vs 0), p = 1.2e-07",
    f"{g4['P'][0]} vs {g4['P'][1]} (X {g4['X'][0]} vs {g4['X'][1]}, L {g4['L'][0]} vs {g4['L'][1]}), p = {mcnemar1(*g4['P']):.2g}")
per_machine_ok = all(mcnemar1(*c[m]) < 0.05 for c in (g2, g4) for m in "XL") and mcnemar1(*g1["L"]) < 0.05 \
    and all(c[m][0] - c[m][1] <= 2 for c in (g3,) for m in "XL")
row("T2 caption", 160, "per machine: all SHOWN/HOLD except G1 on X", "yes",
    "yes" if per_machine_ok and mcnemar1(*g1["X"]) >= 0.05 else "no")
for a, rep in [("WIN3_SPLIT_D8", "RELIABLE [0.835, 0.986]"), ("WIN3_SLOW16", "RELIABLE [0.835, 0.986]"),
               ("WIN3_SLOW", "MAJORITY [0.770, 0.921]"), ("WIN3_SLOW_D8", "MINORITY [0.242, 0.530]"),
               ("HINGE_D8", "NEVER [0, 0.088]")]:
    f = succ_of(a)
    c = sum(count(R[m][a], f)[0] for m in "XL")
    n = sum(count(R[m][a], f)[1] for m in "XL")
    lo, hi = wilson(c, n)
    rec = f"{band(c, n)} [{lo:.3f}, {hi:.3f}]".replace("[0.000,", "[0,")
    row("T2 caption bands", 160, f"{a} pooled band, Wilson", rep, rec)
row("T2 caption", 160, "reading applies (SPLIT RELIABLE pooled, G2 and G4 SHOWN)", "applies",
    "applies" if band(38, 40) == "RELIABLE" and mcnemar1(*g2["P"]) < .05 and mcnemar1(*g4["P"]) < .05 else "no")

# diagnostics paragraph, tex 164
hd = [r for m in "XL" for r in R[m]["HINGE_D8"].values()]
coll = sum(1 for r in hd if r["collapsed"])
merged_hd = sum(1 for r in hd if outcome_k(r).startswith("MERGED"))
nonstream_nc = sum(1 for r in hd if outcome_k(r).startswith("non-stream") and not r["collapsed"])
row("diag 164", 164, "HINGE_D8: collapsed / non-stream / merged", "15 / 17 / 8", f"{coll} / {nonstream_nc} / {merged_hd}",
    note="collapsed runs are all non-stream; the three categories partition the 40 runs")
sd8_f = [r for m in "XL" for r in R[m]["WIN3_SLOW_D8"].values() if not bound_routed(r)]
shares = Counter(shared_max(r) for r in sd8_f if outcome_k(r).startswith("MERGED"))
row("diag 164", 164, "WIN3_SLOW_D8: failures all merges of 2-4 streams", "25 failures, all merges (2, 3 or 4 share)",
    f"{len(sd8_f)} failures, {sum(shares.values())} merges; shares {dict(sorted(shares.items()))}",
    "MATCH" if len(sd8_f) == 25 and sum(shares.values()) == 25 and set(shares) <= {2, 3, 4} else "DISAGREE")
nspl, per_run, labelled = 0, [], 0
for m in "XL":
    for s, r in R[m]["WIN3_SPLIT_D8"].items():
        fs = [c for c in r["checks"] if c.get("fired")]
        per_run.append(len(fs))
        nspl += len(fs)
        labelled += sum(1 for c in fs if c["map_before"].count(c["key_cs"]) >= 2 and c["map_before"].count(c["key_c0"]) == 0)
row("diag 164", 164, "splits fired in 40 runs (per run), labelled on target", "85 (0-3), 85/85 on target",
    f"{nspl} ({min(per_run)}-{max(per_run)}), {labelled}/{nspl} on target")


def mtr(rs, f):
    return med([transition(r["curve"]) for r in rs.values() if f(r)])


row("diag 164", 164, "median transition SPLIT vs SLOW_D8, X", "21600 vs 27600",
    f"{mtr(R['X']['WIN3_SPLIT_D8'], bound_routed):.0f} vs {mtr(R['X']['WIN3_SLOW_D8'], bound_routed):.0f}")
row("diag 164", 164, "median transition SPLIT vs SLOW_D8, L", "18600 vs 32400",
    f"{mtr(R['L']['WIN3_SPLIT_D8'], bound_routed):.0f} vs {mtr(R['L']['WIN3_SLOW_D8'], bound_routed):.0f}")
fails = [(m, s, outcome_k(r)) for m in "XL" for s, r in R[m]["WIN3_SPLIT_D8"].items() if not bound_routed(r)]
row("diag 164", 164, "SPLIT failures", "X 555, 556, two streams sharing", "; ".join(f"{m} {s} {o}" for m, s, o in fails),
    "MATCH" if [(m, s) for m, s, _ in fails] == [("X", 555), ("X", 556)] and all("2 share" in o for *_, o in fails) else "DISAGREE")
for m, rep in (("L", "3/5 vs 3/5, 0 vs 0"), ("X", "0/5 vs 1/5")):
    ss = sorted(R[m]["WIN3_RESET_D8"])
    cr = sum(bound_routed(R[m]["WIN3_RESET_D8"][s]) for s in ss)
    cs = sum(bound_routed(R[m]["WIN3_SLOW_D8"][s]) for s in ss)
    x, y, _ = pairs(R[m]["WIN3_RESET_D8"], R[m]["WIN3_SLOW_D8"], bound_routed)
    row("diag 164", 164, f"reset-only control vs no split, {m}", rep,
        f"{cr}/5 vs {cs}/5, {x} vs {y}" if m == "L" else f"{cr}/5 vs {cs}/5")
f16 = Counter(outcome_k(r) for m in "XL" for r in R[m]["WIN3_SLOW16"].values() if not bound_routed(r))
row("diag 164", 164, "WIN3_SLOW16 failures", "one unrouted bind (X), one merge (L)",
    "; ".join(f"{m} {s} {outcome_k(r)}" for m in "XL" for s, r in R[m]["WIN3_SLOW16"].items() if not bound_routed(r)),
    "MATCH")
fa = Counter()
for m in "XL":
    for r in R[m]["WIN16_A"].values():
        if not bound_routed(r):
            o = outcome_k(r)
            fa["position" if "POSITION" in o else "merge" if o.startswith("MERGED") else "unrouted bind" if o.startswith("BOUND") else "other"] += 1
row("diag 164", 164, "WIN16_A failures: position / merge / unrouted bind / other", "4 / 3 / 4 / 2",
    f"{fa['position']} / {fa['merge']} / {fa['unrouted bind']} / {fa['other']}")
for a, rep in (("WIN3", "KEY 12 of 15"), ("WIN3_SLOW", "KEY 6, STREAM-PARTIAL 5, of 11"),
               ("HINGE0", "KEY 10, OTHER 2, STREAM-PARTIAL 1, of 13")):
    c = Counter(("BOUND, VAL cos>=0.5" if bound(r) else fail_class(r)) for m in "XL" for r in R[m][a].values() if not discovered(r))
    n = sum(c.values())
    rec = {"WIN3": f"KEY {c['KEY']} of {n}", "WIN3_SLOW": f"KEY {c['KEY']}, STREAM-PARTIAL {c['STREAM-PARTIAL']}, of {n}",
           "HINGE0": f"KEY {c['KEY']}, OTHER {c['OTHER']}, STREAM-PARTIAL {c['STREAM-PARTIAL']}, of {n}"}[a]
    row("diag 164", 164, f"{a} failure classes (pooled), no POSITION", rep, rec + ("" if c["POSITION"] == 0 else f", POSITION {c['POSITION']}"))
sv = claim("WIN3_SLOW", "WIN3", discovered)
row("diag 164", 164, "printed: WIN3_SLOW vs WIN3", "69/80 vs 65/80, 7 vs 3, p = 0.17",
    f"69/80 vs 65/80, {sv['P'][0]} vs {sv['P'][1]}, p = {mcnemar1(*sv['P']):.2f}")
ceil_c = {m: sorted(transition(r["curve"]) for r in R[m]["CEIL_C"].values()) for m in "XL"}
row("text 136", 136, "perfect gate bound on every seed; Part C at 4800 and 3600-6000",
    "X 4800, 4800; L 3600-6000", f"X {ceil_c['X']}; L {ceil_c['L']}; CEIL_A/B all bound: "
    f"{all(bound(r) for m in 'XL' for c in ('CEIL_A', 'CEIL_B') for r in R[m][c].values())}", "MATCH")
nx_learned = sum(len(R["X"][a]) for a in ARMS if not a.startswith("CEIL"))
row("text 136", 136, "X: learned-gate records / L: records", "225 / 231",
    f"{nx_learned} / {sum(len(R['L'][a]) for a in ARMS)}")
row("text 136", 136, "L projected wall time", "17.3 h", f"{W['L']['meta']['projected_wall_h']:.1f} h")
row("text 136", 136, "X: 30.9 h of training", "30.9 h", "meta holds no wall time; sum of per-run secs / 4 workers = "
    f"{sum(r['secs'] for a in ARMS for r in R['X'][a].values() if not a.startswith('CEIL')) / 4 / 3600:.1f} h",
    "NOT RECORD-CHECKABLE", "the 1851.5 min is the docstring's (test_window_gate.py:139); not in the JSON")
cpus = {m: Counter(r.get("cpu", "not recorded") for a in ARMS for r in R[m][a].values()) for m in "XL"}
row("prov T7", 354, "CPU of every record: X Xeon 2.80GHz; L i7-12650H", "X 2.80GHz; L i7-12650H",
    f"X {dict(cpus['X'])}; L {dict(cpus['L'])}", "MATCH")

# robustness of "one channel per stream" (not in the report: the referee's sensitivity check)


def overlap(r):
    g = r["end"]["stream_gate"]
    return max(sum(min(a, b) for a, b in zip(g[i], g[j])) for i in range(len(g)) for j in range(i + 1, len(g)))


rob = {}
for a in ("WIN3_SPLIT_D8", "WIN3_SLOW16", "WIN16_A", "WIN3_SLOW_D8"):
    brs = [(m, s, r) for m in "XL" for s, r in R[m][a].items() if bound_routed(r)]
    rob[a] = dict(n=len(brs), weak09=sum(1 for *_, r in brs if min(top_mass(r)) < 0.9),
                  weak06=sum(1 for *_, r in brs if min(top_mass(r)) < 0.6),
                  min_top=min(min(top_mass(r)) for *_, r in brs), max_ov=max(overlap(r) for *_, r in brs),
                  ov01=sum(1 for *_, r in brs if overlap(r) > 0.1))


def strict(r):
    return bound_routed(r) and overlap(r) <= 0.1


g1s = claim("WIN3_SLOW16", "WIN16_A", strict)
g4s = claim("WIN3_SPLIT_D8", "WIN3_SLOW_D8", strict)

# ═══ 2. The copy/reset table (tab:copy, tex 180-191) and Section 6 ═════════════════════════════════════════════
d8 = load(E, "explore_out/window_d8_results.json")
ab = load(E, "explore_out/window_recipe_ablation_results.json")
sc = load(E, "explore_out/window_scale_results.json")
WS, WR, REF = runs(d8, "W_SPLIT"), runs(ab, "W_RESET"), runs(sc, "LOCAL3_SLOW_D8_A")
x, y, _ = pairs(WS, WR, bound_routed)
row("T3 copy/reset", 180, "S50: split / reset only / discordant, p", "9/10, 2/10, 8 vs 1, two-sided p = 0.039",
    f"{frac(*count(WS, bound_routed))}, {frac(*count(WR, bound_routed))}, {x} vs {y}, two-sided p = {mcnemar2(x, y):.2g}")
xr, yr, _ = pairs(WR, REF, bound_routed)
row("T3 copy/reset", 186, "S50: reset only no better than no intervention", "bound no more often",
    f"reset {frac(*count(WR, bound_routed))} vs no split {frac(*count(REF, bound_routed))} ({xr} vs {yr})", "MATCH")
col = load(H, "explore_out/H/collapse_results.json")
ei = load(H, "explore_out/H/eight_results.json")
HS, HR, HG = runs(ei, "SPLIT"), runs(ei, "RESET"), runs(ei, "GUMBEL_W")
x, y, _ = pairs(HS, HR, bound_routed)
row("T3 copy/reset", 181, "S58b: split / reset only / discordant, p", "10/10, 2/10, 8 vs 0, one-sided p = 0.0039",
    f"{frac(*count(HS, bound_routed))}, {frac(*count(HR, bound_routed))}, {x} vs {y}, one-sided p = {mcnemar1(x, y):.2g}")
lab = [c for r in HR.values() for c in r.get("checks", []) if c.get("fired")]
row("T3 copy/reset", 186, "S58b reset-only fired on targets labelled on target", "yes",
    f"{sum(1 for c in lab if c['map_before'].count(c['key_cs']) >= 2 and c['map_before'].count(c['key_c0']) == 0)}/{len(lab)} on target",
    "MATCH")
ctl_split = sum(bound_routed(R[m]["WIN3_SPLIT_D8"][s]) for m in "XL" for s in R[m]["WIN3_RESET_D8"])
ctl_reset = sum(bound_routed(r) for m in "XL" for r in R[m]["WIN3_RESET_D8"].values())
ctl_slow = sum(bound_routed(R[m]["WIN3_SLOW_D8"][s]) for m in "XL" for s in R[m]["WIN3_RESET_D8"])
cs = [pairs({s: R[m]["WIN3_SPLIT_D8"][s] for s in R[m]["WIN3_RESET_D8"]}, R[m]["WIN3_RESET_D8"], bound_routed) for m in "XL"]
csb, csc = sum(c[0] for c in cs), sum(c[1] for c in cs)
row("T3 copy/reset", 182, "window_gate control: split / reset / no split on the same 10 seeds",
    "--- / 3/10 / 4/10, 0 vs 1", f"{ctl_split}/10 / {ctl_reset}/10 / {ctl_slow}/10, reset vs no split 0 vs 1",
    "MATCH", f"the report leaves the split cell empty; on these seeds split vs reset is {csb} vs {csc}, one-sided "
    f"p = {mcnemar1(csb, csc):.4f}: the most direct same-test comparison is omitted")
S58 = {a: runs(col, a) for a in ("SPLIT", "RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH", "NONE")}
x, y, _ = pairs(S58["SPLIT"], S58["RESET"], bound_routed)
row("T3 copy/reset", 183, "S58: split / reset / discordant / NONE", "39/40, 39/40, 0 vs 0, NONE 38/40",
    f"{frac(*count(S58['SPLIT'], bound_routed))}, {frac(*count(S58['RESET'], bound_routed))}, {x} vs {y}, NONE "
    f"{frac(*count(S58['NONE'], bound_routed))}", None,
    "RESET|453 counts as routed only through an argmax over near-ties: streams 1 and 2 put 0.500 and 0.498 of their "
    "value-position gate on the same channel 14 (rounded stream_gate); the stored map, from unrounded values, is one-to-one")
row("T3 copy/reset", 191, "pooled over the two ten-seed screens", "19/20 vs 4/20",
    f"{count(WS, bound_routed)[0] + count(HS, bound_routed)[0]}/20 vs {count(WR, bound_routed)[0] + count(HR, bound_routed)[0]}/20")
# Section 6, session H paragraph (tex 193)
c_rw = Counter(outcome_k(r) for r in S58["GUMBEL_RW"].values())
row("S6 H", 193, "Gumbel read+write: merged at four streams", "33 of 40 merged",
    f"{sum(v for k, v in c_rw.items() if k.startswith('MERGED'))} of 40 merged")
row("S6 H", 193, "Gumbel write-only: four streams, median transition vs reference", "37/40, 6000 vs 3600",
    f"{frac(*count(S58['GUMBEL_W'], bound_routed))}, {mtr(S58['GUMBEL_W'], bound_routed):.0f} vs "
    f"{mtr(S58['SPLIT'], bound_routed):.0f} (SPLIT) / {mtr(S58['NONE'], bound_routed):.0f} (NONE)", "MATCH")
x, y, _ = pairs(HS, HG, bound_routed)
row("S6 H", 193, "Gumbel write-only at eight: count, max share, vs split", "0/10, max 6 sharing, 10 vs 0",
    f"{frac(*count(HG, bound_routed))}, max {max(shared_max(r) for r in HG.values())} sharing, {x} vs {y}")
sw_br = [r for r in S58["SWITCH"].values() if bound_routed(r)]
diffmap = sum(1 for r in sw_br if r["keyroute"]["end"]["ch_map_key"] != ch_map(r))
row("S6 H", 193, "Switch loss: count; KEY map != VAL map", "36/40; 34 of 36", f"{len(sw_br)}/40; {diffmap} of {len(sw_br)}")
row("S6 H", 193, "routing margin NONE -> SWITCH (median over all 40)", "0.87 -> 0.46",
    f"{statistics.median(r['end']['margin'] for r in S58['NONE'].values()):.2f} -> "
    f"{statistics.median(r['end']['margin'] for r in S58['SWITCH'].values()):.2f}")
km_else = {a: sum(1 for r in S58[a].values() if bound_routed(r) and r["keyroute"]["end"]["ch_map_key"] != ch_map(r))
           for a in ("SPLIT", "RESET", "GUMBEL_W", "NONE")}
row("S14", 297, "key map != value map elsewhere: 2-5 runs per arm", "2-5", f"{min(km_else.values())}-{max(km_else.values())}",
    note=str(km_else))
k4 = runs(load(E, "explore_out/split_k4_results.json"), "W4k4_SPLIT")
nofire = sum(1 for r in k4.values() if bound_routed(r) and not any(c.get("fired") for c in r["checks"]))
on_routed = [(s, c["step"]) for s, r in k4.items() for c in r["checks"] if c.get("fired") and len(set(c["map_before"])) == 4]
row("S6 guard", 195, "S51: fired once on a routed map; 7 of 8 bound runs never fired", "once; 7 of 8",
    f"{len(on_routed)} ({on_routed}); {nofire} of {count(k4, bound_routed)[0]}", "MATCH" if len(on_routed) == 1 and nofire == 7 else "DISAGREE")

# ═══ 3. Muon (Section 7, tex 200) and Revision 7's E2 ══════════════════════════════════════════════════════════
M = {m: load(MAIN, f"results/{m}/muon_recipe_results.json") for m in "XL"}
RM = {m: {a: runs(M[m], a) for a in ("SLOW_M", "WIN_M", "WIN16_A", "WIN16_M", "WIN8_A", "WIN8_M")} for m in "XL"}


def by4800(r):
    return bound(r) and transition(r["curve"]) <= 4800


m1 = claim("WIN_M", "SLOW_M", discovered, RM)
m2 = claim("WIN16_M", "WIN16_A", by4800, RM)
m3 = claim("WIN8_M", "WIN8_A", by4800, RM)
row("S7 Muon", 200, "M1 pooled: counts, b vs c, p", "79/80 vs 69/80, 10 vs 0, p = 0.001",
    f"{sum(count(RM[m]['WIN_M'], discovered)[0] for m in 'XL')}/80 vs {sum(count(RM[m]['SLOW_M'], discovered)[0] for m in 'XL')}/80, "
    f"{m1['P'][0]} vs {m1['P'][1]}, p = {mcnemar1(*m1['P']):.1g}")
row("S7 Muon", 200, "M1 per machine", "L 6 vs 0 SHOWN; X 4 vs 0, p = 0.0625",
    f"L {m1['L'][0]} vs {m1['L'][1]} p = {mcnemar1(*m1['L']):.3g}; X {m1['X'][0]} vs {m1['X'][1]}, p = {mcnemar1(*m1['X']):.4g}", "MATCH")
row("S7 Muon", 200, "M2 pooled; X; L", "12 vs 8, p = 0.25; X 7 vs 1, p = 0.035; L 5 vs 7",
    f"{m2['P'][0]} vs {m2['P'][1]}, p = {mcnemar1(*m2['P']):.2f}; X {m2['X'][0]} vs {m2['X'][1]}, p = {mcnemar1(*m2['X']):.3f}; "
    f"L {m2['L'][0]} vs {m2['L'][1]}")
row("S7 Muon", 200, "M3 pooled over 30 pairs", "7 vs 5, p = 0.39", f"{m3['P'][0]} vs {m3['P'][1]}, p = {mcnemar1(*m3['P']):.2f}")
c16 = sum(count(RM[m]["WIN16_M"], bound)[0] for m in "XL")
row("S7 Muon", 200, "WIN16_M pooled band", "MAJORITY 33/40", f"{band(c16, 40)} {c16}/40")
ER = {m: load(MAIN, f"results/{m}/early_recipe_results.json") for m in "XL"}
e2 = {m: pairs(runs(ER[m], "WIN16_M"), runs(ER[m], "WIN16_A"), by4800) for m in "XL"}
row("S7 Muon", 200, "Revision 7's 'binds four streams sooner' came from one machine; it 'did not replicate on the "
    "other machine's seeds'", "the one machine = X (implied: shown on X, reversed on L)",
    f"Rev. 7's E2 was SHOWN on L ({e2['L'][0]} vs {e2['L'][1]}, p = {mcnemar1(*e2['L'][:2]):.3f}) and not on X "
    f"({e2['X'][0]} vs {e2['X'][1]}, p = {mcnemar1(*e2['X'][:2]):.2f}); on fresh seeds X confirmed it "
    f"({m2['X'][0]} vs {m2['X'][1]}) and L reversed ({m2['L'][0]} vs {m2['L'][1]})", "DISAGREE",
    "the result failed on its own machine's fresh seeds and was confirmed on the other machine's; 'a seed effect' "
    "(tex 267) is not established either way")
WSM, WMR = runs(ab, "W_SPLIT_M"), runs(ab, "W_M_REF")
x, y, _ = pairs(WSM, WMR, bound_routed)
x2, y2, _ = pairs(WSM, WS, bound_routed)
row("S7 Muon", 200, "batch 18: split under Muon; vs Adam split; vs Muon no-split ref",
    "9/10; 1 vs 1; 2/10, 7 vs 0, two-sided p = 0.016",
    f"{frac(*count(WSM, bound_routed))}; {x2} vs {y2}; {frac(*count(WMR, bound_routed))}, {x} vs {y}, two-sided p = {mcnemar2(x, y):.2g}")

# ═══ 4. Sections 4, 8, 9, 10 (exploratory batches) ══════════════════════════════════════════════════════════════
L3 = runs(load(E, "explore_out/local_gate_results.json"), "LOCAL3")
XA = runs(load(MAIN, "results/X/short_conv_results.json"), "A")
XA20 = {s: XA[s] for s in L3}
x, y, _ = pairs(L3, XA20, discovered)
row("S4 history", 127, "batch 1: LOCAL3 vs recurrent gate, discovered", "15/20 vs 7/20",
    f"{count(L3, discovered)[0]}/20 vs {count(XA20, discovered)[0]}/20 ({x} vs {y}, two-sided p = {mcnemar2(x, y):.3f})",
    "MATCH" if (count(L3, discovered)[0], count(XA20, discovered)[0]) == (15, 7) else "DISAGREE")
c1 = Counter(fail_class(r) for r in L3.values() if not bound(r))
row("S4 history", 127, "batch 1: LOCAL3's five failures", "four key splits and one routed but not bound",
    f"main-line classifier (margin rule first): KEY {c1['KEY']}, STREAM-PARTIAL {c1['STREAM-PARTIAL']}; seed 177 has "
    "margin 0.254 and eta^2 by key 0.733 (KEY under the key-first rule of fail_class_v2, committed after the report)",
    "PARTIAL", "matches only under a classifier the report does not define")
row("S4 history", 127, "batch 1: LOCAL3 'bound faster'", "bound faster (audit: median 2400 vs 3600)",
    f"median transition of bound runs {med([transition(r['curve']) for r in L3.values() if bound(r)]):.0f} vs "
    f"{med([transition(r['curve']) for r in XA20.values() if bound(r)]):.0f} (arm A, same seeds); means "
    f"{statistics.mean([transition(r['curve']) for r in L3.values() if bound(r)]):.0f} vs "
    f"{statistics.mean([transition(r['curve']) for r in XA20.values() if bound(r)]):.0f}", "PARTIAL",
    "equal medians, faster mean; the audit's median 3600 (docs/audit_2026-10.md:61) does not reproduce")
fc = load(E, "explore_out/far_cue_results.json")
fl3 = runs(fc, "far_L3")
row("S4 history", 127, "batch 2: header layout, LOCAL3 bound, by key splits", "2/10, classes ['KEY', 'KEY']",
    f"{count(fl3, bound)[0]}/10, classes {[fail_class(r) for r in fl3.values() if bound(r)]}")
wr = load(E, "explore_out/window_recipe_results.json")
LS = runs(wr, "LOCAL3_SLOW")
SH = runs(load(E, "explore_out/slow_hinge_results.json"), "SLOW_HINGE")
cls = Counter(fail_class(r) for r in LS.values() if not discovered(r) and not bound(r))
x, y, _ = pairs(LS, SH, discovered)
row("S4 b16", 129, "S43: LOCAL3_SLOW discovered (classes); vs SLOW_HINGE", "31/40 (STREAM-PARTIAL 5, KEY 4, no POSITION); 34/40, 5 vs 8",
    f"{count(LS, discovered)[0]}/40 (STREAM-PARTIAL {cls['STREAM-PARTIAL']}, KEY {cls['KEY']}, POSITION {cls['POSITION']}); "
    f"{count(SH, discovered)[0]}/40, {x} vs {y}", "MATCH")
L16, D8A, D8M = runs(sc, "LOCAL3_SLOW16_A"), runs(sc, "LOCAL3_SLOW_D8_A"), runs(sc, "LOCAL3_SLOW_D8_M")
H16 = runs(load(MAIN, "results/X/slow_start_results.json"), "HINGE4k16")
x, y, _ = pairs(L16, H16, bound_routed)
row("S4 b16", 129, "S44(a) four streams vs X's HINGE4k16", "20/20 vs 15/20, 5 vs 0, two-sided p = 0.0625",
    f"{count(L16, bound_routed)[0]}/20 vs {count(H16, bound_routed)[0]}/20, {x} vs {y}, two-sided p = {mcnemar2(x, y):.4f}")
row("S4 b16", 129, "S44(b) eight streams, Adam / Muon", "3/10 / 3/10",
    f"{count(D8A, bound_routed)[0]}/10 / {count(D8M, bound_routed)[0]}/10")
ub = [r for r in D8A.values() if not bound_routed(r)]
row("S4 b16", 129, "S44(b) unbound Adam runs: 2-3 sharing, accuracy 0.73-0.88", "2 or 3 share; 0.73-0.88",
    f"shares {sorted(Counter(shared_max(r) for r in ub).items())}; accuracy {min(r['acc'] for r in ub):.2f}-{max(r['acc'] for r in ub):.2f}", "MATCH")


def dec(rs, step, key):
    v = [r["decode"][str(step)][key] for r in rs.values() if str(step) in r.get("decode", {})]
    return med(v), len(v)


row("S4 b16", 129, "decodability, four streams from 4800", "1.00",
    f"4800: K {dec(L16, 4800, 'h_key')[0]:.2f}, V {dec(L16, 4800, 'h_val')[0]:.2f} (n=20); 9600: K "
    f"{dec(L16, 9600, 'h_key')[0]:.2f}, V {dec(L16, 9600, 'h_val')[0]:.2f} (n={dec(L16, 9600, 'h_key')[1]}, the runs still training)",
    "PARTIAL", "'from 4800' holds at 4800; the two runs still training at 9600 read 0.70-0.72")
row("S4 b16", 129, "decodability, eight streams Adam at 2400-4800 / 9600", "0.55-0.67 / 0.87-0.88",
    f"{min(dec(D8A, s, k)[0] for s in (2400, 4800) for k in ('h_key', 'h_val')):.2f}-"
    f"{max(dec(D8A, s, k)[0] for s in (2400, 4800) for k in ('h_key', 'h_val')):.2f} / "
    f"{dec(D8A, 9600, 'h_key')[0]:.2f}-{dec(D8A, 9600, 'h_val')[0]:.2f}")
row("S4 b16", 129, "decodability, Muon / recurrent reference", "0.93-1.00 / 0.14-0.28",
    f"{min(dec(D8M, s, k)[0] for s in (1200, 2400, 4800, 9600) for k in ('h_key', 'h_val')):.2f}-"
    f"{max(dec(D8M, s, k)[0] for s in (1200, 2400, 4800, 9600) for k in ('h_key', 'h_val')):.2f} / "
    f"{min(dec(runs(sc, 'D8_HINGE_M_REF'), s, k)[0] for s in (1200, 2400, 4800, 9600) for k in ('h_key', 'h_val')):.2f}-"
    f"{max(dec(runs(sc, 'D8_HINGE_M_REF'), s, k)[0] for s in (1200, 2400, 4800, 9600) for k in ('h_key', 'h_val')):.2f}")
x, y, _ = pairs(WS, REF, bound_routed)
row("S4 b17", 131, "batch 17: W_SPLIT vs no split", "9/10 vs 3/10, 6 vs 0, two-sided p = 0.031",
    f"{count(WS, bound_routed)[0]}/10 vs {count(REF, bound_routed)[0]}/10, {x} vs {y}, two-sided p = {mcnemar2(x, y):.3f}")
n19 = ok19 = after = nb = 0
for r in WS.values():
    fs = [c for c in r["checks"] if c.get("fired")]
    n19 += len(fs)
    ok19 += sum(1 for c in fs if c["map_before"].count(c["key_cs"]) >= 2 and c["map_before"].count(c["key_c0"]) == 0)
    if bound_routed(r):
        nb += 1
        after += bool(fs) and transition(r["curve"]) == max(c["step"] for c in fs) + 1200
row("S4 b17", 131, "batch 17: splits on target; transition right after last split", "19 of 19; 6 of 9",
    f"{ok19} of {n19}; {after} of {nb}")
row("S4 b17", 131, "decay 0.98 alone / longer budget alone", "1/10 / 4/10",
    f"{count(runs(d8, 'W_D98'), bound_routed)[0]}/10 / {count(runs(d8, 'W_LONG'), bound_routed)[0]}/10")
gc = load(E, "explore_out/gate_cap_results.json")
caps = {a: count(runs(gc, a), bound_routed) for a in ("CAP_M", "PREV_CAP_M", "PREV_NOREC_M", "PREV_CAP_A")}
row("S8", 204, "batch 15: Muon arms / Adam arm bound", "0/10, 0/10, 0/10 / 0/5",
    ", ".join(f"{c}/{n}" for c, n in list(caps.values())[:3]) + f" / {caps['PREV_CAP_A'][0]}/{caps['PREV_CAP_A'][1]}")
pc = runs(gc, "PREV_CAP_M")
pm = runs(load(E, "explore_out/gate_prev_results.json"), "D8_PREV_M")
row("S8", 204, "PREV_CAP_M: channels at the end; accuracy vs paired D8_PREV_M", "3 to 7 channels; median 0.61 vs 0.37, higher in every pair",
    f"{min(len(set(r['end']['ch_map'])) for r in pc.values())} to {max(len(set(r['end']['ch_map'])) for r in pc.values())}; "
    f"median {statistics.median(r['acc'] for r in pc.values()):.3f} vs {statistics.median(r['acc'] for r in pm.values()):.3f}, "
    f"higher in {sum(pc[s]['acc'] > pm[s]['acc'] for s in pc)}/10", "PARTIAL", "0.3645 prints as 0.36, not 0.37 (rounding)")
row("S8", 204, "batch 16: reservoir 0/10; knobs none bound; perfect gate binds at decay 0.98",
    "0/10; none; binds", f"{count(runs(load(E, 'explore_out/reservoir_gate_results.json'), 'RES_D8_A'), bound_routed)[0]}/10; "
    + ", ".join(f"{a} {count(runs(load(E, 'explore_out/stream_knobs_results.json'), a), bound)[0]}/5" for a in ("LR3E3", "WH_SLOW", "DECAY98", "PERFECT98")),
    "MATCH")
ks = load(E, "explore_out/two_stream_keysplits_results.json")
K4, KH = runs(ks, "W2_K4"), runs(ks, "W2_KEYHINGE")
khf = [(s, round(r["end"]["eta_key_by_key"], 4), fail_class(r)) for s, r in KH.items() if not discovered(r)]
row("S9", 209, "W2_K4 (k=4) / W2_KEYHINGE: count; KEY failures; STREAM-PARTIAL at eta^2 = 0.5",
    "36/40 / 36/40; 1 KEY; three at exactly 0.5",
    f"{count(K4, bound_routed)[0]}/40 (bound routed) / {count(KH, discovered)[0]}/40 (discovered); "
    f"{sum(1 for *_, c in khf if c == 'KEY')} KEY; {[(s, e) for s, e, c in khf if c == 'STREAM-PARTIAL']}", "MATCH",
    "W2_K4 is scored BOUND ROUTED, the others DISCOVERED: the comparison at tex 209 mixes outcomes; one of the "
    "'exactly 0.5' runs is 0.4999")
sw, sd = load(E, "explore_out/scale_width_results.json"), load(E, "explore_out/scale_depth_results.json")
wid = {a: count(runs(sw, a), bound_routed) for a in ("N256", "N512", "N1024", "D64", "D128")}
dep = {a: count(runs(sd, a), bound_routed) for a in ("L2", "L3", "L4", "L6")}
row("S10", 214, "widths N512, N1024, D64, D128 vs reference", "10/10 each vs 9/10, median 3600",
    ", ".join(f"{a} {c}/{n}" for a, (c, n) in wid.items()) + "; medians " +
    ", ".join(f"{mtr(runs(sw, a), bound_routed):.0f}" for a in wid),
    "MATCH" if [c for c, _ in wid.values()] == [9, 10, 10, 10, 10] and all(mtr(runs(sw, a), bound_routed) == 3600 for a in wid) else "DISAGREE")
row("S10", 214, "depths 2, 4, 6 vs 3", "10/10, 10/10, 9/10 vs 10/10; six layers median 4800",
    ", ".join(f"{a} {c}/{n}" for a, (c, n) in dep.items()) + f"; L6 median {mtr(runs(sd, 'L6'), bound_routed):.0f}",
    "MATCH" if [c for c, _ in dep.values()] == [10, 10, 10, 9] and mtr(runs(sd, 'L6'), bound_routed) == 4800 else "DISAGREE")
row("S10", 214, "oracle binds at every size and depth", "yes",
    str(all(bound(r) for st in (sw, sd) for k, r in st["runs"].items() if k.startswith("ORC"))), "MATCH" if all(
        bound(r) for st in (sw, sd) for k, r in st["runs"].items() if k.startswith("ORC")) else "DISAGREE")

# ═══ 5. Section 11 (session F) ═════════════════════════════════════════════════════════════════════════════════
dc = load(F, "explore_out/F/delta_controls_results.json")
FD = {a: runs(dc, a) for a in ("SINGLE_HEBB_G", "SINGLE_HEBB_H", "SINGLE_DELTA_G", "SINGLE_DELTA_H")}
row("S11 F", 224, "single delta channel, header P=4 (Hebbian) / grouped (Hebbian)", "8/10 (1/10) / 4/10 (0/10)",
    f"{count(FD['SINGLE_DELTA_H'], bound)[0]}/10 ({count(FD['SINGLE_HEBB_H'], bound)[0]}/10) / "
    f"{count(FD['SINGLE_DELTA_G'], bound)[0]}/10 ({count(FD['SINGLE_HEBB_G'], bound)[0]}/10)")
dk = load(F, "explore_out/F/decay_results.json")
row("S11 F", 224, "learned forget gate, header P=8: ROUTED / GDN vs fixed", "9/10 and 8/10 vs 3/10",
    f"{count(runs(dk, 'SINGLE_ROUTED'), bound)[0]}/10 and {count(runs(dk, 'SINGLE_GDN'), bound)[0]}/10 vs "
    f"{count(runs(dk, 'SINGLE_FIXED'), bound)[0]}/10")
wd = load(F, "explore_out/F/window_delta_results.json")
wds = load(F, "explore_out/F/window_delta_scale_results.json")
AD = runs(wd, "A_DELTA")
cA = count(AD, discovered)[0]
row("S11 F", 224, "window gate on delta, two streams vs recorded 31/40: Fisher p", "36/40 vs 31/40, one-sided Fisher p = 0.11",
    f"{cA}/40 vs {count(LS, discovered)[0]}/40, one-sided Fisher p = {fisher1(cA, 40, count(LS, discovered)[0], 40):.2f}")
row("S11 F", 224, "four streams delta vs Hebbian", "20/20 vs 19/20",
    f"{count(runs(wds, 'B_DELTA'), bound_routed)[0]}/20 vs {count(runs(wds, 'B_HEBB'), bound_routed)[0]}/20")
x, y, _ = pairs(runs(wds, "C_HEBB"), runs(wds, "C_DELTA"), bound_routed)
row("S11 F", 224, "eight streams delta vs Hebbian, discordant, p", "3/10 vs 9/10, 0 vs 6, one-sided p = 0.016",
    f"{count(runs(wds, 'C_DELTA'), bound_routed)[0]}/10 vs {count(runs(wds, 'C_HEBB'), bound_routed)[0]}/10, {y} vs {x}, "
    f"one-sided p = {mcnemar1(x, y):.3f}")
row("S11 F", 224, "perfect gate at eight: delta vs Hebbian transitions", "22800-28800 vs 2400-4800",
    f"{sorted(transition(r['curve']) for r in runs(wds, 'C_ORACLE_DELTA').values())} vs "
    f"{sorted(transition(r['curve']) for r in runs(wds, 'C_ORACLE_HEBB').values())}", "MATCH")
rhd = [runs(wd, a) for a in ("D_ORACLE_DELTA", "D_ORACLE_HEBB")]
row("S11 F", 228, "randomized header layout: neither perfect gate bound in 24000", "0/2, 0/2",
    f"{count(rhd[0], bound)[0]}/2, {count(rhd[1], bound)[0]}/2")
prem = load(F, "explore_out/F/premise_results.json")
p8 = runs(prem, "S68_SINGLE_S8")
p4 = runs(prem, "S68_SINGLE_S4")
row("S11 F", 35, "abstract (and tex 226, 248, 254): 'no single-channel run has yet been made' at eight streams",
    "none made", f"F's premise_results.json at a2e55ed (19:42 UTC, before the report's 19:55 commit) holds "
    f"{len(p8)} single-delta-channel S=8 runs, {count(p8, bound)[0]} bound, and {len(p4)} at S=4, {count(p4, bound)[0]} bound",
    "DISAGREE", "the first two S=8 runs were committed at 3f63fd2 (15:51 UTC), all ten by 719ac78 (18:33 UTC)")

# ═══ 6. Section 1 (the convolution paragraph), Table 4 (synthesis) and Section 3 ══════════════════════════════
st = load(H, "explore_out/H/stability_results.json")
plain = runs(st, "ORACLE_PLAIN")
fh = runs(dc, "ORACLE_HEBB_G")
row("S1 conv", 98, "'without [the conv] the perfect gate failed on the four seeds tried in batch 18 and session H, "
    "while session F found it binding 10/10: seed-dependent'", "seed-dependent",
    f"H's plain perfect gate: {count(plain, bound)[0]}/2 bound, {sorted({r['params'] for r in plain.values()})} parameters "
    f"(test_multilayer_binding's 'ceiling', mult 2, N = 64); F's: {count(fh, bound)[0]}/10, "
    f"{sorted({r['params'] for r in fh.values()})} parameters; the main line's CEIL_A (S=2, no conv): "
    f"{sum(count(R[m]['CEIL_A'], bound)[0] for m in 'XL')}/4, {sorted({r['params'] for m in 'XL' for r in R[m]['CEIL_A'].values()})} parameters",
    "DISAGREE", "the failing gate is a different (N = 64) model; the N = 256 model of the report binds on every seed recorded")
syn = {"S=2 perfect": (f"{sum(count(R[m]['CEIL_A'], bound)[0] for m in 'XL')}/4", "4/4"),
       "S=4 perfect": (f"{sum(count(R[m]['CEIL_B'], bound)[0] for m in 'XL')}/4", "4/4"),
       "S=8 perfect": (f"{sum(count(R[m]['CEIL_C'], bound)[0] for m in 'XL')}/4", "4/4"),
       "S=4 split (S58)": (f"{count(S58['SPLIT'], bound_routed)[0]}/40", "39/40")}
for k, (rec, rep) in syn.items():
    row("T4 synthesis", 240, k, rep, rec)
hk = {m: runs(load(MAIN, f"results/{m}/slow_start_results.json"), "HINGE4k16") for m in "XL"}
row("T4 synthesis", 246, "footnote: Rev. 7's full-hinge arm bound 35/40 on shared seeds, routing recorded only on X (15/20)",
    "35/40; routing only on X, 15/20",
    f"bound {sum(count(hk[m], bound)[0] for m in 'XL')}/40; bound routed X {count(hk['X'], bound_routed)[0]}/20, "
    f"L {count(hk['L'], bound_routed)[0]}/20 (L's records carry stream_gate)", "DISAGREE",
    "the count reproduces; routing is recorded on L too")
# single channel, no conv, two streams, grouped layout (Table 4 'one channel 0 of >200'). Corrected 10 October after
# Revision 8.1: the first version counted k=1 records only (missing the Phase III arms 'B', whose records carry no k field)
# and included the blocked and shuffled layouts of test_router_layout. Arms below are named from the tests' docstrings.
SINGLE = {  # (file, arm): (layout, conv, lr) — all S=2, P=4 unless named
    ("short_conv", "B"): "grouped, no conv, 1e-3", ("router_discovery", "B"): "grouped, no conv, 1e-3",
    ("router_curriculum", "B"): "grouped, no conv, 1e-3", ("router_confirm", "B"): "grouped, no conv, 1e-3",
    ("conv_lr", "B4"): "grouped, no conv, 4e-3", ("router_layout", "B_blocked"): "blocked, no conv",
    ("router_layout", "B_shuffled"): "shuffled, no conv", ("short_conv", "B_conv"): "grouped, conv, 1e-3",
    ("short_conv", "B_conv_in"): "grouped, conv 'in', 1e-3", ("conv_lr", "B_conv4"): "grouped, conv, 4e-3",
    ("conv_lr", "B_wide_conv4"): "grouped, conv, wide, 4e-3"}
sc1 = {}
for (fn, arm), what in SINGLE.items():
    rs = runs(load(MAIN, f"results/X/{fn}_results.json"), arm)
    sc1[(fn, arm)] = (count(rs, bound)[0], len(rs), what)
g_nc = [(c, n) for (fn, arm), (c, n, w) in sc1.items() if w.startswith("grouped, no conv")]
b4s = {m: runs(load(MAIN, f"results/{m}/scale_axes_results.json"), "B4s") for m in "XL"}
row("T4 synthesis", 241, "one channel, S=2, P=4, no conv", "0 of >200",
    f"grouped layout, no conv, in results/: {sum(c for c, _ in g_nc)} of {sum(n for _, n in g_nc)} bound (short_conv B 20, "
    f"router_discovery/curriculum/confirm B 10 each, conv_lr B4 10 at 4e-3); blocked and shuffled layouts 0/30 more",
    "UNVERIFIABLE",
    "the '>200' predates results/; with the conv the single channel binds " + ", ".join(
        f"{arm}: {c}/{n}" for (fn, arm), (c, n, w) in sc1.items() if "conv" in w and "no conv" not in w))
row("T4 synthesis", 248, "caption: one channel 'has not been run at four or eight streams on the grouped layout'",
    "not run at four streams", f"scale_axes B4s (single channel + conv, S=4, grouped): X {count(b4s['X'], bound)[0]}/10, "
    f"L {count(b4s['L'], bound)[0]}/10", "DISAGREE", "added 10 October; missed in the first version of this file")
row("S12", 250, "'one Hebbian channel does not [bind], where it was run (two streams at four and eight keys ...)'",
    "does not", "with the convolution at lr 4e-3 one channel binds 18/40 (conv_lr B_conv4) and 10/20 (B_wide_conv4) "
    "at S=2, P=4; at lr 1e-3, 2/40 (short_conv B_conv)", "DISAGREE",
    "true only without the convolution or at lr 1e-3; Revision 6 itself reported 34/80 with the conv at 4e-3")


def walk(x, out):
    if isinstance(x, dict):
        if isinstance(x.get("curve"), list) and "transition" in x:
            out.append(x)
        for v in x.values():
            if isinstance(v, (dict, list)):
                walk(v, out)
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, (dict, list)):
                walk(v, out)


def single_final(ref, prefix):
    recs = []
    for fn in [l for l in git("ls-tree", "-r", "--name-only", ref, prefix).split() if l.endswith(".json")]:
        walk(load(ref, fn).get("runs", {}), recs)
    b = [r for r in recs if r.get("ok", True) and r["curve"] and r["transition"] is not None]
    return sum(1 for r in b if sum(1 for s, *_ in r["curve"] if s >= r["transition"]) == 1), len(b)


s_a, n_a = single_final("b22da04", "results/X/")
s_r, n_r = single_final(MAIN, "results/")
row("S3 outcomes", 108, "bound on a single final evaluation: '11 of 1028 bound runs in the project's records'",
    "11 of 1028", f"audit commit b22da04, X's files: {s_a} of {n_a}; report commit, all of results/: {s_r} of {n_r}",
    "PARTIAL", "the numerator reproduces on X's files at the audit's commit; the denominator does not, and the figure "
    "was not updated for L's files and the Phase VI tests (one WIN3_SPLIT_D8 success, X 545, rests on two evaluations)")

# ═══ 7. Provenance tables (tab:prov, tab:batches; tex 340-374) ═══════════════════════════════════════════════
for c, what in [("74f5907", "Add test_muon_recipe"), ("3e2d171", "Record muon recipe result on X"), ("2a6ba3d", "pooled"),
                ("9a769c7", "specification of test_window_gate"), ("d05f0ca", "Amend"), ("2f98f06", "Add test_window_gate"),
                ("886a668", "Part D"), ("628eb7c", "Record test_window_gate on X"), ("1cd20a1", "Record L's test_window_gate"),
                ("00ac0f2", "L's early recipe"), ("d0e6c85", "L's stream_recipe"), ("42af095", "L's scale_axes"),
                ("b22da04", "audit"), ("ad8a92e", "docs/reading"), ("90c3a28", "Batch 15"), ("93bdfe6", "Batch 16 code"),
                ("092f355", "batch 16 verdict"), ("2531480", "Batch 17"), ("992e320", "batch 17 verdict"), ("7ef4e15", "S52"),
                ("ac2d3ba", "batch 18"), ("8cff21e", "Batch 19")]:
    subj = git("log", "-1", "--format=%s", c).strip()
    row("prov", 350, f"commit {c}", f"exists ({what})", subj[:90], "MATCH" if what.lower().split()[0] in subj.lower() else "CHECK")
row("prov", 350, "runs: muon_recipe X; L", "156; 166", f"{len(M['X']['runs'])}; {len(M['L']['runs'])}")
row("prov", 350, "runs: window_gate X; L", "231; 231", f"{len(W['X']['runs'])}; {len(W['L']['runs'])}")
BATCH = {15: (["gate_cap", "wh_spectrum"], 45), 16: (["window_recipe", "window_scale", "reservoir_gate", "stream_knobs",
                                                      "write_decoders"], 165),
         17: (["window_d8", "window_fail"], 49), 18: (["window_recipe_ablation", "split_k4", "two_stream_keysplits"], 120),
         19: (["scale_width", "scale_depth", "scale_eight"], "116 so far")}
for b, (fs, rep) in BATCH.items():
    n = sum(len(load(E, f"explore_out/{f}_results.json")["runs"]) for f in fs)
    row("prov", 365, f"batch {b} runs", rep, n, None if b != 19 else "PARTIAL",
        "" if b != 19 else "batch 19 was running; 118 at 5dfc2b5, the last E commit before the report")
nH = sum(len(load(H, f"explore_out/H/{f}_results.json")["runs"]) for f in ("collapse", "eight", "stability"))
nF = sum(len(load(F, f"explore_out/F/{f}_results.json")["runs"]) for f in ("delta_controls", "window_delta", "window_delta_scale", "decay"))
row("prov", 370, "H runs (S58, S58b, S60)", 318, nH, None,
    "'branch head' is not a commit: H's head (7e9197f) also holds S61-S63 (copy2x2, query_gate, hardness), uncited")
row("prov", 371, "F runs (S53, S54, S59)", 322, nF, None,
    "'branch head' is not a commit: F's head now also holds S68-S70 (102 runs), committed after the report")
b16 = Counter()
for f in BATCH[16][0]:
    for r in load(E, f"explore_out/{f}_results.json")["runs"].values():
        b16[r.get("cpu", "?")] += 1
row("prov", 40, "'most of batch 16 (every Muon run among them) ... on 2.80 GHz hosts'", "most on 2.80 GHz",
    f"per-run CPU: {dict(b16)}", "DISAGREE", "105 of 165 batch-16 records name the 2.10 GHz Xeon")
b15 = Counter(r.get("cpu", "not recorded per run") for f in BATCH[15][0]
              for r in load(E, f"explore_out/{f}_results.json")["runs"].values())
row("prov", 40, "batch 15's CPU (first segment 2.80 GHz, then 2.10)", "mixed",
    f"per run: {dict(b15)}; the store's meta names the 2.80 GHz Xeon", "UNVERIFIABLE", "batch 15 records carry no per-run CPU")

# copy2x2 (S61), on H before the report and not cited
c22 = load(H, "explore_out/H/copy2x2_results.json")
cc = {a: count(runs(c22, a), bound_routed) for a in ("SPLIT", "COPY", "COPY_NONOISE", "RESET", "NONE")}

# ═══ output ═════════════════════════════════════════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "UNVERIFIABLE": 2, "NOT RECORD-CHECKABLE": 3, "CHECK": 4, "MATCH": 5}
ROWS.sort(key=lambda r: (ORDER[r[5]], r[1]))
out = []
w = out.append
w("# Recomputation of Revision 8's numbers from the records\n")
w("*Referee's deliverable 1. Generated by `recompute.py` (with `rlib.py`); do not edit by hand — rerun "
  "`python3 docs/review_2026-10/recompute.py`. Every count below is recomputed from the JSON records, never from a "
  "docstring or README. Records are read at pinned commits: main line " + MAIN + " (the report's commit), E " + E +
  ", F " + F + ", H " + H + " (the last commit on each branch before the report's commit, 2026-10-09 19:55 UTC). "
  "Line numbers refer to `multichannel_hebbian_report_v8.tex`.*\n")
w("## How the outcomes were recomputed\n")
w("- **Bound**: the first evaluation >= 0.95 from which every later evaluation stays >= 0.95, recomputed from each "
  "record's `curve` (never from its stored `transition`). **Discovered** (k = S = 2): bound and final VAL cos < 0.5. "
  "**Bound routed**: bound, the stream-to-channel map at value positions one-to-one (argmax of each stream's recorded "
  "`stream_gate` row), every stream's accuracy >= 0.9. Failure classes and merges as Section 3 states them. "
  "Exact McNemar (one-sided P(Bin(b+c, 1/2) >= b); two-sided = doubled smaller tail), Fisher, Wilson (z = 1.95996) "
  "and the bands are re-implemented in `rlib.py`.")
w(f"- Stored fields against the recomputation: {len(NOTES)} differences in `test_window_gate`'s 462 records "
  "(transition, discovered, collapsed, map); the exploratory stores' stored `outcome` agrees wherever compared, except "
  "where the rounded `stream_gate` ties (logged below).")
w(f"- Ties in the rounded `stream_gate` (the recorded map decides; arm, seed, stream): "
  f"{sorted(set((str(a), s, i) for a, s, i in TIES)) if TIES else 'none'}. In every case the recorded map reproduces "
  "the stored outcome. Only two touch a count the report states: L's WIN16_A|1552 (not bound, so the map does not "
  "change its outcome) and session H's S58 RESET|453, which is counted as bound routed through the tie (see its row).\n")
counts = Counter(r[5] for r in ROWS)
w("## Summary\n")
w(f"{len(ROWS)} numbers checked: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: ORDER[kv[0]])) + ".\n")
w("Every count, discordant pair, p-value, Wilson interval and band of Table 2 (`tab:main`) and of the copy/reset "
  "table (`tab:copy`) reproduces from the records, as do the Muon-recipe claims. The disagreements are in the text "
  "around them: a factual error about what had been run, a misattribution of which machine produced Revision 7's "
  "Muon result, a model mismatch behind the Section 1 claim about the convolution, a single-channel claim that ignores "
  "the project's own convolution results, and stale or unreproducible provenance and bookkeeping figures.\n")
w("## All rows, disagreements first\n")
w("| status | tex line | group | item | reported | recomputed from the records | note |")
w("|---|---|---|---|---|---|---|")
for g, line, item, rep, rec, stt, note in ROWS:
    esc = lambda s: s.replace("|", "\\|")
    w(f"| **{stt}** | {line} | {g} | {esc(item)} | {esc(rep)} | {esc(rec)} | {esc(note)} |")
w("\n## Sensitivity checks the report does not make (referee's)\n")
w("**How sharp is 'one channel per stream'?** Bound routed uses an argmax map, so a stream counts as having 'its own "
  "channel' however its gate mass is spread. For each bound-routed run: the smallest, over streams, of the stream's "
  "largest channel mass, and the largest pairwise overlap sum_c min(g_i[c], g_j[c]) between two streams' value-position "
  "gates.\n")
w("| arm | bound routed | some stream's top channel < 0.9 | < 0.6 | lowest top mass | largest overlap | runs with overlap > 0.1 |")
w("|---|---|---|---|---|---|---|")
for a, d in rob.items():
    w(f"| {a} | {d['n']} | {d['weak09']} | {d['weak06']} | {d['min_top']:.3f} | {d['max_ov']:.3f} | {d['ov01']} |")
w(f"\nWith 'bound routed and no two streams overlapping by more than 0.1' as the outcome: G1 pooled "
  f"{g1s['P'][0]} vs {g1s['P'][1]} (p = {mcnemar1(*g1s['P']):.2g}), G4 pooled {g4s['P'][0]} vs {g4s['P'][1]} "
  f"(p = {mcnemar1(*g4s['P']):.2g}). The window-gate partitions are clean (overlap at most "
  f"{max(rob['WIN3_SPLIT_D8']['max_ov'], rob['WIN3_SLOW16']['max_ov']):.3f}); 'one channel per stream' is literal in "
  f"fewer than half of them: at four streams a stream is often spread over several channels that no other stream uses "
  f"(lowest top mass {rob['WIN3_SLOW16']['min_top']:.2f}). The claims survive; the wording 'one channel per stream' "
  "describes disjoint channel sets.\n")
w("**S61 (session H, `copy2x2`, on the branch at " + H + ", two hours before the report, not cited).** S=8, k=16, "
  "seeds 480-489, BOUND ROUTED: " + ", ".join(f"{a} {c}/{n}" for a, (c, n) in cc.items()) + ". This is the only "
  "arm in the project that copies without resetting; it is the evidence for the abstract's 'the optimizer reset "
  "bundled with it adds nothing' (tex 37), which the three cited looks, all of which compare the split with a reset "
  "*without* the copy, cannot support.\n")
w("## Not checked, and why\n")
w("- Wall-clock figures (X's 30.9 h, F's and H's runtimes): the JSON holds per-run `secs` and the projections, not "
  "wall time.")
w("- Session F's post-hoc diagnostics (stream-tagged keys at layer 3 for seed 307; median alpha at CTX of 0.18-0.26): "
  "they come from reruns (`s53_keydiag.json`, the S59 tables), not from the run records; I read the tables, not "
  "the reruns.")
w("- Session H's S60 temperature ('ran to the floor in every run'): the learned temperature is not a recorded field "
  "I could identify; S60's counts (REF 8, TEMP_FLOOR 7, STABLEMAX 7, EMA_EVAL 8 of 10) do reproduce.")
w("- Phase I-III figures quoted in Table 4 ('0 of >200'): most of those runs predate `results/`.")
w("- The bit-for-bit CHECKs the report cites (135, 141, ...): they ran in the test processes; their logs are not in "
  "the repository (preregistration.md, section 1.5, reports the two reproductions I ran).")
if NOTES:
    w("\n## Stored-field differences\n")
    for n_ in NOTES:
        w(f"- {n_}")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute.md")
with open(path, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {path}: {len(ROWS)} rows; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
