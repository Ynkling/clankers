#!/usr/bin/env python3
"""recompute_G.py — the referee's recomputation of Session G's follow-up reports 3 and 4 (screens S71, S72, S73)
from the JSON run records at the pinned head of claude/explore-G.

Run from anywhere inside the repository:  python3 docs/review_2026-10/sessions/recompute_G.py
It writes docs/review_2026-10/sessions/recompute_G.md and nothing else. Python 3 stdlib only; no torch; no screen
module is imported. Every record is read with `git show <commit>:<path>`, so the working tree does not matter.

Pinned refs
  G     claude/explore-G at 93422cf (report 4's commit, the branch head): explore_out/G/s7[123]_*_results.json,
        checks_s7[123]_*.json, s7[123].log, s56_register_results.json, s57_latch_results.json
  F     claude/explore-F at the last commit before 93422cf's commit time (F's S70 store rh_lite_results.json),
        for report 4's "F's references" line
  MAIN  the main line's fail_class_v2.py (origin/claude/bdh-growth-hebbian-inference-w90069), used only for the
        key-split sensitivity rows (the screens themselves use v1)

Outcome definitions are re-implemented here from the screens' docstrings (line numbers in explore_g_rhl_parent.py
at 93422cf, which the screen docstrings point to for OUTCOMES and STATISTICS); see OUTCOME_DEFS below.
The reported values are typed in from the report text with their line numbers; none is used as an input.
"""
import os
import re
import statistics
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # docs/review_2026-10
from rlib import load, runs, git, transition, wilson, band, mcnemar1, fail_class, ROOT  # noqa: E402

G = "93422cf"
BRANCH = "origin/claude/explore-G"
MAINREF = "origin/claude/bdh-growth-hebbian-inference-w90069"
R3, R4 = "explore_out/G/report_3.md", "explore_out/G/report_4.md"
OUT = os.path.join(HERE, "recompute_G.md")

ROWS = []      # (file, line, item, reported, recomputed, status, note)


def row(f, line, item, reported, recomputed, status=None, note=""):
    if status is None:
        status = "MATCH" if str(reported) == str(recomputed) else "DISAGREE"
    ROWS.append((f, line, item, str(reported), str(recomputed), status, note))


# ── report text (only to cite lines; the reported values below are typed in, not parsed) ─────────────────────
REPORT_TEXT = {p: git("show", f"{G}:{p}").splitlines() for p in (R3, R4)}


def check_line(f, line, needle):
    """Guard against mistyped line numbers: the typed-in text must occur on the cited line."""
    txt = REPORT_TEXT[f][line - 1]
    if needle not in txt:
        raise SystemExit(f"{f}:{line} does not contain {needle!r}: {txt!r}")


# ── outcome definitions (explore_g_rhl_parent.py at 93422cf, docstring lines 6-18; code lines 35-87) ─────────
ROUTE_MARGIN, ROUTE_ETA, DISC_COS = 0.9, 0.9, 0.5      # rhl_parent.py:35-36
MARGIN_PARTIAL, ETA_SPLIT = 0.25, 0.5                  # test_router_layout.py:198-199 (v1 and v2 share them)


def bound(r):
    """rhl_parent.py:7 BOUND = transition is not None (explore_common.bound). Checked against rlib.transition(curve)."""
    return bool(r.get("ok")) and r["transition"] is not None


def routed_star(r):
    """rhl_parent.py:8-9 ROUTED*@end = end margin >= 0.9 and end eta^2 by stream at KEY > 0.9."""
    e = r["end"]
    return bool(e.get("margin", 0.0) >= ROUTE_MARGIN and e.get("eta_key_by_stream", 0.0) > ROUTE_ETA)


def discovered(r):
    """rhl_parent.py:10 DISCOVERED = BOUND and end VAL cos < 0.5."""
    return bool(bound(r) and r["val_cos"] < DISC_COS)


def routed_bound(r):
    """rhl_parent.py:11 ROUTED-BOUND = BOUND and ROUTED*@end (the S72/S73 readings' outcome)."""
    return bool(bound(r) and routed_star(r))


def bound_routed_mine(r):
    """rhl_parent.py:12-13 my BOUND ROUTED = BOUND, VAL stream->channel map one-to-one at every layer, every
    stream's held-out accuracy >= 0.9."""
    lay = r["end"].get("layers")
    return bool(bound(r) and lay and all(lay[f"L{l}"]["one_to_one_val"] for l in (1, 2, 3))
                and min(r["stream_acc"]) >= 0.9)


def fail_v1(r):
    """rhl_parent.py:14-15 failure class = test_router_layout.fail_class (v1, margin first) = rlib.fail_class."""
    return fail_class(r)


def fail_v2(r):
    """Main line fail_class_v2.py:40-48 (key rule before the margin rule). Sensitivity only."""
    e = r["end"]
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def unbound_class(r, fc=fail_v1):
    """rhl_parent.py:14-15 / outcome() lines 61-69: MERGED if STREAM-PARTIAL and the layer-3 VAL map not one-to-one."""
    c = fc(r)
    if c == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return c


def paired(ra, rb, f):
    """rhl_parent.py:16-18: paired by seed; b = first arm alone; one-sided exact McNemar P(Bin(b+c,1/2) >= b)."""
    ss = sorted(set(ra) & set(rb))
    b = sum(1 for s in ss if f(ra[s]) and not f(rb[s]))
    c = sum(1 for s in ss if f(rb[s]) and not f(ra[s]))
    return b, c, mcnemar1(b, c), len(ss)


OUTCOME_DEFS = [
    ("BOUND", "transition is not None (the run ends >= 0.95 from its transition on)",
     "explore_g_rhl_parent.py:7 (code :39-40); the recorded transition is also recomputed from the curve with rlib.transition"),
    ("ROUTED*@end", "end margin >= 0.9 and end eta^2 by stream at KEY > 0.9", "explore_g_rhl_parent.py:8-9 (code :35, :43-45)"),
    ("DISCOVERED", "BOUND and end VAL cos < 0.5", "explore_g_rhl_parent.py:10 (code :36, :48-49)"),
    ("ROUTED-BOUND", "BOUND and ROUTED*@end; the S72 and S73 readings use it, S71 uses BOUND",
     "explore_g_rhl_parent.py:11; explore_g_register3.py:18; explore_g_register3_delta.py:24; explore_g_layers.py:21"),
    ("my BOUND ROUTED", "BOUND, VAL stream->channel map one-to-one at layers 1, 2, 3, every stream's accuracy >= 0.9",
     "explore_g_rhl_parent.py:12-13 (code :56-58)"),
    ("failure class", "unbound k = 2 runs only: test_router_layout.fail_class v1 (margin >= 0.25 first, then eta^2 by key "
     ">= 0.5 KEY, then half/index POSITION, else OTHER); MERGED if STREAM-PARTIAL and the layer-3 VAL map is not one-to-one",
     "explore_g_rhl_parent.py:14-15 (code :61-69, count_table :105-109); test_router_layout.py:293-301"),
    ("fail_class v2 (sensitivity only)", "eta^2 by key >= 0.5 is KEY before the margin rule", "main line fail_class_v2.py:40-48"),
    ("beats", "one-sided exact McNemar p < 0.05 on the paired seeds", "explore_g_rhl_parent.py:16-18 (code :72-81); "
     "explore_g_layers.py:21; explore_g_register3.py:18; explore_g_register3_delta.py:24"),
    ("A >= B - 1 pair", "(B only) - (A only) <= 1", "explore_g_layers.py:21; explore_g_rhl_parent.py:18 (code :84-87)"),
    ("Wilson / band", "Wilson 95% (rlib.wilson, z = 1.95996; the screens' z = 1.96 gives the same 2-decimal bounds here); "
     "band RELIABLE >= ceil(0.9n), MAJORITY >= ceil(0.5n), MINORITY >= 1, else NEVER", "explore_g_rhl_parent.py:16; explore_g_common.py:51-73"),
    ("validity", "S71: each family VALID if its all-layer oracle binds >= 8/10; S72/S73 use S71's ORACLE_ALL / ORACLE_ALL_D",
     "explore_g_layers.py:19-20; explore_g_register3.py:17; explore_g_register3_delta.py:23"),
]

# ── records ───────────────────────────────────────────────────────────────────────────────────────────────
ST = {n: load(G, f"explore_out/G/{n}_results.json") for n in ("s71_layers", "s72_register3", "s73_register3_delta")}
ARMS = {
    "s71_layers": ("ORACLE_ALL", "ORACLE_23", "ORACLE_1", "ORACLE_ALL_D", "ORACLE_23_D"),
    "s72_register3": ("BASELINE", "REG3", "REG3_NUDGE"),
    "s73_register3_delta": ("BASELINE_D", "REG3_D", "REG3_NUDGE_D", "RESGATE_ONLY_D", "LATCH3_D", "LATCH3_NUDGE_D", "SINGLE_D"),
}
R = {}
for n, al in ARMS.items():
    for a in al:
        R[a] = runs(ST[n], a)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f2(x):
    return f"{x:.2f}"


def rng(xs, nd=2):
    return f"{min(xs):.{nd}f}-{max(xs):.{nd}f}"


def count_like(reported, k, n):
    """Format k/n with exactly the parts the report printed (count, [Wilson], band)."""
    s = f"{k}/{n}"
    if "[" in reported:
        lo, hi = wilson(k, n)
        s += f" [{lo:.2f}, {hi:.2f}]"
    if re.search(r"RELIABLE|MAJORITY|MINORITY|NEVER", reported):
        s += f" {band(k, n)}"
    return s


def trans_like(reported, rr):
    tr = sorted(r["transition"] for r in rr.values() if bound(r))
    if not tr:
        return "–"
    if "(all)" in reported:
        return f"{tr[0]} (all)" if len(set(tr)) == 1 else f"{med(tr):.0f} ({tr[0]}-{tr[-1]})"
    if len(tr) == 1:
        return f"{tr[0]}"
    return f"{med(tr):.0f} ({tr[0]}-{tr[-1]})"


def classes(rr, fc=fail_v1):
    c = Counter(unbound_class(r, fc) for r in rr.values() if not bound(r))
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) or "–"


# ── 0. record integrity ─────────────────────────────────────────────────────────────────────────────────────
def integrity():
    for n, al in ARMS.items():
        st = ST[n]
        tot = len(st["runs"])
        ok = sum(1 for r in st["runs"].values() if r.get("ok"))
        att = sum(1 for r in st["runs"].values() if r.get("attempt", 1) == 1)
        mism = [(r["arm"], r["seed"]) for r in st["runs"].values() if transition(r["curve"]) != r["transition"]]
        seeds_ok = all(sorted(R[a]) == sorted(st["meta"]["arms"][a]["seeds"]) for a in al)
        cells = all(sum(c["n"] for c in r["acc_by_cell"].values()) == 2048 and len(r["acc_by_cell"]) == 12
                    for a in al for r in R[a].values())
        yield n, tot, ok, att, mism, seeds_ok, cells


INTEG = list(integrity())
I = {n: (tot, ok, att, mism, sok, cells) for n, tot, ok, att, mism, sok, cells in INTEG}

# ======================================================================================================
# REPORT 3 (S71)
# ======================================================================================================
check_line(R3, 6, "50/50 runs ok, none retried")
t, ok, att, mism, sok, _ = I["s71_layers"]
row(R3, 6, "S71 runs ok / none retried", "50/50 ok, none retried", f"{ok}/{t} ok, {t - att} retried",
    "MATCH" if (ok, t, att) == (50, 50, 50) else "DISAGREE",
    f"'attempt' == 1 in every record; recorded transition equals rlib.transition(curve) in {t - len(mism)}/{t}")
row(R3, 6, "S71 seeds 700-709 in every arm", "700-709",
    "700-709" if sok and all(sorted(R[a]) == list(range(700, 710)) for a in ARMS["s71_layers"]) else "other",
    note="each arm's seeds equal meta.arms seeds")


def scan_seed_range():
    """'no record anywhere uses 700-759': every *_results.json with a 'runs' dict in the G tree at S71's readings
    commit 9114843 (which carries the main line's and earlier sessions' stores), plus F's S70 store."""
    hits, nfiles = [], 0
    files = [p for p in git("ls-tree", "-r", "--name-only", "9114843").splitlines() if p.endswith("_results.json")]
    for p in files:
        try:
            d = load("9114843", p)
        except Exception:
            continue
        if not isinstance(d, dict) or not isinstance(d.get("runs"), dict):
            continue
        nfiles += 1
        for k in d["runs"]:
            parts = k.split("|")
            if len(parts) > 1 and parts[1].isdigit() and 700 <= int(parts[1]) <= 759:
                hits.append((p, k))
    return nfiles, hits


nf, hits = scan_seed_range()
FREF = git("rev-list", "-1", f"--before={git('log', '-1', '--format=%cI', G).strip()}", "origin/claude/explore-F").strip()[:7]
FST = load(FREF, "explore_out/F/rh_lite_results.json")
fhits = [k for k in FST["runs"] if 700 <= int(k.split("|")[1]) <= 759]
check_line(R3, 6, "no record anywhere uses")
row(R3, 6, "no record anywhere uses seeds 700-759", "none", "none" if not hits and not fhits else f"{len(hits) + len(fhits)} records",
    "MATCH" if not hits and not fhits else "DISAGREE",
    f"scope: {nf} *_results.json stores with a runs dict in the G tree at 9114843, plus F's rh_lite_results.json at {FREF}; "
    "'anywhere' beyond these is not checkable from the records")
row(R3, 6, "S71 wall time", "42.8 min on 4 workers", "42.8 min (s71.log:83 'S71 runs done in 42.8 min')",
    "MATCH" if "S71 runs done in 42.8 min" in git("show", f"{G}:explore_out/G/s71.log") else "DISAGREE",
    "from the screen log, not the run records (records hold per-run secs only)")

# count table, report_3.md lines 13-17
T3 = [  # line, arm, BOUND, ROUTED-BOUND, my BOUND ROUTED, unbound, transitions, median acc
    (13, "ORACLE_ALL", "10/10 [0.72, 1.00] RELIABLE", "10/10", "10/10", "–", "1200 (1200-4800)", "1.000"),
    (14, "ORACLE_23", "0/10 [0.00, 0.28] NEVER", "0/10", "0/10", "STREAM-PARTIAL 10", "–", "0.504"),
    (15, "ORACLE_1", "9/10 [0.60, 0.98] RELIABLE", "9/10", "0/10", "1 (acc 0.937)", "16800 (8400-24000)", "0.955"),
    (16, "ORACLE_ALL_D", "10/10 [0.72, 1.00] RELIABLE", "10/10", "10/10", "–", "1200 (1200-1200)", "1.000"),
    (17, "ORACLE_23_D", "10/10 [0.72, 1.00] RELIABLE", "10/10", "0/10", "–", "1200 (1200-3600)", "1.000"),
]
for line, a, b_, rb_, br_, ub_, tr_, acc_ in T3:
    check_line(R3, line, a)
    rr = R[a]
    n = len(rr)
    row(R3, line, f"{a} BOUND (Wilson, band)", b_, count_like(b_, sum(bound(r) for r in rr.values()), n))
    row(R3, line, f"{a} ROUTED-BOUND", rb_, count_like(rb_, sum(routed_bound(r) for r in rr.values()), n))
    row(R3, line, f"{a} my BOUND ROUTED", br_, count_like(br_, sum(bound_routed_mine(r) for r in rr.values()), n))
    unb = [r for r in rr.values() if not bound(r)]
    if ub_.startswith("1 (acc"):
        rec = f"{len(unb)} (acc {unb[0]['acc']:.3f})" if len(unb) == 1 else f"{len(unb)}"
        row(R3, line, f"{a} unbound", ub_, rec, note=f"class of the unbound run: {classes(rr)}")
    else:
        row(R3, line, f"{a} unbound classes", ub_, classes(rr))
    row(R3, line, f"{a} transitions median (range)", tr_, trans_like(tr_, rr))
    row(R3, line, f"{a} median final acc", acc_, f"{med([r['acc'] for r in rr.values()]):.3f}")

# validity and readings, lines 21, 26-29
nA, nAD = sum(bound(r) for r in R["ORACLE_ALL"].values()), sum(bound(r) for r in R["ORACLE_ALL_D"].values())
check_line(R3, 21, "ORACLE_ALL 10/10 and ORACLE_ALL_D 10/10")
row(R3, 21, "validity: ORACLE_ALL BOUND; Hebbian family", "10/10, VALID", f"{nA}/10, {'VALID' if nA >= 8 else 'UNTESTED'}")
row(R3, 21, "validity: ORACLE_ALL_D BOUND; delta family", "10/10, VALID", f"{nAD}/10, {'VALID' if nAD >= 8 else 'UNTESTED'}")
vh, vd = nA >= 8, nAD >= 8

b, c, p, n = paired(R["ORACLE_ALL"], R["ORACLE_23"], bound)
check_line(R3, 26, "b 10, c 0, one-sided p = 0.00098")
row(R3, 26, "'layer 1 is the block' (ALL beats 23): b, c, p", "b 10, c 0, p = 0.00098", f"b {b}, c {c}, p = {p:.2g}")
row(R3, 26, "'layer 1 is the block' applies?", "APPLIES", "APPLIES" if (p < 0.05 and vh) else "does not apply")
b, c, p, n = paired(R["ORACLE_23"], R["ORACLE_ALL"], bound)
check_line(R3, 27, "ALL only 10")
row(R3, 27, "'layer 1 is not needed' (23 >= ALL - 1): ALL only / 23 only", "ALL only 10 (23 only 0)", f"ALL only {c} (23 only {b})")
row(R3, 27, "'layer 1 is not needed' applies?", "does not apply", "APPLIES" if (c - b <= 1 and vh) else "does not apply")
b, c, p, n = paired(R["ORACLE_1"], R["ORACLE_ALL"], bound)
check_line(R3, 28, "ALL only 1, ORACLE_1 only 0")
row(R3, 28, "'layer 1 alone suffices' (1 >= ALL - 1): ALL only / ORACLE_1 only", "ALL only 1, ORACLE_1 only 0",
    f"ALL only {c}, ORACLE_1 only {b}")
row(R3, 28, "'layer 1 alone suffices' applies?", "APPLIES", "APPLIES" if (c - b <= 1 and vh) else "does not apply")
b, c, p, n = paired(R["ORACLE_23_D"], R["ORACLE_ALL_D"], bound)
check_line(R3, 29, "0 / 0")
row(R3, 29, "'delta tolerates an unrouted layer 1' (23_D >= ALL_D - 1): discordant", "0 / 0", f"{b} / {c}")
row(R3, 29, "'delta tolerates an unrouted layer 1' applies?", "APPLIES", "APPLIES" if (c - b <= 1 and vd) else "does not apply")

# line 33: S56/S57's ORACLE23 0/3
S57 = load(G, "explore_out/G/s57_latch_results.json")
o23 = runs(S57, "ORACLE23")
check_line(R3, 33, "ORACLE23 (0/3, header P = 8)")
row(R3, 33, "S57 store: post-hoc ORACLE23 BOUND (P = 8)", "0/3", f"{sum(bound(r) for r in o23.values())}/{len(o23)}",
    note="explore_out/G/s57_latch_results.json at 93422cf, seeds " + ", ".join(map(str, sorted(o23))))


def cell_medians(rr_list):
    cells = sorted(set(k for r in rr_list for k in r["acc_by_cell"]))
    return {c: med([r["acc_by_cell"][c]["acc"] for r in rr_list if c in r["acc_by_cell"]]) for c in cells}


def flat_run(r):
    """explore_g_register3.py:21-22: a flat plateau = every (stream, block, index) cell within 0.1 of the run's mean."""
    c = [v["acc"] for v in r["acc_by_cell"].values()]
    return max(abs(x - statistics.fmean(c)) for x in c) <= 0.1


# lines 37-38: ORACLE_23 cells and write order
o23r = list(R["ORACLE_23"].values())
cm = cell_medians(o23r)
check_line(R3, 38, "0.46-0.51 in all 12 cells")
row(R3, 38, "ORACLE_23 acc by (stream, block, index), median over runs, range over the 12 cells", "0.46-0.51 (12 cells)",
    f"{rng(list(cm.values()))} ({len(cm)} cells)",
    note="per-run accuracies are bimodal: " + ", ".join(f"{r['acc']:.2f}" for r in o23r) +
    "; the per-cell medians fall between the ~0.32 and ~0.53 groups, so 'at chance between the two streams, everywhere' "
    "describes the median, not every run")
wo = [r["write_order"] for r in o23r]
check_line(R3, 38, "0.52 (last-written) vs 0.49 (not last)")
row(R3, 38, "ORACLE_23 write-order acc last / not-last (median)", "0.52 / 0.49",
    f"{f2(med([w['acc_last'] for w in wo]))} / {f2(med([w['acc_notlast'] for w in wo]))}")
m23 = [(r["end"]["layers"]["L2"]["margin"], r["end"]["layers"]["L3"]["margin"]) for r in o23r]
check_line(R3, 39, "margin 1.00")
row(R3, 39, "ORACLE_23 routing margin at VAL, layers 2-3 (every run)", "1.00",
    f"{min(min(x) for x in m23):.2f}-{max(max(x) for x in m23):.2f}",
    "MATCH" if all(round(x, 2) == 1.0 for t in m23 for x in t) else "DISAGREE")

# lines 42-46: ORACLE_1
o1 = R["ORACLE_1"]
tr1 = sorted(r["transition"] for r in o1.values() if bound(r))
check_line(R3, 42, "8400-24000 (median 16800)")
row(R3, 42, "ORACLE_1 transitions range (median)", "8400-24000 (median 16800)", f"{tr1[0]}-{tr1[-1]} (median {med(tr1):.0f})")
trA = sorted(r["transition"] for r in R["ORACLE_ALL"].values() if bound(r))
row(R3, 42, "ORACLE_ALL transition (median)", "1200", f"{med(trA):.0f}")
check_line(R3, 43, "0.95-0.97")
row(R3, 43, "ORACLE_1 bound runs' final accuracy range", "0.95-0.97", rng([r["acc"] for r in o1.values() if bound(r)]))
ub1 = [(s, r) for s, r in o1.items() if not bound(r)]
row(R3, 43, "ORACLE_1 unbound seed and its final accuracy", "708, 0.937",
    ", ".join(f"{s}, {r['acc']:.3f}" for s, r in ub1))
lm = [(r["end"]["layers"]["L2"]["margin"], r["end"]["layers"]["L3"]["margin"],
       r["end"]["layers"]["L2"]["one_to_one_val"], r["end"]["layers"]["L3"]["one_to_one_val"]) for r in o1.values()]
check_line(R3, 44, "margin 0.00, both streams on one channel")
row(R3, 44, "ORACLE_1 layers 2-3: margin; VAL map one-to-one", "margin 0.00; both streams on one channel (not one-to-one)",
    f"margin {min(min(x[:2]) for x in lm):.2f}..{max(max(x[:2]) for x in lm):.2f}; one-to-one in "
    f"{sum(x[2] or x[3] for x in lm)}/10 runs at L2 or L3",
    "MATCH" if all(abs(x[0]) < 0.005 and abs(x[1]) < 0.005 and not x[2] and not x[3] for x in lm) else "DISAGREE")
check_line(R3, 45, "708's \"MERGED\" label")
row(R3, 45, "ORACLE_1 708 outcome label", "MERGED", unbound_class(o1[708]),
    note=f"708: margin {o1[708]['end']['margin']:.2f} (ROUTED*@end {routed_star(o1[708])}), L3 VAL map "
    f"{o1[708]['end']['layers']['L3']['ch_map_val']}")

# lines 47-50: in-run probe medians


def pmed(a, key):
    return med([r["probe"][key]["acc"] for r in R[a].values()])


def pkv(a, L):
    return f"{f2(pmed(a, L + '|K'))} / {f2(pmed(a, L + '|V'))}"


check_line(R3, 48, "L2 0.64 / 0.51, L3 0.67 / 0.53, final 0.71 / 0.62")
for L, rep in (("L2", "0.64 / 0.51"), ("L3", "0.67 / 0.53"), ("final", "0.71 / 0.62")):
    row(R3, 48, f"probe median K / V, ORACLE_ALL {L}", rep, pkv("ORACLE_ALL", L))
check_line(R3, 48, "L3 0.81 / 0.52")
row(R3, 48, "probe median K / V, ORACLE_ALL_D L3", "0.81 / 0.52", pkv("ORACLE_ALL_D", "L3"))
check_line(R3, 49, "ORACLE_23_D: L3 0.73 / 0.51. ORACLE_1: L3 0.63 / 0.52")
row(R3, 49, "probe median K / V, ORACLE_23_D L3", "0.73 / 0.51", pkv("ORACLE_23_D", "L3"))
row(R3, 49, "probe median K / V, ORACLE_1 L3", "0.63 / 0.52", pkv("ORACLE_1", "L3"))
dK = [pmed(a, "L3|K") for a in ("ORACLE_ALL_D", "ORACLE_23_D")]
row(R3, 49, "delta stack at KEY (median L3 K over the delta arms)", "0.73-0.81", rng(dK))
vmax = max(pmed(a, f"{L}|V") for a in ARMS["s71_layers"] for L in ("L2", "L3", "final"))
check_line(R3, 47, "No arm's stack carries the stream at VAL positions")
row(R3, 47, "no arm's stack carries the stream at VAL (max arm-median probe V over L2/L3/final)", "< 0.9 (not carried)",
    f"max median {vmax:.2f}", "MATCH" if vmax < 0.9 else "DISAGREE")

# line 51: write order in bound oracle arms
wl = {}
for a in ARMS["s71_layers"]:
    w = [r["write_order"] for r in R[a].values()]
    wl[a] = (med([x["acc_last"] for x in w]), med([x["acc_notlast"] for x in w]))
bound_arms = [a for a in ARMS["s71_layers"] if sum(bound(r) for r in R[a].values()) > 0]
all1 = [a for a in bound_arms if round(wl[a][0], 2) == 1.0 and round(wl[a][1], 2) == 1.0]
check_line(R3, 51, "Write-order accuracy is 1.00 / 1.00 in every bound oracle arm")
row(R3, 51, "write-order acc last / not-last = 1.00 / 1.00 in every bound oracle arm (median)", "1.00 / 1.00 (every bound arm)",
    "; ".join(f"{a} {f2(x)} / {f2(y)}" for a, (x, y) in wl.items() if a in bound_arms),
    "MATCH" if len(all1) == len(bound_arms) else "PARTIAL",
    "holds for ORACLE_ALL, ORACLE_ALL_D, ORACLE_23_D; ORACLE_1 binds 9/10 but at ~0.95 accuracy, so its write-order medians "
    "are not 1.00 — true if 'bound oracle arm' means the three arms whose runs end at 1.00")

# CHECK files
CK = {}
for p in git("ls-tree", "-r", "--name-only", G, "explore_out/G/").splitlines():
    m = re.match(r"explore_out/G/checks_(s7[123])_\d+_\d+\.json$", p)
    if m:
        CK.setdefault(m.group(1), []).append(load(G, p))
ck71 = CK["s71"]
check_line(R3, 53, "all asserted, all ok; 23")
row(R3, 53, "S71 CHECKs: count, all ok", "23, all ok", f"{len(ck71[0]['rows'])}, {'all ok' if all(v for _, v in ck71[0]['rows']) else 'FAIL'}",
    note=f"{len(ck71)} checks_s71 file(s) at 93422cf; X's arm A bit for bit is the repro check (not in the file)")
blk = [nm for nm, _ in ck71[0]["rows"] if "block-length frequencies" in nm]
check_line(R3, 55, "0.338 / 0.324 / 0.338, 6 block orders")
row(R3, 55, "check_rand_header block lengths; block orders", "0.338 / 0.324 / 0.338; 6 orders",
    ("0.338 / 0.324 / 0.338" if blk and "{1: 0.338, 2: 0.324, 3: 0.338}" in blk[0] else "?") + "; " +
    ("6 orders" if any("6 distinct orders" in nm for nm, _ in ck71[0]["rows"]) else "?"),
    note="from the CHECK file's row text")
_, _, _, _, _, cells71 = I["s71_layers"]
check_line(R3, 58, "acc_by_cell covers all 2048 queries")
row(R3, 58, "acc_by_cell covers 2048 queries in 12 cells (every S71 run)", "2048 (all runs)",
    "2048 (all runs)" if cells71 else "not in every run")
check_line(R3, 68, "37/37")
row(R3, 68, "S73 CHECKs passed", "37/37", f"{sum(1 for _, v in CK['s73'][0]['rows'] if v)}/{len(CK['s73'][0]['rows'])}")

# ======================================================================================================
# REPORT 4 (S73, S72)
# ======================================================================================================
check_line(R4, 5, "150/150 runs ok, none retried")
t72, ok72, att72, mism72, sok72, cells72 = I["s72_register3"]
t73, ok73, att73, mism73, sok73, cells73 = I["s73_register3_delta"]
row(R4, 5, "S72 + S73 runs ok / none retried", "150/150 ok, none retried",
    f"{ok72 + ok73}/{t72 + t73} ok, {t72 + t73 - att72 - att73} retried",
    "MATCH" if (ok72 + ok73, t72 + t73, att72 + att73) == (150, 150, 150) else "DISAGREE",
    f"recorded transition equals rlib.transition(curve) in {t72 + t73 - len(mism72) - len(mism73)}/{t72 + t73}; "
    f"acc_by_cell covers 2048 queries in every run: {cells72 and cells73}")
check_line(R4, 6, "ORACLE_ALL 10/10, delta ORACLE_ALL_D 10/10")
row(R4, 5, "validity (S71 oracles): ORACLE_ALL / ORACLE_ALL_D; families", "10/10 / 10/10, both VALID",
    f"{nA}/10 / {nAD}/10, {'both VALID' if vh and vd else 'not both VALID'}")

# count table, report_4.md lines 13-23
T4 = [  # line, arm, BOUND, ROUTED-BOUND, ROUTED*@end, DISCOVERED, mine, classes, transitions
    (13, "BASELINE_D", "14/20 [0.48, 0.85]", "0/20 [0.00, 0.16] NEVER", "0/20", "0/20", "11/20", "KEY 6", "3600 (3600-13200)"),
    (14, "REG3_D", "8/20 [0.22, 0.61]", "4/20 [0.08, 0.42] MINORITY", "4/20", "5/20", "5/20", "KEY 12", "2400 (1200-16800)"),
    (15, "REG3_NUDGE_D", "10/10", "10/10 [0.72, 1.00] RELIABLE", "10/10", "10/10", "10/10", "–", "1200 (all)"),
    (16, "RESGATE_ONLY_D", "5/10 [0.24, 0.76]", "0/10 [0.00, 0.28] NEVER", "0/10", "0/10", "0/10", "KEY 5", "6000 (4800-15600)"),
    (17, "LATCH3_D", "1/10 [0.02, 0.40]", "0/10 NEVER", "0/10", "0/10", "0/10", "KEY 9", "13200"),
    (18, "LATCH3_NUDGE_D", "9/10", "9/10 [0.60, 0.98] RELIABLE", "9/10", "9/10", "9/10", "STREAM-PARTIAL 1", "1200 (all)"),
    (19, "SINGLE_D", "7/10 [0.40, 0.89] MAJORITY", None, None, None, None, "3 (acc 0.48-0.50)", "10800 (1200-20400)"),
    (21, "BASELINE", "1/20 [0.01, 0.24]", "0/20 [0.00, 0.16] NEVER", "0/20", "0/20", "1/20", "KEY 16, STREAM-PARTIAL 3", "16800"),
    (22, "REG3", "4/20 [0.08, 0.42]", "4/20 [0.08, 0.42] MINORITY", "4/20", "4/20", "4/20", "KEY 14, OTHER 2", "3000 (2400-6000)"),
    (23, "REG3_NUDGE", "20/20", "20/20 [0.84, 1.00] RELIABLE", "20/20", "20/20", "20/20", "–", "2400 (2400-8400)"),
]
for line, a, b_, rb_, rs_, d_, br_, cl_, tr_ in T4:
    check_line(R4, line, a)
    rr = R[a]
    n = len(rr)
    row(R4, line, f"{a} BOUND", b_, count_like(b_, sum(bound(r) for r in rr.values()), n))
    if rb_ is not None:
        row(R4, line, f"{a} ROUTED-BOUND", rb_, count_like(rb_, sum(routed_bound(r) for r in rr.values()), n))
        row(R4, line, f"{a} ROUTED*@end", rs_, count_like(rs_, sum(routed_star(r) for r in rr.values()), n))
        row(R4, line, f"{a} DISCOVERED", d_, count_like(d_, sum(discovered(r) for r in rr.values()), n))
        row(R4, line, f"{a} my BOUND ROUTED", br_, count_like(br_, sum(bound_routed_mine(r) for r in rr.values()), n))
        row(R4, line, f"{a} unbound classes (v1, unbound runs only)", cl_, classes(rr))
    else:
        unb = [r for r in rr.values() if not bound(r)]
        row(R4, line, f"{a} unbound (k = 1)", cl_, f"{len(unb)} (acc {rng([r['acc'] for r in unb])})")
    row(R4, line, f"{a} transitions median (range)", tr_, trans_like(tr_, rr))

# footnote line 25-28
bd = [r for r in R["BASELINE_D"].values() if bound_routed_mine(r)]
vals = []
for r in bd:
    for l in (1, 2, 3):
        tab = r["end"]["layers"][f"L{l}"]["table"]["VAL"]
        vals += [max(row_) for row_ in tab]
check_line(R4, 25, "(0.50 vs")
row(R4, 25, "BASELINE_D my-BOUND-ROUTED runs: 'near-uniform gates (0.50 vs 0.51)' (VAL contingency, larger channel mass per stream)",
    "0.50 vs 0.51 (near-uniform)", f"larger mass {rng(vals)}; end margin {rng([r['end']['margin'] for r in bd])}",
    "MATCH" if max(vals) <= 0.55 else "DISAGREE",
    "the footnote's example is checked against the VAL contingency tables at every layer of the 11 runs; line 68 of the "
    "same report calls the same runs 'half-routed gates with margin ≈ 0.5'")

# F's references, lines 26-27
for line, arm, rep_b, rep_rs, rep_key in ((26, "S70_HEBB", "1/20", "0/20", "19"), (27, "S70_DELTA", "6/20", "0/20", "14")):
    rr = runs(FST, arm)
    nb = sum(bound(r) for r in rr.values())
    nrs = sum(routed_star(r) for r in rr.values())
    nk = sum(1 for r in rr.values() if not bound(r) and fail_v1(r) == "KEY")
    check_line(R4, line, arm)
    row(R4, line, f"F's {arm} BOUND, ROUTED*@end, KEY (unbound, v1)", f"{rep_b}, {rep_rs}, KEY {rep_key}",
        f"{nb}/{len(rr)}, {nrs}/{len(rr)}, KEY {nk}",
        note=f"explore_out/F/rh_lite_results.json at {FREF} (claude/explore-F), seeds {min(rr)}-{max(rr)}")
check_line(R4, 28, "14/20 against 6/20")
row(R4, 28, "BASELINE_D vs F's S70_DELTA BOUND; routed", "14/20 vs 6/20; routes none either way",
    f"{sum(bound(r) for r in R['BASELINE_D'].values())}/20 vs {sum(bound(r) for r in runs(FST, 'S70_DELTA').values())}/20; "
    f"ROUTED*@end {sum(routed_star(r) for r in R['BASELINE_D'].values())} and "
    f"{sum(routed_star(r) for r in runs(FST, 'S70_DELTA').values())}",
    "MATCH")
row(R4, 27, "'My BASELINE matches F's Hebbian counts' (BOUND, ROUTED*@end, KEY)", "matches",
    f"BOUND {sum(bound(r) for r in R['BASELINE'].values())}/20 vs {sum(bound(r) for r in runs(FST, 'S70_HEBB').values())}/20, "
    f"ROUTED*@end {sum(routed_star(r) for r in R['BASELINE'].values())}/20 vs {sum(routed_star(r) for r in runs(FST, 'S70_HEBB').values())}/20, "
    f"KEY {sum(1 for r in R['BASELINE'].values() if not bound(r) and fail_v1(r) == 'KEY')} vs "
    f"{sum(1 for r in runs(FST, 'S70_HEBB').values() if not bound(r) and fail_v1(r) == 'KEY')}",
    status="PARTIAL",
    note="the BOUND and ROUTED*@end counts match (1/20, 0/20), which is what the sentence contrasts with BASELINE_D; the "
    "failure classes do not (KEY 16 + STREAM-PARTIAL 3 vs F's KEY 19), so 'matches F's Hebbian counts' holds only for the "
    "counts it goes on to compare")

# readings, lines 34-42
nrb = sum(routed_bound(r) for r in R["REG3_NUDGE_D"].values())
check_line(R4, 34, "REG3_NUDGE_D ≥ 9/10")
row(R4, 34, "S73 'the register carries the cue' (REG3_NUDGE_D ROUTED-BOUND >= 9/10)", "APPLIES (10/10)",
    f"{'APPLIES' if nrb >= 9 and vd else 'does not apply'} ({nrb}/10)")
for line, arm, rep in ((35, "REG3_D", "does not apply: 4 / 0, p = 0.0625 (n 20)"),
                       (36, "RESGATE_ONLY_D", "does not apply: 0 / 0"),
                       (37, "LATCH3_D", "does not apply: 0 / 0")):
    b, c, p, n = paired(R[arm], R["BASELINE_D"], routed_bound)
    check_line(R4, line, f"{arm} beats BASELINE_D")
    ap = "APPLIES" if p < 0.05 and vd else "does not apply"
    rec = f"{ap}: {b} / {c}, p = {p:.3g} (n {n})" if "p =" in rep else f"{ap}: {b} / {c}"
    row(R4, line, f"S73 'discovered' {arm} beats BASELINE_D (ROUTED-BOUND)", rep, rec,
        note="" if "p =" in rep else f"p = {p:.3g}, n {n} (not printed)")
zs = [r["end"]["beta"] for r in R["LATCH3_NUDGE_D"].values() if r["end"].get("beta")]
nz = sum(z["CTX"] >= 0.9 and z["KEY"] <= 0.1 and z["VAL"] <= 0.1 for z in zs)
check_line(R4, 38, "(9/10; z = 1.00 / 0.00 / 0.00)")
row(R4, 38, "S73 'the latch holds' (LATCH3_NUDGE_D end z CTX >= 0.9, KEY/VAL <= 0.1 on >= 8/10); median z",
    "APPLIES (9/10; z = 1.00 / 0.00 / 0.00)",
    f"{'APPLIES' if nz >= 8 and vd else 'does not apply'} ({nz}/{len(zs)}; z = "
    f"{f2(med([z['CTX'] for z in zs]))} / {f2(med([z['KEY'] for z in zs]))} / {f2(med([z['VAL'] for z in zs]))})")


def pres(r):
    return r["probe"]["L3|K"]["acc"] >= 0.9 and r["probe"]["L3|V"]["acc"] >= 0.9   # register3_delta.py:30-31


npb = sum(pres(r) for r in R["BASELINE_D"].values())
check_line(R4, 39, "does not apply (7/20)")
row(R4, 39, "S73 'the delta stack carries the cue' (BASELINE_D probe L3 >= 0.9 at K and V on >= 10/20)", "does not apply (7/20)",
    f"{'APPLIES' if npb >= 10 else 'does not apply'} ({npb}/20)")
ns = sum(bound(r) for r in R["SINGLE_D"].values())
check_line(R4, 40, "(7/10, one short)")
row(R4, 40, "S73 'the gate is not needed' (SINGLE_D BOUND >= 8/10)", "does not apply (7/10, one short)",
    f"{'APPLIES' if ns >= 8 else 'does not apply'} ({ns}/10, {8 - ns} short)" if ns < 8 else f"APPLIES ({ns}/10)",
    "MATCH" if ns == 7 else "DISAGREE")
n72 = sum(routed_bound(r) for r in R["REG3_NUDGE"].values())
r1 = "the register carries the cue at all three layers" if n72 >= 17 else "architecture insufficient" if n72 < 12 else "neither"
check_line(R4, 41, "(20/20)")
row(R4, 41, "S72 R1 (REG3_NUDGE ROUTED-BOUND >= 17/20)", "\"the register carries the cue at all three layers\" (20/20)",
    f"\"{r1 if vh else 'UNTESTED'}\" ({n72}/20)")
b, c, p, n = paired(R["REG3"], R["BASELINE"], routed_bound)
check_line(R4, 42, "4 / 0, p = 0.0625 (n 20)")
row(R4, 42, "S72 R2 'discovered' (REG3 beats BASELINE, ROUTED-BOUND)", "not discovered: 4 / 0, p = 0.0625 (n 20)",
    f"{'discovered' if p < 0.05 and vh else 'not discovered'}: {b} / {c}, p = {p:.3g} (n {n})")

# What this says, lines 46-72
S56 = load(G, "explore_out/G/s56_register_results.json")
rn = runs(S56, "REG_NUDGE")
check_line(R4, 47, "bound 0/10 on the P = 8 header")
row(R4, 47, "S56 store: REG_NUDGE BOUND (P = 8)", "0/10", f"{sum(bound(r) for r in rn.values())}/{len(rn)}",
    note="explore_out/G/s56_register_results.json at 93422cf")
nud = list(R["REG3_NUDGE"].values())
o2o = all(all(r["end"]["layers"][f"L{l}"]["one_to_one_val"] for l in (1, 2, 3)) for r in nud)
mg = [r["end"]["layers"][f"L{l}"]["margin"] for r in nud for l in (1, 2, 3)]
check_line(R4, 48, "binds 20/20")
row(R4, 48, "REG3_NUDGE: BOUND, one-to-one at every layer, margin at every layer", "20/20, all one-to-one, margin 1.00",
    f"{sum(bound(r) for r in nud)}/20, {'all one-to-one' if o2o else 'not all one-to-one'}, margin {rng(mg)}",
    "MATCH" if (sum(bound(r) for r in nud) == 20 and o2o and min(mg) >= 0.995) else "DISAGREE")
trn = sorted(r["transition"] for r in nud if bound(r))
check_line(R4, 49, "2400-8400")
row(R4, 49, "REG3_NUDGE transitions range", "2400-8400", f"{trn[0]}-{trn[-1]}")
rs72 = [s for s, r in sorted(R["REG3"].items()) if routed_bound(r)]
rs73 = [s for s, r in sorted(R["REG3_D"].items()) if routed_bound(r)]
check_line(R4, 51, "REG3 Hebbian 700, 703, 714, 715; REG3_D 702, 711, 715, 717; only 715 in common")
row(R4, 51, "REG3 ROUTED-BOUND seeds", "700, 703, 714, 715", ", ".join(map(str, rs72)))
row(R4, 51, "REG3_D ROUTED-BOUND seeds", "702, 711, 715, 717", ", ".join(map(str, rs73)))
row(R4, 51, "seeds in common", "715", ", ".join(map(str, sorted(set(rs72) & set(rs73)))))
mgr = [r["end"]["layers"][f"L{l}"]["margin"] for a, ss in (("REG3", rs72), ("REG3_D", rs73)) for s in ss for l in (1, 2, 3)
       for r in [R[a][s]]]
nbu = sum(1 for r in R["REG3_D"].values() if bound(r) and not routed_bound(r))
check_line(R4, 52, "the bound runs route perfectly, against 0/20 routed for either baseline")
row(R4, 52, "the 8 discovering runs route perfectly (margin at every layer); routed baselines",
    "route perfectly; 0/20 routed for either baseline",
    f"margin {rng(mgr)}; baselines ROUTED*@end {sum(routed_star(r) for r in R['BASELINE'].values())}/20 and "
    f"{sum(routed_star(r) for r in R['BASELINE_D'].values())}/20",
    "PARTIAL" if nbu else "MATCH",
    f"true of the 4 + 4 ROUTED-BOUND runs; 'each time the bound runs route' does not hold for REG3_D's other {nbu} bound runs "
    "(bound, not routed)")
b5, c5, p5, _ = paired(R["REG3_D"], R["BASELINE_D"], routed_bound)
check_line(R4, 52, "one discordant pair")
row(R4, 52, "'one discordant pair short of p < 0.05' (each screen)", "5 / 0 would give p < 0.05",
    f"4 / 0 p = {mcnemar1(4, 0):.4g}; 5 / 0 p = {mcnemar1(5, 0):.4g}", "MATCH" if mcnemar1(5, 0) < 0.05 <= mcnemar1(4, 0) else "DISAGREE")
pb, pc = 0, 0
for a, bse in (("REG3", "BASELINE"), ("REG3_D", "BASELINE_D")):
    b, c, p, n = paired(R[a], R[bse], routed_bound)
    pb, pc = pb + b, pc + c
npairs = len(set(R["REG3"]) & set(R["BASELINE"])) + len(set(R["REG3_D"]) & set(R["BASELINE_D"]))
check_line(R4, 54, "8 / 0 over")
row(R4, 54, "pooled (post hoc, descriptive) discordant pairs over both screens", "8 / 0 over 40 pairs", f"{pb} / {pc} over {npairs} pairs",
    note=f"not pre-registered (the report says so); the one-sided McNemar on it would be p = {mcnemar1(pb, pc):.3g}")


def beta_end(a, seeds):
    return [R[a][s]["end"]["beta"] for s in seeds]


bh, bdl = beta_end("REG3", rs72), beta_end("REG3_D", rs73)
check_line(R4, 57, "0.32-0.64 vs 0.16-0.24")
row(R4, 56, "routed unnudged runs: end beta^R at CTX above K and V in every run", "every run",
    "every run" if all(z["CTX"] > max(z["KEY"], z["VAL"]) for z in bh + bdl) else "not every run")
row(R4, 57, "REG3 routed runs end beta^R CTX vs K/V", "0.32-0.64 vs 0.16-0.24",
    f"{rng([z['CTX'] for z in bh])} vs {rng([z[k] for z in bh for k in ('KEY', 'VAL')])}")
row(R4, 57, "REG3_D routed runs end beta^R CTX vs K/V", "0.36-0.58 vs ≈ 0.2",
    f"{rng([z['CTX'] for z in bdl])} vs {rng([z[k] for z in bdl for k in ('KEY', 'VAL')])}",
    "MATCH" if rng([z['CTX'] for z in bdl]) == "0.36-0.58" else "DISAGREE", "'≈ 0.2' read as 0.19-0.25")
ks = [(a, s, r["end"]["beta"]) for a in ("REG3", "REG3_D") for s, r in R[a].items() if fail_v1(r) == "KEY"]
flat = [x for x in ks if all(0.15 <= x[2][k] <= 0.35 for k in ("CTX", "KEY", "VAL"))]
higherK = [x for x in ks if x[2]["KEY"] >= max(x[2]["CTX"], x[2]["VAL"])]
both = [x for x in ks if x in flat or x in higherK]
check_line(R4, 57, "In the key-split runs it stayed flat")
row(R4, 57, "key-split runs (v1 KEY, all runs of REG3 + REG3_D): end beta^R flat (every role in 0.15-0.35) or highest at KEY",
    "every key-split run",
    f"{len(both)}/{len(ks)} (flat {len(flat)}, KEY highest {len(higherK)}); CTX range {rng([x[2]['CTX'] for x in ks])}",
    "MATCH" if len(both) == len(ks) else "PARTIAL",
    "exceptions: " + ", ".join(f"{a} {s} C/K/V {z['CTX']:.2f}/{z['KEY']:.2f}/{z['VAL']:.2f}" for a, s, z in ks if (a, s, z) not in both)
    if len(both) != len(ks) else "'≈ 0.2-0.3 at every role' read as every role in 0.15-0.35")
nb = [r["end"]["beta"] for r in R["REG3_NUDGE"].values()] + [r["end"]["beta"] for r in R["REG3_NUDGE_D"].values()]
check_line(R4, 58, "CTX 0.45-0.84 vs K/V ≈ 0.18")
row(R4, 58, "nudged (REG3_NUDGE, REG3_NUDGE_D) end beta^R CTX range; K/V median", "CTX 0.45-0.84 vs K/V ≈ 0.18",
    f"CTX {rng([z['CTX'] for z in nb])} vs K/V median {f2(med([z[k] for z in nb for k in ('KEY', 'VAL')]))} "
    f"(range {rng([z[k] for z in nb for k in ('KEY', 'VAL')])})",
    "MATCH" if rng([z['CTX'] for z in nb]) == "0.45-0.84" else "DISAGREE")

# line 59: key splits — the population question
kh_unb = sum(1 for r in R["REG3"].values() if not bound(r) and fail_v1(r) == "KEY")
kd_unb = sum(1 for r in R["REG3_D"].values() if not bound(r) and fail_v1(r) == "KEY")
kh_all = sum(1 for r in R["REG3"].values() if fail_v1(r) == "KEY")
kd_all = sum(1 for r in R["REG3_D"].values() if fail_v1(r) == "KEY")
kh_v2 = sum(1 for r in R["REG3"].values() if fail_v2(r) == "KEY")
kd_v2 = sum(1 for r in R["REG3_D"].values() if fail_v2(r) == "KEY")
nrh = sum(1 for r in R["REG3"].values() if not routed_bound(r))
nrd = sum(1 for r in R["REG3_D"].values() if not routed_bound(r))
check_line(R4, 59, "in 16 of 20 seeds the gate locks into a key split first (KEY 14 Hebbian, 12 delta)")
row(R4, 59, "KEY (v1, unbound only) REG3 / REG3_D", "KEY 14 Hebbian, 12 delta", f"KEY {kh_unb} Hebbian, {kd_unb} delta")
row(R4, 59, "'in 16 of 20 seeds the gate locks into a key split'", "16 of 20 (each memory)",
    f"key-split (v1 KEY, all runs): {kh_all} Hebbian, {kd_all} delta; v2 all runs: {kh_v2}, {kd_v2}; "
    f"not ROUTED-BOUND: {nrh}, {nrd}",
    "DISAGREE",
    "16 is the number of seeds that are not ROUTED-BOUND on each memory, not of key splits: Hebbian adds 2 OTHER "
    "(709, 718: eta^2 by key 0.02, 0.00); delta has 12 unbound KEY + 2 bound KEY (700, 709) + 2 bound STREAM-PARTIAL "
    "(714, 719). No classification of the records gives 16 key splits")

# lines 60-62: latch
ln = R["LATCH3_NUDGE_D"]
check_line(R4, 61, "z = 1 at CTX, 0 at K/V on 9/10")
row(R4, 61, "LATCH3_NUDGE_D exact latch (end z CTX = 1, K/V = 0 to 2 dp) runs", "9/10",
    f"{sum(1 for r in ln.values() if round(r['end']['beta']['CTX'], 2) == 1 and round(r['end']['beta']['KEY'], 2) == 0 and round(r['end']['beta']['VAL'], 2) == 0)}/10")
lrb = [r for r in ln.values() if routed_bound(r)]
row(R4, 61, "LATCH3_NUDGE_D binds routed; transitions", "9/10 at 1200",
    f"{len(lrb)}/10 at {'/'.join(sorted(set(str(r['transition']) for r in lrb)))}")
l3 = R["LATCH3_D"]


def z_at(r, step):
    for s in r["stats"]:
        if s["step"] == step:
            return s.get("beta")
    return None


z1200 = [z_at(r, 1200) for r in l3.values()]
all1_1200 = sum(1 for z in z1200 if z and all(round(z[k], 2) == 1.0 for k in ("CTX", "KEY", "VAL")))
zend_all1 = sum(1 for r in l3.values() if all(round(r["end"]["beta"][k], 2) == 1.0 for k in ("CTX", "KEY", "VAL")))
check_line(R4, 61, "LATCH3_D goes to z = 1 everywhere by update")
row(R4, 61, "LATCH3_D z = 1 at CTX/KEY/VAL by update 1200 (per run)", "z = 1 everywhere by 1200",
    f"{all1_1200}/10 runs at 1200; {zend_all1}/10 at the end; medians at 1200 "
    f"{'/'.join(f2(med([z[k] for z in z1200 if z])) for k in ('CTX', 'KEY', 'VAL'))}",
    "MATCH" if all1_1200 == 10 else "PARTIAL",
    "true of the median run; seeds " + ", ".join(str(s) for s, r in sorted(l3.items())
                                                 if not (z_at(r, 1200) and all(round(z_at(r, 1200)[k], 2) == 1.0 for k in ("CTX", "KEY", "VAL"))))
    + " do not write every token at 1200 (z = 0 everywhere for 700 and 704 from 1200 on)")
kl_unb = sum(1 for r in l3.values() if not bound(r) and fail_v1(r) == "KEY")
kl_all = sum(1 for r in l3.values() if fail_v1(r) == "KEY")
check_line(R4, 62, "key-splits 9/10")
row(R4, 62, "LATCH3_D key-splits", "9/10", f"{kl_unb}/10 (unbound only); {kl_all}/10 (all runs, v1)",
    "PARTIAL" if kl_unb != kl_all else "MATCH",
    "two readings: 9 counts unbound KEY runs; the one bound run (704) is also a key split (eta^2 by key 1.00), so 10/10 runs key-split")

# 63-64 RESGATE
rg = R["RESGATE_ONLY_D"]
check_line(R4, 63, "RESGATE_ONLY_D binds 5/10 but routes 0/10")
row(R4, 63, "RESGATE_ONLY_D BOUND; ROUTED*@end", "5/10; 0/10",
    f"{sum(bound(r) for r in rg.values())}/10; {sum(routed_star(r) for r in rg.values())}/10")


def table_med(rr, l, role):
    return [[round(med([r["end"]["layers"][f"L{l}"]["table"][role][s][c] for r in rr]), 2) for c in range(2)] for s in range(2)]


tm = {(l, role): table_med(list(rg.values()), l, role) for l in (1, 2, 3) for role in ("KEY", "VAL")}
check_line(R4, 64, "its gates are 0.5/0.5 at K and V in every layer")
row(R4, 64, "RESGATE_ONLY_D contingency (median g) at KEY and VAL, every layer", "0.5/0.5",
    "; ".join(f"L{l} {role} {t}" for (l, role), t in tm.items()),
    "MATCH" if all(abs(x - 0.5) <= 0.05 for t in tm.values() for rw in t for x in rw) else "DISAGREE",
    "median over the 10 runs of each cell")
rgb = [(s, r) for s, r in sorted(rg.items()) if bound(r)]
rgk = [s for s, r in rgb if fail_v1(r) == "KEY"]
check_line(R4, 64, "The bound runs are key-split bindings")
row(R4, 64, "RESGATE_ONLY_D bound runs that are key splits (v1 class on the end gate)", "all bound runs (5/5)",
    f"{len(rgk)}/{len(rgb)} ({', '.join(map(str, rgk))})", "DISAGREE" if len(rgk) != len(rgb) else "MATCH",
    "not key splits: " + ", ".join(f"{s} (eta^2 by key {r['end']['eta_key_by_key']:.2f}, margin {r['end']['margin']:.2f}, "
                                  f"class {fail_v1(r)})" for s, r in rgb if s not in rgk))

# 65-68 probes
routed_runs = [(a, s, r) for a in ARMS["s72_register3"] + ARMS["s73_register3_delta"] for s, r in R[a].items() if routed_bound(r)]
pk = [r["probe"]["L3|K"]["acc"] for _, _, r in routed_runs]
pv = [r["probe"]["L3|V"]["acc"] for _, _, r in routed_runs]
check_line(R4, 65, "≈ 0.50-0.57 in every")
row(R4, 65, "probe L3 K / V in every ROUTED-BOUND run (S72 + S73)", "≈ 0.50-0.57 / ≈ 0.50-0.57",
    f"K {rng(pk)} / V {rng(pv)} over {len(routed_runs)} runs",
    "MATCH" if max(pk + pv) <= 0.6 else "DISAGREE",
    f"{sum(1 for x in pk if x > 0.6)} routed runs have probe K > 0.60: " +
    ", ".join(f"{a} {s} {r['probe']['L3|K']['acc']:.2f}" for a, s, r in routed_runs if r["probe"]["L3|K"]["acc"] > 0.6))
sb = [r for r in R["SINGLE_D"].values() if bound(r)]
check_line(R4, 67, "SINGLE_D's bound runs (probe L3 0.93-1.00 / 0.56-1.00, final accuracy 1.00)")
row(R4, 67, "SINGLE_D bound runs probe L3 K / V", "0.93-1.00 / 0.56-1.00",
    f"{rng([r['probe']['L3|K']['acc'] for r in sb])} / {rng([r['probe']['L3|V']['acc'] for r in sb])}")
row(R4, 67, "SINGLE_D bound runs final accuracy", "1.00", rng([r["acc"] for r in sb]),
    "MATCH" if all(round(r["acc"], 2) == 1.0 for r in sb) else "PARTIAL", "two runs end at 0.99 to 2 dp")
bdb = [(s, r) for s, r in sorted(R["BASELINE_D"].items()) if bound(r)]
bdb_sp = [(s, r) for s, r in bdb if fail_v1(r) == "STREAM-PARTIAL"]
check_line(R4, 67, "BASELINE_D's bound runs (0.87-0.98")
row(R4, 67, "BASELINE_D bound runs probe L3 (K, V)", "0.87-0.98",
    f"all {len(bdb)} bound: K {rng([r['probe']['L3|K']['acc'] for _, r in bdb])}, V {rng([r['probe']['L3|V']['acc'] for _, r in bdb])}; "
    f"the {len(bdb_sp)} margin-≈0.5 ones: K {rng([r['probe']['L3|K']['acc'] for _, r in bdb_sp])}, "
    f"V {rng([r['probe']['L3|V']['acc'] for _, r in bdb_sp])}; median over K/V of those {f2(med([r['probe'][k]['acc'] for _, r in bdb_sp for k in ('L3|K', 'L3|V')]))}",
    "DISAGREE",
    f"no population reproduces 0.87-0.98; {len(bdb) - len(bdb_sp)} of the 14 bound runs are key-split bindings "
    "(701, 710, 719: probe K 0.53-0.66) and the half-routed ones span 0.79-1.00")
row(R4, 68, "BASELINE_D bound half-routed runs: margin ≈ 0.5", "≈ 0.5", f"{rng([r['end']['margin'] for _, r in bdb_sp])} ({len(bdb_sp)} runs)",
    "MATCH" if all(0.4 <= r["end"]["margin"] <= 0.6 for _, r in bdb_sp) else "DISAGREE")
t2 = [[sum(1 for r in R["BASELINE_D"].values() if pres(r) == p_ and routed_bound(r) == q) for q in (True, False)] for p_ in (True, False)]
check_line(R4, 68, "[[0, 7], [0, 13]]")
row(R4, 68, "BASELINE_D 2x2 probe >= 0.9 x ROUTED-BOUND", "[[0, 7], [0, 13]]", str(t2))

# 70-72 SINGLE_D
sd = R["SINGLE_D"]
trs = sorted(r["transition"] for r in sd.values() if bound(r))
check_line(R4, 70, "binds this task 7/10** (transitions 1200-20400)")
row(R4, 70, "SINGLE_D BOUND; transitions range", "7/10; 1200-20400", f"{len(trs)}/10; {trs[0]}-{trs[-1]}")
su = [r for r in sd.values() if not bound(r)]
cells_u = [x["acc"] for r in su for x in r["acc_by_cell"].values()]
check_line(R4, 70, "the three unbound runs sit at exactly 0.50")
row(R4, 70, "SINGLE_D unbound runs: final acc; flat in every cell", "exactly 0.50, flat in every cell",
    f"acc {', '.join(f'{r['acc']:.3f}' for r in su)}; flat (register3.py:21-22 rule) {sum(flat_run(r) for r in su)}/{len(su)}; "
    f"cells {rng(cells_u)}",
    "PARTIAL",
    "at chance (0.48-0.50 final, as the table at line 19 prints), not exactly 0.50" +
    ("" if all(flat_run(r) for r in su) else
     "; not every run is flat by the docstring's 0.1 rule: " + ", ".join(
         f"{s} cells {rng([v['acc'] for v in r['acc_by_cell'].values()])} around mean "
         f"{statistics.fmean(v['acc'] for v in r['acc_by_cell'].values()):.3f}"
         for s, r in sorted(sd.items()) if not bound(r) and not flat_run(r)) +
     " (cells hold 60-280 queries, so deviations of this size are within sampling noise of a 0.5 run)"))
swo = [r["write_order"] for r in sb]
check_line(R4, 71, "(1.00 / 1.00)")
row(R4, 71, "SINGLE_D bound runs write-order acc last / not-last (median)", "1.00 / 1.00",
    f"{f2(med([w['acc_last'] for w in swo]))} / {f2(med([w['acc_notlast'] for w in swo]))} "
    f"(min {min(w['acc_last'] for w in swo):.3f} / {min(w['acc_notlast'] for w in swo):.3f})",
    "MATCH" if (f2(med([w['acc_last'] for w in swo])), f2(med([w['acc_notlast'] for w in swo]))) == ("1.00", "1.00") else "DISAGREE")



# Plateaus 76-80
for line, a, rep in ((76, "BASELINE", "19 runs; 0.46-0.55"), (76, "REG3", "16 runs; 0.46-0.55")):
    u = [r for r in R[a].values() if not bound(r)]
    cmed = cell_medians(u)
    row(R4, line, f"{a} plateau: unbound runs; cell medians range", rep, f"{len(u)} runs; {rng(list(cmed.values()))}",
        "MATCH" if (len(u), rng(list(cmed.values()))) == (int(rep.split()[0]), "0.46-0.55") or
        (len(u) == int(rep.split()[0]) and min(cmed.values()) >= 0.455 and max(cmed.values()) < 0.555) else "DISAGREE",
        "combined range over both arms is the report's 0.46-0.55; each arm's own range shown")
check_line(R4, 76, "Hebbian BASELINE (19 runs) and REG3 (16): 0.46-0.55 in all 12 cells")
for a, rep in (("BASELINE_D", "6; 0.83-0.87"), ("REG3_D", "12; 0.78-0.83"), ("RESGATE_ONLY_D", "5; 0.72-0.78"), ("LATCH3_D", "9; 0.75-0.79")):
    check_line(R4, 78, a)
    u = [r for r in R[a].values() if not bound(r)]
    cmed = cell_medians(u)
    row(R4, 78, f"{a} plateau: unbound runs; cell medians range", rep, f"{len(u)}; {rng(list(cmed.values()))}")
for line, arms_, rep_ in ((76, ("BASELINE", "REG3"), "flat"), (78, ("BASELINE_D", "REG3_D", "RESGATE_ONLY_D", "LATCH3_D"), "flat")):
    parts, allf = [], True
    for a in arms_:
        u = [r for r in R[a].values() if not bound(r)]
        cmed = cell_medians(u)
        mf = max(abs(x - statistics.fmean(cmed.values())) for x in cmed.values()) <= 0.1
        nf = sum(flat_run(r) for r in u)
        allf = allf and mf
        parts.append(f"{a} median cells {'flat' if mf else 'not flat'}, runs flat {nf}/{len(u)}")
    row(R4, line + 1 if line == 78 else line, f"plateaus flat across cells ({', '.join(arms_)})", rep_, "; ".join(parts),
        "MATCH" if allf else "DISAGREE",
        "flat = every cell within 0.1 of the mean (explore_g_register3.py:21-22), applied to the per-cell medians the report "
        "prints and, for information, to each run")
c703 = R["LATCH3_NUDGE_D"][703]["acc_by_cell"]
s0 = [c703[k]["acc"] for k in sorted(c703) if k.startswith("0|")]
s1b0 = [c703[f"1|0|{i}"]["acc"] for i in range(3)]
s1b1 = [c703[f"1|1|{i}"]["acc"] for i in range(3)]
check_line(R4, 80, "(0.82 / 0.49 / 0.27, and 0.77 / 0.50 / 0.23)")
row(R4, 79, "LATCH3_NUDGE_D 703: stream 0 cells", "perfect", rng(s0), "MATCH" if min(s0) >= 0.995 else "DISAGREE")
row(R4, 80, "LATCH3_NUDGE_D 703: stream 1 by pair index (block 0; block 1)", "0.82 / 0.49 / 0.27; 0.77 / 0.50 / 0.23",
    f"{' / '.join(f2(x) for x in s1b0)}; {' / '.join(f2(x) for x in s1b1)}")

# Diagnostics 84-93


def beta_med_step(a, step):
    zs = [z_at(r, step) for r in R[a].values()]
    zs = [z for z in zs if z]
    return "/".join(f2(med([z[k] for z in zs])) for k in ("CTX", "KEY", "VAL")), len(zs)


def beta_med_end(a):
    return "/".join(f2(med([r["end"]["beta"][k] for r in R[a].values()])) for k in ("CTX", "KEY", "VAL"))


check_line(R4, 84, "0.10 / 0.10 / 0.10 (0), 0.25 / 0.23 / 0.23 (1200), 0.29 / 0.25 / 0.24 (end)")
for line, a, step, rep in ((84, "REG3_D", 0, "0.10/0.10/0.10"), (84, "REG3_D", 1200, "0.25/0.23/0.23"), (84, "REG3_D", "end", "0.29/0.25/0.24"),
                           (85, "REG3", 1200, "0.24/0.23/0.23"), (85, "REG3", "end", "0.26/0.28/0.21"),
                           (85, "REG3_NUDGE", 0, "0.34/0.21/0.23"), (86, "REG3_NUDGE", 2400, "0.47/0.19/0.23"),
                           (86, "REG3_NUDGE", "end", "0.67/0.18/0.21")):
    if step == "end":
        rec, nn = beta_med_end(a), len(R[a])
    else:
        rec, nn = beta_med_step(a, step)
    row(R4, line, f"{a} beta^R CTX/KEY/VAL median at {step}", rep, rec, note=f"{nn} runs" + (
        "; runs stop early when bound, so a step median covers only the runs still training" if step not in (0, "end") else ""))
zm0 = beta_med_step("LATCH3_D", 0)[0]
check_line(R4, 86, "LATCH3_D's z: 0.50 at 0, then 1.00 / 1.00 / 1.00 from 1200")
row(R4, 86, "LATCH3_D z median CTX/KEY/VAL at 0", "0.50 (at 0)", zm0,
    "MATCH" if zm0 == "0.50/0.50/0.50" else "PARTIAL",
    "the report gives one number for update 0; the per-role medians differ")
ok_from = all(beta_med_step("LATCH3_D", st)[0] == "1.00/1.00/1.00" for st in range(1200, 24001, 1200) if beta_med_step("LATCH3_D", st)[1])
row(R4, 86, "LATCH3_D z median 1.00/1.00/1.00 at every step from 1200", "1.00/1.00/1.00 from 1200",
    "1.00/1.00/1.00 at every step" if ok_from else "not at every step", "MATCH" if ok_from else "DISAGREE")


def lmed(a, l):
    return med([r["end"]["layers"][f"L{l}"]["margin"] for r in R[a].values()])


check_line(R4, 87, "nudged arms 1.00 / 1.00 / 1.00")
for a in ("REG3_NUDGE", "REG3_NUDGE_D", "LATCH3_NUDGE_D"):
    row(R4, 87, f"{a} routing margin at VAL L1/L2/L3 (median)", "1.00 / 1.00 / 1.00", " / ".join(f2(lmed(a, l)) for l in (1, 2, 3)))
for a in ("REG3", "REG3_D"):
    row(R4, 87, f"{a} routing margin at VAL L1/L2/L3 (median)", "0.00 / 0.00 / 0.00", " / ".join(f2(abs(lmed(a, l))) for l in (1, 2, 3)))
    rr = [r for r in R[a].values() if routed_bound(r)]
    row(R4, 88, f"{a} margin at VAL in the routed runs (every layer)", "1.00",
        rng([r["end"]["layers"][f"L{l}"]["margin"] for r in rr for l in (1, 2, 3)]),
        "MATCH" if all(round(r["end"]["layers"][f"L{l}"]["margin"], 2) == 1.0 for r in rr for l in (1, 2, 3))
        else "PARTIAL" if all(round(r["end"]["layers"][f"L{l}"]["margin"], 2) >= 0.99 for r in rr for l in (1, 2, 3)) else "DISAGREE",
        "rounding: " + ", ".join(f"{s} L{l} {r['end']['layers'][f'L{l}']['margin']:.3f}" for s, r in sorted(R[a].items())
                                 if routed_bound(r) for l in (1, 2, 3) if round(r["end"]["layers"][f"L{l}"]["margin"], 2) < 1.0))
row(R4, 88, "BASELINE_D routing margin at VAL (median), every layer", "0.45", " / ".join(f2(lmed("BASELINE_D", l)) for l in (1, 2, 3)),
    "MATCH" if all(f2(lmed("BASELINE_D", l)) == "0.45" for l in (1, 2, 3)) else "DISAGREE")

check_line(R4, 89, "nudged arms KEY and VAL [[1, 0], [0, 1]] at every layer")
for a in ("REG3_NUDGE", "REG3_NUDGE_D", "LATCH3_NUDGE_D"):
    rr = list(R[a].values())
    tabs = {(l, role): table_med(rr, l, role) for l in (1, 2, 3) for role in ("KEY", "VAL", "CTX")}
    kv = [tabs[(l, role)] for l in (1, 2, 3) for role in ("KEY", "VAL")]
    diag = sum(1 for r in rr if r["end"]["layers"]["L3"]["table"]["VAL"][0][0] > 0.5)
    row(R4, 89, f"{a} contingency KEY/VAL (median g), every layer", "[[1, 0], [0, 1]]",
        "; ".join(f"L{l} {role} {tabs[(l, role)]}" for l in (1, 2, 3) for role in ("KEY", "VAL")),
        "MATCH" if all(t == [[1.0, 0.0], [0.0, 1.0]] for t in kv) else "DISAGREE",
        f"stream 0 -> channel 0 at L3 VAL in {diag}/{len(rr)} runs; the channel labels are arbitrary per run, so a "
        "median table over runs is [[1, 0], [0, 1]] only if (almost) every run uses the same assignment")
    ctx = [x for l in (1, 2, 3) for rw in tabs[(l, "CTX")] for x in rw]
    row(R4, 90, f"{a} contingency CTX (median g), every layer", "near 0.5", rng(ctx),
        "MATCH" if all(0.4 <= x <= 0.6 for x in ctx) else "PARTIAL",
        "'near 0.5' read as 0.40-0.60" + ("" if all(0.4 <= x <= 0.6 for x in ctx) else
                                           "; the Hebbian nudged arm's CTX cells lean 0.64 / 0.36, a lean rather than 0.5"))
rr = list(R["REG3"].values())
tabs = {(l, role): table_med(rr, l, role) for l in (1, 2, 3) for role in ("KEY", "VAL", "CTX")}
check_line(R4, 91, "[[0.5, 0.5], [0.5, 0.5]]")
row(R4, 91, "REG3 (unnudged) contingency KEY/VAL (median g)", "[[0.5, 0.5], [0.5, 0.5]]",
    "; ".join(f"L{l} {role} {tabs[(l, role)]}" for l in (1, 2, 3) for role in ("KEY", "VAL")),
    "MATCH" if all(abs(x - 0.5) <= 0.05 for l in (1, 2, 3) for role in ("KEY", "VAL") for rw in tabs[(l, role)] for x in rw) else "DISAGREE")
row(R4, 91, "REG3 (unnudged) contingency CTX (median g): leaning to one channel", "leaning to one channel",
    "; ".join(f"L{l} CTX {tabs[(l, 'CTX')]}" for l in (1, 2, 3)),
    "MATCH" if any(abs(x - 0.5) > 0.05 for l in (1, 2, 3) for rw in tabs[(l, "CTX")] for x in rw) else "DISAGREE")

check_line(R4, 92, "BASELINE 0.61 / 0.52; REG3 0.57 / 0.51; REG3_NUDGE 0.51 / 0.50; BASELINE_D 0.89 / 0.90")
for line, a, rep in ((92, "BASELINE", "0.61 / 0.52"), (92, "REG3", "0.57 / 0.51"), (92, "REG3_NUDGE", "0.51 / 0.50"),
                     (92, "BASELINE_D", "0.89 / 0.90"), (93, "REG3_D", "0.76 / 0.62"), (93, "SINGLE_D", "0.97 / 0.88")):
    row(R4, line, f"{a} probe L3 K / V (median)", rep, pkv(a, "L3"))

# CHECKs 97-109
c72, c73 = CK["s72"][0]["rows"], CK["s73"][0]["rows"]
check_line(R4, 97, "S73 37, S72 32")
row(R4, 97, "CHECK counts S73, S72 (all ok)", "37, 32 (all ok)",
    f"{len(c73)}, {len(c72)} ({'all ok' if all(v for _, v in c72 + c73) else 'FAIL'})",
    note=f"{len(CK['s73'])} checks_s73 and {len(CK['s72'])} checks_s72 file(s) at 93422cf")


def find(rows, pat):
    return [nm for nm, _ in rows if re.search(pat, nm)]


b0 = [re.search(r"max \|diff\| ([0-9.e+-]+)", nm).group(1) for nm in find(c72 + c73, r"REG3, delta with beta 0")]
check_line(R4, 106, "(1.2e-7)")
row(R4, 106, "beta = 0 delta equals Hebbian (REG3 max |diff|)", "1.2e-7", ", ".join(sorted(set(b0))),
    "MATCH" if all(abs(float(x) - 1.2e-7) < 0.05e-7 for x in b0) else "DISAGREE")
lean = []
for nm in find(c72 + c73, r"stream lean at layers 1-3"):
    lean += [float(x) for x in re.search(r"\[([0-9., ]+)\]", nm).group(1).split(",")]
check_line(R4, 108, "(stream lean 0.046-0.050)")
row(R4, 108, "nudge stream lean at layers 1-3 (S72 + S73)", "0.046-0.050", f"{min(lean):.3f}-{max(lean):.3f}")
u = [re.search(r"max \|u\| ([0-9.e+-]+)", nm).group(1) for nm in find(c73, r"dead-STE guard")]
check_line(R4, 109, "(|u| 1.5e-3)")
row(R4, 109, "LATCH3_D dead-STE guard |u|", "1.5e-3", u[0], "MATCH" if abs(float(u[0]) - 1.5e-3) < 0.05e-3 else "DISAGREE",
    "1.49e-03 rounds to 1.5e-3")

check_line(R4, 115, "S71 42.8 min, S73 144.8 min, S72 84.0 min on 4 workers (4.5 h)")
logs = {t: git("show", f"{G}:explore_out/G/{t}.log") for t in ("s71", "s72", "s73")}
mins = {t: float(re.search(rf"{t.upper()} runs done in ([0-9.]+) min", logs[t]).group(1)) for t in logs}
row(R4, 115, "wall times S71 / S73 / S72; total", "42.8 / 144.8 / 84.0 min; 4.5 h",
    f"{mins['s71']} / {mins['s73']} / {mins['s72']} min; {sum(mins.values()) / 60:.1f} h",
    note="from the screen logs ('runs done in'), not the run records")

# ── NOT RECORD-CHECKABLE (said in the reports, not in the records or logs at 93422cf) ───────────────────────
import hashlib  # noqa: E402


def sha1_at(ref, path):
    return hashlib.sha1(subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"))).hexdigest()


prov71 = ST["s71_layers"]["meta"]["provenance"]
shar = {n: h for n, h, _ in ST["s71_layers"]["meta"]["sha_rows"]}
check_line(R3, 3, "torch 2.14.0+cu130, 1 thread per run")
row(R3, 3, "CPU / torch / threads", "Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread",
    f"{prov71['cpu']}, torch {prov71['torch']}, {prov71['threads']} thread", "MATCH",
    "store meta.provenance (identical in the S72 and S73 stores: " +
    str(all(ST[n]['meta']['provenance']['torch'] == prov71['torch'] and ST[n]['meta']['provenance']['threads'] == 1
            for n in ST)) + ")")
h_tasks_G, h_tasks_F = shar["explore_f_tasks"], sha1_at("9a6c2bf", "explore_f_tasks.py")
check_line(R3, 4, "explore_f_tasks copied unchanged at 9a6c2bf")
row(R3, 4, "explore_f_tasks copied unchanged at 9a6c2bf (SHA-1 of the file)", "unchanged",
    "unchanged" if h_tasks_G == h_tasks_F else "changed", None,
    f"S71 store meta.sha_rows explore_f_tasks {h_tasks_G[:12]}; F's explore_f_tasks.py at 9a6c2bf {h_tasks_F[:12]}")
h_dm_G, h_dm_F = shar["explore_delta_mem"], sha1_at(FREF, "explore_delta_mem.py")
check_line(R3, 63, "explore_delta_mem.py is unchanged from F's current branch (same SHA-1)")
row(R3, 63, "explore_delta_mem.py same SHA-1 as F's branch", "same SHA-1", "same SHA-1" if h_dm_G == h_dm_F else "different",
    None, f"S71 store meta.sha_rows {h_dm_G[:12]}; F at {FREF} {h_dm_F[:12]}")
check_line(R4, 115, "explore_delta_mem.py is unchanged from F's")
row(R4, 115, "explore_delta_mem.py same SHA-1 as F's branch (S72/S73 stores)", "same SHA-1",
    "same SHA-1" if all({n_: h for n_, h, _ in ST[n]["meta"]["sha_rows"]}["explore_delta_mem"] == h_dm_F
                        for n in ("s72_register3", "s73_register3_delta")) else "different")
NRC = [
    (R4, 116, "the progress watcher hit the 2-hour limit twice and was restarted, with no effect on the runs", "no effect",
     "operational note; nothing in the records or logs at 93422cf can confirm or refute it (the 150 records are complete "
     "and none retried, which is consistent with it)"),
    (R3, 53, "X's arm A bit for bit (the repro check at every start)", "passed",
     "ec.repro_check() is asserted by the driver (explore_g_rhl_parent.py:189) but its result is not written to the "
     "checks_s7x files or the stores"),
]
for f, line, item, rep, why in NRC:
    row(f, line, item, rep, "—", "NOT RECORD-CHECKABLE", why)

# report text: readings committed before any run (lines that cite the commits)
check_line(R3, 5, "Spec and readings committed at 9114843 before any run")
check_line(R4, 5, "26c4ab1 before any S73 / S72 run")

# ── screen table: readings commits vs first run record ─────────────────────────────────────────────────────
SCREENS = [
    ("S71 layers", "explore_g_layers.py", '"layer 1 alone suffices"', "explore_out/G/s71_layers_results.json", R3),
    ("S72 register3", "explore_g_register3.py", '"the register carries the cue at all three layers"',
     "explore_out/G/s72_register3_results.json", R4),
    ("S73 register3_delta", "explore_g_register3_delta.py", '"the delta stack carries the cue"',
     "explore_out/G/s73_register3_delta_results.json", R4),
]


def first_run_commit(path):
    for line in git("log", "--reverse", "--format=%h %cI", BRANCH, "--", path).splitlines():
        h, d = line.split()
        try:
            st = load(h, path)
        except subprocess.CalledProcessError:
            continue
        if len(st.get("runs", {})) >= 1:
            return h, d, len(st["runs"])
    return None, None, 0


SCREEN_ROWS = []
for name, f, needle, store, rep in SCREENS:
    rc = git("log", "--format=%h %cI", f"-S{needle}", BRANCH, "--", f).split("\n")
    rc = [x for x in rc if x.strip()][-1].split()          # the earliest commit adding the text
    later = [x for x in git("log", "--format=%h", f"{rc[0]}..{G}", "--", f).split() if x]
    fh, fd, nrec = first_run_commit(store)
    prov = ST[os.path.basename(store)[:-len("_results.json")]]["meta"]["provenance"]
    SCREEN_ROWS.append((name, f, f"{rc[0]} ({rc[1]})", f"{fh} ({fd}; {nrec} record(s))",
                        "yes" if rc[1] < fd else "NO",
                        f"needle {needle}; screen file changed after the readings commit: {'no' if not later else ', '.join(later)}; "
                        f"store provenance git {prov['git']}, run started {prov['started']} (local clock)"))

for (name, f, rc, fr, pre, note), line, fname, rep_c in zip(SCREEN_ROWS, (5, 5, 5), (R3, R4, R4),
                                                             ("9114843", "26c4ab1", "26c4ab1")):
    row(fname, line, f"{name}: readings committed at {rep_c} before any run", f"{rep_c}, before any run",
        f"{rc.split()[0]}, {'before' if pre == 'yes' else 'NOT before'} the first record ({fr.split()[0]})",
        "MATCH" if rc.split()[0] == rep_c and pre == "yes" else "DISAGREE", f"readings {rc}; first record {fr}")

# ── readings stated in the reports that the docstrings did not pre-register ────────────────────────────────
UNREG = [
    (R4, "54-55", "pooled REG3 + REG3_D vs both baselines, 8 / 0 over 40 pairs", "the report labels it post hoc, not a reading"),
    (R4, "56-59", "'what separates a discovering run is the write' (beta^R at CTX rises in routed runs; key split first in 16 of 20)",
     "beta^R by step is ALSO REPORTED (register3.py:27, register3_delta.py:35) but no reading is attached to it"),
    (R4, "63-64", "'the residual gate alone does not read the delta memory's stream tag'",
     "the pre-fixed reading is only 'discovered (residual gate)' (register3_delta.py:27); the stronger negative claim is not pre-registered"),
    (R4, "65-69", "'on delta memory the stack can learn stream-tagged keys, and when it does, the gate does not need to route'",
     "register3_delta.py:30-32 fixes the 2x2 as descriptive; the causal reading is not pre-registered"),
    (R4, "70-72", "'the stream is in the key, not in the write order' (SINGLE_D)", "write_order_acc is printed as descriptive (register3_delta.py:33)"),
    (R4, "46-49", "'routing layer 1 was the missing piece on Hebbian memory'", "a cross-screen interpretation (S56 P = 8 vs S72 RHL); "
     "S71's 'layer 1 is the block' is the pre-registered reading behind it"),
    (R3, "31-33", "'explains S56's Hebbian REG_NUDGE plateau'", "interpretation across tasks (S56 is P = 8, S71 is RandHeaderTask-lite)"),
]

# ── key-split counting (the S56 issue) ──────────────────────────────────────────────────────────────────────
KS = []
for a in ("BASELINE_D", "REG3_D", "RESGATE_ONLY_D", "LATCH3_D", "LATCH3_NUDGE_D", "BASELINE", "REG3"):
    rr = R[a]
    KS.append((a, sum(1 for r in rr.values() if not bound(r) and fail_v1(r) == "KEY"),
               sum(1 for r in rr.values() if fail_v1(r) == "KEY"),
               sum(1 for r in rr.values() if not bound(r) and fail_v2(r) == "KEY"),
               sum(1 for r in rr.values() if fail_v2(r) == "KEY"), len(rr)))
V2SAME = all(classes(R[a]) == classes(R[a], fail_v2) for a in ARMS["s71_layers"] + ARMS["s72_register3"] + ARMS["s73_register3_delta"]
             if a != "SINGLE_D")
d5 = paired(R["REG3_D"], R["BASELINE_D"], discovered)
d4 = paired(R["REG3"], R["BASELINE"], discovered)
br5 = paired(R["REG3_D"], R["BASELINE_D"], bound_routed_mine)

# ── write the markdown ─────────────────────────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
rows = sorted(ROWS, key=lambda r: (ORDER[r[5]], r[0], r[1]))
cnt = Counter(r[5] for r in ROWS)


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


L = []
L.append("# Session G reports 3 and 4 — recomputation from the records (referee)")
L.append("")
L.append(f"Pinned commit: `{G}` (claude/explore-G head; report 4's commit, {git('log', '-1', '--format=%cI', G).strip()}). "
         f"Records: `explore_out/G/s71_layers_results.json`, `s72_register3_results.json`, `s73_register3_delta_results.json`, "
         f"the `checks_s7[123]_*.json` files and `s7[123].log` at `{G}`; S56/S57 stores at `{G}` for the cited P = 8 counts; "
         f"F's `explore_out/F/rh_lite_results.json` at `{FREF}` (the last claude/explore-F commit before `{G}`) for F's S70 references. "
         "Generated by `docs/review_2026-10/sessions/recompute_G.py` (stdlib only; records read with `git show`). "
         "Reported values are typed in from the report text with their line numbers (the script asserts the cited line contains them); "
         "none is used as an input.")
L.append("")
L.append(f"Coverage: report 3 covers S71 only; report 4 covers S73 and S72 (it cites S71 only for validity). "
         f"Rows: {len(ROWS)} — " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in ORDER) + ".")
L.append("")
L.append("## Outcome definitions used")
L.append("")
L.append("| outcome | definition | source (at 93422cf unless stated) |")
L.append("|---|---|---|")
for o, d, s in OUTCOME_DEFS:
    L.append(f"| {esc(o)} | {esc(d)} | {esc(s)} |")
L.append("")
L.append("Record integrity: " + "; ".join(
    f"{n}: {ok}/{tot} ok, {tot - att} retried, recorded transition = rlib.transition(curve) in {tot - len(mism)}/{tot}, "
    f"seeds as in meta: {sok}, acc_by_cell 12 cells / 2048 queries in every run: {cells}"
    for n, tot, ok, att, mism, sok, cells in INTEG) + ".")
L.append("")
L.append("## Rows (DISAGREE, then PARTIAL, NOT RECORD-CHECKABLE, MATCH)")
L.append("")
L.append("| report:line | item | reported (as printed) | recomputed | status | note |")
L.append("|---|---|---|---|---|---|")
for f, line, item, rep, rec, st, note in rows:
    L.append(f"| {os.path.basename(f)}:{line} | {esc(item)} | {esc(rep)} | {esc(rec)} | **{st}** | {esc(note)} |")
L.append("")
L.append("## Screens: readings committed vs first run record")
L.append("")
L.append("Readings commit = `git log --format='%h %cI' -S'<reading text>' origin/claude/explore-G -- <file>` (earliest). "
         "First run record = earliest commit on the branch whose results store holds >= 1 record.")
L.append("")
L.append("| screen | file | readings first committed | first run record | readings precede first run? | note |")
L.append("|---|---|---|---|---|---|")
for s in SCREEN_ROWS:
    L.append("| " + " | ".join(esc(x) for x in s) + " |")
L.append("")
L.append("Every reading printed in the reports' Readings tables (report_3.md:26-29; report_4.md:34-42) is one the docstrings fixed "
         "(explore_g_layers.py:21-26; explore_g_register3.py:18-25; explore_g_register3_delta.py:24-33), with the docstring's "
         "outcome (BOUND for S71, ROUTED-BOUND for S72/S73) and test. S72's R1 follow-up (register3.py:20-23) was not triggered (R1 is "
         "'carries'). The screen files are unchanged between their readings commit and 93422cf.")
L.append("")
L.append("### Statements in the reports that the docstrings did not pre-register")
L.append("")
L.append("| report:lines | statement | why it is not a pre-registered reading |")
L.append("|---|---|---|")
for f, ln, s, why in UNREG:
    L.append(f"| {os.path.basename(f)}:{ln} | {esc(s)} | {esc(why)} |")
L.append("")
L.append("## Key-split counting (the S56 R2 issue) — can any reading be read two ways?")
L.append("")
L.append("S56's delta R2 rule counted KEY failures and only among unbound runs. S72 and S73 do not: every pre-fixed "
         "'discovered' reading is a one-sided McNemar on ROUTED-BOUND (explore_g_register3.py:18, :25; "
         "explore_g_register3_delta.py:24, :26-28), and no reading has a KEY-count condition. The failure classes are printed "
         "for unbound runs only (count_table, explore_g_rhl_parent.py:105-109), with v1. So no pre-fixed S72/S73 reading depends "
         f"on how key splits are counted. v1 vs v2 {'changes no printed class count in any S71-S73 arm' if V2SAME else 'CHANGES a printed class count'}. "
         "The counts that do depend on the population are descriptive:")
L.append("")
L.append("| arm | KEY v1, unbound only (printed) | KEY v1, all runs | KEY v2, unbound only | KEY v2, all runs | n |")
L.append("|---|---|---|---|---|---|")
for a, a1, a2, a3, a4, n in KS:
    L.append(f"| {a} | {a1} | {a2} | {a3} | {a4} | {n} |")
L.append("")
L.append("One reading whose wording allows a second reading, and descriptive key-split statements that depend on the population:")
L.append("")
L.append(f"1. S73 'discovered (register)' is fixed on ROUTED-BOUND (register3_delta.py:24, :26): 4 / 0, p = {mcnemar1(4, 0):.4g}, "
         f"'does not apply'. The report's own DISCOVERED column (rhl_parent.py:10, BOUND and VAL cos < 0.5) gives REG3_D vs "
         f"BASELINE_D {d5[0]} / {d5[1]}, p = {d5[2]:.4g} (< 0.05; REG3_D 714 is DISCOVERED but not ROUTED*@end, margin 0.66). "
         f"Read with the outcome that shares the reading's name, 'discovered (register)' would apply. The docstring is explicit, "
         f"so the pre-registered answer is 'does not apply'; the difference should be stated. On S72, DISCOVERED gives "
         f"{d4[0]} / {d4[1]}, p = {d4[2]:.4g}, the same as ROUTED-BOUND. With 'my BOUND ROUTED' (no margin guard), REG3_D vs "
         f"BASELINE_D is {br5[0]} / {br5[1]}, p = {br5[2]:.3g}.")
L.append(f"2. The descriptive key-split statements depend on the population: report_4.md:59 '16 of 20 seeds ... key split' "
         f"(no population gives 16), :62 'LATCH3_D key-splits 9/10' (10/10 counting the bound run 704), :64 'the bound runs are "
         f"key-split bindings' (3 of RESGATE_ONLY_D's 5 bound runs), and the table's 'KEY 6' for BASELINE_D (9 counting the "
         f"three bound key-split runs 701, 710, 719). None of them enters a pre-fixed reading.")
L.append("")
with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")

print(f"wrote {OUT}: {len(ROWS)} rows; " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in ORDER))
for f, line, item, rep, rec, st, note in rows:
    if st in ("DISAGREE", "PARTIAL"):
        print(f"  {st}: {os.path.basename(f)}:{line} {item}: reported {rep!r} recomputed {rec!r}")
