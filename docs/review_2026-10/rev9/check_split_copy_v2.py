"""check_split_copy_v2.py — Referee V, second pass: the SECOND version of the Revision 9 draft
(origin/claude/rev9-draft @ f67bc07, multichannel_hebbian_report_v9.tex) against test_split_copy's records on X.

A copy of check_split_copy.py (first pass, draft 9bfa85a) made to run against the second draft:
  * every first-pass row's quoted fragment is checked at its old line of 9bfa85a (tex:NNN) and then located in f67bc07
    by text search (tex2:NNN, or REMOVED);
  * the record and code checks are kept, recomputed from the same pinned records;
  * one row for every new or changed test_split_copy sentence in f67bc07 (each fragment located by text search).

Inputs, all read with git at pinned refs (nothing is run, nothing is written but stdout):
  draft   9bfa85a (first) and f67bc07 (second): multichannel_hebbian_report_v9.tex
  records 592c574:results/X/split_copy_results.json   (unchanged at the main line's head; L file looked for on every ref)
          HEAD:results/{X,L}/window_gate_results.json  (the window_gate control and the all-position count, tex2:190/195/436)
  test    592c574:test_split_copy.py (docstring CLAIMS / READINGS / RESULT), spec fb3acb7:specs/test_split_copy.md
  code    HEAD:test_window_gate.py (targeting, reset_op, make_trigger)
Outcomes from rlib (re-implemented from the report's Sec. 3), cross-checked against the records' own fields.

    python3 docs/review_2026-10/rev9/check_split_copy_v2.py > docs/review_2026-10/rev9/check_split_copy_v2.md
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402

OLD_REF, NEW_REF, DRAFT = "9bfa85a", "f67bc07", "multichannel_hebbian_report_v9.tex"
HEAD = "origin/claude/bdh-growth-hebbian-inference-w90069"
RES, REC = "592c574", "results/X/split_copy_results.json"
SPEC_C, TEST_C = "fb3acb7", "1f52cd5"
ARMS = ("CEIL_C", "CEIL_DELTA_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8")
PART_A = ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")

tex1 = rlib.git("show", f"{OLD_REF}:{DRAFT}").split("\n")
tex2 = rlib.git("show", f"{NEW_REF}:{DRAFT}").split("\n")
doc = rlib.git("show", f"{RES}:test_split_copy.py")
spec = rlib.git("show", f"{SPEC_C}:specs/test_split_copy.md")
twg = rlib.git("show", f"{HEAD}:test_window_gate.py")
store = rlib.load(RES, REC)
R = {a: rlib.runs(store, a) for a in ARMS}
HEAD_SHORT = rlib.git("rev-parse", "--short", HEAD).strip()


def locate(lines, phrases):
    """First 1-based line holding every phrase, or None."""
    for i, ln in enumerate(lines):
        if all(p in ln for p in phrases):
            return i + 1
    return None


def code_line(src, snippet):
    for i, ln in enumerate(src.split("\n")):
        if snippet in ln:
            return i + 1
    raise AssertionError(snippet)


# ── the records (as in the first pass) ───────────────────────────────────────
def ok_all():
    return all(rlib.ok(r) for a in ARMS for r in R[a].values())


def succ(a, r):
    return rlib.bound_routed(r) if a in PART_A else rlib.bound(r)


def succ_rec(a, r):
    if r["transition"] is None:
        return False
    return bool(r["end"]["one_to_one"] and min(r["end"]["stream_acc"]) >= 0.9) if a in PART_A else True


S = {a: sorted(R[a]) for a in ARMS}
WIN = {a: [s for s in S[a] if succ(a, R[a][s])] for a in ARMS}
agree_rec = all(succ(a, R[a][s]) == succ_rec(a, R[a][s]) for a in ARMS for s in S[a])
agree_t = all(rlib.transition(R[a][s]["curve"]) == R[a][s]["transition"] for a in ARMS for s in S[a])
n_runs = sum(len(R[a]) for a in ARMS)
cpus = {r["cpu"] for a in ARMS for r in R[a].values()}
meta = store["meta"]


def pair(a, b, win=WIN, seeds=S):
    ss = sorted(set(seeds[a]) & set(seeds[b]))
    x = sum(1 for s in ss if s in win[a] and s not in win[b])
    y = sum(1 for s in ss if s in win[b] and s not in win[a])
    return x, y, len(ss)


c1 = pair("WIN3_SPLIT_D8", "WIN3_RESET_D8")
c2 = pair("WIN3_RESET_D8", "WIN3_SLOW_D8")
c3 = pair("WIN3_SPLIT_D8", "SINGLE_DELTA_D8")
p1, p3 = rlib.mcnemar1(c1[0], c1[1]), rlib.mcnemar1(c3[0], c3[1])
d2 = c2[0] - c2[1]
c1_state = "SHOWN" if p1 < 0.05 else "not shown"
c2_state = "HOLDS" if d2 <= 2 else "FAILS"
c3_state = "SHOWN" if p3 < 0.05 else "not shown"
reset_sub_slow = set(WIN["WIN3_RESET_D8"]) <= set(WIN["WIN3_SLOW_D8"])


def ceil_t(a):
    return [R[a][s]["transition"] for s in S[a]]


def fired_in(runs):
    return [(s, c) for s in sorted(runs) for c in runs[s].get("checks", []) if c["fired"]]


def targeting(cmap, cs, c0):  # test_window_gate.targeting, re-implemented
    return cmap.count(cs) >= 2 and cmap.count(c0) == 0


assert "return cmap.count(cs) >= 2 and cmap.count(c0) == 0" in twg
F_SPLIT, F_RESET = fired_in(R["WIN3_SPLIT_D8"]), fired_in(R["WIN3_RESET_D8"])
ok_split = sum(targeting(c["map_before"], c["key_cs"], c["key_c0"]) for _, c in F_SPLIT)
ok_reset = sum(targeting(c["map_before"], c["key_cs"], c["key_c0"]) for _, c in F_RESET)
rec_ok_split = sum(bool(c["key_ok"]) for _, c in F_SPLIT)
rec_ok_reset = sum(bool(c["key_ok"]) for _, c in F_RESET)
ops = {a: {R[a][s].get("trigger_op") for s in S[a]} for a in ("WIN3_SPLIT_D8", "WIN3_RESET_D8")}
per_run = {a: [sum(1 for c in R[a][s].get("checks", []) if c["fired"]) for s in S[a]]
           for a in ("WIN3_SPLIT_D8", "WIN3_RESET_D8")}
acc_d = [R["SINGLE_DELTA_D8"][s]["acc"] for s in S["SINGLE_DELTA_D8"]]
acc_h = [R["SINGLE_HEBB_D8"][s]["acc"] for s in S["SINGLE_HEBB_D8"]]
tr_split = [R["WIN3_SPLIT_D8"][s]["transition"] for s in WIN["WIN3_SPLIT_D8"]]
fail_oc = {a: [rlib.outcome_k(R[a][s]) for s in S[a] if s not in WIN[a]] for a in ("WIN3_RESET_D8", "WIN3_SLOW_D8")}
# the seven seeds C1 rescues (split bound routed, reset not): the reset arm's outcome there
c1_seeds = [s for s in S["WIN3_SPLIT_D8"] if s in WIN["WIN3_SPLIT_D8"] and s not in WIN["WIN3_RESET_D8"]]
c1_reset_oc = {s: rlib.outcome_k(R["WIN3_RESET_D8"][s]) for s in c1_seeds}

# the reset uses no target: test_window_gate.reset_op's body
reset_src = twg[twg.index("def reset_op(m, o):"):twg.index("def make_trigger(")]
reset_ignores_target = "cs" not in reset_src.replace("state_zeroed", "") and "c0" not in reset_src
L_REC = "results/L/split_copy_results.json"
refs = [r.strip() for r in rlib.git("branch", "-r").split("\n") if r.strip() and "->" not in r]
l_file_on = [r for r in refs if L_REC in rlib.git("ls-tree", "-r", "--name-only", r).split("\n")]
l_ever = rlib.git("log", "--all", "--format=%h", "--", L_REC).split()


# window_gate (main line head): the reset-only control and the all-position count
WG = {m: rlib.load(HEAD, f"results/{m}/window_gate_results.json") for m in "XL"}
WGR = {(m, a): rlib.runs(WG[m], a) for m in "XL" for a in ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")}
ctl_seeds = {m: sorted(WGR[(m, "WIN3_RESET_D8")]) for m in "XL"}
ctl_reset = [(m, s) for m in "XL" for s in ctl_seeds[m] if rlib.bound_routed(WGR[(m, "WIN3_RESET_D8")][s])]
ctl_none = [(m, s) for m in "XL" for s in ctl_seeds[m] if rlib.bound_routed(WGR[(m, "WIN3_SLOW_D8")][s])]
ctl_b = sum(1 for x in ctl_reset if x not in ctl_none)
ctl_c = sum(1 for x in ctl_none if x not in ctl_reset)
ctl_f = [c for m in "XL" for s in ctl_seeds[m] for c in WGR[(m, "WIN3_RESET_D8")][s]["checks"] if c["fired"]]
ctl_lab = sum(targeting(c["map_before"], c["key_cs"], c["key_c0"]) for c in ctl_f)
ctl_lab_rec = sum(bool(c["key_ok"]) for c in ctl_f)
ctl_nofire = [f"{m} s{s}" for m in "XL" for s in ctl_seeds[m] if not any(c["fired"] for c in WGR[(m, "WIN3_RESET_D8")][s]["checks"])]
ctl_runs_fired = sum(1 for m in "XL" for s in ctl_seeds[m] if any(c["fired"] for c in WGR[(m, "WIN3_RESET_D8")][s]["checks"]))

ALLPOS = []   # (source, runs, firings, all-position labelled (recomputed), recorded all_ok, key labelled, coincide)
every_row_has_all = True
for name, runs in (("split_copy X", R["WIN3_SPLIT_D8"]), ("window_gate X", WGR[("X", "WIN3_SPLIT_D8")]),
                   ("window_gate L", WGR[("L", "WIN3_SPLIT_D8")])):
    rows_ = [c for r in runs.values() for c in r["checks"]]
    every_row_has_all &= all({"all_cs", "all_c0", "key_cs", "key_c0", "all_masses", "key_masses"} <= set(c) for c in rows_)
    F = [c for r in runs.values() for c in r["checks"] if c["fired"]]
    ALLPOS.append((name, len(runs), len(F), sum(targeting(c["map_before"], c["all_cs"], c["all_c0"]) for c in F),
                   sum(bool(c["all_ok"]) for c in F), sum(targeting(c["map_before"], c["key_cs"], c["key_c0"]) for c in F),
                   sum((c["all_cs"], c["all_c0"]) == (c["key_cs"], c["key_c0"]) for c in F)))
AP = [sum(x[i] for x in ALLPOS) for i in range(1, 7)]


# commits
def cinfo(c):
    return rlib.git("show", "-s", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M:%S", c).strip()


def ctime(c):
    return dt.datetime.strptime(rlib.git("show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M:%S", c).strip(),
                                "%Y-%m-%d %H:%M:%S")


def first_add(path):
    return rlib.git("log", "--diff-filter=A", "--format=%h", HEAD, "--", path).split()


def ancestor(a, b):
    try:
        rlib.git("merge-base", "--is-ancestor", a, b)
        return True
    except Exception:
        return False


spec_add, test_add, rec_add = first_add("specs/test_split_copy.md"), first_add("test_split_copy.py"), first_add(REC)
spec_changed = rlib.git("log", "--format=%h", f"{SPEC_C}..{HEAD}", "--", "specs/test_split_copy.md").split()
py_changed = [f for f in rlib.git("diff", "--name-only", TEST_C, RES, "--", "*.py").split() if not f.startswith("docs/")]
res_py = rlib.git("diff", "--numstat", f"{RES}^", RES, "--", "test_split_copy.py").split()
res_diff = rlib.git("diff", "-U0", f"{RES}^", RES, "--", "test_split_copy.py")
res_doc_only = all(not l.startswith("-") or l.startswith("---") for l in res_diff.split("\n"))
rec_unchanged = rlib.git("diff", "--stat", RES, HEAD, "--", REC, "test_split_copy.py").strip() == ""
fc2_same = rlib.git("diff", "--stat", "ba8901a", TEST_C, "--", "fail_class_v2.py").strip() == ""
lr_diff = rlib.git("diff", "--numstat", "56c44a7", TEST_C, "--", "longrun.py").split()
imports_fc2 = "import fail_class_v2 as fc2" in doc
uses_lr = "import longrun" in doc
t_spec, t_test, t_res = ctime(SPEC_C), ctime(TEST_C), ctime(RES)
started = dt.datetime.strptime(meta["started"], "%Y-%m-%d %H:%M:%S")
secs = sum(r["secs"] for a in ARMS for r in R[a].values())
span_h = (t_res - t_test).total_seconds() / 3600
run_lb_h = secs / meta["workers"] / 3600
run_ub_h = (t_res - started).total_seconds() / 3600
doc_130 = "CHECK 130 (Muon, X's 2.80GHz" in doc and "differs here and was printed, not asserted" in doc
doc_146 = "(printed: other CPU)" in doc


def fs(xs):
    return "{" + ", ".join(f"s{x}" for x in xs) + "}"


def st(cond):
    return "MATCH" if cond else "DISAGREE"


# ── first-pass rows: (old line, item, old status, the quoted fragment) ───────
OLD = [
    (36, "abstract: C1 counts", "MATCH", ("copy and reset 10/10 against the reset alone 3/10, 7 vs 0",)),
    (36, "abstract: reset vs no trigger", "MATCH", ("the reset alone no better than no trigger",)),
    (36, "abstract: C3 counts", "MATCH", ("binds eight streams $0/10$ where the partition binds $10/10$",)),
    (36, "abstract: framing 'the row copy is the ingredient'", "PARTIAL",
     ("pre-registered result on one machine", "the split's row copy is the ingredient")),
    (41, "Rev. 9 paragraph", "MATCH", ("complete on X and running on L",)),
    (53, "index row", "MATCH", ("on X: C1, C3 SHOWN, C2 HOLDS; pooled readings wait for L",)),
    (115, "classifier commit", "MATCH", ("ba8901a",)),
    (178, "'settle the question'", "PARTIAL", ("Two further looks settle the question, one of them pre-registered",)),
    (192, "table row: arms and seeds", "MATCH", ("560--569; live trigger; pre-registered",)),
    (192, "table row: counts", "MATCH", ("$\\mathbf{10/10}$ & --- & $\\mathbf{3/10}$ & $5/10$",)),
    (192, "table row: C1", "MATCH", ("C1 7 vs 0, $p = 0.0078$ SHOWN",)),
    (192, "table row: C2", "MATCH", ("C2 reset vs none 0 vs 2, HOLDS",)),
    (195, "caption: reset-only arm fired under the live rule", "MATCH", ("fired under the live rule",)),
    (195, "caption: targets labelled", "MATCH", ("on targets labelled on target where labelling was recorded",)),
    (195, "caption: reset bound no more often than none", "MATCH", ("bound no more often than the arm with no trigger",)),
    (195, "caption: L half", "NOT RECORD-CHECKABLE", ("L half is running",)),
    (199, "specified before written", "MATCH", ("Specified (\\texttt{specs/test\\_split\\_copy.md}) before it was written",)),
    (199, "claims structure", "MATCH", ("three claims pooled primary and per machine secondary",)),
    (199, "seeds and run count", "MATCH", ("seeds 560--569, 54 runs",)),
    (199, "every CHECK passed", "NOT RECORD-CHECKABLE", ("every CHECK passed, 14.7 h",)),
    (199, "wall clock", "NOT RECORD-CHECKABLE", ("every CHECK passed, 14.7 h)",)),
    (199, "C1 text", "MATCH", ("C1, the split beats the reset alone, SHOWN, $10/10$ against $3/10$, 7 vs 0, $p = 0.0078$",)),
    (199, "C2 text", "MATCH",
     ("C2, the reset alone is not better than no trigger by more than two pairs, HOLDS, $3/10$ against $5/10$, 0 vs 2",)),
    (199, "reset seeds subset of none seeds", "MATCH", ("every seed the reset bound also bound without it",)),
    (199, "splits and resets on labelled targets", "MATCH", ("All 22 splits and all 27 resets were on correctly labelled targets",)),
    (199, "'so the difference is the row copy, not the aim'", "PARTIAL", ("so the difference is the row copy, not the aim",)),
    (199, "validity", "MATCH", ("the Hebbian perfect gate at 4800 and 3600; the delta perfect gate at 1200 and 1200",)),
    (199, "reading 1 quoted", "MATCH", ("``the copy, not the reset'' if C1 is SHOWN pooled and C2 HOLDS",)),
    (199, "per-machine reading", "PARTIAL", ("per machine the reading is met on X",)),
    (242, "C3", "MATCH", ("is SHOWN on X: the recipe $10/10$ against one delta channel $0/10$ (10 vs 0, $p = 0.00098$",)),
    (242, "single delta final accuracy", "MATCH", ("final accuracy $0.08$--$0.76$",)),
    (242, "one Hebbian channel", "MATCH", ("one Hebbian channel $0/10$, the delta",)),
    (242, "delta perfect gate", "MATCH", ("the delta perfect gate $2/2$ at 1200",)),
    (242, "inference from validity", "PARTIAL", ("so that the channel's failure is the missing partition and not the delta rule",)),
    (242, "reading 2 quoted", "MATCH", ("the partition is necessary at eight streams against a single delta channel",)),
    (280, "synthesis: S=8 split column", "MATCH", ("X $10/10^{s}$",)),
    (280, "synthesis: single channels", "MATCH", ("$0/10$ (X) & $0/10$ (X)",)),
    (283, "synthesis caption: X half", "PARTIAL", ("with \\texttt{test\\_split\\_copy}'s X half",)),
    (347, "assessment: copy not reset", "MATCH", ("Pre-registered and shown on one machine",)),
    (351, "assessment: number of contexts", "MATCH", ("$0/10$ in the pre-registered test on X (C3)",)),
    (371, "assessment: row copy breaks merges", "PARTIAL", ("on X, pre-registered",)),
    (459, "conclusion", "PARTIAL", ("The split's row copy is the ingredient",)),
    (482, "provenance commits", "MATCH", ("spec \\texttt{fb3acb7} / \\texttt{1f52cd5} / \\texttt{592c574}",)),
    (482, "classifier / launcher", "MATCH", ("the launcher \\texttt{56c44a7}",)),
    (482, "runs X; L", "MATCH", ("running & 54; ---",)),
    (486, "X CPU for split_copy", "MATCH", ("2.10\\,GHz for \\texttt{split\\_copy}",)),
]
for n, item, _, ph in OLD:
    assert all(p in tex1[n - 1] for p in ph), (n, item, ph)

# ── rows against the second draft ────────────────────────────────────────────
rows = []   # (tex2 line, item, draft, recomputed, status, note, old indices)


def row(phrases, item, draft, recomputed, status, note="", old=()):
    n = locate(tex2, phrases)
    assert n is not None, (item, phrases)
    rows.append((n, item, draft, recomputed, status, note, tuple(old)))


nA = len(S["WIN3_SPLIT_D8"])
nW = {a: len(WIN[a]) for a in ARMS}
# abstract and front matter
row(("the copy with its reset binds eight streams $10/10$ against the reset alone $3/10$ (7 vs 0)",),
    "abstract: C1 counts", "the copy with its reset binds eight streams 10/10 against the reset alone 3/10 (7 vs 0)",
    f"WIN3_SPLIT_D8 BR {nW['WIN3_SPLIT_D8']}/{nA}; WIN3_RESET_D8 BR {nW['WIN3_RESET_D8']}/{nA}; discordant {c1[0]} vs {c1[1]}",
    st((nW['WIN3_SPLIT_D8'], nW['WIN3_RESET_D8'], c1[0], c1[1]) == (10, 3, 7, 0)), "", old=(0,))
row(("the reset alone is within the bound of no trigger ($3/10$ against $5/10$)",),
    "abstract: reset vs no trigger", "the reset alone is within the bound of no trigger (3/10 against 5/10)",
    f"RESET BR {nW['WIN3_RESET_D8']}/10 vs SLOW {nW['WIN3_SLOW_D8']}/10, {c2[0]} vs {c2[1]}, d = {d2} <= 2 -> {c2_state}",
    st(c2_state == "HOLDS" and (nW['WIN3_RESET_D8'], nW['WIN3_SLOW_D8']) == (3, 5)),
    "Now C2's own wording (a bound), not 'no better than'.", old=(1,))
row(("one delta-rule channel binds $0/10$ where the recipe binds $10/10$",), "abstract: C3 counts",
    "one delta-rule channel binds 0/10 where the recipe binds 10/10",
    f"SINGLE_DELTA_D8 BOUND {nW['SINGLE_DELTA_D8']}/10; WIN3_SPLIT_D8 BR {nW['WIN3_SPLIT_D8']}/10; {c3[0]} vs {c3[1]}",
    st(nW['SINGLE_DELTA_D8'] == 0 and nW['WIN3_SPLIT_D8'] == 10), "'the partition' is now 'the recipe'.", old=(2,))
row(("now pre-registered and, on one machine, shown as that machine's secondary claims",
     "the pooled readings wait for it"), "abstract: scope of the result",
    "the second claim of Revision 8 is now pre-registered and, on one machine, shown as that machine's secondary claims "
    "(the other machine's half has not run, and the pooled readings wait for it)",
    f"per machine (SECONDARY) on X: C1 {c1_state}, C2 {c2_state}, C3 {c3_state}; readings pooled only, not evaluated; "
    f"L file at any ref: {l_file_on or 'none'} (ever committed: {l_ever or 'never'})",
    st((c1_state, c2_state, c3_state) == ("SHOWN", "HOLDS", "SHOWN") and not l_file_on),
    "Scoped: per machine, secondary, readings pending. 'The row copy is the ingredient' is gone from the abstract.",
    old=(3,))
row(("\\texttt{test\\_split\\_copy}, complete on X; L's half has not run",), "Rev. 9 paragraph",
    "test_split_copy, complete on X; L's half has not run (and again: 'L's half of test_split_copy has not run')",
    f"X file: {n_runs} records, all ok = {ok_all()}; no L file on any ref",
    st(n_runs == 54 and ok_all() and not l_file_on),
    "'has not run' is consistent with the refs (no L file ever committed) but not provable from git; it contradicts the "
    "first draft's 'running on L' and this draft's own tex2:488 ('running'), see that row.", old=(4,))
row(("on X (secondary): C1, C3 SHOWN, C2 HOLDS; pooled claims and readings wait for L",), "index row",
    "split_copy (X; L to run): on X (secondary): C1, C3 SHOWN, C2 HOLDS; pooled claims and readings wait for L",
    f"C1 {c1_state} (p={p1:.4f}); C3 {c3_state} (p={p3:.5f}); C2 {c2_state} (d={d2})",
    st((c1_state, c2_state, c3_state) == ("SHOWN", "HOLDS", "SHOWN")), "", old=(5,))
row(("\\texttt{fail\\_class\\_v2} (commit \\texttt{ba8901a})", "used by every test from \\texttt{test\\_split\\_copy} on"),
    "classifier commit", "fail_class_v2 (commit ba8901a) ... used by every test from test_split_copy on",
    f"{cinfo('ba8901a')}; test_split_copy imports fail_class_v2: {imports_fc2}; fail_class_v2.py unchanged "
    f"ba8901a..{TEST_C}: {fc2_same}; ancestor of {TEST_C}: {ancestor('ba8901a', TEST_C)}",
    st(imports_fc2 and fc2_same and ancestor('ba8901a', TEST_C)), "", old=(6,))
# Section split
row(("Two further looks address it, one of them pre-registered and complete on one machine",), "'address it' (was 'settle')",
    "Two further looks address it, one of them pre-registered and complete on one machine.",
    f"X complete ({n_runs} records, all ok); L none", st(n_runs == 54 and ok_all()),
    "'settle the question' is gone.", old=(7,))
row(("560--569; live trigger; pre-registered",), "table row: arms and seeds",
    "split_copy, X; S=8, k=16, 560--569; live trigger; pre-registered",
    f"seeds {S['WIN3_SPLIT_D8'][0]}-{S['WIN3_SPLIT_D8'][-1]}; trigger_op SPLIT {ops['WIN3_SPLIT_D8']}, RESET {ops['WIN3_RESET_D8']}; "
    f"k = {R['WIN3_SPLIT_D8'][560]['k']}",
    st(S['WIN3_SPLIT_D8'] == list(range(560, 570)) == S['WIN3_RESET_D8'] == S['WIN3_SLOW_D8']
       and ops['WIN3_RESET_D8'] == {'reset'} and R['WIN3_SPLIT_D8'][560]['k'] == 16), "Unchanged.", old=(8,))
row(("$\\mathbf{10/10}$ & --- & $\\mathbf{3/10}$ & $5/10$",), "table row: counts",
    "copy+reset 10/10; copy only ---; reset only 3/10; none 5/10",
    f"SPLIT {nW['WIN3_SPLIT_D8']}/10; RESET {nW['WIN3_RESET_D8']}/10 {fs(WIN['WIN3_RESET_D8'])}; SLOW {nW['WIN3_SLOW_D8']}/10 "
    f"{fs(WIN['WIN3_SLOW_D8'])}", st([nW[a] for a in PART_A] == [10, 3, 5]),
    f"rlib agrees with the records' own fields on every run: {agree_rec}", old=(9,))
row(("C1 7 vs 0, $p = 0.0078$ SHOWN",), "table row: C1", "C1 7 vs 0, p = 0.0078 SHOWN",
    f"{c1[0]} vs {c1[1]} over {c1[2]} pairs, p = {p1:.4f} -> {c1_state}", st(c1[:2] == (7, 0) and round(p1, 4) == 0.0078),
    "", old=(10,))
row(("C2 reset vs none 0 vs 2, HOLDS",), "table row: C2", "C2 reset vs none 0 vs 2, HOLDS",
    f"RESET only {c2[0]}, SLOW only {c2[1]}, d = {d2} -> {c2_state}", st(c2[:2] == (0, 2) and c2_state == "HOLDS"),
    "Per machine as the code prints it; the spec's bound is pooled.", old=(11,))
row(("\\texttt{window\\_gate} control", "540--544, 1540--1544", "$3/10$ & $4/10$ & 0 vs 1"),
    "table row: window_gate control (cited by the caption)", "window_gate control: reset only 3/10, none 4/10, 0 vs 1",
    f"window_gate WIN3_RESET_D8 BR {len(ctl_reset)}/10 {ctl_reset}; WIN3_SLOW_D8 on the same seeds BR {len(ctl_none)}/10 "
    f"{ctl_none}; {ctl_b} vs {ctl_c}",
    st((len(ctl_reset), len(ctl_none), ctl_b, ctl_c) == (3, 4, 0, 1)), "Records at the main line's head.")
row(("fired under the live rule or at the recorded split updates", "on labelled targets where labelling was recorded"),
    "caption: reset-only arms fired, on labelled targets",
    "In every look the reset-only arm fired under the live rule or at the recorded split updates, on labelled targets "
    "where labelling was recorded",
    f"split_copy RESET: fired in {sum(1 for x in per_run['WIN3_RESET_D8'] if x)}/10 runs, {len(F_RESET)} resets, labelled "
    f"ok {ok_reset} (recorded {rec_ok_reset}); window_gate control: fired in {ctl_runs_fired}/10 runs, {len(ctl_f)} resets, "
    f"labelled ok {ctl_lab} (recorded {ctl_lab_rec})",
    st(len(F_RESET) > 0 and ok_reset == rec_ok_reset == len(F_RESET) and len(ctl_f) > 0
       and ctl_lab == ctl_lab_rec == len(ctl_f)),
    "Read at arm level (the arm fired, and every firing was on a labelled target): true in both main-line looks. One "
    f"control run never fired ({ctl_nofire}: none of its checks was eligible; it bound routed at 13200), so 'fired' is not true "
    "of every run. The garbled 'targets labelled on target' is fixed. Batch 18 / S58b / S61 parts are other sessions' "
    "records, not checked here.", old=(12, 13))
row(("it bound no more often than the arm with no trigger in \\texttt{test\\_split\\_copy} and the \\texttt{window\\_gate} control",
     "and more often in S61"), "caption: reset vs none, per look",
    "it bound no more often than the arm with no trigger in test_split_copy and the window_gate control, and more often in "
    "S61 (4/10 against 3/10, 2 vs 1)",
    f"split_copy {nW['WIN3_RESET_D8']}/10 vs {nW['WIN3_SLOW_D8']}/10; window_gate control {len(ctl_reset)}/10 vs "
    f"{len(ctl_none)}/10 (S61 not checked here)",
    st(nW['WIN3_RESET_D8'] <= nW['WIN3_SLOW_D8'] and len(ctl_reset) <= len(ctl_none)),
    "'In every look' is no longer applied to the comparison (referee B1).", old=(14,))
row(("\\texttt{test\\_split\\_copy}'s L half has not run; its pooled claims and readings wait for it",), "caption: L half",
    "test_split_copy's L half has not run; its pooled claims and readings wait for it",
    f"no L file on any ref ({len(refs)} remote branches searched); never committed: {not l_ever}",
    "NOT RECORD-CHECKABLE", "Consistent with the refs; whether L has started is not visible in git. Contradicts tex2:488.",
    old=(15,))
row(("Specified (\\texttt{specs/test\\_split\\_copy.md}) before it was written",), "specified before written",
    "Specified (specs/test_split_copy.md) before it was written",
    f"spec first added {spec_add} ({t_spec}); test first added {test_add} ({t_test}); spec ancestor of test: "
    f"{ancestor(SPEC_C, TEST_C)}; spec changed after {SPEC_C}: {spec_changed or 'never'}",
    st(spec_add == [SPEC_C] and test_add == [TEST_C] and t_spec < t_test and ancestor(SPEC_C, TEST_C) and not spec_changed),
    "", old=(16,))
row(("three claims pooled primary and per machine secondary",), "claims structure",
    "three claims pooled primary and per machine secondary",
    "spec/docstring: C1, C2, C3; 'POOLED over both machines PRIMARY ... per machine SECONDARY'",
    st("POOLED over both machines PRIMARY" in doc and "Pooled (PRIMARY)" in spec), "", old=(17,))
row(("seeds 560--569, 54 runs",), "seeds and run count", "seeds 560--569, 54 runs",
    f"{n_runs} records: 5 arms x 10 seeds + 2 perfect gates x 2; all ok = {ok_all()}", st(n_runs == 54 and ok_all()), "",
    old=(18,))
row(("every CHECK passed except two printed rather than asserted because another CPU wrote their records",),
    "CHECKs", "every CHECK passed except two printed rather than asserted because another CPU wrote their records",
    f"docstring RESULT: 'every inherited and own CHECK passed. test_muon_recipe's CHECK 130 (Muon, X's 2.80GHz "
    f"test_early_recipe file) differs here and was printed, not asserted' ({doc_130}); 'CHECK 146 reproduced the 2.80GHz "
    f"test_window_gate records ... bit for bit through 7200 (printed: other CPU)' ({doc_146}); split_copy ran on {sorted(cpus)}",
    st(doc_130 and doc_146 and cpus == {"Intel(R) Xeon(R) Processor @ 2.10GHz"}),
    "Correct now. One nuance is dropped: CHECK 146 reproduced bit for bit, but CHECK 130's values differed (the "
    "docstring's 'differs here'); 'printed rather than asserted' does not say that one of the two would have failed.",
    old=(19,))
row(("14.7 h from the test's commit to the result's, the runs about 12 h on four workers",), "wall clock",
    "14.7 h from the test's commit to the result's, the runs about 12 h on four workers",
    f"test commit {t_test:%H:%M:%S} -> result commit {t_res:%H:%M:%S} = {span_h:.2f} h; run phase from meta.started "
    f"{meta['started']} to the result commit {run_ub_h:.2f} h (upper bound); sum of run secs / {meta['workers']} workers "
    f"{run_lb_h:.2f} h (lower bound)",
    st(round(span_h, 1) == 14.7 and meta["workers"] == 4 and 11.0 <= run_lb_h <= 12.5 and run_ub_h <= 12.5),
    "Both numbers now named for what they are.", old=(20,))
row(("C1, the split beats the reset alone, SHOWN, $10/10$ against $3/10$, 7 vs 0, $p = 0.0078$",), "C1 text",
    "C1, the split beats the reset alone, SHOWN, 10/10 against 3/10, 7 vs 0, p = 0.0078",
    f"{nW['WIN3_SPLIT_D8']}/10 vs {nW['WIN3_RESET_D8']}/10, {c1[0]} vs {c1[1]}, p = {p1:.4f}, {c1_state}",
    st(c1[:2] == (7, 0) and round(p1, 4) == 0.0078), "", old=(21,))
row(("C2, the reset alone is not better than no trigger by more than two pairs, HOLDS, $3/10$ against $5/10$, 0 vs 2",),
    "C2 text", "C2 ... HOLDS, 3/10 against 5/10, 0 vs 2",
    f"{nW['WIN3_RESET_D8']}/10 vs {nW['WIN3_SLOW_D8']}/10, {c2[0]} vs {c2[1]}, d = {d2}, {c2_state}",
    st(c2[:2] == (0, 2) and c2_state == "HOLDS"), "", old=(22,))
row(("every seed the reset bound also bound without it",), "reset seeds subset of none seeds",
    "every seed the reset bound also bound without it",
    f"RESET BR {fs(WIN['WIN3_RESET_D8'])} subset of SLOW BR {fs(WIN['WIN3_SLOW_D8'])}: {reset_sub_slow}",
    st(reset_sub_slow), "Descriptive, in a parenthesis to C2.", old=(23,))
row(("All 22 splits and all 27 resets were on labelled targets (printed, not claimed)",), "splits and resets labelled",
    "All 22 splits and all 27 resets were on labelled targets (printed, not claimed)",
    f"splits {len(F_SPLIT)} labelled ok {ok_split} (recorded {rec_ok_split}); resets {len(F_RESET)} labelled ok {ok_reset} "
    f"(recorded {rec_ok_reset})",
    st(len(F_SPLIT) == 22 and len(F_RESET) == 27 and rec_ok_split == ok_split == 22 and rec_ok_reset == ok_reset == 27),
    "Marked as printed. 'labelled ok' = test_window_gate.targeting on map_before, the 256-sequence probe's map "
    "(tsa.routing_k on probe_batch, Generator(seed + 99000)); the first pass's note said the 64-sequence probe, which was "
    "wrong.", old=(24,))
row(("the reset never uses its target, so what the copy adds, on our reading, is the row and not the aim",),
    "the row, not the aim (interpretation)",
    "the reset never uses its target, so what the copy adds, on our reading, is the row and not the aim",
    f"test_window_gate.reset_op (twg:{code_line(twg, 'def reset_op(m, o):')}) takes no (c*, c0) and only zeroes W_g's "
    f"Adam state: {reset_ignores_target}", st(reset_ignores_target),
    "Now marked 'on our reading', with the fact that makes it so.", old=(25,))
row(("the Hebbian perfect gate at 4800 and 3600; the delta perfect gate at 1200 and 1200",), "validity",
    "validity arms bound 2/2 (Hebbian perfect gate at 4800 and 3600; delta perfect gate at 1200 and 1200)",
    f"CEIL_C {nW['CEIL_C']}/2 {ceil_t('CEIL_C')}; CEIL_DELTA_D8 {nW['CEIL_DELTA_D8']}/2 {ceil_t('CEIL_DELTA_D8')}",
    st(ceil_t('CEIL_C') == [4800, 3600] and ceil_t('CEIL_DELTA_D8') == [1200, 1200]), "", old=(26,))
row(("``the copy, not the reset'' if C1 is SHOWN pooled and C2 HOLDS",), "reading 1 quoted",
    "the pre-fixed readings (``the copy, not the reset'' if C1 is SHOWN pooled and C2 HOLDS) wait for L's ten seeds",
    "docstring READINGS (pooled only): '\"the copy, not the reset\" if C1 is SHOWN and C2 HOLDS'",
    st('"the copy, not the reset" if C1 is SHOWN and C2 HOLDS' in doc), "", old=(27,))
row(("the readings are pooled only, and on X alone their conditions hold as secondary claims",),
    "readings pooled only; X's conditions", "the readings are pooled only, and on X alone their conditions hold as secondary claims",
    f"reading 1 on X: C1 {c1_state}, C2 {c2_state}; reading 2's X-side conditions: C3 {c3_state}, SINGLE_DELTA "
    f"{nW['SINGLE_DELTA_D8']}/10 below MAJORITY, CEIL_DELTA X {nW['CEIL_DELTA_D8']}/2 (its L half cannot hold on X)",
    st(c1_state == "SHOWN" and c2_state == "HOLDS"),
    "Replaces 'per machine the reading is met on X'; scoped as secondary claims. Of reading 2's conditions only the X "
    "side can hold on X (it needs CEIL_DELTA_D8 bound on both machines), and its band condition is not a claim; the "
    "sentence's parenthesis names reading 1 only, so read it as reading 1.", old=(28,))
# Section memory
row(("is SHOWN on X: the recipe $10/10$ against one delta channel $0/10$ (10 vs 0, $p = 0.00098$",), "C3",
    "C3 is SHOWN on X: the recipe 10/10 against one delta channel 0/10 (10 vs 0, p = 0.00098)",
    f"SPLIT BR {nW['WIN3_SPLIT_D8']}/10 vs SINGLE_DELTA_D8 BOUND {nW['SINGLE_DELTA_D8']}/10; {c3[0]} vs {c3[1]}; "
    f"p = {p3:.5f} -> {c3_state}", st(c3[:2] == (10, 0) and round(p3, 5) == 0.00098), "", old=(29,))
row(("final accuracy $0.08$--$0.76$",), "single delta final accuracy", "final accuracy 0.08--0.76",
    f"SINGLE_DELTA_D8 final acc {min(acc_d):.4f}-{max(acc_d):.4f}",
    st(f"{min(acc_d):.2f}" == "0.08" and f"{max(acc_d):.2f}" == "0.76"), "", old=(30,))
row(("one Hebbian channel $0/10$ (printed, not claimed)",), "one Hebbian channel", "one Hebbian channel 0/10 (printed, not claimed)",
    f"SINGLE_HEBB_D8 BOUND {nW['SINGLE_HEBB_D8']}/10 (final acc {min(acc_h):.3f}-{max(acc_h):.3f}); docstring PRINTED, NOT "
    f"CLAIMS lists it: {'SINGLE_DELTA_D8 vs SINGLE_HEBB_D8' in doc}", st(nW['SINGLE_HEBB_D8'] == 0), "Now marked.", old=(31,))
row(("the delta perfect gate $2/2$ at 1200",), "delta perfect gate", "the delta perfect gate 2/2 at 1200",
    f"CEIL_DELTA_D8 {nW['CEIL_DELTA_D8']}/2 {ceil_t('CEIL_DELTA_D8')}", st(ceil_t('CEIL_DELTA_D8') == [1200, 1200]), "",
    old=(32,))
row(("which is what would make the channel's failure the missing partition rather than the delta rule, wait for L",),
    "reading 2 and its content", "the pooled claim and the reading ``the partition is necessary at eight streams against a "
    "single delta channel'', which is what would make the channel's failure the missing partition rather than the delta "
    "rule, wait for L",
    "docstring READINGS (pooled only): identical string", st('"the partition is necessary at eight streams against a single delta channel"' in doc),
    "The inference is now conditional on the pooled reading.", old=(33, 34))
# synthesis table
row(("$15/40$; X $5/10^{s}$", "X $10/10^{s}$"), "synthesis S=8 row: window gate + SLOW and split columns",
    "15/40; X 5/10^s  |  38/40; X 10/10^s",
    f"WIN3_SLOW_D8 BR {nW['WIN3_SLOW_D8']}/10; WIN3_SPLIT_D8 BR {nW['WIN3_SPLIT_D8']}/10",
    st(nW['WIN3_SLOW_D8'] == 5 and nW['WIN3_SPLIT_D8'] == 10), "The no-split arm of the X half is now shown.",
    old=(35, 37))
row(("$0/10$ (X) & $0/10$ (X)",), "synthesis: single channels", "one Hebbian channel 0/10 (X); one delta channel 0/10 (X)",
    f"SINGLE_HEBB_D8 {nW['SINGLE_HEBB_D8']}/10, SINGLE_DELTA_D8 {nW['SINGLE_DELTA_D8']}/10 (BOUND)",
    st(nW['SINGLE_HEBB_D8'] == 0 == nW['SINGLE_DELTA_D8']), "k = 1 cells are BOUND (k = 1 cannot route).", old=(36,))
# assessment and conclusion
row(("Pre-registered and shown on one machine", "pooled pending L"), "assessment: copy not reset",
    "Pre-registered and shown on one machine (C1, 7 vs 0; C2 holds), pooled pending L",
    f"C1 {c1[0]} vs {c1[1]} {c1_state}; C2 {c2_state}", st(c1[:2] == (7, 0) and c2_state == "HOLDS"), "", old=(38,))
row(("\\textbf{Supported on X} by the pre-registered C3 ($0/10$ against the recipe's $10/10$)",),
    "assessment: number of contexts", "Supported on X by the pre-registered C3 (0/10 against the recipe's 10/10); session F's "
    "S68 had the same counts (2/10 at four streams, 0/10 at eight)",
    f"SINGLE_DELTA_D8 {nW['SINGLE_DELTA_D8']}/10 vs SPLIT {nW['WIN3_SPLIT_D8']}/10",
    st(nW['SINGLE_DELTA_D8'] == 0 and nW['WIN3_SPLIT_D8'] == 10),
    "Counts match. C3 is at eight streams only; a statement about the number of contexts needs the four-vs-eight contrast, "
    "which only S68 (exploratory, reading 'neither') has. 'the same counts (2/10 at four ...)' compares with a four-stream "
    "count C3 does not have.", old=(39,))
row(("by the pre-registered C1 (the copy with its reset against the reset alone, 7 vs 0)",
     "that the runs it rescues are merges is printed in their failure classes, not claimed"),
    "assessment: row copy breaks merges", "Supported ... and, on X, by the pre-registered C1 (... 7 vs 0); that the runs it "
    "rescues are merges is printed in their failure classes, not claimed",
    f"C1 {c1[0]} vs {c1[1]} {c1_state}; the reset arm's outcome on the {len(c1_seeds)} rescued seeds: "
    f"{sorted(set(c1_reset_oc.values()))}",
    st(c1[:2] == (7, 0) and all(v.startswith("MERGED") for v in c1_reset_oc.values())),
    "C1 is now the pre-registered support; the merge reading is marked as printed.", old=(40,))
row(("On one machine so far, pre-registered, the copy with its reset beats the reset alone",
     "the pooled readings wait for the other machine"), "conclusion",
    "On one machine so far, pre-registered, the copy with its reset beats the reset alone, the reset alone is within the "
    "bound of no trigger, and one delta channel binds none of ten seeds where the recipe binds all ten; the pooled readings "
    "wait for the other machine.",
    f"X: C1 {c1_state}, C2 {c2_state}, C3 {c3_state}; SINGLE_DELTA 0/10, SPLIT 10/10",
    st((c1_state, c2_state, c3_state) == ("SHOWN", "HOLDS", "SHOWN")), "Scoped.", old=(41,))
row(("all-position pair was a labelled target", "at 97 (21 of 22, 46 of 47 and 30 of 38)",
     "the key-position pair at all 107; the two pairs coincide at 46"),
    "limits: all-position targets (new)", "at the 107 firings of the 50 recorded split runs at eight streams (window_gate X "
    "and L, split_copy X) the all-position pair was a labelled target ... at 97 (21 of 22, 46 of 47 and 30 of 38), the "
    "key-position pair at all 107; the two pairs coincide at 46",
    "; ".join(f"{n_}: {r_} runs, {f_} firings, all-position labelled {a_} (recorded all_ok {ar_}), key labelled {k_}, "
              f"coincide {co_}" for n_, r_, f_, a_, ar_, k_, co_ in ALLPOS)
    + f"; total {AP[0]} runs, {AP[1]} firings, {AP[2]}, {AP[4]}, {AP[5]}; every check row stores both pairs: {every_row_has_all}",
    st(AP[0] == 50 and AP[1] == 107 and AP[2] == AP[3] == 97 and [x[3] for x in ALLPOS] == [21, 46, 30]
       and AP[4] == 107 and AP[5] == 46 and every_row_has_all),
    "Marked 'our count, not a reading'. The all-position pair is make_trigger's select(): argmax/argmin of the read gate "
    "mean over all positions of the 64-sequence probe.")
# provenance
row(("spec \\texttt{fb3acb7} / \\texttt{1f52cd5} / \\texttt{592c574}",), "provenance commits",
    "spec fb3acb7 / 1f52cd5 / 592c574",
    f"{cinfo(SPEC_C)} | {cinfo(TEST_C)} | {cinfo(RES)}; meta.git = {meta['git']}; result commit changes the test by "
    f"+{res_py[0]}/-{res_py[1]} docstring lines; JSON and test unchanged to head {HEAD_SHORT}: {rec_unchanged}",
    st(spec_add == [SPEC_C] and test_add == [TEST_C] and rec_add == [RES] and meta['git'] == TEST_C and res_doc_only
       and res_py[1] == "0" and rec_unchanged and py_changed == ["test_split_copy.py"]), "", old=(42,))
row(("the launcher \\texttt{56c44a7}",), "classifier / launcher", "the classifier ba8901a, the launcher 56c44a7",
    f"both ancestors of {TEST_C}: {ancestor('ba8901a', TEST_C) and ancestor('56c44a7', TEST_C)}; longrun.py edited in "
    f"{TEST_C} (+{lr_diff[0]}/-{lr_diff[1]})",
    st(ancestor('ba8901a', TEST_C) and ancestor('56c44a7', TEST_C) and imports_fc2 and uses_lr), "", old=(43,))
row(("running & 54; ---",), "provenance table: L column and runs",
    "split_copy ... & running & 54; ---  (column 'L: results file (commit recorded)')",
    f"X records {n_runs}; no L file on any ref",
    "PARTIAL",
    "Unchanged from the first draft, and now inconsistent with this draft's own text: tex2:41 ('L's half has not run'), "
    "tex2:53 ('L to run'), tex2:195 ('has not run') and tex2:469 (next step). One of the two is wrong; git cannot say "
    "which.", old=(44,))
row(("2.10\\,GHz for \\texttt{split\\_copy}",), "X CPU for split_copy", "2.10 GHz for split_copy",
    f"meta.cpu '{meta['cpu']}'; per-record {sorted(cpus)}", st(cpus == {meta['cpu']} and "2.10GHz" in meta['cpu']), "",
    old=(45,))

# ── output ───────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
new_of = {}
for i, r in enumerate(rows):
    for o in r[6]:
        new_of.setdefault(o, []).append(i)


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


P = print
P("# Revision 9 draft, second version, vs test_split_copy on X — referee check (second pass)\n")
P(f"Second draft `{DRAFT}` at `{NEW_REF}` (`tex2:NNN`); first draft at `{OLD_REF}` (`tex:NNN`). Records "
  f"`{REC}` at `{RES}` (identical at the main line's head `{HEAD_SHORT}`: {rec_unchanged}); window_gate records and "
  f"code at `{HEAD_SHORT}`; test docstring at `{RES}`, spec at `{SPEC_C}`. Outcomes recomputed with `rlib`; they agree "
  f"with the records' own transition ({agree_t}) and bound-routed fields ({agree_rec}); ch_map ties: {len(rlib.TIES)}. "
  f"L's file `{L_REC}`: on no remote branch ({len(refs)} searched), never committed. Generated by "
  f"`check_split_copy_v2.py`.\n")
cnt = {k: sum(1 for r in rows if r[4] == k) for k in ORDER}
removed = [i for i, o in enumerate(OLD) if locate(tex2, o[3]) is None]
P("Counts against the second draft: " + ", ".join(f"{k} {v}" for k, v in cnt.items())
  + f" (of {len(rows)} rows). First-pass fragments: {len(OLD) - len(removed)} of {len(OLD)} found verbatim in the second "
  f"draft, {len(removed)} REMOVED or reworded.\n")

P("## 1. Rows against the second draft\n")
P("| tex2 line | item | draft says | recomputed | status | note | first-pass row |")
P("|---|---|---|---|---|---|---|")
for n, item, d, rc, s, note, old in sorted(rows, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex2:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} | "
      f"{', '.join(f'#{o + 1} (tex:{OLD[o][0]})' for o in old) or 'new'} |")

P("\n## 2. First-pass rows located in the second draft\n")
P("Each first-pass fragment was asserted at its old line of `9bfa85a` and searched verbatim in `f67bc07`.\n")
P("| # | tex line | item | first-pass status | fragment | in second draft | now (section 1) |")
P("|---|---|---|---|---|---|---|")
for i, (n, item, s, ph) in enumerate(OLD):
    m = locate(tex2, ph)
    nows = "; ".join(f"tex2:{rows[j][0]} {rows[j][4]}" for j in new_of.get(i, [])) or "no row"
    P(f"| {i + 1} | tex:{n} | {esc(item)} | {s} | {esc(' / '.join(ph))} | {f'tex2:{m}' if m else 'REMOVED'} | {nows} |")

P("\n## 3. Per-run records (X, seeds 560-569)\n")
P("| seed | WIN3_SLOW_D8 | WIN3_RESET_D8 | WIN3_SPLIT_D8 | SINGLE_DELTA_D8 (final acc) | SINGLE_HEBB_D8 (final acc) |")
P("|---|---|---|---|---|---|")
for s in S["WIN3_SPLIT_D8"]:
    cells = []
    for a in ("WIN3_SLOW_D8", "WIN3_RESET_D8", "WIN3_SPLIT_D8"):
        r = R[a][s]
        lab = "BR" if s in WIN[a] else rlib.outcome_k(r)
        fs_ = [f"{c['step']}:{c['key_cs']}->{c['key_c0']}{'' if targeting(c['map_before'], c['key_cs'], c['key_c0']) else ' NOT'}"
               f"{'' if targeting(c['map_before'], c['all_cs'], c['all_c0']) else ' (all-pos NOT)'}"
               for c in r.get("checks", []) if c["fired"]]
        cells.append(f"{lab} ({r['transition']})" + (f"; {', '.join(fs_)}" if fs_ else ""))
    for a in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8"):
        cells.append(f"{'BOUND' if s in WIN[a] else 'not bound'} ({R[a][s]['acc']:.3f})")
    P(f"| {s} | " + " | ".join(cells) + " |")
P(f"\nPerfect gates: CEIL_C {dict(zip(S['CEIL_C'], ceil_t('CEIL_C')))}, CEIL_DELTA_D8 "
  f"{dict(zip(S['CEIL_DELTA_D8'], ceil_t('CEIL_DELTA_D8')))}. WIN3_SPLIT_D8 transition median {rlib.med(tr_split):.0f} "
  f"({min(tr_split)}-{max(tr_split)}). Bands (Wilson 95%): "
  + "; ".join(f"{a} {nW[a]}/{len(S[a])} {rlib.band(nW[a], len(S[a]))} [{rlib.wilson(nW[a], len(S[a]))[0]:.3f}, "
              f"{rlib.wilson(nW[a], len(S[a]))[1]:.3f}]" for a in PART_A + ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8"))
  + f". RESET failures: {sorted(set(fail_oc['WIN3_RESET_D8']))}; SLOW failures: {sorted(set(fail_oc['WIN3_SLOW_D8']))}.")
P(("\nNo number in the second draft's split_copy statements disagrees with the records. " if cnt["DISAGREE"] == 0 else
   f"\n{cnt['DISAGREE']} row(s) DISAGREE with the records. ")
  + "The first pass's scope points (per machine vs pooled, readings vs claims, printed vs claimed) are addressed; what "
  "remains is listed under PARTIAL above.")
