"""check_split_copy.py — Referee V: Revision 9 draft (origin/claude/rev9-draft @ 9bfa85a,
multichannel_hebbian_report_v9.tex) against test_split_copy's records on X.

Inputs, all read with git at pinned refs (nothing is run, nothing is written but stdout):
  draft   9bfa85a:multichannel_hebbian_report_v9.tex
  records 592c574:results/X/split_copy_results.json   (unchanged at the branch head 32c121b)
  test    592c574:test_split_copy.py (docstring CLAIMS / READINGS / RESULT), spec fb3acb7:specs/test_split_copy.md
  code    592c574:test_window_gate.py (targeting = 'labelled ok')
Outcomes from rlib (re-implemented from the report's Sec. 3), cross-checked against the records' own fields.

    python3 docs/review_2026-10/rev9/check_split_copy.py > docs/review_2026-10/rev9/check_split_copy.md
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402

DRAFT_REF, DRAFT = "9bfa85a", "multichannel_hebbian_report_v9.tex"
HEAD = "origin/claude/bdh-growth-hebbian-inference-w90069"
RES, REC = "592c574", "results/X/split_copy_results.json"
SPEC_C, TEST_C = "fb3acb7", "1f52cd5"
ARMS = ("CEIL_C", "CEIL_DELTA_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8")
PART_A = ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")

tex = rlib.git("show", f"{DRAFT_REF}:{DRAFT}").split("\n")
doc = rlib.git("show", f"{RES}:test_split_copy.py")
spec = rlib.git("show", f"{SPEC_C}:specs/test_split_copy.md")
twg = rlib.git("show", f"{RES}:test_window_gate.py")
store = rlib.load(RES, REC)
R = {a: rlib.runs(store, a) for a in ARMS}


def line(n):
    return tex[n - 1]


def has(n, *phrases):
    for p in phrases:
        assert p in line(n), (n, p)


# ── the records ──────────────────────────────────────────────────────────────
def ok_all():
    return all(rlib.ok(r) for a in ARMS for r in R[a].values())


def succ(a, r):
    """Part A learned gates: BOUND ROUTED; Part B and perfect gates: BOUND (spec 'Outcomes')."""
    return rlib.bound_routed(r) if a in PART_A else rlib.bound(r)


def succ_rec(a, r):
    """The same from the records' own fields (recorded transition, end.one_to_one, stream_acc), as
    test_stream_recipe.routed / test_conv_lr.bound_r read them."""
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


def pair(a, b):
    ss = sorted(set(S[a]) & set(S[b]))
    x = sum(1 for s in ss if s in WIN[a] and s not in WIN[b])
    y = sum(1 for s in ss if s in WIN[b] and s not in WIN[a])
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


def fired(a):
    return [(s, c) for s in S[a] for c in R[a][s].get("checks", []) if c["fired"]]


def targeting(cmap, cs, c0):  # test_window_gate.targeting, re-implemented
    return cmap.count(cs) >= 2 and cmap.count(c0) == 0


assert "return cmap.count(cs) >= 2 and cmap.count(c0) == 0" in twg
F_SPLIT, F_RESET = fired("WIN3_SPLIT_D8"), fired("WIN3_RESET_D8")
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
# fail_class_v2 vs v1 (CHECK 151b): they differ only where margin >= 0.25 and eta^2 by key >= 0.5
v12_conflict = [(a, s) for a in PART_A for s in S[a] if R[a][s]["transition"] is None
                and R[a][s]["end"]["margin"] >= rlib.MARGIN_PARTIAL and R[a][s]["end"]["etak_key_by_key"] >= rlib.ETA_SPLIT]


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
py_changed = [f for f in rlib.git("diff", "--name-only", TEST_C, RES, "--", "*.py").split()
              if not f.startswith("docs/")]   # the referee's own scripts under docs/ are not the test's code
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


def fs(xs):
    return "{" + ", ".join(f"s{x}" for x in xs) + "}"


def st(cond):
    return "MATCH" if cond else "DISAGREE"


rows = []


def row(n, item, draft, recomputed, status, note="", phrases=()):
    has(n, *phrases)
    rows.append((n, item, draft, recomputed, status, note))


nA = len(S["WIN3_SPLIT_D8"])
# ── abstract and front matter ────────────────────────────────────────────────
row(36, "abstract: C1 counts", "copy and reset 10/10 against the reset alone 3/10, 7 vs 0",
    f"WIN3_SPLIT_D8 BR {len(WIN['WIN3_SPLIT_D8'])}/{nA}; WIN3_RESET_D8 BR {len(WIN['WIN3_RESET_D8'])}/{nA}; "
    f"discordant {c1[0]} vs {c1[1]}",
    st((len(WIN['WIN3_SPLIT_D8']), len(WIN['WIN3_RESET_D8']), c1[0], c1[1]) == (10, 3, 7, 0)), "",
    ("copy and reset 10/10 against the reset alone 3/10, 7 vs 0",))
row(36, "abstract: reset vs no trigger", "the reset alone no better than no trigger",
    f"RESET BR {len(WIN['WIN3_RESET_D8'])}/10 vs SLOW {len(WIN['WIN3_SLOW_D8'])}/10, {c2[0]} vs {c2[1]} (d = {d2})",
    st(c2[0] <= c2[1]),
    "True on X's counts. C2 itself is only a bound (d <= 2 pairs, pooled); 'no better' is a descriptive reading of "
    "X's 0 vs 2, not the claim's wording.", ("the reset alone no better than no trigger",))
row(36, "abstract: C3 counts", "one delta-rule channel binds eight streams 0/10 where the partition binds 10/10",
    f"SINGLE_DELTA_D8 BOUND {len(WIN['SINGLE_DELTA_D8'])}/10; WIN3_SPLIT_D8 BR {len(WIN['WIN3_SPLIT_D8'])}/10; "
    f"{c3[0]} vs {c3[1]}",
    st(len(WIN['SINGLE_DELTA_D8']) == 0 and len(WIN['WIN3_SPLIT_D8']) == 10),
    "'the partition' here is the learned recipe (window gate + SLOW + split), not a perfect gate.",
    ("binds eight streams $0/10$ where the partition binds $10/10$",))
row(36, "abstract: framing of C1/C2 as 'the row copy is the ingredient'",
    "the second claim of Revision 8 is now a pre-registered result on one machine ...: the split's row copy is the "
    "ingredient and its optimizer reset alone is not enough",
    f"per machine (SECONDARY) on X: C1 {c1_state}, C2 {c2_state}; the reading 'the copy, not the reset' is "
    "pre-registered POOLED ONLY ('Nothing else is read') and is not yet evaluated",
    "PARTIAL",
    "The per-machine claims C1 and C2 are pre-registered (secondary) and hold on X. 'the row copy is the ingredient' "
    "is the content of the pooled-only reading, stated without the 'per machine / reading pending' qualifier "
    "(the parenthesis 'the other's half is running' is there).",
    ("pre-registered result on one machine", "the split's row copy is the ingredient"))
row(41, "Rev. 9 paragraph", "test_split_copy, complete on X and running on L",
    f"X file: {n_runs} records, all ok = {ok_all()}; meta seeds {sum(len(v) for v in meta['seeds'].values())} runs",
    st(n_runs == 54 and ok_all()), "'running on L' is not record-checkable (no L file at the branch head).",
    ("complete on X and running on L",))
row(53, "index row", "on X: C1, C3 SHOWN, C2 HOLDS; pooled readings wait for L",
    f"C1 {c1_state} (p={p1:.4f}); C3 {c3_state} (p={p3:.5f}); C2 {c2_state} (d={d2})",
    st((c1_state, c2_state, c3_state) == ("SHOWN", "HOLDS", "SHOWN")), "",
    ("on X: C1, C3 SHOWN, C2 HOLDS; pooled readings wait for L",))
row(115, "classifier commit", "fail_class_v2 (commit ba8901a) ... used by every test from test_split_copy on",
    f"{cinfo('ba8901a')}; test_split_copy imports fail_class_v2: {imports_fc2}; fail_class_v2.py unchanged "
    f"ba8901a..{TEST_C}: {fc2_same}; ancestor of {TEST_C}: {ancestor('ba8901a', TEST_C)}",
    st(imports_fc2 and fc2_same and ancestor('ba8901a', TEST_C)), "", ("ba8901a",))

# ── Section split: table, caption, paragraph ─────────────────────────────────
row(178, "'settle the question'", "Two further looks settle the question, one of them pre-registered.",
    "the pre-registered look's PRIMARY (pooled) claims and its reading are untested (L running)",
    "PARTIAL", "Per-machine C1/C2 on X support it; 'settle' outruns the pre-registered status (pooled primary pending), "
    "which tex:195/199 state.", ("Two further looks settle the question, one of them pre-registered",))
row(192, "table row: arms and seeds", "split_copy, X; S=8, k=16, 560--569; live trigger; pre-registered",
    f"seeds {S['WIN3_SPLIT_D8'][0]}-{S['WIN3_SPLIT_D8'][-1]} (all three Part A arms); trigger_op SPLIT {ops['WIN3_SPLIT_D8']}, "
    f"RESET {ops['WIN3_RESET_D8']}; k = {R['WIN3_SPLIT_D8'][560]['k']}",
    st(S['WIN3_SPLIT_D8'] == list(range(560, 570)) == S['WIN3_RESET_D8'] == S['WIN3_SLOW_D8']
       and ops['WIN3_RESET_D8'] == {'reset'} and R['WIN3_SPLIT_D8'][560]['k'] == 16),
    "Arm mapping (test code ARMS): copy + reset = WIN3_SPLIT_D8 (copy with noise 0.1 std + W_g Adam reset); reset only = "
    "WIN3_RESET_D8 (same trigger, W_g untouched); none = WIN3_SLOW_D8 (no trigger); copy only: no such arm ('---').",
    ("560--569; live trigger; pre-registered",))
row(192, "table row: counts (bound routed)", "copy+reset 10/10; copy only ---; reset only 3/10; none 5/10",
    f"SPLIT {len(WIN['WIN3_SPLIT_D8'])}/10 {fs(WIN['WIN3_SPLIT_D8']) if len(WIN['WIN3_SPLIT_D8']) < 10 else ''}; "
    f"RESET {len(WIN['WIN3_RESET_D8'])}/10 {fs(WIN['WIN3_RESET_D8'])}; SLOW {len(WIN['WIN3_SLOW_D8'])}/10 "
    f"{fs(WIN['WIN3_SLOW_D8'])}",
    st([len(WIN[a]) for a in PART_A] == [10, 3, 5]),
    f"rlib bound_routed agrees with the records' own transition/one_to_one/stream_acc on every run: {agree_rec}",
    ("$\\mathbf{10/10}$ & --- & $\\mathbf{3/10}$ & $5/10$",))
row(192, "table row: C1", "C1 7 vs 0, p = 0.0078 SHOWN",
    f"{c1[0]} vs {c1[1]} over {c1[2]} pairs, one-sided exact McNemar p = {p1:.4f} -> {c1_state}",
    st(c1[:2] == (7, 0) and round(p1, 4) == 0.0078), "", ("C1 7 vs 0, $p = 0.0078$ SHOWN",))
row(192, "table row: C2", "C2 reset vs none 0 vs 2, HOLDS",
    f"RESET only {c2[0]}, SLOW only {c2[1]}, d = {d2} <= 2 -> {c2_state}",
    st(c2[:2] == (0, 2) and c2_state == "HOLDS"),
    "Spec: C2's bound is pooled; per machine 'the same d is printed'. The code labels it HOLDS per machine too "
    "(c2 state), as the docstring RESULT does.", ("C2 reset vs none 0 vs 2, HOLDS",))
row(195, "caption: reset-only arm fired under the live rule",
    "In every look the reset-only arm fired under the live rule",
    f"WIN3_RESET_D8 trigger_op {ops['WIN3_RESET_D8']}; fired in {sum(1 for x in per_run['WIN3_RESET_D8'] if x)}/10 runs, "
    f"per run {per_run['WIN3_RESET_D8']}",
    st(ops['WIN3_RESET_D8'] == {'reset'} and all(per_run['WIN3_RESET_D8'])), "",
    ("fired under the live rule",))
row(195, "caption: targets labelled", "on targets labelled on target where labelling was recorded",
    f"RESET firings with key_ok (recorded) {rec_ok_reset}/{len(F_RESET)}; recomputed from map_before "
    f"{ok_reset}/{len(F_RESET)}",
    st(rec_ok_reset == ok_reset == len(F_RESET)),
    "Wording: 'targets labelled on target' is garbled; the code's term is 'labelled ok' (test_window_gate.targeting: "
    "c* holds >= 2 streams and c0 none on the probe's ch_map before the check). For a reset the (c*, c0) choice is "
    "computed but not used.", ("on targets labelled on target where labelling was recorded",))
row(195, "caption: reset bound no more often than none", "bound no more often than the arm with no trigger",
    f"RESET {len(WIN['WIN3_RESET_D8'])}/10 vs SLOW {len(WIN['WIN3_SLOW_D8'])}/10",
    st(len(WIN['WIN3_RESET_D8']) <= len(WIN['WIN3_SLOW_D8'])), "", ("bound no more often than the arm with no trigger",))
row(195, "caption: L half", "test_split_copy's L half is running; its pooled claims and readings wait for it",
    "no L file at the branch head", "NOT RECORD-CHECKABLE", "Consistent with the docstring ('pooled waits for L').",
    ("L half is running",))
row(199, "specified before written", "Specified (specs/test_split_copy.md) before it was written",
    f"spec first added {spec_add} ({t_spec}); test first added {test_add} ({t_test}); spec ancestor of test: "
    f"{ancestor(SPEC_C, TEST_C)}; spec changed after {SPEC_C}: {spec_changed or 'never'}",
    st(spec_add == [SPEC_C] and test_add == [TEST_C] and t_spec < t_test and ancestor(SPEC_C, TEST_C) and not spec_changed),
    "", ("Specified (\\texttt{specs/test\\_split\\_copy.md}) before it was written",))
row(199, "claims structure", "three claims pooled primary and per machine secondary",
    "spec/docstring: C1, C2, C3; 'POOLED over both machines PRIMARY ... per machine SECONDARY'",
    st("POOLED over both machines PRIMARY" in doc and "Pooled (PRIMARY)" in spec), "",
    ("three claims pooled primary and per machine secondary",))
row(199, "seeds and run count", "seeds 560--569, 54 runs",
    f"{n_runs} records: 5 arms x 10 seeds (560-569) + 2 perfect gates x 2 (560-561); all ok = {ok_all()}",
    st(n_runs == 54 and ok_all()), "", ("seeds 560--569, 54 runs",))
row(199, "every CHECK passed", "every CHECK passed",
    "not in the JSON; docstring RESULT: 'every inherited and own CHECK passed'", "NOT RECORD-CHECKABLE",
    "The docstring adds that test_muon_recipe's CHECK 130 'differs here and was printed, not asserted' (2.80 GHz file) "
    "and CHECK 146 was printed (other CPU); the draft's 'every CHECK passed' drops that qualification.",
    ("every CHECK passed",))
row(199, "wall clock", "14.7 h",
    f"JSON has no end time. Run phase started {meta['started']} (meta); sum of run secs / {meta['workers']} workers = "
    f"{run_lb_h:.1f} h (lower bound on the run phase); test commit {t_test:%H:%M:%S} -> result commit {t_res:%H:%M:%S} "
    f"= {span_h:.2f} h", "NOT RECORD-CHECKABLE",
    "Consistent: the docstring's 14.7 h counts from the 05:06 start, verification (~2.7 h) included; the runs "
    f"themselves took about {(t_res - started).total_seconds() / 3600:.1f} h or less.", ("14.7 h",))
row(199, "C1 text", "C1, the split beats the reset alone, SHOWN, 10/10 against 3/10, 7 vs 0, p = 0.0078",
    f"{len(WIN['WIN3_SPLIT_D8'])}/10 vs {len(WIN['WIN3_RESET_D8'])}/10, {c1[0]} vs {c1[1]}, p = {p1:.4f}, {c1_state}",
    st(c1[:2] == (7, 0) and round(p1, 4) == 0.0078 and len(WIN['WIN3_RESET_D8']) == 3),
    "Wording matches C1 ('WIN3_SPLIT_D8 beats WIN3_RESET_D8 (BOUND ROUTED)').",
    ("C1, the split beats the reset alone, SHOWN, $10/10$ against $3/10$, 7 vs 0, $p = 0.0078$",))
row(199, "C2 text", "C2, the reset alone is not better than no trigger by more than two pairs, HOLDS, 3/10 against 5/10, 0 vs 2",
    f"{len(WIN['WIN3_RESET_D8'])}/10 vs {len(WIN['WIN3_SLOW_D8'])}/10, {c2[0]} vs {c2[1]}, d = {d2}, {c2_state}",
    st(c2[:2] == (0, 2) and c2_state == "HOLDS"),
    "Wording matches C2 except the spec's 'pooled' (here applied per machine, as the code prints it).",
    ("C2, the reset alone is not better than no trigger by more than two pairs, HOLDS, $3/10$ against $5/10$, 0 vs 2",))
row(199, "reset seeds subset of none seeds", "every seed the reset bound also bound without it",
    f"RESET BR {fs(WIN['WIN3_RESET_D8'])} subset of SLOW BR {fs(WIN['WIN3_SLOW_D8'])}: {reset_sub_slow}",
    st(reset_sub_slow), "Descriptive (printed in the docstring), not a claim; equivalent to RESET-only = 0.",
    ("every seed the reset bound also bound without it",))
row(199, "splits and resets on labelled targets", "All 22 splits and all 27 resets were on correctly labelled targets",
    f"splits {len(F_SPLIT)} (per run {per_run['WIN3_SPLIT_D8']}), key_ok recorded {rec_ok_split}, recomputed {ok_split}; "
    f"resets {len(F_RESET)} (per run {per_run['WIN3_RESET_D8']}), key_ok recorded {rec_ok_reset}, recomputed {ok_reset}",
    st(len(F_SPLIT) == 22 and len(F_RESET) == 27 and rec_ok_split == ok_split == 22 and rec_ok_reset == ok_reset == 27),
    "'labelled ok' = targeting(map_before, key_cs, key_c0): c* holds >= 2 streams, c0 none, on the 64-sequence probe's "
    "ch_map before the check (test_window_gate.make_trigger). 'correctly labelled' is the draft's paraphrase.",
    ("All 22 splits and all 27 resets were on correctly labelled targets",))
row(199, "'so the difference is the row copy, not the aim'", "so the difference is the row copy, not the aim",
    "an inference from the printed targeting rows (descriptive), not a claim or reading", "PARTIAL",
    "The numbers support it; it is not marked as interpretation. Note the reset never uses its (c*, c0); 'the aim' of a "
    "reset is the trigger's selection, which would have been right.", ("so the difference is the row copy, not the aim",))
row(199, "validity", "validity arms bound 2/2 (Hebbian perfect gate at 4800 and 3600; delta perfect gate at 1200 and 1200)",
    f"CEIL_C {len(WIN['CEIL_C'])}/2 transitions {ceil_t('CEIL_C')} (s560, s561); CEIL_DELTA_D8 "
    f"{len(WIN['CEIL_DELTA_D8'])}/2 transitions {ceil_t('CEIL_DELTA_D8')}",
    st(ceil_t('CEIL_C') == [4800, 3600] and ceil_t('CEIL_DELTA_D8') == [1200, 1200] and len(WIN['CEIL_C']) == 2
       and len(WIN['CEIL_DELTA_D8']) == 2), "Parts A and B VALID on X (VALID_N = 2 of 2).",
    ("the Hebbian perfect gate at 4800 and 3600; the delta perfect gate at 1200 and 1200",))
row(199, "reading 1 quoted", "``the copy, not the reset'' if C1 is SHOWN pooled and C2 HOLDS ... wait for L's ten seeds",
    "docstring/spec: '\"the copy, not the reset\" if C1 is SHOWN and C2 HOLDS' under READINGS (pooled only)",
    st('"the copy, not the reset" if C1 is SHOWN and C2 HOLDS' in doc), "",
    ("``the copy, not the reset'' if C1 is SHOWN pooled and C2 HOLDS",))
row(199, "per-machine reading", "per machine the reading is met on X",
    f"X: C1 {c1_state}, C2 d = {d2} ({c2_state}); but spec: 'Readings (pooled only) ... Otherwise the reading does not "
    "apply. Nothing else is read.'; the code prints '(the readings are pooled only)'", "PARTIAL",
    "Arithmetically true of X's per-machine claims, but there is no pre-registered per-machine reading; the sentence "
    "is not marked as outside the pre-registration.", ("per machine the reading is met on X",))

# ── Section memory ───────────────────────────────────────────────────────────
row(242, "C3", "C3, is SHOWN on X: the recipe 10/10 against one delta channel 0/10 (10 vs 0, p = 0.00098)",
    f"SPLIT BR {len(WIN['WIN3_SPLIT_D8'])}/10 vs SINGLE_DELTA_D8 BOUND {len(WIN['SINGLE_DELTA_D8'])}/10; "
    f"{c3[0]} vs {c3[1]}; p = {p3:.5f} -> {c3_state}",
    st(c3[:2] == (10, 0) and round(p3, 5) == 0.00098), "Outcomes as pre-registered: BOUND ROUTED vs BOUND.",
    ("is SHOWN on X: the recipe $10/10$ against one delta channel $0/10$ (10 vs 0, $p = 0.00098$",))
row(242, "single delta final accuracy", "final accuracy 0.08--0.76",
    f"SINGLE_DELTA_D8 final held-out acc {min(acc_d):.4f}-{max(acc_d):.4f} (s{S['SINGLE_DELTA_D8'][acc_d.index(min(acc_d))]}"
    f", s{S['SINGLE_DELTA_D8'][acc_d.index(max(acc_d))]})",
    st(f"{min(acc_d):.2f}" == "0.08" and f"{max(acc_d):.2f}" == "0.76"), "Printed, not a claim.",
    ("final accuracy $0.08$--$0.76$",))
row(242, "one Hebbian channel", "one Hebbian channel 0/10",
    f"SINGLE_HEBB_D8 BOUND {len(WIN['SINGLE_HEBB_D8'])}/10 (final acc {min(acc_h):.3f}-{max(acc_h):.3f})",
    st(len(WIN['SINGLE_HEBB_D8']) == 0), "Printed, not a claim (the spec's 'SINGLE_DELTA_D8 vs SINGLE_HEBB_D8' is "
    "descriptive); the draft lists it with C3 without that label.", ("one Hebbian channel $0/10$",))
row(242, "delta perfect gate", "the delta perfect gate 2/2 at 1200",
    f"CEIL_DELTA_D8 {len(WIN['CEIL_DELTA_D8'])}/2, transitions {ceil_t('CEIL_DELTA_D8')}",
    st(ceil_t('CEIL_DELTA_D8') == [1200, 1200]), "k = 16 perfect gate on delta memory (F's oracle was k = 8).",
    ("the delta perfect gate $2/2$ at 1200",))
row(242, "inference from validity", "so that the channel's failure is the missing partition and not the delta rule",
    "this is the content of pre-registered reading 2 (pooled only: C3 SHOWN pooled, SINGLE_DELTA below MAJORITY pooled, "
    "CEIL_DELTA bound on X and L)", "PARTIAL",
    f"X's conditions are met (C3 {c3_state}; SINGLE_DELTA {len(WIN['SINGLE_DELTA_D8'])}/10 < ceil(0.5n); CEIL_DELTA 2/2), "
    "and the same sentence says the reading waits for L, but the inference is stated as drawn.",
    ("so that the channel's failure is the missing partition and not the delta rule",))
row(242, "reading 2 quoted", "the reading ``the partition is necessary at eight streams against a single delta channel'' wait for L",
    "docstring READINGS: identical string, pooled only",
    st('"the partition is necessary at eight streams against a single delta channel"' in doc), "",
    ("the partition is necessary at eight streams against a single delta channel",))

# ── synthesis table ──────────────────────────────────────────────────────────
row(280, "synthesis: S=8 split column", "38/40; X 10/10^s",
    f"WIN3_SPLIT_D8 BR {len(WIN['WIN3_SPLIT_D8'])}/10", st(len(WIN['WIN3_SPLIT_D8']) == 10), "",
    ("X $10/10^{s}$",))
row(280, "synthesis: single channels", "one Hebbian channel 0/10 (X); one delta channel 0/10 (X)",
    f"SINGLE_HEBB_D8 {len(WIN['SINGLE_HEBB_D8'])}/10, SINGLE_DELTA_D8 {len(WIN['SINGLE_DELTA_D8'])}/10 (BOUND)",
    st(len(WIN['SINGLE_HEBB_D8']) == 0 == len(WIN['SINGLE_DELTA_D8'])),
    "Caption says 'bound routed, or discovered at S = 2'; the k = 1 cells are BOUND (k = 1 cannot route); no "
    "numeric effect here.", ("$0/10$ (X) & $0/10$ (X)",))
row(283, "synthesis caption: 'with test_split_copy's X half'", "with test_split_copy's X half and the screens",
    f"X half also has WIN3_SLOW_D8 (window gate + SLOW, no split) BR {len(WIN['WIN3_SLOW_D8'])}/10, not shown in the "
    "'window gate + SLOW' column (15/40 there is window_gate's)", "PARTIAL",
    "Selective: the split arm and the single channels of the X half are added, its no-split arm is not.",
    ("with \\texttt{test\\_split\\_copy}'s X half",))

# ── assessment and conclusion ────────────────────────────────────────────────
row(347, "assessment: copy not reset", "Pre-registered and shown on one machine (C1, 7 vs 0; C2 holds), pooled pending L",
    f"C1 {c1[0]} vs {c1[1]} {c1_state}; C2 {c2_state}", st(c1[:2] == (7, 0) and c2_state == "HOLDS"),
    "'Shown on one machine' = the secondary per-machine claims; the reading itself is pooled-only, and 'pooled "
    "pending L' says so.", ("Pre-registered and shown on one machine",))
row(351, "assessment: number of contexts", "0/10 in the pre-registered test on X (C3)",
    f"SINGLE_DELTA_D8 {len(WIN['SINGLE_DELTA_D8'])}/10", st(len(WIN['SINGLE_DELTA_D8']) == 0),
    "C3 is at S = 8 only; the four-vs-eight contrast ('number of contexts') rests on F's exploratory S68.",
    ("$0/10$ in the pre-registered test on X (C3)",))
row(371, "assessment: row copy breaks merges", "Supported ... and, on X, pre-registered",
    f"pre-registered: C1 (BOUND ROUTED) {c1_state}. Descriptive: {len(F_SPLIT)}/{len(F_SPLIT)} splits copied a channel "
    f"holding >= 2 streams; RESET failures {sorted(set(fail_oc['WIN3_RESET_D8']))}, SLOW failures "
    f"{sorted(set(fail_oc['WIN3_SLOW_D8']))}", "PARTIAL",
    "'breaks merges' is not a claim or reading of test_split_copy; the pre-registered support is C1 per machine (secondary), "
    "pooled pending. The merge evidence (failure classes, targeting) is printed, not pre-registered.",
    ("on X, pre-registered",))
row(459, "conclusion", "The split's row copy is the ingredient, and the reset alone is not enough; one channel, Hebbian "
    "or delta, cannot do what the partition does at eight streams.",
    "X only: C1 SHOWN, C2 HOLDS, C3 SHOWN (secondary); both readings pooled-only and not yet evaluated; one Hebbian "
    "channel is printed, not claimed", "PARTIAL",
    "Unscoped: no 'on X' or 'pending L'. The numbers on X support it.", ("The split's row copy is the ingredient",))

# ── provenance ───────────────────────────────────────────────────────────────
row(482, "provenance commits", "spec fb3acb7 / 1f52cd5 / 592c574",
    f"{cinfo(SPEC_C)} | {cinfo(TEST_C)} | {cinfo(RES)}; record meta.git = {meta['git']}; result commit adds the JSON "
    f"({rec_add}) and changes test_split_copy.py only by +{res_py[0]}/-{res_py[1]} docstring lines; code .py changed "
    f"{TEST_C}..{RES} (outside docs/): {py_changed}; JSON and test unchanged to head: "
    f"{rec_unchanged}",
    st(spec_add == [SPEC_C] and test_add == [TEST_C] and rec_add == [RES] and meta['git'] == TEST_C and res_doc_only
       and res_py[1] == "0" and rec_unchanged and py_changed == ["test_split_copy.py"]),
    "Between the test and result commits only docs/review_2026-10/recompute.py (the referee's, not test code) and the "
    "docstring of test_split_copy.py changed among .py files.",
    ("spec \\texttt{fb3acb7} / \\texttt{1f52cd5} / \\texttt{592c574}",))
row(482, "classifier / launcher", "the classifier ba8901a, the launcher 56c44a7",
    f"{cinfo('ba8901a')} | {cinfo('56c44a7')}; both ancestors of {TEST_C}: "
    f"{ancestor('ba8901a', TEST_C) and ancestor('56c44a7', TEST_C)}; test imports longrun: {uses_lr}",
    st(ancestor('ba8901a', TEST_C) and ancestor('56c44a7', TEST_C) and imports_fc2 and uses_lr),
    f"longrun.py was edited in {TEST_C} itself (+{lr_diff[0]}/-{lr_diff[1]}, the progress line only), so the launcher "
    f"that ran is {TEST_C}'s; 56c44a7 is its first commit.", ("the launcher \\texttt{56c44a7}",))
row(482, "runs X; L", "running & 54; ---", f"X records {n_runs}; no L file", st(n_runs == 54),
    "'running' (L) not record-checkable.", ("running & 54; ---",))
row(486, "X CPU for split_copy", "2.10 GHz for split_copy",
    f"meta.cpu '{meta['cpu']}'; per-record cpu {sorted(cpus)}", st(cpus == {meta['cpu']} and "2.10GHz" in meta['cpu']),
    "", ("2.10\\,GHz for \\texttt{split\\_copy}",))

# ── output ───────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
rows.sort(key=lambda r: (ORDER[r[4]], r[0]))


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


P = print
P("# Revision 9 draft vs test_split_copy on X — referee check\n")
P(f"Draft `{DRAFT}` at `{DRAFT_REF}` (origin/claude/rev9-draft); records `{REC}` at `{RES}` (identical at the branch "
  f"head `{rlib.git('rev-parse', '--short', HEAD).strip()}`: {rec_unchanged}); test docstring at `{RES}`, spec at "
  f"`{SPEC_C}`. Outcomes recomputed with `rlib` (transition from the curve, bound routed from the stream gates); they "
  f"agree with the records' own transition on every run ({agree_t}) and with the recorded one_to_one/stream_acc "
  f"reading ({agree_rec}); ch_map ties: {len(rlib.TIES)}. Generated by `check_split_copy.py`.\n")
cnt = {k: sum(1 for r in rows if r[4] == k) for k in ORDER}
P("Counts: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + f" (of {len(rows)} rows).\n")
P("| tex line | item | draft says | recomputed | status | note |")
P("|---|---|---|---|---|---|")
for n, item, d, rc, s, note in rows:
    P(f"| tex:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} |")

P("\n## Per-run records (X, seeds 560-569)\n")
P("Bound routed (Part A) / bound (Part B) with the transition; trigger firings `step:c*->c0` (all labelled ok unless marked).\n")
P("| seed | WIN3_SLOW_D8 | WIN3_RESET_D8 | WIN3_SPLIT_D8 | SINGLE_DELTA_D8 (final acc) | SINGLE_HEBB_D8 (final acc) |")
P("|---|---|---|---|---|---|")
for s in S["WIN3_SPLIT_D8"]:
    cells = []
    for a in ("WIN3_SLOW_D8", "WIN3_RESET_D8", "WIN3_SPLIT_D8"):
        r = R[a][s]
        lab = "BR" if s in WIN[a] else rlib.outcome_k(r)
        fs_ = [f"{c['step']}:{c['key_cs']}->{c['key_c0']}{'' if targeting(c['map_before'], c['key_cs'], c['key_c0']) else ' NOT'}"
               for c in r.get("checks", []) if c["fired"]]
        cells.append(f"{lab} ({r['transition']})" + (f"; {', '.join(fs_)}" if fs_ else ""))
    for a in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8"):
        r = R[a][s]
        cells.append(f"{'BOUND' if s in WIN[a] else 'not bound'} ({r['acc']:.3f})")
    P(f"| {s} | " + " | ".join(cells) + " |")
P(f"\nPerfect gates: CEIL_C {dict(zip(S['CEIL_C'], ceil_t('CEIL_C')))}, CEIL_DELTA_D8 "
  f"{dict(zip(S['CEIL_DELTA_D8'], ceil_t('CEIL_DELTA_D8')))}. WIN3_SPLIT_D8 transition median "
  f"{rlib.med(tr_split):.0f} ({min(tr_split)}-{max(tr_split)}). Bands (Wilson 95%): "
  + "; ".join(f"{a} {len(WIN[a])}/{len(S[a])} {rlib.band(len(WIN[a]), len(S[a]))} "
              f"[{rlib.wilson(len(WIN[a]), len(S[a]))[0]:.3f}, {rlib.wilson(len(WIN[a]), len(S[a]))[1]:.3f}]"
              for a in PART_A + ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8"))
  + f". fail_class v1/v2 conflict cases among unbound Part A runs (margin >= 0.25 and eta^2 by key >= 0.5): "
  f"{v12_conflict or 'none'} (docstring CHECK 151(b): no disagreement). Wall: sum of run secs {secs / 3600:.1f} h over "
  f"{meta['workers']} workers.")

P("\n## Readings vs docstring\n")
P("Pre-registered text (spec `fb3acb7` = docstring at `592c574`, identical meaning):\n")
for ln in doc.split("\n"):
    if ln.startswith(("- C1 ", "- C2 ", "- C3 ", "      <= 2 HOLDS", "READINGS", '- "the copy', '- "the partition',
                      "  below MAJORITY")):
        P(f"    {ln}")
P(f"""
- **C1** draft tex:199 "the split beats the reset alone" = C1. Outcome BOUND ROUTED for both arms (code `success`:
  `outcome_v2(r) == "BOUND ROUTED"` for Part A). X: {c1[0]} vs {c1[1]}, p = {p1:.4f}. Matches.
- **C2** draft tex:199 "the reset alone is not better than no trigger by more than two pairs" = C2, minus the word
  "pooled". The spec makes C2 a pooled bound and says "Per machine the same d is printed"; the code still labels the
  per-machine d HOLDS/FAILS, so "HOLDS" on X follows the code and the docstring. The abstract's (tex:36) "the reset alone
  no better than no trigger" is stronger than the bound; it is true of X's counts (d = {d2}), but it is a description,
  not C2.
- **C3** draft tex:242: BOUND ROUTED (SPLIT) vs BOUND (SINGLE_DELTA_D8), as pre-registered. X: {c3[0]} vs {c3[1]},
  p = {p3:.5f}. Matches.
- **Reading 1**, "the copy, not the reset", pooled only. tex:199 quotes the reading and its condition exactly and says it
  waits for L. It then adds "per machine the reading is met on X". The spec has no per-machine reading ("Readings
  (pooled only) ... Otherwise the reading does not apply. Nothing else is read.") and the code prints "(the readings are
  pooled only)". tex:36 ("the split's row copy is the ingredient") and tex:459 (unscoped) state the reading's content as
  a result. tex:347 is scoped correctly ("shown on one machine ... pooled pending L").
- **Reading 2**, "the partition is necessary at eight streams against a single delta channel", pooled only. tex:242 quotes it
  verbatim and defers it to L. On X its three conditions hold: C3 {c3_state}; SINGLE_DELTA_D8 {len(WIN['SINGLE_DELTA_D8'])}/10, below
  MAJORITY; CEIL_DELTA_D8 2/2. But tex:242's "so that the channel's failure is the missing partition and not the delta
  rule" and tex:459's "one channel, Hebbian or delta, cannot do what the partition does at eight streams" draw the
  reading's conclusion before L has run.
- Arms: copy + reset = WIN3_SPLIT_D8; reset only = WIN3_RESET_D8; none = WIN3_SLOW_D8. The draft maps all three
  correctly (tex:186/192). The "copy" is the copy with noise 0.1 std.
""")

P("## Descriptive statements: marked or not\n")
P("""The test's "printed, not claims" list covers SINGLE_DELTA vs SINGLE_HEBB, every split's c*->c0 and its labelling,
the control's resets, the perfect gates' transitions, the k = 1 final accuracies, the bands and the failure classes.
The draft's split_copy statements that are not pre-registered claims or readings:

| tex line | statement | correct on the records | marked as descriptive / not pre-registered? |
|---|---|---|---|""")
desc = [
    (36, "the reset alone no better than no trigger", f"yes ({c2[0]} vs {c2[1]})", "no: it sits inside 'a pre-registered result'"),
    (36, "the split's row copy is the ingredient", "C1/C2 hold on X", "no: the reading is pooled-only and not yet evaluated"),
    (195, "reset-only arm fired under the live rule, on labelled targets, bound no more often than none",
     f"yes ({len(F_RESET)} resets, {rec_ok_reset} labelled ok; {len(WIN['WIN3_RESET_D8'])} <= {len(WIN['WIN3_SLOW_D8'])})",
     "implicitly (caption); not labelled as printed-not-claim"),
    (199, "every seed the reset bound also bound without it", f"yes ({fs(WIN['WIN3_RESET_D8'])} in {fs(WIN['WIN3_SLOW_D8'])})",
     "no (parenthesis attached to C2)"),
    (199, "All 22 splits and all 27 resets on correctly labelled targets", f"yes ({rec_ok_split}/{len(F_SPLIT)}, {rec_ok_reset}/{len(F_RESET)})",
     "no"),
    (199, "so the difference is the row copy, not the aim", "an inference", "no"),
    (199, "per machine the reading is met on X", "arithmetically", "no: there is no pre-registered per-machine reading"),
    (242, "final accuracy 0.08-0.76", f"yes ({min(acc_d):.3f}-{max(acc_d):.3f})", "no (parenthesis to C3)"),
    (242, "one Hebbian channel 0/10", f"yes ({len(WIN['SINGLE_HEBB_D8'])}/10)", "no: listed with C3, though it is printed, not claimed"),
    (242, "the channel's failure is the missing partition and not the delta rule", "reading 2's content", "no; the same sentence defers the reading to L"),
    (280, "one Hebbian channel 0/10 (X)", "yes", "table cell, source footnote ^s only"),
    (371, "the row copy breaks merges ... on X, pre-registered", "C1 holds; merges are printed failure classes", "no: called pre-registered"),
    (459, "copy is the ingredient; one channel, Hebbian or delta, cannot", "on X", "no: unscoped"),
]
for n, s_, okk, m in desc:
    P(f"| tex:{n} | {esc(s_)} | {esc(okk)} | {esc(m)} |")
P(("\nNo number in the draft's split_copy statements disagrees with the records. " if cnt["DISAGREE"] == 0 else
   f"\n{cnt['DISAGREE']} row(s) DISAGREE with the records (above). ")
  + "The open points are scope: per-machine versus pooled, readings versus claims, and printed versus claimed.")
