"""check_split_copy_v5.py -- Referee V, fifth pass: the FIFTH version of the Revision 9 draft
(origin/claude/rev9-draft @ c5e4508, 19:14:19 UTC on 12 October, multichannel_hebbian_report_v9.tex, 568 lines) against
test_split_copy's records.

A copy of check_split_copy_v4.py (fourth pass, draft c66cc76) made to run against the fifth draft:
  * every row is re-anchored on the fifth draft by text search (tex5:NNN; an assertion fails if a fragment is missing);
    the X-half record and code checks are recomputed from the same pinned records (592c574). Rows whose wording c5e4508
    changed to the pooled result (L's half landed at ffb0771) are re-anchored on the new wording; their X part is
    recomputed as before and the L and pooled counts are recomputed LIGHTLY (counts, discordant pairs and exact p from
    L's results file with the same outcome functions; the CHECKs, firings and run details of L's half are another
    check's);
  * every fourth-pass row's fragment (row and row4 calls, read from check_split_copy_v4.py's source with ast) is asserted
    in c66cc76 and searched in c5e4508;
  * section 4 (not counted in the totals): test_delta_recipe's status lines against the main line as of the fifth draft
    (8ae660d), R's probe-mode count (now 13 of 282), and every line of the two provenance tables: every commit hash
    exists, predates c5e4508 and is what the row says; each session's last cited commit against its branch's last commit
    before 19:14:19 UTC on 12 October.

Inputs, all read with git at pinned refs (nothing is run, nothing is written but stdout):
  draft   9bfa85a (first), c66cc76 (fourth), c5e4508 (fifth): multichannel_hebbian_report_v9.tex
  records 592c574:results/X/split_copy_results.json; ffb0771:results/L/split_copy_results.json; main line as of
          c5e4508: results/{X,L}/window_gate_results.json; replicate-R as of c5e4508: records/phase{2,3,4}.jsonl
  test    592c574:test_split_copy.py, 8ae660d:test_delta_recipe.py, spec fb3acb7:specs/test_split_copy.md

    python3 docs/review_2026-10/rev9/check_split_copy_v5.py > docs/review_2026-10/rev9/check_split_copy_v5.md
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402

OLD_REF, V2_REF, V3_REF, V4_REF, NEW_REF, DRAFT = "9bfa85a", "f67bc07", "a5f67a2", "c66cc76", "c5e4508", "multichannel_hebbian_report_v9.tex"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
CUTOFF = rlib.git("show", "-s", "--format=%cI", NEW_REF).strip()          # the fifth draft's commit time
HEAD = rlib.git("rev-list", "-1", f"--before={CUTOFF}", MAIN).strip()[:7]  # the main line as of the fifth draft
RES, REC = "592c574", "results/X/split_copy_results.json"
SPEC_C, TEST_C = "fb3acb7", "1f52cd5"
ARMS = ("CEIL_C", "CEIL_DELTA_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8")
PART_A = ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")

tex1 = rlib.git("show", f"{OLD_REF}:{DRAFT}").split("\n")
tex2 = rlib.git("show", f"{V2_REF}:{DRAFT}").split("\n")
tex3 = rlib.git("show", f"{V3_REF}:{DRAFT}").split("\n")
tex4 = rlib.git("show", f"{V4_REF}:{DRAFT}").split("\n")
tex5 = rlib.git("show", f"{NEW_REF}:{DRAFT}").split("\n")
doc = rlib.git("show", f"{RES}:test_split_copy.py")
spec = rlib.git("show", f"{SPEC_C}:specs/test_split_copy.md")
twg = rlib.git("show", f"{HEAD}:test_window_gate.py")
store = rlib.load(RES, REC)
R = {a: rlib.runs(store, a) for a in ARMS}
HEAD_SHORT = rlib.git("rev-parse", "--short", HEAD).strip()
# L's half (landed after the fourth pass): counts only, recomputed lightly with the same outcome functions
L_RES = "ffb0771"
LSTORE = rlib.load(L_RES, "results/L/split_copy_results.json")
RL = {a: rlib.runs(LSTORE, a) for a in ARMS}


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
# L and pooled (light: counts, discordant pairs, exact p; the same outcome functions as X's)
SL = {a: sorted(RL[a]) for a in ARMS}
WINL = {a: [s for s in SL[a] if succ(a, RL[a][s])] for a in ARMS}
nL = {a: len(WINL[a]) for a in ARMS}
okL = all(rlib.ok(r) for a in ARMS for r in RL[a].values())
nL_runs = sum(len(RL[a]) for a in ARMS)
c1L, c2L, c3L = (pair(x, y, WINL, SL) for x, y in (("WIN3_SPLIT_D8", "WIN3_RESET_D8"), ("WIN3_RESET_D8", "WIN3_SLOW_D8"),
                                                    ("WIN3_SPLIT_D8", "SINGLE_DELTA_D8")))
c1P, c2P, c3P = (tuple(x + y for x, y in zip(a_, b_)) for a_, b_ in ((c1, c1L), (c2, c2L), (c3, c3L)))
p1L, p3L, p1P, p3P = (rlib.mcnemar1(x[0], x[1]) for x in (c1L, c3L, c1P, c3P))
d2L, d2P = c2L[0] - c2L[1], c2P[0] - c2P[1]
c1L_state, c3L_state = ("SHOWN" if p1L < 0.05 else "not shown"), ("SHOWN" if p3L < 0.05 else "not shown")
c2L_state = "HOLDS" if d2L <= 2 else "FAILS"
c1P_state, c3P_state = ("SHOWN" if p1P < 0.05 else "not shown"), ("SHOWN" if p3P < 0.05 else "not shown")
c2P_state = "HOLDS" if d2P <= 2 else "FAILS"
disjoint = not (set(S["WIN3_SPLIT_D8"]) & set(SL["WIN3_SPLIT_D8"]))
ceilL_t = {a: [RL[a][s]["transition"] for s in SL[a]] for a in ("CEIL_C", "CEIL_DELTA_D8")}
reading1P = c1P_state == "SHOWN" and c2P_state == "HOLDS"
sdP = len(WIN["SINGLE_DELTA_D8"]) + len(WINL["SINGLE_DELTA_D8"])
sdN = len(S["SINGLE_DELTA_D8"]) + len(SL["SINGLE_DELTA_D8"])
validB_both = len(WIN["CEIL_DELTA_D8"]) == len(S["CEIL_DELTA_D8"]) == 2 and len(WINL["CEIL_DELTA_D8"]) == len(SL["CEIL_DELTA_D8"]) == 2
reading2P = c3P_state == "SHOWN" and sdP < -(-sdN // 2) and validB_both
acc_dL = [RL["SINGLE_DELTA_D8"][s]["acc"] for s in SL["SINGLE_DELTA_D8"]]
resetL_eq_slow = set(WINL["WIN3_RESET_D8"]) == set(WINL["WIN3_SLOW_D8"])
L_git = LSTORE["meta"].get("git")
L_add = rlib.git("log", "--diff-filter=A", "--format=%h", L_RES, "--", "results/L/split_copy_results.json").split()


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
TIP = {r: rlib.git("rev-list", "-1", f"--before={CUTOFF}", r).strip()[:7] for r in refs}   # each branch as of c66cc76
TIP = {r: t for r, t in TIP.items() if t}
TREE = {r: rlib.git("ls-tree", "-r", "--name-only", t).split("\n") for r, t in TIP.items()}
l_file_on = [r for r in TIP if L_REC in TREE[r]]
l_ever = rlib.git("log", "--all", "--format=%h", "--", L_REC).split()
# any L split_copy artefact (results/L/**split_copy**) on any branch as of the cutoff
L_ANY = {r: [f for f in TREE[r] if f.startswith("results/L/") and "split_copy" in f] for r in TIP}
L_ANY = {r: v for r, v in L_ANY.items() if v}
L_ATTEMPT = None
for r, fs_ in sorted(L_ANY.items()):
    if "results/L/longrun/split_copy_L.json" in fs_:
        import json as _json
        lj = _json.loads(rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.json"))
        lexit = rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.exit").strip()
        llog = rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.log")
        lc = rlib.git("log", "-1", "--format=%h %cI", TIP[r], "--", "results/L/longrun/split_copy_L.json").strip()
        st0 = lj["starts"][0]
        utc = dt.datetime.fromtimestamp(st0["t"], dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        L_ATTEMPT = dict(ref=r, tip=TIP[r], commit=lc, exit=lexit, started=st0["started"], started_utc=utc,
                         git=st0["git"], cmd=" ".join(lj["cmd"]), total=lj["total"],
                         failed="SOME VERIFICATION CHECKS FAILED" in llog and "verification failed" in llog,
                         check155_wrong="CHECK 155" in llog and "-> WRONG" in llog.split("CHECK 155")[1].split("CHECK 156")[0],
                         run_lines=sum(1 for ln in llog.split("\n") if ln.startswith(("WIN3_", "CEIL_", "SINGLE_")) and " seed " in ln))

l_no_runs = not l_file_on and (L_ATTEMPT is None or (L_ATTEMPT["failed"] and L_ATTEMPT["exit"] == "1"))
LA_TXT = ("no L split_copy artefact on any branch" if L_ATTEMPT is None else
          f"L's one start: {L_ATTEMPT['ref'].replace('origin/', '')} @ {L_ATTEMPT['commit']} "
          f"(results/L/longrun/split_copy_L.{{json,log,exit}}): started {L_ATTEMPT['started']} local = "
          f"{L_ATTEMPT['started_utc']} UTC (epoch in the file), code {L_ATTEMPT['git']}, `{L_ATTEMPT['cmd']}`; "
          f"verification failed {L_ATTEMPT['failed']} (CHECK 155 WRONG: {L_ATTEMPT['check155_wrong']}, the --also path "
          f"results/X/ is a directory), exit {L_ATTEMPT['exit']}, run lines {L_ATTEMPT['run_lines']} of {L_ATTEMPT['total']}")

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
rows = []   # (tex4 line, item, draft, recomputed, status, note, old indices)


def row(phrases, item, draft, recomputed, status, note="", old=()):
    n = locate(tex5, phrases)
    assert n is not None, (item, phrases)
    rows.append((n, item, draft, recomputed, status, note, tuple(old)))


nA = len(S["WIN3_SPLIT_D8"])
nW = {a: len(WIN[a]) for a in ARMS}
nP = {a: nW[a] + nL[a] for a in ARMS}
# abstract and front matter
row(("the copy with its reset binds eight streams $18/20$ against the reset alone $6/20$ (12 vs 0)",),
    "abstract: C1 counts (re-anchored: pooled)", "the copy with its reset binds eight streams 18/20 against the reset alone 6/20 (12 vs 0)",
    f"X: SPLIT {nW['WIN3_SPLIT_D8']}/10, RESET {nW['WIN3_RESET_D8']}/10, SLOW {nW['WIN3_SLOW_D8']}/10, SINGLE_DELTA {nW['SINGLE_DELTA_D8']}/10; L (light): {nL['WIN3_SPLIT_D8']}/10, {nL['WIN3_RESET_D8']}/10, {nL['WIN3_SLOW_D8']}/10, {nL['SINGLE_DELTA_D8']}/10; pooled {nP['WIN3_SPLIT_D8']}/20, {nP['WIN3_RESET_D8']}/20, {nP['WIN3_SLOW_D8']}/20, {nP['SINGLE_DELTA_D8']}/20; C1 X {c1[0]} vs {c1[1]}, L {c1L[0]} vs {c1L[1]}, pooled {c1P[0]} vs {c1P[1]} (p = {p1P:.5f}); C2 pooled {c2P[0]} vs {c2P[1]} (d = {d2P}); C3 pooled {c3P[0]} vs {c3P[1]} (p = {p3P:.2e})",
    st((nW['WIN3_SPLIT_D8'], nW['WIN3_RESET_D8'], c1[0], c1[1]) == (10, 3, 7, 0)
       and (nP['WIN3_SPLIT_D8'], nP['WIN3_RESET_D8'], c1P[0], c1P[1]) == (18, 6, 12, 0)), "X part recomputed as before; L and pooled counts recomputed lightly (L's half in depth is another check's).", old=(0,))
row(("the reset alone is within the bound of no trigger ($6/20$ against $8/20$, 0 vs 2)",),
    "abstract: reset vs no trigger (re-anchored: pooled)", "the reset alone is within the bound of no trigger (6/20 against 8/20, 0 vs 2)",
    f"X: SPLIT {nW['WIN3_SPLIT_D8']}/10, RESET {nW['WIN3_RESET_D8']}/10, SLOW {nW['WIN3_SLOW_D8']}/10, SINGLE_DELTA {nW['SINGLE_DELTA_D8']}/10; L (light): {nL['WIN3_SPLIT_D8']}/10, {nL['WIN3_RESET_D8']}/10, {nL['WIN3_SLOW_D8']}/10, {nL['SINGLE_DELTA_D8']}/10; pooled {nP['WIN3_SPLIT_D8']}/20, {nP['WIN3_RESET_D8']}/20, {nP['WIN3_SLOW_D8']}/20, {nP['SINGLE_DELTA_D8']}/20; C1 X {c1[0]} vs {c1[1]}, L {c1L[0]} vs {c1L[1]}, pooled {c1P[0]} vs {c1P[1]} (p = {p1P:.5f}); C2 pooled {c2P[0]} vs {c2P[1]} (d = {d2P}); C3 pooled {c3P[0]} vs {c3P[1]} (p = {p3P:.2e})", st(c2_state == "HOLDS" and (nW['WIN3_RESET_D8'], nW['WIN3_SLOW_D8']) == (3, 5)
             and (nP['WIN3_RESET_D8'], nP['WIN3_SLOW_D8'], c2P[0], c2P[1]) == (6, 8, 0, 2) and c2P_state == "HOLDS"),
    "C2's own wording (a bound). X part recomputed as before; L and pooled counts recomputed lightly (L's half in depth is another check's).", old=(1,))
row(("one delta-rule channel binds $0/20$ where the recipe binds $18/20$ (18 vs 0)",), "abstract: C3 counts (re-anchored: pooled)",
    "one delta-rule channel binds 0/20 where the recipe binds 18/20 (18 vs 0)", f"X: SPLIT {nW['WIN3_SPLIT_D8']}/10, RESET {nW['WIN3_RESET_D8']}/10, SLOW {nW['WIN3_SLOW_D8']}/10, SINGLE_DELTA {nW['SINGLE_DELTA_D8']}/10; L (light): {nL['WIN3_SPLIT_D8']}/10, {nL['WIN3_RESET_D8']}/10, {nL['WIN3_SLOW_D8']}/10, {nL['SINGLE_DELTA_D8']}/10; pooled {nP['WIN3_SPLIT_D8']}/20, {nP['WIN3_RESET_D8']}/20, {nP['WIN3_SLOW_D8']}/20, {nP['SINGLE_DELTA_D8']}/20; C1 X {c1[0]} vs {c1[1]}, L {c1L[0]} vs {c1L[1]}, pooled {c1P[0]} vs {c1P[1]} (p = {p1P:.5f}); C2 pooled {c2P[0]} vs {c2P[1]} (d = {d2P}); C3 pooled {c3P[0]} vs {c3P[1]} (p = {p3P:.2e})",
    st(nW['SINGLE_DELTA_D8'] == 0 and nW['WIN3_SPLIT_D8'] == 10 and (nP['SINGLE_DELTA_D8'], nP['WIN3_SPLIT_D8'], c3P[0], c3P[1]) == (0, 18, 18, 0)),
    "X part recomputed as before; L and pooled counts recomputed lightly (L's half in depth is another check's).", old=(2,))
row(("is now a pre-registered result on two machines with disjoint seeds",
     "the pre-fixed readings ``the copy, not the reset'' and ``the partition is necessary at eight streams against a single delta channel'' both apply"),
    "abstract: scope of the result (re-anchored)",
    "the second claim of Revision 8 is now a pre-registered result on two machines with disjoint seeds ... the pre-fixed "
    "readings 'the copy, not the reset' and 'the partition is necessary ...' both apply",
    f"seeds X {S['WIN3_SPLIT_D8'][0]}-{S['WIN3_SPLIT_D8'][-1]}, L {SL['WIN3_SPLIT_D8'][0]}-{SL['WIN3_SPLIT_D8'][-1]} (disjoint: {disjoint}); "
    f"pooled C1 {c1P_state}, C2 {c2P_state}, C3 {c3P_state}; reading 1 (C1 SHOWN and C2 HOLDS, pooled) {reading1P}; reading 2 "
    f"(C3 SHOWN, SINGLE_DELTA {sdP}/{sdN} below MAJORITY, CEIL_DELTA bound on both machines {validB_both}) {reading2P}; L results "
    f"{L_RES} ({L_add}), {nL_runs} records, all ok {okL}",
    st(disjoint and reading1P and reading2P and okL and nL_runs == 54),
    "The readings recomputed with the docstring's rules (test_split_copy.py READINGS, pooled only) on the light counts. "
    "'The row copy is the ingredient' stays out of the abstract.", old=(3,))
row(("\\texttt{test\\_split\\_copy}, complete on both machines",), "Rev. 9 paragraph (re-anchored)",
    "test_split_copy, complete on both machines",
    f"X file: {n_runs} records, all ok = {ok_all()}; L file at {L_RES} (added {L_add}, before the draft: "
    f"{ctime(L_RES) < ctime(NEW_REF)}): {nL_runs} records, all ok {okL}, meta.git {L_git}",
    st(n_runs == 54 and ok_all() and nL_runs == 54 and okL and ctime(L_RES) < ctime(NEW_REF)),
    "L's half ran at 592c574 (meta.git) after the earlier failed start.", old=(4,))
row(("C1, C3 SHOWN and C2 HOLDS pooled and on each machine; ``the copy, not the reset''; ``the partition is necessary at eight streams against a single delta channel''",),
    "index row (re-anchored)", "split_copy (X, L, pooled): C1, C3 SHOWN and C2 HOLDS pooled and on each machine; both readings",
    f"X: C1 {c1_state} (p={p1:.4f}), C3 {c3_state} (p={p3:.5f}), C2 {c2_state} (d={d2}); L (light): C1 {c1L_state} (p={p1L:.4f}), "
    f"C3 {c3L_state} (p={p3L:.5f}), C2 {c2L_state} (d={d2L}); pooled: C1 {c1P_state}, C3 {c3P_state}, C2 {c2P_state}; readings "
    f"{reading1P}, {reading2P}",
    st((c1_state, c2_state, c3_state) == ("SHOWN", "HOLDS", "SHOWN") and (c1L_state, c2L_state, c3L_state) == ("SHOWN", "HOLDS", "SHOWN")
       and (c1P_state, c2P_state, c3P_state) == ("SHOWN", "HOLDS", "SHOWN") and reading1P and reading2P), "X part recomputed as before; L and pooled counts recomputed lightly (L's half in depth is another check's).", old=(5,))
row(("\\texttt{fail\\_class\\_v2} (commit \\texttt{ba8901a})", "used by every test from \\texttt{test\\_split\\_copy} on"),
    "classifier commit", "fail_class_v2 (commit ba8901a) ... used by every test from test_split_copy on",
    f"{cinfo('ba8901a')}; test_split_copy imports fail_class_v2: {imports_fc2}; fail_class_v2.py unchanged "
    f"ba8901a..{TEST_C}: {fc2_same}; ancestor of {TEST_C}: {ancestor('ba8901a', TEST_C)}",
    st(imports_fc2 and fc2_same and ancestor('ba8901a', TEST_C)), "", old=(6,))
# Section split
row(("Two further looks settle it, one of them pre-registered on two machines",), "'settle it' (re-anchored)",
    "Two further looks settle it, one of them pre-registered on two machines.",
    f"X complete ({n_runs} records, all ok); L complete ({nL_runs} records, all ok {okL}); pooled C1 {c1P_state}, C2 {c2P_state}",
    st(n_runs == 54 and ok_all() and nL_runs == 54 and okL and c1P_state == "SHOWN"),
    "'Settle' is back (the first pass's PARTIAL was on 'settle the question' with one machine). 'It' is that the reset alone "
    "is not enough; the pre-registered C1 is now SHOWN pooled over two machines, and the other look (the window_gate "
    "control) agrees, so the word is now backed. Wording only.", old=(7,))
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
row(("\\texttt{test\\_split\\_copy}'s pooled claims are primary and its readings are read pooled only",), "caption: pooled primary, readings pooled only (re-anchored)",
    "test_split_copy's pooled claims are primary and its readings are read pooled only",
    "docstring: 'POOLED over both machines PRIMARY'; 'READINGS (pooled only)'",
    st("POOLED over both machines PRIMARY" in doc and "READINGS (pooled only)" in doc), "", old=(15,))
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
row(("every CHECK passed except two whose records another CPU wrote, printed rather than asserted, one of them (130) differing",),
    "CHECKs (re-anchored)", "every CHECK passed except two whose records another CPU wrote, printed rather than asserted, "
    "one of them (130) differing",
    f"docstring RESULT: 'every inherited and own CHECK passed. test_muon_recipe's CHECK 130 (Muon, X's 2.80GHz "
    f"test_early_recipe file) differs here and was printed, not asserted' ({doc_130}); 'CHECK 146 reproduced the 2.80GHz "
    f"test_window_gate records ... bit for bit through 7200 (printed: other CPU)' ({doc_146}); split_copy ran on {sorted(cpus)}",
    st(doc_130 and doc_146 and cpus == {"Intel(R) Xeon(R) Processor @ 2.10GHz"}),
    "Reworded in a5f67a2: the second pass's nuance (CHECK 130 'differs here'; CHECK 146 reproduced bit for bit) is now "
    "carried. 'every CHECK passed except two' still reads as if the two did not pass; the docstring says every CHECK "
    "passed and two were printed rather than asserted. Wording only.",
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
row(("the pre-fixed reading ``the copy, not the reset'' applies",), "reading 1 applies (re-anchored)",
    "the pre-fixed reading 'the copy, not the reset' applies",
    f"docstring READINGS (pooled only): '\"the copy, not the reset\" if C1 is SHOWN and C2 HOLDS'; pooled C1 {c1P_state} "
    f"({c1P[0]} vs {c1P[1]}, p = {p1P:.5f}), C2 {c2P_state} (d = {d2P}) -> {reading1P}",
    st('"the copy, not the reset" if C1 is SHOWN and C2 HOLDS' in doc and reading1P), "Light pooled counts.", old=(27,))
row(("The reset alone bound the same seeds as no trigger on L and a subset of them on X",),
    "reset seeds against none seeds, per machine (re-anchored)", "The reset alone bound the same seeds as no trigger on L and a subset of them on X.",
    f"X: RESET {fs(WIN['WIN3_RESET_D8'])} subset of SLOW {fs(WIN['WIN3_SLOW_D8'])}: {reset_sub_slow}; L: RESET {fs(WINL['WIN3_RESET_D8'])}, "
    f"SLOW {fs(WINL['WIN3_SLOW_D8'])}, equal: {resetL_eq_slow}",
    st(reset_sub_slow and resetL_eq_slow and set(WIN['WIN3_RESET_D8']) != set(WIN['WIN3_SLOW_D8'])),
    "Replaces 'the readings are pooled only, and on X alone their conditions hold as secondary claims'. Light on L.", old=(28,))
# Section memory
row(("is SHOWN pooled over both machines: the recipe $18/20$ against one delta channel $0/20$ (18 vs 0, $p = 3.8\\times10^{-6}$; X 10 vs 0, L 8 vs 0",),
    "C3 (re-anchored: pooled)", "C3 is SHOWN pooled over both machines: the recipe 18/20 against one delta channel 0/20 (18 vs 0, p = 3.8e-6; X 10 vs 0, L 8 vs 0)",
    f"X: SPLIT BR {nW['WIN3_SPLIT_D8']}/10 vs SINGLE_DELTA_D8 BOUND {nW['SINGLE_DELTA_D8']}/10, {c3[0]} vs {c3[1]}, p = {p3:.5f}; L (light): "
    f"{nL['WIN3_SPLIT_D8']}/10 vs {nL['SINGLE_DELTA_D8']}/10, {c3L[0]} vs {c3L[1]}; pooled {c3P[0]} vs {c3P[1]}, p = {p3P:.3e} -> {c3P_state}",
    st(c3[:2] == (10, 0) and c3L[:2] == (8, 0) and c3P[:2] == (18, 0) and f"{p3P:.1e}" == "3.8e-06"), "Light pooled counts.", old=(29,))
row(("final accuracy $0.08$--$0.76$",), "single delta final accuracy", "final accuracy 0.08--0.76",
    f"SINGLE_DELTA_D8 final acc {min(acc_d):.4f}-{max(acc_d):.4f}",
    st(f"{min(acc_d):.2f}" == "0.08" and f"{max(acc_d):.2f}" == "0.76"), "", old=(30,))
row(("one Hebbian channel $0/20$ (printed, not claimed)",), "one Hebbian channel (re-anchored: pooled)",
    "one Hebbian channel 0/20 (printed, not claimed)",
    f"SINGLE_HEBB_D8 BOUND X {nW['SINGLE_HEBB_D8']}/10, L {nL['SINGLE_HEBB_D8']}/10; docstring PRINTED, NOT CLAIMS lists it: "
    f"{'SINGLE_DELTA_D8 vs SINGLE_HEBB_D8' in doc}", st(nW['SINGLE_HEBB_D8'] == 0 == nL['SINGLE_HEBB_D8']), "", old=(31,))
row(("the delta perfect gate $4/4$ at 1200 on both machines",), "delta perfect gate (re-anchored: both machines)",
    "the delta perfect gate 4/4 at 1200 on both machines",
    f"CEIL_DELTA_D8 X {ceil_t('CEIL_DELTA_D8')}, L {ceilL_t['CEIL_DELTA_D8']}",
    st(ceil_t('CEIL_DELTA_D8') == [1200, 1200] and ceilL_t['CEIL_DELTA_D8'] == [1200, 1200]), "", old=(32,))
row(("the pre-fixed reading ``the partition is necessary at eight streams against a single delta channel'' applies, which is what makes the channel's failure the missing partition rather than the delta rule",),
    "reading 2 applies, and its content (re-anchored)", "the pre-fixed reading 'the partition is necessary at eight streams against a "
    "single delta channel' applies, which is what makes the channel's failure the missing partition rather than the delta rule",
    f"docstring READINGS: C3 SHOWN pooled ({c3P_state}), SINGLE_DELTA_D8 below MAJORITY pooled ({sdP}/{sdN}), CEIL_DELTA_D8 bound on "
    f"both machines ({validB_both}) -> {reading2P}",
    st('"the partition is necessary at eight streams against a single delta channel"' in doc and reading2P),
    "The inference ('what makes the channel's failure the missing partition') is the reading's own content, now that it applies.", old=(33, 34))
# synthesis table
row(("$15/40$; $8/20^{s}$", "$\\mathbf{18/20}^{s}$"), "synthesis S=8 row: window gate + SLOW and split columns (re-anchored: pooled)",
    "15/40; 8/20^s  |  38/40; 18/20^s",
    f"WIN3_SLOW_D8 BR pooled {nP['WIN3_SLOW_D8']}/20 (X {nW['WIN3_SLOW_D8']}, L {nL['WIN3_SLOW_D8']}); WIN3_SPLIT_D8 pooled {nP['WIN3_SPLIT_D8']}/20",
    st(nP['WIN3_SLOW_D8'] == 8 and nP['WIN3_SPLIT_D8'] == 18), "Light pooled counts.", old=(35, 37))
row(("$0/20^{s}$ & $0/20^{s}$",), "synthesis: single channels (re-anchored: pooled)",
    "one Hebbian channel 0/20^s; one delta channel 0/20^s",
    f"SINGLE_HEBB_D8 {nP['SINGLE_HEBB_D8']}/20, SINGLE_DELTA_D8 {nP['SINGLE_DELTA_D8']}/20 (BOUND)",
    st(nP['SINGLE_HEBB_D8'] == 0 == nP['SINGLE_DELTA_D8']), "k = 1 cells are BOUND (k = 1 cannot route).", old=(36,))
# assessment and conclusion
row(("Pre-registered and shown}, pooled over two machines with disjoint seeds (C1 12 vs 0, $p = 0.00024$; C2 holds, 0 vs 2) and on each machine; the reading applies",),
    "assessment: copy not reset (re-anchored: pooled)",
    "Pre-registered and shown, pooled over two machines with disjoint seeds (C1 12 vs 0, p = 0.00024; C2 holds, 0 vs 2) and on each machine; the reading applies",
    f"pooled C1 {c1P[0]} vs {c1P[1]} (p = {p1P:.5f}) {c1P_state}; C2 {c2P[0]} vs {c2P[1]} {c2P_state}; per machine C1 X {c1_state}, L {c1L_state}; "
    f"C2 X {c2_state}, L {c2L_state}; reading 1 {reading1P}",
    st(c1P[:2] == (12, 0) and f"{p1P:.5f}" == "0.00024" and c2P[:2] == (0, 2) and c1_state == c1L_state == "SHOWN"
       and c2_state == c2L_state == "HOLDS" and reading1P), "Light pooled counts.", old=(38,))
row(("\\textbf{Supported}, pooled and on each machine, by the pre-registered C3 ($0/20$ against the recipe's $18/20$, 18 vs 0)",),
    "assessment: number of contexts (re-anchored: pooled)", "Supported, pooled and on each machine, by the pre-registered C3 (0/20 "
    "against the recipe's 18/20, 18 vs 0); session F's S68 had 0/10 at eight streams too (2/10 at four)",
    f"pooled SINGLE_DELTA {nP['SINGLE_DELTA_D8']}/20 vs SPLIT {nP['WIN3_SPLIT_D8']}/20, {c3P[0]} vs {c3P[1]}; C3 X {c3_state}, L {c3L_state}; "
    f"tex5 says 'had $0/10$ at eight streams too': {locate(tex5, ('S68 had $0/10$ at eight streams too ($2/10$ at four)',)) is not None}",
    st(nP['SINGLE_DELTA_D8'] == 0 and nP['WIN3_SPLIT_D8'] == 18 and c3P[:2] == (18, 0) and c3_state == c3L_state == "SHOWN"
       and locate(tex5, ('S68 had $0/10$ at eight streams too ($2/10$ at four)',)) is not None),
    "Light pooled counts; S68's counts are check_F_v5's.", old=(39,))
row(("by the pre-registered C1 (the copy with its reset against the reset alone, 7 vs 0)",
     "that the runs it rescues are merges is printed in their failure classes, not claimed"),
    "assessment: row copy breaks merges", "Supported ... and, on X, by the pre-registered C1 (... 7 vs 0); that the runs it "
    "rescues are merges is printed in their failure classes, not claimed",
    f"C1 {c1[0]} vs {c1[1]} {c1_state}; the reset arm's outcome on the {len(c1_seeds)} rescued seeds: "
    f"{sorted(set(c1_reset_oc.values()))}",
    st(c1[:2] == (7, 0) and all(v.startswith("MERGED") for v in c1_reset_oc.values())),
    "C1 is now the pre-registered support; the merge reading is marked as printed.", old=(40,))
row(("Pre-registered on two machines with disjoint seeds, the copy with its reset beats the reset alone (18/20 against 6/20)",
     "one delta channel binds none of twenty seeds where the recipe binds eighteen; both readings apply"), "conclusion (re-anchored: pooled)",
    "Pre-registered on two machines with disjoint seeds, the copy with its reset beats the reset alone (18/20 against 6/20), the "
    "reset alone is within the bound of no trigger, and one delta channel binds none of twenty seeds where the recipe binds "
    "eighteen; both readings apply.",
    f"pooled SPLIT {nP['WIN3_SPLIT_D8']}/20, RESET {nP['WIN3_RESET_D8']}/20, C1 {c1P_state}, C2 {c2P_state}, SINGLE_DELTA "
    f"{nP['SINGLE_DELTA_D8']}/20, C3 {c3P_state}; readings {reading1P}, {reading2P}",
    st((nP['WIN3_SPLIT_D8'], nP['WIN3_RESET_D8'], nP['SINGLE_DELTA_D8']) == (18, 6, 0)
       and (c1P_state, c2P_state, c3P_state) == ("SHOWN", "HOLDS", "SHOWN") and reading1P and reading2P), "Light pooled counts.", old=(41,))
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
row(("\\texttt{ffb0771} (L and pooled;", "& complete (L run at \\texttt{592c574}; one earlier start stopped at verification over an argument) & 54; 54"),
    "provenance table: L column and runs (re-anchored on c5e4508)",
    "split_copy ... ffb0771 (L and pooled; ...) & complete (L run at 592c574; one earlier start stopped at verification over an argument) & 54; 54",
    f"X records {n_runs}; L results file added at {L_add} ({cinfo(L_RES)}), {nL_runs} records, all ok {okL}, meta.git {L_git}; the "
    f"earlier start: {LA_TXT}",
    st(n_runs == 54 and nL_runs == 54 and okL and L_git == "592c574" and L_add == [L_RES] and L_ATTEMPT is not None
       and L_ATTEMPT["failed"] and L_ATTEMPT["run_lines"] == 0),
    "Supersedes the fourth draft's 'not run'. 'Complete' and 'L run at 592c574' match L's file; the earlier start is the failed "
    "one at 84a7e57 (CHECK 155 WRONG, exit 1, no run line).", old=(44,))
row(("2.10\\,GHz for \\texttt{split\\_copy}",), "X CPU for split_copy", "2.10 GHz for split_copy",
    f"meta.cpu '{meta['cpu']}'; per-record {sorted(cpus)}", st(cpus == {meta['cpu']} and "2.10GHz" in meta['cpu']), "",
    old=(45,))

# ── third-pass rows (read from check_split_copy_v3.py's source) located in the fourth draft ───────────────
import ast  # noqa: E402
import json as _json  # noqa: E402

V3_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_split_copy_v3.py")).read()
V3_ROWS = []    # (kind, item, phrases): kind "row" (counted, section 1 of v3) or "row4" (section 4 of v3)
for node in ast.walk(ast.parse(V3_SRC)):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("row", "row4"):
        ph = tuple(ast.literal_eval(node.args[0]))
        V3_ROWS.append((node.func.id, ast.literal_eval(node.args[1]), ph))
V3_ROWS.sort(key=lambda x: (x[0], locate(tex3, x[2]) or 0))
for k_, it, ph in V3_ROWS:
    assert locate(tex3, ph) is not None, (it, ph)          # every third-pass fragment is in a5f67a2


# ── section 4: provenance and main-line rows c66cc76 changed, and the delta_recipe statements ─────────────
def tip(ref):
    return rlib.git("rev-list", "-1", f"--before={CUTOFF}", ref).strip()[:7]


def at(ref, path):
    try:
        return rlib.git("show", f"{ref}:{path}")
    except Exception:
        return None


def subj(c):
    return rlib.git("show", "-s", "--format=%s", c).strip()


def on(c, br):
    return ancestor(c, br)


def tstamp(c):
    return rlib.git("show", "-s", "--format=%cI", c).strip()


def added_in(c, path_sub):
    return [f for f in rlib.git("show", "--name-status", "--format=", c).split("\n") if f and path_sub in f]


rows4 = []   # (tex4 line, item, draft says, record, status, note)


def row4(phrases, item, draft, record, status, note=""):
    n = locate(tex5, phrases)
    assert n is not None, (item, phrases)
    rows4.append((n, item, draft, record, status, note))


# test_delta_recipe: when X's run started (the main line's own statement, committed after this version)
DR = {c: cinfo(c) for c in ("05256c5", "fad1489", "32c121b", "a581027", "16b04f9", "6f2788d", "55dac50")}
DR_PATHS = {c: rlib.git("show", "--name-only", "--format=", c).split() for c in DR}
ALL_HEADS = {r: rlib.git("rev-parse", "--short=7", r).strip() for r in refs}
DR_RES_NOW = {r: [f for f in rlib.git("ls-tree", "-r", "--name-only", h).split("\n")
                  if "delta_recipe" in f and (f.startswith("results/") or f.endswith((".json", ".log")))
                  and not f.startswith("docs/")] for r, h in ALL_HEADS.items()}
DR_RES_NOW = {r: v for r, v in DR_RES_NOW.items() if v}
AM2 = at("55dac50", "specs/test_delta_recipe.md") or ""
AM2_START = "Both reached\nthis session after X's full run had started (02:12 UTC, 11 October, test commit a581027)" in AM2 \
    or "after X's full run had started (02:12 UTC, 11 October, test commit a581027)" in AM2.replace("\n", " ")
AM2_LINES = "the four perfect-gate runs (CEIL_HEBB 580, 581; CEIL_DELTA 580, 581)" in AM2
RUN_START = dt.datetime(2026, 10, 11, 2, 12)             # 55dac50: "started 02:12 UTC"
t_test, t_check, t_rev1, t_rev2, t_amend1 = (ctime(c) for c in ("a581027", "6f2788d", "fad1489", "16b04f9", "32c121b"))
t_v4 = ctime(NEW_REF)
sess = {c: (rlib.git("show", "-s", "--format=%B", c).split("claude.ai/code/")[-1].strip().split()[0]
            if "claude.ai/code/" in rlib.git("show", "-s", "--format=%B", c) else "?")
        for c in ("a581027", "55dac50", NEW_REF)}
dr_rec = (f"spec {DR['05256c5']}; review {DR['fad1489']}; Amendment 1 {DR['32c121b']}; test {DR['a581027']}; second review "
          f"{DR['16b04f9']}; the test check {DR['6f2788d']}; Amendment 2 {DR['55dac50']}, whose text says X's full run "
          f"started 02:12 UTC on 11 October with a581027: {AM2_START} (24 per-run lines read by its commit: {AM2_LINES}); "
          f"delta_recipe result or log files on any branch now: {DR_RES_NOW or 'none'}; sessions: test and Amendment 2 "
          f"{sess['a581027']} / {sess['55dac50']}, this draft {sess[NEW_REF]}")
check_before_run = t_check < RUN_START
dr_short = (f"as the tex4:41 row: test a581027 {t_test}; X's full run started 02:12 UTC per Amendment 2 (55dac50, dated "
            f"{tstamp('55dac50')}): {AM2_START}; delta_recipe result files on any branch now: {DR_RES_NOW or 'none'}")
row4(("and written, its test file checked against the amended spec before any run",),
     "delta_recipe: 'its test file checked against the amended spec before any run' (Rev. 9 paragraph)",
     "a fourth (test_delta_recipe), specified, reviewed by the referee, amended once and written, its test file checked "
     "against the amended spec before any run", dr_rec,
     "MATCH" if check_before_run or not AM2_START else "DISAGREE",
     f"The check (6f2788d, {t_check}) came after X's run started (02:12, a581027 committed {t_test}), by the main line's own "
     "Amendment 2. Unchanged since a5f67a2; the third pass marked it RESOLVED because no commit then gave the start time. "
     "55dac50 is dated 15:36, after this version (07:52), and the draft is written by another session, so the draft's "
     "author may not have known. 'Reviewed by the referee ... before any run' is true of the first review (fad1489).")
row4(("spec reviewed by the referee and amended; the test written and checked against the amended spec; not run",),
     "delta_recipe index row: 'not run'", "spec reviewed by the referee and amended; the test written and checked against "
     "the amended spec; not run", dr_short, "DISAGREE" if AM2_START else "MATCH",
     "No result exists, but X's run had been under way for 5 h 40 min at this version's commit (02:12 to 07:52), per 55dac50. "
     "'Running on X since 02:12 (a581027); no result' would be exact.")
row4(("with the test file checked against the amended spec; not run)",), "delta_recipe, memory section: 'not run'",
     "(specified ..., reviewed by the referee before any run, amended once on its review, and written, with the test file "
     "checked against the amended spec; not run)", dr_short, "DISAGREE" if AM2_START else "MATCH", "As the index row.")
row4(("\\texttt{delta\\_recipe} & spec \\texttt{05256c5}, Amendment~1 \\texttt{32c121b} / test \\texttt{a581027} / not run & not run & ---; ---",),
     "delta_recipe provenance row", "spec 05256c5, Amendment 1 32c121b / test a581027 / not run & not run & ---; ---",
     dr_short, "PARTIAL" if AM2_START and not DR_RES_NOW else ("MATCH" if not AM2_START else "DISAGREE"),
     "The commits are right (spec and Amendment 1 change specs/test_delta_recipe.md; a581027 adds test_delta_recipe.py). "
     "No result commit exists, so the X column's last slot is empty, but 'not run' is not the state: running since 02:12. "
     "The L column ('not run') is not contradicted by any record.")
row4(("then \\texttt{test\\_delta\\_recipe} on both machines (specified, reviewed and amended once)",),
     "next steps: 'L's half of test_split_copy, then test_delta_recipe on both machines'",
     "L's half of test_split_copy, then test_delta_recipe on both machines (specified, reviewed and amended once)", dr_short,
     "PARTIAL" if AM2_START else "MATCH",
     "X's half was already running at this version's commit (per 55dac50), so the list's order does not describe the state. "
     "'Amended once' is true at 07:52 (Amendment 2 is 15:36).")
row4(("from \\texttt{test\\_delta\\_recipe} on it is reviewed by the referee session before any run",),
     "method: 'from test_delta_recipe on it is reviewed by the referee session before any run'",
     "the specification ... reviewed by the referee session before any run",
     f"first review fad1489 {t_rev1} and Amendment 1 {t_amend1}, both before the run's start {RUN_START} (55dac50); "
     f"the second review 16b04f9 {t_rev2} came after it", "MATCH" if t_rev1 < RUN_START else "DISAGREE",
     "Unchanged text. True of the spec's first review; the review of Amendment 1 (16b04f9) and the code check (6f2788d) "
     "came after the run had started.")
# tex4:430 the count test_delta_recipe prints (code at a581027)
tdr = rlib.git("show", "a581027:test_delta_recipe.py").split("\n")
pb = [i + 1 for i, ln in enumerate(tdr) if ln.startswith("def probe_blind")]
pb_body = "\n".join(tdr[pb[0] - 1:pb[0] + 20]) if pb else ""
pb_cond = ">= tgw.ACC_THR" in pb_body or ">= ACC_THR" in pb_body
pb_prn = [i + 1 for i, ln in enumerate(tdr) if "probe-blind checks" in ln and "print" in ln]
row4(("with the count of checks at which the probe read $\\geq 0.95$ while the held-out accuracy was below it, as \\texttt{test\\_delta\\_recipe} prints",),
     "method: the count test_delta_recipe prints (rewritten in c66cc76)",
     "the count of checks at which the probe read >= 0.95 while the held-out accuracy was below it, as test_delta_recipe prints",
     f"a581027:test_delta_recipe.py probe_blind() at :{pb}, the condition on probe >= ACC_THR and held-out < ACC_THR: {pb_cond}; "
     f"printed at :{pb_prn} ('... checks read >= ACC_THR on the probe while the held-out ...')",
     "MATCH" if pb and pb_cond and pb_prn else "DISAGREE",
     "The third pass's wording point is fixed: it is no longer 'checks the probe blocked'. The printed denominator includes "
     "the 2400 reference rows (test_check_delta_recipe.md, addendum); Amendment 2 (55dac50, after this version) counts "
     "from 4800 and adds 'blocked by the probe alone', so the sentence stays true after it.")
# tex4:205 R's Phase 4 probe-mode counts (the window gate's failure mode, counted on R's records)
R_TIP = tip("origin/claude/replicate-R")


def blocks(path, arm):
    n_chk, blk, runs_ = 0, 0, []
    for ln in (at(R_TIP, path) or "").split("\n"):
        if not ln.strip():
            continue
        d = _json.loads(ln)
        if d["arm"] != arm:
            continue
        ev_ = {u: a for u, a in d["evals"]}
        pr = [(u, a) for u, a in d["probes"] if u >= 4800]
        n_chk += len(pr)
        b = [u for u, a in pr if a >= 0.95 and u in ev_ and ev_[u] < 0.95]
        blk += len(b)
        if b:
            runs_.append((d["seed"], d["bound_routed"]))
    return n_chk, blk, runs_


B2, B3, B4 = (blocks("records/phase2.jsonl", "WIN3_SPLIT_D8"), blocks("records/phase3.jsonl", "SPLIT_R1_PROBE"),
              blocks("records/phase4.jsonl", "WIN3_SPLIT_D8_ML"))
fixed_n, fixed_b = B3[0] + B4[0], B3[1] + B4[1]
fixed_runs = B3[2] + B4[2]
dec2 = [s for s, br in B2[2] if not br]
r205_ok = (fixed_b, fixed_n, B2[1], B2[0]) == (3, 263, 13, 192) and all(br for _, br in fixed_runs) and dec2 == [914]
row4(("with the fixed probe the mode occurred at 3 of 263 checks and never decided a run, with one probe drawn per seed at 13 of 192",),
     "window gate: R's count of the probe's failure mode (new in c66cc76; the same numbers at tex4:328)",
     "with the fixed probe the mode occurred at 3 of 263 checks and never decided a run, with one probe drawn per seed at 13 "
     "of 192 and it decided one run in twenty",
     f"R @ {R_TIP}: checks at update >= 4800 with probe >= 0.95 and evaluation < 0.95: R1 (phase3 SPLIT_R1_PROBE, the "
     f"report's probe) {B3[1]} of {B3[0]}, Phase 4 WIN3_SPLIT_D8_ML {B4[1]} of {B4[0]} -> {fixed_b} of {fixed_n}, runs "
     f"{fixed_runs} (seed, bound routed); Phase 2 per-seed probes {B2[1]} of {B2[0]}, runs {B2[2]}; not bound routed among "
     f"them: {dec2}", "MATCH" if r205_ok else "DISAGREE",
     "Recomputed from R's records. 'Phase 4 counted' is R's Phase 4 report; the fixed-probe 263 pools R1 (Phase 3, 91 "
     "checks) with Phase 4 (172), as R's table does. 'Decided' is read as 'the run failed': the three fixed-probe runs all "
     "bound routed; of the two per-seed runs, 914 failed.")

# L's half: on every branch now, not only as of the cutoff
L_NOW = {r: [f for f in rlib.git("ls-tree", "-r", "--name-only", h).split("\n") if f.startswith("results/L/") and "split_copy" in f]
         for r, h in ALL_HEADS.items()}
L_NOW = {r: v for r, v in L_NOW.items() if v}
l_res_now = [r for r, v in L_NOW.items() if L_REC in v]
row4(("complete on X; L's half has not run",), "L's half of split_copy (Rev. 9 paragraph)", "complete on X; L's half has not run",
     f"as of {CUTOFF}: {LA_TXT}; at the branch heads now: L split_copy files {L_NOW or 'none'}; {L_REC} on {l_res_now or 'no branch'}",
     "MATCH" if l_no_runs and not l_res_now else "DISAGREE",
     "True of runs. No later L start or results file is committed on any branch at the time of this check.")

# provenance rows: every commit exists and is what the row says
E_BR, H_BR, G_BR, Rb, V_BR = ("origin/claude/outside-ideas", "origin/claude/explore-H", "origin/claude/explore-G",
                              "origin/claude/replicate-R", "origin/claude/review-V")
E_T, H_T, G_T, R_T, V_T = (tip(b) for b in (E_BR, H_BR, G_BR, Rb, V_BR))


def prov(commits, br, tip_, want_last):
    out = []
    for c, must in commits:
        s = subj(c)
        out.append((c, on(c, br), tstamp(c), s[:110], all(m.lower() in s.lower() for m in must)))
    last = tip_
    return out, last, rlib.git("rev-parse", "--short=7", want_last).strip() == last


def prov_txt(out, last, br, last_ok):
    return ("; ".join(f"{c} on {br.split('/')[-1]}: {o}, {t}, '{s}' (matches: {m})" for c, o, t, s, m in out)
            + f"; the branch's last commit at or before the version: {last} (the row's commit column is it: {last_ok})")


PE, lastE, okE = prov([("e3f5809", ("batch 19 complete", "186/186"))], E_BR, E_T, "e3f5809")
row4(("19 (E) & S64 width, S65 depth, S66 eight streams, S67 capacity (complete, \\texttt{e3f5809}) & \\texttt{e3f5809} & 186",),
     "batch 19's row: commit (content checked elsewhere)", "complete, e3f5809 & e3f5809 & 186", prov_txt(PE, lastE, E_BR, okE),
     "MATCH" if all(x[1] and x[4] for x in PE) and okE else "DISAGREE", "Commit check only; the 186 is the commit's own count.")
PH, lastH, okH = prov([("1e09ecd", ("report 7",)), ("de11b2a", ()), ("9b72bf6", ("S86 complete", "report 8"))], H_BR, H_T, "9b72bf6")
de11 = added_in("de11b2a", "report_7.md")
r8 = added_in("9b72bf6", "report_8.md")
row4(("S86 all-position targeting (report 8, \\texttt{9b72bf6}) & \\texttt{9b72bf6}",), "H's row: commits (content checked elsewhere)",
     "report 7 1e09ecd, corrected de11b2a; S86 report 8 9b72bf6 & 9b72bf6", prov_txt(PH, lastH, H_BR, okH)
     + f"; de11b2a changes {de11}; 9b72bf6 changes {r8}",
     "MATCH" if all(x[1] for x in PH) and PH[2][4] and de11 and r8 and okH else "DISAGREE",
     "Commit check only. The third pass's point (report 7 corrected at de11b2a) is met.")
PG, lastG, okG = prov([("076005d", ("report 6", "S83")), ("b2f9907", ())], G_BR, G_T, "076005d")
spec83 = added_in("b2f9907", "")
row4(("S83 the hinge's inputs, the control and the window (report 6, \\texttt{076005d}; spec \\texttt{b2f9907}) & \\texttt{076005d}",),
     "G's row: commits (content checked elsewhere)", "S83 report 6 076005d; spec b2f9907 & 076005d",
     prov_txt(PG, lastG, G_BR, okG) + f"; b2f9907 changes {spec83[:4]}",
     "MATCH" if all(x[1] for x in PG) and PG[0][4] and okG else "DISAGREE", "Commit check only.")
PR_, lastR, okR = prov([("464be8d", ()), ("1f8be04", ("Phase 4 complete", "35/40"))], Rb, R_T, "1f8be04")
row4(("Phase 4, the 40-seed count (rule \\texttt{464be8d}; report \\texttt{1f8be04}) & \\texttt{1f8be04}",),
     "R's row: commits (content checked elsewhere)", "Phase 4 rule 464be8d; report 1f8be04 & 1f8be04",
     prov_txt(PR_, lastR, Rb, okR) + f"; 464be8d changes {added_in('464be8d', 'phase4')}",
     "MATCH" if all(x[1] for x in PR_) and PR_[1][4] and okR else "DISAGREE",
     "Commit check only. The third pass's DISAGREE (Phase 4 'not started', 464be8d) is resolved in the row.")
V_CITED = (("d2bb959", ()), ("0fdb363", ()), ("ebb9325", ()), ("01bee1f", ()), ("4bf85e9", ("recompute_sessions",)),
           ("0e6c189", ("registry_L_addendum",)), ("fad1489", ("spec_review_test_delta_recipe",)),
           ("16b04f9", ("spec_review_test_delta_recipe_2",)), ("6f2788d", ("test_check_delta_recipe",)),
           ("e8fbb25", ("revision_9_check",)), ("66334b8", ("second_pass",)), ("0d1fa60", ("third_pass",)))
PV, lastV, okV = prov(V_CITED, V_BR, V_T, "0d1fa60")
v_missing = rlib.git("log", "--format=%h %s", f"0d1fa60..{V_T}").strip()
row4(("the test check (\\texttt{6f2788d}); this revision's checks (\\texttt{e8fbb25}, \\texttt{66334b8}, \\texttt{0d1fa60}) & \\texttt{0d1fa60}",),
     "V's row: commits", "... delta spec reviews (fad1489, 16b04f9); the test check (6f2788d); this revision's checks "
     "(e8fbb25, 66334b8, 0d1fa60) & 0d1fa60", prov_txt(PV, lastV, V_BR, okV) + f"; V commits after 0d1fa60 up to the version: "
     f"{v_missing or 'none'}", "MATCH" if all(x[1] and x[4] for x in PV) and okV else "DISAGREE",
     "The third pass's PARTIAL is resolved: 6f2788d, 66334b8 and the third pass 0d1fa60 are cited, and 0d1fa60 is the "
     "branch's last commit before 07:52. The session second-pass commits (fc873d2, dccc5cb, 4f3233a) and the third-pass "
     "session commits (bf62982, b2ee2ec, 67bc1ed, acf76fb) are covered by 'this revision's checks' only through the pass "
     "commits that follow them; that is a choice of listing, not an error.")
# F's row is unchanged and checked in check_F_v4 (S81 still running at the version)
if L_ATTEMPT:
    row4(("L's recorded start times are local time (UTC$-$6)",), "L start times in local time (UTC-6), caption",
         "L's recorded start times are local time (UTC-6)",
         f"L's split_copy start: '{L_ATTEMPT['started']}' local, epoch {L_ATTEMPT['started_utc']} UTC",
         "MATCH" if (dt.datetime.strptime(L_ATTEMPT['started_utc'], "%Y-%m-%d %H:%M:%S")
                     - dt.datetime.strptime(L_ATTEMPT['started'], "%Y-%m-%d %H:%M:%S")) == dt.timedelta(hours=6) else "DISAGREE",
         "Unchanged; checkable for this one L start only.")

# ── output ───────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
new_of = {}
for i, r in enumerate(rows):
    for o in r[6]:
        new_of.setdefault(o, []).append(i)


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


P = print
P("# Revision 9 draft, fourth version, vs test_split_copy on X: referee check (fourth pass)\n")
P(f"Fourth draft `{DRAFT}` at `{NEW_REF}` (committed {CUTOFF}; `tex4:NNN`, {len(tex4) - (tex4[-1] == '')} lines); third "
  f"draft `{V3_REF}` (`tex3:NNN`, {len(tex3) - (tex3[-1] == '')} lines; lines move by +2 after tex3:233 and by +3 after "
  f"tex3:304); first draft `{OLD_REF}` (`tex:NNN`). Records `{REC}` at `{RES}` (identical at the main line's head as of "
  f"the fourth draft, `{HEAD_SHORT}`: {rec_unchanged}); window_gate records and code at `{HEAD_SHORT}`; test docstring "
  f"at `{RES}`, spec at `{SPEC_C}`. Outcomes recomputed with `rlib`; they agree with the records' own transition "
  f"({agree_t}) and bound-routed fields ({agree_rec}); ch_map ties: {len(rlib.TIES)}. L's file `{L_REC}`: on no remote "
  f"branch as of the fourth draft ({len(TIP)} branches) or at any branch head now; never committed. L split_copy "
  f"artefacts at the cutoff: { {r.replace('origin/', ''): v for r, v in L_ANY.items()} or 'none'}. Generated by "
  f"`check_split_copy_v4.py` (a copy of v3 re-anchored on c66cc76).\n")
cnt = {k: sum(1 for r in rows if r[4] == k) for k in ORDER}
removed = [i for i, o in enumerate(OLD) if locate(tex4, o[3]) is None]
v3_main = [x for x in V3_ROWS if x[0] == "row"]
v3_s4 = [x for x in V3_ROWS if x[0] == "row4"]
v3_gone = [x for x in v3_main if locate(tex4, x[2]) is None]
v3_s4_gone = [x for x in v3_s4 if locate(tex4, x[2]) is None]
P("Counts against the fourth draft: " + ", ".join(f"{k} {v}" for k, v in cnt.items())
  + f" (of {len(rows)} rows). Third-pass section 1 fragments: {len(v3_main) - len(v3_gone)} of {len(v3_main)} found "
  f"verbatim in the fourth draft, {len(v3_gone)} reworded (re-anchored in section 1). Third-pass section 4 fragments: "
  f"{len(v3_s4) - len(v3_s4_gone)} of {len(v3_s4)} verbatim, {len(v3_s4_gone)} REMOVED or reworded. First-pass "
  f"fragments: {len(OLD) - len(removed)} of {len(OLD)} verbatim, {len(removed)} REMOVED or reworded.\n")
c4 = {k: sum(1 for r in rows4 if r[4] == k) for k in ORDER}
P("Section 4 (the delta_recipe statements, the window gate's new R count, L's half now, and the provenance rows' "
  "commits; not in the counts above): " + ", ".join(f"{k} {v}" for k, v in c4.items()) + f" (of {len(rows4)} rows).\n")

P("## 1. Rows against the fourth draft\n")
P("| tex4 line | item | draft says | recomputed | status | note | first-pass row |")
P("|---|---|---|---|---|---|---|")
for n, item, d, rc, s, note, old in sorted(rows, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex4:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} | "
      f"{', '.join(f'#{o + 1} (tex:{OLD[o][0]})' for o in old) or 'new'} |")

P("\n## 2. Third-pass rows located in the fourth draft\n")
P("Each third-pass fragment (from `check_split_copy_v3.py`'s `row(...)` and `row4(...)` calls, read with `ast`) was "
  "asserted in `a5f67a2` and searched verbatim in `c66cc76`.\n")
P("| # | v3 section | third-pass item | fragment | tex3 | tex4 |")
P("|---|---|---|---|---|---|")
for i, (k_, it, ph) in enumerate(V3_ROWS):
    m3, m4 = locate(tex3, ph), locate(tex4, ph)
    ra = [r_[0] for r_ in rows if r_[1].startswith(it.split(" (re-anchored")[0])] if not m4 else []
    P(f"| {i + 1} | {'1' if k_ == 'row' else '4'} | {esc(it)} | {esc(' / '.join(ph))} | tex3:{m3} | "
      f"{f'tex4:{m4}' if m4 else (f'REWORDED; re-anchored at tex4:{ra[0]}' if ra else 'REMOVED (see section 4 / resolution_F_sc_appB4.md)')} |")

P("\n## 3. First-pass rows located in the fourth draft\n")
P("| # | tex line | item | first-pass status | fragment | in fourth draft | now (section 1) |")
P("|---|---|---|---|---|---|---|")
for i, (n, item, s, ph) in enumerate(OLD):
    m = locate(tex4, ph)
    nows = "; ".join(f"tex4:{rows[j][0]} {rows[j][4]}" for j in new_of.get(i, [])) or "no row"
    P(f"| {i + 1} | tex:{n} | {esc(item)} | {s} | {esc(' / '.join(ph))} | {f'tex4:{m}' if m else 'REMOVED'} | {nows} |")

P("\n## 4. delta_recipe, the window gate's R count, L's half, and the provenance commits (as of the fourth draft)\n")
P(f"Each branch is read at its last commit dated at or before {CUTOFF}, except where a row says 'now' (the branch heads "
  "at the time of this check). Commit dates are not push dates. Amendment 2 (55dac50), which gives X's start time for "
  "test_delta_recipe, is dated 15:36 UTC, after this version.\n")
P("| tex4 line | item | draft says | record | status | note |")
P("|---|---|---|---|---|---|")
for n, item, d, rc, s, note in sorted(rows4, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex4:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} |")

P("\n## 5. Per-run records (X, seeds 560-569)\n")
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
P(("\nNo number in the fourth draft's split_copy statements disagrees with the records. " if cnt["DISAGREE"] == 0 else
   f"\n{cnt['DISAGREE']} row(s) DISAGREE with the records. ")
  + "The first pass's scope points (per machine vs pooled, readings vs claims, printed vs claimed) are addressed; what "
  "remains is listed under PARTIAL above.")
