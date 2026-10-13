"""check_split_copy_v6.py -- Referee V, sixth pass: the SIXTH version of the Revision 9 draft
(origin/claude/rev9-draft @ 42c4557, 23:14:04 UTC on 12 October, multichannel_hebbian_report_v9.tex, 568 lines) against
test_split_copy's records.

A copy of check_split_copy_v5.py (fifth pass, draft c5e4508) made to run against the sixth draft:
  * every row is re-anchored on the sixth draft by text search (tex6:NNN; an assertion fails if a fragment is missing);
    the X-half record and code checks are recomputed from the same pinned records (592c574), the L and pooled counts
    LIGHTLY from ffb0771 as in v5. Rows whose wording 42c4557 changed are re-anchored on the new wording (the row copy
    breaks merges, tex6:382, now pooled and per machine; the all-position count, tex6:441, now over 60 runs with
    split_copy L); new rows for changed text (L's CHECKs, tex6:201, read from L's log at ffb0771; the 'one-to-one'
    count, tex6:441; the L docstring correction, tex6:495; L's torch, tex6:499);
  * v5 read L's longrun files at every branch tip holding results/L/longrun/split_copy_L.json and assumed a .log beside
    it. At the sixth draft's time the main line (80bfc1d) holds the SECOND start's json and .exit (no .log) and the
    default branch (84a7e57) the FIRST start's json, .log and .exit; both are read, each for what it holds;
  * every fifth-pass row's fragment (row and row4 calls with literal fragments, read from check_split_copy_v5.py's source
    with ast) is asserted in c5e4508 and searched in 42c4557;
  * section 4 (not counted in the totals): test_delta_recipe's status lines against the main line as of the sixth draft
    (80bfc1d), R's probe-mode count, the draft's status lines (tex6:41, tex6:476-478) and every line of the two
    provenance tables: every commit hash exists, predates 42c4557 and is what the row says; each session's last cited
    commit against its branch's last commit before 23:14:04 UTC on 12 October;
  * section 6: what bbb600c and 80bfc1d changed in test_split_copy.py's RESULT (L) docstring, each corrected statement
    against L's log (ffb0771:results/L/split_copy_L.log) and the longrun files.

Inputs, all read with git at pinned refs (nothing is run, nothing is written but stdout):
  draft   9bfa85a (first), c5e4508 (fifth), 42c4557 (sixth): multichannel_hebbian_report_v9.tex
  records 592c574:results/X/split_copy_results.json; ffb0771:results/L/split_copy_results.json and split_copy_L.log;
          80bfc1d:results/L/longrun/split_copy_L.{json,exit}; 84a7e57 (default branch):
          results/L/longrun/split_copy_L.{json,log,exit}; main line as of 42c4557: results/{X,L}/window_gate_results.json;
          replicate-R as of 42c4557: records/phase{2,3,4}.jsonl
  test    592c574:test_split_copy.py, ffb0771/bbb600c/80bfc1d:test_split_copy.py (docstring), 80bfc1d:test_delta_recipe.py,
          spec fb3acb7:specs/test_split_copy.md

    python3 docs/review_2026-10/rev9/check_split_copy_v6.py > docs/review_2026-10/rev9/check_split_copy_v6.md
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402

OLD_REF, V2_REF, V3_REF, V4_REF, V5_REF, NEW_REF, DRAFT = ("9bfa85a", "f67bc07", "a5f67a2", "c66cc76", "c5e4508", "42c4557",
                                                           "multichannel_hebbian_report_v9.tex")
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
CUTOFF = rlib.git("show", "-s", "--format=%cI", NEW_REF).strip()          # the sixth draft's commit time
HEAD = rlib.git("rev-list", "-1", f"--before={CUTOFF}", MAIN).strip()[:7]  # the main line as of the sixth draft
RES, REC = "592c574", "results/X/split_copy_results.json"
SPEC_C, TEST_C = "fb3acb7", "1f52cd5"
ARMS = ("CEIL_C", "CEIL_DELTA_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8")
PART_A = ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")

tex1 = rlib.git("show", f"{OLD_REF}:{DRAFT}").split("\n")
tex2 = rlib.git("show", f"{V2_REF}:{DRAFT}").split("\n")
tex3 = rlib.git("show", f"{V3_REF}:{DRAFT}").split("\n")
tex4 = rlib.git("show", f"{V4_REF}:{DRAFT}").split("\n")
tex5 = rlib.git("show", f"{V5_REF}:{DRAFT}").split("\n")
tex6 = rlib.git("show", f"{NEW_REF}:{DRAFT}").split("\n")
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
L_ATTEMPT = None          # the FIRST start (failed at verification): the branch whose longrun files include the .log
L_SECOND = None           # the SECOND start (the run): longrun json and .exit without a .log (80bfc1d on the main line)
L_STARTS = []             # every branch's longrun json, for the table in section 6
import json as _json
for r, fs_ in sorted(L_ANY.items()):
    if "results/L/longrun/split_copy_L.json" not in fs_:
        continue
    lj = _json.loads(rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.json"))
    lexit = (rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.exit").strip()
             if "results/L/longrun/split_copy_L.exit" in fs_ else None)
    lc = rlib.git("log", "-1", "--format=%h %cI", TIP[r], "--", "results/L/longrun/split_copy_L.json").strip()
    st0 = lj["starts"][0]
    utc = dt.datetime.fromtimestamp(st0["t"], dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ent = dict(ref=r, tip=TIP[r], commit=lc, exit=lexit, started=st0["started"], started_utc=utc, boot=st0["boot"],
               git=st0["git"], cmd=" ".join(lj["cmd"]), total=lj["total"], n_starts=len(lj["starts"]),
               resumes=lj.get("resumes"), has_log="results/L/longrun/split_copy_L.log" in fs_)
    L_STARTS.append(ent)
    if ent["has_log"]:
        llog = rlib.git("show", f"{TIP[r]}:results/L/longrun/split_copy_L.log")
        ent.update(failed="SOME VERIFICATION CHECKS FAILED" in llog and "verification failed" in llog,
                   check155_wrong="CHECK 155" in llog and "-> WRONG" in llog.split("CHECK 155")[1].split("CHECK 156")[0],
                   run_lines=sum(1 for ln in llog.split("\n") if ln.startswith(("WIN3_", "CEIL_", "SINGLE_")) and " seed " in ln))
        L_ATTEMPT = ent
    else:
        L_SECOND = ent

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
ONE2ONE = []  # (source, firings on a map_before that was already one-to-one)
for name, runs in (("split_copy X", R["WIN3_SPLIT_D8"]), ("window_gate X", WGR[("X", "WIN3_SPLIT_D8")]),
                   ("window_gate L", WGR[("L", "WIN3_SPLIT_D8")]), ("split_copy L", RL["WIN3_SPLIT_D8"])):
    rows_ = [c for r in runs.values() for c in r["checks"]]
    every_row_has_all &= all({"all_cs", "all_c0", "key_cs", "key_c0", "all_masses", "key_masses"} <= set(c) for c in rows_)
    F = [c for r in runs.values() for c in r["checks"] if c["fired"]]
    ALLPOS.append((name, len(runs), len(F), sum(targeting(c["map_before"], c["all_cs"], c["all_c0"]) for c in F),
                   sum(bool(c["all_ok"]) for c in F), sum(targeting(c["map_before"], c["key_cs"], c["key_c0"]) for c in F),
                   sum((c["all_cs"], c["all_c0"]) == (c["key_cs"], c["key_c0"]) for c in F)))
    ONE2ONE.append((name, sum(len(set(c["map_before"])) == len(c["map_before"]) for c in F)))
AP = [sum(x[i] for x in ALLPOS) for i in range(1, 7)]
AP3 = [sum(x[i] for x in ALLPOS[:3]) for i in range(1, 7)]       # the fifth draft's three sources (v5's recomputation)


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
json_unchanged = rlib.git("diff", "--stat", RES, HEAD, "--", REC).strip() == ""
_t_res, _t_head = rlib.git("show", f"{RES}:test_split_copy.py"), rlib.git("show", f"{HEAD}:test_split_copy.py")
code_unchanged = _t_res.split('"""', 2)[2] == _t_head.split('"""', 2)[2]          # everything after the docstring
rec_unchanged = json_unchanged and code_unchanged
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
    n = locate(tex6, phrases)
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
    f"tex6 says 'had $0/10$ at eight streams too': {locate(tex6, ('S68 had $0/10$ at eight streams too ($2/10$ at four)',)) is not None}",
    st(nP['SINGLE_DELTA_D8'] == 0 and nP['WIN3_SPLIT_D8'] == 18 and c3P[:2] == (18, 0) and c3_state == c3L_state == "SHOWN"
       and locate(tex6, ('S68 had $0/10$ at eight streams too ($2/10$ at four)',)) is not None),
    "Light pooled counts; S68's counts are check_F_v6's.", old=(39,))
c1L_seeds = [s_ for s_ in SL["WIN3_SPLIT_D8"] if s_ in WINL["WIN3_SPLIT_D8"] and s_ not in WINL["WIN3_RESET_D8"]]
c1L_reset_oc = {s_: rlib.outcome_k(RL["WIN3_RESET_D8"][s_]) for s_ in c1L_seeds}
row(("by the pre-registered C1 (the copy with its reset against the reset alone: 12 vs 0 pooled, 7 vs 0 on X and 5 vs 0 on L)",
     "that the runs it rescues are merges is printed in their failure classes, not claimed"),
    "assessment: row copy breaks merges (re-anchored: pooled and per machine)",
    "Supported ... and by the pre-registered C1 (the copy with its reset against the reset alone: 12 vs 0 pooled, 7 vs 0 on X "
    "and 5 vs 0 on L); that the runs it rescues are merges is printed in their failure classes, not claimed",
    f"C1 X {c1[0]} vs {c1[1]} {c1_state}; L (light) {c1L[0]} vs {c1L[1]} {c1L_state}; pooled {c1P[0]} vs {c1P[1]} {c1P_state}; the "
    f"reset arm's outcome on the rescued seeds: X {len(c1_seeds)} {sorted(set(c1_reset_oc.values()))}, L {len(c1L_seeds)} "
    f"{sorted(set(c1L_reset_oc.values()))}",
    st(c1[:2] == (7, 0) and c1L[:2] == (5, 0) and c1P[:2] == (12, 0) and c1P_state == "SHOWN"
       and all(v.startswith("MERGED") for v in list(c1_reset_oc.values()) + list(c1L_reset_oc.values()))),
    "Reworded in 42c4557 on the fifth pass's PARTIAL (tex5:382: 'on X ... 7 vs 0' only). The rescued runs' failure classes "
    "are merges on both machines (light on L).", old=(40,))
row(("Pre-registered on two machines with disjoint seeds, the copy with its reset beats the reset alone (18/20 against 6/20)",
     "one delta channel binds none of twenty seeds where the recipe binds eighteen; both readings apply"), "conclusion (re-anchored: pooled)",
    "Pre-registered on two machines with disjoint seeds, the copy with its reset beats the reset alone (18/20 against 6/20), the "
    "reset alone is within the bound of no trigger, and one delta channel binds none of twenty seeds where the recipe binds "
    "eighteen; both readings apply.",
    f"pooled SPLIT {nP['WIN3_SPLIT_D8']}/20, RESET {nP['WIN3_RESET_D8']}/20, C1 {c1P_state}, C2 {c2P_state}, SINGLE_DELTA "
    f"{nP['SINGLE_DELTA_D8']}/20, C3 {c3P_state}; readings {reading1P}, {reading2P}",
    st((nP['WIN3_SPLIT_D8'], nP['WIN3_RESET_D8'], nP['SINGLE_DELTA_D8']) == (18, 6, 0)
       and (c1P_state, c2P_state, c3P_state) == ("SHOWN", "HOLDS", "SHOWN") and reading1P and reading2P), "Light pooled counts.", old=(41,))
row(("all-position pair was a labelled target", "at 115 (46 of 47, 30 of 38, 21 of 22 and 18 of 19)",
     "the key-position pair at all 126; the two pairs coincide at 54"),
    "limits: all-position targets (re-anchored: 60 runs, split_copy L added)", "at the 126 firings of the 60 recorded split runs at "
    "eight streams (window_gate X and L, split_copy X and L) the all-position pair was a labelled target ... at 115 (46 of 47, "
    "30 of 38, 21 of 22 and 18 of 19), the key-position pair at all 126; the two pairs coincide at 54",
    "; ".join(f"{n_}: {r_} runs, {f_} firings, all-position labelled {a_} (recorded all_ok {ar_}), key labelled {k_}, "
              f"coincide {co_}" for n_, r_, f_, a_, ar_, k_, co_ in ALLPOS)
    + f"; total {AP[0]} runs, {AP[1]} firings, {AP[2]}, {AP[4]}, {AP[5]}; the fifth draft's three sources (v5): {AP3[0]} runs, "
    f"{AP3[1]} firings, {AP3[2]}, {AP3[4]}, {AP3[5]}; every check row stores both pairs: {every_row_has_all}",
    st(AP[0] == 60 and AP[1] == 126 and AP[2] == AP[3] == 115 and [x[3] for x in ALLPOS] == [21, 46, 30, 18]
       and [x[2] for x in ALLPOS] == [22, 47, 38, 19] and AP[4] == 126 and AP[5] == 54 and every_row_has_all
       and (AP3[0], AP3[1], AP3[2], AP3[4], AP3[5]) == (50, 107, 97, 107, 46)),
    "Marked 'our count, not a reading'. The all-position pair is make_trigger's select(): argmax/argmin of the read gate "
    "mean over all positions of the 64-sequence probe. The split_copy L part is read from L's records at ffb0771 (60 = 20 + 20 + 10 + 10).")
row(("consistent with that rate (11 in 126)",), "limits: S86's 1 in 21 against '11 in 126' (new)",
    "S86 then found 1 mistarget in 21 firings, consistent with that rate (11 in 126)",
    f"firings whose all-position pair was not a labelled target: {AP[1]} - {AP[2]} = {AP[1] - AP[2]} of {AP[1]}",
    st(AP[1] - AP[2] == 11 and AP[1] == 126), "S86's 1 in 21 is session H's record, not recomputed here.")
row(("None of the 126 fired on a map that was already one-to-one (session~H's count)",), "limits: no firing on a one-to-one map (new)",
    "None of the 126 fired on a map that was already one-to-one (session H's count)",
    "firings whose map_before was one-to-one: " + "; ".join(f"{n_} {k_}" for n_, k_ in ONE2ONE) + f"; total {sum(k_ for _, k_ in ONE2ONE)} of {AP[1]}",
    st(sum(k_ for _, k_ in ONE2ONE) == 0 and AP[1] == 126),
    "Recomputed from the records' map_before (one-to-one = eight distinct channels); session H's count is 5b5deec ('0 of 126'). "
    "Whether session I's guard 'has nothing to do at k = 16' follows is a reading, not checked.")
# provenance
row(("spec \\texttt{fb3acb7} / \\texttt{1f52cd5} / \\texttt{592c574}",), "provenance commits",
    "spec fb3acb7 / 1f52cd5 / 592c574",
    f"{cinfo(SPEC_C)} | {cinfo(TEST_C)} | {cinfo(RES)}; meta.git = {meta['git']}; result commit changes the test by "
    f"+{res_py[0]}/-{res_py[1]} docstring lines; X's JSON unchanged to {HEAD_SHORT}: {json_unchanged}; test code after the docstring "
    f"unchanged to {HEAD_SHORT} (ffb0771 added RESULT (L) and pooled to the docstring): {code_unchanged}",
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

# ── fourth-pass rows (read from check_split_copy_v4.py's source) located in the fifth draft ───────────────
import ast  # noqa: E402
import json as _json  # noqa: E402

V3_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_split_copy_v4.py")).read()
V3_ROWS = []    # (kind, item, phrases): kind "row" (counted, section 1 of v3) or "row4" (section 4 of v3)
for node in ast.walk(ast.parse(V3_SRC)):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("row", "row4"):
        ph = tuple(ast.literal_eval(node.args[0]))
        V3_ROWS.append((node.func.id, ast.literal_eval(node.args[1]), ph))
V3_ROWS.sort(key=lambda x: (x[0], locate(tex4, x[2]) or 0))
for k_, it, ph in V3_ROWS:
    assert locate(tex4, ph) is not None, (it, ph)          # every fourth-pass fragment is in c66cc76


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
    n = locate(tex6, phrases)
    assert n is not None, (item, phrases)
    rows4.append((n, item, draft, record, status, note))


# test_delta_recipe's status (the main line as of the fifth draft: X complete at 8ae660d)
DR = {c: cinfo(c) for c in ("05256c5", "fad1489", "32c121b", "a581027", "16b04f9", "6f2788d", "55dac50", "2225f00", "8ae660d")}
ALL_HEADS = {r: rlib.git("rev-parse", "--short=7", r).strip() for r in refs}
DR_RES_5 = {r: [f for f in TREE[r] if "delta_recipe" in f and (f.startswith("results/") or f.endswith((".json", ".log")))
                and not f.startswith("docs/")] for r in TIP}
DR_RES_5 = {r: v for r, v in DR_RES_5.items() if v}
XDR = rlib.load(HEAD, "results/X/delta_recipe_results.json")
XDR_N = sum(1 for r in XDR["runs"].values() if r.get("ok"))
XDR_STARTED = dt.datetime.strptime(XDR["meta"]["started"], "%Y-%m-%d %H:%M:%S")       # X records UTC (caption tex5:499)
TDR_HEAD = rlib.git("show", f"{HEAD}:test_delta_recipe.py")
TDR_DOC = ("The run started 2026-10-11 02:12 UTC" in TDR_HEAD and "runs began at 05:27 after the verification" in TDR_HEAD)
AM2 = at("55dac50", "specs/test_delta_recipe.md") or ""
AM2_START = "after X's full run had started (02:12 UTC, 11 October, test commit a581027)" in " ".join(AM2.split())
AM2_LINES = ("the four perfect-gate runs (CEIL_HEBB 580, 581; CEIL_DELTA 580, 581)" in AM2 and "None of HEBB_SPLIT or HEBB_SLOW had finished" in AM2)
PROC_START = dt.datetime(2026, 10, 11, 2, 12)             # 55dac50 and the RESULT (X) docstring: the process started 02:12 UTC
t_test, t_check, t_rev1, t_rev2, t_amend1 = (ctime(c) for c in ("a581027", "6f2788d", "fad1489", "16b04f9", "32c121b"))
t_am2, t_22, t_x = ctime("55dac50"), ctime("2225f00"), ctime("8ae660d")
L_DR5 = [f"{r}:{f}" for r, v in DR_RES_5.items() for f in v if f.startswith("results/L/")]
dr5 = (f"test a581027 ({t_test}); the referee's second review 16b04f9 ({t_rev2}) and test check 6f2788d ({t_check}); X's process "
       f"started {PROC_START} (Amendment 2; RESULT (X)), the arms' runs at {XDR_STARTED} (meta.started; 'runs began at 05:27 after "
       f"the verification': {TDR_DOC}); Amendment 2 55dac50 ({t_am2}) and 2225f00 ({t_22}); X's result 8ae660d ({t_x}), {XDR_N} records, "
       f"meta.git {XDR['meta']['git']}; L delta_recipe files at the fifth draft: {L_DR5 or 'none'}")
x_done = XDR_N == 84 and t_x < ctime(NEW_REF) and not L_DR5
row4(("amended once before any run and written, its test file checked against the amended spec, and complete on X",),
     "delta_recipe (Rev. 9 paragraph): 'amended once before any run ..., checked against the amended spec, and complete on X (a second "
     "amendment ... during X's run ...); L's half is next' (re-anchored)", "amended once before any run; checked; complete on X; L next",
     dr5, st(x_done and t_amend1 < PROC_START and XDR_STARTED < t_x and XDR_STARTED < t_am2 < t_x),
     "Resolves fourth_pass §1 item 6 (the wording changed). CORRECTION to check_split_copy_v4 §4's DISAGREE on tex4:41: the arms' "
     f"runs began at {XDR_STARTED} UTC, after the check 6f2788d ({t_check}); 02:12 was the start of the process, which ran the "
     "3 h 15 min verification first. tex4:41's 'checked ... before any run' was true of the runs.")
row4(("\\texttt{delta\\_recipe} (specified)", "on X (secondary): D4 SHOWN 10 vs 1", "pooled labels and readings wait for L"),
     "delta_recipe index row (re-anchored): status", "(specified) ... on X (secondary): ...; pooled labels and readings wait for L", dr5,
     "PARTIAL" if x_done else "DISAGREE",
     "Resolves fourth_pass §1 item 7 ('not run' is gone). The label '(specified)' is stale: X's half is complete; '(X; L to run)' "
     "would match split_copy's row. The D numbers are the X-results check's.")
row4(("with the test file checked against the amended spec; running on X, where a second amendment confined to the report's rules",),
     "delta_recipe, memory section (re-anchored): 'running on X, where a second amendment ... was committed during the run'",
     "running on X", dr5, "PARTIAL" if x_done else "DISAGREE",
     "Fourth_pass §1 item 8: PARTLY. 'Running on X' is not the state at c5e4508 (X complete at 8ae660d, 07:25:51 on 12 October); "
     "the same paragraph says 'X's half is complete' further on.")
row4(("the 24 run lines that had been read, none of them a claim, a reading or a Hebbian run",),
     "delta_recipe, memory section: '24 run lines ... none of them a claim, a reading or a Hebbian run' (new)",
     "none a Hebbian run", f"Amendment 2: 24 lines = CEIL_HEBB 580, 581; CEIL_DELTA 580, 581; DELTA_SPLIT 580-598; DELTA_SLOW|580; 'None of "
     f"HEBB_SPLIT or HEBB_SLOW had finished': {AM2_LINES}", "DISAGREE" if AM2_LINES else "MATCH",
     "Two of the 24 were Hebbian runs (the Hebbian perfect gate CEIL_HEBB 580, 581). 'None ... of a Hebbian learned arm' is exact.")
r55, r22, r8a = (rlib.git("show", "--name-only", "--format=", c).split() for c in ("55dac50", "2225f00", "8ae660d"))
row4(("Amendment~2 during the run \\texttt{55dac50}, \\texttt{2225f00} / \\texttt{8ae660d} (X) & X complete; L to run & 84; ---",),
     "delta_recipe provenance row (re-anchored)", "spec 05256c5, Amendment 1 32c121b / test a581027, Amendment 2 during the run 55dac50, "
     "2225f00 / 8ae660d (X) & X complete; L to run & 84; ---",
     dr5 + f"; 55dac50 changes {r55}; 2225f00 changes {r22}; 8ae660d changes {r8a}",
     st(x_done and "specs/test_delta_recipe.md" in r55 and "test_delta_recipe.py" in r22 and "results/X/delta_recipe_results.json" in r8a
        and XDR_STARTED < t_am2 < t_22 < t_x and XDR["meta"]["git"] == "a581027"),
     "Resolves the fourth pass's §2 bullet tex4:492. Both amendment commits fall between the runs' start and the result; the X "
     "records carry a581027 (the report was regenerated with --report at 2225f00).")
row4(("\\texttt{test\\_delta\\_recipe}'s L half (X's is complete)",), "next steps (re-anchored): 'test_delta_recipe's L half (X's is complete)'",
     "L half next", dr5, st(x_done), "Resolves the fourth pass's §2 bullet tex4:472.")
row4(("from \\texttt{test\\_delta\\_recipe} on it is reviewed by the referee session before any run",),
     "method: 'from test_delta_recipe on it is reviewed by the referee session before any run'",
     "the specification ... reviewed by the referee session before any run",
     f"first review fad1489 {t_rev1}, Amendment 1 {t_amend1}, second review 16b04f9 {t_rev2}, test check 6f2788d {t_check}; the arms' "
     f"runs began {XDR_STARTED}", st(t_rev1 < t_rev2 < XDR_STARTED and t_check < XDR_STARTED),
     "Unchanged text. True of both reviews and of the test check (all before 05:27); the fourth pass's note that the second "
     "review came after the run had started counted the verification as the run.")
tdr5 = TDR_HEAD.split("\n")
pb = [i + 1 for i, ln in enumerate(tdr5) if ln.startswith("def probe_blind")]
pb_body = "\n".join(tdr5[pb[0] - 1:pb[0] + 22]) if pb else ""
pb_cond = 'c["acc"] >= tgw.ACC_THR and h < tgw.ACC_THR' in pb_body and 'c["step"] >= tgw.FIRST' in pb_body
pb_prn = [i + 1 for i, ln in enumerate(tdr5) if "probe-blind checks" in ln and "print" in ln]
row4(("with the count of checks at which the probe read $\\geq 0.95$ while the held-out accuracy was below it, as \\texttt{test\\_delta\\_recipe} prints",),
     "method: the count test_delta_recipe prints (code at the main line as of the fifth draft)",
     "the count of checks at which the probe read >= 0.95 while the held-out accuracy was below it, as test_delta_recipe prints",
     f"{HEAD}:test_delta_recipe.py probe_blind() at :{pb}, probe >= ACC_THR and held-out < ACC_THR at updates >= FIRST: {pb_cond}; printed "
     f"at :{pb_prn}; X's RESULT docstring prints 'Probe-blind (checks from 4800)': {'Probe-blind (checks from 4800)' in TDR_HEAD}",
     st(pb and pb_cond and pb_prn), "Since Amendment 2 the count starts at 4800 (the fourth pass's residual is gone).")
# R's probe-mode count, now 13 of 282 per seed (tex5:207; the same at tex5:330 with its parenthesis)
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


B2, B3, B4, BR2 = (blocks("records/phase2.jsonl", "WIN3_SPLIT_D8"), blocks("records/phase3.jsonl", "SPLIT_R1_PROBE"),
                   blocks("records/phase4.jsonl", "WIN3_SPLIT_D8_ML"), blocks("records/phase3.jsonl", "SPLIT_R2_BODYKEYS"))
fixed_n, fixed_b = B3[0] + B4[0], B3[1] + B4[1]
fixed_runs = B3[2] + B4[2]
per_n, per_b = B2[0] + BR2[0], B2[1] + BR2[1]
dec2 = [s for s, br in B2[2] if not br]
r207_ok = (fixed_b, fixed_n, per_b, per_n, B2[0], BR2[1]) == (3, 263, 13, 282, 192, 0) and all(br for _, br in fixed_runs) and dec2 == [914]
row4(("with the fixed probe the mode occurred at 3 of 263 checks and never decided a run, with one probe drawn per seed at 13 of 282 and, on R's inference, it decided one run",),
     "window gate: R's count of the probe's failure mode (re-anchored: 13 of 282, 'on R's inference')",
     "with the fixed probe ... 3 of 263 checks and never decided a run, with one probe drawn per seed at 13 of 282 and, on R's "
     "inference, it decided one run",
     f"R @ {R_TIP}: checks at update >= 4800 with probe >= 0.95 and evaluation < 0.95: fixed probe R1 {B3[1]} of {B3[0]} + Phase 4 "
     f"{B4[1]} of {B4[0]} = {fixed_b} of {fixed_n}, runs {fixed_runs}; per-seed probes Phase 2 {B2[1]} of {B2[0]} + R's second rerun "
     f"(SPLIT_R2_BODYKEYS) {BR2[1]} of {BR2[0]} = {per_b} of {per_n}; not bound routed among them: {dec2}",
     st(r207_ok),
     "Resolves the fourth pass's §2 bullet ('13 of 192' unlabelled): the count is now pooled the same way as the fixed-probe one, "
     "and 'decided one run' is marked as R's inference (tex5:330 gives 13 of 192 for Phase 2 and 13 of 282 with the rerun).")
# L's half now (complete)
L_NOW = {r: [f for f in rlib.git("ls-tree", "-r", "--name-only", h).split("\n") if f.startswith("results/L/") and "split_copy" in f]
         for r, h in ALL_HEADS.items()}
L_NOW = {r: v for r, v in L_NOW.items() if v}
l_res_now = [r for r, v in L_NOW.items() if L_REC in v]
row4(("\\texttt{test\\_split\\_copy}, complete on both machines",), "L's half of split_copy (Rev. 9 paragraph): complete",
     "complete on both machines", f"{L_REC} at {L_RES} ({cinfo(L_RES)}); on branches at the fifth draft: {l_file_on}; {nL_runs} records, all ok {okL}",
     st(bool(l_file_on) and nL_runs == 54 and okL), "L's half landed at ffb0771 (02:39:51 UTC, 12 October), before the fifth draft.")

# ── the provenance tables (tex5:486-524): every commit hash, and each row's commits ─────────────
CUT_DT = int(rlib.git('show', '-s', '--format=%ct', NEW_REF).strip())
P0 = locate(tex5, ("\\begin{table}[H]",)) or 0
PROV_LINES = [i + 1 for i, ln in enumerate(tex5) if "label{tab:prov}" in ln or "label{tab:batches}" in ln]
PROV_A = max(i for i in range(PROV_LINES[0], 0, -1) if tex5[i - 1].startswith("\\begin{table}"))
PROV_B = PROV_LINES[-1]
HEX = {}
for i in range(PROV_A, PROV_B + 1):
    for h in __import__("re").findall(r"\\texttt\{([0-9a-f]{7})\}", tex5[i - 1]):
        HEX.setdefault(h, []).append(i)
TIPS5 = {r: t for r, t in TIP.items()}


def exists(h):
    try:
        rlib.git("cat-file", "-e", f"{h}^{{commit}}")
        return True
    except Exception:
        return False


HROWS = []
for h, lns in sorted(HEX.items(), key=lambda x: (x[1][0], x[0])):
    ok_ = exists(h)
    t_ = int(rlib.git('show', '-s', '--format=%ct', h).strip()) if ok_ else None
    brs = sorted(r.replace("origin/", "").replace("claude/", "") for r, t in TIPS5.items() if ok_ and ancestor(h, t))
    HROWS.append((h, lns, ok_, t_, t_ is not None and t_ < CUT_DT, brs, subj(h)[:100] if ok_ else ""))
all_exist = all(x[2] for x in HROWS)
all_before = all(x[4] for x in HROWS)
on_some = all(x[5] for x in HROWS)
row4(("\\label{tab:batches}",), "provenance tables: every commit hash (tex5 tables and captions)", f"{len(HEX)} distinct hashes",
     f"{len(HEX)} distinct 7-hex hashes on tex5:{PROV_A}-{PROV_B}; exist {sum(x[2] for x in HROWS)}; predate c5e4508 "
     f"{sum(x[4] for x in HROWS)}; on some remote branch as of the cutoff {sum(1 for x in HROWS if x[5])} (each listed in section 4b)",
     st(all_exist and all_before and on_some), "")


def last_report(br):
    return TIP[br]


SESS = [("E 15--18", "origin/claude/outside-ideas", ["90c3a28", "93bdfe6", "2531480", "7ef4e15"], None),
        ("E 19", "origin/claude/outside-ideas", ["e3f5809"], "e3f5809"),
        ("H", "origin/claude/explore-H", ["7e9197f", "72f0bbc", "df57087", "37c1fbf", "1e09ecd", "de11b2a", "9b72bf6"], "9b72bf6"),
        ("F", "origin/claude/explore-F", ["9a6c2bf", "22ec912", "5b9bc2a", "ed76eea", "d5851b1"], "d5851b1"),
        ("G", "origin/claude/explore-G", ["6cd36ff", "427db36", "35de6db", "93422cf", "8973d48", "83699a2", "076005d", "b2f9907"], "076005d"),
        ("I", "origin/claude/explore-I", ["a2588da", "5f03929", "203a5ad", "86652b1", "b649751", "5bc6c2e", "03bffdf"], "03bffdf"),
        ("R", "origin/claude/replicate-R", ["de8df0f", "00990ea", "100dced", "464be8d", "1f8be04"], "1f8be04"),
        ("V", "origin/claude/review-V", ["d2bb959", "0fdb363", "ebb9325", "01bee1f", "4bf85e9", "0e6c189", "fad1489", "16b04f9", "6f2788d",
                                         "e8fbb25", "66334b8", "0d1fa60"], "0d1fa60")]
SESS_OUT = {}
for name, br, cited, col in SESS:
    t_last = TIP[br]
    after = rlib.git("log", "--format=%h %s", f"{col}..{t_last}").strip().split("\n") if col else []
    after = [x for x in after if x.strip()]
    rep_after = [x for x in after if any(w in x.lower() for w in ("report", "pass", "complete", "write-up"))
                 and "progress" not in x.lower() and "EXPLORATORY" not in x]
    SESS_OUT[name] = dict(br=br, cited=cited, on=all(ancestor(c, t_last) for c in cited), last=t_last, col=col,
                          after=after, rep_after=rep_after)
for name, o in SESS_OUT.items():
    if o["col"] is None:
        continue
    lab = {"E 19": "19 (E) &", "H": "H & S58, S58b, S60", "F": "F & S53, S54, S59", "G": "G & S55 probe", "I": "I & S74 pilot",
           "R": "R & Phase 2", "V": "V & review"}[name]
    stv = "MATCH" if o["on"] and not o["rep_after"] else "PARTIAL"
    note = {"V": "V's branch's last commit before the cutoff is the fourth pass (907eb2c, 15:56:25 UTC on 11 October; fourth_pass.md, "
                 "with 1545e88 ... 260a3bd), which the draft's own commit fd2fa89 names; the row still ends at the third pass 0d1fa60.",
            "I": "The S75 paragraph (tex5:458) cites the WIDE_SLOW pairing on nine seeds, recorded at 4c3984a (19:12:59 UTC on 12 October, "
                 "80 s before the draft); the row's records list (86652b1, b649751) and its count (32) do not include it (41 records at 4c3984a).",
            "E 19": "E's later commits are batch 20 (code and readings 4acf915, runs and progress): no report yet, so the column is right; "
                    "batch 20 is running, which bears on tex5:41's 'every session's queue is empty'."}.get(name, "")
    if name == "I":
        i_recs = rlib.load(o["last"], "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json")
        n_i = len(i_recs.get("runs", i_recs))
        stv = "PARTIAL" if n_i > 32 else stv
        note += f" Records at {o['last']}: {n_i}."
    row4((lab,), f"provenance row {name}: commits, and the last cited commit against the branch",
         f"cited {', '.join(o['cited'])}; commit column {o['col']}",
         f"all cited on {o['br'].split('/')[-1]} as of the cutoff: {o['on']}; the branch's last commit before the cutoff {o['last']}; commits "
         f"after the column's {o['col']}: {len(o['after'])}" + (f", of which reports or passes: {o['rep_after']}" if o["rep_after"] else ""),
         stv, note)
MAIN_CITED = ["74f5907", "3e2d171", "2a6ba3d", "9a769c7", "d05f0ca", "2f98f06", "886a668", "628eb7c", "1cd20a1", "fb3acb7", "1f52cd5",
              "592c574", "ffb0771", "ba8901a", "56c44a7", "05256c5", "32c121b", "a581027", "55dac50", "2225f00", "8ae660d",
              "9a53302", "9c53e16", "00ac0f2", "d0e6c85", "42af095", "b22da04", "ad8a92e"]
main_on = {c: ancestor(c, HEAD) for c in MAIN_CITED}
row4(("\\texttt{muon\\_recipe} & --- / \\texttt{74f5907}",), "main-line table and captions: every cited commit on the main line as of the cutoff",
     f"{len(MAIN_CITED)} commits", f"on {HEAD}: {sum(main_on.values())} of {len(MAIN_CITED)}; not: {[c for c, v in main_on.items() if not v] or 'none'}",
     st(all(main_on.values())), "Commit check only; the muon_recipe and window_gate rows are unchanged since the earlier passes.")
# L's start times caption (unchanged)
if L_ATTEMPT:
    row4(("L's recorded start times are local time (UTC$-$6)",), "L start times in local time (UTC-6), caption",
         "L's recorded start times are local time (UTC-6)",
         f"L's failed split_copy start: '{L_ATTEMPT['started']}' local, epoch {L_ATTEMPT['started_utc']} UTC; L's result meta.started "
         f"'{LSTORE['meta'].get('started')}' (the RESULT (L) docstring: 'started 09:31 local = 15:31 UTC')",
         "MATCH" if (dt.datetime.strptime(L_ATTEMPT['started_utc'], "%Y-%m-%d %H:%M:%S")
                     - dt.datetime.strptime(L_ATTEMPT['started'], "%Y-%m-%d %H:%M:%S")) == dt.timedelta(hours=6) else "DISAGREE",
         "Unchanged; checkable for the failed start only (L's result file carries no epoch).")


# ── output ───────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
new_of = {}
for i, r in enumerate(rows):
    for o in r[6]:
        new_of.setdefault(o, []).append(i)


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


P = print
P("# Revision 9 draft, fifth version, vs test_split_copy: referee check (fifth pass)\n")
P(f"Fifth draft `{DRAFT}` at `{NEW_REF}` (committed {CUTOFF}; `tex5:NNN`, {len(tex5) - (tex5[-1] == '')} lines); fourth "
  f"draft `{V4_REF}` (`tex4:NNN`, {len(tex4) - (tex4[-1] == '')} lines; lines move by +2 after tex4:192 and by +4 after "
  f"tex4:454); first draft `{OLD_REF}` (`tex:NNN`). X's records `{REC}` at `{RES}` (X's JSON unchanged at the main line as of "
  f"the fifth draft, `{HEAD_SHORT}`: {json_unchanged}); L's records `{L_REC}` at `{L_RES}` ({nL_runs} records, used LIGHTLY: "
  f"counts, discordant pairs and exact p for the re-anchored pooled sentences; L's half in depth is another check's); "
  f"window_gate records and code at `{HEAD_SHORT}`; test docstring at `{RES}`, spec at `{SPEC_C}`. Outcomes recomputed with "
  f"`rlib`; they agree with X's records' own transition ({agree_t}) and bound-routed fields ({agree_rec}); ch_map ties: "
  f"{len(rlib.TIES)}. Generated by `check_split_copy_v5.py` (a copy of v4 re-anchored on c5e4508).\n")
cnt = {k: sum(1 for r in rows if r[4] == k) for k in ORDER}
removed = [i for i, o in enumerate(OLD) if locate(tex5, o[3]) is None]
v3_main = [x for x in V3_ROWS if x[0] == "row"]
v3_s4 = [x for x in V3_ROWS if x[0] == "row4"]
v3_gone = [x for x in v3_main if locate(tex5, x[2]) is None]
v3_s4_gone = [x for x in v3_s4 if locate(tex5, x[2]) is None]
P("Counts against the fifth draft: " + ", ".join(f"{k} {v}" for k, v in cnt.items())
  + f" (of {len(rows)} rows). Fourth-pass section 1 fragments: {len(v3_main) - len(v3_gone)} of {len(v3_main)} found "
  f"verbatim in the fifth draft, {len(v3_gone)} reworded (re-anchored in section 1). Fourth-pass section 4 fragments: "
  f"{len(v3_s4) - len(v3_s4_gone)} of {len(v3_s4)} verbatim, {len(v3_s4_gone)} REMOVED or reworded (re-anchored in section 4). "
  f"First-pass fragments: {len(OLD) - len(removed)} of {len(OLD)} verbatim, {len(removed)} REMOVED or reworded.\n")
c4 = {k: sum(1 for r in rows4 if r[4] == k) for k in ORDER}
P("Section 4 (test_delta_recipe's status lines, R's probe-mode count, L's half, and the provenance tables; not in the counts "
  "above): " + ", ".join(f"{k} {v}" for k, v in c4.items()) + f" (of {len(rows4)} rows).\n")

P("## 1. Rows against the fifth draft\n")
P("| tex5 line | item | draft says | recomputed | status | note | first-pass row |")
P("|---|---|---|---|---|---|---|")
for n, item, d, rc, s, note, old in sorted(rows, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex5:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} | "
      f"{', '.join(f'#{o + 1} (tex:{OLD[o][0]})' for o in old) or 'new'} |")

MAP4TO5 = {
    "abstract: C1 counts": "abstract: C1 counts (re-anchored", "abstract: reset vs no trigger": "abstract: reset vs no trigger (re-anchored",
    "abstract: C3 counts": "abstract: C3 counts (re-anchored", "abstract: scope of the result": "abstract: scope of the result (re-anchored",
    "Rev. 9 paragraph": "Rev. 9 paragraph (re-anchored", "index row": "index row (re-anchored", "'address it' (was 'settle')": "'settle it' (re-anchored",
    "caption: L half": "caption: pooled primary", "reading 1 quoted": "reading 1 applies", "readings pooled only; X's conditions": "reset seeds against none seeds",
    "C3": "C3 (re-anchored", "one Hebbian channel": "one Hebbian channel (re-anchored", "delta perfect gate": "delta perfect gate (re-anchored",
    "reading 2 and its content": "reading 2 applies", "synthesis S=8 row: window gate + SLOW and split columns": "synthesis S=8 row: window gate + SLOW and split columns (re-anchored",
    "synthesis: single channels": "synthesis: single channels (re-anchored", "assessment: copy not reset": "assessment: copy not reset (re-anchored",
    "assessment: number of contexts": "assessment: number of contexts (re-anchored", "conclusion": "conclusion (re-anchored",
    "provenance table: L column and runs (re-anchored on c66cc76)": "provenance table: L column and runs (re-anchored on c5e4508",
    "delta_recipe: 'its test file checked against the amended spec before any run' (Rev. 9 paragraph)": "delta_recipe (Rev. 9 paragraph)",
    "L's half of split_copy (Rev. 9 paragraph)": "L's half of split_copy (Rev. 9 paragraph): complete",
    "delta_recipe index row: 'not run'": "delta_recipe index row (re-anchored", "delta_recipe, memory section: 'not run'": "delta_recipe, memory section (re-anchored",
    "next steps: 'L's half of test_split_copy, then test_delta_recipe on both machines'": "next steps (re-anchored",
    "delta_recipe provenance row": "delta_recipe provenance row (re-anchored",
    "window gate: R's count of the probe's failure mode (new in c66cc76; the same numbers at tex4:328)": "window gate: R's count of the probe's failure mode (re-anchored",
}
P("\n## 2. Fourth-pass rows located in the fifth draft\n")
P("Each fourth-pass fragment (from `check_split_copy_v4.py`'s `row(...)` and `row4(...)` calls, read with `ast`) was "
  "asserted in `c66cc76` and searched verbatim in `c5e4508`.\n")
P("| # | v4 section | fourth-pass item | fragment | tex4 | tex5 |")
P("|---|---|---|---|---|---|")
for i, (k_, it, ph) in enumerate(V3_ROWS):
    m4, m5 = locate(tex4, ph), locate(tex5, ph)
    pool = rows if k_ == "row" else rows4
    tgt = MAP4TO5.get(it)
    ra = [r_[0] for r_ in pool if tgt and r_[1].startswith(tgt)] if not m5 else []
    P(f"| {i + 1} | {'1' if k_ == 'row' else '4'} | {esc(it)} | {esc(' / '.join(ph))} | tex4:{m4} | "
      f"{f'tex5:{m5}' if m5 else (f'REWORDED; re-anchored at tex5:{ra[0]}' if ra else 'NOT RE-ANCHORED')} |")

P("\n## 3. First-pass rows located in the fifth draft\n")
P("| # | tex line | item | first-pass status | fragment | in fifth draft | now (section 1) |")
P("|---|---|---|---|---|---|---|")
for i, (n, item, s, ph) in enumerate(OLD):
    m = locate(tex5, ph)
    nows = "; ".join(f"tex5:{rows[j][0]} {rows[j][4]}" for j in new_of.get(i, [])) or "no row"
    P(f"| {i + 1} | tex:{n} | {esc(item)} | {s} | {esc(' / '.join(ph))} | {f'tex5:{m}' if m else 'REMOVED'} | {nows} |")

P("\n## 4. test_delta_recipe's status, R's count, L's half, and the provenance tables (as of the fifth draft)\n")
P(f"Each branch is read at its last commit dated at or before {CUTOFF}, except where a row says 'now'. Commit dates are not "
  "push dates.\n")
P("| tex5 line | item | draft says | record | status | note |")
P("|---|---|---|---|---|---|")
for n, item, d, rc, s, note in sorted(rows4, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex5:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} |")
P(f"\n### 4b. Every commit hash in the provenance tables and captions (tex5:{PROV_A}-{PROV_B})\n")
P("| hash | tex5 lines | exists | committed (UTC) | before c5e4508 | on (as of the cutoff) | subject |")
P("|---|---|---|---|---|---|---|")
for h, lns, ok_, t_, bef, brs, sj in HROWS:
    P(f"| {h} | {', '.join(map(str, lns))} | {ok_} | {dt.datetime.fromtimestamp(t_, dt.timezone.utc):%Y-%m-%d %H:%M:%S} | {bef} | "
      f"{', '.join(brs)} | {esc(sj)} |")
P("\n### 4c. Each session's last cited commit against its branch's last commit before the cutoff\n")
P("| row | branch | commit column | branch's last commit before the cutoff | commits after the column | reports or passes after it |")
P("|---|---|---|---|---|---|")
for name, o in SESS_OUT.items():
    P(f"| {name} | {o['br'].replace('origin/', '')} | {o['col'] or '-'} | {o['last']} ({cinfo(o['last'])[8:27]}) | {len(o['after'])} | "
      f"{esc('; '.join(o['rep_after'])[:400]) or 'none'} |")
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
P(("\nNo number in the fifth draft's split_copy statements disagrees with the records (X recomputed; L and pooled lightly). " if cnt["DISAGREE"] == 0 else
   f"\n{cnt['DISAGREE']} row(s) DISAGREE with the records. ")
  + "The first pass's scope points (per machine vs pooled, readings vs claims, printed vs claimed) are addressed; what "
  "remains is listed under PARTIAL above.")
