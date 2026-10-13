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
          f"L's first start: {L_ATTEMPT['ref'].replace('origin/', '')} @ {L_ATTEMPT['commit']} "
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


# ── L's log (ffb0771) and the RESULT (L) docstring at ffb0771, bbb600c and 80bfc1d ───────────────────────────
import re as _re
LLOG_REF, DOC_REFS = L_RES, (L_RES, "bbb600c", "80bfc1d")
LLOG = rlib.git("show", f"{LLOG_REF}:results/L/split_copy_L.log").split("\n")
L_HDR = [(i + 1, ln) for i, ln in enumerate(LLOG) if ln.startswith("=== longrun split_copy_L: start")]
SEG2 = L_HDR[1][0] if len(L_HDR) > 1 else 1                    # first line of the second start
CK = {}                                                           # CHECK id -> dict(line, text) in the second start
cur = None
for i in range(SEG2 - 1, len(LLOG)):
    ln = LLOG[i]
    m = _re.match(r"^\s*CHECK (\d+(?:\([ab]\))?)\s", ln)
    if m:
        cur = m.group(1)
        CK.setdefault(cur, dict(line=i + 1, text=[]))
    elif ln.startswith("=" * 20) or ln.startswith("ALL VERIFICATION") or ln.startswith("SOME VERIFICATION"):
        cur = None
    if cur:
        CK[cur]["text"].append((i + 1, ln))


def marks(k):
    return [m_ for _, ln in CK[k]["text"] for m_ in _re.findall(r"-> ([A-Z][A-Z ]*[A-Z])", ln)]


PRINTED = {k: v for k, v in CK.items() if any("printed, not asserted" in ln for _, ln in v["text"])}
PRINTED_INFO = {k: dict(line=v["line"], marks=marks(k), differs=[n for n, ln in v["text"] if "-> DIFFERS" in ln],
                        written=sorted(set(_re.findall(r"written on (Intel\(R\) Xeon\(R\) Processor @ [0-9.]+GHz|12th Gen Intel\(R\) Core\(TM\) i7-12650H|not recorded|another CPU)",
                                                       " ".join(ln for _, ln in v["text"])))))
                for k, v in PRINTED.items()}
SIDE = {}        # CHECKs that assert something else but print a not-asserted comparison with another host's record
for k, v in CK.items():
    for n, ln in v["text"]:
        if "(not asserted)" in ln and "equal: False" in ln:
            SIDE.setdefault(k, []).append((n, ln.strip()[:150]))
INFORM = {k: [n for n, ln in v["text"] if "curves equal False" in ln] for k, v in CK.items()
          if "informative, not asserted" in LLOG[v["line"] - 1]}
seg2_txt = LLOG[SEG2 - 1:]
n_pass2 = sum(1 for ln in seg2_txt if ln.startswith("ALL VERIFICATION CHECKS PASSED"))
n_fail2 = sum(1 for ln in seg2_txt if ln.startswith("SOME VERIFICATION CHECKS FAILED"))
n_tb2 = sum(1 for ln in seg2_txt if ln.startswith("Traceback"))
last_line = [ln for ln in LLOG if ln.strip()][-1].strip()
ck130, ck146 = marks("130"), marks("146")
ck151a = " ".join(ln for _, ln in CK.get("151(a)", dict(text=[]))["text"])
ck151b = [ln.strip() for ln in seg2_txt if "CHECK 151(b)" in ln]
# the not-asserted CHECKs compare with records not written on this CPU (none on L's own files)
own_cpu = "12th Gen Intel(R) Core(TM) i7-12650H"
printed_other = all(own_cpu not in " ".join(w for w in PRINTED_INFO[k]["written"]) for k in PRINTED_INFO)
first_seg = LLOG[:SEG2 - 1]
first_runs = sum(1 for ln in first_seg if ln.startswith(("  WIN3_", "  CEIL_", "  SINGLE_")) and " seed " in ln)
first_155 = any("CHECK 155" in ln for ln in first_seg) and any("-> WRONG" in ln for ln in first_seg)
first_failed = any(ln.startswith("SOME VERIFICATION CHECKS FAILED") for ln in first_seg)


def docblock(ref):
    t = rlib.git("show", f"{ref}:test_split_copy.py")
    lines = t.split("\n")
    a_ = [i for i, ln in enumerate(lines) if ln.startswith("RESULT (L)")][0]
    b_ = next(i for i in range(a_ + 1, len(lines)) if lines[i].startswith(("POOLED", "RESULT (")) or lines[i].startswith('"""'))
    return a_ + 1, lines[a_:b_], " ".join(" ".join(lines[a_:b_]).split())


DOCV = {r: docblock(r) for r in DOC_REFS}
DOCLN = {r: {k: next((DOCV[r][0] + j for j, ln in enumerate(DOCV[r][1]) if k in ln), None)
             for k in ("(exit 0", "Verification:")} for r in DOC_REFS}
DOC_NUM = {r: rlib.git("diff", "--numstat", f"{r}^", r, "--", "test_split_copy.py").split()[:2] for r in ("bbb600c", "80bfc1d")}
doc_code_same = all(rlib.git("show", f"{r}:test_split_copy.py").split('"""', 2)[2] == _t_res_code
                    for r in DOC_REFS for _t_res_code in [rlib.git("show", f"{RES}:test_split_copy.py").split('"""', 2)[2]])
D80 = DOCV["80bfc1d"][2]
DFF = DOCV[L_RES][2]
DBB = DOCV["bbb600c"][2]
L2 = L_SECOND or {}
l2_utc_off = ((dt.datetime.strptime(L2["started_utc"], "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(L2["started"], "%Y-%m-%d %H:%M:%S"))
              if L2 else None)
l2_boot_in_log = bool(L2) and any(L2["boot"][:8] in ln and L2["started"] in ln for _, ln in L_HDR[1:])
l1_boot_in_log = bool(L_ATTEMPT) and any(L_ATTEMPT["boot"][:8] in ln and L_ATTEMPT["started"] in ln for _, ln in L_HDR[:1])
DROWS = []   # (doc ref:line, statement, record, status, note)


def drow(where, stmt, rec, status, note=""):
    DROWS.append((where, stmt, rec, status, note))


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
    "provenance table: L column and runs",
    "split_copy ... ffb0771 (L and pooled; ...) & complete (L run at 592c574; one earlier start stopped at verification over an argument) & 54; 54",
    f"X records {n_runs}; L results file added at {L_add} ({cinfo(L_RES)}), {nL_runs} records, all ok {okL}, meta.git {L_git}; the "
    f"earlier start: {LA_TXT}",
    st(n_runs == 54 and nL_runs == 54 and okL and L_git == "592c574" and L_add == [L_RES] and L_ATTEMPT is not None
       and L_ATTEMPT["failed"] and L_ATTEMPT["run_lines"] == 0),
    "Unchanged since c5e4508. 'Complete' and 'L run at 592c574' match L's file; the earlier start is the failed one at 84a7e57 "
    "on the default branch (CHECK 155 WRONG, exit 1, no run line).", old=(44,))
row(("2.10\\,GHz for \\texttt{split\\_copy}",), "X CPU for split_copy", "2.10 GHz for split_copy",
    f"meta.cpu '{meta['cpu']}'; per-record {sorted(cpus)}", st(cpus == {meta['cpu']} and "2.10GHz" in meta['cpu']), "",
    old=(45,))

# ── new rows for text 42c4557 changed: L's CHECKs (tex6:201), the docstring correction (tex6:495), L's torch (tex6:499) ──
pr_ids = sorted(PRINTED_INFO, key=lambda k: int(_re.match(r"\d+", k).group()))
pr_txt = "; ".join(f"CHECK {k} (log:{PRINTED_INFO[k]['line']}) -> {', '.join(PRINTED_INFO[k]['marks'])}, records written on "
                   f"{PRINTED_INFO[k]['written']}" for k in pr_ids)
side_txt = "; ".join(f"CHECK {k}: " + "; ".join(f"log:{n} '{t[:110]}'" for n, t in v) for k, v in sorted(SIDE.items()))
inf_txt = "; ".join(f"CHECK {k} (informative, not asserted): 'curves equal False' at log:{v}" for k, v in sorted(INFORM.items()) if v)
chk_l = (f"L's log at {LLOG_REF}, second start (log:{SEG2}-{len(LLOG)}): 'ALL VERIFICATION CHECKS PASSED' x{n_pass2}, 'SOME ... FAILED' "
         f"x{n_fail2}, tracebacks {n_tb2}; CHECKs printed rather than asserted: {pr_txt}; every one of them compares with a record "
         f"another CPU wrote: {printed_other}; CHECK 130 {ck130}; CHECK 146 {ck146}; also printed and not asserted, beside an asserted "
         f"comparison with this machine's rerun: {side_txt}; {inf_txt}")
pr_ok = pr_ids == ["135", "141", "150"] and all(PRINTED_INFO[k]["differs"] for k in pr_ids) and n_fail2 == 0 and printed_other
row(("every CHECK comparing with L's own records asserted and passed, the three comparing with Xeon records printed and differing across CPUs, as the specification expects",
     "536 min of runs on six workers"),
    "L's CHECKs (tex6:201, reworded in 42c4557)",
    "On L (seeds 1560--1569, 54 runs; every CHECK comparing with L's own records asserted and passed, the three comparing with "
    "Xeon records printed and differing across CPUs, as the specification expects; 536 min of runs on six workers)",
    chk_l + f"; L meta.workers {LSTORE['meta'].get('workers')}; '{last_line[:80]}'",
    "PARTIAL" if pr_ok and SIDE else st(pr_ok),
    "Reworded on check_split_copy_L's PARTIAL (tex5:201 'every CHECK asserted on its own records'). The three CHECKs whose subject "
    "is a 2.10GHz Xeon record (135, 141, 150) are printed and DIFFER, and every asserted CHECK passed. Two more CHECKs (122, 123: "
    "Muon screens' code rerun on L, asserted IDENTICAL) also print, not asserted, a comparison with the screens' records written "
    "on the 2.10GHz host, and both print 'equal: False'; CHECK 112 (informative) prints 'curves equal False' against the batch "
    "screens' records (their CPU not recorded). So 'the three comparing with Xeon records' is exact only of the CHECKs whose "
    "subject is such a record. 'As the specification expects': the spec provides for printing them on another CPU (spec "
    "fb3acb7: CHECKs 135/141 'compare with records written on a Xeon @ 2.10GHz'; 150 'Asserted on a matching CPU, printed "
    "otherwise'); it does not predict that they differ. Wording only.")
DOC_SUBJ = {r: rlib.git("show", "-s", "--format=%h %cI %s", r).strip()[:150] for r in ("bbb600c", "80bfc1d")}
corr_ok = ("every CHECK passed; every CHECK that compares with L's own records was asserted" in D80
           and "the three that compare with 2.10GHz Xeon records were printed, not asserted, and DIFFER on this CPU" in D80
           and "every inherited and own CHECK passed, asserted" in DFF and "every inherited and own CHECK passed, asserted" not in D80
           and "(exit 0)" in DFF and "(exit 0: the second start's longrun json and .exit file are at results/L/longrun/" in D80)
row(("L's docstring corrected after the referee's check, \\texttt{80bfc1d}",), "provenance split_copy: 'L's docstring corrected after the referee's check, 80bfc1d' (new in 42c4557)",
    "L's docstring corrected after the referee's check, 80bfc1d",
    f"the referee's check 7b48cda ({cinfo('7b48cda')[8:27]}); {DOC_SUBJ['bbb600c']} (test_split_copy.py +{DOC_NUM['bbb600c'][0]}/-{DOC_NUM['bbb600c'][1]}); "
    f"{DOC_SUBJ['80bfc1d']} (test_split_copy.py +{DOC_NUM['80bfc1d'][0]}/-{DOC_NUM['80bfc1d'][1]}, and the longrun json and .exit); "
    f"bbb600c is 80bfc1d's parent: {rlib.git('rev-parse', '--short', '80bfc1d^').strip() == 'bbb600c'}; code after the docstring "
    f"unchanged at both: {doc_code_same}; 80bfc1d is the main line's last commit before the sixth draft: {HEAD_SHORT == '80bfc1d'}; "
    f"the corrections are in 80bfc1d's docstring: {corr_ok}",
    st(corr_ok and doc_code_same and HEAD_SHORT == "80bfc1d" and ctime("7b48cda") < ctime("bbb600c") < ctime("80bfc1d") < ctime(NEW_REF)),
    "The commit named holds the corrected docstring and is the main-line head the draft saw. The CHECK correction itself "
    "(doc:125) is bbb600c, 80bfc1d's parent; 80bfc1d adds the exit-code provenance and the longrun files. Section 6 checks "
    "the corrected text against L's log.")
row(("PyTorch 2.14.0 (L's \\texttt{channel\\_binding} rerun and its \\texttt{split\\_copy} run 2.14.1)",),
    "caption: L's split_copy run on PyTorch 2.14.1 (new in 42c4557)", "L's channel_binding rerun and its split_copy run 2.14.1",
    f"L's split_copy meta.torch {LSTORE['meta'].get('torch')}; per record {sorted({r.get('torch') for a in ARMS for r in RL[a].values()})}; "
    f"X's split_copy meta.torch {meta.get('torch')}",
    st(LSTORE["meta"].get("torch") == "2.14.1"),
    "The channel_binding rerun's version is not checked here. X's split_copy ran on 2.14.0+cu130, which the caption's '2.14.0' covers.")

# section 6: the RESULT (L) docstring, corrected (bbb600c, 80bfc1d), against L's log and the longrun files
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['(exit 0']}", "exit 0: the second start's longrun json and .exit file are at results/L/longrun/",
     f"80bfc1d results/L/longrun/split_copy_L.exit = '{L2.get('exit')}'; json starts {L2.get('n_starts')}, started '{L2.get('started')}' "
     f"(epoch {L2.get('started_utc')} UTC), boot {L2.get('boot', '')[:8]}, git {L2.get('git')}, resumes {L2.get('resumes')}; the log's second "
     f"header (log:{L_HDR[1][0] if len(L_HDR) > 1 else '-'}) has the same start time and boot: {l2_boot_in_log}",
     st(L2.get("exit") == "0" and l2_boot_in_log and L2.get("git") == "592c574"),
     "Was NOT RECORD-CHECKABLE at ffb0771 (check_split_copy_L doc:122: no exit file committed); now recorded.")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['(exit 0']}", "the json's epoch and local time give L's offset, UTC-6",
     f"second start: local {L2.get('started')}, epoch {L2.get('started_utc')} UTC, offset {l2_utc_off}; first start (84a7e57): local "
     f"{L_ATTEMPT['started'] if L_ATTEMPT else '-'}, epoch {L_ATTEMPT['started_utc'] if L_ATTEMPT else '-'} UTC",
     st(l2_utc_off == dt.timedelta(hours=6)), "")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['(exit 0']}", "as the user sent them", f"80bfc1d's author and message: '{DOC_SUBJ['80bfc1d'][:120]}'",
     "NOT RECORD-CHECKABLE", "Who sent the files is not in git.")
drow(f"80bfc1d doc:{DOCV['80bfc1d'][0] + 2}", "one start for the runs", f"longrun headers {[(n, l_[:60]) for n, l_ in L_HDR]}; run lines before the "
     f"second header {first_runs}", st(len(L_HDR) == 2 and first_runs == 0), "Reworded in bbb600c from 'one start' (check_split_copy_L's PARTIAL).")
drow(f"80bfc1d doc:{DOCV['80bfc1d'][0] + 5}", "the earlier start stopped at verification with 0 runs (CHECK 155: --also a directory); its log at the "
     "top of split_copy_L.log; longrun's counter reset between the two, so both headers read 'start 1'",
     f"first segment log:1-{SEG2 - 1}: CHECK 155 WRONG {first_155}, 'SOME VERIFICATION CHECKS FAILED' {first_failed}, run lines {first_runs}; both "
     f"headers 'start 1': {all('start 1 at' in l_ for _, l_ in L_HDR)}; the first start's own json (84a7e57, default branch) has "
     f"{L_ATTEMPT['n_starts'] if L_ATTEMPT else '-'} start, exit {L_ATTEMPT['exit'] if L_ATTEMPT else '-'}, boot in the first header {l1_boot_in_log}; the "
     f"second's json (80bfc1d) {L2.get('n_starts')} start",
     st(first_155 and first_failed and first_runs == 0 and all("start 1 at" in l_ for _, l_ in L_HDR) and L_ATTEMPT and L_ATTEMPT["n_starts"] == 1
        and L2.get("n_starts") == 1 and l1_boot_in_log),
     "Two separate json files, each with one start: consistent with a reset counter.")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['Verification:']}", "every CHECK passed",
     f"second start: 'ALL VERIFICATION CHECKS PASSED' x{n_pass2}, 'SOME ... FAILED' x{n_fail2}, tracebacks {n_tb2}",
     st(n_fail2 == 0 and n_tb2 == 0 and n_pass2 > 0), "")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['Verification:']}", "every CHECK that compares with L's own records was asserted",
     f"CHECKs that print rather than assert: {pr_ids}; their records were written on {sorted({w for k in pr_ids for w in PRINTED_INFO[k]['written']})} "
     f"(none on this CPU: {printed_other}); CHECK 130 {ck130}, CHECK 146 {ck146}", st(printed_other and set(ck130) == {"IDENTICAL"} and set(ck146) == {"IDENTICAL"}),
     "Replaces 'every inherited and own CHECK passed, asserted' (ffb0771 doc:125, check_split_copy_L's DISAGREE).")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['Verification:']}", "the three that compare with 2.10GHz Xeon records were printed, not asserted, and DIFFER on this CPU "
     "(the inherited CHECKs 135 and 141, and the test's own CHECK 150)", pr_txt + f"; also: {side_txt}; {inf_txt}",
     "PARTIAL" if pr_ok and SIDE else st(pr_ok),
     "Right of the three CHECKs whose subject is a Xeon record. Not the only Xeon comparisons printed without assertion: "
     "CHECKs 122 and 123 (asserted IDENTICAL against L's own rerun of S35's and S25's code) each print a not-asserted comparison "
     "with the screen's record 'written on the 2.10GHz host', 'equal: False'; CHECK 112 is informative ('curves equal False', "
     "the screens' CPU not recorded). CHECK 135's second row compares with local_gate_results.json 'written on not recorded'.")
drow(f"80bfc1d doc:{DOCLN['80bfc1d']['Verification:'] + 2}", "CHECK 130 asserted (L's test_early_recipe file written on this CPU); CHECK 146 "
     "bit for bit at 1541 through 7200 (asserted); CHECK 151(a) 65 records v1 = v2; 151(b) 30 records, no disagreement",
     f"CHECK 130 {ck130} (log:{CK['130']['line']}); CHECK 146 {ck146} (log:{CK['146']['line']}); 151(a) '{' '.join(ck151a.split())[-60:]}'; "
     f"151(b) {ck151b}",
     st(set(ck130) == {"IDENTICAL"} and set(ck146) == {"IDENTICAL"} and "65 records; disagreements 0" in ck151a
        and any("30 records, disagreements 0" in x for x in ck151b)), "Unchanged text.")
drow("docstring code", "bbb600c and 80bfc1d change the docstring only", f"code after the docstring identical at {DOC_REFS}: {doc_code_same}; "
     f"numstat bbb600c {DOC_NUM['bbb600c']}, 80bfc1d {DOC_NUM['80bfc1d']}", st(doc_code_same), "")

# ── fifth-pass rows (read from check_split_copy_v5.py's source) located in the sixth draft ───────────────
import ast  # noqa: E402
import json as _json  # noqa: E402

V3_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_split_copy_v5.py")).read()
V3_ROWS = []    # (kind, item, phrases): kind "row" (counted, section 1 of v5) or "row4" (section 4 of v5)
V5_NONLIT = []  # v5 calls whose fragment or item is built at run time (the provenance-row loop): regenerated below
for node in ast.walk(ast.parse(V3_SRC)):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("row", "row4"):
        try:
            ph = tuple(ast.literal_eval(node.args[0]))
            it_ = ast.literal_eval(node.args[1])
        except ValueError:
            V5_NONLIT.append((node.func.id, node.lineno))
            continue
        V3_ROWS.append((node.func.id, it_, ph))
V3_ROWS.sort(key=lambda x: (x[0], locate(tex5, x[2]) or 0))
for k_, it, ph in V3_ROWS:
    assert locate(tex5, ph) is not None, (it, ph)          # every fifth-pass fragment is in c5e4508


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


# test_delta_recipe's status (the main line as of the sixth draft, 80bfc1d: X complete at 8ae660d; L's half running per
# the main line's round-5 prompts, 0754d61)
DR = {c: cinfo(c) for c in ("05256c5", "fad1489", "32c121b", "a581027", "16b04f9", "6f2788d", "55dac50", "2225f00", "8ae660d")}
ALL_HEADS = {r: rlib.git("rev-parse", "--short=7", r).strip() for r in refs}
DR_RES_5 = {r: [f for f in TREE[r] if "delta_recipe" in f and (f.startswith("results/") or f.endswith((".json", ".log")))
                and not f.startswith("docs/")] for r in TIP}
DR_RES_5 = {r: v for r, v in DR_RES_5.items() if v}
XDR = rlib.load(HEAD, "results/X/delta_recipe_results.json")
XDR_N = sum(1 for r in XDR["runs"].values() if r.get("ok"))
XDR_STARTED = dt.datetime.strptime(XDR["meta"]["started"], "%Y-%m-%d %H:%M:%S")       # X records UTC (caption tex6:499)
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
       f"meta.git {XDR['meta']['git']}; L delta_recipe files at the sixth draft: {L_DR5 or 'none'}")
x_done = XDR_N == 84 and t_x < ctime(NEW_REF) and not L_DR5
ROUND5 = at(HEAD, "docs/reading/parallel_sessions_prompts.md") or ""
r5_hdr = [i + 1 for i, ln in enumerate(ROUND5.split("\n")) if ln.startswith("## Round 5") and "L runs test_delta_recipe" in ln]
r5_c = rlib.git("log", "--diff-filter=AM", "--format=%h", "-1", HEAD, "--", "docs/reading/parallel_sessions_prompts.md").strip()
r5_txt = (f"the main line's round-5 prompts ({r5_c}, {cinfo(r5_c)[8:27]}) docs/reading/parallel_sessions_prompts.md:{r5_hdr}: "
          f"'## Round 5 (12 October, evening): ... while X adopts the register spec, L runs test_delta_recipe ...'" if r5_hdr else "no round-5 header naming L")
L_running = bool(r5_hdr) and not L_DR5
chk_runs_h = (XDR_STARTED - t_check).total_seconds() / 3600
ver_h = (XDR_STARTED - PROC_START).total_seconds() / 3600
row4(("its test file checked against the amended spec before any arm's run (the check preceded the runs by two and a half hours; X's verification took three)",),
     "delta_recipe (Rev. 9 paragraph): 'checked ... before any arm's run (the check preceded the runs by two and a half hours; X's "
     "verification took three)' (new in 42c4557)", "check before any arm's run; 2.5 h before the runs; verification three hours",
     dr5 + f"; check -> runs {chk_runs_h:.2f} h; process start -> runs (the verification) {ver_h:.2f} h",
     st(t_check < XDR_STARTED and 2.25 <= chk_runs_h <= 2.75 and 2.75 <= ver_h <= 3.5 and t_amend1 < PROC_START),
     "Takes the fifth pass's correction (the runs began at 05:27 after a 3 h 15 min verification). 'Two and a half hours' is "
     f"{chk_runs_h:.2f} h; 'three' is {ver_h:.2f} h (rounded down).")
row4(("L's half is next", "\\texttt{test\\_delta\\_recipe}'s L half has not run"),
     "delta_recipe (Rev. 9 paragraph): 'L's half is next'; 'test_delta_recipe's L half has not run'",
     "L next; has not run", f"{r5_txt}; L delta_recipe files at the cutoff: {L_DR5 or 'none'}; tex6:54 says '(X; L running)', tex6:360 'L's half is "
     f"running', tex6:36 'runs on the other'", "PARTIAL" if L_running else st(not L_DR5),
     "Unchanged text made stale by this version's own tex6:36, 54 and 360: by the main line's round-5 prompts (committed 39 min "
     "before the draft) L runs test_delta_recipe. 'Has not run' is true of results (no L file); 'is next' is not the state. The "
     "draft says both.")
row4(("\\texttt{delta\\_recipe} (X; L running)", "on X (secondary): D4 SHOWN 10 vs 1", "pooled labels and readings wait for L"),
     "delta_recipe index row (re-anchored): '(X; L running)'", "(X; L running) ... pooled labels and readings wait for L", f"{dr5}; {r5_txt}",
     st(x_done and L_running),
     "Resolves the fifth pass's PARTIAL ('(specified)'). 'L running' is not in any record (no L file); it is what the main line's "
     "round-5 prompts say. The D numbers are the X-results check's.")
row4(("A pre-registered test is specified and complete on X (Section~\\ref{sec:memory}); L's half is running",),
     "assessment, the Hebbian write withdrawn: 'A pre-registered test is specified and complete on X; L's half is running' (new in 42c4557)",
     "complete on X; L running", f"{dr5}; {r5_txt}", st(x_done and L_running), "As the index row.")
row4(("its pre-registered test is complete on one machine", "and runs on the other"),
     "abstract: 'its pre-registered test is complete on one machine (...) and runs on the other' (status only; new in 42c4557)",
     "complete on X; running on L", f"{dr5}; {r5_txt}", st(x_done and L_running),
     "Status only. The numbers in the parenthesis are check_F_v6's (T rows) and the X-results check's.")
row4(("with the test file checked against the amended spec before any arm's run; run on X, where a second amendment confined to the report's rules",),
     "delta_recipe, memory section (re-anchored): 'checked ... before any arm's run; run on X, where a second amendment ... was committed during the run'",
     "checked before any arm's run; run on X", dr5, st(x_done and t_check < XDR_STARTED < t_am2 < t_x),
     "Resolves the fifth pass's PARTIAL ('running on X').")
row4(("none of them a claim, a reading or a Hebbian learned-gate run; two were the Hebbian perfect gate's",),
     "delta_recipe, memory section (re-anchored): '24 run lines ... none ... a Hebbian learned-gate run; two were the Hebbian perfect gate's'",
     "no Hebbian learned-gate run; two Hebbian perfect-gate runs", f"Amendment 2: 24 lines = CEIL_HEBB 580, 581; CEIL_DELTA 580, 581; "
     f"DELTA_SPLIT 580-598; DELTA_SLOW|580; 'None of HEBB_SPLIT or HEBB_SLOW had finished': {AM2_LINES}", st(AM2_LINES),
     "Resolves the fifth pass's DISAGREE ('none ... a Hebbian run').")
r55, r22, r8a = (rlib.git("show", "--name-only", "--format=", c).split() for c in ("55dac50", "2225f00", "8ae660d"))
prov_dr_ok = (x_done and "specs/test_delta_recipe.md" in r55 and "test_delta_recipe.py" in r22 and "results/X/delta_recipe_results.json" in r8a
              and XDR_STARTED < t_am2 < t_22 < t_x and XDR["meta"]["git"] == "a581027")
row4(("Amendment~2 during the run \\texttt{55dac50}, \\texttt{2225f00} / \\texttt{8ae660d} (X) & X complete; L to run & 84; ---",),
     "delta_recipe provenance row (unchanged): 'X complete; L to run'", "spec 05256c5, Amendment 1 32c121b / test a581027, Amendment 2 during "
     "the run 55dac50, 2225f00 / 8ae660d (X) & X complete; L to run & 84; ---",
     dr5 + f"; 55dac50 changes {r55}; 2225f00 changes {r22}; 8ae660d changes {r8a}; {r5_txt}",
     "PARTIAL" if prov_dr_ok and L_running else st(prov_dr_ok),
     "The commits and the X count are right (as in v5). 'L to run' is stale beside this version's '(X; L running)' (tex6:54), "
     "'L's half is running' (tex6:360) and 'runs on the other' (tex6:36), and against the main line's round-5 prompts.")
row4(("\\texttt{test\\_delta\\_recipe}'s L half (X's is complete)",), "next steps tex6:476: 'test_delta_recipe's L half (X's is complete)'",
     "L half next", f"{dr5}; {r5_txt}", st(x_done),
     "Unchanged. L's half running (the round-5 prompts) and listed as the next step are compatible; the item is not stale.")
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
     "method: the count test_delta_recipe prints (code at the main line as of the sixth draft)",
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
     "complete on both machines", f"{L_REC} at {L_RES} ({cinfo(L_RES)}); on branches at the sixth draft: {l_file_on}; {nL_runs} records, all ok {okL}",
     st(bool(l_file_on) and nL_runs == 54 and okL), "L's half landed at ffb0771 (02:39:51 UTC, 12 October), before the fifth draft.")

# ── the provenance tables (tex6:486-524): every commit hash, and each row's commits ─────────────
CUT_DT = int(rlib.git('show', '-s', '--format=%ct', NEW_REF).strip())
P0 = locate(tex6, ("\\begin{table}[H]",)) or 0
PROV_LINES = [i + 1 for i, ln in enumerate(tex6) if "label{tab:prov}" in ln or "label{tab:batches}" in ln]
PROV_A = max(i for i in range(PROV_LINES[0], 0, -1) if tex6[i - 1].startswith("\\begin{table}"))
PROV_B = PROV_LINES[-1]
HEX = {}
for i in range(PROV_A, PROV_B + 1):
    for h in __import__("re").findall(r"\\texttt\{([0-9a-f]{7})\}", tex6[i - 1]):
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
row4(("\\label{tab:batches}",), "provenance tables: every commit hash (tex6 tables and captions)", f"{len(HEX)} distinct hashes",
     f"{len(HEX)} distinct 7-hex hashes on tex6:{PROV_A}-{PROV_B}; exist {sum(x[2] for x in HROWS)}; predate 42c4557 "
     f"{sum(x[4] for x in HROWS)}; on some remote branch as of the cutoff {sum(1 for x in HROWS if x[5])} (each listed in section 4b)",
     st(all_exist and all_before and on_some), "")


def last_report(br):
    return TIP[br]


SESS = [("E 15--18", "origin/claude/outside-ideas", ["90c3a28", "93bdfe6", "2531480", "7ef4e15"], None),
        ("E 19", "origin/claude/outside-ideas", ["e3f5809"], "e3f5809"),
        ("H", "origin/claude/explore-H", ["7e9197f", "72f0bbc", "df57087", "37c1fbf", "1e09ecd", "de11b2a", "9b72bf6"], "9b72bf6"),
        ("F", "origin/claude/explore-F", ["9a6c2bf", "22ec912", "5b9bc2a", "ed76eea", "d5851b1"], "d5851b1"),
        ("G", "origin/claude/explore-G", ["6cd36ff", "427db36", "35de6db", "93422cf", "8973d48", "83699a2", "076005d", "b2f9907"], "076005d"),
        ("I", "origin/claude/explore-I", ["a2588da", "5f03929", "203a5ad", "86652b1", "b649751", "4c3984a", "5bc6c2e", "03bffdf", "559b942"], "559b942"),
        ("R", "origin/claude/replicate-R", ["de8df0f", "00990ea", "100dced", "464be8d", "1f8be04"], "1f8be04"),
        ("V", "origin/claude/review-V", ["d2bb959", "0fdb363", "ebb9325", "01bee1f", "4bf85e9", "0e6c189", "fad1489", "16b04f9", "6f2788d",
                                         "e8fbb25", "66334b8", "0d1fa60", "907eb2c", "e384a87"], "e384a87")]
SESS_OUT = {}
for name, br, cited, col in SESS:
    t_last = TIP[br]
    after = rlib.git("log", "--format=%h %s", f"{col}..{t_last}").strip().split("\n") if col else []
    after = [x for x in after if x.strip()]
    rep_after = [x for x in after if any(w in x.lower() for w in ("report", "pass", "complete", "write-up"))
                 and "progress" not in x.lower() and "EXPLORATORY" not in x
                 and not any(w in x for w in ("WIP", "interim", "pre-run note", "protocol"))]   # v6: round-5 WIP commits are not reports
    SESS_OUT[name] = dict(br=br, cited=cited, on=all(ancestor(c, t_last) for c in cited), last=t_last, col=col,
                          after=after, rep_after=rep_after)
for name, o in SESS_OUT.items():
    if o["col"] is None:
        continue
    lab = {"E 19": "19 (E) &", "H": "H & S58, S58b, S60", "F": "F & S53, S54, S59", "G": "G & S55 probe", "I": "I & S74 pilot",
           "R": "R & Phase 2", "V": "V & review"}[name]
    stv = "MATCH" if o["on"] and not o["rep_after"] else "PARTIAL"
    note = {"V": "Brought to the fifth pass e384a87 in 42c4557 (the fifth pass's PARTIAL resolved); V's last commit before the cutoff "
                 "is e384a87 (23:11:16 UTC), the column's.",
            "I": "Brought to 559b942 with 4c3984a in 42c4557 (the fifth pass's PARTIAL resolved). I's later commits (30807ec window_gate "
                 "pre-run note, 8cb8dce S75b driver, 9034b83 S75b interim CHECKs) are round 5, not a report.",
            "F": "F's last commit before the cutoff is cdbb886 (S88 queue script and progress, 23:07:46 UTC) after 40087cb (S88a/S88b "
                 "readings fixed before any run, 23:07:15): round 5 started 6 min before the draft; no report after d5851b1, so the column "
                 "and the count (42 for S81) are right.",
            "G": "G's later commits are S87 (specification and readings 3c89f87, progress): round 5, no report.",
            "H": "H's later commits are round 5 (5b5deec, f4fd535 S87a/S87c code and readings, progress): no report.",
            "R": "R's later commit is the Phase 5 protocol (550a736, 22:47:46 UTC), no result: the column is right.",
            "E 19": "E's later commits are batch 20 (code and readings 4acf915, runs and progress, 14 of 19 runs saved at 46cb8d0): no "
                    "report yet, so the column is right; tex6:41 now says batch 20 is running."}.get(name, "")
    if name == "I":
        i_recs = rlib.load(o["last"], "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json")
        n_i = len(i_recs.get("runs", i_recs))
        n_draft_i = 41
        stv = "PARTIAL" if n_i != n_draft_i else stv
        note += f" Records at {o['last']}: {n_i} (the row says 41)."
    row4((lab,), f"provenance row {name}: commits, and the last cited commit against the branch",
         f"cited {', '.join(o['cited'])}; commit column {o['col']}",
         f"all cited on {o['br'].split('/')[-1]} as of the cutoff: {o['on']}; the branch's last commit before the cutoff {o['last']}; commits "
         f"after the column's {o['col']}: {len(o['after'])}" + (f", of which reports or passes: {o['rep_after']}" if o["rep_after"] else ""),
         stv, note)
MAIN_CITED = ["74f5907", "3e2d171", "2a6ba3d", "9a769c7", "d05f0ca", "2f98f06", "886a668", "628eb7c", "1cd20a1", "fb3acb7", "1f52cd5",
              "592c574", "ffb0771", "ba8901a", "56c44a7", "05256c5", "32c121b", "a581027", "55dac50", "2225f00", "8ae660d",
              "9a53302", "9c53e16", "00ac0f2", "d0e6c85", "42af095", "b22da04", "ad8a92e", "80bfc1d"]
main_on = {c: ancestor(c, HEAD) for c in MAIN_CITED}
row4(("\\texttt{muon\\_recipe} & --- / \\texttt{74f5907}",), "main-line table and captions: every cited commit on the main line as of the cutoff",
     f"{len(MAIN_CITED)} commits", f"on {HEAD}: {sum(main_on.values())} of {len(MAIN_CITED)}; not: {[c for c, v in main_on.items() if not v] or 'none'}",
     st(all(main_on.values())), "Commit check only; the muon_recipe and window_gate rows are unchanged since the earlier passes.")
# L's start times caption (unchanged): both starts' epochs now committed
offs = [(e["ref"].replace("origin/", ""), e["started"], e["started_utc"],
         dt.datetime.strptime(e["started_utc"], "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(e["started"], "%Y-%m-%d %H:%M:%S")) for e in L_STARTS]
row4(("L's recorded start times are local time (UTC$-$6)",), "L start times in local time (UTC-6), caption",
     "L's recorded start times are local time (UTC-6)",
     "; ".join(f"{r_}: '{lo_}' local, epoch {ut_} UTC, offset {d_}" for r_, lo_, ut_, d_ in offs)
     + f"; L's result meta.started '{LSTORE['meta'].get('started')}' (RESULT (L): 'started 09:31 local = 15:31 UTC')",
     st(len(offs) == 2 and all(d_ == dt.timedelta(hours=6) for *_, d_ in offs)),
     "Unchanged text; now checkable for both starts (the second start's json landed at 80bfc1d).")
# the draft's status lines at the cutoff: tex6:41 (the sessions) and the next steps tex6:476-478
R5_BR = {"V": "origin/claude/review-V", "G": "origin/claude/explore-G", "H": "origin/claude/explore-H", "F": "origin/claude/explore-F",
         "R": "origin/claude/replicate-R", "I": "origin/claude/explore-I"}
t_r5 = ctime(r5_c) if r5_c else None
R5_AFTER = {k: [x for x in rlib.git("log", "--format=%h %ad %s", "--date=format:%H:%M:%S", f"--since={tstamp(r5_c)}", TIP[b_]).strip().split("\n") if x.strip()]
            for k, b_ in R5_BR.items()}
E_TIP = TIP["origin/claude/outside-ideas"]
E_RD = at(E_TIP, "explore_out/README.md") or ""
e_prog = [ln for ln in E_RD.split("\n") if "batch20.log" in ln and "IN PROGRESS" in ln]
e_code = at(E_TIP, "explore_batch20.py") or ""
e_86 = "86,400" in e_code and "S67" in e_code
row4(("E's batch~20 (S67's unbound runs continued to 86{,}400 updates) and the sessions' fifth round are running",),
     "status: 'E's batch 20 (S67's unbound runs continued to 86,400 updates) and the sessions' fifth round are running' (new in 42c4557)",
     "batch 20 running; round 5 running",
     f"E at {E_TIP} ({cinfo(E_TIP)[8:27]}, '{subj(E_TIP)[:60]}'); README: '{e_prog[0][:200] if e_prog else '-'}'; explore_batch20.py continues "
     f"S67's runs to 86,400: {e_86}; round-5 prompts {r5_c} ({cinfo(r5_c)[8:27]}) for V, G, H, F, R, I; commits on each branch after them "
     f"and before the cutoff: " + "; ".join(f"{k} {len(v)} (last '{v[0][:70]}')" for k, v in R5_AFTER.items() if v)
     + f"; without any: {[k for k, v in R5_AFTER.items() if not v] or 'none'}",
     st(bool(e_prog) and e_86 and all(R5_AFTER.values())),
     "Replaces the fifth draft's 'every session's queue is empty' (the fifth pass's DISAGREE). R's round-5 commit is a protocol "
     "(550a736), not runs; 'running' is read of the round.")
f81 = locate(tex6, ("session~F's trigger on delta channels (S81)",))
w478 = locate(tex6, ("the window gate without the split on the present configuration first",))
i_ws = rlib.git("log", "--format=%h %s", "-1", TIP["origin/claude/explore-I"], "--", "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json").strip()
row4(("a calibrated reading for any further replication phase (Section~\\ref{sec:method})",),
     "next steps tex6:477: S81 no longer listed (changed in 42c4557)", "S81 removed",
     f"'session~F's trigger on delta channels (S81)' in tex6: {f81}; S81 complete at d5851b1 (report 7), in the draft at tex6:205",
     st(f81 is None), "Resolves the fifth pass's PARTIAL (tex5:477).")
row4(("The dense-loss task at four sources again, in session~I's committed follow-up", "a split that does not fire on a one-to-one map."),
     "next steps tex6:478: 'the window gate without the split on the present configuration first' removed (changed in 42c4557)",
     "removed", f"phrase in tex6: {w478}; I's last S75 records commit before the cutoff: '{i_ws[:110]}'; I's report 2 section 6 at 559b942 "
     f"describes WIDE_SLOW 830-838 (seed 839 pending)", st(w478 is None and ("4c3984a" in i_ws or "559b942" in i_ws)),
     "The window gate without the split ran on nine of ten seeds (4c3984a), described at 559b942; the tenth (839) was pending, "
     "which tex6:458 says. Removing it from the next steps matches the records.")

# ── output ───────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
new_of = {}
for i, r in enumerate(rows):
    for o in r[6]:
        new_of.setdefault(o, []).append(i)


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


P = print
P("# Revision 9 draft, sixth version, vs test_split_copy: referee check (sixth pass)\n")
P(f"Sixth draft `{DRAFT}` at `{NEW_REF}` (committed {CUTOFF}; `tex6:NNN`, {len(tex6) - (tex6[-1] == '')} lines); fifth "
  f"draft `{V5_REF}` (`tex5:NNN`, {len(tex5) - (tex5[-1] == '')} lines; 42c4557's changes since c5e4508 replace lines in place, so "
  f"line numbers do not move); first draft `{OLD_REF}` (`tex:NNN`). X's records `{REC}` at `{RES}` (X's JSON unchanged at the main "
  f"line as of the sixth draft, `{HEAD_SHORT}`: {json_unchanged}); L's records `{L_REC}` at `{L_RES}` ({nL_runs} records, used "
  f"LIGHTLY: counts, discordant pairs and exact p for the pooled sentences, firings for the all-position count; L's half in "
  f"depth is check_split_copy_L's); L's log `{L_RES}:results/L/split_copy_L.log` for the CHECK wording; L's longrun files at "
  f"{', '.join(e['ref'].replace('origin/', '') + ' @ ' + e['commit'][:7] for e in L_STARTS)}; window_gate records and code at "
  f"`{HEAD_SHORT}`; test docstring at `{RES}`, `bbb600c`, `80bfc1d`; spec at `{SPEC_C}`. Outcomes recomputed with `rlib`; they "
  f"agree with X's records' own transition ({agree_t}) and bound-routed fields ({agree_rec}); ch_map ties: {len(rlib.TIES)}. "
  f"Generated by `check_split_copy_v6.py` (a copy of v5 re-anchored on 42c4557).\n")
cnt = {k: sum(1 for r in rows if r[4] == k) for k in ORDER}
removed = [i for i, o in enumerate(OLD) if locate(tex6, o[3]) is None]
v3_main = [x for x in V3_ROWS if x[0] == "row"]
v3_s4 = [x for x in V3_ROWS if x[0] == "row4"]
v3_gone = [x for x in v3_main if locate(tex6, x[2]) is None]
v3_s4_gone = [x for x in v3_s4 if locate(tex6, x[2]) is None]
P("Counts against the sixth draft: " + ", ".join(f"{k} {v}" for k, v in cnt.items())
  + f" (of {len(rows)} rows). Fifth-pass section 1 fragments: {len(v3_main) - len(v3_gone)} of {len(v3_main)} found "
  f"verbatim in the sixth draft, {len(v3_gone)} reworded (re-anchored in section 1). Fifth-pass section 4 fragments: "
  f"{len(v3_s4) - len(v3_s4_gone)} of {len(v3_s4)} verbatim, {len(v3_s4_gone)} REMOVED or reworded (re-anchored in section 4); "
  f"{len(V5_NONLIT)} v5 call(s) built their fragment at run time (the provenance-row loop, v5 line(s) {[x[1] for x in V5_NONLIT]}) and "
  f"are regenerated in section 4. First-pass fragments: {len(OLD) - len(removed)} of {len(OLD)} verbatim, {len(removed)} REMOVED or reworded.\n")
c4 = {k: sum(1 for r in rows4 if r[4] == k) for k in ORDER}
P("Section 4 (test_delta_recipe's status lines, R's probe-mode count, L's half, the status lines tex6:41 and tex6:476-478, and "
  "the provenance tables; not in the counts above): " + ", ".join(f"{k} {v}" for k, v in c4.items()) + f" (of {len(rows4)} rows).\n")
c6 = {k: sum(1 for r in DROWS if r[3] == k) for k in ORDER}
P("Section 6 (test_split_copy.py's RESULT (L) docstring as corrected in bbb600c and 80bfc1d, against L's log and longrun files; "
  "not in the counts above): " + ", ".join(f"{k} {v}" for k, v in c6.items()) + f" (of {len(DROWS)} rows).\n")

P("## 1. Rows against the sixth draft\n")
P("| tex6 line | item | draft says | recomputed | status | note | first-pass row |")
P("|---|---|---|---|---|---|---|")
for n, item, d, rc, s, note, old in sorted(rows, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex6:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} | "
      f"{', '.join(f'#{o + 1} (tex:{OLD[o][0]})' for o in old) or 'new'} |")

MAP5TO6 = {
    "assessment: row copy breaks merges": "assessment: row copy breaks merges (re-anchored",
    "limits: all-position targets (new)": "limits: all-position targets (re-anchored",
    "delta_recipe (Rev. 9 paragraph): 'amended once before any run ..., checked against the amended spec, and complete on X (a second "
    "amendment ... during X's run ...); L's half is next' (re-anchored)": "delta_recipe (Rev. 9 paragraph): 'checked",
    "delta_recipe index row (re-anchored): status": "delta_recipe index row (re-anchored): '(X; L running)'",
    "delta_recipe, memory section (re-anchored): 'running on X, where a second amendment ... was committed during the run'":
        "delta_recipe, memory section (re-anchored): 'checked",
    "delta_recipe, memory section: '24 run lines ... none of them a claim, a reading or a Hebbian run' (new)":
        "delta_recipe, memory section (re-anchored): '24 run lines",
}
P("\n## 2. Fifth-pass rows located in the sixth draft\n")
P("Each fifth-pass fragment (from `check_split_copy_v5.py`'s `row(...)` and `row4(...)` calls with literal fragments, read with "
  "`ast`) was asserted in `c5e4508` and searched verbatim in `42c4557`.\n")
P("| # | v5 section | fifth-pass item | fragment | tex5 | tex6 |")
P("|---|---|---|---|---|---|")
for i, (k_, it, ph) in enumerate(V3_ROWS):
    m5, m6 = locate(tex5, ph), locate(tex6, ph)
    pool = rows if k_ == "row" else rows4
    tgt = MAP5TO6.get(it)
    ra = [r_[0] for r_ in pool if tgt and r_[1].startswith(tgt)] if not m6 else []
    P(f"| {i + 1} | {'1' if k_ == 'row' else '4'} | {esc(it)} | {esc(' / '.join(ph))} | tex5:{m5} | "
      f"{f'tex6:{m6}' if m6 else (f'REWORDED; re-anchored at tex6:{ra[0]}' if ra else 'NOT RE-ANCHORED')} |")

P("\n## 3. First-pass rows located in the sixth draft\n")
P("| # | tex line | item | first-pass status | fragment | in sixth draft | now (section 1) |")
P("|---|---|---|---|---|---|---|")
for i, (n, item, s, ph) in enumerate(OLD):
    m = locate(tex6, ph)
    nows = "; ".join(f"tex6:{rows[j][0]} {rows[j][4]}" for j in new_of.get(i, [])) or "no row"
    P(f"| {i + 1} | tex:{n} | {esc(item)} | {s} | {esc(' / '.join(ph))} | {f'tex6:{m}' if m else 'REMOVED'} | {nows} |")

P("\n## 4. test_delta_recipe's status, R's count, L's half, the status lines and the provenance tables (as of the sixth draft)\n")
P(f"Each branch is read at its last commit dated at or before {CUTOFF}, except where a row says 'now'. Commit dates are not "
  "push dates.\n")
P("| tex6 line | item | draft says | record | status | note |")
P("|---|---|---|---|---|---|")
for n, item, d, rc, s, note in sorted(rows4, key=lambda r: (ORDER[r[4]], r[0])):
    P(f"| tex6:{n} | {esc(item)} | {esc(d)} | {esc(rc)} | **{s}** | {esc(note)} |")
P(f"\n### 4b. Every commit hash in the provenance tables and captions (tex6:{PROV_A}-{PROV_B})\n")
P("| hash | tex6 lines | exists | committed (UTC) | before 42c4557 | on (as of the cutoff) | subject |")
P("|---|---|---|---|---|---|---|")
for h, lns, ok_, t_, bef, brs, sj in HROWS:
    P(f"| {h} | {', '.join(map(str, lns))} | {ok_} | {dt.datetime.fromtimestamp(t_, dt.timezone.utc):%Y-%m-%d %H:%M:%S} | {bef} | "
      f"{', '.join(brs)} | {esc(sj)} |")
P("\n### 4c. Each branch's last commit before the cutoff (23:14:04 UTC, 12 October) against the draft's provenance column\n")
P("| row | branch | commit column | branch's last commit before the cutoff | commits after the column | reports or passes after it |")
P("|---|---|---|---|---|---|")
for name, o in SESS_OUT.items():
    P(f"| {name} | {o['br'].replace('origin/', '')} | {o['col'] or '-'} | {o['last']} ({cinfo(o['last'])[8:27]}) | {len(o['after'])} | "
      f"{esc('; '.join(o['rep_after'])[:400]) or 'none'} |")
P(f"| main line | bdh-growth-hebbian-inference-w90069 | split_copy 80bfc1d (tex6:495), delta_recipe 8ae660d (tex6:496) | {HEAD_SHORT} "
  f"({cinfo(HEAD_SHORT)[8:27]}) | after 8ae660d: {len(rlib.git('log', '--format=%h', f'8ae660d..{HEAD_SHORT}').split())} "
  f"({', '.join(rlib.git('log', '--format=%h', f'8ae660d..{HEAD_SHORT}').split())}) | none (prompts 0754d61; the docstring correction "
  f"bbb600c and 80bfc1d) |")
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
P("\n## 6. test_split_copy.py's RESULT (L) docstring: what bbb600c and 80bfc1d changed, against L's log\n")
P(f"bbb600c ({cinfo('bbb600c')[8:27]}) changes test_split_copy.py by +{DOC_NUM['bbb600c'][0]}/-{DOC_NUM['bbb600c'][1]} lines and 80bfc1d "
  f"({cinfo('80bfc1d')[8:27]}) by +{DOC_NUM['80bfc1d'][0]}/-{DOC_NUM['80bfc1d'][1]} (and adds results/L/longrun/split_copy_L.json and .exit); "
  f"the code after the docstring is identical at ffb0771, bbb600c and 80bfc1d: {doc_code_same}. The RESULT (L) block starts at "
  + ", ".join(f"{r}:{DOCV[r][0]}" for r in DOC_REFS) + ". The two sentences check_split_copy_L read at ffb0771:\n")
P("| ref | the exit-code sentence | the verification sentence |")
P("|---|---|---|")
for r in DOC_REFS:
    t_ = DOCV[r][2]
    ex = t_[t_.index("536 min"):t_.index("An earlier start")].strip() if "An earlier start" in t_ else "?"
    ve = t_[t_.index("- Verification:"):t_.index("test_muon_recipe's CHECK 130")].strip()
    P(f"| {r} (doc:{DOCLN[r]['(exit 0']}, doc:{DOCLN[r]['Verification:']}) | {esc(ex)} | {esc(ve)} |")
P("\n| where | statement (80bfc1d) | record | status | note |")
P("|---|---|---|---|---|")
for w, st_, rc, s, note in sorted(DROWS, key=lambda r: ORDER[r[3]]):
    P(f"| {w} | {esc(st_)} | {esc(rc)} | **{s}** | {esc(note)} |")
P(f"\nL's longrun files on the branches as of the cutoff:\n")
P("| branch @ commit | started (local) | epoch (UTC) | boot | git | starts | exit | .log |")
P("|---|---|---|---|---|---|---|---|")
for e in L_STARTS:
    P(f"| {e['ref'].replace('origin/', '')} @ {e['commit']} | {e['started']} | {e['started_utc']} | {e['boot'][:8]} | {e['git']} | "
      f"{e['n_starts']} | {e['exit']} | {e['has_log']} |")
P("\n## 7. Corrections to the referee's earlier files (recorded here, not edited)\n")
P(f"- **`check_split_copy_L.md`, doc:125 and tex5:201 rows.** They list the CHECKs printed rather than asserted on L as 135, 141 "
  f"and 150 only. L's log also prints, not asserted, a comparison with a 2.10GHz-host record inside CHECKs 122 and 123 (both "
  f"'equal: False'; {side_txt}) and CHECK 112 is informative ({inf_txt}). The rows' verdicts (DISAGREE on 'every inherited and "
  f"own CHECK passed, asserted'; PARTIAL on tex5:201) stand; the lists were incomplete.")
P(f"- **`check_split_copy_L.md`, doc:122 '(exit 0)'.** NOT RECORD-CHECKABLE when written (at 7b48cda: no exit file committed); "
  f"80bfc1d has since committed the second start's .exit ('{L2.get('exit')}'), so it is now MATCH (section 6). Not an error, a change of record.")
P(("\nNo number in the sixth draft's split_copy statements disagrees with the records (X recomputed; L and pooled lightly). " if cnt["DISAGREE"] == 0 else
   f"\n{cnt['DISAGREE']} row(s) DISAGREE with the records. ")
  + "What remains is listed under PARTIAL above and in sections 4 and 6.")
