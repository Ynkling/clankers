#!/usr/bin/env python3
"""check_delta_recipe_L.py: Referee V's check of test_delta_recipe's L half and pooled block (main line b03d353,
16:05:18 UTC 13 October) against the records, the same way rev9/check_delta_recipe_X.py checked X's half.

    python3 docs/review_2026-10/check_delta_recipe_L.py      (stdlib only; prints; writes nothing)

Reads only git objects (git show REF:PATH); runs no training and does not run test_delta_recipe.py.
0. Inputs and code identity, by AST: test_delta_recipe.py at b03d353 (= the main line's head) against 8ae660d (L's run
   commit, its meta.git), 2225f00 (Amendment 2's report code) and a581027 (X's run code): which top-level definitions
   differ, and whether any of them is reachable from the run path (run_job, time_job, spec, init_worker).
1. L's run: results/L/delta_recipe_results.json, results/L/delta_recipe_L.log (two longrun starts),
   results/L/longrun/delta_recipe_L.json and .exit; seeds, CPU, torch, git, workers, starts; the times and the UTC
   offset (from the longrun json's epoch); every per-run line of the log against its record.
2. The test's own report: every report function of test_delta_recipe.py at 8ae660d (AST-identical to b03d353's), with
   its helpers extracted from their modules at 8ae660d, exec'd and run on L's records (a) without --also (the in-run
   report, diffed line by line against the log) and (b) with --also X's file (diffed against
   results/L/delta_recipe_L_report.log); (c) the same from X's side with --also L's file (the docstring's "either
   machine's file produces identically").
3. The referee's own recomputation, independent of the test's functions, per machine and pooled: outcomes with rlib,
   McNemar with rlib.mcnemar1, Newcombe with newcombe10 extracted from docs/review_2026-10/spec_checks_delta2.py,
   Wilson and bands with rlib, CHECK 174, D3, probe-blind (Amendment 2 form), firings and the readings written here from
   the specification's words (Amendments 1 and 2).
4. X's run log, committed at 7d7661d after the plan: its two reports against the regenerations, its per-run finish times
   against rev9/check_delta_recipe_X.md's reconstruction (Amendment 2's timing), and RESULT (X)'s verification bullet.
5. The statements: RESULT (L) and RESULT (POOLED, PRIMARY) (docstring at b03d353 = the main line's head), the b03d353
   commit message, results/README.md's row and note, docs/reading/plan_2026-10-13.md (6912e0b), each fragment located
   verbatim; a table of rows (line, statement, recomputed, status, note) and the counts.
"""
import ast
import collections
import contextlib
import datetime as dt
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True       # writes nothing, not even rlib's .pyc
import rlib  # noqa: E402

MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
REC_L, REC_X, A2CODE, RUNX, PLAN_C, XLOG_C = "b03d353", "8ae660d", "2225f00", "a581027", "6912e0b", "7d7661d"
A2SPEC = "55dac50"
TEST = "test_delta_recipe.py"
LRES, XRES = "results/L/delta_recipe_results.json", "results/X/delta_recipe_results.json"
LLOG, LREP = "results/L/delta_recipe_L.log", "results/L/delta_recipe_L_report.log"
LJSON, LEXIT = "results/L/longrun/delta_recipe_L.json", "results/L/longrun/delta_recipe_L.exit"
XLOG = "results/X/delta_recipe_X.log"
PLAN = "docs/reading/plan_2026-10-13.md"
rows = []          # (scope, line, statement, recomputed, status, note)


def say(*a):
    print(" ".join(str(x) for x in a))


def git(*args):
    return subprocess.check_output(("git", "-C", ROOT) + args).decode()


def src(ref, path):
    return git("show", f"{ref}:{path}")


def row(scope, line, statement, recomputed, status, note=""):
    rows.append((scope, line, statement, recomputed, status, note))


def segments(source):
    """{name: source segment} of every top-level def and simple assignment."""
    tree, out = ast.parse(source), {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = ast.get_source_segment(source, n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(e, ast.Name):
                        out[e.id] = ast.get_source_segment(source, n)
    return out


def body_dump(source):
    """ast.dump of the module without its docstring."""
    tree = ast.parse(source)
    b = tree.body
    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
        b = b[1:]
    return ast.dump(ast.Module(body=b, type_ignores=[]))


def extract(source, names, ns):
    """exec the top-level defs / assignments named in `names` (in source order) into ns."""
    tree = ast.parse(source)
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign):
            ids = [e.id for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
            if any(i in names for i in ids):
                body.append(n)
    exec(compile(ast.Module(body=body, type_ignores=[]), "extract", "exec"), ns)
    missing = [x for x in names if x not in ns]
    if missing:
        raise KeyError(f"not found: {missing}")
    return ns


def module(ref, path, names, **inject):
    ns = dict(math=math, statistics=statistics, collections=collections, json=json, os=os, **inject)
    extract(src(ref, path), names, ns)
    return types.SimpleNamespace(**ns)


def utc(s):
    return dt.datetime.fromisoformat(git("show", "-s", "--format=%cI", s).strip()).astimezone(dt.timezone.utc).replace(tzinfo=None)


# ════════════════════════════════════════════════════════════════════════════
say("=" * 100)
say("0. Inputs and code identity")
say("=" * 100)
HEAD = git("rev-parse", "--short=7", MAIN).strip()
for c in (RUNX, A2SPEC, A2CODE, REC_X, REC_L, PLAN_C, XLOG_C):
    say(f"  {git('show', '-s', '--format=%h %cI %s', c).strip()[:160]}")
say(f"  main line head {HEAD}; commits after {REC_L}: "
    + "; ".join(git("log", "--format=%h %cI", f"{REC_L}..{MAIN}").split("\n")[:-1]))
l_files = [LRES, LLOG, LREP, LJSON, LEXIT, TEST]
same_head = git("diff", "--name-only", REC_L, MAIN, "--", *l_files).strip() == ""
say(f"  L's files and {TEST} identical at {REC_L} and the head {HEAD}: {same_head}")
added = git("show", "--name-status", "--format=", REC_L).split("\n")
say(f"  {REC_L} touches: {[x for x in added if x]}")
xlog_refs = sorted({l for l in git("log", "--all", "--format=%h", "--", XLOG).split()})
say(f"  commits on any ref touching {XLOG}: {xlog_refs}")

tsrc = {r: src(r, TEST) for r in (RUNX, A2CODE, REC_X, REC_L)}
dumps = {r: body_dump(s) for r, s in tsrc.items()}
seg = {r: segments(s) for r, s in tsrc.items()}
say(f"  {TEST}: code (module minus docstring) equal by AST: {REC_X} = {REC_L}: {dumps[REC_X] == dumps[REC_L]}; "
    f"{A2CODE} = {REC_X}: {dumps[A2CODE] == dumps[REC_X]}; {RUNX} = {REC_X}: {dumps[RUNX] == dumps[REC_X]}")
doc_x, doc_l = ast.get_docstring(ast.parse(tsrc[REC_X]), clean=False), ast.get_docstring(ast.parse(tsrc[REC_L]), clean=False)
doc_prefix = doc_l.startswith(doc_x.rstrip())
doc_added = doc_l[len(doc_x.rstrip()):].strip()
say(f"  docstring at {REC_L} = {REC_X}'s + an appended block: {doc_prefix}; the appended block starts "
    f"{doc_added[:12]!r} and holds RESULT (L) and RESULT (POOLED, PRIMARY): "
    f"{doc_added.startswith('RESULT (L)') and 'RESULT (POOLED, PRIMARY)' in doc_added}")
diff_run = sorted(k for k in set(seg[RUNX]) | set(seg[REC_X]) if seg[RUNX].get(k) != seg[REC_X].get(k))
say(f"  top-level definitions differing between {RUNX} and {REC_X}: {diff_run}")


def refs_of(source):
    """{top-level name: set of top-level names its body refers to}."""
    tree = ast.parse(source)
    top = set(segments(source))
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.Assign)):
            names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & top
            keys = [n.name] if isinstance(n, ast.FunctionDef) else [e.id for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
            for k in keys:
                out[k] = out.get(k, set()) | names
    return out


RUN_ROOTS = {"run_job", "time_job", "spec", "init_worker"}
closure = {}
for r in (RUNX, REC_X):
    g, todo, seen = refs_of(tsrc[r]), list(RUN_ROOTS), set()
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        todo += list(g.get(x, ()))
    closure[r] = seen
run_path_same = closure[RUNX] == closure[REC_X] and all(seg[RUNX].get(k) == seg[REC_X].get(k) for k in closure[REC_X])
imports = {r: [ast.dump(n) for n in ast.parse(tsrc[r]).body if isinstance(n, (ast.Import, ast.ImportFrom))] for r in (RUNX, REC_X)}
say(f"  the run path ({sorted(RUN_ROOTS)} and every top-level name they reach: {len(closure[REC_X])} names) is identical "
    f"at {RUNX} and {REC_X}: {run_path_same}; differing names on it: {sorted(set(diff_run) & closure[REC_X]) or 'none'}; "
    f"imports identical: {imports[RUNX] == imports[REC_X]}")
other_py = [p for p in git("diff", "--name-only", RUNX, REC_X, "--", "*.py").split() if p != TEST]
other_py_l = [p for p in git("diff", "--name-only", REC_X, REC_L, "--", "*.py").split()
              if p != TEST and not p.startswith("docs/review_2026-10/")]
say(f"  other .py files changed {RUNX}..{REC_X}: {other_py}; {REC_X}..{REC_L} (outside the referee's directory): {other_py_l}")
other_same = {f"{p} {a}..{b}": body_dump(src(a, p)) == body_dump(src(b, p))
              for a, b, ps in ((RUNX, REC_X, other_py), (REC_X, REC_L, other_py_l)) for p in ps}
say(f"  ... their code (minus docstring) equal by AST: {other_same}")
CODE_NOTE = (f"{TEST}: {RUNX} vs {REC_X} differ only in {diff_run} (report rules and CHECKs 159/162's vendored records); "
             f"none is on the run path ({len(closure[REC_X])} names reached from {sorted(RUN_ROOTS)}): identical; "
             f"{REC_X} = {A2CODE} = {REC_L} in code; other .py changed: {list(other_same)}, docstring only "
             f"{all(other_same.values())}")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("1. L's run: records, log, longrun files, times")
say("=" * 100)
storeL = json.loads(src(REC_L, LRES))
storeX = json.loads(src(REC_X, XRES))
assert json.loads(src(MAIN, XRES)) == storeX
metaL, metaX = storeL["meta"], storeX["meta"]
RL, RX = storeL["runs"], storeX["runs"]
say(f"  {LRES}: {len(RL)} records, all ok {all(r.get('ok') for r in RL.values())}; meta.git {metaL['git']}, cpu "
    f"{metaL['cpu']}, torch {metaL['torch']}, workers {metaL['workers']}, started {metaL['started']!r}, 'starts' in meta "
    f"{'starts' in metaL}; projected_wall_h {metaL['projected_wall_h']:.3f}; window {metaL['window']}")
cpusL = collections.Counter(r.get("cpu") for r in RL.values())
say(f"  CPU labels on the records: {dict(cpusL)}; sum of secs {sum(r['secs'] for r in RL.values()) / 3600:.2f} h")
say(f"  {XRES} at {REC_X}: meta.git {metaX['git']}, torch {metaX['torch']}, workers {metaX['workers']}, started {metaX['started']}")
Llog = src(REC_L, LLOG)
LL = Llog.split("\n")
Lrep = src(REC_L, LREP).split("\n")
ljson = json.loads(src(REC_L, LJSON))
lexit = src(REC_L, LEXIT).strip()
hdr_re = re.compile(r"=== longrun delta_recipe_L: start (\d+) at (\S+ \S+), git (\w+), boot (\w+) ===")
hdrs = [(i, hdr_re.match(l).groups()) for i, l in enumerate(LL) if hdr_re.match(l)]
say(f"  log headers: {[(i + 1,) + h for i, h in hdrs]}")
starts = []
for j, (i, h) in enumerate(hdrs):
    end = hdrs[j + 1][0] if j + 1 < len(hdrs) else len(LL)
    part = LL[i:end]
    tl = next(l for l in part if l.strip().startswith("torch "))
    np_ = sorted(set(re.findall(r"NumPy (\S+) float64", "\n".join(part))))
    proj_i = [k for k, l in enumerate(part) if l.startswith("PROJECTION")]
    proj_lines = [l for l in part if "ms/step x" in l]
    starts.append(dict(line=i + 1, start=h[0], at=h[1], git=h[2], boot=h[3], torch_line=tl.strip(), numpy=np_,
                       passed=sum(1 for l in part if l.strip() == "ALL VERIFICATION CHECKS PASSED"),
                       fail=[l for l in part if re.search(r"-> (FAIL|NOT|MISMATCH)", l) or "Traceback" in l],
                       proj_heading=bool(proj_i), proj_lines=len(proj_lines),
                       runs_heading=any(l.startswith("RUNS —") for l in part),
                       run_lines=sum(1 for l in part if re.match(r"   (CEIL|HEBB|DELTA)_\w+\s+seed \d+ ", l)),
                       lines=(i + 1, end)))
for s in starts:
    say(f"  start at log line {s['line']} ('start {s['start']}' {s['at']}, git {s['git']}, boot {s['boot']}): {s['torch_line']}; "
        f"NumPy {s['numpy']}; 'ALL VERIFICATION CHECKS PASSED' x{s['passed']}; failing lines {len(s['fail'])}; PROJECTION "
        f"heading {s['proj_heading']}, projection lines {s['proj_lines']}; RUNS heading {s['runs_heading']}; run lines "
        f"{s['run_lines']}; log lines {s['lines']}")
s1, s2 = starts
proj1 = next(i + 1 for i, l in enumerate(LL) if l.startswith("PROJECTION") and i < s2["line"])
p1, p2 = LL[s1["lines"][0] - 1:s1["lines"][1]], LL[s2["lines"][0] - 1:s2["lines"][1]]
ver1 = p1[1:p1.index(next(l for l in p1 if l.startswith("PROJECTION")))]
ver2 = p2[1:p2.index(next(l for l in p2 if l.startswith("PROJECTION")))]
vdiff = [(a, b) for a, b in zip(ver1, ver2) if a != b]
say(f"  the two starts' verification output: {len(ver1)} vs {len(ver2)} lines; lines that differ {len(vdiff)}:")
for a, b in vdiff:
    say(f"     1st |{a[:150]}\n     2nd |{b[:150]}")
say(f"  the first start's last lines: {[l[:90] for l in p1[-4:]]}")
st0 = ljson["starts"]
t_ep = dt.datetime.fromtimestamp(st0[0]["t"], dt.timezone.utc).replace(tzinfo=None)
loc = dt.datetime.strptime(st0[0]["started"], "%Y-%m-%d %H:%M:%S")
OFF = round((loc - t_ep).total_seconds() / 3600, 3)
off_byte = Llog.encode().find(("=== longrun delta_recipe_L: start 1 at " + st0[0]["started"]).encode())
say(f"  {LJSON}: cmd {' '.join(ljson['cmd'])}; total {ljson['total']}; starts {len(st0)}: {st0}; resumes {ljson['resumes']}; "
    f"offset {ljson['offset']} (byte of the second header in the log: {off_byte}); {LEXIT}: {lexit!r}")
say(f"  the json's start: local {loc} vs epoch {st0[0]['t']:.1f} = {t_ep} UTC -> offset UTC{OFF:+g} h")
TZ = dt.timedelta(hours=-OFF)               # local -> UTC
START1 = dt.datetime.strptime(s1["at"], "%Y-%m-%d %H:%M:%S")
START2 = dt.datetime.strptime(s2["at"], "%Y-%m-%d %H:%M:%S")
STARTED = dt.datetime.strptime(metaL["started"], "%Y-%m-%d %H:%M:%S")
# the per-run lines of the second start
run_re = re.compile(r"   (\w+)\s+seed (\d+)  (.*?)  (\d+)s  \(elapsed ([\d.]+)m\)$")
runl = [run_re.match(l).groups() for l in p2 if run_re.match(l)]
wall_line = next(l for l in LL if "total wall clock after verification" in l)
wall_min = float(re.search(r"verification: ([\d.]+) min", wall_line).group(1))
wall_s = int(re.search(r"min \((\d+)s\)", wall_line).group(1))
T_LREC = utc(REC_L)
# report helpers (needed for the outcome labels of the run lines)
tbo = module(REC_X, "test_binding_onset.py", {"EVAL_EVERY", "fmt_step"})
trd = module(REC_X, "test_router_discovery.py", {"get", "load_store"})
tcl = module(REC_X, "test_conv_lr.py", {"bound_r", "mcnemar_greater"})
tsch = module(REC_X, "test_stream_channels.py", {"med_int", "mstr", "shared_max", "merged"})
trl = module(REC_X, "test_router_layout.py", {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"})
tsa = module(REC_X, "test_scale_axes.py", {"fail_class_k"}, fail_class=trl.fail_class)
tsr = module(REC_X, "test_stream_recipe.py", {"routed", "outcome", "wilson", "ROUTED_ACC"},
             fail_class_k=tsa.fail_class_k, merged=tsch.merged, shared_max=tsch.shared_max)
fc2 = module(REC_X, "fail_class_v2.py", {"fail_class_v2", "multichannel", "fail_class_k_v2", "outcome_v2"},
             MARGIN_PARTIAL=trl.MARGIN_PARTIAL, ETA_SPLIT=trl.ETA_SPLIT, merged=tsch.merged, shared_max=tsch.shared_max)
tscur = module(REC_X, "test_stream_curriculum.py", {"band_n", "RELIABLE_F", "MAJORITY_F"})
tss = module(REC_X, "test_slow_start.py", {"ok_r", "runs_of", "curve_str"}, get=trd.get)
trs = module(REC_X, "test_recipe_scope.py", {"groups_str"})
tgw = module(REC_X, "test_window_gate.py", {"ACC_THR", "RISE", "EVERY", "FIRST", "REF_AT", "MAX_SPLITS", "GAP", "VALID_N"})
assert "MAX_A, MAX_B, MAX_C = tsc.MAX_ITERS, tsr.MAX_ITERS, 43200" in src(REC_X, "test_window_gate.py")
tgw.MAX_C = 43200
REPORT_NAMES = {"MACHINES", "RANGES", "ITERS", "VALID_N", "D1_BOUND", "BETA", "OVERLAP_MAX", "ARMS", "ARM", "KEYS",
                "ARMS_RUN", "CEILS", "PARTS", "NO_ROOM", "D1_LABELS", "SPLIT_READINGS", "D1_OC", "seeds_for", "SEEDS",
                "outcome2", "success", "overlap", "paired", "sign_test", "d3_rows", "d3_of", "validity", "tally",
                "paired_full", "newcombe_paired", "d1_label", "split_verdict", "split_label", "split_reading", "pool2",
                "band_str", "diff_str", "print_claims", "broken_pairs", "check_174", "probe_blind", "print_bands",
                "complete", "stream_str", "raw_rows", "report", "other_of"}
STORES = {XRES: storeX, LRES: storeL, "delta_recipe_results.json": None}


def test_ns(ref):
    names = {n for n in REPORT_NAMES if n in segments(tsrc[ref])}
    ns = dict(math=math, statistics=statistics, collections=collections, tgw=tgw, tsr=tsr, tscur=tscur, tss=tss, fc2=fc2,
              ok_r=tss.ok_r, get=trd.get, bound_r=tcl.bound_r, mcnemar_greater=tcl.mcnemar_greater, med_int=tsch.med_int,
              mstr=tsch.mstr, shared_max=tsch.shared_max, fmt_step=tbo.fmt_step, EVAL_EVERY=tbo.EVAL_EVERY,
              groups_str=trs.groups_str, curve_str=tss.curve_str)
    extract(tsrc[ref], names, ns)
    ns["load_store"] = lambda p: STORES[p]
    return ns


T = test_ns(REC_X)
assert T["seeds_for"]("L") == {k: tuple(v) for k, v in metaL["seeds"].items()}
SEEDS_L = {k: tuple(v) for k, v in metaL["seeds"].items()}
SEEDS_X = {k: tuple(v) for k, v in metaX["seeds"].items()}
exp_keys = {f"{k}|{s}" for k, ss in SEEDS_L.items() for s in ss}
say(f"  seeds: meta.seeds = the test's seeds_for('L') {True}; RANGES['L'] = {T['RANGES']['L']}; perfect gates on "
    f"{SEEDS_L['CEIL_HEBB']}, {SEEDS_L['CEIL_DELTA']}; record keys = arm|seed over meta.seeds: {set(RL) == exp_keys}")
mism = []
for k, s, line, secs, el in runl:
    r = RL[f"{k}|{s}"]
    lab = ("BOUND" if tcl.bound_r(r) else "not bound") if T["ARM"][k]["ceil"] else T["outcome2"](r)
    if int(secs) != round(r["secs"]) or lab not in line or (r["transition"] and f"BOUND at {r['transition']}" not in line):
        mism.append((k, s, line[:60], secs, round(r["secs"])))
say(f"  per-run lines in the second start: {len(runl)}; distinct arm|seed {len({(k, s) for k, s, *_ in runl})}; lines "
    f"disagreeing with their record (secs, outcome, transition): {mism or 'none'}")
fin = {(k, int(s)): STARTED + dt.timedelta(minutes=float(el)) for k, s, _, _, el in runl}
P_MIN = sum(metaL["ms_per_step"].values()) * 1200 / 1000
first_sub = min(fin[(k, int(s))] - dt.timedelta(seconds=int(secs)) for k, s, _, secs, _ in runl[:4])
END_L = STARTED + dt.timedelta(minutes=wall_min)
say(f"  meta.started {STARTED} local = {STARTED + TZ} UTC (verification of the second start: {STARTED - START2}); "
    f"projection >= {P_MIN:.0f} s; first runs submitted about {first_sub:%H:%M:%S} local (first finish minus its secs; "
    f"elapsed is printed to 0.1 min)")
say(f"  last run line elapsed {runl[-1][4]} m = the report's wall clock {wall_min} min ({wall_line.strip()[:70]}); end "
    f"{END_L} local = {END_L + TZ} UTC; result commit {REC_L} {T_LREC} UTC ({(T_LREC - (END_L + TZ)).total_seconds() / 60:.1f} min later)")
say(f"  runs only (first submission to the last finish): {(END_L - first_sub).total_seconds() / 60:.1f} min of the "
    f"{wall_min} min after verification; the projection's share {(first_sub - STARTED).total_seconds() / 60:.1f} min")
say(f"  first start {START1} local = {START1 + TZ} UTC; second {START2} local = {START2 + TZ} UTC; the first was lost "
    f"between its PROJECTION heading and {START2} local (boot ids {s1['boot']} -> {s2['boot']})")
ckblk = {}
cur = None
for l in p2:
    m = re.match(r"CHECK (\d+)", l)
    if m:
        cur = m.group(1)
        ckblk.setdefault(cur, [])
    if cur:
        ckblk[cur].append(l)
na_uneq = sorted((int(k) for k, v in ckblk.items() if "not asserted" in " ".join(v)
                  and ("DIFFERS" in " ".join(v) or "equal False" in " ".join(v) or "equal: False" in " ".join(v))), key=int)
say(f"  second start: CHECK blocks with 'not asserted' and an unequal comparison: {na_uneq}")
for c in ("130", "146", "158", "161", "173", "144", "157", "170", "168"):
    v = ckblk.get(c, [])
    say(f"   CHECK {c}: " + " | ".join(x.strip()[:110] for x in v[1:4] if x.strip()))
first_cached = {c: bool(re.search(r"no cached [^()]*record \(a first start\)", " ".join(ckblk.get(c, [])))) for c in ("144", "157", "170")}
c130 = " ".join(ckblk["130"])
c158 = " ".join(ckblk["158"])
c161 = " ".join(ckblk["161"])
c173 = " ".join(ckblk["173"])
c146 = " ".join(ckblk["146"])
inrun_pooled = [l for l in p2 if l.startswith("POOLED (PRIMARY)")]
say(f"  in-run report's pooled line: {inrun_pooled}; CHECK 168 '--also' line: "
    f"{[x.strip() for x in ckblk['168'] if 'also' in x]}")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("2. The test's own report regenerated from the records (8ae660d's functions = b03d353's)")
say("=" * 100)


def run_report(ns, store, seeds, machine, wall, path, also):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["report"](store, seeds, machine, wall, path, also)
    return buf.getvalue().split("\n")


def linediff(a, b, skip=()):
    n = max(len(a), len(b))
    return [(i, a[i] if i < len(a) else None, b[i] if i < len(b) else None) for i in range(n)
            if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None) and not any(
                k in ((a[i] if i < len(a) else "") or "") for k in skip)]


# (a) the in-run report (no --also) against the log
i0 = next(i for i, l in enumerate(LL) if l.startswith("PER-SEED RAW RESULTS")) - 2
inrun_log = LL[i0:LL.index(wall_line) + 1]
regA = run_report(T, storeL, SEEDS_L, "L", wall_s, "delta_recipe_results.json", None)
regA = regA[:regA.index(next(l for l in regA if "total wall clock" in l)) + 1]
dA = linediff(regA, inrun_log, skip=("total wall clock",))
say(f"  (a) in-run report: log {len(inrun_log)} lines (from log line {i0 + 1}), regenerated {len(regA)}; lines that differ "
    f"(the wall-clock line aside): {len(dA)}")
for i, a, b in dA[:6]:
    say(f"     regen |{(a or '')[:140]}\n     log   |{(b or '')[:140]}")
# (b) the --report --also report against delta_recipe_L_report.log
regB = run_report(T, storeL, SEEDS_L, "L", 0.0, "delta_recipe_results.json", XRES)
while regB and regB[-1] == "":
    regB.pop()
lrep = list(Lrep)
while lrep and lrep[-1] == "":
    lrep.pop()
dB = linediff(regB, lrep)
say(f"  (b) --report --also: {LREP} {len(lrep)} lines, regenerated {len(regB)}; lines that differ: {len(dB)}")
for i, a, b in dB[:6]:
    say(f"     regen |{(a or '')[:140]}\n     log   |{(b or '')[:140]}")
pooled_blk = lambda rep: rep[rep.index(next(l for l in rep if l.startswith("POOLED (PRIMARY)"))):
                             rep.index(next(l for l in rep if l.startswith("PER MACHINE (SECONDARY)")))]
# (c) X's side with --also L
regC = run_report(T, storeX, SEEDS_X, "X", 0.0, "delta_recipe_results.json", LRES)
pbL, pbX = pooled_blk(regB), pooled_blk(regC)
dC = [(a, b) for a, b in zip(pbL, pbX) if a != b]
say(f"  (c) the pooled block from X's file with --also L's: {len(pbX)} lines vs L's {len(pbL)}; differing: {len(dC)}")
for a, b in dC:
    say(f"     L-side |{a[:150]}\n     X-side |{b[:150]}")
say("  -- the pooled block and L's per-machine block, as regenerated --")
for l in pooled_blk(regB) + regB[regB.index(next(l for l in regB if l.startswith("PER MACHINE"))):][:16]:
    say("   |" + l[:200])
tL, tX = T["tally"](storeL, SEEDS_L), T["tally"](storeX, SEEDS_X)

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("3. Independent recomputation (rlib outcomes; own statistics), per machine and pooled")
say("=" * 100)
ARMS = ["CEIL_HEBB", "CEIL_DELTA", "HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW"]
ns_nc = {"math": math}
extract(open(os.path.join(HERE, "spec_checks_delta2.py")).read(), {"wilson_z", "newcombe10"}, ns_nc)
newcombe10 = ns_nc["newcombe10"]


def first_fire(r):
    f = [c["step"] for c in r.get("checks") or [] if c["fired"]]
    return f[0] if f else None


def analyze(store, lo):
    S20 = list(range(lo, lo + 20))
    runs = {a: rlib.runs(store, a) for a in ARMS}
    assert all(sorted(runs[a]) == (S20[:2] if a.startswith("CEIL") else S20) for a in ARMS)
    assert all(r.get("ok") for a in ARMS for r in runs[a].values())
    out = dict(S=S20, runs=runs)
    out["trans_eq"] = all(rlib.transition(r["curve"]) == r["transition"] for a in ARMS for r in runs[a].values())
    br = {a: {s: (rlib.bound(r) if a.startswith("CEIL") else rlib.bound_routed(r)) for s, r in runs[a].items()} for a in ARMS}
    out["br"] = br
    out["agree"] = all(br[a][s] == T["success"](a, r) for a in ARMS for s, r in runs[a].items())
    out["cnt"] = {a: sum(br[a].values()) for a in ARMS}
    out["ceil_tr"] = {a: [runs[a][s]["transition"] for s in S20[:2]] for a in ("CEIL_HEBB", "CEIL_DELTA")}
    out["valid"] = {"HEBB": out["cnt"]["CEIL_HEBB"] == 2, "DELTA": out["cnt"]["CEIL_DELTA"] == 2}
    out["med"] = {a: statistics.median([runs[a][s]["transition"] for s in br[a] if br[a][s]]) for a in ARMS}
    out["rng"] = {a: (min(runs[a][s]["transition"] for s in br[a] if br[a][s]), max(runs[a][s]["transition"] for s in br[a] if br[a][s])) for a in ARMS}
    out["oc"] = {a: collections.Counter(rlib.outcome_k(runs[a][s]) for s in S20) for a in ARMS[2:]}
    out["fails"] = {a: {s: rlib.outcome_k(runs[a][s]) for s in S20 if not br[a][s]} for a in ARMS[2:]}
    out["ov_ok"] = {a: sum(1 for s, r in runs[a].items() if br[a][s] and T["overlap"](r) <= 0.1) for a in ARMS[2:]}
    out["ov_max"] = max(T["overlap"](r) for a in ARMS[2:] for s, r in runs[a].items() if br[a][s])

    def disc(x, y, seeds=S20):
        return ([s for s in seeds if br[x][s] and br[y][s]], [s for s in seeds if br[x][s] and not br[y][s]],
                [s for s in seeds if br[y][s] and not br[x][s]], [s for s in seeds if not br[x][s] and not br[y][s]])
    # CHECK 174 (own)
    broken = {}
    for x, y in (("DELTA_SPLIT", "DELTA_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW")):
        broken[x] = []
        for s in S20:
            f = first_fire(runs[x][s])
            cx = [p for p in runs[x][s]["curve"] if f is None or p[0] <= f]
            cy = [p for p in runs[y][s]["curve"] if f is None or p[0] <= f]
            if cx != cy:
                broken[x].append(s)
    out["broken"] = broken
    out["nofire"] = {x: [s for s in S20 if first_fire(runs[x][s]) is None] for x in ("DELTA_SPLIT", "HEBB_SPLIT")}
    out["D1"] = disc("DELTA_SPLIT", "HEBB_SPLIT")
    out["D2"] = disc("DELTA_SPLIT", "DELTA_SLOW", [s for s in S20 if s not in broken["DELTA_SPLIT"]])
    out["D4"] = disc("HEBB_SPLIT", "HEBB_SLOW", [s for s in S20 if s not in broken["HEBB_SPLIT"]])
    out["SS"] = disc("DELTA_SLOW", "HEBB_SLOW")
    out["d3"] = [(s, runs["HEBB_SPLIT"][s]["transition"], runs["DELTA_SPLIT"][s]["transition"]) for s in S20
                 if br["HEBB_SPLIT"][s] and br["DELTA_SPLIT"][s]]
    out["same_delta"] = [s for s in S20 if rlib.outcome_k(runs["DELTA_SPLIT"][s]) != rlib.outcome_k(runs["DELTA_SLOW"][s])]

    def pblind(arm):
        rows_, blind, alone = 0, 0, 0
        for s in S20:
            r = runs[arm][s]
            cur = {p[0]: p[1] for p in r["curve"]}
            n, last = 0, None
            for c in r["checks"]:
                if c["step"] >= 4800:
                    rows_ += 1
                    h = cur.get(c["step"])
                    if h is not None and c["acc"] >= 0.95 and h < 0.95:
                        blind += 1
                    if (c["prev"] is not None and c["acc"] - c["prev"] < 0.02 and n < 3 and (last is None or c["step"] - last >= 4800)
                            and c["acc"] >= 0.95 and not c["fired"]):
                        alone += 1
                if c["fired"]:
                    n, last = n + 1, c["step"]
        return rows_, blind, alone
    out["pb"] = {a: pblind(a) for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
    out["fire"] = {}
    for a in ("HEBB_SPLIT", "DELTA_SPLIT"):
        fs = [c for s in S20 for c in runs[a][s]["checks"] if c["fired"]]
        out["fire"][a] = (len(fs), sum(1 for c in fs if c["key_ok"]), sum(1 for s in S20 if any(c["fired"] for c in runs[a][s]["checks"])))
    return out


AL, AX = analyze(storeL, 1580), analyze(storeX, 580)
STAT = {}
for nm, A in (("L", AL), ("X", AX)):
    say(f"  -- machine {nm} --")
    say(f"   rlib transition = record's on all 84: {A['trans_eq']}; rlib outcome = the test's success() on all 84: {A['agree']}")
    say(f"   validity: CEIL_HEBB {A['cnt']['CEIL_HEBB']}/2 {A['ceil_tr']['CEIL_HEBB']}; CEIL_DELTA {A['cnt']['CEIL_DELTA']}/2 "
        f"{A['ceil_tr']['CEIL_DELTA']} -> {A['valid']}")
    for a in ARMS[2:]:
        say(f"   {a:<12} {A['cnt'][a]:>2}/20 median {A['med'][a]:.0f} [{A['rng'][a][0]}, {A['rng'][a][1]}]  {dict(A['oc'][a])}  "
            f"not BR: {A['fails'][a]}")
    say(f"   BR with overlap <= 0.1: {A['ov_ok']} (largest {A['ov_max']:.4f}); DELTA_SPLIT vs DELTA_SLOW different outcome on "
        f"{A['same_delta']}")
    a1, b1, c1, d1 = A["D1"]
    nc = newcombe10(len(a1), len(b1), len(c1), len(d1))
    st = dict(D1=(len(a1), len(b1), len(c1), len(d1), len(c1) - len(b1), nc, b1, c1))
    say(f"   D1: both {len(a1)}, DELTA only {len(b1)} {b1}, HEBB only {len(c1)} {c1}, neither {len(d1)}; d = {len(c1) - len(b1)}; "
        f"newcombe10 {nc[0]:+.4f} [{nc[1]:+.4f}, {nc[2]:+.4f}]")
    for name in ("D2", "D4"):
        a, b, c, _ = A[name]
        st[name] = (len(b), len(c), rlib.mcnemar1(len(b), len(c)), b, c)
        say(f"   {name}: split only {len(b)} {b}, no-trigger only {len(c)} {c}; p = {st[name][2]:.6g} -> "
            f"{'SHOWN' if st[name][2] < 0.05 else 'NOT SHOWN'}")
    pos = sum(1 for _, h, d in A["d3"] if d < h)
    neg = sum(1 for _, h, d in A["d3"] if d > h)
    st["D3"] = (len(A["d3"]), pos, neg, len(A["d3"]) - pos - neg, rlib.binom_tail(pos + neg, pos),
                statistics.median([d for *_, d in A["d3"]]), statistics.median([h for _, h, _ in A["d3"]]))
    say(f"   D3: {st['D3'][0]} shared, earlier {pos}, later {neg}, tied {st['D3'][3]}; p = {st['D3'][4]:.6g}; medians DELTA "
        f"{st['D3'][5]} HEBB {st['D3'][6]}")
    for a in ARMS[2:]:
        lo, hi = rlib.wilson(A["cnt"][a], 20)
        say(f"   band {a:<12} {A['cnt'][a]}/20 {rlib.band(A['cnt'][a], 20):<9} [{lo:.3f}, {hi:.3f}]")
    say(f"   CHECK 174 (own): broken {A['broken']}; never fired {A['nofire']}")
    say(f"   probe-blind (own; checks >= 4800): " + "; ".join(f"{k} {v[1]} of {v[0]}, alone {v[2]}" for k, v in A["pb"].items()))
    say(f"   firings (own): " + "; ".join(f"{k} {v[0]} firings, {v[1]} ok, fired in {v[2]} runs" for k, v in A["fire"].items()))
    STAT[nm] = st
# pooled
cntP = {a: AL["cnt"][a] + AX["cnt"][a] for a in ARMS[2:]}
validP = {p: AL["valid"][p] and AX["valid"][p] for p in ("HEBB", "DELTA")}
P = {}
for n, (ia, ib, ic, idd) in (("D1", (0, 1, 2, 3)),):
    a, b, c, d = (len(AL[n][k]) + len(AX[n][k]) for k in (ia, ib, ic, idd))
    P["D1"] = (a, b, c, d, c - b, newcombe10(a, b, c, d))
for n in ("D2", "D4"):
    b = len(AL[n][1]) + len(AX[n][1])
    c = len(AL[n][2]) + len(AX[n][2])
    kept = len(AL[n][0]) + len(AL[n][1]) + len(AL[n][2]) + len(AL[n][3]) + len(AX[n][0]) + len(AX[n][1]) + len(AX[n][2]) + len(AX[n][3])
    slow = "DELTA_SLOW" if n == "D2" else "HEBB_SLOW"
    slow_kept = sum(1 for A in (AL, AX) for s in A["S"] if s not in A["broken"]["DELTA_SPLIT" if n == "D2" else "HEBB_SPLIT"] and A["br"][slow][s])
    P[n] = (b, c, rlib.mcnemar1(b, c), kept, slow_kept)
d3P = AL["d3"] + AX["d3"]
posP = sum(1 for _, h, d in d3P if d < h)
negP = sum(1 for _, h, d in d3P if d > h)
P["D3"] = (len(d3P), posP, negP, len(d3P) - posP - negP, rlib.binom_tail(posP + negP, posP),
           statistics.median([d for *_, d in d3P]), statistics.median([h for _, h, _ in d3P]))
ncP = P["D1"][5]
nc_t = T["newcombe_paired"](*P["D1"][:4])
say(f"  -- pooled (X + L) --")
say(f"   validity: {validP}; counts {cntP}")
say(f"   D1: both {P['D1'][0]}, DELTA only {P['D1'][1]}, HEBB only {P['D1'][2]}, neither {P['D1'][3]}; d = {P['D1'][4]}; "
    f"newcombe10 {ncP[0]:+.4f} [{ncP[1]:+.4f}, {ncP[2]:+.4f}]; test's newcombe_paired {nc_t[0]:+.4f} [{nc_t[1]:+.4f}, "
    f"{nc_t[2]:+.4f}] (largest difference {max(abs(x - y) for x, y in zip(ncP, nc_t)):.1e}); one-sided McNemar of "
    f"{P['D1'][1]} vs {P['D1'][2]} (printed nowhere): p = {rlib.mcnemar1(P['D1'][1], P['D1'][2]):.4f}")


def d1_label_spec(d):
    """Amendment 1's three labels with Amendment 2 (a)'s first: d = HEBB only - DELTA only."""
    if d <= -3:
        return "delta not shown worse; ahead by >= 3 pairs (untested)"
    if d <= 2:
        return "delta not shown worse by more than 2 pairs"
    return "delta not shown non-inferior"


def split_label_spec(p, slow_kept):
    return "no room" if slow_kept >= 36 else ("helps" if p < 0.05 else "not shown")


labD1 = d1_label_spec(P["D1"][4])
labD = split_label_spec(P["D2"][2], P["D2"][4])
labH = split_label_spec(P["D4"][2], P["D4"][4])
NAMED = {("helps", "helps"): "the split helps on both writes",
         ("not shown", "helps"): "the split is shown on Hebbian, not on delta",
         ("helps", "not shown"): "the split helps on delta",
         ("not shown", "not shown"): "the split's effect is not shown on either write"}
readP = NAMED.get((labD, labH)) if "no room" not in (labD, labH) else f"delta: {labD}; Hebbian: {labH}"
for n in ("D2", "D4"):
    say(f"   {n}: {P[n][0]} vs {P[n][1]}, p = {P[n][2]:.6g} -> {'SHOWN' if P[n][2] < 0.05 else 'NOT SHOWN'}; pairs kept by CHECK 174 "
        f"{P[n][3]}; no-trigger arm on them {P[n][4]}/40 -> {'NO ROOM' if P[n][4] >= 36 else 'room'}")
say(f"   D3: {P['D3'][0]} shared, earlier {P['D3'][1]}, later {P['D3'][2]}, tied {P['D3'][3]}; p = {P['D3'][4]:.6g}; medians "
    f"DELTA {P['D3'][5]} HEBB {P['D3'][6]}")
for a in ARMS[2:]:
    lo, hi = rlib.wilson(cntP[a], 40)
    say(f"   band {a:<12} {cntP[a]}/40 {rlib.band(cntP[a], 40):<9} [{lo:.3f}, {hi:.3f}]")
say(f"   labels from the spec's words: D1 '{labD1}'; delta '{labD}', Hebbian '{labH}'; reading '{readP}'")
say(f"   the code's: D1 '{T['d1_label'](T['pool2']([tX['D1'], tL['D1']]))}'")
assert (P["D1"][1], P["D1"][2]) == (5, 5) and (P["D2"][0], P["D2"][1]) == (0, 3) and (P["D4"][0], P["D4"][1]) == (19, 1)

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say(f"4. X's run log ({XLOG}, committed at {XLOG_C}) against the referee's X check (rev9/check_delta_recipe_X.md)")
say("=" * 100)
XL = src(XLOG_C, XLOG).split("\n")
xh = [l for l in XL if l.startswith("=== longrun")]
xr = [l for l in XL if l.startswith("=== REPORT REGENERATED")]
say(f"  longrun headers: {xh}; regenerated-report headers: {[x[:150] for x in xr]}")
xs = [l.strip() for l in XL if l.strip().startswith("torch ")]
say(f"  {xs}")
XSTART = dt.datetime.strptime(re.search(r"start 1 at (\S+ [\d:]+)", xh[0]).group(1), "%Y-%m-%d %H:%M:%S")
XSTARTED = dt.datetime.strptime(metaX["started"], "%Y-%m-%d %H:%M:%S")
xrun = [run_re.match(l).groups() for l in XL if run_re.match(l)]
xfin = {(k, int(s)): XSTARTED + dt.timedelta(minutes=float(el)) for k, s, _, _, el in xrun}
xmism = [(k, s) for k, s, line, secs, el in xrun if int(secs) != round(RX[f"{k}|{s}"]["secs"])]
T55, T22, TXREC = utc(A2SPEC), utc(A2CODE), utc(REC_X)
fin55 = sorted(ks for ks, f in xfin.items() if f + dt.timedelta(seconds=3) <= T55)
amb55 = sorted(ks for ks, f in xfin.items() if abs((f - T55).total_seconds()) <= 3)
fin22 = sorted(ks for ks, f in xfin.items() if f + dt.timedelta(seconds=3) <= T22)
AMEND = sorted([("CEIL_HEBB", 580), ("CEIL_HEBB", 581), ("CEIL_DELTA", 580), ("CEIL_DELTA", 581), ("DELTA_SLOW", 580)]
               + [("DELTA_SPLIT", s) for s in range(580, 599)])
x_wall = float(re.search(r"verification: ([\d.]+) min", next(l for l in XL if "total wall clock" in l)).group(1))
XEND = XSTARTED + dt.timedelta(minutes=x_wall)
say(f"  per-run lines {len(xrun)}; secs disagreeing with the record: {xmism or 'none'}; longrun start {XSTART}; "
    f"meta.started {XSTARTED}; end {XEND} ({(XEND - XSTART).total_seconds() / 3600:.2f} h after the start); result commit {TXREC}")
say(f"  finished (to +-3 s) by 55dac50 {T55}: {len(fin55)} runs, = the amendment's 24: {fin55 == AMEND}; within 3 s of it: "
    f"{amb55 or 'none'}; by 2225f00 {T22}: {len(fin22)}, = the 24: {fin22 == AMEND}")
f599 = xfin[("DELTA_SPLIT", 599)]
say(f"  DELTA_SPLIT|599 finished {f599:%H:%M:%S} ({(f599 - T55).total_seconds():+.0f} s from 55dac50, "
    f"{(f599 - T22).total_seconds():+.0f} s from 2225f00); the X check's window 15:36:04 - 15:41:49, Monte Carlo median +163 s")
hebb_first = min(f for (k, _), f in xfin.items() if k.startswith("HEBB"))
say(f"  first Hebbian learned-gate finish {hebb_first} (X check: at or after 2026-10-11 23:55); last {XEND} (X check: at or "
    f"after 07:20:06; slack to the result commit <= 345 s: actual {(TXREC - XEND).total_seconds():.0f} s)")
# the X check's lower bounds (its scheduler, no overhead)
msx = metaX["ms_per_step"]
CEILS_ = {"CEIL_HEBB", "CEIL_DELTA"}
order = sorted(metaX["seeds"], key=lambda k: (k not in CEILS_, -msx[k]))
specsx = [(k, s) for k in order for s in metaX["seeds"][k]]
PX = sum(msx[k] * 1200 / 1000 for k in msx)


def simulate(P_, specs, R_, workers, started):
    clock, inflight, pending, out = 0.0, [], list(specs), {}
    while pending or inflight:
        while pending and len(inflight) < workers:
            k, s = pending.pop(0)
            inflight.append((clock + R_[f"{k}|{s}"]["secs"], k, s, clock))
        inflight.sort()
        f, k, s, st_ = inflight.pop(0)
        clock = f
        out[(k, s)] = started + dt.timedelta(seconds=P_ + f)
    return out


lbx = simulate(PX, specsx, RX, metaX["workers"], XSTARTED)
viol = [(k, s, f"{(xfin[(k, s)] - lbx[(k, s)]).total_seconds():+.0f}") for (k, s) in lbx if xfin[(k, s)] < lbx[(k, s)] - dt.timedelta(seconds=3)]
excess = sorted((xfin[ks] - lbx[ks]).total_seconds() for ks in lbx)
say(f"  the X check's lower bounds: actual finish earlier than the bound (beyond the 3 s print rounding): {viol or 'none'}; "
    f"actual minus bound: min {excess[0]:+.0f} s, median {excess[len(excess) // 2]:+.0f} s, max {excess[-1]:+.0f} s")
# X's two reports in the log against the regenerations
Trun = test_ns(RUNX)
xi = [i for i, l in enumerate(XL) if l.startswith("PER-SEED RAW RESULTS")]
xw = [i for i, l in enumerate(XL) if "total wall clock after verification" in l]
x_inrun, x_a2 = XL[xi[0] - 2:xw[0] + 1], XL[xi[1] - 2:xw[1] + 1]
r1 = run_report(Trun, storeX, SEEDS_X, "X", x_wall * 60, "delta_recipe_results.json", None)
r1 = r1[:r1.index(next(l for l in r1 if "total wall clock" in l)) + 1]
r2 = run_report(T, storeX, SEEDS_X, "X", 0.0, "delta_recipe_results.json", None)
r2 = r2[:r2.index(next(l for l in r2 if "total wall clock" in l)) + 1]
dx1 = linediff(r1, x_inrun, skip=("total wall clock",))
dx2 = linediff(r2, x_a2)
say(f"  in-run report (log lines {xi[0] - 1}-{xw[0] + 1}) vs {RUNX}'s report functions on the records: {len(x_inrun)} vs "
    f"{len(r1)} lines, differing {len(dx1)}")
say(f"  regenerated report (log lines {xi[1] - 1}-{xw[1] + 1}) vs {REC_X}'s: {len(x_a2)} vs {len(r2)} lines, differing {len(dx2)}")
for i, a, b in (dx1 + dx2)[:6]:
    say(f"     regen |{(a or '')[:140]}\n     log   |{(b or '')[:140]}")
cl = lambda rep, key: [l for l in rep if key in l]
x_pb_inrun = [l.strip()[:120] for l in cl(x_inrun, "probe-blind checks")]
x_pb_a2 = [l.strip()[:150] for l in cl(x_a2, "probe-blind checks")]
say(f"  in-run probe-blind lines: {x_pb_inrun}")
inrun_vs_a2 = [(a, b) for a, b in zip(x_inrun, x_a2) if a != b and "total wall clock" not in a]
say(f"  in-run vs regenerated, in the log itself: {len(inrun_vs_a2)} lines differ: "
    + " || ".join(a.strip()[:70] for a, _ in inrun_vs_a2))
xblk, cur = {}, None
for l in XL[:xi[0]]:
    m = re.match(r"CHECK (\d+)", l)
    if m:
        cur = m.group(1)
        xblk.setdefault(cur, [])
    if cur:
        xblk[cur].append(l)
xv = {c: " ".join(xblk.get(c, [])) for c in ("130", "158", "159", "161", "162", "173")}
x_ver = dict(c130=("DIFFERS" in xv["130"] and "printed, not asserted" in xv["130"]),
             c158=("printed, not asserted" in xv["158"] and "first differing evaluation 6000" in xv["158"]),
             c159=("IDENTICAL" in xv["159"] and "curve equal True" in xv["159"]),
             c162=("IDENTICAL" in xv["162"] and "curve equal True" in xv["162"]),
             c161=("LOCAL3 (k=16)        impl parallel: 1.19e-07" in xv["161"]),
             c173=(xv["173"].count("-> OK") == 2))
say(f"  RESULT (X)'s verification bullet against the log: {x_ver}; 'ALL VERIFICATION CHECKS PASSED' x"
    f"{sum(1 for l in XL if l.strip() == 'ALL VERIFICATION CHECKS PASSED')}")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("5. The statements")
say("=" * 100)
FLAT = " ".join(doc_added.split())
msg = " ".join(git("show", "-s", "--format=%B", REC_L).split())
readme = src(MAIN, "results/README.md")
plan = src(PLAN_C, PLAN)
plan_lines = plan.split("\n")


def has(text, frag, what):
    if " ".join(frag.split()) not in text:
        raise AssertionError(f"{what} lacks {frag!r}")
    return True


def R(frag):
    return has(FLAT, frag, "RESULT (L)/(POOLED)")


def M(frag):
    return has(msg, frag, f"commit {REC_L}")


def pl(frag):
    hits = [i + 1 for i, l in enumerate(plan_lines) if frag in l]
    if len(hits) != 1:
        raise AssertionError(f"plan fragment {frag!r}: {hits}")
    return hits[0]


cL, cX = AL["cnt"], AX["cnt"]
sL, sP = STAT["L"], P
f3 = lambda x: f"{x:+.3f}"
nl = (STAT["L"]["D1"][5])

# --- RESULT (L)
R("RESULT (L): test commit a581027, run at 8ae660d (a581027's test with Amendment 2's report rules, 55dac50 and 2225f00)")
row("RESULT (L)", "doc", "test commit a581027, run at 8ae660d (a581027's test with Amendment 2's report rules, 55dac50 and 2225f00)",
    f"meta.git {metaL['git']}; both log headers git {s1['git']}, {s2['git']}; " + CODE_NOTE, "MATCH",
    "results/README.md's commit column, defined as meta.git, gives a581027 (see the README row)")
R("on 12th Gen Intel(R) Core(TM) i7-12650H, torch 2.14.1, 1 thread per worker")
row("RESULT (L)", "doc", "on 12th Gen Intel(R) Core(TM) i7-12650H, torch 2.14.1, 1 thread per worker",
    f"meta.cpu and all 84 records' cpu {list(cpusL)}; meta.torch {metaL['torch']}; second start '{s2['torch_line'][:80]}'",
    "MATCH", f"The first start ran under '{s1['torch_line'][:28]}' with NumPy {s1['numpy']}, the second under torch 2.14.1 with "
    f"NumPy {s2['numpy']}: the laptop's Python environment changed between the starts (not stated). X's records are torch "
    f"{metaX['torch']}: the pool spans two torch builds (as split_copy's did)")
R("Two starts through longrun.py, both at the top of results/L/delta_recipe_L.log (L's times are local, UTC-6)")
row("RESULT (L)", "doc", "Two starts through longrun.py, both at the top of the log (L's times are local, UTC-6)",
    f"two longrun headers (log lines {s1['line']}, {s2['line']}), both reading 'start 1' (a fresh longrun each time), boot ids "
    f"{s1['boot']} -> {s2['boot']} (a reboot between); the json's start {loc} local = epoch {t_ep} UTC: UTC{OFF:+g}",
    "MATCH", f"The longrun json records only the second start (starts: 1, resumes {ljson['resumes']}); its offset "
    f"{ljson['offset']} is where the second start's output begins in the log (header at byte {off_byte})")
R("the first on 2026-10-12 at 10:04 on 6 workers passed every CHECK")
row("RESULT (L)", "doc", "the first on 2026-10-12 at 10:04 on 6 workers passed every CHECK",
    f"header {START1} local ({START1 + TZ} UTC), '6 worker processes'; 'ALL VERIFICATION CHECKS PASSED' x{s1['passed']} "
    f"(the second start: x{s2['passed']}); no failing line; its verification output equals the second's on all but "
    f"{len(vdiff)} lines (start header, torch line and path, NumPy version, worker count)", "MATCH")
R("and printed its projection")
row("RESULT (L)", "doc", "[the first start] ... and printed its projection",
    f"the first start printed the PROJECTION heading between its two rules (log lines {proj1 - 1}-{proj1 + 1}) "
    f"and no projection line: {s1['proj_lines']} 'ms/step' lines, no '84 runs: serial ...' total; the second start's header "
    f"follows at line {s2['line']}", "DISAGREE",
    "It was lost during the projection's pool-load timing (6 simultaneous copies of each arm's real path), not after it. "
    "Should read 'reached its projection' (or 'was lost during its projection')")
R("the laptop went down out of memory (a GPU job beside it) before any run was recorded")
row("RESULT (L)", "doc", "the laptop went down out of memory (a GPU job beside it) before any run was recorded",
    f"no RUNS heading and {s1['run_lines']} run lines in the first start; the results file is first written after the "
    f"projection, and the second start's CHECKs 144/157/170 found no cached record; boot id changed. The out-of-memory "
    f"cause and the GPU job are in no record", "NOT RECORD-CHECKABLE",
    "'before any run was recorded' is checkable and right (before any run was submitted). Session I's pre-run note "
    "(30807ec, claude/explore-I, 22:44 UTC 12 Oct) says its window_gate run 'moved here from the RTX 3070, which ran out "
    "of host memory running it beside test_delta_recipe'; the plan (6912e0b) gives L the RTX 3070 Laptop GPU: consistent")
R("the second at 13:14 on 5 workers found no cached record (CHECKs 144, 157 and 170: \"a first start\")")
row("RESULT (L)", "doc", "the second at 13:14 on 5 workers found no cached record (CHECKs 144, 157 and 170: 'a first start')",
    f"header {START2} local ({START2 + TZ} UTC), '5 worker processes', json cmd --workers 5, meta.workers "
    f"{metaL['workers']}; 'no cached record (a first start)' in CHECK 144/157/170: {first_cached}; no meta.starts", "MATCH")
R("runs from 15:10 after the verification")
row("RESULT (L)", "doc", "runs from 15:10 after the verification",
    f"meta.started {STARTED} local (stamped after verify, before the projection; elapsed counts from it); projection >= "
    f"{P_MIN:.0f} s; the first runs were submitted at about {first_sub:%H:%M} local ({first_sub + TZ:%H:%M} UTC)",
    "PARTIAL", "15:10 is the end of the verification, as RESULT (X)'s '05:27'; the runs began about 19 min later")
R("84/84, none failed, nothing cut")
row("RESULT (L)", "doc", "84/84, none failed, nothing cut",
    f"{len(RL)} records, all ok; {len(runl)} run lines, none '*** FAILED'; projection line 'nothing is cut'",
    "MATCH" if len(runl) == 84 and not any("FAILED" in x[2] for x in runl) else "DISAGREE")
R("1125.8 min of runs on five workers (the projection said 32.6 h)")
row("RESULT (L)", "doc", "1125.8 min of runs on five workers (the projection said 32.6 h)",
    f"the report's 'total wall clock after verification: {wall_min} min' = the last run line's elapsed; it includes the "
    f"projection (~{(first_sub - STARTED).total_seconds() / 60:.0f} min): runs alone ~{(END_L - first_sub).total_seconds() / 60:.0f} "
    f"min; projected_wall_h {metaL['projected_wall_h']:.2f}", "PARTIAL",
    f"1125.8 min is the wall clock after verification, not of runs. End {END_L} local = {END_L + TZ} UTC; the result commit "
    f"{(T_LREC - (END_L + TZ)).total_seconds() / 60:.1f} min later. From the second start: "
    f"{(END_L - START2).total_seconds() / 3600:.2f} h; from the first: {(END_L - START1).total_seconds() / 3600:.2f} h")
R("exit 0 (results/L/longrun/delta_recipe_L.exit; its json beside it)")
row("RESULT (L)", "doc", "exit 0 (the .exit file; its json beside it)", f".exit {lexit!r}; json total {ljson['total']}, "
    f"starts {len(st0)} ({st0[0]['started']}, git {st0[0]['git']}, records_at_start {st0[0]['records_at_start']})", "MATCH")
R("The run was started without --also, so its own report says \"POOLED (PRIMARY): not computed here\"")
row("RESULT (L)", "doc", "started without --also; its own report says 'POOLED (PRIMARY): not computed here'",
    f"json cmd {ljson['cmd'][2:]}; CHECK 168 'no --also file'; in-run line {inrun_pooled[0][:60]!r}", "MATCH")
R("the pooled block below is the report regenerated with --report --also results/X/delta_recipe_results.json from the same two results files (results/L/delta_recipe_L_report.log)")
row("RESULT (L)", "doc", "the pooled block is the report regenerated with --report --also X's file (delta_recipe_L_report.log)",
    f"report() at {REC_X} on L's records with --also {XRES}: {len(regB)} lines, {len(dB)} differ from {LREP}", "MATCH" if not dB else "DISAGREE")
R("which either machine's file produces identically")
row("RESULT (L)", "doc", "which either machine's file produces identically",
    f"from X's file with --also L's, the pooled block has {len(pbX)} lines; {len(dC)} differ: the header naming the two "
    f"machines in the other order" if len(dC) == 1 and dC[0][0].startswith("POOLED") else f"{len(dC)} lines differ",
    "MATCH" if len(dC) <= 1 and all(a.startswith("POOLED") for a, _ in dC) else "PARTIAL",
    "Every claim, label, band and reading line is identical; the header line differs by construction")
row("L log", "log", "(the in-run report, as the log prints it)", f"report() at {REC_X} on L's records without --also: "
    f"{len(regA)} lines; {len(dA)} differ from the log (lines {i0 + 1}-{LL.index(wall_line) + 1}), the wall-clock line aside "
    f"(equal too: {any(wall_line == x for x in run_report(T, storeL, SEEDS_L, 'L', wall_s, 'delta_recipe_results.json', None))} "
    f"at {wall_s} s)", "MATCH" if not dA else "DISAGREE",
    "Not a docstring statement: the check that the log's report is the records' report")
R("- Verification: every CHECK passed (ALL VERIFICATION CHECKS PASSED at every stage)")
row("RESULT (L)", "doc", "Verification: every CHECK passed (ALL VERIFICATION CHECKS PASSED at every stage)",
    f"second start: x{s2['passed']}, no failing line", "MATCH")
R("CHECK 130 (L's test_early_recipe file), 146 and 158 (L's test_window_gate records; 158 identical through the first split at 4800, different from 6000, the exact copy against the noisy one), 161 (LOCAL3 1.19e-7 under 1e-6), 173 (fused = parallel)")
v130 = "IDENTICAL" in c130 and "not asserted" not in c130
v146 = "asserted on a matching CPU" in c146 and "not asserted" not in c146
v158 = "first differing evaluation 6000" in c158 and "AS STATED" in c158 and "not asserted" not in c158
v161 = "LOCAL3 (k=16)        impl parallel: 1.19e-07 (tolerance 1e-06)" in c161
v173 = c173.count("-> OK") == 2
row("RESULT (L)", "doc", "asserted: CHECK 130, 146, 158 (identical through 4800, different from 6000), 161 (LOCAL3 1.19e-7 "
    "under 1e-6), 173", f"130 {v130}; 146 {v146}; 158 {v158}; 161 {v161} (LOCAL3 fast 1.34e-07 also under 1e-6); 173 {v173}",
    "MATCH" if all((v130, v146, v158, v161, v173)) else "DISAGREE")
R("the inherited CHECKs 112, 122, 123, 135 and 141, and this test's 150 (F's S68_SINGLE_S8|300), 159 (H's COPY_NONOISE|487) and 162 (F's DELTA_SPLIT|1310)")
row("RESULT (L)", "doc", "printed, not asserted, and unequal: inherited 112, 122, 123, 135, 141; own 150, 159, 162",
    f"CHECK blocks of the second start holding 'not asserted' and an unequal comparison: {na_uneq}",
    "MATCH" if na_uneq == [112, 122, 123, 135, 141, 150, 159, 162] else "DISAGREE",
    "122 and 123 are IDENTICAL against the screens' code on this machine (asserted) and unequal only against the "
    "2.10GHz host's recorded runs (printed)")
R("- Validity: CEIL_HEBB bound 2/2 (3600, 3600); CEIL_DELTA 2/2 (1200, 8400). Both parts VALID on L.")
row("RESULT (L)", "doc", "Validity: CEIL_HEBB 2/2 (3600, 3600); CEIL_DELTA 2/2 (1200, 8400); both VALID",
    f"CEIL_HEBB {cL['CEIL_HEBB']}/2 {AL['ceil_tr']['CEIL_HEBB']}; CEIL_DELTA {cL['CEIL_DELTA']}/2 {AL['ceil_tr']['CEIL_DELTA']}; "
    f"code {tL['valid']}", "MATCH")
for a, frag in (("HEBB_SPLIT", "HEBB_SPLIT 19/20 (median transition 20400 [10800, 33600]; s1594 MERGED)"),
                ("HEBB_SLOW", "HEBB_SLOW 10/20 (MERGED 2 share x7, 3 share x3)"),
                ("DELTA_SPLIT", "DELTA_SPLIT 16/20 (12600 [7200, 18000]; s1581, s1591, s1595 MERGED; s1582 BOUND NOT routed)"),
                ("DELTA_SLOW", "DELTA_SLOW 19/20 (14400 [7200, 43200]; s1595 MERGED)")):
    R(frag)
    row("RESULT (L)", "doc", frag, f"{a} {cL[a]}/20, median {AL['med'][a]:.0f} [{AL['rng'][a][0]}, {AL['rng'][a][1]}]; "
        f"not BR {AL['fails'][a]}", "MATCH")
R("every BOUND ROUTED run has overlap <= 0.1")
row("RESULT (L)", "doc", "every BOUND ROUTED run has overlap <= 0.1", f"{AL['ov_ok']} = the counts; largest {AL['ov_max']:.4f}",
    "MATCH" if all(AL["ov_ok"][a] == cL[a] for a in ARMS[2:]) else "DISAGREE")
R("D1: DELTA_SPLIT only 1, HEBB_SPLIT only 4, d = +3 (the mirror of X's -3); rate difference -0.150, 95% [-0.374, +0.082]")
row("RESULT (L)", "doc", "D1: DELTA only 1, HEBB only 4, d = +3 (the mirror of X's -3); -0.150, 95% [-0.374, +0.082]",
    f"DELTA only {sL['D1'][1]} {sL['D1'][6]}, HEBB only {sL['D1'][2]} {sL['D1'][7]}, both {sL['D1'][0]}, neither "
    f"{sL['D1'][3]}; d = {sL['D1'][4]:+d}; newcombe10 {f3(nl[0])} [{f3(nl[1])}, {f3(nl[2])}]; X d = {STAT['X']['D1'][4]}",
    "MATCH", "Per machine, no label (as specified); a d = +3 would be 'delta not shown non-inferior' if it were pooled")
R("D2: 0 vs 3, p = 1, NOT SHOWN (DELTA_SLOW 19/20)")
row("RESULT (L)", "doc", "D2: 0 vs 3, p = 1, NOT SHOWN (DELTA_SLOW 19/20)",
    f"{sL['D2'][0]} vs {sL['D2'][1]} (DELTA_SLOW only {sL['D2'][4]}), p = {sL['D2'][2]:g}; DELTA_SLOW {cL['DELTA_SLOW']}/20", "MATCH")
R("D4: 9 vs 0, p = 0.0020, SHOWN")
row("RESULT (L)", "doc", "D4: 9 vs 0, p = 0.0020, SHOWN",
    f"{sL['D4'][0]} vs {sL['D4'][1]} (HEBB_SPLIT only {sL['D4'][3]}), p = {sL['D4'][2]:.6f} = 1/512", "MATCH")
R("D3: on 15 shared binds, delta earlier on 14, later on 1, tied 0; sign test p = 0.00049; medians 12000 vs 22800")
row("RESULT (L)", "doc", "D3: 15 shared binds, earlier 14, later 1, tied 0; p = 0.00049; medians 12000 vs 22800",
    f"{sL['D3'][0]}, {sL['D3'][1]}, {sL['D3'][2]}, {sL['D3'][3]}; p = {sL['D3'][4]:.6f} = 16/32768; medians {sL['D3'][5]} vs "
    f"{sL['D3'][6]}", "MATCH", "the one later seed is s1589 (10800 vs 12000)")
R("Bands: HEBB_SPLIT RELIABLE [0.764, 0.991]; HEBB_SLOW MAJORITY [0.299, 0.701]; DELTA_SPLIT MAJORITY [0.584, 0.919]; DELTA_SLOW RELIABLE [0.764, 0.991].")
row("RESULT (L)", "doc", "Bands: HEBB_SPLIT RELIABLE [0.764, 0.991]; HEBB_SLOW MAJORITY [0.299, 0.701]; DELTA_SPLIT MAJORITY "
    "[0.584, 0.919]; DELTA_SLOW RELIABLE [0.764, 0.991]", "; ".join(
        f"{a} {rlib.band(cL[a], 20)} [{rlib.wilson(cL[a], 20)[0]:.3f}, {rlib.wilson(cL[a], 20)[1]:.3f}]" for a in ARMS[2:]), "MATCH")
R("Firings: Hebbian 41 in 20 runs, every target labelled ok (41/41); delta 32 in 20 runs, 16 labelled ok.")
row("RESULT (L)", "doc", "Firings: Hebbian 41 in 20 runs, 41/41 ok; delta 32 in 20 runs, 16 ok",
    f"HEBB_SPLIT {AL['fire']['HEBB_SPLIT'][0]} firings, {AL['fire']['HEBB_SPLIT'][1]} ok, fired in {AL['fire']['HEBB_SPLIT'][2]} "
    f"runs; DELTA_SPLIT {AL['fire']['DELTA_SPLIT'][0]}, {AL['fire']['DELTA_SPLIT'][1]} ok, fired in {AL['fire']['DELTA_SPLIT'][2]} runs",
    "MATCH", "'in 20 runs' is the runs counted, as the code prints it; DELTA_SPLIT never fired on "
    f"{['s%d' % s for s in AL['nofire']['DELTA_SPLIT']]}")
R("Probe-blind (checks from 4800): HEBB_SPLIT 3 of 170, blocked by the probe alone 2; DELTA_SPLIT 6 of 149, alone 9.")
pbc = {k: T["probe_blind"](tL, k) for k in ("HEBB_SPLIT", "DELTA_SPLIT")}
row("RESULT (L)", "doc", "Probe-blind: HEBB_SPLIT 3 of 170, alone 2; DELTA_SPLIT 6 of 149, alone 9",
    "; ".join(f"{k} {v[1]} of {v[0]}, alone {v[2]} (code: {pbc[k][1]} of {len(pbc[k][0])}, {pbc[k][2]})" for k, v in AL["pb"].items()),
    "MATCH" if all((AL["pb"][k][1], AL["pb"][k][0], AL["pb"][k][2]) == (pbc[k][1], len(pbc[k][0]), pbc[k][2]) for k in pbc)
    and (AL["pb"]["HEBB_SPLIT"], AL["pb"]["DELTA_SPLIT"]) == ((170, 3, 2), (149, 6, 9)) else "DISAGREE")
n174L, bad174L = T["check_174"](tL)
c171 = next(l for l in regA if "CHECK 171" in l)
R("CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs, none dropped).")
row("RESULT (L)", "doc", "CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs, none dropped)",
    f"code: '{c171.strip()[:95]}'; CHECK 174 {n174L} pairs, broken {bad174L or 'none'} (own: {AL['broken']}); dropped D2 "
    f"{tL['D2']['dropped']}, D4 {tL['D4']['dropped']}", "MATCH")

# --- RESULT (POOLED, PRIMARY)
R("RESULT (POOLED, PRIMARY) - X (results/X/delta_recipe_results.json) + L, b and c summed over both machines' pairs, from the regenerated report (results/L/delta_recipe_L_report.log); X and L VALID on both parts.")
row("POOLED", "doc", "X + L, b and c summed over both machines' pairs; X and L VALID on both parts",
    f"X {AX['valid']}, L {AL['valid']}; the code's pooled D1/D2/D4 = the sums of the per-machine b, c", "MATCH")
R("D1 DELTA_SPLIT vs HEBB_SPLIT: 35/40 vs 35/40; DELTA_SPLIT only 5, HEBB_SPLIT only 5; d = 0; rate difference +0.000, 95% [-0.162, +0.162].")
row("POOLED", "doc", "D1: 35/40 vs 35/40; DELTA only 5, HEBB only 5; d = 0; +0.000, 95% [-0.162, +0.162]",
    f"{cntP['DELTA_SPLIT']}/40 vs {cntP['HEBB_SPLIT']}/40; both {P['D1'][0]}, DELTA only {P['D1'][1]} (X 4, L 1), HEBB only "
    f"{P['D1'][2]} (X 1, L 4), neither {P['D1'][3]}; d = {P['D1'][4]}; newcombe10 {f3(ncP[0])} [{f3(ncP[1])}, {f3(ncP[2])}] "
    f"(test's function: same to {max(abs(x - y) for x, y in zip(ncP, nc_t)):.0e})", "MATCH")
R("Label: \"delta not shown worse by more than 2 pairs\"")
R("at 40 pairs with HEBB_SPLIT at 0.95, P(d <= 2) is 0.29 / 0.22 when DELTA_SPLIT is 0.85 and 0.11 / 0.05 when it is 0.80")
row("POOLED", "doc", "D1 label 'delta not shown worse by more than 2 pairs'; OC: P(d <= 2) 0.29 / 0.22 at 0.85, 0.11 / 0.05 at 0.80",
    f"d = 0 is in -2..2: '{labD1}' (spec's words; the code's d1_label the same); D1_OC as printed: "
    f"{'0.29 (independent outcomes) / 0.22 (nested) when DELTA_SPLIT is 0.85, and 0.11 / 0.05 when it is 0.80' in T['D1_OC']}",
    "MATCH", "A bound, not a non-inferiority test, as the label's printed note says; the interval [-0.162, +0.162] "
    "covers a 16-point deficit")
R("D2 DELTA_SPLIT beats DELTA_SLOW: 35/40 vs 38/40; 0 vs 3; p = 1; NOT SHOWN; DELTA_SLOW 38/40 >= 36: NO ROOM (printed, not read).")
row("POOLED", "doc", "D2: 35/40 vs 38/40; 0 vs 3; p = 1; NOT SHOWN; DELTA_SLOW 38/40 >= 36: NO ROOM",
    f"{cntP['DELTA_SPLIT']}/40 vs {cntP['DELTA_SLOW']}/40; {P['D2'][0]} vs {P['D2'][1]} (all 3 on L: s1581, s1591, s1582); "
    f"p = {P['D2'][2]:g}; pairs kept by CHECK 174 {P['D2'][3]}, DELTA_SLOW on them {P['D2'][4]} >= 36", "MATCH")
R("D4 HEBB_SPLIT beats HEBB_SLOW: 35/40 vs 17/40; 19 vs 1; p = 2.0e-5; SHOWN.")
row("POOLED", "doc", "D4: 35/40 vs 17/40; 19 vs 1; p = 2.0e-5; SHOWN",
    f"{cntP['HEBB_SPLIT']}/40 vs {cntP['HEBB_SLOW']}/40; {P['D4'][0]} vs {P['D4'][1]} (X 10 vs 1, L 9 vs 0); p = "
    f"{P['D4'][2]:.4e} = 21/2^20; HEBB_SLOW {P['D4'][4]}/40 < 36: room", "MATCH")
R("D3 (secondary) on the 30 seeds both split arms are BOUND ROUTED: medians DELTA_SPLIT 13200, HEBB_SPLIT 18600; delta earlier on 24, later on 5, tied 1; one-sided sign test p = 0.00027.")
row("POOLED", "doc", "D3: 30 shared; medians 13200 vs 18600; earlier 24, later 5, tied 1; p = 0.00027",
    f"{P['D3'][0]} shared (X 15, L 15); earlier {P['D3'][1]}, later {P['D3'][2]}, tied {P['D3'][3]}; p = {P['D3'][4]:.6f} "
    f"(= {sum(math.comb(29, k) for k in range(24, 30))}/2^29, P(Bin(29, 1/2) >= 24)); medians {P['D3'][5]} vs {P['D3'][6]}", "MATCH",
    "the report prints the medians as '13200.0' and '18600.0' (the median of an even count)")
R("Bands (pooled): HEBB_SPLIT MAJORITY [0.739, 0.945]; HEBB_SLOW MINORITY [0.285, 0.578]; DELTA_SPLIT MAJORITY [0.739, 0.945]; DELTA_SLOW RELIABLE [0.835, 0.986].")
row("POOLED", "doc", "Bands: HEBB_SPLIT MAJORITY [0.739, 0.945]; HEBB_SLOW MINORITY [0.285, 0.578]; DELTA_SPLIT MAJORITY "
    "[0.739, 0.945]; DELTA_SLOW RELIABLE [0.835, 0.986]", "; ".join(
        f"{a} {cntP[a]}/40 {rlib.band(cntP[a], 40)} [{rlib.wilson(cntP[a], 40)[0]:.3f}, {rlib.wilson(cntP[a], 40)[1]:.3f}]"
        for a in ARMS[2:]), "MATCH", "thresholds at n = 40: RELIABLE >= 36, MAJORITY >= 20")
R("READINGS (pooled; Amendments 1-2): the memory rule: \"delta not shown worse by more than 2 pairs\".")
R("The split: delta \"no room\" (DELTA_SLOW 38/40, b 0, c 3); Hebbian \"helps\" (HEBB_SLOW 17/40, b 19, c 1): printed as \"delta: no room; Hebbian: helps\", and by Amendment 2 (e) no combination is named when either write is \"no room\".")
row("POOLED", "doc", "READINGS: memory rule 'delta not shown worse by more than 2 pairs'; split: delta 'no room' (38/40, b 0, "
    "c 3), Hebbian 'helps' (17/40, b 19, c 1): 'delta: no room; Hebbian: helps'; no combination named (A2 (e))",
    f"from the spec's words: '{labD1}'; delta '{labD}', Hebbian '{labH}'; '{readP}'; the regenerated report: "
    f"{[l.strip()[:60] for l in regB if l.strip().startswith('the split:')]}", "MATCH")

# --- commit message b03d353
for frag, recomputed, status, note in (
        ("84/84 runs at 8ae660d, 2026-10-12 to 13", f"meta.git {metaL['git']}; runs {first_sub:%m-%d %H:%M} to {END_L:%m-%d %H:%M} "
         f"local ({first_sub + TZ:%m-%d %H:%M} to {END_L + TZ:%m-%d %H:%M} UTC)", "MATCH", ""),
        ("two starts, the first lost to the laptop's OOM before any run was recorded", "two headers, boot changed, no run line "
         "in the first; the OOM is in no record", "NOT RECORD-CHECKABLE", "as the RESULT row on the OOM"),
        ("D1 35/40 vs 35/40, d = 0, \"delta not shown worse by more than 2 pairs\"", f"{cntP['DELTA_SPLIT']}/40 vs "
         f"{cntP['HEBB_SPLIT']}/40, d = {P['D1'][4]}, '{labD1}'", "MATCH", ""),
        ("D2 NOT SHOWN, NO ROOM (DELTA_SLOW 38/40)", f"p = {P['D2'][2]:g}; DELTA_SLOW {P['D2'][4]}/40", "MATCH", ""),
        ("D4 SHOWN 19 vs 1 (p = 2e-5)", f"{P['D4'][0]} vs {P['D4'][1]}, p = {P['D4'][2]:.2e}", "MATCH", ""),
        ("D3 delta earlier 24 of 29", f"earlier on {P['D3'][1]} of {P['D3'][0]} shared binds ({P['D3'][1] + P['D3'][2]} untied)",
         "PARTIAL", "29 is the untied count; the RESULT says 'on the 30 seeds ... earlier on 24, later on 5, tied 1' (as the "
         "8ae660d message's '10 of 14' against 10 of 15)"),
        ("the split reading \"delta: no room; Hebbian: helps\"", f"'{readP}'", "MATCH", ""),
        ("the report regenerated with --also", f"{LREP} = report() with --also X's file: {len(dB)} lines differ", "MATCH", "")):
    M(frag)
    row(f"msg {REC_L}", "msg", frag, recomputed, status, note)

# --- results/README.md
rl = next(l for l in readme.split("\n") if l.startswith("| L/delta_recipe_results.json"))
has(readme, "Commit = the git head each run was started from (its `meta.git`); date = its `meta.started`.", "README")
row("README", "results/README.md (L row)", rl.strip(),
    f"meta.git {metaL['git']}; meta.started {metaL['started']}", "PARTIAL",
    f"The table's own rule: 'Commit = the git head each run was started from (its meta.git)'. L's meta.git is 8ae660d, the "
    f"column gives a581027 with '(run at 8ae660d ...)'. X's row (a581027) is right by the rule. The date 2026-10-12 15:10 is "
    f"meta.started (local, UTC-6)")
has(readme, "L's are the laptop's local time (UTC−6)", "README")
row("README", "results/README.md (note)", "L's [times] are the laptop's local time (UTC-6); L/delta_recipe_L_report.log is "
    "the report regenerated with --report --also results/X/delta_recipe_results.json",
    f"longrun json epoch: UTC{OFF:+g}; {LREP} = the regeneration ({len(dB)} lines differ)", "MATCH")

# --- the plan
L12 = pl("| `test_delta_recipe` | complete on both machines (L on 13 October, `b03d353`)")
row("plan", f"plan:{L12}", "complete on both machines (L on 13 October, b03d353)", f"X 84/84 ({REC_X}); L 84/84, last run "
    f"{END_L + TZ:%Y-%m-%d %H:%M} UTC; {REC_L}", "MATCH")
pl("pooled: D1 DELTA_SPLIT 35/40 vs HEBB_SPLIT 35/40, d = 0, [-0.162, +0.162]: \"delta not shown worse by more than 2 pairs\"")
row("plan", f"plan:{L12}", "D1 35/40 vs 35/40, d = 0, [-0.162, +0.162]: 'delta not shown worse by more than 2 pairs'",
    f"{cntP['DELTA_SPLIT']}/40 vs {cntP['HEBB_SPLIT']}/40, d = {P['D1'][4]}, [{f3(ncP[1])}, {f3(ncP[2])}], '{labD1}'", "MATCH")
pl("D2 35/40 vs DELTA_SLOW 38/40, 0 vs 3: NOT SHOWN, NO ROOM")
row("plan", f"plan:{L12}", "D2 35/40 vs 38/40, 0 vs 3: NOT SHOWN, NO ROOM", f"{P['D2'][0]} vs {P['D2'][1]}, p = 1; "
    f"DELTA_SLOW {P['D2'][4]}/40 >= 36", "MATCH")
pl("D4 HEBB_SPLIT 35/40 vs HEBB_SLOW 17/40, 19 vs 1, p = 2e-5: SHOWN")
row("plan", f"plan:{L12}", "D4 35/40 vs 17/40, 19 vs 1, p = 2e-5: SHOWN", f"{P['D4'][0]} vs {P['D4'][1]}, p = {P['D4'][2]:.2e}",
    "MATCH")
pl("D3 delta earlier on 24 of 29")
row("plan", f"plan:{L12}", "D3 delta earlier on 24 of 29", f"{P['D3'][1]} of {P['D3'][0]} shared ({P['D3'][1] + P['D3'][2]} "
    f"untied, 1 tie); p = {P['D3'][4]:.5f}", "PARTIAL", "as the b03d353 message; the RESULT says 24 of 30 with 1 tie")
pl("readings \"delta: no room; Hebbian: helps\"")
row("plan", f"plan:{L12}", "readings 'delta: no room; Hebbian: helps'", f"'{readP}'", "MATCH")
pl("X and L are mirror images on D1 (d = -3 and +3)")
row("plan", f"plan:{L12}", "X and L are mirror images on D1 (d = -3 and +3)",
    f"X: DELTA only {STAT['X']['D1'][1]}, HEBB only {STAT['X']['D1'][2]} (d = {STAT['X']['D1'][4]}), DELTA_SPLIT "
    f"{cX['DELTA_SPLIT']}/20, HEBB_SPLIT {cX['HEBB_SPLIT']}/20; L: {STAT['L']['D1'][1]}, {STAT['L']['D1'][2]} "
    f"(d = {STAT['L']['D1'][4]:+d}), {cL['DELTA_SPLIT']}/20, {cL['HEBB_SPLIT']}/20; intervals {f3(STAT['X']['D1'][5][0])} "
    f"[{f3(STAT['X']['D1'][5][1])}, {f3(STAT['X']['D1'][5][2])}] and {f3(nl[0])} [{f3(nl[1])}, {f3(nl[2])}]", "MATCH",
    "Mirror in counts and intervals; not in seeds or failure kinds (X's DELTA_SPLIT fails once, s594; L's four times, "
    "three MERGED and one BOUND NOT routed)")
L15 = pl("Open on the main line: X's `test_delta_recipe` run log is on no branch")
row("plan", f"plan:{L15}", "X's test_delta_recipe run log is on no branch (ask X to commit it)",
    f"true at {PLAN_C} ({utc(PLAN_C)} UTC); committed 8 min later at {XLOG_C} ({utc(XLOG_C)} UTC) as {XLOG}", "MATCH",
    "Stale since 7d7661d; plan step 2 (line 42) is done")
L35 = pl("idle; `test_delta_recipe` done (one start lost to an OOM when a GPU job ran beside it)")
row("plan", f"plan:{L35}", "L: test_delta_recipe done (one start lost to an OOM when a GPU job ran beside it)",
    "two starts in the log, the first without a run, boot id changed; the cause is in no record", "NOT RECORD-CHECKABLE",
    "consistent with session I's note (30807ec)")

# --- X's log against the referee's X check
row("X log", f"{XLOG}@{XLOG_C}", "X's run log committed (point 4)", f"commits touching {XLOG} on any ref: {xlog_refs}; "
    f"{XLOG_C} at {utc(XLOG_C)} UTC, on the main line", "MATCH", "rev9/check_delta_recipe_X*.md said no log on any ref: true "
    "then (12 Oct); superseded")
row("X log", "RESULT (X)", "The run started 2026-10-11 02:12 UTC through longrun.py (was PARTIAL: not recorded)",
    f"log header: {xh[0][:80]}", "MATCH", "the X check's PARTIAL is resolved: 02:12:34 in the log (X's clock is UTC)")
row("X log", "RESULT (X)", "29.2 h wall clock", f"{XSTART} -> last run {XEND} (meta.started + {x_wall} min) = "
    f"{(XEND - XSTART).total_seconds() / 3600:.2f} h", "MATCH", f"the X check's lower bound for the end (07:20:06) holds; "
    f"the end was {(TXREC - XEND).total_seconds():.0f} s before the result commit (it allowed <= 345 s)")
row("X log", "RESULT (X)", "Verification: CHECK 130 printed, not asserted; 158 identical through 4800, different from 6000, "
    "printed; 159, 162 bit for bit, asserted; 161 LOCAL3 1.19e-7; 173 fused = parallel (was NOT RECORD-CHECKABLE)",
    f"{x_ver}", "MATCH" if all(x_ver.values()) else "DISAGREE", "now checkable in the log")
row("X log", "Amendment 2 / X check §3", "24 runs finished at 55dac50 (15:36:08 UTC): the four perfect gates, DELTA_SPLIT "
    "580-598, DELTA_SLOW 580; no Hebbian learned-gate run",
    f"from the log's elapsed: {len(fin55)} runs finished by 15:36:08 (+-3 s print rounding), equal to the amendment's 24: "
    f"{fin55 == AMEND}; none within 3 s; at 2225f00 the same {len(fin22)}: {fin22 == AMEND}; DELTA_SPLIT|599 at "
    f"{f599:%H:%M:%S} ({(f599 - T55).total_seconds():+.0f} s); first Hebbian learned-gate finish {hebb_first:%m-%d %H:%M:%S}",
    "MATCH" if fin55 == AMEND and fin22 == AMEND else "DISAGREE",
    "The X check's reconstruction holds: 599 inside its window (15:36:04-15:41:49; Monte Carlo median +163 s, actual "
    f"{(f599 - T55).total_seconds():+.0f} s); every run's actual finish is at or after its lower bound "
    f"({'none' if not viol else viol} below; actual minus bound {excess[0]:+.0f} to {excess[-1]:+.0f} s)")
row("X log", "RESULT (X)", "the in-run report and the --report regeneration are both kept in the log; its claims, bands and "
    "checks equal the in-run report's line for line",
    f"in-run report (log) = {RUNX}'s report on the records: {len(dx1)} lines differ; regenerated (log, from line "
    f"{xi[1] - 1}) = {REC_X}'s: {len(dx2)} differ; in-run vs regenerated in the log: {len(inrun_vs_a2)} lines differ (the "
    f"claims header, the D1 interval's name, the two probe-blind lines; in-run {x_pb_inrun[0][44:62]!r}...)",
    "PARTIAL", "Both kept: MATCH. The X check's reconstruction of the in-run report is confirmed line for line. 'Line for "
    "line' holds for the claims and bands and CHECK 171/174, not for the probe-blind lines (A2 (f)) or the D1 interval's "
    "name (A2 (g)), as the X check said")

# --- the referee's earlier rows on L's status (rev9/check_delta_recipe_X_v6.md), now that L's log is in
D6, D9 = "42c4557", "010a085"
t6, t9 = utc(D6), utc(D9)
n6 = sum(1 for f in fin.values() if f + TZ <= t6)
n9 = sum(1 for f in fin.values() if f + TZ <= t9)
tex6 = src(D6, "multichannel_hebbian_report_v9.tex")
tex9 = src(D9, "multichannel_hebbian_report_v9.tex")
assert "L's half is next" in tex6 and "has not run" in tex6 and "(X; L running)" in tex6 and "L's half is running" in tex6
assert "L's half is not yet recorded" in tex9
row("referee v6", f"tex6 ({D6})", "check_delta_recipe_X_v6.md: L's status in the sixth draft NOT RECORD-CHECKABLE / PARTIAL "
    "('runs on the other', '(X; L running)', 'L's half is running' against 'L's half is next', 'has not run')",
    f"at {D6} ({t6} UTC = {t6 - TZ:%H:%M} local) L's second start had been running since {START2 + TZ:%H:%M} UTC and {n6} of "
    f"84 runs had finished (elapsed from {STARTED + TZ:%H:%M} UTC)", "PARTIAL",
    "Now checkable: 'running' was right; 'is next' and 'has not run' were not (L had begun and finished runs). The "
    f"referee's v6 status stands as the right call on the records it had. The merged Revision 9 ({D9}, {t9} UTC, {n9} runs "
    "finished on L) says 'L's half is not yet recorded', which is right")

# ════════════════════════════════════════════════════════════════════════════
say("")
say(f"{'scope':<12} {'status':<22} statement  ||  recomputed")
for sc, ln, st, rc, status, note in rows:
    say(f"{sc:<12} {status:<22} {st[:90]}  ||  {rc[:150]}")
cnt_ = collections.Counter(r[4] for r in rows)
by = collections.OrderedDict()
for r in rows:
    by.setdefault(r[0], collections.Counter())[r[4]] += 1
say("\ncounts by scope:")
for sc, c in by.items():
    say(f"  {sc:<14} {sum(c.values()):>3} rows: " + ", ".join(f"{k} {c[k]}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")))
say(f"counts over {len(rows)} rows: " + ", ".join(f"{k} {cnt_[k]}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")))
say("\n-- markdown table --")
say("| scope | statement | recomputed | status | note |")
say("|---|---|---|---|---|")
esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
for sc, ln, st, rc, status, note in rows:
    say(f"| {esc(sc)} | {esc(st)} | {esc(rc)} | {status if status == 'MATCH' else '**' + status + '**'} | {esc(note)} |")
