#!/usr/bin/env python3
"""check_delta_recipe_X.py: Referee V's check of test_delta_recipe on X (main line 8ae660d) and of the Revision 9
draft's lines that cite it (claude/rev9-draft at c5e4508, X half added in 902c39a).

    python3 docs/review_2026-10/rev9/check_delta_recipe_X.py      (stdlib only; prints; writes nothing)

Reads only git objects (git show REF:PATH); runs no training and does not run test_delta_recipe.py.
1. The test's own report: every report function of test_delta_recipe.py at 8ae660d (= 2225f00, compared here by AST
   segment), with its helpers extracted from their modules at 8ae660d (fail_class_v2, test_stream_recipe, ...), is
   exec'd and `report(store, SEEDS, "X", 0.0, ...)` is run on results/X/delta_recipe_results.json: this is the
   --report regeneration under Amendment 2. The same with a581027's report functions gives the report the running
   process printed at its end (pre-Amendment 2). The two are compared line by line.
2. The referee's own recomputation, independent of the test's functions: outcomes with rlib (transition from the
   curve, map from stream_gate), McNemar with rlib.mcnemar1, Newcombe with spec_checks_delta2.newcombe10 (extracted),
   Wilson with rlib.wilson, CHECK 174, D3, probe-blind and firings written here from the spec's words.
3. Amendment 2's timing: there is no log of X's run on any ref, so the finish time of every run is reconstructed from
   the records: meta.started (UTC, see the .md), the projection's measured cost (a lower bound on its duration), each
   run's own `secs`, and the test's scheduler (4 workers, perfect gates first, then longest first, refill on each
   finish). A Monte Carlo over the unrecorded overheads, constrained by the result commit, gives the sets of runs
   finished at 55dac50 (15:36:08 UTC) and 2225f00 (15:37:52 UTC).
4. The draft's statements (tex5:NNN = c5e4508's multichannel_hebbian_report_v9.tex), each fragment located verbatim.
"""
import ast
import collections
import contextlib
import datetime as dt
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REVIEW))
sys.path.insert(0, REVIEW)
import rlib  # noqa: E402

REC, A2CODE, RUNCODE, A2SPEC, DRAFT, DRAFT_X = "8ae660d", "2225f00", "a581027", "55dac50", "c5e4508", "902c39a"
RESULTS = "results/X/delta_recipe_results.json"
TEST = "test_delta_recipe.py"
TEX = "multichannel_hebbian_report_v9.tex"
rows = []          # (line, statement, recomputed, status, note)


def say(*a):
    print(" ".join(str(x) for x in a))


def git(*args):
    return subprocess.check_output(("git", "-C", ROOT) + args).decode()


def src(ref, path):
    return git("show", f"{ref}:{path}")


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


def row(line, statement, recomputed, status, note=""):
    rows.append((line, statement, recomputed, status, note))


# ════════════════════════════════════════════════════════════════════════════
say("=" * 100)
say(f"0. Inputs")
say("=" * 100)
store = json.loads(src(REC, RESULTS))
meta = store["meta"]
R = store["runs"]
say(f"  {RESULTS} at {REC}: {len(R)} records; meta.git {meta['git']}, cpu {meta['cpu']}, workers {meta['workers']}, "
    f"started {meta['started']!r} (no zone), torch {meta['torch']}, 'starts' (resumes) in meta: {'starts' in meta}")
say(f"  projected_wall_h {meta['projected_wall_h']:.2f}, projected_rest_h {meta['projected_rest_h']:.2f}")
logs = [l for l in git("log", "--all", "--format=", "--name-only").split("\n")
        if ("delta" in l and ("log" in l or "longrun" in l)) or l.startswith("results/X/longrun")]
say(f"  a log of X's delta_recipe run on any ref (any path with 'delta' and 'log'/'longrun', or results/X/longrun/*): "
    f"{sorted(set(logs)) or 'NONE'}")
xfiles = [l for l in git("ls-tree", "-r", "--name-only", REC, "results/X/").split("\n") if "delta" in l]
say(f"  results/X/*delta* at {REC}: {xfiles}")
for c in (RUNCODE, A2SPEC, A2CODE, REC, DRAFT_X, DRAFT):
    say(f"  {git('show', '-s', '--format=%h %cI %s', c).strip()[:150]}")
tsrc = {r: src(r, TEST) for r in (RUNCODE, A2CODE, REC)}
s_a2, s_rec, s_run = segments(tsrc[A2CODE]), segments(tsrc[REC]), segments(tsrc[RUNCODE])
diff_rec = [k for k in set(s_a2) | set(s_rec) if s_a2.get(k) != s_rec.get(k)]
say(f"  {TEST}: top-level defs/assignments that differ between {A2CODE} and {REC}: {diff_rec or 'none'}; "
    f"module docstring differs: {ast.get_docstring(ast.parse(tsrc[A2CODE])) != ast.get_docstring(ast.parse(tsrc[REC]))}")
diff_run = sorted(k for k in set(s_run) | set(s_a2) if s_run.get(k) != s_a2.get(k))
say(f"  ... and between {RUNCODE} and {A2CODE}: {diff_run}")
doc = ast.get_docstring(ast.parse(tsrc[REC]))
RESULT_X = doc[doc.index("RESULT (X)"):]

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("1. The test's own report, regenerated from the records (functions extracted from the sources at 8ae660d)")
say("=" * 100)
tbo = module(REC, "test_binding_onset.py", {"EVAL_EVERY", "fmt_step"})
trd = module(REC, "test_router_discovery.py", {"get", "load_store"})
tcl = module(REC, "test_conv_lr.py", {"bound_r", "mcnemar_greater"})
tsch = module(REC, "test_stream_channels.py", {"med_int", "mstr", "shared_max", "merged"})
trl = module(REC, "test_router_layout.py", {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"})
tsa = module(REC, "test_scale_axes.py", {"fail_class_k"}, fail_class=trl.fail_class)
tsr = module(REC, "test_stream_recipe.py", {"routed", "outcome", "wilson", "ROUTED_ACC"},
             fail_class_k=tsa.fail_class_k, merged=tsch.merged, shared_max=tsch.shared_max)
fc2 = module(REC, "fail_class_v2.py", {"fail_class_v2", "multichannel", "fail_class_k_v2", "outcome_v2"},
             MARGIN_PARTIAL=trl.MARGIN_PARTIAL, ETA_SPLIT=trl.ETA_SPLIT, merged=tsch.merged, shared_max=tsch.shared_max)
tscur = module(REC, "test_stream_curriculum.py", {"band_n", "RELIABLE_F", "MAJORITY_F"})
tss = module(REC, "test_slow_start.py", {"ok_r", "runs_of", "curve_str"}, get=trd.get)
trs = module(REC, "test_recipe_scope.py", {"groups_str"})
tgw_src = src(REC, "test_window_gate.py")
tgw = module(REC, "test_window_gate.py", {"ACC_THR", "RISE", "EVERY", "FIRST", "REF_AT", "MAX_SPLITS", "GAP", "VALID_N"})
assert "MAX_A, MAX_B, MAX_C = tsc.MAX_ITERS, tsr.MAX_ITERS, 43200" in tgw_src
tgw.MAX_C = 43200
say(f"  test_window_gate constants: FIRST {tgw.FIRST}, ACC_THR {tgw.ACC_THR}, RISE {tgw.RISE}, MAX_SPLITS {tgw.MAX_SPLITS}, "
    f"GAP {tgw.GAP}, VALID_N {tgw.VALID_N}; ROUTED_ACC {tsr.ROUTED_ACC}; bands {tscur.RELIABLE_F}/{tscur.MAJORITY_F}")

REPORT_NAMES = {"MACHINES", "RANGES", "ITERS", "VALID_N", "D1_BOUND", "BETA", "OVERLAP_MAX", "ARMS", "ARM", "KEYS",
                "ARMS_RUN", "CEILS", "PARTS", "NO_ROOM", "D1_LABELS", "SPLIT_READINGS", "D1_OC", "seeds_for", "SEEDS",
                "outcome2", "success", "overlap", "paired", "sign_test", "d3_rows", "d3_of", "validity", "tally",
                "paired_full", "newcombe_paired", "d1_label", "split_verdict", "split_label", "split_reading", "pool2",
                "band_str", "diff_str", "print_claims", "broken_pairs", "check_174", "probe_blind", "print_bands",
                "complete", "stream_str", "raw_rows", "report", "other_of"}


def test_ns(ref):
    names = {n for n in REPORT_NAMES if n in segments(tsrc[ref])}
    ns = dict(math=math, statistics=statistics, collections=collections, tgw=tgw, tsr=tsr, tscur=tscur, tss=tss, fc2=fc2,
              ok_r=tss.ok_r, get=trd.get, bound_r=tcl.bound_r, mcnemar_greater=tcl.mcnemar_greater, med_int=tsch.med_int,
              mstr=tsch.mstr, shared_max=tsch.shared_max, fmt_step=tbo.fmt_step, EVAL_EVERY=tbo.EVAL_EVERY,
              groups_str=trs.groups_str, curve_str=tss.curve_str, load_store=trd.load_store)
    return extract(tsrc[ref], names, ns)


def run_report(ns):
    seeds = {k: tuple(v) for k, v in meta["seeds"].items()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["report"](store, seeds, "X", 0.0, "results/X/delta_recipe_results.json", None)
    return buf.getvalue().split("\n"), ns["tally"](store, seeds)


T = test_ns(REC)
rep2, t = run_report(T)
Trun = test_ns(RUNCODE)
rep1, t1 = run_report(Trun)


def section(lines, start, stop):
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next(k for k, l in enumerate(lines) if k > i and l.startswith(stop))
    return lines[i:j]


def grab(lines, *keys):
    return [l for l in lines if all(k in l for k in keys)]


for lab, rep in (("Amendment 2 (--report under 2225f00)", rep2), ("in-run (a581027)", rep1)):
    say(f"\n  -- {lab}: validity, counts, claims, bands, checks (verbatim lines of the regenerated report) --")
    for l in (section(rep, "VALIDITY", "COUNTS")[2:4] + section(rep, "COUNTS", "#")[2:6]
              + section(rep, "PER MACHINE", "=====")
              + grab(rep, "CHECK 171") + grab(rep, "CHECK 174") + grab(rep, "probe-blind checks")
              + grab(rep, "firings in")):
        say("   |" + l)

# line-for-line comparison of claims, bands, checks (the RESULT's statement)
def claims_block(rep):
    blk = section(rep, "PER MACHINE", "=====")
    blk += grab(rep, "CHECK 171") + grab(rep, "CHECK 174") + grab(rep, "probe-blind checks") + grab(rep, "firings in")
    return blk


b1, b2 = claims_block(rep1), claims_block(rep2)
same = [x for x in b2 if x in b1]
only2 = [x for x in b2 if x not in b1]
only1 = [x for x in b1 if x not in b2]
say(f"\n  in-run vs Amendment 2 report, claims/bands/checks block: {len(same)} lines identical; only in Amendment 2's "
    f"{len(only2)}; only in the in-run {len(only1)}")
for x in only1:
    say("   in-run only  |" + x)
for x in only2:
    say("   A2 only      |" + x)
full_diff = sum(1 for a, b in zip(rep1, rep2) if a != b) + abs(len(rep1) - len(rep2))
say(f"  whole reports: {len(rep1)} vs {len(rep2)} lines; lines that differ (position by position): {full_diff}")
hdr = grab(rep2, "PRE-REGISTERED CLAIMS")[0][:120], grab(rep1, "PRE-REGISTERED CLAIMS")[0][:120]
say(f"  claims header A2: {hdr[0]!r}\n  claims header run: {hdr[1]!r}")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("2. Independent recomputation (rlib outcomes; own statistics)")
say("=" * 100)
ARMS = ["CEIL_HEBB", "CEIL_DELTA", "HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW"]
runs = {a: rlib.runs(store, a) for a in ARMS}
SEEDS20 = list(range(580, 600))
assert all(sorted(runs[a]) == (SEEDS20[:2] if a.startswith("CEIL") else SEEDS20) for a in ARMS)
assert all(r.get("ok") for a in ARMS for r in runs[a].values())
trans_eq = all(rlib.transition(r["curve"]) == r["transition"] for a in ARMS for r in runs[a].values())
br = {a: {s: (rlib.bound(r) if a.startswith("CEIL") else rlib.bound_routed(r)) for s, r in runs[a].items()} for a in ARMS}
code = {a: {s: T["success"](a, r) for s, r in runs[a].items()} for a in ARMS}
agree = all(br[a] == code[a] for a in ARMS)
say(f"  rlib transition (first eval >= 0.95 held to the end) = record's transition on all 84: {trans_eq}; "
    f"rlib outcome = the test's success() on all 84: {agree}; stream_gate ties decided by ch_map: {len(rlib.TIES)}")
cnt = {a: sum(br[a].values()) for a in ARMS}
med_tr = {a: statistics.median([runs[a][s]["transition"] for s in br[a] if br[a][s]]) for a in ARMS}
for a in ARMS:
    fails = [s for s in br[a] if not br[a][s]]
    say(f"  {a:<12} {cnt[a]:>2}/{len(br[a])}  median transition {med_tr[a]:.0f}  not bound(-routed): "
        + ", ".join(f"s{s} {rlib.outcome_k(runs[a][s]) if not a.startswith('CEIL') else 'not bound'}" for s in fails))
ceil_tr = {a: [runs[a][s]["transition"] for s in (580, 581)] for a in ("CEIL_HEBB", "CEIL_DELTA")}
say(f"  validity: CEIL_HEBB {cnt['CEIL_HEBB']}/2 {ceil_tr['CEIL_HEBB']}; CEIL_DELTA {cnt['CEIL_DELTA']}/2 {ceil_tr['CEIL_DELTA']}")
ov_ok = {a: sum(1 for s, r in runs[a].items() if br[a][s] and T["overlap"](r) <= 0.1) for a in ARMS[2:]}
ov_max = max(T["overlap"](r) for a in ARMS[2:] for s, r in runs[a].items() if br[a][s])
merged = {a: [s for s, r in runs[a].items() if rlib.outcome_k(r).startswith("MERGED")] for a in ARMS[2:]}
say(f"  BOUND ROUTED with overlap <= 0.1: {ov_ok} (largest overlap over BR runs {ov_max:.4f}); MERGED: {merged}")
same_outcome = [s for s in SEEDS20 if rlib.outcome_k(runs['DELTA_SPLIT'][s]) != rlib.outcome_k(runs['DELTA_SLOW'][s])]
say(f"  DELTA_SPLIT vs DELTA_SLOW: seeds with a different outcome string: {same_outcome or 'none'}; "
    f"different BR: {[s for s in SEEDS20 if br['DELTA_SPLIT'][s] != br['DELTA_SLOW'][s]] or 'none'}")


def disc(x, y):
    b = [s for s in SEEDS20 if br[x][s] and not br[y][s]]
    c = [s for s in SEEDS20 if br[y][s] and not br[x][s]]
    a = [s for s in SEEDS20 if br[x][s] and br[y][s]]
    return a, b, c


# Newcombe method 10, extracted from the referee's spec_checks_delta2.py (written there from the formula)
ns_nc = {"math": math}
sc2 = open(os.path.join(REVIEW, "spec_checks_delta2.py")).read()
extract(sc2, {"wilson_z", "newcombe10"}, ns_nc)
a1, b1_, c1 = disc("DELTA_SPLIT", "HEBB_SPLIT")
dd1 = 20 - len(a1) - len(b1_) - len(c1)
nc = ns_nc["newcombe10"](len(a1), len(b1_), len(c1), dd1)
nc_t = T["newcombe_paired"](len(a1), len(b1_), len(c1), dd1)
d_val = len(c1) - len(b1_)
lab2 = T["D1_LABELS"][0] if d_val <= -3 else (T["D1_LABELS"][1] if d_val <= 2 else T["D1_LABELS"][2])
say(f"  D1: both {len(a1)}, DELTA_SPLIT only {len(b1_)} {b1_}, HEBB_SPLIT only {len(c1)} {c1}, neither {dd1}; "
    f"d = HEBB only - DELTA only = {d_val}; Amendment 2 label (pooled only; here per machine) would be '{lab2}'")
say(f"      DELTA - HEBB: referee's newcombe10 {nc[0]:+.4f} [{nc[1]:+.4f}, {nc[2]:+.4f}]; test's newcombe_paired "
    f"{nc_t[0]:+.4f} [{nc_t[1]:+.4f}, {nc_t[2]:+.4f}]; largest difference {max(abs(x - y) for x, y in zip(nc, nc_t)):.1e}")
say(f"      one-sided McNemar DELTA over HEBB (printed nowhere, for the label's evidence): p = {rlib.mcnemar1(len(b1_), len(c1)):.4f}")
res = {}
for name, x, y in (("D2", "DELTA_SPLIT", "DELTA_SLOW"), ("D4", "HEBB_SPLIT", "HEBB_SLOW")):
    a, b, c = disc(x, y)
    p = rlib.mcnemar1(len(b), len(c))
    res[name] = (len(b), len(c), p)
    say(f"  {name}: {x} {cnt[x]}/20 vs {y} {cnt[y]}/20; {x} only {len(b)} {b}, {y} only {len(c)} {c}; one-sided exact "
        f"McNemar p = {p:.6f} -> {'SHOWN' if p < 0.05 else 'NOT SHOWN'}")
# the no-room rule on one machine
say(f"  no-room rule (pooled >= 36/40): X's DELTA_SLOW {cnt['DELTA_SLOW']}/20, HEBB_SLOW {cnt['HEBB_SLOW']}/20. Per machine "
    f"the code prints no NO ROOM (print_claims: 'pooled and d[\"old\"] >= NO_ROOM'); pooled, delta is NO ROOM iff L's "
    f"DELTA_SLOW >= {36 - cnt['DELTA_SLOW']}/20 (on the pairs CHECK 174 keeps); Hebbian cannot be (max {cnt['HEBB_SLOW'] + 20}/40)")
say(f"  on X alone at most {20 - cnt['DELTA_SLOW']} DELTA_SPLIT-only pair was possible: best p = "
    f"{rlib.mcnemar1(20 - cnt['DELTA_SLOW'], 0):.2f}, so D2 could not be SHOWN on X whatever the split did")
# D3
d3 = [(s, runs["HEBB_SPLIT"][s]["transition"], runs["DELTA_SPLIT"][s]["transition"]) for s in SEEDS20
      if br["HEBB_SPLIT"][s] and br["DELTA_SPLIT"][s]]
pos = sum(1 for _, h, d in d3 if d < h)
neg = sum(1 for _, h, d in d3 if d > h)
p3 = rlib.binom_tail(pos + neg, pos)
say(f"  D3: {len(d3)} seeds both split arms BR; delta earlier {pos}, later {neg}, tied {len(d3) - pos - neg}; one-sided sign "
    f"p = {p3:.4f}; medians DELTA {statistics.median([d for *_, d in d3]):.0f}, HEBB {statistics.median([h for _, h, _ in d3]):.0f}")
say("      rows: " + ", ".join(f"s{s} {h}/{d}" for s, h, d in d3))
# bands
for a in ARMS[2:]:
    lo, hi = rlib.wilson(cnt[a], 20)
    say(f"  band {a:<12} {cnt[a]}/20 {rlib.band(cnt[a], 20):<9} Wilson [{lo:.3f}, {hi:.3f}]")


# CHECK 174, written here from the spec's words
def first_fire(r):
    f = [c["step"] for c in r.get("checks") or [] if c["fired"]]
    return f[0] if f else None


broken = []
for x, y in (("DELTA_SPLIT", "DELTA_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW")):
    for s in SEEDS20:
        f = first_fire(runs[x][s])
        cx = [p for p in runs[x][s]["curve"] if f is None or p[0] <= f]
        cy = [p for p in runs[y][s]["curve"] if f is None or p[0] <= f]
        if cx != cy:
            broken.append(f"{x}|{s}")
nofire = {x: [s for s in SEEDS20 if first_fire(runs[x][s]) is None] for x in ("DELTA_SPLIT", "HEBB_SPLIT")}
say(f"  CHECK 174 (own): 40 pairs, broken {broken or 'none'}; split runs that never fired (whole curve compared): {nofire}")
nf_eq = {x: [s for s in nofire[x] if runs[x][s]["curve"] == runs[x.replace('SPLIT', 'SLOW')][s]["curve"]] for x in nofire}
say(f"      of those, whole curves identical to the no-trigger arm's: {nf_eq}")


# probe-blind (Amendment 2 (f)), own reading
def pblind(arm):
    rows_, blind, alone = 0, 0, 0
    for s in SEEDS20:
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


pb = {a: pblind(a) for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
all_rows = {a: sum(len(runs[a][s]["checks"]) for s in SEEDS20) for a in pb}
say(f"  probe-blind (own; checks >= 4800): " + "; ".join(f"{a} {b} of {n}, blocked by the probe alone {al} "
                                                      f"(all check rows incl. 2400: {all_rows[a]})" for a, (n, b, al) in pb.items()))
fire = {}
for a in ("HEBB_SPLIT", "DELTA_SPLIT"):
    fs = [c for s in SEEDS20 for c in runs[a][s]["checks"] if c["fired"]]
    fire[a] = (len(fs), sum(1 for c in fs if c["key_ok"]), sum(1 for s in SEEDS20 if any(c["fired"] for c in runs[a][s]["checks"])),
               sum(1 for s in SEEDS20 if runs[a][s].get("splits", 0) > 0))
    say(f"  firings {a}: {fire[a][0]} in 20 runs ({fire[a][2]} runs fired), targets labelled ok {fire[a][1]}/{fire[a][0]}")
cpus = collections.Counter(r["cpu"] for a in ARMS for r in runs[a].values())
say(f"  CPU labels in the 84 records: {dict(cpus)}; sum of secs {sum(r['secs'] for r in R.values()) / 3600:.2f} h")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("3. Amendment 2's timing, reconstructed from the records (no log on any ref)")
say("=" * 100)
ms = meta["ms_per_step"]
CEILS = {"CEIL_HEBB", "CEIL_DELTA"}
order = sorted(meta["seeds"], key=lambda k: (k not in CEILS, -ms[k]))
specs = [(k, s) for k in order for s in meta["seeds"][k]]
P_MIN = sum(ms[k] * 1200 / 1000 for k in ms)       # each arm's pool.map of 4 x 1200 steps lasts >= its median
STARTED = dt.datetime.strptime(meta["started"], "%Y-%m-%d %H:%M:%S")      # UTC (see the .md)
T_55, T_22 = dt.datetime(2026, 10, 11, 15, 36, 8), dt.datetime(2026, 10, 11, 15, 37, 52)
T_REC = dt.datetime(2026, 10, 12, 7, 25, 51)
T_RUN = dt.datetime(2026, 10, 11, 2, 12, 29)


def simulate(P, over):
    """The test's scheduler: pending in `specs` order, 4 in flight, refill on each finish; durations secs + over."""
    clock, inflight, pending, fin = 0.0, [], list(specs), {}
    while pending or inflight:
        while pending and len(inflight) < meta["workers"]:
            k, s = pending.pop(0)
            inflight.append((clock + R[f"{k}|{s}"]["secs"] + over.get((k, s), 0.0), k, s, clock))
        inflight.sort()
        f, k, s, st = inflight.pop(0)
        clock = f
        fin[(k, s)] = (STARTED + dt.timedelta(seconds=P + st), STARTED + dt.timedelta(seconds=P + f))
    return fin


base = simulate(P_MIN, {})
end_min = max(f for _, f in base.values())
say(f"  meta.started {meta['started']} (UTC); projection lower bound {P_MIN:.0f} s -> first runs submitted >= "
    f"{STARTED + dt.timedelta(seconds=P_MIN):%H:%M:%S}; last run ends >= {end_min:%Y-%m-%d %H:%M:%S} "
    f"(makespan {(end_min - STARTED).total_seconds() / 3600 - P_MIN / 3600:.3f} h); result commit {T_REC:%Y-%m-%d %H:%M:%S}: "
    f"slack <= {(T_REC - end_min).total_seconds():.0f} s")
say(f"  from the test's commit (02:12:29) to the earliest end: {(end_min - T_RUN).total_seconds() / 3600:.2f} h; to the "
    f"result commit {(T_REC - T_RUN).total_seconds() / 3600:.2f} h; verification + start {(STARTED - T_RUN).total_seconds() / 60:.0f} min")
say(f"  the test's own 'total wall clock after verification' would be >= {(end_min - STARTED).total_seconds() / 3600:.2f} h; "
    f"under --report it prints 0.0 min")
say("  finish times, no overhead (each a lower bound), around the two commits:")
srt = sorted(base.items(), key=lambda x: x[1][1])
for i, ((k, s), (st, f)) in enumerate(srt, 1):
    if dt.datetime(2026, 10, 11, 14, 0) <= f <= dt.datetime(2026, 10, 11, 17, 0) or i <= 4 or i in (23, 24, 25, 26):
        say(f"   #{i:>2} {k:<12} {s}  start {st:%H:%M:%S}  end {f:%H:%M:%S}  ({(f - T_55).total_seconds():+7.0f} s from 55dac50)")
fin55 = [ks for ks, (_, f) in base.items() if f <= T_55]
say(f"  finished at 15:36:08, no overhead: {len(fin55)}: " + ", ".join(f"{k}|{s}" for k, s in sorted(fin55)))
# Monte Carlo over overheads, constrained by the result commit
rng = random.Random(20261012)
sets55, sets22, acc = collections.Counter(), collections.Counter(), 0
t599, t580s, t581s = [], [], []
for _ in range(4000):
    P = P_MIN + rng.uniform(0, 360)
    omax = rng.uniform(0, 8)
    over = {ks: rng.uniform(0, omax) for ks in specs}
    fin = simulate(P, over)
    if max(f for _, f in fin.values()) > T_REC:
        continue
    acc += 1
    f55 = frozenset(ks for ks, (_, f) in fin.items() if f <= T_55)
    f22 = frozenset(ks for ks, (_, f) in fin.items() if f <= T_22)
    sets55[f55] += 1
    sets22[f22] += 1
    t599.append((fin[("DELTA_SPLIT", 599)][1] - T_55).total_seconds())
    t580s.append((fin[("DELTA_SLOW", 580)][1] - T_55).total_seconds())
    t581s.append((fin[("DELTA_SLOW", 581)][1] - T_55).total_seconds())
AMEND = frozenset([("CEIL_HEBB", 580), ("CEIL_HEBB", 581), ("CEIL_DELTA", 580), ("CEIL_DELTA", 581), ("DELTA_SLOW", 580)]
                  + [("DELTA_SPLIT", s) for s in range(580, 599)])


def describe(fs):
    extra, miss = sorted(fs - AMEND), sorted(AMEND - fs)
    return f"{len(fs)} runs" + ("" if not extra and not miss else f" (vs the amendment: + {extra}, - {miss})")


say(f"  Monte Carlo: projection +U(0, 360 s), per-run overhead U(0, omax), omax U(0, 8 s); {acc} of 4000 draws end "
    f"before the result commit")
for fs, n in sets55.most_common():
    say(f"   at 15:36:08: {describe(fs)}: {n / acc:.3f}{'   = the amendment' if fs == AMEND else ''}")
for fs, n in sets22.most_common():
    say(f"   at 15:37:52: {describe(fs)}: {n / acc:.3f}{'   = the amendment' if fs == AMEND else ''}")
t599.sort()
say(f"  DELTA_SPLIT|599's end relative to 15:36:08 over the accepted draws: min {t599[0]:+.0f} s, median "
    f"{t599[len(t599) // 2]:+.0f} s, max {t599[-1]:+.0f} s")
for nm, xs in (("DELTA_SLOW|580", t580s), ("DELTA_SLOW|581", t581s)):
    xs.sort()
    say(f"  {nm}'s end relative to 15:36:08: min {xs[0]:+.0f} s, max {xs[-1]:+.0f} s")
chain, cur = [], ("DELTA_SPLIT", 599)
while cur:
    chain.append(cur)
    stc = base[cur][0]
    cur = next((ks for ks, (_, f) in base.items() if abs((f - stc).total_seconds()) < 1e-6), None)
say(f"  DELTA_SPLIT|599's worker chain (no overhead): {' <- '.join(f'{k}|{s}' for k, s in chain)} ({len(chain)} runs, "
    f"+ the projection): >= 4 s of unrecorded time on it puts its end after 15:36:08, >= 108 s after 15:37:52")
# the zone from git and the records alone: started after the test's commit; the earliest end before the result commit
h_max = (STARTED - T_RUN).total_seconds() / 3600
h_min = (end_min - T_REC).total_seconds() / 3600
say(f"  zone of meta.started from git and records alone: UTC{h_min:+.2f} h <= offset <= UTC+{h_max:.2f} h, i.e. whole-hour "
    f"zones UTC, +1, +2, +3; with the RESULT's 02:12 UTC start and 29.2 h only UTC fits (end >= 07:20 local vs 07:24 UTC)")
hebb_first = min(f for (k, _), (_, f) in base.items() if k.startswith("HEBB"))
slow2 = sorted(f for (k, s), (_, f) in base.items() if k == "DELTA_SLOW")[1]
say(f"  earliest Hebbian learned-gate run end (lower bound) {hebb_first:%Y-%m-%d %H:%M}; second DELTA_SLOW end "
    f"{slow2:%H:%M:%S} (lower bound)")
# other zones: what the amendment's set would require
for h in (-1, 1, 2, 3):
    Tz = T_55 + dt.timedelta(hours=h)
    n = len([ks for ks, (_, f) in base.items() if f <= Tz])
    say(f"   if meta.started were UTC{h:+d}, 15:36:08 UTC would be {Tz:%H:%M:%S} in its clock: {n} runs finished (no overhead)")

# ════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("4. RESULT (X) block and the draft")
say("=" * 100)
tex = src(DRAFT, TEX).split("\n")


def tex_has(n, frag):
    ok = frag in tex[n - 1]
    if not ok:
        raise AssertionError(f"tex5:{n} lacks {frag!r}")
    return ok


def res_has(frag):
    flat = " ".join(RESULT_X.split())
    if " ".join(frag.split()) not in flat:
        raise AssertionError(f"RESULT (X) lacks {frag!r}")
    return True


# split_copy's WIN3_SPLIT_D8, for the draft's comparison
sc = {m: rlib.runs(json.loads(src(REC, f"results/{m}/split_copy_results.json")), "WIN3_SPLIT_D8") for m in "XL"}
sc_br = {m: sum(rlib.bound_routed(r) for r in sc[m].values()) for m in "XL"}
say(f"  test_split_copy WIN3_SPLIT_D8 (noisy copy + Adam reset): X {sc_br['X']}/{len(sc['X'])} seeds {min(sc['X'])}-{max(sc['X'])}, "
    f"L {sc_br['L']}/{len(sc['L'])} seeds {min(sc['L'])}-{max(sc['L'])}")

D1 = t["D1"]
c2, c4, c3 = t["D2"], t["D4"], t["D3"]
pblind_code = {k: T["probe_blind"](t, k) for k in ("HEBB_SPLIT", "DELTA_SPLIT")}
assert (D1["b"], D1["c"], D1["a"], D1["dd"]) == (len(b1_), len(c1), len(a1), dd1)
assert (c2["b"], c2["c"]) == res["D2"][:2] and (c4["b"], c4["c"]) == res["D4"][:2]
assert (c3["pos"], c3["neg"], c3["n"]) == (pos, neg, len(d3))
assert all((len(pblind_code[k][0]), pblind_code[k][1], pblind_code[k][2]) == pb[k] for k in pb)
n174, bad174 = T["check_174"](t)
assert bad174 == [] and broken == [] and n174 == 40

wall_note = (f"Not in the records. Consistent: 02:12:29 + 29.2 h = 07:24; the reconstructed end is >= {end_min:%H:%M} and the "
             f"result commit 07:25:51. The test's own report prints wall time after verification (>= "
             f"{(end_min - STARTED).total_seconds() / 3600:.1f} h in-run; 0.0 min under --report), so 29.2 h is from "
             f"the longrun start, not a printed figure")

# RESULT (X) rows
res_has("84/84 runs, none failed")
row("RESULT", "test commit a581027; 84/84 runs, none failed; one start", f"meta.git {meta['git']}; {len(R)} records, all ok; "
    f"no meta.starts (resume_meta would add one on a restart)", "MATCH")
res_has("The run started 2026-10-11 02:12 UTC")
row("RESULT", "run started 02:12 UTC through longrun.py; runs began at 05:27 after the verification",
    f"meta.started {meta['started']} (UTC by inference, .md §3); 02:12 is not in the records (test commit 02:12:29); runs "
    f"were submitted after the projection, >= {STARTED + dt.timedelta(seconds=P_MIN):%H:%M}", "PARTIAL",
    "05:27:31 is when verification ended and the projection began; the first run started >= 21 min later. 02:12 rests on "
    "the RESULT and Amendment 2 alone (no log)")
res_has("29.2 h wall clock (projection 45.3 h worst case)")
row("RESULT", "29.2 h wall clock (projection 45.3 h worst case)",
    f"projected_wall_h {meta['projected_wall_h']:.2f}; reconstructed 02:12:29 -> end >= {end_min:%m-%d %H:%M} = "
    f">= {(end_min - T_RUN).total_seconds() / 3600:.2f} h", "MATCH", wall_note)
res_has("CEIL_HEBB bound 2/2 (3600, 3600); CEIL_DELTA 2/2 (2400, 1200). Both parts VALID on X")
row("RESULT", "Validity: CEIL_HEBB 2/2 (3600, 3600); CEIL_DELTA 2/2 (2400, 1200); both parts VALID",
    f"CEIL_HEBB {cnt['CEIL_HEBB']}/2 {ceil_tr['CEIL_HEBB']}; CEIL_DELTA {cnt['CEIL_DELTA']}/2 {ceil_tr['CEIL_DELTA']}; "
    f"validity {t['valid']}", "MATCH")
for a, frag in (("HEBB_SPLIT", "HEBB_SPLIT 16/20 (median transition 17400)"), ("HEBB_SLOW", "HEBB_SLOW 7/20"),
                ("DELTA_SPLIT", "DELTA_SPLIT 19/20 (13200)"), ("DELTA_SLOW", "DELTA_SLOW 19/20 (14400")):
    res_has(frag)
    row("RESULT", frag, f"{a} {cnt[a]}/20, median transition {med_tr[a]:.0f}", "MATCH")
res_has("per seed the same outcome as DELTA_SPLIT, both failing only on s594, MERGED")
row("RESULT", "DELTA_SLOW per seed the same outcome as DELTA_SPLIT, both failing only on s594, MERGED",
    f"seeds with different outcomes {same_outcome or 'none'}; MERGED {merged['DELTA_SPLIT']}, {merged['DELTA_SLOW']}", "MATCH")
res_has("every BOUND ROUTED run has overlap <= 0.1")
row("RESULT", "every BOUND ROUTED run has overlap <= 0.1", f"{ov_ok} vs counts; largest {ov_max:.4f}", "MATCH")
res_has("D1: DELTA_SPLIT only 4, HEBB_SPLIT only 1, d = -3; rate difference +0.150, 95% [-0.082, +0.374]")
row("RESULT", "D1: DELTA only 4, HEBB only 1, d = -3; +0.150, 95% [-0.082, +0.374]",
    f"DELTA only {len(b1_)} {b1_}, HEBB only {len(c1)} {c1}, d = {d_val}; newcombe10 {nc[0]:+.3f} [{nc[1]:+.3f}, {nc[2]:+.3f}] "
    f"(test's function: same to {max(abs(x - y) for x, y in zip(nc, nc_t)):.0e})", "MATCH",
    f"Per machine no label is printed (correct per spec); pooled, d = -3 would read '{T['D1_LABELS'][0]}'")
res_has("D2: 0 vs 0, p = 1, NOT SHOWN (DELTA_SLOW 19/20)")
row("RESULT", "D2: 0 vs 0, p = 1, NOT SHOWN (DELTA_SLOW 19/20)", f"{res['D2'][0]} vs {res['D2'][1]}, p = {res['D2'][2]:g}; "
    f"code: {T['split_verdict'](c2)}, dropped by CHECK 174 {c2['dropped']}", "MATCH")
res_has("D4: 10 vs 1, p = 0.0059, SHOWN")
row("RESULT", "D4: 10 vs 1, p = 0.0059, SHOWN", f"{res['D4'][0]} vs {res['D4'][1]}, p = {res['D4'][2]:.6f}; code "
    f"{T['split_verdict'](c4)}", "MATCH", "p = 12/2048")
res_has("D3: on 15 shared binds, delta earlier on 10, later on 4, tied 1; sign test p = 0.090; medians 13200 vs 16800")
row("RESULT", "D3: 15 shared binds, earlier 10, later 4, tied 1; p = 0.090; medians 13200 vs 16800",
    f"{len(d3)} shared, earlier {pos}, later {neg}, tied {len(d3) - pos - neg}; p = {p3:.4f}; medians "
    f"{statistics.median([d for *_, d in d3]):.0f} vs {statistics.median([h for _, h, _ in d3]):.0f}", "MATCH")
res_has("HEBB_SPLIT MAJORITY [0.584, 0.919]; HEBB_SLOW MINORITY [0.181, 0.567]; DELTA_SPLIT RELIABLE [0.764, 0.991]")
row("RESULT", "Bands: HEBB_SPLIT MAJORITY [0.584, 0.919]; HEBB_SLOW MINORITY [0.181, 0.567]; DELTA_SPLIT, DELTA_SLOW "
    "RELIABLE [0.764, 0.991]", "; ".join(f"{a} {rlib.band(cnt[a], 20)} [{rlib.wilson(cnt[a], 20)[0]:.3f}, "
                                         f"{rlib.wilson(cnt[a], 20)[1]:.3f}]" for a in ARMS[2:]), "MATCH")
res_has("Firings: Hebbian 39 in 20 runs, every target labelled ok; delta 31 in 20 runs, 10 labelled ok")
row("RESULT", "Firings: Hebbian 39 in 20 runs, every target ok; delta 31 in 20 runs, 10 ok",
    f"HEBB_SPLIT {fire['HEBB_SPLIT'][0]} firings, {fire['HEBB_SPLIT'][1]} ok, {fire['HEBB_SPLIT'][2]} runs fired; DELTA_SPLIT "
    f"{fire['DELTA_SPLIT'][0]}, {fire['DELTA_SPLIT'][1]} ok, {fire['DELTA_SPLIT'][2]} runs fired", "MATCH",
    "'in 20 runs' is the runs counted, as the code prints it, not the runs that fired")
res_has("Probe-blind (checks from 4800): HEBB_SPLIT 1 of 181, blocked by the probe alone 3; DELTA_SPLIT 1 of 142, alone 5")
row("RESULT", "Probe-blind: HEBB_SPLIT 1 of 181, alone 3; DELTA_SPLIT 1 of 142, alone 5",
    f"HEBB_SPLIT {pb['HEBB_SPLIT'][1]} of {pb['HEBB_SPLIT'][0]}, alone {pb['HEBB_SPLIT'][2]}; DELTA_SPLIT {pb['DELTA_SPLIT'][1]} of "
    f"{pb['DELTA_SPLIT'][0]}, alone {pb['DELTA_SPLIT'][2]} (own and the code's)", "MATCH")
c171 = grab(rep2, "CHECK 171")[0]
res_has("CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs)")
row("RESULT", "CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs)", f"code: '{c171.strip()[:110]}'; CHECK 174 "
    f"{n174} pairs, broken {bad174 or 'none'} (own: {broken or 'none'})", "MATCH")
res_has("Its claims, bands and checks equal the in-run report's line for line: CHECK 174 dropped no pair")
row("RESULT", "the --report figures' claims, bands and checks equal the in-run report's line for line",
    f"regenerated both from the records: {len(same)} lines identical, {len(only2)} only in A2's, {len(only1)} only in the "
    f"in-run (the probe-blind lines, which Amendment 2 (f) changed; the claims and bands identical)", "PARTIAL",
    "True of claims and bands and of CHECKs 171/174; the probe-blind lines differ by construction (A2 (f)); the in-run "
    "line counts all checks incl. 2400. Without the log the in-run report is a reconstruction")
row("RESULT", "the figures are Amendment 2's report regenerated with --report (2225f00) from the same results file",
    "the RESULT's probe-blind figures (1 of 181, alone 3; 1 of 142, alone 5) are what only 2225f00's probe_blind prints "
    "(a581027's prints 1 of 201, 1 of 162, no 'alone'); 8ae660d's report functions = 2225f00's (AST)", "MATCH",
    "That both reports are kept in the log cannot be checked: no log is on any ref")
row("msg 8ae660d", "commit message: 'D3 delta earlier on 10 of 14, p = 0.090'", f"earlier {pos} of {len(d3)} shared binds "
    f"({pos + neg} untied), p = {p3:.4f}", "PARTIAL", "14 is the untied count; the RESULT and the draft say 10 of 15")
row("msg 902c39a", "draft commit title: 'D2 no room'", "the test printed D2 NOT SHOWN; NO ROOM is pooled only and was not "
    "printed", "PARTIAL", "as tex5:256's 'leaving no room'")
row("RESULT", "Verification: CHECKs 130, 158, 159, 162, 161 (1.19e-7), 173", "not in the records (no log)",
    "NOT RECORD-CHECKABLE", "")

# draft rows
tex_has(41, "complete on X (a second amendment, to the report's rules only, was committed during X's run with a statement of what had been seen); L's half is next")
row("tex5:41", "test_delta_recipe ... complete on X (a second amendment ... committed during X's run ...); L's half is next",
    f"{len(R)}/84 records ok; 55dac50 15:36:08 between meta.started and the reconstructed end; no L file at {REC}",
    "MATCH")
tex_has(41, "\\texttt{test\\_delta\\_recipe}'s L half has not run")
row("tex5:41", "test_delta_recipe's L half has not run", f"results/L/delta_recipe_results.json at {REC}: "
    f"{'results/L/delta_recipe_results.json' in git('ls-tree', '-r', '--name-only', REC, 'results/L/')}", "MATCH",
    "As of the record; the round-5 prompts (0754d61) say L is running it")
tex_has(54, "on X (secondary): D4 SHOWN 10 vs 1, D2 0 vs 0 with the delta no-trigger arm at $19/20$, D1 $d = -3$; pooled labels and readings wait for L")
row("tex5:54", "index row: on X (secondary): D4 SHOWN 10 vs 1, D2 0 vs 0 with DELTA_SLOW 19/20, D1 d = -3; pooled waits for L",
    f"D4 {res['D4'][0]} vs {res['D4'][1]} SHOWN; D2 {res['D2'][0]} vs {res['D2'][1]}, DELTA_SLOW {cnt['DELTA_SLOW']}/20; d = {d_val}",
    "MATCH", "The first column still says '(specified)' though X has run")
tex_has(256, "a statement of the 24 run lines that had been read, none of them a claim, a reading or a Hebbian run")
row("tex5:256", "Amendment 2's statement: 24 run lines read, 'none of them a claim, a reading or a Hebbian run'",
    "Amendment 2 lists CEIL_HEBB 580, 581 among the 24: two Hebbian (perfect-gate) runs; it says no HEBB_SPLIT or "
    "HEBB_SLOW run had finished", "DISAGREE",
    "Should read 'no Hebbian learned-gate run' (or 'no HEBB_SPLIT or HEBB_SLOW run'). The count 24 matches the amendment "
    "and the reconstruction (§3)")
tex_has(256, "labelled ``delta binds more often'' ($d \\leq -3$)")
row("tex5:256", "D1 labels: 'delta binds more often' (d <= -3), ...",
    f"Amendment 2 (a) replaced it: '{T['D1_LABELS'][0]}' (D1_LABELS[0] at 2225f00/8ae660d)", "DISAGREE",
    "Pre-Amendment-2 label in the same paragraph that reports the amendment; with d = -3 on X, it is the label the "
    "pooled D1 would print if L's d <= 0")
tex_has(256, "(a ten-point deficit passes it 22--29\\% of the time at 40 pairs)")
row("tex5:256", "a ten-point deficit passes it 22-29% of the time at 40 pairs", "D1_OC: P(d <= 2) 0.29/0.22 at DELTA 0.85 "
    "vs HEBB 0.95", "MATCH", "Unchanged text; quoted from the test's D1_OC")
tex_has(256, "X's half is complete (seeds 580--599, 84 runs, 29.2 h; both perfect gates $2/2$; CHECK 174 dropped no pair)")
row("tex5:256", "X's half is complete (seeds 580-599, 84 runs, 29.2 h; both perfect gates 2/2; CHECK 174 dropped no pair)",
    f"seeds {min(SEEDS20)}-{max(SEEDS20)}, {len(R)} runs; CEIL 2/2 and 2/2; dropped D2 {c2['dropped']}, D4 {c4['dropped']}",
    "MATCH", "29.2 h: " + wall_note)
tex_has(256, "bound routed, DELTA\\_SPLIT $19/20$, DELTA\\_SLOW $19/20$ (the same outcome on every seed, both failing only on one, a merge), HEBB\\_SPLIT $16/20$, HEBB\\_SLOW $7/20$")
row("tex5:256", "BR: DELTA_SPLIT 19/20, DELTA_SLOW 19/20 (same outcome every seed, both failing on one, a merge), HEBB_SPLIT "
    "16/20, HEBB_SLOW 7/20", ", ".join(f"{a} {cnt[a]}/20" for a in ARMS[2:]) + f"; differing seeds {same_outcome or 'none'}; "
    f"MERGED {merged['DELTA_SPLIT']}", "MATCH")
tex_has(256, "D1 $d = -3$ (delta ahead by three pairs; rate difference $+0.15$, 95\\% $[-0.08, +0.37]$)")
row("tex5:256", "D1 d = -3 (delta ahead by three pairs; +0.15, 95% [-0.08, +0.37])",
    f"d = {d_val} ({len(b1_)} vs {len(c1)}); {nc[0]:+.3f} [{nc[1]:+.3f}, {nc[2]:+.3f}]", "MATCH",
    "Rounded from the printed +0.150 [-0.082, +0.374]")
tex_has(256, "D2 0 vs 0, the delta no-trigger arm leaving no room")
row("tex5:256", "D2 0 vs 0, the delta no-trigger arm leaving no room",
    f"0 vs 0, NOT SHOWN; DELTA_SLOW {cnt['DELTA_SLOW']}/20: on X at most 1 split-only pair was possible (best p = 0.5). "
    f"The test printed NOT SHOWN and no NO ROOM: the rule is pooled (>= 36/40) and print_claims applies it only pooled",
    "PARTIAL", "True as a description, but 'no room' is a pre-registered label, which X alone cannot give; it becomes "
    f"the label only if L's DELTA_SLOW >= {36 - cnt['DELTA_SLOW']}/20. Commit 902c39a's title says 'D2 no room'")
tex_has(256, "D4 SHOWN, 10 vs 1, $p = 0.0059$")
row("tex5:256", "D4 SHOWN, 10 vs 1, p = 0.0059", f"{res['D4'][0]} vs {res['D4'][1]}, p = {res['D4'][2]:.4f}", "MATCH")
tex_has(256, "D3 delta sooner on 10 of 15 shared binds, later on 4, sign test $p = 0.09$")
row("tex5:256", "D3 delta sooner on 10 of 15 shared binds, later on 4, p = 0.09",
    f"{pos} of {len(d3)}, later {neg}, tie {len(d3) - pos - neg}; p = {p3:.4f}", "MATCH",
    "Record commit 8ae660d's message says 'earlier on 10 of 14' (14 = the untied); the draft follows the RESULT")
tex_has(256, "every Hebbian firing on a labelled target (39 of 39), the delta firings on 10 of 31")
row("tex5:256", "every Hebbian firing on a labelled target (39 of 39), the delta firings on 10 of 31",
    f"{fire['HEBB_SPLIT'][1]}/{fire['HEBB_SPLIT'][0]}; {fire['DELTA_SPLIT'][1]}/{fire['DELTA_SPLIT'][0]}", "MATCH")
tex_has(256, "the probe blocked 3 Hebbian and 5 delta checks alone")
row("tex5:256", "the probe blocked 3 Hebbian and 5 delta checks alone",
    f"blocked by the probe alone: HEBB_SPLIT {pb['HEBB_SPLIT'][2]}, DELTA_SPLIT {pb['DELTA_SPLIT'][2]}", "MATCH",
    f"The other count (probe >= 0.95 while held-out < 0.95: 1 and 1) is not given")
tex_has(256, "on delta channels at eight streams the split did not change a single seed's outcome, because the no-trigger arm already binds routed $19/20$ where its Hebbian counterpart binds $7/20$")
row("tex5:256", "on delta channels the split did not change a single seed's outcome, because the no-trigger arm already "
    "binds 19/20 where its Hebbian counterpart binds 7/20",
    f"DELTA pairs: outcomes identical on all 20 seeds; DELTA_SLOW {cnt['DELTA_SLOW']}/20, HEBB_SLOW {cnt['HEBB_SLOW']}/20; "
    f"DELTA_SPLIT fired in {fire['DELTA_SPLIT'][2]} of 20 runs ({fire['DELTA_SPLIT'][0]} firings)", "PARTIAL",
    "Counts true; 'because' is the draft's inference (labelled a description pending L). The split fired 31 times in "
    "delta runs, 21 of them off-target, and changed no outcome; the 19/20 bounds the room, it does not show the cause")
tex_has(256, "the exact-copy split on Hebbian channels, $16/20$ here, sits below the noisy copy's $18/20$ of \\texttt{test\\_split\\_copy} on other seeds")
row("tex5:256", "the exact-copy split on Hebbian channels, 16/20 here, sits below the noisy copy's 18/20 of test_split_copy "
    "on other seeds", f"HEBB_SPLIT {cnt['HEBB_SPLIT']}/20 (580-599); split_copy WIN3_SPLIT_D8 X {sc_br['X']}/10 + L "
    f"{sc_br['L']}/10 = {sc_br['X'] + sc_br['L']}/20 (seeds 560-569, 1560-1569)", "MATCH",
    "split_copy's split also zeroes Adam's state (copy + noise + reset); the exact copy here does not, so two things differ")
tex_has(256, "a difference two screens of ten seeds had not seen")
row("tex5:256", "a difference two screens of ten seeds had not seen", "other sessions' screens (H's S79 and the like); "
    "not in this record", "NOT RECORD-CHECKABLE", "")
tex_has(256, "running on X, where a second amendment confined to the report's rules")
row("tex5:256", "'running on X, where a second amendment ... was committed during the run' (the sentence before the "
    "X results)", "historical at the amendment's time; the run is complete by the same paragraph", "MATCH",
    "Reads as the time of the amendment; tense mixes with 'X's half is complete'")
tex_has(476, "\\texttt{test\\_delta\\_recipe}'s L half (X's is complete)")
row("tex5:476", "next: test_delta_recipe's L half (X's is complete)", f"X {len(R)}/84; no L file", "MATCH")
tex_has(496, "test \\texttt{a581027}, Amendment~2 during the run \\texttt{55dac50}, \\texttt{2225f00} / \\texttt{8ae660d} (X) & X complete; L to run & 84; ---")
row("tex5:496", "provenance: spec 05256c5, A1 32c121b / test a581027, A2 55dac50, 2225f00 / 8ae660d (X); X complete; L to run; 84",
    f"meta.git {meta['git']}; record commit {REC}; {len(R)} records; 55dac50 and 2225f00 committed during the run "
    f"(between 05:27:31 and >= {end_min:%m-%d %H:%M})", "MATCH")
tex_has(499, "X's are UTC")
row("tex5:499", "caption: X's recorded start times are UTC", f"meta.started {meta['started']}: UTC is the only zone consistent "
    f"with the 02:12 start and 29.2 h (other zones shift the end by whole hours, §3)", "MATCH",
    "Inferred, not recorded: X's times carry no zone")
tex_has(36, "so the memory rule is open again and is the next main-line test, specified and reviewed")
row("tex5:36", "abstract: the memory rule ... is the next main-line test, specified and reviewed", "X's half complete at "
    f"{REC} (07:25 UTC 12 Oct), before the draft's X-half commit {DRAFT_X} (07:28)", "PARTIAL",
    "Stale: the abstract cites no X result; the test has run on X and its X results are in tex5:54 and 256")
tex_has(472, "The write rule is open: with its write strength fixed, the delta rule is not shown worse than the Hebbian one on screens at every size tried.")
row("tex5:472", "conclusion: the write rule is open ... on screens", "cites no X result; X's D1 (d = -3) does not contradict it",
    "MATCH", "No mention that the pre-registered test has run on X")
tex_has(360, "A pre-registered test is specified.")
row("tex5:360", "assessment: 'A pre-registered test is specified.'", f"it has run on X ({REC})", "PARTIAL",
    "Stale since 902c39a; not an X number")

say("")
say(f"{'line':<10} {'status':<22} statement  ||  recomputed")
for ln, st, rc, status, note in rows:
    say(f"{ln:<10} {status:<22} {st[:90]}  ||  {rc[:150]}")
c = collections.Counter(r[3] for r in rows)
say(f"\ncounts over {len(rows)} rows: " + ", ".join(f"{k} {c[k]}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")))
