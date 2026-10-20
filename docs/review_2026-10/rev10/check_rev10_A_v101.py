#!/usr/bin/env python3
"""check_rev10_A_v101.py: Referee V's check_rev10_A re-run against Revision 10.1 (main line 846d0c8,
multichannel_hebbian_report_v10.tex, 596 lines, cited tex101:NNN). The original, check_rev10_A.py, read Revision 10
(585b666, 591 lines, tex10:NNN). Scope as the original: every count, pair, p-value, interval, band, reading and
factual/process statement concerning test_delta_recipe (X, L, pooled; Amendment 2), I's GPU run of test_window_gate
(report 3), R's Phase 5, S75 (with the tenth WIDE_SLOW seed, 839) and S75b, the referee/review history, the provenance
rows and STATUS-type statements for these, and the abstract/conclusion/limits sentences on them; plus every line of
that scope that 846d0c8 changed or added (rows 83 on).

    python3 -I docs/review_2026-10/rev10/check_rev10_A_v101.py          (from the repository root, after `git fetch`)

What changed against the original:
  * The tex is read at 846d0c8. Every row is anchored twice: its fragment on its tex10 line at 585b666 (as the
    original asserted it) and on the line that line became at 846d0c8 (difflib's line map; asserted). A fragment that
    846d0c8 rewrote is re-anchored on its new wording (L(..., new=...)); its row is then judged against the new words.
    Extra lines a row cites are anchored too (also()), each on a fragment of its own.
  * Branches are pinned to the commits the original read (its run, 19 Oct 17:52-17:59 UTC; the reflog of this clone's
    remote-tracking refs): main 7c4ceb8, explore-I 8d2e89f, replicate-R 31c39fc, review-V 55d2116, rev9-draft 010a085.
    So newer pushes do not change any row. Section 7 lists, for each pinned branch, the commits after its pin up to
    846d0c8's commit time (2026-10-19 18:33:26 UTC).
  * Every record recomputation is kept as it was. Each row carries the original's status (read from
    check_rev10_A.md at 8bc4ebd, matched by row number and statement) and the status against 846d0c8.

Read-only: every file is read with `git show REF:PATH`; nothing is trained, no session code is run and no test is
executed. The main line's outcome and report functions (test_delta_recipe's success/tally/report, test_window_gate's
success/tally, the helpers they call) and R's r_transition / r_bound_routed (docs/review_2026-10/replication_check.py)
are cut out of their files by ast and exec'd in a private namespace, as the referee's earlier scripts do; rlib.py gives
the independent re-implementation. Prints the row table (markdown) and the counts; writes nothing.
"""
import ast
import collections
import contextlib
import datetime as dt
import difflib
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
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
sys.path.insert(0, REV)
sys.dont_write_bytecode = True
import rlib  # noqa: E402

V10 = "585b666"                       # Revision 10's merge (what the original read)
V101 = "846d0c8"                      # Revision 10.1 (this check)
TEX = "multichannel_hebbian_report_v10.tex"
# pins: the commits the original read (remote-tracking refs at its run, 19 Oct 17:52-17:59 UTC)
PIN = {"origin/claude/bdh-growth-hebbian-inference-w90069": "7c4ceb8", "origin/claude/explore-I": "8d2e89f",
       "origin/claude/replicate-R": "31c39fc", "origin/claude/review-V": "55d2116", "origin/claude/rev9-draft": "010a085"}
MAIN = PIN["origin/claude/bdh-growth-hebbian-inference-w90069"]
I_BR, R_BR, V_BR = PIN["origin/claude/explore-I"], PIN["origin/claude/replicate-R"], PIN["origin/claude/review-V"]
ORIG_MD = ("8bc4ebd", "docs/review_2026-10/rev10/check_rev10_A.md")    # the original's output, as committed
ROWS = []


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def ctime(c):
    return dt.datetime.fromisoformat(git("show", "-s", "--format=%cI", c).strip()).astimezone(dt.timezone.utc)


def hm(c):
    return ctime(c).strftime("%d %b %H:%M:%S UTC")


OLD = show(V10, TEX).splitlines()
tex = show(V101, TEX).splitlines()


def line_map(a, b):
    """old line -> new line (1-based), from difflib's opcodes; equal-sized replaced blocks map line for line."""
    m, ops = {}, difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op in ("equal", "replace") and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
    return m, [o for o in ops if o[0] != "equal"]


LMAP, LOPS = line_map(OLD, tex)
LMAP.update({44: 44, 536: 541, 537: 542, 538: 543})     # the two unequal blocks: 44 -> 44-46 (46 new), 536-538 -> 540-543 (540 new)
CHANGED = {i + 1 for op, i1, i2, _, _ in LOPS for i in range(i1, i2)}
PENDING = []


def L(frag, line, new=None, nline=None):
    """Assert the verbatim fragment on tex10:line (585b666) and it (or `new`, its 846d0c8 wording) on the line it became;
    return 'tex101:N'."""
    assert frag in OLD[line - 1], f"fragment not on tex10:{line}: {frag!r}"
    n = nline or LMAP[line]
    f = frag if new is None else new
    assert f in tex[n - 1], f"fragment not on tex101:{n} (tex10:{line}): {f!r}"
    PENDING.append((line, n, new is not None))
    return f"tex101:{n}"


def N(frag, n):
    """A fragment of text 846d0c8 added or rewrote, on tex101:n (no tex10 counterpart is asserted)."""
    assert frag in tex[n - 1], f"fragment not on tex101:{n}: {frag!r}"
    PENDING.append((None, n, True))
    return f"tex101:{n}"


def also(*pairs):
    """Extra lines a row cites: each (tex10 line, fragment[, 846d0c8 wording]) asserted on both; returns ', N, ...'."""
    out = []
    for p in pairs:
        line, frag = p[0], p[1]
        assert frag in OLD[line - 1], f"extra fragment not on tex10:{line}: {frag!r}"
        n = LMAP[line]
        f = p[2] if len(p) > 2 else frag
        assert f in tex[n - 1], f"extra fragment not on tex101:{n}: {f!r}"
        PENDING.append((line, n, len(p) > 2))
        out.append(str(n))
    return ", " + ", ".join(out)


def _old_rows():
    md = show(*ORIG_MD)
    body = md[md.index("## All rows"):]
    out = {}
    for ln_ in body.split("\n"):
        m = re.match(r"^\| (\d+) \| (.*?) \| (.*?) \| (.*) \| (?:\*\*)?(MATCH|PARTIAL|DISAGREE|NOT RECORD-CHECKABLE)(?:\*\*)? \| ", ln_)
        if m:
            out[int(m.group(1))] = (m.group(3), m.group(5))
    return out


OLDROWS = _old_rows()
assert len(OLDROWS) == 82, len(OLDROWS)
NEW_FROM = 83          # rows from here on are new (text 846d0c8 changed or added)


def row(where, stmt, source, recomputed, status, note=""):
    assert status in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")
    i = len(ROWS) + 1
    anchors = list(PENDING)
    PENDING.clear()
    reanch = [f"tex10:{a}" for a, b, ch in anchors if a is not None and ch]
    touched = [f"tex10:{a}" for a, b, ch in anchors if a is not None and a in CHANGED and not ch]
    if i < NEW_FROM:
        ostmt, ost = OLDROWS[i]
        assert ostmt == str(stmt).replace("|", "\\|").replace("\n", " "), (i, ostmt, stmt)
        delta = f"{ost} (unchanged)" if ost == status else f"{ost} \u2192 {status}"
        if reanch:
            delta += "; re-anchored (" + ", ".join(reanch) + " rewritten)"
        elif touched:
            delta += "; line changed elsewhere (" + ", ".join(touched) + "), fragment intact"
    else:
        ost, delta = "\u2014", f"new: {status}"
    ROWS.append((where, stmt, source, recomputed, status, note, ost, delta))


def say(*a):
    print(" ".join(str(x) for x in a))


def segments(source):
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
    assert not missing, missing
    return ns


def module(ref, path, names, **inject):
    ns = dict(math=math, statistics=statistics, collections=collections, json=json, os=os, **inject)
    extract(show(ref, path), names, ns)
    return types.SimpleNamespace(**ns)


def binom_pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_ge(n, k, p):
    return sum(binom_pmf(n, i, p) for i in range(k, n + 1))


say("=" * 100)
say("0. Pins")
say("=" * 100)
say(f"  main line pinned at {MAIN} (the original's read); Revision 10 merge {V10} ({hm(V10)}), {len(OLD)} lines; Revision 10.1 "
    f"{V101} ({hm(V101)}): {TEX}: {len(tex)} lines")
say(f"  explore-I pinned {I_BR}; replicate-R pinned {R_BR}; review-V pinned {V_BR}")
say(f"  tex10 -> tex101 changed blocks: {[(op, f'{i1 + 1}-{i2}', f'{j1 + 1}-{j2}') for op, i1, i2, j1, j2 in LOPS]}")
assert len(OLD) == 591 and len(tex) == 596

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("1. test_delta_recipe: X (8ae660d) and L (b03d353) records; the test's own functions and rlib")
say("=" * 100)
REC_X, REC_L = "8ae660d", "b03d353"
XRES, LRES = "results/X/delta_recipe_results.json", "results/L/delta_recipe_results.json"
storeX, storeL = json.loads(show(REC_X, XRES)), json.loads(show(REC_L, LRES))
assert show(V10, XRES) == show(REC_X, XRES) and show(V10, LRES) == show(REC_L, LRES)
TST = show(V10, "test_delta_recipe.py")
# the test's report functions (code at V10 = 8ae660d's in code; only the docstring changed since)
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
tgw.MAX_C = 43200
REPORT_NAMES = {"MACHINES", "RANGES", "ITERS", "VALID_N", "D1_BOUND", "BETA", "OVERLAP_MAX", "ARMS", "ARM", "KEYS",
                "ARMS_RUN", "CEILS", "PARTS", "NO_ROOM", "D1_LABELS", "SPLIT_READINGS", "D1_OC", "seeds_for", "SEEDS",
                "outcome2", "success", "overlap", "paired", "sign_test", "d3_rows", "d3_of", "validity", "tally",
                "paired_full", "newcombe_paired", "d1_label", "split_verdict", "split_label", "split_reading", "pool2",
                "band_str", "diff_str", "print_claims", "broken_pairs", "check_174", "probe_blind", "print_bands",
                "complete", "stream_str", "raw_rows", "report", "other_of"}
STORES = {XRES: storeX, LRES: storeL, "delta_recipe_results.json": None}
T = dict(math=math, statistics=statistics, collections=collections, tgw=tgw, tsr=tsr, tscur=tscur, tss=tss, fc2=fc2,
         ok_r=tss.ok_r, get=trd.get, bound_r=tcl.bound_r, mcnemar_greater=tcl.mcnemar_greater, med_int=tsch.med_int,
         mstr=tsch.mstr, shared_max=tsch.shared_max, fmt_step=tbo.fmt_step, EVAL_EVERY=tbo.EVAL_EVERY,
         groups_str=trs.groups_str, curve_str=tss.curve_str)
extract(TST, {n for n in REPORT_NAMES if n in segments(TST)}, T)
T["load_store"] = lambda p: STORES[p]


def d1lab(d):
    return T["d1_label"](dict(valid=True, n=40, b=0, c=d))


SEEDS = {"X": {k: tuple(v) for k, v in storeX["meta"]["seeds"].items()},
         "L": {k: tuple(v) for k, v in storeL["meta"]["seeds"].items()}}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    T["report"](storeL, SEEDS["L"], "L", 0.0, "delta_recipe_results.json", XRES)
REP = buf.getvalue().split("\n")
lrep = show(REC_L, "results/L/delta_recipe_L_report.log").rstrip("\n").split("\n")
while REP and REP[-1] == "":
    REP.pop()
say(f"  the test's report() with --also X, regenerated: {len(REP)} lines; results/L/delta_recipe_L_report.log {len(lrep)} "
    f"lines; lines that differ: {sum(1 for a, b in zip(REP, lrep) if a != b) + abs(len(REP) - len(lrep))}")
i0 = next(i for i, l in enumerate(REP) if l.startswith("POOLED (PRIMARY)"))
i1 = next(i for i, l in enumerate(REP) if l.startswith("PER MACHINE (SECONDARY)"))
POOLED_TXT = "\n".join(REP[i0:i1])
for l in REP[i0:i1]:
    say("   |" + l[:190])

# independent (rlib)
ARMS = ["CEIL_HEBB", "CEIL_DELTA", "HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW"]


def analyze(store, lo, mname):
    S20 = list(range(lo, lo + 20))
    runs = {a: rlib.runs(store, a) for a in ARMS}
    br = {a: {s: (rlib.bound(r) if a.startswith("CEIL") else rlib.bound_routed(r)) for s, r in runs[a].items()} for a in ARMS}
    agree = all(br[a][s] == T["success"](a, r) for a in ARMS for s, r in runs[a].items())
    out = dict(S=S20, runs=runs, br=br, agree=agree, cnt={a: sum(br[a].values()) for a in ARMS})
    out["oc"] = {a: {s: rlib.outcome_k(runs[a][s]) for s in S20 if not br[a][s]} for a in ARMS[2:]}

    def disc(x, y):
        return ([s for s in S20 if br[x][s] and not br[y][s]], [s for s in S20 if br[y][s] and not br[x][s]])
    out["D1"] = disc("DELTA_SPLIT", "HEBB_SPLIT")
    out["D2"] = disc("DELTA_SPLIT", "DELTA_SLOW")
    out["D4"] = disc("HEBB_SPLIT", "HEBB_SLOW")
    out["d3"] = [(s, runs["HEBB_SPLIT"][s]["transition"], runs["DELTA_SPLIT"][s]["transition"]) for s in S20
                 if br["HEBB_SPLIT"][s] and br["DELTA_SPLIT"][s]]
    fire = {}
    for a in ("HEBB_SPLIT", "DELTA_SPLIT"):
        fs = [c for s in S20 for c in runs[a][s]["checks"] if c["fired"]]
        fire[a] = (len(fs), sum(1 for c in fs if c["key_ok"]))
    out["fire"] = fire
    # probe alone (Amendment 2 (f) form, as check_delta_recipe_L.py's pblind)
    alone = {}
    for a in ("HEBB_SPLIT", "DELTA_SPLIT"):
        n_alone = 0
        for s in S20:
            n, last = 0, None
            for c in runs[a][s]["checks"]:
                if c["step"] >= 4800 and (c["prev"] is not None and c["acc"] - c["prev"] < 0.02 and n < 3
                                          and (last is None or c["step"] - last >= 4800) and c["acc"] >= 0.95 and not c["fired"]):
                    n_alone += 1
                if c["fired"]:
                    n, last = n + 1, c["step"]
        alone[a] = n_alone
    out["alone"] = alone
    say(f"  {mname}: rlib = the test's success() on all 84: {agree}; counts {out['cnt']}; failures {out['oc']}")
    say(f"     D1 delta only {out['D1'][0]}, Hebbian only {out['D1'][1]}; D2 split only {out['D2'][0]}, slow only "
        f"{out['D2'][1]}; D4 split only {len(out['D4'][0])}, slow only {out['D4'][1]}; firings {fire}; probe alone {alone}")
    return out


AX, AL = analyze(storeX, 580, "X"), analyze(storeL, 1580, "L")
cntP = {a: AX["cnt"][a] + AL["cnt"][a] for a in ARMS}
b1, c1 = len(AX["D1"][0]) + len(AL["D1"][0]), len(AX["D1"][1]) + len(AL["D1"][1])
b2, c2 = len(AX["D2"][0]) + len(AL["D2"][0]), len(AX["D2"][1]) + len(AL["D2"][1])
b4, c4 = len(AX["D4"][0]) + len(AL["D4"][0]), len(AX["D4"][1]) + len(AL["D4"][1])
p2, p4 = rlib.mcnemar1(b2, c2), rlib.mcnemar1(b4, c4)
p2two = rlib.mcnemar2(b2, c2)
d3 = AX["d3"] + AL["d3"]
e3, l3 = sum(1 for _, h, d in d3 if d < h), sum(1 for _, h, d in d3 if d > h)
p3 = rlib.binom_tail(e3 + l3, e3)
md, mh = statistics.median([d for *_, d in d3]), statistics.median([h for _, h, _ in d3])
both = sum(1 for A in (AX, AL) for s in A["S"] if A["br"]["DELTA_SPLIT"][s] and A["br"]["HEBB_SPLIT"][s])
neither = 40 - both - b1 - c1
nc = T["newcombe_paired"](both, b1, c1, neither)
dX, dL = len(AX["D1"][1]) - len(AX["D1"][0]), len(AL["D1"][1]) - len(AL["D1"][0])
bands = {a: (rlib.band(cntP[a], 40), rlib.wilson(cntP[a], 40)) for a in ARMS[2:]}
fireP = {a: (AX["fire"][a][0] + AL["fire"][a][0], AX["fire"][a][1] + AL["fire"][a][1]) for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
aloneP = {a: AX["alone"][a] + AL["alone"][a] for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
valid = {m: (A["cnt"]["CEIL_HEBB"], A["cnt"]["CEIL_DELTA"]) for m, A in (("X", AX), ("L", AL))}
say(f"  pooled: counts {cntP}; D1 {b1} vs {c1} (X {dX:+d}, L {dL:+d}); Newcombe {nc}; D2 {b2} vs {c2} p {p2:.4g} "
    f"(two-sided {p2two:.4g}); D4 {b4} vs {c4} p {p4:.4g}; D3 {len(d3)} shared, earlier {e3}, later {l3}, tied "
    f"{len(d3) - e3 - l3}, p {p3:.6f}, medians {md} vs {mh}; bands {bands}; firings {fireP}; probe alone {aloneP}; "
    f"validity {valid}")
cost3 = sorted(AL["D2"][1])
cost3_oc = {s: rlib.outcome_k(AL["runs"]["DELTA_SPLIT"][s]) for s in cost3}
say(f"  delta seeds bound by DELTA_SLOW and not DELTA_SPLIT: X {AX['D2'][1]}, L {cost3} {cost3_oc}")
# D1's printed operating characteristic, and the same bound at this test's observed Hebbian rate
D1_OC = T["D1_OC"]


def p_d_le2(ph, pd, nested):
    """P(c - b <= 2) over 40 pairs: c = Hebbian only, b = delta only."""
    if nested:     # delta binds only where Hebbian binds
        pc, pb = ph - pd, 0.0
    else:
        pc, pb = ph * (1 - pd), pd * (1 - ph)
    pn = 1 - pc - pb
    tot = 0.0
    for c in range(41):
        for b in range(41 - c):
            if c - b <= 2:
                tot += math.factorial(40) / (math.factorial(c) * math.factorial(b) * math.factorial(40 - b - c)) \
                    * pc ** c * pb ** b * pn ** (40 - b - c)
    return tot


oc95 = (p_d_le2(0.95, 0.85, False), p_d_le2(0.95, 0.85, True))
oc875 = (p_d_le2(0.875, 0.775, False), p_d_le2(0.875, 0.775, True))
say(f"  D1_OC printed: {D1_OC!r}; recomputed at 0.95 vs 0.85: independent {oc95[0]:.3f}, nested {oc95[1]:.3f}; at this "
    f"test's pooled Hebbian rate 0.875 vs 0.775: {oc875[0]:.3f} / {oc875[1]:.3f}")
# L's timings, from the log
LLOG = show(REC_L, "results/L/delta_recipe_L.log").split("\n")
hdr = [l for l in LLOG if l.startswith("=== longrun delta_recipe_L: start")]
wall_line = next(l for l in LLOG if "total wall clock after verification" in l)
wall_min = float(re.search(r"verification: ([\d.]+) min", wall_line).group(1))
run_re = re.compile(r"   (\w+)\s+seed (\d+)  (.*?)  (\d+)s  \(elapsed ([\d.]+)m\)$")
last_start = max(i for i, l in enumerate(LLOG) if l.startswith("=== longrun"))
runl = [run_re.match(l).groups() for l in LLOG[last_start:] if run_re.match(l)]
first_sub = min(float(el) - int(secs) / 60 for k, s, _, secs, el in runl[:4])
runs_only = wall_min - first_sub
p1 = LLOG[:last_start]
say(f"  L log headers: {hdr}; wall after verification {wall_min} min; first submission ~{first_sub:.1f} min after "
    f"meta.started -> runs only {runs_only:.1f} min = {runs_only / 60:.2f} h; first start: PROJECTION heading "
    f"{any(l.startswith('PROJECTION') for l in p1)}, run lines {sum(1 for l in p1 if run_re.match(l))}; "
    f"meta workers {storeL['meta']['workers']}, torch L {storeL['meta']['torch']}, X {storeX['meta']['torch']}; "
    f"X cpu {storeX['meta']['cpu']}; L records {len(storeL['runs'])}, X {len(storeX['runs'])}")
boots = [re.search(r"boot (\w+)", h).group(1) for h in hdr]
xlog = show(MAIN, "results/X/delta_recipe_X.log").split("\n")
xh = next(l for l in xlog if l.startswith("=== longrun"))
xwall = float(re.search(r"verification: ([\d.]+) min", next(l for l in xlog if "total wall clock" in l)).group(1))
xstart = dt.datetime.strptime(re.search(r"start 1 at (\S+ [\d:]+)", xh).group(1), "%Y-%m-%d %H:%M:%S")
xend = dt.datetime.strptime(storeX["meta"]["started"], "%Y-%m-%d %H:%M:%S") + dt.timedelta(minutes=xwall)
xh_h = (xend - xstart).total_seconds() / 3600
say(f"  X: {xh}; end {xend}; {xh_h:.2f} h")
# CHECKs passed
chk = {m: (sum(1 for l in lg if l.strip() == "ALL VERIFICATION CHECKS PASSED"),
           [l for l in lg if re.search(r"-> (FAIL|MISMATCH)", l) or "Traceback" in l])
       for m, lg in (("X", xlog), ("L", LLOG[last_start:]))}
c174 = {m: sorted(set(re.findall(r"CHECK 174 .*?(\d+) pairs, (PASS|FAIL)", "\n".join(lg)))) for m, lg in (("X", xlog), ("L", LLOG[last_start:]))}
say(f"  'ALL VERIFICATION CHECKS PASSED' and failing lines: {chk}; CHECK 174 lines: {c174}")
say(f"  L's RESULT block at {V10}: 'OOM killer' {'OOM killer' in TST}; 'GPU job beside it' {'GPU job beside it' in TST}")

ln = 37
row(L("binds eight streams $35/40$ against the Hebbian recipe's $35/40$ ($d = 0$", ln) + also((60, "($35/40$ against $35/40$)"), (301, "the delta recipe $35/40$ against the Hebbian recipe $35/40$"), (336, "$35/40$ against $35/40$, $d = 0$"), (377, "binds $35/40$ against the Hebbian recipe's $35/40$ ($d = 0$"), (495, "($35/40$ against $35/40$)")),
    "pooled over 40 pairs, delta 35/40 against Hebbian 35/40, d = 0, 'delta not shown worse by more than 2 pairs'",
    "X 8ae660d + L b03d353 records; test_delta_recipe report() --also; rlib",
    f"DELTA_SPLIT {cntP['DELTA_SPLIT']}/40, HEBB_SPLIT {cntP['HEBB_SPLIT']}/40; {b1} vs {c1}; d = {c1 - b1}; label "
    f"'{d1lab(c1 - b1)}'", "MATCH" if (cntP['DELTA_SPLIT'], cntP['HEBB_SPLIT'], b1, c1) == (35, 35, 5, 5) else "DISAGREE")
row(L("without a trigger $38/40$; the split's claim on delta NOT SHOWN, with no room", ln) + also((60, "no room (delta without a trigger $38/40$)"), (264, "DELTA\\_SLOW $38/40$, 0 vs 3, $p = 1$, NOT SHOWN"), (301, "without a trigger, $38/40$"), (336, "``no room'': $38/40$ without a trigger"), (377, "``no room'' ($38/40$ without a trigger)"), (495, "$38/40$ without a trigger")),
    "delta without a trigger 38/40; D2 NOT SHOWN, no room", "records; report()",
    f"DELTA_SLOW {cntP['DELTA_SLOW']}/40 (>= 36 -> no room); D2 {b2} vs {c2}, p = {p2:.3g}",
    "MATCH" if (cntP['DELTA_SLOW'], b2, c2) == (38, 0, 3) and p2 == 1.0 else "DISAGREE")
row(L("the Hebbian write needs the split ($35/40$ against $17/40$, 19 vs 1, $p = 2\\times10^{-5}$)", ln) + also((264, "19 vs 1, $p = 2.0\\times10^{-5}$, SHOWN"), (301, "against $17/40$"), (336, "($35/40$ against $17/40$)"), (377, "``helps'' (19 vs 1, $p = 2\\times10^{-5}$)"), (495, "against $17/40$")),
    "Hebbian 35/40 against 17/40, 19 vs 1, p = 2e-5", "records; rlib.mcnemar1",
    f"HEBB_SLOW {cntP['HEBB_SLOW']}/40; D4 {b4} vs {c4}; p = {p4:.4g} (21/2^20)",
    "MATCH" if (cntP['HEBB_SLOW'], b4, c4) == (17, 19, 1) and round(p4, 6) == 2.0e-5 else "DISAGREE")
row(L("delta binds sooner on 24 of the 30 seeds both bind", ln) + also((60, "D3 delta sooner on 24 of 30"), (264, "delta's transition is earlier on 24"), (377, "delta binds sooner on 24 of the 30 seeds both bind")),
    "delta sooner on 24 of the 30 shared binds", "records", f"{len(d3)} shared; earlier {e3}, later {l3}, tied {len(d3) - e3 - l3}",
    "MATCH" if (len(d3), e3, l3) == (30, 24, 5) else "DISAGREE")
row(L("the readings ``delta: no room; Hebbian: helps'' apply", ln) + also((60, "``delta: no room; Hebbian: helps''"), (264, "``delta: no room; Hebbian: helps''")),
    "readings 'delta: no room; Hebbian: helps'", "report() pooled block",
    f"in the regenerated pooled block: {'delta: no room; Hebbian: helps' in POOLED_TXT}", "MATCH" if 'delta: no room; Hebbian: helps' in POOLED_TXT else "DISAGREE")
ln = 264
row(L("X (seeds 580--599, 84 runs, 29.2 h, Intel Xeon at 2.10\\,GHz)", ln), "X: seeds 580-599, 84 runs, 29.2 h, Xeon 2.10 GHz",
    "X records, results/X/delta_recipe_X.log", f"{len(storeX['runs'])} records; {xh_h:.2f} h start to last run; cpu {storeX['meta']['cpu']}",
    "MATCH" if len(storeX['runs']) == 84 and round(xh_h, 1) == 29.2 and "2.10GHz" in storeX['meta']['cpu'] else "DISAGREE")
row(L("L (seeds 1580--1599, 84 runs, about 18.4 h of runs on five workers of the i7-12650H, PyTorch 2.14.1 against X's 2.14.0", ln),
    "L: seeds 1580-1599, 84 runs, about 18.4 h of runs on five workers, PyTorch 2.14.1 against X's 2.14.0",
    "L records and log", f"{len(storeL['runs'])} records; runs only {runs_only:.1f} min = {runs_only / 60:.2f} h (1125.8 min "
    f"after verification includes the ~19 min projection); workers {storeL['meta']['workers']}; torch L {storeL['meta']['torch']}, "
    f"X {storeX['meta']['torch']}", "MATCH", "X's build is 2.14.0+cu130; 'about 18.4' is 18.45 h rounded down")
row(L("a first start on 12 October was lost before any run to a reboot during the projection's timing, with a GPU job beside it", ln),
    "first start lost before any run to a reboot during the projection's timing, with a GPU job beside it",
    "L log headers; check_delta_recipe_L.md", f"headers {[h[34:70] for h in hdr]}; boot ids {boots}; first start: "
    f"PROJECTION heading only, {sum(1 for l in p1 if run_re.match(l))} run lines", "NOT RECORD-CHECKABLE",
    "the reboot (new boot id), the timing and 'before any run' are recorded; 'with a GPU job beside it' is the author's account "
    "(in no record; tex10:426 attributes it, this line states it as fact)")
row(L("every perfect gate bound $2/2$ on each machine, every CHECK passed, and CHECK 174 dropped no pair on either", ln),
    "perfect gates 2/2 on each machine; every CHECK passed; CHECK 174 dropped no pair", "records; logs",
    f"CEIL (HEBB, DELTA) {valid}; 'ALL VERIFICATION CHECKS PASSED' X {chk['X'][0]}x, L {chk['L'][0]}x, failing lines "
    f"{len(chk['X'][1]) + len(chk['L'][1])}; {c174}", "MATCH" if valid == {"X": (2, 2), "L": (2, 2)} else "DISAGREE")
row(L("on X,", ln) + " / on L", "X: DELTA_SPLIT 19, DELTA_SLOW 19, HEBB_SPLIT 16, HEBB_SLOW 7; L: 16 (three merges, one bound not routed), 19, 19, 10",
    "records (rlib = test success())", f"X {[AX['cnt'][a] for a in ('DELTA_SPLIT', 'DELTA_SLOW', 'HEBB_SPLIT', 'HEBB_SLOW')]}; "
    f"L {[AL['cnt'][a] for a in ('DELTA_SPLIT', 'DELTA_SLOW', 'HEBB_SPLIT', 'HEBB_SLOW')]}; L DELTA_SPLIT failures "
    f"{AL['oc']['DELTA_SPLIT']}",
    "MATCH" if [AX['cnt'][a] for a in ('DELTA_SPLIT', 'DELTA_SLOW', 'HEBB_SPLIT', 'HEBB_SLOW')] == [19, 19, 16, 7]
    and [AL['cnt'][a] for a in ('DELTA_SPLIT', 'DELTA_SLOW', 'HEBB_SPLIT', 'HEBB_SLOW')] == [16, 19, 19, 10] else "DISAGREE")
row(L("delta alone on 5 seeds and Hebbian alone on 5 (X 4 vs 1, L 1 vs 4), $d = 0$, rate difference $+0.000$ with 95\\% interval $[-0.162, +0.162]$", ln),
    "D1 5 vs 5 (X 4 vs 1, L 1 vs 4), +0.000 [-0.162, +0.162]", "records; the test's newcombe_paired",
    f"X {len(AX['D1'][0])} vs {len(AX['D1'][1])}, L {len(AL['D1'][0])} vs {len(AL['D1'][1])}; {nc[0]:+.3f} [{nc[1]:+.3f}, {nc[2]:+.3f}]",
    "MATCH" if (len(AX['D1'][0]), len(AX['D1'][1]), len(AL['D1'][0]), len(AL['D1'][1])) == (4, 1, 1, 4)
    and f"[{nc[1]:+.3f}, {nc[2]:+.3f}]" == "[-0.162, +0.162]" else "DISAGREE")
row(L("a one-sided bound that passes a true ten-point deficit 22--29\\% of the time at this size (printed beside it)", ln),
    "the bound passes a true ten-point deficit 22-29% of the time at this size (printed)", "test_delta_recipe.D1_OC; recomputed",
    f"printed: 0.29 / 0.22 at HEBB 0.95 vs DELTA 0.85; recomputed {oc95[0]:.3f} / {oc95[1]:.3f}; at the observed Hebbian "
    f"0.875 vs 0.775: {oc875[0]:.3f} / {oc875[1]:.3f}", "MATCH",
    "quotes the printed figure exactly; that figure assumes a Hebbian rate of 0.95, and 'at this size' means 40 pairs")
row(L("D2, DELTA\\_SPLIT $35/40$ against DELTA\\_SLOW $38/40$, 0 vs 3, $p = 1$, NOT SHOWN", ln), "D2 35/40 vs 38/40, 0 vs 3, p = 1",
    "records", f"{cntP['DELTA_SPLIT']} vs {cntP['DELTA_SLOW']}; {b2} vs {c2}; p {p2}", "MATCH" if (b2, c2) == (0, 3) else "DISAGREE")
row(L("D4, HEBB\\_SPLIT $35/40$ against HEBB\\_SLOW $17/40$, 19 vs 1, $p = 2.0\\times10^{-5}$, SHOWN", ln), "D4 19 vs 1, p = 2.0e-5",
    "records", f"{b4} vs {c4}; p {p4:.3g}", "MATCH" if (b4, c4) == (19, 1) else "DISAGREE")
row(L("delta's transition is earlier on 24, later on 5 and tied on 1 (medians 13{,}200 against 18{,}600; sign test $p = 0.00027$)", ln),
    "D3 24 / 5 / 1, medians 13,200 vs 18,600, p = 0.00027", "records",
    f"{e3} / {l3} / {len(d3) - e3 - l3}; medians {md:.0f} vs {mh:.0f}; p {p3:.5f}",
    "MATCH" if (e3, l3, md, mh, round(p3, 5)) == (24, 5, 13200, 18600, 0.00027) else "DISAGREE")
bstr = "; ".join(f"{a} {b} [{lo:.3f}, {hi:.3f}]" for a, (b, (lo, hi)) in bands.items())
row(L("Bands: DELTA\\_SLOW RELIABLE $[0.835, 0.986]$; DELTA\\_SPLIT and HEBB\\_SPLIT MAJORITY $[0.739, 0.945]$; HEBB\\_SLOW MINORITY $[0.285, 0.578]$", ln),
    "bands", "records; rlib.wilson/band", bstr,
    "MATCH" if bstr == "HEBB_SPLIT MAJORITY [0.739, 0.945]; HEBB_SLOW MINORITY [0.285, 0.578]; DELTA_SPLIT MAJORITY [0.739, 0.945]; DELTA_SLOW RELIABLE [0.835, 0.986]" else "DISAGREE")
row(L("The two machines mirrored each other on D1 (X $d = -3$, L $d = +3$)", ln) + ", 457",
    "X d = -3, L d = +3", "records", f"X {dX:+d}, L {dL:+d}", "MATCH" if (dX, dL) == (-3, 3) else "DISAGREE")
row(L("every Hebbian firing on a labelled target (80 of 80 over both machines), the delta firings on 26 of 63; the probe alone blocked 5 Hebbian and 14 delta checks", ln),
    "firings 80 of 80 Hebbian, 26 of 63 delta; probe alone 5 and 14", "records (checks[])",
    f"Hebbian {fireP['HEBB_SPLIT'][1]} of {fireP['HEBB_SPLIT'][0]}, delta {fireP['DELTA_SPLIT'][1]} of {fireP['DELTA_SPLIT'][0]}; "
    f"probe alone {aloneP['HEBB_SPLIT']} / {aloneP['DELTA_SPLIT']} (X {AX['alone']}, L {AL['alone']})",
    "MATCH" if fireP == {"HEBB_SPLIT": (80, 80), "DELTA_SPLIT": (63, 26)} and aloneP == {"HEBB_SPLIT": 5, "DELTA_SPLIT": 14} else "DISAGREE")
row(L("help on any seed and cost three (L's 1581, 1582 and 1591 bound routed without a trigger and not with one: two merges and one bound but unrouted; two-sided $p = 0.25$)", ln),
    "the split cost three delta seeds (1581, 1582, 1591: two merges, one bound unrouted), two-sided p = 0.25", "records",
    f"X {AX['D2'][1]}, L {cost3} {cost3_oc}; split only {b2}; two-sided p {p2two}",
    "MATCH" if cost3 == [1581, 1582, 1591] and not AX['D2'][1] and p2two == 0.25 else "DISAGREE")
row(L("binds $35/40$ here ($0.875$), against the noisy copy's $38/40$ of \\texttt{test\\_window\\_gate} and $18/20$", ln),
    "exact copy on Hebbian 35/40 (0.875); X 16/20, L 19/20", "records", f"{cntP['HEBB_SPLIT']}/40; X {AX['cnt']['HEBB_SPLIT']}, L {AL['cnt']['HEBB_SPLIT']}",
    "MATCH", "the 38/40 and 18/20 are the main line's earlier pooled results (round6_A3, check_I_report3, check_split_copy_L_v6)")
row(L("and found all of them to reproduce (\\texttt{docs/review\\_2026-10/check\\_delta\\_recipe\\_L.md}); one sentence of L's RESULT block was false", ln),
    "the referee recomputed every count ... of L's half and the pooled block and found all to reproduce; one sentence false, corrected",
    "check_delta_recipe_L.md; round6_A3.md; 76851eb", "check_delta_recipe_L.md: 66 rows, 55/7/1/3; the DISAGREE row is the 'printed its projection' sentence; "
    "76851eb rewrites it", "MATCH")
# 424 / 426 / 519 / 522
c76 = git("show", "76851eb", "--", "test_delta_recipe.py", "results/README.md")
date_marks = re.findall(r"^\+.*(?:19 October|2026-10-19|10-19|corrected)", c76, re.M)
row(L("Each is corrected where it stands, with the date.", 424),
    "the two statements (L's RESULT block; results/README.md's commit) each corrected where it stands, with the date",
    "76851eb (19 Oct 15:40 UTC) diff", f"both corrected in place at 76851eb; date or 'corrected' markers in the added lines: "
    f"{len(date_marks)}", "PARTIAL",
    "both are corrected where they stand (RESULT (L) now 'lost during the projection's timing'; README row now 8ae660d); neither "
    "corrected text carries a date or a correction mark: the only date is 76851eb's commit time")
row(L("on L (\\texttt{test\\_delta\\_recipe}'s first start, to a reboot that the author attributes to the OOM killer, with a GPU job beside it)", 426),
    "L's first start lost to a reboot the author attributes to the OOM killer, with a GPU job beside it", "RESULT (L) at 585b666; L log",
    f"RESULT (L) says so: {'OOM killer' in TST and 'GPU job beside it' in TST}; boot ids {boots}", "MATCH",
    "attributed to the author, as the record supports (check_delta_recipe_L.md: the cause is in no record)")
row(L("Revision~10 two", 431) + ", 455, 457", "Revision 10 added two lessons", "tex10 lessons section",
    f"new lesson paragraphs in the diff: {sum(1 for i in (455, 457) if tex[i - 1].startswith(chr(92) + 'paragraph'))} (tex10:455, 457)",
    "MATCH")
row(L("\\texttt{b03d353} (L and pooled; L's docstring corrected after the referee's check, \\texttt{76851eb})", 519),
    "provenance: b03d353 (L and pooled), docstring corrected 76851eb; L file recorded at 8ae660d; pooled block regenerated with --report --also; 84; 84",
    "git; L meta.git; L_report.log", f"b03d353 adds {LRES}: {LRES in git('show', '--name-only', '--format=', REC_L)}; "
    f"meta.git {storeL['meta']['git']}; report log regenerated = report(): {REP == lrep}; runs {len(storeX['runs'])}; {len(storeL['runs'])}",
    "MATCH" if storeL['meta']['git'] == "8ae660d" and REP == lrep else "DISAGREE")
row(L("and \\texttt{delta\\_recipe} runs 2.14.1)", 522), "caption: L's delta_recipe run on PyTorch 2.14.1",
    "L meta", f"meta.torch {storeL['meta']['torch']}", "MATCH" if storeL['meta']['torch'] == "2.14.1" else "DISAGREE",
    "the lost first start ran 2.14.0+cu130 (no record from it)")
row(L("Pool before reading the machines", 457), "either half alone would have read as a three-pair lead; pooled d = 0",
    "records; D1_LABELS", f"X d {dX:+d} -> '{d1lab(dX)}'; L d {dL:+d} -> '{d1lab(dL)}'; pooled 0", "MATCH")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("2. I's GPU run of test_window_gate (148c161; note 30807ec; report 3 at 3d7e68e)")
say("=" * 100)
GPU_F = "explore_out/I/gpu/mainline/test_window_gate_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
G = json.loads(show("148c161", GPU_F))
assert show(I_BR, GPU_F) == show("148c161", GPU_F)
NS = {"math": math, "statistics": statistics}
for path, names in [
        ("test_router_layout.py", {"MARGIN_PARTIAL", "ETA_SPLIT", "fail_class"}),
        ("test_scale_axes.py", {"fail_class_k"}),
        ("test_stream_channels.py", {"shared_max", "merged", "med_int"}),
        ("test_stream_recipe.py", {"ROUTED_ACC", "routed", "outcome", "wilson"}),
        ("test_conv_lr.py", {"bound_r", "mcnemar_greater"}),
        ("test_slow_start.py", {"ok_r", "runs_of"}),
        ("test_router_discovery.py", {"get"}),
        ("test_stream_curriculum.py", {"RELIABLE_F", "band_n"}),
        ("test_window_gate.py", {"RANGES", "VALID_N", "G3_BOUND", "RESET_N", "ARMS", "ARM", "KEYS", "ARMS_RUN", "CEILS",
                                 "OUTCOME", "CLAIMS", "G3", "PRINTED", "seeds_for", "success", "validity"})]:
    assert git("rev-parse", f"{MAIN}:{path}") == git("rev-parse", f"30807ec:{path}") == git("rev-parse", f"148c161:{path}"), path
    extract(show(MAIN, path), names, NS)
NS["tsr"] = types.SimpleNamespace(outcome=NS["outcome"], wilson=NS["wilson"])
GA = ("WIN3_SPLIT_D8", "WIN3_SLOW_D8", "HINGE_D8")
GS = range(2540, 2560)
gr = {a: rlib.runs(G, a) for a in GA + ("CEIL_C",)}
gsucc = {a: {s: NS["success"](a, NS["get"](G, a, s)) for s in GS} for a in GA}
grl = {a: {s: rlib.bound_routed(gr[a][s]) for s in GS} for a in GA}
assert gsucc == grl
gceil = sum(rlib.bound(r) for r in gr["CEIL_C"].values())
gc = {a: sum(gsucc[a].values()) for a in GA}
gp = {}
for a, b in (("WIN3_SPLIT_D8", "HINGE_D8"), ("WIN3_SPLIT_D8", "WIN3_SLOW_D8"), ("WIN3_SLOW_D8", "HINGE_D8")):
    x = sum(gsucc[a][s] and not gsucc[b][s] for s in GS)
    y = sum(gsucc[b][s] and not gsucc[a][s] for s in GS)
    gp[(a, b)] = (x, y, rlib.mcnemar1(x, y))
gtorch = sorted({str((r.get("device") or {}).get("torch") or r.get("torch")) for r in G["runs"].values()})
gsha = sorted({str(r.get("code_sha")) for r in G["runs"].values()})
say(f"  seeds {sorted(gr['WIN3_SPLIT_D8'])[0]}-{sorted(gr['WIN3_SPLIT_D8'])[-1]}; CEIL_C {sorted(gr['CEIL_C'])} bound {gceil}; "
    f"counts {gc} (test success = rlib on all 60); pairs {gp}; torch {gtorch}; meta torch "
    f"{G['meta'].get('torch')}; record key SHAs {gsha}")
rep3 = show("3d7e68e", "explore_out/I/report_3.md")
ln = 181
row(L("on twenty fresh seeds (2540--2559", ln) + ", 522", "seeds 2540-2559, fresh", "records; check_I_report3.md (fresh on every branch)",
    f"seeds {min(GS)}-{max(GS)} in the records", "MATCH", "freshness: check_I_report3.md / round6_A3 §3")
row(L("a note fixing the seeds, the arms and the rule that nothing would be read as a claim was committed before any run", ln),
    "the pre-run note (seeds, arms, no claim) committed before any run; records carry its code hash", "git; records; check_I_report3.md §1",
    f"note 30807ec {hm('30807ec')}; records' code_sha {gsha}; first commit with the records 148c161 {hm('148c161')}; meta.started "
    f"{G['meta']['started']} (machine clock)", "MATCH", "check_I_report3: the SHA's code first exists at 30807ec, so no record can precede the note")
row(L("The perfect gate bound $2/2$", ln), "perfect gate 2/2", "records", f"CEIL_C {gceil}/2", "MATCH" if gceil == 2 else "DISAGREE")
row(L("the recipe $\\mathbf{19/20}$, without the split $6/20$, the recurrent-gate recipe $0/20$", ln) + ", 37, 68, 495",
    "19/20, 6/20, 0/20", "records; test_window_gate.success (main line, blob = 30807ec = 148c161) and rlib",
    f"{gc}", "MATCH" if gc == {"WIN3_SPLIT_D8": 19, "WIN3_SLOW_D8": 6, "HINGE_D8": 0} else "DISAGREE")
pv = [gp[k][2] for k in gp]
row(L("paired, 19 vs 0, 13 vs 0 and 6 vs 0 (the referee's exact one-sided $p$: $1.9\\times10^{-6}$, $1.2\\times10^{-4}$ and $0.016$", ln),
    "pairs 19 vs 0, 13 vs 0, 6 vs 0; p 1.9e-6, 1.2e-4, 0.016", "records; rlib.mcnemar1",
    "; ".join(f"{a} vs {b}: {x} vs {y}, p {p:.2g}" for (a, b), (x, y, p) in gp.items()),
    "MATCH" if [(x, y) for x, y, _ in gp.values()] == [(19, 0), (13, 0), (6, 0)] and [f"{p:.2g}" for p in pv] == ["1.9e-06", "0.00012", "0.016"] else "DISAGREE")
row(L("report 3 states that the GPU run is never pooled with it", ln), "report 3 says the GPU run is never pooled; main line 38/40",
    "report_3.md at 3d7e68e", f"'never pooled' in report 3: {'never pooled' in rep3}; '38/40' in report 3: {'38/40' in rep3}", "MATCH")
row(L("64 of 65 rows match; the one that is not record-checkable is a statement about X's commit", ln),
    "the referee's check: 64 of 65 match; one not record-checkable (X's commit)", "check_I_report3.md; round6_A3.md",
    "round6_A3 table: 65 rows, 64 MATCH, 1 not record-checkable ('report 3's account of which commit X's runs used')", "MATCH")
row(L("byte for byte over 29 modules", ln), "the main line's code byte for byte over 29 modules; record code hash first exists at the note's commit",
    "check_I_report3.md (code SHA 05bbaaaac7a3; 29 root modules)", f"records' code_sha {gsha}", "MATCH" if gsha == ["05bbaaaac7a3"] else "DISAGREE",
    "the byte-identity of the 29 modules and the SHA's first commit are check_I_report3's (not re-derived here)")
row(L("PyTorch 2.14.0+cu130, seeds 2540--2559) is on \\texttt{claude/explore-I} at \\texttt{148c161}, its pre-run note at \\texttt{30807ec}", 522),
    "provenance: torch 2.14.0+cu130; records at 148c161; note at 30807ec", "records; git",
    f"torch {gtorch}; records first at 148c161 ({hm('148c161')}); note 30807ec ({hm('30807ec')})",
    "MATCH" if gtorch == ["2.14.0+cu130"] else "PARTIAL")
s75b1660 = "explore_gpu/s75b_checks_cuda_NVIDIA_GeForce_GTX_1660_SUPER_output.txt"
c1660 = git("log", "--format=%h", I_BR, "--", s75b1660).split()[-1]
row(L("The card belonged to the user's server, which left the project after this run and S75b's CUDA CHECKs", ln) + ", 44",
    "the server (1660) left the project on 19 October, after this run and S75b's CUDA CHECKs; the remaining GPU is the 3070",
    "STATUS.md at 585b666; git", f"GPU run 148c161 {hm('148c161')}; S75b CUDA CHECK on the 1660 {c1660} {hm(c1660)}; STATUS.md: "
    f"'left the project on 19 October' {'left the project on 19 October' in show(V10, 'STATUS.md')}", "NOT RECORD-CHECKABLE",
    "the order of the card's last jobs is recorded; the server's leaving is the author's statement (STATUS.md), consistent")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("3. R's Phase 5 (31c39fc; protocol 550a736)")
say("=" * 100)
_ns = {}
_src = open(os.path.join(REV, "replication_check.py")).read()
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("r_transition", "r_bound_routed"):
        exec(compile(ast.Module([node], []), "replication_check.py", "exec"), _ns)
r_tr, r_br = _ns["r_transition"], _ns["r_bound_routed"]
p5 = [json.loads(l) for l in show("31c39fc", "records/phase5.jsonl").split("\n") if l.strip()]
REC5 = {(r["arm"], r["seed"]): r for r in p5}
S5 = range(940, 960)
ARM5 = {"COPY": "WIN3_SPLIT_D8_ML", "RESET": "RESET_D8_ML", "DELTA1": "DELTA1"}


def r5(n, s):
    r = REC5[(ARM5[n], s)]
    return r_tr(r["evals"]) is not None if n == "DELTA1" else r_br(r)


C5 = {n: sum(r5(n, s) for s in S5) for n in ARM5}
W5 = {n: rlib.wilson(C5[n], 20) for n in ARM5}
pr5 = {}
for a, b in (("COPY", "RESET"), ("COPY", "DELTA1")):
    x = sum(r5(a, s) and not r5(b, s) for s in S5)
    y = sum(r5(b, s) and not r5(a, s) for s in S5)
    pr5[(a, b)] = (x, y, rlib.mcnemar1(x, y))
orc = {o: sum(r_tr(REC5[(o, s)]["evals"]) is not None for s in (940, 941)) for o in ("ORACLE_ADAM", "ORACLE_DELTA")}
# the main line's pooled test_split_copy counts (X 592c574, L ffb0771) with rlib
mlX = json.loads(show("592c574", "results/X/split_copy_results.json"))
mlL = json.loads(show("ffb0771", "results/L/split_copy_results.json"))
MLA = {"COPY": "WIN3_SPLIT_D8", "RESET": "WIN3_RESET_D8", "DELTA1": "SINGLE_DELTA_D8"}
ml = {n: sum((rlib.bound(r) if n == "DELTA1" else rlib.bound_routed(r)) for st in (mlX, mlL) for r in rlib.runs(st, a).values())
      for n, a in MLA.items()}
mlpair = {}
for n in ("RESET", "DELTA1"):
    x = y = 0
    for st in (mlX, mlL):
        ra, rb = rlib.runs(st, "WIN3_SPLIT_D8"), rlib.runs(st, MLA[n])
        f = (lambda r: rlib.bound(r)) if n == "DELTA1" else rlib.bound_routed
        x += sum(rlib.bound_routed(ra[s]) and not f(rb[s]) for s in ra)
        y += sum(f(rb[s]) and not rlib.bound_routed(ra[s]) for s in ra)
    mlpair[n] = (x, y)
cons = {}
for n in ARM5:
    lo, hi = rlib.wilson(ml[n], 20)
    rng_ = [x for x in range(21) if round(lo, 3) <= x / 20 <= round(hi, 3)]
    cons[n] = (rng_[0], rng_[-1], rng_[0] <= C5[n] <= rng_[-1])
pc = {p: sum(binom_pmf(20, x, p) for x in range(cons["COPY"][0], cons["COPY"][1] + 1)) for p in (0.70, 0.95)}
uns = {c: len(re.findall(r"^\| U-\d+", show(c, "UNSTATED.md"), re.M)) for c in ("1f8be04", "550a736", "31c39fc")}
rr = show("31c39fc", "replicate_report.md")
say(f"  counts {C5}; Wilson {W5}; pairs {pr5}; oracles {orc}; main line pooled {ml}, pairs {mlpair}; consistent ranges "
    f"{cons}; P(consistent | 0.70, 0.95) {pc}; UNSTATED rows {uns}")
ln = 347
row(L("twenty fresh seeds (940--959)", ln) if L("the records show no run finishing before it, though a start a few minutes earlier cannot be excluded", ln) else "",
    "seeds 940-959; protocol 550a736 before any record; a start a few minutes earlier not excludable",
    "check_R_phase5.md §0-2; git", f"550a736 {hm('550a736')}; first records 29ff87a {hm('29ff87a')}", "MATCH",
    "check_R_phase5: replayed latest start 22:49:48, 2.0 min after the protocol; an earlier start cannot be excluded")
row(L("R was given \\texttt{specs/test\\_split\\_copy.md} and the test's docstring", ln), "R given the spec and the test's docstring",
    "replicate_report.md:172 (31c39fc)", f"'from its specification and docstring at `ffb0771` only' in R's report: "
    f"{'from its specification and docstring at `ffb0771` only' in rr}", "MATCH")
wf = lambda n: f"[{W5[n][0]:.3f}, {W5[n][1]:.3f}]"
row(L("The copy with its reset bound routed $\\mathbf{18/20}$ (Wilson $[0.699, 0.972]$), the reset alone $9/20$ ($[0.258, 0.658]$), one delta channel $0/20$ ($[0, 0.161]$); the perfect gates $2/2$", ln) + ", 37, 69, 472, 495",
    "18/20 [0.699, 0.972], 9/20 [0.258, 0.658], 0/20 [0, 0.161]; perfect gates 2/2", "R's records with R's functions (ast)",
    f"COPY {C5['COPY']} {wf('COPY')}; RESET {C5['RESET']} {wf('RESET')}; DELTA1 {C5['DELTA1']} {wf('DELTA1')}; oracles {orc}",
    "MATCH" if (C5 == {"COPY": 18, "RESET": 9, "DELTA1": 0} and wf("COPY") == "[0.699, 0.972]" and wf("RESET") == "[0.258, 0.658]"
                and wf("DELTA1") == "[0.000, 0.161]" and orc == {"ORACLE_ADAM": 2, "ORACLE_DELTA": 2}) else "DISAGREE")
row(L("the copy against the reset 9 vs 0 ($p = 0.002$) and against one delta channel 18 vs 0 ($p = 3.8\\times10^{-6}$), where the main line has 12 vs 0 and 18 vs 0", ln),
    "9 vs 0 (p 0.002), 18 vs 0 (p 3.8e-6); main line 12 vs 0 and 18 vs 0", "R's records; main line split_copy records (rlib)",
    f"{pr5[('COPY', 'RESET')][:2]} p {pr5[('COPY', 'RESET')][2]:.3g}; {pr5[('COPY', 'DELTA1')][:2]} p {pr5[('COPY', 'DELTA1')][2]:.2g}; "
    f"main line {mlpair}", "MATCH" if pr5[('COPY', 'RESET')][:2] == (9, 0) and pr5[('COPY', 'DELTA1')][:2] == (18, 0)
    and mlpair == {"RESET": (12, 0), "DELTA1": (18, 0)} else "DISAGREE")
row(L("all three do ($18/20$, $6/20$ and $0/20$ there)", ln) + ", 37, 69", "each arm inside the main line's pooled interval (18/20, 6/20, 0/20)",
    "main line records (rlib); Wilson", f"main line {ml}; consistent ranges and verdicts {cons}",
    "MATCH" if ml == {"COPY": 18, "RESET": 6, "DELTA1": 0} and all(v[2] for v in cons.values()) else "DISAGREE")
row(L("the copy arm reads ``consistent'' with probability $0.61$ at a true rate of $0.70$ and $0.64$ at $0.95$", ln),
    "P(consistent) 0.61 at 0.70, 0.64 at 0.95", "binomial over the consistent range", f"range {cons['COPY'][:2]}: {pc[0.70]:.3f}, {pc[0.95]:.3f}",
    "MATCH" if (round(pc[0.70], 2), round(pc[0.95], 2)) == (0.61, 0.64) else "DISAGREE")
row(L("``from the written specification alone'' holds for the delta arm only", ln) + ", 37, 44, 69, 472, 495",
    "'from the specification alone' for the delta arm only; the other two are R's Phase 4 recipe", "check_R_phase5.md row 35; round6_A3 §2",
    "the referee's PARTIAL, quoted as its qualification", "MATCH")
row(L("Every number reproduces from R's records with R's outcome functions and with the main line's, which agree", ln),
    "every number reproduces, both outcome definitions agree (check_R_phase5.md)", "check_R_phase5.md; this script",
    "38 rows: 32 MATCH, 2 PARTIAL, 0 DISAGREE, 4 not record-checkable (none a number)", "MATCH")
row(L("The 44 unstated details of Phases~2--4 became 65 with Phase~5's", ln), "44 unstated details became 65",
    "UNSTATED.md at 1f8be04 / 550a736 / 31c39fc", f"'| U-' rows {uns}", "MATCH" if (uns["1f8be04"], uns["550a736"]) == (44, 65) else "DISAGREE",
    "3 of Phase 5's 21 entries are marked [S] (stated) and 1 [S/A] (check_R_phase5 §), as 10 of the 44 were not guesses")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("4. S75, with the tenth WIDE_SLOW seed (839)")
say("=" * 100)
RES75 = "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
st75 = json.loads(show(I_BR, RES75))
assert show(I_BR, RES75) == show("4371824", RES75)
R75 = {}
for k, r in st75["runs"].items():
    a, s = k.split("|")[:2]
    if r.get("ok"):
        R75.setdefault(a, {})[int(s)] = r
dense = show(I_BR, "explore_i_dense.py")
D = {}
extract(dense, {"routed"}, D)
routed = D["routed"]


def routed_at(e):
    return routed({"end": e})


WL, WS, RG = R75["WIDE_SLOW"], R75["WIDE_SPLIT"], R75["REG3"]
w839, s839 = WL[839], WS[839]


def ev(r):
    return [(e["step"], L_, v) for e in r["curve"] for L_, v in e["routing"].items()]


info = {}
for nm, r in (("WIDE_SLOW", w839), ("WIDE_SPLIT", s839)):
    e = r["end"]
    info[nm] = dict(set=e["set"], routed=routed(r), eta_end=max(v["eta"] for v in e["routing"].values()),
                    oto_end=[v["one_to_one"] for v in e["routing"].values()], ch_map=[v["ch_map"] for v in e["routing"].values()],
                    margin=[v["margin"] for v in e["routing"].values()],
                    eta_max=max(v["eta"] for _, _, v in ev(r)), oto_any=sum(v["one_to_one"] for _, _, v in ev(r)),
                    routed_any=sum(routed_at(c) for c in r["curve"]), n=len(r["curve"]),
                    p1200=r["curve"][0]["probe"]["L2"]["acc"], step0=r["curve"][0]["step"],
                    p3600=max(next(c for c in r["curve"] if c["step"] == 3600)["probe"][L_]["acc"] for L_ in ("L2", "L3", "final")),
                    pend=max(e["probe"][L_]["acc"] for L_ in e["probe"]), sha=r["sha"], hw=r["hw"])
    say(f"  {nm} 839: {info[nm]}")
wl = [r["end"]["set"] for r in WL.values()]
ws = [r["end"]["set"] for r in WS.values()]
diffs = {s: WL[s]["end"]["set"] - WS[s]["end"]["set"] for s in WL}
pair_r = sorted({(routed(WS[s]), routed(WL[s])) for s in WL})
eta_wl = max(v["eta"] for r in WL.values() for _, _, v in ev(r))
oto_wl = sum(v["one_to_one"] for r in WL.values() for _, _, v in ev(r))
nev_wl = sum(1 for r in WL.values() for _ in ev(r))
GATED = ("WIDE_SPLIT", "REG3", "WIDE_SLOW")
n_gated = sum(len(R75[a]) for a in GATED)
p1200 = [r["curve"][0]["probe"]["L2"]["acc"] for a in GATED for r in R75[a].values()]
p3600 = {(a, s): max(next(c for c in r["curve"] if c["step"] == 3600)["probe"][L_]["acc"] for L_ in ("L2", "L3", "final"))
         for a in GATED for s, r in R75[a].items()}
above = sorted((a, s, round(v, 3)) for (a, s), v in p3600.items() if v > 0.27)
eta20 = max(v["eta"] for a in ("WIDE_SPLIT", "REG3") for r in R75[a].values() for _, _, v in ev(r))
oto20 = sum(v["one_to_one"] for a in ("WIDE_SPLIT", "REG3") for r in R75[a].values() for _, _, v in ev(r))
say(f"  WIDE_SLOW ten seeds: SET {min(wl):.4f}-{max(wl):.4f}; WIDE_SPLIT {min(ws):.4f}-{max(ws):.4f}; ROUTED pairs {pair_r}; "
    f"|SET diff| max {max(abs(v) for v in diffs.values()):.5f} (seed {max(diffs, key=lambda s: abs(diffs[s]))}), median "
    f"{statistics.median(abs(v) for v in diffs.values()):.5f}; 839 diff {diffs[839]:+.5f}; eta max {eta_wl:.5f}; one-to-one "
    f"{oto_wl} of {nev_wl}")
say(f"  gated runs {n_gated} ({ {a: len(R75[a]) for a in GATED} }); L2 probe at 1200 {min(p1200):.4f}-{max(p1200):.4f}; above 0.27 at "
    f"3600: {above}; registered learned arms (20 runs) eta max {eta20:.5f}, one-to-one {oto20}")
r2 = show(I_BR, "explore_out/I/report_2.md")
r2_839 = "| WIDE_SLOW (no-split twin, printed) | 830–839 | 0/10 [0.00, 0.28] NEVER | 0.534 (0.530–0.536) |" in r2
w10 = rlib.wilson(0, 10)
ln = 481
row(L("matched the split arm seed for seed on all ten", ln), "WIDE_SLOW matched WIDE_SPLIT seed for seed on all ten (839 included)",
    f"{RES75} at 4371824 (= tip); explore_i_dense.routed (ast)", f"ROUTED (split, slow) pairs {pair_r} on {len(diffs)} seeds; 839: "
    f"slow SET {info['WIDE_SLOW']['set']:.4f} vs split {info['WIDE_SPLIT']['set']:.4f} ({diffs[839]:+.5f}); both not ROUTED",
    "MATCH" if pair_r == [(False, False)] and len(diffs) == 10 else "DISAGREE")
row(L("$0.530$--$0.536$ against $0.529$--$0.536$", ln), "WIDE_SLOW 0.530-0.536 against WIDE_SPLIT 0.529-0.536 (ten seeds)", "records",
    f"WIDE_SLOW {min(wl):.4f}-{max(wl):.4f}; WIDE_SPLIT {min(ws):.4f}-{max(ws):.4f}",
    "MATCH" if (f"{min(wl):.3f}", f"{max(wl):.3f}", f"{min(ws):.3f}", f"{max(ws):.3f}") == ("0.530", "0.536", "0.529", "0.536") else "DISAGREE",
    "839's 0.5333 lies inside the nine-seed range")
row(L("$\\eta^2$ below $0.001$ at every evaluation, no one-to-one map at any", ln), "WIDE_SLOW eta^2 < 0.001 at every evaluation, no one-to-one map at any (ten seeds)",
    "records", f"eta max {eta_wl:.5f} (839: {info['WIDE_SLOW']['eta_max']:.5f}); one-to-one {oto_wl} of {nev_wl} evaluations",
    "MATCH" if eta_wl < 0.001 and oto_wl == 0 else "DISAGREE")
row(L("all but one of the 29 (a register run, at $0.29$) had lost it by 3600", ln),
    "all but one of the 29 gated runs had lost the source by 3600; the residual held it at 1200 in every gated run (0.35-0.61)",
    "records (probe L2/L3/final at 3600; L2 at 1200)", f"gated runs now {n_gated} (WIDE_SPLIT 10, REG3 10, WIDE_SLOW 10); above 0.27 at "
    f"3600: {above}; L2 at 1200 over all {n_gated}: {min(p1200):.4f}-{max(p1200):.4f} (839 slow {info['WIDE_SLOW']['p1200']:.4f}, "
    f"at 3600 {info['WIDE_SLOW']['p3600']:.4f})", "DISAGREE",
    "the count is stale: with seed 839 there are 30 gated runs, and 839 also held the source at 1200 and lost it by 3600, so "
    "'all but one of the 30'; the 0.35-0.61 range and the single exception (REG3 839, 0.29) are unchanged. 'Nine seeds' was "
    "updated to 'all ten' in the same paragraph, this count was not")
row(L("$\\eta^2$ by source was below $0.003$ at every evaluation of every gate layer in all twenty runs", ln),
    "eta^2 < 0.003 and no one-to-one map in the twenty registered learned runs (unchanged by 839)", "records",
    f"max {eta20:.5f}; one-to-one {oto20}", "MATCH" if eta20 < 0.003 and oto20 == 0 else "DISAGREE")
row(L("the window gate with and without the split and the register gate route $0/10$ each", 68),
    "index: window gate with and without the split and the register gate route 0/10 each", "records",
    f"ROUTED at the end: " + ", ".join(f"{a} {sum(routed(r) for r in R75[a].values())}/{len(R75[a])}" for a in GATED),
    "MATCH" if all(sum(routed(r) for r in R75[a].values()) == 0 and len(R75[a]) == 10 for a in GATED) else "DISAGREE")
row("explore-I:explore_out/I/report_2.md (d552c6c)", "report 2 table row: WIDE_SLOW 830-839, 0/10 [0.00, 0.28] NEVER, 0.534 (0.530-0.536)",
    "records; Wilson", f"row present {r2_839}; Wilson 0/10 [{w10[0]:.2f}, {w10[1]:.2f}]; median SET {statistics.median(wl):.3f}",
    "MATCH" if r2_839 and f"{w10[1]:.2f}" == "0.28" and f"{statistics.median(wl):.3f}" == "0.534" else "DISAGREE")
row("explore_out/I/report_2.md section 6 (d552c6c)", "paired with WIDE_SPLIT: neither ROUTED on any of the 10 shared seeds; SET differs by at most 0.003 (median 0.0006)",
    "records", f"pairs {pair_r}; max {max(abs(v) for v in diffs.values()):.5f} (831), median {statistics.median(abs(v) for v in diffs.values()):.5f}",
    "PARTIAL", "as check_S75_v6 row 89 found for nine seeds: 831's 0.00325 exceeds 0.003 unrounded; 839 adds -0.00048 and leaves the median at 0.0006. Not quoted in tex10")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("5. S75b and the S75 note's correction")
say("=" * 100)
mut = show(I_BR, "explore_gpu/s75b_mutants_cpu_output.txt")
mcaught = re.search(r"mutants caught: (\d+)/(\d+)", mut).groups()
cpu = show(I_BR, "explore_gpu/s75b_checks_cpu_output.txt")
cpu_sha = re.search(r"S75b code SHA (\w+)", cpu).group(1)
cuda = {g: re.search(r"S75b code SHA (\w+)", show(I_BR, f"explore_gpu/s75b_checks_cuda_NVIDIA_GeForce_{g}_output.txt")).group(1)
        for g in ("GTX_1660_SUPER", "RTX_3070_Laptop_GPU")}
spec = show(I_BR, "explore_out/I/s75b_spec_draft.md")
pilot_files = [p for p in git("ls-tree", "-r", "--name-only", I_BR, "explore_out/I/gpu/s75b/").split() if "bench" not in p and ".gitignore" not in p]
h3070 = (2 * 1.15 + 10 * 1.0 + 10 * 1.25 + 10 * 1.25 + 10 * 1.7, 2 * 1.15 + 10 * 1.0 + 10 * 1.25 + 10 * 1.25 + 10 * 2.0)
note_f = "explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md"
note_hist = git("log", "--format=%h", I_BR, "--", note_f).split()
r2corr = "Correction to the pre-run note" in show("03bffdf", "explore_out/I/report_2.md")
first_learned = None
for c in reversed(git("log", "--format=%h", I_BR, "--", RES75).split()):
    if any(k.split("|")[0] in GATED for k in json.loads(show(c, RES75))["runs"]):
        first_learned = c
        break
say(f"  mutants caught {mcaught}; CPU CHECK code SHA {cpu_sha} ('ALL CHECKS PASSED' {'ALL CHECKS PASSED' in cpu}); CUDA CHECK SHAs {cuda}; "
    f"S75b result files beyond the benchmark: {pilot_files}; stage 2 on the 3070 alone from the spec's per-run hours: "
    f"{h3070[0]:.1f}-{h3070[1]:.1f} h; note commits {note_hist}; first learned-arm record {first_learned} {hm(first_learned)}; "
    f"5bc6c2e {hm('5bc6c2e')}; 03bffdf {hm('03bffdf')} correction {r2corr}")
row(L("the last with a mutation pass of 48 variants of the failure rules, all caught by its CHECKs", ln),
    "mutation pass: 48 variants, all caught", "explore_gpu/s75b_mutants_cpu_output.txt (8d2e89f)", f"caught {mcaught[0]}/{mcaught[1]}",
    "MATCH" if mcaught == ("48", "48") else "DISAGREE")
row(L("its final driver checked on the CPU (the cards' CUDA CHECKs were run on an earlier driver and are to be rerun)", ln) + ", 500",
    "final driver checked on the CPU; the cards' CUDA CHECKs on an earlier driver", "s75b_checks_*_output.txt",
    f"CPU (a4ab996): SHA {cpu_sha}, all pass {'ALL CHECKS PASSED' in cpu}; CUDA 1660 / 3070: {cuda}",
    "MATCH" if cpu_sha == "ee04da11a164" and set(cuda.values()) == {"0759062126f5"} else "DISAGREE")
row(L("the referee found its provenance claim to hold by the commit graph, two design changes made after the learned-arm records were visible to be disclosed rather than blind, and four pilot outcomes its rules did not name", ln),
    "the referee's S75b review: provenance holds by the commit graph; two design changes not blind by git; four pilot outcomes not covered",
    "spec_review_s75b_draft.md (3f7d84b)", "verdict: 'What git shows matches the claim'; 'Two design changes postdate b649751 ... neither is blind by git. Label them'; "
    "'Four outcomes are not covered'", "MATCH")
row(L("and its pilot not run", ln), "S75b's pilot not run", "explore-I tree", f"S75b record files other than the 3070 benchmark: {pilot_files}",
    "MATCH" if not pilot_files else "DISAGREE")
row(L("about three days for the second stage", ln), "on the 3070 alone, about three days for the second stage",
    "s75b_spec_draft.md 'Cost'; STATUS.md", f"the spec's per-run 3070 hours sum to {h3070[0]:.0f}-{h3070[1]:.0f} h "
    f"({h3070[0] / 24:.1f}-{h3070[1] / 24:.1f} days) for stage 2's 42 runs; 'three days' appears only in STATUS.md", "NOT RECORD-CHECKABLE",
    "an estimate in STATUS.md; the spec's own table gives about 2.3-2.4 days, so 'about three' is a round-up")
row(L("Session~I corrected its pre-run note on 12 October: the note (committed at 00:46 UTC)", ln),
    "note committed 00:46 UTC; corrected in report 2 on 12 October; note not edited; no learned-arm record in git until b649751 at 13:53 UTC",
    "git", f"5bc6c2e {hm('5bc6c2e')}; report 2's correction 03bffdf {hm('03bffdf')} ({r2corr}); note commits {note_hist}; "
    f"first learned-arm record {first_learned} {hm(first_learned)}",
    "MATCH" if note_hist == ["5bc6c2e"] and first_learned == "b649751" and r2corr else "DISAGREE")
row(L("the card's timeline puts up to three before it, which the records cannot fix (the referee: between none and three)", ln),
    "up to three window-gate runs before the note; the referee: none to three", "rev9/fifth_pass.md §2.2; check_S75_v6.md rows 16, 92-93",
    "the referee: 0-3 had finished, under one invocation 832 ends 98 s before the note and 833 runs", "MATCH")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 100)
say("6. The referee and review history; status statements")
say("=" * 100)
passes = ["docs/review_2026-10/revision_9_check.md"] + [f"docs/review_2026-10/rev9/{p}_pass.md" for p in
                                                       ("second", "third", "fourth", "fifth", "sixth", "seventh")]
pt = {p: git("log", "--format=%h %cI", "--diff-filter=A", V_BR, "--", p).split("\n")[0] for p in passes}
first_draft = git("log", "--format=%h %cI", "--reverse", PIN["origin/claude/rev9-draft"], "--", "multichannel_hebbian_report_v9.tex").split("\n")[0]
say(f"  Revision 9 passes: {pt}; first draft {first_draft}; merge c918ff7 {hm('c918ff7')}")
sev = show(V_BR, "docs/review_2026-10/rev9/seventh_pass.md")
six = show(V_BR, "docs/review_2026-10/rev9/sixth_pass.md")
row(L("Revision~9 (10--13 October 2026)", 42), "Revision 9: 10-13 October", "git", f"first draft {first_draft[:40]}; merged c918ff7 {hm('c918ff7')}", "MATCH")
row(L("Revision~9 was checked against the records in seven passes by the referee session before it was merged", 42),
    "seven passes before the merge", "review-V: revision_9_check.md + rev9/second..seventh_pass.md",
    f"{len(passes)} pass files; the seventh added {pt[passes[-1]]}; merge {hm('c918ff7')}",
    "MATCH" if len(passes) == 7 and pt[passes[-1]].startswith("00290a8") else "DISAGREE",
    "the seventh pass checked 084327e; the merged 010a085 only adds c3ad769 to a provenance row (seventh_pass §4 calls that optional)")
row(L("The sixth pass left two items", 364), "sixth pass: two items (L-status, six statements on five lines; R's 44); seventh fixed both and passed",
    "rev9/sixth_pass.md, seventh_pass.md", f"sixth: 'six statements on five lines' {'six statements on five lines' in six}, 'one wording point at tex6:528' "
    f"{'one wording point at tex6:528' in six}; seventh: 'Ready to merge' {'**Ready to merge.**' in sev}", "MATCH")
a3 = show(MAIN, "docs/review_2026-10/round6_A3.md")
a3rows = re.findall(r"^\| [^|]+ \| [^|]+ \| `check_[^|]+ \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", a3, re.M)
say(f"  round6_A3 table rows (rows, MATCH, PARTIAL, DISAGREE, NRC): {a3rows}")
row(L("(66 rows: 55 match, 7 partial, 3 not record-checkable and one disagreement", 366),
    "66 rows: 55/7/3/1", "round6_A3.md (main line 9116a17)", f"{a3rows[0]}", "MATCH" if a3rows[0] == ("66", "55", "7", "1", "3") else "DISAGREE")
row(L("R's Phase~5 (38 rows) and I's GPU run (65 rows) reproduce", 366), "R 38 rows, I 65 rows", "round6_A3.md",
    f"{a3rows[1]}, {a3rows[2]}", "MATCH" if a3rows[1][0] == "38" and a3rows[2][0] == "65" else "DISAGREE")
row(L("the pooled D1 and D4 pairs, the D2 no-room count and D4's $p$ recomputed independently", 366), "what V recomputed independently",
    "round6_A3.md §1", f"'{'The pooled D1 and D4 pairs, the D2 no-room count and D4' in a3}'", "MATCH",
    "check_delta_recipe_L.py §3 in fact recomputes every pooled number independently; the sentence understates, harmlessly")
row(L("is confirmed against X's run log to the minute", 366), "Amendment 2's statement confirmed against X's log to the minute",
    "round6_A3 §1; check_amendment2_log_out.txt", "at 15:36:08 (55dac50) and 15:37:52 (2225f00) exactly the 24 runs had finished; the next at 15:38:19", "MATCH")
# spec reviews
p_one = 2 * binom_ge(20, 18, 0.9) * (1 - binom_ge(20, 18, 0.9))
p_val = 1 - 0.95 ** 2
p_room = binom_ge(40, 36, 0.8625)


def p_shown(pb_, pk, nested):
    """P(one-sided exact McNemar p < 0.05) over 40 pairs; b = key-term only, c = baseline only."""
    if nested:
        pb, pc = pk - pb_, 0.0
    else:
        pb, pc = pk * (1 - pb_), pb_ * (1 - pk)
    pn, tot = 1 - pb - pc, 0.0
    for b in range(41):
        for c in range(41 - b):
            if rlib.mcnemar1(b, c) < 0.05:
                tot += math.factorial(40) / (math.factorial(b) * math.factorial(c) * math.factorial(40 - b - c)) * pb ** b * pc ** c * pn ** (40 - b - c)
    return tot


ps = {(pk, n): p_shown(0.8625, pk, n) for pk in (0.933, 0.95) for n in (False, True)}
say(f"  register review: P(exactly one machine >= 18/20 | 0.90) {p_one:.3f}; validity 2/2 at 0.95 fails {p_val:.4f}; key-term review: "
    f"P(WIN3_SLOW >= 36/40 | 0.8625) {p_room:.3f}; P(K1 SHOWN) {ps}")
row(L("``reliable on one machine'' fires 44\\% of the time at a shared true rate of $0.90$", 366),
    "register review: 18/20 supports 0.70; 'reliable on one machine' 44% at 0.90; validity untested a tenth of the time",
    "spec_review_register_draft.md; binomial", f"P(exactly one of two machines >= 18/20 | 0.90) = {p_one:.3f}; P(2 nudged seeds not both bound | 0.95) "
    f"= {p_val:.4f}; Wilson lower bound at 18/20 {rlib.wilson(18, 20)[0]:.3f}", "MATCH" if round(p_one, 2) == 0.44 and round(p_val, 1) == 0.1 else "DISAGREE")
row(L("its primary claim is SHOWN with probability $0.15$--$0.27$ at the main line's rates, and in a third of outcomes cannot be shown at all", 366),
    "key-term review: 69/80 baseline; P(SHOWN) 0.15-0.27; a third of outcomes cannot show it", "spec_review_two_keyterm_draft.md; binomial",
    f"P(K1 SHOWN), key term 0.933/0.95, independent/nested: " + ", ".join(f"{k[0]} {'nested' if k[1] else 'indep'} {v:.3f}" for k, v in ps.items())
    + f"; P(WIN3_SLOW >= 36/40 | 69/80) {p_room:.3f}", "MATCH" if round(min(ps.values()), 2) == 0.15 and round(max(ps.values()), 2) == 0.27 and round(p_room, 2) == 0.34 else "PARTIAL")
e_draft = "f5d3893"
V_AT = git("rev-list", "-1", "--before=" + ctime(V10).isoformat(), V_BR).strip()[:7]
row(L("Session~E's key-load draft was committed after those reviews", 366), "E's key-load draft committed after the reviews; not reviewed",
    "git", f"reviews 3f7d84b {hm('3f7d84b')} and 723827b {hm('723827b')}; draft {e_draft} {hm(e_draft)}; a key-load review on review-V "
    f"at {V_AT} (its last commit before {V10}): {any('keyload' in p for p in git('ls-tree', '-r', '--name-only', V_AT, 'docs/review_2026-10/').split())}",
    "MATCH", "V's keyload_options_review.md (11a78ee, 19 Oct 17:51 UTC) postdates the merge; it reviews E's five options, not the draft")
row(L("its reviews of three of the four specifications the sessions drafted (three main-line tests and one exploratory screen)", 44) + ", 366",
    "V reviewed three of four drafted specifications (G, H main-line; S75b exploratory); E's not", "review-V files",
    "spec_review_register_draft.md, spec_review_two_keyterm_draft.md, spec_review_s75b_draft.md; none for E's", "MATCH")
row(L("the three drafted specifications are not ready", 70), "index, V's row: 'the three drafted specifications are not ready'",
    "tex10:44, 366, 499; review-V files", "V reviewed G, H and S75b; tex10:499 defines 'the three drafted' as G, H and E", "PARTIAL",
    "credited to V in its row, but the three V judged are G, H and S75b, while the three main-line drafts (tex10:499) include E's, which V "
    "never reviewed (E's own screen found it not ready). Say 'the three specifications it reviewed'")
row(L("the referee's checks of the first three of these (\\texttt{docs/review\\_2026-10/round6\\_A3.md}; the fifth round's screens have not been refereed", 44),
    "V's checks cover delta L/pooled, the GPU run and R's Phase 5; the fifth round's screens are not refereed", "round6_A3.md; review-V tip",
    f"round6_A3 covers exactly those three (+ Amendment 2); V's last commit before this check {git('log', '-1', '--format=%h %cI', V_BR).strip()}", "MATCH")
rev10_files = [p for p in git("ls-tree", "-r", "--name-only", V10, "docs/review_2026-10/rev10/").split()]
fp = show(V10, "docs/review_2026-10/rev10/first_pass.md")
sp = show(V10, "docs/review_2026-10/rev10/second_pass.md")
row(L("this revision's own additions were checked in two passes by a session that did not write them, every number recomputed from the records", 44) + ", 495",
    "two passes by a session that did not write them, in docs/review_2026-10/rev10/", "585b666:docs/review_2026-10/rev10/",
    f"files {[p.split('/')[-1] for p in rev10_files]}; first pass of ea3afef (123 rows), second of 6e75f69's 19 changed lines (31 rows); "
    f"the merged text is d2b7abd (third version, {hm('d2b7abd')}), 4 tex lines (44, 481, 495, 500) changed after the second pass", "PARTIAL",
    "two passes exist and are committed; the second covered only the lines changed since the first and the third version's changes "
    "(the second pass's three items) were not passed; 'every number recomputed' rests on the first pass. Not by V (README: V had closed on "
    "13 October). Note also that this V check writes into the same directory, so the citation will then cover both")
row(L("A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions from the records and found them to reproduce", 495),
    "a referee recomputed the numbers of Rev 8, Rev 9 and this revision's additions and found them to reproduce",
    "round6_A3.md; rev10/README.md", "V recomputed three of this revision's additions (delta L/pooled, GPU run, R Phase 5); the fifth round's screens and "
    "S75's 839 were recomputed only by the unnamed checking session, which the same paragraph distinguishes from 'session V'", "PARTIAL",
    "'A referee' with 'this revision's additions' reads as V covering all additions; the paragraph's last sentence says the screens are "
    "not refereed by V. Say 'the referee recomputed ... the main-line and replication additions'")
st = show(V10, "STATUS.md")
row(L("which left the project on 19 October; the remaining GPU is the laptop's RTX~3070", 44), "the 1660 server left on 19 October; the remaining GPU is the 3070",
    "STATUS.md (585b666)", f"STATUS: 'The server (Ryzen 5 3600; GTX 1660 SUPER) left the project on 19 October' "
    f"{'left the project on 19 October' in st}", "NOT RECORD-CHECKABLE", "the author's statement, consistent with STATUS.md")
row(L("No claim of Revision~9 changes.", 44), "no claim of Revision 9 changes", "c918ff7 vs 585b666 tex",
    "Rev 9 stated test_delta_recipe's per-machine X result (secondary) and left the pooled labels to L; Rev 10 adds the pooled claims; no Rev 9 "
    "claim's count or reading is altered", "MATCH")
row(L("\\texttt{delta\\_recipe} pooled, R's Phase~5 and I's GPU run reproduce", 70), "V's row: delta pooled, R's Phase 5 and I's GPU run reproduce",
    "round6_A3.md", "as rows above", "MATCH")
row(L("The register test is the nearest to running", 499), "next steps: three drafts each need revision; the register test the nearest",
    "spec reviews; E's batch 21", "G's needs a named outcome function, CHECKs, a calibrated rate claim (V); H's a baseline with room (V); E's a "
    "validity check (E's S69)", "MATCH", "'the nearest to running' is a judgment")
row(L("the CUDA CHECKs to rerun on the final driver and the referee's review points to take, then the pilot, on the one remaining GPU", 500),
    "S75b: CUDA CHECKs to rerun, review points to take, then the pilot", "as row (final driver); STATUS.md", f"CUDA SHAs {cuda} vs CPU {cpu_sha}", "MATCH")
row(L("The headline's arms on a GPU, by our own code, are a third machine class and not an independent implementation", 472),
    "limits: the GPU run is our own code on a third machine class; R's fifth phase by a rule that separates little", "check_I_report3 (29 modules byte-identical); check_R_phase5",
    "as rows above", "MATCH")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("ROWS")
print("=" * 100)
print("| # | tex10 line | statement | source | recomputed | status | note |")
print("|---|---|---|---|---|---|---|")
for i, (w, s, src_, rc, stt, nt) in enumerate(ROWS, 1):
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    print(f"| {i} | {cell(w)} | {cell(s)} | {cell(src_)} | {cell(rc)} | **{stt}** | {cell(nt)} |" if stt != "MATCH"
          else f"| {i} | {cell(w)} | {cell(s)} | {cell(src_)} | {cell(rc)} | {stt} | {cell(nt)} |")
cnt = collections.Counter(r[4] for r in ROWS)
print(f"\nCOUNTS: {len(ROWS)} rows: MATCH {cnt['MATCH']}, PARTIAL {cnt['PARTIAL']}, DISAGREE {cnt['DISAGREE']}, "
      f"NOT RECORD-CHECKABLE {cnt['NOT RECORD-CHECKABLE']}")
