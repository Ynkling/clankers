"""check_split_copy_L.py -- Referee V: test_split_copy's L half and pooled result, recorded on the main line at ffb0771
('Record test_split_copy on L and pooled: C1 SHOWN (18/20 vs 6/20, 12 vs 0, p = 0.00024), C2 HOLDS (0 vs 2), C3 SHOWN
...'), against its records, its log, its docstring and the fifth version of the Revision 9 draft (claude/rev9-draft
@ c5e4508, multichannel_hebbian_report_v9.tex; cited tex5:NNN; the L/pooled numbers folded in at ef6d690).

Inputs, all read with git at pinned refs (nothing is run but this script; nothing is written but stdout):
  records  ffb0771:results/L/split_copy_results.json, ffb0771:results/L/split_copy_L.log,
           592c574:results/X/split_copy_results.json (X's half; unchanged at ffb0771, asserted)
  code     ffb0771:test_split_copy.py and the modules its report uses, read at ffb0771; the outcome, pairing,
           McNemar, Wilson, band, verdict, pooling, reading and report functions are EXTRACTED from those sources
           with ast and exec'd (no torch, nothing imported from the working tree); rlib (the referee's own
           re-implementation) is used only as an independent cross-check
  spec     ffb0771:specs/test_split_copy.md (fb3acb7; no amendment exists)
  draft    c5e4508:multichannel_hebbian_report_v9.tex
  other    84a7e57 (origin/claude/bdh-repo-curl-obgdqk): L's longrun state for the first start (log, json, exit)

    python3 -B docs/review_2026-10/rev9/check_split_copy_L.py > docs/review_2026-10/rev9/check_split_copy_L.md
"""
import ast
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
from contextlib import redirect_stdout

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402  (independent cross-check only)

RES, XRES, TEST_C, SPEC_C, DRAFT_C, FOLD_C = "ffb0771", "592c574", "1f52cd5", "fb3acb7", "c5e4508", "ef6d690"
FIRST_L = "84a7e57"
DRAFT = "multichannel_hebbian_report_v9.tex"
LREC, XREC, LLOG = "results/L/split_copy_results.json", "results/X/split_copy_results.json", "results/L/split_copy_L.log"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


# ── 0. sources ───────────────────────────────────────────────────────────────
SRC = show(RES, "test_split_copy.py")
SRC_LINES = SRC.split("\n")
storeL = json.loads(show(RES, LREC))
storeX = json.loads(show(XRES, XREC))
LOG = show(RES, LLOG).split("\n")
NLOG = max(i + 1 for i, x in enumerate(LOG) if x.strip())
TEX = show(DRAFT_C, DRAFT).split("\n")
SPEC = show(RES, "specs/test_split_copy.md")
assert git("diff", "--stat", XRES, RES, "--", XREC).strip() == "", "X's file changed between 592c574 and ffb0771"


def extract(source, names, ns):
    """exec the module-level defs/assignments named in `names` from `source` into ns."""
    body = []
    for n in ast.parse(source).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign):
            tg = set()
            for t in n.targets:
                for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(e, ast.Name):
                        tg.add(e.id)
            if tg & names:
                body.append(n)
    got = {getattr(n, "name", None) for n in body} | {e.id for n in body if isinstance(n, ast.Assign) for t in n.targets
                                                      for e in (t.elts if isinstance(t, ast.Tuple) else [t])
                                                      if isinstance(e, ast.Name)}
    missing = names - got
    assert not missing, f"not found: {missing}"
    exec(compile(ast.Module(body=body, type_ignores=[]), "extract", "exec"), ns)
    return ns


def mod(name, path, names, extra=None):
    ns = {"math": math, "statistics": statistics}
    ns.update(extra or {})
    extract(show(RES, path), set(names), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")}), ns


# the modules test_split_copy's report reaches, at ffb0771 (unchanged since 592c574: asserted below)
HELPERS = ["test_conv_lr.py", "test_router_discovery.py", "test_slow_start.py", "test_stream_channels.py",
           "test_router_layout.py", "test_scale_axes.py", "test_stream_recipe.py", "fail_class_v2.py",
           "test_window_gate.py", "test_stream_curriculum.py", "test_binding_onset.py", "test_recipe_scope.py"]
helpers_same = git("diff", "--stat", XRES, RES, "--", *HELPERS).strip() == ""
conv, _ = mod("conv", "test_conv_lr.py", ["bound_r", "mcnemar_greater"])
rdisc, _ = mod("rdisc", "test_router_discovery.py", ["get"])
sstart, _ = mod("sstart", "test_slow_start.py", ["ok_r", "runs_of", "curve_str"], {"get": rdisc.get})
schan, _ = mod("schan", "test_stream_channels.py", ["shared_max", "merged", "med_int", "mstr"])
rlay, _ = mod("rlay", "test_router_layout.py", ["MARGIN_PARTIAL", "ETA_SPLIT", "fail_class"])
saxes, _ = mod("saxes", "test_scale_axes.py", ["fail_class_k"], {"fail_class": rlay.fail_class})
tsr, _ = mod("tsr", "test_stream_recipe.py", ["routed", "wilson", "outcome", "ROUTED_ACC"],
             {"fail_class_k": saxes.fail_class_k, "shared_max": schan.shared_max, "merged": schan.merged})
fc2, _ = mod("fc2", "fail_class_v2.py", ["fail_class_v2", "multichannel", "fail_class_k_v2", "outcome_v2"],
             {"MARGIN_PARTIAL": rlay.MARGIN_PARTIAL, "ETA_SPLIT": rlay.ETA_SPLIT, "merged": schan.merged,
              "shared_max": schan.shared_max})
tgw, _ = mod("tgw", "test_window_gate.py", ["verdict", "pool_claim", "VALID_N"], {"mcnemar_greater": conv.mcnemar_greater})
tgw.CEIL_C_ARM = {}                                   # ARMS' model dict (not used by any outcome)
tscur, _ = mod("tscur", "test_stream_curriculum.py", ["band_n", "RELIABLE_F", "MAJORITY_F"])
tbo, _ = mod("tbo", "test_binding_onset.py", ["fmt_step", "EVAL_EVERY"])
trs, _ = mod("trs", "test_recipe_scope.py", ["groups_str"])
X_STORE_FOR_ALSO = {}
NS = dict(math=math, statistics=statistics, tgw=tgw, tsr=tsr, tss=sstart, tscur=tscur, fc2=fc2,
          tsc=types.SimpleNamespace(ARM={"B_conv": {}}), get=rdisc.get, ok_r=sstart.ok_r, bound_r=conv.bound_r,
          mcnemar_greater=conv.mcnemar_greater, med_int=schan.med_int, mstr=schan.mstr, shared_max=schan.shared_max,
          fmt_step=tbo.fmt_step, EVAL_EVERY=tbo.EVAL_EVERY, groups_str=trs.groups_str, curve_str=sstart.curve_str,
          load_store=lambda p: X_STORE_FOR_ALSO[p])
TSC_NAMES = {"MACHINES", "RANGES", "VALID_N", "C2_BOUND", "SINGLE_ARM", "ARMS", "ARM", "KEYS", "ARMS_RUN", "CEILS",
             "OUTCOME", "CLAIMS", "C2", "PRINTED", "READING_1", "READING_2", "seeds_for", "outcome2", "success",
             "other_of", "v12_rows", "paired2", "validity", "tally", "verdict", "c2_verdict", "band_of", "band_str",
             "out_of", "print_claims", "print_printed", "print_bands", "readings", "pool_claim", "complete",
             "stream_str", "raw_rows", "report"}
extract(SRC, TSC_NAMES, NS)
T = types.SimpleNamespace(**NS)

# ── 1. recomputation with the test's own functions ───────────────────────────
seedsL, seedsX = T.seeds_for("L"), T.seeds_for("X")
TL, TX = T.tally(storeL, seedsL), T.tally(storeX, seedsX)
TT = {"X": TX, "L": TL}
PC = {name: T.pool_claim([TT[m]["claims"][name] for m in T.MACHINES]) for name, *_ in T.CLAIMS + (T.C2,)}
PB = {k: dict(c=sum(TT[m]["bands"][k]["c"] for m in T.MACHINES), n=sum(TT[m]["bands"][k]["n"] for m in T.MACHINES),
              valid=all(TT[m]["bands"][k]["valid"] for m in T.MACHINES)) for k in T.ARMS_RUN}
VB = all(TT[m]["valid"]["B"][1] for m in T.MACHINES)
(H1, W1), (H2, W2) = T.readings(PC, PB, VB)
PPR = {name: T.pool_claim([TT[m]["printed"][name] for m in T.MACHINES]) for name, *_ in T.PRINTED}
STORE = {"X": storeX, "L": storeL}
SEEDS = {"X": seedsX, "L": seedsL}


def runs(m, k):
    return TT[m]["rA"][k]


def succ_seeds(m, k):
    return [s for s, r in sorted(runs(m, k).items()) if T.success(k, r)]


def verdict_of(d, name):
    return T.c2_verdict(d) if name == "C2" else T.verdict(d)


def wil(k, n):
    return tsr.wilson(k, n)


def fw(k, n):
    lo, hi = wil(k, n)
    return f"[{lo:.3f}, {hi:.3f}]"


def band(b):
    return T.band_of(b)


def trs_of(m, k):
    return [r["transition"] for r in runs(m, k).values() if T.ok_r(r) and r["transition"] is not None]


def fired(m, k):
    return [c for _, r in sorted(runs(m, k).items()) for c in (r.get("checks") or []) if c["fired"]]


def per_run_fired(m, k):
    return {s: sum(1 for c in (r.get("checks") or []) if c["fired"]) for s, r in sorted(runs(m, k).items())}


def oc_classes(m, k):
    out = {}
    for s, r in sorted(runs(m, k).items()):
        out.setdefault(T.outcome2(r), []).append(s)
    return out


def acc_range(m, k):
    a = [r["acc"] for r in runs(m, k).values()]
    return min(a), max(a)


def cross(m, a, b):
    """discordant seeds (a only, b only)"""
    ra, rb = runs(m, a), runs(m, b)
    return ([s for s in SEEDS[m][a] if T.success(a, ra[s]) and not T.success(b, rb[s])],
            [s for s in SEEDS[m][a] if T.success(b, rb[s]) and not T.success(a, ra[s])])


# independent cross-check with rlib (re-implemented outcome: bound + one-to-one recomputed map + every stream >= 0.9)
def rl_succ(k, r):
    return rlib.bound_routed(r) if k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8") else rlib.bound(r)


rl_agree = all(rl_succ(k, r) == T.success(k, r) for m in "XL" for k in T.ARMS_RUN for r in runs(m, k).values())
rl_p = all(abs(rlib.mcnemar1(PC[n]["b"], PC[n]["c"]) - PC[n]["p"]) < 1e-15 for n in PC) and all(
    abs(rlib.mcnemar1(TT[m]["claims"][n]["b"], TT[m]["claims"][n]["c"]) - TT[m]["claims"][n]["p"]) < 1e-15
    for m in "XL" for n in PC)
rl_w = all(rlib.wilson(b["c"], b["n"]) == wil(b["c"], b["n"]) for b in list(PB.values()) + [TT[m]["bands"][k] for m in "XL"
                                                                                          for k in T.ARMS_RUN])
trans_ok = all(rlib.transition(r["curve"]) == r["transition"] for m in "XL" for k in T.KEYS for r in runs(m, k).values())

# ── 2. the report regenerated from the records with the test's own report() and diffed against L's log ───────
X_STORE_FOR_ALSO["results/X/split_copy_results.json"] = storeX
WALL = 32174.0                                      # the log's own figure (not derivable from the records)
buf = io.StringIO()
with redirect_stdout(buf):
    T.report(storeL, seedsL, "L", WALL, "split_copy_results.json", "results/X/split_copy_results.json")
GEN = buf.getvalue().split("\n")
while GEN and GEN[-1] == "":
    GEN.pop()
i0 = next(i for i in range(len(LOG) - 1) if LOG[i].startswith("#" * 100) and LOG[i + 1].startswith("PER-SEED RAW RESULTS"))
g0 = GEN.index("#" * 100)
LOGR = LOG[i0:i0 + len(GEN) - g0]
GENR = GEN[g0:]
rep_diff = [(i0 + j + 1, a, b) for j, (a, b) in enumerate(zip(GENR, LOGR)) if a != b]
rep_tail = LOG[i0 + len(GENR):]
rep_tail = [x for x in rep_tail if x.strip()]

# the RUNS lines (one per record) regenerated with record()'s format, the elapsed field excluded
second = next(i for i, x in enumerate(LOG) if x.startswith("=== longrun") and "2026-10-11" in x)
first = next(i for i, x in enumerate(LOG) if x.startswith("=== longrun"))
runs_lines = {}
for i in range(second, i0):
    mm = re.match(r"^   (\S+)\s+seed (\d+)  (.*)  (\d+)s  \(elapsed ([\d.]+)m\)$", LOG[i])
    if mm:
        runs_lines.setdefault((mm.group(1), int(mm.group(2))), []).append((i + 1, mm.group(3), int(mm.group(4)),
                                                                         float(mm.group(5))))


def rec_line(k, rec):
    tr = rec["transition"]
    lab = (T.outcome2(rec) if (T.ARM[k]["part"] == "A" and not T.ARM[k]["ceil"]) else ("BOUND" if conv.bound_r(rec) else "not bound"))
    fd = ([c["step"] for c in rec["checks"] if c["fired"]] if rec.get("checks") is not None else None)
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} "
            f"{lab}" + (f"  {rec.get('trigger_op')}s at {fd}" if fd is not None else ""))


run_bad = []
for k in T.KEYS:
    for s in seedsL[k]:
        rec = storeL["runs"][f"{k}|{s}"]
        got = runs_lines.get((k, s), [])
        if len(got) != 1 or got[0][1] != rec_line(k, rec) or got[0][2] != round(rec["secs"]):
            run_bad.append((k, s, got, rec_line(k, rec), round(rec["secs"])))
n_run_lines = sum(len(v) for v in runs_lines.values())
last_elapsed = max(v[0][3] for v in runs_lines.values())

# ── 3. L's records: seeds, CPU, code, starts, times ─────────────────────────
mL = storeL["meta"]
spec_seeds = {k: list(range(1560, 1570)) if k not in ("CEIL_C", "CEIL_DELTA_D8") else [1560, 1561] for k in T.KEYS}
seeds_ok = (mL["seeds"] == spec_seeds and set(storeL["runs"]) == {f"{k}|{s}" for k in T.KEYS for s in spec_seeds[k]}
            and "L 1560-1569" in SPEC and "1560-1561" in SPEC)
cpus = {r["cpu"] for r in storeL["runs"].values()}
oks = sum(bool(r.get("ok")) for r in storeL["runs"].values())
errs = [k for k, r in storeL["runs"].items() if r.get("error")]
seed_field_ok = all(r["seed"] == int(k.split("|")[1]) and r["arm"] == k.split("|")[0] for k, r in storeL["runs"].items())
resumed = "starts" in mL
headers = [(i + 1, x) for i, x in enumerate(LOG) if x.startswith("=== longrun")]
cached_lines = [i + 1 for i in range(second, i0) if LOG[i].rstrip().endswith(" cached")]
fail_lines = [i + 1 for i in range(second, i0) if "*** FAILED" in LOG[i]]


def strip_doc(src):
    t = ast.parse(src)
    if t.body and isinstance(t.body[0], ast.Expr) and isinstance(getattr(t.body[0], "value", None), ast.Constant):
        t.body = t.body[1:]
    return ast.dump(t)


code_same_1f_59 = strip_doc(show(TEST_C, "test_split_copy.py")) == strip_doc(show(XRES, "test_split_copy.py"))
code_same_59_ff = strip_doc(show(XRES, "test_split_copy.py")) == strip_doc(SRC)
py_changed_1f_59 = [f for f in git("diff", "--name-only", TEST_C, XRES, "--", "*.py").split() if not f.startswith("docs/")]
py_changed_59_ff = [f for f in git("diff", "--name-only", XRES, RES, "--", "*.py").split() if not f.startswith("docs/")]
src59 = show(XRES, "test_split_copy.py").split("\n")
tb = [(int(a), b) for a, b in re.findall(r'line (\d+), in \S+\n\s+(.+)', "\n".join(LOG[first:second]))]
tb_ok = [(n, txt, src59[n - 1].strip()) for n, txt in tb]
tb_match = all(t == s for _, t, s in tb_ok)
first_log_same = show(FIRST_L, "results/L/longrun/split_copy_L.log").split("\n")[:-1] == LOG[:second - 1]
fj = json.loads(show(FIRST_L, "results/L/longrun/split_copy_L.json"))
fexit = show(FIRST_L, "results/L/longrun/split_copy_L.exit").strip()
f_utc = dt.datetime.fromtimestamp(fj["starts"][0]["t"], dt.timezone.utc).replace(tzinfo=None)
f_loc = dt.datetime.strptime(fj["starts"][0]["started"], "%Y-%m-%d %H:%M:%S")
offset_h = round((f_loc - f_utc).total_seconds() / 3600, 3)
readme_utc6 = "UTC−6" in show(RES, "results/README.md")
OFF = dt.timedelta(hours=-offset_h)


def ctime(c):
    return dt.datetime.fromisoformat(git("show", "-s", "--format=%cI", c).strip()).astimezone(dt.timezone.utc).replace(tzinfo=None)


h2 = re.search(r"start (\d+) at (\S+ \S+), git (\S+), boot (\S+)", headers[1][1])
h1 = re.search(r"start (\d+) at (\S+ \S+), git (\S+), boot (\S+)", headers[0][1])
t_start2_loc = dt.datetime.strptime(h2.group(2), "%Y-%m-%d %H:%M:%S")
t_runs_loc = dt.datetime.strptime(mL["started"], "%Y-%m-%d %H:%M:%S")
t_end_loc = t_runs_loc + dt.timedelta(minutes=last_elapsed)
t_start1_loc = dt.datetime.strptime(h1.group(2), "%Y-%m-%d %H:%M:%S")
C59, CFF, CEF = ctime(XRES), ctime(RES), ctime(FOLD_C)
times_ok = (C59 < t_start1_loc + OFF < t_start2_loc + OFF < t_runs_loc + OFF < t_end_loc + OFF < CFF)
wallmin = re.search(r"total wall clock after verification: ([\d.]+) min", "\n".join(LOG))
also_line = next((x for x in LOG[second:] if x.startswith("  --also")), "")
window_git = mL["window"]["git"]
lr_numstat = git("diff", "--numstat", "56c44a7", TEST_C, "--", "longrun.py").split()
fc_last = git("log", "-1", "--format=%h", XRES, "--", "fail_class_v2.py").strip()
lr_last = git("log", "-1", "--format=%h", XRES, "--", "longrun.py").strip()

# CHECK lines of the second start's verification (asserted vs printed)
VER = LOG[second:i0]
printed_checks = []
for j, x in enumerate(VER):
    if re.search(r"printed, not asserted", x):
        hdr = next(VER[q] for q in range(j, -1, -1) if VER[q].startswith("CHECK"))
        res_ = next((VER[q] for q in range(j, max(j - 6, 0), -1) if "->" in VER[q]), "")
        printed_checks.append((second + j + 1, hdr.split(" ")[1], res_.rsplit("->", 1)[-1].strip()))
all_passed2 = sum(1 for x in VER if x.startswith("ALL VERIFICATION CHECKS PASSED"))
some_failed2 = sum(1 for x in VER if x.startswith("SOME VERIFICATION CHECKS FAILED"))
c155_first = [x for x in LOG[first:second] if "--also: machine None" in x and x.rstrip().endswith("WRONG")]
runs_in_first = [x for x in LOG[first:second] if x.startswith("RUNS")]


def logline(frag, start=second, end=None, exact=False):
    end = len(LOG) if end is None else end
    for i in range(start, end):
        if (LOG[i] == frag) if exact else (frag in LOG[i]):
            return i + 1
    return None


# ── printing helpers ─────────────────────────────────────────────────────────
OUT = []


def P(*a):
    OUT.append(" ".join(str(x) for x in a))


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


ROWS = []
COUNT = {}


def row(where, statement, recomputed, status, note=""):
    assert status in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE"), status
    ROWS.append((where, statement, recomputed, status, note))
    COUNT[status] = COUNT.get(status, 0) + 1


def st(c, bad="DISAGREE"):
    return "MATCH" if c else bad


def dl(frag):
    """docstring line number of a fragment in test_split_copy.py at ffb0771 (asserted present)"""
    for i, x in enumerate(SRC_LINES):
        if frag in x:
            return i + 1
    raise AssertionError(f"docstring fragment not found: {frag!r}")


def tl(frag, near=None):
    """tex5 line of a fragment (asserted present; with `near`, the occurrence on that line)"""
    hits = [i + 1 for i, x in enumerate(TEX) if frag in x]
    assert hits, f"tex fragment not found: {frag!r}"
    if near is not None:
        assert near in hits, f"{frag!r} not on tex5:{near} (on {hits})"
        return near
    return hits[0]


def pfmt(p):
    return f"{p:.2g}"


cl = {m: TT[m]["claims"] for m in "XL"}
bd = {m: TT[m]["bands"] for m in "XL"}
L_, X_ = "L", "X"

# ══ DOCSTRING (RESULT (L), RESULT (POOLED, PRIMARY)) vs log and records ═════
D = "doc:"
row(D + str(dl("RESULT (L) — test commit 1f52cd5")), "test commit 1f52cd5",
    f"test_split_copy.py's code (docstring stripped, ast) equal at 1f52cd5, 592c574 and ffb0771: {code_same_1f_59 and code_same_59_ff}; "
    f".py files changed 1f52cd5..592c574: {py_changed_1f_59}; 592c574..ffb0771: {py_changed_59_ff}",
    st(code_same_1f_59 and code_same_59_ff and py_changed_1f_59 == ["test_split_copy.py"]),
    "the run's HEAD is 592c574; after it, ffb0771 changes test_split_copy.py's docstring only, and test_delta_recipe.py (another test, "
    "not imported by test_split_copy) is the only other .py file changed by ffb0771's parent line")
row(D + str(dl("run at 592c574 on 2026-10-11")), "run at 592c574",
    f"meta.git {mL['git']}; log headers (log:{headers[0][0]}, log:{headers[1][0]}) git {h1.group(3)}, {h2.group(3)}; the run "
    f"line 'git ...' (log:{logline('6 worker processes x 1 thread   git')}) {LOG[logline('6 worker processes x 1 thread   git') - 1].split()[-1]}",
    st(mL["git"] == h2.group(3) == XRES))
row(D + str(dl("started 09:31 local = 15:31 UTC")), "on 2026-10-11 (started 09:31 local = 15:31 UTC through longrun.py",
    f"log:{headers[1][0]} 'start {h2.group(1)} at {h2.group(2)}'; = {(t_start2_loc + OFF):%Y-%m-%d %H:%M:%S} UTC at UTC{offset_h:+g}h "
    f"(offset from 84a7e57's longrun json for the FIRST start: local {fj['starts'][0]['started']} vs epoch {f_utc:%H:%M:%S} UTC; "
    f"results/README.md says UTC−6: {readme_utc6})",
    st(h2.group(2).startswith("2026-10-11 09:31") and f"{(t_start2_loc + OFF):%H:%M}" == "15:31"),
    "the second start's own json is not committed; its offset is assumed equal to the first start's")
row(D + str(dl("from 11:12 local after the verification")), "runs from 11:12 local after the verification",
    f"meta.started {mL['started']}", st(mL["started"].startswith("2026-10-11 11:12")))
row(D + str(dl("12th Gen Intel(R) Core(TM) i7-12650H, torch 2.14.1, 6 workers")),
    "12th Gen Intel(R) Core(TM) i7-12650H, torch 2.14.1, 6 workers x 1 thread",
    f"meta cpu {mL['cpu']!r}, torch {mL['torch']}, workers {mL['workers']}; every record's cpu {sorted(cpus)}",
    st(mL["cpu"] == "12th Gen Intel(R) Core(TM) i7-12650H" and mL["torch"] == "2.14.1" and mL["workers"] == 6 and cpus == {mL["cpu"]}))
row(D + str(dl("start, nothing cut, 54/54 runs")), "one start",
    f"the log holds {len(headers)} longrun headers, both 'start 1' (log:{headers[0][0]} {h1.group(2)} boot {h1.group(4)}; "
    f"log:{headers[1][0]} {h2.group(2)} boot {h2.group(4)}); one start ran the runs (no 'starts' in meta: {not resumed}; no cached lines: "
    f"{not cached_lines})", "PARTIAL",
    "true of the run that made the records; the log has two starts (the docstring's next sentence names the earlier), and the "
    "longrun counter was reset between them (both read 'start 1'), so longrun's own record does not count them")
row(D + str(dl("start, nothing cut, 54/54 runs")), "nothing cut, 54/54 runs, none failed",
    f"records {len(storeL['runs'])}, ok {oks}, with an error field {errs}; RUNS lines {n_run_lines} (one per record: "
    f"{all(len(v) == 1 for v in runs_lines.values())}); FAILED lines {fail_lines}; 'nothing is cut' at log:{logline('  nothing is cut')}",
    st(len(storeL["runs"]) == 54 == oks and not errs and n_run_lines == 54 and not fail_lines))
row(D + str(dl("536 min of runs after the verification")), "536 min of runs after the verification",
    f"log 'total wall clock after verification: {wallmin.group(1)} min'; last RUNS line elapsed {last_elapsed}m",
    st(round(float(wallmin.group(1))) == 536))
row(D + str(dl("(exit 0)")), "(exit 0)",
    f"no exit file or longrun json for the second start is committed anywhere; the log ends with the report's last line "
    f"(log:{NLOG}) and has no traceback after log:{headers[1][0]}: "
    f"{not any('Traceback' in x for x in LOG[second:])}", "NOT RECORD-CHECKABLE",
    "consistent with the log, not recorded")
row(D + str(dl("An earlier start on")), "An earlier start on 2026-10-10 stopped at verification with 0 runs (CHECK 155: --also was "
    "given a directory, not X's file); its log is kept at the top of results/L/split_copy_L.log",
    f"log:{headers[0][0]} {h1.group(2)} local ({(t_start1_loc + OFF):%H:%M} UTC, after 592c574 at {C59:%H:%M} UTC); CHECK 155 "
    f"'--also: machine None ... -> WRONG' ({len(c155_first)} line); 'SOME VERIFICATION CHECKS FAILED' and AssertionError; RUNS "
    f"blocks in it {len(runs_in_first)}; log:1-{second - 1} equal to 84a7e57's results/L/longrun/split_copy_L.log: "
    f"{first_log_same} (its .exit {fexit})",
    st(len(c155_first) == 1 and not runs_in_first and first_log_same and fexit == "1"))
row(D + str(dl("every inherited and own CHECK passed, asserted")), "Verification: every inherited and own CHECK passed, asserted",
    f"second start: 'ALL VERIFICATION CHECKS PASSED' x{all_passed2}, 'SOME ... FAILED' x{some_failed2}; printed, not asserted: "
    + "; ".join(f"CHECK {c} -> {r} (log:{n})" for n, c, r in printed_checks),
    "DISAGREE",
    "every CHECK passed, but not every one was asserted: the inherited CHECKs 135 and 141 (asserted on X, printed on L) and the "
    "own CHECK 150 were printed, not asserted, and all three DIFFER; the docstring names only 150 (next lines)")
row(D + str(dl("CHECK 130 asserted (L's")), "test_muon_recipe's CHECK 130 asserted (L's test_early_recipe file was written on this CPU)",
    f"log:{logline('CHECK 130 is asserted')} 'CHECK 130 is asserted'; log:{logline('CHECK 130 code identity')}: two runs IDENTICAL "
    f"({sum(1 for x in LOG[second:i0] if x.strip().startswith(('WIN_M|300', 'WIN16_M|340')) and x.endswith('IDENTICAL'))})",
    st(sum(1 for x in LOG[second:i0] if x.strip().startswith(("WIN_M|300", "WIN16_M|340")) and x.endswith("IDENTICAL")) == 2))
n146 = [x for x in LOG[second:i0] if re.match(r"\s+WIN3_(SPLIT|RESET)_D8\|1541:", x)]
row(D + str(dl("CHECK 146 reproduced L's")), "CHECK 146 reproduced L's test_window_gate records at seed 1541 bit for bit through 7200 "
    "(asserted: same CPU)", f"log:{logline('CHECK 146 WIN3_SPLIT_D8 and WIN3_RESET_D8 at seed 1541')}: {len(n146)} lines, all "
    f"IDENTICAL {all(x.endswith('IDENTICAL') for x in n146)}; asserted on a matching CPU (window file CPU = this CPU)",
    st(len(n146) == 2 and all(x.endswith("IDENTICAL") for x in n146)))
row(D + str(dl("CHECK 150 (F's S68_SINGLE_S8|300")), "CHECK 150 ... differs here and was printed, not asserted",
    "; ".join(f"CHECK {c} -> {r} (log:{n})" for n, c, r in printed_checks if c == "150"),
    st(any(c == "150" and r == "DIFFERS" for _, c, r in printed_checks)))
c151a = next(x for x in LOG[second:i0] if re.match(r"\s+65 records; disagreements 0", x))
c151b = next(x for x in LOG[i0:] if "CHECK 151(b)" in x)
row(D + str(dl("CHECK 151(a): 65 Part C/D records, v1 = v2; 151(b)")), "CHECK 151(a): 65 Part C/D records, v1 = v2; 151(b): 30 Part A "
    "records, no disagreement", f"log '{c151a.strip()}'; '{c151b.strip()}'; recomputed v1 vs v2 on the 30 Part A records: "
    f"{sum(1 for k in ('WIN3_SPLIT_D8', 'WIN3_RESET_D8', 'WIN3_SLOW_D8') for r in runs(L_, k).values() if tsr.outcome(r) != T.outcome2(r))} disagreements",
    st("30 records, disagreements 0" in c151b and all(tsr.outcome(r) == T.outcome2(r) for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8",
                                                                                                   "WIN3_SLOW_D8") for r in runs(L_, k).values())))
vA, vB = TL["valid"]["A"], TL["valid"]["B"]
ceil_tr = {k: [runs(L_, k)[s]["transition"] for s in seedsL[k]] for k in ("CEIL_C", "CEIL_DELTA_D8")}
row(D + str(dl("- Validity: CEIL_C bound 2/2 (4800, 3600); CEIL_DELTA_D8 bound 2/2 (1200, 1200). Parts A and B VALID on L")),
    "Validity: CEIL_C 2/2 (4800, 3600); CEIL_DELTA_D8 2/2 (1200, 1200); Parts A and B VALID on L",
    f"CEIL_C {vA[0]}/2 {ceil_tr['CEIL_C']}; CEIL_DELTA_D8 {vB[0]}/2 {ceil_tr['CEIL_DELTA_D8']}; VALID A {vA[1]}, B {vB[1]}",
    st(vA == (2, True) and vB == (2, True) and ceil_tr == {"CEIL_C": [4800, 3600], "CEIL_DELTA_D8": [1200, 1200]}))
spl = trs_of(L_, "WIN3_SPLIT_D8")
spl_fail = {s: T.outcome2(r) for s, r in runs(L_, "WIN3_SPLIT_D8").items() if not T.success("WIN3_SPLIT_D8", r)}
row(D + str(dl("- Counts: WIN3_SPLIT_D8 BOUND ROUTED 8/10 (transition median 19200")),
    "WIN3_SPLIT_D8 BOUND ROUTED 8/10 (transition median 19200, 8400-25200; s1566, s1569 MERGED 2 share)",
    f"{bd[L_]['WIN3_SPLIT_D8']['c']}/10; median {statistics.median(spl):.0f}, {min(spl)}-{max(spl)}; failures {spl_fail}",
    st(bd[L_]["WIN3_SPLIT_D8"]["c"] == 8 and statistics.median(spl) == 19200 and (min(spl), max(spl)) == (8400, 25200)
       and spl_fail == {1566: "MERGED (2 share)", 1569: "MERGED (2 share)"}))
row(D + str(dl("WIN3_RESET_D8 3/10 (s1560, s1565, s1568); WIN3_SLOW_D8 3/10")),
    "WIN3_RESET_D8 3/10 (s1560, s1565, s1568); WIN3_SLOW_D8 3/10 (s1560, s1565, s1568)",
    f"RESET {succ_seeds(L_, 'WIN3_RESET_D8')}; SLOW {succ_seeds(L_, 'WIN3_SLOW_D8')}",
    st(succ_seeds(L_, "WIN3_RESET_D8") == [1560, 1565, 1568] == succ_seeds(L_, "WIN3_SLOW_D8")))
sdr, shr = acc_range(L_, "SINGLE_DELTA_D8"), acc_range(L_, "SINGLE_HEBB_D8")
row(D + str(dl("accuracy 0.13-0.55); SINGLE_HEBB_D8 BOUND 0/10 (0.08-0.18)")),
    "SINGLE_DELTA_D8 BOUND 0/10 (final accuracy 0.13-0.55); SINGLE_HEBB_D8 BOUND 0/10 (0.08-0.18)",
    f"SINGLE_DELTA {bd[L_]['SINGLE_DELTA_D8']['c']}/10, {sdr[0]:.4f}-{sdr[1]:.4f}; SINGLE_HEBB {bd[L_]['SINGLE_HEBB_D8']['c']}/10, "
    f"{shr[0]:.4f}-{shr[1]:.4f}",
    st(bd[L_]["SINGLE_DELTA_D8"]["c"] == 0 == bd[L_]["SINGLE_HEBB_D8"]["c"] and f"{sdr[0]:.2f}-{sdr[1]:.2f}" == "0.13-0.55"
       and f"{shr[0]:.2f}-{shr[1]:.2f}" == "0.08-0.18"))


def claim_doc(m, name, frag, pdoc):
    d = cl[m][name] if m in "XL" else PC[name]
    ln = dl(frag)
    lab = {"X": "machine X block", "L": "per machine L block", "P": "POOLED block"}[m]
    if name == "C2":
        rec = (f"{d['new']}/{d['n']} vs {d['old']}/{d['n']}; reset only {d['b']}, slow only {d['c']}; d = {d['b'] - d['c']}; "
               f"{verdict_of(d, name)}")
        ok_ = frag.startswith(f"C2 {verdict_of(d, name)}") and f"{d['new']}/{d['n']}" in frag and f"{d['old']}/{d['n']}" in frag and \
            f"reset only {d['b']}, slow only {d['c']}, d = {d['b'] - d['c']}" in frag
    else:
        rec = (f"{d['new']}/{d['n']} vs {d['old']}/{d['n']}; b {d['b']}, c {d['c']}; one-sided p = {d['p']:.6g}; "
               f"{verdict_of(d, name)}")
        ok_ = frag.startswith(f"{name} {verdict_of(d, name)}") and f"{d['new']}/{d['n']}" in frag and f"{d['old']}/{d['n']}" in frag \
            and f"{d['b']}" in frag and pfmt(d["p"]) == pdoc
    return row(D + str(ln), frag + f" [{lab}]", rec, st(ok_))


claim_doc("L", "C1", "C1 SHOWN: 8/10 vs 3/10, split only 5, reset only 0, p = 0.031.", "0.031")
claim_doc("L", "C2", "C2 HOLDS: 3/10 vs 3/10, reset only 0, slow only 0, d = 0 (bound d <= 2).", None)
claim_doc("L", "C3", "C3 SHOWN: 8/10 vs 0/10, 8 vs 0, p = 0.0039.", "0.0039")


def bands_doc(frag, which, n_):
    ln = dl(frag)
    bb = PB if which == "P" else bd[which]
    s = "; ".join(f"{k} {band(bb[k])} {fw(bb[k]['c'], bb[k]['n'])} ({bb[k]['c']}/{bb[k]['n']})" for k in T.ARMS_RUN)
    text = " ".join(SRC_LINES[ln - 1:ln + 1])
    want = [f"{k} {band(bb[k])} {fw(bb[k]['c'], bb[k]['n']).replace('[0.000,', '[0,')}" for k in T.ARMS_RUN]
    flat = re.sub(r"\s+", " ", text)
    row(D + f"{ln}-{ln + 1}", flat.strip(), s, st(all(w in flat for w in want)))


bands_doc("  Bands: WIN3_SPLIT_D8 MAJORITY [0.490, 0.943]", "L", 10)
pr = TL["printed"]["one delta channel vs one Hebbian channel"]
fs = fired(L_, "WIN3_SPLIT_D8")
frs = fired(L_, "WIN3_RESET_D8")
row(D + str(dl("- Printed: SINGLE_DELTA_D8 vs SINGLE_HEBB_D8 0 vs 0 (both 0/10). Splits: 19")),
    "Printed: SINGLE_DELTA_D8 vs SINGLE_HEBB_D8 0 vs 0 (both 0/10). Splits: 19 in 10 runs, every target labelled ok (19/19). Resets: 26 "
    "in 10 runs, all targets labelled ok",
    f"{pr['b']} vs {pr['c']} ({pr['new']}/{pr['n']}, {pr['old']}/{pr['n']}); splits {len(fs)} in {len(runs(L_, 'WIN3_SPLIT_D8'))} runs "
    f"{list(per_run_fired(L_, 'WIN3_SPLIT_D8').values())}, key_ok {sum(c['key_ok'] for c in fs)}/{len(fs)}; resets {len(frs)} "
    f"{list(per_run_fired(L_, 'WIN3_RESET_D8').values())}, key_ok {sum(c['key_ok'] for c in frs)}/{len(frs)}",
    st(pr["b"] == pr["c"] == 0 and len(fs) == 19 == sum(c["key_ok"] for c in fs) and len(frs) == 26 == sum(c["key_ok"] for c in frs)))
ocR, ocS = oc_classes(L_, "WIN3_RESET_D8"), oc_classes(L_, "WIN3_SLOW_D8")
row(D + str(dl("(s1560, s1565, s1568). The failures of RESET and SLOW are all merges")),
    "The reset seeds that bound are the slow seeds that bound (s1560, s1565, s1568). The failures of RESET and SLOW are all merges "
    "(RESET 2 share x6, 3 share x1; SLOW 2 share x5, 3 share x2)",
    f"RESET {{{', '.join(f'{k}: {len(v)}' for k, v in ocR.items())}}}; SLOW {{{', '.join(f'{k}: {len(v)}' for k, v in ocS.items())}}}",
    st(succ_seeds(L_, "WIN3_RESET_D8") == succ_seeds(L_, "WIN3_SLOW_D8") and len(ocR.get("MERGED (2 share)", [])) == 6 and
       len(ocR.get("MERGED (3 share)", [])) == 1 and len(ocS.get("MERGED (2 share)", [])) == 5 and len(ocS.get("MERGED (3 share)", [])) == 2
       and set(ocR) | set(ocS) <= {"BOUND ROUTED", "MERGED (2 share)", "MERGED (3 share)"}))
pool_hdr = next(x for x in LOG[i0:] if x.startswith("POOLED (PRIMARY)"))
row(D + str(dl("RESULT (POOLED, PRIMARY) — X (results/X/split_copy_results.json)")),
    "RESULT (POOLED, PRIMARY) — X (results/X/split_copy_results.json) + L, b and c summed ... printed by L's run with --also "
    "results/X/split_copy_results.json; X and L VALID on both parts",
    f"log:{logline('POOLED (PRIMARY)', i0)} '{pool_hdr[:150]}...'; run header {also_line.strip()!r}; X valid {TX['valid']}, "
    f"L valid {TL['valid']}; X file at ffb0771 = 592c574's (asserted)",
    st("results/X/split_copy_results.json" in pool_hdr and "machine X" in also_line and all(v[1] for m in "XL" for v in TT[m]["valid"].values())))
claim_doc("P", "C1", "C1 SHOWN: WIN3_SPLIT_D8 18/20 vs WIN3_RESET_D8 6/20, split only 12, reset only 0, p = 0.00024.", "0.00024")
claim_doc("P", "C2", "C2 HOLDS: WIN3_RESET_D8 6/20 vs WIN3_SLOW_D8 8/20, reset only 0, slow only 2, d = -2 (bound d <= 2).", None)
claim_doc("P", "C3", "C3 SHOWN: WIN3_SPLIT_D8 18/20 vs SINGLE_DELTA_D8 0/20, 18 vs 0, p = 3.8e-06.", "3.8e-06")
bands_doc("  Pooled bands: WIN3_SPLIT_D8 RELIABLE [0.699, 0.972]", "P", 20)
row(D + f"{dl('READINGS (pooled): \"the copy, not the reset\"')}-{dl('machines). Printed, not a claim')}",
    "READINGS (pooled): both apply (C1 SHOWN; C2 HOLDS / C3 SHOWN; SINGLE_DELTA_D8 0/20 below MAJORITY; CEIL_DELTA_D8 bound on both "
    "machines). Printed, not a claim: SINGLE_DELTA_D8 vs SINGLE_HEBB_D8 0/20 vs 0/20",
    f"reading 1 {H1} ({W1}); reading 2 {H2} ({W2}); printed pooled {PPR['one delta channel vs one Hebbian channel']['new']}/20 vs "
    f"{PPR['one delta channel vs one Hebbian channel']['old']}/20",
    st(H1 and H2 and PPR["one delta channel vs one Hebbian channel"]["new"] == 0 == PPR["one delta channel vs one Hebbian channel"]["old"]))
N_DOC = len(ROWS)

# ══ the draft (tex5) ═════════════════════════════════════════════════════════
pC1, pC2, pC3 = PC["C1"], PC["C2"], PC["C3"]
lC1, lC2, lC3 = cl[L_]["C1"], cl[L_]["C2"], cl[L_]["C3"]
xC1, xC2, xC3 = cl[X_]["C1"], cl[X_]["C2"], cl[X_]["C3"]
alls = all(verdict_of(cl[m][n], n) in ("SHOWN", "HOLDS") for m in "XL" for n in PC) and all(verdict_of(PC[n], n) in ("SHOWN", "HOLDS") for n in PC)
cnt = lambda m, k: f"{bd[m][k]['c']}/{bd[m][k]['n']}"  # noqa: E731
pc_ = lambda k: f"{PB[k]['c']}/{PB[k]['n']}"  # noqa: E731
summary = (f"pooled C1 {pC1['new']}/20 vs {pC1['old']}/20, {pC1['b']} vs {pC1['c']}, p = {pC1['p']:.3g} {T.verdict(pC1)}; C2 "
           f"{pC2['new']}/20 vs {pC2['old']}/20, {pC2['b']} vs {pC2['c']}, d = {pC2['b'] - pC2['c']} {T.c2_verdict(pC2)}; C3 {pC3['new']}/20 "
           f"vs {pC3['old']}/20, {pC3['b']} vs {pC3['c']}, p = {pC3['p']:.3g} {T.verdict(pC3)}; readings {H1}, {H2}")
n = tl("the copy with its reset binds eight streams $18/20$ against the reset alone $6/20$ (12 vs 0)")
row(f"tex5:{n}", "abstract: the copy with its reset binds eight streams 18/20 against the reset alone 6/20 (12 vs 0), the reset alone is "
    "within the bound of no trigger (6/20 against 8/20, 0 vs 2), one delta-rule channel binds 0/20 where the recipe binds 18/20 (18 vs 0); "
    "both pre-fixed readings apply; 'a pre-registered result on two machines with disjoint seeds'", summary,
    st(pC1["new"] == 18 and pC1["old"] == 6 and (pC1["b"], pC1["c"]) == (12, 0) and (pC2["new"], pC2["old"]) == (6, 8) and
       (pC2["b"], pC2["c"]) == (0, 2) and T.c2_verdict(pC2) == "HOLDS" and (pC3["new"], pC3["old"], pC3["b"], pC3["c"]) == (18, 0, 18, 0)
       and H1 and H2))
n = tl("(\\texttt{test\\_split\\_copy}, complete on both machines)")
row(f"tex5:{n}", "Revision 9 paragraph: test_split_copy, complete on both machines",
    f"X {len(storeX['runs'])} records ok {sum(bool(r.get('ok')) for r in storeX['runs'].values())}; L {len(storeL['runs'])} ok {oks}",
    st(len(storeX["runs"]) == 54 == sum(bool(r.get("ok")) for r in storeX["runs"].values()) and oks == 54))
n = tl("C1, C3 SHOWN and C2 HOLDS pooled and on each machine")
row(f"tex5:{n}", "index row (Table 1): C1, C3 SHOWN and C2 HOLDS pooled and on each machine; both readings",
    "; ".join(f"{m}: " + ", ".join(f"{nm} {verdict_of(cl[m][nm], nm)}" for nm in PC) for m in "XL") + "; pooled: "
    + ", ".join(f"{nm} {verdict_of(PC[nm], nm)}" for nm in PC) + f"; readings {H1}, {H2}", st(alls and H1 and H2))
n = tl("\\texttt{split\\_copy}, X & $S{=}8$, $k{=}16$, 560--569")
row(f"tex5:{n}", "Table 3 (tab:copy), X row (context): 10/10, reset 3/10, none 5/10; C1 7 vs 0, p = 0.0078 SHOWN; C2 reset vs none 0 vs 2, HOLDS",
    f"{cnt(X_, 'WIN3_SPLIT_D8')}, {cnt(X_, 'WIN3_RESET_D8')}, {cnt(X_, 'WIN3_SLOW_D8')}; C1 {xC1['b']} vs {xC1['c']} p = {xC1['p']:.4g} "
    f"{T.verdict(xC1)}; C2 {xC2['b']} vs {xC2['c']} {T.c2_verdict(xC2)}",
    st((bd[X_]["WIN3_SPLIT_D8"]["c"], bd[X_]["WIN3_RESET_D8"]["c"], bd[X_]["WIN3_SLOW_D8"]["c"]) == (10, 3, 5) and (xC1["b"], xC1["c"]) == (7, 0)
       and pfmt(xC1["p"]) == "0.0078" and (xC2["b"], xC2["c"]) == (0, 2) and T.c2_verdict(xC2) == "HOLDS"))
n = tl("\\texttt{split\\_copy}, L & $S{=}8$, $k{=}16$, 1560--1569")
row(f"tex5:{n}", "Table 3, L row: seeds 1560-1569; copy + reset 8/10, reset only 3/10, none 3/10; C1 5 vs 0, p = 0.031 SHOWN; C2 0 vs 0, HOLDS",
    f"{cnt(L_, 'WIN3_SPLIT_D8')}, {cnt(L_, 'WIN3_RESET_D8')}, {cnt(L_, 'WIN3_SLOW_D8')}; C1 {lC1['b']} vs {lC1['c']} p = {lC1['p']:.4g} "
    f"{T.verdict(lC1)}; C2 {lC2['b']} vs {lC2['c']} {T.c2_verdict(lC2)}",
    st((bd[L_]["WIN3_SPLIT_D8"]["c"], bd[L_]["WIN3_RESET_D8"]["c"], bd[L_]["WIN3_SLOW_D8"]["c"]) == (8, 3, 3) and (lC1["b"], lC1["c"]) == (5, 0)
       and pfmt(lC1["p"]) == "0.031" and T.verdict(lC1) == "SHOWN" and (lC2["b"], lC2["c"]) == (0, 0) and T.c2_verdict(lC2) == "HOLDS"))
n = tl("\\texttt{split\\_copy}, pooled (primary) & 20 pairs, disjoint seeds")
row(f"tex5:{n}", "Table 3, pooled row: 18/20, 6/20, 8/20; C1 12 vs 0, p = 0.00024 SHOWN; C2 0 vs 2, HOLDS; 'the copy, not the reset'",
    f"{pc_('WIN3_SPLIT_D8')}, {pc_('WIN3_RESET_D8')}, {pc_('WIN3_SLOW_D8')}; C1 {pC1['b']} vs {pC1['c']} p = {pC1['p']:.4g} "
    f"{T.verdict(pC1)}; C2 {pC2['b']} vs {pC2['c']} {T.c2_verdict(pC2)}; reading 1 {H1}; X and L seed sets disjoint "
    f"{not (set(range(560, 570)) & set(range(1560, 1570)))}",
    st((PB["WIN3_SPLIT_D8"]["c"], PB["WIN3_RESET_D8"]["c"], PB["WIN3_SLOW_D8"]["c"]) == (18, 6, 8) and (pC1["b"], pC1["c"]) == (12, 0)
       and pfmt(pC1["p"]) == "0.00024" and (pC2["b"], pC2["c"]) == (0, 2) and H1))
n = tl("it bound no more often than the arm with no trigger in \\texttt{test\\_split\\_copy}")
row(f"tex5:{n}", "Table 3 caption: in test_split_copy the reset-only arm fired on labelled targets and bound no more often than the arm "
    "with no trigger; the pooled claims are primary and its readings pooled only",
    f"RESET vs SLOW: X {cnt(X_, 'WIN3_RESET_D8')} vs {cnt(X_, 'WIN3_SLOW_D8')}, L {cnt(L_, 'WIN3_RESET_D8')} vs {cnt(L_, 'WIN3_SLOW_D8')}, "
    f"pooled {pc_('WIN3_RESET_D8')} vs {pc_('WIN3_SLOW_D8')}; resets on labelled targets X {sum(c['key_ok'] for c in fired(X_, 'WIN3_RESET_D8'))}/"
    f"{len(fired(X_, 'WIN3_RESET_D8'))}, L {sum(c['key_ok'] for c in frs)}/{len(frs)}",
    st(all(bd[m]["WIN3_RESET_D8"]["c"] <= bd[m]["WIN3_SLOW_D8"]["c"] for m in "XL") and all(c["key_ok"] for m in "XL" for c in fired(m, "WIN3_RESET_D8"))))

# tex5:201, the pre-registered look
L201 = tl("On L (seeds 1560--1569, 54 runs, every CHECK asserted on its own records, 536 min of runs on six workers)")
row(f"tex5:{L201}", "On L: seeds 1560-1569, 54 runs", f"meta seeds = spec ({seeds_ok}); {len(storeL['runs'])} records",
    st(seeds_ok and len(storeL["runs"]) == 54))
row(f"tex5:{L201}", "On L: 'every CHECK asserted on its own records'",
    "printed, not asserted on L: " + "; ".join(f"CHECK {c} -> {r} (log:{nn})" for nn, c, r in printed_checks)
    + "; asserted against L's own files: 130 (2 IDENTICAL), 146 (2 IDENTICAL) and the rest of the chain",
    "PARTIAL",
    "true of the CHECKs that compare with L's own records; but CHECKs 135 and 141 (inherited, against 2.10GHz Xeon records) and the "
    "test's own CHECK 150 (F's Xeon record) were printed, not asserted, and DIFFER on L; the same sentence's X clause counts its "
    "printed CHECKs ('except two ... printed rather than asserted'), the L clause does not")
row(f"tex5:{L201}", "On L: 536 min of runs on six workers", f"log {wallmin.group(1)} min; meta.workers {mL['workers']}",
    st(round(float(wallmin.group(1))) == 536 and mL["workers"] == 6))
row(f"tex5:{L201}", "On L: C1 SHOWN, 8/10 against 3/10 (5 vs 0, p = 0.031); C2 HOLDS, 3/10 against 3/10 (0 vs 0)",
    f"C1 {lC1['new']}/10 vs {lC1['old']}/10, {lC1['b']} vs {lC1['c']}, p = {lC1['p']:.5g} {T.verdict(lC1)}; C2 {lC2['new']}/10 vs "
    f"{lC2['old']}/10, {lC2['b']} vs {lC2['c']} {T.c2_verdict(lC2)}",
    st((lC1["new"], lC1["old"], lC1["b"], lC1["c"]) == (8, 3, 5, 0) and pfmt(lC1["p"]) == "0.031" and (lC2["new"], lC2["old"], lC2["b"], lC2["c"]) == (3, 3, 0, 0)))
spf = {s: (T.outcome2(r), sum(1 for c in r["checks"] if c["fired"])) for s, r in runs(L_, "WIN3_SPLIT_D8").items() if not T.success("WIN3_SPLIT_D8", r)}
row(f"tex5:{L201}", "On L: the two split failures are merges of two streams after three splits", f"{spf}",
    st(len(spf) == 2 and all(v == ("MERGED (2 share)", 3) for v in spf.values())))
row(f"tex5:{L201}", "On L: all 19 splits and 26 resets on labelled targets; both perfect gates 2/2",
    f"splits {sum(c['key_ok'] for c in fs)}/{len(fs)}, resets {sum(c['key_ok'] for c in frs)}/{len(frs)}; CEIL_C {vA[0]}/2, CEIL_DELTA_D8 {vB[0]}/2",
    st(len(fs) == 19 == sum(c["key_ok"] for c in fs) and len(frs) == 26 == sum(c["key_ok"] for c in frs) and vA[0] == vB[0] == 2))
row(f"tex5:{L201}", "Pooled over the 20 pairs (primary): C1 SHOWN, 18/20 against 6/20, 12 vs 0, p = 0.00024; C2 HOLDS, 6/20 against 8/20, "
    "0 vs 2 (d = -2); the reading 'the copy, not the reset' applies",
    f"C1 {pC1['new']}/{pC1['n']} vs {pC1['old']}/{pC1['n']}, {pC1['b']} vs {pC1['c']}, p = {pC1['p']:.6g} {T.verdict(pC1)}; C2 "
    f"{pC2['new']}/{pC2['n']} vs {pC2['old']}/{pC2['n']}, {pC2['b']} vs {pC2['c']}, d = {pC2['b'] - pC2['c']} {T.c2_verdict(pC2)}; reading 1 {H1}",
    st((pC1["new"], pC1["old"], pC1["b"], pC1["c"], pC1["n"]) == (18, 6, 12, 0, 20) and pfmt(pC1["p"]) == "0.00024" and
       (pC2["new"], pC2["old"], pC2["b"], pC2["c"]) == (6, 8, 0, 2) and T.c2_verdict(pC2) == "HOLDS" and H1))
rsL, ssL = set(succ_seeds(L_, "WIN3_RESET_D8")), set(succ_seeds(L_, "WIN3_SLOW_D8"))
rsX, ssX = set(succ_seeds(X_, "WIN3_RESET_D8")), set(succ_seeds(X_, "WIN3_SLOW_D8"))
row(f"tex5:{tl('The reset alone bound the same seeds as no trigger on L and a subset of them on X')}",
    "The reset alone bound the same seeds as no trigger on L and a subset of them on X",
    f"L RESET {sorted(rsL)} SLOW {sorted(ssL)}; X RESET {sorted(rsX)} SLOW {sorted(ssX)}", st(rsL == ssL and rsX < ssX))
x201 = tl("C1, the split beats the reset alone, SHOWN, $10/10$ against $3/10$, 7 vs 0, $p = 0.0078$")
row(f"tex5:{x201}", "X clause (context): C1 SHOWN 10/10 against 3/10, 7 vs 0, p = 0.0078; C2 HOLDS 3/10 against 5/10, 0 vs 2 (every seed the "
    "reset bound also bound without it); all 22 splits and 27 resets on labelled targets; validity 2/2 (4800, 3600; 1200, 1200)",
    f"C1 {xC1['new']}/10 vs {xC1['old']}/10, {xC1['b']} vs {xC1['c']}, p = {xC1['p']:.4g}; C2 {xC2['new']}/10 vs {xC2['old']}/10, "
    f"{xC2['b']} vs {xC2['c']}; RESET seeds subset of SLOW {rsX <= ssX}; splits {sum(c['key_ok'] for c in fired(X_, 'WIN3_SPLIT_D8'))}/"
    f"{len(fired(X_, 'WIN3_SPLIT_D8'))}, resets {sum(c['key_ok'] for c in fired(X_, 'WIN3_RESET_D8'))}/{len(fired(X_, 'WIN3_RESET_D8'))}; "
    f"CEIL_C {[runs(X_, 'CEIL_C')[s]['transition'] for s in seedsX['CEIL_C']]}, CEIL_DELTA {[runs(X_, 'CEIL_DELTA_D8')[s]['transition'] for s in seedsX['CEIL_DELTA_D8']]}",
    st((xC1["new"], xC1["old"], xC1["b"], xC1["c"]) == (10, 3, 7, 0) and (xC2["new"], xC2["old"], xC2["b"], xC2["c"]) == (3, 5, 0, 2)
       and rsX <= ssX and len(fired(X_, "WIN3_SPLIT_D8")) == 22 == sum(c["key_ok"] for c in fired(X_, "WIN3_SPLIT_D8"))
       and len(fired(X_, "WIN3_RESET_D8")) == 27 == sum(c["key_ok"] for c in fired(X_, "WIN3_RESET_D8"))
       and [runs(X_, "CEIL_C")[s]["transition"] for s in seedsX["CEIL_C"]] == [4800, 3600]
       and [runs(X_, "CEIL_DELTA_D8")[s]["transition"] for s in seedsX["CEIL_DELTA_D8"]] == [1200, 1200]))

# tex5:246 (C3), 256, 286/289 (Table 4), 319, 343, 358, 362, 382, 472, 495, 499
sdx, sdl = acc_range(X_, "SINGLE_DELTA_D8"), acc_range(L_, "SINGLE_DELTA_D8")
pool_lo, pool_hi = min(sdx[0], sdl[0]), max(sdx[1], sdl[1])
cd_tr = {m: [runs(m, "CEIL_DELTA_D8")[s]["transition"] for s in SEEDS[m]["CEIL_DELTA_D8"]] for m in "XL"}
n = tl("is SHOWN pooled over both machines: the recipe $18/20$ against one delta channel $0/20$")
row(f"tex5:{n}", "C3 SHOWN pooled: the recipe 18/20 against one delta channel 0/20 (18 vs 0, p = 3.8e-6; X 10 vs 0, L 8 vs 0)",
    f"C3 pooled {pC3['new']}/20 vs {pC3['old']}/20, {pC3['b']} vs {pC3['c']}, p = {pC3['p']:.4g} {T.verdict(pC3)}; X {xC3['b']} vs {xC3['c']}, "
    f"L {lC3['b']} vs {lC3['c']}",
    st((pC3["new"], pC3["old"], pC3["b"], pC3["c"]) == (18, 0, 18, 0) and pfmt(pC3["p"]) == "3.8e-06" and (xC3["b"], xC3["c"]) == (10, 0)
       and (lC3["b"], lC3["c"]) == (8, 0)))
row(f"tex5:{n}", "the channel's final accuracy 0.08-0.76",
    f"SINGLE_DELTA_D8 final accuracy X {sdx[0]:.4f}-{sdx[1]:.4f}, L {sdl[0]:.4f}-{sdl[1]:.4f}; pooled {pool_lo:.2f}-{pool_hi:.2f}",
    st(f"{pool_lo:.2f}-{pool_hi:.2f}" == "0.08-0.76"),
    "the pooled range equals X's (L's 0.13-0.55 lies inside it)")
row(f"tex5:{n}", "one Hebbian channel 0/20 (printed, not claimed) and the delta perfect gate 4/4 at 1200 on both machines; the reading "
    "'the partition is necessary ...' applies",
    f"SINGLE_HEBB pooled {pc_('SINGLE_HEBB_D8')}; CEIL_DELTA_D8 transitions {cd_tr}; reading 2 {H2} ({W2})",
    st(PB["SINGLE_HEBB_D8"]["c"] == 0 and PB["SINGLE_HEBB_D8"]["n"] == 20 and all(t == 1200 for v in cd_tr.values() for t in v)
       and sum(len(v) for v in cd_tr.values()) == 4 and H2))
n = tl("the pre-registered support for the partition is C3 on X")
row(f"tex5:{n}", "'the pre-registered support for the partition is C3 on X'",
    f"C3 SHOWN on X ({xC3['b']} vs {xC3['c']}), on L ({lC3['b']} vs {lC3['c']}, p = {lC3['p']:.4g}) and pooled ({pC3['b']} vs {pC3['c']}); "
    f"reading 2 {H2}", "PARTIAL",
    "true of X but stale since ef6d690: the support is now C3 pooled (primary) and on each machine, as tex5:246 and tex5:362 say")
n = tl("sits below the noisy copy's $18/20$ of \\texttt{test\\_split\\_copy} on other seeds")
row(f"tex5:{n}", "the noisy copy's 18/20 of test_split_copy on other seeds", f"WIN3_SPLIT_D8 pooled {pc_('WIN3_SPLIT_D8')}",
    st(PB["WIN3_SPLIT_D8"]["c"] == 18))
n = tl("$S{=}8$, $P{=}4$, $k{=}16$, conv. & $4/4$ & $0/40$ & $15/40$; $8/20^{s}$")
row(f"tex5:{n}", "Table 4 (tab:synthesis), S=8 row, the ^s cells: window gate + SLOW 8/20; + split 18/20; one Hebbian channel 0/20; one "
    "delta channel 0/20", f"WIN3_SLOW {pc_('WIN3_SLOW_D8')}, WIN3_SPLIT {pc_('WIN3_SPLIT_D8')}, SINGLE_HEBB {pc_('SINGLE_HEBB_D8')}, "
    f"SINGLE_DELTA {pc_('SINGLE_DELTA_D8')}",
    st((PB["WIN3_SLOW_D8"]["c"], PB["WIN3_SPLIT_D8"]["c"], PB["SINGLE_HEBB_D8"]["c"], PB["SINGLE_DELTA_D8"]["c"]) == (8, 18, 0, 0)
       and all(PB[k]["n"] == 20 for k in PB)),
    "the row's 4/4 perfect gate, 0/40, 15/40, 38/40 and 9/10 vs 10/10 are not split_copy's (its own CEIL_C pooled is "
    f"{sum(TT[m]['valid']['A'][0] for m in 'XL')}/4)")
n = tl("with \\texttt{test\\_split\\_copy} (pooled over X and L, 20 pairs)")
row(f"tex5:{n}", "Table 4 caption: test_split_copy pooled over X and L, 20 pairs; ^s test_split_copy, pooled",
    f"pairs per pooled claim {pC1['n']}, {pC2['n']}, {pC3['n']}", st(pC1["n"] == pC2["n"] == pC3["n"] == 20))
n = tl("the pre-registered version of that comparison is SHOWN pooled over both machines")
row(f"tex5:{n}", "one delta channel ... none at eight, and the pre-registered version of that comparison is SHOWN pooled over both machines",
    f"C3 pooled {T.verdict(pC3)}; SINGLE_DELTA pooled {pc_('SINGLE_DELTA_D8')}", st(T.verdict(pC3) == "SHOWN" and PB["SINGLE_DELTA_D8"]["c"] == 0))
n = tl("the copy against the reset is pre-registered and shown on both machines")
row(f"tex5:{n}", "the copy against the reset is pre-registered and shown on both machines",
    f"C1 X {T.verdict(xC1)}, L {T.verdict(lC1)}, pooled {T.verdict(pC1)}", st(T.verdict(xC1) == T.verdict(lC1) == "SHOWN"))
n = tl("\\paragraph{``The copy, not the reset.''} \\textbf{Pre-registered and shown}")
row(f"tex5:{n}", "'The copy, not the reset': pre-registered and shown, pooled (C1 12 vs 0, p = 0.00024; C2 holds, 0 vs 2) and on each "
    "machine; the reading applies",
    f"pooled C1 {pC1['b']} vs {pC1['c']} p = {pC1['p']:.4g}; C2 {pC2['b']} vs {pC2['c']} {T.c2_verdict(pC2)}; X C1 {T.verdict(xC1)} C2 "
    f"{T.c2_verdict(xC2)}; L C1 {T.verdict(lC1)} C2 {T.c2_verdict(lC2)}; reading {H1}",
    st((pC1["b"], pC1["c"]) == (12, 0) and pfmt(pC1["p"]) == "0.00024" and (pC2["b"], pC2["c"]) == (0, 2) and
       all(T.verdict(cl[m]["C1"]) == "SHOWN" and T.c2_verdict(cl[m]["C2"]) == "HOLDS" for m in "XL") and H1))
n = tl("by the pre-registered C3 ($0/20$ against the recipe's $18/20$, 18 vs 0)")
row(f"tex5:{n}", "Supported, pooled and on each machine, by the pre-registered C3 (0/20 against the recipe's 18/20, 18 vs 0)",
    f"C3 pooled {pC3['old']}/20 vs {pC3['new']}/20, {pC3['b']} vs {pC3['c']} {T.verdict(pC3)}; X {T.verdict(xC3)}, L {T.verdict(lC3)}",
    st((pC3["old"], pC3["new"], pC3["b"], pC3["c"]) == (0, 18, 18, 0) and T.verdict(xC3) == T.verdict(lC3) == "SHOWN"))
disc = {m: cross(m, "WIN3_SPLIT_D8", "WIN3_RESET_D8")[0] for m in "XL"}
disc_oc = {m: sorted({T.outcome2(runs(m, "WIN3_RESET_D8")[s]) for s in disc[m]}) for m in "XL"}
n = tl("and, on X, by the pre-registered C1 (the copy with its reset against the reset alone, 7 vs 0)")
row(f"tex5:{n}", "The row copy breaks merges: supported ... and, on X, by the pre-registered C1 (7 vs 0); that the runs it rescues are merges "
    "is printed in their failure classes",
    f"C1 X {xC1['b']} vs {xC1['c']}, L {lC1['b']} vs {lC1['c']} {T.verdict(lC1)}, pooled {pC1['b']} vs {pC1['c']} {T.verdict(pC1)}; the "
    f"rescued seeds' RESET outcomes X {disc_oc['X']}, L {disc_oc['L']}", "PARTIAL",
    "true of X but stale since ef6d690: C1 is also SHOWN on L (5 vs 0) and pooled (12 vs 0, primary); the rescued runs are merges on "
    "both machines")
n = tl("the copy with its reset beats the reset alone (18/20 against 6/20)")
row(f"tex5:{n}", "conclusion: pre-registered on two machines with disjoint seeds, the copy with its reset beats the reset alone (18/20 "
    "against 6/20), the reset alone is within the bound of no trigger, and one delta channel binds none of twenty seeds where the recipe "
    "binds eighteen; both readings apply", summary,
    st((pC1["new"], pC1["old"]) == (18, 6) and T.verdict(pC1) == "SHOWN" and T.c2_verdict(pC2) == "HOLDS" and PB["SINGLE_DELTA_D8"]["c"] == 0
       and PB["WIN3_SPLIT_D8"]["c"] == 18 and H1 and H2))
n = tl("\\texttt{ffb0771} (L and pooled; the classifier \\texttt{ba8901a}, the launcher \\texttt{56c44a7})")
row(f"tex5:{n}", "provenance (tab:prov): spec fb3acb7 / 1f52cd5 / 592c574 (X), ffb0771 (L and pooled; the classifier ba8901a, the launcher "
    "56c44a7) & complete (L run at 592c574; one earlier start stopped at verification over an argument) & 54; 54",
    f"spec's only commit {git('log', '--format=%h', RES, '--', 'specs/test_split_copy.md').split()}; ffb0771 adds {LREC}: "
    f"{LREC in git('show', '--name-only', '--format=', RES).split()}; last fail_class_v2.py commit before the run {fc_last}; last longrun.py "
    f"commit {lr_last} (56c44a7..1f52cd5 longrun.py +{lr_numstat[0]}/-{lr_numstat[1]}); meta.git {mL['git']}; first start stopped at "
    f"CHECK 155 (--also given 'results/X/'); runs X {len(storeX['runs'])}, L {len(storeL['runs'])}",
    st(git("log", "--format=%h", RES, "--", "specs/test_split_copy.md").split() == [SPEC_C] and fc_last == "ba8901a" and mL["git"] == XRES
       and len(storeX["runs"]) == len(storeL["runs"]) == 54),
    "the launcher that ran is longrun.py as edited in 1f52cd5 (the progress estimate only), as the fourth pass found; rated as the "
    "fourth pass rated it")
n = tl("PyTorch 2.14.0 (L's rerun 2.14.1)")
row(f"tex5:{n}", "tab:prov caption: 'PyTorch 2.14.0 (L's rerun 2.14.1)'; L's start times local (UTC-6); X's split_copy on 2.10 GHz",
    f"L's split_copy meta.torch {mL['torch']} (L's window_gate {json.loads(show(RES, 'results/L/window_gate_results.json'))['meta']['torch']}); "
    f"X's split_copy torch {storeX['meta']['torch']}, CPU {storeX['meta']['cpu']!r}; L offset from 84a7e57 {offset_h:+g} h",
    "DISAGREE" if mL["torch"] != "2.14.0" else "MATCH",
    "the table's split_copy row includes L's run, recorded on torch 2.14.1; the caption gives 2.14.1 only for L's channel_binding rerun "
    "(the caption predates ef6d690's L row); the UTC-6 and 2.10 GHz statements hold")
N_TEX = len(ROWS) - N_DOC

# ══ output ═══════════════════════════════════════════════════════════════════
P("# test_split_copy, L half and pooled (ffb0771): records, log, docstring and the fifth draft (c5e4508)")
P()
P(f"Generated by `check_split_copy_L.py` (Referee V). Records: `{RES}:{LREC}`, `{RES}:{LLOG}` ({NLOG} lines), "
  f"`{XRES}:{XREC}` (unchanged at {RES}). Code: `{RES}:test_split_copy.py` and its helper modules at {RES} (unchanged since {XRES}: "
  f"{helpers_same}); every outcome, pairing, McNemar, Wilson, band, verdict, pooling, reading and report function below is the test's "
  f"own, extracted with `ast` and exec'd; nothing is trained or imported from the working tree. Draft: `{DRAFT_C}:{DRAFT}` "
  f"({len(TEX)} lines, cited tex5:NNN). Spec: `{SPEC_C}` (no amendment: commits touching the spec up to {RES}: "
  f"{git('log', '--format=%h', RES, '--', 'specs/test_split_copy.md').split()}).")
P()
P("## 1. Recomputed from the records (the test's functions)")
P()
P("| arm | X | L | pooled | band X / L / pooled | Wilson 95% X / L / pooled |")
P("|---|---|---|---|---|---|")
for k in T.ARMS_RUN:
    P(f"| {k} ({T.out_of(k)}) | {cnt(X_, k)} | {cnt(L_, k)} | {pc_(k)} | {band(bd[X_][k])} / {band(bd[L_][k])} / {band(PB[k])} | "
      f"{fw(bd[X_][k]['c'], 10)} / {fw(bd[L_][k]['c'], 10)} / {fw(PB[k]['c'], 20)} |")
for part, k in T.CEILS:
    P(f"| {k} (validity, Part {part}) | {TX['valid'][part][0]}/2 {[runs(X_, k)[s]['transition'] for s in seedsX[k]]} | "
      f"{TL['valid'][part][0]}/2 {[runs(L_, k)[s]['transition'] for s in seedsL[k]]} | "
      f"{TX['valid'][part][0] + TL['valid'][part][0]}/4 | VALID {TX['valid'][part][1]} / {TL['valid'][part][1]} / {TX['valid'][part][1] and TL['valid'][part][1]} | |")
P()
P("| claim | X: b vs c, p, verdict | L: b vs c, p, verdict | pooled: b vs c, p, verdict | discordant seeds X; L |")
P("|---|---|---|---|---|")
for name, a, b_, *_ in sorted(T.CLAIMS + (T.C2,)):
    cells = []
    for d in (cl[X_][name], cl[L_][name], PC[name]):
        extra = f"d = {d['b'] - d['c']}" if name == "C2" else f"p = {d['p']:.6g}"
        cells.append(f"{d['new']}/{d['n']} vs {d['old']}/{d['n']}; {d['b']} vs {d['c']}; {extra}; **{verdict_of(d, name)}**")
    dsx, dsl = cross(X_, a, b_), cross(L_, a, b_)
    P(f"| {name} {a} vs {b_} | {cells[0]} | {cells[1]} | {cells[2]} | {dsx[0]} / {dsx[1]}; {dsl[0]} / {dsl[1]} |")
pp = TL["printed"]["one delta channel vs one Hebbian channel"]
P(f"| printed: SINGLE_DELTA vs SINGLE_HEBB | {TX['printed']['one delta channel vs one Hebbian channel']['b']} vs "
  f"{TX['printed']['one delta channel vs one Hebbian channel']['c']} | {pp['b']} vs {pp['c']} | "
  f"{PPR['one delta channel vs one Hebbian channel']['b']} vs {PPR['one delta channel vs one Hebbian channel']['c']}, p = "
  f"{PPR['one delta channel vs one Hebbian channel']['p']:.3g} both ways | |")
P()
P(f"Readings (pooled, `readings()`): \"{T.READING_1}\" **{H1}** ({W1}); \"{T.READING_2}\" **{H2}** ({W2}).")
P()
P("Secondary numbers (L): " + "; ".join(
    f"{k} transitions median {statistics.median(trs_of(L_, k)):.0f} [{min(trs_of(L_, k))}, {max(trs_of(L_, k))}] ({len(trs_of(L_, k))} bound)"
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8")) +
  f"; collapsed SINGLE_DELTA {sum(bool(r.get('collapsed')) for r in runs(L_, 'SINGLE_DELTA_D8').values())}, SINGLE_HEBB "
  f"{sum(bool(r.get('collapsed')) for r in runs(L_, 'SINGLE_HEBB_D8').values())}; splits per run {list(per_run_fired(L_, 'WIN3_SPLIT_D8').values())}, "
  f"resets per run {list(per_run_fired(L_, 'WIN3_RESET_D8').values())}; outcome classes RESET {ocR}, SLOW {ocS}; final accuracy SINGLE_DELTA "
  f"{sdl[0]:.4f}-{sdl[1]:.4f}, SINGLE_HEBB {shr[0]:.4f}-{shr[1]:.4f}.")
P()
P(f"Cross-checks: rlib's independent outcome (recomputed one-to-one map, every stream >= 0.9) agrees with the test's on all "
  f"{sum(len(runs(m, k)) for m in 'XL' for k in T.ARMS_RUN)} learned-arm records: {rl_agree}; rlib's McNemar equals the test's for every "
  f"claim, per machine and pooled: {rl_p}; rlib's Wilson equals the test's for every band: {rl_w}; every record's transition equals the "
  f"first held evaluation >= 0.95 of its curve: {trans_ok}; ch_map ties met by rlib: {len(rlib.TIES)}.")
P()
P("## 2. The log against the records")
P()
P(f"* The report section of the log (log:{i0 + 1}-{i0 + len(GENR)}, {len(GENR)} lines: per-seed rows, validity, counts, the pooled and "
  f"per-machine claims, bands, readings, X's block, diagnostics, splits and resets, the side-by-side table, every curve) was "
  f"regenerated from the two records files with the test's own `report()` (wall clock passed as the log's {WALL:.0f} s, the one "
  f"number not in the records): **{len(rep_diff)} differing lines**" + (": " + "; ".join(f"log:{a} gen {b!r} log {c!r}" for a, b, c in rep_diff[:8]) if rep_diff else "")
  + f"; lines after it in the log: {len(rep_tail)}.")
P(f"* The {n_run_lines} RUNS lines (log:{min(v[0][0] for v in runs_lines.values())}-{max(v[0][0] for v in runs_lines.values())}) "
  f"regenerated with `main().record`'s format from each record (outcome, transition, accuracy, stop, firings, seconds; the "
  f"elapsed field excluded): one line per record and all equal: **{not run_bad}**" + (f" ({run_bad[:3]})" if run_bad else "") + ".")
P()
P("## 3. L's records: seeds, CPU, code, starts and times")
P()
P(f"* Seeds: meta and record keys equal the spec's (every arm 1560-1569, the perfect gates 1560-1561): {seeds_ok}; each record's arm "
  f"and seed fields equal its key: {seed_field_ok}; X's 560-569 disjoint: True.")
P(f"* CPU: meta {mL['cpu']!r}; every record {sorted(cpus)}; torch {mL['torch']}; workers {mL['workers']}; machine {mL['machine']}.")
P(f"* Code: the records carry no file hash; meta.git = {mL['git']} (HEAD's short sha when the run started), meta.window.git = "
  f"{window_git} (L's window_gate file), f_sha {mL['f_sha']}. test_split_copy.py's syntax tree without its docstring is the same at "
  f"{TEST_C}, {XRES} and {RES}: {code_same_1f_59 and code_same_59_ff}; .py files changed {TEST_C}..{XRES}: {py_changed_1f_59} and "
  f"{XRES}..{RES}: {py_changed_59_ff} (docstring only). The first start's traceback cites lines {[n_ for n_, _ in tb]} of "
  f"test_split_copy.py, whose text at {XRES} is the traceback's: {tb_match} ({'; '.join(f'{n_}: {s!r}' for n_, _, s in tb_ok)}), so "
  f"L's working copy had {XRES}'s line layout. The log's CHECK 152 found the copied modules equal F's byte for byte and CHECK 145 found "
  f"the knob's verify trees equal fb3acb7's (log:{logline('CHECK 145')}, {logline('CHECK 152')}). Whether L's working tree was clean is "
  f"not recorded.")
P(f"* Starts: {len(headers)} longrun headers: " + "; ".join(f"log:{a} {b!r}" for a, b in headers) + f". The first stopped at "
  f"verification (CHECK 155 WRONG: '--also' given 'results/X/'; AssertionError), with no RUNS block; its log, json and exit code "
  f"({fexit}) were pushed at {FIRST_L} (origin/claude/bdh-repo-curl-obgdqk) and log:1-{second - 1} equals that log: {first_log_same}. "
  f"The second ran all 54 runs with no cached record (CHECK 157 'a first start'; no 'cached' line; no meta.starts: {not resumed}); no "
  f"retry or resume is recorded.")
P(f"* Times (L local = UTC{offset_h:+g} h, from {FIRST_L}'s json: local {fj['starts'][0]['started']} = epoch {f_utc:%Y-%m-%d %H:%M:%S} UTC; "
  f"results/README.md says UTC−6: {readme_utc6}): {XRES} committed {C59:%Y-%m-%d %H:%M:%S} UTC; first start "
  f"{(t_start1_loc + OFF):%Y-%m-%d %H:%M} UTC; second start {(t_start2_loc + OFF):%Y-%m-%d %H:%M} UTC; runs from "
  f"{(t_runs_loc + OFF):%Y-%m-%d %H:%M} UTC; last run done about {(t_end_loc + OFF):%Y-%m-%d %H:%M} UTC (meta.started + the last "
  f"elapsed {last_elapsed} min); {RES} committed {CFF:%Y-%m-%d %H:%M:%S} UTC; the draft fold-in {FOLD_C} {CEF:%H:%M:%S} UTC. In order: {times_ok}.")
P()
P("## 4. Rows")
P()
P("Rows marked 'doc:N' are lines of test_split_copy.py at ffb0771 (the RESULT (L) and RESULT (POOLED, PRIMARY) blocks), each against the "
  "log line that carries it and the recomputation; rows marked 'tex5:N' are lines of the fifth draft.")
P()
P("| line | statement | recomputed (records, log, git) | status | note |")
P("|---|---|---|---|---|")
for w, s_, r_, stt, nt in ROWS:
    P(f"| {w} | {esc(s_)} | {esc(r_)} | **{stt}** | {esc(nt)} |")
P()
P("## 5. Counts")
P()
cd_ = {}
ct = {}
for w, *_x, stt, _n in ROWS:
    (cd_ if w.startswith("doc:") else ct)[stt] = (cd_ if w.startswith("doc:") else ct).get(stt, 0) + 1
P(f"* Docstring rows ({N_DOC}): " + ", ".join(f"{k} {v}" for k, v in sorted(cd_.items())) + ".")
P(f"* Draft rows ({N_TEX}): " + ", ".join(f"{k} {v}" for k, v in sorted(ct.items())) + ".")
P(f"* All ({len(ROWS)}): " + ", ".join(f"{k} {v}" for k, v in sorted(COUNT.items())) + ".")
P(f"* Every number of a claim, verdict, band, interval and reading in the docstring and the draft reproduces; the regenerated report "
  f"differs from the log in {len(rep_diff)} lines.")
P()
P("## 6. What could not be checked")
P()
P("* The second start's exit code ('exit 0'): no longrun json or .exit for it is committed; only the log, which ends with the report "
  "and has no traceback.")
P("* That the L run used the code as committed: the records hold no file hash; meta.git and the longrun headers give HEAD (592c574), "
  "and the first start's traceback line numbers match 592c574's file, but a dirty working tree is not recorded. The run itself was "
  "not repeated (no training).")
P("* The second start's UTC offset: taken equal to the first start's (UTC-6, from 84a7e57's json epoch) and results/README.md.")
P("* The 'longest first' order of the runs and the projection's figures (pool-load timing) are printed, not recorded per run.")
print("\n".join(OUT))
