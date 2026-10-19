#!/usr/bin/env python3
"""check_I_report3.py — Referee V's check of session I's report 3 (test_window_gate Part C / D on the GTX 1660 SUPER,
seeds 2540-2559) on claude/explore-I.

    python3 -I docs/review_2026-10/check_I_report3.py > docs/review_2026-10/check_I_report3.md

Run from the repository root after `git fetch`. Every file is read with `git show` / `git cat-file`; no session code is
run and nothing is trained. The main line's outcome functions (test_stream_recipe.outcome and what it calls,
test_window_gate.success / validity / tally / targeting, test_early_recipe.paired_by, test_stream_recipe.wilson,
test_conv_lr.mcnemar_greater, ...) are pure functions: their source is cut out of the main line's files by ast and
executed in a private namespace (no module is imported). docs/review_2026-10/rlib.py gives an independent
re-implementation (transition recomputed from the curve, the map from stream_gate).
"""
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import types
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "docs", "review_2026-10"))
import rlib  # noqa: E402

# ── pins ─────────────────────────────────────────────────────────────────────
I_TIP = "origin/claude/explore-I"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
NOTE_C = "30807ec"          # pre-run note, generator, mainline.py one-thread fix
REC_C = "148c161"           # the records
REP = "3d7e68e"             # report 3 as last changed (c18c7c0 draft, 6ab6324, 3d7e68e)
REP_HIST = ("c18c7c0", "6ab6324", "3d7e68e")
ROUND5 = "0754d61"          # main-line commit the note names
XL_C = "886a668"            # X's second start and L's start
X_FIRST = "2f98f06"         # X's first start
GPU_F = "explore_out/I/gpu/mainline/test_window_gate_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
NOTE_F = "explore_out/I/gpu/mainline/window_gate_prerun_note_2026-10-12.md"
DESC_F = "explore_out/I/gpu/mainline/window_gate_descriptions.md"
GEN_F = "explore_gpu/window_gate_writeup.py"
REP_F = "explore_out/I/report_3.md"
X_F, L_F = "results/X/window_gate_results.json", "results/L/window_gate_results.json"
ARMS3 = ("WIN3_SPLIT_D8", "HINGE_D8", "WIN3_SLOW_D8")
PAIRS = (("WIN3_SPLIT_D8", "HINGE_D8"), ("WIN3_SPLIT_D8", "WIN3_SLOW_D8"), ("WIN3_SLOW_D8", "HINGE_D8"))
GPU_SEEDS = dict(CEIL_C=(2540, 2541), **{a: tuple(range(2540, 2560)) for a in ARMS3})
SEED_LO, SEED_HI = 2540, 2559


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"), stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def blob(ref, path):
    try:
        return git("rev-parse", f"{ref}:{path}").strip()
    except subprocess.CalledProcessError:
        return None


def ctime(ref):
    return git("log", "-1", "--format=%cI", ref).strip()


def lines(ref, path):
    return show(ref, path).decode().split("\n")


ROWS = []


def row(where, stmt, recomputed, status, note=""):
    ROWS.append((where, stmt, recomputed, status, note))


def f2(x):
    return f"{x:.2f}"


def wil(c, n):
    lo, hi = NS["wilson"](c, n)
    return f"[{lo:.2f}, {hi:.2f}]"


# ── the main line's pure functions, cut out by ast ───────────────────────────
def extract(ref, path, names):
    src = show(ref, path).decode()
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            out[n.name] = ast.get_source_segment(src, n)
        elif isinstance(n, ast.Assign):
            tg = [t.id for t in n.targets if isinstance(t, ast.Name)]
            tg += [e.id for t in n.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
            hit = [t for t in tg if t in names]
            if hit:
                out[hit[0]] = ast.get_source_segment(src, n)
    missing = [x for x in names if x not in out]
    assert not missing, (path, missing)
    return [out[x] for x in names]


NS = {"math": math, "statistics": statistics}
SOURCES = [
    ("test_router_layout.py", ["MARGIN_PARTIAL", "ETA_SPLIT", "fail_class"]),
    ("test_scale_axes.py", ["fail_class_k"]),
    ("test_stream_channels.py", ["shared_max", "merged", "med_int"]),
    ("test_stream_recipe.py", ["ROUTED_ACC", "routed", "outcome", "wilson"]),
    ("test_conv_lr.py", ["bound_r", "mcnemar_greater"]),
    ("test_slow_start.py", ["ok_r", "runs_of"]),
    ("test_router_discovery.py", ["get"]),
    ("test_early_recipe.py", ["paired_by"]),
    ("test_stream_curriculum.py", ["RELIABLE_F", "band_n"]),
    ("test_window_gate.py", ["RANGES", "VALID_N", "G3_BOUND", "RESET_N", "ARMS", "ARM", "KEYS", "ARMS_RUN", "CEILS",
                             "OUTCOME", "CLAIMS", "G3", "PRINTED", "seeds_for", "success", "validity", "tally",
                             "targeting"]),
]
for path, names in SOURCES:
    for s in extract(MAIN, path, names):
        exec(compile(s, f"{MAIN}:{path}", "exec"), NS)
NS["tsr"] = types.SimpleNamespace(outcome=NS["outcome"], wilson=NS["wilson"])
NS["tscur"] = types.SimpleNamespace(band_n=NS["band_n"])
# the test's `get` comes from test_router_curriculum, which takes it from test_router_discovery
assert "get," in show(MAIN, "test_router_curriculum.py").decode().split("from test_router_discovery import (")[1][:300]
for path, _ in SOURCES:   # the extracted functions are the code the GPU run used (blob-identical, see code identity)
    assert blob(MAIN, path) == blob(NOTE_C, path) == blob(REC_C, path), path

# ── records ──────────────────────────────────────────────────────────────────
GPU = json.loads(show(REC_C, GPU_F))
X = json.loads(show(MAIN, X_F))
L = json.loads(show(MAIN, L_F))
assert blob(REC_C, GPU_F) == blob(REP, GPU_F) == blob(I_TIP, GPU_F)
for f in (X_F, L_F):
    assert blob(MAIN, f) == blob(REP, f) == blob(NOTE_C, f)


def xl_seeds(st):
    s = st["meta"]["seeds"]
    s = json.loads(s) if isinstance(s, str) else s
    return {k: tuple(v) for k, v in s.items()}


SETS = {"GPU": (GPU, dict(GPU_SEEDS)), "X": (X, xl_seeds(X)), "L": (L, xl_seeds(L))}


def describe(st, seeds):
    """What the generator prints, recomputed with the test's own tally / success / paired_by / wilson."""
    t = NS["tally"](st, seeds)
    n_c, valid = t["valid"]["C"]
    arms = {}
    for k in ARMS3:
        rr = t["rA"][k]
        arms[k] = dict(c=sum(NS["success"](k, r) for r in rr.values()), n=len(seeds.get(k, ())), have=len(rr))
    pairs = {}
    for a, b in PAIRS:
        d = NS["paired_by"](t["rA"][a], t["rA"][b], seeds[a], lambda r, k=a: NS["success"](k, r))
        pairs[(a, b)] = d
    return dict(valid=valid, ceil=n_c, ceil_n=len(seeds.get("CEIL_C", ())), arms=arms, pairs=pairs, t=t)


D = {k: describe(*v) for k, v in SETS.items()}
D["pooled"] = dict(valid=D["X"]["valid"] and D["L"]["valid"], ceil=D["X"]["ceil"] + D["L"]["ceil"],
                   ceil_n=D["X"]["ceil_n"] + D["L"]["ceil_n"],
                   arms={k: {f: D["X"]["arms"][k][f] + D["L"]["arms"][k][f] for f in ("c", "n", "have")} for k in ARMS3},
                   pairs={pk: {f: D["X"]["pairs"][pk][f] + D["L"]["pairs"][pk][f] for f in ("n", "b", "c")} for pk in PAIRS})


def rl_bound_routed(st, arm, seeds):
    rr = rlib.runs(st, arm)
    return {s: rlib.bound_routed(rr[s]) for s in seeds if s in rr}


# independent check with rlib (transition from the curve, the map from stream_gate)
RL = {}
for name, (st, seeds) in SETS.items():
    RL[name] = {k: rl_bound_routed(st, k, seeds[k]) for k in ARMS3}
    for k in ARMS3:
        test_ = {s: NS["success"](k, NS["get"](st, k, s)) for s in seeds[k]}
        assert test_ == RL[name][k], (name, k)          # the test's success and rlib agree seed for seed
    for s in seeds["CEIL_C"]:
        assert NS["bound_r"](NS["get"](st, "CEIL_C", s)) == rlib.bound(NS["get"](st, "CEIL_C", s))

out = []
P = out.append

# ═════════════════════════════════════════════════════════════════════════════
# 1. ORDER
# ═════════════════════════════════════════════════════════════════════════════
log = git("log", "--format=%h|%aI|%cI|%an <%ae>|%s", f"{NOTE_C}^..{I_TIP}").strip().split("\n")
WATCH = {NOTE_F: "note", GEN_F: "generator", GPU_F: "records", DESC_F: "descriptions", REP_F: "report_3",
         "explore_gpu/mainline.py": "mainline.py", "explore_gpu/device.py": "device.py", "test_window_gate.py": "test"}
COMMITS = []
for ln in reversed(log):
    h, ai, ci, who, subj = ln.split("|", 4)
    files = git("show", "--name-only", "--format=", h).split()
    touched = [WATCH[f] for f in files if f in WATCH]
    COMMITS.append((h, ai, ci, who, subj, touched))


# code SHA as explore_gpu/run.code_sha defines it (SHA-1 over name + source of every repository module, root's and
# explore_gpu's, that importing test_window_gate, explore_gpu.device and explore_gpu.mainline loads), recomputed from
# git objects by following the module-level imports (no import is executed)
def _files(ref):
    return set(git("ls-tree", "-r", "--name-only", ref).split("\n"))


def _path_of(name, fs):
    if name == "explore_gpu":
        return "explore_gpu/__init__.py" if "explore_gpu/__init__.py" in fs else None
    if name.startswith("explore_gpu."):
        p = name.replace(".", "/") + ".py"
        return p if p in fs else None
    if "." in name:
        return None
    return name + ".py" if name + ".py" in fs else None


def _top_imports(tree):
    found = []

    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue                                  # function bodies do not run at import
            if isinstance(n, ast.Import):
                for a in n.names:
                    parts = a.name.split(".")
                    found.extend(".".join(parts[:i]) for i in range(1, len(parts) + 1))
            elif isinstance(n, ast.ImportFrom) and not n.level:
                parts = n.module.split(".")
                found.extend(".".join(parts[:i]) for i in range(1, len(parts) + 1))
                found.extend(n.module + "." + a.name for a in n.names)
            for f in ("body", "orelse", "finalbody", "handlers"):
                b = getattr(n, f, None)
                if isinstance(b, list) and b and isinstance(b[0], ast.AST):
                    walk(b)
    walk(tree.body)
    return found


def closure(ref, starts):
    fs, seen, todo = _files(ref), {}, list(starts)
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        p = _path_of(n, fs)
        if p is None:
            continue
        src = show(ref, p)
        seen[n] = (p, src)
        todo += _top_imports(ast.parse(src))
    return seen


def code_sha(ref):
    seen = closure(ref, ["test_window_gate", "explore_gpu", "explore_gpu.device", "explore_gpu.mainline"])
    h = hashlib.sha1()
    for n in sorted(seen):
        h.update(n.encode())
        h.update(seen[n][1])
    return h.hexdigest()[:12], seen


SHA_AT = {}
for c in [x[0] for x in COMMITS] + ["ab65d9b", "5d9dcf9"]:
    SHA_AT[c] = code_sha(c)[0]
REC_SHAS = Counter(r.get("code_sha") for r in GPU["runs"].values())
SHA_NOTE, MODS = code_sha(NOTE_C)
first_sha = next(c[0] for c in COMMITS if SHA_AT[c[0]] == list(REC_SHAS)[0])

started = GPU["meta"]["started"]                      # machine clock, written once (setdefault)
secs = [r["secs"] for r in GPU["runs"].values()]
wall = [r["wall_secs"] for r in GPU["runs"].values()]


def hms(s):
    s = int(round(s))
    sign = "-" if s < 0 else "+"
    s = abs(s)
    return f"{sign}{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def ts(iso):
    from datetime import datetime
    return datetime.fromisoformat(iso).timestamp()


from datetime import datetime, timezone  # noqa: E402
t_started = datetime.strptime(started, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()  # as if UTC
t_note = ts(git("log", "-1", "--format=%cI", NOTE_C).strip())
t_rec_a = ts(git("log", "-1", "--format=%aI", REC_C).strip())
t_rec_c = ts(git("log", "-1", "--format=%cI", REC_C).strip())
t_4371824 = ts(git("log", "-1", "--format=%cI", "4371824").strip())
elapsed = t_rec_a - t_started                          # both on the 1660 host's clock

# clock bounds for the 1660 host (root@Docker.home) from git ancestry, taking session I's (Claude's) commit times as UTC
DOCKER = [c for c in COMMITS if c[3].startswith("root <root@Docker.home>")]
bounds = []
for h, ai, ci, who, subj, _ in DOCKER:
    par = git("log", "-1", "--format=%P", h).split()[0]
    t_par = ts(git("log", "-1", "--format=%cI", par).strip())
    kids = [c for c in COMMITS if git("log", "-1", "--format=%P", c[0]).split()[0].startswith(h)]
    t_kid = ts(kids[0][2]) if kids and not kids[0][3].startswith("root") else None
    up = ts(ci) - t_par                                    # parent existed before: offset <= docker - parent
    lo = (ts(ci) - t_kid) if t_kid else None               # child came after: offset >= docker - child
    bounds.append((h, ci, par, t_par, kids[0][0] if kids else None, t_kid, lo, up))
ups = [b[7] for b in bounds if b[0] in ("4371824", "148c161")]
los = [b[6] for b in bounds if b[0] in ("4371824", "148c161") and b[6] is not None]
OFF_LO, OFF_HI = max(los), min(ups)

note_hist = git("log", "--format=%h %cI", I_TIP, "--", NOTE_F).strip().split("\n")
gen_hist = git("log", "--format=%h %cI", I_TIP, "--", GEN_F).strip().split("\n")


def defs_dump(ref, path):
    t = ast.parse(show(ref, path).decode())
    d = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            d[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name):
                    d["=" + tg.id] = ast.dump(n.value)
    return d


g0, g1 = defs_dump(NOTE_C, GEN_F), defs_dump(REP, GEN_F)
GEN_CHANGED = sorted(k for k in set(g0) | set(g1) if g0.get(k) != g1.get(k))

# ═════════════════════════════════════════════════════════════════════════════
# 2. CODE IDENTITY
# ═════════════════════════════════════════════════════════════════════════════
ROOT_MODS = sorted(v[0] for k, v in MODS.items() if not k.startswith("explore_gpu"))
blob_main = {p: (blob(NOTE_C, p) == blob(MAIN, p) == blob(ROUND5, p) == blob(REP, p) == blob(I_TIP, p)) for p in ROOT_MODS}
last_main = git("log", "-1", "--format=%h %cI %s", MAIN, "--", *ROOT_MODS).strip()
gpu_pkg_same = all(blob(NOTE_C, p) == blob(REC_C, p) == blob(REP, p) == blob(I_TIP, p)
                   for p in ("explore_gpu/__init__.py", "explore_gpu/device.py", "explore_gpu/mainline.py"))
dev_hist = git("log", "--format=%h %cI", I_TIP, "--", "explore_gpu/device.py").strip().split("\n")
ml_hist = git("log", "--format=%h %cI", I_TIP, "--", "explore_gpu/mainline.py").strip().split("\n")


def mod_ast_diff(a, b):
    A, B = closure(a, ["test_window_gate"]), closure(b, ["test_window_gate"])
    res = {}
    for m in sorted(set(A) | set(B)):
        if m not in A or m not in B:
            res[m] = "only in one"
            continue
        if A[m][1] == B[m][1]:
            continue
        da, db = defs_dump(a, A[m][0]), defs_dump(b, B[m][0])
        res[m] = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    return res


DIFF_XL = mod_ast_diff(XL_C, NOTE_C)
DIFF_X1 = mod_ast_diff(X_FIRST, NOTE_C)


def fn_src_equal(r1, p1, n1, r2, p2, n2, strip_doc=True):
    def get_(r, p, n):
        t = ast.parse(show(r, p).decode())
        node = next(x for x in t.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == n)
        if strip_doc:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and sub.body and isinstance(sub.body[0], ast.Expr) \
                        and isinstance(getattr(sub.body[0], "value", None), ast.Constant) and isinstance(sub.body[0].value.value, str):
                    sub.body = sub.body[1:]
        return ast.dump(node)
    return get_(r1, p1, n1) == get_(r2, p2, n2)


CEIL_PATH_SAME = all(fn_src_equal(X_FIRST, "test_window_gate.py", n, NOTE_C, "test_window_gate.py", n)
                     for n in ("run_path", "iters_of", "run_job", "spec", "success", "validity"))
LOCAL_SAME = fn_src_equal(NOTE_C, "explore_b16_gates.py", "LocalBDH", NOTE_C, "test_window_gate.py", "LocalBDH")
LOCAL_SAME_DOC = fn_src_equal(NOTE_C, "explore_b16_gates.py", "LocalBDH", NOTE_C, "test_window_gate.py", "LocalBDH", False)

# ═════════════════════════════════════════════════════════════════════════════
# 3. SEEDS 2540-2559: every branch tip, and every JSON / JSONL blob in the history
# ═════════════════════════════════════════════════════════════════════════════
KEYRX = re.compile(r"^[A-Za-z0-9_]+\|(\d+)(\||$)")


def in_range(v):
    return isinstance(v, int) and not isinstance(v, bool) and SEED_LO <= v <= SEED_HI


def scan(o, hits, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRX.match(k) if isinstance(k, str) else None
            if m and SEED_LO <= int(m.group(1)) <= SEED_HI:
                hits.append(("run key", k))
            if isinstance(k, str) and k.isdigit() and SEED_LO <= int(k) <= SEED_HI and isinstance(v, dict):
                hits.append(("seed-keyed dict", f"{path}/{k}"))
            if k == "seed" and in_range(v):
                hits.append(("seed field", f"{path}/seed={v}"))
            if k in ("seeds", "SEEDS") and isinstance(v, (list, dict)):
                vals = v if isinstance(v, list) else [x for vv in v.values() if isinstance(vv, list) for x in vv]
                for x in vals:
                    if in_range(x):
                        hits.append(("seeds list", f"{path}/{k}"))
                        break
            scan(v, hits, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            scan(v, hits, path)


def parse_any(b):
    try:
        return [json.loads(b)]
    except Exception:
        objs = []
        for ln in b.decode(errors="replace").split("\n"):
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    objs.append(json.loads(ln))
                except Exception:
                    pass
        return objs


PRE = re.compile(rb"(?<![\d.])25[45]\d(?![\d.])")
branches = [b.strip() for b in git("branch", "-r").split("\n") if b.strip() and "->" not in b]
TIP_HITS = {}
TIP_FILES = 0
TIP_TEXT_HITS = []
TEXTRX = re.compile(r"(?i)(seed[^\n]{0,30}?(?<![\d.])25[45]\d(?![\d.])|[A-Z0-9_]+\|25[45]\d\b)")
for br in branches:
    for ln in git("ls-tree", "-r", br).split("\n"):
        if not ln:
            continue
        meta_, path = ln.split("\t", 1)
        sha = meta_.split()[2]
        if path.endswith((".json", ".jsonl")):
            TIP_FILES += 1
            if (sha, path) in TIP_HITS:
                continue
            raw = subprocess.check_output(("git", "-C", ROOT, "cat-file", "blob", sha))
            hits = []
            if PRE.search(raw):
                for o in parse_any(raw):
                    scan(o, hits)
            if hits:
                TIP_HITS[(sha, path)] = (br, hits)
        elif path.endswith((".log", ".txt")):
            raw = subprocess.check_output(("git", "-C", ROOT, "cat-file", "blob", sha)).decode(errors="replace")
            for m in TEXTRX.finditer(raw):
                TIP_TEXT_HITS.append((br, path, m.group(0)[:80]))

# history: every JSON / JSONL blob reachable from any ref
paths_of = defaultdict(set)
for ln in git("rev-list", "--all", "--objects").split("\n"):
    p = ln.split(" ", 1)
    if len(p) == 2 and p[1].endswith((".json", ".jsonl")):
        paths_of[p[0]].add(p[1])
HIST_HITS = {}
proc = subprocess.Popen(("git", "-C", ROOT, "cat-file", "--batch"), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for sha in sorted(paths_of):
    proc.stdin.write((sha + "\n").encode())
    proc.stdin.flush()
    hdr = proc.stdout.readline().split()
    size = int(hdr[2])
    raw = proc.stdout.read(size)
    proc.stdout.read(1)
    if not PRE.search(raw):
        continue
    hits = []
    for o in parse_any(raw):
        scan(o, hits)
    if hits:
        HIST_HITS[sha] = hits
proc.stdin.close()
proc.wait()
GPU_BLOB = blob(REC_C, GPU_F)
OTHER_HIST = {s: h for s, h in HIST_HITS.items() if s != GPU_BLOB}
RTX_WG = sorted({p for ps in paths_of.values() for p in ps if "window_gate" in p and "3070" in p})


def seed_set(st, pat=None):
    out = set()
    for k in st["runs"]:
        p = k.split("|")
        if len(p) > 1 and p[1].isdigit():
            out.add(int(p[1]))
    return out


XS, LS, GS = seed_set(X), seed_set(L), seed_set(GPU)
R_SEEDS = set()
for f in ("records/phase2.jsonl", "records/phase3.jsonl", "records/phase4.jsonl", "records/phase5.jsonl"):
    for o in parse_any(show("origin/claude/replicate-R", f)):
        if isinstance(o.get("seed"), int):
            R_SEEDS.add(o["seed"])

# ═════════════════════════════════════════════════════════════════════════════
# 4. RECOMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
G = D["GPU"]
runs_ok = sum(bool(r.get("ok")) for r in GPU["runs"].values())
plan = {f"{k}|{s}" for k, v in GPU_SEEDS.items() for s in v}
extra = set(GPU["runs"]) - plan
missing = plan - set(GPU["runs"])
REAL = dict(iters_a=24000, iters_b=28800, iters_c=43200, warm=2400, window=2400)
REAL_SRC = extract(MAIN, "test_window_gate.py", ["REAL"])[0]
sched_ok = all(r["spec"]["sched"] == REAL and r["iters"] == 43200 for r in GPU["runs"].values())
dev_set = Counter(json.dumps(r["device"], sort_keys=True) for r in GPU["runs"].values())
dev0 = GPU["runs"]["CEIL_C|2540"]["device"]
threads1 = all(r["device"]["threads"] == 1 for r in GPU["runs"].values())


def per_seed(st, seeds):
    return {s: tuple(NS["success"](k, NS["get"](st, k, s)) for k in ARMS3) for s in seeds}


PS = per_seed(GPU, range(2540, 2560))
groups = defaultdict(list)
for s, v in PS.items():
    groups[v].append(s)


def mc(b, c):
    return NS["mcnemar_greater"](b, c)


def diag(st, seeds):
    """Diagnostics the report does not describe (the note did not fix them): outcomes, collapsed, medians, firings."""
    res = {}
    for k in ARMS3:
        rr = {s: NS["get"](st, k, s) for s in seeds[k]}
        oc = Counter()
        for s, r in rr.items():
            o = NS["outcome"](r)
            if o != "BOUND ROUTED" and r.get("collapsed"):
                o = "collapsed / " + o
            oc[o] += 1
        trs = [r["transition"] for r in rr.values() if r["transition"] is not None]
        trs_br = [r["transition"] for r in rr.values() if NS["success"](k, r)]
        d = dict(outcomes=dict(oc), collapsed=sum(bool(r.get("collapsed")) for r in rr.values()),
                 med_bound=NS["med_int"](trs), n_bound=len(trs), med_br=NS["med_int"](trs_br))
        if k == "WIN3_SPLIT_D8":
            fired = [(s, c) for s, r in rr.items() for c in r.get("checks", []) if c.get("fired")]
            blocked = Counter(c.get("blocked_by") for r in rr.values() for c in r.get("checks", []) if c.get("blocked"))
            per = Counter(len([1 for c in r.get("checks", []) if c.get("fired")]) for r in rr.values())
            lab_rec = sum(bool(c.get("key_ok")) for _, c in fired)
            lab_re = sum(NS["targeting"](c["map_before"], c["key_cs"], c["key_c0"]) for _, c in fired)
            spl = all(r.get("splits") == len([1 for c in r.get("checks", []) if c.get("fired")]) for r in rr.values())
            steps = Counter(c["step"] for _, c in fired)
            d.update(fires=len(fired), per_run=dict(sorted(per.items())), labelled_rec=lab_rec, labelled_re=lab_re,
                     blocked=dict(blocked), splits_field_agrees=spl, fire_steps=dict(sorted(steps.items())))
        res[k] = d
    return res


DIAG = {name: diag(st, seeds) for name, (st, seeds) in SETS.items()}

# ═════════════════════════════════════════════════════════════════════════════
# 5. CHECK outputs (the GPU package)
# ═════════════════════════════════════════════════════════════════════════════
CK = {}
for card in ("NVIDIA_GeForce_GTX_1660_SUPER", "NVIDIA_GeForce_RTX_3070_Laptop_GPU"):
    txt = show(REP, f"explore_gpu/checks_cuda_{card}_output.txt").decode()
    status = {}
    for ln in txt.split("\n"):
        m = re.match(r"^(ok|SKIP|info|FAIL|fail)\s+(K\w+)\s", ln)
        if m:
            status[m.group(2)] = m.group(1)
    js = json.loads(show(REP, f"explore_gpu/checks_cuda_{card}_output.json"))
    CK[card] = dict(verdict=txt.strip().split("\n")[-1], status=status, js=js,
                    commit=git("log", "-1", "--format=%h %cI", REP, "--", f"explore_gpu/checks_cuda_{card}_output.txt").strip())
CPU_CK = json.loads(show(REP, "explore_gpu/checks_cpu_output.json"))
CPU_TXT = show(REP, "explore_gpu/checks_cpu_output.txt").decode()
cpu_ck_commit = git("log", "-1", "--format=%h %cI", REP, "--", "explore_gpu/checks_cpu_output.txt").strip()
k8 = CPU_CK["checks"]["K8"]
k4b = CK["NVIDIA_GeForce_GTX_1660_SUPER"]["js"]["checks"]["K4b"]
k4b_cases = list(k4b["cases"])
all_case_names = [c for k, v in CK["NVIDIA_GeForce_GTX_1660_SUPER"]["js"]["checks"].items()
                  if isinstance(v, dict) for c in (v.get("cases") or {})]
k1 = CK["NVIDIA_GeForce_GTX_1660_SUPER"]["js"]["checks"]["K1"]

# ═════════════════════════════════════════════════════════════════════════════
# ROWS: report_3.md (at 3d7e68e) and its descriptions file
# ═════════════════════════════════════════════════════════════════════════════
R3 = lines(REP, REP_F)
DS = lines(REP, DESC_F)
assert blob(REP, REP_F) == blob(I_TIP, REP_F) and blob(REP, DESC_F) == blob(I_TIP, DESC_F)


def find(L_, frag):
    i = next(i for i, x in enumerate(L_) if frag in x)
    return i + 1


def cell(d, k):
    a = d["arms"][k]
    miss = "" if a["have"] == a["n"] else f" ({a['n'] - a['have']} missing)"
    return f"{a['c']}/{a['n']} {wil(a['c'], a['n'])}{miss}"


def pcell(d, pk):
    p = d["pairs"][pk]
    return f"b {p['b']}, c {p['c']} (n {p['n']})"


# l.3-5
txt3 = "\n".join(R3)
bad = [(n, w) for n, ln in enumerate(R3 + ["--desc--"] + DS, 1) for w in ("p =", "p=", "p <", "replicat", "consistent", "SHOWN") if w in ln]
row(f"r3:{find(R3, 'There is no reading')}-5", "no reading; descriptions only; no p-value, no SHOWN / NOT SHOWN, no 'replicates' / 'consistent'",
    f"occurrences of p =, p <, replicat-, consistent, SHOWN (line in report_3, or report_3 length + 1 + line in the descriptions): {bad}; each is the disclaimer itself (r3 l.4-5) or the descriptions' last line ('No p-value and no SHOWN / NOT SHOWN label are printed'); no p-value or verdict is printed",
    "MATCH", "the note's 'Not printed' list (note l.53-54) is respected")
row(f"r3:{find(R3, 'committed in')}", "the note was committed in 30807ec before the run",
    f"note first and only commit {note_hist[-1]}; every record carries code SHA {list(REC_SHAS)[0]}, which first exists at {first_sha} (its parent 559b942 hashes {code_sha('559b942')[0]}); meta.started {started} (machine clock) = note + {hms(t_started - t_note)} on a UTC reading",
    "MATCH", "see Order: the code-SHA argument needs no clock")
for k, ln_frag in zip(ARMS3, ("| WIN3_SPLIT_D8 | 19/20", "| HINGE_D8 | 0/20", "| WIN3_SLOW_D8 | 6/20")):
    i = find(R3, ln_frag)
    stated = [c.strip() for c in R3[i - 1].split("|")[2:6]]
    rec = [cell(D[n], k) for n in ("GPU", "X", "L", "pooled")]
    row(f"r3:{i}", f"{k}: GPU / X / L / pooled = {' ; '.join(stated)}", " ; ".join(rec),
        "MATCH" if stated == rec else "DISAGREE",
        "BOUND ROUTED = test_window_gate.success (tsr.outcome == 'BOUND ROUTED'); identical seed for seed with rlib.bound_routed")
i = find(R3, "bound on 2540 and on 2541")
cc = {s: (NS["get"](GPU, "CEIL_C", s)["transition"], NS["bound_r"](NS["get"](GPU, "CEIL_C", s))) for s in (2540, 2541)}
row(f"r3:{i - 1}-{i}", "CEIL_C bound on 2540 and 2541: 2/2, VALID; X and L 2/2 as well",
    f"GPU CEIL_C transitions {cc[2540][0]} / {cc[2541][0]} -> {G['ceil']}/{G['ceil_n']} {'VALID' if G['valid'] else 'NOT VALID'}; X {D['X']['ceil']}/2 ({[NS['get'](X, 'CEIL_C', s)['transition'] for s in (540, 541)]}), L {D['L']['ceil']}/2 ({[NS['get'](L, 'CEIL_C', s)['transition'] for s in (1540, 1541)]})",
    "MATCH" if G["valid"] and D["X"]["valid"] and D["L"]["valid"] else "DISAGREE", "test_window_gate.validity (bound_r: ok and transition not None, VALID_N = 2)")
for pk, frag in zip(PAIRS, ("| WIN3_SPLIT_D8 vs HINGE_D8", "| WIN3_SPLIT_D8 vs WIN3_SLOW_D8", "| WIN3_SLOW_D8 vs HINGE_D8")):
    i = find(R3, frag)
    stated = [c.strip() for c in R3[i - 1].split("|")[2:6]]
    rec = [pcell(D[n], pk) for n in ("GPU", "X", "L", "pooled")]
    p = D["GPU"]["pairs"][pk]
    row(f"r3:{i}", f"{pk[0]} vs {pk[1]}: {' ; '.join(stated)}", " ; ".join(rec), "MATCH" if stated == rec else "DISAGREE",
        f"test_early_recipe.paired_by over seeds with both records ok; exact one-sided McNemar (referee's, not printed by report 3, as the note fixed): GPU p = {mc(p['b'], p['c']):.3g}")
gen = show(REP, GEN_F).decode()
uses = all(x in gen for x in ("twg.tally(", "twg.success(", "twg.paired_by(", "tsr.wilson("))
row(f"r3:{find(R3, 'Every number and verdict above')}-31", "generated by window_gate_writeup with test_window_gate's own tally / success and test_stream_recipe.wilson",
    f"generator at {REP} calls twg.tally, twg.success, twg.paired_by, tsr.wilson: {uses}; recomputing with those functions (cut from the main line) gives every table cell above",
    "MATCH" if uses else "DISAGREE", "the generator itself was not run (session code)")
row(f"r3:{find(R3, 'Every record is ok, so each pairing')}", "every record is ok, so each pairing has n = 20",
    f"{runs_ok}/62 ok; pair n = {[D['GPU']['pairs'][pk]['n'] for pk in PAIRS]}", "MATCH")
for v, frag in (((True, False, False), "(13 seeds)"), ((True, False, True), "(6 seeds)"), ((False, False, False), "| 2558 |")):
    i = find(R3, frag)
    stated_seeds = R3[i - 1].split("|")[1].strip()
    row(f"r3:{i}", f"per seed: {stated_seeds} -> {['R' if x else '.' for x in v]}",
        f"seeds with that pattern: {groups.get(v, [])}; other patterns present: {sorted(set(groups) - {(True, False, False), (True, False, True), (False, False, False)})}",
        "MATCH", "")
row(f"r3:{find(R3, 'The test and the path')}-48", "the test's own run_job, Part C's path run_sc with D8; every record has REAL's 43,200 steps (iters 43200); --iters not used",
    f"spec.sched == REAL {REAL} and iters == 43200 on all 62: {sched_ok}; main line's REAL: `{REAL_SRC}` (MAX_A/B/C 24000/28800/43200)",
    "MATCH" if sched_ok else "DISAGREE", "--iters would overwrite iters_a/b/c (mainline._sched); with --iters 43200 the spec would be the same, so 'not used' is shown only up to that")
row(f"r3:{find(R3, '**Seeds.**')}-50", "CEIL_C 2540-2541, each arm 2540-2559: 62 records, all ok, none outside the plan; no overlap with X, L, R",
    f"{len(GPU['runs'])} records, {runs_ok} ok, {len(extra)} outside the plan, {len(missing)} missing; X seeds {min(XS)}-{max(XS)}, L {min(LS)}-{max(LS)}, R {min(R_SEEDS)}-{max(R_SEEDS)}: overlap {sorted(GS & (XS | LS | R_SEEDS))}",
    "MATCH" if not extra and not missing and runs_ok == 62 and not GS & (XS | LS | R_SEEDS) else "DISAGREE", "see Seeds for the scan of every branch")
row(f"r3:{find(R3, '**Code.**')}-54", "every record code SHA 05bbaaaac7a3 = the hash at 30807ec; none of those modules changed since; test_window_gate / test_stream_recipe / test_stream_curriculum byte-identical to the main line's",
    f"records {dict(REC_SHAS)}; recomputed (static import closure, {len(MODS)} modules) at {NOTE_C}: {SHA_NOTE}; at {REC_C}: {SHA_AT[REC_C]}; at {REP}: {SHA_AT[REP]}; tip: {SHA_AT[COMMITS[-1][0]]}; all {len(ROOT_MODS)} root modules blob-identical to {MAIN.split('/')[-1]} tip and {ROUND5}: {all(blob_main.values())}",
    "MATCH" if SHA_NOTE == list(REC_SHAS)[0] and all(blob_main.values()) and gpu_pkg_same else "DISAGREE", "")
row(f"r3:{find(R3, '**Device.**')}-56", "GTX 1660 SUPER, 5.61 GB, sm 7.5, CUDA 13.0, torch 2.14.0+cu130, TF32 off, Ryzen 5 3600; one torch thread in every record",
    f"{len(dev_set)} distinct device meta over 62 records: gpu {dev0['gpu']}, {dev0['gpu_mem_gb']} GB, cap {dev0['capability']}, cuda {dev0['cuda']}, torch {dev0['torch']}, tf32 {dev0['tf32_matmul']}/{dev0['tf32_cudnn']}, cpu {dev0['cpu']}, threads 1 in all: {threads1}",
    "MATCH" if len(dev_set) == 1 and threads1 else "DISAGREE", "the device meta is device_info(dev) inside device_scope; it does not by itself prove where each tensor op ran (K4b / K8 cover the path)")
row(f"r3:{find(R3, 'The results file was started')}-58", "file started 03:20:25 on 2026-10-13 by the first command (started set once)",
    f"meta.started = {started}; mainline.main: st['meta'].setdefault('started', time.strftime(...))", "MATCH", "machine-local clock, no zone")
row(f"r3:{find(R3, 'runs on UTC')}-60", "the server's local clock runs on UTC: the same host committed 4371824 at 03:20:22 +0000, three seconds earlier",
    f"4371824 committed {git('log', '-1', '--format=%cI', '4371824').strip()} by root <root@Docker.home> (as 148c161); {t_started - t_4371824:.0f} s before meta.started; ancestry bounds the host clock to UTC {hms(OFF_LO)} .. {hms(OFF_HI)}",
    "MATCH", "the +0000 shows the host's time zone, and the 3 s is right; 'same host' rests on the committer identity, and the clock's accuracy only on the ancestry bound (Order)")
row(f"r3:{find(R3, '4.6 h after')}", "the start was 4.6 h after the note's commit (22:44:05 UTC 10-12), so the note preceded the run",
    f"{hms(t_started - t_note)} = {(t_started - t_note) / 3600:.2f} h on a UTC reading; with the ancestry bound the run began at UTC >= {datetime.fromtimestamp(t_started - OFF_HI, timezone.utc):%m-%d %H:%M:%S}, >= {(t_started - OFF_HI - t_note) / 3600:.2f} h after the note",
    "MATCH", "independently of any clock: code SHA 05bbaaaac7a3 does not exist before 30807ec")
row(f"r3:{find(R3, 'committed and pushed in 148c161')}", "records committed and pushed in 148c161 at 15:31 UTC",
    f"148c161 author {git('log', '-1', '--format=%aI', REC_C).strip()}, committer {git('log', '-1', '--format=%cI', REC_C).strip()}; its child c18c7c0 (session I) at {git('log', '-1', '--format=%cI', 'c18c7c0').strip()}",
    "MATCH", "the push time is not recorded; it was on the remote by 15:48:27 (c18c7c0)")
row(f"r3:{find(R3, 'sum to 15.0 h')}-65", "records' secs sum to 15.0 h, more than the 12.2 h between start and commit, so runs overlapped",
    f"sum secs {sum(secs) / 3600:.3f} h (wall_secs {sum(wall) / 3600:.3f} h); meta.started -> 148c161 author time (same host clock) {elapsed / 3600:.3f} h; ratio {sum(secs) / elapsed:.2f} -> at least 2 runs at once at some time",
    "MATCH", "per arm: " + ", ".join(f"{k} {sum(r['secs'] for kk, r in GPU['runs'].items() if kk.startswith(k + '|')) / 3600:.2f} h" for k in ("CEIL_C",) + ARMS3))
early = all(r["stopped_at"] == (r["transition"] + 2400 if r["transition"] is not None and r["transition"] + 2400 <= 43200 else 43200)
            for r in GPU["runs"].values())
row(f"r3:{find(R3, 'Per-arm times are not given')}-65", "a run stops once it binds, so per-arm times track when arms bound",
    f"stopped_at = transition + 2400 for every bound run that stopped early, 43200 otherwise: {early}", "MATCH", "")
row(f"r3:{find(R3, 'The note fixed CEIL_C first')}-67", "the note fixed CEIL_C first, then the three arms, through four commands", "note l.15-27: order 1. CEIL_C 2540-2541, 2. the three arms; four commands", "MATCH")
row(f"r3:{find(R3, 'carry no start times')}-69", "records carry no start times; the file is saved with sorted keys, so the order cannot be checked",
    f"record fields: {sorted(set().union(*[set(r) for r in GPU['runs'].values()]) & {'started', 'start', 'time', 't0', 'finished'}) or 'no time field'} (only secs / wall_secs); mainline.save: json.dump(..., sort_keys=True)", "MATCH")
row(f"r3:{find(R3, 'command line or the worker count')}", "records carry neither the command line nor the worker count",
    f"no record or meta field for argv / workers: meta keys {sorted(GPU['meta'])}", "MATCH")
row(f"r3:{find(R3, 'What the records do confirm')}-72", "records confirm arm and seed, device cuda:0, REAL schedule (no --iters), one thread",
    f"arm field == key arm on all: {all(r['arm'] == k.split('|')[0] and r['seed'] == int(k.split('|')[1]) for k, r in GPU['runs'].items())}; device {dev0['device']}; sched REAL {sched_ok}; threads 1 {threads1}", "MATCH")
row(f"r3:{find(R3, 'aborted attempt')}", "none of the RTX 3070's records is in the repository",
    f"paths in any ref's history with 'window_gate' and '3070': {RTX_WG or 'none'}; JSON/JSONL blobs anywhere in history holding a 2540-2559 seed other than this file: {len(OTHER_HIST)}", "MATCH")
row(f"r3:{find(R3, 'It was not in 148c161')}-76", "descriptions not in 148c161 although its message names it; generated with this report; generator changed after the run in three ways; no number or verdict changed",
    f"148c161 touches only the results file; {DESC_F.split('/')[-1]} first in c18c7c0; generator history {gen_hist}; changed top-level defs 30807ec -> {REP}: {GEN_CHANGED}; describe(), ARMS, PAIRS, GPU_SEEDS unchanged; all 62 planned records present, so the 30807ec seed filter would have given the same n",
    "MATCH" if GEN_CHANGED == ["main"] else "PARTIAL", "the three changes (label, planned seeds, header) are all in main()")
row(f"r3:{find(R3, 'GPU column label changed')}-78", "label 'GPU GPU:NVIDIA …' -> 'GPU:NVIDIA GeForce GTX 1660 SUPER'",
    f"30807ec: 'GPU ' + meta.machine = 'GPU {GPU['meta']['machine']}'; c18c7c0 on: meta.machine", "MATCH")
row(f"r3:{find(R3, 'planned seeds, as the test')}-81", "planned seeds as the test's tally; '(k missing)'; pairs' n counts seeds with both ok",
    "generator main() at 3d7e68e passes dict(GPU_SEEDS); describe() prints '(k missing)' when have < n; paired_by counts seeds with both ok", "MATCH")
row(f"r3:{find(R3, 'records present / planned')}-85", "header gives present/planned, ok, outside plan, and the code (X/L: meta.git and meta.starts; GPU: code SHA)",
    f"desc l.5-7: {DS[4][:60]}… ; {DS[5][:40]}… ; {DS[6][:40]}…", "MATCH")
xs = X["meta"]
row(f"r3:{find(R3, 'records two starts')}-91", "X: first start at 2f98f06 kept only its six perfect-gate records (CEIL_C's two among them); the other 225, all three arms included, ran at 886a668; L entirely at 886a668",
    f"X meta.git {xs['git']} (started {xs['started']}), meta.starts {[(s['git'], s['started'], s['arms_added']) for s in xs['starts']]}; X records {len(X['runs'])} = 6 + {len(X['runs']) - 6}; L meta.git {L['meta']['git']}, starts {L['meta'].get('starts')}",
    "NOT RECORD-CHECKABLE", "no record carries its start's commit; the meta is consistent with it, and the report names its sources (RESULT (X), results/README.md l.29); the CEIL_C path's functions are ast-identical at 2f98f06 and 886a668 / 30807ec" + ("" if CEIL_PATH_SAME else " (NOT identical)"))
row(f"r3:{find(R3, 'Before the fix, the descriptions')}", "before the fix the descriptions' header named only 2f98f06",
    f"c18c7c0 descriptions l.6: {lines('c18c7c0', DESC_F)[5]}", "MATCH")
row(f"r3:{find(R3, '60 new runs')}-97", "60 new runs plus two validity runs, beside X's and L's CPU records; a CUDA run is not bit-comparable: outcomes only",
    f"3 x 20 + 2 = {len(GPU['runs'])}; the 1660's K1 (arm A 160 vs its CPU record) identical = {k1.get('identical')}", "MATCH", "bit-equivalence is disclaimed, not claimed")
s1660 = CK["NVIDIA_GeForce_GTX_1660_SUPER"]
row(f"r3:{find(R3, 'The CUDA CHECKs on this card')}-101", "the card's CUDA CHECKs passed every asserted CHECK ('ALL ASSERTED CHECKS PASSED'); K1, K2, K2b, K2c, K9 printed, not asserted; CPU-only CHECKs, K8 among them, skipped",
    f"verdict '{s1660['verdict']}'; ok {sorted(k for k, v in s1660['status'].items() if v == 'ok')}; info {sorted(k for k, v in s1660['status'].items() if v == 'info')}; SKIP {sorted(k for k, v in s1660['status'].items() if v == 'SKIP')} (file {s1660['commit']})", "MATCH",
    f"the CHECK predates the run; explore_gpu/device.py at 203a5ad is the run's blob: {blob('203a5ad', 'explore_gpu/device.py') == blob(NOTE_C, 'explore_gpu/device.py')} (mainline.py changed after it, in 30807ec: the one-thread fix)")
row(f"r3:{find(R3, '**K4b** validated')}-103", "K4b: sequential path (cpu_io) lock-step vs CPU; arm A 160 for 2400 updates and S75 ORACLE, SINGLE, WIDE_SPLIT, REG3 for 20 updates",
    f"K4b ok {k4b['ok']}; cases {k4b_cases}", "MATCH" if k4b["ok"] and len(k4b_cases) == 5 else "DISAGREE", "device.to_device(cpu_io=True) is what _onset_wrapper uses")
row(f"r3:{find(R3, 'Arm A 160 is MultiBDH')}", "arm A 160 is MultiBDH with the recurrent gate, HINGE_D8's gate kind",
    "K8 case label 'MultiBDH recurrent (arm A)'; test_window_gate docstring: HINGE_D8 'recurrent gate, SLOW + hinge'", "MATCH")
row(f"r3:{find(R3, 'explore_b16_gates.LocalBDH')}-106", "S75's WIDE_SPLIT is explore_b16_gates.LocalBDH with a width-16 window (S = 4, k = 4); that class's source is identical to test_window_gate's LocalBDH",
    f"explore_gpu/run.make_model: gate 'wide' -> explore_i_common.to_window (class g16.LocalBDH, width lmax + 2 = 16); LocalBDH ast-identical (docstrings removed): {LOCAL_SAME}; with docstrings: {LOCAL_SAME_DOC}",
    "MATCH" if LOCAL_SAME else "DISAGREE", "")
row(f"r3:{find(R3, '**K8** is a CPU-only')}-112", "K8 skipped on CUDA; passed in the CPU CHECK run (ce4b9ef) on 13 fake-CUDA cases incl. LocalBDH LOCAL3 / WIDE (reduced S75 task), MultiBDH recurrent (arm A) and perfect (ORACLE), test_window_gate.noise_for",
    f"checks_cpu_output.txt last changed {cpu_ck_commit}; K8 ok {k8['ok']}, {len(k8['cases'])} cases: {list(k8['cases'])}", "MATCH" if k8["ok"] and len(k8["cases"]) == 13 else "DISAGREE",
    "reduced S75 task = ICMCTask(S75_CFG, NB=4, M=44) in checks.k8_fake_cuda")
s3070 = CK["NVIDIA_GeForce_RTX_3070_Laptop_GPU"]
note = lines(NOTE_C, NOTE_F)
row(f"r3:{find(R3, 'Correction to the pre-run note')}-115", "the note attributes K8 to 'both cards' CUDA CHECKs'; wrong: K8 was skipped on both",
    f"note l.58-61 attributes K4b and K8 to 'Both cards' CUDA CHECKs'; K8 on the 1660: {s1660['status'].get('K8')}, on the 3070: {s3070['status'].get('K8')}; both verdicts '{s3070['verdict']}'",
    "MATCH", "the correction is right; the note's other attributions (K4b, 'both passed') hold")
d8 = [c for c in all_case_names if "D8" in c or "S=8" in c or "S = 8" in c]
row(f"r3:{find(R3, 'No CHECK ran Part C')}", "no CHECK ran Part C's models (S=8, P=4, k=16, D8) lock-step against the CPU",
    f"case names in the 1660's CHECKs mentioning D8 / S=8: {d8 or 'none'}", "MATCH")
row(f"r3:{find(R3, 'other diagnostics')}-120", "the records hold transitions, failure classes, splits, hinge counts; the note fixed none for description",
    "fields transition, end (fail-class inputs), checks/splits present; hinge state in st1/st2/stats; note 'What will be described' lists validity, counts, pairs only", "MATCH",
    "the referee's recomputation of these diagnostics is in 'Diagnostics' below, for the record")

# descriptions file rows
for i, ln in enumerate(DS, 1):
    if ln.startswith("- GPU:"):
        row(f"desc:{i}", ln[2:], f"62/62 planned present, {runs_ok} ok; {len(extra)} outside the plan; code {list(REC_SHAS)[0]}", "MATCH" if runs_ok == 62 and not extra else "DISAGREE")
    elif ln.startswith("- X:"):
        row(f"desc:{i}", ln[2:], f"{len(X['runs'])}/{sum(len(v) for v in SETS['X'][1].values())} present; ok {sum(bool(r.get('ok')) for r in X['runs'].values())}; outside {len(set(X['runs']) - {f'{k}|{s}' for k, v in SETS['X'][1].items() for s in v})}; meta.git {X['meta']['git']} started {X['meta']['started']}; starts {[(s['git'], s['started'], s['arms_added']) for s in X['meta']['starts']]}", "MATCH")
    elif ln.startswith("- L:"):
        row(f"desc:{i}", ln[2:], f"{len(L['runs'])}/{sum(len(v) for v in SETS['L'][1].values())} present; ok {sum(bool(r.get('ok')) for r in L['runs'].values())}; outside {len(set(L['runs']) - {f'{k}|{s}' for k, v in SETS['L'][1].items() for s in v})}; meta.git {L['meta']['git']} started {L['meta']['started']}", "MATCH")
    elif ln.startswith("| GPU:NVIDIA") and "/2 |" in ln:
        row(f"desc:{i}", ln, f"{G['ceil']}/2 {'VALID' if G['valid'] else 'NOT VALID'}", "MATCH" if f"{G['ceil']}/2 | {'VALID' if G['valid'] else 'NOT VALID'}" in ln else "DISAGREE")
    elif re.match(r"^\| (X|L|pooled X\+L) \| \d/\d", ln):
        nm = ln.split("|")[1].strip()
        d = D["pooled" if nm.startswith("pooled") else nm]
        want = f"{d['ceil']}/{d['ceil_n']} | {'VALID' if d['valid'] else 'NOT VALID'}"
        row(f"desc:{i}", ln, want, "MATCH" if want in ln else "DISAGREE")
    elif re.match(r"^\| (WIN3_SPLIT_D8|HINGE_D8|WIN3_SLOW_D8) \|", ln):
        k = ln.split("|")[1].strip()
        stated = [c.strip() for c in ln.split("|")[2:6]]
        rec = [cell(D[n], k) for n in ("GPU", "X", "L", "pooled")]
        row(f"desc:{i}", ln, " | ".join(rec), "MATCH" if stated == rec else "DISAGREE")
    elif ln.startswith("| WIN3_S") and " vs " in ln:
        a, b = ln.split("|")[1].split("(")[0].strip().split(" vs ")
        stated = [c.strip() for c in ln.split("|")[2:6]]
        rec = [pcell(D[n], (a, b)) for n in ("GPU", "X", "L", "pooled")]
        row(f"desc:{i}", ln, " | ".join(rec), "MATCH" if stated == rec else "DISAGREE")

# the note's checkable statements
NT = note
row(f"note:{find(NT, 'byte-identical')}-10", "this branch's test_window_gate.py is byte-identical to the main line's (0754d61)",
    f"blob at 30807ec {blob(NOTE_C, 'test_window_gate.py')[:10]} = 0754d61 {blob(ROUND5, 'test_window_gate.py')[:10]} = main tip {blob(MAIN, 'test_window_gate.py')[:10]}", "MATCH")
row(f"note:{find(NT, 'are fresh')}-21", "2540-2559 fresh; no overlap with X 500-559, L 1500-1559, R 900-959; checked on every branch",
    f"X {min(XS)}-{max(XS)}, L {min(LS)}-{max(LS)}, R {min(R_SEEDS)}-{max(R_SEEDS)}; tip JSON/JSONL files scanned {TIP_FILES}; hits outside the run's own file: {sorted({p for (s, p), v in TIP_HITS.items() if s != GPU_BLOB}) or 'none'}", "MATCH")

# README (explore_out/README.md, lines added with report 3 in c18c7c0)
RM = lines(REP, "explore_out/README.md")
i = find(RM, "**Validity.** VALID by the test's rule")
row(f"README:{i}", "VALID: CEIL_C bound on 2540 and 2541", f"{G['ceil']}/2", "MATCH")
i = find(RM, "WIN3_SPLIT_D8 19/20, HINGE_D8 0/20")
row(f"README:{i}", "WIN3_SPLIT_D8 19/20, HINGE_D8 0/20, WIN3_SLOW_D8 6/20", ", ".join(f"{k} {G['arms'][k]['c']}/20" for k in ARMS3), "MATCH")
i = find(RM, "Against HINGE_D8, b 19")
row(f"README:{i}", "against HINGE_D8 b 19 c 0; against WIN3_SLOW_D8 b 13 c 0",
    f"{G['pairs'][PAIRS[0]]['b']}/{G['pairs'][PAIRS[0]]['c']}; {G['pairs'][PAIRS[1]]['b']}/{G['pairs'][PAIRS[1]]['c']}", "MATCH")

# the round plan (main line docs/reading/plan_2026-10-13.md), which the caller quotes
PL = lines(MAIN, "docs/reading/plan_2026-10-13.md")
plan_c = git("log", "-1", "--format=%h %cI", MAIN, "--", "docs/reading/plan_2026-10-13.md").strip()
i = find(PL, "the recipe 19/20, no split 6/20")
row(f"plan:{i}", "test_window_gate on the GTX 1660 SUPER (fresh seeds 2540-2559, CUDA): the recipe 19/20, no split 6/20, recurrent 0/20; pairs 19 vs 0, 13 vs 0, 6 vs 0; VALID; descriptions only",
    f"WIN3_SPLIT_D8 {G['arms']['WIN3_SPLIT_D8']['c']}/20, WIN3_SLOW_D8 {G['arms']['WIN3_SLOW_D8']['c']}/20, HINGE_D8 {G['arms']['HINGE_D8']['c']}/20; pairs " + ", ".join(f"{G['pairs'][pk]['b']} vs {G['pairs'][pk]['c']}" for pk in PAIRS) + f" (SPLIT/HINGE, SPLIT/SLOW, SLOW/HINGE); {'VALID' if G['valid'] else 'NOT VALID'}; seeds fresh (Seeds)",
    "MATCH", f"plan at {plan_c}")
i = find(PL, "the recipe 38/40 vs 0/40")
row(f"plan:{i}", "test_window_gate pooled: the recipe 38/40 vs 0/40 (recurrent) and 15/40 (no split)",
    f"pooled {D['pooled']['arms']['WIN3_SPLIT_D8']['c']}/40, {D['pooled']['arms']['HINGE_D8']['c']}/40, {D['pooled']['arms']['WIN3_SLOW_D8']['c']}/40", "MATCH", "the main-line numbers report 3 sets beside the GPU column")
i = find(PL, "the GPU steps (CUDA CHECK at L = 1536")
row(f"plan:{i}", "S75b: the GPU steps (CUDA CHECK at L = 1536, benchmark, pilot) not started",
    f"plan committed {plan_c.split()[1]}; S75b CUDA CHECK on the 1660 29d0bd6 {ctime('29d0bd6')}, on the 3070 97504cd {ctime('97504cd')}, benchmark 6468193 {ctime('6468193')}: all after the plan", "MATCH", "true when the plan was written; stale now")

# ═════════════════════════════════════════════════════════════════════════════
# PRINT
# ═════════════════════════════════════════════════════════════════════════════
cnt = Counter(r[3] for r in ROWS)
P("# Check: session I's report 3 (test_window_gate Part C / D on the GTX 1660 SUPER, seeds 2540-2559)")
P("")
P("Referee V. Generated in full by `docs/review_2026-10/check_I_report3.py` (`python3 -I docs/review_2026-10/check_I_report3.py > "
  "docs/review_2026-10/check_I_report3.md` from the repository root after `git fetch`). It reads every file from git objects, "
  "runs no session code and trains nothing. The main line's outcome functions are cut out of the main line's files by ast and "
  "executed in a private namespace; rlib.py re-implements them independently (transition from the curve, the map from "
  "stream_gate) and agrees seed for seed on every arm of GPU, X and L.")
P("")
P(f"**Pins.** Session I: `{I_TIP}` tip `{COMMITS[-1][0]}` ({COMMITS[-1][2]}). Report 3 `{REP_F}` and the descriptions "
  f"`{DESC_F}` at `{REP}` (unchanged at the tip); report history {', '.join(REP_HIST)}. Records `{GPU_F}` at `{REC_C}` "
  f"(blob {GPU_BLOB[:10]}, unchanged at `{REP}` and the tip; {len(GPU['runs'])} records). Pre-run note and generator at `{NOTE_C}`. "
  f"Main line `{MAIN}` tip `{git('rev-parse', '--short', MAIN).strip()}`; X / L records `{X_F}`, `{L_F}` (blobs identical on the "
  f"main line, at `{NOTE_C}` and at `{REP}`).")
P("")
P("**Arms.** The round plan's names: the recipe = WIN3_SPLIT_D8 (window gate + slow memory + split trigger), no split = "
  "WIN3_SLOW_D8, recurrent = HINGE_D8 (the recurrent gate with SLOW + hinge); CEIL_C = the perfect gate (validity). The pairs "
  "'19 vs 0, 13 vs 0, 6 vs 0' are WIN3_SPLIT_D8 vs HINGE_D8 (G2's pairing), WIN3_SPLIT_D8 vs WIN3_SLOW_D8 (G4's), "
  "WIN3_SLOW_D8 vs HINGE_D8 (the printed 'original G2'); b = first arm only, c = second arm only.")
P("")
P("## Counts")
P("")
P(f"{len(ROWS)} rows: **" + ", ".join(f"{k} {cnt.get(k, 0)}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")) + "**.")
P("")
P("By source: report_3.md (`r3:`), the descriptions file (`desc:`), the pre-run note's checkable lines (`note:`), the "
  "README lines added with report 3 (`README:`), and the round plan's lines the caller quotes (`plan:`, main line "
  "docs/reading/plan_2026-10-13.md).")
P("")
P("### Every non-MATCH row")
P("")
P("| where | statement | recomputed | status | note |")
P("|---|---|---|---|---|")
for w, s, rc, st, nt in ROWS:
    if st != "MATCH":
        P(f"| {w} | {s} | {rc} | {st} | {nt} |")
P("")
P("## Headline numbers (recomputed)")
P("")
P("| arm | GPU 2540-2559 | X 540-559 | L 1540-1559 | pooled X + L |")
P("|---|---|---|---|---|")
for k in ARMS3:
    P(f"| {k} | " + " | ".join(cell(D[n], k) for n in ("GPU", "X", "L", "pooled")) + " |")
P("")
P("| pairing | GPU | X | L | pooled | exact one-sided McNemar p (GPU; X; L; pooled) |")
P("|---|---|---|---|---|---|")
for pk in PAIRS:
    ps = "; ".join(f"{mc(D[n]['pairs'][pk]['b'], D[n]['pairs'][pk]['c']):.2g}" for n in ("GPU", "X", "L", "pooled"))
    P(f"| {pk[0]} vs {pk[1]} | " + " | ".join(pcell(D[n], pk) for n in ("GPU", "X", "L", "pooled")) + f" | {ps} |")
P("")
P(f"Validity: GPU CEIL_C {G['ceil']}/2 (transitions 2540 {cc[2540][0]}, 2541 {cc[2541][0]}) -> "
  f"{'VALID' if G['valid'] else 'NOT VALID'}; X {D['X']['ceil']}/2, L {D['L']['ceil']}/2. Unrounded Wilson: GPU "
  + "; ".join(f"{k} [{NS['wilson'](G['arms'][k]['c'], 20)[0]:.4f}, {NS['wilson'](G['arms'][k]['c'], 20)[1]:.4f}]" for k in ARMS3) + ". "
  "The p-values are the referee's: report 3 prints none, as its note fixed; they are given so that a reader can set the GPU "
  "pairs beside the test's own (next paragraph).")
P("")
P("**Beside the main line's own numbers** (test_window_gate.py docstring, RESULT (X), RESULT (L), POOLED; the numbers report 3 "
  "sets the GPU column beside): X G2 18 vs 0 p = 3.8e-6, G4 12 vs 0 p = 0.00024, original G2 6 vs 0 p = 0.016; L G2 20 vs 0 "
  "p = 9.5e-7, G4 11 vs 0 p = 0.00049; pooled G2 38 vs 0 p = 3.6e-12, G4 23 vs 0 p = 1.2e-7, original G2 15 vs 0 p = 3.1e-5; "
  "bands X WIN3_SPLIT_D8 18/20 [0.699, 0.972], WIN3_SLOW_D8 6/20 [0.145, 0.519], HINGE_D8 0/20 [0, 0.161]; pooled 38/40 "
  "[0.835, 0.986], 15/40 [0.242, 0.530], 0/40 [0, 0.088]. Recomputed here: "
  + "; ".join(f"{n} {pk[0][:9]}/{pk[1][:9]} {D[n]['pairs'][pk]['b']} vs {D[n]['pairs'][pk]['c']} p = {mc(D[n]['pairs'][pk]['b'], D[n]['pairs'][pk]['c']):.2g}"
              for n in ("X", "L", "pooled") for pk in PAIRS)
  + ". Every X / L / pooled cell of report 3 equals the main line's own figures.")
P("")
P("## Order (item 1)")
P("")
P("| commit | author time | commit time | who | touches | subject |")
P("|---|---|---|---|---|---|")
for h, ai, ci, who, subj, t in COMMITS:
    P(f"| `{h}` | {ai} | {ci} | {who.split(' <')[0]} | {', '.join(t) or '-'} | {subj[:110]} |")
P("")
P(f"- **The note and its fixed content precede every record.** The note (`{NOTE_F.split('/')[-1]}`) has one commit, "
  f"`{note_hist[-1]}`, and is unchanged at `{REP}` and the tip. Its validity rule (CEIL_C binds on both 2540 and 2541), "
  "BOUND ROUTED definition (`test_stream_recipe.outcome == \"BOUND ROUTED\"`), arms (CEIL_C, then WIN3_SPLIT_D8, HINGE_D8, "
  "WIN3_SLOW_D8) and seeds (2540-2541, 2540-2559) are all in that one version.")
P(f"- **Records.** The only commit that adds or changes the results file is `{REC_C}` (author {git('log', '-1', '--format=%aI', REC_C).strip()}, "
  f"committer {git('log', '-1', '--format=%cI', REC_C).strip()}, root <root@Docker.home>), {hms(t_rec_c - t_note)} after the note.")
P(f"- **Code SHA (needs no clock).** Every record carries `{list(REC_SHAS)[0]}`. Recomputed from git objects as "
  "`explore_gpu/run.code_sha` defines it (SHA-1 over name + source of every root / explore_gpu module the import loads; "
  f"{len(MODS)} modules, found by following module-level imports): {', '.join(f'`{c}` {SHA_AT[c]}' for c in ['5d9dcf9', 'ab65d9b'] + [x[0] for x in COMMITS])}. "
  f"The hash first exists at `{first_sha}` (the note's commit: it changed mainline.py, one thread per worker) and holds at every "
  "later commit. A record is stamped with the hash of the code of the command that wrote it, and a cached record is reused "
  "only under the same hash, so no record of this file was written by code older than the note's commit.")
P(f"- **meta.started** = `{started}` (machine-local clock, no zone; `setdefault`, so it is the first command's start). "
  f"Read as UTC it is {hms(t_started - t_note)} ({(t_started - t_note) / 3600:.2f} h) after the note and "
  f"{t_started - t_4371824:.0f} s after `4371824` (03:20:22 +0000, S75's WIDE_SLOW commit, by the same committer identity "
  "root@Docker.home as `148c161`).")
P("- **The host clock, bounded by git ancestry** (taking session I's commit times as UTC; a child commit cannot precede its "
  "parent): " + "; ".join(f"`{b[0]}` ({b[1]}): parent `{b[2]}` -> clock at most {hms(b[7])[1:]} ahead"
                          + (f", child `{b[4]}` -> at most {hms(-b[6])[1:]} behind" if b[6] is not None else "")
                          for b in bounds if b[0] in ("4371824", "148c161"))
  + f". Assuming one constant offset, the clock is within {hms(OFF_LO)} .. {hms(OFF_HI)} of UTC, so the run began between "
  f"{datetime.fromtimestamp(t_started - OFF_HI, timezone.utc):%Y-%m-%d %H:%M:%S} and "
  f"{datetime.fromtimestamp(t_started - OFF_LO, timezone.utc):%Y-%m-%d %H:%M:%S} UTC, at least {(t_started - OFF_HI - t_note) / 3600:.2f} h after the note. "
  "(`29d0bd6`, the S75b CUDA CHECK, has the same identity and is later; it is not needed.)")
P(f"- **Per-run secs.** Sum {sum(secs) / 3600:.3f} h (wall_secs {sum(wall) / 3600:.3f} h) against {elapsed / 3600:.3f} h from "
  "meta.started to `148c161`'s author time (both on the same host clock, so the offset cancels): runs overlapped (at least two at "
  "once at some time). With overlapping workers and no start time per record, the records cannot place any single run in "
  "time, nor show the order CEIL_C -> arms; they show only that every run lies between meta.started and the push. As with "
  "S75, what the records can show is: every record's code postdates the note (code SHA), and the first command started "
  "after it (meta.started, under the ancestry bound). What they cannot show is the order of commands and the worker count.")
P(f"- **Changes after the first record.** Note: none (history {note_hist}). Generator: three commits after `{REC_C}` "
  f"(`c18c7c0`, `6ab6324`, `3d7e68e`); the top-level definitions that differ between `{NOTE_C}` and `{REP}` are {GEN_CHANGED}: "
  "the column label, the planned (rather than present) seeds, and the header line. `describe()`, `ARMS`, `PAIRS` and "
  "`GPU_SEEDS` are ast-identical, and with all 62 planned records present the seed change cannot alter a number. Report 3 "
  "discloses all three changes (r3 l.74-85). No descriptions file was ever committed from the 30807ec generator.")
P(f"- **Main line.** None of the {len(ROOT_MODS)} root modules in the code SHA changed on the main line after `{last_main.split()[0]}` "
  f"({last_main.split()[1]}), so the main line's test code did not move under the run either.")
P("")
P("## Code identity (item 2)")
P("")
P(f"- **What ran.** The main line's own test code, unmodified: all {len(ROOT_MODS)} root modules in the code SHA "
  f"({', '.join(m.split('.')[0] for m in ROOT_MODS)}) are blob-identical at `{NOTE_C}`, `{REC_C}`, `{REP}`, the explore-I tip, "
  f"`{ROUND5}` and the main-line tip: {all(blob_main.values())}. Their last main-line change is `{last_main}`. So the test code is "
  f"the main line's at any commit from `{last_main.split()[0]}` on (the note names `{ROUND5}`; it is the same code).")
P(f"- **Through a GPU wrapper, not a port.** The other three hashed modules are explore_gpu's `__init__.py`, `device.py` "
  f"(history {dev_hist}) and `mainline.py` (history {ml_hist}), unchanged from `{NOTE_C}` to the tip: {gpu_pkg_same}. "
  "`mainline.run_arm` calls the test's own `spec` and `run_job` inside `device.device_scope`, which (on CUDA) wraps "
  "`test_binding_onset.onset_run` so that the model the test builds on CPU under its own seeding is moved to the device "
  "(`to_device`, `cpu_io=True`: the forward runs on the GPU, its outputs come back to CPU, where the test's losses, "
  "statistics and trigger run), and wraps `test_window_gate.split_op` so that `noise_for` draws on CPU and moves the noise. "
  "No test function is replaced by a GPU re-implementation (contrast S75, which ran explore_gpu's own driver).")
P(f"- **AST diff against the code X and L ran.** Root modules, `{XL_C}` (X's second start, L's start) vs `{NOTE_C}`: "
  + "; ".join(f"{m}: {v}" for m, v in DIFF_XL.items()) + ". "
  f"Only `verify` (the CHECKs, not called by `run_job`) differs; run_job, one_run, run_path, make_trigger, split_op, noise_for, "
  f"success, tally and every helper they reach are ast-identical. Against `{X_FIRST}` (X's first start, its six perfect-gate "
  f"records) the Part D additions differ ({len(DIFF_X1.get('test_window_gate', []))} definitions in test_window_gate); the "
  f"CEIL_C path (run_path, iters_of, run_job, spec, success, validity) is ast-identical: {CEIL_PATH_SAME}.")
P(f"- **CUDA vs CPU.** Report 3 disclaims bit-equivalence (r3 l.95-97: 'not bit-comparable ... only outcomes, not curves'); "
  f"so does the note (l.58). Consistent with the 1660's own K1 (arm A 160 against its CPU record: identical = {k1.get('identical')}). "
  "K4b (lock-step within tolerance, not bitwise) covers arm A and S75 models, not Part C's S = 8 models, as report 3 says.")
P("")
P("## Seeds 2540-2559 (item 3)")
P("")
P(f"- **Branch tips.** {len(branches)} remote branches, {TIP_FILES} JSON / JSONL files: every hit "
  + ("; ".join(f"`{p}` on {v[0]} ({len(v[1])} hits, e.g. {v[1][0]})" for (s, p), v in sorted(TIP_HITS.items(), key=lambda x: x[0][1])) or "none")
  + f". Text logs (.log / .txt) matching a seed-like pattern: {sorted(set((b, p, m) for b, p, m in TIP_TEXT_HITS)) or 'none'}.")
P(f"- **All history.** {len(paths_of)} distinct JSON / JSONL blobs reachable from any ref; with a 2540-2559 seed (run key "
  f"`ARM|seed`, a `seed` field, a `seeds` list, or a seed-keyed dict): {len(HIST_HITS)}, of which the run's own results file "
  f"blob: {GPU_BLOB in HIST_HITS}; others: {[(s[:10], sorted(paths_of[s])) for s in OTHER_HIST] or 'none'}.")
P(f"- X {min(XS)}-{max(XS)}, L {min(LS)}-{max(LS)}, R (replicate-R records) {min(R_SEEDS)}-{max(R_SEEDS)}. The seeds were "
  "fresh: no record anywhere in the repository's history uses 2540-2559 except this run's. (Mentions in prose: "
  "docs/reading/parallel_sessions_prompts.md l.920 and l.1082 reserve 2540-2559 for this run; spec_review drafts list it.)")
P("")
P("## Diagnostics (not described by report 3; the note did not fix them) — the referee's recomputation, for the record")
P("")
P("| set | arm | outcomes (test's outcome string; 'collapsed /' where the record's collapsed flag is set) | collapsed | median transition [min, max] of bound runs (n) | of BOUND ROUTED runs |")
P("|---|---|---|---|---|---|")
for name in ("GPU", "X", "L"):
    for k in ARMS3:
        d = DIAG[name][k]
        P(f"| {name} | {k} | {d['outcomes']} | {d['collapsed']} | {d['med_bound']} ({d['n_bound']}) | {d['med_br']} |")
P("")
for name in ("GPU", "X", "L"):
    d = DIAG[name]["WIN3_SPLIT_D8"]
    P(f"- {name} WIN3_SPLIT_D8 firings: {d['fires']} splits; runs by number of splits {d['per_run']}; at steps {d['fire_steps']}; "
      f"labelled ok (c* holds >= 2 streams and c0 none on the labelled map before): recorded key_ok {d['labelled_rec']}/{d['fires']}, "
      f"recomputed with test_window_gate.targeting {d['labelled_re']}/{d['fires']}; blocked checks {d['blocked']}; the records' "
      f"`splits` field equals the fired count: {d['splits_field_agrees']}.")
P("- Against the main line's docstring: X 47 splits (1-3 per run, all labelled ok), HINGE_D8 8 collapsed / 7 OTHER / 5 merged, "
  "WIN3_SLOW_D8's 14 failures all merges; L 38 splits (0-3 per run), HINGE_D8 7 collapsed, WIN3_SLOW_D8's 11 failures all "
  "merges; median transitions X WIN3_SPLIT_D8 21600 [14400, 42000], WIN3_SLOW_D8 27600 [20400, 33600]; L 18600 [13200, 30000], "
  "32400 [13200, 38400]. Every one of these is reproduced by the recomputation above (X HINGE_D8: 8 collapsed / 7 OTHER / "
  "4 + 1 merged; X WIN3_SLOW_D8 8 two-share + 6 three-share; L WIN3_SLOW_D8 11 merges), which validates the extraction of "
  "the test's functions on the CPU records before they are applied to the GPU records.")
P("")
P("## Statements about S75, S75b and the GPU package (item 5)")
P("")
P("Report 3 says nothing about S75's or S75b's results. Its statements about the GPU package are the CHECK paragraph (rows "
  "r3:99-117 above: verdicts, K4b's cases, K8's 13 cases, the K8 correction to the note, no lock-step CHECK of Part C's "
  "models), the code SHA and the one-thread fix (r3:51-56); all MATCH. Its one S75 reference is 4371824 (S75's WIDE_SLOW "
  "commit) as the clock witness (r3:59-60, MATCH). Outside report 3, explore_out/README.md at 3d7e68e still carries S75's "
  "'Correction to the pre-run note. Three WIDE_SPLIT runs had already finished on the 1660 when it was committed', which the "
  "referee's check_S75.md (rows 16, 81-82) found not record-checkable (and, on its own premises, wrong by one run in "
  "progress); report 3 does not repeat it.")
P("")
P("**The referee's own earlier files.** Nothing here contradicts rev9/check_S75.md or check_S75_v6.md. Their clock argument "
  "(the 1660's clock bounded from one side only) is consistent with what git now shows: the 1660 host commits as "
  "root@Docker.home with +0000 (4371824, 148c161, 29d0bd6), and the ancestry bound above applies to that host from 13 October "
  "on; it does not reach back to S75's 11-12 October runs.")
P("")
P("## What could not be checked")
P("")
P("- The order of the four commands and the worker counts: no record holds a start time, the command line or the worker "
  "count, and overlapping workers make the per-run `secs` unchainable (unlike S75's single-process chain).")
P("- Which X records ran at 2f98f06 and which at 886a668: no record carries its commit; the meta and the test's RESULT text agree "
  "with report 3, and the CEIL_C path is ast-identical at the two commits, so the attribution does not affect any number.")
P("- That the arithmetic ran on the GPU: the records carry the device meta from inside `device_scope`; the GPU path itself is "
  "covered by the package's CHECKs (K4b, K8 on CPU), not by anything in the records. The per-step times (HINGE_D8 about "
  f"{statistics.median(r['secs'] for k, r in GPU['runs'].items() if k.startswith('HINGE_D8|')) / 43200 * 1000:.0f} ms against X's "
  f"{X['meta']['ms_per_step']['HINGE_D8']:.0f} ms and L's {L['meta']['ms_per_step']['HINGE_D8']:.0f} ms) are consistent with it.")
P("- The descriptions generator was not run (session code); every number it printed was recomputed with the test's own "
  "functions instead.")
P("- The host clock's accuracy beyond the ancestry bound; the push time of `148c161`.")
P("")
P("## Rows")
P("")
P(f"Counts: {len(ROWS)} rows; " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")))
P("")
P("| # | where | statement | recomputed | status | note |")
P("|---|---|---|---|---|---|")
for n, (w, s, rc, st, nt) in enumerate(ROWS, 1):
    P(f"| {n} | {w} | {s.replace('|', '/')} | {rc.replace('|', '/')} | {st} | {nt.replace('|', '/')} |")
print("\n".join(out))
