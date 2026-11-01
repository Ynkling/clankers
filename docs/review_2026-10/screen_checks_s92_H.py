#!/usr/bin/env python3
"""screen_checks_s92_H.py — Referee V's checks for the screen review of session H's round-7 screen (the prompt's S92; H's
"S90") on claude/explore-H at 23dccf24, with its CHECK log 117ded72.

    python3 -I docs/review_2026-10/screen_checks_s92_H.py > docs/review_2026-10/screen_checks_s92_H_out.txt

Run from the repository root after `git fetch`. Prints only. Every file is read with `git show`; nothing is trained, no
torch is imported and no session code is run as a program. H's explore_h_s90.py is exec'd from its source with the harness
stubbed (stores and runs_of served from synthetic records); the main line's transition() and outcome are cut out by ast.
"""
import ast
import collections
import itertools
import json
import math
import os
import re
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = "23dccf24"
LOG = "117ded72"
H = "origin/claude/explore-H"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
MAIN9C = "9c5939e"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"), stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return None


def say(*a):
    print(*a, flush=True)


# ══ 0. pins ══════════════════════════════════════════════════════════════════
say("== 0. Pins")
for r in (REF, LOG, H, MAIN, "origin/claude/explore-G", "origin/claude/explore-F"):
    say(f"   {r:<52} {git('log', '-1', '--format=%h %cI %s', r).strip()[:150]}")
say(f"   files changed {REF}: {git('show', '--stat', '--format=', REF).strip().splitlines()}")
say(f"   files changed {LOG}: {git('show', '--stat', '--format=', LOG).strip().splitlines()}")
say(f"   commits on {H} after {LOG}: {git('rev-list', '--count', LOG + '..' + H).strip()}")
lg = show(H, "explore_out/H/s90_phase1.log")
say(f"   s90_phase1.log at the head: {len(lg.splitlines())} lines; CHECK rows: {lg.count('CHECK ')}; 'passed': {lg.count('passed')}; "
    f"runs saved: {len(re.findall(r'^  \\[ *[0-9]+\\]', lg, re.M))}")
say("   " + "\n   ".join(lg.splitlines()[:4]))
ck = json.loads(show(H, "explore_out/H/checks.json"))
say(f"   checks.json at the head has H/s90: {'H/s90' in ck}")
say(f"   s90_results.json at the head: {show(H, 'explore_out/H/s90_results.json') is not None}")

# ══ 1. name ══════════════════════════════════════════════════════════════════
say("\n== 1. The screen's name: 'S90' / 'S92' on every head")
heads = [b.strip() for b in git("branch", "-r").splitlines() if "->" not in b]
for b in heads:
    try:
        out = git("grep", "-n", "-E", r"\bS9[02]\b", b, "--", "*.py", "*.md", "*.sh")
    except subprocess.CalledProcessError:
        continue
    files = collections.Counter(ln.split(":")[1] for ln in out.splitlines())
    s92 = sum(1 for ln in out.splitlines() if re.search(r"\bS92\b", ln))
    say(f"   {b}: {sum(files.values())} lines in {len(files)} files ({s92} with S92): {sorted(files)[:6]}")
pr = show(MAIN, "docs/reading/parallel_sessions_prompts.md").splitlines()
for i, ln in enumerate(pr, 1):
    if ln.startswith("## Session") and "round 7" in ln:
        say(f"   prompts:{i}: {ln[:120]}")
src = show(REF, "explore_h_s90.py")
say(f"   explore_h_s90.py:3 {src.splitlines()[2][:110]}")

# ══ 2. seeds ═════════════════════════════════════════════════════════════════
say("\n== 2. Seeds 3020-3059 (H's block) on every head and in history")
BLOCK = set(range(3020, 3060))
KEYRE = re.compile(r"^[A-Za-z0-9_.\-]+\|(-?\d+)(\|.*)?$")


def walk(o, hit, parent=""):
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRE.match(k) if isinstance(k, str) else None
            if m and int(m.group(1)) in BLOCK:
                hit.add(("key", k.split("|")[0], int(m.group(1))))
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool) and v in BLOCK:
                hit.add(("seed", parent, v))
            if k == "seeds" and isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and not isinstance(x, bool) and x in BLOCK:
                        hit.add(("seeds", parent, x))
            walk(v, hit, k)
    elif isinstance(o, list):
        for x in o:
            walk(x, hit, parent)


def parse(text):
    try:
        return [json.loads(text)]
    except ValueError:
        objs = []
        for ln in text.splitlines():
            try:
                objs.append(json.loads(ln))
            except ValueError:
                pass
        return objs


def scan(blobs):
    hits = {}
    p = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha in blobs:
        p.stdin.write((sha + "\n").encode())
        p.stdin.flush()
        size = int(p.stdout.readline().split()[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        h = set()
        for o in parse(data.decode("utf-8", "replace")):
            walk(o, h)
        if h:
            hits[sha] = h
    p.stdin.close()
    p.wait()
    return hits


tip = {}
for b in heads:
    for line in git("ls-tree", "-r", b).splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and path.endswith((".json", ".jsonl")):
            tip.setdefault(sha, []).append((b.split("/")[-1], path))
hits = scan(tip)
say(f"   {len(heads)} heads, {len(tip)} distinct JSON/JSONL blobs; blobs with a 3020-3059 key / seed / seeds entry: {len(hits)}")
for s, v in hits.items():
    say(f"     {tip[s][:2]}: {sorted(v)[:5]} ({len(v)} hits)")
hist = {}
for line in git("rev-list", "--all", "--objects").splitlines():
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl")):
        hist.setdefault(parts[0], []).append(parts[1])
hh = scan(hist)
say(f"   history: {len(hist)} distinct JSON/JSONL blobs; with 3020-3059: {len(hh)}")
for s, v in hh.items():
    say(f"     {hist[s][0]}: seeds {sorted({x[2] for x in v})}; kinds {sorted({x[0] for x in v})}")
pat = r"(^|[^0-9.e])(30[2-5][0-9])([^0-9]|$)"
txt = collections.defaultdict(set)
for b in heads:
    try:
        r = git("grep", "-n", "-E", pat, b, "--", "*.py", "*.md", "*.sh")
    except subprocess.CalledProcessError:
        r = ""
    for ln in r.splitlines():
        txt[ln.split(":")[1]].add(b.split("/")[-1])
say("   text mentions of 3020-3059 in .py/.md/.sh on the heads (path: branches):")
for p, bs in sorted(txt.items()):
    say(f"     {p}: {sorted(bs)[:4]}{' ...' if len(bs) > 4 else ''}")
g90 = show("origin/claude/explore-G", "explore_g_s90.py") or ""
f90 = show("origin/claude/explore-F", "explore_f_s90_ctx.py") or ""
say(f"   G's explore_g_s90.py seeds mentioned: {sorted(set(re.findall(r'\b(2[79][0-9]{2}|30[0-9]{2})\b', g90)))[:12]}")
say(f"   F's explore_f_s90_ctx.py seeds mentioned: {sorted(set(re.findall(r'\b(30[0-9]{2})\b', f90)))[:12]}")

# ══ 3. phase 1, exec'd ═══════════════════════════════════════════════════════
say("\n== 3. Phase 1: cell_reading / phase1 / report exec'd from explore_h_s90.py at " + REF)
SYN, STORE = {}, {"meta": {}, "runs": {}}


def _stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


_stub("explore_common", load_store=lambda name: STORE, cpu_model=lambda: "stub", OUT_DIR="/nonexistent")
_stub("explore_common16", runs_of=lambda screen, arm, seeds=None, store=None: dict(SYN.get(arm, {})), med=_med,
      STATE={"shas": {"H/s90": "stub"}}, code_sha=lambda *a: ("stub", []), key=lambda *a: "stub",
      run_child=lambda *a, **k: None)
_stub("explore_muon_scale")
_stub("explore_h_collapse")
_stub("explore_h_s89", FIELDS=(), NAME="H/s89", MAINREF="628eb7c", CHILD="explore_h_s89_child")
_stub("explore_h_s89b", NAME="H/s89b", DELTA_REC="8ae660d")


def load(name, ref, path=None):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(ROOT, "_nonexistent_", name + ".py")
    sys.modules[name] = m
    exec(compile(show(ref, path or name + ".py"), f"{name}.py@{ref}", "exec"), m.__dict__)
    return m


hc = load("explore_h_common", REF)
s90 = load("explore_h_s90", REF)

# 3a. cell_reading on every class pattern of four seeds
CL = {"<=43200": 12000, "43200 exactly": 43200, "44400-86400": 60000, "86400 exactly": 86400, "none": None}
names = collections.Counter()
bad = []
for pat in itertools.product(CL, repeat=4):
    trs = [CL[c] for c in pat]
    got = s90.cell_reading(trs)
    want = ("reliable" if all(t is not None and t <= 43200 for t in trs) else
            "needs the longer budget" if all(t is not None for t in trs) else "the oracle is the limit")
    names[got] += 1
    if got != want:
        bad.append((pat, got, want))
say(f"   cell_reading on all {5 ** 4} patterns of 4 seeds (5 transition classes): {dict(names)}; disagreements with the "
    f"prompt's text {len(bad)}; 3 seeds stored -> {s90.cell_reading([12000] * 3)!r}")

# 3b. the main line's transition(): a run read 'by 43200' from an 86400 curve vs a 43200 run
tbo = show(MAIN9C, "test_binding_onset.py")
tree = ast.parse(tbo)
ns = {"BIND_PASS": 0.95}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name == "transition":
        exec(compile(ast.Module([n], []), "transition@9c5939e", "exec"), ns)
tr = ns["transition"]


def curve(points, upto):
    """points: {step: acc}; others 0.5 before the first listed step and the last listed acc after; early stop at 3."""
    c, streak, last = [], 0, 0.5
    for st in range(1200, upto + 1, 1200):
        last = points.get(st, last if st > min(points) else 0.5)
        c.append([st, last, 0.0])
        streak = streak + 1 if last >= 0.95 else 0
        if streak >= 3:
            break
    return c


ex = {"(a) >= 0.95 at 42000 and 43200, 0.94 from 44400 on": {42000: 0.96, 43200: 0.96, 44400: 0.94},
      "(a') as (a), then >= 0.95 from 50400 on": {42000: 0.96, 43200: 0.96, 44400: 0.94, 50400: 0.97},
      "(b) >= 0.95 from 43200 on": {43200: 0.96},
      "(c) >= 0.95 from 40800 on": {40800: 0.96}}
for lab, pts in ex.items():
    full = curve(pts, 86400)
    t_full = tr(full)
    t_cut = tr([x for x in full if x[0] <= 43200])
    t_short = tr(curve(pts, 43200))
    say(f"   {lab}: 86400 run transition {t_full} -> H 'by 43200' {t_full is not None and t_full <= 43200}; "
        f"the same curve cut at 43200 {t_cut}; a 43200-budget run {t_short}")

# 3c. P* over every combination of the eight cell readings
R3 = {"reliable": [12000] * 4, "needs the longer budget": [12000, 12000, 12000, 60000],
      "the oracle is the limit": [12000, 12000, 12000, None]}


def recs(trs):
    return {s: dict(ok=True, transition=t, acc=0.97 if t else 0.8, curve=[[43200, 0.9, 0.0]], stopped_at=t)
            for s, t in zip(s90.P1_SEEDS, trs)}


pst = collections.Counter()
nonmono = pool4fail_p = 0
budgets = collections.Counter()
ex_nm = None
cells = list(s90.CELL)
for combo in itertools.product(R3, repeat=len(cells)):
    rd = dict(zip(cells, combo))
    out, complete, pstar, budget = s90.phase1(lambda c: recs(R3[rd[c]]))
    pst[pstar] += 1
    if pstar:
        budgets[tuple(sorted(budget.items()))] += 1
        below = [k for k in s90.POOLS if k < pstar and any(rd[f"{m}_P{k}"] == "the oracle is the limit" for m in s90.MEMS)]
        if below:
            nonmono += 1
            ex_nm = ex_nm or (pstar, {c: v for c, v in rd.items()})
        if any(rd[f"{m}_P4"] == "the oracle is the limit" for m in s90.MEMS):
            pool4fail_p += 1
say(f"   phase1() over all {3 ** 8} combinations of the 8 cell readings: P* {dict(sorted(pst.items(), key=str))}")
say(f"     P* > 4 with a smaller pool failing on some memory (non-monotone): {nonmono}; P* > 4 with pool 4 itself failing: "
    f"{pool4fail_p}; budgets named {dict(budgets)}")
say(f"     example: P* {ex_nm[0]} with {ex_nm[1]}")
out, complete, pstar, budget = s90.phase1(lambda c: recs(R3["reliable"]) if c != "DELTA_P8" else {3020: recs([1])[3020]})
say(f"     one cell with 1 of 4 records: complete {complete}, P* {pstar}")


def report_with(rd, p2=None):
    SYN.clear()
    for c in cells:
        SYN[c] = recs(R3[rd[c]])
    for a, v in (p2 or {}).items():
        SYN[a] = v
    res, _ = s90.report()
    return res


for lab, rd in (("every cell 'the oracle is the limit' (pool 4 too)", {c: "the oracle is the limit" for c in cells}),
                ("pool 4 reliable, 8/16/32 limit", {c: ("reliable" if c.endswith("_P4") else "the oracle is the limit") for c in cells}),
                ("all reliable (P* 32), no phase-2 record", {c: "reliable" for c in cells})):
    res = report_with(rd)
    say(f"   report(), {lab}: {res['readings']}; complete {res['complete']}")

# ══ 4. phase 2, exec'd on every composition ═════════════════════════════════
say("\n== 4. Phase 2: report() at P* = 16 (Hebbian reliable, delta needs the longer budget) on every composition of 10 seeds "
    "into the 8 per-seed patterns of (BASE, DELTA_SPLIT, DELTA_NOSPLIT)")
rd16 = {c: ("reliable" if (c.startswith("HEBB") and c != "HEBB_P32") or c in ("DELTA_P4", "DELTA_P8")
            else "needs the longer budget" if c == "DELTA_P16" else "the oracle is the limit") for c in cells}
PAT = list(itertools.product((0, 1), repeat=3))


def comps(n, k):
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        ext = (-1,) + bars + (n + k - 1,)
        yield tuple(ext[i + 1] - ext[i] - 1 for i in range(k))


def p2rec(b):
    return dict(ok=True, outcome="BOUND ROUTED" if b else "MERGED (2 share)", transition=24000 if b else None, acc=0.97 if b else 0.7,
                checks=[dict(step=4800, fired=True, acc=0.3, op_cs=1, op_c0=0, op_ok=True, map_before=list(range(8)))])


nm = collections.Counter()
bad, moot, hurt, n_all = [], 0, 0, 0
for c in comps(10, 8):
    seeds = [p for p, k in zip(PAT, c) for _ in range(k)]
    p2 = {a: {s: p2rec(seeds[i][j]) for i, s in enumerate(s90.SEEDS)} for j, a in enumerate(s90.P2)}
    res = report_with(rd16, p2)
    r = res["readings"]
    n_all += 1
    kb, ks, kn = (sum(x[j] for x in seeds) for j in range(3))
    cS = sum(1 for x in seeds if x[1] and not x[2])
    bN = sum(1 for x in seeds if x[2] and not x[1])

    def per(k):
        return "the recipe carries to P*" if k >= 7 else "the recipe does not carry to P*" if k <= 3 else "no reading applies"
    want_h, want_d = per(kb), per(ks)
    want_s = "the split is not needed on delta at P*" if cS - bN <= 1 else "the split is needed on delta"
    got_h = r["Hebbian (BASE)"].split(" (")[0]
    got_d = r["delta (DELTA_SPLIT)"].split(" (")[0]
    got_s = r["delta split"].split(" (")[0]
    nm[("H", got_h)] += 1
    nm[("D", got_d)] += 1
    nm[("S", got_s)] += 1
    if (got_h, got_d, got_s) != (want_h, want_d, want_s):
        bad.append(c)
    if got_s.startswith("the split is not needed") and ks <= 3 and kn <= 3:
        moot += 1
    if got_s.startswith("the split is not needed") and bN - cS >= 2:
        hurt += 1
say(f"   {n_all} compositions; disagreements with the docstring's rules {len(bad)}")
for k in sorted(nm):
    say(f"     {k[0]} {k[1]}: {nm[k]}")
say(f"   'the split is not needed' printed with DELTA_SPLIT <= 3/10 and DELTA_NOSPLIT <= 3/10 (moot): {moot}; printed with "
    f"DELTA_NOSPLIT ahead by >= 2 pairs (outside a two-sided 'within 1 pair'): {hurt}")
seeds = [(0, 0, 0)] * 10
p2 = {a: {s: p2rec(0) for s in s90.SEEDS} for a in s90.P2}
say(f"   example, all 30 runs unbound: {report_with(rd16, p2)['readings']}")
p2 = {a: {s: p2rec(1) for s in s90.SEEDS} for a in s90.P2}
p2["BASE"] = {s: p2rec(1) for s in s90.SEEDS[:9]}
say(f"   example, BASE 9 of 10 stored: {report_with(rd16, p2)['readings']}")

# ══ 5. operating characteristics ═════════════════════════════════════════════
say("\n== 5. Operating characteristics (the referee's own; V's model: with probability rho one shared U ~ U(0,1) decides "
    "every arm, else independent)")


def binom_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


rates = (0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
say("   P(>= 7/10): " + ", ".join(f"{p}: {binom_ge(7, 10, p):.3f}" for p in rates))
say("   P(<= 3/10): " + ", ".join(f"{p}: {1 - binom_ge(4, 10, p):.3f}" for p in rates))
say("   P(4/4) = q^4: " + ", ".join(f"{q}: {q ** 4:.3f}" for q in (0.99, 0.95, 0.9, 0.8, 0.7, 0.5)))
say("   P(both memories 4/4 at a pool) = (q_h q_d)^4 at q_h = q_d: " + ", ".join(f"{q}: {q ** 8:.3f}" for q in (0.99, 0.95, 0.9, 0.8)))
say("   P(a cell reads 'the oracle is the limit' | q) = 1 - q^4 (one failure in four): " +
    ", ".join(f"{q}: {1 - q ** 4:.3f}" for q in (0.99, 0.95, 0.9, 0.8)))
say("   Wilson 95%% of 3/4: %.2f-%.2f; of 4/4: %.2f-%.2f" % (hc.wilson(3, 4) + hc.wilson(4, 4)))


def pstar_dist(q):
    """q: {pool: per-run probability of binding by 86400, both memories alike, independent cells}."""
    d = collections.Counter()
    for bits in itertools.product((0, 1), repeat=4):
        pr = math.prod((q[k] ** 8 if b else 1 - q[k] ** 8) for k, b in zip(s90.POOLS, bits))
        ok = [k for k, b in zip(s90.POOLS, bits) if b]
        d[max(ok) if ok else None] += pr
    return {k: round(v, 3) for k, v in sorted(d.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}


for lab, q in (("q 0.95 at every pool", {4: .95, 8: .95, 16: .95, 32: .95}),
               ("q 0.9 at every pool", {4: .9, 8: .9, 16: .9, 32: .9}),
               ("q 1.0 / 0.95 / 0.9 / 0.7 at 4 / 8 / 16 / 32", {4: 1.0, 8: .95, 16: .9, 32: .7}),
               ("q 1.0 / 0.9 / 0.7 / 0.5", {4: 1.0, 8: .9, 16: .7, 32: .5})):
    say(f"   P* distribution, {lab}: {pstar_dist(q)}")


def probs(rates, rho):
    pats_ = list(itertools.product((0, 1), repeat=len(rates)))
    out_ = []
    for p in pats_:
        ind = math.prod(r if b else 1 - r for r, b in zip(rates, p))
        lo = max([0.0] + [r for r, b in zip(rates, p) if not b])
        hi = min([1.0] + [r for r, b in zip(rates, p) if b])
        out_.append(rho * max(0.0, hi - lo) + (1 - rho) * ind)
    return pats_, out_


def oc2(split, nosplit, rho):
    pats_, pr = probs((split, nosplit), rho)
    res = collections.Counter()
    for c in comps(10, 4):
        if any(k and q == 0 for k, q in zip(c, pr)):
            continue
        lp = math.lgamma(11) - sum(math.lgamma(k + 1) for k in c) + sum(k * math.log(q) for k, q in zip(c, pr) if k)
        cnt = dict(zip(pats_, c))
        cS = cnt[(1, 0)]
        bN = cnt[(0, 1)]
        ks = cnt[(1, 0)] + cnt[(1, 1)]
        kn = cnt[(0, 1)] + cnt[(1, 1)]
        p = math.exp(lp)
        if cS - bN <= 1:
            res["not needed (H)"] += p
            if ks <= 3 and kn <= 3:
                res["moot"] += p
        if abs(cS - bN) <= 1:
            res["two-sided"] += p
    return res


for sp, ns_ in ((0.9, 0.9), (0.9, 0.7), (0.9, 0.5), (0.7, 0.7), (0.7, 0.4), (0.3, 0.3), (0.1, 0.1), (0.5, 0.9)):
    rs = [oc2(sp, ns_, r) for r in (0.0, 0.5, 1.0)]
    say(f"   SPLIT {sp} / NOSPLIT {ns_}: 'not needed' (H's one-sided rule) " + "/".join(f"{x['not needed (H)']:.2f}" for x in rs) +
        "; of which moot (both <= 3/10) " + "/".join(f"{x['moot']:.2f}" for x in rs) +
        "; two-sided |c-b| <= 1 " + "/".join(f"{x['two-sided']:.2f}" for x in rs))

# ══ 6. the main line's outcome; the delta recipe's code ═════════════════════
say("\n== 6. The main line's outcome and the delta recipe, by ast")


def extract(ref, path, names):
    s = show(ref, path)
    t = ast.parse(s)
    out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name) and tg.id in names:
                    out[tg.id] = ast.dump(n)
    return out


SOURCES = [("test_router_layout.py", ["MARGIN_PARTIAL", "ETA_SPLIT", "fail_class"]),
           ("test_scale_axes.py", ["fail_class_k"]),
           ("test_stream_channels.py", ["shared_max", "merged"]),
           ("test_stream_recipe.py", ["ROUTED_ACC", "routed", "outcome", "wilson"]),
           ("test_binding_onset.py", ["BIND_PASS", "STOP_AFTER", "transition"])]
for path, nms in SOURCES:
    a, b = extract(MAIN, path, nms), extract(MAIN9C, path, nms)
    say(f"   {path} {nms}: equal at the main head and {MAIN9C}: {a == b and len(a) == len(nms)}")
for nmx in (["with_delta", "copy_op", "BETA"],):
    a, b = extract(MAIN, "test_delta_recipe.py", nmx), extract("a581027", "test_delta_recipe.py", nmx)
    say(f"   test_delta_recipe.py {nmx}: equal at the main head and a581027: {{{', '.join(f'{k}: {a.get(k) == b.get(k)}' for k in nmx)}}}")
say(f"   explore_b16_main.finish sets outcome = tsr.outcome(rec): {'rec[\"outcome\"] = tsr.outcome(rec)' in show(REF, 'explore_b16_main.py')}")

# ══ 7. CHECK targets and pins ════════════════════════════════════════════════
say("\n== 7. The CHECK targets: records, CPUs, first commits")
mx = json.loads(show("628eb7c", "results/X/window_gate_results.json"))["runs"]["WIN3_SPLIT_D8|540"]
mh = (json.loads(show(MAIN, "results/X/window_gate_results.json") or "{}").get("runs") or {}).get("WIN3_SPLIT_D8|540")
say(f"   WIN3_SPLIT_D8|540 at 628eb7c: cpu {mx.get('cpu')}; transition {mx.get('transition')}; fired "
    f"{[c['step'] for c in mx.get('checks') or [] if c.get('fired')]}; identical at the main head: {mh == mx}")


def store_rec(ref, path, arm, seed):
    st = json.loads(show(ref, path) or '{"runs": {}}')
    for k, r in st["runs"].items():
        a, s = k.split("|")[:2]
        if a == arm and int(s) == seed and r.get("ok"):
            return k, r
    return None, None


for path, arm, seed in (("explore_out/H/s89_results.json", "ORACLE", 2900), ("explore_out/H/s89b_results.json", "ORACLE_DELTA", 2913),
                        ("explore_out/H/s89_results.json", "BASE", 2902)):
    k, r = store_rec(REF, path, arm, seed)
    firsts = git("log", "--format=%h %cI", "--reverse", "-S", f'"{arm}|{seed}|', H, "--", path).strip().splitlines()
    first = firsts[0] if firsts else None
    k0, r0 = store_rec(first.split()[0], path, arm, seed) if first else (None, None)
    say(f"   {arm}|{seed}: key {k}; cpu {r and r.get('cpu')}; transition {r and r.get('transition')}; fired "
        f"{[c['step'] for c in (r or {}).get('checks') or [] if c.get('fired')]}; first committed {first}; identical then and at {REF}: {r0 == r}")
xd = json.loads(show("8ae660d", "results/X/delta_recipe_results.json"))["runs"]
for k in ("DELTA_SPLIT|583", "DELTA_SLOW|583", "CEIL_DELTA|580"):
    say(f"   X {k} at 8ae660d: cpu {xd[k].get('cpu')}; transition {xd[k].get('transition')}; fired "
        f"{[c['step'] for c in xd[k].get('checks') or [] if c.get('fired')]}")
for f in ("explore_h_s89_child.py", "explore_h_s89b_child.py", "explore_h_ops.py", "explore_b16_main.py", "explore_common16.py"):
    revs = git("log", "--format=%h", "4360cafa.." + REF, "--", f).split()
    say(f"   {f}: changed between 4360cafa and {REF}: {revs or 'no'}")
say(f"   explore_h_s89_child's pool-4 CHECK (PoolTask(4) = TASK48 bit for bit) is in S89's checks(), not in S90's check(): "
    f"{'PoolTask(4) batches equal' in show(REF, 'explore_h_s89_child.py')} / "
    f"{'PoolTask(4)' in show(REF, 'explore_h_s90_child.py') or 'PoolTask(4)' in src}")
a87 = json.loads(show(REF, "explore_out/H/allmass2_results.json") or '{"runs": {}}')["runs"]
o87 = sorted((k.split("|")[0], k.split("|")[1], r.get("cpu"), r.get("transition")) for k, r in a87.items() if k.startswith("ORACLE"))
say(f"   H's own pool-4 perfect-gate records (S87a, allmass2_results.json): {o87[:6]}")

# ══ 8. the budget's path to the child ════════════════════════════════════════
say("\n== 8. The budget: payload() exec'd")
s90.PLAN.clear()
s90.PLAN.update(K=16, budget={"HEBB": 43200, "DELTA": 86400})
for a in ("HEBB_P4", "DELTA_P32", "BASE", "DELTA_SPLIT", "DELTA_NOSPLIT"):
    say(f"   payload({a}, seed) = {s90.payload(a, 3024)}")
ch = show(REF, "explore_h_s90_child.py")
say(f"   child: 'total = p.get(\"total\", p[\"iters\"])' {('total = p.get(\"total\", p[\"iters\"])' in ch)}; b16.path(..., p[\"iters\"], ...) "
    f"{ch.count('p[\"iters\"], h89c.pool_rc')} calls; trigger last 'total - 2400' {ch.count('total - 2400')}")
ms = show(REF, "explore_muon_scale_child.py")
say(f"   explore_muon_scale_child.path: sched total=iters: {'dict(tscur.REAL, total=iters)' in ms}")
c16src = show(REF, "explore_common16.py")
say(f"   explore_common16._job calls mod.run_job(arm, seed): {'rec = mod.run_job(j[\"arm\"], j[\"seed\"])' in c16src}; "
    f"run_pool uses the fork context: {'get_context(\"fork\")' in c16src}")

# ══ 9. printing ══════════════════════════════════════════════════════════════
say("\n== 9. What the driver prints")
run = show(REF, "explore_h_run.py")
for i, ln in enumerate(run.splitlines(), 1):
    if "readings" in ln and "print" in ln:
        say(f"   explore_h_run.py:{i}: {ln.strip()[:130]}")
rep = show(REF, "explore_h_report.py")
say(f"   explore_h_report.py mentions s90: {'s90' in rep}; CAVEAT / OC strings inside the readings dict: "
    f"{'CAVEAT' in src.split('def report')[1].split('res[\"readings\"] = rd')[0].split('rd = {}')[1]}")

# ══ 10. push ═════════════════════════════════════════════════════════════════
say("\n== 10. Push after every saved run; no pull")
say(f"   'pull' in explore_h_run.py code lines: {[i for i, ln in enumerate(run.splitlines(), 1) if 'pull' in ln and not ln.strip().startswith('#')]}")
say(f"   --push-each is opt-in (store_true): {'ap.add_argument(\"--push-each\", action=\"store_true\")' in run}; "
    f"push output discarded: {'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL' in run}")
say(f"   commits on {H} since {REF} whose subject says 'saved': "
    f"{len([s for s in git('log', '--format=%s', REF + '..' + H).splitlines() if 'saved' in s])}")
