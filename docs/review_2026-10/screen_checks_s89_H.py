#!/usr/bin/env python3
"""screen_checks_s89_H.py: the referee's checks behind screen_review_s89_H.md (Session H's S89 screen,
claude/explore-H at d4a9a7d, CHECK log at 7820b12).

    python3 docs/review_2026-10/screen_checks_s89_H.py     (stdlib + numpy; reads git only; trains nothing;
                                                              writes screen_checks_s89_H_out.txt beside itself)

0. State: the branch heads, H's head, the CHECK log, any S89 store or results file on any head.
1. Seeds 2900-2959: every JSON / JSONL blob on every remote branch head, and every JSON / JSONL blob reachable
   in the history of every ref, parsed; "ARM|seed[|...]" keys, "seed" fields, "seeds" lists; integer dict keys
   reported apart (update steps); text mentions on the heads.
2. The readings: explore_h_s89.report() at d4a9a7d is exec'd from its source (git show) with the harness modules
   stubbed (the stubs only hand it synthetic run records; hc's paired / br / band / wilson are H's own code, exec'd
   from explore_h_common.py at d4a9a7d) on one representative of every distinct sufficient statistic over all
   16^10 per-seed outcome patterns of the four learned arms (all compositions of 10 seeds into 16 patterns);
   the names it prints are tabulated and compared with the round-6 prompt's rules re-implemented here.
3. The outcome: H's vendored fail_class_v2 against the main line's, byte for byte; test_stream_recipe.outcome,
   routed and their dependencies, syntax tree at 9c5939e (the served tree) against the main line at 7c4ceb8.
4. The CHECKs' records: the records the CHECK compares against, their CPUs and firings; the moving ref.
5. Operating characteristics of the readings by exact multinomial enumeration (BASE, T, NOSPLIT; 10 and 20 seeds;
   independent, half-shared and fully shared seed difficulty).
6. The task and the trigger: the pool task's statistics (numpy re-implementation of PoolTask.make_batch's
   distribution, not its bit stream; and exact values); the trigger's syntax-tree diff against S86's; what each
   target pair is computed from.
7. Other: F's task module at 50dc8e4; S86 + S87a's BASE / ALLMASS discordance, firings on one-to-one maps, run times.
"""
import ast
import collections
import difflib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a)


def show(ref, path):
    return git("show", f"{ref}:{path}").decode()


H, LOG, MAIN9C, MAIN, F6 = "d4a9a7d", "7820b12", "9c5939e", "7c4ceb8", "50dc8e4"
BLOCK = set(range(2900, 2960))

# ══ 0. state ═════════════════════════════════════════════════════════════════
say("== 0. State")
heads = [b for b in git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").decode().split()
         if not b.endswith("/HEAD")]
for b in heads:
    say(f"  {b} at {git('log', '-1', '--format=%h %cI', b).decode().strip()}")
say(f"  explore-H head: {git('log', '-1', '--format=%h %s', 'origin/claude/explore-H').decode().strip()}")
log = show(LOG, "explore_out/H/s89_oracle.log")
say(f"  s89_oracle.log at {LOG}: {len(log.splitlines())} lines; lines with 'CHECK': "
    f"{sum('CHECK' in ln for ln in log.splitlines())}; with 'ORACLE': {sum('ORACLE' in ln for ln in log.splitlines())}; "
    f"last line: {log.splitlines()[-1].strip()[:110]}")
HT = git("rev-parse", "--short", "origin/claude/explore-H").decode().strip()
if HT != LOG[:len(HT)]:
    lh = show(HT, "explore_out/H/s89_oracle.log").splitlines()
    say(f"  s89_oracle.log at the head {HT}: {len(lh)} lines; CHECK lines {sum('CHECK H/s89' in ln for ln in lh)} "
        f"(ok {sum('CHECK H/s89' in ln and ln.rstrip().endswith(': ok') for ln in lh)}); "
        f"'all CHECKs of H/s89 passed' {any('all CHECKs of H/s89 passed' in ln for ln in lh)}; run lines:")
    for ln in lh:
        if ln.strip().startswith(("start", "done", "saved")) or "runs queued" in ln or "printed" in ln:
            say("    " + ln.strip()[:240])
    try:
        st = json.loads(show(HT, "explore_out/H/s89_results.json"))
        say(f"  explore_out/H/s89_results.json at {HT}: stored runs {sorted(st['runs'])}")
    except subprocess.CalledProcessError:
        pass
s89files = []
for b in heads:
    for p in git("ls-tree", "-r", "--name-only", b).decode().split():
        if "s89" in p.lower():
            s89files.append((b.split("/")[-1], p))
say(f"  files named *s89* on the heads: {s89files}")
hstore = [p for br, p in s89files if br == "explore-H" and p.endswith(".json")]
say(f"  S89 store (explore_out/H/s89_results.json) on explore-H: {'present' if hstore else 'absent'}; "
    f"checks.json has H/s89: {'H/s89' in json.loads(show(LOG, 'explore_out/H/checks.json'))}")

# ══ 1. seeds ═════════════════════════════════════════════════════════════════
say("\n== 1. Seeds 2900-2959")
KEYRE = re.compile(r"^[A-Za-z0-9_.\-]+\|(-?\d+)(\|.*)?$")


def walk(o, hit, ikeys, parent="", cnt=None):
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRE.match(k) if isinstance(k, str) else None
            if m:
                cnt[0] += 1
                if int(m.group(1)) in BLOCK:
                    hit.add(("key", k.split("|")[0], int(m.group(1))))
            elif isinstance(k, str) and k.isdigit() and int(k) in BLOCK:
                ikeys.add((int(k), str(parent)))
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool):
                cnt[0] += 1
                if v in BLOCK:
                    hit.add(("seed field", parent, v))
            if k == "seeds" and isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and not isinstance(x, bool):
                        cnt[0] += 1
                        if x in BLOCK:
                            hit.add(("seeds list", parent, x))
            walk(v, hit, ikeys, k, cnt)
    elif isinstance(o, list):
        for x in o:
            walk(x, hit, ikeys, parent, cnt)


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
    hits, ikeys, cnt, unparsed = {}, {}, [0], []
    shas = list(blobs)
    p = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha in shas:
        p.stdin.write((sha + "\n").encode())
        p.stdin.flush()
        hdr = p.stdout.readline().split()
        size = int(hdr[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        objs = parse(data.decode("utf-8", "replace"))
        if not objs:
            unparsed.append(blobs[sha][0])
        h, ik = set(), set()
        for o in objs:
            walk(o, h, ik, "", cnt)
        if h:
            hits[sha] = h
        if ik:
            ikeys[sha] = ik
    p.stdin.close()
    p.wait()
    return hits, ikeys, cnt[0], unparsed


tip_blobs = {}
for b in heads:
    for line in git("ls-tree", "-r", b).decode().splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and path.endswith((".json", ".jsonl")):
            tip_blobs.setdefault(sha, []).append((b.split("/")[-1], path))
hits, ikeys, n, unp = scan(tip_blobs)
say(f"  heads: {len(tip_blobs)} distinct JSON/JSONL blobs ({sum(len(v) for v in tip_blobs.values())} branch-paths); "
    f"{n} seed values read; unparsed: {unp or 'none'}")
say(f"  heads: records using a seed in 2900-2959: {sum(len(v) for v in hits.values())}"
    + "".join(f"\n    {tip_blobs[s][:2]}: {sorted(v)[:6]}" for s, v in hits.items()))
say(f"  heads: integer dict keys in 2900-2959 (parents; update steps, not seeds): "
    f"{sorted({(k, par) for v in ikeys.values() for k, par in v})[:12] or 'none'}"
    + (f" in {sorted({tip_blobs[s][0] for s in ikeys})[:6]}" if ikeys else ""))
hist_blobs = {}
for line in git("rev-list", "--all", "--objects").decode().splitlines():
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl")):
        hist_blobs.setdefault(parts[0], []).append(("history", parts[1]))
hh, hik, hn, hunp = scan(hist_blobs)
say(f"  history (every ref, every commit): {len(hist_blobs)} distinct JSON/JSONL blobs; {hn} seed values read; "
    f"unparsed: {len(hunp)}")
say(f"  history: blobs with a record using a seed in 2900-2959: {len(hh)}"
    + "".join(f"\n    {hist_blobs[s][0][1]}: {sorted(v)[:6]}" for s, v in list(hh.items())[:10]))
pat = r"(^|[^0-9.e])(29[0-5][0-9])([^0-9]|$)"
txt = collections.defaultdict(set)
for b in heads:
    try:
        r = git("grep", "-n", "-E", pat, b, "--", "*.py", "*.md", "*.tex", "*.txt", "*.log").decode()
    except subprocess.CalledProcessError:
        r = ""
    for ln in r.splitlines():
        _, path, lno = ln.split(":", 3)[:3]
        txt[(b.split("/")[-1], path)].add(int(lno))
say("  text mentions of a number 2900-2959 on the heads (*.py *.md *.tex *.txt *.log; branch, path, lines):")
for (b, p), ls in sorted(txt.items()):
    say(f"    {b} {p} {sorted(ls)[:8]}{' ...' if len(ls) > 8 else ''}")

# ══ 2. the readings ══════════════════════════════════════════════════════════
say("\n== 2. The readings: explore_h_s89.report() at d4a9a7d on every distinct outcome statistic (10 seeds, 4 arms)")
SYN = {}


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


_stub("explore_common", load_store=lambda name: {"meta": {}, "runs": {}}, cpu_model=lambda: "stub", OUT_DIR="/nonexistent")
_stub("explore_common16", runs_of=lambda screen, arm, seeds=None, store=None: dict(SYN[arm]), med=_med, STATE={},
      code_sha=lambda *a: ("stub", []), key=lambda *a: "stub", run_child=lambda *a, **k: None)
_stub("explore_muon_scale")
_stub("explore_h_collapse")
_stub("explore_h_allmass2", NAME="H/allmass2")
hc = types.ModuleType("explore_h_common")
hc.__file__ = "explore_h_common.py"
sys.modules["explore_h_common"] = hc
exec(compile(show(H, "explore_h_common.py"), "explore_h_common.py@" + H, "exec"), hc.__dict__)
s89 = types.ModuleType("explore_h_s89")
s89.__file__ = os.path.join(HERE, "_nonexistent_", "explore_h_s89.py")
sys.modules["explore_h_s89"] = s89
SRC89 = show(H, "explore_h_s89.py")
exec(compile(SRC89, "explore_h_s89.py@" + H, "exec"), s89.__dict__)
LEARN = ("BASE", "ALLMASS", "REPEATMASS", "NOSPLIT")       # bit 0..3 of a per-seed pattern
SEEDS = s89.SEEDS
assert tuple(s89.LEARNED) == ("BASE", "ALLMASS", "REPEATMASS", "NOSPLIT") and len(SEEDS) == 10


def compositions(n, k):
    """All compositions of n into k parts (numpy, rows)."""
    bars = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n + k - 1), k - 1)),
                       dtype=np.int16).reshape(-1, k - 1)
    ext = np.concatenate([np.full((bars.shape[0], 1), -1, np.int16), bars,
                          np.full((bars.shape[0], 1), n + k - 1, np.int16)], 1)
    return (np.diff(ext, axis=1) - 1).astype(np.int16)


comp = compositions(10, 16)
pats = np.arange(16)
bit = {a: ((pats >> i) & 1).astype(bool) for i, a in enumerate(LEARN)}


def only(x, y):
    return comp[:, bit[x] & ~bit[y]].sum(1)


stat = np.stack([comp[:, bit["BASE"]].sum(1),
                 only("NOSPLIT", "BASE"), only("BASE", "NOSPLIT"),
                 only("ALLMASS", "BASE"), only("BASE", "ALLMASS"), only("ALLMASS", "NOSPLIT"), only("NOSPLIT", "ALLMASS"),
                 only("REPEATMASS", "BASE"), only("BASE", "REPEATMASS"), only("REPEATMASS", "NOSPLIT"),
                 only("NOSPLIT", "REPEATMASS")], 1)
uniq, first = np.unique(stat, axis=0, return_index=True)
say(f"  {comp.shape[0]} compositions of 10 seeds into 16 patterns; {len(uniq)} distinct sufficient statistics "
    f"(BASE count; b, c of NOSPLIT/BASE, ALLMASS/BASE, ALLMASS/NOSPLIT, REPEATMASS/BASE, REPEATMASS/NOSPLIT)")
# every predicate the prompt's rules (and report()) test, per statistic: one representative per predicate pattern, plus
# a random sample of 20000 distinct statistics
nB_, NBb, NBc, ABb, ABc, ANb, ANc, RBb, RBc, RNb, RNc = stat.T
coarse = np.stack([nB_ < 7, NBc - NBb <= 1,
                   ABc - ABb <= 1, ANb >= 4, ANc == 0, ABc >= 4, ABb == 0,
                   RBc - RBb <= 1, RNb >= 4, RNc == 0, RBc >= 4, RBb == 0], 1)
cu, cfirst = np.unique(coarse, axis=0, return_index=True)
rs = np.random.default_rng(7)
pick = np.unique(np.concatenate([cfirst, rs.choice(first, size=min(20000, len(first)), replace=False)]))
say(f"  {len(cu)} distinct patterns of the predicates the rules test (BASE < 7; NOSPLIT within 1; per trigger arm: within "
    f"1 of BASE, beats NOSPLIT >= 4, NOSPLIT only 0, BASE only >= 4, T only 0); report() run on one composition per "
    f"pattern and on 20000 random distinct statistics: {len(pick)} calls")
first = pick


def rec(ok):
    return dict(ok=True, outcome="BOUND ROUTED" if ok else "non-stream OTHER", outcome_v2="BOUND ROUTED" if ok else
                "non-stream OTHER", transition=12000 if ok else None, checks=[])


def prompt_read(nB, NBb, NBc, ABb, ABc, ANb, ANc, RBb, RBc, RNb, RNc):
    """The round-6 prompt's rules (parallel_sessions_prompts.md at 7c4ceb8, Session H round 6), 'within 1 pair of BASE'
    as S87's 'X >= BASE - 1 pair' (= BASE only - X only <= 1 on paired seeds)."""
    if nB < 7:
        return ("FAILS",)
    moot = NBc - NBb <= 1
    r = ["NOT NEEDED" if moot else "NEEDED"]
    for b, c, bn, cn in ((ABb, ABc, ANb, ANc), (RBb, RBc, RNb, RNc)):
        if c - b <= 1 and bn >= 4 and cn == 0:
            r.append("SURVIVES")
        elif c >= 4 and b == 0:
            r.append("NEEDS KEYPOS")
        else:
            r.append("NONE")
    return tuple(r)


def h_names(rd):
    if "S89" in rd:
        return ("FAILS",) if "labelled recipe itself fails" in rd["S89"] else (rd["S89"][:20],)
    r = ["NOT NEEDED" if rd["split"].startswith("the split is not needed") else "NEEDED"]
    for t in ("ALLMASS", "REPEATMASS"):
        s = rd[t]
        n_ = [nm for key, nm in (("the trigger survives", "SURVIVES"), ("the trigger needs key positions", "NEEDS KEYPOS"),
                                 ("no reading applies", "NONE")) if key in s]
        assert len(n_) == 1, s
        r.append(n_[0])
    return tuple(r)


table = collections.Counter()
mism, moot_named = [], collections.Counter()
SYN["ORACLE"] = {2900: rec(True), 2901: rec(True)}
for i in first:
    row = comp[i]
    seq = [p for p in range(16) for _ in range(int(row[p]))]
    for a in LEARN:
        SYN[a] = {s: rec(bool(bit[a][p])) for s, p in zip(SEEDS, seq)}
    res, _ = s89.report()
    assert res["valid"] and res["complete"]
    hn = h_names(res["readings"])
    pn = prompt_read(*[int(v) for v in stat[i]])
    if hn != pn:
        mism.append((stat[i].tolist(), hn, pn))
    table[hn] += 1
    if hn[0] == "NOT NEEDED":
        for t, nm in zip(("ALLMASS", "REPEATMASS"), hn[1:]):
            if nm != "NONE":
                moot_named[(t, nm)] += 1
say(f"  report() vs the prompt's rules re-implemented: {len(mism)} disagreements over {len(uniq)} statistics"
    + (f" {mism[:5]}" if mism else ""))
say(f"  every statistic got exactly one name per reading key (S89 | split, ALLMASS, REPEATMASS); combinations "
    f"(name tuple: number of distinct statistics):")
for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
    say(f"    {k}: {v}")
say(f"  'moot' statistics where a trigger reading still carries a name (printed 'moot (...): T: the trigger ...'): "
    f"{dict(moot_named)}")
cand = []
for i in first:
    nB, NBb, NBc, ABb, ABc, ANb, ANc = [int(v) for v in stat[i][:7]]
    if nB >= 7 and NBc - NBb <= 1 and ABc - ABb <= 1 and ANb >= 4 and ANc == 0:
        nN = nB - NBc + NBb
        nA = nB - ABc + ABb
        cand.append((nB, nN, nA))
say(f"  (BASE, NOSPLIT, ALLMASS) counts at which 'the split is not needed here' and 'ALLMASS: the trigger survives' "
    f"are printed together: {sorted(set(cand))}")
# the incomplete / invalid paths
SYN["ORACLE"] = {2900: rec(True), 2901: rec(False)}
SYN["ORACLE"][2901]["transition"] = None
res, _ = s89.report()
say(f"  ORACLE 1/2: {res['readings']}")
SYN["ORACLE"] = {2900: rec(True), 2901: rec(True)}
SYN["BASE"] = {s: rec(True) for s in SEEDS[:9]}
res, _ = s89.report()
say(f"  ORACLE 2/2, BASE 9 of 10 runs stored: {res['readings']}")

# ══ 3. the outcome ═══════════════════════════════════════════════════════════
say("\n== 3. The outcome")
v2h = show(H, "explore_h_fail_class_v2.py")
v2m = show(MAIN, "fail_class_v2.py")
say(f"  explore_h_fail_class_v2.py at {H} vs fail_class_v2.py at {MAIN}: byte-identical {v2h == v2m}")


def fdefs(src):
    t = ast.parse(src)
    return {n.name: ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef)}


for mod, fs in (("test_stream_recipe", ("outcome", "routed", "padded_task", "model_fn")),
                ("test_scale_axes", ("routing_k", "fail_class_k", "scale_stats", "stream_acc")),
                ("test_stream_channels", ("merged", "shared_max"))):
    a, b = fdefs(show(MAIN9C, mod + ".py")), fdefs(show(MAIN, mod + ".py"))
    say(f"  {mod} at {MAIN9C} (served to the child) vs {MAIN}: " + ", ".join(f"{f} {a.get(f) == b.get(f)}" for f in fs))
a = fdefs(show(H, "test_router_layout.py"))
b = fdefs(show(MAIN, "test_router_layout.py"))
say(f"  test_router_layout.fail_class at {H} vs {MAIN}: {a.get('fail_class') == b.get('fail_class')}")
say("  where the outcome is set: explore_b16_main.finish -> rec['outcome'] = tsr.outcome(rec) (imported); "
    "hc.br(r) = r['outcome'] == 'BOUND ROUTED'; outcome_v2 = explore_h_fail_class_v2.outcome_v2(rec, tsr.routed)")

# ══ 4. the CHECKs' records ═══════════════════════════════════════════════════
say("\n== 4. The records the CHECKs compare against")
st87 = json.loads(show(H, "explore_out/H/allmass2_results.json"))["runs"]
for k, r in st87.items():
    if k.split("|")[1] == "2600":
        f = [c["step"] for c in r.get("checks") or [] if c.get("fired")]
        say(f"  S87a {k.split('|')[0]}|2600 (code {k.split('|')[2]}): cpu {r['cpu']}; outcome {r['outcome']}; "
            f"transition {r['transition']}; fired at {f}")
ws = json.loads(show(H, "explore_out/window_scale_results.json"))["runs"]
for k, r in ws.items():
    if k.startswith("LOCAL3_SLOW_D8_A|260|"):
        say(f"  batch 16 {k.split('|')[0]}|260 (code {k.split('|')[2]}): cpu {r['cpu']}; outcome {r['outcome']}")
x = json.loads(show(MAIN, "results/X/window_gate_results.json"))["runs"]["WIN3_SPLIT_D8|540"]
say(f"  X WIN3_SPLIT_D8|540 (main at {MAIN}): cpu {x.get('cpu')}; transition {x.get('transition')}; fired at "
    f"{[c['step'] for c in x.get('checks') or [] if c.get('fired')]}")
lg = git("log", "--format=%h %cI", "origin/claude/bdh-growth-hebbian-inference-w90069", "--",
         "results/X/window_gate_results.json").decode().split("\n")
say(f"  commits touching results/X/window_gate_results.json on the main line: {[l_ for l_ in lg if l_][:5]}")
m = re.search(r'MAINREF = "([^"]+)"', SRC89)
say(f"  explore_h_s89.py MAINREF = {m.group(1)!r} (a branch name, not a commit)")
for log_name in ("s79.log", "s86.log", "s87.log"):
    t = show(H, "explore_out/H/" + log_name)
    say(f"  {log_name}: CHECK lines reproducing the 2.80GHz record LOCAL3_SLOW_D8_A|260 on this CPU: "
        f"{sum('LOCAL3_SLOW_D8_A|260' in ln and ln.rstrip().endswith('ok') for ln in t.splitlines())}")

# ══ 5. operating characteristics ═════════════════════════════════════════════
say("\n== 5. Operating characteristics (exact multinomial over the 8 per-seed patterns of BASE, T, NOSPLIT)")


def pattern_probs(pB, pT, pN, rho):
    """Per-seed pattern probabilities (bits B, T, N): with probability rho one shared uniform U decides all three arms
    (arm binds iff U < p), else three independent uniforms."""
    ps = (pB, pT, pN)
    q = np.zeros(8)
    for p in range(8):
        ind = 1.0
        for i in range(3):
            ind *= ps[i] if (p >> i) & 1 else 1 - ps[i]
        lo, hi = 0.0, 1.0
        for i in range(3):
            if (p >> i) & 1:
                hi = min(hi, ps[i])
            else:
                lo = max(lo, ps[i])
        com = max(0.0, hi - lo)
        q[p] = rho * com + (1 - rho) * ind
    return q


OC = {}
for N in (10, 20):
    c3 = compositions(N, 8).astype(np.int64)
    pb = np.arange(8)
    bB, bT, bN = [((pb >> i) & 1).astype(bool) for i in range(3)]

    def on(x, y):
        return c3[:, x & ~y].sum(1)
    nB = c3[:, bB].sum(1)
    TB, BT, TN, NT, NB, BN = on(bT, bB), on(bB, bT), on(bT, bN), on(bN, bT), on(bN, bB), on(bB, bN)
    gate = nB < math.ceil(0.7 * N)
    moot = (BN - NB <= 1) & ~gate
    surv = (BT - TB <= 1) & (TN >= 4) & (NT == 0) & ~gate
    need = (BT >= 4) & (TB == 0) & ~gate
    lf = np.array([math.lgamma(i + 1) for i in range(N + 1)])
    logmult = lf[N] - lf[c3].sum(1)
    for pT in (0.9, 0.8, 0.7, 0.5):
        for pN in (0.3, 0.6):
            for rho in (0.0, 0.5, 1.0):
                q = pattern_probs(0.9, pT, pN, rho)
                with np.errstate(divide="ignore"):
                    lq = np.log(q)
                lp = logmult + (c3 * np.where(q > 0, lq, 0)).sum(1)
                ok = ~((c3 > 0) & (q == 0)).any(1)
                pr = np.where(ok, np.exp(lp), 0.0)
                OC[(N, pT, pN, rho)] = dict(total=pr.sum(), fails=pr[gate].sum(), moot=pr[moot].sum(),
                                            surv=pr[surv].sum(), surv_live=pr[surv & ~moot].sum(),
                                            need=pr[need].sum(), need_live=pr[need & ~moot].sum(),
                                            none_live=pr[~gate & ~moot & ~surv & ~need].sum())
say("  BASE 0.9 throughout; 'fails' = BASE < 7/10 (14/20 at 20 seeds, the same 0.7); 'moot' = NOSPLIT within 1 pair;"
    " survives / needs counted with and without the moot runs; rho = share of seeds on which one difficulty decides all "
    "three arms")
say(f"  {'N':>3} {'pT':>4} {'pN':>4} {'rho':>4} | {'fails':>6} {'moot':>6} | {'surv':>6} {'surv*':>6} | {'needs':>6} "
    f"{'needs*':>6} | {'none*':>6}    (* = not moot, not failed: the reading printed without 'moot')")
for (N, pT, pN, rho), d in sorted(OC.items()):
    assert abs(d["total"] - 1) < 1e-9
    say(f"  {N:>3} {pT:>4} {pN:>4} {rho:>4} | {d['fails']:6.3f} {d['moot']:6.3f} | {d['surv']:6.3f} {d['surv_live']:6.3f} "
        f"| {d['need']:6.3f} {d['need_live']:6.3f} | {d['none_live']:6.3f}")
# the exact one-sided McNemar at 4 vs 0
say(f"  exact one-sided McNemar at 4 vs 0: p = {0.5 ** 4:.4f}; at 5 vs 0: {0.5 ** 5:.4f}")

# ══ 6. the task and the trigger ══════════════════════════════════════════════
say("\n== 6. The pool task and the trigger")
S, P, K, NV = 8, 4, 32, 16
rng = np.random.default_rng(89)
n = 4096
vals = np.argsort(rng.random((n, K, NV)), -1)[..., :S]
kord = np.argsort(rng.random((n, K)), -1)[:, :P]
sord = np.argsort(rng.random((n, P, S)), -1)
vg = np.take_along_axis(np.take_along_axis(vals, kord[..., None].repeat(S, -1), 1), sord, 2)
keys = np.broadcast_to((S + kord)[..., None], (n, P, S))
body = np.stack([sord, keys, S + K + vg], -1).reshape(n, -1)
q = np.argsort(rng.random((n, S * P)), -1)[:, :1]
qs, qi = q // P, q % P
qk = np.take_along_axis(np.sort(kord, -1), qi, 1)
qv = np.take_along_axis(vals.reshape(n, K * S), qk * S + qs, 1)
x = np.concatenate([body, np.stack([qs, S + qk, S + K + qv], -1).reshape(n, -1)], 1)
inp = x[:, :-1]
V = S + K + NV
cnt = np.stack([np.bincount(r, minlength=V) for r in inp])
rep = np.take_along_axis(cnt, inp, 1) > 1
j = np.arange(S * P)
say(f"  numpy re-implementation of PoolTask(32).make_batch's distribution, {n} sequences (not torch's bit stream):")
say(f"    input positions {inp.size}; repeat set {int(rep.sum())} ({rep.mean():.3f}); CTX {int(rep[:, 3 * j].sum())} of "
    f"{rep[:, 3 * j].size}; KEY {int(rep[:, 3 * j + 1].sum())} of {rep[:, 3 * j + 1].size}; VAL "
    f"{int(rep[:, 3 * j + 2].sum())} of {rep[:, 3 * j + 2].size} ({rep[:, 3 * j + 2].mean():.4f}); query CTX "
    f"{int(rep[:, 96].sum())}, query KEY {int(rep[:, 97].sum())} of {n}")
say(f"    key positions / repeat set = {rep[:, 3 * j + 1].sum() / rep.sum():.3f}")
kc = cnt[:, S:S + K]
say(f"    within a sequence: distinct key types {sorted(set((kc > 0).sum(1).tolist()))}; count of each present key type "
    f"{sorted(set(kc[kc > 0].tolist()))} (8 in the body, 9 for the queried one); CTX type counts "
    f"{sorted(set(cnt[:, :S].ravel().tolist()))}; VAL type counts (present) "
    f"{sorted(set(cnt[:, S + K:][cnt[:, S + K:] > 0].tolist()))}")
say("  exact: P(a VAL position's type repeats) = 1 - (1/2)^3 = 0.875 (each other key holds a given value with prob. 8/16); "
    f"expected repeat set per sequence = 32 + 32 + 28 + 2 = 94 of 98 = {94 / 98:.4f}; H's docstring: 385,149 of "
    f"401,408 = {385149 / 401408:.4f}; VAL 114,813 of 131,072 = {114813 / 131072:.4f}")
say(f"  exact corpus frequencies per type: CTX 33/98/8 = {33 / 98 / 8:.4f}; KEY 33/98/32 = {33 / 98 / 32:.4f}; VAL "
    f"32/98/16 = {32 / 98 / 16:.4f}; main line KEY (4 types) 33/98/4 = {33 / 98 / 4:.4f}")
say(f"  across the 64-sequence trigger probe: a key type appears in Bin(64, 1/8) sequences (mean 8); P(absent) = "
    f"{(7 / 8) ** 64:.5f}")

src_child = show(H, "explore_h_s89_child.py")
src86 = show(H, "explore_h_allmass_child.py")


def fsrc(src, name):
    t = ast.parse(src)
    for nd in t.body:
        if isinstance(nd, ast.FunctionDef) and nd.name == name:
            return ast.unparse(nd).splitlines()
    raise KeyError(name)


a = fsrc(src86, "make_trigger_target")
b = fsrc(src_child, "make_trigger_s89")
d = [ln for ln in difflib.unified_diff(a, b, "S86 make_trigger_target", "S89 make_trigger_s89", lineterm="", n=0)]
say("  syntax-tree (unparsed) diff, S86 explore_h_allmass_child.make_trigger_target -> S89 make_trigger_s89:")
for ln in d:
    say("    " + ln)
for fn in ("repeat_mask", "select_repeat"):
    say(f"  {fn}: " + " | ".join(fsrc(src_child, fn)))
say("  explore_split_target_child.select: " + " | ".join(fsrc(show(H, "explore_split_target_child.py"), "select")))
say("  explore_split_child.probe_data: " + " | ".join(fsrc(show(H, "explore_split_child.py"), "probe_data")))
say(f"  PoolTask.make_batch (explore_h_s89_child.py:80-93) and BindTask.make_batch (test_binding_capacity.py at "
    f"{MAIN9C}) draw order: vals (B,K|P,nv), korder (B,K|P), sorder (B,P,S), q (B,S*P)")

# ══ 7. other ═════════════════════════════════════════════════════════════════
say("\n== 7. Other")
ft = show(F6, "explore_f_tasks.py")
cls = [nd.name for nd in ast.parse(ft).body if isinstance(nd, ast.ClassDef)]
say(f"  F's explore_f_tasks.py at {F6} ({git('log', '-1', '--format=%cI', F6).decode().strip()}): classes {cls}; "
    f"H's commit {H} at {git('log', '-1', '--format=%cI', H).decode().strip()}")
say(f"  'grouped' in a pool class of F's module: "
    f"{[c for c in cls if 'Pool' in c]} subclass RandHeaderTask (S70's lite layout, S = 2); "
    f"H's tree has explore_f_tasks.py: {'explore_f_tasks.py' in git('ls-tree', '--name-only', H).decode().split()}")
FCITE = re.findall(r"F.s\s+module\s+has\s+no\s+grouped-layout\s+pool\s+variant\s+at\s+(\w+)", SRC89)
say(f"  explore_h_s89.py cites F at: {FCITE}")
st86 = json.loads(show(H, "explore_out/H/allmass_results.json"))["runs"]


def br(r):
    return r.get("outcome") == "BOUND ROUTED"


pairs = collections.Counter()
o2o = collections.Counter()
secs = collections.defaultdict(list)
for st in (st86, st87):
    by = collections.defaultdict(dict)
    for k, r in st.items():
        a_, s_ = k.split("|")[:2]
        if r.get("ok"):
            by[a_][int(s_)] = r
            if r.get("secs"):
                secs[a_].append(r["secs"] / 60)
    for s_ in by["BASE"]:
        if s_ in by["ALLMASS"]:
            pairs[(br(by["BASE"][s_]), br(by["ALLMASS"][s_]))] += 1
    for a_ in ("BASE", "ALLMASS"):
        for r in by[a_].values():
            for c in r.get("checks") or []:
                if c.get("fired"):
                    o2o[(a_, "fired")] += 1
                    o2o[(a_, "one-to-one before")] += len(set(c["map_before"])) == 8
say(f"  S86 + S87a BASE vs ALLMASS on the same seeds (BASE, ALLMASS bound routed): {dict(pairs)}")
say(f"  S86 + S87a firings and firings on a one-to-one map: {dict(o2o)}")
for a_, v in secs.items():
    v = sorted(v)
    say(f"  S86 + S87a run time, {a_}: median {v[len(v) // 2]:.0f} min (range {v[0]:.0f}-{v[-1]:.0f}), {len(v)} runs")
lsd = ws
cnt_ = collections.Counter()
for k, r in lsd.items():
    a_ = k.split("|")[0]
    if a_.startswith("LOCAL3_SLOW_D8") and r.get("ok"):
        cnt_[(a_, br(r))] += 1
say(f"  batch 16 LOCAL3_SLOW_D8* (the recipe without a trigger): {dict(cnt_)}")
xw = json.loads(show(MAIN, "results/X/window_gate_results.json"))["runs"]
for a_ in ("WIN3_SLOW_D8", "WIN3_SPLIT_D8"):
    rs = [r for k, r in xw.items() if k.split("|")[0] == a_ and r.get("ok")]
    nb = sum(1 for r in rs if r.get("transition") is not None and r["end"]["one_to_one"]
             and min(r["end"]["stream_acc"]) >= 0.9)
    say(f"  X test_window_gate {a_}: BOUND ROUTED {nb}/{len(rs)} (tsr.outcome's rule applied to the record)")

with open(os.path.join(HERE, "screen_checks_s89_H_out.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
