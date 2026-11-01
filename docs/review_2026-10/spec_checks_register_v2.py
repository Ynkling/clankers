#!/usr/bin/env python3
"""spec_checks_register_v2.py: the referee's checks behind spec_review_register_v2.md (Session G's
explore_out/G/spec_draft_register.md, revision 1, at claude/explore-G c98648e6, against the first review
spec_review_register_draft.md on the draft at 076005d).

    python3 docs/review_2026-10/spec_checks_register_v2.py      (stdlib; reads git only; writes ..._out.txt)

1. Seeds: every JSON / JSONL blob on every remote branch head (today's heads), every "ARM|seed" key, "seed" field and
   "seeds" list; hits on X 600-619 / L 1600-1619; text mentions of the blocks.
2. The named outcome function: does test_register_gate.py (or test_register.py, the plan's name) exist on any head;
   explore_g_rhl_parent.py's blob and line numbers at 076005d and c98648e6; files changed by c98648e6.
3. routed_bound (as at explore_g_rhl_parent.py:35-53, 076005d) re-scored on S82, S83 (the draft's CHECK list), S87 and
   S89 (background); the references the reproduction CHECK names.
4. The rate claim R: intervals, P(pass) per machine, on both, on exactly one, pooled, at 0.80/0.85/0.90/0.95.
5. McNemar power for C1-C4 (and for the arms the plan asked for) at the records' paired effect sizes; Holm over four.
6. Readings: the 16 combinations of C1-C4 and the 9 rate combinations, as revision 1 words them.
7. The task structure: what the key, frequency and repeat hinges select on lite (exact, plus a stdlib simulation of
   RandHeaderTask's value draw).
8. Dates: the plan, S89's records, report 8, revision 1.
"""
import collections
import json
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rlib import load, mcnemar1, wilson  # noqa: E402

out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a)


def exists(ref, path):
    return subprocess.call(("git", "-C", ROOT, "cat-file", "-e", f"{ref}:{path}"), stderr=subprocess.DEVNULL) == 0


MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
G6, G8, GR1 = "076005d8", "8e9e3187", "c98648e6"


def pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def p_ge(n, k, p):
    return sum(pmf(n, i, p) for i in range(k, n + 1))


def clopper_pearson(x, n, a=0.05):
    def bis(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lo = 0.0 if x == 0 else bis(lambda p: p_ge(n, x, p) >= a / 2)
    hi = 1.0 if x == n else bis(lambda p: 1 - p_ge(n, x + 1, p) <= a / 2)
    return lo, hi


def multinom(n, qa, qb):
    qn = 1 - qa - qb
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield a, b, math.comb(n, a) * math.comb(n - a, b) * qa ** a * qb ** b * max(qn, 0.0) ** (n - a - b)


def power_q(n, qb, qc, alpha=0.05):
    return sum(pr for b, c, pr in multinom(n, qb, qc) if mcnemar1(b, c) < alpha)


def within1_q(n, qb, qc):
    """P(b - c <= 1): the screens' 'within 1 pair of the labelled hinge' rule, applied to n pairs."""
    return sum(pr for b, c, pr in multinom(n, qb, qc) if b - c <= 1)


# ══ 1. seeds ═════════════════════════════════════════════════════════════════
say("== 1. Seeds: every JSON / JSONL blob on every remote branch head (today)")
heads = [b for b in git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").decode().split()
         if not b.endswith("/HEAD")]
for b in heads:
    say(f"  scanned {b} at {git('rev-parse', '--short', b).decode().strip()} "
        f"({git('log', '-1', '--format=%cI', b).decode().strip()})")
BLOCK = set(range(600, 620)) | set(range(1600, 1620))
KEYRE = re.compile(r"^(?:[A-Za-z0-9_]+\.)?[A-Za-z0-9_\-]+\|(-?\d+)(\|.*)?$")
blobs = {}
for b in heads:
    for line in git("ls-tree", "-r", b).decode().splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and path.endswith((".json", ".jsonl")):
            blobs.setdefault(sha, []).append((b, path))
seed_uses = collections.defaultdict(set)
int_keys = collections.defaultdict(set)
n_vals = 0


def walk(o, locs, parent=""):
    global n_vals
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRE.match(k) if isinstance(k, str) else None
            if m:
                n_vals += 1
                for b, p in locs:
                    seed_uses[int(m.group(1))].add((b, p))
            elif isinstance(k, str) and k.isdigit():
                int_keys[int(k)].add(str(parent))
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool):
                n_vals += 1
                for b, p in locs:
                    seed_uses[v].add((b, p))
            if k == "seeds" and isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and not isinstance(x, bool):
                        n_vals += 1
                        for b, p in locs:
                            seed_uses[x].add((b, p))
            walk(v, locs, k)
    elif isinstance(o, list):
        for x in o:
            walk(x, locs, parent)


unparsed = []
for sha, locs in blobs.items():
    text = git("cat-file", "blob", sha).decode("utf-8", "replace")
    try:
        objs = [json.loads(text)]
    except ValueError:
        objs = []
        for ln in text.splitlines():
            try:
                objs.append(json.loads(ln))
            except ValueError:
                pass
        if not objs:
            unparsed.append(locs[0])
    for o in objs:
        walk(o, locs)
say(f"  {len(blobs)} distinct JSON/JSONL blobs ({sum(len(v) for v in blobs.values())} branch-paths); {n_vals} seed "
    f"values read; unparsed: {unparsed or 'none'}")
hits = sorted((s, u) for s in BLOCK for u in seed_uses.get(s, ()))
say(f"  records using a seed of X 600-619 / L 1600-1619: {len(hits)}" + (f" {hits[:10]}" if hits else ""))
say(f"  integer dict keys in the block (parents): {({v: sorted(int_keys[v]) for v in sorted(BLOCK) if v in int_keys}) or 'none'}")
for lo, hi in ((2700, 2730), (2840, 2900)):
    used = sorted(s for s in seed_uses if lo <= s < hi)
    say(f"  G's later screens' seeds in use in {lo}-{hi - 1}: {used[0] if used else '-'}..{used[-1] if used else '-'} "
        f"({len(used)} values); overlap with the block: {sorted(set(used) & BLOCK) or 'none'}")
pat = r"(^|[^0-9.e])(6[01][0-9]|16[01][0-9]) ?(-|–|\.\.) ?(6[01][0-9]|16[01][0-9])([^0-9]|$)|range\((6[01][0-9]|16[01][0-9]),"
txt = set()
for b in heads:
    try:
        r = git("grep", "-n", "-E", pat, b, "--", "*.py", "*.md", "*.tex", "*.txt").decode()
    except subprocess.CalledProcessError:
        r = ""
    for ln in r.splitlines():
        _, path, _lno = ln.split(":", 3)[:3]
        txt.add((path, b.split("/")[-1]))
say("  text mentions of 600-619 / 1600-1619 ranges (*.py, *.md, *.tex, *.txt; path, branch):")
for path, b in sorted(txt):
    say(f"    {path} ({b})")

# ══ 2. the named function ════════════════════════════════════════════════════
say("\n== 2. The named outcome function")
found = {}
for b in heads:
    names = [p for p in git("ls-tree", "-r", "--name-only", b).decode().splitlines()
             if re.search(r"(^|/)(test_register(_gate)?\.py|register_gate\.py|specs/test_register[^/]*\.md)$", p)]
    if names:
        found[b] = names
say(f"  test_register_gate.py / register_gate.py / test_register.py / specs/test_register*.md on any of the "
    f"{len(heads)} heads: {found or 'none'}")
try:
    gr = git("grep", "-l", "-E", r"def routed_bound", *heads, "--", "*.py").decode().split()
except subprocess.CalledProcessError:
    gr = []
say(f"  'def routed_bound' on any head: {sorted(set(gr))}")
for ref in (G6, GR1, "origin/claude/explore-G"):
    src = git("show", f"{ref}:explore_g_rhl_parent.py").decode().splitlines()
    lines = {n: i + 1 for i, l in enumerate(src) for n in ("ROUTE_MARGIN, ROUTE_ETA =", "def bound(", "def routed_star(",
                                                             "def routed_bound(") if l.startswith(n)}
    say(f"  {ref}: explore_g_rhl_parent.py blob {git('rev-parse', f'{ref}:explore_g_rhl_parent.py').decode()[:10]}; "
        f"lines {lines}")
src = git("show", f"{G6}:explore_g_rhl_parent.py").decode().splitlines()
say("  routed_bound at 076005d, :39-53 verbatim:")
for l in src[38:53]:
    say("    | " + l)
say(f"  files changed by {GR1}: {git('show', '--name-only', '--format=', GR1).decode().split()}")
say(f"  files changed by {G8} (report 8): {git('show', '--name-only', '--format=', G8).decode().split()}")
for f in ("explore_f_tasks.py",):
    say(f"  {f}: blob at 9a6c2bf {git('rev-parse', f'9a6c2bf:{f}').decode()[:10]}, at 076005d "
        f"{git('rev-parse', f'{G6}:{f}').decode()[:10]}, at {GR1} {git('rev-parse', f'{GR1}:{f}').decode()[:10]}")
old = git("show", f"9a6c2bf:explore_f_tasks.py")
new = git("show", f"{GR1}:explore_f_tasks.py")
dup = [n for n, c in collections.Counter(re.findall(rb"^(?:def|class) (\w+)", new, re.M)).items() if c > 1]
say(f"  9a6c2bf's module is a byte prefix of {GR1}'s: {new.startswith(old)} ({len(old)} of {len(new)} bytes); "
    f"top-level names defined twice in the new module: {dup or 'none'}")
plan = git("show", f"{MAIN}:docs/reading/plan_2026-10-13.md").decode().splitlines()
say("  plan_2026-10-13.md (main line), the register test's names: " + "; ".join(
    f":{i + 1} {l.strip()[:150]}" for i, l in enumerate(plan) if "test_register" in l))

# ══ 3. routed_bound on the records ═══════════════════════════════════════════
say("\n== 3. routed_bound (explore_g_rhl_parent.py:35-53 at 076005d) on G's records")


def bound(r):
    return r["transition"] is not None


def routed_star(r):
    e = r["end"]
    return bool(e.get("margin", 0.0) >= 0.9 and e.get("eta_key_by_stream", 0.0) > 0.9)


def routed_bound(r):
    return bool(bound(r) and routed_star(r))


def arms_of(store, prefix=""):
    A = collections.defaultdict(dict)
    for k, r in store["runs"].items():
        p = k.split("|")
        if not p[0].startswith(prefix):
            continue
        a = p[0][len(prefix):]
        s = int(p[1])
        assert s not in A[a], (a, s)
        A[a][s] = r
    return A


S82 = arms_of(load(G6, "explore_out/G/s82_devices_results.json"))
S83 = arms_of(load(G6, "explore_out/G/s83_results.json"))
S87 = arms_of(load(G8, "explore_out/G/s87_results.json"))
S89s = load(G8, "explore_out/G/s89_results.json")
POOL, ALLP = arms_of(S89s, "pool."), arms_of(S89s, "allpool.")
draft_check = [("S82", "BASELINE", 0), ("S82", "REG3", 2), ("S82", "REG3_KEYHINGE", 20), ("S82", "REG3_SLOWGATE", 0),
               ("S82", "REG3_NOWIN", 5), ("S83", "LOCAL3_KEYHINGE", 0), ("S83", "REG3_KEYHINGE_POS", 1),
               ("S83", "REG3_FLAT", 0), ("S83", "REG3_KEYHINGE_1200", 18), ("S82", "REG3_NUDGE", 2), ("S83", "REG3_NUDGE", 2)]
n_rec = 0
for src, a, want in draft_check:
    R = (S82 if src == "S82" else S83)[a]
    got = sum(routed_bound(r) for r in R.values())
    say(f"  draft CHECK l.218-220: {src} {a:20s} routed_bound {got}/{len(R)} (draft: {want}) "
        f"{'MATCH' if got == want else 'DIFFERS'}")


def tbo_transition(curve, bind=0.95):
    t = None
    for step, acc, *_ in reversed(curve):
        if acc >= bind:
            t = step
        else:
            break
    return t


allr = [r for A in (S82, S83) for a in A for r in A[a].values() if not a.endswith("_D")]
allr_all = [r for A in (S82, S83) for a in A for r in A[a].values()]
say(f"  records in s82_devices_results.json + s83_results.json at 076005d: {len(allr_all)} (Hebbian {len(allr)}); "
    f"recorded transition = test_binding_onset.transition(curve) on "
    f"{sum(tbo_transition(r['curve']) == r['transition'] for r in allr_all)}/{len(allr_all)}")
say("  S82 + S83 arms present: " + ", ".join(f"{k} {len(v)}" for k, v in sorted({**{f'S82 {a}': S82[a] for a in S82},
                                                                                  **{f'S83 {a}': S83[a] for a in S83}}.items())))
for a in ("BASELINE", "REG3_KEYHINGE", "REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP", "BASELINE_D", "REG3_D",
          "REG3_KEYHINGE_D", "REG3_FREQHINGE_D", "ORACLE_ALL"):
    R = S87[a]
    say(f"  S87 {a:18s} ({min(R)}-{max(R)}): routed_bound {sum(routed_bound(r) for r in R.values())}/{len(R)}")
for a in ("REG3_FREQHINGE", "REG3_KEYHINGE", "BASELINE"):
    m = sorted(r["secs"] / 60 for r in S87[a].values())
    say(f"  S87 {a} minutes per run: median {m[len(m) // 2]:.1f} [{m[0]:.1f}, {m[-1]:.1f}]; cpu {sorted({r['cpu'] for r in S87[a].values()})}")
for nm, T in (("POOL", POOL), ("ALLPOOL", ALLP)):
    say(f"  S89 {nm} (pooled eta^2 view): " + ", ".join(
        f"{a} {sum(routed_bound(r) for r in R.values())}/{len(R)}" for a, R in sorted(T.items())))
say(f"  reproduction references (draft l.254-258): S82 BASELINE|2100 {2100 in S82['BASELINE']}, REG3|2100 "
    f"{2100 in S82['REG3']}, REG3_KEYHINGE|2100 {2100 in S82['REG3_KEYHINGE']}, REG3_NUDGE|2100 "
    f"{2100 in S82['REG3_NUDGE']} (seeds {sorted(S82['REG3_NUDGE'])}), ORACLE_ALL|2100 {2100 in S82['ORACLE_ALL']} "
    f"(recipe {S82['ORACLE_ALL'][2100].get('recipe')}); S83 LOCAL3_KEYHINGE|2100 {2100 in S83['LOCAL3_KEYHINGE']}; "
    f"a Hebbian reg = False record anywhere in S71-S89: "
    f"{any('RESGATE' in a and not a.endswith('_D') for A in (S82, S83, S87) for a in A)}")
S73 = arms_of(load(G6, "explore_out/G/s73_register3_delta_results.json"))
say(f"  S73 (delta, 076005d) RESGATE_ONLY_D routed_bound {sum(routed_bound(r) for r in S73['RESGATE_ONLY_D'].values())}"
    f"/{len(S73['RESGATE_ONLY_D'])}, BOUND {sum(bound(r) for r in S73['RESGATE_ONLY_D'].values())}; REG3_D "
    f"{sum(routed_bound(r) for r in S73['REG3_D'].values())}/{len(S73['REG3_D'])}")


def pair(Ra, Rb):
    s = sorted(set(Ra) & set(Rb))
    b = sum(routed_bound(Ra[x]) and not routed_bound(Rb[x]) for x in s)
    c = sum(routed_bound(Rb[x]) and not routed_bound(Ra[x]) for x in s)
    return b, c, len(s)


PAIRS = {
    "C1 S82 KH vs BASELINE": pair(S82["REG3_KEYHINGE"], S82["BASELINE"]),
    "C1 S87 KH vs BASELINE": pair(S87["REG3_KEYHINGE"], S87["BASELINE"]),
    "C2 S83 KH(S82) vs LOCAL3_KEYHINGE": pair(S82["REG3_KEYHINGE"], S83["LOCAL3_KEYHINGE"]),
    "C3 S82 KH vs REG3": pair(S82["REG3_KEYHINGE"], S82["REG3"]),
    "C3 S89 POOL (delta) KH vs REG3_D": pair(POOL["REG3_KEYHINGE_D"], POOL["REG3_D"]),
    "F1 S87 FREQ vs BASELINE": pair(S87["REG3_FREQHINGE"], S87["BASELINE"]),
    "F2 S87 KH vs FREQ": pair(S87["REG3_KEYHINGE"], S87["REG3_FREQHINGE"]),
    "F2 S89 POOL (delta) KH vs FREQ": pair(POOL["REG3_KEYHINGE_D"], POOL["REG3_FREQHINGE_D"]),
    "RP1 S89 POOL (delta) REPEAT vs BASELINE": pair(POOL["REG3_REPEATHINGE_D"], POOL["BASELINE_D"]),
    "RP2 S89 POOL (delta) KH vs REPEAT": pair(POOL["REG3_KEYHINGE_D"], POOL["REG3_REPEATHINGE_D"]),
    "RP3 S89 POOL (delta) REPEAT vs REG3_D": pair(POOL["REG3_REPEATHINGE_D"], POOL["REG3_D"]),
}
for k, (b, c, n) in PAIRS.items():
    say(f"  pairs {k}: b {b}, c {c}, n {n}, one-sided p {mcnemar1(b, c):.3g}")

# ══ 4. R ═════════════════════════════════════════════════════════════════════
say("\n== 4. The rate claim R (>= 18/20 per machine; pooled RELIABLE >= 36/40)")
for x, n in ((18, 20), (19, 20), (20, 20), (36, 40), (38, 40), (40, 40)):
    wl, wu = wilson(x, n)
    cl, cu = clopper_pearson(x, n)
    say(f"  {x}/{n}: Wilson [{wl:.3f}, {wu:.3f}]; Clopper-Pearson [{cl:.3f}, {cu:.3f}]")
for p in (0.80, 0.85, 0.90, 0.95):
    one = p_ge(20, 18, p)
    say(f"  p {p:.2f}: MET on a machine {one:.3f}; on both {one * one:.3f}; on exactly one {2 * one * (1 - one):.3f}; "
        f"on neither {(1 - one) ** 2:.3f}; pooled >= 36/40 {p_ge(40, 36, p):.3f}")
say("  revision 1 l.162-163 prints 0.04, 0.16, 0.46, 0.85 for 'both': " + ", ".join(
    f"{p_ge(20, 18, p) ** 2:.2f}" for p in (0.80, 0.85, 0.90, 0.95)))

# ══ 5. power ═════════════════════════════════════════════════════════════════
say("\n== 5. McNemar power (exact one-sided; 40 pooled pairs / 20 per machine; at 0.05 and at Holm's first step 0.05/4)")
say("  smallest b with c = 0 that shows: " + ", ".join(
    f"alpha {a:.4f}: b >= {next(b for b in range(1, 30) if mcnemar1(b, 0) < a)}" for a in (0.05, 0.05 / 3, 0.05 / 4)))
for k, (b, c, n) in PAIRS.items():
    qb, qc = b / n, c / n
    say(f"  {k} (discordance {b}/{n} vs {c}/{n}): P(SHOWN) 40 pairs {power_q(40, qb, qc):.3f} (Holm 0.0125: "
        f"{power_q(40, qb, qc, 0.0125):.3f}); 20 pairs {power_q(20, qb, qc):.3f}; P(within 1 pair) 40 pairs "
        f"{within1_q(40, qb, qc):.3f}, 20 pairs {within1_q(20, qb, qc):.3f}")
say("  C4 (no Hebbian record of RESGATE_KEYHINGE): KH at p1 vs RESGATE_KEYHINGE at p2, nested (RESGATE's successes a "
    "subset) / independent; 40 pairs at 0.05:")
for p1 in (1.0, 0.95, 0.90):
    row = []
    for p2 in (0.0, 0.25, 0.5, 0.75, 0.85, 0.90):
        if p2 > p1:
            continue
        qbn, qcn = p1 - p2, 0.0
        qbi, qci = p1 * (1 - p2), p2 * (1 - p1)
        row.append(f"p2 {p2:.2f}: {power_q(40, qbn, qcn):.3f}/{power_q(40, qbi, qci):.3f}")
    say(f"    p1 {p1:.2f}: " + "; ".join(row))
say("  the one-pair rule at 40 pairs (FREQ vs KH, nested): P(KH - FREQ <= 1) at FREQ's true rate (KH 1.0): " + ", ".join(
    f"{p2:.2f}: {within1_q(40, 1 - p2, 0.0):.3f}" for p2 in (0.85, 0.90, 0.95, 0.975, 1.0)))
say("  the one-pair rule at 20 pairs per machine, same: " + ", ".join(
    f"{p2:.2f}: {within1_q(20, 1 - p2, 0.0):.3f}" for p2 in (0.85, 0.90, 0.95, 0.975, 1.0)))
say("  a two-pair-per-40 rule (KH - FREQ <= 2 pooled): " + ", ".join(
    f"{p2:.2f}: {sum(pr for b, c, pr in multinom(40, 1 - p2, 0.0) if b - c <= 2):.3f}" for p2 in (0.85, 0.90, 0.95, 0.975)))

# ══ 6. readings ══════════════════════════════════════════════════════════════
say("\n== 6. Readings as revision 1 words them (l.170-193)")


def read(c1, c2, c3, c4):
    n = []
    if not c1:
        n.append("not shown ... than the baseline")
    if c1 and not c2:
        n.append("... than the baseline; not shown ... than the key term on the window gate")
    if c1 and c2 and not c3:
        n.append("... baseline and window gate; not shown ... than REG3 without the term")
    if c1 and c2 and c3 and not c4:
        n.append("... baseline, window gate, REG3 without the term; not shown ... than the same gate without the register")
    if c1 and c2 and c3 and c4:
        n.append("... all four")
    return n


bad = 0
for c in range(16):
    v = [(c >> i) & 1 == 1 for i in range(4)]
    nm = read(*v)
    bad += len(nm) != 1
    say(f"  C1-C4 {''.join('S' if x else 'N' for x in v)} -> {nm}")
say(f"  combinations with no name or two: {bad}")


def rate(x, l):
    if x is None and l is None:
        return ["R UNTESTED on <machine>: both untested; 'the other machine's verdict' is itself UNTESTED"]
    if x is None or l is None:
        return [f"R UNTESTED on {'X' if x is None else 'L'}, other: {'MET' if (l if x is None else x) else 'NOT MET'}"]
    return [{(1, 1): "R MET on both machines", (1, 0): "R MET on X only", (0, 1): "R MET on L only",
             (0, 0): "R MET on neither"}[(int(x), int(l))]]


for x in (True, False, None):
    for l in (True, False, None):
        say(f"  R X {x}, L {l} -> {rate(x, l)}")

# ══ 7. task structure ════════════════════════════════════════════════════════
say("\n== 7. What each hinge selects on lite (RandHeaderTask P=4, S=2, n_vals=16, nb=2, lmin 1, lmax 3)")
P, S, NV = 4, 2, 16
say(f"  input positions 0-21: CTX {S * 2 + 1} (4 block heads + the query's), KEY {S * P + 1} (8 body + the query's), "
    f"VAL {S * P} (body); the answer (position 22) is the target only")
say("  key term: the 8 body KEY positions, grouped by key id (token - S)")
say("  frequency hinge (S87, report 7:51-53, frequent types {0..5}): 5 CTX + 9 KEY = 14 positions, grouped by token id; "
    "covers the key term's 8 of 8; at KEY, token id = key id; at CTX, token id = the stream")
p_rep = 1 - (1 - 2 / NV) ** (P - 1)
say(f"  repeat hinge (S89's rule: types occurring more than once in the sequence): every CTX and KEY type repeats on lite "
    f"(2 heads per stream; each key once per stream), so 14 positions as the frequency hinge, plus the body VAL positions "
    f"whose value recurs: P(a VAL position repeats) = 1 - (1 - 2/16)^3 = {p_rep:.4f}, expected {8 * p_rep:.2f} per sequence")
rng = random.Random(600)
cnt = collections.Counter()
N = 200000
for _ in range(N):
    vals = [rng.sample(range(NV), S) for _ in range(P)]       # per key, S distinct values (RandHeaderTask.make_batch:69)
    flat = [v for kv in vals for v in kv]
    c = collections.Counter(flat)
    cnt[sum(1 for v in flat if c[v] > 1)] += 1
mean = sum(k * v for k, v in cnt.items()) / N
say(f"  simulation ({N} sequences, stdlib): repeated VAL positions per sequence mean {mean:.3f}; distribution "
    f"{ {k: round(v / N, 4) for k, v in sorted(cnt.items())} }; P(none) {cnt[0] / N:.4f}")
say(f"  so on lite: key term 8 positions; frequency hinge 14 (the 8 + 5 CTX + the query's key); repeat hinge 14 + "
    f"{mean:.2f} VAL; both label-free hinges contain the key term's positions and grouping exactly")

# ══ 8. dates ═════════════════════════════════════════════════════════════════
say("\n== 8. Dates")


def cdate(ref):
    return git("log", "-1", "--format=%cI %h %s", ref).decode().strip()[:140]


say(f"  first review (this branch): {git('log', '--format=%cI %h', '--diff-filter=A', '--', 'docs/review_2026-10/spec_review_register_draft.md').decode().split()[:2]}")
say(f"  plan_2026-10-13.md on the main line: {git('log', '--format=%cI %h', '--diff-filter=A', MAIN, '--', 'docs/reading/plan_2026-10-13.md').decode().split()[:2]}")
lg = git("log", "--format=%cI %h", f"{G6}..{GR1}", "--", "explore_out/G/s89_results.json").decode().split("\n")
lg = [x for x in lg if x]
say(f"  S89 records: first commit {lg[-1]}, last {lg[0]} ({len(lg)} commits)")
say(f"  S89 spec / readings: {cdate('7ab98c16')}; {cdate('db14e9b9')}")
say(f"  report 8: {cdate(G8)}")
say(f"  revision 1: {cdate(GR1)}")
say(f"  S87 report 7 commit: {cdate('e46799a')}")

with open(os.path.join(HERE, "spec_checks_register_v2_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
