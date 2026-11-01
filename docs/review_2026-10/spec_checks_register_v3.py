#!/usr/bin/env python3
"""spec_checks_register_v3.py: the referee's checks behind spec_review_register_v3.md (Session G's
explore_out/G/spec_draft_register.md, revision 2, at claude/explore-G 6fc55b17, against the second review
spec_review_register_v2.md on revision 1 at c98648e6).

    python3 docs/review_2026-10/spec_checks_register_v3.py      (stdlib; reads git only; writes ..._out.txt)

0. Dates: S89's, S89H's and S90's records, report 9, the round-7 prompt, V's S90 screen review and check_G_report9,
   S90 done, revision 2; when S90's spec decision became fixed by the records (per progress commit).
1. Seeds: every JSON / JSONL blob on every remote branch head (today's heads); X 600-619 / L 1600-1619.
2. Files: the diff c98648e6 -> 6fc55b17; explore_g_rhl_parent.py's blob and lines; explore_f_tasks.py's blobs at
   9a6c2bf / 6fc55b17 / the head (the 14,101-byte prefix); make_reg3 (:137) and make_model (:151-159) at 076005d;
   the plan's names; report_10.md and the README's S90 section at 6fc55b17; V's out file on claude/review-V.
3. routed_bound (explore_g_rhl_parent.py:35-53 at 076005d) on S82/S83 (the CHECK), S87, S89, S89H, S90; the pairs the
   draft quotes (S90: REPEAT vs KH, FREQ vs REPEAT, KH vs REG3; S89H's 9 vs 0); iters on S89H ALLPOOL.
4. S90's rule ("FOR THE SPEC", explore_g_s90.py at b57c1867) re-applied, with V's S90 fixes 1-2; its OC.
5. The rate claim R's OC at the screens' KH rates.
6. McNemar power per claim at the screens' rates (S82, S83, S87, S89, S89H, S90); C4's table; Holm over four.
7. Readings: C1-C4 (16 combinations), R (9 machine states) as revision 2 words them.
8. The repeat / frequency sets on lite (stdlib simulation of make_batch's value draw) against the draft's counts.
9. float32: the mean of three identical values (the A_r = 0 CHECK's 'key term bit for bit').
"""
import collections
import datetime
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
from rlib import mcnemar1, wilson  # noqa: E402

out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a)


def exists(ref, path):
    return subprocess.call(("git", "-C", ROOT, "cat-file", "-e", f"{ref}:{path}"), stderr=subprocess.DEVNULL) == 0


def load(ref, path):
    return json.loads(git("show", f"{ref}:{path}"))


MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
GH = "origin/claude/explore-G"
G6, G8, GR1, GR2 = "076005d8", "8e9e3187", "c98648e6", "6fc55b17"
S90A, S90Z, R9, PROMPT = "b57c1867", "2c1913b2", "996e38b7", "5e51a294"


def cdate(ref):
    return git("log", "-1", "--format=%cI", ref).decode().strip()


def cline(ref):
    return git("log", "-1", "--format=%cI %h %s", ref).decode().strip()[:150]


def pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def p_ge(n, k, p):
    return sum(pmf(n, i, p) for i in range(k, n + 1))


def multinom(n, qa, qb):
    qn = 1 - qa - qb
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield a, b, math.comb(n, a) * math.comb(n - a, b) * qa ** a * qb ** b * max(qn, 0.0) ** (n - a - b)


def power_q(n, qb, qc, alpha=0.05):
    return sum(pr for b, c, pr in multinom(n, qb, qc) if mcnemar1(b, c) < alpha)


# ══ 0. dates ═════════════════════════════════════════════════════════════════
say("== 0. Dates (commit times, UTC)")


def file_commits(rng, path):
    lg = [x for x in git("log", "--format=%cI %h", rng, "--", path).decode().split("\n") if x]
    return lg[-1], lg[0], len(lg)


f, l, n = file_commits(f"{G6}..{G8}", "explore_out/G/s89_results.json")
say(f"  S89 records (s89_results.json): first {f}, last {l}, {n} commits; report 8 {cline(G8)}")
f, l, n = file_commits(f"{G8}..{R9}", "explore_out/G/s89h_results.json")
say(f"  S89H records (s89h_results.json): first {f}, last {l}, {n} commits")
say(f"  report 9: {cline(R9)}")
say(f"  round-7 prompt: {cline(PROMPT)}")
say(f"  S90 parent/child: {cline(S90A)}")
f, l, n = file_commits(f"{S90A}..{S90Z}", "explore_out/G/s90_results.json")
say(f"  S90 records (s90_results.json): first {f}, last {l}, {n} commits")
f, l, n = file_commits(f"{S90A}..{S90Z}", "explore_out/G/checks_s90_20261022_050720.json")
say(f"  S90 CHECK file: {f}")
say(f"  S90 done: {cline(S90Z)}")
say(f"  revision 2: {cline(GR2)}")
d = (datetime.datetime.fromisoformat(cdate(GR2)) - datetime.datetime.fromisoformat(cdate(S90Z))).total_seconds()
say(f"  revision 2 - S90 done = {d:.0f} s")
say(f"  S91 parent: {cline('60b1932a')}")
say(f"  G's head now: {cline(GH)}")
for p in ("docs/review_2026-10/check_G_report9.md", "docs/review_2026-10/screen_review_s90_G.md",
          "docs/review_2026-10/spec_review_register_v2.md", "docs/review_2026-10/spec_checks_register_v2_out.txt"):
    lg = git("log", "--format=%cI %h", "--diff-filter=A", "origin/claude/review-V", "--", p).decode().split()
    say(f"  V's {os.path.basename(p)} first on claude/review-V: {lg[:2]}")
say(f"  .py files changed {S90A}..{S90Z}: "
    f"{[x for x in git('diff', '--name-only', S90A, S90Z).decode().split() if x.endswith('.py')] or 'none'}")
say(f"  explore_g_s90.py blob at {S90A} {git('rev-parse', f'{S90A}:explore_g_s90.py').decode()[:10]}, at {GR2} "
    f"{git('rev-parse', f'{GR2}:explore_g_s90.py').decode()[:10]}, at the head "
    f"{git('rev-parse', f'{GH}:explore_g_s90.py').decode()[:10]}")


# ══ 3 (helpers first). routed_bound ══════════════════════════════════════════
def bound(r):
    return r.get("transition") is not None


def routed_star(r):
    e = r.get("end") or {}
    return bool(e.get("margin", 0.0) >= 0.9 and e.get("eta_key_by_stream", 0.0) > 0.9)


def rb(r):
    return bool(r.get("ok", True) and bound(r) and routed_star(r))


def arms_of(store, prefix=""):
    A = collections.defaultdict(dict)
    for k, r in store["runs"].items():
        p = k.split("|")
        if not p[0].startswith(prefix):
            continue
        a, s = p[0][len(prefix):], int(p[1])
        if r.get("ok", True):
            assert s not in A[a], (a, s)
            A[a][s] = r
    return A


def pair(Ra, Rb):
    s = sorted(set(Ra) & set(Rb))
    b = sum(rb(Ra[x]) and not rb(Rb[x]) for x in s)
    c = sum(rb(Rb[x]) and not rb(Ra[x]) for x in s)
    return b, c, len(s)


say("\n   S90's spec decision as the records came in (per progress commit; the rule: the repeat hinge iff REPEAT >= 19/20 "
    "and REG3 <= 15/20, V's screen review §1a):")
for ln in reversed([x for x in git("log", "--format=%h %cI", f"{S90A}..{S90Z}", "--",
                                   "explore_out/G/s90_results.json").decode().split("\n") if x]):
    h, t = ln.split()
    A = arms_of(load(h, "explore_out/G/s90_results.json"), "lite.")
    rp, r3 = A.get("REG3_REPEATHINGE", {}), A.get("REG3", {})
    fails = sorted(s for s, r in rp.items() if not rb(r))
    fixed = "FIXED: falls short" if len(fails) >= 2 else ""
    say(f"    {h} {t[11:19]}: REPEAT {sum(map(rb, rp.values()))}/{len(rp)} (fails {fails}), REG3 "
        f"{sum(map(rb, r3.values()))}/{len(r3)} {fixed}")

# ══ 1. seeds ═════════════════════════════════════════════════════════════════
say("\n== 1. Seeds: every JSON / JSONL blob on every remote branch head (today)")
heads = [b for b in git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").decode().split()
         if not b.endswith("/HEAD")]
for b in heads:
    say(f"  scanned {b} at {git('rev-parse', '--short', b).decode().strip()} ({cdate(b)})")
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
say(f"  {len(heads)} heads; {len(blobs)} distinct JSON/JSONL blobs ({sum(len(v) for v in blobs.values())} "
    f"branch-paths); {n_vals} seed values read; unparsed: {unparsed or 'none'}")
hits = sorted((s, u) for s in BLOCK for u in seed_uses.get(s, ()))
say(f"  records using a seed of X 600-619 / L 1600-1619: {len(hits)}" + (f" {hits[:10]}" if hits else ""))
say(f"  integer dict keys in the block (their parents): "
    f"{ {v: sorted(int_keys[v])[:8] for v in sorted(BLOCK) if v in int_keys} or 'none'}")
for lo, hi in ((2700, 2730), (2840, 2862), (2960, 2992)):
    used = sorted(s for s in seed_uses if lo <= s < hi)
    say(f"  G's screens' seeds in use in {lo}-{hi - 1}: {used[0] if used else '-'}..{used[-1] if used else '-'} "
        f"({len(used)} values); overlap with the block: {sorted(set(used) & BLOCK) or 'none'}")

# ══ 2. files ═════════════════════════════════════════════════════════════════
say("\n== 2. Files")
say(f"  files changed by {GR2}: {git('show', '--name-only', '--format=', GR2).decode().split()}")
say(f"  diff {GR1} -> {GR2}, the draft: {git('diff', '--shortstat', GR1, GR2, '--', 'explore_out/G/spec_draft_register.md').decode().strip()}")
say(f"  the draft at the head {GH} equals {GR2}'s: "
    f"{git('rev-parse', f'{GH}:explore_out/G/spec_draft_register.md') == git('rev-parse', f'{GR2}:explore_out/G/spec_draft_register.md')}")
for ref in (G6, GR2, GH):
    src = git("show", f"{ref}:explore_g_rhl_parent.py").decode().splitlines()
    lines = {nm: i + 1 for i, l in enumerate(src) for nm in ("ROUTE_MARGIN, ROUTE_ETA =", "def bound(", "def routed_star(",
                                                               "def routed_bound(") if l.startswith(nm)}
    say(f"  {ref}: explore_g_rhl_parent.py blob {git('rev-parse', f'{ref}:explore_g_rhl_parent.py').decode()[:10]}; "
        f"lines {lines}; length {len(src)}")
old = git("show", "9a6c2bf:explore_f_tasks.py")
for ref in ("9a6c2bf", G6, GR1, GR2, GH):
    new = git("show", f"{ref}:explore_f_tasks.py")
    dup = [nm for nm, c in collections.Counter(re.findall(rb"^(?:def|class) (\w+)", new, re.M)).items() if c > 1]
    say(f"  explore_f_tasks.py at {ref}: blob {git('rev-parse', f'{ref}:explore_f_tasks.py').decode()[:10]}, "
        f"{len(new)} bytes; 9a6c2bf's ({len(old)} bytes) is its prefix: {new.startswith(old)}; names defined twice: "
        f"{dup or 'none'}")
for ref, path, pat in ((G6, "explore_g_rhl_child.py", r"def make_reg3"), (G6, "explore_g_devices_child.py", r"def make_model"),
                       (G6, "explore_g_rhl_child.py", r"RESGATE")):
    src = git("show", f"{ref}:{path}").decode().splitlines()
    hit = [(i + 1, l.strip()[:120]) for i, l in enumerate(src) if re.search(pat, l)]
    say(f"  {path} at {ref}, '{pat}': {hit[:4]}")
src = git("show", f"{G6}:explore_g_devices_child.py").decode().splitlines()
say("  explore_g_devices_child.py:151-159 at 076005d:")
for i in range(150, 159):
    say(f"    {i + 1}| {src[i]}")
plan = git("show", f"{MAIN}:docs/reading/plan_2026-10-13.md").decode().splitlines()
say("  plan_2026-10-13.md (main line), the register test's names: " + "; ".join(
    f":{i + 1} {l.strip()[:160]}" for i, l in enumerate(plan) if "test_register" in l))
say(f"  report_10.md at {GR2}: {exists(GR2, 'explore_out/G/report_10.md')}; at the head: "
    f"{exists(GH, 'explore_out/G/report_10.md')} (added {git('log', '--format=%cI %h', '--diff-filter=A', GH, '--', 'explore_out/G/report_10.md').decode().split()[:2]})")
rd = git("show", f"{GR2}:explore_out/README.md").decode()
say(f"  README 'Session G — S90' section at {GR2}: {'## Session G — S90' in rd}")
say(f"  spec_checks_register_v2_out.txt §3 on claude/review-V: "
    f"{exists('origin/claude/review-V', 'docs/review_2026-10/spec_checks_register_v2_out.txt')}")
for ref in (MAIN, GH):
    names = [p for p in git("ls-tree", "-r", "--name-only", ref).decode().splitlines()
             if re.search(r"(^|/)(test_register(_gate)?\.py|register_gate\.py|specs/test_register[^/]*\.md)$", p)]
    say(f"  test_register*.py / register_gate.py / specs/test_register*.md on {ref}: {names or 'none'}")

# ══ 3. routed_bound ══════════════════════════════════════════════════════════
say("\n== 3. routed_bound (explore_g_rhl_parent.py:35-53 at 076005d) on G's records")
S82 = arms_of(load(G6, "explore_out/G/s82_devices_results.json"))
S83 = arms_of(load(G6, "explore_out/G/s83_results.json"))
S87 = arms_of(load(G8, "explore_out/G/s87_results.json"))
S89s = load(G8, "explore_out/G/s89_results.json")
POOL, ALLP = arms_of(S89s, "pool."), arms_of(S89s, "allpool.")
S89Hs = load(S90Z, "explore_out/G/s89h_results.json")
HPOOL, HALLP = arms_of(S89Hs, "pool."), arms_of(S89Hs, "allpool.")
S90 = arms_of(load(S90Z, "explore_out/G/s90_results.json"), "lite.")
draft_check = [("S82", "BASELINE", 0), ("S82", "REG3", 2), ("S82", "REG3_KEYHINGE", 20), ("S82", "REG3_SLOWGATE", 0),
               ("S82", "REG3_NOWIN", 5), ("S83", "LOCAL3_KEYHINGE", 0), ("S83", "REG3_KEYHINGE_POS", 1),
               ("S83", "REG3_FLAT", 0), ("S83", "REG3_KEYHINGE_1200", 18), ("S82", "REG3_NUDGE", 2), ("S83", "REG3_NUDGE", 2)]
m = 0
for src, a, want in draft_check:
    R = (S82 if src == "S82" else S83)[a]
    got = sum(rb(r) for r in R.values())
    m += got == want
say(f"  r2 l.268-271 CHECK counts: {m}/{len(draft_check)} MATCH")
heb = sum(len(v) for A in (S82, S83) for a, v in A.items() if not a.endswith("_D"))
alln = sum(len(v) for A in (S82, S83) for v in A.values())
say(f"  records in the two files: {alln} ({heb} Hebbian, {alln - heb} delta)")
for nm, T in (("S87", S87), ("S89 POOL (delta)", POOL), ("S89 ALLPOOL (delta)", ALLP), ("S89H POOL", HPOOL),
              ("S89H ALLPOOL", HALLP), ("S90 (lite)", S90)):
    say(f"  {nm}: " + ", ".join(f"{a} {sum(rb(r) for r in R.values())}/{len(R)}" for a, R in sorted(T.items())))
say(f"  S89H iters: POOL {sorted({r['iters'] for R in HPOOL.values() for r in R.values()})}, ALLPOOL "
    f"{sorted({r['iters'] for R in HALLP.values() for r in R.values()})}; ALLPOOL max stopped_at "
    f"{max(r['stopped_at'] for R in HALLP.values() for r in R.values())}")
say(f"  S90 iters {sorted({r['iters'] for R in S90.values() for r in R.values()})}; completed REG3 {len(S90['REG3'])}, "
    f"REPEAT {len(S90['REG3_REPEATHINGE'])}; attempts {sorted({r['attempt'] for R in S90.values() for r in R.values()})}")
fails = {a: sorted(s for s, r in R.items() if not rb(r)) for a, R in
         (("KH", S87["REG3_KEYHINGE"]), ("FREQ", S87["REG3_FREQHINGE"]), ("REPEAT", S90["REG3_REPEATHINGE"]))}
say(f"  S87/S90 failing seeds: {fails}")
P = {
    "S90 KH(S87) vs REPEAT": pair(S87["REG3_KEYHINGE"], S90["REG3_REPEATHINGE"]),
    "S90 FREQ(S87) vs REPEAT": pair(S87["REG3_FREQHINGE"], S90["REG3_REPEATHINGE"]),
    "S90 KH(S87) vs REG3": pair(S87["REG3_KEYHINGE"], S90["REG3"]),
    "S90 REPEAT vs REG3": pair(S90["REG3_REPEATHINGE"], S90["REG3"]),
    "S90 REPEAT vs BASELINE(S87)": pair(S90["REG3_REPEATHINGE"], S87["BASELINE"]),
    "S89H KH vs REG3": pair(HPOOL["REG3_KEYHINGE_H"], HPOOL["REG3_H"]),
    "S89H REPEAT vs REG3": pair(HPOOL["REG3_REPEATHINGE_H"], HPOOL["REG3_H"]),
    "S89H KH vs REPEAT": pair(HPOOL["REG3_KEYHINGE_H"], HPOOL["REG3_REPEATHINGE_H"]),
    "S89H KH vs BASELINE": pair(HPOOL["REG3_KEYHINGE_H"], HPOOL["BASELINE_H"]),
    "S89 POOL KH vs REPEAT (delta)": pair(POOL["REG3_KEYHINGE_D"], POOL["REG3_REPEATHINGE_D"]),
    "S89 POOL KH vs REG3 (delta)": pair(POOL["REG3_KEYHINGE_D"], POOL["REG3_D"]),
    "S89 POOL KH vs BASELINE (delta)": pair(POOL["REG3_KEYHINGE_D"], POOL["BASELINE_D"]),
    "S82 KH vs BASELINE": pair(S82["REG3_KEYHINGE"], S82["BASELINE"]),
    "S82 KH vs REG3": pair(S82["REG3_KEYHINGE"], S82["REG3"]),
    "S83 KH(S82) vs LOCAL3_KEYHINGE": pair(S82["REG3_KEYHINGE"], S83["LOCAL3_KEYHINGE"]),
    "S87 KH vs BASELINE": pair(S87["REG3_KEYHINGE"], S87["BASELINE"]),
}
for k, (b, c, n) in P.items():
    say(f"  pairs {k}: b {b}, c {c}, n {n}, one-sided p(b > c) {mcnemar1(b, c):.3g}")

# ══ 4. S90's rule ════════════════════════════════════════════════════════════
say("\n== 4. S90's spec rule (explore_g_s90.py:48-50, :99-101 at b57c1867), re-applied")
src = git("show", f"{S90A}:explore_g_s90.py").decode().splitlines()
for i in list(range(40, 53)) + list(range(98, 101)):
    say(f"    s90:{i + 1}| {src[i]}")


def s90_rule(kh, base, r3, rp, n=20, v_fixes=True):
    """Marginal form of explore_g_s89h.readings with KH n/n and BASELINE 0/n fixed (V's S90 review §1a), and V's
    fixes 1-2 (UNTESTED -> no decision; REG3 < 18 completed -> UNTESTED)."""
    if v_fixes and (len(rp) < n or len(r3) < 18):
        return "UNTESTED: no spec decision"
    c = sum(rp)
    r = sum(r3)
    if c >= n - 1 and r <= n - 5:
        return "the repeat hinge (holds, and the hinge is needed here)"
    if c >= n - 1:
        return "falls short (holds, descriptive)"
    return "falls short (does not hold: " + ("is weaker" if c >= n - 3 else "needs key identities") + ")"


rp = [rb(r) for r in S90["REG3_REPEATHINGE"].values()]
r3 = [rb(r) for r in S90["REG3"].values()]
say(f"  S90: REPEAT {sum(rp)}/{len(rp)}, REG3 {sum(r3)}/{len(r3)}, KH (S87) {sum(rb(r) for r in S87['REG3_KEYHINGE'].values())}/20, "
    f"BASELINE (S87) {sum(rb(r) for r in S87['BASELINE'].values())}/20 -> with V's fixes 1-2: {s90_rule(1, 0, r3, rp)}; "
    f"without: {s90_rule(1, 0, r3, rp, v_fixes=False)}")
say(f"  G's s90_summary.json at {S90Z}: question / FOR THE SPEC = "
    f"{load(S90Z, 'explore_out/G/s90_summary.json')['readings']['question']!r} / "
    f"{load(S90Z, 'explore_out/G/s90_summary.json')['readings']['FOR THE SPEC']!r}")
say("  OC of the rule (REG3 at 0.2): P(the repeat hinge) = P(REPEAT >= 19/20) * P(REG3 <= 15/20): " + ", ".join(
    f"{p:.2f}: {p_ge(20, 19, p) * (1 - p_ge(20, 16, 0.2)):.3f}" for p in (1.0, 0.95, 0.9, 0.85, 0.8)))
say(f"  'falls short' vs a test: p(KH > REPEAT) at 3 vs 0 = {mcnemar1(3, 0):.3f}; two-sided {min(1, 2 * mcnemar1(3, 0)):.3f}")
for p in (0.85, 0.9, 0.95):
    say(f"  P(REPEAT <= 17/20 | true rate {p}) = {1 - p_ge(20, 18, p):.3f}; P(<= 18/20) = {1 - p_ge(20, 19, p):.3f}")
# what REPEAT's 17/20 supports
lo, hi = wilson(17, 20)
say(f"  REPEAT 17/20: Wilson [{lo:.3f}, {hi:.3f}]")

# ══ 5. R ═════════════════════════════════════════════════════════════════════
say("\n== 5. The rate claim R (>= 18/20 per machine) at the screens' KH rates")
for nm, p in (("S82/S87/S89 KH 20/20 (as 0.975)", 0.975), ("0.95", 0.95), ("S89H KH 18/20", 0.90), ("0.85", 0.85),
              ("0.80", 0.80)):
    one = p_ge(20, 18, p)
    say(f"  {nm}: MET on a machine {one:.3f}; on both {one * one:.3f}; on exactly one {2 * one * (1 - one):.3f}; "
        f"on neither {(1 - one) ** 2:.3f}; pooled >= 36/40 {p_ge(40, 36, p):.3f}")
say("  r2 l.208-209 prints 0.04, 0.16, 0.46, 0.85: " + ", ".join(f"{p_ge(20, 18, p) ** 2:.2f}" for p in (0.80, 0.85, 0.90, 0.95)))

# ══ 6. power ═════════════════════════════════════════════════════════════════
say("\n== 6. McNemar power per claim (exact one-sided; observed discordance as the per-pair rates; 40 pooled / 20 per machine)")
say("  smallest b with c = 0 that shows: " + ", ".join(
    f"alpha {a:.4f}: b >= {next(b for b in range(1, 30) if mcnemar1(b, 0) < a)}" for a in (0.05, 0.05 / 4)))
CL = [("C1", "S82 KH vs BASELINE"), ("C1", "S87 KH vs BASELINE"), ("C1", "S89 POOL KH vs BASELINE (delta)"),
      ("C1", "S89H KH vs BASELINE"), ("C2", "S83 KH(S82) vs LOCAL3_KEYHINGE"), ("C3", "S82 KH vs REG3"),
      ("C3", "S90 KH(S87) vs REG3"), ("C3", "S89 POOL KH vs REG3 (delta)"), ("C3", "S89H KH vs REG3")]
for cl, k in CL:
    b, c, n = P[k]
    qb, qc = b / n, c / n
    say(f"  {cl} at {k} ({b} vs {c} of {n}): P(SHOWN) 40 pairs {power_q(40, qb, qc):.3f} (Holm 0.0125: "
        f"{power_q(40, qb, qc, 0.0125):.3f}); 20 pairs {power_q(20, qb, qc):.3f}")
say("  C4 (RESGATE_KEYHINGE unscreened): KH at p1, RESGATE at p2, nested; 40 pairs at 0.05 (Holm 0.0125):")
for p1 in (1.0, 0.95, 0.90):
    say(f"    KH {p1:.2f}: " + "; ".join(f"p2 {p2:.2f}: {power_q(40, p1 - p2, 0.0):.3f} ({power_q(40, p1 - p2, 0.0, 0.0125):.3f})"
                                     for p2 in (0.0, 0.5, 0.75, 0.85, 0.90) if p2 <= p1))
say("  r2 l.200-202 prints 1.00, 0.98, 0.74, 0.37 at 0.50, 0.75, 0.85, 0.90 (KH 1.0): " + ", ".join(
    f"{power_q(40, 1 - p2, 0.0):.2f}" for p2 in (0.5, 0.75, 0.85, 0.90)))
say("  for comparison, the arm left out (REPEAT, at S90's discordance KH 3 vs 0): P(KH > REPEAT SHOWN) 40 pairs "
    f"{power_q(40, 3 / 20, 0):.3f}, 20 pairs {power_q(20, 3 / 20, 0):.3f}; REPEAT vs BASELINE 17 vs 0: 40 pairs "
    f"{power_q(40, 17 / 20, 0):.3f}")

# ══ 7. readings ══════════════════════════════════════════════════════════════
say("\n== 7. Readings as revision 2 words them (l.216-244)")


def read(c1, c2, c3, c4):
    n = []
    if not c1:
        n.append("l.218 not shown ... baseline")
    if c1 and not c2:
        n.append("l.220 ... baseline; not shown ... window gate")
    if c1 and c2 and not c3:
        n.append("l.222 ... baseline and window gate; not shown ... REG3 without the term")
    if c1 and c2 and c3 and not c4:
        n.append("l.224 ...; not shown ... the same gate and term without the register")
    if c1 and c2 and c3 and c4:
        n.append("l.227 all four")
    return n


bad = sum(len(read(*[(c >> i) & 1 == 1 for i in range(4)])) != 1 for c in range(16))
say(f"  C1-C4: 16 combinations, with no name or two names: {bad}")


def rate_literal(x, l):
    """Each listed name, read literally: MET/NOT MET/UNTESTED per machine."""
    n = []
    if x == "M" and l == "M":
        n.append("R MET on both machines")
    if x == "M" and l != "M":
        n.append("R MET on X only")
    if l == "M" and x != "M":
        n.append("R MET on L only")
    if x != "M" and l != "M":
        n.append("R MET on neither")
    if (x == "U") != (l == "U"):
        n.append(f"R UNTESTED on {'X' if x == 'U' else 'L'}")
    if x == "U" and l == "U":
        n.append("R UNTESTED on both machines")
    return n


def rate_prec(x, l):
    """The same with UNTESTED taking precedence (the MET names only when both machines are VALID)."""
    if "U" in (x, l):
        return ["R UNTESTED on both machines"] if x == l else [f"R UNTESTED on {'X' if x == 'U' else 'L'}"]
    return rate_literal(x, l)


for x in "MNU":
    for l in "MNU":
        say(f"  R X {x}, L {l}: literal {rate_literal(x, l)}; UNTESTED first {rate_prec(x, l)}")
say(f"  states with exactly one name: literal {sum(len(rate_literal(x, l)) == 1 for x in 'MNU' for l in 'MNU')}/9; "
    f"UNTESTED first {sum(len(rate_prec(x, l)) == 1 for x in 'MNU' for l in 'MNU')}/9")

# ══ 8. the sets on lite ══════════════════════════════════════════════════════
say("\n== 8. The frequency / repeat sets on lite (RandHeaderTask P=4, S=2, n_vals=16)")
Pk, S, NV = 4, 2, 16
# make_batch:69 draws, per key, S distinct values; a body VAL repeats if its value occurs at another body VAL.
rng = random.Random(600)
cnt = collections.Counter()
N = 200000
pos_rep = 0
for _ in range(N):
    vals = [rng.sample(range(NV), S) for _ in range(Pk)]
    flat = [v for kv in vals for v in kv]
    c = collections.Counter(flat)
    k = sum(1 for v in flat if c[v] > 1)
    cnt[k] += 1
    pos_rep += k
mean = pos_rep / N
say(f"  simulated ({N} sequences): repeated VAL positions per sequence mean {mean:.3f} (per position {mean / 8:.3f}); "
    f"P(none) {cnt[0] / N:.4f}; distribution { {k: round(v / N, 4) for k, v in sorted(cnt.items())} }")
say(f"  exact per position: 1 - (1 - 2/16)^3 = {1 - (1 - 2 / 16) ** 3:.4f}; x 8 = {8 * (1 - (1 - 2 / 16) ** 3):.3f} "
    "(independence approximation)")
say(f"  draft: 5 CTX + 9 KEY + 2.63 VAL = {5 + 9 + 2.63:.2f} (draft '16.6'); frequency 8 + 1 + 5 = 14; G's CHECK rows: "
    "16.63 = CTX 4 + KEY 8 + VAL 2.63 + QUERY 2; frequency 14.00 = CTX 4 + KEY 8 + QUERY 2")
say("  so the repeat set = the frequency set (on every sequence) + the chance-repeated VALs, grouped by the same token id")

# ══ 9. the A_r = 0 CHECK's "key term bit for bit" (r2 l.278-279) ═══════════════
say("\n== 9. float32: is the mean of three identical values the value? (RESGATE's term is a mean over 3 layers, "
    "LOCAL3_KEYHINGE's over one gate)")
import struct  # noqa: E402


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


rng = random.Random(0)
NB = 100000
bad = 0
for _ in range(NB):
    x = f32(rng.random())
    bad += f32(f32(f32(x + x) + x) / 3) != x
say(f"  sequential float32 sum then /3: differs from x on {bad}/{NB} uniform draws ({bad / NB:.3f})")
say("  and at initialisation relu(eta2_key - 0.2) is 0 whenever eta2_key < 0.2, so 0 = 0 passes whatever the gates")

with open(os.path.join(HERE, "spec_checks_register_v3_out.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
