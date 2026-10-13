#!/usr/bin/env python3
"""spec_checks_register_keyterm.py: the referee's checks behind spec_review_register_draft.md (Session G's
explore_out/G/spec_draft_register.md at 076005d) and spec_review_two_keyterm_draft.md (Session H's
explore_out/H/spec_draft_two_keyterm.md at 9b72bf6).

    python3 docs/review_2026-10/spec_checks_register_keyterm.py      (stdlib; reads git only; writes ..._out.txt)

1. Seeds: every JSON / JSONL blob on every remote branch head, parsed; every "ARM|seed[|...]" key, every "seed" field and
   every "seeds" list collected; hits on X 600-639 / L 1600-1639 (both drafts' blocks); integer dict keys in those ranges
   reported with their parent key (update steps, not seeds); the neighbouring blocks in use; text mentions of the blocks.
2. Facts G's draft cites: S82 / S83 / S72 counts on ROUTED-BOUND (re-implemented here from the definition in
   explore_g_rhl_parent.py at 076005d), paired b / c, the thresholds' margin of safety, the nudge's history, timings.
3. Facts H's draft cites: S85, S52 (and batch 16's LOCAL3_SLOW on the same seeds), S78, the main line's Part A WIN3_SLOW.
4. The rate claims (G's R, H's K3): Wilson and Clopper-Pearson bounds at the thresholds; P(pass) by exact binomial per
   machine, on both machines, and for pooled alternatives; P("one machine only") under equal rates; predictive
   probabilities from the screens.
5. McNemar power: G's C1-C3 at the S82 / S83 rates; H's K1 at the screens' and the main line's rates; H's no-room
   region; the n H's K1 needs; Holm's thresholds.
6. Readings: every combination of outcomes enumerated for both drafts, with the reading each prints.
7. Validity: P(the oracle passes 2/2) for G's REG3_NUDGE at its observed record.
"""
import collections
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rlib import load, runs, mcnemar1, wilson  # noqa: E402

out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a)


MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
G6, H8, OI = "076005d", "9b72bf6", "origin/claude/outside-ideas"

# ── statistics ───────────────────────────────────────────────────────────────


def pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def p_ge(n, k, p):
    return sum(pmf(n, i, p) for i in range(k, n + 1))


def clopper_pearson(x, n, a=0.05):
    def bis(f, lo=0.0, hi=1.0):
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
            yield a, b, math.comb(n, a) * math.comb(n - a, b) * qa ** a * qb ** b * qn ** (n - a - b)


def cells(p1, p2, nested):
    """(P(first only), P(second only)) per seed: independent outcomes, or nested (the worse arm's successes a subset)."""
    return (max(p1 - p2, 0.0), max(p2 - p1, 0.0)) if nested else (p1 * (1 - p2), p2 * (1 - p1))


def power(n, p1, p2, nested, alpha=0.05):
    qb, qc = cells(p1, p2, nested)
    return sum(pr for b, c, pr in multinom(n, qb, qc) if mcnemar1(b, c) < alpha)


def power_q(n, qb, qc, alpha=0.05):
    return sum(pr for b, c, pr in multinom(n, qb, qc) if mcnemar1(b, c) < alpha)


def betabin_ge(n, k, a, b):
    """P(X >= k), X ~ BetaBinomial(n, a, b)."""
    lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return sum(math.exp(math.log(math.comb(n, i)) + lb(i + a, n - i + b) - lb(a, b)) for i in range(k, n + 1))


# ══ 1. seeds ═════════════════════════════════════════════════════════════════
say("== 1. Seeds: every JSON / JSONL blob on every remote branch head")
heads = [b for b in git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").decode().split()
         if not b.endswith("/HEAD")]
for b in heads:
    say(f"  scanned {b} at {git('rev-parse', '--short', b).decode().strip()}")
G_BLOCK = set(range(600, 620)) | set(range(1600, 1620))
H_BLOCK = set(range(620, 640)) | set(range(1620, 1640))
KEYRE = re.compile(r"^[A-Za-z0-9_.\-]+\|(-?\d+)(\|.*)?$")
blobs = {}
for b in heads:
    for line in git("ls-tree", "-r", b).decode().splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and path.endswith((".json", ".jsonl")):
            blobs.setdefault(sha, []).append((b, path))
seed_uses = collections.defaultdict(set)      # seed -> {(branch, path, how)}
int_keys = collections.defaultdict(set)       # value -> {parent key}
n_vals = 0


def walk(o, locs, parent=""):
    global n_vals
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRE.match(k) if isinstance(k, str) else None
            if m:
                n_vals += 1
                for b, p in locs:
                    seed_uses[int(m.group(1))].add((b, p, "key " + k.split("|")[0]))
            elif isinstance(k, str) and k.isdigit():
                int_keys[int(k)].add(str(parent))
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool):
                n_vals += 1
                for b, p in locs:
                    seed_uses[v].add((b, p, "seed field"))
            if k == "seeds" and isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and not isinstance(x, bool):
                        n_vals += 1
                        for b, p in locs:
                            seed_uses[x].add((b, p, "seeds list"))
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
    f"values read (keys, 'seed' fields, 'seeds' lists); unparsed: {unparsed or 'none'}")
for name, blk in (("G (X 600-619, L 1600-1619)", G_BLOCK), ("H (X 620-639, L 1620-1639)", H_BLOCK)):
    hits = sorted((s, u) for s in blk for u in seed_uses.get(s, ()))
    say(f"  {name}: records using a seed of the block: {len(hits)}" + (f" {hits[:10]}" if hits else ""))
    ik = {v: sorted(int_keys[v]) for v in sorted(blk) if v in int_keys}
    say(f"    integer dict keys with these values (parents): {ik or 'none'}")


def blocks(lo, hi):
    used = sorted(s for s in seed_uses if lo <= s < hi)
    out_, start = [], None
    for i, s in enumerate(used):
        if start is None:
            start = s
        if i + 1 == len(used) or used[i + 1] != s + 1:
            out_.append(f"{start}-{s}" if s > start else str(start))
            start = None
    return out_


say(f"  seeds in use in 500-699: {blocks(500, 700)}")
say(f"  seeds in use in 1500-1699: {blocks(1500, 1700)}")
for s in (599, 1599, 640, 1640, 700, 1700):
    say(f"    seed {s}: {sorted(set((b.split('/')[-1], p) for b, p, _ in seed_uses.get(s, ())))[:4] or 'unused'}")
pat = r"(^|[^0-9.e])(6[0-3][0-9]|16[0-3][0-9]) ?(-|–|\.\.) ?(6[0-3][0-9]|16[0-3][0-9])([^0-9]|$)|range\((6[0-3][0-9]|16[0-3][0-9]),"
txt = set()
for b in heads:
    try:
        r = git("grep", "-n", "-E", pat, b, "--", "*.py", "*.md", "*.tex", "*.txt").decode()
    except subprocess.CalledProcessError:
        r = ""
    for ln in r.splitlines():
        _, path, lno = ln.split(":", 3)[:3]
        txt.add((path, b.split("/")[-1]))
say("  text mentions of a block (*.py, *.md, *.tex, *.txt on every head; path, branch):")
for path, b in sorted(txt):
    say(f"    {path} ({b})")

# ══ 2. G's facts ═════════════════════════════════════════════════════════════
say("\n== 2. G's draft: the records it cites (claude/explore-G at 076005d)")
RM, RE = 0.9, 0.9      # explore_g_rhl_parent.py:35 ROUTE_MARGIN, ROUTE_ETA


def g_bound(r):
    return r["transition"] is not None


def g_routed_star(r):
    e = r["end"]
    return bool(e.get("margin", 0.0) >= RM and e.get("eta_key_by_stream", 0.0) > RE)


def g_rb(r):
    return bool(r.get("ok", True) and g_bound(r) and g_routed_star(r))


S82 = load(G6, "explore_out/G/s82_devices_results.json")
S83 = load(G6, "explore_out/G/s83_results.json")
S72 = load(G6, "explore_out/G/s72_register3_results.json")
A = {}
for store, arms in ((S82, ("BASELINE", "REG3", "REG3_KEYHINGE", "REG3_SLOWGATE", "REG3_NOWIN", "REG3_NUDGE", "ORACLE_ALL")),
                    (S83, ("LOCAL3_KEYHINGE", "REG3_KEYHINGE_POS", "REG3_FLAT", "REG3_KEYHINGE_1200", "REG3_NUDGE"))):
    for a in arms:
        A[(a, "S83" if store is S83 else "S82")] = runs(store, a)
for (a, src), R in A.items():
    secs = sorted(r["secs"] / 60 for r in R.values())
    say(f"  {src} {a:20s} n {len(R):2d}: BOUND {sum(g_bound(r) for r in R.values()):2d}, ROUTED-BOUND "
        f"{sum(g_rb(r) for r in R.values()):2d}, recorded discovered {sum(bool(r['discovered']) for r in R.values()):2d}; "
        f"minutes median {secs[len(secs) // 2]:.1f} [{secs[0]:.1f}, {secs[-1]:.1f}]; cpu {sorted({r['cpu'] for r in R.values()})}")


def pair(Ra, Rb, f):
    s = sorted(set(Ra) & set(Rb))
    b = sum(f(Ra[x]) and not f(Rb[x]) for x in s)
    c = sum(f(Rb[x]) and not f(Ra[x]) for x in s)
    return b, c, len(s), mcnemar1(b, c)


KH = A[("REG3_KEYHINGE", "S82")]
for name, other in (("C1 vs BASELINE", A[("BASELINE", "S82")]), ("C2 vs LOCAL3_KEYHINGE", A[("LOCAL3_KEYHINGE", "S83")]),
                    ("C3 vs REG3", A[("REG3", "S82")]), ("KH_1200 vs BASELINE", None)):
    if other is None:
        b, c, n, p = pair(A[("REG3_KEYHINGE_1200", "S83")], A[("BASELINE", "S82")], g_rb)
    else:
        b, c, n, p = pair(KH, other, g_rb)
    say(f"  {name}: b {b}, c {c}, n {n}, one-sided p {p:.3g}")
allr = [r for R in A.values() for r in R.values()]
near = [(r["arm"], r["seed"], round(r["end"]["margin"], 3), round(r["end"]["eta_key_by_stream"], 3)) for r in allr
        if g_bound(r) and (0.75 <= r["end"]["margin"] < 0.98 or 0.75 <= r["end"]["eta_key_by_stream"] < 0.98)]
say(f"  bound records ({sum(g_bound(r) for r in allr)}) with end margin or eta_key_by_stream in [0.75, 0.98): {near or 'none'}")
mg = sorted(round(r["end"]["margin"], 2) for r in allr if g_bound(r))
say(f"  end margins of bound records: min {mg[0]}, max {mg[-1]}; values: {collections.Counter(mg).most_common(6)}")
bnr = [(r["arm"], r["seed"], round(r["end"]["margin"], 2), round(r["end"]["eta_key_by_stream"], 2)) for r in allr
       if g_bound(r) and not g_routed_star(r)]
say(f"  bound but not ROUTED* (the key-split bindings ROUTED-BOUND excludes): {len(bnr)}: {bnr}")


def tbo_transition(curve, bind=0.95):
    """test_binding_onset.transition (main line, :247-255; BIND_PASS 0.95), re-implemented."""
    t = None
    for step, acc, _ in reversed(curve):
        if acc >= bind:
            t = step
        else:
            break
    return t


mism = [(r["arm"], r["seed"]) for r in allr if tbo_transition(r["curve"]) != r["transition"]]
say(f"  recorded 'transition' against test_binding_onset.transition(curve) on all {len(allr)} S82/S83 records: "
    f"{len(mism)} differ {mism[:5]}")
lay = collections.Counter()
for r in allr:
    L = r["end"].get("layers")
    if g_bound(r) and L and all(L.get(f"L{i}", {}).get("margin") is not None for i in (1, 2, 3)):
        lay[(all(L[f"L{i}"]["margin"] >= RM and L[f"L{i}"]["eta_key_by_stream"] > RE for i in (1, 2, 3)), g_routed_star(r))] += 1
say(f"  bound records with per-layer statistics, (ROUTED* at every layer, ROUTED* on the returned gate): {dict(lay)}")
nud = runs(S72, "REG3_NUDGE")
nn = (sum(g_rb(r) for r in nud.values()), len(nud))
say(f"  REG3_NUDGE ROUTED-BOUND: S72 {nn[0]}/{nn[1]} (seeds {min(nud)}-{max(nud)}), S82 "
    f"{sum(g_rb(r) for r in A[('REG3_NUDGE', 'S82')].values())}/2, S83 {sum(g_rb(r) for r in A[('REG3_NUDGE', 'S83')].values())}/2")
say(f"  S72 (700-719) ROUTED-BOUND: BASELINE {sum(g_rb(r) for r in runs(S72, 'BASELINE').values())}/20, REG3 "
    f"{sum(g_rb(r) for r in runs(S72, 'REG3').values())}/20; S71 (700-709) BOUND: " + ", ".join(
    f"{a} {sum(g_bound(r) for r in runs(load(G6, 'explore_out/G/s71_layers_results.json'), a).values())}/10"
    for a in ("ORACLE_ALL", "ORACLE_23", "ORACLE_1")))
src = git("show", f"{G6}:explore_g_devices.py").decode().splitlines()
say(f"  S82's validity rule as coded (explore_g_devices.py:{[i + 1 for i, l in enumerate(src) if 'VALID' in l and 'no >=' in l]}): "
    f"{[l.strip() for l in src if 'VALID' in l and 'no >=' in l]}")
for f in ("explore_f_tasks.py", "explore_b18_gates.py", "explore_aux_gate.py", "explore_keysplits_child.py"):
    shas = {ref: git("rev-parse", f"{ref}:{f}").decode().strip()[:10] for ref in ("9a6c2bf", G6, H8, OI)
            if subprocess.call(("git", "-C", ROOT, "cat-file", "-e", f"{ref}:{f}"), stderr=subprocess.DEVNULL) == 0}
    say(f"  blob of {f}: {shas}; on the main line: "
        f"{subprocess.call(('git', '-C', ROOT, 'cat-file', '-e', f'{MAIN}:{f}'), stderr=subprocess.DEVNULL) == 0}")
for f in ("test_binding_onset.py", "test_window_gate.py", "fail_class_v2.py", "test_slow_start.py", "test_short_conv.py"):
    s = git("show", f"{MAIN}:{f}").decode()
    names = [n for n in ("def transition(", "def success(", "def to_local(", "def fail_class_v2(", "def outcome_v2(",
                         "class Inject(", "def eta2_hinge(") if n in s]
    say(f"  main line {f}: {names}")

# ══ 3. H's facts ═════════════════════════════════════════════════════════════
say("\n== 3. H's draft: the records it cites")
S85 = load(H8, "explore_out/H/two_b_results.json")
disc = lambda r: bool(r.get("ok", True) and r["discovered"])
for a in ("REF", "KEYHINGE", "SLOWGATE", "SLOWGATE_KEYHINGE", "ORACLE"):
    R = runs(S85, a)
    secs = sorted(r["secs"] / 60 for r in R.values())
    say(f"  S85 {a:18s} n {len(R):2d} discovered {sum(disc(r) for r in R.values()):2d}; minutes median "
        f"{secs[len(secs) // 2]:.1f} [{secs[0]:.1f}, {secs[-1]:.1f}]")
b85 = pair(runs(S85, "KEYHINGE"), runs(S85, "REF"), disc)
say(f"  S85 KEYHINGE vs REF: b {b85[0]}, c {b85[1]}, n {b85[2]}, p {b85[3]:.4f}")
S52 = load(OI, "explore_out/two_stream_keysplits_results.json")
B16 = load(OI, "explore_out/window_recipe_results.json")
kh52, l3 = runs(S52, "W2_KEYHINGE"), runs(B16, "LOCAL3_SLOW")
b52 = pair(kh52, l3, disc)
say(f"  S52 W2_KEYHINGE (seeds {min(kh52)}-{max(kh52)}) discovered {sum(disc(r) for r in kh52.values())}/{len(kh52)}; "
    f"batch 16 LOCAL3_SLOW {sum(disc(r) for r in l3.values())}/{len(l3)}; b {b52[0]}, c {b52[1]}, p {b52[3]:.3f}")
r160 = kh52[160]
say(f"  S52 W2_KEYHINGE|160 hinge record (for T3): fired {r160['hinge']['fired']}, eta_at {r160['hinge']['eta_at']}, "
    f"cpu {r160['cpu']}; S85 KEYHINGE|2310 present: {2310 in runs(S85, 'KEYHINGE')}, its hinge record "
    f"{'hinge' in runs(S85, 'KEYHINGE')[2310]}")
S78 = load(H8, "explore_out/H/two_results.json")
ref78 = runs(S78, "REF")
say(f"  S78 REF (WIN3_SLOW, {min(ref78)}-{max(ref78)}): discovered {sum(disc(r) for r in ref78.values())}/{len(ref78)}")
wg = {m: runs(load(MAIN, f"results/{m}/window_gate_results.json"), "WIN3_SLOW") for m in ("X", "L")}
for m, R in wg.items():
    say(f"  main line test_window_gate Part A WIN3_SLOW {m} ({min(R)}-{max(R)}): discovered "
        f"{sum(disc(r) for r in R.values())}/{len(R)}")
tot_ws = sum(disc(r) for R in list(wg.values()) + [l3, ref78, runs(S85, 'REF')] for r in R.values())
n_ws = sum(len(R) for R in list(wg.values()) + [l3, ref78, runs(S85, 'REF')])
say(f"  WIN3_SLOW (the same arm, CHECK 135) over every record above: {tot_ws}/{n_ws} = {tot_ws / n_ws:.3f}; key term "
    f"(S52 + S85): {sum(disc(r) for r in kh52.values()) + sum(disc(r) for r in runs(S85, 'KEYHINGE').values())}/60")

# ══ 4. rate claims ═══════════════════════════════════════════════════════════
say("\n== 4. The rate claims (G's R and H's K3: >= 18/20 on each machine)")
for n, xs in ((20, range(16, 21)), (40, range(34, 41))):
    for x in xs:
        wl, wu = wilson(x, n)
        cl, cu = clopper_pearson(x, n)
        say(f"  {x}/{n}: Wilson 95% [{wl:.3f}, {wu:.3f}]; Clopper-Pearson 95% [{cl:.3f}, {cu:.3f}]")
say("  P(pass) by exact binomial at the true rate p (machines independent, equal rates):")
for p in (0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.975, 0.99):
    one = p_ge(20, 18, p)
    say(f"    p {p:.3f}: >= 18/20 on one machine {one:.3f}; on both {one * one:.3f}; on exactly one {2 * one * (1 - one):.3f}; "
        f"pooled >= 36/40 {p_ge(40, 36, p):.3f}; pooled >= 37/40 {p_ge(40, 37, p):.3f}; pooled >= 38/40 {p_ge(40, 38, p):.3f}")
say("  the 'both machines' rule is not monotone in the pooled count: X 20/20 + L 17/20 = 37/40 fails it, X 18/20 + L 18/20 = "
    "36/40 passes")
for lbl, s, f in (("G: S82 REG3_KEYHINGE 20/20", 20, 0), ("G: S82 20/20 + S83 KH_1200 18/20 (a different window)", 38, 2),
                  ("H: S52 36/40 + S85 20/20", 56, 4)):
    for pr, (a0, b0) in (("uniform", (1, 1)), ("Jeffreys", (0.5, 0.5))):
        a, b = a0 + s, b0 + f
        one = betabin_ge(20, 18, a, b)
        # E[P(>= 18 | p)^2] over p ~ Beta(a, b), with p = 1 - u^2 (removes the b < 1 singularity at p = 1)
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        us = [(i + 0.5) / 20000 for i in range(20000)]
        w = [2 * math.exp((a - 1) * math.log(1 - u * u) + (2 * b - 1) * math.log(u) - lbeta) / 20000 for u in us]
        shared = sum(wi * p_ge(20, 18, 1 - u * u) ** 2 for wi, u in zip(w, us)) / sum(w)
        say(f"  predictive, {lbl}, {pr} prior: P(>= 18/20 on a machine) {one:.3f}; on both: {one * one:.3f} if each "
            f"machine has its own rate, {shared:.3f} if they share one")

# ══ 5. power ═════════════════════════════════════════════════════════════════
say("\n== 5. McNemar power (exact one-sided, SHOWN at p < 0.05; pooled 40 pairs, per machine 20)")
say("  G: REG3_KEYHINGE at p1 against a comparator at p2 (S82/S83: BASELINE 0/20, LOCAL3_KEYHINGE 0/20, REG3 2/20; Wilson "
    "upper bounds 0.16, 0.16, 0.30)")
for p1 in (0.50, 0.70, 0.80, 0.90):
    row = []
    for p2 in (0.0, 0.10, 0.16, 0.30):
        row.append(f"p2 {p2:.2f}: 40 {power(40, p1, p2, False):.3f}/{power(40, p1, p2, True):.3f}, "
                   f"20 {power(20, p1, p2, False):.3f}/{power(20, p1, p2, True):.3f}")
    say(f"    p1 {p1:.2f} (indep/nested): " + "; ".join(row))
say("    at Holm's or Bonferroni's 0.05/3: " + "; ".join(
    f"p1 {p1}, p2 {p2}: 40 {power(40, p1, p2, False, 0.05 / 3):.3f}" for p1, p2 in ((0.8, 0.3), (0.7, 0.3), (0.5, 0.16))))
say("  smallest b with c = 0 that shows: " + ", ".join(
    f"alpha {a:.4f}: b >= {next(b for b in range(1, 20) if mcnemar1(b, 0) < a)}" for a in (0.05, 0.025, 0.05 / 3)))
say("  H: K1, WIN3_SLOW_KEYHINGE at p1 against WIN3_SLOW at p2 (indep / nested)")
for p2 in (0.75, 0.80, 0.8125, 0.8625, 0.90):
    row = [f"p1 {p1:.3f}: 40 {power(40, p1, p2, False):.3f}/{power(40, p1, p2, True):.3f}, 20 "
           f"{power(20, p1, p2, False):.3f}/{power(20, p1, p2, True):.3f}" for p1 in (0.933, 0.95, 1.0)]
    say(f"    WIN3_SLOW {p2:.4f}: " + "; ".join(row))
for lbl, qb, qc in (("S52 + S85 discordance (12 vs 2 of 60)", 12 / 60, 2 / 60), ("S52 alone (7 vs 2 of 40)", 7 / 40, 2 / 40),
                    ("S85 alone (5 vs 0 of 20)", 5 / 20, 0.0)):
    say(f"    empirical {lbl}: P(K1 SHOWN) at 40 pairs {power_q(40, qb, qc):.3f}, at 20 {power_q(20, qb, qc):.3f}, "
        f"at 80 {power_q(80, qb, qc):.3f}")
for lbl, f in (("indep 0.95 vs 0.8625", lambda n: power(n, 0.95, 0.8625, False)),
               ("nested 0.95 vs 0.8625", lambda n: power(n, 0.95, 0.8625, True)),
               ("empirical 12 vs 2 of 60", lambda n: power_q(n, 0.2, 2 / 60)),
               ("indep 0.95 vs 0.80", lambda n: power(n, 0.95, 0.80, False))):
    need = next((n for n in range(20, 201, 2) if f(n) >= 0.8), None)
    say(f"    pooled pairs for 80% power, {lbl}: {need}")
say("  H: K1 at 80 pooled pairs (40 seeds per machine), indep / nested: " + "; ".join(
    f"{p1} vs {p2}: {power(80, p1, p2, False):.3f}/{power(80, p1, p2, True):.3f}"
    for p2 in (0.8125, 0.8625) for p1 in (0.933, 0.95)))
say("  H: with 40 seeds per machine, the RELIABLE band per machine (>= 36/40): P(on both machines) " + ", ".join(
    f"{p}: {p_ge(40, 36, p) ** 2:.3f}" for p in (0.80, 0.85, 0.90, 0.933, 0.95, 0.975)) + "; the 18/20 rule for comparison: "
    + ", ".join(f"{p}: {p_ge(20, 18, p) ** 2:.3f}" for p in (0.90, 0.933, 0.95, 0.975)))
say("  H's no-room region: K1 cannot be SHOWN when WIN3_SLOW binds >= 36/40 pooled (b <= 4, p >= 0.0625). P(WIN3_SLOW >= "
    "36/40): " + ", ".join(f"{p}: {p_ge(40, 36, p):.3f}" for p in (0.80, 0.8125, 0.8625, 0.90)))

# ══ 6. readings ══════════════════════════════════════════════════════════════
say("\n== 6. Readings: every combination (both machines VALID unless stated)")


def g_read(c1, c2, c3):
    names = []
    if c1 and c2 and c3:
        names.append("the register, released by the key term, discovers the stream")
    if c1 and c2 and not c3:
        names.append("the term and the register discover it; the register alone is not shown worse")
    if c1 and not c2:
        names.append("discovered, but the register is not shown to be needed")
    if not c1:
        names.append("not replicated")
    return names


for c1 in (True, False):
    for c2 in (True, False):
        for c3 in (True, False):
            nm = g_read(c1, c2, c3)
            say(f"  G: C1 {'S' if c1 else 'N'} C2 {'S' if c2 else 'N'} C3 {'S' if c3 else 'N'} -> {nm} "
                f"{'' if len(nm) == 1 else '<- GAP/OVERLAP'}")


def g_rate(x, l):
    if x is None or l is None:
        return "not defined (a machine UNTESTED)"
    return {2: "reliable on both machines", 1: "reliable on one machine", 0: "not reliable"}[x + l]


for x in (True, False, None):
    for l in (True, False, None):
        say(f"  G rate: X {x}, L {l} -> {g_rate(x, l)}")
say(f"  G: KH 40/40 with BASELINE 36/40 gives C1 p = {mcnemar1(4, 0):.4f} -> 'not replicated' printed beside 'reliable on "
    "both machines'")


def h_read(k1, x, l, valid=True):
    if not valid:
        return ["UNTESTED"]
    both = x and l
    names = []
    if k1 and both:
        names.append("the key term makes two streams reliable")
    if k1 and not both:
        names.append("the key term helps, but not reliably")
    if not k1 and both:
        names.append("reliable, but not shown better than WIN3_SLOW")
    if not k1 and not both:
        names.append("not shown")
    return names


for k1 in (True, False):
    for x in (True, False):
        for l in (True, False):
            nm = h_read(k1, x, l)
            say(f"  H: K1 {'S' if k1 else 'N'} K3 X {'MET' if x else 'NOT'} L {'MET' if l else 'NOT'} -> {nm} "
                f"{'' if len(nm) == 1 else '<- GAP/OVERLAP'}")
say("  H: K1 SHOWN with WIN3_SLOW >= 18/20 on both machines is impossible (WIN3_SLOW >= 36/40 leaves b <= 4, p >= "
    f"{mcnemar1(4, 0):.4f}), so 'makes two streams reliable' never fires while the baseline is itself >= 18/20 on both")
say("  H: P('the key term helps, but not reliably' | K1 SHOWN) is P(K3 fails somewhere) = 1 - P(both): "
    + ", ".join(f"{p}: {1 - p_ge(20, 18, p) ** 2:.3f}" for p in (0.90, 0.933, 0.95, 0.975)))

# ══ 7. validity ══════════════════════════════════════════════════════════════
say("\n== 7. Validity: G's REG3_NUDGE must be ROUTED-BOUND on both validity seeds of a machine")
n_obs = nn[0] + 2 + 2
cpl = clopper_pearson(n_obs, n_obs)[0]
for p in (0.99, 0.95, 0.90, round(cpl, 4)):
    one = p * p
    say(f"  nudge rate {p}: P(machine VALID) {one:.3f}; P(both machines VALID) {one * one:.3f}; P(>= 1 of 2 on a machine) "
        f"{1 - (1 - p) ** 2:.4f}")
say(f"  ({n_obs}/{n_obs} nudge runs recorded routed (S72, S82, S83): Clopper-Pearson 95% lower bound {cpl:.3f})")

with open(os.path.join(HERE, "spec_checks_register_keyterm_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
