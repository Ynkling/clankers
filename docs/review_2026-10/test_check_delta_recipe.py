#!/usr/bin/env python3
"""test_check_delta_recipe.py: the referee's checks behind test_check_delta_recipe.md (test_delta_recipe.py at a581027).

    python3 docs/review_2026-10/test_check_delta_recipe.py     (stdlib; writes test_check_delta_recipe_out.txt)

1. CHECK 172 reproduced here: the test's copied `overlap` (a581027) against the referee's (recompute.py at 0fdb363,
   the commit CHECK 172 names) and against rlib-free recomputation, on every WIN3_SPLIT_D8 record of results/X and
   results/L window_gate files; the largest overlap over bound-routed runs against recompute.md's 0.047.
2. The reading functions (paired, paired_full, d1_label, split_verdict, split_label, split_reading, newcombe_paired,
   sign_test, d3_of), extracted from the test's source and run on constructed stores, to show each label follows from
   the code's own counts (no number typed in).
"""
import ast
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rlib import load, runs, bound_routed  # noqa: E402

TEST = "a581027"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def src(ref, path):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{ref}:{path}"]).decode()


def extract(source, names, ns):
    tree = ast.parse(source)
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
           (getattr(n, "name", None) in names or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets)))]
    exec(compile(ast.Module(body=fns, type_ignores=[]), "extract", "exec"), ns)
    return ns


tsrc = src(TEST, "test_delta_recipe.py")
say(f"== 1. CHECK 172: the test's overlap (test_delta_recipe.py at {TEST}) against recompute.py's at 0fdb363")
ns_t = extract(tsrc, {"overlap"}, {})
ns_r = extract(src("0fdb363", "docs/review_2026-10/recompute.py"), {"overlap"}, {})
for m in "XL":
    W = load(MAIN, f"results/{m}/window_gate_results.json")
    R = runs(W, "WIN3_SPLIT_D8")
    vals = []
    for s, r in sorted(R.items()):
        if not (r.get("end") or {}).get("stream_gate"):
            continue
        a, b = ns_t["overlap"](r), ns_r["overlap"](r)
        vals.append((s, a, b, bound_routed(r)))
    eq = sum(1 for _, a, b, _ in vals if a == b)
    br = [a for _, a, _, ok in vals if ok]
    say(f"  {m}: {eq}/{len(vals)} records equal (bitwise); largest overlap over all {max(a for _, a, _, _ in vals):.4f}, "
        f"over the {len(br)} bound-routed runs {max(br):.4f}")
    say("     per seed (overlap): " + ", ".join(f"{s} {a:.3f}" for s, a, _, _ in vals))
say("  recompute.md:189: 'The window-gate partitions are clean (overlap at most 0.047)' over the bound-routed runs")

say("\n== 2. The reading functions on constructed stores (extracted from the test's source)")
names = {"paired", "paired_full", "d1_label", "split_verdict", "split_label", "split_reading", "newcombe_paired",
         "sign_test", "d3_of", "D1_LABELS", "SPLIT_READINGS", "NO_ROOM", "D1_BOUND"}


def ok_r(r):
    return bool(r and r.get("ok"))


def mcnemar_greater(b, c):
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


ns = extract(tsrc, names, {"ok_r": ok_r, "mcnemar_greater": mcnemar_greater, "math": math, "statistics": statistics,
                           "tsr": type("tsr", (), {"wilson": staticmethod(wilson)})})


def store(pattern_new, pattern_old):
    """Two arms over 40 seeds from per-seed success flags."""
    seeds = list(range(40))
    new = {s: {"ok": True, "s": f} for s, f in zip(seeds, pattern_new)}
    old = {s: {"ok": True, "s": f} for s, f in zip(seeds, pattern_old)}
    return new, old, seeds


def make(a, b, c, d):
    return ([1] * a + [1] * b + [0] * c + [0] * d, [1] * a + [0] * b + [1] * c + [0] * d)


cases = [("D1", (34, 0, 3, 3)), ("D1", (34, 3, 0, 3)), ("D1", (34, 2, 2, 2)), ("D1", (33, 0, 2, 5)), ("D1", (33, 3, 0, 4))]
for what, (a, b, c, d) in cases:
    pn, po = make(a, b, c, d)
    new, old, seeds = store(pn, po)
    pf = ns["paired_full"](new, old, seeds, lambda r: r["s"], lambda r: r["s"])
    pf["valid"] = True
    nc = ns["newcombe_paired"](pf["a"], pf["b"], pf["c"], pf["dd"])
    say(f"  D1 table a={a} DELTA only={b} HEBB only={c} neither={d}: code counts a={pf['a']} b={pf['b']} c={pf['c']} dd={pf['dd']}; "
        f"d = c - b = {pf['c'] - pf['b']} -> label '{ns['d1_label'](pf)}'; DELTA - HEBB {nc[0]:+.3f} [{nc[1]:+.3f}, {nc[2]:+.3f}]")
for slow, b, c in ((36, 4, 0), (35, 5, 0), (30, 5, 0), (30, 4, 0), (20, 12, 2)):
    split_n = slow + b - c
    pn = [1] * (slow - c) + [1] * b + [0] * c + [0] * (40 - slow - b)
    po = [1] * (slow - c) + [0] * b + [1] * c + [0] * (40 - slow - b)
    new, old, seeds = store(pn, po)
    pd = ns["paired"](new, old, seeds, lambda r: r["s"], lambda r: r["s"])
    pd["valid"] = True
    say(f"  D2/D4 no-trigger {pd['old']}/40, split {pd['new']}/40, {pd['b']} vs {pd['c']}, p = {pd['p']:.4f}: verdict "
        f"{ns['split_verdict'](pd)}, label '{ns['split_label'](pd, pd['old'])}'")
for ld, lh in (("helps", "helps"), ("not shown", "helps"), ("helps", "not shown"), ("not shown", "not shown"),
               ("no room", "helps"), ("no room", "no room")):
    say(f"  split reading delta={ld!r}, Hebbian={lh!r}: {ns['split_reading'](ld, lh)!r}")
rows = [(s, 24000, 14400) for s in range(5)] + [(5, 12000, 12000), (6, 9600, 12000)]
d3 = ns["d3_of"](rows)
say(f"  D3 on 7 constructed rows (5 delta earlier, 1 tie, 1 later): pos {d3['pos']}, neg {d3['neg']}, ties {d3['ties']}, "
    f"p = {d3['p']:.4f} (exact: {sum(math.comb(6, i) for i in range(5, 7)) / 64:.4f}), medians H {d3['med_h']} D {d3['med_d']}")

with open(os.path.join(HERE, "test_check_delta_recipe_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
