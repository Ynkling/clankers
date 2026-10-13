#!/usr/bin/env python3
"""spec_checks_delta2.py: the referee's checks behind spec_review_test_delta_recipe_2.md (Amendment 1, 32c121b).

    python3 docs/review_2026-10/spec_checks_delta2.py      (stdlib; writes spec_checks_delta2_out.txt)

1. The no-room rule: the smallest one-sided McNemar p any outcome can reach when the no-trigger arm binds s of 40.
2. D2 / D4 power around the threshold (split arm at 0.95), and how often each label comes out.
3. D1's three labels: how often each fires under equal and unequal rates (independent and nested pairing).
4. Newcombe's hybrid score interval for paired proportions (1998, method 10), written here from the formula, against
   test_delta_recipe.newcombe_paired as committed on the main line (a581027), on every 2x2 table with n = 40 and on
   n = 20, 80; and on the referee's own G3 figures as a sanity check.
"""
import math
import os
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


def multinom(n, qa, qb):
    qn = 1 - qa - qb
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield a, b, math.comb(n, a) * math.comb(n - a, b) * qa ** a * qb ** b * qn ** (n - a - b)


say("== 1. No-room rule (Amendment 1: a no-trigger arm at >= 36 of the pooled 40 makes the split claim NO ROOM)")
for s in range(32, 41):
    bmax = 40 - s
    say(f"  no-trigger arm {s}/40: largest possible b (split only) = {bmax}, smallest p = {mcnemar1(bmax, 0):.4f} "
        f"-> SHOWN {'possible' if mcnemar1(bmax, 0) < 0.05 else 'impossible'}")


def labels(n, ps, pl, nested):
    qs, ql = (max(ps - pl, 0), max(pl - ps, 0)) if nested else (ps * (1 - pl), pl * (1 - ps))
    # the slow arm's count is needed for no-room: approximate by its binomial margin, independent of (b, c)? No: compute
    # exactly by enumerating the four cells.
    pa = ps * pl if not nested else min(ps, pl)       # both bind
    pd = 1 - ps - pl + pa                               # neither
    res = {"helps": 0.0, "no room": 0.0, "not shown": 0.0}
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                pr = math.factorial(n) / (math.factorial(a) * math.factorial(b) * math.factorial(c) * math.factorial(d)) \
                    * pa ** a * qs ** b * ql ** c * pd ** d if min(pa, qs, ql, pd) >= 0 else 0
                slow = a + c
                lab = "no room" if slow >= 36 else ("helps" if mcnemar1(b, c) < 0.05 else "not shown")
                res[lab] += pr
    return res


say("\n== 2. D2 / D4 labels over 40 pairs, split arm at 0.95 (independent | nested pairing)")
for pl in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    ri, rn = labels(40, 0.95, pl, False), labels(40, 0.95, pl, True)
    say(f"  no-trigger arm {pl:.2f}: indep " + ", ".join(f"{k} {v:.3f}" for k, v in ri.items())
        + " | nested " + ", ".join(f"{k} {v:.3f}" for k, v in rn.items()))

say("\n== 3. D1 labels over 40 pairs (d = HEBB only - DELTA only): P(d <= -3), P(-2 <= d <= 2), P(d >= 3)")
for ph, pd in ((0.95, 0.95), (0.95, 0.90), (0.95, 0.85), (0.90, 0.95), (0.85, 0.95), (0.90, 0.90)):
    for nested in (False, True):
        qh, qd = (max(ph - pd, 0), max(pd - ph, 0)) if nested else (ph * (1 - pd), pd * (1 - ph))
        lo = mid = hi = 0.0
        for h, dl, p in multinom(40, qh, qd):
            dd = h - dl
            if dd <= -3:
                lo += p
            elif dd <= 2:
                mid += p
            else:
                hi += p
        say(f"  HEBB {ph:.2f}, DELTA {pd:.2f}, {'nested' if nested else 'indep '}: 'delta binds more often' {lo:.3f}; "
            f"'not shown worse by > 2' {mid:.3f}; 'not shown non-inferior' {hi:.3f}")
say(f"  the label's own evidence: d = -3 as 0 vs 3 gives one-sided p = {mcnemar1(3, 0):.3f}; d = 3 as 3 vs 0 the same")


def wilson_z(x, n, z):
    if n == 0:
        return 0.0, 1.0
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe10(a, b, c, d, z=1.959963984540054):
    """Newcombe (1998) method 10 for theta = p1 - p2, p1 = (a+b)/n, p2 = (a+c)/n; Wilson margins; phi corrected for
    continuity when ad > bc (max(ad - bc - n/2, 0)), as Fagerland, Lydersen and Laake (2014) state it."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_z(a + b, n, z)
    l2, u2 = wilson_z(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2, 0)
        phi = num / math.sqrt(den)
    th = p1 - p2
    L = th - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    U = th + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return th, max(-1.0, L), min(1.0, U)


say("\n== 4. Newcombe method 10: this file's implementation against test_delta_recipe.newcombe_paired (main line a581027)")
src = subprocess.check_output(["git", "-C", ROOT, "show", "a581027:test_delta_recipe.py"]).decode()
start = src.index("def newcombe_paired(")
end = src.index("\ndef ", start + 10)
ns = {"math": math, "tsr": type("tsr", (), {"wilson": staticmethod(lambda x, n, z=1.959963984540054: wilson_z(x, n, z))})}
exec(src[start:end], ns)
theirs = ns["newcombe_paired"]
worst = 0.0
tables = 0
for n in (20, 40, 80):
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                m, t = newcombe10(a, b, c, d), theirs(a, b, c, d)
                worst = max(worst, max(abs(x - y) for x, y in zip(m, t)))
                tables += 1
say(f"  {tables} tables (n = 20, 40, 80): largest absolute difference {worst:.2e}")
# the code's tsr.wilson must be the plain Wilson score interval: compare with rlib.wilson (z = 1.959964)
diffw = max(abs(x - y) for n in (20, 40) for k in range(n + 1) for x, y in zip(wilson(k, n), wilson_z(k, n, 1.959964)))
say(f"  rlib.wilson against the Wilson formula used here: largest difference {diffw:.2e}")
for tab in ((36, 2, 0, 2), (34, 4, 1, 1), (30, 0, 3, 7), (38, 0, 0, 2), (25, 8, 3, 4)):
    th, lo, hi = newcombe10(*tab)
    say(f"  a, b, c, d = {tab}: difference {th:+.3f}, 95% [{lo:+.3f}, {hi:+.3f}]")
# sanity: the referee's G3 (window_gate Part A, 80 pairs: WIN3_SLOW only 11, HINGE0 only 9; both 58 (69 - 11), neither 2)
th, lo, hi = newcombe10(58, 11, 9, 2)
say(f"  G3 (WIN3_SLOW vs HINGE0, 80 pairs, 11 vs 9): {th:+.3f} [{lo:+.3f}, {hi:+.3f}] "
    f"(statistics.md's conditional interval [-0.079, +0.121])")

with open(os.path.join(HERE, "spec_checks_delta2_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
