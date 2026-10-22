#!/usr/bin/env python
"""
explore_gpu/s75b_rates.py — the numbers behind S75b's pilot proceed rule, for the spec's Appendix C (Referee V's review,
fix-list items 4 and 5). EXPLORATORY; a description: it changes no constant. The driver does not import it, so it is not
in the driver's code SHA.

Model: the two pilot runs, and the two stage-2 ORACLE runs, are independent normal draws with means mu_1, mu_2 and one
seed sd sigma.
  - proceed (the rule, s75b.pilot_decision): mean >= 0.912 and each >= 0.9; the design commit 1e31409 had both >= 0.912;
  - VALID: both stage-2 ORACLE runs >= 0.9.
The means are the spec's ORACLE predictions at B (per-bin transfer 0.928 / 0.916, mean 0.922; fixed reach 0.920 /
0.909, mean 0.914), taken either as one common mean or as two seed-specific means. sigma 0.008 is S75's ORACLE spread
over sqrt(2), from one pair (1 df); 0.004 and 0.012 bracket it.

    python -m explore_gpu.s75b_rates
"""

import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S75 = os.path.join(HERE, "explore_out", "I", "gpu", "s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json")
MEAN, EACH, BAR, OLD = 0.912, 0.9, 0.9, 0.912


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_proceed(m1, m2, s, n=8001):
    """P(X1 >= EACH, X2 >= EACH, X1 + X2 >= 2 MEAN): X1 integrated on a grid of n points over +-8 sigma."""
    lo, hi = m1 - 8 * s, m1 + 8 * s
    dx = (hi - lo) / (n - 1)
    tot = 0.0
    for i in range(n):
        x = lo + i * dx
        if x < EACH:
            continue
        f = math.exp(-0.5 * ((x - m1) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        tot += f * (1.0 - phi((max(EACH, 2 * MEAN - x) - m2) / s)) * dx
    return tot


def p_old(m1, m2, s):
    return (1.0 - phi((OLD - m1) / s)) * (1.0 - phi((OLD - m2) / s))


def p_valid(m1, m2, s):
    return (1.0 - phi((BAR - m1) / s)) * (1.0 - phi((BAR - m2) / s))


def chi2_1_quantile(p):
    """The p-quantile of chi-square with 1 df: (Phi^-1((1 + p) / 2))^2, Phi^-1 by bisection."""
    q = (1.0 + p) / 2.0
    a, b = 0.0, 10.0
    for _ in range(200):
        m = (a + b) / 2.0
        a, b = (m, b) if phi(m) < q else (a, m)
    return ((a + b) / 2.0) ** 2


def main():
    with open(S75) as f:
        st = json.load(f)
    o = sorted(r["end"]["set"] for k, r in st["runs"].items() if k.startswith("ORACLE|"))
    spread = o[1] - o[0]
    sd = spread / math.sqrt(2.0)
    lo95, hi95 = sd * math.sqrt(1.0 / chi2_1_quantile(0.975)), sd * math.sqrt(1.0 / chi2_1_quantile(0.025))
    print("S75b's pilot rule: the numbers behind it (spec Appendix C; a description, no constant changes)")
    print(f"S75's ORACLE 830 / 831 end.set: {o[1]:.5f} / {o[0]:.5f}; spread {spread:.5f}; 0.9 + spread = {0.9 + spread:.4f} "
          f"(the rule's 0.912 rounds each value first: 0.892 - 0.880 = 0.012)")
    print(f"sigma = spread / sqrt(2) = {sd:.5f}, from one pair (1 df): 95% interval {lo95:.4f} to {hi95:.3f}")
    print()
    print("| ORACLE prediction at B | sigma | P(stop), old rule (both >= 0.912) | P(stop), the rule | P(stage-2 VALID) |")
    print("|---|---|---|---|---|")
    for name, m1, m2 in (("per-bin, common mean 0.922", .922, .922), ("per-bin, seed means 0.928 / 0.916", .928, .916),
                         ("fixed reach, common mean 0.914", .914, .914),
                         ("fixed reach, seed means 0.920 / 0.909", .920, .909)):
        for s in (0.004, 0.008, 0.012):
            print(f"| {name} | {s} | {1 - p_old(m1, m2, s):.3f} | {1 - p_proceed(m1, m2, s):.3f} | "
                  f"{p_valid(m1, m2, s):.3f} |")
    print()
    print("| true common mean (sigma 0.008) | P(proceed) | P(stage-2 VALID) |")
    print("|---|---|---|")
    for mu in (0.905, 0.910, 0.914, 0.922):
        print(f"| {mu} | {p_proceed(mu, mu, 0.008):.3f} | {p_valid(mu, mu, 0.008):.3f} |")


if __name__ == "__main__":
    main()
