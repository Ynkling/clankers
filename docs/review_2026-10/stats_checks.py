#!/usr/bin/env python3
"""stats_checks.py — the referee's statistical checks behind statistics.md (deliverable 4).

python3 docs/review_2026-10/stats_checks.py   -> prints every number statistics.md quotes (stats_checks_out.txt)
Exact computations only (binomial, hypergeometric, enumeration); counts come from the records through rlib.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import load, runs, bound_routed, discovered, bound, transition, pairs, mcnemar1, mcnemar2, fisher1, wilson  # noqa

MAIN = "4a3eebc"
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def binom_pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def fisher_two(a, b, c, d):
    """Two-sided Fisher exact on [[a, b], [c, d]] (sum of tables no more likely than the observed)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9))


W = {m: load(MAIN, f"results/{m}/window_gate_results.json") for m in "XL"}
R = {m: {a: runs(W[m], a) for a in ("WIN3", "WIN3_SLOW", "HINGE0", "WIN3_SLOW16", "WIN16_A", "WIN3_SPLIT_D8",
                                      "WIN3_SLOW_D8", "HINGE_D8")} for m in "XL"}
M = {m: load(MAIN, f"results/{m}/muon_recipe_results.json") for m in "XL"}


def by4800(r):
    return bound(r) and transition(r["curve"]) <= 4800


say("== 1. One-sided vs two-sided, and paired vs unpaired, for the claims of test_window_gate and test_muon_recipe")
for name, a, b, f, rr in (("G1", "WIN3_SLOW16", "WIN16_A", bound_routed, R), ("G2", "WIN3_SPLIT_D8", "HINGE_D8", bound_routed, R),
                          ("G4", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", bound_routed, R)):
    bb = sum(pairs(rr[m][a], rr[m][b], f)[0] for m in "XL")
    cc = sum(pairs(rr[m][a], rr[m][b], f)[1] for m in "XL")
    na = sum(f(r) for m in "XL" for r in rr[m][a].values())
    nb = sum(f(r) for m in "XL" for r in rr[m][b].values())
    n = sum(len(rr[m][a]) for m in "XL")
    say(f"{name}: {na}/{n} vs {nb}/{n}; McNemar {bb} vs {cc}: one-sided {mcnemar1(bb, cc):.3g}, two-sided "
        f"{mcnemar2(bb, cc):.3g}; unpaired Fisher one-sided {fisher1(na, n, nb, n):.3g}")
    # concordance: how much does pairing matter?
    both = sum(1 for m in "XL" for s in rr[m][a] if f(rr[m][a][s]) and f(rr[m][b][s]))
    neither = sum(1 for m in "XL" for s in rr[m][a] if not f(rr[m][a][s]) and not f(rr[m][b][s]))
    say(f"    table: both {both}, {a} only {bb}, {b} only {cc}, neither {neither}")

say("\n== 2. Heterogeneity between machines (does the discordant split differ between X and L?)")
for name, a, b, f, rr in (("G1", "WIN3_SLOW16", "WIN16_A", bound_routed, R), ("G4", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", bound_routed, R),
                          ("G3", "HINGE0", "WIN3_SLOW", discovered, R)):
    x = pairs(rr["X"][a], rr["X"][b], f)
    l_ = pairs(rr["L"][a], rr["L"][b], f)
    say(f"{name}: X {x[0]} vs {x[1]}, L {l_[0]} vs {l_[1]}; Fisher two-sided on the discordant splits p = {fisher_two(x[0], x[1], l_[0], l_[1]):.3f}")
RM = {m: {a: runs(M[m], a) for a in ("WIN16_A", "WIN16_M", "WIN_M", "SLOW_M", "WIN8_A", "WIN8_M")} for m in "XL"}
x = pairs(RM["X"]["WIN16_M"], RM["X"]["WIN16_A"], by4800)
l_ = pairs(RM["L"]["WIN16_M"], RM["L"]["WIN16_A"], by4800)
say(f"M2: X {x[0]} vs {x[1]}, L {l_[0]} vs {l_[1]}; Fisher two-sided on the discordant splits p = {fisher_two(x[0], x[1], l_[0], l_[1]):.3f}")
for a, f in (("WIN3_SLOW_D8", bound_routed), ("WIN16_A", bound_routed), ("WIN3", discovered), ("WIN3_SLOW", discovered),
             ("HINGE0", discovered)):
    cx, nx = sum(f(r) for r in R["X"][a].values()), len(R["X"][a])
    cl, nl = sum(f(r) for r in R["L"][a].values()), len(R["L"][a])
    say(f"  arm {a}: X {cx}/{nx} vs L {cl}/{nl}, Fisher two-sided p = {fisher_two(cx, nx - cx, cl, nl - cl):.3f}")
cx = sum(by4800(r) for r in RM["X"]["WIN16_M"].values())
cl = sum(by4800(r) for r in RM["L"]["WIN16_M"].values())
say(f"  arm WIN16_M bound by 4800: X {cx}/20 vs L {cl}/20, Fisher two-sided p = {fisher_two(cx, 20 - cx, cl, 20 - cl):.3f}")

say("\n== 3. Muon's four-stream speed across all four looks (E2 on X and L shared seeds 340-359; M2 on disjoint seeds)")
ER = {m: load(MAIN, f"results/{m}/early_recipe_results.json") for m in "XL"}
e2 = {m: pairs(runs(ER[m], "WIN16_M"), runs(ER[m], "WIN16_A"), by4800) for m in "XL"}
say(f"E2 X {e2['X'][0]} vs {e2['X'][1]} (n={e2['X'][2]}), E2 L {e2['L'][0]} vs {e2['L'][1]} (n={e2['L'][2]}); "
    f"M2 X {x[0]} vs {x[1]}, M2 L {l_[0]} vs {l_[1]}")
say(f"  L's own two looks (E2 seeds 340-359, M2 seeds 1440-1459): {e2['L'][0] + l_[0]} vs {e2['L'][1] + l_[1]}, "
    f"one-sided p = {mcnemar1(e2['L'][0] + l_[0], e2['L'][1] + l_[1]):.3f}")
say(f"  X's own two looks: {e2['X'][0] + x[0]} vs {e2['X'][1] + x[1]}, one-sided p = {mcnemar1(e2['X'][0] + x[0], e2['X'][1] + x[1]):.4f}")
say(f"  M2 pooled plus E2 on L only (disjoint seeds): {x[0] + l_[0] + e2['L'][0]} vs {x[1] + l_[1] + e2['L'][1]}, one-sided p = "
    f"{mcnemar1(x[0] + l_[0] + e2['L'][0], x[1] + l_[1] + e2['L'][1]):.4f}  (descriptive; not pre-registered)")
# power of M2 to find an effect of E2-on-L's size
nd = x[0] + x[1] + l_[0] + l_[1]
crit = min(b for b in range(nd + 1) if mcnemar1(b, nd - b) < 0.05)
for q in (0.7, 0.75, 0.8, 0.9):
    pw = sum(binom_pmf(nd, b, q) for b in range(crit, nd + 1))
    say(f"  power of M2's pooled test with {nd} discordant pairs (reject if b >= {crit}) when P(Muon-only | discordant) = {q}: {pw:.2f}")

say("\n== 4. G3's non-inferiority rule 'd = (HINGE0 only) - (WIN3_SLOW only) <= 2': operating characteristics")
bH = sum(pairs(R[m]["HINGE0"], R[m]["WIN3_SLOW"], discovered)[0] for m in "XL")
bW = sum(pairs(R[m]["HINGE0"], R[m]["WIN3_SLOW"], discovered)[1] for m in "XL")
nd3 = bH + bW
N3 = 80
say(f"observed: HINGE0 only {bH}, WIN3_SLOW only {bW}, discordant {nd3} of {N3} pairs")
lo, hi = wilson(bW, nd3)
say(f"  paired difference in success rate (WIN3_SLOW - HINGE0) = {(bW - bH) / N3:+.3f}; conditional 95% interval from the "
    f"Wilson interval of the discordant split [{lo:.3f}, {hi:.3f}]: [{(2 * lo - 1) * nd3 / N3:+.3f}, {(2 * hi - 1) * nd3 / N3:+.3f}]")
for q in (0.5, 0.4, 0.3, 0.25):
    # q = P(WIN3_SLOW only | discordant); d = c - b where c = HINGE0-only ~ Bin(nd, 1-q)
    hold = sum(binom_pmf(nd3, w, q) for w in range(nd3 + 1) if (nd3 - w) - w <= 2)
    say(f"  with {nd3} discordant pairs, P(rule HOLDS) if P(WIN3_SLOW only | discordant) = {q} "
        f"(true difference {(2 * q - 1) * nd3 / N3:+.3f}): {hold:.2f}")
for nd_ in (10, 20, 40):
    hold = sum(binom_pmf(nd_, w, 0.5) for w in range(nd_ + 1) if (nd_ - w) - w <= 2)
    say(f"  at exact equality with {nd_} discordant pairs, P(rule FAILS) = {1 - hold:.2f}")

say("\n== 5. Bands and Wilson lower bounds")
for c, n in ((38, 40), (18, 20), (20, 20), (19, 20), (36, 40), (9, 10), (10, 10)):
    lo, hi = wilson(c, n)
    say(f"  {c}/{n}: Wilson 95% [{lo:.3f}, {hi:.3f}]; one-sided exact (Clopper-Pearson) 95% lower bound "
        f"{next(p / 10000 for p in range(10000, -1, -1) if sum(binom_pmf(n, k, p / 10000) for k in range(c, n + 1)) <= 0.05):.3f}")
say("  (test_muon_recipe's reading required a Wilson lower bound >= 0.80; test_window_gate's reading only the band)")

say("\n== 6. Multiplicity")
for k in (60, 120):
    say(f"  Bonferroni over {k} claims: G1 pooled p 0.0037 -> {min(1, 0.0037 * k):.2f}; G4 1.2e-7 -> {1.2e-7 * k:.1e}; "
        f"G2 3.6e-12 -> {3.6e-12 * k:.1e}; M1 0.00098 -> {min(1, 0.00098 * k):.3f}")
say(f"  expected false positives among 60 one-sided claims at alpha 0.05 if all nulls true: {60 * 0.05:.1f}; "
    f"probability that a given null passes on both machines independently: {0.05 ** 2:.4f}; expected among 60: {60 * 0.05 ** 2:.2f}")
holm = sorted((("G2", 3.638e-12), ("G4", 1.192e-7), ("G1", 0.003693)), key=lambda t: t[1])
say("  Holm over the three pooled window-gate claims (G1, G2, G4): " + ", ".join(f"{n_} {min(1, p * (3 - i)):.2g}" for i, (n_, p) in enumerate(holm)))

say("\n== 7. The 'scale-free' rule (batch 19: d = arm - reference >= -1 over 10 paired seeds)")
# independence approximation: arm and reference each bind with their own rates, per seed independently
for pr, pa in ((0.9, 0.9), (0.9, 0.7), (0.9, 0.6), (0.95, 0.75)):
    p_hold = 0.0
    for b in range(11):
        for c in range(11 - b):
            # b = arm only, c = ref only, under independent per-seed outcomes
            pass
    # enumerate per-seed joint outcomes: arm-only q1 = pa(1-pr), ref-only q2 = pr(1-pa)
    q1, q2 = pa * (1 - pr), pr * (1 - pa)
    q0 = 1 - q1 - q2
    p_hold = sum(math.factorial(10) / (math.factorial(b) * math.factorial(c) * math.factorial(10 - b - c))
                 * q1 ** b * q2 ** c * q0 ** (10 - b - c) for b in range(11) for c in range(11 - b) if b - c >= -1)
    say(f"  reference rate {pr}, arm rate {pa}: P(d >= -1, 'scale-free') = {p_hold:.2f}")

say("\n== 8. Copy vs reset, pooled over the two ten-seed eight-stream screens (not pre-registered)")
say(f"  S50 split-only 8 / reset-only 1, S58b 8 / 0 -> 16 vs 1: one-sided p = {mcnemar1(16, 1):.2g}, two-sided {mcnemar2(16, 1):.2g}")
say(f"  the window_gate control seeds (same test, same trigger): split vs reset 7 vs 0, one-sided p = {mcnemar1(7, 0):.4f}")
say(f"  S61 (uncited): COPY 9/10 vs SPLIT 9/10; the 'reset adds nothing' comparison has at most a handful of discordant pairs")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_checks_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
