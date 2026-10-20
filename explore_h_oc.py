#!/usr/bin/env python
"""
explore_h_oc.py — EXPLORATORY, not a result. Session H's exact operating characteristics of a screen's readings (Referee
V's fix 4 for S89, applied to S89b and S89c). No runs, no torch.

Model (V's, screen_review_s89_H.md §5): n paired seeds; arms with true rates p_i; with probability rho a seed has one
shared difficulty u ~ U(0, 1) and arm i succeeds iff u < p_i; otherwise the arms succeed independently. The per-seed
pattern over the arms is then multinomial over the 2^m patterns, and every reading is a function of the pattern counts.
oc(rates, rule, n, rho) enumerates the compositions of n into the 2^m patterns exactly.

Run:  python explore_h_oc.py      (prints the tables quoted in explore_h_s89c.py's and explore_h_s89b.py's docstrings)
"""

import itertools
import math


def pattern_probs(rates, rho):
    m = len(rates)
    pats = list(itertools.product((0, 1), repeat=m))
    ind = {p: math.prod(r if b else 1 - r for r, b in zip(rates, p)) for p in pats}
    cuts = sorted(set([0.0, 1.0] + list(rates)))
    sh = {p: 0.0 for p in pats}
    for lo, hi in zip(cuts, cuts[1:]):
        u = 0.5 * (lo + hi)
        sh[tuple(int(u < r) for r in rates)] += hi - lo
    return pats, [rho * sh[p] + (1 - rho) * ind[p] for p in pats]


def compositions(n, k):
    for c in itertools.combinations(range(n + k - 1), k - 1):
        yield [b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,))]


def oc(rates, rule, n=10, rho=0.0):
    """{reading: probability}; rule(counts) gets {pattern tuple: count} and returns a reading name."""
    pats, pr = pattern_probs(rates, rho)
    out = {}
    lf = [math.lgamma(i + 1) for i in range(n + 1)]
    for c in compositions(n, len(pats)):
        lp = lf[n] - sum(lf[x] for x in c)
        if any(x and q == 0 for x, q in zip(c, pr)):
            continue
        lp += sum(x * math.log(q) for x, q in zip(c, pr) if x)
        name = rule(dict(zip(pats, c)))
        out[name] = out.get(name, 0.0) + math.exp(lp)
    return out


def only(cnt, i, j):
    """Seeds where arm i succeeded and arm j did not."""
    return sum(v for p, v in cnt.items() if p[i] and not p[j])


def total(cnt, i):
    return sum(v for p, v in cnt.items() if p[i])


def rule_s89c(cnt):
    """Arms (BASE, BASE_LATE, BASE_CAP6)."""
    if any(only(cnt, v, 0) >= 4 and only(cnt, 0, v) == 0 for v in (1, 2)):
        return "the schedule was the limit"
    if all(total(cnt, v) < 4 for v in (1, 2)):
        return "the channel is the limit"
    return "no reading applies"


def rule_s89b(cnt):
    """Arms (DELTA_SPLIT, DELTA_ALLMASS, DELTA_NOSPLIT)."""
    if total(cnt, 0) < 7:
        return "the labelled recipe itself fails here"
    if only(cnt, 0, 2) - only(cnt, 2, 0) <= 1:
        return "the split is not needed here"
    if only(cnt, 0, 1) - only(cnt, 1, 0) <= 1 and only(cnt, 1, 2) >= 4 and only(cnt, 2, 1) == 0:
        return "survives"
    if only(cnt, 0, 1) >= 4 and only(cnt, 1, 0) == 0:
        return "needs key positions"
    return "no reading applies"


def table(rows, rule, names, rhos=(0.0, 0.5, 1.0)):
    for lab, rates in rows:
        res = [oc(rates, rule, rho=r) for r in rhos]
        print(f"  {lab}: " + "; ".join(f"{nm} " + "/".join(f"{x.get(nm, 0.0):.2f}" for x in res) for nm in names))


if __name__ == "__main__":
    print("S89c (BASE, BASE_LATE, BASE_CAP6); P(reading) at rho = 0 / 0.5 / 1; both new arms at the stated rate")
    names = ("the schedule was the limit", "the channel is the limit", "no reading applies")
    table([(f"BASE {b}, new arms {v}", (b, v, v)) for b in (0.1, 0.3) for v in (0.1, 0.3, 0.5, 0.7)], rule_s89c, names)
    print("S89c, one new arm at the rate and the other at BASE's")
    table([(f"BASE {b}, one arm {v}", (b, v, b)) for b in (0.1,) for v in (0.3, 0.5, 0.7)], rule_s89c, names)
    print("S89b (DELTA_SPLIT, DELTA_ALLMASS, DELTA_NOSPLIT); P(reading) at rho = 0 / 0.5 / 1")
    names = ("the labelled recipe itself fails here", "the split is not needed here", "survives", "needs key positions",
             "no reading applies")
    table([(f"SPLIT {s}, ALLMASS {t}, NOSPLIT {ns}", (s, t, ns)) for s, ns in ((0.9, 0.9), (0.95, 0.95), (0.9, 0.5))
           for t in (0.9, 0.7, 0.5)], rule_s89b, names)
