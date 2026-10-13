#!/usr/bin/env python3
"""spec_checks_delta.py: the referee's checks behind spec_review_test_delta_recipe.md (stdlib; reads git only).

    python3 docs/review_2026-10/spec_checks_delta.py      (writes spec_checks_delta_out.txt)

1. Facts the draft cites: the CHECK 158 seeds' first splits; F's S76 arms (is there a DELTA_SLOW?); H's COPY_NONOISE|487.
2. Power at the planned 40 pooled pairs:
   D1 ('HOLDS if d = HEBB only - DELTA only <= 2') and D2 (exact one-sided McNemar, SHOWN if p < 0.05),
   under two pairing extremes: independent outcomes per seed (most discordance) and nested outcomes
   (the better arm binds on every seed the worse one binds; least discordance).
"""
import collections
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rlib import load, runs, bound_routed, mcnemar1  # noqa: E402

out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
say("== 1. Facts the draft cites")
for m, rng in (("X", range(540, 560)), ("L", range(1540, 1560))):
    R = runs(load(MAIN, f"results/{m}/window_gate_results.json"), "WIN3_SPLIT_D8")
    first = [(s, next((c["step"] for c in R[s]["checks"] if c.get("fired")), None)) for s in rng if s in R]
    say(f"  {m} WIN3_SPLIT_D8 first firing by seed (Part D): {first[:4]}; CPU {R[rng[0]].get('cpu')}")
E = load("5b9bc2a", "explore_out/F/eight_results.json")
arms = collections.Counter(k.split("|")[0] for k in E["runs"])
say(f"  F S76 arms at 5b9bc2a: {dict(arms)}  (DELTA_SLOW present: {'DELTA_SLOW' in arms})")
for a in ("HEBB_SPLIT", "DELTA_SPLIT"):
    rs = runs(E, a)
    say(f"  F S76 {a}: bound routed {sum(bound_routed(r) for r in rs.values())}/{len(rs)}")
H = load("7e9197f", "explore_out/H/copy2x2_results.json")
r = runs(H, "COPY_NONOISE")[487]
say(f"  H COPY_NONOISE|487: firings {[c['step'] for c in r['checks'] if c.get('fired')]}; CPU {r.get('cpu')}")


def multinom_d(n, qa, qb):
    """distribution of (A only, B only) counts over n pairs; returns P over (a, b)."""
    qn = 1 - qa - qb
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield a, b, math.comb(n, a) * math.comb(n - a, b) * qa ** a * qb ** b * qn ** (n - a - b)


def p_holds(n, ph, pd, nested):
    if nested:
        qh, qd = max(ph - pd, 0), max(pd - ph, 0)
    else:
        qh, qd = ph * (1 - pd), pd * (1 - ph)
    return sum(p for h, d_, p in multinom_d(n, qh, qd) if h - d_ <= 2)


def p_shown(n, ps, pslow, nested):
    if nested:
        qs, ql = max(ps - pslow, 0), max(pslow - ps, 0)
    else:
        qs, ql = ps * (1 - pslow), pslow * (1 - ps)
    return sum(p for b, c, p in multinom_d(n, qs, ql) if mcnemar1(b, c) < 0.05)


say("\n== 2. D1: P(HOLDS) = P(HEBB only - DELTA only <= 2) over 40 pairs, HEBB_SPLIT at 0.95")
for pd in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
    say(f"  DELTA_SPLIT {pd:.2f}: independent {p_holds(40, 0.95, pd, False):.3f}; nested {p_holds(40, 0.95, pd, True):.3f}")
say("  per machine (20 pairs, same margin 2), HEBB 0.95: " + ", ".join(
    f"DELTA {pd:.2f}: {p_holds(20, 0.95, pd, False):.3f}/{p_holds(20, 0.95, pd, True):.3f}" for pd in (0.95, 0.85, 0.75)))

say("\n== 3. D2: P(SHOWN) over 40 pairs, DELTA_SPLIT at 0.90 (S76: 9/10); DELTA_SLOW unknown (no S76 arm)")
for pl in (0.30, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90):
    say(f"  DELTA_SLOW {pl:.2f}: independent {p_shown(40, 0.90, pl, False):.3f}; nested {p_shown(40, 0.90, pl, True):.3f}")
say("  smallest nested discordance that can show: b >= 5 with c = 0 (p = 1/32); b = 4, c = 0 gives p = 0.0625")

say("\n== 4. What a FAIL can rest on: d = 3 with no DELTA-only pair is 3 vs 0, one-sided McNemar p = "
    f"{mcnemar1(3, 0):.3f}; 4 vs 1 p = {mcnemar1(4, 1):.3f}")


say("\n== 5. A calibrated D1: threshold t on d at n pairs, P(HOLDS) at a 10-point deficit (0.95 vs 0.85) and at equal 0.95")


def p_le(n, ph, pd, t, nested):
    qh, qd = (max(ph - pd, 0), max(pd - ph, 0)) if nested else (ph * (1 - pd), pd * (1 - ph))
    return sum(p for h, d_, p in multinom_d(n, qh, qd) if h - d_ <= t)


for n in (40, 60, 80):
    for t in (-1, 0, 1, 2):
        say(f"  n = {n}, HOLDS if d <= {t:2d}: false HOLD at 0.85 indep {p_le(n, .95, .85, t, False):.3f} / nested "
            f"{p_le(n, .95, .85, t, True):.3f}; HOLDS at equal 0.95 indep {p_le(n, .95, .95, t, False):.3f}")

with open(os.path.join(HERE, "spec_checks_delta_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
