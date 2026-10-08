#!/usr/bin/env python
"""
explore_h_common.py — EXPLORATORY, not a result. Session H's shared parent-side helpers (no runs): the statistics the
harness defines, outcome extraction, and the per-arm report table. Imports the harness read-only (explore_common,
explore_common16, explore_muon_scale for the channel-map helpers); edits nothing.

STATISTICS (as the harness defines them):
  bands       RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n), MINORITY >= 1, NEVER 0 (test_stream_channels.band at n = 20:
              18 / 10).
  wilson      the 95% Wilson score interval (z = 1.959963984540054), test_stream_recipe.wilson's formula.
  mcnemar1    exact one-sided McNemar on paired seeds: P(Bin(b + c, 1/2) >= b), b = pairs where the first arm alone
              succeeded (test_conv_lr.mcnemar_greater's formula; CHECK against it).
  fisher1     one-sided Fisher exact (test_router_confirm.fisher_greater; used only for unpaired comparisons).
OUTCOMES (test_stream_recipe.outcome, stored in each record by the child): BOUND ROUTED / BOUND NOT routed / MERGED (n share)
  / STREAM-PARTIAL one-to-one / non-stream KEY | POSITION | OTHER. Failure classes are read from the outcome string.
"""

import math
from collections import Counter

import explore_common as ec
import explore_common16 as c16
import explore_muon_scale as s33

Z95 = 1.959963984540054


def band(c, n):
    if n == 0:
        return "--"
    return ("RELIABLE" if c >= math.ceil(0.9 * n) else "MAJORITY" if c >= math.ceil(0.5 * n)
            else "MINORITY" if c >= 1 else "NEVER")


def wilson(k, n, z=Z95):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar1(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def fisher1(a, n1, b, n2):
    N, K = n1 + n2, a + b
    tot = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / tot


def check_stats():
    from test_conv_lr import mcnemar_greater
    from test_router_confirm import fisher_greater
    rows = []
    ok = all(abs(mcnemar1(b, c) - mcnemar_greater(b, c)) < 1e-15 for b in range(12) for c in range(12))
    rows.append(("mcnemar1 equals test_conv_lr.mcnemar_greater on b, c in 0..11", ok))
    ok = all(abs(fisher1(a, 10, b, 10) - fisher_greater(a, 10, b, 10)) < 1e-15 for a in range(11) for b in range(11))
    rows.append(("fisher1 equals test_router_confirm.fisher_greater on a, b in 0..10 of 10", ok))
    lo, hi = wilson(18, 20)
    rows.append((f"wilson(18, 20) = [{lo:.4f}, {hi:.4f}] (the audit: lower bound 0.70)", abs(lo - 0.6990) < 5e-4))
    rows.append((f"bands at n = 20: 18 {band(18, 20)}, 17 {band(17, 20)}, 10 {band(10, 20)}, 9 {band(9, 20)}, 0 {band(0, 20)} "
                 f"(test_stream_channels: 18 / 10)", (band(18, 20), band(17, 20), band(10, 20), band(9, 20), band(0, 20))
                 == ("RELIABLE", "MAJORITY", "MAJORITY", "MINORITY", "NEVER")))
    return rows


def br(r):
    return bool(r is not None and r.get("outcome") == "BOUND ROUTED")


def fclass(r):
    """The failure class of a run that is not BOUND ROUTED: BOUND-NOT-ROUTED, MERGED, STREAM-PARTIAL, KEY, POSITION, OTHER."""
    o = r.get("outcome") or ""
    if o == "BOUND ROUTED":
        return None
    if o.startswith("BOUND NOT"):
        return "BOUND-NOT-ROUTED"
    if o.startswith("MERGED"):
        return "MERGED"
    if o.startswith("STREAM-PARTIAL"):
        return "STREAM-PARTIAL"
    if o.startswith("non-stream "):
        return o.split(" ", 1)[1]
    return o or "?"


def classes(runs):
    return dict(Counter(c for c in (fclass(r) for r in runs.values()) if c))


def acc_at(r, step):
    for c in r.get("curve", []):
        if c[0] == step:
            return c[1]
    return None


def flat_at(r, t, prev, rise=0.02, thr=0.95):
    """(unbound at t, flat at t): the run was still training at t with held-out accuracy < thr there; flat = and it rose
    < rise since the evaluation at prev (the trigger's plateau rule)."""
    a = acc_at(r, t)
    if a is None or a >= thr:
        return False, False
    p = acc_at(r, prev)
    return True, (p is not None and a - p < rise)


def paired(x, y, seeds):
    """{seed: bool} x, y -> (n, x count, y count, b = x only, c = y only, one-sided p that x > y)."""
    ss = [s for s in seeds if s in x and s in y]
    b = sum(1 for s in ss if x[s] and not y[s])
    c = sum(1 for s in ss if y[s] and not x[s])
    return dict(n=len(ss), x=sum(bool(x[s]) for s in ss), y=sum(bool(y[s]) for s in ss), b=b, c=c, p=mcnemar1(b, c),
                p_rev=mcnemar1(c, b))


def count_line(lab, k, n):
    lo, hi = wilson(k, n)
    return f"{lab} {k}/{n} {band(k, n)} [Wilson {lo:.2f}-{hi:.2f}]"


def per_channel(m):
    return s33.per_channel(m)


def map_at(r, t):
    return s33.map_at(r, t)
