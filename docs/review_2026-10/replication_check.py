#!/usr/bin/env python3
"""replication_check.py — the referee's checks behind replication_review.md.

    python3 docs/review_2026-10/replication_check.py            (stdlib; writes replication_check_out.txt)
    <venv with torch>/bin/python docs/review_2026-10/replication_check.py --probe
        (also regenerates the main line's fixed 64-sequence split probe and counts its queries per stream;
         no training, only test_window_gate.probe_data on test_stream_curriculum.TASK48)

1. Session R's counts recomputed from records/phase2.jsonl and phase3.jsonl on claude/replicate-R (pinned 100dced):
   bound from the evaluation curve, bound routed = bound + eight distinct channels in the map + every stream >= 0.9.
2. The binomial arithmetic behind R's reading and its addition 6.
3. How often the main line's split checks read the probe at >= 0.95 while the 2048-query evaluation was below 0.95.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rlib import load, runs, wilson, mcnemar1, transition  # noqa: E402

R = "100dced"
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def jsonl(path):
    txt = subprocess.check_output(["git", "-C", ROOT, "show", f"{R}:{path}"]).decode()
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def binom_cdf(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def r_transition(evals):
    t = None
    for step, acc in reversed(evals):
        if acc >= 0.95:
            t = step
        else:
            break
    return t


def r_bound_routed(r):
    return (r_transition(r["evals"]) is not None and len(set(r["map"])) == len(r["map"]) == 8
            and min(r["per_stream_acc"]) >= 0.9)


say("== 1. Session R's Phase 2 and Phase 3 records (claude/replicate-R at %s)" % R)
p2 = jsonl("records/phase2.jsonl")
by = {}
for r in p2:
    by.setdefault(r["arm"], {})[r["seed"]] = r
for arm, rs in by.items():
    mism = [s for s, r in rs.items() if r_bound_routed(r) != r["bound_routed"] or r_transition(r["evals"]) != r["transition"]]
    c = sum(r_bound_routed(r) for r in rs.values())
    lo, hi = wilson(c, len(rs))
    say(f"  {arm}: bound routed {c}/{len(rs)}, Wilson [{lo:.3f}, {hi:.3f}]; stored flags disagreeing with the recount: {mism}")
sp, sl = by["WIN3_SPLIT_D8"], by["WIN3_SLOW_D8"]
b = sum(1 for s in sp if r_bound_routed(sp[s]) and not r_bound_routed(sl[s]))
c = sum(1 for s in sp if r_bound_routed(sl[s]) and not r_bound_routed(sp[s]))
say(f"  paired SPLIT vs SLOW: {b} vs {c}, exact one-sided p = {mcnemar1(b, c):.5f}")
fails = sorted(s for s, r in sp.items() if not r_bound_routed(r))
say(f"  SPLIT failures: {fails}; maps: " + "; ".join(f"{s}: {sp[s]['map']} acc {sp[s]['final_acc']:.3f}" for s in fails))
nspl = sum(len(r["splits"]) for r in sp.values())
ok = sum(1 for r in sp.values() for x in r["splits"] if x.get("streams_on_c_star", 0) >= 2 and x.get("streams_on_c0", 1) == 0)
say(f"  splits fired: {nspl} (per run {min(len(r['splits']) for r in sp.values())}-{max(len(r['splits']) for r in sp.values())}); "
    f"labelled on target: {ok}")
blind = [(s, st, a, dict(r["evals"]).get(st)) for s, r in sp.items() for st, a in r["probes"]
         if a >= 0.95 and dict(r["evals"]).get(st, 1) < 0.95]
say(f"  checks where R's probe read >= 0.95 while the evaluation was < 0.95: {len(blind)}, seeds {sorted({x[0] for x in blind})}")
med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else (sorted(xs)[len(xs) // 2 - 1] + sorted(xs)[len(xs) // 2]) / 2
for arm in ("WIN3_SPLIT_D8", "WIN3_SLOW_D8"):
    t = [r_transition(r["evals"]) for r in by[arm].values() if r_bound_routed(r)]
    say(f"  median transition over bound-routed {arm}: {med(t):.0f}")
p3 = jsonl("records/phase3.jsonl")
for tag in sorted({r.get("rerun", r.get("arm")) for r in p3}):
    rs = [r for r in p3 if r.get("rerun", r.get("arm")) == tag]
    say(f"  Phase 3 {tag}: bound routed {sum(r_bound_routed(r) for r in rs)}/{len(rs)}")

say("\n== 2. The reading's arithmetic")
for p in (0.95, 0.90, 0.85):
    say(f"  true rate {p}: P(>= 18 of 20) = {1 - binom_cdf(20, 17, p):.3f}; P(<= 17 of 20) = {binom_cdf(20, 17, p):.3f}; "
        f"P(>= 12 of 20) = {1 - binom_cdf(20, 11, p):.4f}")
# R's 17/20 against the main line's 38/40: Fisher exact two-sided
def fisher_two(a, b_, c_, d):
    r1, r2, c1, n = a + b_, c_ + d, a + c_, a + b_ + c_ + d
    pr = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))
say(f"  17/20 (R) against 38/40 (main line): Fisher two-sided p = {fisher_two(17, 3, 38, 2):.3f}; "
    f"pooled X + L + R: 55/60, Wilson [{wilson(55, 60)[0]:.3f}, {wilson(55, 60)[1]:.3f}]")
for n in (20, 40, 60):
    for p in (0.95, 0.85):
        k = math.ceil(0.9 * n)
        say(f"  n = {n}: P(>= {k}, i.e. >= 90%) at true {p}: {1 - binom_cdf(n, k - 1, p):.3f}")

say("\n== 3. The main line's split checks: probe >= 0.95 while the 2048-query evaluation is below 0.95")
for m in "XL":
    W = load(None, f"results/{m}/window_gate_results.json")
    for arm in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        tot = 0
        hits = []
        for s, r in runs(W, arm).items():
            ev = {st: a for st, a, *_ in r["curve"]}
            for ck in r["checks"]:
                tot += 1
                if ck["step"] in ev and ck["acc"] >= 0.95 and ev[ck["step"]] < 0.95:
                    hits.append((s, ck["step"], round(ck["acc"], 3), round(ev[ck["step"]], 3), transition(r["curve"])))
        say(f"  {m} {arm}: {len(hits)} of {tot} checks; (seed, step, probe, eval, transition): {hits}")

say("\n== 5. R's Phase 4 reading (pre-registered at 464be8d, phase4.py:25-29) and calibrated alternatives")
from rlib import wilson as _w  # noqa: E402
xmin = min(x for x in range(41) if _w(x, 40)[1] >= 0.835)
say(f"  'consistent' needs the Wilson upper bound >= 0.835: x >= {xmin}/40; with 17/20 of Phase 2 pooled, "
    f">= {xmin - 17} of the 20 new seeds")
for p in (0.95, 0.85, 0.75, 0.70, 0.65):
    say(f"  true rate {p}: P('a lower rate' | 17/20 kept) = {binom_cdf(20, xmin - 17 - 1, p):.3f}; "
        f"on 40 fresh seeds = {binom_cdf(40, xmin - 1, p):.3f}")
for n in (40, 60):
    k = max(k for k in range(n + 1) if binom_cdf(n, k, 0.835) <= 0.05)
    say(f"  exact one-sided test against 0.835 at n = {n}: 'lower' if x <= {k} (size {binom_cdf(n, k, 0.835):.3f}); "
        f"power at 0.75 {binom_cdf(n, k, 0.75):.3f}, 0.70 {binom_cdf(n, k, 0.70):.3f}, 0.65 {binom_cdf(n, k, 0.65):.3f}")
for alt in (0.85, 0.80):
    for n in range(20, 400):
        ks = [k for k in range(n + 1) if binom_cdf(n, k, 0.95) <= 0.05]
        if ks and binom_cdf(n, ks[-1], alt) >= 0.8:
            say(f"  smallest n telling {alt} from 0.95 (one-sided alpha 0.05, power >= 0.8): n = {n}, "
                f"'below 0.95' if x <= {ks[-1]} (size {binom_cdf(n, ks[-1], 0.95):.3f}, power {binom_cdf(n, ks[-1], alt):.3f})")
            break

if "--probe" in sys.argv:
    import collections
    sys.path.insert(0, ROOT)
    import test_window_gate as twg
    import test_stream_curriculum as tscur
    task = tscur.TASK48
    data = twg.probe_data(task)
    qs = data[:, task.qpos[0] - 1]
    say("\n== 4. The main line's fixed probe (Generator 12345, S=8, P=4): queries per stream "
        + str(dict(sorted(collections.Counter(qs.tolist()).items()))))
else:
    say("\n== 4. (run with --probe and torch to regenerate the main line's probe composition)")

with open(os.path.join(HERE, "replication_check_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
