"""Mechanical reading of Phase 2 records. Independent replication, EXPLORATORY, not a result.

Usage: python analyze.py records/phase2.jsonl [--arm-a WIN3_SPLIT_D8] [--arm-b WIN3_SLOW_D8] [--seeds 900-919]
"""

import argparse
import json
import math
from statistics import median


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_one_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def load(path):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs[(r["arm"], r["seed"])] = r
    return recs


def fmt_count(x, n):
    lo, hi = wilson(x, n)
    return f"{x}/{n} [{lo:.3f}, {hi:.3f}]"


def reading(recs, arm_a="WIN3_SPLIT_D8", arm_b="WIN3_SLOW_D8", seeds=range(900, 920), oracle_seeds=(900, 901)):
    out = []
    oracle = [recs.get(("ORACLE", s)) for s in oracle_seeds]
    oracle_ok = all(r is not None and r["bound"] for r in oracle)
    out.append(f"ORACLE bound: {sum(bool(r and r['bound']) for r in oracle)}/{len(oracle)} "
               f"(transitions {[r and r['transition'] for r in oracle]}) -> validity {'PASS' if oracle_ok else 'FAIL'}")
    pairs = [(recs.get((arm_a, s)), recs.get((arm_b, s))) for s in seeds]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    xa = sum(a["bound_routed"] for a, _ in pairs)
    xb = sum(b["bound_routed"] for _, b in pairs)
    bb = sum(a["bound_routed"] and not b["bound_routed"] for a, b in pairs)
    cc = sum(b["bound_routed"] and not a["bound_routed"] for a, b in pairs)
    p = mcnemar_one_sided(bb, cc)
    for arm, rs in ((arm_a, [a for a, _ in pairs]), (arm_b, [b for _, b in pairs])):
        br = sum(r["bound_routed"] for r in rs)
        bd = sum(r["bound"] for r in rs)
        tr = [r["transition"] for r in rs if r["bound_routed"]]
        out.append(f"{arm}: bound routed {fmt_count(br, n)}; bound {fmt_count(bd, n)}; "
                   f"median transition (bound routed) {median(tr) if tr else None}")
    out.append(f"McNemar {arm_a} vs {arm_b}: {bb} vs {cc}, exact one-sided p = {p:.3g}")
    if xa >= 18 and p < 0.05:
        r = "replicated"
    elif xa >= 12:
        r = "partially replicated"
    else:
        r = "not replicated"
    if n != len(list(seeds)):
        r += f" (INCOMPLETE: {n} pairs)"
    if not oracle_ok:
        r += "  [INVALID: oracle did not bind]"
    out.append(f"READING: {r}")
    return "\n".join(out), dict(xa=xa, xb=xb, b=bb, c=cc, p=p, n=n, reading=r, oracle_ok=oracle_ok)


def table(recs, arms=("ORACLE", "WIN3_SLOW_D8", "WIN3_SPLIT_D8")):
    rows = ["| arm | seed | bound | transition | bound routed | map (stream→channel) | failure | splits (update, c*, c0, streams on c*/c0) | per-stream acc | final acc | updates | wall (min) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for (a, s), r in sorted(recs.items()):
            if a != arm:
                continue
            sp = "; ".join(f"({x['update']}, {x['c_star']}, {x['c0']}, {x['streams_on_c_star']}/{x['streams_on_c0']})" for x in r["splits"]) or "—"
            rows.append(f"| {arm} | {s} | {int(r['bound'])} | {r['transition']} | {int(r['bound_routed'])} | {r['map']} | {r['failure']} | {sp} | "
                        f"{' '.join(f'{v:.2f}' for v in r['per_stream_acc'])} | {r['final_acc']:.3f} | {r['updates']} | {r['wall_s'] / 60:.1f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--arm-a", default="WIN3_SPLIT_D8")
    ap.add_argument("--arm-b", default="WIN3_SLOW_D8")
    ap.add_argument("--seeds", default="900-919")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    recs = load(a.path)
    text, _ = reading(recs, a.arm_a, a.arm_b, range(lo, hi + 1))
    print(text)
    if a.table:
        print()
        print(table(recs))
