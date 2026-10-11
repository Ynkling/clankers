"""Phase 4: the 40-seed count. Independent replication, EXPLORATORY, not a result.

PRE-REGISTERED DESIGN AND READINGS (fixed and committed before any Phase 4 run)
-------------------------------------------------------------------------------
Runs (replicate.py --phase 4, records/phase4.jsonl; Phase 2's records are not touched):
  WIN3_SPLIT_D8_ML  the split as the main line specifies it: the probe is the main line's
                    (BindTask.make_batch(64, Generator(12345)), ported token-identical, the same
                    64 sequences for every seed), c*/c0 from the mean read-gate mass over the
                    32 body key positions only. Everything else is WIN3_SPLIT_D8's. Seeds 920-939.
  WIN3_SLOW_D8      unchanged (it has no split). Seeds 920-939.
  ORACLE_ADAM       perfect gate, single-rate Adam at 1e-3 (the main line's validity arm).
                    Seeds 920-921. Validity: both must bind.

The 40 seeds (as instructed: "on all 40 seeds"):
  split arm = WIN3_SPLIT_D8 on 900-919 (Phase 2, my probe and key positions) together with
              WIN3_SPLIT_D8_ML on 920-939;
  no-split  = WIN3_SLOW_D8 on 900-939 (the same code in both phases, so this pooling is exact).
  The split arm's 40 therefore mixes two probe/key specifications. On 900-909, R1 and R2 each
  changed one of the two and moved nothing (9/10 -> 9/10 both). The 920-939 count alone, which is
  entirely under the main line's specification, is printed beside it, as a secondary count.

Outcome: bound routed (as in Phase 2). Pairing by seed.

READINGS (mechanical, on the 40 seeds):
  "consistent with the report"  if the Wilson 95% interval of the split arm's rate overlaps
                                [0.835, 0.986] (upper >= 0.835 and lower <= 0.986) AND the split arm
                                beats WIN3_SLOW_D8 by exact one-sided McNemar p < 0.05 (b = split
                                only, c = no-split only, p = P[Bin(b+c, 1/2) >= b]);
  "a lower rate"                if the interval's upper bound is below 0.835;
  "neither"                     otherwise (the interval overlaps but p >= 0.05).
  If ORACLE_ADAM does not bind on both seeds, "INVALID (oracle)" is printed beside the reading.

PROBE-VERSUS-EVALUATION BLOCKS (counted on every split run: Phase 2, R1, R2 and Phase 4)
  A check is a probe read at an update u >= 4800 (u a multiple of 2400). It is BLOCKED BY THE
  PROBE if the probe's accuracy is >= 0.95 while the held-out evaluation at the same update is
  < 0.95: the rule's "accuracy < 0.95" fails although the model has not bound (seed 914's mode).
  Counted: such checks; those that were otherwise eligible (fewer than 3 splits so far and >= 4800
  updates since the last split, the probe-side rise condition ignored); runs with >= 1 such check,
  and how many of them ended not bound routed. The converse (probe < 0.95 while evaluation
  >= 0.95) is counted too.

RESULT (filled in after the run; records/phase4.jsonl, records/phase4_table.md)
-------------------------------------------------------------------------------
  ORACLE_ADAM 2/2 (both at 2400) -> validity PASS.
  Split arm, 40 seeds: 35/40, Wilson [0.739, 0.945] (overlaps [0.835, 0.986]);
  WIN3_SLOW_D8 12/40 [0.181, 0.454]; McNemar 23 vs 0, p = 1.19e-7.
  Secondary, 920-939 only (all under the main line's split): WIN3_SPLIT_D8_ML 18/20 [0.699, 0.972];
  WIN3_SLOW_D8 5/20 [0.112, 0.469].
  Probe-versus-evaluation blocks: Phase 4 2 of 172 checks, 0 otherwise eligible, no run affected;
  Phase 2 (per-seed probes) 13 of 192 checks, 12 eligible, in seeds 906 and 914 (914 not bound routed);
  R1 (the fixed probe) 1 of 91, in 905 (bound routed); R2 0 of 90.
READING (mechanical): "consistent with the report".
Independent replication, EXPLORATORY, not a result.
"""

import json
import sys

from analyze import load, mcnemar_one_sided, wilson

REPORT_LO, REPORT_HI = 0.835, 0.986


def probe_blocks(r):
    ev = dict((u, a) for u, a in r["evals"])
    splits = [x["update"] for x in r["splits"]]
    out = dict(blocked=0, eligible=0, converse=0)
    for u, pa in r["probes"]:
        if u < 4800 or u not in ev:
            continue
        before = [s for s in splits if s < u]
        free = len(before) < 3 and (not before or u - before[-1] >= 4800)
        if pa >= 0.95 and ev[u] < 0.95:
            out["blocked"] += 1
            out["eligible"] += int(free)
        if pa < 0.95 and ev[u] >= 0.95:
            out["converse"] += 1
    return out


def blocks_summary(runs):
    tot = dict(blocked=0, eligible=0, converse=0)
    hit, hit_fail = [], []
    for r in runs:
        b = probe_blocks(r)
        for k in tot:
            tot[k] += b[k]
        if b["eligible"]:
            hit.append(r["seed"])
            if not r["bound_routed"]:
                hit_fail.append(r["seed"])
    n_checks = sum(1 for r in runs for u, _ in r["probes"] if u >= 4800)
    return (f"{len(runs)} runs, {n_checks} checks: probe >= 0.95 > evaluation at {tot['blocked']} checks "
            f"({tot['eligible']} otherwise eligible) in runs {hit} (not bound routed: {hit_fail}); "
            f"converse (probe < 0.95 <= evaluation) at {tot['converse']} checks")


def count(rs):
    x, n = sum(r["bound_routed"] for r in rs), len(rs)
    lo, hi = wilson(x, n)
    return x, n, lo, hi


def main(p2="records/phase2.jsonl", p3="records/phase3.jsonl", p4="records/phase4.jsonl"):
    r2, r3, r4 = load(p2), load(p3), load(p4)
    split = [r2[("WIN3_SPLIT_D8", s)] for s in range(900, 920)] + \
            [r4[("WIN3_SPLIT_D8_ML", s)] for s in range(920, 940) if ("WIN3_SPLIT_D8_ML", s) in r4]
    slow = {s: r for (a, s), r in list(r2.items()) + list(r4.items()) if a == "WIN3_SLOW_D8"}
    pairs = [(r, slow[r["seed"]]) for r in split if r["seed"] in slow]
    oracle = [r4.get(("ORACLE_ADAM", s)) for s in (920, 921)]
    ok = all(o is not None and o["bound"] for o in oracle)
    print(f"ORACLE_ADAM: {[o and (o['bound'], o['transition']) for o in oracle]} -> validity {'PASS' if ok else 'FAIL'}")
    xs, n, lo, hi = count([a for a, _ in pairs])
    xb, _, blo, bhi = count([b for _, b in pairs])
    b = sum(a["bound_routed"] and not c["bound_routed"] for a, c in pairs)
    c = sum(c["bound_routed"] and not a["bound_routed"] for a, c in pairs)
    p = mcnemar_one_sided(b, c)
    print(f"split arm (40 seeds): {xs}/{n} [{lo:.3f}, {hi:.3f}]; WIN3_SLOW_D8 {xb}/{n} [{blo:.3f}, {bhi:.3f}]; "
          f"McNemar {b} vs {c}, p = {p:.3g}")
    ml = [r4[("WIN3_SPLIT_D8_ML", s)] for s in range(920, 940) if ("WIN3_SPLIT_D8_ML", s) in r4]
    sl4 = [r4[("WIN3_SLOW_D8", s)] for s in range(920, 940) if ("WIN3_SLOW_D8", s) in r4]
    if ml:
        x, m, l_, h_ = count(ml)
        x2, m2, l2, h2 = count(sl4)
        print(f"secondary, 920-939 only: WIN3_SPLIT_D8_ML {x}/{m} [{l_:.3f}, {h_:.3f}]; WIN3_SLOW_D8 {x2}/{m2} [{l2:.3f}, {h2:.3f}]")
    overlap = hi >= REPORT_LO and lo <= REPORT_HI
    if overlap and p < 0.05:
        reading = "consistent with the report"
    elif hi < REPORT_LO:
        reading = "a lower rate"
    else:
        reading = "neither"
    if n != 40:
        reading += f" (INCOMPLETE: {n} pairs)"
    if not ok:
        reading += "  [INVALID (oracle)]"
    print(f"READING: {reading}")
    print("probe-versus-evaluation blocks:")
    print("  Phase 2 WIN3_SPLIT_D8 (900-919): " + blocks_summary([r2[("WIN3_SPLIT_D8", s)] for s in range(900, 920)]))
    for arm in ("SPLIT_R1_PROBE", "SPLIT_R2_BODYKEYS"):
        rs = [r3[(arm, s)] for s in range(900, 910) if (arm, s) in r3]
        print(f"  Phase 3 {arm} (900-909): " + blocks_summary(rs))
    if ml:
        print("  Phase 4 WIN3_SPLIT_D8_ML (920-939): " + blocks_summary(ml))


if __name__ == "__main__":
    main(*sys.argv[1:])
