"""Phase 5: the split's copy against its reset, and one delta-rule channel. Independent replication, EXPLORATORY,
not a result.

What is replicated: the main line's test_split_copy (specs/test_split_copy.md and the docstring of
test_split_copy.py at ffb0771; the code below that docstring was not read). From its pooled result:
  C1  WIN3_SPLIT_D8 18/20 BOUND ROUTED against WIN3_RESET_D8 6/20 (12 vs 0);
  C3  WIN3_SPLIT_D8 18/20 against SINGLE_DELTA_D8 0/20 BOUND (18 vs 0).
Not replicated here: C2 (RESET against no trigger) and SINGLE_HEBB_D8. Details I had to guess are in UNSTATED.md,
"Phase 5" (U-50 to U-70).

PROTOCOL (fixed and committed before any Phase 5 run)
-------------------------------------------------------------------------------
Runs (replicate.py --phase 5, records/phase5.jsonl). Seeds 940-959 for every arm, paired by seed; the validity
oracles on 940-941. Phases 2-4 records are not touched.
  COPY          = WIN3_SPLIT_D8_ML, my Phase 4 recipe unchanged: the window gate + SLOW, and the main line's trigger
                  and copy: the probe from Generator(12345), c*/c0 from body key positions, checks every 2400 from
                  4800 (2400 the reference), fire if probe acc < 0.95 and rise < 0.02, copy W_g[c*] onto W_g[c0]
                  with noise 0.1 std on both, W_g's Adam state zeroed, <= 3 splits >= 4800 apart.
  RESET         = RESET_D8_ML: the same trigger on its own run; where it fires, W_g's Adam state is zeroed and W_g is
                  left untouched (no copy).
  DELTA1        = one channel (k = 1, no gate) with the delta-rule write at the spec's fixed strength: beta = 1, L2
                  keys, tied write, decay 0.95; conv; the same N, D, layers; single-rate Adam 1e-3 on every
                  parameter throughout.
  ORACLE_ADAM   = the perfect gate, Hebbian, k = 16, single-rate Adam (validity for COPY and RESET).
  ORACLE_DELTA  = the perfect gate on delta-rule channels, k = 16, single-rate Adam (validity for DELTA1).
Outcomes: COPY and RESET BOUND ROUTED; DELTA1 BOUND (k = 1 has no routing). Budget 43,200; evaluation and early
stop as in Phases 2-4.

READING RULE (the amended rule as stated by the user for this phase):
  Per arm: "consistent" if my count's rate x/20 lies in the main line's pooled Wilson 95% interval for that arm
  (endpoints included); otherwise "outside the main line's interval". No "replicates" label is used.
    COPY   (main line WIN3_SPLIT_D8 18/20):   [0.699, 0.972]  -> consistent iff 14 <= x <= 19
    RESET  (main line WIN3_RESET_D8 6/20):    [0.145, 0.519]  -> consistent iff  3 <= x <= 10
    DELTA1 (main line SINGLE_DELTA_D8 0/20):  [0.000, 0.161]  -> consistent iff  0 <= x <= 3
  (A 20/20 copy count would read "outside", since 1.0 > 0.972. The rule is applied as written.)
  The discordant counts are printed, not read: COPY vs RESET (b = COPY only, c = RESET only, BOUND ROUTED), and COPY
  (BOUND ROUTED) vs DELTA1 (BOUND), with the exact one-sided McNemar p labelled descriptive.
  Validity: if ORACLE_ADAM does not bind on both seeds, COPY's and RESET's labels are printed as INVALID; if
  ORACLE_DELTA does not bind on both seeds, DELTA1's label is UNTESTED, not a negative (the spec's rule).
  Note: the "Phase 4, amended" text in docs/reading/parallel_sessions_prompts.md at ffb0771 states a different rule
  (a fresh-seed count x <= 27/32 reads "below the report's 0.95"). The rule above is the one the user stated for
  this phase. Both are recorded here; only this one is applied.
Also printed: every split and reset (update, c*, c0, streams on c*/c0 by the labels); the probe-versus-evaluation
blocks (phase4.probe_blocks); DELTA1's final accuracy by stream; Wilson intervals for every arm.

RESULT (filled in after the run)
-------------------------------------------------------------------------------
(not yet run)
"""

import sys

from analyze import load, mcnemar_one_sided, wilson
from phase4 import blocks_summary

SEEDS = range(940, 960)
MAIN = {"COPY": (18, 20, 0.699, 0.972), "RESET": (6, 20, 0.145, 0.519), "DELTA1": (0, 20, 0.0, 0.161)}
ARM = {"COPY": "WIN3_SPLIT_D8_ML", "RESET": "RESET_D8_ML", "DELTA1": "DELTA1"}


def success(name, r):
    return bool(r["bound"]) if name == "DELTA1" else bool(r["bound_routed"])


def main(path="records/phase5.jsonl"):
    recs = load(path)
    ok_h = all(recs.get(("ORACLE_ADAM", s), {}).get("bound") for s in (940, 941))
    ok_d = all(recs.get(("ORACLE_DELTA", s), {}).get("bound") for s in (940, 941))
    for o in ("ORACLE_ADAM", "ORACLE_DELTA"):
        print(o, [(s, recs[(o, s)]["bound"], recs[(o, s)]["transition"]) for s in (940, 941) if (o, s) in recs])
    runs = {n: {s: recs[(a, s)] for s in SEEDS if (a, s) in recs} for n, a in ARM.items()}
    for name, rs in runs.items():
        x, n = sum(success(name, r) for r in rs.values()), len(rs)
        lo, hi = wilson(x, n)
        mx, mn, mlo, mhi = MAIN[name]
        label = "consistent" if n and mlo <= x / n <= mhi else "outside the main line's interval"
        if n != 20:
            label += f" (INCOMPLETE: {n} runs)"
        if name in ("COPY", "RESET") and not ok_h:
            label += "  [INVALID (ORACLE_ADAM)]"
        if name == "DELTA1" and not ok_d:
            label = "UNTESTED (ORACLE_DELTA did not bind on both seeds)"
        print(f"{name:6s} {x}/{n} [{lo:.3f}, {hi:.3f}]   main line {mx}/{mn} [{mlo:.3f}, {mhi:.3f}]   -> {label}")
    for a, b in (("COPY", "RESET"), ("COPY", "DELTA1")):
        common = sorted(set(runs[a]) & set(runs[b]))
        bb = sum(success(a, runs[a][s]) and not success(b, runs[b][s]) for s in common)
        cc = sum(success(b, runs[b][s]) and not success(a, runs[a][s]) for s in common)
        print(f"discordant {a} vs {b}: {bb} vs {cc} over {len(common)} pairs "
              f"(exact one-sided McNemar p = {mcnemar_one_sided(bb, cc):.3g}, descriptive)")
    for name in ("COPY", "RESET"):
        rs = list(runs[name].values())
        ev = [x for r in rs for x in r["splits"]]
        ok = sum(1 for x in ev if x["streams_on_c_star"] >= 2 and x["streams_on_c0"] == 0)
        print(f"{name} trigger fired {len(ev)} times in {len(rs)} runs; targets labelled ok {ok}/{len(ev)}")
        print(f"  probe-versus-evaluation blocks: {blocks_summary(rs)}")
    for s, r in sorted(runs["DELTA1"].items()):
        print(f"DELTA1 {s}: bound {r['bound']} T {r['transition']} final {r['final_acc']:.3f} "
              f"per stream {' '.join(f'{v:.2f}' for v in r['per_stream_acc'])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
