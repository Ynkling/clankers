**Which reading applies: R4 "neither device".** No arm beats REF. SPLIT2 vs REF is 1 vs 0, SPLIT2_NOISE vs REF 2 vs 0, and K4
vs REF 3 vs 3; the best one-sided p is 0.25. R1, R2 and R3 do not apply. All four arms sit in the MAJORITY band (15-17/20),
with overlapping Wilson intervals.

**What the copy does at k = 2** (from the split table)
- **The exact copy erases the gate's partition, and the same split comes back.** After W_g[c0] := W_g[c*] the two rows are
  equal, so the gate is uniform: every eta^2 is 0.00 right after the copy. By the next check, 2400 updates later, the KEY
  split had re-formed exactly (KEY/stream eta^2 1.00/0.00) after every firing on seeds 1311, 1313 and 1315 (8 of 8
  firings). On 1317 the half-key pattern (0.50/0.25) re-formed after all 3 firings.
- **Why the same split returns.** The copy leaves W_g's Adam moments untouched. At k = 2 the two identical rows differ only
  in those moments, and the moments still point along the old split. This is our reading of the mechanism; we did not
  measure the moments. The one rescue is 1309, where the gate was not yet a key split at the firing (0.20/0.60): after the
  copy the streams separated (0.00/1.00) and the run discovered.
- **Noise + reset (the main line's form) breaks it more often, but not cleanly.** The same 5 runs fired in both arms. It
  rescued 1309 and 1317 (both DISCOVERED). On 1311 and 1313 it turned a full key split into a partial one (key eta^2
  0.5-0.75, v1 STREAM-PARTIAL, v2 KEY). On 1315 the key split held through 3 firings. So at k = 2 symmetry-breaking helps a
  little (+2 vs REF, +1 vs SPLIT2), well short of R2's 4 vs 0.
- **Spare channels (K4) are not better on these seeds.** K4 is 15/20, the same count as REF, with 3 vs 3 discordant. Its
  failures are STREAM-PARTIAL one-to-one 3 and non-stream KEY 2 under both v1 and v2. S52's 36/40 on seeds 160-199 is the
  same arm on other seeds.
- **REF here: 15/20.** The main line's WIN3_SLOW discovered 69/80 (86%). 15/20 (Wilson 0.53-0.89) is consistent with that.
  The failures are key splits, as the main line reports (v1 KEY 3 + STREAM-PARTIAL 2; v2 KEY 4 + 1).

**Classifier.** On S78's 17 unbound runs v2 differs from v1 on 4: REF 1317, SPLIT2 1317, SPLIT2_NOISE 1311 and 1313. Each is
STREAM-PARTIAL -> KEY, and they are exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25 (the property holds). On
this branch's 29 recorded unbound two-stream runs (the pre-run CHECK), v2 differed on 6, again exactly the predicted ones.
Two of them are S52's W2_K4 MERGED runs (169, 196), which v2 calls non-stream KEY.

**Seeds and setup.** The block moved from 1100-1159 to 1300-1359:
- 1100-1149 are test_router_reliability's trial seeds in results/X and results/L.
- 1200-1259 hold restart-attempt seeds in stream_recipe and curriculum_confirm, and appear in explore-G's logs.
- 1300-1359 appears nowhere in explore_out/ or results/ on any branch.

The other choices:
- REF = S43's LOCAL3_SLOW path, which test_window_gate's CHECK 135 equates with WIN3_SLOW. It reproduced S43's record bit
  for bit through 2400.
- K4 = S52's W2_K4 code, identical to claude/outside-ideas (same blob). It reproduced its record bit for bit through 2400.

The first driver attempt crashed before any run, on a leftover line in my CHECK code; the log is kept as s78_check_crashed.log.

**Take-away (exploratory).** At k = S the KEYMASS split has no idle channel to copy onto. An exact copy only resets the gate
to uniform, and the same key split comes back. The copy works at eight streams because there c0 is empty, so the copy
creates a near-tie between a crowded channel and a free one. Two-stream key splits need another device.
