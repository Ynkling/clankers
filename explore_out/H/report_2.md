# Session H — report 2 (EXPLORATORY, not a result)

## S58b explore_h_eight — S=8, P=4, k=16, conv, 43200 updates, LOCAL3 + SLOW, seeds 460-469

Selected from S58 by the fixed rule: ['RESET', 'GUMBEL_W'] (S58 basis {'GUMBEL_RW': {'br': 5, 'merged': 33, 'trans_med': 6000}, 'GUMBEL_W': {'br': 37, 'merged': 2, 'trans_med': 6000}, 'RESET': {'br': 39, 'merged': 0, 'trans_med': 3600}, 'SWITCH': {'br': 36, 'merged': 0, 'trans_med': 3600.0}}).

| arm | BOUND ROUTED | band | Wilson 95% | BR by 28800 | BOUND (any) | failure classes | MERGED | transition median (all) | BR & 1:1 at KEY | unbound maps at end |
|---|---|---|---|---|---|---|---|---|---|---|
| SPLIT | 10/10 | RELIABLE | 0.72-1.00 | 10 | 10/10 | — | 0 | 21000 (14400, 18000, 18000, 18000, 19200, 22800, 25200, 26400, 26400, 27600) | 10/10 | — |
| RESET | 2/10 | MINORITY | 0.06-0.51 | 1 | 2/10 | MERGED 8 | 8 | 29400 (22800, 36000) | 2/2 | 2+1+1+1+1+1+1 2+1+1+1+1+1+1 2+1+1+1+1+1+1 2+1+1+1+1+1+1 2+1+1+1+1+1+1 2+1+1+1+1+1+1 2+2+1+1+1+1 2+2+1+1+1+1 |
| GUMBEL_W | 0/10 | NEVER | 0.00-0.28 | 0 | 0/10 | MERGED 10 | 10 | -- () | 0/0 | 2+2+2+1+1 2+2+2+1+1 2+2+2+1+1 3+2+1+1+1 3+2+1+1+1 3+2+2+1 3+2+2+1 4+3+1 4+3+1 6+2 |

ORACLE (ceil8_D8 + conv, seeds 460-461): bound 2/2, transitions {460: 3600, 461: 2400} → VALID.

- RESET vs SPLIT: 2/10 vs 10/10; RESET only 0, SPLIT only 8; p(RESET>SPLIT) 1, p(SPLIT>RESET) 0.00391
- GUMBEL_W vs SPLIT: 0/10 vs 10/10; GUMBEL_W only 0, SPLIT only 10; p(GUMBEL_W>SPLIT) 1, p(SPLIT>GUMBEL_W) 0.000977

**Readings (fixed in explore_h_eight's docstring before any run):**

- RESET: **RESET does not carry (SPLIT only 8 - RESET only 0 [<= 1])**
- GUMBEL_W: **GUMBEL_W does not carry (SPLIT only 10 - GUMBEL_W only 0 [<= 1])**
- S58b: **does not carry (better arm: RESET)**

## S60 explore_h_stability — S=2, P=4, k=2, no conv, 24000 updates, LOCAL3 + SLOW, seeds 470-479

| arm | DISCOVERED | band | Wilson 95% | BOUND | bound not disc. | failures | transition median (all) | sat. evals before trans. | at/after | unbound runs | runs sat. before binding | single-final BOUND | runs with flips | tau at end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | 8/10 | MAJORITY | 0.49-0.94 | 8 | 0 | KEY 1, STREAM-PARTIAL 1 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600) | 0/10 | 0/24 | 19/40 | 0 | 0 | 0 | — |
| TEMP_FLOOR | 7/10 | MAJORITY | 0.40-0.89 | 7 | 0 | KEY 1, STREAM-PARTIAL 2 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 4800) | 0/9 | 0/21 | 44/60 | 0 | 0 | 0 | 0.47-0.50 |
| STABLEMAX | 7/10 | MAJORITY | 0.40-0.89 | 7 | 0 | KEY 1, STREAM-PARTIAL 2 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 3600) | 0/8 | 0/21 | 38/60 | 0 | 0 | 0 | — |
| EMA_EVAL | 8/10 | MAJORITY | 0.49-0.94 | 8 | 0 | KEY 1, STREAM-PARTIAL 1 | 3600 (3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600) | 0/16 | 0/24 | 19/40 | 0 | 0 | 0 | — |

ORACLE_CONV (ceiling_conv, seeds 470-471): bound 2/2, transitions {470: 1200, 471: 1200} → VALID; ORACLE_PLAIN (printed): transitions {470: None, 471: None}, final acc {470: 0.31640625, 471: 0.529296875}.

- TEMP_FLOOR vs REF (DISCOVERED): 7/10 vs 8/10; TEMP_FLOOR only 0, REF only 1; p(TEMP_FLOOR>REF) 1, p(REF>TEMP_FLOOR) 0.5
- STABLEMAX vs REF (DISCOVERED): 7/10 vs 8/10; STABLEMAX only 0, REF only 1; p(STABLEMAX>REF) 1, p(REF>STABLEMAX) 0.5
- EMA_EVAL vs REF (DISCOVERED): 8/10 vs 8/10; EMA_EVAL only 0, REF only 0; p(EMA_EVAL>REF) 1, p(REF>EMA_EVAL) 1

**Readings (descriptive, fixed in explore_h_stability's docstring before any run):**

- TEMP_FLOOR: **D1 no change at n = 10 (vs REF 0 vs 1); D2 does not delay saturation (before-transition saturated fraction 0.0 vs REF 0.0)**
- STABLEMAX: **D1 no change at n = 10 (vs REF 0 vs 1); D2 does not delay saturation (before-transition saturated fraction 0.0 vs REF 0.0)**
- EMA_EVAL: **D1 no change at n = 10 (vs REF 0 vs 0); D2 does not delay saturation (before-transition saturated fraction 0.0 vs REF 0.0)**
- D3: **does not shrink (single-final-evaluation bindings + runs with flips: EMA_EVAL 0 + 0, REF 0 + 0); uninformative: REF has none at n = 10**

**H/eight:** 32 runs, 21.4 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 91494b36315e; CHECKs all passed (2026-10-08 10:50:38 UTC, git 6f67a6b+dirty)

**H/stability:** 44 runs, 1.8 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 49ece189ac1a; CHECKs all passed (2026-10-08 10:51:15 UTC, git 6f67a6b+dirty)

**Which readings apply.** Both screens are VALID: the oracles bound 2/2 (S58b at updates 3600 and 2400; S60's
ceiling_conv at 1200).
- **S58b "does not carry".** At eight streams the split bound routed 10/10. The two best S58 arms did not come close:
  RESET 2/10 (8 vs 0 discordant, one-sided p = 0.0039), GUMBEL_W 0/10 (10 vs 0, p = 0.00098).
- **S60: no device changed discovery at n = 10.**
- **D3 (does EMA evaluation shrink the BOUND asymmetry?) is uninformative.** Neither REF nor EMA_EVAL had a run that bound on a single final evaluation, or a run whose accuracy flipped back below 0.95.

**Unexpected / worth a look**
- **At eight streams the row copy is the active part of the split, not the reset.** RESET fired under the same plateau rule
  as SPLIT: 3 operations in every run, every target labelled correct (c* held >= 2 streams, c0 none). It still ended
  MERGED (2 share) on 8/10 seeds. SPLIT bound routed on all 10, by 27600. At four streams (S58) the two arms could not be
  told apart (the split itself did nothing there). The audit 2.3 question is answered for eight streams only: zeroing W_g's
  Adam state alone does not reproduce the split.
- **Write-side Gumbel noise makes eight-stream merges worse.** All 10 GUMBEL_W runs merged, with more sharing than the
  other arms (end maps up to 3+2+2+1, 4+3+1 and 6+2), against 2+1+...+1 for RESET. Noise helped nowhere in this session.
- **The saturation criterion (max prob > 0.99 at > 90% of positions) never fires in a run that binds.** Discovered runs sit
  at a constant 0.65 of probe positions above 0.99 from the transition on, in every arm. The failed runs (KEY,
  STREAM-PARTIAL) are the ones that reach 0.9-1.0. So in these runs saturation marks a committed wrong partition. It is not
  a plateau before binding, and Prieto et al.'s mechanism has no handle here. STABLEMAX starts much softer (0.01-0.38 at
  1200 against REF's 0.52-0.65) but binds at the same updates.
- **TEMP_FLOOR's learned temperature ran straight to the floor.** In every run τ fell to 0.47-0.50 by the first evaluation
  and stayed there. The clamp max(τ, 0.5) cuts the gradient below 0.5, so this arm is in effect "logits x 2". The gate
  wants to be sharper, not softer.
- **EMA_EVAL had the same outcome as REF on every seed (8/10, same seeds).** Its transitions are 1200 later on the 6 seeds
  where REF bound at 2400 (the EMA lags about 1000 updates); training is REF's bit for bit by construction.
- **ORACLE_PLAIN** (the perfect gate without the conv layer) again did not bind this layout (final accuracy 0.32 and 0.53),
  as batch 18 found. Validity rests on ceiling_conv, as fixed in the docstring.

**Overall (exploratory).** Nothing from this session replaces the KEYMASS split. At four streams, k = 16, every arm except
GUMBEL_RW is at the ceiling. At eight streams the split works on its own and its continuous stand-ins do not. What
matters is the row copy (S24's mechanism: the copy makes the busy and idle channels' logits nearly tie). Candidate for the
main line: a pre-registered SPLIT vs RESET at eight streams on disjoint seeds, to confirm "the copy, not the reset".
