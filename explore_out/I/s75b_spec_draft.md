# S75b: S75 at S = 4 on a configuration where the bar is reachable (DRAFT spec, 2026-10-12, EXPLORATORY)

**Status: draft for the user.** Nothing here has run, and no S75b code exists yet. The spec follows the follow-up rule
committed in `explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md` (5bc6c2e):
- fresh seeds;
- every constant derived only from ORACLE and SINGLE quantities (S75's ORACLE 830–831 and SINGLE 830–839, the
  ideal-learner computations, a new 2-seed ORACLE pilot of the exact configuration), never from S75's WIDE_SPLIT, REG3
  or WIDE_SLOW outcomes;
- the configuration changed so that the 0.9 bar is reachable, instead of moving the bar;
- the split guarded at k = S.

**Written blind to the learned arms.** This draft was written and committed before S75's WIDE_SPLIT and REG3 records
were read. Its commit's parent is 5bc6c2e, whose tree holds only the ORACLE and SINGLE records. The learned-arm records
(b649751) were merged after it.

## Question

The question is S75's at S = 4: does the channel partition emerge label-free at four sources under the dense loss? S75
could not answer it, because its ORACLE stopped at 0.892 and 0.880, under the 0.9 bar (UNTESTED).

## Configuration B

ICMC with S = 4, V = 4, blocks of 6–14 words, NB = 32 blocks per source, M = 352 words per source, L = 1536 (the model
reads 1535). Model, B = 32, Adam 1e-3, evaluation and probe exactly as S75. 24,000 updates.

**Why B.** It keeps S75's block structure and doubles each source's history.
- Block lengths, S = 4 and random interleaving are unchanged, so the distribution of gaps between a source's blocks is
  unchanged. That gap is what made S75's first word of a block hard: bin 1 was 0.719 and 0.712 against a bound of 0.943.
- Doubling the history raises the Bayes bound from 0.927 to 0.964.
- The bin shares stay put: 0.088 / 0.091 / 0.182 / 0.638 at B, against 0.086 / 0.091 / 0.183 / 0.640 at S75.

**Ideal learners** (`explore_i_task.ideal_set_acc`, 2 × 128 sequences):

| learner | S75 (L = 768) | B (L = 1536) |
|---|---|---|
| own (the Bayes bound) | 0.927 | 0.964 |
| own_fade (decay 0.95, 58-token horizon) | 0.679 | 0.691 |
| mixed (sources pooled) | 0.496 | 0.517 |

**Predicted ORACLE at B.** Take S75's ORACLE shortfall below the bound in each bin, subtract it from B's bound in that
bin, and weight by B's bin shares. This gives 0.928 (from seed 830) and 0.916 (from seed 831); mean 0.922.
- A second transfer gives a similar figure: S75's share of the faded-to-bound gap that ORACLE closed (0.83), applied to
  B, gives 0.918.
- The prediction assumes the model uses history up to ~1500 tokens back as well as it used ~768. The faded learner
  gains almost nothing from the longer history (0.679 → 0.691), so this is the assumption most at risk.
- In the pessimistic case the model's reach is limited in absolute tokens, and ORACLE stays near S75's 0.886. That is
  why the configuration is piloted before the screen.

**Predicted SINGLE at B.** The mixed learner (0.517) plus S75's SINGLE excess over it (median 0.538 − 0.496 = +0.043)
gives about 0.56, far below ORACLE − 0.2.

**Configurations not chosen.** A, with NB = 24 and L = 1152, is predicted at 0.909, too close to the bar. Longer blocks
(12–28 or 16–32 words, L of about 1200–1350) are predicted at about 0.92. But they change the gap distribution, so their
predictions rest on a shakier transfer.

## Stage 1: ORACLE pilot of B (2 runs)

- **Runs.** ORACLE, seeds 840 and 841, 24,000 updates.
- **Proceed rule, fixed now.** Go to stage 2 if both pilot runs end with SET ≥ 0.912. The margin, 0.012, is S75's ORACLE
  seed spread (0.892 − 0.880), so the screen's own two ORACLE runs should clear 0.9.
- **Stop rule.** Otherwise S75b stops, and the report says that S = 4 is not usable at B. No further configuration is
  tried under this spec; trying one would need a new spec.
- **SPLIT_THR, fixed by rule.** SPLIT_THR := (the lower pilot ORACLE final SET) − 0.03. This is S75's own rule (lowest
  ORACLE final SET − 0.03), now applied to the same configuration's pilot instead of being carried from S = 2.
- **Records.** The pilot's curves, per-bin values, time per update and peak memory are written up, and the stage-1
  record is committed before stage 2 starts.

## Stage 2: the screen (seeds 850–859)

| arm | seeds | what it is | role |
|---|---|---|---|
| ORACLE | 850–851 | perfect gate, k = 4, ADAM | validity |
| SINGLE | 850–859 | one channel, ADAM | the interference baseline |
| WIDE_SPLIT_G | 850–859 | WIDE window gate (width 16), k = 4, SLOW, split with the guard | read |
| REG3 | 850–859 | Session G's register gate at all three layers, k = 4, SLOW, as S75 | read |
| WIDE_SLOW | 850–859 | WIDE_SPLIT without the split | printed, never read |

**The guard.** The split trigger is S75's (`explore_i_common.make_split`: checks every 2400 from 4800 to 21,600; probe
SET < SPLIT_THR and rise < 0.02; at most 3 fires, at least 4800 apart), with one addition. **No fire when the source →
channel map at word positions is one-to-one.** Such a check is recorded as blocked by `one-to-one`. With four sources on
four channels there is nothing to separate, and a fire would overwrite a live channel. This is the hazard S75's note
declared.

**Outcomes, VALID and READING (S75's, unchanged).**
- ROUTED := at every gate layer, η² by source at word positions ≥ 0.9 and a one-to-one map, at the end.
- GAP := (SET(arm) − SET(SINGLE, same seed)) / (mean SET(ORACLE) − SET(SINGLE, same seed)), and by bin.
- VALID := both ORACLE runs (850–851) end with SET ≥ 0.9.
- The reading is "carries to four sources" if the better of WIDE_SPLIT_G and REG3 is ROUTED on ≥ 8/10, and "does not
  carry" otherwise. It is UNTESTED if not VALID.
- **Printed:** ROUTED with Wilson interval and band; GAP and GAP by bin; SET; η² and margin per layer; probe per layer;
  REG3's β; S74's usability count; every split check (fired, blocked by gap, cap or one-to-one); WIDE_SLOW beside
  WIDE_SPLIT_G, paired by seed.

## Where each constant comes from

| constant | value | from (an allowed input) |
|---|---|---|
| S, V, block lengths | 4, 4, 6–14 | S75 (unchanged; registered before any S75 run) |
| NB, M, L | 32, 352, 1536 | the ideal-learner bound at B (0.964) and the ORACLE prediction from S75's ORACLE per-bin shortfalls |
| VALID bar | 0.9 | S74's registered bar (unchanged) |
| pilot proceed margin | 0.012 | S75's ORACLE seed spread |
| SPLIT_THR | lower pilot ORACLE − 0.03 | S75's rule; the pilot's ORACLE |
| the guard | no fire on a one-to-one map | the pre-run note's commitment (written blind to the learned arms) |
| updates, B, lr, evaluation, probe | as S75 | S75 (unchanged) |
| arms, outcomes, reading | as S75, plus the guard; WIDE_SLOW printed | S75's registered design |
| seeds | pilot 840–841; screen 850–859 | fresh, within Session I's 800–859 |

S75's WIDE_SPLIT, REG3 and WIDE_SLOW records feed nothing in this table. If the user changes S75b after reading S75's
descriptions, that change is not blind, and S75b's record must say so.

## Cost (CPU profile of one update at L = 1536; K9 on the GPU measures it)

__COST__

## Code S75b needs (not written; written only after the user approves this draft)

- **A new driver, `explore_gpu/s75b.py`.** It must not import-change S75's modules, so that S75's code SHA stays
  5fb120a92d69. It holds configuration B, the two stages with their rules, the seeds and the guard. The guard is a
  wrapper of `explore_i_common.make_split` that skips `split_op` on a one-to-one `map_before` and records
  `blocked_by = "one-to-one"`.
- **CHECKs before any run.**
  - the task CHECK at B (`explore_i_task.check_task`);
  - the guard's mechanics, C6-style: with threshold 1.01 and rise 1.0, a one-to-one map never fires and a merged map
    does;
  - a lock-step (Judge) of batched against sequential on the GPU at L = 1536 for each S75b arm, 20 updates; the CUDA
    CHECK validated L = 768 only;
  - a K9-style time and memory benchmark at L = 1536.

## Order

1. Write the driver and its CPU CHECKs, then commit.
2. Run the CUDA CHECK at L = 1536.
3. Run the stage-1 pilot, apply its rules, and commit the stage-1 record.
4. Run stage 2.
5. Write the report, counts table first.
