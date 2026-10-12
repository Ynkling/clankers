# S75 at S = 4: pre-run note, 2026-10-12 (EXPLORATORY)

This note was written after the ORACLE and SINGLE runs on the GTX 1660 SUPER, and committed on `claude/explore-I` before
any WIDE_SPLIT or REG3 run. **It is not an amendment.** `explore_gpu/s75_s4.py` and every module it imports are
unchanged, so the code SHA is still 5fb120a92d69, the SHA of the ORACLE and SINGLE records. The RTX 3070's log must show
that SHA at start.

## Decision: no amendment

The following stay as committed in 5d9dcf9 (2026-10-10): VALID (both ORACLE runs end with SET ≥ 0.9), the reading rule,
24,000 updates, the seeds, the arms, SPLIT_THR = 0.88, and the code.

- **The registered result is UNTESTED.** Any report states this first.
- **ROUTED counts are descriptions only.** The ROUTED counts (with Wilson intervals), η², margin, GAP and GAP by bin,
  and probe accuracy are printed without a "carries" or "does not carry" label, whatever the counts.
- **No later re-reading.** S75's learned-arm runs are never re-read under a later bar.

## What came in (GTX 1660 SUPER; the CUDA CHECK passed)

| arm | seeds | final SET | note |
|---|---|---|---|
| ORACLE | 830, 831 | 0.892, 0.880 | no evaluation of either ever reached 0.9 (bests 0.892 at 24k, 0.881 at 21.6k) |
| SINGLE | 830–839 | 0.534–0.540 on 9 seeds, 0.595 on 838 | SINGLE ≤ ORACLE − 0.2 on 10/10 (smallest margin 0.29) |

So the SINGLE half of S74's usability rule, which C1 failed at S = 2 (2/4), holds at S = 4 on every seed. Only the
registered validity half fails.

## Options considered and rejected

1. **A longer budget, for every arm or for ORACLE only.**
   - *Projections.* After ~10k updates the ORACLE curves rise only slowly: +0.003 to +0.005 SET per doubling of the
     updates. Seed 831 is the binding seed. It projects to 0.880–0.888 at 48k and 0.883–0.896 at 96k, depending on the
     fit window. The chance that both seeds end ≥ 0.9 at 72k is about 0.03 (AICc-weighted saturating and log fits). A
     straight line gives about 0.23, but backtests show the straight line overshoots.
   - *Problems.* A pass would mostly be a noisy final evaluation crossing the bar. An ORACLE-only budget would validate
     a configuration the learned arms do not run. Extending every arm roughly doubles the GPU time, forces SINGLE to be
     rerun and the split schedule to be re-specified, and still most likely ends UNTESTED.
2. **A re-derived bar.** Any rule written now is chosen knowing 0.880 and 0.892, and the candidates disagree:
   - relative to the Bayes bound: 0.907–0.918 (fails);
   - bound − 0.05 = 0.877 (passes);
   - relative to the decay-faded learner: 0.851 (passes).

   S74's practice with an unusable configuration was to change the configuration or stop, never to move the bar.
3. **Validity redefined through the SINGLE half, which passes 10/10.** This would switch the gate to whichever half
   passed.
4. **Fresh ORACLE seeds (832–833) as a new validity probe.** This retests a failed gate.

What an amendment would have bought is the label, not the data. Under UNTESTED, every learned-arm quantity is still
printed.

## Diagnosis (an audit made after ORACLE missed the bar)

- **The bar was feasible.** The Bayes bound at S = 4, L = 768 is 0.927 (`explore_i_task.ideal_set_acc`, 144–512
  sequences). C1's bound, recomputed the same way, is also 0.927; S74's 0.932 came from 32–48 sequences.
- **The memory horizon did not transfer from C1.** The model's decay (0.95 per token) runs over global positions even
  under the perfect gate. The decay-faded per-source learner scores 0.680 at S = 4, against 0.840 at C1.
- **The mis-transfer was foreseeable.** S74's table already showed this drop (0.606 at S = 4 against 0.703 at S = 2,
  M = 44), and the docstring states that S = 4 was not piloted.
- **Where ORACLE falls short.** ORACLE beats the faded learner by ~0.20 (C1: ~0.07) but ends 0.035 and 0.047 under the
  bound (C1: ~0.014). About half of the shortfall is in bin 1, a block's first word, where the source's last word must
  be recalled across the gap since its previous block: 0.719 and 0.712, against a bound of ~0.944. The rest is a uniform
  loss of 0.017–0.036 on bins 2+.
- **What it would take.** With bin 1 at its bound and the other bins unchanged, seed 831 would reach about 0.900,
  exactly the bar.

## A hazard in the registered WIDE_SPLIT arm, declared before it runs

**The hazard.** SPLIT_THR = 0.88 was set as C1's lowest ORACLE (0.910) minus 0.03, "so a run routed as well as the oracle
does not fire". At S = 4 the oracle itself sits below 0.88. The trigger (`explore_i_common.make_split`) checks every
2400 updates. From 4800 on, a split is eligible when probe SET < 0.88 and the rise since the previous check is < 0.02;
at most 3 splits fire, at least 4800 apart. Replaying that rule on the ORACLE evaluation curves:

| curves | fires at 0.88 | 0.87 | 0.86 | 0.85 |
|---|---|---|---|---|
| S = 4 ORACLE 830 | 7200 | – | – | – |
| S = 4 ORACLE 831 | 4800, 9600, 14400 | 4800 | 7200 | – |
| S = 2 ORACLEs (C1, C2; 8 runs) | – | | | |

The trigger reads a 64-sequence probe, which is noisier than these 1024-sequence evaluations, so the replay is a
stand-in. The trigger does not check whether the map is already one-to-one, and at k = S a split overwrites a live
channel. **WIDE_SPLIT's ROUTED count is therefore expected to be biased low.** REG3 has no split and is unaffected.

**Not amended.** The screen is UNTESTED under any threshold. Changing SPLIT_THR changes the code SHA, which orphans the
ORACLE and SINGLE records or forces them to be rerun. And 0.85 has not been piloted on learned gates.

**How it will be described.** For each WIDE_SPLIT run the write-up lists every split that fired, with its step, probe
accuracy, map before, `on_target`, whether the run was ROUTED at the evaluation before it, and whether the run is ROUTED
at the end. It also counts fires on a one-to-one map, and runs that were ROUTED before a fire and not at the end.

## Run selection (fixed now, before any learned-arm data exist)

- **RTX 3070: the registered arms only.**

      python3 longrun.py start s75_S4 -- python3 -u -m explore_gpu.s75_s4 --device cuda --arms WIDE_SPLIT,REG3

- **GTX 1660 SUPER, if it runs alongside: WIDE_SLOW only.** WIDE_SLOW is WIDE_SPLIT without the split, registered under
  `--extended` as "printed, never read". It runs seeds 830–839 in about 5 h. `--extended` is required, and no other
  `--extended` arm is run.

      python3 longrun.py start s75_S4_slow -- python3 -u -m explore_gpu.s75_s4 --device cuda --extended --arms WIDE_SLOW

  The write-up describes its ROUTED counts beside WIDE_SPLIT's, paired by seed from the per-run lines. The pairing
  separates "the window gate does not route at S = 4" from "the split undid routing". If WIDE_SLOW does not run,
  WIDE_SPLIT's count is read as a likely underestimate of what the window gate does.
- **Report.** When the runs are done, gather every `s75_S4_*_results.json` in `explore_out/I/gpu/` and run
  `python -m explore_gpu.s75_s4 --report-only`.

## Follow-up commitment (fixed while blind to the learned arms)

These rules apply if an S = 4 follow-up screen is made:
- **Fresh seeds.** Every arm uses fresh seeds (840 and up).
- **Allowed inputs.** Its configuration, VALID bar, budget and split threshold come only from ORACLE and SINGLE
  quantities: S75's ORACLE 830–831 and SINGLE 830–839, the ideal-learner computations, and a new 2-seed ORACLE pilot of
  the exact configuration.
- **Excluded inputs.** They never come from S75's WIDE_SPLIT, REG3 or WIDE_SLOW outcomes.
- **Preferred route.** Following S74's practice, change the configuration so that 0.9 is reachable instead of moving
  the bar. For example, NB = 32, M = 352, L = 1536 has a bound of 0.964. Carrying S75's per-bin shortfalls over predicts
  ORACLE at about 0.92. It costs about 4× the attention per sequence and must be piloted before its bar is fixed.
- **The split.** At k = S the split gets a guard (no fire on a one-to-one map) or a spare channel.
