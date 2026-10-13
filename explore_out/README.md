# explore_out (branch claude/explore-I)

This branch holds only Session I's outputs; the other sessions' sections are on claude/outside-ideas and
claude/explore-F/G/H.

## Session I

EXPLORATORY, not a result. Subject: does the channel partition emerge under a dense next-token loss on interleaved
sources (ICMC: in-context Markov chains with sources)? Code: explore_i_task.py, explore_i_common.py, explore_i_checks.py,
explore_i_pilot.py (S74), explore_i_dense.py (S75, a draft with placeholders, not run). Seeds used: 800-803.

- **S74 pilot (report_1.md, s74_pilot_log.md): nothing usable; stopped.** The usability bar (ORACLE SET ≥ 0.9) forces
  V = 4 and L ≥ 384 at S = 2 (exact Bayes bound) and puts S = 4 out of reach (L ≈ 768). C1 (blocks 6..14, L = 384):
  ORACLE 4/4 ≥ 0.9, SINGLE ≤ ORACLE − 0.2 on 2/4. C2 (blocks 3..7, L = 480): ORACLE 4/4, SINGLE on 1/4, because three
  single-channel runs learned to carry the source in the stack (probe from the residual entering layer 3: 0.52-0.58 →
  0.98 at one evaluation, SET +0.10-0.21). Under a dense loss the stack does carry the cue, where the sparse-loss stacks
  did not.
- **S75: not run** (no usable configuration). Options for the user are listed at the end of report_1.md.
- **GPU package (explore_gpu/, its README; tooling, no result).** A device-agnostic runner for the project's model and
  tasks. The sequential path is the project's own code with the model on the GPU; it also runs a main-line test's arm,
  through `explore_gpu.mainline`. The batched path stacks R runs of one arm, with a per-run Adam. Its CHECKs ran here on
  CPU (explore_gpu/checks_cpu_output.txt): X's record A|160 is reproduced bit for bit, and the batched path equals the
  sequential path bit for bit over 2400 free-running updates. The CUDA CHECK is lock-step, with stated tolerances, and is
  for the user to run. S75 at S = 4 (seeds 830-839, L = 768) is fixed in explore_gpu/s75_s4.py, to run on the user's GPU
  once the CUDA CHECK passes. Its outputs go to explore_out/I/gpu/. The runtime estimate is explore_gpu/runtime_estimate.md.
- **S75 at S = 4, ORACLE and SINGLE (GTX 1660 SUPER, 2026-10-11).**
  - The CUDA CHECK passed.
  - ORACLE: 0.892 and 0.880, under the 0.9 bar, so the registered result is **UNTESTED**.
  - SINGLE: 0.534–0.595, so SINGLE ≤ ORACLE − 0.2 holds on 10/10.
  - **Pre-run note** (gpu/s75_S4_prerun_note_2026-10-12.md, committed before any learned-arm run): no amendment, and
    the reasons. It also records a diagnosis: the bar was feasible (bound 0.927), but the decay horizon did not
    transfer from S = 2.
  - **Declared hazard.** SPLIT_THR = 0.88 fires on oracle-quality runs at S = 4, so WIDE_SPLIT's ROUTED count is
    expected to be biased low.
  - **Fixed run selection.** WIDE_SPLIT and REG3 on the 3070, and WIDE_SLOW as the no-split twin on the 1660.
  - **Follow-up rule.** Any later S = 4 screen derives its constants from ORACLE and SINGLE data only.
- **S75 at S = 4, all registered arms (GTX 1660 SUPER; report_2.md).**
  - **UNTESTED, as registered.** ORACLE ended at 0.892 and 0.880.
  - **Descriptions only.** WIDE_SPLIT 0/10 and REG3 0/10 ROUTED. Every gate stayed source-blind (η² 0.00) at every
    evaluation, and SET stayed at SINGLE's level, with GAP ≈ 0.
  - **The split.** 30 fires, all on merged maps; the declared hazard did not arise.
  - **Correction to the pre-run note.** Three WIDE_SPLIT runs had already finished on the 1660 when it was committed.
    It was still blind: no learned-arm record was pushed until 13:53.
  - **WIDE_SLOW (seeds 830–839).** 0/10 ROUTED and source-blind, as WIDE_SPLIT. Neither arm routed on any shared
    seed.
- **S75b draft spec (s75b_spec_draft.md, for the user).** The design commit 1e31409 was built on a tree without S75's
  learned arms; that it was blind is the author's statement. Every change after the merge 142f47e (ee97443, 114cd08
  and the driver's) is not blind by git: Appendix B.
  - Configuration B: S = 4, NB = 32, M = 352, L = 1536, bound 0.963.
  - A 2-seed ORACLE pilot with proceed and stop rules fixed in advance.
  - SPLIT_THR set from the pilot, and the split guarded against one-to-one maps.
  - Fresh seeds 850–859, and an estimated ~1.4 days on both cards.
- **S75b driver** (explore_gpu/s75b.py, with CHECKs in s75b_checks.py). It implements the spec as approved at 114cd08.
  - **Changes since approval (not blind).** Every change is listed in the spec's appendix and is written into each
    record. None changes a design constant.
  - **Failure rules (2026-10-13, before the pilot; spec Appendix B).** These cover the four outcomes Referee V found
    uncovered:
    - a failed pilot run;
    - a code change between the stages;
    - a failed stage-2 ORACLE run;
    - a failed benchmark.
  - **Divergence.** A diverged run is an outcome, never rerun. Any other failure is rerun on the same seed.
  - **CPU CHECKs B1–B8 pass.** The cards' CUDA CHECKs and benchmarks are rerun on the commit that adds these rules; the
    stages follow.
- **window_gate on the GTX 1660 SUPER** (gpu/mainline/window_gate_prerun_note_2026-10-12.md). The pre-run note was
  committed before the run, with no reading. It covers:
  - CEIL_C on 2540–2541 first;
  - three arms on 2540–2559;
  - what is described, beside X, L and the pooled counts.

  `mainline.py` now uses one thread per worker on every device.
- **window_gate on the GTX 1660 SUPER: the run (report_3.md).** Descriptions only, with no reading, as the note fixed.
  The records are 148c161.
  - **Validity.** VALID by the test's rule: CEIL_C bound on 2540 and 2541.
  - **BOUND ROUTED on 2540–2559.** WIN3_SPLIT_D8 19/20, HINGE_D8 0/20, WIN3_SLOW_D8 6/20.
  - **Discordant pairs.** Against HINGE_D8, b 19 and c 0. Against WIN3_SLOW_D8, b 13 and c 0.
  - **Beside the CPU runs.** These sit beside X's, L's and the pooled counts in `gpu/mainline/window_gate_descriptions.md`
    and are never pooled with them.
