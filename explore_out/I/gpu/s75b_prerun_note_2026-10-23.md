# S75b pre-run note (2026-10-23, before the pilot; EXPLORATORY)

This note is committed before S75b's first run: stage 1, the pilot ORACLE_PILOT 840–841 on the GTX 1660 SUPER. At this
commit no S75b results file exists on the branch. `explore_out/I/gpu/s75b/` holds only the two benchmark files and their
archive, and no record of seeds 840–841 or 850–859 is in any file. The spec is `explore_out/I/s75b_spec_draft.md`,
approved at 114cd08, with Appendices A, B and C. The driver is `explore_gpu/s75b.py` at 3cef609, code SHA
ee04da11a164, unchanged since.

## What has been seen before the pilot

**1. S75's records at S = 4, including its learned arms.** These are seeds 830–839, L = 768, on the GTX 1660 SUPER, in
`explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json`: ORACLE 2, SINGLE 10, WIDE_SPLIT 10, REG3 10 and
WIDE_SLOW 10 records.
- **ORACLE 830–831 and SINGLE 830–839** are the allowed inputs of the follow-up rule (5bc6c2e). ORACLE ended at 0.892 and
  0.880.
- **The learned arms have been seen and read.** WIDE_SPLIT and REG3 came at b649751 and were merged at 142f47e; WIDE_SLOW
  came at 4c3984a and 4371824. Their write-ups were read too: `s75_S4_summary.md`, `s75_S4_descriptions.md`, and
  `report_2.md` (03bffdf, d552c6c), which this session wrote.
  - What they show, as report_2 describes it: WIDE_SPLIT 0/10, REG3 0/10 and WIDE_SLOW 0/10 ROUTED. Every gate is
    source-blind (η² about 0). The split fired 30 times, all on merged maps.
  - Referee V's review of the spec quotes the same quantities (η² ≤ 0.0004; no one-to-one map). This session's reviews
    of the driver ran with the records in the tree.

**2. The spec's appendices, and the review they answer.**
- Appendix A (13 October): the changes after approval.
- Appendix B (13 October): the provenance labels and the failure rules.
- Appendix C (22 October; c78d5d2): Referee V's review (`docs/review_2026-10/spec_review_s75b_draft.md`, 3f7d84b), taken
  item by item.
- Two description scripts read only allowed inputs:
  - `explore_gpu/s75b_rates.py`, the proceed rule's stop and VALID rates, from S75's ORACLE spread and the spec's ORACLE
    predictions;
  - `explore_gpu/s75b_triggers.py --check`, the trigger replay on S75's ORACLE curves, which reproduces S75's pre-run note
    table.

**3. Both cards' CUDA CHECK and benchmark outputs under the final driver** (code SHA ee04da11a164). Each card's latest
benchmark finished with every case ok, so every arm is usable on both cards.

| | GTX 1660 SUPER (the server) | RTX 3070 Laptop (the laptop) |
|---|---|---|
| CUDA CHECK | a7e7e12, "ALL CHECKS PASSED" | fb3066c, "ALL CHECKS PASSED" |
| B7 (lock-step at L = 1536, R = 1, 20 updates on 850 and 851) | every arm passes, REG3 in both register modes; the doubled-W_g control fails as it should | the same |
| B9 (a real divergence) | every case is classed diverged; `run_arm` makes one attempt | the same |
| benchmark | b06a81a: status done, every case ok, written 2026-10-22 23:11:48Z | eb3e819: status done, every case ok, written 2026-10-23 01:45:50Z |
| `--host-ram-gb`, `--beside` | 35, none (MemAvailable 34.8 GB at the start) | 15.2, none (MemAvailable 9.0 GB at the start) |
| REG3's register mode, by rule | scan: peak reserved 5.19 + context 0.11 GB < 5.62 GB | scan: 5.19 + 1.14 GB < 7.71 GB |
| `--stats-workers`, by rule | 1, which keeps up: budget 33.5 GB (bound: the machine's total); need 4.6 GB | 1, which keeps up: budget 8.5 GB (bound: MemAvailable); need 4.8 GB |
| seconds per update (benchmark, 12 updates) | ORACLE 0.27 (whole run), SINGLE 0.11 (whole run), WIDE 0.14–0.16, REG3 scan 0.22, loop 0.57 | ORACLE 0.11 (whole run), SINGLE 0.08 (whole run), WIDE 0.11–0.18, REG3 scan 0.17, loop 0.37 |
| archived, counted as failed or superseded | the first 1660 benchmark (68284a1, status "in progress"), archived in b06a81a | the 13 October benchmark (code 0759062126f5), archived in eb3e819 |

- **The laptop's commits are stamped −0600.** The user reports that its clock is misconfigured. The UTC times the driver
  wrote agree with the commits' instants: the CUDA CHECK started at 01:43:41Z and the benchmark at 01:45:50Z on 23
  October, and the commits fall at 01:45:26 and 01:57:58 UTC. No rule compares times across machines.
- **The spec's cost section predicted the loop for REG3 on the 1660** (Machines and run mechanics). The rule, fixed
  there, reads the benchmark and gives the scan: 5.30 GB with context, under the card's 5.62 GB. The rule decides. The
  prediction was an estimate.

## No learned-arm constant was chosen from what was seen

- **Every design constant is the approved spec's (114cd08)** and has not changed since:
  - configuration B (S = 4, V = 4, blocks of 6–14, NB = 32, M = 352, L = 1536, the model, 24,000 updates);
  - the seeds (pilot 840–841, screen 850–859) and the arms;
  - the 0.9 bar and the proceed rule (mean ≥ 0.912 and each ≥ 0.9);
  - SPLIT_THR's rule (min of the pilot − 0.03), the trigger's constants, and the guard with its precedence;
  - ROUTED, VALID and the reading.
- **Appendices A, B and C change no design constant.** Each says so, and Appendix C changes no code.
- **The settings the benchmarks gave were set by rules fixed before the benchmarks ran**, from each benchmark's own
  measurements of this card: REG3's register mode (`register_rule`), `--stats-workers` (`workers_rule`), and each arm's
  usability. These are run mechanics, not design constants. No learned-arm outcome enters them.
- **SPLIT_THR will be set by its rule from the pilot**, and from nothing seen here.
- **The two design changes made after the merge 142f47e** came in 114cd08: the loosened proceed rule, and WIDE_SLOW
  made unconditional. They were made with the learned-arm records in the tree. They are disclosed, and not blind by git
  (Appendix C.5). Neither is chosen again here.
- **What this note does not claim.** It does not make S75b blind. The design commit 1e31409 is blind on the author's
  statement only, and everything after the merge 142f47e is not blind by git, this note included. The note records what
  had been seen when the pilot started. Any later change to a constant is not blind and must say so in S75b's record.

## What runs next (fixed)

1. **Stage 1 on the GTX 1660 SUPER:** `python -m explore_gpu.s75b --stage 1 --device cuda`.
   - It takes about 3–4 h: two runs of 24,000 updates.
   - Its start is pushed (the results file and the running stage-1 record), and so is its end.
   - The proceed rule is applied by the driver to the committed records. The stage-1 record says PROCEED or STOP.
2. **After stage 1, a description that decides nothing:** `python -m explore_gpu.s75b_triggers --replay`.
3. **Stage 2 on both cards, only if the record says PROCEED:** `--machine 1660` on the server, `--machine 3070` on the
   laptop.
