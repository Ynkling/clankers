# Replicating the eight-stream window-gate result from the Revision 8 report

**Independent replication, EXPLORATORY, not a result.** One container, one implementer, 20 paired seeds.

Sources for Phases 1–2: `multichannel_hebbian_report_v8.pdf` and github.com/pathwaycom/bdh only. The main line was fetched for the first time in Phase 3, after the Phase 2 records had been pushed (`de8df0f`).

## 1. Counts

Primary outcome: **bound routed**. Seeds 900–919, S = 8, P = 4, k = 16, budget 43,200 updates. Wilson 95% intervals in brackets.

| arm | this replication | report (X / L / pooled) |
|---|---|---|
| WIN3_SPLIT_D8 (window gate + SLOW + SPLIT) | **17/20** [0.640, 0.948] | 18/20 / 20/20 / 38/40 [0.835, 0.986] |
| WIN3_SLOW_D8 (window gate + SLOW) | **7/20** [0.181, 0.567] | 6/20 / 9/20 / 15/40 [0.242, 0.530] |
| paired SPLIT vs SLOW, discordant pairs | **10 vs 0**, exact one-sided McNemar p = 0.00098 | 23 vs 0, p = 1.2×10⁻⁷ |
| perfect gate (validity) | 2/2 (transitions 6000, 10,800) | 4/4 (3600–6000) |
| **pre-registered reading** | **"partially replicated"** (needs ≥ 18/20 and p < 0.05 for "replicated") | — |

Phase 3 reruns: WIN3_SPLIT_D8 on seeds 900–909, with one discrepancy set to the main line's choice.

| rerun | change | bound routed | Phase 2 on the same seeds |
|---|---|---|---|
| R1 | D-1: the main line's probe (one probe from `Generator(12345)` for every seed) | **9/10** [0.596, 0.982] | 9/10 |
| R2 | D-2: key mass over body keys only | **9/10** [0.596, 0.982] (7 of 10 runs bit-identical to Phase 2) | 9/10 |

**Reading.**
- 17/20 sits one run below the threshold fixed before the run, and its interval contains the report's 38/40 (0.95).
- If the true rate were 0.95, a count of 17 or fewer would occur with probability 0.075.
- The no-split arm (7/20) agrees with the report's 15/40, and so does the direction and size of the paired effect: every discordant pair favours the split (10 vs 0 here, 23 vs 0 there).
- Neither rerun moved the count. **The shortfall is consistent with seed-level variation, not with an identified implementation difference.** With 20 seeds this replication cannot tell 0.85 from 0.95.

**Diagnostics.**
- The trigger fired 45 splits in the 20 SPLIT runs (one to three each). 44 copied a channel holding ≥ 2 streams onto one holding none, by the labels.
- Every one of the 16 Phase 2 failures (3 SPLIT, 13 SLOW) ended with two or three streams merged on one channel, as in the report.
- Median transition over bound-routed runs: SPLIT 20,400, SLOW 22,800. The report gives 21,600 / 18,600 and 27,600 / 32,400.
- The three SPLIT failures:
  - seed 907: its first split came late (14,400) and all three were spent with a merge left;
  - seed 914: the probe read ≥ 0.95 while the held-out evaluation sat at 0.88, so the plateau rule stopped firing after one split;
  - seed 915: two three-stream merges remained after three splits.
- Per-run records: `records/phase2.jsonl` and `records/phase2_table.md` (map, splits as (update, c*, c0), per-stream accuracy, wall time). Training took 37.4 h over 4 workers × 1 thread.

## 2. What the report left open (UNSTATED.md, 44 entries; summary)

I logged 44 choices before running.

**Most are harmless:** RNG streams, the probe's draw, which held-out set, the vocabulary, the order of operations at a shared step.

**Several were genuinely ambiguous and could each have changed the outcome. In every one my reading turned out to match the main line:**
- decay replaces RoPE (U-2);
- the embedding init std is 0.02, which sets the scale of the gate's input (U-6);
- the gate reads the model's own raw embedding (U-9);
- the window gate's width (32) and init (N(0, 0.1²)) are carried over from the recurrent gate (U-10, U-11);
- "conv" is one shared width-4 identity filter on the D-dim layer input, feeding Q=K and V (U-14, U-15);
- Adam, not AdamW, with no weight decay (U-24);
- the probe is also read at 2400 so the first check at 4800 has a reference (U-37);
- "zero W_g's Adam state" includes the step counter (U-39);
- the stream-to-channel map is the argmax of the mean gate at value positions (U-32; the main line's README 8.1 confirms "by largest mass").

**Where my choice differed from theirs:**
- the oracle's learning rate (U-28; theirs: single-rate Adam, mine: SLOW);
- the probe's draw (U-36);
- the key positions for the split's mass (U-38).

**Disclosure.** Before pre-registering I ran one oracle-only pilot on seed 1, to catch bugs (U-42). It bound at 4800. No learned-gate run was piloted.

## 3. Where the implementations differ (DIFF.md, 17 items; summary)

**Mechanical check.** I loaded my weights into the main line's model (`test_window_gate.to_local(MultiBDH)` on the D8 path). Gates agree to 7×10⁻⁹ and query logits to 7×10⁻⁸. Model, gate, conv, init distributions, SLOW groups and switch, batch, evaluation and the bound-routed definition are the same.

**Labels.** No item was "contradicts the report" or "I misread it". All are "unstated", except the PyTorch patch version (2.14.1 here against the report's 2.14.0; stated, not available).

**The differences that act on the split comparison:**
- **D-1, the probe.** One fixed probe for all seeds against my per-seed draw. Rerun R1 changed when and where the splits fired on every seed, but not the count (9/10 → 9/10). Seed 907 failed again.
- **D-2, the key positions.** Their 32 body keys against my 33 (with the query key). Rerun R2 left c* and c0 unchanged in 7 of 10 runs (bit-identical to Phase 2). In the other 3 a split chose a different target with the same outcome: 9/10 → 9/10.

**Differences that only re-draw random numbers, or are inert in my records:**
- D-3: the oracle's learning rate. Validity only; it explains my later oracle transitions.
- D-4: the order of the trigger and the early stop. Inert in practice.
- D-5: the last check at 40,800. Inert, since my latest split was at 36,000.
- D-6 to D-10: data, held-out, init, noise generators, and the map measured on a 256-sequence probe rather than 2048 held-out sequences.
- D-11 to D-16: the vocabulary (the `?` token), the tap order and similar. No effect on the forward pass.

## 4. What the report should add for the next reader

1. **The validity arm's training.** The perfect gate uses single-rate Adam at 1e-3, not the recipe's SLOW schedule. A reader who uses SLOW gets later oracle transitions and may suspect a bug.
2. **The probe.** "A fixed 64-sequence probe" should say: the same 64 sequences for every seed (generator 12345); only the query is scored; the probe is also read at 2400 as the reference for the check at 4800; the last check is at total − 2400.
3. **The key positions** whose mass picks c* and c0: the body's 32 keys, not the query key.
4. **The probe can be wrong about a plateau.** One of my three failures (seed 914) was a probe reading ≥ 0.95 while the model sat at 0.88 with two streams merged, so the rule stopped. A 64-query probe has a standard error of about 0.04 near 0.9. The report should say how often its checks were blocked this way, or score more queries per probe sequence.
5. **The architecture choices the reader must otherwise guess**, each of which I had to guess (and guessed right):
   - decay instead of RoPE;
   - the embedding init std (0.02), since the gate reads the raw embedding;
   - the window gate's hidden width and init;
   - that the gate shares the model's embedding table;
   - "conv" as one shared filter on the D-dim layer input, with its output also used as V;
   - Adam with no weight decay and no dropout;
   - the 28-token vocabulary with the query's VAL as the target (no `?` token).
6. **A rate with its uncertainty, not only a pooled count.** 38/40 has Wilson [0.835, 0.986]. A 20-seed replication at 17/20 is inside it. The report should say what count an outside replication should expect. At a true rate of 0.95, a 20-seed replication reaches ≥ 18 with probability 0.92 and ≥ 17 with 0.98. At 0.90 it reaches ≥ 18 with only 0.68. So my "≥ 18/20" threshold would fail a recipe that really binds 90% of the time about one time in three.
7. **The errata in README 8.1** ("one channel per stream" by largest mass; the withdrawn conv argument; the per-part seed ranges) belong in the PDF as well.

## Provenance and isolation

- **Branch:** `claude/replicate-R` (orphan). Commits:
  - pre-registration `786028e`
  - Phase 2 complete `de8df0f`
  - rerun design `e370c14`
  - DIFF.md `00990ea`
  - this report: the branch head
- **What I saw of the clankers repository before Phase 3.** The attached PDF was inside the session's checkout of it (`/home/user/clankers`). A filename search to locate the PDF printed six filenames from that checkout: `bdh_multichannel.py`, the `.tex` sources of v7 and v8, and older report PDFs. I opened none of them and saw none of their contents. Apart from the v8 PDF, nothing from that checkout was read. I judge the replication not void, but the reader should know.
- **Environment:** 4 workers × 1 thread, PyTorch 2.14.1 (CPU).

---

## Phase 4: the 40-seed count (added after the report above; Phases 1–3 unchanged)

**Independent replication, EXPLORATORY, not a result.** The design and readings were fixed in `phase4.py` and pushed (`464be8d`) before any Phase 4 run.

**Design.**
- **Split arm:** WIN3_SPLIT_D8 on seeds 900–919 (Phase 2, my probe and key positions), together with WIN3_SPLIT_D8_ML on 920–939. WIN3_SPLIT_D8_ML is the split as the main line specifies it: one probe from `Generator(12345)` for every seed, and c*/c0 from the 32 body key positions.
- **No-split arm:** WIN3_SLOW_D8 on 900–939. It has no split, so pooling it across phases is exact.
- **Validity:** the perfect gate under single-rate Adam (ORACLE_ADAM) on seeds 920–921.

| | count | Wilson 95% |
|---|---|---|
| split arm, 40 seeds | **35/40** | [0.739, 0.945] |
| WIN3_SLOW_D8, 40 seeds | 12/40 | [0.181, 0.454] |
| paired split vs no split | **23 vs 0**, exact one-sided McNemar p = 1.2×10⁻⁷ | |
| secondary: WIN3_SPLIT_D8_ML, 920–939 only | 18/20 | [0.699, 0.972] |
| secondary: WIN3_SLOW_D8, 920–939 only | 5/20 | [0.112, 0.469] |
| ORACLE_ADAM (validity) | 2/2, both at 2400 | |
| report, pooled | 38/40 vs 15/40, 23 vs 0 | [0.835, 0.986] |

**READING (mechanical): "consistent with the report".** The interval [0.739, 0.945] overlaps [0.835, 0.986], and the split arm beats the no-split arm at p = 1.2×10⁻⁷.

Under the main line's specification alone, 920–939 gave 18/20. That is the count Phase 2's threshold asked for.

**What the 40 seeds can and cannot say:**
- The point estimate, 0.875, sits below the report's 0.95.
- The interval excludes rates below 0.74 and includes 0.95.
- 40 seeds can still not tell 0.85 from 0.95. Separating them at this precision needs on the order of 150 seeds per arm.
- The no-split arm (12/40, 0.30) matches the report's 15/40 (0.375). So does the paired effect: no seed in either replication or in the report bound without the split and failed with it.

**Failures.**
- WIN3_SPLIT_D8_ML failed on seeds 920 and 933. Both ended with two streams sharing a channel after three splits, the same failure the report gives for X's seeds 555 and 556.
- All 44 Phase 4 splits were on target by the labels (c* held ≥ 2 streams, c0 none).
- All 15 WIN3_SLOW_D8 failures were merges of two or three streams.
- Median transition over bound-routed runs: 19,800 with the split, 19,200 without.

**Probe-versus-evaluation blocks (seed 914's failure mode).** A blocked check is one at update ≥ 4800 where the probe reads ≥ 0.95 while the held-out evaluation at the same update is < 0.95.

| split runs | checks | blocked | of them otherwise eligible (cap and gap allowing) | runs affected | of those, not bound routed |
|---|---|---|---|---|---|
| Phase 2, per-seed probes (900–919) | 192 | 13 | 12 | 906, 914 | 914 |
| R1, the report's probe (900–909) | 91 | 1 | 1 | 905 | none |
| R2, per-seed probes, body keys (900–909) | 90 | 0 | 0 | — | — |
| **Phase 4, the report's probe (920–939)** | **172** | **2** | **0** | **none** | **none** |

The opposite disagreement (probe < 0.95 while the evaluation is ≥ 0.95) never occurred.

With the report's probe, the failure mode appeared at 3 of 263 checks, and never decided a run. With per-seed probes it decided one run in twenty: one probe drawn per seed is sometimes easy enough to read ≥ 0.95 while two streams are still merged. The fixed probe from `Generator(12345)` happened not to be easy in that way on these seeds. That is a property of one particular 64-sequence draw, not a guarantee.

**For the report (adding to Section 4):**
- Specify the probe as one fixed draw, as the main line does.
- Say that its 64 queries can over-read a merged model.
- Better still, gate the trigger on more queries per probe sequence.

---

## Phase 5: the copy against its reset, and one delta-rule channel (added after Phase 4; earlier sections unchanged)

**Independent replication, EXPLORATORY, not a result.**

**What was replicated.** The first and third results of the main line's `test_split_copy`, from its specification and docstring at `ffb0771` only.
- **Not read:** no code below that docstring, and nothing from Session F's delta-memory modules.
- **Protocol:** committed before the first run (`550a736`). It is the reading rule you stated, "consistent" if the count lies in the main line's pooled Wilson interval for that arm, applied per arm, with no "replicates" label. The "Phase 4, amended" text in the main line's file states a different rule (a ≤ 27/32 threshold on fresh seeds). That difference is recorded in `phase5.py`.
- **Guesses:** every detail I had to guess is in UNSTATED.md, U-50 to U-70. The ones that matter most:
  - RESET zeroes W_g's Adam state only (the spec's wording, against "the gate group" in your message);
  - the delta memory keeps raw queries with L2-normalised keys;
  - the decay is placed so that β = 0 recovers the Hebbian memory exactly;
  - DELTA1 has the same width as the other arms.

**Arms.** Seeds 940–959, paired by seed.

| arm | what it is |
|---|---|
| COPY | WIN3_SPLIT_D8_ML: my Phase 4 recipe, unchanged |
| RESET | the same trigger; W_g's Adam state is zeroed and nothing is copied |
| DELTA1 | k = 1, no gate, a delta-rule write at β = 1 (L2 keys, tied write, decay 0.95), single-rate Adam 1e-3 |
| validity | the Hebbian perfect gate (ORACLE_ADAM) and the perfect gate on delta channels (ORACLE_DELTA), each on 940–941 |

**Results.**

| arm | this replication | Wilson 95% | main line, pooled | main line's interval | reading |
|---|---|---|---|---|---|
| COPY (bound routed) | **18/20** | [0.699, 0.972] | 18/20 | [0.699, 0.972] | **consistent** |
| RESET (bound routed) | **9/20** | [0.258, 0.658] | 6/20 | [0.145, 0.519] | **consistent** (0.45 is inside) |
| DELTA1 (bound) | **0/20** | [0, 0.161] | 0/20 | [0, 0.161] | **consistent** |
| validity | ORACLE_ADAM 2/2 (6000, 6000); ORACLE_DELTA 2/2 (1200, 1200) | | CEIL_C 4/4; CEIL_DELTA_D8 4/4 (all at 1200) | | valid |

Discordant counts (printed, not read; p is the exact one-sided McNemar, descriptive):
- COPY vs RESET: 9 vs 0 (p = 0.0020). The main line had 12 vs 0.
- COPY vs DELTA1: 18 vs 0 (p = 3.8×10⁻⁶). The main line had 18 vs 0.

**What the runs show (diagnostics, not claims).**
- **No seed went the other way.** On no seed did the reset bind where the copy failed.
- **What the 9 shared binds were.**
  - On two of them (941, 943) the trigger never fired. Those seeds bind on their own, and both arms are bit-identical there.
  - On the other seven the reset fired and the run still bound.
- **RESET runs higher here than on the main line** (0.45 against 0.30). It is still inside their interval.
- **Every failure was a merge.** All 11 RESET failures and both COPY failures (945 and 947, three streams sharing a channel) ended with two or three streams sharing one channel. RESET failed on both of COPY's failing seeds too.
- **Trigger targets.** The trigger fired 37 copies and 46 resets. By the labels, every target was a channel holding ≥ 2 streams copied or reset onto one holding none.
- **Probe-versus-evaluation blocks** (seed 914's failure mode): 1 of 175 checks in COPY (seed 958, which bound), and 2 of 246 in RESET (neither eligible). Again they decided no run.
- **DELTA1.**
  - It never bound. Its final accuracy ranged 0.09–0.78 (median 0.29); the main line reports 0.08–0.76.
  - Per-stream accuracy is roughly even within a run (for example 0.72–0.79 on seed 942). The single channel learns some bindings for every stream rather than separating them.
  - The same delta memory under the perfect gate bound at 1200 on both seeds, so the failure is not in the delta write.
- **Cost.** 78.3 h of training on 4 workers × 1 thread; each DELTA1 run took about 2 h.

**What this phase adds.** From the written specification alone, an independent implementation reproduces:
- the copy arm's count exactly;
- the reset arm within the main line's interval;
- the single delta channel's 0/20.

The copy's advantage over its reset holds in direction (9 vs 0 against 12 vs 0). It is smaller here because RESET bound more often. With 20 seeds, 9 vs 0 and 12 vs 0 are not distinguishable. Only a pre-registered paired test would turn "the copy, not the reset" into a claim; this phase reads per-arm consistency only.
