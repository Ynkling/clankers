# Session F — report 10 (round 7, S90-F: the context-pool family). EXPLORATORY, not a result

Branch `claude/explore-F`.
- **Task module:** `explore_f_tasks.py` at **9017b82** (`9017b823e69bde2ddbe7502c61fefc8fe530be2b`), the commit G
  vendors. It is append-only: b0c6332's file is unchanged above it (CHECKed).
- **Screen and readings:** `explore_f_s90_ctx.py`, committed at 73749e8 before any run.
- **Setup:** Intel Xeon @ 2.10GHz for every run; torch 2.14.0+cu130; 1 thread per run; code SHA 809fde1f5168.
- **Runs:** 96/96 ok, none retried, none diverged.
- **Seeds:** 3060-3071, free on all 13 branches.

Full tables are in `s90_ctx_tables.md`; the decision file is `s90_decision.json`.

## Counts first

**The screen** (one channel, no gate; seeds 3062-3071 paired; 24000 updates, SLOW):

| task | one delta channel | one Hebbian + conv channel | paired: delta only / Hebbian only (one-sided McNemar) |
|---|---|---|---|
| -ctxpool (D = 2) | **10/10** [0.72, 1.00] RELIABLE; median 9600 (7200-15600) | 0/10 [0.00, 0.28] NEVER; final 0.48-0.51 | 10 / 0 (p = 0.00098) |
| -docs3 (D = 3) | 0/10 [0.00, 0.28] NEVER; final 0.30-0.43, one at 0.81 | 0/10 NEVER; final 0.32-0.35 | 0 / 0 |
| -docs4 (D = 4) | 0/10 [0.00, 0.28] NEVER; final 0.20-0.25 | 0/10 NEVER; final 0.24-0.26 | 0 / 0 |

**SCREEN READING (fixed before the runs): "one channel binds without a context identity"**: one delta channel binds
-ctxpool 10/10, which is ≥ 7/10.

**Validity pilots** (the perfect gate handed the per-sequence index, k = D; seeds 3060 / 3061; SLOW, 24000):

| task | Hebbian + conv | delta (β 1, L2) | reading, both memories |
|---|---|---|---|
| -ctxpool | 2/2 (2400, 4800) | 2/2 (1200, 2400) | **the task is valid** |
| -docs3 | 2/2 (3600, 3600) | 2/2 (1200, 2400) | **the task is valid** |
| -docs4 | 2/2 (3600, 3600) | 2/2 (1200, 2400) | **the task is valid** |

- All six task × memory pairs are valid at 24000, so no 48000 fallback ran. Each pair is 2/2 [0.34, 1.00] RELIABLE.
- **Phase 0 stopped nothing**, and every no-memory accuracy is below its positional bound:

| task | conv on the embeddings (reach 4) | positional bound | conv into the memory (reach 10) | positional bound |
|---|---|---|---|---|
| -ctxpool | 0.17 / 0.17 | 0.19 | 0.37 / 0.37 | 0.42 |
| -docs3 | 0.12 / 0.12 | 0.16 | 0.19 / 0.20 | 0.31 |
| -docs4 | 0.11 / 0.11 | 0.14 | 0.16 / 0.12 | 0.25 |

  The stop rule (0.6) could not have fired. "Not stopped" says nothing beyond these bounds.

## The tasks and the cap

-ctxpool is lite with lite's 4 key types kept; only the contexts are drawn, 2 of 16 types per sequence. -docs is the
same at D = 3 and 4 documents. Every batch is lite's batch (at S = D) with only the contexts renamed (CHECKed). The stream
index is by first appearance.

**Frequencies** (S87's sample):
- KEY 0.101-0.103 (4 types), VAL 0.022-0.024 (16), CTX 0.013-0.015 (16).
- On all three tasks the largest-ratio cut selects the 4 KEY types, and the half-mass cut 4 KEY + 4 VAL. No cut selects
  a context.

**The cap of a fixed type → channel map** (the fraction of sequences whose D contexts land on D distinct channels; exact
over every split of the 16 types; the balanced split is best):

| D | k = D | k = 2D | k = 16 |
|---|---|---|---|
| 2 | 64/120 = **0.533** | 0.800 | 1 |
| 3 | 150/560 = **0.268** | 0.664 | 1 |
| 4 | 256/1820 = **0.141** | 0.615 | 1 |

The cap is a property of k = D. With one channel per type it vanishes.

## What the runs show (descriptive)

- **A delta channel holds two documents; a Hebbian channel holds one.**
  - One delta channel binds the two-document task on every seed. That holds without any routing and without a fixed
    context identity, so the contexts' missing identity is no obstacle to a single delta channel at D = 2.
  - On lite's header layout one delta channel bound 8/10 (Revision 10 §11). -ctxpool's 10/10 is no worse once the
    contexts lose their identity.
  - At D = 3 and 4 the delta channel fails on every seed. The one exception, -docs3 3065, climbs to 0.79 and is still
    rising at 24000.
- **The Hebbian channel's failures sit at 1/D.**
  - With the conv, one Hebbian channel ends at 0.48-0.51 (D = 2), 0.32-0.35 (D = 3) and 0.24-0.26 (D = 4): one of the key's
    D values at random.
  - Among its misses, the last-written value's share is also about 1/D (0.52, 0.34, 0.26). The channel retrieves the
    key's values without telling the documents apart, and without preferring the last write.
  - The failing delta channels at D = 3 and 4 sit at the same level (0.36 and 0.22 mean; last-written share 0.34 and
    0.26).
- **The type-level perfect gate** (a fixed balanced map at k = D; descriptive, 2 seeds):
  - Hebbian + conv: unbound at 0.77 / 0.76, 0.73 / 0.72 and 0.81 / 0.73.
  - Delta: binds all three tasks (8400-10800), final 0.96-1.00.
  - Within each sequence its statistics equal the cap: separated 0.53 / 0.32 / 0.14 against 0.533 / 0.268 / 0.141 (the
    probe's 256 sequences).
  - **A post-hoc account fits these numbers.** Suppose a channel holding m documents answers 1/m of their queries if
    Hebbian, and all of them if delta with m ≤ 2. Then the type-level gate's expected accuracy is:
    - Hebbian: 0.767 / 0.732 / 0.728 (observed 0.77 / 0.76, 0.73 / 0.72, 0.81 / 0.73);
    - delta: 1.000 / 0.952 / 0.946 (observed 0.98 / 1.00, 0.96 / 0.97, 0.97 / 0.96).

    The delta runs sit slightly above the prediction at D = 3 and 4, so a delta channel holds a third document on some
    queries. That fits the climbing -docs3 3065 run.
- **Routing statistics in the pilots:** every perfect-gate pilot ends with margin 1.00 and every probe sequence separated
  (1.00), within sequences.
- **The corrected write-order split** (last write / not last) is flat wherever a run binds. Example: one delta channel on
  -ctxpool, 0.99 / 0.99.

## Design decisions (stated in the docstrings before any run)

- **-ctxpool keeps lite's key identities exactly.** There is no key draw; only one draw renames the contexts. So it
  differs from lite only in the contexts' identity. -docs is the same class at S = D.
- **The stream index is by first appearance at any S.** At S = 2 it is -allpool's convention (CHECKed).
- **Within-sequence statistics at S > 2:**
  - the mean pairwise TV distance between the streams' mean KEY gates;
  - the fraction of sequences with D distinct argmax channels each ≥ 0.9;
  - the mean pairwise VAL cosine.

  η² by stream is not used at S > 2: it is 1 whenever each stream's gate is constant, even when two streams share a
  channel. The lite-id statistics (`key_sep`, `key_part`) are NaN on these tasks.
- **The 48000 fallback is per task and memory.** A memory that bound 2/2 at 24000 is not rerun (V's point (b)). It was
  not needed.
- **The perfect gate the screen is read against is the pilot** (2/2, which is the validity). It was not rerun on
  3062-3071.
- **Stage timing:** the screen's runs ran in the "phase1b" stage. The 24000 budgets were recorded at phase 1's decision,
  so the arms were queued with the (empty) fallback stage. The "phase2" and "final" stages found nothing left. This
  changes when runs ran, not what they are.

## CHECKs (all passed, asserted before every stage)

- explore_f_delta_checks 26/26.
- **S90-F's 107 CHECKs**, in three parts:
  - **tasks:**
    - the module equals 9017b82's and starts with b0c6332's file;
    - per task: lite's layout at S = D (check_rand_header), the batch as lite's with only the contexts renamed, the
      index, every key once under each stream with distinct values, the query answered by exactly one pair, the
      perfect gate, no CTX type as frequent as a value type, and the cap (exact and against the balanced map's
      empirical rate);
    - the synthetic gates: relative and per-sequence-permuted gates score separated 1 and VAL cos 0, and the type-level
      gate's separated fraction equals the probe's;
    - the corrected write order;
    - the frequency tables and cuts;
  - **ties:** this screen's code path reproduces three recorded runs bit for bit through 2400, with the new tasks' code
    disabled:
    - S89's NOMEM_SELF_LITE|2800 and O_DELTA_ALLPOOL|2800, which are 2.80GHz records;
    - S89-post's O_HEBB_CONV_ALLPOOL|2801;
  - **models:**
    - the gates are one-hot at k = D on both memories;
    - the single channels have k = 1 with the right memory and conv;
    - the no-memory models' reaches are 4 and 10;
    - groups and lrs at updates 1, 2400 and 2401, with the conv in the rest group.
- **No SLOW single-channel record exists** to tie the single channels to. They are S89's single-channel construction on
  the new tasks (CHECKed).
- **Disclosure:** before the readings' commit I ran each CHECK part once by hand, plus the task CHECKs at B = 1024 while
  writing the module. Nothing was stored, and no pilot or screen arm was trained before the commit.

## Erratum (also appended to report 9)

`write_order_acc` compared the query's CTX token with the stream index. That is invalid on -allpool: they differ in 485
of 512 sequences. Report 9's -allpool write-order splits are withdrawn. `write_order_acc_idx` (9017b82) is the corrected
form, and every S90 run uses it. **G and H vendor the module, so their -allpool write-order splits are affected the same
way.**

## Runtime

11.48 h of run time on 4 workers, 05:14-08:27 UTC on 22 October:

| stage | wall clock (UTC) | runs |
|---|---|---|
| phase 0 | 05:14-05:34 | 12 |
| phase 1 | 05:34-06:07 | 24 |
| screen (in the phase1b stage) | 06:07-08:22 | 60 |

## Anything unexpected

- **The type-level gate binds on delta at a fraction of the cap.** It separates only 14% of -docs4 sequences and still
  reaches 0.96. A delta channel's capacity is two documents, not one, so a fixed map needs far fewer channels than D on
  delta.
- **Single-channel Hebbian + conv fails exactly at 1/D.** The perfect gate with the same memory binds every task, so the
  failure is the missing partition, not the memory.
- **For G (S91):**
  - -ctxpool is valid on both memories at 24000. -allpool, by contrast, needs 48000 on Hebbian + conv.
  - On -ctxpool a single delta channel binds without any gate, so on delta the partition is not needed at D = 2. It is
    needed at D ≥ 3 (0/10 at both).
