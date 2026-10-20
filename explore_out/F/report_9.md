# Session F — report 9 (round 6, S89-F: the lite task without marked keys). EXPLORATORY, not a result

Branch `claude/explore-F`.
- **Task module:** `explore_f_tasks.py` at **50dc8e4** (`50dc8e460b2c89ab6946f03b628598d7d6223d5e`). This is the commit
  G and H vendor. It was pushed before any S89 run.
- **Screen and readings:** `explore_f_s89_tasks.py`, committed at 8ba99bb before any run.
- **Setup:** Intel Xeon @ 2.80GHz; torch 2.14.0+cu130; 1 thread per run; code SHA ea9728c636c6.
- **Runs:** 28/28 ok, none retried, none diverged.
- **Seeds:** 2800-2801. The baseline block 2802-2811 was not used. The block 2800-2839 appears in no record on any branch.

Full tables are in `s89_tasks_tables.md`; the decision file is `s89_decision.json`.

## Counts first: the validity pilots (the perfect gate, k = 2, SLOW; seeds 2800 / 2801)

| task | memory | 24000: bound (Wilson 95%; band) | transitions / final acc | 48000: bound | transitions / final acc |
|---|---|---|---|---|---|
| POOL | Hebbian | **0/2** [0.00, 0.66] NEVER | unbound 0.62 / 0.48 | **0/2** [0.00, 0.66] NEVER | unbound 0.72 / 0.51 |
| POOL | delta (β 1, L2) | 2/2 [0.34, 1.00] RELIABLE | 3600 / 2400 | 2/2 [0.34, 1.00] RELIABLE | 3600 / 2400 |
| ALLPOOL | Hebbian | **0/2** [0.00, 0.66] NEVER | unbound 0.33 / 0.50 | **0/2** [0.00, 0.66] NEVER | unbound 0.32 / 0.48 |
| ALLPOOL | delta (β 1, L2) | 2/2 [0.34, 1.00] RELIABLE | 3600 / 3600 | 2/2 [0.34, 1.00] RELIABLE | 3600 / 3600 |

**READINGS (fixed in the docstring before any run): POOL "untested"; ALLPOOL "untested".**
- The rule needs the perfect gate to bind 2/2 on Hebbian and 2/2 on delta, at 24000 or else at 48000. Delta does on both
  tasks; Hebbian does on neither, at either budget.
- So **neither task is valid for this round, and G and H have no task from it.** The rule's fallback ("G and H run the
  other only") has no other task to fall back on.
- No baseline ran: phase 2 is conditional on a task reading "valid" or "needs the longer budget".

**Phase 0, positional solvability** (no memory; held-out accuracy at 24000; stop rule > 0.6). No task was stopped:

| task | NOMEM (conv on the embeddings, zero read; sees 4 tokens) | NOMEM_SELF (conv into the memory, own V read; sees 10) |
|---|---|---|
| LITE (reference) | 0.166 / 0.168 | 0.407 / 0.374 |
| POOL | 0.167 / 0.168 | 0.401 / 0.398 |
| ALLPOOL | 0.169 / 0.169 | 0.381 / 0.372 |

## The cuts (descriptive; S87's sample: 4096 sequences from torch.Generator(87_000), input tokens)

| task | CTX types (frequency) | KEY types | VAL types | largest-ratio cut | half-mass cut |
|---|---|---|---|---|---|
| LITE | 2 (0.114) | 4 (0.102-0.103) | 16 (0.022-0.024) | 2 CTX + 4 KEY (= G's S87 set) | 2 CTX + 3 KEY (mass 0.535) |
| POOL | 2 (0.114) | 32 (0.012-0.014) | 16 (0.022-0.024) | **the 2 CTX only** | 2 CTX + 12 VAL (mass 0.502) |
| ALLPOOL | 16 (0.013-0.015) | 32 (0.012-0.014) | 16 (0.022-0.024) | **the 16 VAL** (ratio 1.46) | 16 VAL + 10 CTX (mass 0.509) |

- **POOL:** the pool removes the keys from the largest-ratio cut, as expected; the cut selects only the cues.
  - The half-mass cut does not stop at the cues. The 2 CTX types hold 0.227 of the mass, so it adds 12 of the 16 value
    types to reach half.
- **ALLPOOL:** a frequency cut cannot select the cues. The values are the commonest types (8 value tokens per sequence
  over 16 types), so both cuts select them first.
  - Every type is below 1/8 (max 0.0236).
  - Anything that finds the cues on ALLPOOL must use position (a CTX token opens its block) or relation (each key appears
    under both contexts), not frequency.
- The docstring says what MQAR proper adds:
  - many documents, not two;
  - many queries interleaved with the pairs, not one at the end;
  - no context token, since MQAR's keys are unique within a sequence;
  - a far larger vocabulary and longer sequences.

## What the failure looks like (descriptive)

- **The Hebbian perfect gate routes perfectly and still does not bind.**
  - At the end of every Hebbian pilot the margin is 1.00 and η² by stream at KEY is 1.00. Each channel sees exactly one
    stream's pairs, so the failure is inside a channel: the channel does not store and retrieve its own 4 pairs.
  - The curves sit at 0.29-0.34, then some step to about 0.48-0.50. POOL 2800 alone climbs on, to 0.72 at 32400, and
    holds 0.69-0.72 to 48000.
  - Whether or not the queried pair was its key's last write, accuracy differs by at most 0.06 (POOL 2800 at 24000:
    0.65 vs 0.59).
- **The delta perfect gate binds 8/8, at 2400-3600**, on the same seeds, batches and recipe.
- On lite, S70's Hebbian perfect gate bound 3/3 (1200, 1200, 8400) under one Adam at 1e-3.
- The pilots differ from S70's oracle in two ways: the key pool (4 of 32 key types per sequence, not 4 fixed) and SLOW.
  The post-hoc diagnostic below separates them.

## Post-hoc diagnostic (written and committed at 4639d16 after the readings came in; changes no reading)

The question: does SLOW or the key pool stop the Hebbian perfect gate? Three arms, seeds 2800-2801, 24000 updates, the
S89 models unchanged (`explore_f_s89_diag.py`):

| arm | task | recipe | bound (Wilson 95%; band) | 2800 | 2801 |
|---|---|---|---|---|---|
| D_HEBB_LITE_SLOW | lite | SLOW | 2/2 [0.34, 1.00] RELIABLE | bound @2400 | bound @2400 |
| D_HEBB_POOL_ADAM | -pool | one Adam at 1e-3 (S70's oracle) | 0/2 [0.00, 0.66] NEVER | unbound 0.32 | unbound 0.33 |
| D_HEBB_ALLPOOL_ADAM | -allpool | one Adam at 1e-3 | 0/2 [0.00, 0.66] NEVER | unbound 0.31 | unbound 0.31 |

- **SLOW is not what stops it.** On lite the Hebbian perfect gate binds at 2400 under SLOW. S70's Adam oracle bound at
  1200, 1200 and 8400.
- **The key pool is.** Under S70's own Adam the Hebbian perfect gate stays at 0.29-0.34 for all 24000 updates on both pool
  tasks.
  - Routing is perfect throughout (end margin 1.00, η² 1.00).
  - These runs never reach the step to about 0.5 that SLOW gave on 3 of 4 runs at 24000 (all but ALLPOOL 2800). Here
    SLOW is the better recipe, not the stricter one.
- So both tasks read untested because of how the Hebbian channel handles 4 keys drawn from 32 types. The recipe is not
  the cause.
  - This screen does not separate which part of the channel fails: the key code over 32 types, crosstalk in a sum with
    no erase, or the query's match.
  - The delta channel binds the same task at 2400-3600.
- **Runs and CHECKs:** 6/6 runs ok, 0.68 h of run time, 19:18-19:31 UTC. explore_f_delta_checks passed 26/26 before the
  runs. The diagnostic adds no CHECK of its own; it reuses S89's models, whose CHECKs passed.

## Design decisions (stated in the docstrings before any run)

- **Pool tasks are relabellings of lite.** For a given generator state, the -pool batch is lite's batch with each
  sequence's 4 keys renamed to 4 types drawn from 32. ALLPOOL also renames the 2 CTX tokens to 2 types drawn from 16.
  Streams, blocks, positions, values and the query are lite's (CHECKed). The tasks differ from lite only in identities.
- **ALLPOOL's stream index is by first appearance.** No CTX type maps to a fixed stream, so stream 0 is the first block's
  context.
  - `stream_labels`, the query stream and `task.perfect_gate` use this index.
  - The model's built-in perfect gate maps fixed CTX ids and does not apply here. The CHECK shows that the Hebbian logits
    under `task.perfect_gate` equal the built-in gate's over the 16 CTX types bit for bit, because the same-stream
    matrix is the same.
- **"No memory" has two forms.** With the convolution in 'layer' mode, a zero read makes the model per-token, because the
  conv feeds only the memory's input.
  - NOMEM: the conv on the embeddings with a zero read; it sees 4 tokens.
  - NOMEM_SELF: the conv into the memory, with the read returning the position's own V; it sees 10 tokens.
  - Both reaches are CHECKed with randomised conv weights, since CausalConv starts at the identity. The rule stops a task
    if either form exceeds 0.6.
- **The pilots used SLOW, as the round's prompt states S70's recipe.** S70's own oracle used one Adam at 1e-3; the
  docstring calls SLOW the stricter pilot. The diagnostic shows the opposite on the pool tasks.
  - The perfect gate has no parameters, so under SLOW every parameter is at 1e-4 for updates 1-2400 and 1e-3 after.
  - The 48000 arms are fresh runs on the same seeds. SLOW does not depend on the budget, so their curves equal the 24000
    runs' bit for bit through 24000 (verified). The extra 24000 updates are a continuation in effect.
- **How "lite's batches at seed 2700 equal G's S87 first batch" was checked.** G's S87 records hold no tokens, so two
  CHECKs:
  - (i) the first training batch at seed 2700 from this module equals the one from G's vendored copy (claude/explore-G
    e46799a);
  - (ii) S70's own path on seed 2700 reproduces G's S87 `BASELINE|2700` record bit for bit through 2400: the curve and
    the gradient norms at updates 1, 10 and 100.

## CHECKs (all passed, asserted before every phase)

- explore_f_delta_checks 26/26 at the start of every segment.
- **Lite and G:**
  - lite reproduces G's S87 first batch, in both senses above;
  - S70_DELTA|330 (a 2.10GHz record) reproduces bit for bit through 2400 with S70's own path;
  - explore_f_tasks.py's first 13771 bytes (everything before the round-6 section) equal a199d5e's file.
- **Each new task (4096 sequences):**
  - lite's batch with the identities renamed;
  - key types distinct and from the pool;
  - every key exactly twice, once in one block of each stream, with two distinct values;
  - the query's (stream, key) occurs exactly once, so the answer is determined;
  - one CTX per block;
  - stream labels and the perfect gate;
  - POOL: max KEY frequency 0.0138 < min VAL 0.0220;
  - ALLPOOL: no type above 1/8 (max 0.0236).
- The frequency tables. LITE's equals G's published S87 table and cut.
- **Models:** the perfect gates are one-hot and equal to `task.perfect_gate`; the NOMEM reaches are 4 and 10; NOMEM's
  read is zero; the groups and lrs are right at updates 1, 2400 and 2401.
- **Smoke runs, not stored:**
  - the -pool and -allpool checks at B = 1024 and 2048, before the module commit;
  - 300-update runs of each arm type on seed 99, outside the block, before 8ba99bb.

## Runtime

4.05 h of run time on 4 workers, 19 October UTC. The diagnostic's runtime is in its section.

| phase | runs | wall clock (UTC) |
|---|---|---|
| phase 0 | 12 | 18:02-18:29 |
| phase 1 | 8 | 18:30-18:47 |
| phase 1b (48000) | 8 | 18:47-19:15 |
| phase 2 | none queued | 19:15-19:18 |

## Anything unexpected

- **The Hebbian perfect gate fails on both pool tasks while the delta perfect gate binds.** Routing is perfect, so the
  difference is inside a channel.
  - Renaming lite's 4 fixed keys to 4 of 32 is enough to cause it. Per the diagnostic this holds under both recipes, and
    SLOW binds lite.
  - The validity rule asks both memories to bind, so it is the Hebbian half that leaves both tasks untested.
- **On ALLPOOL the largest-ratio cut selects the values, not nothing.** No role is frequent, but the values are still the
  commonest types.
- **About 0.4 without memory, lite included.** A model that sees the last 10 tokens reaches 0.37-0.41 on all three
  tasks. The renaming does not change positional solvability: a pair inside the conv's reach can be answered without
  memory. This is below the 0.6 stop. It also means any accuracy near 0.4 on these tasks is not evidence of memory use.

---

# Addendum (20 October): referee V's fixes and the follow-up pilots

Referee V reviewed S89-F at 8ba99bb (`docs/review_2026-10/screen_review_s89_F.md`, claude/review-V 3ae815b) and gave the
verdict "fix first". The pilots and the diagnostic stand as run, and **no reading above changes**. The S89 records stay
under code SHA ea9728c636c6, and S89's queue is closed.

**This container is a Xeon @ 2.10GHz again.** The 2.80GHz S89 records reproduce bit for bit here: two pilots through 2400,
by the revised path. So every run below is on 2.10GHz, and the comparisons with the S89 pilots are for reference only,
not pairs.

## Counts first: the follow-up pilots (seeds 2800 / 2801; Hebbian, k = 2, SLOW, 24000)

| arm | bound (Wilson 95%; band) | 2800 | 2801 | the S89 pilot on the same seeds (no conv, relative gate) |
|---|---|---|---|---|
| O_HEBB_CONV_POOL (perfect gate + conv 'layer') | 2/2 [0.34, 1.00] RELIABLE | bound @4800 | bound @13200 | unbound 0.62 / 0.48 |
| O_HEBB_CONV_ALLPOOL (perfect gate + conv 'layer') | 1/2 [0.09, 0.91] MAJORITY | bound @6000 | unbound 0.87 | unbound 0.33 / 0.50 |
| conv pilots together | **3/4** [0.30, 0.95] MAJORITY | | | 0/4 |
| O_TYPE_HEBB_ALLPOOL (type-level perfect gate, no conv) | 0/2 [0.00, 0.66] NEVER | unbound 0.28 | unbound 0.28 | (relative gate: 0.33 / 0.50) |

**READING (post hoc; hypothesis and readings committed at 3dd5b74 before the runs): "partly"** (3/4 bind). "Restores"
needed 4/4 and "it does not" 2 or fewer.

**The conv pilots (descriptive).**
- Every conv run routes perfectly: end margin 1.00, η² by stream 1.00 in both forms, VAL cos 0.00 in both forms.
- The three bound runs reach 0.96-0.99, against 0.32-0.72 without the conv at 24000 and 48000.
- The unbound run, ALLPOOL 2801, rises 0.67 → 0.78 by 7200, holds about 0.77 to 20400, then climbs to 0.87 at 24000. It
  was still rising when the budget ended.
- Accuracy is the same whether or not the queried pair was its key's last write (0.86 / 0.87 on ALLPOOL 2801).
- **The convolution is the only difference** from the S89 pilot. It starts as a no-op: at initialisation the conv model's
  logits equal the S89 pilot's bit for bit, its shared parameters are equal, and the conv weights are its only extra
  parameter (CHECKed).
- So adding the trainable convolution on the memory's input turns a Hebbian channel that cannot use 4 keys out of 32 into
  one that binds on 3 of 4 seeds.
  - This fits the hypothesis: H's Hebbian perfect gate with the conv bound the eight-stream grouped layout with the same
    pool 2/2. The delta memory binds these tasks without the conv (S89: 2400-3600).
  - Why the conv helps is not separated here. One candidate is the short-convolution account from associative recall (H3,
    Based): with the conv, the write at a VAL position carries its KEY one token back in the same layer. So the
    key-value association needs no earlier layer to copy the key forward.

**The type-level gate (fix 2's fifth pilot; descriptive).**
- Both runs stay at 0.25-0.28 for all 24000 updates, below the relative gate's 0.33 / 0.50 on the same seeds.
- Within each sequence the statistics do what the cap predicts:
  - η² by stream 0.51 / 0.49, equal to the fraction of probe sequences the balanced map separates (0.51 / 0.49);
  - margin 0.51 / 0.49;
  - VAL cos 0.49 / 0.51.
- Pooled, they read η² 0.00 / 0.01 and VAL cos 1.00 / 0.99. Those are the labelling, not the routing.
- As stated before the run, a Hebbian channel without the conv is unbound under the relative gate too. So this pilot
  cannot separate the 0.533 cap from the memory. A type-level gate with the conv, or on delta, would; it was not run.

## The three fixes taken (in the review's order)

1. **Routing statistics invariant to a per-sequence channel swap** (`explore_f_tasks.py`, appended).
   - `pool_routing_stats_inv` / `pool_stats_inv` record η² by stream at KEY and the VAL cosine within each sequence,
     averaged over the probe (`eta_key_by_stream_inv`, `val_cos_inv`, `seq_routed_frac`), beside the pooled forms.
   - `pool_outcome(rec, invariant)` reads ROUTED*@end, ROUTED-BOUND and DISCOVERED from either form. On ALLPOOL the
     within-sequence form is read; POOL records and prints both.
   - The synthetic-gate CHECK passes on both tasks:
     - the relative gate, the same gate with the channels swapped in a random half of sequences, and a gate that sends
       each sequence's lower CTX type to channel 0 all score within-sequence η² 1 and cos 0;
     - on ALLPOOL the last two score pooled η² 0.0000 / 0.0015 and VAL cos 1.000 / 0.997;
     - the balanced type-level gate's within-sequence η² equals the probe's separated fraction exactly (0.5117 on 256
       sequences);
     - the vectorised per-sequence η² equals `test_router_layout.eta2` on each sequence (max |diff| 0).
   - The S89 pilots' gates are relative, so for them the two forms coincide (1.00 / 0.00 at 2400, CHECKed).
2. **ALLPOOL's routing cap is stated** in the S89 docstring and the module.
   - No fixed context-type → channel map separates more than 64/120 = 0.533 of sequences.
   - So a valid relative perfect gate on ALLPOOL says the memory can use a relative routing, not that a gate under test
     can express one.
   - The type-level perfect gate ran as the fifth pilot (above).
3. **The screen's own path is tied to a recorded run, and the references are read at commits.**
   - `explore_f_s89_tasks_child.run` on BASELINE_LITE, seed 2700, reproduces G's S87 `BASELINE|2700` (read at e46799a,
     not the branch tip) bit for bit through 2400. That covers the curve, the gradient norms at updates 1, 10 and 100,
     and all 19 statistic fields this path records at 0, 1200 and 2400. G's record also holds `beta` and `layers`, G's
     own additions, which this path does not compute.
   - S70's record is read at 61ba46e.
   - The revised path also reproduces O_HEBB_POOL|2800 and O_HEBB_ALLPOOL|2800 (read at 52e2ea7) bit for bit through
     2400.
   - The docstring now says the gradient norms at 10 and 100 are functions of the first 10 and 100 batches.

**Not taken:** the review's fixes 4-10 concern phase 2 (the baselines' role, failed records, label-free gates, v2
classes, lite-id statistics) or G's and H's arms. Phase 2 did not run.

**The module for G and H is `explore_f_tasks.py` at b0c6332** (`b0c6332b766b833df83826e36a3b8c2d9b58e583`).
- It is append-only: its first 34603 bytes are 50dc8e4's file (CHECKed), so both classes are unchanged.
- The new helpers are `pool_stats_inv`, `pool_outcome`, `type_level_gate` / `BALANCED_MAP` and `check_inv_stats`.

## CHECKs, disclosure, runtime

- **CHECKs, asserted at the queue's start, all passed:**
  - explore_f_delta_checks 26/26;
  - S89-F's 52 CHECKs: tasks, g87, s70, models, inv;
  - the follow-up child's 6:
    - the type gate is BALANCED_MAP's one-hot, with the S89 pilot's parameters;
    - the conv arms are the S89 pilot plus a conv at the identity: equal logits at init, the conv the only extra
      parameter;
    - groups and lrs at updates 1, 2400 and 2401, with the conv in the rest group.
- **Before the commits** I ran each CHECK part once by hand: S89's five parts and the follow-up child's. No pilot arm was
  trained outside those CHECKs before the runs.
- **The queue started on a dirty tree** (git 3dd5b74+dirty). The uncommitted files were the queue script and the
  keepalive, with no code; both were committed at 4c8a343. The later `+dirty` tags in the log are progress files.
- **Runtime:** 0.50 h of run time, 21:47-22:02 UTC on 20 October, CHECKs included. Code SHA 267d08456c38. Tables:
  `s89_post_tables.md`.

## What this leaves (descriptive)

- No Hebbian configuration is valid under S89's rule, whose pilots had no convolution, so the baselines stay unrun.
- With the convolution, the Hebbian perfect gate binds -pool 2/2 and -allpool 1/2 in 24000; the unbound run reached
  0.87 and was still rising.
- **Whether a conv configuration should be this round's valid Hebbian one is the user's decision.** If so:
  - it would need a validity reading of its own, fixed before runs;
  - -allpool would likely need a longer budget;
  - the ALLPOOL cap still applies to any learned gate that reads context identity.
