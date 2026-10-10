# DIFF.md — my implementation against the main line's

Independent replication, EXPLORATORY, not a result.

**What was compared.** Written in Phase 3, after the Phase 2 records were pushed (`de8df0f`).
- Mine: `replicate.py` at `de8df0f`.
- Theirs: branch `claude/bdh-growth-hebbian-inference-w90069` at `404e3a0`. The arms are WIN3_SLOW_D8, WIN3_SPLIT_D8 and CEIL_C, which `test_window_gate.py` builds through `test_stream_curriculum.run_sc` (D8) → `test_router_reliability.run_one` → `test_binding_onset.onset_run`. The model is `test_multilayer_binding.MultiBDH`, converted by `to_local`, on `test_binding_capacity.BindTask`. I read `README.md`, `test_window_gate.py` and the parts of its imports that the D8 path executes: `bdh.py`, `test_multilayer_binding`, `test_binding_onset`, `test_binding_capacity.BindTask`, `test_instrument_v2` (constants, `decay_mask`), `test_stream_curriculum`, `test_stream_recipe` (`builder_for`, `outcome`, `routed`), `test_scale_axes` (`routing_k`, `stream_acc`, `scale_stats`), `test_router_reliability.run_one`, `test_router_discovery.model_fn`, `test_slow_start` (`make_recipe`, `part_run`, constants) and `test_channel_binding`.
- I did **not** read the inherited CHECK chain (CHECKs 1–134), the report printing, or the HINGE/WIN16_A paths.

**Mechanical check.** I built their D8 window-gate model, loaded my weights into it (vocabulary permuted) and compared outputs:
- gates agree to 7×10⁻⁹;
- query logits agree to 7×10⁻⁸.

Their printed config is `n_layer=3, n_embd=32, n_head=1, mult=8 (N=256), dropout 0, decay 0.95, conv "layer" k=4, h_gate 32, k=16`. The architecture, gate, conv, init distributions, SLOW groups and switch, batch, evaluation and the outcome definition all match my reading. Every difference below is in the protocol.

**Labels.**
- **unstated**: the report does not determine it.
- **misread**: stated in the report, and I misread it.
- **contradicts**: the code contradicts the report.

I found **no "contradicts"** and **no "misread"** in the paths I read. One row (D-17) does not fit the three labels: the report states it and I did not match it, because the exact version was not installable.

## Differences that can change a run's outcome (systematic)

| # | What | Theirs | Mine | Label | Expected effect |
|---|---|---|---|---|---|
| D-1 | The split's 64-sequence probe | **One probe for every seed and arm**: `BindTask.make_batch(64, Generator(12345))` | A probe drawn per run from `Generator(seed+1000003)` (U-36) | unstated ("a fixed 64-sequence probe" admits both) | Changes when the trigger fires. My seed 914 failed because its probe read ≥ 0.95 while the 2048-query evaluation sat at 0.88 with two streams merged, so the rule stopped firing. **Rerun R1.** |
| D-2 | Positions for "mean read-gate mass at key positions" | The 32 body key positions (`3j+1`) | 33: the body keys plus the query's KEY (U-38) | unstated | Can change c* or c0 when masses are close. **Rerun R2.** |
| D-3 | Perfect gate (validity) training | Single Adam at 1e-3 throughout (no SLOW); arm SC8_ceil with context padding to 16 channels, curriculum off | SLOW schedule (1e-4 for updates 1–2400) (U-28) | unstated | Validity only. It explains my later oracle transitions (6000 and 10,800 against their 3600–6000). No effect on the split comparison. |
| D-4 | Order at an evaluation step | Evaluation → statistics → path check → LR switch → **trigger** → early-stop test | Evaluation → early-stop test → probe/trigger (U-30) | unstated | Theirs can split at the step a run stops, before the final statistics. In my 20 SPLIT runs, 5 stopped on a probe step, after three evaluations ≥ 0.95, where a probe below 0.95 is very unlikely. Practically inert. |
| D-5 | Last split check | `total − 2400` = 40,800 | Up to 43,200 | unstated | Inert here: my latest split was at 36,000. |

## Differences in which random draw is taken (same distribution)

| # | What | Theirs | Mine | Label |
|---|---|---|---|---|
| D-6 | Training batches | `Generator(seed + 10000)` | `Generator(seed)` | unstated |
| D-7 | Held-out set | 2048 sequences from `Generator(424242)` | 2048 from `Generator(123457)` | unstated |
| D-8 | Initialisation | `torch.manual_seed(seed)`; Pathway's BDH draws (decoder, encoder, encoder_v, lm_head, embedding at 0.02); then W_in, **W_h** (drawn, frozen, unused), W_g at 0.1; the window from `Generator(seed + 31337)`, U(±1/√3) | `manual_seed(seed)`; embedding, encoder, encoder_v, decoder, lm_head, then the window (PyTorch default init, the same U(±1/√3)), W_in, W_g. No W_h | unstated |
| D-9 | Split noise | Fresh `Generator(12_000_000 + 1000·seed + n_splits)` per split; c*'s noise drawn first | One `Generator(seed+2000003)` per run; c0's noise drawn first | unstated |
| D-10 | Stream→channel map | Mean read gate per stream at the body's value positions on a **256-sequence probe** (`Generator(seed+99000)`); argmax | The same statistic on the 2048 held-out sequences (U-32) | unstated |

## Differences with no effect on the forward pass

| # | What | Theirs | Mine | Label |
|---|---|---|---|---|
| D-11 | Vocabulary and sequence | 28 tokens (CTX 0–7, KEY 8–11, VAL 12–27). The sequence is the 99 tokens including the query's VAL; the model input is the first 98 | 29 tokens (VAL 0–15, KEY, CTX, and a `?` token). Input of 99, ending in `?` (U-17) | unstated (the report's "[CTX_q, KEY_q, ?]" is notation, not a token) |
| D-12 | Window tap order | `conv_w[:, 0]` = current token | Conv1d last tap = current token | — (equal by construction; checked numerically) |
| D-13 | Recurrent W_h | Kept in the model, frozen, in no optimizer group | Absent | unstated (inert) |
| D-14 | Failure labels | η²-based `fail_class_k`; MERGED (j share) | Map-based simplification (U-35) | unstated (diagnostic only) |
| D-15 | "Bound" | `transition()`: the first evaluation ≥ 0.95 that holds to the end | The same, written as "≥ 3 trailing or at the budget" | — (equal under early stopping) |
| D-16 | Extra statistics | `test_slow_start.extra_stats` and `scale_stats` at every evaluation (no state change) | None | — |
| D-17 | PyTorch | 2.14.0 (report, Table 5) | 2.14.1 (the version pip resolved here) | stated in the report; not matched (unavailable), not misread |

## Points where my reading of the report matched theirs

These are worth confirming, because the report leaves them open:
- Scalar decay replaces RoPE.
- The gate reads the model's raw embedding.
- The window gate's H is 32 with N(0, 0.1²) init and no biases.
- The width-4 conv is **one shared** filter on the D-dim layer input, identity-initialised, feeding Q=K and V (U-14/15).
- Adam has default betas and no weight decay. The SLOW groups put the embedding and the conv in the slow group. The switch happens after the evaluation at 2400.
- The loss and accuracy are cross-entropy and argmax over the full vocabulary at the query's KEY position.
- The probe is also read at 2400 as the reference for the first check at 4800 (U-37).
- `W_g`'s Adam state is zeroed, `step` included.
- The noise is 0.1·std of the copied row, independent for each row.
- The ≥ 4800 gap and the cap of 3 match.
- Bound routed = bound + one-to-one map at value positions + every stream ≥ 0.9.

## Corrections in the main line's README (Revision 8.1) that bear on the PDF I worked from

- **"One channel per stream" is "by largest mass".** The README's corrected Section 3 and its §5 text both say so: the streams sit on disjoint channel sets, one per stream by the largest mean gate mass. My U-32 map, an argmax of the mean gate, is that definition. The v8 PDF's wording ("every stream has its own channel") also admits a hard per-token routing reading.
- **The PDF's "conv" paragraph is withdrawn.** The PDF says the perfect gate without the conv failed at two streams. Revision 8.1 says those runs used a different, smaller model (N = 64), and that whether the conv carries part of the binding is open. This does not change the eight-stream configuration, which uses the conv in both versions.
- **Seed ranges.** The PDF says "X 500–559, L 1500–1559" for the test as a whole. The eight-stream parts used 540–559 and 1540–1559 (`test_window_gate.RANGES`). The PDF does not give the per-part split.

## Reruns (Phase 3; WIN3_SPLIT_D8 on seeds 900–909, one discrepancy fixed at a time)

Chosen as the two systematic differences in the split itself (D-1, D-2). The others were not rerun:
- D-3 touches only the oracle.
- D-4 and D-5 are inert in the Phase 2 records.
- D-6 to D-10 only re-draw random numbers, so a rerun would measure seed noise, not the discrepancy.

| Rerun | Change | Bound routed on 900–909 | vs Phase 2 (9/10) |
|---|---|---|---|
| R1 | D-1: their probe (`Generator(12345)`, token-identical port) | see `replicate_report.md` | |
| R2 | D-2: key mass over body keys only | see `replicate_report.md` | |
