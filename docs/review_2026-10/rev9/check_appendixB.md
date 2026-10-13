# Revision 9, Appendix B ("What a replication needs") checked against the main line's code and records

Draft: `origin/claude/rev9-draft` at 9bfa85a, `multichannel_hebbian_report_v9.tex` (`tex:NNN`).
Code and records: `origin/claude/bdh-growth-hebbian-inference-w90069` at 32c121b (read with `git show`). The path
checked is `test_window_gate.py`'s Part C/D path: `one_run` -> `run_path` -> `test_slow_start.part_run("C")` ->
`test_stream_curriculum.run_sc(ARM["D8"])` -> `test_router_reliability.run_one` -> `test_binding_onset.onset_run`,
with the model `test_multilayer_binding.MultiBDH` (on `bdh.BDH`) converted by `test_window_gate.to_local`.
Helper: `docs/review_2026-10/rev9/check_appendixB.py` (read-only; reads the records through `git show`, prints the
check-row fields, the probe/evaluation comparison, the meta's start time and the split probe's per-stream counts).
One read-only model build was run (Part C model, seed 540, and W_h for seeds 540-559) to print shapes and
initialisations; nothing was trained.

Abbreviations: twg = test_window_gate.py, tmb = test_multilayer_binding.py, tbo = test_binding_onset.py,
tbc = test_binding_capacity.py, trr = test_router_reliability.py, tsa = test_scale_axes.py, tsr = test_stream_recipe.py,
tscur = test_stream_curriculum.py, tss = test_slow_start.py, tiv = test_instrument_v2.py, tcb = test_channel_binding.py,
tsc = test_short_conv.py.

## Counts

| verdict | items |
|---|---|
| FALSE | 3 |
| PARTIAL | 2 |
| UNVERIFIABLE | 3 |
| TRUE | 79 |

## Table (FALSE and PARTIAL first)

| # | tex line | item | verdict | code evidence (file:line) | note |
|---|---|---|---|---|---|
| 1 | tex:527 | Records: "the start time with its time zone" | FALSE | twg:1340 `time.strftime("%Y-%m-%d %H:%M:%S")`, stored at twg:1343/1346-1347 | No zone in code or records: X `meta.started = "2026-10-07 23:30:25"`, `starts[0].started = "2026-10-08 11:48:32"`; L `meta.started = "2026-10-08 11:33:10"`. |
| 2 | tex:205 | "How often the main line's checks were blocked this way cannot be read from its records, which store neither the probe's reading nor the per-stream masses at each check" | FALSE | twg:524-530 (`row = dict(step, acc=<probe accuracy>, prev, map_before, key_masses, all_masses, ...)`), twg:546 (`info["checks"].append(row)`), twg:591 (`rec["checks"]`); per-stream masses: tsa:292-297 (`stream_gate`, per stream x channel mean gate at value positions) in `scale_stats` (tsa:316-322) at every evaluation, trr:311-312 | Every `checks` row of WIN3_SPLIT_D8 in results/X and results/L holds `acc` (the 64-sequence probe's accuracy) and `prev`; 393 rows (X 216, L 177). Every check step also has `stats[*].stream_gate` (8 x 16, 256-sequence probe, value positions) and the row holds the per-channel key masses on the 64 probe (`key_masses`, `all_masses`) and the labelled map (`map_before`). The count can be read: 6 of 393 checks had probe >= 0.95 while the 2048-query evaluation at the same step was < 0.95 (X 1: s552@24000; L 5: s1550@14400, s1553@16800, s1553@19200, s1554@19200, s1554@21600), in 4 runs. 4 of the 6 would not have fired anyway (gap/cap block, or the probe's rise was >= 0.02); 2 were blocked by the probe alone (L s1553@19200, probe 0.953 vs eval 0.867, streams 0 and 2 merged on channel 1; L s1554@21600, probe 0.969 vs eval 0.869, streams 1 and 6 on channel 13). Both fired at the next check (21600, 24000) and both runs ended BOUND ROUTED. WIN3_RESET_D8: 0 of 145 rows. |
| 3 | tex:384 | "Neither was noticed in the project's own runs because the records do not store the probe's readings" | FALSE | twg:524-527, 546, 591 | Same records as row 2: the probe's reading is stored at every check (`checks[*].acc`). Whether anyone noticed is not a code question; the causal clause is false. |
| 4 | tex:516 | Model forward pass written as x = LN(emb); x_s = ReLU(xE); y = LN(attn(x_s, x_s, x)); y_s = ReLU(yE_v) | PARTIAL | bdh.py:117-146; tmb:303-320 (the conv path used here) | Each listed step is right, but the layer does not end at y_s: `xy_sparse = x_sparse * y_sparse` (tmb:315, bdh.py:136), `yMLP = xy @ decoder` (tmb:317), `y = LN(yMLP)` (tmb:318), `x = LN(x + y)` (tmb:319), logits `x @ lm_head` (tmb:320). So there is the BDH product with x_s and a LayerNormed residual. "Pathway's BDH forward pass" covers these only by reference; the formulas as written stop early. With the 'layer' conv, x_s = ReLU(conv(x)E) and V = conv(x) (tmb:308-311); the residual is not convolved. |
| 5 | tex:516 | score (x_t . x_s)(g_t . g_s) 0.95^{t-s} | PARTIAL | tmb:311 (`attn(Q=x_sparse, K=x_sparse, V=xc)`), tmb:216-219, tiv:243-247, twg:340 | The structure is right (product of the dot product, the gate overlap g_t.g_s and the decay, for s < t). The vectors in the dot product are the sparse codes ReLU(conv(x)E) (N = 256), not the layer input x. The text's "x_s" is defined as the sparse code and is then reused as "x at position s", which is ambiguous. |
| 6 | tex:524 | "(none is needed at k = 16, S = 8)", the reason no guard is used | UNVERIFIABLE | twg:531-532 (no guard in code) | This is a claim about need and cannot be checked in code. Supporting evidence: twg docstring 166 and 188 record every target as labelled ok (X 47 splits, L 38). |
| 7 | tex:529 | for a replication: one-sided exact test against the reported rate with size and power stated in advance, a Fisher comparison, the arms paired on seeds | UNVERIFIABLE | Fisher exists at test_router_confirm.py:134; no rate-vs-reported-rate test on the Part C/D path | This prescribes what a replication should do. It does not describe the main line's code. |
| 8 | tex:122 | "session H's exact copy does neither" | UNVERIFIABLE | not on the main line | Session H's code is on another branch and outside this scope. |
| 9 | tex:516 | x = LN(emb) | TRUE | tmb:303-306; bdh.py:117-120 | |
| 10 | tex:516 | x_s = ReLU(xE) | TRUE | tmb:308-310 | The input is the conv output xc here (see row 37). |
| 11 | tex:516 | y = LN(attn(x_s, x_s, x)) | TRUE | tmb:311-312 | V = xc |
| 12 | tex:516 | y_s = ReLU(yE_v) | TRUE | tmb:313-314 | |
| 13 | tex:516 | one head | TRUE | tmb:253 `n_head=1` | |
| 14 | tex:516 | N = 256 | TRUE | tcb:137 `SUB_MULT = 8`; D8 arm `mult: 8` (tscur:258,263 via tsr:270); tmb:253-254 | The built model has encoder (1, 32, 256). |
| 15 | tex:516 | d = 32 | TRUE | test_interference.py:36 `D = 32` -> tmb:157,253 | |
| 16 | tex:516 | three applications of one shared layer | TRUE | tcb:136 `ARCH = dict(n_layer=3, ...)`; tmb:307 loop reuses encoder/encoder_v/decoder/short_conv | |
| 17 | tex:516 | scalar decay 0.95^{t-s} instead of RoPE | TRUE | tiv:124 `OPERATING_DECAY = 0.95`; tiv:243-247; tmb:215-216 (positional "decay", tcb:136) | |
| 18 | tex:516 | strictly causal s < t | TRUE | tmb:219 `tril(diagonal=-1)`; tiv:247 | |
| 19 | tex:516 | no dropout | TRUE | tmb:253 `dropout=0.0` | |
| 20 | tex:516 | encoder, decoder, head, embedding N(0, 0.02^2) | TRUE | bdh.py:85-107 | encoder_v is also 0.02 (bdh.py:93). The embedding is set by `_init_weights` (bdh.py:106-107). Built stds at seed 540: 0.020/0.020/0.020/0.020. |
| 21 | tex:516 | which sets the scale of the gate's input | TRUE | twg:339 (gate input `self.embed(tokens)`) | |
| 22 | tex:516 | one gate per token for read and write | TRUE | twg:332-336 `return g, g` | |
| 23 | tex:516 | the same gate at every layer | TRUE | twg:340 (`attn.G` set once per forward), tmb:217-218 | |
| 24 | tex:516 | gate reads the model's own raw embedding (same table) | TRUE | twg:339 | No LN: `gate_ln` is False (asserted twg:356). |
| 25 | tex:516 | width-3 depthwise causal convolution | TRUE | twg:256 `WIDTH = 3`, twg:359; tmb:225-242 | |
| 26 | tex:516 | weights uniform in +-1/sqrt(3) | TRUE | twg:348-351 | Drawn from `Generator(seed + 31_337)` (twg:256,350), not from the global RNG. The appendix does not say this; it matters only for bit-for-bit replication. |
| 27 | tex:516 | no bias | TRUE | tmb:230-234 (CausalConv has only conv_w) | |
| 28 | tex:516 | left-padded with zeros | TRUE | tmb:238 `F.pad(x, (0, 0, K - 1, 0))` | |
| 29 | tex:516 | hidden layer of 32 units with tanh | TRUE | tiv:125 `H_GATE = 32`; twg:334 | |
| 30 | tex:516 | W_in, W_g ~ N(0, 0.1^2) | TRUE | tmb:266, 268 | W_h (tmb:267) is drawn between them and then frozen and unused (twg:362). This affects the RNG stream only. The appendix does not say it. |
| 31 | tex:516 | no biases (gate) | TRUE | twg:334-335; assert twg:497 | |
| 32 | tex:516 | the conv is one shared depthwise causal filter of width 4 | TRUE | tsc:212 `CONV_K = 4`, tsc:222; tmb:291-292, 308 | |
| 33 | tex:516 | on the d-dimensional layer input | TRUE | tmb:308 | |
| 34 | tex:516 | initialised to the identity | TRUE | tmb:232-233 | |
| 35 | tex:516 | output feeds keys (= queries) and values | TRUE | tmb:308-311 | |
| 36 | tex:518 | 28 tokens: 16 values, P = 4 keys, S = 8 contexts, no query token | TRUE | tbc:214 `vocab = S + P + n_vals`; tbc:249-250 (the query reuses context and key ids) | |
| 37 | tex:518 | sequence of 99 tokens ends with the query's value | TRUE | tbc:216 `L = 3*(S*P + n_q)` = 99; tbc:249-251 | |
| 38 | tex:518 | reads the first 98, predicts the 99th from the query's KEY position | TRUE | tbc:218 `qpos = [97]`; tbo:226-227; tbc:327-330 | |
| 39 | tex:518 | per sequence and key, S distinct values without replacement, assigned to streams in random order | TRUE | tbc:241-245 | |
| 40 | tex:518 | query's stream and key uniform and independent | TRUE | tbc:247-248 | One draw from the S*P pairs. |
| 41 | tex:518 | cross-entropy over the full vocabulary at the query's KEY position | TRUE | tbo:226-228; tbc:327-330 | |
| 42 | tex:518 | accuracy argmax over the full vocabulary | TRUE | tbo:187 | |
| 43 | tex:518 | 2048 held-out sequences, fixed generator, same for every run and arm | TRUE | tbo:115-116, 177-179 (`EVAL_SEED = 424_242`, `2048 // n_q`, n_q = 1 at tcb:139) | |
| 44 | tex:518 | torch.manual_seed(seed) before initialisation | TRUE | tbo:207 (and tmb:525 in `build`) | |
| 45 | tex:518 | training batches from Generator(seed + 10000) | TRUE | tbo:215, 225 | |
| 46 | tex:520 | Adam, default betas, no weight decay | TRUE | tbo:213; twg:438-441 (the capturing factory passes only lr) | |
| 47 | tex:520 | batch 32 | TRUE | test_interference.py:38 -> tbo:108, 225 | |
| 48 | tex:520 | constant rates, no warm-up | TRUE | tbo:193-195 (`warmup=0`, `lr_at=None`); only SLOW's switch changes a rate | |
| 49 | tex:520 | no clipping | TRUE | tbo:216-243 (no clip anywhere on the path) | |
| 50 | tex:520 | SLOW's groups and its switch after the evaluation at 2400 | TRUE | twg:257, 367-378, 404-409; tss:239-241 | |
| 51 | tex:520 | evaluation every 1200 and at the budget | TRUE | tbo:112, 235 | The code has no separate final evaluation. The budget 43200 = 36 x 1200 falls on the grid. |
| 52 | tex:520 | early stop after three consecutive evaluations >= 0.95 | TRUE | tbo:113-114, 240-243 | D8 has no `early_stop_after` (tscur:345 run_kw None) |
| 53 | tex:520 | order at a shared step: evaluation, statistics, path's check, lr switch, split trigger, early-stop test | TRUE | tbo:236-243; trr:311-314; twg:404-409 (path check then switch); twg:550-560 (trigger after both) | |
| 54 | tex:520 | perfect gate at single rate 1e-3, without SLOW | TRUE | twg:419-425, 583 (rc None); tscur:236 `LR = SUB_LR` = 1e-3 (tcb:138) | |
| 55 | tex:520 | final per-stream accuracy and map from a final evaluation of the final model | TRUE | trr:324-330 (`stats_fn(m, ..., "end")`); tsa:322-324 | |
| 56 | tex:522 | Bound = first evaluation >= 0.95 that holds to the end; can bind on its last evaluation | TRUE | tbo:247-255 | |
| 57 | tex:522 | map: per stream, argmax over channels of the mean gate at body value positions | TRUE | tsa:292-294 | |
| 58 | tex:522 | on a 256-sequence probe Generator(seed + 99000) | TRUE | tmb:352-357; tiv:132 `PROBE_B = 256`; trr:303 | |
| 59 | tex:522 | ties on unrounded values | TRUE | tsa:294 (argmax of unrounded means; rounding only for the stored `stream_gate`, tsa:297) | |
| 60 | tex:522 | bound routed = bound, one-to-one, every stream's held-out accuracy >= 0.9 | TRUE | tsr:250, 751-760; tsa:296, 303-313 | |
| 61 | tex:524 | probe: 64 sequences, BindTask.make_batch(64, Generator(12345)), the same for every seed and arm | TRUE | twg:299, 449-450, 517, 577 (task TASK48 for every seed) | |
| 62 | tex:524 | scored on its queries | TRUE | twg:524; tbo:183-189 | |
| 63 | tex:524 | 5, 5, 6, 8, 10, 12, 5, 13 queries per stream | TRUE | computed by check_appendixB.py --code | [5, 5, 6, 8, 10, 12, 5, 13] |
| 64 | tex:524 | read at 2400 as the reference; checks from 4800 every 2400 to budget - 2400 | TRUE | twg:301, 522, 531, 577 (`last = iters_c - EVERY` = 40800) | |
| 65 | tex:524 | fire if probe < 0.95 and rose < 0.02 since previous check | TRUE | twg:300, 531 | `prev` is updated at every check, blocked ones included (twg:545). |
| 66 | tex:524 | c*, c0 by mean read-gate mass over the probe's 32 body key positions (not the query's) | TRUE | twg:463-468 (`j = arange(S*P)`, positions 3j+1) | |
| 67 | tex:524 | W_g[c0] <- W_g[c*] | TRUE | twg:500-504 | |
| 68 | tex:524 | Part D adds 0.1 sigma noise to both rows, sigma = copied row's std | TRUE | twg:302, 477, 502-504 | |
| 69 | tex:524 | c*'s noise drawn first, Generator(12_000_000 + 1000*seed + n_splits) | TRUE | twg:478-480, 538 | |
| 70 | tex:524 | zeroes W_g's Adam state including its step counter | TRUE | twg:484-492, 505 | Records: every fired row has `state_zeroed = [exp_avg, exp_avg_sq, step]`. |
| 71 | tex:524 | a split blocks further splits for 4800 updates | TRUE | twg:303, 532 | |
| 72 | tex:524 | at most three | TRUE | twg:303, 532 | |
| 73 | tex:524 | no guard against a target that carries key mass | TRUE | twg:531-532 | |
| 74 | tex:527 | curve at every evaluation | TRUE | tbo:237; record `curve` | |
| 75 | tex:527 | per-stream gate mass at value positions and per-stream accuracy at the end | TRUE | records `end.stream_gate`, `end.stream_acc` (tsa:297, 324) | `stream_gate` is also stored at every evaluation (`stats[*]`). |
| 76 | tex:527 | each check's probe reading and the evaluation's reading at the same step | TRUE | `checks[*].acc` (twg:527); `curve` at the same step | |
| 77 | tex:527 | each split's update, c*, c0 and the map before it | TRUE | `checks[*].step/key_cs/key_c0/map_before` (twg:527-530) | Also `map_after`, noise seed and sd, and row distances. |
| 78 | tex:527 | the CPU | TRUE | twg:587 (`rec.cpu`), twg:1341 (`meta.cpu`) | |
| 79 | tex:527 | the code commit | TRUE | twg:1295, 1341, 1346-1347 | X: `meta.git = 2f98f06` (first start), `starts[0].git = 886a668` (Part D). L: 886a668. |
| 80 | tex:529 | exact one-sided McNemar, p = P[Bin(b+c, 1/2) >= b] | TRUE | test_conv_lr.py:260-266 | |
| 81 | tex:529 | Wilson 95% with z = 1.96 | TRUE | tsr:332 (`z=1.959963984540054`); used at twg:1036 | |
| 82 | tex:87 | recurrent gate W_in, W_h, W_g ~ N(0, 0.1^2); rho(W_h) ~ 0.6, sigma_max ~ 1.08 | TRUE | tmb:266-268 | Measured on Part C builds, seeds 540-559: rho mean 0.606 [0.554, 0.678], sigma_max mean 1.083 [1.008, 1.168]. |
| 83 | tex:92 | window gate: width-3 depthwise causal over embeddings, uniform +-1/sqrt(3), no bias, no recurrence | TRUE | twg:332-336, 348-351; tmb:225-242 | W_h is frozen and unused (twg:362). |
| 84 | tex:120 | SLOW: gate group at 1e-3 throughout; everything else (embedding, conv included) at 1e-4 for 1-2400, then 1e-3; perfect gate single 1e-3 | TRUE | twg:257, 367-378, 404-409; tss:239-241; twg:583 | |
| 85 | tex:122 | SPLIT description (checks, trigger, KEYMASS c*/c0, cap 3, gap 4800, noise and Adam reset, probe 64 at 12345 scored on queries, 5,5,6,8,10,12,5,13, no stream labels, needs key positions) | TRUE | as rows 61-73; twg:463-468 (positions only; stream labels only in the logged map, twg:526) | |
| 86 | tex:114 | map from a 256-sequence probe, ties resolved on unrounded values | TRUE | tsa:292-297; tmb:352-357 | The counts in the same item (38 runs, 0.047, 18) were not checked here. |
| 87 | tex:112 | a run that reaches its budget counts as bound on a single final evaluation | TRUE | tbo:247-255 (`transition` takes the last evaluation alone if it is >= 0.95) | The "14 of 1869" count is left to others. |

(87 rows: 1-3 FALSE, 4-5 PARTIAL, 6-8 UNVERIFIABLE, 9-87 TRUE.)

## Omissions noticed (not verdicts)

- The window's own generator `seed + 31_337` (twg:256, 350) and W_h's draw (frozen and unused) are not stated. Both
  matter only for bit-for-bit reproduction.
- The forward pass leaves out the x_s * y_s product, the decoder and the LN residual (row 4).
- The split probe is read in eval mode with `TAU_END` (twg:453-460). This has no effect, since there is no dropout
  and the gate is a plain softmax.
