# test_key_load.py — specification (DRAFT, EXPLORATORY, not a result)

A draft written by the exploratory session on claude/outside-ideas (batch 20), in the form of specs/test_split_copy.md, for
the user and X to adopt, change or drop. Nothing here is fixed on the main line until it is committed there. Every number
below that comes from a screen (S64-S68) is EXPLORATORY, not a result. The CHECK numbers are provisional: they continue
from the main line's last (test_delta_recipe's 173 at eb5fd49) and are renumbered on adoption.

## The user's specification (verbatim, from the batch 20 prompt)

> Then write explore_out/spec_draft_keyload.md: a main-line specification in the form of
>  specs/test_split_copy.md for the key-load test the user and X may adopt: S = 4, k = 16, conv, P = 16 at
>  D = 128 against D = 32 (N = 256), the window recipe with the exact-copy split, the budget from S68, 20
>  seeds per machine (propose blocks; check every branch and the main line's seed list), the oracle at
>  both widths on the first two seeds as validity; claims pooled primary and per machine secondary: K1
>  D128 beats D32 on BOUND ROUTED (exact one-sided McNemar); a rate claim on D128; how a run that ends
>  one-to-one and unbound at the budget counts (pre-registered: it is unbound); readings named so each
>  outcome has exactly one name; CHECKs: the batch-19 reproduction, each arm against the recorded S67 run
>  of the same seed through 2400.

**One change, the user's (AskUserQuestion, batch 20, before any S68 run): the recipe is the one screened, with no
split.** S67 and S68 ran LOCAL3 + SLOW with no trigger at S = 4. The exact-copy split has not been run at P = 16 or at
D = 128 with S = 4, so the draft fixes the screened recipe. The split is kept as a flagged option (Option S, at the end),
not part of the test.

## Background (EXPLORATORY, claude/outside-ideas)

- test_window_gate Part B (main line; S = 4, P = 4, k = 16, conv, 28800, X 540-559 / L 1540-1559): WIN3_SLOW16 (LOCAL3 +
  SLOW) BOUND ROUTED 38/40 pooled, RELIABLE. This test's KL_D32 is WIN3_SLOW16 with P = 16 in place of P = 4.
- S64 (batch 19; S = 4, P = 4, 28800, seeds 270-279): the same recipe at N 256-1024 and D 32-128 is "scale-free". D128
  (mult 2, D 128, N 256) bound 10/10 against N256's 9/10; transitions 3600 at every size.
- S67 (batch 19; S = 4, P = 8 and 16, 43200, seeds 270-279, Xeon @ 2.80GHz, code SHA 17e56410a1db):
  - Oracles (the perfect gate; a single Adam at 1e-3): P = 16 bound at N256 2/2 (27600, 24000) and at D128 1/2 (271 at
    12000; 270 unbound at 0.32).
  - Learned, P = 16: D128 BOUND ROUTED 4/10 (19200-38400), N256 0/10. Paired: 4 / 0, exact McNemar p = 0.125.
  - 12 of the 16 learned failures ended one-to-one and unbound (STREAM-PARTIAL one-to-one): six at each width. The other
    four (all N256) ended MERGED. The screen could not tell a slow run from a stuck one.
- S68 (batch 20; this draft's budget source): every unbound S67 run, regenerated from update 0 to 86,400 under the same
  recipe on a Xeon @ 2.10GHz, each equal to its S67 record at 43,200 (curve, statistics and the weights' SHA-1).
  - The twelve one-to-one runs (six per width): **none bound by 86,400** (k_D128 = 0/6, k_N256 = 0/6; both widths
    "stuck" by the readings fixed before any run; joint reading "neither"). Accuracy at 86,400: D128 0.29-0.49, N256
    0.14-0.80 (N256 270 and 272 flat at about 0.15; 273-276 at 0.66-0.80, 274 and 275 flat or falling since
    about 57,600, 273 and 276 still creeping).
  - Three were still rising by >= 0.02 between 76,800 and 86,400 (D128 271 +0.032, D128 272 +0.023, N256 276 +0.05), so
    the pre-registered rule prints "86,400 is not enough" beside the budget.
  - The budget rule (the latest transition of any BOUND ROUTED P = 16 learned run, S67 and S68, + 2400, rounded up to a
    multiple of 2400) gives **40,800**: the latest transition is still S67's 38,400 (D128 270); no run bound after
    43,200.
  - The MERGED N256 runs did not bind (271, 277, 278 still MERGED; 279 ended one-to-one at 0.20). The D128 oracle on
    seed 270 ended unbound at 0.38 (a single Adam at 1e-3).
  - Read together: S67's 4/10 against 0/10 is not a budget effect. Every learned P = 16 binding seen in S67 and S68 came
    before 38,400; doubling the budget converted none of the twelve one-to-one runs.

## Interpretation, fixed here before any code

**The path.** Every learned arm runs `test_stream_recipe.run_attempt` with ARM["A4k16"] (S = 4, k = 16, conv 'layer'
K = 4, batch 32, evaluation every 1200 on held-out sequences, run_one's early stop: three evaluations >= 0.95 after 4800)
on `test_channel_binding.task_for(16, 4)` (S = 4, P = 16, n_vals 16, n_q 1, grouped layout, length 195), registered in
`test_stream_recipe.TASKS` as "P16S4" (as explore_scale_child does), for **40,800 updates** (`dict(tsr.REAL,
total=40800)`). The oracles run `ceiling4k16` (the perfect gate: stream s -> channel s, channels 4-15 unused,
ctx_pad 16) on the same task, path and budget.

**The budget: 40,800 updates** (S68's pre-registered rule; see Background). Every P = 16 learned binding in S67 and S68
came at or before 38,400. S68 also printed "86,400 is not enough": three one-to-one runs were still rising slowly at 86,400.
Under this test such a run counts as unbound at the budget (Outcomes). The draft does not lengthen the budget: doubling
it in S68 turned none of the twelve one-to-one runs into a binding.

**The widths.** N = mult x D = 256 at both:
- **D32**: mult 8, D 32. The run path's own model; no knob.
- **D128**: mult 2, D 128. A width knob, the only new model code: the builder sets `test_multilayer_binding.D` to 128
  for the duration of `tmb.build` and restores it (explore_b19_model.embed_dim / builder_of at claude/outside-ideas,
  copied). The gate's hidden width stays 32: W_in is 32 x D, W_g is 16 x 32.
- The window at D = 128: test_window_gate's `window_init` with D rows, `(torch.rand(D, 3, generator=g) * 2 - 1) /
  sqrt(3)`, g seeded seed + 31337. Its first 32 rows are the D = 32 window of the same seed (the generator draws row by
  row). `to_local` reads D from the model (`m.config.n_embd`) instead of the module constant. At D = 32 both are
  test_window_gate's own functions, unchanged.

**The recipe (as screened; no split).** Learned arms: test_window_gate's LOCAL3 + SLOW (`make_local_rc(slow=True)`):
the gate group (W_in, W_g, the window) at 1e-3 throughout; every other trainable parameter at 1e-4 for updates 1-2400
and 1e-3 after (the switch after the evaluation at 2400); W_h frozen and in no group; no hinge, no trigger. Oracles: the
recorded validity recipe, a single Adam at 1e-3 on every parameter (test_window_gate's CEIL_B at this task and size).

**Arms (20 seeds per machine, paired by seed: the same batches and the same initial gate window rows 1-32).**
- KL_D32: LOCAL3 + SLOW at D32.
- KL_D128: LOCAL3 + SLOW at D128.
- CEIL_D32, CEIL_D128: the oracle at each width, on the machine's first two seeds (validity).

**Outcomes.** BOUND ROUTED (`test_stream_recipe.outcome(r) == "BOUND ROUTED"`), with the transition.
- **A run that ends one-to-one and unbound at the budget is unbound.** It is not BOUND ROUTED, whatever its accuracy or
  its slope. Its failure class (`fail_class_v2.outcome_v2`, e.g. STREAM-PARTIAL) is printed as a diagnostic and enters
  no claim.
- An oracle is BOUND if `test_conv_lr.bound_r` holds (a transition held to the end).

**Validity** (test_window_gate's rule, VALID_N = 2). A width is VALID on a machine if its oracle binds on both of the
machine's first two seeds within the budget. K1 uses both widths; K2 uses D128. A claim is UNTESTED on a machine where a
width it uses is not VALID, and UNTESTED pooled unless it is VALID on both machines. UNTESTED is not a negative.
**A validity risk, for the user and X to decide before adoption.** At D = 128 the oracle on its recorded recipe (a single
Adam at 1e-3) bound one of S67's two seeds (271 at 12000); seed 270 did not bind by 43,200, and S68 ran it to 86,400:
still unbound (0.38, all four streams 0.36-0.40). With VALID_N = 2 at D128, K1 and K2 may be UNTESTED on a machine for a
reason that has nothing to do with the learned gate. The draft keeps the user's rule as written. Two alternatives, not
adopted here: (a) the D128 oracle on the SLOW recipe (other parameters at 1e-4 for 2400 updates, as the learned arms);
(b) VALID_N = 1 at D128. Either would need its own screen first: neither was run.

**Seeds: X 660-679, L 1660-1679; the oracles on 660-661 / 1660-1661.** The blocks were checked against:
- the main line's seed list (claude/bdh-growth-hebbian-inference-w90069 at eb5fd49): `test_split_copy.earlier_seeds()`
  and every range in the test files and specs since: 0-359, 400-479, 500-569, 580-599, 1000-1149, 1160-1189, 1200-1219,
  1240-1279, 1400-1479, 1500-1569, 1580-1599, 2160-2189, 2200-2219, 2240-2279 and the same at 3xxx and 4xxx.
  No match for 660-679 or 1660-1679 (a full-text search finds only digits inside other numbers);
- the other sessions' pending drafts: Session G's register draft proposes X 600-619 / L 1600-1619, Session H's
  two-key-term draft X 620-639 / L 1620-1639;
- generator seeds: 640-659 is skipped because test_p_scaling's CHECK uses generator seeds 650 and 651;
- every exploratory branch's screens: claude/outside-ideas through batch 20 uses seeds below 300; Sessions F, G and H's
  screens use 300-489 (e.g. F's S68 300-309, H's S61 480-489). On every branch, the only hits for 660-679 are JSON step
  keys and accuracies, not seeds.
The seeds are disjoint from every earlier main-line seed and from the other machine's.

## Claims (fixed now)

Exact tests on the seed pairs. Pooled (PRIMARY): over both machines' pairs, UNTESTED unless VALID on both machines. Per
machine (SECONDARY): the same tests on each machine's 20 pairs.

- **K1** KL_D128 beats KL_D32 on BOUND ROUTED: exact one-sided McNemar (b = D128 only, c = D32 only), SHOWN if p < 0.05.
  (Pooled n = 40: with c = 0, b >= 5 is SHOWN; per machine the same.)
- **K2** (the rate claim on D128) KL_D128's BOUND ROUTED rate exceeds 0.2: exact one-sided binomial test of
  H0: rate <= 0.2, SHOWN if p < 0.05. Pooled n = 40 needs >= 13/40; per machine n = 20 needs >=
  8/20. The Wilson 95% interval is printed beside it. The rate 0.2
  is half of S67's point estimate (4/10, Wilson [0.17, 0.69]); at a true rate of 0.4 the pooled test has power 0.87 and
  the per-machine test 0.58.

**Readings** (pooled, primary; per machine the same names, printed as secondary). Each outcome has exactly one name:

| K1 pooled | K2 pooled | reading |
|---|---|---|
| UNTESTED (either width not VALID on a machine) | any | "UNTESTED at this budget" |
| SHOWN | SHOWN | "the width carries the key load" |
| SHOWN | NOT SHOWN | "wider binds more, not yet reliably" |
| NOT SHOWN | SHOWN | "the key load binds at D=128; the width effect is not shown" |
| NOT SHOWN | NOT SHOWN | "no key-load lead at this budget" |

(K2 is UNTESTED only where K1 is, since D128's validity is part of both; the table's first row covers it.) Nothing
else is read.

**Printed, not claims:**
- Per seed: the outcome, the transition, the stop, the final accuracy, the end map and fail_class_v2 for every unbound
  run; the per-stream held-out accuracy at the budget.
- Each unbound run's accuracy at the budget minus 9600 and at the budget, flagged "still rising" if it rose >= 0.02.
  It is unbound either way.
- The transitions of the runs both arms bind routed (median per arm; paired differences).
- The oracles' transitions at each width.
- Bands with Wilson 95% intervals for every arm, per machine and pooled (RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5
  n), MINORITY >= 1, else NEVER).

## CHECKs (all before any run unless stated; provisionally numbered from 174)

- **The inherited chain:** test_window_gate's verification (its CHECKs 135-144 and below), on this machine's own earlier
  results files, with test_split_copy's `early_assert` knob as committed.
- **174** the batch-19 reproduction: each arm against the recorded S67 run of the same seed through 2400 (curve,
  gradient norms and every statistic both records hold). Records: claude/outside-ideas at e3f5809,
  explore_out/capacity_results.json (code SHA 17e56410a1db, Xeon @ 2.80GHz), copied byte for byte into the main line:
  - KL_D32 at seed 270 = P16_N256|270; KL_D128 at seed 270 = P16_D128|270;
  - CEIL_D32 at seed 270 = ORC_P16_N256|270; CEIL_D128 at seed 270 = ORC_P16_D128|270.
  Asserted on a Xeon @ 2.80 or 2.10GHz. Adam runs have reproduced across those two, and batch 20 reproduced all 18 S67
  runs it regenerated through 43,200 on a 2.10GHz Xeon against the 2.80GHz records. Printed otherwise (L's i7-12650H).
- **175** the width knob is inert at D = 32: the knob's builder at (mult 8, D 32) gives the run path's initial parameters
  bit for bit; the D-reading `to_local` equals test_window_gate's at D = 32; KL_D32 at seed 1 through 1200 equals the
  same run without the knob. After every build, `test_multilayer_binding.D` is 32 again.
- **176** the D128 model: n_embd 128, N = 256; W_in 32 x 128, W_g 16 x 32; the window is 128 x 3 and its first 32 rows
  equal the D = 32 window of the same seed. The parameter count is printed for both widths.
- **177** the recipe, through each arm's real path (forward stubbed), at updates 1, 2400 and 2401: the learned arms' gate
  group (W_in, W_g, the window) at 1e-3 throughout and every other trainable parameter at 1e-4 then 1e-3, W_h in no
  group; the oracles one Adam group at 1e-3 holding every trainable parameter.
- **178** the task: task_for(16, 4) is S = 4, P = 16, n_vals 16, n_q 1, length 195, and "P16S4" in TASKS is that task.
- **179** the seeds: every arm on the machine's range, the oracles on its first two; disjoint from every earlier
  main-line seed and from the other machine's range; with --also, the other file's machine, CPU and seeds.
- **180** a worker's run is bit-identical to the same run here (KL_D128 and KL_D32, seed 1, 1200 steps).
- **181** (a resume) records cached by an earlier start reproduce under this code through 2400.
- **182** (checkpoints; needed only if a run can outlive a session, see Runtime) a run halted at 1200 and resumed from its
  checkpoint equals the uninterrupted run through 2400 bit for bit (curve, statistics, weights' SHA-1). Every resume is
  recorded cumulatively across processes (batch 20's 44,400 check failed only because its record kept the last resume).

## Runtime

Measured in S68 (EXPLORATORY, a Xeon @ 2.10GHz, 4 workers, 1 torch thread each): a D128 run advances about 11,000
updates in 116 minutes (about 0.63 s per update including evaluations), a D32 (N256) run about 0.2 s per update. At 40,800
updates: about 7.1 h per D128 run and 2.3 h per D32 run in the worst case; per machine (20 + 20 learned runs and 2 + 2
oracles) about 200 worker-hours, a makespan near 50 h on 4 workers. A run that binds stops early (three evaluations
>= 0.95 after 4800). Runs of this length outlive a session: each run checkpoints after every evaluation and resumes from
its checkpoint (CHECK 182).

Projection before the first run, with pool-load timing (each arm's real path for 1200 steps as simultaneous copies on the
workers, the median). The worst case runs every run to the budget. Nothing is cut. Order: oracles first, then longest
first. Runs are cached in key_load_results.json (gitignored, written atomically after every run); a restart resumes
from the cached records. Long runs go through `longrun.py`.

## Output

Per-seed raw values before any aggregate; validity; counts; the claims pooled (with `--also`, the other machine's
complete file) and per machine; bands; readings; the printed comparisons; curves. RESULT sections go in the docstring,
the JSON to results/<machine>/, and a row to results/README.md. Both machines, disjoint seeds, --machine X|L; --also
pools. New tests use fail_class_v2.

## Option S (flagged; not part of the test above)

The user's prompt named "the window recipe with the exact-copy split". If it is wanted, it would be a third learned arm,
or a replacement for both, with test_delta_recipe's HEBB_SPLIT trigger: the KEYMASS check every 2400 from 4800, <= 3
splits >= 4800 apart, the exact copy of W_g's row c* onto c0, and W_g's Adam state zeroed. At S = 4 with the budget above,
the trigger's checks would run through 38,400. Not screened at S = 4, P = 16 or D = 128. Every screen at S = 4
(S64, S65, S67, S68) ran with no trigger, so the budget above was not measured for it. A screen of the split at this
configuration would come first.
