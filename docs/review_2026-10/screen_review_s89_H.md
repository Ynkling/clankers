# Screen review: Session H's S89 (the split at eight streams, keys drawn from a pool of 32)

*Referee V, 19 October 2026, before any learned-arm run. Reviewed: `explore_h_s89.py` and `explore_h_s89_child.py` at claude/explore-H `d4a9a7d`, the CHECK log `explore_out/H/s89_oracle.log` at `7820b12`, at `0dd88bf` (18:16 UTC) and at the head `42d86ae` (18:32 UTC). "s89:NN" is a line of `explore_h_s89.py` at `d4a9a7d`, "ch:NN" a line of `explore_h_s89_child.py` at `d4a9a7d`; "prompt:NN" a line of `docs/reading/parallel_sessions_prompts.md` at `7c4ceb8` (Session H, round 6, prompt:1198-1229). "out §N" is a section of `screen_checks_s89_H_out.txt`, written by `screen_checks_s89_H.py` (stdlib + numpy; reads git; trains nothing; it execs `report()` from s89's source with the harness stubbed, and re-implements the task's distribution in numpy).*

**Has a learned arm started? No.** At `0dd88bf` all 14 CHECK rows of H/s89 passed (20.4 min) and the two perfect-gate runs (ORACLE 2900, 2901) started at 18:16. At the head `42d86ae` the store holds one run, ORACLE|2901: BOUND ROUTED, transition 10800, 16 min. ORACLE 2900 is still running (out §0). The learned arms are queued only by a second invocation, and only after ORACLE binds 2/2 under the same code SHA (s89:129-141). So every fix below can land before them.

**Verdict: fix first (text and the report path only; no fix changes a run's bit stream, and the perfect gate can keep running).**

What is sound:
- **Readings.** They give every outcome exactly one name per key, and agree with the prompt's rules on all 1,261,260 distinct outcome statistics (out §2).
- **Outcome.** The outcome is the main line's `test_stream_recipe.outcome`, imported (out §3).
- **Seeds.** 2900-2959 are free on every branch head and in all history (out §1).
- **CHECKs.** They tie the code path to three recorded runs bit for bit and passed. The main line's X record `WIN3_SPLIT_D8|540` also reproduces bit for bit through 4800. It was printed and not asserted, but true on all four fields (§4).
- **Labels.** No label reaches the label-free arms: ALLMASS's pair comes from the read gate alone, REPEATMASS's from the read gate and token ids (§6).

What must change before the learned arms:
1. **The screen does not remove what the key-position rule reads.** On the grouped layout the keys stay at the fixed positions 3j+1. They are still the most repeated tokens within a sequence (8 each, against CTX 4 and VAL 1-4). Each key still sits next to its stream's CTX token. The pool makes key *types* rare in the corpus (0.0105 against 0.084), nothing else.
   - The prompt's "the key-position rule has nothing to read" (prompt:1205) is therefore not what this screen tests. s89:18 says so, but s89:3-4 and the readings do not.
   - A "survives" here is a result about rare key types on a layout with positional keys, not about unmarked keys (§6, §7; fix 1).
2. **REPEATMASS is ALLMASS minus about 4% of positions.**
   - The repeat set is 94 of 98 input positions per sequence: every CTX and KEY position, 7/8 of the VAL positions, and both query inputs (out §6). The key positions are 34% of it.
   - In H's own CHECK on the pool-4 harness, the repeat pair equalled the all-position pair at all three firings (`0dd88bf` log).
   - Its two readings are not a second finding and must not be counted as one (fix 2).
3. **"Moot" readings still carry names.** When NOSPLIT is within one pair of BASE, report() prints "moot (...): T: the trigger survives" or "... needs key positions" (s89:283).
   - This happens for 25 / 23 statistics (survives) and 253 / 255 (needs key positions), for ALLMASS / REPEATMASS respectively (out §2).
   - One of them is self-contradictory: at BASE 7, NOSPLIT 6, ALLMASS 10, the screen prints both "the split is not needed here" and "the trigger survives".
   - The prompt says the trigger readings are moot. A moot reading should carry no name (fix 3).
4. **"Survives" has little power at 10 pairs.** It is likely at a true learned rate of 0.8 against the ceiling's 0.9, and possible at 0.7:

   | true ALLMASS rate (ceiling 0.9, NOSPLIT 0.3) | P(survives), independent → fully paired |
   |---|---|
   | 0.9 | 0.66 → 0.94 |
   | 0.8 | 0.39 → 0.64 |
   | 0.7 | 0.20 → 0.29 |

   At NOSPLIT 0.6 "survives" is at most 0.35 at any learned rate (out §5). The prompt fixed the rule, so the fix is to print the operating characteristics with it (fix 4).

The numbered fixes at the end are replacement text.

## 0. State (out §0)

| ref | time (UTC) | contents |
|---|---|---|
| `d4a9a7d` | 17:54:28 | the screen |
| `7820b12` | 17:56:24 | 9 log lines: banner, code SHA, `repro_check` BIT-IDENTICAL to X's arm A seed 160. No CHECK row yet |
| `0dd88bf` | 18:16:29 | the full CHECK log, `checks.json` with H/s89, and `s89_results.json` with `meta` only (`runs` empty) |
| `42d86ae` (head) | 18:32:42 | ORACLE\|2901 saved: BOUND ROUTED, transition 10800 (perfect gate on pool 32; S87a's pool-4 perfect gate bound at 2400) |

At `0dd88bf`:
- All 14 CHECK rows read ok, then "2 runs queued on 4 workers"; ORACLE 2900 and 2901 started.
- The stage is the oracle stage (s89:133-135). In the other stage `configure()` would print its ORACLE line before the banner and drop the screen (s89:136-141).
- **No learned-arm screen run has started.**
- The CHECK did run BASE, ALLMASS, REPEATMASS and NOSPLIT for 4800 updates on screen seed 2902 at pool 32 with the threshold at 0 (s89:171-172). Those runs are not stored and give no outcome, so I count them as CHECK compute, not screen runs (item 7.5).

## 1. The readings (s89:42-51, s89:241-285; out §2)

**Method.**
- report() was exec'd from its own source at `d4a9a7d`. H's `explore_h_common` (`paired`, `br`, `band`, `wilson`) was exec'd too. Only the store loaders were stubbed, to hand report() synthetic records.
- Outcome space: every per-seed pattern of the four learned arms over the 10 seeds. That is 3,268,760 compositions and 1,261,260 distinct sufficient statistics (BASE count; b and c of NOSPLIT/BASE, ALLMASS/BASE, ALLMASS/NOSPLIT, REPEATMASS/BASE, REPEATMASS/NOSPLIT).
- report() ran on one representative of each of the 1,772 patterns of the predicates the rules test, plus 20,000 random statistics. My re-implementation of the prompt's rules (prompt:1216-1220) ran on all of them. "Within 1 pair of BASE" was read as S87's "≥ BASE − 1 pair", i.e. BASE only − X only ≤ 1 on paired seeds, which equals the count difference.

**Results.**
- **0 disagreements** between report() and the prompt's rules.
- **Every outcome gets exactly one name per reading key.** The keys are S89 (UNTESTED / INCOMPLETE / "the labelled recipe itself fails here") or else split, ALLMASS and REPEATMASS. No gaps, by the if/elif/else at s89:262-283.
- **The validity and incomplete paths** print as fixed: ORACLE 1/2 gives "UNTESTED (ORACLE bound 1/2); no learned arm runs"; 9 of 10 BASE runs gives "INCOMPLETE, no reading".
- **The discordant pairs are labelled correctly.** `hc.paired(x, y)` returns b = x only and c = y only (explore_h_common.py:114-120). s89:269's `c - b <= 1` on "X vs BASE" is therefore BASE only − X only ≤ 1, as the docstring says.

**Overlaps of meaning.** Within a key there are none; across keys there are two.

- **(a) Named readings under "moot".**
  - With the split "not needed", a trigger reading still prints its name behind the word "moot" (s89:283). Counts: ALLMASS survives 25, needs key positions 253; REPEATMASS 23 and 255.
  - "Moot, needs key positions" is the common case. BASE beats T 4-0 while NOSPLIT is within one pair of BASE, which says T is *worse than no trigger*. The prompt calls it moot, but the name invites reading it.
  - "Not needed + survives" occurs only at BASE 7, NOSPLIT 6, ALLMASS 10, with ALLMASS ⊇ NOSPLIT. There the all-position trigger beats no trigger 4-0 and the screen still calls the split not needed.
  - The prompt's "moot, say so" is better served by printing no name (fix 3).
- **(b) The two trigger arms are near-copies** (§6). "ALLMASS survives" + "REPEATMASS survives" is one observation printed twice. "ALLMASS survives" + "REPEATMASS needs key positions" is possible in the rule space (out §2: 51 + 3 statistics), but on this layout it would be noise between two near-identical arms (fix 2).

**"Needs key positions" carries no significance.** At its threshold (4 vs 0) the exact one-sided McNemar p is 0.0625 (out §5). The rule does not claim significance, and the docstring prints p (s89:42); the reading text should carry p beside the name (fix 4).

## 2. The outcome (s89:38-39; out §3)

**Imported, not re-implemented.**
- The child's `run()` calls `b16.finish` (ch:204), which sets `rec["outcome"] = tsr.outcome(rec)` (explore_b16_main.py:199-206).
- `tsr` is the main line's `test_stream_recipe` at `9c5939e`, served to the child by `explore_main9c`.
- The parent reads `hc.br(r) = r["outcome"] == "BOUND ROUTED"` (explore_h_common.py:73-74).

**Same as the main line's head.** Syntax trees at `9c5939e` against `7c4ceb8` are equal for:
- `outcome`, `routed`, `padded_task` and `model_fn` (test_stream_recipe.py:751-762 for the first two);
- `routing_k`, `fail_class_k`, `scale_stats` and `stream_acc` (test_scale_axes);
- `merged` and `shared_max` (test_stream_channels);
- `test_router_layout.fail_class`.

`ROUTED_ACC` = 0.9 in both.

**v2.** `explore_h_fail_class_v2.py` is byte-identical to the main line's `fail_class_v2.py` at `7c4ceb8`. `outcome_v2` is its function, called with `tsr.routed` (ch:208-209).

**On the pool task.**
- `routing_k` labels streams by CTX token, and keys by `token − S` (0..31). Its `ch_map` and `one_to_one`, which `routed` reads, do not depend on P or K (test_scale_axes.py:273-299).
- `stream_acc` and the held-out set use the task passed in (`eval_batch(task)`, `probe_batch(task, seed)`, test_router_reliability.py:303). Neither is cached by task.
- `PoolTask.key` is distinct ("P4K32", ch:76).
- **So BOUND ROUTED is computed on the pool task.**

`key_sep` and `key_part` index key ids 8-11 only (ch:20-21). They are descriptive, and no outcome reads them.

## 3. Seeds (out §1)

| scan | blobs read | seed values | hits in 2900-2959 |
|---|---|---|---|
| every JSON / JSONL blob on the 13 remote heads | 339 (1,061 branch-paths) | 31,860 | only H's own `s89_results.json`: its meta lists ORACLE 2900, 2901, and it holds the ORACLE\|2901 run (`42d86ae`) |
| every JSON / JSONL blob reachable in history from every ref | 2,291 | 319,838 | only the two versions of that file |

No integer dictionary key falls in the block.

**Text mentions on the heads** (`*.py`, `*.md`, `*.tex`, `*.txt`, `*.log`):
- the prompt itself;
- H's two S89 files and its log;
- elsewhere, numbers that are not seeds: seconds (`delta_recipe_L.log:2556` "2922s"; R's phase4/5 logs "wall=2936s/2944s/2952s"), a line count (`batch2.log` "2952 +++"), and a time offset (`check_delta_recipe_X_v6.md:254` "+2909 s").

**Neighbouring blocks.** F 2800-2839 and G 2840-2899 are disjoint.

**The seeds are free.**

## 4. Do the CHECKs tie the new code to a recorded run? (s89:57-67, s89:161-219, ch:256-338; out §4; log at `0dd88bf`)

**Yes, three ways, all asserted and all ok.**

| CHECK (pool size 4) | record | record's CPU | result |
|---|---|---|---|
| BASE\|2600 and ALLMASS\|2600 through 9600: curve, statistics, the trigger rows' common fields, first firing at 4800 | S87a's records (`explore_out/H/allmass2_results.json`, code `898867f69ad4`) | 2.10 GHz, as here | ok |
| NOSPLIT\|260 through 6000 | batch 16's `LOCAL3_SLOW_D8_A\|260` (`explore_out/window_scale_results.json`, code `01767493e840`) | 2.80 GHz | ok |
| ORACLE\|2600 through 2400 | S87a's `ORACLE\|2600`, bound at 2400 | | ok |

The 2.80 GHz batch-16 record has reproduced bit for bit on this CPU before: 6 CHECK lines in s79.log, 1 in s86.log, 1 in s87.log.

**What the ties cover.**
- **The task.** PoolTask(4)'s batches equal TASK48's on six generators, with the generator state after the draw. This covers the pool draw and its sizes (ch:259-282).
- **The model builder override.** `pool_rc` (ch:171-180) builds the model through `tsr.model_fn` on the passed task.
- **The trigger.** At target key / all, S89's trigger equals S86's row for row (ch:317-332). Out §6 gives the syntax-tree diff: three added lines compute the repeat pair from the same read gate, and the target lookup is a dict.

**The main line's record** (prompt:1222-1223).
- Pool 4 with the main line's split op, against X's `WIN3_SPLIT_D8|540` (recorded on a 2.80 GHz CPU; committed to the main line in `628eb7c`), is printed, not asserted (s89:204-214). The prompt allows this when the CPU differs.
- It printed True on the curve, the statistics, the gradient norms and the trigger rows at 2400 and 4800 (fired at 4800 in both).
- **So the code is tied bit for bit to a named main-line record through 4800**, as well as to S87a.
- **One defect.** The CHECK reads the record through `MAINREF = "origin/claude/bdh-growth-hebbian-inference-w90069"` (s89:106, s89:164), a branch name. A later commit on the main line would silently change what is compared. Pin it to `628eb7c` (fix 6).

**At pool 32 nothing can be tied to a record**, because none exists. Three CHECKs bear on the pool-32 path:
- with the threshold at 0, BASE, ALLMASS and REPEATMASS equal NOSPLIT bit for bit through 4800 on seed 2902, none fired. The trigger's side computations (repeat mask, routing_k, overlap) do not perturb the run;
- the structure CHECK (ch:283-289);
- the token-ids-only CHECK on `repeat_mask` (ch:290-300).

That is what can be asked; the perfect gate is the validity link.

**What s89_oracle.log shows so far.**
- At `7820b12`: only the provenance and `repro_check` lines.
- At `0dd88bf`: the 14 CHECK rows (all ok), REPEATMASS's printed probe overlap (6,005 of 6,272 input positions repeat; key positions 2,048 of 2,048; not key positions 3,957), the MAINSPLIT line (all True), and the two ORACLE starts.
- At `42d86ae`: ORACLE|2901 bound routed at 10800; ORACLE 2900 still running. The pool task binds under the perfect gate, about 4.5 times later than at pool size 4 (2400 in S87a).

## 5. Operating characteristics of the readings (out §5)

**Model.**
- Exact multinomial over the 8 per-seed patterns of (BASE, T, NOSPLIT), with BASE at 0.9.
- T is ALLMASS or REPEATMASS. NOSPLIT is 0.3 (X's `WIN3_SLOW_D8` 6/20, batch 16's `LOCAL3_SLOW_D8_A` 3/10) or 0.6.
- ρ is the share of seeds on which one difficulty decides all three arms: 0 is independent, 1 fully paired.
- S86 + S87a's BASE / ALLMASS pairs were 23 both, 3 neither, 3 BASE only, 1 ALLMASS only (out §7), which is strongly paired: ρ ≈ 0.5-1 is the plausible range.

**At 10 pairs, NOSPLIT 0.3** (fails = BASE < 7/10, 0.013 throughout; moot ≤ 0.005):

| true T | P(survives) ρ = 0 / 0.5 / 1 | P(needs key positions) | P(no reading applies) |
|---|---|---|---|
| 0.9 | 0.66 / 0.80 / 0.94 | 0.005 / 0.001 / 0 | 0.32 / 0.19 / 0.05 |
| 0.8 | 0.39 / 0.50 / 0.64 | 0.05 / 0.03 / 0.01 | 0.54 / 0.45 / 0.34 |
| 0.7 | 0.20 / 0.26 / 0.29 | 0.16 / 0.15 / 0.12 | 0.63 / 0.58 / 0.58 |
| 0.5 | 0.03 / 0.03 / 0.02 | 0.47 / 0.55 / 0.62 | 0.48 / 0.41 / 0.35 |

**At NOSPLIT 0.6:**
- "moot" takes 0.15-0.19;
- "survives" falls to ≤ 0.35 even at T = 0.9, because beating NOSPLIT by 4 vs 0 needs NOSPLIT ≤ 6/10;
- "needs key positions" is unchanged.

**What the rule can and cannot say.**
- **"Needs key positions" is a real detector only of a large loss.** It fires with probability about 0.5-0.6 when ALLMASS falls to 0.5, and rarely (≤ 0.16) at 0.7.
- **"Survives" does not separate equality from a 10-point loss.** Under strong pairing it reads 0.94 at equality and 0.64 at 0.8. At ALLMASS 0.7 it still reads "survives" about one time in four.
- **"No reading applies" is the modal outcome between 0.7 and 0.8.** It means "ten pairs cannot tell", not "neither".
- **At the screens' own rates** (BASE 26/30 ≈ 0.87, ALLMASS 24/30 = 0.8, NOSPLIT 0.3, ρ = 0.5) the expected print is "survives" about half the time and "no reading applies" about 45%.

**At 20 seeds** (BASE gate 14/20) the fixed one-pair margin becomes stricter relative to the noise. At T = 0.9 "survives" falls to 0.48 independent (0.69 at ρ = 0.5), and at T = 0.8 it is 0.17-0.39. So more seeds alone do not make "within 1 pair" more informative. A fixed count margin does not scale with n, which is a point for the round's design, not for H.

## 6. Does anything read a label it claims not to? (ch:100-168; out §6)

### The call paths at a check (ch:134-167)

**Probe sequences.**
- `data = s36c.probe_data(task)[:64]` (ch:129) is `PoolTask.make_batch(64, Generator(12345))[0]` (explore_split_child.py:69-70).
- These are token sequences only. The `[0]` drops the query-stream label that `make_batch` returns second.
- This is what replaces the main line's probe: the same `BindTask.make_batch` code path (PoolTask subclasses it, ch:71, and overrides only `make_batch` and `role_masks`), at the same seed and size, drawing from the pool.

**Read-gate mass.**
- `gr = read_gate(m, data)` (ch:138) is the model's read gate on `data[:, :-1]`: 98 input positions, the body and the query's CTX and KEY.

**Target pairs** (all three from `gr`):
- **KEYMASS (BASE):** `select(gr, S, P)` takes `gr[:, 3j+1]`, j < 32 (explore_split_target_child.py:63-68). This is the only use of S and P and of key positions.
- **ALLMASS:** `gr.mean((0, 1))`. **No input besides the read gate.**
- **REPEATMASS:** `select_repeat(gr, data[:, :-1])` (ch:105-107), with `repeat_mask(x)` of one argument, the token ids (ch:100-102). CHECKed:
  - invariant under a random permutation of the vocabulary;
  - equal to a Counter reference;
  - signatures `["x"]` and `["gr", "x"]` (ch:290-300).

**The decision to fire** (ch:147-149) reads:
- `acc` (`tbo.evaluate` on the probe, which reads the query's answer, i.e. the training loss's own target, as on the main line);
- `prev`, `thr`, `rise`, `cap`, `gap`;
- `guard`, which is None (S84's constants: every 2400, first 4800, gap 4800, cap 3, probe 64, rise 0.02, thr 0.95, explore_h_constants_child.py:41).

**The op** (ch:158-159) receives `(m, opt, cs, c0, seed)`.

**Labelled quantities stay descriptive.** `cmap` comes from `routing_k(m, task, rprobe)` with stream labels (ch:141). It feeds only `key_ok`, `all_ok`, `rep_ok`, `op_ok` and `map_after`, which are recorded and not read by the decision or the op. `overlap()` (ch:110-124, ch:207) uses key positions, and runs after the run, into the record only.

**Conclusion: no key position, key identity, stream label or derived mask reaches ALLMASS's or REPEATMASS's pair, or the decision to fire.** BASE alone reads key positions, and ORACLE alone reads stream labels (the perfect gate). Both are labelled arms.

### What the pool changes, and what "key positions" mean (out §6)

These figures are from a numpy re-implementation of `PoolTask.make_batch`'s distribution, plus exact values. H's docstring figures (s89:14-24) reproduce: 385,149 / 401,408 = 0.9595 against an expected 94/98 = 0.9592, and VAL 0.8760 against an expected 0.875.

**Key repeats within a sequence.**
- The four keys of a sequence are drawn without replacement, so no key type repeats across key slots (the CHECK asserts 4 distinct keys, ch:285).
- Each key type occurs exactly 8 times in the body (once per stream, in one group of 8 adjacent triples), and 9 times for the queried key.
- CTX types occur 4 (or 5) times, VAL types 1-4 times.
- **Within a sequence the keys are the most repeated class.** "Not a frequent class" holds across the corpus only:

  | class | corpus frequency per type |
  |---|---|
  | KEY (pool 32) | 0.0105 |
  | VAL | 0.0204 |
  | CTX | 0.0421 |
  | KEY on the main line (4 types) | 0.084 |

- A key type appears in about 8 of the probe's 64 sequences; it is absent with probability 0.0002.

**What "key positions" means for the labelled ceiling: unchanged.**
- BASE's mask is the 32 body positions 3j+1, positional on the grouped layout and identical to the main line's.
- On the pool task the positions holding key tokens are exactly those 32, plus the query's KEY input (position 97, which KEYMASS excludes, as on the main line).
- The pool changes the key *types* at these positions, and nothing BASE reads. A label-free rule could find the same mask from the layout's period of 3.
- **So the screen does not instantiate "the key-position rule has nothing to read"** (prompt:1205). s89:16-18 says this plainly; s89:3-4 ("the split without marked key positions") and the readings do not (fix 1).

**REPEATMASS on this layout.**
- The repeat set is 34% key positions. Every CTX and KEY position and both query inputs repeat, as do 87.5% of VAL positions: each other key holds a given value with probability 1/2, so 1 − (1/2)³.
- REPEATMASS's mean differs from ALLMASS's only by dropping ~4 singleton-value positions of 98.
- The prompt's expectation ("on this layout, the keys", prompt:1214) is false, and s89:19-24 says so before any run, which is right.
- The pool-4 CHECK on S48's harness shows the consequence: repeat pair = all pair at 4800, 9600 and 14400 (`0dd88bf` log, "target repeat copies onto the repeat pair").

## 7. Other points bearing on interpretation

1. **What a null means.**
   - "No reading applies" is the likely print for a true ALLMASS between 0.7 and 0.85 (§5). It must be printed as "10 pairs do not decide", not as a failure of either trigger.
   - "Survives" means: on the grouped layout, with keys at fixed positions and each key next to its stream's CTX token, an all-position trigger keeps binding when key *types* are rare in the corpus.
   - The window gate's three-token window at a key position includes that triple's CTX token, so the cue the gate routes by is unchanged by the pool. The trigger's targets come from that gate's mass.
   - A "survives" therefore does not extend S86/S87a to unmarked key positions or to the MQAR-style layout F builds. A "needs key positions" would be informative, because the pool would then have changed something the trigger relies on.
2. **Validity.**
   - ORACLE (perfect gate `ceil8_D8` + conv) validates the pool task, the memory and the path at 43,200 updates. It does not validate the learned gate's capacity.
   - The BASE ≥ 7/10 gate covers that, and is fixed (s89:44, s89:266-267). P(BASE < 7 | true 0.9) = 0.013.
   - Both rules are correct as coded, including the UNTESTED and INCOMPLETE paths (§1).
3. **Budget.**
   - S86 + S87a's runs on the same recipe and CPU took a median of 33-35 min (range 18-85), and the perfect gate 10 min (out §7). The pool task may be slower to bind.
   - 40 learned runs on 4 workers is roughly 6-12 h, inside the prompt's two days (prompt:1225). `--push-each` pushes after each saved run, as asked.
4. **The one-to-one guard.**
   - The trigger has none: `guard` is None and there is no one-to-one test (ch:147-149).
   - In S86 + S87a, 1 of 61 BASE firings and 0 of 63 ALLMASS firings had a one-to-one map before the fire (out §7). H's report 9 found 0 of 107 on the main line's records.
   - The pool task could change this if binding, not routing, becomes the bottleneck: accuracy flat below 0.95 with a one-to-one map, so a fire overwrites a live channel.
   - `summarize` already counts `one_to_one_before` per arm (s89:236). It should be printed beside each trigger reading, so that a "needs key positions" caused by firing on routed maps is visible (fix 5).
5. **Seed 2902 in the CHECK.**
   - The pool-32 CHECK ran four learned arms on screen seed 2902 for 4800 updates before the perfect gate validated the task (s89:171-172).
   - Nothing was stored and no outcome was computed, so no reading can be affected. The CHECK seed could have come from outside the screen seeds, e.g. 2959. **No fix needed now.**
6. **F's module.**
   - s89:11 cites F's module "at a199d5e" (round 5). F's round-6 module is `50dc8e4`, pushed at 17:54:31, three seconds after `d4a9a7d`.
   - It has no grouped-layout variant either: `RandHeaderTaskPool` and `RandHeaderTaskAllPool` subclass `RandHeaderTask`, S70's lite layout with S = 2 (out §7).
   - So H's own pool draw is what the prompt asks for (prompt:1207-1209), and nothing of F's is vendored, so there is no byte-equality to check.
   - F's construction relabels the base batch: for a given generator state the pool batch is the lite batch with its keys renamed. H's draws values per pool key (ch:82), so at K = 32 the batches are a different random stream from the main line's. This is equivalent in distribution, and is not a defect. The citation should name `50dc8e4` (fix 7).
7. **The trigger target's span.** The all-position mean covers the query's CTX and KEY inputs as well as the 96 body positions (98 in all). This is the same code as S86/S87a (`select`, explore_split_target_child.py:67), so it is consistent with the arm it continues.

## Fixes (replacement text; most important first)

1. **What the screen tests** (`explore_h_s89.py`).
   - Replace s89:3-4 with:
     > "explore_h_s89.py — EXPLORATORY, not a result. Session H, round 6, screen S89: the split with rare key types, at eight streams. Does the all-position trigger, with the exact-copy split, still bind eight streams when each key type is rare across the corpus (32-key pool), on the grouped layout where key positions are still fixed (3j+1)?"
   - Add after s89:51:
     > "CAVEAT (printed with every reading): the pool makes key types rare across the corpus (0.0105 per type against 0.084 at pool size 4). It does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. BASE's key-position mask is the main line's, unchanged. A reading here is about rare key types on a layout with positional keys, not about unmarked key positions."
2. **REPEATMASS** (append to s89:35-36 and to the readings).
   > "On this layout the repeat set is 94 of 98 input positions (every CTX and KEY position, 7/8 of the VAL positions, the query's CTX and KEY); REPEATMASS is ALLMASS without ~4 singleton-value positions, not a key-selecting statistic. Its reading is printed beside ALLMASS's as 'REPEATMASS (repeat set = X% of input positions; same pair as ALLMASS at s of n firings): <reading>' and is not a second finding; if the two arms chose the same pair at every firing, it is printed as 'REPEATMASS: as ALLMASS (same pair at every firing)'."
   - In report(), compute the per-run share from `same_all_rep` and `ops` (already in `summarize`, s89:235).
   - Whether to keep the arm at all (10 runs, about 6 CPU-hours) is X's call. As fixed by the prompt, it does not test what the prompt expected it to.
3. **Moot readings carry no name.**
   - Replace s89:45-46 with:
     > "'the split is not needed here' NOSPLIT within 1 pair of BASE. Then each trigger arm is printed 'T: moot (the split is not needed here); BASE only b, T only c; T only d vs NOSPLIT only e', with no other name."
   - Replace s89:283 with:
     > `rd[t] = (f"{t}: moot (the split is not needed here); " + s.split(" (", 1)[1].rstrip(")")) if moot else s`
     
     (or any form that prints the counts without "survives", "needs key positions" or "no reading applies" when moot).
4. **Print what the rule can detect** (add to the readings block, s89:42-51).
   > "Operating characteristics (exact, BASE 0.9, NOSPLIT 0.3, 10 pairs; independent / half-shared / fully shared seed difficulty): P('survives') at a true trigger rate of 0.9, 0.8, 0.7, 0.5 = 0.66/0.80/0.94, 0.39/0.50/0.64, 0.20/0.26/0.29, 0.03/0.03/0.02; P('needs key positions') = 0.005/0.001/0, 0.05/0.03/0.01, 0.16/0.15/0.12, 0.47/0.55/0.62. 'Survives' does not distinguish a trigger as good as BASE from one 10 points worse; 'needs key positions' detects a loss of about 40 points half the time; 4 vs 0 is one-sided p = 0.0625. 'No reading applies' means ten pairs do not decide."
   - Each printed reading carries its exact one-sided McNemar p (`p` / `p_rev` from `hc.paired`, already computed).
5. **One-to-one firings** (s89:52-56, the descriptive block).
   > "Printed with each trigger arm's reading: its firings, and how many had a one-to-one map before the fire (one_to_one_before), and how many were labelled ok (op_ok)."
6. **Pin the main-line record.** s89:106:
   > `MAINREF = "628eb7c"   # the main-line commit that recorded results/X/window_gate_results.json (WIN3_SPLIT_D8|540)`
   
   This does not change the CHECK's result. The record is the same at `7c4ceb8`.
7. **The citation.** s89:10-11:
   > "F's module has no grouped-layout pool variant at 50dc8e4 (RandHeaderTask-pool / -allpool are S70's lite layout, S = 2)"
8. **The MAINSPLIT row's rationale** (s89:64-67; record only, no code change). Add:
   > "printed True on curve, statistics, gradient norms and trigger rows through 4800 at 0dd88bf"
   
   This removes the reason to doubt the main-line link. Batch 16's 2.80 GHz record is asserted (s89:62-63) and has reproduced on this CPU in S79, S86 and S87a, so the CPU difference is not by itself a reason to print rather than assert.

**Commit reviewed:** claude/explore-H `d4a9a7d` (screen), with the CHECK log at `7820b12`, `0dd88bf` and the head `42d86ae`. Fixes 1-8 change `explore_h_s89.py` (the parent: docstring and report path) only. That file is not in the child's code SHA (the 51 modules listed in the log). The stored ORACLE runs and the passed CHECKs therefore stay valid, no run's bit stream changes, and ORACLE can keep running. They should land before the second invocation that queues the learned arms.
