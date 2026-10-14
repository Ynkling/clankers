# Project status — paused on 14 October 2026

This file is the entry point for anyone picking the project up, including its author after a break. It says what the
project is, what it found, what is pre-registered and what is not, where every thread stood when work paused, and how
to resume. The technical report (Revision 9, `multichannel_hebbian_report_v9.pdf`) is the refereed account; the README
follows it; this file covers what the report does not yet.

## 1. What this is

A research fork of Pathway's BDH (Dragon Hatchling). BDH's Hebbian working memory is split into *k* channels, and a
per-token gate distributes each token's write over the channels and blends its read from them. On a synthetic task,
*S* streams bind the same keys to different values, cued by a stream token; one channel cannot hold the conflicting
bindings at four or eight streams, and a hand-set gate, one channel per stream, can. The question throughout has been
whether a gate trained only on the task loss finds that partition.

The project is run as a pre-registered study (`CONTRIBUTING.md`): every claim is written down before the runs that
test it, every run leaves a record that an outsider can re-score (`results/`), and the report's text is checked
against those records by a referee that had no part in the work (`docs/review_2026-10/`). The work was done by one
person orchestrating Claude Code sessions in named roles, on two CPU machines and two GPUs the author owns.

## 2. What was found

**Pre-registered, on two machines with disjoint seeds (the main line):**

| test | result |
|---|---|
| `test_window_gate` | A width-3 *window gate*, slow memory for the first 2400 updates and a plateau-triggered *channel split* bind eight streams, each stream on its own set of channels, in **38/40** runs, against 0/40 for the recurrent-gate recipe (38 vs 0 discordant pairs) and 15/40 without the split (23 vs 0). No stream labels, no restarts, no hinge; the split reads the task's key positions. |
| `test_split_copy` | The split's row copy, not its optimizer reset, is the active part: copy with reset 18/20 against reset alone 6/20 (12 vs 0, *p* = 0.00024); the reset alone is within the bound of no trigger (6/20 against 8/20). One delta-rule channel binds 0/20 where the recipe binds 18/20. |
| `test_delta_recipe` | Complete on both machines on 13 October (not yet in the report). Pooled over 40 pairs: the delta-rule write with a fixed write strength binds eight streams as often as the Hebbian write (35/40 against 35/40, *d* = 0; the pre-fixed label "delta not shown worse by more than 2 pairs"), it needs no split to do so (no trigger 38/40; the split's claim NOT SHOWN, NO ROOM), and the Hebbian write needs the split (35/40 against 17/40, 19 vs 1, *p* = 2×10⁻⁵). Readings: "delta: no room; Hebbian: helps". |

**Independently checked:**

- A referee session recomputed every number of the report's tables from the records (Revision 8: 140 of 156
  reproduce, the rest statements, corrected), audited the pre-registration of every claim (none edited after a run),
  compiled the claim registry (53 pre-registered tests per machine), and checked seven versions of Revision 9's draft
  before passing it (`docs/review_2026-10/rev9/`).
- A clean-room replication from the report's text alone (session R, its own implementation) bound eight streams 17/20
  against 7/20 without the split, 18/20 on twenty further seeds with the split's probe and key positions taken from
  our code, and reproduced `test_split_copy`'s results from the specification (18/20, 9/20, 0/20).
- The headline arms ran on a third machine class, a GPU with CUDA arithmetic, on fresh seeds: the recipe 19/20, no
  split 6/20, the recurrent gate 0/20 (descriptions only, by that run's pre-run note).

**Exploratory, not claims** (screens of ten to twenty seeds on the sessions' branches; a screen's finding becomes a
claim only when a main-line test repeats it): the recipe's constants are scale-free to four times the width and twice
the depth; at a distance, a register that overwrites at the context token is found label-free on a fifth of seeds and
on every seed when a hinge holds key splits off during the slow phase; a hinge that reads only a frequency table of
token ids, no labels, does most of that work (17/20); a split trigger that reads no task structure keeps binding
within two pairs of the key-position rule on 30 seeds; under a dense next-token loss one channel learns to carry the
source at two sources and nothing routes at four on the one screen run, which its own validity rule left untested.

## 3. Where things are

| place | what it holds | state at the pause |
|---|---|---|
| `claude/bdh-repo-curl-obgdqk` (default) | the merged main line | equal to the main line after PR #26 |
| `claude/bdh-growth-hebbian-inference-w90069` (the main line) | the tests (`test_*.py`), their specifications (`specs/`), the records (`results/`), the report, the referee's files, this file | Revision 9 merged; `test_delta_recipe` complete on both machines |
| `claude/rev9-draft` | the seven versions of Revision 9's draft that the referee checked | merged into the main line; kept because the report cites its commits |
| `claude/review-V` | Referee V's working branch (`docs/review_2026-10/` only) | merged at 00290a8; its last items were the reviews of the three spec drafts below |
| `claude/outside-ideas` | session E: batches 1–21 of exploratory screens (`explore_*.py`, `explore_out/`, verdicts in `explore_out/README.md`) | batch 20 complete; `explore_out/spec_draft_keyload.md` written; batch 21 (the key-load test's validity screen, S69) still running at 18:12 UTC on 14 October: 13 of 20 runs saved (`4fedf9c`), records in `explore_out/oracle_d128_results.json`, pushed after every run; its SUMMARY and reading come at the end of `explore_out/batch21.log` |
| `claude/explore-F` | session F: delta-rule channels, forget gates, the lite task, the trigger on delta (reports 1–7) | session closed on 14 October with S88 half done: S88a (the write strength) complete at `401d25a` (`explore_out/F/write_results.json`, `write_tables.md`); S88b (the horizon) stopped after 6 of its 44 runs (`horizon_results.json` at `99482c5`); no report 8 |
| `claude/explore-G` | session G: distant cues, the register, the hinges (reports 1–7) | S87 complete; `explore_out/G/spec_draft_register.md` awaits revision after the referee's review |
| `claude/explore-H` | session H: anti-collapse devices, the copy, the split's constants, two streams (reports 1–9) | round 5 complete; `explore_out/H/spec_draft_two_keyterm.md` awaits revision after the referee's review |
| `claude/explore-I` | session I: the dense-loss task (S74, S75), the GPU package (`explore_gpu/`), the GPU run of `test_window_gate` (report 3), S75b (`explore_out/I/s75b_spec_draft.md`, `explore_gpu/s75b.py`) | S75 closed; S75b's driver final (3cef609, mutation pass 8d2e89f), CPU CHECKs pass, the cards' CUDA CHECKs passed on the driver of 13 October and must be rerun on the final one; the pilot was not run |
| `claude/replicate-R` | session R: the clean-room replication, Phases 2–5, `UNSTATED.md`, `replicate_report.md` | Phase 5 complete |
| `docs/reading/parallel_sessions_prompts.md` | every prompt each session was given, round by round | round 5 is the last |
| `docs/reading/plan_2026-10-13.md` | the state and plan written the day before the pause | superseded by this file where they differ |

## 4. Open threads, and the next step of each

1. **Revision 10 of the report.** Not drafted. It folds in: `test_delta_recipe` pooled; the GPU run of the headline;
   R's Phase 5; G's S87; H's round 5; E's batch 20; S75 closed and S75b's state; the referee's reviews of the spec
   drafts. The referee checks the L half and pooled block of `test_delta_recipe` first (as it checked X's,
   `docs/review_2026-10/rev9/check_delta_recipe_X.md`), then the draft, pass by pass, until "ready to merge".
2. **The register test** (the next pre-registered test; the project's open question, a distant cue discovered
   label-free). G's draft specification exists; the referee's review (`docs/review_2026-10/spec_review_register_draft.md`)
   lists what must change (a named outcome function with a CHECK against the recorded verdicts, the validity rule's
   dependence on a nudged arm, the probe named, a seed-collision CHECK), and S87's frequency hinge belongs in it as an
   arm. Pipeline: G revises → X adopts into `specs/test_register.md` → the referee reviews → X writes the test → the
   referee checks it → X and L run it (seeds X 600–619, L 1600–1619).
3. **The two-stream key-term test.** H's draft is underpowered against a 69/80 baseline
   (`spec_review_two_keyterm_draft.md`); it needs more seeds, a harder configuration or a different primary claim before
   adoption. H's dose screen found the drafted constants robust.
4. **The key-load test** (sixteen keys at *D* = 128 against *D* = 32). E's draft (`explore_out/spec_draft_keyload.md`,
   budget 40,800 by its rule) is unreviewed; its validity rule was at risk because the *D* = 128 oracle on the one seed
   tried never bound, so E ran the validity screen first (batch 21, S69: the perfect gate at *D* = 128, sixteen keys,
   seeds 270–279, the recorded single-Adam recipe against SLOW, 40,800 updates; readings fixed before any run). State at
   the pause: the single-Adam arm is done at **4 of 10** bound (271, 273, 274, 278; three more ended at 0.91–0.92, just
   under the 0.95 bar) — "unreliable" by the pre-fixed rule, so two validity seeds would both bind with probability
   about 0.13; the SLOW arm stands at 0 of 3 (270–272 ended near 0.35; 271 had bound at 12,000 under single Adam)
   with seven seeds running or queued. The spec's reading now rests on SLOW: 5 or more of its remaining seven bind →
   the draft falls back to one validity seed at *D* = 128 (VALID_N = 1) on SLOW; fewer → the draft has no validity
   check at 40,800 and does not go to the referee as written. Whoever resumes reads the SUMMARY at the end of
   `explore_out/batch21.log` (or finishes the batch from its checkpoints if the container died) before touching the
   draft.
5. **S75b** (the dense loss at four sources on a configuration where the perfect gate can reach its bar). Approved spec
   with a mechanical pilot rule; what remains is in `explore_gpu/README.md` (d): rerun the CUDA CHECK and the benchmark
   on each card with the final driver, run the pilot (two perfect-gate runs, about three hours), and stage 2 on both
   cards (about a day and a half) only if the pilot record says PROCEED. The referee's review of the draft
   (`spec_review_s75b_draft.md`) has points to take before the pilot.
6. **F's S88**, half done when its session closed (readings fixed before the first run, `40087cb`; EXPLORATORY,
   not a result). **S88a**, the write strength (34 runs, 160 h, seeds 2400–2409, the delta recipe with the exact-copy
   split at eight streams): the fixed write strength binds and routes 10/10, a learned per-token strength 7/10 (0 vs 3
   discordant pairs) and the learned strength on the slow schedule 3/10 (0 vs 7, *p* = 0.0078); the pre-fixed reading
   is "no reading applies" (the better learned arm is 3 pairs behind the fixed one — more than the 1 pair that "keeps the
   partition" allowed, fewer than the 4 that "costs it" needed). Printed, not a reading: every learned run drives its
   write strength to about 0 by update 4800 and it never diverges by stream (0/20), so the learned arms are mostly
   writing nothing. Tables in `explore_out/F/write_tables.md`. **S88b**, the horizon (one delta channel on the lite
   task at 1× and 2× cue-to-key distance, fixed decay against the learned forget gate; 44 runs planned): 6 ROUTED_2X
   runs recorded (4 bound) before the session closed; no reading can be taken from a single arm. To finish it, a new
   session on `claude/explore-F` runs `explore_f_horizon.py` (resumes from the cached records) and writes report 8.
7. **Round 6 of screens**, proposed in `docs/reading/plan_2026-10-13.md` §4E, none started.

## 5. How to resume

**The method.** Read `CONTRIBUTING.md` (one page). Every test's docstring states its design and, after the run, its
result on each machine and pooled; `specs/` holds the specifications; `results/README.md` says which machine and
commit produced each record. Nothing is a result until the docstring carries the recorded outcome, and nothing from an
exploratory branch is a claim.

**The sessions.** Each role is a Claude Code session on its own branch (table above). To restart one, open a session
on that branch and give it the latest entry for its role in `docs/reading/parallel_sessions_prompts.md` together with
its branch's `explore_out/<role>/README.md` (or `explore_out/README.md` for E); the branches hold the state, and every
segment of a screen opens by reproducing a recorded run bit for bit. The referee is restarted the same way on
`claude/review-V` and must change nothing outside `docs/review_2026-10/`.

**The machines.**

| machine | clone | branch | environment |
|---|---|---|---|
| X (a cloud container) | the session's own | the main line | the session's |
| the laptop (i7-12650H, 15 GB; RTX 3070 Laptop GPU) | `~/Documents/clankers` | the default branch, pulled before each test | system Python with torch 2.14.1 (CPU); main-line tests run here with `--machine L` |
| the laptop, GPU work | `~/Documents/clankers-gpu` | `claude/explore-I` | `source ~/venv-gpu/bin/activate` (torch 2.14.0+cu130) |
| the server (Ryzen 5 3600, 35 GB; GTX 1660 SUPER) | `~/jellyfin/clankers` (single-branch) | `claude/explore-I` | `source .venv/bin/activate` in the clone (torch 2.14.0+cu130) |

Long runs go through `longrun.py` (`start`, `watch --resume`, `status`; a stopped run resumes from its records with
`start --again`); the README's "Repository layout and how to run the tests" has the commands. Two lessons from the
last week: do not run a GPU job beside a main-line test on the laptop (15 GB is not enough; it lost a start to the
OOM killer), and start a test's L half with `--also results/X/<test>_results.json` so the pooled block is printed in
the run itself (`test_delta_recipe`'s was regenerated afterwards with `--report --also`).

**The report and the README.** The report is the tex in the root (`pdflatex` twice). The README is generated from it:
`python3 tools/readme/build_readme.py` (needs pandoc) rebuilds it from the tex and the hand-written parts in
`tools/readme/parts/` (the Summary, the status section, the repository layout, the references), which are updated by
hand with each revision.

## 6. Housekeeping known at the pause

- `Test cross channel result.odt` in the root is an early note of the author's, kept as found.
- `LICENSE.md` is Pathway's; an attribution line for this fork is the author's decision.
- The exploratory branches are not merged into the main line, by design; their records are large.
