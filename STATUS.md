# Project status — concluded on 1 November 2026

This file is the entry point for anyone picking the project up. It says what the project is, what it found, what is
pre-registered and what is not, where every thread stood when the work stopped, and how to resume it. The technical
report, **Revision 11 (final), `multichannel_hebbian_report_v11.pdf`**, is the account; the README follows it; this
file covers what the report does not. The author concluded the project on 1 November 2026, by choice, with two
pre-registered tests specified and ready to run and one exploratory frontier named but not crossed (§4). Nothing here
is abandoned mid-run: every screen that started has its report, every report was read by the referee or by a check
the report names, and every record is on its branch.

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
| `test_delta_recipe` | Complete on both machines on 13 October (Revision 10, §11). Pooled over 40 pairs: the delta-rule write with a fixed write strength binds eight streams as often as the Hebbian write (35/40 against 35/40, *d* = 0; the pre-fixed label "delta not shown worse by more than 2 pairs"), it needs no split to do so (no trigger 38/40; the split's claim NOT SHOWN, NO ROOM), and the Hebbian write needs the split (35/40 against 17/40, 19 vs 1, *p* = 2×10⁻⁵). Readings: "delta: no room; Hebbian: helps". |

**Independently checked:**

- A referee session recomputed every number of the report's tables from the records (Revision 8: 140 of 156
  reproduce, the rest statements, corrected), audited the pre-registration of every claim (none edited after a run),
  compiled the claim registry (53 pre-registered tests per machine), checked seven versions of Revision 9's draft
  before passing it (`docs/review_2026-10/rev9/`), found nine false statements in the merged Revision 10 (corrected in
  10.1 and 10.2, the latter passed before its merge; `rev10/`), and checked every exploratory report of rounds 5–7
  against its records (`check_*_report*.md`). Revision 11 was checked by a separate checking agent, not by the referee;
  the report says so.
- A clean-room replication from the report's text alone (session R, its own implementation) bound eight streams 17/20
  against 7/20 without the split, 18/20 on twenty further seeds with the split's probe and key positions taken from
  our code, and reproduced `test_split_copy`'s three counts on fresh seeds (18/20, 9/20, 0/20; the delta arm from the
  specification alone).
- The headline arms ran on a third machine class, a GPU with CUDA arithmetic, on fresh seeds: the recipe 19/20, no
  split 6/20, the recurrent gate 0/20 (descriptions only, by that run's pre-run note).

**Exploratory, not claims** (screens of ten to twenty seeds on the sessions' branches; a screen's finding becomes a
claim only when a main-line test repeats it). Through round 5 (Revision 10): the recipe's constants are scale-free
to four times the width and twice the depth; at a distance, a register that overwrites at the context token is found
label-free on a fifth of seeds and on every seed when a hinge holds key splits off during the slow phase; a hinge that
reads only a frequency table of token ids does most of that work (17/20); a split trigger that reads no task structure
keeps binding within two pairs of the key-position rule on 30 seeds; under a dense next-token loss one channel learns
to carry the source at two sources and nothing routes at four on the one screen run, untested by its own rule. Rounds
6–7 (19–23 October; Revision 11 §11–12), which removed the task's marked keys and then its fixed context identities:

- With routing given, a **Hebbian channel cannot bind four keys drawn from 32 types** on the lite task (the perfect gate
  0/2 at 24,000 and 48,000, under either recipe) where **a delta channel binds in 2400–3600**; the short convolution
  restores the Hebbian oracle (POOL 2/2, ALLPOOL 2/2 at 48,000). One delta channel with no gate binds the two-document
  task whose contexts are drawn from 16 types **10/10** (F, reports 9–10).
- On the lite task with pooled keys, a hinge that reads only **which token types repeat within the sequence** — no
  labels, no corpus table — equals the labelled key hinge: 20/20 against 20/20 on delta memory, where the hingeless
  register already binds 16/20; **19/20 against 18/20 on Hebbian memory with the convolution**, where the register
  alone binds 9/20 and the hinge is needed (*p* = 0.002). On the original lite task (Hebbian, no convolution) it falls
  short, 17/20 against 20/20 (G, reports 8–10). The frequency hinge hurts on delta (7/20) and survives on Hebbian (18/20).
- When the **context tokens have no fixed identity** (drawn from 16 types), every learned arm is 0/20 or 0/10 on both
  memories, with two channels and with sixteen, while the perfect gate handed the per-sequence index binds in 1200–9600:
  no learned gate found the relative routing; a comparing gate and a novelty signal did nothing (G, reports 8, 9, 11;
  the referee: a learning negative, not a capacity limit).
- On the eight-stream grouped layout with keys from 32 types, the main line's recipe binds **1/10** where it binds 38/40
  at four keys; neither a later first check nor a larger split cap helps (1/10, 2/10 against a 2/10 rerun); the delta
  perfect gate is itself unreliable at pools of 16 and 32. At the largest pool where both oracles are reliable (8),
  **the delta recipe with the split carries, 10/10**, the Hebbian recipe 6/10, and the delta recipe without a trigger
  7/10 (H, reports 10–12).

## 3. Where things are

| place | what it holds | final state |
|---|---|---|
| `claude/bdh-repo-curl-obgdqk` (default) | the merged main line | equal to the main line after the Revision 11 merge |
| `claude/bdh-growth-hebbian-inference-w90069` (the main line) | the tests (`test_*.py`), their specifications (`specs/`), the records (`results/`), the report, the referee's files, this file | Revision 11 (final) merged |
| `claude/rev9-draft`, `claude/rev10-draft`, `claude/rev11-draft` | the versions of each revision's draft that were checked | merged into the main line; kept because the report cites their commits |
| `claude/review-V` | Referee V's working branch (`docs/review_2026-10/` only) | merged in full; its last file is `check_G_report11.md` (23 October) |
| `claude/outside-ideas` | session E: batches 1–21 of exploratory screens (`explore_*.py`, `explore_out/`, verdicts in `explore_out/README.md`) | batch 21 complete (15 October); `explore_out/spec_draft_keyload.md` carries a NOT READY status section with five unscreened options |
| `claude/explore-F` | session F: delta-rule channels, forget gates, the lite task, the trigger on delta; the pool and context-pool tasks (reports 1–10) | complete; the task module G and H vendor is `explore_f_tasks.py` at `9017b82` |
| `claude/explore-G` | session G: distant cues, the register, the hinges (reports 1–11) | complete; `explore_out/G/spec_draft_register.md` revision 2 is "ready for adoption after ten text fixes" (`spec_review_register_v3.md`), not adopted (see `specs/drafts/`) |
| `claude/explore-H` | session H: anti-collapse devices, the copy, the split's constants, two streams, the split on pooled keys (reports 1–12) | complete; report 12 (S92) is the one exploratory report the referee did not check; `explore_out/H/spec_draft_two_keyterm.md` awaits revision |
| `claude/explore-I` | session I: the dense-loss task (S74, S75), the GPU package (`explore_gpu/`), the GPU run of `test_window_gate` (report 3), S75b | S75b specified (Appendices A–C), both cards' CUDA CHECKs and benchmarks on the final driver, the pre-run note committed (`005a59a`); **the pilot was never run** |
| `claude/replicate-R` | session R: the clean-room replication, Phases 2–5, `UNSTATED.md`, `replicate_report.md` | Phase 5 complete |
| `docs/reading/parallel_sessions_prompts.md` | every prompt each session was given, round by round | round 7 is the last |
| `specs/drafts/` | specifications drafted by the sessions and reviewed by the referee but never adopted into `specs/` or run | the register test (G, revision 2) |

## 4. What was left ready, and what was left open

Two tests were ready to run when the author stopped. Either can be run by whoever resumes, in the order below, and
nothing about them needs re-deciding:

1. **The register test** (`specs/drafts/test_register_draft_rev2.md`, G's revision 2; the referee's verdict
   `docs/review_2026-10/spec_review_register_v3.md`: "ready for X's adoption after ten text fixes", listed there). It is
   the main line's fifth pre-registered test and the one that answers the project's own open question — a distant cue
   discovered by a register gate — with a labelled hinge; its label-free arm was decided against after S90 (the repeat
   hinge 17/20 against 20/20 on the test's own setting), and that decision is dated in the draft. To run it: take the
   ten fixes, move the file to `specs/test_register.md`, write `test_register.py` to the discipline in `CONTRIBUTING.md`
   (the referee reviews the test file against the spec before any run), then X and L with disjoint seeds.
2. **S75b** (`explore_out/I/s75b_spec_draft.md` with Appendices A–C; `explore_gpu/README.md` (d) has every command).
   Both cards are checked and benchmarked on the final driver and the pre-run note is committed. The pilot is
   `python -m explore_gpu.s75b --stage 1 --device cuda` on the 1660 (about three hours); stage 2 on both cards
   (about a day and a half) only if the stage-1 record says PROCEED. Session I reads the record.

Three threads were left open with no runnable next step:

3. **Contexts without fixed identity.** The one frontier rounds 6–7 named and did not cross: when the cue token is
   drawn from a pool, every learned gate fails while the perfect gate succeeds, at two channels and at sixteen. A
   comparing gate and a novelty signal (G, S91) did nothing. Whoever takes this up starts from G's report 11 and F's
   report 10 (the cap table: with a channel per context type the capacity problem vanishes, so what remains is learning).
4. **The two-stream key-term test** (H's draft): underpowered against a 69/80 baseline (`spec_review_two_keyterm_draft.md`).
5. **The key-load test** (E's draft): no validity check at its budget (batch 21); five options, none screened.

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
`start --again`); the README's "Repository layout and how to run the tests" has the commands. Two lessons from
October: do not run a GPU job beside a main-line test on the laptop (15 GB is not enough; it lost a start to the
OOM killer), and start a test's L half with `--also results/X/<test>_results.json` so the pooled block is printed in
the run itself (`test_delta_recipe`'s was regenerated afterwards with `--report --also`).

**The report and the README.** The report is the tex in the root (`pdflatex` twice). The README is generated from it:
`python3 tools/readme/build_readme.py` (needs pandoc) rebuilds it from the tex and the hand-written parts in
`tools/readme/parts/` (the Summary, the status section, the repository layout, the references), which are updated by
hand with each revision.

## 6. Housekeeping

- `Test cross channel result.odt` in the root is an early note of the author's, kept as found.
- `LICENSE.md` is Pathway's; an attribution line for this fork is the author's decision.
- The exploratory branches are not merged into the main line, by design; their records are large.
- The laptop clone's git identity carries a clock six hours ahead of UTC (its commits of 22 October on
  `claude/explore-I` are stamped UTC+6); the server's is right.
