# explore_gpu — the project's model and tasks on one GPU (EXPLORATORY tooling)

**What it is.** A device-agnostic runner made of new files only: no project module is edited, and wrappers handle what
needs it. Float32, TF32 off. Seeding follows the harness: models are built on CPU by the project's builders, and batches,
evaluation and probe sets come from the project's CPU generators. Only the arithmetic runs on the GPU.

- **Sequential path** (`device.py`): the project's own code with the model on the GPU; (c) uses it.
- **Batched path** (`batched.py`, `run.py`): *R* runs of one arm stacked into one model, with a per-run Adam. It covers:
  - BDH with *k* channels;
  - the LOCAL3 / WIDE window, recurrent, perfect and REG3 gates;
  - Hebbian and delta memories.

  The trainers support ADAM, SLOW and the split on ICMC, and ADAM only on RandHeaderTask-lite and the grouped BindTask.
  Other recipes (reset-only, Muon, curricula) run on the sequential path.

**Install** (Python ≥ 3.11), then run every command from the repository root on branch `claude/explore-I`:

    pip install torch==2.14.0 numpy scikit-learn   # PyPI's Linux wheel: CUDA 13.0, driver >= 580

On an older driver, use the CUDA 12.6 build instead (`--index-url https://download.pytorch.org/whl/cu126`). Both builds
support the GTX 1660 SUPER (sm_75) and the RTX 3070 Laptop (sm_86).

**(a) The CUDA CHECK.** It takes about 1–2 h and writes `explore_gpu/checks_cuda_<GPU>_output.txt` and `.json`; send
both back. A failing CHECK does not stop the others, and the last line names every failure.

    python -m explore_gpu.checks --device cuda

Training is *chaotic*. A one-ulp change to arm A's initial parameters moves its logits by 3e-3 after 100 updates and by
O(1) after 200 (K0). So two free-running paths agree to 1e-5 over 2400 updates only if they are bit-identical. On CPU
they are bit-identical for every Hebbian model (the next paragraph). On CUDA they are not, because cuBLAS batches GEMMs
differently.

The CUDA CHECK is therefore **lock-step**: before every update, the model under test is set to the reference's parameters,
and both compute logits and gradients on the same batch.
- **Tolerance.** An update passes if max |Δlogits| ≤ tol_logits and, for each parameter, ‖Δg‖ ≤ tol_grad · ‖g_ref‖, using
  that parameter's own Frobenius norm.
- **Arbitration.** Otherwise a float64 copy of the reference computes the same update. The update counts as *test-worse*
  (A) if the tested path is farther from float64 than 4× the reference's own float32 error plus the tolerance. It counts
  as *reference-worse* (B) in the mirror case.
- **Pass.** A case passes if no value is non-finite and a one-sided sign test does not reject "equally accurate":
  P(Binomial(A + B, ½) ≥ A) ≥ 1e-4.

A fixed bound alone would fail correct code. Two float32 paths that round differently sometimes disagree by more than any
tolerance, and the reference is as often the inaccurate one:
- a relu pre-activation within rounding of zero, active in one path only (about 6 million relu units per update at
  L = 768);
- the delta memory's ill-conditioned solve;
- a cancelling sum.

A wrong gradient makes nearly every update A and none B.

| CHECK | Comparison on CUDA | tol_logits, tol_grad |
|---|---|---|
| K3 | batched vs sequential, both on the GPU: arm A, seeds 160–163, all 2400 updates | 1e-5, 1e-4 |
| K3b | the same for each S75 arm at L = 768 and for the delta memory, 300 updates | 1e-5, 1e-4 |
| K4 | GPU batched vs CPU sequential: arm A 160, 2400 updates; each S75 arm, 20 updates | 1e-4, 1e-3 |
| K4b | GPU sequential as (c) runs it vs CPU sequential, the same runs | 1e-4, 1e-3 |
| K6 | the per-run Adam vs `torch.optim.Adam`, including the split's state reset | 1e-6 |
| K1, K2, K2b, K2c, K9 | X's record `A\|160`; free-running agreement; the benchmark (time and memory per arm, R = 1 and 2) | printed |

On CPU (`checks_cpu_output.txt`, this container) these are asserted bit for bit:
- K1: X's recorded `A|160` is reproduced exactly.
- K2, K2b, K2c: batched and sequential agree exactly in free-running training, over 2400, 600 and 7200 updates; the 7200
  include a split that fires.
- K5: the ICMC trainer equals `explore_i_common.run_one`, and the register's scan equals its loop.

K3c (the criterion's controls) also runs on CPU, and `calibrate_judge.py` runs the CUDA lengths on CPU
(`calibrate_judge_output.txt`). Two correct paths that round differently pass, with A ≈ B:

| Case | Run-updates | A | B | p |
|---|---|---|---|---|
| REG3's register scan vs its loop, L = 768 | 600 | 21 | 24 | 0.72 |
| arm A vs its neuron-permuted twin | 2400 | 5 | 3 | 0.36 |
| the delta memory (GDN) | 900 | 0 | 2 | 1 |

Wrong gradients fail at every update (A = 20, B = 0, p = 1e-6): a zeroed W_in, W_h or W_g gradient, or W_g's gradient
scaled by 1.01. This includes W_in at L = 768, where its gradient is ~1e-7 of the largest.

K7 checks the data path. K8 runs every forward on fake CUDA tensors to show that no CPU tensor is mixed in.

**(b) S75 at S = 4** (`s75_s4.py`; its docstring fixes the readings before any run). The task is ICMC with S = 4, V = 4,
blocks of 6–14, NB = 16, M = 176 and L = 768. Per source this is S74's C1 (Bayes bound 0.93). Each run is 24,000
updates.

The arms:
- ORACLE, seeds 830–831: validity (both must reach SET ≥ 0.9).
- SINGLE, seeds 830–839.
- WIDE_SPLIT, seeds 830–839: WIDE window, k = 4, SLOW and the split.
- REG3, seeds 830–839: REG3, k = 4, SLOW.

S75 picks its learned arms by their S = 2 counts, but S = 2 never ran. WIDE_SPLIT and REG3 are fixed instead: they are
the two label-free gates that see a block's source token at every word.

**Reading:** "carries to four sources" if the better of WIDE_SPLIT and REG3 is ROUTED on ≥ 8/10. ROUTED means, at every
gate layer, η² by source ≥ 0.9 at word positions and a one-to-one source → channel map.

    python3 longrun.py start s75_S4 -- python3 -u -m explore_gpu.s75_s4 --device cuda --arms WIDE_SPLIT,REG3   # 8 GB laptop
    python3 longrun.py start s75_S4 -- python3 -u -m explore_gpu.s75_s4 --device cuda --arms ORACLE,SINGLE     # 6 GB server
    python3 longrun.py watch s75_S4 --minutes 9 --resume                                   # follow; restarts resume
    python -m explore_gpu.s75_s4 --report-only     # merge -> explore_out/I/gpu/s75_S4_summary.md

Each machine writes `explore_out/I/gpu/s75_S4_<GPU>_results.json` and a summary. Records are cached by code SHA and GPU.

When a run fails:
- a failed batch (e.g. out of memory) is retried with half as many runs;
- a single run is retried once;
- the command exits non-zero, and the summary lists runs without an ok record.

When both machines have finished (both on the same commit), push one machine's results file and pull it on the other.
Then run `--report-only`; it lists records from another code SHA and does not count them. Without `--arms`, a machine
runs all four arms.

Flags:
- `--runs-per-batch R` (default 1 on a 6 GB card, 2 on 8 GB);
- `--stats-workers N` (default 3; the CPU processes for the evaluation statistics);
- `--extended` adds WIN3_SPLIT, WIDE_SLOW and REG3_NUDGE, which are printed and never read.

**(c) A main-line test's arm on the GPU**, through the test's own `run_job`, with its seeding and record format:

    python -m explore_gpu.mainline --test test_window_gate --arm WIN3_SPLIT_D8 --seeds 540-549 --device cuda --workers 3
    python -m explore_gpu.mainline --list test_split_copy     # its arms and seeds

All 16 tests that have `run_job` and `spec` are supported (`mainline.py` lists them). Phase I–III,
test_router_reliability and test_readout_path are not. Records are keyed `ARM|seed`. Each carries the device (GPU model,
CUDA, torch, CPU), the spec and the code SHA, and is reused only when spec and code match. They go to
`explore_out/I/gpu/mainline/<test>_<GPU>_results.json`, with machine `GPU:<model>` in the meta, never to `results/X` or
`results/L`. A GPU run is a new run of its seed: compare outcomes, not curves.

**(d) S75b** (`s75b.py`; spec `../explore_out/I/s75b_spec_draft.md`, approved at 114cd08, with Appendices A and B
of post-approval changes, none of them blind).

**General rules.**
- Run every step from the repository root, on an up-to-date checkout of `claude/explore-I`.
- Commit and push each step's output before the next step: `git add explore_out/I/gpu/s75b` takes everything there,
  including `bench_archive/` and any archived stage-1 record. The driver refuses to go out of order. With commits
  required, the stages also refuse when `git fetch` fails, when the checkout is behind or ahead of its upstream, or when
  an S75b JSON file is uncommitted.
- Any change to a module in the code SHA changes it: `s75b.code_sha()[1]` lists the 33 modules, including
  `explore_gpu/run.py`, `explore_i_*` and `test_*`. The CUDA CHECKs and the benchmarks must then be rerun on that commit.
  - Before stage 2, the first stage 1 under the new code needs `--code-change-note PATH` (a committed, dated note).
  - After stage 2 starts, a change halts the screen.
- **Do not pull or edit code while a stage runs.** Each run's workers load the code from disk, so the driver compares
  it before and after every run. A change stops the invocation, and a change during a run halts the screen for you.
  - If the push at the start of a stage is rejected because the other machine pushed first, run `git fetch` and then
    `git diff --stat HEAD @{u}`. If that lists no `.py` file, `git pull --no-rebase` and push again. If it does, stop
    and tell the user at once: that code change must be reverted before the other machine runs any stage, and only then
    is this start pulled and pushed.
- **One invocation per card.** A second stage or benchmark on a card refuses while the first runs, from any checkout on
  that machine. A stage also refuses while the other card's last invocation, as pushed, has not ended and runs any of
  the same arms.
  - If that process was killed: first commit and push that card's results file from its own machine, because it holds
    the runs that finished before the kill. Then pull, and add `--other-card-idle` (recorded) once you are sure no S75b
    process runs there.
  - If that machine cannot be reached, do not fall back: tell the user, because its unpushed runs would be drawn again.

1. **The CUDA CHECK**, on each card while no other job is on that GPU. Then commit
   `explore_gpu/s75b_checks_cuda_<GPU>_output.*`. It takes about 2 minutes. It runs B7, the lock-step, and B9, a real
   divergence, on the card.

       python -m explore_gpu.s75b_checks --device cuda

2. **The benchmark**, on each card. Then commit `explore_out/I/gpu/s75b/s75b_bench_<GPU>.json`.
   - `--host-ram-gb` is the machine's RAM: 15.2 on the laptop, and the user's figure on the server.
   - `--beside` names a string in the command line of the job that will run beside S75b. That job must be running
     during the benchmark, so its peak memory is measured; use `none` if nothing will.
   - The benchmark may be rerun at any time; earlier files go to `bench_archive/`.
   - Usability is per arm: an arm runs on a card if its latest benchmark finished, passed that arm's cases, and its
     host-memory rule gave workers. The pilot needs only ORACLE's case.

       python -m explore_gpu.s75b --bench --device cuda --host-ram-gb 15.2 --beside test_delta_recipe   # laptop
       python -m explore_gpu.s75b --bench --device cuda --host-ram-gb <server RAM> --beside none        # server

3. **The pilot**, on one card: the 1660 after the window_gate run, or the 3070 if that is free first. The other card
   waits until the stage-1 record is pushed.
   - As soon as it starts, commit and push the results file and the stage-1 record, which is marked running. The
     other card's stages then refuse while the pilot runs.
   - When it ends, commit and push the stage-1 record (`.json` / `.md`) and every results file, whatever the decision.
   - Stage 2 refuses a record still marked running; after a hard kill, run stage 1 again.
   - A run below 0.9, or a diverged run, is a final STOP.
   - Any other failure stops the invocation. Re-invoke the same command, on either card: it reruns the same seed.
   - Two invocations with failures halt the seed. Then you decide: add `--retry-halted` to clear it, or fix the code.

       python -m explore_gpu.s75b --stage 1 --device cuda

4. **Stage 2**, only if the record says PROCEED. Commit and push the results file as soon as stage 2 has started, so the
   other machine sees the freeze and which arms run there.
   - If one card is unusable or unavailable, run everything on the other card, with a reason:
     `--machine fallback --reason "..."` on the 1660, or `--machine fallback-3070 --reason "..."` on the 3070.
   - A card runs only the arms its benchmark allows, and names the rest. To run just the refused arms on the other
     card, use the fallback with `--arms`, after that card's own stage 2 has ended. For example:
     `--machine fallback-3070 --arms WIDE_SLOW --reason "the 1660's benchmark refuses WIDE_SLOW"`.
   - A fallback for a card that died mid-stage needs `--other-card-idle` as well, after that card's results file has
     been pushed from its own machine (general rules).
   - Failures are handled as in stage 1: re-invoke, and use `--retry-halted` after a halt.
   - If the code changed on disk during a run, the screen halts. Restore the code, and record your decision on that
     card with `--code-changed-decision keep` (the run counts) or `rerun` (it is set aside, and its seed is rerun).
   - Spec Appendix B has the rules.

       python -m explore_gpu.s75b --stage 2 --device cuda --machine 1660      # ORACLE 850-851, SINGLE, WIDE_SLOW
       python -m explore_gpu.s75b --stage 2 --device cuda --machine 3070      # REG3, WIDE_SPLIT_G

5. **The report**, once both cards' results files are pulled: `python -m explore_gpu.s75b --report`, which writes
   `s75b_summary.md`.

**Runtime estimate for S75 at S = 4** (`runtime_estimate.md`; method and assumptions in `profile_s75.py`). It is a CPU
profile of one update per arm (FLOPs, bytes, kernel launches, live tensor memory) put through a roofline model of each
GPU with a stated efficiency bracket. K9 measures the real numbers.

| | GTX 1660 SUPER (6 GB) | RTX 3070 Laptop (8 GB) |
|---|---|---|
| s per update, one run (SINGLE … REG3) | 0.05–0.08 … 0.08–0.11 | 0.03–0.05 … 0.05–0.08 |
| one run of 24,000 updates | 0.3–0.7 h | 0.2–0.5 h |
| all 32 runs on this GPU alone | 14–21 h | 9–13 h |
| shared as in (b) | 4.3–6.4 h (ORACLE, SINGLE) | 6.4–9.5 h (WIDE_SPLIT, REG3) |
| memory per run incl. 15% slack; runs that fit | 0.6–1.5 GB; 3 (REG3) to 8 | 0.6–1.5 GB; 4 (REG3) to 11 |

Shared as in (b), S75 at S = 4 takes about 6.4–9.5 h of wall time; on one CPU thread a single run takes about 15 h. The
evaluation statistics, mostly the probe, take 40–90 s of one CPU core per evaluation of a run (here, depending on load).
They run in 3 worker processes beside the training and keep up, since a run is evaluated every 1200 updates (≥ 38 s of
GPU time).

**Files:**
- `device.py`, `batched.py`, `run.py`: the runner.
- `s75_s4.py`, `mainline.py`, `s75b.py`: (b), (c) and (d); `s75b_checks.py` and `s75b_checks_cpu_output.*`: (d)'s
  CHECKs and this container's outputs; `s75b_mutants.py` and `s75b_mutants_cpu_output.txt`: the mutation pass on (d)'s
  rules.
- `checks.py` and `checks_cpu_output.*`: the CHECKs and this container's outputs.
- `calibrate_judge.py` and its output: the lock-step criterion at the CUDA lengths.
- `profile_s75.py`, `runtime_estimate.md`, `profile_s75.json`: the estimate.
- `../explore_out/I/gpu/audit.json`: the audit of the project's CUDA hazards behind `device.py`.
