# explore_gpu — the project's model and tasks on one GPU (EXPLORATORY tooling)

**What it is.** A device-agnostic runner, new files only; no project module is edited, and project modules are wrapped
where they need it.
- **Sequential path** (`device.py`): the project's own code, with the model's parameters on the GPU. It can run any
  main-line test's arm through that test's own `run_job`.
- **Batched path** (`batched.py`, `run.py`): *R* runs of one arm stacked into one model, trained with a per-run Adam. It
  covers the BDH stack with *k* channels, the LOCAL3 / WIDE window gates, the recurrent and perfect gates, Session G's REG3
  (`explore_i_common.Reg3Conv`), the Hebbian memory and the delta memory (`explore_delta_mem`), SLOW, and the split (row
  copy with noise, W_g's Adam state zeroed) or reset-only. Tasks: ICMC, RandHeaderTask-lite and the grouped BindTask.
- **Numerics and seeding:** float32 with TF32 off. Seeding follows the harness: models are built on CPU by the project's
  builders, batches, evaluation and probe sets come from the project's CPU generators, and only the arithmetic runs on
  the GPU.

**Install** (Python ≥ 3.11; an NVIDIA driver that supports the wheel's CUDA):

    pip install torch==2.14.0 numpy scikit-learn   # PyPI's Linux wheel is built for CUDA 13.0 (driver >= 580)

With an older driver, install the CUDA 12.x build of the same torch from pytorch.org's selector (`--index-url
https://download.pytorch.org/whl/cu126`). The GTX 1660 SUPER (sm_75) and the RTX 3070 Laptop (sm_86) are supported by
both. Run every command from the repository root on branch `claude/explore-I`.

**(a) The CUDA CHECK** (about 1–2 h; it writes `explore_gpu/checks_cuda_<GPU>_output.txt` and `.json`. Push or send
them back):

    python -m explore_gpu.checks --device cuda

Training here is *chaotic*: a one-ulp change to arm A's initial parameters moves its logits by 3e-3 after 100 updates
and by O(1) after 200 (K0, measured on CPU). So two paths agree to 1e-5 over 2400 free-running updates only if they are
bit-identical. On CPU they are (K2: 0.0 over 2400 updates, and for every Hebbian model in K2b). On CUDA the batched and
sequential paths batch their GEMMs differently, so free-running agreement is printed, not asserted.

The CUDA CHECK is **lock-step**: before each of 2400 updates, the batched (or CUDA) model is set to the sequential (or CPU)
model's parameters, and both compute logits and gradients on the same batch.

| CHECK | Comparison | Asserted tolerance |
|---|---|---|
| K3 | batched vs sequential on the GPU (arm A, 4 runs, 2400 updates) | logits ≤ 1e-5, gradients ≤ 1e-4 relative |
| K3b | the same for each S75 arm at L = 768 and for the delta memory | logits ≤ 1e-5, gradients ≤ 1e-4 relative |
| K4 | the GPU vs the CPU (arm A, 2400 updates; the S75 arms) | logits ≤ 1e-4, gradients ≤ 1e-3 relative |
| K1 | X's recorded run `A|160` on the GPU | printed only; a GPU run cannot reproduce it bit for bit |
| K9 | benchmark: time per update and peak memory of each S75 arm with *R* = 1 and 2, projected S75 hours | printed |

**(b) S75 at S = 4** (`s75_s4.py`; the readings are fixed in its docstring before any run). The task is ICMC with S = 4,
V = 4, blocks of 6–14 words, NB = 16, M = 176 and L = 768 (per source, S74's C1 exactly: Bayes bound 0.93). Seeds
830–839, 24,000 updates.

| Arm | Seeds | Role |
|---|---|---|
| ORACLE | 830–831 | validity: both must reach SET ≥ 0.9 |
| SINGLE | 830–839 | the interference baseline |
| WIDE_SPLIT | 830–839 | WIDE window gate, k = 4 + SLOW + split |
| REG3 | 830–839 | register gate at all three layers, k = 4 + SLOW |

S75's rule picks the learned arms by their S = 2 counts. S = 2 never ran, so the docstring fixes WIDE_SPLIT and REG3
instead: they are the two label-free gates that can see a block's source token at every word.

The reading is **"carries to four sources" if the better of WIDE_SPLIT and REG3 is ROUTED on ≥ 8/10**. ROUTED means η²
by source at word positions ≥ 0.9 and a one-to-one source → channel map at every gate layer.

    python3 longrun.py start s75_S4 -- python3 -u -m explore_gpu.s75_s4 --device cuda      # detached, resumable
    python3 longrun.py watch s75_S4 --minutes 9 --resume                                   # follow it
    python -m explore_gpu.s75_s4 --report-only     # counts table -> explore_out/I/gpu/s75_S4_summary.md

`--arms ORACLE,SINGLE` on one machine and `--arms WIDE_SPLIT,REG3` on the other share the work. Each machine writes
`explore_out/I/gpu/s75_S4_<GPU>_results.json`, and the report merges both. Records are cached and a restart resumes.
`--runs-per-batch R` stacks *R* runs; the default is 1 on 6 GB and 2 on 8 GB. `--stats-workers` sets the number of CPU
processes for the evaluation statistics; the default is 3.

**(c) Any main-line test's arm on the GPU**, through the test's own `run_job` with its own seeding and record format:

    python -m explore_gpu.mainline --test test_window_gate --arm WIN3_SPLIT_D8 --seeds 540-549 --device cuda --workers 3
    python -m explore_gpu.mainline --list test_split_copy     # its arms and seeds

Records are keyed `ARM|seed` and carry `device` (GPU model, CUDA, torch, CPU) and `gpu`. They are written to
`explore_out/I/gpu/mainline/<test>_<GPU>_results.json`, never to `results/X` or `results/L`. The meta names the machine
`GPU:<model>`, so no `--also` pooling mistakes these records for X's or L's. A GPU run is a new run of that arm and seed:
compare outcomes, not curves. `--workers N` shares one GPU between N processes; the L ≤ 99 models are small.

**Runtime estimate for S75 at S = 4** (`runtime_estimate.md`; from a CPU profile of one update, with the assumptions
stated there; K9 measures the real numbers):

__RUNTIME__

**Files.**

| File | What it holds |
|---|---|
| `device.py` | the sequential path and `device_scope` |
| `batched.py` | `BatchedBDH`, `StackedAdam` |
| `run.py` | tasks, arms, trainers, evaluation (the probe's statistics in CPU workers) |
| `s75_s4.py` | S75 at S = 4 |
| `mainline.py` | main-line arms on a device |
| `checks.py` | the CHECKs |
| `profile_s75.py` | the estimate |
| `checks_cpu_output.txt`, `.json` | this container's CHECK outputs |
| `../explore_out/I/gpu/audit.json` | the read-only audit of the project's CUDA hazards behind `device.py` |
