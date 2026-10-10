# S75 at S = 4 on a GPU: runtime estimate (from a CPU profile; EXPLORATORY)

Configuration: ICMC S=4 V=4 NB=16 block words 6..14 M=176 words/source L=768; B = 32; N = 256, D = 32, 3 layers, k = 4; 24,000 updates, 20 evaluations; runs: ORACLE 2, SINGLE 10, WIDE_SPLIT 10, REG3 10 (32 runs). Method and assumptions: explore_gpu/profile_s75.py's docstring. `python -m explore_gpu.checks --device cuda` (K9) measures the real numbers on the GPU.

## One training update, per run (CPU profile)

| arm | GFLOP | GB moved | kernel launches | peak memory GB (update) | eval chunk GB | saved activations GB | CPU s/update |
|---|---|---|---|---|---|---|---|
| ORACLE | 108.8 | 8.60 | 374 | 0.60 | 0.35 | 0.56 | 2.25 |
| SINGLE | 108.7 | 7.24 | 341 | 0.53 | 0.28 | 0.49 | 2.12 |
| WIDE_SPLIT | 109.4 | 10.77 | 558 | 0.89 | 0.34 | 0.79 | 2.71 |
| REG3 | 112.8 | 12.47 | 698 | 1.28 | 0.42 | 1.18 | 3.42 |

## Estimated GPU time per update and memory

Memory per run: the update's peak (parameters, gradients, Adam state and the update's live tensors) x 1.15 for the allocator's rounding and fragmentation; one evaluation chunk of 32 sequences beside it; 0.6 GB CUDA context.

| GPU | arm | s/update (low - high) | runs that fit at once (R) | 24k updates, one run (h, low - high) |
|---|---|---|---|---|
| GTX 1660 SUPER (6 GB) | ORACLE | 0.057 - 0.086 | 7 | 0.4 - 0.6 |
| GTX 1660 SUPER (6 GB) | SINGLE | 0.052 - 0.078 | 8 | 0.3 - 0.5 |
| GTX 1660 SUPER (6 GB) | WIDE_SPLIT | 0.068 - 0.100 | 4 | 0.5 - 0.7 |
| GTX 1660 SUPER (6 GB) | REG3 | 0.075 - 0.112 | 3 | 0.5 - 0.7 |
| RTX 3070 Laptop (8 GB) | ORACLE | 0.036 - 0.054 | 10 | 0.2 - 0.4 |
| RTX 3070 Laptop (8 GB) | SINGLE | 0.032 - 0.048 | 11 | 0.2 - 0.3 |
| RTX 3070 Laptop (8 GB) | WIDE_SPLIT | 0.044 - 0.066 | 6 | 0.3 - 0.4 |
| RTX 3070 Laptop (8 GB) | REG3 | 0.050 - 0.075 | 4 | 0.3 - 0.5 |

## Evaluation

The CPU part of one evaluation (explore_gpu.run.eval_cpu_part: routing on 256 sequences with the float64 margin, the probe's four logistic regressions, SET accuracy) took 88 s (WIDE_SPLIT) and 90 s (REG3) here, 1 thread. On the GPU path it runs in --stats-workers CPU processes (default 3) while training continues: one evaluation per run every 1200 updates, i.e. every 38 s or more per run, so three workers keep up (the probe needs 90 s / 3 workers). The evaluation forwards (1024 + 256 + 256 sequences, forward only, ~1/3 of an update per 32 sequences) cost ~16 updates' worth of GPU time per evaluation: 20 evaluations add ~1.3% to a run.

## S75 at S = 4, all 32 runs

- **GTX 1660 SUPER (6 GB)**: 14 - 20 GPU hours of training (+1.3% for the evaluation forwards; the evaluation statistics run on the CPU alongside). Stacking R runs costs about the same GPU time per run (L = 768 already fills the GPU); it hides Python and launch overhead.
- **RTX 3070 Laptop (8 GB)**: 9 - 13 GPU hours of training (+1.3% for the evaluation forwards; the evaluation statistics run on the CPU alongside). Stacking R runs costs about the same GPU time per run (L = 768 already fills the GPU); it hides Python and launch overhead.

Caveats: the bracket is wide because consumer-GPU SGEMM efficiency on these shapes (batched 767 x 767 x 256 GEMMs) and the eager-mode launch overhead are not known here; the peak-memory column counts live tensor bytes (the CUDA caching allocator adds block rounding and fragmentation, taken as 15%); K9 on the GPU replaces all of this with measurements in a minute.
