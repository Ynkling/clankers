# explore_out — EXPLORATORY, not a result

Outputs of the outside-ideas screens on branch `claude/outside-ideas` (code: `explore_*.py`).
Nothing here is a result. A screen that comes out promising is handed back to the main branch
for a pre-registered test.

| file | what |
|---|---|
| `batch1_aborted_threads.log` | first launch of batch 1: stopped by the reproduction check. It ran in torch's default 4 threads and did not match X; at 1 thread it matches X bit for bit. |
| `batch1.log` | batch 1 (S1 local_gate, S2 kwta, S3 aux_gate), git 8544207, 1 thread per run. **In progress until the report and verdicts appear at the end.** |
| `<screen>_results.json` | every run's record (run_one's fields plus `lr`, `secs_wall`); `meta.provenance` holds CPU, torch, git, threads. **Partial while batch 1 runs.** |
