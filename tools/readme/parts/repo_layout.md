## Repository layout and how to run the tests

- **`bdh.py`:** Pathway's BDH model, used unmodified. The multi-channel model subclasses it:
  - it replaces the attention module, multiplying each score by the gate match at every layer;
  - it adds the gate's parameters (the recurrent gate of Phases II–V, or the window gate of Phase VI, converted in place);
  - with the convolution, it runs a copy of BDH's layer loop with the causal convolution inserted.
- **`test_*.py`:** one file per experiment.
  - Each test's docstring holds its pre-registered design (background, arms, seeds, claims and thresholds), and after the run, its recorded result on each machine and pooled.
  - Every Phase III–VI test first runs its inherited verification chain and its own numbered CHECKs, then prints a runtime projection, then trains.
  - Phase VI's tests are `test_muon_recipe.py`, `test_window_gate.py`, `test_split_copy.py` and `test_delta_recipe.py`; Phase V's are `test_stream_recipe.py`, `test_stream_curriculum.py`, `test_slow_start.py`, `test_recipe_scope.py` and `test_early_recipe.py`.
  - Tests from Phases I–II (for example `test_interference*.py`, `test_asymmetric_routing.py`, `test_instrument_v2.py`) and earlier modules (`bdh_multichannel.py`, `bdh_mc.py`, `bdh_recurrent.py`) are kept for the record.
- **`longrun.py`:** the detached run, its watcher and the resume and projection helpers (below). **`fail_class_v2.py`:** the failure classes with the key rule ahead of the margin rule, used by new tests; `docs/fail_class_v2.md` reclassifies the recorded two-stream runs (no recorded verdict changes).
- **`specs/`:** each Phase VI test's specification from `test_window_gate` on, committed before the test was written and, from `test_delta_recipe` on, reviewed by the referee session before any run; amendments are dated and state what had been seen when they were made. `CONTRIBUTING.md` states the discipline (specification, readings in the docstring, CHECKs, amendments) and `.github/` carries the pull-request and discrepancy templates.
- **`results/X/`, `results/L/`:** copies of each machine's result JSON files, with `results/README.md` recording each file's machine, commit and start time, and the run logs of the long tests beside them (`results/L/split_copy_L.log`; `results/L/longrun/`). Results files are gitignored where they are written. Both machines' files for every Phase III–VI test are recorded; L's Phase III–IV files were recovered in this revision ([15](#15-the-referees-report-and-the-claim-registry)).
- **`multichannel_hebbian_report_v10.pdf`** and **`multichannel_hebbian_report_v10.tex`:** Revision 10 of the technical report, which this README follows. `multichannel_hebbian_report_v9.pdf`, `_v8.pdf`, `_v7.pdf`, `multichannel_hebbian_report.pdf` and `_v2.pdf` are earlier revisions.
- **`docs/audit_2026-10.md`:** the October 2026 audit of the project's records (code, beliefs, literature, process), whose Section 3.1 brought the window gate back.
- **`docs/reading/`:** notes on twelve papers on context routing and memory, read in full, with the synthesis `distant_cues_2026-10.md` and the prompts for the parallel exploratory sessions.
- **`docs/review_2026-10/`:** the referee's report on Revision 8 (`review.md`, `recompute.py`/`recompute.md`, `preregistration.md`, `claims.md`, `statistics.md`, `literature.md`, `repro.py`), its claim registry (`claims_registry.md`), its check of Revision 8.1 (`revision_8_1_check.md`), its reviews of the `test_delta_recipe` specification and test file, its review of the replication, and its passes over this revision's draft (`revision_9_check.md`, `rev9/second_pass.md` to `rev9/sixth_pass.md`, each with the scripts that recompute every number from the records).
- **`docs/revision9.md`**, **`docs/revision8.md`**, **`docs/revision7.md`**, **`docs/revision6.md`:** this README's research text as of Revisions 9, 8.1, 7 and 6, kept for their accounts of Phase VI's third test and first replication, of Phase VI's first tests, of Phase V and of Phases I–IV.
- **Branch `claude/outside-ideas`:** the exploratory screens (`explore_*.py`, outputs and per-batch logs in `explore_out/`, verdicts in `explore_out/README.md`). **Branches `claude/explore-F`, `claude/explore-G`, `claude/explore-H`, `claude/explore-I`:** the parallel exploratory sessions (delta-rule channels and forget gates; distant cues; the split's devices and constants; a dense-loss task and the GPU package `explore_gpu/`), each with its reports under `explore_out/<session>/`. **Branch `claude/replicate-R`:** the clean-room replication (its own implementation, protocol, `UNSTATED.md` and `replicate_report.md`). **Branch `claude/review-V`:** the referee's working branch, merged here under `docs/review_2026-10/`. The prompts that started each session and each round are in `docs/reading/parallel_sessions_prompts.md`. Nothing on these branches is a result.

**Running a test** (it runs the CHECKs, the projection, then the full test; runs are cached, so an interrupted test resumes). Phase VI tests take a required `--machine X|L` flag, and each machine runs its own seed block:

```bash
pip install -r requirements.txt
python3 -u test_window_gate.py --machine L --workers 6
```

**To compute the pooled claims** once both machines' files are complete (the pooled block is identical from either side):

```bash
python3 -u test_window_gate.py --machine L --report --also results/X/window_gate_results.json
```

**Tests that pair with earlier recorded runs** read *this machine's* own earlier results files: by default from the working directory, or from flags such as `--early`, `--slow`, `--recipe` or `--short`. Before pairing, they reproduce a recorded run from each file bit for bit. If a file does not reproduce, the test stops before training, or reports the paired claims as UNTESTED.

Training is deterministic for a given processor, kernel path and thread count; workers run one thread each.

**Long runs** (`longrun.py`). Long runs in cloud containers have been lost to container restarts. `longrun.py start` launches the test detached (its own session, stdin closed, output appended to `longrun/<name>.log`, gitignored) and records each start's process, boot id and commit. `longrun.py watch --resume` prints the new log lines and the results file's record count for up to a few minutes. It relaunches the same command when the run has stopped without an exit code (a container restart or a kill), and the test then resumes from its cached records. A run that exits with an error is reported, not relaunched. Tests resume the way `test_window_gate`'s Part D did: runs already in the results file are skipped, and the meta keeps the first start's commit and time with every later start listed under `starts` (`longrun.resume_meta`). Each test prints its runtime projection, of all runs and of the runs still to do, before the first run (`longrun.projection`).

```bash
python3 longrun.py start window_gate_L --results window_gate_results.json --total 231 -- \
    python3 -u test_window_gate.py --machine L --workers 6
python3 longrun.py watch window_gate_L --minutes 9 --resume    # repeat until FINISHED
```

Without `longrun.py`, `setsid nohup python3 -u test_window_gate.py --machine L --workers 6 > window_gate.log 2>&1 &` detaches a run, and rerunning the same command resumes it.

