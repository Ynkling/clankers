# Contributing

This repository is a research fork of Pathway's BDH, run as a pre-registered project: every claim is
fixed in writing before the runs that test it, every run leaves a record that an outsider can re-score,
and the text of the report is checked against those records by a referee who had no part in the work.
Contributions are welcome under the same rules. They are short, and they are the whole method.

## The branches

- **The main line** (`claude/bdh-growth-hebbian-inference-w90069`, merged into the default branch by
  pull request) holds the pre-registered tests (`test_*.py`), their specifications (`specs/`), the
  records (`results/`), the report (`multichannel_hebbian_report_v*.tex` / `.pdf`), and the
  referee's review (`docs/review_2026-10/`). Nothing on it is a result until the test's docstring
  carries the recorded outcome.
- **Exploratory branches** (`claude/outside-ideas`, `claude/explore-*`, `claude/replicate-R`) hold
  screens. They add new files only and never edit a main-line module; every output they produce is
  labelled "EXPLORATORY, not a result". A screen's finding becomes a claim only when a main-line
  test repeats it.
- **The referee's branch** (`claude/review-V`) writes only under `docs/review_2026-10/` and changes
  nothing else.

## Before a run

1. **Write the specification first.** For a main-line test, commit `specs/test_<name>.md` before the
   test file exists: the arms, the seeds per machine, the outcome fixed by a named function, the
   claims (pooled over both machines primary, per machine secondary), the readings each outcome
   maps to, and the CHECKs. The referee reviews it before any run; amendments are dated and say
   what had been seen when they were made.
2. **Fix the readings in the docstring** before the first run, for a screen as for a test. A reading
   is a sentence the code prints when a pre-fixed condition holds; it is never chosen after the
   counts are known. Name outcomes so that each reading rests on exactly one outcome.
3. **Choose fresh seeds.** Check every branch for the seeds you plan to use; the main line's ranges
   are listed in the tests' CHECKs. A collision is a reason to shift your block, not to share it.
4. **Run the CHECKs.** Every test verifies, before training, that it reproduces a recorded run bit for
   bit on the same CPU model, that each arm with its device switched off equals its base, and that
   its outcome functions agree with the recorded verdicts. A CHECK that fails stops the run.

## During and after a run

- Use `longrun.py` for anything longer than an hour; it survives container restarts and resumes from
  the cached records. Do not change a run's code while it runs. A change to the *report* rules during
  a run is an amendment that states what had been read at the time it was committed.
- Record everything the outcome needs: the curve at every evaluation, the per-stream gate masses,
  every trigger check's probe reading beside the evaluation's, every firing with its target and the
  map before it, the CPU, the code commit, the start time.
- Commit the results file under `results/<machine>/` with a row in `results/README.md` (commit and
  start date from the file's own `meta`). Pool with the other machine through `--also`.
- Write what the pre-fixed readings say, and nothing else as a result. Descriptions of what happened
  are welcome and are labelled as descriptions; interpretations are labelled as whose they are.

## The report

Every number in the report must be recomputable from a committed record by a script. Every
statement about a branch is checked against that branch at its head before the commit. Statements
the referee finds false are listed in the next revision, not silently fixed. The README is generated
from the report's tex and is not edited by hand.

## Pull requests

The pull request template asks for the items above. A pull request that adds a result without its
specification, its readings' commit before its first record, or its records, is not mergeable.
