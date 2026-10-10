#!/usr/bin/env python
"""
test_muon_recipe.py — the candidate GPU recipe (MUON + WINDOW) confirmed on independent seeds per machine,
and tried at eight keys, where Muon has not been run. Everything below is fixed before any run. Both
machines run it, each on its own seeds.

Run it directly (X first; on X the recorded files are results/X/...):

    python test_muon_recipe.py --machine X --workers 4 --early results/X/early_recipe_results.json \\
        --slow results/X/slow_start_results.json --recipe results/X/stream_recipe_results.json \\
        --curriculum results/X/stream_curriculum_results.json --scale results/X/scale_axes_results.json \\
        --short results/X/short_conv_results.json
    (the inherited CHECKs 122-123 need claude/outside-ideas fetched: git fetch origin claude/outside-ideas)
    L: python test_muon_recipe.py --machine L --workers 6 --also results/X/muon_recipe_results.json
       (--early defaults to L's own early_recipe_results.json; --slow etc. to L's own files)
    The report alone, from a finished results file (no verification, no training):
       python test_muon_recipe.py --machine L --report --also results/X/muon_recipe_results.json

PURPOSE
Confirm the candidate GPU recipe (MUON + WINDOW) on independent seeds per machine, so a claim shown on
both machines is two independent confirmations and pooled tests are valid; and try it at eight keys,
where Muon has not been run.

BACKGROUND
- test_early_recipe (seeds 300-359 on both machines): E1 WIN_M 40/40 vs SLOW_M 36/40 on X (4 vs 0,
  NOT SHOWN), 40/40 vs 34/40 on L (6 vs 0, SHOWN). E2 bound by 4800 WIN16_M vs WIN16_A 10/12 vs 6/12
  on X (5 vs 1, NOT SHOWN; Part B cut to 12 seeds), 19/20 vs 12/20 on L (7 vs 0, SHOWN). Same seeds on
  both machines, so not independent (Revision 7, Section 15).
- test_slow_start: HINGE8 (Adam, S=2, P=8, conv) 20/20 on X, 17/20 on L. Muon untried at P=8.
- Muon's bf16 Newton-Schulz step is processor-specific; nothing here pairs across machines.

DEFINITIONS
ADAM, MUON, WINDOW, the outcomes, the failure classes and the diagnostics exactly as in
test_early_recipe, imported from it (make_rc, running, one_run, run_path, the report helpers). Part C
runs on test_slow_start's Part A path (test_scale_axes' DIRECT8: S=2, P=8, k=2, conv; part_run "A"),
the path of HINGE8; its perfect gate is test_scale_axes' ceiling8 on the same run_job. test_early_recipe
gained two knobs for this test, inert at their defaults (CHECK 129): one_run(path=) and
lr_job(arm=, path=), the run path and arm of another test.

SEEDS (disjoint by machine; the required flag --machine X|L is printed with the CPU string)
- X: Part A 400-439, Part B 440-459, Part C 460-479.
- L: Part A 1400-1439, Part B 1440-1459, Part C 1460-1479.

ARMS (per machine)
- Part A: S=2, P=4, k=2, no conv, 24000 steps; outcome DISCOVERED.
    SLOW_M   MUON, no hinge
    WIN_M    MUON + WINDOW
- Part B: S=4, P=4, k=16, conv, 28800 steps; outcome BOUND, with the transition.
    WIN16_A  ADAM + WINDOW
    WIN16_M  MUON + WINDOW
- Part C: S=2, P=8, k=2, conv, 28800 steps; outcome BOUND, with the transition.
    WIN8_A   ADAM + WINDOW
    WIN8_M   MUON + WINDOW
- Validity per part: the perfect gate under MUON at 0.005 (no slow phase) binds on the part's first two
  seeds (CEIL_A: test_short_conv's perfect gate on Part A's run_one call; CEIL_B: test_stream_recipe's
  ceiling4k16; CEIL_C: test_scale_axes' ceiling8). Otherwise that part's MUON claims are UNTESTED.

CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05)
Pooled (primary): printed when --also gives the other machine's complete file (its meta names the other
machine; every run its meta schedules is recorded and finished), over both machines' pairs (a pair = a
seed on which one machine ran both arms; b and c are summed over the two machines). A pooled claim is
UNTESTED if its part is not valid on either machine.
- M1: WIN_M beats SLOW_M (Part A, DISCOVERED; 80 pairs).
- M2: WIN16_M is bound by update 4800 more often than WIN16_A (Part B; 40 pairs).
- M3: WIN8_M is bound by update 4800 more often than WIN8_A (Part C; 40 pairs, fewer if the drop rule
  cut WIN8_A on a machine).
Bands, per machine and pooled (RELIABLE >= 90%, MAJORITY >= 50%, MINORITY >= 1 run, NEVER 0; of the
scheduled runs; test_stream_curriculum.band_n), with the Wilson 95% interval (test_stream_recipe.wilson):
WIN_M, WIN16_M, WIN8_M. A MUON arm's band is UNTESTED where its part is not valid.
Reading, pooled: "the Muon recipe is reliable at the working configurations" if all three are RELIABLE
pooled and each pooled Wilson 95% lower bound is >= 0.80; otherwise each configuration that falls short
is named (its band and Wilson lower bound, or UNTESTED).
Per machine (secondary): M1-M3 on that machine's pairs, same tests; UNTESTED where that machine's part
is not valid.
Printed, not claims: BOUND comparisons in Parts B and C (WIN16_M vs WIN16_A, WIN8_M vs WIN8_A; per
machine and pooled); median transitions.

DIAGNOSTICS (labelled, not part of the verdict), as in test_early_recipe: every firing's update and the
hinge/task gradient-norm ratio at each firing; would-fires after the window; failure classes (k=2:
test_router_layout.fail_class through test_slow_start.tag; k=16: test_stream_recipe.outcome); merged runs,
and the distinct channels holding the streams at 4800, 9600 and the end (Part B); the hosts (CPU label,
oneDNN bf16 kernel, Newton-Schulz fingerprint).

CHECKS (after the inherited chain, which ends with test_early_recipe's CHECK 128)
- 129 test_early_recipe's new knobs are inert at their defaults: its runs (WIN_M at TAU = 0 and WIN16_M,
      seed 3, 1200 steps; records, firings and diagnostic rows) and lr_job("WIN16_M") equal 102ca96's module.
- 130 Code identity: this machine's own recorded test_early_recipe runs WIN_M|300 and WIN16_M|340
      reproduce bit for bit through 1200 (curve, statistics, gradient norms; the firings and their
      diagnostic rows through 1200), from --early (X: results/X/early_recipe_results.json; L: its own
      early_recipe_results.json). The test stops before verification if that file's CPU label is not
      this machine's.
- 131 WIN8_A seed 220 equals this machine's recorded test_slow_start HINGE8 seed 220 (--slow) bit for bit
      through 2400 (curve, statistics, gradient norms); every firing has a diagnostic row.
- 132 Seeds: this machine's seeds are its ranges (the perfect gates on each part's first two seeds) and
      disjoint from every earlier main-line seed (test_early_recipe's 300-359 included); with --also,
      the other file's meta names the other machine, its seeds lie in the other machine's ranges, its
      CPU label differs from this machine's and no seed overlaps.
- 133 The optimizer groups cover every parameter exactly once and every group's lr is correct at updates
      1, 2400 and 2401 (every arm, through this test's run path; test_early_recipe's lr_job); the
      hinge's weight is 1 at 2400 and 0 at 2401 for WIN8_A and WIN8_M (real 2402-step runs at TAU = 0:
      2400 firings with a diagnostic row each, would-fires at 2401 and 2402; Parts A and B: CHECK 124).
- 134 A worker's run is bit-identical to the same run here (WIN8_M seed 1, 1200 steps; WIN8_A at TAU = 0
      seed 1, 1200 steps, a diagnostic row per update).

RUNTIME
- The projection uses pool-load timing (test_early_recipe's): each arm's real path for 1200 steps, as
  simultaneous copies on the workers; the median per-step time; worst case = every run to its last step.
- If over 10 h, WIN8_A is cut to 10 seeds (Part C's first ten: X 460-469, L 1460-1469). Parts A and B
  and WIN8_M are never cut. The decision is stored at the first start of the results file and kept on a
  resume.
- Runs are scheduled perfect gates first, then longest first, workers x 1 thread; finished runs are cached.

OUTPUT
Per-seed raw values first, then validity, counts, the claims (pooled with --also, then per machine),
readings, diagnostics and curves; with --also, X and L side by side.
Results: muon_recipe_results.json (gitignored; X's copy in results/X/).

RESULT (X, test commit 74f5907; results/X/muon_recipe_results.json)
Run conditions:
- 156/156 records, none failed; 325.6 min of training after 3.2 h of verification, 4 workers x 1
  thread, one start (2.80GHz host, started 2026-10-06 06:19, detached with setsid).
- Every CHECK passed (inherited 1-128, own 129-134). Every record: 2.80GHz, gemm:jit:bf16, NS
  fingerprint f209ef7b8c61.
- Drop rule: projection 17.96 h > 10 h, so WIN8_A was cut to seeds 460-469 (projection 16.45 h; the
  rule has no further step). Parts A and B and WIN8_M ran every seed.
- Validity: CEIL_A 2/2 (1200), CEIL_B 2/2 (1200), CEIL_C 2/2 (2400) -> all three parts VALID.
Claims:
- Pooled (primary): not computed on X alone; it needs L's complete file (--also).
- Per machine (secondary):
  - M1 WIN_M 39/40 vs SLOW_M 35/40 DISCOVERED, 4 vs 0, p = 0.0625: NOT SHOWN.
  - M2 bound by 4800: WIN16_M 15/20 vs WIN16_A 9/20, 7 vs 1, p = 0.035: SHOWN.
  - M3 bound by 4800: WIN8_M 9/10 vs WIN8_A 9/10 (the 10 seeds both ran), 1 vs 1, p = 0.75: NOT SHOWN.
- Bands: WIN_M RELIABLE (39/40, Wilson 95% [0.871, 0.996]); WIN16_M RELIABLE (19/20, [0.764, 0.991]);
  WIN8_M RELIABLE (20/20, [0.839, 1.000]).
- Printed, not claims: BOUND WIN16_M 19/20 vs WIN16_A 17/20 (3 vs 1, p = 0.31); WIN8_M 10/10 vs WIN8_A
  9/10 on 460-469 (1 vs 0, p = 0.5). Median transition SLOW_M 2400, WIN_M 2400, WIN16_A 4800 [3600,
  27600], WIN16_M 2400 [2400, 19200], WIN8_A 3600 [3600, 4800], WIN8_M 4800 [3600, 6000].
Diagnostics (not part of the verdict):
- SLOW_M's 5 failures: 3 POSITION (s407, s418, s434) and 2 KEY (s400, s421). WIN_M discovered on all
  but s421 (KEY under both; its one firing at update 551).
- The hinge fired on 20/40 WIN_M runs (first firing 150 [50, 551]), 13/20 WIN16_A (385 [183, 726]),
  18/20 WIN16_M (293 [185, 565]), 7/10 WIN8_A (362 [218, 669]) and 7/20 WIN8_M (198 [141, 400]).
  Hinge/task ratio over every firing: WIN_M 1127.6 [0.1, 132443.4], WIN16_A 21.6 [1.4, 631101.0],
  WIN16_M 110.0 [6.6, 24919.6], WIN8_A 20.5 [2.9, 269637.5], WIN8_M 3440.6 [1.9, 63263.6].
  Would-fires after the window: WIN_M 5 runs (6 updates), WIN16_A 7 runs (51454 updates), WIN8_A 2
  runs (15982), WIN16_M and WIN8_M none.
- Part B: merged at the end WIN16_A 5 (s440, s447, s448, s456, s458), WIN16_M 3 (s445, s453, s459;
  s445 and s453 bound, not routed). At 4800, 17/20 WIN16_M gates held the 4 streams on 4 distinct
  channels; WIN16_A 9/20.
- Part C: WIN8_M bound on every seed, 17 by 4800 (s468, s478, s479 at 6000); WIN8_A's one failure,
  s461, is POSITION (WIN8_M bound it at 3600 after firings at 141 and 163).

RESULT (L, test commit 74f5907; results/L/muon_recipe_results.json; recomputed from that file with
this file's report(), --report --also results/X/muon_recipe_results.json)
Run conditions:
- 166/166 records, none failed; 12th Gen Intel(R) Core(TM) i7-12650H, torch 2.14.0, 6 workers, started
  2026-10-05 23:08. Projection 7.21 h: nothing cut (WIN8_A ran 1460-1479).
- The meta is written only after verify() passes, so L's CHECKs passed (L's log is not recorded here).
  Every record: NS fingerprint 91a18153b4d5 (X: f209ef7b8c61); oneDNN bf16 kernel not identified ("?").
- CHECK 132's --also part holds both ways: each file names the other machine, its seeds lie in the
  other machine's ranges (X 400-479, L 1400-1479), the CPU labels differ and no seed overlaps.
- Validity: CEIL_A 2/2, CEIL_B 2/2, CEIL_C 2/2 -> all three parts VALID.
Per machine (secondary):
- M1 WIN_M 40/40 vs SLOW_M 34/40 DISCOVERED, 6 vs 0, p = 0.0156: SHOWN.
- M2 bound by 4800: WIN16_M 9/20 vs WIN16_A 11/20, 5 vs 7, p = 0.81: NOT SHOWN.
- M3 bound by 4800: WIN8_M 13/20 vs WIN8_A 11/20, 6 vs 4, p = 0.38: NOT SHOWN.
- Bands: WIN_M RELIABLE (40/40, Wilson [0.912, 1.000]); WIN16_M MAJORITY (14/20, [0.481, 0.855]);
  WIN8_M RELIABLE (19/20, [0.764, 0.991]).
- Printed, not claims: BOUND WIN16_M 14/20 vs WIN16_A 16/20 (3 vs 5); WIN8_M 19/20 vs WIN8_A 18/20
  (2 vs 1). Median transition SLOW_M 2400, WIN_M 2400 [1200, 14400], WIN16_A 4800, WIN16_M 3600
  [2400, 21600], WIN8_A 4800 [3600, 26400], WIN8_M 4800 [3600, 8400].
Diagnostics (not part of the verdict):
- SLOW_M's 6 failures are all POSITION (s1409, s1418, s1419, s1424, s1430, s1438); WIN_M discovered
  on every seed (s1413 only at 14400).
- WIN16_M's 6 failures: 5 MERGED (2 share) and 1 STREAM-PARTIAL; 7 merged at the end (2 of them bound,
  not routed). WIN16_A's 4: 3 non-stream POSITION, 1 MERGED. At 4800, WIN16_M held the 4 streams on 4
  distinct channels in 10/20 runs, WIN16_A in 11/20.
- WIN8_M's one failure, s1461, OTHER (no firing; 1331 would-fires after the window); WIN8_A's two:
  s1468 KEY, s1476 OTHER.
- The hinge fired on 23/40 WIN_M runs (first 136 [54, 278]), 16/20 WIN16_A (380 [220, 1036]), 19/20
  WIN16_M (225 [129, 435]), 15/20 WIN8_A (281 [33, 654]) and 6/20 WIN8_M (174 [121, 405]).

POOLED (primary; X + L, b and c summed over both machines' pairs; computed identically from either
machine's side)
- M1 WIN_M 79/80 vs SLOW_M 69/80 DISCOVERED, 10 vs 0 (X 4 vs 0, L 6 vs 0), p = 0.00098: SHOWN.
- M2 bound by 4800: WIN16_M 24/40 vs WIN16_A 20/40, 12 vs 8 (X 7 vs 1, L 5 vs 7), p = 0.25: NOT SHOWN.
- M3 bound by 4800: WIN8_M 22/30 vs WIN8_A 20/30 (X's 10 pairs + L's 20), 7 vs 5 (X 1 vs 1, L 6 vs 4),
  p = 0.39: NOT SHOWN.
- Bands: WIN_M RELIABLE (79/80, Wilson [0.933, 0.998]); WIN16_M MAJORITY (33/40, [0.681, 0.913]);
  WIN8_M RELIABLE (39/40, [0.871, 0.996]).
- Reading: not "the Muon recipe is reliable at the working configurations"; it falls short at WIN16_M
  (S=4, P=4, k=16, conv): MAJORITY 33/40, Wilson lower bound 0.681.
- Printed, not claims: BOUND WIN16_M 33/40 vs WIN16_A 33/40 (6 vs 6); WIN8_M 29/30 vs WIN8_A 27/30 on
  the paired seeds (3 vs 1).
- X and L disagree at four streams: WIN16_M bound 19/20 on X and 14/20 on L, and M2 is 7 vs 1 on X but
  5 vs 7 on L. These are independent seeds, so neither machine's count reproduces the other's.
"""

import argparse
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

from test_binding_onset import EVAL_EVERY, fmt_step
from test_router_curriculum import get, load_store, init_worker, makespan, cpu_model, save_results
from test_router_discovery import load_at
from test_router_reliability import strip_all
from test_conv_lr import bound_r, mcnemar_greater
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_stream_channels import med_int, merged, shared_max
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import (LR, LR_WARM, WARM, TAU, GATE, ok_r, hstr, tag, runs_of, curve_str, raw_rows, success,
                             subset_equal, trunc, outcome_key)
import test_early_recipe as ter
from test_early_recipe import MUON_LR, SLOW_SCALE, BY, firings, fire_str, paired_by, by_update, host_info
from test_recipe_scope import cmap_at, groups_str

# ── Settings (fixed before any run) ──────────────────────────────────────────
MACHINES = ("X", "L")
RANGES = {"X": dict(A=range(400, 440), B=range(440, 460), C=range(460, 480)),
          "L": dict(A=range(1400, 1440), B=range(1440, 1460), C=range(1460, 1480))}
MAX_C = tsa.MAX_ITERS                      # 28800
VALID_N = 2
DROP_H = 10.0
CUT_N = 10                                 # the drop rule: WIN8_A on Part C's first ten seeds
RELIABLE_LO = 0.80                         # the reading: pooled Wilson 95% lower bound
LEGACY_SHA = "102ca96"                     # head before this test's changes (test_early_recipe's knobs)
EARLY_REPRO = (("WIN_M", 300), ("WIN16_M", 340))
HINGE8_SEED = 220
TIME_ITERS = ter.TIME_ITERS
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "muon_recipe_results.json"

REAL = dict(ter.REAL, iters_c=MAX_C)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ARMS = [dict(ter.ARM[k]) for k in ("SLOW_M", "WIN_M", "WIN16_A", "WIN16_M")] + [
    dict(key="WIN8_A", part="C", opt="adam", slow=True, tau=TAU, ceil=False, label="WIN8_A   ADAM + WINDOW (S=2, P=8, k=2, conv)"),
    dict(key="WIN8_M", part="C", opt="muon", slow=True, tau=TAU, ceil=False, label="WIN8_M   MUON + WINDOW (S=2, P=8, k=2, conv)"),
] + [dict(ter.ARM[k]) for k in ("CEIL_A", "CEIL_B")] + [
    dict(key="CEIL_C", part="C", opt="muon", slow=False, tau=None, ceil=True,
         label="CEIL_C   perfect gate ceiling8 under MUON (validity, Part C)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
CEILS = (("A", "CEIL_A"), ("B", "CEIL_B"), ("C", "CEIL_C"))
TPART = {"A": "0", "B": "B", "C": "A"}     # test_slow_start's part names (tag, raw_rows, outcome)
CONFIG = {"WIN_M": "S=2, P=4, k=2, no conv", "WIN16_M": "S=4, P=4, k=16, conv", "WIN8_M": "S=2, P=8, k=2, conv"}
BAND_KEYS = ("WIN_M", "WIN16_M", "WIN8_M")
READING = "the Muon recipe is reliable at the working configurations"


def discovered(r):
    return success("0", r)


def by_4800(r):
    return by_update(r, BY)


CLAIMS = (("M1", "WIN_M", "SLOW_M", "A", discovered, "WIN_M beats SLOW_M (DISCOVERED)"),
          ("M2", "WIN16_M", "WIN16_A", "B", by_4800, f"WIN16_M bound by update {BY} more often than WIN16_A"),
          ("M3", "WIN8_M", "WIN8_A", "C", by_4800, f"WIN8_M bound by update {BY} more often than WIN8_A"))
PRINTED = (("WIN16_M", "WIN16_A", "B"), ("WIN8_M", "WIN8_A", "C"))


def seeds_for(machine):
    r = RANGES[machine]
    A, B, C = (tuple(r[p]) for p in "ABC")
    return dict(SLOW_M=A, WIN_M=A, WIN16_A=B, WIN16_M=B, WIN8_A=C, WIN8_M=C, CEIL_A=A[:2], CEIL_B=B[:2], CEIL_C=C[:2])


SEEDS = seeds_for("X")                     # main() sets the machine's


# ── Runs ─────────────────────────────────────────────────────────────────────
def run_path(a, seed, iters, rc):
    """The arm's run path: Parts A and B test_early_recipe's; Part C test_slow_start's Part A path
    (test_scale_axes' DIRECT8, the path of HINGE8), its perfect gate test_scale_axes' ceiling8 on the same
    run_job."""
    if a["part"] != "C":
        return ter.run_path(a, seed, iters, rc)
    if a["ceil"]:
        return tsa.run_job(dict(arm="ceiling8", seed=seed, iters=iters), recipe=rc)
    return tss.part_run("A", seed, iters, rc)


def iters_of(a, sched):
    return sched[{"A": "iters_a", "B": "iters_b", "C": "iters_c"}[a["part"]]]


def one_run(a, seed, iters, sched, **kw):
    return ter.one_run(a, seed, iters, sched, path=run_path, **kw)


def run_job(sp):
    a = ARM[sp["arm"]]
    return one_run(a, sp["seed"], iters_of(a, sp["sched"]), sp["sched"])[0]


def spec(key, seed, sched):
    return dict(arm=key, seed=seed, sched=dict(sched))


def check_job(key, seed, iters, tau=None):
    """A CHECK run of arm `key` (record, hinge state summary)."""
    rec, state, _ = one_run(ARM[key], seed, iters, REAL, tau=tau)
    return dict(rec=rec, fired=state.get("fired"), n=state.get("n"), fired_at=list(state.get("fired_at", [])),
                would_at=list(state.get("would_at", [])), w_log=dict(state.get("w_log", {})),
                diag=list(state.get("firing_diag", [])))


def legacy_job(which, what):
    """CHECK 129: through 102ca96's test_early_recipe (legacy) or this one at the knobs' defaults (new)."""
    mod = load_at(LEGACY_SHA, "test_early_recipe.py", "test_early_recipe_legacy") if which == "legacy" else ter
    if what == "lr":
        return mod.lr_job("WIN16_M")
    key, tau = {"A": ("WIN_M", 0.0), "B": ("WIN16_M", None)}[what]
    rec, state, _ = mod.one_run(mod.ARM[key], 3, EVAL_EVERY, mod.REAL, tau=tau)
    return dict(rec=strip_all(rec), fired_at=list(state.get("fired_at", [])), diag=list(state.get("firing_diag", [])))


def lr_job(key):
    """CHECK 133: test_early_recipe's lr_job through this test's arm and run path."""
    return ter.lr_job(key, arm=ARM[key], path=run_path)


def time_job(key):
    """Pool-load timing: the arm's real path for TIME_ITERS steps (one evaluation included); seconds per step."""
    t0 = time.time()
    one_run(ARM[key], 0, TIME_ITERS, REAL)
    return (time.time() - t0) / TIME_ITERS


def projection(cost, sched, workers):
    full = {k: cost[k] * iters_of(ARM[k], sched) for k in KEYS}
    d = [full[k] for k in KEYS for _ in SEEDS[k]]
    return full, d, makespan(d, workers) / 3600


# ── Verification ─────────────────────────────────────────────────────────────
def earlier_seeds():
    """test_early_recipe's earlier seeds and its own 300-359."""
    return ter.earlier_seeds() | set(range(300, 360))


def complete(store):
    """(complete, missing): every run the file's meta schedules is recorded and finished."""
    seeds = store.get("meta", {}).get("seeds") or {}
    missing = [f"{k}|{s}" for k in KEYS for s in seeds.get(k, []) if not ok_r(get(store, k, s))]
    return bool(seeds) and set(seeds) == set(KEYS) and not missing, missing


def other_of(machine):
    return "L" if machine == "X" else "X"


def check_also(machine, also):
    """CHECK 132's part for --also: (ok, lines)."""
    m = also.get("meta", {})
    oth = other_of(machine)
    seeds = m.get("seeds") or {}
    in_range = all(s in RANGES[oth][ARM[k]["part"]] for k in KEYS for s in seeds.get(k, []))
    mine = {s for k in KEYS for s in SEEDS[k]}
    theirs = {s for k in KEYS for s in seeds.get(k, [])}
    good = m.get("machine") == oth and bool(seeds) and in_range and m.get("cpu") != cpu_model() and not (mine & theirs)
    lines = [f"--also: machine {m.get('machine')} (should be {oth}); CPU {m.get('cpu')} (this machine {cpu_model()}; "
             f"differs: {m.get('cpu') != cpu_model()}); its seeds {tsr.ranges(sorted(theirs)) if theirs else None} in "
             f"{oth}'s ranges: {in_range}; overlap with this machine's: {sorted(mine & theirs) or 'none'}"]
    return good, lines


def verify(pool, files, machine, early, also, early_assert=True):
    f = {}
    for part in ("A", "B"):
        for which in ("legacy", "new"):
            f["129", part, which] = pool.submit(legacy_job, which, part)
    for which in ("legacy", "new"):
        f["129", "lr", which] = pool.submit(legacy_job, which, "lr")
    for k, s in EARLY_REPRO:
        f["130", k] = pool.submit(check_job, k, s, EVAL_EVERY)
    f["131"] = pool.submit(check_job, "WIN8_A", HINGE8_SEED, WARM)
    for k in KEYS:
        f["133", k] = pool.submit(lr_job, k)
    for k in ("WIN8_A", "WIN8_M"):
        f["133w", k] = pool.submit(check_job, k, 3, WARM + 2, 0.0)
    f["134", "M"] = pool.submit(check_job, "WIN8_M", 1, EVAL_EVERY)
    f["134", "A"] = pool.submit(check_job, "WIN8_A", 1, EVAL_EVERY, 0.0)

    print("test_early_recipe.py's verification (which runs test_recipe_scope's, and so on down to")
    print("test_multilayer_binding's):")
    info = ter.verify(pool, files)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 129 test_early_recipe's new knobs (one_run path, lr_job arm and path) are inert at their defaults: its "
          f"runs (seed 3, {EVAL_EVERY} steps) and lr_job vs {LEGACY_SHA}'s module:")
    good = True
    for part, what in (("A", "WIN_M at TAU = 0"), ("B", "WIN16_M")):
        L_, n_ = f["129", part, "legacy"].result(), f["129", part, "new"].result()
        same = (subset_equal(L_["rec"], n_["rec"]) and subset_equal(n_["rec"], L_["rec"]) and L_["fired_at"] == n_["fired_at"]
                and L_["diag"] == n_["diag"])
        g = same and n_["rec"]["ok"]
        good &= g
        print(f"     {what:<17}: records, firings and diagnostic rows equal {same}   firings {len(n_['fired_at'])}   curve "
              f"{n_['rec']['curve']}  -> {'UNCHANGED' if g else 'DIFFERS'}")
    L_, n_ = f["129", "lr", "legacy"].result(), f["129", "lr", "new"].result()
    g = L_ == n_ and n_["ok"]
    good &= g
    print(f"     lr_job WIN16_M   : equal {L_ == n_}   lrs {n_['at']}  -> {'UNCHANGED' if g else 'DIFFERS'}")
    ok &= good
    print()

    em = early.get("meta", {})
    print(f"CHECK 130 code identity: this machine's recorded test_early_recipe runs ({files['early_path']}, written on "
          f"{em.get('cpu')}, git {em.get('git')}; this machine {cpu_model()}) reproduce bit for bit through {EVAL_EVERY}:")
    good = True
    for k, s in EARLY_REPRO:
        r = f["130", k].result()
        old = get(early, k, s)
        same = old is not None and subset_equal(trunc(r["rec"], EVAL_EVERY), trunc(old, EVAL_EVERY))
        end = (old or {}).get("end") or {}
        fa_old = [u for u in end.get("hs_fired_at") or [] if u <= EVAL_EVERY]
        dg_old = [d for d in end.get("hs_firing_diag") or [] if d["update"] <= EVAL_EVERY]
        g = same and r["fired_at"] == fa_old and tss.norm(r["diag"]) == dg_old and r["rec"]["ok"]
        good &= g
        print(f"     {k}|{s}: curve {r['rec']['curve']} vs recorded {[c for c in old['curve'] if c[0] <= EVAL_EVERY] if old else None}; "
              f"statistics and gradient norms equal {same}; firings {r['fired_at']} vs {fa_old}; diagnostic rows equal "
              f"{tss.norm(r['diag']) == dg_old}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    if early_assert:
        ok &= good
    else:
        print(f"     (the file was written on {em.get('cpu')}, this machine is {cpu_model()}: printed, not asserted)")
    print()

    print(f"CHECK 131 WIN8_A seed {HINGE8_SEED} vs the recorded test_slow_start HINGE8 seed {HINGE8_SEED} ({files['slow_path']}) "
          f"through {WARM}:")
    r = f["131"].result()
    old = get(files["slow"], "HINGE8", HINGE8_SEED)
    rep = old is not None and subset_equal(trunc(r["rec"], WARM), trunc(old, WARM))
    g = rep and r["rec"]["ok"] and len(r["diag"]) == len(r["fired_at"])
    m_ = files["slow"].get("meta", {})
    print(f"     curve {r['rec']['curve']} vs recorded {[c for c in old['curve'] if c[0] <= WARM] if old else None}; statistics and "
          f"gradient norms equal: {rep}")
    print(f"     firings {r['fired_at']} (each with a diagnostic row: {len(r['diag']) == len(r['fired_at'])}; hinge/task ratio "
          f"{[round(d['ratio'], 1) for d in r['diag']]})   the file was written on {m_.get('cpu')} (git {m_.get('git')}), "
          f"this machine {cpu_model()}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()

    print(f"CHECK 132 the seeds (machine {machine}):")
    rg = RANGES[machine]
    parts_ok = all(set(SEEDS[k]) <= set(rg[ARM[k]["part"]]) for k in KEYS)
    full_ok = all(SEEDS[k] == tuple(rg[ARM[k]["part"]]) for k in KEYS if not ARM[k]["ceil"] and k != "WIN8_A")
    ceil_ok = all(SEEDS[k] == tuple(rg[p])[:VALID_N] for p, k in CEILS)
    cut_ok = SEEDS["WIN8_A"] in (tuple(rg["C"]), tuple(rg["C"])[:CUT_N])
    mine = {s for k in KEYS for s in SEEDS[k]}
    other = {s for p in "ABC" for s in RANGES[other_of(machine)][p]}
    earlier = earlier_seeds()
    g = parts_ok and full_ok and ceil_ok and cut_ok and not (mine & earlier) and not (mine & other)
    print(f"     this machine's seeds {tsr.ranges(sorted(mine))}: in its ranges {parts_ok}; Parts A and B and WIN8_M full "
          f"{full_ok}; WIN8_A full or its first {CUT_N} {cut_ok}; the perfect gates on each part's first two seeds {ceil_ok}")
    print(f"     disjoint from the other machine's ranges {not (mine & other)}; from every earlier main-line seed "
          f"{not (mine & earlier)} (earlier: {tsr.ranges(sorted(earlier))})")
    if also is not None:
        ga, lines = check_also(machine, also)
        for ln in lines:
            print(f"     {ln}  -> {'OK' if ga else 'WRONG'}")
        g &= ga
    else:
        print("     no --also file")
    ok &= g
    print()

    print(f"CHECK 133 the optimizer groups, the lrs at updates 1, {WARM} and {WARM + 1}, and the hinge's weight:")
    good = True
    for k in KEYS:
        a = ARM[k]
        r = f["133", k].result()
        if a["opt"] == "muon":
            sc = SLOW_SCALE if a["slow"] else 1.0
            kinds = [(g_["tag"], g_["kind"]) for g_ in r["groups"]]
            w1 = [MUON_LR] * (1 if ("gate", "muon") in kinds else 0) + [sc * MUON_LR, sc * LR]
            w2 = [MUON_LR] * (1 if ("gate", "muon") in kinds else 0) + [MUON_LR, LR]
            adam_names = [n for g_ in r["groups"] if g_["kind"] == "adam" for n in g_["names"]]
            g = (r["cover"] and r["at"]["1"] == w1 and r["at"][str(WARM)] == w1 and r["at"][str(WARM + 1)] == w2
                 and "embed.weight" in adam_names and (a["part"] == "A" or "short_conv.conv_w" in adam_names)
                 and (a["ceil"] or ("gate", "muon") in kinds) and r["n"] == WARM + 1 and r["ok"])
        else:
            g = (r["cover"] and r["at"]["1"] == [LR, LR_WARM] and r["at"][str(WARM)] == [LR, LR_WARM]
                 and r["at"][str(WARM + 1)] == [LR, LR] and r["groups"][0]["names"] == list(GATE) and r["n"] == WARM + 1
                 and r["ok"])
        good &= g
        print(f"     {k:<8} groups {[(g_['tag'], g_['kind'], g_['names']) for g_ in r['groups']]}: every parameter exactly once "
              f"{r['cover']}; lrs at update 1 {r['at'].get('1')}, {WARM} {r['at'].get(str(WARM))}, {WARM + 1} "
              f"{r['at'].get(str(WARM + 1))}  -> {'OK' if g else 'WRONG'}")
    for k in ("WIN8_A", "WIN8_M"):
        r = f["133w", k].result()
        want = {"1": 1.0, str(WARM - 1): 1.0, str(WARM): 1.0, str(WARM + 1): 0.0, str(WARM + 2): 0.0}
        rows = r["diag"]
        g = (r["w_log"] == want and r["fired"] == WARM and r["fired_at"] == list(range(1, WARM + 1))
             and r["would_at"] == [WARM + 1, WARM + 2] and [d["update"] for d in rows] == list(range(1, WARM + 1))
             and all(math.isfinite(d["ratio"]) and d["ratio"] > 0 for d in rows) and r["rec"]["ok"])
        good &= g
        print(f"     {k:<8} TAU = 0, {WARM + 2} updates: the hinge's weight {r['w_log']}; fired {r['fired']} of {r['n']} "
              f"(updates 1-{WARM}: {r['fired_at'] == list(range(1, WARM + 1))}); would-fires {r['would_at']}; diagnostic rows "
              f"{len(rows)}, one per firing, ratios finite (median {statistics.median(d['ratio'] for d in rows):.1f})  -> "
              f"{'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 134 a worker's run is bit-identical to the same run here:")
    good = True
    for which, key, tau, what in (("M", "WIN8_M", None, "WIN8_M seed 1"), ("A", "WIN8_A", 0.0, "WIN8_A at TAU = 0 seed 1")):
        rw = f["134", which].result()
        rh = check_job(key, 1, EVAL_EVERY, tau)
        same = (subset_equal(strip_all(rw["rec"]), strip_all(rh["rec"])) and subset_equal(strip_all(rh["rec"]), strip_all(rw["rec"]))
                and rw["diag"] == rh["diag"] and rw["fired_at"] == rh["fired_at"])
        g = same and rh["rec"]["ok"] and (tau != 0.0 or len(rh["diag"]) == EVAL_EVERY)
        good &= g
        print(f"         {what}, {EVAL_EVERY} steps: records, firings and diagnostic rows equal {same}   firings "
              f"{len(rh['fired_at'])}   acc {rh['rec']['curve'][-1][1] if rh['rec']['curve'] else None}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"     this machine ({machine}): {cpu_model()}; bf16 path (Muon): {host_info()}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(host=host_info(), inherited=info)


# ── Reporting ────────────────────────────────────────────────────────────────
def validity(store, seeds):
    out = {}
    for part, k in CEILS:
        n = sum(bound_r(get(store, k, s)) for s in seeds[k])
        out[part] = (n, n >= VALID_N)
    return out


def tally(store, seeds):
    """One machine's runs, validity, claims (paired counts), bands and printed comparisons."""
    rA = {k: runs_of(store, k, seeds[k]) for k in KEYS}
    valid = validity(store, seeds)
    out = dict(rA=rA, valid=valid, claims={}, bands={}, printed={}, seeds=seeds)
    for name, new, old, part, fn, _ in CLAIMS:
        out["claims"][name] = dict(paired_by(rA[new], rA[old], seeds[new], fn), valid=valid[part][1])
    for k in BAND_KEYS:
        part = TPART[ARM[k]["part"]]
        out["bands"][k] = dict(c=sum(success(part, r) for r in rA[k].values()), n=len(seeds[k]), valid=valid[ARM[k]["part"]][1])
    for new, old, part in PRINTED:
        out["printed"][new] = paired_by(rA[new], rA[old], seeds[new], bound_r)
    return out


def pool_claim(ds):
    b, c = sum(d["b"] for d in ds), sum(d["c"] for d in ds)
    return dict(n=sum(d["n"] for d in ds), new=sum(d["new"] for d in ds), old=sum(d["old"] for d in ds), b=b, c=c,
                p=mcnemar_greater(b, c), valid=all(d.get("valid", True) for d in ds))


def verdict(d):
    return "UNTESTED" if not d["valid"] or d["n"] == 0 else ("SHOWN" if d["p"] < 0.05 else "NOT SHOWN")


def band_str(b):
    lo, hi = tsr.wilson(b["c"], b["n"])
    band = "UNTESTED" if not b["valid"] else tscur.band_n(b["c"], b["n"])
    return band, lo, (f"{b['c']}/{b['n']} -> {band}   Wilson 95% [{lo:.3f}, {hi:.3f}]  (RELIABLE >= {math.ceil(0.9 * b['n'])}, "
                      f"MAJORITY >= {math.ceil(0.5 * b['n'])})")


def print_claims(t, who):
    for name, new, old, part, _, what in CLAIMS:
        d = t["claims"][name]
        print(f"  {name}  {what}: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, {old} only {d['c']}; "
              f"p = {d['p']:.4g}" + ("" if d["valid"] else f"   (Part {part} not valid{who})"))
        print(f"     *** {name}: {verdict(d)} ***")
    for k in BAND_KEYS:
        print(f"  band {k:<8} ({CONFIG[k]}): {band_str(t['bands'][k])[2]}")


def report(store, seeds, machine, drop, wall, path, also):
    t = tally(store, seeds)
    rA, valid = t["rA"], t["valid"]
    print()
    print("#" * 100)
    print(f"PER-SEED RAW RESULTS — every value, before any aggregate (machine {machine}, {store.get('meta', {}).get('cpu')})")
    print("#" * 100)
    print("  (columns as test_slow_start's: acc = final held-out accuracy; transition; ROUTED* = margin >= 0.9 and "
          "eta_key_by_stream > 0.9; eta i/h/k/s = pooled eta^2 of the read gate at key positions on the probe batch by "
          "index / half / key / stream; hinge = training batches with a hinge term > 0 in updates 1-1200 / 1-2400 / the "
          "whole run; flag = eta^2 by index, half or key >= 0.5 at 1200; map = stream -> channel argmax; ROUTED = one-to-one "
          "and every stream >= 0.9. Firings: each firing's update with the hinge/task gradient-norm ratio on the gate)")
    if drop:
        print(f"  cut by the drop rule: {drop}")
    print()
    for k in KEYS:
        a = ARM[k]
        part = TPART[a["part"]]
        print(f"  {a['label']}   (outcome {'perfect gate: BOUND' if a['ceil'] else outcome_key(part)})")
        raw_rows(k, part, rA[k], seeds[k])
        if a["tau"] is not None:
            print(f"  {k} firings (update(ratio)):")
            for s in seeds[k]:
                if ok_r(rA[k].get(s)):
                    print(f"    s{s}  {fire_str(rA[k][s])}")
        print()

    print("=" * 100)
    print("VALIDITY (the perfect gate under MUON at 0.005, no slow phase, binds on the part's first two seeds)")
    print("=" * 100)
    for part, k in CEILS:
        print(f"  Part {part}: {k} bound {valid[part][0]}/{len(seeds[k])} "
              f"({', '.join(f's{s} ' + (('BOUND at ' + str(r['transition'])) if bound_r(r) else 'not bound') for s, r in sorted(rA[k].items()))})"
              f"  -> {'VALID' if valid[part][1] else 'NOT VALID: the MUON claims of this part are UNTESTED'}")
    print()

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in KEYS:
        if ARM[k]["ceil"]:
            continue
        part = TPART[ARM[k]["part"]]
        done = [r for r in rA[k].values() if ok_r(r)]
        trs_ = [r["transition"] for r in done if r["transition"] is not None]
        hs = [len(firings(r)[0]) for r in done if firings(r)[0] is not None]
        print(f"  {ARM[k]['label']:<50} {outcome_key(part)} {sum(success(part, r) for r in done):>2}/{len(seeds[k])}   bound "
              f"{len(trs_):>2}   bound by {BY} {sum(by_update(r, BY) for r in done):>2}   collapsed "
              f"{sum(bool(r['collapsed']) for r in done):>2}   completed {len(done)}/{len(seeds[k])}   transition "
              f"{med_int(trs_) if trs_ else '--'}" + (f"   hinge never fired {sum(1 for x in hs if x == 0)}/{len(hs)}" if hs else ""))
    print()

    other = None
    if also:
        other = load_store(also)
        okc, missing = complete(other)
        om = other.get("meta", {})
        if om.get("machine") != other_of(machine) or not okc:
            print(f"  --also {also}: machine {om.get('machine')} (should be {other_of(machine)}), complete {okc} "
                  f"({len(missing)} runs missing or failed: {missing[:6]}{' ...' if len(missing) > 6 else ''}): NOT POOLED")
            other = None
    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05)")
    print("#" * 100)
    to = None
    if other is not None:
        om = other.get("meta", {})
        to = tally(other, {k: tuple(v) for k, v in om["seeds"].items()})
        both = {machine: t, other_of(machine): to}
        print(f"POOLED (PRIMARY): machine {machine} ({store.get('meta', {}).get('cpu')}) + machine {other_of(machine)} "
              f"({om.get('cpu')}, {also}); b and c summed over both machines' pairs")
        pc = {name: pool_claim([both[m_]["claims"][name] for m_ in MACHINES]) for name, *_ in CLAIMS}
        for name, new, old, part, _, what in CLAIMS:
            d = pc[name]
            per = "; ".join(f"{m_} {both[m_]['claims'][name]['b']} vs {both[m_]['claims'][name]['c']}" for m_ in MACHINES)
            print(f"  {name}  {what}: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, {old} only {d['c']} "
                  f"({per}); p = {d['p']:.4g}" + ("" if d["valid"] else f"   (Part {part} not valid on a machine)"))
            print(f"     *** {name} (pooled): {verdict(d)} ***")
        short = []
        for k in BAND_KEYS:
            bs = [both[m_]["bands"][k] for m_ in MACHINES]
            b = dict(c=sum(x["c"] for x in bs), n=sum(x["n"] for x in bs), valid=all(x["valid"] for x in bs))
            band, lo, s_ = band_str(b)
            print(f"  band {k:<8} pooled ({CONFIG[k]}): {s_}")
            if band == "UNTESTED":
                short.append(f"{k} ({CONFIG[k]}): UNTESTED")
            elif band != "RELIABLE" or lo < RELIABLE_LO:
                short.append(f"{k} ({CONFIG[k]}): {band} {b['c']}/{b['n']}, Wilson lower {lo:.3f}")
        print()
        print("  READING (pooled):")
        if not short:
            print(f"     \"{READING}\" (all three RELIABLE pooled, each Wilson 95% lower bound >= {RELIABLE_LO})")
        else:
            print(f"     not \"{READING}\"; falls short at:")
            for s_ in short:
                print(f"       {s_}")
        print()
        print("  Printed, not claims (pooled):")
        for new, old, part in PRINTED:
            d = pool_claim([both[m_]["printed"][new] for m_ in MACHINES])
            print(f"     {new} vs {old}, BOUND: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, {old} only {d['c']}; "
                  f"one-sided p = {d['p']:.4g} ({new} higher), {mcnemar_greater(d['c'], d['b']):.4g} ({old} higher)")
        print()
    else:
        print("POOLED (PRIMARY): not computed here — it needs --also with the other machine's complete file")
        print()
    print(f"PER MACHINE (SECONDARY): machine {machine}")
    print_claims(t, " on this machine")
    print()
    print("  Printed, not claims:")
    for new, old, part in PRINTED:
        d = t["printed"][new]
        print(f"     {new} vs {old}, BOUND: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, {old} only {d['c']}; "
              f"one-sided p = {d['p']:.4g} ({new} higher), {mcnemar_greater(d['c'], d['b']):.4g} ({old} higher)")
    for k in KEYS:
        if ARM[k]["ceil"]:
            continue
        trs_ = [r["transition"] for r in rA[k].values() if ok_r(r) and r["transition"] is not None]
        print(f"     median transition {k:<8} {med_int(trs_) if trs_ else 'none bound'} ({len(trs_)} bound)")
    print()
    if to is not None:
        print(f"  Machine {other_of(machine)} (from --also, the same tests on its pairs):")
        print_claims(to, f" on {other_of(machine)}")
        print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  failure classes and outcomes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome):")
    for k in KEYS:
        if ARM[k]["ceil"]:
            continue
        part = TPART[ARM[k]["part"]]
        cls = {}
        for s, r in sorted(rA[k].items()):
            cls.setdefault(tag(part, r), []).append(s)
        print(f"    {k:<8} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<32} {' '.join('s' + str(x) for x in v)}")
    print()
    print("  Part A per seed: SLOW_M | WIN_M (WIN_M's firings):")
    for s in seeds["WIN_M"]:
        a_, b_ = rA["SLOW_M"].get(s), rA["WIN_M"].get(s)
        print(f"    s{s}  {tag('0', a_):<24} | {tag('0', b_):<24} {fire_str(b_) if ok_r(b_) else ''}")
    print()
    print("  Part C per seed: WIN8_A | WIN8_M (transition; WIN8_M's firings):")
    for s in seeds["WIN8_M"]:
        a_, b_ = rA["WIN8_A"].get(s), rA["WIN8_M"].get(s)
        ta = f"{tag('A', a_)} ({fmt_step(a_['transition']) if ok_r(a_) else '--'})"
        tb = f"{tag('A', b_)} ({fmt_step(b_['transition']) if ok_r(b_) else '--'})"
        print(f"    s{s}  {ta:<34} | {tb:<34} {fire_str(b_) if ok_r(b_) else ''}")
    print()
    print("  firings: per arm, runs with a firing; the first firing's update, median [min, max]; the hinge/task gradient-"
          "norm ratio over every firing, median [min, max]; would-fires after the window:")
    for k in KEYS:
        if ARM[k]["tau"] is None:
            continue
        done = [r for r in rA[k].values() if ok_r(r) and firings(r)[0] is not None]
        fa = [firings(r)[0] for r in done]
        rat = [d_["ratio"] for r in done for d_ in (firings(r)[1] or [])]
        wa = [len(firings(r)[2] or []) for r in done]
        first = [x[0] for x in fa if x]
        print(f"    {k:<8} runs with a firing {sum(1 for x in fa if x)}/{len(done)}; first firing {med_int(first) if first else '--'}; "
              f"last {max(u for x in fa for u in x) if any(fa) else '--'}; firings in all {sum(len(x) for x in fa)}; ratio "
              f"{(f'{statistics.median(rat):.1f} [{min(rat):.1f}, {max(rat):.1f}]') if rat else '--'}; runs with would-fires "
              f"{sum(1 for x in wa if x)} ({sum(wa)} updates)")
    print()
    hosts = {}
    for k in KEYS:
        for r in rA[k].values():
            if ok_r(r) and r.get("host"):
                h_ = r["host"]
                key_ = (h_.get("cpu"), h_.get("onednn_bf16"), h_.get("ns_fp"))
                hosts.setdefault(key_, {}).setdefault(k, 0)
                hosts[key_][k] += 1
    print("  hosts in the records (CPU label, oneDNN bf16 matmul kernel, Newton-Schulz fingerprint): runs per arm:")
    for key_, d_ in hosts.items():
        print(f"    {key_}: {d_}")
    print()
    print("  Part B: merged runs (two or more streams on one channel at the end) and the map at 4800, 9600 and the end "
          "(distinct channels holding the 4 streams, streams per channel):")
    for k in ("WIN16_A", "WIN16_M"):
        mg = [s for s, r in sorted(rA[k].items()) if ok_r(r) and merged(r)]
        print(f"    {k}: merged at the end {len(mg)} ({' '.join('s' + str(x) for x in mg) or '-'})")
        for s in seeds[k]:
            r = rA[k].get(s)
            if not ok_r(r):
                continue
            print(f"      s{s}  4800: {groups_str(cmap_at(r, 4800)):<28} 9600: {groups_str(cmap_at(r, 9600)):<28} end: "
                  f"{groups_str(cmap_at(r, 'end')):<28} {tag('B', r)}  (transition {fmt_step(r['transition'])}; shared "
                  f"{shared_max(r)})")
    print()
    print("  Part B per seed: WIN16_A | WIN16_M (transition):")
    for s in seeds["WIN16_M"]:
        a_, b_ = rA["WIN16_A"].get(s), rA["WIN16_M"].get(s)
        ta = f"{tag('B', a_)} ({fmt_step(a_['transition']) if ok_r(a_) else '--'})"
        tb = f"{tag('B', b_)} ({fmt_step(b_['transition']) if ok_r(b_) else '--'})"
        print(f"    s{s}  {ta:<40} | {tb}")
    print()

    if to is not None:
        print("=" * 100)
        print(f"X AND L SIDE BY SIDE (each machine's own seeds; pooled counts are the sums)")
        print("=" * 100)
        both = {machine: t, other_of(machine): to}
        print(f"  {'':34} {'X':>14} {'L':>14} {'pooled':>14}")
        for k in KEYS:
            part = TPART[ARM[k]["part"]]
            cs = [(sum(success(part, r) for r in both[m_]["rA"][k].values()), len(both[m_]["seeds"][k])) for m_ in MACHINES]
            print(f"  {k:<8} {('perfect gate BOUND' if ARM[k]['ceil'] else outcome_key(part)):<25} "
                  + " ".join(f"{f'{c}/{n}':>14}" for c, n in cs) + f" {f'{sum(c for c, _ in cs)}/{sum(n for _, n in cs)}':>14}")
        for k in ("WIN16_A", "WIN16_M", "WIN8_A", "WIN8_M"):
            cs = [(sum(by_update(r, BY) for r in both[m_]["rA"][k].values()), len(both[m_]["seeds"][k])) for m_ in MACHINES]
            print(f"  {k:<8} {f'bound by {BY}':<25} " + " ".join(f"{f'{c}/{n}':>14}" for c, n in cs)
                  + f" {f'{sum(c for c, _ in cs)}/{sum(n for _, n in cs)}':>14}")
        for name, *_ in CLAIMS:
            ds = [both[m_]["claims"][name] for m_ in MACHINES]
            cells = [f"{d_['b']} vs {d_['c']}, {d_['p']:.3g}" for d_ in ds] + [f"{sum(d_['b'] for d_ in ds)} vs {sum(d_['c'] for d_ in ds)}"]
            print(f"  {name + ' discordant, p':<34} " + " ".join(f"{c_:>14}" for c_ in cells))
        print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps)")
    print("=" * 100)
    for k in KEYS:
        part = TPART[ARM[k]["part"]]
        for s in seeds[k]:
            r = get(store, k, s)
            if ok_r(r):
                print(f"  {k:<8} s{s} {curve_str(r['curve'])}  -> {tag(part, r)}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


def main():
    global SEEDS
    ap = argparse.ArgumentParser(description="MUON + WINDOW on independent seeds per machine, and at eight keys.")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--early", default=None, help="this machine's test_early_recipe results (X: results/X/early_recipe_results.json; "
                                                  "L: early_recipe_results.json)")
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (HINGE0 280, HINGE8 220)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (inherited)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (inherited)")
    ap.add_argument("--scale", default=tsa.RESULTS_FILE, help="this machine's test_scale_axes results (inherited)")
    ap.add_argument("--short", default=ter.tsc.RESULTS_FILE, help="this machine's test_short_conv results (inherited)")
    ap.add_argument("--also", default=None, help="the other machine's muon_recipe_results.json (pooled claims)")
    ap.add_argument("--report", action="store_true", help="only the report, from a finished results file")
    args = ap.parse_args()
    torch.set_num_threads(1)
    SEEDS = seeds_for(args.machine)
    early_path = args.early or ("results/X/early_recipe_results.json" if args.machine == "X" else ter.RESULTS_FILE)

    if args.report:
        store = load_store(args.results)
        m = store.get("meta", {})
        assert m.get("machine") == args.machine, f"{args.results} was written by machine {m.get('machine')}"
        SEEDS = {k: tuple(v) for k, v in m["seeds"].items()}
        report(store, SEEDS, args.machine, m.get("drop", ""), 0.0, args.results, args.also)
        return

    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    early = load_store(early_path)
    if early.get("meta", {}).get("cpu") != cpu_model():
        print(f"STOP: {early_path} was written on {early.get('meta', {}).get('cpu')}, this machine is {cpu_model()} "
              f"(CHECK 130 needs this machine's own test_early_recipe file)")
        sys.exit(2)
    also = load_store(args.also) if args.also else None
    files = {"early": early, "early_path": early_path}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    sched = dict(SCHED)

    print("=" * 100)
    print("Muon recipe: MUON + WINDOW on independent seeds per machine (Parts A and B), and at eight keys (Part C)")
    print(f"  machine {args.machine}: {cpu_model()}")
    print(f"  MUON, ADAM and WINDOW as test_early_recipe (Muon lr {MUON_LR:g}; slow phase and window 1-{sched['warm']})")
    print(f"  Part A: S=2, P=4, k=2, no conv, {sched['iters_a']} steps (DISCOVERED); Part B: S=4, P=4, k=16, conv, "
          f"{sched['iters_b']} steps; Part C: S=2, P=8, k=2, conv, {sched['iters_c']} steps (BOUND; M2, M3 bound by {BY})")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<62} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  --early      {early_path} (CPU {early['meta'].get('cpu')}, git {early['meta'].get('git')}, {len(early.get('runs', {}))} records)")
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        m_ = files[k].get("meta", {})
        print(f"  --{k:<11} {files[k + '_path']} (CPU {m_.get('cpu')}, git {m_.get('git')}, {len(files[k].get('runs', {}))} records)")
    if also is not None:
        print(f"  --also       {args.also} (machine {also.get('meta', {}).get('machine')}, CPU {also.get('meta', {}).get('cpu')})")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        info = verify(pool, files, args.machine, early, also)
        store = load_store(args.results)
        old_m = store.get("meta", {})
        if old_m and old_m.get("machine") not in (None, args.machine):
            print(f"STOP: {args.results} holds machine {old_m.get('machine')}'s runs")
            sys.exit(2)
        old_drop = None if args.force else old_m.get("drop_decision")
        store["meta"] = dict(machine=args.machine, torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head,
                             lr=LR, lr_warm=LR_WARM, muon_lr=MUON_LR, tau=TAU, sched=sched, host=info["host"],
                             early=dict(path=early_path, git=early["meta"].get("git")),
                             started=time.strftime("%Y-%m-%d %H:%M:%S"))
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — pool-load timing: each arm's real path for {TIME_ITERS} steps (one evaluation and its statistics "
              f"included), {args.workers} simultaneous copies on the {args.workers} workers, the median; worst case = every "
              f"run to its last step")
        print("=" * 100)
        cost = {}
        for k in KEYS:
            ts = list(pool.map(time_job, [k] * args.workers))
            cost[k] = statistics.median(ts)
        full, d, total_h = projection(cost, sched, args.workers)
        for k in KEYS:
            print(f"  {k:<8} {cost[k] * 1000:6.1f} ms/step x {iters_of(ARM[k], sched)} steps x {len(SEEDS[k])} seeds   "
                  f"(a full run {full[k] / 60:.1f} min)")
        print(f"  {len(d)} runs: serial {sum(d) / 3600:.2f} h, {total_h:.2f} h on {args.workers} workers")
        drop = ""
        if old_drop is not None:
            if old_drop:
                SEEDS["WIN8_A"] = tuple(old_drop)
                drop = f"WIN8_A cut to seeds {old_drop[0]}-{old_drop[-1]}"
            full, d, total_h = projection(cost, sched, args.workers)
            print(f"  the drop decision recorded at the first start of this results file is kept: {drop or 'nothing cut'} "
                  f"(projection now {total_h:.2f} h)")
            store["meta"]["drop_decision"] = old_drop
        else:
            decision = []
            if total_h > DROP_H:
                SEEDS["WIN8_A"] = SEEDS["WIN8_A"][:CUT_N]
                decision = list(SEEDS["WIN8_A"])
                drop = f"WIN8_A cut to seeds {decision[0]}-{decision[-1]}"
                full, d, total_h = projection(cost, sched, args.workers)
                print(f"  *** above {DROP_H:g} h: {drop} (drop rule); projection now {total_h:.2f} h ***")
                if total_h > DROP_H:
                    print(f"  (still above {DROP_H:g} h; the rule has no further step: Parts A and B and WIN8_M are never cut)")
            else:
                print(f"  within {DROP_H:g} h: nothing is cut")
            store["meta"]["drop_decision"] = decision
        store["meta"].update(projected_wall_h=total_h, drop=drop, seeds={k: list(v) for k, v in SEEDS.items()},
                             ms_per_step={k: cost[k] * 1000 for k in KEYS})
        save_results(args.results, store)
        print()

        def describe(key, rec):
            if not rec.get("ok"):
                return f"*** FAILED: {rec.get('error')}"
            part = TPART[ARM[key]["part"]]
            tr = rec["transition"]
            mid = (f"map [{','.join(str(c) for c in rec['end']['ch_map'])}] streams {[round(v, 2) for v in rec['end']['stream_acc']]}"
                   if part == "B" else f"VALcos {rec['val_cos']:.3f}" if rec.get("val_cos") is not None else "")
            return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} {mid}  "
                    f"{tag(part, rec)}  hinge {hstr(rec)}  firings {fire_str(rec)}")

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<8} seed {sp['seed']}  {describe(sp['arm'], rec)}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

        specs = []
        for k in sorted(KEYS, key=lambda k: (not ARM[k]["ceil"], -full[k])):
            specs += [spec(k, s, sched) for s in SEEDS[k]]
        todo = []
        for sp in specs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<8} seed {sp['seed']}  cached")
            else:
                todo.append(sp)
        print("=" * 100)
        print(f"RUNS — {len(todo)} runs, the perfect gates first, then longest first")
        print("=" * 100)
        pending, inflight = list(todo), {}
        while pending or inflight:
            while pending and len(inflight) < args.workers:
                sp = pending.pop(0)
                inflight[pool.submit(run_job, sp)] = sp
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fu in done:
                sp = inflight.pop(fu)
                try:
                    rec = fu.result()
                except Exception as e:
                    rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
                record(sp, rec)
        print()

    report(store, SEEDS, args.machine, drop, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
