#!/usr/bin/env python
"""
explore_gpu/s75b.py — EXPLORATORY, not a result. S75b: S75 at S = 4 on a configuration where the bar is reachable. The
specification is explore_out/I/s75b_spec_draft.md at 114cd08, approved by the user on 2026-10-12. This driver implements
it; nothing below changes a constant of it.

PROVENANCE, AS GIT SHOWS IT (SPEC_PROVENANCE below). It is written into the benchmark files, the results files' meta, the
stage-1 record, stage 2's meta and the report.
  - The design commit, 1e31409, was built on 5bc6c2e. That tree holds only S75's ORACLE 830-831 and SINGLE 830-839
    records. That the design was written blind to S75's learned arms is the author's statement: b649751, holding them,
    was on the remote 23 min earlier.
  - Every change after the merge 142f47e, which brought b649751's WIDE_SPLIT and REG3 records into the tree, is NOT
    BLIND BY GIT:
    - the cost section (ee97443);
    - the revision 114cd08, including the loosened proceed rule and WIDE_SLOW made unconditional;
    - every change after 114cd08 (CHANGES_SINCE_SPEC).

CONFIGURATION B: ICMC S = 4, V = 4, blocks 6..14, NB = 32, M = 352, L = 1536 (the model reads 1535); the model, B = 32,
  Adam 1e-3, evaluation every 1200 updates on 1024 held-out sequences, the probe and the routing statistics as S75;
  24000 updates per run.
STAGE 1, the pilot: ORACLE_PILOT (perfect gate, k = 4, ADAM), seeds 840 and 841; its own arm and records, never in VALID,
  GAP or any stage-2 aggregate.
  PROCEED (fixed): both pilot records exist and ok, mean(end.set) >= 0.912 and each end.set >= 0.9 (end.set = the
  final evaluation's overall SET at 24000 updates on the 1024 held-out sequences, unrounded).
  SPLIT_THR := min(end.set of ORACLE_PILOT 840, 841) - 0.03, unrounded.
  STOP otherwise: S75b stops; "S = 4 is not usable at B" only if a pilot run ended below 0.9 or diverged (FAILURES); no
  other configuration.
  The stage-1 record (s75b_stage1_record.json / .md, written by --stage 1) names the machine the pilot ran on and must be
  committed before stage 2: --stage 2 refuses unless the record file is committed and clean, both pilot records exist
  under the driver's code SHA and the proceed rule holds; it writes SPLIT_THR and the record's commit into its meta.
STAGE 2, seeds 850-859: ORACLE (850-851, validity), SINGLE, WIDE_SPLIT_G (WIDE window gate, k = 4, SLOW, the split with
  the one-to-one guard), REG3 (as S75), WIDE_SLOW (WIDE_SPLIT_G without the split; printed, never read; runs
  unconditionally). Every arm runs all its seeds whatever ORACLE 850-851 show.
  THE GUARDED TRIGGER (make_split_guarded): explore_i_common.make_split transcribed, installed in this process by
  assigning ic.make_split for WIDE_SPLIT_G's runs (run_batched_icmc looks it up at run time); every helper is called
  through explore_i_common, so the CUDA patches of explore_gpu.run.eval_scope apply. Precedence at a check: (1) not
  eligible: as S75; (2) eligible and capped or gapped: blocked by "cap" / "gap", as S75; (3) eligible, not capped or
  gapped, and the source -> channel map at word positions one-to-one (routing_layer's own one_to_one): blocked by
  "one-to-one", the fire count and the last-fire step unchanged; (4) otherwise the split fires. prev is updated at
  every check. Every row also records the map's eta, margin and one_to_one.
OUTCOMES, VALID, READING (S75's): ROUTED := at every gate layer eta^2 by source at word positions >= 0.9 and a one-to-one
  map, at the end. GAP := (SET(arm) - SET(SINGLE, same seed)) / (mean SET(ORACLE 850-851) - SET(SINGLE, same seed)),
  overall and by bin. VALID := both ORACLE 850-851 end with SET >= 0.9. READING: "carries to four sources" if the better
  of WIDE_SPLIT_G and REG3 is ROUTED on >= 8/10, "does not carry" otherwise; UNTESTED if not VALID (then ROUTED counts,
  GAP, eta^2, margin and the probe are descriptions only, never re-read under a later bar, and the pilot is never pooled
  with 850-851).
MACHINES (fixed): the GTX 1660 SUPER runs ORACLE 850-851, SINGLE and WIDE_SLOW; the RTX 3070 Laptop runs REG3 and
  WIDE_SPLIT_G; if the 3070 is unavailable or unusable when stage 2 starts, every arm runs on the 1660 (--machine
  fallback) and nothing is dropped; if the 1660 is unavailable or its benchmark does not allow its arms, every arm runs on the 3070
  (--machine fallback-3070); both fallbacks need --reason (FAILURES). R = 1 for every arm on both cards. REG3's register mode by rule from this card's benchmark: "scan" if
  the benchmark's peak reserved memory with the scan, plus the context (memory in use beyond this process's reserve),
  is below the card's total; otherwise "loop". --stats-workers from the benchmark's host memory, by this driver's
  formula (workers_rule; listed under CHANGES).

FAILURES (rules added on 2026-10-13, before the pilot, for the outcomes Referee V's review found uncovered, then revised
after four reviews of this driver; NOT BLIND; spec Appendix B):
  - KINDS.
    - DIVERGED. A run whose evaluation logits are not all finite. runtime() checks every evaluation and the split's probe
      before anything else, on CPU and on CUDA. The record's non-finite-loss check, or a NaN or inf reaching a
      statistic, also count. It is an outcome: never retried, never rerun, under any code, on any card.
    - OPERATIONAL. Any other failure: out of memory, a crash, a driver error. A process that ended without a record
      (killed, interrupted) leaves an open start marker, which counts as an operational failure of that invocation.
  - AN INVOCATION stops at the first operational failure that survives the driver's one retry (InvocationAborted), so
    only that seed is charged.
  - RERUNS. An operational failure is rerun on the same seed, never a fresh one, on whichever card runs that arm (the
    record keeps each seed's card).
  - HALTS.
    - A seed with operational failures in OPERATIONAL_LIMIT = 2 invocations is HALTED. The driver does not run it again
      until the user clears it (--retry-halted) or the code changes before stage 2.
    - A clear names exactly the invocations it clears; no rule compares times across machines. Every time S75b writes is
      UTC with its zone.
    - A halt waits on the user. It is never a STOP, never final, and says nothing about B.
  - THE PILOT.
    - A value STOP is decided as soon as a pilot run ends below 0.9 or diverges, whatever the other seed: "S = 4 is not
      usable at B".
    - Otherwise the proceed rule applies once both runs are ok.
    - A halted pilot seed leaves the decision to the user.
  - A CODE CHANGE.
    - Before stage 2 starts: the code (every module in the code SHA) is frozen from stage 1. A change that cannot be
      avoided (a bug that stops a stage) needs a committed, dated note, passed as --code-change-note to the first stage 1
      under the new code; the CUDA CHECKs, the benchmarks and the pilot (840-841) are rerun under it. The note is needed
      whenever another code has a stage-1 record (current or archived) or pilot entries.
    - Stage 1 writes its record before the pilot's first run (marked running) and rewrites it when it ends, so a later
      code's stage 1 archives the earlier record as soon as it starts, even if it is then killed.
    - A value STOP under any code is final. It is recomputed from every code's pilot records, not only from the record
      files.
    - Stage 2 needs every complete stage-1 decision, under every code, to be PROCEED. SPLIT_THR comes from the pilot
      under the code stage 2 runs. Earlier stage-1 records are kept as s75b_stage1_record_<sha>.*, printed, never pooled.
    - A superseded code (its record archived, or its pilot entries left while the record is under another code) is
      never used again: a revert cannot re-adopt an earlier pilot. The report gives no reading if the stage-2 code was
      superseded.
    - While a stage runs: the code on disk is compared with the invocation's before and after every run (the data and
      statistics workers import it from disk). A change before a run stops the invocation, with nothing charged; a
      change during a run keeps that record, marked code_changed, and halts the screen: stages refuse and the report
      gives no reading until the user decides, recorded with --code-changed-decision keep (the record counts) or rerun
      (it is set aside, kept and printed, and its seed is rerun) on the card that ran it, with the code on disk the
      invocation's again. Do not pull or edit code while a stage runs.
    - Once stage 2 has a record or a failure under one code, the code is frozen for good. Stage 1 and stage 2 refuse
      under any other code: a change halts the screen for the user, and no outcome is re-drawn. The other machine sees
      the freeze once stage 2's results file is pushed, so it is pushed as soon as stage 2 starts.
  - STAGE 2.
    - A diverged ORACLE 850 / 851 run counts as below 0.9: NOT VALID, so the reading is UNTESTED.
    - A diverged learned-arm run is not ROUTED, and its seed stays in n (n stays 10).
    - A diverged SINGLE run gives no GAP for its seed.
    - A halted ORACLE or read-arm seed leaves validity or the reading incomplete until the user decides; a halted SINGLE
      seed gives no GAP for it, a halted WIDE_SLOW seed no paired row.
    - A seed with an outcome (an ok record or a divergence) from two cards was drawn twice: stages refuse and the report
      marks the reading incomplete until the user decides which draw counts.
  - THE BENCHMARK.
    - Usability is per arm: an arm runs on a card iff the card's latest benchmark under this code finished, passed every
      case the arm uses (REG3 in the chosen register mode), and its host-memory rule gave workers.
    - A stage runs the arms its card allows and names the ones it refuses; a stage with no allowed arm refuses.
    - A benchmark may be rerun under the same code at any time: it measures and decides no outcome. Each earlier file is
      kept in bench_archive/. A benchmark that crashes leaves its file "in progress", which counts as failed. Each run
      record carries the benchmark it ran under (commit, written) and its workers. An archived file is never
      overwritten.
    - If one card is unusable or unavailable, the other runs everything: --machine fallback on the 1660, --machine
      fallback-3070 on the 3070. Both need --reason, which is written into stage 2's meta and records. If the other card
      refuses only some of its arms, the fallback runs those with --arms. A fallback never runs arms that the other
      card's unfinished invocation, as pushed, runs (ONE INVOCATION PER CARD).
    - If no card's benchmark allows the pilot, S75b cannot start. If no card's benchmark allows a stage-2 arm, that arm
      cannot run: ORACLE leaves validity incomplete and a read arm the reading; SINGLE gives no GAP and WIDE_SLOW no
      paired rows. No configuration, R or batch is changed to make a card fit; the user decides.
  - SYNC. With commits required, stage 1 and stage 2 refuse unless a git fetch succeeds, the checkout is neither
    behind nor ahead of its upstream, and every S75b JSON file in OUT is committed (stage 2 checks this first). Stage 2
    reads the pilot from every card's committed results file. The stage-1 record and the results files are committed
    and pushed whatever the decision. --device cpu is for the CHECKs only.
  - ONE INVOCATION PER CARD. A stage or a benchmark holds locks on its card (card_lock: one in the checkout, one per
    machine) for its whole run; a second one on the card, from any checkout on the machine, refuses. Each results file's
    meta holds its card's last invocation and whether it ended, and both stages ask for a push at their start. A stage
    refuses while another card's last invocation, as pushed, has not ended and runs any of its arms (stage 1: the
    pilot). If that process was killed, its card's results file is first committed and pushed from its own machine (it
    holds the runs finished before the kill); only then does the user state with --other-card-idle (recorded) that no
    invocation runs there. If that machine cannot be reached, the user decides: its unpushed runs would be drawn again.
    Two starts that were not pushed are caught afterwards, as a seed with outcomes from two cards. Stage 2 with --arms
    runs only the named arms of its machine (recorded), e.g. a fallback for the arms the other card's benchmark
    refuses.

CHANGES SINCE 114cd08 (spec Appendix A; CHANGES_SINCE_SPEC below holds these items and Appendix B's failure, sync and
  provenance items, and is written into the benchmark files, the results files' meta, the stage-1 record, stage 2's meta
  and the report). They were made after S75's learned-arm records were read, so they are NOT BLIND. None changes a
  design constant (configuration, seeds, arms, bars, the proceed rule, SPLIT_THR's rule); the failure rules (FAILURES)
  extend the outcomes, VALID and the reading to failed runs:
  - The pilot may run on the RTX 3070 if it is free before the GTX 1660 (the user's message of 12 October, round 3); the
    stage-1 record names the card.
  - Host memory: in S75b's processes the routing margin (explore_i_common.margin_of) runs in chunks of MARGIN_CHUNK = 16
    sequences instead of 64 (the same value up to float64 summation order: CHECK B3), and the evaluation statistics'
    worker pool is this module's StatsPool (explore_gpu.run.AsyncStats with that patch applied in each worker).
  - --stats-workers: the spec says only "from the benchmark's host memory"; the formula is this driver's (workers_rule).
    The benchmark measures the main process's, one statistics worker's and the data process's own peak RSS (VmHWM), the
    tensors one evaluation hands to the statistics, the seconds one evaluation's statistics take and the seconds per
    update. The budget is the smaller of (the machine's RAM, the user's figure, - the co-running job's measured peak
    (--beside) - OS_RESERVE_GB) and (MemAvailable at the benchmark's start - the co-running job's possible regrowth to its
    peak - AVAIL_RESERVE_GB). Workers: the smallest w <= 3 that keeps up with the evaluations and fits; else the largest
    that fits (the evaluations it falls behind on are held in the main process and counted); else 0 (the statistics in
    line) if that fits; else the benchmark refuses.
  - Records of one arm and seed from two cards (an ok record or a divergence on each) are not resolved by the driver:
    stage 1 / 2 refuse, and the report prints them and marks the reading incomplete.
  - Order: stage 1 needs the driver's code, this card's benchmark and its CUDA CHECK output committed; stage 2 needs the
    driver's code, the stage-1 record, every results file, and this card's benchmark and CUDA CHECK output committed.
  - CHECK B7 (the spec's lock-step at L = 1536): R = 1 as the runs, each seed alone, batch 8 on CUDA (the R = 2, batch-32
    lock-step does not fit either card), REG3 in both register modes on both cards.

RUN (from the repository root; every step writes into explore_out/I/gpu/s75b/):
  python -m explore_gpu.s75b_checks --device cpu            CHECKs (this container); --device cuda on each card (commit
                                                            explore_gpu/s75b_checks_cuda_<GPU>_output.*)
  python -m explore_gpu.s75b --bench --device cuda --host-ram-gb G --beside NAME|none
        each card, before its first stage: G = the machine's RAM in GB (the user's figure: 15.2 on the laptop), NAME = a
        string in the command line of the job running beside S75b (the laptop: test_delta_recipe), "none" if nothing
        does; commit the bench file
  python -m explore_gpu.s75b --stage 1 --device cuda        the pilot (one card only); commit and push the results file
                                                            and the running record as soon as it starts, and the record
                                                            and every results file when it ends
  python -m explore_gpu.s75b --stage 2 --device cuda --machine 1660|3070|fallback|fallback-3070 [--reason TEXT]
        [--arms A,B]; commit and push the results file as soon as it starts; do not pull or edit code until it ends
  add --retry-halted to clear this stage's halted seeds (the user's decision; recorded); --other-card-idle if the other
  card's last invocation was killed, after its results file was pushed from its own machine (recorded);
  --code-changed-decision keep|rerun on runs whose code changed while they ran (recorded)
  python -m explore_gpu.s75b --report                       the counts table -> s75b_summary.md
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

import explore_i_task as it
import explore_i_common as ic
import explore_i_dense as idense
from explore_gpu import device as dv
from explore_gpu import run as gr

NAME = "s75b"
OUT = os.path.join(HERE, "explore_out", "I", "gpu", "s75b")
CONFIG = dict(S=4, V=4, NB=32, lmin=6, lmax=14, M=352)
ITERS = 24000
EVAL_EVERY = 1200                          # S75's (explore_gpu.run.EVAL_EVERY); the split trigger runs after evaluations
PILOT_SEEDS = (840, 841)
ORACLE_SEEDS = (850, 851)
SEEDS = tuple(range(850, 860))
BAR = 0.9                                  # VALID (S74's registered bar)
PROCEED_MEAN, PROCEED_EACH = 0.912, 0.9    # 0.9 + S75's ORACLE seed spread (0.892 - 0.880); each run >= 0.9
THR_MARGIN = 0.03                          # SPLIT_THR := min(pilot) - 0.03 (S75's rule)
READ_ARMS = ("WIDE_SPLIT_G", "REG3")
HI = 8
ARMS = {                                   # arm -> (explore_gpu.run.S75_ARMS key, seeds)
    "ORACLE_PILOT": ("ORACLE", PILOT_SEEDS),
    "ORACLE": ("ORACLE", ORACLE_SEEDS),
    "SINGLE": ("SINGLE", SEEDS),
    "WIDE_SPLIT_G": ("WIDE_SPLIT", SEEDS),
    "REG3": ("REG3", SEEDS),
    "WIDE_SLOW": ("WIDE_SLOW", SEEDS),
}
STAGE2 = ("ORACLE", "SINGLE", "WIDE_SPLIT_G", "REG3", "WIDE_SLOW")
MACHINE_ARMS = {"1660": ("ORACLE", "SINGLE", "WIDE_SLOW"), "3070": ("REG3", "WIDE_SPLIT_G"), "fallback": STAGE2,
                "fallback-3070": STAGE2}
MACHINE_GPU = {"1660": "GTX 1660", "3070": "RTX 3070", "fallback": "GTX 1660", "fallback-3070": "RTX 3070"}
OPERATIONAL_LIMIT = 2                      # invocations with an operational failure of one seed before it is HALTED
MARGIN_CHUNK = 16
MAX_WORKERS = 3                            # explore_gpu.run's default on CUDA
DATA_PROC_GB = 0.5                         # the data process, where its peak cannot be measured (no /proc)
OS_RESERVE_GB = 1.5                        # the operating system and everything else not measured
AVAIL_RESERVE_GB = 0.5                     # kept free of MemAvailable
CHECKS_DIR = os.path.join(HERE, "explore_gpu")   # where explore_gpu.s75b_checks writes its outputs
MACHINE_LOCK_DIR = __import__("tempfile").gettempdir()   # the per-machine card lock (every checkout); CHECKs repoint it

SPEC_COMMIT = "114cd08"
SPEC_PROVENANCE = dict(
    design="1e31409", design_parent="5bc6c2e", merge="142f47e", learned_arms="b649751",
    blind_by_git=[
        "1e31409, the design: built on 5bc6c2e, whose tree holds only S75's ORACLE 830-831 and SINGLE 830-839 records. "
        "That it was written blind is the author's statement; git shows the tree, not what was read (b649751 was on "
        "the remote at 13:53:11 UTC, 23 min before 1e31409). The same commit adds explore_gpu/s75_writeup.py, the "
        "learned-arm descriptions generator; git cannot show whether it was run on b649751's records before the commit"],
    not_blind_by_git=[
        "ee97443, the cost section: made after the merge 142f47e, with b649751's WIDE_SPLIT and REG3 records and "
        "s75_S4_summary.md in the tree",
        "114cd08, the revision (with b649751 in the tree; the same session committed the S75 write-up 03bffdf 130 s "
        "later). It loosened the proceed rule from 'both >= 0.912' to 'mean >= 0.912 and each >= 0.9', and made "
        "WIDE_SLOW unconditional. It made corrections and operational choices: the bound 0.963, the gap share, the "
        "windowed learner and the fixed-reach case, the stop wording, the guard's precedence, the allocation, R = 1, "
        "the register rule, the workers, n = 4096, the memory and the 3070's factor. It added the other rules in "
        "15c5e4f..114cd08: ORACLE_PILOT as its own arm, never substituting for ORACLE 850-851; every arm runs all its "
        "seeds; not VALID -> descriptions only, never re-read, the pilot never pooled; the successor rule; the trigger's "
        "transcription and precedence; the CHECK list and the Order",
        "every change after 114cd08 (CHANGES_SINCE_SPEC)",
    ])
CHANGES_SINCE_SPEC = dict(
    spec=SPEC_COMMIT, blind=False,
    note="made after S75's learned-arm records were read and with them in the tree, so not blind (by git either); none "
         "changes a constant of the design (configuration, seeds, arms, bars, the proceed rule, SPLIT_THR's rule); the "
         "failure rules extend the outcomes, VALID and the reading to failed runs",
    items=[
        "the pilot may run on the RTX 3070 if it is free before the GTX 1660 (the user, 12 October); the stage-1 record "
        "names the card",
        "the routing margin in chunks of 16 sequences in S75b's processes and the statistics pool StatsPool (host memory; "
        "the same value up to float64 summation order, CHECK B3)",
        "--stats-workers by this driver's formula (workers_rule; the spec says only 'from the benchmark's host memory'): "
        "measured peaks (VmHWM) of the main process, one statistics worker and the data process, the evaluations held "
        "while the workers lag, and a budget from the machine's RAM (the user's figure), the co-running job's measured "
        "peak (--beside) and MemAvailable",
        "records of one arm and seed from two cards (an ok record or a divergence on each) are not resolved: stage 1 / 2 "
        "refuse, the report marks the reading incomplete",
        "order: stage 1 needs the driver's code, this card's benchmark and CUDA CHECK output committed; stage 2 needs the "
        "driver's code, the stage-1 record, every results file and this card's benchmark and CUDA CHECK output "
        "committed",
        "CHECK B7 (lock-step at L = 1536): R = 1 as the runs, each seed alone, batch 8 on CUDA, REG3 in both register "
        "modes (the R = 2, batch-32 lock-step does not fit either card)",
        "failed runs (rules added 2026-10-13, before the pilot, revised after four reviews): DIVERGED = the evaluation's "
        "logits not all finite (checked before anything else), the non-finite-loss check, or a NaN / inf in a "
        "statistic: an outcome, never retried or rerun, under any code, on any card; any other failure, including a "
        "process that ended without a record (an open start marker), is OPERATIONAL: rerun on the same seed, never a "
        "fresh seed, on whichever card runs the arm; an invocation stops at the first operational failure that "
        "survives the retry; operational failures in 2 invocations HALT the seed until the user clears it "
        "(--retry-halted, naming the invocations it clears) or the code changes before stage 2; a halt is never a STOP",
        "the pilot: a value STOP as soon as a pilot run ends below 0.9 or diverges, whatever the other seed ('S = 4 is "
        "not usable at B'); a halted seed leaves the decision to the user; the record lists every invocation and each "
        "seed's card, and stage 2 reads the pilot from every card's committed results file",
        "a code change: before stage 2, an unavoidable change needs a committed, dated note (--code-change-note, "
        "whenever another code has a stage-1 record or pilot entries) and the CUDA CHECKs, the benchmarks and the pilot "
        "(840-841) are rerun under it; stage 1 writes its record before its first run, so a later code supersedes the "
        "earlier one as soon as its stage 1 starts; a value STOP under any code is final (recomputed from every code's "
        "pilot records); stage 2 needs every complete stage-1 decision to PROCEED, SPLIT_THR from the pilot under its "
        "code; earlier records kept as s75b_stage1_record_<sha>.*, never pooled; a superseded code is never used again; "
        "once stage 2 has a record or failure, the code is frozen for good: a change halts the screen for the user, no "
        "outcome re-drawn; the code on disk is compared with the invocation's before and after every run: a change "
        "stops the invocation (nothing charged before a run; a run it overlapped is kept, marked code_changed, and the "
        "user decides: --code-changed-decision keep or rerun, recorded)",
        "stage 2: a diverged ORACLE run counts as below 0.9 (UNTESTED); a diverged learned-arm run is not ROUTED (n "
        "stays 10); a diverged SINGLE run gives no GAP for its seed; a halted ORACLE or read-arm seed leaves validity or "
        "the reading incomplete until the user decides; the report gives no reading after a value STOP, a code change "
        "in stage 2, a superseded stage-2 code or an undecided run whose code changed",
        "the benchmark: usability per arm (finished, the arm's cases passed, workers given); a stage runs the arms its "
        "card allows and names the rest; a benchmark may be rerun at any time (earlier files kept in bench_archive/; a "
        "crashed one counts as failed; each run record carries its benchmark); one card unusable or unavailable: the "
        "other runs everything (--machine fallback or fallback-3070, each with --reason in stage 2's meta and records); "
        "no card allows the pilot: S75b cannot start; no card allows an arm: the arm cannot run (ORACLE: validity "
        "incomplete; a read arm: the reading incomplete; SINGLE: no GAP; WIDE_SLOW: no paired rows); no configuration, R "
        "or batch change; the user decides",
        "sync: with commits required, stages refuse unless git fetch succeeds, the checkout is neither behind nor ahead "
        "of its upstream, and every S75b JSON file in OUT is committed; results files are pushed whatever the decision, "
        "and as soon as stage 2 starts; --device cpu is for the CHECKs only",
        "one invocation per card (locks per checkout and per machine for a stage's or a benchmark's whole run); both "
        "stages are pushed at their start; a stage refuses while another card's last invocation, as pushed, has not "
        "ended and runs any of its arms, unless the user states with --other-card-idle (recorded) that it was killed, "
        "after pushing that card's results file from its own machine; the pilot runs on one card; stage 2's --arms runs "
        "only the named arms of the machine (recorded)",
        "every time written is UTC with its zone",
        "provenance labels: SPEC_PROVENANCE marks every change after the merge 142f47e (ee97443, 114cd08 and this "
        "list) as not blind by git",
    ])


def task():
    return it.ICMCTask(**CONFIG)


def arm_spec(arm):
    return gr.arm_spec(ARMS[arm][0], CONFIG["S"])


def seeds_of(arm):
    return ARMS[arm][1]


_CODE_NAMES = {}                           # code SHA -> the modules it hashes (code_now)


def code_sha():
    sha, names = gr.code_sha("explore_gpu.s75b")
    _CODE_NAMES[sha] = list(names)
    return sha, names


def module_file(n):
    """The source file of a module in the code SHA (explore_gpu.run.code_sha hashes the root's and explore_gpu's)."""
    if n == "explore_gpu":
        return os.path.join(HERE, "explore_gpu", "__init__.py")
    return os.path.join(HERE, *n.split(".")) + ".py"


def code_now(sha):
    """The code SHA of the files on disk now, over the modules that made sha, in this process (code_sha's hash, no child
    process). The data and statistics workers each run spawns import these files from disk, so a pull or an edit while
    a stage runs reaches the runs (FAILURES: run_arm compares this with the invocation's SHA before and after each
    run)."""
    import hashlib
    names = _CODE_NAMES.get(sha)
    if names is None:
        return code_sha()[0]
    h = hashlib.sha1()
    for n in names:
        h.update(n.encode())
        try:
            with open(module_file(n), "rb") as f:
                h.update(f.read())
        except OSError:
            return f"unreadable ({n})"
    return h.hexdigest()[:12]


def utc_now():
    """UTC to the second, with its zone (every time S75b writes: FAILURES)."""
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def hw_slug(hw):
    return "".join(c if c.isalnum() else "_" for c in hw)


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def one_thread():
    """One thread per process on every device: the BLAS pools through the environment (inherited by spawned workers),
    torch's intra-op pool directly."""
    for k in THREAD_ENV:
        os.environ[k] = "1"
    torch.set_num_threads(1)


# ── host memory ───────────────────────────────────────────────────────────────
def _status_kb(pid, key):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def ru_maxrss_gb():
    """getrusage's peak RSS in GB. In a spawned worker it starts at the parent's peak (exec inherits it), so it is not the
    worker's own: recorded as a description only."""
    try:
        import resource
        k = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return k / 2 ** 30 if sys.platform == "darwin" else k / 2 ** 20
    except Exception:
        return None


def own_peak_gb():
    """This process's own peak RSS in GB: VmHWM (exec gives a spawned worker a fresh one); ru_maxrss where /proc is
    missing."""
    k = _status_kb("self", "VmHWM")
    return k / 2 ** 20 if k is not None else ru_maxrss_gb()


def rss_gb():
    return own_peak_gb()


def meminfo_gb():
    """MemTotal and MemAvailable in GB from /proc/meminfo ({} where it cannot be read)."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    out[k] = int(v.split()[0]) / 2 ** 20
    except (OSError, ValueError):
        pass
    return out


def _procs():
    """{pid: (ppid, command line)} of every process visible in /proc."""
    out = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            ppid = int(s[s.rfind(")") + 2:].split()[1])
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            out[int(d)] = (ppid, cmd)
        except (OSError, ValueError, IndexError):
            continue
    return out


def _descendants(roots, procs):
    kids = {}
    for p, (pp, _) in procs.items():
        kids.setdefault(pp, []).append(p)
    seen, todo = set(), list(roots)
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo += kids.get(p, [])
    return seen


def _ancestors(pid, procs):
    out = set()
    while pid in procs and pid not in out:
        out.add(pid)
        pid = procs[pid][0]
    return out


def beside_mem(name):
    """The job running beside S75b: every process whose command line contains `name`, with its descendants (its
    workers), never this process, its ancestors or its descendants. Summed VmRSS (now) and VmHWM (each process's peak so
    far) in GB."""
    procs = _procs()
    me = os.getpid()
    mine = _descendants([me], procs) | _ancestors(me, procs)
    roots = [p for p, (_, cmd) in procs.items() if name in cmd and p not in mine]
    pids = sorted(_descendants(roots, procs) - mine)
    return dict(name=name, roots=[dict(pid=p, cmd=procs[p][1][:160]) for p in roots], pids=pids,
                rss_gb=sum(_status_kb(p, "VmRSS") or 0 for p in pids) / 2 ** 20,
                hwm_gb=sum(_status_kb(p, "VmHWM") or 0 for p in pids) / 2 ** 20)


def children_mem():
    """{pid: dict(cmd, hwm_gb)} of this process's descendants."""
    procs = _procs()
    me = os.getpid()
    return {p: dict(cmd=procs[p][1][:160], hwm_gb=(_status_kb(p, "VmHWM") or 0) / 2 ** 20)
            for p in sorted(_descendants([me], procs) - {me})}


def payload_gb(*xs):
    """Bytes of the tensors in xs (tensors, lists / tuples and dicts of them, None), in GB."""
    n = 0
    todo = list(xs)
    while todo:
        x = todo.pop()
        if torch.is_tensor(x):
            n += x.numel() * x.element_size()
        elif isinstance(x, (list, tuple)):
            todo += list(x)
        elif isinstance(x, dict):
            todo += list(x.values())
    return n / 2 ** 30


# ── the routing margin and the statistics pool ───────────────────────────────
_MARGIN = {}


def small_margin():
    """explore_i_common.margin_of in chunks of MARGIN_CHUNK sequences (routing_layer looks it up at call time)."""
    if "orig" not in _MARGIN:
        _MARGIN["orig"] = ic.margin_of

        def margin_of(g, src, word, chunk=MARGIN_CHUNK):
            return _MARGIN["orig"](g, src, word, chunk=min(chunk, MARGIN_CHUNK))
        ic.margin_of = margin_of


_JOBS = []                                 # one dict per statistics job collected in this process (host-memory facts)


def _stats_init_b(task_, ev, pdata):
    small_margin()
    gr._stats_init(task_, ev, pdata)


def _stats_job_b(lg, gates, feats):
    t0 = time.time()
    out = gr._stats_job(lg, gates, feats)
    return dict(out, _worker_gb=own_peak_gb(), _worker_ru_maxrss_gb=ru_maxrss_gb(), _worker_secs=time.time() - t0,
                _worker_pid=os.getpid())


class StatsPool(gr.AsyncStats):
    """explore_gpu.run.AsyncStats with the small-margin patch in each worker; records each job's worker peak RSS, seconds
    and the tensors handed over (_JOBS). sync (the benchmark only): wait for each job when it is submitted, so this
    process holds one evaluation's tensors at a time."""
    sync = False

    def __init__(self, task_, ev, pdata, workers):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_stats_init_b, initargs=(task_, ev, pdata))
        self.jobs = []

    def submit(self, key, extra, lg, gates, feats):
        f = self.pool.submit(_stats_job_b, lg, gates, feats)
        if StatsPool.sync:
            f.result()
        self.jobs.append((key, extra, f, payload_gb(lg, gates, feats)))

    def collect(self):
        out = []
        for key, extra, f, pl in self.jobs:
            r = dict(f.result())
            w = {k[1:]: r.pop(k) for k in list(r) if k.startswith("_worker")}
            _JOBS.append(dict(w, payload_gb=pl))
            out.append((key, dict(r, **extra)))
        self.pool.shutdown()
        return out


class NonFinite(ValueError):
    pass


def _finite_logits(orig):
    """explore_i_common.logits_chunked, raising NonFinite when the logits are not all finite. Every evaluation (the
    device part and explore_i_common.evaluate) and the split trigger's probe compute the logits first, before the
    residual replay's assert, so a diverged run is recognised as such on CPU and on CUDA (FAILURES)."""
    def logits_chunked(*a, **kw):
        lg = orig(*a, **kw)
        if not bool(torch.isfinite(lg).all()):
            raise NonFinite("non-finite logits in the evaluation (the run diverged)")
        return lg
    logits_chunked.s75b_finite = True
    return logits_chunked


@contextlib.contextmanager
def runtime():
    """S75b's process-level patches: the statistics pool and the finiteness check on every evaluation's logits (both
    restored on exit), and the small routing margin (permanent in S75b's processes)."""
    small_margin()
    old, old_lc = gr.AsyncStats, ic.logits_chunked
    gr.AsyncStats = StatsPool
    if not getattr(old_lc, "s75b_finite", False):
        ic.logits_chunked = _finite_logits(old_lc)
    try:
        yield
    finally:
        gr.AsyncStats, ic.logits_chunked = old, old_lc


# ── the guarded split trigger ─────────────────────────────────────────────────
def make_split_guarded(task_, seed, last, thr, rise=None, max_splits=None, gap=None):
    """explore_i_common.make_split, transcribed, with the one-to-one guard (precedence in the module docstring)."""
    rise = ic.SPLIT_RISE if rise is None else rise
    max_splits = ic.SPLIT_MAX if max_splits is None else max_splits
    gap = ic.SPLIT_GAP if gap is None else gap
    pe = ic.eval_set(task_, ic.SPLIT_PROBE_N, ic.SPLIT_PROBE_SEED)
    px = pe["tokens"][:, :-1]
    st = dict(prev=None, n=0, last=None)
    rows = []

    def trig(m, step, opt):
        if step % ic.SPLIT_EVERY or step < ic.SPLIT_REF or step > last:
            return
        lg = ic.logits_chunked(m, px)
        acc = float(it.set_hits(task_, lg, pe["sets"])[pe["ann"]["mask"]].float().mean())
        g = ic.model_gates(m, px)[0]
        word = px >= task_.S
        km = g[word].mean(0)
        cs, c0 = int(km.argmax()), int(km.argmin())
        rl = ic.routing_layer(g, px, task_)
        cmap = rl["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap, masses=[round(float(v), 5) for v in km],
                   cs=cs, c0=c0, on_target=cmap.count(cs) >= 2 and cmap.count(c0) == 0, fired=False, blocked=False,
                   map_eta=rl["eta"], map_margin=rl["margin"], map_one_to_one=bool(rl["one_to_one"]))
        eligible = step >= ic.SPLIT_FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row.update(blocked=True, blocked_by="cap" if st["n"] >= max_splits else "gap")
        elif eligible and rl["one_to_one"]:
            row.update(blocked=True, blocked_by="one-to-one")
        elif eligible:
            res = ic.split_op(m, opt, cs, c0, ic.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            g2 = ic.model_gates(m, px)[0]
            row.update(fired=True, map_after=ic.routing_layer(g2, px, task_)["ch_map"], **res)
        st["prev"] = acc
        rows.append(row)
    trig.rows = rows
    return trig


@contextlib.contextmanager
def guarded(rise=None):
    """ic.make_split -> make_split_guarded in this process (run_batched_icmc looks ic.make_split up at run time)."""
    old = ic.make_split
    ic.make_split = lambda t, s, last, thr: make_split_guarded(t, s, last, thr, rise=rise)
    try:
        yield
    finally:
        ic.make_split = old


# ── stores ─────────────────────────────────────────────────────────────────────
def store_path(hw):
    return os.path.join(OUT, f"{NAME}_{hw_slug(hw)}_results.json")


def load(hw):
    p = store_path(hw)
    if not os.path.exists(p):
        return {"meta": {}, "runs": {}}
    with open(p) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def load_all():
    import glob
    out = {"meta": {}, "runs": {}, "failures": {}, "set_aside": {}}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        out["meta"][os.path.basename(p)] = st["meta"]
        out["runs"].update(st["runs"])
        out["failures"].update(st.get("failures") or {})
        out["set_aside"].update(st.get("set_aside") or {})
    return out


# ── failed runs (FAILURES in the module docstring) ────────────────────────────
INVOCATION = f"{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} pid {os.getpid()}"   # reset by each stage
OPEN_ATTEMPT = "started; the process ended without a record (killed or interrupted)"


def now_ts():
    """UTC with its offset, to the microsecond (failure entries; descriptions only: no rule compares times)."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


class InvocationAborted(RuntimeError):
    """An operational failure survived the driver's retry: the invocation stops, so only that seed is charged."""


class CodeChanged(InvocationAborted):
    """The code on disk is no longer the invocation's (a pull or an edit while a stage runs): the invocation stops. Before
    a run, nothing is charged; after a run, its record is kept, marked code_changed, and the screen halts for the user
    (FAILURES)."""


def code_changed_runs(st, sha):
    """The keys of records under sha whose code changed on disk while they ran and that no decision of the user covers
    (FAILURES: the screen halts until the user decides, --code-changed-decision)."""
    return sorted(k for k, r in st["runs"].items() if k.split("|")[2] == sha and r.get("code_changed")
                  and not r.get("decided"))


def decide_code_changed(st, hw, sha, choice, log=print):
    """--code-changed-decision keep|rerun: the user's decision on this card's undecided code_changed records under sha,
    recorded with this invocation and its time. keep: the record counts, marked with the decision. rerun: the record is
    set aside (st['set_aside'], kept and printed, never counted), and the failure entries of that attempt count as an
    operational failure of their invocation, so the seed is rerun on the same seed (FAILURES)."""
    dec = dict(choice=choice, invocation=INVOCATION, at=now_ts())
    done = []
    for k in [k for k, r in st["runs"].items() if k.split("|")[2] == sha and r.get("code_changed")
              and not r.get("decided")]:
        if choice == "keep":
            st["runs"][k]["decided"] = dec
        else:
            sa = st.setdefault("set_aside", {})
            sa[f"{k}|{1 + sum(x.startswith(k + '|') for x in sa)}"] = dict(st["runs"].pop(k), decided=dec)
        for f in (st.get("failures") or {}).get(k, []):
            if f.get("code_changed") and not f.get("decided"):
                f["decided"] = dec
        done.append(k)
    if done:
        save_json(st, store_path(hw))
        log(f"  the user's decision on runs whose code changed while they ran: {choice} {done} (recorded)")
    return done


def _code_changed_or_exit(st, sha, what):
    cc = code_changed_runs(st, sha)
    if cc:
        raise SystemExit(f"{what}: the code on disk changed while these runs ran: {cc}; their records are kept and the "
                         f"screen halts until the user decides. With the code on disk this invocation's again, re-invoke "
                         f"the stage on the card that ran them with --code-changed-decision keep (the records count) or "
                         f"rerun (they are set aside, kept and printed, and their seeds are rerun); the decision is "
                         f"recorded (FAILURES)")


def lock_paths(hw):
    """This card's locks: one in this checkout's OUT and one per machine (MACHINE_LOCK_DIR), so a second checkout on the
    same machine is refused too."""
    return list(dict.fromkeys([os.path.join(OUT, f"{NAME}_{hw_slug(hw)}.lock"),
                               os.path.join(MACHINE_LOCK_DIR, f"{NAME}_{hw_slug(hw)}.lock")]))


@contextlib.contextmanager
def card_lock(hw, what):
    """One S75b invocation at a time per card (FAILURES): exclusive locks (fcntl.flock) on lock_paths(hw), taken before
    the stage reads this card's results file and released when the process ends, however it ends (the operating system
    releases them, so a killed process leaves no stale lock). Each file holds the holder's host, pid and invocation,
    for the refusal; none is committed (OUT/.gitignore)."""
    import fcntl
    import socket
    os.makedirs(OUT, exist_ok=True)
    held = []
    try:
        for p in lock_paths(hw):
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    info = os.read(fd, 4096).decode(errors="replace").strip()
                except OSError:
                    info = "?"
                os.close(fd)
                raise SystemExit(f"{what}: another S75b invocation is running on this card (lock {p}: {info}): one "
                                 f"invocation per card at a time, from any checkout; wait for it to end (FAILURES)")
            held.append(fd)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), invocation=INVOCATION, out=OUT,
                                         at=now_ts())).encode())
        yield lock_paths(hw)[0]
    finally:
        for fd in held:
            try:
                os.ftruncate(fd, 0)
            finally:
                os.close(fd)


def others_running(st, hw, arms):
    """[(results file, last invocation)] of every other card whose last invocation, as its results file in this checkout
    shows it, has not ended and runs any of arms (FAILURES: its seeds would be drawn twice)."""
    own = os.path.basename(store_path(hw))
    return [(f, li) for f, m in sorted(st["meta"].items()) if f != own
            for li in [m.get("last_invocation")] if li and not li.get("ended") and set(li.get("arms") or []) & set(arms)]


def _others_running_or_exit(st, hw, arms, what, other_card_idle):
    busy = others_running(st, hw, arms)
    if busy and not other_card_idle:
        raise SystemExit(
            f"{what}: another card's last invocation, as pushed, has not ended and runs some of these arms: "
            f"{[(f, li.get('invocation'), li.get('arms')) for f, li in busy]}. Wait until it has ended and its results "
            f"file is pushed, then pull; or run only other arms (--arms). If that process was killed: first commit and "
            f"push its results file from its own machine (it holds the runs that finished before the kill), pull, and "
            f"only then say with --other-card-idle that no S75b invocation runs on that card (recorded; its open "
            f"markers then count as killed). If that machine cannot be reached, stop and tell the user: its unpushed "
            f"runs would be drawn again (FAILURES)")
    return busy


def failure_kind(err):
    """'diverged' if a run's error reports a non-finite value (the finiteness check on the evaluation's logits, the
    record's non-finite-loss check, a NaN / inf reaching a statistic), else 'operational'."""
    import re
    return "diverged" if re.search(r"non-finite|\bnan\b|\binf\b|infinity", str(err or ""), re.I) else "operational"


def _effective(f):
    """A failure entry as the rules count it: an entry of a run whose code changed while it ran, which the user decided
    to rerun, counts as an operational failure of its invocation (its recorded kind is kept in the file)."""
    if f.get("code_changed") and (f.get("decided") or {}).get("choice") == "rerun" and f.get("kind") != "operational":
        return dict(f, kind="operational", recorded_kind=f.get("kind"))
    return f


def _fails(st, arm, seed, sha=None):
    """Failure entries of (arm, seed), under sha or under any code (sha None), from every card, as the rules count
    them (_effective)."""
    return [_effective(f) for k, fl in (st.get("failures") or {}).items()
            if k.split("|")[:2] == [arm, str(seed)] and (sha is None or k.split("|")[2] == sha) for f in fl]


def seed_status(st, arm, seed, sha):
    """'diverged' (under any code, on any card: a divergence is the seed's outcome, whatever came after), 'ok' (an ok
    record under sha, any card), 'halted' (operational failures in OPERATIONAL_LIMIT invocations since the user last
    cleared the seed: the driver does not run it again until the user clears it or the code changes), 'failed'
    (operational failures, rerun) or 'missing'. An attempt left open by an earlier invocation (its start marker, no
    record) counts as an operational failure of that invocation."""
    if any(f.get("kind") == "diverged" for f in _fails(st, arm, seed)):
        return "diverged"
    if any(k.split("|")[:3] == [arm, str(seed), sha] and r.get("ok") for k, r in st["runs"].items()):
        return "ok"
    fl = _fails(st, arm, seed, sha)
    inv = counted_invocations(fl)
    if len(inv) >= OPERATIONAL_LIMIT:
        return "halted"
    return "failed" if inv else "missing"


def counted_invocations(fl):
    """The invocations whose operational failures count toward a halt: every invocation with an operational entry (an
    open marker of the current invocation excluded: that attempt is running), less those a user's clear named."""
    cleared = {i for f in fl if f.get("kind") == "cleared" for i in (f.get("clears") or [])}
    return {f.get("invocation") for f in fl if f.get("kind") == "operational"
            and not (f.get("open") and f.get("invocation") == INVOCATION)} - cleared


def statuses_of(st, arm, sha, seeds=None):
    return {s: seed_status(st, arm, s, sha) for s in (seeds or seeds_of(arm))}


def ok_runs(st, arm, sha):
    """runs_of without seeds whose status is diverged (a divergence is the outcome even if a later run completed)."""
    stt = statuses_of(st, arm, sha)
    return {s: r for s, r in runs_of(st, arm, sha).items() if stt.get(s) != "diverged"}


def shas_with(st, arms):
    """The code SHAs under which any of arms has a record or a failure entry."""
    return {k.split("|")[2] for k in list(st["runs"]) + list((st.get("failures") or {})) if k.split("|")[0] in arms}


def _card(k, r):
    return r.get("hw") or k.split("|")[3]


def runs_of(st, arm, sha):
    """{seed: the ok record of arm under sha}. A seed with ok records from two cards is never resolved here: callers
    check duplicates() first (stage 1 / 2 refuse; the report prints them and marks the reading incomplete)."""
    out = {}
    for k in sorted(st["runs"]):
        r = st["runs"][k]
        a, s, sh = k.split("|")[:3]
        if a == arm and r.get("ok") and sh == sha and int(s) not in out:
            out[int(s)] = r
    return out


def duplicates(st, sha, arms=None):
    """[(arm, seed, cards)] for each arm and seed with an outcome under sha from more than one card: an ok record or a
    divergence (a seed drawn on two cards; which draw counts is the user's decision, FAILURES)."""
    seen = {}
    for k, r in st["runs"].items():
        a, s, sh = k.split("|")[:3]
        if sh == sha and r.get("ok") and (arms is None or a in arms):
            seen.setdefault((a, int(s)), set()).add(_card(k, r))
    for k, fl in (st.get("failures") or {}).items():
        a, s, sh, card = k.split("|")[:4]
        if sh == sha and (arms is None or a in arms) and any(_effective(f).get("kind") == "diverged" for f in fl):
            seen.setdefault((a, int(s)), set()).add(card)
    return [(a, s, sorted(h)) for (a, s), h in sorted(seen.items()) if len(h) > 1]


def cards_of(rr):
    return sorted({r.get("hw") for r in rr.values()}, key=str)


# ── the stage-1 rules ─────────────────────────────────────────────────────────
def pilot_decision(pilot, statuses=None):
    """pilot: {seed: ok record}; statuses: {seed: seed_status} (FAILURES). Returns the stage-1 decision (the module
    docstring's rules), all values unrounded. A value STOP is decided as soon as a pilot run ended below 0.9 or
    diverged, whatever the other seed; a halted seed leaves the decision to the user."""
    statuses = statuses or {}
    stt = {s: ("ok" if (pilot.get(s) or {}).get("ok") else statuses.get(s, "missing")) for s in PILOT_SEEDS}
    stt.update({s: "diverged" for s in PILOT_SEEDS if statuses.get(s) == "diverged"})
    sets0 = {s: float(pilot[s]["end"]["set"]) if stt[s] == "ok" else None for s in PILOT_SEEDS}
    div = [s for s in PILOT_SEEDS if stt[s] == "diverged"]
    low = [s for s in PILOT_SEEDS if stt[s] == "ok" and sets0[s] < PROCEED_EACH]
    halted = [s for s in PILOT_SEEDS if stt[s] == "halted"]
    if div or low:
        why = "; ".join(([f"pilot run(s) {div} diverged (a non-finite value): below 0.9"] if div else [])
                        + ([f"pilot run(s) {low} ended below {PROCEED_EACH}: {[sets0[s] for s in low]}"] if low else []))
        return dict(complete=True, proceed=False, statuses=stt, sets=sets0, split_thr=None, diverged=div, halted=halted,
                    not_usable=True, why=why)
    if halted:
        return dict(complete=False, proceed=False, statuses=stt, sets=sets0, halted=halted,
                    why=f"pilot seed(s) {halted} HALTED (operational failures in {OPERATIONAL_LIMIT} invocations): the "
                        f"user decides (clear with --retry-halted, or fix the code: FAILURES)")
    if any(v != "ok" for v in stt.values()):
        return dict(complete=False, proceed=False, statuses=stt, sets=sets0, halted=[],
                    why="a pilot record is missing or failed operationally: rerun --stage 1 (the same seeds)")
    sets = {s: float(v) for s, v in sets0.items()}
    mean = statistics.fmean(sets.values())
    each = all(v >= PROCEED_EACH for v in sets.values())
    proceed = mean >= PROCEED_MEAN and each
    thr = min(sets.values()) - THR_MARGIN
    why = (f"mean {mean:.6f} {'>=' if mean >= PROCEED_MEAN else '<'} {PROCEED_MEAN}; each >= {PROCEED_EACH}: {each}")
    return dict(complete=True, proceed=proceed, statuses=stt, sets=sets, mean=mean, each_ok=each,
                split_thr=thr if proceed else None, diverged=[], halted=[],
                not_usable=any(v < BAR for v in sets.values()), why=why)


def pilot_of(st, sha):
    """The stage-1 decision from a store (load_all() or a committed copy) under sha."""
    return pilot_decision(runs_of(st, "ORACLE_PILOT", sha), statuses_of(st, "ORACLE_PILOT", sha, PILOT_SEEDS))


def stop_words(d):
    """The stop's wording: always that B's pilot did not clear the proceed rule; "S = 4 is not usable at B" only if a pilot
    run ended below 0.9 or diverged."""
    s = "S75b stops: B's pilot did not clear the proceed rule"
    if d.get("diverged"):
        s += f"; pilot run(s) {d['diverged']} diverged (a non-finite value) and so did not reach 0.9"
    if d.get("not_usable"):
        s += ("; a pilot run ended below 0.9" if not d.get("diverged") else "") + ": S = 4 is not usable at B"
    return s


def record_paths(sha=None):
    """The stage-1 record; with sha, its archived copy under that code (FAILURES: a code change between the stages)."""
    tag = "" if sha is None else f"_{sha}"
    return os.path.join(OUT, f"s75b_stage1_record{tag}.json"), os.path.join(OUT, f"s75b_stage1_record{tag}.md")


def stage1_records():
    """[(path, record)] of the stage-1 record and every archived one."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, "s75b_stage1_record*.json"))):
        with open(p) as f:
            out.append((p, json.load(f)))
    return out


def earlier_stops(sha, st=None):
    """[(where, code SHA)] of every value STOP under another code: a stage-1 record whose decision is a complete STOP,
    and any other code's pilot records that decide one (a run below 0.9 or diverged), recomputed from the stores, so a
    STOP never written to a record (the process killed) counts too. A halt is not a STOP."""
    st = st or load_all()
    out = [(os.path.basename(p), r.get("sha")) for p, r in stage1_records()
           if r.get("sha") != sha and (r.get("decision") or {}).get("complete") and not r["decision"].get("proceed")]
    for s_ in sorted(shas_with(st, ("ORACLE_PILOT",)) - {sha}):
        d = pilot_of(st, s_)
        if d["complete"] and not d["proceed"] and s_ not in [x for _, x in out]:
            out.append(("pilot records", s_))
    return out


def write_stage1_record(rec, md_lines):
    """Writes the stage-1 record; a record there under another code SHA is first archived as s75b_stage1_record_<sha>.*"""
    jp, mp_ = record_paths()
    if os.path.exists(jp):
        with open(jp) as f:
            old = json.load(f)
        if old.get("sha") != rec["sha"]:
            ajp, amp = record_paths(old.get("sha"))
            os.replace(jp, ajp)
            if os.path.exists(mp_):
                os.replace(mp_, amp)
    save_json(rec, jp)
    with open(mp_, "w") as f:
        f.write("\n".join(md_lines) + "\n")


# ── git: what is committed ────────────────────────────────────────────────────
def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def _repo_rel(path):
    path = os.path.abspath(path)
    top = _git(["rev-parse", "--show-toplevel"], os.path.dirname(path)).strip()
    return top, os.path.relpath(path, top)


def git_committed(path):
    """(commit, clean) in the repository that contains path: the last commit touching it, and whether the working tree's
    file equals that commit's (tracked, no staged or unstaged change). (None, False) if it was never committed or there
    is no repository."""
    try:
        top, rel = _repo_rel(path)
        c = _git(["log", "-1", "--format=%H", "--", rel], top).strip()
        s = _git(["status", "--porcelain", "--", rel], top).strip()
        return (c or None), bool(c) and not s
    except Exception:
        return None, False


def git_committed_json(path):
    """path's JSON as of the last commit touching it (None if never committed)."""
    c, _ = git_committed(path)
    if c is None:
        return None
    try:
        top, rel = _repo_rel(path)
        return json.loads(_git(["show", f"{c}:{rel}"], top))
    except Exception:
        return None


def dirty_code():
    """The source files of the modules in this driver's code SHA that differ from the last commit (or are untracked)."""
    _, names = code_sha()
    files = [module_file(n) for n in names]
    try:
        out = _git(["status", "--porcelain", "--", *[os.path.relpath(f, HERE) for f in files]], HERE)
    except Exception as e:
        return [f"git failed: {e}"]
    return [line[3:] for line in out.splitlines() if line.strip()]


def code_or_exit(what, require_commit=True):
    if require_commit:
        d = dirty_code()
        if d:
            raise SystemExit(f"{what}: the driver's code is not committed ({d}): run from a clean checkout")


def committed_store():
    """The S75b state as committed: every s75b_*_results.json in OUT read at its last commit, merged as load_all does."""
    import glob
    out = {"meta": {}, "runs": {}, "failures": {}}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        st = git_committed_json(p)
        if st:
            out["meta"][os.path.basename(p)] = st.get("meta", {})
            out["runs"].update(st.get("runs", {}))
            out["failures"].update(st.get("failures") or {})
    return out


def superseded(sha, st=None):
    """A code whose stage-1 record was archived when a later code's stage 1 started (the record is written before the
    pilot's first run), or whose pilot entries exist while the stage-1 record is under another code: never used again
    (FAILURES)."""
    if os.path.exists(record_paths(sha)[0]):
        return True
    jp = record_paths()[0]
    if not os.path.exists(jp):
        return False
    with open(jp) as f:
        cur = json.load(f).get("sha")
    return cur != sha and sha in shas_with(st or load_all(), ("ORACLE_PILOT",))


def committed_or_exit(path, what, require_commit=True):
    c, clean = git_committed(path)
    if require_commit and (c is None or not clean):
        raise SystemExit(f"{what} ({path}) is not committed, or has local changes: commit and push it first")
    return c


# ── the benchmark and the run-mechanics rules ─────────────────────────────────
def bench_path(hw):
    return os.path.join(OUT, f"{NAME}_bench_{hw_slug(hw)}.json")


def checks_output_path(gpu):
    return os.path.join(CHECKS_DIR, f"s75b_checks_cuda_{hw_slug(gpu)}_output.json")


def checks_status(gpu, sha):
    """None if this card's CUDA CHECK output exists, was written under sha, holds B1-B9 (B7 and B9 run, not skipped) and
    every CHECK passed; otherwise why not."""
    p = checks_output_path(gpu)
    if not os.path.exists(p):
        return "missing"
    with open(p) as f:
        d = json.load(f)
    ch = d.get("checks") or {}
    if (d.get("meta") or {}).get("s75b_sha") != sha:
        return f"written under code SHA {(d.get('meta') or {}).get('s75b_sha')}, not {sha}"
    keys = [f"B{i}" for i in range(1, 10)]
    if any(k not in ch for k in keys):
        return f"incomplete: {[k for k in keys if k not in ch]} missing"
    bad = [k for k in keys if not ch[k].get("ok")]
    if bad:
        return f"failed: {bad}"
    if ch["B7"].get("skipped") or ch["B9"].get("skipped"):
        return "B7 or B9 skipped"
    return None


def held_evals(w, n_evals, gap_secs, stats_secs):
    """The evaluations whose tensors the main process holds at the worst point of a run with w statistics workers: those
    in the workers and those queued while the workers lag (one job per evaluation at R = 1, one every gap_secs, each
    taking stats_secs)."""
    if w <= 0:
        return 0
    lag = max(0.0, 1.0 - w * gap_secs / stats_secs) if stats_secs and gap_secs else 0.0
    return min(n_evals, w + math.ceil(n_evals * lag))


def ram_budget(host_ram_gb, mem, beside):
    """(budget GB, which bound): the smaller of (min(the user's figure, MemTotal) - the co-running job's peak -
    OS_RESERVE_GB) and (MemAvailable at the benchmark's start - the co-running job's regrowth to its peak -
    AVAIL_RESERVE_GB)."""
    total = min(host_ram_gb, mem["MemTotal"]) if mem.get("MemTotal") else host_ram_gb
    b_total = total - beside["hwm_gb"] - OS_RESERVE_GB
    if mem.get("MemAvailable") is None:
        return b_total, "total (MemAvailable not readable)"
    b_avail = mem["MemAvailable"] - max(0.0, beside["hwm_gb"] - beside["rss_gb"]) - AVAIL_RESERVE_GB
    return (b_total, "total") if b_total <= b_avail else (b_avail, "available")


def workers_rule(budget_gb, main_gb, data_gb, worker_gb, payload, gap_secs, stats_secs, n_evals):
    """--stats-workers (this driver's formula; CHANGES_SINCE_SPEC). need(w) = main_gb (the benchmark's main-process
    peak, one evaluation's tensors included) + data_gb + w * worker_gb + (held_evals(w) - 1) * payload for w >= 1;
    need(0) = main_gb + data_gb + worker_gb (the statistics in line). Workers: the smallest w in 1..MAX_WORKERS with
    w >= needed (= ceil(stats_secs / gap_secs): no evaluation waits) whose need fits the budget; else the largest w >= 1
    that fits; else 0 if need(0) fits; else None (refuse). Returns (w, needed, need, why)."""
    if None in (budget_gb, main_gb, worker_gb) or not gap_secs or stats_secs is None:
        return None, None, {}, "not measured"
    needed = max(1, math.ceil(stats_secs / gap_secs))
    need = {0: main_gb + data_gb + worker_gb}
    for w in range(1, MAX_WORKERS + 1):
        need[w] = main_gb + data_gb + w * worker_gb + max(0, held_evals(w, n_evals, gap_secs, stats_secs) - 1) * payload
    fits = [w for w in range(1, MAX_WORKERS + 1) if need[w] <= budget_gb]
    keep_up = [w for w in fits if w >= needed]
    if keep_up:
        return min(keep_up), needed, need, "keeps up"
    if fits:
        return max(fits), needed, need, "falls behind (the evaluations held are counted)"
    if need[0] <= budget_gb:
        return 0, needed, need, "in line (no worker fits)"
    return None, needed, need, "refuse: not even the statistics in line fit the budget"


def register_rule(bench):
    s = bench["arms"].get("REG3 scan") or {}
    if not s.get("ok") or s.get("peak_reserved_gb") is None:
        return "loop"
    return "scan" if s["peak_reserved_gb"] + bench["context_gb"] < bench["total_gb"] else "loop"


def bench(device, host_ram_gb, beside="none", iters=12, every=3, log=print, path=None):
    """Each S75b arm at R = 1 on this card for `iters` updates with an evaluation every `every` (the device part here, the
    statistics in one StatsPool worker, waited for at each evaluation) and, for WIDE_SPLIT_G, the guarded split checks
    (the trigger's schedule scaled to `every`, threshold 1.01, rise 1.0); REG3 in both register modes. Reads peak device
    memory, the host memory of this process, the statistics worker and the data process (VmHWM), the co-running job's
    (beside), MemTotal / MemAvailable, the seconds per update and per evaluation's statistics; applies register_rule and
    workers_rule. With path, the file is written before the first case (status "in progress") and after every case, so a
    benchmark that crashes leaves a file that counts as failed (FAILURES)."""
    dev = dv.setup(device)
    mem0 = meminfo_gb()
    if beside == "none":
        bs = dict(name="none", roots=[], pids=[], rss_gb=0.0, hwm_gb=0.0)
    else:
        bs = beside_mem(beside)
        if not bs["pids"]:
            raise SystemExit(f"--beside {beside!r}: no running process has it in its command line (pass --beside none "
                             f"if nothing runs beside S75b)")
    T = task()
    out = dict(device=dv.device_info(dev), host_ram_gb=host_ram_gb, beside=bs, meminfo_start=mem0, iters=iters,
               eval_every=every, arms={}, status="in progress", all_ok=False, sha=code_sha()[0],
               provenance=SPEC_PROVENANCE, changes=CHANGES_SINCE_SPEC, written=utc_now())
    if path:
        archive_bench(path)
        save_json(out, path)
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST)
    cases = [("ORACLE", "ORACLE", "auto"), ("SINGLE", "SINGLE", "auto"), ("WIDE_SLOW", "WIDE_SLOW", "auto"),
             ("WIDE_SPLIT_G", "WIDE_SPLIT_G", "auto"), ("REG3 scan", "REG3", "scan"), ("REG3 loop", "REG3", "loop")]
    StatsPool.sync = True
    try:
        with runtime():
            for name, arm, reg in cases:
                _JOBS.clear()
                kids = {}

                def sample(bm, views, step):
                    for p, v in children_mem().items():
                        k = kids.setdefault(p, dict(v))
                        k["hwm_gb"] = max(k["hwm_gb"], v["hwm_gb"])
                ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = every, every, 2 * every
                try:
                    with guarded(rise=1.0) if arm == "WIDE_SPLIT_G" else contextlib.nullcontext():
                        t0 = time.time()
                        recs = gr.run_batched_icmc(T, arm_spec(arm), [PILOT_SEEDS[0]], iters, device=dev, split_thr=1.01,
                                                   eval_every=every, register=reg, stats_workers=1, prefetch=True,
                                                   on_eval_extra=sample)
                    r = recs[0]
                    cv = r.get("curve") or []
                    wp = {j.get("worker_pid") for j in _JOBS}
                    data = [v["hwm_gb"] for p, v in kids.items() if p not in wp and "resource_tracker" not in v["cmd"]]
                    # seconds per update between the first and the last evaluation (the first update's warm-up left
                    # out) when that is >= 1 s of training (train_secs is rounded to 0.1 s); else over the whole run
                    dt = cv[-1]["train_secs"] - cv[0]["train_secs"] if len(cv) >= 2 else 0.0
                    spu = (dt / (cv[-1]["step"] - cv[0]["step"]) if dt >= 1.0 else
                           (r["train_secs"] / iters if r.get("train_secs") else None))
                    row = dict(ok=bool(r.get("ok")), error=r.get("error"), secs=time.time() - t0, register=r.get("register"),
                               secs_per_update=spu, secs_per_update_from="evaluations" if dt >= 1.0 else "whole run",
                               worker_gb=max((j["worker_gb"] for j in _JOBS if j.get("worker_gb")), default=None),
                               worker_ru_maxrss_gb=max((j["worker_ru_maxrss_gb"] for j in _JOBS
                                                        if j.get("worker_ru_maxrss_gb")), default=None),
                               stats_secs=max((j["worker_secs"] for j in _JOBS), default=None),
                               payload_gb=max((j["payload_gb"] for j in _JOBS), default=None),
                               data_gb=max(data) if data else None, main_peak_gb=own_peak_gb(),
                               checks=[(c["step"], c.get("eligible"), c.get("blocked_by"), c["fired"], c.get("map_one_to_one"))
                                       for c in r.get("checks", [])])
                    if dev.type == "cuda":
                        free, total = torch.cuda.mem_get_info(dev)
                        row.update(peak_alloc_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                                   peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2 ** 30,
                                   device_used_gb=(total - free) / 2 ** 30,
                                   reserved_now_gb=torch.cuda.memory_reserved(dev) / 2 ** 30)
                except Exception as e:
                    row = dict(ok=False, error=f"{type(e).__name__}: {e}")
                finally:
                    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = sched
                    if dev.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception as e:                  # a sticky CUDA error: the case fails, the file stays
                            row = dict(row, ok=False, error=f"{row.get('error')}; empty_cache: {type(e).__name__}: {e}")
                out["arms"][name] = row
                if path:
                    save_json(out, path)
                log(f"  bench {name}: {json.dumps(row, default=str)[:400]}")
    finally:
        StatsPool.sync = False
    out["meminfo_end"] = meminfo_gb()
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        out["total_gb"] = total / 2 ** 30
        ctx = [r["device_used_gb"] - r["reserved_now_gb"] for r in out["arms"].values() if "device_used_gb" in r]
        out["context_gb"] = max(ctx) if ctx else 0.0
        out["register_reg3"] = register_rule(out)
    else:
        out["register_reg3"] = "loop"
    unused = "REG3 loop" if out["register_reg3"] == "scan" else "REG3 scan"   # a failed scan (out of memory) -> loop
    out["cases_used"] = [n for n in out["arms"] if n != unused]
    used = [out["arms"][n] for n in out["cases_used"] if out["arms"][n].get("ok")]

    def mx(k):
        v = [r[k] for r in used if r.get(k) is not None]
        return max(v) if v else None
    spu = [r["secs_per_update"] for r in used if r.get("secs_per_update")]
    out["main_gb"] = own_peak_gb()
    out["main_ru_maxrss_gb"] = ru_maxrss_gb()
    out["worker_gb"], out["payload_gb"], out["stats_secs"] = mx("worker_gb"), mx("payload_gb"), mx("stats_secs")
    out["data_gb"] = mx("data_gb")
    out["data_gb_source"] = "measured" if out["data_gb"] is not None else f"DATA_PROC_GB = {DATA_PROC_GB} (not measured)"
    out["secs_per_update"] = min(spu) if spu else None          # the shortest interval between evaluations: the most lag
    out["n_evals"] = ITERS // EVAL_EVERY
    out["gap_secs"] = EVAL_EVERY * out["secs_per_update"] if out["secs_per_update"] else None
    out["budget_gb"], out["budget_bound"] = ram_budget(host_ram_gb, mem0, bs)
    w, needed, need, why = workers_rule(out["budget_gb"], out["main_gb"],
                                        out["data_gb"] if out["data_gb"] is not None else DATA_PROC_GB,
                                        out["worker_gb"], out["payload_gb"] or 0.0, out["gap_secs"], out["stats_secs"],
                                        out["n_evals"])
    out.update(stats_workers=w, workers_needed=needed, need_gb=need, workers_why=why,
               held_evals=held_evals(w, out["n_evals"], out["gap_secs"], out["stats_secs"]) if w else 0,
               all_ok=all(out["arms"][n].get("ok") for n in out["cases_used"]), status="done")
    if path:
        save_json(out, path)
    return out


def archive_bench(path):
    """An earlier benchmark file is kept: moved to OUT/bench_archive/ under its 'written' time (FAILURES)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            old = json.load(f)
    except Exception:
        old = {}
    arch = os.path.join(os.path.dirname(path), "bench_archive")
    os.makedirs(arch, exist_ok=True)
    stem = os.path.join(arch, os.path.basename(path)[:-5] + "_" + hw_slug(str(old.get("written") or time.time())))
    dst, i = stem + ".json", 1
    while os.path.exists(dst):                     # never overwrite an earlier archive (two in one second)
        i += 1
        dst = f"{stem}_{i}.json"
    os.replace(path, dst)
    return dst


def load_bench(hw, sha):
    p = bench_path(hw)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        b = json.load(f)
    return b if b.get("sha") == sha else None


def bench_cases(arms, register):
    """The benchmark cases the arms use (REG3 in the register mode the benchmark chose)."""
    m = {"ORACLE_PILOT": "ORACLE", "ORACLE": "ORACLE", "SINGLE": "SINGLE", "WIDE_SLOW": "WIDE_SLOW",
         "WIDE_SPLIT_G": "WIDE_SPLIT_G", "REG3": f"REG3 {register}"}
    return sorted({m[a] for a in arms})


def bench_why(b, arms):
    """None if the benchmark makes this card usable for the arms (FAILURES: it finished; every case the arms use passed;
    the host-memory rule gave workers), else why not."""
    if not b:
        return "no benchmark"
    if b.get("status") != "done":
        return f"the benchmark did not finish (status {b.get('status')!r}): it counts as failed"
    bad = [c for c in bench_cases(arms, b.get("register_reg3") or "loop") if not (b["arms"].get(c) or {}).get("ok")]
    if bad:
        return f"failed cases {bad}"
    if b.get("stats_workers") is None:
        return f"the host-memory rule refuses ({b.get('workers_why')}; budget {b.get('budget_gb')} GB, need {b.get('need_gb')})"
    return None


def bench_usable(b, arms=STAGE2):
    return bench_why(b, arms) is None


def benches(sha):
    """{card: (path, benchmark)} for every benchmark file under sha."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_bench_*.json"))):
        with open(p) as f:
            b = json.load(f)
        if b.get("sha") == sha:
            dvc = b.get("device") or {}
            out[dvc.get("gpu") or dvc.get("cpu")] = (p, b)
    return out


def card_setup(dev, hw, sha, require_commit, what, arms):
    """This card's run mechanics from its latest benchmark under this code (CUDA: committed; and the CUDA CHECK output
    committed and passing). Usability is per arm (FAILURES): an arm whose cases the benchmark did not pass is refused
    and named, the others run; if none of the arms is allowed, the stage refuses. CPU: every arm, the statistics in
    line, REG3 with the loop. Returns (workers, register, bench_commit, checks_commit, allowed arms, {refused arm: why},
    the benchmark's 'written')."""
    if dev.type != "cuda":
        return 0, "loop", None, None, tuple(arms), {}, None
    b = load_bench(hw, sha)
    if b is None:
        raise SystemExit(f"{what}: no benchmark for {hw} under code SHA {sha}: run --bench on this card and commit it")
    refused = {a: bench_why(b, (a,)) for a in arms if bench_why(b, (a,))}
    allowed = tuple(a for a in arms if a not in refused)
    if not allowed:
        raise SystemExit(f"{what}: this card's benchmark allows none of {sorted(arms)}: {refused} (FAILURES: "
                         + ("run --stage 1 on the other card, or rerun --bench on this one)" if what == "stage 1" else
                            "the other card runs them, --machine fallback / fallback-3070 with --reason and --arms)"))
    bc = committed_or_exit(bench_path(hw), f"{what}: this card's benchmark", require_commit)
    why = checks_status(hw, sha)
    if why is not None:
        raise SystemExit(f"{what}: this card's CUDA CHECK output ({checks_output_path(hw)}) is {why}: run python -m "
                         f"explore_gpu.s75b_checks --device cuda and commit its output")
    cc = committed_or_exit(checks_output_path(hw), f"{what}: this card's CUDA CHECK output", require_commit)
    return b["stats_workers"], b["register_reg3"], bc, cc, allowed, refused, b.get("written")


def sync_state():
    """(ok, why): with commits required, the repository holding OUT must be up to date with its upstream (after git
    fetch) and every S75b JSON file in OUT committed, so the finality rules see both machines' state (FAILURES)."""
    try:
        os.makedirs(OUT, exist_ok=True)
        top, rel = _repo_rel(os.path.join(OUT, "x"))
    except Exception as e:
        return False, f"OUT ({OUT}) is not in a git repository ({type(e).__name__})"
    try:
        r = subprocess.run(["git", "fetch", "--quiet"], cwd=top, capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    except subprocess.TimeoutExpired:
        return False, "git fetch timed out: the other machine's state cannot be seen"
    if r.returncode != 0:
        return False, (f"git fetch failed (rc {r.returncode}: {r.stderr.strip()[:160]}): the other machine's state "
                       f"cannot be seen")
    try:
        up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], top).strip()
        behind = int(_git(["rev-list", "--count", "HEAD..@{u}"], top).strip() or 0)
        ahead = int(_git(["rev-list", "--count", "@{u}..HEAD"], top).strip() or 0)
    except Exception as e:
        return False, f"no upstream branch to compare with ({type(e).__name__}): push the branch and set its upstream"
    if behind:
        return False, f"the checkout is {behind} commits behind {up}: pull first"
    if ahead:
        return False, f"the checkout has {ahead} unpushed commits: push first, so the other machine sees them"
    try:
        dirty = [l[3:] for l in _git(["status", "--porcelain", "--untracked-files=all", "--", os.path.dirname(rel)],
                                     top).splitlines() if l.strip().endswith(".json")]
    except Exception as e:
        return False, f"git status failed ({type(e).__name__})"
    if dirty:
        return False, f"uncommitted S75b state {dirty}: commit and push it first"
    return True, ""


def sync_or_exit(what, require_commit=True):
    if require_commit:
        ok, why = sync_state()
        if not ok:
            raise SystemExit(f"{what}: {why}")


# ── running arms ──────────────────────────────────────────────────────────────
def run_arm(arm, seeds, dev, hw, sha, st, meta_extra, register, workers, split_thr=None, log=print):
    """Each seed at R = 1 (FAILURES in the module docstring). A seed that is ok, diverged or halted (on any card) is not
    run. Before each attempt an open marker is saved in st["failures"][key]; it is replaced by the attempt's failure
    entry (attempt, invocation, kind, error) or removed when the run completes, so a process that dies leaves an
    operational failure. A divergence is never retried. An operational failure is retried once; if it survives the
    retry, InvocationAborted stops the invocation, so only this seed is charged. Before each attempt the code on disk is
    compared with sha (code_now): if it changed, CodeChanged stops the invocation and nothing is charged; if it changed
    during the attempt, the record is kept with code_changed and CodeChanged stops the invocation (the user decides).
    Records keyed arm|seed|sha|hw, with the host memory of the run (descriptions). Returns the seeds without an ok
    record."""
    T = task()
    missing = []
    for s in seeds:
        key = f"{arm}|{s}|{sha}|{hw}"
        if (st["runs"].get(key) or {}).get("ok"):
            continue
        status = seed_status(load_all(), arm, s, sha)
        if status in ("ok", "diverged", "halted"):
            log(f"  {arm} seed {s}: {status} (not run)")
            if status != "ok":
                missing.append(s)
            continue
        kind = None
        for attempt in (1, 2):
            now = code_now(sha)
            if now != sha:
                raise CodeChanged(f"the code on disk is {now}, not this invocation's {sha} (a pull or an edit while the "
                                  f"stage ran): the invocation stops before {arm} seed {s}, which is not charged; a code "
                                  f"change halts the screen (FAILURES)")
            log(f"  {arm} seed {s} (attempt {attempt}) ...")
            fl = st.setdefault("failures", {}).setdefault(key, [])
            mark = dict(attempt=attempt, invocation=INVOCATION, kind="operational", open=True, error=OPEN_ATTEMPT,
                        at=now_ts())
            fl.append(mark)
            save_json(st, store_path(hw))
            _JOBS.clear()
            ctx = guarded() if arm == "WIDE_SPLIT_G" else contextlib.nullcontext()
            with ctx:
                rec = gr.run_batched_icmc(T, arm_spec(arm), [s], ITERS, device=dev, split_thr=split_thr, eval_every=EVAL_EVERY,
                                          register=register if arm == "REG3" else "auto", stats_workers=workers, tag=arm)[0]
            rec.update(arm=arm, sha=sha, hw=hw, attempt=attempt, threads=torch.get_num_threads(),
                       host=dict(main_peak_gb=own_peak_gb(), meminfo=meminfo_gb(),
                                 worker_peak_gb=max((j["worker_gb"] for j in _JOBS if j.get("worker_gb")), default=None),
                                 stats_secs_max=max((j["worker_secs"] for j in _JOBS), default=None)), **meta_extra)
            now = code_now(sha)
            changed = None if now == sha else dict(sha_on_disk=now, at=now_ts())
            if changed:
                rec["code_changed"] = changed
            fl.remove(mark)
            if not fl:
                del st["failures"][key]
            st["runs"][key] = rec
            stop = CodeChanged(f"the code on disk changed to {now} while {arm} seed {s} ran under {sha}: its record is kept "
                               f"(code_changed) and the screen halts: the user decides (FAILURES)") if changed else None
            if rec.get("ok"):
                save_json(st, store_path(hw))
                e = rec["end"]
                log(f"    {arm} {s}: SET {e['set']:.4f} ROUTED {idense.routed(rec)}"
                    + (f" splits {rec.get('splits')}" if "splits" in rec else ""))
                if stop:
                    raise stop
                break
            kind = failure_kind(rec.get("error"))
            st["failures"].setdefault(key, []).append(dict(
                attempt=attempt, invocation=INVOCATION, kind=kind, error=str(rec.get("error"))[:600],
                at=now_ts(), **({"code_changed": changed} if changed else {})))
            save_json(st, store_path(hw))
            log(f"    {arm} {s}: FAILED ({kind}) {rec.get('error')}")
            if stop:
                raise stop
            if kind == "diverged":                         # an outcome: never retried
                break
        if not (st["runs"].get(key) or {}).get("ok"):
            missing.append(s)
            if kind == "operational":
                raise InvocationAborted(f"{arm} seed {s} failed operationally twice in this invocation "
                                        f"({str(rec.get('error'))[:200]}): the invocation stops (FAILURES); re-invoke "
                                        f"the stage after checking the card")
    return missing


def _dup_exit(dup, what):
    if dup:
        raise SystemExit(f"{what}: ok records of one arm and seed from two cards (arm, seed, cards): {dup}; the driver "
                         f"does not choose between them: the user decides")


def _clear_halted(st, hw, arm, seeds, sha, log):
    """--retry-halted: the user clears halted seeds (a 'cleared' entry with its time and invocation, kept and printed)."""
    stt = load_all()
    for s in seeds:
        if seed_status(stt, arm, s, sha) == "halted":
            clears = sorted(counted_invocations(_fails(stt, arm, s, sha)))
            st.setdefault("failures", {}).setdefault(f"{arm}|{s}|{sha}|{hw}", []).append(dict(
                kind="cleared", invocation=INVOCATION, at=now_ts(), clears=clears,
                error=f"cleared by the user (--retry-halted): invocations {clears}"))
            log(f"  {arm} seed {s}: halt cleared by the user (--retry-halted)")
    save_json(st, store_path(hw))


def _frozen_or_exit(st, sha, what):
    """After stage 2 has started under one code, no stage runs under another (FAILURES: a code change halts the screen
    for the user; no outcome is re-drawn)."""
    other = sorted(shas_with(st, STAGE2) - {sha})
    if other:
        raise SystemExit(f"{what}: stage 2 has records or failures under code SHA {other}; the code is frozen once stage 2 "
                         f"starts: a code change halts the screen and the user decides (FAILURES)")


def _pilot_files(sha):
    """The results files (relative paths) that hold an ORACLE_PILOT record or failure entry under sha."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        keys = list(st.get("runs", {})) + list(st.get("failures") or {})
        if any(k.split("|")[0] == "ORACLE_PILOT" and k.split("|")[2] == sha for k in keys):
            out.append(os.path.relpath(p, HERE))
    return out


def stage1(device, log=print, require_commit=True, retry_halted=False, code_change_note=None, other_card_idle=False,
           code_changed_decision=None):
    global INVOCATION
    INVOCATION = f"stage 1 {utc_now()} pid {os.getpid()}"
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    if dev.type == "cpu" and require_commit:
        raise SystemExit("stage 1: --device cpu is for the CHECKs only")
    with card_lock(hw, "stage 1"):
        return _stage1(dev, info, hw, log, require_commit, retry_halted, code_change_note, other_card_idle,
                       code_changed_decision)


def _stage1(dev, info, hw, log, require_commit, retry_halted, code_change_note, other_card_idle, code_changed_decision):
    sha = code_sha()[0]
    sync_or_exit("stage 1", require_commit)
    st_all = load_all()
    if superseded(sha, st_all):
        raise SystemExit(f"stage 1: code SHA {sha} was superseded (a later code's stage 1 started after it): it is never "
                         f"used again (FAILURES)")
    if code_changed_decision:
        decide_code_changed(load(hw), hw, sha, code_changed_decision, log)
        st_all = load_all()
    _code_changed_or_exit(st_all, sha, "stage 1")
    stops = earlier_stops(sha, st_all)
    if stops:
        raise SystemExit(f"stage 1: S75b stopped at stage 1 under code SHA {[x for _, x in stops]} ({[w for w, _ in stops]}); "
                         f"a value STOP under any code is final (FAILURES)")
    _frozen_or_exit(st_all, sha, "stage 1")
    _dup_exit(duplicates(st_all, sha, ("ORACLE_PILOT",)), "stage 1")
    idle = _others_running_or_exit(st_all, hw, ("ORACLE_PILOT",), "stage 1", other_card_idle)
    jp, _ = record_paths()
    prev = json.load(open(jp)) if os.path.exists(jp) else None
    own = bool(prev) and prev.get("sha") == sha
    others = sorted((shas_with(st_all, ("ORACLE_PILOT",)) | {r.get("sha") for _, r in stage1_records()}) - {sha, None})
    note = (prev or {}).get("code_change_note") if own else None
    if others and not own:
        if not code_change_note or not os.path.exists(code_change_note):
            raise SystemExit(f"stage 1: another code {others} has a stage-1 record or pilot entries; a code change before "
                             f"stage 2 needs --code-change-note PATH, a committed, dated note saying why the change could "
                             f"not be avoided (FAILURES)")
        nc = committed_or_exit(code_change_note, "stage 1: the code-change note", require_commit)
        note = dict(path=os.path.relpath(os.path.abspath(code_change_note), HERE), commit=nc,
                    from_sha=(prev or {}).get("sha"), other_codes=others)
    code_or_exit("stage 1", require_commit)
    workers, _, bench_commit, checks_commit, _, _, bench_written = card_setup(dev, hw, sha, require_commit, "stage 1",
                                                                             ("ORACLE_PILOT",))
    T = task()
    chk = it.check_task(T)
    assert all(chk.values()), f"task CHECK failed: {chk}"
    st = load(hw)
    if retry_halted:
        _clear_halted(st, hw, "ORACLE_PILOT", PILOT_SEEDS, sha, log)
    inv = dict(invocation=INVOCATION, sha=sha, card=hw, workers=workers, threads=torch.get_num_threads(),
               bench_commit=bench_commit, bench_written=bench_written, checks_commit=checks_commit,
               retry_halted=retry_halted, other_card_idle=bool(other_card_idle),
               other_cards_not_ended=[(f, li.get("invocation")) for f, li in idle], started=utc_now())
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, provenance=SPEC_PROVENANCE,
                      changes=CHANGES_SINCE_SPEC, stage1=inv,
                      last_invocation=dict(invocation=INVOCATION, stage=1, arms=["ORACLE_PILOT"], ended=False,
                                           started=inv["started"]))
    st["meta"].setdefault("stage1_invocations", []).append(inv)
    save_json(st, store_path(hw))
    # the record is written before the first run, so a later code's stage 1 supersedes this one even if it is killed
    write_stage1_record(*_stage1_record(sha, note, running=INVOCATION))
    log(f"[S75b stage 1] {T.label}; code SHA {sha}; {info}; task CHECK ok; stats workers {workers}. Commit and push "
        f"the results file and this running stage-1 record now (git add explore_out/I/gpu/s75b), so the other card "
        f"sees the pilot running; push again when it ends. Do not pull or edit code until this invocation ends.")
    aborted = None
    try:
        with runtime():
            run_arm("ORACLE_PILOT", PILOT_SEEDS, dev, hw, sha, st, dict(stage=1, workers=workers, bench_commit=bench_commit,
                                                                      bench_written=bench_written), "auto", workers, log=log)
    except InvocationAborted as e:
        aborted = str(e)
        log(f"[S75b stage 1] {aborted}")
    except BaseException as e:                    # interrupted: the invocation ends; its record says so
        aborted = f"interrupted ({type(e).__name__}: {str(e)[:120]})"
        raise
    finally:
        statuses = statuses_of(load_all(), "ORACLE_PILOT", sha, PILOT_SEEDS)
        inv.update(aborted=aborted, statuses_after=statuses, ended=utc_now())
        st["meta"]["last_invocation"].update(ended=True, at=inv["ended"], aborted=aborted)
        save_json(st, store_path(hw))
        rec, lines = _stage1_record(sha, note, running=None)
        write_stage1_record(rec, lines)
    log("\n".join(lines))
    return rec["decision"], rec["missing"]


def _stage1_record(sha, note, running=None):
    """The stage-1 record from the state on disk: the decision from every card's pilot records under sha, each seed's
    card, the failure log, every stage-1 invocation under sha (from every results file's meta) and the code-change
    note. running: the invocation that is still running (the record written before the first run)."""
    st_all = load_all()
    pilot = runs_of(st_all, "ORACLE_PILOT", sha)
    d = pilot_of(st_all, sha)
    missing = [s for s, v in d["statuses"].items() if v != "ok"]
    cards = {s: r.get("hw") for s, r in pilot.items()}
    fails = {k: v for k, v in st_all["failures"].items() if k.startswith("ORACLE_PILOT|")}
    files = _pilot_files(sha)
    invs = sorted((i for m in st_all["meta"].values() for i in (m.get("stage1_invocations") or [])
                   if i.get("sha") == sha), key=lambda i: str(i.get("invocation")))
    rec = dict(written=utc_now(), sha=sha, running=running, machine=sorted(set(cards.values())), cards=cards,
               config=CONFIG, iters=ITERS, pilot_seeds=list(PILOT_SEEDS), decision=d, missing=missing, failures=fails,
               results_files=files, invocations=invs, code_change_note=note, provenance=SPEC_PROVENANCE,
               changes=CHANGES_SINCE_SPEC,
               note=(f"Commit and push this record together with every results file {files}, whatever the decision: the "
                     f"finality rules on both machines read the committed state, and --stage 2 refuses unless every "
                     f"results file is committed and the committed pilot records give this decision. Each pilot seed's "
                     f"card: {cards} (the spec at {SPEC_COMMIT} assigned the GTX 1660 SUPER; the user allowed the RTX 3070 "
                     f"if it was free first). Every stage-1 invocation under this code, with its card, benchmark and "
                     f"workers, is listed."))
    sets = d.get("sets") or {}
    head = (f"RUNNING: written when stage-1 invocation {running} started, before its first run; --stage 2 refuses "
            f"until a stage 1 has ended and rewritten it" if running else None)
    lines = [f"# S75b stage-1 record (EXPLORATORY), {rec['written']}", ""] + ([f"**{head}**", ""] if head else []) + [
             f"- Pilot: ORACLE_PILOT, seeds {PILOT_SEEDS}, {ITERS:,} updates, code SHA {sha}; cards per seed {cards}; "
             f"results files {files}.",
             "- Invocations: " + "; ".join(f"{i['invocation']} on {i['card']} (workers {i['workers']}, threads "
                                           f"{i['threads']}, benchmark {i['bench_commit']} written {i['bench_written']}, "
                                           f"CUDA CHECK {i['checks_commit']}"
                                           + (", --retry-halted" if i.get("retry_halted") else "")
                                           + (", --other-card-idle" if i.get("other_card_idle") else "")
                                           + (f"; stopped: {i['aborted'][:120]}" if i.get("aborted") else "")
                                           + ("" if i.get("ended") else "; not ended") + ")"
                                           for i in invs),
             "- end.set: " + ", ".join(f"{s}: {v}" for s, v in sets.items()),
             "- Seed status (FAILURES): " + ", ".join(f"{s}: {v}" for s, v in (d.get("statuses") or {}).items())
             + (f"; failure log: {[(k, [(f.get('kind'), f.get('invocation'), str(f.get('error'))[:80]) for f in v]) for k, v in fails.items()]}"
                if fails else ""),
             f"- Proceed rule (mean >= {PROCEED_MEAN} and each >= {PROCEED_EACH}): {d.get('why')} -> "
             f"**{'PROCEED' if d['proceed'] else ('STOP' if d['complete'] else ('HALTED: the user decides' if d.get('halted') else 'INCOMPLETE'))}**"]
    if note:
        lines.append(f"- Code change before stage 2 from {note.get('from_sha')} (other codes {note.get('other_codes')}): "
                     f"note {note.get('path')} (commit {note.get('commit')}).")
    if d["proceed"]:
        lines.append(f"- SPLIT_THR = min(end.set) - {THR_MARGIN} = {d['split_thr']!r}")
    elif d["complete"]:
        lines.append(f"- {stop_words(d)}. No other configuration.")
    lines += ["", rec["note"], "", "Provenance as git shows it (SPEC_PROVENANCE):"]
    lines += [f"- built on a tree without the learned arms (git); blind on the author's statement only: {x}"
              for x in SPEC_PROVENANCE["blind_by_git"]]
    lines += [f"- NOT BLIND BY GIT: {x}" for x in SPEC_PROVENANCE["not_blind_by_git"]]
    lines += ["", f"Changes since the approved spec ({SPEC_COMMIT}), not blind ({CHANGES_SINCE_SPEC['note']}):"]
    lines += [f"- {x}" for x in CHANGES_SINCE_SPEC["items"]]
    return rec, lines


def stage2(device, machine, log=print, require_commit=True, reason=None, retry_halted=False, arms=None,
           other_card_idle=False, code_changed_decision=None):
    global INVOCATION
    INVOCATION = f"stage 2 {utc_now()} pid {os.getpid()}"
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    if dev.type == "cuda" and MACHINE_GPU[machine] not in hw:
        raise SystemExit(f"--machine {machine} expects a {MACHINE_GPU[machine]}, this card is {hw}")
    if dev.type == "cpu" and require_commit:
        raise SystemExit("stage 2: --device cpu is for the CHECKs only")
    if machine.startswith("fallback") and not reason:
        raise SystemExit(f"--machine {machine} needs --reason (why the other card cannot run its arms: unavailable, or its "
                         f"benchmark does not allow them); it is written into stage 2's meta and records (FAILURES)")
    if arms is not None:
        bad = [a for a in arms if a not in MACHINE_ARMS[machine]]
        if bad or not arms:
            raise SystemExit(f"--arms {list(arms)}: must name some of --machine {machine}'s arms {list(MACHINE_ARMS[machine])}")
    with card_lock(hw, "stage 2"):
        return _stage2(dev, info, hw, machine, log, require_commit, reason, retry_halted,
                       tuple(a for a in MACHINE_ARMS[machine] if arms is None or a in arms), arms is not None,
                       other_card_idle, code_changed_decision)


def _stage2(dev, info, hw, machine, log, require_commit, reason, retry_halted, arms, arms_given, other_card_idle,
            code_changed_decision):
    sha = code_sha()[0]
    sync_or_exit("stage 2", require_commit)          # first: a checkout that is behind is told to pull
    jp, _ = record_paths()
    if not os.path.exists(jp):
        raise SystemExit("no stage-1 record: run --stage 1 first (on one card; the other card waits until the record is "
                         "pushed)")
    commit = committed_or_exit(jp, "the stage-1 record", require_commit)
    with open(jp) as f:
        rec1 = json.load(f)
    st_all = load_all()
    if superseded(sha, st_all):
        raise SystemExit(f"stage 2: code SHA {sha} was superseded: it is never used again (FAILURES)")
    if code_changed_decision:
        decide_code_changed(load(hw), hw, sha, code_changed_decision, log)
        st_all = load_all()
    _code_changed_or_exit(st_all, sha, "stage 2")
    stops = earlier_stops(sha, st_all)
    if stops:
        raise SystemExit(f"S75b stopped at stage 1 under code SHA {[x for _, x in stops]}: a value STOP under any code is "
                         f"final (FAILURES)")
    _frozen_or_exit(st_all, sha, "stage 2")
    _dup_exit(duplicates(st_all, sha), "stage 2")
    if rec1.get("running"):
        raise SystemExit(f"the stage-1 record (code {rec1.get('sha')}) was written when stage-1 invocation "
                         f"{rec1['running']} started, and that invocation has not rewritten it (still running, or "
                         f"killed)" + ("" if rec1.get("sha") == sha else f"; it is not this code's ({sha})")
                         + ": run --stage 1 again (it reruns only what is missing; under a code other than the "
                           "record's, it needs --code-change-note) and commit the record")
    d = pilot_of(st_all, sha)
    if not d["complete"]:
        raise SystemExit("both pilot records under this code SHA are needed (" + d["why"] + ")")
    if (rec1.get("sha") != sha or rec1["decision"].get("split_thr") != d["split_thr"]
            or rec1["decision"].get("proceed") != d["proceed"]):
        raise SystemExit("the stage-1 record does not match the pilot records under this code SHA")
    if require_commit:
        dc = pilot_of(committed_store(), sha)
        if not dc["complete"] or dc.get("split_thr") != d["split_thr"] or dc["proceed"] != d["proceed"]:
            raise SystemExit("the committed pilot records (every results file) do not give the stage-1 record's decision: "
                             "commit and push every results file")
    if not d["proceed"]:
        raise SystemExit(f"{stop_words(d)} (S75b stops at stage 1: {d['why']})")
    idle = _others_running_or_exit(st_all, hw, arms, "stage 2", other_card_idle)
    code_or_exit("stage 2", require_commit)
    workers, register, bench_commit, checks_commit, allowed, refused, bench_written = card_setup(
        dev, hw, sha, require_commit, "stage 2", arms)
    st = load(hw)
    if retry_halted:
        for arm in allowed:
            _clear_halted(st, hw, arm, seeds_of(arm), sha, log)
    meta2 = dict(machine=machine, reason=reason, invocation=INVOCATION, sha=sha, split_thr=d["split_thr"],
                 stage1_commit=commit, bench_commit=bench_commit, bench_written=bench_written, checks_commit=checks_commit,
                 register_reg3=register, workers=workers, threads=torch.get_num_threads(), arms=list(allowed),
                 arms_given=list(arms) if arms_given else None, refused_arms=refused, retry_halted=retry_halted,
                 other_card_idle=bool(other_card_idle), other_cards_not_ended=[(f, li.get("invocation")) for f, li in idle],
                 provenance=SPEC_PROVENANCE, changes=CHANGES_SINCE_SPEC, started=utc_now())
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, stage2=meta2,
                      last_invocation=dict(invocation=INVOCATION, stage=2, arms=list(allowed), ended=False,
                                           started=meta2["started"]))
    st["meta"].setdefault("stage2_invocations", []).append(meta2)
    save_json(st, store_path(hw))
    log(f"[S75b stage 2] machine {machine} ({hw}){' (' + reason + ')' if reason else ''}; arms {list(allowed)}"
        + (f" (--arms {list(arms)})" if arms_given else "")
        + (f"; REFUSED by this card's benchmark: {refused}" if refused else "")
        + f"; SPLIT_THR {d['split_thr']!r}; stage-1 commit {commit}; REG3 register {register}; stats workers {workers}; "
        f"threads {meta2['threads']}. Commit and push the results file now, so the other machine sees that stage 2 has "
        f"started (the freeze) and which arms run here. Do not pull or edit code until this invocation ends.")
    missing, aborted, reached = {}, None, []
    if refused:
        missing["(refused)"] = sorted(refused)
    try:
        with runtime():
            for arm in allowed:
                reached.append(arm)
                m = run_arm(arm, seeds_of(arm), dev, hw, sha, st,
                            dict(stage=2, machine=machine, reason=reason, workers=workers, bench_commit=bench_commit,
                                 bench_written=bench_written, invocation=INVOCATION), register, workers,
                            split_thr=d["split_thr"] if arm == "WIDE_SPLIT_G" else None, log=log)
                if m:
                    missing[arm] = m
    except InvocationAborted as e:
        aborted = str(e)
        log(f"[S75b stage 2] {e}")
        missing["(aborted)"] = [aborted]
    except BaseException as e:
        aborted = f"interrupted ({type(e).__name__}: {str(e)[:120]})"
        raise
    finally:
        meta2.update(ended=utc_now(), aborted=aborted)
        st["meta"]["last_invocation"].update(ended=True, at=meta2["ended"], aborted=aborted)
        save_json(st, store_path(hw))
    if aborted:
        stt = load_all()
        for arm in allowed:
            m = [s for s, v in statuses_of(stt, arm, sha).items() if v != "ok"]
            if m:
                missing[arm] = m
        missing["(not reached)"] = [a for a in allowed if a not in reached]
    if missing:
        stt = load_all()
        log(f"[S75b stage 2] seeds without an ok record: "
            f"{ {a: {s: seed_status(stt, a, s, sha) for s in m} for a, m in missing.items() if a in ARMS} }"
            + (f"; arms not reached: {missing['(not reached)']}" if missing.get("(not reached)") else "")
            + (f"; arms this card's benchmark refused: {refused}" if refused else "")
            + " (FAILURES: an operational failure is rerun by re-invoking --stage 2; a divergence is an outcome; a halted "
              "seed waits for the user)")
    if not missing.get("(not reached)"):
        missing.pop("(not reached)", None)
    return missing


# ── report ────────────────────────────────────────────────────────────────────
def report(log=print, sha=None, out=None):
    st = load_all()
    s2 = sorted(shas_with(st, STAGE2))
    sha = sha or (s2[0] if len(s2) == 1 else code_sha()[0])
    STT = {a: statuses_of(st, a, sha) for a in ARMS}
    R = {a: ok_runs(st, a, sha) for a in ARMS}
    DIV = {a: [s for s, v in STT[a].items() if v == "diverged"] for a in ARMS}
    HALT = {a: [s for s, v in STT[a].items() if v == "halted"] for a in ARMS}
    dup = duplicates(st, sha)
    L = [f"# S75b (EXPLORATORY): {task().label}; {ITERS} updates; code SHA {sha}", "",
         "Provenance as git shows it: " + "; ".join(SPEC_PROVENANCE["blind_by_git"]) + ". NOT BLIND BY GIT: "
         + "; ".join(SPEC_PROVENANCE["not_blind_by_git"]) + ".", "",
         f"Changes since the approved spec ({SPEC_COMMIT}), not blind ({CHANGES_SINCE_SPEC['note']}): "
         + "; ".join(CHANGES_SINCE_SPEC["items"]), "",
         "Cards per arm: " + "; ".join(f"{a} {cards_of(rr)}" for a, rr in R.items() if rr)]
    code_note = None
    if len(s2) > 1:
        code_note = f"CODE CHANGED DURING STAGE 2 (stage-2 records under {s2}): no reading; the user decides"
    elif s2 and sha not in s2:
        code_note = f"NOT THE STAGE-2 CODE (stage 2 ran under {s2}): no reading under {sha}"
    elif s2 and superseded(sha, st):
        cur = (json.load(open(record_paths()[0])) if os.path.exists(record_paths()[0]) else {}).get("sha")
        code_note = (f"CODE CHANGED AFTER STAGE 2 STARTED (a later code's stage 1 ran; the stage-1 record is now under "
                     f"{cur}): no reading; the user decides")
    cc = code_changed_runs(st, sha)
    if cc and not code_note:
        code_note = f"CODE CHANGED WHILE RUNS RAN ({cc}): no reading; the user decides"
    if code_note:
        L.append(code_note)
    bs = benches(sha)
    if bs:
        own = lambda c: MACHINE_ARMS["1660"] if "GTX 1660" in c else MACHINE_ARMS["3070"] if "RTX 3070" in c else STAGE2
        L.append("BENCHMARKS (this code): " + "; ".join(
            f"{card}: pilot {bench_why(b, ('ORACLE_PILOT',)) or 'usable'}, its arms {bench_why(b, own(card)) or 'usable'}, "
            f"all arms {bench_why(b, STAGE2) or 'usable'} (REG3 {b.get('register_reg3')}, {b.get('stats_workers')} stats "
            f"workers)" for card, (p, b) in bs.items()))
        cards = {("1660" if "GTX 1660" in c else "3070" if "RTX 3070" in c else c): bench_usable(b, ("ORACLE_PILOT",))
                 for c, (p, b) in bs.items()}
        if {"1660", "3070"} <= set(cards) and not any(cards.values()):
            L.append("S75b CANNOT START: no card's benchmark allows the pilot (FAILURES: no configuration, R or batch is "
                     "changed to make a card fit; the user decides).")
        if {"1660", "3070"} <= set(cards):
            nobody = [a for a in STAGE2 if not any(bench_usable(b, (a,)) for p, b in bs.values())]
            if nobody:
                eff = {"ORACLE": "validity stays incomplete", "SINGLE": "no GAP", "WIDE_SLOW": "no paired rows",
                       "WIDE_SPLIT_G": "the reading stays incomplete", "REG3": "the reading stays incomplete"}
                L.append(f"ARMS NO CARD'S BENCHMARK ALLOWS: {nobody}. They cannot run until the user decides: "
                         + "; ".join(f"{a}: {eff[a]}" for a in nobody) + " (FAILURES).")
    other = {}
    for k, r in st["runs"].items():
        sh = k.split("|")[2]
        if sh != sha:
            other[(sh, _card(k, r))] = other.get((sh, _card(k, r)), 0) + 1
    if other:
        L.append("NOT COUNTED (another code SHA): " + "; ".join(f"{n} records with SHA {sh} on {hw}"
                                                              for (sh, hw), n in sorted(other.items())))
    fl = [(k, f) for k, v in sorted(st["failures"].items()) for f in v]
    if fl:
        L.append("FAILURE LOG, every code (FAILURES: diverged = an outcome; operational = rerun on the same seed; open = "
                 "an attempt whose process ended without a record; cleared = the user's --retry-halted): " + "; ".join(
                     f"{k.split('|')[0]} {k.split('|')[1]} code {k.split('|')[2]} on {k.split('|')[3]} "
                     f"[{f.get('kind')}{' open' if f.get('open') else ''}] attempt {f.get('attempt')} invocation "
                     f"{f.get('invocation')}: {str(f.get('error'))[:100]}" for k, f in fl))
    s2inv = [m for meta in st["meta"].values() for m in (meta.get("stage2_invocations") or []) if meta.get("sha") == sha]
    if s2inv:
        L.append("STAGE-2 INVOCATIONS: " + "; ".join(
            f"{m.get('invocation')} --machine {m.get('machine')}" + (f" (reason: {m.get('reason')})" if m.get("reason") else "")
            + (f" --arms {m.get('arms_given')}" if m.get("arms_given") else "")
            + (" --other-card-idle" if m.get("other_card_idle") else "")
            + f": arms {m.get('arms')}" + (f", refused {sorted(m.get('refused_arms') or {})}" if m.get("refused_arms") else "")
            + ("" if m.get("ended") else " (not ended)") + (f"; stopped: {str(m.get('aborted'))[:100]}" if m.get("aborted")
                                                            else "")
            for m in s2inv))
    kept = sorted(k for k, r in st["runs"].items() if (r.get("decided") or {}).get("choice") == "keep")
    aside = sorted(st.get("set_aside") or {})
    if kept or aside:
        L.append("RUNS WHOSE CODE CHANGED WHILE THEY RAN, DECIDED BY THE USER (--code-changed-decision): "
                 + (f"kept (counted): {kept}" if kept else "") + ("; " if kept and aside else "")
                 + (f"set aside (not counted; their seeds rerun): {aside}" if aside else ""))
    if dup:
        L.append("DUPLICATE RECORDS (arm, seed, cards; an ok record or a divergence on each): "
                 + "; ".join(f"{a} {s} {h}" for a, s, h in dup)
                 + ". The reading is incomplete until the user decides which card's draw counts.")
    L.append("")
    for p, r in stage1_records():
        if r.get("sha") != sha:
            dd = r.get("decision") or {}
            L.append(f"Stage 1 under code {r.get('sha')} (written {r.get('written')}; {os.path.basename(p)}; never pooled): "
                     f"end.set {dd.get('sets')}; " + ("PROCEED" if dd.get("proceed") else "STOP (final)" if dd.get("complete")
                                                     else "halted" if dd.get("halted") else "incomplete"))
    if os.path.exists(record_paths()[0]):
        r_ = json.load(open(record_paths()[0]))
        if r_.get("running"):
            L.append(f"STAGE-1 RECORD WRITTEN BY A STAGE 1 THAT HAS NOT ENDED ({r_['running']}; code {r_.get('sha')}): "
                     f"still running, or killed (FAILURES).")
    for w, x in earlier_stops(sha, st):
        L.append(f"VALUE STOP under code {x} ({w}): final under any code.")
    pil = R["ORACLE_PILOT"]
    d = pilot_of(st, sha)
    L.append(f"Stage 1 (pilot, never pooled): end.set {[(s, round(r['end']['set'], 4)) for s, r in sorted(pil.items())]}; "
             f"seed status {d.get('statuses')}; {d['why']}; " + (f"PROCEED, SPLIT_THR {d['split_thr']:.6f}" if d["proceed"] else
                                                                  stop_words(d) if d["complete"] else
                                                                  "HALTED (the user decides)" if d.get("halted") else
                                                                  "incomplete"))
    ost = STT["ORACLE"]
    orc = [R["ORACLE"].get(s) for s in ORACLE_SEEDS]
    complete_orc = all(r is not None for r in orc)
    if DIV["ORACLE"]:
        vtag, valid, resolved = f"NOT VALID: ORACLE {DIV['ORACLE']} diverged (counts as below 0.9) -> UNTESTED", False, True
    elif any(r is not None and r["end"]["set"] < BAR for r in orc):
        vtag, valid, resolved = "NOT VALID -> UNTESTED", False, True
    elif HALT["ORACLE"]:
        vtag, valid, resolved = f"incomplete: ORACLE {HALT['ORACLE']} halted (the user decides)", False, False
    elif complete_orc:
        vtag, valid, resolved = "VALID", True, True
    else:
        vtag, valid, resolved = "incomplete", False, False
    oset = {f: statistics.fmean(r["end"][f] for r in orc) for f in ["set"] + [f"set_{b}" for b in it.BIN_NAMES]} \
        if complete_orc else None
    L.append(f"VALIDITY: ORACLE 850-851 final SET {[round(r['end']['set'], 4) for r in orc if r]}; seed status {ost} -> {vtag}")
    sing = R["SINGLE"]
    L += ["", "| arm | ROUTED (Wilson; band; a diverged run counts as not ROUTED) | median GAP | GAP by bin 1 / 2 / 3-4 / 5+ | "
              "median SET | splits fired / blocked one-to-one | median eta (layers) | median margin (layers) |",
          "|---|---|---|---|---|---|---|---|"]
    ks, ns = {}, {}
    for a in STAGE2:
        rr = R[a]
        k_, n_ = sum(idense.routed(r) for r in rr.values()), len(rr) + len(DIV[a])
        ks[a], ns[a] = k_, n_
        extra = (f" (diverged {DIV[a]})" if DIV[a] else "") + (f" (halted {HALT[a]})" if HALT[a] else "")
        cl = (f"{k_}/{n_} [{ic.wilson(k_, n_)[0]:.2f}, {ic.wilson(k_, n_)[1]:.2f}] {ic.band(k_, n_)}" if n_ else "-") + extra
        if not rr:
            L.append(f"| {a} | {cl} | | | | | | |")
            continue
        g = [idense.gap(r, sing[s], oset) for s, r in rr.items() if s in sing] if oset else []
        gb = [idense.med([idense.gap(r, sing[s], oset, f"set_{b}") for s, r in rr.items() if s in sing]) if oset else None
              for b in it.BIN_NAMES]
        fired = sum(c["fired"] for r in rr.values() for c in r.get("checks", []))
        oto = sum(c.get("blocked_by") == "one-to-one" for r in rr.values() for c in r.get("checks", []))
        lay = sorted(next(iter(rr.values()))["end"].get("routing") or {})
        rt = [r for r in rr.values() if r["end"].get("routing")]
        etas = [idense.med([r["end"]["routing"][y]["eta"] for r in rt]) for y in lay]
        mgs = [idense.med([r["end"]["routing"][y]["margin"] for r in rt]) for y in lay]
        L.append(f"| {a} | {cl} | {idense.fmt(idense.med(g))} | " + " / ".join(idense.fmt(x) for x in gb) +
                 f" | {idense.fmt(idense.med([r['end']['set'] for r in rr.values()]), 3)} | "
                 f"{(str(fired) + ' / ' + str(oto)) if a == 'WIDE_SPLIT_G' else '-'} | "
                 f"{' / '.join(idense.fmt(x) for x in etas) or '-'} | {' / '.join(idense.fmt(x) for x in mgs) or '-'} |")
    kr = {a: ks[a] for a in READ_ARMS}
    halt_read = {a: HALT[a] for a in READ_ARMS if HALT[a]}
    done = resolved and all(ns[a] == len(SEEDS) for a in READ_ARMS)
    stops_ = earlier_stops(sha, st)
    if dup:
        tag = "incomplete: records from two cards (the user decides)"
    elif code_note:
        tag = code_note
    elif stops_:
        tag = (f"NO READING: S75b stopped at stage 1 under code {[x for _, x in stops_]} (a value STOP is final under any "
               f"code; FAILURES); the stage-2 rows are descriptions only")
    elif d["complete"] and not d["proceed"]:
        tag = ("NO READING: S75b stopped at stage 1 under this code (a value STOP is final; FAILURES)"
               + ("; the stage-2 rows are descriptions only" if any(R[a] or DIV[a] for a in STAGE2) else ""))
    elif not d["proceed"] and any(R[a] or DIV[a] for a in STAGE2):
        tag = "NO READING: the pilot under this code did not PROCEED; the stage-2 rows are descriptions only"
    elif resolved and not valid:
        tag = "UNTESTED (descriptions only; never re-read under a later bar)"
    elif halt_read:
        tag = f"incomplete: read-arm seeds halted {halt_read} (the user decides)"
    elif not done:
        tag = "incomplete"
    else:
        tag = "carries to four sources" if max(kr.values()) >= HI else "does not carry"
    L += ["", f"READING (pre-fixed): {tag}  (ROUTED {kr} of {ns['WIDE_SPLIT_G']} / {ns['REG3']}; the better arm needs "
              f">= {HI}/10)"]
    if DIV["SINGLE"]:
        L.append(f"SINGLE diverged on {DIV['SINGLE']}: no GAP for those seeds.")
    if oset and sing:
        us = [s for s, r in sing.items() if r["end"]["set"] <= oset["set"] - 0.2]
        L.append(f"Printed: S74's usability count: ORACLE mean {oset['set']:.4f}; SINGLE <= ORACLE - 0.2 on {len(us)}/{len(sing)}")
    L += ["", "DIAGNOSTICS (medians at the end): eta^2 by source at word index 1-2 / 5+ (layer 1); the probe per layer; "
              "REG3's beta^R at SRC / word; train minutes per run; peak device GB; cards"]
    for a, rr in R.items():
        if not rr:
            continue
        rt = [r for r in rr.values() if r["end"].get("routing")]
        e12 = idense.med([r["end"]["routing"]["L1"].get("eta_pos12") for r in rt if "L1" in r["end"]["routing"]])
        e5 = idense.med([r["end"]["routing"]["L1"].get("eta_pos5p") for r in rt if "L1" in r["end"]["routing"]])
        pk = sorted({k for r in rr.values() for k in (r["end"].get("probe") or {})})
        pr = [(k, idense.med([r["end"]["probe"][k]["acc"] for r in rr.values() if k in (r["end"].get("probe") or {})]))
              for k in pk]
        L.append(f"  {a:<12} eta 1-2 {idense.fmt(e12)} / 5+ {idense.fmt(e5)}; probe "
                 + (" / ".join(f"{k} {idense.fmt(v)}" for k, v in pr) or "-")
                 + (f"; beta {idense.fmt(idense.med([r.get('beta_src') for r in rr.values()]))} / "
                    f"{idense.fmt(idense.med([r.get('beta_word') for r in rr.values()]))}" if a == "REG3" else "")
                 + f"; train {idense.fmt(idense.med([r.get('train_secs', 0) / 60 for r in rr.values()]), 0)} min; "
                 f"peak {idense.fmt(idense.med([r.get('peak_mem_gb_batch') for r in rr.values()]))} GB; {cards_of(rr)}")
    L += ["", "WIDE_SPLIT_G's split checks (every eligible check): seed step probe-SET map-before one-to-one outcome"]
    for s, r in sorted(R["WIDE_SPLIT_G"].items()):
        for c in r.get("checks", []):
            if c.get("eligible"):
                L.append(f"  {s} {c['step']} {c['acc']:.3f} {c['map_before']} {c.get('map_one_to_one')} "
                         f"{'fired' if c['fired'] else 'blocked: ' + str(c.get('blocked_by'))}")
    L += ["", "WIDE_SPLIT_G beside WIDE_SLOW, paired by seed: seed ROUTED(split_g) ROUTED(slow) SET(split_g) SET(slow)"]
    for s in SEEDS:
        a, b_ = R["WIDE_SPLIT_G"].get(s), R["WIDE_SLOW"].get(s)
        if not (a or b_):
            continue
        cells = [("R" if idense.routed(r) else ".") if r else "-" for r in (a, b_)]
        cells += [f"{r['end']['set']:.4f}" if r else "-" for r in (a, b_)]
        L.append(f"  {s} " + " ".join(cells))
    L += ["", "PER RUN: arm seed ROUTED SET GAP; map / eta / margin per layer; splits"]
    for a, rr in R.items():
        for s, r in sorted(rr.items()):
            rt = r["end"].get("routing") or {}
            g = idense.gap(r, sing[s], oset) if (a != "ORACLE_PILOT" and s in sing and oset) else None
            L.append(f"  {a:<12} {s} {'R' if idense.routed(r) else '.'} {r['end']['set']:.4f} {idense.fmt(g)} "
                     + " ".join(f"{y}:{v['ch_map']}/{v['eta']:.2f}/{v['margin']:.2f}" for y, v in rt.items())
                     + (f" splits {r.get('splits')} at {[c['step'] for c in r.get('checks', []) if c['fired']]}"
                        if "splits" in r else ""))
    txt = "\n".join(L) + "\n"
    with open(out or os.path.join(OUT, "s75b_summary.md"), "w") as f:
        f.write(txt)
    log(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", type=int, choices=(1, 2))
    ap.add_argument("--machine", choices=tuple(MACHINE_ARMS))
    ap.add_argument("--reason", default=None, help="--machine fallback / fallback-3070: why the other card cannot run its "
                                                    "arms (written into stage 2's meta)")
    ap.add_argument("--retry-halted", action="store_true", help="the user clears this stage's halted seeds (recorded)")
    ap.add_argument("--arms", default=None, help="--stage 2: run only these of the machine's arms (comma-separated; "
                                                 "recorded), e.g. a fallback for the arms the other card's benchmark refuses")
    ap.add_argument("--other-card-idle", action="store_true",
                    help="the user states that no S75b invocation runs on the other card although its last invocation "
                         "shows no end (it was killed), after that card's results file was committed and pushed from "
                         "its own machine; recorded, and its open markers count as killed")
    ap.add_argument("--code-changed-decision", choices=("keep", "rerun"), default=None,
                    help="--stage 1 / 2: the user's decision on this card's runs whose code changed while they ran "
                         "(recorded): keep counts them; rerun sets them aside (kept, printed) and reruns their seeds")
    ap.add_argument("--code-change-note", default=None, help="--stage 1 under a new code before stage 2: a committed, "
                                                            "dated note on why the change could not be avoided")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--host-ram-gb", type=float, default=None, help="the machine's RAM in GB (the user's figure)")
    ap.add_argument("--beside", default=None,
                    help="a string in the command line of the job running beside S75b on this machine, or 'none'")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sha", default=None, help="--report: the code SHA whose records count (default: stage 2's)")
    a = ap.parse_args()
    if (a.arms is not None and a.stage != 2) or ((a.other_card_idle or a.code_changed_decision)
                                                  and a.stage not in (1, 2)):
        raise SystemExit("--arms is for --stage 2; --other-card-idle and --code-changed-decision for --stage 1 or 2")
    one_thread()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        report(sha=a.sha)
        return
    if a.bench:
        if a.host_ram_gb is None or a.beside is None:
            raise SystemExit("--bench needs --host-ram-gb (the machine's RAM, the user's figure) and --beside NAME|none")
        info = dv.device_info(dv.setup(a.device))
        path = bench_path(info["gpu"] or info["cpu"])   # an earlier file is archived by bench() (FAILURES: reruns allowed)
        with card_lock(info["gpu"] or info["cpu"], "--bench"):     # never beside a stage on this card
            b = bench(a.device, a.host_ram_gb, a.beside, path=path)
        print(json.dumps({k: v for k, v in b.items() if k not in ("arms", "changes", "provenance")}, indent=1, default=str))
        why = bench_why(b, STAGE2)
        if why:
            print(f"this card is not usable for every arm: {why}; for the pilot: {bench_why(b, ('ORACLE_PILOT',)) or 'usable'}"
                  f"; for its own arms: see --report (FAILURES)")
        return
    if a.stage == 1:
        d, missing = stage1(a.device, retry_halted=a.retry_halted, code_change_note=a.code_change_note,
                            other_card_idle=a.other_card_idle, code_changed_decision=a.code_changed_decision)
        sys.exit(0 if d["complete"] else 1)
    if a.stage == 2:
        if a.machine is None:
            raise SystemExit("--stage 2 needs --machine 1660, 3070, fallback or fallback-3070")
        missing = stage2(a.device, a.machine, reason=a.reason, retry_halted=a.retry_halted,
                         arms=None if a.arms is None else tuple(x.strip() for x in a.arms.split(",") if x.strip()),
                         other_card_idle=a.other_card_idle, code_changed_decision=a.code_changed_decision)
        stt = load_all()
        sha = code_sha()[0]
        open_ = [(arm, s) for arm, m in missing.items() if arm in ARMS for s in m
                 if seed_status(stt, arm, s, sha) != "diverged"]
        sys.exit(1 if open_ or "(aborted)" in missing or "(refused)" in missing else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
