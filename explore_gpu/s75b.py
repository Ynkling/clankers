#!/usr/bin/env python
"""
explore_gpu/s75b.py — EXPLORATORY, not a result. S75b: S75 at S = 4 on a configuration where the bar is reachable. The
specification is explore_out/I/s75b_spec_draft.md at 114cd08, approved by the user on 2026-10-12. This driver implements
it; nothing below changes a constant of it.

PROVENANCE, AS GIT SHOWS IT (SPEC_PROVENANCE below; written into every record and the report):
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
  STOP otherwise: S75b stops; "S = 4 is not usable at B" only if a pilot run ended below 0.9 (or diverged, below); no
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
  WIDE_SPLIT_G; if the 3070 is unavailable when stage 2 starts, every arm runs on the 1660 (--machine fallback) and
  nothing is dropped; if the 1660's benchmark fails, every arm runs on the 3070 (--machine fallback-3070; FAILURES
  below). R = 1 for every arm on both cards. REG3's register mode by rule from this card's benchmark: "scan" if
  the benchmark's peak reserved memory with the scan, plus the context (memory in use beyond this process's reserve),
  is below the card's total; otherwise "loop". --stats-workers from the benchmark's host memory, by this driver's
  formula (workers_rule; listed under CHANGES).

FAILURES (rules added on 2026-10-13, before the pilot, for the outcomes Referee V's review found uncovered; NOT BLIND):
  - A RUN.
    - A failed run whose error reports a non-finite value is DIVERGED. That is an outcome: it is never retried and never
      rerun.
    - Any other failed run (out of memory, a crash) is OPERATIONAL. It is retried once by the driver and is otherwise
      rerun on the same seed by re-invoking the stage. No seed is ever replaced by a fresh one.
    - A seed with operational failures in OPERATIONAL_LIMIT = 2 invocations, and no ok record, CANNOT RUN.
  - THE PILOT.
    - A diverged pilot run did not reach 0.9: S75b stops, "S = 4 is not usable at B".
    - A pilot seed that cannot run: S75b stops, "B's pilot could not be run".
    - Otherwise a pilot seed without an ok record is rerun (--stage 1 again).
  - A CODE CHANGE BETWEEN THE STAGES.
    - The driver's code is frozen from stage 1 to the end of stage 2. A change that cannot be avoided (a bug that stops a
      stage) is committed with a dated note.
    - The CUDA CHECKs and the benchmarks are then rerun, and the pilot is rerun on 840-841 under the new code.
    - A stage-1 STOP under any code is final.
    - Stage 2 runs only if every complete stage-1 record, under every code, says PROCEED. SPLIT_THR comes from the pilot
      under the code stage 2 runs.
    - Earlier stage-1 records are kept as s75b_stage1_record_<sha>.*, printed, and never pooled.
  - STAGE 2.
    - A diverged ORACLE 850 / 851 run counts as below 0.9: NOT VALID, so the reading is UNTESTED.
    - An ORACLE seed that cannot run leaves VALID unestablished: UNTESTED.
    - A diverged run of a learned arm is not ROUTED, and its seed stays in n.
    - A diverged SINGLE run gives no GAP for its seed.
    - A read-arm seed that cannot run leaves the reading INCOMPLETE: descriptions only.
  - THE BENCHMARK.
    - A card whose L = 1536 benchmark fails a case it would run, or whose host-memory rule refuses, runs no S75b arm.
    - If only one card is usable, it runs everything, the pilot included: --machine fallback on the 1660, or --machine
      fallback-3070 on the 3070. The 3070 fallback is allowed only when the 1660's committed benchmark under this code
      failed or refused.
    - If neither card is usable, S75b stops before the pilot: "B cannot be run on these cards".

CHANGES SINCE 114cd08 (CHANGES_SINCE_SPEC below; written into the benchmark files, the stage-1 record, stage 2's meta and
  the report). They were made after S75's learned-arm records were read, so they are NOT BLIND; none changes a constant
  of the design (configuration, seeds, arms, outcomes, bars, the proceed rule, SPLIT_THR's rule, the reading):
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
  - Records of one arm and seed from two cards are not resolved by the driver: stage 1 / 2 refuse, and the report prints
    them and marks the reading incomplete.
  - Order: stage 1 needs the driver's code, this card's benchmark and its CUDA CHECK output committed; stage 2 needs the
    driver's code, the stage-1 record, the pilot results file and this card's benchmark and CUDA CHECK output committed,
    and the pilot card's benchmark unchanged since the stage-1 record.
  - CHECK B7 (the spec's lock-step at L = 1536): R = 1 as the runs, each seed alone, batch 8 on CUDA (the R = 2, batch-32
    lock-step does not fit either card), REG3 in both register modes on both cards.

RUN (from the repository root; every step writes into explore_out/I/gpu/s75b/):
  python -m explore_gpu.s75b_checks --device cpu            CHECKs (this container); --device cuda on each card (commit
                                                            explore_gpu/s75b_checks_cuda_<GPU>_output.*)
  python -m explore_gpu.s75b --bench --device cuda --host-ram-gb G --beside NAME|none
        each card, before its first stage: G = the machine's RAM in GB (the user's figure: 15.2 on the laptop), NAME = a
        string in the command line of the job running beside S75b (the laptop: test_delta_recipe), "none" if nothing
        does; commit the bench file
  python -m explore_gpu.s75b --stage 1 --device cuda        the pilot (either card); then commit and push the record and
                                                            the pilot results file
  python -m explore_gpu.s75b --stage 2 --device cuda --machine 1660|3070|fallback|fallback-3070
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
OPERATIONAL_LIMIT = 2                      # invocations with an operational failure of one seed before it CANNOT RUN
MARGIN_CHUNK = 16
MAX_WORKERS = 3                            # explore_gpu.run's default on CUDA
DATA_PROC_GB = 0.5                         # the data process, where its peak cannot be measured (no /proc)
OS_RESERVE_GB = 1.5                        # the operating system and everything else not measured
AVAIL_RESERVE_GB = 0.5                     # kept free of MemAvailable
CHECKS_DIR = os.path.join(HERE, "explore_gpu")   # where explore_gpu.s75b_checks writes its outputs

SPEC_COMMIT = "114cd08"
SPEC_PROVENANCE = dict(
    design="1e31409", design_parent="5bc6c2e", merge="142f47e", learned_arms="b649751",
    blind_by_git=[
        "1e31409, the design: built on 5bc6c2e, whose tree holds only S75's ORACLE 830-831 and SINGLE 830-839 records. "
        "That it was written blind is the author's statement; git shows the tree, not what was read (b649751 was on "
        "the remote at 13:53:11 UTC, 23 min before 1e31409)"],
    not_blind_by_git=[
        "ee97443, the cost section: made after the merge 142f47e, with b649751's WIDE_SPLIT and REG3 records and "
        "s75_S4_summary.md in the tree",
        "114cd08, the revision (with b649751 in the tree; the same session committed the S75 write-up 03bffdf 130 s "
        "later). It loosened the proceed rule from 'both >= 0.912' to 'mean >= 0.912 and each >= 0.9', and made "
        "WIDE_SLOW unconditional. It also made corrections and operational choices: the bound 0.963, the gap share, "
        "the windowed learner and the fixed-reach case, the stop wording, the guard's precedence, the allocation, R = 1, "
        "the register rule, the workers, n = 4096, the memory and the 3070's factor",
        "every change after 114cd08 (CHANGES_SINCE_SPEC)",
    ])
CHANGES_SINCE_SPEC = dict(
    spec=SPEC_COMMIT, blind=False,
    note="made after S75's learned-arm records were read and with them in the tree, so not blind (by git either); none "
         "changes a constant of the design (configuration, seeds, arms, outcomes, bars, the proceed rule, SPLIT_THR's "
         "rule, the reading)",
    items=[
        "the pilot may run on the RTX 3070 if it is free before the GTX 1660 (the user, 12 October); the stage-1 record "
        "names the card",
        "the routing margin in chunks of 16 sequences in S75b's processes and the statistics pool StatsPool (host memory; "
        "the same value up to float64 summation order, CHECK B3)",
        "--stats-workers by this driver's formula (workers_rule; the spec says only 'from the benchmark's host memory'): "
        "measured peaks (VmHWM) of the main process, one statistics worker and the data process, the evaluations held "
        "while the workers lag, and a budget from the machine's RAM (the user's figure), the co-running job's measured "
        "peak (--beside) and MemAvailable",
        "records of one arm and seed from two cards are not resolved: stage 1 / 2 refuse, the report marks the reading "
        "incomplete",
        "order: stage 1 needs the driver's code, this card's benchmark and CUDA CHECK output committed; stage 2 needs the "
        "driver's code, the stage-1 record, the pilot results file and this card's benchmark and CUDA CHECK output "
        "committed",
        "CHECK B7 (lock-step at L = 1536): R = 1 as the runs, each seed alone, batch 8 on CUDA, REG3 in both register "
        "modes (the R = 2, batch-32 lock-step does not fit either card)",
        "failed runs (rules added 2026-10-13, before the pilot): an error reporting a non-finite value is DIVERGED, an "
        "outcome, never retried or rerun; any other failure is OPERATIONAL, rerun on the same seed, never a fresh "
        "seed; operational failures in 2 invocations mean the seed CANNOT RUN",
        "the pilot: a diverged run stops S75b ('S = 4 is not usable at B'); a seed that cannot run stops it ('B's pilot "
        "could not be run'); otherwise a seed without an ok record is rerun",
        "a code change between the stages: the code is frozen; an unavoidable change is committed with a dated note, "
        "the CUDA CHECKs, benchmarks and the pilot (840-841) are rerun under the new code; a stage-1 STOP under any "
        "code is final; stage 2 needs every complete stage-1 record to say PROCEED; SPLIT_THR from the pilot under "
        "the code stage 2 runs; earlier records kept as s75b_stage1_record_<sha>.*, never pooled",
        "stage 2: a diverged ORACLE run counts as below 0.9 (UNTESTED); an ORACLE seed that cannot run leaves VALID "
        "unestablished (UNTESTED); a diverged learned-arm run is not ROUTED (n stays 10); a diverged SINGLE run gives "
        "no GAP for its seed; a read-arm seed that cannot run leaves the reading INCOMPLETE",
        "the benchmark: a card whose benchmark fails or refuses runs nothing; one usable card runs everything "
        "(--machine fallback on the 1660, fallback-3070 on the 3070, the latter only if the 1660's committed "
        "benchmark failed or refused); neither usable: S75b stops before the pilot",
        "provenance labels: SPEC_PROVENANCE marks every change after the merge 142f47e (ee97443, 114cd08 and this "
        "list) as not blind by git",
    ])


def task():
    return it.ICMCTask(**CONFIG)


def arm_spec(arm):
    return gr.arm_spec(ARMS[arm][0], CONFIG["S"])


def seeds_of(arm):
    return ARMS[arm][1]


def code_sha():
    return gr.code_sha("explore_gpu.s75b")


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


@contextlib.contextmanager
def runtime():
    """S75b's process-level patches: the small routing margin, the statistics pool. Restored on exit."""
    small_margin()
    old = gr.AsyncStats
    gr.AsyncStats = StatsPool
    try:
        yield
    finally:
        gr.AsyncStats = old


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
    out = {"meta": {}, "runs": {}, "failures": {}}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        out["meta"][os.path.basename(p)] = st["meta"]
        out["runs"].update(st["runs"])
        out["failures"].update(st.get("failures") or {})
    return out


# ── failed runs (FAILURES in the module docstring) ────────────────────────────
INVOCATION = f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}"   # reset at the start of each stage


def failure_kind(err):
    """'diverged' if a run's error reports a non-finite value (the record's non-finite-loss check, or a NaN / inf
    reaching a statistic), else 'operational'."""
    import re
    return "diverged" if re.search(r"non-finite|\bnan\b|\binf\b|infinity", str(err or ""), re.I) else "operational"


def seed_status(st, arm, seed, sha):
    """'diverged' (a divergence is the outcome, whatever came after), 'ok', 'cannot run' (operational failures in
    OPERATIONAL_LIMIT invocations, no ok record), 'failed' (operational failures, rerun) or 'missing'."""
    want = [arm, str(seed), sha]
    fails = [f for k, fl in (st.get("failures") or {}).items() if k.split("|")[:3] == want for f in fl]
    if any(f.get("kind") == "diverged" for f in fails):
        return "diverged"
    if any(k.split("|")[:3] == want and r.get("ok") for k, r in st["runs"].items()):
        return "ok"
    if len({f.get("invocation") for f in fails if f.get("kind") == "operational"}) >= OPERATIONAL_LIMIT:
        return "cannot run"
    return "failed" if fails else "missing"


def statuses_of(st, arm, sha, seeds=None):
    return {s: seed_status(st, arm, s, sha) for s in (seeds or seeds_of(arm))}


def ok_runs(st, arm, sha):
    """runs_of without seeds whose status is diverged (a divergence is the outcome even if a later run completed)."""
    stt = statuses_of(st, arm, sha)
    return {s: r for s, r in runs_of(st, arm, sha).items() if stt.get(s) != "diverged"}


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
    """[(arm, seed, cards)] for each arm and seed with ok records under sha from more than one card."""
    seen = {}
    for k, r in st["runs"].items():
        a, s, sh = k.split("|")[:3]
        if sh == sha and r.get("ok") and (arms is None or a in arms):
            seen.setdefault((a, int(s)), set()).add(_card(k, r))
    return [(a, s, sorted(h)) for (a, s), h in sorted(seen.items()) if len(h) > 1]


def other_cards(st, sha, arms, hw):
    """{arm: sorted seeds} with ok records under sha from a card other than hw."""
    out = {}
    for k, r in st["runs"].items():
        a, s, sh = k.split("|")[:3]
        if sh == sha and r.get("ok") and a in arms and _card(k, r) != hw:
            out.setdefault(a, set()).add(int(s))
    return {a: sorted(v) for a, v in sorted(out.items())}


def cards_of(rr):
    return sorted({r.get("hw") for r in rr.values()}, key=str)


# ── the stage-1 rules ─────────────────────────────────────────────────────────
def pilot_decision(pilot, statuses=None):
    """pilot: {seed: ok record}; statuses: {seed: seed_status} (FAILURES). Returns the stage-1 decision (the module
    docstring's rules), all values unrounded."""
    statuses = statuses or {}
    stt = {s: ("ok" if (pilot.get(s) or {}).get("ok") else statuses.get(s, "missing")) for s in PILOT_SEEDS}
    stt.update({s: "diverged" for s in PILOT_SEEDS if statuses.get(s) == "diverged"})
    have = {s: pilot.get(s) for s in PILOT_SEEDS}
    sets0 = {s: ((r or {}).get("end") or {}).get("set") if stt[s] == "ok" else None for s, r in have.items()}
    div = [s for s in PILOT_SEEDS if stt[s] == "diverged"]
    cant = [s for s in PILOT_SEEDS if stt[s] == "cannot run"]
    if div:
        return dict(complete=True, proceed=False, statuses=stt, sets=sets0, split_thr=None, diverged=div, cannot_run=cant,
                    not_usable=True, not_runnable=bool(cant),
                    why=f"pilot run(s) {div} diverged (a non-finite value): below 0.9")
    if cant:
        return dict(complete=True, proceed=False, statuses=stt, sets=sets0, split_thr=None, diverged=[], cannot_run=cant,
                    not_usable=False, not_runnable=True,
                    why=f"pilot seed(s) {cant} failed operationally in {OPERATIONAL_LIMIT} invocations: they cannot run")
    if any(v != "ok" for v in stt.values()):
        return dict(complete=False, proceed=False, statuses=stt, sets=sets0,
                    why="a pilot record is missing or failed operationally: rerun --stage 1 (the same seeds)")
    sets = {s: float(r["end"]["set"]) for s, r in have.items()}
    mean = statistics.fmean(sets.values())
    each = all(v >= PROCEED_EACH for v in sets.values())
    proceed = mean >= PROCEED_MEAN and each
    thr = min(sets.values()) - THR_MARGIN
    why = (f"mean {mean:.6f} {'>=' if mean >= PROCEED_MEAN else '<'} {PROCEED_MEAN}; each >= {PROCEED_EACH}: {each}")
    return dict(complete=True, proceed=proceed, statuses=stt, sets=sets, mean=mean, each_ok=each,
                split_thr=thr if proceed else None, diverged=[], cannot_run=[], not_runnable=False,
                not_usable=any(v < BAR for v in sets.values()), why=why)


def pilot_of(st, sha):
    """The stage-1 decision from a store (load_all() or a committed copy) under sha."""
    return pilot_decision(runs_of(st, "ORACLE_PILOT", sha), statuses_of(st, "ORACLE_PILOT", sha, PILOT_SEEDS))


def stop_words(d):
    """The stop's wording: always that B's pilot did not clear the proceed rule; "S = 4 is not usable at B" only if a pilot
    run ended below 0.9 or diverged; "B's pilot could not be run" if a pilot seed cannot run (FAILURES)."""
    s = "S75b stops: B's pilot did not clear the proceed rule"
    if d.get("diverged"):
        s += f"; pilot run(s) {d['diverged']} diverged (a non-finite value) and so did not reach 0.9: S = 4 is not usable at B"
    elif d.get("not_usable"):
        s += "; a pilot run ended below 0.9: S = 4 is not usable at B"
    if d.get("not_runnable"):
        s += (f"; pilot seed(s) {d.get('cannot_run')} failed operationally in {OPERATIONAL_LIMIT} invocations: B's pilot "
              f"could not be run on these cards")
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


def earlier_stops(sha):
    """Stage-1 records under another code whose decision is a complete STOP (final under any code)."""
    return [(p, r) for p, r in stage1_records()
            if r.get("sha") != sha and (r.get("decision") or {}).get("complete") and not r["decision"].get("proceed")]


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
    files = [os.path.join(HERE, *n.split(".")) + ".py" if n != "explore_gpu" else os.path.join(HERE, "explore_gpu",
                                                                                               "__init__.py")
             for n in names]
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
    """None if this card's CUDA CHECK output exists, was written under sha, holds B1-B8 (B7 run, not skipped) and every
    CHECK passed; otherwise why not."""
    p = checks_output_path(gpu)
    if not os.path.exists(p):
        return "missing"
    with open(p) as f:
        d = json.load(f)
    ch = d.get("checks") or {}
    if (d.get("meta") or {}).get("s75b_sha") != sha:
        return f"written under code SHA {(d.get('meta') or {}).get('s75b_sha')}, not {sha}"
    keys = [f"B{i}" for i in range(1, 9)]
    if any(k not in ch for k in keys):
        return f"incomplete: {[k for k in keys if k not in ch]} missing"
    bad = [k for k in keys if not ch[k].get("ok")]
    if bad:
        return f"failed: {bad}"
    if ch["B7"].get("skipped"):
        return "B7 skipped"
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


def bench(device, host_ram_gb, beside="none", iters=12, every=3, log=print):
    """Each S75b arm at R = 1 on this card for `iters` updates with an evaluation every `every` (the device part here, the
    statistics in one StatsPool worker, waited for at each evaluation) and, for WIDE_SPLIT_G, the guarded split checks
    (the trigger's schedule scaled to `every`, threshold 1.01, rise 1.0); REG3 in both register modes. Reads peak device
    memory, the host memory of this process, the statistics worker and the data process (VmHWM), the co-running job's
    (beside), MemTotal / MemAvailable, the seconds per update and per evaluation's statistics; applies register_rule and
    workers_rule."""
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
               eval_every=every, arms={}, changes=CHANGES_SINCE_SPEC, written=time.strftime("%Y-%m-%d %H:%M:%S"))
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
                        torch.cuda.empty_cache()
                out["arms"][name] = row
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
               all_ok=all(out["arms"][n].get("ok") for n in out["cases_used"]), sha=code_sha()[0])
    return out


def load_bench(hw, sha):
    p = bench_path(hw)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        b = json.load(f)
    return b if b.get("sha") == sha else None


def bench_usable(b):
    """A card is usable for S75b iff its benchmark passed every case it would run and its host-memory rule gave workers."""
    return bool(b and b.get("all_ok") and b.get("stats_workers") is not None)


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


def card_setup(dev, hw, sha, require_commit, what):
    """This card's run mechanics from its benchmark (CUDA: required, committed, every case ok, a worker count; and the
    CUDA CHECK output committed and passing). CPU: the statistics in line, REG3 with the loop. Returns (workers,
    register, bench_commit, checks_commit)."""
    if dev.type != "cuda":
        return 0, "loop", None, None
    b = load_bench(hw, sha)
    if b is None:
        raise SystemExit(f"{what}: no benchmark for {hw} under code SHA {sha}: run --bench on this card and commit it")
    if not b.get("all_ok"):
        raise SystemExit(f"{what}: this card's benchmark has failed cases: "
                         f"{[n for n, r in b['arms'].items() if n in b.get('cases_used', b['arms']) and not r.get('ok')]}")
    if b.get("stats_workers") is None:
        raise SystemExit(f"{what}: the benchmark's host-memory rule refuses ({b.get('workers_why')}; budget "
                         f"{b.get('budget_gb')} GB, need {b.get('need_gb')})")
    bc = committed_or_exit(bench_path(hw), f"{what}: this card's benchmark", require_commit)
    why = checks_status(hw, sha)
    if why is not None:
        raise SystemExit(f"{what}: this card's CUDA CHECK output ({checks_output_path(hw)}) is {why}: run python -m "
                         f"explore_gpu.s75b_checks --device cuda and commit its output")
    cc = committed_or_exit(checks_output_path(hw), f"{what}: this card's CUDA CHECK output", require_commit)
    return b["stats_workers"], b["register_reg3"], bc, cc


# ── running arms ──────────────────────────────────────────────────────────────
def run_arm(arm, seeds, dev, hw, sha, st, meta_extra, register, workers, split_thr=None, log=print):
    """Each seed at R = 1 (FAILURES in the module docstring): a seed already ok, diverged or unable to run (on any card) is
    not run; an operational failure is retried once in this invocation; a divergence is never retried. Every failed
    attempt is logged in st["failures"][key] (attempt, invocation, kind, error). Records keyed arm|seed|sha|hw, with the
    host memory of the run (descriptions). Returns the seeds without an ok record."""
    T = task()
    missing = []
    for s in seeds:
        key = f"{arm}|{s}|{sha}|{hw}"
        if (st["runs"].get(key) or {}).get("ok"):
            continue
        status = seed_status(load_all(), arm, s, sha)
        if status in ("ok", "diverged", "cannot run"):
            log(f"  {arm} seed {s}: {status} (not run)")
            if status != "ok":
                missing.append(s)
            continue
        for attempt in (1, 2):
            log(f"  {arm} seed {s} (attempt {attempt}) ...")
            _JOBS.clear()
            ctx = guarded() if arm == "WIDE_SPLIT_G" else contextlib.nullcontext()
            with ctx:
                rec = gr.run_batched_icmc(T, arm_spec(arm), [s], ITERS, device=dev, split_thr=split_thr, eval_every=EVAL_EVERY,
                                          register=register if arm == "REG3" else "auto", stats_workers=workers, tag=arm)[0]
            rec.update(arm=arm, sha=sha, hw=hw, attempt=attempt, threads=torch.get_num_threads(),
                       host=dict(main_peak_gb=own_peak_gb(), meminfo=meminfo_gb(),
                                 worker_peak_gb=max((j["worker_gb"] for j in _JOBS if j.get("worker_gb")), default=None),
                                 stats_secs_max=max((j["worker_secs"] for j in _JOBS), default=None)), **meta_extra)
            st["runs"][key] = rec
            save_json(st, store_path(hw))
            if rec.get("ok"):
                e = rec["end"]
                log(f"    {arm} {s}: SET {e['set']:.4f} ROUTED {idense.routed(rec)}"
                    + (f" splits {rec.get('splits')}" if "splits" in rec else ""))
                break
            kind = failure_kind(rec.get("error"))
            st.setdefault("failures", {}).setdefault(key, []).append(dict(
                attempt=attempt, invocation=INVOCATION, kind=kind, error=str(rec.get("error"))[:600],
                at=time.strftime("%Y-%m-%d %H:%M:%S")))
            save_json(st, store_path(hw))
            log(f"    {arm} {s}: FAILED ({kind}) {rec.get('error')}")
            if kind == "diverged":                         # an outcome: never retried
                break
        if not (st["runs"].get(key) or {}).get("ok"):
            missing.append(s)
    return missing


def _dup_exit(dup, what):
    if dup:
        raise SystemExit(f"{what}: ok records of one arm and seed from two cards (arm, seed, cards): {dup}; the driver "
                         f"does not choose between them: the user decides")


def stage1(device, log=print, require_commit=True):
    global INVOCATION
    INVOCATION = f"stage 1 {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}"
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    sha = code_sha()[0]
    stops = earlier_stops(sha)
    if stops:
        raise SystemExit(f"stage 1: S75b stopped at stage 1 under code SHA {[r.get('sha') for _, r in stops]} "
                         f"({[os.path.basename(p) for p, _ in stops]}); a stage-1 STOP under any code is final (FAILURES)")
    st_all = load_all()
    _dup_exit(duplicates(st_all, sha, ("ORACLE_PILOT",)), "stage 1")
    oth = other_cards(st_all, sha, ("ORACLE_PILOT",), hw)
    if oth:
        raise SystemExit(f"stage 1: pilot records under code SHA {sha} exist from another card ({oth}): the pilot runs on "
                         f"one card")
    code_or_exit("stage 1", require_commit)
    workers, _, bench_commit, checks_commit = card_setup(dev, hw, sha, require_commit, "stage 1")
    T = task()
    chk = it.check_task(T)
    assert all(chk.values()), f"task CHECK failed: {chk}"
    st = load(hw)
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, changes=CHANGES_SINCE_SPEC,
                      stage1=dict(workers=workers, threads=torch.get_num_threads(), bench_commit=bench_commit,
                                  checks_commit=checks_commit, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    save_json(st, store_path(hw))
    log(f"[S75b stage 1] {T.label}; code SHA {sha}; {info}; task CHECK ok; stats workers {workers}")
    with runtime():
        missing = run_arm("ORACLE_PILOT", PILOT_SEEDS, dev, hw, sha, st, dict(stage=1), "auto", workers, log=log)
    st_all = load_all()
    pilot = runs_of(st_all, "ORACLE_PILOT", sha)
    d = pilot_of(st_all, sha)
    cards = {s: r.get("hw") for s, r in pilot.items()}
    fails = {k: v for k, v in st_all["failures"].items() if k.startswith("ORACLE_PILOT|") and k.split("|")[2] == sha}
    rec = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), sha=sha, machine=hw, cards=cards, device=info, config=CONFIG,
               iters=ITERS, pilot_seeds=list(PILOT_SEEDS), decision=d, missing=missing, failures=fails,
               results_file=os.path.relpath(store_path(hw), HERE), bench_commit=bench_commit, checks_commit=checks_commit,
               threads=torch.get_num_threads(), provenance=SPEC_PROVENANCE, changes=CHANGES_SINCE_SPEC,
               note=(f"Commit and push this record together with the pilot results file {os.path.relpath(store_path(hw), HERE)} "
                     f"before stage 2: --stage 2 refuses unless both are committed and the committed pilot records give this "
                     f"decision. The pilot ran on {hw} (the spec at {SPEC_COMMIT} assigned the GTX 1660 SUPER; the user "
                     f"allowed the RTX 3070 if it was free first)."))
    sets = d.get("sets") or {}
    lines = [f"# S75b stage-1 record (EXPLORATORY), {rec['written']}", "",
             f"- Pilot: ORACLE_PILOT, seeds {PILOT_SEEDS}, {ITERS:,} updates, on **{hw}** (code SHA {sha}; stats workers "
             f"{workers}; threads {rec['threads']}; benchmark commit {bench_commit}; CUDA CHECK commit {checks_commit}).",
             "- end.set: " + ", ".join(f"{s}: {v}" for s, v in sets.items()),
             "- Seed status (FAILURES): " + ", ".join(f"{s}: {v}" for s, v in (d.get("statuses") or {}).items())
             + (f"; failed attempts: {[(k, [(f['kind'], f['error'][:80]) for f in v]) for k, v in fails.items()]}"
                if fails else ""),
             f"- Proceed rule (mean >= {PROCEED_MEAN} and each >= {PROCEED_EACH}): {d.get('why')} -> "
             f"**{'PROCEED' if d['proceed'] else ('STOP' if d['complete'] else 'INCOMPLETE')}**"]
    if d["proceed"]:
        lines.append(f"- SPLIT_THR = min(end.set) - {THR_MARGIN} = {d['split_thr']!r}")
    elif d["complete"]:
        lines.append(f"- {stop_words(d)}. No other configuration.")
    lines += ["", rec["note"], "", "Provenance as git shows it (SPEC_PROVENANCE):"]
    lines += [f"- blind by git only as the author's statement: {x}" for x in SPEC_PROVENANCE["blind_by_git"]]
    lines += [f"- NOT BLIND BY GIT: {x}" for x in SPEC_PROVENANCE["not_blind_by_git"]]
    lines += ["", f"Changes since the approved spec ({SPEC_COMMIT}), not blind ({CHANGES_SINCE_SPEC['note']}):"]
    lines += [f"- {x}" for x in CHANGES_SINCE_SPEC["items"]]
    write_stage1_record(rec, lines)
    log("\n".join(lines))
    return d, missing


def stage2(device, machine, log=print, require_commit=True):
    global INVOCATION
    INVOCATION = f"stage 2 {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}"
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    if dev.type == "cuda" and MACHINE_GPU[machine] not in hw:
        raise SystemExit(f"--machine {machine} expects a {MACHINE_GPU[machine]}, this card is {hw}")
    sha = code_sha()[0]
    jp, _ = record_paths()
    if not os.path.exists(jp):
        raise SystemExit("no stage-1 record: run --stage 1 first")
    commit = committed_or_exit(jp, "the stage-1 record", require_commit)
    with open(jp) as f:
        rec1 = json.load(f)
    stops = earlier_stops(sha)
    if stops:
        raise SystemExit(f"S75b stopped at stage 1 under code SHA {[r.get('sha') for _, r in stops]}: a stage-1 STOP under "
                         f"any code is final (FAILURES)")
    st_all = load_all()
    _dup_exit(duplicates(st_all, sha), "stage 2")
    d = pilot_of(st_all, sha)
    if not d["complete"]:
        raise SystemExit("both pilot records under this code SHA are needed (" + d["why"] + ")")
    if (rec1.get("sha") != sha or rec1["decision"].get("split_thr") != d["split_thr"]
            or rec1["decision"].get("proceed") != d["proceed"]):
        raise SystemExit("the stage-1 record does not match the pilot records under this code SHA")
    if require_commit:
        pf = os.path.join(HERE, rec1.get("results_file") or "")
        cj = git_committed_json(pf) if rec1.get("results_file") else None
        if cj is None:
            raise SystemExit(f"the pilot results file ({rec1.get('results_file')}) is not committed: commit and push it")
        dc = pilot_of(cj, sha)
        if not dc["complete"] or dc.get("split_thr") != d["split_thr"] or dc["proceed"] != d["proceed"]:
            raise SystemExit("the committed pilot records do not give the stage-1 record's decision: commit and push the "
                             "pilot results file")
    if not d["proceed"]:
        raise SystemExit(f"{stop_words(d)} (S75b stops at stage 1: {d['why']})")
    oth = other_cards(st_all, sha, MACHINE_ARMS[machine], hw)
    if oth:
        raise SystemExit(f"--machine {machine}: records of its arms under code SHA {sha} exist from another card ({oth}); "
                         f"running them here would give two cards' records for one seed: the user decides")
    if machine == "fallback-3070":
        g = [(p, b) for card, (p, b) in benches(sha).items() if card and "GTX 1660" in card]
        if not g or bench_usable(g[0][1]):
            raise SystemExit("--machine fallback-3070 runs every arm on the RTX 3070 only when the GTX 1660's benchmark "
                             f"under code SHA {sha} exists and failed or refused (FAILURES)")
        committed_or_exit(g[0][0], "the GTX 1660's benchmark", require_commit)
    code_or_exit("stage 2", require_commit)
    workers, register, bench_commit, checks_commit = card_setup(dev, hw, sha, require_commit, "stage 2")
    if require_commit and rec1.get("machine") == hw and rec1.get("bench_commit") and bench_commit != rec1["bench_commit"]:
        raise SystemExit(f"this card's benchmark changed since the stage-1 record (commit {bench_commit}, the record names "
                         f"{rec1['bench_commit']})")
    st = load(hw)
    meta2 = dict(machine=machine, split_thr=d["split_thr"], stage1_commit=commit, bench_commit=bench_commit,
                 checks_commit=checks_commit, register_reg3=register, workers=workers, threads=torch.get_num_threads(),
                 provenance=SPEC_PROVENANCE, changes=CHANGES_SINCE_SPEC, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, stage2=meta2)
    save_json(st, store_path(hw))
    log(f"[S75b stage 2] machine {machine} ({hw}); arms {MACHINE_ARMS[machine]}; SPLIT_THR {d['split_thr']!r}; "
        f"stage-1 commit {commit}; REG3 register {register}; stats workers {workers}; threads {meta2['threads']}")
    missing = {}
    with runtime():
        for arm in MACHINE_ARMS[machine]:
            m = run_arm(arm, seeds_of(arm), dev, hw, sha, st, dict(stage=2, machine=machine), register, workers,
                        split_thr=d["split_thr"] if arm == "WIDE_SPLIT_G" else None, log=log)
            if m:
                missing[arm] = m
    if missing:
        stt = load_all()
        log(f"[S75b stage 2] runs without an ok record: "
            f"{ {a: {s: seed_status(stt, a, s, sha) for s in m} for a, m in missing.items()} } (FAILURES: an operational "
            f"failure is rerun by re-invoking --stage 2; a divergence is an outcome)")
    return missing


# ── report ────────────────────────────────────────────────────────────────────
def report(log=print, sha=None, out=None):
    st = load_all()
    sha = sha or code_sha()[0]
    STT = {a: statuses_of(st, a, sha) for a in ARMS}
    R = {a: ok_runs(st, a, sha) for a in ARMS}
    DIV = {a: [s for s, v in STT[a].items() if v == "diverged"] for a in ARMS}
    CANT = {a: [s for s, v in STT[a].items() if v == "cannot run"] for a in ARMS}
    dup = duplicates(st, sha)
    L = [f"# S75b (EXPLORATORY): {task().label}; {ITERS} updates; code SHA {sha}", "",
         "Provenance as git shows it: " + "; ".join(SPEC_PROVENANCE["blind_by_git"]) + ". NOT BLIND BY GIT: "
         + "; ".join(SPEC_PROVENANCE["not_blind_by_git"]) + ".", "",
         f"Changes since the approved spec ({SPEC_COMMIT}), not blind ({CHANGES_SINCE_SPEC['note']}): "
         + "; ".join(CHANGES_SINCE_SPEC["items"]), "",
         "Cards per arm: " + "; ".join(f"{a} {cards_of(rr)}" for a, rr in R.items() if rr)]
    bs = benches(sha)
    if bs:
        L.append("BENCHMARKS (this code): " + "; ".join(
            f"{card}: {'usable' if bench_usable(b) else ('refused (' + str(b.get('workers_why')) + ')' if b.get('all_ok') else 'failed ' + str([n for n in b.get('cases_used', b['arms']) if not b['arms'][n].get('ok')]))}"
            f" (REG3 {b.get('register_reg3')}, {b.get('stats_workers')} stats workers)" for card, (p, b) in bs.items()))
        cards = {("1660" if "GTX 1660" in c else "3070" if "RTX 3070" in c else c): bench_usable(b) for c, (p, b) in bs.items()}
        if {"1660", "3070"} <= set(cards) and not any(cards.values()):
            L.append("S75b STOPS BEFORE THE PILOT: B cannot be run on these cards (neither card's benchmark passed; FAILURES).")
    other = {}
    for k, r in st["runs"].items():
        sh = k.split("|")[2]
        if sh != sha:
            other[(sh, _card(k, r))] = other.get((sh, _card(k, r)), 0) + 1
    if other:
        L.append("NOT COUNTED (another code SHA): " + "; ".join(f"{n} records with SHA {sh} on {hw}"
                                                              for (sh, hw), n in sorted(other.items())))
    fl = [(k, f) for k, v in sorted(st["failures"].items()) if k.split("|")[2] == sha for f in v]
    if fl:
        L.append("FAILED ATTEMPTS (FAILURES: diverged = an outcome; operational = rerun on the same seed): " + "; ".join(
            f"{k.split('|')[0]} {k.split('|')[1]} on {k.split('|')[3]} attempt {f.get('attempt')} [{f.get('kind')}]: "
            f"{str(f.get('error'))[:120]}" for k, f in fl))
    if dup:
        L.append("DUPLICATE RECORDS (arm, seed, cards): " + "; ".join(f"{a} {s} {h}" for a, s, h in dup)
                 + ". The reading is incomplete until the user decides which card's records count.")
    L.append("")
    for p, r in stage1_records():
        if r.get("sha") != sha:
            dd = r.get("decision") or {}
            L.append(f"Stage 1 under earlier code {r.get('sha')} ({os.path.basename(p)}; never pooled): end.set "
                     f"{dd.get('sets')}; " + ("PROCEED" if dd.get("proceed") else "STOP (final)" if dd.get("complete")
                                             else "incomplete"))
    pil = R["ORACLE_PILOT"]
    d = pilot_of(st, sha)
    L.append(f"Stage 1 (pilot, never pooled): end.set {[(s, round(r['end']['set'], 4)) for s, r in sorted(pil.items())]}; "
             f"seed status {d.get('statuses')}; {d['why']}; " + (f"PROCEED, SPLIT_THR {d['split_thr']:.6f}" if d["proceed"] else
                                                                  stop_words(d) if d["complete"] else "incomplete"))
    ost = STT["ORACLE"]
    orc = [R["ORACLE"].get(s) for s in ORACLE_SEEDS]
    complete_orc = all(r is not None for r in orc)
    if DIV["ORACLE"]:
        vtag, valid, resolved = f"NOT VALID: ORACLE {DIV['ORACLE']} diverged (counts as below 0.9) -> UNTESTED", False, True
    elif any(r is not None and r["end"]["set"] < BAR for r in orc):
        vtag, valid, resolved = "NOT VALID -> UNTESTED", False, True
    elif CANT["ORACLE"]:
        vtag, valid, resolved = f"VALID cannot be established: ORACLE {CANT['ORACLE']} cannot run -> UNTESTED", False, True
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
        extra = (f" (diverged {DIV[a]})" if DIV[a] else "") + (f" (cannot run {CANT[a]})" if CANT[a] else "")
        if not rr:
            L.append(f"| {a} | {('0/' + str(n_)) if n_ else '-'}{extra} | | | | | | |")
            continue
        lo, hi = ic.wilson(k_, n_)
        cl = f"{k_}/{n_} [{lo:.2f}, {hi:.2f}] {ic.band(k_, n_)}{extra}"
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
    cant_read = {a: CANT[a] for a in READ_ARMS if CANT[a]}
    done = resolved and all(ns[a] == len(SEEDS) for a in READ_ARMS)
    if dup:
        tag = "incomplete: records from two cards (the user decides)"
    elif resolved and not valid:
        tag = "UNTESTED (descriptions only; never re-read under a later bar)"
    elif cant_read and resolved:
        tag = f"INCOMPLETE: read-arm seeds cannot run {cant_read} (descriptions only)"
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
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--host-ram-gb", type=float, default=None, help="the machine's RAM in GB (the user's figure)")
    ap.add_argument("--beside", default=None,
                    help="a string in the command line of the job running beside S75b on this machine, or 'none'")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sha", default=None, help="--report: the code SHA whose records count (default: this code's)")
    a = ap.parse_args()
    one_thread()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        report(sha=a.sha)
        return
    if a.bench:
        if a.host_ram_gb is None or a.beside is None:
            raise SystemExit("--bench needs --host-ram-gb (the machine's RAM, the user's figure) and --beside NAME|none")
        b = bench(a.device, a.host_ram_gb, a.beside)
        hw = b["device"]["gpu"] or b["device"]["cpu"]
        save_json(b, bench_path(hw))
        print(json.dumps({k: v for k, v in b.items() if k not in ("arms", "changes")}, indent=1, default=str))
        if b["stats_workers"] is None or not b["all_ok"]:
            raise SystemExit(f"the benchmark refuses: {b['workers_why']}; failed cases "
                             f"{[n for n in b['cases_used'] if not b['arms'][n].get('ok')]}")
        return
    if a.stage == 1:
        d, missing = stage1(a.device)
        sys.exit(1 if missing else 0)
    if a.stage == 2:
        if a.machine is None:
            raise SystemExit("--stage 2 needs --machine 1660, 3070, fallback or fallback-3070")
        missing = stage2(a.device, a.machine)
        sys.exit(1 if missing else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
