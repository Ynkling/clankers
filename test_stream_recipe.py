#!/usr/bin/env python
"""
test_stream_recipe.py — confirms on fresh seeds a recipe for four streams: spare channels
(k=16), plus a restart rule with an early accuracy check. It also takes a first descriptive look
at S=8. lr 1e-3 (SUB_LR, passed explicitly), conv 'layer' width 4. Everything below is fixed
before any result.

Run it directly:

    python test_stream_recipe.py
    python test_stream_recipe.py --also results/L/stream_recipe_results.json

BACKGROUND
test_stream_channels, both machines, seeds 220-239 (L ran e5a24ae):
- Claims:
  - X: K8, K16 and MG NOT SHOWN (A4k8 10/20, A4k16 10/20 vs the recorded A4k4 9/20).
  - L: all three SHOWN (13/20, 13/20 vs 5/20).
  - Pooled, post hoc: A4k16 vs A4k4 was 13 vs 4 discordant (p = 0.049).
- Consistent on both machines:
  - Streams partly merged in a channel at the end: 15 runs at k=4, 4 at k=8, 1 at k=16.
  - Non-stream splits (all streams on one channel by stream average; position or key splits; at
    chance 0.2-0.25 throughout): 11 -> 13 -> 16.
  - Every k=8 and k=16 binder had a one-to-one stream -> channel map. Median transitions were
    3600-8400 steps, against about 14400 at k=4.
- The margin >= 0.9 criterion undercounts routing at k > 4 (gate mass spreads over 6-9
  channels). Here a run is ROUTED if it has a one-to-one map and every stream's accuracy is
  >= 0.9.
- Post hoc, from both machines' k=8 and k=16 curves (80 runs):
  - "held-out accuracy >= 0.4 at step 4800" passed 38 runs, all 38 bound, and missed 8 late
    binders;
  - at step 6000 the same threshold passed 42, of which 41 bound.
  At k=4 no threshold worked, because two-stream merges plateau at 0.75. This test confirms the
  chosen check (4800, 0.4) on fresh seeds.

DESIGN
- Run path: test_stream_channels' (test_router_reliability.run_one with test_scale_axes's
  scale_stats: routing margin, multichannel eta^2 and the stream -> channel map at every
  evaluation, per-stream held-out accuracy at the end; conv_grad_norms; lr 1e-3), on each part's
  task. The k=16 perfect gates are built as test_stream_channels built ceiling4k8: the existing
  perfect_gate_general over the task's context tokens padded with token ids that never occur, so
  channels S..15 stay at 0 (CHECK 92).
- The check: at every evaluation up to step 4800 the held-out accuracy is read (tbo.evaluate on
  the same model and batch, so it is the value onset_run just recorded: test_router_reliability.
  run_attempt's recipe); at step 4800 test_router_reliability.make_decide(4800, 0.4) keeps the
  run or not. The plain arms record the decision and always continue.
- Every run in this test (all arms) counts onset_run's early-stop streak only at evaluations
  after step 4800 (onset_run's early_stop_after). Without it, a run that binds at 1200 would stop
  at 3600, before the check; with it, every run reaches its evaluation at 4800. It changes only
  when a run stops (a run bound by 4800 stops at 8400 at the earliest), not what it computes
  (CHECKs 90 and 91).
- The restart rule (the _R arms): at step 4800, continue if held-out accuracy >= 0.4; otherwise
  restart with seed s + 1000*j (j = 1..4), for at most 5 attempts. If none passes, the last
  attempt continues. Attempt 0 of A4k16_R and A4k4_R is the plain arm's run with the same seed:
  its record is reused when it passes; otherwise the trial runs attempts 1-4 (the plain run still
  continues to 28800 as the plain arm). A8k16_R has no plain arm; its trials start at attempt 0.
  Every attempt seed is disjoint from every earlier seed (CHECK 91).
- The plain and _R arms of the same seed share attempt 0 bit for bit (CHECK 90). A4k16 and A4k4
  of the same seed share every parameter except W_g, and the batches (test_stream_channels'
  CHECK 85), and so do attempts with the same seed; McNemar pairs by seed.

PART 1: S=4 (task_for(4, 4)), fresh seeds 240-259, up to 28800 steps per continued run.
- A4k16:       arm A, k=16, + conv (plain; test_stream_channels' A4k16).  Seeds 240-259 (20)
- A4k16_R:     A4k16 with the restart rule.                              Seeds 240-259 (20)
- A4k4:        arm A, k=4, + conv (plain; test_scale_axes's A4k4).        Seeds 240-259 (20)
- A4k4_R:      A4k4 with the restart rule (control: does the check need spare channels?).
                                                                         Seeds 240-259 (20)
- ceiling4k16: perfect gate, k=16 (stream s -> channel s), + conv. Validity; runs first.
                                                                         Seeds 240-242 (3)
If ceiling4k16 binds fewer than 2/3, Part 1 is UNTESTED and nothing else runs.

PART 2: S=8, descriptive only (task_for(4, 8): P=4, n_vals=16, length 99).
- A8k16_R:     arm A, k=16, + conv, with the same restart rule.  Seeds 240-244 (5)
- ceiling8k16: perfect gate, k=16, + conv. Runs first in Part 2. Seeds 240-241 (2)
- If ceiling8k16 binds 0/2, A8k16_R is skipped (said so).
- Drop rule: if the printed projection exceeds 12 h on the workers, Part 2 is dropped (said so).
  The projection is the worst case (every continued run to 28800, every trial needing all of its
  attempts) over both parts, as in test_curriculum_confirm.

DEFINITIONS
- BOUND: transition not None (held-out accuracy >= 0.95 held to the end).
- Stream -> channel map, one_to_one, shared (streams on the most crowded channel), eff_ch and
  the routing margin: as in test_stream_channels.
- ROUTED: a one-to-one map at the end AND every stream's held-out accuracy at the end >= 0.9.
- MERGED: shared >= 2 at the end (this includes every stream on one channel).
- Channels used: >= 5% of any stream's mean read gate at value positions (test_stream_channels).
- The check passes: held-out accuracy >= 0.4 at step 4800.

CLAIMS (Part 1; outcome BOUND)
- VALID: ceiling4k16 binds >= 2/3, and the k=16 gate fits the perfect 4-way routing (CHECK 45's
  recipe, test_router_reliability.gate_fit_k: argmax >= 0.99; CHECK 92).
- R1 THE RECIPE IS RELIABLE: A4k16_R's band: RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9,
  NEVER 0.
- R2 THE CHECK IS PRECISE: over every A4k16 / A4k16_R attempt that passed the check at 4800 (each
  distinct attempt once: a reused attempt 0 is the plain run), the fraction that went on to bind.
  PRECISE if >= 0.9, with the Wilson 95% interval printed; NOT PRECISE below; UNTESTABLE if fewer
  than 10 attempts passed.
- R3 SPARE CHANNELS MAKE RESTARTS WORK: A4k16_R vs A4k4_R. Fisher one-sided (A4k16_R higher);
  exact McNemar (one-sided, the same direction, paired by seed) printed alongside.
- R4 REPLICATION OF K16 ON FRESH SEEDS: A4k16 vs A4k4, plain arms. Exact McNemar, one-sided
  (A4k16 higher).
- R3 and R4 are SHOWN if p < 0.05, else NOT SHOWN. If not VALID, every claim is UNTESTED.
- Readings, printed verbatim where they apply:
  - R1 RELIABLE and R2 PRECISE: "spare channels plus an early check make four streams reliable:
    the recipe carries to S=4."
  - R3 SHOWN: "the early check works only with spare channels: at k=4 merged runs pass it and
    stall."
  - R1 MINORITY or NEVER: "the recipe does not carry to four streams."
Fisher is test_router_confirm's; the one-sided McNemar is test_conv_lr's.

DIAGNOSTICS, not part of the verdict
- Per seed: the plain outcomes (A4k4, A4k16); the _R outcomes, attempts used and each attempt's
  accuracy at 4800; transitions; the end stream -> channel map, one_to_one, shared, eff_ch and
  per-stream accuracy; channels used.
- Check performance per arm: passes, passes that bound, and binders missed; the same counts at
  the alternatives (6000, 0.4) and (4800, 0.5), on the plain arms, for reference only.
- Failure classes: MERGED (by how many streams share), non-stream (POSITION, KEY, OTHER),
  collapsed.
- A4k4_R: how many continued attempts were merges that passed the check (A4k16_R alongside).
- Part 2, per seed: attempts used, bound, transition, map, one_to_one, per-stream accuracy and the
  final accuracy.

CHECKS (printed before any training): test_stream_channels' verification (which runs every
earlier test's), then
  90 every arm's optimizer runs at lr 1e-3 (step pre-hook, through run_job; for the _R arms
     through a trial whose attempts all fail a check at step 1); every arm dict equals its source
     arm's except the named fields; the plain and _R arms of the same seed share attempt 0 bit
     for bit (the weights right after step 4800, and every evaluation up to it; A4k16/A4k16_R and
     A4k4/A4k4_R, seed 3); a plain run here equals test_stream_channels.run_job's record except
     the fields the check adds (A4k16 seed 3, 1200 steps, check at 1200);
  91 the restart rule reads only the held-out accuracy at step 4800 (make_decide's closure);
     attempt seeds are s + 1000*j and disjoint from every earlier seed; on synthetic curves
     (tbo.evaluate stubbed) a trial continues and restarts as specified, including "none passes
     -> the last attempt continues", abandoned attempts stop at the check, and a run at 1.0
     from its first evaluation still reaches the check (without early_stop_after it would stop
     before it); a plain run that fails the check continues;
  92 task_for(4, 8) over 10,000 sequences: each (stream, key) once, S distinct values per key,
     the query answered by its own stream's value, length 99; the k=16 perfect gate zeroes every
     cross-stream score at every layer at S=4 and at S=8 and leaves channels S..15 at 0; this
     test's builder equals test_stream_channels.builder_for at S=4; the margin at S=8, k=16 is 1
     on the perfect gate and 0 on the uniform gate; the recurrent k=16 gates' fits to the perfect
     routing at S=4 (for VALID) and S=8 (recorded), not asserted;
  93 ROUTED (one-to-one map and every stream >= 0.9) is true on a perfect-gate model (ceiling4k16
     trained 1200 steps) and false on a uniform-gate model (the same run path), and on synthetic
     records as defined; the Wilson interval reproduces hand values;
  94 a worker's run is bit-identical to the same run in the main process (A4k16_R through one
     restart, 6000 steps: attempt 0 failed and attempt 1 passed by forced decisions, so the
     trial path is exercised whatever the accuracies are).

OUTPUT AND LOGISTICS. Parallel single-thread workers; the wall clock is projected before training
(worst case). Results persist atomically to stream_recipe_results.json (gitignored; copied into
results/X/ after the run). Flags: --workers, --force, --results, --also (a second machine's file,
for pooled counts).
"""

import argparse
import copy
import hashlib
import inspect
import itertools
import json
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
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import time_per_step, fmt_step, EVAL_EVERY
from test_multilayer_binding import D, probe_batch
import test_channel_binding as tcb
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater
from test_router_curriculum import (
    ARCH, SUB_LR, get, load_store, init_worker, makespan, cpu_model, same_weights, save_results,
    TAU_END, TIME_STEPS,
)
from test_router_discovery import model_fn
import test_router_reliability as trr
from test_router_reliability import run_one, strip_all, med, gate_fit_k
from test_router_layout import routing_stats
from test_short_conv import conv_grad_norms, CONV_K
from test_conv_lr import bound_r, mcnemar_greater
import test_load_curriculum as tlc
import test_curriculum_confirm as tcc
import test_scale_axes as tsa
from test_scale_axes import scale_stats, routing_k, fail_class_k, TASK44
import test_stream_channels as tsch
from test_stream_channels import shared_max, merged, used, med_int, mstr, fmt, at, band

# ── Settings (fixed before any result) ───────────────────────────────────────
LR = SUB_LR                                # 1e-3, passed explicitly to every run
MAX_ITERS = 28800
T_CHECK, A_CHECK = 4800, 0.4               # the restart rule's check
R_MAX, STRIDE = 5, 1000                    # at most 5 attempts, seeds s + 1000*j
REAL = dict(t_check=T_CHECK, a_check=A_CHECK, total=MAX_ITERS, eval_every=EVAL_EVERY)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
CEIL4_MIN, CEIL8_MIN = 2, 1                # ceiling4k16 >= 2/3 (VALID); ceiling8k16 >= 1/2 (else A8k16_R skipped)
FIT_MIN = 0.99
ALPHA = 0.05
PRECISE_FRAC, PRECISE_N = 0.9, 10          # R2
ROUTED_ACC = 0.9                           # ROUTED: one-to-one and every stream >= 0.9
ALT_CHECKS = ((6000, 0.4), (4800, 0.5))    # reference only
DROP_H = 12.0                              # Part 2 is dropped if the projection exceeds this
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "stream_recipe_results.json"

TASK48 = tcb.task_for(4, 8)                # S=8, P=4, n_vals=16, n_q=1, length 99
TASKS = {"P4S4": TASK44, "P4S8": TASK48}
A4K4 = tsa.ARM["A4k4"]
CEIL = tsa.ARM["ceiling4k4"]
ARMS1 = [
    dict(CEIL, key="ceiling4k16", n_ch=16, ctx_pad=16, label="ceiling4k16 perfect gate k=16 (4-15 unused) + conv"),
    dict(A4K4, key="A4k16", n_ch=16, label="A4k16       arm A k=16 + conv"),
    dict(A4K4, key="A4k16_R", n_ch=16, restart=True, label="A4k16_R     arm A k=16 + conv + restart rule"),
    dict(A4K4, label="A4k4        arm A k=4 + conv"),
    dict(A4K4, key="A4k4_R", restart=True, label="A4k4_R      arm A k=4 + conv + restart rule"),
]
ARMS2 = [
    dict(CEIL, key="ceiling8k16", n_ch=16, ctx_pad=16, task="P4S8",
         label="ceiling8k16 perfect gate k=16 (8-15 unused) + conv, S=8"),
    dict(A4K4, key="A8k16_R", n_ch=16, task="P4S8", restart=True,
         label="A8k16_R     arm A k=16 + conv + restart rule, S=8"),
]
ARM = {a["key"]: a for a in ARMS1 + ARMS2}
PART1 = tuple(a["key"] for a in ARMS1)
PART2 = tuple(a["key"] for a in ARMS2)
KEYS = PART1 + PART2
PLAIN = {"A4k16_R": "A4k16", "A4k4_R": "A4k4"}
S20 = tuple(range(240, 260))
SEEDS = dict(ceiling4k16=tuple(range(240, 243)), A4k16=S20, A4k16_R=S20, A4k4=S20, A4k4_R=S20,
             ceiling8k16=tuple(range(240, 242)), A8k16_R=tuple(range(240, 245)))
READINGS = {
    "R1+ R2+": "spare channels plus an early check make four streams reliable: the recipe carries to S=4.",
    "R3+": "the early check works only with spare channels: at k=4 merged runs pass it and stall.",
    "R1-": "the recipe does not carry to four streams.",
}


def attempt_seeds(s):
    return [s + STRIDE * j for j in range(R_MAX)]


def earlier_seeds():
    """Every seed an earlier test used: base seeds 0-239 (test_stream_channels and test_scale_axes
    used 220-239), test_router_reliability's trial seeds 1000-1149, and the attempt seeds of
    test_load_curriculum and test_curriculum_confirm (from their own tables)."""
    out = set(tlc.EARLIER_SEEDS) | set(range(240))
    out |= {x for s in tlc.SEEDS["CUR_A_R"] for x in tlc.attempt_seeds(s)}
    out |= {x for s in tcc.SEEDS["CUR_lo_R"] for x in tcc.attempt_seeds(s)}
    for mod in (tcc, tsa, tsch):
        out |= {x for v in mod.SEEDS.values() for x in v}
    return out


def ranges(xs):
    xs = sorted(xs)
    out, lo, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x != prev + 1:
            out.append((lo, prev))
            lo = x
        prev = x
    out.append((lo, prev))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


def padded_task(task, k):
    """test_stream_channels.padded_task for any task: its context tokens padded with k - S token
    ids that never occur (-1, -2, ...), so perfect_gate_general is one-hot over k channels with
    channels S..k-1 unused. Used only by the model builder."""
    t = copy.copy(task)
    t.ctx_tokens = tuple(task.ctx_tokens) + tuple(-(i + 1) for i in range(k - task.S))
    return t


def builder_for(a):
    task = TASKS[a["task"]]
    if a.get("ctx_pad"):
        task = padded_task(task, a["ctx_pad"])
    return lambda aa, seed: model_fn(task, aa, seed)


def wilson(k, n, z=1.959963984540054):
    """The Wilson score interval for k successes of n (95% at the default z)."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ── Runs: one attempt, and a trial ───────────────────────────────────────────
def run_attempt(a, seed, sched, decide=None, abandon=False, keep=None):
    """One run of arm a from `seed` on test_stream_channels' run path, on the arm's task, with the
    early-stop streak counted only after t_check. check(): at each evaluation up to t_check the
    held-out accuracy is read (test_router_reliability.run_attempt's recipe); at t_check,
    decide(step, acc); with abandon=True a failing attempt stops there (trr.Abandon)."""
    task = TASKS[a["task"]]
    tc = sched["t_check"]
    decide = trr.make_decide(tc, sched["a_check"]) if decide is None else decide
    info = dict(early=[])

    def check(m, step, data):
        if step > tc:
            return
        acc, el = tbo.evaluate(m, task, data)
        info["early"].append([step, acc, el])
        if step == tc:
            info["acc_check"] = acc
            info["passed"] = bool(decide(step, acc))
            if abandon and not info["passed"]:
                raise trr.Abandon(f"held-out accuracy {acc:.4f} at step {step}")

    rec = run_one(a, seed, sched["total"], sched["eval_every"], check=check, keep=keep, task=task,
                  stats_fn=scale_stats, grad_fn=conv_grad_norms, lr=LR, builder=builder_for(a),
                  run_kw=dict(early_stop_after=tc))
    rec.update(early=info["early"], acc_check=info.get("acc_check"), passed=info.get("passed"))
    return rec


def run_trial(a, seed, sched, start=0, decide=None):
    """A trial from attempt `start`: attempts s + 1000*j in order; a failing attempt is abandoned
    at t_check, except the last (j = R_MAX-1), which continues (test_load_curriculum.run_trial's
    loop, on this test's run path)."""
    decide = trr.make_decide(sched["t_check"], sched["a_check"]) if decide is None else decide
    ts0 = time.time()
    attempts = []
    for j in range(start, R_MAX):
        sj, last = seed + STRIDE * j, j == R_MAX - 1
        rec = run_attempt(a, sj, sched, decide=decide, abandon=not last)
        cut = rec["passed"] is False and not last
        attempts.append(dict(j=j, seed=sj, acc_check=rec["acc_check"], passed=rec["passed"], early=rec["early"],
                             steps=sched["t_check"] if cut else rec.get("stopped_at")))
        if cut:
            continue
        rec["trial"] = dict(start=start, attempts=attempts, used=j + 1, continued=sj, reused=False,
                            secs=time.time() - ts0)
        return rec
    raise AssertionError("unreachable")


class Forced:
    """CHECK 94 only: a decide that returns preset decisions in order (at t_check only)."""
    def __init__(self, seq, t_check):
        self.seq, self.t_check = list(seq), t_check

    def __call__(self, step, acc):
        assert step == self.t_check
        return self.seq.pop(0)


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_job(sp):
    a = sp.get("arm_dict") or ARM[sp["arm"]]
    sched = sp["sched"]
    decide = Forced(sp["force"], sched["t_check"]) if sp.get("force") else None
    if sp.get("trial"):
        return run_trial(a, sp["seed"], sched, start=sp["start"], decide=decide)
    return run_attempt(a, sp["seed"], sched, decide=decide)


def spec(key, seed, sched, trial=False, start=0):
    sp = dict(arm=key, seed=seed, sched=dict(sched))
    if trial:
        sp.update(trial=True, start=start)
    return sp


def lr_job(key):
    """CHECK 90: the optimizer lr at every step through run_job (step pre-hook): 2 steps for a
    plain arm; for an _R arm, a trial with the check at step 1 of 2 (every attempt fails it, so
    attempts 0-3 stop at step 1 and the last continues to step 2)."""
    seen, n_opt = [], [0]

    def pre(opt, args, kw):
        if not getattr(opt, "_c90", False):
            opt._c90 = True
            n_opt[0] += 1
        seen.append(sorted({g["lr"] for g in opt.param_groups}))
    h = register_optimizer_step_pre_hook(pre)
    try:
        if ARM[key].get("restart"):
            r = run_job(spec(key, 3, dict(t_check=1, a_check=A_CHECK, total=2, eval_every=1), trial=True))
        else:
            r = run_job(spec(key, 3, dict(REAL, total=2)))
    finally:
        h.remove()
    return seen, n_opt[0], r["ok"], (r.get("trial") or {}).get("used")


def digest(snap):
    h = hashlib.sha256()
    for n in sorted(snap):
        h.update(n.encode())
        h.update(bytes(snap[n].detach().contiguous().view(-1).view(torch.uint8).tolist()))
    return h.hexdigest()[:16]


def share_job(key, seed):
    """CHECK 90: arm `key`'s attempt 0 of `seed` through the check at 4800 — the plain path for a
    plain arm, the trial's attempt path (abandoning if it fails) for an _R arm: a digest of the
    weights right after step 4800, and every evaluation up to it."""
    a = ARM[key]
    keep = {"at": T_CHECK}
    rec = run_attempt(a, seed, dict(REAL, total=T_CHECK), abandon=bool(a.get("restart")), keep=keep)
    return dict(digest=digest(keep["snap"]) if "snap" in keep else None, early=rec["early"],
                curve=[c for c in (rec.get("curve") or []) if c[0] <= T_CHECK], acc_check=rec["acc_check"],
                passed=rec["passed"], ok=rec["ok"])


def time_job(key):
    """test_scale_axes.time_job's recipe on the arm's own task."""
    a = ARM[key]
    task = TASKS[a["task"]]
    mk = builder_for(a)(a, 0)
    per = time_per_step(task, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(task)
    t0 = time.time()
    tbo.evaluate(m, task, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


# ── Verification ─────────────────────────────────────────────────────────────
def cross_scores(a, task, gseed):
    """CHECK 6's test on arm a (conv on, random weights): the max |cross-stream score| and max
    |same-stream score| per layer, and the read gate."""
    x = task.make_batch(16, torch.Generator().manual_seed(gseed))[0][:, :-1]
    mdl = builder_for(a)(a, 6)().eval()
    with torch.no_grad():
        mdl.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(gseed + 1)))
    mdl.attn.record = []
    with torch.no_grad():
        gr = mdl(x, TAU_END)[2]
    lab = task.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s_[:, 0][cross].abs().max().item() for s_ in mdl.attn.record]
    live = [s_[:, 0][~cross].abs().max().item() for s_ in mdl.attn.record]
    mdl.attn.record = None
    return maxc, live, gr


def verify(pool):
    print("test_stream_channels.py's verification (which runs test_scale_axes's, and so on down to")
    print(f"test_multilayer_binding's), with its default --prev ({tsch.PREV_FILE}):")
    tsch.verify(pool, tsch.PREV_FILE)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f90 = {k: pool.submit(lr_job, k) for k in KEYS}
    s1200 = dict(REAL, total=EVAL_EVERY, t_check=EVAL_EVERY)
    f90d = (pool.submit(run_job, spec("A4k16", 3, s1200)), pool.submit(tsch.run_job, tsch.spec("A4k16", 3, EVAL_EVERY)))
    uni16 = dict(ARM["A4k16"], key="uniform4k16", gate="uniform")
    f93 = {"ceiling4k16": pool.submit(run_job, spec("ceiling4k16", 1, s1200)),
           "uniform4k16": pool.submit(run_job, dict(spec("uniform4k16", 1, s1200), arm_dict=uni16))}
    f90s = {k: pool.submit(share_job, k, 3) for k in ("A4k16", "A4k16_R", "A4k4", "A4k4_R")}
    f94 = pool.submit(run_job, dict(spec("A4k16_R", 1, dict(REAL, total=6000), trial=True), force=[False, True]))

    print(f"CHECK 90 every arm runs at lr {LR}; the arms are the source arms; the plain and _R arms of a seed share "
          f"attempt 0:")
    good = True
    for k in KEYS:
        seen, n_opt, rok, used_ = f90[k].result()
        lrs = sorted({tuple(x) for x in seen})
        if ARM[k].get("restart"):
            g = lrs == [(LR,)] and rok and n_opt == R_MAX and len(seen) == R_MAX + 1 and used_ == R_MAX
            how = f"through a trial ({n_opt} optimizers, {len(seen)} steps, attempts used {used_})"
        else:
            g = lrs == [(LR,)] and rok and len(seen) == 2
            how = f"over {len(seen)} steps"
        good &= g
        print(f"     (a) {k:<11} optimizer lr at every step {[list(x) for x in lrs]} {how}  -> {'OK' if g else 'WRONG'}")
    src = [("ceiling4k16", tsch.ARM["ceiling4k8"], "test_stream_channels' ceiling4k8", ("key", "label", "n_ch", "ctx_pad")),
           ("A4k16", tsch.ARM["A4k16"], "test_stream_channels' A4k16", ("label",)),
           ("A4k4", tsa.ARM["A4k4"], "test_scale_axes's A4k4", ("label",)),
           ("A4k16_R", ARM["A4k16"], "A4k16", ("key", "label", "restart")),
           ("A4k4_R", ARM["A4k4"], "A4k4", ("key", "label", "restart")),
           ("ceiling8k16", ARM["ceiling4k16"], "ceiling4k16", ("key", "label", "task")),
           ("A8k16_R", ARM["A4k16_R"], "A4k16_R", ("key", "label", "task"))]
    for k, ref, name, drop in src:
        same = ({x: v for x, v in ARM[k].items() if x not in drop} == {x: v for x, v in ref.items() if x not in drop})
        good &= same
        print(f"     (b) {k:<11} = {name:<33} except {', '.join(drop)}: {same}")
    sh = {k: f.result() for k, f in f90s.items()}
    for kp, kr in (("A4k16", "A4k16_R"), ("A4k4", "A4k4_R")):
        p_, r_ = sh[kp], sh[kr]
        g = (p_["digest"] is not None and p_["digest"] == r_["digest"] and p_["early"] == r_["early"]
             and p_["curve"] == p_["early"] and len(p_["early"]) == T_CHECK // EVAL_EVERY
             and p_["acc_check"] == r_["acc_check"] and p_["passed"] == r_["passed"] and p_["ok"])
        good &= g
        print(f"     (c) seed 3, through step {T_CHECK}: {kp} (plain path) vs {kr} (attempt 0 of its trial, "
              f"{'continued' if r_['passed'] else 'abandoned'}): weights right after step {T_CHECK} equal "
              f"{p_['digest'] == r_['digest']} ({p_['digest']})   evaluations equal {p_['early'] == r_['early']}   "
              f"accuracy read == the recorded curve {p_['curve'] == p_['early']}   acc@{T_CHECK} "
              f"{fmt(p_['acc_check'], 4)} (passed {p_['passed']})  -> {'SHARED' if g else 'DIFFERS'}")
    mine, theirs = f90d[0].result(), f90d[1].result()
    added = ("early", "acc_check", "passed")
    same = strip_all({x: v for x, v in mine.items() if x not in added}) == strip_all(theirs)
    g = same and mine["ok"] and mine["early"] == mine["curve"] and mine["passed"] is not None
    good &= g
    print(f"     (d) this test's plain run of A4k16 (seed 3, {EVAL_EVERY} steps, check at {EVAL_EVERY}) == "
          f"test_stream_channels.run_job's record (in workers), except the fields the check adds {added}: {same}   "
          f"accuracy read == curve {mine['early'] == mine['curve']}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print("CHECK 91 the restart rule:")
    dec = trr.make_decide(T_CHECK, A_CHECK)
    params = list(inspect.signature(dec).parameters)
    cv = inspect.getclosurevars(dec)
    good_a = (params == ["step", "acc"] and not cv.globals and not cv.builtins
              and cv.nonlocals == {"t_check": T_CHECK, "a_check": A_CHECK})
    print(f"     (a) decide = test_router_reliability.make_decide({T_CHECK}, {A_CHECK}): parameters {params}; it reads no "
          f"global name ({dict(cv.globals)}) and only the constants {dict(cv.nonlocals)}: {good_a}")
    earlier = earlier_seeds()
    good_b = True
    for part, keys in (("Part 1", ("A4k16_R", "A4k4_R")), ("Part 2", ("A8k16_R",))):
        base = sorted({s for k in keys for s in SEEDS[k]})
        new = [x for s in base for x in attempt_seeds(s)[1:]]
        g = (len(set(new)) == len(new) == len(base) * (R_MAX - 1) and not (set(new) | set(base)) & earlier
             and all(x == s + STRIDE * j for s in base for j, x in enumerate(attempt_seeds(s))))
        good_b &= g
        print(f"     (b) {part}: base seeds {ranges(base)}; attempt seeds s + {STRIDE}*j (j = 1..{R_MAX - 1}): {ranges(new)}; "
              f"{len(set(new))} distinct, all disjoint from every earlier seed: {g}")
    print(f"         (earlier seeds: {ranges(earlier)})")
    real_eval = tbo.evaluate
    tiny = dict(t_check=8, a_check=A_CHECK, total=20, eval_every=2)
    cnt = {"opt": None, "n": 0, "k": -1, "steps": []}

    def pre(opt, args, kw):
        if cnt["opt"] is not opt:
            cnt["opt"], cnt["n"], cnt["k"] = opt, 0, cnt["k"] + 1
            cnt["steps"].append(0)
        cnt["n"] += 1
        cnt["steps"][-1] += 1

    def run_synth(script, fn):
        cnt.update(opt=None, n=0, k=-1, steps=[])
        tbo.evaluate = lambda model, task, d: (script(cnt["k"], cnt["n"]), 0.0)
        h = register_optimizer_step_pre_hook(pre)
        try:
            return fn(), list(cnt["steps"])
        finally:
            h.remove()
            tbo.evaluate = real_eval
    cases = [("passes at once", lambda j, st: 0.45 if st == 8 else (0.3 if st < 8 else 0.6),
              dict(calls=[(8, 0.45)], used=1, continued=3, steps=[20])),
             ("restarts, then passes", lambda j, st: (0.5 if j == 2 else 0.3) if st == 8 else 0.2,
              dict(calls=[(8, 0.3), (8, 0.3), (8, 0.5)], used=3, continued=2003, steps=[8, 8, 20])),
             ("none passes", lambda j, st: 0.3,
              dict(calls=[(8, 0.3)] * R_MAX, used=R_MAX, continued=4003, steps=[8, 8, 8, 8, 20])),
             ("binds before the check", lambda j, st: 1.0,
              dict(calls=[(8, 1.0)], used=1, continued=3, steps=[14]))]
    good_c = True
    for name, script, want in cases:
        calls, rule = [], trr.make_decide(tiny["t_check"], A_CHECK)

        def spy(step, acc, rule=rule, calls=calls):
            calls.append((step, acc))
            return rule(step, acc)
        tr, steps = run_synth(script, lambda: run_trial(ARM["A4k16_R"], 3, tiny, start=0, decide=spy))
        t_ = tr["trial"]
        g = (calls == want["calls"] and t_["used"] == want["used"] and t_["continued"] == want["continued"]
             and steps == want["steps"] and [x["seed"] for x in t_["attempts"]] == attempt_seeds(3)[:want["used"]]
             and tr["ok"] and tr["stopped_at"] == want["steps"][-1]
             and [x["steps"] for x in t_["attempts"]] == want["steps"])
        good_c &= g
        print(f"     (c) synthetic '{name}' (check at step 8 of 20, evaluations every 2): decide saw {calls}; attempts used "
              f"{t_['used']}, seeds {[x['seed'] for x in t_['attempts']]}, steps per attempt {steps}, continued "
              f"{t_['continued']}  -> {'OK' if g else 'WRONG'}")
    (_, curve_ns, _), _ = run_synth(lambda j, st: 1.0, lambda: tbo.onset_run(
        TASK44, builder_for(ARM["A4k16"])(ARM["A4k16"], 3), 3, max_iters=20, eval_every=2, data=eval_batch(TASK44), lr=LR))
    g_d = curve_ns[-1][0] == 6
    print(f"     (d) the same 1.0-from-the-start curve without early_stop_after (onset_run's default): stops at step "
          f"{curve_ns[-1][0]}, before the check at 8: {g_d}  -> {'OK' if g_d else 'WRONG'}")
    rp, steps = run_synth(lambda j, st: 0.3, lambda: run_attempt(ARM["A4k16"], 3, tiny))
    g_e = rp["ok"] and rp["passed"] is False and rp["stopped_at"] == 20 and steps == [20]
    print(f"     (e) a plain run failing the check (synthetic 'none passes'): passed {rp['passed']}, continued to step "
          f"{rp['stopped_at']} of 20  -> {'OK' if g_e else 'WRONG'}")
    good = good_a and good_b and good_c and g_d and g_e
    ok &= good
    print()

    print("CHECK 92 task_for(4, 8), and the k=16 perfect gates:")
    t, N = TASK48, 10_000
    S, P = t.S, t.P
    tok, qs = t.make_batch(N, torch.Generator().manual_seed(92))
    j = torch.arange(S * P)
    ctx, key, val = tok[:, 3 * j], tok[:, 3 * j + 1] - S, tok[:, 3 * j + 2]
    once = bool((torch.sort(ctx * P + key, 1).values == torch.arange(S * P)).all())
    kv = key.view(N, P, S)
    grouped = bool((kv == kv[..., :1]).all()) and bool((torch.sort(ctx.view(N, P, S), -1).values == torch.arange(S)).all())
    distinct = bool((torch.sort(val.view(N, P, S), -1).values.diff(dim=-1) != 0).all())
    n = S * P
    qc, qk, qv = tok[:, 3 * n], tok[:, 3 * n + 1] - S, tok[:, 3 * n + 2]
    hit = (ctx == qc[:, None]) & (key == qk[:, None])
    answered = (bool((hit.sum(1) == 1).all()) and bool((val[hit] == qv).all())
                and bool((qc == qs.reshape(-1)).all()))
    voc = t.vocab == S + P + t.n_vals and bool((tok >= 0).all()) and bool((tok < t.vocab).all())
    length = t.L == 99 and tok.shape[1] == 99 and t.qpos == [97]
    g1 = once and grouped and distinct and answered and voc and length
    print(f"     (a) over {N:,} sequences (generator seed 92): each (stream, key) once: {once}   grouped (each key's S triples "
          f"adjacent, one per stream): {grouped}   S distinct values per key: {distinct}")
    print(f"         the query answered by its own stream's value: {answered}   vocabulary {t.vocab} = S+P+n_vals = "
          f"{S}+{P}+{t.n_vals}, token ids in range: {voc}   length {t.L}, query position {t.qpos}: {length}")
    print(f"         e.g. {tlc.tps_decode(tok[0], t)}")
    g2 = True
    for k, task, gs in (("ceiling4k16", TASK44, 920), ("ceiling8k16", TASK48, 922)):
        maxc, live, gr = cross_scores(ARM[k], task, gs)
        unused = gr.shape[-1] == 16 and bool((gr[..., task.S:] == 0).all())
        g = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0 and unused
        g2 &= g
        print(f"     (b) {k}, random conv weights: layers seen {len(maxc)} (n_layer={ARCH['n_layer']}); read gate shape "
              f"{tuple(gr.shape)}, channels {task.S}-15 exactly 0: {unused}")
        for i, (c, l_) in enumerate(zip(maxc, live)):
            print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l_:.2e}")
    x = TASK44.make_batch(16, torch.Generator().manual_seed(924))[0][:, :-1]
    a = ARM["ceiling4k16"]
    m_mine, m_tsch = builder_for(a)(a, 3)().eval(), tsch.builder_for(a)(a, 3)().eval()
    with torch.no_grad():
        dl = (tbo.logits_of(m_mine, x) - tbo.logits_of(m_tsch, x)).abs().max().item()
    a = ARM["A4k16"]
    sw = same_weights(builder_for(a)(a, 3)(), tsch.builder_for(a)(a, 3)())
    g3 = dl == 0.0 and sw
    print(f"         this test's builder vs test_stream_channels.builder_for at S=4 (seed 3): ceiling4k16 max |logits diff| "
          f"{dl:.1e}; A4k16 weights equal {sw}: {g3}")
    probe = probe_batch(TASK48, 3)
    ap, au = ARM["ceiling8k16"], dict(ARM["A8k16_R"], gate="uniform")
    mp_, mu_ = builder_for(ap)(ap, 3)(), builder_for(au)(au, 3)()
    rp_, ru_ = routing_stats(mp_, TASK48, probe)["margin"], routing_stats(mu_, TASK48, probe)["margin"]
    rk = routing_k(mp_, TASK48, probe)
    g4 = abs(rp_ - 1.0) < 1e-6 and abs(ru_) < 1e-6 and rk["ch_map"] == list(range(8)) and rk["one_to_one"]
    print(f"     (c) S=8, k=16: margin on the perfect gate {rp_:.6f} (want 1), on the uniform gate {ru_:.6f} (want 0); "
          f"perfect gate's map {rk['ch_map']} one-to-one {rk['one_to_one']}: {g4}")
    fits = {}
    for k, task, what in (("A4k16", TASK44, "4-way routing (for VALID)"), ("A8k16_R", TASK48, "8-way routing (Part 2, recorded)")):
        mse, am = gate_fit_k(task, ARM[k])
        fits[k] = dict(mse=mse, argmax=am)
        print(f"     (d) {k}'s recurrent k=16 gate fit to the perfect {what} (test_router_reliability.gate_fit_k): mse "
              f"{mse:.2e}   argmax match {am:.4f}  -> {'FITS' if am >= FIT_MIN else 'DOES NOT FIT'} (>= {FIT_MIN}; not asserted)")
    good = g1 and g2 and g3 and g4
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 93 ROUTED (a one-to-one map and every stream's accuracy >= {ROUTED_ACC}), and the Wilson interval:")
    synth = [([0, 1, 2, 3], [1.0, 1.0, 1.0, 0.95], True), ([0, 1, 2, 3], [1.0, 1.0, 1.0, 0.85], False),
             ([0, 0, 1, 2], [1.0, 1.0, 1.0, 1.0], False), ([5, 1, 9, 12], [0.9, 0.9, 0.9, 0.9], True),
             (list(range(8)), [0.92] * 8, True), ([0, 1, 2, 3, 4, 5, 6, 6], [1.0] * 8, False)]
    g_a = all(routed(dict(end=dict(ch_map=cm, one_to_one=len(set(cm)) == len(cm), stream_acc=sa))) == want
              for cm, sa, want in synth)
    print(f"     (a) synthetic records (map; per-stream accuracy -> want): "
          + "; ".join(f"{cm} {sa} -> {w}" for cm, sa, w in synth) + f": {g_a}")
    g_b = True
    for k, want in (("ceiling4k16", True), ("uniform4k16", False)):
        r = f93[k].result()
        e = r["end"]
        g = r["ok"] and routed(r) == want
        g_b &= g
        print(f"     (b) {k:<11} seed 1, {EVAL_EVERY} steps (this test's run path): map {e['ch_map']} one-to-one "
              f"{e['one_to_one']}   per-stream acc {[round(v, 3) for v in e['stream_acc']]}   ROUTED {routed(r)} "
              f"(want {want})  -> {'OK' if g else 'WRONG'}")
    hand = [((9, 10), (0.5958, 0.9821)), ((10, 10), (0.7225, 1.0)), ((0, 10), (0.0, 0.2775)), ((5, 10), (0.2366, 0.7634))]
    g_c = all(abs(wilson(*kn)[0] - lo) < 5e-4 and abs(wilson(*kn)[1] - hi) < 5e-4 for kn, (lo, hi) in hand)
    print("     (c) Wilson 95%: " + "   ".join(f"{k_}/{n_} [{wilson(k_, n_)[0]:.4f}, {wilson(k_, n_)[1]:.4f}] (hand "
                                              f"[{lo:.4f}, {hi:.4f}])" for (k_, n_), (lo, hi) in hand) + f": {g_c}")
    good = g_a and g_b and g_c
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 94 a worker's run is bit-identical to the same run here:")
    rw = f94.result()
    rh = run_job(dict(spec("A4k16_R", 1, dict(REAL, total=6000), trial=True), force=[False, True]))
    t_ = rh.get("trial", {})
    g = (strip_all(rw) == strip_all(rh) and rh["ok"] and t_.get("used") == 2 and t_.get("continued") == 1001
         and rh["stopped_at"] == 6000)
    print(f"         A4k16_R seed 1, one restart (decisions forced: attempt 0 fails, attempt 1 passes), 6000 steps: equal "
          f"{strip_all(rw) == strip_all(rh)}   attempts {[(x['seed'], round(x['acc_check'], 4)) for x in t_.get('attempts', [])]}"
          f"   continued {t_.get('continued')} to step {rh.get('stopped_at')}   acc {rh.get('acc', float('nan')):.4f}   map "
          f"{rh['end']['ch_map'] if rh.get('ok') else '--'}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return fits


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds) if (r := get(store, key, s)) and r.get("ok")]


def routed(r):
    e = r["end"]
    return bool(e["one_to_one"] and min(e["stream_acc"]) >= ROUTED_ACC)


def outcome(r):
    if r["transition"] is not None:
        return "BOUND ROUTED" if routed(r) else "BOUND NOT routed"
    fc = fail_class_k(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if merged(r) else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def chk(x):
    return "--" if x is None else ("pass" if x else "fail")


def apath(r):
    """A trial's attempts: seed:accuracy at the check, + passed, - failed."""
    t_ = r.get("trial") or {}
    return " ".join(f"{a['seed']}:{fmt(a['acc_check'], 2)}{'+' if a['passed'] else '-'}" for a in t_.get("attempts", []))


def raw_rows(store, key, seeds, tc):
    trial = bool(ARM[key].get("restart"))
    print(f"  {'arm':<11} {'seed':>4} {'acc':>7} {'transition':>10} {'acc@' + str(tc):>8} {'chk':>4} {'m end':>6} {'eff_ch':>6} "
          f"{'map':>11} {'1:1':>3} {'sh':>2} {'used':>4}  per-stream acc                 outcome"
          + ("                 attempts (seed:acc@check)" if trial else ""))
    for s in seeds:
        r = get(store, key, s)
        if r is None:
            print(f"  {key:<11} {s:>4}  NOT RUN")
            continue
        if not r.get("ok"):
            print(f"  {key:<11} {s:>4}  FAILED — {r.get('error')}" + (f"   attempts {apath(r)}" if trial else ""))
            continue
        e = r["end"]
        extra = ""
        if trial:
            t_ = r["trial"]
            extra = f"  {t_['used']} used: {apath(r)}" + ("  (reused)" if t_["reused"] else "")
        print(f"  {key:<11} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {fmt(r.get('acc_check')):>8} "
              f"{chk(r.get('passed')):>4} {fmt(e.get('margin')):>6} {e['eff_ch']:>6.2f} {mstr(e['ch_map']):>11} "
              f"{'y' if e['one_to_one'] else 'n':>3} {shared_max(r):>2} {len(used(r)):>4}  "
              f"{'/'.join(f'{v:.2f}' for v in e['stream_acc']):<30} {outcome(r) + ('  collapsed' if r['collapsed'] else ''):<22}"
              + extra)
    print()


def passed_attempts(store):
    """R2's attempts: every distinct A4k16 / A4k16_R attempt that passed the check (and so was
    continued), by attempt seed -> bound. A reused attempt 0 is the plain run, counted once."""
    out = {}
    for s in SEEDS["A4k16"]:
        r = get(store, "A4k16", s)
        if r and r.get("ok") and r.get("passed"):
            out[s] = bound_r(r)
    for s in SEEDS["A4k16_R"]:
        r = get(store, "A4k16_R", s)
        if r and r.get("ok") and r.get("passed") and r["trial"]["continued"] not in out:
            out[r["trial"]["continued"]] = bound_r(r)
    return out


def check_perf(runs, step, thr):
    """Over plain runs: passes of (step, thr), passes that bound, fails that bound (missed)."""
    ps = [r for r in runs if (x := at(r["curve"], step)) is not None and x >= thr]
    fs = [r for r in runs if not ((x := at(r["curve"], step)) is not None and x >= thr)]
    return len(ps), sum(1 for r in ps if r["transition"] is not None), sum(1 for r in fs if r["transition"] is not None)


def report_stop(store, wall, path, tc):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — ceiling4k16 only (nothing else was run)")
    print("#" * 100)
    raw_rows(store, "ceiling4k16", SEEDS["ceiling4k16"], tc)
    c = sum(bound(store, "ceiling4k16", s) for s in SEEDS["ceiling4k16"])
    need = min(CEIL4_MIN, len(SEEDS["ceiling4k16"]))
    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    print(f"  ceiling4k16 binds {c}/{len(SEEDS['ceiling4k16'])} (needs >= {need}): the validity condition FAILED; nothing "
          f"else was run")
    print("     *** PART 1 UNTESTED ***")
    print()
    print("PRE-REGISTERED CLAIMS: R1, R2, R3 and R4 UNTESTED; no reading. Part 2 not run.")
    store["verdict"] = dict(valid=False, stopped="ceiling", ceiling=c, R1="UNTESTED", R2="UNTESTED", R3="UNTESTED",
                            R4="UNTESTED", readings=[])
    save_results(path, store)
    print(f"\n  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}\n")


def report(store, sched, fits, part2, wall, path, also):
    tc = sched["t_check"]
    print("#" * 100)
    print("PER-SEED RAW RESULTS (Part 1) — every value, before any aggregate")
    print("#" * 100)
    print(f"  (acc@{tc} and chk = the held-out accuracy at the check and the rule's decision; m end = routing margin; map = "
          f"the stream -> channel argmax at value positions, stream 0 first; 1:1 = one-to-one; sh = streams on the most "
          f"crowded channel; used = channels with >= 5% of any stream's mean gate; ROUTED = one-to-one and every stream "
          f">= {ROUTED_ACC}; for _R arms the row is the continued attempt)")
    for k in PART1:
        raw_rows(store, k, SEEDS[k], tc)

    print("=" * 100)
    print(f"COUNTS (BOUND = transition not None; ROUTED = one-to-one and every stream >= {ROUTED_ACC}; MERGED = shared >= 2 "
          f"at the end)")
    print("=" * 100)
    c = {}
    for k in PART1:
        rs = ok_runs(store, k)
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        extra = ""
        if ARM[k].get("restart"):
            extra = f"   attempts used {sum(r['trial']['used'] for _, r in rs)} (reused attempt 0: " \
                    f"{sum(1 for _, r in rs if r['trial']['reused'])})"
        print(f"  {ARM[k]['label']:<52} bound {c[k]:>2}/{len(SEEDS[k])}   bound and ROUTED "
              f"{sum(1 for _, r in rs if r['transition'] is not None and routed(r)):>2}   one-to-one at the end "
              f"{sum(1 for _, r in rs if r['end']['one_to_one']):>2}   MERGED {sum(1 for _, r in rs if merged(r)):>2}/{len(rs)}   "
              f"completed {len(rs)}/{len(SEEDS[k])}" + extra)
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["ceiling4k16"])
    need = min(CEIL4_MIN, nc)
    fit_ok = fits["A4k16"]["argmax"] >= FIT_MIN
    valid = c["ceiling4k16"] >= need and fit_ok
    print(f"  ceiling4k16 binds {c['ceiling4k16']}/{nc} (needs >= {need})   the k=16 gate fits the perfect 4-way routing: "
          f"argmax {fits['A4k16']['argmax']:.4f} (need >= {FIT_MIN}): {fit_ok}")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (Part 1; outcome: BOUND)")
    print("#" * 100)
    v = dict(bound=c, valid=valid, fits=fits)
    ver = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    b1 = band(c["A4k16_R"])
    v["R1"] = b1 if valid else "UNTESTED"
    print(f"  R1 THE RECIPE IS RELIABLE (band: RELIABLE >= 18, MAJORITY 10-17, MINORITY 1-9, NEVER 0): A4k16_R "
          f"{c['A4k16_R']}/{len(SEEDS['A4k16_R'])}")
    print(f"     *** R1: {v['R1']} ***")
    pa = passed_attempts(store)
    n_p, k_p = len(pa), sum(pa.values())
    lo, hi = wilson(k_p, n_p)
    frac = k_p / n_p if n_p else float("nan")
    r2 = ("UNTESTABLE" if n_p < PRECISE_N else "PRECISE" if frac >= PRECISE_FRAC else "NOT PRECISE") if valid else "UNTESTED"
    v.update(R2=r2, R2_passed=n_p, R2_bound=k_p, R2_wilson=[lo, hi])
    print(f"  R2 THE CHECK IS PRECISE: A4k16 / A4k16_R attempts that passed the check at {tc}: {n_p} (need >= {PRECISE_N}); "
          f"went on to bind {k_p}/{n_p} = {frac:.3f} (PRECISE if >= {PRECISE_FRAC}); Wilson 95% [{lo:.3f}, {hi:.3f}]")
    miss = sorted(s for s, b in pa.items() if not b)
    print(f"     passed and did not bind: {', '.join(f's{s}' for s in miss) if miss else 'none'}")
    print(f"     *** R2: {r2} ***")
    p3 = fisher_greater(c["A4k16_R"], len(SEEDS["A4k16_R"]), c["A4k4_R"], len(SEEDS["A4k4_R"]))
    b3 = sum(bound(store, "A4k16_R", s) and not bound(store, "A4k4_R", s) for s in SEEDS["A4k16_R"])
    c3 = sum(bound(store, "A4k4_R", s) and not bound(store, "A4k16_R", s) for s in SEEDS["A4k16_R"])
    m3 = mcnemar_greater(b3, c3)
    v.update(R3=ver(p3), R3_p=p3, R3_mcnemar=[b3, c3, m3])
    print(f"  R3 SPARE CHANNELS MAKE RESTARTS WORK (Fisher one-sided): A4k16_R {c['A4k16_R']}/{len(SEEDS['A4k16_R'])} vs A4k4_R "
          f"{c['A4k4_R']}/{len(SEEDS['A4k4_R'])}; p = {p3:.3g}   (exact McNemar one-sided alongside: A4k16_R only {b3}, "
          f"A4k4_R only {c3}, p = {m3:.3g})")
    print(f"     *** R3: {v['R3']} ***")
    b4 = sum(bound(store, "A4k16", s) and not bound(store, "A4k4", s) for s in SEEDS["A4k16"])
    c4 = sum(bound(store, "A4k4", s) and not bound(store, "A4k16", s) for s in SEEDS["A4k16"])
    p4 = mcnemar_greater(b4, c4)
    v.update(R4=ver(p4), R4_p=p4, R4_discordant=[b4, c4])
    print(f"  R4 REPLICATION OF K16 ON FRESH SEEDS (exact McNemar, one-sided): A4k16 {c['A4k16']}/{len(SEEDS['A4k16'])} vs "
          f"A4k4 {c['A4k4']}/{len(SEEDS['A4k4'])}; A4k16 only {b4}, A4k4 only {c4}; p = {p4:.3g}")
    print(f"     *** R4: {v['R4']} ***")
    print()
    print("  READINGS:")
    rd = []
    if valid:
        if v["R1"] == "RELIABLE" and v["R2"] == "PRECISE":
            rd.append(("R1 RELIABLE and R2 PRECISE", READINGS["R1+ R2+"]))
        if v["R3"] == "SHOWN":
            rd.append(("R3 SHOWN", READINGS["R3+"]))
        if v["R1"] in ("MINORITY", "NEVER"):
            rd.append((f"R1 {v['R1']}", READINGS["R1-"]))
        for cond, text in rd:
            print(f"     {cond}: \"{text}\"")
        if not rd:
            print("     (none of the pre-registered readings applies)")
    else:
        print("     (UNTESTED: no reading)")
    v["readings"] = [cond for cond, _ in rd]
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print(f"  per seed: plain outcomes, then the _R trials (attempts used; each attempt's acc@{tc}, + passed, - failed); "
          f"B = bound (transition), . = not bound:")
    for s in SEEDS["A4k16"]:
        cells = []
        for k in ("A4k4", "A4k16", "A4k4_R", "A4k16_R"):
            r = get(store, k, s)
            if not r or not r.get("ok"):
                cells.append(f"{k}: --")
                continue
            head = f"{k}: {'B' if r['transition'] is not None else '.'} {fmt_step(r['transition']):>5}"
            if ARM[k].get("restart"):
                head += f" [{r['trial']['used']}: " + " ".join(
                    f"{fmt(a['acc_check'], 2)}{'+' if a['passed'] else '-'}" for a in r["trial"]["attempts"]) + "]"
            cells.append(head)
        print(f"    s{s}  {cells[0]:<17} {cells[1]:<18} {cells[2]:<48} {cells[3]}")
    print()
    print(f"  check performance at ({tc}, {A_CHECK}) — passes, passes that bound, binders missed (failed the check but bound):")
    for k in ("A4k16", "A4k4"):
        rs = [r for _, r in ok_runs(store, k)]
        n_, b_, m_ = check_perf(rs, tc, A_CHECK)
        print(f"    {k:<8} (plain, {len(rs)} runs)   passed {n_:>2}   passed and bound {b_:>2}   missed {m_:>2}")
    for k in ("A4k16_R", "A4k4_R"):
        rs = [r for _, r in ok_runs(store, k)]
        att = [a for r in rs for a in r["trial"]["attempts"]]
        npass = sum(1 for a in att if a["passed"])
        cont_fail = [r for r in rs if r.get("passed") is False]
        a0_miss = sum(1 for s in SEEDS[k] if (rp := get(store, PLAIN[k], s)) and rp.get("ok") and rp.get("passed") is False
                      and rp["transition"] is not None)
        print(f"    {k:<8} ({len(att)} attempts in {len(rs)} trials)   passed {npass:>2}   passed and bound "
              f"{sum(1 for r in rs if r.get('passed') and r['transition'] is not None):>2}   abandoned "
              f"{sum(1 for a in att if a['passed'] is False) - len(cont_fail):>2} (attempt 0s among them that bound as the plain "
              f"run: {a0_miss})   last attempts continued after failing {len(cont_fail)}, of which bound "
              f"{sum(1 for r in cont_fail if r['transition'] is not None)}")
    print("  the alternatives, on the plain arms (for reference only):")
    for st, thr in ALT_CHECKS:
        for k in ("A4k16", "A4k4"):
            rs = [r for _, r in ok_runs(store, k)]
            n_, b_, m_ = check_perf(rs, st, thr)
            print(f"    ({st}, {thr})  {k:<6} passed {n_:>2}   passed and bound {b_:>2}   missed {m_:>2}")
    print()
    print("  failure classes and outcomes (MERGED = STREAM-PARTIAL with streams sharing a channel; non-stream = "
          "test_scale_axes's generalized classes):")
    for k in ("A4k16", "A4k16_R", "A4k4", "A4k4_R"):
        rs = ok_runs(store, k)
        tally = {}
        for s, r in rs:
            tally.setdefault(outcome(r), []).append(s)
        print(f"    {k:<8} " + "   ".join(f"{o} {len(x)}" for o, x in sorted(tally.items())))
        for o, xs in sorted(tally.items()):
            print(f"    {'':<8}   {o:<26} " + " ".join(f"s{s}" for s in xs))
        mgs = {}
        for _, r in rs:
            mgs[shared_max(r)] = mgs.get(shared_max(r), 0) + 1
        print(f"    {'':<8}   shared at the end, all runs: " + "   ".join(f"{n}: {mgs[n]}" for n in sorted(mgs))
              + f"   collapsed {sum(1 for _, r in rs if r['collapsed'])}")
    print()
    print("  continued attempts that passed the check and then did not bind (merges that passed the check and stalled):")
    for k in ("A4k4_R", "A4k16_R"):
        rs = ok_runs(store, k)
        pp = [(s, r) for s, r in rs if r.get("passed")]
        stall = [(s, r) for s, r in pp if r["transition"] is None]
        mg = [(s, r) for s, r in stall if merged(r)]
        print(f"    {k:<8} passed and continued {len(pp):>2}   of those not bound {len(stall):>2}   MERGED among them "
              f"{len(mg):>2}" + ("   " + "; ".join(f"s{s} (attempt {r['trial']['continued']}) {outcome(r)} acc {r['acc']:.3f} "
                                                 f"streams {[round(x, 2) for x in r['end']['stream_acc']]}" for s, r in stall)
                                 if stall else ""))
    print()
    print("  transitions, median [min, max]:")
    for k in PART1:
        trs = [r["transition"] for _, r in ok_runs(store, k) if r["transition"] is not None]
        print(f"    {k:<11} " + (f"{statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]" if trs else "none bound"))
    print()
    print("  channels used (>= 5% of any stream's mean gate at the end), median [min, max], and eff_ch, by outcome:")
    for k in ("A4k16", "A4k16_R", "A4k4", "A4k4_R"):
        rs = ok_runs(store, k)
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {k:<8} {lab:<9} ({len(sub):>2})  {med_int([len(used(r)) for r in sub]):<14}  eff_ch "
                      f"{med([r['end']['eff_ch'] for r in sub])}")
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git {other.get('meta', {}).get('git')}")
            for k in KEYS:
                rs2 = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                b2_ = sum(1 for r in rs2 if r.get("ok") and r.get("transition") is not None)
                nh = sum(1 for s in SEEDS[k] if get(store, k, s) is not None)
                bh = sum(bound(store, k, s) for s in SEEDS[k])
                print(f"    {k:<11} bound here {bh}/{nh}   other {b2_}/{len(rs2)}   pooled {bh + b2_}/{nh + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    report_part2(store, sched, fits, part2)

    print("=" * 100)
    print(f"CURVES — per evaluation (every {sched['eval_every']} steps): held-out accuracy x100; '^' marks the check at {tc}")
    print("=" * 100)
    for k in KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                continue
            curve_lines(k, s, r, tc)
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


def curve_str(curve, tc):
    return "".join(f"{'^' if st == tc else ' '}{round(acc * 100):>3}" for st, acc, _ in curve)


def curve_lines(k, s, r, tc):
    tag = f"{k:<11} s{s}"
    if ARM[k].get("restart"):
        for a in (r.get("trial") or {}).get("attempts", []):
            if a["passed"] is False and a["seed"] != (r.get("trial") or {}).get("continued"):
                print(f"  {tag} a{a['j']} {a['seed']:>4} acc {curve_str(a['early'] or [], tc)}  -> abandoned "
                      f"({fmt(a['acc_check'], 4)} < {A_CHECK})")
    if not r.get("ok"):
        print(f"  {tag} FAILED — {r.get('error')}")
        return
    if ARM[k].get("restart"):
        t_ = r["trial"]
        j = next(a["j"] for a in t_["attempts"] if a["seed"] == t_["continued"])
        tag += f" a{j} {t_['continued']:>4}"
    end = f"bound @{r['transition']}" if r["transition"] else "not bound"
    note = "  (attempt 0 = the plain run)" if (r.get("trial") or {}).get("reused") else ""
    print(f"  {tag} acc {curve_str(r['curve'], tc)}  -> {end}  [{mstr(r['end']['ch_map'])}]{note}")


def report_part2(store, sched, fits, part2):
    tc = sched["t_check"]
    print("=" * 100)
    print("PART 2 — S=8 (task_for(4, 8)), descriptive only")
    print("=" * 100)
    if part2.get("dropped"):
        print(f"  PART 2 DROPPED: the projection was {part2['projected_h']:.2f} h > {DROP_H:g} h on the workers; nothing "
              f"was run (drop rule).")
        print()
        return
    raw_rows(store, "ceiling8k16", SEEDS["ceiling8k16"], tc)
    n8 = sum(bound(store, "ceiling8k16", s) for s in SEEDS["ceiling8k16"])
    print(f"  ceiling8k16 bound {n8}/{len(SEEDS['ceiling8k16'])}   the recurrent k=16 gate's fit to the perfect 8-way routing "
          f"(CHECK 92): argmax {fits['A8k16_R']['argmax']:.4f}")
    if part2.get("skipped"):
        print(f"  A8k16_R SKIPPED: ceiling8k16 bound {n8}/{len(SEEDS['ceiling8k16'])} (as pre-registered: skipped at 0/2).")
        print()
        return
    raw_rows(store, "A8k16_R", SEEDS["A8k16_R"], tc)
    print(f"  A8k16_R per seed (attempts used; each attempt's acc@{tc}; bound; transition; map; one-to-one; per-stream "
          f"accuracy; final accuracy):")
    for s in SEEDS["A8k16_R"]:
        r = get(store, "A8k16_R", s)
        if not r or not r.get("ok"):
            print(f"    s{s}  " + ("NOT RUN" if r is None else f"FAILED — {r.get('error')}"))
            continue
        e = r["end"]
        print(f"    s{s}  attempts {r['trial']['used']} [{apath(r)}]   {'BOUND' if r['transition'] is not None else 'not bound'}"
              f" {fmt_step(r['transition']):>5}   map [{mstr(e['ch_map'])}] one-to-one {e['one_to_one']}   per-stream "
              f"{[round(x, 2) for x in e['stream_acc']]}   acc {r['acc']:.4f}   {outcome(r)}")
    b8 = sum(bound(store, "A8k16_R", s) for s in SEEDS["A8k16_R"])
    print(f"  A8k16_R bound {b8}/{len(SEEDS['A8k16_R'])} (descriptive; no claim)")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def describe(rec):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr, e = rec["transition"], rec["end"]
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} chk "
            f"{fmt(rec.get('acc_check'))} {chk(rec.get('passed')):<4} map [{mstr(e['ch_map'])}] shared {shared_max(rec)} "
            f"eff_ch {e['eff_ch']:.2f} streams {[round(v, 2) for v in e['stream_acc']]}")


def att0(rp, s, tc):
    return dict(j=0, seed=s, acc_check=rp.get("acc_check"), passed=rp.get("passed"), early=rp.get("early"),
                steps=rp.get("stopped_at") if rp.get("passed") else tc)


def r_step(store, key, s, sched, force, record):
    """key's trial for seed s once its plain run is known: reuse the plain record if attempt 0
    passed, else a trial from attempt 1 (from attempt 0 if the plain run failed)."""
    if get(store, key, s) is not None and not force:
        print(f"   {key:<11} seed {s}  cached")
        return None
    rp = get(store, PLAIN[key], s)
    if rp and rp.get("ok") and rp.get("passed"):
        rec = dict(rp, trial=dict(start=0, attempts=[att0(rp, s, sched["t_check"])], used=1, continued=s, reused=True,
                                  secs=0.0))
        record(dict(arm=key, seed=s), rec)
        return None
    return spec(key, s, sched, trial=True, start=1 if (rp and rp.get("ok")) else 0)


def finish_trial(store, sp, rec):
    """Prepend attempt 0 (the plain run) to a trial that started at attempt 1."""
    if "trial" in rec and sp["start"] == 1:
        rp = get(store, PLAIN[sp["arm"]], sp["seed"])
        rec["trial"]["attempts"] = [att0(rp, sp["seed"], sp["sched"]["t_check"])] + rec["trial"]["attempts"]
    elif "trial" not in rec:
        rec["trial"] = dict(start=sp["start"], attempts=[], used=None, continued=None, reused=False, secs=0.0)
    return rec


def main():
    ap = argparse.ArgumentParser(description="The four-stream recipe (k=16 + an early check with restarts), and S=8.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None, help="a second machine's stream_recipe_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    sched = dict(SCHED)
    tc, total = sched["t_check"], sched["total"]

    print("=" * 100)
    print("Stream recipe: spare channels (k=16) plus an early check with restarts, for four streams; and a first look at S=8")
    print(f"  Part 1: task_for(4, 4) (S=4, length {TASK44.L}); Part 2: task_for(4, 8) (S=8, length {TASK48.L}); lr {LR} (SUB_LR, "
          f"explicit); conv 'layer' width {CONV_K}; up to {total} steps per continued run, evaluation every {sched['eval_every']}; "
          f"early stop counted after {tc}")
    print(f"  restart rule (_R arms): held-out accuracy >= {sched['a_check']} at step {tc}, else restart with seed s + "
          f"{STRIDE}*j, at most {R_MAX} attempts, the last continues; attempt 0 of A4k16_R / A4k4_R = the plain run")
    for a in ARMS1 + ARMS2:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<52} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        fits = verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR, sched=sched,
                             r_max=R_MAX, stride=STRIDE, fits=fits, seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool, one evaluation charged per "
              f"{sched['eval_every']}; worst case = every continued run to {total} steps, every trial needing all its attempts "
              f"(A4k16_R / A4k4_R: attempts 1-3 to {tc} + attempt 4 in full; A8k16_R: attempts 0-3 to {tc} + attempt 4)")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        full = {k: cost[k] * total for k in KEYS}
        for k in KEYS:
            print(f"  {k:<11} {cost[k] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        d_ceil = [full["ceiling4k16"]] * len(SEEDS["ceiling4k16"])
        d1 = ([full[k] for k in ("A4k16", "A4k4") for _ in SEEDS[k]]
              + [full[k] + (R_MAX - 2) * tc * cost[k] for k in ("A4k16_R", "A4k4_R") for _ in SEEDS[k]])
        d2 = ([full["ceiling8k16"]] * len(SEEDS["ceiling8k16"])
              + [full["A8k16_R"] + (R_MAX - 1) * tc * cost["A8k16_R"]] * len(SEEDS["A8k16_R"]))
        mk1, mk2 = makespan(d_ceil, args.workers), makespan(d1 + d2, args.workers)
        total_h = (mk1 + mk2) / 3600
        print(f"  ceiling4k16 first: {len(d_ceil)} runs, {mk1 / 3600:.2f} h;  then Part 1's {len(d1)} runs/trials and Part 2's "
              f"{len(d2)} runs/trials: serial {(sum(d1) + sum(d2)) / 3600:.2f} h, {mk2 / 3600:.2f} h on {args.workers} "
              f"workers;  total {total_h:.2f} h")
        part2 = dict(projected_h=total_h, dropped=total_h > DROP_H, ran=False, skipped=False)
        if part2["dropped"]:
            mk2b = makespan(d1, args.workers)
            print(f"  *** the projection exceeds {DROP_H:g} h: PART 2 IS DROPPED (drop rule). Part 1 alone: "
                  f"{(mk1 + mk2b) / 3600:.2f} h ***")
            store["meta"]["projected_wall_h"] = (mk1 + mk2b) / 3600
        else:
            print(f"  within {DROP_H:g} h: Part 2 runs")
            store["meta"]["projected_wall_h"] = total_h
        store["meta"].update(projected_with_part2_h=total_h, part2_dropped=part2["dropped"])
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            extra, secs = "", rec.get("secs", 0.0)
            if ARM[sp["arm"]].get("restart"):
                t_ = rec.get("trial") or {}
                secs = t_.get("secs", secs)
                extra = f"  attempts {t_.get('used')} [{apath(rec)}]" + ("  (plain run reused)" if t_.get("reused") else "")
            print(f"   {sp['arm']:<11} seed {sp['seed']}  {describe(rec)}{extra}  {secs:.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)")

        def run_specs(specs, on_done=None):
            pending, inflight = list(specs), {}
            while pending or inflight:
                while pending and len(inflight) < args.workers:
                    sp = pending.pop(0)
                    inflight[pool.submit(run_job, sp)] = sp
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done:
                    sp = inflight.pop(f)
                    try:
                        rec = f.result()
                    except Exception as e:
                        rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
                    if sp.get("trial"):
                        rec = finish_trial(store, sp, rec)
                    record(sp, rec)
                    if on_done is not None:
                        pending[:0] = on_done(sp)

        def todo(k):
            out = []
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<11} seed {s}  cached")
                else:
                    out.append(spec(k, s, sched, trial=bool(ARM[k].get("restart"))))
            return out

        print("=" * 100)
        print(f"RUNS — ceiling4k16 first ({len(SEEDS['ceiling4k16'])} runs)")
        print("=" * 100)
        run_specs(todo("ceiling4k16"))
        n_ceil = sum(bound(store, "ceiling4k16", s) for s in SEEDS["ceiling4k16"])
        need = min(CEIL4_MIN, len(SEEDS["ceiling4k16"]))
        print(f"   ceiling4k16 bound {n_ceil}/{len(SEEDS['ceiling4k16'])} (needs >= {need})")
        print()
        if n_ceil < need:
            print("   *** the ceiling failed: nothing else is run ***\n")
            report_stop(store, time.time() - t0, args.results, tc)
            return

        def ceil8_known():
            return all(get(store, "ceiling8k16", s) is not None for s in SEEDS["ceiling8k16"])

        def a8_step():
            n8 = sum(bound(store, "ceiling8k16", s) for s in SEEDS["ceiling8k16"])
            need8 = min(CEIL8_MIN, len(SEEDS["ceiling8k16"]))
            print(f"   ceiling8k16 bound {n8}/{len(SEEDS['ceiling8k16'])}"
                  + ("" if n8 >= need8 else ": A8k16_R is skipped, as pre-registered"))
            if n8 < need8:
                part2["skipped"] = True
                return []
            return todo("A8k16_R")

        def after(sp):
            if sp["arm"] in ("A4k16", "A4k4"):
                nxt = r_step(store, sp["arm"] + "_R", sp["seed"], sched, args.force, record)
                return [] if nxt is None else [nxt]
            if sp["arm"] == "ceiling8k16" and ceil8_known():
                return a8_step()
            return []

        part2["ran"] = not part2["dropped"]
        n_rest = sum(len(SEEDS[k]) for k in PART1 if k != "ceiling4k16")
        print("=" * 100)
        print(f"RUNS — the rest ({n_rest} Part 1 runs/trials" + ("" if part2["dropped"] else
              f", and Part 2's {len(SEEDS['ceiling8k16'])} + {len(SEEDS['A8k16_R'])} runs/trials, ceiling8k16 first") +
              "; each _R trial is queued, first, when its plain run is known)")
        print("=" * 100)
        first = []
        if part2["ran"]:
            first = todo("ceiling8k16")
            if ceil8_known():
                first = a8_step() + first
        pre = []
        for key in ("A4k16_R", "A4k4_R"):
            for s in SEEDS[key]:
                if get(store, PLAIN[key], s) is not None and not args.force:
                    nxt = r_step(store, key, s, sched, args.force, record)
                    if nxt is not None:
                        pre.append(nxt)
        plain = [sp for pr in itertools.zip_longest(todo("A4k16"), todo("A4k4")) for sp in pr if sp is not None]
        run_specs(pre + first + plain, on_done=after)
        print()

    report(store, sched, fits, part2, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
