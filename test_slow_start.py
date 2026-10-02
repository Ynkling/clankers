#!/usr/bin/env python
"""
test_slow_start.py — does slow memory plus a hinge on the gate's position information, with no
restarts, hold on fresh seeds and carry to the working configuration (convolution, eight keys,
four streams)? lr 1e-3 (SUB_LR, passed explicitly). Everything below is fixed before any run.

Run it directly (on X the paired files are results/X/...):

    python test_slow_start.py --workers 4 --scale results/X/scale_axes_results.json \\
        --recipe results/X/stream_recipe_results.json --curriculum results/X/stream_curriculum_results.json \\
        --short results/X/short_conv_results.json
    python test_slow_start.py ... --also results/L/slow_start_results.json      (pooled counts, descriptive)

BACKGROUND
- Exploratory screens on branch claude/outside-ideas (container E, which reproduces X bit for bit
  at 1 thread; S=2, P=4, k=2, grouped layout, no convolution, lr 1e-3, seeds 160-199; screens,
  not results):
  - SLOW (batches 2-3): DISCOVERED 24/40 vs X's arm A 12/40 (15 vs 3, p = 0.0075); failures:
    position splits 18 -> 11, key splits 9 -> 3.
  - A full eta^2 penalty on the read gate's position information (batch 4) removed position
    splits but sat at its sampling floor (median 0.025 at step 1200) and delayed commitment
    (routed at 1200: 0/20).
  - HINGE (batch 5, bd9adb6): DISCOVERED 34/40 vs SLOW 24/40 (10 vs 0, p = 0.002) and vs arm A
    12/40 (23 vs 1, p = 3e-6); failures 2 KEY, 4 OTHER, no POSITION; routed at 1200 28/40. The
    hinge never fired on 20 of the 40 seeds and fired a few times, early, on most others. A
    random-direction control runs on the side branch; this test does not need it.
- Main line:
  - test_scale_axes' DIRECT8 (S=2, P=8, conv, from scratch): bound X 13/20, L 12/20.
  - test_stream_recipe's A4k16 (S=4, P=4, k=16, conv): bound X 12/20, L 9/20.
  - test_stream_curriculum's D8 (S=8, P=4, k=16, conv, from scratch): 0/16 over both machines,
    every run at chance. On L, 5 of 10 D8 gates were position splits at 4800 (eta2 by index
    0.70-0.96); SC8's collapses were position splits too (0.88-0.99).
- Question: does slow memory plus the hinge, with no restarts, hold on fresh seeds and carry to
  the working configuration (convolution, eight keys, four streams)?

THE RECIPE
- SLOW: one Adam optimizer, two parameter groups, moments carried over. Gate group W_in, W_h, W_g
  at 1e-3 throughout; every other parameter (embedding and convolution included) at 1e-4
  (SUB_LR / 10, equal to 1e-4 exactly) for updates 1-2400, 1e-3 after.
- HINGE = SLOW + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] on every training batch:
  - p = the read gate at key positions 3j+1, j = 0..S*P-1 (the query's key excluded), shape
    (B, S*P, k);
  - eta2 pooled over channels: the between-group sum of squares summed over channels, divided by
    the total sum of squares summed over channels (+ EPS, 1e-12 as the screens); groups: triple
    index j (eta2_index), and j < S*P/2 vs j >= S*P/2 (eta2_half);
  - its gradient reaches the gate parameters and the embedding only.
- Each new arm uses the run path of the recorded arm it pairs with, changing only the optimizer
  groups and the loss term, so a seed fixes the same initial parameters and batches.
- Implementation (the screens' mechanism, explore_slow_mem / explore_slow_hinge at bd9adb6):
  - the four run paths gained one knob, recipe(kw) -> kw, which rewrites the keyword arguments
    they pass to test_router_reliability.run_one (None = unchanged; CHECK 101): test_short_conv.
    run_job (Part 0), test_scale_axes.run_job (Part A), test_stream_recipe.run_attempt (Part B)
    and test_stream_curriculum.run_sc (Part C);
  - SLOW passes onset_run's param_groups, and sets the second group's lr to 1e-3 in run_one's
    per-evaluation check right after the evaluation at step 2400 (after the run path's own check);
  - HINGE adds a forward hook to the built model: on every training forward (training mode,
    gradients on) it computes the two terms from the read gate the forward returned, and passes
    the logits through the screens' gradient-injection node (identity on the logits; in backward
    it also sends gradient 1 into the penalty), so onset_run's loss and loop are untouched. The
    hook counts the training batches with either term > 0, per 1200 updates.
  - Every arm here also records, at every evaluation, the pooled eta2 of the read gate at key
    positions on the run's probe batch by index, half, key and stream (test_scale_axes.eta2_multi,
    float64), and at k=2 the routing statistics (test_router_layout.routing_stats) where the run
    path does not already record them. Statistics run under eval() and no_grad and do not touch
    training (CHECKs 103 and 104).

ARMS (lr 1e-3; no restarts anywhere)
- Part 0, the screens' configuration on fresh seeds 280-299 (20): S=2, P=4, k=2, grouped, no
  convolution, MAX_ITERS 24000, test_short_conv's arm A run path. Outcome DISCOVERED (bound and
  final VAL cos < 0.5).
    A0      arm A
    SLOW0   arm A + SLOW
    HINGE0  arm A + HINGE
- Part A, eight keys: HINGE8 = DIRECT8 + HINGE (S=2, P=8, conv, MAX_ITERS 28800), seeds 220-239,
  paired with this machine's recorded DIRECT8. Outcome BOUND.
- Part B, four streams: HINGE4k16 = A4k16 + HINGE (S=4, P=4, k=16, conv, MAX_ITERS 28800), seeds
  240-259, paired with this machine's recorded A4k16. Outcome BOUND.
- Part C, eight streams (descriptive): HINGE_D8 = D8 + HINGE (S=8, P=4, k=16, conv, all 8 streams
  from step 1, MAX_ITERS 28800), seeds 260-269, paired with this machine's recorded D8 (X has
  260-265, L 260-269). Outcome BOUND.
- Flags name this machine's recorded files: --scale (test_scale_axes), --recipe
  (test_stream_recipe), --curriculum (test_stream_curriculum), and --short (test_short_conv, for
  CHECK 104). Defaults are the files a run writes in this directory; on X, results/X/....

PAIRING
- A recorded file pairs if one of its recorded runs reproduces bit for bit on this machine,
  whatever the CPU label. X moved from a 2.10GHz to a 2.80GHz host; its recorded runs still
  reproduce (test_stream_curriculum's inherited CHECKs 59, 64, 72, 85). Both labels and the
  reproduction are printed (CHECK 110: the recorded run of the part's first seed, re-run on its
  plain run path through step 2400; curve, statistics and gradient norms equal). A failed
  reproduction makes that part's claims UNTESTED.

CLAIMS (per machine; exact McNemar, one-sided, new arm higher; SHOWN if p < 0.05)
- H0   HINGE0 beats A0 (DISCOVERED).
- HA   HINGE8 beats DIRECT8 (BOUND).
- HB   HINGE4k16 beats A4k16 (BOUND).
- S0   SLOW0 beats A0 (DISCOVERED).
- H0S  HINGE0 beats SLOW0 (DISCOVERED).
- Bands without restarts for HINGE0, HINGE8 and HINGE4k16: RELIABLE >= 18/20, MAJORITY 10-17,
  MINORITY 1-9, NEVER 0.
- Part C reading: "the hinge prevents the collapse" if at most 2 of the 10 HINGE_D8 runs end
  collapsed (as test_stream_curriculum flags it: final accuracy below COLLAPSE = 0.15);
  otherwise "it does not".
- Readings, printed: "the screen replicates on fresh seeds" if H0 SHOWN; "it carries to the
  working configuration" if HA and HB SHOWN; "it carries to eight keys only" or "it carries to
  four streams only" if just one of them is.

DIAGNOSTICS (labelled, not part of the verdict)
- Per seed: outcome, transition, final accuracy, failure class; at k=2, routed at 1200, 2400 and
  the end (ROUTED*, the screens' criterion: margin >= 0.9 and eta_key_by_stream > 0.9); at k=16,
  the stream -> channel map, one-to-one, streams sharing a channel, per-stream accuracy, ROUTED
  (one-to-one and every stream >= 0.9); eta2 by index, half, key and stream at 1200, 4800 and
  the end.
- Hinge firing counts per run: updates 1-1200, 1-2400, whole run.
- Part 0: the three arms' per-seed outcomes side by side; failure classes per arm.
- The label-free check (any eta2 >= 0.5 at 1200; the groups that need no stream label: index,
  half and key, as explore_gate_reset's detector) against the outcome, per arm.
- Failure classes: k=2, test_router_layout.fail_class; k=16, test_stream_recipe.outcome.

CHECKS (after the inherited chain, which ends with test_stream_curriculum's CHECK 100)
- 101 The recipe knob is inert: each of the four run paths with recipe=None and with the identity
      recipe gives the same record as 520fe51's module (legacy modules loaded from that commit).
- 102 With TAU = 1.0, HINGE equals SLOW bit for bit (curves and weights, 3600 steps) in each
      part's configuration; the hinge counted 0 firings.
- 103 With both groups at 1e-3 from step 1 and TAU = 1.0, each new arm equals its plain
      counterpart bit for bit through 2400 steps: Part 0 against A0 on the same seed; Parts A-C
      against the recorded paired run (and, weights included, against a fresh plain run).
- 104 A0's run path reproduces the recorded test_short_conv arm A run (seed 160) bit for bit.
- 105 At k = 2 the pooled eta2 equals routing_stats' eta_key_by_index and eta_key_by_half on a
      probe batch (to 1e-6); at k = 16 it equals a NumPy recomputation (to 1e-6). Two gates per
      part: the learned gate redrawn far from uniform, and a synthetic gate with strong index and
      half effects. (NumPy is in requirements.txt; it was installed into this container for this.)
- 106 The perfect gate scores below 0.1 on both terms on 20 probe batches in each part's
      configuration; a gate set by sequence half scores above 0.5 on eta2_half.
- 107 The hinge's gradient reaches only W_in, W_h, W_g and the embedding (directly, and through
      the hook with the task loss zeroed), and is exactly zero on a batch with both terms below
      0.2.
- 108 An optimizer pre-hook confirms the lrs (gate 1e-3 at every update; the rest 1e-4 through
      update 2400, 1e-3 from 2401) and that the gate group holds exactly W_in, W_h, W_g, through
      every arm's real run path and schedule (the forward stubbed, held-out accuracy stubbed to
      0.5, so no early stop).
- 109 No earlier test used seeds 280-299, restart reseeds included.
- 110 The pairing (above). Not asserted: a file that does not pair makes its part's claims
      UNTESTED.
- 111 A worker's run is bit-identical to the same run here (HINGE0 with TAU = 0, so the hinge
      fires on every batch; and HINGE4k16).
- 112 (informative, not asserted) SLOW0 and HINGE0 against the screens' recorded runs
      (explore_out at claude/outside-ideas 0c5fa01), seeds 160 and 162, through 3600 steps.

RUNTIME
- The worst-case projection (every run to its MAX_ITERS) is printed before training.
- If it exceeds 12 h: drop Part C first, then SLOW0 (S0 and H0S become UNTESTED). Never cut A0,
  HINGE0, HINGE8 or HINGE4k16. If it is still over 12 h, stop before training and report. The
  decision is stored in the results file at its first start and kept on a resume.
- Runs are scheduled longest first on 4 workers x 1 thread; finished runs are cached in the
  results file, so an interrupted run resumes.
- Both machines run it, as usual.

OUTPUT
Per-seed raw values first, then counts, pairing, claims, readings, diagnostics and curves.
Results: slow_start_results.json (gitignored; X's copy in results/X/).

RESULT (X, test commit 9c5939e; results/X/slow_start_results.json)
Run conditions:
- 110/110 records, none failed; 383.4 min of training after verification, 4 workers x 1 thread.
- The first start (2.10GHz host, 2026-10-02 01:08) recorded its drop decision (projection 6.69 h:
  nothing dropped) and was then killed by a container restart before any run finished. The
  resumed start (2.80GHz host, 05:21) re-ran the whole verification, kept the recorded decision
  (its own projection was 10.54 h) and ran every run. Every recorded file paired by reproduction
  on both hosts (CHECK 110); all CHECKs passed on both.
- CHECK 112 (informative): SLOW0 reproduces the screens' SLOW_MEM (seed 160) and HINGE0 their
  SLOW_HINGE where the hinge never fired (seed 160); on seed 162 (3 firings in both) the curves
  differ slightly (0.2153 vs 0.2158 at 1200), as expected: the pooled statistic equals the
  screens' channel-0 one only up to float rounding.
Claims:
- H0   HINGE0 19/20 vs A0 8/20, 11 vs 0, p = 0.00049: SHOWN.
- HA   HINGE8 20/20 vs DIRECT8 13/20, 7 vs 0, p = 0.0078: SHOWN.
- HB   HINGE4k16 17/20 vs A4k16 12/20, 7 vs 2, p = 0.090: NOT SHOWN.
- S0   SLOW0 15/20 vs A0 8/20, 9 vs 2, p = 0.033: SHOWN.
- H0S  HINGE0 19/20 vs SLOW0 15/20, 4 vs 0, p = 0.0625: NOT SHOWN.
- Bands: HINGE0 RELIABLE (19/20), HINGE8 RELIABLE (20/20), HINGE4k16 MAJORITY (17/20).
- Readings: "the screen replicates on fresh seeds"; "it carries to eight keys only".
- Part C: 4 of 10 HINGE_D8 runs end collapsed: "it does not". None bound (D8 0/6 recorded here).
Diagnostics (not part of the verdict):
- Part 0 failures: A0 9 POSITION, 2 KEY, 1 STREAM-PARTIAL; SLOW0 4 POSITION, 1 OTHER; HINGE0 one
  OTHER (s287, the only run where the hinge kept firing: 950 batches). HINGE0's hinge never fired
  on 11/20 seeds and at most twice on the others; ROUTED* at 1200: A0 8, SLOW0 15, HINGE0 14.
- HINGE8 bound all 20, including the 5 POSITION and 2 STREAM-PARTIAL failures of the recorded
  DIRECT8 (17 by step 4800; s222, s223, s237 later). It fired on 12/20 runs, median once.
- HINGE4k16: 15 BOUND ROUTED, 2 bound without a one-to-one map, 2 merges, 1 one-to-one partial;
  no non-stream failure (A4k16 had 4: 2 POSITION, 1 KEY, 1 OTHER). Discordant pairs 7 vs 2: it
  lost s242 (partial) and s243 (merge), which A4k16 bound late (18000, 16800).
- HINGE_D8: no binder; 6 of 10 ended above chance as merges or partial routes (0.16-0.47, best
  s265 at 0.47 with two streams near 0.9), against D8's flat 0.08-0.10; the hinge fired 3-83
  times per run, mostly after step 2400.
- The label-free check at 1200 (any eta2 by index, half or key >= 0.5) flagged 8 A0 and 5 SLOW0
  runs, none of which discovered; under HINGE it flagged 1/20 (Part 0) and 1/20 (Part A), and
  those runs bound.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import time_per_step, fmt_step, EVAL_EVERY, BATCH
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_router_curriculum import (
    SUB_LR, get, load_store, init_worker, makespan, cpu_model, save_results, TAU_END, TIME_STEPS,
)
from test_router_discovery import load_at, GATE_PARAMS, COLLAPSE
import test_router_reliability as trr
from test_router_reliability import strip_all
from test_router_layout import routing_stats, fail_class
from test_conv_lr import bound_r, mcnemar_greater
import test_short_conv as tsc
import test_scale_axes as tsa
from test_scale_axes import eta2_multi, band
import test_stream_recipe as tsr
from test_stream_recipe import outcome, routed
from test_stream_channels import shared_max, mstr, fmt, med_int
import test_stream_curriculum as tscur

# ── Settings (fixed before any run) ──────────────────────────────────────────
LR = SUB_LR                                # 1e-3: the gate group throughout, every group after the warm-up
LR_WARM = SUB_LR / 10                      # 1e-4 (== 1e-4 exactly): the non-gate group, updates 1-WARM
WARM = 2400
LAMBDA, TAU = 1.0, 0.2
EPS = 1e-12                                # the screens' (explore_slow_pos.EPS)
GATE = tuple(GATE_PARAMS)                  # ("W_in", "W_h", "W_g")
MAX0, MAXK = tsc.MAX_ITERS, 28800          # Part 0 (24000); Parts A-C
REAL = dict(iters0=MAX0, iters=MAXK, warm=WARM)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9         # ROUTED* at k=2 (the screens')
FLAG_ETA, FLAG_GROUPS = 0.5, ("index", "half", "key")
ETA_GROUPS = ("index", "half", "key", "stream")
ETA_STEPS = (EVAL_EVERY, 4800, "end")
COLLAPSE_MAX = 2                           # Part C: "prevents the collapse" if at most 2 of 10 end collapsed
ALPHA = 0.05
PERFECT_MAX, HALF_MIN = 0.1, 0.5           # CHECK 106
DROP_H = 12.0
LEGACY_SHA = "520fe51"                     # head before this test's changes
SCREEN_SHA = "0c5fa01"                     # claude/outside-ideas, the screens' recorded runs (CHECK 112)
CHECK_SEED = 1
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "slow_start_results.json"

TASK0, TASK8, TASK44, TASK48 = tsc.TASK, tsa.TASKS["P8S2"], tsa.TASK44, tsr.TASK48
PART = {
    "0": dict(task=TASK0, k=2, src="test_short_conv's arm A", file=None, rec_arm="A", seed0=None,
              ceil=("tsc", "ceiling")),
    "A": dict(task=TASK8, k=2, src="test_scale_axes' DIRECT8", file="scale", rec_arm="DIRECT8", seed0=220,
              ceil=("tsa", "ceiling8")),
    "B": dict(task=TASK44, k=16, src="test_stream_recipe's A4k16", file="recipe", rec_arm="A4k16", seed0=240,
              ceil=("tsr", "ceiling4k16")),
    "C": dict(task=TASK48, k=16, src="test_stream_curriculum's D8", file="curriculum", rec_arm="D8", seed0=260,
              ceil=("tsr", "ceiling8k16")),
}
ARMS = [
    dict(key="A0", part="0", slow=False, tau=None, label="A0         arm A (S=2, P=4, k=2, no conv)"),
    dict(key="SLOW0", part="0", slow=True, tau=None, label="SLOW0      arm A + SLOW"),
    dict(key="HINGE0", part="0", slow=True, tau=TAU, label="HINGE0     arm A + HINGE"),
    dict(key="HINGE8", part="A", slow=True, tau=TAU, label="HINGE8     DIRECT8 + HINGE (S=2, P=8, conv)"),
    dict(key="HINGE4k16", part="B", slow=True, tau=TAU, label="HINGE4k16  A4k16 + HINGE (S=4, P=4, k=16, conv)"),
    dict(key="HINGE_D8", part="C", slow=True, tau=TAU, label="HINGE_D8   D8 + HINGE (S=8, P=4, k=16, conv, 8 streams)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
S20 = tuple(range(280, 300))
SEEDS = dict(A0=S20, SLOW0=S20, HINGE0=S20, HINGE8=tuple(range(220, 240)), HINGE4k16=tuple(range(240, 260)),
             HINGE_D8=tuple(range(260, 270)))
PAIRED = {"HINGE8": "A", "HINGE4k16": "B", "HINGE_D8": "C"}
CUTS = ("HINGE_D8", "SLOW0")               # the drop order; A0, HINGE0, HINGE8 and HINGE4k16 are never cut
CLAIMS = (("H0", "HINGE0", "A0"), ("HA", "HINGE8", "A"), ("HB", "HINGE4k16", "B"), ("S0", "SLOW0", "A0"),
          ("H0S", "HINGE0", "SLOW0"))
READINGS = {
    "H0": "the screen replicates on fresh seeds",
    "HA HB": "it carries to the working configuration",
    "HA": "it carries to eight keys only",
    "HB": "it carries to four streams only",
    "C+": "the hinge prevents the collapse",
    "C-": "it does not",
}


def iters_of(part, sched):
    return sched["iters0"] if part == "0" else sched["iters"]


def outcome_key(part):
    return "DISCOVERED" if part == "0" else "BOUND"


def success(part, r):
    """The part's outcome: DISCOVERED (bound and final VAL cos < 0.5) in Part 0, BOUND elsewhere."""
    if not (r and r.get("ok")):
        return False
    return bool(r["discovered"]) if part == "0" else bound_r(r)


# ── The recipe ───────────────────────────────────────────────────────────────
class Inject(torch.autograd.Function):
    """The screens' gradient-injection node (explore_aux_gate._Inject): the identity on the
    logits; in backward it also sends gradient 1 into the penalty."""

    @staticmethod
    def forward(ctx, logits, aux):
        return logits.clone()

    @staticmethod
    def backward(ctx, g):
        return g, torch.ones((), dtype=g.dtype)


def eta2_hinge(gr, S, P, eps=EPS):
    """The hinge's two statistics, differentiable: eta^2 of the read gate at the body's key
    positions 3j+1 (j = 0..S*P-1, the query's key excluded), pooled over the k channels (between-
    group over total sum of squares, both summed over channels, + eps), by triple index j and by
    sequence half (j < S*P/2 vs j >= S*P/2). At k = 2 this is explore_slow_pos.pos_penalty's
    channel-0 statistic up to float rounding (CHECK 105)."""
    n = S * P
    j = torch.arange(n)
    p = gr[:, 3 * j + 1, :]                                          # (B, n, k)
    B = p.shape[0]
    mu = p.mean((0, 1))
    tot = ((p - mu) ** 2).sum() + eps
    between_idx = B * ((p.mean(0) - mu) ** 2).sum()
    h = j >= n // 2
    between_half = B * (int((~h).sum()) * ((p[:, ~h].mean((0, 1)) - mu) ** 2).sum()
                        + int(h.sum()) * ((p[:, h].mean((0, 1)) - mu) ** 2).sum())
    return between_idx / tot, between_half / tot


def attach_hinge(m, S, P, tau, lam, state):
    """The hinge on model m (a forward hook): on every training forward (training mode, gradients
    on), lam * [relu(eta2_index - tau) + relu(eta2_half - tau)] enters the loss through Inject;
    state counts the training batches and those with either term > 0, per EVAL_EVERY updates."""
    state.update(n=0, fired=0, windows=[], last=None)

    def hook(mod, args, out):
        if not (mod.training and torch.is_grad_enabled()):
            return None
        logits, sat, gr, gw = out
        e_i, e_h = eta2_hinge(gr, S, P)
        h_i, h_h = F.relu(e_i - tau), F.relu(e_h - tau)
        w = state["n"] // EVAL_EVERY
        state["n"] += 1
        while len(state["windows"]) <= w:
            state["windows"].append(0)
        if bool(h_i > 0) or bool(h_h > 0):
            state["fired"] += 1
            state["windows"][w] += 1
        state["last"] = [float(e_i.detach()), float(e_h.detach())]
        return Inject.apply(logits, lam * (h_i + h_h)), sat, gr, gw

    return m.register_forward_hook(hook)


def slow_groups(state, warm_lr):
    """onset_run's param_groups: the gate group (W_in, W_h, W_g) at LR, every other parameter at
    warm_lr; the group dicts are kept so the lr switch can set the second one."""
    def param_groups(model):
        named = dict(model.named_parameters())
        state["group_names"] = [list(GATE), [n for n, _ in model.named_parameters() if n not in GATE]]
        state["groups"] = [dict(params=[named[n] for n in GATE], lr=LR),
                           dict(params=[p for n, p in model.named_parameters() if n not in GATE], lr=warm_lr)]
        return state["groups"]
    return param_groups


@torch.no_grad()
def extra_stats(model, task, probe, step, base, state):
    """Pooled eta^2 of the read gate at key positions on the probe batch by index, half, key and
    stream; at k=2 the routing statistics where the run path has none; the hinge's counts."""
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    S, n = task.S, task.S * task.P
    B = pinp.shape[0]
    j = torch.arange(n)
    labels = dict(index=j.expand(B, n), half=(j >= n // 2).long().expand(B, n), key=pinp[:, 3 * j + 1] - S,
                  stream=pinp[:, 3 * j])
    out = {f"hs_eta_{g}": eta2_multi(gr[:, 3 * j + 1], lab) for g, lab in labels.items()}
    if model.n_ch == 2 and "margin" not in base:
        out.update(routing_stats(model, task, probe))
    if "n" in state:
        out["hs_hinge"] = [state["fired"], state["n"]]
        if step == "end":
            out["hs_windows"] = list(state["windows"])
    return out


def make_recipe(slow, tau, warm=WARM, warm_lr=LR_WARM, lam=LAMBDA, keep=None):
    """recipe(kw) -> kw for a run path's run_one call, and its state. slow: SLOW's groups and lr
    switch; tau (None = no hinge): HINGE's hook; every arm: extra_stats; keep: run_one's keep."""
    state = {}

    def recipe(kw):
        kw = dict(kw)
        task = kw["task"]
        if tau is not None:
            b0 = kw.get("builder") or trr.make_fn

            def builder(a, seed):
                mk = b0(a, seed)

                def make():
                    m = mk()
                    attach_hinge(m, task.S, task.P, tau, lam, state)
                    return m
                return make
            kw["builder"] = builder
        if slow:
            kw["run_kw"] = dict(kw.get("run_kw") or {}, param_groups=slow_groups(state, warm_lr))
            c0 = kw.get("check")

            def check(m, step, data):
                if c0 is not None:
                    c0(m, step, data)
                if step >= warm:
                    state["groups"][1]["lr"] = LR
            kw["check"] = check
        s0 = kw["stats_fn"]

        def stats_fn(model, task_, probe, step):
            out = s0(model, task_, probe, step)
            out.update(extra_stats(model, task_, probe, step, out, state))
            return out
        kw["stats_fn"] = stats_fn
        if keep is not None:
            kw["keep"] = keep
        return kw
    return recipe, state


# ── Runs ─────────────────────────────────────────────────────────────────────
def part_run(part, seed, iters, recipe=None):
    """One run on the part's recorded run path (recipe None = that path unchanged)."""
    if part == "0":
        return tsc.run_job(dict(arm="A", seed=seed, iters=iters), recipe=recipe)
    if part == "A":
        return tsa.run_job(dict(arm="DIRECT8", seed=seed, iters=iters), recipe=recipe)
    if part == "B":
        return tsr.run_attempt(tsr.ARM["A4k16"], seed, dict(tsr.REAL, total=iters), recipe=recipe)
    return tscur.run_sc(tscur.ARM["D8"], seed, dict(tscur.REAL, total=iters), recipe=recipe)


def run_job(sp):
    a = ARM[sp["arm"]]
    sched = sp["sched"]
    rc, _ = make_recipe(a["slow"], a["tau"], warm=sched["warm"])
    return part_run(a["part"], sp["seed"], iters_of(a["part"], sched), rc)


def spec(key, seed, sched):
    return dict(arm=key, seed=seed, sched=dict(sched))


def variant_job(part, seed, iters, slow=False, tau=None, warm_lr=LR_WARM, keep_at=None, plain=False):
    """A check run on the part's path: plain (recipe None, or one that only sets keep), or the recipe
    (slow, tau, warm_lr); with keep_at, the weights right after that step."""
    keep = None if keep_at is None else {"at": keep_at}
    if plain:
        rc = None if keep is None else (lambda kw: dict(kw, keep=keep))
    else:
        rc, _ = make_recipe(slow, tau, warm_lr=warm_lr, keep=keep)
    rec = part_run(part, seed, iters, rc)
    return dict(rec=rec, snap=None if keep is None else keep.get("snap"))


def knob_job(path, variant):
    """CHECK 101: one short run of a run path, through 520fe51's module (legacy) or this one with
    recipe None (new) or the identity recipe (ident)."""
    tiny_r = dict(tsr.REAL, total=60, eval_every=20, t_check=40)
    tiny_c = dict(tscur.REAL, t1=20, t2=40, total=60, eval_every=20, t_check=20)
    fname, modname, cur = {"tsc": ("test_short_conv.py", "test_short_conv_legacy", tsc),
                           "tsa": ("test_scale_axes.py", "test_scale_axes_legacy", tsa),
                           "tsr": ("test_stream_recipe.py", "test_stream_recipe_legacy", tsr),
                           "tscur": ("test_stream_curriculum.py", "test_stream_curriculum_legacy", tscur)}[path]
    mod = load_at(LEGACY_SHA, fname, modname) if variant == "legacy" else cur
    kw = {} if variant != "ident" else dict(recipe=lambda k: k)
    if path == "tsc":
        r = mod.run_job(dict(arm="A", seed=3, iters=EVAL_EVERY), **kw)
    elif path == "tsa":
        r = mod.run_job(dict(arm="DIRECT8", seed=3, iters=EVAL_EVERY), **kw)
    elif path == "tsr":
        r = mod.run_attempt(mod.ARM["A4k16"], 3, tiny_r, **kw)
    else:
        r = mod.run_sc(mod.ARM["D8"], 3, tiny_c, **kw)
    return strip_all(r)


def lr_job(key):
    """CHECK 108: through the arm's run_job path and real schedule, an optimizer step pre-hook records
    each group's lr at every update and the groups' parameters; the forward is stubbed and every
    held-out accuracy is 0.5, so the run goes to MAX_ITERS (the statistics still run on the model)."""
    a = ARM[key]
    lrs, groups = [], {}

    def hook(opt, args, kwargs):
        if not groups:
            groups["ids"] = [[id(p) for p in g["params"]] for g in opt.param_groups]
        lrs.append(tuple(g["lr"] for g in opt.param_groups))

    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    keep = {"at": 1}
    try:
        rc, _ = make_recipe(a["slow"], a["tau"], keep=keep)
        rec = part_run(a["part"], 3, iters_of(a["part"], REAL), rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    names = {id(p): n for n, p in keep["model"].named_parameters()}
    runs = []
    for x in lrs:
        if runs and runs[-1][0] == list(x):
            runs[-1][1] += 1
        else:
            runs.append([list(x), 1])
    return dict(ok=rec["ok"], n=len(lrs), runs=runs, names=[[names[i] for i in g] for g in groups.get("ids", [])],
                all_names=[n for n, _ in keep["model"].named_parameters()], stopped=rec.get("stopped_at"))


def source(part):
    """The part's recorded learned-gate arm and its builder (the run path's own)."""
    if part == "0":
        return tsc.ARM["A"], trr.make_fn
    if part == "A":
        return tsa.ARM["DIRECT8"], tsa.builder_for(tsa.ARM["DIRECT8"])
    if part == "B":
        return tsr.ARM["A4k16"], tsr.builder_for(tsr.ARM["A4k16"])
    return tscur.ARM["D8"], tsr.builder_for(tscur.ARM["D8"])


def ceiling(part):
    """The part's perfect-gate arm and its builder (CHECK 106)."""
    if part == "0":
        a = tsc._BASE["ceiling"]
        return a, trr.make_fn
    a = tsa.ARM["ceiling8"] if part == "A" else tsr.ARM["ceiling4k16" if part == "B" else "ceiling8k16"]
    return a, (tsa.builder_for(a) if part == "A" else tsr.builder_for(a))


def maker(part, seed, slow=False, tau=None):
    """make() for the part's learned-gate model at `seed`, through the recipe's builder."""
    a, b0 = source(part)
    rc, state = make_recipe(slow, tau)
    kw = rc(dict(task=PART[part]["task"], builder=b0, stats_fn=lambda *x: {}))
    return kw["builder"](a, seed), state


def time_job(key):
    """ms/step through the arm's builder and hinge (test_scale_axes.time_job's recipe), with the
    held-out evaluation charged once per EVAL_EVERY."""
    a = ARM[key]
    task = PART[a["part"]]["task"]
    mk, _ = maker(a["part"], 0, a["slow"], a["tau"])
    per = time_per_step(task, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(task)
    t0 = time.time()
    tbo.evaluate(m, task, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


def screen_job(key, seed, iters):
    """CHECK 112: SLOW0 or HINGE0 (the arms' real recipe) for `iters` steps."""
    a = ARM[key]
    rc, st = make_recipe(a["slow"], a["tau"])
    return part_run("0", seed, iters, rc)
# ── Verification ─────────────────────────────────────────────────────────────
def norm(x):
    """A record as the results file holds it (JSON round trip)."""
    return json.loads(json.dumps(x))


def subset_equal(new, old):
    """Every field of `old` is in `new` with the same value, at any depth (dicts: old's keys;
    lists: same length, elementwise); wall time is ignored."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and subset_equal(new[k], v) for k, v in old.items() if k != "secs")
    if isinstance(old, list):
        return isinstance(new, list) and len(new) == len(old) and all(subset_equal(a, b) for a, b in zip(new, old))
    if isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new):
        return True
    return new == old


def trunc(r, t):
    """A record's curve, statistics, gradient norms and early/stage curves through step t."""
    out = dict(curve=[c for c in r["curve"] if c[0] <= t],
               stats=[x for x in r["stats"] if x["step"] != "end" and x["step"] <= t],
               grad={k: v for k, v in r["grad"].items() if int(k) <= t})
    for f in ("early", "curve_stage"):
        if f in r:
            out[f] = [c for c in r[f] if c[0] <= t]
    return out


def same_snap(a, b):
    return a is not None and b is not None and sorted(a) == sorted(b) and all(torch.equal(a[n], b[n]) for n in a)


def fired_of(r):
    e = r.get("end") or {}
    return e.get("hs_hinge", [None, None])


def gate_model(part, seed):
    """The part's learned-gate model at `seed` with its gate redrawn far from uniform (its own
    generator): W_g at scale 3, W_in and W_h at 0.3, three times their initial 0.1 (CHECK 105)."""
    mk, _ = maker(part, seed)
    torch.manual_seed(seed)
    m = mk()
    g = torch.Generator().manual_seed(seed + 105)
    with torch.no_grad():
        for name, sc in (("W_g", 3.0), ("W_in", 0.3), ("W_h", 0.3)):
            w = getattr(m, name)
            w.copy_(torch.randn(w.shape, generator=g) * sc)
    return m


class GateStub:
    """Stands in for a model in routing_stats (CHECK 105): it returns a fixed gate as read and write gate."""
    def __init__(self, gr):
        self.gr, self.training = gr, False

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, x, tau=None):
        return None, None, self.gr, self.gr


def synth_gate(B, T, k, S, P, seed):
    """A gate with strong index and half effects at the key positions: softmax of noise plus a
    per-index offset and, in the second half, +4 on channel 0 (CHECK 105)."""
    g = torch.Generator().manual_seed(seed)
    n = S * P
    j = torch.arange(n)
    z = torch.randn(B, T, k, generator=g)
    z[:, 3 * j + 1, :] += 1.5 * torch.randn(n, k, generator=g) + 4.0 * (j >= n // 2).float()[:, None] * torch.eye(k)[0]
    return F.softmax(z, -1)


def eta2_numpy(gr, S, P):
    """CHECK 105: the pooled eta^2 by index and by half, recomputed in NumPy (float64)."""
    import numpy as np
    n = S * P
    p = gr[:, [3 * j + 1 for j in range(n)], :].double().numpy()
    B, k = p.shape[0], p.shape[-1]
    mu = p.reshape(-1, k).mean(0)
    tot = float(((p - mu) ** 2).sum())
    bi = float(sum(B * ((p[:, j, :].mean(0) - mu) ** 2).sum() for j in range(n)))
    lo, hi = p[:, : n // 2, :].reshape(-1, k), p[:, n // 2:, :].reshape(-1, k)
    bh = float(lo.shape[0] * ((lo.mean(0) - mu) ** 2).sum() + hi.shape[0] * ((hi.mean(0) - mu) ** 2).sum())
    return bi / tot, bh / tot, np.__version__


def screen_records():
    """CHECK 112: the screens' recorded SLOW_MEM and SLOW_HINGE runs (explore_out at SCREEN_SHA)."""
    try:
        out = {}
        for f in ("slow_mem", "slow_hinge"):
            src = subprocess.run(["git", "-C", HERE, "show", f"{SCREEN_SHA}:explore_out/{f}_results.json"],
                                 capture_output=True, text=True, check=True).stdout
            out[f] = json.loads(src)["runs"]
        return out
    except Exception as e:
        return f"{type(e).__name__}"


def earlier_seeds():
    """test_stream_curriculum's earlier seeds, plus its own base seeds (260-279) and attempt seeds."""
    out = tscur.earlier_seeds() | {x for v in tscur.SEEDS.values() for x in v}
    out |= {x for s in tscur.SEEDS["SC8_R"] for x in tscur.attempt_seeds(s)}
    return out


def verify(pool, files):
    print("test_stream_curriculum.py's verification (which runs test_stream_recipe's, and so on down to")
    print("test_multilayer_binding's):")
    tscur.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    T2, T3 = WARM, 3 * EVAL_EVERY
    f = {}
    for part in ("C", "B"):
        f["102", part, "slow"] = pool.submit(variant_job, part, CHECK_SEED, T3, True, None, LR_WARM, T3)
        f["102", part, "hinge"] = pool.submit(variant_job, part, CHECK_SEED, T3, True, 1.0, LR_WARM, T3)
        s0 = PART[part]["seed0"]
        f["103", part, "plain"] = pool.submit(variant_job, part, s0, T2, keep_at=T2, plain=True)
        f["103", part, "hinge"] = pool.submit(variant_job, part, s0, T2, True, 1.0, LR, T2)
    f["104"] = pool.submit(variant_job, "0", 160, MAX0)
    for part in ("A",):
        f["102", part, "slow"] = pool.submit(variant_job, part, CHECK_SEED, T3, True, None, LR_WARM, T3)
        f["102", part, "hinge"] = pool.submit(variant_job, part, CHECK_SEED, T3, True, 1.0, LR_WARM, T3)
        s0 = PART[part]["seed0"]
        f["103", part, "plain"] = pool.submit(variant_job, part, s0, T2, keep_at=T2, plain=True)
        f["103", part, "hinge"] = pool.submit(variant_job, part, s0, T2, True, 1.0, LR, T2)
    for v in ("legacy", "new", "ident"):
        f["101", "tsa", v] = pool.submit(knob_job, "tsa", v)
    f["112", "SLOW0", 160] = pool.submit(screen_job, "SLOW0", 160, T3)
    f["112", "HINGE0", 160] = pool.submit(screen_job, "HINGE0", 160, T3)
    f["112", "HINGE0", 162] = pool.submit(screen_job, "HINGE0", 162, T3)
    f["102", "0", "slow"] = pool.submit(variant_job, "0", CHECK_SEED, T3, True, None, LR_WARM, T3)
    f["102", "0", "hinge"] = pool.submit(variant_job, "0", CHECK_SEED, T3, True, 1.0, LR_WARM, T3)
    f["103", "0", "A0"] = pool.submit(variant_job, "0", CHECK_SEED, T2, False, None, LR_WARM, T2)
    f["103", "0", "SLOW0"] = pool.submit(variant_job, "0", CHECK_SEED, T2, True, None, LR, T2)
    f["103", "0", "HINGE0"] = pool.submit(variant_job, "0", CHECK_SEED, T2, True, 1.0, LR, T2)
    for k in KEYS:
        f["108", k] = pool.submit(lr_job, k)
    f["111", "0"] = pool.submit(variant_job, "0", CHECK_SEED, EVAL_EVERY, True, 0.0)
    f["111", "B"] = pool.submit(variant_job, "B", CHECK_SEED, EVAL_EVERY, True, TAU)
    for path in ("tsc", "tsr", "tscur"):
        for v in ("legacy", "new", "ident"):
            f["101", path, v] = pool.submit(knob_job, path, v)

    print(f"CHECK 101 the recipe knob is inert: each run path with recipe=None and with the identity recipe vs {LEGACY_SHA}'s module:")
    good = True
    for path, what in (("tsc", f"test_short_conv.run_job      arm A    seed 3, {EVAL_EVERY} steps"),
                       ("tsa", f"test_scale_axes.run_job      DIRECT8  seed 3, {EVAL_EVERY} steps"),
                       ("tsr", "test_stream_recipe.run_attempt  A4k16 seed 3, 60 steps (evaluations every 20, check at 40)"),
                       ("tscur", "test_stream_curriculum.run_sc   D8    seed 3, 60 steps (evaluations every 20)")):
        L_, n_, i_ = (f["101", path, v].result() for v in ("legacy", "new", "ident"))
        g = L_ == n_ == i_ and L_["ok"]
        good &= g
        print(f"     {what}: legacy == recipe None {L_ == n_}, == identity recipe {L_ == i_}   curve "
              f"{[[c[0], round(c[1], 4)] for c in L_['curve']]}  -> {'UNCHANGED' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 102 with TAU = 1.0, HINGE equals SLOW bit for bit (curves and weights, {T3} steps, seed {CHECK_SEED}) in each "
          f"part's configuration:")
    good = True
    for part in ("0", "A", "B", "C"):
        rs, rh = f["102", part, "slow"].result(), f["102", part, "hinge"].result()
        fired, n = fired_of(rh["rec"])
        same = subset_equal(norm(strip_all(rh["rec"])), norm(strip_all(rs["rec"])))
        w = same_snap(rh["snap"], rs["snap"])
        g = same and w and fired == 0 and n == T3 and rs["rec"]["ok"]
        good &= g
        print(f"     Part {part} ({PART[part]['src']:<28}): records equal {same}   weights at step {T3} equal {w}   hinge fired "
              f"{fired} of {n} batches   curve {[round(c[1], 4) for c in rs['rec']['curve']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 103 with both groups at {LR:g} from step 1 and TAU = 1.0, each new arm equals its plain counterpart bit for "
          f"bit through {T2} steps:")
    good = True
    r_a0 = f["103", "0", "A0"].result()
    for k in ("SLOW0", "HINGE0"):
        r_ = f["103", "0", k].result()
        same = subset_equal(norm(strip_all(r_["rec"])), norm(strip_all(r_a0["rec"])))
        w = same_snap(r_["snap"], r_a0["snap"])
        g = same and w and r_a0["rec"]["ok"]
        good &= g
        print(f"     Part 0 {k:<9} vs A0 (seed {CHECK_SEED}): records equal {same}   weights at {T2} equal {w}   curve "
              f"{[round(c[1], 4) for c in r_a0['rec']['curve']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    plain = {}
    for part, k in (("A", "HINGE8"), ("B", "HINGE4k16"), ("C", "HINGE_D8")):
        rp, rh = f["103", part, "plain"].result(), f["103", part, "hinge"].result()
        plain[part] = rp["rec"]
        s0 = PART[part]["seed0"]
        rec_old = get(files[PART[part]["file"]], PART[part]["rec_arm"], s0)
        same_p = subset_equal(norm(strip_all(rh["rec"])), norm(strip_all(rp["rec"])))
        w = same_snap(rh["snap"], rp["snap"])
        rep = rec_old is not None and subset_equal(trunc(norm(rp["rec"]), T2), trunc(rec_old, T2))
        same_r = rec_old is not None and subset_equal(trunc(norm(rh["rec"]), T2), trunc(rec_old, T2))
        g = same_p and w and rp["rec"]["ok"] and (same_r or not rep)
        good &= g
        print(f"     Part {part} {k:<9} vs the recorded {PART[part]['rec_arm']} seed {s0} (curve, statistics, gradient norms "
              f"through {T2}): {same_r if rep else 'not compared: the recorded run does not reproduce here (CHECK 110)'}; "
              f"vs a fresh plain run: records equal {same_p}, weights at {T2} equal {w}   curve "
              f"{[round(c[1], 4) for c in rp['rec']['curve']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print("CHECK 104 A0's run path reproduces the recorded test_short_conv arm A run, seed 160:")
    rec_old = get(files["short"], "A", 160)
    r104 = f["104"].result()["rec"]
    g = rec_old is not None and subset_equal(norm(strip_all(r104)), rec_old) and r104["ok"]
    print(f"     file {files['short_path']} (written on {files['short'].get('meta', {}).get('cpu')}, git "
          f"{files['short'].get('meta', {}).get('git')}): recorded run present {rec_old is not None}")
    print(f"     A0's path, seed 160, {MAX0} steps: stopped at {r104.get('stopped_at')}, transition {r104.get('transition')}, "
          f"acc {r104.get('acc')}, VAL cos {round(r104.get('val_cos', float('nan')), 6)}; every recorded field equal (curve, "
          f"statistics, gradient norms, end): {g}  -> {'REPRODUCED' if g else 'DIFFERS'}")
    ok &= g
    print()

    print("CHECK 105 the pooled eta2 at k = 2 equals routing_stats' channel-0 eta^2, and at k = 16 a NumPy recomputation, "
          "on the probe batch of seed 5: (a) learned-gate models with the gate redrawn far from uniform (W_g at scale 3, "
          "W_in and W_h at 0.3); (b) a synthetic gate with strong index and half effects (routing_stats reads it through a stub):")
    good = True
    for part in ("0", "A", "B", "C"):
        task = PART[part]["task"]
        probe = probe_batch(task, 5)
        m = gate_model(part, 5)
        m.eval()
        with torch.no_grad():
            gr_m = m(probe[0], TAU_END)[2]
        gr_s = synth_gate(probe[0].shape[0], probe[0].shape[1], gr_m.shape[-1], task.S, task.P, 105)
        for how, gr, mod in (("(a) model", gr_m, m), ("(b) synthetic", gr_s, GateStub(gr_s))):
            e_i, e_h = (float(x) for x in eta2_hinge(gr, task.S, task.P))
            n_i, n_h, npv = eta2_numpy(gr, task.S, task.P)
            if PART[part]["k"] == 2:
                rs = routing_stats(mod, task, probe)
                d = max(abs(e_i - rs["eta_key_by_index"]), abs(e_h - rs["eta_key_by_half"]))
                what = f"routing_stats index {rs['eta_key_by_index']:.6f}, half {rs['eta_key_by_half']:.6f}"
            else:
                d = max(abs(e_i - n_i), abs(e_h - n_h))
                what = f"NumPy {npv} float64 index {n_i:.6f}, half {n_h:.6f}"
            g = d < 1e-6 and max(abs(e_i - n_i), abs(e_h - n_h)) < 1e-6
            good &= g
            print(f"     Part {part} (S={task.S}, P={task.P}, k={PART[part]['k']:<2}) {how:<13}: pooled index {e_i:.6f}, half "
                  f"{e_h:.6f}; {what}; max |diff| {d:.1e}  -> {'EQUAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print("CHECK 106 the perfect gate on 20 probe batches (seeds 0-19), and a gate set by sequence half:")
    good = True
    for part in ("0", "A", "B", "C"):
        task = PART[part]["task"]
        a, b = ceiling(part)
        torch.manual_seed(3)
        m = b(a, 3)()
        m.eval()
        vals = []
        for s in range(20):
            with torch.no_grad():
                gr = m(probe_batch(task, s)[0], TAU_END)[2]
            vals.append([float(x) for x in eta2_hinge(gr, task.S, task.P)])
        mi, mh = max(v[0] for v in vals), max(v[1] for v in vals)
        n = task.S * task.P
        k = gr.shape[-1]
        half = torch.zeros(BATCH, task.L - 1, k)
        for j in range(n):
            half[:, 3 * j + 1, 0 if j < n // 2 else 1] = 1.0
        hh = [float(x) for x in eta2_hinge(half, task.S, task.P)]
        g = mi < PERFECT_MAX and mh < PERFECT_MAX and hh[1] > HALF_MIN
        good &= g
        print(f"     Part {part} ({a['key']}, k={k}): perfect gate max eta2_index {mi:.4f}, max eta2_half {mh:.4f} (< {PERFECT_MAX}); "
              f"half gate eta2_half {hh[1]:.4f} (> {HALF_MIN})  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 107 the hinge's gradient (learned-gate models at init, seed 5, training batches from the run's generator):")
    good = True
    want = set(GATE) | {"embed.weight"}
    for part in ("0", "A", "B", "C"):
        task = PART[part]["task"]
        rng = torch.Generator().manual_seed(5 + 10_000)
        tok, _ = task.make_batch(BATCH, rng)
        x = tok[:, :-1]
        mk, _ = maker(part, 5)
        torch.manual_seed(5)
        m = mk()
        m.train()
        e_i, e_h = eta2_hinge(m(x)[2], task.S, task.P)
        (F.relu(e_i - 0.0) + F.relu(e_h - 0.0)).backward()
        got = {n for n, p in m.named_parameters() if p.grad is not None and bool((p.grad != 0).any())}
        mk2, st2 = maker(part, 5, tau=0.0)
        torch.manual_seed(5)
        m2 = mk2()
        m2.train()
        (m2(x)[0] * 0.0).sum().backward()
        got2 = {n for n, p in m2.named_parameters() if p.grad is not None and bool((p.grad != 0).any())}
        below, zero, vals = None, False, None
        for b_ in range(20):
            tok, _ = task.make_batch(BATCH, rng)
            torch.manual_seed(5)
            m3 = mk()
            m3.train()
            f_i, f_h = eta2_hinge(m3(tok[:, :-1])[2], task.S, task.P)
            if float(f_i) < TAU and float(f_h) < TAU:
                (F.relu(f_i - TAU) + F.relu(f_h - TAU)).backward()
                below, vals = b_ + 2, (float(f_i), float(f_h))
                zero = all(p.grad is None or bool((p.grad == 0).all()) for p in m3.parameters())
                break
        g = got == want and got2 == want and st2["fired"] == 1 and below is not None and zero
        good &= g
        print(f"     Part {part}: with TAU = 0 the gradient reaches {sorted(got)}; through the hook (task loss x 0) "
              f"{sorted(got2)}; on batch {below} (eta2 index {vals[0] if vals else float('nan'):.3f}, half "
              f"{vals[1] if vals else float('nan'):.3f}, both below {TAU}) the gradient is exactly zero: {zero}  -> "
              f"{'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 108 the optimizer groups and lrs at every update, through each arm's run path and real schedule (forward "
          "stubbed, held-out accuracy 0.5; step pre-hook):")
    good = True
    for k in KEYS:
        a = ARM[k]
        r = f["108", k].result()
        it = iters_of(a["part"], REAL)
        if a["slow"]:
            want_runs = [[[LR, LR_WARM], WARM], [[LR, LR], it - WARM]]
            rest = [n for n in r["all_names"] if n not in GATE]
            g = (r["runs"] == want_runs and r["names"] == [list(GATE), rest] and "embed.weight" in rest
                 and r["n"] == it and r["ok"] and r["stopped"] == it)
            conv = "short_conv.conv_w" in rest
            print(f"     {k:<9} {r['n']} updates: group lrs {r['runs'][0][0]} x {r['runs'][0][1]}"
                  + (f", then {r['runs'][1][0]} x {r['runs'][1][1]}" if len(r["runs"]) > 1 else "")
                  + f"   gate group {r['names'][0]}   other group {len(rest)} tensors (embed.weight included: "
                  f"{'embed.weight' in rest}; convolution included: {conv})  -> {'OK' if g else 'WRONG'}")
        else:
            g = r["runs"] == [[[LR], it]] and len(r["names"]) == 1 and r["n"] == it and r["ok"]
            print(f"     {k:<9} {r['n']} updates: one group at {r['runs'][0][0]} x {r['runs'][0][1]} (arm A's optimizer)  -> "
                  f"{'OK' if g else 'WRONG'}")
        good &= g
    ok &= good
    print()

    print("CHECK 109 the seeds:")
    earlier = earlier_seeds()
    new = set(S20)
    g = not (new & earlier)
    print(f"     Part 0's seeds {tsr.ranges(S20)}: disjoint from every earlier seed: {g}")
    print(f"     (earlier seeds, restart reseeds included: {tsr.ranges(earlier)})")
    print(f"     Parts A-C reuse their recorded arms' seeds by design (paired): {tsr.ranges(SEEDS['HINGE8'])}, "
          f"{tsr.ranges(SEEDS['HINGE4k16'])}, {tsr.ranges(SEEDS['HINGE_D8'])}")
    ok &= g
    print()

    print(f"CHECK 110 the pairing with this machine's recorded files (a file pairs if its recorded run of the part's first seed "
          f"reproduces bit for bit through step {T2} on the plain run path, whatever the CPU label):")
    pairs = {}
    for part in ("A", "B", "C"):
        fk = PART[part]["file"]
        store, path = files[fk], files[f"{fk}_path"]
        arm, s0 = PART[part]["rec_arm"], PART[part]["seed0"]
        meta = store.get("meta", {})
        seeds = SEEDS[{"A": "HINGE8", "B": "HINGE4k16", "C": "HINGE_D8"}[part]]
        have = [s for s in seeds if (r_ := get(store, arm, s)) and r_.get("ok")]
        rec_old = get(store, arm, s0)
        rp = plain[part]
        rep = rec_old is not None and subset_equal(trunc(norm(rp), T2), trunc(rec_old, T2))
        pairs[part] = bool(rep)
        print(f"     Part {part}: {path}: exists {os.path.exists(path)}   written on: {meta.get('cpu')} (git {meta.get('git')})   "
              f"this machine: {cpu_model()}   labels match: {meta.get('cpu') == cpu_model()}")
        print(f"         recorded {arm} runs on seeds {tsr.ranges(seeds)}: {len(have)}/{len(seeds)}")
        print(f"         {arm} seed {s0} re-run (plain path), {T2} steps: curve {rp['curve']} vs recorded "
              f"{[c for c in rec_old['curve'] if c[0] <= T2] if rec_old else None}; statistics and gradient norms equal: {rep}  "
              f"-> {'PAIRS' if rep else 'DOES NOT PAIR (that part is UNTESTED)'}")
    print()

    print("CHECK 111 a worker's run is bit-identical to the same run here:")
    good = True
    for part, tau, what in (("0", 0.0, "HINGE0 with TAU = 0 (fires on every batch)"), ("B", TAU, "HINGE4k16")):
        rw = f["111", part].result()["rec"]
        rh = variant_job(part, CHECK_SEED, EVAL_EVERY, True, tau)["rec"]
        fired, n = fired_of(rh)
        g = strip_all(rw) == strip_all(rh) and rh["ok"] and (tau != 0.0 or fired == n == EVAL_EVERY)
        good &= g
        print(f"         {what}, seed {CHECK_SEED}, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   hinge fired "
              f"{fired} of {n} batches   acc {rh['curve'][-1][1]:.4f}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 112 (informative, not asserted) SLOW0 and HINGE0 against the screens' recorded runs (explore_out at "
          f"claude/outside-ideas {SCREEN_SHA}), {T3} steps:")
    scr = screen_records()
    if isinstance(scr, str):
        print(f"     skipped: the screens' records are not readable here ({scr})")
    for key, seed, sname, sk in (("SLOW0", 160, "slow_mem", "SLOW_MEM"), ("HINGE0", 160, "slow_hinge", "SLOW_HINGE"),
                                 ("HINGE0", 162, "slow_hinge", "SLOW_HINGE")):
        r = f["112", key, seed].result()
        if isinstance(scr, str):
            continue
        old = scr[sname].get(f"{sk}|{seed}")
        oc = [c for c in old["curve"] if c[0] <= T3] if old else None
        hc = [x.get("hinge_counts") for x in old["stats"] if x["step"] in (EVAL_EVERY, 2 * EVAL_EVERY, T3)] if old else None
        mine = [x.get("hs_hinge") for x in r["stats"] if x["step"] in (EVAL_EVERY, 2 * EVAL_EVERY, T3)]
        print(f"     {key:<6} seed {seed} vs the screen's {sk}: curves equal {r['curve'] == oc}   "
              f"curve {[round(c[1], 4) for c in r['curve']]} vs {[round(c[1], 4) for c in oc] if oc else None}"
              + (f"   hinge fired/batches at 1200, 2400, 3600: {mine} vs the screen's {hc}" if key == "HINGE0" else ""))
    print()

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}"
          + ("" if all(pairs.values()) else f"   (CHECK 110: Part(s) {', '.join(p for p in pairs if not pairs[p])} do not pair: "
                                              f"see PAIRING)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(pairs=pairs)


# ── Reporting ────────────────────────────────────────────────────────────────
def ok_r(r):
    return bool(r and r.get("ok"))


def st(r, step):
    if step == "end":
        return r.get("end")
    return next((x for x in r.get("stats", []) if x.get("step") == step), None)


def routed_star(x):
    """The screens' ROUTED* at k=2: margin >= 0.9 and eta_key_by_stream > 0.9."""
    return bool(x) and x.get("margin", -1.0) >= ROUTE_MARGIN and x.get("eta_key_by_stream", 0.0) > ROUTE_ETA


def rstar(r, step):
    x = st(r, step)
    return "?" if not x or "margin" not in x else ("Y" if routed_star(x) else "-")


def eta_str(r, step):
    x = st(r, step)
    if not x or "hs_eta_index" not in x:
        return "--"
    return "/".join(f"{x[f'hs_eta_{g}']:.2f}" for g in ETA_GROUPS)


def hinge_counts(r):
    w = (r.get("end") or {}).get("hs_windows")
    return None if w is None else (sum(w[:1]), sum(w[:2]), sum(w))


def hstr(r):
    h = hinge_counts(r)
    return "--" if h is None else "/".join(str(x) for x in h)


def flagged(r):
    """The label-free check at 1200: any pooled eta^2 by index, half or key >= 0.5."""
    x = st(r, EVAL_EVERY)
    if not x or "hs_eta_index" not in x:
        return None
    return any(x[f"hs_eta_{g}"] >= FLAG_ETA for g in FLAG_GROUPS)


def tag(part, r):
    if r is None:
        return "not run"
    if not r.get("ok"):
        return "FAILED"
    if PART[part]["k"] == 16:
        return outcome(r) + ("  collapsed" if r["collapsed"] else "")
    if r["transition"] is not None:
        if part == "0":
            return "DISCOVERED" if r["discovered"] else "BOUND, VAL cos >= 0.5"
        return "BOUND"
    return fail_class(r) + ("  collapsed" if r["collapsed"] else "")


def runs_of(store, key, seeds):
    return {s: get(store, key, s) for s in seeds if get(store, key, s) is not None}


def recorded(files, part, seeds):
    return runs_of(files[PART[part]["file"]], PART[part]["rec_arm"], seeds)


def curve_str(curve):
    return " ".join(f"{round(100 * a):>3}" for _, a, _ in curve)


def raw_rows(name, part, runs, seeds):
    k16 = PART[part]["k"] == 16
    if k16:
        print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'map':>23} {'1:1':>3} {'sh':>2}  "
              f"{'per-stream acc':<39} {'ROUTED':>6}  {'eta i/h/k/s @1200':>19} {'@4800':>19} {'@end':>19} "
              f"{'hinge 1200/2400/all':>19} {'flag':>4}  outcome")
    else:
        print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'VALcos':>7} {'m end':>6} {'ROUTED* 1200/2400/end':>21}  "
              f"{'eta i/h/k/s @1200':>19} {'@4800':>19} {'@end':>19} {'hinge 1200/2400/all':>19} {'flag':>4}  outcome")
    for s in seeds:
        r = runs.get(s)
        if not ok_r(r):
            print(f"  {name:<10} {s:>4}  {'NOT RUN' if r is None else 'FAILED — ' + str(r.get('error'))}")
            continue
        fl = flagged(r)
        fls = "--" if fl is None else ("Y" if fl else "-")
        tail = (f"{eta_str(r, EVAL_EVERY):>19} {eta_str(r, 4800):>19} {eta_str(r, 'end'):>19} {hstr(r):>19} {fls:>4}  "
                f"{tag(part, r)}")
        if k16:
            e = r["end"]
            print(f"  {name:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {mstr(e['ch_map']):>23} "
                  f"{'y' if e['one_to_one'] else 'n':>3} {shared_max(r):>2}  "
                  f"{'/'.join(f'{v:.2f}' for v in e['stream_acc']):<39} {'Y' if routed(r) else '-':>6}  " + tail)
        else:
            rt = "/".join(rstar(r, x) for x in (EVAL_EVERY, 2 * EVAL_EVERY, "end"))
            print(f"  {name:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {r['val_cos']:>7.3f} "
                  f"{fmt(r['end'].get('margin')):>6} {rt:>21}  " + tail)
    print()


def paired(new, old, seeds, fn):
    both = [s for s in seeds if ok_r(new.get(s)) and ok_r(old.get(s))]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    return dict(n=len(both), new=sum(bool(fn(new[s])) for s in both), old=sum(bool(fn(old[s])) for s in both), b=b, c=c,
                p=mcnemar_greater(b, c))


def describe(key, rec):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    part = ARM[key]["part"]
    tr = rec["transition"]
    head = f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5}"
    if PART[part]["k"] == 16:
        e = rec["end"]
        mid = f"map [{mstr(e['ch_map'])}] shared {shared_max(rec)} streams {[round(v, 2) for v in e['stream_acc']]}"
    else:
        mid = (f"VALcos {rec['val_cos']:.3f} ROUTED* 1200/2400/end "
               f"{'/'.join(rstar(rec, x) for x in (EVAL_EVERY, 2 * EVAL_EVERY, 'end'))}")
    return f"{head} {mid}  {tag(part, rec)}  hinge {hstr(rec)}"


def report(store, files, info, drop, wall, path, also):
    pairs = info["pairs"]
    print()
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print("  (acc = final held-out accuracy; VALcos = final value-role gate cosine between streams; m end = final routing margin; "
          "ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9 (the screens'); eta i/h/k/s = pooled eta^2 of the read gate at "
          "key positions on the probe batch by index / half / key / stream; hinge = training batches with a hinge term > 0 in "
          "updates 1-1200 / 1-2400 / the whole run; flag = the label-free check at 1200 (eta^2 by index, half or key >= 0.5); "
          "map = stream -> channel argmax at value positions, stream 0 first; sh = streams on the most crowded channel; "
          "ROUTED = one-to-one and every stream >= 0.9)")
    if drop:
        print(f"  dropped by the drop rule: {', '.join(drop)}")
    print()
    for key in KEYS:
        part = ARM[key]["part"]
        print(f"  {ARM[key]['label']}   (outcome {outcome_key(part)})")
        raw_rows(key, part, runs_of(store, key, SEEDS[key]), SEEDS[key])
        if key in PAIRED and SEEDS[key]:
            arm = PART[part]["rec_arm"]
            print(f"  the recorded {arm} on the same seeds ({files[PART[part]['file'] + '_path']}; "
                  f"{'pairs' if pairs[part] else 'DOES NOT PAIR'}):")
            raw_rows(arm, part, recorded(files, part, SEEDS[key]), SEEDS[key])

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for key in KEYS:
        part = ARM[key]["part"]
        rs = runs_of(store, key, SEEDS[key])
        done = [s for s in SEEDS[key] if ok_r(rs.get(s))]
        line = (f"  {ARM[key]['label']:<52} {outcome_key(part)} {sum(success(part, rs[s]) for s in done):>2}/{len(SEEDS[key])}"
                f"   bound {sum(bound_r(rs[s]) for s in done):>2}   collapsed {sum(bool(rs[s]['collapsed']) for s in done):>2}"
                f"   completed {len(done)}/{len(SEEDS[key])}")
        if PART[part]["k"] == 16:
            line += f"   ROUTED at the end {sum(routed(rs[s]) for s in done)}   one-to-one {sum(bool(rs[s]['end']['one_to_one']) for s in done)}"
        else:
            line += "   ROUTED* 1200/2400/end " + "/".join(str(sum(rstar(rs[s], x) == "Y" for s in done))
                                                         for x in (EVAL_EVERY, 2 * EVAL_EVERY, "end"))
        print(line)
        if key in PAIRED and SEEDS[key]:
            rr = recorded(files, part, SEEDS[key])
            dn = [s for s in SEEDS[key] if ok_r(rr.get(s))]
            print(f"  {'  recorded ' + PART[part]['rec_arm'] + ' (same seeds)':<52} BOUND {sum(bound_r(rr[s]) for s in dn):>2}/"
                  f"{len(SEEDS[key])}   collapsed {sum(bool(rr[s]['collapsed']) for s in dn):>2}   present {len(dn)}/{len(SEEDS[key])}")
    print()

    print("#" * 100)
    print("PAIRING")
    print("#" * 100)
    for part in ("A", "B", "C"):
        fk = PART[part]["file"]
        m_ = files[fk].get("meta", {})
        print(f"  Part {part}: {files[fk + '_path']} (written on {m_.get('cpu')}, git {m_.get('git')}; this machine {cpu_model()}): "
              f"{'PAIRS (a recorded run reproduced bit for bit, CHECK 110)' if pairs[part] else 'DOES NOT PAIR: UNTESTED'}")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, new arm higher; SHOWN if p < 0.05)")
    print("#" * 100)
    res = {}
    for name, new_k, old in CLAIMS:
        part = ARM[new_k]["part"]
        fn = (lambda r, part=part: success(part, r))
        new = runs_of(store, new_k, SEEDS[new_k])
        if old in ("A", "B"):
            old_name, old_runs, untested = PART[old]["rec_arm"], recorded(files, old, SEEDS[new_k]), not pairs[old]
            why = f"the {PART[old]['file']} file does not pair"
        else:
            old_name, old_runs = old, runs_of(store, old, SEEDS[old])
            untested = not SEEDS[old] or not SEEDS[new_k]
            why = "SLOW0 dropped by the drop rule"
        d = paired(new, old_runs, SEEDS[new_k], fn)
        if untested or d["n"] == 0:
            verdict = "UNTESTED"
        else:
            verdict = "SHOWN" if d["p"] < ALPHA else "NOT SHOWN"
        res[name] = verdict
        print(f"  {name:<4} {new_k} beats {old_name} ({outcome_key(part)}): {new_k} {d['new']}/{d['n']} vs {old_name} "
              f"{d['old']}/{d['n']}; {new_k} only {d['b']}, {old_name} only {d['c']}; p = {d['p']:.4g}"
              + (f"   ({why})" if verdict == "UNTESTED" and untested else ""))
        print(f"     *** {name}: {verdict} ***")
    for key in ("HINGE0", "HINGE8", "HINGE4k16"):
        part = ARM[key]["part"]
        rs = runs_of(store, key, SEEDS[key])
        c = sum(success(part, r) for r in rs.values())
        print(f"  band {key:<10} ({outcome_key(part)}, no restarts; RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9, NEVER 0): "
              f"{c}/{len(SEEDS[key])}  -> {band(c)}")
    print()
    print("  READINGS:")
    lines = []
    if res["H0"] == "SHOWN":
        lines.append(("H0 SHOWN", READINGS["H0"]))
    ha, hb = res["HA"] == "SHOWN", res["HB"] == "SHOWN"
    if ha and hb:
        lines.append(("HA and HB SHOWN", READINGS["HA HB"]))
    elif ha:
        lines.append(("HA SHOWN, HB not", READINGS["HA"]))
    elif hb:
        lines.append(("HB SHOWN, HA not", READINGS["HB"]))
    for why, txt in lines:
        print(f"     {why}: \"{txt}\"")
    if not lines:
        print(f"     none of the claim readings applies (H0 {res['H0']}, HA {res['HA']}, HB {res['HB']})")
    if SEEDS["HINGE_D8"]:
        rs = runs_of(store, "HINGE_D8", SEEDS["HINGE_D8"])
        done = [r for r in rs.values() if ok_r(r)]
        nc = sum(bool(r["collapsed"]) for r in done)
        txt = READINGS["C+"] if nc <= COLLAPSE_MAX else READINGS["C-"]
        print(f"     Part C: {nc} of {len(done)} HINGE_D8 runs end collapsed (final accuracy < {COLLAPSE}; at most "
              f"{COLLAPSE_MAX} -> prevents): \"{txt}\"")
    else:
        print("     Part C: not run (drop rule)")
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  Part 0 per seed, side by side (hinge = HINGE0's firing counts 1-1200/1-2400/all):")
    r0 = {k: runs_of(store, k, SEEDS[k]) for k in ("A0", "SLOW0", "HINGE0")}
    for s in S20:
        print(f"    s{s}  A0 {tag('0', r0['A0'].get(s)):<24} SLOW0 {tag('0', r0['SLOW0'].get(s)) if SEEDS['SLOW0'] else '--':<24} "
              f"HINGE0 {tag('0', r0['HINGE0'].get(s)):<24} hinge {hstr(r0['HINGE0'][s]) if ok_r(r0['HINGE0'].get(s)) else '--'}")
    print()
    print("  failure classes and outcomes per arm (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome):")
    groups = [(k, ARM[k]["part"], runs_of(store, k, SEEDS[k])) for k in KEYS if SEEDS[k]]
    groups += [(PART[ARM[k]["part"]]["rec_arm"] + " (recorded)", ARM[k]["part"], recorded(files, ARM[k]["part"], SEEDS[k]))
               for k in PAIRED if SEEDS[k]]
    for name, part, rs in groups:
        cls = {}
        for s, r in sorted(rs.items()):
            cls.setdefault(tag(part, r), []).append(s)
        print(f"    {name:<20} " + "   ".join(f"{c} {len(v)}" for c, v in sorted(cls.items())))
        for c, v in sorted(cls.items()):
            print(f"      {c:<32} {' '.join('s' + str(x) for x in v)}")
    print()
    print(f"  the label-free check at 1200 (any pooled eta^2 by index, half or key >= {FLAG_ETA}) against the outcome:")
    for k in KEYS:
        if not SEEDS[k]:
            continue
        part = ARM[k]["part"]
        rs = [r for r in runs_of(store, k, SEEDS[k]).values() if ok_r(r)]
        fl = [r for r in rs if flagged(r)]
        un = [r for r in rs if flagged(r) is False]
        print(f"    {k:<10} flagged {len(fl):>2}: {outcome_key(part)} {sum(success(part, r) for r in fl):>2}   not flagged {len(un):>2}: "
              f"{outcome_key(part)} {sum(success(part, r) for r in un):>2}")
    print()
    print("  hinge firing (training batches with a term > 0), per arm: runs that never fired; median [min, max] over runs of "
          "updates 1-1200, 1-2400, whole run:")
    for k in ("HINGE0", "HINGE8", "HINGE4k16", "HINGE_D8"):
        if not SEEDS[k]:
            continue
        hs = [hinge_counts(r) for r in runs_of(store, k, SEEDS[k]).values() if ok_r(r) and hinge_counts(r) is not None]
        if not hs:
            continue
        never = sum(1 for h in hs if h[2] == 0)
        print(f"    {k:<10} never fired {never}/{len(hs)}   1-1200 {med_int([h[0] for h in hs])}   1-2400 "
              f"{med_int([h[1] for h in hs])}   whole {med_int([h[2] for h in hs])}")
    print()
    print("  transitions, median [min, max]:")
    for name, part, rs in groups:
        trs = [r["transition"] for r in rs.values() if ok_r(r) and r["transition"] is not None]
        print(f"    {name:<20} {med_int(trs) if trs else 'none bound'}   ({len(trs)} bound)")
    print()
    print("  HINGE vs the recorded arm per seed (descriptive):")
    for k in PAIRED:
        if not SEEDS[k]:
            continue
        part = ARM[k]["part"]
        rs, rr = runs_of(store, k, SEEDS[k]), recorded(files, part, SEEDS[k])
        print(f"    Part {part}: {k} | recorded {PART[part]['rec_arm']}")
        for s in SEEDS[k]:
            a_, b_ = rs.get(s), rr.get(s)
            ta = f"{tag(part, a_)} ({fmt_step(a_['transition']) if ok_r(a_) else '--'})"
            tb = f"{tag(part, b_)} ({fmt_step(b_['transition']) if ok_r(b_) else '--'})"
            print(f"      s{s}  {ta:<40} | {tb}")
    print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps); HINGE runs are followed by the recorded run")
    print("=" * 100)
    for k in KEYS:
        part = ARM[k]["part"]
        rr = recorded(files, part, SEEDS[k]) if k in PAIRED else {}
        for s in SEEDS[k]:
            r = get(store, k, s)
            if ok_r(r):
                print(f"  {k:<10} s{s} {curve_str(r['curve'])}  -> {tag(part, r)}")
            if ok_r(rr.get(s)):
                print(f"  {'(rec ' + PART[part]['rec_arm'] + ')':<10} s{s} {curve_str(rr[s]['curve'])}  -> {tag(part, rr[s])}")
    print()
    if also:
        other = load_store(also)
        print("=" * 100)
        print(f"THE OTHER MACHINE (descriptive): {also} (CPU {other.get('meta', {}).get('cpu')}, git {other.get('meta', {}).get('git')})")
        print("=" * 100)
        for k in KEYS:
            part = ARM[k]["part"]
            ro = {int(x.split("|")[1]): r for x, r in other["runs"].items() if x.split("|")[0] == k}
            done = [r for r in ro.values() if ok_r(r)]
            if done:
                print(f"  {k:<10} {outcome_key(part)} {sum(success(part, r) for r in done)}/{len(done)}   here "
                      f"{sum(success(part, r) for r in runs_of(store, k, SEEDS[k]).values())}/{len(SEEDS[k])}")
        print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


def projection(cost, sched, workers):
    full = {k: iters_of(ARM[k]["part"], sched) * cost[k] for k in KEYS}
    d = [full[k] for k in KEYS for _ in SEEDS[k]]
    return full, d, makespan(d, workers) / 3600


def main():
    ap = argparse.ArgumentParser(description="Slow memory plus a hinge on the gate's position information, no restarts.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--scale", default=tsa.RESULTS_FILE, help="this machine's test_scale_axes results (DIRECT8)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (A4k16)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (D8)")
    ap.add_argument("--short", default=tsc.RESULTS_FILE, help="this machine's test_short_conv results (CHECK 104)")
    ap.add_argument("--also", default=None, help="the other machine's slow_start_results.json, for counts (descriptive)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    sched = dict(SCHED)
    files = {}
    for k in ("scale", "recipe", "curriculum", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)

    print("=" * 100)
    print("Slow start: slow memory plus a hinge on the gate's position information, no restarts")
    print(f"  SLOW: one Adam, gate W_in/W_h/W_g at {LR:g} throughout, every other parameter at {LR_WARM:g} for updates 1-"
          f"{sched['warm']}, {LR:g} after; HINGE = SLOW + {LAMBDA} * [relu(eta2_index - {TAU}) + relu(eta2_half - {TAU})]")
    print(f"  Part 0: S=2, P=4, k=2, no conv, {sched['iters0']} steps (DISCOVERED); Parts A-C: conv, {sched['iters']} steps (BOUND), "
          f"paired with the recorded runs")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<56} seeds {s[0]}-{s[-1]} ({len(s)})")
    for k in ("scale", "recipe", "curriculum", "short"):
        m_ = files[k].get("meta", {})
        print(f"  --{k:<11} {files[k + '_path']} (CPU {m_.get('cpu')}, git {m_.get('git')}, {len(files[k].get('runs', {}))} records)")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        info = verify(pool, files)
        store = load_store(args.results)
        old_drop = None if args.force else store.get("meta", {}).get("drop_decision")
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR, lr_warm=LR_WARM,
                             tau=TAU, lam=LAMBDA, sched=sched, pairs=info["pairs"],
                             files={k: dict(path=files[k + "_path"], cpu=files[k].get("meta", {}).get("cpu"),
                                            git=files[k].get("meta", {}).get("git"))
                                    for k in ("scale", "recipe", "curriculum", "short")},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"), note="seed-level outcomes differ between machines")
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool through each arm's builder and hinge, its "
              f"evaluation charged once per {EVAL_EVERY}; worst case = every run to its MAX_ITERS")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        full, d, total_h = projection(cost, sched, args.workers)
        for k in KEYS:
            print(f"  {k:<10} {cost[k] * 1000:6.1f} ms/step x {iters_of(ARM[k]['part'], sched)} steps x {len(SEEDS[k])} seeds"
                  f"   (a full run {full[k] / 60:.1f} min)")
        print(f"  {len(d)} runs: serial {sum(d) / 3600:.2f} h, {total_h:.2f} h on {args.workers} workers")
        drop, stop = [], False
        if old_drop is not None:
            for k in old_drop["dropped"]:
                SEEDS[k] = ()
                drop.append(k)
            stop = old_drop["stop"]
            full, d, total_h = projection(cost, sched, args.workers)
            print(f"  the drop decision recorded at the first start of this results file is kept: "
                  f"{'dropped ' + ', '.join(drop) if drop else 'nothing dropped'}{'; stop' if stop else ''} (projection now "
                  f"{total_h:.2f} h)")
        else:
            for k in CUTS:
                if total_h <= DROP_H:
                    break
                SEEDS[k] = ()
                drop.append(k)
                full, d, total_h = projection(cost, sched, args.workers)
                print(f"  *** above {DROP_H:g} h: {k} dropped (drop rule); projection now {total_h:.2f} h ***")
            stop = total_h > DROP_H
            if not drop:
                print(f"  within {DROP_H:g} h: nothing is dropped")
            store["meta"]["drop_decision"] = dict(dropped=drop, stop=stop)
        if old_drop is not None:
            store["meta"]["drop_decision"] = old_drop
        store["meta"].update(projected_wall_h=total_h, drop=drop, seeds={k: list(v) for k, v in SEEDS.items()})
        save_results(args.results, store)
        print()
        if stop:
            print(f"  *** still above {DROP_H:g} h after every permitted cut: stopping before training, as the rule says ***")
            return

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<10} seed {sp['seed']}  {describe(sp['arm'], rec)}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

        specs = []
        order = sorted(KEYS, key=lambda k: -full[k])
        part0 = [k for k in order if ARM[k]["part"] == "0"]
        for k in order:
            if ARM[k]["part"] == "0":
                continue
            specs += [spec(k, s, sched) for s in SEEDS[k]]
        for s in S20:
            specs += [spec(k, s, sched) for k in part0 if s in SEEDS[k]]
        todo = []
        for sp in specs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<10} seed {sp['seed']}  cached")
            else:
                todo.append(sp)
        print("=" * 100)
        print(f"RUNS — {len(todo)} runs, longest first")
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

    report(store, files, info, drop, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
