#!/usr/bin/env python
"""
test_early_recipe.py — the hinge only during the slow phase (updates 1-2400), under Muon and under
Adam: does the early window work under Muon (two streams), and does Muon bind four streams sooner?
Everything below is fixed before any run. Both machines run it.

Run it directly (on X the recorded files are results/X/...):

    python test_early_recipe.py --workers 4 --slow results/X/slow_start_results.json \\
        --recipe results/X/stream_recipe_results.json --curriculum results/X/stream_curriculum_results.json \\
        --scale results/X/scale_axes_results.json --short results/X/short_conv_results.json
    (CHECKs 122-123 need claude/outside-ideas fetched: git fetch origin claude/outside-ideas)

BACKGROUND (screens on claude/outside-ideas, EXPLORATORY, 1 thread, X's CPU, bit-identical repro checks)
- S34 (batch 11): the hinge on updates 1-2400 only, against the full hinge:
  - Muon: window 38/40, full 39/40, no hinge 31/40;
  - Adam: window 34/40, full 34/40, no hinge 24/40.
  On every rescued seed the first firing was at updates 49-168. At a firing, the hinge's gradient on
  the gate was a median ~850x (Muon) / 726x (Adam) the task gradient.
- S35: Muon + slow + hinge at S=4, P=4, k=16: 17/20 BOUND (X's HINGE4k16 17/20, 3 vs 3). 15/17 bound
  by 2400, all by 3600 (HINGE4k16 median 7200). Bound by 4800: 12 vs 1.
- S32: Muon + slow without the hinge 31/40 vs with it 39/40 (8 vs 0).
- Main line: test_slow_start H0 SHOWN on X and L; test_recipe_scope Q2 SHOWN on X and L.

OPTIMIZERS
- ADAM: test_slow_start's SLOW schedule. Gate (W_in, W_h, W_g) at 1e-3 throughout; every other
  parameter at 1e-4 for updates 1-2400, then 1e-3.
- MUON: S25's construction, copied from explore_muon_recipe.py at d0c31ce on claude/outside-ideas
  (read with git show): SliceMuon (torch.optim.Muon's update, _single_tensor_muon line for line with
  torch's own Newton-Schulz and lr adjustment; momentum 0.95, Nesterov, 5 Newton-Schulz steps,
  coefficients (3.4445, -4.775, 2.0315), eps 1e-7, weight decay 0) on every 2-D weight and on each
  head slice of a 3-D weight; torch.optim.Adam on the embedding, the conv, biases and 1-D parameters
  (S33's grouping rule, explore_muon_scale_child: Muon decoder, encoder, encoder_v, lm_head, W_in,
  W_h, W_g; Adam embed.weight and short_conv.conv_w); MuonAdam combines them and replaces onset_run's
  torch.optim.Adam for the run only (muon_in_onset_run); run_one's param_groups callback gives the
  groups. Muon lr 0.005: the gate group at 0.005 throughout; the other Muon weights x0.1 for updates
  1-2400; the Adam groups 1e-4 for 1-2400, then 1e-3 (every group set to its full lr right after the
  evaluation at 2400, as SLOW). The perfect gate (validity) runs without the slow phase (S25's lr
  selection).
- WINDOW: test_slow_start's hinge (1.0 x [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)], pooled
  over channels, through its forward hook and gradient-injection node) with weight 1.0 on updates
  1-2400 and 0 afterwards: from update 2401 the penalty is not injected (the model's own forward);
  its two terms are still computed without gradient and an update where one exceeds 0.2 is logged
  as a would-fire (not a firing). test_slow_start gained three knobs for this test, inert at their
  defaults (CHECK 120): attach_hinge / make_recipe window, diag (the firing diagnostics) and stat
  (the statistic function; CHECK 123 only).

ARMS (seeds disjoint from every earlier main-line seed, CHECK 126)
- Part A: S=2, P=4, k=2, no conv (test_short_conv's arm A run path, through test_slow_start.part_run
  "0"), 24000 steps, seeds 300-339 (40). Outcome DISCOVERED (bound and final VAL cos < 0.5).
    SLOW_M   MUON, no hinge
    WIN_M    MUON + WINDOW
- Part B: S=4, P=4, k=16, conv (test_stream_recipe's A4k16 run path, through part_run "B"), 28800
  steps, seeds 340-359 (20). Outcome BOUND, with the transition step.
    WIN16_A  ADAM + WINDOW
    WIN16_M  MUON + WINDOW
- Validity per part: the perfect gate under MUON at 0.005 binds on 2 seeds (300-301: test_short_conv's
  perfect-gate arm without conv on Part A's run_one call; 340-341: test_stream_recipe's ceiling4k16 on
  Part B's path). Otherwise the MUON claims of that part are UNTESTED.

CLAIMS (per machine; exact McNemar one-sided, first arm higher; SHOWN if p < 0.05)
- E1: WIN_M beats SLOW_M (DISCOVERED).
- E2: WIN16_M is bound by update 4800 more often than WIN16_A (transition <= 4800).
- Bands (RELIABLE >= 90%, MAJORITY >= 50%, MINORITY >= 1 run, NEVER 0) for WIN_M, WIN16_A and
  WIN16_M.
- Readings:
  - E1 SHOWN: "the early hinge window works under Muon";
  - E2 SHOWN: "Muon binds four streams sooner".
- Printed, not claims: WIN16_M vs WIN16_A BOUND (McNemar); median transitions.

DIAGNOSTICS (labelled, not part of the verdict)
- Every firing's update; the hinge/task gradient-norm ratio at each firing (on the gate W_in, W_h,
  W_g pooled: the hinge's gradient by torch.autograd.grad of the injected term, the task's as the
  accumulated gradient minus the hinge's, read by an optimizer step pre-hook before the first step of
  that update, S34's method); would-fires after the window.
- Failure classes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome); merged runs
  (two or more streams on one channel at the end, k=16); distinct channels holding the streams and
  streams per channel at 4800, 9600 and the end (k=16).

CHECKS (after the inherited chain, which ends with test_recipe_scope's CHECK 119)
- 120 test_slow_start's new knobs are inert at their defaults: its runs (Parts 0 and B, the hinge at
      TAU = 0 so it fires on every batch) equal c69f1e6's module.
- 121 With window end 24000, an ADAM WINDOW run at Part A's configuration (firing diagnostics on)
      equals the recorded HINGE0 seed 280 (slow_start) bit for bit through 2400 (curve, statistics,
      gradient norms).
- 122 WIN16_M at seed 240 equals S35's MUON_HINGE16 seed 240 through 1200 (curve, statistics,
      gradient norms, the early curve; and the firing updates).
- 123 MUON at Part A's configuration with S13's hinge (explore_slow_pos.pos_penalty at d0c31ce, the
      read gate's channel 0, through the stat knob) equals S25's MUON_HINGE seed 160 through 2400
      (curve; the hinge's counts). The difference between test_slow_start's hinge statistic and S13's
      on a probe batch is printed; the test uses test_slow_start's.
      Reference for 122-123, by the user's decision before any run: the screens' own code
      (explore_muon_k16.child "run" and explore_muon_recipe.run_path at 831c485, run from a temporary
      git worktree of claude/outside-ideas in its own process on this machine), bit for bit. The
      screens' recorded runs (explore_out/muon_k16_results.json, muon_recipe_results.json at 831c485)
      are compared and printed, not asserted: Muon's Newton-Schulz runs in bfloat16 through oneDNN
      (gemm:jit:bf16), whose output bits depend on the CPU's instruction set (on this test's first
      host, a 2.80GHz Xeon without avx512_bf16, the screens' own code reproduced this test's runs
      bit for bit but not the records the screens made on the 2.10GHz host). Muon runs are therefore
      host-dependent; every record carries its host's CPU label, oneDNN's bf16 matmul kernel and a
      fingerprint of the Newton-Schulz output on a fixed probe.
- 124 The optimizer groups cover every parameter exactly once (every arm, both perfect gates); every
      group's lr is correct at updates 1, 2400 and 2401 (each arm's real path and schedule, forward
      and evaluation stubbed); the hinge's weight is 1 at 2400 and 0 at 2401 (real 2402-step runs at
      TAU = 0, so it fires on every batch of the window: 2400 firings, one diagnostic row each, would-
      fires at 2401 and 2402).
- 125 SliceMuon equals torch.optim.Muon (weight decay 0) bit for bit on 2-D weights (S25's CHECK).
- 126 No earlier main-line test used seeds 300-359, restart reseeds included.
- 127 A worker's run is bit-identical to the same run here (WIN16_M seed 1, 1200 steps; WIN_M at
      TAU = 0, 1200 steps, a diagnostic row per update).
- 128 Recorded files pair by bit-for-bit reproduction, whatever the CPU label (slow_start's HINGE0
      seed 280, CHECK 121; no claim pairs with a recorded run). The host's bf16 fingerprint is
      printed.

RUNTIME
- The projection uses pool-load timing: each arm's real path for 1200 steps (one evaluation
  included), run as 4 simultaneous copies on the 4 workers; the median per-step time; worst case =
  every run to its last step.
- If over 10 h, Part B is cut to seeds 340-351 (WIN16_A and WIN16_M). Part A is never cut. The
  decision is stored at the first start of the results file and kept on a resume.
- Runs are scheduled longest first on 4 workers x 1 thread; finished runs are cached.

OUTPUT
Per-seed raw values first, then validity, counts, claims, readings, diagnostics and curves.
Results: early_recipe_results.json (gitignored; X's copy in results/X/).

RESULT (X, test commit 9f25dc8; results/X/early_recipe_results.json)
Run conditions:
- 108/108 records, none failed; 159.3 min of training after verification, 4 workers x 1 thread,
  one start (2.80GHz host, started 2026-10-04 06:26, detached with setsid).
- Every CHECK passed (inherited 1-119, own 120-128). CHECKs 122-123 compare with the screens' own
  code (831c485) run on this machine (the user's decision): both IDENTICAL. The screens' recorded
  runs, written on the 2.10GHz host, differ (Muon's bf16 Newton-Schulz follows the host's oneDNN
  path) and are printed, not asserted. Every record: 2.80GHz, gemm:jit:bf16, NS fingerprint
  f209ef7b8c61.
- Drop rule: projection 10.62 h > 10 h, so Part B was cut to seeds 340-351 (projection 8.35 h).
  Part A ran all 40 seeds.
- Validity: CEIL_A 2/2 and CEIL_B 2/2 bound -> both parts VALID.
Claims:
- E1 WIN_M 40/40 vs SLOW_M 36/40 DISCOVERED, 4 vs 0, p = 0.0625: NOT SHOWN.
- E2 bound by 4800: WIN16_M 10/12 vs WIN16_A 6/12, 5 vs 1, p = 0.11: NOT SHOWN.
- Bands: WIN_M RELIABLE (40/40), WIN16_A MAJORITY (10/12), WIN16_M RELIABLE (11/12).
- Reading: none applies.
- Printed, not claims: BOUND WIN16_M 11/12 vs WIN16_A 10/12 (2 vs 1, p = 0.5); median transition
  SLOW_M 2400, WIN_M 2400, WIN16_A 4800 [3600, 25200], WIN16_M 2400 [2400, 6000].
Diagnostics (not part of the verdict):
- SLOW_M's 4 failures are all POSITION (s304, s314, s320, s332); WIN_M discovered on each of them,
  and on those four seeds the hinge fired early (updates 88-230) at hinge/task ratios 603-50566.
- The hinge fired on 19/40 WIN_M runs (first firing 137 [59, 242], last at 358), 8/12 WIN16_A
  (first 342 [248, 427], last 2303) and 12/12 WIN16_M (first 331 [140, 546], last 761).
  Hinge/task gradient-norm ratio over every firing: WIN_M 3385.6 [7.1, 57759.7], WIN16_A 81.5
  [12.1, 638188.6], WIN16_M 36.3 [8.6, 110420.5].
- Would-fires after the window: WIN_M 2 runs (1 update each), WIN16_A 4 runs (8394 updates, 8384
  of them on s351, which bound at 20400), WIN16_M none.
- WIN16_A failures: s342 non-stream OTHER (3 streams on one channel throughout), s343
  STREAM-PARTIAL one-to-one (all 4 on one channel through 9600, split by the end). WIN16_M's one
  failure, s345, MERGED (2 share) from 4800 on. At 4800, 11/12 WIN16_M gates held the 4 streams on
  4 distinct channels; WIN16_A 6/12.
"""

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import EVAL_EVERY, fmt_step
from test_multilayer_binding import probe_batch
from test_router_curriculum import get, load_store, init_worker, makespan, cpu_model, save_results, TAU_END
from test_router_discovery import load_at
import test_router_reliability as trr
from test_router_reliability import strip_all
from test_conv_lr import bound_r, mcnemar_greater
import test_short_conv as tsc
from test_short_conv import conv_stats, conv_grad_norms
import test_stream_recipe as tsr
from test_stream_channels import med_int, merged, shared_max
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import (LR, LR_WARM, WARM, TAU, GATE, ok_r, hstr, tag, runs_of, curve_str, raw_rows, success,
                             norm, subset_equal, trunc, outcome_key)
import test_recipe_scope as trs
from test_recipe_scope import cmap_at, groups_str

# ── Settings (fixed before any run) ──────────────────────────────────────────
MUON_LR = 0.005
SLOW_SCALE = 0.1
WINDOW = 2400
LOG_UPD = 2400                             # MuonAdam logs the gate's update norm for these updates (S25)
MUON_KW = dict(momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7, ns_steps=5,
               weight_decay=0.0, adjust_lr_fn=None)
MUON_CODE_SHA = "d0c31ce"                  # claude/outside-ideas: explore_muon_recipe.py, the construction copied here
SCREEN_SHA = "831c485"                     # claude/outside-ideas: the screens' recorded runs (CHECKs 122-123)
LEGACY_SHA = "c69f1e6"                     # head before this test's changes
MAX_A, MAX_B = tsc.MAX_ITERS, tsr.MAX_ITERS   # 24000, 28800
BY = 4800                                  # E2: bound by this update
VALID_N = 2
DROP_H = 10.0
CUT_B = tuple(range(340, 352))
TIME_ITERS = 1200
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "early_recipe_results.json"
_ADAM = torch.optim.Adam

REAL = dict(iters_a=MAX_A, iters_b=MAX_B, warm=WARM, window=WINDOW)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ARMS = [
    dict(key="SLOW_M", part="A", opt="muon", slow=True, tau=None, ceil=False, label="SLOW_M   MUON, no hinge (S=2, P=4, k=2)"),
    dict(key="WIN_M", part="A", opt="muon", slow=True, tau=TAU, ceil=False, label="WIN_M    MUON + WINDOW (S=2, P=4, k=2)"),
    dict(key="WIN16_A", part="B", opt="adam", slow=True, tau=TAU, ceil=False,
         label="WIN16_A  ADAM + WINDOW (S=4, P=4, k=16, conv)"),
    dict(key="WIN16_M", part="B", opt="muon", slow=True, tau=TAU, ceil=False,
         label="WIN16_M  MUON + WINDOW (S=4, P=4, k=16, conv)"),
    dict(key="CEIL_A", part="A", opt="muon", slow=False, tau=None, ceil=True,
         label="CEIL_A   perfect gate under MUON (validity, Part A)"),
    dict(key="CEIL_B", part="B", opt="muon", slow=False, tau=None, ceil=True,
         label="CEIL_B   perfect gate ceiling4k16 under MUON (validity, Part B)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
SEEDS = dict(SLOW_M=tuple(range(300, 340)), WIN_M=tuple(range(300, 340)), WIN16_A=tuple(range(340, 360)),
             WIN16_M=tuple(range(340, 360)), CEIL_A=(300, 301), CEIL_B=(340, 341))
TPART = {"A": "0", "B": "B"}               # test_slow_start's part names (tag, raw_rows, outcome)
READINGS = {"E1": "the early hinge window works under Muon", "E2": "Muon binds four streams sooner"}


# ── MUON: S25's construction (explore_muon_recipe.py at d0c31ce), with S33's Adam rule for the conv ──
class SliceMuon(torch.optim.Optimizer):
    """torch.optim.Muon's update (_single_tensor_muon, line for line, torch's own Newton-Schulz and lr
    adjustment) on every 2-D weight and on each 2-D slice of a 3-D weight (one momentum buffer per
    weight, orthogonalised per slice)."""

    def __init__(self, params, lr, momentum, nesterov, ns_coefficients, eps, ns_steps, weight_decay, adjust_lr_fn):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_coefficients=ns_coefficients,
                                      eps=eps, ns_steps=ns_steps, weight_decay=weight_decay,
                                      adjust_lr_fn=adjust_lr_fn))
        for g in self.param_groups:
            for p in g["params"]:
                assert p.ndim in (2, 3), p.shape

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st_ = self.state[p]
                if "momentum_buffer" not in st_:
                    st_["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buf = st_["momentum_buffer"]
                buf.lerp_(grad, 1 - mom)
                update = grad.lerp(buf, mom) if group["nesterov"] else buf
                pairs = [(p, update)] if p.ndim == 2 else [(p[i], update[i]) for i in range(p.shape[0])]
                for w, u in pairs:
                    o = _zeropower_via_newtonschulz(u, group["ns_coefficients"], group["ns_steps"], group["eps"])
                    alr = _adjust_lr(lr, group["adjust_lr_fn"], w.shape)
                    w.mul_(1 - lr * wd)
                    w.add_(o, alpha=-alr)
        return None


class MuonAdam:
    """Adam groups (kind 'adam') on torch.optim.Adam, Muon groups (kind 'muon') on SliceMuon; the group
    dicts are the caller's (an lr switch on them reaches the right optimizer). Logs the gate's update norm
    for the first LOG_UPD steps (the parameters of groups tagged 'gate'), and optionally each group's lr."""

    def __init__(self, groups, lr):
        groups = [g for g in groups if len(g["params"])]
        ad = [g for g in groups if g["kind"] == "adam"]
        mu = [g for g in groups if g["kind"] == "muon"]
        self.adam = _ADAM(ad, lr=lr) if ad else None
        self.muon = SliceMuon(mu, lr=lr, **MUON_KW) if mu else None
        self.opts = [o for o in (self.adam, self.muon) if o is not None]
        self.param_groups = groups
        self.gate = [p for g in groups if g.get("tag") == "gate" for p in g["params"]]
        self.n, self.upd, self.lr_log = 0, [], None

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.n += 1
        if self.lr_log is not None:
            self.lr_log.append([g["lr"] for g in self.param_groups])
        log = self.n <= LOG_UPD and self.gate
        before = [p.detach().clone() for p in self.gate] if log else None
        for o in self.opts:
            o.step()
        if log:
            self.upd.append(float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(self.gate, before)))))


@contextlib.contextmanager
def muon_in_onset_run(holder):
    """onset_run builds torch.optim.Adam(param_groups(model), lr=lr); for the duration of one run that
    name builds a MuonAdam on the same groups instead."""
    def factory(params, lr=1e-3, **kw):
        assert not kw, kw
        opt = MuonAdam(list(params), lr)
        if holder.get("log_lr"):
            opt.lr_log = []
        holder.setdefault("opts", []).append(opt)
        return opt
    torch.optim.Adam = factory
    try:
        yield
    finally:
        torch.optim.Adam = _ADAM


def is_adam(n, p):
    """S33's rule: the embedding, the convolution, biases and 1-D parameters go to Adam."""
    return n == "embed.weight" or "conv" in n or p.ndim == 1 or n.endswith("bias")


def make_groups(holder, mlr, slow):
    """onset_run's param_groups: gate (W_in, W_h, W_g; Muon at mlr throughout), the other 2-D / 3-D weights
    (Muon), the Adam parameters (embedding, conv; Adam at LR). slow: every non-gate group at SLOW_SCALE x its
    lr until the switch (lr_after)."""
    def param_groups(model):
        named = list(model.named_parameters())
        nm = dict(named)
        gate = [n for n, _ in named if n in GATE]
        adam = [n for n, p in named if n not in GATE and is_adam(n, p)]
        rest = [n for n, p in named if n not in GATE and n not in adam]
        sc = SLOW_SCALE if slow else 1.0
        spec = [("gate", gate, "muon", mlr, mlr), ("rest", rest, "muon", sc * mlr, mlr), ("rest", adam, "adam", sc * LR, LR)]
        holder["groups"] = [dict(params=[nm[n] for n in ns], lr=l0, kind=k, tag=t, lr_after=l1, names=ns)
                            for t, ns, k, l0, l1 in spec if ns]
        return holder["groups"]
    return param_groups


def pos_penalty_s12(gr, S, P):
    """explore_slow_pos.pos_penalty at d0c31ce (S13's hinge statistic): eta^2 by triple index and by half of
    gr[..., 0] at key positions 3j+1, differentiable. CHECK 123 only."""
    n = S * P
    j = torch.arange(n)
    p = gr[:, 3 * j + 1, 0]
    B = p.shape[0]
    mu = p.mean()
    tot = ((p - mu) ** 2).sum() + tss.EPS
    between_idx = B * ((p.mean(0) - mu) ** 2).sum()
    h = j >= P
    between_half = B * (int((~h).sum()) * (p[:, ~h].mean() - mu) ** 2 + int(h.sum()) * (p[:, h].mean() - mu) ** 2)
    return between_idx / tot, between_half / tot


# ── Recipes and runs ─────────────────────────────────────────────────────────
def make_rc(opt, slow, tau, window=WINDOW, warm=WARM, keep=None, stat=None, mlr=MUON_LR):
    """(recipe, hinge state, holder): ADAM = test_slow_start.make_recipe (slow, tau, window, diag); MUON =
    make_recipe without its Adam groups (tau, window, diag, extra statistics) plus S25's groups and switch
    (holder: the groups and the MuonAdam built; the run must be inside muon_in_onset_run(holder))."""
    hk = dict(window=window if tau is not None else None, diag=tau is not None, stat=stat)
    if opt == "adam":
        rc, state = tss.make_recipe(slow, tau, warm=warm, keep=keep, **hk)
        return rc, state, None
    holder = {}
    base, state = tss.make_recipe(False, tau, keep=keep, **hk)

    def rc(kw):
        kw = base(kw)
        kw["run_kw"] = dict(kw.get("run_kw") or {}, param_groups=make_groups(holder, mlr, slow))
        if slow:
            c0 = kw.get("check")

            def check(m, step, data):
                if c0 is not None:
                    c0(m, step, data)
                if step >= warm:
                    for g in holder["groups"]:
                        g["lr"] = g["lr_after"]
            kw["check"] = check
        return kw
    return rc, state, holder


@contextlib.contextmanager
def running(state, holder):
    """One run's context: MuonAdam in onset_run (holder not None) and the firing diagnostics' optimizer step
    pre-hook (S34's: before the first optimizer step of an update, a pending firing's hinge gradient on the
    gate is set against the accumulated gradient; task = accumulated - hinge)."""
    def pre(opt, args, kwargs):
        p = state.get("pending")
        if p is None:
            return
        state["pending"] = None
        hn = math.sqrt(sum(float((h ** 2).sum()) for h in p["hinge"]))
        tn = math.sqrt(sum(float(((q.grad - h) ** 2).sum()) for q, h in zip(state["gate"], p["hinge"])))
        state["firing_diag"].append(dict(update=p["update"], hinge=hn, task=tn, ratio=hn / tn if tn > 0 else float("inf")))
    h = register_optimizer_step_pre_hook(pre)
    try:
        with (muon_in_onset_run(holder) if holder is not None else contextlib.nullcontext()):
            yield
    finally:
        h.remove()


def run_path(a, seed, iters, rc):
    """The arm's run path: Part A's run_one call (test_short_conv.run_job's, through test_slow_start.part_run
    "0"; the perfect gate on the same call), Part B's test_stream_recipe.run_attempt (part_run "B"; the
    perfect gate ceiling4k16 on the same path)."""
    if a["ceil"]:
        if a["part"] == "A":
            kw = dict(task=tsc.TASK, stats_fn=conv_stats, grad_fn=conv_grad_norms)
            return trr.run_one(tsc._BASE["ceiling"], seed, iters, **(kw if rc is None else rc(kw)))
        return tsr.run_attempt(tsr.ARM["ceiling4k16"], seed, dict(tsr.REAL, total=iters), recipe=rc)
    return tss.part_run(TPART[a["part"]], seed, iters, rc)


def iters_of(a, sched):
    return sched["iters_a"] if a["part"] == "A" else sched["iters_b"]


def one_run(a, seed, iters, sched, tau=None, keep=None, stat=None, log_lr=False):
    """A run of arm a (tau None = the arm's own; returns the record with the optimizer's fields)."""
    t = a["tau"] if tau is None else tau
    rc, state, holder = make_rc(a["opt"], a["slow"], t, window=sched["window"], warm=sched["warm"], keep=keep, stat=stat)
    if holder is not None and log_lr:
        holder["log_lr"] = True
    with running(state, holder):
        rec = run_path(a, seed, iters, rc)
    if rec.get("ok"):
        rec["opt"] = a["opt"]
        rec["host"] = host_info()
        if holder is not None:
            opts = holder.get("opts", [])
            u = opts[0].upd if opts else []
            rec.update(muon_lr=MUON_LR, n_opts_built=len(opts),
                       opt_groups=[dict(tag=g["tag"], kind=g["kind"], names=g["names"], lr_end=g["lr"]) for g in holder["groups"]],
                       gate_upd=dict(n=len(u), median=statistics.median(u), max=max(u)) if u else None)
    return norm(rec), state, holder


def run_job(sp):
    a = ARM[sp["arm"]]
    return one_run(a, sp["seed"], iters_of(a, sp["sched"]), sp["sched"])[0]


def spec(key, seed, sched):
    return dict(arm=key, seed=seed, sched=dict(sched))


def check_job(key, seed, iters, tau=None, keep_at=None, stat=None, window=WINDOW, warm=WARM):
    """A CHECK run of arm `key` (record, hinge state summary, weights at keep_at)."""
    keep = None if keep_at is None else {"at": keep_at}
    rec, state, _ = one_run(ARM[key], seed, iters, dict(REAL, window=window, warm=warm), tau=tau, keep=keep, stat=stat)
    return dict(rec=rec, snap=None if keep is None else keep.get("snap"),
                fired=state.get("fired"), n=state.get("n"), fired_at=list(state.get("fired_at", [])),
                would_at=list(state.get("would_at", [])), w_log=dict(state.get("w_log", {})),
                diag=list(state.get("firing_diag", [])))


def adam_window_job(seed, iters, window):
    """CHECK 121: ADAM + WINDOW (diagnostics on) on Part A's path."""
    a = dict(ARM["WIN16_A"], part="A", key="ADAM_WIN_A")
    rec, state, _ = one_run(a, seed, iters, dict(REAL, window=window))
    return dict(rec=rec, fired_at=list(state.get("fired_at", [])), diag=list(state.get("firing_diag", [])))


def legacy_job(part, which):
    """CHECK 120: test_slow_start.variant_job(part, 3, 1200, slow, TAU 0) through c69f1e6's module or this one."""
    mod = load_at(LEGACY_SHA, "test_slow_start.py", "test_slow_start_legacy") if which == "legacy" else tss
    return strip_all(mod.variant_job(part, 3, EVAL_EVERY, True, 0.0)["rec"])


def lr_job(key):
    """CHECK 124: through the arm's real path and schedule, each group's lr at every update (MuonAdam's lr log;
    under Adam an optimizer step pre-hook) and the groups' parameters; the forward stubbed and every held-out
    accuracy 0.5 (no early stop), 2401 updates."""
    a = ARM[key]
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if isinstance(opt, _ADAM) and a["opt"] == "adam":
            if not ids:
                ids["g"] = [[id(p) for p in g["params"]] for g in opt.param_groups]
            lrs.append([g["lr"] for g in opt.param_groups])

    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    keep = {"at": 1}
    try:
        rc, state, holder = make_rc(a["opt"], a["slow"], a["tau"], keep=keep)
        if holder is not None:
            holder["log_lr"] = True
        with running(state, holder):
            rec = run_path(a, 3, WARM + 1, rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    names = {id(p): n for n, p in m.named_parameters()}
    if holder is not None:
        opt = holder["opts"][0]
        lg = opt.lr_log
        gs = [dict(tag=g["tag"], kind=g["kind"], names=g["names"]) for g in holder["groups"]]
        ids_all = [id(p) for g in holder["groups"] for p in g["params"]]
    else:
        lg = lrs
        gs = [dict(tag=t, kind="adam", names=[names[i] for i in g]) for t, g in zip(("gate", "rest"), ids["g"])]
        ids_all = [i for g in ids["g"] for i in g]
    cover = len(ids_all) == len(set(ids_all)) and set(ids_all) == {id(p) for p in m.parameters()}
    return dict(ok=rec["ok"], n=len(lg), at={str(u): lg[u - 1] for u in (1, WARM, WARM + 1) if u <= len(lg)}, groups=gs,
                cover=cover, all_names=[n for n, _ in m.named_parameters()])


def time_job(key):
    """Pool-load timing: the arm's real path for TIME_ITERS steps (one evaluation and its statistics
    included); seconds per step."""
    a = ARM[key]
    t0 = time.time()
    one_run(a, 0, TIME_ITERS, REAL)
    return (time.time() - t0) / TIME_ITERS


def full_cost(key, cost, sched):
    return cost[key] * iters_of(ARM[key], sched)


def projection(cost, sched, workers):
    full = {k: full_cost(k, cost, sched) for k in KEYS}
    d = [full[k] for k in KEYS for _ in SEEDS[k]]
    return full, d, makespan(d, workers) / 3600


# ── The host's bf16 path (Muon is host-dependent) ───────────────────────────
_HOST = {}


def host_info():
    """This process's CPU label, oneDNN's bf16 matmul kernel (ONEDNN_VERBOSE in a subprocess) and a fingerprint
    of torch's Newton-Schulz (bfloat16) on fixed Gaussian probes; computed once per process."""
    if not _HOST:
        g = torch.Generator().manual_seed(0)
        outs = [_zeropower_via_newtonschulz(torch.randn(s_, generator=g), MUON_KW["ns_coefficients"], MUON_KW["ns_steps"],
                                            MUON_KW["eps"]).float() for s_ in ((32, 256), (16, 32), (256, 32))]
        fp = hashlib.sha1(b"".join(bytes(t.contiguous().view(-1).view(torch.uint8).tolist()) for t in outs)).hexdigest()[:12]
        code = "import torch; torch.set_num_threads(1); a = torch.randn(32, 256).bfloat16(); b = torch.randn(256, 32).bfloat16(); a @ b"
        try:
            r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, ONEDNN_VERBOSE="1", OMP_NUM_THREADS="1"),
                               capture_output=True, text=True, timeout=120)
            impl = [ln.split(",")[6] for ln in r.stdout.splitlines() if ln.startswith("onednn_verbose") and ",exec,cpu,matmul," in ln]
            impl = impl[0] if impl else "?"
        except Exception as e:
            impl = f"? ({type(e).__name__})"
        _HOST.update(cpu=cpu_model(), onednn_bf16=impl, ns_fp=fp)
    return dict(_HOST)


SCREEN_CODE = {
    "S25": ("import explore_muon_recipe as s\n"
            "rec, _ = s.run_path(dict(s.ARMS['MUON_HINGE'], muon_lr={mlr}), {seed}, {iters}, model='hinge')\n"),
    "S35": ("import explore_muon_k16 as s\n"
            "rec = s.child('run', dict(kind='HINGE', seed={seed}, iters={iters}, mlr={mlr}))\n"),
}


def screen_code_job(tree, which, seed, iters):
    """CHECKs 122-123: the screen's own code at SCREEN_SHA, in its own process in the worktree `tree` (its own
    modules), one thread; returns its record."""
    code = ("import sys, os, json\nsys.path.insert(0, os.getcwd())\nimport torch\ntorch.set_num_threads(1)\n"
            + SCREEN_CODE[which].format(mlr=MUON_LR, seed=seed, iters=iters)
            + "print('@@REC@@' + json.dumps(rec, default=float))\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=tree, env=env, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("@@REC@@")]
    if not lines:
        raise RuntimeError(f"{which}'s code failed (exit {r.returncode}): {r.stderr[-1500:]}")
    return json.loads(lines[-1][len("@@REC@@"):])


# ── Verification ─────────────────────────────────────────────────────────────
def screen_runs(fname):
    src = subprocess.run(["git", "-C", HERE, "show", f"{SCREEN_SHA}:explore_out/{fname}"], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(src)["runs"]


def earlier_seeds():
    """test_slow_start's earlier seeds (0-279 with every restart reseed) and its own 280-299 (test_recipe_scope
    reused the recorded seeds)."""
    return tss.earlier_seeds() | set(range(280, 300))


def verify(pool, files):
    print("test_recipe_scope.py's verification (which runs test_slow_start's, and so on down to")
    print("test_multilayer_binding's), with X's recorded files:")
    trs.verify(pool, files)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f = {}
    tree = tempfile.mkdtemp(prefix="early_recipe_screens_")
    wt = subprocess.run(["git", "-C", HERE, "worktree", "add", "--detach", tree, SCREEN_SHA], capture_output=True, text=True)
    if wt.returncode == 0:
        f["122", "code"] = pool.submit(screen_code_job, tree, "S35", 240, EVAL_EVERY)
        f["123", "code"] = pool.submit(screen_code_job, tree, "S25", 160, WARM)
    f["122"] = pool.submit(check_job, "WIN16_M", 240, EVAL_EVERY)
    for k in ("WIN16_M", "WIN16_A", "WIN_M"):
        f["124w", k] = pool.submit(check_job, k, 3, WARM + 2, 0.0)
    f["123"] = pool.submit(check_job, "WIN_M", 160, WARM, None, None, pos_penalty_s12)
    f["121"] = pool.submit(adam_window_job, 280, WARM, MAX_A)
    f["127", "B"] = pool.submit(check_job, "WIN16_M", 1, EVAL_EVERY)
    f["127", "A"] = pool.submit(check_job, "WIN_M", 1, EVAL_EVERY, 0.0)
    for part in ("B", "0"):
        for which in ("legacy", "new"):
            f["120", part, which] = pool.submit(legacy_job, part, which)
    for k in KEYS:
        f["124", k] = pool.submit(lr_job, k)

    print(f"CHECK 120 test_slow_start's new knobs (window, diag, stat) are inert at their defaults: its runs (seed 3, "
          f"{EVAL_EVERY} steps, SLOW + the hinge at TAU = 0, firing on every batch) vs {LEGACY_SHA}'s module:")
    good = True
    for part in ("0", "B"):
        L_, n_ = f["120", part, "legacy"].result(), f["120", part, "new"].result()
        g = L_ == n_ and n_["ok"] and "hs_fired_at" not in n_["end"]
        good &= g
        print(f"     Part {part} ({tss.PART[part]['src']}): legacy == default {L_ == n_}   no new record fields "
              f"{'hs_fired_at' not in n_['end']}   hinge {n_['end'].get('hs_hinge')}   curve "
              f"{[[c[0], round(c[1], 4)] for c in n_['curve']]}  -> {'UNCHANGED' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 121 ADAM + WINDOW with the window end at {MAX_A} (firing diagnostics on), Part A's configuration, seed 280, "
          f"vs the recorded HINGE0 seed 280 ({files['slow_path']}) through {WARM}:")
    r = f["121"].result()
    rec_old = get(files["slow"], "HINGE0", 280)
    rep = rec_old is not None and subset_equal(trunc(r["rec"], WARM), trunc(rec_old, WARM))
    g = rep and r["rec"]["ok"] and len(r["diag"]) == len(r["fired_at"])
    m_ = files["slow"].get("meta", {})
    print(f"     curve {r['rec']['curve']} vs recorded {[c for c in rec_old['curve'] if c[0] <= WARM] if rec_old else None}; "
          f"statistics and gradient norms equal: {rep}")
    print(f"     firings {r['fired_at']} (each with a diagnostic row: {len(r['diag']) == len(r['fired_at'])}; hinge/task ratio "
          f"{[round(d['ratio'], 1) for d in r['diag']]})   the file was written on {m_.get('cpu')} (git {m_.get('git')}), "
          f"this machine {cpu_model()}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    pairs = dict(slow=bool(rep))
    ok &= g
    print()

    def code_rec(key):
        if wt.returncode != 0:
            print(f"     the screens' code is not available here: git worktree add {SCREEN_SHA} failed ({wt.stderr.strip()[-200:]})")
            return None
        try:
            return f[key, "code"].result()
        except Exception as e:
            print(f"     the screen's code failed here: {type(e).__name__}: {str(e)[-600:]}")
            return None

    print(f"CHECK 122 WIN16_M seed 240 vs S35's own code (explore_muon_k16 at {SCREEN_SHA}, its child on main at 9c5939e) on "
          f"this machine, through {EVAL_EVERY}:")
    r = f["122"].result()
    sc = code_rec("122")
    same = sc is not None and subset_equal(trunc(r["rec"], EVAL_EVERY), trunc(norm(sc), EVAL_EVERY))
    fa_sc = [u for u in (sc or {}).get("fired_at") or [] if u <= EVAL_EVERY]
    g = same and r["fired_at"] == fa_sc
    print(f"     curve {r['rec']['curve']} vs {sc['curve'] if sc else None}; statistics, gradient norms and the early curve "
          f"equal: {same}; firings {r['fired_at']} vs {fa_sc}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    try:
        old = screen_runs("muon_k16_results.json")["MUON_HINGE16|240"]
    except Exception as e:
        old = None
        print(f"     the screen's record is not readable here ({type(e).__name__})")
    if old is not None:
        same_o = subset_equal(trunc(r["rec"], EVAL_EVERY), trunc(old, EVAL_EVERY))
        print(f"     (not asserted) the screen's recorded run (muon_k16_results.json, written on {old.get('host') or 'the 2.10GHz host'}): "
              f"curve {[c for c in old['curve'] if c[0] <= EVAL_EVERY]}; firings {[u for u in old.get('fired_at') or [] if u <= EVAL_EVERY]}; "
              f"equal: {same_o and r['fired_at'] == [u for u in old.get('fired_at') or [] if u <= EVAL_EVERY]}")
    print()

    print(f"CHECK 123 MUON at Part A's configuration with S13's hinge statistic (channel 0) vs S25's own code "
          f"(explore_muon_recipe.run_path at {SCREEN_SHA}, MUON_HINGE) on this machine, through {WARM}:")
    r = f["123"].result()
    sc = code_rec("123")
    hc_new = [x.get("hs_hinge") for x in r["rec"]["stats"] if x["step"] in (EVAL_EVERY, WARM)]
    hc_sc = [x.get("hinge_counts") for x in sc["stats"] if x["step"] in (EVAL_EVERY, WARM)] if sc else None
    g = sc is not None and r["rec"]["curve"] == sc["curve"] and hc_new == hc_sc
    print(f"     curve {r['rec']['curve']} vs {sc['curve'] if sc else None}; the hinge's firings/batches at {EVAL_EVERY} and "
          f"{WARM} {hc_new} vs {hc_sc}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    try:
        old = screen_runs("muon_recipe_results.json")["MUON_HINGE|160"]
    except Exception as e:
        old = None
        print(f"     the screen's record is not readable here ({type(e).__name__})")
    if old is not None:
        oc = [c for c in old["curve"] if c[0] <= WARM]
        hc_old = [x.get("hinge_counts") for x in old["stats"] if x["step"] in (EVAL_EVERY, WARM)]
        print(f"     (not asserted) the screen's recorded run (muon_recipe_results.json, the 2.10GHz host): curve {oc}; firings/"
              f"batches {hc_old}; equal: {r['rec']['curve'] == oc and hc_new == hc_old}")
    subprocess.run(["git", "-C", HERE, "worktree", "remove", "--force", tree], capture_output=True)
    m = tss.gate_model("0", 5)
    m.eval()
    task = tsc.TASK
    with torch.no_grad():
        gr = m(probe_batch(task, 5)[0], TAU_END)[2]
    a_ = [float(x) for x in tss.eta2_hinge(gr, task.S, task.P)]
    b_ = [float(x) for x in pos_penalty_s12(gr, task.S, task.P)]
    print(f"     the two statistics on a probe batch (seed 5; arm A's gate redrawn far from uniform): test_slow_start's "
          f"(pooled) index {a_[0]:.9f}, half {a_[1]:.9f}; S13's (channel 0) {b_[0]:.9f}, {b_[1]:.9f}; difference "
          f"{a_[0] - b_[0]:.2e}, {a_[1] - b_[1]:.2e} (the test uses test_slow_start's)")
    print()

    print(f"CHECK 124 the optimizer groups, the lrs at updates 1, {WARM} and {WARM + 1}, and the hinge's weight:")
    good = True
    for k in KEYS:
        a = ARM[k]
        r = f["124", k].result()
        if a["opt"] == "muon":
            sc = SLOW_SCALE if a["slow"] else 1.0
            kinds = [(g["tag"], g["kind"]) for g in r["groups"]]
            w1 = [MUON_LR] * (1 if ("gate", "muon") in kinds else 0) + [sc * MUON_LR, sc * LR]
            w2 = [MUON_LR] * (1 if ("gate", "muon") in kinds else 0) + [MUON_LR, LR]
            adam_names = [n for g in r["groups"] if g["kind"] == "adam" for n in g["names"]]
            g = (r["cover"] and r["at"]["1"] == w1 and r["at"][str(WARM)] == w1 and r["at"][str(WARM + 1)] == w2
                 and "embed.weight" in adam_names and (a["part"] == "A" or "short_conv.conv_w" in adam_names)
                 and r["n"] == WARM + 1 and r["ok"])
        else:
            g = (r["cover"] and r["at"]["1"] == [LR, LR_WARM] and r["at"][str(WARM)] == [LR, LR_WARM]
                 and r["at"][str(WARM + 1)] == [LR, LR] and r["groups"][0]["names"] == list(GATE) and r["n"] == WARM + 1
                 and r["ok"])
        good &= g
        print(f"     {k:<8} groups {[(g_['tag'], g_['kind'], g_['names']) for g_ in r['groups']]}: every parameter exactly once "
              f"{r['cover']}; lrs at update 1 {r['at'].get('1')}, {WARM} {r['at'].get(str(WARM))}, {WARM + 1} "
              f"{r['at'].get(str(WARM + 1))}  -> {'OK' if g else 'WRONG'}")
    for k in ("WIN_M", "WIN16_A", "WIN16_M"):
        r = f["124w", k].result()
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

    print("CHECK 125 SliceMuon equals torch.optim.Muon (weight decay 0) bit for bit on 2-D weights (3 shapes, 5 steps):")
    gen = torch.Generator().manual_seed(1)
    w1 = [torch.randn(s, generator=gen).requires_grad_() for s in ((32, 256), (2, 32), (256, 32))]
    w2 = [w.detach().clone().requires_grad_() for w in w1]
    o1, o2 = SliceMuon(w1, lr=0.01, **MUON_KW), torch.optim.Muon(w2, lr=0.01, **MUON_KW)
    for _ in range(5):
        gr_ = [torch.randn(w.shape, generator=gen) for w in w1]
        for w, x in zip(w1, gr_):
            w.grad = x.clone()
        for w, x in zip(w2, gr_):
            w.grad = x.clone()
        o1.step()
        o2.step()
    g = all(torch.equal(a_, b_) for a_, b_ in zip(w1, w2))
    print(f"     equal: {g}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()

    print("CHECK 126 the seeds:")
    earlier = earlier_seeds()
    new = {s for k in KEYS for s in SEEDS[k]}
    g = not (new & earlier)
    print(f"     this test's seeds {tsr.ranges(sorted(new))}: disjoint from every earlier main-line seed: {g}")
    print(f"     (earlier seeds, restart reseeds included: {tsr.ranges(sorted(earlier))})")
    ok &= g
    print()

    print("CHECK 127 a worker's run is bit-identical to the same run here:")
    good = True
    for part, key, it, tau, what in (("B", "WIN16_M", EVAL_EVERY, None, "WIN16_M seed 1"),
                                     ("A", "WIN_M", EVAL_EVERY, 0.0, "WIN_M at TAU = 0 seed 1")):
        rw = f["127", part].result()
        rh = check_job(key, 1, it, tau)
        same = (subset_equal(strip_all(rw["rec"]), strip_all(rh["rec"])) and subset_equal(strip_all(rh["rec"]), strip_all(rw["rec"]))
                and rw["diag"] == rh["diag"] and rw["fired_at"] == rh["fired_at"])
        g = same and rh["rec"]["ok"] and (tau != 0.0 or len(rh["diag"]) == it)
        good &= g
        print(f"         {what}, {it} steps: records, firings and diagnostic rows equal {same}   firings {len(rh['fired_at'])}"
              f"   acc {rh['rec']['curve'][-1][1] if rh['rec']['curve'] else None}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print("CHECK 128 the pairing (a recorded file pairs if a recorded run reproduces bit for bit, whatever the CPU label):")
    m_ = files["slow"].get("meta", {})
    print(f"     {files['slow_path']} (written on {m_.get('cpu')}, git {m_.get('git')}; this machine {cpu_model()}): "
          f"{'PAIRS' if pairs['slow'] else 'DOES NOT PAIR'} (HINGE0 seed 280 through {WARM}, CHECK 121). No claim of this "
          f"test pairs with a recorded run; Parts A and B use fresh seeds.")
    print(f"     this machine's bf16 path (Muon): {host_info()}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(pairs=pairs, host=host_info())


# ── Reporting ────────────────────────────────────────────────────────────────
def by_update(r, u):
    return bool(ok_r(r) and r["transition"] is not None and r["transition"] <= u)


def firings(r):
    e = r.get("end") or {}
    return e.get("hs_fired_at"), e.get("hs_firing_diag"), e.get("hs_would_at")


def fire_str(r):
    fa, dg, wa = firings(r)
    if fa is None:
        return "--"
    rat = {d["update"]: d["ratio"] for d in dg or []}
    s = " ".join(f"{u}({rat[u]:.0f}x)" if u in rat else str(u) for u in fa[:8]) + (f" ...+{len(fa) - 8}" if len(fa) > 8 else "")
    return (s or "none") + (f"   would-fire after {WINDOW}: {len(wa)}" if wa else "")


def paired_by(new, old, seeds, fn):
    both = [s for s in seeds if ok_r(new.get(s)) and ok_r(old.get(s))]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    return dict(n=len(both), new=sum(bool(fn(new[s])) for s in both), old=sum(bool(fn(old[s])) for s in both), b=b, c=c,
                p=mcnemar_greater(b, c))


def report(store, info, valid, drop, wall, path, also):
    print()
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
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
        rs = runs_of(store, k, SEEDS[k])
        print(f"  {a['label']}   (outcome {'perfect gate: BOUND' if a['ceil'] else outcome_key(part)})")
        raw_rows(k, part, rs, SEEDS[k])
        if a["tau"] is not None:
            print(f"  {k} firings (update(ratio)):")
            for s in SEEDS[k]:
                if ok_r(rs.get(s)):
                    print(f"    s{s}  {fire_str(rs[s])}")
            print()

    print("=" * 100)
    print("VALIDITY (the perfect gate under MUON at 0.005, no slow phase, binds on 2 seeds)")
    print("=" * 100)
    for part, k in (("A", "CEIL_A"), ("B", "CEIL_B")):
        rs = runs_of(store, k, SEEDS[k])
        print(f"  Part {part}: {k} bound {valid[part][0]}/{len(SEEDS[k])} "
              f"({', '.join(f's{s} ' + (('BOUND at ' + str(r['transition'])) if bound_r(r) else 'not bound') for s, r in sorted(rs.items()))})"
              f"  -> {'VALID' if valid[part][1] else 'NOT VALID: the MUON claims of this part are UNTESTED'}")
    print()

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in KEYS:
        if ARM[k]["ceil"]:
            continue
        part = TPART[ARM[k]["part"]]
        rs = runs_of(store, k, SEEDS[k])
        done = [r for r in rs.values() if ok_r(r)]
        trs_ = [r["transition"] for r in done if r["transition"] is not None]
        hs = [len(firings(r)[0]) for r in done if firings(r)[0] is not None]
        print(f"  {ARM[k]['label']:<50} {outcome_key(part)} {sum(success(part, r) for r in done):>2}/{len(SEEDS[k])}   bound "
              f"{len(trs_):>2}   bound by {BY} {sum(by_update(r, BY) for r in done):>2}   collapsed "
              f"{sum(bool(r['collapsed']) for r in done):>2}   completed {len(done)}/{len(SEEDS[k])}   transition "
              f"{med_int(trs_) if trs_ else '--'}" + (f"   hinge never fired {sum(1 for x in hs if x == 0)}/{len(hs)}" if hs else ""))
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05)")
    print("#" * 100)
    res = {}
    rA = {k: runs_of(store, k, SEEDS[k]) for k in KEYS}
    d = paired_by(rA["WIN_M"], rA["SLOW_M"], SEEDS["WIN_M"], lambda r: success("0", r))
    res["E1"] = "UNTESTED" if not valid["A"][1] or d["n"] == 0 else ("SHOWN" if d["p"] < 0.05 else "NOT SHOWN")
    print(f"  E1  WIN_M beats SLOW_M (DISCOVERED): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; WIN_M only {d['b']}, SLOW_M only "
          f"{d['c']}; p = {d['p']:.4g}" + ("   (Part A not valid)" if not valid["A"][1] else ""))
    print(f"     *** E1: {res['E1']} ***")
    d = paired_by(rA["WIN16_M"], rA["WIN16_A"], SEEDS["WIN16_M"], lambda r: by_update(r, BY))
    res["E2"] = "UNTESTED" if not valid["B"][1] or d["n"] == 0 else ("SHOWN" if d["p"] < 0.05 else "NOT SHOWN")
    print(f"  E2  WIN16_M bound by update {BY} more often than WIN16_A: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; WIN16_M only "
          f"{d['b']}, WIN16_A only {d['c']}; p = {d['p']:.4g}" + ("   (Part B not valid)" if not valid["B"][1] else ""))
    print(f"     *** E2: {res['E2']} ***")
    for k in ("WIN_M", "WIN16_A", "WIN16_M"):
        part = TPART[ARM[k]["part"]]
        n = len(SEEDS[k])
        c = sum(success(part, r) for r in rA[k].values())
        untested = ARM[k]["opt"] == "muon" and not valid[ARM[k]["part"]][1]
        print(f"  band {k:<8} ({outcome_key(part)}, no restarts; RELIABLE >= {-(-9 * n // 10)}, MAJORITY >= {-(-n // 2)}, "
              f"MINORITY >= 1, NEVER 0, of {n}): {c}/{n}  -> {'UNTESTED' if untested else tscur.band_n(c, n)}")
    print()
    print("  READINGS:")
    lines = [(f"{k} SHOWN", READINGS[k]) for k in ("E1", "E2") if res[k] == "SHOWN"]
    for why, txt in lines:
        print(f"     {why}: \"{txt}\"")
    if not lines:
        print(f"     none of the readings applies (E1 {res['E1']}, E2 {res['E2']})")
    print()
    print("  Printed, not claims:")
    d = paired_by(rA["WIN16_M"], rA["WIN16_A"], SEEDS["WIN16_M"], bound_r)
    print(f"     WIN16_M vs WIN16_A, BOUND: {d['new']}/{d['n']} vs {d['old']}/{d['n']}; WIN16_M only {d['b']}, WIN16_A only {d['c']}; "
          f"one-sided p = {d['p']:.4g} (WIN16_M higher), {mcnemar_greater(d['c'], d['b']):.4g} (WIN16_A higher)")
    for k in ("SLOW_M", "WIN_M", "WIN16_A", "WIN16_M"):
        trs_ = [r["transition"] for r in rA[k].values() if ok_r(r) and r["transition"] is not None]
        print(f"     median transition {k:<8} {med_int(trs_) if trs_ else 'none bound'} ({len(trs_)} bound)")
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  failure classes and outcomes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome):")
    for k in ("SLOW_M", "WIN_M", "WIN16_A", "WIN16_M"):
        part = TPART[ARM[k]["part"]]
        cls = {}
        for s, r in sorted(rA[k].items()):
            cls.setdefault(tag(part, r), []).append(s)
        print(f"    {k:<8} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<32} {' '.join('s' + str(x) for x in v)}")
    print()
    print("  Part A per seed: SLOW_M | WIN_M (WIN_M's firings):")
    for s in SEEDS["WIN_M"]:
        a_, b_ = rA["SLOW_M"].get(s), rA["WIN_M"].get(s)
        print(f"    s{s}  {tag('0', a_):<24} | {tag('0', b_):<24} {fire_str(b_) if ok_r(b_) else ''}")
    print()
    print("  firings: per arm, runs with a firing; the first firing's update, median [min, max]; the hinge/task gradient-"
          "norm ratio over every firing, median [min, max]; would-fires after the window:")
    for k in ("WIN_M", "WIN16_A", "WIN16_M"):
        done = [r for r in rA[k].values() if ok_r(r) and firings(r)[0] is not None]
        fa = [firings(r)[0] for r in done]
        rat = [d_["ratio"] for r in done for d_ in (firings(r)[1] or [])]
        wa = [len(firings(r)[2] or []) for r in done]
        first = [x[0] for x in fa if x]
        print(f"    {k:<8} runs with a firing {sum(1 for x in fa if x)}/{len(done)}; first firing {med_int(first) if first else '--'}; "
              f"firings in all {sum(len(x) for x in fa)}; ratio "
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
    print("  hosts in the records (CPU label, oneDNN bf16 matmul kernel, Newton-Schulz fingerprint): runs per arm; Muon runs "
          "are bit-reproducible only on the same bf16 path:")
    for key_, d_ in hosts.items():
        print(f"    {key_}: {d_}")
    print()
    print("  Part B: merged runs (two or more streams on one channel at the end) and the map at 4800, 9600 and the end "
          "(distinct channels holding the 4 streams, streams per channel):")
    for k in ("WIN16_A", "WIN16_M"):
        mg = [s for s, r in sorted(rA[k].items()) if ok_r(r) and merged(r)]
        print(f"    {k}: merged at the end {len(mg)} ({' '.join('s' + str(x) for x in mg) or '-'})")
        for s in SEEDS[k]:
            r = rA[k].get(s)
            if not ok_r(r):
                continue
            print(f"      s{s}  4800: {groups_str(cmap_at(r, 4800)):<28} 9600: {groups_str(cmap_at(r, 9600)):<28} end: "
                  f"{groups_str(cmap_at(r, 'end')):<28} {tag('B', r)}  (transition {fmt_step(r['transition'])}; shared "
                  f"{shared_max(r)})")
    print()
    print("  Part B per seed: WIN16_A | WIN16_M (transition):")
    for s in SEEDS["WIN16_M"]:
        a_, b_ = rA["WIN16_A"].get(s), rA["WIN16_M"].get(s)
        ta = f"{tag('B', a_)} ({fmt_step(a_['transition']) if ok_r(a_) else '--'})"
        tb = f"{tag('B', b_)} ({fmt_step(b_['transition']) if ok_r(b_) else '--'})"
        print(f"    s{s}  {ta:<40} | {tb}")
    print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps)")
    print("=" * 100)
    for k in KEYS:
        part = TPART[ARM[k]["part"]]
        for s in SEEDS[k]:
            r = get(store, k, s)
            if ok_r(r):
                print(f"  {k:<8} s{s} {curve_str(r['curve'])}  -> {tag(part, r)}")
    print()
    if also:
        other = load_store(also)
        print("=" * 100)
        print(f"THE OTHER MACHINE (descriptive): {also} (CPU {other.get('meta', {}).get('cpu')}, git {other.get('meta', {}).get('git')})")
        print("=" * 100)
        for k in KEYS:
            part = TPART[ARM[k]["part"]]
            ro = [r for x, r in other["runs"].items() if x.split("|")[0] == k and ok_r(r)]
            if ro:
                print(f"  {k:<8} {outcome_key(part)} {sum(success(part, r) for r in ro)}/{len(ro)}   bound by {BY} "
                      f"{sum(by_update(r, BY) for r in ro)}   here {sum(success(part, r) for r in rA[k].values())}/{len(SEEDS[k])}")
        print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


def validity(store):
    out = {}
    for part, k in (("A", "CEIL_A"), ("B", "CEIL_B")):
        n = sum(bound_r(get(store, k, s)) for s in SEEDS[k])
        out[part] = (n, n >= VALID_N)
    return out


def main():
    ap = argparse.ArgumentParser(description="The hinge only during the slow phase, under Muon and under Adam.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (HINGE0 seed 280)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (inherited)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (inherited)")
    ap.add_argument("--scale", default=tss.tsa.RESULTS_FILE, help="this machine's test_scale_axes results (inherited)")
    ap.add_argument("--short", default=tsc.RESULTS_FILE, help="this machine's test_short_conv results (inherited)")
    ap.add_argument("--also", default=None, help="the other machine's early_recipe_results.json, for counts (descriptive)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = {}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    sched = dict(SCHED)

    print("=" * 100)
    print("Early recipe: the hinge on updates 1-2400 only, under Muon (Parts A and B) and under Adam (Part B)")
    print(f"  MUON: S25's construction (explore_muon_recipe.py at {MUON_CODE_SHA}): Muon lr {MUON_LR:g} on the gate throughout, "
          f"the other Muon weights x{SLOW_SCALE:g} and the Adam groups (embedding, conv) at {LR_WARM:g} for updates 1-"
          f"{sched['warm']}, then {MUON_LR:g} and {LR:g}; ADAM: test_slow_start's SLOW")
    print(f"  WINDOW: test_slow_start's hinge with weight 1.0 on updates 1-{sched['window']}, 0 after")
    print(f"  Part A: S=2, P=4, k=2, no conv, {sched['iters_a']} steps (DISCOVERED); Part B: S=4, P=4, k=16, conv, "
          f"{sched['iters_b']} steps (BOUND; E2 bound by {BY})")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<62} seeds {s[0]}-{s[-1]} ({len(s)})")
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
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
                             muon_lr=MUON_LR, tau=TAU, sched=sched, pairs=info["pairs"], host=info["host"],
                             started=time.strftime("%Y-%m-%d %H:%M:%S"), note="seed-level outcomes differ between machines")
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
                for k in ("WIN16_A", "WIN16_M"):
                    SEEDS[k] = tuple(old_drop)
                drop = f"Part B cut to seeds {old_drop[0]}-{old_drop[-1]}"
            full, d, total_h = projection(cost, sched, args.workers)
            print(f"  the drop decision recorded at the first start of this results file is kept: {drop or 'nothing cut'} "
                  f"(projection now {total_h:.2f} h)")
        else:
            decision = []
            if total_h > DROP_H:
                for k in ("WIN16_A", "WIN16_M"):
                    SEEDS[k] = tuple(s for s in SEEDS[k] if s in CUT_B)
                decision = list(SEEDS["WIN16_A"])
                drop = f"Part B cut to seeds {decision[0]}-{decision[-1]}"
                full, d, total_h = projection(cost, sched, args.workers)
                print(f"  *** above {DROP_H:g} h: {drop} (drop rule); projection now {total_h:.2f} h ***")
                if total_h > DROP_H:
                    print(f"  (still above {DROP_H:g} h; the rule has no further step: Part A is never cut)")
            else:
                print(f"  within {DROP_H:g} h: nothing is cut")
            store["meta"]["drop_decision"] = decision
        if old_drop is not None:
            store["meta"]["drop_decision"] = old_drop
        store["meta"].update(projected_wall_h=total_h, drop=drop, seeds={k: list(v) for k, v in SEEDS.items()},
                             ms_per_step={k: cost[k] * 1000 for k in KEYS})
        save_results(args.results, store)
        print()

        def describe(key, rec):
            if not rec.get("ok"):
                return f"*** FAILED: {rec.get('error')}"
            part = TPART[ARM[key]["part"]]
            tr = rec["transition"]
            mid = (f"VALcos {rec['val_cos']:.3f}" if part == "0" else
                   f"map [{','.join(str(c) for c in rec['end']['ch_map'])}] streams {[round(v, 2) for v in rec['end']['stream_acc']]}")
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

    report(store, info, validity(store), drop, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
