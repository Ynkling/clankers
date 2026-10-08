#!/usr/bin/env python
"""
test_window_gate.py — batch 1's width-3 window gate (LOCAL3) with slow memory and no hinge, against the current
recipes at two, four and eight streams. Specification: specs/test_window_gate.md (committed before this file).
Everything below is fixed before any run. Both machines run it, each on its own seeds.

Run it directly (X first; on X the recorded files are results/X/...):

    python test_window_gate.py --machine X --workers 4 --early results/X/early_recipe_results.json \\
        --slow results/X/slow_start_results.json --recipe results/X/stream_recipe_results.json \\
        --curriculum results/X/stream_curriculum_results.json --scale results/X/scale_axes_results.json \\
        --short results/X/short_conv_results.json
    (the inherited CHECKs 122-123 need claude/outside-ideas fetched: git fetch origin claude/outside-ideas;
     CHECK 135 reads batch 1's and batch 16's records from that branch at 092f355)
    L: python test_window_gate.py --machine L --workers 6 --also results/X/window_gate_results.json
       (--early defaults to L's own early_recipe_results.json; --slow etc. to L's own files)
    The report alone, from a finished results file: python test_window_gate.py --machine L --report --also <X's file>

BACKGROUND (batch 16, EXPLORATORY, claude/outside-ideas; verdicts in explore_out/README.md at 092f355)
- S43 LOCAL3_SLOW (S=2, P=4, k=2, no conv; seeds 160-199): DISCOVERED 31/40 (STREAM-PARTIAL 5, KEY 4, no POSITION)
  vs S13's SLOW_HINGE 34/40 (5 vs 8, p = 0.58): inconclusive. LOCAL3 alone over 160-199: 33/40.
- S44(a) LOCAL3_SLOW16_A (S=4, P=4, k=16, conv; 240-259): BOUND ROUTED 20/20 vs X's HINGE4k16 15/20 (5 vs 0,
  p = 0.0625): "carries to four streams".
- S44(b) LOCAL3_SLOW_D8_A (S=8, P=4, k=16, conv; 260-269; 28800 steps): BOUND ROUTED 3/10 (transitions 16800-21600)
  vs X's HINGE_D8 0/10 (3 vs 0, p = 0.25): "breaks the eight-stream stall".

THE GATE. LOCAL3, copied from explore_b16_gates.py at 93bdfe6 (LocalBDH, window_init, to_local), which is batch 1's
explore_local_gate.LocalGateBDH line for line: u_t = sum_{j<3} w_j v_{t-j} (depthwise, causal, zero-padded; the window
drawn from a generator seeded seed + 31337, uniform +-1/sqrt(3), no bias), h_t = tanh(W_in u_t), g_t = softmax(W_g h_t),
read gate = write gate. The run path's model is built as before and converted in place: every existing parameter is
unchanged, the window is added, W_h stays in the model unused, frozen and in no optimizer group.

RECIPES (all Adam, lr 1e-3 = test_slow_start.LR; no Muon)
- WIN3: LOCAL3, onset_run's single Adam at 1e-3 throughout (batch 1's LOCAL3), no hinge.
- WIN3_SLOW, WIN3_SLOW16, WIN3_SLOW_D8: LOCAL3 + SLOW, no hinge (batch 16's LOCAL3_SLOW / LOCAL3_SLOW16_A /
  LOCAL3_SLOW_D8_A): the gate group (W_in, W_g, gate_conv.conv_w) at 1e-3 throughout; every other trainable parameter
  at 1e-4 for updates 1-2400 and 1e-3 after (the switch right after the evaluation at 2400). test_slow_start's extra
  statistics are recorded (make_recipe with no hinge and no SLOW); this test's groups and switch are composed on top.
  No existing module changes.
- HINGE0, HINGE_D8: test_slow_start's HINGE recipe (make_recipe(True, TAU)), recurrent gate, on part_run "0" and "C";
  HINGE_D8 at 43200 steps.
- WIN16_A: test_early_recipe's arm, unchanged (ADAM + WINDOW, firing diagnostics), through its one_run.

ARMS AND SEEDS (--machine X|L, printed with the CPU string)
- Part A: test_short_conv's arm A path, S=2, P=4, k=2, no conv, 24000 steps; X 500-539, L 1500-1539; DISCOVERED.
    WIN3, WIN3_SLOW, HINGE0
- Part B: test_stream_recipe.run_attempt with A4k16, S=4, P=4, k=16, conv, 28800; X 540-559, L 1540-1559; BOUND ROUTED.
    WIN3_SLOW16, WIN16_A
- Part C: test_stream_curriculum.run_sc with D8 (all 8 streams from step 1), S=8, P=4, k=16, conv, 43200; X 540-559,
  L 1540-1559; BOUND ROUTED, with the transition.
    WIN3_SLOW_D8, HINGE_D8
OUTCOMES: Part A DISCOVERED (bound and final VAL cos < 0.5); Parts B and C BOUND ROUTED = test_stream_recipe.outcome
== "BOUND ROUTED" (bound, one-to-one, every stream >= 0.9).
VALIDITY per part: the perfect gate, Adam 1e-3 single rate, on the part's path and the part's first two seeds of this
machine (CEIL_A test_short_conv's perfect gate on Part A's run_one call; CEIL_B test_stream_recipe's ceiling4k16;
CEIL_C test_stream_curriculum's SC8_ceil with the curriculum off, 43200 steps), VALID if it binds on both; otherwise
that part's claims are UNTESTED on that machine, and pooled if the part is not valid on either machine.

CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05)
Pooled (primary): computed when --also gives the other machine's complete file (its meta names the other machine;
every run its meta schedules is recorded and finished); b and c summed over both machines' pairs.
- G1: WIN3_SLOW16 beats WIN16_A (BOUND ROUTED).
- G2: WIN3_SLOW_D8 beats HINGE_D8 (BOUND ROUTED).
- G3: d = (HINGE0 only) - (WIN3_SLOW only) on DISCOVERED; HOLDS if d <= 2, FAILS if d > 2 (a bound, no p-value).
Per machine (secondary): G1, G2, G3 on that machine's pairs, same rules.
Bands, per machine and pooled, every arm: RELIABLE >= 90%, MAJORITY >= 50%, MINORITY >= 1 run, NEVER 0
(test_stream_curriculum.band_n), with the Wilson 95% interval (test_stream_recipe.wilson).
Printed, not claims: WIN3_SLOW vs WIN3 (McNemar, both directions); median transitions for every arm.

DIAGNOSTICS (not part of the verdict): failure classes; merged runs; the stream -> channel map with distinct channels at
4800, 9600, 19200 and the end (Parts B and C); WIN16_A's firings and hinge/task ratios; HINGE0's and HINGE_D8's hinge
counts; X and L side by side with --also.

CHECKS (after the inherited chain: test_muon_recipe's verify, through CHECK 134)
- 135 WIN3_SLOW at seed 160 equals batch 16's LOCAL3_SLOW|160 (explore_out/window_recipe_results.json on
      claude/outside-ideas at 092f355; written on a Xeon @ 2.10GHz) bit for bit through 2400 (curve, gradient norms
      and every statistic both records hold); WIN3 at seed 160 likewise against batch 1's LOCAL3|160
      (explore_out/local_gate_results.json). Asserted on machine X ("on X's CPU"); on L printed, not asserted.
- 136 The gate (Parts A, B, C models): at t it ignores token t-3 and sees token t-2; the conversion keeps every
      parameter of the run path's model bitwise and adds only the window, equal to batch 1's for the seed; W_h frozen;
      with the window at [1, 0, 0] and W_h = 0 the gate equals the recurrent gate with W_h = 0.
- 137 Groups and lrs through the real paths (forward and evaluation stubbed, 2401 updates): every WIN3_SLOW arm has two
      groups covering every trainable parameter once, gate group (W_in, W_g, gate_conv.conv_w), lrs [1e-3, 1e-4] at
      updates 1 and 2400 and [1e-3, 1e-3] at 2401, W_h in no group and unchanged; WIN3 one group at 1e-3 at 1, 2400
      and 2401.
- 138 HINGE_D8 at seed 260 on Part C's path (run_sc uses the total only as run_one's step count) equals this
      machine's recorded test_slow_start HINGE_D8|260 (--slow) bit for bit through 2400.
- 139 Seeds: this machine's seeds are its ranges, the perfect gates on each part's first two seeds; disjoint from the
      other machine's ranges and from every earlier main-line seed (test_muon_recipe's 400-479 and 1400-1479
      included); with --also: the other file's meta names the other machine, its seeds lie in the other machine's
      ranges, its CPU label differs and no seed overlaps.
- 140 A worker's run is bit-identical to the same run here (WIN3_SLOW_D8 seed 1 and WIN3 seed 1, 1200 steps).

RUNTIME: projection with pool-load timing (each arm's real path for 1200 steps, as simultaneous copies on the workers,
the median per-step time; worst case = every run to its last step), printed. Nothing is cut: Parts B and C are never
cut, and no cut rule was given for Part A. Perfect gates first, then longest first; finished runs are cached.

OUTPUT: per-seed raw values first, then validity, counts, claims (pooled with --also, then per machine), bands,
diagnostics, curves. Results: window_gate_results.json (gitignored; each machine's copy in results/X or results/L).

AMENDMENT OF 8 OCTOBER 2026: PART D (specs/test_window_gate.md, "Amendment"; X's run had started at 2f98f06)
The user's answers before any Part D code: Part D runs 43200 steps (matching Part C, so G2 and G4 pair on equal
budgets); the original G2 is printed, not a claim.
- Part D (S=8, P=4, k=16, conv; Part C's path, run_sc with D8, 43200 steps; Part C's seeds; validity = Part C's CEIL_C):
    WIN3_SPLIT_D8  WIN3_SLOW_D8 + batch 17's W_SPLIT trigger (explore_window_d8_child.make_trigger at 992e320 = S37's
                   make_trigger with the cap as a parameter; copied here with S36's probe_data, split_op and
                   with_trigger and S37's read_gate, select and targeting): checks after the evaluations at 2400
                   (reference), 4800, 7200, ... through total - 2400 = 40800, after the path's check and the lr switch, on
                   the 64-sequence probe (generator 12345); fire if probe accuracy < 0.95 and it rose < 0.02 since the
                   previous check; at most 3 splits, none within 4800 updates of the previous one (capped or gapped
                   checks logged as blocked); c* / c0 = the largest / smallest mean read-gate mass at the probe's key
                   positions; W_g[c0] = w + n2, W_g[c*] = w + n1 (w = W_g[c*], noise 0.1 std(w), generator 12_000_000 +
                   1000 x seed + split count); W_g's Adam state zeroed. onset_run's Adam built by a capturing factory.
    WIN3_RESET_D8  the reset-only control: the same rule on its own run; where it fires, W_g's Adam state is zeroed, W_g
                   untouched. Part C's first five seeds (X 540-544, L 1540-1544). Printed, not claimed.
- Claims: G2 becomes WIN3_SPLIT_D8 beats HINGE_D8 (BOUND ROUTED); G4: WIN3_SPLIT_D8 beats WIN3_SLOW_D8 (BOUND ROUTED);
  G1 and G3 unchanged; pooled primary, per machine secondary. Printed, not claims: the original G2 (WIN3_SLOW_D8 vs
  HINGE_D8) and WIN3_RESET_D8 vs WIN3_SLOW_D8. Bands with Wilson for every arm. Reading (pooled): "eight streams bind
  without labels or restarts" if WIN3_SPLIT_D8 is RELIABLE pooled (>= 36/40) and G2 and G4 are SHOWN pooled.
- Diagnostics: every split's update, c* -> c0 and whether the target is labelled ok (c* holds >= 2 streams and c0 none
  on the labelled map before it); blocked checks and why; the labelled map at every check; the control's resets.
- CHECKs:
  - 141 WIN3_SPLIT_D8 at seed 263 equals batch 17's W_SPLIT|263 (explore_out/window_d8_results.json at 992e320; 2.10GHz;
        first split at 4800) bit for bit through 7200: curve, gradient norms, every statistic both records hold, and
        the trigger's rows at 2400, 4800 and 7200 (the fields both hold). Asserted on X, printed on L.
  - 142 The trigger's mechanics (a model after one Adam step; the trigger called at each check with no training in
        between; threshold above 1): fires at 4800, 9600, 14400; gap-blocked at 7200 and 12000; cap-blocked from 16800;
        each split leaves W_g's rows c* and c0 apart and zeroes W_g's Adam state; the reset-only variant at the same
        checks zeroes the state and leaves W_g unchanged.
  - 143 With the threshold at 0, WIN3_SPLIT_D8 and WIN3_RESET_D8 at seed 1 equal WIN3_SLOW_D8 bit for bit through 6000.
  - 144 (a resume) Part A-C records cached by an earlier start reproduce under this code: the first cached WIN3_SLOW
        and WIN3_SLOW_D8 records through 2400.
  - 139 extended: WIN3_SPLIT_D8 on Part C's range, WIN3_RESET_D8 on its first five seeds.
- Runtime: the split arm is never cut (nothing is cut). On X the run started at 2f98f06 finishes Parts A-C untouched;
  this code then resumes the same results file: the full verification, the projection, and only Part D's runs. The meta
  keeps the first start's commit and time and records the amendment's start (commit, time, arms). L runs it all at once.
"""

import argparse
import contextlib
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
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

from bdh import BDH
from test_multilayer_binding import MultiBDH, CausalConv, D
import test_binding_onset as tbo
from test_binding_onset import EVAL_EVERY, fmt_step
from test_router_curriculum import get, load_store, init_worker, makespan, cpu_model, save_results
from test_instrument_v2 import TAU_END
from test_multilayer_binding import probe_batch
from test_router_reliability import strip_all
import test_router_reliability as trr
from test_conv_lr import bound_r, mcnemar_greater
import test_short_conv as tsc
from test_short_conv import conv_stats, conv_grad_norms
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_stream_channels import med_int, merged, shared_max
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import LR, LR_WARM, WARM, TAU, ok_r, hstr, tag, runs_of, curve_str, raw_rows, subset_equal, trunc
import test_early_recipe as ter
from test_early_recipe import firings, fire_str, paired_by
import test_muon_recipe as tmr
from test_recipe_scope import cmap_at, groups_str

# ── Settings (fixed before any run) ──────────────────────────────────────────
MACHINES = ("X", "L")
RANGES = {"X": dict(A=range(500, 540), B=range(540, 560), C=range(540, 560)),
          "L": dict(A=range(1500, 1540), B=range(1540, 1560), C=range(1540, 1560))}
MAX_A, MAX_B, MAX_C = tsc.MAX_ITERS, tsr.MAX_ITERS, 43200      # 24000, 28800, 43200
VALID_N = 2
G3_BOUND = 2
WIDTH, GCONV_OFFSET = 3, 31_337            # batch 1's window (explore_local_gate.WIDTH, GCONV_OFFSET)
GATE_L = ("W_in", "W_g", "gate_conv.conv_w")
SCREEN_SHA = "092f355"                     # claude/outside-ideas: batch 1's and batch 16's records (CHECK 135)
REPRO_SEED = 160
D8_SEED = 260                              # CHECK 138
MAP_AT = (4800, 9600, 19200)
TIME_ITERS = 1200
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "window_gate_results.json"

REAL = dict(iters_a=MAX_A, iters_b=MAX_B, iters_c=MAX_C, warm=WARM, window=ter.WINDOW)
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ARMS = [
    dict(key="WIN3", part="A", kind="local", slow=False, ceil=False, label="WIN3          LOCAL3, Adam 1e-3 single rate (S=2, P=4, k=2)"),
    dict(key="WIN3_SLOW", part="A", kind="local", slow=True, ceil=False, label="WIN3_SLOW     LOCAL3 + SLOW, no hinge (S=2, P=4, k=2)"),
    dict(key="HINGE0", part="A", kind="hinge", slow=True, ceil=False, label="HINGE0        recurrent gate, SLOW + hinge (S=2, P=4, k=2)"),
    dict(key="WIN3_SLOW16", part="B", kind="local", slow=True, ceil=False, label="WIN3_SLOW16   LOCAL3 + SLOW, no hinge (S=4, P=4, k=16, conv)"),
    dict(key="WIN16_A", part="B", kind="early", slow=True, ceil=False, label="WIN16_A       ADAM + WINDOW, recurrent gate (S=4, P=4, k=16, conv)"),
    dict(key="WIN3_SLOW_D8", part="C", kind="local", slow=True, ceil=False, label="WIN3_SLOW_D8  LOCAL3 + SLOW, no hinge (S=8, P=4, k=16, conv)"),
    dict(key="HINGE_D8", part="C", kind="hinge", slow=True, ceil=False, label="HINGE_D8      recurrent gate, SLOW + hinge (S=8, P=4, k=16, conv)"),
    dict(key="WIN3_SPLIT_D8", part="C", kind="split", slow=True, ceil=False, amend=True,
         label="WIN3_SPLIT_D8 WIN3_SLOW_D8 + batch 17's split trigger (Part D)"),
    dict(key="WIN3_RESET_D8", part="C", kind="reset", slow=True, ceil=False, amend=True,
         label="WIN3_RESET_D8 WIN3_SLOW_D8 + the trigger, Adam reset only (Part D control)"),
    dict(key="CEIL_A", part="A", kind="ceil", slow=False, ceil=True, label="CEIL_A        perfect gate (validity, Part A)"),
    dict(key="CEIL_B", part="B", kind="ceil", slow=False, ceil=True, label="CEIL_B        perfect gate ceiling4k16 (validity, Part B)"),
    dict(key="CEIL_C", part="C", kind="ceil", slow=False, ceil=True, label="CEIL_C        perfect gate SC8_ceil, no curriculum (validity, Part C)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
ARMS_RUN = tuple(k for k in KEYS if not ARM[k]["ceil"])
CEILS = (("A", "CEIL_A"), ("B", "CEIL_B"), ("C", "CEIL_C"))
TPART = {"A": "0", "B": "B", "C": "C"}     # test_slow_start's part names (tag, raw_rows)
OUTCOME = {"A": "DISCOVERED", "B": "BOUND ROUTED", "C": "BOUND ROUTED"}
CLAIMS = (("G1", "WIN3_SLOW16", "WIN16_A", "B"), ("G2", "WIN3_SPLIT_D8", "HINGE_D8", "C"),
          ("G4", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", "C"))
G3 = ("G3", "WIN3_SLOW", "HINGE0", "A")
PRINTED = (("slow memory on the window gate", "WIN3_SLOW", "WIN3", "A"),
           ("the original G2", "WIN3_SLOW_D8", "HINGE_D8", "C"),
           ("the reset-only control", "WIN3_RESET_D8", "WIN3_SLOW_D8", "C"))
READING_D = "eight streams bind without labels or restarts"
RESET_N = 5                                # the reset-only control: Part C's first five seeds
# The trigger (batch 17's W_SPLIT: S37's with the cap as a parameter; S36's probe and operation)
PROBE_N, PROBE_SEED = 64, 12345
ACC_THR, RISE = 0.95, 0.02
EVERY, FIRST, REF_AT = 2400, 4800, 2400
NOISE_REL, NOISE_BASE = 0.1, 12_000_000
MAX_SPLITS, GAP = 3, 4800
SPLIT_SHA = "992e320"                      # claude/outside-ideas: batch 17's W_SPLIT records (CHECK 141)
SPLIT_SEED, SPLIT_THROUGH = 263, 7200
CEIL_C_ARM = dict(tscur.ARM["SC8_ceil"], key="CEIL_C", cur=False)


def seeds_for(machine):
    r = RANGES[machine]
    A, B, C = (tuple(r[p]) for p in "ABC")
    return dict(WIN3=A, WIN3_SLOW=A, HINGE0=A, WIN3_SLOW16=B, WIN16_A=B, WIN3_SLOW_D8=C, HINGE_D8=C,
                WIN3_SPLIT_D8=C, WIN3_RESET_D8=C[:RESET_N], CEIL_A=A[:VALID_N], CEIL_B=B[:VALID_N], CEIL_C=C[:VALID_N])


SEEDS = seeds_for("X")                     # main() sets the machine's


def success(key, r):
    """The part's outcome: DISCOVERED in Part A, BOUND ROUTED in Parts B and C."""
    if not ok_r(r):
        return False
    if ARM[key]["part"] == "A":
        return bool(r["discovered"])
    return tsr.outcome(r) == "BOUND ROUTED"


# ── LOCAL3: copied from explore_b16_gates.py at 93bdfe6 (batch 1's LocalGateBDH, line for line) ──────────
class LocalBDH(MultiBDH):
    """MultiBDH with LOCAL3's gate: explore_local_gate.LocalGateBDH's local_gates and forward, line for line."""

    def local_gates(self, v):
        u = self.gate_conv(v)                                         # (B, T, D), causal
        h = torch.tanh(u @ self.W_in.T)
        g = F.softmax(h @ self.W_g.T, dim=-1)
        return g, g

    def forward(self, tokens, tau=None, track_sat=False):
        gr, gw = self.local_gates(self.embed(tokens))
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        if self.conv is None:
            logits, _ = BDH.forward(self, tokens)
        else:
            logits = self.forward_conv(tokens)
        return logits, None, gr, gw


def window_init(seed):
    """Batch 1's window for a seed: uniform +-1/sqrt(WIDTH) from a generator seeded seed + GCONV_OFFSET."""
    g = torch.Generator().manual_seed(int(seed) + GCONV_OFFSET)
    return (torch.rand(D, WIDTH, generator=g) * 2 - 1) * (1.0 / math.sqrt(WIDTH))


def to_local(m, seed):
    assert type(m) is MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
    assert not m.gate_to_readout and not m.gate_ln and m.gate_noise == 0.0
    m.__class__ = LocalBDH
    m.local = True
    m.gate_conv = CausalConv(D, WIDTH)
    with torch.no_grad():
        m.gate_conv.conv_w.copy_(window_init(seed))
    m.W_h.requires_grad_(False)
    return m


# ── Recipes and runs ─────────────────────────────────────────────────────────
def local_groups(holder):
    """LOCAL3 + SLOW's groups over the trainable parameters (batch 16's): the gate group (W_in, W_g, the window) at LR,
    every other trainable parameter at LR_WARM; W_h (frozen) in no group."""
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gate = [n for n in GATE_L if n in nm]
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]
        return holder["groups"]
    return param_groups


def make_local_rc(slow, warm=WARM, keep=None):
    """(recipe, holder): the run path's model converted to LOCAL3; test_slow_start's extra statistics (make_recipe with
    no hinge and no SLOW); slow: this file's groups and the switch after the evaluation at `warm` (after the path's own
    check)."""
    base, _ = tss.make_recipe(False, None, keep=keep)
    holder = {}

    def rc(kw):
        kw = dict(kw)
        b0 = kw.get("builder") or trr.make_fn

        def builder(a, seed):
            mk = b0(a, seed)

            def make():
                return to_local(mk(), seed)
            return make
        kw["builder"] = builder
        kw = base(kw)
        if slow:
            kw["run_kw"] = dict(kw.get("run_kw") or {}, param_groups=local_groups(holder))
            c0 = kw.get("check")

            def check(m, step, data):
                if c0 is not None:
                    c0(m, step, data)
                if step >= warm:
                    holder["groups"][1]["lr"] = LR
            kw["check"] = check
        return kw
    return rc, holder


def run_path(a, seed, iters, rc):
    """The part's run path: Part A test_short_conv's arm A (part_run "0"; the perfect gate on the same run_one call),
    Part B run_attempt with A4k16 (part_run "B"; ceiling4k16), Part C run_sc with D8 (part_run "C"; SC8_ceil without
    the curriculum)."""
    part = a["part"]
    if a["ceil"]:
        if part == "A":
            kw = dict(task=tsc.TASK, stats_fn=conv_stats, grad_fn=conv_grad_norms)
            return trr.run_one(tsc._BASE["ceiling"], seed, iters, **(kw if rc is None else rc(kw)))
        if part == "B":
            return tsr.run_attempt(tsr.ARM["ceiling4k16"], seed, dict(tsr.REAL, total=iters), recipe=rc)
        return tscur.run_sc(CEIL_C_ARM, seed, dict(tscur.REAL, total=iters), recipe=rc)
    return tss.part_run(TPART[part], seed, iters, rc)


def iters_of(a, sched):
    return sched[{"A": "iters_a", "B": "iters_b", "C": "iters_c"}[a["part"]]]


# ── Part D: batch 17's W_SPLIT trigger (explore_window_d8_child.make_trigger at 992e320, S37's with the cap as a
#    parameter), with S36's probe_data, split_op and with_trigger and S37's read_gate, select and targeting ─────
@contextlib.contextmanager
def adam_capture(holder):
    """onset_run's torch.optim.Adam(param_groups(model), lr=lr), built by the same constructor and kept (S36's)."""
    def factory(params, lr=1e-3, **kw):
        opt = ter._ADAM(params, lr=lr, **kw)
        holder.setdefault("opts", []).append(opt)
        return opt
    torch.optim.Adam = factory
    try:
        yield
    finally:
        torch.optim.Adam = ter._ADAM


def probe_data(task):
    return task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]


@torch.no_grad()
def read_gate(m, data):
    was = m.training
    m.eval()
    try:
        return m(data[:, :-1], TAU_END)[2]
    finally:
        m.train(was)


def select(gr, S, P):
    """(KEYMASS c*, c0, key masses), (all-position c*, c0, masses) from a read gate gr (B, T, k)."""
    j = torch.arange(S * P)
    km = gr[:, 3 * j + 1].mean((0, 1))
    am = gr.mean((0, 1))
    return (int(km.argmax()), int(km.argmin()), km), (int(am.argmax()), int(am.argmin()), am)


def targeting(cmap, cs, c0):
    """Labelled: does (c*, c0) copy a channel holding >= 2 streams onto one holding none?"""
    return cmap.count(cs) >= 2 and cmap.count(c0) == 0


def noise_for(w, seed):
    sd = NOISE_REL * w.std()
    g = torch.Generator().manual_seed(seed)
    n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c*
    n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c0
    return sd, n1, n2


def zero_state(o, W):
    st = o.state.get(W)
    zeroed = []
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                v.zero_()
                zeroed.append(k)
    return zeroed


def split_op(m, o, cs, c0, seed):
    """S24's SPLIT on W_g, then W_g's optimizer state (in o) zeroed in place (S36's)."""
    assert not any(n.startswith("b_g") or n == "W_g_bias" for n, _ in m.named_parameters())
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = noise_for(w, seed)
        W[c0] = w + n2
        W[cs] = w + n1
    zeroed = zero_state(o, W)
    return dict(noise_sd=float(sd), noise_seed=seed, row_dist_before=before,
                row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=zeroed)


def reset_op(m, o):
    """The reset-only control: W_g's optimizer state zeroed in place; W_g untouched."""
    return dict(state_zeroed=zero_state(o, m.W_g))


def make_trigger(task, seed, info, opt_of, last, max_splits=MAX_SPLITS, gap=GAP, thr=ACC_THR, rise=RISE, op="split"):
    """Batch 17's make_trigger line for line; op="reset" applies reset_op where the rule fires (the control)."""
    data = probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        (kc, k0, km), (ac, a0, am) = select(read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=targeting(cmap, kc, k0), all_ok=targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= max_splits else "gap"
        elif eligible:
            res = (split_op(m, opt_of(), kc, k0, NOISE_BASE + 1000 * seed + st["n"]) if op == "split"
                   else reset_op(m, opt_of()))
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in select(read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def with_trigger(rc, trig):
    def rc2(kw):
        kw = rc(kw)
        c0 = kw.get("check")

        def check(m, step, data):
            if c0 is not None:
                c0(m, step, data)
            trig(m, step)
        kw["check"] = check
        return kw
    return rc2


def one_run(a, seed, iters, sched, keep=None, thr=ACC_THR):
    """A run of arm a: (record, hinge state or trigger info or None). Part D's trigger looks ahead to the part's full
    length (its last check is iters_c - 2400), whatever this run's length."""
    state = None
    if a["kind"] == "early":
        rec, state, _ = ter.one_run(ter.ARM["WIN16_A"], seed, iters, dict(ter.REAL, window=sched["window"], warm=sched["warm"]),
                                    keep=keep)
        return rec, state
    ctx = contextlib.nullcontext()
    if a["kind"] in ("local", "split", "reset"):
        rc, holder = make_local_rc(a["slow"], warm=sched["warm"], keep=keep)
        if a["kind"] != "local":
            state = dict(checks=[])
            rc = with_trigger(rc, make_trigger(tscur.TASK48, seed, state, lambda: holder["opts"][0], sched["iters_c"] - EVERY,
                                               thr=thr, op=a["kind"]))
            ctx = adam_capture(holder)
    elif a["kind"] == "hinge":
        rc, state = tss.make_recipe(True, TAU, warm=sched["warm"], keep=keep)
    else:
        rc = None if keep is None else (lambda kw: dict(kw, keep=keep))
    with ctx:
        rec = run_path(a, seed, iters, rc)
    if rec.get("ok"):
        rec.update(arm=a["key"], cpu=cpu_model())
        if a["kind"] != "hinge" and a["slow"]:
            rec["group_names"] = holder.get("names")
        if a["kind"] in ("split", "reset"):
            rec.update(checks=state["checks"], splits=sum(1 for c in state["checks"] if c["fired"]), max_splits=MAX_SPLITS,
                       trigger_op=a["kind"])
    return tss.norm(rec), state


def run_job(sp):
    a = ARM[sp["arm"]]
    return one_run(a, sp["seed"], iters_of(a, sp["sched"]), sp["sched"])[0]


def spec(key, seed, sched):
    return dict(arm=key, seed=seed, sched=dict(sched))


def check_job(key, seed, iters, thr=ACC_THR):
    rec, state = one_run(ARM[key], seed, iters, REAL, thr=thr)
    st = state or {}
    return dict(rec=rec, fired_at=list(st.get("fired_at", [])), diag=list(st.get("firing_diag", [])),
                checks=list(st.get("checks", [])))


def trigger_job(op):
    """CHECK 142: Part C's LOCAL3 model (seed 260) after one Adam step (so W_g has Adam state), the trigger called at
    each check with no training in between, threshold above 1 and rise 1 (every check from 4800 eligible)."""
    task = tscur.TASK48
    m = to_local(build_plain("C", D8_SEED), D8_SEED)
    opt = ter._ADAM(m.parameters(), lr=LR)
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    inp, tgt = x[:, :-1], x[:, 1:]
    ql, qt, _ = task.select(tbo.logits_of(m, inp), tgt, None)
    loss = F.cross_entropy(ql, qt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    info = dict(checks=[])
    trig = make_trigger(task, D8_SEED, info, lambda: opt, 26400, thr=1.01, rise=1.0, op=op)
    before = {k: float(v.abs().max()) for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
    wg0 = m.W_g.detach().clone()
    for t in range(EVERY, 26400 + 1, EVERY):
        trig(m, t)
    after = {k: float(v.abs().max()) for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
    rows = info["checks"]
    return dict(fired=[r["step"] for r in rows if r["fired"]], gap=[r["step"] for r in rows if r.get("blocked_by") == "gap"],
                cap=[r["step"] for r in rows if r.get("blocked_by") == "cap"], before=before, after=after,
                wg_same=torch.equal(wg0, m.W_g.detach()), dist=[r.get("row_dist_after") for r in rows if r["fired"]],
                targets=[(r["key_cs"], r["key_c0"], r["key_ok"]) for r in rows if r["fired"]])


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
def screen_record(fname, prefix):
    return screen_record_at(SCREEN_SHA, fname, prefix)


def screen_record_at(sha, fname, prefix):
    src = subprocess.run(["git", "-C", HERE, "show", f"{sha}:explore_out/{fname}"], capture_output=True, text=True,
                         check=True).stdout
    runs = json.loads(src)["runs"]
    ks = [k for k in runs if k == prefix or k.startswith(prefix + "|")]
    assert len(ks) == 1, (fname, prefix, ks)
    return ks[0], runs[ks[0]]


def same_through(new, old, t):
    """Curve, gradient norms and every statistic both records hold, through step t."""
    c = [x for x in new["curve"] if x[0] <= t] == [x for x in old["curve"] if x[0] <= t]
    g = {k: v for k, v in new["grad"].items() if int(k) <= t} == {k: v for k, v in old["grad"].items() if int(k) <= t}
    sn = [x for x in new["stats"] if x["step"] != "end" and x["step"] <= t]
    so = [x for x in old["stats"] if x["step"] != "end" and x["step"] <= t]
    s = (len(sn) == len(so) and len(sn) > 0
         and all(set(a) & set(b) and all(a[k] == b[k] for k in set(a) & set(b)) for a, b in zip(sn, so)))
    shared = sum(len(set(a) & set(b)) for a, b in zip(sn, so))
    return c, g, s, f"{shared} values at steps {[x['step'] for x in so]}"


def build_plain(part, seed):
    """The part's recurrent-gate model as its run path builds it."""
    if part == "A":
        return trr.make_fn(tsc.ARM["A"], seed)()
    a = tsr.ARM["A4k16"] if part == "B" else tscur.ARM["D8"]
    return tsr.builder_for(a)(a, seed)()


def task_of(part):
    return {"A": tsc.TASK, "B": tsr.TASKS["P4S4"], "C": tscur.TASK48}[part]


def gate_job(part):
    """CHECK 136 for one part's model (seed 160)."""
    task = task_of(part)
    p0 = dict(build_plain(part, REPRO_SEED).named_parameters())
    m = to_local(build_plain(part, REPRO_SEED), REPRO_SEED)
    p1 = dict(m.named_parameters())
    keep = all(torch.equal(p1[n], p0[n]) for n in p0)
    extra = [n for n in p1 if n not in p0]
    win = torch.equal(m.gate_conv.conv_w.detach(), window_init(REPRO_SEED))
    m.eval()
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    t = x.shape[1] - 1
    with torch.no_grad():
        g0 = m(x)[2]
        x3 = x.clone(); x3[:, t - 3] = (x3[:, t - 3] + 1) % task.vocab
        x2 = x.clone(); x2[:, t - 2] = (x2[:, t - 2] + 1) % task.vocab
        blind = torch.equal(m(x3)[2][:, t], g0[:, t])
        sees = not torch.equal(m(x2)[2][:, t], g0[:, t])
    ref = build_plain(part, REPRO_SEED)
    with torch.no_grad():
        ref.W_h.zero_()
        w = torch.zeros_like(m.gate_conv.conv_w)
        w[:, 0] = 1.0
        m.gate_conv.conv_w.copy_(w)
    ref.eval()
    with torch.no_grad():
        d = float((ref(x)[2] - m(x)[2]).abs().max())
    return dict(n=len(p0), keep=keep, extra=extra, win=win, wh_frozen=not m.W_h.requires_grad, t=t, blind=blind, sees=sees,
                d=d)


def lr_job(key):
    """CHECK 137: each group's lr at every update through the arm's real path and recipe, the forward and evaluation
    stubbed (no early stop), 2401 updates; the groups' parameters; W_h after the run."""
    a = ARM[key]
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
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
        rc, holder = make_local_rc(a["slow"], keep=keep)
        rec = run_path(a, 3, WARM + 1, rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    names = {id(p): n for n, p in m.named_parameters()}
    flat = [i for g in ids["g"] for i in g]
    train = {id(p) for p in m.parameters() if p.requires_grad}
    wh_same = torch.equal(m.W_h.detach(), build_plain(a["part"], 3).W_h.detach())
    return dict(ok=rec["ok"], n=len(lrs), at={str(u): lrs[u - 1] for u in (1, WARM, WARM + 1) if u <= len(lrs)},
                groups=[[names[i] for i in g] for g in ids["g"]], cover=len(flat) == len(set(flat)) and set(flat) == train,
                wh_in_group=id(m.W_h) in set(flat), wh_frozen=not m.W_h.requires_grad, wh_same=wh_same)


def earlier_seeds():
    """test_muon_recipe's earlier seeds and both machines' 400-479 / 1400-1479."""
    return tmr.earlier_seeds() | {s for m in MACHINES for p in "ABC" for s in tmr.RANGES[m][p]}


def complete(store):
    seeds = store.get("meta", {}).get("seeds") or {}
    missing = [f"{k}|{s}" for k in KEYS for s in seeds.get(k, []) if not ok_r(get(store, k, s))]
    return bool(seeds) and set(seeds) == set(KEYS) and not missing, missing


def other_of(machine):
    return "L" if machine == "X" else "X"


def check_also(machine, also):
    m = also.get("meta", {})
    oth = other_of(machine)
    seeds = m.get("seeds") or {}
    in_range = all(s in RANGES[oth][ARM[k]["part"]] for k in KEYS for s in seeds.get(k, []))
    mine = {s for k in KEYS for s in SEEDS[k]}
    theirs = {s for k in KEYS for s in seeds.get(k, [])}
    good = m.get("machine") == oth and bool(seeds) and in_range and m.get("cpu") != cpu_model() and not (mine & theirs)
    return good, (f"--also: machine {m.get('machine')} (should be {oth}); CPU {m.get('cpu')} (this machine {cpu_model()}; "
                  f"differs: {m.get('cpu') != cpu_model()}); its seeds {tsr.ranges(sorted(theirs)) if theirs else None} in "
                  f"{oth}'s ranges: {in_range}; overlap with this machine's: {sorted(mine & theirs) or 'none'}")


def check_144(f, cached):
    """CHECK 144: cached Part A-C records reproduce under this code through WARM (f: futures keyed ("144", arm))."""
    print("CHECK 144 (a resume) Part A-C records cached by an earlier start reproduce under this code through "
          f"{WARM}:")
    ok = True
    if not cached:
        print("     no cached Part A-C record (a first start)")
    for k in ("WIN3_SLOW", "WIN3_SLOW_D8"):
        if k not in cached:
            continue
        s0, rec_old = cached[k]
        r = f["144", k].result()["rec"]
        g = subset_equal(trunc(r, WARM), trunc(rec_old, WARM)) and subset_equal(trunc(rec_old, WARM), trunc(r, WARM))
        ok &= g
        print(f"     {k}|{s0}: curve {r['curve']} vs {[x for x in rec_old['curve'] if x[0] <= WARM]}; curve, statistics and "
              f"gradient norms equal {g}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    return ok


def verify(pool, files, machine, early, also, cached=None):
    f = {}
    f["141"] = pool.submit(check_job, "WIN3_SPLIT_D8", SPLIT_SEED, SPLIT_THROUGH)
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8"):
        f["143", k] = pool.submit(check_job, k, 1, 6000, 0.0)
    for op in ("split", "reset"):
        f["142", op] = pool.submit(trigger_job, op)
    cached = cached or {}
    for k in ("WIN3_SLOW", "WIN3_SLOW_D8"):
        if k in cached:
            f["144", k] = pool.submit(check_job, k, cached[k][0], WARM)
    f["135", "WIN3_SLOW"] = pool.submit(check_job, "WIN3_SLOW", REPRO_SEED, WARM)
    f["135", "WIN3"] = pool.submit(check_job, "WIN3", REPRO_SEED, WARM)
    for part in "ABC":
        f["136", part] = pool.submit(gate_job, part)
    for k in ARMS_RUN:
        if ARM[k]["kind"] == "local":
            f["137", k] = pool.submit(lr_job, k)
    f["138"] = pool.submit(check_job, "HINGE_D8", D8_SEED, WARM)
    f["140", "D8"] = pool.submit(check_job, "WIN3_SLOW_D8", 1, EVAL_EVERY)
    f["140", "A"] = pool.submit(check_job, "WIN3", 1, EVAL_EVERY)

    print("test_muon_recipe.py's verification (which runs test_early_recipe's, and so on down to")
    print("test_multilayer_binding's):")
    tmr.SEEDS = tmr.seeds_for(machine)
    tmr.verify(pool, files, machine, early, None)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 135 the window gate's code identity through {WARM} (records on claude/outside-ideas at {SCREEN_SHA}; asserted "
          f"on machine X, printed on L; this machine {machine}, {cpu_model()}):")
    good = True
    for key, fname, prefix in (("WIN3_SLOW", "window_recipe_results.json", f"LOCAL3_SLOW|{REPRO_SEED}"),
                               ("WIN3", "local_gate_results.json", f"LOCAL3|{REPRO_SEED}")):
        r = f["135", key].result()["rec"]
        k_old, old = screen_record(fname, prefix)
        c, g, s, shared = same_through(r, old, WARM)
        same = c and g and s and r["ok"]
        good &= same
        print(f"     {key}|{REPRO_SEED} vs {fname} {k_old} (written on {old.get('cpu') or 'not recorded'}): curve "
              f"{[x for x in r['curve'] if x[0] <= WARM]} vs {[x for x in old['curve'] if x[0] <= WARM]}; curve equal {c}; "
              f"gradient norms equal {g}; statistics equal {s} (shared {shared})  -> {'IDENTICAL' if same else 'DIFFERS'}")
    if machine == "X":
        ok &= good
    else:
        print(f"     (machine {machine}: printed, not asserted)")
    print()

    print("CHECK 136 the window gate (seed 160, each part's model):")
    good = True
    for part in "ABC":
        r = f["136", part].result()
        g = (r["keep"] and r["extra"] == ["gate_conv.conv_w"] and r["win"] and r["wh_frozen"] and r["blind"] and r["sees"]
             and r["d"] <= 1e-6)
        good &= g
        print(f"     Part {part}: the conversion keeps all {r['n']} parameters bitwise {r['keep']}, adds only {r['extra']} equal to "
              f"batch 1's window {r['win']}; W_h frozen {r['wh_frozen']}; the gate at t = {r['t']} ignores token t-3 {r['blind']} "
              f"and sees token t-2 {r['sees']}; window [1, 0, 0] and W_h = 0 vs the recurrent gate with W_h = 0: max |diff| "
              f"{r['d']:.2e}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 137 groups and lrs at updates 1, {WARM}, {WARM + 1} through the real paths (forward stubbed):")
    good = True
    for k in ARMS_RUN:
        if ARM[k]["kind"] != "local":
            continue
        r = f["137", k].result()
        if ARM[k]["slow"]:
            g = (r["cover"] and len(r["groups"]) == 2 and r["groups"][0] == list(GATE_L) and r["at"]["1"] == [LR, LR_WARM]
                 and r["at"][str(WARM)] == [LR, LR_WARM] and r["at"][str(WARM + 1)] == [LR, LR] and not r["wh_in_group"])
        else:
            g = len(r["groups"]) == 1 and all(r["at"][str(u)] == [LR] for u in (1, WARM, WARM + 1))
        g = g and r["wh_frozen"] and r["wh_same"] and r["n"] == WARM + 1 and r["ok"]
        good &= g
        print(f"     {k:<12} groups {[len(x) for x in r['groups']]} (gate group {r['groups'][0] if ARM[k]['slow'] else 'all'}); "
              f"every trainable parameter once {r['cover']}; lrs at 1 {r['at'].get('1')}, {WARM} {r['at'].get(str(WARM))}, "
              f"{WARM + 1} {r['at'].get(str(WARM + 1))}; W_h frozen {r['wh_frozen']}, in a group {r['wh_in_group']}, unchanged "
              f"{r['wh_same']}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 138 HINGE_D8 seed {D8_SEED} on Part C's path vs the recorded test_slow_start HINGE_D8|{D8_SEED} "
          f"({files['slow_path']}) through {WARM}:")
    r = f["138"].result()["rec"]
    old = get(files["slow"], "HINGE_D8", D8_SEED)
    rep = old is not None and subset_equal(trunc(r, WARM), trunc(old, WARM))
    m_ = files["slow"].get("meta", {})
    print(f"     curve {r['curve']} vs {[c for c in old['curve'] if c[0] <= WARM] if old else None}; statistics and gradient norms "
          f"equal {rep}; the file was written on {m_.get('cpu')} (git {m_.get('git')})  -> {'IDENTICAL' if rep else 'DIFFERS'}")
    ok &= rep
    print()

    print(f"CHECK 139 the seeds (machine {machine}):")
    rg = RANGES[machine]
    parts_ok = all(SEEDS[k] == (tuple(rg[ARM[k]["part"]])[:RESET_N] if k == "WIN3_RESET_D8" else tuple(rg[ARM[k]["part"]]))
                   for k in ARMS_RUN)
    ceil_ok = all(SEEDS[k] == tuple(rg[p])[:VALID_N] for p, k in CEILS)
    mine = {s for k in KEYS for s in SEEDS[k]}
    other = {s for p in "ABC" for s in RANGES[other_of(machine)][p]}
    earlier = earlier_seeds()
    g = parts_ok and ceil_ok and not (mine & earlier) and not (mine & other)
    print(f"     this machine's seeds {tsr.ranges(sorted(mine))}: every arm on its part's range (WIN3_RESET_D8 on its first "
          f"{RESET_N}) {parts_ok}; the perfect gates on "
          f"each part's first two seeds {ceil_ok}; disjoint from the other machine's ranges {not (mine & other)} and from every "
          f"earlier main-line seed {not (mine & earlier)} (earlier: {tsr.ranges(sorted(earlier))})")
    if also is not None:
        ga, line = check_also(machine, also)
        print(f"     {line}  -> {'OK' if ga else 'WRONG'}")
        g &= ga
    else:
        print("     no --also file")
    ok &= g
    print()

    print("CHECK 140 a worker's run is bit-identical to the same run here:")
    good = True
    for which, key in (("D8", "WIN3_SLOW_D8"), ("A", "WIN3")):
        rw = f["140", which].result()
        rh = check_job(key, 1, EVAL_EVERY)
        same = subset_equal(strip_all(rw["rec"]), strip_all(rh["rec"])) and subset_equal(strip_all(rh["rec"]), strip_all(rw["rec"]))
        g = same and rh["rec"]["ok"]
        good &= g
        print(f"         {key} seed 1, {EVAL_EVERY} steps: records equal {same}   acc {rh['rec']['curve'][-1][1]}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 141 WIN3_SPLIT_D8 seed {SPLIT_SEED} vs batch 17's W_SPLIT|{SPLIT_SEED} (window_d8_results.json on "
          f"claude/outside-ideas at {SPLIT_SHA}) through {SPLIT_THROUGH} (asserted on machine X, printed on L; this machine "
          f"{machine}):")
    r = f["141"].result()
    k_old, old = screen_record_at(SPLIT_SHA, "window_d8_results.json", f"W_SPLIT|{SPLIT_SEED}")
    c, g_, s_, shared = same_through(r["rec"], old, SPLIT_THROUGH)
    rn = [x for x in r["checks"] if x["step"] <= SPLIT_THROUGH]
    ro = [x for x in old["checks"] if x["step"] <= SPLIT_THROUGH]
    rows_eq = len(rn) == len(ro) > 0 and all(all(a[k] == b[k] for k in set(a) & set(b)) for a, b in zip(rn, ro))
    g = c and g_ and s_ and rows_eq and r["rec"]["ok"]
    print(f"     {k_old} (written on {old.get('cpu')}): curve {r['rec']['curve']} vs {[x for x in old['curve'] if x[0] <= SPLIT_THROUGH]}; "
          f"curve equal {c}; gradient norms equal {g_}; statistics equal {s_} ({shared}); the trigger's rows at "
          f"{[x['step'] for x in ro]} equal {rows_eq} (fired at {[x['step'] for x in rn if x['fired']]} vs "
          f"{[x['step'] for x in ro if x['fired']]}; targets {[(x['key_cs'], x['key_c0'], x['key_ok']) for x in rn if x['fired']]})  -> "
          f"{'IDENTICAL' if g else 'DIFFERS'}")
    if machine == "X":
        ok &= g
    else:
        print(f"     (machine {machine}: printed, not asserted)")
    print()

    print("CHECK 142 the trigger's mechanics (Part C's LOCAL3 model, seed 260, after one Adam step; checks 2400..26400 with no "
          "training between; threshold above 1):")
    good = True
    for op in ("split", "reset"):
        r = f["142", op].result()
        zeroed = all(v == 0.0 for v in r["after"].values()) and any(v > 0 for k, v in r["before"].items() if k != "step")
        g = (r["fired"] == [4800, 9600, 14400] and r["gap"] == [7200, 12000] and r["cap"] == [16800, 19200, 21600, 24000, 26400]
             and zeroed and (all(d > 0 for d in r["dist"]) and not r["wg_same"] if op == "split" else r["wg_same"]))
        good &= g
        print(f"     {op:<5}: fired at {r['fired']}, blocked by the gap at {r['gap']}, by the cap at {r['cap']}; W_g's Adam state "
              f"max |.| before {r['before']} -> after {r['after']} (zeroed {zeroed}); W_g unchanged {r['wg_same']}"
              + (f"; rows c*, c0 apart after each split {r['dist']}; targets {r['targets']}" if op == "split" else "")
              + f"  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 143 with the threshold at 0 (never eligible) the Part D arms equal WIN3_SLOW_D8 (seed 1) bit for bit through 6000:")
    base = f["143", "WIN3_SLOW_D8"].result()["rec"]
    good = True
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        r = f["143", k].result()
        same = subset_equal(trunc(r["rec"], 6000), trunc(base, 6000)) and subset_equal(trunc(base, 6000), trunc(r["rec"], 6000))
        g = same and r["rec"]["ok"] and not any(x["fired"] for x in r["checks"]) and len(r["checks"]) == 2
        good &= g
        print(f"     {k}: curve, statistics and gradient norms equal {same}; trigger checks at {[x['step'] for x in r['checks']]}, "
              f"none fired  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    ok &= check_144(f, cached)
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(cpu=cpu_model())


# ── Reporting ────────────────────────────────────────────────────────────────
def validity(store, seeds):
    out = {}
    for part, k in CEILS:
        n = sum(bound_r(get(store, k, s)) for s in seeds[k])
        out[part] = (n, n >= VALID_N)
    return out


def tally(store, seeds):
    seeds = {k: tuple(seeds.get(k, ())) for k in KEYS}
    rA = {k: runs_of(store, k, seeds[k]) for k in KEYS}
    valid = validity(store, seeds)
    out = dict(rA=rA, valid=valid, seeds=seeds, claims={}, bands={}, printed={})
    for name, new, old, part in CLAIMS + (G3,):
        out["claims"][name] = dict(paired_by(rA[new], rA[old], seeds[new], lambda r, k=new: success(k, r)), valid=valid[part][1])
    for k in ARMS_RUN:
        out["bands"][k] = dict(c=sum(success(k, r) for r in rA[k].values()), n=len(seeds[k]), valid=valid[ARM[k]["part"]][1])
    for name, new, old, part in PRINTED:
        out["printed"][name] = paired_by(rA[new], rA[old], seeds[new], lambda r, k=new: success(k, r))
    return out


def reading_d(claims, bands):
    """The pooled reading: WIN3_SPLIT_D8 RELIABLE pooled and G2 and G4 SHOWN pooled."""
    b = bands["WIN3_SPLIT_D8"]
    band = "UNTESTED" if not b["valid"] else tscur.band_n(b["c"], b["n"])
    why = [f"WIN3_SPLIT_D8 {band} {b['c']}/{b['n']}" + ("" if band == "RELIABLE" else " (needs RELIABLE)")]
    why += [f"{g} {verdict(claims[g])}" for g in ("G2", "G4")]
    holds = band == "RELIABLE" and all(verdict(claims[g]) == "SHOWN" for g in ("G2", "G4"))
    return holds, "; ".join(why)


def pool_claim(ds):
    b, c = sum(d["b"] for d in ds), sum(d["c"] for d in ds)
    return dict(n=sum(d["n"] for d in ds), new=sum(d["new"] for d in ds), old=sum(d["old"] for d in ds), b=b, c=c,
                p=mcnemar_greater(b, c), valid=all(d.get("valid", True) for d in ds))


def verdict(d):
    return "UNTESTED" if not d["valid"] or d["n"] == 0 else ("SHOWN" if d["p"] < 0.05 else "NOT SHOWN")


def g3_verdict(d):
    return "UNTESTED" if not d["valid"] or d["n"] == 0 else ("HOLDS" if d["c"] - d["b"] <= G3_BOUND else "FAILS")


def band_str(b):
    lo, hi = tsr.wilson(b["c"], b["n"])
    band = "UNTESTED" if not b["valid"] else tscur.band_n(b["c"], b["n"])
    return (f"{b['c']}/{b['n']} -> {band}   Wilson 95% [{lo:.3f}, {hi:.3f}]  (RELIABLE >= {math.ceil(0.9 * b['n'])}, "
            f"MAJORITY >= {math.ceil(0.5 * b['n'])})")


def print_claims(cl, who):
    for name, new, old, part in sorted(CLAIMS + (G3,)):
        d = cl[name]
        if name == G3[0]:
            print(f"  {name}  {new} not worse than {old} by more than {G3_BOUND} discordant pairs ({OUTCOME[part]}): "
                  f"{d['new']}/{d['n']} vs {d['old']}/{d['n']}; {old} only {d['c']}, {new} only {d['b']}; d = {d['c'] - d['b']} "
                  f"(bound: d <= {G3_BOUND}; no p-value)" + ("" if d["valid"] else f"   (Part {part} not valid{who})"))
            print(f"     *** {name}{' (pooled)' if who == ' on a machine' else ''}: {g3_verdict(d)} ***")
            continue
        print(f"  {name}  {new} beats {old} ({OUTCOME[part]}): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, "
              f"{old} only {d['c']}; p = {d['p']:.4g}" + ("" if d["valid"] else f"   (Part {part} not valid{who})"))
        print(f"     *** {name}{' (pooled)' if who == ' on a machine' else ''}: {verdict(d)} ***")


def print_printed(pr, label):
    for name, new, old, part in PRINTED:
        d = pr[name]
        print(f"  Printed, not a claim{label}: {name}, {new} vs {old} ({OUTCOME[part]}): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; "
              f"{new} only {d['b']}, {old} only {d['c']}; one-sided p = {d['p']:.4g} ({new} higher), "
              f"{mcnemar_greater(d['c'], d['b']):.4g} ({old} higher)")


def print_bands(bands, label):
    for k in ARMS_RUN:
        print(f"  band {k:<12} {label}({OUTCOME[ARM[k]['part']]}): {band_str(bands[k])}")


def report(store, seeds, machine, wall, path, also):
    t = tally(store, seeds)
    rA, valid = t["rA"], t["valid"]
    print()
    print("#" * 100)
    print(f"PER-SEED RAW RESULTS — every value, before any aggregate (machine {machine}, {store.get('meta', {}).get('cpu')})")
    print("#" * 100)
    print("  (columns as test_slow_start's; Parts B and C: outcome = test_stream_recipe.outcome; WIN16_A's firings: each "
          "firing's update with the hinge/task gradient-norm ratio)")
    print()
    for k in KEYS:
        a = ARM[k]
        part = TPART[a["part"]]
        print(f"  {a['label']}   (outcome {'perfect gate: BOUND' if a['ceil'] else OUTCOME[a['part']]})")
        raw_rows(k, part, rA[k], seeds[k])
        if a["kind"] == "early":
            print(f"  {k} firings (update(ratio)):")
            for s in seeds[k]:
                if ok_r(rA[k].get(s)):
                    print(f"    s{s}  {fire_str(rA[k][s])}")
        print()

    print("=" * 100)
    print("VALIDITY (the perfect gate, Adam 1e-3, binds on the part's first two seeds)")
    print("=" * 100)
    for part, k in CEILS:
        print(f"  Part {part}: {k} bound {valid[part][0]}/{len(seeds[k])} "
              f"({', '.join(f's{s} ' + (('BOUND at ' + str(r['transition'])) if bound_r(r) else 'not bound') for s, r in sorted(rA[k].items()))})"
              f"  -> {'VALID' if valid[part][1] else 'NOT VALID: the claims of this part are UNTESTED'}")
    print()

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in ARMS_RUN:
        done = [r for r in rA[k].values() if ok_r(r)]
        trs_ = [r["transition"] for r in done if r["transition"] is not None]
        print(f"  {ARM[k]['label']:<66} {OUTCOME[ARM[k]['part']]} {sum(success(k, r) for r in done):>2}/{len(seeds[k])}   bound "
              f"{len(trs_):>2}   collapsed {sum(bool(r['collapsed']) for r in done):>2}   completed {len(done)}/{len(seeds[k])}   "
              f"transition {med_int(trs_) if trs_ else '--'}")
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
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05; G3 a bound)")
    print("#" * 100)
    to = None
    if other is not None:
        om = other.get("meta", {})
        to = tally(other, {k: tuple(v) for k, v in om["seeds"].items()})
        both = {machine: t, other_of(machine): to}
        print(f"POOLED (PRIMARY): machine {machine} ({store.get('meta', {}).get('cpu')}) + machine {other_of(machine)} "
              f"({om.get('cpu')}, {also}); b and c summed over both machines' pairs")
        pc = {name: pool_claim([both[m_]["claims"][name] for m_ in MACHINES]) for name, *_ in CLAIMS + (G3,)}
        print("  (per machine: " + "; ".join(f"{name} " + ", ".join(f"{m_} {both[m_]['claims'][name]['b']} vs "
                                                                     f"{both[m_]['claims'][name]['c']}" for m_ in MACHINES)
                                             for name, *_ in CLAIMS + (G3,)) + ")")
        print_claims(pc, " on a machine")
        pb = {k: dict(c=sum(both[m_]["bands"][k]["c"] for m_ in MACHINES), n=sum(both[m_]["bands"][k]["n"] for m_ in MACHINES),
                      valid=all(both[m_]["bands"][k]["valid"] for m_ in MACHINES)) for k in ARMS_RUN}
        print_bands(pb, "pooled ")
        holds, why = reading_d(pc, pb)
        print("  READING (pooled):")
        print(f"     {'' if holds else 'not '}\"{READING_D}\" ({why})")
        print_printed({name: pool_claim([both[m_]["printed"][name] for m_ in MACHINES]) for name, *_ in PRINTED}, " (pooled)")
        print()
    else:
        print("POOLED (PRIMARY): not computed here — it needs --also with the other machine's complete file")
        print()
    print(f"PER MACHINE (SECONDARY): machine {machine}")
    print_claims(t["claims"], " on this machine")
    print_bands(t["bands"], "")
    print(f"  (the reading \"{READING_D}\" is pooled only)")
    print_printed(t["printed"], "")
    for k in ARMS_RUN:
        trs_ = [r["transition"] for r in rA[k].values() if ok_r(r) and r["transition"] is not None]
        print(f"     median transition {k:<12} {med_int(trs_) if trs_ else 'none bound'} ({len(trs_)} bound)")
    print()
    if to is not None:
        print(f"  Machine {other_of(machine)} (from --also, the same tests on its pairs):")
        print_claims(to["claims"], f" on {other_of(machine)}")
        print_bands(to["bands"], "")
        print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  failure classes and outcomes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome):")
    for k in ARMS_RUN:
        part = TPART[ARM[k]["part"]]
        cls = {}
        for s, r in sorted(rA[k].items()):
            cls.setdefault(tag(part, r), []).append(s)
        print(f"    {k:<12} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<32} {' '.join('s' + str(x) for x in v)}")
    print()
    print("  Part A per seed: WIN3 | WIN3_SLOW | HINGE0 (HINGE0's hinge firings 1-1200 / 1-2400 / all):")
    for s in seeds["WIN3_SLOW"]:
        a_, b_, c_ = (rA[k].get(s) for k in ("WIN3", "WIN3_SLOW", "HINGE0"))
        print(f"    s{s}  {tag('0', a_):<26} | {tag('0', b_):<26} | {tag('0', c_):<26} {hstr(c_) if ok_r(c_) else ''}")
    print()
    for part, pair in (("B", ("WIN3_SLOW16", "WIN16_A")), ("C", ("WIN3_SLOW_D8", "HINGE_D8"))):
        print(f"  Part {part}: merged runs (two or more streams on one channel at the end) and the map at "
              f"{', '.join(str(x) for x in MAP_AT)} and the end (distinct channels holding the streams, streams per channel):")
        for k in pair:
            mg = [s for s, r in sorted(rA[k].items()) if ok_r(r) and merged(r)]
            print(f"    {k}: merged at the end {len(mg)} ({' '.join('s' + str(x) for x in mg) or '-'})")
            for s in seeds[k]:
                r = rA[k].get(s)
                if not ok_r(r):
                    continue
                print(f"      s{s}  " + "  ".join(f"{x}: {groups_str(cmap_at(r, x)):<30}" for x in MAP_AT)
                      + f"end: {groups_str(cmap_at(r, 'end')):<30} {tag(TPART[part], r)}  (transition {fmt_step(r['transition'])}; "
                        f"shared {shared_max(r)}; hinge {hstr(r)})")
        print(f"  Part {part} per seed: {pair[0]} | {pair[1]} (transition):")
        for s in seeds[pair[0]]:
            a_, b_ = rA[pair[0]].get(s), rA[pair[1]].get(s)
            ta = f"{tag(TPART[part], a_)} ({fmt_step(a_['transition']) if ok_r(a_) else '--'})"
            tb = f"{tag(TPART[part], b_)} ({fmt_step(b_['transition']) if ok_r(b_) else '--'})"
            print(f"    s{s}  {ta:<40} | {tb}")
        print()
    def outc(r):
        return f"{tag('C', r)} ({fmt_step(r['transition']) if ok_r(r) else '--'})"
    print("  Part D per seed: HINGE_D8 | WIN3_SLOW_D8 | WIN3_SPLIT_D8 | WIN3_RESET_D8 (transition):")
    for s in seeds["WIN3_SPLIT_D8"]:
        print(f"    s{s}  " + " | ".join(f"{outc(rA[k].get(s)) if rA[k].get(s) else '--':<34}"
                                     for k in ("HINGE_D8", "WIN3_SLOW_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8")))
    print()
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        rows_all = [(s, r) for s, r in sorted(rA[k].items()) if ok_r(r)]
        fired = [c for _, r in rows_all for c in r.get("checks", []) if c["fired"]]
        print(f"  {k}: the trigger's {'splits' if k == 'WIN3_SPLIT_D8' else 'resets'} — {len(fired)} in {len(rows_all)} runs "
              f"(per run {[sum(1 for c in r.get('checks', []) if c['fired']) for _, r in rows_all]}); targets labelled ok "
              f"{sum(1 for c in fired if c['key_ok'])}/{len(fired)}")
        for s, r in rows_all:
            ch = r.get("checks", [])
            ev = [f"{c['step']}: {c['key_cs']}->{c['key_c0']} {'ok' if c['key_ok'] else 'NOT labelled'} "
                  f"(map {groups_str(c['map_before'])} -> {groups_str(c.get('map_after'))})" for c in ch if c["fired"]]
            bl = [f"{c['step']} ({c.get('blocked_by')})" for c in ch if c["blocked"]]
            print(f"    s{s}  {tag('C', r)} (transition {fmt_step(r['transition'])}); fired: {'; '.join(ev) or 'none'}; "
                  f"blocked: {', '.join(bl) or 'none'}")
            print(f"           maps at the checks (step: distinct channels / probe acc): "
                  + " ".join(f"{c['step']}:{len(set(c['map_before']))}/{c['acc']:.2f}{'*' if c['fired'] else ''}" for c in ch))
        print()
    done = [r for r in rA["WIN16_A"].values() if ok_r(r) and firings(r)[0] is not None]
    fa = [firings(r)[0] for r in done]
    rat = [d_["ratio"] for r in done for d_ in (firings(r)[1] or [])]
    first = [x[0] for x in fa if x]
    print(f"  WIN16_A firings: runs with a firing {sum(1 for x in fa if x)}/{len(done)}; first firing "
          f"{med_int(first) if first else '--'}; ratio {(f'{statistics.median(rat):.1f} [{min(rat):.1f}, {max(rat):.1f}]') if rat else '--'}")
    cpus = {}
    for k in KEYS:
        for r in rA[k].values():
            if ok_r(r):
                cpus.setdefault(r.get("cpu") or (r.get("host") or {}).get("cpu"), set()).add(k)
    print(f"  CPU labels in the records: {{{', '.join(f'{c!r}: {len(v)} arms' for c, v in cpus.items())}}}")
    print()

    if to is not None:
        print("=" * 100)
        print("X AND L SIDE BY SIDE (each machine's own seeds; pooled counts are the sums)")
        print("=" * 100)
        both = {machine: t, other_of(machine): to}
        print(f"  {'':40} {'X':>14} {'L':>14} {'pooled':>14}")
        for k in KEYS:
            cs = [(sum((bound_r(r) if ARM[k]['ceil'] else success(k, r)) for r in both[m_]["rA"][k].values()),
                   len(both[m_]["seeds"][k])) for m_ in MACHINES]
            cells = [f"{c}/{n}" for c, n in cs] + [f"{sum(c for c, _ in cs)}/{sum(n for _, n in cs)}"]
            lab = "perfect gate BOUND" if ARM[k]["ceil"] else OUTCOME[ARM[k]["part"]]
            print(f"  {k:<12} {lab:<27} " + " ".join(f"{c_:>14}" for c_ in cells))
        for name, *_ in CLAIMS + (G3,):
            ds = [both[m_]["claims"][name] for m_ in MACHINES]
            cells = [f"{d_['b']} vs {d_['c']}" for d_ in ds] + [f"{sum(d_['b'] for d_ in ds)} vs {sum(d_['c'] for d_ in ds)}"]
            print(f"  {name + ' discordant (first only vs second only)':<40} " + " ".join(f"{c_:>14}" for c_ in cells))
        print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps)")
    print("=" * 100)
    for k in KEYS:
        part = TPART[ARM[k]["part"]]
        for s in seeds[k]:
            r = get(store, k, s)
            if ok_r(r):
                print(f"  {k:<12} s{s} {curve_str(r['curve'])}  -> {tag(part, r)}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


def main():
    global SEEDS
    ap = argparse.ArgumentParser(description="The window gate (LOCAL3) with slow memory, no hinge, at two, four and eight streams.")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--early", default=None, help="this machine's test_early_recipe results (inherited CHECK 130)")
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (CHECK 138; inherited)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (inherited)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (inherited)")
    ap.add_argument("--scale", default=tsa.RESULTS_FILE, help="this machine's test_scale_axes results (inherited)")
    ap.add_argument("--short", default=tsc.RESULTS_FILE, help="this machine's test_short_conv results (inherited)")
    ap.add_argument("--also", default=None, help="the other machine's window_gate_results.json (pooled claims)")
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
        report(store, SEEDS, args.machine, 0.0, args.results, args.also)
        return

    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    early = load_store(early_path)
    if early.get("meta", {}).get("cpu") != cpu_model():
        print(f"STOP: {early_path} was written on {early.get('meta', {}).get('cpu')}, this machine is {cpu_model()} "
              f"(the inherited CHECK 130 needs this machine's own test_early_recipe file)")
        sys.exit(2)
    also = load_store(args.also) if args.also else None
    files = {"early": early, "early_path": early_path}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    sched = dict(SCHED)

    print("=" * 100)
    print("Window gate: batch 1's LOCAL3 with slow memory, no hinge, at two, four and eight streams")
    print(f"  machine {args.machine}: {cpu_model()}")
    print(f"  Part A {sched['iters_a']} steps (DISCOVERED); Part B {sched['iters_b']} steps, Part C {sched['iters_c']} steps "
          f"(BOUND ROUTED); SLOW: gate group at {LR:g}, the rest at {LR_WARM:g} for updates 1-{sched['warm']}")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<70} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  --early      {early_path} (CPU {early['meta'].get('cpu')}, git {early['meta'].get('git')})")
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
        store = load_store(args.results)
        old_m = store.get("meta", {})
        if old_m and old_m.get("machine") not in (None, args.machine):
            print(f"STOP: {args.results} holds machine {old_m.get('machine')}'s runs")
            sys.exit(2)
        cached = {}
        for k in ("WIN3_SLOW", "WIN3_SLOW_D8"):
            hit = [(s, get(store, k, s)) for s in SEEDS[k] if ok_r(get(store, k, s))]
            if hit and not args.force:
                cached[k] = hit[0]
        verify(pool, files, args.machine, early, also, cached)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = dict(machine=args.machine, torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head,
                    lr=LR, lr_warm=LR_WARM, tau=TAU, sched=sched, early=dict(path=early_path, git=early["meta"].get("git")),
                    seeds={k: list(v) for k, v in SEEDS.items()}, drop="", started=now)
        if old_m.get("started") and not args.force:
            added = [k for k in KEYS if k not in (old_m.get("seeds") or {})]
            meta.update(git=old_m.get("git"), started=old_m["started"],
                        starts=list(old_m.get("starts", [])) + [dict(git=head, started=now, workers=args.workers, arms_added=added)])
            print(f"  resuming {args.results}: first started {old_m['started']} at {old_m.get('git')}; this start {now} at {head}; "
                  f"arms added {added or 'none'}")
        store["meta"] = meta
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
            print(f"  {k:<12} {cost[k] * 1000:6.1f} ms/step x {iters_of(ARM[k], sched)} steps x {len(SEEDS[k])} seeds   "
                  f"(a full run {full[k] / 60:.1f} min)")
        print(f"  {len(d)} runs: serial {sum(d) / 3600:.2f} h, {total_h:.2f} h on {args.workers} workers; nothing is cut "
              f"(Parts B and C are never cut; no cut rule for Part A)")
        store["meta"].update(projected_wall_h=total_h, ms_per_step={k: cost[k] * 1000 for k in KEYS})
        save_results(args.results, store)
        print()

        def describe(key, rec):
            if not rec.get("ok"):
                return f"*** FAILED: {rec.get('error')}"
            part = TPART[ARM[key]["part"]]
            tr = rec["transition"]
            mid = (f"map [{','.join(str(c) for c in rec['end']['ch_map'])}]" if part != "0" and rec["end"].get("ch_map")
                   else f"VALcos {rec['val_cos']:.3f}" if rec.get("val_cos") is not None else "")
            return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} {mid}  "
                    f"{tag(part, rec)}  hinge {hstr(rec)}" + (f"  {rec.get('trigger_op')}s at "
                                                               f"{[c['step'] for c in rec['checks'] if c['fired']]}"
                                                               if rec.get("checks") is not None else ""))

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<12} seed {sp['seed']}  {describe(sp['arm'], rec)}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

        specs = []
        for k in sorted(KEYS, key=lambda k: (not ARM[k]["ceil"], -full[k])):
            specs += [spec(k, s, sched) for s in SEEDS[k]]
        todo = []
        for sp in specs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<12} seed {sp['seed']}  cached")
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

    report(store, SEEDS, args.machine, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
