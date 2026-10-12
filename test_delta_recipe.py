#!/usr/bin/env python
"""
test_delta_recipe.py — at eight streams, does the delta-rule memory (β = 1, L2 keys) bind as well as the Hebbian one
under the window gate's recipe with the exact-copy split, and is the split an ingredient on delta channels too?
Everything is fixed before any run (specification: specs/test_delta_recipe.md, committed before this file; no arm
runs before Session V's review of it is read and resolved by a dated amendment there).

    python3 -u test_delta_recipe.py --machine X --workers 4          # CHECKs, projection, all runs, report
    python3 -u test_delta_recipe.py --machine L --report --also results/X/delta_recipe_results.json
    python3 longrun.py start delta_recipe_X --results delta_recipe_results.json --total 84 -- \\
        python3 -u test_delta_recipe.py --machine X --workers 4     # detached, resumable

BACKGROUND (exploratory)
- F's S76 (claude/explore-F, report 5 at 5b9bc2a; seeds 1310-1319, paired): LOCAL3 + SLOW + H's exact-copy split bound
  routed 9/10 on delta channels (β 1, L2 keys) and 10/10 on Hebbian ones (Hebbian only 1, delta only 0). Delta bound
  sooner on 8 of the 9 seeds both bound (median 14400 vs 24000). The delta perfect gate bound at 1200 (Hebbian
  2400-6000). 6 of the 12 delta splits fired on targets not labelled ok.
- S54(c), with a learned β and S48's noisy split with a reset: 3/10 against 9/10.
- H's S61 (claude/explore-H at 7e9197f): the exact copy alone (COPY_NONOISE) 9/10 on 480-489.

CONFIGURATION (test_window_gate's Part C/D path): test_stream_curriculum.run_sc with D8's settings: task_for(4, 8)
(S = 8, P = 4, n_vals 16, n_q 1, L = 99), conv 'layer' K = 4, batch 32, 43200 updates, evaluation every 1200 on 2048
held-out sequences, run_one's early stop (three evaluations >= 0.95).

THE RECIPE of the learned-gate arms (test_window_gate's WIN3_SLOW_D8, imported):
- The gate is LOCAL3 (test_window_gate.to_local).
- Adam 1e-3 on the gate group (W_in, W_g, the window); every other trainable parameter at 1e-4 for updates 1-2400 and
  1e-3 after the evaluation at 2400 (SLOW). W_h is frozen.
- No hinge.

THE MEMORIES
- HEBB: the model's Hebbian memory.
- DELTA: explore_delta_mem.py (claude/explore-F at 9a6c2bf, byte for byte) with β = 1 fixed, L2-normalised keys, tied
  write, decay 0.95. It is applied after the LOCAL3 conversion through explore_f_delta_fast.to_delta_fast(m, seed=seed,
  beta=1.0), F's fused evaluation of the same rule (S76's form). It adds no parameter.
- Arms at a seed share their initial parameters and their training batches (run_one's generator, seeded by the seed).

THE SPLIT (H's exact-copy form, in test_window_gate's trigger)
- Checks at updates 2400, 4800, ..., 40800. 2400 is the reference only; the first eligible check is 4800.
- The probe: task.make_batch(64, torch.Generator().manual_seed(12345)). That is 64 sequences, one query each; queries
  per stream (streams 0-7): 5, 5, 6, 8, 10, 12, 5, 13.
- A check fires if the probe accuracy < 0.95 and it rose < 0.02 since the previous check; at most 3 operations, >= 4800
  updates apart.
- c* and c0 are the argmax and argmin over the 16 channels of the read gate's mean mass. The mass is taken over the 64
  probe sequences and the 32 body-key positions 3j + 1, j = 0..31, of the input; the query's key at 97 is excluded. The
  read gate is computed in eval mode at τ_end.
- The operation: W_g[c0] := W_g[c*] exactly. No noise; c*'s row unchanged; W_g's Adam state untouched.
- Labelled ok if c* holds >= 2 streams and c0 none on test_scale_axes.routing_k's stream -> channel map (statistics
  probe probe_batch(task, seed)) just before the firing.

ARMS: X 580-599, L 1580-1599 (20 seeds each), paired by seed.
- HEBB_SPLIT: HEBB + the recipe + the split.
- HEBB_SLOW: HEBB + the recipe, no trigger (test_window_gate's WIN3_SLOW_D8; Amendment 1).
- DELTA_SPLIT: DELTA + the recipe + the split.
- DELTA_SLOW: DELTA + the recipe, no trigger.
- CEIL_HEBB: test_window_gate's CEIL_C (SC8_ceil's perfect gate, k = 16, ctx_pad 16, no curriculum), single-rate Adam
  1e-3, on 580-581 / 1580-1581 (validity, HEBB).
- CEIL_DELTA: the same, converted to DELTA (validity, DELTA).

OUTCOME and VALIDITY
- Outcome: BOUND ROUTED (test_stream_recipe.outcome: bound, one-to-one map by the largest mean read-gate mass at the
  body's value positions, every stream >= 0.9 at the end), with the transition.
- A part (HEBB, DELTA) is VALID on a machine if its perfect gate binds both seeds. A claim is UNTESTED (not a negative)
  on a machine where a part it uses is not VALID; pooled claims need both machines VALID. D1 and D3 use both parts,
  D2 uses DELTA, D4 uses HEBB.

CLAIMS (Amendment 1, after Referee V's review; seed pairs; pooled PRIMARY with b and c summed, per machine SECONDARY):
- D1  d = HEBB_SPLIT only - DELTA_SPLIT only, pooled, labelled: d <= -3 "delta not shown worse; ahead by >= 3 pairs
      (untested)" (Amendment 2); -2 <= d <= 2 "delta not shown worse by more than 2 pairs"; d >= 3 "delta not shown
      non-inferior". Printed beside it: the paired
      difference of BOUND ROUTED rates with Newcombe's 95% interval, and the review's operating characteristics (a bound,
      not a non-inferiority test). Per machine: the rate difference and interval only.
- D2  DELTA_SPLIT beats DELTA_SLOW; D4 HEBB_SPLIT beats HEBB_SLOW: exact one-sided McNemar, SHOWN if p < 0.05. If the
      no-trigger arm binds routed on >= 36 of the pooled 40, that claim is NO ROOM (printed, not read). Amendment 2: a
      seed whose split arm's curve differs from its no-trigger arm's before the first firing (CHECK 174) is dropped from
      that claim, with the count printed.
- D3  (secondary, printed) on the seeds both split arms are BOUND ROUTED: the median transitions and an exact one-sided
      sign test of "DELTA earlier" (ties dropped).
READINGS (pooled): the memory rule = D1's label. The split, per write: "helps" (SHOWN), "no room", or "not shown"; named
combinations: helps/helps "the split helps on both writes"; delta not shown / Hebbian helps "the split is shown on
Hebbian, not on delta" (Amendment 2); delta helps / Hebbian not shown "the split helps on delta"; not shown / not shown
"the split's effect is not shown on either write". If either write is "no room", both labels are printed and no
combination is named. Each label is printed with its no-trigger count and McNemar's b and c. UNTESTED gives no reading.
PRINTED, NOT CLAIMS: the largest pairwise overlap sum_c min(g_i[c], g_j[c]) of the streams' value-position gates (the
referee's function, copied; CHECK 172) and "BOUND ROUTED with overlap <= 0.1" per arm; MERGED counts and failure
classes (fail_class_v2.outcome_v2); every firing with its label; every check's probe accuracy beside the held-out
accuracy and the count of probe-blind checks; bands with Wilson 95% intervals for every arm. Map ties: torch.argmax on
unrounded values (lowest index); the label and the outcome use the same map function on the same probe.

CHECKS (before any run of an arm): the inherited chain (test_split_copy's verification, with test_window_gate's and
everything below), then 158-170, 172 and 173 as specs/test_delta_recipe.md and its Amendments 1-2 state them; 171 and
174 in the report. CHECKs 159 and 162 read their reference records from results/refs/ (Amendment 2 (h)).

RUNTIME: projection with pool-load timing before the first run (longrun.projection). Nothing is cut. Perfect gates
first, then longest first. Runs are cached in delta_recipe_results.json (gitignored, written atomically after each
run); a restart resumes from the cached records.

RESULT (X): test commit a581027 (spec 05256c5 + Amendment 1 32c121b before the run). The run started 2026-10-11 02:12 UTC
through longrun.py; runs began at 05:27 after the verification. Intel(R) Xeon(R) Processor @ 2.10GHz, torch
2.14.0+cu130, 4 workers x 1 thread, one start (two restarts of the agent's worker did not touch it), nothing cut. 84/84
runs, none failed; 29.2 h wall clock (projection 45.3 h worst case). Amendment 2 (55dac50, report rules only) was
committed during the run; the figures below are its report, regenerated with --report (2225f00) from the same results
file. Its claims, bands and checks equal the in-run report's line for line: CHECK 174 dropped no pair.
- Verification: every inherited and own CHECK passed.
  - CHECK 130 (Muon, X's 2.80GHz file) printed, not asserted.
  - 158: identical through the first split at 4800, different from 6000; printed, the record is from 2.80GHz.
  - 159 (H's COPY_NONOISE|487) and 162 (F's DELTA_SPLIT|1310) bit for bit, asserted.
  - 161: LOCAL3 1.19e-7 under 1e-6. 173: fused = parallel.
- Validity: CEIL_HEBB bound 2/2 (3600, 3600); CEIL_DELTA 2/2 (2400, 1200). Both parts VALID on X.
- Counts (BOUND ROUTED):
  - HEBB_SPLIT 16/20 (median transition 17400);
  - HEBB_SLOW 7/20;
  - DELTA_SPLIT 19/20 (13200);
  - DELTA_SLOW 19/20 (14400; per seed the same outcome as DELTA_SPLIT, both failing only on s594, MERGED);
  - every BOUND ROUTED run has overlap <= 0.1.
- Per machine (SECONDARY; the labels and readings are pooled only and wait for L):
  - D1: DELTA_SPLIT only 4, HEBB_SPLIT only 1, d = -3; rate difference +0.150, 95% [-0.082, +0.374].
  - D2: 0 vs 0, p = 1, NOT SHOWN (DELTA_SLOW 19/20).
  - D4: 10 vs 1, p = 0.0059, SHOWN.
  - D3: on 15 shared binds, delta earlier on 10, later on 4, tied 1; sign test p = 0.090; medians 13200 vs 16800.
  - Bands: HEBB_SPLIT MAJORITY [0.584, 0.919]; HEBB_SLOW MINORITY [0.181, 0.567]; DELTA_SPLIT RELIABLE [0.764, 0.991];
    DELTA_SLOW RELIABLE [0.764, 0.991].
- Printed:
  - Firings: Hebbian 39 in 20 runs, every target labelled ok; delta 31 in 20 runs, 10 labelled ok.
  - Probe-blind (checks from 4800): HEBB_SPLIT 1 of 181, blocked by the probe alone 3; DELTA_SPLIT 1 of 142, alone 5.
  - CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs).
"""

import argparse
import collections
import contextlib
import hashlib
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

import test_binding_onset as tbo
from test_binding_onset import EVAL_EVERY, fmt_step
from test_router_curriculum import get, load_store, init_worker, cpu_model, save_results
from test_router_reliability import strip_all
from test_conv_lr import bound_r, mcnemar_greater
import test_stream_recipe as tsr
from test_stream_channels import med_int, mstr, shared_max
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import ok_r, subset_equal, trunc, curve_str
from test_recipe_scope import groups_str
import test_window_gate as tgw
import test_split_copy as tsp
import explore_f_delta_fast as ff
import fail_class_v2 as fc2
import longrun

# ── Settings (fixed before any run) ──────────────────────────────────────────
MACHINES = ("X", "L")
RANGES = {"X": range(580, 600), "L": range(1580, 1600)}
ITERS = tgw.MAX_C                          # 43200
VALID_N = tgw.VALID_N                      # 2 of 2
D1_BOUND = 2
BETA = 1.0
OVERLAP_MAX = 0.1                          # the referee's strict count (printed)
PROBE_COUNTS = (5, 5, 6, 8, 10, 12, 5, 13)  # the trigger probe's queries per stream (CHECK 163)
H_SHA, H_FILE, H_KEY, H_SEED = "7e9197f", "explore_out/H/copy2x2_results.json", "COPY_NONOISE|487", 487
F_SHA, F_FILE, F_KEY, F_SEED = "5b9bc2a", "explore_out/F/eight_results.json", "DELTA_SPLIT|1310", 1310
H_THROUGH, F_THROUGH = 7200, 2400
D_THROUGH = 7200                           # CHECK 158
D_SEED = {"X": 540, "L": 1541}             # the rule's seed per machine (CHECK 158 recomputes it)
PAIR_SEED = 580                            # CHECK 166
TIME_ITERS = 1200
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "delta_recipe_results.json"

ARMS = [
    dict(key="CEIL_HEBB", part="HEBB", ceil=True, label="CEIL_HEBB    perfect gate (k=16), Hebbian, Adam 1e-3 (validity, HEBB)"),
    dict(key="CEIL_DELTA", part="DELTA", ceil=True, label="CEIL_DELTA   perfect gate (k=16), delta (beta 1), Adam 1e-3 (validity, DELTA)"),
    dict(key="HEBB_SPLIT", part="HEBB", ceil=False, delta=False, trigger=True,
         label="HEBB_SPLIT   LOCAL3 + SLOW + the exact-copy split, Hebbian"),
    dict(key="HEBB_SLOW", part="HEBB", ceil=False, delta=False, trigger=False,
         label="HEBB_SLOW    LOCAL3 + SLOW, no trigger, Hebbian (Amendment 1)"),
    dict(key="DELTA_SPLIT", part="DELTA", ceil=False, delta=True, trigger=True,
         label="DELTA_SPLIT  LOCAL3 + SLOW + the exact-copy split, delta (beta 1, L2 keys)"),
    dict(key="DELTA_SLOW", part="DELTA", ceil=False, delta=True, trigger=False,
         label="DELTA_SLOW   LOCAL3 + SLOW, no trigger, delta (beta 1, L2 keys)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
ARMS_RUN = tuple(k for k in KEYS if not ARM[k]["ceil"])
CEILS = (("HEBB", "CEIL_HEBB"), ("DELTA", "CEIL_DELTA"))
PARTS = {"D1": ("HEBB", "DELTA"), "D2": ("DELTA",), "D3": ("HEBB", "DELTA"), "D4": ("HEBB",)}
NO_ROOM = 36                               # a no-trigger arm at >= 36 of the pooled 40 makes its split claim NO ROOM
D1_LABELS = ("delta not shown worse; ahead by >= 3 pairs (untested)", "delta not shown worse by more than 2 pairs",
             "delta not shown non-inferior")                                   # Amendment 2 (a)
SPLIT_READINGS = {("helps", "helps"): "the split helps on both writes",
                  ("not shown", "helps"): "the split is shown on Hebbian, not on delta",   # Amendment 2 (c)
                  ("helps", "not shown"): "the split helps on delta",
                  ("not shown", "not shown"): "the split's effect is not shown on either write"}
D1_OC = ("at 40 pairs with HEBB_SPLIT at 0.95, P(d <= 2) is 0.29 (independent outcomes) / 0.22 (nested) when DELTA_SPLIT "
         "is 0.85, and 0.11 / 0.05 when it is 0.80 (Referee V's spec review, fad1489): a bound, not a non-inferiority test")
RECOMPUTE_SHA = "0fdb363"                  # docs/review_2026-10/recompute.py's overlap (CHECK 172)


def seeds_for(machine):
    s = tuple(RANGES[machine])
    return {k: (s[:VALID_N] if ARM[k]["ceil"] else s) for k in KEYS}


SEEDS = seeds_for("X")                     # main() sets the machine's


def outcome2(r):
    return fc2.outcome_v2(r, tsr.routed)


def success(key, r):
    """BOUND ROUTED for the learned-gate arms; BOUND for the perfect gates."""
    if not ok_r(r):
        return False
    return bound_r(r) if ARM[key]["ceil"] else outcome2(r) == "BOUND ROUTED"


def overlap(r):
    """The largest pairwise overlap sum_c min(g_i[c], g_j[c]) of the streams' value-position gates (the referee's)."""
    g = (r.get("end") or {}).get("stream_gate")
    if not g:
        return None
    return max(sum(min(a, b) for a, b in zip(g[i], g[j])) for i in range(len(g)) for j in range(i + 1, len(g)))


# ── The exact-copy operation (H's copy_nonoise_op, claude/explore-H at 7e9197f, line for line) ──────────────────────
def copy_op(m, o, cs, c0, seed):
    W = m.W_g
    with torch.no_grad():
        before = float((W[cs] - W[c0]).norm())
        W[c0] = W[cs].clone()
    return dict(row_dist_before=before, row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=[])


@contextlib.contextmanager
def exact_copy():
    """test_window_gate's trigger applies split_op at a firing; here it applies copy_op instead (the trigger's code,
    probe, rule and caps unchanged)."""
    saved = tgw.split_op
    tgw.split_op = copy_op
    try:
        yield
    finally:
        tgw.split_op = saved


# ── Runs ─────────────────────────────────────────────────────────────────────
def with_delta(rc):
    """The recipe's model converted to the delta memory after every other conversion (LOCAL3 first, as S76)."""
    def rc2(kw):
        kw = rc(kw)
        b0 = kw["builder"]

        def builder(a, seed):
            mk = b0(a, seed)
            return lambda: ff.to_delta_fast(mk(), seed=seed, beta=BETA)
        return dict(kw, builder=builder)
    return rc2


def with_capture(rc, out):
    """Records the built model's initial parameters in out["init"] (CHECK 166)."""
    def rc2(kw):
        kw = rc(kw)
        b0 = kw["builder"]

        def builder(a, seed):
            mk = b0(a, seed)

            def make():
                m = mk()
                out["init"] = {n: p.detach().clone() for n, p in m.named_parameters()}
                out["cls"] = (type(m).__name__, type(m.attn).__name__)
                return m
            return make
        return dict(kw, builder=builder)
    return rc2


def run_learned(key, seed, iters, thr=tgw.ACC_THR, keep=None, capture=None):
    a = ARM[key]
    rc, holder = tgw.make_local_rc(True, warm=tgw.WARM, keep=keep)
    if a["delta"]:
        rc = with_delta(rc)
    if capture is not None:
        rc = with_capture(rc, capture)
    state = None
    with contextlib.ExitStack() as es:
        if a["trigger"]:
            state = dict(checks=[])
            rc = tgw.with_trigger(rc, tgw.make_trigger(tscur.TASK48, seed, state, lambda: holder["opts"][0],
                                                       tgw.MAX_C - tgw.EVERY, thr=thr, op="split"))
            es.enter_context(tgw.adam_capture(holder))
            es.enter_context(exact_copy())
        rec = tgw.run_path(tgw.ARM["WIN3_SLOW_D8"], seed, iters, rc)
    if rec.get("ok"):
        rec.update(arm=key, cpu=cpu_model(), group_names=holder.get("names"), memory="delta" if a["delta"] else "hebb",
                   delta_cfg=dict(beta=BETA, write="tied", normalize=True, decay="fixed", impl="fast") if a["delta"] else None)
        if a["trigger"]:
            rec.update(checks=state["checks"], splits=sum(1 for c in state["checks"] if c["fired"]),
                       max_splits=tgw.MAX_SPLITS, trigger_op="copy")
    return tss.norm(rec)


def run_ceil(key, seed, iters, keep=None, capture=None):
    if key == "CEIL_HEBB":
        rc = (lambda kw: dict(kw, keep=keep)) if keep is not None else (lambda kw: kw)
    else:
        rc = tsp.b_recipe(True, keep)
    if capture is not None:
        rc = with_capture(rc, capture)
    rec = tscur.run_sc(tgw.CEIL_C_ARM, seed, dict(tscur.REAL, total=iters), recipe=rc)
    if rec.get("ok"):
        rec.update(arm=key, cpu=cpu_model(), memory="delta" if key == "CEIL_DELTA" else "hebb")
    return tss.norm(rec)


def one_run(key, seed, iters, thr=tgw.ACC_THR, keep=None, capture=None):
    if ARM[key]["ceil"]:
        return run_ceil(key, seed, iters, keep=keep, capture=capture)
    return run_learned(key, seed, iters, thr=thr, keep=keep, capture=capture)


def run_job(sp):
    return one_run(sp["arm"], sp["seed"], sp["iters"])


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def time_job(key):
    t0 = time.time()
    one_run(key, 0, TIME_ITERS)
    return (time.time() - t0) / TIME_ITERS


# ── CHECK jobs ───────────────────────────────────────────────────────────────
def check_job(key, seed, iters, thr=tgw.ACC_THR):
    return dict(rec=one_run(key, seed, iters, thr=thr))


def copy_unit_job():
    """CHECK 160: the operation on Part C's LOCAL3 model (seed 260) after one Adam step."""
    task = tscur.TASK48
    m = tgw.to_local(tgw.build_plain("C", tgw.D8_SEED), tgw.D8_SEED)
    opt = tgw.ter._ADAM(m.parameters(), lr=tgw.LR)
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    ql, qt, _ = task.select(tbo.logits_of(m, x[:, :-1]), x[:, 1:], None)
    opt.zero_grad()
    F.cross_entropy(ql, qt).backward()
    opt.step()
    st0 = {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
    W0 = m.W_g.detach().clone()
    res = copy_op(m, opt, 3, 7, 0)
    W1 = m.W_g.detach()
    others = [r for r in range(W0.shape[0]) if r not in (3, 7)]
    return dict(copied=torch.equal(W1[7], W0[3]), cs_same=torch.equal(W1[3], W0[3]),
                others=all(torch.equal(W1[r], W0[r]) for r in others),
                state=all(torch.equal(opt.state[m.W_g][k], st0[k]) for k in st0) and bool(st0), res=res)


def trigger_copy_job():
    """CHECK 165: CHECK 142's harness (test_window_gate.trigger_job) with the exact copy."""
    with exact_copy():
        return tgw.trigger_job("split")


def beta0_job():
    """CHECK 161: the delta memory at beta 0 (erase off, write 1, raw keys) vs the Hebbian model, on the perfect-gate
    model and the LOCAL3 model."""
    import copy
    import explore_delta_mem as dm
    task = tscur.TASK48
    out = []
    a = tgw.CEIL_C_ARM
    tp = tsr.padded_task(tsr.TASKS[a["task"]], a["ctx_pad"])
    models = (("perfect gate (k=16)", tsr.builder_for(a)(a, PAIR_SEED)(), tp),
              ("LOCAL3 (k=16)", tgw.to_local(tgw.build_plain("C", PAIR_SEED), PAIR_SEED), task))
    for name, m0, t in models:
        x = t.make_batch(16, torch.Generator().manual_seed(3))[0][:, :-1]
        m0.eval()
        with torch.no_grad():
            l0 = m0(x)[0]
        for impl in ("parallel", "fast", "hebb"):
            m = (ff.to_delta_fast(copy.deepcopy(m0), beta=0.0, write=1.0, normalize=False) if impl == "fast"
                 else dm.to_delta(copy.deepcopy(m0), beta=0.0, write=1.0, normalize=False, impl=impl))
            m.eval()
            with torch.no_grad():
                lg = m(x)[0]
            out.append(dict(name=name, impl=impl, d=float((lg - l0).abs().max()), equal=torch.equal(lg, l0)))
    return out


def fused_job():
    """CHECK 173: at beta 1 with L2 keys, the fused form (impl fast, the arms') against the parallel form, on the LOCAL3
    and the perfect-gate models: logits and every parameter gradient on 16 training sequences."""
    import copy
    import explore_delta_mem as dm
    task = tscur.TASK48
    a = tgw.CEIL_C_ARM
    tp = tsr.padded_task(tsr.TASKS[a["task"]], a["ctx_pad"])
    out = []
    for name, m0, t in (("LOCAL3 (k=16)", tgw.to_local(tgw.build_plain("C", PAIR_SEED), PAIR_SEED), task),
                        ("perfect gate (k=16)", tsr.builder_for(a)(a, PAIR_SEED)(), tp)):
        x = t.make_batch(16, torch.Generator().manual_seed(5))[0]
        res = []
        for conv in (lambda m: dm.to_delta(m, seed=PAIR_SEED, beta=BETA, impl="parallel"),
                     lambda m: ff.to_delta_fast(m, seed=PAIR_SEED, beta=BETA)):
            m = conv(copy.deepcopy(m0))
            m.train()
            m.zero_grad()
            ql, qt, _ = t.select(tbo.logits_of(m, x[:, :-1]), x[:, 1:], None)
            F.cross_entropy(ql, qt).backward()
            lg = tbo.logits_of(m, x[:, :-1]).detach()
            res.append((type(m.attn).__name__, lg, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
        (cp, lp, gp), (cf, lf, gf) = res
        rel = float((lp - lf).abs().max() / lp.abs().max())
        relg = max(float((gp[n] - gf[n]).abs().max() / (gp[n].abs().max() + 1e-12)) for n in gp)
        out.append(dict(name=name, cls=(cp, cf), rel=rel, relg=relg, same_set=set(gp) == set(gf), n=len(gp)))
    return out


def probe_job():
    """CHECK 163: the trigger's probe and the mass positions."""
    task = tscur.TASK48
    d = tgw.probe_data(task)
    q = task.stream_labels(d)[:, task.L - 2]
    counts = [int((q == s).sum()) for s in range(task.S)]
    j = torch.arange(task.S * task.P)
    pos = (3 * j + 1).tolist()
    keys_ok = bool(((d[:, :-1][:, pos] >= task.S) & (d[:, :-1][:, pos] < task.S + task.P)).all())
    return dict(n=int(d.shape[0]), counts=counts, pos=pos, keys_ok=keys_ok, qpos=task.L - 2,
                same=torch.equal(d, task.make_batch(64, torch.Generator().manual_seed(12345))[0]))


def pairing_job(key):
    """CHECK 166: the initial parameters and the first three training batches of a 3-update run at PAIR_SEED."""
    task = tscur.TASK48
    seen = []
    orig = task.make_batch

    def mb(B, gen):
        out = orig(B, gen)
        if B == tbo.BATCH and len(seen) < 3:
            seen.append(hashlib.sha1(out[0].numpy().tobytes()).hexdigest()[:16])
        return out
    task.make_batch = mb
    cap = {}
    try:
        one_run(key, PAIR_SEED, 3, capture=cap)
    finally:
        del task.make_batch
    init = {n: hashlib.sha1(p.numpy().tobytes()).hexdigest()[:16] for n, p in cap["init"].items()}
    out = dict(init=init, batches=seen, cls=cap["cls"])
    if key == "DELTA_SPLIT":
        m = run_learned_model(key)
        at = m.attn
        out["mem_ok"] = (type(at) is ff.FastDeltaMemory and at.beta_fixed == BETA and at.normalize and at.write == "tied"
                         and at.decay == 0.95 and at.decay_mod is None)
    if key == "CEIL_DELTA":
        a = tgw.CEIL_C_ARM
        tp = tsr.padded_task(tsr.TASKS[a["task"]], a["ctx_pad"])
        m = tsp.b_recipe(True)(dict(builder=tsr.builder_for(a)))["builder"](a, PAIR_SEED)()
        x = tp.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
        m.eval()
        with torch.no_grad():
            g = m(x)[2]
        out["onehot"] = bool((g.argmax(-1) == tp.stream_labels(x)).all()) and bool(((g == 0) | (g == 1)).all())
    return out


def referee_overlap():
    """docs/review_2026-10/recompute.py's overlap at RECOMPUTE_SHA, extracted from that commit's source and run."""
    import ast
    src = git_show(RECOMPUTE_SHA, "docs/review_2026-10/recompute.py")
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "overlap"]
    assert len(fn) == 1
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "recompute.py@" + RECOMPUTE_SHA, "exec"), ns)
    return ns["overlap"]


def run_learned_model(key):
    """The learned-gate arm's model as its path builds it (PAIR_SEED), through a 1-update run's keep."""
    keep = {"at": 1}
    run_learned(key, PAIR_SEED, 1, keep=keep)
    return keep["model"]


def lr_job(key):
    """CHECK 167: each group's lr at every update through the arm's real path (forward and evaluation stubbed), 2401
    updates; the groups' parameters."""
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
        rec = one_run(key, 3, tss.WARM + 1, keep=keep)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    names = {id(p): n for n, p in m.named_parameters()}
    flat = [i for g in ids["g"] for i in g]
    train = {id(p) for p in m.parameters() if p.requires_grad}
    wh = getattr(m, "W_h", None)
    return dict(ok=rec["ok"], n=len(lrs), groups=[[names[i] for i in g] for g in ids["g"]],
                at={str(u): lrs[u - 1] for u in (1, tss.WARM, tss.WARM + 1) if u <= len(lrs)},
                cover=len(flat) == len(set(flat)) and set(flat) == train,
                wh_in_group=wh is not None and id(wh) in set(flat), wh_frozen=wh is None or not wh.requires_grad)


# ── Verification ─────────────────────────────────────────────────────────────
def git_show(sha, path):
    return subprocess.run(["git", "-C", HERE, "show", f"{sha}:{path}"], capture_output=True, text=True, check=True).stdout


REFS = {H_KEY: "results/refs/H_copy2x2_COPY_NONOISE_487.json", F_KEY: "results/refs/F_eight_DELTA_SPLIT_1310.json"}


def record_at(sha, path, prefix):
    """Amendment 2 (h): the reference record from its vendored copy in results/refs/; where the source commit is in
    this clone, whether the copy equals the commit's record (None if the commit is absent)."""
    import json
    ref = json.load(open(os.path.join(HERE, REFS[prefix])))
    src = ref["source"]
    assert src["commit"].startswith(sha) and src["path"] == path and src["key"].split("|")[0] + "|" + src["key"].split("|")[1] == prefix
    try:
        runs = json.loads(git_show(sha, path))["runs"]
        same = runs.get(src["key"]) == ref["record"]
    except subprocess.CalledProcessError:
        same = None
    return src["key"], ref["record"], same


def d_seed(window, machine):
    """CHECK 158's seed: the first of the machine's Part D range on which the recorded WIN3_SPLIT_D8 fired by 7200."""
    for s in tgw.RANGES[machine]["C"]:
        r = get(window, "WIN3_SPLIT_D8", s)
        if ok_r(r) and any(c["fired"] and c["step"] <= D_THROUGH for c in r.get("checks", [])):
            return s
    return None


ROW_KEYS = ("step", "acc", "prev", "map_before", "key_cs", "key_c0", "fired", "eligible")
H_ROW_KEYS = ("step", "acc", "prev", "map_before", "key_cs", "key_c0", "fired", "map_after", "row_dist_before",
              "row_dist_after")


def rows_equal(a, b, keys):
    return len(a) == len(b) > 0 and all({k: x.get(k) for k in keys} == {k: y.get(k) for k in keys} for x, y in zip(a, b))


def first_diff(new, old):
    """The first evaluation step at which the two curves differ (None if equal over both)."""
    for x, y in zip(new["curve"], old["curve"]):
        if x != y:
            return x[0]
    return None


def earlier_seeds():
    return tsp.earlier_seeds() | {s for m in MACHINES for s in tsp.RANGES[m]}


def other_of(machine):
    return "L" if machine == "X" else "X"


def check_also(machine, also):
    m = also.get("meta", {})
    oth = other_of(machine)
    seeds = m.get("seeds") or {}
    in_range = all(s in RANGES[oth] for k in KEYS for s in seeds.get(k, []))
    mine = {s for k in KEYS for s in SEEDS[k]}
    theirs = {s for k in KEYS for s in seeds.get(k, [])}
    good = m.get("machine") == oth and bool(seeds) and in_range and m.get("cpu") != cpu_model() and not (mine & theirs)
    return good, (f"--also: machine {m.get('machine')} (should be {oth}); CPU {m.get('cpu')} (this machine {cpu_model()}; "
                  f"differs: {m.get('cpu') != cpu_model()}); its seeds {tsr.ranges(sorted(theirs)) if theirs else None} in "
                  f"{oth}'s range: {in_range}; overlap with this machine's: {sorted(mine & theirs) or 'none'}")


def verify(pool, files, machine, early, also, window, cached):
    f = {}
    ds = d_seed(window, machine)
    if ds is not None:
        f["158"] = pool.submit(check_job, "HEBB_SPLIT", ds, D_THROUGH)
    f["159"] = pool.submit(check_job, "HEBB_SPLIT", H_SEED, H_THROUGH)
    f["162"] = pool.submit(check_job, "DELTA_SPLIT", F_SEED, F_THROUGH)
    f["164", "HEBB_SPLIT"] = pool.submit(check_job, "HEBB_SPLIT", 1, 6000, 0.0)
    f["164", "DELTA_SPLIT"] = pool.submit(check_job, "DELTA_SPLIT", 1, 6000, 0.0)
    f["164", "DELTA_SLOW"] = pool.submit(check_job, "DELTA_SLOW", 1, 6000)
    f["164", "HEBB_SLOW"] = pool.submit(check_job, "HEBB_SLOW", 1, 6000)
    f["164", "WIN3_SLOW_D8"] = pool.submit(tsp.check_a, "WIN3_SLOW_D8", 1, 6000)
    f["160"] = pool.submit(copy_unit_job)
    f["165"] = pool.submit(trigger_copy_job)
    f["161"] = pool.submit(beta0_job)
    f["163"] = pool.submit(probe_job)
    f["173"] = pool.submit(fused_job)
    for k in KEYS:
        f["166", k] = pool.submit(pairing_job, k)
        f["167", k] = pool.submit(lr_job, k)
    f["169", "DELTA_SPLIT"] = pool.submit(check_job, "DELTA_SPLIT", 1, EVAL_EVERY)
    f["169", "HEBB_SPLIT"] = pool.submit(check_job, "HEBB_SPLIT", 1, EVAL_EVERY)
    for k, (s0, _) in cached.items():
        f["170", k] = pool.submit(check_job, k, s0, tss.WARM)

    print("test_split_copy.py's verification (which runs test_window_gate's, and so on down to test_multilayer_binding's):")
    tsp.SEEDS = tsp.seeds_for(machine)
    tsp.verify(pool, files, machine, early, None, window, {})
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    wm = window.get("meta", {})
    match = wm.get("cpu") == cpu_model()
    print(f"CHECK 158 HEBB_SPLIT at seed {ds} (the first of machine {machine}'s Part D range on which the recorded WIN3_SPLIT_D8 "
          f"fired by {D_THROUGH}; fixed: {D_SEED[machine]}) vs this machine's test_window_gate record ({files['window_path']}, "
          f"written on {wm.get('cpu')}): equal through the first split's check, then different (the exact copy vs the noisy "
          f"split with the reset); asserted on a matching CPU, this machine {cpu_model()}:")
    if ds is None or "158" not in f:
        print("     no seed  -> WRONG")
        g = False
    else:
        r = f["158"].result()["rec"]
        old = get(window, "WIN3_SPLIT_D8", ds)
        t1 = min(c["step"] for c in old["checks"] if c["fired"])
        c, g_, s_, shared = tgw.same_through(r, old, t1)
        rn = [x for x in r["checks"] if x["step"] <= t1]
        ro = [x for x in old["checks"] if x["step"] <= t1]
        rows = rows_equal(rn, ro, ROW_KEYS)
        fd = first_diff(r, old)
        op_new = [{k: x.get(k) for k in ("row_dist_after", "state_zeroed")} for x in rn if x["fired"]]
        op_old = [{k: x.get(k) for k in ("row_dist_after", "state_zeroed")} for x in ro if x["fired"]]
        g = c and g_ and s_ and rows and fd is not None and fd > t1 and r["ok"] and ds == D_SEED[machine]
        print(f"     through the first split ({t1}): curve equal {c}; gradient norms equal {g_}; statistics equal {s_} ({shared}); "
              f"the trigger's rows up to the firing equal {rows}")
        print(f"     the operation at {t1}: here {op_new} vs recorded {op_old}; first differing evaluation {fd} "
              f"(curve here {r['curve']} vs {[x for x in old['curve'] if x[0] <= D_THROUGH]})  -> {'AS STATED' if g else 'WRONG'}")
    if match:
        ok &= g
    else:
        print("     (the record's CPU differs from this machine's: printed, not asserted)")
    print()

    k_h, rec_h, same_h = record_at(H_SHA, H_FILE, H_KEY)
    match = rec_h.get("cpu") == cpu_model()
    r = f["159"].result()["rec"]
    c, g_, s_, shared = tgw.same_through(r, rec_h, H_THROUGH)
    rows = rows_equal([x for x in r["checks"] if x["step"] <= H_THROUGH], [x for x in rec_h["checks"] if x["step"] <= H_THROUGH],
                      H_ROW_KEYS)
    g = c and g_ and s_ and rows and r["ok"]
    print(f"CHECK 159 HEBB_SPLIT|{H_SEED} vs H's {k_h} ({H_FILE} at {H_SHA}, written on {rec_h.get('cpu')}) through {H_THROUGH} "
          f"(asserted on a matching CPU; this machine {cpu_model()}); the record from {REFS[H_KEY]}, equal to the commit's "
          f"{'(commit not in this clone)' if same_h is None else same_h}:")
    print(f"     curve {r['curve']} vs {[x for x in rec_h['curve'] if x[0] <= H_THROUGH]}; curve equal {c}; gradient norms equal "
          f"{g_}; statistics equal {s_} ({shared}); trigger rows equal {rows} (fired at "
          f"{[x['step'] for x in r['checks'] if x['fired']]})  -> {'IDENTICAL' if g else 'DIFFERS'}")
    if same_h is False:
        ok = False
        print("     the vendored record differs from the commit's  -> WRONG")
    if match:
        ok &= g
    else:
        print("     (H's record was written on another CPU: printed, not asserted)")
    print()

    r = f["160"].result()
    g = r["copied"] and r["cs_same"] and r["others"] and r["state"] and r["res"]["row_dist_after"] == 0.0
    print(f"CHECK 160 the operation (Part C's LOCAL3 model, seed {tgw.D8_SEED}, after one Adam step; c* = 3, c0 = 7): W_g[c0] = "
          f"W_g[c*] exactly {r['copied']}; c*'s row unchanged {r['cs_same']}; every other row unchanged {r['others']}; W_g's "
          f"Adam state untouched {r['state']}; rows apart before {r['res']['row_dist_before']:.4f}, after "
          f"{r['res']['row_dist_after']}  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print("CHECK 161 the delta memory at beta 0 (erase off, write 1, raw keys) vs the Hebbian memory, max |logits diff| on 16 "
          "sequences (tolerance: see the specification's amendment):")
    good = True
    for x in f["161"].result():
        tol = BETA0_TOL[x["name"]]
        g = x["equal"] if x["impl"] == "hebb" else x["d"] <= tol
        good &= g
        print(f"     {x['name']:<20} impl {x['impl']:<8}: {x['d']:.2e} (tolerance {'bit-identical' if x['impl'] == 'hebb' else f'{tol:g}'}); "
              f"bit-identical {x['equal']}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    k_f, rec_f, same_f = record_at(F_SHA, F_FILE, F_KEY)
    match = rec_f.get("cpu") == cpu_model()
    r = f["162"].result()["rec"]
    c, g_, s_, shared = tgw.same_through(r, rec_f, F_THROUGH)
    g = c and g_ and s_ and r["ok"]
    print(f"CHECK 162 DELTA_SPLIT|{F_SEED} vs F's {k_f} ({F_FILE} at {F_SHA}, written on {rec_f.get('cpu')}) through {F_THROUGH} "
          f"(asserted on a matching CPU; this machine {cpu_model()}); the record from {REFS[F_KEY]}, equal to the commit's "
          f"{'(commit not in this clone)' if same_f is None else same_f}:")
    print(f"     curve {r['curve']} vs {[x for x in rec_f['curve'] if x[0] <= F_THROUGH]}; curve equal {c}; gradient norms "
          f"equal {g_}; statistics equal {s_} ({shared})  -> {'IDENTICAL' if g else 'DIFFERS'}")
    if same_f is False:
        ok = False
        print("     the vendored record differs from the commit's  -> WRONG")
    if match:
        ok &= g
    else:
        print("     (F's record was written on another CPU: printed, not asserted)")
    print()

    r = f["163"].result()
    g = r["n"] == 64 and tuple(r["counts"]) == PROBE_COUNTS and r["keys_ok"] and r["same"] and r["pos"] == [3 * j + 1 for j in range(32)]
    print(f"CHECK 163 the trigger's probe: {r['n']} sequences from Generator(12345) (test_window_gate.probe_data; equal to a fresh "
          f"draw {r['same']}); queries per stream (streams 0-7) {r['counts']} (fixed: {list(PROBE_COUNTS)}); the mass positions "
          f"3j+1, j = 0..31 ({r['pos'][0]}..{r['pos'][-1]}), all key tokens {r['keys_ok']}; the query's key at {r['qpos']} not "
          f"among them  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print("CHECK 164 with the threshold at 0 (never eligible) both split arms equal their no-trigger forms (seed 1) bit for bit "
          "through 6000:")
    good = True
    for k, base_k in (("HEBB_SPLIT", "HEBB_SLOW"), ("HEBB_SLOW", "WIN3_SLOW_D8"), ("DELTA_SPLIT", "DELTA_SLOW")):
        r = f["164", k].result()["rec"]
        base = f["164", base_k].result()["rec"]
        same = subset_equal(trunc(r, 6000), trunc(base, 6000)) and subset_equal(trunc(base, 6000), trunc(r, 6000))
        trig = ARM[k]["trigger"]
        g = same and r["ok"] and (not trig or (not any(x["fired"] for x in r["checks"]) and len(r["checks"]) == 2))
        good &= g
        print(f"     {k} vs {base_k}: curve, statistics and gradient norms equal {same}"
              + (f"; trigger checks at {[x['step'] for x in r['checks']]}, none fired" if trig else "")
              + f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    r = f["165"].result()
    unchanged = all(r["before"].get(k) == v for k, v in r["after"].items()) and any(v > 0 for k, v in r["before"].items()
                                                                                     if k != "step")
    g = (r["fired"] == [4800, 9600, 14400] and r["gap"] == [7200, 12000] and r["cap"] == [16800, 19200, 21600, 24000, 26400]
         and all(d == 0.0 for d in r["dist"]) and unchanged and not r["wg_same"])
    print(f"CHECK 165 the trigger with the exact copy (CHECK 142's harness: Part C's LOCAL3 model, seed {tgw.D8_SEED}, after one "
          f"Adam step; checks 2400..26400, threshold above 1): fired at {r['fired']}, blocked by the gap at {r['gap']}, by the "
          f"cap at {r['cap']}; rows c*, c0 apart after each firing {r['dist']}; W_g's Adam state unchanged {unchanged}; W_g "
          f"changed {not r['wg_same']}  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print(f"CHECK 166 the pairing at seed {PAIR_SEED}: initial parameters and the first three training batches:")
    pr = {k: f["166", k].result() for k in KEYS}
    same_init = pr["HEBB_SPLIT"]["init"] == pr["DELTA_SPLIT"]["init"] == pr["DELTA_SLOW"]["init"]
    same_ceil = pr["CEIL_HEBB"]["init"] == pr["CEIL_DELTA"]["init"]
    same_b = all(len(pr[k]["batches"]) == 3 for k in KEYS) and len({tuple(pr[k]["batches"]) for k in KEYS}) == 1
    g = (same_init and same_ceil and same_b and pr["DELTA_SPLIT"].get("mem_ok") and pr["CEIL_DELTA"].get("onehot"))
    for k in KEYS:
        print(f"     {k:<12} {pr[k]['cls'][0]} / {pr[k]['cls'][1]}: {len(pr[k]['init'])} parameter tensors; batches "
              f"{pr[k]['batches']}")
    print(f"     learned-gate arms' initial parameters equal {same_init}; CEIL_DELTA's equal CEIL_HEBB's {same_ceil}; the first "
          f"three training batches equal across all {len(KEYS)} arms {same_b}; DELTA's memory FastDeltaMemory beta {BETA:g}, L2 keys, "
          f"tied write, decay 0.95 {pr['DELTA_SPLIT'].get('mem_ok')}; CEIL_DELTA's gate one-hot on each position's stream "
          f"{pr['CEIL_DELTA'].get('onehot')}  -> {'OK' if g else 'WRONG'}")
    ok &= bool(g)
    print()

    print(f"CHECK 167 groups and lrs at updates 1, {tss.WARM}, {tss.WARM + 1} through the real paths (forward stubbed):")
    good = True
    for k in KEYS:
        r = f["167", k].result()
        if ARM[k]["ceil"]:
            g = len(r["groups"]) == 1 and all(r["at"][str(u)] == [tgw.LR] for u in (1, tss.WARM, tss.WARM + 1))
        else:
            g = (len(r["groups"]) == 2 and r["groups"][0] == list(tgw.GATE_L) and r["at"]["1"] == [tgw.LR, tgw.LR_WARM]
                 and r["at"][str(tss.WARM)] == [tgw.LR, tgw.LR_WARM] and r["at"][str(tss.WARM + 1)] == [tgw.LR, tgw.LR]
                 and not r["wh_in_group"] and r["wh_frozen"])
        g = g and r["cover"] and r["n"] == tss.WARM + 1 and r["ok"]
        good &= g
        print(f"     {k:<12} groups {[len(x) for x in r['groups']]} (first {r['groups'][0] if not ARM[k]['ceil'] else 'all'}); every "
              f"trainable parameter once {r['cover']}; lrs at 1 {r['at'].get('1')}, {tss.WARM} {r['at'].get(str(tss.WARM))}, "
              f"{tss.WARM + 1} {r['at'].get(str(tss.WARM + 1))}; W_h in a group {r['wh_in_group']}, frozen {r['wh_frozen']}  -> "
              f"{'OK' if g else 'WRONG'}")
    ok &= good
    print()

    ref = referee_overlap()
    recs = [r for s_, r in sorted(tss.runs_of(window, "WIN3_SPLIT_D8", (wm.get("seeds") or {}).get("WIN3_SPLIT_D8", [])).items())
            if ok_r(r) and (r.get("end") or {}).get("stream_gate")]
    eq = [overlap(r) == ref(r) for r in recs]
    g = bool(recs) and all(eq)
    print(f"CHECK 172 the copied overlap equals the referee's (docs/review_2026-10/recompute.py at {RECOMPUTE_SHA}, extracted "
          f"and run) on every WIN3_SPLIT_D8 record of {files['window_path']}: {sum(eq)}/{len(recs)} equal; largest "
          f"{max((overlap(r) for r in recs), default=None)}  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print("CHECK 173 the arms' memory (beta 1, L2 keys): the fused form vs the parallel form, logits to 1e-5 relative and every "
          "parameter gradient to 1e-4 relative (16 training sequences, seed 580; asserted on every machine):")
    good = True
    for x in f["173"].result():
        g = x["rel"] <= 1e-5 and x["relg"] <= 1e-4 and x["same_set"] and x["cls"] == ("DeltaMemory", "FastDeltaMemory")
        good &= g
        print(f"     {x['name']:<20} {x['cls'][1]} vs {x['cls'][0]}: logits {x['rel']:.1e}, gradients {x['relg']:.1e} over "
              f"{x['n']} tensors  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 168 the seeds (machine {machine}):")
    rg = tuple(RANGES[machine])
    arms_ok = all(SEEDS[k] == (rg[:VALID_N] if ARM[k]["ceil"] else rg) for k in KEYS)
    mine = {s for k in KEYS for s in SEEDS[k]}
    other = set(RANGES[other_of(machine)])
    earlier = earlier_seeds()
    g = arms_ok and not (mine & earlier) and not (mine & other)
    print(f"     this machine's seeds {tsr.ranges(sorted(mine))}: every arm on the range, the perfect gates on its first "
          f"{VALID_N} {arms_ok}; disjoint from the other machine's range {not (mine & other)} and from every earlier "
          f"main-line seed {not (mine & earlier)} (earlier: {tsr.ranges(sorted(earlier))})")
    if also is not None:
        ga, line = check_also(machine, also)
        print(f"     {line}  -> {'OK' if ga else 'WRONG'}")
        g &= ga
    else:
        print("     no --also file")
    ok &= g
    print()

    print("CHECK 169 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("DELTA_SPLIT", "HEBB_SPLIT"):
        rw = f["169", key].result()["rec"]
        rh = check_job(key, 1, EVAL_EVERY)["rec"]
        same = subset_equal(strip_all(rw), strip_all(rh)) and subset_equal(strip_all(rh), strip_all(rw))
        g = same and rh["ok"]
        good &= g
        print(f"         {key} seed 1, {EVAL_EVERY} steps: records equal {same}   acc {rh['curve'][-1][1]}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 170 (a resume) records cached by an earlier start reproduce under this code through {tss.WARM}:")
    good = True
    if not cached:
        print("     no cached record (a first start)")
    for k, (s0, old) in cached.items():
        r = f["170", k].result()["rec"]
        g = subset_equal(trunc(r, tss.WARM), trunc(old, tss.WARM)) and subset_equal(trunc(old, tss.WARM), trunc(r, tss.WARM))
        good &= g
        print(f"     {k}|{s0}: curve {r['curve']} vs {[x for x in old['curve'] if x[0] <= tss.WARM]}; equal {g}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


BETA0_TOL = {"perfect gate (k=16)": 1e-7, "LOCAL3 (k=16)": 1e-6}   # Amendment 1 (Referee V: 7x the measured 1.49e-7)


# ── Reporting ────────────────────────────────────────────────────────────────
def paired(new, old, seeds, fn_new, fn_old):
    both = [s for s in seeds if ok_r(new.get(s)) and ok_r(old.get(s))]
    b = sum(1 for s in both if fn_new(new[s]) and not fn_old(old[s]))
    c = sum(1 for s in both if fn_old(old[s]) and not fn_new(new[s]))
    return dict(n=len(both), new=sum(bool(fn_new(new[s])) for s in both), old=sum(bool(fn_old(old[s])) for s in both),
                b=b, c=c, p=mcnemar_greater(b, c))


def sign_test(pos, neg):
    """Exact one-sided sign test: P(X >= pos), X ~ Bin(pos + neg, 1/2)."""
    n = pos + neg
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n


def d3_rows(rA, seeds):
    """Seeds where both DELTA_SPLIT and HEBB_SPLIT are BOUND ROUTED: (seed, hebb transition, delta transition)."""
    out = []
    for s in seeds:
        h, d = rA["HEBB_SPLIT"].get(s), rA["DELTA_SPLIT"].get(s)
        if success("HEBB_SPLIT", h) and success("DELTA_SPLIT", d):
            out.append((s, h["transition"], d["transition"]))
    return out


def d3_of(rows):
    pos = sum(1 for _, h, d in rows if d < h)
    neg = sum(1 for _, h, d in rows if d > h)
    return dict(n=len(rows), pos=pos, neg=neg, ties=len(rows) - pos - neg, p=sign_test(pos, neg),
                med_h=statistics.median([h for _, h, _ in rows]) if rows else None,
                med_d=statistics.median([d for _, _, d in rows]) if rows else None)


def validity(store, seeds):
    out = {}
    for part, k in CEILS:
        n = sum(bound_r(get(store, k, s)) for s in seeds[k])
        out[part] = (n, n >= VALID_N)
    return out


def tally(store, seeds):
    seeds = {k: tuple(seeds.get(k, ())) for k in KEYS}
    rA = {k: tss.runs_of(store, k, seeds[k]) for k in KEYS}
    valid = validity(store, seeds)
    sv = lambda parts: all(valid[p][1] for p in parts)
    out = dict(rA=rA, valid=valid, seeds=seeds, bands={})
    out["D1"] = dict(paired_full(rA["DELTA_SPLIT"], rA["HEBB_SPLIT"], seeds["DELTA_SPLIT"], lambda r: success("DELTA_SPLIT", r),
                            lambda r: success("HEBB_SPLIT", r)), valid=sv(PARTS["D1"]))
    drop2 = broken_pairs(rA, seeds, "DELTA_SPLIT", "DELTA_SLOW")          # Amendment 2 (d)
    drop4 = broken_pairs(rA, seeds, "HEBB_SPLIT", "HEBB_SLOW")
    out["D2"] = dict(paired(rA["DELTA_SPLIT"], rA["DELTA_SLOW"], [s for s in seeds["DELTA_SPLIT"] if s not in drop2],
                            lambda r: success("DELTA_SPLIT", r), lambda r: success("DELTA_SLOW", r)),
                     valid=sv(PARTS["D2"]), dropped=drop2)
    out["D4"] = dict(paired(rA["HEBB_SPLIT"], rA["HEBB_SLOW"], [s for s in seeds["HEBB_SPLIT"] if s not in drop4],
                            lambda r: success("HEBB_SPLIT", r), lambda r: success("HEBB_SLOW", r)),
                     valid=sv(PARTS["D4"]), dropped=drop4)
    out["D3rows"] = d3_rows(rA, seeds["DELTA_SPLIT"])
    out["D3"] = dict(d3_of(out["D3rows"]), valid=sv(PARTS["D3"]))
    for k in ARMS_RUN:
        out["bands"][k] = dict(c=sum(success(k, r) for r in rA[k].values()), n=len(seeds[k]), valid=valid[ARM[k]["part"]][1])
    return out


def paired_full(new, old, seeds, fn_new, fn_old):
    """paired() plus the concordant counts (for the paired-difference interval)."""
    d = paired(new, old, seeds, fn_new, fn_old)
    both = [s for s in seeds if ok_r(new.get(s)) and ok_r(old.get(s))]
    d["a"] = sum(1 for s in both if fn_new(new[s]) and fn_old(old[s]))
    d["dd"] = d["n"] - d["a"] - d["b"] - d["c"]
    return d


def newcombe_paired(a, b, c, d, z=1.959963984540054):
    """The difference of paired proportions (b - c) / n with Newcombe's hybrid score interval (1998, method 10):
    Wilson limits for each margin, combined with the continuity-corrected phi."""
    n = a + b + c + d
    if n == 0:
        return None
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = tsr.wilson(a + b, n, z)
    l2, u2 = tsr.wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0) if num > 0 else num) / math.sqrt(den)
    th = p1 - p2
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    eu = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return th, max(-1.0, th - dl), min(1.0, th + eu)


def d1_label(d):
    """Amendment 1: d = HEBB only - DELTA only."""
    if not d["valid"] or d["n"] == 0:
        return "UNTESTED"
    dd = d["c"] - d["b"]
    return D1_LABELS[0] if dd <= -3 else (D1_LABELS[1] if dd <= D1_BOUND else D1_LABELS[2])


def split_verdict(d):
    return "UNTESTED" if not d["valid"] or d["n"] == 0 else ("SHOWN" if d["p"] < 0.05 else "NOT SHOWN")


def split_label(d, slow_count):
    """Per write: helps / no room / not shown (Amendment 1); None if UNTESTED."""
    v = split_verdict(d)
    if v == "UNTESTED":
        return None
    if slow_count >= NO_ROOM:
        return "no room"
    return "helps" if v == "SHOWN" else "not shown"


def split_reading(ld, lh):
    """Amendment 2 (e): if either write is "no room", both labels are printed and no combination is named."""
    if ld is None or lh is None:
        return None
    return SPLIT_READINGS.get((ld, lh)) or f"delta: {ld}; Hebbian: {lh}"


def pool2(ds):
    b, c = sum(d["b"] for d in ds), sum(d["c"] for d in ds)
    out = dict(n=sum(d["n"] for d in ds), new=sum(d["new"] for d in ds), old=sum(d["old"] for d in ds), b=b, c=c,
               p=mcnemar_greater(b, c), valid=all(d["valid"] for d in ds))
    if all("a" in d for d in ds):
        out.update(a=sum(d["a"] for d in ds), dd=sum(d["dd"] for d in ds))
    if all("dropped" in d for d in ds):
        out["dropped"] = [s for d in ds for s in d["dropped"]]
    return out


def band_str(b):
    lo, hi = tsr.wilson(b["c"], b["n"])
    band = "UNTESTED" if not b["valid"] else tscur.band_n(b["c"], b["n"])
    return (f"{b['c']}/{b['n']} -> {band}   Wilson 95% [{lo:.3f}, {hi:.3f}]  (RELIABLE >= {math.ceil(0.9 * b['n'])}, "
            f"MAJORITY >= {math.ceil(0.5 * b['n'])})")


def diff_str(d):
    nc = newcombe_paired(d["a"], d["b"], d["c"], d["dd"])
    return "--" if nc is None else (f"{nc[0]:+.3f}, 95% [{nc[1]:+.3f}, {nc[2]:+.3f}] (the hybrid score interval with "
                                    f"continuity-corrected phi; method 10 as Fagerland, Lydersen and Laake 2014 give it)")


def print_claims(t, who, pooled):
    nv = lambda d: "" if d["valid"] else f"   (a part it uses is not valid{who})"
    tag = " (pooled)" if pooled else ""
    d = t["D1"]
    print(f"  D1  DELTA_SPLIT vs HEBB_SPLIT (BOUND ROUTED): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; DELTA_SPLIT only {d['b']}, "
          f"HEBB_SPLIT only {d['c']}; d = HEBB only - DELTA only = {d['c'] - d['b']}" + nv(d))
    print(f"      rate difference DELTA_SPLIT - HEBB_SPLIT {diff_str(d)}")
    if pooled:
        print(f"     *** D1{tag}: {d1_label(d)} ***   ({D1_OC})")
    else:
        print(f"     (per machine: the rate difference only, no label)")
    for name, slow in (("D2", "DELTA_SLOW"), ("D4", "HEBB_SLOW")):
        d = t[name]
        new = "DELTA_SPLIT" if name == "D2" else "HEBB_SPLIT"
        drop = d.get("dropped") or []
        print(f"  {name}  {new} beats {slow} (BOUND ROUTED): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, "
              f"{slow} only {d['c']}; p = {d['p']:.4g}" + nv(d)
              + (f"   ({len(drop)} pairs dropped by CHECK 174: {' '.join('s' + str(x) for x in drop)})" if drop else ""))
        room = (f"; {slow} {d['old']}/{d['n']} >= {NO_ROOM}: NO ROOM (printed, not read)" if pooled and d["old"] >= NO_ROOM else "")
        print(f"     *** {name}{tag}: {split_verdict(d)}{room} ***")
    d = t["D3"]
    print(f"  D3  (secondary) on the {d['n']} seeds both split arms are BOUND ROUTED: median transition DELTA_SPLIT {d['med_d']}, "
          f"HEBB_SPLIT {d['med_h']}; DELTA earlier on {d['pos']}, later on {d['neg']}, tied {d['ties']}; one-sided sign test "
          f"p = {d['p']:.4g} (DELTA earlier)" + nv(d))


def broken_pairs(rA, seeds, sk, nk):
    """CHECK 174 for one pair of arms: the seeds whose split arm's curve differs from its no-trigger arm's through the
    split arm's first firing (the whole run if it never fired). Amendment 2 (d): these pairs are dropped."""
    bad = []
    for s in seeds[sk]:
        a, b = rA[sk].get(s), rA[nk].get(s)
        if not (ok_r(a) and ok_r(b)):
            continue
        fs = [c["step"] for c in a.get("checks", []) if c["fired"]]
        if fs:
            same = [x for x in a["curve"] if x[0] <= fs[0]] == [x for x in b["curve"] if x[0] <= fs[0]]
        else:
            same = a["curve"] == b["curve"]
        if not same:
            bad.append(s)
    return bad


def check_174(t):
    n = sum(1 for sk, nk in (("DELTA_SPLIT", "DELTA_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW")) for s in t["seeds"][sk]
            if ok_r(t["rA"][sk].get(s)) and ok_r(t["rA"][nk].get(s)))
    bad = [f"{sk}|{s}" for sk, nk in (("DELTA_SPLIT", "DELTA_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW"))
           for s in broken_pairs(t["rA"], t["seeds"], sk, nk)]
    return n, bad


def probe_blind(t, k):
    """Amendment 2 (f): the trigger checks at update >= 4800 (seed, step, probe acc, held-out acc at that update); the
    count where the probe read >= 0.95 while the held-out accuracy was < 0.95; and the count blocked by the probe alone
    (a previous reading, rise < 0.02, under the cap and the gap, probe >= 0.95, not fired)."""
    rows, blind, alone = [], 0, 0
    for s, r in sorted(t["rA"][k].items()):
        if not ok_r(r):
            continue
        cur = {x[0]: x[1] for x in r["curve"]}
        n, last = 0, None
        for c in r.get("checks", []):
            if c["step"] >= tgw.FIRST:
                h = cur.get(c["step"])
                rows.append((s, c["step"], c["acc"], h))
                if h is not None and c["acc"] >= tgw.ACC_THR and h < tgw.ACC_THR:
                    blind += 1
                room = n < tgw.MAX_SPLITS and (last is None or c["step"] - last >= tgw.GAP)
                if (c["prev"] is not None and c["acc"] - c["prev"] < tgw.RISE and room and c["acc"] >= tgw.ACC_THR
                        and not c["fired"]):
                    alone += 1
            if c["fired"]:
                n, last = n + 1, c["step"]
    return rows, blind, alone


def print_bands(bands, label):
    for k in ARMS_RUN:
        print(f"  band {k:<12} {label}(BOUND ROUTED): {band_str(bands[k])}")


def complete(store):
    seeds = store.get("meta", {}).get("seeds") or {}
    missing = [f"{k}|{s}" for k in KEYS for s in seeds.get(k, []) if not ok_r(get(store, k, s))]
    return bool(seeds) and set(seeds) == set(KEYS) and not missing, missing


def stream_str(r):
    sa = (r.get("end") or {}).get("stream_acc")
    return "/".join(f"{v:.2f}" for v in sa) if sa else "--"


def raw_rows(k, runs, seeds):
    if ARM[k]["ceil"]:
        print(f"  {'arm':<12} {'seed':>4} {'acc':>7} {'transition':>10}  {'per-stream acc (end)':<39} outcome")
    else:
        print(f"  {'arm':<12} {'seed':>4} {'acc':>7} {'transition':>10} {'map':>23} {'1:1':>3} {'sh':>2}  {'per-stream acc':<39} "
              f"{'ROUTED':>6} {'overlap':>7}  {'fired (ok?)':<26} outcome (v2)")
    for s in seeds:
        r = runs.get(s)
        if not ok_r(r):
            print(f"  {k:<12} {s:>4}  {'NOT RUN' if r is None else 'FAILED — ' + str(r.get('error'))}")
            continue
        if ARM[k]["ceil"]:
            print(f"  {k:<12} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10}  {stream_str(r):<39} "
                  f"{'BOUND' if bound_r(r) else 'not bound'}")
            continue
        e = r["end"]
        fired = ("--" if r.get("checks") is None else
                 " ".join(f"{c['step']}{'' if c['key_ok'] else '!'}" for c in r["checks"] if c["fired"]) or "none")
        ov = overlap(r)
        print(f"  {k:<12} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {mstr(e['ch_map']):>23} "
              f"{'y' if e['one_to_one'] else 'n':>3} {shared_max(r):>2}  {stream_str(r):<39} {'Y' if tsr.routed(r) else '-':>6} "
              f"{(f'{ov:.3f}' if ov is not None else '--'):>7}  {fired:<26} {outcome2(r)}")
    print()


def report(store, seeds, machine, wall, path, also):
    t = tally(store, seeds)
    rA, valid = t["rA"], t["valid"]
    print()
    print("#" * 100)
    print(f"PER-SEED RAW RESULTS — every value, before any aggregate (machine {machine}, {store.get('meta', {}).get('cpu')})")
    print("#" * 100)
    print("  (map = stream -> channel argmax at the end; 1:1 = one-to-one; sh = most streams on one channel; ROUTED = one-to-one "
          "and every stream >= 0.9; overlap = the largest pairwise sum_c min(g_i[c], g_j[c]) of the value-position gates; fired = "
          "the firings' updates, '!' = target not labelled ok; outcome = fail_class_v2.outcome_v2)")
    print()
    for k in KEYS:
        print(f"  {ARM[k]['label']}   (outcome {'perfect gate: BOUND' if ARM[k]['ceil'] else 'BOUND ROUTED'})")
        raw_rows(k, rA[k], t["seeds"][k])

    print("=" * 100)
    print("VALIDITY (each part's perfect gate binds both of its seeds)")
    print("=" * 100)
    for part, k in CEILS:
        print(f"  {part}: {k} bound {valid[part][0]}/{len(t['seeds'][k])} "
              f"({', '.join(f's{s} ' + (('BOUND at ' + str(r['transition'])) if bound_r(r) else 'not bound') for s, r in sorted(rA[k].items()))})"
              f"  -> {'VALID' if valid[part][1] else 'NOT VALID: the claims using ' + part + ' are UNTESTED here'}")
    print()

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in ARMS_RUN:
        done = [r for r in rA[k].values() if ok_r(r)]
        trs_ = [r["transition"] for r in done if success(k, r)]
        strict = sum(1 for r in done if success(k, r) and (overlap(r) or 0) <= OVERLAP_MAX)
        merged = sum(1 for r in done if outcome2(r).startswith("MERGED"))
        print(f"  {ARM[k]['label']:<74} BOUND ROUTED {len(trs_):>2}/{len(t['seeds'][k])}   overlap <= {OVERLAP_MAX} "
              f"{strict:>2}   MERGED {merged:>2}   completed {len(done)}/{len(t['seeds'][k])}   transition "
              f"{med_int(trs_) if trs_ else '--'}")
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
    print("PRE-REGISTERED CLAIMS (Amendments 1-2: D1 labelled by d with a paired interval; D2, D4 exact one-sided McNemar, SHOWN "
          "if p < 0.05, NO ROOM if the no-trigger arm binds >= 36/40 pooled; D3 secondary, printed)")
    print("#" * 100)
    to = None
    if other is not None:
        om = other.get("meta", {})
        to = tally(other, {k: tuple(v) for k, v in om["seeds"].items()})
        both = {machine: t, other_of(machine): to}
        print(f"POOLED (PRIMARY): machine {machine} ({store.get('meta', {}).get('cpu')}) + machine {other_of(machine)} "
              f"({om.get('cpu')}, {also}); b and c summed over both machines' pairs")
        pc = {n: pool2([both[m_][n] for m_ in MACHINES]) for n in ("D1", "D2", "D4")}
        rows = [x for m_ in MACHINES for x in both[m_]["D3rows"]]
        pc["D3"] = dict(d3_of(rows), valid=all(both[m_]["D3"]["valid"] for m_ in MACHINES))
        print("  (per machine: " + "; ".join(f"{n} " + ", ".join(f"{m_} {both[m_][n]['b']} vs {both[m_][n]['c']}" for m_ in MACHINES)
                                             for n in ("D1", "D2", "D4")) + ")")
        print_claims(pc, " on a machine", True)
        pb = {k: dict(c=sum(both[m_]["bands"][k]["c"] for m_ in MACHINES), n=sum(both[m_]["bands"][k]["n"] for m_ in MACHINES),
                      valid=all(both[m_]["bands"][k]["valid"] for m_ in MACHINES)) for k in ARMS_RUN}
        print_bands(pb, "pooled ")
        l1 = d1_label(pc["D1"])
        ld = split_label(pc["D2"], pc["D2"]["old"])
        lh = split_label(pc["D4"], pc["D4"]["old"])
        sr = split_reading(ld, lh)
        print("  READINGS (pooled; Amendments 1-2):")
        print(f"     the memory rule: " + (f"\"{l1}\"" if l1 != "UNTESTED" else "UNTESTED, no reading"))
        cnt = lambda d, slow: f"{slow} {d['old']}/{d['n']}, b {d['b']}, c {d['c']}"
        print(f"     the split: delta {ld or 'UNTESTED'} ({cnt(pc['D2'], 'DELTA_SLOW')}), Hebbian {lh or 'UNTESTED'} "
              f"({cnt(pc['D4'], 'HEBB_SLOW')}): " + (f"\"{sr}\"" if sr else "no reading"))
        print()
    else:
        print("POOLED (PRIMARY): not computed here — it needs --also with the other machine's complete file")
        print()
    print(f"PER MACHINE (SECONDARY): machine {machine}")
    print_claims(t, " on this machine", False)
    print_bands(t["bands"], "")
    print("  (the readings are pooled only)")
    print()
    if to is not None:
        print(f"  Machine {other_of(machine)} (from --also):")
        print_claims(to, f" on {other_of(machine)}", False)
        print_bands(to["bands"], "")
        print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  outcomes (fail_class_v2.outcome_v2):")
    for k in ARMS_RUN:
        cls = {}
        for s, r in sorted(rA[k].items()):
            if ok_r(r):
                cls.setdefault(outcome2(r) + ("  collapsed" if r.get("collapsed") else ""), []).append(s)
        print(f"    {k:<12} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<34} {' '.join('s' + str(x) for x in v)}")
    rows = [(k, s, tsr.outcome(r), outcome2(r), (r.get("end") or {}).get("etak_key_by_key"), (r.get("end") or {}).get("margin"))
            for k in ARMS_RUN for s, r in sorted(rA[k].items()) if ok_r(r)]
    diff = [x for x in rows if x[2] != x[3]]
    print(f"  CHECK 171 fail_class_v2 vs v1 on every record of this test: {len(rows)} records, disagreements {len(diff)}  -> "
          f"{'PASS' if not diff else 'DISAGREE (printed)'}")
    for k, s, a1, a2, eta, mg in diff:
        print(f"     {k}|{s}: v1 {a1}, v2 {a2}; eta^2 by key (multichannel) {eta}, margin {mg}")
    n174, bad = check_174(t)
    print(f"  CHECK 174 each split arm's curve equals its no-trigger arm's on the same seed through the first firing: {n174} pairs, "
          f"{'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")
    for k in ("HEBB_SPLIT", "DELTA_SPLIT"):
        rows_pb, blind, alone = probe_blind(t, k)
        print(f"  probe-blind checks, {k} (checks at update >= {tgw.FIRST}): {blind} of {len(rows_pb)} read >= {tgw.ACC_THR} on the "
              f"probe while the held-out accuracy was < {tgw.ACC_THR}; blocked by the probe alone {alone}")
        print("    (seed@update probe/held-out) " + " ".join(f"s{s_}@{st}:{pa:.2f}/{('%.2f' % h) if h is not None else '--'}"
                                                         for s_, st, pa, h in rows_pb))
    print()
    print("  per seed: HEBB_SPLIT | HEBB_SLOW | DELTA_SPLIT | DELTA_SLOW (transition):")

    def oc(r):
        return "--" if not ok_r(r) else f"{outcome2(r)} ({fmt_step(r['transition'])})"
    for s in t["seeds"]["DELTA_SPLIT"]:
        print(f"    s{s}  " + " | ".join(f"{oc(rA[k].get(s)):<30}" for k in ("HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW")))
    print()
    print(f"  D3 rows (seed: HEBB_SPLIT transition, DELTA_SPLIT transition): "
          + ", ".join(f"s{s} {h}/{d}" for s, h, d in t["D3rows"]))
    print()
    for k in ("HEBB_SPLIT", "DELTA_SPLIT"):
        rows_all = [(s, r) for s, r in sorted(rA[k].items()) if ok_r(r)]
        fired = [c for _, r in rows_all for c in r.get("checks", []) if c["fired"]]
        print(f"  {k}: {len(fired)} firings in {len(rows_all)} runs (per run "
              f"{[sum(1 for c in r.get('checks', []) if c['fired']) for _, r in rows_all]}); targets labelled ok "
              f"{sum(1 for c in fired if c['key_ok'])}/{len(fired)}")
        for s, r in rows_all:
            ch = r.get("checks", [])
            ev = [f"{c['step']}: {c['key_cs']}->{c['key_c0']} {'ok' if c['key_ok'] else 'NOT labelled'} "
                  f"(map {groups_str(c['map_before'])} -> {groups_str(c.get('map_after'))})" for c in ch if c["fired"]]
            bl = [f"{c['step']} ({c.get('blocked_by')})" for c in ch if c["blocked"]]
            print(f"    s{s}  {outcome2(r)} (transition {fmt_step(r['transition'])}); fired: {'; '.join(ev) or 'none'}; "
                  f"blocked: {', '.join(bl) or 'none'}")
            print(f"           maps at the checks (step: distinct channels / probe acc): "
                  + " ".join(f"{c['step']}:{len(set(c['map_before']))}/{c['acc']:.2f}{'*' if c['fired'] else ''}" for c in ch))
        print()
    for k in ("CEIL_HEBB", "CEIL_DELTA"):
        print(f"  {k} transitions: " + ", ".join(f"s{s} {fmt_step(r['transition']) if ok_r(r) else '--'}"
                                               for s, r in sorted(rA[k].items())))
    cpus = collections.Counter(r.get("cpu") for k in KEYS for r in rA[k].values() if ok_r(r))
    print(f"  CPU labels in the records: {dict(cpus)}")
    print()

    if to is not None:
        print("=" * 100)
        print("X AND L SIDE BY SIDE (each machine's own seeds; pooled counts are the sums)")
        print("=" * 100)
        both = {machine: t, other_of(machine): to}
        print(f"  {'':40} {'X':>14} {'L':>14} {'pooled':>14}")
        for k in KEYS:
            cs = [(sum(success(k, r) for r in both[m_]["rA"][k].values()), len(both[m_]["seeds"][k])) for m_ in MACHINES]
            cells = [f"{c}/{n}" for c, n in cs] + [f"{sum(c for c, _ in cs)}/{sum(n for _, n in cs)}"]
            print(f"  {k:<12} {'perfect gate BOUND' if ARM[k]['ceil'] else 'BOUND ROUTED':<27} " + " ".join(f"{c_:>14}" for c_ in cells))
        for n in ("D1", "D2", "D4"):
            ds = [both[m_][n] for m_ in MACHINES]
            cells = [f"{d_['b']} vs {d_['c']}" for d_ in ds] + [f"{sum(d_['b'] for d_ in ds)} vs {sum(d_['c'] for d_ in ds)}"]
            print(f"  {n + ' discordant (split arm only vs other only)':<40} " + " ".join(f"{c_:>14}" for c_ in cells))
        print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps)")
    print("=" * 100)
    for k in KEYS:
        for s in t["seeds"][k]:
            r = get(store, k, s)
            if ok_r(r):
                lab = ("BOUND" if bound_r(r) else "not bound") if ARM[k]["ceil"] else outcome2(r)
                print(f"  {k:<12} s{s} {curve_str(r['curve'])}  -> {lab}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global SEEDS
    ap = argparse.ArgumentParser(description="The memory rule at eight streams: delta (beta 1, L2 keys) vs Hebbian, with the exact-copy split.")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--window", default=tgw.RESULTS_FILE, help="this machine's test_window_gate results (CHECKs 146, 151, 158)")
    ap.add_argument("--early", default=None, help="this machine's test_early_recipe results (inherited CHECK 130)")
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (inherited)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (inherited)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (inherited)")
    ap.add_argument("--scale", default=tgw.tsa.RESULTS_FILE, help="this machine's test_scale_axes results (inherited)")
    ap.add_argument("--short", default=tgw.tsc.RESULTS_FILE, help="this machine's test_short_conv results (inherited)")
    ap.add_argument("--also", default=None, help="the other machine's delta_recipe_results.json (pooled claims)")
    ap.add_argument("--report", action="store_true", help="only the report, from a finished results file")
    args = ap.parse_args()
    torch.set_num_threads(1)
    SEEDS = seeds_for(args.machine)

    if args.report:
        store = load_store(args.results)
        m = store.get("meta", {})
        assert m.get("machine") == args.machine, f"{args.results} was written by machine {m.get('machine')}"
        SEEDS = {k: tuple(v) for k, v in m["seeds"].items()}
        report(store, SEEDS, args.machine, 0.0, args.results, args.also)
        return

    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    early_path = args.early or tgw.ter.RESULTS_FILE
    early = load_store(early_path)
    window = load_store(args.window)
    also = load_store(args.also) if args.also else None
    files = {"early": early, "early_path": early_path, "window_path": args.window}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    if window.get("meta", {}).get("machine") != args.machine:
        print(f"STOP: {args.window} holds machine {window.get('meta', {}).get('machine')}'s test_window_gate runs")
        sys.exit(2)

    print("=" * 100)
    print("Delta recipe: the memory rule at eight streams (delta beta 1, L2 keys, vs Hebbian; LOCAL3 + SLOW + the exact-copy split)")
    print(f"  machine {args.machine}: {cpu_model()}")
    print(f"  test_window_gate's Part C/D path: run_sc D8, S=8, P=4, k=16, conv, {ITERS} updates, all eight streams from update 1")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<92} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  the trigger's probe: 64 sequences, Generator(12345); queries per stream {list(PROBE_COUNTS)}; mass at body keys 3j+1, "
          f"j = 0..31")
    print(f"  --window     {args.window} (CPU {window['meta'].get('cpu')}, git {window['meta'].get('git')}, "
          f"{len(window.get('runs', {}))} records)")
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
        if not args.force:
            for k in ("HEBB_SPLIT", "DELTA_SPLIT"):
                hit = [(s, get(store, k, s)) for s in SEEDS[k] if ok_r(get(store, k, s))]
                if hit:
                    cached[k] = hit[0]
        verify(pool, files, args.machine, early, also, window, cached)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = dict(machine=args.machine, torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, beta=BETA,
                    iters=ITERS, probe_counts=list(PROBE_COUNTS), window=dict(path=args.window, git=window["meta"].get("git")),
                    seeds={k: list(v) for k, v in SEEDS.items()}, started=now)
        store["meta"] = longrun.resume_meta(old_m, meta, head, args.workers, KEYS, force=args.force)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — pool-load timing: each arm's real path for {TIME_ITERS} steps (one evaluation and its statistics "
              f"included), {args.workers} simultaneous copies on the {args.workers} workers, the median; worst case = every "
              f"run to its last step")
        print("=" * 100)
        cost = {}
        for k in KEYS:
            cost[k] = statistics.median(pool.map(time_job, [k] * args.workers))
        done = {(k, s) for k in KEYS for s in SEEDS[k] if get(store, k, s) is not None and not args.force}
        h_all, h_rest = longrun.projection([(k, cost[k], ITERS, SEEDS[k]) for k in KEYS], args.workers, done)
        print(f"  delta / Hebbian per step: learned gate {cost['DELTA_SPLIT'] / cost['HEBB_SPLIT']:.2f}x, perfect gate "
              f"{cost['CEIL_DELTA'] / cost['CEIL_HEBB']:.2f}x; nothing is cut")
        store["meta"].update(projected_wall_h=h_all, projected_rest_h=h_rest, ms_per_step={k: cost[k] * 1000 for k in KEYS})
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            k = sp["arm"]
            if not rec.get("ok"):
                line = f"*** FAILED: {rec.get('error')}"
            else:
                tr = rec["transition"]
                lab = ("BOUND" if bound_r(rec) else "not bound") if ARM[k]["ceil"] else outcome2(rec)
                fired = ([f"{c['step']}{'' if c['key_ok'] else '!'}" for c in rec["checks"] if c["fired"]]
                         if rec.get("checks") is not None else None)
                line = (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} {lab}"
                        + (f"  copies at {fired}" if fired is not None else ""))
            print(f"   {k:<12} seed {sp['seed']}  {line}  {rec.get('secs', 0.0):.0f}s  (elapsed {(time.time() - t0) / 60:.1f}m)",
                  flush=True)

        specs = []
        for k in sorted(KEYS, key=lambda k: (not ARM[k]["ceil"], -cost[k])):
            specs += [spec(k, s, ITERS) for s in SEEDS[k]]
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
            done_, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fu in done_:
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
