#!/usr/bin/env python
"""
test_recipe_scope.py — does the hinge need the slow schedule, and does the recipe combine with
the stream curriculum? lr 1e-3 (SUB_LR), no restarts. Everything below is fixed before any run.

Run it directly (on X the paired files are results/X/...):

    python test_recipe_scope.py --workers 4 --slow results/X/slow_start_results.json \\
        --recipe results/X/stream_recipe_results.json --curriculum results/X/stream_curriculum_results.json
    python test_recipe_scope.py ... --also results/L/recipe_scope_results.json   (counts, descriptive)

BACKGROUND
- test_slow_start, both machines (X 9c5939e/b8c6007; L 9c5939e). Pooled post hoc:
  - HINGE0 39/40 vs A0 16/40 (23 vs 0);
  - HINGE8 37/40 vs DIRECT8 25/40 (12 vs 0);
  - HINGE4k16 35/40 vs A4k16 21/40 (18 vs 4), matching test_stream_recipe's restart recipe
    (A4k16_R 36/40) without restarts;
  - HINGE0 vs SLOW0: 9 vs 0;
  - HINGE_D8 0/20. On L, 4 of 10 split the 8 streams into two channels of four (eta2 by stream
    0.95-0.99) and stalled near 0.2.
- test_stream_curriculum's SC8 bound 10/36. Its failures were 8 collapses (position splits) and
  18 merges.
- Open: does the hinge need the slow schedule? Does the recipe combine with the stream
  curriculum?

THE RECIPE (test_slow_start's, reused: its make_recipe knob on the recorded run paths)
- HINGE = SLOW + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] on every training batch
  (the read gate at the body's key positions, eta2 pooled over channels); SLOW = gate W_in, W_h,
  W_g at 1e-3 throughout, every other parameter at 1e-4 for updates 1-2400, 1e-3 after (one Adam,
  two groups, moments carried over).
- "The hinge alone" (HONLY) = the hinge with ONE optimizer group at 1e-3 throughout (no slow
  phase): test_slow_start.make_recipe(slow=False, tau=0.2), the run path's own optimizer.
- On the stream curriculum the hinge reads each stage's own key positions: test_slow_start gained
  one knob, make_recipe(stage=True) -> attach_hinge(S=None), which takes each training batch's
  stream count from its length (stage_S: T = 3*(S*P + 1) - 1), so a stage with s_active streams
  is grouped by its s_active*P body triples (triple index j, and j < s_active*P/2 vs the rest).
  stage=False (the default) is unchanged (CHECK 113).

ARMS (lr 1e-3, no restarts)
- Part 0 (S=2, P=4, k=2, grouped, no conv, 24000 steps; test_short_conv's arm A run path),
  seeds 280-299, paired with this machine's recorded A0, SLOW0, HINGE0 (slow_start_results.json).
  Outcome DISCOVERED (bound and final VAL cos < 0.5).
    HONLY0    arm A + the hinge, ONE optimizer group at 1e-3 throughout (no slow phase)
- Part B (S=4, P=4, k=16, conv, 28800 steps; test_stream_recipe's A4k16 run path), seeds
  240-259, paired with the recorded A4k16 (stream_recipe) and HINGE4k16 (slow_start). Outcome
  BOUND.
    HONLY4k16 A4k16 + the hinge, one group at 1e-3
- Part C (S=8, P=4, k=16, conv; test_stream_curriculum's SC8 schedule and run path: 2, 4, then 8
  streams, switches at 4800 and 9600, 28800 steps), paired with the recorded SC8 (X 260-275,
  L 260-279: the seeds of this machine's recorded SC8 in 260-279). Outcome BOUND on the 8-stream
  set (the transition counted in stage 3 only, as SC8's).
    SC8_H     SC8 + the HINGE recipe (slow schedule for updates 1-2400 as in test_slow_start; the
              hinge computed on the current stage's key positions, groups by triple index and by
              half of the current sequence)
- Flags name this machine's recorded files: --slow (test_slow_start), --recipe
  (test_stream_recipe), --curriculum (test_stream_curriculum). Defaults are the files a run
  writes in this directory; on X, results/X/....

PAIRING
- A recorded file pairs if one of its recorded runs reproduces bit for bit on this machine
  (through step 2400: curve, statistics and gradient norms), whatever the CPU label (CHECK 118):
  slow_start's HINGE0 seed 280 (Part 0) and HINGE4k16 seed 240 (Part B), stream_recipe's A4k16
  seed 240, stream_curriculum's SC8 seed 260. A claim whose file does not pair is UNTESTED.

CLAIMS (per machine; exact McNemar one-sided, first arm higher; SHOWN if p < 0.05)
- Q1 HONLY0 beats A0.
- Q2 HINGE0 beats HONLY0 (the slow schedule adds to the hinge).
- Q3 HONLY4k16 beats A4k16.
- Q4 HINGE4k16 beats HONLY4k16.
- Q5 SC8_H beats SC8.
- Readings, printed:
  - "the hinge alone suffices" if Q1 and Q3 are SHOWN and Q2 and Q4 are not;
  - "both parts are needed" if Q2 or Q4 is SHOWN;
  - "the recipe helps the curriculum" if Q5 is SHOWN.
- Band for SC8_H (test_stream_curriculum's bands as fractions of its seeds: RELIABLE >= 0.9,
  MAJORITY >= 0.5, MINORITY >= 1 run, NEVER 0; 15 and 8 of 16, 18 and 10 of 20).

DIAGNOSTICS (labelled, not part of the verdict)
- Failure classes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome); hinge
  firing counts (updates 1-1200, 1-2400, whole run); Part 0 per seed beside A0, SLOW0, HINGE0.
- Part C: at 4800, 9600 and the end, the stream -> channel map on the 8-stream probe, the number
  of distinct channels holding the streams, and streams per channel; for SC8_H and the recorded
  SC8. Runs that stall (not bound) with the streams split into exactly 2 or 4 groups (distinct
  channels at the end) are counted.

CHECKS (after the inherited chain, which ends with test_slow_start's CHECK 112)
- 113 The stage knob is inert: test_slow_start's runs with make_recipe's default equal b8c6007's
      module (the hinge firing on every batch, TAU = 0; Parts 0 and B); and stage=True on a task
      with fixed S equals stage=False.
- 114 HONLY arms with TAU = 1.0 equal the plain recorded arm bit for bit through 2400 steps
      (HONLY0 vs the recorded A0 seed 280, HONLY4k16 vs the recorded A4k16 seed 240; and, weights
      included, vs a fresh plain run).
- 115 SC8_H with TAU = 1.0 and both groups at 1e-3 equals the recorded SC8 (seed 260) through 2400
      steps (and, weights included, a fresh plain SC8 run).
- 116 The hinge's groups at each curriculum stage cover exactly that stage's key positions (the
      body's key tokens, the query's key excluded), the hook's statistic equals a NumPy
      recomputation on the token-located key positions, and the perfect gate scores below 0.1 on
      both terms at each stage (20 probe batches per stage).
- 117 An optimizer pre-hook confirms the groups and lrs through each arm's real run path and
      schedule (forward stubbed, held-out accuracy 0.5): HONLY arms one group at 1e-3 at every
      update; SC8_H the gate group (W_in, W_h, W_g) at 1e-3, the rest 1e-4 through update 2400 and
      1e-3 after.
- 118 The pairing (above). Not asserted: a file that does not pair makes its claims UNTESTED.
- 119 A worker's run is bit-identical to the same run here (SC8_H seed 1, 6000 steps, through the
      switch to four streams).

RUNTIME
- The worst-case projection (every run to its last step) is printed before training.
- If it is over 12 h, SC8_H is cut to seeds 260-269 first. The Part 0 and Part B arms are never
  cut. The decision is stored at the first start of the results file and kept on a resume.
- Runs are scheduled longest first on 4 workers x 1 thread; finished runs are cached.
- Both machines run it.

OUTPUT
Per-seed raw values first, then counts, pairing, claims, readings, diagnostics and curves.
Results: recipe_scope_results.json (gitignored; X's copy in results/X/).
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import time_per_step, EVAL_EVERY
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_router_curriculum import get, load_store, init_worker, makespan, cpu_model, save_results, TAU_END, TIME_STEPS
from test_router_discovery import load_at
from test_router_reliability import strip_all
from test_conv_lr import bound_r
import test_stream_recipe as tsr
from test_stream_channels import med_int
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import (LR, LR_WARM, WARM, TAU, GATE, MAX0, MAXK, ALPHA, ok_r, st, hstr, hinge_counts, tag,
                             runs_of, curve_str, raw_rows, success, paired, norm, subset_equal, trunc, same_snap,
                             fired_of, outcome_key)

# ── Settings (fixed before any run) ──────────────────────────────────────────
LEGACY_SHA = "b8c6007"                     # head before this test's changes
DROP_H = 12.0
CUT_C = tuple(range(260, 270))             # SC8_H cut to these seeds if the projection is over DROP_H
PERFECT_MAX = 0.1                          # CHECK 116
T2 = WARM                                  # the CHECKs' 2400 steps
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "recipe_scope_results.json"

TASK48 = tsr.TASK48
STAGE = tscur.STAGE
REAL = dict(iters0=MAX0, iters=MAXK, warm=WARM, C=dict(tscur.REAL))
SCHED = dict(REAL)                         # the run's schedule (a dry run shortens it; the CHECKs use REAL)
ARMS = [
    dict(key="HONLY0", part="0", slow=False, tau=TAU, stage=False,
         label="HONLY0     arm A + the hinge, one group at 1e-3 (S=2, P=4, k=2, no conv)"),
    dict(key="HONLY4k16", part="B", slow=False, tau=TAU, stage=False,
         label="HONLY4k16  A4k16 + the hinge, one group at 1e-3 (S=4, P=4, k=16, conv)"),
    dict(key="SC8_H", part="C", slow=True, tau=TAU, stage=True,
         label="SC8_H      SC8 + HINGE (slow 1-2400; hinge on each stage's key positions)"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
SEEDS = dict(HONLY0=tuple(range(280, 300)), HONLY4k16=tuple(range(240, 260)), SC8_H=tuple(range(260, 280)))
# Recorded arms: (file flag, arm, part, seeds of the new arm they pair with)
REC = {"A0": ("slow", "A0", "0", "HONLY0"), "SLOW0": ("slow", "SLOW0", "0", "HONLY0"),
       "HINGE0": ("slow", "HINGE0", "0", "HONLY0"), "A4k16": ("recipe", "A4k16", "B", "HONLY4k16"),
       "HINGE4k16": ("slow", "HINGE4k16", "B", "HONLY4k16"), "SC8": ("curriculum", "SC8", "C", "SC8_H")}
# Each file's reproduction (CHECK 118): pairing key -> (file, arm, seed, run path, recipe (slow, tau) or None)
REPRO = {"slow0": ("slow", "HINGE0", 280, "0", (True, TAU)), "slowB": ("slow", "HINGE4k16", 240, "B", (True, TAU)),
         "recipe": ("recipe", "A4k16", 240, "B", None), "curriculum": ("curriculum", "SC8", 260, "C", None)}
REC_PAIR = {"A0": "slow0", "SLOW0": "slow0", "HINGE0": "slow0", "A4k16": "recipe", "HINGE4k16": "slowB", "SC8": "curriculum"}
CLAIMS = (("Q1", ("new", "HONLY0"), ("rec", "A0")), ("Q2", ("rec", "HINGE0"), ("new", "HONLY0")),
          ("Q3", ("new", "HONLY4k16"), ("rec", "A4k16")), ("Q4", ("rec", "HINGE4k16"), ("new", "HONLY4k16")),
          ("Q5", ("new", "SC8_H"), ("rec", "SC8")))
READINGS = {
    "alone": "the hinge alone suffices",
    "both": "both parts are needed",
    "cur": "the recipe helps the curriculum",
}


# ── Runs ─────────────────────────────────────────────────────────────────────
def run_path(where, seed, sched, recipe=None):
    """One run on a recorded run path: "0" and "B" are test_slow_start's part paths (arm A and A4k16),
    "C" test_stream_curriculum's SC8 path (the 2 -> 4 -> 8 stream curriculum)."""
    if where == "C":
        return tscur.run_sc(tscur.ARM["SC8"], seed, sched["C"], recipe=recipe)
    return tss.part_run(where, seed, sched["iters0"] if where == "0" else sched["iters"], recipe)


def run_job(sp):
    a = ARM[sp["arm"]]
    sched = sp["sched"]
    rc, _ = tss.make_recipe(a["slow"], a["tau"], warm=sched["warm"], stage=a["stage"])
    return run_path(a["part"], sp["seed"], sched, rc)


def spec(key, seed, sched):
    return dict(arm=key, seed=seed, sched=dict(sched))


def sched_for(iters):
    """REAL with every path cut to `iters` steps (the CHECKs)."""
    return dict(REAL, iters0=iters, iters=iters, C=dict(tscur.REAL, total=iters))


def var_job(where, seed, iters, slow=False, tau=None, warm_lr=LR_WARM, keep_at=None, plain=False, stage=False):
    """A check run on a recorded run path: plain (recipe None, or one that only sets keep), or
    test_slow_start's recipe (slow, tau, warm_lr, stage); with keep_at, the weights right after that step."""
    keep = None if keep_at is None else {"at": keep_at}
    if plain:
        rc = None if keep is None else (lambda kw: dict(kw, keep=keep))
    else:
        rc, _ = tss.make_recipe(slow, tau, warm_lr=warm_lr, keep=keep, stage=stage)
    rec = run_path(where, seed, sched_for(iters), rc)
    return dict(rec=rec, snap=None if keep is None else keep.get("snap"))


def legacy_job(part, tau):
    """CHECK 113: test_slow_start.variant_job (slow, tau; 1200 steps, seed 3) through b8c6007's module."""
    leg = load_at(LEGACY_SHA, "test_slow_start.py", "test_slow_start_legacy")
    return strip_all(leg.variant_job(part, 3, EVAL_EVERY, True, tau)["rec"])


def new_job(part, tau, stage=False):
    """CHECK 113: the same run through this module's test_slow_start (make_recipe's stage knob)."""
    if not stage:
        return strip_all(tss.variant_job(part, 3, EVAL_EVERY, True, tau)["rec"])
    return strip_all(var_job(part, 3, EVAL_EVERY, True, tau, stage=True)["rec"])


def lr_job(key):
    """CHECK 117: test_slow_start.lr_job's recipe for this test's arms: an optimizer step pre-hook records
    each group's lr at every update and the groups' parameters, through the arm's run path and real
    schedule; the forward is stubbed and every held-out accuracy is 0.5, so the run goes to its last step."""
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
        rc, _ = tss.make_recipe(a["slow"], a["tau"], keep=keep, stage=a["stage"])
        rec = run_path(a["part"], 3, REAL, rc)
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


def sc8_maker(seed, tau=None, stage=True):
    """make() for SC8's model (test_stream_recipe's builder for the k=16 arm A) through the recipe's builder."""
    a = tscur.ARM["SC8"]
    rc, state = tss.make_recipe(False, tau, stage=stage)
    kw = rc(dict(task=TASK48, builder=tsr.builder_for(a), stats_fn=lambda *x: {}))
    return kw["builder"](a, seed), state


def time_job(key):
    """ms/step through the arm's builder and hinge, evaluations charged once per EVAL_EVERY; for SC8_H on
    each stage's task (test_stream_curriculum.time_job's recipe)."""
    a = ARM[key]
    if a["part"] != "C":
        task = tss.PART[a["part"]]["task"]
        mk, _ = tss.maker(a["part"], 0, a["slow"], a["tau"])
        per = time_per_step(task, mk, steps=TIME_STEPS)
        m, data = mk(), eval_batch(task)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        ev = time.time() - t0
        return {0: max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY}
    mk, _ = sc8_maker(0, a["tau"])
    out = {}
    for n in (2, 4, 8):
        task = STAGE[n]
        per = time_per_step(task, mk, steps=TIME_STEPS)
        m, data = mk(), eval_batch(task)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        ev = time.time() - t0
        out[n] = (max(per - ev / TIME_STEPS, 1e-6), ev)
    ev8 = out[8][1]
    return {n: out[n][0] + ((out[n][1] if n != 8 else 0.0) + ev8) / EVAL_EVERY for n in out}


def full_cost(key, cost, sched):
    a = ARM[key]
    if a["part"] == "C":
        c, t1, t2, total = cost[key], sched["C"]["t1"], sched["C"]["t2"], sched["C"]["total"]
        return t1 * c[2] + (t2 - t1) * c[4] + (total - t2) * c[8]
    return cost[key][0] * (sched["iters0"] if a["part"] == "0" else sched["iters"])


def projection(cost, sched, workers):
    full = {k: full_cost(k, cost, sched) for k in KEYS}
    d = [full[k] for k in KEYS for _ in SEEDS[k]]
    return full, d, makespan(d, workers) / 3600


# ── Verification ─────────────────────────────────────────────────────────────
def key_positions(inp, task):
    """The body's key positions of each row, located from the tokens (key ids S..S+P-1), the query's
    key excluded (the last key token of the row)."""
    S, P = task.S, task.P
    mask = (inp >= S) & (inp < S + P)
    rows = [torch.nonzero(mask[b]).flatten().tolist() for b in range(inp.shape[0])]
    return [r[:-1] for r in rows], [r[-1] for r in rows]


def eta2_numpy_rows(gr, pos):
    """CHECK 116: the pooled eta^2 by order index and by half over the given key positions per row, in
    NumPy (float64)."""
    import numpy as np
    p = np.stack([gr[b, pos[b], :].double().numpy() for b in range(gr.shape[0])])     # (B, n, k)
    B, n, k = p.shape
    mu = p.reshape(-1, k).mean(0)
    tot = float(((p - mu) ** 2).sum())
    bi = float(sum(B * ((p[:, j, :].mean(0) - mu) ** 2).sum() for j in range(n)))
    lo, hi = p[:, : n // 2, :].reshape(-1, k), p[:, n // 2:, :].reshape(-1, k)
    bh = float(lo.shape[0] * ((lo.mean(0) - mu) ** 2).sum() + hi.shape[0] * ((hi.mean(0) - mu) ** 2).sum())
    return bi / tot, bh / tot


def verify(pool, files):
    print("test_slow_start.py's verification (which runs test_stream_curriculum's, and so on down to")
    print("test_multilayer_binding's), with X's recorded files:")
    tss.verify(pool, files["tss_files"])
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f = {}
    f["115", "hinge"] = pool.submit(var_job, "C", 260, T2, True, 1.0, LR, T2, False, True)
    f["115", "plain"] = pool.submit(var_job, "C", 260, T2, keep_at=T2, plain=True)
    f["114", "B", "honly"] = pool.submit(var_job, "B", 240, T2, False, 1.0, LR_WARM, T2)
    f["114", "B", "plain"] = pool.submit(var_job, "B", 240, T2, keep_at=T2, plain=True)
    f["118", "slowB"] = pool.submit(var_job, "B", 240, T2, True, TAU)
    f["119"] = pool.submit(run_job, spec("SC8_H", 1, dict(REAL, C=dict(tscur.REAL, total=6000))))
    for k in KEYS:
        f["117", k] = pool.submit(lr_job, k)
    for part in ("B", "0"):
        f["113", part, "legacy"] = pool.submit(legacy_job, part, 0.0)
        f["113", part, "new"] = pool.submit(new_job, part, 0.0)
    f["113", "0", "stage"] = pool.submit(new_job, "0", 0.0, True)
    f["114", "0", "honly"] = pool.submit(var_job, "0", 280, T2, False, 1.0, LR_WARM, T2)
    f["114", "0", "a0"] = pool.submit(var_job, "0", 280, T2, False, None, LR_WARM, T2)
    f["118", "slow0"] = pool.submit(var_job, "0", 280, T2, True, TAU)

    print(f"CHECK 113 the stage knob is inert: test_slow_start runs (seed 3, {EVAL_EVERY} steps, SLOW + the hinge at TAU = 0, so "
          f"it fires on every batch) with make_recipe's default vs {LEGACY_SHA}'s module, and stage=True on a fixed-S task:")
    good = True
    for part in ("0", "B"):
        L_, n_ = f["113", part, "legacy"].result(), f["113", part, "new"].result()
        fired, n = fired_of(n_)
        g = L_ == n_ and n_["ok"] and fired == n == EVAL_EVERY
        good &= g
        print(f"     Part {part} ({tss.PART[part]['src']}): legacy == default {L_ == n_}   hinge fired {fired} of {n}   curve "
              f"{[[c[0], round(c[1], 4)] for c in n_['curve']]}  -> {'UNCHANGED' if g else 'DIFFERS'}")
    s_ = f["113", "0", "stage"].result()
    g = s_ == f["113", "0", "new"].result()
    good &= g
    print(f"     Part 0 with stage=True (S read from each batch's length) == stage=False: {g}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 114 the HONLY arms with TAU = 1.0 equal the plain recorded arm bit for bit through {T2} steps:")
    good = True
    pl = {}
    for part, k, (fk, arm, s0) in (("0", "HONLY0", ("slow", "A0", 280)), ("B", "HONLY4k16", ("recipe", "A4k16", 240))):
        rh = f["114", part, "honly"].result()
        rp = f["114", part, "a0" if part == "0" else "plain"].result()
        pl[part] = rp["rec"]
        rec_old = get(files[fk], arm, s0)
        fired, n = fired_of(rh["rec"])
        same_r = rec_old is not None and subset_equal(trunc(norm(rh["rec"]), T2), trunc(rec_old, T2))
        rep = rec_old is not None and subset_equal(trunc(norm(rp["rec"]), T2), trunc(rec_old, T2))
        same_p = subset_equal(norm(strip_all(rh["rec"])), norm(strip_all(rp["rec"])))
        w = same_snap(rh["snap"], rp["snap"])
        g = same_p and w and fired == 0 and rh["rec"]["ok"] and (same_r or not rep)
        good &= g
        print(f"     {k:<9} seed {s0} vs the recorded {arm} ({files[fk + '_path']}; curve, statistics, gradient norms): "
              f"{same_r if rep else 'not compared: the recorded run does not reproduce here'}; vs a fresh "
              f"{'A0' if part == '0' else 'plain'} run: records equal {same_p}, weights at {T2} equal {w}   hinge fired "
              f"{fired} of {n}   curve {[round(c[1], 4) for c in rp['rec']['curve']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 115 SC8_H with TAU = 1.0 and both groups at {LR:g} equals the recorded SC8 through {T2} steps:")
    rh, rp = f["115", "hinge"].result(), f["115", "plain"].result()
    pl["C"] = rp["rec"]
    rec_old = get(files["curriculum"], "SC8", 260)
    fired, n = fired_of(rh["rec"])
    same_r = rec_old is not None and subset_equal(trunc(norm(rh["rec"]), T2), trunc(rec_old, T2))
    rep = rec_old is not None and subset_equal(trunc(norm(rp["rec"]), T2), trunc(rec_old, T2))
    same_p = subset_equal(norm(strip_all(rh["rec"])), norm(strip_all(rp["rec"])))
    w = same_snap(rh["snap"], rp["snap"])
    g = same_p and w and fired == 0 and rh["rec"]["ok"] and (same_r or not rep)
    print(f"     SC8_H seed 260 vs the recorded SC8 ({files['curriculum_path']}; 8-stream curve, stage curve, statistics, "
          f"gradient norms): {same_r if rep else 'not compared: the recorded run does not reproduce here'}; vs a fresh plain "
          f"SC8 run: records equal {same_p}, weights at {T2} equal {w}   hinge fired {fired} of {n}   stage curve "
          f"{[round(c[1], 4) for c in rp['rec']['curve_stage']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()

    print("CHECK 116 the hinge at each curriculum stage (S=8, P=4, k=16):")
    good = True
    P = TASK48.P
    a_ceil = tsr.ARM["ceiling8k16"]
    torch.manual_seed(3)
    m_ceil = tsr.builder_for(a_ceil)(a_ceil, 3)()
    m_ceil.eval()
    for n in (2, 4, 8):
        task = STAGE[n]
        tok, _ = task.make_batch(64, torch.Generator().manual_seed(116 + n))
        inp = tok[:, :-1]
        T = inp.shape[1]
        Z = tss.stage_S(T, P)
        pos = [3 * j + 1 for j in range(Z * P)]
        body, qk = key_positions(inp, task)
        cover = Z == n and all(r == pos for r in body) and all(q == 3 * Z * P + 1 for q in qk)
        mk, state = sc8_maker(5, tau=0.0)
        torch.manual_seed(5)
        m = mk()
        gg = torch.Generator().manual_seed(5 + 116)
        with torch.no_grad():
            for name, sc in (("W_g", 3.0), ("W_in", 0.3), ("W_h", 0.3)):
                wt = getattr(m, name)
                wt.copy_(torch.randn(wt.shape, generator=gg) * sc)
        m.train()
        gr = m(inp)[2]
        e_hook = state["last"]
        e_np = eta2_numpy_rows(gr.detach(), body)
        d = max(abs(e_hook[0] - e_np[0]), abs(e_hook[1] - e_np[1]))
        vals = []
        for s in range(20):
            with torch.no_grad():
                pinp = probe_batch(task, s)[0]
                g_ = m_ceil(pinp, TAU_END)[2]
            vals.append([float(x) for x in tss.eta2_hinge(g_, tss.stage_S(pinp.shape[1], P), P)])
        mi, mh = max(v[0] for v in vals), max(v[1] for v in vals)
        g = cover and d < 1e-6 and state["n"] == 1 and mi < PERFECT_MAX and mh < PERFECT_MAX
        good &= g
        print(f"     {n} streams (input length {T}): stage_S = {Z}; the hinge's {Z * P} positions 3j+1 (j < {Z * P}; halves "
              f"j < {Z * P // 2} and j >= {Z * P // 2}) == the body's key tokens in all 64 rows, the query's key at "
              f"{3 * Z * P + 1} excluded: {cover}")
        print(f"         the hook's statistic (TAU = 0, gate redrawn: W_g 3, W_in and W_h 0.3): index {e_hook[0]:.6f}, half "
              f"{e_hook[1]:.6f}; NumPy on the token-located positions {e_np[0]:.6f}, {e_np[1]:.6f}; max |diff| {d:.1e}")
        print(f"         the perfect gate (ceiling8k16) on 20 probe batches: max eta2_index {mi:.4f}, max eta2_half {mh:.4f} "
              f"(< {PERFECT_MAX})  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 117 the optimizer groups and lrs at every update, through each arm's run path and real schedule (forward "
          "stubbed, held-out accuracy 0.5; step pre-hook):")
    good = True
    for k in KEYS:
        a = ARM[k]
        r = f["117", k].result()
        it = REAL["C"]["total"] if a["part"] == "C" else (MAX0 if a["part"] == "0" else MAXK)
        if a["slow"]:
            rest = [n_ for n_ in r["all_names"] if n_ not in GATE]
            g = (r["runs"] == [[[LR, LR_WARM], WARM], [[LR, LR], it - WARM]] and r["names"] == [list(GATE), rest]
                 and "embed.weight" in rest and r["n"] == it and r["ok"] and r["stopped"] == it)
            print(f"     {k:<9} {r['n']} updates: group lrs {r['runs'][0][0]} x {r['runs'][0][1]}"
                  + (f", then {r['runs'][1][0]} x {r['runs'][1][1]}" if len(r["runs"]) > 1 else "")
                  + f"   gate group {r['names'][0]}   other group {len(rest)} tensors (embed.weight included: "
                  f"{'embed.weight' in rest}; convolution included: {'short_conv.conv_w' in rest})  -> {'OK' if g else 'WRONG'}")
        else:
            g = (r["runs"] == [[[LR], it]] and len(r["names"]) == 1 and sorted(r["names"][0]) == sorted(r["all_names"])
                 and r["n"] == it and r["ok"])
            print(f"     {k:<9} {r['n']} updates: one group at {r['runs'][0][0]} x {r['runs'][0][1]}, every parameter "
                  f"({len(r['names'][0])} tensors; no slow phase)  -> {'OK' if g else 'WRONG'}")
        good &= g
    ok &= good
    print()

    print(f"CHECK 118 the pairing with this machine's recorded files (a file pairs if a recorded run reproduces bit for bit "
          f"through step {T2}, whatever the CPU label):")
    pairs = {}
    runs118 = {"slow0": f["118", "slow0"].result()["rec"], "slowB": f["118", "slowB"].result()["rec"],
               "recipe": pl["B"], "curriculum": pl["C"]}
    for pk, (fk, arm, s0, where, rcp) in REPRO.items():
        store, path = files[fk], files[fk + "_path"]
        meta = store.get("meta", {})
        rec_old = get(store, arm, s0)
        rp = runs118[pk]
        rep = rec_old is not None and subset_equal(trunc(norm(rp), T2), trunc(rec_old, T2))
        pairs[pk] = bool(rep)
        print(f"     {path}: written on {meta.get('cpu')} (git {meta.get('git')})   this machine: {cpu_model()}   labels match: "
              f"{meta.get('cpu') == cpu_model()}")
        print(f"         {arm} seed {s0} re-run ({'its recipe' if rcp else 'plain path'}), {T2} steps: curve "
              f"{rp['curve']} vs recorded {[c for c in rec_old['curve'] if c[0] <= T2] if rec_old else None}; statistics and "
              f"gradient norms equal: {rep}  -> {'PAIRS' if rep else 'DOES NOT PAIR (its claims are UNTESTED)'}")
    print()

    print("CHECK 119 a worker's run is bit-identical to the same run here:")
    rw = f["119"].result()
    rh = run_job(spec("SC8_H", 1, dict(REAL, C=dict(tscur.REAL, total=6000))))
    fired, n = fired_of(rh)
    g = strip_all(rw) == strip_all(rh) and rh["ok"] and rh["stopped_at"] == 6000
    print(f"         SC8_H seed 1, 6000 steps (through the switch to four streams at 4800): equal {strip_all(rw) == strip_all(rh)}"
          f"   hinge fired {fired} of {n}   stage acc {[round(c[1], 3) for c in rh['curve_stage']]}   8-stream acc "
          f"{[round(c[1], 3) for c in rh['curve']]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}"
          + ("" if all(pairs.values()) else f"   (CHECK 118: {', '.join(p for p in pairs if not pairs[p])} do not pair: "
                                              f"see PAIRING)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return dict(pairs=pairs)


# ── Reporting ────────────────────────────────────────────────────────────────
def rec_runs(files, rk):
    fk, arm, part, new = REC[rk]
    return runs_of(files[fk], arm, SEEDS[new])


def runs_for(src, store, files):
    kind, k = src
    if kind == "new":
        return runs_of(store, k, SEEDS[k]), ARM[k]["part"], k
    fk, arm, part, new = REC[k]
    return rec_runs(files, k), part, f"{arm} (recorded)"


def cmap_at(r, step):
    x = st(r, step)
    return None if not x or "ch_map" not in x else x["ch_map"]


def groups_str(cm):
    if cm is None:
        return "--"
    sizes = sorted((cm.count(c) for c in set(cm)), reverse=True)
    return f"[{','.join(str(c) for c in cm)}] {len(set(cm))} ch {'+'.join(str(x) for x in sizes)}"


def report(store, files, info, drop, wall, path, also):
    pairs = info["pairs"]
    print()
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print("  (columns as test_slow_start's: acc = final held-out accuracy (8-stream set in Part C); transition (Part C: "
          "stage 3 only); ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9; eta i/h/k/s = pooled eta^2 of the read gate "
          "at key positions on the probe batch by index / half / key / stream (Part C: the 8-stream probe); hinge = "
          "training batches with a hinge term > 0 in updates 1-1200 / 1-2400 / the whole run; flag = eta^2 by index, half or "
          "key >= 0.5 at 1200; map = stream -> channel argmax; ROUTED = one-to-one and every stream >= 0.9)")
    if drop:
        print(f"  cut by the drop rule: {drop}")
    print()
    for k in KEYS:
        part = ARM[k]["part"]
        print(f"  {ARM[k]['label']}   (outcome {outcome_key(part)})")
        raw_rows(k, part, runs_of(store, k, SEEDS[k]), SEEDS[k])
        for rk, (fk, arm, p_, new) in REC.items():
            if new != k:
                continue
            print(f"  the recorded {arm} on the same seeds ({files[fk + '_path']}; {'pairs' if pairs[REC_PAIR[rk]] else 'DOES NOT PAIR'}):")
            raw_rows(arm, part, rec_runs(files, rk), SEEDS[k])

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in KEYS:
        part = ARM[k]["part"]
        groups = [(ARM[k]["label"][:10].strip(), runs_of(store, k, SEEDS[k]))] + \
                 [(f"  {arm} (recorded)", rec_runs(files, rk)) for rk, (fk, arm, p_, new) in REC.items() if new == k]
        for name, rs in groups:
            done = [s for s in SEEDS[k] if ok_r(rs.get(s))]
            hs = [hinge_counts(rs[s]) for s in done if hinge_counts(rs[s]) is not None]
            print(f"  {name:<22} {outcome_key(part)} {sum(success(part, rs[s]) for s in done):>2}/{len(SEEDS[k])}   bound "
                  f"{sum(bound_r(rs[s]) for s in done):>2}   collapsed {sum(bool(rs[s]['collapsed']) for s in done):>2}   "
                  f"present {len(done)}/{len(SEEDS[k])}" + (f"   hinge never fired {sum(1 for h in hs if h[2] == 0)}/{len(hs)}" if hs else ""))
        print()

    print("#" * 100)
    print("PAIRING")
    print("#" * 100)
    for pk, (fk, arm, s0, where, rcp) in REPRO.items():
        m_ = files[fk].get("meta", {})
        print(f"  {files[fk + '_path']} via {arm} seed {s0} (written on {m_.get('cpu')}, git {m_.get('git')}; this machine "
              f"{cpu_model()}): {'PAIRS (reproduced bit for bit, CHECK 118)' if pairs[pk] else 'DOES NOT PAIR: UNTESTED'}")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05)")
    print("#" * 100)
    res = {}
    for name, a_src, b_src in CLAIMS:
        ra, part, na = runs_for(a_src, store, files)
        rb, _, nb = runs_for(b_src, store, files)
        new_k = a_src[1] if a_src[0] == "new" else b_src[1]
        rec_k = a_src[1] if a_src[0] == "rec" else b_src[1]
        fn = (lambda r, part=part: success(part, r))
        d = paired(ra, rb, SEEDS[new_k], fn)
        untested = not pairs[REC_PAIR[rec_k]]
        verdict = "UNTESTED" if untested or d["n"] == 0 else ("SHOWN" if d["p"] < ALPHA else "NOT SHOWN")
        res[name] = verdict
        print(f"  {name}  {na} beats {nb} ({outcome_key(part)}): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; {na} only {d['b']}, "
              f"{nb} only {d['c']}; p = {d['p']:.4g}" + ("   (its recorded file does not pair)" if untested else ""))
        print(f"     *** {name}: {verdict} ***")
    rs = runs_of(store, "SC8_H", SEEDS["SC8_H"])
    c = sum(success("C", r) for r in rs.values())
    n = len(SEEDS["SC8_H"])
    print(f"  band SC8_H (BOUND, no restarts; RELIABLE >= {-(-9 * n // 10)}, MAJORITY >= {-(-n // 2)}, MINORITY >= 1, NEVER 0, of "
          f"{n}): {c}/{n}  -> {tscur.band_n(c, n)}")
    print()
    print("  READINGS:")
    lines = []
    if res["Q1"] == "SHOWN" and res["Q3"] == "SHOWN" and res["Q2"] != "SHOWN" and res["Q4"] != "SHOWN":
        lines.append(("Q1 and Q3 SHOWN, Q2 and Q4 not", READINGS["alone"]))
    if res["Q2"] == "SHOWN" or res["Q4"] == "SHOWN":
        lines.append(("Q2 or Q4 SHOWN", READINGS["both"]))
    if res["Q5"] == "SHOWN":
        lines.append(("Q5 SHOWN", READINGS["cur"]))
    for why, txt in lines:
        print(f"     {why}: \"{txt}\"")
    if not lines:
        print(f"     none of the readings applies ({', '.join(f'{k} {v}' for k, v in res.items())})")
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  Part 0 per seed (recorded A0, SLOW0, HINGE0 beside HONLY0; hinge = firing counts 1-1200/1-2400/all):")
    r0 = {k: rec_runs(files, k) for k in ("A0", "SLOW0", "HINGE0")}
    rh0 = runs_of(store, "HONLY0", SEEDS["HONLY0"])
    for s in SEEDS["HONLY0"]:
        print(f"    s{s}  A0 {tag('0', r0['A0'].get(s)):<22} SLOW0 {tag('0', r0['SLOW0'].get(s)):<22} HINGE0 "
              f"{tag('0', r0['HINGE0'].get(s)):<22} {hstr(r0['HINGE0'][s]) if ok_r(r0['HINGE0'].get(s)) else '--':<9} HONLY0 "
              f"{tag('0', rh0.get(s)):<22} {hstr(rh0[s]) if ok_r(rh0.get(s)) else '--'}")
    print()
    print("  Part B per seed (recorded A4k16 and HINGE4k16 beside HONLY4k16):")
    rb = {k: rec_runs(files, k) for k in ("A4k16", "HINGE4k16")}
    rhb = runs_of(store, "HONLY4k16", SEEDS["HONLY4k16"])
    for s in SEEDS["HONLY4k16"]:
        print(f"    s{s}  A4k16 {tag('B', rb['A4k16'].get(s)):<26} HINGE4k16 {tag('B', rb['HINGE4k16'].get(s)):<26} "
              f"{hstr(rb['HINGE4k16'][s]) if ok_r(rb['HINGE4k16'].get(s)) else '--':<9} HONLY4k16 {tag('B', rhb.get(s)):<26} "
              f"{hstr(rhb[s]) if ok_r(rhb.get(s)) else '--'}")
    print()
    print("  failure classes and outcomes (k=2: test_router_layout.fail_class; k=16: test_stream_recipe.outcome):")
    allg = [(k, ARM[k]["part"], runs_of(store, k, SEEDS[k])) for k in KEYS] + \
           [(f"{REC[rk][1]} (recorded)", REC[rk][2], rec_runs(files, rk)) for rk in REC]
    for name, part, rs_ in allg:
        cls = {}
        for s, r in sorted(rs_.items()):
            cls.setdefault(tag(part, r), []).append(s)
        print(f"    {name:<22} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<32} {' '.join('s' + str(x) for x in v)}")
    print()
    print("  hinge firing (training batches with a term > 0): runs that never fired; median [min, max] of updates 1-1200, "
          "1-2400, whole run:")
    for name, part, rs_ in allg:
        hs = [hinge_counts(r) for r in rs_.values() if ok_r(r) and hinge_counts(r) is not None]
        if hs:
            print(f"    {name:<22} never fired {sum(1 for h in hs if h[2] == 0)}/{len(hs)}   1-1200 {med_int([h[0] for h in hs])}"
                  f"   1-2400 {med_int([h[1] for h in hs])}   whole {med_int([h[2] for h in hs])}")
    print()
    print("  transitions, median [min, max]:")
    for name, part, rs_ in allg:
        trs = [r["transition"] for r in rs_.values() if ok_r(r) and r["transition"] is not None]
        print(f"    {name:<22} {med_int(trs) if trs else 'none bound'}   ({len(trs)} bound)")
    print()
    t1, t2 = SCHED["C"]["t1"], SCHED["C"]["t2"]
    print(f"  Part C: the 8-stream probe's stream -> channel map at {t1}, {t2} and the end, the number of distinct channels "
          f"holding the streams, and streams per channel:")
    rc8 = rec_runs(files, "SC8")
    rh8 = runs_of(store, "SC8_H", SEEDS["SC8_H"])
    for name, rs_ in (("SC8_H", rh8), ("SC8 (recorded)", rc8)):
        print(f"    {name}:")
        for s in SEEDS["SC8_H"]:
            r = rs_.get(s)
            if not ok_r(r):
                print(f"      s{s}  {'not run' if r is None else 'FAILED'}")
                continue
            print(f"      s{s}  {t1}: {groups_str(cmap_at(r, t1)):<34} {t2}: {groups_str(cmap_at(r, t2)):<34} end: "
                  f"{groups_str(cmap_at(r, 'end')):<34} {tag('C', r)}")
    print("  stalls with the streams split into exactly 2 or 4 groups (not bound; distinct channels at the end):")
    for name, rs_ in (("SC8_H", rh8), ("SC8 (recorded)", rc8)):
        st2 = [s for s, r in sorted(rs_.items()) if ok_r(r) and not bound_r(r) and len(set(r["end"]["ch_map"])) == 2]
        st4 = [s for s, r in sorted(rs_.items()) if ok_r(r) and not bound_r(r) and len(set(r["end"]["ch_map"])) == 4]
        nb = sum(1 for r in rs_.values() if ok_r(r) and not bound_r(r))
        print(f"    {name:<16} not bound {nb}:  2 groups {len(st2)} ({' '.join('s' + str(x) for x in st2) or '-'})   "
              f"4 groups {len(st4)} ({' '.join('s' + str(x) for x in st4) or '-'})")
    print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps; Part C: the 8-stream set, then the stage "
          f"set in stages 1-2); each run is followed by the recorded runs on its seed")
    print("=" * 100)
    for k in KEYS:
        part = ARM[k]["part"]
        recs = [(REC[rk][1], rec_runs(files, rk)) for rk in REC if REC[rk][3] == k]
        for s in SEEDS[k]:
            r = get(store, k, s)
            if ok_r(r):
                print(f"  {k:<11} s{s} {curve_str(r['curve'])}  -> {tag(part, r)}")
                if part == "C":
                    print(f"  {'  stage':<11} s{s} {curve_str(r['curve_stage'])}")
            for arm, rr in recs:
                if ok_r(rr.get(s)):
                    print(f"  {'(' + arm + ')':<11} s{s} {curve_str(rr[s]['curve'])}  -> {tag(part, rr[s])}")
    print()
    if also:
        other = load_store(also)
        print("=" * 100)
        print(f"THE OTHER MACHINE (descriptive): {also} (CPU {other.get('meta', {}).get('cpu')}, git {other.get('meta', {}).get('git')})")
        print("=" * 100)
        for k in KEYS:
            part = ARM[k]["part"]
            ro = [r for x, r in other["runs"].items() if x.split("|")[0] == k and ok_r(r)]
            if ro:
                print(f"  {k:<10} {outcome_key(part)} {sum(success(part, r) for r in ro)}/{len(ro)}   here "
                      f"{sum(success(part, r) for r in runs_of(store, k, SEEDS[k]).values())}/{len(SEEDS[k])}")
        print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


def main():
    ap = argparse.ArgumentParser(description="Does the hinge need the slow schedule; does the recipe combine with the curriculum?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (A0, SLOW0, HINGE0, HINGE4k16)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (A4k16)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (SC8)")
    ap.add_argument("--scale", default=tss.tsa.RESULTS_FILE, help="for test_slow_start's inherited CHECKs (DIRECT8)")
    ap.add_argument("--short", default=tss.tsc.RESULTS_FILE, help="for test_slow_start's inherited CHECK 104")
    ap.add_argument("--also", default=None, help="the other machine's recipe_scope_results.json, for counts (descriptive)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = {}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    have = sorted(int(x.split("|")[1]) for x in files["curriculum"].get("runs", {}) if x.split("|")[0] == "SC8")
    have = tuple(s for s in have if s in SEEDS["SC8_H"])
    if have:
        SEEDS["SC8_H"] = have
    sched = dict(SCHED)

    print("=" * 100)
    print("Recipe scope: the hinge without the slow schedule, and the HINGE recipe on the stream curriculum")
    print(f"  HONLY: the hinge (1.0 * [relu(eta2_index - {TAU}) + relu(eta2_half - {TAU})]) with one optimizer group at {LR:g}; "
          f"SC8_H: SLOW (gate {LR:g}, the rest {LR_WARM:g} for updates 1-{sched['warm']}) + the hinge on each stage's key positions")
    print(f"  Part 0: {sched['iters0']} steps (DISCOVERED); Part B: {sched['iters']} steps (BOUND); Part C: stages switch at "
          f"{sched['C']['t1']} and {sched['C']['t2']}, {sched['C']['total']} steps (BOUND on the 8-stream set, stage 3)")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<80} seeds {s[0]}-{s[-1]} ({len(s)})")
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
                             tau=TAU, sched=sched, pairs=info["pairs"],
                             files={k: dict(path=files[k + "_path"], cpu=files[k].get("meta", {}).get("cpu"),
                                            git=files[k].get("meta", {}).get("git")) for k in ("slow", "recipe", "curriculum")},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"), note="seed-level outcomes differ between machines")
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool through each arm's builder and hinge (SC8_H on "
              f"each stage's task), evaluations charged once per {EVAL_EVERY}; worst case = every run to its last step")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        full, d, total_h = projection(cost, sched, args.workers)
        for k in KEYS:
            per = ", ".join(f"{('stage ' + str(n) + ' streams') if n else 'all'} {v * 1000:.1f}" for n, v in sorted(cost[k].items()))
            print(f"  {k:<10} ms/step: {per:<55} x {len(SEEDS[k])} seeds   (a full run {full[k] / 60:.1f} min)")
        print(f"  {len(d)} runs: serial {sum(d) / 3600:.2f} h, {total_h:.2f} h on {args.workers} workers")
        drop = ""
        if old_drop is not None:
            if old_drop:
                SEEDS["SC8_H"] = tuple(s for s in SEEDS["SC8_H"] if s in old_drop)
                drop = f"SC8_H cut to seeds {SEEDS['SC8_H'][0]}-{SEEDS['SC8_H'][-1]}"
            full, d, total_h = projection(cost, sched, args.workers)
            print(f"  the drop decision recorded at the first start of this results file is kept: {drop or 'nothing cut'} "
                  f"(projection now {total_h:.2f} h)")
        else:
            decision = []
            if total_h > DROP_H:
                SEEDS["SC8_H"] = tuple(s for s in SEEDS["SC8_H"] if s in CUT_C)
                decision = list(SEEDS["SC8_H"])
                drop = f"SC8_H cut to seeds {SEEDS['SC8_H'][0]}-{SEEDS['SC8_H'][-1]}"
                full, d, total_h = projection(cost, sched, args.workers)
                print(f"  *** above {DROP_H:g} h: {drop} (drop rule); projection now {total_h:.2f} h ***")
                if total_h > DROP_H:
                    print(f"  (still above {DROP_H:g} h; the rule has no further step: the Part 0 and Part B arms are never cut)")
            else:
                print(f"  within {DROP_H:g} h: nothing is cut")
            store["meta"]["drop_decision"] = decision
        if old_drop is not None:
            store["meta"]["drop_decision"] = old_drop
        store["meta"].update(projected_wall_h=total_h, drop=drop, seeds={k: list(v) for k, v in SEEDS.items()})
        save_results(args.results, store)
        print()

        def describe(key, rec):
            if not rec.get("ok"):
                return f"*** FAILED: {rec.get('error')}"
            part = ARM[key]["part"]
            tr = rec["transition"]
            head_ = f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5}"
            if part == "0":
                mid = f"VALcos {rec['val_cos']:.3f}"
            else:
                e = rec["end"]
                mid = f"map [{','.join(str(c) for c in e['ch_map'])}] streams {[round(v, 2) for v in e['stream_acc']]}"
                if part == "C":
                    mid = f"8@{sched['C']['t1']} {tss.fmt(tsr.at(rec['curve'], sched['C']['t1']))} " \
                          f"8@{sched['C']['t2']} {tss.fmt(tsr.at(rec['curve'], sched['C']['t2']))} " + mid
            return f"{head_} {mid}  {tag(part, rec)}  hinge {hstr(rec)}"

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<10} seed {sp['seed']}  {describe(sp['arm'], rec)}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

        specs = []
        for k in sorted(KEYS, key=lambda k: -full[k]):
            specs += [spec(k, s, sched) for s in SEEDS[k]]
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
