#!/usr/bin/env python
"""
test_router_reliability.py — two questions about arm A (the k=2 recurrent gate on raw
embeddings, no readout), both fixed here before any result:
  Part 1: do more channels let the learned gate find the stream routing more often?
  Part 2: does a restart rule that uses only held-out accuracy make routing reliable?
Part 2 is a PROCEDURE, not the gate learning more reliably. Every attempt in it is exactly
an arm-A run (CHECK 46); the procedure only decides, from held-out accuracy at one early
evaluation, whether to keep a run or start again from a fresh seed. A RELIABLE verdict in
Part 2 says the procedure reliably ends with the routing; it says nothing about the gate's
own per-run rate, which is Part 1's arm A.

Run it directly:

    python test_router_reliability.py                  # CHECKs, projection, Part 1 + Part 2
    python test_router_reliability.py --also results/L/router_reliability_results.json

BACKGROUND
- Across both machines, arm A (k=2 recurrent gate on raw embeddings, no readout) found the
  stream routing on 94/160 seeds: X 49/80, L 45/80.
  - The two machines agree on 68 of 80 seeds, so a second machine is only a partial
    replication.
  - The labelled 5% nudge on arm A found it on 56/60.
  - Readout arms found it on 4/80. test_readout_path confirmed, on both machines, that the
    cause is the readout's gradient into the gate.
  - Single-channel runs: 0/100 distinct.
- The outcome is decided early: 87 of the 160 arm-A runs had split the streams (VAL cos <
  0.5) by step 2400 and went on to discover.
- A label-free early signal exists. In those 160 runs:
  - held-out accuracy at step 2400 was >= 0.6 in 82 runs, and all 82 discovered;
  - no non-discovering run exceeded 0.50 at step 2400;
  - 12 discoveries came later than that.
  T_CHECK = 2400 and A_CHECK = 0.6 were chosen from these data. This test uses fresh seeds.
- Channel-count hypothesis:
  - With k = 2 the gate can express only one binary split.
  - Early training is dominated by key confusion, which pulls a moving gate toward a
    key-identity split; once saturated, it cannot also split by stream.
  - With k = P*S = 8 the gate can split by key and by stream at once, so the early key pull
    need not exclude the stream split.
  - At k = 4 the gate may give each of the four keys its own channel and saturate again;
    k = 4 tests that.
- For any k, g_t.g_s = 1/k + d_t.d_s with sum_c d[c] = 0. So the uniform gate is still a
  stationary point, and more channels do not remove it (CHECK 44).
- Seed-level outcomes differ between machines; the verdict is per machine.

SUBSTRATE AND TASK: test_router_confirm's arm A, reused, not rewritten. BindTask P=4, S=2,
n_vals=16, n_q=1; MultiBDH n_layer=3, decay, N=256; symmetric recurrent gate on raw
embeddings, h_gate 32, no readout; lr 1e-3, BATCH 32; evaluation every 1200 steps on
eval_batch; early stop after 3 evaluations >= 0.95; MAX_ITERS = 24000.
DISCOVERED = transition is not None AND final VAL cos < 0.5. VAL cos is the cosine between
the two streams' mean read-gate vectors at value positions; it is defined for any k.

PART 1: CHANNEL COUNT
  A      k=2 (test_router_confirm's arm A)
  A_k4   k=4
  A_k8   k=8
The arms differ only in n_ch. Seeds 80-119, 40 per arm, disjoint from every earlier seed.
Paired: the same seed gives the same body, W_in, W_h and batch stream at every k; only W_g's
shape differs (it is drawn last, CHECK 43).

PART 2: RESTART PROTOCOL (arm A, k=2)
- 30 trials. Trial i uses attempt seeds 1000 + 5i + j, j = 0..4 (R_MAX = 5), in order; all
  are disjoint from every earlier seed.
- Each attempt trains arm A from its seed. At the held-out evaluation at step T_CHECK = 2400
  the protocol reads the held-out accuracy and nothing else:
  - if it is >= A_CHECK = 0.6 the attempt continues to MAX_ITERS with the usual early stop,
    and the trial's outcome is that attempt's (no further restarts);
  - otherwise the attempt is abandoned and the next seed starts.
- A trial SUCCEEDS if its continued attempt ends DISCOVERED. It fails if that attempt does
  not, or if all five attempts fail the check.
- The decision function takes only (step, held-out accuracy); it cannot read the gate, the
  stream labels or VAL cos (CHECK 46c).
- Each trial is one pool job, its attempts sequential inside it. Recorded: per attempt the
  seed, held-out accuracy at 1200 and 2400 and whether it passed; per continued attempt the
  full record, as in test_router_confirm; per trial success, attempts used and total
  training steps.
Fixed before any result: the arms, the seeds, MAX_ITERS, T_CHECK, A_CHECK, R_MAX and every
threshold below.

PRE-REGISTERED CLAIMS (each printed with its numbers)
  K8  more channels help at k=8: A_k8 vs A, discovered of 40, Fisher's exact test one-sided
      (A_k8 higher). HELPS if p < 0.05, else NOT SHOWN.
  K4  the same test for A_k4 vs A.
  Band for each Part 1 arm: RELIABLE >= 36/40, MAJORITY 20-35, MINORITY <= 19.
  RESTART  trials succeeded of 30: RELIABLE >= 27, PARTIAL 19-26, NOT RELIABLE <= 18.
No validity gate: Part 1 is comparative against its own k=2 arm, and Part 2 is judged on its
outcome. Fisher and McNemar are test_router_confirm's.

GATE STATISTICS, GENERALIZED TO ANY k (read gate, noise-free, eval mode, on the probe batch)
  conc(g) = (k / (2(k-1))) * sum_c |g[c] - 1/k|: 0 at uniform, 1 at one-hot.
  spread  = mean of conc over probe positions.
  sep     = 1/2 * sum_c |gbar_s0[c] - gbar_s1[c]|, gbar_s the mean gate over stream s's
            positions (all roles, as before).
  key_sep = mean, over key-token pairs i < j, of the same distance between the two keys'
            mean gates (both streams pooled).
  eff_ch  = exp(entropy of the overall mean gate), the number of channels in use.
At k = 2, spread and sep equal test_router_discovery's definitions (CHECK 42); at k = 2 the
records also carry key_part, and the old failure classification is printed for comparison.

DIAGNOSTICS, not part of the verdict
  Part 1: per-seed outcome table across k with exact McNemar p (k8 vs k2, k4 vs k2); escape
  step (first evaluation with VAL cos < 0.5); per arm spread, sep, key_sep and eff_ch at the
  first evaluation and at the end; failure modes (saddle: spread < 0.1; key split: key_sep >
  0.5 and sep < 0.2; other); key_sep of discovered runs at k = 4 and 8 (do they split keys as
  well as streams?); gate-parameter and rest-of-model gradient norms at steps 1, 10 and 100
  per k; runs that bound but did not discover.
  Part 2: the per-attempt pass rate at the check; the check's precision (continued attempts
  that did not discover); predicted vs observed (from Part 1's arm A: the fraction p_hat of
  runs that would pass the check, whether each of those discovered, and the predicted trial
  success 1 - (1 - p_hat)^5 against the observed trial rate); cost (mean and max training
  steps per trial, against the mean steps per run of Part 1's arm A).
  --also pooled counts, if a second machine's router_reliability_results.json is supplied.

CHECKS (printed before any training): test_readout_path's verification (which runs every
earlier test's), then
  42 the generalized statistics reduce to the old ones at k = 2 on a trained arm-A model;
  43 the pairing across k at seed 80 (parameters other than W_g, first 3 batches);
  44 the uniform gate is still a stationary point at k = 4 and 8;
  45 the recurrent gate can express the stream routing at k = 4 and 8 (argmax >= 0.99);
  46 the restart protocol: (a) an attempt is bit-identical to a plain arm-A run through the
     check and after it; (b) the attempt seeds; (c) the decision function reads only the
     step and the held-out accuracy (one pass and one fail on short synthetic curves);
  47 a worker's run, for a Part 1 run and a Part 2 trial, is bit-identical to the main
     process's.

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to router_reliability_results.json (gitignored; copied
into results/X/ after the run).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import onset_run, attempt, fmt_step, logits_of, EVAL_EVERY, BATCH
import test_multilayer_binding as tmb
from test_multilayer_binding import build, FIT_ITERS, FIT_PASS
from test_instrument_v2 import Instrument, perfect_gate_general
from test_router_confirm import fisher_greater, mcnemar_exact
import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, make_fn, escape, get, count, mode_of, load_store, init_worker,
    makespan, cpu_model, same_weights, save_results, probe_batch, gate_stats, time_arm,
    DISC_COS, COLLAPSE, TAU_END,
)
from test_router_discovery import SADDLE_SPREAD, LOCK_SEP
import test_readout_path as trp
from test_readout_path import grad_norms, GRAD_STEPS
from test_channel_binding import eval_batch

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
SEEDS = tuple(range(80, 120))              # Part 1, 40 per arm
TRIALS = 30                                # Part 2
R_MAX = 5
TRIAL_BASE = 1000
T_CHECK = 2400
A_CHECK = 0.6
KEY_SPLIT = 0.5                            # failure mode "key split": key_sep > 0.5, sep < 0.2
ALPHA = 0.05
RELIABLE_1, MAJORITY_1 = 36, 20            # Part 1 bands, of 40
RELIABLE_R, PARTIAL_R = 27, 19             # Part 2 bands, of 30
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "router_reliability_results.json"
EARLIER_SEEDS = tuple(range(80))           # every seed used by earlier router tests

A = dict(trc.ARM["A"], key="A", label="A      k=2 (test_router_confirm's arm A)")
ARMS = [A,
        dict(A, key="A_k4", n_ch=4, label="A_k4   k=4"),
        dict(A, key="A_k8", n_ch=8, label="A_k8   k=8")]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)


def trial_seeds(i):
    return tuple(TRIAL_BASE + R_MAX * i + j for j in range(R_MAX))


# ── Gate statistics for any k ────────────────────────────────────────────────
def tv(a, b):
    """Half the L1 distance between two mean gate vectors."""
    return 0.5 * (a - b).abs().sum().item()


@torch.no_grad()
def gate_stats_k(model, task, probe, step):
    """The docstring's generalized statistics, noise-free (eval mode). The per-role cosines
    are test_router_discovery.gate_stats's (defined for any k), and so is key_part at k=2."""
    base = gate_stats(model, task, probe, step)
    was = model.training
    model.eval()
    try:
        pinp, plab, _ = probe
        g = model(pinp, TAU_END)[2]                              # read gate, (B, T, k)
    finally:
        model.train(was)
    k = g.shape[-1]
    spread = ((k / (2 * (k - 1))) * (g - 1.0 / k).abs().sum(-1)).mean().item()
    sep = tv(g[plab == 0].mean(0), g[plab == 1].mean(0))
    km = [g[pinp == c].mean(0) for c in range(task.S, task.S + task.P)]
    key_sep = statistics.fmean(tv(km[i], km[j]) for i in range(len(km))
                               for j in range(i + 1, len(km)))
    gm = g.reshape(-1, k).mean(0)
    eff_ch = math.exp(-torch.special.xlogy(gm, gm).sum().item())
    out = dict(step=step, k=k, sep=sep, spread=spread, key_sep=key_sep, eff_ch=eff_ch,
               key_cos=base["key_cos"], val_cos=base["val_cos"], ctx_cos=base["ctx_cos"])
    if k == 2:
        out["key_part"] = base["key_part"]
    return out


def mode_k(r):
    e = r["end"]
    if e["spread"] < SADDLE_SPREAD:
        return "saddle"
    if e["key_sep"] > KEY_SPLIT and e["sep"] < LOCK_SEP:
        return "key split"
    return "other"


# ── One arm-A-style run (Part 1, and every Part 2 attempt) ──────────────────
class Abandon(Exception):
    """Raised from onset_run's on_eval when the restart rule abandons an attempt."""


def run_one(a, seed, iters, eval_every=EVAL_EVERY, check=None, keep=None):
    """test_readout_path.run_spec's record for one arm and seed, with the generalized gate
    statistics. check(model, step, data) runs after each evaluation's statistics (Part 2);
    keep (CHECK 46 only) receives the weights after opt.step at keep["at"] and the model."""
    task = TASK
    probe, data = probe_batch(task, seed), eval_batch(task)
    make, stats, grad = make_fn(a, seed), [], {}

    def make_and_measure():
        m = make()
        stats.append(gate_stats_k(m, task, probe, 0))
        return m

    def on_eval(m, step):
        stats.append(gate_stats_k(m, task, probe, step))
        if check is not None:
            check(m, step, data)

    def grad_hook(m, step):
        if step in GRAD_STEPS:
            grad[str(step)] = grad_norms(m)

    def post_step(m, step):
        if step == keep["at"]:
            keep["snap"] = {n: p.detach().clone() for n, p in m.named_parameters()}

    def post(m):
        if keep is not None:
            keep["model"] = m
        end = gate_stats_k(m, task, probe, "end")
        return dict(key_cos=end["key_cos"], val_cos=end["val_cos"], ctx_cos=end["ctx_cos"],
                    end=end)

    rec = attempt(task, make_and_measure, seed, data, post=post, max_iters=iters,
                  eval_every=eval_every, lr=SUB_LR, on_eval=on_eval, grad_hook=grad_hook,
                  post_step=post_step if keep is not None else None)
    rec.update(stats=stats, grad=grad, iters=iters, k=a["n_ch"])
    if rec["ok"]:
        rec["acc"] = rec["curve"][-1][1] if rec["curve"] else float("nan")
        rec["collapsed"] = rec["acc"] < COLLAPSE
        rec["discovered"] = bool(rec["transition"] is not None and rec["val_cos"] < DISC_COS)
    return rec


# ── Part 2: the restart protocol ─────────────────────────────────────────────
def make_decide(t_check, a_check):
    def decide(step, acc):
        """The restart rule: the step and the held-out accuracy at it, nothing else."""
        assert step == t_check
        return acc >= a_check
    return decide


DECIDE = make_decide(T_CHECK, A_CHECK)


def curve_acc(rec, step):
    return next((a for st, a, _ in rec["curve"] if st == step), None)


def run_attempt(seed, iters, decide, t_check, eval_every=EVAL_EVERY, keep=None):
    """One attempt: arm A from `seed`. At each evaluation up to t_check the held-out
    accuracy is read (tbo.evaluate on the same model and batch, so it is the value onset_run
    just recorded); at t_check decide(step, acc) keeps the run or abandons it."""
    assert iters >= t_check
    info = {"accs": {}}

    def check(m, step, data):
        if step > t_check:
            return
        acc, _ = tbo.evaluate(m, TASK, data)
        info["accs"][str(step)] = acc
        if step == t_check:
            info["passed"] = bool(decide(step, acc))
            if not info["passed"]:
                raise Abandon(f"held-out accuracy {acc:.4f} at step {step}")

    rec = run_one(ARM["A"], seed, iters, eval_every, check=check, keep=keep)
    out = dict(seed=seed, accs=info["accs"], passed=info.get("passed"))
    if info.get("passed") is False:
        out.update(steps=t_check, rec=None)
    elif not rec["ok"]:
        out.update(steps=None, rec=rec, error=rec["error"])
    else:
        out.update(steps=rec["stopped_at"], rec=rec,
                   acc_match=all(curve_acc(rec, int(s)) == v for s, v in info["accs"].items()))
    return out


def run_trial(sp, decide=None):
    """One trial: attempts in order until one passes the check (or R_MAX fail it)."""
    decide = make_decide(sp["t_check"], sp["a_check"]) if decide is None else decide
    ts = time.time()
    attempts = []
    for seed in sp["seeds"]:
        at = run_attempt(seed, sp["iters"], decide, sp["t_check"], sp["eval_every"])
        attempts.append(at)
        if at["passed"] is not False:            # continued (or an error)
            break
    last = attempts[-1]
    cont = last if last["passed"] else None
    err = last.get("error")
    success = bool(cont is not None and cont["rec"]["ok"] and cont["rec"]["discovered"])
    return dict(ok=err is None, trial=sp["trial"], seeds=list(sp["seeds"]), attempts=attempts,
                used=len(attempts), continued=cont["seed"] if cont else None, success=success,
                steps=sum(x["steps"] for x in attempts if x["steps"] is not None), error=err,
                secs=time.time() - ts)


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_job(sp):
    if sp["part"] == 2:
        return run_trial(sp)
    return run_one(ARM[sp["arm"]], sp["seed"], sp["iters"])


def spec1(key, seed, iters):
    return dict(part=1, arm=key, seed=seed, iters=iters)


def spec2(i, iters):
    return dict(part=2, trial=i, seeds=trial_seeds(i), iters=iters, t_check=T_CHECK,
                a_check=A_CHECK, eval_every=EVAL_EVERY)


def strip_all(x):
    """Drop wall-clock fields, at any depth."""
    if isinstance(x, dict):
        return {k: strip_all(v) for k, v in x.items() if k != "secs"}
    if isinstance(x, list):
        return [strip_all(v) for v in x]
    return x


# ── CHECK 45's fit: test_multilayer_binding.gate_fit on arm a's model ────────
def gate_fit_k(task, a):
    """tmb.gate_fit line for line, on build(task, a) with the perfect gate zero-padded to
    k = a["n_ch"] channels (channel s for stream s, zero on the other k - S)."""
    g = torch.Generator().manual_seed(11)
    tok, _ = task.make_batch(64, g)
    x = tok[:, :-1]
    tgt = F.pad(perfect_gate_general(x, task.ctx_tokens), (0, a["n_ch"] - task.S))
    m = build(task, a, ARCH, 3)
    opt = torch.optim.Adam([m.W_in, m.W_h, m.W_g, m.embed.weight], lr=1e-2)
    for _ in range(FIT_ITERS):
        gr, _ = Instrument.gates(m, x, m.embed(x))
        loss = F.mse_loss(gr, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gr, _ = Instrument.gates(m, x, m.embed(x))
        return F.mse_loss(gr, tgt).item(), (gr.argmax(-1) == tgt.argmax(-1)).float().mean().item()


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_readout_path.py's verification (which runs test_router_confirm's, and so on")
    print("down to test_multilayer_binding's):")
    trp.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    t = TASK
    data = eval_batch(t)

    # the plain arm-A run that CHECKs 42 and 46(a) compare against: seed 3, 3600 steps,
    # weights kept after opt.step at T_CHECK
    s = 3
    plain = {"at": T_CHECK}

    def snap(m, step):
        if step == T_CHECK:
            plain["snap"] = {n: p.detach().clone() for n, p in m.named_parameters()}
    m_plain, c_plain, _ = onset_run(t, make_fn(A, s), s, max_iters=3 * EVAL_EVERY, data=data,
                                    lr=SUB_LR, post_step=snap)

    print(f"CHECK 42 the generalized statistics reduce to the old ones at k = 2: arm A, seed {s},")
    print(f"         trained {c_plain[-1][0]} steps, probe batch of seed {s}:")
    probe = probe_batch(t, s)
    old, new = gate_stats(m_plain, t, probe, "end"), gate_stats_k(m_plain, t, probe, "end")
    d_sp, d_se = abs(old["spread"] - new["spread"]), abs(old["sep"] - new["sep"])
    same_rest = all(old[f] == new[f] for f in ("key_part", "key_cos", "val_cos", "ctx_cos"))
    good = d_sp < 1e-6 and d_se < 1e-6 and same_rest and old["spread"] > 0.05
    ok &= good
    print(f"         spread old {old['spread']:.7f} new {new['spread']:.7f} (|diff| {d_sp:.1e})   "
          f"sep old {old['sep']:.7f} new {new['sep']:.7f} (|diff| {d_se:.1e})")
    print(f"         key_part and the role cosines carried over unchanged: {same_rest}   "
          f"(key_sep {new['key_sep']:.4f}, eff_ch {new['eff_ch']:.4f})  -> "
          f"{'REDUCES TO test_router_discovery' if good else 'DIFFERS'}")
    print()

    print("CHECK 43 the pairing: seed 80, A, A_k4 and A_k8 at init and over the first 3 batches:")
    models = {k: make_fn(ARM[k], 80)() for k in KEYS}
    pa = dict(models["A"].named_parameters())
    same_init, shapes = True, {}
    for k in KEYS:
        pk = dict(models[k].named_parameters())
        shapes[k] = tuple(pk["W_g"].shape)
        same_init &= sorted(pk) == sorted(pa) and all(
            torch.equal(pa[n], pk[n]) for n in pa if n != "W_g")
    fed = {}
    for k in KEYS:
        rec, mk = [], make_fn(ARM[k], 80)

        def mk2(mk=mk, rec=rec):
            mm = mk()
            mm.register_forward_pre_hook(
                lambda mod, args: rec.append(args[0].clone()) if mod.training else None)
            return mm
        onset_run(t, mk2, 80, max_iters=3, lr=SUB_LR)
        fed[k] = rec
    same_b = all(len(fed[k]) == 3 and all(torch.equal(u, v) for u, v in zip(fed["A"], fed[k]))
                 for k in KEYS)
    good = same_init and same_b and [shapes[k][0] for k in KEYS] == [2, 4, 8]
    ok &= good
    print(f"         parameters other than W_g bitwise identical: {same_init}   first 3 batches "
          f"identical: {same_b}")
    print(f"         W_g shapes: " + "   ".join(f"{k} {shapes[k]}" for k in KEYS)
          + f"  -> {'PAIRED' if good else 'WRONG'}")
    print()

    print("CHECK 44 the stationary point generalizes: W_g = 0, seed 5, one training batch (seed 44):")
    tok = t.make_batch(BATCH, torch.Generator().manual_seed(44))[0]
    x, y = tok[:, :-1], tok[:, 1:]
    good = True
    for k in KEYS:
        m = make_fn(ARM[k], 5)()
        with torch.no_grad():
            m.W_g.zero_()
        kk = ARM[k]["n_ch"]
        uni = True
        for train in (True, False):
            m.train(train)
            with torch.no_grad():
                gr = m(x, TAU_END)[2]
            uni &= torch.equal(gr, torch.full_like(gr, 1.0 / kk))
        m.train()
        m.zero_grad(set_to_none=True)
        ql, qt, _ = t.select(logits_of(m, x), y, None)
        F.cross_entropy(ql, qt).backward()
        gn, rn = grad_norms(m)
        g = uni and gn <= 1e-6 * rn
        good &= g
        print(f"         k={kk}: gate exactly 1/k at every position (train and eval): {uni}   "
              f"|grad| gate (W_in, W_h, W_g) {gn:.3e}   rest {rn:.3e}  -> "
              f"{'STATIONARY' if g else 'NOT STATIONARY'}")
    ok &= good
    print()

    print(f"CHECK 45 expressibility: tmb.gate_fit's recipe ({FIT_ITERS} Adam steps at lr 0.01 on "
          f"W_in, W_h, W_g, embed) toward")
    print("         the stream routing padded to k channels (one-hot on channel s for stream s):")
    ref_a = dict(A, mult=tmb.MULT)
    same_fit = gate_fit_k(t, ref_a) == tmb.gate_fit(t, ARCH)
    good = same_fit
    print(f"         the recipe reproduces tmb.gate_fit exactly (k=2, its own model at mult "
          f"{tmb.MULT}): {same_fit}")
    for k in KEYS:
        mse, am = gate_fit_k(t, ARM[k])
        g = am >= FIT_PASS or ARM[k]["n_ch"] == 2
        good &= g
        print(f"         {k:<5} k={ARM[k]['n_ch']}: mse {mse:.2e}   argmax match {am:.4f}"
              + ("" if ARM[k]["n_ch"] == 2 else f"  -> {'FITS' if am >= FIT_PASS else 'FAILS'} "
                 f"(>= {FIT_PASS})"))
    ok &= good
    print()

    print("CHECK 46 the restart protocol:")
    print(f"     (a) an attempt (seed {s}, run to {3 * EVAL_EVERY}) vs the plain arm-A run above, with the "
          f"decision forced both ways:")
    keep = {"at": T_CHECK}
    at_p = run_attempt(s, 3 * EVAL_EVERY, lambda step, acc: True, T_CHECK, keep=keep)
    thr_p = all(torch.equal(plain["snap"][n], keep["snap"][n]) for n in plain["snap"])
    aft_p = same_weights(m_plain, keep["model"]) and at_p["rec"]["curve"] == c_plain
    keep_f = {"at": T_CHECK}
    at_f = run_attempt(s, 3 * EVAL_EVERY, lambda step, acc: False, T_CHECK, keep=keep_f)
    thr_f = all(torch.equal(plain["snap"][n], keep_f["snap"][n]) for n in plain["snap"])
    accs_f = [at_f["accs"].get(str(st)) for st in (EVAL_EVERY, T_CHECK)]
    accs_plain = [curve_acc(dict(curve=c_plain), st) for st in (EVAL_EVERY, T_CHECK)]
    real = DECIDE(T_CHECK, accs_plain[1])
    good_a = (thr_p and aft_p and at_p["passed"] and at_p["acc_match"] and at_p["steps"] == 3600
              and thr_f and at_f["passed"] is False and at_f["steps"] == T_CHECK
              and accs_f == accs_plain and at_f["rec"] is None)
    print(f"         forced pass: weights at step {T_CHECK} equal {thr_p}   weights at step 3600 and "
          f"curve equal {aft_p}   accuracy read == curve {at_p['acc_match']}")
    print(f"         forced fail: weights at step {T_CHECK} equal {thr_f}   abandoned at step "
          f"{at_f['steps']}   accuracies read {[round(v, 4) for v in accs_f]} == plain curve's: "
          f"{accs_f == accs_plain}")
    print(f"         the real rule on this seed: accuracy {accs_plain[1]:.4f} at step {T_CHECK} -> "
          f"{'pass' if real else 'fail'}  -> {'IDENTICAL' if good_a else 'DIFFERS'}")
    all_seeds = [x for i in range(TRIALS) for x in trial_seeds(i)]
    good_b = (all_seeds == [TRIAL_BASE + R_MAX * i + j for i in range(TRIALS) for j in range(R_MAX)]
              and len(set(all_seeds)) == TRIALS * R_MAX
              and not set(all_seeds) & (set(EARLIER_SEEDS) | set(SEEDS)))
    print(f"     (b) attempt seeds {all_seeds[0]}-{all_seeds[-1]}, {len(set(all_seeds))} distinct, "
          f"1000 + 5i + j: {good_b}; disjoint from earlier seeds 0-79 and Part 1's "
          f"{SEEDS[0]}-{SEEDS[-1]}: {not set(all_seeds) & (set(EARLIER_SEEDS) | set(SEEDS))}")
    params = list(inspect.signature(DECIDE).parameters)
    cv = inspect.getclosurevars(DECIDE)
    inputs_ok = (params == ["step", "acc"] and not cv.globals and not cv.builtins
                 and cv.nonlocals == {"t_check": T_CHECK, "a_check": A_CHECK})
    print(f"     (c) decide's parameters: {params}; it reads no global name ({dict(cv.globals)}) "
          f"and only the constants {dict(cv.nonlocals)}: {inputs_ok}")
    # synthetic curves: onset_run's evaluate stubbed to return a fixed accuracy per training
    # step (the step counted by an optimizer pre-hook); every evaluation 2 steps, check at 4
    synth = {"pass": {2: 0.31, 4: 0.64, 6: 0.90, 8: 0.97},
             "fail": {2: 0.35, 4: 0.52, 6: 0.55, 8: 0.58}}
    cnt = {"opt": None, "n": 0}

    def pre(opt, args, kw):
        if cnt["opt"] is not opt:
            cnt["opt"], cnt["n"] = opt, 0
        cnt["n"] += 1
    real_eval = tbo.evaluate
    good_c = inputs_ok
    for name, sc in synth.items():
        calls, rule = [], make_decide(4, A_CHECK)

        def spy(step, acc, rule=rule, calls=calls):
            calls.append((step, acc))
            return rule(step, acc)
        tbo.evaluate = lambda model, task, d, sc=sc: (sc[cnt["n"]], 0.0)
        h = register_optimizer_step_pre_hook(pre)
        try:
            tr = run_trial(dict(trial=-1, seeds=(3, 4, 5, 6, 7), iters=8, t_check=4,
                                a_check=A_CHECK, eval_every=2), decide=spy)
        finally:
            h.remove()
            tbo.evaluate = real_eval
        types_ok = all(type(st) is int and type(ac) is float for st, ac in calls)
        if name == "pass":
            g = (types_ok and calls == [(4, 0.64)] and tr["used"] == 1 and tr["continued"] == 3
                 and tr["steps"] == 8 and [c[1] for c in tr["attempts"][0]["rec"]["curve"]]
                 == [0.31, 0.64, 0.90, 0.97])
        else:
            g = (types_ok and calls == [(4, 0.52)] * 5 and tr["used"] == 5
                 and tr["continued"] is None and not tr["success"] and tr["steps"] == 20)
        good_c &= g
        print(f"         synthetic '{name}' curve {list(sc.values())}: decide saw {calls}; attempts "
              f"used {tr['used']}, continued seed {tr['continued']}, training steps {tr['steps']}"
              f"  -> {'OK' if g else 'WRONG'}")
    good = good_a and good_b and good_c
    ok &= good
    print()

    print("CHECK 47 a worker's run is bit-identical to the same run here:")
    good = True
    sp = spec1("A_k8", 1, EVAL_EVERY)
    rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
    g = strip_all(rw) == strip_all(rh) and rh["ok"]
    good &= g
    print(f"         Part 1  A_k8 seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   "
          f"eff_ch at the evaluation {rh['stats'][-1]['eff_ch']:.4f}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    for a_check in (A_CHECK, 0.0):
        sp = dict(part=2, trial=-1, seeds=(1, 2, 3, 4, 5), iters=60, t_check=40, a_check=a_check,
                  eval_every=20)
        rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        print(f"         Part 2  trial, seeds 1-5, check at step 40 of 60 (evaluations every 20), "
              f"A_CHECK={a_check}: equal {strip_all(rw) == strip_all(rh)}   attempts used "
              f"{rh['used']}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def band1(c):
    return "RELIABLE" if c >= RELIABLE_1 else "MAJORITY" if c >= MAJORITY_1 else "MINORITY"


def band2(c):
    return "RELIABLE" if c >= RELIABLE_R else "PARTIAL" if c >= PARTIAL_R else "NOT RELIABLE"


def med(xs):
    return f"{statistics.median(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]" if xs else "--"


def fmt_acc(x):
    return "--" if x is None else f"{x:.4f}"


def curve_lines(tag, r, mark_at=None):
    ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
    mark = lambda st: "^" if st == mark_at else " "
    accs = "".join(f"{mark(st)}{round(acc * 100):>3}" for st, acc, _ in r["curve"])
    coss = "".join(f"{mark(st)}{round(ev[st]['val_cos'] * 100):>3}" if st in ev else "  --"
                   for st, _, _ in r["curve"])
    print(f"  {tag} acc {accs}")
    print(f"  {'':<{len(tag)}} cos {coss}   -> "
          f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}"
          f"{'  DISCOVERED' if r['discovered'] else ''}")


def report(store, wall, path, also):
    seeds, n = list(SEEDS), len(SEEDS)
    trials = [store["trials"].get(str(i)) for i in range(TRIALS)]
    nt = TRIALS
    print("#" * 100)
    print("PER-SEED RAW RESULTS (Part 1) — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<5} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>7} "
          f"{'spread':>7} {'key_sep':>7} {'eff_ch':>6} {'key_part':>8}  flags")
    for k in KEYS:
        for s in seeds:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<5} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<5} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            fl = (["DISCOVERED"] if r["discovered"] else []) + (["collapsed"] if r["collapsed"] else [])
            kp = f"{e['key_part']:>8.4f}" if "key_part" in e else f"{'--':>8}"
            print(f"  {k:<5} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                  f"{e['val_cos']:>8.4f} {e['sep']:>7.4f} {e['spread']:>7.4f} {e['key_sep']:>7.4f} "
                  f"{e['eff_ch']:>6.3f} {kp}  {' '.join(fl)}")
        print()

    print("#" * 100)
    print(f"PER-TRIAL RAW RESULTS (Part 2) — check at step {T_CHECK}: continue if held-out accuracy "
          f">= {A_CHECK}")
    print("#" * 100)
    for i, tr in enumerate(trials):
        if tr is None:
            print(f"  trial {i:>2}  NOT RUN")
            continue
        for j, at in enumerate(tr["attempts"]):
            res = ("PASS -> continued" if at["passed"] else "fail -> abandoned" if at["passed"] is False
                   else "ERROR")
            print(f"  trial {i:>2}  attempt {j + 1}  seed {at['seed']}  acc@{EVAL_EVERY} "
                  f"{fmt_acc(at['accs'].get(str(EVAL_EVERY)))}  acc@{T_CHECK} "
                  f"{fmt_acc(at['accs'].get(str(T_CHECK)))}  {res}")
            r = at.get("rec")
            if at["passed"] and r and r.get("ok"):
                e = r["end"]
                print(f"  {'':<22}continued: acc {r['acc']:.4f}  transition {fmt_step(r['transition'])}"
                      f"  VAL cos {e['val_cos']:.4f}  sep {e['sep']:.4f}  key_part {e['key_part']:.4f}"
                      f"  stopped at {r['stopped_at']}{'  DISCOVERED' if r['discovered'] else ''}")
        print(f"  trial {i:>2}  {'SUCCESS' if tr['success'] else 'failed'}   attempts used {tr['used']}   "
              f"training steps {tr['steps']}" + (f"   [ERROR: {tr['error']}]" if tr.get("error") else ""))
        print()

    print("=" * 100)
    print(f"COUNTS (discovered = bound AND final VAL cos < {DISC_COS})")
    print("=" * 100)
    c = {}
    for a in ARMS:
        k = a["key"]
        c[k] = sum(disc(store, k, s) for s in seeds)
        rs = [get(store, k, s) for s in seeds]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        print(f"  Part 1  {a['label']:<42} discovered {c[k]:>2}/{n}   bound "
              f"{count(store, k, seeds, 'transition')}/{n}   completed "
              f"{sum(1 for r in rs if r and r.get('ok'))}/{n}" + (f"   [{failed} FAILED]" if failed else ""))
    done = [tr for tr in trials if tr is not None]
    succ = sum(1 for tr in done if tr["success"])
    errs = sum(1 for tr in done if not tr["ok"])
    print(f"  Part 2  restart protocol (arm A, R_MAX={R_MAX})            trials succeeded {succ:>2}/{nt}"
          f"   completed {len(done)}/{nt}" + (f"   [{errs} with an ERROR]" if errs else ""))
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS")
    print("#" * 100)
    v = dict(counts=c, trials_succeeded=succ)
    for key, name in (("A_k8", "K8"), ("A_k4", "K4")):
        p = fisher_greater(c[key], n, c["A"], n)
        verdict = "HELPS" if p < ALPHA else "NOT SHOWN"
        print(f"  {name} MORE CHANNELS HELP (Fisher one-sided): {key} {c[key]}/{n} vs A {c['A']}/{n}; "
              f"p = {p:.3g}")
        print(f"     *** {name}: {verdict} ***")
        v[name], v[f"{name}_p"] = verdict, p
    print(f"  BANDS (RELIABLE >= {RELIABLE_1}/40, MAJORITY {MAJORITY_1}-{RELIABLE_1 - 1}, MINORITY "
          f"<= {MAJORITY_1 - 1}):")
    for k in KEYS:
        print(f"     *** {k:<5} {c[k]}/{n}: {band1(c[k])} ***")
        v[f"band_{k}"] = band1(c[k])
    print(f"  RESTART (RELIABLE >= {RELIABLE_R}/30, PARTIAL {PARTIAL_R}-{RELIABLE_R - 1}, NOT RELIABLE "
          f"<= {PARTIAL_R - 1}): {succ}/{nt} trials succeeded")
    print(f"     *** RESTART: {band2(succ)} ***   (a procedure: restarts chosen by held-out accuracy; "
          f"not the gate learning more reliably)")
    v["RESTART"] = band2(succ)
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  PART 1")
    print("  per-seed outcome (D = discovered, . = not):")
    print("    seed  " + "  ".join(f"{k:>5}" for k in KEYS))
    for s in seeds:
        print(f"    {s:>4}  " + "  ".join(f"{('D' if disc(store, k, s) else '.'):>5}" for k in KEYS))
    for k1 in ("A_k8", "A_k4"):
        b = sum(disc(store, k1, s) and not disc(store, "A", s) for s in seeds)
        cc = sum(disc(store, "A", s) and not disc(store, k1, s) for s in seeds)
        print(f"    McNemar exact two-sided, {k1} vs A: {k1} only {b}, A only {cc}; "
              f"p = {mcnemar_exact(b, cc):.3g}")
    print()
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}) per discovering run "
          f"(escape/transition):")
    for k in KEYS:
        es = [(s, escape(get(store, k, s)), get(store, k, s)["transition"]) for s in seeds
              if disc(store, k, s)]
        print(f"    {k:<5} " + ("  ".join(f"s{s}:{fmt_step(e)}/{fmt_step(tr)}" for s, e, tr in es)
                              if es else "none"))
    print()
    print(f"  gate statistics per arm, median [min, max] over completed runs, at the first "
          f"evaluation ({EVAL_EVERY}) and at the end:")
    for k in KEYS:
        rs = [r for s in seeds if (r := get(store, k, s)) and r.get("ok")]
        first = [next((x for x in r["stats"] if x["step"] == EVAL_EVERY), None) for r in rs]
        first = [x for x in first if x is not None]
        for when, xs in ((f"@{EVAL_EVERY}", first), ("end", [r["end"] for r in rs])):
            print(f"    {k:<5} {when:<6} " + "   ".join(f"{f} {med([x[f] for x in xs])}"
                                                    for f in ("spread", "sep", "key_sep", "eff_ch")))
    print()
    print(f"  failure modes of non-discovering runs (saddle: spread < {SADDLE_SPREAD}; key split: "
          f"key_sep > {KEY_SPLIT} and sep < {LOCK_SEP}; other):")
    for k in KEYS:
        modes = {"saddle": 0, "key split": 0, "other": 0}
        for s in seeds:
            r = get(store, k, s)
            if r and r.get("ok") and not r["discovered"]:
                modes[mode_k(r)] += 1
        print(f"    {k:<5} " + "   ".join(f"{m}: {x}" for m, x in modes.items()))
    old = {"saddle": 0, "locked on keys": 0, "other": 0}
    for s in seeds:
        r = get(store, "A", s)
        if r and r.get("ok") and not r["discovered"]:
            old[mode_of(r)] += 1
    print(f"    A (k=2) in the old classification (locked on keys: key_part > 0.8 and sep < 0.2): "
          + "   ".join(f"{m}: {x}" for m, x in old.items()))
    print()
    print("  key_sep at the end of discovered runs (do they split keys as well as streams?):")
    for k in KEYS:
        ks = [(s, get(store, k, s)["end"]["key_sep"]) for s in seeds if disc(store, k, s)]
        print(f"    {k:<5} median {med([x for _, x in ks])}   "
              + " ".join(f"s{s}:{x:.2f}" for s, x in ks))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; rest = all other parameters), median [min, max] "
          "over seeds:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [r["grad"][str(step)] for s in seeds
                 if (r := get(store, k, s)) and r.get("ok") and str(step) in r.get("grad", {})]
            if not g:
                continue
            ga, re = [x[0] for x in g], [x[1] for x in g]
            print(f"    step {step:>3} {k:<5} gate {statistics.median(ga):.3e} [{min(ga):.3e}, "
                  f"{max(ga):.3e}]   rest {statistics.median(re):.3e} [{min(re):.3e}, {max(re):.3e}]")
    print()
    print("  runs that bound but did not discover:")
    for k in KEYS:
        bd = [(s, get(store, k, s)) for s in seeds
              if (r := get(store, k, s)) and r.get("ok") and r["transition"] is not None
              and not r["discovered"]]
        print(f"    {k:<5} " + ("  ".join(f"s{s} (bound @{r['transition']}, VAL cos "
                                          f"{r['val_cos']:.3f})" for s, r in bd) if bd else "none"))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, seeds, 'collapsed')}" for k in KEYS))
    print()

    print("  PART 2 (a procedure: each attempt is an arm-A run)")
    ats = [at for tr in done for at in tr["attempts"] if at["passed"] is not None]
    npass = sum(1 for at in ats if at["passed"])
    print(f"  per-attempt pass rate at the check: {npass}/{len(ats)}"
          + (f" = {npass / len(ats):.3f}" if ats else ""))
    conts = [at for tr in done for at in tr["attempts"] if at["passed"] and at.get("rec")]
    miss = [at for at in conts if not (at["rec"].get("ok") and at["rec"]["discovered"])]
    print(f"  the check's precision: continued attempts that did not discover: {len(miss)}/{len(conts)}"
          + (" (" + ", ".join(f"seed {at['seed']}" for at in miss) + ")" if miss else ""))
    a_runs = [(s, r) for s in seeds if (r := get(store, "A", s)) and r.get("ok")]
    would = [(s, r) for s, r in a_runs if (x := curve_acc(r, T_CHECK)) is not None and x >= A_CHECK]
    late = [s for s, r in a_runs if r["discovered"] and (curve_acc(r, T_CHECK) or 0.0) < A_CHECK]
    p_hat = len(would) / len(a_runs) if a_runs else float("nan")
    pred = 1 - (1 - p_hat) ** R_MAX if a_runs else float("nan")
    print(f"  predicted vs observed, from Part 1's arm A ({len(a_runs)} runs):")
    print(f"    would pass the check (acc@{T_CHECK} >= {A_CHECK}): {len(would)}/{len(a_runs)}, p_hat = "
          f"{p_hat:.3f}; of those, discovered {sum(r['discovered'] for _, r in would)}/{len(would)}")
    print(f"    discoveries the check would have abandoned (acc@{T_CHECK} < {A_CHECK}): {len(late)}"
          + (" (" + ", ".join(f"s{s}" for s in late) + ")" if late else ""))
    print(f"    predicted trial success 1 - (1 - p_hat)^{R_MAX} = {pred:.3f}   observed "
          f"{succ}/{len(done)}" + (f" = {succ / len(done):.3f}" if done else ""))
    steps = [tr["steps"] for tr in done]
    a_steps = [r["stopped_at"] for _, r in a_runs]
    if steps and a_steps:
        print(f"  cost: training steps per trial mean {statistics.fmean(steps):.0f}, max {max(steps)}; "
              f"Part 1 arm A mean steps per run {statistics.fmean(a_steps):.0f} (ratio "
              f"{statistics.fmean(steps) / statistics.fmean(a_steps):.2f})")
    used = [tr["used"] for tr in done]
    if used:
        print(f"  attempts used per trial: " + "  ".join(f"{u}: {used.count(u)}"
                                                     for u in range(1, R_MAX + 1)))
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for k in KEYS:
                runs = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                k2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                print(f"    {k:<5} discovered here {c[k]}/{n}   other {k2}/{len(runs)}   pooled "
                      f"{c[k] + k2}/{n + len(runs)}")
            ot = [tr for tr in other.get("trials", {}).values()]
            o2 = sum(1 for tr in ot if tr.get("success"))
            print(f"    trials succeeded here {succ}/{len(done)}   other {o2}/{len(ot)}   pooled "
                  f"{succ + o2}/{len(done) + len(ot)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — accuracy x100 and VAL cos x100 at each evaluation every {EVAL_EVERY} steps "
          f"('^' marks the check at {T_CHECK} for Part 2's continued attempts)")
    print("=" * 100)
    for k in KEYS:
        for s in seeds:
            r = get(store, k, s)
            if r and r.get("ok"):
                curve_lines(f"{k:<5} s{s}", r)
    for i, tr in enumerate(trials):
        for at in (tr or {}).get("attempts", []):
            r = at.get("rec")
            if at["passed"] and r and r.get("ok"):
                curve_lines(f"trial {i:>2} s{at['seed']}", r, mark_at=T_CHECK)
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="More channels, and a restart rule, for arm A.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's router_reliability_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    t_seeds = [x for i in range(TRIALS) for x in trial_seeds(i)]

    print("=" * 100)
    print("Router reliability: more channels (Part 1), and a restart rule on held-out accuracy (Part 2)")
    print(f"  test_router_confirm's arm A (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{SUB_LR}), MAX_ITERS={MAX_ITERS}")
    print(f"  Part 1: k = 2, 4, 8; seeds {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} per arm, paired)")
    print(f"  Part 2: {TRIALS} trials x up to {R_MAX} attempts, seeds {t_seeds[0]}-{t_seeds[-1]}; "
          f"check at step {T_CHECK}: continue if held-out accuracy >= {A_CHECK}")
    print(f"  all seeds disjoint from earlier seeds 0-79: "
          f"{not (set(SEEDS) | set(t_seeds)) & set(EARLIER_SEEDS)}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_store(args.results)
        store.setdefault("trials", {})
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, seeds=list(SEEDS), trials=TRIALS,
                             r_max=R_MAX, t_check=T_CHECK, a_check=A_CHECK,
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step (200 steps) inside the pool, evaluation charged once per "
              f"{EVAL_EVERY}; worst case = every run to {MAX_ITERS}, every trial "
              f"{R_MAX - 1} x {T_CHECK} + {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_arm, [ARM[k] for k in KEYS])))
        for k in KEYS:
            print(f"  {k:<5} {costs[k][0] * 1000:6.1f} ms/step   x {len(SEEDS)} seeds")
        trial_steps = max(R_MAX * T_CHECK, (R_MAX - 1) * T_CHECK + MAX_ITERS)
        print(f"  trial {costs['A'][0] * trial_steps / 60:6.1f} min worst case   x {TRIALS} trials")
        cost_of = lambda sp: (costs["A"][0] * trial_steps if sp["part"] == 2
                              else costs[sp["arm"]][0] * sp["iters"])
        allspecs = [spec1(k, s, MAX_ITERS) for k in KEYS for s in SEEDS] + \
                   [spec2(i, MAX_ITERS) for i in range(TRIALS)]
        durs = [cost_of(sp) for sp in allspecs]
        print(f"  {len(durs)} jobs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(SEEDS) * len(KEYS)} Part 1 runs and {TRIALS} Part 2 trials")
        print("=" * 100)
        todo = []
        for sp in allspecs:
            cached = (get(store, sp["arm"], sp["seed"]) if sp["part"] == 1
                      else store["trials"].get(str(sp["trial"])))
            if cached is not None and not args.force:
                print(f"   {'trial ' + str(sp['trial']) if sp['part'] == 2 else sp['arm'] + ' seed ' + str(sp['seed'])}"
                      f"  cached")
            else:
                todo.append(sp)
        todo.sort(key=lambda sp: -cost_of(sp))
        futs = {pool.submit(run_job, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            el = (time.time() - t0) / 60
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, error=f"worker {type(e).__name__}: {e}")
                if sp["part"] == 1:
                    rec["seed"] = sp["seed"]
                else:
                    rec.update(trial=sp["trial"], attempts=[], used=0, success=False, steps=0,
                               continued=None)
            if sp["part"] == 1:
                store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
                if rec.get("ok"):
                    e, tr = rec["end"], rec["transition"]
                    print(f"   {sp['arm']:<5} seed {sp['seed']}  "
                          f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                          f"@{rec['stopped_at']:<5}  VALcos={e['val_cos']:.3f} sep={e['sep']:.3f} "
                          f"key_sep={e['key_sep']:.3f} eff_ch={e['eff_ch']:.2f}"
                          f"{'  DISCOVERED' if rec['discovered'] else ''}  {rec['secs']:.0f}s  "
                          f"(elapsed {el:.1f}m)")
                else:
                    print(f"   {sp['arm']:<5} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                          f"(elapsed {el:.1f}m)")
            else:
                store["trials"][str(sp["trial"])] = rec
                ats = rec.get("attempts", [])
                path = " ".join(f"{at['seed']}:{fmt_acc(at['accs'].get(str(T_CHECK)))}"
                                f"{'+' if at['passed'] else '-'}" for at in ats)
                print(f"   trial {sp['trial']:>2}  {'SUCCESS' if rec.get('success') else 'failed':<7}  "
                      f"attempts {rec.get('used', 0)} [{path}]  steps {rec.get('steps', 0)}"
                      + (f"  *** ERROR: {rec['error']}" if rec.get("error") else "")
                      + f"  {rec.get('secs', 0):.0f}s  (elapsed {el:.1f}m)")
            save_results(args.results, store)
        print()

    report(store, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
