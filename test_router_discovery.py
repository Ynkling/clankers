#!/usr/bin/env python
"""
test_router_discovery.py — can the learned channel gate discover the stream routing from the
task loss alone, reliably, without being given the stream labels?

Run it directly:

    python test_router_discovery.py                 # CHECKs, projection, 6 arms x 10 seeds
    python test_router_discovery.py --workers 2 --force

test_channel_binding.py's NOT SUPPORTED stands; this test does not revise it. It asks one
narrower question about why the channel arms failed there.

BACKGROUND
- test_channel_binding gave NOT SUPPORTED on both machines at P=4, the only config that
  passed Gates 0 and 1 on both. Pooled over both machines: perfect gate 10/10 bound,
  single-channel controls 0/50, learned channels (A, A_ro) 1/20. The one success was on the
  other machine (i7-12650H, run at 7cf7d5a): A_ro seed 4 bound at 3600 with VAL cos 0.0009.
  On that machine A seed 3 also collapsed to 0.085 (loss ~ ln 16).
- Mechanism, from a scratch diagnostic on arm A, P=4 (not in the repo):
  - G_ts = g_t . g_s with a softmax gate. For k=2 write g_t = (p_t, 1-p_t). Then
    dG_ts/dp_t = 2 p_s - 1, exactly 0 when every p_s = 0.5. The uniform gate is a saddle:
    the task loss has no first-order pull toward routing.
  - The gate reads the raw embeddings, which bdh.py initialises at std 0.02 (the one-layer
    instrument used std 1), so the gate starts within +-0.002 of uniform. At step 1 its
    gradient norm was ~1e-5, against ~2 for the rest of the model.
  - The labelled nudge: pre-fitting the gate toward the correct stream by only 52.5/47.5,
    then training normally. The task loss amplified the lean; seeds 0, 1 and 2 bound at
    2400, 2400 and 7200.
  - Non-oracle quick tries (seed 0 only, short runs): LayerNorm on the gate input moved the
    gate off the saddle but hard-partitioned the four key tokens by identity
    (p ~ 0/1/1/0); stream separation stayed ~0.03 and accuracy 0.50. Freezing the gate for
    6000 steps then training, and Gumbel writes, showed no stream separation within the
    steps run.

SUBSTRATE AND TASK: test_channel_binding's P=4 config, reused, not rewritten.
BindTask P=4, S=2, n_vals=16, n_q=1 (test_channel_binding.task_for(4, 2)); MultiBDH n_layer=3,
decay, N=256 via build(); Adam lr 1e-3, BATCH 32, warmup 0; onset_run, held-out evaluation
every 1200 steps on eval_batch(task), early stop after 3 evaluations >= 0.95. MAX_ITERS =
19200, a pre-registered scope: on both machines every P=4 S=2 run that bound did so by 7200,
and none of the 79 runs still unbound at step 13200 bound later. Seeds 0-9.

ARMS, 10 seeds each
  A_ro        test_channel_binding's A_ro exactly. The baseline rate, never a candidate.
  A_ro_lr10   candidate: Adam groups {W_in, W_h, W_g} at lr 1e-2, everything else (W_ro and
              the embedding included) at 1e-3.
  A_ro_ln     candidate: the gate reads F.layer_norm(embed(tokens), (D,)), no affine; the BDH
              stack is untouched (BDH.forward embeds tokens itself).
  A_ro_noise  candidate: in training forwards only, gate = softmax(log(gr) + sigma*eps),
              sigma = 0.5, eps iid N(0,1) per (sequence, position, channel) from the model's
              own generator seeded seed + 77_777; read and write use the same noisy gate.
  A_ro_nudge  POSITIVE CONTROL, uses the stream labels, never a candidate: inside
              make_model, pre-fit only {W_in, W_h, W_g} for 300 Adam steps at lr 1e-2 on MSE
              between Instrument.gates' read gate and 0.475 + 0.05 * perfect_gate_general,
              batches of 64 from a generator seeded seed + 55_555; then train normally.
  B           plain single channel, a reference.
Fixed before any result: sigma = 0.5, the x10 multiplier, the 5% nudge, MAX_ITERS, the seeds.

PER RUN: test_channel_binding's fields (curve, transition, stopped_at, acc, KEY/VAL/CTX cos
at the end). For channel arms, gate stats at step 0, after every evaluation and at the end,
on probe_batch(task, seed), always under model.eval() and torch.no_grad(); p = read gate on
channel 0:
  sep       |mean p over stream-0 positions - mean p over stream-1 positions| (1 = routed)
  spread    2 * mean |p - 0.5|                                    (0 = uniform, 1 = hard)
  key_part  mean over the four key tokens of 2 * |mean p at that token - 0.5|
            (1 = the gate partitions by key identity; routing by stream gives ~0)
  VAL cos   role_cosines at each evaluation
collapsed = final accuracy < 0.15.   DISCOVERED = transition is not None AND final VAL cos < 0.5.

PRE-REGISTERED VERDICT, exactly one of
  INVALID       A_ro_nudge discovers on fewer than 8/10 seeds: the saddle account does not
                reproduce on this machine, and no candidate is interpreted.
  ROUTER FOUND  some candidate (A_ro_lr10, A_ro_ln, A_ro_noise) discovers on >= 8/10.
  PARTIAL       the best candidate discovers on 3-7/10.
  NOT FOUND     every candidate discovers on <= 2/10.
A_ro's count is printed beside the candidates as the baseline (it pools with
test_channel_binding's A_ro, 1/10 over both machines); it is never a candidate. Any B run
that binds is printed prominently: a single channel would escape the 0.5 plateau.

DIAGNOSTICS, not part of the verdict: escape step (first evaluation with VAL cos < 0.5) vs
body step (first evaluation with accuracy >= 0.45) per discovering run; failure mode of each
non-discovering channel run from its final gate stats ("saddle": spread < 0.1; "locked on
keys": key_part > 0.8 and sep < 0.2; "other"); collapsed runs per arm; and, if
channel_binding_results.json from this same CPU is present, whether A_ro seeds 0-4 reproduce
its curves exactly up to 19200.

LOGISTICS. Parallel single-thread workers as in test_binding_recipe.py; wall clock projected
before training (no drop rule). Results persist atomically to router_discovery_results.json
(gitignored). Seed-level outcomes differ between machines.

RESULT (full run, 60/60 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz,
4 workers x 1 thread; 87 min after verification).
  PRE-REGISTERED VERDICT: INVALID. The positive control A_ro_nudge discovered on 2/10
  (seed 3 bound at 4800; seed 5 only at its final evaluation, 0.961 at 19200, VAL cos
  0.238), fewer than 8/10: the saddle account does not reproduce on this machine and no
  candidate is interpreted. Counts, for the record: A_ro_lr10 0/10, A_ro_ln 2/10 (seeds 0
  and 9), A_ro_noise 0/10; baseline A_ro 1/10 (seed 8, bound at 6000, VAL cos 0.009).
  B bound 0/10. No run collapsed.

  Diagnostics (not part of the verdict):
  - The nudge's lean was erased, not amplified. Every nudged seed starts at sep 0.050; by
    the first evaluation (step 1200) 8/10 are at sep <= 0.015 while spread has jumped from
    0.05 to 0.44-0.89. The gate does not sit at the uniform saddle on this machine: in
    every channel arm it goes hard within the first 1200 steps (A_ro: spread 0.00 -> 0.23-
    0.88), in a direction that is not the stream.
  - Final gate states of the 45 non-discovering channel runs: "saddle" (spread < 0.1) 1,
    "locked on keys" (key_part > 0.8, sep < 0.2) 15, "other" 29. Final spread is 0.59-1.00
    in all of them but one (A_ro_lr10 seed 7, 0.08). The failure is a hard gate routing by
    something other than the stream, not a gate stuck at uniform.
  - Discovery, when it happened, happened early and abruptly: escape (VAL cos < 0.5) at
    1200 for both A_ro_ln seeds, 3600 for nudge seed 3, 4800 for A_ro seed 8, each at or
    before the body step (accuracy >= 0.45), except nudge seed 5 (escape 19200, body 9600).
  - A_ro seeds 0-4 reproduce test_channel_binding's curves exactly for all 16 evaluations
    up to step 19200 (this also shows on_eval does not perturb training). With those five,
    A_ro on this machine is 1/10 over seeds 0-9; the other machine's A_ro was 1/5.
  - The scratch diagnostic's nudge successes (seeds 0, 1, 2 at 2400, 2400, 7200) were on
    arm A on the other machine; here, on A_ro, seeds 0, 1 and 2 stayed at 0.48-0.51.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

from bdh import BDH
import test_multilayer_binding as tmb
from test_multilayer_binding import build, probe_batch, role_cosines, rkey, save_results, D
from test_instrument_v2 import Instrument, perfect_gate_general, TAU_END
from test_binding_onset import (
    onset_run, evaluate, transition, attempt, time_per_step, fmt_step, BIND_PASS, STOP_AFTER,
    EVAL_EVERY,
)
import test_channel_binding as tcb
from test_channel_binding import task_for, eval_batch, ARCH, SUB_LR, arms as cb_arms
from test_binding_recipe import makespan, cpu_model, same_weights

# ── Settings (fixed before any result) ───────────────────────────────────────
P, S = 4, 2
MAX_ITERS = 19200
SEEDS = tuple(range(10))
SIGMA = 0.5
FAST_MULT = 10
NUDGE_STEPS, NUDGE_LR, NUDGE_B = 300, 1e-2, 64
NUDGE_BASE, NUDGE_AMP = 0.475, 0.05
NOISE_OFFSET, NUDGE_OFFSET = 77_777, 55_555
DISC_COS = 0.5
FOUND_K, PARTIAL_K = 8, 3
COLLAPSE = 0.15
BODY_ACC = 0.45
SADDLE_SPREAD, LOCK_KEYPART, LOCK_SEP = 0.1, 0.8, 0.2
GATE_PARAMS = ("W_in", "W_h", "W_g")
LEGACY_SHA = "66754b2"
WORKERS = min(4, os.cpu_count() or 1)
TIME_STEPS = 200
RESULTS_FILE = "router_discovery_results.json"
CB_RESULTS = "channel_binding_results.json"
CANDIDATES = ("A_ro_lr10", "A_ro_ln", "A_ro_noise")

TASK = task_for(P, S)
_BASE = {a["key"]: a for a in cb_arms(TASK)}
ARMS = [
    dict(_BASE["A_ro"], key="A_ro", label="A_ro        baseline (test_channel_binding)"),
    dict(_BASE["A_ro"], key="A_ro_lr10", label="A_ro_lr10   gate params at lr x10",
         lr10=True),
    dict(_BASE["A_ro"], key="A_ro_ln", label="A_ro_ln     LayerNorm on the gate input",
         model_kw=dict(gate_ln=True)),
    dict(_BASE["A_ro"], key="A_ro_noise", label=f"A_ro_noise  gate-logit noise sigma={SIGMA}",
         noise=SIGMA),
    dict(_BASE["A_ro"], key="A_ro_nudge", label="A_ro_nudge  POSITIVE CONTROL (labels)",
         nudge=True),
    dict(_BASE["B"], key="B", label="B           plain single channel (reference)"),
]
ARM = {a["key"]: a for a in ARMS}
CHANNEL_KEYS = tuple(a["key"] for a in ARMS if a["n_ch"] > 1)


# ── The arms' pieces ─────────────────────────────────────────────────────────
def fast_groups(model):
    """A_ro_lr10: the gate's own weights at lr x10; every other parameter at the base lr."""
    named = list(model.named_parameters())
    fast = [p for n, p in named if n in GATE_PARAMS]
    rest = [p for n, p in named if n not in GATE_PARAMS]
    return [dict(params=fast, lr=SUB_LR * FAST_MULT), dict(params=rest)]


def nudge(model, task, seed):
    """A_ro_nudge's labelled pre-fit of the gate toward 52.5/47.5 routing."""
    g = torch.Generator().manual_seed(seed + NUDGE_OFFSET)
    opt = torch.optim.Adam([getattr(model, n) for n in GATE_PARAMS], lr=NUDGE_LR)
    for _ in range(NUDGE_STEPS):
        x = task.make_batch(NUDGE_B, g)[0][:, :-1]
        target = NUDGE_BASE + NUDGE_AMP * perfect_gate_general(x, task.ctx_tokens)
        gr, _ = Instrument.gates(model, x, model.embed(x))
        loss = F.mse_loss(gr, target)
        opt.zero_grad(); loss.backward(); opt.step()
    model.zero_grad(set_to_none=True)


def model_fn(task, a, seed, sigma=None):
    """make_model for onset_run: build(), plus the arm's noise seed or labelled nudge."""
    def make():
        aa = dict(a)
        s = a.get("noise") if sigma is None else sigma
        if s is not None:
            aa["model_kw"] = dict(aa.get("model_kw", {}), gate_noise=s,
                                  noise_seed=seed + NOISE_OFFSET)
        m = build(task, aa, ARCH, seed)
        if a.get("nudge"):
            nudge(m, task, seed)
        return m
    return make


def run_kw(a):
    return dict(lr=SUB_LR, param_groups=fast_groups if a.get("lr10") else None)


@torch.no_grad()
def gate_stats(model, task, probe, step):
    """sep, spread, key_part and the per-role cosines, noise-free (eval mode)."""
    was = model.training
    model.eval()
    try:
        pinp, plab, _ = probe
        p = model(pinp, TAU_END)[2][..., 0]
        sep = (p[plab == 0].mean() - p[plab == 1].mean()).abs().item()
        spread = (2 * (p - 0.5).abs().mean()).item()
        kp = [2 * abs(p[pinp == k].mean().item() - 0.5) for k in range(task.S, task.S + task.P)]
        d = role_cosines(model, probe, step)
    finally:
        model.train(was)
    return dict(step=step, sep=sep, spread=spread, key_part=sum(kp) / len(kp),
                key_cos=d["key_cos"], val_cos=d["val_cos"], ctx_cos=d["ctx_cos"])


# ── Worker-side ──────────────────────────────────────────────────────────────
def init_worker():
    torch.set_num_threads(1)


def run_spec(sp):
    """One arm, one seed: onset_run's recipe, with gate stats for channel arms."""
    task, a, seed = TASK, ARM[sp["arm"]], sp["seed"]
    chan = a["n_ch"] > 1
    probe = probe_batch(task, seed) if chan else None
    make, stats = model_fn(task, a, seed), []

    def make_and_measure():
        m = make()
        if chan:
            stats.append(gate_stats(m, task, probe, 0))
        return m

    def on_eval(m, step):
        stats.append(gate_stats(m, task, probe, step))

    def post(m):
        if not chan:
            return {}
        end = gate_stats(m, task, probe, "end")
        return dict(key_cos=end["key_cos"], val_cos=end["val_cos"], ctx_cos=end["ctx_cos"],
                    end=end)

    rec = attempt(task, make_and_measure, seed, eval_batch(task), post=post,
                  max_iters=sp["iters"], on_eval=on_eval if chan else None, **run_kw(a))
    rec["stats"] = stats
    rec["iters"] = sp["iters"]
    if rec["ok"]:
        rec["acc"] = rec["curve"][-1][1] if rec["curve"] else float("nan")
        rec["collapsed"] = rec["acc"] < COLLAPSE
        rec["discovered"] = bool(chan and rec["transition"] is not None
                                 and rec["val_cos"] < DISC_COS)
    return rec


def time_arm(key):
    """ms/step with evaluation charged once per EVAL_EVERY, plus a fixed per-run cost (the
    nudge pre-fit), each timed on its own."""
    a = ARM[key]
    t0 = time.time()
    model_fn(TASK, a, 0)()
    fixed = time.time() - t0 if a.get("nudge") else 0.0
    per = time_per_step(TASK, model_fn(TASK, a, 0), steps=TIME_STEPS)
    m, data = model_fn(TASK, a, 0)(), eval_batch(TASK)
    t0 = time.time()
    evaluate(m, TASK, data)
    ev = time.time() - t0
    return max(per - (fixed + ev) / TIME_STEPS, 1e-6) + ev / EVAL_EVERY, fixed


# ── Verification ─────────────────────────────────────────────────────────────
def load_at(sha, fname, modname):
    src = subprocess.run(["git", "-C", HERE, "show", f"{sha}:{fname}"],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType(modname)
    mod.__file__ = os.path.join(HERE, fname)
    exec(compile(src, f"{fname}@{sha}", "exec"), mod.__dict__)
    return mod


def load_legacy_pair():
    """test_multilayer_binding and test_binding_onset at LEGACY_SHA, the onset module bound to
    the legacy multilayer module (not to the current one)."""
    leg_ml = load_at(LEGACY_SHA, "test_multilayer_binding.py", "test_multilayer_binding")
    saved = sys.modules["test_multilayer_binding"]
    sys.modules["test_multilayer_binding"] = leg_ml
    try:
        leg_on = load_at(LEGACY_SHA, "test_binding_onset.py", "test_binding_onset_legacy")
    finally:
        sys.modules["test_multilayer_binding"] = saved
    return leg_ml, leg_on


def strip(r):
    return {k: v for k, v in r.items() if k != "secs"}


def verify(pool):
    print("test_channel_binding.py's verification (which runs test_binding_onset's, which runs")
    print("test_multilayer_binding's):")
    tcb.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    t = TASK
    data = eval_batch(t)

    print(f"CHECK 19 every new knob is inert at its default: 60-step runs (evaluations every 20)")
    print(f"         through the new code vs {LEGACY_SHA}'s modules:")
    leg_ml, leg_on = load_legacy_pair()
    print(f"         legacy MultiBDH is a distinct class: {leg_ml.MultiBDH is not tmb.MultiBDH}   "
          f"legacy onset_run takes on_eval: "
          f"{'on_eval' in leg_on.onset_run.__code__.co_varnames}")
    for key in ("A_ro", "B"):
        a = ARM[key]
        m0, c0, _ = leg_on.onset_run(t, lambda: leg_ml.build(t, a, ARCH, 3), 3, max_iters=60,
                                     eval_every=20, data=data, early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(t, model_fn(t, a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, **run_kw(a))
        good = same_weights(m0, m1) and c0 == c1
        ok &= good
        print(f"         {key:<5} weights equal {same_weights(m0, m1)}   curves equal {c0 == c1}  "
              f"accs {[round(x, 4) for _, x, _ in c1]}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    m = build(t, ARM["A_ro"], ARCH, 0)
    good = m.gate_ln is False and m.gate_noise == 0.0 and m.noise_gen is None
    ok &= good
    print(f"         defaults: gate_ln={m.gate_ln} gate_noise={m.gate_noise} noise_gen="
          f"{m.noise_gen}  -> {'INERT' if good else 'WRONG'}")
    print()

    print("CHECK 20 A_ro_lr10's optimizer, as onset_run built it (captured by a step pre-hook):")
    a = ARM["A_ro_lr10"]
    holder, seen = [], []
    make = model_fn(t, a, 0)

    def mk():
        holder.append(make())
        return holder[-1]

    h = register_optimizer_step_pre_hook(lambda opt, args, kw: seen.append(
        [(g["lr"], [id(p) for p in g["params"]]) for g in opt.param_groups]))
    try:
        onset_run(t, mk, 0, max_iters=1, **run_kw(a))
    finally:
        h.remove()
    names = {id(p): n for n, p in holder[0].named_parameters()}
    groups = seen[0]
    for i, (lr, ids) in enumerate(groups):
        print(f"         group {i}: lr={lr:g}  {len(ids)} params: {sorted(names[x] for x in ids)}")
    good = (len(groups) == 2 and {names[x] for x in groups[0][1]} == set(GATE_PARAMS)
            and groups[0][0] == SUB_LR * FAST_MULT and groups[1][0] == SUB_LR
            and {names[x] for x in groups[1][1]} == set(names.values()) - set(GATE_PARAMS)
            and "W_ro" in {names[x] for x in groups[1][1]}
            and "embed.weight" in {names[x] for x in groups[1][1]})
    ok &= good
    print(f"         -> {'TWO GROUPS, fast = exactly {W_in, W_h, W_g}' if good else 'WRONG'}")
    print()

    print("CHECK 21 A_ro_ln changes only the gate's input:")
    ma, ml = build(t, ARM["A_ro"], ARCH, 5), build(t, ARM["A_ro_ln"], ARCH, 5)
    pa, pl = dict(ma.named_parameters()), dict(ml.named_parameters())
    same_init = sorted(pa) == sorted(pl) and all(torch.equal(pa[n], pl[n]) for n in pa)
    x = t.make_batch(16, torch.Generator().manual_seed(21))[0][:, :-1]
    G = torch.rand(16, 1, x.shape[1], x.shape[1], generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ma.attn.G, ml.attn.G = G, G
        la, ll = BDH.forward(ma, x)[0], BDH.forward(ml, x)[0]
        ga, gl = ma(x, TAU_END)[2], ml(x, TAU_END)[2]
    good = same_init and torch.equal(la, ll) and not torch.equal(ga, gl)
    ok &= good
    pr = probe_batch(t, 0)
    s_a = gate_stats(build(t, ARM["A_ro"], ARCH, 0), t, pr, 0)["spread"]
    s_l = gate_stats(build(t, ARM["A_ro_ln"], ARCH, 0), t, pr, 0)["spread"]
    print(f"         init parameters identical to A_ro's (seed 5): {same_init}   BDH.forward logits "
          f"with the same G forced: identical {torch.equal(la, ll)}   gates differ: "
          f"{not torch.equal(ga, gl)}")
    print(f"         step-0 spread (seed 0 probe): A_ro {s_a:.6f}   A_ro_ln {s_l:.6f}  -> "
          f"{'ONLY THE GATE INPUT CHANGES' if good else 'WRONG'}")
    print()

    print(f"CHECK 22 A_ro_noise (sigma={SIGMA}, own generator seeded seed + {NOISE_OFFSET}):")
    a = ARM["A_ro_noise"]
    m0, c0, _ = onset_run(t, model_fn(t, ARM["A_ro"], 4), 4, max_iters=60, eval_every=20,
                          data=data, early_stop=False, **run_kw(ARM["A_ro"]))
    m1, c1, _ = onset_run(t, model_fn(t, a, 4, sigma=0.0), 4, max_iters=60, eval_every=20,
                          data=data, early_stop=False, **run_kw(a))
    ga_ = same_weights(m0, m1) and c0 == c1
    mn, mp_ = model_fn(t, a, 6)(), model_fn(t, ARM["A_ro"], 6)()
    with torch.no_grad():
        mn.eval(); mp_.eval()
        e1, e2, ef = mn(x, TAU_END), mn(x, TAU_END), mp_(x, TAU_END)
        gb = (torch.equal(e1[0], e2[0]) and torch.equal(e1[0], ef[0])
              and torch.equal(e1[2], ef[2]))
        mn.train()
        tr = mn(x, TAU_END)
        gc = not torch.equal(tr[2], e1[2]) and not torch.equal(tr[0], e1[0])
    onset_run(t, model_fn(t, ARM["A_ro"], 7), 7, max_iters=1, **run_kw(ARM["A_ro"]))
    st_a = torch.get_rng_state()
    holder = []
    mk = model_fn(t, a, 7)
    onset_run(t, lambda: holder.append(mk()) or holder[-1], 7, max_iters=1, **run_kw(a))
    st_n = torch.get_rng_state()
    fresh = torch.Generator().manual_seed(7 + NOISE_OFFSET).get_state()
    gd = torch.equal(st_a, st_n) and not torch.equal(holder[0].noise_gen.get_state(), fresh)
    ok &= ga_ and gb and gc and gd
    print(f"         (a) sigma=0, 60 steps vs A_ro: weights equal {same_weights(m0, m1)}, curves "
          f"equal {c0 == c1}  -> {'IDENTICAL' if ga_ else 'DIFFERS'}")
    print(f"         (b) eval mode: two forwards identical and equal to the noise-free forward: {gb}")
    print(f"         (c) train mode: the forward differs (gate and logits): {gc}")
    print(f"         (d) global RNG state after one training step equals A_ro's: "
          f"{torch.equal(st_a, st_n)}   the noise generator was drawn from: "
          f"{not torch.equal(holder[0].noise_gen.get_state(), fresh)}")
    print()

    print(f"CHECK 23 A_ro_nudge (positive control): {NUDGE_STEPS} Adam steps at lr {NUDGE_LR} on "
          f"{{W_in, W_h, W_g}} toward {NUDGE_BASE} + {NUDGE_AMP} * perfect gate:")
    a = ARM["A_ro_nudge"]
    mz, mb = model_fn(t, a, 8)(), model_fn(t, ARM["A_ro"], 8)()
    sep0 = gate_stats(mz, t, probe_batch(t, 8), 0)["sep"]
    pz, pb = dict(mz.named_parameters()), dict(mb.named_parameters())
    others = [n for n in pz if n not in GATE_PARAMS]
    same_rest = all(torch.equal(pz[n], pb[n]) for n in others)
    moved = all(not torch.equal(pz[n], pb[n]) for n in GATE_PARAMS)
    fed = {}
    for key in ("A_ro", "A_ro_nudge"):
        rec, mkf = [], model_fn(t, ARM[key], 8)

        def mk2():
            mm = mkf()
            mm.register_forward_pre_hook(
                lambda mod, args: rec.append(args[0].clone()) if mod.training else None)
            return mm
        onset_run(t, mk2, 8, max_iters=3, **run_kw(ARM[key]))
        fed[key] = rec
    same_b = len(fed["A_ro"]) == 3 and all(torch.equal(u, v) for u, v in
                                            zip(fed["A_ro"], fed["A_ro_nudge"]))
    good = abs(sep0 - 0.05) <= 0.015 and same_rest and moved and same_b
    ok &= good
    print(f"         step-0 sep {sep0:.4f} (0.05 +- 0.015): {abs(sep0 - 0.05) <= 0.015}   "
          f"{len(others)} non-gate parameters bitwise equal to A_ro's init: {same_rest}   "
          f"gate parameters moved: {moved}")
    print(f"         first 3 training batches equal A_ro's: {same_b}  -> {'OK' if good else 'WRONG'}")
    print()

    print("CHECK 24 gate stats and cosines are measured noise-free: twice on an A_ro_noise model")
    print("         left in train mode:")
    mn = model_fn(t, ARM["A_ro_noise"], 9)()
    mn.train()
    pr = probe_batch(t, 9)
    s1, s2 = gate_stats(mn, t, pr, 0), gate_stats(mn, t, pr, 0)
    good = s1 == s2 and mn.training
    ok &= good
    print(f"         identical: {s1 == s2}   still in train mode afterwards: {mn.training}   "
          f"sep {s1['sep']:.6f} spread {s1['spread']:.6f} VAL cos {s1['val_cos']:.6f}  -> "
          f"{'NOISE-FREE' if good else 'WRONG'}")
    print()

    print("CHECK 25 a worker's run (record and gate stats) is bit-identical to the same run here:")
    sp = dict(arm="A_ro_noise", seed=1, iters=1200)
    rw = pool.submit(run_spec, sp).result()
    rh = run_spec(sp)
    good = strip(rw) == strip(rh) and rh["ok"] and len(rh["stats"]) == 2
    ok &= good
    print(f"         A_ro_noise seed 1, 1200 steps: equal {strip(rw) == strip(rh)}   stats "
          f"recorded {len(rh['stats'])} (step 0 + 1 evaluation)   VAL cos "
          f"{rh.get('val_cos', float('nan')):.6f}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def load_store(path):
    if not os.path.exists(path):
        return {"meta": {}, "runs": {}}
    try:
        with open(path) as f:
            d = json.load(f)
        d.setdefault("meta", {}); d.setdefault("runs", {})
        return d
    except Exception as e:
        print(f"  ! could not read {path} ({e}); starting fresh")
        return {"meta": {}, "runs": {}}


def get(store, key, seed):
    return store["runs"].get(f"{key}|{seed}")


def count(store, key, seeds, field):
    return sum(1 for s in seeds if (r := get(store, key, s)) and r.get("ok") and r.get(field))


def mode_of(r):
    e = r["end"]
    if e["spread"] < SADDLE_SPREAD:
        return "saddle"
    if e["key_part"] > LOCK_KEYPART and e["sep"] < LOCK_SEP:
        return "locked on keys"
    return "other"


def report(store, seeds, wall, path):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<11} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>7} "
          f"{'key_part':>8} {'spread':>7}  flags")
    for a in ARMS:
        for s in seeds:
            r = get(store, a["key"], s)
            if r is None:
                print(f"  {a['key']:<11} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {a['key']:<11} {s:>4}  FAILED — {r['error']}")
                continue
            e = r.get("end")
            fl = []
            if r.get("discovered"):
                fl.append("DISCOVERED")
            if a["key"] == "B" and r["transition"] is not None:
                fl.append("B BOUND")
            if r["collapsed"]:
                fl.append("collapsed")
            if e is None:
                print(f"  {a['key']:<11} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{'--':>8} {'--':>7} {'--':>8} {'--':>7}  {' '.join(fl)}")
            else:
                print(f"  {a['key']:<11} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{e['val_cos']:>8.4f} {e['sep']:>7.4f} {e['key_part']:>8.4f} "
                      f"{e['spread']:>7.4f}  {' '.join(fl)}")
        print()

    n = len(seeds)
    print("=" * 100)
    print(f"DISCOVERY COUNTS (discovered = bound AND final VAL cos < {DISC_COS}), of {n} seeds")
    print("=" * 100)
    disc = {}
    for a in ARMS:
        k = a["key"]
        rs = [get(store, k, s) for s in seeds]
        done = sum(1 for r in rs if r is not None and r.get("ok"))
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        bound = count(store, k, seeds, "transition")
        disc[k] = count(store, k, seeds, "discovered") if k in CHANNEL_KEYS else None
        role = ("candidate" if k in CANDIDATES else "positive control" if k == "A_ro_nudge"
                else "baseline" if k == "A_ro" else "reference")
        dtxt = "--" if disc[k] is None else f"{disc[k]}/{n}"
        print(f"  {a['label']:<46} {role:<17} discovered {dtxt:>6}   bound {bound}/{n}   "
              f"completed {done}/{n}" + (f"   [{failed} FAILED]" if failed else ""))
    print(f"  A_ro's count is the baseline and never a candidate; test_channel_binding's A_ro was "
          f"1/10 over both machines.")
    nb = count(store, "B", seeds, "transition")
    if nb:
        print()
        print("  " + "!" * 96)
        print(f"  !!! B BOUND ON {nb}/{n} SEEDS: a single channel escaped the 0.5 plateau within "
              f"{MAX_ITERS} steps !!!")
        print("  " + "!" * 96)
    print()

    print("#" * 100)
    print("PRE-REGISTERED VERDICT")
    print("#" * 100)
    print(f"  INVALID       A_ro_nudge discovers on fewer than {FOUND_K}/{n} seeds.")
    print(f"  ROUTER FOUND  some candidate discovers on >= {FOUND_K}/{n}.")
    print(f"  PARTIAL       the best candidate discovers on {PARTIAL_K}-{FOUND_K - 1}/{n}.")
    print(f"  NOT FOUND     every candidate discovers on <= {PARTIAL_K - 1}/{n}.")
    print()
    print(f"  A_ro_nudge (positive control) {disc['A_ro_nudge']}/{n};  candidates: "
          + ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES)
          + f";  baseline A_ro {disc['A_ro']}/{n}")
    best = max(disc[k] for k in CANDIDATES)
    if disc["A_ro_nudge"] < FOUND_K:
        verdict = "INVALID"
        why = (f"the positive control discovered on {disc['A_ro_nudge']}/{n} < {FOUND_K}: the "
               f"saddle account does not reproduce on this machine; no candidate is interpreted.")
    elif best >= FOUND_K:
        verdict = "ROUTER FOUND"
        why = ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES if disc[k] >= FOUND_K)
    elif best >= PARTIAL_K:
        verdict = "PARTIAL"
        why = "best candidate " + ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES
                                             if disc[k] == best)
    else:
        verdict = "NOT FOUND"
        why = f"every candidate discovered on <= {PARTIAL_K - 1}/{n}."
    print(f"  *** {verdict}: {why} ***")
    store["verdict"] = dict(verdict=verdict, counts=disc)
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}) vs body step (first "
          f"evaluation with accuracy >= {BODY_ACC}), per discovering run:")
    any_d = False
    for k in CHANNEL_KEYS:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok") and r.get("discovered")):
                continue
            any_d = True
            ev = [x for x in r["stats"] if x["step"] not in (0, "end")]
            esc = next((x["step"] for x in ev if x["val_cos"] < DISC_COS), None)
            body = next((st for st, acc, _ in r["curve"] if acc >= BODY_ACC), None)
            rel = ("escape BEFORE body" if esc is not None and body is not None and esc < body
                   else "same evaluation" if esc == body else "escape AFTER body")
            print(f"    {k:<11} seed {s}: escape {fmt_step(esc):>6}   body {fmt_step(body):>6}   "
                  f"transition {fmt_step(r['transition']):>6}   -> {rel}")
    if not any_d:
        print("    none: no run discovered.")
    print()
    print(f"  failure modes of non-discovering channel runs, from final gate stats (saddle: "
          f"spread < {SADDLE_SPREAD}; locked on keys: key_part > {LOCK_KEYPART} and sep < "
          f"{LOCK_SEP}; other):")
    for k in CHANNEL_KEYS:
        modes = {"saddle": 0, "locked on keys": 0, "other": 0}
        for s in seeds:
            r = get(store, k, s)
            if r and r.get("ok") and not r.get("discovered"):
                modes[mode_of(r)] += 1
        print(f"    {k:<11} " + "   ".join(f"{m}: {c}" for m, c in modes.items()))
    print()
    print(f"  collapsed runs (final accuracy < {COLLAPSE}), per arm: "
          + ", ".join(f"{a['key']} {count(store, a['key'], seeds, 'collapsed')}" for a in ARMS))
    print()
    print(f"  reproduction of test_channel_binding's A_ro, seeds 0-4, up to step {MAX_ITERS}:")
    if not os.path.exists(CB_RESULTS):
        print(f"    {CB_RESULTS} not present: skipped.")
    else:
        with open(CB_RESULTS) as f:
            cb = json.load(f)
        if cb.get("meta", {}).get("cpu") != cpu_model():
            print(f"    {CB_RESULTS} is from another CPU ({cb.get('meta', {}).get('cpu')}): "
                  f"skipped.")
        else:
            old = cb["runs"].get(rkey("P4", "A_ro", ARCH, tcb.MAX_ITERS), {})
            for s in range(5):
                r, o = get(store, "A_ro", s), old.get(str(s))
                if not (r and r.get("ok") and o and o.get("ok")):
                    print(f"    seed {s}: missing")
                    continue
                oc = [c for c in o["curve"] if c[0] <= MAX_ITERS]
                nn_ = min(len(oc), len(r["curve"]))
                same = nn_ > 0 and oc[:nn_] == r["curve"][:nn_]
                print(f"    seed {s}: {nn_} evaluations compared -> "
                      f"{'EXACT MATCH' if same else 'DIFFERS'}")
    print()

    print("=" * 100)
    print(f"CURVES — channel runs: accuracy x100 and VAL cos x100 at each evaluation every "
          f"{EVAL_EVERY} steps")
    print("=" * 100)
    for k in CHANNEL_KEYS:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok")):
                continue
            ev = {x["step"]: x for x in r["stats"] if x["step"] not in (0, "end")}
            accs = "".join(f"{round(acc * 100):>4}" for _, acc, _ in r["curve"])
            coss = "".join(f"{round(ev[st]['val_cos'] * 100):>4}" if st in ev else "  --"
                           for st, _, _ in r["curve"])
            print(f"  {k:<11} s{s} acc {accs}")
            print(f"  {'':<11}    cos {coss}   -> "
                  f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}"
                  f"{'  DISCOVERED' if r.get('discovered') else ''}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw "
          f"records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Can the gate discover the routing by itself?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    seeds = list(SEEDS)

    print("=" * 100)
    print("Router discovery: can the learned gate find the stream routing from the task loss?")
    print(f"  test_channel_binding's P=4 config: S={S} P={P} n_vals=16 n_q=1, n_layer=3 decay "
          f"N={8 * D} lr={SUB_LR}, MAX_ITERS={MAX_ITERS}, seeds {seeds}")
    print(f"  fixed before any result: sigma={SIGMA}, gate lr x{FAST_MULT}, nudge "
          f"{NUDGE_BASE}/{NUDGE_BASE + NUDGE_AMP}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x "
          f"1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, seeds=seeds, sigma=SIGMA,
                             fast_mult=FAST_MULT, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool, evaluation "
              f"charged once per {EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        keys = [a["key"] for a in ARMS]
        costs = dict(zip(keys, pool.map(time_arm, keys)))
        for k in keys:
            per, fixed = costs[k]
            print(f"  {k:<11} {per * 1000:6.1f} ms/step" + (f"   + {fixed:.1f}s pre-fit per run"
                                                            if fixed else ""))
        durs = [MAX_ITERS * costs[k][0] + costs[k][1] for k in keys for _ in seeds]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(ARMS)} arms x seeds {seeds}, MAX_ITERS={MAX_ITERS}")
        print("=" * 100)
        todo = []
        for k in keys:
            for s in seeds:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<11} seed {s}  cached")
                else:
                    todo.append(dict(arm=k, seed=s, iters=MAX_ITERS))
        todo.sort(key=lambda sp: -costs[sp["arm"]][0])
        futs = {pool.submit(run_spec, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            if rec.get("ok"):
                e = rec.get("end")
                gs = ("" if e is None else f"  VALcos={e['val_cos']:.3f} sep={e['sep']:.3f} "
                      f"key_part={e['key_part']:.3f}")
                tr = rec["transition"]
                print(f"   {sp['arm']:<11} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}{gs}{'  DISCOVERED' if rec.get('discovered') else ''}"
                      f"  {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<11} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, seeds, time.time() - t0, args.results)


if __name__ == "__main__":
    main()
