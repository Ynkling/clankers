#!/usr/bin/env python
"""
test_stream_channels.py — with S=4 streams, do spare channels (k=8, k=16) reduce the
two-stream merges that stall the k=4 gate? lr 1e-3 (SUB_LR, passed explicitly), conv 'layer'
width 4. Everything below is fixed before any result.

Run it directly:

    python test_stream_channels.py --prev results/X/scale_axes_results.json
    python test_stream_channels.py --prev ... --also results/L/stream_channels_results.json

BACKGROUND
test_scale_axes, both machines, seeds 220-239 (L ran 248c482):
- M1 SHOWN on both: DIRECT8 13/20 (X) and 12/20 (L), against B_wide8 0/12 on each (same
  fast-weight memory, 1.8x the parameters). B8: 0/10 (X), 1/10 (L).
- S1 MINORITY on both: A4k4 bound 9/20 (X) and 5/20 (L). B4s 0/10 on each.
- S2 SHOWN on X (p = 0.012), NOT SHOWN on L (p = 0.109).
- A4k4 outcomes, pooled over both machines:
  - 13 bound with a one-to-one stream -> channel map;
  - 1 bound with streams sharing a channel;
  - 13 failed with exactly two streams merged into one channel, stalled at 0.75;
  - 13 failed with a non-stream split.
- Routing arrives late and in steps (plateaus near 0.5 and 0.75; binders' median transition
  around 14400). No accuracy threshold at any step up to 12000 separated binders from
  failures well (post hoc, pooled).
- Motivating model, not a claim: if the 4 streams were assigned to k channels at random, a
  one-to-one map would occur 9% of the time at k=4, 41% at k=8 and 67% at k=16. The observed
  k=4 rate is above random, but merges look like early collisions that never separate.
- Seed-level outcomes differ between machines; the verdict is per machine.

DESIGN
- Task: task_for(4, 4) (S=4, P=4, n_vals=16, n_q=1). MAX_ITERS 28800, as A4k4. Run path:
  test_scale_axes's (run_one with its scale_stats: routing margin, multichannel eta^2 and the
  stream -> channel map at every evaluation, per-stream accuracy at the end), CHECK 86.
- Seeds 220-239 are reused on purpose: each run is paired with this machine's recorded A4k4
  run (the --prev file, default scale_axes_results.json; on X, results/X/scale_axes_results.
  json). As in test_router_reliability's CHECK 43, all parameters other than W_g are bitwise
  identical to A4k4's for the same seed, and so are the batches (CHECK 85).
- If the --prev file is missing, or its CPU does not match, or the recorded A4k4 run is not
  reproduced, the claims are UNTESTED.
Arms:
- A4k8:       arm A with k=8 channels + conv (test_scale_axes's A4k4 with n_ch=8). Seeds 220-239
- A4k16:      arm A with k=16 channels + conv (n_ch=16).                          Seeds 220-239
- ceiling4k8: perfect gate at k=8 (stream s -> channel s, channels 4-7 unused) + conv.
              Validity; runs first.                                               Seeds 220-222
  No new model code: it is test_scale_axes's ceiling4k4 (the existing perfect_gate_general)
  built with the context tokens (0, 1, 2, 3) padded by four token ids that never occur (-1 to
  -4), so the gate is one-hot over 8 channels and channels 4-7 stay at 0 (CHECK 87).
If ceiling4k8 binds fewer than 2/3, every claim is UNTESTED and nothing else runs.

DEFINITIONS (as in test_scale_axes, for any k)
- Stream -> channel map: the argmax of each stream's mean read gate at value positions.
- one_to_one: every stream has a different channel. shared: the number of streams on the most
  crowded channel.
- eff_ch, and the routing margin: g_q.g_tgt minus the mean over the other streams' bindings of
  g_q.g_dis. Routed = margin >= 0.9.
- MERGED: the run ends with at least two streams sharing a channel (shared >= 2; this
  includes runs with every stream on one channel).
- Channels used: channels carrying at least 5% of any stream's mean read-gate mass at value
  positions (from the recorded per-stream means).

CLAIMS (outcome BOUND)
- VALID: ceiling4k8 binds >= 2/3, the pairing CHECK (85) passes, and the k=8 and k=16 recurrent
  gates fit the perfect routing (CHECK 45's recipe, test_router_reliability.gate_fit_k: argmax
  >= 0.99).
- K8 EIGHT CHANNELS BIND MORE: A4k8 vs the recorded A4k4, same seeds. Exact McNemar, one-sided
  (A4k8 higher).
- K16 SIXTEEN CHANNELS BIND MORE: A4k16 vs the recorded A4k4, the same test.
- MG FEWER MERGES: the fraction of MERGED runs, A4k16 vs the recorded A4k4. Fisher one-sided
  (fewer at k=16).
- Each claim is SHOWN if p < 0.05, else NOT SHOWN.
- Bands for A4k8 and A4k16: RELIABLE >= 18/20, MAJORITY 10-17, MINORITY 1-9, NEVER 0.
- Readings, printed verbatim where they apply:
  - K8 or K16 SHOWN, with MG SHOWN: "spare channels let four streams bind: merges were early
    collisions."
  - K8 and K16 NOT SHOWN: "spare channels do not help four streams at this size."
  - MG SHOWN, with K8 and K16 NOT SHOWN: "spare channels reduce merges without raising
    binding: the failures move elsewhere."
Fisher is test_router_confirm's; the one-sided McNemar is test_conv_lr's.

DIAGNOSTICS, not part of the verdict
- Per seed: the recorded k=4 outcome, then the k=8 and k=16 outcomes, with transitions; the map,
  shared and eff_ch at the end; margin at 1200, 2400, 6000 and the end.
- Failure classes: MERGED (STREAM-PARTIAL with streams sharing a channel, by how many share),
  non-stream (test_scale_axes's generalized POSITION, KEY, OTHER), collapsed.
- Plateau structure: steps at held-out accuracy 0.45-0.55 and 0.70-0.80 before the transition
  (the whole run for failures).
- Channels used.
- The restart check per arm (held-out accuracy >= 0.6 at 2400; reported, not used).
- Gradient norms at steps 1, 10 and 100 (at k=16 the uniform start gives each channel 1/16).

CHECKS (printed before any training): test_scale_axes's verification (which runs every earlier
test's), then
  85 the pairing: the --prev file exists with a matching CPU; for seeds 220 and 221, every A4k8
     and A4k16 parameter except W_g is bitwise equal to A4k4's, and the first 3 batches are
     identical (asserted); re-running A4k4 seed 220 for 1200 steps (test_scale_axes.run_job)
     reproduces the recorded curve (recorded, not asserted: a failure makes the claims
     UNTESTED);
  86 every arm's optimizer runs at lr 1e-3 (step pre-hook, through run_job); this test's run_job
     on test_scale_axes's A4k4 gives test_scale_axes.run_job's record (1200 steps);
  87 the perfect gate at k=8 zeroes every cross-stream score at every layer, and leaves
     channels 4-7 at 0; the k-general margin equals 1 on the perfect gate and 0 on the uniform
     gate, at k=8 and k=16; the recurrent k=8 and k=16 gates' fits to the perfect routing
     (recorded for VALID, not asserted);
  88 the uniform gate is a stationary point at k=16 (CHECK 44's test; k=8 alongside);
  89 a worker's run is bit-identical to the same run in the main process (A4k16, 1200 steps).

LOGISTICS. Parallel single-thread workers; wall clock projected before training (worst case,
no drop rule). Results persist atomically to stream_channels_results.json (gitignored; copied
into results/X/ after the run).

RESULT (full run, 43/43 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz, 4 workers
x 1 thread; projection 4.60 h worst case; 174.0 min after verification; paired with
results/X/scale_axes_results.json; no --also file; machine X, commit e5a24ae).
  VALID: ceiling4k8 bound 3/3 (all at 1200); the pairing CHECK (85) passed; the k=8 and k=16
  recurrent gates fit the perfect routing (argmax 1.0000 each).
  K8 EIGHT CHANNELS BIND MORE: NOT SHOWN. A4k8 10/20 vs the recorded A4k4 9/20; A4k8 only 4
     (s221, s227, s234, s235), A4k4 only 3 (s222, s230, s231); one-sided McNemar p = 0.5.
  K16 SIXTEEN CHANNELS BIND MORE: NOT SHOWN. A4k16 10/20 vs 9/20; A4k16 only 4 (s224, s225, s227,
     s232), A4k4 only 3 (s222, s230, s237); p = 0.5.
  MG FEWER MERGES: NOT SHOWN. MERGED A4k16 10/20 vs the recorded A4k4 12/20; Fisher one-sided
     p = 0.376.
  Bands: A4k8 10/20 MAJORITY, A4k16 10/20 MAJORITY.
  The pre-registered reading: K8 and K16 NOT SHOWN: "spare channels do not help four streams at
  this size."

  Diagnostics (not part of the verdict):
  - The kind of merge changes, not the count. MERGED as defined (shared >= 2) includes runs with
    every stream on one channel. By how many streams share: A4k4 had 5 two-stream merges
    (STREAM-PARTIAL at ~0.75), 1 three-stream binder and 6 runs with all four streams on one
    channel; A4k8 had 1 two-stream merge (s232, 0.75) and 9 all-on-one-channel runs; A4k16 had 1
    two-stream merge (s237, at 0.51 with two pairs) and 9 all-on-one-channel runs. Every k=8 and
    k=16 all-on-one-channel run ended at chance (0.21-0.26) with the margin 0 at every evaluation
    (non-stream POSITION or OTHER). Post hoc and descriptive: the spare channels removed most
    two-stream stalls, and the failures moved to runs that never split the streams at all.
  - Every k=8 and k=16 binder ended with a one-to-one stream -> channel map (10/10 and 10/10;
    A4k4 8/9) and bound earlier: transition median 5400 (k=8) and 3600 (k=16) vs 14400 for A4k4;
    time on the 0.70-0.80 plateau before the transition, median 1800 and 1200 vs 4800.
  - The margin's "routed" line (0.9) is rarely crossed at k > 4: 3 of 10 k=8 binders and 0 of 10
    k=16 binders, all with every stream at 1.00. Their streams spread over more channels than
    four (channels used by binders, median 6 at both k=8 and k=16; eff_ch median 6.2 and 9.2),
    so the query's gate is less peaked and g_q.g_tgt smaller. "Routed" at k > 4 therefore
    undercounts; one-to-one and per-stream accuracy are the better signs here.
  - Seeds: 5 seeds bound at every k (220, 228, 229, 233, 236); 4 failed at every k (223, 226,
    238, 239).
  - The restart check (>= 0.6 at 2400) passed 3/20 (k=8) and 4/20 (k=16), all of which bound;
    A4k4 1/20 (not bound).
"""

import argparse
import copy
import itertools
import json
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
from test_binding_onset import onset_run, time_per_step, fmt_step, EVAL_EVERY, BATCH
from test_multilayer_binding import D, probe_batch
from test_channel_binding import eval_batch
from test_router_confirm import fisher_greater
from test_router_curriculum import (
    ARCH, SUB_LR, get, count, load_store, init_worker, makespan, cpu_model, save_results,
    COLLAPSE, TAU_END, TIME_STEPS,
)
from test_router_discovery import model_fn
from test_router_reliability import run_one, strip_all, med, gate_fit_k, T_CHECK, A_CHECK
from test_router_layout import routing_stats
from test_readout_path import GRAD_STEPS
from test_short_conv import conv_grad_norms, wstr, CONV_K, ROUTED_MARGIN
from test_conv_lr import bound_r, mcnemar_greater
import test_scale_axes as tsa
from test_scale_axes import scale_stats, routing_k, fail_class_k, TASK44

# ── Settings (fixed before any result) ───────────────────────────────────────
LR = SUB_LR                                # 1e-3, passed explicitly to every run
MAX_ITERS = 28800
CEIL_MIN = 2                               # VALID needs ceiling4k8 >= 2/3
FIT_MIN = 0.99
ALPHA = 0.05
RELIABLE_K, MAJORITY_K = 18, 10            # bands, of 20
USED_MASS = 0.05                           # a channel is used if >= 5% of any stream's mean gate
PLATEAUS = ((0.45, 0.55), (0.70, 0.80))
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "stream_channels_results.json"
PREV_FILE = "scale_axes_results.json"
PAIR_SEEDS = (220, 221)

A4K4 = tsa.ARM["A4k4"]
ARMS = [
    dict(tsa.ARM["ceiling4k4"], key="ceiling4k8", n_ch=8, ctx_pad=8,
         label="ceiling4k8  perfect gate k=8 (4-7 unused) + conv"),
    dict(A4K4, key="A4k8", n_ch=8, label="A4k8        arm A k=8 + conv"),
    dict(A4K4, key="A4k16", n_ch=16, label="A4k16       arm A k=16 + conv"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
SEEDS = dict(ceiling4k8=tuple(range(220, 223)), A4k8=tuple(range(220, 240)),
             A4k16=tuple(range(220, 240)))
PAIRED = tuple(range(220, 240))            # the recorded A4k4 seeds
READINGS = {
    "K+ MG+": "spare channels let four streams bind: merges were early collisions.",
    "K-": "spare channels do not help four streams at this size.",
    "MG+ K-": "spare channels reduce merges without raising binding: the failures move elsewhere.",
}


def band(c):
    return ("RELIABLE" if c >= RELIABLE_K else "MAJORITY" if c >= MAJORITY_K
            else "MINORITY" if c >= 1 else "NEVER")


def padded_task(k):
    """TASK44 for building a perfect-gate model at k channels: the context tokens (0, 1, 2, 3)
    padded with k - 4 token ids that never occur (-1, -2, ...), so perfect_gate_general is one-hot
    over k channels with channels 4..k-1 unused. Used only by the model builder."""
    t = copy.copy(TASK44)
    t.ctx_tokens = tuple(TASK44.ctx_tokens) + tuple(-(i + 1) for i in range(k - TASK44.S))
    return t


def builder_for(a):
    task = padded_task(a["ctx_pad"]) if a.get("ctx_pad") else TASK44
    return lambda aa, seed: model_fn(task, aa, seed)


# ── Jobs ─────────────────────────────────────────────────────────────────────
def run_job(sp):
    """test_scale_axes.run_job for this test's arms (and for any arm dict passed in the spec)."""
    a = sp.get("arm_dict") or ARM[sp["arm"]]
    return run_one(a, sp["seed"], sp["iters"], task=TASK44, stats_fn=scale_stats, grad_fn=conv_grad_norms,
                   lr=LR, builder=builder_for(a))


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def lr_job(key):
    seen = []
    h = register_optimizer_step_pre_hook(lambda opt, args, kw: seen.append(sorted({g["lr"] for g in opt.param_groups})))
    try:
        r = run_job(spec(key, 3, 2))
    finally:
        h.remove()
    return seen, r["ok"]


def time_job(key):
    a = ARM[key]
    mk = builder_for(a)(a, 0)
    per = time_per_step(TASK44, mk, steps=TIME_STEPS)
    m, data = mk(), eval_batch(TASK44)
    t0 = time.time()
    tbo.evaluate(m, TASK44, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


# ── The pairing (CHECK 85) ───────────────────────────────────────────────────
def pairing(pool, path):
    info = dict(path=path, exists=os.path.exists(path))
    if info["exists"]:
        prev = load_store(path)
        info["cpu"], info["git"] = prev.get("meta", {}).get("cpu"), prev.get("meta", {}).get("git")
        info["cpu_match"] = info["cpu"] == cpu_model()
        have = [bool((r := get(prev, "A4k4", s)) and r.get("ok")) for s in PAIRED]
        info["n_recorded"], info["complete"] = sum(have), all(have)
    else:
        info.update(cpu_match=False, complete=False, n_recorded=0)
    # structural: parameters other than W_g, and the first 3 batches (asserted by the caller)
    struct = []
    for s in PAIR_SEEDS:
        mods = {k: builder_for(a)(a, s)() for k, a in (("A4k4", A4K4), ("A4k8", ARM["A4k8"]), ("A4k16", ARM["A4k16"]))}
        p4 = dict(mods["A4k4"].named_parameters())
        fed = {}
        for k, a in (("A4k4", A4K4), ("A4k8", ARM["A4k8"]), ("A4k16", ARM["A4k16"])):
            rec, mk = [], builder_for(a)(a, s)

            def mk2(mk=mk, rec=rec):
                mm = mk()
                mm.register_forward_pre_hook(lambda mod, args: rec.append(args[0].clone()) if mod.training else None)
                return mm
            onset_run(TASK44, mk2, s, max_iters=3, lr=LR)
            fed[k] = rec
        for k in ("A4k8", "A4k16"):
            pk = dict(mods[k].named_parameters())
            same = sorted(pk) == sorted(p4) and all(torch.equal(p4[n], pk[n]) for n in p4 if n != "W_g")
            batches = len(fed[k]) == 3 and all(torch.equal(u, v) for u, v in zip(fed["A4k4"], fed[k]))
            struct.append((k, s, same, batches, tuple(pk["W_g"].shape)))
    info["struct"] = struct
    info["struct_ok"] = all(x[2] and x[3] for x in struct)
    if info["exists"]:
        r = pool.submit(tsa.run_job, tsa.spec("A4k4", 220, EVAL_EVERY)).result()
        now = json.loads(json.dumps(r["curve"]))
        old = ((get(prev, "A4k4", 220) or {}).get("curve") or [])[:len(now)]
        info["curve"] = (now, old)
        info["reproduced"] = bool(r["ok"] and now and now == old)
    else:
        info["curve"], info["reproduced"] = ([], []), False
    info["ok"] = info["exists"] and info["cpu_match"] and info["complete"] and info["reproduced"]
    return info


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool, prev_path):
    print("test_scale_axes.py's verification (which runs test_curriculum_confirm's, and so on down")
    print("to test_multilayer_binding's):")
    tsa.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    f89 = pool.submit(run_job, spec("A4k16", 1, EVAL_EVERY))
    f86 = {k: pool.submit(lr_job, k) for k in KEYS}
    f86b = pool.submit(tsa.run_job, tsa.spec("A4k4", 3, EVAL_EVERY))

    print(f"CHECK 85 the pairing with this machine's test_scale_axes results ({prev_path}):")
    pair = pairing(pool, prev_path)
    if pair["exists"]:
        print(f"         file exists: True   written on: {pair['cpu']} (git {pair['git']})   this machine: {cpu_model()}   "
              f"match: {pair['cpu_match']}   recorded A4k4 runs (220-239): {pair['n_recorded']}/{len(PAIRED)}")
    else:
        print("         file exists: False")
    for k, s, same, batches, shp in pair["struct"]:
        print(f"         {k:<6} seed {s}: parameters other than W_g bitwise equal to A4k4's: {same}   first 3 batches "
              f"identical: {batches}   W_g {shp} (A4k4's (4, 32))")
    now, old = pair["curve"]
    print(f"         A4k4 seed 220 re-run (test_scale_axes.run_job), {EVAL_EVERY} steps: {now} vs recorded {old}")
    print(f"         reproduced exactly: {pair['reproduced']}  -> "
          + ("PAIRED" if pair["ok"] and pair["struct_ok"] else
             "PAIRING FAILS (every claim will be UNTESTED)" if pair["struct_ok"] else "WRONG"))
    ok &= pair["struct_ok"]
    print()

    print(f"CHECK 86 every arm runs at lr {LR}, through test_scale_axes's run path:")
    good = True
    for k in KEYS:
        seen, rok = f86[k].result()
        g = seen == [[LR], [LR]] and rok
        good &= g
        print(f"         {k:<10} optimizer lr over 2 steps {seen}  -> {'OK' if g else 'WRONG'}")
    r_mine = run_job(dict(arm="A4k4", arm_dict=A4K4, seed=3, iters=EVAL_EVERY))
    r_tsa = f86b.result()
    g = strip_all(r_mine) == strip_all(r_tsa) and r_mine["ok"]
    good &= g
    print(f"         this test's run_job on test_scale_axes's A4k4 (seed 3, {EVAL_EVERY} steps) == test_scale_axes.run_job's record "
          f"(in a worker): "
          f"{strip_all(r_mine) == strip_all(r_tsa)}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print("CHECK 87 the k=8 perfect gate and the k-general margin:")
    a = ARM["ceiling4k8"]
    x = TASK44.make_batch(16, torch.Generator().manual_seed(870))[0][:, :-1]
    mdl = builder_for(a)(a, 6)().eval()
    with torch.no_grad():
        mdl.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=torch.Generator().manual_seed(871)))
    mdl.attn.record = []
    with torch.no_grad():
        gr8 = mdl(x, TAU_END)[2]
    lab = TASK44.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s_[:, 0][cross].abs().max().item() for s_ in mdl.attn.record]
    live = [s_[:, 0][~cross].abs().max().item() for s_ in mdl.attn.record]
    mdl.attn.record = None
    unused = bool((gr8[..., 4:] == 0).all()) and gr8.shape[-1] == 8
    g1 = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0 and unused
    print(f"     (a) ceiling4k8, random conv weights: layers seen {len(maxc)} (n_layer={ARCH['n_layer']}); read gate shape "
          f"{tuple(gr8.shape)}, channels 4-7 exactly 0: {unused}")
    for i, (c, l_) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l_:.2e}")
    a4 = tsa.ARM["ceiling4k4"]
    m4c, m8c = tsa.builder_for(a4)(a4, 3)(), builder_for(a)(a, 3)()
    with torch.no_grad():
        d48 = (tbo.logits_of(m4c.eval(), x) - tbo.logits_of(m8c.eval(), x)).abs().max().item()
    print(f"         (informative) ceiling4k8 vs ceiling4k4, same seed: max |logits diff| {d48:.1e} (the zero channels add "
          f"exact zeros)")
    probe = probe_batch(TASK44, 3)
    good2 = True
    for k in (8, 16):
        ap = dict(tsa.ARM["ceiling4k4"], n_ch=k, ctx_pad=k)
        au = dict(A4K4, gate="uniform", n_ch=k)
        mp_, mu_ = builder_for(ap)(ap, 3)(), builder_for(au)(au, 3)()
        rp, ru = routing_stats(mp_, TASK44, probe)["margin"], routing_stats(mu_, TASK44, probe)["margin"]
        rk = routing_k(mp_, TASK44, probe)
        g = abs(rp - 1.0) < 1e-6 and abs(ru) < 1e-6 and rk["ch_map"] == [0, 1, 2, 3] and rk["one_to_one"]
        good2 &= g
        print(f"     (b) k={k:<2}: margin on the perfect gate {rp:.6f} (want 1), on the uniform gate {ru:.6f} (want 0); "
              f"perfect gate's map {rk['ch_map']} one-to-one {rk['one_to_one']}: {g}")
    fits = {}
    for k in ("A4k8", "A4k16"):
        fits[k] = gate_fit_k(TASK44, ARM[k])
        print(f"     (c) {k}'s recurrent gate fit to the perfect 4-way routing (test_router_reliability.gate_fit_k): "
              f"mse {fits[k][0]:.2e}   argmax match {fits[k][1]:.4f}  -> {'FITS' if fits[k][1] >= FIT_MIN else 'DOES NOT FIT'} "
              f"(>= {FIT_MIN}; recorded for VALID, not asserted)")
    good = g1 and good2
    print(f"         -> {'OK' if good else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 88 the uniform gate is a stationary point (CHECK 44's test): W_g = 0, seed 5, one training batch (seed 88):")
    tok = TASK44.make_batch(BATCH, torch.Generator().manual_seed(88))[0]
    xx, yy = tok[:, :-1], tok[:, 1:]
    good = True
    for k in ("A4k8", "A4k16"):
        a = ARM[k]
        m = builder_for(a)(a, 5)()
        with torch.no_grad():
            m.W_g.zero_()
        kk = a["n_ch"]
        uni = True
        for train in (True, False):
            m.train(train)
            with torch.no_grad():
                gr = m(xx, TAU_END)[2]
            uni &= torch.equal(gr, torch.full_like(gr, 1.0 / kk))
        m.train()
        m.zero_grad(set_to_none=True)
        ql, qt, _ = TASK44.select(tbo.logits_of(m, xx), yy, None)
        F.cross_entropy(ql, qt).backward()
        gn, rn, _ = conv_grad_norms(m)
        g = uni and gn <= 1e-6 * rn
        good &= g
        print(f"         k={kk:<2}: gate exactly 1/k at every position (train and eval): {uni}   |grad| gate (W_in, W_h, W_g) "
              f"{gn:.3e}   rest {rn:.3e}  -> {'STATIONARY' if g else 'NOT STATIONARY'}")
    ok &= good
    print()

    print("CHECK 89 a worker's run is bit-identical to the same run here:")
    rw, rh = f89.result(), run_job(spec("A4k16", 1, EVAL_EVERY))
    g = strip_all(rw) == strip_all(rh) and rh["ok"]
    print(f"         A4k16 seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   acc {rh['acc']:.4f}   margin "
          f"{rh['end']['margin']:.4f}   map {rh['end']['ch_map']}   eff_ch {rh['end']['eff_ch']:.3f}  -> "
          f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= g
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}"
          + ("" if pair["ok"] else "   (CHECK 85's recorded pairing did not pass: see VALIDITY)") + "\n")
    assert ok, "verification failed — do not trust the results below"
    return pair, {k: dict(mse=v[0], argmax=v[1]) for k, v in fits.items()}


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    return bound_r(get(store, key, s))


def ok_runs(store, key, seeds=None):
    return [(s, r) for s in (SEEDS[key] if seeds is None else seeds) if (r := get(store, key, s)) and r.get("ok")]


def at(curve, step):
    return next((a for st, a, _ in curve if st == step), None)


def margin_at(r, step):
    x = next((x for x in r["stats"] if x["step"] == step), None)
    return None if x is None else x.get("margin")


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def routed(r):
    return r["end"].get("margin", -1) >= ROUTED_MARGIN


def shared_max(r):
    cm = r["end"]["ch_map"]
    return max(cm.count(c) for c in cm)


def merged(r):
    return shared_max(r) >= 2


def used(r):
    """Channels carrying >= 5% of any stream's mean read-gate mass at value positions."""
    sg = r["end"]["stream_gate"]
    return sorted({c for row in sg for c, v in enumerate(row) if v >= USED_MASS})


def outcome(r):
    if r["transition"] is not None:
        return "BOUND routed" if routed(r) else "BOUND NOT routed"
    fc = fail_class_k(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if merged(r) else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def plateau(r, lo, hi):
    tr = r["transition"]
    return sum(EVAL_EVERY for st, a_, _ in r["curve"] if (tr is None or st < tr) and lo <= a_ <= hi)


def med_int(xs):
    return f"{statistics.median(xs):.0f} [{min(xs)}, {max(xs)}]" if xs else "--"


def mstr(cm):
    return ",".join(str(c) for c in cm)


def raw_rows(tag, store, key, seeds):
    print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'acc@2400':>8} {'m@1200':>7} {'m@2400':>7} {'m@6000':>7} "
          f"{'m end':>7} {'eff_ch':>6} {'map':>9} {'sh':>2} {'used':>4}  per-stream acc        conv |w| lag 0/1/2/3   outcome")
    for s in seeds:
        r = get(store, key, s)
        if r is None:
            print(f"  {tag:<10} {s:>4}  NOT RUN")
            continue
        if not r.get("ok"):
            print(f"  {tag:<10} {s:>4}  FAILED — {r.get('error')}")
            continue
        e = r["end"]
        print(f"  {tag:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {fmt(at(r['curve'], T_CHECK)):>8} "
              f"{fmt(margin_at(r, EVAL_EVERY)):>7} {fmt(margin_at(r, 2 * EVAL_EVERY)):>7} {fmt(margin_at(r, 5 * EVAL_EVERY)):>7} "
              f"{fmt(e.get('margin')):>7} {e['eff_ch']:>6.3f} {mstr(e['ch_map']):>9} {shared_max(r):>2} {len(used(r)):>4}  "
              f"{'/'.join(f'{v:.2f}' for v in e['stream_acc']):<21} {wstr(e):<23}  {outcome(r)}"
              + ("  collapsed" if r["collapsed"] else ""))
    print()


def report_stop(store, wall, path):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — ceiling4k8 only (nothing else was run)")
    print("#" * 100)
    raw_rows("ceiling4k8", store, "ceiling4k8", SEEDS["ceiling4k8"])
    c = sum(bound(store, "ceiling4k8", s) for s in SEEDS["ceiling4k8"])
    need = min(CEIL_MIN, len(SEEDS["ceiling4k8"]))
    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    print(f"  ceiling4k8 binds {c}/{len(SEEDS['ceiling4k8'])} (needs >= {need}): the validity condition FAILED; nothing "
          f"else was run")
    print("     *** UNTESTED ***")
    print()
    print("PRE-REGISTERED CLAIMS: K8 UNTESTED, K16 UNTESTED, MG UNTESTED; no reading.")
    store["verdict"] = dict(valid=False, stopped="ceiling", ceiling=c, K8="UNTESTED", K16="UNTESTED", MG="UNTESTED",
                            readings=[])
    save_results(path, store)
    print(f"\n  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}\n")


def report(store, prev, pair, fits, wall, path, also):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print("  (m = routing margin at the query; map = the stream -> channel argmax at value positions, stream 0 first; "
          "sh = streams on the most crowded channel; used = channels with >= 5% of any stream's mean gate)")
    for k in KEYS:
        raw_rows(k, store, k, SEEDS[k])
    if prev is not None:
        print("  the recorded A4k4 runs (test_scale_axes, the --prev file), for the pairing:")
        raw_rows("A4k4 (rec)", prev, "A4k4", PAIRED)

    print("=" * 100)
    print("COUNTS (BOUND = transition not None; routed = end margin >= 0.9; MERGED = shared >= 2 at the end)")
    print("=" * 100)
    c, mg = {}, {}
    rows = [(k, store, SEEDS[k], ARM[k]["label"]) for k in KEYS]
    if prev is not None:
        rows.append(("A4k4", prev, PAIRED, "A4k4        recorded (test_scale_axes)"))
    for k, st, sd, label in rows:
        rs = ok_runs(st, k, sd)
        c[k] = sum(bound(st, k, s) for s in sd)
        mg[k] = sum(1 for _, r in rs if merged(r))
        print(f"  {label:<46} bound {c[k]:>2}/{len(sd)}   bound and routed "
              f"{sum(1 for _, r in rs if r['transition'] is not None and routed(r)):>2}   one-to-one at the end "
              f"{sum(1 for _, r in rs if r['end']['one_to_one']):>2}   MERGED {mg[k]:>2}/{len(rs)}   completed {len(rs)}/{len(sd)}")
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    nc = len(SEEDS["ceiling4k8"])
    need = min(CEIL_MIN, nc)
    fit_ok = all(v["argmax"] >= FIT_MIN for v in fits.values())
    valid = c["ceiling4k8"] >= need and pair["ok"] and fit_ok
    print(f"  ceiling4k8 binds {c['ceiling4k8']}/{nc} (needs >= {need})   the pairing CHECK (85) passed: {pair['ok']} "
          f"(file {pair['path']} exists {pair['exists']}, CPU match {pair.get('cpu_match')}, complete {pair.get('complete')}, "
          f"reproduced {pair.get('reproduced')})")
    print("  the recurrent gates fit the perfect routing: " + "   ".join(
        f"{k} argmax {v['argmax']:.4f}" for k, v in fits.items()) + f" (need >= {FIT_MIN}): {fit_ok}")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND)")
    print("#" * 100)
    v = dict(bound=c, merged=mg, valid=valid, pairing=pair["ok"], fits=fits)
    ver = lambda p: ("SHOWN" if p < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    n4 = len(PAIRED)
    for kk, arm, title in (("K8", "A4k8", "EIGHT CHANNELS BIND MORE"), ("K16", "A4k16", "SIXTEEN CHANNELS BIND MORE")):
        if prev is None:
            b_ = c_ = 0
            p = float("nan")
        else:
            b_ = sum(bound(store, arm, s) and not bound(prev, "A4k4", s) for s in PAIRED)
            c_ = sum(bound(prev, "A4k4", s) and not bound(store, arm, s) for s in PAIRED)
            p = mcnemar_greater(b_, c_)
        vv = ver(p) if prev is not None else "UNTESTED"
        print(f"  {kk} {title} (exact McNemar, one-sided): {arm} {c[arm]}/{len(SEEDS[arm])} vs recorded A4k4 "
              f"{c.get('A4k4', '?')}/{n4}; {arm} only {b_}, A4k4 only {c_}; p = {p:.3g}")
        print(f"     *** {kk}: {vv} ***")
        v.update({kk: vv, kk + "_p": p, kk + "_discordant": [b_, c_]})
    if prev is None:
        pm, vm = float("nan"), "UNTESTED"
    else:
        n16 = len(ok_runs(store, "A4k16"))
        n4r = len(ok_runs(prev, "A4k4", PAIRED))
        pm = fisher_greater(mg["A4k4"], n4r, mg["A4k16"], n16)
        vm = ver(pm)
    print(f"  MG FEWER MERGES (Fisher one-sided, fewer at k=16): MERGED A4k16 {mg['A4k16']}/{len(ok_runs(store, 'A4k16'))} vs "
          f"recorded A4k4 {mg.get('A4k4', '?')}/{n4}; p = {pm:.3g}")
    print(f"     *** MG: {vm} ***")
    v.update(MG=vm, MG_p=pm)
    print(f"  bands (RELIABLE >= {RELIABLE_K}, MAJORITY {MAJORITY_K}-{RELIABLE_K - 1}, MINORITY 1-{MAJORITY_K - 1}, NEVER 0): "
          f"A4k8 {c['A4k8']}/{len(SEEDS['A4k8'])} {band(c['A4k8'])}   A4k16 {c['A4k16']}/{len(SEEDS['A4k16'])} {band(c['A4k16'])}")
    v.update(band_A4k8=band(c["A4k8"]), band_A4k16=band(c["A4k16"]))
    print()
    print("  READINGS:")
    rd = []
    if valid:
        k_any = "SHOWN" in (v["K8"], v["K16"])
        k_none = v["K8"] == "NOT SHOWN" and v["K16"] == "NOT SHOWN"
        if k_any and v["MG"] == "SHOWN":
            rd.append(("K8 or K16 SHOWN, with MG SHOWN", READINGS["K+ MG+"]))
        if k_none:
            rd.append(("K8 and K16 NOT SHOWN", READINGS["K-"]))
        if v["MG"] == "SHOWN" and k_none:
            rd.append(("MG SHOWN, with K8 and K16 NOT SHOWN", READINGS["MG+ K-"]))
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
    print("  per seed: the recorded k=4 outcome, then k=8 and k=16 (transition; map; shared; eff_ch):")
    for s in PAIRED:
        cells = []
        for k, st in (("A4k4", prev), ("A4k8", store), ("A4k16", store)):
            r = get(st, k, s) if st is not None else None
            if not r or not r.get("ok"):
                cells.append(f"{k}: --")
                continue
            cells.append(f"{k}: {'B' if r['transition'] is not None else '.'} {fmt_step(r['transition']):>5} "
                         f"[{mstr(r['end']['ch_map'])}] sh {shared_max(r)} eff {r['end']['eff_ch']:.2f}")
        print(f"    s{s}  " + "   ".join(f"{x:<44}" for x in cells))
    print()
    print("  outcomes (failures: MERGED = STREAM-PARTIAL with streams sharing a channel; non-stream = test_scale_axes's "
          "generalized classes):")
    for k, st, sd in (("A4k4", prev, PAIRED), ("A4k8", store, SEEDS["A4k8"]), ("A4k16", store, SEEDS["A4k16"])):
        if st is None:
            continue
        rs = ok_runs(st, k, sd)
        tally = {}
        for s, r in rs:
            tally.setdefault(outcome(r), []).append(s)
        print(f"    {k:<6} " + "   ".join(f"{o} {len(x)}" for o, x in sorted(tally.items())))
        for o, xs in sorted(tally.items()):
            print(f"    {'':<6}   {o:<26} " + " ".join(f"s{s}" for s in xs))
        mgs = {}
        for _, r in rs:
            mgs[shared_max(r)] = mgs.get(shared_max(r), 0) + 1
        print(f"    {'':<6}   shared at the end (streams on the most crowded channel), all runs: "
              + "   ".join(f"{n}: {mgs[n]}" for n in sorted(mgs))
              + f"   collapsed {sum(1 for _, r in rs if r['collapsed'])}")
    print()
    print("  plateau structure (steps at held-out accuracy 0.45-0.55 and 0.70-0.80 before the transition, or the whole run), "
          "median [min, max]:")
    for k, st, sd in (("A4k4", prev, PAIRED), ("A4k8", store, SEEDS["A4k8"]), ("A4k16", store, SEEDS["A4k16"])):
        if st is None:
            continue
        rs = ok_runs(st, k, sd)
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {k:<6} {lab:<9} ({len(sub):>2})  0.45-0.55: {med_int([plateau(r, *PLATEAUS[0]) for r in sub]):<20} "
                      f"0.70-0.80: {med_int([plateau(r, *PLATEAUS[1]) for r in sub])}")
    print()
    print(f"  channels used (>= {USED_MASS:.0%} of any stream's mean gate at the end), median [min, max], by outcome:")
    for k, st, sd in (("A4k4", prev, PAIRED), ("A4k8", store, SEEDS["A4k8"]), ("A4k16", store, SEEDS["A4k16"])):
        if st is None:
            continue
        rs = ok_runs(st, k, sd)
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {k:<6} {lab:<9} ({len(sub):>2})  {med_int([len(used(r)) for r in sub]):<14}  eff_ch "
                      f"{med([r['end']['eff_ch'] for r in sub])}")
    print()
    print(f"  the restart check (held-out accuracy >= {A_CHECK} at {T_CHECK}; reported, not used): passed and of those bound; "
          f"failed and of those bound:")
    for k, st, sd in (("A4k4", prev, PAIRED),) + tuple((k, store, SEEDS[k]) for k in KEYS):
        if st is None:
            continue
        rs = ok_runs(st, k, sd)
        ps = [r for _, r in rs if (x := at(r["curve"], T_CHECK)) is not None and x >= A_CHECK]
        fs = [r for _, r in rs if not ((x := at(r["curve"], T_CHECK)) is not None and x >= A_CHECK)]
        print(f"    {k:<10} passed {len(ps):>2}/{len(rs):<2} bound {sum(1 for r in ps if r['transition'] is not None):>2}   "
              f"failed {len(fs):>2} bound {sum(1 for r in fs if r['transition'] is not None):>2}")
    print()
    print("  transitions, median [min, max]:")
    for k, st, sd in (("A4k4", prev, PAIRED),) + tuple((k, store, SEEDS[k]) for k in KEYS):
        if st is None:
            continue
        trs = [r["transition"] for _, r in ok_runs(st, k, sd) if r["transition"] is not None]
        print(f"    {k:<10} " + (f"{statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]" if trs else "none bound"))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median [min, max]:")
    for step in GRAD_STEPS:
        for k, st, sd in (("A4k4", prev, PAIRED),) + tuple((k, store, SEEDS[k]) for k in KEYS):
            if st is None:
                continue
            g = [r["grad"][str(step)] for _, r in ok_runs(st, k, sd) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>4} {k:<10} " + "   ".join(parts))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): " + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
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
                print(f"    {k:<10} bound here {c[k]}/{nh}   other {b2_}/{len(rs2)}   pooled {c[k] + b2_}/{nh + len(rs2)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — per evaluation (every {EVAL_EVERY} steps): held-out accuracy x100 and the routing margin x100")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            tag = f"{k:<10} s{s}"
            acc = "".join(f"{round(a_ * 100):>4}" for _, a_, _ in r["curve"])
            mg_ = "".join((f"{round(m_ * 100):>4}" if (m_ := margin_at(r, st)) is not None else "  --") for st, _, _ in r["curve"])
            end = f"bound @{r['transition']}" if r["transition"] else "not bound"
            print(f"  {tag} acc {acc}")
            print(f"  {'':<{len(tag)}} mrg {mg_}  -> {end}  [{mstr(r['end']['ch_map'])}]")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def describe(rec):
    if not rec.get("ok"):
        return f"*** FAILED: {rec.get('error')}"
    tr, e = rec["transition"], rec["end"]
    return (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} margin "
            f"{fmt(margin_at(rec, EVAL_EVERY))}->{fmt(e.get('margin'))} map [{mstr(e['ch_map'])}] shared {shared_max(rec)} "
            f"eff_ch {e['eff_ch']:.2f} streams {[round(v, 2) for v in e['stream_acc']]}")


def main():
    ap = argparse.ArgumentParser(description="Spare channels for four streams: k=8 and k=16 vs the recorded k=4.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None, help="a second machine's stream_channels_results.json, for pooled counts")
    ap.add_argument("--prev", default=PREV_FILE, help="this machine's scale_axes_results.json (the recorded A4k4 runs)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    iters = MAX_ITERS

    print("=" * 100)
    print("Stream channels: do spare channels (k=8, k=16) reduce the two-stream merges of the k=4 gate at S=4?")
    print(f"  task_for(4, 4) (S=4, P=4, length {TASK44.L}); lr {LR} (SUB_LR, explicit); conv 'layer' width {CONV_K}; up to "
          f"{iters} steps, evaluation every {EVAL_EVERY}")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<46} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  paired with {args.prev}: the recorded A4k4 runs, seeds 220-239 (all parameters but W_g equal, same batches)")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        pair, fits = verify(pool, args.prev)
        prev = load_store(args.prev) if pair["exists"] else None
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR,
                             max_iters=iters, prev=args.prev, fits=fits,
                             pairing={k: v for k, v in pair.items() if k not in ("curve", "struct")},
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool, one evaluation charged per {EVAL_EVERY}; "
              f"worst case = every run to {iters} steps")
        print("=" * 100)
        cost = dict(zip(KEYS, pool.map(time_job, KEYS)))
        for k in KEYS:
            print(f"  {k:<10} {cost[k] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds   (a full run {cost[k] * iters / 60:.1f} min)")
        d1 = [cost["ceiling4k8"] * iters] * len(SEEDS["ceiling4k8"])
        d2 = [cost[k] * iters for k in ("A4k8", "A4k16") for _ in SEEDS[k]]
        mk1, mk2 = makespan(d1, args.workers), makespan(d2, args.workers)
        print(f"  ceiling4k8 first: {len(d1)} runs, {mk1 / 3600:.2f} h;  then {len(d2)} runs: serial {sum(d2) / 3600:.2f} h, "
              f"{mk2 / 3600:.2f} h on {args.workers} workers;  total {(mk1 + mk2) / 3600:.2f} h")
        store["meta"]["projected_wall_h"] = (mk1 + mk2) / 3600
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            print(f"   {sp['arm']:<10} seed {sp['seed']}  {describe(rec)}  {rec.get('secs', 0.0):.0f}s  "
                  f"(elapsed {(time.time() - t0) / 60:.1f}m)")

        def run_specs(specs):
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
                    record(sp, rec)

        def todo(k):
            out = []
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<10} seed {s}  cached")
                else:
                    out.append(spec(k, s, iters))
            return out

        print("=" * 100)
        print(f"RUNS — ceiling4k8 first ({len(SEEDS['ceiling4k8'])} runs)")
        print("=" * 100)
        run_specs(todo("ceiling4k8"))
        n_ceil = sum(bound(store, "ceiling4k8", s) for s in SEEDS["ceiling4k8"])
        need = min(CEIL_MIN, len(SEEDS["ceiling4k8"]))
        print(f"   ceiling4k8 bound {n_ceil}/{len(SEEDS['ceiling4k8'])} (needs >= {need})")
        print()
        if n_ceil < need:
            print("   *** the ceiling failed: nothing else is run ***\n")
            report_stop(store, time.time() - t0, args.results)
            return
        rest = [sp for pair_ in itertools.zip_longest(todo("A4k16"), todo("A4k8")) for sp in pair_ if sp is not None]
        print("=" * 100)
        print(f"RUNS — A4k8 and A4k16 ({len(rest)} runs, interleaved)")
        print("=" * 100)
        run_specs(rest)
        print()

    report(store, prev, pair, fits, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
