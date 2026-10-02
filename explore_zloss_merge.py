#!/usr/bin/env python
"""
explore_zloss_merge.py — EXPLORATORY, not a result. Screen S27 (batch 9): does a router z-loss
(ST-MoE) on the gate's logits un-merge two streams that share a channel at S=4, k=4?

BACKGROUND
- S24's SPLIT (copy the busiest channel's W_g row to the idlest, plus noise, W_g's Adam state reset)
  un-merged 8/8 merged runs by 19200 (NOISE 1/8, CONTROL 0/8). It worked by bringing the spare
  channel's logit level with the shared one (row distance 1.96-3.32 -> 0.24-0.40).
- ST-MoE's router z-loss (penalising the gate's log-partition) keeps logits from saturating,
  continuously and without a discrete operation.

THE RUNS AND THE CODE PATH: S24's (and S21's): X's recorded A4k4 runs (arm A, k=4, conv 'layer'
width 4, S=4, P=4, task_for(4, 4), lr 1e-3) merged at 9600 that stayed merged: stream_recipe 247,
248, 251, 258 and scale_axes 221, 223, 225, 226 (main-branch modules loaded read-only at S21's MAIN_SHA
520fe51), re-run to T0 = 9600 through the recorded run's own path (S21's rerun), then T_CONT = 9600
updates on a deep copy of the model and of the optimizer state (S21/S24's continuation: onset_run's
step line for line, the batch stream replayed from the rerun's generator):
  ZLOSS  from 9600, the loss is the task loss + 1e-3 x mean over every position of the batch of
         (logsumexp_c of the gate's logits)^2. The logits are the read gate's pre-softmax values,
         recomputed differentiably from the model's own embedding exactly as
         test_instrument_v2.Instrument.gates computes them (h_t = tanh(W_in v_t + W_h h_{t-1}),
         logits_t = W_g h_t; softmax checked against the model's read gate); the term's gradient
         reaches W_in, W_h, W_g and the embedding only. Same optimizer (Adam, lr 1e-3, moments carried).
At every evaluation (every 1200): S24's (stream -> channel map, per-stream read gates at values,
per-stream and overall held-out accuracy), plus on the probe batch the mean (logsumexp)^2 and each
channel's mean logit over all positions.
Compared, run by run, with S24's recorded CONTROL, SPLIT and NOISE on the same runs (same state at
9600, same batches: the continuation's batch digest must equal S24's recorded one).
Per-run reading as S21/S24: "split" if at some evaluation up to 19200 the merged pair (the recorded end
map's) is on different channels and the held-out accuracy is >= 0.95.

RULE (fixed before any run): "z-loss un-merges" if >= 4/8 split by 19200; "it does not" if <= 1/8;
otherwise neither.

CHECKS: before the batch, on one run of each path (rerun to 1200, continuations of 1200): with weight 0
the continuation equals S24's CONTROL continuation bit for bit (curves, maps, batch digest, weights);
with weight 1e-3 it differs; the z-loss gradient reaches only W_in, W_h, W_g and the embedding; the
recomputed logits' softmax equals the model's read gate. In each job: the rerun reproduces its
recorded curve to 9600; the continuation starts from an optimizer state equal to the run's and stored
apart from it; its batches equal S24's recorded CONTROL batches (digest).
"""

import copy
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_merge_kick as s21
import explore_merge_split as s24
import test_binding_onset as tbo
from test_binding_onset import BATCH, EVAL_EVERY
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_instrument_v2 import TAU_END

tsa = s21.tsa
NAME = "zloss_merge"
IDEA = "a router z-loss (ST-MoE) on the gate's logits from 9600, on S=4 runs with two streams merged"
SOURCE = ("ST-MoE's router z-loss; batch 8's S24 (SPLIT un-merged 8/8 by bringing the spare channel's logit "
          "level with the shared one)")
CHANGE = ("none to the recorded runs up to 9600; then 9600 updates with + 1e-3 x mean over positions of "
          "(logsumexp of the gate logits)^2")
PAIRING = "each run against S24's recorded CONTROL, SPLIT and NOISE (same state at 9600, same batches)"
TASK = s21.TASK
T0, T_CONT = s21.T0, s21.T_CONT
Z_WEIGHT = 1e-3
BIND = s21.BIND
SPLIT_MIN, NOT_MAX = 4, 1
S24_STORE = s24.NAME

_A = dict(tsa.ARM["A4k4"], lr=tsa.LR, iters=T0 + T_CONT, prio=1, z_weight=Z_WEIGHT,
          sched=f"lr {tsa.LR:g} throughout; from {T0}: + {Z_WEIGHT:g} x mean over positions of (logsumexp of the "
                f"gate logits)^2")
ARMS = {
    "SR_A4k4": dict(_A, key="SR_A4k4", seeds=s21.SEEDS_SR, label="test_stream_recipe's A4k4 (X), merged pair stalled"),
    "SA_A4k4": dict(_A, key="SA_A4k4", seeds=s21.SEEDS_SA, label="test_scale_axes's A4k4 (X), merged pair stalled"),
}


# ── the z-loss ───────────────────────────────────────────────────────────────
def gate_logits(m, x):
    """The read gate's logits (B, T, k), differentiable: Instrument.gates' recurrence on m.embed(x)."""
    assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
    v = m.embed(x)
    h = torch.zeros(v.shape[0], m.h_gate, dtype=v.dtype)
    lg = []
    for t in range(v.shape[1]):
        h = torch.tanh(v[:, t] @ m.W_in.T + h @ m.W_h.T)
        lg.append(h @ m.W_g.T)
    return torch.stack(lg, dim=1)


def zloss(m, x):
    return (torch.logsumexp(gate_logits(m, x), dim=-1) ** 2).mean()


@torch.no_grad()
def logit_stats(m, x):
    was = m.training
    m.eval()
    try:
        lg = gate_logits(m, x)
    finally:
        m.train(was)
    return dict(z=float((torch.logsumexp(lg, -1) ** 2).mean()), ch_logit=[round(v, 4) for v in lg.mean((0, 1)).tolist()],
                max_abs=float(lg.abs().max()))


def continuation(m0, opt0, rng0, t0, steps, weight, data, probe, lr, ref):
    """S24's continuation for CONTROL (S21's: onset_run's step, line for line, on deep copies), with
    + weight x zloss in the loss (weight None: the term is not computed)."""
    m = copy.deepcopy(m0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(opt0.state_dict()))
    start = dict(equal=s21.same_state(ref, s21.opt_state(opt)), separate=not (s21.ptrs(opt) & s21.ptrs(opt0)))
    g = torch.Generator()
    g.set_state(rng0.get_state())
    dig, curve, evals = hashlib.sha256(), [], []
    zs = [logit_stats(m, probe[0])]
    for step in range(1, steps + 1):
        tokens, _ = TASK.make_batch(BATCH, g)
        dig.update(tokens.numpy().tobytes())
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        ql, qt, _ = TASK.select(m(inp, TAU_END)[0], tgt, None)
        loss = F.cross_entropy(ql, qt)
        if weight is not None:
            loss = loss + weight * zloss(m, inp)
        opt.zero_grad(); loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            acc, el = tbo.evaluate(m, TASK, data)
            curve.append([t0 + step, acc, el])
            rk = tsa.routing_k(m, TASK, probe)
            evals.append(dict(step=t0 + step, acc=acc, ch_map=rk["ch_map"], shared=rk["shared"],
                              stream_gate=rk["stream_gate"], stream_acc=tsa.stream_acc(m, TASK, data)))
            zs.append(logit_stats(m, probe[0]))
    return dict(curve=curve, evals=evals, digest=dig.hexdigest(), start_equal=start["equal"],
                start_separate=start["separate"], zstats=zs), m


def split_at(evals, pair):
    return s21.split_at(evals, pair)


def run_job(arm, seed):
    a = ARMS[arm]
    dry = a["iters"] < T0 + T_CONT
    t0, cont = (2 * EVAL_EVERY, EVAL_EVERY) if dry else (T0, T_CONT)
    rec0 = s21.recorded(arm, seed)
    pair = s21.merged_pair(rec0)
    want0 = [c for c in rec0["curve"] if c[0] <= t0]
    rec, m, opt, rng = s21.rerun(arm, seed, t0)
    data, probe = eval_batch(TASK), probe_batch(TASK, seed)
    snap, psnap = s21.opt_state(opt), s24.params_snapshot(m)
    rk = tsa.routing_k(m, TASK, probe)
    z, mz = continuation(m, opt, rng, t0, cont, a["z_weight"], data, probe, a["lr"], snap)
    s24r = ec.load_store(("dry_" if dry else "") + S24_STORE)["runs"].get(f"{arm}|{seed}")
    ref_dig = s24r["control"]["digest"] if s24r and s24r.get("ok") and s24r.get("t0") == t0 else None
    end = tsa.scale_stats(mz, TASK, probe, "end")
    out = dict(ok=True, seed=seed, k=4, dry=dry, t0=t0, cont=cont, merged_pair=pair, z_weight=a["z_weight"],
               recorded_end_map=rec0["end"]["ch_map"], recorded_acc=rec0["acc"], rerun_curve=rec["curve"],
               map_t0=rk["ch_map"], stream_gate_t0=rk["stream_gate"], zloss=z,
               reproduces=rec["curve"] == want0,
               opt_separate=z["start_equal"] and z["start_separate"],
               run_unchanged=s21.same_state(snap, s21.opt_state(opt)) and s24.same_params(psnap, m),
               s24_digest=ref_dig, batches_as_s24=(ref_dig == z["digest"]) if ref_dig else None,
               split_zloss=split_at(z["evals"], pair))
    out.update(curve=[c for c in rec["curve"]] + z["curve"], acc=z["curve"][-1][1] if z["curve"] else None,
               val_cos=end["val_cos"], key_cos=end["key_cos"], ctx_cos=end["ctx_cos"], end=end, stats=[])
    out["transition"] = tbo.transition(out["curve"])
    out["reading"] = "split" if out["split_zloss"] is not None else "merged"
    return out


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], TASK, tsa.builder_for(tsa.ARM["A4k4"])(tsa.ARM["A4k4"], a["seeds"][0]), a["lr"],
             tsa.scale_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    der = s21.derived_runs()
    rows = [(f"the runs are S21's derived list (stream_recipe {der['SR_A4k4']}, scale_axes {der['SA_A4k4']})",
             der["SR_A4k4"] == s21.SEEDS_SR and der["SA_A4k4"] == s21.SEEDS_SA)]
    for arm, seed in (("SA_A4k4", s21.SEEDS_SA[0]), ("SR_A4k4", s21.SEEDS_SR[0])):
        t0 = cont = EVAL_EVERY
        lr = ARMS[arm]["lr"]
        rec0 = s21.recorded(arm, seed)
        rec, m, opt, rng = s21.rerun(arm, seed, t0)
        data, probe = eval_batch(TASK), probe_batch(TASK, seed)
        snap = s21.opt_state(opt)
        pair = s21.merged_pair(rec0)
        ctl, mc = s24.continuation(m, opt, rng, seed, t0, cont, "CONTROL", 0, 1, data, probe, lr, snap, pair)
        z0, m0 = continuation(m, opt, rng, t0, cont, 0.0, data, probe, lr, snap)
        z1, m1 = continuation(m, opt, rng, t0, cont, Z_WEIGHT, data, probe, lr, snap)
        same0 = (z0["curve"] == ctl["curve"] and [e["ch_map"] for e in z0["evals"]] == [e["ch_map"] for e in ctl["evals"]]
                 and z0["digest"] == ctl["digest"] and s24.same_params(s24.params_snapshot(mc), m0))
        differs = not s24.same_params(s24.params_snapshot(mc), m1) and z1["digest"] == ctl["digest"]
        want0 = [c for c in rec0["curve"] if c[0] <= t0]
        rows.append((f"{s21.SRC[arm]['path']} seed {seed}: rerun to {t0} reproduces the record ({rec['curve'] == want0}); "
                     f"with weight 0 the continuation equals S24's CONTROL continuation bit for bit over {cont} updates "
                     f"(curves {z0['curve']} vs {ctl['curve']}, maps, digest, weights: {same0}); with weight {Z_WEIGHT:g} "
                     f"the weights differ on the same batches ({differs}; probe mean lse^2 {z1['zstats'][0]['z']:.3f} -> "
                     f"{z1['zstats'][-1]['z']:.3f}, CONTROL {z0['zstats'][-1]['z']:.3f}); start states equal and separate "
                     f"({z0['start_equal'] and z0['start_separate'] and z1['start_equal'] and z1['start_separate']})",
                     rec["curve"] == want0 and same0 and differs and z0["start_equal"] and z0["start_separate"]
                     and z1["start_equal"] and z1["start_separate"]))
    a = tsa.ARM["A4k4"]
    mm = tsa.builder_for(a)(a, 221)()
    mm.train()
    x = TASK.make_batch(BATCH, torch.Generator().manual_seed(5))[0][:, :-1]
    zloss(mm, x).backward()
    got = {n for n, p in mm.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    rows.append((f"the z-loss gradient reaches only {sorted(got)}", got == {"W_in", "W_h", "W_g", "embed.weight"}))
    mm.eval()
    probe = probe_batch(TASK, 221)
    with torch.no_grad():
        d = float((F.softmax(gate_logits(mm, probe[0]), -1) - mm(probe[0], TAU_END)[2]).abs().max())
    rows.append((f"softmax of the recomputed logits equals the model's read gate (max diff {d:.1e})", d < 1e-6))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def rule(n):
    return "z-loss un-merges" if n >= SPLIT_MIN else ("it does not" if n <= NOT_MAX else "neither")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    done = [r for r in store["runs"].values() if r.get("ok")]
    t0, t1 = (done[0]["t0"], done[0]["t0"] + done[0]["cont"]) if done else (T0, T0 + T_CONT)
    s24s = ec.load_store(("dry_" if done and done[0]["dry"] else "") + S24_STORE)["runs"]
    print(f"  per run (map = stream -> channel at each evaluation, as acc:map; '|' = the merged pair on different "
          f"channels; continuation from {t0} to {t1}; S24's CONTROL / SPLIT / NOISE split steps from its record; z = "
          f"probe mean (logsumexp)^2, logits = each channel's mean gate logit over all positions):")
    n_z, n_done = 0, 0
    cnt24 = {k: 0 for k in ("control", "split", "noise")}
    yn = lambda v: "yes" if v else ("n/a" if v is None else "NO")
    for arm, a in ARMS.items():
        for s in a["seeds"]:
            r = store["runs"].get(f"{arm}|{s}")
            if r is None or not r.get("ok"):
                print(f"    {arm:<8} {s:>4} {ec.tag(r)}")
                continue
            n_done += 1
            pair = r["merged_pair"]
            r24 = s24s.get(f"{arm}|{s}") or {}
            sp24 = {k: r24.get(f"split_{k}") for k in cnt24} if r24.get("ok") else {}
            for k in cnt24:
                cnt24[k] += sp24.get(k) is not None
            z = r["zloss"]
            zs = z["zstats"]
            print(f"    {arm:<8} {s:>4}  merged pair streams {pair} (recorded end map {r['recorded_end_map']}, rec acc "
                  f"{r['recorded_acc']:.3f}; map at {r['t0']} {r['map_t0']})")
            print(f"    {'':<8} {'':>4}  rerun reproduces: {yn(r['reproduces'])}; start state equal and separate: "
                  f"{yn(r['opt_separate'])}; run unchanged: {yn(r['run_unchanged'])}; batches as S24's: "
                  f"{yn(r['batches_as_s24'])}")
            print(f"    {'':<8} {'':>4}  ZLOSS   {s24.ev_str(z['evals'], pair)}")
            last = z["evals"][-1] if z["evals"] else None
            sa = "/".join(f"{v:.2f}" for v in last["stream_acc"]) if last else "--"
            print(f"    {'':<8} {'':>4}  {'':<7} at {last['step'] if last else '--'}: acc "
                  f"{last['acc'] if last else float('nan'):.3f}, per stream {sa}, merged pair on channels "
                  f"{[last['ch_map'][pair[0]], last['ch_map'][pair[1]]] if last else '--'}; z {zs[0]['z']:.3f} -> "
                  f"{zs[-1]['z']:.3f}; logits {zs[0]['ch_logit']} -> {zs[-1]['ch_logit']}; max |logit| "
                  f"{zs[0]['max_abs']:.2f} -> {zs[-1]['max_abs']:.2f}")
            s24txt = ("; S24: " + ", ".join(f"{k.upper()} {('split at ' + str(sp24.get(k))) if sp24.get(k) else 'merged'}"
                                            for k in cnt24)) if sp24 else "; S24: no record"
            print(f"    {'':<8} {'':>4}  READING: ZLOSS {r['reading']}"
                  + (f" (at {r['split_zloss']})" if r["split_zloss"] else "") + s24txt)
            n_z += r["split_zloss"] is not None
    rd = rule(n_z)
    print(f"  split by {t1}: ZLOSS {n_z}/{n_done}; S24's records on these runs: CONTROL {cnt24['control']}, SPLIT "
          f"{cnt24['split']}, NOISE {cnt24['noise']} of {n_done}")
    print(f"  RULE S27 ('z-loss un-merges' if >= {SPLIT_MIN}/8 split, 'it does not' if <= {NOT_MAX}/8): {rd}")
    return dict(n=n_done, split=n_z, s24=cnt24, rule=rd)
