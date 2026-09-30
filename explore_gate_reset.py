#!/usr/bin/env python
"""
explore_gate_reset.py — EXPLORATORY, not a result. Screen S11 (batch 4): detect a wrong gate
split early, without labels, and give the gate a fresh start.

WHY. Arm A's failures are gates that commit early (by step 1200) to a POSITION split (by triple
index or sequence half) or a KEY split (by key identity) instead of the stream split. The restart
rule (test_router_reliability) restarts the whole run from a fresh seed, using held-out accuracy.
This screen restarts only the gate, using the gate itself.

THE ARM (RESET). Arm A (test_short_conv's recipe: grouped BindTask P=4, S=2, n_vals=16, n_q=1;
lr test_channel_binding.SUB_LR = 1e-3 passed explicitly; MAX_ITERS 24000). Seeds 160-179, paired
with X's recorded arm A.
- DETECTOR (label-free: token ids and positions only). At step 1200 (right after that
  evaluation), on the run's probe batch tokens (probe_batch(task, seed), its stream labels and
  role masks unused), the read gate's channel-0 probability at the body's key positions (3j+1)
  gives eta^2 by triple index j, by sequence half (j >= P) and by key id (the key token) —
  test_router_layout.routing_stats' eta_key_by_index / _by_half / _by_key.
- If any is >= 0.5: W_in, W_h, W_g are re-drawn exactly as at build (randn * 0.1, in that order)
  from a fresh generator seeded seed + 50000*r (r = 1, 2, 3 for the r-th reset), their Adam state is
  cleared (so their moments and step count restart), and training continues; every other
  parameter and its Adam state are untouched.
- The detector runs again 1200 steps after each reset. At most 3 resets; the detection 1200 steps
  after the 3rd reset is recorded but cannot reset. A detection below 0.5 ends the checks.
The run's optimizer is captured by an optimizer step pre-hook registered for the run only.

RULE: batch 1's (DISCOVERED vs X's arm A; promising if b - c >= 4 and McNemar p < 0.10; not if
b - c <= 0; otherwise inconclusive). Also reported: resets used per run, and outcome by number of
resets.

CHECKS
- the detector reads no stream label: its inputs are the model and the token ids; it equals
  routing_stats' eta_key_by_index/half/key on a probe, computed with the probe's stream labels and
  role masks withheld;
- a reset changes only W_in, W_h, W_g and their Adam state (forced reset: threshold -1);
- with the threshold at 2.0 (never fires), a run equals arm A bit for bit (curve = X's record
  through 2400; weights after update 2400 = a fresh arm-A run's).
"""

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH
from test_router_layout import eta2, routing_stats
from test_binding_onset import EVAL_EVERY
from test_instrument_v2 import TAU_END

NAME = "gate_reset"
IDEA = "a label-free detector of position/key gate splits at step 1200; reset only the gate"
SOURCE = ("restart heuristics for bad initial commitments (e.g. dead-unit / expert re-initialization "
          "in MoE training); the project's restart rule, applied to the gate alone")
CHANGE = ("arm A; at 1200 (and 1200 after each reset, max 3) if eta^2 of the read gate at key "
          "positions by index, half or key >= 0.5: re-draw W_in, W_h, W_g, clear their Adam state")
PAIRING = ("seeds 160-179, paired with X's recorded arm A (test_short_conv, lr 1e-3; same initial "
           "parameters and batches)")
SEEDS = tuple(range(160, 180))
THRESH = 0.5
MAX_RESETS = 3
RESET_OFFSET = 50_000
GATE = ("W_in", "W_h", "W_g")

ARMS = {
    "RESET": dict(ec.ARM_A, key="RESET", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1,
                  label="arm A + label-free gate reset (max 3)",
                  sched=f"lr {ec.SUB_LR:g} throughout; gate reset if eta^2(index/half/key) >= "
                        f"{THRESH} at 1200 and 1200 after each reset (max {MAX_RESETS})"),
}


@torch.no_grad()
def detect(model, tokens, task):
    """eta^2 of the read gate's channel-0 probability at key positions by triple index, half and
    key id. Inputs: the model and token ids only."""
    was = model.training
    model.eval()
    try:
        gr = model(tokens, TAU_END)[2]
    finally:
        model.train(was)
    S, P, n = task.S, task.P, task.S * task.P
    B = tokens.shape[0]
    j = torch.arange(n)
    p = gr[..., 0][:, 3 * j + 1]
    return dict(index=eta2(p, j.expand(B, n)), half=eta2(p, (j >= P).long().expand(B, n)),
                key=eta2(p, tokens[:, 3 * j + 1] - S))


def reset_gate(m, opt, seed, r):
    g = torch.Generator().manual_seed(seed + RESET_OFFSET * r)
    with torch.no_grad():
        for n in GATE:                                   # MultiBDH's order and scale
            p = getattr(m, n)
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    for n in GATE:
        opt.state.pop(getattr(m, n), None)


def run_path(a, seed, iters, thresh=THRESH, keep=None, trace=None):
    task = ec.TASK
    tokens = probe_batch(task, seed)[0]
    holder, st = {}, dict(next=EVAL_EVERY, resets=0, log=[])

    def hook(opt, args, kwargs):
        holder.setdefault("opt", opt)

    def check(m, step, data):
        if st["next"] is None or step != st["next"]:
            return
        e = detect(m, tokens, task)
        fire = max(e.values()) >= thresh
        entry = dict(step=step, eta=e, fired=bool(fire))
        if fire and st["resets"] < MAX_RESETS:
            st["resets"] += 1
            if trace is not None:
                trace["before"] = {n: p.detach().clone() for n, p in m.named_parameters()}
                trace["state_before"] = {n: {k: (v.clone() if torch.is_tensor(v) else v)
                                             for k, v in holder["opt"].state[p].items()}
                                         for n, p in m.named_parameters() if p in holder["opt"].state}
            reset_gate(m, holder["opt"], seed, st["resets"])
            if trace is not None:
                trace["after"] = {n: p.detach().clone() for n, p in m.named_parameters()}
                trace["state_after"] = {n: p in holder["opt"].state for n, p in m.named_parameters()}
                trace["state_objs"] = {n: holder["opt"].state[p] for n, p in m.named_parameters()
                                       if p in holder["opt"].state}
            entry["reset"] = st["resets"]
            st["next"] = step + EVAL_EVERY
        elif fire and st["resets"] >= MAX_RESETS:
            entry["reset"] = None                        # recorded; cannot reset
            st["next"] = None
        else:
            st["next"] = None
        st["log"].append(entry)

    h = register_optimizer_step_pre_hook(hook)
    try:
        rec = ec.run_one(a, seed, iters, check=check, keep=keep, task=task, stats_fn=c2.stats_b2,
                         grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                         builder=lambda aa, s: (lambda: build(task, aa, ARCH, s)))
    finally:
        h.remove()
    rec.update(resets=st["resets"], reset_log=st["log"])
    return rec


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: build(ec.TASK, ec.ARM_A, ARCH, 0), a["lr"], c2.stats_b2)]


def check():
    ok = True
    task, seed = ec.TASK, 160
    # (1) the detector reads no stream label
    m = build(task, ec.ARM_A, ARCH, seed)
    with torch.no_grad():
        m.W_in.mul_(30.0); m.W_g.mul_(10.0)             # a gate far from uniform
    pinp, plab, prole = probe_batch(task, seed)
    e = detect(m, pinp, task)
    full = routing_stats(m, task, (pinp, plab, prole))
    params = list(inspect.signature(detect).parameters)
    same = (e["index"] == full["eta_key_by_index"] and e["half"] == full["eta_key_by_half"]
            and e["key"] == full["eta_key_by_key"])
    # (2) a forced reset changes only W_in, W_h, W_g and their Adam state
    trace = {}
    r = run_path(ARMS["RESET"], seed, EVAL_EVERY, thresh=-1.0, trace=trace)
    changed = {n for n in trace["before"] if not torch.equal(trace["before"][n], trace["after"][n])}
    cleared = {n for n, present in trace["state_after"].items() if not present}
    kept_ok = all(n in trace["state_objs"] and all(
        (torch.equal(v, trace["state_objs"][n][k]) if torch.is_tensor(v) else v == trace["state_objs"][n][k])
        for k, v in trace["state_before"][n].items()) for n in trace["state_before"] if n not in GATE)
    # (3) threshold 2.0: bit for bit arm A
    k_a, k_r = {"at": 2 * EVAL_EVERY}, {"at": 2 * EVAL_EVERY}
    ra = ec.run_one(ec.ARM_A, seed, 2 * EVAL_EVERY, keep=k_a, task=task, stats_fn=tsc.conv_stats,
                    grad_fn=tsc.conv_grad_norms, lr=ec.SUB_LR)
    rr = run_path(ARMS["RESET"], seed, 2 * EVAL_EVERY, thresh=2.0, keep=k_r)
    want = [c for c in ec.recorded("A")[seed]["curve"] if c[0] <= 2 * EVAL_EVERY]
    same_run = (rr["curve"] == want == ra["curve"] and rr["resets"] == 0
                and all(torch.equal(k_a["snap"][n], k_r["snap"][n]) for n in k_a["snap"]))
    for nm, v in ((f"detector inputs are {params} (no labels); with the probe's stream labels and "
                   f"role masks withheld it equals routing_stats' eta_key_by_index/half/key "
                   f"({e['index']:.4f}/{e['half']:.4f}/{e['key']:.4f})",
                   same and params == ["model", "tokens", "task"]),
                  (f"a forced reset changes only W_in, W_h, W_g (changed: {sorted(changed)})",
                   changed == set(GATE) and r["resets"] == 1),
                  (f"... and clears exactly their Adam state (cleared: {sorted(cleared)}); every other "
                   f"parameter's Adam state is untouched", cleared == set(GATE) and kept_ok),
                  (f"threshold 2.0 (never fires): curve equals X's record through 2400 and a fresh arm-A "
                   f"run's, weights after update 2400 equal ({rr['curve']})", same_run)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store)
    s = ARMS["RESET"]["seeds"]
    runs = {x: r for x in s if (r := store["runs"].get(f"RESET|{x}")) and r.get("ok")}
    print("  resets per run (step: eta index/half/key -> reset r):")
    for x, r in runs.items():
        log = "; ".join(f"{e['step']}: {e['eta']['index']:.2f}/{e['eta']['half']:.2f}/"
                        f"{e['eta']['key']:.2f}" + (f" -> reset {e['reset']}" if e.get("reset") else
                                                     (" -> fired, no resets left" if e["fired"] else ""))
                        for e in r["reset_log"])
        print(f"    {x:>4} resets {r['resets']}  {ec.tag(r):<15} {log}")
    print("  paired with X's arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "RESET", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "RESET", s, ec.bound, label="BOUND")
    for step in (1200, "end"):
        c2.paired_routed(store, "RESET", s, step, {})
    done = list(runs)
    print(f"    failures RESET {ec.fail_counts(store, 'RESET', done)}   "
          f"X arm A {ec.recorded_fail_counts('A', done)}")
    hist = {}
    for r in runs.values():
        h = hist.setdefault(r["resets"], [0, 0])
        h[0] += ec.discovered(r); h[1] += 1
    print("  outcome by number of resets: " + "   ".join(
        f"{k} resets: DISCOVERED {v[0]}/{v[1]}" for k, v in sorted(hist.items())))
    print(f"  resets used: {sum(r['resets'] for r in runs.values())} over {len(runs)} runs")
    return d
