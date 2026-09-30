#!/usr/bin/env python
"""
explore_stall_probe.py — EXPLORATORY, not a result. Screen S14 (batch 5), diagnostic, no verdict:
why do gates that route by stream fail to bind?

BACKGROUND (post hoc). Routed-but-unbound runs on the header layout: batch 2's far_A 165, 166 (2 of
2 routed) and batch 4's far_nudge 164, 165 (2 of 7 routed), all at 0.52-0.55; far_nudge 161
(stream-partial) also stalled at 0.55. far_ceil (the perfect gate) bound 3/3. The main line saw the
same in the blocked layout (test_router_layout: gates that routed and never bound, 8 on X, 6 on L).

RUNS, each re-run to its recorded end through its own recorded path (onset_run with the recorded
builder, task and lr 1e-3 = test_channel_binding.SUB_LR; early stopping as recorded):
  hdr_far_nudge  batch 4's far_nudge 161, 164, 165 (stalled); 160 (control: discovered)
  hdr_far_A      batch 2's far_A 165, 166 (stalled)
  blk_A          blocked layout: the first 3 seeds, in seed order, of X's recorded test_router_layout
                 A_blocked runs with end margin >= 0.9 and not bound (results/X/router_layout_
                 results.json): 120, 130, 136
A rerun whose curve differs from the recorded one at any evaluation is reported UNTESTED and not
replaced (no probe is run on it).

AT EACH RUN'S END STATE
  (a) SWAP: held-out accuracy (eval_batch, 2048 queries) with the learned gate replaced by the
      perfect gate, channels assigned by the learned stream-to-channel map (stream s -> the channel
      its mean read gate prefers over the probe batch's body positions; if both streams map to one
      channel, the identity is used and said).
  (b) CONTINUE: 4800 more updates from the end state (a copy of the model and of the optimizer's
      state, the run's own batch stream continued from its generator state), twice: with the
      perfect gate fixed (as in SWAP) and with the learned gate (control); evaluation every 1200.
  (c) GATE PROFILE: mean read and write gate (channel-0 probability) per stream at each position
      type (CTX, KEY, VAL, query CTX, query KEY) on the probe batch, learned vs perfect, and the
      routing margin at value positions (header_routing_stats / routing_stats).
READINGS, per run: SWAP-FIXES if (a) >= 0.95 (the memory holds stream-separated bindings and the
learned gate's reads miss them; where (c) differs is printed); MEMORY-STUCK if (a) < 0.95, the
perfect-gate continuation reaches >= 0.95 at some evaluation and the learned one does not;
SLOW-ONLY if both continuations reach >= 0.95; NEITHER otherwise. Summary: the stalled runs'
readings by layout (the control is shown apart).

CHECKS (before the batch): the perfect gate (identity and swapped maps) zeroes every cross-stream
score at every layer on both layouts; the continuation code continues a run exactly (far_nudge 160
re-run to 2400 and continued 1200 updates with the learned gate reproduces its recorded
evaluation at 3600); both continuations see identical batches; SWAP's evaluation path with the
learned gate reproduces the recorded accuracy. In every job: the rerun reproduces the recorded
curve at every evaluation, SWAP's learned-gate path reproduces the recorded final accuracy, and the
two continuations' batch digests are equal.
"""

import copy
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import explore_far_cue as s6
import explore_far_express as s10
import test_binding_onset as tbo
import test_router_layout as trl
import test_router_curriculum as trc
from test_binding_onset import BATCH, EVAL_EVERY, eval_batch
from test_multilayer_binding import BDH, build, probe_batch
from test_channel_binding import ARCH
from test_instrument_v2 import Instrument, perfect_gate_general, TAU_END
from test_router_layout import routing_stats, fail_class

NAME = "stall_probe"
IDEA = "why routed gates fail to bind: perfect-gate swap, continued training, gate profile"
SOURCE = ("stalled runs of batches 2 and 4 (header layout) and of the main line's test_router_layout "
          "(blocked layout)")
CHANGE = "none to the runs; probes at their end states"
PAIRING = "each probe against its own recorded run (reproduced exactly first)"
CONT_STEPS = 4 * EVAL_EVERY
BIND = 0.95
X_LAYOUT = os.path.join(HERE, "results", "X", "router_layout_results.json")
BLOCKED = trl.TASKS["blocked"]
CONTROL = ("hdr_far_nudge", 160)

SRC = {
    "hdr_far_nudge": dict(task=s6.HEADER, store=("explore", "far_express"), key="far_nudge",
                          make=lambda seed: s10.builder(s10.ARMS["far_nudge"], seed),
                          routing=s6.header_routing_stats, layout="header"),
    "hdr_far_A": dict(task=s6.HEADER, store=("explore", "far_cue"), key="far_A",
                      make=lambda seed: s6.builder_for("far_A")(s6.ARMS["far_A"], seed),
                      routing=s6.header_routing_stats, layout="header"),
    "blk_A": dict(task=BLOCKED, store=("x", X_LAYOUT), key="A_blocked",
                  make=lambda seed: trc.make_fn(trl.ARM["A_blocked"], seed),
                  routing=routing_stats, layout="blocked"),
}


def blocked_stalled(n=3):
    import json
    with open(X_LAYOUT) as f:
        d = json.load(f)["runs"]
    out = sorted(int(k.split("|")[1]) for k, r in d.items() if k.startswith("A_blocked|")
                 and r["end"].get("margin", 0) >= 0.9 and r["transition"] is None)
    return tuple(out[:n])


_A = dict(ec.ARM_A, lr=ec.SUB_LR, iters=ec.MAX_ITERS, prio=1)
ARMS = {
    "hdr_far_nudge": dict(_A, key="hdr_far_nudge", seeds=(160, 161, 164, 165),
                          label="batch 4's far_nudge (160 = control)"),
    "hdr_far_A": dict(_A, key="hdr_far_A", seeds=(165, 166), label="batch 2's far_A"),
    "blk_A": dict(_A, key="blk_A", seeds=blocked_stalled(), label="X's test_router_layout A_blocked"),
}


def recorded(arm, seed):
    kind, where = SRC[arm]["store"]
    if kind == "explore":
        return ec.load_store(where)["runs"][f"{SRC[arm]['key']}|{seed}"]
    import json
    with open(where) as f:
        return json.load(f)["runs"][f"{SRC[arm]['key']}|{seed}"]


# ── the probes' machinery ────────────────────────────────────────────────────
def rerun(arm, seed, iters):
    """onset_run with the recorded builder, task and lr; returns model, curve, rng, optimizer."""
    src, holder = SRC[arm], {}

    def hook(opt, args, kwargs):
        holder["opt"] = opt              # the last optimizer to step: the run's (after any nudge)

    h = register_optimizer_step_pre_hook(hook)
    try:
        m, curve, rng = tbo.onset_run(src["task"], src["make"](seed), seed, max_iters=iters,
                                      eval_every=EVAL_EVERY, data=eval_batch(src["task"]),
                                      lr=ARMS[arm]["lr"])
    finally:
        h.remove()
    return m, curve, rng, holder["opt"]


def forward_with_gate(m, tokens, g):
    m.attn.G = torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
    return BDH.forward(m, tokens)[0]


def learned_gate(m, x):
    return Instrument.gates(m, x, m.embed(x))[0]


def perfect_gate(task, x, cmap):
    pg = perfect_gate_general(x, task.ctx_tokens)                    # stream s -> channel s
    inv = [cmap.index(c) for c in range(len(cmap))]                  # channel c <- stream inv[c]
    return pg[..., inv]


@torch.no_grad()
def acc_with(m, task, data, gate_fn):
    was = m.training
    m.eval()
    try:
        x = data[:, :-1]
        ql, qt, _ = task.select(forward_with_gate(m, x, gate_fn(x)), data[:, 1:], None)
        return (ql.argmax(-1) == qt).float().mean().item(), F.cross_entropy(ql, qt).item()
    finally:
        m.train(was)


@torch.no_grad()
def stream_map(m, task, pinp):
    lab = task.stream_labels(pinp)
    g = learned_gate(m, pinp)
    body = torch.zeros_like(lab, dtype=torch.bool)
    body[:, :task.qpos[0] - 1] = True
    cmap = [int(g[(lab == s) & body].mean(0).argmax()) for s in range(task.S)]
    if len(set(cmap)) < task.S:
        return list(range(task.S)), True
    return cmap, False


def continue_run(m0, opt0, rng0, task, data, lr, gate_fn=None, steps=CONT_STEPS):
    """steps more updates from (m0, opt0, rng0), mirroring onset_run's step; gate_fn None = the
    learned gate (the model's own forward), else a fixed gate."""
    m = copy.deepcopy(m0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(opt0.state_dict())
    g = torch.Generator()
    g.set_state(rng0.get_state())
    dig, curve = hashlib.sha256(), []
    for step in range(1, steps + 1):
        tokens, _ = task.make_batch(BATCH, g)
        dig.update(tokens.numpy().tobytes())
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = m(inp, TAU_END)[0] if gate_fn is None else forward_with_gate(m, inp, gate_fn(inp))
        ql, qt, _ = task.select(logits, tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            if gate_fn is None:
                curve.append([step] + list(tbo.evaluate(m, task, data)))
            else:
                curve.append([step] + list(acc_with(m, task, data, gate_fn)))
    return curve, dig.hexdigest()


TYPES = ("CTX", "KEY", "VAL", "qCTX", "qKEY")


def positions(task):
    S, P = task.S, task.P
    q = task.qpos[0]
    if task.layout == "header":
        kpos, vpos, cpos = s6.positions(task)
        return dict(CTX=list(cpos), KEY=kpos.tolist(), VAL=vpos.tolist(), qCTX=[q - 1], qKEY=[q])
    n = S * P
    j = list(range(n))
    return dict(CTX=[3 * i for i in j], KEY=[3 * i + 1 for i in j], VAL=[3 * i + 2 for i in j],
                qCTX=[q - 1], qKEY=[q])


@torch.no_grad()
def profile(m, task, pinp, cmap):
    lab = task.stream_labels(pinp)
    gl = learned_gate(m, pinp)
    gp = perfect_gate(task, pinp, cmap)
    out = {}
    for t, pos in positions(task).items():
        pos = torch.tensor(pos)
        for s in range(task.S):
            sel = lab[:, pos] == s
            if not bool(sel.any()):
                continue
            lr_ = gl[:, pos, 0][sel].mean().item()
            out[f"{t}|{s}"] = dict(read=lr_, write=lr_, perfect=gp[:, pos, 0][sel].mean().item(),
                                   n=int(sel.sum()))
    return out


def reading(swap, cp, cl):
    reach = lambda cur: any(a >= BIND for _, a, _ in cur)
    if swap >= BIND:
        return "SWAP-FIXES"
    if reach(cp) and not reach(cl):
        return "MEMORY-STUCK"
    if reach(cp) and reach(cl):
        return "SLOW-ONLY"
    return "NEITHER"


def run_job(arm, seed):
    src = SRC[arm]
    task = src["task"]
    rec = recorded(arm, seed)
    cap = ARMS[arm]["iters"]                   # MAX_ITERS; lower only in explore_batch5 --dry
    dry = cap < ec.MAX_ITERS
    iters = min(rec["stopped_at"], cap)
    want = [c for c in rec["curve"] if c[0] <= iters]
    cont = EVAL_EVERY if dry else CONT_STEPS
    m, curve, rng, opt = rerun(arm, seed, iters)
    data = eval_batch(task)
    pinp, plab, prole = probe_batch(task, seed)
    end = trl.layout_stats(m, task, (pinp, plab, prole), "end") if task.layout != "header" else \
        s6.header_stats(m, task, (pinp, plab, prole), "end")
    out = dict(ok=True, seed=seed, k=2, layout=src["layout"], iters=iters, curve=curve,
               transition=tbo.transition(curve), acc=curve[-1][1] if curve else float("nan"),
               val_cos=end["val_cos"], key_cos=end["key_cos"], ctx_cos=end["ctx_cos"], end=end,
               stats=[], recorded_acc=want[-1][1] if want else float("nan"),
               recorded_transition=rec["transition"], reproduces=curve == want,
               control=(arm, seed) == CONTROL, dry=dry, cont_steps=cont)
    if not out["reproduces"]:
        out["reading"] = "UNTESTED"
        return out
    learned_acc = acc_with(m, task, data, lambda x: learned_gate(m, x))
    out["learned_path_acc"] = learned_acc[0]
    out["learned_path_ok"] = learned_acc[0] == want[-1][1]
    cmap, fallback = stream_map(m, task, pinp)
    out.update(cmap=cmap, cmap_identity_fallback=fallback)
    pg = lambda x: perfect_gate(task, x, cmap)
    out["swap_acc"], out["swap_loss"] = acc_with(m, task, data, pg)
    out["cont_perfect"], dp = continue_run(m, opt, rng, task, data, ARMS[arm]["lr"], gate_fn=pg,
                                           steps=cont)
    out["cont_learned"], dl = continue_run(m, opt, rng, task, data, ARMS[arm]["lr"], steps=cont)
    out["batches_identical"] = dp == dl
    out["profile"] = profile(m, task, pinp, cmap)
    out["margin_learned"] = src["routing"](m, task, (pinp, plab, prole))["margin"]
    if not out["learned_path_ok"] or not out["batches_identical"]:
        out["reading"] = "UNTESTED"
    else:
        out["reading"] = reading(out["swap_acc"], out["cont_perfect"], out["cont_learned"])
    return out


def segments(arm):
    a = ARMS[arm]
    src = SRC[arm]
    return [(a["iters"] + 2 * CONT_STEPS, src["task"], src["make"](a["seeds"][0]), a["lr"],
             s6.header_stats if src["layout"] == "header" else trl.layout_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    # (1) the perfect gate zeroes every cross-stream score at every layer, both layouts and maps
    zero = True
    for arm in ("hdr_far_A", "blk_A"):
        task = SRC[arm]["task"]
        m = SRC[arm]["make"](ARMS[arm]["seeds"][0])()
        m.eval()
        x = probe_batch(task, 0)[0][:32]
        lab = task.stream_labels(x)
        cross = (lab[:, :, None] != lab[:, None, :]).unsqueeze(1)
        for cmap in ([0, 1], [1, 0]):
            m.attn.record = []
            with torch.no_grad():
                forward_with_gate(m, x, perfect_gate(task, x, cmap))
            rec, m.attn.record = m.attn.record, None
            zero &= len(rec) == ARCH["n_layer"] and all(bool((sc[cross] == 0).all()) for sc in rec)
    # (2) the continuation code continues a run exactly; (3) identical batches; (4) SWAP's
    # learned-gate path reproduces the recorded accuracy
    arm, seed = CONTROL
    task = SRC[arm]["task"]
    rec = recorded(arm, seed)
    m, curve, rng, opt = rerun(arm, seed, 2 * EVAL_EVERY)
    data = eval_batch(task)
    same_prefix = curve == [c for c in rec["curve"] if c[0] <= 2 * EVAL_EVERY]
    cl, dl = continue_run(m, opt, rng, task, data, ARMS[arm]["lr"], steps=EVAL_EVERY)
    want = [c for c in rec["curve"] if c[0] == 3 * EVAL_EVERY]
    cont_ok = bool(want) and cl[-1][1:] == want[0][1:]
    cmap, _ = stream_map(m, task, probe_batch(task, seed)[0])
    _, dp = continue_run(m, opt, rng, task, data, ARMS[arm]["lr"],
                         gate_fn=lambda x: perfect_gate(task, x, cmap), steps=50)
    _, dl50 = continue_run(m, opt, rng, task, data, ARMS[arm]["lr"], steps=50)
    lp = acc_with(m, task, data, lambda x: learned_gate(m, x))
    lp_ok = lp[0] == rec["curve"][1][1]
    for nm, v in (("the perfect gate (maps identity and swapped) zeroes every cross-stream score at "
                   "every layer, header and blocked layouts", zero),
                  (f"far_nudge 160 re-run to 2400 reproduces its recorded curve, and 1200 continued "
                   f"updates (learned gate) reproduce its recorded evaluation at 3600 ({cl[-1][1:]} vs "
                   f"{want[0][1:] if want else None})", same_prefix and cont_ok),
                  ("the perfect-gate and learned-gate continuations see identical batches (digests)",
                   dp == dl50),
                  (f"SWAP's evaluation path with the learned gate reproduces the recorded accuracy at "
                   f"2400 ({lp[0]} vs {rec['curve'][1][1]})", lp_ok),
                  (f"the blocked runs are X's first 3 stalled A_blocked seeds: {ARMS['blk_A']['seeds']}",
                   ARMS["blk_A"]["seeds"] == (120, 130, 136))):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    print("  per run (rec = recorded final accuracy; rerun reproduces its recorded curve at every "
          "evaluation; SWAP = perfect gate at the end state; continuations: accuracy at +1200/+2400/"
          "+3600/+4800):")
    rows, readings = [], {}
    for arm, a in ARMS.items():
        for s in a["seeds"]:
            r = store["runs"].get(f"{arm}|{s}")
            if r is None or not r.get("ok"):
                print(f"    {arm:<14} {s:>4} {ec.tag(r)}")
                continue
            ctl = " (control)" if r.get("control") else ""
            if not r["reproduces"]:
                print(f"    {arm:<14} {s:>4}{ctl} UNTESTED: rerun differs from the recorded curve")
                readings.setdefault(("control" if ctl else r["layout"]), []).append((arm, s, "UNTESTED"))
                continue
            cp = "/".join(f"{x[1]:.2f}" for x in r["cont_perfect"])
            cl = "/".join(f"{x[1]:.2f}" for x in r["cont_learned"])
            print(f"    {arm:<14} {s:>4}{ctl}  rec {r['recorded_acc']:.3f}  rerun reproduces: yes  "
                  f"learned-path acc {r['learned_path_acc']:.3f} ({'=' if r['learned_path_ok'] else '!='} "
                  f"rec)  map stream->channel {r['cmap']}{' (IDENTITY FALLBACK: both streams on one channel)' if r['cmap_identity_fallback'] else ''}")
            print(f"    {'':<14} {'':>4}  SWAP {r['swap_acc']:.3f}   perfect-gate continuation {cp}   "
                  f"learned-gate continuation {cl}   batches identical: {r['batches_identical']}   "
                  f"margin (learned) {r['margin_learned']:.2f}   READING: {r['reading']}")
            prof = r["profile"]
            diff = [f"{t} s{sx}: {prof[f'{t}|{sx}']['read']:.2f} vs {prof[f'{t}|{sx}']['perfect']:.2f}"
                    for t in TYPES for sx in (0, 1) if f"{t}|{sx}" in prof
                    and abs(prof[f"{t}|{sx}"]["read"] - prof[f"{t}|{sx}"]["perfect"]) > 0.1]
            prof_s = "  ".join(f"{t}: " + "/".join(f"{prof[f'{t}|{sx}']['read']:.2f}" for sx in (0, 1)
                                                    if f"{t}|{sx}" in prof) for t in TYPES)
            print(f"    {'':<14} {'':>4}  gate profile, channel-0 read = write, streams 0/1: {prof_s}")
            print(f"    {'':<14} {'':>4}  differs from perfect (> 0.1): {'; '.join(diff) if diff else 'nowhere'}")
            readings.setdefault(("control" if ctl else r["layout"]), []).append((arm, s, r["reading"]))
    print("  SUMMARY of readings (stalled runs by layout; control apart):")
    for lay in ("header", "blocked", "control"):
        items = readings.get(lay, [])
        cnt = {}
        for _, _, rd in items:
            cnt[rd] = cnt.get(rd, 0) + 1
        print(f"    {lay:<8} {cnt}   " + ", ".join(f"{a} {s}: {rd}" for a, s, rd in items))
    return dict(readings=readings)
