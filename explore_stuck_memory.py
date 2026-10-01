#!/usr/bin/env python
"""
explore_stuck_memory.py — EXPLORATORY, not a result. Screen S18 (batch 6), diagnostic, no verdict:
what do the stalled routed runs answer, and does a long run with the perfect gate free them?

BACKGROUND (post hoc, from batch 5's S14)
- In far_nudge 164, 165 and far_A 165, 166 (header layout) the learned gate matches the perfect
  gate at keys, values and query keys; swapping in the perfect gate, and 4800 more updates with it,
  leave them at 0.52-0.55. Blocked-layout A 120, 130, 136 stay at 0.32-0.34 the same way.
- far_nudge 161 is a gate failure (body keys 0.25/0.75); it rebinds with the perfect gate in 3600
  updates.
- With SLOW_MEM's schedule the same header seeds bound 10/10 (S15), all three far_nudge stalls
  included; far_nudge 165 had routed by 1200 and still stalled at full speed.

RUNS: S14's 8 stalled runs, each re-run to its recorded end state through S14's own path
(explore_stall_probe.rerun: onset_run with the recorded builder, task and lr 1e-3 =
test_channel_binding.SUB_LR), reproducing the recorded curve at every evaluation (a rerun that
does not is reported UNTESTED and gets no probe):
  header   hdr_far_nudge 161, 164, 165 (batch 4's far_nudge); hdr_far_A 165, 166 (batch 2's far_A)
  blocked  blk_A 120, 130, 136 (X's test_router_layout A_blocked)

(a) BREAKDOWN at the end state, on the held-out evaluation batch (eval_batch, 2048 queries), with
    the learned gate (the evaluation's own forward) and with the perfect gate (S14's SWAP: the
    perfect gate with channels assigned by the learned stream-to-channel map):
    - accuracy by whether the queried stream's block comes first or last in the sequence;
    - accuracy by the queried key's position within its block (0..P-1);
    - the share of wrong answers that are the same key's value in the other stream.
    The queried pair is located in the body (the pair whose stream and key match the query's);
    pairs are indexed in order of appearance, block = index // P, position = index % P.
(b) LONG CONTINUATION: 24000 more updates from the end state with the perfect gate fixed (S14's
    continuation, line for line, run longer: a copy of the model and of the optimizer's state, the
    run's own batch stream continued from its generator state); evaluation every 1200 with the
    perfect gate. Its first 4800 updates must reproduce S14's recorded perfect-gate continuation.
    The perfect-gate breakdown is repeated at its end (extra).
    NOTE (a bug in batch 5's S14, found while writing this screen): S14's continue_run loaded the
    run's optimizer state with opt.load_state_dict(opt0.state_dict()), which SHARES the state
    tensors (exp_avg, exp_avg_sq, step) with the run's optimizer (torch 2.14; data pointers equal).
    Its perfect-gate continuation, run first, was correct and advanced that state in place; its
    learned-gate continuation (the control) then started from the end-state weights with Adam's
    moments and step count from the end of the perfect-gate continuation. SWAP and the perfect-gate
    continuations are unaffected; so are the NEITHER and SWAP-FIXES readings; the MEMORY-STUCK
    reading (far_nudge 161) depends on the control. Here every continuation deep-copies the state
    (continue_long), and the run's optimizer is checked unchanged after it.
(c) EXTRA (not in the readings; corrects S14): S14's learned-gate continuation redone correctly,
    4800 updates from the end state with a deep copy of the run's optimizer state; S14's reading
    recomputed with it (explore_stall_probe.reading, unchanged).

READINGS per run (fixed before any run), printed:
  "recency" if, with the learned gate, last-block queries >= 0.9 and first-block ones <= 0.2;
  "primacy" for the reverse; otherwise neither. Separately: "slow" if (b) reaches >= 0.95 at
  some evaluation (the update is given), "basin" if it does not.

CHECKS: every rerun reproduces its recorded curve (each job); the breakdown's overall accuracy
equals the evaluation's (each job: the learned gate = evaluate's accuracy = the recorded final
accuracy; the perfect gate = S14's SWAP path, and S14's recorded SWAP accuracy); before the batch:
the breakdown's bookkeeping on synthetic answers on both layouts (the true values: accuracy 1; the
other stream's values: accuracy 0, other-stream share 1; the located pair holds the query's value;
the blocks and positions partition the queries), its overall accuracy on untrained models of both
layouts equals evaluate's and S14's SWAP path's, and the long continuation's code equals S14's
continue_run (curves and batch digests) on far_nudge 160 re-run to 1200, given a copy of the
optimizer; continue_long leaves the run's optimizer state unchanged, while S14's continue_run
changes the optimizer it is given (the bug above, shown). In each job: the run's optimizer state is
unchanged after (b) and (c).
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
import explore_far_cue as s6
import explore_stall_probe as s14
import test_binding_onset as tbo
import test_router_layout as trl
from test_binding_onset import BATCH, EVAL_EVERY, eval_batch
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

NAME = "stuck_memory"
IDEA = "what the stalled routed runs answer (block order, key position, the other stream's value), and a long perfect-gate run"
SOURCE = "batch 5's S14 (the stalls are not in the gate) and S15 (SLOW_MEM's schedule prevents them)"
CHANGE = "none to the runs; probes at their end states"
PAIRING = "each probe against its own recorded run (reproduced exactly first) and S14's probes"
LONG_STEPS = 20 * EVAL_EVERY
BIND = 0.95
REC_HI, REC_LO = 0.9, 0.2
S14_STORE = "stall_probe"
S14_CONT = s14.CONT_STEPS

_A = dict(ec.ARM_A, lr=ec.SUB_LR, iters=ec.MAX_ITERS, prio=1)
ARMS = {
    "hdr_far_nudge": dict(_A, key="hdr_far_nudge", seeds=(161, 164, 165), label="batch 4's far_nudge"),
    "hdr_far_A": dict(_A, key="hdr_far_A", seeds=(165, 166), label="batch 2's far_A"),
    "blk_A": dict(_A, key="blk_A", seeds=(120, 130, 136), label="X's test_router_layout A_blocked"),
}


# ── (a) the breakdown ────────────────────────────────────────────────────────
def pair_positions(task):
    if task.layout == "header":
        kpos, vpos, _ = s6.positions(task)
        return kpos, vpos
    j = torch.arange(task.S * task.P)
    return 3 * j + 1, 3 * j + 2


def locate(task, x):
    """For each sequence: the queried pair's index (order of appearance), and the value of the same
    key in the other stream. x = the input tokens (data[:, :-1])."""
    assert task.n_q == 1 and task.S == 2
    kpos, vpos = pair_positions(task)
    lab = task.stream_labels(x)[:, kpos]
    keys = x[:, kpos]
    q = task.qpos[0]
    qs, qk = x[:, q - 1], x[:, q]
    hit = (lab == qs[:, None]) & (keys == qk[:, None])
    oth = (lab != qs[:, None]) & (keys == qk[:, None])
    assert bool((hit.sum(1) == 1).all()) and bool((oth.sum(1) == 1).all())
    j, jo = hit.float().argmax(1), oth.float().argmax(1)
    rows = torch.arange(x.shape[0])
    return dict(j=j, val=x[:, vpos][rows, j], other_val=x[:, vpos][rows, jo])


def tally(task, pred, qt, loc):
    P = task.P
    correct = pred == qt
    blk, pos = loc["j"] // P, loc["j"] % P
    wrong = ~correct
    nw = int(wrong.sum())
    acc_of = lambda mask: ((correct & mask).sum().item() / int(mask.sum())) if int(mask.sum()) else None
    out = dict(acc=correct.float().mean().item(), n=int(qt.shape[0]), n_wrong=nw,
               other_share=(pred[wrong] == loc["other_val"][wrong]).float().mean().item() if nw else None,
               lookup_ok=bool((loc["val"] == qt).all()),
               block={b: dict(acc=acc_of(blk == b), n=int((blk == b).sum())) for b in range(task.S)},
               pos={i: dict(acc=acc_of(pos == i), n=int((pos == i).sum())) for i in range(P)},
               block_pos={f"{b}|{i}": acc_of((blk == b) & (pos == i)) for b in range(task.S)
                          for i in range(P)})
    return out


@torch.no_grad()
def breakdown(m, task, data, gate_fn=None):
    """The evaluation's computation (tbo.evaluate, or S14's acc_with for a fixed gate), per query."""
    was = m.training
    m.eval()
    try:
        x = data[:, :-1]
        logits = tbo.logits_of(m, x) if gate_fn is None else s14.forward_with_gate(m, x, gate_fn(x))
        ql, qt, _ = task.select(logits, data[:, 1:], None)
        pred = ql.argmax(-1)
    finally:
        m.train(was)
    return tally(task, pred, qt, locate(task, x))


# ── (b) the long continuation: S14's continue_run line for line, also returning the model, with
#    the optimizer's state deep-copied (S14's load_state_dict shared it with the run's optimizer) ──
def opt_state(opt):
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
            for g in opt.param_groups for p in g["params"]]


def same_state(a, b):
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k]
                                     for k in x) for x, y in zip(a, b))


def clone_opt(m, opt0, lr):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(opt0.state_dict()))
    return opt


def continue_long(m0, opt0, rng0, task, data, lr, gate_fn, steps=LONG_STEPS):
    m = copy.deepcopy(m0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(opt0.state_dict()))
    g = torch.Generator()
    g.set_state(rng0.get_state())
    dig, curve = hashlib.sha256(), []
    for step in range(1, steps + 1):
        tokens, _ = task.make_batch(BATCH, g)
        dig.update(tokens.numpy().tobytes())
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = m(inp, TAU_END)[0] if gate_fn is None else s14.forward_with_gate(m, inp, gate_fn(inp))
        ql, qt, _ = task.select(logits, tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            if gate_fn is None:
                curve.append([step] + list(tbo.evaluate(m, task, data)))
            else:
                curve.append([step] + list(s14.acc_with(m, task, data, gate_fn)))
    return curve, dig.hexdigest(), m


def order_reading(bd):
    f, l = bd["block"][0]["acc"], bd["block"][1]["acc"]
    if l >= REC_HI and f <= REC_LO:
        return "recency"
    if f >= REC_HI and l <= REC_LO:
        return "primacy"
    return "neither"


def first_reach(curve):
    return next((s for s, a, _ in curve if a >= BIND), None)


def run_job(arm, seed):
    src = s14.SRC[arm]
    task = src["task"]
    rec = s14.recorded(arm, seed)
    cap = ARMS[arm]["iters"]                   # MAX_ITERS; lower only in explore_batch6 --dry
    dry = cap < ec.MAX_ITERS
    iters = min(rec["stopped_at"], cap)
    want = [c for c in rec["curve"] if c[0] <= iters]
    long = EVAL_EVERY if dry else LONG_STEPS
    m, curve, rng, opt = s14.rerun(arm, seed, iters)
    data = eval_batch(task)
    pinp, plab, prole = probe_batch(task, seed)
    end = trl.layout_stats(m, task, (pinp, plab, prole), "end") if task.layout != "header" else \
        s6.header_stats(m, task, (pinp, plab, prole), "end")
    out = dict(ok=True, seed=seed, k=2, layout=src["layout"], iters=iters, curve=curve,
               transition=tbo.transition(curve), acc=curve[-1][1] if curve else float("nan"),
               val_cos=end["val_cos"], key_cos=end["key_cos"], ctx_cos=end["ctx_cos"], end=end,
               stats=[], recorded_acc=want[-1][1] if want else float("nan"),
               recorded_transition=rec["transition"], reproduces=curve == want, dry=dry,
               long_steps=long)
    if not out["reproduces"]:
        out["reading"] = "UNTESTED"
        return out
    eval_acc = tbo.evaluate(m, task, data)[0]
    cmap, fallback = s14.stream_map(m, task, pinp)
    pg = lambda x: s14.perfect_gate(task, x, cmap)
    swap_acc = s14.acc_with(m, task, data, pg)[0]
    bl, bp = breakdown(m, task, data), breakdown(m, task, data, pg)
    out.update(cmap=cmap, cmap_identity_fallback=fallback, eval_acc=eval_acc, swap_acc=swap_acc,
               bd_learned=bl, bd_perfect=bp,
               bd_learned_ok=bl["acc"] == eval_acc == want[-1][1] and bl["lookup_ok"],
               bd_perfect_ok=bp["acc"] == swap_acc and bp["lookup_ok"])
    if not dry:
        s14r = ec.load_store(S14_STORE)["runs"].get(f"{arm}|{seed}")
        out["s14_swap_acc"] = s14r["swap_acc"] if s14r else None
        out["bd_perfect_ok"] &= s14r is not None and swap_acc == s14r["swap_acc"]
    snap = opt_state(opt)
    cont, dig, m2 = continue_long(m, opt, rng, task, data, ARMS[arm]["lr"], pg, steps=long)
    out.update(cont_perfect=cont, cont_digest=dig, cont_first=first_reach(cont),
               bd_after=breakdown(m2, task, data, pg))
    del m2
    # (c) extra: S14's learned-gate continuation, corrected (the run's own optimizer state)
    short = EVAL_EVERY if dry else S14_CONT
    cl, dl, _ = continue_long(m, opt, rng, task, data, ARMS[arm]["lr"], None, steps=short)
    out.update(cont_learned_fixed=cl, cont_learned_digest=dl, opt_unchanged=same_state(snap, opt_state(opt)),
               s14_reading_fixed=s14.reading(swap_acc, cont[:short // EVAL_EVERY], cl))
    if not dry:
        out["cont_matches_s14"] = (s14r is not None
                                   and cont[:S14_CONT // EVAL_EVERY] == s14r["cont_perfect"])
        out["s14_reading"] = s14r["reading"] if s14r else None
        out["s14_cont_learned"] = s14r["cont_learned"] if s14r else None
    out["order"] = order_reading(bl)
    out["speed"] = "slow" if out["cont_first"] is not None else "basin"
    out["reading"] = f"{out['order']}; {out['speed']}"
    return out


def segments(arm):
    a = ARMS[arm]
    src = s14.SRC[arm]
    return [(a["iters"] + LONG_STEPS, src["task"], src["make"](a["seeds"][0]), a["lr"],
             s6.header_stats if src["layout"] == "header" else trl.layout_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    s14runs = ec.load_store(S14_STORE)["runs"]
    want = {(a, s) for a, d in ARMS.items() for s in d["seeds"]}
    stalled = {tuple(k.split("|")) for k, r in s14runs.items() if r.get("reproduces")
               and not r.get("control")}
    stalled = {(a, int(s)) for a, s in stalled}
    runs_ok = want <= stalled and stalled == {(a, s) for a, d in s14.ARMS.items() for s in d["seeds"]
                                              if (a, s) != s14.CONTROL} and s14.blocked_stalled() == (120, 130, 136)
    # bookkeeping on synthetic answers, both layouts
    book = True
    acc_same = True
    for arm in ("hdr_far_A", "blk_A"):
        task = s14.SRC[arm]["task"]
        data = eval_batch(task)
        x = data[:, :-1]
        loc = locate(task, x)
        _, qt, _ = task.select(torch.zeros(x.shape[0], x.shape[1], task.vocab), data[:, 1:], None)
        t_true = tally(task, qt, qt, loc)
        t_other = tally(task, loc["other_val"], qt, loc)
        book &= (t_true["acc"] == 1.0 and t_true["lookup_ok"] and t_other["acc"] == 0.0
                 and t_other["other_share"] == 1.0
                 and sum(v["n"] for v in t_true["block"].values()) == t_true["n"]
                 and sum(v["n"] for v in t_true["pos"].values()) == t_true["n"]
                 and min(v["n"] for v in t_true["block"].values()) > 0.4 * t_true["n"])
        m = s14.SRC[arm]["make"](ARMS[arm]["seeds"][0])()
        pinp = probe_batch(task, 0)[0]
        cmap, _ = s14.stream_map(m, task, pinp)
        pg = lambda xx: s14.perfect_gate(task, xx, cmap)
        acc_same &= (breakdown(m, task, data)["acc"] == tbo.evaluate(m, task, data)[0]
                     and breakdown(m, task, data, pg)["acc"] == s14.acc_with(m, task, data, pg)[0])
    # the long continuation's code = S14's continue_run
    arm, seed = s14.CONTROL
    task = s14.SRC[arm]["task"]
    m, curve, rng, opt = s14.rerun(arm, seed, EVAL_EVERY)
    data = eval_batch(task)
    cmap, _ = s14.stream_map(m, task, probe_batch(task, seed)[0])
    pg = lambda xx: s14.perfect_gate(task, xx, cmap)
    lr = ARMS["hdr_far_A"]["lr"]
    snap = opt_state(opt)
    c_mine, d_mine, _ = continue_long(m, opt, rng, task, data, lr, pg, steps=EVAL_EVERY)
    mine_leaves = same_state(snap, opt_state(opt))
    opt_b = clone_opt(m, opt, lr)
    c_s14, d_s14 = s14.continue_run(m, opt_b, rng, task, data, lr, gate_fn=pg, steps=EVAL_EVERY)
    s14_changes = not same_state(snap, opt_state(opt_b))
    rec = s14.recorded(arm, seed)
    cont_same = (curve == [c for c in rec["curve"] if c[0] <= EVAL_EVERY]
                 and c_mine == c_s14 and d_mine == d_s14)
    for nm, v in ((f"the runs are S14's 8 stalled runs, each reproduced in S14 "
                   f"({sorted(want)})", runs_ok),
                  ("the breakdown's bookkeeping on synthetic answers, header and blocked layouts (true "
                   "values: accuracy 1; the other stream's values: accuracy 0, other-stream share 1; the "
                   "located pair holds the query's value; blocks and positions partition the queries)",
                   book),
                  ("the breakdown's overall accuracy equals evaluate's (learned gate) and S14's SWAP "
                   "path's (perfect gate) on untrained models of both layouts", acc_same),
                  (f"the long continuation's code equals S14's continue_run (far_nudge 160 re-run to "
                   f"{EVAL_EVERY}, reproducing its record, then {EVAL_EVERY} perfect-gate updates: curves "
                   f"{c_mine} vs {c_s14}, batch digests equal: {d_mine == d_s14}; S14's run given a copy "
                   f"of the optimizer)", cont_same),
                  ("continue_long leaves the run's optimizer state unchanged; S14's continue_run changes "
                   "the optimizer it is given (batch 5's S14 bug, shown: its learned-gate control started "
                   "from the state its perfect-gate continuation left)", mine_leaves and s14_changes)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def bd_str(bd):
    fa = lambda v: "  -- " if v is None else f"{v:5.3f}"
    pos = "/".join(fa(bd["pos"][i]["acc"]).strip() for i in sorted(bd["pos"], key=int))
    oth = "--" if bd["other_share"] is None else f"{bd['other_share']:.3f}"
    return (f"overall {bd['acc']:.3f}  first-block {fa(bd['block'][0]['acc'])} (n {bd['block'][0]['n']})  "
            f"last-block {fa(bd['block'][1]['acc'])} (n {bd['block'][1]['n']})  key position 0-3 {pos}  "
            f"wrong {bd['n_wrong']}: other stream's value {oth}")


def bp_str(bd, S=2, P=4):
    fa = lambda v: "--" if v is None else f"{v:.2f}"
    return "  ".join(("first" if b == 0 else "last") + " " +
                     "/".join(fa(bd["block_pos"][f"{b}|{i}"]) for i in range(P)) for b in range(S))


def norm(bd):
    """JSON turns the int keys into strings; accept both."""
    return dict(bd, block={int(k): v for k, v in bd["block"].items()},
                pos={int(k): v for k, v in bd["pos"].items()})


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    print("  per run (rec = recorded final accuracy; (a) on the held-out batch, 2048 queries; (b) "
          f"{LONG_STEPS} perfect-gate updates from the end state, accuracy every {EVAL_EVERY}):")
    readings = {}
    for arm, a in ARMS.items():
        for s in a["seeds"]:
            r = store["runs"].get(f"{arm}|{s}")
            if r is None or not r.get("ok"):
                print(f"    {arm:<14} {s:>4} {ec.tag(r)}")
                continue
            if not r["reproduces"]:
                print(f"    {arm:<14} {s:>4} UNTESTED: rerun differs from the recorded curve")
                readings.setdefault(r["layout"], []).append((arm, s, "UNTESTED"))
                continue
            bl, bp, ba = norm(r["bd_learned"]), norm(r["bd_perfect"]), norm(r["bd_after"])
            chk = (f"breakdown = evaluation: learned {'yes' if r['bd_learned_ok'] else 'NO'}, perfect "
                   f"{'yes' if r['bd_perfect_ok'] else 'NO'}"
                   + ("" if r["dry"] else f"; (b)'s first {S14_CONT} updates = S14's: "
                      f"{'yes' if r.get('cont_matches_s14') else 'NO'}"))
            print(f"    {arm:<14} {s:>4}  rec {r['recorded_acc']:.3f}  rerun reproduces: yes  map "
                  f"stream->channel {r['cmap']}{' (IDENTITY FALLBACK)' if r['cmap_identity_fallback'] else ''}"
                  f"  {chk}")
            print(f"    {'':<14} {'':>4}  (a) learned gate:  {bd_str(bl)}")
            print(f"    {'':<14} {'':>4}      by block x key position: {bp_str(bl)}")
            print(f"    {'':<14} {'':>4}  (a) perfect gate:  {bd_str(bp)}")
            print(f"    {'':<14} {'':>4}      by block x key position: {bp_str(bp)}")
            cv = " ".join(f"{x[1]:.2f}" for x in r["cont_perfect"])
            fr = r["cont_first"]
            print(f"    {'':<14} {'':>4}  (b) perfect-gate continuation: {cv}   first >= {BIND}: "
                  f"{'+' + str(fr) if fr is not None else 'never'}")
            print(f"    {'':<14} {'':>4}      after (b), perfect gate (extra): {bd_str(ba)}")
            clf = "/".join(f"{x[1]:.2f}" for x in r["cont_learned_fixed"])
            old = r.get("s14_cont_learned")
            print(f"    {'':<14} {'':>4}  (c) extra, S14's learned-gate continuation corrected: {clf}"
                  + (f" (batch 5's: {'/'.join(f'{x[1]:.2f}' for x in old)})" if old else "")
                  + f"; S14 reading with it: {r['s14_reading_fixed']}"
                  + (f" (batch 5's: {r['s14_reading']})" if r.get("s14_reading") else "")
                  + f"; run's optimizer unchanged by (b), (c): {'yes' if r['opt_unchanged'] else 'NO'}")
            print(f"    {'':<14} {'':>4}  READING: {r['order']}; {r['speed']}"
                  + (f" (>= {BIND} at +{fr})" if fr is not None else ""))
            readings.setdefault(r["layout"], []).append((arm, s, r["reading"]))
    print("  SUMMARY of readings (by layout):")
    for lay in ("header", "blocked"):
        items = readings.get(lay, [])
        print(f"    {lay:<8} " + ", ".join(f"{a} {s}: {rd}" for a, s, rd in items))
    fixed = [(arm, s, r["s14_reading"], r["s14_reading_fixed"]) for arm, a in ARMS.items()
             for s in a["seeds"] if (r := store["runs"].get(f"{arm}|{s}")) and r.get("ok")
             and r.get("reproduces") and r.get("s14_reading")]
    if fixed:
        print("  EXTRA, S14's readings with the corrected learned-gate control (batch 5's -> corrected): "
              + ", ".join(f"{a} {s}: {o} -> {n}" for a, s, o, n in fixed))
    return dict(readings=readings, s14_fixed=fixed)
