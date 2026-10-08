#!/usr/bin/env python
"""
explore_h_collapse_child.py — EXPLORATORY, not a result. Session H's S58 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e), as batches 10-18's. Parent: explore_h_collapse (the readings are
in its docstring). Imports batch 16/17/18 modules read-only (explore_b16_main, explore_b16_gates, explore_window_d8_child,
explore_split_child, explore_split_target_child, explore_b18_gates); edits none.

CONFIGURATION: batch 16's S44(a) path (explore_b16_main config "k16": test_stream_recipe.run_attempt with ARM["A4k16"]:
S=4, P=4, k=16, conv; its restart check recorded, never applied), LOCAL3's gate (explore_b16_gates.to_local), the SLOW
schedule (Adam: gate W_in, W_g, window at 1e-3 throughout; every other trainable parameter 1e-4 for updates 1-2400, 1e-3
after), no hinge, evaluation every 1200, 28800 updates. This is batch 16's LOCAL3_SLOW16_A recipe exactly (CHECK: NONE|240
equals its record bit for bit).

ARMS (each = NONE + one change):
  NONE       batch 16's LOCAL3_SLOW16_A.
  SPLIT      + S48's KEYMASS plateau trigger (explore_window_d8_child.make_trigger, read-only: checks every 2400 from 4800,
             reference 2400, through 26400, on S36's 64-sequence probe; fires if probe acc < 0.95 and it rose < 0.02 since
             the previous check; at most 3 splits, >= 4800 apart; S24's split of W_g's row c* (KEYMASS) onto c0, then W_g's
             Adam state zeroed). onset_run's Adam built through S36's capturing factory (the same constructor).
  RESET      the same trigger (make_trigger_op below: S48's make_trigger line for line with the operation as a parameter;
             CHECK: with the split as the operation it equals S48's row for row), the operation = W_g's Adam state zeroed
             only (explore_b18_gates.zero_state: S36's split_op's reset lines, no row copy). The targets c*, c0 are computed
             and logged as for SPLIT but not used. Self-contained (not timed from SPLIT's runs): RESET equals SPLIT bit for
             bit until the first firing, then each run follows its own plateaus.
  GUMBEL_W   the write gate g^w = softmax(z + n), n ~ Gumbel(0, 1) i.i.d. per (sequence, position, channel), drawn from the
             run's own generator (torch.Generator seeded seed + 58_000; never the global RNG), on training forwards only
             (model.training and grad enabled); the read gate g^r = softmax(z) (no noise). No annealing, no split.
  GUMBEL_RW  the same noise, one draw per forward, on the gate logits for both sides: g^r = g^w = softmax(z + n). No split.
  SWITCH     no split, no noise; L_aux = 1e-3 * k * sum_c f_c P_c added to the training loss (test_slow_start.Inject, the
             hinge's injection node, through a forward hook), with P_c = the mean of g_t[c] and f_c = the fraction of tokens
             whose argmax_c g_t = c, both over every input position of the training batch (B x T); g = the gate (read =
             write for LOCAL3). f is not differentiated (Switch's form).
  ORACLE     the perfect gate (test_stream_recipe.ARM["ceiling4k16"]: stream s -> channel s at every position, channels
             4-15 unused) + conv on the same path with its own recipe (one Adam at 1e-3 for every parameter), main's ceiling
             arms' recipe. Validity only.
MEASURED (every learned arm): the run path's statistics at every evaluation (routing_k's map at VAL positions, eta^2 by
stream / key / block / index at KEY and VAL, the query routing margin), plus key_routing below at every evaluation (the
stream -> channel map from the read gate at KEY positions, test_scale_axes.routing_k's rule moved from 3j+2 to 3j+1);
SWITCH: L_aux, max_c f_c and the number of argmax-used channels on the last training batch before each evaluation;
GUMBEL: the number of noisy forwards.
"""

import contextlib
import json
import math

import torch
import torch.nn.functional as F

import explore_main9c as mt
import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import explore_split_target_child as s37c
import explore_b18_gates as g18
import explore_muon_k16_child as s35c
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

CFG = "k16"
EVERY, FIRST, REF_AT = s48c.EVERY, s48c.FIRST, s48c.REF_AT          # 2400, 4800, 2400
ACC_THR, RISE, GAP = s48c.ACC_THR, s48c.RISE, s48c.GAP              # 0.95, 0.02, 4800
MAX_SPLITS = 3                                                      # S48's
GUMBEL_OFFSET = 58_000
SWITCH_ALPHA = 1e-3
TOTAL = 28800
ARM = {
    "NONE": dict(),
    "SPLIT": dict(trigger="split"),
    "RESET": dict(trigger="reset"),
    "GUMBEL_W": dict(gumbel="w"),
    "GUMBEL_RW": dict(gumbel="rw"),
    "SWITCH": dict(switch=SWITCH_ALPHA),
    "ORACLE": dict(oracle=True),
}


# ── Gumbel gate ──────────────────────────────────────────────────────────────
def gumbel_noise(shape, gen):
    """Gumbel(0, 1) samples: -log(-log U), U ~ Uniform[0, 1) from gen, clamped away from 0 (finite everywhere)."""
    u = torch.rand(shape, generator=gen).clamp_(min=1e-20)
    return -torch.log(-torch.log(u))


class GumbelLocalBDH(g16.LocalBDH):
    """LOCAL3 (explore_b16_gates.LocalBDH) with Gumbel noise on the gate logits on training forwards: side 'w' -> the write
    gate only, 'rw' -> both gates (one draw). gumbel_scale multiplies the noise (1.0 in the arms; 0.0 in the CHECK)."""

    def local_gates(self, v):
        u = self.gate_conv(v)
        h = torch.tanh(u @ self.W_in.T)
        z = h @ self.W_g.T
        g = F.softmax(z, dim=-1)
        if self.training and torch.is_grad_enabled():
            n = gumbel_noise(z.shape, self.gumbel_gen)
            gn = F.softmax(z + self.gumbel_scale * n, dim=-1)
            self.gumbel_calls += 1
            return (g, gn) if self.gumbel_side == "w" else (gn, gn)
        return g, g


def to_gumbel(side, scale=1.0):
    def conv(m, seed):
        assert type(m) is g16.LocalBDH, type(m)
        m.__class__ = GumbelLocalBDH
        m.gumbel_side = side
        m.gumbel_scale = float(scale)
        m.gumbel_gen = torch.Generator().manual_seed(int(seed) + GUMBEL_OFFSET)
        m.gumbel_calls = 0
        return m
    return conv


# ── Switch balance loss ──────────────────────────────────────────────────────
def switch_loss(g, alpha):
    """alpha * k * sum_c f_c P_c over every position of g (B, T, k); f from the argmax (no gradient)."""
    k = g.shape[-1]
    P = g.mean((0, 1))
    f = F.one_hot(g.detach().argmax(-1), k).to(g.dtype).mean((0, 1))
    return alpha * k * (f * P).sum(), f


def attach_switch(alpha, log):
    def conv(m, seed):
        def hook(mod, args, out):
            if not (mod.training and torch.is_grad_enabled()):
                return None
            logits, sat, gr, gw = out
            L, f = switch_loss(gr, alpha)
            log["last"] = dict(aux=float(L.detach()), fmax=float(f.max()), used=int((f > 0).sum()))
            log["n"] = log.get("n", 0) + 1
            return tss.Inject.apply(logits, L), sat, gr, gw
        m.register_forward_hook(hook)
        return m
    return conv


# ── The trigger with the operation as a parameter (RESET) ────────────────────
def make_trigger_op(task, seed, info, opt_of, last, op, max_splits=MAX_SPLITS, gap=GAP, thr=ACC_THR, rise=RISE):
    """explore_window_d8_child.make_trigger line for line, except that the operation applied at a firing is op(m, o, c*,
    c0, noise_seed) -> dict (split: S36's split_op; reset: W_g's optimizer state zeroed only)."""
    data = s36c.probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        (kc, k0, km), (ac, a0, am) = s37c.select(s37c.read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= max_splits else "gap"
        elif eligible:
            res = op(m, opt_of(), kc, k0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in s37c.select(s37c.read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def reset_op(m, o, cs, c0, seed):
    W = m.W_g
    w0 = W.detach().clone()
    zeroed = g18.zero_state(o, W)
    return dict(state_zeroed=zeroed, w_g_unchanged=bool(torch.equal(w0, W.detach())))


# ── Routing at KEY positions ─────────────────────────────────────────────────
@torch.no_grad()
def key_routing(m, task, probe):
    """routing_k's stream -> channel map with the read gate at KEY positions 3j+1 (routing_k uses VAL, 3j+2)."""
    was = m.training
    m.eval()
    try:
        pinp = probe[0]
        gr = m(pinp, TAU_END)[2]
    finally:
        m.train(was)
    n = task.S * task.P
    j = torch.arange(n)
    ctx = pinp[:, 3 * j]
    gk = gr[:, 3 * j + 1]
    means = [gk[ctx == s].mean(0) for s in range(task.S)]
    cmap = [int(x.argmax()) for x in means]
    return dict(ch_map_key=cmap, one_to_one_key=len(set(cmap)) == task.S,
                stream_gate_key=[[round(v, 4) for v in x.tolist()] for x in means])


# ── Runs ─────────────────────────────────────────────────────────────────────
def build_rc(arm_name, p, holder, out, info, log):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    rprobe = probe_batch(task, p["seed"])

    def probe(m, step):
        if p.get("probe", True):
            out[str(step)] = dict(key_routing(m, task, rprobe), **({"switch": dict(log["last"])} if log.get("last") else {}),
                                  **({"gumbel_calls": m.gumbel_calls} if hasattr(m, "gumbel_calls") else {}))
    convert = [g16.to_local]
    if arm.get("gumbel"):
        convert.append(to_gumbel(arm["gumbel"], p.get("gumbel_scale", 1.0)))
    if arm.get("switch") is not None:
        convert.append(attach_switch(p.get("alpha", arm["switch"]), log))
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=tuple(convert), probe=probe, keep=p.get("keep"))
    if arm.get("trigger"):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        last = p.get("total", TOTAL) - EVERY
        thr = p.get("thr", ACC_THR)
        if arm["trigger"] == "split":
            trig = s48c.make_trigger(task, p["seed"], info, opt_of, last, max_splits=MAX_SPLITS, thr=thr)
        else:
            trig = make_trigger_op(task, p["seed"], info, opt_of, last, reset_op, thr=thr)
        rc = s36c.with_trigger(rc, trig)
    else:
        ctx = contextlib.nullcontext()
    return rc, ctx


def run(p):
    """p: arm, seed, iters; CHECK knobs: thr / total (trigger), gumbel_scale, alpha, keep, probe, eval_every."""
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        rec = b16.path(CFG, s35c.arm("CEIL"), p["seed"], p["iters"], None, eval_every=p.get("eval_every"))
        return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="oracle")
    holder, out, info, log = {}, {}, dict(checks=[]), {}
    rc, ctx = build_rc(p["arm"], p, holder, out, info, log)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", keyroute=out, groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                      switch_n=log.get("n"), gumbel=arm.get("gumbel"), switch_alpha=arm.get("switch"),
                      max_splits=MAX_SPLITS if arm.get("trigger") else None, trigger=arm.get("trigger"))


# ── CHECKs (unit; the record equalities run as runs in the parent) ──────────
def _model(seed, arm_name=None, gumbel_scale=1.0):
    a = b16.arm_of(CFG)
    m = g16.to_local(tsr.builder_for(a)(a, seed)(), seed)
    arm = ARM.get(arm_name) or {}
    if arm.get("gumbel"):
        m = to_gumbel(arm["gumbel"], gumbel_scale)(m, seed)
    return m


def checks(p):
    rows = []
    task = b16.task_of(CFG)
    seed = 420
    x = task.make_batch(8, torch.Generator().manual_seed(5))[0][:, :-1]
    # (1) the Gumbel sampler
    gen = torch.Generator().manual_seed(1)
    n = gumbel_noise((400_000,), gen).double()
    mu, var = float(n.mean()), float(n.var())
    rows.append((f"Gumbel(0,1) sampler: mean {mu:.4f} (Euler-Mascheroni 0.5772), variance {var:.4f} (pi^2/6 = 1.6449), all finite",
                 abs(mu - 0.5772) < 0.01 and abs(var - math.pi ** 2 / 6) < 0.02 and bool(torch.isfinite(n).all())))
    # (2) the Gumbel gates: noise on training forwards only, on the stated side; the global RNG untouched
    for arm_name in ("GUMBEL_W", "GUMBEL_RW"):
        m0, m1 = _model(seed), _model(seed, arm_name)
        v = m1.embed(x)
        rs = torch.get_rng_state()
        m1.train()
        gr1, gw1 = m1.local_gates(v)
        rng_same = torch.equal(rs, torch.get_rng_state())
        gr0, gw0 = m0.local_gates(m0.embed(x))
        with torch.no_grad():
            gr_ng, gw_ng = m1.local_gates(v)                     # grad disabled: no noise
        m1.eval()
        gr_e, gw_e = m1.local_gates(v)
        m1.train()
        if arm_name == "GUMBEL_W":
            side_ok = torch.equal(gr1, gr0) and not torch.equal(gw1, gw0)
        else:
            side_ok = torch.equal(gr1, gw1) and not torch.equal(gr1, gr0)
        clean = all(torch.equal(a_, gr0) for a_ in (gr_ng, gw_ng, gr_e, gw_e))
        ok = side_ok and clean and rng_same and m1.gumbel_calls == 1 and torch.allclose(gw1.sum(-1), torch.ones(()))
        rows.append((f"{arm_name}: on a training forward the noise is on the {'write gate only (read gate = LOCAL3)' if arm_name == 'GUMBEL_W' else 'read and write gates, one draw'} "
                     f"({side_ok}); without grad and in eval mode both gates equal LOCAL3's ({clean}); the global RNG is untouched "
                     f"({rng_same}); one noisy forward counted ({m1.gumbel_calls})", ok))
        m2 = _model(seed, arm_name, gumbel_scale=0.0)
        m2.train()
        a2, b2 = m2.local_gates(m2.embed(x))
        rows.append((f"{arm_name} at noise scale 0: both gates bitwise LOCAL3's on a training forward", torch.equal(a2, gr0)
                     and torch.equal(b2, gr0)))
    # (3) the Switch loss
    g = torch.softmax(torch.randn(4, 10, 16, generator=torch.Generator().manual_seed(3)) * 2, -1)
    L, f = switch_loss(g, 1e-3)
    am = g.argmax(-1)
    hand = 1e-3 * 16 * sum(float((am == c).float().mean()) * float(g[..., c].mean()) for c in range(16))
    Lu, _ = switch_loss(torch.full((2, 3, 16), 1 / 16), 1e-3)
    gq = g.clone().requires_grad_(True)
    Lq, _ = switch_loss(gq, 1e-3)
    Lq.backward()
    grad_ok = torch.allclose(gq.grad, (1e-3 * 16 * f / (4 * 10)).expand_as(gq))
    rows.append((f"Switch loss: equals 1e-3 * k * sum f_c P_c computed by hand ({float(L):.6g} vs {hand:.6g}); a uniform gate gives "
                 f"alpha exactly {float(Lu):.6g}; d/dg = alpha * k * f_c / (B*T) (f not differentiated): {grad_ok}",
                 abs(float(L) - hand) < 1e-9 and abs(float(Lu) - 1e-3) < 1e-9 and grad_ok))
    m, mp = _model(seed), _model(seed)
    log = {}
    m = attach_switch(1e-3, log)(m, seed)
    m.train(), mp.train()
    torch.manual_seed(0)
    lo = m(x, TAU_END)
    torch.manual_seed(0)
    lp = mp(x, TAU_END)
    m.eval()
    with torch.no_grad():
        m(x, TAU_END)
    same = torch.equal(lo[0].detach(), lp[0].detach())
    rows.append((f"Switch hook: fires on training forwards only (count {log.get('n')} after one training and one eval forward); "
                 f"the logits equal the hook-free model's from the same RNG state ({same})", log.get("n") == 1 and same))
    # (4) RESET: zeroes W_g's Adam state, leaves W_g; RESET's trigger with the split as op equals S48's row for row
    a = b16.arm_of(CFG)
    mm = s48c._model(a, seed)
    opt = torch.optim.Adam(mm.parameters(), lr=b16.LR)
    s48c._one_step(mm, task, opt)
    res = reset_op(mm, opt, 0, 1, 7)
    st = opt.state[mm.W_g]
    zero = all(float(v.abs().max()) == 0 for v in st.values() if torch.is_tensor(v))
    rows.append((f"RESET's operation zeroes W_g's Adam state {res['state_zeroed']} ({zero}) and leaves W_g unchanged ({res['w_g_unchanged']})",
                 zero and res["w_g_unchanged"] and sorted(res["state_zeroed"]) == ["exp_avg", "exp_avg_sq", "step"]))
    steps = list(range(EVERY, 26400 + 1, EVERY))
    keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
            "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "noise_sd", "noise_seed",
            "row_dist_before", "row_dist_after", "state_zeroed")
    mine, m1, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_op(
        task, seed, info, oo, 26400, s36c.split_op, thr=1.01, rise=1.0), steps)
    theirs, m2, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: s48c.make_trigger(
        task, seed, info, oo, 26400, thr=1.01, rise=1.0), steps)
    eq = (len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys} for r, q in zip(mine, theirs))
          and torch.equal(m1.W_g.detach(), m2.W_g.detach()))
    rows.append((f"RESET's trigger with S36's split as the operation equals S48's make_trigger row for row on config k16 (fired at "
                 f"{[r['step'] for r in mine if r['fired']]} vs {[r['step'] for r in theirs if r['fired']]}; W_g equal after)", eq))
    rr, m3, o3, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_op(
        task, seed, info, oo, 26400, reset_op, thr=1.01, rise=1.0), steps)
    w0 = s48c._model(a, seed)
    rows.append((f"RESET's trigger fires at {[r['step'] for r in rr if r['fired']]} (cap 3, gap 4800) and leaves W_g as after one "
                 f"step (the reset copies no row: {all(r.get('w_g_unchanged') for r in rr if r['fired'])})",
                 [r["step"] for r in rr if r["fired"]] == [4800, 9600, 14400]
                 and all(r.get("w_g_unchanged") for r in rr if r["fired"])))
    # (5) key_routing: on the perfect gate the KEY map is the identity
    ca = s35c.arm("CEIL")
    mc = tsr.builder_for(ca)(ca, seed)()
    kr = key_routing(mc, task, probe_batch(task, seed))
    rows.append((f"key_routing on the perfect gate: KEY-position map {kr['ch_map_key']} (identity, one-to-one)",
                 kr["ch_map_key"] == list(range(task.S)) and kr["one_to_one_key"]))
    return [[n_, bool(v_)] for n_, v_ in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False)), True)
        m = tsr.builder_for(s35c.arm("CEIL"))(s35c.arm("CEIL"), 0)() if ARM[arm_name].get("oracle") else _model(0)
        out[arm_name] = [st, b16.eval_timing(CFG, m)]
    return out


def constants(p):
    return dict(main_sha=mt.MAIN_SHA, LR=b16.LR, LR_WARM=b16.LR_WARM, WARM=b16.WARM, EVERY=EVERY, FIRST=FIRST, REF_AT=REF_AT,
                ACC_THR=ACC_THR, RISE=RISE, GAP=GAP, MAX_SPLITS=MAX_SPLITS, S48_MAX_SPLITS=s48c.MAX_SPLITS,
                TOTAL=tsr.MAX_ITERS, T_CHECK=tsr.T_CHECK, task=[b16.task_of(CFG).S, b16.task_of(CFG).P])
