#!/usr/bin/env python
"""
explore_window_d8_child.py — EXPLORATORY, not a result. S48's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_window_d8. Recipes: explore_b16_main (batch 16's,
imported read-only); batch 16's S44 child is not imported (its LOCAL3_SLOW_D8_A recipe is make_rc("adam", GATE_LOCAL,
convert=(to_local,)), composed the same way here; CHECK: with the trigger's threshold at 0, W_SPLIT|260 equals the
LOCAL3_SLOW_D8_A|260 record of batch 16 through 6000).

CONFIGURATION: S44(b)'s: test_stream_curriculum.run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams from step 1),
LOCAL3's gate (explore_b16_gates.to_local: the width-3 window from a generator seeded seed + 31337, W_h frozen, unused),
evaluation every 1200, no hinge. Every arm is LOCAL3_SLOW_D8_A (Adam: gate W_in, W_g, window at 1e-3 throughout; every
other trainable parameter 1e-4 for updates 1-2400, 1e-3 after) or LOCAL3_SLOW_D8_M (S25's groups at Muon lr 0.005) with
one change:
  W_SPLIT      + the KEYMASS plateau trigger (make_trigger below: S37's explore_split_target_child.make_trigger line for
               line with the cap as a parameter): checks at 2400 (reference), 4800, 7200, ... through iters - 2400 on S36's
               64-sequence probe; fire if the probe accuracy is < 0.95 and it rose < 0.02 since the previous check; at
               most 3 splits per run (S37: 2), none within 4800 updates of the previous one (S37's gap); S24's split of
               W_g's row c* (largest mean read-gate mass at the probe's key positions) onto c0 (smallest), W_g's optimizer
               state zeroed. onset_run's Adam is built through S36's capturing factory (the same constructor).
  W_D98        the memory's decay 0.98 (S46's set_decay: GatedAttention.decay on the built model; the mask only).
  W_SPLIT_D98  both.
  W_LONG       43200 updates (the parent's iters), nothing else changed (the run path has no schedule tied to the length:
               its first 28800 updates are LOCAL3_SLOW_D8_A's run; the report checks this on every seed).
  W_NOSLOW     one rate: every trainable parameter at 1e-3 throughout (the same two groups, warm lr = lr).
  W_WIDTH4     the window of width 4 (gate at t sees tokens t-3 .. t): uniform +-1/sqrt(4) from the same generator
               (seed + 31337), shape (D, 4) — batch 1's rule (the conv's default bound 1/sqrt(width)) at width 4.
  W_SPLIT_M    LOCAL3_SLOW_D8_M + the trigger (Muon: W_g's momentum buffer zeroed; S37's SPLITK_M wiring).
  W_M_REF      LOCAL3_SLOW_D8_M fresh on this CPU (a reference for W_SPLIT_M when the batch's CPU is not the CPU of
               batch 16's LOCAL3_SLOW_D8_M records).
  W_BASE       LOCAL3_SLOW_D8_A (CHECK only).
MEASURED: S38's decoder (explore_b16_gates.decode: the gate state h and its input u at CTX, KEY and VAL positions) after
every evaluation ("end" = the last); routing_k's map at every evaluation (the run path's statistics); the trigger's rows
(every check: probe accuracy, KEYMASS c* / c0 and their key masses, the labelled map before, eligible / fired / blocked
and why, the map after a split).
"""

import contextlib
import math

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_muon_recipe as s25
import explore_stream_knobs_child as s46c
import explore_b16_gates as g16
import explore_b16_main as b16
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch, CausalConv, D, MultiBDH

CFG = "b"
EVERY, FIRST, REF_AT = s37c.EVERY, s37c.FIRST, s37c.REF_AT          # 2400, 4800, 2400
ACC_THR, RISE, GAP = s37c.ACC_THR, s37c.RISE, s37c.GAP              # 0.95, 0.02, 4800
MAX_SPLITS = 3                                                      # the user (S37: 2)
DECAY = s46c.DECAY                                                  # 0.98
ARM = {
    "W_SPLIT": dict(opt="adam", trigger=True),
    "W_D98": dict(opt="adam", decay=DECAY),
    "W_SPLIT_D98": dict(opt="adam", trigger=True, decay=DECAY),
    "W_LONG": dict(opt="adam"),
    "W_NOSLOW": dict(opt="adam", noslow=True),
    "W_WIDTH4": dict(opt="adam", width=4),
    "W_SPLIT_M": dict(opt="muon", trigger=True),
    "W_M_REF": dict(opt="muon"),
    "W_BASE": dict(opt="adam"),
}


# ── The window of any width ──────────────────────────────────────────────────
def window_init_w(seed, width):
    """Batch 1's window rule at a width: uniform +-1/sqrt(width) from a generator seeded seed + 31337."""
    g = torch.Generator().manual_seed(int(seed) + g16.GCONV_OFFSET)
    return (torch.rand(D, width, generator=g) * 2 - 1) * (1.0 / math.sqrt(width))


def to_local_w(width):
    """explore_b16_gates.to_local at width 3; the same conversion with a width-`width` window otherwise."""
    if width == g16.WIDTH:
        return g16.to_local

    def conv(m, seed):
        assert type(m) is MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
        assert not m.gate_to_readout and not m.gate_ln and m.gate_noise == 0.0
        m.__class__ = g16.LocalBDH
        m.local = True
        m.gate_conv = CausalConv(D, width)
        with torch.no_grad():
            m.gate_conv.conv_w.copy_(window_init_w(seed, width))
        m.W_h.requires_grad_(False)
        return m
    return conv


# ── The trigger (S37's, with the cap as a parameter) ─────────────────────────
def make_trigger(task, seed, info, opt_of, last, max_splits=MAX_SPLITS, gap=GAP, thr=ACC_THR, rise=RISE):
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
            res = s36c.split_op(m, opt_of(), kc, k0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in s37c.select(s37c.read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


# ── Runs ─────────────────────────────────────────────────────────────────────
def every_probe(task, out):
    """S38's decoder after every evaluation (eval mode, no gradient, its own data: training is untouched)."""
    data = g16.probe_data(task)

    def probe(m, step):
        out[str(step)] = g16.decode(m, task, data)
    return probe


def convert_of(arm):
    return (to_local_w(arm.get("width", g16.WIDTH)),) + ((s46c.set_decay(arm["decay"]),) if arm.get("decay") else ())


def build_rc(arm_name, p, holder, out, info):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    probe = every_probe(task, out) if p.get("probe", True) else None
    rc, _ = b16.make_rc(arm["opt"], holder, b16.GATE_LOCAL, tau=None, mlr=p.get("mlr"),
                        warm_lr=b16.LR if arm.get("noslow") else b16.LR_WARM, convert=convert_of(arm), probe=probe,
                        keep=p.get("keep"))
    if arm.get("trigger"):
        if arm["opt"] == "muon":
            ctx, opt_of = s25.muon_in_onset_run(holder), (lambda: holder["opts"][0].muon)
        else:
            ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        rc = s36c.with_trigger(rc, make_trigger(task, p["seed"], info, opt_of, p["iters"] - EVERY,
                                                max_splits=p.get("max_splits", MAX_SPLITS), thr=p.get("thr", ACC_THR)))
    elif arm["opt"] == "muon":
        ctx = s25.muon_in_onset_run(holder)
    else:
        ctx = contextlib.nullcontext()
    return rc, ctx


def run(p):
    """p: arm, seed, iters, mlr (Muon arms), eval_every / probe (timing), thr (CHECK), keep (CHECK)."""
    arm = ARM[p["arm"]]
    holder, out, info = {}, {}, dict(checks=[])
    rc, ctx = build_rc(p["arm"], p, holder, out, info)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt=arm["opt"], kind="local", width=arm.get("width", g16.WIDTH),
                      decay=arm.get("decay", 0.95), noslow=bool(arm.get("noslow")),
                      muon_lr=p.get("mlr") if arm["opt"] == "muon" else None, decode=out, groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                      max_splits=MAX_SPLITS if arm.get("trigger") else None)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _model(a, seed, width=3, decay=None):
    m = to_local_w(width)(tsr.builder_for(a)(a, seed)(), seed)
    return s46c.set_decay(decay)(m, seed) if decay else m


def _one_step(m, task, opt):
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    inp, tgt = x[:, :-1], x[:, 1:]
    ql, qt, _ = task.select(tbo.logits_of(m, inp), tgt, None)
    loss = torch.nn.functional.cross_entropy(ql, qt)
    opt.zero_grad()
    loss.backward()
    opt.step()


def _trigger_rows(a, task, seed, maker, steps, muon=False, mlr=None):
    """A LOCAL3 model after one optimizer step (so W_g has optimizer state), the trigger called at each step in turn
    with no training between (thr above 1, rise 1: every check from 4800 is eligible)."""
    m = _model(a, seed)
    if muon:
        groups = b16.muon_groups({}, mlr, b16.GATE_LOCAL)(m)
        opt = s25.MuonAdam(groups, b16.LR)
        opt_of = lambda: opt.muon                                                     # noqa: E731
    else:
        opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
        opt_of = lambda: opt                                                          # noqa: E731
    _one_step(m, task, opt)
    info = dict(checks=[])
    trig = maker(info, opt_of)
    state_before = {k: v.clone() for k, v in opt_of().state[m.W_g].items() if torch.is_tensor(v)}
    for t in steps:
        trig(m, t)
    return info["checks"], m, opt_of(), state_before


def checks(p):
    mlr = p["mlr"]
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 260
    # (1) the conversions
    rows.append(("to_local_w(3) is explore_b16_gates.to_local (batch 16's LOCAL3 conversion)", to_local_w(3) is g16.to_local))
    m0 = tsr.builder_for(a)(a, seed)()
    m4 = _model(a, seed, width=4)
    p0, p4 = dict(m0.named_parameters()), dict(m4.named_parameters())
    same = all(torch.equal(p4[n], p0[n]) for n in p0)
    extra = [n for n in p4 if n not in p0]
    w4 = m4.gate_conv.conv_w.detach()
    rows.append((f"width 4: the conversion keeps every parameter of the run path's model bitwise ({len(p0)}), adds only {extra} "
                 f"of shape {tuple(w4.shape)} = window_init_w(seed, 4) (|w| max {float(w4.abs().max()):.4f} <= 0.5), W_h frozen "
                 f"({not m4.W_h.requires_grad}); window_init_w(seed, 3) equals batch 1's window "
                 f"({torch.equal(window_init_w(seed, 3), g16.window_init(seed))})",
                 same and extra == ["gate_conv.conv_w"] and tuple(w4.shape) == (D, 4) and torch.equal(w4, window_init_w(seed, 4))
                 and float(w4.abs().max()) <= 0.5 and not m4.W_h.requires_grad
                 and torch.equal(window_init_w(seed, 3), g16.window_init(seed))))
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    t = 30
    for width in (3, 4):
        m = _model(a, seed, width=width)
        m.eval()
        with torch.no_grad():
            g0 = m(x)[2]
            xa = x.clone(); xa[:, t - width] = (xa[:, t - width] + 1) % task.vocab
            xb = x.clone(); xb[:, t - width + 1] = (xb[:, t - width + 1] + 1) % task.vocab
            blind = torch.equal(m(xa)[2][:, t], g0[:, t])
            sees = not torch.equal(m(xb)[2][:, t], g0[:, t])
        rows.append((f"width {width}: the gate at t = {t} ignores token t-{width} ({blind}) and sees token t-{width - 1} ({sees})",
                     blind and sees))
    m3, m4 = _model(a, seed, width=3), _model(a, seed, width=4)
    with torch.no_grad():
        w = torch.zeros_like(m3.gate_conv.conv_w); w[:, 0] = 1.0
        m3.gate_conv.conv_w.copy_(w)
        w = torch.zeros_like(m4.gate_conv.conv_w); w[:, 0] = 1.0
        m4.gate_conv.conv_w.copy_(w)
    m3.eval(), m4.eval()
    with torch.no_grad():
        d = float((m3(x)[2] - m4(x)[2]).abs().max())
    rows.append((f"with the windows at [1, 0, 0] and [1, 0, 0, 0] the width-3 and width-4 gates are equal (max |diff| {d:.2e})",
                 d == 0.0))
    # (2) the decay on the LOCAL3 model: the mask and nothing else
    m5, m8 = _model(a, seed), _model(a, seed, decay=DECAY)
    sd5, sd8 = m5.state_dict(), m8.state_dict()
    same = sd5.keys() == sd8.keys() and all(torch.equal(sd5[k], sd8[k]) for k in sd5)
    attrs = [k for k in vars(m8.attn) if not k.startswith("_") and vars(m8.attn)[k] != vars(m5.attn).get(k)]
    m5.eval(), m8.eval()
    with torch.no_grad():
        o5, o8 = m5(x), m8(x)
        m8.attn.decay = 0.95
        back = torch.equal(m8(x)[0], o5[0])
        m8.attn.decay = DECAY
    rows.append((f"decay 0.98 on the LOCAL3 model: parameters and buffers equal ({same}); attention attributes that differ {attrs}; "
                 f"gate equal ({torch.equal(o5[2], o8[2])}); logits differ ({not torch.equal(o5[0], o8[0])}) and equal the 0.95 "
                 f"model's bit for bit with the decay set back ({back}); set_decay is S46's",
                 same and attrs == ["decay"] and torch.equal(o5[2], o8[2]) and not torch.equal(o5[0], o8[0]) and back))
    # (3) groups and lrs through the real run path (forward and evaluation stubbed), 2401 updates
    W = b16.WARM
    for arm_name, want1, want2 in (("W_NOSLOW", [b16.LR, b16.LR], [b16.LR, b16.LR]),
                                   ("W_WIDTH4", [b16.LR, b16.LR_WARM], [b16.LR, b16.LR]),
                                   ("W_D98", [b16.LR, b16.LR_WARM], [b16.LR, b16.LR])):
        h, rec, lg, ids, keep = b16.stub_lrs(CFG, a, lambda hd, kp: build_rc(arm_name, dict(keep=kp, probe=False, seed=3,
                                                                                            iters=W + 1), hd, {}, {})[0])
        mm = keep["model"]
        names = [g["names"] for g in h["groups"]]
        ok = (b16.cover(ids, mm) and names[0] == list(b16.GATE_LOCAL) and len(names) == 2 and len(lg) == W + 1
              and lg[0] == want1 and lg[W - 1] == want1 and lg[W] == want2 and rec.get("ok") and not mm.W_h.requires_grad
              and mm.gate_conv.conv_w.shape[1] == ARM[arm_name].get("width", 3)
              and mm.attn.decay == ARM[arm_name].get("decay", 0.95))
        rows.append((f"{arm_name}: groups {[(g['tag'], len(g['names'])) for g in h['groups']]} cover every trainable parameter "
                     f"once, gate {names[0]}; lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}; window width "
                     f"{mm.gate_conv.conv_w.shape[1]}, decay {mm.attn.decay}", ok))
    # (4) the trigger: cap 2 equals S37's row for row; cap 3 fires three times, 4800 apart, then is capped
    steps = list(range(EVERY, 26400 + 1, EVERY))
    s37keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0",
               "key_ok", "all_ok", "fired", "blocked", "eligible", "map_after", "key_masses_after", "noise_sd", "noise_seed",
               "row_dist_before", "row_dist_after", "state_zeroed")
    mine, mm, om, _ = _trigger_rows(a, task, seed, lambda info, oo: make_trigger(task, seed, info, oo, 26400, max_splits=2,
                                                                                 thr=1.01, rise=1.0), steps)
    theirs, ms, os_, _ = _trigger_rows(a, task, seed, lambda info, oo: s37c.make_trigger(task, seed, info, oo, 26400,
                                                                                       thr=1.01, rise=1.0), steps)
    eq = (len(mine) == len(theirs) and all({k: r.get(k) for k in s37keys} == {k: q.get(k) for k in s37keys}
                                           for r, q in zip(mine, theirs))
          and torch.equal(mm.W_g.detach(), ms.W_g.detach()))
    rows.append((f"the trigger at cap 2 equals S37's make_trigger row for row over checks {steps[0]}..{steps[-1]} (fired at "
                 f"{[r['step'] for r in mine if r['fired']]} vs {[r['step'] for r in theirs if r['fired']]}; W_g equal after)", eq))
    r3, m3, o3, sb = _trigger_rows(a, task, seed, lambda info, oo: make_trigger(task, seed, info, oo, 26400, thr=1.01, rise=1.0),
                                   steps)
    fired = [r["step"] for r in r3 if r["fired"]]
    gap = [r["step"] for r in r3 if r.get("blocked_by") == "gap"]
    cap = [r["step"] for r in r3 if r.get("blocked_by") == "cap"]
    st = o3.state[m3.W_g]
    zeroed = all(float(v.abs().max()) == 0.0 for k, v in st.items() if torch.is_tensor(v) and k != "step") and \
        any(float(v.abs().max()) > 0 for k, v in sb.items() if k != "step")
    rows.append((f"the trigger at cap 3 (Adam): fired at {fired}, blocked by the 4800 gap at {gap}, by the cap at {cap} (from the third split on the cap is named); each split "
                 f"copies W_g's row c* onto c0 with noise (row distances after {[round(r['row_dist_after'], 4) for r in r3 if r['fired']]}"
                 f" > 0) and zeroes W_g's Adam state (exp_avg, exp_avg_sq zero after: {zeroed}); targets labelled "
                 f"{[(r['key_cs'], r['key_c0'], r['key_ok']) for r in r3 if r['fired']]}",
                 fired == [4800, 9600, 14400] and gap == [7200, 12000] and cap == [16800, 19200, 21600, 24000, 26400] and zeroed
                 and all(r["row_dist_after"] > 0 for r in r3 if r["fired"])))
    if p.get("muon"):
        rm, mmu, omu, sbm = _trigger_rows(a, task, seed, lambda info, oo: make_trigger(task, seed, info, oo, 26400, thr=1.01,
                                                                                      rise=1.0), [2400, 4800], muon=True, mlr=mlr)
        st = omu.state[mmu.W_g]
        zeroed = all(float(v.abs().max()) == 0.0 for v in st.values() if torch.is_tensor(v)) and \
            any(float(v.abs().max()) > 0 for v in sbm.values())
        rows.append((f"the trigger under Muon: W_g is in the Muon optimizer ({any(q is mmu.W_g for g in omu.param_groups for q in g['params'])}); "
                     f"fired at {[r['step'] for r in rm if r['fired']]}; W_g's Muon state {sorted(sbm)} nonzero before and zero "
                     f"after ({zeroed})", [r["step"] for r in rm if r["fired"]] == [4800] and zeroed
                     and any(q is mmu.W_g for g in omu.param_groups for q in g["params"])))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"], eval_every=10 ** 9,
                                                probe=False)), arm["opt"] == "adam")
        a = b16.arm_of(CFG)
        m = _model(a, 0, width=arm.get("width", 3), decay=arm.get("decay"))
        out[arm_name] = [st, b16.eval_timing(CFG, m), b16.decode_timing(CFG, m)]
    return out


def constants(p):
    return dict(main_sha=mt.MAIN_SHA, LR=b16.LR, LR_WARM=b16.LR_WARM, WARM=b16.WARM, EVERY=EVERY, FIRST=FIRST, REF_AT=REF_AT,
                ACC_THR=ACC_THR, RISE=RISE, GAP=GAP, MAX_SPLITS=MAX_SPLITS, S37_MAX_SPLITS=s37c.MAX_SPLITS, DECAY=DECAY,
                WIDTH=g16.WIDTH, TOTAL=tsr.__dict__.get("TOTAL"))
