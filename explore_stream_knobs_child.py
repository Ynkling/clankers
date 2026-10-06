#!/usr/bin/env python
"""
explore_stream_knobs_child.py — EXPLORATORY, not a result. S46's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_stream_knobs. Recipes: explore_b16_main.

Configuration: X's HINGE_D8 (test_stream_curriculum.run_sc with ARM["D8"]: S=8, P=4, k=16, conv, all 8 streams from step
1, 28800 steps), Adam, test_slow_start's HINGE recipe composition (explore_b16_main.make_rc + adam_groups; CHECK: at the
default lrs it reproduces X's HINGE_D8|260 through 1200), with one knob changed per arm:
  LR3E4      every lr x 0.3: gate W_in/W_h/W_g 3e-4 throughout; every other parameter 3e-5 for updates 1-2400, 3e-4 after.
  LR3E3      every lr x 3: gate 3e-3 throughout; the rest 3e-4 for 1-2400, 3e-3 after.
  WH_SLOW    W_h in its own group at 1e-4 (the gate's lr / 10) throughout; W_in, W_g 1e-3 throughout; the rest 1e-4 for
             1-2400, 1e-3 after.
  DECAY98    the memory's decay 0.98 instead of 0.95 (GatedAttention.decay, set on the built model: the decay mask and
             nothing else).
  PERFECT98  the perfect gate (test_stream_recipe.ARM["ceiling8k16"], cur False: S33's config b CEIL arm) at decay 0.98,
             with the perfect gate's own recipe (one Adam at 1e-3 for every parameter, no slow phase, no hinge; main's
             ceiling arms and S25's selection); a validity check of the decay.
Measured: S38's decoder (CTX, KEY, VAL) at 1200 and 2400 (learned gates); rho(W_h) and sigma_max(W_h) at update 0 and
every 400 updates (an optimizer step post-hook counting the run's Adam steps).
"""

import torch
import torch.optim.optimizer as topt

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_b16_gates as g16
import explore_b16_main as b16
import test_slow_start as tss
import test_stream_recipe as tsr
from test_instrument_v2 import decay_mask

CFG = "b"
AT = (1200, 2400)
RHO_EVERY = 400
DECAY = 0.98
ARM = {"LR3E4": dict(lr=3e-4), "LR3E3": dict(lr=3e-3), "WH_SLOW": dict(wh=True), "DECAY98": dict(decay=DECAY),
       "PERFECT98": dict(perfect=True, decay=DECAY), "HINGE_D8_DEFAULT": dict()}


def set_decay(d):
    def f(m, seed):
        m.attn.decay = d
        return m
    return f


def arm_model(arm):
    return s33c.cfg_arm(CFG, "CEIL") if arm.get("perfect") else b16.arm_of(CFG)


def build_rc(arm_name, p, holder, out, rho, box):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    conv = (set_decay(arm["decay"]),) if arm.get("decay") else ()

    def on_build(m):
        box["m"] = m
        if hasattr(m, "W_h"):
            rho["0"] = dict(rho=g16.rho(m.W_h), sigma=g16.sigma_max(m.W_h))
    if arm.get("perfect"):
        return b16.make_rc("adam", holder, (), tau=None, convert=conv, on_build=on_build, keep=p.get("keep"), groups=False)[0]
    lr = arm.get("lr", b16.LR)
    probe = b16.decoder_probe(task, out, AT)[0] if p.get("probe", True) else None
    if arm.get("wh"):
        return b16.make_rc("adam", holder, ("W_in", "W_g"), tau=b16.TAU, fixed=((("W_h",), b16.LR / 10),), convert=conv,
                           probe=probe, on_build=on_build, keep=p.get("keep"))[0]
    return b16.make_rc("adam", holder, b16.GATE_REC, tau=b16.TAU, lr=lr, warm_lr=lr / 10, convert=conv, probe=probe,
                       on_build=on_build, keep=p.get("keep"))[0]


def run(p):
    holder, out, rho, box = {}, {}, {}, {}
    rc = build_rc(p["arm"], p, holder, out, rho, box)
    n = [0]

    def post(opt, args, kwargs):
        n[0] += 1
        m = box.get("m")
        if m is not None and hasattr(m, "W_h") and n[0] % RHO_EVERY == 0:
            rho[str(n[0])] = dict(rho=g16.rho(m.W_h), sigma=g16.sigma_max(m.W_h))
    h = topt.register_optimizer_step_post_hook(post)
    try:
        rec = b16.path(CFG, arm_model(ARM[p["arm"]]), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        h.remove()
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", decode=out, rho=rho, groups=b16.group_names(holder),
                      decay=box["m"].attn.decay if box.get("m") is not None else None)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    b = tsr.builder_for(a)
    m5, m8 = b(a, 260)(), set_decay(DECAY)(b(a, 260)(), 260)
    sd5, sd8 = m5.state_dict(), m8.state_dict()
    same = sd5.keys() == sd8.keys() and all(torch.equal(sd5[k], sd8[k]) for k in sd5)
    attrs = [k for k in vars(m8.attn) if not k.startswith("_") and vars(m8.attn)[k] != vars(m5.attn).get(k)]
    x = task.make_batch(4, torch.Generator().manual_seed(7))[0][:, :-1]
    T = x.shape[1]
    m5.eval(), m8.eval()
    with torch.no_grad():
        m5.attn.record, m8.attn.record = [], []
        l5, l8 = m5(x)[0], m8(x)[0]
        s5, s8 = m5.attn.record[0], m8.attn.record[0]
        m5.attn.record = m8.attn.record = None
        ratio = (decay_mask(T, DECAY) / decay_mask(T, 0.95).clamp(min=1e-30)).tril(-1)
        rel = float(((s8 - s5 * ratio).abs() / (s5 * ratio).abs().clamp(min=1e-12)).max())
        m8.attn.decay = 0.95
        back = torch.equal(m8(x)[0], l5)
        m8.attn.decay = DECAY
    rows.append((f"decay 0.98 enters the decay mask and nothing else: parameters and buffers equal ({same}); attention attributes "
                 f"that differ {attrs}; layer-1 scores = the 0.95 model's x (0.98/0.95)^(t-s) (max rel. diff {rel:.1e} <= 1e-5); "
                 f"logits differ ({not torch.equal(l5, l8)}) and equal the 0.95 model's bit for bit with the decay set back ({back})",
                 same and attrs == ["decay"] and rel <= 1e-5 and back and not torch.equal(l5, l8)))
    c = s33c.cfg_arm(CFG, "CEIL")
    bc = tsr.builder_for(c)
    p5, p8 = bc(c, 260)(), set_decay(DECAY)(bc(c, 260)(), 260)
    p5.eval(), p8.eval()
    with torch.no_grad():
        g5, g8 = p5(x)[2], p8(x)[2]
    rows.append((f"the perfect gate's routing is unchanged at decay 0.98 (read gate bitwise equal: {torch.equal(g5, g8)}; one-hot "
                 f"over {g5.shape[-1]} channels: {bool(((g5 == 0) | (g5 == 1)).all())})", torch.equal(g5, g8)))
    # the generic Adam recipe at the default lrs reproduces X's HINGE_D8|260 through 1200
    t = 1200
    xs = mt.recorded("slow_start", mt.RESULTS_SHA)["HINGE_D8|260"]
    r = run(dict(arm="HINGE_D8_DEFAULT", seed=260, iters=t, probe=False))
    ok, want = b16.same_upto(r, xs, t)
    rows.append((f"explore_b16_main's Adam recipe at the default lrs (gate W_in/W_h/W_g 1e-3; the rest 1e-4 -> 1e-3; the hinge) "
                 f"reproduces X's HINGE_D8|260 through {t} (curve {r['curve']} vs {want}; statistics equal)", ok))
    # lrs per group through the real run path (forward and evaluation stubbed), 2401 updates
    W = b16.WARM
    for arm_name, want1, want2 in (("LR3E4", [3e-4, 3e-4 / 10], [3e-4, 3e-4]), ("LR3E3", [3e-3, 3e-3 / 10], [3e-3, 3e-3]),
                                   ("WH_SLOW", [1e-3, 1e-4, 1e-4], [1e-3, 1e-4, 1e-3])):
        h, rec, lg, ids, keep = b16.stub_lrs(CFG, a, lambda hd, kp: build_rc(arm_name, dict(keep=kp, probe=False), hd, {}, {}, {}))
        names = [g["names"] for g in h["groups"]]
        ok = (b16.cover(ids, keep["model"]) and len(lg) == W + 1 and lg[0] == want1 and lg[W - 1] == want1 and lg[W] == want2
              and rec.get("ok") and (arm_name != "WH_SLOW" or (names[0] == ["W_in", "W_g"] and names[1] == ["W_h"])))
        rows.append((f"{arm_name}: groups {[(g['tag'], g['names'] if len(g['names']) < 4 else len(g['names'])) for g in h['groups']]} "
                     f"cover every parameter once; lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}", ok))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False)), True)
        arm = ARM[arm_name]
        a = arm_model(arm)
        m = tsr.builder_for(a)(a, 0)()
        out[arm_name] = [st, b16.eval_timing(CFG, m), 0.0 if arm.get("perfect") else b16.decode_timing(CFG, m)]
    return out


def recorded(p):
    runs = mt.recorded("slow_start", mt.RESULTS_SHA)
    return {k: dict(outcome=tsr.outcome(runs[k]), transition=runs[k].get("transition"), acc=runs[k].get("acc"))
            for k in p["keys"] if k in runs}
