#!/usr/bin/env python
"""
explore_window_scale_child.py — EXPLORATORY, not a result. S44's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_window_scale. Recipes: explore_b16_main.

ARMS (conv on, 28800 steps; the recorded runs' paths, so a seed fixes the initial parameters and batches):
  (a) S=4, P=4, k=16 (run_attempt with A4k16, the path of X's HINGE4k16 and S35's MUON_HINGE16):
      LOCAL3_SLOW16_A   LOCAL3 (explore_b16_gates.to_local) + SLOW, no hinge, Adam: gate W_in, W_g, window at 1e-3
                        throughout; every other trainable parameter 1e-4 for updates 1-2400, 1e-3 after.
      LOCAL3_SLOW16_M   LOCAL3 + S25's groups at Muon lr 0.005 (W_in, W_g on Muon at 0.005 throughout; the window on Adam
                        at 1e-3 throughout; other 2-D / 3-D weights on Muon at 0.0005 for 1-2400; embedding, stack conv
                        and 1-D on Adam at 1e-4 for 1-2400; full after), no hinge.
      MUON_HINGE16_REF  S35's MUON_HINGE16 (explore_muon_k16_child.make_rc "HINGE"), fresh on this CPU.
  (b) S=8, P=4, k=16, all 8 streams from step 1 (run_sc with D8, the path of X's HINGE_D8 and S33's D8_HINGE):
      LOCAL3_SLOW_D8_A, LOCAL3_SLOW_D8_M   as (a) on this configuration.
      LOCAL3_SLOW_D8_A_SPLIT  LOCAL3_SLOW_D8_A + S37's KEYMASS plateau trigger (explore_split_target_child.make_trigger:
                        checks at 2400 (reference), 4800, 7200, ... 26400 on S36's 64-sequence probe; fire if accuracy
                        < 0.95 and it rose < 0.02 since the previous check; at most 2 splits, >= 4800 apart; S24's split
                        of W_g's row c* onto c0 and W_g's Adam state zeroed), onset_run's Adam built through S36's
                        capturing factory (the same constructor).
      D8_HINGE_M_REF    S33's D8_HINGE (explore_muon_scale_child.make_rc "HINGE"), fresh on this CPU.
DECODER: explore_b16_gates.decode (S38's decoder at CTX, KEY, VAL; h and u) at 1200, 2400, 4800 and 9600, after the
evaluation's statistics, every arm.
"""

import contextlib

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_k16_child as s35c
import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_muon_recipe as s25
import explore_b16_gates as g16
import explore_b16_main as b16
import test_slow_start as tss
import test_stream_recipe as tsr

AT = (1200, 2400, 4800, 9600)
ARM = {
    "LOCAL3_SLOW16_A": dict(cfg="k16", kind="local", opt="adam"),
    "LOCAL3_SLOW16_M": dict(cfg="k16", kind="local", opt="muon"),
    "MUON_HINGE16_REF": dict(cfg="k16", kind="ref", opt="muon"),
    "LOCAL3_SLOW_D8_A": dict(cfg="b", kind="local", opt="adam"),
    "LOCAL3_SLOW_D8_M": dict(cfg="b", kind="local", opt="muon"),
    "LOCAL3_SLOW_D8_A_SPLIT": dict(cfg="b", kind="local", opt="adam", trigger=True),
    "D8_HINGE_M_REF": dict(cfg="b", kind="ref", opt="muon"),
}


def to_local(m, seed):
    return g16.to_local(m, seed)


def build_rc(arm_name, p, holder, out, info, log):
    arm = ARM[arm_name]
    cfg = arm["cfg"]
    task = b16.task_of(cfg)
    probe = b16.decoder_probe(task, out, AT)[0] if p.get("probe", True) else None
    if arm["kind"] == "local":
        rc, _ = b16.make_rc(arm["opt"], holder, b16.GATE_LOCAL, tau=None, mlr=p.get("mlr"), convert=(to_local,),
                            probe=probe, keep=p.get("keep"))
        if arm.get("trigger"):
            ctx = s36c.adam_capture(holder)
            trig = s37c.make_trigger(task, p["seed"], info, lambda: holder["opts"][0], s37c.TOTAL - s37c.EVERY,
                                     thr=p.get("thr", s37c.ACC_THR))
            rc = s36c.with_trigger(rc, trig)
        elif arm["opt"] == "muon":
            ctx = s25.muon_in_onset_run(holder)
        else:
            ctx = contextlib.nullcontext()
    else:
        if cfg == "k16":
            rc, _ = s35c.make_rc("HINGE", p["mlr"], holder, log, keep=p.get("keep"))
        else:
            rc, _ = s33c.make_rc("HINGE", p["mlr"], holder, keep=p.get("keep"))
        if probe is not None:
            rc = b16.with_probe(rc, probe)
        ctx = s25.muon_in_onset_run(holder)
    return rc, ctx


def run(p):
    """p: arm, seed, iters, mlr (Muon arms), eval_every / probe (timing), thr (CHECK), keep (CHECK)."""
    arm = ARM[p["arm"]]
    holder, out, info, log = {}, {}, dict(checks=[]), []
    rc, ctx = build_rc(p["arm"], p, holder, out, info, log)
    with ctx:
        rec = b16.path(arm["cfg"], b16.arm_of(arm["cfg"]), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    return b16.finish(rec, arm=p["arm"], cfg=arm["cfg"], opt=arm["opt"], kind=arm["kind"],
                      muon_lr=p.get("mlr") if arm["opt"] == "muon" else None, decode=out, groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                      fired_at=list(log) if arm["kind"] == "ref" and arm["cfg"] == "k16" else None)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    mlr = p["mlr"]
    rows = []
    for cfg in ("k16", "b"):
        a = b16.arm_of(cfg)
        b = tsr.builder_for(a)
        m0 = b(a, 260)()
        m1 = to_local(b(a, 260)(), 260)
        p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
        same = all(torch.equal(p1[n], p0[n]) for n in p0)
        extra = [n for n in p1 if n not in p0]
        rows.append((f"config {cfg}: LOCAL3's conversion keeps every parameter of the run path's model bitwise ({len(p0)}), "
                     f"adds only {extra}, equal to batch 1's window for the seed ({torch.equal(m1.gate_conv.conv_w.detach(), g16.window_init(260))}); "
                     f"W_h frozen ({not m1.W_h.requires_grad}); class {type(m1).__name__}",
                     same and extra == ["gate_conv.conv_w"] and torch.equal(m1.gate_conv.conv_w.detach(), g16.window_init(260))
                     and not m1.W_h.requires_grad))
    cfg = "b"
    a = b16.arm_of(cfg)
    task = b16.task_of(cfg)
    m = to_local(tsr.builder_for(a)(a, 260)(), 260)
    m.eval()
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    t = 30
    with torch.no_grad():
        g0 = m(x)[2]
        x3 = x.clone(); x3[:, t - 3] = (x3[:, t - 3] + 1) % task.vocab
        x2 = x.clone(); x2[:, t - 2] = (x2[:, t - 2] + 1) % task.vocab
        blind = torch.equal(m(x3)[2][:, t], g0[:, t])
        sees = not torch.equal(m(x2)[2][:, t], g0[:, t])
    rows.append((f"config b: the LOCAL3 gate at t = {t} ignores token t-3 ({blind}) and sees token t-2 ({sees})", blind and sees))
    ref = tsr.builder_for(a)(a, 260)()
    with torch.no_grad():
        ref.W_h.zero_()
        w = torch.zeros_like(m.gate_conv.conv_w)
        w[:, 0] = 1.0
        m.gate_conv.conv_w.copy_(w)
    ref.eval()
    with torch.no_grad():
        d = float((ref(x)[2] - m(x)[2]).abs().max())
    rows.append((f"config b: with the window at [1, 0, 0] and W_h = 0 the LOCAL3 gate equals arm A's gate with W_h = 0 "
                 f"(max |diff| {d:.2e} <= 1e-6)", d <= 1e-6))
    # groups and lrs through the real run paths (forward and evaluation stubbed), 2401 updates
    W = b16.WARM
    for cfg in ("k16", "b"):
        a = b16.arm_of(cfg)
        h, rec, lg, ids, keep = b16.stub_lrs(cfg, a, lambda hd, kp: b16.make_rc(
            "adam", hd, b16.GATE_LOCAL, convert=(to_local,), keep=kp)[0])
        mm = keep["model"]
        names = [g["names"] for g in h["groups"]]
        wh_same = torch.equal(mm.W_h.detach(), tsr.builder_for(a)(a, 3)().W_h.detach())
        ok = (b16.cover(ids, mm) and names[0] == list(b16.GATE_LOCAL) and len(names) == 2 and len(lg) == W + 1
              and lg[0] == [b16.LR, b16.LR_WARM] and lg[W - 1] == [b16.LR, b16.LR_WARM] and lg[W] == [b16.LR, b16.LR]
              and wh_same and rec.get("ok"))
        rows.append((f"config {cfg} LOCAL3 Adam: groups {[(g['tag'], len(g['names'])) for g in h['groups']]} cover every trainable "
                     f"parameter once, gate {names[0]}; lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}; W_h in no group "
                     f"and unchanged after {len(lg)} updates ({wh_same})", ok))
        h, rec, lg, ids, keep = b16.stub_lrs(cfg, a, lambda hd, kp: b16.make_rc(
            "muon", hd, b16.GATE_LOCAL, mlr=mlr, convert=(to_local,), keep=kp)[0], muon=True)
        mm = keep["model"]
        spec = [(g["tag"], g["kind"]) for g in h["groups"]]
        names = [g["names"] for g in h["groups"]]
        w1 = [mlr, b16.LR, b16.SLOW * mlr, b16.SLOW * b16.LR]
        w2 = [mlr, b16.LR, mlr, b16.LR]
        ok = (b16.cover(ids, mm) and spec == [("gate", "muon"), ("gate", "adam"), ("rest", "muon"), ("rest", "adam")]
              and names[0] == ["W_in", "W_g"] and names[1] == ["gate_conv.conv_w"] and "embed.weight" in names[3]
              and any("short_conv" in n for n in names[3]) and len(lg) == W + 1 and lg[0] == w1 and lg[W - 1] == w1
              and lg[W] == w2 and rec.get("ok"))
        rows.append((f"config {cfg} LOCAL3 Muon (S25's groups): {list(zip(spec, names))}; lrs at update 1 {lg[0]}, {W} "
                     f"{lg[W - 1]}, {W + 1} {lg[W]}", ok))
    m = tsr.builder_for(a)(a, 260)()
    mine = [g["names"] for g in b16.muon_groups({}, mlr, b16.GATE_REC)(m)]
    s33 = [g["names"] for g in s33c.make_groups({}, mlr, True)(m)]
    rows.append((f"explore_b16_main.muon_groups with the recurrent gate's names equals S33's groups ({mine == s33})", mine == s33))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        cfg = arm["cfg"]
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"], eval_every=10 ** 9,
                                                probe=False)), arm["opt"] == "adam")
        a = b16.arm_of(cfg)
        m = tsr.builder_for(a)(a, 0)()
        if arm["kind"] == "local":
            m = to_local(m, 0)
        out[arm_name] = [st, b16.eval_timing(cfg, m), b16.decode_timing(cfg, m)]
    return out


def recorded(p):
    """Outcomes of X's recorded runs (test_stream_recipe.outcome) for the pairing: HINGE4k16 and HINGE_D8."""
    runs = mt.recorded("slow_start", mt.RESULTS_SHA)
    out = {}
    for k in p["keys"]:
        r = runs.get(k)
        if r is None:
            continue
        out[k] = dict(outcome=tsr.outcome(r), transition=r.get("transition"), acc=r.get("acc"), end_map=r["end"].get("ch_map"),
                      maps=s33c.maps_of(r), stopped_at=r.get("stopped_at"))
    return out
