#!/usr/bin/env python
"""
explore_f_window_delta_scale_child.py — EXPLORATORY, not a result. S54 (b) and (c)'s run code (parent:
explore_f_window_delta_scale). Runs INSIDE a child process on the main line's module tree at explore_main9c.MAIN_SHA
(9c5939e), as batches 16-17's S44 / S48; recipes from explore_b16_main, the trigger from explore_window_d8_child (both
imported read-only). 1 torch thread.

CONFIGURATIONS (the recorded runs' own paths, so a seed fixes the initial parameters and batches):
  "k16"  S=4, P=4, k=16, conv (test_stream_recipe.run_attempt with A4k16; batch 16's LOCAL3_SLOW16_A path), 28800 updates
  "b"    S=8, P=4, k=16, conv, all 8 streams from step 1 (test_stream_curriculum.run_sc with D8; batch 17's W_SPLIT path),
         43200 updates
ARMS
  {B,C}_HEBB    batch 17's W_SPLIT recipe: LOCAL3 (explore_b16_gates.to_local) + SLOW (Adam: gate W_in, W_g, window at 1e-3
                throughout; every other trainable parameter 1e-4 for updates 1-2400, 1e-3 after) + S37's KEYMASS plateau
                trigger (explore_window_d8_child.make_trigger: checks every 2400 from 4800 with 2400 the reference, on S36's
                64-sequence probe; fire if accuracy < 0.95 and it rose < 0.02 since the previous check; at most 3 splits,
                >= 4800 apart; the last check at the run's length - 2400; S24's split of W_g's row c* onto c0, W_g's Adam
                state zeroed); no hinge; the Hebbian memory. S38's decoder after every evaluation.
  {B,C}_DELTA   the same, the model converted by explore_delta_mem.to_delta(beta="learned") after to_local (L2 keys, tied
                write, decay 0.95); attn.w_b, attn.b_b in the 'rest' group.
  {B,C}_ORACLE_{HEBB,DELTA}  the perfect gate (ceiling4k16 / ceil8_D8, k = 16), the perfect gate's own recipe (one Adam
                at 1e-3 on every parameter, no slow phase, no trigger), Hebbian or delta (learned beta).
"""

import contextlib
import time

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_k16_child as s35c
import explore_split_child as s36c
import explore_window_d8_child as wd8c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_delta_mem as dm
import test_stream_recipe as tsr

MAX_SPLITS = 3
DELTA_KW = dict(beta="learned", normalize=True)
ARM = {
    "B_HEBB": dict(cfg="k16", delta=False), "B_DELTA": dict(cfg="k16", delta=True),
    "C_HEBB": dict(cfg="b", delta=False), "C_DELTA": dict(cfg="b", delta=True),
    "B_ORACLE_HEBB": dict(cfg="k16", delta=False, perfect=True), "B_ORACLE_DELTA": dict(cfg="k16", delta=True, perfect=True),
    "C_ORACLE_HEBB": dict(cfg="b", delta=False, perfect=True), "C_ORACLE_DELTA": dict(cfg="b", delta=True, perfect=True),
}
REC = {"k16": ("window_scale", "LOCAL3_SLOW16_A", 242, "01767493e840"), "b": ("window_d8", "W_SPLIT", 260, "375b30d487df")}
REC_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def arm_model(arm):
    if arm.get("perfect"):
        return s35c.arm("CEIL") if arm["cfg"] == "k16" else s33c.cfg_arm("b", "CEIL")
    return b16.arm_of(arm["cfg"])


def convert_of(arm, impl="parallel"):
    conv = () if arm.get("perfect") else (g16.to_local,)
    if arm["delta"]:
        conv = conv + (dm.converter(impl=impl, **DELTA_KW),)
    return conv


def build_rc(arm_name, p, holder, out, info):
    arm = ARM[arm_name]
    task = b16.task_of(arm["cfg"])
    conv = convert_of(arm, p.get("impl", "parallel"))
    box = p.get("box")
    on_build = (lambda m: box.__setitem__("m", m)) if box is not None else None
    if arm.get("perfect"):
        rc, _ = b16.make_rc("adam", holder, (), tau=None, convert=conv, on_build=on_build, keep=p.get("keep"), groups=False)
        return rc, contextlib.nullcontext()
    probe = wd8c.every_probe(task, out) if p.get("probe", True) else None
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=conv, probe=probe, on_build=on_build,
                        keep=p.get("keep"))
    ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
    rc = s36c.with_trigger(rc, wd8c.make_trigger(task, p["seed"], info, opt_of, p.get("total", p["iters"]) - wd8c.EVERY,
                                                 max_splits=MAX_SPLITS, thr=p.get("thr", wd8c.ACC_THR)))
    return rc, ctx


@torch.no_grad()
def beta_by_role(m, task, seed):
    from test_multilayer_binding import probe_batch
    pinp, _, roles = probe_batch(task, seed)
    out = []

    def hook(mod, args, kwargs, o):
        b = mod.beta_of(kwargs["V"][:, 0])
        out.append({r: float(b[mask].mean()) for r, mask in roles.items()})
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    return out


def run(p):
    arm = ARM[p["arm"]]
    holder, out, info, box = {}, {}, dict(checks=[]), {}
    p = dict(p, box=box)
    rc, ctx = build_rc(p["arm"], p, holder, out, info)
    with ctx:
        rec = b16.path(arm["cfg"], arm_model(arm), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    extra = {}
    m = box.get("m")
    if rec.get("ok") and arm["delta"] and m is not None and p.get("probe", True):
        extra["beta_by_role"] = beta_by_role(m, b16.task_of(arm["cfg"]), p["seed"])
    return b16.finish(rec, arm=p["arm"], cfg=arm["cfg"], opt="adam", kind="perfect" if arm.get("perfect") else "local",
                      memory="delta" if arm["delta"] else "hebb", delta_cfg=getattr(m, "delta_cfg", None), decode=out,
                      groups=b16.group_names(holder), checks=info["checks"],
                      splits=sum(1 for c in info["checks"] if c["fired"]), max_splits=None if arm.get("perfect") else MAX_SPLITS,
                      **extra)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import explore_common as ec
    rows = []
    cpu = ec.cpu_model()
    # (1) the recorded runs, the new code path disabled (impl "hebb") and the Hebbian arm itself
    for cfg, arm_h, arm_d in (("b", "C_HEBB", "C_DELTA"), ("k16", "B_HEBB", "B_DELTA")):
        store, rarm, seed, sha = REC[cfg]
        key = f"{rarm}|{seed}|{sha}|{REC_CPU}"
        ref = ec.load_store(store)["runs"][key]
        t = 2400
        total = 43200 if cfg == "b" else 28800
        if cpu != REC_CPU:
            rows.append((f"{cfg}: recorded {key} NOT APPLICABLE on {cpu}", True))
            continue
        for arm_name, impl in ((arm_h, "parallel"), (arm_d, "hebb")):
            t0 = time.time()
            r = run(dict(arm=arm_name, seed=seed, iters=t, total=total, impl=impl))
            same, want = b16.same_upto(r, ref, t)
            dec = all(r["decode"].get(s) == ref["decode"].get(s) for s in ("1200", "2400") if s in ref.get("decode", {}))
            rows.append((f"{cfg}: {arm_name} (memory impl {impl}) on seed {seed} reproduces {key} through {t} bit for bit "
                         f"(curve {r['curve']} vs {want}; statistics equal {same}; decoder at 1200/2400 equal {dec}; "
                         f"{time.time() - t0:.0f} s)", same and dec))
    # (2) groups: the memory's parameters are in the 'rest' group; the gate group is LOCAL3's
    for cfg in ("k16", "b"):
        a = b16.arm_of(cfg)
        m = tsr.builder_for(a)(a, 310)()
        for f in convert_of(ARM["C_DELTA" if cfg == "b" else "B_DELTA"]):
            m = f(m, 310)
        h = {}
        gs = b16.adam_groups(h, b16.GATE_LOCAL)(m)
        names = {g["tag"]: g["names"] for g in gs}
        flat = [n for g in gs for n in g["names"]]
        train = [n for n, q in m.named_parameters() if q.requires_grad]
        ok = (names["gate"] == list(b16.GATE_LOCAL) and all(n in names["rest"] for n in dm.delta_params(m))
              and sorted(flat) == sorted(train) and len(flat) == len(set(flat)) and isinstance(m.attn, dm.DeltaMemory)
              and m.attn.beta_mode == "learned" and type(m).__name__ == "DeltaLocalBDH")
        rows.append((f"{cfg}: DELTA model {type(m).__name__}, memory {type(m.attn).__name__}; groups gate {names['gate']}, rest "
                     f"includes {dm.delta_params(m)}; every trainable parameter once", ok))
    # (3) the perfect gates on delta keep the routing one-hot and convert cleanly
    for cfg in ("k16", "b"):
        arm = dict(cfg=cfg, delta=True, perfect=True)
        a = arm_model(arm)
        m0 = tsr.builder_for(a)(a, 310)()
        m = convert_of(arm)[0](tsr.builder_for(a)(a, 310)(), 310)
        task = b16.task_of(cfg)
        x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
        m.eval(), m0.eval()
        with torch.no_grad():
            g, g0 = m(x)[2], m0(x)[2]
        rows.append((f"{cfg}: the perfect gate (k {m.n_ch}) on the delta memory: read gate equal to the Hebbian model's and "
                     f"one-hot ({torch.equal(g, g0)}, {bool(((g == 0) | (g == 1)).all())})",
                     torch.equal(g, g0) and bool(((g == 0) | (g == 1)).all()) and isinstance(m.attn, dm.DeltaMemory)))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    arm = ARM[p["arm"]]
    st = b16.step_timing(lambda: run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False)), True)
    a = arm_model(arm)
    m = tsr.builder_for(a)(a, 0)()
    for f in convert_of(arm):
        m = f(m, 0)
    return [st, b16.eval_timing(arm["cfg"], m) + (0.0 if arm.get("perfect") else b16.decode_timing(arm["cfg"], m))]


def constants(p):
    return dict(main_sha=mt.MAIN_SHA, LR=b16.LR, LR_WARM=b16.LR_WARM, WARM=b16.WARM, EVERY=wd8c.EVERY, FIRST=wd8c.FIRST,
                ACC_THR=wd8c.ACC_THR, RISE=wd8c.RISE, GAP=wd8c.GAP, MAX_SPLITS=MAX_SPLITS)
