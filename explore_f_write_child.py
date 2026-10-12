#!/usr/bin/env python
"""
explore_f_write_child.py — EXPLORATORY, not a result. S88a's run code (parent: explore_f_write, whose docstring holds the
readings). Runs INSIDE a child process on the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S76 and S81;
1 torch thread.

PATH: S76's Phase 2 exactly (explore_f_eight_child): test_window_gate's Part C/D path, test_stream_curriculum.run_sc with
ARM["D8"] (S = 8, P = 4, k = 16, conv, all 8 streams from step 1) via explore_b16_main.path("b", ...), 43200 updates,
evaluation every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95; LOCAL3 + SLOW (Adam: the gate
group W_in, W_g, gate_conv.conv_w at 1e-3 throughout; every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3
after; no hinge); S76's KEYMASS trigger with H's exact-copy split (explore_f_eight_child.make_trigger_op, copy_nonoise_op).
MEMORY: the delta rule with L2 keys, tied write, decay 0.95 (explore_f_delta_fast), with the write strength
  DELTA_FIXED         β = 1 (DELTA*, S76's).
  DELTA_LEARNED       β_t = 2·sigmoid(w·v_t + b) per token (Beta2Memory), w = 0 (d = 32) and b = 0 at init, so β = 1.0
                      exactly at init with a live gradient (sigmoid'(0) = 1/4); v_t the layer's write vector V (its input,
                      as the delta memory's learned β everywhere in Session F); w, b shared by the 3 layers; w and b in
                      the GATE group (1e-3 throughout).
  DELTA_LEARNED_SLOW  the same, w and b in the slow ("rest") group: 1e-4 for updates 1-2400, 1e-3 after.
  O_DELTA, O_DELTA_LEARNED   the perfect gate (ceil8_D8), one Adam at 1e-3 on every parameter, β = 1 / β learned.
β TRAJECTORY (learned arms, after every evaluation, on the run's probe batch, test_multilayer_binding.probe_batch): per
layer the median of β over all positions, over CTX / KEY / VAL positions, per stream at KEY positions (8 medians), η² of β
by stream at KEY and at VAL positions, the fraction of positions with β > 1, min and max.
DIVERGENCE GUARD (S76's): a non-finite held-out loss or accuracy at an evaluation stops the run as DIVERGED (ok = False).
"""

import contextlib
import json
import math
import os

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import explore_f_eight_child as e8
import test_binding_onset as tbo
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = "b"
TOTAL = 43200
STAR = os.path.join(HERE, "explore_out", "F", "eight_delta_star.json")
BETA_NAMES = ("attn.w_b", "attn.b_b")
ARM = {"DELTA_FIXED": dict(beta="fixed"), "DELTA_LEARNED": dict(beta="learned", group="gate"),
       "DELTA_LEARNED_SLOW": dict(beta="learned", group="rest"),
       "O_DELTA": dict(beta="fixed", perfect=True), "O_DELTA_LEARNED": dict(beta="learned", perfect=True)}
S76_SHA, S76_CPU = "2336ceff7760", "Intel(R) Xeon(R) Processor @ 2.10GHz"


class Beta2Memory(ff.FastDeltaMemory):
    """The delta memory with β_t = 2·sigmoid(w_b·v_t + b_b) (explore_delta_mem's learned β, doubled: 1 at init)."""

    def beta_of(self, v):
        return 2.0 * torch.sigmoid(v @ self.w_b + self.b_b)


def conv_of(arm_name, freeze=False):
    if ARM[arm_name]["beta"] == "fixed":
        return ff.converter(beta=1.0, normalize=True)

    def conv(m, seed):
        m = ff.to_delta_fast(m, seed=seed, beta="learned", normalize=True)
        m.attn.__class__ = Beta2Memory
        m.delta_cfg = dict(m.delta_cfg, beta="2*sigmoid(w.v+b), w=0, b=0 at init", frozen=bool(freeze))
        if freeze:
            for n, q in m.named_parameters():
                if n in BETA_NAMES:
                    q.requires_grad_(False)
        return m
    return conv


def gate_names(arm_name):
    return b16.GATE_LOCAL + (BETA_NAMES if ARM[arm_name].get("group") == "gate" else ())


# ── The β trajectory ─────────────────────────────────────────────────────────
def eta2(x, lab):
    """η² of x by the labels (1-D tensors): between-group sum of squares over the total."""
    x = x.double()
    tot = ((x - x.mean()) ** 2).sum()
    if tot <= 0:
        return 0.0
    btw = sum(int((lab == g).sum()) * (x[lab == g].mean() - x.mean()) ** 2 for g in lab.unique())
    return float(btw / tot)


@torch.no_grad()
def beta_stats(m, task, probe):
    pinp, plab, roles = probe
    rows = []

    def hook(mod, args, kwargs, o):
        rows.append(mod.beta_of(kwargs["V"][:, 0]))
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    out = []
    for b in rows:
        key, val = roles["key"], roles["val"]
        r = dict(median=float(b.median()), min=float(b.min()), max=float(b.max()), frac_gt1=float((b > 1).float().mean()),
                 by_role={rn: float(b[rm].median()) for rn, rm in roles.items()},
                 key_by_stream=[float(b[key & (plab == s)].median()) for s in range(task.S)],
                 eta_key_by_stream=eta2(b[key], plab[key]), eta_val_by_stream=eta2(b[val], plab[val]))
        out.append(r)
    return out


def beta_probe(task, seed, out):
    probe = probe_batch(task, seed)

    def p(m, step):
        out[str(step)] = beta_stats(m, task, probe)
    return p


# ── A run ────────────────────────────────────────────────────────────────────
def build_rc(arm_name, p, holder, info, out):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    conv = (conv_of(arm_name, p.get("freeze", False)),)
    learned = arm["beta"] == "learned" and not p.get("freeze", False)
    probe = beta_probe(task, p["seed"], out) if (arm["beta"] == "learned" and p.get("beta_probe", True)) else None
    if arm.get("perfect"):
        rc, _ = b16.make_rc("adam", holder, (), tau=None, convert=conv, probe=probe, keep=p.get("keep"), groups=False)
        return rc, contextlib.nullcontext()
    rc, _ = b16.make_rc("adam", holder, gate_names(arm_name) if learned else b16.GATE_LOCAL, tau=None,
                        convert=(g16.to_local,) + conv, probe=probe, keep=p.get("keep"))
    ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
    trig = e8.make_trigger_op(task, p["seed"], info, opt_of, p.get("total", TOTAL) - e8.EVERY, e8.copy_nonoise_op)
    return s36c.with_trigger(rc, trig), ctx


def run(p):
    arm_name = p["arm"]
    arm = ARM[arm_name]
    holder, info, out = {}, dict(checks=[]), {}
    rc, ctx = build_rc(arm_name, p, holder, info, out)
    orig_eval = tbo.evaluate

    def evaluate(model, task, data):                     # the divergence guard (S76's)
        acc, loss = orig_eval(model, task, data)
        if not (math.isfinite(acc) and math.isfinite(loss)):
            raise e8.Diverged(f"DIVERGED (non-finite held-out loss {loss})")
        return acc, loss
    tbo.evaluate = evaluate
    try:
        with ctx:
            a = s33c.cfg_arm(CFG, "CEIL") if arm.get("perfect") else b16.arm_of(CFG)
            rec = b16.path(CFG, a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        tbo.evaluate = orig_eval
    rec = b16.finish(rec, arm=arm_name, cfg=CFG, opt="adam", kind="perfect" if arm.get("perfect") else "local",
                     memory="delta", beta=arm["beta"], beta_group=arm.get("group"), frozen=bool(p.get("freeze", False)),
                     groups=b16.group_names(holder), checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]), beta_traj=out or None)
    if rec.get("ok") and not arm.get("perfect"):
        import explore_f_configs_b_child as cb
        rec.update(cb.classify_k(rec))
    return rec


# ── CHECKs ───────────────────────────────────────────────────────────────────
PARTS = ("s76", "frozen", "frozen_slow", "mech")
KEYS = ("step", "acc", "prev", "map_before", "key_cs", "key_c0", "fired", "map_after", "row_dist_before", "row_dist_after",
        "key_ok", "eligible", "blocked")


def _rows_equal(a, b, t):
    ra = [c for c in a if c["step"] <= t]
    rb = [c for c in b if c["step"] <= t]
    return len(ra) == len(rb) and all({k: x.get(k) for k in KEYS} == {k: y.get(k) for k in KEYS} for x, y in zip(ra, rb))


def s76_record(seed):
    import explore_common as ec
    return ec.load_store("F/eight")["runs"][f"DELTA_SPLIT|{seed}|{S76_SHA}|{S76_CPU}"]


def checks(p):
    """p["part"]: one of PARTS (the parent runs them in parallel). "Equals DELTA_FIXED" is asserted through S76's record,
    which DELTA_FIXED reproduces (part "s76")."""
    import explore_common as ec
    part = p.get("part", "mech")
    rows = []
    cpu = ec.cpu_model()
    seed, t = 1315, 7200                                   # S76's DELTA_SPLIT|1315: first split at 4800
    variants = {"s76": ("DELTA_FIXED", dict(arm="DELTA_FIXED")),
                "frozen": ("DELTA_LEARNED with w and b frozen at their initial values", dict(arm="DELTA_LEARNED", freeze=True)),
                "frozen_slow": ("DELTA_LEARNED_SLOW with w and b frozen", dict(arm="DELTA_LEARNED_SLOW", freeze=True))}
    if part in variants:
        ref = s76_record(seed)
        nm, kw = variants[part]
        r = run(dict(kw, seed=seed, iters=t, total=TOTAL))
        same, want = b16.same_upto(r, ref, t)
        trig = _rows_equal(r["checks"], ref["checks"], t)
        extra, ok_x = "", True
        if part != "s76":
            bt = r.get("beta_traj") or {}
            allone = all(L["min"] == 1.0 and L["max"] == 1.0 for v in bt.values() for L in v)
            extra = f"; β = 1.0 exactly at every probe position and evaluation ({allone}, {len(bt)} evaluations)"
            ok_x = allone and len(bt) >= 5
        what = "reproduces S76's" if part == "s76" else "equals DELTA_FIXED, i.e. reproduces S76's"
        rows.append((f"{nm} on seed {seed}, on this CPU ({cpu}), {what} DELTA_SPLIT|{seed} ({S76_SHA}, recorded on {S76_CPU}) "
                     f"bit for bit through {t} (its first "
                     f"split at 4800 + one check): curve {r['curve']} vs {want}; statistics {same}; trigger rows (fired at "
                     f"{[c['step'] for c in r['checks'] if c['fired']]}) {trig}{extra}", same and trig and ok_x))
        return [[n, bool(v)] for n, v in rows]
    star = json.load(open(STAR))
    rows.append((f"DELTA* = {star['arm']} (beta {star['beta']}, L2 keys {star['normalize']}) as committed (DELTA_FIXED uses "
                 f"beta 1, L2 keys)", star["beta"] == 1.0 and star["normalize"]))
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
    # the learned memory: β = 1 exactly at init; the same parameters as DELTA_FIXED plus w, b; the same logits
    mf = conv_of("DELTA_FIXED")(g16.to_local(tsr.builder_for(a)(a, 2400)(), 2400), 2400)
    ml = conv_of("DELTA_LEARNED")(g16.to_local(tsr.builder_for(a)(a, 2400)(), 2400), 2400)
    pf, pl = dict(mf.named_parameters()), dict(ml.named_parameters())
    extra = sorted(set(pl) - set(pf))
    same_p = all(torch.equal(pl[n], pf[n]) for n in pf)
    mf.eval(), ml.eval()
    with torch.no_grad():
        lf, ll = mf(x)[0], ml(x)[0]
    bs = beta_stats(ml, task, probe_batch(task, 2400))
    rows.append((f"DELTA_LEARNED's model = DELTA_FIXED's + {extra} (zeros), every other parameter equal ({same_p}); memory "
                 f"{type(ml.attn).__name__}; β at init in [{min(L['min'] for L in bs)}, {max(L['max'] for L in bs)}] (exactly 1); "
                 f"logits equal bit for bit ({torch.equal(lf, ll)})",
                 extra == sorted(BETA_NAMES) and same_p and isinstance(ml.attn, Beta2Memory) and torch.equal(lf, ll)
                 and all(L["min"] == 1.0 and L["max"] == 1.0 for L in bs)))
    # groups: w, b in the gate group (LEARNED) or the rest group (LEARNED_SLOW); lrs
    for arm_name, want_tag, lr0 in (("DELTA_LEARNED", "gate", b16.LR), ("DELTA_LEARNED_SLOW", "rest", b16.LR_WARM)):
        m = conv_of(arm_name)(g16.to_local(tsr.builder_for(a)(a, 2400)(), 2400), 2400)
        h = {}
        gs = b16.adam_groups(h, gate_names(arm_name))(m)
        where = {n: g["tag"] for g in gs for n in g["names"]}
        lrs = {g["tag"]: (g["lr"], g["lr_after"]) for g in gs}
        flat = [n for g in gs for n in g["names"]]
        train = [n for n, q in m.named_parameters() if q.requires_grad]
        ok = (all(where.get(n) == want_tag for n in BETA_NAMES) and lrs[want_tag][0] == lr0 and lrs[want_tag][1] == b16.LR
              and sorted(flat) == sorted(train) and len(flat) == len(set(flat))
              and [n for n in where if where[n] == "gate"][:3] == list(b16.GATE_LOCAL))
        rows.append((f"{arm_name}: w, b in the {want_tag} group (lr {lrs[want_tag][0]} for updates 1-2400, then "
                     f"{lrs[want_tag][1]}); every trainable parameter in exactly one group", ok))
    # the learned β has a live gradient: one Adam step moves w and b
    import explore_window_d8_child as s48c
    m = conv_of("DELTA_LEARNED")(g16.to_local(tsr.builder_for(a)(a, 2400)(), 2400), 2400)
    opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
    w0, b0 = m.attn.w_b.detach().clone(), m.attn.b_b.detach().clone()
    s48c._one_step(m, task, opt)
    gw_, gb_ = float(m.attn.w_b.grad.norm()), float(m.attn.b_b.grad.abs())
    rows.append((f"the learned β trains: after one Adam step |grad w| {gw_:.2e}, |grad b| {gb_:.2e}, w and b moved "
                 f"({not torch.equal(m.attn.w_b, w0)}, {not torch.equal(m.attn.b_b, b0)})",
                 gw_ > 0 and gb_ > 0 and not torch.equal(m.attn.w_b, w0) and not torch.equal(m.attn.b_b, b0)))
    # the fast kernel with a per-token β (explore_f_delta_fast's own checks)
    for nm, v in ff.fast_checks():
        rows.append((f"explore_f_delta_fast: {nm}", v))
    # the perfect gate on the learned memory: one-hot, logits equal to the fixed memory's at init
    ac = s33c.cfg_arm(CFG, "CEIL")
    m0 = conv_of("O_DELTA")(tsr.builder_for(ac)(ac, 2400)(), 2400)
    m1 = conv_of("O_DELTA_LEARNED")(tsr.builder_for(ac)(ac, 2400)(), 2400)
    m0.eval(), m1.eval()
    with torch.no_grad():
        o0, o1 = m0(x), m1(x)
    rows.append((f"the perfect gate (k {m1.n_ch}) on both memories: one-hot ({bool(((o1[2] == 0) | (o1[2] == 1)).all())}); logits "
                 f"equal at init ({torch.equal(o0[0], o1[0])})", bool(((o1[2] == 0) | (o1[2] == 1)).all())
                 and torch.equal(o0[0], o1[0])))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    st = b16.step_timing(lambda: run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9)), True)
    return [st, 25.0]
