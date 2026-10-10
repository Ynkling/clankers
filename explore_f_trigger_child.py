#!/usr/bin/env python
"""
explore_f_trigger_child.py — EXPLORATORY, not a result. S81's run code (parent: explore_f_trigger, whose docstring holds
the readings). Runs INSIDE a child process on the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S76;
1 torch thread.

PATH: S76's Phase 2 exactly (explore_f_eight_child): test_window_gate's Part C/D path, test_stream_curriculum.run_sc with
ARM["D8"] (S = 8, P = 4, k = 16, conv, all 8 streams from step 1) via explore_b16_main.path("b", ...), 43200 updates,
evaluation every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95; LOCAL3 + SLOW (gate W_in, W_g,
window at 1e-3 throughout; the rest at 1e-4 for updates 1-2400, 1e-3 after; no hinge; no decoder); DELTA* channels
(explore_f_delta_fast: β = 1, L2 keys, tied write, decay 0.95); H's exact-copy split (explore_f_eight_child.copy_nonoise_op).
THE TRIGGER (make_trigger_v): explore_f_eight_child.make_trigger_op (H's, from claude/explore-H 7e9197f) line for line, with
two switches:
  source  "probe": the rule reads S36's 64-sequence probe (generator 12345), as BASE; "eval": the rule reads the run's own
          held-out evaluation at the check's update (the 2048-sequence evaluation the path has just computed and recorded
          in its curve; captured from test_binding_onset.evaluate). The probe is evaluated at every check in every arm (as
          in BASE), so both readings are recorded everywhere.
  guard   None, or g: a check is eligible only if the rule's reading is >= g (0.5: "no split while the accuracy is below
          0.5").
  At a check (every 2400 from 2400; 2400 is the reference; the last at 40800): reading r and the previous check's reading
  r_prev of the same source; eligible = step >= 4800 and r_prev exists and r < 0.95 and r - r_prev < 0.02 (and r >= g
  under a guard); then H's cap (<= 3 splits) and gap (>= 4800 updates since the last). Every row records both readings
  and both decisions (the rule's eligibility read on the probe and read on the evaluation, with the arm's guard), and
  whether they disagree.
ARMS: BASE (probe, no guard: S76's DELTA_SPLIT), GUARD (probe, 0.5), EVAL (eval, no guard), GUARD_EVAL (eval, 0.5);
  O_DELTA_C: the perfect gate (ceil8_D8) on DELTA*, one Adam at 1e-3 on every parameter (validity).
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
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import explore_f_eight_child as e8
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = "b"
TOTAL = 43200
EVERY, FIRST, REF_AT = e8.EVERY, e8.FIRST, e8.REF_AT                 # 2400, 4800, 2400
ACC_THR, RISE, GAP, MAX_SPLITS = e8.ACC_THR, e8.RISE, e8.GAP, e8.MAX_SPLITS   # 0.95, 0.02, 4800, 3
GUARD = 0.5
STAR = os.path.join(HERE, "explore_out", "F", "eight_delta_star.json")
DELTA_KW = dict(beta=1.0, normalize=True)                           # DELTA*; asserted against the committed choice
ARM = {"BASE": dict(source="probe", guard=None), "GUARD": dict(source="probe", guard=GUARD),
       "EVAL": dict(source="eval", guard=None), "GUARD_EVAL": dict(source="eval", guard=GUARD),
       "O_DELTA_C": dict(perfect=True)}
S76_SHA, S76_CPU = "2336ceff7760", "Intel(R) Xeon(R) Processor @ 2.10GHz"


def _elig(step, r, prev, thr, rise, guard):
    return (step >= FIRST and prev is not None and r < thr and r - prev < rise
            and (guard is None or r >= guard))


def make_trigger_v(task, seed, info, opt_of, last, op, evals, source="probe", guard=None, max_splits=MAX_SPLITS, gap=GAP,
                   thr=ACC_THR, rise=RISE):
    """explore_f_eight_child.make_trigger_op line for line, plus the source and the guard (see the module docstring).
    evals: the list the patched evaluate appends the held-out evaluations to ([acc, loss] per evaluation)."""
    assert source in ("probe", "eval")
    data = s36c.probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None, prev_p=None, prev_e=None)
    info["probe_id"] = id(data)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc_p, _ = tbo.evaluate(m, task, data)
        assert len(evals) == step // tbo.EVAL_EVERY, f"held-out evaluations {len(evals)} at step {step}"
        acc_e = evals[-1][0]
        acc = acc_p if source == "probe" else acc_e
        (kc, k0, km), (ac, a0, am) = s37c.select(s37c.read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        elig_p = _elig(step, acc_p, st["prev_p"], thr, rise, guard)
        elig_e = _elig(step, acc_e, st["prev_e"], thr, rise, guard)
        row.update(source=source, guard=guard, acc_probe=acc_p, acc_eval=acc_e, prev_probe=st["prev_p"], prev_eval=st["prev_e"],
                   elig_probe=elig_p, elig_eval=elig_e, disagree=elig_p != elig_e)
        if guard is not None:
            row["guarded"] = bool(eligible and acc < guard)
            eligible = eligible and acc >= guard
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
        st["prev_p"], st["prev_e"] = acc_p, acc_e
        info["checks"].append(row)
    return trig


# ── Arms ─────────────────────────────────────────────────────────────────────
def mem_conv(impl="fast"):
    if impl == "hebb":
        return (dm.converter(impl="hebb", **DELTA_KW),)
    return (ff.converter(**DELTA_KW),)


def build_rc(arm_name, p, holder, info, evals):
    task = b16.task_of(CFG)
    conv = mem_conv(p.get("impl", "fast"))
    if ARM[arm_name].get("perfect"):
        rc, _ = b16.make_rc("adam", holder, (), tau=None, convert=conv, keep=p.get("keep"), groups=False)
        return rc, contextlib.nullcontext()
    arm = ARM[arm_name]
    source = p.get("source", arm["source"])
    guard = p.get("guard", arm["guard"])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,) + conv, probe=None, keep=p.get("keep"))
    ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
    if p.get("h_trigger"):                     # H's make_trigger_op itself (S76's DELTA_SPLIT), for the CHECKs
        trig = e8.make_trigger_op(task, p["seed"], info, opt_of, p.get("total", TOTAL) - EVERY, e8.copy_nonoise_op,
                                  thr=p.get("thr", ACC_THR))
    else:
        trig = make_trigger_v(task, p["seed"], info, opt_of, p.get("total", TOTAL) - EVERY, e8.copy_nonoise_op, evals,
                              source=source, guard=guard, thr=p.get("thr", ACC_THR), rise=p.get("rise", RISE))
    return s36c.with_trigger(rc, trig), ctx


def run(p):
    arm_name = p["arm"]
    holder, info, evals = {}, dict(checks=[]), []
    rc, ctx = build_rc(arm_name, p, holder, info, evals)
    orig_eval = tbo.evaluate

    def evaluate(model, task, data):                     # the divergence guard (S76's) + the held-out readings
        acc, loss = orig_eval(model, task, data)
        if not (math.isfinite(acc) and math.isfinite(loss)):
            raise e8.Diverged(f"DIVERGED (non-finite held-out loss {loss})")
        if id(data) != info.get("probe_id"):
            assert data.shape[0] == tbo.EVAL_QUERIES // task.n_q, f"an evaluation on {data.shape[0]} sequences"
            evals.append([acc, loss])
        return acc, loss
    tbo.evaluate = evaluate
    try:
        with ctx:
            a = s33c.cfg_arm(CFG, "CEIL") if ARM[arm_name].get("perfect") else b16.arm_of(CFG)
            rec = b16.path(CFG, a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        tbo.evaluate = orig_eval
    perfect = ARM[arm_name].get("perfect", False)
    rec = b16.finish(rec, arm=arm_name, cfg=CFG, opt="adam", kind="perfect" if perfect else "local", memory="delta",
                     delta=DELTA_KW, source=None if perfect else p.get("source", ARM[arm_name]["source"]),
                     guard=None if perfect else p.get("guard", ARM[arm_name]["guard"]),
                     groups=b16.group_names(holder), checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]))
    if rec.get("ok") and not perfect:
        import explore_f_configs_b_child as cb
        rec.update(cb.classify_k(rec))
        n = len(rec["curve"])
        rec["evals_match_curve"] = n == len(evals) and all(c[1] == e[0] for c, e in zip(rec["curve"], evals))
    return rec


# ── CHECKs ───────────────────────────────────────────────────────────────────
KEYS = ("step", "acc", "prev", "map_before", "key_cs", "key_c0", "fired", "map_after", "row_dist_before", "row_dist_after",
        "key_ok", "eligible", "blocked")


def _rows_equal(a, b, t, keys=KEYS):
    ra = [c for c in a if c["step"] <= t]
    rb = [c for c in b if c["step"] <= t]
    return len(ra) == len(rb) and all({k: x.get(k) for k in keys} == {k: y.get(k) for k in keys} for x, y in zip(ra, rb))


def s76_record(seed):
    import explore_common as ec
    return ec.load_store("F/eight")["runs"][f"DELTA_SPLIT|{seed}|{S76_SHA}|{S76_CPU}"]


PARTS = ("base", "guard0", "guardnone", "evalprobe", "geprobe", "evalread", "mech")


def checks(p):
    """p["part"]: one of PARTS (the parent runs them in parallel); every disabled variant is compared with S76's record,
    which BASE reproduces (part "base"), so "equals BASE" is asserted through the record."""
    import explore_common as ec
    part = p.get("part", "mech")
    rows = []
    cpu = ec.cpu_model()
    seed, t = 1315, 7200                                   # S76's DELTA_SPLIT|1315: first split at 4800 (labelled ok)
    variants = {"base": ("BASE", dict(arm="BASE")),
                "guard0": ("GUARD with the guard at 0", dict(arm="GUARD", guard=0.0)),
                "guardnone": ("GUARD with no guard", dict(arm="GUARD", guard=None)),
                "evalprobe": ("EVAL reading the probe", dict(arm="EVAL", source="probe")),
                "geprobe": ("GUARD_EVAL reading the probe, guard 0", dict(arm="GUARD_EVAL", source="probe", guard=0.0))}
    if part in variants:
        if cpu != S76_CPU:
            return [[f"{part}: S76 record reproduction NOT APPLICABLE on {cpu}", True]]
        ref = s76_record(seed)
        first = [c["step"] for c in ref["checks"] if c["fired"]][0]
        assert first + EVERY == t
        nm, kw = variants[part]
        r = run(dict(kw, seed=seed, iters=t, total=TOTAL))
        same, want = b16.same_upto(r, ref, t)
        trig = _rows_equal(r["checks"], ref["checks"], t)
        what = ("reproduces S76's" if part == "base" else "(its condition disabled) equals BASE, i.e. reproduces S76's")
        rows.append((f"{nm} on seed {seed} {what} DELTA_SPLIT|{seed} ({S76_SHA}, this CPU) bit for bit through {t} (first "
                     f"split at {first} + one check): curve {r['curve']} vs {want}; statistics {same}; trigger rows (fired at "
                     f"{[c['step'] for c in r['checks'] if c['fired']]}) {trig}", same and trig))
    elif part == "evalread":
        if cpu != S76_CPU:
            return [[f"evalread: NOT APPLICABLE on {cpu}", True]]
        re_ = run(dict(arm="EVAL", seed=seed, iters=t, total=TOTAL))
        cur = {c[0]: c[1] for c in re_["curve"]}
        ok_e = (all(c["acc"] == cur[c["step"]] and c["acc_eval"] == cur[c["step"]] for c in re_["checks"])
                and re_["evals_match_curve"] and len(re_["checks"]) >= 3)
        rows.append((f"EVAL on seed {seed}: at every check the rule's reading is the curve's held-out accuracy at that update "
                     f"exactly ({[(c['step'], round(c['acc'], 4), round(c['acc_probe'], 4)) for c in re_['checks']]}); the "
                     f"captured evaluations equal the curve ({re_['evals_match_curve']}); fired at "
                     f"{[c['step'] for c in re_['checks'] if c['fired']]}", ok_e))
    else:
        star = json.load(open(STAR))
        rows.append((f"DELTA* = {star['arm']} (beta {star['beta']}, L2 keys {star['normalize']}) as committed; this child uses "
                     f"{DELTA_KW}", star["beta"] == DELTA_KW["beta"] and star["normalize"] == DELTA_KW["normalize"]))
        # the trigger's mechanics: a model after one Adam step, the trigger at each check with no training in between
        import explore_window_d8_child as s48c
        a = b16.arm_of(CFG)
        task = b16.task_of(CFG)

        def mech(source, guard, fake):
            m = mem_conv()[0](g16.to_local(tsr.builder_for(a)(a, 2020)(), 2020), 2020)
            opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
            s48c._one_step(m, task, opt)
            info, evals = dict(checks=[]), []
            trig = make_trigger_v(task, 2020, info, lambda: opt, TOTAL - EVERY, e8.copy_nonoise_op, evals, source=source,
                                  guard=guard, thr=1.1, rise=1.0)
            for step in range(1200, 19201, 1200):
                evals.append([fake(step), 1.0])
                trig(m, step)
            return info["checks"]
        c_base = mech("probe", None, lambda s: 0.0)
        c_guard = mech("probe", GUARD, lambda s: 0.0)
        c_ge = mech("eval", GUARD, lambda s: 0.3 if s < 7200 else 0.6)
        fired = lambda cs: [c["step"] for c in cs if c["fired"]]
        rows.append((f"trigger mechanics (untrained model, threshold 1.1, rise 1): BASE fires at {fired(c_base)} (expect 4800, "
                     f"9600, 14400); GUARD (probe {c_guard[1]['acc_probe']:.3f} < 0.5) fires at {fired(c_guard)} (expect none; "
                     f"guarded {sum(c.get('guarded', False) for c in c_guard)}); GUARD_EVAL on fake held-out readings 0.3 then 0.6 "
                     f"from 7200 fires at {fired(c_ge)} (expect 7200, 12000, 16800)",
                     fired(c_base) == [4800, 9600, 14400] and fired(c_guard) == [] and fired(c_ge) == [7200, 12000, 16800]
                     and all(c["acc_probe"] < 0.5 for c in c_guard)))
        # the perfect gate on DELTA* at S = 8
        ac = s33c.cfg_arm(CFG, "CEIL")
        x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
        m0 = tsr.builder_for(ac)(ac, 2020)()
        md = mem_conv()[0](tsr.builder_for(ac)(ac, 2020)(), 2020)
        m0.eval(), md.eval()
        with torch.no_grad():
            g0, g = m0(x)[2], md(x)[2]
        rows.append((f"the perfect gate on DELTA* (k {md.n_ch}): one-hot ({bool(((g == 0) | (g == 1)).all())}), equal to the "
                     f"Hebbian model's ({torch.equal(g, g0)})", torch.equal(g, g0) and bool(((g == 0) | (g == 1)).all())))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    st = b16.step_timing(lambda: run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9)), True)
    return [st, 25.0]
