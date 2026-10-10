#!/usr/bin/env python
"""
explore_f_configs_b_child.py — EXPLORATORY, not a result. S80 (b)'s run code (parent: explore_f_configs_b; the readings
are in explore_f_configs). Runs INSIDE a child process on the main line's module tree at explore_main9c.MAIN_SHA
(9c5939e), as S54(b) and S76; 1 torch thread. Also the classifier (v1 / v2) for S80's report and its CHECK, here because
explore_f_fail_class_v2 imports test_stream_channels, which only the main line's tree has.

PATH: test_window_gate's Part B path = test_stream_recipe.run_attempt with ARM["A4k16"] (S = 4, P = 4, k = 16, conv) via
explore_b16_main.path("k16", ...), 28800 updates, evaluation every 1200 on 2048 held-out sequences, early stop after 3
evaluations >= 0.95.
  HEBB_B, DELTA_B   WIN3_SLOW16 = LOCAL3 (explore_b16_gates.to_local) + SLOW (explore_b16_main.make_rc: gate W_in, W_g,
                    window at 1e-3 throughout; the rest at 1e-4 for updates 1-2400, 1e-3 after; no hinge; no decoder) +
                    H's exact-copy split with the KEYMASS plateau trigger (explore_f_eight_child.make_trigger_op and
                    copy_nonoise_op, S76's Phase 2, copied there from claude/explore-H at 7e9197f): checks every 2400 from
                    4800 (2400 the reference) through 26400; DELTA_B's memory is DELTA* (explore_f_delta_fast, β = 1, L2
                    keys, tied write, decay 0.95).
  O_DELTA_B, O_HEBB_B   the perfect gate (test_stream_recipe's ceiling4k16), one Adam at 1e-3 on every parameter.
DIVERGENCE GUARD (S76's): a non-finite held-out loss or accuracy at an evaluation stops the run as DIVERGED (ok = False).
"""

import contextlib
import json
import math
import os
import subprocess

import torch

import explore_main9c as mt
import explore_muon_k16_child as s35c
import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import explore_f_eight_child as e8
import test_binding_onset as tbo
import test_stream_recipe as tsr

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = "k16"
TOTAL = 28800
STAR = os.path.join(HERE, "explore_out", "F", "eight_delta_star.json")
DELTA_KW = dict(beta=1.0, normalize=True)                           # DELTA*; asserted against the committed choice
ARM = {"HEBB_B": dict(delta=False), "DELTA_B": dict(delta=True),
       "O_DELTA_B": dict(delta=True, perfect=True), "O_HEBB_B": dict(delta=False, perfect=True)}
REC = ("window_scale", "LOCAL3_SLOW16_A", 242, "01767493e840")      # batch 16, S44(a), Xeon @ 2.10GHz
REC_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"
FC_SHA = "ba8901a947854f78557447fcaf04c6d1220300f0"


def mem_conv(arm_name, impl="fast"):
    if not ARM[arm_name]["delta"]:
        return ()
    if impl == "hebb":
        return (dm.converter(impl="hebb", **DELTA_KW),)
    return (ff.converter(**DELTA_KW),)


def build_rc(arm_name, p, holder, info):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    conv = mem_conv(arm_name, p.get("impl", "fast"))
    if arm.get("perfect"):
        rc, _ = b16.make_rc("adam", holder, (), tau=None, convert=conv, keep=p.get("keep"), groups=False)
        return rc, contextlib.nullcontext()
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,) + conv, probe=None, keep=p.get("keep"))
    if p.get("trigger", True):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        trig = e8.make_trigger_op(task, p["seed"], info, opt_of, p.get("total", TOTAL) - e8.EVERY, e8.copy_nonoise_op,
                                  thr=p.get("thr", e8.ACC_THR))
        return s36c.with_trigger(rc, trig), ctx
    return rc, contextlib.nullcontext()


def run(p):
    arm_name = p["arm"]
    arm = ARM[arm_name]
    holder, info = {}, dict(checks=[])
    rc, ctx = build_rc(arm_name, p, holder, info)
    orig_eval = tbo.evaluate

    def evaluate(model, task, data):                     # the divergence guard (S76's)
        acc, loss = orig_eval(model, task, data)
        if not (math.isfinite(acc) and math.isfinite(loss)):
            raise e8.Diverged(f"DIVERGED (non-finite held-out loss {loss})")
        return acc, loss
    tbo.evaluate = evaluate
    try:
        with ctx:
            a = s35c.arm("CEIL") if arm.get("perfect") else b16.arm_of(CFG)
            rec = b16.path(CFG, a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        tbo.evaluate = orig_eval
    rec = b16.finish(rec, arm=arm_name, cfg=CFG, opt="adam", kind="perfect" if arm.get("perfect") else "local",
                     memory="delta" if arm["delta"] else "hebb", delta=DELTA_KW if arm["delta"] else None,
                     groups=b16.group_names(holder), checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]))
    if rec.get("ok") and not arm.get("perfect"):
        rec.update(classify_k(rec))
    return rec


# ── The classifier (v1 = the recorded functions; v2 = explore_f_fail_class_v2, ba8901a's) ──────────────────────────
def classify_k(r):
    import explore_f_fail_class_v2 as v2
    from test_scale_axes import fail_class_k
    e = r["end"]
    return dict(outcome_v2=v2.outcome_v2(r, tsr.routed), fail_k_v1=fail_class_k(r), fail_k_v2=v2.fail_class_k_v2(r),
                etak_key_by_key=e.get("etak_key_by_key"))


def classify(p):
    """p: {"two": {key: end dict}, "k": {key: record}} -> v1 / v2 per key (two-stream: test_router_layout.fail_class
    and fail_class_v2; multichannel: test_scale_axes.fail_class_k and fail_class_k_v2)."""
    import explore_f_fail_class_v2 as v2
    from test_router_layout import fail_class
    from test_scale_axes import fail_class_k
    out = {}
    for k, e in p.get("two", {}).items():
        r = {"end": e}
        out[k] = dict(v1=fail_class(r), v2=v2.fail_class_v2(r))
    for k, r in p.get("k", {}).items():
        out[k] = dict(v1=fail_class_k(r), v2=v2.fail_class_k_v2(r))
    return out


def stored_records():
    """Every stored Session F record (explore_out/F/*_results.json, ok) whose end statistics a classifier reads."""
    import glob
    two, multi = {}, {}
    for path in sorted(glob.glob(os.path.join(HERE, "explore_out", "F", "*_results.json"))):
        runs = json.load(open(path)).get("runs", {})
        for key, r in runs.items():
            if not isinstance(r, dict) or not r.get("ok") or not isinstance(r.get("end"), dict):
                continue
            e = r["end"]
            tag = f"{os.path.basename(path)[:-13]}:{key.split('|')[0]}|{key.split('|')[1]}"
            if r.get("k") == 2 and all(f in e for f in ("margin", "eta_key_by_key", "eta_key_by_half", "eta_key_by_index")):
                two[tag] = dict(end=e, bound=r.get("transition") is not None)
            elif all(f in e for f in ("margin", "etak_key_by_key", "etak_key_by_block", "etak_key_by_index")):
                multi[tag] = dict(r=r, bound=r.get("transition") is not None)
    return two, multi


def classifier_check():
    """v2 equals v1 except where η² by key >= 0.5 and the margin rule fired (margin >= MARGIN_PARTIAL); (ok, rows)."""
    import explore_f_fail_class_v2 as v2
    from test_router_layout import fail_class, MARGIN_PARTIAL, ETA_SPLIT
    from test_scale_axes import fail_class_k
    two, multi = stored_records()
    diffs, bad, n = [], [], 0
    for tag, d in two.items():
        r = {"end": d["end"]}
        a, b = fail_class(r), v2.fail_class_v2(r)
        e = d["end"]
        exp = e["eta_key_by_key"] >= ETA_SPLIT and e["margin"] >= MARGIN_PARTIAL
        n += 1
        if a != b:
            diffs.append((tag, "two", d["bound"], a, b, e["eta_key_by_key"], e["margin"]))
        if (a != b) != exp or (a != b and (a, b) != ("STREAM-PARTIAL", "KEY")):
            bad.append(tag)
    for tag, d in multi.items():
        r = d["r"]
        a, b = fail_class_k(r), v2.fail_class_k_v2(r)
        e = r["end"]
        exp = e["etak_key_by_key"] >= ETA_SPLIT and e["margin"] >= MARGIN_PARTIAL
        n += 1
        if a != b:
            diffs.append((tag, "multi", d["bound"], a, b, e["etak_key_by_key"], e["margin"]))
        if (a != b) != exp or (a != b and (a, b) != ("STREAM-PARTIAL", "KEY")):
            bad.append(tag)
    return not bad, dict(n=n, n_two=len(two), n_multi=len(multi), diffs=diffs, bad=bad)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import explore_common as ec
    rows = []
    star = json.load(open(STAR))
    rows.append((f"DELTA* = {star['arm']} (beta {star['beta']}, L2 keys {star['normalize']}) as committed; this child uses "
                 f"{DELTA_KW}", star["beta"] == DELTA_KW["beta"] and star["normalize"] == DELTA_KW["normalize"]))
    cpu = ec.cpu_model()
    # (1) the recorded run, with the Hebbian arm and with the delta path disabled
    store, rarm, seed, sha = REC
    key = f"{rarm}|{seed}|{sha}|{REC_CPU}"
    if cpu == REC_CPU:
        ref = ec.load_store(store)["runs"][key]
        for arm_name, impl in (("HEBB_B", "fast"), ("DELTA_B", "hebb")):
            r = run(dict(arm=arm_name, seed=seed, iters=2400, total=TOTAL, impl=impl))
            same, want = b16.same_upto(r, ref, 2400)
            rows.append((f"{arm_name} (memory impl {impl}) on seed {seed} reproduces batch 16's {key} through 2400 bit for bit: "
                         f"curve {r['curve']} vs {want}; statistics equal {same}", same))
    else:
        rows.append((f"recorded {key} NOT APPLICABLE on {cpu}", True))
    # (2) the split arm without a firing is WIN3_SLOW16's path: threshold 0 vs no trigger, through 7200
    r0 = run(dict(arm="HEBB_B", seed=2010, iters=7200, total=TOTAL, thr=0.0))
    r1 = run(dict(arm="HEBB_B", seed=2010, iters=7200, total=TOTAL, trigger=False))
    same, _ = b16.same_upto(r0, r1, 7200)
    rows.append((f"HEBB_B on seed 2010 with the threshold at 0 (checks at {[c['step'] for c in r0['checks']]}, fired "
                 f"{r0['splits']}) equals the path without a trigger bit for bit through 7200: curve {r0['curve'] == r1['curve']}, "
                 f"statistics {same}", same and r0["splits"] == 0 and len(r0["checks"]) >= 1))
    # (3) the copy op at k = 16, S = 4 (H's check)
    import explore_window_d8_child as s48c
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    m = g16.to_local(tsr.builder_for(a)(a, 2010)(), 2010)
    opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
    s48c._one_step(m, task, opt)
    st0 = {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
    W0 = m.W_g.detach().clone()
    e8.copy_nonoise_op(m, opt, 3, 7, 0)
    W1 = m.W_g.detach()
    others = [i for i in range(W0.shape[0]) if i not in (3, 7)]
    rows.append((f"copy_nonoise_op at k = {W0.shape[0]}: W_g[c0] = W_g[c*] exactly, c*'s and the other rows unchanged, Adam "
                 f"state untouched", torch.equal(W1[7], W0[3]) and torch.equal(W1[3], W0[3])
                 and all(torch.equal(W1[i], W0[i]) for i in others) and all(torch.equal(opt.state[m.W_g][k], st0[k]) for k in st0)))
    # (4) the classifier copy equals ba8901a's file; v2 = v1 except where eta2 by key >= 0.5 and the margin rule fired
    mt.ensure_main()
    orig = subprocess.run(["git", "-C", HERE, "show", f"{FC_SHA}:fail_class_v2.py"], capture_output=True, text=True,
                          check=True).stdout
    src = open(os.path.join(HERE, "explore_f_fail_class_v2.py")).read()
    head = '#!/usr/bin/env python\n"""\n'
    cut = src.index("\n\n", src.index("explore_f_fail_class_v2.py")) + 2
    rows.append((f"explore_f_fail_class_v2.py minus its note equals fail_class_v2.py at {FC_SHA[:7]} byte for byte",
                  src.startswith(head) and head + src[cut:] == orig))
    ok, res = classifier_check()
    lst = "; ".join(f"{t} ({'bound' if bd else 'unbound'}) {a} -> {b} (eta2 by key {eta:.4f}, margin {mg:.3f})"
                    for t, kind, bd, a, b, eta, mg in res["diffs"]) or "none"
    rows.append((f"v2 equals v1 on {res['n']} stored Session F records ({res['n_two']} two-stream, {res['n_multi']} multichannel) "
                 f"except where eta2 by key >= 0.5 and margin >= 0.25, all STREAM-PARTIAL -> KEY: {len(res['diffs'])} differ: "
                 f"{lst}; violations {res['bad']}", ok))
    # (5) the perfect gate on DELTA* at k = 16: one-hot, equal to the Hebbian model's
    ac = s35c.arm("CEIL")
    x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
    m0 = tsr.builder_for(ac)(ac, 2010)()
    md = mem_conv("O_DELTA_B")[0](tsr.builder_for(ac)(ac, 2010)(), 2010)
    m0.eval(), md.eval()
    with torch.no_grad():
        g0, g = m0(x)[2], md(x)[2]
    rows.append((f"the perfect gate on DELTA* (k {md.n_ch}): read gate one-hot ({bool(((g == 0) | (g == 1)).all())}) and equal to "
                 f"the Hebbian model's ({torch.equal(g, g0)}); memory {type(md.attn).__name__} beta {md.attn.beta_fixed} L2 "
                 f"{md.attn.normalize}", torch.equal(g, g0) and bool(((g == 0) | (g == 1)).all())
                 and isinstance(md.attn, ff.FastDeltaMemory) and md.attn.beta_fixed == 1.0 and md.attn.normalize))
    return [[n, bool(v)] for n, v in rows]


def classifier_rows(p):
    """For the report: the classifier CHECK's list on the stored records at report time."""
    ok, res = classifier_check()
    return dict(ok=ok, **res)


def timing(p):
    st = b16.step_timing(lambda: run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9)), True)
    a = s35c.arm("CEIL") if ARM[p["arm"]].get("perfect") else b16.arm_of(CFG)
    m = tsr.builder_for(a)(a, 0)()
    for f in ((() if ARM[p["arm"]].get("perfect") else (g16.to_local,)) + mem_conv(p["arm"])):
        m = f(m, 0)
    return [st, b16.eval_timing(CFG, m)]
