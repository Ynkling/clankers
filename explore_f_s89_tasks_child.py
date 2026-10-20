#!/usr/bin/env python
"""
explore_f_s89_tasks_child.py — EXPLORATORY, not a result. Run code of Session F's round-6 screen (parent:
explore_f_s89_tasks, whose docstring holds the phases, the rules and the readings). Runs INSIDE a child process on this
branch's own modules (no main-line hook), 1 torch thread. Training: test_router_reliability.run_one (onset_run: batch 32,
evaluation every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95).

ARMS
  NOMEM_<T>       test_short_conv's B_conv_in on the task's vocabulary (k = 1; the convolution 'in', K = 4, on the
        embeddings, entering the residual stream) with the memory replaced by ZeroRead (returns zeros): the layers add
        nothing but their LayerNorm, so a position's logits are a function of its last 4 tokens.
  NOMEM_SELF_<T>  test_short_conv's B_conv (the convolution 'layer', K = 4, on each layer's input to the memory) with the
        memory replaced by SelfRead (returns V, the position's own conv'd input): the MLP path is kept; a position's logits
        are a function of its last 10 tokens. (With 'layer' and a zero read the convolution would reach nothing.)
  Both: Adam 1e-3 on every parameter. CausalConv starts at the identity (lag 0 = 1), so the reach is the trained one.
  O_{HEBB,DELTA}_<T>[_48K]   the perfect gate (test_channel_binding's 'ceiling', k = 2) on the task's vocabulary; on ALLPOOL
        the model is AllpoolPerfectBDH (task.perfect_gate in place of the built-in gate); delta: explore_f_delta_fast,
        beta 1, L2 keys. SLOW: s43c.groups (the gate group empty: the perfect gate has no parameter) + the switch at 2400.
  BASELINE_<T>   explore_f_followup_child's S70_HEBB rule on the task: explore_f_window_delta_child.local_model (the window
        gate on the task's vocabulary), s43c.groups + the switch; statistics explore_f_tasks.pool_stats(ROUTE_AT).
  SINGLE_HEBB_<T>, SINGLE_DELTA_<T>   test_short_conv's arm B (k = 1) on the task's vocabulary, Hebbian or delta (beta 1,
        L2 keys), SLOW (the gate group empty).

REVISION (20 October, after referee V's review of 8ba99bb at claude/review-V 3ae815b; fixes 1 and 3). The pilots and the
diagnostic stand as run under code SHA ea9728c636c6; this revision changes what later runs record and what the CHECKs
tie, not those records (CHECKed: the revised path reproduces two recorded pilots bit for bit through 2400).
  Statistics: STATS_POOL is explore_f_tasks.pool_stats_inv (pool_stats plus the within-sequence forms
    eta_key_by_stream_inv, val_cos_inv, seq_routed_frac at every evaluation); run() adds outcome_pooled and
    outcome_inv (explore_f_tasks.pool_outcome) to records on the pool tasks.
  CHECKs added: part "g87" runs this module's own run() on BASELINE_LITE, seed 2700, through 2400 against G's S87
    record read at e46799a (G_REF is the commit, not the branch tip): the curve, the gradient norms at updates 1, 10
    and 100 (functions of the first 1, 10 and 100 batches) and every statistic this path records at 0, 1200 and 2400;
    part "s70" reads S70's record at 61ba46e (not the working tree); part "inv": explore_f_tasks.check_inv_stats on
    POOL and ALLPOOL, and this module's run() on O_HEBB_POOL|2800 and O_HEBB_ALLPOOL|2800 through 2400 against the S89
    records at 52e2ea7 (curve and every recorded statistic bit for bit; the new fields printed); part "tasks": the
    revised explore_f_tasks.py starts with 50dc8e4's file byte for byte (the revision is appended).
"""

import json
import os
import subprocess
import time

import torch
import torch.nn as nn

import explore_common as ec
import explore_common2 as c2
import explore_f_delta_fast as ff
import explore_f_tasks as ft
import explore_f_window_delta_child as wdc
import explore_window_recipe_child as s43c
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_multilayer_binding import build, MultiBDH
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
LR, LR_WARM, WARM = s43c.LR, s43c.LR_WARM, s43c.WARM
TASKS = {"LITE": ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), "POOL": ft.RandHeaderTaskPool(),
         "ALLPOOL": ft.RandHeaderTaskAllPool()}
STATS_POOL = ft.pool_stats_inv(c2.ROUTE_AT)
G_SHA = "e46799a"
G_REF, G_FILE = G_SHA, "explore_out/G/s87_results.json"
S70_REF, S70_FILE = "61ba46e", "explore_out/F/rh_lite_results.json"
S89_REF, S89_FILE, S89_SHA = "52e2ea7", "explore_out/F/s89_tasks_results.json", "ea9728c636c6"
F_OLD, F_POOL = "a199d5e", "50dc8e4"


# ── Models ───────────────────────────────────────────────────────────────────
class ZeroRead(nn.Module):
    """No memory: the read is zero (the layer adds nothing but its LayerNorm)."""
    def forward(self, Q, K, V):
        return torch.zeros_like(V)


class SelfRead(nn.Module):
    """No memory: each position reads its own write vector V and nothing else."""
    def forward(self, Q, K, V):
        return V


class AllpoolPerfectBDH(MultiBDH):
    """The perfect gate on RandHeaderTask-allpool: task.perfect_gate (one channel per stream index) in place of
    Instrument.gates; for a delta memory the gate is passed to it (set_gates) after the delta class's own call."""

    def forward(self, tokens, tau=None, track_sat=False):
        gr = gw = self.pool_task.perfect_gate(tokens, self.lm_head.dtype)
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(gr, gw)
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits = BDH.forward(self, tokens)[0] if self.conv is None else self.forward_conv(tokens)
        return logits, None, gr, gw


def parse(arm):
    """(kind, task name, memory, budget tag)."""
    parts = arm.split("_")
    if parts[-1] == "48K":
        parts = parts[:-1]
    t = parts[-1]
    head = "_".join(parts[:-1])
    if head in ("NOMEM", "NOMEM_SELF"):
        return dict(kind=head, task=t, memory=None)
    if head in ("O_HEBB", "O_DELTA"):
        return dict(kind="ORACLE", task=t, memory=head[2:])
    if head == "BASELINE":
        return dict(kind="BASELINE", task=t, memory="HEBB")
    if head in ("SINGLE_HEBB", "SINGLE_DELTA"):
        return dict(kind="SINGLE", task=t, memory=head.split("_")[1])
    raise KeyError(arm)


def arm_def(sp):
    task = TASKS[sp["task"]]
    if sp["kind"] == "ORACLE":
        return {a["key"]: a for a in cb_arms(task)}["ceiling"]
    if sp["kind"] == "BASELINE":
        return s43c.A
    if sp["kind"] == "NOMEM":
        return tsc.ARM["B_conv_in"]
    if sp["kind"] == "NOMEM_SELF":
        return tsc.ARM["B_conv"]
    return tsc.ARM["B"]


def make_model(sp, a, seed):
    task = TASKS[sp["task"]]
    if sp["kind"] == "BASELINE":
        return wdc.local_model(task, a, seed)
    m = build(task, a, ARCH, seed)
    if sp["kind"] == "NOMEM":
        m.attn = ZeroRead()
        return m
    if sp["kind"] == "NOMEM_SELF":
        m.attn = SelfRead()
        return m
    if sp["kind"] == "ORACLE" and sp["task"] == "ALLPOOL":
        m.__class__ = AllpoolPerfectBDH
        m.pool_task = task
    if sp["memory"] == "DELTA":
        m = ff.to_delta_fast(m, seed=seed, beta=1.0, normalize=True)
    return m


@torch.no_grad()
def single_stats(model, task, probe, step):
    nan = float("nan")
    return dict(step=step, key_cos=nan, val_cos=nan, ctx_cos=nan)


def run_kw(sp, holder, seed):
    task = TASKS[sp["task"]]
    kw = dict(task=task, grad_fn=tsc.conv_grad_norms, lr=LR,
              builder=lambda a_, s_: (lambda: make_model(sp, a_, s_)),
              stats_fn=STATS_POOL if sp["kind"] in ("BASELINE", "ORACLE") and sp["task"] != "LITE" else
              (ft.rand_header_stats(c2.ROUTE_AT) if sp["kind"] in ("BASELINE", "ORACLE") else single_stats))
    if sp["kind"] != "NOMEM" and sp["kind"] != "NOMEM_SELF":
        kw.update(run_kw=dict(param_groups=s43c.groups(holder)), check=wdc.switch_check(holder))
    return kw


def run(p):
    arm, seed, iters = p["arm"], p["seed"], p["iters"]
    sp = parse(arm)
    task = TASKS[sp["task"]]
    holder, keep = {}, {"at": -1}
    rec = run_one(arm_def(sp), seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **run_kw(sp, holder, seed))
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm, kind=sp["kind"], task=sp["task"], memory=sp["memory"], k=arm_def(sp)["n_ch"],
                   group_names=holder.get("names"), delta_cfg=getattr(m, "delta_cfg", None))
        if sp["kind"] == "BASELINE":
            rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
        if sp["kind"] in ("BASELINE", "SINGLE", "ORACLE"):
            rec["write_order"] = ft.write_order_acc(m, task, tbo.eval_batch(task))
        if sp["kind"] in ("BASELINE", "ORACLE") and sp["task"] != "LITE":
            rec["outcome_pooled"] = ft.pool_outcome(rec, invariant=False)
            rec["outcome_inv"] = ft.pool_outcome(rec, invariant=True)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def _module_at(ref, path, name):
    import importlib.util
    import tempfile
    src = _git("show", f"{ref}:{path}")
    d = tempfile.mkdtemp()
    fpath = os.path.join(d, name + ".py")
    open(fpath, "w").write(src)
    spec = importlib.util.spec_from_file_location(name, fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, src


def _lrs(arm, seed=2802, iters=WARM + 1):
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    sp = parse(arm)
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["n"] = [len(g["params"]) for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1]) + model.lm_head.sum() * 0.0)
    holder, keep = {}, {"at": iters}
    try:
        kw = run_kw(sp, holder, seed)
        kw["stats_fn"] = single_stats
        rec = run_one(arm_def(sp), seed, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, holder, rec


def checks(p):
    part = p.get("part", "tasks")
    rows = []
    cpu = ec.cpu_model()
    if part == "tasks":
        old = _git("show", f"{F_OLD}:explore_f_tasks.py")
        new = open(os.path.join(HERE, "explore_f_tasks.py")).read()
        rows.append((f"explore_f_tasks.py's first {len(old)} bytes (everything before the round-6 section) equal {F_OLD}'s file",
                     new.startswith(old)))
        v1 = _git("show", f"{F_POOL}:explore_f_tasks.py")
        rows.append((f"explore_f_tasks.py's first {len(v1)} bytes equal {F_POOL}'s file (the round-6 classes as G and H vendored "
                     f"them; the revision is appended after them, {len(new) - len(v1)} bytes)", new.startswith(v1)))
        for t in ("POOL", "ALLPOOL"):
            for nm, v in ft.check_pool_task(TASKS[t]):
                rows.append((f"{t}: {nm}", v))
        for t, T in TASKS.items():
            r = ft.freq_report(T)
            br = r["by_role"]
            rows.append((f"{t} frequencies (S87's sample): " + "; ".join(f"{k} {v['n']} types {v['min']:.4f}-{v['max']:.4f}"
                                                                    for k, v in br.items())
                         + f"; largest-ratio cut {len(r['largest_ratio'])} types {sorted(set(r['largest_ratio_roles']))}; "
                         f"half-mass cut {len(r['half_mass'])} types "
                         f"{ {x: r['half_mass_roles'].count(x) for x in sorted(set(r['half_mass_roles']))} }", True))
        lr, tab = ft.largest_ratio_cut(ft.freq_sample(TASKS["LITE"]))
        g = [(0, 0.1137), (1, 0.1136), (3, 0.1027), (4, 0.1026), (5, 0.1022), (2, 0.1017)]
        mine = [(t, round(f, 4)) for t, f in tab[:6]]
        rows.append((f"LITE's table equals G's published S87 table ({mine}) and its cut G's set {{0, 1, 2, 3, 4, 5}} ({sorted(lr)})",
                     mine == g and sorted(lr) == [0, 1, 2, 3, 4, 5]))
        return [[n, bool(v)] for n, v in rows]
    if part == "g87":
        gmod, _ = _module_at(G_SHA, "explore_f_tasks.py", "explore_f_tasks_g87")
        gl = gmod.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)
        b_mine = TASKS["LITE"].make_batch(tbo.BATCH, torch.Generator().manual_seed(2700 + 10_000))[0]
        b_g = gl.make_batch(tbo.BATCH, torch.Generator().manual_seed(2700 + 10_000))[0]
        rows.append((f"the first training batch at seed 2700 (onset_run's generator, seed + 10000, batch {tbo.BATCH}) from this "
                     f"module equals the one from G's vendored explore_f_tasks.py at {G_SHA}", torch.equal(b_mine, b_g)))
        import explore_f_followup_child as fu
        d = json.loads(_git("show", f"{G_REF}:{G_FILE}"))
        key = [k for k in d["runs"] if k.startswith("BASELINE|2700|")][0]
        ref = d["runs"][key]
        r = fu.run(dict(arm="S70_HEBB", seed=2700, iters=2400))
        want = [c for c in ref["curve"] if c[0] <= 2400]
        gk = [k for k in ref["grad"] if int(k) <= 2400]
        gr_ok = all(r["grad"].get(k) == ref["grad"][k] for k in gk)
        rows.append((f"S70's own path (S70_HEBB = G's BASELINE) on seed 2700, on this CPU ({cpu}), reproduces G's S87 {key} "
                     f"at {G_REF} through 2400 bit for bit: curve {r['curve']} vs {want}; gradient norms at updates {gk} "
                     f"(functions of the first 1, 10 and 100 batches) {gr_ok}", r["curve"] == want and gr_ok))
        # this screen's own path (referee V's fix 3)
        r = run(dict(arm="BASELINE_LITE", seed=2700, iters=2400))
        gr_ok = all(r["grad"].get(k) == ref["grad"][k] for k in gk)
        steps = (0, 1200, 2400)
        mine = {s["step"]: s for s in r["stats"] if s["step"] in steps}
        theirs = {s["step"]: s for s in ref["stats"] if s["step"] in steps}
        st_ok, extra = set(mine) == set(steps) == set(theirs), set()
        for t in steps:
            for f, v in mine.get(t, {}).items():
                st_ok &= f in theirs.get(t, {}) and theirs[t][f] == v
            extra |= set(theirs.get(t, {})) - set(mine.get(t, {}))
        nf = sorted({f for t in steps for f in mine.get(t, {})})
        rows.append((f"this screen's own path (explore_f_s89_tasks_child.run, BASELINE_LITE) on seed 2700, on this CPU ({cpu}), "
                     f"reproduces G's S87 {key} at {G_REF} through 2400 bit for bit: curve {r['curve']} vs {want}; gradient "
                     f"norms at updates {gk} {gr_ok}; every statistic this path records at {list(steps)} ({len(nf)} fields: "
                     f"{', '.join(nf)}) equal {st_ok} (G's record also holds {sorted(extra)}, G's own additions, not "
                     f"computed on this path)", r["curve"] == want and gr_ok and st_ok))
        return [[n, bool(v)] for n, v in rows]
    if part == "s70":
        import explore_f_followup_child as fu
        key = "S70_DELTA|330|ff908bdbc0b2|Intel(R) Xeon(R) Processor @ 2.10GHz"
        ref = json.loads(_git("show", f"{S70_REF}:{S70_FILE}"))["runs"][key]
        r = fu.run(dict(arm="S70_DELTA", seed=330, iters=2400))
        want = [c for c in ref["curve"] if c[0] <= 2400]
        sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
        so = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= 2400]
        rows.append((f"S70 reproduction on this CPU ({cpu}): {key} (read at {S70_REF}) through 2400 bit for bit with S70's own path: curve "
                     f"{r['curve']} vs {want}; statistics {sn == so}", r["curve"] == want and sn == so))
        return [[n, bool(v)] for n, v in rows]
    if part == "inv":
        for t in ("POOL", "ALLPOOL"):
            rows.extend(ft.check_inv_stats(TASKS[t]))
        d = json.loads(_git("show", f"{S89_REF}:{S89_FILE}"))
        for arm in ("O_HEBB_POOL", "O_HEBB_ALLPOOL"):
            key = [k for k in d["runs"] if k.startswith(f"{arm}|2800|{S89_SHA}|")][0]
            ref = d["runs"][key]
            r = run(dict(arm=arm, seed=2800, iters=2400))
            want = [c for c in ref["curve"] if c[0] <= 2400]
            so = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= 2400]
            sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
            st_ok = len(so) == len(sn) and all(all(b.get(f) == v for f, v in a.items()) for a, b in zip(so, sn))
            new = {f: round(v, 4) for f, v in sn[-1].items() if f.endswith("_inv") or f == "seq_routed_frac"}
            rows.append((f"the revised path reproduces the S89 record {key} (read at {S89_REF}) through 2400 bit for bit on this "
                         f"CPU ({cpu}): curve {r['curve']} vs {want}; every recorded statistic at {[s['step'] for s in so]} "
                         f"equal {st_ok}; added at 2400: {new}", r["curve"] == want and st_ok))
        return [[n, bool(v)] for n, v in rows]
    # part == "models"
    from test_instrument_v2 import perfect_gate_general
    for t in ("POOL", "ALLPOOL"):
        task = TASKS[t]
        x = task.make_batch(8, torch.Generator().manual_seed(3))[0][:, :-1]
        for mem in ("HEBB", "DELTA"):
            sp = dict(kind="ORACLE", task=t, memory=mem)
            m = make_model(sp, arm_def(sp), 2800)
            m.eval()
            with torch.no_grad():
                lg, _, gr, gw = m(x)
            pg = task.perfect_gate(x)
            ok = torch.equal(gr, pg) and bool(((gr == 0) | (gr == 1)).all()) and gr.shape[-1] == 2
            extra = ""
            if mem == "DELTA":
                ok &= torch.equal(m.attn._gates[0], pg)
                extra = f"; the delta memory receives the k = 2 gate ({torch.equal(m.attn._gates[0], pg)})"
            if t == "ALLPOOL" and mem == "HEBB":
                mb = build(task, arm_def(sp), ARCH, 2800)                      # the built-in gate over the 16 CTX types
                mb.eval()
                with torch.no_grad():
                    lb = mb(x)[0]
                ok &= torch.equal(lb, lg)
                extra = f"; logits equal the built-in perfect gate over the 16 CTX types bit for bit ({torch.equal(lb, lg)})"
            if t == "POOL":
                ok &= torch.equal(pg, perfect_gate_general(x, (0, 1)))
            rows.append((f"{t} perfect gate on {mem}: one-hot with k = 2, equal to task.perfect_gate ({ok}){extra}", ok))
    # the no-memory models: nothing beyond the convolution's reach changes a position's logits (conv weights randomised,
    # since CausalConv starts at the identity)
    for kind, reach, conv in (("NOMEM", 4, "in"), ("NOMEM_SELF", 10, "layer")):
        for t in ("POOL", "ALLPOOL"):
            task = TASKS[t]
            sp = dict(kind=kind, task=t, memory=None)
            m = make_model(sp, arm_def(sp), 2800)
            with torch.no_grad():
                m.short_conv.conv_w.copy_(torch.randn(m.short_conv.conv_w.shape, generator=torch.Generator().manual_seed(9)))
            m.eval()
            x = task.make_batch(8, torch.Generator().manual_seed(4))[0][:, :-1]
            tq = task.qpos[0]

            def bump(j):
                y = x.clone()
                y[:, tq - j] = (y[:, tq - j] + 1) % task.vocab
                return y
            with torch.no_grad():
                l0 = m(x)[0][:, tq]
                far = torch.equal(m(bump(reach))[0][:, tq], l0)
                near = not torch.equal(m(bump(reach - 1))[0][:, tq], l0)
            zero = True
            if kind == "NOMEM":
                zero = torch.equal(m.attn(None, None, torch.ones(2, 1, 3, 4)), torch.zeros(2, 1, 3, 4))
            rows.append((f"{kind} on {t} (conv '{conv}'): the query position's logits ignore a token {reach} back ({far}) and see one "
                         f"{reach - 1} back ({near}); the read is {'zero' if kind == 'NOMEM' else 'the own V'} ({zero}); k {m.n_ch}",
                         far and near and zero and m.conv == conv))
    # groups and lrs through the real path
    for arm in ("BASELINE_POOL", "SINGLE_DELTA_ALLPOOL", "O_DELTA_ALLPOOL"):
        lrs, ids, holder, rec = _lrs(arm)
        n = len(lrs)
        want_gate = arm.startswith("BASELINE")
        ok = (n == WARM + 1 and lrs[0] == [LR, LR_WARM] and lrs[WARM - 1] == [LR, LR_WARM] and lrs[WARM] == [LR, LR]
              and (ids["n"][0] == 3) == want_gate and bool(rec.get("ok")))
        rows.append((f"{arm}: groups {holder.get('names', [[]])[0]} / {len(holder.get('names', [[], []])[1])} rest parameters; lrs at "
                     f"1 {lrs[0] if n else '?'}, {WARM} {lrs[WARM - 1] if n >= WARM else '?'}, {WARM + 1} "
                     f"{lrs[WARM] if n > WARM else '?'}", ok))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    from torch.optim.optimizer import register_optimizer_step_post_hook
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    sp = parse(p["arm"])
    task = TASKS[sp["task"]]
    m = make_model(sp, arm_def(sp), 0)
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    return [dd[len(dd) // 2], 2 * (time.time() - t0)]
