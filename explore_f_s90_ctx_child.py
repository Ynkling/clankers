#!/usr/bin/env python
"""
explore_f_s90_ctx_child.py — EXPLORATORY, not a result. Run code of Session F's S90-F (parent: explore_f_s90_ctx, whose
docstring holds the phases, the rules and the readings). Runs INSIDE a child process on this branch's own modules (no
main-line hook), 1 torch thread. Training: test_router_reliability.run_one (onset_run: batch 32, evaluation every 1200 on
2048 held-out sequences, early stop after 3 evaluations >= 0.95), as S89-F.

TASKS: CTXPOOL = explore_f_tasks.RandHeaderTaskCtxPool() (S = 2); DOCS3 / DOCS4 = RandHeaderTaskDocs(3 / 4). For the
CHECKs only: LITE, POOL, ALLPOOL as S89-F's.
ARMS (<T> a task; models as explore_f_s89_tasks_child's, which this module imports):
  NOMEM_<T>, NOMEM_SELF_<T>     S89-F's no-memory models (conv 'in' + zero read; conv 'layer' + own-V read), Adam 1e-3.
  O_HEBBCONV_<T>[_48K]          the perfect gate (k = S, task.perfect_gate: the per-sequence index; S89's
                                AllpoolPerfectBDH), Hebbian, + test_short_conv.LAYER (conv 'layer', K = 4); SLOW.
  O_DELTA_<T>[_48K]             the perfect gate, the delta memory (explore_f_delta_fast, beta 1, L2 keys), no conv; SLOW.
  TL_HEBBCONV_<T>, TL_DELTA_<T> the type-level perfect gate (explore_f_tasks.type_level_gate with balanced_map(16, S):
                                TypeGateK), the same two memories; SLOW. Descriptive.
  SINGLE_DELTA_<T>              one delta channel, no gate (test_short_conv's arm B, k = 1; beta 1, L2 keys); SLOW.
  SINGLE_HEBBCONV_<T>           one Hebbian channel + conv 'layer' (test_short_conv's B_conv, k = 1); SLOW.
  SLOW: s43c.groups + wdc.switch_check (every parameter at 1e-4 for updates 1-2400, 1e-3 after; the gate group empty).
STATISTICS: perfect and type-level gates on the new tasks: explore_f_tasks.ctx_stats(ROUTE_AT); on POOL / ALLPOOL (CHECK
reproductions only) S89's STATS_POOL; single channels and no-memory models: S89's single_stats (NaN). Every gated or
single run records explore_f_tasks.write_order_acc_idx (the corrected write order).
"""

import json

import torch

import explore_common as ec
import explore_common2 as c2
import explore_f_delta_fast as ff
import explore_f_s89_tasks_child as s89
import explore_f_tasks as ft
import explore_f_window_delta_child as wdc
import explore_window_recipe_child as s43c
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_channel_binding import ARCH, arms as cb_arms
from test_multilayer_binding import build
from test_router_reliability import run_one

LR, LR_WARM, WARM = s43c.LR, s43c.LR_WARM, s43c.WARM
NEW = {"CTXPOOL": ft.RandHeaderTaskCtxPool(), "DOCS3": ft.RandHeaderTaskDocs(3), "DOCS4": ft.RandHeaderTaskDocs(4)}
TASKS = dict(s89.TASKS, **NEW)
STATS_CTX = ft.ctx_stats(c2.ROUTE_AT)


class TypeGateK(s89.AllpoolPerfectBDH):
    """The type-level perfect gate at k = S: explore_f_tasks.type_level_gate with balanced_map(n_ctx, S)."""

    def forward(self, tokens, tau=None, track_sat=False):
        cm = ft.balanced_map(self.pool_task.n_ctx, self.pool_task.S)
        gr = gw = ft.type_level_gate(self.pool_task, tokens, cm, self.lm_head.dtype)
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(gr, gw)
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits = BDH.forward(self, tokens)[0] if self.conv is None else self.forward_conv(tokens)
        return logits, None, gr, gw


def parse(arm):
    """(kind, task, memory, conv)."""
    parts = arm.split("_")
    if parts[-1] == "48K":
        parts = parts[:-1]
    t = parts[-1]
    head = "_".join(parts[:-1])
    if head in ("NOMEM", "NOMEM_SELF"):
        return dict(kind=head, task=t, memory=None, conv=head == "NOMEM_SELF")
    kind = {"O": "ORACLE", "TL": "TYPE", "SINGLE": "SINGLE"}[parts[0]]
    mem = parts[1]
    assert mem in ("HEBBCONV", "DELTA", "HEBB"), arm
    return dict(kind=kind, task=t, memory="DELTA" if mem == "DELTA" else "HEBB", conv=mem == "HEBBCONV")


def arm_def(sp):
    task = TASKS[sp["task"]]
    if sp["kind"] == "NOMEM":
        return tsc.ARM["B_conv_in"]
    if sp["kind"] == "NOMEM_SELF":
        return tsc.ARM["B_conv"]
    a = {x["key"]: x for x in cb_arms(task)}["ceiling"] if sp["kind"] in ("ORACLE", "TYPE") else tsc.ARM["B"]
    return dict(a, model_kw=tsc.LAYER) if sp["conv"] else a


def make_model(sp, a, seed):
    task = TASKS[sp["task"]]
    if sp["task"] in s89.TASKS and sp["kind"] in ("NOMEM", "NOMEM_SELF", "ORACLE", "SINGLE"):
        m = s89.make_model(dict(kind=sp["kind"], task=sp["task"], memory=sp["memory"]), a, seed)   # S89's path, unchanged
        return m
    m = build(task, a, ARCH, seed)
    if sp["kind"] == "NOMEM":
        m.attn = s89.ZeroRead()
        return m
    if sp["kind"] == "NOMEM_SELF":
        m.attn = s89.SelfRead()
        return m
    if sp["kind"] == "ORACLE":
        m.__class__ = s89.AllpoolPerfectBDH
        m.pool_task = task
    if sp["kind"] == "TYPE":
        m.__class__ = TypeGateK
        m.pool_task = task
    if sp["memory"] == "DELTA":
        m = ff.to_delta_fast(m, seed=seed, beta=1.0, normalize=True)
    return m


def stats_for(sp):
    if sp["kind"] in ("ORACLE", "TYPE"):
        return STATS_CTX if sp["task"] in NEW else (s89.STATS_POOL if sp["task"] != "LITE" else ft.rand_header_stats(c2.ROUTE_AT))
    return s89.single_stats


def run_kw(sp, holder, seed):
    kw = dict(task=TASKS[sp["task"]], grad_fn=tsc.conv_grad_norms, lr=LR,
              builder=lambda a_, s_: (lambda: make_model(sp, a_, s_)), stats_fn=stats_for(sp))
    if sp["kind"] not in ("NOMEM", "NOMEM_SELF"):
        kw.update(run_kw=dict(param_groups=s43c.groups(holder)), check=wdc.switch_check(holder))
    return kw


def run(p):
    arm, seed, iters = p["arm"], p["seed"], p["iters"]
    sp = parse(arm)
    task = TASKS[sp["task"]]
    a = arm_def(sp)
    holder, keep = {}, {"at": -1}
    rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **run_kw(sp, holder, seed))
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm, kind=sp["kind"], task=sp["task"], memory=sp["memory"], conv=m.conv, k=a["n_ch"],
                   group_names=holder.get("names"), delta_cfg=getattr(m, "delta_cfg", None))
        if sp["kind"] in ("ORACLE", "TYPE", "SINGLE"):
            rec["write_order"] = ft.write_order_acc_idx(m, task, tbo.eval_batch(task))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _lrs(arm, seed=3060, iters=WARM + 1):
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
        kw["stats_fn"] = s89.single_stats
        rec = run_one(arm_def(sp), seed, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, holder, rec


def _store_at(ref, path):
    return json.loads(s89._git("show", f"{ref}:{path}"))


def _tie(arm, seed, ref, path, sha, cpu_pref=None):
    """this module's run() on a recorded arm through 2400 against the record (curve and every recorded statistic)."""
    d = _store_at(ref, path)
    keys = [k for k in d["runs"] if k.startswith(f"{arm}|{seed}|{sha}|") and d["runs"][k].get("ok")]
    if cpu_pref:
        keys = sorted(keys, key=lambda k: not k.endswith(cpu_pref))
    key = keys[0]
    rec = d["runs"][key]
    r = run(dict(arm=arm, seed=seed, iters=2400))
    want = [c for c in rec["curve"] if c[0] <= 2400]
    so = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= 2400]
    sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
    st_ok = len(so) == len(sn) and all(all((b.get(f) == v) or (v != v and b.get(f) != b.get(f)) for f, v in a.items())
                                       for a, b in zip(so, sn))
    return key, r["curve"] == want and st_ok, f"curve {r['curve']} vs {want}; statistics at {[s['step'] for s in so]} equal {st_ok}"


def checks(p):
    part = p.get("part", "tasks")
    rows = []
    cpu = ec.cpu_model()
    if part == "tasks":
        old = s89._git("show", "9017b82:explore_f_tasks.py")
        new = open(s89.os.path.join(s89.HERE, "explore_f_tasks.py")).read()
        prev = s89._git("show", "b0c6332:explore_f_tasks.py")
        rows.append((f"explore_f_tasks.py equals 9017b82's file (the module G vendors) and starts with b0c6332's "
                     f"{len(prev)} bytes (round 6's classes byte-identical)", new == old and new.startswith(prev)))
        for t, T in NEW.items():
            rows.extend(ft.check_ctx_task(T))
            rows.extend(ft.check_inv_stats_k(T))
            rows.extend(ft.check_write_order_idx(T))
            r = ft.freq_report(T)
            br = r["by_role"]
            cnt = lambda roles: {x: roles.count(x) for x in sorted(set(roles))}
            rows.append((f"{t} frequencies (S87's sample): " + "; ".join(f"{k} {v['n']} types {v['min']:.4f}-{v['max']:.4f}"
                                                                    for k, v in br.items())
                         + f"; largest-ratio cut {len(r['largest_ratio'])} types {cnt(r['largest_ratio_roles'])}; half-mass "
                         f"cut {len(r['half_mass'])} types {cnt(r['half_mass_roles'])} (mass {r['half_mass_mass']:.3f})", True))
        # the corrected write order equals the old one where CTX ids are the indices (-pool), on a model
        sp = dict(kind="ORACLE", task="POOL", memory="HEBB", conv=False)
        m = s89.make_model(dict(kind="ORACLE", task="POOL", memory="HEBB"), s89.arm_def(dict(kind="ORACLE", task="POOL")), 3)
        d = tbo.eval_batch(TASKS["POOL"])[:512]
        a, b = ft.write_order_acc(m, TASKS["POOL"], d), ft.write_order_acc_idx(m, TASKS["POOL"], d)
        rows.append((f"write_order_acc_idx equals write_order_acc on -pool (an initial model, 512 held-out sequences): {a == b}",
                     a == b))
        x = TASKS["ALLPOOL"].make_batch(512, torch.Generator().manual_seed(5))[0][:, :-1]
        qp = TASKS["ALLPOOL"].qpos[0]
        bad = bool((x[:, qp - 1] != TASKS["ALLPOOL"].stream_labels(x)[:, qp - 1]).any())
        rows.append((f"ERRATUM: on -allpool the query's CTX token differs from its stream index in "
                     f"{int((x[:, qp - 1] != TASKS['ALLPOOL'].stream_labels(x)[:, qp - 1]).sum())} of 512 sequences, so "
                     f"write_order_acc's split is invalid there (printed)", bad))
        rows.extend(ft.check_write_order_idx(TASKS["ALLPOOL"]))
        return [[n, bool(v)] for n, v in rows]
    if part == "ties":
        # this module's path on recorded arms (the new task code disabled: S89's tasks), on this CPU
        for arm, seed, ref, path, sha in (
                ("NOMEM_SELF_LITE", 2800, "52e2ea7", "explore_out/F/s89_tasks_results.json", "ea9728c636c6"),
                ("O_DELTA_ALLPOOL", 2800, "52e2ea7", "explore_out/F/s89_tasks_results.json", "ea9728c636c6"),
                ("O_HEBBCONV_ALLPOOL", 2801, "3e3b1bf", "explore_out/F/s89_post_results.json", "267d08456c38")):
            src = arm.replace("HEBBCONV", "HEBB_CONV") if arm.startswith("O_HEBBCONV") else arm
            if arm.startswith("O_HEBBCONV"):
                # S89-post's arm name; this module's O_HEBBCONV_ALLPOOL is the same model (S89's AllpoolPerfectBDH + LAYER)
                d = _store_at(ref, path)
                key = [k for k in d["runs"] if k.startswith(f"{src}|{seed}|{sha}|") and d["runs"][k].get("ok")][0]
                rec = d["runs"][key]
                r = run(dict(arm=arm, seed=seed, iters=2400))
                want = [c for c in rec["curve"] if c[0] <= 2400]
                so = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= 2400]
                sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
                ok = r["curve"] == want and so == sn
                msg = f"curve {r['curve']} vs {want}; statistics at {[s['step'] for s in so]} equal {so == sn}"
            else:
                key, ok, msg = _tie(arm, seed, ref, path, sha)
            rows.append((f"this module's path reproduces {key} (read at {ref}) through 2400 bit for bit on this CPU ({cpu}): "
                         f"{msg}", ok))
        return [[n, bool(v)] for n, v in rows]
    # part == "models"
    for t, task in NEW.items():
        x = task.make_batch(8, torch.Generator().manual_seed(3))[0][:, :-1]
        S = task.S
        for kind in ("ORACLE", "TYPE"):
            for mem, conv in (("HEBB", True), ("DELTA", False)):
                sp = dict(kind=kind, task=t, memory=mem, conv=conv)
                m = make_model(sp, arm_def(sp), 3060)
                m.eval()
                with torch.no_grad():
                    lg, _, gr, gw = m(x)
                want = task.perfect_gate(x) if kind == "ORACLE" else ft.type_level_gate(task, x, ft.balanced_map(task.n_ctx, S))
                ok = torch.equal(gr, want) and gr.shape[-1] == S and m.n_ch == S and (m.conv == ("layer" if conv else None))
                extra = ""
                if mem == "DELTA":
                    ok &= torch.equal(m.attn._gates[0], want) and isinstance(m.attn, ff.FastDeltaMemory)
                    extra = f"; the delta memory (beta {m.attn.beta_fixed}) receives the gate"
                rows.append((f"{t} {kind} {'HEBBCONV' if conv else mem}: the gate is "
                             f"{'task.perfect_gate' if kind == 'ORACLE' else 'the balanced type-level gate'} (one-hot, k = "
                             f"{S}) ({ok}); conv {m.conv}{extra}", ok))
        for mem, conv in (("DELTA", False), ("HEBB", True)):
            sp = dict(kind="SINGLE", task=t, memory=mem, conv=conv)
            m = make_model(sp, arm_def(sp), 3062)
            ok = m.n_ch == 1 and m.conv == ("layer" if conv else None) and (isinstance(m.attn, ff.FastDeltaMemory) == (mem == "DELTA"))
            rows.append((f"{t} SINGLE_{'HEBBCONV' if conv else 'DELTA'}: one channel, no gate (k {m.n_ch}), conv {m.conv}, "
                         f"memory {type(m.attn).__name__}", ok))
        for kind, reach in (("NOMEM", 4), ("NOMEM_SELF", 10)):
            sp = dict(kind=kind, task=t, memory=None, conv=kind == "NOMEM_SELF")
            m = make_model(sp, arm_def(sp), 3060)
            with torch.no_grad():
                m.short_conv.conv_w.copy_(torch.randn(m.short_conv.conv_w.shape, generator=torch.Generator().manual_seed(9)))
            m.eval()
            xx = task.make_batch(8, torch.Generator().manual_seed(4))[0][:, :-1]
            tq = task.qpos[0]

            def bump(j):
                y = xx.clone()
                y[:, tq - j] = (y[:, tq - j] + 1) % task.vocab
                return y
            with torch.no_grad():
                l0 = m(xx)[0][:, tq]
                far = torch.equal(m(bump(reach))[0][:, tq], l0)
                near = not torch.equal(m(bump(reach - 1))[0][:, tq], l0)
            rows.append((f"{t} {kind}: the query's logits ignore a token {reach} back ({far}) and see one {reach - 1} back "
                         f"({near})", far and near))
    # the positional bound (descriptive, approximate as V's): P(the target pair, its KEY and VAL, inside the reach of the
    # query's logits) + the rest at chance among the n_vals - S + 1 values left when the key's other S - 1 are excluded
    for t, task in NEW.items():
        x = task.make_batch(4096, torch.Generator().manual_seed(87_000))[0]
        qp = task.qpos[0]
        lab = task.stream_labels(x)
        body = x[:, :qp - 1]
        hit = (body == x[:, qp][:, None]) & (lab[:, :qp - 1] == lab[:, qp][:, None])
        kp = torch.where(hit, torch.arange(qp - 1).expand_as(hit), torch.full_like(hit, -1, dtype=torch.long)).max(1).values
        out = []
        for reach in (4, 10):
            seen = (kp >= qp - reach + 1).double().mean().item()
            out.append(f"reach {reach}: {seen:.3f}, bound {seen + (1 - seen) / (task.n_vals - task.S + 1):.3f}")
        rows.append((f"{t} positional bound (descriptive): P(the target pair inside the reach) and P + rest/(n_vals - S + 1): "
                     + "; ".join(out), True))
    for arm in ("O_HEBBCONV_DOCS4", "O_DELTA_CTXPOOL", "TL_DELTA_DOCS3", "SINGLE_DELTA_CTXPOOL", "SINGLE_HEBBCONV_DOCS4"):
        lrs, ids, holder, rec = _lrs(arm)
        n = len(lrs)
        conv = parse(arm)["conv"]
        ok = (n == WARM + 1 and lrs[0] == [LR, LR_WARM] and lrs[WARM - 1] == [LR, LR_WARM] and lrs[WARM] == [LR, LR]
              and ids["n"][0] == 0 and bool(rec.get("ok")) and (("short_conv.conv_w" in holder["names"][1]) == conv))
        rows.append((f"{arm}: groups {holder['names'][0]} / {len(holder['names'][1])} rest parameters (conv in rest: "
                     f"{'short_conv.conv_w' in holder['names'][1]}); lrs at 1 {lrs[0]}, {WARM} {lrs[WARM - 1]}, {WARM + 1} "
                     f"{lrs[WARM]}", ok))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
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
