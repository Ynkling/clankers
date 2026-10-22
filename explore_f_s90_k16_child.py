#!/usr/bin/env python
"""
explore_f_s90_k16_child.py — EXPLORATORY, not a result. Run code of Session F's k = 16 type-level pilot (parent:
explore_f_s90_k16, whose docstring holds the reading). Runs INSIDE a child process on this branch's own modules (no
main-line hook), 1 torch thread; run_one / onset_run as S89-F and S90-F.

ARMS TL16_<M>_<T>, M in HEBBCONV (Hebbian + conv 'layer', K = 4) / DELTA (beta 1, L2 keys, no conv), T in CTXPOOL /
ALLPOOL: the perfect-gate pilot of S90-F (CTXPOOL) or S89-F / S89-post (ALLPOOL) with one change: k = 16 channels and the
type-level perfect gate with one channel per CTX type (explore_f_tasks.type_level_gate with the identity map, TypeGate16),
in place of the per-sequence index at k = 2. SLOW. A sequence's contexts are distinct types, so the same-channel matrix
G = gr·gwᵀ equals the per-sequence perfect gate's exactly, and the model's parameters do not depend on k for the
perfect-gate kind (CHECKed: the parameters and the logits at initialisation equal the k = 2 pilot's bit for bit).
STATISTICS (k16_stats): explore_f_tasks.pool_stats plus inv_from_gate_k (sep_key_inv, seq_separated_frac, val_cos_inv_k)
at every evaluation. Set to NaN: key_sep and key_part (lite ids), every eta_* (they read channel 0, which is CTX type 0
here), and the pooled sep / key_cos / val_cos / ctx_cos (pooled over sequences under the first-appearance index, so near
1 for any type-level gate; V's S90 review fix 6). So run_one's DISCOVERED (end val_cos < 0.5) is not computed here.
"""

import json

import torch

import explore_f_delta_fast as ff
import explore_f_s89_tasks_child as s89
import explore_f_s90_ctx_child as s90
import explore_f_tasks as ft
import explore_common2 as c2
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_channel_binding import ARCH, arms as cb_arms
from test_multilayer_binding import build
from test_router_reliability import run_one

K = 16
TASKS = {"CTXPOOL": s90.TASKS["CTXPOOL"], "ALLPOOL": s90.TASKS["ALLPOOL"]}


class TypeGate16(s89.AllpoolPerfectBDH):
    """The type-level perfect gate with one channel per CTX type (k = n_ctx = 16)."""

    def forward(self, tokens, tau=None, track_sat=False):
        gr = gw = ft.type_level_gate(self.pool_task, tokens, tuple(range(self.pool_task.n_ctx)), self.lm_head.dtype)
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(gr, gw)
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits = BDH.forward(self, tokens)[0] if self.conv is None else self.forward_conv(tokens)
        return logits, None, gr, gw


def parse(arm):
    _, mem, t = arm.split("_")
    return dict(task=t, memory="DELTA" if mem == "DELTA" else "HEBB", conv=mem == "HEBBCONV")


def arm_def(sp, k=K):
    a = dict({x["key"]: x for x in cb_arms(TASKS[sp["task"]])}["ceiling"], n_ch=k)
    return dict(a, model_kw=tsc.LAYER) if sp["conv"] else a


def make_model(sp, a, seed):
    task = TASKS[sp["task"]]
    m = build(task, a, ARCH, seed)
    m.__class__ = TypeGate16
    m.pool_task = task
    if sp["memory"] == "DELTA":
        m = ff.to_delta_fast(m, seed=seed, beta=1.0, normalize=True)
    return m


def pilot_model(sp, seed):
    """the k = 2 per-sequence perfect-gate pilot's model (S90's path on CTXPOOL, S89's on ALLPOOL)."""
    sp2 = dict(kind="ORACLE", task=sp["task"], memory=sp["memory"], conv=sp["conv"])
    return s90.make_model(sp2, s90.arm_def(sp2), seed)


NAN_FIELDS = ("key_sep", "key_part", "sep", "key_cos", "val_cos", "ctx_cos")


def k16_stats(route_at):
    base = ft.pool_stats(route_at)
    nan = float("nan")

    @torch.no_grad()
    def stats(model, task, probe, step):
        out = base(model, task, probe, step)
        for f in list(out):
            if f in NAN_FIELDS or f.startswith("eta_"):
                out[f] = nan
        out.update(ft.pool_routing_stats_inv_k(model, task, probe))
        return out
    return stats


STATS = k16_stats(c2.ROUTE_AT)


def run_kw(sp, holder):
    kw = dict(task=TASKS[sp["task"]], grad_fn=tsc.conv_grad_norms, lr=s90.LR,
              builder=lambda a_, s_: (lambda: make_model(sp, a_, s_)), stats_fn=STATS)
    kw.update(run_kw=dict(param_groups=s90.s43c.groups(holder)), check=s90.wdc.switch_check(holder))
    return kw


def run(p):
    arm, seed, iters = p["arm"], p["seed"], p["iters"]
    sp = parse(arm)
    task = TASKS[sp["task"]]
    a = arm_def(sp)
    holder, keep = {}, {"at": -1}
    rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **run_kw(sp, holder))
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm, kind="TYPE16", task=sp["task"], memory=sp["memory"], conv=m.conv, k=a["n_ch"],
                   group_names=holder.get("names"), delta_cfg=getattr(m, "delta_cfg", None))
        rec["write_order"] = ft.write_order_acc_idx(m, task, tbo.eval_batch(task))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import explore_common as ec
    rows = []
    cpu = ec.cpu_model()
    for arm in ("TL16_HEBBCONV_CTXPOOL", "TL16_DELTA_CTXPOOL", "TL16_HEBBCONV_ALLPOOL", "TL16_DELTA_ALLPOOL"):
        sp = parse(arm)
        task = TASKS[sp["task"]]
        m = make_model(sp, arm_def(sp), 3060)
        ref = pilot_model(sp, 3060)
        x = task.make_batch(64, torch.Generator().manual_seed(3))[0][:, :-1]
        m.eval()
        ref.eval()
        with torch.no_grad():
            lg, _, gr, gw = m(x)
            lr_, _, g2, _ = ref(x)
        want = ft.type_level_gate(task, x, tuple(range(task.n_ctx)))
        G16, G2 = torch.einsum("btk,bsk->bts", gr, gw), torch.einsum("btk,bsk->bts", g2, g2)
        p16, p2 = dict(m.named_parameters()), dict(ref.named_parameters())
        same_p = set(p16) == set(p2) and all(torch.equal(p16[n], p2[n]) for n in p16)
        ok = (torch.equal(gr, want) and gr.shape[-1] == K and m.n_ch == K and torch.equal(G16, G2) and same_p
              and torch.equal(lg, lr_) and m.conv == ("layer" if sp["conv"] else None))
        extra = ""
        if sp["memory"] == "DELTA":
            ok &= torch.equal(m.attn._gates[0], want) and isinstance(m.attn, ff.FastDeltaMemory)
            extra = "; the delta memory receives the 16-channel gate"
        rows.append((f"{arm}: the gate is one channel per CTX type (k = {gr.shape[-1]}); its same-channel matrix equals the "
                     f"per-sequence perfect gate's ({torch.equal(G16, G2)}); the parameters equal the k = 2 pilot's at seed "
                     f"3060 ({same_p}); the logits at initialisation equal it bit for bit ({torch.equal(lg, lr_)}){extra}", ok))
    # groups and lrs through the real path
    for arm in ("TL16_HEBBCONV_ALLPOOL", "TL16_DELTA_CTXPOOL"):
        sp = parse(arm)
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        lrs, ids = [], {}

        def hook(opt, args, kwargs):
            if not ids:
                ids["n"] = [len(g["params"]) for g in opt.param_groups]
            lrs.append([g["lr"] for g in opt.param_groups])
        saved = (tbo.evaluate, tbo.logits_of)
        h = register_optimizer_step_pre_hook(hook)
        tbo.evaluate = lambda model, task, data: (0.5, 1.0)
        tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1]) + model.lm_head.sum() * 0.0)
        holder = {}
        try:
            kw = run_kw(sp, holder)
            kw["stats_fn"] = s89.single_stats
            rec = run_one(arm_def(sp), 3060, s90.WARM + 1, keep={"at": s90.WARM + 1}, **kw)
        finally:
            tbo.evaluate, tbo.logits_of = saved
            h.remove()
        W = s90.WARM
        ok = (len(lrs) == W + 1 and lrs[0] == [s90.LR, s90.LR_WARM] and lrs[W - 1] == [s90.LR, s90.LR_WARM]
              and lrs[W] == [s90.LR, s90.LR] and ids["n"][0] == 0 and bool(rec.get("ok"))
              and ("short_conv.conv_w" in holder["names"][1]) == sp["conv"])
        rows.append((f"{arm}: groups {holder['names'][0]} / {len(holder['names'][1])} rest parameters; lrs at 1 {lrs[0]}, {W} "
                     f"{lrs[W - 1]}, {W + 1} {lrs[W]}", ok))
    # the tie: the k = 2 pilot's record (S90, read at d9783bb) reproduces on this CPU through 2400 with the new path disabled
    # (S90's path), and the new path (k = 16) gives the same curve and the same k-invariant statistics
    d = json.loads(s89._git("show", "d9783bb:explore_out/F/s90_ctx_results.json"))
    for arm2, arm16 in (("O_DELTA_CTXPOOL", "TL16_DELTA_CTXPOOL"), ("O_HEBBCONV_CTXPOOL", "TL16_HEBBCONV_CTXPOOL")):
        key = [k for k in d["runs"] if k.startswith(f"{arm2}|3060|809fde1f5168|") and d["runs"][k].get("ok")][0]
        rec = d["runs"][key]
        want = [c for c in rec["curve"] if c[0] <= 2400]
        r2 = s90.run(dict(arm=arm2, seed=3060, iters=2400))
        so = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= 2400]
        sn = [s for s in r2["stats"] if s["step"] != "end" and s["step"] <= 2400]
        same = lambda a, b: all((b.get(f) == v) or (v != v and b.get(f) != b.get(f)) for f, v in a.items())
        ok2 = r2["curve"] == want and len(so) == len(sn) and all(same(a, b) for a, b in zip(so, sn))
        rows.append((f"the new path disabled: S90's path reproduces {key} (read at d9783bb) through 2400 bit for bit on this CPU "
                     f"({cpu}): curve {r2['curve']} vs {want}; statistics equal {ok2}", ok2))
        r16 = run(dict(arm=arm16, seed=3060, iters=2400))
        inv = ("margin", "sep_key_inv", "seq_separated_frac", "val_cos_inv_k")
        s16 = [s for s in r16["stats"] if s["step"] != "end" and s["step"] <= 2400]
        st_ok = len(s16) == len(so) and all(all(a.get(f) == b.get(f) for f in inv if f in b) for a, b in zip(s16, so))
        ok16 = r16["curve"] == want and st_ok
        rows.append((f"the new path: {arm16}|3060 through 2400 gives the k = 2 pilot's curve bit for bit ({r16['curve'] == want}) "
                     f"and its k-invariant statistics ({', '.join(inv)}) where recorded ({st_ok})", ok16))
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
    m = make_model(sp, arm_def(sp), 0)
    t0 = time.time()
    tbo.evaluate(m, TASKS[sp["task"]], tbo.eval_batch(TASKS[sp["task"]]))
    return [dd[len(dd) // 2], 2 * (time.time() - t0)]
