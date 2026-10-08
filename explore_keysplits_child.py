#!/usr/bin/env python
"""
explore_keysplits_child.py — EXPLORATORY, not a result. S52's run code. It runs INSIDE a child process on this branch's own
modules (no main-line hook), as S43's and S49's. Parent: explore_keysplits.

THE RUN PATH: S43's LOCAL3_SLOW (explore_window_recipe_child: batch 1's LocalGateBDH with W_h frozen, S5's SLOW schedule —
gate W_in, W_g, window at 1e-3 throughout, every other trainable parameter 1e-4 for updates 1-2400 and 1e-3 after —
test_short_conv's TASK (S=2, P=4, grouped, no convolution), 24000 updates, S38's decoder at every evaluation), with:
  W2_K4        k = 4 (the arm's n_ch; two spare channels). Statistics: explore_common2.stats_b2 + routing_k at every
               evaluation + stream_acc at the end, so the outcome is test_stream_recipe.outcome's (BOUND ROUTED: bound, and
               one-to-one map + every stream's accuracy >= 0.9). torch.manual_seed(seed) then the constructor, as every
               arm: W_g has 4 rows, so the parameters drawn after it differ from LOCAL3_SLOW's; the window and the batches
               are the seed's.
  W2_KEYHINGE  k = 2 + the hinge's KEY term only, on the read gate at the body's key positions:
               LAM x relu(eta2_key - TAU), LAM 1.0, TAU 0.2, weight LAM on updates 1-2400 and 0 after (WINDOW, the
               key-position analogue of S34's window). eta2_key = explore_b18_gates.eta2_key (test_slow_start.eta2_hinge's
               pooled computation grouped by the key id). It enters the loss through explore_aux_gate._Inject from a forward
               hook on training forwards (main's attach_hinge), counted per update; after the window the hook does nothing.
               Statistics: stats_b2 (S43's).
Both: the gate measured after updates 0, 25, ..., 1200, 1300, ..., 2400, then every 1200 to the end (cells: routing_k's
pooled eta^2 at KEY and VAL positions, routing_stats' margin and channel-0 eta^2, the map), on the run's probe, in an
optimizer step post-hook (no gradient, eval mode: training untouched).

COPIED FROM MAIN at 9c5939e, line for line (CHECK: their syntax trees equal main's): test_scale_axes.eta2_multi, routing_k,
fail_class_k, stream_acc; test_stream_recipe.routed, outcome (ROUTED_ACC 0.9);
test_stream_channels.shared_max, merged. The branch has no test_scale_axes; these are pure computations on the model's
gates, and every name they use is a module both trees share unchanged.
"""

import ast
import json

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_window_recipe_child as s43c
import explore_local_gate as s1
import explore_aux_gate as aux
import explore_b16_gates as g16
import explore_b18_gates as g18
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_layout import routing_stats, fail_class
from test_router_reliability import run_one
from test_multilayer_binding import probe_batch, MultiBDH, arms_for
from test_channel_binding import eval_batch, ARCH
from test_instrument_v2 import TAU_END
from test_binding_onset import EVAL_EVERY

LAM, TAU, WINDOW = 1.0, 0.2, 2400
ROUTED_ACC = 0.9
ARM = {"W2_K4": dict(k=4), "W2_KEYHINGE": dict(k=2, hinge=True)}
DENSE = tuple(range(25, 1201, 25)) + tuple(range(1300, 2401, 100))
W_LOG_AT = (1, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2)
COPIED = {"eta2_multi": "test_scale_axes.py", "routing_k": "test_scale_axes.py", "fail_class_k": "test_scale_axes.py",
          "stream_acc": "test_scale_axes.py", "routed": "test_stream_recipe.py", "outcome": "test_stream_recipe.py",
          "shared_max": "test_stream_channels.py", "merged": "test_stream_channels.py"}


# ── Copied from main at 9c5939e (CHECK: syntax trees equal) ──────────────────
def eta2_multi(x, lab):
    """eta^2 of a vector-valued x (..., k) grouped by lab: between-group over total sum of
    squares, both summed over the k channels; 0 if x is constant."""
    x, lab = x.reshape(-1, x.shape[-1]).double(), lab.reshape(-1)
    mu = x.mean(0)
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    between = sum((lab == g).sum() * ((x[lab == g].mean(0) - mu) ** 2).sum() for g in lab.unique())
    return float(between / tot)


@torch.no_grad()
def routing_k(model, task, probe):
    """The multichannel eta^2 of the read gate over the body's key and value positions (groups
    stream / key / block / index), and the stream -> channel map at value positions."""
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    S, P, n = task.S, task.P, task.S * task.P
    B = pinp.shape[0]
    j = torch.arange(n)
    ctx, key = pinp[:, 3 * j], pinp[:, 3 * j + 1] - S
    labels = dict(stream=ctx, key=key, block=(j // P).expand(B, n), index=j.expand(B, n))
    out = {}
    for role, off in (("key", 1), ("val", 2)):
        for name, lab in labels.items():
            out[f"etak_{role}_by_{name}"] = eta2_multi(gr[:, 3 * j + off], lab)
    gv = gr[:, 3 * j + 2]
    means = [gv[ctx == s].mean(0) for s in range(S)]
    cmap = [int(m_.argmax()) for m_ in means]
    out["ch_map"] = cmap
    out["one_to_one"] = len(set(cmap)) == S
    out["shared"] = sum(1 for c in cmap if cmap.count(c) > 1)
    out["stream_gate"] = [[round(v, 4) for v in m_.tolist()] for m_ in means]
    return out


def fail_class_k(r):
    """test_router_layout.fail_class on the multichannel eta^2 (block for half)."""
    e = r["end"]
    return fail_class({"end": dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                                   eta_key_by_half=e["etak_key_by_block"],
                                   eta_key_by_index=e["etak_key_by_index"])})


@torch.no_grad()
def stream_acc(model, task, data):
    """Held-out accuracy per queried stream (tbo.evaluate's computation, split by stream)."""
    was = model.training
    model.eval()
    try:
        ql, qt, _ = task.select(tbo.logits_of(model, data[:, :-1]), data[:, 1:], None)
    finally:
        model.train(was)
    qs = data[:, task.qpos[0] - 1]
    hit = (ql.argmax(-1) == qt).float()
    return [hit[qs == s].mean().item() for s in range(task.S)]


def routed(r):
    e = r["end"]
    return bool(e["one_to_one"] and min(e["stream_acc"]) >= ROUTED_ACC)


def outcome(r):
    if r["transition"] is not None:
        return "BOUND ROUTED" if routed(r) else "BOUND NOT routed"
    fc = fail_class_k(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if merged(r) else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def shared_max(r):
    cm = r["end"]["ch_map"]
    return max(cm.count(c) for c in cm)


def merged(r):
    return shared_max(r) >= 2


# ── This screen ──────────────────────────────────────────────────────────────
def stats_k(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out.update(routing_k(model, task, probe))
    if step == "end":
        out["stream_acc"] = stream_acc(model, task, eval_batch(task))
    return out


def at_steps(iters):
    return sorted(set(t for t in DENSE if t <= iters) | set(range(1200, iters + 1, 1200)))


@torch.no_grad()
def cells(m, task, probe):
    rk = routing_k(m, task, probe)
    rs = routing_stats(m, task, probe)
    out = {k: round(rk[k], 5) for k in ("etak_key_by_key", "etak_key_by_stream", "etak_key_by_block", "etak_key_by_index",
                                         "etak_val_by_stream", "etak_val_by_key")}
    out.update(margin=round(rs["margin"], 5), eta0_key_by_key=round(rs["eta_key_by_key"], 5), ch_map=rk["ch_map"])
    return out


def attach_key_hinge(m, S, P, tau, lam, window, state):
    state.update(n=0, fired=0, windows=[], weights={}, last=None, eta_at={})

    def hook(mod, args, out):
        if not (mod.training and torch.is_grad_enabled()):
            return None
        logits, sat, gr, gw = out
        state["n"] += 1
        u = state["n"]
        w = lam if u <= window else 0.0
        if u in W_LOG_AT:
            state["weights"][str(u)] = w
        if w == 0.0:
            return None
        e = g18.eta2_key(gr, args[0], S, P)
        h = F.relu(e - tau)
        state["last"] = float(e.detach())
        if u in (1, 300, 600, 1200, 2400):
            state["eta_at"][str(u)] = round(float(e.detach()), 5)
        win = (u - 1) // EVAL_EVERY
        while len(state["windows"]) <= win:
            state["windows"].append(0)
        if bool(h > 0):
            state["fired"] += 1
            state["windows"][win] += 1
        return aux._Inject.apply(logits, w * h), sat, gr, gw
    return m.register_forward_hook(hook)


def run(p):
    """p: arm, seed, iters, window (CHECK), dense (default True), eval_every (timing), keep (CHECK)."""
    arm = ARM[p["arm"]]
    seed, iters = p["seed"], p["iters"]
    task = ec.TASK
    holder, out, dense, box, hs = {}, {}, {}, {}, {}
    data = g16.probe_data(task)
    probe = probe_batch(task, seed)
    a = dict(s43c.A, n_ch=4) if arm["k"] == 4 else s43c.A
    window = p.get("window", WINDOW)
    want = set(at_steps(iters)) if p.get("dense", True) else set()
    b0 = s43c.builder(True)

    def builder(aa, s):
        mk = b0(aa, s)

        def make():
            m = mk()
            if arm.get("hinge"):
                attach_key_hinge(m, task.S, task.P, TAU, LAM, window, hs)
            box["m"] = m
            if want:
                dense["0"] = cells(m, task, probe)
            return m
        return make
    n = [0]

    def post(opt, args, kwargs):
        n[0] += 1
        if n[0] in want:
            dense[str(n[0])] = cells(box["m"], task, probe)
    h = register_optimizer_step_post_hook(post)
    try:
        kw = dict(check=s43c.make_check(True, holder, out, data), task=task, grad_fn=tsc.conv_grad_norms, lr=s43c.LR,
                  builder=builder, stats_fn=stats_k if arm["k"] > 2 else c2.stats_b2,
                  run_kw=dict(param_groups=s43c.groups(holder)))
        if p.get("keep") is not None:
            kw["keep"] = p["keep"]
        rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), **kw)
    finally:
        h.remove()
    if rec.get("ok"):
        rec.update(arm=p["arm"], decode=out, dense=dense, n_updates=n[0], group_names=holder.get("names"),
                   hinge=(dict(window=window, lam=LAM, tau=TAU, fired=hs.get("fired"), n=hs.get("n"), windows=hs.get("windows"),
                               weights=hs.get("weights"), eta_at=hs.get("eta_at")) if arm.get("hinge") else None))
        if arm["k"] > 2:
            rec["outcome"] = outcome(rec)
            rec["fail"] = None if rec["outcome"].startswith("BOUND") else fail_class_k(rec)
        else:
            rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
            rec["tag"] = ec.tag(rec)
    return json.loads(json.dumps(rec, default=float))


def perfect_k4(p):
    """The perfect gate with k = p["k"] (default 4): test_instrument_v2.perfect_gate_general with the context tokens padded
    by k - 2 ids that never occur, so streams 0, 1 go to channels 0, 1 and the other channels stay empty; p["arm"]:
    "ceiling_conv" = test_short_conv.ARM["ceiling_conv"] (the perfect gate + conv "layer", mult 8: this layout's recorded
    validity arm, 5/5 at 1200) or "ceiling" = test_multilayer_binding's plain perfect gate (no convolution, mult 2); the
    perfect gate's plain recipe (one Adam at 1e-3); W2_K4's statistics."""
    task = ec.TASK
    k = p.get("k", 4)
    base = tsc.ARM["ceiling_conv"] if p.get("arm", "ceiling_conv") == "ceiling_conv" else \
        next(a for a in arms_for(task) if a["key"] == "ceiling")
    ceil = dict(base, n_ch=k)
    pad = tuple(-1 - i for i in range(k - task.S))

    def builder(aa, s):
        def make():
            torch.manual_seed(s)
            return MultiBDH(task.vocab, ARCH["n_layer"], aa["gate"], aa["n_ch"], ARCH["positional"], gate_to_readout=aa["g2r"],
                            h_gate=aa["h_gate"], mult=aa["mult"], ctx_tokens=tuple(task.ctx_tokens) + pad,
                            **aa.get("model_kw", {}))
        return make
    out = {}
    for s in p["seeds"]:
        rec = run_one(ceil, s, p["iters"], tbo.EVAL_EVERY, task=task, grad_fn=tsc.conv_grad_norms, lr=s43c.LR, builder=builder,
                      stats_fn=stats_k)
        rec = json.loads(json.dumps(rec, default=float))
        out[str(s)] = dict(ok=rec.get("ok"), transition=rec.get("transition"), curve=rec.get("curve"),
                           outcome=outcome(rec) if rec.get("ok") else rec.get("error"), ch_map=rec["end"].get("ch_map"),
                           stream_acc=rec["end"].get("stream_acc"), k=rec.get("k"),
                           gate_at_val=rec["end"].get("stream_gate"))
    return out


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _defs(src):
    return {n.name: n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def copied_check():
    import explore_main9c as mt
    with open(__file__) as f:
        mine = _defs(f.read())
    rows = {}
    for name, path in COPIED.items():
        theirs = _defs(mt.main_blob(path))
        rows[name] = name in theirs and ast.dump(theirs[name]) == ast.dump(mine[name])
    main_acc = [n for n in ast.parse(mt.main_blob("test_stream_recipe.py")).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "ROUTED_ACC" for t in n.targets)]
    rows["ROUTED_ACC"] = bool(main_acc) and ast.literal_eval(main_acc[0].value) == ROUTED_ACC
    return rows


def checks(p):
    rows = []
    task = ec.TASK
    S, P = task.S, task.P
    rr = copied_check()
    rows.append((f"routing_k, eta2_multi, fail_class_k, stream_acc, routed, outcome, shared_max, merged and ROUTED_ACC are main's "
                 f"at 9c5939e line for line (syntax trees equal: {rr})", all(rr.values())))
    # pooled eta^2 at k = 2 equals routing_stats' channel-0 eta^2; eta2_key equals S19's key_penalty and eta_key_by_key
    import explore_token_hinge as s19
    seed = 160
    m = g16.freeze_wh(s1.make_model(s43c.A, seed))
    pr = probe_batch(task, seed)
    worst = 0.0
    for scale in (1.0, 20.0):
        with torch.no_grad():
            m.W_g.mul_(scale)
        rk, rs = routing_k(m, task, pr), routing_stats(m, task, pr)
        m.eval()
        with torch.no_grad():
            gr = m(pr[0], TAU_END)[2]
        m.train()
        j = torch.arange(S * P)
        e18 = float(g18.eta2_key(gr, pr[0], S, P))
        e19 = float(s19.key_penalty(gr, pr[0], 3 * j + 1, S, P))
        worst = max(worst, abs(rk["etak_key_by_key"] - rs["eta_key_by_key"]), abs(rk["etak_key_by_stream"] - rs["eta_key_by_stream"]),
                    abs(rk["etak_key_by_block"] - rs["eta_key_by_half"]), abs(e18 - rs["eta_key_by_key"]), abs(e19 - e18))
        last = (round(rk["etak_key_by_key"], 6), round(rs["eta_key_by_key"], 6), round(e18, 6), round(e19, 6))
    rows.append((f"at k = 2 the pooled eta^2 (routing_k) equals routing_stats' channel-0 eta^2 by key, stream and half, and the "
                 f"hinge's eta2_key equals S19's key_penalty and eta_key_by_key, at init and with W_g x 20 (last: pooled / channel-0 / "
                 f"eta2_key / S19 = {last}; max |diff| {worst:.1e} <= 1e-4: float32, and eta2_key keeps test_slow_start's "
                 f"uncentred group means where S19's key_penalty centres them)", worst <= 1e-4 and last[0] > 0.01))
    # the key term's gradient reaches only the gate and the embedding
    m = g16.freeze_wh(s1.make_model(s43c.A, seed))
    x = task.make_batch(16, torch.Generator().manual_seed(3))[0][:, :-1]
    gr = m(x)[2]
    g18.eta2_key(gr, x, S, P).backward()
    hit = sorted(n for n, q in m.named_parameters() if q.grad is not None and float(q.grad.abs().sum()) > 0)
    rows.append((f"the key term's gradient reaches only {hit}", set(hit) <= {"W_in", "W_g", "gate_conv.conv_w", "embed.weight"}
                 and "W_g" in hit))
    # W2_K4's model
    m4 = g16.freeze_wh(s1.make_model(dict(s43c.A, n_ch=4), seed))
    rows.append((f"W2_K4's model: W_g {tuple(m4.W_g.shape)} (4 channels), the seed's window (equal to batch 1's: "
                 f"{torch.equal(m4.gate_conv.conv_w.detach(), g16.window_init(seed))}), W_h frozen ({not m4.W_h.requires_grad})",
                 m4.W_g.shape[0] == 4 and m4.n_ch == 4 and torch.equal(m4.gate_conv.conv_w.detach(), g16.window_init(seed))
                 and not m4.W_h.requires_grad))
    # the hinge's weight through the real run path, and its inertness at window 0
    r = run(dict(arm="W2_KEYHINGE", seed=seed, iters=WINDOW + 2, dense=False, eval_every=10 ** 9))
    wts = r["hinge"]["weights"]
    rows.append((f"W2_KEYHINGE's weight at updates {list(W_LOG_AT)}: {wts}; training forwards counted {r['hinge']['n']} = updates "
                 f"{r['n_updates']}; fired on {r['hinge']['fired']} of the first {WINDOW} updates",
                 wts == {"1": LAM, str(WINDOW - 1): LAM, str(WINDOW): LAM, str(WINDOW + 1): 0.0, str(WINDOW + 2): 0.0}
                 and r["hinge"]["n"] == r["n_updates"] == WINDOW + 2))
    r0 = run(dict(arm="W2_KEYHINGE", seed=seed, iters=1200, window=0, dense=False))
    rl = s43c.run(dict(arm="LOCAL3_SLOW", seed=seed, iters=1200))
    same = r0["curve"] == rl["curve"] and r0["stats"] == rl["stats"] and r0["decode"] == rl["decode"]
    rows.append((f"with weight 0 (window 0) W2_KEYHINGE|{seed} equals S43's LOCAL3_SLOW|{seed} bit for bit through 1200 (curve "
                 f"{r0['curve']}; statistics and decoder equal {same})", same))
    # the dense measurements are inert (W2_K4, 1200 updates with and without them)
    a1 = run(dict(arm="W2_K4", seed=seed, iters=1200))
    a0 = run(dict(arm="W2_K4", seed=seed, iters=1200, dense=False))
    same = a1["curve"] == a0["curve"] and a1["stats"] == a0["stats"] and a1["decode"] == a0["decode"]
    rows.append((f"W2_K4|{seed}: the dense measurements are inert through 1200 (curve {a1['curve']}; statistics and decoder equal "
                 f"{same}; {len(a1['dense'])} measurements; routing_k map at 1200 {a1['stats'][-1].get('ch_map')})",
                 same and len(a1["dense"]) == len(at_steps(1200)) + 1))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
    from torch.optim.optimizer import register_optimizer_step_post_hook as post_hook
    out = {}
    task = ec.TASK
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        stamps = []
        hh = post_hook(lambda o, a_, k_: stamps.append(time.perf_counter()))
        try:
            run(dict(arm=arm_name, seed=0, iters=p["steps"], eval_every=10 ** 9, dense=False))
        finally:
            hh.remove()
        dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        m = g16.freeze_wh(s1.make_model(dict(s43c.A, n_ch=arm["k"]), 0))
        pr = probe_batch(task, 0)
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        (stats_k if arm["k"] > 2 else c2.stats_b2)(m, task, pr, 2400)
        te = time.time() - t0
        t0 = time.time()
        g16.decode(m, task, g16.probe_data(task))
        td = time.time() - t0
        t0 = time.time()
        for _ in range(5):
            cells(m, task, pr)
        out[arm_name] = [dd[len(dd) // 2], te, td, (time.time() - t0) / 5]
    return out
