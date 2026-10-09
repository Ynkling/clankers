#!/usr/bin/env python
"""
explore_scale_child.py — EXPLORATORY, not a result. Batch 19's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e), for all four screens (S64 explore_scale_width, S65 explore_scale_depth,
S66 explore_scale_eight, S67 explore_capacity). Model-side helpers: explore_b19_model. Recipes: explore_b16_main.

AN ARM is a spec from the parent:
  cfg      "k16": test_stream_recipe.run_attempt with ARM["A4k16"] (S=4, k=16, conv 'layer' width 4; the path of batch 16's
                  LOCAL3_SLOW16_A), on the spec's task: "P4S4" (P=4, the recorded one), "P8S4" / "P16S4" (S67:
                  test_channel_binding.task_for(8, 4) / (16, 4), registered in test_stream_recipe.TASKS in this process;
                  n_vals 16, n_q 1, grouped layout);
           "b":   test_stream_curriculum.run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams from step 1; the path
                  of batches 16-18's LOCAL3_SLOW_D8_A / W_SPLIT).
  size     [mult, D, n_layer]: N = mult x D (explore_b19_model.builder_of).
  gate     "local":   LOCAL3 + SLOW (explore_b16_main.make_rc "adam" with GATE_LOCAL: W_in, W_g, window at 1e-3 throughout;
                      every other trainable parameter 1e-4 for updates 1-2400, 1e-3 after; W_h frozen, unused; no hinge);
           "perfect": the oracle: the perfect gate (stream s -> channel s, channels S..15 unused), test_stream_recipe's
                      ceiling4k16 at this size and task, on the recorded validity recipe (the run path's single Adam at
                      1e-3, no groups, no switch);
           "resgate": RES6 (explore_b19_model.ResGateBDH, Session G's 4.3) with the SLOW recipe, the gate group W_in, W_g,
                      window, A.
  trigger  S66: + S37's KEYMASS trigger as run in batches 17-18 (explore_window_d8_child.make_trigger: checks at 2400
           (reference), 4800, 7200, ... through iters - 2400 on S36's 64-sequence probe; fire if accuracy < 0.95 and it rose
           < 0.02; <= 3 splits, >= 4800 apart; S24's split of W_g's row c* onto c0, W_g's Adam state zeroed).
Every run checkpoints after each evaluation and resumes from its checkpoint (explore_b19_model, CHECKed bit for bit here).
MEASURED: the run path's statistics (routing_k's map and multichannel eta^2 at every evaluation, per-stream accuracy at
the end), the trigger's rows, the initial parameter scales, the final weights' SHA-1, RES6's per-layer routing at the end.
"""

import contextlib
import hashlib

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import explore_b19_model as b19
import test_binding_onset as tbo
import test_multilayer_binding as tmb
import test_scale_axes as tsa
import test_stream_curriculum as tscur
import test_stream_recipe as tsr

for _k, _P in (("P8S4", 8), ("P16S4", 16)):
    tsr.TASKS.setdefault(_k, tsr.tcb.task_for(_P, 4))

EVERY = s48c.EVERY
MAX_SPLITS = 3
WRAP = ((tbo, "evaluate", True), (s37c, "read_gate", False), (s37c, "select", True), (tsa, "routing_k", True),
        (s36c, "split_op", True))


def base_arm(spec):
    if spec["cfg"] == "b":
        assert spec["gate"] != "perfect" and spec.get("task", "P4S8") == "P4S8"
        return dict(tscur.ARM["D8"])
    src = tsr.ARM["ceiling4k16"] if spec["gate"] == "perfect" else tsr.ARM["A4k16"]
    return dict(src, task=spec.get("task", "P4S4"))


def task_of(spec):
    return tsr.TASK48 if spec["cfg"] == "b" else tsr.TASKS[spec.get("task", "P4S4")]


def convert_of(spec):
    return {"local": b19.to_local_any, "resgate": b19.to_resgate, "perfect": None}[spec["gate"]]


def build_rc(spec, p, holder, info):
    mult, d, L = spec["size"]
    task = task_of(spec)
    conv = convert_of(spec)

    def on_build(m):
        holder["model"] = m
        holder.setdefault("init", b19.init_scales(m))

    if spec["gate"] == "perfect":
        rc0, _ = b16.make_rc("adam", holder, (), tau=None, on_build=on_build, keep=None, groups=False)
    else:
        gate = b19.GATE_RES if spec["gate"] == "resgate" else b16.GATE_LOCAL
        rc0, _ = b16.make_rc("adam", holder, gate, tau=None, convert=(conv,), on_build=on_build, keep=None)

    def rc(kw):
        kw = dict(kw)
        kw["builder"] = b19.builder_of(mult, d, L)
        return rc0(kw)
    ctx = contextlib.nullcontext()
    if spec.get("trigger"):
        ctx = s36c.adam_capture(holder)
        rc = s36c.with_trigger(rc, s48c.make_trigger(task, p["seed"], info, lambda: holder["opts"][0],
                                                     p.get("total", p["iters"]) - EVERY, max_splits=MAX_SPLITS,
                                                     thr=p.get("thr", s48c.ACC_THR), rise=p.get("rise", s48c.RISE)))
    rc_last = rc

    def rc_wrapped(kw):
        return b19.wrap_kw(rc_last(kw))
    return rc_wrapped, ctx


def w_sha(m):
    h = hashlib.sha1()
    for n, t in sorted(m.state_dict().items()):
        h.update(n.encode())
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(p):
    """p: spec, seed, iters, ckpt (path or None), halt_at (CHECK), eval_every (timing), thr / rise / total (CHECK)."""
    spec = p["spec"]
    b19.install(WRAP)
    b19.REC.reset()
    b19.CK.update(path=p.get("ckpt"), halt_at=p.get("halt_at"), resumed=[], saved=0)
    holder, info = {}, dict(checks=[])
    rc, ctx = build_rc(spec, p, holder, info)
    a = base_arm(spec)
    try:
        with ctx:
            rec = b16.path(spec["cfg"], a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    except b19.Halt as h:
        return dict(halted=True, step=int(h.args[0]), saved=b19.CK["saved"], resumed=list(b19.CK["resumed"]))
    m = holder.get("model")
    extra = dict(spec=spec, cfg=spec["cfg"], kind=spec["gate"], size=list(spec["size"]), task_key=spec.get("task", "P4S4"),
                 groups=b16.group_names(holder), init=holder.get("init"), checks=info["checks"],
                 splits=sum(1 for c in info["checks"] if c["fired"]), max_splits=MAX_SPLITS if spec.get("trigger") else None,
                 resumed=list(b19.CK["resumed"]), ckpt_saved=b19.CK["saved"], w_sha=w_sha(m) if m is not None else None,
                 n_embd=m.config.n_embd if m is not None else None, n_feat=m.n_feat if m is not None else None,
                 n_layer=m.config.n_layer if m is not None else None)
    if spec["gate"] == "resgate" and m is not None and rec.get("ok"):
        task = task_of(spec)
        lm = b19.layer_maps(m, task, tmb.probe_batch(task, p["seed"]), b19.orig(tsa.routing_k))
        extra["layer_maps"] = [{k: v for k, v in x.items() if k in ("ch_map", "one_to_one", "shared", "etak_val_by_stream",
                                                                    "etak_val_by_key", "etak_key_by_stream", "etak_key_by_key")}
                               for x in lm]
    out = b16.finish(rec, **extra)
    if spec["gate"] == "perfect" and out.get("ok"):
        out["outcome"] = "BOUND" if out.get("transition") is not None else "not bound"
    if p.get("ckpt") and out.get("ok"):
        import os
        for f in (p["ckpt"], p["ckpt"] + ".tmp", p["ckpt"] + ".step"):
            if os.path.exists(f):
                os.remove(f)
    return out


# ── CHECKs ───────────────────────────────────────────────────────────────────
SIZES_ALL = [(8, 32, 3), (16, 32, 3), (32, 32, 3), (4, 64, 3), (2, 128, 3), (8, 32, 2), (8, 32, 4), (8, 32, 6)]


def checks(p):
    rows = []
    seed = 270
    # (1) the sized builder at N=256, D=32, 3 layers is the run path's builder, bit for bit
    for cfg, spec_task, gate in (("k16", "P4S4", "local"), ("k16", "P4S4", "perfect"), ("b", "P4S8", "local"),
                                 ("k16", "P8S4", "local"), ("k16", "P16S4", "perfect")):
        spec = dict(cfg=cfg, task=spec_task, gate=gate, size=[8, 32, 3])
        a = base_arm(spec)
        m0 = tsr.builder_for(a)(a, seed)()
        m1 = b19.builder_of(8, 32, 3)(a, seed)()
        s0, s1 = m0.state_dict(), m1.state_dict()
        same = list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)
        rows.append((f"{cfg} {spec_task} {gate}: the sized builder at N=256, D=32, 3 layers equals test_stream_recipe.builder_for "
                     f"bit for bit ({len(s0)} tensors; task S={task_of(spec).S} P={task_of(spec).P}, length "
                     f"{task_of(spec).make_batch(1, torch.Generator().manual_seed(0))[0].shape[1]})", same))
    # (2) every size: shapes, N = mult x D, depth, LOCAL3 at D (the D=32 conversion is batch 16's object; the window's first
    #     32 rows are the D=32 window), D restored after construction, one training step on the real task through the
    #     k16 and b statistics
    rows.append(("to_local_any at D=32 is explore_b16_gates.to_local's conversion (same window, same class)",
                 torch.equal(b19.window_init_d(seed, 32), g16.window_init(seed))))
    for mult, d, L in SIZES_ALL:
        for spec_task, cfg in (("P4S4", "k16"), ("P4S8", "b")):
            spec = dict(cfg=cfg, task=spec_task, gate="local", size=[mult, d, L])
            a = base_arm(spec)
            m = b19.to_local_any(b19.builder_of(mult, d, L)(a, seed)(), seed)
            ok = (m.config.n_embd == d and m.n_feat == mult * d and m.config.n_layer == L and tuple(m.W_in.shape) == (32, d)
                  and tuple(m.gate_conv.conv_w.shape) == (d, 3) and tuple(m.short_conv.conv_w.shape) == (d, 4)
                  and torch.equal(m.gate_conv.conv_w[:32], g16.window_init(seed)) and tmb.D == 32 and type(m) is g16.LocalBDH
                  and not m.W_h.requires_grad)
            task = task_of(spec)
            x = task.make_batch(4, torch.Generator().manual_seed(5))[0]
            ql, qt, _ = task.select(tbo.logits_of(m, x[:, :-1]), x[:, 1:], None)
            loss = torch.nn.functional.cross_entropy(ql, qt)
            loss.backward()
            st = tsa.scale_stats(m, task, tmb.probe_batch(task, seed), 0)
            ok &= bool(torch.isfinite(loss)) and len(st["ch_map"]) == task.S and all(
                p_.grad is not None for n_, p_ in m.named_parameters() if p_.requires_grad)
            if cfg == "k16" or (mult, d, L) in ((8, 32, 3), (32, 32, 3), (8, 32, 6), (4, 64, 3)):
                rows.append((f"size N={mult * d} (mult {mult}) D={d} L={L} on {cfg}: n_embd {m.config.n_embd}, N {m.n_feat}, layers "
                             f"{m.config.n_layer}, W_in {tuple(m.W_in.shape)}, window {tuple(m.gate_conv.conv_w.shape)} (first 32 "
                             f"rows = the D=32 window), stack conv {tuple(m.short_conv.conv_w.shape)}; one backward on the task, "
                             f"statistics map {st['ch_map']}; test_multilayer_binding.D restored ({tmb.D})", ok))
    # (3) S67's tasks and their oracle at N=256 and N=1024
    for tk in ("P8S4", "P16S4"):
        task = tsr.TASKS[tk]
        for mult in (8, 32):
            spec = dict(cfg="k16", task=tk, gate="perfect", size=[mult, 32, 3])
            a = base_arm(spec)
            m = b19.builder_of(mult, 32, 3)(a, seed)()
            x = task.make_batch(2, torch.Generator().manual_seed(4))[0][:, :-1]
            with torch.no_grad():
                gr = m(x)[2]
            ctx = x[:, 0]
            ok = (m.gate_kind == "perfect" and m.n_ch == 16 and gr.shape[-1] == 16 and float(gr[..., task.S:].abs().max()) == 0.0
                  and torch.equal(gr[torch.arange(2), 0].argmax(-1), ctx))
            rows.append((f"{tk} (S={task.S}, P={task.P}, vocab {task.vocab}, length {x.shape[1] + 1}) oracle at N={mult * 32}: perfect "
                         f"gate k=16, channels {task.S}..15 at 0, the first block on its stream's channel", ok))
    # (4) RES6: A = [0 | W_in] at init, so every layer's gate at step 0 is softmax(W_g tanh(W_in v)); layer 1 is LOCAL3's;
    #     forward_res with layer 1's gate at every layer equals LocalBDH's forward (the forward_conv copy)
    spec = dict(cfg="k16", task="P4S4", gate="resgate", size=[8, 32, 6])
    a = base_arm(spec)
    mr = b19.to_resgate(b19.builder_of(8, 32, 6)(a, seed)(), seed)
    ml = b19.to_local_any(b19.builder_of(8, 32, 6)(a, seed)(), seed)
    task = task_of(spec)
    x = task.make_batch(3, torch.Generator().manual_seed(6))[0][:, :-1]
    mr.eval(), ml.eval()
    with torch.no_grad():
        _, _, g1r, _ = mr(x)
        ll_, _, g1l, _ = ml(x)
        v = mr.embed(x)
        want = torch.softmax(torch.tanh(v @ mr.W_in.T) @ mr.W_g.T, -1)
        lay = list(mr._layer_gates)
        d_init = max(float((g - want).abs().max()) for g in lay[1:])
        mr.force_g1 = True
        try:
            lf = mr(x)[0]
        finally:
            mr.force_g1 = False
    names = [n for n, _ in mr.named_parameters()]
    rows.append((f"RES6 (L=6): A {tuple(mr.A.shape)} = [0 | W_in] at init (a parameter: {'A' in names}); layer 1's gate equals "
                 f"LOCAL3's ({torch.equal(g1r, g1l)}); layers 2-6 at init equal softmax(W_g tanh(W_in v)) (max |diff| {d_init:.1e}); "
                 f"{len(lay)} layer gates", torch.equal(g1r, g1l) and d_init <= 1e-6 and len(lay) == 6 and "A" in names
                 and torch.equal(mr.A.detach()[:, :32], torch.zeros(32, 32)) and torch.equal(mr.A.detach()[:, 32:], mr.W_in.detach())))
    df = float((lf - ll_).abs().max())
    rows.append((f"RES6's forward_res with every layer's gate forced to layer 1's equals LocalBDH's forward (max |logit diff| "
                 f"{df:.1e} == 0)", df == 0.0))
    # (5) the chunked evaluation forward equals the one-forward logits bit for bit (and evaluate's accuracy and loss)
    from test_instrument_v2 import TAU_END
    for tk, mult, cfg in (("P4S4", 32, "k16"), ("P16S4", 8, "k16"), ("P4S8", 8, "b"), ("P8S4", 8, "k16")):
        spec = dict(cfg=cfg, task=tk, gate="local", size=[mult, 32, 3])
        a = base_arm(spec)
        m = b19.to_local_any(b19.builder_of(mult, 32, 3)(a, seed)(), seed)
        task = task_of(spec)
        data = tbo.eval_batch(task)[:1024]
        m.eval()
        with torch.no_grad():
            one = b19.ORIG_LOGITS(m, data[:, :-1])
            ch = b19.logits_chunked(m, data[:, :-1])
        m.train()
        old = tbo.logits_of
        tbo.logits_of = b19.ORIG_LOGITS
        ev_one = b19.orig(tbo.evaluate)(m, task, data)
        tbo.logits_of = b19.logits_chunked
        ev_ch = b19.orig(tbo.evaluate)(m, task, data)
        tbo.logits_of = old
        rows.append((f"{cfg} {tk} N={mult * 32}: the evaluation forward in chunks of {b19.CHUNK} equals the one forward on 1024 held-out "
                     f"sequences bit for bit (logits {torch.equal(one, ch)}; evaluate {ev_one} vs {ev_ch})",
                     torch.equal(one, ch) and ev_one == ev_ch))
    # (6) groups and lrs through the real run path at D=64 and for RES6 (forward and evaluation stubbed), 2401 updates
    W = b16.WARM
    tbo.onset_run = b19.ORIG_ONSET                                  # stub_lrs needs run_one's keep (no checkpoints)
    for spec in (dict(cfg="k16", task="P4S4", gate="local", size=[4, 64, 3]), dict(cfg="k16", task="P4S4", gate="resgate", size=[8, 32, 4]),
                 dict(cfg="k16", task="P4S4", gate="perfect", size=[16, 32, 3])):
        h, rec, lg, ids, keep = b16.stub_lrs("k16", base_arm(spec), lambda hd, kp, sp=spec: _stub_rc(sp, hd, kp))
        mm = keep["model"]
        names = [g["names"] for g in h.get("groups", [])]
        if spec["gate"] == "perfect":
            ok = len(lg) == W + 1 and all(x == [b16.LR] for x in lg) and rec.get("ok") and mm.n_feat == 512
            desc = f"one group at {lg[0]} throughout"
        else:
            gate = list(b19.GATE_RES if spec["gate"] == "resgate" else b16.GATE_LOCAL)
            ok = (b16.cover(ids, mm) and names[0] == gate and len(names) == 2 and len(lg) == W + 1 and lg[0] == [b16.LR, b16.LR_WARM]
                  and lg[W - 1] == [b16.LR, b16.LR_WARM] and lg[W] == [b16.LR, b16.LR] and rec.get("ok"))
            desc = (f"groups {[(g['tag'], len(g['names'])) for g in h['groups']]} cover every trainable parameter once, gate {names[0]}; "
                    f"lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}")
        rows.append((f"{spec['gate']} at N={spec['size'][0] * spec['size'][1]} D={spec['size'][1]} L={spec['size'][2]}: {desc}", ok))
    return [[n, bool(v)] for n, v in rows]


def _stub_rc(spec, holder, keep):
    """build_rc's recipe without the checkpoint wrappers, for stub_lrs (keep needs run_one's post_step)."""
    mult, d, L = spec["size"]
    if spec["gate"] == "perfect":
        rc0, _ = b16.make_rc("adam", holder, (), tau=None, keep=keep, groups=False)
    else:
        gate = b19.GATE_RES if spec["gate"] == "resgate" else b16.GATE_LOCAL
        rc0, _ = b16.make_rc("adam", holder, gate, tau=None, convert=(convert_of(spec),), keep=keep)

    def rc(kw):
        kw = dict(kw)
        kw["builder"] = b19.builder_of(mult, d, L)
        return rc0(kw)
    return rc


def resume_check(p):
    """A run halted after its checkpoint at halt_at and resumed, against the same run uninterrupted (both from scratch)."""
    import os
    import tempfile
    base = dict(spec=p["spec"], seed=p["seed"], iters=p["iters"])
    for k in ("thr", "rise", "total"):
        if k in p:
            base[k] = p[k]
    full = run(dict(base, ckpt=None))
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "ck.pt")
        h = run(dict(base, ckpt=ck, halt_at=p["halt_at"]))
        r = run(dict(base, ckpt=ck))
    keys = ("curve", "stats", "grad", "end", "transition", "stopped_at", "early", "acc_check", "passed", "checks", "splits",
            "w_sha", "outcome", "init", "layer_maps")
    diff = [k for k in keys if full.get(k) != r.get(k)]
    return dict(halted=h, resumed=r.get("resumed"), saved=r.get("ckpt_saved"), diff=diff, n_keys=len(keys),
                splits_full=[c["step"] for c in full.get("checks") or [] if c.get("fired")],
                splits_res=[c["step"] for c in r.get("checks") or [] if c.get("fired")], ok=bool(full.get("ok") and r.get("ok")),
                curve=full.get("curve"))


def timing(p):
    """{name: [step s, evaluation s (+ trigger check s for trigger arms)]} under the caller's pool load."""
    import time
    out = {}
    for name, spec in p["which"].items():
        st = b16.step_timing(lambda: run(dict(spec=spec, seed=0, iters=p["steps"], eval_every=10 ** 9, ckpt=None)), True)
        mult, d, L = spec["size"]
        a = base_arm(spec)
        m = b19.builder_of(mult, d, L)(a, 0)()
        conv = convert_of(spec)
        if conv is not None:
            m = conv(m, 0)
        task = task_of(spec)
        from test_channel_binding import eval_batch
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        tsa.scale_stats(m, task, tmb.probe_batch(task, 0), 1200)
        te = time.time() - t0
        if spec.get("trigger"):
            data = s36c.probe_data(task)
            t0 = time.time()
            tbo.evaluate(m, task, data)
            s37c.select(s37c.read_gate(m, data), task.S, task.P)
            tsa.routing_k(m, task, tmb.probe_batch(task, 0))
            te += (time.time() - t0) / 2                                       # a check every 2400 = every other evaluation
        out[name] = [st, te]
    return out


def constants(p):
    return dict(main_sha=mt.MAIN_SHA, LR=b16.LR, LR_WARM=b16.LR_WARM, WARM=b16.WARM, EVERY=EVERY, FIRST=s48c.FIRST,
                REF_AT=s48c.REF_AT, ACC_THR=s48c.ACC_THR, RISE=s48c.RISE, GAP=s48c.GAP, MAX_SPLITS=MAX_SPLITS, WIDTH=g16.WIDTH,
                EVAL_EVERY=tbo.EVAL_EVERY, BIND_PASS=tbo.BIND_PASS, STOP_AFTER=tbo.STOP_AFTER, tsr_LR=tsr.LR, tscur_LR=tscur.LR,
                T_CHECK=tsr.T_CHECK, ARCH=dict(b19.trd.ARCH), D=tmb.D, MULT_A4k16=tsr.ARM["A4k16"]["mult"],
                H_GATE=tsr.ARM["A4k16"].get("h_gate"), conv=tsr.ARM["A4k16"].get("model_kw"),
                tasks={k: dict(S=t.S, P=t.P, vocab=t.vocab, n_vals=t.n_vals, n_q=t.n_q) for k, t in tsr.TASKS.items()},
                task48=dict(S=tsr.TASK48.S, P=tsr.TASK48.P))


def inits(p):
    """Initial parameter scales per size (one seed), for the report."""
    out = {}
    for name, spec in p["which"].items():
        mult, d, L = spec["size"]
        a = base_arm(spec)
        m = b19.builder_of(mult, d, L)(a, p["seed"])()
        conv = convert_of(spec)
        if conv is not None:
            m = conv(m, p["seed"])
        out[name] = b19.init_scales(m)
    return out
