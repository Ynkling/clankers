"""
explore_gpu/run.py — EXPLORATORY tooling, not a result. The runner: tasks, arms, the sequential and the batched trainers,
evaluation, records. Device-agnostic (cpu or cuda), float32, the project's seeding conventions.

SEEDING (every path, the harness's): torch.manual_seed(seed) then the model is built on CPU by the project's builders
(test_multilayer_binding.build -> explore_b16_gates.to_local / explore_i_common.to_window / to_reg3 / nudge_reg /
explore_delta_mem.to_delta), so initial parameters do not depend on the device; the training batches come from a CPU
torch.Generator seeded seed + 10_000 (one per run), in the same order; evaluation and probe sets from their fixed CPU
generators; the split's noise from its CPU generator (12_000_000 + 1000 seed + split count). Only the arithmetic runs on
the device.

TASKS (task_from): "icmc:S=4,V=4,NB=16,lmin=6,lmax=14,M=176" (explore_i_task.ICMCTask; dense masked loss), "rhl"
(RandHeaderTask-lite: explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), Session F's S70 / Session G's S71-S73),
"grouped:P=4,S=2" (test_binding_capacity.BindTask, grouped, n_vals 16, n_q 1; "grouped" alone = test_short_conv.TASK).
ARMS (spec dicts; arm_spec): gate none / perfect / recurrent / win3 / wide / reg3, k, conv (None or "layer"), mem "hebb"
or "delta" (+ delta keyword arguments of explore_delta_mem.to_delta; not with reg3), recipe "adam" (Adam 1e-3
throughout) or "slow" (SLOW: gate group at 1e-3, the rest 1e-4 for updates 1-2400, 1e-3 after the evaluation at 2400),
split None / "split" (ICMC: explore_i_common.make_split), nudge (REG3's 5% labelled nudge, run on CPU while the model is
built: 300 Adam steps of a full REG3 forward/backward at B = 64, minutes per run at L = 767).
TRAINERS (what each supports)
  run_sequential_icmc  explore_i_common.run_one, transcribed with the model on the device (CHECKed equal to run_one on CPU
                       bit for bit); ICMC; ADAM / SLOW / split; one run.
  run_batched_icmc     R runs of one ICMC arm stacked (explore_gpu.batched): one forward, one backward, StackedAdam;
                       ADAM / SLOW / split (explore_i_common.make_split, the row copy + W_g's Adam state of that run
                       zeroed, step included); each run's evaluation, statistics and split read through its view (the
                       run's own sequential model, parameters aliasing the stacked slices), with explore_i_common's own
                       functions; the evaluation's CPU part optionally in worker processes.
  run_sequential_bind  test_router_reliability.run_one (the harness) inside explore_gpu.device.device_scope: every
                       BindTask / RandHeaderTask arm and recipe the harness has (main-line arms: explore_gpu.mainline).
  run_batched_bind     R runs of a BindTask / RandHeaderTask arm stacked, ADAM only; run_one's record fields; early stop
                       per run (a stopped run is frozen).
  Reset-only (the split's control) is a main-line recipe: it runs on the sequential path (test_window_gate's arms).
RECORDS carry device, gpu (or CPU) model, torch and CUDA versions (device.device_info) besides the harness's fields.
"""

import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_i_task as it
import explore_i_common as ic
import explore_b16_gates as g16
import explore_g_regmodel as rm
import explore_delta_mem as dm
import explore_f_tasks as ft
import test_short_conv as tsc
import test_binding_onset as tbo
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, eval_batch
from test_binding_capacity import BindTask

from explore_gpu import device as dv
from explore_gpu.batched import BatchedBDH, StackedAdam, adam_groups

BATCH = 32
LR, LR_WARM, WARM = 1e-3, 1e-4, 2400
EVAL_EVERY = 1200


# ── code identity ─────────────────────────────────────────────────────────────
def code_sha(module):
    """SHA-1 over the source of every repository module (the root's and explore_gpu's) that importing `module` loads, in
    a fresh child process (12 hex), and their names."""
    import subprocess
    prog = ("import sys, json, hashlib, os; sys.path.insert(0, %r); import %s; h = hashlib.sha1(); names = []\n"
            "for n, m in sorted(sys.modules.items()):\n"
            "    f = getattr(m, '__file__', None)\n"
            "    if isinstance(f, str) and os.path.isabs(f) and os.path.dirname(f) in (%r, %r):\n"
            "        h.update(n.encode()); h.update(open(f, 'rb').read()); names.append(n)\n"
            "print(json.dumps([h.hexdigest()[:12], names]))") % (HERE, module, HERE, os.path.join(HERE, "explore_gpu"))
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True, cwd=HERE).stdout
    sha, names = json.loads(out.strip().splitlines()[-1])
    return sha, names


# ── tasks ─────────────────────────────────────────────────────────────────────
def task_from(s):
    name, _, args = s.partition(":")
    kw = {k: int(v) for k, v in (a.split("=") for a in args.split(",") if a)}
    if name == "icmc":
        return it.ICMCTask(**kw)
    if name == "rhl":
        return ft.RandHeaderTask(**({"P": 4, "nb": 2, "lmin": 1, "lmax": 3} | kw))
    if name == "grouped":
        return tsc.TASK if not kw else BindTask(kw.get("P", 4), S=kw.get("S", 2), n_vals=kw.get("n_vals", 16),
                                                n_q=kw.get("n_q", 1))
    raise ValueError(s)


def icmc_loss_mask(task, tokens):
    """explore_i_task.ICMCTask.loss_mask, vectorised (CHECKed equal): targets that are words, except the first word of
    each source's first block (the target right after the source's first SRC token)."""
    S = task.S
    mask = tokens[:, 1:] >= S
    inp = tokens[:, :-1]
    rows = torch.arange(tokens.shape[0])
    for s in range(S):
        hit = inp == s
        first = hit.int().argmax(1)
        has = hit.any(1)
        mask[rows[has], first[has]] = False
    return mask


# ── arms ──────────────────────────────────────────────────────────────────────
S75_ARMS = {
    "ORACLE": dict(gate="perfect", recipe="adam"),
    "SINGLE": dict(gate="none", k=1, recipe="adam"),
    "WIN3_SPLIT": dict(gate="win3", recipe="slow", split="split"),
    "WIDE_SPLIT": dict(gate="wide", recipe="slow", split="split"),
    "REG3": dict(gate="reg3", recipe="slow"),
    "WIDE_SLOW": dict(gate="wide", recipe="slow"),
    "REG3_NUDGE": dict(gate="reg3", recipe="slow", nudge=True),
}


def arm_spec(name, k):
    s = dict(S75_ARMS[name])
    s.setdefault("k", k)
    return s


def make_model(spec, task, seed):
    """The project's builders on CPU (see the module docstring); the caller has already called torch.manual_seed(seed)."""
    gate, k = spec["gate"], int(spec.get("k", 1))
    conv = spec.get("conv", "layer")
    kw = {} if conv is None else dict(tsc.LAYER)
    if gate == "perfect":
        m = build(task, dict(tsc.ARM["ceiling_conv"], n_ch=k, model_kw=kw), ARCH, seed)
    elif gate == "none":
        assert k == 1
        m = build(task, dict(tsc.ARM["B_conv"], model_kw=kw), ARCH, seed)
    else:
        m = build(task, dict(tsc.ARM["A_conv"], n_ch=k, model_kw=kw), ARCH, seed)
        if gate == "win3":
            m = g16.to_local(m, seed)
        elif gate == "wide":
            m = ic.to_window(m, seed, spec.get("width") or task.lmax + 2)
        elif gate == "reg3":
            m = ic.to_reg3(g16.to_local(m, seed), seed)
            if spec.get("nudge"):
                ic.nudge_reg(m, task, seed)
        else:
            assert gate == "recurrent", gate
    if spec.get("mem", "hebb") == "delta":
        assert gate != "reg3", "REG3 on the delta memory is built by to_reg(mem='delta'), not supported here"
        m = dm.to_delta(m, seed=seed, **spec.get("delta", dict(beta=1.0)))
    return m


def gate_names(m):
    return ic.gate_names(m)


# ── data prefetch (a child process owns every run's generator; values and order unchanged) ───────────────────
def _producer(task, seeds, B, icmc, q, stop):
    torch.set_num_threads(1)
    gens = [torch.Generator().manual_seed(s + 10_000) for s in seeds]
    while not stop.is_set():
        toks = torch.stack([task.make_batch(B, g)[0] for g in gens])
        masks = torch.stack([icmc_loss_mask(task, t) for t in toks]) if icmc else None
        item = (toks.numpy(), None if masks is None else masks.numpy())
        while not stop.is_set():
            try:
                q.put(item, timeout=1.0)
                break
            except queue.Full:
                continue


class Batches:
    """Per step: (tokens (R, B, L), loss mask (R, B, L-1) or None), each run's batches drawn from its own generator."""

    def __init__(self, task, seeds, B=BATCH, icmc=False, prefetch=True, depth=6):
        self.task, self.seeds, self.B, self.icmc, self.prefetch = task, list(seeds), B, icmc, prefetch
        if prefetch:
            ctx = mp.get_context("spawn")
            self.q = ctx.Queue(maxsize=depth)
            self.stop = ctx.Event()
            self.p = ctx.Process(target=_producer, args=(task, self.seeds, B, icmc, self.q, self.stop), daemon=True)
            self.p.start()
        else:
            self.gens = [torch.Generator().manual_seed(s + 10_000) for s in self.seeds]

    def next(self):
        if self.prefetch:
            t, m = self.q.get(timeout=600)
            return torch.from_numpy(t), (None if m is None else torch.from_numpy(m))
        toks = torch.stack([self.task.make_batch(self.B, g)[0] for g in self.gens])
        masks = torch.stack([icmc_loss_mask(self.task, t) for t in toks]) if self.icmc else None
        return toks, masks

    def close(self):
        if self.prefetch:
            self.stop.set()
            self.p.join(timeout=5)
            if self.p.is_alive():
                self.p.terminate()


# ── optimizer state proxy (lets explore_i_common's split zero ONE run's W_g state) ─────────────────────────────
class _RunState:
    def __init__(self, opt, r, by_id):
        self.opt, self.r, self.by_id = opt, r, by_id

    def get(self, p, default=None):
        sp = self.by_id.get(id(p))
        st = None if sp is None else self.opt.run_state(sp, self.r)
        return default if st is None else st


class RunOpt:
    def __init__(self, opt, r, view, bm):
        named = dict(view.named_parameters())
        self.state = _RunState(opt, r, {id(named[n]): bm.P(n) for n in bm.names})


# ── evaluation helpers on a device ────────────────────────────────────────────
EVAL_CHUNK = 32                       # sequences per forward in evaluation on CUDA (T = 767: a (32, 1, T, T) score = 75 MB)
REPLAY_TOL = 1e-4


@torch.no_grad()
def residuals_dev(model, x):
    """explore_i_common.residuals, transcribed for a device: the same replay of the model's layer loop with its own gates,
    with the bit-for-bit assert replaced by max |d| <= REPLAY_TOL (on CUDA the replay's kernels are not guaranteed to be
    the forward's). Called through dv.on_device (inputs moved, outputs to CPU)."""
    import explore_g_regmodel as rm_
    was = model.training
    model.eval()
    try:
        if isinstance(model, rm_.RegBDH):
            model.keep_diag = True
            logits = model(x)[0]
            model.keep_diag = False
            Gs = [torch.einsum("btk,bsk->bts", g, g).unsqueeze(1) for g in model.diag["gates"]]
        else:
            logits = model(x, ic.TAU_END)[0]
            Gs = [model.attn.G] * model.config.n_layer
        C = model.config
        B, T = x.shape
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        h = model.embed(x).unsqueeze(1)
        if model.conv == "in":
            h = model.short_conv(h)
        h = model.ln(h)
        feats = {}
        for level in range(C.n_layer):
            feats[f"L{level + 1}"] = h.reshape(B, T, D_)
            model.attn.G = Gs[level]
            hc = model.short_conv(h) if model.conv == "layer" else h
            x_sparse = F.relu(hc @ model.encoder)
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=hc))
            y_sparse = F.relu(yKV @ model.encoder_v)
            xy = model.drop(x_sparse * y_sparse)
            y = model.ln(xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder)
            h = model.ln(h + y)
        feats["final"] = h.reshape(B, T, D_)
        lg = h.reshape(B, T, D_) @ model.lm_head
        assert float((lg - logits).abs().max()) <= REPLAY_TOL, "residual replay differs from the model's forward"
        return feats, logits
    finally:
        model.train(was)


def _cat(outs):
    a = outs[0]
    if torch.is_tensor(a):
        return torch.cat(outs, 0)
    if isinstance(a, dict):
        return {k: _cat([o[k] for o in outs]) for k in a}
    if isinstance(a, (list, tuple)):
        return type(a)(_cat([o[i] for o in outs]) for i in range(len(a)))
    return a


def chunked(fn, chunk=EVAL_CHUNK):
    """f(model, x, ...) over chunks of x's sequences, results concatenated (every sequence is computed alone)."""
    def g(model, x, *a, **kw):
        if x.shape[0] <= chunk:
            return fn(model, x, *a, **kw)
        return _cat([fn(model, x[i:i + chunk], *a, **kw) for i in range(0, x.shape[0], chunk)])
    return g


# ── evaluation split in two: the device part (forwards) and the CPU part (statistics), the latter possibly in workers ─
def eval_device_part(model, task, ev, pdata, with_probe=True):
    """explore_i_common.evaluate's model calls, in its order: logits on the evaluation set, gates on its first ROUTE_N
    sequences, the residual features on the probe set (CPU tensors; through eval_scope's chunking on CUDA)."""
    x = ev["tokens"][:, :-1]
    lg = ic.logits_chunked(model, x)
    gates = ic.model_gates(model, x[:ic.ROUTE_N]) if model.n_ch > 1 else None
    feats = None
    if with_probe and pdata is not None and task.S >= 2:
        feats = ic.residuals(model, pdata[0])[0]
    return lg, gates, feats


def eval_cpu_part(task, ev, pdata, lg, gates, feats):
    """explore_i_common.evaluate's statistics from the device part's tensors: the same operations in the same order
    (CHECKed equal to evaluate bit for bit on CPU)."""
    x, tok = ev["tokens"][:, :-1], ev["tokens"]
    tgt = tok[:, 1:]
    m, bins, sets, src = ev["ann"]["mask"], ev["bins"], ev["sets"], ev["ann"]["src"]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    hit = it.set_hits(task, lg, sets)
    out = dict(loss=float(ce[m].mean()), set=float(hit[m].float().mean()))
    for i, name in enumerate(it.BIN_NAMES):
        mm = m & (bins == i)
        out[f"loss_{name}"] = float(ce[mm].mean()) if mm.any() else None
        out[f"set_{name}"] = float(hit[mm].float().mean()) if mm.any() else None
    out["set_by_src"] = [float(hit[m & (src == s)].float().mean()) for s in range(task.S)]
    if gates is not None:
        xr = x[:ic.ROUTE_N]
        out["routing"] = {f"L{i + 1}": ic.routing_layer(g, xr, task) for i, g in enumerate(gates)}
    if feats is not None:
        px, tr, te = pdata
        lab = task.stream_labels(px)
        word = px >= task.S
        sub = torch.Generator().manual_seed(ic.PROBE_SEED + 2)
        pr = {}
        for name, f in feats.items():
            Xtr, ytr = f[tr][word[tr]], lab[tr][word[tr]]
            Xte, yte = f[te][word[te]], lab[te][word[te]]
            ptr = torch.randperm(Xtr.shape[0], generator=sub)[:ic.PROBE_TR]
            pte = torch.randperm(Xte.shape[0], generator=sub)[:ic.PROBE_TE]
            h, tracc = ic.fit_score(Xtr[ptr].double().numpy(), ytr[ptr].numpy(), Xte[pte].double().numpy(),
                                    yte[pte].numpy())
            pr[name] = dict(acc=float(h.mean()), n=int(h.size), train_acc=tracc)
        out["probe"] = pr
    return out


_W = {}


def _stats_init(task, ev, pdata):
    torch.set_num_threads(1)
    _W.update(task=task, ev=ev, pdata=pdata)


def _stats_job(lg, gates, feats):
    return eval_cpu_part(_W["task"], _W["ev"], _W["pdata"], lg, gates, feats)


class AsyncStats:
    """The CPU part of every evaluation in a pool of worker processes (spawn), so the GPU does not wait for the probe's
    logistic regressions; results are collected in submission order."""

    def __init__(self, task, ev, pdata, workers):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_stats_init, initargs=(task, ev, pdata))
        self.jobs = []

    def submit(self, key, extra, lg, gates, feats):
        self.jobs.append((key, extra, self.pool.submit(_stats_job, lg, gates, feats)))

    def collect(self):
        out = []
        for key, extra, f in self.jobs:
            out.append((key, dict(f.result(), **extra)))
        self.pool.shutdown()
        return out


class eval_scope:
    """explore_i_common's evaluation and split on device models (CUDA only; on CPU nothing changes): the residual replay on
    the device (residuals_dev), the gates and the replay in chunks of EVAL_CHUNK sequences, logits in chunks of
    EVAL_CHUNK, the split's noise drawn on CPU and moved."""

    def __init__(self, dev):
        self.dev = dev
        self.saved = []

    def __enter__(self):
        if self.dev.type != "cpu":
            lc = ic.logits_chunked
            for attr, new in (("residuals", chunked(dv.on_device(residuals_dev))),
                              ("model_gates", chunked(ic.model_gates)),
                              ("logits_chunked", lambda model, x, chunk=EVAL_CHUNK, _lc=lc: _lc(model, x, chunk)),
                              ("split_op", dv._split_wrapper(ic.split_op))):
                self.saved.append((attr, getattr(ic, attr)))
                setattr(ic, attr, new)
        return self

    def __exit__(self, *a):
        for attr, orig in reversed(self.saved):
            setattr(ic, attr, orig)


@dv.on_device
def _beta_by_role(model, x, S):
    beta = torch.sigmoid(model.embed(x) @ model.w_R + model.b_R)
    return torch.stack([beta[x < S].mean(), beta[x >= S].mean()])


def _icmc_record(curve, ts, iters, k, holder_names, trig, model, task, ev):
    if not all(math.isfinite(c["loss"]) for c in curve):
        raise ValueError("non-finite loss in the curve")
    rec = dict(ok=True, curve=curve, end=curve[-1], iters=iters, k=k, group_names=holder_names, secs=time.time() - ts)
    if trig is not None:
        rec.update(checks=trig.rows, splits=sum(r["fired"] for r in trig.rows))
    if isinstance(model, rm.RegBDH):
        with torch.no_grad():
            b = _beta_by_role(model, ev["tokens"][:ic.ROUTE_N, :-1], task.S)
        rec["beta_src"], rec["beta_word"] = float(b[0]), float(b[1])
    return rec


def _tag_print(tag, step, e):
    rt = " ".join(f"{L} eta {r['eta']:.2f} m {r['margin']:.2f}" for L, r in (e.get("routing") or {}).items())
    pr = " ".join(f"{k} {v['acc']:.2f}" for k, v in (e.get("probe") or {}).items())
    print(f"    .. {tag} step {step}: SET {e['set']:.3f} loss {e['loss']:.3f} {rt} probe {pr} ({e['secs'] / 60:.1f} min)",
          flush=True)


# ── ICMC: sequential ──────────────────────────────────────────────────────────
def run_sequential_icmc(task, spec, seed, iters, device="cpu", split_thr=None, with_probe=True, tag=None,
                        eval_every=EVAL_EVERY, on_eval_extra=None):
    """explore_i_common.run_one, transcribed: the model built on CPU, then moved to the device (explore_gpu.device)."""
    dev = dv.setup(device)
    ts = time.time()
    holder = {}
    ev, pdata = ic.eval_set(task), ic.probe_set(task)
    curve = []
    trig = ic.make_split(task, seed, iters - ic.SPLIT_EVERY, split_thr) if spec.get("split") else None
    slow = spec.get("recipe") == "slow"
    assert spec.get("split") in (None, "split"), "reset-only on ICMC: not provided by explore_i_common"
    try:
        with eval_scope(dev):
            torch.manual_seed(seed)
            model = dv.to_device(make_model(spec, task, seed), dev)
            groups = ic.slow_groups(holder)(model) if slow else None
            opt = torch.optim.Adam(model.parameters() if groups is None else groups, lr=LR)
            rng = torch.Generator(); rng.manual_seed(seed + 10_000)
            for step in range(1, iters + 1):
                tokens, _ = task.make_batch(BATCH, rng)
                inp, tgt = tokens[:, :-1], tokens[:, 1:]
                logits = tbo.logits_of(model, inp)
                mask = task.loss_mask(tokens)
                loss = F.cross_entropy(logits[mask], tgt[mask])
                opt.zero_grad(); loss.backward()
                opt.step()
                if step % eval_every == 0:
                    e = ic.evaluate(model, task, ev, pdata, with_probe)
                    e["step"] = step
                    e["secs"] = round(time.time() - ts, 1)
                    curve.append(e)
                    if tag is not None:
                        _tag_print(tag, step, e)
                    if slow and step >= WARM:
                        holder["groups"][1]["lr"] = LR
                    if trig is not None:
                        trig(model, step, opt)
                    if on_eval_extra is not None:
                        on_eval_extra(model, step)
            rec = _icmc_record(curve, ts, iters, model.n_ch, holder.get("names"), trig, model, task, ev)
            rec["seed"] = seed
    except Exception as e:
        rec = dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:],
                   secs=time.time() - ts)
    rec.update(device=dv.device_info(dev), path="sequential")
    return json.loads(json.dumps(rec, default=float))


# ── ICMC: batched ─────────────────────────────────────────────────────────────
def run_batched_icmc(task, spec, seeds, iters, device="cpu", split_thr=None, with_probe=True, tag=None,
                     eval_every=EVAL_EVERY, prefetch=True, register="auto", on_eval_extra=None, log=print,
                     stats_workers=None):
    """R = len(seeds) runs of one arm, stacked; one record per seed (see the module docstring). stats_workers: the CPU
    part of each evaluation (explore_i_common.evaluate's statistics; the probe's logistic regressions dominate) runs in
    that many worker processes while training continues (default: 3 on CUDA, 0 = in line on CPU)."""
    dev = dv.setup(device)
    ts = time.time()
    seeds = list(seeds)
    R = len(seeds)
    ev, pdata = ic.eval_set(task), ic.probe_set(task)
    curves = [[] for _ in seeds]
    slow = spec.get("recipe") == "slow"
    assert spec.get("split") in (None, "split")
    data = None
    try:
        models = []
        for s in seeds:
            torch.manual_seed(s)
            models.append(make_model(spec, task, s))
        names = None
        if slow:
            gnames = gate_names(models[0])
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        bm = BatchedBDH(models, dev, register=register)
        groups, names = adam_groups(bm, gnames if slow else None, LR, LR_WARM)
        names = names if slow else None                     # explore_i_common.run_one records None for ADAM arms
        opt = StackedAdam(groups, R)
        views = [bm.view(r, wrap=lambda m: dv.to_device(m, dev)) for r in range(R)]
        ropts = [RunOpt(opt, r, views[r], bm) for r in range(R)]
        trigs = [ic.make_split(task, s, iters - ic.SPLIT_EVERY, split_thr) if spec.get("split") else None for s in seeds]
        data = Batches(task, seeds, BATCH, icmc=True, prefetch=prefetch)
        nw = (3 if dev.type == "cuda" else 0) if stats_workers is None else stats_workers
        astats = AsyncStats(task, ev, pdata, nw) if nw > 0 else None
        t_train = 0.0
        with eval_scope(dev):
            for step in range(1, iters + 1):
                t0 = time.time()
                tok, mask = data.next()
                tok, mask = tok.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
                inp, tgt = tok[..., :-1], tok[..., 1:]
                logits, _ = bm(inp)
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                     reduction="none").view(tgt.shape)
                mf = mask.to(ce.dtype)
                loss_r = (ce * mf).sum((1, 2)) / mf.sum((1, 2))
                opt.zero_grad()
                loss_r.sum().backward()
                opt.step()
                if dev.type == "cuda" and step % 50 == 0:
                    torch.cuda.synchronize()
                t_train += time.time() - t0
                if step % eval_every == 0:
                    for r in range(R):
                        extra = dict(step=step, secs=round(time.time() - ts, 1), train_secs=round(t_train, 1))
                        if astats is None:
                            e = dict(ic.evaluate(views[r], task, ev, pdata, with_probe), **extra)
                            curves[r].append(e)
                            if tag is not None:
                                _tag_print(f"{tag}|{seeds[r]}", step, e)
                        else:
                            astats.submit(r, extra, *eval_device_part(views[r], task, ev, pdata, with_probe))
                            if tag is not None:
                                print(f"    .. {tag}|{seeds[r]} step {step} (statistics queued; "
                                      f"{t_train / 60:.1f} min training)", flush=True)
                    if slow and step >= WARM:
                        groups[1]["lr"] = LR
                    for r in range(R):
                        if trigs[r] is not None:
                            trigs[r](views[r], step, ropts[r])
                    if on_eval_extra is not None:
                        on_eval_extra(bm, views, step)
            if astats is not None:
                for r, e in astats.collect():
                    curves[r].append(e)
                    if tag is not None:
                        _tag_print(f"{tag}|{seeds[r]}", e["step"], e)
                astats = None
            recs = []
            peak = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3) if dev.type == "cuda" else None
            for r, s in enumerate(seeds):
                try:                                         # each run's record on its own (one run's failure is its own)
                    rec = _icmc_record(curves[r], ts, iters, views[r].n_ch, names, trigs[r], views[r], task, ev)
                except Exception as e:
                    rec = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:],
                               secs=time.time() - ts)
                rec.update(seed=s, batched_with=seeds, R=R, register=bm.register_mode, train_secs=t_train / R,
                           train_secs_batch=t_train, peak_mem_gb_batch=peak)
                recs.append(rec)
    except Exception as e:
        err = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:],
                   secs=time.time() - ts, batched_with=seeds)
        recs = [dict(err, seed=s) for s in seeds]
    finally:
        if data is not None:
            data.close()
        if "astats" in locals() and astats is not None:
            astats.pool.shutdown(cancel_futures=True)
    info = dv.device_info(dev)
    for rec in recs:
        rec.update(device=info, path="batched")
    return [json.loads(json.dumps(rec, default=float)) for rec in recs]


# ── BindTask / RandHeaderTask: sequential (the harness on a device) ───────────
def run_sequential_bind(arm, seed, iters, device="cpu", **run_one_kw):
    """test_router_reliability.run_one(arm, seed, iters, **kw) with the model on the device."""
    import test_router_reliability as trr
    with dv.device_scope(device) as dev:
        rec = trr.run_one(arm, seed, iters, **run_one_kw)
    rec.update(device=dv.device_info(dev), path="sequential")
    return rec


# ── BindTask / RandHeaderTask: batched ────────────────────────────────────────
def run_batched_bind(task, builder, seeds, iters, device="cpu", eval_every=EVAL_EVERY, early_stop=True, stats_fn=None,
                     grad_fn=None, grad_steps=(1, 10, 100), lr=LR, on_eval_extra=None, prefetch=False):
    """R runs of a BindTask-family arm: builder(seed) -> make() (as test_router_reliability.make_fn(a, seed)); ADAM only
    (no recipe, no trigger: main-line recipes run on the sequential path, explore_gpu.mainline). Records with run_one's
    fields (curve [step, acc, loss], transition, stopped_at, stats, grad, acc, ...); stats_fn and grad_fn as run_one's
    (defaults gate_stats_k and test_readout_path.grad_norms), grad_fn reading run r's gradients through its view; a run
    with a non-finite evaluation is ok = False (test_binding_onset.attempt's rule)."""
    import test_router_reliability as trr
    dev = dv.setup(device)
    ts = time.time()
    seeds = list(seeds)
    R = len(seeds)
    stats_fn = stats_fn or trr.gate_stats_k
    grad_fn = grad_fn or trr.grad_norms
    data_eval = eval_batch(task)
    probes = [probe_batch(task, s) for s in seeds]
    models, stats = [], [[] for _ in seeds]
    for s in seeds:
        torch.manual_seed(s)
        models.append(builder(s)())
    bm = BatchedBDH(models, dev)
    views = [bm.view(r, wrap=lambda m: dv.to_device(m, dev)) for r in range(R)]
    for r in range(R):
        stats[r].append(stats_fn(views[r], task, probes[r], 0))
    groups, _ = adam_groups(bm, None, lr)
    opt = StackedAdam(groups, R)
    data = Batches(task, seeds, BATCH, icmc=False, prefetch=prefetch)
    curves, grads = [[] for _ in seeds], [{} for _ in seeds]
    streak, active = [0] * R, [True] * R
    try:
        for step in range(1, iters + 1):
            tok, _ = data.next()
            tok = tok.to(dev)
            inp, tgt = tok[..., :-1], tok[..., 1:]
            logits, _ = bm(inp)
            Rr, B, T, V = logits.shape
            ql, qt, _ = task.select(logits.reshape(Rr * B, T, V), tgt.reshape(Rr * B, T), None)
            loss_r = F.cross_entropy(ql, qt, reduction="none").view(Rr, -1).mean(1)
            act = torch.tensor(active, dtype=loss_r.dtype, device=loss_r.device)
            opt.zero_grad()
            (loss_r * act).sum().backward()
            if step in grad_steps:
                for r in range(R):
                    if active[r]:
                        vp = dict(views[r].named_parameters())
                        for n in bm.names:
                            g = bm.P(n).grad
                            vp[n].grad = None if g is None else g[r]
                        grads[r][str(step)] = grad_fn(views[r])
                        for n in bm.names:
                            vp[n].grad = None
            opt.step(active)
            if step % eval_every == 0:
                for r in range(R):
                    if not active[r]:
                        continue
                    acc, el = tbo.evaluate(views[r], task, data_eval)
                    curves[r].append([step, acc, el])
                    stats[r].append(stats_fn(views[r], task, probes[r], step))
                    streak[r] = streak[r] + 1 if acc >= tbo.BIND_PASS else 0
                    if early_stop and streak[r] >= tbo.STOP_AFTER:
                        active[r] = False
                if on_eval_extra is not None:
                    on_eval_extra(bm, views, step, active)
                if not any(active):
                    break
    finally:
        data.close()
    recs = []
    info = dv.device_info(dev)
    for r, s in enumerate(seeds):
        c = curves[r]
        end = stats_fn(views[r], task, probes[r], "end")
        finite = all(math.isfinite(a_) and math.isfinite(l_) for _, a_, l_ in c)
        rec = dict(ok=finite, seed=s, curve=c, transition=tbo.transition(c), stopped_at=c[-1][0] if c else 0,
                   params=sum(p.numel() for p in models[r].parameters() if p.requires_grad), secs=time.time() - ts,
                   key_cos=end.get("key_cos"), val_cos=end.get("val_cos"), ctx_cos=end.get("ctx_cos"), end=end,
                   stats=stats[r], grad=grads[r], iters=iters, k=models[r].n_ch, device=info, path="batched",
                   batched_with=seeds)
        rec["acc"] = c[-1][1] if c else float("nan")
        rec["collapsed"] = rec["acc"] < trr.COLLAPSE
        rec["discovered"] = bool(rec["transition"] is not None and (rec["val_cos"] or 1.0) < trr.DISC_COS)
        if not finite:
            rec["error"] = "ValueError: non-finite accuracy or loss in the curve"
        recs.append(json.loads(json.dumps(rec, default=float)))
    return recs
