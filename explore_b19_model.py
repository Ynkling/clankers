#!/usr/bin/env python
"""
explore_b19_model.py — EXPLORATORY, not a result. Batch 19's model-side helpers, used INSIDE child processes on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Imports existing modules read-only; changes no file. Nothing here
draws from the global RNG except the model construction it wraps.

SIZE (width and depth). test_stream_recipe.builder_for(a) builds through test_router_discovery.model_fn(task, a, seed) =
test_multilayer_binding.build(task, a, ARCH, seed) (ARCH = n_layer 3, positional decay). builder_of(a, mult, d, n_layer)
is that builder with three numbers replaced and nothing else: the arm's mult (N = mult x D, the BDH sparse width),
ARCH's n_layer, and the embedding width D (test_multilayer_binding.D, a module global read only while MultiBDH is
constructed: n_embd, the gate's W_in (h_gate x D) and the stack's convolution (D x 4); set for the construction and
restored). Initial scales are the code's defaults at the new size (bdh.BDH: encoder, encoder_v, decoder, lm_head and the
embedding N(0, 0.02^2); the gate's W_in, W_h, W_g N(0, 0.1^2); the stack conv the identity; h_gate 32 at every size);
init_scales() records them. At N=256, D=32, 3 layers the builder's model equals the run path's own bit for bit (CHECK).

LOCAL3 at any D: to_local_any(m, seed) is explore_b16_gates.to_local at D = 32 (batch 16's conversion, the same object);
at another D the same conversion with a (D x 3) window drawn by batch 1's rule from the same generator (seed + 31337,
uniform +-1/sqrt(3); its first 32 rows are the D = 32 window, because torch.rand fills row-major).

RES6 (S65's conditional diagnostic arm): Session G's per-layer gate over the residual stream, copied from
docs/reading/distant_cues_2026-10.md section 4.3 (main branch):
    u_t^(l) = A [ RMSNorm(r_t^(l)) ; v_t ]          A learned, (D + D) -> h_gate
    g_t^(l) = softmax( W_g tanh(u_t^(l)) ),       layer 1 keeps the window gate (no residual yet)
    init A so that u^(l) = W_in v_t at step 0     (A = [0 | W_in], the residual block zero)
r^(l) is the residual stream entering layer l (forward_conv's x, already LayerNormed by the stack, so RMSNorm of it equals
the user's "LN(r)" up to the norm's eps); v_t the raw embedding; W_g one matrix shared by every layer (TRM's shared gate);
read gate = write gate at every layer (LOCAL3's symmetry); no stop-gradient (the document's "let gradient flow into layer
l-1"). A trains in the gate group. The model returns layer 1's gate as its read/write gate (the run path's statistics and
outcome are layer 1's); layer_maps() gives every layer's routing (test_scale_axes.routing_k on that layer's gate).

EVALUATION IN CHUNKS (memory). test_binding_onset.evaluate and test_scale_axes.stream_acc run the 2048-sequence held-out
batch through the model in one forward; at P=16 and N=1024 that needs about 10.6 GB, more than four parallel runs can
have. install() makes test_binding_onset.logits_of run a gradient-free forward of more than CHUNK sequences in chunks of
CHUNK and concatenate the logits; training forwards (gradient on) are unchanged. On this machine the chunked logits are
bit-identical to the one-forward logits (CHECK in explore_scale_child, several sizes and tasks), so every number is
unchanged.

CHECKPOINTS (runs longer than a segment). A run executes in one child process; a segment ends by killing it. Every
evaluation of test_binding_onset.onset_run is followed by a checkpoint (onset_run_ckpt: onset_run line for line plus the
save, written atomically), and a run started with a checkpoint present resumes from it:
  - saved: the step, the model's state_dict, the optimizer's per-parameter state and per-group hyperparameters, the data
    generator's state, the global torch RNG state, onset_run's curve and streak, and the RECORDER's log;
  - the RECORDER wraps every measurement whose result the run's callbacks keep (test_binding_onset.evaluate; the run
    path's stats_fn and grad_fn as composed by the recipe; the trigger's explore_split_target_child.read_gate / select,
    test_scale_axes.routing_k and explore_split_child.split_op) and logs each result in call order (a call nested inside a
    logged call is not logged);
  - on resume the run is rebuilt as at step 0 (the same seed, construction, optimizer) and steps 1..K are replayed
    WITHOUT training: every callback runs as before with each recorded measurement returned from the log in order (names
    asserted), so every list, dict and counter the callbacks keep (the statistics, the check's early curve, the lr switch,
    the trigger's state and rows) is rebuilt exactly; at K the saved model, optimizer, generator and RNG states are loaded
    (the optimizer's in place, so the recipe's group dicts remain the optimizer's own; the group lrs reached by the replay
    are asserted equal to the saved ones) and training continues at K + 1.
  CHECK (explore_scale_child): a run interrupted after a checkpoint and resumed equals the uninterrupted run bit for bit
  (curve, statistics, gradient norms, the trigger's rows, the end statistics, the final weights), on the k16 path and on
  the S=8 path with forced splits before the checkpoint.
"""

import contextlib
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_b16_gates as g16
import test_binding_onset as tbo
import test_multilayer_binding as tmb
import test_router_discovery as trd
import test_stream_recipe as tsr

SIZES = {"N256": (8, 32), "N512": (16, 32), "N1024": (32, 32), "D64": (4, 64), "D128": (2, 128)}   # name: (mult, D)
D0 = 32
GATE_RES = ("W_in", "W_g", "gate_conv.conv_w", "A")


# ── Size ─────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def embed_dim(d):
    old = tmb.D
    tmb.D = int(d)
    try:
        yield
    finally:
        tmb.D = old


def task_of_arm(a):
    task = tsr.TASKS[a["task"]]
    return tsr.padded_task(task, a["ctx_pad"]) if a.get("ctx_pad") else task


def builder_of(mult, d, n_layer):
    """builder(a, seed) -> make: test_router_discovery.model_fn's build with the arm's mult, ARCH's n_layer and the
    embedding width replaced (the arm dict passed by the run path, its task as test_stream_recipe.builder_for's)."""
    arch = dict(trd.ARCH, n_layer=int(n_layer))

    def builder(a, seed):
        assert a.get("noise") is None and not a.get("nudge"), "model_fn's noise / nudge are not used here"
        task = task_of_arm(a)

        def make():
            with embed_dim(d):
                return tmb.build(task, dict(a, mult=int(mult)), arch, seed)
        return make
    return builder


def init_scales(m):
    out = {}
    for n, p in m.named_parameters():
        x = p.detach()
        out[n] = dict(shape=list(x.shape), std=float(x.std()) if x.numel() > 1 else 0.0, absmax=float(x.abs().max()),
                      trainable=bool(p.requires_grad))
    return out


# ── LOCAL3 at any D ──────────────────────────────────────────────────────────
def window_init_d(seed, d, width=g16.WIDTH):
    g = torch.Generator().manual_seed(int(seed) + g16.GCONV_OFFSET)
    return (torch.rand(int(d), width, generator=g) * 2 - 1) * (1.0 / math.sqrt(width))


def to_local_any(m, seed):
    d = m.config.n_embd
    if d == g16.D:
        return g16.to_local(m, seed)
    assert type(m) is tmb.MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
    assert not m.gate_to_readout and not m.gate_ln and m.gate_noise == 0.0
    m.__class__ = g16.LocalBDH
    m.local = True
    m.gate_conv = tmb.CausalConv(d, g16.WIDTH)
    with torch.no_grad():
        m.gate_conv.conv_w.copy_(window_init_d(seed, d))
    m.W_h.requires_grad_(False)
    return m


# ── RES6: Session G's per-layer gate over the residual stream (4.3) ─────────
class ResGateBDH(g16.LocalBDH):
    """LOCAL3 at layer 1; at layer l >= 2 g^(l) = softmax(W_g tanh(A [RMSNorm(r^(l)); v])), read = write. forward_res is
    test_multilayer_binding.MultiBDH.forward_conv line for line with the gate set before each layer's attention.
    force_g1 (CHECK only) gives every layer layer 1's gate, so forward_res must equal LocalBDH's forward."""
    force_g1 = False

    def forward(self, tokens, tau=None, track_sat=False):
        assert self.conv == "layer", "RES6 is defined on the run path's conv 'layer' stack"
        v = self.embed(tokens)
        g1, _ = self.local_gates(v)
        self._layer_gates = [g1]
        logits = self.forward_res(tokens, v, g1)
        return logits, None, g1, g1

    def forward_res(self, idx, v, g1):
        C = self.config
        B, T = idx.size()
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        for level in range(C.n_layer):
            if level == 0 or self.force_g1:
                g = g1
            else:
                r = x.squeeze(1)                                              # r^(l): (B, T, D)
                u = torch.cat([F.rms_norm(r, (D_,)), v], dim=-1) @ self.A.T
                g = F.softmax(torch.tanh(u) @ self.W_g.T, dim=-1)
                self._layer_gates.append(g)
            self.attn.G = torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
            xc = self.short_conv(x) if self.conv == "layer" else x
            x_latent = xc @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=xc)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        return x.view(B, T, D_) @ self.lm_head


def to_resgate(m, seed):
    m = to_local_any(m, seed)
    m.__class__ = ResGateBDH
    d = m.config.n_embd
    m.A = nn.Parameter(torch.cat([torch.zeros(m.h_gate, d), m.W_in.detach().clone()], dim=1))
    return m


class _LayerGate:
    """A stand-in model for test_scale_axes.routing_k: returns one layer's gate as the read gate."""

    def __init__(self, m, layer):
        self.m, self.layer, self.training = m, layer, m.training

    def eval(self):
        self.m.eval()

    def train(self, mode=True):
        self.m.train(mode)

    def __call__(self, x, tau=None):
        self.m(x, tau)
        return None, None, self.m._layer_gates[self.layer]


@torch.no_grad()
def layer_maps(m, task, probe, routing_k):
    return [routing_k(_LayerGate(m, l), task, probe) for l in range(m.config.n_layer)]


# ── Checkpoints ──────────────────────────────────────────────────────────────
class Halt(BaseException):
    """Raised after the checkpoint at CK['halt_at'] (CHECK only); a BaseException so test_binding_onset.attempt does not
    turn it into a failed record."""


class Recorder:
    def __init__(self):
        self.log, self.replay, self.pos, self.depth = [], None, 0, 0

    def reset(self):
        self.log, self.replay, self.pos, self.depth = [], None, 0, 0

    def wrap(self, name, f, keep=True):
        """keep=False: the result is logged as None and replayed as None (read_gate: only select reads it)."""
        def g(*a, **k):
            if self.depth > 0:
                return f(*a, **k)
            if self.replay is not None:
                assert self.pos < len(self.replay), f"replay ran past the log at {name}"
                nm, val = self.replay[self.pos]
                assert nm == name, f"replay out of order: logged {nm}, called {name} (entry {self.pos})"
                self.pos += 1
                out = copy.deepcopy(val)
                self.log.append((name, copy.deepcopy(val)))
                return out
            self.depth += 1
            try:
                out = f(*a, **k)
            finally:
                self.depth -= 1
            self.log.append((name, copy.deepcopy(out) if keep else None))
            return out
        g._b19_orig = f                     # not __wrapped__: torch.no_grad's functools.wraps already sets that
        return g


REC = Recorder()
CK = dict(path=None, halt_at=None, resumed=[], saved=0)
ORIG_ONSET = tbo.onset_run
ORIG_LOGITS = tbo.logits_of
CHUNK = 256


def logits_chunked(model, x):
    """test_binding_onset.logits_of, in chunks of CHUNK sequences when no gradient is taken (evaluation)."""
    if x.shape[0] <= CHUNK or torch.is_grad_enabled():
        return ORIG_LOGITS(model, x)
    return torch.cat([ORIG_LOGITS(model, x[i:i + CHUNK]) for i in range(0, x.shape[0], CHUNK)])


def _params(opt):
    return [p for g in opt.param_groups for p in g["params"]]


def opt_state(opt):
    st = []
    for p in _params(opt):
        s = opt.state.get(p)
        st.append(None if not s else {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in s.items()})
    groups = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    return dict(state=st, groups=groups)


def opt_restore(opt, s):
    ps = _params(opt)
    assert len(ps) == len(s["state"]), "optimizer parameters differ from the checkpoint's"
    for p, st in zip(ps, s["state"]):
        if st is None:
            opt.state.pop(p, None)
        else:
            opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    assert len(opt.param_groups) == len(s["groups"])
    for g, sg in zip(opt.param_groups, s["groups"]):
        assert g["lr"] == sg["lr"], f"the replayed group lr {g['lr']} differs from the checkpoint's {sg['lr']}"
        for k, v in sg.items():
            g[k] = v


def save_ckpt(path, d):
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)
    with open(path + ".step.tmp", "w") as f:
        f.write(str(d["step"]))
    os.replace(path + ".step.tmp", path + ".step")


def load_ckpt(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:                                                    # noqa: BLE001
        print(f"  checkpoint {path} unreadable ({type(e).__name__}: {e}); starting fresh", flush=True)
        return None


def onset_run_ckpt(task, make_model, seed, max_iters=tbo.MAX_ITERS, eval_every=tbo.EVAL_EVERY, data=None, early_stop=True,
                   lr=tbo.LR, warmup=0, param_groups=None, on_eval=None, grad_hook=None, post_step=None,
                   early_stop_after=None, task_at=None, lr_at=None):
    """test_binding_onset.onset_run line for line, with a checkpoint after every evaluation and a resume (module doc)."""
    assert post_step is None, "keep is not used with checkpoints"
    ck = load_ckpt(CK["path"])
    K = 0
    REC.reset()
    if ck is not None:
        REC.replay = ck["log"]
        K = ck["step"]
        CK["resumed"].append(K)
    torch.manual_seed(seed)
    model = make_model()
    if param_groups is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        assert warmup == 0, "param_groups sets per-group lrs; warmup would overwrite them"
        opt = torch.optim.Adam(param_groups(model), lr=lr)
    assert lr_at is None or (warmup == 0 and param_groups is None), "lr_at sets every group's lr"
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    curve, streak = [], 0
    for step in range(1, max_iters + 1):
        if warmup > 0:
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, step / warmup)
        if lr_at is not None:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
        if step <= K:                                                         # replay: no training
            if grad_hook is not None:
                grad_hook(model, step)
            if data is not None and step % eval_every == 0:
                acc, el = tbo.evaluate(model, task, data)
                curve.append([step, acc, el])
                if on_eval is not None:
                    on_eval(model, step)
                counts = early_stop_after is None or step > early_stop_after
                streak = streak + 1 if (acc >= tbo.BIND_PASS and counts) else 0
            if step == K:
                assert curve == ck["curve"] and streak == ck["streak"], "the replayed curve differs from the checkpoint's"
                assert REC.pos == len(REC.replay), f"replay ended at entry {REC.pos} of {len(REC.replay)}"
                REC.replay = None
                model.load_state_dict(ck["model"])
                opt_restore(opt, ck["opt"])
                rng.set_state(ck["rng"])
                torch.set_rng_state(ck["torch_rng"])
            continue
        bt = task if task_at is None else task_at(step)
        tokens, _ = bt.make_batch(tbo.BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        ql, qt, _ = bt.select(tbo.logits_of(model, inp), tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward()
        if grad_hook is not None:
            grad_hook(model, step)
        opt.step()
        if data is not None and step % eval_every == 0:
            acc, el = tbo.evaluate(model, task, data)
            curve.append([step, acc, el])
            if on_eval is not None:
                on_eval(model, step)
            counts = early_stop_after is None or step > early_stop_after
            streak = streak + 1 if (acc >= tbo.BIND_PASS and counts) else 0
            if early_stop and streak >= tbo.STOP_AFTER:
                break
            if CK["path"]:
                save_ckpt(CK["path"], dict(step=step, model=model.state_dict(), opt=opt_state(opt), rng=rng.get_state(),
                                           torch_rng=torch.get_rng_state(), curve=copy.deepcopy(curve), streak=streak,
                                           log=list(REC.log)))
                CK["saved"] += 1
                if CK["halt_at"] == step:
                    raise Halt(step)
    return model, curve, rng


def install(wrap_mods):
    """In the child, before the run: onset_run -> onset_run_ckpt; the listed (module, attribute, keep) measurements
    wrapped by the recorder (once)."""
    tbo.onset_run = onset_run_ckpt
    tbo.logits_of = logits_chunked
    for mod, attr, keep in wrap_mods:
        f = getattr(mod, attr)
        if getattr(f, "_b19_orig", None) is None:
            setattr(mod, attr, REC.wrap(f"{mod.__name__}.{attr}", f, keep=keep))


def orig(f):
    """The unwrapped measurement (the recorder's wrapper's original, else f)."""
    return getattr(f, "_b19_orig", f)


def wrap_kw(kw):
    """The run path's stats_fn and grad_fn as composed by the recipe, wrapped (applied last in the recipe chain)."""
    kw = dict(kw)
    kw["stats_fn"] = REC.wrap("stats_fn", kw["stats_fn"])
    kw["grad_fn"] = REC.wrap("grad_fn", kw["grad_fn"])
    return kw
