"""
explore_gpu/device.py — EXPLORATORY tooling, not a result. The SEQUENTIAL path on any device: the project's own model
classes, run by the project's own code, with the model's parameters on the device. No project module is edited; the
model's class is swapped at runtime for a thin subclass, and a few module functions are wrapped (see device_scope).

  setup(device)        float32 everywhere: on CUDA, TF32 off for matmuls and cuDNN, matmul precision "highest";
                       returns the torch.device.
  device_info(device)  the meta every record and results file carries: device, GPU model, capability, memory, CUDA and
                       cuDNN versions, torch version, CPU model.
  to_device(m, dev)    moves a built model to dev and swaps its class for Dev<Class>: forward moves its tensor inputs
                       to dev, runs the class's own forward inside `with torch.device(dev)` (so the tensors the project's
                       forwards create without a device — Instrument.gates' zeros / ones, perfect_gate_general's arange,
                       decay_mask, explore_delta_mem's full / ones / zeros — are created on dev), and, with cpu_io
                       (default), returns its tensor outputs on CPU (the copies are differentiable, so the project's
                       losses, hinges, statistics and triggers, which build CPU tensors, run unchanged). Gate noise
                       (MultiBDH.gate_noise > 0) is drawn from the model's CPU generator and then moved. On CPU the
                       wrapper changes nothing: every value is bit for bit the unwrapped model's (CHECKed).
  on_device(fn)        wraps a function f(model, *args) that reaches into the model's parameters directly (not through
                       forward): its tensor arguments move to the model's device, it runs inside `with
                       torch.device(dev)`, its tensor results come back to CPU.
  device_scope(dev)    a context manager for running a main-line test's own run code on dev: test_binding_onset.onset_run
                       (and every module that bound it by name) builds the model with the test's own make_model on CPU
                       under the test's own seeding, then to_device; the split's noise and the other direct parameter
                       accesses found by the audit are wrapped (PATCHES below). On exit everything is restored.
"""

import contextlib
import functools
import importlib
import os
import platform
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def setup(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but torch.cuda.is_available() is False"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        assert torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
        fp = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        assert fp in (None, "ieee", "none"), f"float32 matmuls must be IEEE (got {fp})"
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def device_info(device):
    dev = torch.device(device)
    out = dict(device=str(dev), torch=torch.__version__, cpu=cpu_model(), threads=torch.get_num_threads())
    if dev.type == "cuda":
        i = dev.index if dev.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        out.update(gpu=p.name, capability=f"{p.major}.{p.minor}", gpu_mem_gb=round(p.total_memory / 2 ** 30, 2),
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
    else:
        out.update(gpu=None)
    return out


def hw_label(device):
    """The hardware part of a cache key: the GPU model on CUDA, the CPU model on CPU."""
    i = device_info(device)
    return i["gpu"] or i["cpu"]


# ── moving tensors ────────────────────────────────────────────────────────────
def to_cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, tuple):
        return tuple(to_cpu(y) for y in x)
    if isinstance(x, list):
        return [to_cpu(y) for y in x]
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, tuple):
        return tuple(to_dev(y, dev) for y in x)
    if isinstance(x, list):
        return [to_dev(y, dev) for y in x]
    return x


def model_device(m):
    for p in m.parameters():
        return p.device
    return torch.device("cpu")


# ── the model wrapper ─────────────────────────────────────────────────────────
_CLASSES = {}


def _noisy_forward(self, tokens, tau=None, track_sat=False):
    """MultiBDH.forward with the gate noise drawn on CPU from the model's own generator, then moved (the only change)."""
    from test_multilayer_binding import MultiBDH  # noqa: F401
    from test_instrument_v2 import Instrument
    from bdh import BDH
    v = self.embed(tokens)
    vg = F.layer_norm(v, (v.shape[-1],)) if self.gate_ln else v
    gr, gw = Instrument.gates(self, tokens, vg, tau)
    if self.gate_noise > 0 and self.training:
        eps = torch.randn(gr.shape, generator=self.noise_gen, dtype=gr.dtype, device="cpu").to(gr.device)
        gr = F.softmax(torch.log(gr) + self.gate_noise * eps, dim=-1)
        gw = gr
    self.attn.G = (None if self.gate_kind == "none" else torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1))
    if self.conv is None:
        logits, _ = BDH.forward(self, tokens)
    else:
        logits = self.forward_conv(tokens)
    if self.gate_to_readout:
        h = self._h_seq.detach() if self.readout_sg else self._h_seq
        logits = logits + (h @ self.W_ro.T) @ self.lm_head
    return logits, None, gr, gw


def device_class(cls):
    if getattr(cls, "_dev_wrapped", False):
        return cls
    if cls in _CLASSES:
        return _CLASSES[cls]

    def forward(self, *args, **kw):
        dev = self._dev
        args = tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)
        with torch.device(dev):
            if getattr(self, "gate_noise", 0.0) > 0 and self.training and dev.type != "cpu":
                out = _noisy_forward(self, *args, **kw)
            else:
                out = cls.forward(self, *args, **kw)
        if self._cpu_io and dev.type != "cpu":
            out = to_cpu(out)
            if getattr(self, "diag", None) is not None:
                self.diag = to_cpu(self.diag)
        return out

    # the same __name__ / __qualname__ as the project class: hooks that test type(m).__name__ (test_stream_curriculum's
    # CHECK 97) still match; `type(m) is X` does not, so every conversion (to_local, to_delta, to_reg3, nudge) must run
    # BEFORE to_device (the onset_run wrapper does exactly that: it moves what make_model returns)
    new = type(cls.__name__, (cls,), dict(forward=forward, _dev_wrapped=True, _base_cls=cls,
                                          __qualname__=cls.__qualname__, __module__=cls.__module__))
    _CLASSES[cls] = new
    return new


def to_device(m, device, cpu_io=True):
    dev = torch.device(device)
    m.to(dev)
    m.__class__ = device_class(type(m))
    m._dev = dev
    m._cpu_io = bool(cpu_io)
    return m


def unwrap_class(m):
    """The project class underneath (for code that tests type(m) is X)."""
    return getattr(type(m), "_base_cls", type(m))


def on_device(fn):
    """f(model, *args, **kw) run on the model's device; tensor arguments moved there, tensor results moved to CPU."""
    @functools.wraps(fn)
    def g(model, *args, **kw):
        dev = model_device(model)
        if dev.type == "cpu":
            return fn(model, *args, **kw)
        args = tuple(to_dev(a, dev) for a in args)
        kw = {k: to_dev(v, dev) for k, v in kw.items()}
        io = getattr(model, "_cpu_io", None)
        if io is not None:
            model._cpu_io = False
        try:
            with torch.device(dev):
                out = fn(model, *args, **kw)
        finally:
            if io is not None:
                model._cpu_io = io
        return to_cpu(out)
    g._on_device = True
    return g


# ── running a main-line test's own code on a device ──────────────────────────
# (module, attribute, kind): kind "onset" = test_binding_onset.onset_run or a by-name binding of it; "model_fn" = a
# function f(model, ...) to run on the model's device (on_device); "split_op" = a split / reset op whose noise is drawn
# on CPU and must reach the device. Filled from the audit (explore_out/I/gpu/audit.md).
PATCHES = [
    ("test_binding_onset", "onset_run", "onset"),              # every harness run (attempt / run_one resolve it at call time)
    ("test_window_gate", "split_op", "split_op"),              # WIN3_SPLIT_D8's row copy (test_split_copy reaches it too)
    ("explore_i_common", "split_op", "split_op"),              # ICMC's split (S75)
]


def _onset_wrapper(orig, dev):
    @functools.wraps(orig)
    def onset_run(task, make_model, seed, *a, **kw):
        return orig(task, lambda: to_device(make_model(), dev), seed, *a, **kw)
    return onset_run


def noise_on_device(nf):
    """noise_for(w, seed) -> (sd, n1, n2) drawn exactly as nf draws them (from the row's CPU copy, the CPU generator), then
    moved to the row's device."""
    def noise_dev(w, seed):
        sd, n1, n2 = nf(w.cpu(), seed)
        return sd.to(w.device) if torch.is_tensor(sd) else sd, n1.to(w.device), n2.to(w.device)
    return noise_dev


def _split_wrapper(orig):
    """op(m, o, cs, c0, seed): the noise is drawn exactly as the op draws it (CPU generator) and then moved."""
    @functools.wraps(orig)
    def op(m, o, *a, **kw):
        dev = model_device(m)
        if dev.type == "cpu":
            return orig(m, o, *a, **kw)
        mod = sys.modules[orig.__module__]
        nf = getattr(mod, "noise_for", None)
        if nf is None:
            return orig(m, o, *a, **kw)
        mod.noise_for = noise_on_device(nf)
        try:
            return orig(m, o, *a, **kw)
        finally:
            mod.noise_for = nf
    return op


@contextlib.contextmanager
def device_scope(device, patches=None):
    """Patch the run path for `device` (see the module docstring); restore everything on exit."""
    dev = setup(device)
    saved = []
    if dev.type != "cpu":
        for modname, attr, kind in (PATCHES if patches is None else patches):
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            if not hasattr(mod, attr):
                continue
            orig = getattr(mod, attr)
            if kind == "onset":
                new = _onset_wrapper(orig, dev)
            elif kind == "model_fn":
                new = on_device(orig)
            elif kind == "split_op":
                new = _split_wrapper(orig)
            else:
                raise ValueError(kind)
            saved.append((mod, attr, orig))
            setattr(mod, attr, new)
    try:
        yield dev
    finally:
        for mod, attr, orig in reversed(saved):
            setattr(mod, attr, orig)
