#!/usr/bin/env python
"""
explore_continue_child.py — EXPLORATORY, not a result. Batch 20's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e) for S68 (explore_continue). It is batch 19's child (explore_scale_child,
imported read-only, its run path unchanged) with three additions, each read from the arm's spec:

  marks        [steps]: at each listed evaluation step, the per-stream held-out accuracy (test_scale_axes.stream_acc on
               test_channel_binding.eval_batch, the measurement behind the run path's end statistics);
  sha_at       [steps]: at each listed evaluation step, the SHA-1 of every weight (explore_scale_child.w_sha);
  halt_once_at step: the run halts right after its checkpoint at this step, writes a marker next to the checkpoint and
               replaces its own process (os.execv of the same command line) with a new one, which resumes from that
               checkpoint (the marker stops a second halt) — the user's "resumed run" for the 44,400 reproduction;
  ckpt_until   step: no checkpoint is written after this step (the reproduction's FRESH run: its 43,200 -> 44,400 updates
               are always computed in one process from a state at or before this step, never resumed at 43,200).

The two measurements are wrapped by batch 19's recorder (explore_b19_model.REC), so a run resumed from a checkpoint
replays them exactly as it replays every other measurement. They read the model without a gradient and touch no random
state (CHECKed: a run with them equals the same run without them, bit for bit). Everything else — the recipe, the
evaluation every 1200, the early stop — is explore_scale_child.run as batch 19 ran it.
"""

import os
import sys

import torch

import explore_b19_model as b19
import explore_scale_child as esc
import test_scale_axes as tsa
from test_channel_binding import eval_batch

STASH = {}
CTX = {}


def _stream_acc(model, task):
    with torch.no_grad():
        return [float(v) for v in tsa.stream_acc(model, task, eval_batch(task))]


def _sha(model):
    return esc.w_sha(model)


MEAS_ACC = b19.REC.wrap("s68.stream_acc", _stream_acc)
MEAS_SHA = b19.REC.wrap("s68.w_sha", _sha)

_ORIG_BUILD = esc.build_rc
_ORIG_SAVE = b19.save_ckpt


def build_rc(spec, p, holder, info):
    rc, ctx = _ORIG_BUILD(spec, p, holder, info)
    marks = set(spec.get("marks") or ())
    shas = set(spec.get("sha_at") or ())
    task = esc.task_of(spec)
    STASH["info"] = info
    info["marks"], info["shas"] = [], []

    def rc2(kw):
        kw = rc(kw)
        c0 = kw.get("check")

        def check(m, step, d):
            if c0 is not None:
                c0(m, step, d)
            if step in marks:
                info["marks"].append([step, MEAS_ACC(m, task)])
            if step in shas:
                info["shas"].append([step, MEAS_SHA(m)])
        kw["check"] = check
        return kw
    return rc2, ctx


def save_ckpt(path, d):
    lim = CTX.get("ckpt_until")
    if lim is not None and d["step"] > lim:
        return
    _ORIG_SAVE(path, d)


esc.build_rc = build_rc
b19.save_ckpt = save_ckpt


def _marker(ck):
    return None if not ck else ck + ".halted"


def run(p):
    """p: spec (with marks / sha_at / halt_once_at / ckpt_until), seed, iters, ckpt."""
    spec = p["spec"]
    ck = p.get("ckpt")
    mk = _marker(ck)
    halted = None
    if mk and os.path.exists(mk):
        with open(mk) as f:
            halted = int(f.read().strip())
    q = dict(p)
    h = spec.get("halt_once_at")
    if h and halted is None and ck:
        q["halt_at"] = h
    CTX["ckpt_until"] = spec.get("ckpt_until")
    out = esc.run(q)
    if out.get("halted"):
        with open(mk, "w") as f:
            f.write(str(out["step"]))
        args = open("/proc/self/cmdline", "rb").read().split(b"\0")[:-1]
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + [a.decode() for a in args[1:]])
    info = STASH.get("info") or {}
    out["marks"] = info.get("marks")
    out["shas"] = info.get("shas")
    out["halted_once_at"] = halted
    if out.get("ok") and mk and os.path.exists(mk):
        os.remove(mk)
    return out


# ── CHECKs (run in the first segment; the parent compares) ──────────────────
def plain(p):
    """explore_scale_child.run without any batch-20 addition (no marks, no sha_at, no halt, no checkpoint)."""
    spec = {k: v for k, v in p["spec"].items() if k not in ("marks", "sha_at", "halt_once_at", "ckpt_until")}
    CTX["ckpt_until"] = None
    return esc.run(dict(spec=spec, seed=p["seed"], iters=p["iters"], ckpt=None))


def save_guard_check(p):
    """save_ckpt writes nothing after ckpt_until and writes at or before it."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.pt")
        CTX["ckpt_until"] = 1200
        save_ckpt(path, dict(step=2400, x=1))
        after = os.path.exists(path)
        save_ckpt(path, dict(step=1200, x=1))
        at = os.path.exists(path) and open(path + ".step").read().strip() == "1200"
        CTX["ckpt_until"] = None
    return dict(ok=bool(not after and at), wrote_after=after, wrote_at=at)


def constants(p):
    return esc.constants(p)
