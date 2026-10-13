#!/usr/bin/env python
"""
explore_oracle_child.py — EXPLORATORY, not a result. Batch 21's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e), for S69 (explore_oracle_d128). It is batch 19's child
(explore_scale_child, imported read-only, its run path unchanged) with one addition, read from the arm's spec:

  slow  True on a perfect-gate spec (gate "perfect"): the oracle on SLOW instead of the recorded validity recipe. Every
        trainable parameter is at 1e-4 for updates 1-2400 and at 1e-3 after (the switch after the evaluation at 2400:
        explore_b16_main.make_rc "adam" with an empty gate group, so every parameter is in the group the learned arms
        call "rest"). The perfect gate has no parameters, so there is no gate group to keep at 1e-3.

Without `slow` a spec runs exactly as batch 19's child runs it (the recorded validity recipe: one Adam at 1e-3 on every
parameter, no groups, no switch). The model, the task, the path, the evaluation, the early stop, the outcome ("BOUND" /
"not bound") and the checkpoints are explore_scale_child's.
"""

import contextlib

import explore_b16_main as b16
import explore_b19_model as b19
import explore_scale_child as esc
import test_binding_onset as tbo

_ORIG_BUILD = esc.build_rc


def build_rc(spec, p, holder, info):
    if not spec.get("slow"):
        return _ORIG_BUILD(spec, p, holder, info)
    assert spec["gate"] == "perfect" and not spec.get("trigger"), spec
    mult, d, L = spec["size"]

    def on_build(m):
        holder["model"] = m
        holder.setdefault("init", b19.init_scales(m))
    rc0, _ = b16.make_rc("adam", holder, (), tau=None, on_build=on_build, keep=None, groups=True)

    def rc(kw):
        kw = dict(kw)
        kw["builder"] = b19.builder_of(mult, d, L)
        return rc0(kw)

    def rc_wrapped(kw):
        return b19.wrap_kw(rc(kw))
    return rc_wrapped, contextlib.nullcontext()


esc.build_rc = build_rc


def run(p):
    return esc.run(p)


def resume_check(p):
    return esc.resume_check(p)


def constants(p):
    return esc.constants(p)


# ── CHECK (run in the first segment; the parent compares) ───────────────────
def lrs_check(p):
    """Each group's lr at every update through the real run path (forward and evaluation stubbed), 2401 updates, for the
    oracle on SLOW and on the recorded recipe at the spec's size and task."""
    W = b16.WARM
    tbo.onset_run = b19.ORIG_ONSET                     # stub_lrs needs run_one's keep (no checkpoints)
    out = {}
    for name, spec in p["which"].items():
        mult, d, L = spec["size"]

        def rc_of(hd, kp, sp=spec, mult=mult, d=d, L=L):
            rc0, _ = b16.make_rc("adam", hd, (), tau=None, keep=kp, groups=bool(sp.get("slow")))

            def rc(kw):
                kw = dict(kw)
                kw["builder"] = b19.builder_of(mult, d, L)
                return rc0(kw)
            return rc
        h, rec, lg, ids, keep = b16.stub_lrs("k16", esc.base_arm(spec), rc_of)
        mm = keep["model"]
        out[name] = dict(ok=bool(rec.get("ok")), n=len(lg), at1=lg[0], at_w=lg[W - 1], at_w1=lg[W],
                         groups=[(g["tag"], len(g["names"])) for g in h.get("groups", [])],
                         cover=b16.cover(ids, mm) if ids else None, n_embd=mm.config.n_embd, gate_kind=mm.gate_kind,
                         n_trainable=sum(1 for q in mm.parameters() if q.requires_grad))
    return out
