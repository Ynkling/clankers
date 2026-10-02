#!/usr/bin/env python
"""
explore_muon_recipe.py — EXPLORATORY, not a result. Screen S25 (batch 9): arm A and the HINGE recipe
under Muon (orthogonalized momentum, no per-weight second moment) instead of Adam.

BACKGROUND (docstring)
- test_slow_start (main branch, both machines): slow memory + hinge (HINGE) without restarts: S=2
  P=4 39/40, P=8 37/40, S=4 k=16 35/40 pooled; S=8 0/20. On L, 4 of 10 HINGE_D8 runs split the 8
  streams into two channels of four and stalled near 0.2.
- Batches 6-7: a gradient far above a weight's running gradient gives Adam a fixed-size step
  (sign-like) and inflates the second moment for thousands of updates (W_g x1e6 at 4800). The
  hinge's effect partly runs through this. Muon has no per-weight second moment.
- Ideas from the literature: Langevin noise for saddle escape (Muon meets Tamed Langevin, arXiv
  2610.02158); balanced routing (Sinkhorn/BASE layers; usage-balance losses) against collapse.

THE OPTIMIZER (MUON). This torch (2.14) has torch.optim.Muon; it accepts 2-D parameters only. Its
update (torch.optim._muon._single_tensor_muon, line for line, with torch's own Newton-Schulz
_zeropower_via_newtonschulz and lr adjustment _adjust_lr) is applied here by SliceMuon to every 2-D
weight and to each head slice of a 3-D weight: momentum 0.95, Nesterov, 5 Newton-Schulz steps with
torch's coefficients (3.4445, -4.775, 2.0315) in bfloat16, eps 1e-7, the default "original" lr
adjustment (lr x sqrt(max(1, rows/cols))), weight decay 0 (the spec names none; Adam here has none;
torch's default is 0.1). Muon: decoder, encoder and encoder_v (per head slice), lm_head, W_in, W_h,
W_g. Adam (torch.optim.Adam, lr 1e-3 = test_channel_binding.SUB_LR): the embedding (the model has no
conv, bias or 1-D parameter at arm A). The combined optimizer (MuonAdam) replaces onset_run's Adam
for the run only (torch.optim.Adam is swapped in its module for the duration of run_one and restored
after; onset_run's loop is untouched); run_one's param_groups callback supplies the groups.

LR SELECTION (before any screen run; stored in explore_out/muon_recipe_lrsel_results.json): the
perfect gate (test_short_conv's ceiling arm, S=2, P=4, k=2, grouped) under MUON at Muon lr 0.005,
0.01, 0.02, seeds 160-161, 6000 steps (Adam's lr 1e-3, no slow phase); the smallest lr at which both
seeds bind is used for both arms below and printed; if none binds, S25 is UNTESTED.

THE ARMS (S=2, P=4, k=2, grouped, no conv, MAX_ITERS 24000; seeds 160-179)
  MUON_A      arm A under MUON (Muon lr as selected for every Muon weight, Adam 1e-3 for the
              embedding, throughout). Paired with X's recorded arm A (7/20 DISCOVERED on these seeds).
  MUON_HINGE  S13's SLOW_HINGE recipe under MUON: S13's HingeBDH (LAMBDA 1.0 x [relu(eta2_index -
              0.2) + relu(eta2_half - 0.2)], TAU 0.2); the slow phase scales every non-gate learning
              rate by 0.1 for updates 1-2400 (Muon and Adam groups alike: the non-gate Muon weights at
              0.1 x Muon lr, the embedding at 1e-4), then back to full after the evaluation at 2400 (as
              explore_slow_mem); the gate W_in, W_h, W_g at the Muon lr throughout. Paired with S13's
              SLOW_HINGE (18/20 on these seeds).
Logged per run: the gate's update norm per step (||delta of (W_in, W_h, W_g)||, all three together)
for updates 1-2400: median and max (and the first and last values).

RULES (fixed before any run): MUON_A vs X's arm A, MUON_HINGE vs SLOW_HINGE, outcome DISCOVERED
(McNemar printed). "the recipe survives Muon" if MUON_HINGE >= 15/20; "it does not" if MUON_HINGE <=
10/20; otherwise neither. Failure classes and hinge firings printed.

CHECKS: the optimizer groups cover every parameter exactly once (MUON_A, MUON_HINGE, the ceiling);
SliceMuon equals torch.optim.Muon (weight decay 0) bit for bit on 2-D weights (5 steps); the swap
reaches onset_run: with every parameter in an Adam group the MuonAdam path reproduces X's arm A record
(seed 160, 2400 steps) bit for bit; MUON_HINGE's lrs per group at updates 1, 2400 and 2401 (an
optimizer log through its real run path, forward and evaluation stubbed); with the hinge's TAU at
1.0, MUON_HINGE equals a MUON run with the slow schedule alone bit for bit (curves and weights, 1200
steps); Muon's update on a probe matrix (Gaussian, the encoder head slice's shape 32 x 256, generator
seed 0) has all singular values within [0.5, 1.5] after Newton-Schulz (scaled as implemented: torch's
Frobenius normalisation, before the lr and its adjustment). The pre-registered range was [0.7, 1.3];
the first dry run (explore_out/batch9_dry_check_failed.log) measured [0.6835, 1.0490]: these
coefficients map 1 to 0.70 and 1.05 to 0.68 by design. Widened to [0.5, 1.5], the range Keller Jordan
gives for them (S' ~ Uniform(0.5, 1.5)), by the user's decision before any run.
"""

import contextlib
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr

import explore_common as ec
import explore_common2 as c2
import explore_slow_hinge as s13
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_discovery import GATE_PARAMS
from test_binding_onset import EVAL_EVERY

NAME = "muon_recipe"
IDEA = "arm A and the HINGE recipe with Muon (orthogonalized momentum) in place of Adam"
SOURCE = ("Muon (Jordan et al.; torch.optim.Muon): no per-weight second moment; batches 6-7 (Adam's sign-like "
          "step and inflated second moment under the hinge)")
CHANGE = ("Muon on every 2-D weight (per head slice of 3-D), Adam 1e-3 on the embedding; MUON_HINGE: S13's hinge, "
          "the slow phase scaling every non-gate lr by 0.1 for updates 1-2400")
PAIRING = ("seeds 160-179: MUON_A with X's recorded arm A, MUON_HINGE with S13's SLOW_HINGE (same initial parameters "
           "and batches)")
_ADAM = torch.optim.Adam
GATE = tuple(GATE_PARAMS)
ADAM_NAMES = ("embed.weight",)
MUON_KW = dict(momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7, ns_steps=5,
               weight_decay=0.0, adjust_lr_fn=None)
LR_CANDIDATES = (0.005, 0.01, 0.02)
LRSEL_SEEDS = (160, 161)
LRSEL_ITERS = 6000
WARM = 2400
SLOW_SCALE = 0.1
LOG_UPD = 2400
SV_LO, SV_HI = 0.5, 1.5                    # pre-registered 0.7, 1.3; widened by the user after the first dry run
SEEDS = tuple(range(160, 180))
SURVIVES, FAILS = 15, 10
CEIL = tsc._BASE["ceiling"]
STATE = dict(muon_lr=None, untested=False, choice_note="")

ARMS = {
    "MUON_A": dict(ec.ARM_A, key="MUON_A", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1, slow=False,
                   hinge=False, label="arm A under Muon (Adam 1e-3 on the embedding)", muon_lr=None,
                   sched="Muon lr (selected) on every 2-D weight, Adam 0.001 on the embedding, throughout"),
    "MUON_HINGE": dict(ec.ARM_A, key="MUON_HINGE", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1, slow=True,
                       hinge=True, label="S13's SLOW_HINGE recipe under Muon", muon_lr=None,
                       sched="Muon lr (selected): gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates "
                             "1-2400; Adam on the embedding 0.0001 for 1-2400, 0.001 after; + 1.0 * [relu(eta2_index "
                             "- 0.2) + relu(eta2_half - 0.2)] on each training batch"),
}


# ── the optimizer ────────────────────────────────────────────────────────────
class SliceMuon(torch.optim.Optimizer):
    """torch.optim.Muon's update (_single_tensor_muon, line for line, torch's own Newton-Schulz and lr
    adjustment) on every 2-D weight and on each 2-D slice of a 3-D weight (one momentum buffer per
    weight, orthogonalised per slice)."""

    def __init__(self, params, lr, momentum, nesterov, ns_coefficients, eps, ns_steps, weight_decay, adjust_lr_fn):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_coefficients=ns_coefficients,
                                      eps=eps, ns_steps=ns_steps, weight_decay=weight_decay,
                                      adjust_lr_fn=adjust_lr_fn))
        for g in self.param_groups:
            for p in g["params"]:
                assert p.ndim in (2, 3), p.shape

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buf = st["momentum_buffer"]
                buf.lerp_(grad, 1 - mom)
                update = grad.lerp(buf, mom) if group["nesterov"] else buf
                pairs = [(p, update)] if p.ndim == 2 else [(p[i], update[i]) for i in range(p.shape[0])]
                for w, u in pairs:
                    o = _zeropower_via_newtonschulz(u, group["ns_coefficients"], group["ns_steps"], group["eps"])
                    alr = _adjust_lr(lr, group["adjust_lr_fn"], w.shape)
                    w.mul_(1 - lr * wd)
                    w.add_(o, alpha=-alr)
        return None


class MuonAdam:
    """Adam groups (kind 'adam') on torch.optim.Adam, Muon groups (kind 'muon') on SliceMuon; the group
    dicts are the caller's (an lr switch on them reaches the right optimizer). Logs the gate's update norm
    for the first LOG_UPD steps (the parameters of groups tagged 'gate'), and optionally each group's lr."""

    def __init__(self, groups, lr):
        groups = [g for g in groups if len(g["params"])]
        ad = [g for g in groups if g["kind"] == "adam"]
        mu = [g for g in groups if g["kind"] == "muon"]
        self.adam = _ADAM(ad, lr=lr) if ad else None
        self.muon = SliceMuon(mu, lr=lr, **MUON_KW) if mu else None
        self.opts = [o for o in (self.adam, self.muon) if o is not None]
        self.param_groups = groups
        self.gate = [p for g in groups if g.get("tag") == "gate" for p in g["params"]]
        self.n, self.upd, self.lr_log = 0, [], None

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.n += 1
        if self.lr_log is not None:
            self.lr_log.append([g["lr"] for g in self.param_groups])
        log = self.n <= LOG_UPD and self.gate
        before = [p.detach().clone() for p in self.gate] if log else None
        for o in self.opts:
            o.step()
        if log:
            self.upd.append(float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(self.gate, before)))))


@contextlib.contextmanager
def muon_in_onset_run(holder):
    """onset_run builds torch.optim.Adam(param_groups(model), lr=lr); for the duration of one run that
    name builds a MuonAdam on the same groups instead."""
    def factory(params, lr=1e-3, **kw):
        assert not kw, kw
        opt = MuonAdam(list(params), lr)
        if holder.get("log_lr"):
            opt.lr_log = []
        holder.setdefault("opts", []).append(opt)
        return opt
    torch.optim.Adam = factory
    try:
        yield
    finally:
        torch.optim.Adam = _ADAM


def make_groups(holder, muon_lr, slow, all_adam=False):
    """onset_run's param_groups: gate (W_in, W_h, W_g; Muon), the other Muon weights, the embedding
    (Adam). slow: every non-gate group at SLOW_SCALE x its lr until the switch. all_adam: every
    parameter in Adam groups at 1e-3 (the plumbing CHECK)."""
    def param_groups(model):
        named = dict(model.named_parameters())
        gate = [n for n in GATE if n in named]
        rest_m = [n for n in named if n not in GATE and n not in ADAM_NAMES]
        adam = [n for n in named if n in ADAM_NAMES]
        assert all(named[n].ndim in (2, 3) for n in gate + rest_m)
        sc = SLOW_SCALE if slow else 1.0
        if all_adam:
            spec = [("gate", gate, "adam", ec.SUB_LR, ec.SUB_LR), ("rest", rest_m + adam, "adam", ec.SUB_LR, ec.SUB_LR)]
        else:
            spec = [("gate", gate, "muon", muon_lr, muon_lr), ("rest", rest_m, "muon", sc * muon_lr, muon_lr),
                    ("rest", adam, "adam", sc * ec.SUB_LR, ec.SUB_LR)]
        holder["groups"] = [dict(params=[named[n] for n in ns], lr=lr0, kind=kind, tag=tag, lr_after=lr1, names=ns)
                            for tag, ns, kind, lr0, lr1 in spec if ns]
        return holder["groups"]
    return param_groups


def make_check(holder):
    def check(m, step, data):
        if step >= WARM:
            for g in holder["groups"]:
                g["lr"] = g["lr_after"]
    return check


def run_path(a, seed, iters, keep=None, muon_lr=None, tau=s13.TAU, model="arm", all_adam=False, task=None):
    """One run on run_one with MuonAdam in onset_run. model: 'arm' (build), 'hinge' (S13's HingeBDH at tau)
    or 'ceil' (the perfect gate)."""
    holder = {}
    task = ec.TASK if task is None else task
    mlr = a["muon_lr"] if muon_lr is None else muon_lr
    if model == "hinge":
        b, sf = (lambda aa, s: (lambda: s13.make_model(aa, s, tau))), s13.stats_fn
    else:
        b, sf = (lambda aa, s: (lambda: build(task, aa, ARCH, s))), (tsc.conv_stats if model == "ceil" else c2.stats_b2)
    with muon_in_onset_run(holder):
        rec = ec.run_one(a, seed, iters, check=make_check(holder) if a.get("slow") else None, keep=keep, task=task,
                         stats_fn=sf, grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=b,
                         run_kw=dict(param_groups=make_groups(holder, mlr, a.get("slow", False), all_adam)))
    opts = holder.get("opts", [])
    if rec.get("ok"):
        u = opts[0].upd if opts else []
        rec["muon_lr"] = mlr
        rec["opt_groups"] = [dict(tag=g["tag"], kind=g["kind"], names=g["names"], lr_end=g["lr"]) for g in holder["groups"]]
        rec["gate_upd"] = (dict(n=len(u), median=statistics.median(u), max=max(u), first=u[:3], last=u[-1])
                           if u else None)
        rec["n_opts_built"] = len(opts)
    return rec, holder


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"], model="hinge" if a["hinge"] else "arm")[0]


def segments(arm):
    a = ARMS[arm]
    mk = (lambda: s13.make_model(a, 0)) if a["hinge"] else (lambda: build(ec.TASK, a, ARCH, 0))
    return [(a["iters"], ec.TASK, mk, a["lr"], s13.stats_fn if a["hinge"] else c2.stats_b2)]


# ── LR selection ─────────────────────────────────────────────────────────────
def lrsel_name():
    return NAME + "_lrsel"


def _lrsel_worker(job):
    torch.set_num_threads(ec.THREADS)
    t = time.time()
    a = dict(CEIL, lr=ec.SUB_LR, slow=False)
    rec, _ = run_path(a, job["seed"], job["iters"], muon_lr=job["lr"], model="ceil")
    rec.update(secs_wall=time.time() - t, lr=ec.SUB_LR, muon_lr=job["lr"])
    return job, rec


def select_lr(workers, dry=False, run=True):
    """The LR selection (stored; a resume reuses it). Sets STATE and the arms' muon_lr (or UNTESTED)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    seeds = LRSEL_SEEDS[:1] if dry else LRSEL_SEEDS
    iters = min(LRSEL_ITERS, c2.DRY.get("iters", LRSEL_ITERS)) if dry else LRSEL_ITERS
    st = ec.load_store(lrsel_name())
    st["meta"].setdefault("provenance", ec.provenance())
    st["meta"].update(candidates=list(LR_CANDIDATES), seeds=list(seeds), iters=iters, arm=CEIL["key"], dry=dry)
    jobs = [dict(lr=lr, seed=s, iters=iters) for lr in LR_CANDIDATES for s in seeds if f"{lr}|{s}" not in st["runs"]]
    print(f"  S25 LR SELECTION ({ec.BANNER}): perfect gate under MUON, Muon lr {list(LR_CANDIDATES)}, seeds "
          f"{list(seeds)}, {iters} steps, Adam lr {ec.SUB_LR:g} on the embedding; {len(jobs)} runs to do "
          f"({len(LR_CANDIDATES) * len(seeds) - len(jobs)} stored)", flush=True)
    if jobs and run:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("fork"),
                                 initializer=ec._worker_init) as ex:
            futs = [ex.submit(_lrsel_worker, j) for j in jobs]
            for n, f in enumerate(as_completed(futs), 1):
                job, rec = f.result()
                st["runs"][f"{job['lr']}|{job['seed']}"] = rec
                ec.save_store(lrsel_name(), st)
                print(f"  [lrsel {n}/{len(jobs)}] {(time.time() - t0) / 60:5.1f} min  Muon lr {job['lr']:g} seed "
                      f"{job['seed']} (Adam lr {ec.SUB_LR:g}, {iters} steps): {ec.line(rec)}", flush=True)
    ok_lr = []
    for lr in LR_CANDIDATES:
        rs = [st["runs"].get(f"{lr}|{s}") for s in seeds]
        b = [ec.bound(r) for r in rs]
        print(f"    Muon lr {lr:<6g} " + "  ".join(
            f"seed {s}: {('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'} (acc {r['acc']:.3f})"
            if r and r.get("ok") else f"seed {s}: {ec.tag(r)}" for s, r in zip(seeds, rs)))
        if all(b):
            ok_lr.append(lr)
    if ok_lr:
        STATE.update(muon_lr=ok_lr[0], untested=False, choice_note=f"smallest lr at which every seed binds: {ok_lr[0]:g}")
    elif dry:
        STATE.update(muon_lr=LR_CANDIDATES[0], untested=False,
                     choice_note=f"DRY: no lr bound in the shortened runs; {LR_CANDIDATES[0]:g} used to exercise the code")
    else:
        STATE.update(muon_lr=None, untested=True, choice_note="no lr bound both seeds: S25 UNTESTED")
    for a in ARMS.values():
        a["muon_lr"] = STATE["muon_lr"]
        if STATE["untested"]:
            a["seeds"] = ()
        else:
            a["sched"] = a["sched"].replace("Muon lr (selected)", f"Muon lr {STATE['muon_lr']:g} (selected)")
    print(f"  S25 LR CHOICE: {STATE['choice_note']}", flush=True)
    return STATE


def muon_overhead(n=50):
    """ms/step of arm A under MUON vs plain Adam (onset_run, 1 thread), for the projection note."""
    torch.set_num_threads(ec.THREADS)
    a = dict(ARMS["MUON_A"], muon_lr=LR_CANDIDATES[0])
    out = {}
    for lab in ("adam", "muon"):
        t = time.time()
        if lab == "adam":
            ec.run_one(a, 0, n, task=ec.TASK, stats_fn=lambda *x: dict(key_cos=0, val_cos=0, ctx_cos=0), lr=a["lr"],
                       builder=lambda aa, s: (lambda: build(ec.TASK, aa, ARCH, s)), eval_every=10 ** 9)
        else:
            holder = {}
            with muon_in_onset_run(holder):
                ec.run_one(a, 0, n, task=ec.TASK, stats_fn=lambda *x: dict(key_cos=0, val_cos=0, ctx_cos=0),
                           lr=a["lr"], builder=lambda aa, s: (lambda: build(ec.TASK, aa, ARCH, s)), eval_every=10 ** 9,
                           run_kw=dict(param_groups=make_groups(holder, a["muon_lr"], False)))
        out[lab] = (time.time() - t) / n * 1e3
    return out


# ── CHECKs ───────────────────────────────────────────────────────────────────
def cover(groups, model):
    ids = [id(p) for g in groups for p in g["params"]]
    allp = {id(p) for p in model.parameters()}
    return len(ids) == len(set(ids)) and set(ids) == allp


def check():
    ok = True
    rows = []
    # (1) groups cover every parameter exactly once
    cov = []
    for lab, a, mk, slow in (("MUON_A", ARMS["MUON_A"], lambda: build(ec.TASK, ARMS["MUON_A"], ARCH, 160), False),
                             ("MUON_HINGE", ARMS["MUON_HINGE"], lambda: s13.make_model(ARMS["MUON_HINGE"], 160), True),
                             ("ceiling", CEIL, lambda: build(ec.TASK, CEIL, ARCH, 160), False)):
        h = {}
        m = mk()
        gs = make_groups(h, LR_CANDIDATES[0], slow)(m)
        cov.append((lab, cover(gs, m), [(g["tag"], g["kind"], g["names"]) for g in gs]))
    rows.append(("the optimizer groups cover every parameter exactly once: "
                 + "; ".join(f"{lab} {gs}" for lab, _, gs in cov), all(c for _, c, _ in cov)))
    # (2) SliceMuon = torch.optim.Muon (weight decay 0) on 2-D weights
    g = torch.Generator().manual_seed(1)
    w1 = [torch.randn(s, generator=g).requires_grad_() for s in ((32, 256), (2, 32), (256, 32))]
    w2 = [w.detach().clone().requires_grad_() for w in w1]
    o1 = SliceMuon(w1, lr=0.01, **MUON_KW)
    o2 = torch.optim.Muon(w2, lr=0.01, **MUON_KW)
    for _ in range(5):
        gr = [torch.randn(w.shape, generator=g) for w in w1]
        for w, x in zip(w1, gr):
            w.grad = x.clone()
        for w, x in zip(w2, gr):
            w.grad = x.clone()
        o1.step()
        o2.step()
    rows.append(("SliceMuon equals torch.optim.Muon (weight decay 0) bit for bit on 2-D weights (3 shapes, 5 steps)",
                 all(torch.equal(a_, b_) for a_, b_ in zip(w1, w2))))
    # (3) the swap reaches onset_run: all-Adam MuonAdam = X's arm A record (repro seed and steps)
    r_ad, h_ad = run_path(dict(ec.ARM_A, lr=ec.SUB_LR, slow=False, muon_lr=None), 160, 2 * EVAL_EVERY, all_adam=True,
                          muon_lr=0.0)
    xrec = ec.recorded("A")[160]
    want = [c for c in xrec["curve"] if c[0] <= 2 * EVAL_EVERY]
    rows.append((f"the swap reaches onset_run: with every parameter in Adam groups the MuonAdam path reproduces X's arm A "
                 f"record (seed 160, {2 * EVAL_EVERY} steps; {len(h_ad.get('opts', []))} optimizer built): "
                 f"{r_ad['curve']} vs {want}", r_ad["curve"] == want and len(h_ad.get("opts", [])) == 1))
    # (4) MUON_HINGE's lrs at updates 1, 2400, 2401 (stubbed forward and evaluation, as explore_slow_mem's check)
    saved = (tbo.evaluate, tbo.logits_of)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    holder, hh = {"log_lr": True}, {}
    mlr = LR_CANDIDATES[0]
    a = ARMS["MUON_HINGE"]
    try:
        with muon_in_onset_run(holder):
            r_lr = ec.run_one(a, 160, WARM + 1, check=make_check(hh), task=ec.TASK, stats_fn=c2.stats_b2,
                              grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                              builder=lambda aa, s: (lambda: s13.make_model(aa, s)),
                              run_kw=dict(param_groups=make_groups(hh, mlr, True)))
    finally:
        tbo.evaluate, tbo.logits_of = saved
    opt = holder["opts"][0]
    lg = opt.lr_log
    tags = [(gg["tag"], gg["kind"]) for gg in hh["groups"]]
    want1 = [mlr, SLOW_SCALE * mlr, SLOW_SCALE * ec.SUB_LR]
    want2 = [mlr, mlr, ec.SUB_LR]
    lr_ok = (len(lg) == WARM + 1 and lg[0] == want1 and lg[WARM - 1] == want1 and lg[WARM] == want2
             and tags == [("gate", "muon"), ("rest", "muon"), ("rest", "adam")])
    rows.append((f"MUON_HINGE's lrs per group {tags}: update 1 {lg[0]}, {WARM} {lg[WARM - 1]}, {WARM + 1} {lg[WARM]} "
                 f"({len(lg)} updates, forward and evaluation stubbed)", lr_ok and r_lr.get("ok")))
    # (5) TAU = 1.0: MUON_HINGE = a MUON run with the slow schedule alone
    k1, k2 = {"at": EVAL_EVERY}, {"at": EVAL_EVERY}
    r1, _ = run_path(a, 160, EVAL_EVERY, keep=k1, muon_lr=mlr, tau=1.0, model="hinge")
    r2, _ = run_path(dict(a, hinge=False), 160, EVAL_EVERY, keep=k2, muon_lr=mlr, model="arm")
    same = r1["curve"] == r2["curve"] and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"])
    fired = r1["end"].get("hinge_counts")
    rows.append((f"with the hinge's TAU at 1.0, MUON_HINGE equals a MUON run with the slow schedule alone bit for bit "
                 f"(seed 160, {EVAL_EVERY} steps, Muon lr {mlr:g}; curves {r1['curve']} vs {r2['curve']}; weights "
                 f"equal; hinge on {fired})", same and fired and fired[0] == 0))
    # (6) singular values of the Newton-Schulz output on the probe (last: the pre-registered range)
    gp = torch.Generator().manual_seed(0)
    probe = torch.randn((32, 256), generator=gp)
    o = _zeropower_via_newtonschulz(probe, MUON_KW["ns_coefficients"], MUON_KW["ns_steps"], MUON_KW["eps"]).float()
    sv = torch.linalg.svdvals(o)
    extra = []
    for shp in ((256, 32), (32, 32), (2, 32), (32, 22)):
        oo = _zeropower_via_newtonschulz(torch.randn(shp, generator=gp), MUON_KW["ns_coefficients"],
                                         MUON_KW["ns_steps"], MUON_KW["eps"]).float()
        s_ = torch.linalg.svdvals(oo)
        extra.append(f"{shp} [{float(s_.min()):.3f}, {float(s_.max()):.3f}]")
    rows.append((f"Muon's update on the probe (Gaussian 32 x 256, seed 0) after Newton-Schulz has all singular values "
                 f"within [{SV_LO}, {SV_HI}]: min {float(sv.min()):.4f}, max {float(sv.max()):.4f} (not asserted, other "
                 f"Gaussian shapes: {'; '.join(extra)})", SV_LO <= float(sv.min()) and float(sv.max()) <= SV_HI))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def compare(label, new, old, seeds, fn, what, arm):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    nn = sum(bool(fn(new[s])) for s in both)
    no = sum(bool(fn(old[s])) for s in both)
    print(f"    {what:<13} {arm} {nn:>2}/{len(both)}   {label} {no:>2}/{len(both)}   {arm} only {b}, "
          f"{label} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=nn, old=no, b=b, c=c, p=p)


def rule(n):
    return "the recipe survives Muon" if n >= SURVIVES else ("it does not" if n <= FAILS else "neither")


def lrsel_report():
    st = ec.load_store(lrsel_name())
    print(f"  LR selection (perfect gate under MUON, {st['meta'].get('iters')} steps, seeds {st['meta'].get('seeds')}):")
    for lr in LR_CANDIDATES:
        rs = [(s, st["runs"].get(f"{lr}|{s}")) for s in st["meta"].get("seeds", [])]
        print(f"    Muon lr {lr:<6g} " + "  ".join(
            f"seed {s}: {('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'} (acc {r['acc']:.3f})"
            if r and r.get("ok") else f"seed {s}: {ec.tag(r)}" for s, r in rs))
    print(f"    choice: {STATE['choice_note'] or 'not made in this call'}")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    lrsel_report()
    if STATE["untested"]:
        print("  S25 UNTESTED: no Muon lr bound the perfect gate on both seeds")
        return dict(untested=True, rule="UNTESTED", a=None, h=None)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    xa = ec.recorded("A")
    sh = s13.runs_of((s13.NAME,), "SLOW_HINGE")
    lrs = sorted({r.get("muon_lr") for d in got.values() for r in d.values()})
    print(f"  Muon lr in the records: {lrs}")
    out = {}
    for arm, ref, lab in (("MUON_A", xa, "X arm A"), ("MUON_HINGE", sh, "SLOW_HINGE")):
        runs = got[arm]
        print(f"   arm {arm} (lr {ARMS[arm]['lr']:g}; {c2.sched(ARMS[arm])})")
        print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} {'ROUTED*':>8}  "
              f"{'outcome':<15} {lab:<15} {'gate |dW| 1-2400 med/max':>25} {'hinge on of n':>14}")
        for s in SEEDS:
            r = runs.get(s)
            if r is None:
                print(f"    {s:>4} {ec.tag(store['runs'].get(f'{arm}|{s}')):<70} {ec.tag(ref.get(s)):<15}")
                continue
            m = r["end"].get("margin")
            gu = r.get("gate_upd")
            gus = f"{gu['median']:.2e}/{gu['max']:.2e}" if gu else "--"
            hc = r["end"].get("hinge_counts")
            hcs = f"{hc[0]} of {hc[1]}" if hc else "--"
            print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
                  f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  {ec.tag(r):<15} "
                  f"{ec.tag(ref.get(s)):<15} {gus:>25} {hcs:>14}")
        done = [s for s in SEEDS if s in runs]
        print(f"   {arm} vs {lab}, {len(done)} seeds:")
        d = compare(lab, runs, ref, done, ec.discovered, "DISCOVERED", arm)
        compare(lab, runs, ref, done, ec.bound, "BOUND", arm)
        compare(lab, runs, ref, done, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", arm)
        print(f"    failures {arm} {s13.fails(runs, done)}   {lab} {s13.fails(ref, done)}")
        if arm == "MUON_HINGE":
            hcs = [runs[s]["end"].get("hinge_counts") for s in done]
            on = [h[0] for h in hcs if h]
            print(f"    hinge firings: {sum(on)} batches in total; runs with the hinge on at least once "
                  f"{sum(1 for x in on if x > 0)}/{len(done)}; per run {on}")
        gus = [runs[s]["gate_upd"]["median"] for s in done if runs[s].get("gate_upd")]
        if gus:
            print(f"    gate update norm per step, updates 1-{LOG_UPD}: median over runs of the per-run median "
                  f"{statistics.median(gus):.2e}, of the per-run max "
                  f"{statistics.median(runs[s]['gate_upd']['max'] for s in done if runs[s].get('gate_upd')):.2e}")
        out[arm] = d
    rd = rule(out["MUON_HINGE"]["new"])
    print(f"  RULE S25 (MUON_HINGE DISCOVERED {out['MUON_HINGE']['new']}/{out['MUON_HINGE']['n']}; 'the recipe survives "
          f"Muon' if >= {SURVIVES}/20, 'it does not' if <= {FAILS}/20): {rd}")
    return dict(untested=False, rule=rd, a=out["MUON_A"], h=out["MUON_HINGE"], muon_lr=STATE["muon_lr"])
