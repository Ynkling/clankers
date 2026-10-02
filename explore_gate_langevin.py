#!/usr/bin/env python
"""
explore_gate_langevin.py — EXPLORATORY, not a result. Screen S26 (batch 9): SLOW_MEM plus decaying
Gaussian (Langevin-style) noise on the gate's weights for the first 2400 updates, no hinge.

BACKGROUND (docstring)
- test_slow_start (main branch, both machines): slow memory + hinge (HINGE) without restarts: S=2
  P=4 39/40, P=8 37/40, S=4 k=16 35/40 pooled; S=8 0/20. On L, 4 of 10 HINGE_D8 runs split the 8
  streams into two channels of four and stalled near 0.2.
- Batches 6-7: a gradient far above a weight's running gradient gives Adam a fixed-size step
  (sign-like) and inflates the second moment for thousands of updates (W_g x1e6 at 4800). The
  hinge's effect partly runs through this. Muon has no per-weight second moment.
- Ideas from the literature: Langevin noise for saddle escape (Muon meets Tamed Langevin, arXiv
  2610.02158); balanced routing (Sinkhorn/BASE layers; usage-balance losses) against collapse.

THE ARM (the screens' configuration: S=2, P=4, k=2, grouped, no conv, MAX_ITERS 24000; S5's
SLOW_MEM run path unchanged: one Adam, gate W_in/W_h/W_g at lr 1e-3 = test_channel_binding.SUB_LR
throughout, every other parameter 1e-4 for updates 1-2400, 1e-3 after)
  LANGEVIN  after each optimizer step t (an optimizer step post-hook), for t = 1..2399, add
            N(0, sigma_t^2) to every element of W_in, W_h and W_g, with
                sigma_t = 0.01 x (the tensor's std at initialisation) x (1 - t / 2400),
            decaying linearly to 0 at update 2400 (from update 2400 on nothing is added); the noise
            comes from a dedicated torch.Generator seeded NOISE_BASE + seed, drawn W_in, W_h, W_g in
            that order at each update (the global RNG and the batches are untouched); no hinge.
Seeds 160-199, paired with SLOW_MEM (S5 160-179, S7 180-199; 24/40 DISCOVERED) and S13's
SLOW_HINGE (34/40).

RULE (fixed before any run): vs SLOW_MEM over 40 seeds, DISCOVERED: promising if b - c >= 4 and
McNemar p < 0.1; not if b - c <= 0; otherwise inconclusive (explore_batch1.verdict). Also printed:
vs SLOW_HINGE, and ROUTED*@1200.

CHECKS: with sigma = 0 the run equals SLOW_MEM bit for bit (curves and weights, 3600 steps; also
S5's record); the noise per tensor at updates 1, 1200 and 2399 has std sigma_t within 2%, zero from
update 2400 (2400 and 2401 checked: the hook leaves the weights exactly as the optimizer left them).
How the 2% is measured: W_g has 64 elements, so one draw's sample std has a standard error of ~9%
and a single draw cannot test 2%. The check therefore asserts (i) the noise added in the run at each
of those updates is sigma_t times the unit draw (std(added) / (sigma_t x std(unit draw)) within 2% of
1, per tensor) and (ii) the unit draws have std 1 within 2% per tensor shape, pooled over 200 draws
from the noise generator; the single-draw ratios std(added) / sigma_t are printed too.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_slow_hinge as s13
import test_short_conv as tsc
from explore_batch1 import verdict
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY

NAME = "gate_langevin"
IDEA = "Langevin-style noise on the gate's weights, decaying to 0 at update 2400, on SLOW_MEM (no hinge)"
SOURCE = ("Muon meets Tamed Langevin (arXiv 2610.02158): noise for saddle escape; batches 6-7 (the hinge's "
          "effect partly runs through Adam's sign-like step)")
CHANGE = ("SLOW_MEM + after each optimizer step t < 2400, N(0, sigma_t^2) on every element of W_in, W_h, W_g, "
          "sigma_t = 0.01 x init std x (1 - t/2400)")
PAIRING = ("seeds 160-199, paired with SLOW_MEM (S5 160-179, S7 180-199) and S13's SLOW_HINGE (same initial "
           "parameters and batches)")
GATE = s5.GATE
SIGMA_REL = 0.01
WARM = s5.WARM_UPDATES                                     # 2400
NOISE_BASE = 9_100_000
SEEDS = tuple(range(160, 200))
CHECK_AT = (1, 1200, WARM - 1)
POOL_DRAWS = 200

ARMS = {
    "LANGEVIN": dict(s5.ARMS["SLOW_MEM"], key="LANGEVIN", seeds=SEEDS, prio=1,
                     label="SLOW_MEM + decaying Gaussian noise on W_in, W_h, W_g (updates 1-2399)",
                     sigma_rel=SIGMA_REL,
                     sched=s5.ARMS["SLOW_MEM"]["sched"] + f"; + N(0, sigma_t^2) on W_in, W_h, W_g after each "
                           f"step t < {WARM}, sigma_t = {SIGMA_REL:g} x init std x (1 - t/{WARM})"),
}


def sigma_at(std0, t, rel=SIGMA_REL):
    return rel * std0 * max(0.0, 1.0 - t / WARM)


class GateNoise:
    """The noise hook: after each optimizer step (global post-hook, registered for one run), add
    N(0, sigma_t^2) to W_in, W_h, W_g of the run's model. log_at: updates whose added noise is
    recorded (for the CHECK)."""

    def __init__(self, seed, rel=SIGMA_REL, log_at=()):
        self.rel, self.log_at, self.log = rel, set(log_at), {}
        self.gen = torch.Generator().manual_seed(NOISE_BASE + seed)
        self.model, self.std0, self.t = None, None, 0

    def bind(self, model):
        self.model = model
        named = dict(model.named_parameters())
        self.params = [named[n] for n in GATE]
        self.std0 = [float(p.detach().std()) for p in self.params]

    def __call__(self, opt, args, kwargs):
        self.t += 1
        t = self.t
        if self.model is None:
            return
        logging = t in self.log_at
        with torch.no_grad():
            rec = []
            for p, s0 in zip(self.params, self.std0):
                sig = sigma_at(s0, t, self.rel)
                before = p.detach().clone() if logging else None
                z_std = None
                if sig > 0:
                    z = torch.randn(p.shape, generator=self.gen, dtype=p.dtype)
                    p.add_(z * sig)
                    z_std = float(z.std())
                if logging:
                    d = p.detach() - before
                    rec.append(dict(sigma=sig, z_std=z_std, added_std=float(d.std()),
                                    added_max=float(d.abs().max())))
            if logging:
                self.log[t] = rec


def builder_with(noise):
    def builder(a, seed):
        def make():
            m = build(ec.TASK, a, ARCH, seed)
            noise.bind(m)
            return m
        return make
    return builder


def run_path(a, seed, iters, keep=None, rel=None, log_at=()):
    """S5's SLOW_MEM run path (its groups, lr switch, statistics and builder), plus the noise hook."""
    noise = GateNoise(seed, a["sigma_rel"] if rel is None else rel, log_at)
    holder = {}
    h = register_optimizer_step_post_hook(noise)
    try:
        rec = ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK,
                         stats_fn=c2.stats_b2, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                         builder=builder_with(noise), run_kw=dict(param_groups=s5.make_groups(holder)))
    finally:
        h.remove()
    if rec.get("ok"):
        rec["noise"] = dict(std0=noise.std0, steps=noise.t, seed=NOISE_BASE + seed, rel=noise.rel)
    return rec, noise


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])[0]


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, s5.builder(a, 0), a["lr"], c2.stats_b2)]


def check():
    ok = True
    rows = []
    it = 3 * EVAL_EVERY
    a = ARMS["LANGEVIN"]
    # (1) sigma = 0: SLOW_MEM bit for bit (a fresh SLOW_MEM run and S5's record)
    k1, k2 = {"at": it}, {"at": it}
    r1, _ = run_path(a, 160, it, keep=k1, rel=0.0)
    r2 = s5.run_path(s5.ARMS["SLOW_MEM"], 160, it, keep=k2)
    rec = [c for c in ec.load_store(s5.NAME)["runs"]["SLOW_MEM|160"]["curve"] if c[0] <= it]
    same = (r1["curve"] == r2["curve"] == rec
            and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"]))
    rows.append((f"with sigma = 0 LANGEVIN equals SLOW_MEM bit for bit (seed 160, {it} steps; curves = a fresh "
                 f"SLOW_MEM run = S5's record; weights equal)", same))
    # (2) the noise: sigma_t at updates 1, 1200, 2399; nothing from 2400
    k3 = {"at": WARM + 1}
    r3, nz = run_path(a, 160, WARM + 1, keep=k3, log_at=CHECK_AT + (WARM, WARM + 1))
    scale_ok, desc = True, []
    for t in CHECK_AT:
        entry = nz.log.get(t) or []
        scale_ok &= len(entry) == 3
        for nm, e, s0 in zip(GATE, entry, nz.std0):
            want = sigma_at(s0, t)
            ratio = e["added_std"] / (want * e["z_std"]) if e["z_std"] else float("nan")
            scale_ok &= e["sigma"] == want and abs(ratio - 1) < 0.02
            desc.append(f"{nm}@{t} sigma {want:.2e} std(added)/(sigma x std(unit)) {ratio:.4f} "
                        f"[one draw: std(added)/sigma {e['added_std'] / want:.3f}]")
    zero_ok = all(len(nz.log.get(t) or []) == 3 and all(e["z_std"] is None and e["added_max"] == 0.0
                                                         for e in nz.log[t]) for t in (WARM, WARM + 1))
    g = torch.Generator().manual_seed(NOISE_BASE + 160)
    pooled = {}
    for nm, s0 in zip(GATE, nz.std0):
        shp = dict(build(ec.TASK, a, ARCH, 160).named_parameters())[nm].shape
        zs = torch.cat([torch.randn(shp, generator=g).reshape(-1) for _ in range(POOL_DRAWS)])
        pooled[nm] = float(zs.std())
    pool_ok = all(abs(v - 1) < 0.02 for v in pooled.values())
    rows.append(("the noise added at updates 1, 1200, 2399 is sigma_t x the unit draw, per tensor: "
                 + "; ".join(desc), scale_ok))
    rows.append((f"the unit draws have std 1 within 2% per tensor shape, pooled over {POOL_DRAWS} draws: "
                 + ", ".join(f"{k} {v:.4f}" for k, v in pooled.items()), pool_ok))
    rows.append((f"nothing is added from update {WARM} (the hook's change to W_in, W_h, W_g at updates {WARM} and "
                 f"{WARM + 1} is exactly 0 and no draw is made; the run took {nz.t} optimizer steps)",
                 zero_ok and nz.t == WARM + 1 and r3.get("ok")))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def compare(label, new, old, seeds, fn, what, arm="LANGEVIN"):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    nn = sum(bool(fn(new[s])) for s in both)
    no = sum(bool(fn(old[s])) for s in both)
    print(f"    {what:<13} {arm} {nn:>2}/{len(both)}   {label} {no:>2}/{len(both)}   {arm} only {b}, "
          f"{label} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=nn, old=no, b=b, c=c, p=p)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    new = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith("LANGEVIN|") and r.get("ok")}
    sm = s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")
    sh = s13.runs_of((s13.NAME,), "SLOW_HINGE")
    print("  per seed (ROUTED* at 1200/2400/3600/end):")
    print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} {'SLOW_MEM':<15} {'SLOW_HINGE':<15}")
    for s in SEEDS:
        r = new.get(s)
        tags = f"{ec.tag(sm.get(s)):<15} {ec.tag(sh.get(s)):<15}"
        if r is None:
            print(f"    {s:>4} {ec.tag(store['runs'].get(f'LANGEVIN|{s}')):<70} {tags}")
            continue
        m = r["end"].get("margin")
        print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  {ec.tag(r):<15} {tags}")
    done = [s for s in SEEDS if s in new]
    xr = lambda step: (lambda r: c2.routed_at(r, step))
    print(f"  vs SLOW_MEM, {len(done)} seeds (the rule):")
    d = compare("SLOW_MEM", new, sm, done, ec.discovered, "DISCOVERED")
    compare("SLOW_MEM", new, sm, done, ec.bound, "BOUND")
    compare("SLOW_MEM", new, sm, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  vs SLOW_HINGE, {len(done)} seeds:")
    dh = compare("SLOW_HINGE", new, sh, done, ec.discovered, "DISCOVERED")
    compare("SLOW_HINGE", new, sh, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  failure classes ({len(done)} seeds): LANGEVIN {s13.fails(new, done)}   SLOW_MEM {s13.fails(sm, done)}   "
          f"SLOW_HINGE {s13.fails(sh, done)}")
    v = verdict(d)
    print(f"  RULE S26 (vs SLOW_MEM, DISCOVERED {d['b']} vs {d['c']}, p = {d['p']:.3g}; promising if b - c >= 4 and "
          f"p < 0.1, not if b - c <= 0): {v}")
    return dict(d, verdict=v, slow_hinge=dh)
