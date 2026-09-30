#!/usr/bin/env python
"""
explore_slow_hinge.py — EXPLORATORY, not a result. Screen S13 (batch 5): SLOW_MEM plus a HINGE on
the read gate's position information — zero until a position split starts to form.

BACKGROUND (post hoc, from batch 4)
- S12 SLOW_POS (SLOW_MEM + 1.0 * (eta^2 by triple index + eta^2 by half)) discovered 15/20 vs
  SLOW_MEM 12/20 on seeds 160-179 (6 vs 3, p = 0.51). It removed every POSITION failure, but
  routed at 1200 fell to 0/20 (SLOW_MEM 12/20, p = 0.0005); its 5 failures are gates that never
  left uniform (every eta^2 ~0, margin 0); it lost 3 of SLOW_MEM's discoveries (160, 161, 164).
- The penalty at 1200 had median 0.025 (range 0.01-0.09). With no position effect, eta^2 by triple
  index on one batch (32 sequences x 8 key positions, 8 groups) has mean 0.028, median 0.025, 95th
  percentile 0.055 (simulation): the penalty sat at its sampling floor.
- eta^2 is a ratio of two quadratics in the gate's deviations: it does not shrink as the gate
  nears uniform, and its gradient grows as the deviations shrink, while the task's routing
  gradient shrinks with them (the product gate's stationary point). Reading: near uniform the
  penalty's noise swamps the routing pull, delaying or preventing commitment.
- S11: after a gate reset the fresh gate landed in a key split 24 of 27 times and in a position
  split never. Reading: position splits are the early attractor of a naive memory, key splits
  the attractor once the memory has learned the keys; SLOW_MEM delays the second, a position
  penalty blocks the first.

THE ARM (SLOW_HINGE). explore_slow_mem's SLOW_MEM (param groups and lr switch unchanged: gate at
lr 1e-3 = test_channel_binding.SUB_LR throughout; everything else 1e-4 for updates 1-2400, 1e-3
after; MAX_ITERS 24000) plus, on each training batch,
    LAMBDA * [relu(eta2_index - TAU) + relu(eta2_half - TAU)],   LAMBDA = 1.0, TAU = 0.2,
with eta2_index and eta2_half computed exactly as S12's (explore_slow_pos.pos_penalty: the read
gate's channel-0 probability at the body's key positions, differentiable) and added through S12's
gradient-injection node. TAU is ~7x the index floor and far below the position splits seen at 1200
(0.95-1.00): the penalty is zero until a position split starts to form.
Logged per run: the fraction of training batches with either hinge term > 0 over updates 1-1200,
1-2400 and the whole run.
Seeds 160-199 (40), paired with SLOW_MEM (batch 2's S5 on 160-179, batch 3's S7 on 180-199), with
X's recorded arm A, and on 160-179 with batch 4's SLOW_POS (same initial parameters and batches).

RULE (fixed before any run): vs SLOW_MEM over the 40 seeds, outcome DISCOVERED: promising if
b - c >= 4 and McNemar p < 0.10; not if b - c <= 0; otherwise inconclusive. Also printed: vs X's
arm A (40 seeds) and vs SLOW_POS (160-179), with ROUTED* at 1200 / 2400 / end and McNemar for each.
READING (seeds 160-179): "noise floor" if ROUTED* at 1200 >= 8/20 and at most 2 failures are
uncommitted gates (OTHER with every eta^2 < 0.1 at the end); "not the noise floor" if ROUTED* at
1200 <= 3/20; otherwise neither.

CHECKS: with TAU = 1.0 the run equals SLOW_MEM bit for bit, and with TAU = 0 it equals S12's
SLOW_POS bit for bit (curves and weights, 3600 steps each; also against the recorded runs); the
hinge's gradient reaches only the gate and the embedding, and is exactly zero on a batch where
both terms are below TAU; the logged fraction is 0 with TAU = 1 and 1 with TAU = 0.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_slow_pos as s12
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH, probe_batch
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY

NAME = "slow_hinge"
IDEA = "SLOW_MEM + a hinge on the read gate's position eta^2 (zero below TAU = 0.2)"
SOURCE = ("batch 4's SLOW_POS (a full eta^2 penalty sat at its sampling floor and delayed "
          "commitment); hinge / margin penalties that act only past a threshold")
CHANGE = ("SLOW_MEM + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] of the read gate at "
          "key positions, on each training batch")
PAIRING = ("seeds 160-199, paired with SLOW_MEM (S5 160-179, S7 180-199), X's recorded arm A, and "
           "batch 4's SLOW_POS on 160-179 (same initial parameters and batches)")
SEEDS = tuple(range(160, 200))
SEEDS_POS = tuple(range(160, 180))
LAMBDA, TAU = 1.0, 0.2
UNCOMMITTED_ETA = 0.1
READ_ROUTED_HI, READ_ROUTED_LO, READ_UNCOMMITTED_MAX = 8, 3, 2
STORES_SLOW_MEM = ("slow_mem", "slow_mem_ext")
STORE_SLOW_POS = "slow_pos"
REF_STORE = "ref_a2400"

ARMS = {
    "SLOW_HINGE": dict(s5.ARMS["SLOW_MEM"], key="SLOW_HINGE", seeds=SEEDS, prio=1,
                       label="SLOW_MEM + hinge on position eta^2 (TAU 0.2)",
                       sched=s5.ARMS["SLOW_MEM"]["sched"] + f"; + {LAMBDA} * [relu(eta2_index - "
                             f"{TAU}) + relu(eta2_half - {TAU})] on each training batch"),
}


class HingeBDH(MultiBDH):
    def __init__(self, *args, pen_lambda=None, tau=TAU, pen_SP=(2, 4), **kw):
        super().__init__(*args, **kw)
        self.pen_lambda, self.tau, self.pen_SP = pen_lambda, tau, pen_SP
        self.pen_last, self.n_batches, self.n_active = None, 0, 0

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.pen_lambda is not None and torch.is_grad_enabled():
            e_i, e_h = s12.pos_penalty(gr, *self.pen_SP)
            h_i, h_h = F.relu(e_i - self.tau), F.relu(e_h - self.tau)
            self.n_batches += 1
            self.n_active += int(bool(h_i > 0) or bool(h_h > 0))
            self.pen_last = (float(e_i.detach()), float(e_h.detach()))
            logits = _Inject.apply(logits, self.pen_lambda * (h_i + h_h))
        return logits, sat, gr, gw


def make_model(a, seed, tau=TAU, lam=LAMBDA):
    task = ec.TASK
    torch.manual_seed(seed)
    return HingeBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                    gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                    ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), pen_lambda=lam, tau=tau,
                    pen_SP=(task.S, task.P))


def stats_fn(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out.update(pos_pen_last_train=model.pen_last, hinge_counts=[model.n_active, model.n_batches])
    return out


def run_path(a, seed, iters, tau=TAU, keep=None):
    holder = {}
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK,
                      stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s, tau)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: make_model(a, 0), a["lr"], stats_fn)]


def frac(r, step):
    d = c2.stat_at(r, step)
    if not d or not d.get("hinge_counts") or not d["hinge_counts"][1]:
        return None
    a, n = d["hinge_counts"]
    return a / n


def check():
    ok = True
    seed, it = 160, 3 * EVAL_EVERY
    a = ARMS["SLOW_HINGE"]
    # (1) TAU = 1.0 is SLOW_MEM bit for bit
    k1, k5 = {"at": it}, {"at": it}
    r1 = run_path(a, seed, it, tau=1.0, keep=k1)
    r5 = s5.run_path(s5.ARMS["SLOW_MEM"], seed, it, keep=k5)
    rec5 = [c for c in ec.load_store("slow_mem")["runs"][f"SLOW_MEM|{seed}"]["curve"] if c[0] <= it]
    same1 = (r1["curve"] == r5["curve"] == rec5
             and all(torch.equal(k1["snap"][n], k5["snap"][n]) for n in k5["snap"]))
    f1 = frac(r1, "end")
    # (2) TAU = 0 is S12's SLOW_POS bit for bit
    k0, kp = {"at": it}, {"at": it}
    r0 = run_path(a, seed, it, tau=0.0, keep=k0)
    rp = s12.run_path(s12.ARMS["SLOW_POS"], seed, it, keep=kp)
    recp = [c for c in ec.load_store(STORE_SLOW_POS)["runs"][f"SLOW_POS|{seed}"]["curve"] if c[0] <= it]
    same0 = (r0["curve"] == rp["curve"] == recp
             and all(torch.equal(k0["snap"][n], kp["snap"][n]) for n in kp["snap"]))
    f0 = frac(r0, "end")
    # (3) the hinge's gradient: only the gate and the embedding; exactly zero below TAU
    x = probe_batch(ec.TASK, seed)[0][:32]
    m = make_model(a, seed, tau=0.0)
    m.train()
    gr = MultiBDH.forward(m, x)[2]
    e_i, e_h = s12.pos_penalty(gr, ec.TASK.S, ec.TASK.P)
    (F.relu(e_i - 0.0) + F.relu(e_h - 0.0)).backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    reach = got == {"W_in", "W_h", "W_g", "embed.weight"}
    m2 = make_model(a, seed)                                      # TAU = 0.2
    m2.train()
    gr2 = MultiBDH.forward(m2, x)[2]
    f_i, f_h = s12.pos_penalty(gr2, ec.TASK.S, ec.TASK.P)
    below = float(f_i) < TAU and float(f_h) < TAU
    (F.relu(f_i - TAU) + F.relu(f_h - TAU)).backward()
    zero = all(p.grad is None or bool((p.grad == 0).all()) for p in m2.parameters())
    for nm, v in ((f"TAU = 1.0 equals SLOW_MEM bit for bit through {it} steps (curves = a fresh SLOW_MEM "
                   f"run = batch 2's record; weights equal)", same1),
                  (f"TAU = 0 equals S12's SLOW_POS bit for bit through {it} steps (curves = a fresh "
                   f"SLOW_POS run = batch 4's record; weights equal)", same0),
                  (f"the logged hinge fraction is 0 with TAU = 1 ({f1}) and 1 with TAU = 0 ({f0})",
                   f1 == 0.0 and f0 == 1.0),
                  (f"the hinge's gradient (TAU = 0) reaches only {sorted(got)}", reach),
                  (f"with both terms below TAU = {TAU} (eta2 index {float(f_i):.3f}, half "
                   f"{float(f_h):.3f}) the hinge's gradient is exactly zero", below and zero)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def runs_of(names, key):
    out = {}
    for nm in names:
        for k, r in ec.load_store(nm)["runs"].items():
            if k.split("|")[0] == key and r.get("ok"):
                out[int(k.split("|")[1])] = r
    return out


def uncommitted(r):
    return (not ec.bound(r) and fail_class(r) == "OTHER"
            and all(v < UNCOMMITTED_ETA for k, v in r["end"].items() if k.startswith("eta_")))


def compare(label, new, old, seeds, fn, what):
    both = [s for s in seeds if s in new and s in old and fn(old[s]) is not None]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    print(f"    {what:<13} SLOW_HINGE {sum(bool(fn(new[s])) for s in both):>2}/{len(both)}   {label} "
          f"{sum(bool(fn(old[s])) for s in both):>2}/{len(both)}   SLOW_HINGE only {b}, {label} only {c}, "
          f"McNemar two-sided p = {p:.3g}" + (f"   [{len(seeds) - len(both)} seeds without a value]"
                                              if len(both) < len(seeds) else ""))
    return dict(n=len(both), new=sum(bool(fn(new[s])) for s in both),
                old=sum(bool(fn(old[s])) for s in both), b=b, c=c, p=p)


def fails(runs, seeds):
    out = {}
    for s in seeds:
        r = runs.get(s)
        if r is not None and not ec.bound(r):
            out[fail_class(r)] = out.get(fail_class(r), 0) + 1
    return out


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    refs = c2.ref_runs(ec.load_store(REF_STORE))
    c2.per_seed_table2(me, store, refs=refs)
    new = {int(k.split("|")[1]): r for k, r in store["runs"].items()
           if k.startswith("SLOW_HINGE|") and r.get("ok")}
    sm = runs_of(STORES_SLOW_MEM, "SLOW_MEM")
    sp = runs_of((STORE_SLOW_POS,), "SLOW_POS")
    xa = ec.recorded("A")
    print("  per seed: hinge-active fraction of training batches (updates 1-1200 / 1-2400 / whole run); "
          "SLOW_MEM and SLOW_POS outcomes on the same seed:")
    fm = lambda v: "  -- " if v is None else f"{v:5.3f}"
    for s in SEEDS:
        r = new.get(s)
        if r is None:
            print(f"    {s:>4} not run")
            continue
        print(f"    {s:>4} {fm(frac(r, EVAL_EVERY))} / {fm(frac(r, 2 * EVAL_EVERY))} / {fm(frac(r, 'end'))}   "
              f"SLOW_HINGE {ec.tag(r):<15} SLOW_MEM {ec.tag(sm.get(s)):<15} "
              f"SLOW_POS {ec.tag(sp.get(s)) if s in SEEDS_POS else '--':<15}"
              + ("  uncommitted" if uncommitted(r) else ""))
    xr = lambda step: (lambda r: c2.routed_at(r, step))
    out = {}
    print("  vs SLOW_MEM (40 seeds; S5 160-179, S7 180-199):")
    out["slow_mem"] = compare("SLOW_MEM", new, sm, SEEDS, ec.discovered, "DISCOVERED")
    for st in (EVAL_EVERY, 2 * EVAL_EVERY, "end"):
        compare("SLOW_MEM", new, sm, SEEDS, xr(st), f"ROUTED*@{st}")
    print("  vs X's arm A (40 seeds; X's ROUTED* at 2400 from batch 2's re-run, seeds 160-179 only):")
    xa_runs = {s: xa[s] for s in SEEDS}
    out["x_a"] = compare("X A", new, xa_runs, SEEDS, ec.discovered, "DISCOVERED")
    compare("X A", new, xa_runs, SEEDS, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    x24 = {s: r for s, r in xa_runs.items() if c2.x_routed(s, 2 * EVAL_EVERY, refs) is not None}
    compare("X A", new, {s: dict(r, _r24=c2.x_routed(s, 2 * EVAL_EVERY, refs)) for s, r in x24.items()},
            SEEDS, lambda r: r["_r24"] if "_r24" in r else c2.routed_at(r, 2 * EVAL_EVERY),
            f"ROUTED*@{2 * EVAL_EVERY}")
    compare("X A", new, xa_runs, SEEDS, xr("end"), "ROUTED*@end")
    print("  vs SLOW_POS (160-179):")
    out["slow_pos"] = compare("SLOW_POS", new, sp, SEEDS_POS, ec.discovered, "DISCOVERED")
    for st in (EVAL_EVERY, 2 * EVAL_EVERY, "end"):
        compare("SLOW_POS", new, sp, SEEDS_POS, xr(st), f"ROUTED*@{st}")
    unc = [s for s in SEEDS if s in new and uncommitted(new[s])]
    done = [s for s in SEEDS if s in new]
    done_pos = [s for s in SEEDS_POS if s in new]
    print(f"  failure classes ({len(done)} seeds run of 160-199): SLOW_HINGE {fails(new, done)}   "
          f"SLOW_MEM {fails(sm, done)}   X arm A {fails(xa_runs, done)}")
    print(f"  failure classes ({len(done_pos)} seeds run of 160-179): SLOW_HINGE {fails(new, done_pos)}   "
          f"SLOW_POS {fails(sp, done_pos)}")
    print(f"  uncommitted gates (OTHER, every eta^2 < {UNCOMMITTED_ETA} at the end): SLOW_HINGE {unc}; "
          f"SLOW_POS {[s for s in SEEDS_POS if s in sp and uncommitted(sp[s])]}; "
          f"SLOW_MEM {[s for s in SEEDS if s in sm and uncommitted(sm[s])]}")
    r12 = sum(c2.routed_at(new[s], EVAL_EVERY) for s in SEEDS_POS if s in new)
    n_pos = sum(1 for s in SEEDS_POS if s in new)
    unc_pos = sum(1 for s in SEEDS_POS if s in new and uncommitted(new[s]))
    if r12 >= READ_ROUTED_HI and unc_pos <= READ_UNCOMMITTED_MAX:
        reading = "noise floor"
    elif r12 <= READ_ROUTED_LO:
        reading = "not the noise floor"
    else:
        reading = "neither"
    print(f"  READING S13 (seeds 160-179): ROUTED*@1200 {r12}/{n_pos}, uncommitted failures {unc_pos}: "
          f"{reading}")
    d = dict(out["slow_mem"], reading=reading, x_a=out["x_a"], slow_pos=out["slow_pos"])
    return d
