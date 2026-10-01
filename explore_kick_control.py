#!/usr/bin/env python
"""
explore_kick_control.py — EXPLORATORY, not a result. Screen S16 (batch 6): does S13's hinge help
through its DIRECTION, or would any kick of that size do?

BACKGROUND (post hoc, from batch 5)
- S13 SLOW_HINGE (SLOW_MEM + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] on the read
  gate at key positions) discovered 34/40 vs SLOW_MEM 24/40 (10 vs 0). The hinge never fired on 20
  seeds (all DISCOVERED in both arms, the runs identical). On the 20 seeds where it fired, SLOW_MEM
  4/20 -> SLOW_HINGE 14/20, mostly from a few firings in the first 1200 updates (1-3 batches on
  the runs that bound).
- Near uniform the gate's task gradient is tiny, so under Adam one batch with a real gradient is a
  large step. Open: does the hinge's direction matter, or would any kick of that size do?

THE ARM (KICK). explore_slow_mem's SLOW_MEM (param groups and lr switch unchanged: gate W_in/W_h/
W_g at lr 1e-3 = test_channel_binding.SUB_LR throughout; every other parameter 1e-4 for updates
1-2400, 1e-3 after; MAX_ITERS 24000). On every training batch, S13's hinge is computed exactly as
S13 computes it (explore_slow_pos.pos_penalty on the read gate at the key positions 3j+1; TAU 0.2,
LAMBDA 1.0). Whenever it would be nonzero (either term above TAU), and while the seed's kick
budget lasts:
  - its gradient on each tensor it reaches (W_in, W_h, W_g, embed.weight) is taken
    (torch.autograd.grad on the current batch and weights; nothing is accumulated);
  - for each tensor, a random tensor of the same shape is drawn from a dedicated generator seeded
    KICK_BASE + 100_000 * seed + update (Gaussian, the four tensors in the order W_in, W_h, W_g,
    embed.weight), then scaled to the Frobenius norm of the hinge's gradient on that tensor;
  - the random tensor is added to that tensor's .grad: through explore_aux_gate's gradient-
    injection node on the logits, aux = sum_p (p * kick_p).sum() with kick_p constant, whose
    gradient on p is exactly kick_p; it enters loss.backward() with the task's gradient, so
    onset_run is unchanged. The hinge's own gradient is not applied.
Budget: at most N_s kicks per seed, N_s = the number of batches on which S13's hinge fired on that
seed (explore_out/slow_hinge_results.json, end hinge_counts[0]); after that, nothing. Logged per
run: every kick (update, eta2 index/half, the hinge gradient's norms and the kick's norms per
tensor, the cosine between the two), kicks used, and the batches on which the hinge would have
fired (with or without budget).
Seeds: the 20 seeds on which S13's hinge fired. Paired with SLOW_HINGE (14/20 there) and SLOW_MEM
(4/20; batch 2's S5 on 160-179, batch 3's S7 on 180-199): same initial parameters and batches.

RULE (fixed before any run), outcome DISCOVERED on the 20 seeds: "the direction matters" if KICK
<= 7/20; "a kick of that size suffices" if KICK >= 12/20; otherwise neither. Also printed: failure
classes, ROUTED* at 1200, kicks used per seed, and McNemar against SLOW_HINGE and SLOW_MEM.

CHECKS: with N_s = 0, KICK equals SLOW_MEM bit for bit (curves and weights, 3600 steps; seed 162,
where the hinge fired); with the random tensor replaced by the hinge's own gradient, the path
reproduces S13's SLOW_HINGE on seed 162 (bit for bit if it does; otherwise the largest curve
difference through 3600 steps is reported, and the batch continues only if the outcome matches);
each logged kick's norm equals the hinge gradient's it replaced (a 1200-step random KICK run on
seed 162); the kick enters .grad exactly (gradient with a kick minus without = the kick); the
budget stops the kicks.
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
import explore_slow_hinge as s13
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH, probe_batch
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY, BATCH

NAME = "kick_control"
IDEA = "S13's hinge replaced by a random kick of the same size, on the batches where it would fire"
SOURCE = ("batch 5's S13 (a few hinge firings flipped POSITION failures to DISCOVERED); random-"
          "direction controls for a gradient intervention")
CHANGE = ("SLOW_MEM; on each batch where S13's hinge would be nonzero (at most N_s per seed), add to "
          "W_in, W_h, W_g, embed.weight a Gaussian tensor scaled to the hinge gradient's norm on "
          "that tensor; the hinge's own gradient is not applied")
PAIRING = ("the 20 seeds where S13's hinge fired, paired with SLOW_HINGE (S13) and SLOW_MEM (S5 / S7) "
           "on the same seeds (same initial parameters and batches)")
TAU, LAMBDA = s13.TAU, s13.LAMBDA
KICK_BASE = 7_000_000
KICK_PARAMS = ("W_in", "W_h", "W_g", "embed.weight")
S13_STORE = "slow_hinge"
DIRECTION_MAX, SUFFICES_MIN = 7, 12
CHECK_SEED = 162


def s13_counts():
    runs = ec.load_store(S13_STORE)["runs"]
    out = {}
    for k, r in runs.items():
        if k.startswith("SLOW_HINGE|") and r.get("ok"):
            out[int(k.split("|")[1])] = int(r["end"]["hinge_counts"][0])
    return out


N_S = {s: n for s, n in s13_counts().items() if n > 0}
SEEDS = tuple(sorted(N_S))

ARMS = {
    "KICK": dict(s5.ARMS["SLOW_MEM"], key="KICK", seeds=SEEDS, prio=1,
                 label="SLOW_MEM + random kicks sized as S13's hinge gradient (at most N_s per seed)",
                 sched=s5.ARMS["SLOW_MEM"]["sched"] + "; + on a batch where S13's hinge (TAU 0.2) "
                       "would fire, a Gaussian tensor of the hinge gradient's norm added to W_in, "
                       "W_h, W_g, embed.weight's .grad (at most N_s kicks per seed)"),
}


def kick_seed(seed, update):
    return KICK_BASE + 100_000 * seed + update


class KickBDH(MultiBDH):
    """mode 'random': the kick is a scaled Gaussian; 'hinge': the hinge's own gradient (CHECK)."""

    def __init__(self, *args, tau=TAU, lam=LAMBDA, n_max=0, seed=0, mode="random",
                 pen_SP=(2, 4), **kw):
        super().__init__(*args, **kw)
        self.tau, self.lam, self.n_max, self.seed, self.mode, self.pen_SP = \
            tau, lam, n_max, seed, mode, pen_SP
        self.n_batches, self.n_active, self.n_kicks = 0, 0, 0
        self.kick_log, self.last_kick = [], None

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and torch.is_grad_enabled():
            self.n_batches += 1
            e_i, e_h = s12.pos_penalty(gr, *self.pen_SP)
            h_i, h_h = F.relu(e_i - self.tau), F.relu(e_h - self.tau)
            fires = bool(h_i > 0) or bool(h_h > 0)
            self.n_active += int(fires)
            if fires and self.n_kicks < self.n_max:
                named = dict(self.named_parameters())
                params = [named[n] for n in KICK_PARAMS]
                hg = torch.autograd.grad(self.lam * (h_i + h_h), params, retain_graph=True,
                                         allow_unused=True)
                hg = [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(hg, params)]
                if self.mode == "random":
                    gen = torch.Generator().manual_seed(kick_seed(self.seed, self.n_batches))
                    kicks = []
                    for g, p in zip(hg, params):
                        z = torch.randn(p.shape, generator=gen, dtype=p.dtype)
                        kicks.append(z * (g.norm() / z.norm()))
                else:
                    kicks = hg
                self.n_kicks += 1
                self.last_kick = dict(zip(KICK_PARAMS, kicks))
                cos = [float((g * k).sum() / (g.norm() * k.norm())) if float(g.norm()) > 0 else 0.0
                       for g, k in zip(hg, kicks)]
                self.kick_log.append(dict(update=self.n_batches, eta2=[float(e_i.detach()),
                                                                       float(e_h.detach())],
                                          hinge_norm=[float(g.norm()) for g in hg],
                                          kick_norm=[float(k.norm()) for k in kicks], cos=cos))
                aux = sum((p * k).sum() for p, k in zip(params, kicks))
                logits = _Inject.apply(logits, aux)
        return logits, sat, gr, gw


def make_model(a, seed, n_max=None, mode="random", tau=TAU):
    task = ec.TASK
    torch.manual_seed(seed)
    return KickBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                   gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                   ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), tau=tau, lam=LAMBDA,
                   n_max=N_S.get(seed, 0) if n_max is None else n_max, seed=seed, mode=mode,
                   pen_SP=(task.S, task.P))


def stats_fn(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out.update(kick_counts=[model.n_kicks, model.n_active, model.n_batches], kick_budget=model.n_max)
    if step == "end":
        out["kick_log"] = model.kick_log
    return out


def run_path(a, seed, iters, n_max=None, mode="random", keep=None):
    holder = {}
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK,
                      stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s, n_max, mode)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: make_model(a, CHECK_SEED), a["lr"], stats_fn)]


def norm_diff(log):
    """Largest relative difference between a kick's norm and the hinge gradient's, per tensor."""
    d = 0.0
    for e in log:
        for h, k in zip(e["hinge_norm"], e["kick_norm"]):
            d = max(d, abs(k - h) / h if h > 0 else abs(k))
    return d


def check():
    ok = True
    seed, it = CHECK_SEED, 3 * EVAL_EVERY
    a = ARMS["KICK"]
    # (1) N_s = 0 is SLOW_MEM bit for bit
    k0, k5 = {"at": it}, {"at": it}
    r0 = run_path(a, seed, it, n_max=0, keep=k0)
    r5 = s5.run_path(s5.ARMS["SLOW_MEM"], seed, it, keep=k5)
    rec5 = [c for c in ec.load_store("slow_mem")["runs"][f"SLOW_MEM|{seed}"]["curve"] if c[0] <= it]
    same0 = (r0["curve"] == r5["curve"] == rec5
             and all(torch.equal(k0["snap"][n], k5["snap"][n]) for n in k5["snap"]))
    would0 = r0["end"]["kick_counts"]
    # (2) the hinge's own gradient through the kick path vs S13's SLOW_HINGE (run to its natural end)
    rh = run_path(a, seed, a["iters"], mode="hinge")
    r13 = ec.load_store(S13_STORE)["runs"][f"SLOW_HINGE|{seed}"]
    c_new = {c[0]: c for c in rh["curve"]}
    c_old = {c[0]: c for c in r13["curve"]}
    steps = [s for s in sorted(c_old) if s <= it]
    bit = [c for c in rh["curve"] if c[0] <= it] == [c for c in r13["curve"] if c[0] <= it]
    full = rh["curve"] == r13["curve"]
    dacc = max(abs(c_new[s][1] - c_old[s][1]) if s in c_new else float("inf") for s in steps)
    dloss = max(abs(c_new[s][2] - c_old[s][2]) if s in c_new else float("inf") for s in steps)
    same_outcome = (ec.tag(rh) == ec.tag(r13) and rh["transition"] == r13["transition"])
    kh = rh["end"]["kick_counts"]
    # (3) a 1200-step random KICK run: each kick's norm equals the hinge gradient's it replaced
    rr = run_path(a, seed, EVAL_EVERY)
    log = rr["end"]["kick_log"]
    nd = norm_diff(log)
    maxcos = max((abs(c) for e in log for c in e["cos"]), default=float("nan"))
    # (4) the kick enters .grad exactly; (5) the budget stops the kicks
    x = probe_batch(ec.TASK, seed)[0][:BATCH + 1]
    gen = torch.Generator().manual_seed(5)
    tok, _ = ec.TASK.make_batch(BATCH, gen)
    inp, tgt = tok[:, :-1], tok[:, 1:]
    grads = []
    for n_max in (1, 0):
        m = make_model(a, seed, n_max=n_max, tau=0.0)
        m.train()
        ql, qt, _ = ec.TASK.select(m(inp)[0], tgt, None)
        F.cross_entropy(ql, qt).backward()
        grads.append((m, {n: p.grad.clone() for n, p in m.named_parameters()}))
    (mk, gk), (_, g0) = grads
    enter = max(float((gk[n] - g0[n] - mk.last_kick[n]).abs().max()) for n in KICK_PARAMS)
    scale = max(max(float(mk.last_kick[n].abs().max()), float(g0[n].abs().max())) for n in KICK_PARAMS)
    others = all(torch.equal(gk[n], g0[n]) for n in gk if n not in KICK_PARAMS)
    mb = make_model(a, seed, n_max=2, tau=0.0)
    mb.train()
    for _ in range(5):
        mb(x)[0].sum().backward()
    budget = mb.n_kicks == 2 and mb.n_active == 5 and mb.n_batches == 5
    for nm, v in ((f"the seeds are the {len(SEEDS)} on which S13's hinge fired (N_s from "
                   f"{S13_STORE}_results.json): {list(SEEDS)}; budgets {[N_S[s] for s in SEEDS]}",
                   len(SEEDS) == 20 and len(ARMS["KICK"]["seeds"]) >= 1
                   and set(ARMS["KICK"]["seeds"]) <= set(SEEDS)),
                  (f"N_s = 0 equals SLOW_MEM bit for bit through {it} steps on seed {seed} (curves = a "
                   f"fresh SLOW_MEM run = batch 2's record; weights equal; the hinge would have fired "
                   f"on {would0[1]} batches, kicks used {would0[0]})", same0),
                  (f"the kick path with the hinge's own gradient on seed {seed} vs S13's SLOW_HINGE: "
                   + (f"BIT-IDENTICAL curve through {it} steps (whole curve "
                      f"{'identical' if full else 'differs'})" if bit else
                      f"not bit-identical; largest curve difference through {it} steps |acc| {dacc:.3g}, "
                      f"|loss| {dloss:.3g}") +
                   f"; outcome {ec.tag(rh)} trans {rh['transition']} vs S13 {ec.tag(r13)} trans "
                   f"{r13['transition']} (kicks used {kh[0]} of budget {N_S.get(seed)}, would fire "
                   f"{kh[1]}); the batch continues only if the outcome matches", same_outcome),
                  (f"each logged kick's norm equals the hinge gradient's it replaced (random KICK, seed "
                   f"{seed}, {EVAL_EVERY} steps: {len(log)} kicks, max relative norm difference "
                   f"{nd:.1e} < 1e-5; max |cos(kick, hinge gradient)| {maxcos:.2f})",
                   len(log) >= 1 and nd < 1e-5),
                  (f"the kick enters .grad exactly: grad with a kick - grad without = the kick on "
                   f"{', '.join(KICK_PARAMS)} (max abs diff {enter:.1e}; largest |kick| or |task gradient| "
                   f"{scale:.1e}); every other parameter's gradient bitwise unchanged",
                   enter <= 1e-6 * scale and others),
                  (f"the budget stops the kicks (budget 2, hinge firing on 5 batches: kicks "
                   f"{mb.n_kicks}, would fire {mb.n_active})", budget)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def rule(n):
    if n <= DIRECTION_MAX:
        return "the direction matters"
    if n >= SUFFICES_MIN:
        return "a kick of that size suffices"
    return "neither"


def compare(label, new, old, seeds, fn, what):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    nn = sum(bool(fn(new[s])) for s in both)
    no = sum(bool(fn(old[s])) for s in both)
    print(f"    {what:<13} KICK {nn:>2}/{len(both)}   {label} {no:>2}/{len(both)}   KICK only {b}, "
          f"{label} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=nn, old=no, b=b, c=c, p=p)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    new = {int(k.split("|")[1]): r for k, r in store["runs"].items()
           if k.startswith("KICK|") and r.get("ok")}
    sh = s13.runs_of((S13_STORE,), "SLOW_HINGE")
    sm = s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")
    seeds = [s for s in ARMS["KICK"]["seeds"]]
    print(f"  per seed (ROUTED* at 1200/2400/3600/end; N_s = S13's hinge firings on the seed = the kick "
          f"budget; kicks = kicks used; fires = batches where the hinge would have fired in this run; "
          f"first = update of the first kick):")
    print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} {'SLOW_HINGE':<15} {'SLOW_MEM':<15} {'N_s':>4} {'kicks':>5} "
          f"{'fires':>5} {'first':>6} {'norm diff':>9}")
    for s in seeds:
        r = new.get(s)
        tags = f"{ec.tag(sh.get(s)):<15} {ec.tag(sm.get(s)):<15}"
        if r is None:
            print(f"    {s:>4} {ec.tag(store['runs'].get(f'KICK|{s}')):<70} {tags}")
            continue
        m = r["end"].get("margin")
        kc = r["end"]["kick_counts"]
        log = r["end"].get("kick_log", [])
        print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
              f"{ec.tag(r):<15} {tags} {N_S[s]:>4} {kc[0]:>5} {kc[1]:>5} "
              f"{(log[0]['update'] if log else '--'):>6} {norm_diff(log):9.1e}")
    done = [s for s in seeds if s in new]
    xr = lambda step: (lambda r: c2.routed_at(r, step))
    print(f"  vs SLOW_HINGE (S13; the hinge's own direction), {len(done)} seeds:")
    d_sh = compare("SLOW_HINGE", new, sh, done, ec.discovered, "DISCOVERED")
    compare("SLOW_HINGE", new, sh, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  vs SLOW_MEM (no kick), {len(done)} seeds:")
    d_sm = compare("SLOW_MEM", new, sm, done, ec.discovered, "DISCOVERED")
    compare("SLOW_MEM", new, sm, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  failure classes ({len(done)} seeds): KICK {s13.fails(new, done)}   SLOW_HINGE "
          f"{s13.fails(sh, done)}   SLOW_MEM {s13.fails(sm, done)}")
    used = {s: new[s]["end"]["kick_counts"][0] for s in done}
    print(f"  kicks used per seed (budget N_s): " + ", ".join(f"{s}: {used[s]}/{N_S[s]}" for s in done))
    print(f"  kicks in updates 1-{EVAL_EVERY}: " + ", ".join(
        f"{s}: {sum(1 for e in new[s]['end'].get('kick_log', []) if e['update'] <= EVAL_EVERY)}"
        for s in done))
    nd = sum(ec.discovered(new[s]) for s in done)
    rd = rule(nd)
    print(f"  READING S16 (KICK DISCOVERED {nd}/{len(done)}; 'the direction matters' if <= "
          f"{DIRECTION_MAX}/20, 'a kick of that size suffices' if >= {SUFFICES_MIN}/20): {rd}")
    return dict(n=len(done), discovered=nd, rule=rd, slow_hinge=d_sh, slow_mem=d_sm)
