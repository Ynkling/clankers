#!/usr/bin/env python
"""
explore_kick_untimed.py — EXPLORATORY, not a result. Screen S20 (batch 7): one random kick at a
fixed update, with no trigger.

BACKGROUND (post hoc, from batches 5-6)
- S16: random kicks sized as S13's hinge gradient, on the batches where the hinge would fire,
  discovered 12/20 vs SLOW_HINGE 14/20 and SLOW_MEM 4/20 on the 20 seeds where the hinge fired.
  Direction mattered on 3 seeds (173, 190, 196); kicks routed later (at 1200: 4/20 vs 8/20).
- Under Adam, one gradient far above a parameter's running gradient moves it by roughly the same
  amount whatever its size, and inflates its second-moment estimate for thousands of updates.
- On the 20 seeds where S13's hinge never fired, SLOW_MEM and SLOW_HINGE are the same runs, all 20
  DISCOVERED. S16's first kick came at update 49-164 on 18 of its 20 seeds (3572 and 2267 on two).

THE ARM (KICK120). explore_slow_mem's SLOW_MEM (param groups and lr switch unchanged: gate
W_in/W_h/W_g at lr 1e-3 = test_channel_binding.SUB_LR throughout; every other parameter 1e-4 for
updates 1-2400, 1e-3 after; MAX_ITERS 24000) plus exactly one kick, at update KICK_AT = 120: a
Gaussian tensor added to the .grad of each of W_in, W_h, W_g and embed.weight (drawn in that order
from a dedicated generator seeded KICK_BASE + 100_000 * seed + 120), each scaled to the median,
over S16's 20 first kicks (explore_out/kick_control_results.json, each seed's kick_log[0]), of
that tensor's kick norm: W_in 0.8126, W_h 2.5223, W_g 0.4003, embed.weight 5.7174 (computed at
import; the CHECK prints them). The kick enters loss.backward() through explore_aux_gate's
gradient-injection node, as in S16 (aux = sum_p (p * kick_p).sum()). No trigger: every seed gets
the same kick sizes at the same update.
Logged per run: the kick (update, norms), and the mean Adam exp_avg_sq of each gate tensor (W_in,
W_h, W_g) after updates 119, 121, 600, 2400 and 4800 (an optimizer step post-hook on the run's
two-group optimizer; a run that stops earlier has no later entries).
Seeds 160-199 (40), paired with SLOW_MEM (24/40; batch 2's S5 on 160-179, batch 3's S7 on
180-199), SLOW_HINGE (34/40), and on S16's 20 seeds with KICK (12/20).

RULE (fixed before any run), vs SLOW_MEM over the 40 seeds, outcome DISCOVERED: promising if
b - c >= 4 and McNemar p < 0.10; not if b - c <= 0; otherwise inconclusive. Printed separately:
the 20 seeds where S13's hinge never fired (SLOW_MEM 20/20) and the 20 where it fired (SLOW_MEM
4/20). READING: "the trigger is not needed" if KICK120 >= 18/20 on the first group and >= 10/20
on the second; "an untimed kick costs the easy seeds" if <= 16/20 on the first group; otherwise
neither.

CHECKS: with the kick's norm set to 0 the run equals SLOW_MEM bit for bit (curves and weights,
3600 steps; also batch 2's record); a run past update 120 logs exactly one kick, at update 120;
its logged norms equal the targets; the exp_avg_sq log has the listed updates.
"""

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_slow_hinge as s13
import explore_kick_control as s16
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY
from explore_batch1 import verdict

NAME = "kick_untimed"
IDEA = "one Gaussian kick at update 120, sized as S16's median first kick, with no trigger"
SOURCE = ("batch 6's S16 (random kicks on the hinge's batches did about as well as the hinge); Adam's "
          "response to a single large gradient")
CHANGE = ("SLOW_MEM + one Gaussian tensor added to the .grad of W_in, W_h, W_g, embed.weight at update "
          "120, each at the median norm of S16's first kicks on that tensor")
PAIRING = ("seeds 160-199, paired with SLOW_MEM (S5 / S7), SLOW_HINGE (S13), and on S16's 20 seeds with "
           "KICK (same initial parameters and batches)")
SEEDS = tuple(range(160, 200))
KICK_AT = 120
KICK_BASE = 9_000_000
KICK_PARAMS = s16.KICK_PARAMS
GATE = s5.GATE
LOG_AT = (119, 121, 600, 2400, 4800)
S16_STORE = "kick_control"
TRIG_GOOD, TRIG_HARD, COST_EASY = 18, 10, 16


def target_norms():
    runs = ec.load_store(S16_STORE)["runs"]
    firsts = [r["end"]["kick_log"][0]["kick_norm"] for k, r in sorted(runs.items())
              if k.startswith("KICK|") and r.get("ok") and r["end"].get("kick_log")]
    assert len(firsts) == 20, f"S16 first kicks: {len(firsts)}"
    return [statistics.median(f[i] for f in firsts) for i in range(len(KICK_PARAMS))]


TARGET = target_norms()

ARMS = {
    "KICK120": dict(s5.ARMS["SLOW_MEM"], key="KICK120", seeds=SEEDS, prio=1,
                    label="SLOW_MEM + one Gaussian kick at update 120 (S16's median first-kick norms)",
                    sched=s5.ARMS["SLOW_MEM"]["sched"] + f"; + one Gaussian tensor on W_in, W_h, W_g, "
                          f"embed.weight's .grad at update {KICK_AT} (norms " +
                          ", ".join(f"{v:.4f}" for v in TARGET) + ")"),
}


def kick_seed(seed, update):
    return KICK_BASE + 100_000 * seed + update


class KickOnceBDH(MultiBDH):
    def __init__(self, *args, seed=0, scale=1.0, **kw):
        super().__init__(*args, **kw)
        self.seed, self.scale = seed, scale
        self.n_batches, self.kick_log = 0, []

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and torch.is_grad_enabled():
            self.n_batches += 1
            if self.n_batches == KICK_AT:
                named = dict(self.named_parameters())
                params = [named[n] for n in KICK_PARAMS]
                gen = torch.Generator().manual_seed(kick_seed(self.seed, self.n_batches))
                kicks = []
                for t, p in zip(TARGET, params):
                    z = torch.randn(p.shape, generator=gen, dtype=p.dtype)
                    kicks.append(z * (self.scale * t / z.norm()))
                self.kick_log.append(dict(update=self.n_batches, kick_norm=[float(k.norm()) for k in kicks],
                                          target=[self.scale * t for t in TARGET]))
                aux = sum((p * k).sum() for p, k in zip(params, kicks))
                logits = _Inject.apply(logits, aux)
        return logits, sat, gr, gw


def make_model(a, seed, scale=1.0):
    task = ec.TASK
    torch.manual_seed(seed)
    return KickOnceBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                       gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                       ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), seed=seed, scale=scale)


def stats_fn(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out["kicks"] = len(model.kick_log)
    if step == "end":
        out["kick_log"] = model.kick_log
    return out


def run_path(a, seed, iters, scale=1.0, keep=None):
    holder, log = {}, {"n": 0}

    def post(opt, args, kwargs):
        if len(opt.param_groups) != 2:
            return
        log["n"] += 1
        if log["n"] in LOG_AT:
            log[str(log["n"])] = [float(opt.state[p]["exp_avg_sq"].mean()) for p in opt.param_groups[0]["params"]]

    h = register_optimizer_step_post_hook(post)
    try:
        r = ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK,
                       stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                       builder=lambda aa, s: (lambda: make_model(aa, s, scale)),
                       run_kw=dict(param_groups=s5.make_groups(holder)))
    finally:
        h.remove()
    r["exp_avg_sq"] = {k: v for k, v in log.items() if k != "n"}
    r["exp_avg_sq_names"] = list(holder.get("names", [[]])[0])
    return r


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: make_model(a, 0), a["lr"], stats_fn)]


def check():
    ok = True
    seed, it = 160, 3 * EVAL_EVERY
    a = ARMS["KICK120"]
    k0, k5 = {"at": it}, {"at": it}
    r0 = run_path(a, seed, it, scale=0.0, keep=k0)
    r5 = s5.run_path(s5.ARMS["SLOW_MEM"], seed, it, keep=k5)
    rec5 = [c for c in ec.load_store("slow_mem")["runs"][f"SLOW_MEM|{seed}"]["curve"] if c[0] <= it]
    same0 = (r0["curve"] == r5["curve"] == rec5
             and all(torch.equal(k0["snap"][n], k5["snap"][n]) for n in k5["snap"]))
    rk = run_path(a, seed, EVAL_EVERY)
    log = rk["end"]["kick_log"]
    one = len(log) == 1 and log[0]["update"] == KICK_AT
    nd = max((abs(k - t) / t for k, t in zip(log[0]["kick_norm"], TARGET)), default=float("inf")) if log else float("inf")
    eas = rk["exp_avg_sq"]
    have = sorted(int(u) for u in eas) == [u for u in LOG_AT if u <= EVAL_EVERY]
    jump = (eas["121"][2] / eas["119"][2]) if have else float("nan")
    for nm, v in ((f"target norms = the median over S16's 20 first kicks: W_in {TARGET[0]:.4f}, W_h {TARGET[1]:.4f}, "
                   f"W_g {TARGET[2]:.4f}, embed.weight {TARGET[3]:.4f}", len(TARGET) == 4 and all(t > 0 for t in TARGET)),
                  (f"with the kick's norm set to 0 the run equals SLOW_MEM bit for bit through {it} steps on seed "
                   f"{seed} (curves = a fresh SLOW_MEM run = batch 2's record; weights equal; kicks logged "
                   f"{len(r0['end']['kick_log'])} at norm 0)", same0 and len(r0["end"]["kick_log"]) == 1),
                  (f"a {EVAL_EVERY}-step run logs exactly one kick, at update {KICK_AT} "
                   f"({[e['update'] for e in log]})", one),
                  (f"the logged kick norms equal the targets (max relative difference {nd:.1e} < 1e-5)", nd < 1e-5),
                  (f"the exp_avg_sq log has updates {sorted(int(u) for u in eas)} (W_g's mean exp_avg_sq 121/119 "
                   f"= {jump:.3g})", have)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def reading(n_easy, n_hard):
    if n_easy >= TRIG_GOOD and n_hard >= TRIG_HARD:
        return "the trigger is not needed"
    if n_easy <= COST_EASY:
        return "an untimed kick costs the easy seeds"
    return "neither"


def compare(label, new, old, seeds, fn, what):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    nn = sum(bool(fn(new[s])) for s in both)
    no = sum(bool(fn(old[s])) for s in both)
    print(f"    {what:<13} KICK120 {nn:>2}/{len(both)}   {label} {no:>2}/{len(both)}   KICK120 only {b}, "
          f"{label} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=nn, old=no, b=b, c=c, p=p)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    new = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith("KICK120|") and r.get("ok")}
    sm = s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")
    sh = s13.runs_of((s13.NAME,), "SLOW_HINGE")
    kc = s13.runs_of((S16_STORE,), "KICK")
    fired = set(s16.SEEDS)
    easy = [s for s in SEEDS if s not in fired]
    hard = [s for s in SEEDS if s in fired]
    print("  per seed (ROUTED* at 1200/2400/3600/end; group: easy = S13's hinge never fired, hard = it fired; "
          "W_g exp_avg_sq = mean Adam second moment of W_g after updates 119/121/600/2400/4800):")
    print(f"    {'seed':>4} {'grp':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} {'SLOW_MEM':<15} {'SLOW_HINGE':<15} {'S16 KICK':<15} "
          f"W_g exp_avg_sq 119/121/600/2400/4800")
    for s in SEEDS:
        r = new.get(s)
        grp = "hard" if s in fired else "easy"
        tags = (f"{ec.tag(sm.get(s)):<15} {ec.tag(sh.get(s)):<15} "
                f"{(ec.tag(kc.get(s)) if s in fired else '--'):<15}")
        if r is None:
            print(f"    {s:>4} {grp:>4} {ec.tag(store['runs'].get(f'KICK120|{s}')):<70} {tags}")
            continue
        m = r["end"].get("margin")
        eas = r.get("exp_avg_sq", {})
        es = "/".join(f"{eas[str(u)][2]:.1e}" if str(u) in eas else "--" for u in LOG_AT)
        print(f"    {s:>4} {grp:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
              f"{ec.tag(r):<15} {tags} {es}")
    done = [s for s in SEEDS if s in new]
    xr = lambda step: (lambda r: c2.routed_at(r, step))
    out = {}
    print(f"  vs SLOW_MEM, {len(done)} seeds (the rule):")
    out["slow_mem"] = compare("SLOW_MEM", new, sm, done, ec.discovered, "DISCOVERED")
    compare("SLOW_MEM", new, sm, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    for nm, grp in (("easy (S13's hinge never fired)", easy), ("hard (S13's hinge fired)", hard)):
        g = [s for s in grp if s in new]
        print(f"   {nm}, {len(g)} seeds:")
        out[nm.split()[0]] = compare("SLOW_MEM", new, sm, g, ec.discovered, "DISCOVERED")
        compare("SLOW_MEM", new, sm, g, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  vs SLOW_HINGE, {len(done)} seeds:")
    out["slow_hinge"] = compare("SLOW_HINGE", new, sh, done, ec.discovered, "DISCOVERED")
    compare("SLOW_HINGE", new, sh, done, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    g = [s for s in hard if s in new]
    print(f"  vs S16's KICK, its {len(g)} seeds:")
    out["kick"] = compare("KICK", new, kc, g, ec.discovered, "DISCOVERED")
    compare("KICK", new, kc, g, xr(EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}")
    print(f"  failure classes ({len(done)} seeds): KICK120 {s13.fails(new, done)}   SLOW_MEM {s13.fails(sm, done)}   "
          f"SLOW_HINGE {s13.fails(sh, done)}")
    print("  Adam's second moment, median over seeds of the mean exp_avg_sq after each update (ratio to update 119):")
    for i, nm in enumerate(GATE):
        vals = {u: [new[s]["exp_avg_sq"][str(u)][i] for s in done if str(u) in new[s].get("exp_avg_sq", {})]
                for u in LOG_AT}
        med = {u: statistics.median(v) if v else None for u, v in vals.items()}
        base = med[LOG_AT[0]]
        print(f"    {nm:<5} " + "   ".join(f"{u}: {'--' if med[u] is None else f'{med[u]:.2e}'}"
                                            + ("" if med[u] is None or not base else f" (x{med[u] / base:.3g})")
                                            + f" [n {len(vals[u])}]" for u in LOG_AT))
    n_easy = sum(ec.discovered(new[s]) for s in easy if s in new)
    n_hard = sum(ec.discovered(new[s]) for s in hard if s in new)
    rd = reading(n_easy, n_hard)
    v = verdict(out["slow_mem"])
    print(f"  READING S20 (easy {n_easy}/{sum(1 for s in easy if s in new)}, hard {n_hard}/"
          f"{sum(1 for s in hard if s in new)}; 'the trigger is not needed' if easy >= {TRIG_GOOD} and hard >= "
          f"{TRIG_HARD}; 'an untimed kick costs the easy seeds' if easy <= {COST_EASY}): {rd}")
    return dict(out["slow_mem"], verdict=v, reading=rd, n_easy=n_easy, n_hard=n_hard,
                n_easy_run=sum(1 for s in easy if s in new), n_hard_run=sum(1 for s in hard if s in new),
                slow_hinge=out["slow_hinge"], kick=out["kick"])
