#!/usr/bin/env python
"""
explore_muon_slow.py — EXPLORATORY, not a result. Screen S32 (batch 10): is the hinge still needed
under Muon? S25's MUON_HINGE without the hinge (MUON_SLOW), on 40 seeds.

BACKGROUND (docstring)
- S25 (batch 9, d0c31ce): MUON_HINGE 20/20 DISCOVERED on 160-179, with eta2 by stream = 1.00 at the
  first evaluation (1200) on every seed. The hinge fired 0-4 times on 19 seeds (never on 165, 168,
  169, 170, 174, 179) and 281 times on 166, which was already stream-routed at 1200. MUON_A 9/20;
  every MUON_A run was decided by 1200: stream-routed, or position-split (eta2 by index 0.98-1.00).
  Gate update norm under Muon: max/median 1.15-1.24 (MUON_HINGE), 1.11-1.28 (MUON_A) per seed.
- Same seeds under Adam: SLOW_MEM 12/20, SLOW_HINGE 18/20. On 166, 173, 175, 176 and 177 SLOW_HINGE
  gained after a single hinge firing; on 162, after three.
- S27: the z-loss lowered all logits together (lse^2 -> 0.1) without separating merges (2/8). It is
  the only gate term with a gradient along the all-ones logit direction.

THE CONFIGURATION: the screens' (S=2, P=4, k=2, grouped, no conv, MAX_ITERS 24000), S25's optimizer
(explore_muon_recipe: SliceMuon on every 2-D weight and head slice, Adam on the embedding, torch's
Muon update, momentum 0.95, Nesterov, weight decay 0) and S25's groups and slow schedule, Muon lr 0.005
(S25's selection): the gate group (W_in, W_h, W_g) at the Muon lr throughout; the other Muon weights at
0.1 x the Muon lr and the embedding's Adam at 1e-4 (0.1 x SUB_LR = 1e-3) for updates 1-2400, full
after the evaluation at 2400.
  MUON_SLOW       MUON_HINGE without the hinge (S25's model: test_multilayer_binding.build). Seeds
                  160-199.
  MUON_HINGE      S25's MUON_HINGE (S13's hinge, LAMBDA 1.0, TAU 0.2). Seeds 180-199 here; 160-179
                  are S25's runs, re-run here (arm MUON_HINGE_S25) to the step S25's record stopped
                  at, only to record the extra evaluations and the hinge firings below: each must
                  reproduce S25's curve bit for bit (printed per seed), and its outcome is S25's.
Extra labelled evaluations at updates 300, 600 and 900, in every arm: after the optimizer step of that
update, the routing statistics (test_router_layout.routing_stats: eta2 of the read gate by stream,
key, half and index at key and value positions, and the margin) on the run's own probe batch
(test_multilayer_binding.probe_batch, its own generator; eval mode, no gradients: the training batches
and the RNG are untouched). ROUTED* (margin >= 0.9 and eta_key_by_stream > 0.9) at 300/600/900 from
them, at 1200 from the run's first evaluation.
Logged for MUON_HINGE: the update of every hinge firing (the training batch on which a term was above
TAU).

READINGS (fixed before any run; DISCOVERED; exact McNemar, two-sided; b = MUON_HINGE-only seeds, c =
MUON_SLOW-only seeds, 160-199):
- "the hinge is unnecessary under Muon" if MUON_SLOW >= 37/40 and b - c <= 1;
- "the hinge matters under Muon" if b - c >= 4 and p < 0.1;
- otherwise neither.
Also printed: MUON_SLOW vs SLOW_MEM (160-199), ROUTED* at 300/600/900/1200 per arm, failure classes,
MUON_HINGE's firing updates.

CHECKS: with the extra evaluations, MUON_HINGE|160 equals S25's recorded run bit for bit through 2400
(curve and the statistics at 1200 and 2400); the evaluations are logged at 300, 600, 900; the firing
log has one entry per counted firing; with the hinge's TAU at 1.0, MUON_HINGE equals MUON_SLOW bit for
bit (curves and weights, 1200 steps).
"""

import contextlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import explore_common as ec
import explore_common2 as c2
import explore_slow_hinge as s13
import explore_muon_recipe as s25
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import routing_stats, fail_class
from test_binding_onset import EVAL_EVERY

NAME = "muon_slow"
IDEA = "S25's MUON_HINGE without the hinge (MUON_SLOW): is the hinge still needed under Muon?"
SOURCE = ("batch 9's S25 (MUON_HINGE 20/20, stream-routed at 1200 on every seed; the hinge fired 0-4 times on "
          "19 seeds); Muon has no per-weight second moment")
CHANGE = "S25's MUON_HINGE without the hinge; extra routing evaluations at updates 300, 600, 900"
PAIRING = ("seeds 160-199: MUON_SLOW with MUON_HINGE (S25's runs on 160-179, new runs on 180-199) and with "
           "SLOW_MEM (S5 160-179, S7 180-199); same initial parameters and batches")
MUON_LR = 0.005
EARLY_AT = (300, 600, 900)
SEEDS = tuple(range(160, 200))
S25_STORE = s25.NAME
UNNEEDED_MIN, MATTERS_D, MATTERS_P = 37, 4, 0.1

_base = dict(ec.ARM_A, lr=ec.SUB_LR, iters=ec.MAX_ITERS, prio=1, slow=True, muon_lr=MUON_LR)
_sched = ("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on the "
          "embedding 0.0001 for 1-2400, 0.001 after")
ARMS = {
    "MUON_SLOW": dict(_base, key="MUON_SLOW", seeds=SEEDS, hinge=False,
                      label="S25's MUON_HINGE without the hinge", sched=_sched),
    "MUON_HINGE": dict(_base, key="MUON_HINGE", seeds=tuple(range(180, 200)), hinge=True,
                       label="S25's MUON_HINGE, new seeds", sched=_sched + "; + 1.0 * [relu(eta2_index - 0.2) + "
                                                                        "relu(eta2_half - 0.2)] on each training batch"),
    "MUON_HINGE_S25": dict(_base, key="MUON_HINGE_S25", seeds=tuple(range(160, 180)), hinge=True,
                           label="S25's MUON_HINGE runs re-run to their recorded stop, for the logs only",
                           sched=_sched + "; + the hinge; to S25's recorded stop step"),
}


# ── the optimizer with the extra evaluations ─────────────────────────────────
class MuonAdamEval(s25.MuonAdam):
    def __init__(self, groups, lr, after=None):
        super().__init__(groups, lr)
        self.after = after

    def step(self, closure=None):
        super().step()
        if self.after is not None:
            self.after(self.n)


@contextlib.contextmanager
def muon_eval_in_onset_run(holder, after):
    """s25.muon_in_onset_run, building MuonAdamEval (the after-step callback) instead."""
    def factory(params, lr=1e-3, **kw):
        assert not kw, kw
        opt = MuonAdamEval(list(params), lr, after)
        holder.setdefault("opts", []).append(opt)
        return opt
    torch.optim.Adam = factory
    try:
        yield
    finally:
        torch.optim.Adam = s25._ADAM


class HingeLog(s13.HingeBDH):
    """S13's HingeBDH, logging the training batch (= update) of every firing."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.fired_at = []

    def forward(self, tokens, tau=None, track_sat=False):
        n0 = self.n_active
        out = super().forward(tokens, tau, track_sat)
        if self.n_active > n0:
            self.fired_at.append(self.n_batches)
        return out


def make_hinge_model(a, seed, tau=s13.TAU, lam=s13.LAMBDA):
    task = ec.TASK
    torch.manual_seed(seed)
    return HingeLog(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                    gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"], ctx_tokens=task.ctx_tokens,
                    **a.get("model_kw", {}), pen_lambda=lam, tau=tau, pen_SP=(task.S, task.P))


def hinge_stats(model, task, probe, step):
    out = s13.stats_fn(model, task, probe, step)
    if step == "end":
        out["fired_at"] = list(model.fired_at)
    return out


def run_path(a, seed, iters, keep=None, tau=s13.TAU, early=True):
    holder, box, ev = {}, {}, {}
    probe = probe_batch(ec.TASK, seed)

    def after(n):
        if n in EARLY_AT:
            ev[str(n)] = {k: float(v) for k, v in routing_stats(box["m"], ec.TASK, probe).items()}

    def keep_model(m):
        box["m"] = m
        return m

    if a["hinge"]:
        b, sf = (lambda aa, s: (lambda: keep_model(make_hinge_model(aa, s, tau)))), hinge_stats
    else:
        b, sf = (lambda aa, s: (lambda: keep_model(build(ec.TASK, aa, ARCH, s)))), c2.stats_b2
    with muon_eval_in_onset_run(holder, after if early else None):
        rec = ec.run_one(a, seed, iters, check=s25.make_check(holder), keep=keep, task=ec.TASK, stats_fn=sf,
                         grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=b,
                         run_kw=dict(param_groups=s25.make_groups(holder, a["muon_lr"], True)))
    opts = holder.get("opts", [])
    if rec.get("ok"):
        u = opts[0].upd if opts else []
        rec.update(muon_lr=a["muon_lr"], early_evals=ev, n_opts_built=len(opts),
                   opt_groups=[dict(tag=g["tag"], kind=g["kind"], names=g["names"], lr_end=g["lr"])
                               for g in holder["groups"]],
                   gate_upd=dict(n=len(u), median=statistics.median(u), max=max(u)) if u else None)
    return rec


def s25_record(seed):
    return ec.load_store(S25_STORE)["runs"][f"MUON_HINGE|{seed}"]


def iters_for(arm, seed):
    a = ARMS[arm]
    if arm == "MUON_HINGE_S25":
        return min(a["iters"], s25_record(seed)["stopped_at"])
    return a["iters"]


def run_job(arm, seed):
    a = ARMS[arm]
    rec = run_path(a, seed, iters_for(arm, seed))
    if arm == "MUON_HINGE_S25" and rec.get("ok"):
        ref = s25_record(seed)
        rec["s25_curve_equal"] = rec["curve"] == ref["curve"] if rec["stopped_at"] == ref["stopped_at"] else \
            rec["curve"] == [c for c in ref["curve"] if c[0] <= rec["stopped_at"]]
        rec["s25_stopped_at"] = ref["stopped_at"]
    return rec


def segments(arm):
    a = ARMS[arm]
    mk = (lambda: make_hinge_model(a, 0)) if a["hinge"] else (lambda: build(ec.TASK, a, ARCH, 0))
    return [(a["iters"], ec.TASK, mk, a["lr"], hinge_stats if a["hinge"] else c2.stats_b2)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def norm(x):
    return json.loads(json.dumps(x))


def check():
    ok = True
    rows = []
    t = 2 * EVAL_EVERY
    a = ARMS["MUON_HINGE"]
    r = run_path(a, 160, t)
    ref = s25_record(160)
    want = [c for c in ref["curve"] if c[0] <= t]
    st_new = {s["step"]: s for s in norm(r["stats"]) if s["step"] in (EVAL_EVERY, t)}
    st_ref = {s["step"]: s for s in ref["stats"] if s["step"] in (EVAL_EVERY, t)}
    same_st = st_new.keys() == st_ref.keys() and all(st_new[k] == st_ref[k] for k in st_ref)
    rows.append((f"with the extra evaluations, MUON_HINGE|160 equals S25's recorded run bit for bit through {t} (curve "
                 f"{r['curve'] == want}; statistics at {EVAL_EVERY} and {t} {same_st})", r["curve"] == want and same_st))
    rows.append((f"the extra evaluations are logged at updates {sorted(int(k) for k in r['early_evals'])} (ROUTED* "
                 + ", ".join(f"{k}: {c2.routed_star(v)}" for k, v in sorted(r['early_evals'].items(), key=lambda x: int(x[0])))
                 + ")", sorted(int(k) for k in r["early_evals"]) == list(EARLY_AT)))
    fa, hc = r["end"].get("fired_at"), r["end"].get("hinge_counts")
    rows.append((f"the firing log has one entry per counted firing ({fa} vs hinge counts {hc})",
                 fa is not None and hc is not None and len(fa) == hc[0]))
    k1, k2 = {"at": EVAL_EVERY}, {"at": EVAL_EVERY}
    r1 = run_path(a, 160, EVAL_EVERY, keep=k1, tau=1.0)
    r2 = run_path(ARMS["MUON_SLOW"], 160, EVAL_EVERY, keep=k2)
    same = r1["curve"] == r2["curve"] and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"])
    rows.append((f"with the hinge's TAU at 1.0, MUON_HINGE equals MUON_SLOW bit for bit (seed 160, {EVAL_EVERY} steps; "
                 f"curves {r1['curve']} vs {r2['curve']}; weights equal; hinge on {r1['end'].get('hinge_counts')})",
                 same and r1["end"]["hinge_counts"][0] == 0))
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


def routed_early(r, step):
    if step == EVAL_EVERY:
        return c2.routed_at(r, EVAL_EVERY)
    d = (r.get("early_evals") or {}).get(str(step))
    return c2.routed_star(d)


def reading(d, n_slow):
    if n_slow >= UNNEEDED_MIN and d["b"] - d["c"] <= 1:
        return "the hinge is unnecessary under Muon"
    if d["b"] - d["c"] >= MATTERS_D and d["p"] < MATTERS_P:
        return "the hinge matters under Muon"
    return "neither"


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    s25r = {s: r for s, r in ((int(k.split("|")[1]), r) for k, r in ec.load_store(S25_STORE)["runs"].items()
                              if k.startswith("MUON_HINGE|"))}
    hinge = dict(got["MUON_HINGE"])
    for s, r in got["MUON_HINGE_S25"].items():
        hinge[s] = r
    slow = got["MUON_SLOW"]
    sm = s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")
    sh = s13.runs_of((s13.NAME,), "SLOW_HINGE")
    steps = EARLY_AT + (EVAL_EVERY,)
    flag = lambda r: "".join("Y" if routed_early(r, t) else "-" for t in steps)
    print(f"  per seed (ROUTED* at 300/600/900/1200 as Y/-; MUON_HINGE 160-179 = S25's runs, re-run here for the logs "
          f"and checked against S25's curve):")
    print(f"    {'seed':>4} | {'MUON_SLOW':<15} {'R*300-1200':>10} {'acc':>6} {'trans':>6} | {'MUON_HINGE':<15} "
          f"{'R*300-1200':>10} {'trans':>6} {'firings at updates':<34} {'=S25':>5} | {'SLOW_MEM':<15} {'SLOW_HINGE':<15}")
    for s in SEEDS:
        rs, rh = slow.get(s), hinge.get(s)
        a1 = (f"{ec.tag(rs):<15} {flag(rs):>10} {rs['acc']:6.3f} {str(rs['transition']):>6}" if rs
              else f"{'not run':<15} {'':>10} {'':>6} {'':>6}")
        if rh:
            fa = rh["end"].get("fired_at") or []
            fs = (str(fa[:6])[:-1] + (f", ... ({len(fa)})]" if len(fa) > 6 else "]")) if fa else "none"
            eq = rh.get("s25_curve_equal")
            a2 = (f"{ec.tag(rh):<15} {flag(rh):>10} {str(rh['transition']):>6} {fs:<34} "
                  f"{('yes' if eq else 'NO') if eq is not None else '--':>5}")
        else:
            a2 = f"{'not run':<15} {'':>10} {'':>6} {'':<34} {'':>5}"
        print(f"    {s:>4} | {a1} | {a2} | {ec.tag(sm.get(s)):<15} {ec.tag(sh.get(s)):<15}")
    rer = got["MUON_HINGE_S25"]
    neq = [s for s, r in rer.items() if not r.get("s25_curve_equal")]
    print(f"  MUON_HINGE 160-179 re-runs reproduce S25's curves: {len(rer) - len(neq)}/{len(rer)}"
          + (f" (NOT: {neq})" if neq else ""))
    out = {}
    print(f"  MUON_HINGE vs MUON_SLOW ({sum(1 for s in SEEDS if s in hinge and s in slow)} seeds; the rule):")
    d = compare("MUON_SLOW", hinge, slow, SEEDS, ec.discovered, "DISCOVERED", "MUON_HINGE")
    compare("MUON_SLOW", hinge, slow, SEEDS, ec.bound, "BOUND", "MUON_HINGE")
    for t in steps:
        compare("MUON_SLOW", hinge, slow, SEEDS, lambda r, t=t: routed_early(r, t), f"ROUTED*@{t}", "MUON_HINGE")
    print("  MUON_SLOW vs SLOW_MEM (Adam), 160-199:")
    dsm = compare("SLOW_MEM", slow, sm, SEEDS, ec.discovered, "DISCOVERED", "MUON_SLOW")
    compare("SLOW_MEM", slow, sm, SEEDS, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "MUON_SLOW")
    print("  MUON_HINGE vs SLOW_HINGE (Adam), 160-199 (printed):")
    compare("SLOW_HINGE", hinge, sh, SEEDS, ec.discovered, "DISCOVERED", "MUON_HINGE")
    print(f"  ROUTED* at 300/600/900/1200: MUON_SLOW " + "/".join(str(sum(routed_early(r, t) for r in slow.values()))
                                                               for t in steps) + f" of {len(slow)}; MUON_HINGE "
          + "/".join(str(sum(routed_early(r, t) for r in hinge.values())) for t in steps) + f" of {len(hinge)}")
    print(f"  failure classes: MUON_SLOW {s13.fails(slow, SEEDS)}   MUON_HINGE {s13.fails(hinge, SEEDS)}   SLOW_MEM "
          f"{s13.fails(sm, SEEDS)}")
    fired = {s: (r["end"].get("fired_at") or []) for s, r in hinge.items()}
    print(f"  MUON_HINGE hinge firings: {sum(len(v) for v in fired.values())} in total; runs with at least one "
          f"{sum(1 for v in fired.values() if v)}/{len(fired)}; firings by update 300/600/900/1200/after: "
          + "/".join(str(sum(1 for v in fired.values() for u in v if lo < u <= hi))
                     for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1200), (1200, 10 ** 9))))
    rd = reading(d, d["old"])
    print(f"  RULE S32 (MUON_SLOW DISCOVERED {d['old']}/{d['n']}; MUON_HINGE only {d['b']}, MUON_SLOW only {d['c']}, p = "
          f"{d['p']:.3g}; 'unnecessary' if MUON_SLOW >= {UNNEEDED_MIN}/40 and b - c <= 1, 'matters' if b - c >= "
          f"{MATTERS_D} and p < {MATTERS_P}): {rd}")
    return dict(d, rule=rd, slow_mem=dsm)
