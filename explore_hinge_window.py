#!/usr/bin/env python
"""
explore_hinge_window.py — EXPLORATORY, not a result. Screen S34 (batch 11): is the hinge needed
after the slow phase? The hinge active only on updates 1-2400 (H2400), under Muon and under Adam.

BACKGROUND (docstring)
- S32 (batch 10): MUON_HINGE 39/40 vs MUON_SLOW 31/40 (8 vs 0, p = 0.0078); MUON_SLOW's failures are
  8 position splits and 1 key split. On the 8 rescued seeds the hinge first fired at updates 50-168.
  ROUTED* 0/40 at 300 and 600 in both arms; at 900, 38/40 vs 27/40. Firings by update
  300/600/900/1200/after: 47/9/0/0/283; the late firings are on 166 (281 in total, bound at 10800),
  172, 180, 190, 191.
- Under Adam: SLOW_HINGE 34/40 vs SLOW_MEM 24/40 (S13); the firing timing was not logged.
- S33(a), S=4 k=4: Muon does not break merges. The main line's working four-stream configuration is
  k=16 (HINGE4k16 17/20 on X, 35/40 pooled), untested under Muon (S35).

THE CONFIGURATION: the screens' (S=2, P=4, k=2, grouped, no conv, MAX_ITERS test_short_conv.MAX_ITERS
= 24000, lr test_channel_binding.SUB_LR = 1e-3 for the Adam groups, passed explicitly).
  H2400 = S13's hinge (explore_slow_hinge.HingeBDH's terms, LAMBDA 1.0, TAU 0.2, through S12's
  gradient-injection node) with weight 1.0 on the training batches of updates 1-2400 (the slow phase)
  and 0 afterwards: from update 2401 the penalty is not injected at all (the base model's forward); its
  two terms are still computed without gradient and an update where one is above TAU is logged as a
  "would-fire" (not a firing).
  MUON_H2400  S25's MUON_HINGE (S25's groups and slow schedule, Muon lr 0.005; explore_muon_recipe)
              with the window. Seeds 160-199, paired with MUON_HINGE (S25 160-179 + S32 180-199) and
              MUON_SLOW (S32).
  ADAM_H2400  S13's SLOW_HINGE (explore_slow_mem's groups and lr switch, Adam) with the window. Seeds
              160-199, paired with SLOW_HINGE (S13) and SLOW_MEM (S5 160-179, S7 180-199).
READINGS (fixed before any run; per optimizer; DISCOVERED; b = full-hinge-only seeds, c = window-only
seeds; exact McNemar, two-sided):
  - "an early window suffices" if b - c <= 1;
  - "the late hinge matters" if b - c >= 4 and p < 0.1;
  - otherwise neither.
  Also printed: each window arm against its no-hinge arm (MUON_SLOW; SLOW_MEM).
DIAGNOSTICS (every new run):
  - the update of every firing (and of every would-fire after 2400);
  - at each firing in updates 1-2400: the norm of the hinge's gradient on the gate (W_in, W_h, W_g
    pooled; torch.autograd.grad of the injected term, retain_graph, which leaves the training
    gradient untouched) and of the task's gradient (the accumulated gradient minus the hinge's, read
    before the optimizer step), and their ratio;
  - MUON_H2400 only: for the 200 updates after each firing, the cosine between W_g's Muon momentum
    buffer (after each update, the firing's own included as offset 0) and that firing's hinge
    gradient on W_g; per firing, the first offset at which it is below 0.5 (none within 200: censored,
    counted as 200 in the median).
  - Labelled description: "directional kick" if the median ratio (MUON_H2400's firings) is >= 100 and
    the median offset at which the cosine falls below 0.5 is >= 50 updates; otherwise "not a
    directional kick".

CHECKS: with the window end at 24000, MUON_H2400 equals S25's recorded MUON_HINGE|160 and ADAM_H2400
equals S13's recorded SLOW_HINGE|160 bit for bit through 2400 (curve, statistics at 1200 and 2400;
diagnostics on); with the window end at 0, they equal MUON_SLOW|160 and SLOW_MEM|160 through 1200
(fresh runs: curves and weights; and the recorded curves); with the window end at 2400 the hinge's
weight is 1.0 at update 2400 and 0 at 2401 (2401-step runs, both arms); every firing has one
diagnostic row (and, under Muon, one cosine track).
"""

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import torch.optim.optimizer as topt

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_slow_pos as s12
import explore_slow_hinge as s13
import explore_muon_recipe as s25
import explore_muon_slow as s32
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY

NAME = "hinge_window"
IDEA = "the hinge only during the slow phase (updates 1-2400), under Muon and under Adam"
SOURCE = ("batch 10's S32 (the hinge matters under Muon; it fired at updates 50-168 on the rescued seeds, "
          "late firings on 5 seeds); S13 (SLOW_HINGE under Adam, firing timing not logged)")
CHANGE = "S13's hinge with weight 1.0 on updates 1-2400 and 0 after (not injected), in MUON_HINGE and SLOW_HINGE"
PAIRING = ("seeds 160-199: MUON_H2400 with MUON_HINGE (S25 160-179, S32 180-199) and MUON_SLOW (S32); ADAM_H2400 "
           "with SLOW_HINGE (S13) and SLOW_MEM (S5 160-179, S7 180-199); same initial parameters and batches")
WINDOW = 2400
MUON_LR = s32.MUON_LR                          # 0.005 (S25's selection)
SEEDS = tuple(range(160, 200))
GATE = tuple(s25.GATE)                         # W_in, W_h, W_g
COS_SPAN, COS_THR = 200, 0.5
COS_AT = (0, 1, 5, 10, 25, 50, 100, 200)
KICK_RATIO, KICK_SPAN = 100.0, 50
SUFFICES_D, MATTERS_D, MATTERS_P = 1, 4, 0.1
W_LOG_AT = (1, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2)

_hinge = f"; + {s13.LAMBDA} * [relu(eta2_index - {s13.TAU}) + relu(eta2_half - {s13.TAU})] on updates 1-{WINDOW} only"
ARMS = {
    "MUON_H2400": dict(s32.ARMS["MUON_SLOW"], key="MUON_H2400", seeds=SEEDS, hinge=True, opt="muon", window=WINDOW,
                       prio=1, label="S25's MUON_HINGE with the hinge on updates 1-2400 only",
                       sched=s32.ARMS["MUON_SLOW"]["sched"] + _hinge),
    "ADAM_H2400": dict(s13.ARMS["SLOW_HINGE"], key="ADAM_H2400", seeds=SEEDS, opt="adam", window=WINDOW, prio=1,
                       label="S13's SLOW_HINGE with the hinge on updates 1-2400 only",
                       sched=s13.ARMS["SLOW_HINGE"]["sched"].split("; +")[0] + _hinge),
}


class WindowHinge(s13.HingeBDH):
    """S13's HingeBDH with a weight window: on training batches 1..window the penalty is computed and
    injected exactly as HingeBDH does (same expression); afterwards the base forward only, the two
    terms computed without gradient for the would-fire log. Logs firings and, at each firing, the hinge's
    gradient on the gate (for the diagnostics; resolved against the accumulated gradient before the
    optimizer step)."""

    def __init__(self, *args, window=None, diag=True, **kw):
        super().__init__(*args, **kw)
        self.window, self.diag = window, diag
        self.fired_at, self.would_at, self.w_log = [], [], {}
        self.pending, self.firing_diag, self.tracks = None, [], []

    def weight(self, u):
        return self.pen_lambda if (self.window is None or u <= self.window) else 0.0

    def forward(self, tokens, tau=None, track_sat=False):
        if not (self.training and self.pen_lambda is not None and torch.is_grad_enabled()):
            return MultiBDH.forward(self, tokens, tau, track_sat)
        u = self.n_batches + 1
        w = self.weight(u)
        if u in W_LOG_AT:
            self.w_log[str(u)] = float(w)
        logits, sat, gr, gw = MultiBDH.forward(self, tokens, tau, track_sat)
        if w:
            e_i, e_h = s12.pos_penalty(gr, *self.pen_SP)
            h_i, h_h = F.relu(e_i - self.tau), F.relu(e_h - self.tau)
            self.n_batches += 1
            fired = bool(h_i > 0) or bool(h_h > 0)
            self.n_active += int(fired)
            self.pen_last = (float(e_i.detach()), float(e_h.detach()))
            pen = self.pen_lambda * (h_i + h_h)
            if fired:
                self.fired_at.append(u)
                if self.diag:
                    ps = [getattr(self, n) for n in GATE]
                    hg = torch.autograd.grad(pen, ps, retain_graph=True, allow_unused=True)
                    self.pending = dict(update=u, hinge=[torch.zeros_like(p) if g is None else g.detach().clone()
                                                         for p, g in zip(ps, hg)])
            logits = _Inject.apply(logits, pen)
        else:
            with torch.no_grad():
                e_i, e_h = s12.pos_penalty(gr, *self.pen_SP)
            self.n_batches += 1
            self.pen_last = (float(e_i), float(e_h))
            if float(e_i) > self.tau or float(e_h) > self.tau:
                self.would_at.append(u)
        return logits, sat, gr, gw


def make_model(a, seed, window, tau=s13.TAU, lam=s13.LAMBDA, diag=True):
    """s13.make_model's construction (same RNG draws) with WindowHinge."""
    task = ec.TASK
    torch.manual_seed(seed)
    return WindowHinge(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                       gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"], ctx_tokens=task.ctx_tokens,
                       **a.get("model_kw", {}), pen_lambda=lam, tau=tau, pen_SP=(task.S, task.P), window=window,
                       diag=diag)


def _cos(x, y):
    nx, ny = float(x.norm()), float(y.norm())
    return float((x * y).sum()) / (nx * ny) if nx > 0 and ny > 0 else float("nan")


def install_diag(box, muon):
    """Global optimizer hooks for one run (removed by the caller): before the first optimizer step of an
    update, resolve a pending firing (task gradient = accumulated gradient - hinge gradient on the gate);
    after each Muon step, the cosine tracks of the last COS_SPAN updates' firings."""
    def pre(opt, args, kwargs):
        m = box.get("m")
        if m is None or m.pending is None:
            return
        p, m.pending = m.pending, None
        ps = [getattr(m, n) for n in GATE]
        hn = math.sqrt(sum(float((h ** 2).sum()) for h in p["hinge"]))
        tn = math.sqrt(sum(float(((q.grad - h) ** 2).sum()) for q, h in zip(ps, p["hinge"])))
        m.firing_diag.append(dict(update=p["update"], hinge=hn, task=tn, ratio=hn / tn if tn > 0 else float("inf")))
        if muon:
            m.tracks.append(dict(update=p["update"], g=p["hinge"][GATE.index("W_g")], below=None, cos={}, done=False))

    def post(opt, args, kwargs):
        if not isinstance(opt, s25.SliceMuon):
            return
        m = box.get("m")
        if m is None or not m.tracks:
            return
        st = opt.state.get(m.W_g)
        if not st or "momentum_buffer" not in st:
            return
        buf = st["momentum_buffer"]
        u = m.n_batches
        for t in m.tracks:
            if t["done"]:
                continue
            d = u - t["update"]
            c = _cos(buf, t["g"])
            if d in COS_AT:
                t["cos"][str(d)] = c
            if t["below"] is None and c < COS_THR:
                t["below"] = d
            if d >= COS_SPAN:
                t["done"] = True
    return [topt.register_optimizer_step_pre_hook(pre), topt.register_optimizer_step_post_hook(post)]


def stats_fn(model, task, probe, step):
    out = s13.stats_fn(model, task, probe, step)
    if step == "end":
        out.update(fired_at=list(model.fired_at), would_at=list(model.would_at), w_log=dict(model.w_log),
                   firing_diag=list(model.firing_diag),
                   cos_tracks=[dict(update=t["update"], below=t["below"], cos=t["cos"],
                                    complete=t["done"]) for t in model.tracks])
    return out


def run_path(a, seed, iters, window=None, keep=None, diag=True):
    """One run of arm a (opt 'muon' or 'adam') with the hinge window (None: a["window"])."""
    window = a["window"] if window is None else window
    box, holder = {}, {}

    def builder(aa, s):
        def make():
            m = make_model(aa, s, window, diag=diag)
            box["m"] = m
            return m
        return make
    hooks = install_diag(box, a["opt"] == "muon") if diag else []
    try:
        if a["opt"] == "muon":
            with s25.muon_in_onset_run(holder):
                rec = ec.run_one(a, seed, iters, check=s25.make_check(holder), keep=keep, task=ec.TASK, stats_fn=stats_fn,
                                 grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder,
                                 run_kw=dict(param_groups=s25.make_groups(holder, a["muon_lr"], True)))
        else:
            rec = ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK, stats_fn=stats_fn,
                             grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder,
                             run_kw=dict(param_groups=s5.make_groups(holder)))
    finally:
        for h in hooks:
            h.remove()
    if rec.get("ok"):
        rec.update(window=window, opt=a["opt"], muon_lr=a.get("muon_lr") if a["opt"] == "muon" else None)
    return json.loads(json.dumps(rec, default=float))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: make_model(a, 0, a["window"]), a["lr"], stats_fn)]


def time_step(arm, steps=400):
    """(seconds per training step: the median interval between optimizer steps over one run of `steps`
    steps, diagnostics on; seconds per evaluation) for the projection, run concurrently with the other arms'
    timing (the pool's load)."""
    import time
    a = ARMS[arm]
    stamps = []

    def post(opt, args, kwargs):
        if a["opt"] == "adam" or isinstance(opt, s25.SliceMuon):
            stamps.append(time.perf_counter())
    h = topt.register_optimizer_step_post_hook(post)
    try:
        run_path(a, 0, steps)
    finally:
        h.remove()
    d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    te = c2.time_eval(ec.TASK, lambda: make_model(a, 0, a["window"]), stats_fn)
    return d[len(d) // 2], te


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _stats_at(stats, steps):
    return {s["step"]: s for s in stats if s["step"] in steps}


def check():
    ok = True
    rows = []
    t2, t1 = 2 * EVAL_EVERY, EVAL_EVERY
    s25rec = ec.load_store(s25.NAME)["runs"]["MUON_HINGE|160"]
    s13rec = s13.runs_of((s13.NAME,), "SLOW_HINGE")[160]
    for arm, ref, lab in (("MUON_H2400", s25rec, "S25's MUON_HINGE|160"), ("ADAM_H2400", s13rec, "S13's SLOW_HINGE|160")):
        r = run_path(ARMS[arm], 160, t2, window=24000)
        want = [c for c in ref["curve"] if c[0] <= t2]
        sn, sr = _stats_at(r["stats"], (t1, t2)), _stats_at(ref["stats"], (t1, t2))
        same_st = sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr)
        fd, fa = r["end"]["firing_diag"], r["end"]["fired_at"]
        tr = r["end"]["cos_tracks"]
        diag_ok = len(fd) == len(fa) and [d["update"] for d in fd] == fa and (
            arm != "MUON_H2400" or [t["update"] for t in tr] == fa)
        rows.append((f"{arm} with the window end at 24000 equals {lab} bit for bit through {t2} (curve {r['curve'] == want}; "
                     f"statistics at {t1} and {t2} {same_st}; diagnostics on)", r["curve"] == want and same_st))
        rows.append((f"{arm}: one diagnostic row per firing (firings {fa}; ratios "
                     f"{[round(d['ratio'], 1) for d in fd]}" + (f"; cosine below {COS_THR} after "
                     f"{[t['below'] for t in tr]} updates" if arm == "MUON_H2400" else "") + ")", diag_ok))
    k1, k2 = {"at": t1}, {"at": t1}
    r1 = run_path(ARMS["MUON_H2400"], 160, t1, window=0, keep=k1)
    r2 = s32.run_path(s32.ARMS["MUON_SLOW"], 160, t1, keep=k2, early=False)
    rec = [c for c in ec.load_store(s32.NAME)["runs"]["MUON_SLOW|160"]["curve"] if c[0] <= t1]
    same = r1["curve"] == r2["curve"] == rec and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"])
    rows.append((f"MUON_H2400 with the window end at 0 equals MUON_SLOW|160 bit for bit through {t1} (curves {r1['curve']} "
                 f"= a fresh MUON_SLOW run = S32's record; weights equal; firings {r1['end']['fired_at'][:3]})",
                 same and r1["end"]["fired_at"] == []))
    k1, k2 = {"at": t1}, {"at": t1}
    r1 = run_path(ARMS["ADAM_H2400"], 160, t1, window=0, keep=k1)
    r2 = s5.run_path(s5.ARMS["SLOW_MEM"], 160, t1, keep=k2)
    rec = [c for c in s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")[160]["curve"] if c[0] <= t1]
    same = r1["curve"] == r2["curve"] == rec and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"])
    rows.append((f"ADAM_H2400 with the window end at 0 equals SLOW_MEM|160 bit for bit through {t1} (curves {r1['curve']} "
                 f"= a fresh SLOW_MEM run = S5's record; weights equal; firings {r1['end']['fired_at'][:3]})",
                 same and r1["end"]["fired_at"] == []))
    for arm in ARMS:
        r = run_path(ARMS[arm], 160, WINDOW + 1)
        wl = r["end"]["w_log"]
        rows.append((f"{arm} with the window end at {WINDOW}: the hinge's weight at updates {sorted(int(k) for k in wl)} "
                     f"is {[wl[str(k)] for k in sorted(int(k) for k in wl)]} (1.0 at {WINDOW}, 0 at {WINDOW + 1})",
                     wl.get(str(WINDOW)) == 1.0 and wl.get(str(WINDOW + 1)) == 0.0 and wl.get("1") == 1.0))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def full_hinge_muon():
    """MUON_HINGE per seed: S25's outcome on 160-179 (S32's re-runs for the firing logs, curves equal to
    S25's), S32's runs on 180-199."""
    st = ec.load_store(s32.NAME)["runs"]
    s25r = ec.load_store(s25.NAME)["runs"]
    out = {}
    for s in SEEDS:
        if s < 180:
            r = s25r.get(f"MUON_HINGE|{s}")
            log = st.get(f"MUON_HINGE_S25|{s}")
            if r is not None and log is not None:
                r = dict(r, end=dict(r["end"], fired_at=log["end"].get("fired_at")))
        else:
            r = st.get(f"MUON_HINGE|{s}")
        if r is not None and r.get("ok"):
            out[s] = r
    return out


def compare(new, old, seeds, fn, lab_new, lab_old, what="DISCOVERED"):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    p = mcnemar_exact(b, c)
    nn = sum(bool(fn(new[s])) for s in both)
    no = sum(bool(fn(old[s])) for s in both)
    print(f"    {what:<13} {lab_new} {nn:>2}/{len(both)}   {lab_old} {no:>2}/{len(both)}   {lab_new} only {b}, "
          f"{lab_old} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=nn, old=no, b=b, c=c, p=p)


def reading(d):
    """d from compare(full, window): b = full only, c = window only."""
    if d["b"] - d["c"] <= SUFFICES_D:
        return "an early window suffices"
    if d["b"] - d["c"] >= MATTERS_D and d["p"] < MATTERS_P:
        return "the late hinge matters"
    return "neither"


def short(xs, n=5):
    if not xs:
        return "none"
    return str(xs[:n])[:-1] + (f", ... ({len(xs)})]" if len(xs) > n else "]")


def bins(xs):
    edges = ((0, 300), (300, 600), (600, 900), (900, 1200), (1200, 2400), (2400, 10 ** 9))
    return "/".join(str(sum(1 for u in xs if lo < u <= hi)) for lo, hi in edges)


def med(xs):
    return statistics.median(xs) if xs else None


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    mw, aw = got["MUON_H2400"], got["ADAM_H2400"]
    mh, ms = full_hinge_muon(), s13.runs_of((s32.NAME,), "MUON_SLOW")
    ah, am = s13.runs_of((s13.NAME,), "SLOW_HINGE"), s13.runs_of(s13.STORES_SLOW_MEM, "SLOW_MEM")
    print(f"  per seed (firings = training batches with a hinge term above TAU {s13.TAU}; 'after' = would-fire updates "
          f"after {WINDOW}, weight 0; full-hinge firings: S32's logs (MUON); S13's hinge count (ADAM, timing not logged)):")
    print(f"    {'seed':>4} | {'MUON_H2400':<11} {'trans':>5} {'firings':<26} {'after':>5} | {'MUON_HINGE':<11} "
          f"{'>2400':>5} | {'MUON_SLOW':<11} || {'ADAM_H2400':<11} {'trans':>5} {'firings':<26} {'after':>5} | "
          f"{'SLOW_HINGE':<11} {'count':>5} | {'SLOW_MEM':<11}")
    for s in SEEDS:
        def cell(r):
            if r is None:
                return f"{'not run':<11} {'':>5} {'':<26} {'':>5}"
            return (f"{ec.tag(r):<11} {str(r['transition']):>5} {short(r['end'].get('fired_at') or [], 4):<26} "
                    f"{len(r['end'].get('would_at') or []):>5}")
        h = mh.get(s)
        late = len([u for u in (h["end"].get("fired_at") or []) if u > WINDOW]) if h else None
        hc = (ah.get(s) or {}).get("end", {}).get("hinge_counts")
        print(f"    {s:>4} | {cell(mw.get(s))} | {ec.tag(h):<11} {'' if late is None else late:>5} | {ec.tag(ms.get(s)):<11} || "
              f"{cell(aw.get(s))} | {ec.tag(ah.get(s)):<11} {'' if not hc else hc[0]:>5} | {ec.tag(am.get(s)):<11}")
    out = {}
    for opt, win, full, base, lw, lf, lb in (("MUON", mw, mh, ms, "MUON_H2400", "MUON_HINGE", "MUON_SLOW"),
                                              ("ADAM", aw, ah, am, "ADAM_H2400", "SLOW_HINGE", "SLOW_MEM")):
        print(f"  {opt}: {lf} (full hinge) vs {lw} (window; the rule):")
        d = compare(full, win, SEEDS, ec.discovered, lf, lw)
        compare(full, win, SEEDS, ec.bound, lf, lw, "BOUND")
        compare(full, win, SEEDS, lambda r: c2.routed_at(r, EVAL_EVERY), lf, lw, f"ROUTED*@{EVAL_EVERY}")
        print(f"  {opt}: {lw} vs {lb} (no hinge; printed):")
        db = compare(win, base, SEEDS, ec.discovered, lw, lb)
        diff = sorted(s for s in SEEDS if s in win and s in full and ec.discovered(win[s]) != ec.discovered(full[s]))
        same_curve = sum(1 for s in SEEDS if s in win and s in full and win[s]["curve"] == full[s]["curve"])
        print(f"    seeds where the window and the full hinge differ (DISCOVERED): {diff}; identical curves "
              f"{same_curve}/{sum(1 for s in SEEDS if s in win and s in full)}")
        print(f"    failure classes: {lw} {s13.fails(win, SEEDS)}   {lf} {s13.fails(full, SEEDS)}   {lb} {s13.fails(base, SEEDS)}")
        fa = [u for r in win.values() for u in (r["end"].get("fired_at") or [])]
        wa = {s: len(r["end"].get("would_at") or []) for s, r in sorted(win.items()) if r["end"].get("would_at")}
        print(f"    {lw} firings: {len(fa)} in {sum(1 for r in win.values() if r['end'].get('fired_at'))}/{len(win)} runs; by "
              f"update 300/600/900/1200/2400/after: {bins(fa)}; would-fire after {WINDOW} (weight 0), per seed: {wa}")
        fd = [d_ for r in win.values() for d_ in (r["end"].get("firing_diag") or [])]
        ratios = [d_["ratio"] for d_ in fd]
        print(f"    {lw} at its {len(fd)} firings: hinge-gradient norm on the gate median {med([d_['hinge'] for d_ in fd])}, "
              f"task-gradient norm median {med([d_['task'] for d_ in fd])}; ratio median {med(ratios)}, range "
              f"{(min(ratios), max(ratios)) if ratios else '--'}")
        rd = reading(d)
        print(f"  RULE S34 {opt} ({lf} only {d['b']}, {lw} only {d['c']}, p = {d['p']:.3g}; 'suffices' if b - c <= "
              f"{SUFFICES_D}, 'matters' if b - c >= {MATTERS_D} and p < {MATTERS_P}): {rd}")
        out[opt] = dict(d, rule=rd, vs_base=db, ratio_median=med(ratios), n_fire=len(fd),
                        complete=len(win) == len(ARMS[lw]["seeds"]))
    tr = [t for r in mw.values() for t in (r["end"].get("cos_tracks") or [])]
    below = [COS_SPAN if t["below"] is None else t["below"] for t in tr]
    cens = sum(1 for t in tr if t["below"] is None)
    cos_med = {k: med([t["cos"][str(k)] for t in tr if str(k) in t["cos"]]) for k in COS_AT}
    rmed, bmed = out["MUON"]["ratio_median"], med(below)
    kick = rmed is not None and bmed is not None and rmed >= KICK_RATIO and bmed >= KICK_SPAN
    print(f"  MUON_H2400 cosine of W_g's momentum with each firing's hinge gradient ({len(tr)} firings): median updates "
          f"until below {COS_THR} = {bmed} ({cens} censored at {COS_SPAN}); per firing {below}; median cosine at offsets "
          + ", ".join(f"{k}: {'--' if v is None else f'{v:.3f}'}" for k, v in cos_med.items()))
    lab = "directional kick" if kick else "not a directional kick"
    print(f"  DESCRIPTION S34 (median ratio {rmed} vs {KICK_RATIO:g}; median updates until the cosine is below {COS_THR}: "
          f"{bmed} vs {KICK_SPAN}): {lab}")
    out.update(kick=lab, cos_below_median=bmed, censored=cens, n_tracks=len(tr))
    return out
