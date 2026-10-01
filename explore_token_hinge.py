#!/usr/bin/env python
"""
explore_token_hinge.py — EXPLORATORY, not a result. Screen S19 (batch 7): S13's SLOW_HINGE with a
third hinge term on the key token's identity.

BACKGROUND (post hoc, from batches 5-6)
- S16: random kicks sized as the hinge gradient, on the batches where it would fire, discovered
  12/20 vs SLOW_HINGE 14/20 and SLOW_MEM 4/20 on the 20 seeds where the hinge fired. Direction
  mattered on 3 seeds (173, 190, 196); kicks routed later (at 1200: 4/20 vs 8/20).
- S11: after a gate reset the fresh gate landed in a key split 24 of 27 times. S17: on the header
  layout the hinge removed far_A_slow's position splits and all 10 runs fell into key splits
  (far_EMA and far_SEL did the same in batch 3). Reading: blocking one cheap split exposes the
  next; on the header layout the next is the key token.
- S18: the stalled runs answer ~0.53 (header) and ~0.33 (blocked) with no pattern by block or key
  position; 24000 perfect-gate updates leave them there. S15: slow memory prevents this.

THE ARMS. SLOW_HINGE (explore_slow_mem's param groups and lr switch, unchanged: gate W_in/W_h/W_g
at lr 1e-3 = test_channel_binding.SUB_LR throughout; every other parameter 1e-4 for updates 1-2400,
1e-3 after; MAX_ITERS 24000) plus, on each training batch,
    LAMBDA * [relu(eta2_index - TAU) + relu(eta2_half - TAU) + relu(eta2_key - TAU)],
LAMBDA 1.0, TAU 0.2, added through explore_aux_gate's gradient-injection node. eta2_index and
eta2_half are S13's / S17's (explore_far_slow_hinge.pos_penalty_at: the read gate's channel-0
probability at the key positions, grouped by the pair's index in the sequence and by its first or
second half); eta2_key groups the same values by the key token (between-group over total sum of
squares, total + 1e-12; it equals the routing statistics' eta_key_by_key).
  FAR_TOK   header layout (explore_far_cue.HEADER; S17's hinge positions and groups), arm A's
            parameters, no nudge; statistics explore_far_cue.header_stats. Seeds 160-169, paired
            with S17's FAR_SLOW_HINGE and batch 3's far_A_slow (same initial parameters, batches).
  NEAR_TOK  grouped layout (explore_common.TASK; S13's positions 3j+1 and groups); statistics
            explore_common2.stats_b2. Seeds 160-179, paired with S13's SLOW_HINGE (18/20 there).
Logged per run: the batches on which each hinge term (index, half, key) and any term was on.
On a perfectly routed gate eta2_key is exactly 0 on both layouts (every key appears once per
stream in every sequence), so the key term never fires on a routed gate.

RULES (fixed before any run), outcome DISCOVERED:
  FAR_TOK:  promising if >= 4/10; not if <= 1/10; otherwise inconclusive.
  NEAR_TOK: "keeps the near case" if >= 16/20; "costs the near case" if <= 13/20; otherwise neither.
Also printed: failure classes and hinge firings per term; McNemar against the paired arms.

CHECKS: with the key term's TAU at 1.0, FAR_TOK equals S17's FAR_SLOW_HINGE and NEAR_TOK equals
S13's SLOW_HINGE bit for bit (curves and weights, 3600 steps; also the recorded runs); eta2_key
equals the routing statistics' eta_key_by_key on a probe batch (at init and on a gate far from
uniform), on both layouts; on 20 probe batches of each layout the perfect gate scores below 0.1 on
eta2_key and a gate set by key token scores above 0.5; the key term's gradient reaches only the
gate and the embedding.
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
import explore_slow_hinge as s13
import explore_far_cue as s6
import explore_far_gates as s8
import explore_far_slow_hinge as s17
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH, probe_batch
from test_channel_binding import ARCH
from test_instrument_v2 import perfect_gate_general
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class, routing_stats
from test_binding_onset import EVAL_EVERY, BATCH

NAME = "token_hinge"
IDEA = "S13's hinge plus a third term on the key token's identity (eta2_key), near and far"
SOURCE = ("batch 6's S17 (blocking the position split on the header layout exposed key splits); S11 "
          "(fresh gates land in key splits)")
CHANGE = ("SLOW_MEM + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2) + relu(eta2_key - 0.2)] of "
          "the read gate at key positions, on each training batch")
PAIRING = ("FAR_TOK seeds 160-169 with S17's FAR_SLOW_HINGE and batch 3's far_A_slow; NEAR_TOK seeds "
           "160-179 with S13's SLOW_HINGE (same initial parameters and batches)")
LAMBDA, TAU = 1.0, 0.2
EPS = s17.EPS
HEADER = s6.HEADER
KPOS_FAR = s17.KPOS
KPOS_NEAR = 3 * torch.arange(ec.TASK.S * ec.TASK.P) + 1
SEEDS_FAR, SEEDS_NEAR = tuple(range(160, 170)), tuple(range(160, 180))
S17_STORE, B3_STORE, S13_STORE = "far_slow_hinge", "far_gates", "slow_hinge"
FAR_PROMISING, FAR_NOT = 4, 1
NEAR_KEEPS, NEAR_COSTS = 16, 13
PROBE_BATCHES = 20
TERMS = ("index", "half", "key")

_sched = (s5.ARMS["SLOW_MEM"]["sched"] + f"; + {LAMBDA} * [relu(eta2_index - {TAU}) + relu(eta2_half - "
          f"{TAU}) + relu(eta2_key - {TAU})] at the key positions on each training batch")
ARMS = {
    "FAR_TOK": dict(s8.ARMS["far_A_slow"], key="FAR_TOK", seeds=SEEDS_FAR, prio=1, layout="header",
                    label="far_A_slow + hinge on position and key-token eta^2 (TAU 0.2), header layout",
                    sched=_sched + " (header KEY positions)"),
    "NEAR_TOK": dict(s5.ARMS["SLOW_MEM"], key="NEAR_TOK", seeds=SEEDS_NEAR, prio=1, layout="grouped",
                     label="SLOW_MEM + hinge on position and key-token eta^2 (TAU 0.2), grouped layout",
                     sched=_sched),
}
TASK_OF = {"header": HEADER, "grouped": ec.TASK}
KPOS_OF = {"header": KPOS_FAR, "grouped": KPOS_NEAR}


def key_penalty(gr, x, kpos, S, P):
    """eta^2 of gr[..., 0] at kpos grouped by the key token there (x = the input tokens)."""
    p = gr[:, kpos, 0]                                                  # (B, n)
    key = x[:, kpos] - S                                                # (B, n) in 0..P-1
    oh = F.one_hot(key, P).to(p.dtype)                                  # (B, n, P)
    mu = p.mean()
    tot = ((p - mu) ** 2).sum() + EPS
    cnt = oh.sum((0, 1))
    mean_k = (p.unsqueeze(-1) * oh).sum((0, 1)) / cnt.clamp(min=1)
    between = (cnt * (mean_k - mu) ** 2).sum()
    return between / tot


class TokHingeBDH(MultiBDH):
    def __init__(self, *args, pen_lambda=None, tau=TAU, tau_key=TAU, kpos=None, pen_SP=(2, 4), **kw):
        super().__init__(*args, **kw)
        self.pen_lambda, self.tau, self.tau_key, self.kpos, self.pen_SP = \
            pen_lambda, tau, tau_key, kpos, pen_SP
        self.pen_last, self.n_batches, self.n_any = None, 0, 0
        self.n_term = [0, 0, 0]

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.pen_lambda is not None and torch.is_grad_enabled():
            S, P = self.pen_SP
            e_i, e_h = s17.pos_penalty_at(gr, self.kpos, P)
            e_k = key_penalty(gr, tokens, self.kpos, S, P)
            h_i, h_h, h_k = F.relu(e_i - self.tau), F.relu(e_h - self.tau), F.relu(e_k - self.tau_key)
            on = [bool(h_i > 0), bool(h_h > 0), bool(h_k > 0)]
            self.n_batches += 1
            self.n_any += int(any(on))
            self.n_term = [a + int(b) for a, b in zip(self.n_term, on)]
            self.pen_last = (float(e_i.detach()), float(e_h.detach()), float(e_k.detach()))
            logits = _Inject.apply(logits, self.pen_lambda * (h_i + h_h + h_k))
        return logits, sat, gr, gw


def make_model(a, seed, tau_key=TAU, lam=LAMBDA):
    task = TASK_OF[a["layout"]]
    torch.manual_seed(seed)
    return TokHingeBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                       gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                       ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), pen_lambda=lam, tau=TAU,
                       tau_key=tau_key, kpos=KPOS_OF[a["layout"]], pen_SP=(task.S, task.P))


def stats_far(model, task, probe, step):
    out = s6.header_stats(model, task, probe, step)
    out.update(pen_last_train=model.pen_last, hinge_counts=[model.n_any, model.n_batches],
               hinge_terms=list(model.n_term))
    return out


def stats_near(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out.update(pen_last_train=model.pen_last, hinge_counts=[model.n_any, model.n_batches],
               hinge_terms=list(model.n_term))
    return out


STATS_OF = {"header": stats_far, "grouped": stats_near}


def run_path(a, seed, iters, tau_key=TAU, keep=None):
    holder = {}
    lay = a["layout"]
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=TASK_OF[lay],
                      stats_fn=STATS_OF[lay], grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s, tau_key)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], TASK_OF[a["layout"]], lambda: make_model(a, 0), a["lr"], STATS_OF[a["layout"]])]


def counts(r):
    e = r["end"] if r else {}
    return e.get("hinge_terms", [0, 0, 0]), e.get("hinge_counts", [0, 0])


def check():
    ok = True
    it = 3 * EVAL_EVERY
    rows = []
    # (1) the key term at TAU 1.0: FAR_TOK = S17's FAR_SLOW_HINGE, NEAR_TOK = S13's SLOW_HINGE
    for arm, other, store, key, seed in (
            ("FAR_TOK", lambda s, k: s17.run_path(s17.ARMS["FAR_SLOW_HINGE"], s, it, keep=k), S17_STORE,
             "FAR_SLOW_HINGE", 160),
            ("NEAR_TOK", lambda s, k: s13.run_path(s13.ARMS["SLOW_HINGE"], s, it, keep=k), S13_STORE,
             "SLOW_HINGE", 160)):
        k1, k2 = {"at": it}, {"at": it}
        r1 = run_path(ARMS[arm], seed, it, tau_key=1.0, keep=k1)
        r2 = other(seed, k2)
        rec = [c for c in ec.load_store(store)["runs"][f"{key}|{seed}"]["curve"] if c[0] <= it]
        same = (r1["curve"] == r2["curve"] == rec
                and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"]))
        rows.append((f"{arm} with the key term's TAU at 1.0 equals {key} bit for bit through {it} steps "
                     f"on seed {seed} (curves = a fresh {key} run = the recorded run; weights equal; key "
                     f"term fired {r1['end']['hinge_terms'][2]} times)", same and r1["end"]["hinge_terms"][2] == 0))
    # (2) eta2_key = the routing statistics' eta_key_by_key; (3) 20 probe batches per layout
    diffs, pf, kg = [], {}, {}
    for lay, task in TASK_OF.items():
        a = ARMS["FAR_TOK" if lay == "header" else "NEAR_TOK"]
        kpos = KPOS_OF[lay]
        probe = probe_batch(task, 160)
        for scale in (1.0, 30.0):
            mm = make_model(a, 160)
            with torch.no_grad():
                mm.W_in.mul_(scale)
            mm.eval()
            with torch.no_grad():
                g = mm(probe[0], None)[2]
                ek = key_penalty(g, probe[0], kpos, task.S, task.P).item()
            st = (s6.header_routing_stats(mm, task, probe) if lay == "header"
                  else routing_stats(mm, task, probe))
            diffs.append(abs(ek - st["eta_key_by_key"]))
        pf[lay], kg[lay] = [], []
        for s in range(PROBE_BATCHES):
            x = probe_batch(task, s)[0]
            pg = perfect_gate_general(x, task.ctx_tokens)
            pf[lay].append(key_penalty(pg, x, kpos, task.S, task.P).item())
            kgate = torch.zeros_like(pg)
            first = (x - task.S) < task.P // 2                          # key tokens 0..P/2-1
            kgate[..., 0] = first.float()
            kgate[..., 1] = 1.0 - first.float()
            kg[lay].append(key_penalty(kgate, x, kpos, task.S, task.P).item())
    # (4) the key term's gradient reaches only the gate and the embedding
    a = ARMS["NEAR_TOK"]
    m = make_model(a, 160)
    m.train()
    x = probe_batch(ec.TASK, 160)[0][:BATCH]
    gr = MultiBDH.forward(m, x)[2]
    key_penalty(gr, x, KPOS_NEAR, ec.TASK.S, ec.TASK.P).backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    rows += [(f"eta2_key equals the routing statistics' eta_key_by_key on a probe batch, both layouts, at "
              f"init and on a gate far from uniform (max diff {max(diffs):.1e} < 1e-5)", max(diffs) < 1e-5),
             (f"on {PROBE_BATCHES} probe batches per layout the perfect gate scores below 0.1 on eta2_key "
              f"(max: header {max(pf['header']):.2e}, grouped {max(pf['grouped']):.2e}) and a gate set by "
              f"key token above 0.5 (min: header {min(kg['header']):.4f}, grouped {min(kg['grouped']):.4f})",
              max(pf["header"] + pf["grouped"]) < 0.1 and min(kg["header"] + kg["grouped"]) > 0.5),
             (f"the key term's gradient reaches only {sorted(got)}", got == {"W_in", "W_h", "W_g", "embed.weight"})]
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def far_rule(n):
    return "promising" if n >= FAR_PROMISING else ("not" if n <= FAR_NOT else "inconclusive")


def near_rule(n):
    if n >= NEAR_KEEPS:
        return "keeps the near case"
    if n <= NEAR_COSTS:
        return "costs the near case"
    return "neither"


def stall(r):
    return bool(r and r.get("ok") and not ec.bound(r)
                and (c2.routed_at(r, "end") or fail_class(r) == "STREAM-PARTIAL"))


def fails(get, seeds):
    f = {}
    for s in seeds:
        r = get(s)
        if r and r.get("ok") and not ec.bound(r):
            f[fail_class(r)] = f.get(fail_class(r), 0) + 1
    return f


def pair(runs, other, fn, what, label, arm):
    b = sum(1 for s, r in runs.items() if fn(r) and not fn(other.get(s)))
    c = sum(1 for s, r in runs.items() if fn(other.get(s)) and not fn(r))
    nn = sum(bool(fn(r)) for r in runs.values())
    no = sum(bool(fn(other.get(s))) for s in runs)
    print(f"    {what:<13} {arm} {nn:>2}/{len(runs)}   {label} {no:>2}/{len(runs)}   {arm} only {b}, "
          f"{label} only {c}, McNemar two-sided p = {mcnemar_exact(b, c):.3g}")
    return dict(n=len(runs), new=nn, old=no, b=b, c=c, p=mcnemar_exact(b, c))


def table(arm, runs_all, refs):
    print(f"   arm {arm} (lr {ARMS[arm]['lr']:g}; {c2.sched(ARMS[arm])})")
    print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} " + " ".join(f"{lab:<15}" for lab, _ in refs)
          + f" {'hinge on: index/half/key/any of batches':>40}")
    for s in ARMS[arm]["seeds"]:
        r = runs_all.get(s)
        tags = " ".join(f"{ec.tag(d.get(s)):<15}" for _, d in refs)
        if r is None or not r.get("ok"):
            print(f"    {s:>4} {ec.tag(r):<70} {tags}")
            continue
        m = r["end"].get("margin")
        t, hc = counts(r)
        print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
              f"{ec.tag(r):<15} {tags} {t[0]:>8}/{t[1]}/{t[2]}/{hc[0]} of {hc[1]}")


def firings(runs):
    tot = [0, 0, 0]
    any_, nb = 0, 0
    per = {}
    for s, r in runs.items():
        t, hc = counts(r)
        tot = [a + b for a, b in zip(tot, t)]
        any_ += hc[0]
        nb += hc[1]
        per[s] = t
    runs_on = [sum(1 for t in per.values() if t[i] > 0) for i in range(3)]
    print(f"    hinge firings (batches): index {tot[0]}, half {tot[1]}, key {tot[2]}, any {any_} of {nb}; runs "
          f"with the term on at least once: index {runs_on[0]}, half {runs_on[1]}, key {runs_on[2]} of {len(runs)}")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items()
                 if k.startswith(arm + "|") and r.get("ok")} for arm in ARMS}
    s17r = {int(k.split("|")[1]): r for k, r in ec.load_store(S17_STORE)["runs"].items()
            if k.startswith("FAR_SLOW_HINGE|")}
    b3 = {int(k.split("|")[1]): r for k, r in ec.load_store(B3_STORE)["runs"].items()
          if k.startswith("far_A_slow|")}
    s13r = s13.runs_of((S13_STORE,), "SLOW_HINGE")
    xa = ec.recorded("A")
    print("  per seed (ROUTED* at 1200/2400/3600/end; hinge on = training batches with that term above TAU):")
    table("FAR_TOK", got["FAR_TOK"], (("S17 FAR_SLOW_H", s17r), ("b3 far_A_slow", b3)))
    table("NEAR_TOK", got["NEAR_TOK"], (("S13 SLOW_HINGE", s13r), ("X A", xa)))
    out = {}
    far = got["FAR_TOK"]
    print(f"  FAR_TOK (header layout), {len(far)} seeds:")
    print("   paired with S17's FAR_SLOW_HINGE:")
    d = pair(far, s17r, ec.discovered, "DISCOVERED", "FAR_SLOW_HINGE", "FAR_TOK")
    pair(far, s17r, ec.bound, "BOUND", "FAR_SLOW_HINGE", "FAR_TOK")
    pair(far, s17r, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "FAR_SLOW_HINGE", "FAR_TOK")
    pair(far, s17r, lambda r: c2.routed_at(r, "end"), "ROUTED*@end", "FAR_SLOW_HINGE", "FAR_TOK")
    print("   paired with batch 3's far_A_slow:")
    pair(far, b3, ec.discovered, "DISCOVERED", "far_A_slow", "FAR_TOK")
    pair(far, b3, stall, "STALLS", "far_A_slow", "FAR_TOK")
    print(f"    failures FAR_TOK {fails(far.get, far)}   FAR_SLOW_HINGE {fails(s17r.get, far)}   far_A_slow "
          f"{fails(b3.get, far)}")
    print(f"    stalls (routed at the end or stream-partial, not bound): FAR_TOK "
          f"{[s for s, r in far.items() if stall(r)]}")
    firings(far)
    out["far"] = dict(d, rule=far_rule(d["new"]))
    near = got["NEAR_TOK"]
    print(f"  NEAR_TOK (grouped layout), {len(near)} seeds:")
    print("   paired with S13's SLOW_HINGE:")
    d = pair(near, s13r, ec.discovered, "DISCOVERED", "SLOW_HINGE", "NEAR_TOK")
    pair(near, s13r, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "SLOW_HINGE", "NEAR_TOK")
    pair(near, s13r, lambda r: c2.routed_at(r, "end"), "ROUTED*@end", "SLOW_HINGE", "NEAR_TOK")
    print("   paired with X's arm A:")
    pair(near, xa, ec.discovered, "DISCOVERED", "X A", "NEAR_TOK")
    print(f"    failures NEAR_TOK {fails(near.get, near)}   SLOW_HINGE {fails(s13r.get, near)}   X arm A "
          f"{fails(xa.get, near)}")
    firings(near)
    out["near"] = dict(d, rule=near_rule(d["new"]))
    print(f"  RULE FAR_TOK (DISCOVERED {out['far']['new']}/{out['far']['n']}; promising >= {FAR_PROMISING}/10, "
          f"not <= {FAR_NOT}/10): {out['far']['rule']}")
    print(f"  RULE NEAR_TOK (DISCOVERED {out['near']['new']}/{out['near']['n']}; 'keeps the near case' >= "
          f"{NEAR_KEEPS}/20, 'costs the near case' <= {NEAR_COSTS}/20): {out['near']['rule']}")
    return out
