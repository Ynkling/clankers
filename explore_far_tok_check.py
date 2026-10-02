#!/usr/bin/env python
"""
explore_far_tok_check.py — EXPLORATORY, not a result. Screen S22 (batch 8): does S19's FAR_TOK
replicate on fresh seeds, and does starting the key-token term late help?

BACKGROUND (post hoc, from batch 7)
- S19 FAR_TOK (header layout, 160-169): DISCOVERED 6/10, BOUND 9/10 (vs FAR_SLOW_HINGE 1/10, 8 vs
  0). ROUTED* at the end only 3/10 (163, 164, 168); 166 and 167 separate streams at values only;
  160 partial; 161, 162, 165 bound with every eta^2 near 0 and VAL cos 0.74-1.00. No single-channel
  run exists on the header layout (S23).
- NEAR_TOK: the key term fired 1-9 times in all 20 runs (incl. the 10 where SLOW_HINGE's hinge
  never fired); ROUTED*@1200 0/20 vs 15/20. eta2 is scale-free and at init the read gate's tiny
  deviations are a function of the current token (eta_key_by_key 0.3-0.8 at step 0 in main-line
  k=16 runs), so the key term likely fires at the start. Key splits appeared from 2400 in S11, once
  the memory learns the keys.

THE ARMS (S19's recipe: explore_slow_mem's param groups and lr switch, gate W_in/W_h/W_g at lr 1e-3 =
test_channel_binding.SUB_LR throughout, every other parameter 1e-4 for updates 1-2400 and 1e-3 after,
MAX_ITERS 24000; on each training batch LAMBDA * [relu(eta2_index - TAU) + relu(eta2_half - TAU) +
relu(eta2_key - TAU)], LAMBDA 1.0, TAU 0.2, through explore_aux_gate's gradient-injection node; the
terms are S19's own functions: explore_far_slow_hinge.pos_penalty_at and explore_token_hinge.
key_penalty, at S19's key positions and groups for each layout)
  FAR_TOK_NEW    S19's FAR_TOK on fresh seeds 170-179 (header layout).
  FAR_TOK_LATE   FAR_TOK with the key term active only from update 2401 (the position terms from
                 update 1 as before): through update 2400 it is S17's FAR_SLOW_HINGE. Seeds 160-179,
                 paired with FAR_TOK (160-169 S19's records, 170-179 FAR_TOK_NEW).
  NEAR_TOK_LATE  the same change on the grouped layout (through update 2400 it is S13's SLOW_HINGE).
                 Seeds 160-179, paired with S19's NEAR_TOK and S13's SLOW_HINGE.
Logged per run (all arms): the update of every key-term firing (the training batch n, n = 1, 2, ...,
whose gradient enters update n; firings are the applied ones; FAR_TOK_LATE and NEAR_TOK_LATE also
log the updates <= 2400 on which the key term was above TAU but not applied), and eta2_key, eta2_index
and eta2_half after 0, 1, 10 and 100 updates (on the training batch of the next update, i.e. batch
n = 1, 2, 11, 101). The hinge counts per term are S19's.

RULES (fixed before any run)
- FAR_TOK_NEW (DISCOVERED and ROUTED* at the end both printed): "replicates" if DISCOVERED >= 4/10
  and ROUTED*@end >= 2/10; "does not" if DISCOVERED <= 1/10; otherwise neither.
- FAR_TOK_LATE vs FAR_TOK (20 seeds), ROUTED*@end: "the late start helps" if b - c >= 4 and McNemar
  p < 0.1; "it costs" if c - b >= 4; otherwise neither (DISCOVERED and BOUND printed alongside).
- NEAR_TOK_LATE: "routing returns" if ROUTED*@1200 >= 10/20 and DISCOVERED >= 16/20 (otherwise the
  rule is not met).

CHECKS: FAR_TOK_NEW's code path equals S19's (a seed-160 run reproduces S19's FAR_TOK record to 3600);
FAR_TOK_LATE equals FAR_SLOW_HINGE bit for bit through update 2400 on seed 160 (curves and weights;
also S17's record), and NEAR_TOK_LATE equals SLOW_HINGE the same way; the logs: the number of logged
key firings equals the key term's hinge count, the early eta2 values are logged at updates 0, 1, 10,
100, and no key firing is applied at or before update 2400 in the late arms.
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
import explore_far_slow_hinge as s17
import explore_token_hinge as s19
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY

NAME = "far_tok_check"
IDEA = "S19's FAR_TOK on fresh seeds, and the key-token term started late (from update 2401)"
SOURCE = ("batch 7's S19 (FAR_TOK 6/10 DISCOVERED, 3/10 routed; NEAR_TOK's key term fired at the start "
          "and delayed routing)")
CHANGE = ("FAR_TOK_NEW: S19's FAR_TOK on seeds 170-179; FAR_TOK_LATE / NEAR_TOK_LATE: the key term only "
          "from update 2401, the position terms from update 1")
PAIRING = ("FAR_TOK_LATE seeds 160-179 with FAR_TOK (S19 160-169, FAR_TOK_NEW 170-179); NEAR_TOK_LATE seeds "
           "160-179 with S19's NEAR_TOK and S13's SLOW_HINGE (same initial parameters and batches)")
LAMBDA, TAU = s19.LAMBDA, s19.TAU
LATE_FROM = s5.WARM_UPDATES + 1                        # 2401
EARLY_AT = (0, 1, 10, 100)
LOG_CAP = 5000
SEEDS_NEW, SEEDS_20 = tuple(range(170, 180)), tuple(range(160, 180))
S19_STORE, S17_STORE, S13_STORE = s19.NAME, s17.NAME, s13.NAME

_far, _near = s19.ARMS["FAR_TOK"], s19.ARMS["NEAR_TOK"]
_late = f"; the key term only from update {LATE_FROM}"
ARMS = {
    "FAR_TOK_NEW": dict(_far, key="FAR_TOK_NEW", seeds=SEEDS_NEW, key_from=1,
                        label="S19's FAR_TOK on fresh seeds, header layout"),
    "FAR_TOK_LATE": dict(_far, key="FAR_TOK_LATE", seeds=SEEDS_20, key_from=LATE_FROM,
                         label="FAR_TOK with the key term from update 2401, header layout",
                         sched=_far["sched"] + _late),
    "NEAR_TOK_LATE": dict(_near, key="NEAR_TOK_LATE", seeds=SEEDS_20, key_from=LATE_FROM,
                          label="NEAR_TOK with the key term from update 2401, grouped layout",
                          sched=_near["sched"] + _late),
}


class TokLogBDH(MultiBDH):
    """S19's TokHingeBDH (the same terms, in the same order), with the key term applied from batch
    key_from on, and the firing / early-eta2 logs."""

    def __init__(self, *args, pen_lambda=None, tau=TAU, key_from=1, kpos=None, pen_SP=(2, 4), **kw):
        super().__init__(*args, **kw)
        self.pen_lambda, self.tau, self.key_from, self.kpos, self.pen_SP = \
            pen_lambda, tau, key_from, kpos, pen_SP
        self.pen_last, self.n_batches, self.n_any = None, 0, 0
        self.n_term = [0, 0, 0]
        self.key_fire, self.key_would, self.early = [], [], {}

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.pen_lambda is not None and torch.is_grad_enabled():
            S, P = self.pen_SP
            self.n_batches += 1
            n = self.n_batches
            e_i, e_h = s17.pos_penalty_at(gr, self.kpos, P)
            e_k = s19.key_penalty(gr, tokens, self.kpos, S, P)
            h_i, h_h = F.relu(e_i - self.tau), F.relu(e_h - self.tau)
            if n - 1 in EARLY_AT:
                self.early[str(n - 1)] = [float(e_k.detach()), float(e_i.detach()), float(e_h.detach())]
            key_on = n >= self.key_from
            if key_on:
                h_k = F.relu(e_k - self.tau)
                on = [bool(h_i > 0), bool(h_h > 0), bool(h_k > 0)]
                if on[2] and len(self.key_fire) < LOG_CAP:
                    self.key_fire.append(n)
                total = h_i + h_h + h_k
            else:
                on = [bool(h_i > 0), bool(h_h > 0), False]
                if float(e_k.detach()) > self.tau and len(self.key_would) < LOG_CAP:
                    self.key_would.append(n)
                total = h_i + h_h
            self.n_any += int(any(on))
            self.n_term = [a + int(b) for a, b in zip(self.n_term, on)]
            self.pen_last = (float(e_i.detach()), float(e_h.detach()), float(e_k.detach()))
            logits = _Inject.apply(logits, self.pen_lambda * total)
        return logits, sat, gr, gw


def make_model(a, seed, key_from=None):
    task = s19.TASK_OF[a["layout"]]
    torch.manual_seed(seed)
    return TokLogBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                     gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                     ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), pen_lambda=LAMBDA, tau=TAU,
                     key_from=a["key_from"] if key_from is None else key_from,
                     kpos=s19.KPOS_OF[a["layout"]], pen_SP=(task.S, task.P))


def stats_fn(model, task, probe, step):
    out = s19.STATS_OF["header" if task is s19.HEADER else "grouped"](model, task, probe, step)
    if step == "end":
        out.update(key_fire=list(model.key_fire), key_would=list(model.key_would),
                   early_eta2=dict(model.early))
    return out


def run_path(a, seed, iters, keep=None):
    holder = {}
    task = s19.TASK_OF[a["layout"]]
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=task,
                      stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], s19.TASK_OF[a["layout"]], lambda: make_model(a, 0), a["lr"], stats_fn)]


def check():
    ok = True
    rows = []
    it = 3 * EVAL_EVERY
    # (1) FAR_TOK_NEW's path = S19's: seed 160 to 3600 reproduces S19's FAR_TOK record
    r_new = run_path(ARMS["FAR_TOK_NEW"], 160, it)
    rec19 = [c for c in ec.load_store(S19_STORE)["runs"]["FAR_TOK|160"]["curve"] if c[0] <= it]
    e = r_new["end"]
    logs_ok = (len(e["key_fire"]) == e["hinge_terms"][2] and sorted(e["early_eta2"]) == sorted(map(str, EARLY_AT))
               and not e["key_would"])
    rows.append((f"FAR_TOK_NEW's code path equals S19's: seed 160 to {it} reproduces S19's FAR_TOK record "
                 f"({r_new['curve'] == rec19}); its logs: {len(e['key_fire'])} key firings = the key hinge count "
                 f"{e['hinge_terms'][2]}, early eta2_key at updates 0/1/10/100 "
                 + "/".join(f"{e['early_eta2'][str(u)][0]:.3f}" for u in EARLY_AT if str(u) in e["early_eta2"]),
                 r_new["curve"] == rec19 and logs_ok))
    # (2) the late arms equal FAR_SLOW_HINGE / SLOW_HINGE through update 2400
    t = s5.WARM_UPDATES
    for arm, other, store, key in (
            ("FAR_TOK_LATE", lambda s, k: s17.run_path(s17.ARMS["FAR_SLOW_HINGE"], s, t, keep=k), S17_STORE,
             "FAR_SLOW_HINGE"),
            ("NEAR_TOK_LATE", lambda s, k: s13.run_path(s13.ARMS["SLOW_HINGE"], s, t, keep=k), S13_STORE,
             "SLOW_HINGE")):
        k1, k2 = {"at": t}, {"at": t}
        r1 = run_path(ARMS[arm], 160, t, keep=k1)
        r2 = other(160, k2)
        rec = [c for c in ec.load_store(store)["runs"][f"{key}|160"]["curve"] if c[0] <= t]
        same = (r1["curve"] == r2["curve"] == rec
                and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"]))
        e1 = r1["end"]
        late_ok = not e1["key_fire"] and e1["hinge_terms"][2] == 0
        rows.append((f"{arm} equals {key} bit for bit through update {t} on seed 160 (curves = a fresh {key} run "
                     f"= the recorded run; weights after update {t} equal); no key firing applied (the key term was "
                     f"above TAU on {len(e1['key_would'])} batches, first {e1['key_would'][:5]})", same and late_ok))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def new_rule(nd, nr):
    if nd >= 4 and nr >= 2:
        return "replicates"
    if nd <= 1:
        return "does not"
    return "neither"


def late_rule(d):
    if d["b"] - d["c"] >= 4 and d["p"] < 0.1:
        return "the late start helps"
    if d["c"] - d["b"] >= 4:
        return "it costs"
    return "neither"


def near_rule(r12, nd):
    return "routing returns" if r12 >= 10 and nd >= 16 else "rule not met"


def runs_of(store_name, key):
    return {int(k.split("|")[1]): r for k, r in ec.load_store(store_name)["runs"].items()
            if k.startswith(key + "|") and r.get("ok")}


def logs_str(r):
    e = r["end"]
    kf, kw = e.get("key_fire", []), e.get("key_would", [])
    early = e.get("early_eta2", {})
    eb = "/".join(f"{early[str(u)][0]:.2f}" if str(u) in early else "--" for u in EARLY_AT)
    return (f"eta2_key@0/1/10/100 {eb}; key fired {len(kf)}x at {kf[:6]}{'...' if len(kf) > 6 else ''}"
            + (f"; above TAU while off {len(kw)}x (first {kw[:3]}, last {kw[-1:]})" if kw else ""))


def table(arm, runs, refs):
    print(f"   arm {arm} (lr {ARMS[arm]['lr']:g}; {c2.sched(ARMS[arm])})")
    print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} " + " ".join(f"{lab:<15}" for lab, _ in refs) + "  logs")
    for s in ARMS[arm]["seeds"]:
        r = runs.get(s)
        tags = " ".join(f"{ec.tag(d.get(s)):<15}" for _, d in refs)
        if r is None:
            print(f"    {s:>4} {'not run':<70} {tags}")
            continue
        m = r["end"].get("margin")
        print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  {ec.tag(r):<15} "
              f"{tags}  {logs_str(r)}")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items()
                 if k.startswith(arm + "|") and r.get("ok")} for arm in ARMS}
    far19 = runs_of(S19_STORE, "FAR_TOK")
    near19 = runs_of(S19_STORE, "NEAR_TOK")
    s13r = runs_of(S13_STORE, "SLOW_HINGE")
    s17r = runs_of(S17_STORE, "FAR_SLOW_HINGE")
    far_all = dict(far19)
    far_all.update(got["FAR_TOK_NEW"])                         # FAR_TOK: S19 160-169, NEW 170-179
    print("  per seed (ROUTED* at 1200/2400/3600/end; logs: eta2_key after 0/1/10/100 updates, the updates of the "
          "applied key firings, and in the late arms the updates where the key term was above TAU but off):")
    table("FAR_TOK_NEW", got["FAR_TOK_NEW"], ())
    table("FAR_TOK_LATE", got["FAR_TOK_LATE"], (("FAR_TOK", far_all), ("S17 FAR_SLOW_H", s17r)))
    table("NEAR_TOK_LATE", got["NEAR_TOK_LATE"], (("S19 NEAR_TOK", near19), ("S13 SLOW_HINGE", s13r)))
    out = {}
    new = got["FAR_TOK_NEW"]
    nd = sum(ec.discovered(r) for r in new.values())
    nr = sum(c2.routed_at(r, "end") for r in new.values())
    nb = sum(ec.bound(r) for r in new.values())
    print(f"  FAR_TOK_NEW (seeds 170-179): DISCOVERED {nd}/{len(new)}, ROUTED*@end {nr}/{len(new)}, BOUND {nb}/{len(new)}; "
          f"failures {s19.fails(new.get, new)}; with S19's 160-169: DISCOVERED {nd + sum(ec.discovered(r) for r in far19.values())}"
          f"/{len(new) + len(far19)}, ROUTED*@end {nr + sum(c2.routed_at(r, 'end') for r in far19.values())}/"
          f"{len(new) + len(far19)}")
    out["new"] = dict(n=len(new), discovered=nd, routed=nr, rule=new_rule(nd, nr))
    late = got["FAR_TOK_LATE"]
    print(f"  FAR_TOK_LATE vs FAR_TOK ({len(late)} seeds; FAR_TOK = S19 on 160-169, FAR_TOK_NEW on 170-179):")
    d = s19.pair(late, far_all, lambda r: c2.routed_at(r, "end"), "ROUTED*@end", "FAR_TOK", "FAR_TOK_LATE")
    s19.pair(late, far_all, ec.discovered, "DISCOVERED", "FAR_TOK", "FAR_TOK_LATE")
    s19.pair(late, far_all, ec.bound, "BOUND", "FAR_TOK", "FAR_TOK_LATE")
    s19.pair(late, far_all, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "FAR_TOK", "FAR_TOK_LATE")
    print(f"    failures FAR_TOK_LATE {s19.fails(late.get, late)}   FAR_TOK {s19.fails(far_all.get, late)}")
    s19.firings(late)
    out["late"] = dict(d, rule=late_rule(d))
    nl = got["NEAR_TOK_LATE"]
    print(f"  NEAR_TOK_LATE ({len(nl)} seeds):")
    print("   paired with S19's NEAR_TOK:")
    s19.pair(nl, near19, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "NEAR_TOK", "NEAR_TOK_LATE")
    dn = s19.pair(nl, near19, ec.discovered, "DISCOVERED", "NEAR_TOK", "NEAR_TOK_LATE")
    print("   paired with S13's SLOW_HINGE:")
    s19.pair(nl, s13r, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", "SLOW_HINGE", "NEAR_TOK_LATE")
    s19.pair(nl, s13r, ec.discovered, "DISCOVERED", "SLOW_HINGE", "NEAR_TOK_LATE")
    print(f"    failures NEAR_TOK_LATE {s19.fails(nl.get, nl)}   NEAR_TOK {s19.fails(near19.get, nl)}   SLOW_HINGE "
          f"{s19.fails(s13r.get, nl)}")
    s19.firings(nl)
    r12 = sum(c2.routed_at(r, EVAL_EVERY) for r in nl.values())
    out["near"] = dict(n=len(nl), routed1200=r12, discovered=dn["new"], rule=near_rule(r12, dn["new"]))
    allr = list(new.values()) + list(late.values()) + list(nl.values())
    e0 = [r["end"]["early_eta2"]["0"][0] for r in allr if "0" in r["end"].get("early_eta2", {})]
    early_fire = sum(1 for r in allr if any(u <= EVAL_EVERY for u in r["end"].get("key_fire", [])
                                            + r["end"].get("key_would", [])))
    if e0:
        print(f"  eta2_key on the first training batch (0 updates), all {len(e0)} runs: min {min(e0):.3f}, median "
              f"{sorted(e0)[len(e0) // 2]:.3f}, max {max(e0):.3f}; runs with the key term above TAU in updates "
              f"1-{EVAL_EVERY} (applied or not): {early_fire}/{len(allr)}")
    print(f"  RULE FAR_TOK_NEW (DISCOVERED {nd}/{len(new)}, ROUTED*@end {nr}/{len(new)}; 'replicates' if DISCOVERED >= 4 "
          f"and ROUTED*@end >= 2, 'does not' if DISCOVERED <= 1): {out['new']['rule']}")
    print(f"  RULE FAR_TOK_LATE vs FAR_TOK (ROUTED*@end {d['b']} vs {d['c']}, p = {d['p']:.3g}; 'the late start helps' "
          f"if b - c >= 4 and p < 0.1, 'it costs' if c - b >= 4): {out['late']['rule']}")
    print(f"  RULE NEAR_TOK_LATE (ROUTED*@1200 {r12}/{len(nl)}, DISCOVERED {dn['new']}/{len(nl)}; 'routing returns' if "
          f">= 10/20 and >= 16/20): {out['near']['rule']}")
    return out
