#!/usr/bin/env python
"""
explore_g_s87_child.py — EXPLORATORY, not a result. Run code of Session G's screen S87 (explore_g_s87): can a term that
reads no task labels do the key term's work? Runs INSIDE a child process on this branch's own modules, 1 torch thread.
It imports S82's child (explore_g_devices_child) and S71-S73's (explore_g_rhl_child) unchanged.

TASK, MEMORY, RECIPE: RandHeaderTask-lite exactly as F's S70; Hebbian (delta: beta 1, L2 keys, tied write, decay 0.95);
k = 2, no conv, 24000 updates; SLOW (S56's groups). The model of every new arm is S72's REG3 (the register gate shared by
the three layers); each new device acts on updates 1-2400 only (the update count = training forwards).

THE NEW DEVICES (each with a switch that makes the model REG3 bit for bit when off; CHECKed through 2400)
  FREQ   REG3_FREQHINGE: S52's key term with the key positions and identities replaced by a corpus statistic:
             LAM x mean_l relu(eta2_freq(g^(l)) - TAU),  LAM 1.0, TAU 0.2 (S52's constants),
         eta2_freq = explore_b18_gates.eta2_pool of layer l's gate at every input position whose token is one of the
         FREQUENT types, grouped by the token's id. The frequent types come from freq_types(tokens): the token-type
         frequencies over a sample of 4096 training sequences (make_batch with torch.Generator(87_000); the input tokens
         tokens[:, :-1], every position; the function receives the token-id tensor only, CHECKed), sorted, cut at the
         LARGEST RATIO between consecutive frequencies; the types above the cut are frequent.
         On this task (computed before the spec was committed): CTX 0 0.1137, CTX 1 0.1136, KEY 3 0.1027, KEY 4 0.1026,
         KEY 5 0.1022, KEY 2 0.1017, then VAL 18 0.0236 (the largest ratio, 4.3x) and the other 15 values 0.022-0.024.
         The frequent set is {both CTX types, all four key types}: 6 types, 63.6% of the token mass.
         NOT AS THE USER PRESUMED: on this task the context tokens are the MOST frequent types (5 of the 22 input
         tokens per sequence over 2 types), the keys next (9 per sequence over 4 types), the values rare (8 over 16).
         The suggested rule "the smallest K covering half the token mass" gives K = 5 here: both CTX types and three of the
         four key types, the fourth dropped by sampling noise (0.1017 vs 0.1022). The largest-ratio cut is used instead
         (fixed now): it keeps the four equally frequent key types together and does not depend on sampling noise.
         Coincidence with the key term's positions, per sequence (every sequence alike): the frequent positions are the 8
         body KEY positions (all of the key term's positions), the query's KEY position (1) and the 5 CTX positions (4
         block headers + the query's); 8 of the 14 frequent positions are the key term's, and all 8 of the key term's
         positions are frequent. At the 5 CTX positions the grouping by token id is a grouping by STREAM, so the term also
         penalises the gate's dependence on the stream token at the stream token itself. READS: token ids and a frequency
         table of token ids; no label, no role.
  SELFKL REG3_SELFKL: LAM_S x mean_l mean_{sequences, body positions t} KL(g^(l)_t || gbar^(l)), LAM_S 1.0, gbar^(l) the
         sequence's own mean gate over its body positions (gradient flows through both); penalises any within-sequence
         variation of the gate. READS: nothing (body_len is a fixed index range).
  TEMP   REG3_TEMP: the gate's softmax at temperature T = 4 for updates 1-2400 (training and evaluation forwards alike,
         by the training-forward count) and T = 1 after: g = softmax(W_g2 tanh(u) / T). A schedule, not a penalty; at T = 1
         the division is skipped (bit for bit REG3). READS: nothing.
CONTROLS (rerun on these seeds through S82's own run path, explore_g_devices_child.run): BASELINE (LOCAL3 + SLOW),
REG3_KEYHINGE (the key term), ORACLE_ALL (the perfect gate, validity); for phase 2 also BASELINE_D, REG3_D, REG3_KEYHINGE_D.
COMMITMENT (for the "clamps" reading): test_router_reliability.gate_stats_k's "spread" at the evaluation at 2400 — the mean
over the probe's positions of the returned gate's total-variation distance from uniform (scaled to [0, 1]); the returned
gate is layer 3's for REG3 models and the window gate for BASELINE. Recorded in every run's stats.
"""

import inspect
import json
import os

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_aux_gate as aux
import explore_b18_gates as g18
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

LAM, TAU, LAM_S, TEMP, WINDOW = dc.LAM, dc.TAU, 1.0, 4.0, 2400
RHL = rc.RHL
FREQ_SAMPLE, FREQ_SEED = 4096, 87_000
LOG_AT = (1, 300, 600, 1200, 2400)

ARMS = {
    "REG3_FREQHINGE": ("freq", "hebb"), "REG3_SELFKL": ("selfkl", "hebb"), "REG3_TEMP": ("temp", "hebb"),
    "REG3_FREQHINGE_D": ("freq", "delta"), "REG3_SELFKL_D": ("selfkl", "delta"), "REG3_TEMP_D": ("temp", "delta"),
    "BASELINE": ("dc", "BASELINE"), "REG3_KEYHINGE": ("dc", "REG3_KEYHINGE"),
    "ORACLE_ALL": ("dc", "ORACLE_ALL"), "BASELINE_D": ("dc", "BASELINE_D"), "REG3_D": ("dc", "REG3_D"), "REG3_KEYHINGE_D": ("dc", "REG3_KEYHINGE_D"),
}


# ── the frequency table (token ids only) ─────────────────────────────────────
def freq_types(tokens):
    """(frequent type ids, table): token-type frequencies of a LongTensor of token ids, sorted, cut at the largest ratio
    between consecutive frequencies. Receives the token ids and nothing else."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long
    cnt = torch.bincount(tokens.reshape(-1)).double()
    fr = cnt / cnt.sum()
    order = [int(t) for t in fr.argsort(descending=True) if fr[t] > 0]
    vals = [float(fr[t]) for t in order]
    cut = max(range(len(vals) - 1), key=lambda i: vals[i] / vals[i + 1])
    return order[:cut + 1], [(t, round(v, 5)) for t, v in zip(order, vals)]


def sample_tokens(task=RHL):
    return task.make_batch(FREQ_SAMPLE, torch.Generator().manual_seed(FREQ_SEED))[0][:, :-1]


FREQ = None


def frequent():
    global FREQ
    if FREQ is None:
        FREQ = freq_types(sample_tokens())
    return FREQ


# ── the terms ────────────────────────────────────────────────────────────────
def term_value(kind, gates, tokens):
    if kind == "freq":
        sel = torch.tensor(frequent()[0])
        mask = (tokens.unsqueeze(-1) == sel).any(-1)                       # (B, T) frequent positions
        out = []
        for g in gates:
            gk = g[mask]                                                    # (n, k)
            out.append(g18.eta2_pool(gk, tokens[mask], int(tokens.max()) + 1))
        return F.relu(torch.stack(out) - TAU)
    if kind == "selfkl":
        out = []
        for g in gates:
            gb = g[:, :RHL.body_len]
            m = gb.mean(1, keepdim=True)
            out.append((gb * (torch.log(gb) - torch.log(m))).sum(-1).mean())
        return torch.stack(out)
    raise KeyError(kind)


class Term87(rc.Reg3BDH):
    def forward(self, tokens, tau=None, track_sat=False):
        st = self.t_state
        training = self.training and torch.is_grad_enabled() and st["count"]
        if training:
            st["n"] += 1
        u = st["n"]
        on = self.t_on and u <= WINDOW
        self.t_T = TEMP if (self.t_kind == "temp" and on) else 1.0
        live = []
        prev = self.__dict__.get("layer_gate")
        if training and on and self.t_kind != "temp":
            inner = prev if prev is not None else (lambda level, h, v, Rprev: Term87.layer_gate(self, level, h, v, Rprev))

            def lg(level, h, v, Rprev):
                g = inner(level, h, v, Rprev)
                live.append(g)
                return g
            self.layer_gate = lg
        try:
            out = rm.RegBDH.forward(self, tokens, tau, track_sat)
        finally:
            if prev is None:
                self.__dict__.pop("layer_gate", None)
            else:
                self.layer_gate = prev
        if training and u in (1, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2):
            st["weights"][str(u)] = (self.t_T if self.t_kind == "temp" else (1.0 if on else 0.0))
        if not live:
            return out
        v = term_value(self.t_kind, live, tokens)
        if u in LOG_AT:
            st["value_at"][str(u)] = [round(float(x), 6) for x in v.detach()]
        if bool((v > 0).any()):
            st["fired"] += 1
        lam = LAM if self.t_kind == "freq" else LAM_S
        return (aux._Inject.apply(out[0], lam * v.mean()),) + tuple(out[1:])

    def layer_gate(self, level, h, v, Rprev):
        if getattr(self, "t_T", 1.0) == 1.0:
            return rc.Reg3BDH.layer_gate(self, level, h, v, Rprev)
        x_in = self.gate_conv(v)
        u = x_in @ self.A_v.T
        r = h.detach() if self.sg else h
        u = u + F.layer_norm(r, (r.shape[-1],)) @ self.A_r.T
        if Rprev is not None:
            u = u + Rprev @ self.A_R.T
        return F.softmax((torch.tanh(u) @ self.W_g2.T) / self.t_T, dim=-1)


def make_model(arm, seed, task=RHL, on=True):
    kind, mem = ARMS[arm]
    if kind == "dc":
        return dc.make_model(mem, seed, task)
    m = rc.make_model("REG3_D" if mem == "delta" else "REG3", seed, task)
    m.__class__ = Term87
    m.t_kind, m.t_on, m.t_T = kind, bool(on), 1.0
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


def run(p):
    arm, seed = p["arm"], p["seed"]
    kind, mem = ARMS[arm]
    if kind == "dc":
        r = dc.run(dict(p, arm=mem))
        r["arm"] = arm
        return r
    a = dict(ec.ARM_A, key=arm)
    holder = {}
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), task=RHL, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, on=not p.get("off"))),
                  run_kw=dict(param_groups=dc.groups(holder, dc.LR)), check=dc.switch(holder, 0), keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(RHL)
    rec.update(arm=arm, recipe="slow", group_names=holder.get("names"), stream_acc=gc.stream_acc(m, RHL, ev),
               probe=rc.probe_rh(m, RHL, gc.probe_data(RHL)), acc_by_cell=rc.acc_by_cell(m, RHL, ev),
               write_order=ft.write_order_acc(m, RHL, ev), term={k: v for k, v in m.t_state.items() if k != "count"})
    if kind == "freq":
        rec["freq"] = dict(types=frequent()[0], table=frequent()[1][:8])
    rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── reproduction and CHECKs ──────────────────────────────────────────────────
def s82_record(arm, seed):
    with open(os.path.join(gc.G_DIR, "s82_devices_results.json")) as f:
        st = json.load(f)
    return st["runs"][f"{arm}|{seed}|{st['meta']['sha'][:12]}|{ec.cpu_model()}"]


def trunc(r, upto=2400):
    return dict(curve=[c for c in r["curve"] if c[0] <= upto],
                stats=[s for s in r["stats"] if isinstance(s["step"], int) and s["step"] <= upto])


def reproduce(p):
    out = {}
    for arm in p.get("arms", ("BASELINE", "REG3_KEYHINGE")):
        r = dc.run(dict(arm=arm, seed=2100, iters=2400))
        out[arm] = trunc(r) == trunc(s82_record(arm, 2100))
    return out


def checks(p):
    rows = [(f"S83 inherited: {n}", v) for n, v in __import__("explore_g_s83_child").checks({})]
    seed = 2700
    x = RHL.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
    # (1) each device with its switch off equals REG3 bit for bit through 2400 (REG3 rerun on this path at 2700)
    ref = dc.run(dict(arm="REG3", seed=seed, iters=2400))
    for arm in ("REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP"):
        r = run(dict(arm=arm, seed=seed, iters=2400, off=True))
        rows.append((f"{arm} with its switch off equals REG3|{seed} through 2400 bit for bit (curve {r['curve']})",
                     trunc(r) == trunc(ref)))
    # (2) the frequency table from token ids only: freq_types has one parameter, asserts a LongTensor; the sample is
    #     make_batch's token tensor alone (the stream labels it also returns are dropped before the call)
    sig = list(inspect.signature(freq_types).parameters)
    types, table = frequent()
    bad = False
    try:
        freq_types(torch.zeros(4, 2))                                   # a float tensor (as a label or gate would be) is refused
    except AssertionError:
        bad = True
    rows.append((f"the frequency table is computed from token ids only: freq_types{tuple(sig)} asserts a LongTensor of ids "
                 f"(a float tensor is refused: {bad}); frequent types {types}; table head {table[:7]}",
                 sig == ["tokens"] and bad and sorted(types) == [0, 1, 2, 3, 4, 5]))
    # (3) the coincidence count over the sample: per sequence, frequent positions vs the key term's positions
    xs = sample_tokens()
    sel = torch.tensor(types)
    mask = (xs.unsqueeze(-1) == sel).any(-1)
    kpos, _ = ft.body_positions(RHL, xs)
    kmask = torch.zeros_like(mask)
    kmask.scatter_(1, kpos, True)
    nf, nk, both = mask.sum(1), kmask.sum(1), (mask & kmask).sum(1)
    rows.append((f"coincidence over {FREQ_SAMPLE} sequences: frequent positions per sequence {sorted(set(nf.tolist()))}, key-term "
                 f"positions {sorted(set(nk.tolist()))}, both {sorted(set(both.tolist()))} (every key-term position is frequent; "
                 f"the other frequent positions are the 5 CTX and the query's KEY)",
                 set(nf.tolist()) == {14} and set(nk.tolist()) == {8} and set(both.tolist()) == {8}))
    # (4) the freq term restricted to the key term's positions and types equals the key term (sanity of the grouping)
    g = torch.softmax(torch.randn(32, x.shape[1], 2, generator=torch.Generator().manual_seed(3)), -1)
    kp, _ = ft.body_positions(RHL, x)
    e_key = float(dc.key_eta([g], x, RHL)[0])
    gk = g.gather(1, kp.unsqueeze(-1).expand(-1, -1, 2))
    e_by_tok = float(g18.eta2_pool(gk, x.gather(1, kp), int(x.max()) + 1))
    rows.append((f"grouping by token id at the key positions equals the key term's grouping by key id ({e_by_tok:.6f} vs {e_key:.6f})",
                 abs(e_by_tok - e_key) < 1e-7))
    # (5) gradients: freq and selfkl reach the gate's parameters (and the embedding through its input), not the stack
    stack = ("encoder", "encoder_v", "decoder", "lm_head")
    for arm in ("REG3_FREQHINGE", "REG3_SELFKL"):
        m = make_model(arm, seed)
        m.train()
        m.t_state["count"] = False
        gates = s56.live_gates(m, x)
        v = term_value(m.t_kind, gates, x)
        if m.t_kind == "freq":                                          # the hinge's argument (relu may be 0 at init)
            v = v + TAU
        names = [n for n, q in m.named_parameters() if q.requires_grad]
        gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
        nz = {n: (0.0 if g_ is None else float(g_.abs().max())) for n, g_ in zip(names, gr)}
        rows.append((f"{arm}: the term's gradient reaches {sorted(n for n, v_ in nz.items() if v_ > 0)}; the stack "
                     f"{[n for n in stack if nz.get(n, 0) > 0] or 'none'}",
                     nz.get("W_g2", 0) > 0 and nz.get("A_r", 0) > 0 and all(nz.get(n, 0) == 0 for n in stack)))
    # (6) TEMP: during the window the gate is softmax(logits / 4); after it the same model is REG3's gate; no parameter added
    mt = make_model("REG3_TEMP", seed)
    mr = rc.make_model("REG3", seed)
    with torch.no_grad():
        mt.eval()
        mt.t_T = TEMP
        g4 = s56.all_gates(mt, x)[0]
        mt.t_state["n"] = WINDOW + 1                                    # past the window: T = 1
        g1 = s56.all_gates(mt, x)[0]
        gr_ = s56.all_gates(mr, x)[0]
    rows.append(("REG3_TEMP: after the window its gates equal REG3's bit for bit; inside it the gates are flatter "
                 f"(mean |g - 1/2| {float((g4[2] - 0.5).abs().mean()):.2e} vs {float((gr_[2] - 0.5).abs().mean()):.2e}); "
                 "no parameter added",
                 all(torch.equal(a_, b_) for a_, b_ in zip(g1, gr_)) and float((g4[2] - 0.5).abs().mean()) < float((gr_[2] - 0.5).abs().mean())
                 and dict(mt.named_parameters()).keys() == dict(mr.named_parameters()).keys()))
    # (7) the schedules: weights / temperatures at 1, 2399, 2400, 2401, 2402 over training forwards
    for arm in ("REG3_SELFKL", "REG3_TEMP"):
        m = make_model(arm, seed)
        m.train()
        for _ in range(WINDOW + 2):                                     # real training forwards (no optimizer step)
            out = m(x[:1])[0]
            out.sum().backward()
            m.zero_grad(set_to_none=True)
        w = m.t_state["weights"]
        want = (TEMP, 1.0) if arm == "REG3_TEMP" else (1.0, 0.0)
        rows.append((f"{arm} schedule over training forwards {w}", w["1"] == want[0] and w[str(WINDOW)] == want[0]
                     and w[str(WINDOW + 1)] == want[1] and w[str(WINDOW + 2)] == want[1]))
    # (8) the temperature in a real forward pass: one training forward at update 1 uses T = 4, and the weight log says so
    m = make_model("REG3_TEMP", seed)
    m.train()
    m(x[:2])
    rows.append((f"REG3_TEMP: the first training forward ran at T {m.t_T} (log {m.t_state['weights']})", m.t_T == TEMP
                 and m.t_state["weights"].get("1") == TEMP))
    return [[n, bool(v)] for n, v in rows]
