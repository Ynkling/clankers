#!/usr/bin/env python
"""
explore_g_s89_child.py — EXPLORATORY, not a result. Run code of Session G's screen S89 (explore_g_s89): the frequency
hinge and a within-sequence repeat hinge on tasks where keys are not a frequent class. Runs INSIDE a child process on this
branch's own modules, 1 torch thread. It imports S82's (explore_g_devices_child), S87's (explore_g_s87_child) and
S71-S73's (explore_g_rhl_child) children unchanged; every arm is built and trained on the task named in the payload.

TASKS: "lite" (F's RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S70's; used for the CHECKs and the S87 reproduction) and the
round-6 tasks from explore_f_tasks.py vendored byte for byte from claude/explore-F at the commit F names (TASKS below;
filled in when F's module is vendored, before any S89 run).
MODEL AND RECIPE (S87's): Hebbian (delta: beta 1, L2 keys, tied write, decay 0.95); k = 2, no conv, 24000 updates; SLOW
(S56's groups) for every learned arm; the perfect gate (ORACLE_ALL) Adam 1e-3 throughout. REG3 = S72's register gate shared
by the three layers. Every hinge acts on updates 1-2400 only (the update count = training forwards), through
explore_aux_gate._Inject, as LAM x mean_l relu(eta2(g^(l) at the hinge's positions, grouped by token id) - TAU), LAM 1.0,
TAU 0.2 (S52's constants); eta2 = explore_b18_gates.eta2_pool.
  REG3_KEYHINGE    S82's key term: the body's KEY positions and each key's id (task structure: the labelled ceiling).
  REG3_FREQHINGE   S87's frequency hinge: the positions of the frequent token types, the types cut from the task's own
                   4096-sequence sample (torch.Generator(87_000)) at the largest ratio between consecutive sorted
                   frequencies (explore_g_s87_child.freq_types, which receives the token-id tensor only).
  REG3_REPEATHINGE the positions of every token type that occurs more than once in the same input sequence (tokens[:, :-1];
                   repeat_mask receives the token-id tensor only); a within-sequence statistic: no corpus table, no label.
                   On the lite task (4096 sequences, Generator(87_000)) it selects per sequence all 5 CTX positions (each
                   stream's 2 block headers repeat, and the query's CTX), all 9 KEY positions (each key once per stream,
                   and the query's key), and 2.63 VAL positions on average (values repeated by chance across keys): 16.6
                   positions, of which the key term's 8 are 48%. NOT "the keys and nothing else": at the CTX positions the
                   grouping by token id is a grouping by stream, as in S87's frequency hinge.
Records: explore_g_rhl_child's statistics (gate_stats_k's "spread" is the commitment), layer statistics, in-run probe,
per-cell accuracy, write order; the hinge's firings, its value at 1, 300, 600, 1200, 2400, its weights at 1 and 2399-2402,
and the positions it selected (counts per role).
"""

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
import explore_g_s87_child as s87
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

LAM, TAU, WINDOW = dc.LAM, dc.TAU, 2400
LOG_AT = (1, 300, 600, 1200, 2400)
TASKS = {"lite": rc.RHL}                    # the round-6 tasks are added when F's module is vendored


def task_of(name):
    return TASKS[name]


# ── the statistics (token ids only) ──────────────────────────────────────────
def repeat_mask(tokens):
    """(B, T) bool: positions whose token type occurs more than once in the same sequence. Receives token ids only."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long and tokens.dim() == 2
    V = int(tokens.max()) + 1
    cnt = torch.zeros(tokens.shape[0], V, dtype=torch.long).scatter_add_(1, tokens, torch.ones_like(tokens))
    return cnt.gather(1, tokens) > 1


_FREQ = {}


def frequent(task_name):
    if task_name not in _FREQ:
        t = task_of(task_name)
        _FREQ[task_name] = s87.freq_types(t.make_batch(s87.FREQ_SAMPLE, torch.Generator().manual_seed(s87.FREQ_SEED))[0][:, :-1])
    return _FREQ[task_name]


def hinge_mask(kind, tokens, task_name):
    if kind == "freq":
        sel = torch.tensor(frequent(task_name)[0])
        return (tokens.unsqueeze(-1) == sel).any(-1)
    if kind == "repeat":
        return repeat_mask(tokens)
    raise KeyError(kind)


def term_value(kind, gates, tokens, task_name):
    if kind == "key":
        return F.relu(dc.key_eta(gates, tokens, task_of(task_name)) - TAU)
    mask = hinge_mask(kind, tokens, task_name)
    if not bool(mask.any()):
        return torch.zeros(len(gates))
    V = int(tokens.max()) + 1
    return F.relu(torch.stack([g18.eta2_pool(g[mask], tokens[mask], V) for g in gates]) - TAU)


class Term89(rc.Reg3BDH):
    def forward(self, tokens, tau=None, track_sat=False):
        st = self.t_state
        training = self.training and torch.is_grad_enabled() and st["count"]
        if training:
            st["n"] += 1
        u = st["n"]
        on = self.t_on and u <= WINDOW
        live = []
        prev = self.__dict__.get("layer_gate")
        if training and on:
            inner = prev if prev is not None else (lambda level, h, v, Rprev: rc.Reg3BDH.layer_gate(self, level, h, v, Rprev))

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
            st["weights"][str(u)] = 1.0 if on else 0.0
        if not live:
            return out
        v = term_value(self.t_kind, live, tokens, self.t_task)
        if u in LOG_AT:
            st["value_at"][str(u)] = [round(float(x), 6) for x in v.detach()]
        if bool((v > 0).any()):
            st["fired"] += 1
        return (aux._Inject.apply(out[0], LAM * v.mean()),) + tuple(out[1:])


# arm -> (kind, memory)
ARMS = {
    "ORACLE_ALL": ("oracle", "hebb"), "BASELINE": ("base", "hebb"), "REG3": ("reg3", "hebb"),
    "REG3_KEYHINGE": ("keyhinge", "hebb"), "REG3_FREQHINGE": ("freq", "hebb"), "REG3_REPEATHINGE": ("repeat", "hebb"),
    "BASELINE_D": ("base", "delta"), "REG3_D": ("reg3", "delta"), "REG3_KEYHINGE_D": ("keyhinge", "delta"),
    "REG3_FREQHINGE_D": ("freq", "delta"), "REG3_REPEATHINGE_D": ("repeat", "delta"),
}


def make_model(arm, seed, task_name, on=True):
    kind, mem = ARMS[arm]
    task = task_of(task_name)
    d = "_D" if mem == "delta" else ""
    if kind == "oracle":
        return rc.make_model("ORACLE_ALL" + d, seed, task)
    if kind == "base":
        return rc.make_model("BASELINE" + d, seed, task)
    if kind == "reg3":
        return rc.make_model("REG3" + d, seed, task)
    if kind == "keyhinge":
        m = dc.make_model("REG3_KEYHINGE" + d, seed, task, lam=dc.LAM if on else 0.0)
        return m
    m = rc.make_model("REG3" + d, seed, task)
    m.__class__ = Term89
    m.t_kind, m.t_on, m.t_task = kind, bool(on), task_name
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


def selected(task_name, kind):
    """Per role, the mean count of positions the hinge selects in the task's 4096-sequence sample (token ids only)."""
    t = task_of(task_name)
    x = t.make_batch(s87.FREQ_SAMPLE, torch.Generator().manual_seed(s87.FREQ_SEED))[0][:, :-1]
    m = hinge_mask(kind, x, task_name)
    kpos, _ = ft.body_positions(t, x)
    km = torch.zeros_like(m)
    km.scatter_(1, kpos, True)
    S = t.S
    is_ctx, is_key = x < S, (x >= S) & (x < S + t.P)
    out = dict(total=float(m.sum(1).float().mean()), ctx=float((m & is_ctx).sum(1).float().mean()),
               key=float((m & is_key).sum(1).float().mean()), val=float((m & ~is_ctx & ~is_key).sum(1).float().mean()),
               key_term=float((m & km).sum(1).float().mean()), key_term_of=float(km.sum(1).float().mean()))
    if kind == "freq":
        out["types"], out["table"] = frequent(task_name)[0], frequent(task_name)[1][:12]
    return out


def run(p):
    """p: arm ("<task>.<ARM>" or ARM with p["task"]), seed, iters, off."""
    arm, seed = p["arm"], p["seed"]
    if "." in arm:
        tname, arm = arm.split(".", 1)
    else:
        tname = p["task"]
    task = task_of(tname)
    kind, mem = ARMS[arm]
    a = dict(rc.CEIL if kind == "oracle" else ec.ARM_A, key=arm)
    holder, kw, check = {}, {}, None
    if kind != "oracle":
        kw = dict(param_groups=dc.groups(holder, dc.LR))
        check = dc.switch(holder, 0)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), task=task, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, tname, on=not p.get("off"))), run_kw=kw, check=check,
                  keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=p["arm"], task=tname, recipe="adam" if kind == "oracle" else "slow", group_names=holder.get("names"),
               stream_acc=gc.stream_acc(m, task, ev), probe=rc.probe_rh(m, task, gc.probe_data(task)),
               acc_by_cell=rc.acc_by_cell(m, task, ev), write_order=ft.write_order_acc(m, task, ev))
    if isinstance(m, Term89):
        rec["term"] = {k: v for k, v in m.t_state.items() if k != "count"}
    if isinstance(m, dc.KeyHingeBDH):
        rec["hinge"] = {k: v for k, v in m.kh.items() if k != "count"}
    if kind != "oracle":
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def s87_record(arm, seed):
    with open(os.path.join(gc.G_DIR, "s87_results.json")) as f:
        st = json.load(f)
    return st["runs"][f"{arm}|{seed}|{st['meta']['sha'][:12]}|{ec.cpu_model()}"]


def checks(p):
    rows = []
    tasks = p.get("tasks", ["lite"])
    # (1) S87's REG3_FREQHINGE|2700 reproduces bit for bit through 2400 under the current code: through S87's own run path,
    #     and through this screen's run path on the lite task
    ref = s87.trunc(s87_record("REG3_FREQHINGE", 2700))
    r87 = s87.run(dict(arm="REG3_FREQHINGE", seed=2700, iters=2400))
    r89 = run(dict(arm="REG3_FREQHINGE", seed=2700, iters=2400, task="lite"))
    rows.append((f"S87's REG3_FREQHINGE|2700 reproduces bit for bit through 2400 under the current code (S87's path "
                 f"{s87.trunc(r87) == ref}; S89's path on the lite task {s87.trunc(r89) == ref}; curve {r89['curve']})",
                 s87.trunc(r87) == ref and s87.trunc(r89) == ref))
    # (2) the statistics from token ids only: one parameter each, a LongTensor asserted, a float tensor refused
    refused = []
    for fn in (repeat_mask, s87.freq_types):
        try:
            fn(torch.zeros(4, 3))
            refused.append(False)
        except AssertionError:
            refused.append(True)
    import inspect
    rows.append(("the frequency and repeat statistics receive token ids only: repeat_mask(tokens), freq_types(tokens), each "
                 f"asserts a LongTensor and refuses a float tensor ({refused})",
                 all(refused) and list(inspect.signature(repeat_mask).parameters) == ["tokens"]
                 and list(inspect.signature(s87.freq_types).parameters) == ["tokens"]))
    xr = torch.tensor([[0, 5, 7, 5, 1, 9, 0]])
    rows.append((f"repeat_mask by hand: {repeat_mask(xr).tolist()} (types 0 and 5 repeat)",
                 repeat_mask(xr).tolist() == [[True, True, False, True, False, False, True]]))
    for tname in tasks:
        task = task_of(tname)
        seed = p.get("seed", 2842)
        x = task.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
        # (3) each hinge with its switch off equals REG3 bit for bit through 2400 on this task
        refr = run(dict(arm="REG3", seed=seed, iters=2400, task=tname))
        for arm in ("REG3_FREQHINGE", "REG3_REPEATHINGE", "REG3_KEYHINGE"):
            r = run(dict(arm=arm, seed=seed, iters=2400, task=tname, off=True))
            rows.append((f"{tname}: {arm} with its switch off equals REG3|{seed} through 2400 bit for bit (curve {r['curve']})",
                         s87.trunc(r) == s87.trunc(refr)))
        # (4) what each hinge selects, per sequence (counts per role) and the frequency cut
        for kind in ("freq", "repeat"):
            s = selected(tname, kind)
            rows.append((f"{tname}: the {kind} hinge selects per sequence {s['total']:.2f} positions (CTX {s['ctx']:.2f}, KEY "
                         f"{s['key']:.2f}, VAL {s['val']:.2f}); of the key term's {s['key_term_of']:.0f} positions {s['key_term']:.2f}"
                         + (f"; frequent types {s['types']}; table head {s['table']}" if kind == "freq" else ""), True))
        # (5) gradients: each term reaches the gate's parameters (and the embedding through its input), not the stack
        stack = ("encoder", "encoder_v", "decoder", "lm_head")
        for arm in ("REG3_FREQHINGE", "REG3_REPEATHINGE"):
            m = make_model(arm, seed, tname)
            m.train()
            m.t_state["count"] = False
            gates = s56.live_gates(m, x)
            mask = hinge_mask(m.t_kind, x, tname)
            V = int(x.max()) + 1
            v = torch.stack([g18.eta2_pool(g[mask], x[mask], V) for g in gates]) if bool(mask.any()) else None
            if v is None:
                rows.append((f"{tname}: {arm} selects no position on this batch: the term is identically 0 and the arm is REG3",
                             True))
                continue
            names = [n for n, q in m.named_parameters() if q.requires_grad]
            gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
            nz = {n: (0.0 if g_ is None else float(g_.abs().max())) for n, g_ in zip(names, gr)}
            rows.append((f"{tname}: {arm}'s gradient reaches {sorted(n for n, v_ in nz.items() if v_ > 0)}; the stack "
                         f"{[n for n in stack if nz.get(n, 0) > 0] or 'none'}",
                         nz.get("W_g2", 0) > 0 and nz.get("A_r", 0) > 0 and all(nz.get(n, 0) == 0 for n in stack)))
        # (6) the repeat hinge's schedule over real training forwards
        m = make_model("REG3_REPEATHINGE", seed, tname)
        m.train()
        for _ in range(WINDOW + 2):
            out = m(x[:1])[0]
            out.sum().backward()
            m.zero_grad(set_to_none=True)
        w = m.t_state["weights"]
        rows.append((f"{tname}: REG3_REPEATHINGE schedule over training forwards {w}",
                     w["1"] == 1.0 and w[str(WINDOW)] == 1.0 and w[str(WINDOW + 1)] == 0.0 and w[str(WINDOW + 2)] == 0.0))
    return [[n, bool(v)] for n, v in rows]


def oracle_check(p):
    """The perfect gate on the task's validity seeds (run in full by the parent before any learned arm)."""
    return run(dict(arm="ORACLE_ALL", seed=p["seed"], task=p["task"]))
