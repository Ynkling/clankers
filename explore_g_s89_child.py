#!/usr/bin/env python
"""
explore_g_s89_child.py — EXPLORATORY, not a result. Run code of Session G's screen S89 (explore_g_s89): the frequency
hinge and a within-sequence repeat hinge on tasks where keys are not a frequent class. Delta memory is the primary
memory (Referee V's review and F's report 9: the Hebbian perfect gate binds neither round-6 task). Runs INSIDE a child
process on this branch's own modules, 1 torch thread. It imports S82's (explore_g_devices_child), S87's
(explore_g_s87_child) and S71-S73's (explore_g_rhl_child) children unchanged; every arm is built and trained on the task
named in the payload.

TASKS (explore_f_tasks.py vendored byte for byte from claude/explore-F at 50dc8e4, blob ffe249bb..., asserted by the
CHECKs): "lite" (F's RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S70's; the CHECKs and the reproductions of S87's records),
"pool" (RandHeaderTaskPool(): lite's batch with each sequence's 4 keys renamed to 4 of 32 types; vocabulary 50) and
"allpool" (RandHeaderTaskAllPool(): also the 2 CTX tokens renamed to 2 of 16 types; vocabulary 64; stream index by first
appearance).
TASK-AWARE CODE (new here; on lite each equals the lite function it replaces, CHECKed bit for bit):
  key positions and ids     lite: explore_f_tasks.body_positions, id = token - S over P types; round-6 tasks:
                            task.body_positions, id = task.key_index over task.k_pool types.
  the key term (key_eta89)  explore_b18_gates.eta2_pool of each layer's gate at those positions grouped by those ids
                            (on lite: explore_g_devices_child.key_eta).
  roles, the query          CTX / KEY / VAL from task.role_masks (lite: token ranges); the query's stream is
                            task.stream_labels at qpos (the stream index; lite: the CTX token before it).
  statistics                lite: explore_g_rhl_child.stats_fn unchanged; round-6 tasks: explore_f_tasks.pool_stats(ROUTE_AT)
                            (gate_stats_k + pool_routing_stats) + beta^R by role + layer_stats89 (explore_g_rhl_child.
                            layer_stats with the positions, ids and query above). After the run: stream_acc89, probe_rh89,
                            acc_by_cell89, write_order_acc89 (the lite functions with the stream index and the task's roles).
  the perfect gate          TaskPerfectBDH: test_channel_binding's 'ceiling' model whose gate is task.perfect_gate on the
                            round-6 tasks (perfect_gate_general(tokens, ctx_tokens) on lite); delta: explore_delta_mem
                            (beta 1, L2 keys, tied write, decay 0.95), the gate passed by set_gates. CHECKed equal to
                            explore_g_rhl_child's ORACLE_ALL_D model on lite and pool (parameters, logits, gates bit for bit).
MODEL AND RECIPE (S87's): k = 2, no conv, 24000 updates; SLOW (S56's groups: the gate group at 1e-3, the rest 1e-4 for
updates 1-2400 and 1e-3 after) for every arm, the perfect gate included (its gate group is empty: every parameter at 1e-4
for updates 1-2400, 1e-3 after; F's S89 pilots' recipe). REG3 = S72's register gate shared by the three layers. Every
hinge acts on updates 1-2400 only (the update count = training forwards), through explore_aux_gate._Inject, as
LAM x mean_l relu(eta2(g^(l) at the hinge's positions, grouped by its ids) - TAU), LAM 1.0, TAU 0.2 (S52's constants).
  REG3_KEYHINGE    the key term (key_eta89): the body's 8 KEY positions and each key's id (the labelled ceiling). On lite
                   computed exactly as S82's KeyHingeBDH (CHECKed: S87's REG3_KEYHINGE|2700 and REG3_KEYHINGE_D|2720
                   reproduce through this run()).
  REG3_FREQHINGE   S87's frequency hinge: the positions of the frequent token types (explore_g_s87_child.freq_types,
                   the largest-ratio cut, = F's largest_ratio_cut, CHECKed) on the task's own 4096-sequence sample
                   (explore_f_tasks.freq_sample: torch.Generator(87_000), tokens[:, :-1]), grouped by token id. On lite it
                   selects 2 CTX + 4 KEY types (14 positions, the key term's 8 among them); on pool the 2 CTX types only
                   (5 CTX positions, none of the key term's): a CTX hinge; on allpool the 16 VAL types only (8 VAL
                   positions, none of the key term's): a value hinge.
  REG3_REPEATHINGE the positions of every token type that occurs more than once in the same input sequence, grouped by
                   token id; repeat_mask receives the token-id tensor only. On every task 5 CTX + 9 KEY + ~2.67 VAL
                   positions per sequence, identical position by position across the three tasks (F's renaming is
                   one-to-one within each role and sequence), covering the key term's 8 (48% of what it selects).
TOKEN IDS ONLY (fix 7): frequent() asserts its sample has shape (4096, L - 1), values in [0, vocab) and equals
ft.freq_sample(task); hinge_mask() asserts (B, L - 1) and [0, vocab). A CHECK shows frequent() raises when handed
task.stream_labels(x) or the key ids instead.
Records: the statistics above (gate_stats_k's "spread" is the commitment), layer statistics, in-run probe, per-cell
accuracy, write order; the hinge's firings, its value at 1, 300, 600, 1200, 2400, its weights at 1 and 2399-2402.
"""

import inspect
import json
import os
import subprocess

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_aux_gate as aux
import explore_b18_gates as g18
import explore_delta_mem as dm
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import explore_g_s87_child as s87
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_channel_binding import ARCH, arms as cb_arms
from test_instrument_v2 import perfect_gate_general
from test_multilayer_binding import MultiBDH, build, probe_batch
from test_router_layout import eta2
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
LAM, TAU, WINDOW = dc.LAM, dc.TAU, 2400
LOG_AT = (1, 300, 600, 1200, 2400)
F_COMMIT, F_BLOB = "50dc8e4", "ffe249bb644423479e869cff11d2d723ac35450b"
PREFIX_COMMIT, PREFIX_BLOB, PREFIX_LEN = "a199d5e", "de4746371dfa63b05630a644c4fe0b2bdce29e32", 14101
TASKS = {"lite": rc.RHL, "pool": ft.RandHeaderTaskPool(), "allpool": ft.RandHeaderTaskAllPool()}
POOLED = ("pool", "allpool")
STATS_POOL = ft.pool_stats(c2.ROUTE_AT)


def task_of(name):
    return TASKS[name]


def is_pool(task):
    return isinstance(task, ft.RandHeaderTaskPool)


# ── task-aware positions, ids, roles, query ──────────────────────────────────
def key_pos_ids(task, x):
    """(kpos, kid, n_types): the body's KEY positions in order of appearance, the key id at each, the number of ids."""
    if is_pool(task):
        kpos, _ = task.body_positions(x)
        return kpos, task.key_index(x).gather(1, kpos), task.k_pool
    kpos, _ = ft.body_positions(task, x)
    return kpos, x.gather(1, kpos) - task.S, task.P


def body_pos(task, x):
    return task.body_positions(x) if is_pool(task) else ft.body_positions(task, x)


def roles89(task, x):
    if not is_pool(task):
        return rc.roles(task, x)
    body = torch.zeros_like(x, dtype=torch.bool)
    body[:, :task.body_len] = True
    rm_ = task.role_masks(x)
    return dict(CTX=rm_["ctx"] & body, KEY=rm_["key"] & body, VAL=rm_["val"] & body, QUERY=~body)


def query(task, x):
    """(query stream index, query key id) at qpos."""
    q = task.qpos[0]
    if is_pool(task):
        return task.stream_labels(x)[:, q], x[:, q] - task.key0
    return x[:, q - 1], x[:, q] - task.S


def key_eta89(gates, tokens, task):
    """the key term's eta2 per layer, on any of the three tasks (= explore_g_devices_child.key_eta on lite)."""
    kpos, kid, G = key_pos_ids(task, tokens)
    out = []
    for g in gates:
        gk = g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, g.shape[-1]))
        out.append(g18.eta2_pool(gk, kid, G))
    return torch.stack(out)


# ── the statistics (token ids only) ──────────────────────────────────────────
def _assert_tokens(tokens, task):
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long and tokens.dim() == 2, "token ids only"
    assert tokens.shape[1] == task.L - 1, f"token ids only: width {tokens.shape[1]} != L - 1 = {task.L - 1}"
    assert int(tokens.min()) >= 0 and int(tokens.max()) < task.vocab, "token ids only: values outside [0, vocab)"


def repeat_mask(tokens):
    """(B, T) bool: positions whose token type occurs more than once in the same sequence. Receives token ids only."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long and tokens.dim() == 2
    V = int(tokens.max()) + 1
    cnt = torch.zeros(tokens.shape[0], V, dtype=torch.long).scatter_add_(1, tokens, torch.ones_like(tokens))
    return cnt.gather(1, tokens) > 1


_FREQ = {}


def frequent(task_name, sample=None):
    """S87's largest-ratio cut on the task's own token sample. The sample must be ft.freq_sample(task) itself."""
    if sample is None and task_name in _FREQ:
        return _FREQ[task_name]
    task = task_of(task_name)
    ref = ft.freq_sample(task)
    sample = ref if sample is None else sample
    _assert_tokens(sample, task)
    assert sample.shape == ref.shape and torch.equal(sample, ref), "frequent(): not the task's token sample"
    out = s87.freq_types(sample)
    _FREQ[task_name] = out
    return out


def hinge_mask(kind, tokens, task_name):
    _assert_tokens(tokens, task_of(task_name))
    if kind == "freq":
        sel = torch.tensor(frequent(task_name)[0])
        return (tokens.unsqueeze(-1) == sel).any(-1)
    if kind == "repeat":
        return repeat_mask(tokens)
    raise KeyError(kind)


def term_value(kind, gates, tokens, task_name):
    if kind == "key":
        return F.relu(key_eta89(gates, tokens, task_of(task_name)) - TAU)
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


class TaskPerfectBDH(MultiBDH):
    """The perfect gate from the task: task.perfect_gate on the round-6 tasks, perfect_gate_general on lite. For a delta
    memory the gate is passed to it by set_gates."""

    def forward(self, tokens, tau=None, track_sat=False):
        t = self.pt_task
        gr = gw = (t.perfect_gate(tokens, self.lm_head.dtype) if is_pool(t)
                   else perfect_gate_general(tokens, self.ctx_tokens, self.lm_head.dtype))
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(gr, gw)
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits, _ = BDH.forward(self, tokens)
        return logits, None, gr, gw


def ceil_arm(task):
    return {a["key"]: a for a in cb_arms(task)}["ceiling"]


# arm -> (kind, memory). The parent runs the _D arms; the Hebbian ones serve the reproductions of S87's records.
ARMS = {
    "ORACLE": ("oracle", "hebb"), "BASELINE": ("base", "hebb"), "REG3": ("reg3", "hebb"),
    "REG3_KEYHINGE": ("key", "hebb"), "REG3_FREQHINGE": ("freq", "hebb"), "REG3_REPEATHINGE": ("repeat", "hebb"),
    "ORACLE_D": ("oracle", "delta"), "BASELINE_D": ("base", "delta"), "REG3_D": ("reg3", "delta"),
    "REG3_KEYHINGE_D": ("key", "delta"), "REG3_FREQHINGE_D": ("freq", "delta"), "REG3_REPEATHINGE_D": ("repeat", "delta"),
}


def make_model(arm, seed, task_name, on=True):
    kind, mem = ARMS[arm]
    task = task_of(task_name)
    d = "_D" if mem == "delta" else ""
    if kind == "oracle":
        m = build(task, ceil_arm(task), ARCH, seed)
        if mem == "delta":
            dm.to_delta(m, **s56.DELTA_KW)
        m.__class__ = TaskPerfectBDH
        m.pt_task = task
        return m
    if kind == "base":
        return rc.make_model("BASELINE" + d, seed, task)
    m = rc.make_model("REG3" + d, seed, task)
    if kind == "reg3":
        return m
    m.__class__ = Term89
    m.t_kind, m.t_on, m.t_task = kind, bool(on), task_name
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


# ── task-aware statistics (each equals the lite function on lite; CHECKed) ───
@torch.no_grad()
def write_rate89(model, task, probe):
    if not isinstance(model, rm.RegBDH) or not getattr(model, "reg", False):
        return None
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        w = torch.sigmoid(model.embed(pinp) @ model.w_R + model.b_R)
    finally:
        model.train(was)
    return {r: w[m_].mean().item() for r, m_ in roles89(task, pinp).items()}


@torch.no_grad()
def layer_stats89(model, task, probe):
    pinp = probe[0]
    gates = s56.all_gates(model, pinp)[0]
    S, n = task.S, task.S * task.P
    B = pinp.shape[0]
    kpos, kid, _ = key_pos_ids(task, pinp)
    vpos = kpos + 1
    lab = task.stream_labels(pinp)
    ctx = lab.gather(1, kpos)
    j = torch.arange(n)
    labels = dict(stream=ctx, key=kid, half=(j >= n // 2).long().expand(B, n), index=j.expand(B, n))
    rl = roles89(task, pinp)
    qs, qk = query(task, pinp)
    qpos = task.qpos[0]
    same_key = kid == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    out = {}
    for li, g in enumerate(gates, 1):
        d = {}
        p = g[..., 0]
        k_ = g.shape[-1]
        for role, pos in (("key", kpos), ("val", vpos)):
            for name, lb in labels.items():
                d[f"eta_{role}_by_{name}"] = eta2(p.gather(1, pos), lb)
            gg = g.gather(1, pos.unsqueeze(-1).expand(-1, -1, k_))
            ll = lab.gather(1, pos)
            means = [gg[ll == s].mean(0) for s in range(S)]
            cmap = [int(m_.argmax()) for m_ in means]
            d[f"ch_map_{role}"] = cmap
            d[f"one_to_one_{role}"] = len(set(cmap)) == S
        gv = g.gather(1, vpos.unsqueeze(-1).expand(-1, -1, k_))
        dots = (gv * g[:, qpos][:, None, :]).sum(-1)
        d["margin"] = ((dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)).mean().item()
        d["table"] = {r: [[round(v, 4) for v in g[m_ & (lab == s)].mean(0).tolist()] for s in range(S)]
                      for r, m_ in rl.items()}
        out[f"L{li}"] = d
    return out


def stats_fn89(model, task, probe, step):
    if not is_pool(task):
        return rc.stats_fn(model, task, probe, step)
    out = STATS_POOL(model, task, probe, step)
    out["beta"] = write_rate89(model, task, probe)
    if step in rc.LAYER_AT:
        out["layers"] = layer_stats89(model, task, probe)
    return out


@torch.no_grad()
def _logits(model, inp):
    was = model.training
    model.eval()
    try:
        return tbo.logits_of(model, inp)
    finally:
        model.train(was)


def stream_acc89(model, task, data):
    ql, qt, _ = task.select(_logits(model, data[:, :-1]), data[:, 1:], None)
    qs, _ = query(task, data[:, :-1])
    hit = (ql.argmax(-1) == qt).float()
    return [hit[qs == s].mean().item() for s in range(task.S)]


def probe_rh89(model, task, data, layers=("L2", "L3", "final")):
    x, tr, te = data
    feats, _ = rc.capture(model, x)
    kpos, vpos = body_pos(task, x)
    lab = task.stream_labels(x)
    out = {}
    for name in layers:
        f = feats[name]
        for role, pos in (("K", kpos), ("V", vpos)):
            X = f.gather(1, pos.unsqueeze(-1).expand(-1, -1, f.shape[-1])).double().numpy()
            y = lab.gather(1, pos).numpy()
            Xtr, ytr = X[tr.numpy()].reshape(-1, X.shape[-1]), y[tr.numpy()].reshape(-1)
            Xte, yte = X[te.numpy()].reshape(-1, X.shape[-1]), y[te.numpy()].reshape(-1)
            hit, tracc = gc.fit_score(Xtr, ytr, Xte, yte)
            k, n = int(hit.sum()), int(hit.size)
            out[f"{name}|{role}"] = dict(acc=k / n, k=k, n=n, wilson=gc.wilson(k, n), train_acc=tracc)
    return out


def acc_by_cell89(model, task, data):
    """Held-out accuracy by (query stream, block of the target pair among its stream's blocks, pair index in the block)."""
    ql, qt, _ = task.select(_logits(model, data[:, :-1]), data[:, 1:], None)
    inp = data[:, :-1]
    S = task.S
    qpos = task.qpos[0]
    qs, _ = query(task, inp)
    qtok = inp[:, qpos]
    lab = task.stream_labels(inp)
    rl = roles89(task, inp)
    hit = (ql.argmax(-1) == qt)
    cells = {}
    for b in range(inp.shape[0]):
        s_q, k_q = int(qs[b]), int(qtok[b])
        cur, nblk, idx, found = None, {s: -1 for s in range(S)}, 0, None
        tok, lb, isc, isk = inp[b].tolist(), lab[b].tolist(), rl["CTX"][b].tolist(), rl["KEY"][b].tolist()
        for pos in range(task.body_len):
            if isc[pos]:
                cur = lb[pos]
                nblk[cur] += 1
                idx = 0
            elif isk[pos]:
                if cur == s_q and tok[pos] == k_q:
                    found = (nblk[cur], idx)
                idx += 1
        key = f"{s_q}|{found[0]}|{found[1]}"
        c = cells.setdefault(key, [0, 0])
        c[0] += int(hit[b]); c[1] += 1
    return {k: dict(acc=v[0] / v[1], n=v[1]) for k, v in sorted(cells.items())}


def write_order_acc89(model, task, data):
    """explore_f_tasks.write_order_acc with the query's stream index from stream_labels."""
    inp, tgt = data[:, :-1], data[:, 1:]
    lg = _logits(model, inp)
    S = task.S
    qpos = task.qpos[0]
    pred = lg[:, qpos].argmax(-1)
    ans = tgt[:, qpos]
    qs, _ = query(task, inp)
    qk = inp[:, qpos]
    body = inp[:, :qpos - 1]
    lab = task.stream_labels(inp)[:, :qpos - 1]
    idx = torch.arange(body.shape[1]).expand_as(body)
    hit = body == qk[:, None]
    assert bool((hit.sum(1) == S).all())
    last_pos = torch.where(hit, idx, torch.full_like(idx, -1)).max(1).values
    last_stream = lab.gather(1, last_pos[:, None])[:, 0]
    last_val = inp.gather(1, (last_pos + 1)[:, None])[:, 0]
    is_last = last_stream == qs
    ok = pred == ans
    out = dict(n=int(ok.numel()), acc=float(ok.float().mean()),
               n_last=int(is_last.sum()), acc_last=float(ok[is_last].float().mean()) if is_last.any() else None,
               n_notlast=int((~is_last).sum()),
               acc_notlast=float(ok[~is_last].float().mean()) if (~is_last).any() else None,
               lastval_notlast=float((pred[~is_last] == last_val[~is_last]).float().mean()) if (~is_last).any() else None)
    out["acc_by_stream"] = [float(ok[qs == s].float().mean()) if (qs == s).any() else None for s in range(S)]
    return out


def selected(task_name, kind):
    """Per role, the mean count of positions a hinge selects in the task's 4096-sequence sample."""
    t = task_of(task_name)
    x = ft.freq_sample(t)
    kpos, _, _ = key_pos_ids(t, x)
    km = torch.zeros_like(x, dtype=torch.bool).scatter_(1, kpos, True)
    m = km if kind == "key" else hinge_mask(kind, x, task_name)
    rl = roles89(t, x)
    out = dict(total=float(m.sum(1).float().mean()), key_term=float((m & km).sum(1).float().mean()),
               key_term_of=float(km.sum(1).float().mean()))
    for r in ("CTX", "KEY", "VAL", "QUERY"):
        out[r.lower()] = float((m & rl[r]).sum(1).float().mean())
    if kind == "freq":
        out["types"], out["table"] = frequent(task_name)[0], frequent(task_name)[1][:20]
        out["roles"] = [ft.role_of(t, i) for i in out["types"]]
    return out


# ── run ──────────────────────────────────────────────────────────────────────
def run(p):
    """p: arm ("<task>.<ARM>" or ARM with p["task"]), seed, iters, off, eval_every."""
    arm, seed = p["arm"], p["seed"]
    if "." in arm:
        tname, arm = arm.split(".", 1)
    else:
        tname = p["task"]
    task = task_of(tname)
    kind, mem = ARMS[arm]
    a = dict(ceil_arm(task) if kind == "oracle" else ec.ARM_A, key=arm)
    holder = {}
    kw = dict(param_groups=dc.groups(holder, dc.LR))
    check = dc.switch(holder, 0)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), p.get("eval_every", tbo.EVAL_EVERY), task=task, stats_fn=stats_fn89,
                  grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, tname, on=not p.get("off"))), run_kw=kw, check=check,
                  keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=p["arm"], task=tname, recipe="slow", group_names=holder.get("names"),
               stream_acc=stream_acc89(m, task, ev), probe=probe_rh89(m, task, gc.probe_data(task)),
               acc_by_cell=acc_by_cell89(m, task, ev), write_order=write_order_acc89(m, task, ev))
    if isinstance(m, Term89):
        rec["term"] = {k: v for k, v in m.t_state.items() if k != "count"}
    if kind != "oracle":
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── records of earlier screens (a CPU-tolerant lookup, printed) ──────────────
def record(store, arm, seed):
    """(record, its CPU): the single record of arm|seed under the store's SHA, whatever the CPU it ran on."""
    with open(os.path.join(gc.G_DIR, store)) as f:
        st = json.load(f)
    pre = f"{arm}|{seed}|{st['meta']['sha'][:12]}|"
    hits = [(k, v) for k, v in st["runs"].items() if k.startswith(pre) and v.get("ok")]
    assert len(hits) == 1, (pre, [k for k, _ in hits])
    return hits[0][1], hits[0][0][len(pre):]


def _git(*a):
    return subprocess.check_output(["git", "-C", HERE] + list(a), stderr=subprocess.DEVNULL).decode().strip()


def _has(rev):
    try:
        _git("rev-parse", "--verify", "--quiet", rev + "^{commit}")
        return True
    except subprocess.CalledProcessError:
        return False


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks_static(p):
    """Everything that trains for less than one evaluation."""
    rows = []
    # (fix 2) the vendored module
    path = os.path.join(HERE, "explore_f_tasks.py")
    h = _git("hash-object", path)
    hf = _git("rev-parse", F_COMMIT + ":explore_f_tasks.py") if _has(F_COMMIT) else None
    rows.append((f"explore_f_tasks.py is F's {F_COMMIT} blob: git hash-object {h[:12]}; expected {F_BLOB[:12]}; "
                 f"git rev-parse {F_COMMIT}:explore_f_tasks.py {hf[:12] if hf else 'not in this clone'}",
                 h == F_BLOB and (hf is None or hf == F_BLOB)))
    data = open(path, "rb").read()
    pre = subprocess.check_output(["git", "-C", HERE, "cat-file", "blob", PREFIX_BLOB])
    rows.append((f"its first {PREFIX_LEN} bytes equal {PREFIX_COMMIT}'s blob {PREFIX_BLOB[:12]} (the lite module vendored "
                 f"before; F appended only): {len(pre)} bytes, prefix equal {data[:PREFIX_LEN] == pre}",
                 len(pre) == PREFIX_LEN and data[:PREFIX_LEN] == pre))
    for tname in POOLED:
        for nm, v in ft.check_pool_task(task_of(tname)):
            rows.append((f"{tname} (F's check_pool_task): {nm}", v))
    lite = task_of("lite")
    rows.append((f"tasks: lite {lite.key} vocab {lite.vocab}; pool {TASKS['pool'].key} vocab {TASKS['pool'].vocab}; allpool "
                 f"{TASKS['allpool'].key} vocab {TASKS['allpool'].vocab}; L 23, qpos [21] on all three",
                 lite.vocab == 22 and TASKS["pool"].vocab == 50 and TASKS["allpool"].vocab == 64
                 and all(t.L == 23 and t.qpos == [21] for t in TASKS.values())))
    # (fix 7) token ids only: the cut on the task's own sample; refusals
    for tname in TASKS:
        t = task_of(tname)
        x = ft.freq_sample(t)
        fr = frequent(tname)
        lr, _ = ft.largest_ratio_cut(x)
        raised = []
        for bad in (t.stream_labels(x), t.key_index(x) if is_pool(t) else x - t.S, x.float(), x[:, :-1],
                    t.make_batch(ft.FREQ_SAMPLE, torch.Generator().manual_seed(1))[0][:, :-1]):
            try:
                frequent(tname, bad)
                raised.append(False)
            except AssertionError:
                raised.append(True)
        rows.append((f"{tname}: frequent() is S87's freq_types on ft.freq_sample(task) = F's largest_ratio_cut "
                     f"({fr[0] == lr}): types {fr[0]} ({sorted(set(ft.role_of(t, i) for i in fr[0]))}); it raises when "
                     f"handed stream_labels(x), the key ids, a float tensor, a narrower tensor, another sample ({raised})",
                     fr[0] == lr and all(raised)))
    rows.append(("lite: the frequent types are S87's {0, 1, 2, 3, 4, 5}", sorted(frequent("lite")[0]) == [0, 1, 2, 3, 4, 5]))
    sig = [list(inspect.signature(f_).parameters) for f_ in (repeat_mask, s87.freq_types)]
    xr = torch.tensor([[0, 5, 7, 5, 1, 9, 0]])
    rows.append((f"repeat_mask(tokens) and freq_types(tokens) take one parameter ({sig}); repeat_mask by hand "
                 f"{repeat_mask(xr).tolist()}", sig == [["tokens"], ["tokens"]]
                 and repeat_mask(xr).tolist() == [[True, True, False, True, False, False, True]]))
    xl = ft.freq_sample(lite)
    same_rep = all(torch.equal(repeat_mask(ft.freq_sample(task_of(n))), repeat_mask(xl)) for n in POOLED)
    rows.append((f"the repeat hinge's positions are identical on the three tasks' samples (same generator state): {same_rep}",
                 same_rep))
    for tname in TASKS:
        for kind in ("key", "freq", "repeat"):
            s = selected(tname, kind)
            extra = f"; types {s['types']} {sorted(set(s['roles']))}" if kind == "freq" else ""
            rows.append((f"{tname}: the {kind} hinge selects per sequence {s['total']:.2f} positions (CTX {s['ctx']:.2f}, KEY "
                         f"{s['key']:.2f}, VAL {s['val']:.2f}, QUERY {s['query']:.2f}); of the key term's "
                         f"{s['key_term_of']:.0f}: {s['key_term']:.2f}{extra}", True))
    sp, sa = selected("pool", "freq"), selected("allpool", "freq")
    rows.append((f"pool: the frequency hinge is a CTX hinge (roles {sorted(set(sp['roles']))}, key-term positions "
                 f"{sp['key_term']:.2f}); allpool: a value hinge (roles {sorted(set(sa['roles']))}, key-term positions "
                 f"{sa['key_term']:.2f})", set(sp["roles"]) == {"CTX"} and set(sa["roles"]) == {"VAL"}
                 and sp["key_term"] == 0 and sa["key_term"] == 0))
    # (fix 3) the key term: equal to dc.key_eta on lite bit for bit; on the new tasks its positions are task.key_mask's 8
    m = make_model("REG3_KEYHINGE_D", 2842, "lite")
    x = lite.make_batch(64, torch.Generator().manual_seed(5))[0][:, :-1]
    with torch.no_grad():
        gs = s56.all_gates(m, x)[0]
    rows.append(("lite: the task-aware key term equals explore_g_devices_child.key_eta bit for bit (3 layers' gates, 64 seqs)",
                 torch.equal(key_eta89(gs, x, lite), dc.key_eta(gs, x, lite))))
    for tname in POOLED:
        t = task_of(tname)
        x = t.make_batch(64, torch.Generator().manual_seed(5))[0][:, :-1]
        kpos, kid, G = key_pos_ids(t, x)
        km = torch.zeros_like(x, dtype=torch.bool).scatter_(1, kpos, True)
        body = torch.zeros_like(km)
        body[:, :t.body_len] = True
        rows.append((f"{tname}: the key term's positions are task.key_mask's {int(km[0].sum())} body positions per sequence, "
                     f"ids in [0, {G}) = task.key_index there",
                     torch.equal(km, t.key_mask(x) & body) and bool((kid >= 0).all() and (kid < G).all())
                     and torch.equal(kid, (x - t.key0).gather(1, kpos))))
    # the task-aware statistics equal the lite ones on lite (untrained models)
    probe = probe_batch(lite, 2842)
    ev = tbo.eval_batch(lite)
    for arm in ("REG3_D", "BASELINE_D"):
        m = make_model(arm, 2842, "lite")
        eq = dict(layer_stats=layer_stats89(m, lite, probe) == rc.layer_stats(m, lite, probe),
                  write_rate=write_rate89(m, lite, probe) == rc.write_rate(m, lite, probe),
                  stream_acc=stream_acc89(m, lite, ev) == gc.stream_acc(m, lite, ev),
                  acc_by_cell=acc_by_cell89(m, lite, ev) == rc.acc_by_cell(m, lite, ev),
                  write_order=write_order_acc89(m, lite, ev) == ft.write_order_acc(m, lite, ev),
                  roles=all(torch.equal(a_, b_) for a_, b_ in zip(roles89(lite, probe[0]).values(),
                                                                 rc.roles(lite, probe[0]).values())))
        rows.append((f"lite: the task-aware statistics equal the lite functions on {arm}|2842 untrained: {eq}", all(eq.values())))
    # on pool (stream index = CTX id there) the statistics equal F's rand_header-style ones read with the pool's roles
    t = task_of("pool")
    pr = probe_batch(t, 2842)
    m = make_model("REG3_D", 2842, "pool")
    a_, b_ = ft.pool_routing_stats(m, t, pr), {k: v for k, v in layer_stats89(m, t, pr)["L3"].items() if k.startswith("eta")}
    rows.append(("pool: layer_stats89's layer-3 eta^2 equal pool_routing_stats' (the shared gate's last layer is the returned "
                 "gate)", all(abs(a_[k] - b_[k]) < 1e-12 for k in b_)))
    # the perfect gate: = explore_g_rhl_child's ORACLE_ALL_D on lite and pool; task.perfect_gate at every layer
    for tname in TASKS:
        t = task_of(tname)
        x = t.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
        m = make_model("ORACLE_D", 2840, tname).eval()
        with torch.no_grad():
            lg, _, gr, gw = m(x)
        pg = t.perfect_gate(x) if is_pool(t) else perfect_gate_general(x, t.ctx_tokens)
        ok = torch.equal(gr, pg) and torch.equal(gw, pg) and gr.shape[-1] == 2 and torch.equal(m.attn._gates[0], pg)
        extra = ""
        if tname != "allpool":
            ref = rc.make_model("ORACLE_ALL_D", 2840, t).eval()
            with torch.no_grad():
                lr_, _, grr, _ = ref(x)
            same_p = all(torch.equal(u, w) for u, w in zip(m.state_dict().values(), ref.state_dict().values()))
            extra = (f"; = explore_g_rhl_child's ORACLE_ALL_D (parameters {same_p}, logits {torch.equal(lg, lr_)}, gates "
                     f"{torch.equal(gr, grr)})")
            ok = ok and same_p and torch.equal(lg, lr_) and torch.equal(gr, grr)
        rows.append((f"{tname}: ORACLE_D's gate is task.perfect_gate (k = 2), passed to the delta memory{extra}", ok))
    # (fix 3) the untrained perfect gate reads ROUTED* on each task
    for tname in TASKS:
        t = task_of(tname)
        m = make_model("ORACLE_D", 2840, tname)
        st = stats_fn89(m, t, probe_batch(t, 2840), "end")
        rows.append((f"{tname}: the untrained perfect gate reads ROUTED*: eta_key_by_stream {st['eta_key_by_stream']:.4f}, "
                     f"margin {st['margin']:.4f}, eta_val_by_stream {st['eta_val_by_stream']:.4f}",
                     st["eta_key_by_stream"] > 0.999 and st["margin"] >= 0.9))
    return rows


def lr_rows(p):
    """The recipe: SLOW for every arm, the perfect gate's gate group empty (lrs at updates 1 / 2400 / 2401)."""
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    rows = []
    for arm in ("ORACLE_D", "REG3_FREQHINGE_D"):
        lrs, cnt = {}, [0]

        def pre(opt, args, kwargs):
            cnt[0] += 1
            if cnt[0] in (1, 2400, 2401):
                lrs[cnt[0]] = [(len(g["params"]), g["lr"]) for g in opt.param_groups]
        hk = register_optimizer_step_pre_hook(pre)
        try:
            run(dict(arm=arm, seed=2842, task="pool", iters=2401))
        finally:
            hk.remove()
        ok = (all(k in lrs for k in (1, 2400, 2401)) and lrs[1][0][1] == dc.LR and lrs[1][1][1] == dc.LR_SLOW
              and lrs[2400][1][1] == dc.LR_SLOW and lrs[2401][1][1] == dc.LR and (arm != "ORACLE_D" or lrs[1][0][0] == 0))
        rows.append([f"pool: {arm}'s (n params, lr) per group at updates 1 / 2400 / 2401: {lrs}", bool(ok)])
    return rows


def checks_repro(p):
    """An earlier record reproduces bit for bit through 2400 through this screen's run() on lite."""
    arm, seed, store = p["arm"], p["seed"], p.get("store", "s87_results.json")
    ref, cpu = record(store, arm, seed)
    r = run(dict(arm=arm, seed=seed, iters=2400, task="lite"))
    ok = s87.trunc(r) == s87.trunc(ref)
    return [[f"{store[:3].upper()}'s {arm}|{seed} (recorded on {cpu}; here {ec.cpu_model()}) reproduces bit for bit through "
             f"2400 through S89's run() on lite: curve {r['curve']}", bool(ok)]]


def checks_off(p):
    """On a task: each hinge with its switch off equals REG3_D through 2400 bit for bit."""
    tname = p["task"]
    seed = p.get("seed", 2842)
    ref = run(dict(arm="REG3_D", seed=seed, iters=2400, task=tname))
    rows = []
    for arm in ("REG3_KEYHINGE_D", "REG3_FREQHINGE_D", "REG3_REPEATHINGE_D"):
        r = run(dict(arm=arm, seed=seed, iters=2400, task=tname, off=True))
        rows.append([f"{tname}: {arm} with its switch off equals REG3_D|{seed} through 2400 bit for bit (curve {r['curve']})",
                     s87.trunc(r) == s87.trunc(ref)])
    return rows


def checks_on(p):
    """On a task: every arm runs end to end (20 updates, an evaluation every 10: the statistics at 10, 20 and 'end' and
    every post-run record); each term's gradient reaches the gate's parameters only; the schedule over training forwards."""
    tname = p["task"]
    t = task_of(tname)
    rows = []
    for arm in ("ORACLE_D", "BASELINE_D", "REG3_D", "REG3_KEYHINGE_D", "REG3_FREQHINGE_D", "REG3_REPEATHINGE_D"):
        r = run(dict(arm=arm, seed=2842, task=tname, iters=20, eval_every=10))
        e = r.get("end", {})
        need = ("margin", "eta_key_by_stream", "eta_key_by_key", "spread", "layers") + (() if arm == "ORACLE_D" else ("beta",))
        fired = r.get("term", {}).get("fired")
        rows.append([f"{tname}: {arm} runs 20 updates end to end (ok {r.get('ok')}; end statistics "
                     f"{sorted(k for k in need if k in e)}; stream_acc {r.get('stream_acc')}; cells {len(r.get('acc_by_cell', {}))}; "
                     f"term fired {fired}/20)",
                     bool(r.get("ok")) and all(k in e for k in need) and len(r["stream_acc"]) == 2
                     and len(r["acc_by_cell"]) > 0 and (fired is None or fired > 0)])
    x = t.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
    stack = ("encoder", "encoder_v", "decoder", "lm_head")
    for arm in ("REG3_KEYHINGE_D", "REG3_FREQHINGE_D", "REG3_REPEATHINGE_D"):
        m = make_model(arm, 2842, tname)
        m.train()
        m.t_state["count"] = False
        gates = s56.live_gates(m, x)
        if m.t_kind == "key":
            v = key_eta89(gates, x, t)
        else:
            mask = hinge_mask(m.t_kind, x, tname)
            V = int(x.max()) + 1
            v = torch.stack([g18.eta2_pool(g[mask], x[mask], V) for g in gates])
        names = [n for n, q in m.named_parameters() if q.requires_grad]
        gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
        nz = {n: (0.0 if g_ is None else float(g_.abs().max())) for n, g_ in zip(names, gr)}
        reach = sorted(n for n, v_ in nz.items() if v_ > 0)
        rows.append([f"{tname}: {arm}'s eta^2 gradient reaches {reach}; the stack {[n for n in stack if nz.get(n, 0) > 0] or 'none'}; "
                     f"the memory {[n for n in reach if n.startswith('attn.')] or 'none'}",
                     nz.get("W_g2", 0) > 0 and nz.get("A_r", 0) > 0 and all(nz.get(n, 0) == 0 for n in stack)
                     and not any(n.startswith("attn.") for n in reach)])
    m = make_model("REG3_REPEATHINGE_D", 2842, tname)
    m.train()
    for _ in range(WINDOW + 2):
        out = m(x[:1])[0]
        out.sum().backward()
        m.zero_grad(set_to_none=True)
    w = m.t_state["weights"]
    rows.append([f"{tname}: REG3_REPEATHINGE_D schedule over training forwards {w}",
                 w["1"] == 1.0 and w[str(WINDOW)] == 1.0 and w[str(WINDOW + 1)] == 0.0 and w[str(WINDOW + 2)] == 0.0])
    return rows


def checks(p):
    return [[n, bool(v)] for n, v in checks_static(p)]
