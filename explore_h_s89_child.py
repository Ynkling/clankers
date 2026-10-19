#!/usr/bin/env python
"""
explore_h_s89_child.py — EXPLORATORY, not a result. Session H's S89 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S86's and S87a's. Parent: explore_h_s89 (readings in its
docstring).

THE TASK (PoolTask): test_window_gate Part C's task (test_stream_recipe.TASK48 = task_for(4, 8): S = 8, P = 4, n_vals =
16, n_q = 1, the grouped layout, length 99), subclassed, with the P = 4 keys of each sequence drawn without replacement
from a pool of K key types (K = 32 in the screen). make_batch draws, in the main line's order and shapes with P replaced
by K where a key's identity is drawn:
  vals   (B, K, n_vals) uniform -> argsort -> first S   (S distinct values for every pool key; the main line: (B, P, ...))
  korder (B, K) uniform -> argsort -> first P            (the sequence's P keys, without replacement, in body order)
  sorder (B, P, S) uniform -> argsort                    (stream order within each key's group; unchanged)
  q      (B, S*P) uniform -> argsort -> first n_q        (the query's (stream, key slot); the key slot indexes the
                                                          sequence's keys in increasing id, so at K = P it is the key id)
Tokens: CTX_s = s (0..S-1); KEY_i = S + i (i < K); VAL_j = S + K + j. vocab = S + K + n_vals. At K = P (= 4) every
draw has the main line's shape and order and every token its id, so the batches equal the main line's bit for bit (CHECKed
on the training, evaluation and probe generators). role_masks reads K. Every other method is the main line's: the key
positions are still 3j+1 (the grouped layout); routing_k labels keys by token id - S (0..K-1).
Inherited statistics that index key ids S..S+P-1 (gate_stats_k's key_sep, gate_stats' key_part) cover 4 of the 32 pool
keys at K = 32: descriptive, used by no outcome; printed as such.

THE TRIGGER (make_trigger_s89): S86's explore_h_allmass_child.make_trigger_target line for line (the same read gate on the
same 64-sequence probe, read once per check), with a third target pair computed at every check beside the other two:
  the repeat pair (c*, c0 = the largest / smallest mean read-gate mass over the probe's input positions whose token type
  occurs more than once in that sequence's input; repeat_mask, token ids only).
target "key" copies onto the key-position pair (KEYMASS: positions 3j+1), "all" onto the all-position pair, "repeat"
onto the repeat pair. All three pairs and their labels (key_ok, all_ok, rep_ok: c* holds >= 2 streams and c0 none on
routing_k's stream -> channel map at VAL positions, before the operation) are recorded at every check.
  BASE        target key, exact copy (explore_h_ops.copy_exact = S61's copy_nonoise_op).
  ALLMASS     target all (S86 / S87a's arm).
  REPEATMASS  target repeat.
  NOSPLIT     no trigger (LOCAL3 + SLOW alone; = WIN3_SLOW_D8's recipe).
  ORACLE      perfect gate ceil8_D8 + conv (test_stream_recipe's ceiling8k16, curriculum off), on the pool task.
  MAINSPLIT   (CHECK only) target key with the main line's split op (copy + noise + Adam reset: S36's split_op =
              test_window_gate's), for the WIN3_SPLIT_D8|540 comparison.
"""

import ast
import collections
import inspect
import textwrap

import torch

import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_muon_scale_child as s33c
import explore_h_allmass_child as h86c
import explore_h_copy2x2_child as h61c
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import explore_window_d8_child as s48c
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_channel_binding import eval_batch
from test_multilayer_binding import probe_batch

CFG = "b"
TOTAL = 43200
K_POOL = 32
T48 = tsr.TASK48
BASE = dict(h86c.BASE)                     # S84's constants, target key, probe as source, no guard
ARM = {"BASE": dict(), "ALLMASS": dict(target="all"), "REPEATMASS": dict(target="repeat"), "NOSPLIT": dict(nosplit=True),
       "ORACLE": dict(oracle=True), "MAINSPLIT": dict(op="split")}


class PoolTask(type(T48)):
    def __init__(self, K):
        super().__init__(T48.P, S=T48.S, n_vals=T48.n_vals, n_q=T48.n_q)
        assert T48.layout == "grouped" and K >= T48.P
        self.K = K
        self.key = T48.key if K == T48.P else f"{T48.key}K{K}"
        self.label = T48.label + ("" if K == T48.P else f", keys drawn from a pool of {K} (S89)")
        self.vocab = self.S + K + self.n_vals

    def make_batch(self, B, gen):
        S, P, K, nv, nq = self.S, self.P, self.K, self.n_vals, self.n_q
        vals = torch.rand(B, K, nv, generator=gen).argsort(-1)[..., :S]    # (B,K,S) distinct per pool key
        korder = torch.rand(B, K, generator=gen).argsort(-1)[:, :P]         # the sequence's keys, no replacement
        sorder = torch.rand(B, P, S, generator=gen).argsort(-1)             # stream order
        vg = vals.gather(1, korder.unsqueeze(-1).expand(B, P, S)).gather(2, sorder)
        keys = (S + korder).unsqueeze(-1).expand(B, P, S)
        body = torch.stack([sorder, keys, S + K + vg], dim=-1).reshape(B, -1)
        q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :nq]         # no replacement
        qs, qi = q // P, q % P
        qk = korder.sort(-1).values.gather(1, qi)                           # = qi at K = P
        qv = vals.reshape(B, K * S).gather(1, qk * S + qs)
        queries = torch.stack([qs, S + qk, S + K + qv], dim=-1).reshape(B, -1)
        return torch.cat([body, queries], dim=1), qs

    def role_masks(self, x):
        S, K = self.S, self.K
        return {'ctx': x < S, 'key': (x >= S) & (x < S + K), 'val': x >= S + K}


def repeat_mask(x):
    """(B, T) bool: the positions of x whose token type occurs more than once in that row of x. Token ids only."""
    return (x.unsqueeze(2) == x.unsqueeze(1)).sum(-1) > 1


def select_repeat(gr, x):
    rm = gr[repeat_mask(x)].mean(0)
    return int(rm.argmax()), int(rm.argmin()), rm


def overlap(task, x):
    """Counts over the input rows x (B, T = L - 1): positions in the repeat set by role (body CTX / KEY / VAL, the query's
    CTX and KEY), and the key positions (3j+1) in it."""
    n = task.S * task.P
    m = repeat_mask(x)
    j = torch.arange(n)
    pos = dict(ctx=3 * j, key=3 * j + 1, val=3 * j + 2, q_ctx=torch.tensor([3 * n]), q_key=torch.tensor([3 * n + 1]))
    B = x.shape[0]
    out = dict(sequences=B, positions=int(m.numel()), repeat=int(m.sum()), key_positions=B * n)
    for r, p in pos.items():
        out[f"repeat_{r}"] = int(m[:, p].sum())
        out[f"all_{r}"] = int(B * len(p))
    out["key_in_repeat"] = out["repeat_key"]
    out["repeat_not_key"] = out["repeat"] - out["repeat_key"]
    return out


def make_trigger_s89(task, seed, info, opt_of, last, op, every, ref_at, first, gap, cap, probe_n, rise, thr, guard, source,
                     target):
    data = s36c.probe_data(task)[:probe_n]
    edata = eval_batch(task) if source == "eval" else None
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % every or step < ref_at or step > last:
            return
        acc, _ = tbo.evaluate(m, task, edata if source == "eval" else data)
        gr = s37c.read_gate(m, data)
        (kc, k0, km), (ac, a0, am) = s37c.select(gr, task.S, task.P)
        rc, r0, rm = select_repeat(gr, data[:, :-1])
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, source=source, target=target, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False,
                   rep_masses=[round(float(v), 5) for v in rm], rep_cs=rc, rep_c0=r0, rep_ok=s37c.targeting(cmap, rc, r0))
        eligible = step >= first and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        guarded = guard is not None and acc < guard
        capped = st["n"] >= cap or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and guarded:
            row["blocked"] = True
            row["blocked_by"] = "guard"
        elif eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= cap else "gap"
        elif eligible:
            cs, c0 = {"key": (kc, k0), "all": (ac, a0), "repeat": (rc, r0)}[target]
            res = op(m, opt_of(), cs, c0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, op_cs=cs, op_c0=c0, op_ok=s37c.targeting(cmap, cs, c0),
                       map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in s37c.select(s37c.read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def pool_rc(task, a, rc):
    """The run path's run_one keywords with the task and the model builder on `task` (test_stream_recipe.builder_for's
    model_fn, with padded_task for the perfect gate's ctx_pad), then the recipe rc (None = none)."""
    bt = tsr.padded_task(task, a["ctx_pad"]) if a.get("ctx_pad") else task

    def rc2(kw):
        kw = dict(kw, task=task, builder=lambda aa, seed: tsr.model_fn(bt, aa, seed))
        return kw if rc is None else rc(kw)
    return rc2


def run(p):
    """p: arm, seed, iters; K (pool size, default 32); CHECK knobs: thr, total."""
    arm = ARM[p["arm"]]
    K = p.get("K", K_POOL)
    task = PoolTask(K)
    if arm.get("oracle"):
        a = s33c.cfg_arm(CFG, "CEIL")
        rec = b16.path(CFG, a, p["seed"], p["iters"], pool_rc(task, a, None), eval_every=p.get("eval_every"))
        return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="oracle", pool=K)
    a = b16.arm_of(CFG)
    cfg = dict(BASE, target=arm.get("target", "key"))
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    trig = not arm.get("nosplit")
    if trig:
        op = s36c.split_op if arm.get("op") == "split" else ops.copy_exact
        rc = s36c.with_trigger(rc, make_trigger_s89(task, p["seed"], info, lambda: holder["opts"][0],
                                                    p.get("total", TOTAL) - 2400, op, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, a, p["seed"], p["iters"], pool_rc(task, a, rc), eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder),
                     trig_cfg=cfg if trig else None, checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]), pool=K,
                     overlap=overlap(task, s36c.probe_data(task)[:BASE["probe_n"]][:, :-1]))
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _body(f):
    t = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
    return [ast.dump(s) for s in t.body]


def describe(p):
    """The pool task's structure and statistics over n sequences (no model)."""
    K = p.get("K", K_POOL)
    t = PoolTask(K)
    n = p.get("n", 4096)
    x, qs = t.make_batch(n, torch.Generator().manual_seed(p.get("gen", 89_000)))
    S, P, nb = t.S, t.P, t.S * t.P
    j = torch.arange(nb)
    ctx, key, val = x[:, 3 * j], x[:, 3 * j + 1], x[:, 3 * j + 2]
    cnt = torch.bincount(x[:, :-1].reshape(-1), minlength=t.vocab).double() / x[:, :-1].numel()
    seq_cnt = torch.stack([torch.bincount(r, minlength=t.vocab) for r in x[:, :-1]])            # (n, vocab)
    role = lambda v: "ctx" if v < S else ("key" if v < S + K else "val")                         # noqa: E731
    by_role = collections.defaultdict(list)
    for v in range(t.vocab):
        by_role[role(v)].append(float(cnt[v]))
    per_seq = {r: [] for r in ("ctx", "key", "val")}
    for v in range(t.vocab):
        c = seq_cnt[:, v]
        per_seq[role(v)].append(float(c[c > 0].float().mean()) if (c > 0).any() else 0.0)
    keys_ok = all(len(set(r.tolist())) == P for r in key[:, ::S])
    groups_ok = bool((key.reshape(n, P, S) == key.reshape(n, P, S)[..., :1]).all())
    streams_ok = bool((ctx.reshape(n, P, S).sort(-1).values == torch.arange(S)).all())
    vals_ok = bool(all(len(set(r.tolist())) == S for r in val.reshape(n * P, S)))
    qpos = 3 * nb
    qk, qv = x[:, qpos + 1], x[:, qpos + 2]
    hit = (ctx == x[:, qpos:qpos + 1]) & (key == qk[:, None])
    answered = bool((hit.sum(1) == 1).all()) and bool((val[hit] == qv).all())
    ov = overlap(t, x[:, :-1])
    return dict(K=K, vocab=t.vocab, L=t.L, n=n,
                freq=dict(ctx_max=max(by_role["ctx"]), key_max=max(by_role["key"]), key_min=min(by_role["key"]),
                          val_max=max(by_role["val"]), val_min=min(by_role["val"])),
                within_seq=dict(ctx=sum(per_seq["ctx"]) / S, key=sum(per_seq["key"]) / K,
                                val=sum(per_seq["val"]) / t.n_vals),
                keys_distinct=keys_ok, key_groups=groups_ok, streams_per_group=streams_ok, values_distinct=vals_ok,
                answered=answered, ids_in_vocab=bool((x >= 0).all() and (x < t.vocab).all()), overlap=ov)


def checks(p):
    rows = []
    t4, t32 = PoolTask(4), PoolTask(K_POOL)
    # (1) pool size 4 = the main line's batches, bit for bit
    gens = [("training", 540 + 10_000, 32), ("evaluation", None, None), ("trigger probe", None, None),
            ("routing probe 540", None, None), ("seed 2902", 2902, 64), ("seed 1", 1, 7)]
    same = []
    for name, sd, B in gens:
        if name == "evaluation":
            a, b = eval_batch(t4), eval_batch(T48)
        elif name == "trigger probe":
            a, b = s36c.probe_data(t4), s36c.probe_data(T48)
        elif name == "routing probe 540":
            a, b = probe_batch(t4, 540)[0], probe_batch(T48, 540)[0]
        else:
            ga, gb = torch.Generator().manual_seed(sd), torch.Generator().manual_seed(sd)
            xa, qa = t4.make_batch(B, ga)
            xb, qb = T48.make_batch(B, gb)
            a, b = torch.cat([xa, qa.reshape(B, -1)], 1), torch.cat([xb, qb.reshape(B, -1)], 1)
            a = torch.cat([a, torch.rand(1, generator=ga).expand(a.shape[0], 1).mul(1e6).long()], 1)  # generator state after
            b = torch.cat([b, torch.rand(1, generator=gb).expand(b.shape[0], 1).mul(1e6).long()], 1)
        same.append((name, torch.equal(a, b)))
    attrs = all(getattr(t4, k) == getattr(T48, k) for k in ("S", "P", "n_vals", "n_q", "vocab", "L", "T", "qpos", "ctx_tokens",
                                                             "key", "layout"))
    rows.append((f"PoolTask(4) batches equal test_stream_recipe.TASK48's bit for bit ({same}; generator state after the draw "
                 f"included) and its attributes (S, P, n_vals, n_q, vocab, L, T, qpos, ctx_tokens, key, layout) equal: {attrs}",
                 all(v for _, v in same) and attrs))
    # (2) the pool-32 task's structure
    d = describe(dict(K=K_POOL, n=4096))
    ok = (d["keys_distinct"] and d["key_groups"] and d["streams_per_group"] and d["values_distinct"] and d["answered"]
          and d["ids_in_vocab"] and d["vocab"] == 8 + K_POOL + 16 and d["L"] == T48.L)
    rows.append((f"PoolTask(32): 4 distinct keys per sequence from 32, each in one group of 8 triples with the 8 streams once, 8 "
                 f"distinct values per key, the query answered by its (stream, key)'s value, ids in [0, {d['vocab']}), length "
                 f"{d['L']}; frequencies over 4096 sequences {d['freq']}", ok))
    # (3) the repeat statistic reads token ids only
    x = s36c.probe_data(t32)[:, :-1]
    g = torch.Generator().manual_seed(3)
    perm = torch.randperm(t32.vocab, generator=g)
    rm = repeat_mask(x)
    ref = torch.tensor([[collections.Counter(r.tolist())[v] > 1 for v in r.tolist()] for r in x])
    inv = torch.equal(rm, repeat_mask(perm[x])) and torch.equal(rm, ref)
    sig = list(inspect.signature(repeat_mask).parameters) == ["x"] and list(inspect.signature(select_repeat).parameters) == ["gr", "x"]
    rows.append((f"repeat_mask is a function of the token ids alone (one argument; invariant under a random permutation of the "
                 f"vocabulary; equals a per-row Counter reference); select_repeat takes the read gate and the ids only: "
                 f"{inv}, {sig}", inv and sig))
    # (4) the exact copy is S61's copy_nonoise_op
    src = _body(ops.copy_exact) == _body(h61c.copy_nonoise_op)
    a = b16.arm_of(CFG)
    res = []
    for f in (ops.copy_exact, h61c.copy_nonoise_op):
        m = s48c._model(a, 1320)
        o = torch.optim.Adam(m.parameters(), lr=1e-3)
        s48c._one_step(m, t4, o)
        r = f(m, o, 3, 7, 12_000_000 + 1000 * 1320)
        res.append((m.W_g.detach().clone(), {k: v.clone() for k, v in o.state[m.W_g].items() if torch.is_tensor(v)}, r))
    beh = torch.equal(res[0][0], res[1][0]) and res[0][1].keys() == res[1][1].keys() and \
        all(torch.equal(res[0][1][k], res[1][1][k]) for k in res[0][1]) and res[0][2] == res[1][2]
    rows.append((f"explore_h_ops.copy_exact is S61's copy_nonoise_op (explore_h_copy2x2_child) statement for statement "
                 f"(syntax trees of the bodies equal: {src}) and bit for bit (W_g, its Adam state, the returned record: {beh})",
                 src and beh))
    # (5) the trigger's mechanics on S48's harness (pool 4 = the main line's task; threshold above 1)
    steps = list(range(1200, 40801, 1200))
    c0 = {k: v for k, v in BASE.items() if k != "target"}
    seed = 2902
    mk = lambda tgt: (lambda info, oo: make_trigger_s89(t4, seed, info, oo, 40800, ops.copy_exact,    # noqa: E731
                                                        **dict(c0, thr=1.01, rise=1.0, target=tgt)))
    keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
            "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
            "row_dist_after", "op_cs", "op_c0", "op_ok")
    for tgt in ("key", "all"):
        mine, m1, _, _ = s48c._trigger_rows(a, t4, seed, mk(tgt), steps)
        theirs, m2, _, _ = s48c._trigger_rows(a, t4, seed, lambda info, oo: h86c.make_trigger_target(
            t4, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0, target=tgt)), steps)
        eq = len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys}
                                              for r, q in zip(mine, theirs)) and torch.equal(m1.W_g.detach(), m2.W_g.detach())
        rows.append((f"target {tgt} equals S86's make_trigger_target row for row (fired at {[r['step'] for r in mine if r['fired']]})",
                     eq))
    rp, _, _, _ = s48c._trigger_rows(a, t4, seed, mk("repeat"), steps)
    f = [r for r in rp if r["fired"]]
    ok = all(r["op_cs"] == r["rep_cs"] and r["op_c0"] == r["rep_c0"] for r in f) and [r["step"] for r in f] == [4800, 9600, 14400]
    rows.append((f"target repeat copies onto the repeat pair at every firing ({[(r['step'], r['rep_cs'], r['rep_c0'], r['key_cs'], r['key_c0'], r['all_cs'], r['all_c0']) for r in f]}: "
                 f"(update, repeat c*, c0, key c*, c0, all c*, c0))", ok))
    return [[n, bool(v)] for n, v in rows]
