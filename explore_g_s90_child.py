#!/usr/bin/env python
"""
explore_g_s90_child.py — EXPLORATORY, not a result. CHECKs of Session G's screen S90 (explore_g_s90). S90's runs go
through explore_g_s89_child.run on the lite task unchanged (its Hebbian REG3 and REG3_REPEATHINGE arms); this module only
CHECKs, 1 torch thread: S87's ORACLE_ALL|2700 through S87's own path, the repeat set on lite in counts, the switch-off and
the term's gradient on lite at seed 2700.
"""

import torch

import explore_common as ec
import explore_f_tasks as ft
import explore_g_register_child as s56
import explore_g_s87_child as s87
import explore_g_s89_child as c89


def check_oracle(p):
    ref, cpu = c89.record("s87_results.json", "ORACLE_ALL", 2700)
    r = s87.run(dict(arm="ORACLE_ALL", seed=2700, iters=2400))
    return [[f"S87's ORACLE_ALL|2700 (on {cpu}; here {ec.cpu_model()}) reproduces bit for bit through 2400 through S87's "
             f"path: curve {r['curve']}", s87.trunc(r) == s87.trunc(ref)]]


def check_lite(p):
    rows = []
    t = c89.task_of("lite")
    s = c89.selected("lite", "repeat")
    rows.append([f"lite: the repeat set per sequence (4096 sequences, Generator(87000)): {s['total']:.2f} positions = CTX "
                 f"{s['ctx']:.2f} + KEY {s['key']:.2f} + VAL {s['val']:.2f} + QUERY {s['query']:.2f} (the query's CTX and "
                 f"KEY); the key term's {s['key_term_of']:.0f} positions inside it: {s['key_term']:.2f}",
                 s["key_term"] == s["key_term_of"] == 8.0 and abs(s["ctx"] + s["key"] + s["val"] + s["query"] - s["total"]) < 1e-6])
    # the key term's grouping is inside the repeat hinge's: at the key positions the repeat hinge groups by token id,
    # which on lite is the key id (token - S), the key term's grouping
    x = ft.freq_sample(t)
    kpos, kid, _ = c89.key_pos_ids(t, x)
    rm_ = c89.repeat_mask(x)
    rows.append(["lite: every key-term position is a repeat position, and there the repeat hinge's group (token id) is the "
                 "key term's (token - S) one to one", bool(rm_.gather(1, kpos).all())
                 and torch.equal(x.gather(1, kpos) - t.S, kid)])
    r0 = c89.run(dict(arm="REG3", seed=2700, iters=2400, task="lite"))
    r1 = c89.run(dict(arm="REG3_REPEATHINGE", seed=2700, iters=2400, task="lite", off=True))
    rows.append([f"lite: REG3_REPEATHINGE with its switch off equals REG3|2700 through 2400 bit for bit (curve {r1['curve']})",
                 s87.trunc(r1) == s87.trunc(r0)])
    m = c89.make_model("REG3_REPEATHINGE", 2700, "lite")
    m.train()
    m.t_state["count"] = False
    xb = x[:32]
    gates = s56.live_gates(m, xb)
    import explore_b18_gates as g18
    mask = c89.hinge_mask("repeat", xb, "lite")
    v = torch.stack([g18.eta2_pool(g[mask], xb[mask], int(xb.max()) + 1) for g in gates])
    names = [n for n, q in m.named_parameters() if q.requires_grad]
    gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
    nz = {n: (0.0 if g_ is None else float(g_.abs().max())) for n, g_ in zip(names, gr)}
    stack = ("encoder", "encoder_v", "decoder", "lm_head")
    rows.append([f"lite: the repeat term's gradient reaches {sorted(n for n, v_ in nz.items() if v_ > 0)}; the stack "
                 f"{[n for n in stack if nz.get(n, 0) > 0] or 'none'}",
                 nz.get("W_g2", 0) > 0 and all(nz.get(n, 0) == 0 for n in stack)])
    return rows
