#!/usr/bin/env python
"""
explore_h_classes_child.py — EXPLORATORY, not a result. Session H's helper that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e): failure classes under v1 (test_router_layout.fail_class,
test_scale_axes.fail_class_k, test_stream_recipe.outcome) and v2 (explore_h_fail_class_v2 = fail_class_v2.py at ba8901a,
copied unchanged: CHECK on the git blob SHA), from saved records' "end" statistics (no runs); and the CHECK that
explore_h_ops' operations equal S36's split_op and S61's copy_nonoise_op bit for bit.
"""

import torch

import explore_h_fail_class_v2 as v2
import explore_h_ops as ops
import explore_split_child as s36c
import explore_h_copy2x2_child as s61c
import explore_window_d8_child as s48c
import explore_b16_main as b16
from test_router_layout import fail_class, MARGIN_PARTIAL, ETA_SPLIT
from test_scale_axes import fail_class_k
import test_stream_recipe as tsr


def classify(p):
    """p: {"runs": {key: {"kind": "k2" | "multi" | "outcome", "end": ..., "transition": ..., (outcome: the record)}}}."""
    out = {}
    for key, r in p["runs"].items():
        if r["kind"] == "k2":
            a, b = fail_class(r), v2.fail_class_v2(r)
            e = r["end"]
            kk, mg = e["eta_key_by_key"], e["margin"]
        elif r["kind"] == "multi":
            a, b = fail_class_k(r), v2.fail_class_k_v2(r)
            e = r["end"]
            kk, mg = e["etak_key_by_key"], e["margin"]
        else:
            a, b = tsr.outcome(r), v2.outcome_v2(r, tsr.routed)
            e = r["end"]
            kk, mg = e["etak_key_by_key"], e["margin"]
        out[key] = dict(v1=a, v2=b, eta_key=kk, margin=mg,
                        predicted_differ=bool(kk >= ETA_SPLIT and mg >= MARGIN_PARTIAL))
    return out


def ops_check(p):
    rows = []
    a = b16.arm_of("b")
    task = b16.task_of("b")
    for name, mine, theirs in (("split_noise_reset vs S36's split_op", ops.split_noise_reset, s36c.split_op),
                               ("copy_exact vs S61's copy_nonoise_op", ops.copy_exact, s61c.copy_nonoise_op)):
        res = []
        for f in (mine, theirs):
            m = s48c._model(a, 1320)
            o = torch.optim.Adam(m.parameters(), lr=1e-3)
            s48c._one_step(m, task, o)
            f(m, o, 3, 7, 12_000_000 + 1000 * 1320)
            res.append((m.W_g.detach().clone(), {k: v.clone() for k, v in o.state[m.W_g].items() if torch.is_tensor(v)}))
        ok = torch.equal(res[0][0], res[1][0]) and res[0][1].keys() == res[1][1].keys() and \
            all(torch.equal(res[0][1][k], res[1][1][k]) for k in res[0][1])
        rows.append((f"{name}: W_g and W_g's Adam state bit for bit equal", ok))
    rows.append((f"the copied fail_class_v2 reads v1's thresholds (ETA_SPLIT {v2.ETA_SPLIT}, MARGIN_PARTIAL {v2.MARGIN_PARTIAL})",
                 v2.ETA_SPLIT == ETA_SPLIT and v2.MARGIN_PARTIAL == MARGIN_PARTIAL))
    return [[n, bool(v)] for n, v in rows]
