#!/usr/bin/env python
"""
explore_f_s53_keydiag.py — EXPLORATORY, not a result; POST HOC (written after S53's results, no reading depends on it).
Why does a single delta channel (β = 1, L2 keys) bind the header layout (S53: 8/10)? The erase removes the old value only
along the new key direction x̂_t; if the two writes of a key (one per stream) use different key vectors, both bindings
survive. Measured on a rerun of S53's SINGLE_DELTA_H and SINGLE_HEBB_H on seed 307 (S53's run path, so the records are
reproduced; checked): per layer, over 512 probe sequences, the cosine between the normalised keys x̂ at the two VAL
positions holding the queried key's two values, and between the query's key x̂_q and each of them (the target stream's
and the other stream's).
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(1)
import explore_f_delta_controls_child as c
import explore_common as ec
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

def keys_by_layer(m, x):
    ks = []
    h = m.attn.register_forward_hook(lambda mod, a, kw, o: ks.append(torch.nn.functional.normalize(kw["Q"][:, 0], dim=-1)), with_kwargs=True)
    m.eval()
    with torch.no_grad():
        m(x)
    h.remove()
    return ks

def diag(kind, seed=307, iters=3600):
    task = c.TASKS["header"]
    a = c.arm_def(kind, task); b = c.builder(kind); b.task = task
    keep = {"at": -1}
    rec = run_one(a, seed, iters, task=task, stats_fn=c.stats_fn(kind, "header"), grad_fn=tsc.conv_grad_norms, lr=c.LR, builder=b, keep=keep)
    m = keep["model"]
    x = task.make_batch(512, torch.Generator().manual_seed(77))[0][:, :-1]
    qpos = task.qpos[0]
    qk, qs = x[:, qpos], x[:, qpos - 1]
    body = x[:, :qpos - 1]
    lab = task.stream_labels(x)[:, :qpos - 1]
    idx = torch.arange(body.shape[1]).expand_as(body)
    hit = body == qk[:, None]
    pos = torch.where(hit, idx, torch.full_like(idx, 10**6)).sort(1).values[:, :2] + 1   # the two VAL positions
    st = lab.gather(1, pos - 1)
    tgt = torch.where(st[:, 0] == qs, pos[:, 0], pos[:, 1]); oth = torch.where(st[:, 0] == qs, pos[:, 1], pos[:, 0])
    out = []
    for L, K in enumerate(keys_by_layer(m, x)):
        B = torch.arange(K.shape[0])
        kt, ko, kq = K[B, tgt], K[B, oth], K[B, qpos]
        out.append(dict(layer=L + 1, cos_two_writes=float((kt * ko).sum(-1).median()), cos_query_target=float((kq * kt).sum(-1).median()),
                        cos_query_other=float((kq * ko).sum(-1).median())))
    return dict(kind=kind, curve=rec["curve"], transition=rec["transition"], layers=out)

if __name__ == "__main__":
    st = ec.load_store("F/delta_controls")
    res = []
    for kind in ("SINGLE_DELTA", "SINGLE_HEBB"):
        r = diag(kind)
        recd = [v for k, v in st["runs"].items() if k.startswith(f"{kind}_H|307|")][0]
        r["reproduces_record_curve"] = r["curve"] == [cc for cc in recd["curve"] if cc[0] <= 3600]
        res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open("explore_out/F/s53_keydiag.json", "w"), indent=1)
